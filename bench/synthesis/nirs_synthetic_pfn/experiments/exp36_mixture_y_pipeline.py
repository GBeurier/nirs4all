"""exp36 mixture decomposition + Y predictor pipeline (Phase B/C).

Per `docs/19_DIESEL_FIX_AND_MIXTURE_TO_Y_PLAN.md` after Codex review:

For one dataset where the Step 0 feasibility gate (exp35) returned GO:

1. Read official `Xtrain.csv` + `Ytrain.csv` ONLY. Drop `-999` sentinels.
2. Centered PCA on official-train X. Pick K via 5-fold CV-BIC-style
   (we use CV-R^2 of Ridge on PCA scores, the same selection
   procedure as exp35 because BIC for Ridge regression on PCA scores
   has no closed form here; documented as deviation).
3. Fit `f(C) -> Y` on official-train ONLY: pick best of
   `Ridge(alpha=1.0)` and `RandomForestRegressor(200)` by 5-fold CV.
4. Compute residuals on official-train and store with their
   corresponding `C`. Used for conditional residual sampling.
5. Generate `X'` synthetic spectra with the Phase 18 winner
   (`knn_mixup k=5 alpha=1` + Gaussian noise, PCA rank from CV).
6. Project `X'` -> `C'` via the centered PCA (same projection as in
   step 2).
7. Predict `Y'_mean = f(C')`. For each synthetic row, find K-nearest
   real C samples by Euclidean distance and bootstrap one of their
   residuals as `eps'`. Output `Y' = Y'_mean + eps'`.
8. Save `(X', Y')` CSV for downstream use.

The Y model is fit only on official-train and frozen before any
synthetic projection (no oracle leakage).

Constraints:

- No `nirs4all/` import.
- `Xtest.csv` / `Ytest.csv` not read here. They are reserved for
  Phase D (`exp37`) TSTR / TRTS.
- All work under `bench/nirs_synthetic_pfn/`.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

EXP36_AUDIT_SCOPE = "bench_only_phase_bc_mixture_y_pipeline"
DEFAULT_K_GRID: tuple[int, ...] = (3, 5, 10, 20, 40)
DEFAULT_REPORT = Path("bench/nirs_synthetic_pfn/reports/mixture_y_pipeline.md")
DEFAULT_CSV = Path("bench/nirs_synthetic_pfn/reports/mixture_y_pipeline.csv")
DEFAULT_SYNTHETIC_DIR = Path("bench/nirs_synthetic_pfn/reports/synthetic_xy")


def _load_exp32() -> ModuleType:
    name = "exp32_hybrid_xrealism_discriminator"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_exp35() -> ModuleType:
    name = "exp35_y_predictor_feasibility"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class PipelineRow:
    status: str
    relative_path: str
    n_train_after_sentinel: int
    n_features: int
    target_name: str
    K_selected: int
    cv_r2_ridge: float
    cv_r2_rf: float
    chosen_regressor: str
    train_residual_std: float
    train_y_std: float
    n_synthetic: int
    synthetic_y_mean: float
    synthetic_y_std: float
    real_y_mean: float
    real_y_std: float
    synthetic_csv_path: str
    error_message: str
    audit_scope: str = EXP36_AUDIT_SCOPE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cv_score_pca_ridge(x: np.ndarray, y: np.ndarray, k: int, seed: int) -> float:
    kf = KFold(n_splits=min(5, max(2, x.shape[0] // 5)), shuffle=True, random_state=seed)
    scores: list[float] = []
    for tr_idx, va_idx in kf.split(x):
        x_tr, x_va = x[tr_idx], x[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        scaler = StandardScaler(with_std=False).fit(x_tr)
        pca = PCA(n_components=min(k, *x_tr.shape, x_tr.shape[1])).fit(scaler.transform(x_tr))
        c_tr = pca.transform(scaler.transform(x_tr))
        c_va = pca.transform(scaler.transform(x_va))
        ridge = Ridge(alpha=1.0).fit(c_tr, y_tr)
        scores.append(float(r2_score(y_va, ridge.predict(c_va))))
    return float(np.mean(scores))


def _cv_score_pca_rf(x: np.ndarray, y: np.ndarray, k: int, seed: int) -> float:
    kf = KFold(n_splits=min(5, max(2, x.shape[0] // 5)), shuffle=True, random_state=seed)
    scores: list[float] = []
    for tr_idx, va_idx in kf.split(x):
        x_tr, x_va = x[tr_idx], x[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        scaler = StandardScaler(with_std=False).fit(x_tr)
        pca = PCA(n_components=min(k, *x_tr.shape, x_tr.shape[1])).fit(scaler.transform(x_tr))
        c_tr = pca.transform(scaler.transform(x_tr))
        c_va = pca.transform(scaler.transform(x_va))
        rf = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1).fit(c_tr, y_tr)
        scores.append(float(r2_score(y_va, rf.predict(c_va))))
    return float(np.mean(scores))


def _select_k(x: np.ndarray, y: np.ndarray, k_grid: tuple[int, ...], seed: int) -> int:
    valid = [k for k in sorted(set(k_grid)) if 0 < k < min(x.shape)]
    if not valid:
        return 1
    best_k = valid[0]
    best_score = -np.inf
    for k in valid:
        s = _cv_score_pca_ridge(x, y, k, seed)
        if s > best_score:
            best_score = s
            best_k = k
    return best_k


def run_pipeline_for_dataset(
    directory: Path,
    *,
    n_synthetic: int,
    k_grid: tuple[int, ...] = DEFAULT_K_GRID,
    seed: int = 20260502,
    knn_mixup_k: int = 5,
    knn_mixup_alpha: float = 1.0,
    residual_neighbors: int = 10,
    synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR,
    progress: bool = True,
) -> PipelineRow:
    exp32 = _load_exp32()
    exp35 = _load_exp35()
    relative = str(directory)
    try:
        x_train, y_train, axis, target_name = exp35.load_train_x_and_y(directory)
    except Exception as exc:
        return PipelineRow(
            status="error",
            relative_path=relative,
            n_train_after_sentinel=0,
            n_features=0,
            target_name="",
            K_selected=0,
            cv_r2_ridge=float("nan"),
            cv_r2_rf=float("nan"),
            chosen_regressor="",
            train_residual_std=float("nan"),
            train_y_std=float("nan"),
            n_synthetic=0,
            synthetic_y_mean=float("nan"),
            synthetic_y_std=float("nan"),
            real_y_mean=float("nan"),
            real_y_std=float("nan"),
            synthetic_csv_path="",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    if progress:
        print(f"  loaded x={x_train.shape}, y={y_train.shape}", flush=True)

    k_star = _select_k(x_train, y_train, k_grid, seed)
    cv_ridge = _cv_score_pca_ridge(x_train, y_train, k_star, seed)
    cv_rf = _cv_score_pca_rf(x_train, y_train, k_star, seed)
    chosen = "ridge" if cv_ridge >= cv_rf else "rf"
    if progress:
        print(f"  K*={k_star}, cv_ridge={cv_ridge:.4f}, cv_rf={cv_rf:.4f}, chosen={chosen}", flush=True)

    scaler = StandardScaler(with_std=False).fit(x_train)
    pca = PCA(n_components=k_star).fit(scaler.transform(x_train))
    c_train = pca.transform(scaler.transform(x_train))
    if chosen == "ridge":
        regressor = Ridge(alpha=1.0).fit(c_train, y_train)
    else:
        regressor = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1).fit(c_train, y_train)
    train_pred = regressor.predict(c_train)
    train_residuals = y_train - train_pred

    nn_residuals = NearestNeighbors(n_neighbors=min(residual_neighbors, c_train.shape[0])).fit(c_train)

    rng = np.random.default_rng(seed)
    synth_config = exp32.HybridConfig(
        baseline_degree=3,
        max_peaks=16,
        n_pca_components=k_star,
        add_per_channel_noise=True,
        score_sampling_mode="knn_mixup",
        noise_sampling_mode="gaussian",
        score_knn_mixup_k=knn_mixup_k,
        score_knn_mixup_dirichlet_alpha=knn_mixup_alpha,
        seed=seed,
    )
    synth_gen = exp32.HybridGenerator(synth_config).fit(x_train, axis)
    x_prime = synth_gen.sample(n_synthetic, rng)

    c_prime = pca.transform(scaler.transform(x_prime))
    y_prime_mean = regressor.predict(c_prime)
    _, neighbor_idx = nn_residuals.kneighbors(c_prime, return_distance=True)
    sampled_residuals = np.empty(n_synthetic, dtype=np.float64)
    for i in range(n_synthetic):
        choice = rng.choice(neighbor_idx[i])
        sampled_residuals[i] = train_residuals[choice]
    y_prime = y_prime_mean + sampled_residuals

    synthetic_dir.mkdir(parents=True, exist_ok=True)
    synth_filename = synthetic_dir / f"{directory.name}_synthetic.csv"
    header = ",".join(f"{a:g}" for a in axis) + f",{target_name}"
    with synth_filename.open("w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for row, y_value in zip(x_prime, y_prime, strict=True):
            fh.write(",".join(f"{v:.8g}" for v in row) + f",{y_value:.8g}\n")

    return PipelineRow(
        status="ok",
        relative_path=relative,
        n_train_after_sentinel=int(x_train.shape[0]),
        n_features=int(x_train.shape[1]),
        target_name=target_name,
        K_selected=int(k_star),
        cv_r2_ridge=cv_ridge,
        cv_r2_rf=cv_rf,
        chosen_regressor=chosen,
        train_residual_std=float(np.std(train_residuals)),
        train_y_std=float(np.std(y_train)),
        n_synthetic=int(n_synthetic),
        synthetic_y_mean=float(np.mean(y_prime)),
        synthetic_y_std=float(np.std(y_prime)),
        real_y_mean=float(np.mean(y_train)),
        real_y_std=float(np.std(y_train)),
        synthetic_csv_path=str(synth_filename),
        error_message="",
    )


def run_pipeline(
    datasets: list[Path],
    *,
    n_synthetic_factor: float,
    k_grid: tuple[int, ...] = DEFAULT_K_GRID,
    seed: int = 20260502,
    knn_mixup_k: int = 5,
    knn_mixup_alpha: float = 1.0,
    residual_neighbors: int = 10,
    synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR,
    progress: bool = True,
) -> dict[str, Any]:
    exp35 = _load_exp35()
    rows: list[PipelineRow] = []
    for index, directory in enumerate(datasets, start=1):
        if progress:
            print(f"[{index}/{len(datasets)}] {directory}", flush=True)
        try:
            x_train, _, _, _ = exp35.load_train_x_and_y(directory)
            n_synth = max(1, int(round(x_train.shape[0] * n_synthetic_factor)))
        except Exception as exc:
            rows.append(
                PipelineRow(
                    status="error",
                    relative_path=str(directory),
                    n_train_after_sentinel=0,
                    n_features=0,
                    target_name="",
                    K_selected=0,
                    cv_r2_ridge=float("nan"),
                    cv_r2_rf=float("nan"),
                    chosen_regressor="",
                    train_residual_std=float("nan"),
                    train_y_std=float("nan"),
                    n_synthetic=0,
                    synthetic_y_mean=float("nan"),
                    synthetic_y_std=float("nan"),
                    real_y_mean=float("nan"),
                    real_y_std=float("nan"),
                    synthetic_csv_path="",
                    error_message=f"load failed: {type(exc).__name__}: {exc}",
                )
            )
            continue
        rows.append(
            run_pipeline_for_dataset(
                directory,
                n_synthetic=n_synth,
                k_grid=k_grid,
                seed=seed,
                knn_mixup_k=knn_mixup_k,
                knn_mixup_alpha=knn_mixup_alpha,
                residual_neighbors=residual_neighbors,
                synthetic_dir=synthetic_dir,
                progress=progress,
            )
        )
    return {
        "status": "done",
        "n_datasets": len(rows),
        "n_ok": sum(1 for r in rows if r.status == "ok"),
        "rows": rows,
    }


def write_csv(rows: list[PipelineRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(PipelineRow)], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def render_markdown(result: dict[str, Any], *, report_path: Path, csv_path: Path | None) -> str:
    rows: list[PipelineRow] = list(result["rows"])
    csv_line = f"- csv: `{csv_path}`" if csv_path is not None else "- csv: `not_written`"

    lines: list[str] = [
        "# exp36 Mixture-Y Pipeline (Phase B/C)",
        "",
        f"- audit_scope: `{EXP36_AUDIT_SCOPE}`",
        f"- report: `{report_path}`",
        csv_line,
        f"- n_datasets: `{len(rows)}`",
        f"- n_ok: `{result['n_ok']}`",
        "",
        "## Method",
        "",
        "- Read official `Xtrain.csv` + `Ytrain.csv` only. `Xtest.csv` / `Ytest.csv` are reserved for Phase D.",
        "- Centered PCA on official-train X with K* selected by 5-fold CV-R^2 of Ridge on PCA scores.",
        "- Fit `f(C) -> Y` on official-train: best CV-R^2 of `Ridge(alpha=1)` vs `RandomForestRegressor(200)`.",
        "- Generate `X'` with `knn_mixup k=5 alpha=1` + Gaussian noise (Phase 18 winner).",
        "- Project `X' -> C'` via the same PCA + scaler.",
        "- Predict `Y' = f(C') + eps'` where `eps'` is bootstrapped from the residuals of the K-nearest-C real samples (calibrated residuals).",
        "- Save `(X', Y')` per dataset under `bench/nirs_synthetic_pfn/reports/synthetic_xy/`.",
        "",
        "## Per-Dataset Results",
        "",
        "| dataset | n_train | K* | cv_R2 ridge | cv_R2 rf | chosen | resid_std | y_std | y'_mean / y_mean | y'_std / y_std |",
        "|---|---:|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for row in rows:
        if row.status != "ok":
            lines.append(f"| `{row.relative_path}` | n/a | n/a | n/a | n/a | `{row.error_message}` | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| `{row.relative_path}` | `{row.n_train_after_sentinel}` | `{row.K_selected}` | "
            f"`{row.cv_r2_ridge:.4f}` | `{row.cv_r2_rf:.4f}` | `{row.chosen_regressor}` | "
            f"`{row.train_residual_std:.4f}` | `{row.train_y_std:.4f}` | "
            f"`{row.synthetic_y_mean:.4f} / {row.real_y_mean:.4f}` | "
            f"`{row.synthetic_y_std:.4f} / {row.real_y_std:.4f}` |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--synthetic-dir", type=Path, default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument("--n-synthetic-factor", type=float, default=1.0)
    parser.add_argument("--k-grid", type=str, default="3,5,10,20,40")
    parser.add_argument("--knn-mixup-k", type=int, default=5)
    parser.add_argument("--knn-mixup-alpha", type=float, default=1.0)
    parser.add_argument("--residual-neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    k_grid = tuple(int(t) for t in args.k_grid.split(",") if t.strip())
    result = run_pipeline(
        list(args.datasets),
        n_synthetic_factor=args.n_synthetic_factor,
        k_grid=k_grid,
        seed=args.seed,
        knn_mixup_k=args.knn_mixup_k,
        knn_mixup_alpha=args.knn_mixup_alpha,
        residual_neighbors=args.residual_neighbors,
        synthetic_dir=args.synthetic_dir,
        progress=not args.no_progress,
    )
    if args.csv is not None:
        write_csv(list(result["rows"]), args.csv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(result, report_path=args.report, csv_path=args.csv), encoding="utf-8")
    print(f"wrote {args.report}")
    if args.csv is not None:
        print(f"wrote {args.csv}")
    print(json.dumps({"n_ok": result["n_ok"], "n_datasets": result["n_datasets"]}, indent=2))


if __name__ == "__main__":
    main()
