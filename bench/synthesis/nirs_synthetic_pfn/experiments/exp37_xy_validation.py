"""exp37 joint (X, Y) validation (Phase D).

For each dataset where exp36 produced a synthetic `(X', Y')` CSV:

1. Load the official `Xtrain.csv` + `Ytrain.csv` AND `Xtest.csv` + `Ytest.csv`.
   Drop sentinel rows.
2. Load the synthetic `(X', Y')` from `bench/nirs_synthetic_pfn/reports/synthetic_xy/<name>_synthetic.csv`.
3. Joint-distribution adversarial AUC: train RandomForest on stacked
   `[X, Y]` real vs synthetic. Report mean / std AUC across CV splits.
4. KS test on Y marginals (real vs synthetic).
5. TSTR (Train-Synthetic, Test-Real): fit `RandomForestRegressor` on
   `(X', Y')`, evaluate on `(X_test_real, Y_test_real)`. Report R^2.
6. TRTS (Train-Real, Test-Synthetic): fit on `(X_train_real,
   Y_train_real)`, evaluate on `(X', Y')`. Report R^2.
7. Real-only baseline (anchor): fit on `(X_train_real, Y_train_real)`,
   evaluate on `(X_test_real, Y_test_real)`. Report R^2.

The Y model used in exp36 was frozen on official-train only; this
script may compare against `Xtest`/`Ytest` because that's the
TSTR/TRTS use case. No tuning happens here.

Constraints carried forward:

- No `nirs4all/` import.
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
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

EXP37_AUDIT_SCOPE = "bench_only_phase_d_xy_validation"
DEFAULT_REPORT = Path("bench/nirs_synthetic_pfn/reports/xy_validation.md")
DEFAULT_CSV = Path("bench/nirs_synthetic_pfn/reports/xy_validation.csv")
DEFAULT_SYNTH_DIR = Path("bench/nirs_synthetic_pfn/reports/synthetic_xy")


def _load_module(name: str, filename: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_official_train_test(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    exp32 = _load_module("exp32_hybrid_xrealism_discriminator", "exp32_hybrid_xrealism_discriminator.py")
    exp35 = _load_module("exp35_y_predictor_feasibility", "exp35_y_predictor_feasibility.py")
    axis_tr, x_train = exp32._read_data_file(directory / "Xtrain.csv")
    axis_te, x_test = exp32._read_data_file(directory / "Xtest.csv")
    if axis_tr.shape != axis_te.shape or not np.allclose(axis_tr, axis_te):
        raise ValueError(f"axis mismatch in {directory}")
    header_train, y_train_raw = exp35._read_y_file(directory / "Ytrain.csv")
    header_test, y_test_raw = exp35._read_y_file(directory / "Ytest.csv")
    y_train = y_train_raw[:, 0] if y_train_raw.ndim > 1 else y_train_raw
    y_test = y_test_raw[:, 0] if y_test_raw.ndim > 1 else y_test_raw
    sentinels = {-999.0, -9999.0, -99.0}
    train_mask = ~np.isin(y_train, list(sentinels))
    test_mask = ~np.isin(y_test, list(sentinels))
    target_name = header_train[0] if header_train else "target"
    return x_train[train_mask], y_train[train_mask], x_test[test_mask], y_test[test_mask], axis_tr, target_name


def _load_synthetic(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not raw:
        raise ValueError(f"empty synthetic file: {path}")
    sep = ","
    rows = []
    for line in raw[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        rows.append([float(t) for t in stripped.split(sep)])
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, :-1], arr[:, -1]


@dataclass(frozen=True)
class ValidationRow:
    status: str
    relative_path: str
    n_train_real: int
    n_test_real: int
    n_synthetic: int
    n_features: int
    target_name: str
    joint_auc_rf_mean: float
    joint_auc_rf_std: float
    ks_y_statistic: float
    ks_y_pvalue: float
    real_y_test_std: float
    synthetic_y_std: float
    real_only_baseline_r2: float
    real_only_baseline_rmse: float
    tstr_r2: float
    tstr_rmse: float
    trts_r2: float
    trts_rmse: float
    tstr_to_baseline_ratio: float
    error_message: str
    audit_scope: str = EXP37_AUDIT_SCOPE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _joint_auc(real_x: np.ndarray, real_y: np.ndarray, synth_x: np.ndarray, synth_y: np.ndarray, *, n_splits: int = 3, n_estimators: int = 80, seed: int = 20260502) -> tuple[float, float]:
    if real_x.shape[1] != synth_x.shape[1]:
        raise ValueError(f"feature mismatch real={real_x.shape[1]} synth={synth_x.shape[1]}")
    real_xy = np.column_stack([real_x, real_y])
    synth_xy = np.column_stack([synth_x, synth_y])
    x = np.vstack([real_xy, synth_xy])
    y = np.concatenate([np.ones(len(real_xy)), np.zeros(len(synth_xy))])
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=seed)
    aucs: list[float] = []
    for train_idx, test_idx in splitter.split(x, y):
        x_tr, x_te = x[train_idx], x[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=1).fit(x_tr, y_tr)
        prob = clf.predict_proba(x_te)[:, 1]
        aucs.append(float(roc_auc_score(y_te, prob)))
    return float(np.mean(aucs)), float(np.std(aucs))


def evaluate_one(directory: Path, *, synth_dir: Path, n_splits: int, n_estimators: int, seed: int, progress: bool) -> ValidationRow:
    relative = str(directory)
    synth_path = synth_dir / f"{directory.name}_synthetic.csv"
    if not synth_path.exists():
        return ValidationRow(
            status="missing_synthetic",
            relative_path=relative,
            n_train_real=0,
            n_test_real=0,
            n_synthetic=0,
            n_features=0,
            target_name="",
            joint_auc_rf_mean=float("nan"),
            joint_auc_rf_std=float("nan"),
            ks_y_statistic=float("nan"),
            ks_y_pvalue=float("nan"),
            real_y_test_std=float("nan"),
            synthetic_y_std=float("nan"),
            real_only_baseline_r2=float("nan"),
            real_only_baseline_rmse=float("nan"),
            tstr_r2=float("nan"),
            tstr_rmse=float("nan"),
            trts_r2=float("nan"),
            trts_rmse=float("nan"),
            tstr_to_baseline_ratio=float("nan"),
            error_message=f"synthetic CSV not found: {synth_path}",
        )
    try:
        x_train, y_train, x_test, y_test, axis, target_name = _load_official_train_test(directory)
        synth_x, synth_y = _load_synthetic(synth_path)
    except Exception as exc:
        return ValidationRow(
            status="error",
            relative_path=relative,
            n_train_real=0,
            n_test_real=0,
            n_synthetic=0,
            n_features=0,
            target_name="",
            joint_auc_rf_mean=float("nan"),
            joint_auc_rf_std=float("nan"),
            ks_y_statistic=float("nan"),
            ks_y_pvalue=float("nan"),
            real_y_test_std=float("nan"),
            synthetic_y_std=float("nan"),
            real_only_baseline_r2=float("nan"),
            real_only_baseline_rmse=float("nan"),
            tstr_r2=float("nan"),
            tstr_rmse=float("nan"),
            trts_r2=float("nan"),
            trts_rmse=float("nan"),
            tstr_to_baseline_ratio=float("nan"),
            error_message=f"{type(exc).__name__}: {exc}",
        )

    if progress:
        print(
            f"  loaded train={x_train.shape}, test={x_test.shape}, synth={synth_x.shape}",
            flush=True,
        )

    auc_mean, auc_std = _joint_auc(
        np.vstack([x_train, x_test]),
        np.concatenate([y_train, y_test]),
        synth_x,
        synth_y,
        n_splits=n_splits,
        n_estimators=n_estimators,
        seed=seed,
    )

    ks_stat, ks_p = ks_2samp(np.concatenate([y_train, y_test]), synth_y)

    def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    rf_real = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=1).fit(x_train, y_train)
    real_baseline_pred = rf_real.predict(x_test)
    real_baseline_r2 = float(r2_score(y_test, real_baseline_pred))
    real_baseline_rmse = _safe_rmse(y_test, real_baseline_pred)

    rf_synth = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=1).fit(synth_x, synth_y)
    tstr_pred = rf_synth.predict(x_test)
    tstr_r2 = float(r2_score(y_test, tstr_pred))
    tstr_rmse = _safe_rmse(y_test, tstr_pred)

    trts_pred = rf_real.predict(synth_x)
    trts_r2 = float(r2_score(synth_y, trts_pred))
    trts_rmse = _safe_rmse(synth_y, trts_pred)

    ratio = tstr_r2 / real_baseline_r2 if real_baseline_r2 > 0 else float("nan")

    return ValidationRow(
        status="ok",
        relative_path=relative,
        n_train_real=int(x_train.shape[0]),
        n_test_real=int(x_test.shape[0]),
        n_synthetic=int(synth_x.shape[0]),
        n_features=int(x_train.shape[1]),
        target_name=target_name,
        joint_auc_rf_mean=auc_mean,
        joint_auc_rf_std=auc_std,
        ks_y_statistic=float(ks_stat),
        ks_y_pvalue=float(ks_p),
        real_y_test_std=float(np.std(y_test)),
        synthetic_y_std=float(np.std(synth_y)),
        real_only_baseline_r2=real_baseline_r2,
        real_only_baseline_rmse=real_baseline_rmse,
        tstr_r2=tstr_r2,
        tstr_rmse=tstr_rmse,
        trts_r2=trts_r2,
        trts_rmse=trts_rmse,
        tstr_to_baseline_ratio=ratio,
        error_message="",
    )


def run_validation(datasets: list[Path], *, synth_dir: Path, n_splits: int, n_estimators: int, seed: int, progress: bool) -> dict[str, Any]:
    rows: list[ValidationRow] = []
    for index, directory in enumerate(datasets, start=1):
        if progress:
            print(f"[{index}/{len(datasets)}] {directory}", flush=True)
        row = evaluate_one(directory, synth_dir=synth_dir, n_splits=n_splits, n_estimators=n_estimators, seed=seed, progress=progress)
        if progress and row.status == "ok":
            print(
                f"  -> joint_AUC={row.joint_auc_rf_mean:.4f} TSTR_R2={row.tstr_r2:.4f} "
                f"baseline_R2={row.real_only_baseline_r2:.4f} ratio={row.tstr_to_baseline_ratio:.3f} "
                f"KS_p={row.ks_y_pvalue:.3g}",
                flush=True,
            )
        rows.append(row)
    return {"status": "done", "n_datasets": len(rows), "n_ok": sum(1 for r in rows if r.status == "ok"), "rows": rows}


def write_csv(rows: list[ValidationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(ValidationRow)], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def render_markdown(result: dict[str, Any], *, report_path: Path, csv_path: Path | None) -> str:
    rows: list[ValidationRow] = list(result["rows"])
    csv_line = f"- csv: `{csv_path}`" if csv_path is not None else "- csv: `not_written`"

    lines: list[str] = [
        "# exp37 Joint (X, Y) Validation (Phase D)",
        "",
        f"- audit_scope: `{EXP37_AUDIT_SCOPE}`",
        f"- report: `{report_path}`",
        csv_line,
        f"- n_datasets: `{len(rows)}`",
        f"- n_ok: `{result['n_ok']}`",
        "",
        "## Method",
        "",
        "- Joint-distribution adversarial AUC: RandomForest on stacked `[X, Y]` real vs synthetic, 3-fold StratifiedShuffleSplit.",
        "- KS test on Y marginals (real Y over train+test vs synthetic Y).",
        "- TSTR (Train-Synthetic, Test-Real): RandomForestRegressor on `(X', Y')`, evaluated on official `(Xtest, Ytest)` real.",
        "- TRTS (Train-Real, Test-Synthetic): RandomForestRegressor on real train, evaluated on `(X', Y')`.",
        "- Real-only baseline (anchor): RandomForestRegressor on real train, evaluated on real test.",
        "- The synthetic `(X', Y')` was produced by exp36 with the Y model frozen on official-train only.",
        "",
        "## Per-Dataset Results",
        "",
        "| dataset | n_train | n_test | n_synth | joint AUC (RF) | KS Y p-val | baseline R^2 | TSTR R^2 | TRTS R^2 | TSTR/base ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.status != "ok":
            lines.append(f"| `{row.relative_path}` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | `{row.error_message}` |")
            continue
        lines.append(
            f"| `{row.relative_path}` | `{row.n_train_real}` | `{row.n_test_real}` | `{row.n_synthetic}` | "
            f"`{row.joint_auc_rf_mean:.4f} +/- {row.joint_auc_rf_std:.4f}` | "
            f"`{row.ks_y_pvalue:.3g}` | "
            f"`{row.real_only_baseline_r2:.4f}` | `{row.tstr_r2:.4f}` | `{row.trts_r2:.4f}` | "
            f"`{row.tstr_to_baseline_ratio:.3f}` |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--synth-dir", type=Path, default=DEFAULT_SYNTH_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    result = run_validation(
        list(args.datasets),
        synth_dir=args.synth_dir,
        n_splits=args.n_splits,
        n_estimators=args.n_estimators,
        seed=args.seed,
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
