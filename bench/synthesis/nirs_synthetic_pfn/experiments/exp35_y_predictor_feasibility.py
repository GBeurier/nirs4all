"""exp35 y-predictor feasibility gate (real-only, no synthetic).

Codex-mandated Step 0 before any Phase B/C/D mixture+Y work
(`docs/19_DIESEL_FIX_AND_MIXTURE_TO_Y_PLAN.md`).

For each dataset:

1. Read official `Xtrain.csv` and `Ytrain.csv` only. Never touch
   `Xtest.csv`/`Ytest.csv` here.
2. Inner split: 80/20 train/val.
3. Centered PCA on inner-train X with K swept across a small grid;
   pick `K*` via 5-fold CV on inner-train using R^2 of a Ridge
   regressor on PCA scores.
4. Fit `f(C) -> Y` on inner-train: Ridge and RandomForestRegressor.
5. Direct baseline: Ridge on raw `X -> Y` (inner train).
6. Project inner-val X -> C_val. Predict Y_val_hat. Report R^2 / RMSE.
7. Decision (per dataset):
   - **GO** if `R^2(f(C)) >= 0.8 * R^2(direct_X)` on the inner-val,
     and residual variance is not collapsed.
   - **KILL** otherwise.

The whole Phase B/C/D mixture-Y chain depends on this gate. If most
hard datasets KILL, ship X-only (Phase 18 is the final deliverable
for those families).

No synthetic data is generated here. No adversarial AUC. No oracle
abuse — only standard supervised CV on real data.

Constraints carried forward:

- No `nirs4all/` import.
- All work under `bench/nirs_synthetic_pfn/`.
- `Xtest.csv` and `Ytest.csv` are NEVER read by this script.
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
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

EXP35_AUDIT_SCOPE = "bench_only_step0_y_predictor_feasibility_real_only"
DEFAULT_REPORT = Path("bench/nirs_synthetic_pfn/reports/y_predictor_feasibility.md")
DEFAULT_CSV = Path("bench/nirs_synthetic_pfn/reports/y_predictor_feasibility.csv")
DEFAULT_K_GRID: tuple[int, ...] = (3, 5, 10, 20, 40)
SENTINEL_VALUES: tuple[str, ...] = ("-999", "-9999", "-99")


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


def _read_y_file(path: Path) -> tuple[list[str], np.ndarray]:
    """Return (header_columns, Y) for one Ytrain/Ytest file. Drops empty rows."""
    raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not raw_lines:
        raise ValueError(f"empty Y file: {path}")
    header_line = raw_lines[0]
    sep = ";" if header_line.count(";") >= header_line.count(",") else ","
    header_tokens = [tok.strip().strip('"').strip("'") for tok in header_line.split(sep) if tok.strip()]
    rows: list[list[float]] = []
    for line in raw_lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        cleaned = stripped.replace('"', "").replace("'", "")
        tokens = [tok.strip() for tok in cleaned.split(sep) if tok.strip()]
        try:
            row = [float(tok) for tok in tokens]
        except ValueError as exc:
            raise ValueError(f"non-numeric Y token in {path}: {exc}") from exc
        rows.append(row)
    if not rows:
        raise ValueError(f"no numeric Y rows in {path}")
    y = np.asarray(rows, dtype=np.float64)
    return header_tokens, y


def load_train_x_and_y(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load X_train, Y_train (first column), axis, and target column name from official train files."""
    exp32 = _load_exp32()
    axis, x_train = exp32._read_data_file(directory / "Xtrain.csv")
    header, y = _read_y_file(directory / "Ytrain.csv")
    if y.ndim == 1:
        y_col = y
    else:
        y_col = y[:, 0]
    if x_train.shape[0] != y_col.shape[0]:
        raise ValueError(f"row mismatch: X_train={x_train.shape[0]}, Y_train={y_col.shape[0]} in {directory}")
    target_name = header[0] if header else "target"
    # Drop sentinel rows
    mask = np.ones(y_col.shape[0], dtype=bool)
    for sentinel in SENTINEL_VALUES:
        try:
            value = float(sentinel)
        except ValueError:
            continue
        mask &= y_col != value
    if not mask.any():
        raise ValueError(f"all rows are sentinels in {directory / 'Ytrain.csv'}")
    return x_train[mask], y_col[mask], axis, target_name


@dataclass(frozen=True)
class FeasibilityRow:
    status: str
    relative_path: str
    n_train_after_sentinel: int
    n_features: int
    target_name: str
    axis_min: float
    axis_max: float
    inner_split_test_size: float
    K_grid_used: str
    K_selected: int
    direct_ridge_r2: float
    direct_ridge_rmse: float
    fc_ridge_r2: float
    fc_ridge_rmse: float
    fc_rf_r2: float
    fc_rf_rmse: float
    best_fc_r2: float
    best_fc_rmse: float
    fc_to_direct_r2_ratio: float
    fc_residual_std: float
    real_y_std_inner_val: float
    decision: str
    error_message: str
    audit_scope: str = EXP35_AUDIT_SCOPE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _select_k_by_cv(x: np.ndarray, y: np.ndarray, k_grid: tuple[int, ...], seed: int = 20260502) -> int:
    """Select K maximizing 5-fold CV R^2 of a Ridge on PCA scores fit inside each fold."""
    valid_k = sorted({k for k in k_grid if k > 0 and k < min(x.shape)})
    if not valid_k:
        return min(k_grid) if k_grid else 1
    kf = KFold(n_splits=min(5, max(2, x.shape[0] // 5)), shuffle=True, random_state=seed)
    best_k = valid_k[0]
    best_score = -np.inf
    for k in valid_k:
        scores = []
        for tr_idx, va_idx in kf.split(x):
            x_tr, x_va = x[tr_idx], x[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            scaler = StandardScaler(with_std=False).fit(x_tr)
            x_tr_c = scaler.transform(x_tr)
            x_va_c = scaler.transform(x_va)
            pca = PCA(n_components=k).fit(x_tr_c)
            c_tr = pca.transform(x_tr_c)
            c_va = pca.transform(x_va_c)
            ridge = Ridge(alpha=1.0).fit(c_tr, y_tr)
            scores.append(r2_score(y_va, ridge.predict(c_va)))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_k = k
    return best_k


def _evaluate_dataset(directory: Path, *, k_grid: tuple[int, ...], inner_test_size: float, seed: int) -> FeasibilityRow:
    relative = str(directory)
    try:
        x_train, y_train, axis, target_name = load_train_x_and_y(directory)
    except Exception as exc:
        return FeasibilityRow(
            status="error",
            relative_path=relative,
            n_train_after_sentinel=0,
            n_features=0,
            target_name="",
            axis_min=0.0,
            axis_max=0.0,
            inner_split_test_size=inner_test_size,
            K_grid_used=",".join(str(k) for k in k_grid),
            K_selected=0,
            direct_ridge_r2=float("nan"),
            direct_ridge_rmse=float("nan"),
            fc_ridge_r2=float("nan"),
            fc_ridge_rmse=float("nan"),
            fc_rf_r2=float("nan"),
            fc_rf_rmse=float("nan"),
            best_fc_r2=float("nan"),
            best_fc_rmse=float("nan"),
            fc_to_direct_r2_ratio=float("nan"),
            fc_residual_std=float("nan"),
            real_y_std_inner_val=float("nan"),
            decision="error",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    if x_train.shape[0] < 30:
        return FeasibilityRow(
            status="skipped_too_few_samples",
            relative_path=relative,
            n_train_after_sentinel=int(x_train.shape[0]),
            n_features=int(x_train.shape[1]),
            target_name=target_name,
            axis_min=float(axis.min()),
            axis_max=float(axis.max()),
            inner_split_test_size=inner_test_size,
            K_grid_used=",".join(str(k) for k in k_grid),
            K_selected=0,
            direct_ridge_r2=float("nan"),
            direct_ridge_rmse=float("nan"),
            fc_ridge_r2=float("nan"),
            fc_ridge_rmse=float("nan"),
            fc_rf_r2=float("nan"),
            fc_rf_rmse=float("nan"),
            best_fc_r2=float("nan"),
            best_fc_rmse=float("nan"),
            fc_to_direct_r2_ratio=float("nan"),
            fc_residual_std=float("nan"),
            real_y_std_inner_val=float("nan"),
            decision="skipped",
            error_message=f"n_train_after_sentinel={x_train.shape[0]} < 30",
        )

    x_inner_tr, x_inner_va, y_inner_tr, y_inner_va = train_test_split(
        x_train,
        y_train,
        test_size=inner_test_size,
        random_state=seed,
    )

    k_selected = _select_k_by_cv(x_inner_tr, y_inner_tr, k_grid, seed=seed)
    scaler = StandardScaler(with_std=False).fit(x_inner_tr)
    x_inner_tr_c = scaler.transform(x_inner_tr)
    x_inner_va_c = scaler.transform(x_inner_va)
    pca = PCA(n_components=k_selected).fit(x_inner_tr_c)
    c_tr = pca.transform(x_inner_tr_c)
    c_va = pca.transform(x_inner_va_c)

    ridge_fc = Ridge(alpha=1.0).fit(c_tr, y_inner_tr)
    rf_fc = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1).fit(c_tr, y_inner_tr)
    ridge_direct = Ridge(alpha=1.0).fit(x_inner_tr, y_inner_tr)

    pred_fc_ridge = ridge_fc.predict(c_va)
    pred_fc_rf = rf_fc.predict(c_va)
    pred_direct = ridge_direct.predict(x_inner_va)

    def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    fc_ridge_r2 = float(r2_score(y_inner_va, pred_fc_ridge))
    fc_rf_r2 = float(r2_score(y_inner_va, pred_fc_rf))
    direct_r2 = float(r2_score(y_inner_va, pred_direct))

    best_fc_r2 = max(fc_ridge_r2, fc_rf_r2)
    best_fc_pred = pred_fc_ridge if fc_ridge_r2 >= fc_rf_r2 else pred_fc_rf
    best_fc_rmse = _safe_rmse(y_inner_va, best_fc_pred)
    fc_residual_std = float(np.std(y_inner_va - best_fc_pred))
    real_y_std = float(np.std(y_inner_va))

    if direct_r2 <= 0.0 or np.isnan(direct_r2):
        ratio = float("nan")
        decision = "kill_baseline_useless"
    else:
        ratio = best_fc_r2 / direct_r2
        if best_fc_r2 < 0:
            decision = "kill_fc_negative_r2"
        elif fc_residual_std < 0.1 * real_y_std:
            decision = "kill_residual_collapsed"
        elif ratio >= 0.8:
            decision = "go"
        else:
            decision = "kill_below_80pct_ratio"

    return FeasibilityRow(
        status="ok",
        relative_path=relative,
        n_train_after_sentinel=int(x_train.shape[0]),
        n_features=int(x_train.shape[1]),
        target_name=target_name,
        axis_min=float(axis.min()),
        axis_max=float(axis.max()),
        inner_split_test_size=inner_test_size,
        K_grid_used=",".join(str(k) for k in k_grid),
        K_selected=int(k_selected),
        direct_ridge_r2=direct_r2,
        direct_ridge_rmse=_safe_rmse(y_inner_va, pred_direct),
        fc_ridge_r2=fc_ridge_r2,
        fc_ridge_rmse=_safe_rmse(y_inner_va, pred_fc_ridge),
        fc_rf_r2=fc_rf_r2,
        fc_rf_rmse=_safe_rmse(y_inner_va, pred_fc_rf),
        best_fc_r2=best_fc_r2,
        best_fc_rmse=best_fc_rmse,
        fc_to_direct_r2_ratio=ratio,
        fc_residual_std=fc_residual_std,
        real_y_std_inner_val=real_y_std,
        decision=decision,
        error_message="",
    )


def run_feasibility(
    datasets: list[Path],
    *,
    k_grid: tuple[int, ...] = DEFAULT_K_GRID,
    inner_test_size: float = 0.2,
    seed: int = 20260502,
    progress: bool = True,
) -> dict[str, Any]:
    rows: list[FeasibilityRow] = []
    for index, directory in enumerate(datasets, start=1):
        if progress:
            print(f"[{index}/{len(datasets)}] {directory}", flush=True)
        row = _evaluate_dataset(directory, k_grid=k_grid, inner_test_size=inner_test_size, seed=seed)
        if progress and row.status == "ok":
            print(
                f"  -> direct_R2={row.direct_ridge_r2:.4f} fc_best_R2={row.best_fc_r2:.4f} "
                f"ratio={row.fc_to_direct_r2_ratio:.3f} decision={row.decision}",
                flush=True,
            )
        rows.append(row)
    decisions = [r.decision for r in rows if r.status == "ok"]
    n_go = decisions.count("go")
    n_kill = sum(1 for d in decisions if d.startswith("kill"))
    return {
        "status": "done",
        "n_datasets": len(rows),
        "n_ok": sum(1 for r in rows if r.status == "ok"),
        "n_go": n_go,
        "n_kill": n_kill,
        "rows": rows,
    }


def write_csv(rows: list[FeasibilityRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(FeasibilityRow)], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def render_markdown(result: dict[str, Any], *, report_path: Path, csv_path: Path | None) -> str:
    rows: list[FeasibilityRow] = list(result["rows"])
    csv_line = f"- csv: `{csv_path}`" if csv_path is not None else "- csv: `not_written`"
    n_total = len(rows)
    n_ok = result["n_ok"]
    n_go = result["n_go"]
    n_kill = result["n_kill"]
    overall = "GO (proceed with Phase B/C/D)" if n_go > n_kill else "KILL (ship X-only)"

    lines: list[str] = [
        "# exp35 Y-predictor Feasibility Gate (Codex Step 0)",
        "",
        f"- audit_scope: `{EXP35_AUDIT_SCOPE}`",
        f"- report: `{report_path}`",
        csv_line,
        f"- n_datasets: `{n_total}`",
        f"- n_ok: `{n_ok}`",
        f"- n_go: `{n_go}`",
        f"- n_kill: `{n_kill}`",
        f"- overall_decision: `{overall}`",
        "",
        "## Method",
        "",
        "- Read official `Xtrain.csv` + `Ytrain.csv` only. **`Xtest.csv` / `Ytest.csv` are never read by this script** (no oracle leakage into the future TSTR step).",
        "- Drop sentinel Y rows (`-999`, `-9999`, `-99`).",
        "- Inner split: 80/20 train/val on official-train rows (seed=20260502).",
        "- Centered PCA on inner-train X, K swept over `{3, 5, 10, 20, 40}`, K* picked by 5-fold CV R^2 of Ridge on PCA scores.",
        "- Fit f(C)->Y on inner-train: Ridge(alpha=1) and RandomForestRegressor(n_estimators=200).",
        "- Direct baseline: Ridge(alpha=1) on raw X->Y on inner-train.",
        "- Evaluate on inner-val: R^2 / RMSE.",
        "- Decision per dataset:",
        "  - `go` if best `R^2(f(C)) >= 0.8 * R^2(direct_X)` AND residual std >= 10% of real Y std.",
        "  - `kill_*` otherwise (suffix indicates failure mode).",
        "",
        "## Per-Dataset Results",
        "",
        "| dataset | n_train | n_feat | K* | direct R^2 | best f(C) R^2 | ratio | decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.relative_path}` | `{row.n_train_after_sentinel}` | `{row.n_features}` | `{row.K_selected}` | "
            f"`{row.direct_ridge_r2:.4f}` | `{row.best_fc_r2:.4f}` | `{row.fc_to_direct_r2_ratio:.3f}` | "
            f"`{row.decision}` |"
        )

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "PYTHONPATH=bench/nirs_synthetic_pfn/src python \\",
            "  bench/nirs_synthetic_pfn/experiments/exp35_y_predictor_feasibility.py \\",
            "  --datasets <space-separated paths> \\",
            f"  --report {report_path} \\",
            f"  --csv {csv_path if csv_path is not None else 'bench/nirs_synthetic_pfn/reports/y_predictor_feasibility.csv'}",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, nargs="+", required=True, help="One or more dataset directories.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--k-grid", type=str, default="3,5,10,20,40")
    parser.add_argument("--inner-test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    k_grid = tuple(int(token) for token in args.k_grid.split(",") if token.strip())
    result = run_feasibility(
        list(args.datasets),
        k_grid=k_grid,
        inner_test_size=args.inner_test_size,
        seed=args.seed,
        progress=not args.no_progress,
    )
    if args.csv is not None:
        write_csv(list(result["rows"]), args.csv)
    markdown = render_markdown(result, report_path=args.report, csv_path=args.csv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown, encoding="utf-8")
    print(f"wrote {args.report}")
    if args.csv is not None:
        print(f"wrote {args.csv}")
    print(json.dumps({"n_go": result["n_go"], "n_kill": result["n_kill"], "n_ok": result["n_ok"]}, indent=2))


if __name__ == "__main__":
    main()
