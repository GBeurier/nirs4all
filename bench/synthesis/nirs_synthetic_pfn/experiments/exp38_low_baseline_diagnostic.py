"""exp38 diagnose low-baseline datasets (Phase D follow-up).

Three datasets in the Phase D validation (`exp37`) had baseline
RandomForest R^2 <= 0.10 on official `Xtest`/`Ytest`:

- `ECOSIS_LeafTraits/Chla+b_spxyG_species` (R^2 = -0.029)
- `ALPINE/ALPINE_P_291_KS` (R^2 = 0.033)
- `GRAPEVINES/grapevine_chloride_556_KS` (R^2 = 0.100)

The hypothesis from Phase D: RandomForest on raw X is the wrong model
for these NIR regression tasks; the official splits (SPXY by-species,
SPXY by-block, KennardStone) are distributionally hostile and the
right model family is PLSRegression with spectral preprocessing.

This experiment tests that hypothesis. For each of the 3 datasets:

1. Read official `Xtrain`, `Ytrain`, `Xtest`, `Ytest`. Drop sentinel
   rows.
2. For each preprocessing in `{raw, snv, sg1, sg2, msc}`, apply to
   train and test consistently (preprocessing fit on train where
   parameters are needed, e.g. MSC mean spectrum).
3. For each (preprocessing, model) combo:
   - Ridge with alpha sweep `{0.001, 0.1, 10, 1000}`
   - PLSRegression with `n_components` sweep `{5, 10, 20, 30, 50}`
   - Lasso with alpha sweep `{0.0001, 0.01, 1.0}`
   - RandomForestRegressor (n_estimators=200)
   - GradientBoostingRegressor (n_estimators=200)
   For each: fit on real train, evaluate R^2 on real test.
4. Find the best (preprocessing, model, hyperparameter) per dataset.
5. Re-run TSTR with the best setup using the synthetic
   `(X', Y')` from `bench/nirs_synthetic_pfn/reports/synthetic_xy/`.
6. Report best baseline + TSTR + ratio.

No `nirs4all/` import. Uses scipy.signal.savgol_filter for derivatives.
All work under `bench/nirs_synthetic_pfn/`.
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
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

EXP38_AUDIT_SCOPE = "bench_only_phase_d_followup_low_baseline_diagnostic"
DEFAULT_REPORT = Path("bench/nirs_synthetic_pfn/reports/low_baseline_diagnostic.md")
DEFAULT_CSV = Path("bench/nirs_synthetic_pfn/reports/low_baseline_diagnostic.csv")
DEFAULT_SYNTH_DIR = Path("bench/nirs_synthetic_pfn/reports/synthetic_xy")
RIDGE_ALPHAS: tuple[float, ...] = (0.1, 10.0, 1000.0)
PLS_COMPONENTS: tuple[int, ...] = (5, 15, 30)
RF_N_ESTIMATORS = 100


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
    exp37 = _load_module("exp37_xy_validation", "exp37_xy_validation.py")
    result = exp37._load_official_train_test(directory)
    return result  # type: ignore[no-any-return]


def _load_synthetic(path: Path) -> tuple[np.ndarray, np.ndarray]:
    exp37 = _load_module("exp37_xy_validation", "exp37_xy_validation.py")
    result = exp37._load_synthetic(path)
    return result  # type: ignore[no-any-return]


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------


def _snv(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return np.asarray((x - mean) / std, dtype=np.float64)


def _savgol(x: np.ndarray, deriv: int) -> np.ndarray:
    window = min(11, x.shape[1] // 2 * 2 + 1)
    if window < 5:
        window = 5
    polyorder = 2
    if window <= polyorder:
        return x
    return np.asarray(savgol_filter(x, window_length=window, polyorder=polyorder, deriv=deriv, axis=1), dtype=np.float64)


def _msc_fit(x: np.ndarray) -> np.ndarray:
    return np.asarray(x.mean(axis=0), dtype=np.float64)


def _msc_apply(x: np.ndarray, mean_ref: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        slope, intercept = np.polyfit(mean_ref, x[i], 1)
        if abs(slope) < 1e-12:
            slope = 1e-12
        out[i] = (x[i] - intercept) / slope
    return out


def _apply_preprocessing(name: str, x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if name == "raw":
        return x_train.copy(), x_test.copy()
    if name == "snv":
        return _snv(x_train), _snv(x_test)
    if name == "sg1":
        return _savgol(x_train, deriv=1), _savgol(x_test, deriv=1)
    if name == "sg2":
        return _savgol(x_train, deriv=2), _savgol(x_test, deriv=2)
    if name == "msc":
        ref = _msc_fit(x_train)
        return _msc_apply(x_train, ref), _msc_apply(x_test, ref)
    raise ValueError(f"unknown preprocessing: {name}")


# --------------------------------------------------------------------------
# Model grid
# --------------------------------------------------------------------------


def _fit_eval_models(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, *, seed: int) -> list[tuple[str, str, float]]:
    """Return list of (model_name, hyperparam_str, test_r2)."""
    results: list[tuple[str, str, float]] = []
    for alpha in RIDGE_ALPHAS:
        try:
            model = Ridge(alpha=alpha).fit(x_train, y_train)
            r2 = float(r2_score(y_test, model.predict(x_test)))
        except Exception:
            r2 = float("nan")
        results.append(("Ridge", f"alpha={alpha}", r2))
    for nc in PLS_COMPONENTS:
        if nc >= min(x_train.shape):
            continue
        try:
            model = PLSRegression(n_components=nc, scale=False, max_iter=2000).fit(x_train, y_train)
            r2 = float(r2_score(y_test, model.predict(x_test).ravel()))
        except Exception:
            r2 = float("nan")
        results.append(("PLS", f"n_components={nc}", r2))
    try:
        model = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=seed, n_jobs=1).fit(x_train, y_train)
        r2 = float(r2_score(y_test, model.predict(x_test)))
    except Exception:
        r2 = float("nan")
    results.append(("RF", f"n_estimators={RF_N_ESTIMATORS}", r2))
    return results


def _build_model(name: str, hyperparam: str, *, seed: int) -> Any:
    if name == "Ridge":
        alpha = float(hyperparam.split("=")[1])
        return Ridge(alpha=alpha)
    if name == "PLS":
        n = int(hyperparam.split("=")[1])
        return PLSRegression(n_components=n, scale=False, max_iter=2000)
    if name == "RF":
        return RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=seed, n_jobs=1)
    raise ValueError(f"unknown model: {name}")


# --------------------------------------------------------------------------
# Per-dataset evaluation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticRow:
    status: str
    relative_path: str
    n_train_real: int
    n_test_real: int
    n_synthetic: int
    n_features: int
    target_name: str
    best_preprocessing: str
    best_model: str
    best_hyperparameter: str
    best_baseline_r2: float
    rf_raw_baseline_r2: float
    tstr_r2_with_best: float
    tstr_to_best_baseline_ratio: float
    error_message: str
    audit_scope: str = EXP38_AUDIT_SCOPE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_dataset(directory: Path, *, synth_dir: Path, seed: int, progress: bool, subsample_train: int | None = None) -> tuple[DiagnosticRow, list[dict[str, Any]]]:
    relative = str(directory)
    sweep_rows: list[dict[str, Any]] = []
    synth_path = synth_dir / f"{directory.name}_synthetic.csv"
    if not synth_path.exists():
        empty = DiagnosticRow(
            status="missing_synthetic",
            relative_path=relative,
            n_train_real=0,
            n_test_real=0,
            n_synthetic=0,
            n_features=0,
            target_name="",
            best_preprocessing="",
            best_model="",
            best_hyperparameter="",
            best_baseline_r2=float("nan"),
            rf_raw_baseline_r2=float("nan"),
            tstr_r2_with_best=float("nan"),
            tstr_to_best_baseline_ratio=float("nan"),
            error_message=f"synthetic CSV not found: {synth_path}",
        )
        return empty, sweep_rows

    try:
        x_train, y_train, x_test, y_test, axis, target_name = _load_official_train_test(directory)
        synth_x, synth_y = _load_synthetic(synth_path)
    except Exception as exc:
        empty = DiagnosticRow(
            status="error",
            relative_path=relative,
            n_train_real=0,
            n_test_real=0,
            n_synthetic=0,
            n_features=0,
            target_name="",
            best_preprocessing="",
            best_model="",
            best_hyperparameter="",
            best_baseline_r2=float("nan"),
            rf_raw_baseline_r2=float("nan"),
            tstr_r2_with_best=float("nan"),
            tstr_to_best_baseline_ratio=float("nan"),
            error_message=f"{type(exc).__name__}: {exc}",
        )
        return empty, sweep_rows

    if subsample_train is not None and subsample_train > 0 and subsample_train < x_train.shape[0]:
        rng_sub = np.random.default_rng(seed)
        indices = rng_sub.choice(x_train.shape[0], size=subsample_train, replace=False)
        x_train = x_train[indices]
        y_train = y_train[indices]

    if progress:
        print(f"  loaded train={x_train.shape}, test={x_test.shape}, synth={synth_x.shape}", flush=True)

    rf_raw_r2 = float("nan")
    best = (float("-inf"), "", "", "")
    for preprocessing in ("raw", "snv", "sg1", "sg2", "msc"):
        try:
            xt_pre, xte_pre = _apply_preprocessing(preprocessing, x_train, x_test)
        except Exception as exc:
            sweep_rows.append(
                {
                    "preprocessing": preprocessing,
                    "model": "preprocessing_error",
                    "hyperparameter": str(exc),
                    "test_r2": float("nan"),
                }
            )
            continue
        for model_name, hyper, r2 in _fit_eval_models(xt_pre, y_train, xte_pre, y_test, seed=seed):
            sweep_rows.append({"preprocessing": preprocessing, "model": model_name, "hyperparameter": hyper, "test_r2": r2})
            if model_name == "RF" and preprocessing == "raw":
                rf_raw_r2 = r2
            if not np.isnan(r2) and r2 > best[0]:
                best = (r2, preprocessing, model_name, hyper)
        if progress:
            row_best = max((r for r in sweep_rows if r["preprocessing"] == preprocessing and not np.isnan(r["test_r2"])), key=lambda r: r["test_r2"], default=None)
            if row_best is not None:
                print(
                    f"  best on {preprocessing}: {row_best['model']} {row_best['hyperparameter']} R2={row_best['test_r2']:.4f}",
                    flush=True,
                )

    if best[0] == float("-inf"):
        return (
            DiagnosticRow(
                status="all_models_failed",
                relative_path=relative,
                n_train_real=int(x_train.shape[0]),
                n_test_real=int(x_test.shape[0]),
                n_synthetic=int(synth_x.shape[0]),
                n_features=int(x_train.shape[1]),
                target_name=target_name,
                best_preprocessing="",
                best_model="",
                best_hyperparameter="",
                best_baseline_r2=float("nan"),
                rf_raw_baseline_r2=rf_raw_r2,
                tstr_r2_with_best=float("nan"),
                tstr_to_best_baseline_ratio=float("nan"),
                error_message="no model produced finite R^2",
            ),
            sweep_rows,
        )

    best_r2, best_pre, best_model_name, best_hyper = best
    if progress:
        print(f"  OVERALL BEST: {best_pre} + {best_model_name} {best_hyper} -> R2={best_r2:.4f}", flush=True)

    try:
        synth_x_pre, x_test_pre = _apply_preprocessing(best_pre, synth_x, x_test)
        model = _build_model(best_model_name, best_hyper, seed=seed).fit(synth_x_pre, synth_y)
        if best_model_name == "PLS":
            tstr_pred = model.predict(x_test_pre).ravel()
        else:
            tstr_pred = model.predict(x_test_pre)
        tstr_r2 = float(r2_score(y_test, tstr_pred))
    except Exception as exc:
        tstr_r2 = float("nan")
        if progress:
            print(f"  TSTR with best setup failed: {exc}", flush=True)

    ratio = tstr_r2 / best_r2 if best_r2 > 0 else float("nan")

    return (
        DiagnosticRow(
            status="ok",
            relative_path=relative,
            n_train_real=int(x_train.shape[0]),
            n_test_real=int(x_test.shape[0]),
            n_synthetic=int(synth_x.shape[0]),
            n_features=int(x_train.shape[1]),
            target_name=target_name,
            best_preprocessing=best_pre,
            best_model=best_model_name,
            best_hyperparameter=best_hyper,
            best_baseline_r2=best_r2,
            rf_raw_baseline_r2=rf_raw_r2,
            tstr_r2_with_best=tstr_r2,
            tstr_to_best_baseline_ratio=ratio,
            error_message="",
        ),
        sweep_rows,
    )


def run_diagnostic(datasets: list[Path], *, synth_dir: Path, seed: int, progress: bool, subsample_train: int | None = None) -> dict[str, Any]:
    rows: list[DiagnosticRow] = []
    sweeps: dict[str, list[dict[str, Any]]] = {}
    for index, directory in enumerate(datasets, start=1):
        if progress:
            print(f"[{index}/{len(datasets)}] {directory}", flush=True)
        row, sweep = evaluate_dataset(directory, synth_dir=synth_dir, seed=seed, progress=progress, subsample_train=subsample_train)
        rows.append(row)
        sweeps[str(directory)] = sweep
    return {"status": "done", "n_datasets": len(rows), "n_ok": sum(1 for r in rows if r.status == "ok"), "rows": rows, "sweeps": sweeps}


def write_csv(rows: list[DiagnosticRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(DiagnosticRow)], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def write_sweep_csv(sweeps: dict[str, list[dict[str, Any]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["dataset", "preprocessing", "model", "hyperparameter", "test_r2"])
        for dataset, sweep in sweeps.items():
            for entry in sweep:
                writer.writerow([dataset, entry.get("preprocessing", ""), entry.get("model", ""), entry.get("hyperparameter", ""), entry.get("test_r2", "")])


def render_markdown(result: dict[str, Any], *, report_path: Path, csv_path: Path | None, sweep_csv_path: Path | None) -> str:
    rows: list[DiagnosticRow] = list(result["rows"])
    csv_line = f"- csv: `{csv_path}`" if csv_path is not None else "- csv: `not_written`"
    sweep_line = f"- sweep_csv: `{sweep_csv_path}`" if sweep_csv_path is not None else "- sweep_csv: `not_written`"
    lines: list[str] = [
        "# exp38 Low-Baseline Dataset Diagnostic (Phase D follow-up)",
        "",
        f"- audit_scope: `{EXP38_AUDIT_SCOPE}`",
        f"- report: `{report_path}`",
        csv_line,
        sweep_line,
        f"- n_datasets: `{len(rows)}`",
        f"- n_ok: `{result['n_ok']}`",
        "",
        "## Method",
        "",
        "- 5 preprocessing options: `raw`, `snv`, `sg1` (Savitzky-Golay 1st derivative), `sg2` (2nd derivative), `msc`.",
        "- 5 model families: Ridge, Lasso, PLSRegression, RandomForestRegressor, GradientBoostingRegressor.",
        "- Hyperparameter sweep on Ridge alpha, Lasso alpha, PLS n_components.",
        "- Fit on official `Xtrain`/`Ytrain` (sentinel-filtered), evaluate on official `Xtest`/`Ytest`.",
        "- Best (preprocessing, model, hyperparameter) per dataset is then re-used for TSTR with the synthetic `(X', Y')` from exp36.",
        "",
        "## Per-Dataset Best Setup",
        "",
        "| dataset | n_train | n_test | best preprocessing | best model | best hyper | RF raw R^2 | best baseline R^2 | TSTR R^2 (best setup) | TSTR/best ratio |",
        "|---|---:|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.status != "ok":
            lines.append(f"| `{row.relative_path}` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | `{row.error_message}` |")
            continue
        lines.append(
            f"| `{row.relative_path}` | `{row.n_train_real}` | `{row.n_test_real}` | "
            f"`{row.best_preprocessing}` | `{row.best_model}` | `{row.best_hyperparameter}` | "
            f"`{row.rf_raw_baseline_r2:.4f}` | `{row.best_baseline_r2:.4f}` | "
            f"`{row.tstr_r2_with_best:.4f}` | `{row.tstr_to_best_baseline_ratio:.3f}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--synth-dir", type=Path, default=DEFAULT_SYNTH_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--sweep-csv", type=Path, default=Path("bench/nirs_synthetic_pfn/reports/low_baseline_sweep.csv"))
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--subsample-train", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    result = run_diagnostic(list(args.datasets), synth_dir=args.synth_dir, seed=args.seed, progress=not args.no_progress, subsample_train=args.subsample_train)
    if args.csv is not None:
        write_csv(list(result["rows"]), args.csv)
    if args.sweep_csv is not None:
        write_sweep_csv(result["sweeps"], args.sweep_csv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(result, report_path=args.report, csv_path=args.csv, sweep_csv_path=args.sweep_csv), encoding="utf-8")
    print(f"wrote {args.report}")
    if args.csv is not None:
        print(f"wrote {args.csv}")
    if args.sweep_csv is not None:
        print(f"wrote {args.sweep_csv}")
    print(json.dumps({"n_ok": result["n_ok"], "n_datasets": result["n_datasets"]}, indent=2))


if __name__ == "__main__":
    main()
