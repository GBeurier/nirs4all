"""Resumable smoke benchmark for AOM-Ridge.

Reads ``bench/AOM_v0/benchmarks/cohort_regression.csv`` (built by the existing
AOM-PLS benchmark tooling), filters to a small representative subset (or the
full cohort), and runs every requested AOM-Ridge variant on every dataset and
seed. Results are appended row-by-row so the runner is fully resumable.

Cross-validation defaults to ``SPXYFold`` from nirs4all when available, which
is the standard NIRS-aware splitter for AOM_v0; the user can override with a
plain ``KFold`` via ``--cv-kind kfold``.

Usage:

```bash
PYTHONPATH=bench/AOM_v0:bench/AOM_v0/Ridge python \\
  bench/AOM_v0/Ridge/benchmarks/run_aomridge_benchmark.py \\
  --workspace bench/AOM_v0/Ridge/benchmark_runs/smoke \\
  --cohort smoke --variants smoke --cv 3
```
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AOM_ROOT = ROOT.parent
REPO_ROOT = AOM_ROOT.parent.parent
for path in (ROOT, AOM_ROOT, REPO_ROOT):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from aomridge.estimators import AOMRidgeRegressor  # noqa: E402

CODE_VERSION = "AOM_v0/Ridge/0.1.0"

RESULT_COLUMNS = [
    "dataset_group",
    "dataset",
    "task",
    "variant",
    "status",
    "error",
    "selection",
    "operator_bank",
    "alpha",
    "alpha_index",
    "alpha_at_boundary",
    "grid_expansions",
    "cv_min_score",
    "block_scaling",
    "scale_power",
    "x_scale",
    "active_operator_names",
    "selected_operator_names",
    "rmsep",
    "mae",
    "r2",
    "ref_rmse_ridge",
    "ref_rmse_pls",
    "relative_rmsep_vs_ridge_raw",
    "relative_rmsep_vs_paper_ridge",
    "relative_rmsep_vs_pls_standard",
    "fit_time_s",
    "predict_time_s",
    "random_state",
    "version",
]

SMOKE_DATASETS = [
    "Beer_OriginalExtract_60_KS",
    "Rice_Amylose_313_YbasedSplit",
    "ALPINE_P_291_KS",
]

# Larger smoke set used once iter results stabilise
EXTENDED_SMOKE_DATASETS = SMOKE_DATASETS + [
    "Tablet5_KS",
    "Tablet9_KS",
    "Tleaf_grp70_30",
]


# ----------------------------------------------------------------------
# Data loading (mirrors AOM-PLS benchmark conventions)
# ----------------------------------------------------------------------


def _coerce_numeric(df: pd.DataFrame) -> np.ndarray:
    return df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _load_csv_array(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep=";")
    return _coerce_numeric(df)


def _load_csv_target(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(float).to_numpy()
    return df.iloc[:, 0].astype(float).to_numpy()


# ----------------------------------------------------------------------
# CV builders
# ----------------------------------------------------------------------


def _build_cv(kind: str, n_splits: int, seed: int):
    if kind == "spxy":
        try:
            from nirs4all.operators.splitters import SPXYFold
        except Exception as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "SPXYFold requires nirs4all to be installed in the environment"
            ) from exc
        return SPXYFold(n_splits=n_splits, random_state=seed)
    if kind == "kfold":
        from sklearn.model_selection import KFold

        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    raise ValueError(f"unknown cv kind: {kind!r}")


# ----------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------


@dataclass
class Variant:
    label: str
    selection: str
    operator_bank: str = "compact"
    block_scaling: str = "rms"
    extra: dict[str, object] = field(default_factory=dict)


SMOKE_VARIANTS: list[Variant] = [
    # Baseline (no AOM): identity + center
    Variant("Ridge-raw", selection="superblock", operator_bank="identity",
            block_scaling="none"),
    # Same baseline + StandardScaler (paper-style preprocessing)
    Variant("Ridge-raw-stdscale", selection="superblock", operator_bank="identity",
            block_scaling="none", extra={"x_scale": "feature_std"}),
    # Best variants from iter 1 — kept for regression comparison
    Variant("AOMRidge-superblock-compact-none", selection="superblock",
            block_scaling="none"),
    Variant("AOMRidge-global-compact-none", selection="global",
            block_scaling="none"),
    Variant("AOMRidge-active-compact-none", selection="active_superblock",
            block_scaling="none", extra={"active_top_m": 6}),
    # Iter 2 — feature_std variants (Codex backlog item #6)
    Variant("AOMRidge-superblock-compact-none-stdscale", selection="superblock",
            block_scaling="none", extra={"x_scale": "feature_std"}),
    Variant("AOMRidge-global-compact-none-stdscale", selection="global",
            block_scaling="none", extra={"x_scale": "feature_std"}),
    Variant("AOMRidge-active-compact-none-stdscale", selection="active_superblock",
            block_scaling="none", extra={"active_top_m": 6, "x_scale": "feature_std"}),
    # Iter 2b — family-balanced active (Codex backlog item #4)
    Variant("AOMRidge-active-compact-none-blend-fam", selection="active_superblock",
            block_scaling="none", extra={
                "active_top_m": 6,
                "active_score_method": "blend",
                "active_max_per_family": 1,
            }),
]

FULL_VARIANTS: list[Variant] = SMOKE_VARIANTS + [
    Variant("AOMRidge-superblock-default-none", selection="superblock",
            operator_bank="default", block_scaling="none"),
    Variant("AOMRidge-global-default", selection="global",
            operator_bank="default"),
    Variant("AOMRidge-active-default-none", selection="active_superblock",
            operator_bank="default", block_scaling="none",
            extra={"active_top_m": 12}),
]


def _resolve_variants(name: str) -> list[Variant]:
    if name == "smoke":
        return SMOKE_VARIANTS
    if name == "full":
        return FULL_VARIANTS
    raise ValueError(f"unknown variants set: {name!r}")


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
    return float(np.sqrt(np.mean(diff * diff)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true).ravel() - np.asarray(y_pred).ravel())))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ----------------------------------------------------------------------
# Single-variant runner
# ----------------------------------------------------------------------


def _existing_keys(results_path: Path) -> set:
    if not results_path.exists():
        return set()
    df = pd.read_csv(results_path, dtype=str)
    if df.empty:
        return set()
    return {(row["dataset_group"], row["dataset"], row["variant"], row["random_state"])
            for _, row in df.iterrows()}


def _append_row(results_path: Path, row: dict[str, object]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_path.exists()
    with results_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _run_variant(
    variant: Variant,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    seed: int,
    cv_obj,
) -> dict[str, object]:
    kwargs = {
        "selection": variant.selection,
        "operator_bank": variant.operator_bank,
        "block_scaling": variant.block_scaling,
        "cv": cv_obj,
        "random_state": seed,
    }
    kwargs.update(variant.extra)
    est = AOMRidgeRegressor(**kwargs)
    t0 = time.perf_counter()
    est.fit(Xtr, ytr)
    fit_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    yhat = est.predict(Xte)
    predict_time = time.perf_counter() - t1
    diag = est.get_diagnostics()
    out = {
        "selection": variant.selection,
        "operator_bank": variant.operator_bank,
        "alpha": float(diag["alpha"]),
        "alpha_index": diag.get("alpha_index"),
        "alpha_at_boundary": diag.get("alpha_at_boundary"),
        "grid_expansions": diag.get("grid_expansions", 0),
        "cv_min_score": diag.get("cv_min_score"),
        "block_scaling": variant.block_scaling,
        "scale_power": float(diag.get("scale_power", 1.0)),
        "x_scale": diag.get("x_scale", "center"),
        "active_operator_names": json.dumps(
            diag.get("active_operator_names", [])
        ) if variant.selection == "active_superblock" else "",
        "selected_operator_names": json.dumps(diag["selected_operator_names"]),
        "rmsep": rmse(yte, yhat),
        "mae": mae(yte, yhat),
        "r2": r2(yte, yhat),
        "fit_time_s": float(fit_time),
        "predict_time_s": float(predict_time),
    }
    return out


def run_dataset(
    cohort_row: pd.Series,
    variants: Sequence[Variant],
    results_path: Path,
    seeds: Sequence[int],
    cv_kind: str,
    cv_splits: int,
    existing_keys: set,
) -> int:
    Xtr = _load_csv_array(cohort_row["train_path"])
    Xte = _load_csv_array(cohort_row["test_path"])
    ytr = _load_csv_target(cohort_row["ytrain_path"])
    yte = _load_csv_target(cohort_row["ytest_path"])
    ref_pls = cohort_row.get("ref_rmse_pls", "")
    ref_ridge = cohort_row.get("ref_rmse_ridge", "")
    n_added = 0
    for seed in seeds:
        cv_obj = _build_cv(cv_kind, cv_splits, seed)
        # Compute the in-cohort raw-Ridge RMSE first so we can report ratios.
        raw_label = "Ridge-raw"
        raw_row_key = (
            cohort_row["database_name"], cohort_row["dataset"], raw_label, str(seed),
        )
        if raw_row_key in existing_keys:
            raw_rmsep = None
        else:
            raw_variant = next(v for v in variants if v.label == raw_label)
            try:
                raw_metrics = _run_variant(
                    raw_variant, Xtr, ytr, Xte, yte, seed=seed, cv_obj=cv_obj,
                )
                raw_row = _row_record(
                    cohort_row, raw_label, seed, raw_variant, raw_metrics,
                    ref_pls=ref_pls, ref_ridge=ref_ridge, raw_rmsep=None,
                )
                _append_row(results_path, raw_row)
                existing_keys.add(raw_row_key)
                raw_rmsep = float(raw_metrics["rmsep"])
                n_added += 1
            except Exception as exc:
                _append_row(results_path, _error_record(
                    cohort_row, raw_label, seed, raw_variant, exc, ref_pls, ref_ridge,
                ))
                existing_keys.add(raw_row_key)
                raw_rmsep = None
                n_added += 1
        # Run remaining variants
        for variant in variants:
            if variant.label == raw_label:
                continue
            key = (cohort_row["database_name"], cohort_row["dataset"], variant.label, str(seed))
            if key in existing_keys:
                continue
            try:
                metrics = _run_variant(
                    variant, Xtr, ytr, Xte, yte, seed=seed, cv_obj=cv_obj,
                )
                row = _row_record(
                    cohort_row, variant.label, seed, variant, metrics,
                    ref_pls=ref_pls, ref_ridge=ref_ridge, raw_rmsep=raw_rmsep,
                )
            except Exception as exc:
                row = _error_record(
                    cohort_row, variant.label, seed, variant, exc, ref_pls, ref_ridge,
                )
            _append_row(results_path, row)
            existing_keys.add(key)
            n_added += 1
    return n_added


def _row_record(
    cohort_row: pd.Series,
    label: str,
    seed: int,
    variant: Variant,
    metrics: dict[str, object],
    ref_pls,
    ref_ridge,
    raw_rmsep: float | None,
) -> dict[str, object]:
    rmsep = float(metrics["rmsep"])
    rel_ridge_raw = ""
    if raw_rmsep is not None and raw_rmsep > 0:
        rel_ridge_raw = rmsep / raw_rmsep
    rel_paper_ridge = ""
    if pd.notna(ref_ridge) and float(ref_ridge) > 0:
        rel_paper_ridge = rmsep / float(ref_ridge)
    rel_pls = ""
    if pd.notna(ref_pls) and float(ref_pls) > 0:
        rel_pls = rmsep / float(ref_pls)
    return {
        "dataset_group": cohort_row["database_name"],
        "dataset": cohort_row["dataset"],
        "task": "regression",
        "variant": label,
        "status": "ok",
        "error": "",
        "selection": metrics["selection"],
        "operator_bank": metrics["operator_bank"],
        "alpha": metrics["alpha"],
        "alpha_index": metrics.get("alpha_index"),
        "alpha_at_boundary": metrics.get("alpha_at_boundary"),
        "grid_expansions": metrics.get("grid_expansions", 0),
        "cv_min_score": metrics.get("cv_min_score"),
        "block_scaling": metrics["block_scaling"],
        "scale_power": metrics.get("scale_power", 1.0),
        "x_scale": metrics.get("x_scale", "center"),
        "active_operator_names": metrics["active_operator_names"],
        "selected_operator_names": metrics["selected_operator_names"],
        "rmsep": rmsep,
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "ref_rmse_ridge": ref_ridge if pd.notna(ref_ridge) else "",
        "ref_rmse_pls": ref_pls if pd.notna(ref_pls) else "",
        "relative_rmsep_vs_ridge_raw": rel_ridge_raw,
        "relative_rmsep_vs_paper_ridge": rel_paper_ridge,
        "relative_rmsep_vs_pls_standard": rel_pls,
        "fit_time_s": metrics["fit_time_s"],
        "predict_time_s": metrics["predict_time_s"],
        "random_state": seed,
        "version": CODE_VERSION,
    }


def _error_record(
    cohort_row: pd.Series,
    label: str,
    seed: int,
    variant: Variant,
    exc: Exception,
    ref_pls,
    ref_ridge,
) -> dict[str, object]:
    return {
        "dataset_group": cohort_row["database_name"],
        "dataset": cohort_row["dataset"],
        "task": "regression",
        "variant": label,
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
        "selection": variant.selection,
        "operator_bank": variant.operator_bank,
        "alpha": "",
        "alpha_index": "",
        "alpha_at_boundary": "",
        "grid_expansions": "",
        "cv_min_score": "",
        "block_scaling": variant.block_scaling,
        "scale_power": variant.extra.get("scale_power", 1.0),
        "x_scale": variant.extra.get("x_scale", "center"),
        "active_operator_names": "",
        "selected_operator_names": "",
        "rmsep": "",
        "mae": "",
        "r2": "",
        "ref_rmse_ridge": ref_ridge if pd.notna(ref_ridge) else "",
        "ref_rmse_pls": ref_pls if pd.notna(ref_pls) else "",
        "relative_rmsep_vs_ridge_raw": "",
        "relative_rmsep_vs_paper_ridge": "",
        "relative_rmsep_vs_pls_standard": "",
        "fit_time_s": "",
        "predict_time_s": "",
        "random_state": seed,
        "version": CODE_VERSION,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _select_cohort_rows(cohort_path: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(cohort_path)
    df_ok = df[df["status"] == "ok"].copy()
    if name == "smoke":
        preferred = df_ok[df_ok["dataset"].isin(SMOKE_DATASETS)]
        if not preferred.empty:
            return preferred
        return df_ok.head(3)
    if name == "smoke6":
        preferred = df_ok[df_ok["dataset"].isin(EXTENDED_SMOKE_DATASETS)]
        return preferred
    if name == "full":
        return df_ok
    raise ValueError(f"unknown cohort selection: {name!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AOM-Ridge benchmark runner")
    parser.add_argument("--workspace", required=True, help="output workspace")
    parser.add_argument(
        "--cohort", default="smoke", choices=["smoke", "smoke6", "full"],
        help="dataset cohort selection",
    )
    parser.add_argument(
        "--variants", default="smoke", choices=["smoke", "full"],
        help="variant set",
    )
    parser.add_argument("--cv", type=int, default=3, help="CV split count")
    parser.add_argument(
        "--cv-kind", default="spxy", choices=["spxy", "kfold"],
        help="splitter kind for inner CV",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--cohort-path",
        default="bench/AOM_v0/benchmarks/cohort_regression.csv",
        help="path to the AOM_v0 regression cohort CSV",
    )
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    results_path = workspace / "results.csv"

    cohort = _select_cohort_rows(args.cohort_path, args.cohort)
    print(
        f"[aomridge] {len(cohort)} datasets, variants={args.variants}, "
        f"cv={args.cv_kind}({args.cv})"
    )
    variants = _resolve_variants(args.variants)
    existing = _existing_keys(results_path)
    total = 0
    for _, row in cohort.iterrows():
        try:
            n = run_dataset(
                cohort_row=row,
                variants=variants,
                results_path=results_path,
                seeds=args.seeds,
                cv_kind=args.cv_kind,
                cv_splits=args.cv,
                existing_keys=existing,
            )
        except Exception as exc:
            print(f"[aomridge] dataset {row['dataset']} failed: {exc}")
            n = 0
        total += n
        print(f"[aomridge] {row['database_name']}/{row['dataset']} +{n} rows")
    print(f"[aomridge] wrote {total} rows -> {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
