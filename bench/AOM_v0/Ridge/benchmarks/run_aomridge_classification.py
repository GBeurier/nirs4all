"""Resumable classification benchmark for AOM-Ridge.

Mirrors :mod:`run_aomridge_benchmark` but for classification tasks. Reads
``bench/AOM_v0/benchmarks/cohort_classification.csv``, runs every requested
variant on every dataset and seed, and appends results row-by-row so the
runner is fully resumable.

The variants are the classifier-equivalents of the five AOM-Ridge selection
modes (``superblock``, ``global``, ``active_superblock``, ``mkl``,
``branch_global``) on the compact bank. Each variant builds an
:class:`AOMRidgeClassifier` and reports balanced accuracy, macro F1,
log loss, and expected calibration error (ECE).

Cross-validation defaults to ``KFold`` because the underlying
:class:`AOMRidgeRegressor` invokes ``cv.split(X, Y_encoded)`` where
``Y_encoded`` is the multi-output one-hot encoding (which sklearn's
``StratifiedKFold`` rejects). Use ``--cv-kind stratified`` only if you
provide a custom splitter that handles multi-output targets.

Usage:

```bash
PYTHONPATH=bench/AOM_v0:bench/AOM_v0/Ridge python \\
  bench/AOM_v0/Ridge/benchmarks/run_aomridge_classification.py \\
  --workspace bench/AOM_v0/Ridge/benchmark_runs/cls_smoke \\
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

from aompls.metrics import (  # noqa: E402
    balanced_accuracy,
    expected_calibration_error,
    log_loss,
    macro_f1,
)
from aomridge.branches import make_branch_preproc  # noqa: E402
from aomridge.classification import AOMRidgeClassifier  # noqa: E402

CODE_VERSION = "AOM_v0/Ridge/0.1.0-cls"

RESULT_COLUMNS = [
    "dataset_group",
    "dataset",
    "task",
    "variant",
    "status",
    "error",
    "selection",
    "operator_bank",
    "calibrator_kind",
    "alpha",
    "alpha_index",
    "alpha_at_boundary",
    "grid_expansions",
    "cv_min_score",
    "block_scaling",
    "scale_power",
    "x_scale",
    "branch_preproc",
    "active_operator_names",
    "selected_operator_names",
    "n_classes",
    "balanced_accuracy",
    "macro_f1",
    "log_loss",
    "ece",
    "fit_time_s",
    "predict_time_s",
    "random_state",
    "version",
]

# Smallest classification cohort entries (chosen for fast smoke runs).
SMOKE_DATASETS = [
    "Beef_Impurity_60_AlJowder",
    "CoffeeType_kenstone70_strat",
]


# ----------------------------------------------------------------------
# Data loading (mirrors AOM-PLS classification benchmark conventions)
# ----------------------------------------------------------------------


def _coerce_numeric(df: pd.DataFrame) -> np.ndarray:
    return df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _load_csv_array(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep=";")
    arr = _coerce_numeric(df)
    if np.isnan(arr).any():
        col_mean = np.nanmean(arr, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        idx = np.where(np.isnan(arr))
        arr[idx] = np.take(col_mean, idx[1])
    return arr


def _load_csv_target(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep=";")
    series = df.iloc[:, 0]
    # Classification targets are integer class labels. Float CSVs (e.g.
    # ``0.0`` / ``1.0``) round-trip cleanly to int.
    return series.astype(float).round().astype(int).to_numpy()


# ----------------------------------------------------------------------
# CV builders
# ----------------------------------------------------------------------


def _build_cv(kind: str, n_splits: int, seed: int):
    if kind == "stratified":
        from sklearn.model_selection import StratifiedKFold

        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if kind == "kfold":
        from sklearn.model_selection import KFold

        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    raise ValueError(f"unknown cv kind: {kind!r}")


# ----------------------------------------------------------------------
# Variants — classifier-equivalents of the 5 selection modes
# ----------------------------------------------------------------------


@dataclass
class Variant:
    label: str
    selection: str
    operator_bank: str = "compact"
    block_scaling: str = "none"
    branch_preproc: str | None = None
    extra: dict[str, object] = field(default_factory=dict)


SMOKE_VARIANTS: list[Variant] = [
    Variant("AOMRidgeCls-superblock-compact", selection="superblock"),
    Variant("AOMRidgeCls-global-compact", selection="global"),
    Variant("AOMRidgeCls-active-compact", selection="active_superblock",
            extra={"active_top_m": 6}),
    Variant("AOMRidgeCls-mkl-compact", selection="mkl",
            extra={"mkl_top_k": 6}),
    Variant("AOMRidgeCls-branch_global-compact", selection="branch_global",
            extra={"branches": ("none", "snv", "msc")}),
]


def _resolve_variants(name: str) -> list[Variant]:
    if name == "smoke":
        return SMOKE_VARIANTS
    raise ValueError(f"unknown variants set: {name!r}")


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
    est = AOMRidgeClassifier(**kwargs)
    preproc = make_branch_preproc(variant.branch_preproc) if variant.branch_preproc else None
    t0 = time.perf_counter()
    if preproc is not None:
        Xtr_used = preproc.fit_transform(Xtr, ytr)
        est.fit(Xtr_used, ytr)
    else:
        est.fit(Xtr, ytr)
    fit_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    if preproc is not None:
        Xte_used = preproc.transform(Xte)
        yhat = est.predict(Xte_used)
        proba = est.predict_proba(Xte_used)
    else:
        yhat = est.predict(Xte)
        proba = est.predict_proba(Xte)
    predict_time = time.perf_counter() - t1
    diag = est.get_diagnostics()
    return {
        "selection": variant.selection,
        "operator_bank": variant.operator_bank,
        "calibrator_kind": diag.get("calibrator_kind", ""),
        "alpha": float(diag["alpha"]),
        "alpha_index": diag.get("alpha_index"),
        "alpha_at_boundary": diag.get("alpha_at_boundary"),
        "grid_expansions": diag.get("grid_expansions", 0),
        "cv_min_score": diag.get("cv_min_score"),
        "block_scaling": variant.block_scaling,
        "scale_power": float(diag.get("scale_power", 1.0)),
        "x_scale": diag.get("x_scale", "center"),
        "branch_preproc": variant.branch_preproc or "",
        "active_operator_names": json.dumps(
            diag.get("active_operator_names", [])
        ) if variant.selection == "active_superblock" else "",
        "selected_operator_names": json.dumps(diag["selected_operator_names"]),
        "n_classes": int(diag.get("n_classes", est.classes_.shape[0])),
        "balanced_accuracy": balanced_accuracy(yte, yhat),
        "macro_f1": macro_f1(yte, yhat),
        "log_loss": log_loss(yte, proba, classes=est.classes_),
        "ece": expected_calibration_error(yte, proba, n_bins=10),
        "fit_time_s": float(fit_time),
        "predict_time_s": float(predict_time),
    }


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
    n_added = 0
    for seed in seeds:
        cv_obj = _build_cv(cv_kind, cv_splits, seed)
        for variant in variants:
            key = (
                cohort_row["database_name"], cohort_row["dataset"],
                variant.label, str(seed),
            )
            if key in existing_keys:
                continue
            try:
                metrics = _run_variant(
                    variant, Xtr, ytr, Xte, yte, seed=seed, cv_obj=cv_obj,
                )
                row = _row_record(cohort_row, variant.label, seed, variant, metrics)
            except Exception as exc:
                row = _error_record(cohort_row, variant.label, seed, variant, exc)
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
) -> dict[str, object]:
    return {
        "dataset_group": cohort_row["database_name"],
        "dataset": cohort_row["dataset"],
        "task": "classification",
        "variant": label,
        "status": "ok",
        "error": "",
        "selection": metrics["selection"],
        "operator_bank": metrics["operator_bank"],
        "calibrator_kind": metrics["calibrator_kind"],
        "alpha": metrics["alpha"],
        "alpha_index": metrics.get("alpha_index"),
        "alpha_at_boundary": metrics.get("alpha_at_boundary"),
        "grid_expansions": metrics.get("grid_expansions", 0),
        "cv_min_score": metrics.get("cv_min_score"),
        "block_scaling": metrics["block_scaling"],
        "scale_power": metrics.get("scale_power", 1.0),
        "x_scale": metrics.get("x_scale", "center"),
        "branch_preproc": metrics.get("branch_preproc", ""),
        "active_operator_names": metrics["active_operator_names"],
        "selected_operator_names": metrics["selected_operator_names"],
        "n_classes": metrics["n_classes"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "log_loss": metrics["log_loss"],
        "ece": metrics["ece"],
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
) -> dict[str, object]:
    return {
        "dataset_group": cohort_row["database_name"],
        "dataset": cohort_row["dataset"],
        "task": "classification",
        "variant": label,
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
        "selection": variant.selection,
        "operator_bank": variant.operator_bank,
        "calibrator_kind": "",
        "alpha": "",
        "alpha_index": "",
        "alpha_at_boundary": "",
        "grid_expansions": "",
        "cv_min_score": "",
        "block_scaling": variant.block_scaling,
        "scale_power": variant.extra.get("scale_power", 1.0),
        "x_scale": variant.extra.get("x_scale", "center"),
        "branch_preproc": variant.branch_preproc or "",
        "active_operator_names": "",
        "selected_operator_names": "",
        "n_classes": "",
        "balanced_accuracy": "",
        "macro_f1": "",
        "log_loss": "",
        "ece": "",
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
        # Fallback: smallest n_train datasets
        return df_ok.sort_values("n_train").head(2)
    if name == "full":
        return df_ok
    raise ValueError(f"unknown cohort selection: {name!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AOM-Ridge classification benchmark runner",
    )
    parser.add_argument("--workspace", required=True, help="output workspace")
    parser.add_argument(
        "--cohort", default="smoke", choices=["smoke", "full"],
        help="dataset cohort selection",
    )
    parser.add_argument(
        "--variants", default="smoke", choices=["smoke"],
        help="variant set",
    )
    parser.add_argument("--cv", type=int, default=3, help="CV split count")
    parser.add_argument(
        "--cv-kind", default="kfold", choices=["stratified", "kfold"],
        help="splitter kind for inner CV (default: kfold; stratified only "
        "works with a custom multi-output-aware wrapper)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--cohort-path",
        default="bench/AOM_v0/benchmarks/cohort_classification.csv",
        help="path to the AOM_v0 classification cohort CSV",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="optional subset of dataset names to run (overrides --cohort)",
    )
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    results_path = workspace / "results.csv"

    cohort = _select_cohort_rows(args.cohort_path, args.cohort)
    if args.datasets:
        cohort = cohort[cohort["dataset"].isin(args.datasets)]
        if cohort.empty:
            print(
                f"[aomridge-cls] no ok-status rows match --datasets={args.datasets}",
            )
            return 1
    print(
        f"[aomridge-cls] {len(cohort)} datasets, variants={args.variants}, "
        f"cv={args.cv_kind}({args.cv})",
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
            print(f"[aomridge-cls] dataset {row['dataset']} failed: {exc}")
            n = 0
        total += n
        print(f"[aomridge-cls] {row['database_name']}/{row['dataset']} +{n} rows")
    print(f"[aomridge-cls] wrote {total} rows -> {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
