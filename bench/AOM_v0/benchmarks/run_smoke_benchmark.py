"""Smoke benchmark: run the AOM_v0 benchmark on a small representative subset.

Builds the regression cohort (if missing), filters to up to 5 datasets, and
runs all numpy + torch variants with covariance criterion (fastest). Writes
results into `bench/AOM_v0/benchmark_runs/smoke/results.csv`.

This script is the deterministic "always-runs" smoke check used by
`Phase 11`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.build_cohorts import build_classification_cohort, build_regression_cohort  # noqa: E402
from benchmarks.run_aompls_benchmark import (  # noqa: E402
    CLASSIFICATION_VARIANTS,
    REGRESSION_VARIANTS,
    _existing_keys,
    run_dataset,
)


SMOKE_DATASETS = [
    "Beer_OriginalExtract_60_KS",
    "Rice_Amylose_313_YbasedSplit",
    "ALPINE_P_291_KS",
    "Tleaf_grp70_30",
    "Tablet5_KS",
]


def _select_smoke_rows(cohort_path: str, max_rows: int = 5) -> pd.DataFrame:
    df = pd.read_csv(cohort_path)
    df_ok = df[df["status"] == "ok"].copy()
    if df_ok.empty:
        return df_ok
    preferred = df_ok[df_ok["dataset"].isin(SMOKE_DATASETS)]
    if len(preferred) >= max_rows:
        return preferred.head(max_rows)
    selected = preferred.copy()
    remaining = max_rows - len(selected)
    others = df_ok[~df_ok["dataset"].isin(SMOKE_DATASETS)].head(remaining)
    return pd.concat([selected, others], ignore_index=True)


def run_smoke(workspace: str = "bench/AOM_v0/benchmark_runs/smoke") -> int:
    cohort_reg_path = "bench/AOM_v0/benchmarks/cohort_regression.csv"
    cohort_clf_path = "bench/AOM_v0/benchmarks/cohort_classification.csv"
    if not Path(cohort_reg_path).exists():
        build_regression_cohort(out_path=cohort_reg_path)
    if not Path(cohort_clf_path).exists():
        build_classification_cohort(out_path=cohort_clf_path)
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    results_reg = workspace / "results.csv"
    results_clf = workspace / "results_classification.csv"
    smoke_reg = _select_smoke_rows(cohort_reg_path, max_rows=3)
    smoke_clf = _select_smoke_rows(cohort_clf_path, max_rows=2)
    print(f"[smoke] regression rows: {len(smoke_reg)}, classification rows: {len(smoke_clf)}")
    # Use covariance criterion for speed; max_components=8 to keep wall-clock low.
    existing = _existing_keys(results_reg)
    total_reg = 0
    for _, row in smoke_reg.iterrows():
        n = run_dataset(
            cohort_row=row,
            variants=REGRESSION_VARIANTS,
            results_path=results_reg,
            seeds=[0],
            criterion="covariance",
            max_components=8,
            cv=3,
            classification=False,
            existing_keys=existing,
        )
        total_reg += n
        print(f"[smoke regression] {row['database_name']}/{row['dataset']} +{n} rows")
    existing_clf = _existing_keys(results_clf)
    total_clf = 0
    for _, row in smoke_clf.iterrows():
        n = run_dataset(
            cohort_row=row,
            variants=CLASSIFICATION_VARIANTS,
            results_path=results_clf,
            seeds=[0],
            criterion="covariance",
            max_components=6,
            cv=3,
            classification=True,
            existing_keys=existing_clf,
        )
        total_clf += n
        print(f"[smoke classification] {row['database_name']}/{row['dataset']} +{n} rows")
    print(f"smoke benchmark wrote {total_reg} regression rows and {total_clf} classification rows")
    print(f"  results: {results_reg}")
    print(f"  classification results: {results_clf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_smoke())
