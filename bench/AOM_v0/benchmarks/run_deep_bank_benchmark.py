"""Deep-bank benchmark: order 3 / 4 composition chains.

Compares:

- AOM-default (100 ops, our parity-validated baseline) — REFERENCE
- AOM-extended (default + Whittaker)
- AOM-deep3 (default + order-3 chains, 116 ops)
- AOM-deep4 (default + order-3 + order-4 chains, 132 ops)
- ActiveSuperblock with deep3 / deep4
- Operator explorer (beam search, max_degree=2, 3, 4)

Records `fit_time_s` and `predict_time_s` for every (dataset, variant)
combination so we can plot cost-vs-precision curves.

Usage:

    PYTHONPATH=bench/AOM_v0 .venv/bin/python \
      bench/AOM_v0/benchmarks/run_deep_bank_benchmark.py \
      --workspace bench/AOM_v0/benchmark_runs/deep --limit 20

Output: `<workspace>/results.csv` (resumable).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.build_cohorts import build_regression_cohort  # noqa: E402
from benchmarks.run_aompls_benchmark import _existing_keys, run_dataset  # noqa: E402


DEEP_VARIANTS = [
    {"label": "PLS-standard-numpy", "kind": "regression", "selection": "none",
     "engine": "pls_standard", "operator_bank": "identity", "backend": "numpy"},
    {"label": "AOM-default-nipals-adjoint-numpy", "kind": "regression", "selection": "global",
     "engine": "nipals_adjoint", "operator_bank": "default", "backend": "numpy"},
    {"label": "AOM-extended-nipals-adjoint-numpy", "kind": "regression", "selection": "global",
     "engine": "nipals_adjoint", "operator_bank": "extended", "backend": "numpy"},
    {"label": "AOM-deep3-nipals-adjoint-numpy", "kind": "regression", "selection": "global",
     "engine": "nipals_adjoint", "operator_bank": "deep3", "backend": "numpy"},
    {"label": "AOM-deep4-nipals-adjoint-numpy", "kind": "regression", "selection": "global",
     "engine": "nipals_adjoint", "operator_bank": "deep4", "backend": "numpy"},
    {"label": "ActiveSuperblock-deep3-numpy", "kind": "regression", "selection": "active_superblock",
     "engine": "simpls_covariance", "operator_bank": "deep3", "backend": "numpy"},
    {"label": "AOM-explorer-deep3-numpy", "kind": "regression", "selection": "explorer_global",
     "engine": "simpls_covariance", "operator_bank": "explorer", "backend": "numpy"},
    {"label": "nirs4all-AOM-PLS-default", "kind": "regression", "selection": "external",
     "engine": "nirs4all_aom", "operator_bank": "production_default", "backend": "numpy"},
]


def _select_cohort(cohort_path: str, limit: int = 20, max_n: int = 1500) -> pd.DataFrame:
    df = pd.read_csv(cohort_path)
    df_ok = df[df["status"] == "ok"].copy()
    df_ok = df_ok[df_ok["n_train"].fillna(0) <= max_n].sort_values("n_train")
    if limit > 0:
        df_ok = df_ok.head(limit)
    return df_ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="bench/AOM_v0/benchmark_runs/deep")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-n-train", type=int, default=1500)
    parser.add_argument("--max-components", type=int, default=15)
    parser.add_argument("--criterion", default="holdout")
    args = parser.parse_args(argv)
    cohort_path = "bench/AOM_v0/benchmarks/cohort_regression.csv"
    if not Path(cohort_path).exists():
        build_regression_cohort(out_path=cohort_path)
    workspace = Path(args.workspace); workspace.mkdir(parents=True, exist_ok=True)
    results = workspace / "results.csv"
    cohort = _select_cohort(cohort_path, limit=args.limit, max_n=args.max_n_train)
    print(f"[deep] {len(cohort)} datasets x {len(DEEP_VARIANTS)} variants")
    existing = _existing_keys(results)
    total = 0
    for _, row in cohort.iterrows():
        n = run_dataset(
            cohort_row=row, variants=DEEP_VARIANTS, results_path=results, seeds=[0],
            criterion=args.criterion, max_components=args.max_components, cv=3,
            classification=False, existing_keys=existing,
        )
        total += n
        print(f"[deep] {row['database_name']}/{row['dataset']} (n={row['n_train']}) +{n} rows")
    print(f"[deep] wrote {total} new rows -> {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
