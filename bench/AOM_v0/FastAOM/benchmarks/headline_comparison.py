"""Headline FastAOM vs AOM_v0 champion comparison.

Reads:
  * FastAOM results CSV (this run's output, e.g. ``paper_aom_fastaom_full60_seed0/results.csv``).
  * AOM_v0 ``full/results.csv`` (already computed at
    ``bench/AOM_v0/benchmark_runs/full/results.csv``, contains 134 model
    variants × 59 datasets).
  * Cohort manifest with ``ref_rmse_pls`` per dataset.

Produces:
  * A leaderboard joining matched ``(database_name, dataset)`` rows across
    FastAOM variants and a configurable set of baseline variants.
  * Per-baseline win counts vs FastAOM (lowest median RMSEP wins on each
    dataset, ties counted as half).
  * A wide CSV with one row per (database, dataset) and one column per model.
  * A Pareto-curve text dump (RMSEP vs fit-time) so we can see if FastAOM
    is dominated.

Usage:

    python bench/AOM_v0/FastAOM/benchmarks/headline_comparison.py \\
        --fastaom bench/scenarios/runs/paper_aom_fastaom_full60_seed0/results.csv \\
        --baseline bench/AOM_v0/benchmark_runs/full/results.csv \\
        --cohort bench/AOM_v0/benchmarks/cohort_regression.csv \\
        --out bench/scenarios/runs/paper_aom_fastaom_full60_seed0/headline.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd


DEFAULT_BASELINE_MODELS = (
    "PLS-standard-numpy",
    "AOM-compact-cv5-numpy",
    "ASLS-AOM-compact-cv5-numpy",
    "SPXY-AOM-compact-cv5-numpy",
    "AOM-default-nipals-adjoint-numpy",
    "POP-nipals-adjoint-numpy",
    "nirs4all-AOM-PLS-default",
    "nirs4all-POP-PLS-default",
)


def _load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["RMSEP"] = pd.to_numeric(df.get("RMSEP", df.get("rmsep")), errors="coerce")
    df["fit_time_s"] = pd.to_numeric(df.get("fit_time_s", df.get("fit_time")), errors="coerce")
    return df


def _aggregate(df: pd.DataFrame, models: Sequence[str] | None = None) -> pd.DataFrame:
    """Median RMSEP per (database, dataset, model), restricted to ok rows."""
    sub = df[df.get("status", "ok") == "ok"]
    if models is not None:
        sub = sub[sub["model"].isin(models)]
    agg = (
        sub.groupby(["database_name", "dataset", "model"], dropna=False)
        .agg(
            median_rmsep=("RMSEP", "median"),
            mean_fit_time=("fit_time_s", "mean"),
            n_seeds=("RMSEP", "size"),
        )
        .reset_index()
    )
    return agg


def _attach_ref_pls(df: pd.DataFrame, cohort_csv: str | Path) -> pd.DataFrame:
    cohort = pd.read_csv(cohort_csv)
    ref = cohort.set_index(["database_name", "dataset"])["ref_rmse_pls"].to_dict()
    df = df.copy()
    df["ref_pls_rmse"] = df.apply(
        lambda r: ref.get((r["database_name"], r["dataset"]), np.nan), axis=1
    )
    df["rel_rmse_vs_pls"] = df["median_rmsep"] / df["ref_pls_rmse"]
    df["delta_vs_pls"] = df["median_rmsep"] - df["ref_pls_rmse"]
    return df


def _win_counts(wide: pd.DataFrame) -> pd.DataFrame:
    """Per-model win count: for each dataset, the model with lowest median RMSEP."""
    cols = [c for c in wide.columns if c not in ("database_name", "dataset")]
    rows = []
    for _, row in wide.iterrows():
        values = {c: row[c] for c in cols if pd.notna(row[c])}
        if not values:
            continue
        winner = min(values, key=values.get)
        rows.append({"database_name": row["database_name"], "dataset": row["dataset"], "winner": winner})
    wins = pd.DataFrame(rows)
    return wins["winner"].value_counts().reset_index().rename(columns={"index": "model", "winner": "model"})


def build_comparison(
    fastaom_csv: str,
    baseline_csv: str,
    cohort_csv: str,
    baseline_models: Sequence[str] = DEFAULT_BASELINE_MODELS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fast_df = _load_csv(fastaom_csv)
    base_df = _load_csv(baseline_csv)
    fast_agg = _aggregate(fast_df)
    fast_models = list(fast_agg["model"].unique())
    base_agg = _aggregate(base_df, models=list(baseline_models))
    merged = pd.concat([fast_agg, base_agg], ignore_index=True)
    merged = _attach_ref_pls(merged, cohort_csv)

    # Per-model summary
    summary = (
        merged.groupby("model", as_index=False)
        .agg(
            n_datasets=("median_rmsep", "size"),
            median_rel_rmse=("rel_rmse_vs_pls", "median"),
            mean_rel_rmse=("rel_rmse_vs_pls", "mean"),
            median_fit_time=("mean_fit_time", "median"),
        )
        .sort_values("median_rel_rmse")
    )

    # Wide table for win counts
    wide = merged.pivot_table(
        index=["database_name", "dataset"], columns="model", values="median_rmsep"
    ).reset_index()
    winners = _win_counts(wide)

    return merged, summary, wide, winners


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Headline FastAOM vs baselines comparison.")
    parser.add_argument("--fastaom", required=True, help="FastAOM results.csv")
    parser.add_argument("--baseline", required=True, help="AOM_v0 full/results.csv")
    parser.add_argument("--cohort", required=True, help="Cohort CSV with ref_rmse_pls")
    parser.add_argument(
        "--baseline-models",
        nargs="+",
        default=list(DEFAULT_BASELINE_MODELS),
        help="Baseline model labels to include",
    )
    parser.add_argument("--out", default=None, help="Output wide CSV (one row per dataset)")
    args = parser.parse_args(argv)

    merged, summary, wide, winners = build_comparison(
        args.fastaom, args.baseline, args.cohort, args.baseline_models
    )

    with pd.option_context("display.max_rows", 100, "display.max_columns", 15, "display.width", 220):
        print("\n=== Per-model summary (median rel-RMSE vs PLS) ===")
        print(summary.to_string(index=False, float_format="%.4f"))
        print("\n=== Dataset winners ===")
        print(winners.to_string(index=False))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        wide.to_csv(out_path, index=False)
        summary_path = out_path.with_name(out_path.stem + "_summary.csv")
        summary.to_csv(summary_path, index=False)
        winners_path = out_path.with_name(out_path.stem + "_winners.csv")
        winners.to_csv(winners_path, index=False)
        print(f"\nWrote {out_path}")
        print(f"Wrote {summary_path}")
        print(f"Wrote {winners_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
