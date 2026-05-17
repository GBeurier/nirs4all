"""Compare FastAOM results against existing AOM_v0 baselines.

Takes one or more results CSV files (each following the AOM_v0 master
schema: ``database_name, dataset, model, RMSEP, seed, ...``) and prints
a side-by-side leaderboard per ``(database_name, dataset)`` plus a
relative-RMSEP summary against the cohort's PLS reference.

Typical use after running the FastAOM smoke benchmark::

    python bench/AOM_v0/FastAOM/benchmarks/compare_to_baselines.py \\
        --files \\
            bench/scenarios/runs/paper_aom_fastaom_seed0/results.csv \\
            bench/AOM_v0/benchmark_runs/smoke_old_11ds/results.csv \\
        --cohort bench/AOM_v0/Ridge/benchmark_runs/diverse11_cohort.csv \\
        --out bench/scenarios/runs/paper_aom_fastaom_seed0/comparison.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd


def load_results(files: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for f in files:
        path = Path(f)
        if not path.exists():
            print(f"WARNING: missing {path}", file=sys.stderr)
            continue
        df = pd.read_csv(path, dtype=str)
        if df.empty:
            continue
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        raise SystemExit("No non-empty input files")
    out = pd.concat(frames, ignore_index=True)
    return out


def coerce_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (database, dataset, model) with median + min RMSEP across seeds."""
    df = coerce_numeric(df, ["RMSEP", "seed", "fit_time_s"])
    ok = df[(df.get("status", "ok") == "ok") & df["RMSEP"].notna()]
    grouped = ok.groupby(
        ["database_name", "dataset", "model"], dropna=False
    ).agg(
        median_rmsep=("RMSEP", "median"),
        min_rmsep=("RMSEP", "min"),
        max_rmsep=("RMSEP", "max"),
        n_seeds=("RMSEP", "size"),
        mean_fit_time=("fit_time_s", "mean"),
    ).reset_index()
    return grouped


def add_relative_rmse(
    leaderboard: pd.DataFrame,
    cohort_csv: str,
) -> pd.DataFrame:
    cohort = pd.read_csv(cohort_csv)
    ref = cohort.set_index(["database_name", "dataset"])["ref_rmse_pls"].to_dict()
    leaderboard = leaderboard.copy()
    leaderboard["ref_pls_rmse"] = leaderboard.apply(
        lambda r: ref.get((r["database_name"], r["dataset"]), np.nan), axis=1
    )
    leaderboard["rel_rmse_vs_pls"] = leaderboard["median_rmsep"] / leaderboard["ref_pls_rmse"]
    leaderboard["delta_rmse_vs_pls"] = leaderboard["median_rmsep"] - leaderboard["ref_pls_rmse"]
    return leaderboard


def best_per_dataset(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """For each (database, dataset), the model with the lowest median RMSEP."""
    return (
        leaderboard.sort_values(["database_name", "dataset", "median_rmsep"])
        .groupby(["database_name", "dataset"], as_index=False)
        .head(1)
    )


def win_counts(leaderboard: pd.DataFrame, ref_pls_only: bool = True) -> pd.DataFrame:
    """Count how many datasets each model wins (lowest median RMSEP)."""
    best = best_per_dataset(leaderboard)
    counts = best["model"].value_counts().reset_index()
    counts.columns = ["model", "n_wins"]
    return counts


def fastaom_summary(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """Per-model summary: median rel-RMSE, number of datasets, mean fit time."""
    summary = (
        leaderboard.groupby("model", as_index=False)
        .agg(
            n_datasets=("median_rmsep", "size"),
            median_rel_rmse=("rel_rmse_vs_pls", "median"),
            mean_rel_rmse=("rel_rmse_vs_pls", "mean"),
            mean_delta_rmse=("delta_rmse_vs_pls", "mean"),
            mean_fit_time=("mean_fit_time", "mean"),
        )
        .sort_values("median_rel_rmse")
    )
    return summary


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare FastAOM results to AOM_v0 baselines.")
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="One or more results.csv files to merge.",
    )
    parser.add_argument(
        "--cohort",
        required=True,
        help="Cohort CSV with ref_rmse_pls column.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV for the merged leaderboard.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Print top-N models per dataset in stdout summary.",
    )
    args = parser.parse_args(argv)

    df = load_results(args.files)
    leaderboard = build_leaderboard(df)
    leaderboard = add_relative_rmse(leaderboard, args.cohort)
    summary = fastaom_summary(leaderboard)
    winners = win_counts(leaderboard)

    print("\n=== Per-model summary (median rel-RMSE vs PLS) ===")
    with pd.option_context("display.max_rows", 80, "display.max_columns", 12, "display.width", 200):
        print(summary.to_string(index=False, float_format="%.4f"))

    print("\n=== Dataset winners (lowest median RMSEP) ===")
    with pd.option_context("display.max_rows", 80, "display.max_columns", 12, "display.width", 200):
        print(winners.to_string(index=False))

    print("\n=== Top FastAOM rows ===")
    fast_only = leaderboard[leaderboard["model"].str.startswith("FastAOM")]
    if not fast_only.empty:
        with pd.option_context("display.max_rows", 80, "display.max_columns", 12, "display.width", 220):
            print(
                fast_only.sort_values("rel_rmse_vs_pls").head(40).to_string(
                    index=False, float_format="%.4f"
                )
            )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(out_path, index=False)
        print(f"\nMerged leaderboard written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
