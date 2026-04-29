"""Summarise AOM-Ridge benchmark results.

Computes per-variant median relative RMSEP versus Ridge-raw and PLS-standard,
win counts, failure counts, and median fit/predict times. Reads from the CSV
produced by ``run_aomridge_benchmark.py``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def summarise(results_path: Path) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    df_ok = df[df["status"] == "ok"].copy()
    if df_ok.empty:
        return pd.DataFrame()
    df_ok["rmsep"] = pd.to_numeric(df_ok["rmsep"], errors="coerce")
    df_ok["relative_rmsep_vs_ridge_raw"] = pd.to_numeric(
        df_ok["relative_rmsep_vs_ridge_raw"], errors="coerce"
    )
    df_ok["relative_rmsep_vs_pls_standard"] = pd.to_numeric(
        df_ok["relative_rmsep_vs_pls_standard"], errors="coerce"
    )
    df_ok["fit_time_s"] = pd.to_numeric(df_ok["fit_time_s"], errors="coerce")
    df_ok["predict_time_s"] = pd.to_numeric(df_ok["predict_time_s"], errors="coerce")

    failures = df[df["status"] != "ok"].groupby("variant").size().rename("failures")
    summary = (
        df_ok.groupby("variant")
        .agg(
            median_relative_rmsep_vs_ridge_raw=("relative_rmsep_vs_ridge_raw", "median"),
            median_relative_rmsep_vs_pls=("relative_rmsep_vs_pls_standard", "median"),
            wins_vs_ridge_raw=(
                "relative_rmsep_vs_ridge_raw",
                lambda s: int((s < 1.0).sum()),
            ),
            wins_vs_pls=(
                "relative_rmsep_vs_pls_standard",
                lambda s: int((s < 1.0).sum()),
            ),
            median_fit_time_s=("fit_time_s", "median"),
            median_predict_time_s=("predict_time_s", "median"),
            n_runs=("rmsep", "size"),
        )
        .join(failures, how="left")
        .fillna({"failures": 0})
        .sort_values("median_relative_rmsep_vs_ridge_raw")
    )
    summary["failures"] = summary["failures"].astype(int)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise AOM-Ridge results")
    parser.add_argument("--results", required=True, help="path to results.csv")
    parser.add_argument(
        "--out",
        default=None,
        help="optional path to write the summary CSV (default: stdout)",
    )
    args = parser.parse_args(argv)
    summary = summarise(Path(args.results))
    if summary.empty:
        print("no successful runs to summarise")
        return 0
    if args.out:
        summary.to_csv(args.out)
        print(f"summary -> {args.out}")
    else:
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
