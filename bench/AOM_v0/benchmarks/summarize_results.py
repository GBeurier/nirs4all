"""Summarize benchmark results into LaTeX tables and CSV pivots.

Inputs:

- `--results`: path to the AOM_v0 benchmark `results.csv` (regression or
  classification).
- `--master`: optional path to the regression `master_results.csv` for cross
  comparison.
- `--out`: directory to write tables.

Outputs:

- `summary_per_dataset.csv`: per-dataset best-variant RMSEP / accuracy.
- `summary_per_variant.csv`: aggregate rank table.
- `table_regression_main.tex` (or classification): paper-ready LaTeX table.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def summarize(results_path: str, master_path: Optional[str], out_dir: str) -> int:
    df = pd.read_csv(results_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        (out / "summary_per_dataset.csv").write_text("")
        (out / "summary_per_variant.csv").write_text("")
        return 0
    if "RMSEP" in df.columns and df["RMSEP"].notna().any():
        df["RMSEP"] = pd.to_numeric(df["RMSEP"], errors="coerce")
        per_var = df.dropna(subset=["RMSEP"]).groupby(["aom_variant"]).agg(
            rmsep_mean=("RMSEP", "mean"),
            rmsep_median=("RMSEP", "median"),
            n_runs=("RMSEP", "count"),
        ).reset_index()
        per_var.to_csv(out / "summary_per_variant.csv", index=False)
        wide = df.pivot_table(
            index=["database_name", "dataset"],
            columns="aom_variant",
            values="RMSEP",
            aggfunc="mean",
        ).reset_index()
        wide.to_csv(out / "summary_per_dataset.csv", index=False)
        # LaTeX table
        latex_path = out / "table_regression_main.tex"
        with latex_path.open("w") as f:
            f.write("\\begin{tabular}{lrr}\n\\toprule\n")
            f.write("AOM variant & mean RMSEP & median RMSEP \\\\\n\\midrule\n")
            for _, row in per_var.sort_values("rmsep_mean").iterrows():
                f.write(f"{row['aom_variant']} & {row['rmsep_mean']:.4f} & {row['rmsep_median']:.4f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
    if "balanced_accuracy" in df.columns and df["balanced_accuracy"].notna().any():
        df["balanced_accuracy"] = pd.to_numeric(df["balanced_accuracy"], errors="coerce")
        per_var = df.dropna(subset=["balanced_accuracy"]).groupby(["aom_variant"]).agg(
            balanced_acc_mean=("balanced_accuracy", "mean"),
            balanced_acc_median=("balanced_accuracy", "median"),
            n_runs=("balanced_accuracy", "count"),
        ).reset_index()
        per_var.to_csv(out / "summary_classification_per_variant.csv", index=False)
        latex_path = out / "table_classification_main.tex"
        with latex_path.open("w") as f:
            f.write("\\begin{tabular}{lrr}\n\\toprule\n")
            f.write("AOM-DA variant & mean balanced accuracy & median \\\\\n\\midrule\n")
            for _, row in per_var.sort_values("balanced_acc_mean", ascending=False).iterrows():
                f.write(f"{row['aom_variant']} & {row['balanced_acc_mean']:.4f} & {row['balanced_acc_median']:.4f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
    if master_path:
        master_path_p = Path(master_path)
        if master_path_p.exists():
            master = pd.read_csv(master_path_p)
            pivot = master.pivot_table(
                index=["database_name", "dataset"],
                columns="model",
                values="RMSEP",
                aggfunc="mean",
            ).reset_index()
            pivot.to_csv(out / "master_pivot.csv", index=False)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--master", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return summarize(args.results, args.master, args.out)


if __name__ == "__main__":
    sys.exit(main())
