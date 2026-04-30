"""Publication figures from a benchmark CSV.

Currently produces:
  * fig_per_dataset_delta_vs_ridge.pdf — bar chart of per-dataset Δ% rmsep vs Ridge
  * fig_cumulative_rmsep.pdf           — sorted Δ% line plot
  * fig_cost_vs_precision.pdf          — fit_time vs rmsep scatter

Usage::

  python publication/scripts/make_figures.py \
    --csv bench/nicon_v2/benchmark_runs/stack_extended/results.csv \
    --out bench/nicon_v2/publication/figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df[df["status"].astype(str) == "OK"].copy()


def fig_delta_per_dataset(df: pd.DataFrame, control: str, out_dir: Path) -> Path:
    pivot = df.groupby(["dataset", "variant"])["rmsep"].median().unstack()
    if control not in pivot.columns:
        raise ValueError(f"control {control!r} missing")
    deltas = (pivot.div(pivot[control], axis=0) - 1.0) * 100
    deltas = deltas.drop(columns=[control])

    fig, ax = plt.subplots(figsize=(11, 4))
    deltas.plot.bar(ax=ax)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel(f"Δ rmsep vs {control} (%)")
    ax.set_xlabel("dataset")
    ax.set_title("Per-dataset relative rmsep")
    ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    path = out_dir / "fig_per_dataset_delta_vs_ridge.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_cumulative_rmsep(df: pd.DataFrame, control: str, out_dir: Path) -> Path:
    pivot = df.groupby(["dataset", "variant"])["rmsep"].median().unstack()
    deltas = (pivot.div(pivot[control], axis=0) - 1.0) * 100
    deltas = deltas.drop(columns=[control])

    fig, ax = plt.subplots(figsize=(8, 4))
    for variant in deltas.columns:
        sorted_d = np.sort(deltas[variant].dropna().values)
        ax.plot(sorted_d, label=variant, linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel(f"Δ rmsep vs {control} (%)")
    ax.set_xlabel("rank (sorted ascending)")
    ax.set_title("Cumulative distribution of per-dataset deltas")
    ax.legend(fontsize=7)
    plt.tight_layout()
    path = out_dir / "fig_cumulative_rmsep.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_cost_vs_precision(df: pd.DataFrame, control: str, out_dir: Path) -> Path:
    agg = df.groupby("variant").agg(rmsep=("rmsep", "median"), fit=("fit_time_s", "median")).dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant, row in agg.iterrows():
        ax.scatter(row["fit"], row["rmsep"], label=variant)
        ax.annotate(variant, (row["fit"], row["rmsep"]), fontsize=6, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("median fit time (s)")
    ax.set_ylabel("median rmsep")
    ax.set_title("Cost vs precision")
    plt.tight_layout()
    path = out_dir / "fig_cost_vs_precision.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control", type=str, default="Ridge-baseline")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = load(args.csv)
    figs = [
        fig_delta_per_dataset(df, args.control, args.out),
        fig_cumulative_rmsep(df, args.control, args.out),
        fig_cost_vs_precision(df, args.control, args.out),
    ]
    for p in figs:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
