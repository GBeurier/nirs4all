"""Analyse a benchmark CSV: per-dataset medians, paired Wilcoxon vs control, ablation pivot.

Usage::

    python publication/scripts/analyze_results.py \
      --csv bench/nicon_v2/benchmark_runs/stack_extended/results.csv \
      --control Ridge-baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--control", type=str, default="Ridge-baseline")
    parser.add_argument("--metric", type=str, default="rmsep")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["status"].astype(str) == "OK"]
    metric = args.metric

    pivot = df.pivot_table(index=["dataset", "seed"], columns="variant", values=metric).dropna()
    print(f"=== Median {metric} per (dataset, variant) ===")
    print(df.groupby(["dataset", "variant"])[metric].median().unstack().round(4).to_string())
    print()

    if args.control not in pivot.columns:
        print(f"control variant {args.control!r} not in CSV")
        return 1

    print(f"=== Paired Wilcoxon vs {args.control} (across {len(pivot)} (dataset, seed) pairs) ===")
    print(f"{'variant':<35} {'med_Δ%':>10} {'wins':>6} {'losses':>7} {'p (Wilcoxon)':>14}")
    control = pivot[args.control]
    rows: list[dict] = []
    for variant in pivot.columns:
        if variant == args.control:
            continue
        treatment = pivot[variant]
        delta = (treatment - control) / control
        delta_pct = delta * 100
        try:
            stat, p = wilcoxon(treatment, control, zero_method="zsplit")
        except ValueError:
            stat, p = float("nan"), float("nan")
        wins = int((treatment < control).sum())
        losses = int((treatment > control).sum())
        rows.append({"variant": variant, "med_delta_pct": float(np.median(delta_pct)),
                     "wins": wins, "losses": losses, "p_wilcoxon": float(p)})
        print(f"{variant:<35} {np.median(delta_pct):>+9.2f}% {wins:>6d} {losses:>7d} {p:>14.4g}")

    print()
    print(f"=== Per-dataset median Δ% vs {args.control} ===")
    per_ds = df.groupby(["dataset", "variant"])[metric].median().unstack()
    if args.control in per_ds.columns:
        ratios = (per_ds.div(per_ds[args.control], axis=0) - 1.0) * 100
        print(ratios.drop(columns=[args.control]).round(2).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
