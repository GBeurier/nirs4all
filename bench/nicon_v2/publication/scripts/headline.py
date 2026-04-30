"""One-line headline numbers for the manuscript.

Reads any nicon_v2 benchmark CSV and prints the median Δ% rmsep of a chosen
variant against every available reference, plus the paired Wilcoxon p vs the
internal Ridge baseline. Designed for "when you wake up tomorrow morning, run
this once and you have the manuscript headline".

Usage::

    python publication/scripts/headline.py \
      --csv bench/nicon_v2/benchmark_runs/stack_curated/results.csv \
      --variant Stack-Ridge-PLS-V1c
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


REFS = {
    "internal Ridge (paired)": ("Ridge-baseline", "paired"),
    "internal PLS (paired)":   ("PLS-baseline",   "paired"),
    "paper Ridge":             ("ref_rmse_paper_ridge", "ref"),
    "paper PLS":               ("ref_rmse_pls",         "ref"),
    "paper TabPFN-raw":        ("ref_rmse_tabpfn_raw",  "ref"),
    "paper TabPFN-opt":        ("ref_rmse_tabpfn_opt",  "ref"),
    "paper CNN":               ("ref_rmse_cnn",         "ref"),
    "paper CatBoost":          ("ref_rmse_catboost",    "ref"),
    "AOM-Ridge curated best":  ("ref_rmse_aom_ridge_curated_best", "ref"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--variant", type=str, default="Stack-Ridge-PLS-V1c")
    parser.add_argument("--metric", type=str, default="rmsep")
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"[ERR] csv missing: {args.csv}")
        return 1
    df = pd.read_csv(args.csv)
    df = df[df["status"].astype(str) == "OK"]

    if args.variant not in df["variant"].unique():
        print(f"[ERR] variant {args.variant!r} missing; have {sorted(df['variant'].unique())}")
        return 1

    print(f"=== nicon_v2 headline — variant: {args.variant} ===")
    print(f"  csv: {args.csv}")
    print(f"  rows total: {len(df)}, OK: {(df['status']=='OK').sum()}, datasets: {df['dataset'].nunique()}, seeds: {sorted(df['seed'].unique())}")
    print()

    treatment = df[df["variant"] == args.variant]
    print(f"  median {args.metric} ({args.variant}): {treatment[args.metric].median():.4f}")
    print()

    pivot = df.pivot_table(index=["dataset", "seed"], columns="variant", values=args.metric).dropna()
    if args.variant not in pivot.columns:
        return 1
    treat_arr = pivot[args.variant]

    print(f"=== reference comparisons (median Δ% rmsep) ===")
    print(f"  {'reference':<30} {'Δ%':>10} {'n_ds':>5} {'n_pairs':>8} {'wins':>6} {'losses':>7} {'wilcoxon_p':>12}")
    print(f"  {'-'*30} {'-'*10} {'-'*5} {'-'*8} {'-'*6} {'-'*7} {'-'*12}")

    for label, (key, kind) in REFS.items():
        if kind == "paired":
            if key not in pivot.columns:
                continue
            ctrl = pivot[key]
            delta = (treat_arr - ctrl) / ctrl
            wins = int((treat_arr < ctrl).sum())
            losses = int((treat_arr > ctrl).sum())
            try:
                _, p = wilcoxon(treat_arr, ctrl, zero_method="zsplit")
            except ValueError:
                p = float("nan")
            n_ds = pivot.index.get_level_values(0).nunique()
            n_pairs = len(pivot)
            print(f"  {label:<30} {np.median(delta)*100:>+9.2f}% {n_ds:>5d} {n_pairs:>8d} {wins:>6d} {losses:>7d} {p:>12.4g}")
        else:
            sub = df[df["variant"] == args.variant].dropna(subset=[key])
            if sub.empty:
                continue
            delta_pct = (sub[args.metric] - sub[key]) / sub[key] * 100
            n_ds = sub["dataset"].nunique()
            n_pairs = len(sub)
            print(f"  {label:<30} {np.median(delta_pct):>+9.2f}% {n_ds:>5d} {n_pairs:>8d} {'-':>6} {'-':>7} {'descriptive':>12}")

    print()
    # Per-dataset wins
    print(f"=== per-dataset rmsep ({args.variant} vs Ridge-baseline) ===")
    if "Ridge-baseline" in pivot.columns:
        ds_med = df[df["variant"].isin([args.variant, "Ridge-baseline"])].groupby(["dataset", "variant"])[args.metric].median().unstack()
        if "Ridge-baseline" in ds_med.columns and args.variant in ds_med.columns:
            deltas = (ds_med[args.variant] - ds_med["Ridge-baseline"]) / ds_med["Ridge-baseline"] * 100
            wins = (deltas < 0).sum()
            losses = (deltas > 0).sum()
            n = len(deltas)
            print(f"  per-dataset wins / total: {wins} / {n}")
            print(f"  best:  {deltas.idxmin():<40} Δ = {deltas.min():+.2f}%")
            print(f"  worst: {deltas.idxmax():<40} Δ = {deltas.max():+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
