"""Synthesize smoke-4/smoke-10 results into a comparison table.

Produces a per-dataset, per-variant RMSEP matrix and computes:
- Relative RMSEP vs PLS-standard.
- Relative RMSEP vs `ref_rmse_pls` from cohort_regression.csv.
- Win count vs PLS-standard, vs AOM-PLS-compact.
- Median improvement.

Usage:

    .venv/bin/python bench/AOM_v0/multiview/benchmarks/synth_results.py \
        --results bench/AOM_v0/multiview/results/smoke10.csv \
        --baseline-variant PLS-standard-numpy \
        --aom-variant AOM-PLS-compact-numpy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True)
    parser.add_argument(
        "--cohort", default="bench/AOM_v0/benchmarks/cohort_regression.csv"
    )
    parser.add_argument("--baseline-variant", default="PLS-standard-numpy")
    parser.add_argument("--aom-variant", default="AOM-PLS-compact-numpy")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.results)
    df = df[df["status"] == "ok"].copy()
    df["rmsep"] = pd.to_numeric(df["rmsep"], errors="coerce")
    df = df.dropna(subset=["rmsep"])

    # Pivot to dataset x variant matrix.
    pivot = df.pivot_table(
        index="dataset", columns="variant", values="rmsep", aggfunc="mean"
    )

    cohort = pd.read_csv(args.cohort).set_index("dataset")[
        ["ref_rmse_pls", "ref_rmse_tabpfn_opt", "ref_rmse_ridge", "ref_rmse_catboost"]
    ]

    print("\n=== Absolute RMSEP per dataset x variant ===\n")
    print(pivot.round(4).to_string())

    if args.baseline_variant in pivot.columns:
        baseline = pivot[args.baseline_variant]
        rel_to_baseline = pivot.div(baseline, axis=0)
        print(f"\n=== Relative RMSEP vs {args.baseline_variant} ===\n")
        print(rel_to_baseline.round(3).to_string())

    if args.aom_variant in pivot.columns:
        aom = pivot[args.aom_variant]
        rel_to_aom = pivot.div(aom, axis=0)
        print(f"\n=== Relative RMSEP vs {args.aom_variant} ===\n")
        print(rel_to_aom.round(3).to_string())

    # Win count vs baselines.
    print("\n=== Wins vs PLS-standard / AOM-PLS / TabPFN-opt ===\n")
    summary_rows = []
    for variant in pivot.columns:
        if variant in (args.baseline_variant, args.aom_variant):
            continue
        v_rmse = pivot[variant]
        wins_baseline = (v_rmse < pivot[args.baseline_variant]).sum() if args.baseline_variant in pivot.columns else 0
        wins_aom = (v_rmse < pivot[args.aom_variant]).sum() if args.aom_variant in pivot.columns else 0
        # vs TabPFN-opt (using cohort reference).
        ref_tabpfn = cohort["ref_rmse_tabpfn_opt"].reindex(v_rmse.index)
        wins_tabpfn = (v_rmse < ref_tabpfn).sum()
        # Median rel-RMSEP vs baseline.
        if args.baseline_variant in pivot.columns:
            med_rel = float((v_rmse / pivot[args.baseline_variant]).median())
        else:
            med_rel = float("nan")
        summary_rows.append({
            "variant": variant,
            "wins_vs_baseline": int(wins_baseline),
            "wins_vs_aom": int(wins_aom),
            "wins_vs_tabpfn_opt": int(wins_tabpfn),
            "median_rel_rmsep_vs_baseline": med_rel,
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        "median_rel_rmsep_vs_baseline"
    )
    print(summary.to_string(index=False))

    # Per-dataset winner.
    print("\n=== Per-dataset winner ===\n")
    for ds in pivot.index:
        finite = pivot.loc[ds].dropna()
        if finite.empty:
            continue
        winner = finite.idxmin()
        print(f"  {ds:<48s}  winner={winner:<42s}  rmsep={finite[winner]:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
