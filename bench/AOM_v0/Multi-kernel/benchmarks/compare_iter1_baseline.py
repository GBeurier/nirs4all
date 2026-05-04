"""Compare Iter 1 (active-screened default-15) vs the diverse10 baseline.

Reads:
- diverse10/results.csv (compact-bank baseline, 9 datasets)
- iter1_active15/results.csv (default+active15, 8 datasets)

Produces a per-dataset comparison: best baseline variant vs best Iter 1
variant for each dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-csv", type=Path,
        default=Path("bench/AOM_v0/Multi-kernel/benchmark_runs/diverse10/results.csv"),
    )
    parser.add_argument(
        "--iter-csv", type=Path,
        default=Path("bench/AOM_v0/Multi-kernel/benchmark_runs/iter1_active15/results.csv"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("bench/AOM_v0/Multi-kernel/benchmark_runs/iter1_active15/comparison.md"),
    )
    args = parser.parse_args(argv)

    bl = pd.read_csv(args.baseline_csv)
    it = pd.read_csv(args.iter_csv) if args.iter_csv.exists() else pd.DataFrame()

    bl_ok = bl[bl.status == "ok"].dropna(subset=["rmsep", "rel_rmsep_vs_pls"]).copy()
    bl_idx = bl_ok.groupby(["dataset_group", "dataset"])["rmsep"].idxmin()
    bl_best = bl_ok.loc[bl_idx, [
        "dataset_group", "dataset", "variant",
        "rmsep", "rel_rmsep_vs_pls", "rel_rmsep_vs_ridge", "rel_rmsep_vs_tabpfn_opt",
    ]].rename(columns={
        "variant": "baseline_variant",
        "rmsep": "baseline_rmsep",
        "rel_rmsep_vs_pls": "baseline_rel_pls",
        "rel_rmsep_vs_ridge": "baseline_rel_ridge",
        "rel_rmsep_vs_tabpfn_opt": "baseline_rel_tabpfn_opt",
    })

    if len(it):
        it_ok = it[it.status == "ok"].dropna(subset=["rmsep", "rel_rmsep_vs_pls"]).copy()
        it_idx = it_ok.groupby(["dataset_group", "dataset"])["rmsep"].idxmin()
        it_best = it_ok.loc[it_idx, [
            "dataset_group", "dataset", "variant",
            "rmsep", "rel_rmsep_vs_pls", "rel_rmsep_vs_ridge", "rel_rmsep_vs_tabpfn_opt",
        ]].rename(columns={
            "variant": "iter1_variant",
            "rmsep": "iter1_rmsep",
            "rel_rmsep_vs_pls": "iter1_rel_pls",
            "rel_rmsep_vs_ridge": "iter1_rel_ridge",
            "rel_rmsep_vs_tabpfn_opt": "iter1_rel_tabpfn_opt",
        })
        merged = pd.merge(bl_best, it_best, on=["dataset_group", "dataset"], how="left")
        merged["lift_pct"] = (
            (merged["baseline_rmsep"] - merged["iter1_rmsep"])
            / merged["baseline_rmsep"] * 100
        ).round(2)
    else:
        merged = bl_best.copy()
        merged["iter1_variant"] = None
        merged["iter1_rmsep"] = None
        merged["iter1_rel_pls"] = None
        merged["iter1_rel_tabpfn_opt"] = None
        merged["lift_pct"] = None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out.parent / "comparison.csv", index=False)

    md = ["# Iter 1 (default-active15) vs baseline (compact)", ""]
    cols = [
        "dataset", "baseline_variant", "baseline_rel_pls",
        "baseline_rel_tabpfn_opt", "iter1_variant", "iter1_rel_pls",
        "iter1_rel_tabpfn_opt", "lift_pct",
    ]
    cols = [c for c in cols if c in merged.columns]
    md.append(merged[cols].sort_values("baseline_rel_tabpfn_opt").to_markdown(index=False))
    md.append("")

    if "iter1_rel_pls" in merged.columns and merged["iter1_rel_pls"].notna().any():
        md.append("## Summary statistics")
        md.append("")
        md.append(f"- median baseline rel-PLS:        **{merged['baseline_rel_pls'].median():.3f}**")
        md.append(f"- median Iter 1 rel-PLS:          **{merged['iter1_rel_pls'].median():.3f}**")
        md.append(f"- median baseline rel-TabPFN-opt: **{merged['baseline_rel_tabpfn_opt'].median():.3f}**")
        md.append(f"- median Iter 1 rel-TabPFN-opt:   **{merged['iter1_rel_tabpfn_opt'].median():.3f}**")
        md.append(f"- median lift_pct:                **{merged['lift_pct'].median():.2f} %**")
        md.append(f"- iter1 wins on:                  **{(merged['lift_pct'] > 0).sum()} / {merged['lift_pct'].notna().sum()}** datasets")
    args.out.write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
