"""Compare Stack-5 results vs the per-dataset best of the 6-variant baseline.

Reads:
- diverse10/results.csv (the 6-variant baseline + manual Ridge-raw on species)
- diverse8_stack5/results.csv (Stack-5 only, 8 datasets)

Produces:
- Per-dataset comparison: best baseline variant + Stack-5 → which wins?
- Per-cohort medians.
- Stack-5 meta-coefficients (one row per dataset).

Usage:

```
.venv/bin/python bench/AOM_v0/Multi-kernel/benchmarks/compare_stack_vs_baselines.py
```
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
        "--stack-csv", type=Path,
        default=Path("bench/AOM_v0/Multi-kernel/benchmark_runs/diverse8_stack5/results.csv"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("bench/AOM_v0/Multi-kernel/benchmark_runs/diverse8_stack5/comparison.md"),
    )
    args = parser.parse_args(argv)

    bl = pd.read_csv(args.baseline_csv)
    bl_ok = bl[bl.status == "ok"]
    # Best baseline per dataset.
    bl_idx = bl_ok.dropna(subset=["rmsep"]).groupby(["dataset_group", "dataset"])["rmsep"].idxmin()
    bl_best = bl_ok.loc[bl_idx, [
        "dataset_group", "dataset", "variant", "rmsep",
        "rel_rmsep_vs_pls", "rel_rmsep_vs_ridge", "rel_rmsep_vs_tabpfn_opt",
        "fit_time_s",
    ]].rename(columns={
        "variant": "best_baseline_variant",
        "rmsep": "rmsep_baseline",
        "rel_rmsep_vs_pls": "baseline_rel_pls",
        "rel_rmsep_vs_ridge": "baseline_rel_ridge",
        "rel_rmsep_vs_tabpfn_opt": "baseline_rel_tabpfn_opt",
        "fit_time_s": "baseline_fit_t",
    })

    if args.stack_csv.exists():
        st = pd.read_csv(args.stack_csv)
        st_ok = st[st.status == "ok"]
        if len(st_ok):
            st_view = st_ok[[
                "dataset_group", "dataset", "rmsep",
                "rel_rmsep_vs_pls", "rel_rmsep_vs_ridge", "rel_rmsep_vs_tabpfn_opt",
                "fit_time_s", "boundary_components",
            ]].rename(columns={
                "rmsep": "rmsep_stack",
                "rel_rmsep_vs_pls": "stack_rel_pls",
                "rel_rmsep_vs_ridge": "stack_rel_ridge",
                "rel_rmsep_vs_tabpfn_opt": "stack_rel_tabpfn_opt",
                "fit_time_s": "stack_fit_t",
                "boundary_components": "stack_meta_coefs",
            })
            merged = pd.merge(bl_best, st_view, on=["dataset_group", "dataset"], how="left")
        else:
            merged = bl_best.copy()
            merged["rmsep_stack"] = float("nan")
    else:
        merged = bl_best.copy()
        merged["rmsep_stack"] = float("nan")

    # Compute who wins per dataset.
    if "rmsep_stack" in merged.columns:
        merged["stack_better"] = merged["rmsep_stack"] < merged["rmsep_baseline"]
        merged["stack_lift_pct"] = (
            (merged["rmsep_baseline"] - merged["rmsep_stack"]) / merged["rmsep_baseline"] * 100
        ).round(2)

    # Save / print.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out.parent / "comparison.csv", index=False)
    md = ["# Stack-5 vs Best-baseline-per-dataset", ""]
    cols = [
        "dataset", "best_baseline_variant", "baseline_rel_pls",
        "baseline_rel_tabpfn_opt", "stack_rel_pls", "stack_rel_tabpfn_opt",
        "stack_lift_pct",
    ]
    cols = [c for c in cols if c in merged.columns]
    md.append(merged[cols].sort_values("baseline_rel_tabpfn_opt").to_markdown(index=False))
    md.append("")
    md.append(f"**Median baseline rel-PLS:** {merged['baseline_rel_pls'].median():.3f}")
    md.append(f"**Median baseline rel-TabPFN-opt:** {merged['baseline_rel_tabpfn_opt'].median():.3f}")
    if merged.get("stack_rel_pls", pd.Series()).notna().any():
        md.append(f"**Median Stack-5 rel-PLS:** {merged['stack_rel_pls'].median():.3f}")
        md.append(f"**Median Stack-5 rel-TabPFN-opt:** {merged['stack_rel_tabpfn_opt'].median():.3f}")
        md.append(f"**Stack wins on:** "
                  f"{merged['stack_better'].sum()} / {merged['stack_better'].notna().sum()} datasets")
    args.out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\n=> {args.out}")
    print(f"=> {args.out.parent / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
