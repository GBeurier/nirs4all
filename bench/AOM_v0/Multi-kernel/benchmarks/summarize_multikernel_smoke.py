"""Summarize the smoke benchmark results.

Reads the raw ``results.csv`` produced by ``run_multikernel_smoke.py`` and
emits:

- ``summary_per_variant.csv`` — one row per variant, with median rel-RMSEP
  vs PLS / Ridge / TabPFN-opt across datasets, plus median fit time.
- ``summary_per_dataset.csv`` — one row per dataset, with the best variant
  and its key metrics.
- ``per_dataset_table.md`` — a Markdown table, useful for the manuscript /
  IMPLEMENTATION_LOG.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    out_dir = args.out_dir or args.results_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results_csv)
    if df.empty:
        print("[summary] empty results.csv; nothing to summarise")
        return 0

    ok = df[df.status == "ok"].copy()

    # --- Per-variant summary ------------------------------------------
    grp = ok.groupby("variant")
    summary = grp.agg(
        median_rel_pls=("rel_rmsep_vs_pls", "median"),
        median_rel_ridge=("rel_rmsep_vs_ridge", "median"),
        median_rel_tabpfn_opt=("rel_rmsep_vs_tabpfn_opt", "median"),
        median_rmsep=("rmsep", "median"),
        median_fit_time_s=("fit_time_s", "median"),
        n_datasets=("rmsep", "count"),
    ).round(4).reset_index().sort_values("median_rel_pls")
    sum_path = out_dir / "summary_per_variant.csv"
    summary.to_csv(sum_path, index=False)
    print(f"[summary] {sum_path}")
    print(summary.to_string(index=False))
    print()

    # --- Per-dataset best variant -------------------------------------
    best_idx = ok.groupby(["dataset_group", "dataset"])["rmsep"].idxmin()
    best = ok.loc[best_idx, [
        "dataset_group", "dataset", "variant",
        "rmsep", "rel_rmsep_vs_pls", "rel_rmsep_vs_ridge",
        "rel_rmsep_vs_tabpfn_opt", "fit_time_s",
    ]].sort_values(["dataset_group", "dataset"]).round(4)
    ds_path = out_dir / "summary_per_dataset.csv"
    best.to_csv(ds_path, index=False)
    print(f"[summary] {ds_path}")
    print(best.to_string(index=False))
    print()

    # --- Markdown table ------------------------------------------------
    lines = ["# Multi-kernel smoke benchmark", ""]
    lines.append("## Per-variant median performance")
    lines.append("")
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Best variant per dataset")
    lines.append("")
    lines.append(best.to_markdown(index=False))
    lines.append("")
    md_path = out_dir / "per_dataset_table.md"
    md_path.write_text("\n".join(lines))
    print(f"[summary] {md_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
