"""Generate LaTeX / CSV tables for the multi-kernel paper.

Outputs:

- ``table_per_variant.{csv,tex}`` — median rel-RMSEP per variant.
- ``table_per_dataset.{csv,tex}`` — best variant per dataset.
- ``relative_rmsep_pivot.csv`` — full variant × dataset pivot of
  rel-RMSEP vs PLS.

Usage:

```bash
.venv/bin/python bench/AOM_v0/Multi-kernel/publication/scripts/make_multikernel_tables.py \
  bench/AOM_v0/Multi-kernel/benchmark_runs/smoke3_branches/results.csv \
  --out-dir bench/AOM_v0/Multi-kernel/publication/tables
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", type=Path)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("bench/AOM_v0/Multi-kernel/publication/tables"),
    )
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results_csv)
    if df.empty:
        print("[tables] empty results csv; nothing to do")
        return 0
    ok = df[df.status == "ok"].copy()

    # Per-variant median table.
    grp = ok.groupby("variant")
    summary = grp.agg(
        median_rel_pls=("rel_rmsep_vs_pls", "median"),
        median_rel_ridge=("rel_rmsep_vs_ridge", "median"),
        median_rel_tabpfn_opt=("rel_rmsep_vs_tabpfn_opt", "median"),
        wins_vs_pls=("rel_rmsep_vs_pls", lambda s: int((s < 1.0).sum())),
        n_datasets=("rmsep", "count"),
        median_fit_time_s=("fit_time_s", "median"),
    ).round(4).reset_index().sort_values("median_rel_pls")
    csv_path = args.out_dir / "table_per_variant_multikernel.csv"
    summary.to_csv(csv_path, index=False)
    print(f"[tables] {csv_path}")
    tex_path = args.out_dir / "table_per_variant_multikernel.tex"
    summary.to_latex(tex_path, index=False, float_format="%.3f")
    print(f"[tables] {tex_path}")

    # Best per dataset.
    best_idx = ok.groupby(["dataset_group", "dataset"])["rmsep"].idxmin()
    best = ok.loc[best_idx, [
        "dataset_group", "dataset", "variant", "rmsep",
        "rel_rmsep_vs_pls", "rel_rmsep_vs_ridge", "rel_rmsep_vs_tabpfn_opt",
    ]].sort_values(["dataset_group", "dataset"]).round(4)
    bcsv = args.out_dir / "table_best_per_dataset_multikernel.csv"
    best.to_csv(bcsv, index=False)
    print(f"[tables] {bcsv}")
    btex = args.out_dir / "table_best_per_dataset_multikernel.tex"
    best.to_latex(btex, index=False, float_format="%.3f")
    print(f"[tables] {btex}")

    # Pivot table.
    pivot = ok.pivot_table(
        index="variant", columns="dataset", values="rel_rmsep_vs_pls",
        aggfunc="median",
    ).round(4)
    pcsv = args.out_dir / "relative_rmsep_pivot_multikernel.csv"
    pivot.to_csv(pcsv)
    print(f"[tables] {pcsv}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
