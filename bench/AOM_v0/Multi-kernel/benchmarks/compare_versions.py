"""Compare multiple benchmark cohorts side-by-side.

Reads any number of `results.csv` files (each with the same schema) and
computes per-dataset best variant + per-cohort medians, joined into one
table.

Usage:

```
.venv/bin/python bench/AOM_v0/Multi-kernel/benchmarks/compare_versions.py \
  --csv "diverse10:bench/AOM_v0/Multi-kernel/benchmark_runs/diverse10/results.csv" \
  --csv "iter1:bench/AOM_v0/Multi-kernel/benchmark_runs/iter1_active15/results.csv" \
  --out bench/AOM_v0/Multi-kernel/benchmark_runs/iter1_active15/comparison.md
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def best_per_dataset(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df.status == "ok"].dropna(subset=["rmsep", "rel_rmsep_vs_pls"])
    if ok.empty:
        return ok
    idx = ok.groupby(["dataset_group", "dataset"])["rmsep"].idxmin()
    return ok.loc[idx, [
        "dataset_group", "dataset", "variant",
        "rmsep", "rel_rmsep_vs_pls", "rel_rmsep_vs_ridge", "rel_rmsep_vs_tabpfn_opt",
        "fit_time_s",
    ]].copy()


def median_per_cohort(df: pd.DataFrame) -> dict:
    ok = df[df.status == "ok"].dropna(subset=["rel_rmsep_vs_pls"])
    if ok.empty:
        return {}
    best = best_per_dataset(df)
    return {
        "n_datasets": int(best.shape[0]),
        "median_best_rel_pls": float(best["rel_rmsep_vs_pls"].median()),
        "median_best_rel_tabpfn_opt": float(best["rel_rmsep_vs_tabpfn_opt"].median()),
        "wins_vs_pls": int((best["rel_rmsep_vs_pls"] < 1.0).sum()),
        "wins_vs_tabpfn_opt": int((best["rel_rmsep_vs_tabpfn_opt"] < 1.0).sum()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="append", required=True,
                        help="format `name:/path/to/results.csv`")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    cohorts = {}
    for spec in args.csv:
        name, path = spec.split(":", 1)
        cohorts[name] = pd.read_csv(path)

    md = ["# Multi-cohort comparison", "",
          "## Per-cohort summary (best variant per dataset)", ""]
    rows = []
    for name, df in cohorts.items():
        stats = median_per_cohort(df)
        if stats:
            stats["cohort"] = name
            rows.append(stats)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary[["cohort", "n_datasets", "median_best_rel_pls",
                           "median_best_rel_tabpfn_opt",
                           "wins_vs_pls", "wins_vs_tabpfn_opt"]]
        md.append(summary.to_markdown(index=False, floatfmt=".3f"))
    md.append("")

    # Per-dataset side-by-side, computing best variant in each cohort.
    md.append("## Per-dataset best variant per cohort", "")
    bests = {name: best_per_dataset(df) for name, df in cohorts.items()}
    all_datasets = set()
    for b in bests.values():
        all_datasets.update(zip(b.dataset_group, b.dataset))
    rows = []
    for grp, ds in sorted(all_datasets):
        row = {"dataset_group": grp, "dataset": ds}
        for name, b in bests.items():
            sub = b[(b.dataset_group == grp) & (b.dataset == ds)]
            if not sub.empty:
                r = sub.iloc[0]
                row[f"{name}_variant"] = r["variant"]
                row[f"{name}_rel_pls"] = r["rel_rmsep_vs_pls"]
                row[f"{name}_rel_tabpfn"] = r["rel_rmsep_vs_tabpfn_opt"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.out.parent / "per_dataset_compare.csv", index=False)
    md.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"))
    md.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
