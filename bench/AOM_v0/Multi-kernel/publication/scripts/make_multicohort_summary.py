"""Multi-cohort summary figure: per-variant median rel-PLS across smoke3,
smoke3_branches, curated10, all54_stageA.

Shows the consistency of MKM/mkR rankings across cohorts of different sizes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_COHORTS = {
    "smoke3 (3 ds)": "bench/AOM_v0/Multi-kernel/benchmark_runs/smoke3/results.csv",
    "smoke3+branches (3 ds, 8 var)":
        "bench/AOM_v0/Multi-kernel/benchmark_runs/smoke3_branches/results.csv",
    "curated10 (10 div ds)":
        "bench/AOM_v0/Multi-kernel/benchmark_runs/curated10/results.csv",
    "diverse10 (10 user-curated v2)":
        "bench/AOM_v0/Multi-kernel/benchmark_runs/diverse10/results.csv",
    "all54_stageA (51 ds)":
        "bench/AOM_v0/Multi-kernel/benchmark_runs/all54_stageA/results.csv",
}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("bench/AOM_v0/Multi-kernel/publication/figures"),
    )
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    rows = []
    for label, csv_path in DEFAULT_COHORTS.items():
        path = Path(csv_path)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        ok = df[df.status == "ok"]
        if ok.empty:
            continue
        for variant, sub in ok.groupby("variant"):
            med = float(sub["rel_rmsep_vs_pls"].median())
            n = int(sub["rmsep"].count())
            wins = int((sub["rel_rmsep_vs_pls"] < 1.0).sum())
            rows.append({
                "cohort": label,
                "variant": variant,
                "median_rel_pls": med,
                "n_datasets": n,
                "wins": wins,
                "wins_frac": wins / n if n else float("nan"),
            })
    if not rows:
        print("No data found")
        return 1
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir.parent / "tables" / "multicohort_summary.csv", index=False)

    # Build the "median rel-PLS per variant per cohort" heatmap.
    pivot = summary.pivot(index="variant", columns="cohort", values="median_rel_pls")
    cohort_order = list(DEFAULT_COHORTS.keys())
    pivot = pivot[[c for c in cohort_order if c in pivot.columns]]
    # Order variants by median across cohorts.
    pivot = pivot.loc[pivot.median(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * pivot.shape[1]), max(4, 0.4 * len(pivot))))
    vmin = max(pivot.min().min(), 0.4)
    vmax = min(pivot.max().max(), 2.0)
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=15, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 1.5 or v < 0.7 else "black")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("median relative RMSEP vs PLS (1.0 = PLS reference)")
    ax.set_title("Multi-kernel performance across cohorts (median rel-RMSEP vs PLS)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"fig_multicohort_summary.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[multicohort] wrote fig_multicohort_summary.{{pdf,png}}")

    # Wins-fraction figure.
    pivot_w = summary.pivot(index="variant", columns="cohort", values="wins_frac")
    pivot_w = pivot_w[[c for c in cohort_order if c in pivot_w.columns]]
    pivot_w = pivot_w.loc[pivot_w.median(axis=1).sort_values(ascending=False).index]

    fig2, ax2 = plt.subplots(figsize=(max(8, 1.4 * pivot_w.shape[1]), max(4, 0.4 * len(pivot_w))))
    im2 = ax2.imshow(pivot_w.values, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax2.set_xticks(range(len(pivot_w.columns)))
    ax2.set_xticklabels(pivot_w.columns, rotation=15, ha="right")
    ax2.set_yticks(range(len(pivot_w.index)))
    ax2.set_yticklabels(pivot_w.index)
    for i in range(pivot_w.shape[0]):
        for j in range(pivot_w.shape[1]):
            v = pivot_w.values[i, j]
            if np.isnan(v):
                continue
            ax2.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                     color="white" if v > 0.6 else "black")
    cb2 = fig2.colorbar(im2, ax=ax2)
    cb2.set_label("fraction of datasets where rel-PLS < 1.0 (variant beats PLS)")
    ax2.set_title("Win-rate vs PLS across cohorts")
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(args.out_dir / f"fig_multicohort_winrate.{ext}",
                     dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[multicohort] wrote fig_multicohort_winrate.{{pdf,png}}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
