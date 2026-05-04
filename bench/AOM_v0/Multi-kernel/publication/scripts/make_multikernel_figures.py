"""Generate the per-variant and per-dataset figures for the multi-kernel
paper from a smoke or extended benchmark CSV.

Usage:

```bash
.venv/bin/python bench/AOM_v0/Multi-kernel/publication/scripts/make_multikernel_figures.py \
  bench/AOM_v0/Multi-kernel/benchmark_runs/smoke3_branches/results.csv \
  --out-dir bench/AOM_v0/Multi-kernel/publication/figures
```

Outputs:

- ``fig_per_variant_summary.{pdf,png}`` — bar chart of median rel-RMSEP
  vs PLS per variant.
- ``fig_per_dataset_heatmap.{pdf,png}`` — heatmap of variant × dataset
  rel-RMSEP vs PLS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Lazy imports of matplotlib so this script can be sourced without it.


def _load(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    return df[df.status == "ok"].copy()


def fig_per_variant_summary(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    grp = df.groupby("variant")["rel_rmsep_vs_pls"].median().sort_values()
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(grp))))
    colors = ["tab:green" if v < 1.0 else "tab:red" for v in grp.values]
    ax.barh(grp.index, grp.values, color=colors)
    ax.axvline(1.0, color="black", lw=1, ls="--", label="PLS reference")
    ax.set_xlabel("median relative RMSEP vs PLS (lower = better)")
    n_datasets = df["dataset"].nunique()
    ax.set_title(f"Per-variant median performance ({n_datasets} datasets)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_per_variant_summary.{ext}", dpi=150)
    plt.close(fig)
    print(f"[figs] wrote fig_per_variant_summary.{{pdf,png}}")


def fig_per_dataset_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    pivot = df.pivot_table(
        index="variant", columns="dataset", values="rel_rmsep_vs_pls", aggfunc="median",
    )
    if pivot.empty:
        print("[figs] no data for heatmap; skipping")
        return
    # Order rows by median across datasets.
    pivot = pivot.loc[pivot.median(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(pivot.columns)), max(3, 0.4 * len(pivot))))
    vmin = max(pivot.min().min(), 0.4)
    vmax = min(pivot.max().max(), 3.0)
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color=("white" if v > 1.5 or v < 0.7 else "black"))
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("rel-RMSEP vs PLS (1.0 = PLS reference)")
    ax.set_title("Variant × dataset — relative RMSEP vs PLS")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_per_dataset_heatmap.{ext}", dpi=150)
    plt.close(fig)
    print(f"[figs] wrote fig_per_dataset_heatmap.{{pdf,png}}")


def fig_kernel_alignment(df: pd.DataFrame, out_dir: Path) -> None:
    """Per-dataset boxplot of max kernel alignment across variants."""
    import matplotlib.pyplot as plt

    if df["kernel_alignment_max"].dropna().empty:
        print("[figs] no kernel alignment data; skipping")
        return
    fig, ax = plt.subplots(figsize=(6, 3))
    sub = df.dropna(subset=["kernel_alignment_max"])
    if sub.empty:
        plt.close(fig)
        return
    grp = sub.groupby("dataset")["kernel_alignment_max"].apply(list)
    ax.boxplot(grp.values, tick_labels=list(grp.index))
    ax.set_ylabel("max off-diagonal kernel alignment")
    ax.set_title("Kernel alignment per dataset (across variants)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_kernel_alignment.{ext}", dpi=150)
    plt.close(fig)
    print(f"[figs] wrote fig_kernel_alignment.{{pdf,png}}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("bench/AOM_v0/Multi-kernel/publication/figures"))
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = _load(args.results_csv)
    if df.empty:
        print("[figs] empty results csv; nothing to do")
        return 0
    fig_per_variant_summary(df, args.out_dir)
    fig_per_dataset_heatmap(df, args.out_dir)
    fig_kernel_alignment(df, args.out_dir)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
