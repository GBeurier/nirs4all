"""Publication-quality figures inspired by the TabPFN paper.

Produces:

1. `fig_heatmap_relative_rmsep.pdf` — heatmap of relative RMSEP
   (variant / PLS-standard) per dataset, with a diverging colormap.
2. `fig_critical_difference.pdf` — Friedman + Nemenyi critical-difference
   diagram across all variants.
3. `fig_cost_vs_precision.pdf` — scatter of mean fit time vs median
   relative RMSEP, with each variant labelled.
4. `fig_per_dataset_delta_vs_pls.pdf` — vertical strip plot of the
   per-dataset delta-RMSEP versus standard PLS for every variant.
5. `fig_cohort_timing.pdf` — bar/dot plot of training time by variant
   averaged across the cohort.

Inputs:

- `bench/AOM_v0/benchmark_runs/full/results.csv` (extended + parity benchmark).
- Optional `bench/tabpfn_paper/master_results.csv` for TabPFN-Raw / TabPFN-opt
  reference series.

Outputs into `bench/AOM_v0/publication/figures/`.

The styling aims for journal quality: serif fonts, tight margins,
publication colour palette, and clear annotations.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Curated palette: AOM_v0 core variants in cool blues, multi-view in greens,
# production baselines in warm reds, PLS in neutral grey, non-linear preprocessing
# in golden orange, alternative criteria in muted purple.
VARIANT_COLOURS = {
    "PLS-standard-numpy": "#666666",
    "AOM-default-simpls-covariance-numpy": "#1f77b4",
    "AOM-default-nipals-adjoint-numpy": "#2c7fb8",
    "AOM-compact-simpls-covariance-numpy": "#74c0e1",
    "POP-simpls-covariance-numpy": "#9467bd",
    "POP-nipals-adjoint-numpy": "#c39bd3",
    "ActiveSuperblock-simpls-numpy": "#2ca02c",
    "Superblock-raw-simpls-numpy": "#7fbf7f",
    "AOM-explorer-simpls-numpy": "#17becf",
    "nirs4all-AOM-PLS-default": "#d62728",
    "nirs4all-POP-PLS-default": "#ff9896",
    # Alternative criteria (no holdout)
    "AOM-compact-press-numpy": "#8c6bb1",
    "AOM-compact-cv3-numpy": "#bcbddc",
    # Non-linear preprocessing -> AOM (TabPFN-paper-style baselines)
    "SNV-AOM-default-numpy": "#fdae6b",
    "MSC-AOM-default-numpy": "#fd8d3c",
    "OSC-AOM-default-numpy": "#e6550d",
    "SNV-AOM-compact-numpy": "#a6cee3",
    "MSC-AOM-compact-numpy": "#fdc086",
    "OSC-AOM-compact-numpy": "#bf5b17",
    "SNV-OSC-AOM-default-numpy": "#7fcdbb",
    "MSC-OSC-AOM-default-numpy": "#41b6c4",
    # P1-P5 + stabilization variants
    "ASLS-AOM-compact-cv5-numpy": "#006837",  # darkest green = champion
    "ASLS-AOM-response-dedup-cv3-numpy": "#1a9850",
    "ASLS-AOM-family-pruned-cv3-numpy": "#66bd63",
    "ASLS-AOM-compact-repcv3-numpy": "#a6d96a",
    "ASLS-AOM-compact-cv3-numpy": "#d9ef8b",
    "AOM-compact-repcv3-numpy": "#fee08b",
    "AOM-compact-cv5-numpy": "#fdae61",
    "AOM-response-dedup-numpy": "#b2abd2",
    "AOM-family-pruned-numpy": "#9d96c4",
    "ActiveSuperblock-deep3-numpy": "#5e3c99",
}

VARIANT_DISPLAY = {
    "PLS-standard-numpy": "PLS",
    "AOM-default-simpls-covariance-numpy": "AOM-SIMPLS-cov (default)",
    "AOM-default-nipals-adjoint-numpy": "AOM-NIPALS-adj (default)",
    "AOM-compact-simpls-covariance-numpy": "AOM-SIMPLS (compact)",
    "POP-simpls-covariance-numpy": "POP-SIMPLS",
    "POP-nipals-adjoint-numpy": "POP-NIPALS",
    "ActiveSuperblock-simpls-numpy": "Active Superblock",
    "Superblock-raw-simpls-numpy": "Superblock raw",
    "AOM-explorer-simpls-numpy": "AOM explorer",
    "nirs4all-AOM-PLS-default": "AOM-PLS (production)",
    "nirs4all-POP-PLS-default": "POP-PLS (production)",
    "AOM-compact-press-numpy": "AOM (compact, PRESS)",
    "AOM-compact-cv3-numpy": "AOM (compact, CV-3)",
    "SNV-AOM-default-numpy": "SNV+AOM (default)",
    "MSC-AOM-default-numpy": "MSC+AOM (default)",
    "OSC-AOM-default-numpy": "OSC+AOM (default)",
    "SNV-AOM-compact-numpy": "SNV+AOM (compact)",
    "MSC-AOM-compact-numpy": "MSC+AOM (compact)",
    "OSC-AOM-compact-numpy": "OSC+AOM (compact)",
    "SNV-OSC-AOM-default-numpy": "SNV+OSC+AOM",
    "MSC-OSC-AOM-default-numpy": "MSC+OSC+AOM",
    "TabPFN-Raw": "TabPFN-Raw",
    "TabPFN-opt": "TabPFN-opt",
    "master-PLS": "PLS (TabPFN paper)",
    "master-Catboost": "Catboost (TabPFN paper)",
    "master-Ridge": "Ridge (TabPFN paper)",
    # P1-P5 + stabilization variants
    "ASLS-AOM-compact-cv5-numpy": "ASLS+AOM (compact, CV-5)",
    "ASLS-AOM-compact-cv3-numpy": "ASLS+AOM (compact, CV-3)",
    "ASLS-AOM-compact-repcv3-numpy": "ASLS+AOM (compact, repCV-3)",
    "ASLS-AOM-response-dedup-cv3-numpy": "ASLS+AOM (response-dedup, CV-3)",
    "ASLS-AOM-family-pruned-cv3-numpy": "ASLS+AOM (family-pruned, CV-3)",
    "AOM-compact-repcv3-numpy": "AOM (compact, repCV-3)",
    "AOM-compact-cv5-numpy": "AOM (compact, CV-5)",
    "AOM-response-dedup-numpy": "AOM (response-dedup)",
    "AOM-family-pruned-numpy": "AOM (family-pruned)",
    "ActiveSuperblock-deep3-numpy": "Active Superblock (deep3)",
}


def _load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df["RMSEP"] = pd.to_numeric(df["RMSEP"], errors="coerce")
    df["fit_time_s"] = pd.to_numeric(df["fit_time_s"], errors="coerce")
    df = df.dropna(subset=["RMSEP"])
    return df


def _attach_master_baselines(df: pd.DataFrame, master_path: Path) -> pd.DataFrame:
    """Append TabPFN-Raw, TabPFN-opt, and master PLS rows from master_results.csv.

    The TabPFN paper benchmark stores RMSEP for several reference models keyed
    by `(database_name, dataset, model)`. We pivot it into the same long format
    used by the AOM_v0 results (one row per dataset / variant) so the heatmap
    can show TabPFN baselines alongside our variants.
    """
    if not master_path.exists():
        return df
    master = pd.read_csv(master_path)
    keep = master[master["model"].isin(["TabPFN-Raw", "TabPFN-opt", "PLS", "Catboost", "Ridge"])].copy()
    keep["RMSEP"] = pd.to_numeric(keep["RMSEP"], errors="coerce")
    keep = keep.dropna(subset=["RMSEP"])
    keep = keep[["database_name", "dataset", "model", "RMSEP"]]
    keep = keep.rename(columns={"model": "aom_variant"})
    keep["aom_variant"] = keep["aom_variant"].map({
        "TabPFN-Raw": "TabPFN-Raw",
        "TabPFN-opt": "TabPFN-opt",
        "PLS": "master-PLS",
        "Catboost": "master-Catboost",
        "Ridge": "master-Ridge",
    })
    keep["status"] = "ok"
    keep["fit_time_s"] = float("nan")
    return pd.concat([df, keep], ignore_index=True)


def heatmap_relative_rmsep(df: pd.DataFrame, out: Path, top_datasets: Optional[int] = None) -> Path:
    wide = df.pivot_table(index="dataset", columns="aom_variant", values="RMSEP", aggfunc="mean")
    if "PLS-standard-numpy" not in wide.columns:
        raise ValueError("PLS-standard-numpy must be in the results to compute relative RMSEP")
    rel = wide.div(wide["PLS-standard-numpy"], axis=0)
    rel = rel.drop(columns=["PLS-standard-numpy"])
    # Order columns by median ascending (best first)
    col_order = rel.median(axis=0).sort_values().index.tolist()
    rel = rel[col_order]
    if top_datasets is not None:
        # Keep the top_datasets rows by largest variance (most informative)
        var = rel.var(axis=1)
        rel = rel.loc[var.sort_values(ascending=False).head(top_datasets).index]
    rel = rel.sort_index()
    fig, ax = plt.subplots(figsize=(min(14, 0.6 * len(rel.columns) + 4), max(4, 0.18 * len(rel.index) + 1)))
    # Diverging colormap centred at 1.0.
    vmax = max(2.0, float(np.nanpercentile(rel.values, 95)))
    vmin = min(0.5, float(np.nanpercentile(rel.values, 5)))
    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    im = ax.imshow(rel.values, aspect="auto", cmap="RdYlGn_r", norm=norm)
    ax.set_xticks(range(len(rel.columns)))
    ax.set_xticklabels([VARIANT_DISPLAY.get(c, c) for c in rel.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(rel.index)))
    ax.set_yticklabels(rel.index, fontsize=7)
    ax.set_title("Relative RMSEP / PLS (lower = better, green = beats PLS)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
    cbar.set_label("RMSEP / RMSEP(PLS-standard)")
    # Annotate cells with values for compact tables
    if len(rel.index) <= 30:
        for i in range(len(rel.index)):
            for j in range(len(rel.columns)):
                v = rel.values[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="black", fontsize=6)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def critical_difference_diagram(df: pd.DataFrame, out: Path) -> Path:
    """Friedman/Nemenyi CD diagram (Demšar 2006)."""
    from scipy.stats import friedmanchisquare
    wide = df.pivot_table(index="dataset", columns="aom_variant", values="RMSEP", aggfunc="mean")
    wide = wide.dropna()
    n_datasets = len(wide.index)
    k = len(wide.columns)
    if n_datasets < 2 or k < 2:
        # Empty plot with explanatory message
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, f"Not enough data for CD: {n_datasets} datasets x {k} variants", ha="center")
        fig.savefig(out)
        plt.close(fig)
        return out
    # Per-row ranks (lower RMSE = lower rank). Average over datasets.
    ranks = wide.rank(axis=1)
    mean_ranks = ranks.mean(axis=0).sort_values()
    # Critical difference (Nemenyi) at alpha=0.05
    # CD = q_alpha * sqrt(k(k+1) / (6N))
    q_alpha_05 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
        8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
        14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517,
        20: 3.544, 21: 3.569, 22: 3.593, 23: 3.616, 24: 3.637, 25: 3.658,
    }.get(k, 3.658)
    CD = q_alpha_05 * np.sqrt(k * (k + 1) / (6.0 * n_datasets))
    fig, ax = plt.subplots(figsize=(10, max(2.4, 0.25 * k + 1.5)))
    ax.set_xlim(min(mean_ranks) - 0.2, max(mean_ranks) + 0.2)
    ax.set_ylim(-1, 1)
    # Plot ranks as horizontal lines from each label to the rank position
    for i, (variant, rank) in enumerate(mean_ranks.items()):
        ypos = 0.0
        ax.plot([rank, rank], [-0.05, 0.05], color="black", linewidth=1.2)
        # alternate label sides
        side = -1 if i % 2 == 0 else 1
        label_y = 0.4 * side
        label_x = rank
        ax.annotate(VARIANT_DISPLAY.get(variant, variant), xy=(rank, 0.05 if side > 0 else -0.05),
                    xytext=(label_x, label_y), ha="center", fontsize=8,
                    arrowprops=dict(arrowstyle="-", color="grey", linewidth=0.5))
    ax.axhline(0, color="black", linewidth=0.5)
    # CD bar above the axis
    cd_y = 0.15
    ax.plot([min(mean_ranks), min(mean_ranks) + CD], [cd_y, cd_y], color="red", linewidth=2)
    ax.text(min(mean_ranks) + CD / 2, cd_y + 0.05, f"CD = {CD:.3f}", ha="center", color="red", fontsize=8)
    ax.set_yticks([])
    ax.set_title(f"Critical-difference diagram (k={k}, n_datasets={n_datasets}, alpha=0.05)")
    ax.set_xlabel("Mean rank (lower is better)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def cost_vs_precision(df: pd.DataFrame, out: Path) -> Path:
    wide = df.pivot_table(index="dataset", columns="aom_variant", values="RMSEP", aggfunc="mean")
    if "PLS-standard-numpy" not in wide.columns:
        raise ValueError("need PLS-standard-numpy for relative RMSEP")
    rel = wide.div(wide["PLS-standard-numpy"], axis=0)
    median_rel = rel.median(axis=0)
    times = df.groupby("aom_variant")["fit_time_s"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in median_rel.index:
        if variant not in times.index:
            continue
        x = float(times[variant])
        y = float(median_rel[variant])
        c = VARIANT_COLOURS.get(variant, "#888888")
        ax.scatter(x, y, color=c, s=80, edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(VARIANT_DISPLAY.get(variant, variant), (x, y),
                    xytext=(6, 4), textcoords="offset points", fontsize=7)
    ax.axhline(1.0, color="grey", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Mean fit time (s)")
    ax.set_ylabel("Median RMSEP / PLS")
    ax.set_xscale("log")
    ax.set_title("Cost vs precision: training time vs median relative RMSEP")
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def per_dataset_delta_strip(df: pd.DataFrame, out: Path) -> Path:
    wide = df.pivot_table(index="dataset", columns="aom_variant", values="RMSEP", aggfunc="mean")
    if "PLS-standard-numpy" not in wide.columns:
        raise ValueError("need PLS-standard-numpy")
    delta = wide.subtract(wide["PLS-standard-numpy"], axis=0).drop(columns=["PLS-standard-numpy"])
    col_order = delta.median(axis=0).sort_values().index.tolist()
    delta = delta[col_order]
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(delta.columns) + 2), 5))
    rng = np.random.default_rng(0)
    for j, variant in enumerate(delta.columns):
        vals = delta[variant].dropna().values
        x = j + (rng.random(len(vals)) - 0.5) * 0.3
        c = VARIANT_COLOURS.get(variant, "#888888")
        ax.scatter(x, vals, color=c, s=20, alpha=0.6, edgecolor="white", linewidth=0.4)
        ax.scatter([j], [np.median(vals)], color="black", marker="_", s=200, linewidth=1.5, zorder=4)
    ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
    ax.set_xticks(range(len(delta.columns)))
    ax.set_xticklabels([VARIANT_DISPLAY.get(c, c) for c in delta.columns], rotation=45, ha="right")
    ax.set_ylabel("RMSEP - RMSEP(PLS-standard)")
    ax.set_title("Per-dataset RMSEP delta vs PLS-standard")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def cohort_timing_chart(df: pd.DataFrame, out: Path) -> Path:
    times = df.groupby("aom_variant")["fit_time_s"].agg(["mean", "median"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(times) + 1)))
    y = np.arange(len(times))
    colours = [VARIANT_COLOURS.get(v, "#888888") for v in times.index]
    ax.barh(y, times["mean"].values, color=colours, alpha=0.8, edgecolor="white")
    for i, (v, mean, median) in enumerate(zip(times.index, times["mean"], times["median"])):
        ax.text(mean, i, f" {mean:.1f}s (med {median:.1f})", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([VARIANT_DISPLAY.get(v, v) for v in times.index])
    ax.invert_yaxis()
    ax.set_xlabel("Mean fit time (s)")
    ax.set_title("Training time by variant (averaged across the cohort)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="bench/AOM_v0/benchmark_runs/full/results.csv")
    parser.add_argument("--master", default="bench/tabpfn_paper/master_results.csv")
    parser.add_argument("--out", default="bench/AOM_v0/publication/figures")
    args = parser.parse_args(argv)
    df = _load_results(Path(args.results))
    if args.master:
        df = _attach_master_baselines(df, Path(args.master))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [
        heatmap_relative_rmsep(df, out_dir / "fig_heatmap_relative_rmsep.pdf"),
        critical_difference_diagram(df, out_dir / "fig_critical_difference.pdf"),
        cost_vs_precision(df, out_dir / "fig_cost_vs_precision.pdf"),
        per_dataset_delta_strip(df, out_dir / "fig_per_dataset_delta_vs_pls.pdf"),
        cohort_timing_chart(df, out_dir / "fig_cohort_timing.pdf"),
    ]
    print("Generated:")
    for f in files:
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
