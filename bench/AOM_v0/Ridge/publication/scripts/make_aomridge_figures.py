"""Generate publication figures for the AOM-Ridge paper.

Figures produced (all written under ``../figures`` as both ``.pdf`` and
``.png``):

- ``fig_aomridge_framework``           : schematic of the dual-Ridge AOM
                                         pipeline
                                         ``X -> (branch?) -> X' -> [s_b X' A_b^T] -> dual ridge -> y_hat``
                                         with the five selection modes
                                         shown as branches (superblock,
                                         global, active_superblock,
                                         branch_global, mkl).
- ``fig_alpha_grid``                   : illustration of the log-spaced
                                         alpha grid and the boundary
                                         expansion mechanism on a
                                         synthetic loss curve.
- ``fig_per_dataset_delta_vs_paper_ridge``
                                       : per-dataset delta of the best
                                         AOM-Ridge variant versus the
                                         paper Ridge HPO baseline
                                         (sorted, coloured by
                                         dataset_group).
- ``fig_critical_difference``          : Demsar / Iman-Davenport CD
                                         diagram comparing the best
                                         AOM-Ridge variant to the six
                                         TabPFN paper baselines (Ridge,
                                         PLS, CNN, Catboost, TabPFN-Raw,
                                         TabPFN-opt).
- ``fig_heatmap_methods_x_datasets``   : heatmap (datasets x methods) of
                                         relative RMSEP versus paper
                                         Ridge, including AOM-Ridge as a
                                         column.

The script is designed to be runnable today on the smoke / curated
results: when a method is missing for a dataset we keep the cell empty
rather than fabricating values, and when a join leaves no overlapping
rows we render a placeholder figure rather than failing.

Usage::

    python bench/AOM_v0/Ridge/publication/scripts/make_aomridge_figures.py

Optional flags::

    --results        path to the AOM-Ridge results CSV (default:
                     curated_v2 if present, else curated).
    --master         path to the TabPFN master pivot CSV.
    --out            output directory (default: ../figures).

This script never modifies the production library or the AOM-Ridge
package; it is a read-only consumer of the curated benchmark CSV.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PUB_ROOT = HERE.parent
RIDGE_ROOT = PUB_ROOT.parent  # bench/AOM_v0/Ridge
AOM_ROOT = RIDGE_ROOT.parent   # bench/AOM_v0

DEFAULT_OUT = PUB_ROOT / "figures"
DEFAULT_RESULTS_V2 = RIDGE_ROOT / "benchmark_runs" / "curated_v2" / "results.csv"
DEFAULT_RESULTS_V1 = RIDGE_ROOT / "benchmark_runs" / "curated" / "results.csv"
DEFAULT_MASTER = AOM_ROOT / "publication" / "tables" / "master_pivot.csv"

# Datasets to drop from analysis (degenerate references).
EXCLUDED_DATASETS = {"QUARTZ"}

# Color-blind friendly palette (Okabe-Ito) used for the dataset groups.
OKABE_ITO = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

# Per-method palette for the comparison figures.
METHOD_COLOURS = {
    "AOM-Ridge": "#1f77b4",
    "Ridge": "#666666",
    "PLS": "#999999",
    "CNN": "#bcbd22",
    "Catboost": "#8c564b",
    "TabPFN-Raw": "#ff7f0e",
    "TabPFN-opt": "#d62728",
}

PAPER_BASELINES = ["Ridge", "PLS", "CNN", "Catboost", "TabPFN-Raw", "TabPFN-opt"]


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# IO + cohort helpers
# ---------------------------------------------------------------------------

def _resolve_results_path(arg: Path | None) -> Path:
    if arg is not None:
        return Path(arg)
    if DEFAULT_RESULTS_V2.exists():
        return DEFAULT_RESULTS_V2
    return DEFAULT_RESULTS_V1


def _load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df["rmsep"] = pd.to_numeric(df["rmsep"], errors="coerce")
    df["fit_time_s"] = pd.to_numeric(df.get("fit_time_s"), errors="coerce")
    df = df.dropna(subset=["rmsep"])
    df = df[~df["dataset_group"].isin(EXCLUDED_DATASETS)].copy()
    return df


def _aomridge_best_per_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """For each dataset, pick the AOM-Ridge variant with the best RMSEP.

    "Best" is min ``relative_rmsep_vs_paper_ridge`` when available and
    finite; otherwise we fall back to min ``rmsep``. Ridge baselines
    (Ridge-raw, Ridge-raw-stdscale) are excluded so the "best variant"
    truly reflects an AOM-Ridge configuration.
    """
    is_aomridge = df["variant"].str.startswith("AOMRidge-")
    aom = df[is_aomridge].copy()
    if aom.empty:
        return aom
    rel = pd.to_numeric(aom.get("relative_rmsep_vs_paper_ridge"), errors="coerce")
    aom = aom.assign(_rel=rel.fillna(np.inf))
    # Tie-break by lower fit time, then by variant name for determinism.
    aom = aom.sort_values(
        ["dataset_group", "dataset", "_rel", "rmsep", "fit_time_s", "variant"]
    )
    best = aom.groupby(["dataset_group", "dataset"], as_index=False).first()
    best = best.drop(columns=["_rel"])
    return best


def _attach_master_baselines(best_aom: pd.DataFrame, master_path: Path) -> pd.DataFrame:
    """Long-format frame with one row per (database, dataset, method).

    Columns: database_name, dataset, method, rmsep.
    """
    rows = []
    if not best_aom.empty:
        rows.append(pd.DataFrame({
            "database_name": best_aom["dataset_group"].values,
            "dataset": best_aom["dataset"].values,
            "method": "AOM-Ridge",
            "rmsep": best_aom["rmsep"].values,
        }))
    if master_path.exists():
        master = pd.read_csv(master_path)
        master = master[~master["database_name"].isin(EXCLUDED_DATASETS)].copy()
        for col in PAPER_BASELINES:
            if col not in master.columns:
                continue
            sub = master[["database_name", "dataset", col]].copy()
            sub["method"] = col
            sub = sub.rename(columns={col: "rmsep"})
            sub["rmsep"] = pd.to_numeric(sub["rmsep"], errors="coerce")
            sub = sub.dropna(subset=["rmsep"])
            rows.append(sub[["database_name", "dataset", "method", "rmsep"]])
    if not rows:
        return pd.DataFrame(columns=["database_name", "dataset", "method", "rmsep"])
    return pd.concat(rows, ignore_index=True)


def _save(fig: plt.Figure, out_path: Path) -> list[Path]:
    """Save figure as both PDF (vector) and PNG (300 DPI). Returns paths."""
    out_path = Path(out_path)
    out_pdf = out_path.with_suffix(".pdf")
    out_png = out_path.with_suffix(".png")
    fig.savefig(out_pdf, format="pdf")
    fig.savefig(out_png, format="png", dpi=300)
    plt.close(fig)
    return [out_pdf, out_png]


# ---------------------------------------------------------------------------
# Figure 1 - AOM-Ridge framework schematic
# ---------------------------------------------------------------------------

def _draw_box(ax, xy, w, h, text, fc="white", ec="black", fontsize=9):
    rect = plt.Rectangle(xy, w, h, fill=True, fc=fc, ec=ec, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
            ha="center", va="center", fontsize=fontsize, wrap=True)


def _draw_arrow(ax, x0, y0, x1, y1, color="black"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "->", "linewidth": 1.3, "color": color})


def fig_framework(out: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Stage 1 - input + branch
    _draw_box(ax, (0.2, 4.8), 1.8, 1.0, "$X$\n(spectra)", fc="#e8f3ff")
    _draw_box(ax, (2.4, 4.8), 1.8, 1.0,
              "Branch\n(none / SNV / MSC)",
              fc="#fff5e6")
    _draw_arrow(ax, 2.0, 5.3, 2.4, 5.3)

    # Stage 2 - operator bank applied to X'
    _draw_box(ax, (4.7, 5.1), 2.6, 0.7,
              "Operator bank\n$\\{A_b\\}_{b=1}^{B}$",
              fc="#e8f3ff", fontsize=9)
    _draw_box(ax, (4.7, 4.2), 2.6, 0.7,
              "Identity, SG (smooth/d1/d2),\nFD, Detrend, NW",
              fc="#f0f0f0", fontsize=8)
    _draw_arrow(ax, 4.2, 5.3, 4.65, 5.45)

    # Stage 3 - block-scaled superblock features
    _draw_box(ax, (7.8, 4.7), 2.4, 1.0,
              "Block-scaled views\n$[s_b X' A_b^\\top]_{b=1}^{B}$",
              fc="#e6f5e6", fontsize=8)
    _draw_arrow(ax, 7.35, 5.3, 7.85, 5.2)

    # Stage 4 - dual ridge
    _draw_box(ax, (10.6, 4.7), 2.6, 1.0,
              "Dual Ridge\n$K = \\sum_b s_b^2 X' A_b^\\top A_b X'^\\top$\n$C = (K + \\alpha I)^{-1} Y$",
              fc="#f5e6f5", fontsize=8)
    _draw_arrow(ax, 10.2, 5.2, 10.65, 5.2)

    # Stage 5 - prediction box and connecting arrow from dual-ridge box.
    _draw_box(ax, (13.3, 4.8), 0.6, 1.0, "$\\hat y$",
              fc="#fff5e6", fontsize=11)
    _draw_arrow(ax, 13.2, 5.2, 13.3, 5.2)

    # The five selection-mode branches (below the main horizontal flow)
    selection_modes = [
        ("superblock", "All operators,\nfold-CV $\\alpha$", "#74c0e1"),
        ("global", "Hard $(b, \\alpha)$,\nfold-CV", "#1f77b4"),
        ("active_superblock", "Pruned subset\nby relevance", "#2ca02c"),
        ("branch_global", "Branch x op x $\\alpha$\nselection", "#9467bd"),
        ("mkl", "Supervised block\nweights $w_b$", "#ff7f0e"),
    ]
    n = len(selection_modes)
    width = 2.3
    spacing = 0.2
    total = n * width + (n - 1) * spacing
    x0 = (14 - total) / 2
    y_box = 1.7
    y_top = 3.6
    for i, (name, descr, colour) in enumerate(selection_modes):
        x = x0 + i * (width + spacing)
        _draw_box(ax, (x, y_box), width, 1.0, f"{name}\n{descr}",
                  fc=colour, ec="black", fontsize=8)
        # Connector from dual-ridge band to each branch
        _draw_arrow(ax, x + width / 2, y_top, x + width / 2, y_box + 1.0,
                    color="grey")
    ax.text(7.0, 3.85, "Selection modes",
            ha="center", va="center", fontsize=9, fontstyle="italic", color="grey")

    ax.text(7, 6.7, "AOM-Ridge: dual Ridge over a bank of strict-linear operator views",
            ha="center", va="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Figure 2 - alpha grid + boundary-expansion concept
# ---------------------------------------------------------------------------

def fig_alpha_grid(out: Path) -> list[Path]:
    rng = np.random.default_rng(0)
    # Synthetic bowl-shaped CV loss curve with a minimum near the right end
    # of the initial grid, illustrating the boundary-hit -> expand mechanism.
    initial_low, initial_high, n = -6.0, 6.0, 50
    grid_initial = np.logspace(initial_low, initial_high, n)
    log_alpha = np.log10(grid_initial)
    optimum = 7.0  # outside the initial high cap of 1e6
    loss = 1.0 + 0.04 * (log_alpha - optimum) ** 2
    loss += rng.normal(0, 0.005, size=loss.shape)

    expanded_high = 9.0
    grid_expanded = np.logspace(initial_low, expanded_high, n)
    log_alpha_e = np.log10(grid_expanded)
    loss_e = 1.0 + 0.04 * (log_alpha_e - optimum) ** 2
    loss_e += rng.normal(0, 0.005, size=loss_e.shape)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)

    ax = axes[0]
    ax.plot(grid_initial, loss, "-", color="#1f77b4", linewidth=1.3,
            label="initial grid (50 points, $10^{-6}$..$10^{6}$)")
    idx_min = int(np.argmin(loss))
    ax.scatter([grid_initial[idx_min]], [loss[idx_min]], color="#d62728",
               s=70, zorder=5, label="argmin (boundary hit)")
    ax.axvline(grid_initial[idx_min], color="#d62728", linestyle=":", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("CV RMSE")
    ax.set_title("Initial log-spaced grid (boundary hit)")
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(grid_expanded, loss_e, "-", color="#1f77b4", linewidth=1.3,
            label="expanded grid (3-decade shift on the high side)")
    idx_min2 = int(np.argmin(loss_e))
    ax.scatter([grid_expanded[idx_min2]], [loss_e[idx_min2]], color="#2ca02c",
               s=70, zorder=5, label="new argmin (interior)")
    ax.axvline(grid_expanded[idx_min2], color="#2ca02c", linestyle=":", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_title("After boundary expansion")
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    ax.legend(loc="upper left")

    fig.suptitle("Adaptive log-spaced $\\alpha$ grid with boundary expansion",
                 y=1.02, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Figure 3 - per-dataset delta vs paper Ridge
# ---------------------------------------------------------------------------

def fig_per_dataset_delta_vs_paper_ridge(
    df: pd.DataFrame, master_path: Path, out: Path,
) -> list[Path]:
    best = _aomridge_best_per_dataset(df)
    if best.empty or not master_path.exists():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No AOM-Ridge results available", ha="center")
        return _save(fig, out)
    master = pd.read_csv(master_path)
    master = master[~master["database_name"].isin(EXCLUDED_DATASETS)]
    merged = best.merge(
        master[["database_name", "dataset", "Ridge"]],
        left_on=["dataset_group", "dataset"],
        right_on=["database_name", "dataset"],
        how="inner",
    )
    if merged.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No overlap with master_pivot", ha="center")
        return _save(fig, out)
    merged["Ridge"] = pd.to_numeric(merged["Ridge"], errors="coerce")
    merged = merged.dropna(subset=["Ridge"])
    merged["delta_pct"] = (
        100.0 * (merged["rmsep"] - merged["Ridge"]) / merged["Ridge"]
    )
    merged = merged.sort_values("delta_pct")

    groups = sorted(merged["dataset_group"].unique())
    palette = {g: OKABE_ITO[i % len(OKABE_ITO)] for i, g in enumerate(groups)}
    colours = [palette[g] for g in merged["dataset_group"]]

    n = len(merged)
    fig, ax = plt.subplots(figsize=(max(8, 0.28 * n + 4), 5.0))
    x = np.arange(n)
    ax.bar(x, merged["delta_pct"].values, color=colours,
           edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(merged["dataset"].values, rotation=75, ha="right",
                       fontsize=7)
    ax.set_ylabel(r"$\Delta$ RMSEP vs paper Ridge HPO (%)")
    pos = (merged["delta_pct"] > 0).sum()
    neg = (merged["delta_pct"] < 0).sum()
    ax.set_title(
        "AOM-Ridge (best variant) per-dataset delta vs paper Ridge HPO  "
        f"[{neg} wins / {pos} losses / {n} datasets]"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.3)

    # Build a compact group legend.
    handles = [plt.Line2D([0], [0], marker="s", color="w",
                          markerfacecolor=palette[g], markersize=8,
                          label=g) for g in groups]
    ax.legend(handles=handles, loc="upper left", fontsize=7,
              ncol=2, frameon=False, title="Dataset group")
    fig.tight_layout()
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Figure 4 - critical-difference diagram
# ---------------------------------------------------------------------------

def _q_alpha_lookup(k: int) -> float:
    """Studentized range divisor for Demsar Nemenyi-like CD bars (alpha=0.05)."""
    table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
        8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
        14: 3.354, 15: 3.391,
    }
    return table.get(k, 3.4)


def _wilcoxon_post_hoc(wide: pd.DataFrame, alpha: float = 0.05) -> list[tuple[str, str]]:
    """Return sorted list of (m1, m2) pairs whose Wilcoxon test is NOT significant.

    Two methods that are not significantly different at level alpha are
    rendered as connected on the CD diagram (cliques).
    """
    from scipy.stats import wilcoxon
    methods = list(wide.columns)
    pairs = []
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1:]:
            x = wide[m1].values
            y = wide[m2].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 5:
                continue
            try:
                _, p = wilcoxon(x[mask], y[mask], zero_method="wilcox")
            except Exception:
                continue
            if p >= alpha:
                pairs.append((m1, m2))
    return pairs


def fig_critical_difference(df: pd.DataFrame, master_path: Path, out: Path) -> list[Path]:
    best_aom = _aomridge_best_per_dataset(df)
    long = _attach_master_baselines(best_aom, master_path)
    if long.empty:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No data for CD diagram", ha="center")
        return _save(fig, out)

    wide = long.pivot_table(
        index=["database_name", "dataset"], columns="method", values="rmsep",
        aggfunc="mean",
    )
    # Keep AOM-Ridge plus paper baselines, in a stable column order.
    cols = ["AOM-Ridge"] + [c for c in PAPER_BASELINES if c in wide.columns]
    wide = wide[cols]
    # CD requires complete cases on the columns we compare. Drop datasets
    # missing one or more methods.
    wide_complete = wide.dropna()
    n_data = len(wide_complete)
    k = len(wide_complete.columns)
    if n_data < 2 or k < 2:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            f"Not enough complete data: {n_data} datasets x {k} methods",
            ha="center",
        )
        return _save(fig, out)

    ranks = wide_complete.rank(axis=1, method="average")
    mean_ranks = ranks.mean(axis=0).sort_values()
    q = _q_alpha_lookup(k)
    cd = q * np.sqrt(k * (k + 1) / (6 * n_data))

    # Wilcoxon cliques on the same complete-case wide table.
    not_sig_pairs = _wilcoxon_post_hoc(wide_complete, alpha=0.05)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(mean_ranks.min() - 0.4, mean_ranks.max() + 0.4)
    ax.set_ylim(0, 1)
    ax.invert_xaxis()
    ax.set_yticks([])
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    # Axis tick line
    ax.plot([mean_ranks.min(), mean_ranks.max()], [0.55, 0.55],
            color="black", linewidth=1.2)
    for r, m in mean_ranks.items():
        ax.plot([m, m], [0.55, 0.7], color="black", linewidth=1.0)
        colour = METHOD_COLOURS.get(r, "black")
        ax.text(m, 0.74, f"{r}\n({m:.2f})", ha="center", va="bottom",
                fontsize=8, color=colour)
    # CD bar
    cd_x0 = mean_ranks.min()
    ax.plot([cd_x0, cd_x0 + cd], [0.40, 0.40], color="red", linewidth=2)
    ax.text(cd_x0 + cd / 2, 0.34, f"CD = {cd:.2f}", ha="center",
            fontsize=9, color="red")

    # Wilcoxon cliques: render as black horizontal bars under the axis,
    # one per non-significant pair. We stack pairs in vertical lanes to
    # avoid overlap.
    used_lanes: list[tuple[float, float]] = []
    lane_y = 0.20
    lane_dy = 0.04
    for m1, m2 in not_sig_pairs:
        if m1 not in mean_ranks.index or m2 not in mean_ranks.index:
            continue
        r1, r2 = mean_ranks[m1], mean_ranks[m2]
        x0, x1 = (r1, r2) if r1 < r2 else (r2, r1)
        # Find the first lane that does not overlap with x0..x1
        lane = 0
        while True:
            if lane >= len(used_lanes):
                used_lanes.append((x0, x1))
                break
            ux0, ux1 = used_lanes[lane]
            if x1 < ux0 or x0 > ux1:
                used_lanes[lane] = (min(ux0, x0), max(ux1, x1))
                break
            lane += 1
        y = lane_y - lane * lane_dy
        ax.plot([x0, x1], [y, y], color="black", linewidth=1.4)

    ax.text(
        mean_ranks.mean(), 0.06,
        f"Wilcoxon (alpha=0.05) cliques shown as black bars  |  "
        f"k={k} methods, N={n_data} datasets",
        ha="center", fontsize=8, color="grey", style="italic",
    )
    ax.set_title("Critical-difference diagram: AOM-Ridge vs TabPFN paper baselines")
    fig.tight_layout()
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Figure 5 - heatmap (datasets x methods) of relative RMSEP vs paper Ridge
# ---------------------------------------------------------------------------

def fig_heatmap_methods_x_datasets(
    df: pd.DataFrame, master_path: Path, out: Path,
) -> list[Path]:
    best_aom = _aomridge_best_per_dataset(df)
    long = _attach_master_baselines(best_aom, master_path)
    if long.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No data for heatmap", ha="center")
        return _save(fig, out)
    wide = long.pivot_table(
        index="dataset", columns="method", values="rmsep", aggfunc="mean",
    )
    if "Ridge" not in wide.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "Ridge baseline missing in master_pivot", ha="center")
        return _save(fig, out)
    rel = wide.div(wide["Ridge"], axis=0)
    method_order = ["AOM-Ridge"] + [c for c in PAPER_BASELINES if c in rel.columns]
    method_order = [m for m in method_order if m in rel.columns]
    rel = rel[method_order]
    # Sort datasets by the AOM-Ridge column when present, else by median.
    sort_col = "AOM-Ridge" if "AOM-Ridge" in rel.columns else rel.columns[0]
    rel = rel.sort_values(sort_col)

    fig, ax = plt.subplots(
        figsize=(max(7, 0.6 * len(rel.columns) + 3),
                 max(4, 0.2 * len(rel.index) + 1)),
    )
    finite = rel.values[np.isfinite(rel.values)]
    if finite.size == 0:
        vmin, vmax = 0.5, 2.0
    else:
        vmin = max(0.4, float(np.nanpercentile(finite, 5)))
        vmax = min(2.0, float(np.nanpercentile(finite, 95)))
        vmax = max(vmax, 1.05)
        vmin = min(vmin, 0.95)
    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    im = ax.imshow(rel.values, aspect="auto", cmap="RdYlGn_r", norm=norm)
    ax.set_xticks(range(len(rel.columns)))
    ax.set_xticklabels(rel.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(rel.index)))
    ax.set_yticklabels(rel.index, fontsize=7)
    ax.set_title(
        "Relative RMSEP / paper Ridge HPO (lower = better, green = beats Ridge)"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("RMSEP / RMSEP(paper Ridge)")
    if len(rel.index) <= 60:
        for i in range(len(rel.index)):
            for j in range(len(rel.columns)):
                v = rel.values[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black", fontsize=6)
    fig.tight_layout()
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AOM-Ridge publication figures.")
    parser.add_argument("--results", type=Path, default=None,
                        help="Path to AOM-Ridge results CSV (default: curated_v2 if present, else curated).")
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER,
                        help="Path to TabPFN master pivot CSV.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    results_path = _resolve_results_path(args.results)
    master_path = Path(args.master)
    out: Path = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[make_aomridge_figures] results: {results_path}")
    print(f"[make_aomridge_figures] master:  {master_path}")
    print(f"[make_aomridge_figures] out:     {out}")

    if not results_path.exists():
        print(f"ERROR: results CSV not found: {results_path}", file=sys.stderr)
        return 2
    df = _load_results(results_path)
    print(f"[make_aomridge_figures] loaded {len(df)} rows over "
          f"{df['dataset'].nunique()} datasets and {df['variant'].nunique()} variants")

    artefacts: list[Path] = []
    artefacts += fig_framework(out / "fig_aomridge_framework")
    print("  - fig_aomridge_framework.{pdf,png}")
    artefacts += fig_alpha_grid(out / "fig_alpha_grid")
    print("  - fig_alpha_grid.{pdf,png}")
    artefacts += fig_per_dataset_delta_vs_paper_ridge(
        df, master_path, out / "fig_per_dataset_delta_vs_paper_ridge",
    )
    print("  - fig_per_dataset_delta_vs_paper_ridge.{pdf,png}")
    artefacts += fig_critical_difference(
        df, master_path, out / "fig_critical_difference",
    )
    print("  - fig_critical_difference.{pdf,png}")
    artefacts += fig_heatmap_methods_x_datasets(
        df, master_path, out / "fig_heatmap_methods_x_datasets",
    )
    print("  - fig_heatmap_methods_x_datasets.{pdf,png}")

    print("\nGenerated artefacts:")
    for p in artefacts:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
