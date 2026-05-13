#!/usr/bin/env python3
"""Generate publication-style figures for the AOM NIRS manuscript.

Design principles
-----------------
* Okabe-Ito colorblind-safe palette.
* Sans-serif typography (Liberation Sans / DejaVu fallback) embedded via
  ``pdf.fonttype = 42`` so journals can re-render the type.
* Rounded boxes (``FancyBboxPatch``) and FancyArrow connectors for a clean
  vector look without rendering artifacts.
* Each figure is sized for a single 7.2-inch column block and avoids the small
  decorative spectra that previously created stray glyphs at the top of
  fig_concept.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"

# --- Okabe-Ito colorblind-safe palette + neutrals -----------------------------
INK = "#1f2933"
MUTED = "#52606d"
GRID = "#dde2e8"

BLUE = "#0072B2"      # PLS / cross-covariance
GREEN = "#009E73"     # AOM / Ridge geometry
ORANGE = "#D55E00"    # CatBoost
PURPLE = "#7E57C2"    # CNN-1D
GOLD = "#E69F00"      # AOM-PLS family
RED = "#B03A48"       # warning / branch
SKY = "#56B4E9"       # accent fills

# Soft tints for panel backgrounds (RGBA hex with alpha).
TINT_LEFT = "#eef1f5"
TINT_MID = "#fbeeee"
TINT_RIGHT = "#e9f6f0"


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Liberation Sans",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "mathtext.fontset": "dejavusans",
        }
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def strip(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    edge: str = GRID,
    face: str = "white",
    lw: float = 0.9,
    fontsize: float = 8.0,
    text_color: str = INK,
    weight: str = "normal",
    pad: float = 0.012,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size={min(w, h) * 0.18:.4f}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        joinstyle="round",
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        weight=weight,
        linespacing=1.25,
        clip_on=False,
    )


def arrow(
    ax,
    xy0,
    xy1,
    *,
    color: str = MUTED,
    lw: float = 0.9,
    style: str = "-|>",
    mutation_scale: float = 9.0,
    alpha: float = 0.95,
) -> None:
    patch = FancyArrowPatch(
        xy0,
        xy1,
        arrowstyle=style,
        color=color,
        linewidth=lw,
        mutation_scale=mutation_scale,
        alpha=alpha,
        shrinkA=2,
        shrinkB=2,
        clip_on=False,
    )
    ax.add_patch(patch)


def panel_label(ax, label: str, color: str = INK) -> None:
    ax.text(
        0.0,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        weight="bold",
        color=color,
        fontsize=10,
    )


# ---------------------------------------------------------------------------
# Fig 1 — Concept / workflow
# ---------------------------------------------------------------------------


def fig_concept() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    strip(ax)

    # Column background tints + accent bars to anchor the workflow.
    columns = [
        dict(x=0.020, w=0.300, face=TINT_LEFT,  bar=MUTED,  title="Conventional NIRS calibration",       subtitle="external pipeline search"),
        dict(x=0.350, w=0.300, face=TINT_MID,   bar=RED,    title="Analytical bottleneck",               subtitle="cost, variance, opacity"),
        dict(x=0.680, w=0.300, face=TINT_RIGHT, bar=GREEN,  title="Operator-adaptive calibration",       subtitle="model-internal & auditable"),
    ]
    for col in columns:
        bg = FancyBboxPatch(
            (col["x"], 0.060),
            col["w"],
            0.870,
            boxstyle="round,pad=0.004,rounding_size=0.022",
            facecolor=col["face"],
            edgecolor="none",
            zorder=-5,
            clip_on=False,
        )
        ax.add_patch(bg)
        # Accent ribbon under the title.
        ax.add_patch(
            Rectangle(
                (col["x"] + 0.016, 0.866),
                col["w"] - 0.032,
                0.012,
                facecolor=col["bar"],
                edgecolor="none",
                zorder=-3,
            )
        )
        ax.text(
            col["x"] + col["w"] / 2,
            0.905,
            col["title"],
            ha="center",
            va="bottom",
            weight="bold",
            color=INK,
            fontsize=9.4,
        )
        ax.text(
            col["x"] + col["w"] / 2,
            0.884,
            col["subtitle"],
            ha="center",
            va="bottom",
            color=MUTED,
            fontsize=7.6,
            style="italic",
        )

    rows_y = [0.755, 0.625, 0.495, 0.365, 0.235]
    left_text = [
        "many preprocessing\npipelines",
        "CV selects a\ncomplete pipeline",
        "final model depends\non pipeline choice",
        "external choices are\nhard to audit",
        "refitting repeats\nthe search",
    ]
    middle_text = [
        "high calibration\ncost",
        "winner's curse\n& instability",
        "deployment\ncomplexity",
        "poor traceability",
        "limited routine\nusability",
    ]
    right_text = [
        "compact linear\noperator bank",
        "selection inside the\ncalibration model",
        "original-wavelength\ncoefficients",
        "selected operators\nlogged",
        "fast refit of one\nlinear model",
    ]

    box_h = 0.088
    for y, lt, mt, rt in zip(rows_y, left_text, middle_text, right_text):
        rounded_box(
            ax, 0.045, y - box_h / 2, 0.250, box_h, lt,
            face="white", edge="#c5ccd4", fontsize=7.7, pad=0.008,
        )
        rounded_box(
            ax, 0.375, y - box_h / 2, 0.250, box_h, mt,
            face="white", edge="#d49aa3", fontsize=7.7, pad=0.008,
        )
        rounded_box(
            ax, 0.705, y - box_h / 2, 0.250, box_h, rt,
            face="white", edge="#7fc4ad", fontsize=7.7, pad=0.008,
        )
        arrow(ax, (0.301, y), (0.369, y), color=MUTED, lw=1.0, mutation_scale=8.5)
        arrow(ax, (0.631, y), (0.699, y), color=GREEN, lw=1.0, mutation_scale=8.5)

    # Bottom caption strip.
    ax.text(
        0.500,
        0.110,
        "AOM reframes preprocessing selection as a model-internal, "
        "auditable part of analytical calibration.",
        ha="center",
        va="center",
        color=INK,
        fontsize=8.3,
        weight="medium",
    )

    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.005)
    fig.savefig(FIG_DIR / "fig_concept.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2 — Mathematical structure (three panels)
# ---------------------------------------------------------------------------


def math_panel(
    ax,
    title: str,
    accent: str,
    rows,
    tint: str,
) -> None:
    strip(ax)

    # Soft panel background.
    ax.add_patch(
        FancyBboxPatch(
            (0.020, 0.020),
            0.960,
            0.940,
            boxstyle="round,pad=0.002,rounding_size=0.045",
            facecolor=tint,
            edgecolor="none",
            zorder=-4,
            clip_on=False,
        )
    )
    # Accent stripe.
    ax.add_patch(
        Rectangle(
            (0.020, 0.910),
            0.960,
            0.030,
            facecolor=accent,
            edgecolor="none",
        )
    )
    ax.text(
        0.060,
        0.870,
        title,
        ha="left",
        va="top",
        weight="bold",
        color=INK,
        fontsize=10,
    )

    y_positions = [0.735, 0.490, 0.245]
    box_h = 0.155
    for (main, sub, color), y in zip(rows, y_positions):
        rounded_box(
            ax,
            0.080,
            y - box_h / 2,
            0.840,
            box_h,
            main + ("\n" + sub if sub else ""),
            edge=color,
            face="white",
            fontsize=9.0,
            pad=0.010,
        )
    # Top-down flow arrows between successive rows (must clear the box edges).
    arrow(ax, (0.500, 0.735 - box_h / 2 - 0.005),
              (0.500, 0.490 + box_h / 2 + 0.005),
          color=accent, lw=1.3, mutation_scale=12)
    arrow(ax, (0.500, 0.490 - box_h / 2 - 0.005),
              (0.500, 0.245 + box_h / 2 + 0.005),
          color=accent, lw=1.3, mutation_scale=12)


def fig_math() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), constrained_layout=True)

    math_panel(
        axes[0],
        "A   Shared operator view",
        GREEN,
        [
            (r"$\mathbf{X}_b = \mathbf{X}\,\mathbf{A}_b^\mathrm{T}$", "linear spectral operator", GREEN),
            (r"$\mathbf{A}_b \in \mathbb{R}^{p \times p}$", "identity always included", GREEN),
            ("fold-local branches", "SNV / MSC / ASLS", RED),
        ],
        TINT_RIGHT,
    )

    math_panel(
        axes[1],
        "B   AOM-PLS",
        BLUE,
        [
            (r"$\mathbf{S} = \mathbf{X}^\mathrm{T}\mathbf{Y}$", "cross-covariance", BLUE),
            (r"$\mathbf{S}_b = \mathbf{A}_b\,\mathbf{S}$",
             r"because $(\mathbf{X}\mathbf{A}_b^\mathrm{T})^\mathrm{T}\mathbf{Y} = \mathbf{A}_b\mathbf{X}^\mathrm{T}\mathbf{Y}$", BLUE),
            (r"$\mathbf{B} = \mathbf{Z}(\mathbf{P}^\mathrm{T}\mathbf{Z})^{+}\mathbf{Q}^\mathrm{T}$",
             "original-space coefficients", BLUE),
        ],
        "#e8f1fa",
    )

    math_panel(
        axes[2],
        "C   AOM-Ridge",
        GREEN,
        [
            (r"$\mathbf{K}_b = \mathbf{X}_c\,\mathbf{A}_b^\mathrm{T}\mathbf{A}_b\,\mathbf{X}_c^\mathrm{T}$",
             "operator geometry", GREEN),
            (r"$\mathbf{C} = (\mathbf{K}_b + \alpha\mathbf{I})^{-1}\mathbf{Y}_c$",
             "dual ridge solve", GREEN),
            (r"$\boldsymbol{\beta}_b = \mathbf{A}_b^\mathrm{T}\mathbf{A}_b\,\mathbf{X}_c^\mathrm{T}\mathbf{C}$",
             "original-space coefficients", GREEN),
        ],
        TINT_RIGHT,
    )

    fig.savefig(FIG_DIR / "fig_math.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3 — Headline results (deployable vs oracle envelope)
# ---------------------------------------------------------------------------


def fig_results() -> None:
    rows = [
        ("AOM-PLS compact  vs PLS",        4.00, 42, 57, "deployable", GOLD),
        ("AOM-Ridge Blender  vs Ridge",    2.22, 35, 52, "deployable", GREEN),
        ("AOM-Ridge oracle  vs Ridge",     4.73, 45, 52, "oracle",     GREEN),
        ("AOM-Ridge oracle  vs PLS",       7.73, 47, 52, "oracle",     BLUE),
        ("AOM-Ridge oracle  vs CatBoost", 13.27, 38, 52, "oracle",     ORANGE),
    ]
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    wins   = [r[2] for r in rows]
    totals = [r[3] for r in rows]
    kinds  = [r[4] for r in rows]
    colors = [r[5] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)

    # Separation band between deployable and oracle blocks.
    deploy_n = sum(1 for k in kinds if k == "deployable")
    ax.axhspan(-0.5, deploy_n - 0.5, facecolor="#f3f7f3", edgecolor="none", zorder=0)
    ax.axhspan(deploy_n - 0.5, len(rows) - 0.5, facecolor="#f5f3fa", edgecolor="none", zorder=0)

    ax.text(
        16.5, (deploy_n - 1) / 2,
        "DEPLOYABLE",
        ha="right", va="center", color="#789686", fontsize=7.4,
        weight="bold", alpha=0.9,
    )
    ax.text(
        16.5, deploy_n + (len(rows) - deploy_n - 1) / 2,
        "ORACLE  ENVELOPE",
        ha="right", va="center", color="#8c81b4", fontsize=7.4,
        weight="bold", alpha=0.9,
    )

    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    for yi, value, kind, color in zip(y, values, kinds, colors):
        ax.hlines(yi, 0, value, color=color, lw=2.2, alpha=0.85, zorder=2)
        if kind == "oracle":
            ax.scatter(value, yi, s=70, facecolor="white",
                       edgecolor=color, linewidth=1.7, zorder=3)
        else:
            ax.scatter(value, yi, s=70, facecolor=color,
                       edgecolor=color, linewidth=1.0, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.4, 17.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel("Median RMSEP reduction relative to baseline (%)")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, width=0.7)

    for yi, value, win, total in zip(y, values, wins, totals):
        ax.text(
            value + 0.45, yi - 0.05,
            f"{value:.2f}%",
            va="center", ha="left",
            fontsize=8.4, weight="bold", color=INK,
        )
        ax.text(
            value + 0.45, yi + 0.27,
            f"wins {win}/{total}",
            va="center", ha="left",
            fontsize=7.4, color=MUTED,
        )

    # Custom legend.
    ax.scatter([], [], s=70, facecolor=INK, edgecolor=INK, label="Deployable result")
    ax.scatter([], [], s=70, facecolor="white", edgecolor=INK,
               linewidth=1.4, label="Oracle envelope")
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.995, 1.00),
        frameon=False,
        fontsize=7.8,
        handletextpad=0.6,
    )

    fig.savefig(FIG_DIR / "fig_results.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4 — Search-budget contrast (log-scale bars)
# ---------------------------------------------------------------------------


def fig_budget() -> None:
    rows = [
        ("AOM-PLS compact",    135, GOLD,    "structured operator bank"),
        ("CatBoost",            90, ORANGE,  "fixed HPs, preprocessing search"),
        ("CNN-1D",            2700, PURPLE,  "preprocessing search + arch"),
        ("PLS-HPO",           5400, BLUE,    "hierarchical preprocessing-HPO"),
        ("Ridge-HPO",         5400, GREEN,   "hierarchical preprocessing-HPO"),
    ]
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    notes  = [r[3] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.2, 3.1), constrained_layout=True)

    # Shaded "compact" region as a visual reference (≤ 200 evaluations).
    ax.axvspan(50, 200, facecolor="#fff6e0", edgecolor="none", zorder=0)
    ax.text(
        140, -0.85,
        "compact zone",
        ha="center", va="center",
        fontsize=7.4, color="#a37200", style="italic",
    )

    # Lollipop bars with rounded markers.
    for yi, value, color in zip(y, values, colors):
        ax.hlines(yi, 50, value, color=color, lw=2.6, alpha=0.85, zorder=2)
        ax.scatter(value, yi, s=84, facecolor=color, edgecolor="white",
                   linewidth=1.4, zorder=3)

    ax.set_xscale("log")
    ax.set_xlim(50, 12000)
    ax.set_ylim(len(rows) - 0.5, -1.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Candidate fold-level evaluations per dataset (log scale)")
    ax.grid(axis="x", color=GRID, lw=0.7, which="both")
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, width=0.7)

    for yi, value, note in zip(y, values, notes):
        ax.text(
            value * 1.16, yi - 0.06,
            f"{value:,}",
            va="center", ha="left",
            fontsize=8.4, weight="bold", color=INK,
        )
        ax.text(
            value * 1.16, yi + 0.30,
            note,
            va="center", ha="left",
            fontsize=7.3, color=MUTED, style="italic",
        )

    fig.savefig(FIG_DIR / "fig_budget.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    setup()
    fig_concept()
    fig_math()
    fig_results()
    fig_budget()
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
