"""Generate figures comparing AOM-Ridge variants to all TabPFN-paper baselines.

Produces:
- ``fig_aomridge_vs_baselines_grid``: a 2D grid of median Δ vs each
  baseline for each AOM-Ridge variant (heatmap with annotations).
- ``fig_aomridge_vs_baselines_bars``: grouped bar chart, one bar per
  (AOM-Ridge variant, baseline) cell, win-rate annotations.
- ``fig_per_dataset_winners``: per-dataset bar chart showing which
  baseline AOM-Ridge-Blender beats / loses to.
- ``fig_radar_oracle_vs_baselines``: spider chart of AOM-Ridge oracle
  envelope vs all six baselines.

All figures use the combined results CSV
(``benchmark_runs/all54_combined/results.csv``) plus the master pivot
(``bench/AOM_v0/publication/tables/master_pivot.csv``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RIDGE_ROOT = HERE.parent.parent
PROJECT_ROOT = RIDGE_ROOT.parent.parent
DEFAULT_RESULTS = RIDGE_ROOT / "benchmark_runs" / "all54_combined" / "results.csv"
DEFAULT_MASTER = PROJECT_ROOT / "AOM_v0" / "publication" / "tables" / "master_pivot.csv"
DEFAULT_OUT = RIDGE_ROOT / "publication" / "figures"

BASELINES = ["Ridge", "PLS", "TabPFN-Raw", "TabPFN-opt", "Catboost", "CNN"]
BASELINE_LABELS = {
    "Ridge": "Ridge HPO",
    "PLS": "PLS",
    "TabPFN-Raw": "TabPFN-Raw",
    "TabPFN-opt": "TabPFN-opt",
    "Catboost": "Catboost",
    "CNN": "CNN",
}

TOP_VARIANTS = [
    "AOMRidge-Blender-headline-spxy3",
    "AOMRidge-AutoSelect-headline-spxy3",
    "AOMRidge-global-compact-none-split_aware",
    "AOMRidge-global-compact-none",
    "AOMRidge-Local-compact-cv-blended",
    "AOMRidge-global-compact-none-asls",
    "AOMRidge-global-compact-none-msc",
    "AOMRidge-global-compact-none-snv",
    "AOMRidgePLS-compact-colscale-cv-relative",
    "AOMRidge-Local-compact-knn50",
    "AOMRidge-MultiBranchMKL-compact-shrink03",
]
VARIANT_LABELS = {
    "AOMRidge-Blender-headline-spxy3": "Blender",
    "AOMRidge-AutoSelect-headline-spxy3": "AutoSelect",
    "AOMRidge-global-compact-none-split_aware": "split_aware",
    "AOMRidge-global-compact-none": "global",
    "AOMRidge-Local-compact-cv-blended": "Local-cv",
    "AOMRidge-global-compact-none-asls": "global-asls",
    "AOMRidge-global-compact-none-msc": "global-msc",
    "AOMRidge-global-compact-none-snv": "global-snv",
    "AOMRidgePLS-compact-colscale-cv-relative": "RidgePLS-cv",
    "AOMRidge-Local-compact-knn50": "Local-knn50",
    "AOMRidge-MultiBranchMKL-compact-shrink03": "MultiBrMKL",
}

EXCLUDED_DATASETS = {"QUARTZ", "LUCAS_SOC_Cropland_8731_NocitaKS"}


def _save(fig: plt.Figure, out: Path) -> list[Path]:
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out.with_suffix(f".{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200)
        paths.append(p)
    plt.close(fig)
    return paths


def _load(results_path: Path, master_path: Path) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    df = df.dropna(subset=["rmsep"])
    df = df.drop_duplicates(["dataset", "variant"], keep="last")
    df = df[~df["dataset_group"].isin(EXCLUDED_DATASETS)]
    df = df[~df["dataset"].isin(EXCLUDED_DATASETS)]
    master = pd.read_csv(master_path)
    master = master[~master["database_name"].isin(EXCLUDED_DATASETS)]
    master = master[~master["dataset"].isin(EXCLUDED_DATASETS)]
    df = df.merge(
        master,
        left_on=["dataset_group", "dataset"],
        right_on=["database_name", "dataset"],
        how="inner",
    )
    return df


def fig_grid_median_delta(df: pd.DataFrame, out: Path) -> list[Path]:
    """Heatmap of median Δ vs each baseline for each variant."""
    rows = []
    win_rows = []
    for v in TOP_VARIANTS:
        sv = df[df["variant"] == v]
        if len(sv) == 0:
            continue
        row = {"variant": VARIANT_LABELS[v]}
        win_row = {"variant": VARIANT_LABELS[v]}
        for b in BASELINES:
            s = sv.dropna(subset=[b])
            if len(s) == 0:
                row[b] = np.nan
                win_row[b] = np.nan
                continue
            d = 100.0 * (s["rmsep"] - s[b]) / s[b]
            row[b] = d.median()
            win_row[b] = 100.0 * (d < 0).sum() / len(s)
        rows.append(row)
        win_rows.append(win_row)

    median_df = pd.DataFrame(rows).set_index("variant")
    win_df = pd.DataFrame(win_rows).set_index("variant")

    # Add an Oracle row
    oracle_idx = df[df["variant"].str.startswith("AOMRidge")].groupby("dataset")["rmsep"].idxmin()
    oracle = df.loc[oracle_idx]
    oracle_row = {"variant": "Oracle"}
    oracle_win = {"variant": "Oracle"}
    for b in BASELINES:
        s = oracle.dropna(subset=[b])
        if len(s) == 0:
            oracle_row[b] = np.nan
            oracle_win[b] = np.nan
            continue
        d = 100.0 * (s["rmsep"] - s[b]) / s[b]
        oracle_row[b] = d.median()
        oracle_win[b] = 100.0 * (d < 0).sum() / len(s)
    median_df = pd.concat([pd.DataFrame([oracle_row]).set_index("variant"), median_df])
    win_df = pd.concat([pd.DataFrame([oracle_win]).set_index("variant"), win_df])

    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             gridspec_kw={"width_ratios": [1, 1]})

    # Median delta heatmap
    ax = axes[0]
    data = median_df[BASELINES].values
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
    vmax = min(vmax, 30)  # cap colormap
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(BASELINES)))
    ax.set_xticklabels([BASELINE_LABELS[b] for b in BASELINES], rotation=30, ha="right")
    ax.set_yticks(range(len(median_df.index)))
    ax.set_yticklabels(median_df.index)
    ax.set_title("Median Δ (%) vs.\\ baseline\n(green = AOM-Ridge wins)")
    for i, row in enumerate(median_df.index):
        for j, b in enumerate(BASELINES):
            v = median_df.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if abs(v) > vmax * 0.6 else "black")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    # Win rate heatmap
    ax = axes[1]
    data = win_df[BASELINES].values
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(BASELINES)))
    ax.set_xticklabels([BASELINE_LABELS[b] for b in BASELINES], rotation=30, ha="right")
    ax.set_yticks(range(len(win_df.index)))
    ax.set_yticklabels(win_df.index)
    ax.set_title("Win-rate (%) vs.\\ baseline\n(green = AOM-Ridge wins more often)")
    for i, row in enumerate(win_df.index):
        for j, b in enumerate(BASELINES):
            v = win_df.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v < 30 or v > 70 else "black")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle("AOM-Ridge variants vs.\\ TabPFN-paper baselines (52 datasets)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return _save(fig, out)


def fig_per_dataset_blender(df: pd.DataFrame, out: Path) -> list[Path]:
    """For each dataset, show Blender's delta vs each baseline as a grouped bar."""
    sv = df[df["variant"] == "AOMRidge-Blender-headline-spxy3"].copy()
    if len(sv) == 0:
        return []
    sv = sv.sort_values("dataset").reset_index(drop=True)
    n = len(sv)
    deltas = {}
    for b in BASELINES:
        deltas[b] = 100.0 * (sv["rmsep"] - sv[b]) / sv[b]

    fig, ax = plt.subplots(figsize=(13, 8))
    bar_h = 0.13
    x = np.arange(n)
    colors = plt.cm.tab10(np.arange(len(BASELINES)))
    for i, b in enumerate(BASELINES):
        d = deltas[b].values
        d_clipped = np.clip(d, -50, 50)  # cap for readability
        bars = ax.barh(x + (i - len(BASELINES) / 2 + 0.5) * bar_h, d_clipped,
                       height=bar_h, color=colors[i], label=BASELINE_LABELS[b])
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(x)
    ax.set_yticklabels(sv["dataset"], fontsize=6)
    ax.set_xlabel("Δ (%) — Blender vs.\\ baseline (negative = Blender wins)")
    ax.set_xlim(-55, 55)
    ax.set_title("Per-dataset Blender vs.\\ TabPFN-paper baselines (capped at ±50%)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.invert_yaxis()
    fig.tight_layout()
    return _save(fig, out)


def fig_radar_oracle(df: pd.DataFrame, out: Path) -> list[Path]:
    """Radar / spider chart: oracle envelope vs each baseline."""
    oracle_idx = df[df["variant"].str.startswith("AOMRidge")].groupby("dataset")["rmsep"].idxmin()
    oracle = df.loc[oracle_idx]
    blender = df[df["variant"] == "AOMRidge-Blender-headline-spxy3"]

    metrics = []
    for variant_name, sub in [("Oracle", oracle), ("Blender", blender)]:
        row = {"variant": variant_name}
        for b in BASELINES:
            s = sub.dropna(subset=[b])
            if len(s) == 0:
                row[b] = 50.0
                continue
            wins = 100.0 * ((s["rmsep"] < s[b]).sum()) / len(s)
            row[b] = wins
        metrics.append(row)

    angles = np.linspace(0, 2 * np.pi, len(BASELINES), endpoint=False)
    angles_close = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    for i, m in enumerate(metrics):
        vals = [m[b] for b in BASELINES]
        vals_close = vals + [vals[0]]
        color = "tab:red" if m["variant"] == "Oracle" else "tab:blue"
        ax.plot(angles_close, vals_close, color=color, lw=2.5,
                label=f'{m["variant"]} (mean win-rate={np.mean(vals):.1f}%)')
        ax.fill(angles_close, vals_close, color=color, alpha=0.20)

    ax.set_xticks(angles)
    ax.set_xticklabels([BASELINE_LABELS[b] for b in BASELINES], fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"])
    # Mark the 50% threshold (tie line)
    ax.plot(angles_close, [50] * len(angles_close), color="black", ls="--", lw=0.8, alpha=0.5)
    ax.set_title("Win-rate of AOM-Ridge variants vs.\\ TabPFN-paper baselines\n"
                 "(52 datasets; 50% dashed line = tie)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    return _save(fig, out)


def fig_winner_distribution(df: pd.DataFrame, out: Path) -> list[Path]:
    """Bar chart: which AOM-Ridge variant wins on each dataset (oracle composition)."""
    aom_only = df[df["variant"].str.startswith("AOMRidge")].copy()
    aom_only = aom_only[aom_only["variant"].isin(TOP_VARIANTS)]
    idx = aom_only.groupby("dataset")["rmsep"].idxmin()
    winners = aom_only.loc[idx]
    counts = winners["variant"].value_counts()
    labels = [VARIANT_LABELS.get(v, v) for v in counts.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, counts.values, color=plt.cm.tab10(np.arange(len(labels))))
    ax.set_xlabel("Number of datasets where this variant is the AOM-Ridge winner")
    ax.set_title(f"Per-dataset winner among AOM-Ridge top variants\n"
                 f"({len(winners)} datasets, top variants only)")
    for bar, v in zip(bars, counts.values):
        ax.text(v + 0.1, bar.get_y() + bar.get_height() / 2, str(v),
                va="center", fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    return _save(fig, out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render TabPFN comparison figures.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    print(f"[fig] results: {args.results}")
    print(f"[fig] master:  {args.master}")
    print(f"[fig] out:     {args.out}")

    df = _load(args.results, args.master)
    print(f"[fig] loaded {len(df)} (dataset, variant) rows")

    artefacts = []
    artefacts += fig_grid_median_delta(df, args.out / "fig_aomridge_vs_baselines_grid")
    print("  - fig_aomridge_vs_baselines_grid.{pdf,png}")
    artefacts += fig_per_dataset_blender(df, args.out / "fig_per_dataset_blender_vs_baselines")
    print("  - fig_per_dataset_blender_vs_baselines.{pdf,png}")
    artefacts += fig_radar_oracle(df, args.out / "fig_radar_oracle_vs_baselines")
    print("  - fig_radar_oracle_vs_baselines.{pdf,png}")
    artefacts += fig_winner_distribution(df, args.out / "fig_aom_winner_distribution")
    print("  - fig_aom_winner_distribution.{pdf,png}")

    print("\nGenerated:")
    for p in artefacts:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
