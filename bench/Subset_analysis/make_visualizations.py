#!/usr/bin/env python3
"""Genere les figures explicatives pour la selection du sous-ensemble representatif.

Usage:
    python3 bench/Subset_analysis/make_visualizations.py

Sorties: bench/Subset_analysis/figures/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 1234

THRESHOLDS = {
    "spearman": 0.98,
    "kendall": 0.95,
    "pairwise": 0.97,
}


def load_data():
    search = pd.read_csv(ROOT / "subset_search_results.csv")
    random_bl = pd.read_csv(ROOT / "random_baselines.csv")
    zscores = pd.read_csv(ROOT / "model_dataset_zscores.csv").set_index("model")
    coverage = pd.read_csv(ROOT / "dataset_coverage.csv")
    selected = json.loads((ROOT / "selected_subset.json").read_text())
    return search, random_bl, zscores, coverage, selected


def fig_metrics_vs_size(search: pd.DataFrame, selected: dict) -> Path:
    selected_size = int(selected["selected_size"])
    conservative = selected.get("conservative_alternative")
    conservative_size = int(conservative["size"]) if conservative else None

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(search["size"], search["greedy_spearman"], marker="o",
            label="Spearman (glouton)", color="#1f77b4")
    ax.plot(search["size"], search["greedy_kendall"], marker="s",
            label="Kendall (glouton)", color="#2ca02c")
    ax.plot(search["size"], search["greedy_pairwise"], marker="^",
            label="Accord par paires (glouton)", color="#d62728")

    ax.axhline(THRESHOLDS["spearman"], color="#1f77b4", linestyle=":", alpha=0.6,
               label=f"Seuil Spearman = {THRESHOLDS['spearman']}")
    ax.axhline(THRESHOLDS["kendall"], color="#2ca02c", linestyle=":", alpha=0.6,
               label=f"Seuil Kendall = {THRESHOLDS['kendall']}")
    ax.axhline(THRESHOLDS["pairwise"], color="#d62728", linestyle=":", alpha=0.6,
               label=f"Seuil accord = {THRESHOLDS['pairwise']}")

    ax.axvline(selected_size, color="black", linestyle="--", alpha=0.7,
               label=f"Choix retenu (n={selected_size})")
    if conservative_size is not None:
        ax.axvline(conservative_size, color="grey", linestyle="--", alpha=0.7,
                   label=f"Alternative conservatrice (n={conservative_size})")

    ax.set_xlabel("Taille du sous-ensemble")
    ax.set_ylabel("Metrique de fidelite vs classement complet")
    ax.set_title("Fidelite du classement des modeles selon la taille du sous-ensemble")
    ax.set_ylim(0.5, 1.005)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    out = FIG_DIR / "fig_metrics_vs_size.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_greedy_vs_random(search: pd.DataFrame, selected: dict) -> Path:
    selected_size = int(selected["selected_size"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(search["size"], search["greedy_spearman"], marker="o",
            label="Glouton", color="#1f77b4")
    ax.plot(search["size"], search["random_spearman_mean"], marker="x",
            label="Aleatoire (moyenne)", color="#ff7f0e")
    ax.fill_between(search["size"], search["random_spearman_p05"],
                    search["random_spearman_mean"], color="#ff7f0e", alpha=0.2,
                    label="Aleatoire (p05 - moyenne)")
    ax.axhline(THRESHOLDS["spearman"], color="grey", linestyle=":", alpha=0.7)
    ax.axvline(selected_size, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Taille du sous-ensemble")
    ax.set_ylabel("Spearman")
    ax.set_title("Spearman : glouton vs baseline aleatoire")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(search["size"], search["greedy_pairwise"], marker="^",
            label="Glouton", color="#d62728")
    ax.plot(search["size"], search["random_pairwise_mean"], marker="x",
            label="Aleatoire (moyenne)", color="#ff7f0e")
    ax.fill_between(search["size"], search["random_pairwise_p05"],
                    search["random_pairwise_mean"], color="#ff7f0e", alpha=0.2,
                    label="Aleatoire (p05 - moyenne)")
    ax.axhline(THRESHOLDS["pairwise"], color="grey", linestyle=":", alpha=0.7)
    ax.axvline(selected_size, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Taille du sous-ensemble")
    ax.set_ylabel("Accord par paires")
    ax.set_title("Accord par paires : glouton vs baseline aleatoire")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("La selection gloutonne domine clairement les sous-ensembles aleatoires",
                 fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "fig_greedy_vs_random.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_zscore_heatmap(zscores: pd.DataFrame, selected: dict) -> Path:
    cols = selected["selected_datasets"]
    Z = zscores[cols]
    mean_perf = Z.mean(axis=1)
    order = mean_perf.sort_values().index.tolist()
    n = len(order)
    # Take 12 best, 12 worst, 12 around the median => max 36 models.
    k = 12
    median_start = max(0, n // 2 - k // 2)
    picks = order[:k] + order[median_start:median_start + k] + order[-k:]
    seen = set()
    picks_unique = [m for m in picks if not (m in seen or seen.add(m))]
    Zsub = Z.loc[picks_unique]

    fig, ax = plt.subplots(figsize=(11, max(6, 0.22 * len(picks_unique))))
    vmax = float(np.nanpercentile(np.abs(Zsub.values), 98))
    im = ax.imshow(Zsub.values, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(picks_unique)))
    ax.set_yticklabels(picks_unique, fontsize=7)
    ax.set_title("Empreintes z-score (log-RMSEP) sur les 11 datasets retenus\n"
                 "Modeles : meilleurs / mediane / pires (par moyenne globale)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("z-score (negatif = meilleur)")
    fig.tight_layout()
    out = FIG_DIR / "fig_zscore_heatmap.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_pca_datasets(zscores: pd.DataFrame, selected: dict) -> Path:
    # Datasets as rows, models as features.
    M = zscores.T  # (n_datasets, n_models)
    M = M.fillna(M.mean(axis=0))
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(M.values)
    var = pca.explained_variance_ratio_

    selected_set = set(selected["selected_datasets"])
    conservative_set = set(selected["conservative_alternative"]["datasets"]) - selected_set

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, name in enumerate(M.index):
        if name in selected_set:
            color, marker, size, label = "#d62728", "*", 240, "Retenu (n=11)"
        elif name in conservative_set:
            color, marker, size, label = "#2ca02c", "D", 90, "Ajout conservateur (n=26)"
        else:
            color, marker, size, label = "#7f7f7f", "o", 35, "Autre dataset"
        ax.scatter(coords[i, 0], coords[i, 1], c=color, marker=marker, s=size,
                   alpha=0.85, edgecolors="black", linewidths=0.4, label=label)

    # Dedup legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=9)

    # Annotate the 11 selected.
    for i, name in enumerate(M.index):
        if name in selected_set:
            ax.annotate(name, (coords[i, 0], coords[i, 1]),
                        fontsize=7, alpha=0.75, xytext=(4, 4),
                        textcoords="offset points")

    ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}% var.)")
    ax.set_title("PCA des 57 datasets a partir de leurs empreintes de performance modele")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig_pca_datasets.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_coverage(coverage: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = coverage.groupby("in_core_57").size().rename({True: "Coeur 57", False: "Hors coeur"})
    src_counts = coverage.groupby(["in_core_57", "n_sources"]).size().unstack(fill_value=0)
    src_counts.index = src_counts.index.map({True: "Coeur 57", False: "Hors coeur"})
    src_counts.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_ylabel("Nombre de datasets")
    ax.set_xlabel("")
    ax.set_title("Couverture des datasets : appartenance au coeur 57 et nombre de sources")
    ax.legend(title="Nombre de sources", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for container in ax.containers:
        ax.bar_label(container, fontsize=8, label_type="center")
    fig.tight_layout()
    out = FIG_DIR / "fig_coverage.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main():
    search, random_bl, zscores, coverage, selected = load_data()
    outputs = [
        fig_metrics_vs_size(search, selected),
        fig_greedy_vs_random(search, selected),
        fig_zscore_heatmap(zscores, selected),
        fig_pca_datasets(zscores, selected),
        fig_coverage(coverage),
    ]
    print("Figures generees :")
    for p in outputs:
        print(f"  - {p.relative_to(ROOT.parent.parent)}")


if __name__ == "__main__":
    main()
