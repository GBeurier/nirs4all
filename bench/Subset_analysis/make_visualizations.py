#!/usr/bin/env python3
"""Generate class-balanced, paper-aware subset visualizations."""
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


def load_data():
    search = pd.read_csv(ROOT / "class_balanced_subset_search_results.csv")
    z = pd.read_csv(ROOT / "class_dataset_zscores.csv").set_index("model_class")
    paper = pd.read_csv(ROOT / "tabpfn_paper_scores_core_pivot.csv")
    selected = json.loads((ROOT / "selected_subset.json").read_text())
    return search, z, paper, selected


def fig_paper_coverage(paper: pd.DataFrame) -> Path:
    model_cols = [c for c in paper.columns if c.startswith("Paper_")]
    avail = paper.set_index("dataset")[model_cols].notna().T.astype(int)
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.imshow(avail.values, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_yticks(range(len(model_cols)))
    ax.set_yticklabels(model_cols)
    ax.set_xticks(range(len(avail.columns)))
    ax.set_xticklabels(avail.columns, rotation=80, ha="right", fontsize=6)
    ax.set_title("Couverture explicite des scores du papier TabPFN sur le coeur 57")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Classe papier")
    for y, model in enumerate(model_cols):
        n = int(avail.loc[model].sum())
        ax.text(len(avail.columns) + 0.5, y, f"{n}/57", va="center", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "fig_paper_score_coverage.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_metrics(search: pd.DataFrame, selected: dict) -> Path:
    sel_size = int(selected["selected_size"])
    cons = selected.get("conservative_alternative")
    cons_size = int(cons["size"]) if cons else None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for col, label, color in [
        ("spearman", "Spearman", "#1f77b4"),
        ("kendall", "Kendall", "#2ca02c"),
        ("pairwise_agreement", "Accord par paires", "#d62728"),
    ]:
        ax.plot(search["size"], search[col], marker="o", label=label, color=color)
    ax.axhline(0.99, color="#1f77b4", ls=":", alpha=0.5)
    ax.axhline(0.95, color="#2ca02c", ls=":", alpha=0.5)
    ax.axhline(0.97, color="#d62728", ls=":", alpha=0.5)
    ax.axvline(sel_size, color="black", ls="--", label=f"Retenu n={sel_size}")
    if cons_size:
        ax.axvline(cons_size, color="grey", ls="--", label=f"Conservateur n={cons_size}")
    ax.set_title("Fidelite de rang entre classes")
    ax.set_xlabel("Taille du sous-ensemble")
    ax.set_ylim(0.75, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(search["size"], search["agg_mae"], marker="s", label="Erreur agregee", color="#9467bd")
    ax.axhline(0.08, color="#9467bd", ls=":", alpha=0.6, label="Seuil principal 0.08")
    ax.axhline(0.05, color="#9467bd", ls="--", alpha=0.5, label="Seuil conservateur 0.05")
    ax.plot(search["size"], 1 - search["selected_coverage_ratio"], marker="^",
            label="Taux manquant selection", color="#ff7f0e")
    ax.axvline(sel_size, color="black", ls="--")
    if cons_size:
        ax.axvline(cons_size, color="grey", ls="--")
    ax.set_title("Garde-fous : erreur absolue et couverture")
    ax.set_xlabel("Taille du sous-ensemble")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "fig_class_metrics_vs_size.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_random(search: pd.DataFrame, selected: dict) -> Path:
    sel_size = int(selected["selected_size"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(search["size"], search["agg_mae"], marker="o", label="Glouton", color="#9467bd")
    ax.plot(search["size"], search["random_agg_mae_mean"], marker="x", label="Aleatoire moyen", color="#ff7f0e")
    ax.axvline(sel_size, color="black", ls="--")
    ax.set_title("Erreur agregee : glouton vs aleatoire")
    ax.set_xlabel("Taille")
    ax.set_ylabel("MAE agregee (plus bas = mieux)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(search["size"], search["selected_coverage_ratio"], marker="o",
            label="Glouton", color="#1f77b4")
    ax.plot(search["size"], search["random_coverage_mean"], marker="x",
            label="Aleatoire moyen", color="#ff7f0e")
    ax.axhline(0.98, color="grey", ls=":", alpha=0.6)
    ax.axvline(sel_size, color="black", ls="--")
    ax.set_title("Couverture des classes sur le subset")
    ax.set_xlabel("Taille")
    ax.set_ylabel("Ratio de scores disponibles")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "fig_class_greedy_vs_random.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_heatmap(z: pd.DataFrame, selected: dict) -> Path:
    cols = selected["selected_datasets"]
    Z = z[cols]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    vmax = float(np.nanpercentile(np.abs(Z.values), 98))
    im = ax.imshow(Z.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(Z.index)))
    ax.set_yticklabels(Z.index)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=65, ha="right", fontsize=8)
    ax.set_title("Empreintes log-RMSEP z-score par classe sur les datasets retenus")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("z-score (negatif = meilleur)")
    fig.tight_layout()
    out = FIG_DIR / "fig_class_zscore_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_pca(z: pd.DataFrame, selected: dict) -> Path:
    data = z.T.fillna(z.T.mean(axis=0))
    coords = PCA(n_components=2, random_state=1234).fit_transform(data.values)
    sel = set(selected["selected_datasets"])
    cons = selected.get("conservative_alternative") or {}
    cons_only = set(cons.get("datasets", [])) - sel

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, dataset in enumerate(data.index):
        if dataset in sel:
            marker, color, size, label = "*", "#d62728", 240, "Retenu"
        elif dataset in cons_only:
            marker, color, size, label = "D", "#2ca02c", 85, "Ajout conservateur"
        else:
            marker, color, size, label = "o", "#8c8c8c", 35, "Autre"
        ax.scatter(coords[i, 0], coords[i, 1], marker=marker, c=color, s=size,
                   edgecolors="black", linewidths=0.4, alpha=0.9, label=label)
        if dataset in sel:
            ax.annotate(dataset, (coords[i, 0], coords[i, 1]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=9)
    ax.set_title("PCA des 57 datasets sur empreintes de classes de modeles")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig_class_pca_datasets.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    search, z, paper, selected = load_data()
    outputs = [
        fig_paper_coverage(paper),
        fig_metrics(search, selected),
        fig_random(search, selected),
        fig_heatmap(z, selected),
        fig_pca(z, selected),
    ]
    print("Figures generees :")
    for path in outputs:
        print(f"  - {path.relative_to(ROOT.parent.parent)}")


if __name__ == "__main__":
    main()
