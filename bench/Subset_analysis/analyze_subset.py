#!/usr/bin/env python3
"""Paper-aware, class-balanced subset selection for TabPFN-paper datasets.

The first version of this analysis optimized over 128 AOM variants. That was
useful as a stress test, but it over-weighted one model family. This script
keeps the 57 regression datasets as the evaluation universe, makes the TabPFN
paper scores explicit, collapses model variants into model classes, and selects
datasets from an equal-class matrix.

Run:
    python3 bench/Subset_analysis/analyze_subset.py
"""
from __future__ import annotations

import json
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

SRC_AOM = ROOT / "bench/AOM_v0/publication/tables/tabpfn_comparison_per_dataset.csv"
SRC_MASTER = ROOT / "bench/AOM_v0/publication/tables/master_pivot.csv"
SRC_NICON = ROOT / "bench/nicon_v2/publication/tables/full_comparison/long_per_dataset.csv"

PAPER_COLUMNS = {
    "CNN": "Paper_CNN",
    "Catboost": "Paper_CatBoost",
    "PLS": "Paper_PLS",
    "Ridge": "Paper_Ridge",
    "TabPFN-Raw": "Paper_TabPFN_Raw",
    "TabPFN-opt": "Paper_TabPFN_Opt",
}

NICON_CLASS_MAP = {
    "AOM-PLS PLS-standard": "AOM_PLS_Standard",
    "AOM-PLS-best": "AOM_PLS_Best",
    "AOM-Ridge-best": "AOM_Ridge_Best",
    "nicon_v2 internal Ridge": "Nicon_Internal_Ridge",
    "nicon_v2 internal PLS": "Nicon_Internal_PLS",
    "nicon_v2 V1c-concat-bjerrum (CNN-only)": "Nicon_CNN_V1c",
    "nicon_v2 Stack-Ridge-PLS-V1c": "Nicon_Stack_Ridge_PLS_V1c",
}

MAIN_CLASSES = [
    "Paper_CNN",
    "Paper_CatBoost",
    "Paper_PLS",
    "Paper_Ridge",
    "Paper_TabPFN_Raw",
    "Paper_TabPFN_Opt",
    "AOM_PLS_Standard",
    "AOM_PLS_Best",
]

THRESHOLDS = {
    "min_subset_size": 10,
    "spearman": 0.99,
    "kendall": 0.95,
    "pairwise_agreement": 0.97,
    "agg_mae": 0.08,
    "selected_coverage_ratio": 0.98,
    "min_class_selected_count": 3,
}

CONSERVATIVE_THRESHOLDS = {
    "agg_mae": 0.05,
    "selected_coverage_ratio": 0.97,
}


def rankdata(a: np.ndarray) -> np.ndarray:
    s = pd.Series(a)
    return s.rank(method="average").to_numpy(dtype=float)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    rx = rankdata(x[mask])
    ry = rankdata(y[mask])
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    vals_x = x[mask]
    vals_y = y[mask]
    c = d = 0
    for i, j in combinations(range(len(vals_x)), 2):
        prod = (vals_x[i] - vals_x[j]) * (vals_y[i] - vals_y[j])
        if prod > 0:
            c += 1
        elif prod < 0:
            d += 1
    total = c + d
    return (c - d) / total if total else 0.0


def pairwise_sign_agreement(x: np.ndarray, y: np.ndarray, tol: float = 1e-3) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    vals_x = x[mask]
    vals_y = y[mask]
    if len(vals_x) < 2:
        return 0.0
    agreements = []
    for i, j in combinations(range(len(vals_x)), 2):
        dx = vals_x[i] - vals_x[j]
        dy = vals_y[i] - vals_y[j]
        sx = 0 if abs(dx) <= tol else np.sign(dx)
        sy = 0 if abs(dy) <= tol else np.sign(dy)
        agreements.append(sx == sy)
    return float(np.mean(agreements))


def core_datasets() -> list[str]:
    df = pd.read_csv(SRC_AOM, low_memory=False)
    df = df[df["task"] == "regression"].dropna(subset=["model", "dataset", "RMSEP"])
    coverage = df.groupby("model")["dataset"].nunique()
    eligible = coverage[coverage >= 57].index.tolist()
    sets = [set(df.loc[df["model"] == m, "dataset"]) for m in eligible]
    if not sets:
        raise RuntimeError("Unable to derive the 57-dataset core from AOM results.")
    return sorted(set.intersection(*sets))


def export_paper_scores(core: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = pd.read_csv(SRC_MASTER)
    long_rows = []
    for raw_col, model_class in PAPER_COLUMNS.items():
        keep = master[["database_name", "dataset", raw_col]].rename(columns={raw_col: "RMSEP"})
        keep = keep.dropna(subset=["RMSEP"])
        for row in keep.itertuples(index=False):
            long_rows.append(
                {
                    "source": "TabPFN_paper/master_pivot",
                    "model_class": model_class,
                    "paper_model_label": raw_col,
                    "dataset_group": row.database_name,
                    "dataset": row.dataset,
                    "metric_name": "RMSEP",
                    "metric_value": float(row.RMSEP),
                    "lower_is_better": True,
                    "in_core_57": row.dataset in core,
                }
            )
    long = pd.DataFrame(long_rows)
    long.to_csv(OUT / "tabpfn_paper_scores_long.csv", index=False)

    pivot = pd.DataFrame({"dataset": core})
    groups = master.set_index("dataset")["database_name"].to_dict()
    pivot.insert(0, "dataset_group", [groups.get(d, "") for d in core])
    for raw_col, model_class in PAPER_COLUMNS.items():
        vals = master.set_index("dataset")[raw_col].reindex(core)
        pivot[model_class] = vals.to_numpy()
    pivot.to_csv(OUT / "tabpfn_paper_scores_core_pivot.csv", index=False)
    return long, pivot


def build_class_matrix(core: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, paper_pivot = export_paper_scores(core)
    rows = {}
    source_model = {}

    for model_class in PAPER_COLUMNS.values():
        rows[model_class] = paper_pivot.set_index("dataset")[model_class].reindex(core)
        source_model[model_class] = "TabPFN paper master_pivot"

    nicon = pd.read_csv(SRC_NICON)
    for raw_class, model_class in NICON_CLASS_MAP.items():
        vals = (
            nicon[nicon["model_class"] == raw_class]
            .groupby("dataset")["rmsep"]
            .median()
            .reindex(core)
        )
        rows[model_class] = vals
        source_model[model_class] = f"nicon_v2 long_per_dataset::{raw_class}"

    matrix = pd.DataFrame(rows).T
    matrix.index.name = "model_class"
    matrix.to_csv(OUT / "class_score_matrix.csv")
    matrix.loc[[c for c in MAIN_CLASSES if c in matrix.index]].to_csv(
        OUT / "main_class_score_matrix.csv"
    )

    mapping_rows = []
    for model_class, vals in matrix.iterrows():
        n_scores = int(vals.notna().sum())
        included = model_class in MAIN_CLASSES
        if included:
            reason = "included_main_equal_class_weight"
        elif n_scores < 49:
            reason = "excluded_from_main_low_core_coverage"
        else:
            reason = "excluded_from_main_extra_diagnostic_class"
        mapping_rows.append(
            {
                "model_class": model_class,
                "source_model": source_model.get(model_class, ""),
                "n_core_scores": n_scores,
                "coverage_ratio": n_scores / len(core),
                "included_in_balanced_analysis": included,
                "reason": reason,
            }
        )
    mapping = pd.DataFrame(mapping_rows).sort_values(
        ["included_in_balanced_analysis", "model_class"], ascending=[False, True]
    )
    mapping.to_csv(OUT / "model_class_mapping.csv", index=False)
    return matrix, mapping


def transform_class_scores(matrix: pd.DataFrame, included_classes: list[str]) -> pd.DataFrame:
    included = matrix.loc[included_classes]
    logged = np.log(included.clip(lower=1e-12))
    z = logged.copy()
    for col in z.columns:
        vals = z[col]
        mu = vals.mean(skipna=True)
        sd = vals.std(skipna=True)
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        z[col] = (vals - mu) / sd
    z.to_csv(OUT / "class_dataset_zscores.csv")
    return z


def evaluate_subset(z: pd.DataFrame, subset: list[str], full_mean: np.ndarray) -> dict[str, float]:
    selected_counts = z[subset].notna().sum(axis=1)
    if (selected_counts == 0).any():
        return {
            "spearman": -1.0,
            "kendall": -1.0,
            "pairwise_agreement": 0.0,
            "rank_mae": 999.0,
            "agg_mae": 999.0,
            "selected_coverage_ratio": 0.0,
            "min_class_selected_count": 0,
            "composite": -999.0,
        }

    subset_mean = z[subset].mean(axis=1, skipna=True).to_numpy(dtype=float)
    mask = np.isfinite(subset_mean) & np.isfinite(full_mean)
    sp = spearman(subset_mean, full_mean)
    kt = kendall(subset_mean, full_mean)
    pa = pairwise_sign_agreement(subset_mean, full_mean)
    rank_mae = float(np.mean(np.abs(rankdata(subset_mean[mask]) - rankdata(full_mean[mask]))))
    agg_mae = float(np.mean(np.abs(subset_mean[mask] - full_mean[mask])))
    coverage = float(z[subset].notna().to_numpy().mean())
    min_count = int(selected_counts.min())

    composite = (
        0.25 * sp
        + 0.20 * kt
        + 0.20 * pa
        - 0.20 * agg_mae
        + 0.15 * coverage
    )
    return {
        "spearman": sp,
        "kendall": kt,
        "pairwise_agreement": pa,
        "rank_mae": rank_mae,
        "agg_mae": agg_mae,
        "selected_coverage_ratio": coverage,
        "min_class_selected_count": min_count,
        "composite": composite,
    }


def greedy_search(z: pd.DataFrame, max_size: int = 30) -> tuple[list[dict], np.ndarray]:
    full_mean = z.mean(axis=1, skipna=True).to_numpy(dtype=float)
    chosen: list[str] = []
    rows = []
    for size in range(1, max_size + 1):
        best_dataset = None
        best_metrics = None
        best_score = -np.inf
        for dataset in z.columns:
            if dataset in chosen:
                continue
            metrics = evaluate_subset(z, chosen + [dataset], full_mean)
            if metrics["composite"] > best_score:
                best_score = metrics["composite"]
                best_dataset = dataset
                best_metrics = metrics
        if best_dataset is None or best_metrics is None:
            break
        chosen.append(best_dataset)
        row = {"size": size, "added": best_dataset, "subset": "|".join(chosen)}
        row.update(best_metrics)
        rows.append(row)
    return rows, full_mean


def random_baselines(z: pd.DataFrame, sizes: list[int], full_mean: np.ndarray, n_trials: int = 200) -> pd.DataFrame:
    rng = np.random.RandomState(SEED)
    cols = np.array(z.columns)
    rows = []
    for size in sizes:
        metrics = []
        for _ in range(n_trials):
            subset = rng.choice(cols, size=size, replace=False).tolist()
            metrics.append(evaluate_subset(z, subset, full_mean))
        frame = pd.DataFrame(metrics)
        rows.append(
            {
                "size": size,
                "random_n_trials": n_trials,
                "random_spearman_mean": frame["spearman"].mean(),
                "random_spearman_p05": frame["spearman"].quantile(0.05),
                "random_kendall_mean": frame["kendall"].mean(),
                "random_pairwise_mean": frame["pairwise_agreement"].mean(),
                "random_agg_mae_mean": frame["agg_mae"].mean(),
                "random_agg_mae_p05": frame["agg_mae"].quantile(0.05),
                "random_coverage_mean": frame["selected_coverage_ratio"].mean(),
            }
        )
    return pd.DataFrame(rows)


def leave_one_class_out(z: pd.DataFrame, subset: list[str]) -> dict[str, float]:
    vals = []
    for model_class in z.index:
        reduced = z.drop(index=model_class)
        full = reduced.mean(axis=1, skipna=True).to_numpy(dtype=float)
        vals.append(evaluate_subset(reduced, subset, full))
    frame = pd.DataFrame(vals)
    return {
        "loco_spearman_min": float(frame["spearman"].min()),
        "loco_kendall_min": float(frame["kendall"].min()),
        "loco_pairwise_min": float(frame["pairwise_agreement"].min()),
        "loco_agg_mae_max": float(frame["agg_mae"].max()),
    }


def pick_selection(rows: list[dict]) -> tuple[dict, dict | None]:
    def passes(row: dict) -> bool:
        return (
            row["size"] >= THRESHOLDS["min_subset_size"]
            and row["spearman"] >= THRESHOLDS["spearman"]
            and row["kendall"] >= THRESHOLDS["kendall"]
            and row["pairwise_agreement"] >= THRESHOLDS["pairwise_agreement"]
            and row["agg_mae"] <= THRESHOLDS["agg_mae"]
            and row["selected_coverage_ratio"] >= THRESHOLDS["selected_coverage_ratio"]
            and row["min_class_selected_count"] >= THRESHOLDS["min_class_selected_count"]
        )

    selected = next((r for r in rows if r["size"] >= 3 and passes(r)), None)
    if selected is None:
        best = max(rows, key=lambda r: r["composite"])
        selected = next(
            r
            for r in rows
            if r["size"] >= 3 and r["composite"] >= best["composite"] - 0.01 * abs(best["composite"])
        )

    conservative = next(
        (
            r
            for r in rows
            if r["size"] >= selected["size"]
            and passes(r)
            and r["agg_mae"] <= CONSERVATIVE_THRESHOLDS["agg_mae"]
            and r["selected_coverage_ratio"] >= CONSERVATIVE_THRESHOLDS["selected_coverage_ratio"]
        ),
        None,
    )
    return selected, conservative


def write_reports(selected: dict, conservative: dict | None, mapping: pd.DataFrame, core: list[str]) -> None:
    selected_datasets = selected["subset"].split("|")
    conservative_datasets = conservative["subset"].split("|") if conservative is not None else []
    included = mapping[mapping["included_in_balanced_analysis"]].copy()
    excluded = mapping[~mapping["included_in_balanced_analysis"]].copy()

    selected_json = {
        "protocol_version": "class_balanced_paper_aware",
        "seed": SEED,
        "n_core_datasets": len(core),
        "selected_size": int(selected["size"]),
        "selected_datasets": selected_datasets,
        "metrics_full_vs_subset": {k: selected[k] for k in [
            "spearman",
            "kendall",
            "pairwise_agreement",
            "rank_mae",
            "agg_mae",
            "selected_coverage_ratio",
            "min_class_selected_count",
            "composite",
        ]},
        "thresholds": THRESHOLDS,
        "model_classes_included": included.to_dict(orient="records"),
        "model_classes_excluded_from_main": excluded.to_dict(orient="records"),
        "leave_one_class_out": leave_one_class_out(
            pd.read_csv(OUT / "class_dataset_zscores.csv").set_index("model_class"),
            selected_datasets,
        ),
        "conservative_alternative": None
        if conservative is None
        else {
            "size": int(conservative["size"]),
            "datasets": conservative_datasets,
            "metrics_full_vs_subset": {k: conservative[k] for k in [
                "spearman",
                "kendall",
                "pairwise_agreement",
                "rank_mae",
                "agg_mae",
                "selected_coverage_ratio",
                "min_class_selected_count",
                "composite",
            ]},
            "extra_thresholds": CONSERVATIVE_THRESHOLDS,
        },
        "method_note": (
            "Main selection uses one equal-weight row per model class. AOM variants are not "
            "individual voting rows; they are represented by AOM_PLS_Standard and AOM_PLS_Best only."
        ),
    }
    (OUT / "selected_subset.json").write_text(json.dumps(selected_json, indent=2, ensure_ascii=False), encoding="utf-8")

    report = [
        "# Rapport - protocole class-balanced et paper-aware\n",
        "## Correction apportee\n",
            "La premiere analyse utilisait 128 variantes AOM comme lignes de la matrice principale. "
            "Cela surrepresentait une seule famille. Le protocole courant remplace cette matrice par "
            "une matrice de **classes de modeles a poids egal** et rend les scores du papier TabPFN explicites.\n",
            f"Un plancher de **{THRESHOLDS['min_subset_size']} datasets** est impose : avec 8 classes seulement, les metriques de rang peuvent saturer artificiellement sur des subsets trop petits.\n",
        "## Scores papier TabPFN\n",
        "Les scores papier sont exportes dans `tabpfn_paper_scores_long.csv` et `tabpfn_paper_scores_core_pivot.csv`. "
        "Les classes visibles sont `Paper_CNN`, `Paper_CatBoost`, `Paper_PLS`, `Paper_Ridge`, "
        "`Paper_TabPFN_Raw`, `Paper_TabPFN_Opt`. CatBoost n'est donc plus anonymise.\n",
        "## Classes dans l'analyse principale\n",
    ]
    for row in included.itertuples(index=False):
        report.append(f"- `{row.model_class}` : {row.n_core_scores}/57 scores coeur, source `{row.source_model}`.")
    report.append("\nClasses exclues de la selection principale faute de couverture suffisante ou parce qu'elles sont diagnostiques :")
    for row in excluded.itertuples(index=False):
        report.append(f"- `{row.model_class}` : {row.n_core_scores}/57, raison `{row.reason}`.")
    report.extend(
        [
            "\nLes deux voix AOM conservees sont intentionnelles : `AOM_PLS_Standard` represente la version standard, et `AOM_PLS_Best` represente la meilleure variante AOM-PLS agregee. Aucune variante AOM individuelle ne vote dans la selection.\n",
            "\n## Methode\n",
            "1. Coeur 57 reconstruit depuis `tabpfn_comparison_per_dataset.csv`, pas depuis les artefacts legacy.",
            "2. `class_score_matrix.csv` conserve toutes les classes candidates pour audit ; `main_class_score_matrix.csv` garde les 8 classes a poids egal utilisees pour la selection.",
            "3. Transformation `log(RMSEP)` puis z-score par dataset entre classes disponibles dans `class_dataset_zscores.csv`.",
            "4. Recherche gloutonne sur les 8 classes principales, avec penalite de couverture et `agg_mae` dans l'objectif.",
            f"5. Baselines aleatoires, plancher de taille n>={THRESHOLDS['min_subset_size']} et test leave-one-class-out pour eviter de surinterpreter les rangs, car 8 classes seulement font saturer Spearman/Kendall.\n",
            "## Recommandation revisee\n",
            f"- Taille retenue : **{selected['size']} datasets**.",
            f"- Critere principal : `agg_mae` {selected['agg_mae']:.4f}, couverture selectionnee {selected['selected_coverage_ratio']:.4f}, min scores par classe {selected['min_class_selected_count']}.",
            f"- Diagnostics de rang, a interpreter avec prudence car seulement 8 classes : Spearman {selected['spearman']:.4f}, Kendall {selected['kendall']:.4f}, accord {selected['pairwise_agreement']:.4f}.",
            "- Datasets retenus :",
        ]
    )
    report.extend([f"  - `{d}`" for d in selected_datasets])
    if conservative is not None:
        report.extend(
            [
                "\n## Alternative conservatrice\n",
                f"Plus petite taille avec `agg_mae <= {CONSERVATIVE_THRESHOLDS['agg_mae']}` et couverture >= {CONSERVATIVE_THRESHOLDS['selected_coverage_ratio']} : **{conservative['size']} datasets**.",
            ]
        )
    report.extend(
        [
            "\n## Limites\n",
            "- Les rangs sur 8 classes saturent vite ; le critere discriminant devient surtout `agg_mae` et la couverture.",
            "- Les scores papier CNN/PLS/Ridge/CatBoost ont quelques NaN sur les 57 datasets ; ils sont traites explicitement, pas imputes.",
            "- Les sorties legacy sont regroupees dans `legacy_variant_heavy/` pour audit, mais ne portent plus la recommandation principale.\n",
            "## Reproduction\n",
            "```\npython3 bench/Subset_analysis/analyze_subset.py\npython3 bench/Subset_analysis/make_visualizations.py\n```\n",
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    readme = [
        "# bench/Subset_analysis\n",
        "Analyse paper-aware et class-balanced pour selectionner un sous-ensemble representatif des 57 datasets de regression communs.\n",
        "## Sorties principales\n",
        "- `tabpfn_paper_scores_long.csv` / `tabpfn_paper_scores_core_pivot.csv` : scores explicites du papier TabPFN.",
        "- `model_class_mapping.csv` : classes incluses/exclues et couverture.",
        "- `class_score_matrix.csv` : RMSEP par classe candidate x dataset, incluant les classes diagnostiques exclues.",
        "- `main_class_score_matrix.csv` : RMSEP des 8 classes principales utilisees a poids egal.",
        "- `class_dataset_zscores.csv` : log-RMSEP z-score par dataset sur les 8 classes principales.",
        "- `class_balanced_subset_search_results.csv` : recherche gloutonne class-balanced + baselines aleatoires.",
        "- `selected_subset.json` : recommandation principale class-balanced.",
        "- `REPORT.md` : rapport de protocole.",
        "- `SYNTHESE_TECHNIQUE.md` : synthese technique avec figures.",
        "- `make_visualizations.py` : genere les figures dans `figures/`.",
        "- `subset_transfer_analysis.py` : teste le transfert selection-sur-subset -> resultats globaux depuis `benchmark_master_results.csv`.",
        "- `SUBSET_TRANSFER_REPORT.md` / `subset_transfer_summary.csv` : diagnostic direct sur les subsets courants, baselines aleatoires, representativite et garde-fous.\n",
        "Les anciens artefacts variant-heavy sont regroupes dans `legacy_variant_heavy/` ; ils ne doivent plus etre utilises comme recommandation principale.\n",
        "## Execution\n",
        "```\npython3 bench/Subset_analysis/analyze_subset.py\npython3 bench/Subset_analysis/make_visualizations.py\npython3 bench/Subset_analysis/subset_transfer_analysis.py\n```\n",
    ]
    (OUT / "README.md").write_text("\n".join(readme), encoding="utf-8")

    synth = [
        "# Synthese technique - protocole class-balanced paper-aware\n",
        "## Pourquoi revoir le protocole ?\n",
        "L'analyse initiale contenait beaucoup de variantes AOM. Meme si elle etait statistiquement stable, elle ponderait trop fortement une seule famille. La version revisee donne une voix a chaque **classe de modeles** : CNN papier, CatBoost papier, PLS papier, Ridge papier, TabPFN raw/opt, et deux classes AOM maximum.\n",
        "## Scores papier visibles\n",
        "Les scores du papier TabPFN sont dans `tabpfn_paper_scores_core_pivot.csv` et visualises par la figure de couverture.\n",
        "![Couverture scores papier](figures/fig_paper_score_coverage.png)\n",
        "## Selection class-balanced\n",
        "La matrice principale est `main_class_score_matrix.csv`. Elle contient des NaN assumes pour les scores papier manquants. Les scores sont transformes en `log(RMSEP)`, puis z-score par dataset entre classes disponibles dans `class_dataset_zscores.csv`. `class_score_matrix.csv` reste un fichier d'audit avec les classes candidates exclues.\n",
        "Les deux lignes AOM conservees sont `AOM_PLS_Standard` et `AOM_PLS_Best`; aucune variante AOM individuelle ne vote dans la selection.\n",
        "![Z-scores classes](figures/fig_class_zscore_heatmap.png)\n",
        "La recherche gloutonne maximise un composite qui combine fidelite de rang, erreur agregee et couverture. Les rangs saturant vite avec 8 classes, la decision se lit surtout via `agg_mae` et la couverture.\n",
        "![Metriques class-balanced](figures/fig_class_metrics_vs_size.png)\n",
        "Les baselines aleatoires montrent ce qui est gagne par la selection active plutot que par la seule taille du sous-ensemble.\n",
        "![Glouton vs aleatoire class-balanced](figures/fig_class_greedy_vs_random.png)\n",
        "La PCA illustre la repartition des datasets retenus dans l'espace des empreintes de performance par classe.\n",
        "![PCA class-balanced](figures/fig_class_pca_datasets.png)\n",
        "## Recommandation\n",
        f"Le sous-ensemble principal contient **{selected['size']} datasets** : " + ", ".join(f"`{d}`" for d in selected_datasets) + ".\n",
        "Pour une decision finale ou une comparaison serree, utiliser l'alternative conservatrice indiquee dans `selected_subset.json`.\n",
    ]
    (OUT / "SYNTHESE_TECHNIQUE.md").write_text("\n".join(synth), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    core = core_datasets()
    class_matrix, mapping = build_class_matrix(core)
    included_classes = mapping.loc[mapping["included_in_balanced_analysis"], "model_class"].tolist()
    z = transform_class_scores(class_matrix, included_classes)

    rows, full_mean = greedy_search(z, max_size=min(30, len(core)))
    search = pd.DataFrame(rows)
    sizes = search.loc[search["size"] >= 3, "size"].astype(int).tolist()
    random_df = random_baselines(z, sizes, full_mean, n_trials=200)
    search_main = search[search["size"] >= 3].merge(random_df, on="size", how="left")
    search_main.to_csv(OUT / "class_balanced_subset_search_results.csv", index=False)

    selected, conservative = pick_selection(rows)
    write_reports(selected, conservative, mapping, core)

    print("[OK] Class-balanced paper-aware analysis")
    print(f"[OK] Core datasets: {len(core)}; main classes: {len(included_classes)}")
    print(f"[OK] Selected size: {selected['size']}")
    print(f"     subset={selected['subset']}")
    print(f"     agg_mae={selected['agg_mae']:.4f} coverage={selected['selected_coverage_ratio']:.4f}")
    print(f"[OK] Outputs in {OUT}")


if __name__ == "__main__":
    main()
