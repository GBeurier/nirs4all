#!/usr/bin/env python3
"""Subset selection analysis for TabPFN-paper regression datasets.

Reads benchmark CSVs, builds a normalized long table of model x dataset RMSEP
scores, defines the core 57-dataset intersection, then performs greedy forward
selection (sizes 3..30) to identify a representative subset whose mean ranking
of models agrees with the full ranking. Outputs CSVs, JSON and markdown reports.

Run:
    python3 bench/Subset_analysis/analyze_subset.py
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

# Optional deps
try:
    from scipy.stats import spearmanr, kendalltau
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

SRC_AOM = ROOT / "bench/AOM_v0/publication/tables/tabpfn_comparison_per_dataset.csv"
SRC_AOM_MK = ROOT / "bench/AOM_v0/Multi-kernel/publication/tables/tabpfn_comparison_per_dataset.csv"
SRC_PIVOT = ROOT / "bench/AOM_v0/publication/tables/master_pivot.csv"
SRC_PIVOT_MK = ROOT / "bench/AOM_v0/Multi-kernel/publication/tables/master_pivot.csv"
SRC_NICON = ROOT / "bench/nicon_v2/publication/tables/full_comparison/long_per_dataset.csv"


# -------------------- Rank helpers (fallback when scipy missing) --------------------

def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank tie handling, mimicking scipy.stats.rankdata."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_a = a[order]
    n = len(a)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    if HAVE_SCIPY:
        r, _ = spearmanr(x, y)
        return float(r) if r == r else 0.0
    rx = _rankdata(np.asarray(x))
    ry = _rankdata(np.asarray(y))
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall(x, y):
    if HAVE_SCIPY:
        t, _ = kendalltau(x, y)
        return float(t) if t == t else 0.0
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    c = d = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]; dy = y[i] - y[j]
            s = dx * dy
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    tot = n * (n - 1) / 2
    return (c - d) / tot if tot else 0.0


# -------------------- Loading and normalization --------------------

def load_long() -> pd.DataFrame:
    frames = []

    # AOM tabpfn_comparison: long, regression. RMSEP column.
    df = pd.read_csv(SRC_AOM, low_memory=False)
    df = df[df["task"] == "regression"].copy()
    keep = df.dropna(subset=["RMSEP", "model", "dataset"])
    long = pd.DataFrame({
        "source": "AOM_v0/tabpfn_comparison_per_dataset",
        "model": keep["model"].astype(str),
        "dataset_group": keep["database_name"].astype(str),
        "dataset": keep["dataset"].astype(str),
        "task": "regression",
        "metric_name": "RMSEP",
        "metric_value": keep["RMSEP"].astype(float),
        "lower_is_better": True,
    })
    frames.append(long)

    # AOM Multi-kernel: skip if identical
    try:
        if SRC_AOM_MK.exists():
            same = SRC_AOM_MK.read_bytes() == SRC_AOM.read_bytes()
            if not same:
                df2 = pd.read_csv(SRC_AOM_MK, low_memory=False)
                df2 = df2[df2["task"] == "regression"].copy()
                k2 = df2.dropna(subset=["RMSEP", "model", "dataset"])
                frames.append(pd.DataFrame({
                    "source": "AOM_v0_Multi-kernel/tabpfn_comparison_per_dataset",
                    "model": k2["model"].astype(str),
                    "dataset_group": k2["database_name"].astype(str),
                    "dataset": k2["dataset"].astype(str),
                    "task": "regression",
                    "metric_name": "RMSEP",
                    "metric_value": k2["RMSEP"].astype(float),
                    "lower_is_better": True,
                }))
    except Exception:
        pass

    # Master pivot: wide -> long, paper baselines
    for path, src in [(SRC_PIVOT, "AOM_v0/master_pivot"),
                      (SRC_PIVOT_MK, "AOM_v0_Multi-kernel/master_pivot")]:
        if not path.exists():
            continue
        if path is SRC_PIVOT_MK and SRC_PIVOT.exists() and path.read_bytes() == SRC_PIVOT.read_bytes():
            continue
        wide = pd.read_csv(path)
        id_cols = [c for c in ("database_name", "dataset") if c in wide.columns]
        model_cols = [c for c in wide.columns if c not in id_cols]
        m = wide.melt(id_vars=id_cols, value_vars=model_cols,
                      var_name="model", value_name="RMSEP")
        m = m.dropna(subset=["RMSEP"])
        frames.append(pd.DataFrame({
            "source": src,
            "model": m["model"].astype(str),
            "dataset_group": m.get("database_name", pd.Series([""] * len(m))).astype(str),
            "dataset": m["dataset"].astype(str),
            "task": "regression",
            "metric_name": "RMSEP",
            "metric_value": m["RMSEP"].astype(float),
            "lower_is_better": True,
        }))

    # nicon long
    if SRC_NICON.exists():
        nic = pd.read_csv(SRC_NICON)
        nic = nic.dropna(subset=["rmsep"])
        frames.append(pd.DataFrame({
            "source": "nicon_v2/long_per_dataset",
            "model": nic["model_class"].astype(str),
            "dataset_group": nic.get("database_name", pd.Series([""] * len(nic))).astype(str),
            "dataset": nic["dataset"].astype(str),
            "task": "regression",
            "metric_name": "RMSEP",
            "metric_value": nic["rmsep"].astype(float),
            "lower_is_better": True,
        }))

    out = pd.concat(frames, ignore_index=True)
    return out


# -------------------- Core matrix --------------------

def build_core_matrix(long: pd.DataFrame):
    """Core 57 = intersection of datasets covered by AOM models with coverage>=57.

    Build matrix from AOM source only, restricted to models that fully cover the
    intersection. RMSEP per (model, dataset) is the median across duplicates.
    """
    aom = long[long["source"].str.startswith("AOM_v0/tabpfn_comparison")].copy()
    cov = aom.groupby("model")["dataset"].nunique().sort_values(ascending=False)
    eligible_models = cov[cov >= 57].index.tolist()
    sub = aom[aom["model"].isin(eligible_models)]

    sets = [set(sub[sub["model"] == m]["dataset"].unique()) for m in eligible_models]
    if not sets:
        raise RuntimeError("No eligible models with >=57 datasets")
    inter = set.intersection(*sets)
    core_datasets = sorted(inter)

    sub2 = sub[sub["dataset"].isin(core_datasets)]
    matrix = (sub2.groupby(["model", "dataset"])["metric_value"]
              .median().unstack("dataset"))
    matrix = matrix.reindex(columns=core_datasets)
    matrix = matrix.dropna(axis=0, how="any")
    return matrix, core_datasets, cov


# -------------------- Score transformation --------------------

def transform_scores(matrix: pd.DataFrame) -> pd.DataFrame:
    """log(RMSEP) then z-score per dataset across models; lower is better."""
    log_m = np.log(matrix.clip(lower=1e-12))
    mu = log_m.mean(axis=0)
    sd = log_m.std(axis=0).replace(0, 1.0)
    z = (log_m - mu) / sd
    return z


# -------------------- Subset evaluation --------------------

def pairwise_sign_agreement(a: np.ndarray, b: np.ndarray, tol: float = 1e-3) -> float:
    """Fraction of model pairs with same sign of difference (within tolerance)."""
    n = len(a)
    if n < 2:
        return 1.0
    da = a[:, None] - a[None, :]
    db = b[:, None] - b[None, :]
    sa = np.where(np.abs(da) <= tol, 0, np.sign(da))
    sb = np.where(np.abs(db) <= tol, 0, np.sign(db))
    iu = np.triu_indices(n, 1)
    return float(np.mean(sa[iu] == sb[iu]))


def evaluate_subset(z: pd.DataFrame, subset_cols, full_mean: np.ndarray):
    sub_mean = z[list(subset_cols)].mean(axis=1).values
    sp = spearman(sub_mean, full_mean)
    kt = kendall(sub_mean, full_mean)
    pa = pairwise_sign_agreement(sub_mean, full_mean, tol=1e-3)
    rank_full = _rankdata(full_mean)
    rank_sub = _rankdata(sub_mean)
    rank_mae = float(np.mean(np.abs(rank_full - rank_sub)))
    agg_mae = float(np.mean(np.abs(sub_mean - full_mean)))
    composite = 0.4 * sp + 0.3 * kt + 0.3 * pa - 0.05 * agg_mae
    return {
        "spearman": sp, "kendall": kt, "pairwise_agreement": pa,
        "rank_mae": rank_mae, "agg_mae": agg_mae, "composite": composite,
    }


def greedy_forward(z: pd.DataFrame, max_size: int):
    """Greedy: start empty; at each step add the dataset that maximizes composite."""
    full_mean = z.mean(axis=1).values
    all_ds = list(z.columns)
    chosen: list[str] = []
    rows = []
    while len(chosen) < max_size:
        best = None
        best_score = -1e18
        for c in all_ds:
            if c in chosen:
                continue
            cand = chosen + [c]
            s = evaluate_subset(z, cand, full_mean)
            if s["composite"] > best_score:
                best_score = s["composite"]; best = c; best_metrics = s
        chosen.append(best)
        rec = {"size": len(chosen), "added": best, "subset": list(chosen)}
        rec.update(best_metrics)
        rows.append(rec)
    return rows, full_mean


def random_baselines(z: pd.DataFrame, sizes, full_mean, n_trials: int = 200):
    rng = np.random.RandomState(SEED)
    cols = np.array(z.columns)
    out = []
    for sz in sizes:
        sps, kts, pas, rmaes = [], [], [], []
        for _ in range(n_trials):
            idx = rng.choice(len(cols), size=sz, replace=False)
            s = evaluate_subset(z, cols[idx].tolist(), full_mean)
            sps.append(s["spearman"]); kts.append(s["kendall"])
            pas.append(s["pairwise_agreement"]); rmaes.append(s["rank_mae"])
        out.append({
            "size": sz, "n_trials": n_trials,
            "spearman_mean": float(np.mean(sps)), "spearman_p05": float(np.percentile(sps, 5)),
            "kendall_mean": float(np.mean(kts)), "kendall_p05": float(np.percentile(kts, 5)),
            "pairwise_mean": float(np.mean(pas)), "pairwise_p05": float(np.percentile(pas, 5)),
            "rank_mae_mean": float(np.mean(rmaes)),
        })
    return out


def bootstrap_models(z: pd.DataFrame, subset_cols, n_boot: int = 400):
    rng = np.random.RandomState(SEED + 7)
    n = z.shape[0]
    sps, kts, pas = [], [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        zb = z.iloc[idx]
        full_mean = zb.mean(axis=1).values
        sub_mean = zb[list(subset_cols)].mean(axis=1).values
        sps.append(spearman(sub_mean, full_mean))
        kts.append(kendall(sub_mean, full_mean))
        pas.append(pairwise_sign_agreement(sub_mean, full_mean, tol=1e-3))
    return {
        "spearman_mean": float(np.mean(sps)),
        "spearman_p05": float(np.percentile(sps, 5)),
        "spearman_p95": float(np.percentile(sps, 95)),
        "kendall_mean": float(np.mean(kts)),
        "kendall_p05": float(np.percentile(kts, 5)),
        "pairwise_mean": float(np.mean(pas)),
        "pairwise_p05": float(np.percentile(pas, 5)),
        "pairwise_p95": float(np.percentile(pas, 95)),
        "n_boot": n_boot,
    }


# -------------------- Main --------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    long = load_long()
    long_path = OUT / "all_scores_long.csv"
    long.to_csv(long_path, index=False)

    matrix, core_ds, coverage = build_core_matrix(long)
    matrix.to_csv(OUT / "model_dataset_matrix.csv")

    coverage_df = coverage.rename("n_datasets").reset_index()
    coverage_df.to_csv(OUT / "model_coverage.csv", index=False)

    # Dataset coverage across all sources
    core_set = set(core_ds)
    grp = (long.groupby("dataset")
           .agg(n_models_all_sources=("model", "nunique"),
                n_sources=("source", "nunique"),
                dataset_group=("dataset_group",
                               lambda s: next((v for v in s if isinstance(v, str) and v), "")))
           .reset_index())
    grp["in_core_57"] = grp["dataset"].isin(core_set)
    grp = grp.sort_values(["in_core_57", "n_models_all_sources"],
                          ascending=[False, False])
    grp.to_csv(OUT / "dataset_coverage.csv", index=False)

    z = transform_scores(matrix)
    z.to_csv(OUT / "model_dataset_zscores.csv")

    n_core = len(core_ds)
    max_size = min(30, n_core)
    if max_size < 30:
        max_size = n_core

    greedy_rows, full_mean = greedy_forward(z, max_size)
    greedy_df = pd.DataFrame(greedy_rows)
    greedy_df["subset_str"] = greedy_df["subset"].apply(lambda L: "|".join(L))
    greedy_df.drop(columns=["subset"]).to_csv(OUT / "greedy_progression.csv", index=False)

    sizes = [r["size"] for r in greedy_rows if r["size"] >= 3]
    rand = random_baselines(z, sizes, full_mean, n_trials=200)
    rand_df = pd.DataFrame(rand)
    rand_df.to_csv(OUT / "random_baselines.csv", index=False)

    # Bootstrap CI for each greedy size >=3
    boot_rows = []
    for r in greedy_rows:
        if r["size"] < 3:
            continue
        b = bootstrap_models(z, r["subset"], n_boot=300)
        b["size"] = r["size"]
        boot_rows.append(b)
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(OUT / "bootstrap_ci.csv", index=False)

    # Joined user-friendly search results (greedy >=3 with bootstrap + random)
    greedy_df3 = greedy_df[greedy_df["size"] >= 3].copy()
    g_keep = greedy_df3[["size", "added", "spearman", "kendall",
                         "pairwise_agreement", "rank_mae", "agg_mae",
                         "composite", "subset_str"]].rename(columns={
        "spearman": "greedy_spearman",
        "kendall": "greedy_kendall",
        "pairwise_agreement": "greedy_pairwise",
        "rank_mae": "greedy_rank_mae",
        "agg_mae": "greedy_agg_mae",
        "composite": "greedy_composite",
        "subset_str": "subset",
    })
    b_keep = boot_df.rename(columns={
        "spearman_mean": "boot_spearman_mean",
        "spearman_p05": "boot_spearman_p05",
        "spearman_p95": "boot_spearman_p95",
        "kendall_mean": "boot_kendall_mean",
        "kendall_p05": "boot_kendall_p05",
        "pairwise_mean": "boot_pairwise_mean",
        "pairwise_p05": "boot_pairwise_p05",
        "pairwise_p95": "boot_pairwise_p95",
    })[["size", "boot_spearman_mean", "boot_spearman_p05", "boot_spearman_p95",
        "boot_kendall_mean", "boot_kendall_p05",
        "boot_pairwise_mean", "boot_pairwise_p05", "boot_pairwise_p95"]]
    r_keep = rand_df.rename(columns={
        "spearman_mean": "random_spearman_mean",
        "spearman_p05": "random_spearman_p05",
        "kendall_mean": "random_kendall_mean",
        "kendall_p05": "random_kendall_p05",
        "pairwise_mean": "random_pairwise_mean",
        "pairwise_p05": "random_pairwise_p05",
        "rank_mae_mean": "random_rank_mae_mean",
        "n_trials": "random_n_trials",
    })
    search = g_keep.merge(b_keep, on="size", how="left").merge(r_keep, on="size", how="left")
    search.to_csv(OUT / "subset_search_results.csv", index=False)

    # Selection
    th = {"spearman": 0.98, "kendall": 0.95, "pairwise_agreement": 0.97,
          "boot_spearman_p05": 0.95, "boot_pairwise_p05": 0.95}
    chosen = None
    chosen_reason = ""
    for r, b in zip([rr for rr in greedy_rows if rr["size"] >= 3], boot_rows):
        if (r["spearman"] >= th["spearman"]
                and r["kendall"] >= th["kendall"]
                and r["pairwise_agreement"] >= th["pairwise_agreement"]
                and b["spearman_p05"] >= th["boot_spearman_p05"]
                and b["pairwise_p05"] >= th["boot_pairwise_p05"]):
            chosen = (r, b)
            chosen_reason = "Tous les seuils sont satisfaits."
            break

    if chosen is None:
        # Plateau: smallest within 1% of best composite
        best_c = max(r["composite"] for r in greedy_rows)
        plateau = [r for r in greedy_rows if r["composite"] >= best_c - 0.01 * abs(best_c) and r["size"] >= 3]
        plateau.sort(key=lambda r: r["size"])
        r = plateau[0]
        b = next(bb for bb in boot_rows if bb["size"] == r["size"])
        chosen = (r, b)
        chosen_reason = ("Aucun sous-ensemble n'atteint l'ensemble complet des seuils ; "
                         "selection sur le plus petit du plateau (a 1% du meilleur composite).")

    sel_r, sel_b = chosen

    # Conservative alternative: smallest greedy size whose bootstrap kendall_p05 >= 0.95
    # in addition to existing thresholds.
    conservative = None
    for r, b in zip([rr for rr in greedy_rows if rr["size"] >= 3], boot_rows):
        if (r["spearman"] >= th["spearman"]
                and r["kendall"] >= th["kendall"]
                and r["pairwise_agreement"] >= th["pairwise_agreement"]
                and b["spearman_p05"] >= th["boot_spearman_p05"]
                and b["pairwise_p05"] >= th["boot_pairwise_p05"]
                and b["kendall_p05"] >= 0.95):
            conservative = {
                "size": int(r["size"]),
                "datasets": list(r["subset"]),
                "metrics_full_vs_subset": {
                    "spearman": r["spearman"],
                    "kendall": r["kendall"],
                    "pairwise_agreement": r["pairwise_agreement"],
                    "rank_mae": r["rank_mae"],
                    "agg_mae": r["agg_mae"],
                    "composite": r["composite"],
                },
                "bootstrap": b,
                "extra_threshold": {"boot_kendall_p05": 0.95},
            }
            break

    selected = {
        "seed": SEED,
        "n_core_datasets": n_core,
        "n_models_core": int(matrix.shape[0]),
        "selected_size": int(sel_r["size"]),
        "selected_datasets": list(sel_r["subset"]),
        "metrics_full_vs_subset": {
            "spearman": sel_r["spearman"],
            "kendall": sel_r["kendall"],
            "pairwise_agreement": sel_r["pairwise_agreement"],
            "rank_mae": sel_r["rank_mae"],
            "agg_mae": sel_r["agg_mae"],
            "composite": sel_r["composite"],
        },
        "bootstrap": sel_b,
        "thresholds": th,
        "selection_reason": chosen_reason,
        "conservative_alternative": conservative,
    }
    (OUT / "selected_subset.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False))

    # Report (FR)
    rand_for_size = next((rr for rr in rand if rr["size"] == sel_r["size"]), None)
    lines = []
    lines.append("# Rapport - Selection d'un sous-ensemble representatif (TabPFN-paper, regression)\n")
    lines.append("## Sources de donnees\n")
    lines.append("- `bench/AOM_v0/publication/tables/tabpfn_comparison_per_dataset.csv` (matrice principale, 7298 lignes, regression).")
    lines.append("- `bench/AOM_v0/Multi-kernel/publication/tables/tabpfn_comparison_per_dataset.csv` : doublon binaire du fichier ci-dessus, ignore.")
    lines.append("- `bench/AOM_v0/publication/tables/master_pivot.csv` et la version Multi-kernel : baselines de l'article (61 lignes, valeurs manquantes), incluses dans `all_scores_long.csv` et la couverture, mais pas dans la matrice de selection.")
    lines.append("- `bench/nicon_v2/publication/tables/full_comparison/long_per_dataset.csv` : 61 datasets, couverture incomplete, idem ci-dessus.\n")

    lines.append("## Pourquoi 57 et non 59 / 61 ?\n")
    lines.append(f"Le coeur d'analyse est l'intersection des datasets couverts par les modeles AOM avec couverture >= 57. Cette intersection compte **{n_core} datasets**. Les datasets supplementaires presents dans `master_pivot.csv` ou `nicon_v2` (jusqu'a 61) ne sont pas couverts par tous les modeles -- les inclure introduirait des NaN ou forcerait l'imputation, ce qui biaiserait les agregats.\n")

    lines.append("## Methode\n")
    lines.append("1. Normalisation longue (`all_scores_long.csv`).")
    lines.append(f"2. Construction de la matrice modele x dataset sur les modeles AOM couvrant les {n_core} datasets ({matrix.shape[0]} modeles retenus).")
    lines.append("3. Transformation `log(RMSEP)` puis z-score par dataset (suppression des differences d'echelle).")
    lines.append("4. Selection avant gloutonne (sizes 3..30) avec score composite : 0.4*Spearman + 0.3*Kendall + 0.3*accord par paires (tolerance 1e-3) - 0.05*MAE agrege.")
    lines.append("5. Baselines aleatoires (200 tirages par taille) et bootstrap par modeles (300 reechantillonnages) pour les IC.")
    lines.append("6. Selection automatique : plus petit sous-ensemble respectant Spearman>=0.98, Kendall>=0.95, accord>=0.97, IC bootstrap p05 Spearman>=0.95 et p05 accord>=0.95. Sinon, plus petit element du plateau a 1% du meilleur composite.\n")
    lines.append("L'analyse principale est volontairement limitee a la regression : le coeur 57 est defini par des scores RMSEP comparables. Les resultats classification disponibles dans certains sous-repertoires utilisent d'autres metriques et une couverture insuffisante, donc ils ne sont pas melanges a cette selection.\n")

    lines.append("## Recommandation\n")
    lines.append(f"- Taille selectionnee : **{sel_r['size']}** datasets.")
    lines.append(f"- Spearman = {sel_r['spearman']:.4f}, Kendall = {sel_r['kendall']:.4f}, accord par paires = {sel_r['pairwise_agreement']:.4f}.")
    lines.append(f"- Bootstrap : Spearman p05 = {sel_b['spearman_p05']:.4f}, accord p05 = {sel_b['pairwise_p05']:.4f}.")
    if rand_for_size:
        lines.append(f"- Comparaison aleatoire (meme taille) : Spearman moyen {rand_for_size['spearman_mean']:.4f}, accord moyen {rand_for_size['pairwise_mean']:.4f}.")
    lines.append("- Datasets retenus :")
    for d in sel_r["subset"]:
        lines.append(f"  - `{d}`")
    lines.append("")

    lines.append("## Confiance et alternative conservatrice\n")
    lines.append(f"- Justification de la selection : {chosen_reason}")
    lines.append("- Le sous-ensemble recommande de **{n} datasets** satisfait les seuils **directs** (Spearman >= 0.98, Kendall >= 0.95, accord par paires >= 0.97) ainsi que les seuils bootstrap p05 sur Spearman et l'accord. En revanche, le critere **plus strict** exigeant aussi `bootstrap kendall_p05 >= 0.95` est plus exigeant : la borne basse (p05) du Kendall sur le bootstrap des modeles vaut {kp05:.4f} pour la taille recommandee.".format(n=sel_r["size"], kp05=sel_b["kendall_p05"]))
    if conservative is not None:
        lines.append(f"- **Alternative conservatrice** (plus petite taille satisfaisant aussi `kendall_p05 >= 0.95`) : **{conservative['size']} datasets** "
                     f"(Spearman {conservative['metrics_full_vs_subset']['spearman']:.4f}, "
                     f"Kendall {conservative['metrics_full_vs_subset']['kendall']:.4f}, "
                     f"accord {conservative['metrics_full_vs_subset']['pairwise_agreement']:.4f}, "
                     f"bootstrap kendall_p05 {conservative['bootstrap']['kendall_p05']:.4f}).")
    else:
        lines.append("- **Alternative conservatrice** : aucune taille testee n'atteint `kendall_p05 >= 0.95` en plus des autres seuils ; cf. `bootstrap_ci.csv` pour la progression.")
    lines.append("- Les IC bootstrap sont calcules en reechantillonnant les modeles ; ils mesurent la robustesse au pool de modeles. Nous ne surestimons pas la confiance : la stabilite du Kendall (statistique discrete a faible effectif) est plus difficile a garantir.\n")

    lines.append("## Limites\n")
    lines.append("- L'analyse est restreinte aux modeles ayant >=57 datasets dans la table AOM ; modeles a couverture partielle exclus pour eviter l'imputation.")
    lines.append("- Les datasets hors intersection (jusqu'a 61) ne sont pas evalues -- leur ajout depend de runs supplementaires.")
    lines.append("- Le score composite pondere arbitrairement Spearman/Kendall/accord ; un autre arbitrage pourrait modifier marginalement le classement entre tailles voisines.\n")

    lines.append("## Reproduction\n")
    lines.append("```\npython3 bench/Subset_analysis/analyze_subset.py\n```\n")

    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    readme = []
    readme.append("# bench/Subset_analysis\n")
    readme.append("Analyse reproductible pour selectionner un sous-ensemble representatif des 57 datasets de regression communs (papier TabPFN).\n")
    readme.append("L'analyse principale est regression-only : le coeur 57 repose sur des RMSEP comparables. Les resultats classification disponibles ailleurs dans `bench` ont des metriques et une couverture differentes, donc ils ne sont pas melanges a cette selection.\n")
    readme.append("## Fichiers generes\n")
    readme.append("- `all_scores_long.csv` : table longue normalisee (toutes sources).")
    readme.append("- `model_coverage.csv` : couverture par modele dans la table AOM.")
    readme.append("- `model_dataset_matrix.csv` : matrice RMSEP modele x dataset (coeur 57).")
    readme.append("- `model_dataset_zscores.csv` : z-scores par dataset apres log.")
    readme.append("- `dataset_coverage.csv` : par dataset, nombre de modeles toutes sources, nombre de sources, appartenance au coeur 57, et `dataset_group`.")
    readme.append("- `greedy_progression.csv` : progression de la selection avant gloutonne.")
    readme.append("- `random_baselines.csv` : statistiques de sous-ensembles aleatoires.")
    readme.append("- `bootstrap_ci.csv` : IC bootstrap (sur les modeles) pour chaque taille.")
    readme.append("- `subset_search_results.csv` : table jointe lisible (greedy, bootstrap CI, baselines aleatoires) pour chaque taille >=3.")
    readme.append("- `selected_subset.json` : sous-ensemble retenu, metriques, et `conservative_alternative` (plus petite taille satisfaisant aussi `bootstrap kendall_p05 >= 0.95`, ou null).")
    readme.append("- `SYNTHESE_TECHNIQUE.md` : synthese explicative des techniques statistiques avec figures illustratives.")
    readme.append("- `make_visualizations.py` : genere les figures dans `figures/` a partir des sorties CSV/JSON.")
    readme.append("- `REPORT.md` : rapport detaille en francais.\n")
    readme.append("## Execution\n")
    readme.append("```\npython3 bench/Subset_analysis/analyze_subset.py\n```\n")
    readme.append("Figures et synthese :\n")
    readme.append("```\npython3 bench/Subset_analysis/make_visualizations.py\n```\n")
    readme.append(f"Seed deterministe : {SEED}.\n")
    readme.append("## Note sur la confiance\n")
    readme.append(f"Le sous-ensemble recommande ({sel_r['size']} datasets) satisfait les seuils **directs** (Spearman/Kendall/accord par paires) ainsi que les bornes bootstrap p05 sur Spearman et accord. Le critere **plus strict** ajoutant `bootstrap kendall_p05 >= 0.95` selectionne plutot l'**alternative conservatrice** decrite dans `selected_subset.json` (champ `conservative_alternative`). Nous ne surestimons pas la confiance : sur cette dimension Kendall, la borne basse au p05 reste sous le seuil pour la taille recommandee.\n")
    (OUT / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"[OK] Core datasets: {n_core}; eligible models: {matrix.shape[0]}")
    print(f"[OK] Selected size: {sel_r['size']}")
    print(f"     spearman={sel_r['spearman']:.4f} kendall={sel_r['kendall']:.4f} "
          f"pairwise={sel_r['pairwise_agreement']:.4f}")
    print(f"     bootstrap spearman_p05={sel_b['spearman_p05']:.4f} pairwise_p05={sel_b['pairwise_p05']:.4f}")
    print(f"[OK] Outputs in {OUT}")


if __name__ == "__main__":
    main()
