"""Build comprehensive benchmark dataviz dashboard.

Reads:
  - bench/benchmark_master_results.csv (MASTER — 24K rows, 451 model_name,
    112 source_runs, 85 datasets, 14 model_classes)
  - bench/scenarios/runs/<preset>_full57_seed0/results.csv (Phase 2-5 production)

Produces:
  - bench/scenarios/dataviz/dashboard_data.json

The HTML page (dashboard.html) consumes this JSON.

Sections produced:
  preset_*:        per-preset leaderboards (production runs only)
  master_*:        aggregations across ALL observed + reference_paper rows
  datasets:        per-dataset best-overall
  head_to_head:    8x8 win matrix on best_current
  preproc_pairs:   preprocessing influence
  failures:        failure patterns
  tabpfn_paper:    legacy TabPFN paper rows

Owner: Agent C. Analysis of locked data only.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BENCH = Path(__file__).resolve().parents[2]
RUNS = BENCH / "scenarios" / "runs"
MASTER = BENCH / "benchmark_master_results.csv"
OUT_JSON = BENCH / "scenarios" / "dataviz" / "dashboard_data.json"

PRESETS = (
    ("fast_reliable", "fast_reliable_full57_seed0"),
    ("strong_practical", "strong_practical_full57_seed0"),
    ("best_current", "best_current_full57_seed0"),
    ("exhaustive_research", "exhaustive_research_full57_seed0"),
)


# Map canonical_name → (family, friendly_short_name, color_hex)
FAMILIES: dict[str, tuple[str, str, str]] = {
    "AOM-PLS-compact-numpy": ("AOM-PLS", "AOM-PLS", "#1f77b4"),
    "AOM-default-nipals-adjoint-numpy": ("AOM-PLS", "AOM-PLS-default", "#3d8bcc"),
    "POP-PLS-compact-numpy": ("AOM-PLS", "POP-PLS", "#5fa8e3"),
    "AOMRidge-global-compact-none": ("AOM-Ridge", "AR-glb-none", "#d62728"),
    "AOMRidge-global-compact-snv": ("AOM-Ridge", "AR-glb-snv", "#e3493b"),
    "AOMRidge-Local-compact-knn50": ("AOM-Ridge", "AR-loc-50", "#e87e6f"),
    "AOMRidge-Local-compact-knn-sweep": ("AOM-Ridge", "AR-loc-sweep", "#ec9d8f"),
    "AOMRidge-MultiBranchMKL-compact-shrink03": ("AOM-Ridge", "AR-MBMKL", "#fcc8b9"),
    "AOMRidge-Blender-headline-spxy3": ("AOM-Ridge", "AR-Blender", "#b0140e"),
    "AOMRidge-AutoSelect-headline-spxy3": ("AOM-Ridge", "AR-AutoSelect", "#92100a"),
    "AOMRidgePLSCV-compact-with-fck": ("AOM-Ridge", "AR-PLSCV-fck", "#7a0a06"),
    "ASLS-AOM-compact-cv5-numpy": ("ASLS", "ASLS-AOM", "#9467bd"),
    "ASLS-FCK-PLS-static": ("ASLS", "ASLS-FCK-PLS", "#a988c9"),
    "Ridge-tuned-cv5": ("Ridge", "Ridge-cv5", "#2ca02c"),
    "PLS-tuned-cv5": ("PLS", "PLS-cv5", "#8c564b"),
    "AOMMultiView-MeanEnsemble4-fixed": ("MoE/Stack", "AOMMean-Ens4", "#ff7f0e"),
    "moe-preproc-soft-pls-compact": ("MoE/Stack", "MoE-preproc", "#ff9d3a"),
    "AdaptiveSuperLearner-recipe-nnls": ("MoE/Stack", "ASL-recipe", "#ffb066"),
    "AdaptiveSuperLearner-bigN-guarded": ("MoE/Stack", "ASL-bigN", "#ffc391"),
    "Stack-Ridge-PLS-V1c": ("MoE/Stack", "Stack-Ridge-PLS", "#ffd7bd"),
    "FCK-AOMPLS-static": ("FCK", "FCK-AOMPLS", "#17becf"),
    "FCK-PLS-static": ("FCK", "FCK-PLS", "#3fcedd"),
    "Concat-SNV-FCK-AOMPLS-static": ("FCK", "Concat-SNV-FCK", "#67dde6"),
    "AOMPLS-compact-with-fck-full57": ("FCK", "AOMPLS-fck", "#8ee9ef"),
    "FCKResidual-AOMPLS-teacher": ("FCK", "FCK-Residual", "#b5f1f4"),
    "V2L-Residual-AOMPLS": ("Residual NN", "V2L-Residual", "#bcbd22"),
    "V2L-Boost-AOMPLS": ("Residual NN", "V2L-Boost", "#cecf3d"),
    "MKM-reml-default": ("Multi-Kernel", "MKM-reml", "#7f7f7f"),
    "mkR-softmax-cv-default": ("Multi-Kernel", "mkR-softmax", "#9a9a9a"),
    "TabPFN-Raw": ("TabPFN", "TabPFN-Raw", "#e377c2"),
    "TabPFN-opt": ("TabPFN", "TabPFN-opt", "#eb95cf"),
    "TabPFN-HPO-preprocessing": ("TabPFN", "TabPFN-HPO", "#f2b4dc"),
    "paper-CNN-reference": ("Paper-ref", "paper-CNN", "#aaaaaa"),
    "paper-CatBoost-reference": ("Paper-ref", "paper-CatBoost", "#bbbbbb"),
}

# model_class → color for the master view
MODEL_CLASS_COLORS = {
    "AOM-PLS": "#1f77b4",
    "AOM-Ridge": "#d62728",
    "PLS": "#8c564b",
    "NICON/CNN": "#bcbd22",
    "Ridge": "#2ca02c",
    "Meta-selector/MoE": "#ff7f0e",
    "Hybrid CNN+AOM": "#17becf",
    "POP-PLS": "#5fa8e3",
    "Hybrid CNN+linear": "#bcbd22",
    "Multi-kernel ridge": "#7f7f7f",
    "TabPFN": "#e377c2",
    "CatBoost": "#9467bd",
    "FCK-PLS": "#17becf",
    "Stacked NN+Linear": "#ffaa00",
}


def family_of(name: str) -> tuple[str, str, str]:
    return FAMILIES.get(name, ("Other", name[:15], "#cccccc"))


def safe_float(v: str | None) -> float | None:
    if v is None or v == "":
        return None
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def safe_int(v: str | None) -> int | None:
    f = safe_float(v)
    return int(f) if f is not None else None


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = int(q * (len(s) - 1))
    return s[idx]


def load_preset_rows(workspace: str) -> list[dict[str, str]]:
    path = RUNS / workspace / "results.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def load_master_rows() -> list[dict[str, str]]:
    if not MASTER.exists():
        return []
    with MASTER.open() as f:
        return list(csv.DictReader(f))


def build_preset_section(preset_name: str, workspace: str) -> dict:
    rows = load_preset_rows(workspace)
    if not rows:
        return {"name": preset_name, "n_planned": 0, "candidates": [], "available": False}

    by_cand: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_cand[r["canonical_name"]].append(r)

    candidates_out = []
    for cand, cand_rows in by_cand.items():
        rmseps = [safe_float(r.get("rmsep")) for r in cand_rows if r["status"] == "ok"]
        rmseps = [x for x in rmseps if x is not None]
        fit_times = [safe_float(r.get("fit_time_s")) for r in cand_rows if r["status"] == "ok"]
        fit_times = [x for x in fit_times if x is not None]
        family, short, color = family_of(cand)
        candidates_out.append({
            "canonical_name": cand,
            "short_name": short,
            "family": family,
            "color": color,
            "n_ok": sum(1 for r in cand_rows if r["status"] == "ok"),
            "n_failed": sum(1 for r in cand_rows if r["status"] == "failed"),
            "n_failed_terminal": sum(1 for r in cand_rows if r["status"] == "failed_terminal"),
            "median_rmsep": median(rmseps),
            "q25_rmsep": quantile(rmseps, 0.25),
            "q75_rmsep": quantile(rmseps, 0.75),
            "q90_rmsep": quantile(rmseps, 0.90),
            "max_rmsep": max(rmseps) if rmseps else None,
            "median_fit_s": median(fit_times),
            "p90_fit_s": quantile(fit_times, 0.90),
            "max_fit_s": max(fit_times) if fit_times else None,
        })

    candidates_out.sort(key=lambda c: c["median_rmsep"] if c["median_rmsep"] is not None else float("inf"))

    return {
        "name": preset_name,
        "workspace": workspace,
        "n_planned": len(rows),
        "n_ok": sum(1 for r in rows if r["status"] == "ok"),
        "n_failed": sum(1 for r in rows if r["status"] == "failed"),
        "n_failed_terminal": sum(1 for r in rows if r["status"] == "failed_terminal"),
        "candidates": candidates_out,
        "available": True,
    }


def build_dataset_section(presets: dict) -> list[dict]:
    """Per-dataset analysis across preset workspaces."""
    ds_data: dict[str, dict] = {}
    for preset_name, info in presets.items():
        rows = load_preset_rows(info["workspace"])
        for r in rows:
            if r["status"] != "ok":
                continue
            ds_name = r["dataset"]
            if ds_name not in ds_data:
                ds_data[ds_name] = {
                    "name": ds_name,
                    "n_train": safe_int(r.get("n_train")),
                    "n_test": safe_int(r.get("n_test")),
                    "n_features": safe_int(r.get("n_features")),
                    "candidates": {},
                }
            rmsep = safe_float(r.get("rmsep"))
            if rmsep is None:
                continue
            ds_data[ds_name]["candidates"][r["canonical_name"]] = rmsep

    for ds in ds_data.values():
        if ds["candidates"]:
            best_cand, best_rmsep = min(ds["candidates"].items(), key=lambda x: x[1])
            ds["best_candidate"] = best_cand
            ds["best_rmsep"] = best_rmsep
            ds["worst_rmsep"] = max(ds["candidates"].values())
            family, _, _ = family_of(best_cand)
            ds["best_family"] = family

    return list(ds_data.values())


def build_head_to_head(workspace: str) -> dict:
    rows = load_preset_rows(workspace)
    pair_rmsep: dict[tuple[str, str], float] = {}
    candidates = set()
    for r in rows:
        if r["status"] != "ok":
            continue
        rmsep = safe_float(r.get("rmsep"))
        if rmsep is None:
            continue
        candidates.add(r["canonical_name"])
        pair_rmsep[(r["canonical_name"], r["dataset"])] = rmsep

    candidates = sorted(candidates)
    h2h: dict[str, dict[str, int]] = {a: {} for a in candidates}
    for a in candidates:
        for b in candidates:
            if a == b:
                h2h[a][b] = 0
                continue
            wins = 0
            for ds in {k[1] for k in pair_rmsep if k[0] == a}:
                ra = pair_rmsep.get((a, ds))
                rb = pair_rmsep.get((b, ds))
                if ra is not None and rb is not None and ra < rb:
                    wins += 1
            h2h[a][b] = wins
    return {"workspace": workspace, "candidates": candidates, "wins": h2h}


def build_preprocessing_pairs(workspace: str) -> list[dict]:
    rows = load_preset_rows(workspace)
    pair_rmsep: dict[tuple[str, str], float] = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        rmsep = safe_float(r.get("rmsep"))
        if rmsep is None:
            continue
        pair_rmsep[(r["canonical_name"], r["dataset"])] = rmsep

    pairs = [
        ("AOMRidge-global-compact-none", "AOMRidge-global-compact-snv", "preprocessing: none vs SNV"),
    ]
    out = []
    for a, b, label in pairs:
        deltas = []
        for ds in {k[1] for k in pair_rmsep if k[0] == a}:
            ra = pair_rmsep.get((a, ds))
            rb = pair_rmsep.get((b, ds))
            if ra is None or rb is None:
                continue
            deltas.append({"dataset": ds, "delta": rb - ra})
        if not deltas:
            continue
        delta_vals = [d["delta"] for d in deltas]
        out.append({
            "candidate_a": a,
            "candidate_b": b,
            "label": label,
            "n_datasets": len(deltas),
            "median_delta": median(delta_vals),
            "mean_delta": sum(delta_vals) / len(delta_vals),
            "max_abs_delta": max(abs(d) for d in delta_vals),
            "deltas": deltas,
        })
    return out


def build_failures(presets: dict) -> dict:
    failed_by_ds_cand: dict[str, list[str]] = defaultdict(list)
    timeout_by_ds_cand: dict[str, list[str]] = defaultdict(list)
    workspace = presets["best_current"]["workspace"]
    rows = load_preset_rows(workspace)
    for r in rows:
        if r["status"] == "failed":
            failed_by_ds_cand[r["dataset"]].append(r["canonical_name"])
        elif r["status"] == "failed_terminal":
            timeout_by_ds_cand[r["dataset"]].append(r["canonical_name"])
    return {
        "failed_by_dataset": {k: sorted(v) for k, v in failed_by_ds_cand.items()},
        "timeout_by_dataset": {k: sorted(v) for k, v in timeout_by_ds_cand.items()},
    }


def build_tabpfn_paper(master: list[dict]) -> list[dict]:
    out = []
    for r in master:
        if r.get("record_type") != "reference_paper":
            continue
        mn = (r.get("model_name") or "").strip()
        if "tabpfn" not in mn.lower():
            continue
        rmsep = safe_float(r.get("rmsep"))
        if rmsep is None:
            continue
        out.append({"model_name": mn, "dataset": r.get("dataset", ""), "rmsep": rmsep})
    return out


# ====================================================================
# MASTER aggregations — the big additions
# ====================================================================

def build_master_aggregations(master: list[dict]) -> dict:
    """Aggregate the ENTIRE master CSV (observed + reference_paper).

    Produces:
      - top_models: top N model_name by median rmsep (with >= min_ds datasets)
      - by_model_class: 14 model_classes aggregated
      - by_source_run: 112 source_runs aggregated
      - by_source_family: 9 source families
      - by_maturity: locked vs exploratory comparison
      - all_observations: lightweight list for plotting (down-sampled if huge)
      - per_dataset_best: best (model_name, rmsep) per dataset across all sources
    """
    # Filter: keep observed/paper rows with rmsep AND a real dataset name
    # (some legacy ingests have dataset='dataset' or empty due to parse bugs)
    _DS_BLACKLIST = {"dataset", "runs", "results", ""}
    obs = [
        r for r in master
        if r.get("record_type") in ("observed", "reference_paper")
        and r.get("status", "") in ("ok", "")
        and safe_float(r.get("rmsep")) is not None
        and (r.get("dataset") or "").strip() not in _DS_BLACKLIST
    ]

    # Group by model_name
    by_model_name: dict[str, list[dict]] = defaultdict(list)
    for r in obs:
        mn = (r.get("model_name") or "").strip()
        if mn:
            by_model_name[mn].append(r)

    # Best rmsep per (model_name, dataset) — used for both leaderboard + per-model ranks
    best_per_mn_ds: dict[str, dict[str, float]] = defaultdict(dict)
    for r in obs:
        mn = (r.get("model_name") or "").strip()
        ds = (r.get("dataset") or "").strip()
        if not mn or not ds:
            continue
        v = safe_float(r.get("rmsep"))
        if v is None:
            continue
        if ds not in best_per_mn_ds[mn] or v < best_per_mn_ds[mn][ds]:
            best_per_mn_ds[mn][ds] = v

    # Per-dataset rank table at model_name granularity
    # For each dataset, sort all model_names by their best rmsep → rank
    per_ds_model_ranks: dict[str, dict[str, int]] = defaultdict(dict)
    ds_universe: set[str] = set()
    for mn, ds_map in best_per_mn_ds.items():
        for ds in ds_map:
            ds_universe.add(ds)
    for ds in ds_universe:
        contestants = [(mn, m[ds]) for mn, m in best_per_mn_ds.items() if ds in m]
        contestants.sort(key=lambda x: x[1])
        for rk, (mn, _) in enumerate(contestants, start=1):
            per_ds_model_ranks[ds][mn] = rk

    # Per-model_name aggregation
    top_models = []
    for mn, mn_rows in by_model_name.items():
        if mn not in best_per_mn_ds:
            continue
        best_per_ds = best_per_mn_ds[mn]
        ds_rmseps = list(best_per_ds.values())
        if not ds_rmseps:
            continue
        # Most-common model_class for this model_name
        mclass_counter: dict[str, int] = defaultdict(int)
        for r in mn_rows:
            mclass_counter[r.get("model_class", "")] += 1
        model_class = max(mclass_counter, key=lambda k: mclass_counter[k]) if mclass_counter else ""
        # Source runs covered
        source_runs = sorted({r.get("source_run", "") for r in mn_rows if r.get("source_run")})
        # Maturity tags
        maturities = sorted({r.get("protocol_maturity", "") for r in mn_rows if r.get("protocol_maturity")})
        # Normalized score metrics — built into master CSV (score_ratio_to_dataset_oracle has 100% coverage)
        oracle_ratios = []
        pls_ratios = []
        for r in mn_rows:
            v = safe_float(r.get("score_ratio_to_dataset_oracle"))
            if v is not None and math.isfinite(v):
                oracle_ratios.append(v)
            v = safe_float(r.get("score_ratio_vs_dataset_pls"))
            if v is not None and math.isfinite(v) and v > 0:
                pls_ratios.append(v)
        # Per-dataset ranks for this model
        my_ranks = [per_ds_model_ranks[ds].get(mn) for ds in best_per_ds]
        my_ranks = [r for r in my_ranks if r is not None]
        # Wins (rank 1 on at least 1 dataset)
        wins = sum(1 for r in my_ranks if r == 1)
        top_models.append({
            "model_name": mn,
            "model_class": model_class,
            "n_datasets": len(ds_rmseps),
            "n_observations": len(mn_rows),
            "median_rmsep": median(ds_rmseps),
            "q25_rmsep": quantile(ds_rmseps, 0.25),
            "q75_rmsep": quantile(ds_rmseps, 0.75),
            "q90_rmsep": quantile(ds_rmseps, 0.90),
            "best_rmsep": min(ds_rmseps),
            "median_oracle_ratio": median(oracle_ratios) if oracle_ratios else None,
            "median_pls_ratio": median(pls_ratios) if pls_ratios else None,
            "n_oracle": len(oracle_ratios),
            "median_rank": median(my_ranks) if my_ranks else None,
            "mean_rank": statistics.mean(my_ranks) if my_ranks else None,
            "wins": wins,
            "n_ranks": len(my_ranks),
            "source_runs": source_runs[:10],
            "n_source_runs": len(source_runs),
            "maturities": maturities,
        })
    top_models.sort(key=lambda x: x["median_rmsep"] if x["median_rmsep"] is not None else float("inf"))

    # Per-model_class aggregation (median best-per-dataset across all models in class)
    by_class: dict[str, dict] = defaultdict(lambda: {
        "n_models": 0,
        "n_observations": 0,
        "datasets": set(),
        "best_per_ds_per_model": defaultdict(dict),  # ds → model_name → best rmsep
    })
    for r in obs:
        mc = r.get("model_class", "")
        if not mc:
            continue
        mn = (r.get("model_name") or "").strip()
        v = safe_float(r.get("rmsep"))
        ds = r.get("dataset", "")
        if v is None or not ds:
            continue
        by_class[mc]["n_observations"] += 1
        by_class[mc]["datasets"].add(ds)
        # Track best rmsep per (model_class, dataset, model_name)
        prev = by_class[mc]["best_per_ds_per_model"][ds].get(mn)
        if prev is None or v < prev:
            by_class[mc]["best_per_ds_per_model"][ds][mn] = v

    by_model_class = []
    for mc, info in by_class.items():
        # For each dataset, take the BEST rmsep across all models in this class
        ds_bests = []
        for ds, model_rmseps in info["best_per_ds_per_model"].items():
            ds_bests.append(min(model_rmseps.values()))
        models_in_class = {
            mn for ds_dict in info["best_per_ds_per_model"].values() for mn in ds_dict.keys()
        }
        by_model_class.append({
            "model_class": mc,
            "color": MODEL_CLASS_COLORS.get(mc, "#888888"),
            "n_models": len(models_in_class),
            "n_datasets_covered": len(info["datasets"]),
            "n_observations": info["n_observations"],
            "median_best_rmsep": median(ds_bests),
            "q25_best_rmsep": quantile(ds_bests, 0.25),
            "q75_best_rmsep": quantile(ds_bests, 0.75),
            "q90_best_rmsep": quantile(ds_bests, 0.90),
        })
    by_model_class.sort(key=lambda x: x["median_best_rmsep"] if x["median_best_rmsep"] is not None else float("inf"))

    # Per-source_run aggregation
    by_source: dict[str, dict] = defaultdict(lambda: {
        "n_observations": 0,
        "datasets": set(),
        "model_classes": set(),
        "rmseps": [],
    })
    for r in obs:
        sr = r.get("source_run", "")
        if not sr:
            continue
        v = safe_float(r.get("rmsep"))
        if v is None:
            continue
        by_source[sr]["n_observations"] += 1
        by_source[sr]["datasets"].add(r.get("dataset", ""))
        by_source[sr]["model_classes"].add(r.get("model_class", ""))
        by_source[sr]["rmseps"].append(v)

    by_source_run = []
    for sr, info in by_source.items():
        rmseps = info["rmseps"]
        by_source_run.append({
            "source_run": sr,
            "n_observations": info["n_observations"],
            "n_datasets": len(info["datasets"]),
            "n_model_classes": len(info["model_classes"]),
            "model_classes": sorted(info["model_classes"]),
            "median_rmsep": median(rmseps),
        })
    by_source_run.sort(key=lambda x: -x["n_observations"])

    # Per-source_family aggregation
    by_family: dict[str, dict] = defaultdict(lambda: {
        "n_observations": 0,
        "datasets": set(),
        "rmseps": [],
    })
    for r in obs:
        sf = r.get("source_family", "")
        if not sf:
            continue
        v = safe_float(r.get("rmsep"))
        if v is None:
            continue
        by_family[sf]["n_observations"] += 1
        by_family[sf]["datasets"].add(r.get("dataset", ""))
        by_family[sf]["rmseps"].append(v)

    by_source_family = []
    for sf, info in by_family.items():
        by_source_family.append({
            "source_family": sf,
            "n_observations": info["n_observations"],
            "n_datasets": len(info["datasets"]),
            "median_rmsep": median(info["rmseps"]),
        })
    by_source_family.sort(key=lambda x: -x["n_observations"])

    # Per-maturity aggregation
    by_maturity: dict[str, list[float]] = defaultdict(list)
    for r in obs:
        m = r.get("protocol_maturity", "")
        if not m:
            continue
        v = safe_float(r.get("rmsep"))
        if v is None:
            continue
        by_maturity[m].append(v)
    maturity_summary = [
        {
            "maturity": m,
            "n_observations": len(values),
            "median_rmsep": median(values),
            "q25_rmsep": quantile(values, 0.25),
            "q75_rmsep": quantile(values, 0.75),
        }
        for m, values in by_maturity.items()
    ]
    maturity_summary.sort(key=lambda x: x["maturity"])

    # Per-dataset best-overall (across all models in master)
    ds_best: dict[str, dict] = {}
    for r in obs:
        ds = r.get("dataset", "")
        v = safe_float(r.get("rmsep"))
        if not ds or v is None:
            continue
        n_train = safe_int(r.get("n_train"))
        n_features = safe_int(r.get("n_features"))
        if ds not in ds_best:
            ds_best[ds] = {
                "dataset": ds,
                "n_train": n_train,
                "n_features": n_features,
                "best_rmsep": v,
                "best_model_name": r.get("model_name", ""),
                "best_model_class": r.get("model_class", ""),
                "best_source_run": r.get("source_run", ""),
            }
        else:
            # update best if better, fill metadata if missing
            if v < ds_best[ds]["best_rmsep"]:
                ds_best[ds]["best_rmsep"] = v
                ds_best[ds]["best_model_name"] = r.get("model_name", "")
                ds_best[ds]["best_model_class"] = r.get("model_class", "")
                ds_best[ds]["best_source_run"] = r.get("source_run", "")
            if ds_best[ds]["n_train"] is None and n_train is not None:
                ds_best[ds]["n_train"] = n_train
            if ds_best[ds]["n_features"] is None and n_features is not None:
                ds_best[ds]["n_features"] = n_features
    per_dataset_best = sorted(ds_best.values(), key=lambda x: x["dataset"])

    # All observations (lean) — for scatter / interactive plots
    # Cap to N_max to keep JSON size manageable
    N_max = 6000
    obs_lean = []
    for r in obs:
        v = safe_float(r.get("rmsep"))
        if v is None:
            continue
        obs_lean.append({
            "model_name": (r.get("model_name") or "")[:80],
            "model_class": r.get("model_class", ""),
            "dataset": r.get("dataset", ""),
            "source_run": r.get("source_run", ""),
            "rmsep": v,
            "n_train": safe_int(r.get("n_train")),
            "maturity": r.get("protocol_maturity", ""),
        })
    if len(obs_lean) > N_max:
        # Sample: keep all top-models rows + random sample of rest
        import random
        random.seed(0)
        obs_lean = sorted(obs_lean, key=lambda x: x["rmsep"])[:1000] + random.sample(obs_lean, N_max - 1000)

    # Observations per dataset (FULL detail, every row, for drill-down)
    # Each entry: {model_name, model_class, source_run, seed, rmsep, mae, r2,
    #              fit_time_s, maturity, n_train, n_features, dataset_group}
    per_dataset_obs: dict[str, list[dict]] = defaultdict(list)
    for r in obs:
        ds = r.get("dataset", "")
        if not ds:
            continue
        rmsep = safe_float(r.get("rmsep"))
        if rmsep is None:
            continue
        per_dataset_obs[ds].append({
            "model_name": (r.get("model_name") or "")[:80],
            "model_class": r.get("model_class", ""),
            "variant": (r.get("variant") or "")[:60],
            "source_run": r.get("source_run", ""),
            "source_family": r.get("source_family", ""),
            "seed": r.get("seed", ""),
            "rmsep": rmsep,
            "mae": safe_float(r.get("mae")),
            "r2": safe_float(r.get("r2")),
            "fit_time_s": safe_float(r.get("fit_time_s")),
            "maturity": r.get("protocol_maturity", ""),
            "record_type": r.get("record_type", ""),
            "preprocessing": (r.get("preprocessing_pipeline") or "")[:60],
            "n_components": r.get("n_components", ""),
        })
    # Sort each dataset's observations by rmsep ascending
    for ds_name in per_dataset_obs:
        per_dataset_obs[ds_name].sort(key=lambda x: x["rmsep"])

    # Per-dataset metadata: pick first non-null n_train/n_features per dataset
    per_dataset_meta = {}
    for r in obs:
        ds = r.get("dataset", "")
        if not ds:
            continue
        if ds not in per_dataset_meta:
            per_dataset_meta[ds] = {
                "dataset": ds,
                "n_train": safe_int(r.get("n_train")),
                "n_test": safe_int(r.get("n_test")),
                "n_features": safe_int(r.get("n_features")),
                "dataset_group": r.get("dataset_group", ""),
            }
        else:
            if per_dataset_meta[ds]["n_train"] is None:
                v = safe_int(r.get("n_train"))
                if v is not None:
                    per_dataset_meta[ds]["n_train"] = v
            if per_dataset_meta[ds]["n_features"] is None:
                v = safe_int(r.get("n_features"))
                if v is not None:
                    per_dataset_meta[ds]["n_features"] = v
            if per_dataset_meta[ds]["n_test"] is None:
                v = safe_int(r.get("n_test"))
                if v is not None:
                    per_dataset_meta[ds]["n_test"] = v

    preprocessing = build_preprocessing_master(obs)
    ranks = build_rank_analysis(obs)

    return {
        "top_models": top_models,
        "by_model_class": by_model_class,
        "by_source_run": by_source_run,
        "by_source_family": by_source_family,
        "by_maturity": maturity_summary,
        "per_dataset_best": per_dataset_best,
        "observations_lean": obs_lean,
        "observations_per_dataset": dict(per_dataset_obs),
        "per_dataset_meta": per_dataset_meta,
        "preprocessing": preprocessing,
        "ranks": ranks,
        "n_observations_total": len(obs),
    }


def build_rank_analysis(obs: list[dict]) -> dict:
    """For each model_class, compute its rank on each dataset (1 = best).

    Produces:
      - rank_per_dataset: dict[class] -> [{dataset, rank, rmsep, n_competitors}]
      - mean_ranks: list of {class, mean_rank, median_rank, n_datasets, std_rank}
      - friedman: { chi_squared, df, p_value_approx } across all classes
      - critical_difference: per Nemenyi α=0.05 (q_alpha * sqrt(k(k+1)/6N))

    Each class is collapsed to its best (min rmsep) on each dataset to avoid
    inflating rank distributions with intra-class variance.
    """
    by_ds_cls: dict[str, dict[str, float]] = defaultdict(dict)
    for r in obs:
        rmsep = safe_float(r.get("rmsep"))
        if rmsep is None or not math.isfinite(rmsep):
            continue
        ds = (r.get("dataset") or "").strip()
        cls = (r.get("model_class") or "").strip()
        if not ds or not cls:
            continue
        cur = by_ds_cls[ds].get(cls)
        if cur is None or rmsep < cur:
            by_ds_cls[ds][cls] = rmsep

    # Per-class rank list (only datasets where the class has a score)
    rank_per_class: dict[str, list[dict]] = defaultdict(list)
    valid_datasets_per_class: dict[str, set] = defaultdict(set)
    full_classes_per_dataset: dict[str, list[str]] = {}

    for ds, cls_scores in by_ds_cls.items():
        sorted_cls = sorted(cls_scores.items(), key=lambda x: x[1])
        n = len(sorted_cls)
        # Average ranks for ties
        for rank_idx, (cls, val) in enumerate(sorted_cls, start=1):
            rank_per_class[cls].append({
                "dataset": ds,
                "rank": rank_idx,
                "rmsep": val,
                "n_competitors": n,
            })
            valid_datasets_per_class[cls].add(ds)
        full_classes_per_dataset[ds] = [c for c, _ in sorted_cls]

    # Datasets where ALL N classes have a score (for fair Friedman test)
    all_classes = sorted(rank_per_class.keys())
    complete_datasets = [
        ds for ds, sc in by_ds_cls.items() if len(sc) == len(all_classes)
    ]

    # Mean rank, std, only over datasets where the class participated
    mean_ranks = []
    for cls in all_classes:
        ranks_list = [r["rank"] for r in rank_per_class[cls]]
        if not ranks_list:
            continue
        mean_ranks.append({
            "class": cls,
            "mean_rank": statistics.mean(ranks_list),
            "median_rank": statistics.median(ranks_list),
            "std_rank": statistics.stdev(ranks_list) if len(ranks_list) >= 2 else 0.0,
            "n_datasets": len(ranks_list),
            "wins": sum(1 for r in ranks_list if r == 1),
        })
    mean_ranks.sort(key=lambda x: x["mean_rank"])

    # Friedman χ² stat (rough; exact requires balanced design)
    friedman = None
    if complete_datasets:
        N = len(complete_datasets)
        k = len(all_classes)
        # Sum-of-rank-squared per class on complete-design datasets
        rank_sums: dict[str, list[float]] = defaultdict(list)
        for ds in complete_datasets:
            for cls, rec in zip(all_classes, sorted_cls):
                pass
        # Re-do cleanly: rebuild ranks on complete datasets only
        complete_rank_sums = defaultdict(float)
        for ds in complete_datasets:
            sorted_cls = sorted(by_ds_cls[ds].items(), key=lambda x: x[1])
            for rk, (cls, _) in enumerate(sorted_cls, start=1):
                complete_rank_sums[cls] += rk
        # chi² = 12 / (N*k*(k+1)) * sum(Ri²) - 3*N*(k+1)
        sum_r_sq = sum((rs ** 2) for rs in complete_rank_sums.values())
        chi2 = (12 / (N * k * (k + 1))) * sum_r_sq - 3 * N * (k + 1)
        friedman = {
            "chi_squared": chi2,
            "df": k - 1,
            "n_datasets_balanced": N,
            "n_classes": k,
        }

    # Nemenyi critical difference at α=0.05 (k≤20 tabulated)
    # q_alpha values for α=0.05 (two-tailed Studentized range / sqrt(2))
    _Q_05 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
        8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
        14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517,
        20: 3.544,
    }
    cd_value = None
    if complete_datasets and len(all_classes) >= 2:
        k = len(all_classes)
        N = len(complete_datasets)
        q = _Q_05.get(k, 3.544)  # fallback for k>20
        cd_value = q * math.sqrt(k * (k + 1) / (6 * N))

    # Subset analysis: pick top-K classes with highest dataset overlap
    # so we can actually compute Friedman + CD on a balanced design.
    subset_results = []
    for K in (6, 8, 10):
        # Greedy pick of K classes that maximise common dataset count
        cls_dsets = {c: valid_datasets_per_class[c] for c in all_classes}
        # Start from class with most datasets
        sorted_by_size = sorted(cls_dsets.items(), key=lambda x: -len(x[1]))
        picked = [sorted_by_size[0][0]]
        running_intersection = set(cls_dsets[picked[0]])
        for cls, dsets in sorted_by_size[1:]:
            cand_inter = running_intersection & dsets
            if len(cand_inter) >= 20:  # require ≥20 common datasets
                picked.append(cls)
                running_intersection = cand_inter
            if len(picked) >= K:
                break
        if len(picked) < K or len(running_intersection) < 10:
            continue
        # Compute Friedman + CD on this subset
        subset_ds = sorted(running_intersection)
        N = len(subset_ds)
        sub_k = len(picked)
        rank_sums_sub = defaultdict(float)
        rank_lists_sub: dict[str, list[float]] = defaultdict(list)
        for ds in subset_ds:
            sub_scores = sorted(((c, by_ds_cls[ds][c]) for c in picked), key=lambda x: x[1])
            for rk, (cls, _) in enumerate(sub_scores, start=1):
                rank_sums_sub[cls] += rk
                rank_lists_sub[cls].append(rk)
        sum_r_sq = sum((rs ** 2) for rs in rank_sums_sub.values())
        chi2 = (12 / (N * sub_k * (sub_k + 1))) * sum_r_sq - 3 * N * (sub_k + 1)
        q = _Q_05.get(sub_k, 3.544)
        cd = q * math.sqrt(sub_k * (sub_k + 1) / (6 * N))
        subset_results.append({
            "k": sub_k,
            "n_datasets": N,
            "classes": picked,
            "mean_ranks": sorted(
                [{"class": c, "mean_rank": statistics.mean(rank_lists_sub[c]),
                  "std_rank": statistics.stdev(rank_lists_sub[c]) if N >= 2 else 0.0}
                 for c in picked],
                key=lambda x: x["mean_rank"],
            ),
            "friedman_chi2": chi2,
            "friedman_df": sub_k - 1,
            "critical_difference": cd,
            "alpha": 0.05,
        })

    return {
        "rank_per_class": dict(rank_per_class),
        "mean_ranks": mean_ranks,
        "complete_datasets_count": len(complete_datasets),
        "friedman": friedman,
        "critical_difference": cd_value,
        "alpha": 0.05,
        "subset_analyses": subset_results,
    }


def _quantile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _truncate_pp(pp: str, max_len: int = 60) -> str:
    if len(pp) <= max_len:
        return pp
    return pp[: max_len - 1] + "…"


# Canonical taxonomy: heterogeneous pp strings → structured fields
_BANK_KEYWORDS = {"compact", "default", "extended", "deep3", "deep4",
                  "production_default", "production_compact", "custom"}
_BANK_FILTER_KEYWORDS = {"family_pruned", "response_dedup"}
_SCALING_TOKENS = {"center", "feature_std", "identity",
                   "minmaxscaler", "standardscaler", "normalize"}
_SCATTER_TOKENS = {"snv", "msc", "standardnormalvariate",
                   "multiplicativescattercorrection"}
_BASELINE_TOKENS = {"asls", "aslsbaseline", "baseline", "detrend"}
_DERIVATIVE_TOKENS = {"savitzkygolay", "sg", "firstderivative", "secondderivative"}


def _norm_token(name: str) -> str:
    """Map verbose class name → short canonical token."""
    n = name.lower().strip()
    if "minmax" in n: return "MinMax"
    if "standardscaler" in n: return "StdScaler"
    if "normalize" in n: return "Normalize"
    if "asls" in n or n == "aslsbaseline": return "ASLS"
    if "standardnormalvariate" in n or n == "snv": return "SNV"
    if "multiplicativescatter" in n or n == "msc": return "MSC"
    if "savitzkygolay" in n: return "SG"
    if "firstderivative" in n: return "Deriv1"
    if "secondderivative" in n: return "Deriv2"
    if "detrend" in n: return "Detrend"
    if n == "feature_std": return "FeatureStd"
    if n == "center": return "Center"
    if n == "identity": return "Identity"
    if n == "none": return None  # type: ignore[return-value]
    return name.strip()[:30]


def parse_preprocessing(raw: str) -> dict:
    """Parse a heterogeneous preprocessing_pipeline string into canonical fields.

    Returns a dict with keys:
      - raw: original string
      - category: 'none' | 'atom_bank' | 'hpo_compound' | 'pipeline_path' | 'ensemble' | 'unknown'
      - atom_bank: short bank name (compact/default/…) or None
      - bank_filtering: family_pruned / response_dedup or None
      - baseline: ASLS / None / etc.
      - scaling: MinMax / StdScaler / Center / FeatureStd / Identity / None
      - scatter: SNV / MSC / Normalize / None
      - derivative: SG / Deriv1 / Deriv2 / None
      - pca: 'PCA_features_0.25' / None
      - components: list of detected canonical tokens (lossy summary)
    """
    p = (raw or "").strip()
    out = {
        "raw": p,
        "category": "none" if not p else "unknown",
        "atom_bank": None,
        "bank_filtering": None,
        "baseline": None,
        "scaling": None,
        "scatter": None,
        "derivative": None,
        "pca": None,
        "components": [],
    }
    if not p:
        return out

    # 1. HPO compound (key=value | …)
    if " | " in p and "=" in p:
        out["category"] = "hpo_compound"
        for chunk in p.split(" | "):
            if "=" not in chunk:
                continue
            k, v = (s.strip() for s in chunk.split("=", 1))
            kl, vl = k.lower(), v.lower()
            if vl == "none":
                continue
            if kl == "scaler":
                out["scaling"] = _norm_token(v)
            elif kl == "baseline":
                out["baseline"] = _norm_token(v)
            elif kl == "simple":
                if "savitzky" in vl:
                    out["derivative"] = "SG"
                elif "standardnormalvariate" in vl or vl == "snv":
                    out["scatter"] = "SNV"
                elif "msc" in vl:
                    out["scatter"] = "MSC"
                elif "normalize" in vl:
                    out["scaling"] = "Normalize"
                elif "baseline" in vl:
                    out["baseline"] = _norm_token(v)
            elif kl == "pca":
                out["pca"] = v
            elif kl == "paper_best":
                # reference annotation, ignore
                continue
            out["components"].append(f"{k}={v}")
        return out

    # 2. Pipeline class path (Python paths chained with ' > ')
    if " > " in p or (">" in p and not p.startswith("baseline")):
        out["category"] = "pipeline_path"
        parts = [x.strip() for x in p.split(">") if x.strip()]
        for part in parts:
            cls = part.rsplit(".", 1)[-1]
            cl = cls.lower()
            if not cls:
                continue
            if "minmax" in cl:
                out["scaling"] = "MinMax"
            elif "standardscaler" in cl:
                out["scaling"] = "StdScaler"
            elif "asls" in cl:
                out["baseline"] = "ASLS"
            elif "standardnormalvariate" in cl or cl == "snv":
                out["scatter"] = "SNV"
            elif "multiplicativescatter" in cl or cl == "msc":
                out["scatter"] = "MSC"
            elif "savitzkygolay" in cl:
                out["derivative"] = "SG"
            elif "firstderivative" in cl:
                out["derivative"] = "Deriv1"
            elif "secondderivative" in cl:
                out["derivative"] = "Deriv2"
            elif "detrend" in cl:
                out["baseline"] = "Detrend"
            elif "spxy" in cl or "splitter" in cl:
                continue  # splitters aren't preprocessing
            out["components"].append(_norm_token(cls) or cls[:20])
        return out

    # 3. List notation (ensemble of pp atoms)
    if p.startswith("[") and p.endswith("]"):
        try:
            import ast
            items = ast.literal_eval(p)
            if isinstance(items, (list, tuple)) and items:
                out["category"] = "ensemble"
                tokens = []
                for x in items:
                    s = str(x).lower()
                    if "snv" in s: tokens.append("SNV")
                    elif "msc" in s: tokens.append("MSC")
                    elif "asls" in s: tokens.append("ASLS")
                    else: tokens.append(str(x)[:20])
                out["components"] = tokens
                return out
        except Exception:
            pass

    # 4. Single atom-bank / classical token
    pl = p.lower().strip()
    out["category"] = "atom_bank"
    out["components"] = [p]
    if pl in _BANK_KEYWORDS:
        out["atom_bank"] = pl
    elif pl in _BANK_FILTER_KEYWORDS:
        out["bank_filtering"] = pl
    elif pl == "center":
        out["scaling"] = "Center"
    elif pl == "feature_std":
        out["scaling"] = "FeatureStd"
    elif pl == "identity":
        out["scaling"] = "Identity"
    elif pl in {"snv", "msc"}:
        out["scatter"] = pl.upper()
    elif pl == "asls":
        out["baseline"] = "ASLS"
    elif pl == "none":
        out["category"] = "none"
    return out


def build_preprocessing_master(obs: list[dict]) -> dict:
    """Aggregate preprocessing influence across all observed/paper rows.

    Produces:
      - leaderboard: per raw pp stats
      - class_pp_heatmap: median rmsep per (class, raw pp) cell
      - best_pp_per_dataset: per-dataset best/worst pp + spread
      - components: per-canonical-component aggregations
        (baseline, scaling, scatter, derivative, atom_bank) → list of {value, n_obs, median, …}
      - component_pairs: { (baseline, scatter): {n, median} } for cross-component matrix
    """
    pp_obs: dict[str, list[float]] = defaultdict(list)
    pp_dataset: dict[str, set] = defaultdict(set)
    pp_class: dict[str, set] = defaultdict(set)
    pp_model: dict[str, set] = defaultdict(set)
    class_pp_obs: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    ds_pp_obs: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    # Component-level aggregations (the actual useful per-component stats)
    component_obs: dict[str, dict[str, list[float]]] = {
        "baseline": defaultdict(list),
        "scaling": defaultdict(list),
        "scatter": defaultdict(list),
        "derivative": defaultdict(list),
        "atom_bank": defaultdict(list),
        "bank_filtering": defaultdict(list),
        "category": defaultdict(list),
    }
    component_classes: dict[str, dict[str, set]] = {
        k: defaultdict(set) for k in component_obs
    }
    component_datasets: dict[str, dict[str, set]] = {
        k: defaultdict(set) for k in component_obs
    }
    # Cross matrix: baseline × scatter (most informative pp pair)
    pair_obs: dict[tuple, list[float]] = defaultdict(list)

    for r in obs:
        rmsep = safe_float(r.get("rmsep"))
        if rmsep is None or not math.isfinite(rmsep):
            continue
        pp = (r.get("preprocessing_pipeline") or r.get("preprocessing") or "").strip()
        if not pp:
            continue
        cls = (r.get("model_class") or "").strip()
        ds = (r.get("dataset") or "").strip()
        model = (r.get("model_name") or "").strip()

        pp_obs[pp].append(rmsep)
        pp_dataset[pp].add(ds)
        if cls:
            pp_class[pp].add(cls)
            class_pp_obs[cls][pp].append(rmsep)
        if model:
            pp_model[pp].add(model)
        if ds:
            ds_pp_obs[ds][pp].append(rmsep)

        # Component breakdown
        parsed = parse_preprocessing(pp)
        for comp_key in ("baseline", "scaling", "scatter", "derivative",
                         "atom_bank", "bank_filtering", "category"):
            val = parsed.get(comp_key)
            if val is None:
                val = "none"
            component_obs[comp_key][val].append(rmsep)
            if cls:
                component_classes[comp_key][val].add(cls)
            if ds:
                component_datasets[comp_key][val].add(ds)
        pair_obs[(parsed.get("baseline") or "none",
                  parsed.get("scatter") or "none")].append(rmsep)

    # Leaderboard (filter: ≥5 obs to cut noise)
    leaderboard = []
    for pp, vals in pp_obs.items():
        if len(vals) < 5:
            continue
        leaderboard.append({
            "pp": _truncate_pp(pp),
            "pp_full": pp if len(pp) > 60 else None,
            "n_obs": len(vals),
            "n_datasets": len(pp_dataset[pp]),
            "n_classes": len(pp_class[pp]),
            "n_models": len(pp_model[pp]),
            "median": statistics.median(vals),
            "q25": _quantile(vals, 0.25),
            "q75": _quantile(vals, 0.75),
            "best": min(vals),
        })
    leaderboard.sort(key=lambda x: x["median"])

    # Class × PP heatmap — top 25 pp by n_obs
    top_pps = [p["pp_full"] or p["pp"] for p in sorted(leaderboard, key=lambda x: -x["n_obs"])[:25]]
    classes = sorted(class_pp_obs.keys())
    heatmap_values = [[
        statistics.median(class_pp_obs[c][p]) if class_pp_obs[c].get(p) else None
        for p in top_pps
    ] for c in classes]
    heatmap_counts = [[len(class_pp_obs[c].get(p, [])) for p in top_pps] for c in classes]
    heatmap = {
        "classes": classes,
        "pps": [_truncate_pp(p, 40) for p in top_pps],
        "pps_full": top_pps,
        "values": heatmap_values,
        "counts": heatmap_counts,
    }

    # Best PP per dataset
    best_pp_per_dataset = []
    for ds, pps in ds_pp_obs.items():
        if len(pps) < 2:
            continue
        ranked = sorted(
            ((p, statistics.median(v), len(v)) for p, v in pps.items() if len(v) >= 2),
            key=lambda x: x[1],
        )
        if not ranked:
            continue
        best = ranked[0]
        worst = ranked[-1]
        best_pp_per_dataset.append({
            "dataset": ds,
            "n_pps": len(pps),
            "best_pp": _truncate_pp(best[0]),
            "best_median": best[1],
            "best_n": best[2],
            "worst_pp": _truncate_pp(worst[0]),
            "worst_median": worst[1],
            "spread": worst[1] - best[1],
            "rel_spread": (worst[1] - best[1]) / best[1] if best[1] > 0 else 0.0,
        })
    best_pp_per_dataset.sort(key=lambda x: -x["spread"])

    # Component leaderboards
    components_out: dict[str, list[dict]] = {}
    for comp_key, obs_map in component_obs.items():
        entries = []
        for val, vals in obs_map.items():
            if len(vals) < 5:
                continue
            entries.append({
                "value": val,
                "n_obs": len(vals),
                "n_classes": len(component_classes[comp_key][val]),
                "n_datasets": len(component_datasets[comp_key][val]),
                "median": statistics.median(vals),
                "q25": _quantile(vals, 0.25),
                "q75": _quantile(vals, 0.75),
                "best": min(vals),
            })
        entries.sort(key=lambda x: x["median"])
        components_out[comp_key] = entries

    # baseline × scatter pair matrix
    baselines = sorted({k[0] for k in pair_obs})
    scatters = sorted({k[1] for k in pair_obs})
    pair_matrix = {
        "rows": baselines,
        "cols": scatters,
        "values": [[
            statistics.median(pair_obs[(b, s)]) if pair_obs.get((b, s)) and len(pair_obs[(b, s)]) >= 5 else None
            for s in scatters
        ] for b in baselines],
        "counts": [[len(pair_obs.get((b, s), [])) for s in scatters] for b in baselines],
    }

    return {
        "leaderboard": leaderboard,
        "class_pp_heatmap": heatmap,
        "best_pp_per_dataset": best_pp_per_dataset,
        "components": components_out,
        "baseline_scatter_matrix": pair_matrix,
    }


def build_preset_pools_section(master: list[dict]) -> dict:
    """Surface the data-driven pool re-design (`preset_audit/proposed_presets.json`).

    For each preset:
      - pool: list of canonical_name in the pool
      - sequential q90 fit-time budget (per dataset)
      - per-dataset best-in-pool rmsep/PLS ratio
      - per-dataset cumulative pool time (sum of q90_fit_time across pool)
    Both proposed and current pools are scored, so the dashboard can show the
    delta the new pools deliver.
    """
    audit_path = BENCH / "scenarios" / "preset_audit" / "proposed_presets.json"
    evidence_path = BENCH / "scenarios" / "preset_audit" / "per_model_evidence.csv"
    registry_path = BENCH / "scenarios" / "model_registry.yaml"
    if not audit_path.exists() or not evidence_path.exists():
        return {"available": False, "reason": "preset_audit artefacts missing"}

    proposed = json.loads(audit_path.read_text(encoding="utf-8"))

    # Build alias→canonical index by reading the registry as YAML-ish text
    # (avoid pulling pyyaml here; the file has a strict structure).
    alias_to_canonical: dict[str, str] = {}
    canonical_name: str | None = None
    in_models_section = False
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        s = line.rstrip()
        if s.startswith("models:"):
            in_models_section = True
            continue
        if in_models_section and (s.startswith("cohorts:") or s.startswith("presets:")):
            in_models_section = False
        if not in_models_section:
            continue
        stripped = s.lstrip()
        if stripped.startswith("- canonical_name:"):
            canonical_name = stripped.split(":", 1)[1].strip()
            alias_to_canonical[_norm_token(canonical_name)] = canonical_name
        elif stripped.startswith("- ") and canonical_name and ":" not in stripped:
            alias = stripped[2:].strip()
            if alias:
                alias_to_canonical[_norm_token(alias)] = canonical_name
    # Hard extras same as audit script (TabPFN raw fallbacks).
    for raw, canon in (
        ("tabpfnregressor", "TabPFN-Raw"),
        ("tabpfn-standalone", "TabPFN-Raw"),
        ("tabpfn-oracle", "TabPFN-opt"),
        ("aomridgeregressor", "AOMRidge-global-compact-none"),
        ("aomplsregressor", "AOM-PLS-compact-numpy"),
        ("aomlocalridge", "AOMRidge-Local-compact-knn50"),
    ):
        alias_to_canonical.setdefault(raw, canon)

    # Per-model q90_fit_time_s for budget bookkeeping in the dashboard.
    q90_time: dict[str, float | None] = {}
    median_ratio: dict[str, float | None] = {}
    with evidence_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            canon = row["canonical_name"]
            q90_time[canon] = safe_float(row.get("q90_fit_time_s"))
            median_ratio[canon] = safe_float(row.get("median_ratio"))

    # Aggregate master → per (canonical, dataset) best rmsep + per-dataset PLS.
    by_cell: dict[tuple[str, str], float] = {}
    pls_per_dataset: dict[str, float] = {}
    for row in master:
        if row.get("record_type") not in ("observed", "reference_paper"):
            continue
        if (row.get("status") or "ok").lower() not in ("", "ok", "success", "done", "complete", "completed"):
            continue
        if (row.get("evaluation_split") or "").lower() in ("train", "cv", "cross_val", "cross-validation", "cros val"):
            continue
        if row.get("score_metric") != "rmsep":
            continue
        dataset = (row.get("dataset") or "").strip()
        model_name = (row.get("model_name") or "").strip()
        variant = (row.get("variant") or "").strip()
        if not dataset or not model_name:
            continue
        rmsep = safe_float(row.get("rmsep")) or safe_float(row.get("score_value"))
        if rmsep is None:
            continue
        canon = alias_to_canonical.get(_norm_token(model_name)) or alias_to_canonical.get(_norm_token(variant))
        if canon is None:
            continue
        key = (canon, dataset)
        prev = by_cell.get(key)
        if prev is None or rmsep < prev:
            by_cell[key] = rmsep
        if canon == "PLS-tuned-cv5":
            cur = pls_per_dataset.get(dataset)
            if cur is None or rmsep < cur:
                pls_per_dataset[dataset] = rmsep

    # For each pool, score each dataset.
    def score_pool(pool: list[str]) -> dict[str, Any]:
        per_dataset: list[dict] = []
        for dataset in sorted(pls_per_dataset.keys()):
            pls = pls_per_dataset[dataset]
            if pls <= 0:
                continue
            best_canon = None
            best_ratio = None
            best_rmsep = None
            for canon in pool:
                rms = by_cell.get((canon, dataset))
                if rms is None:
                    continue
                ratio = min(rms / pls, 5.0)
                if best_ratio is None or ratio < best_ratio:
                    best_ratio = ratio
                    best_canon = canon
                    best_rmsep = rms
            per_dataset.append({
                "dataset": dataset,
                "pls_rmsep": pls,
                "best_rmsep": best_rmsep,
                "best_ratio": best_ratio,
                "best_model": best_canon,
            })
        total_time = sum((q90_time.get(c) or 0.0) for c in pool)
        scored = [d for d in per_dataset if d["best_ratio"] is not None]
        ratios = [d["best_ratio"] for d in scored]
        return {
            "pool": pool,
            "pool_q90_times": [q90_time.get(c) for c in pool],
            "total_seq_time_s": total_time,
            "per_dataset": per_dataset,
            "n_datasets_scored": len(scored),
            "mean_ratio": statistics.fmean(ratios) if ratios else None,
            "median_ratio": (quantile(ratios, 0.5) if ratios else None),
            "q90_ratio": (quantile(ratios, 0.9) if ratios else None),
        }

    # Score the *current* pre-redesign preset members the same way for delta.
    CURRENT_PRESETS = {
        "fast_reliable": [
            "PLS-tuned-cv5", "Ridge-tuned-cv5", "ASLS-AOM-compact-cv5-numpy",
            "AOM-PLS-compact-numpy", "AOMRidge-global-compact-none",
            "AOMRidge-global-compact-snv",
        ],
        "strong_practical": [
            "PLS-tuned-cv5", "Ridge-tuned-cv5", "ASLS-AOM-compact-cv5-numpy",
            "AOM-PLS-compact-numpy", "AOMRidge-global-compact-none",
            "AOMRidge-global-compact-snv", "AOMRidge-Local-compact-knn50",
        ],
        "best_current": [
            "PLS-tuned-cv5", "Ridge-tuned-cv5", "ASLS-AOM-compact-cv5-numpy",
            "AOM-PLS-compact-numpy", "AOMRidge-global-compact-none",
            "AOMRidge-global-compact-snv", "AOMRidge-Local-compact-knn50",
            "AOMRidge-MultiBranchMKL-compact-shrink03",
        ],
        "exhaustive_research": [
            "PLS-tuned-cv5", "Ridge-tuned-cv5", "ASLS-AOM-compact-cv5-numpy",
            "AOM-PLS-compact-numpy", "AOM-default-nipals-adjoint-numpy",
            "POP-PLS-compact-numpy", "AOMRidge-global-compact-none",
            "AOMRidge-global-compact-snv", "AOMRidge-Local-compact-knn50",
            "AOMRidge-MultiBranchMKL-compact-shrink03",
            "AOMRidge-Blender-headline-spxy3",
            "AOMRidge-AutoSelect-headline-spxy3",
            "V2L-Residual-AOMPLS", "V2L-Boost-AOMPLS",
            "FCK-AOMPLS-static", "FCK-PLS-static",
        ],
    }

    out: dict[str, Any] = {
        "available": True,
        "generated_on": proposed.get("generated_on") or "2026-05-12",
        "presets": {},
    }
    for preset, plan in proposed.items():
        proposed_scored = score_pool(plan["pool"])
        current_scored = score_pool(CURRENT_PRESETS.get(preset, []))
        out["presets"][preset] = {
            "budget_seconds": plan.get("budget_seconds"),
            "candidate_pool_size": plan.get("candidate_pool_size"),
            "expected_pool_mean_ratio": plan.get("expected_pool_mean_ratio"),
            "selected_search": plan.get("selected_search"),
            "gate": plan.get("gate"),
            "proposed": proposed_scored,
            "current": current_scored,
        }
    return out


def main() -> None:
    master = load_master_rows()

    presets_out: dict[str, dict] = {}
    for name, ws in PRESETS:
        presets_out[name] = build_preset_section(name, ws)

    datasets_out = build_dataset_section(presets_out)
    h2h_out = build_head_to_head("best_current_full57_seed0")
    preproc_out = build_preprocessing_pairs("best_current_full57_seed0")
    failures_out = build_failures(presets_out)
    tabpfn_out = build_tabpfn_paper(master)
    master_aggregations = build_master_aggregations(master)
    preset_pools_out = build_preset_pools_section(master)

    payload = {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "n_datasets_total": len(datasets_out),
            "n_candidates_preset": len({c["canonical_name"] for p in presets_out.values() for c in p["candidates"]}),
            "phases_complete": sum(1 for p in presets_out.values() if p["available"]),
            "master_rows_total": len(master),
            "master_observed_total": master_aggregations["n_observations_total"],
            "master_distinct_models": len(master_aggregations["top_models"]),
            "master_model_classes": len(master_aggregations["by_model_class"]),
            "master_source_runs": len(master_aggregations["by_source_run"]),
        },
        "presets": presets_out,
        "datasets": datasets_out,
        "head_to_head": h2h_out,
        "preprocessing_pairs": preproc_out,
        "failures": failures_out,
        "tabpfn_paper": tabpfn_out,
        "master": master_aggregations,
        "preset_pools": preset_pools_out,
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    size_kb = OUT_JSON.stat().st_size / 1024
    print(f"Wrote {OUT_JSON} ({size_kb:.1f} KB)")
    print(f"  presets: {sum(1 for p in presets_out.values() if p['available'])}/{len(PRESETS)}")
    print(f"  datasets (preset): {len(datasets_out)}")
    print(f"  candidates (preset): {payload['metadata']['n_candidates_preset']}")
    print(f"  master observed/paper rows: {master_aggregations['n_observations_total']}")
    print(f"  master distinct models: {len(master_aggregations['top_models'])}")
    print(f"  master model classes: {len(master_aggregations['by_model_class'])}")
    print(f"  master source runs: {len(master_aggregations['by_source_run'])}")
    print(f"  observations_lean (for plots): {len(master_aggregations['observations_lean'])}")
    print(f"  per_dataset_best: {len(master_aggregations['per_dataset_best'])}")


if __name__ == "__main__":
    main()
