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
            for ds in {k[1] for k in pair_rmsep.keys() if k[0] == a}:
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
        for ds in {k[1] for k in pair_rmsep.keys() if k[0] == a}:
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
    # Filter
    obs = [
        r for r in master
        if r.get("record_type") in ("observed", "reference_paper")
        and r.get("status", "") in ("ok", "")
        and safe_float(r.get("rmsep")) is not None
    ]

    # Group by model_name
    by_model_name: dict[str, list[dict]] = defaultdict(list)
    for r in obs:
        mn = (r.get("model_name") or "").strip()
        if mn:
            by_model_name[mn].append(r)

    # Per-model_name aggregation
    top_models = []
    for mn, mn_rows in by_model_name.items():
        rmseps = [safe_float(r.get("rmsep")) for r in mn_rows]
        rmseps = [x for x in rmseps if x is not None]
        if not rmseps:
            continue
        # Best rmsep per dataset for this model_name (collapse duplicates)
        best_per_ds: dict[str, float] = {}
        for r in mn_rows:
            v = safe_float(r.get("rmsep"))
            if v is None:
                continue
            ds = r.get("dataset", "")
            if not ds:
                continue
            if ds not in best_per_ds or v < best_per_ds[ds]:
                best_per_ds[ds] = v
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
            "source_runs": source_runs[:10],  # cap list length
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

    return {
        "top_models": top_models,
        "by_model_class": by_model_class,
        "by_source_run": by_source_run,
        "by_source_family": by_source_family,
        "by_maturity": maturity_summary,
        "per_dataset_best": per_dataset_best,
        "observations_lean": obs_lean,
        "n_observations_total": len(obs),
    }


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
