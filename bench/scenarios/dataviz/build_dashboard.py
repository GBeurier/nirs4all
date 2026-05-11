"""Build comprehensive benchmark dataviz dashboard.

Reads:
  - bench/scenarios/runs/<preset>_full57_seed0/results.csv (Phase 2-5 production)
  - bench/benchmark_master_results.csv (master, for legacy TabPFN paper rows
    and historical context)

Produces:
  - bench/scenarios/dataviz/dashboard_data.json (data blob for the HTML page)

The HTML page (dashboard.html) consumes this JSON inline.

Sections:
  1. Header / overview cards
  2. Per-preset leaderboards
  3. Model class / family aggregation
  4. Per-dataset rmsep heatmap (normalized)
  5. Dataset size analysis (rmsep vs n_train)
  6. Preprocessing influence (per-pair delta analysis)
  7. Robustness / consistency (cross-preset variance)
  8. Runtime / speed-accuracy Pareto
  9. Failure pattern analysis
 10. Head-to-head win matrix (best_current)
 11. MoE / Stack performances breakdown
 12. AOMRidge-Local k-tuning cross-retention (D-A-009)

Owner: Agent C. No new decisions; analysis of locked data.
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
# Family taxonomy aligned with the registry's intent + ml-modeling tradition.
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


def build_preset_section(preset_name: str, workspace: str) -> dict:
    rows = load_preset_rows(workspace)
    if not rows:
        return {"name": preset_name, "n_planned": 0, "candidates": [], "available": False}

    # Aggregate per candidate
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

    # Sort by median rmsep ascending (best first)
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


def build_dataset_section(presets: dict) -> dict:
    """Per-dataset analysis across all presets.

    For each dataset, collect rmsep + metadata (n_train, n_features) from
    whichever preset provides them. Best candidate is computed across the
    union of all preset rows.
    """
    # Map dataset → {meta, candidates: {cand: rmsep}}
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
            # Keep best rmsep per candidate × dataset (across presets — should be identical)
            ds_data[ds_name]["candidates"][r["canonical_name"]] = rmsep

    # Compute best candidate per dataset
    for ds in ds_data.values():
        if ds["candidates"]:
            best_cand, best_rmsep = min(ds["candidates"].items(), key=lambda x: x[1])
            ds["best_candidate"] = best_cand
            ds["best_rmsep"] = best_rmsep
            ds["worst_rmsep"] = max(ds["candidates"].values())
            family, short, color = family_of(best_cand)
            ds["best_family"] = family

    return list(ds_data.values())


def build_family_section(presets: dict) -> dict:
    """Aggregate candidates by family across all presets."""
    fam_data: dict[str, dict] = defaultdict(lambda: {
        "candidates": set(),
        "rmseps_per_dataset": defaultdict(list),
    })

    for preset_name, info in presets.items():
        rows = load_preset_rows(info["workspace"])
        for r in rows:
            if r["status"] != "ok":
                continue
            rmsep = safe_float(r.get("rmsep"))
            if rmsep is None:
                continue
            family, _, _ = family_of(r["canonical_name"])
            fam_data[family]["candidates"].add(r["canonical_name"])
            fam_data[family]["rmseps_per_dataset"][r["dataset"]].append(rmsep)

    out = []
    for family, info in fam_data.items():
        # Take best (min) rmsep per dataset across family candidates
        best_per_ds = {ds: min(rmseps) for ds, rmseps in info["rmseps_per_dataset"].items()}
        family_rmseps = list(best_per_ds.values())
        out.append({
            "family": family,
            "candidates": sorted(info["candidates"]),
            "n_candidates": len(info["candidates"]),
            "n_datasets_covered": len(best_per_ds),
            "median_best_rmsep": median(family_rmseps),
            "q90_best_rmsep": quantile(family_rmseps, 0.90),
        })
    out.sort(key=lambda x: x["median_best_rmsep"] if x["median_best_rmsep"] is not None else float("inf"))
    return out


def build_head_to_head(workspace: str) -> dict:
    """Build head-to-head win matrix for a single preset (best_current default).

    For each (cand_a, cand_b), count datasets where cand_a rmsep < cand_b rmsep.
    """
    rows = load_preset_rows(workspace)
    # (cand, ds) → rmsep (status=ok only)
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
    return {
        "workspace": workspace,
        "candidates": candidates,
        "wins": h2h,
    }


def build_preprocessing_pairs(workspace: str) -> list[dict]:
    """Compare candidate pairs that differ only in preprocessing knob.

    Currently the cleanest pair is `AOMRidge-global-compact-none` vs `-snv`.
    Future pairs can be added here.
    """
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
            deltas.append({"dataset": ds, "delta": rb - ra, "ratio": rb / ra if ra != 0 else None})
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
    """Failure pattern analysis across all presets."""
    failed_by_ds_cand: dict[str, list[str]] = defaultdict(list)
    timeout_by_ds_cand: dict[str, list[str]] = defaultdict(list)

    # We use best_current as canonical view (most candidates seen there)
    workspace = presets["best_current"]["workspace"]
    rows = load_preset_rows(workspace)
    for r in rows:
        key = r["dataset"]
        if r["status"] == "failed":
            failed_by_ds_cand[key].append(r["canonical_name"])
        elif r["status"] == "failed_terminal":
            timeout_by_ds_cand[key].append(r["canonical_name"])

    return {
        "failed_by_dataset": {k: sorted(v) for k, v in failed_by_ds_cand.items()},
        "timeout_by_dataset": {k: sorted(v) for k, v in timeout_by_ds_cand.items()},
    }


def build_tabpfn_paper(master_path: Path) -> list[dict]:
    """Extract TabPFN paper rows from master CSV (reference_paper records)."""
    if not master_path.exists():
        return []
    tabpfn_rows = []
    with master_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("record_type") != "reference_paper":
                continue
            mn = (r.get("model_name") or "").strip()
            if "tabpfn" not in mn.lower():
                continue
            rmsep = safe_float(r.get("rmsep"))
            if rmsep is None:
                continue
            tabpfn_rows.append({
                "model_name": mn,
                "dataset": r.get("dataset", ""),
                "rmsep": rmsep,
            })
    return tabpfn_rows


def main() -> None:
    presets_out: dict[str, dict] = {}
    for name, ws in PRESETS:
        presets_out[name] = build_preset_section(name, ws)

    datasets_out = build_dataset_section(presets_out)
    families_out = build_family_section(presets_out)
    h2h_out = build_head_to_head("best_current_full57_seed0")
    preproc_out = build_preprocessing_pairs("best_current_full57_seed0")
    failures_out = build_failures(presets_out)
    tabpfn_out = build_tabpfn_paper(MASTER)

    payload = {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "n_datasets_total": len(datasets_out),
            "n_candidates_total": len({c["canonical_name"] for p in presets_out.values() for c in p["candidates"]}),
            "phases_complete": sum(1 for p in presets_out.values() if p["available"]),
        },
        "presets": presets_out,
        "datasets": datasets_out,
        "families": families_out,
        "head_to_head": h2h_out,
        "preprocessing_pairs": preproc_out,
        "failures": failures_out,
        "tabpfn_paper": tabpfn_out,
        "family_colors": {f: family_of(list(FAMILIES.keys())[0])[2] for f in {family_of(c)[0] for c in FAMILIES}},
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT_JSON} ({OUT_JSON.stat().st_size:,} bytes)")
    print(f"  presets: {sum(1 for p in presets_out.values() if p['available'])}/{len(PRESETS)}")
    print(f"  datasets: {len(datasets_out)}")
    print(f"  candidates: {payload['metadata']['n_candidates_total']}")
    print(f"  families: {len(families_out)}")
    print(f"  tabpfn paper rows: {len(tabpfn_out)}")


if __name__ == "__main__":
    main()
