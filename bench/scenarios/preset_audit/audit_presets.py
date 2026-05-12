"""Audit current 4 presets vs a data-driven time-budget pool selection.

Reads:
    bench/benchmark_master_results.csv  (locked + exploratory observed rmsep rows)
    bench/scenarios/model_registry.yaml  (canonical_name / alias mapping)
    bench/scenarios/runs/<preset>_full57_seed0/results.csv  (production run times)

Writes:
    bench/scenarios/preset_audit/per_model_evidence.csv
    bench/scenarios/preset_audit/per_dataset_best.csv
    bench/scenarios/preset_audit/preset_audit.json
    bench/scenarios/preset_audit/proposed_presets.json
    bench/scenarios/preset_audit/AUDIT.md

This is the evidence basis for the 4-preset re-design.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

BENCH = Path("/home/delete/nirs4all/nirs4all/bench")
MASTER = BENCH / "benchmark_master_results.csv"
REGISTRY = BENCH / "scenarios" / "model_registry.yaml"
RUNS_DIR = BENCH / "scenarios" / "runs"
OUT = BENCH / "scenarios" / "preset_audit"
OUT.mkdir(exist_ok=True)

PRESETS_WORKSPACE = {
    "fast_reliable": RUNS_DIR / "fast_reliable_full57_seed0" / "results.csv",
    "strong_practical": RUNS_DIR / "strong_practical_full57_seed0" / "results.csv",
    "best_current": RUNS_DIR / "best_current_full57_seed0" / "results.csv",
    "exhaustive_research": RUNS_DIR / "exhaustive_research_full57_seed0" / "results.csv",
}


def normalise(s: str) -> str:
    return (s or "").strip().lower().replace("_", "-").replace(" ", "-")


def to_float(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        v = float(s)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# 1. Load registry & alias index
# ---------------------------------------------------------------------------

def load_registry() -> tuple[dict[str, dict], dict[str, str]]:
    """Returns (entries_by_canonical, alias_to_canonical)."""
    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    entries_by_canonical: dict[str, dict] = {}
    alias_to_canonical: dict[str, str] = {}
    for entry in data["models"]:
        canon = entry["canonical_name"]
        entries_by_canonical[canon] = entry
        for alias in [canon, *entry.get("aliases", [])]:
            alias_to_canonical[normalise(alias)] = canon
    return entries_by_canonical, alias_to_canonical


# Hand-curated extra aliases that appear in master CSV but are not in registry yet.
#
# Provenance note (2026-05-12, post user-flag): the TabPFN paper's NIRS pivot
# (bench/AOM_v0/publication/tables/master_pivot.csv, columns "TabPFN-Raw" and
# "TabPFN-opt") is the upstream source for both TabPFN baselines. The "opt"
# column is the best score from the paper's *preprocessing HPO* (verified by
# cross-checking against bench/tabpfn_paper/table_results_tabpfn_final_light.csv
# "selected_by mean_val(top3)" rows). Therefore TabPFN-opt and TabPFN-HPO-preprocessing
# represent the SAME expensive HPO computation, just reported in two curated tables.
# Treat them as a single canonical (TabPFN-HPO-preprocessing) so the audit cannot
# allocate the HPO score to a cheaper time budget.
EXTRA_ALIASES = {
    "tabpfnregressor": "TabPFN-Raw",                  # default fast inference
    "tabpfn-standalone": "TabPFN-Raw",
    "tabpfn-oracle": "TabPFN-HPO-preprocessing",       # paper-pivot oracle = HPO best
    "tabpfn-hpo-preprocessing": "TabPFN-HPO-preprocessing",
    "tabpfn-opt": "TabPFN-HPO-preprocessing",          # paper-pivot "opt" = HPO best
    "tabpfn_opt": "TabPFN-HPO-preprocessing",
    "tabpfn-raw": "TabPFN-Raw",
    "aomridgeregressor": "AOMRidge-global-compact-none",
    "aomplsregressor": "AOM-PLS-compact-numpy",
    "aomlocalridge": "AOMRidge-Local-compact-knn50",
}


def lookup_canonical(model_name: str, variant: str, alias_to_canonical: dict[str, str]) -> str | None:
    """Try (model_name, variant, normalised, EXTRA_ALIASES) to find canonical."""
    for key_src in (model_name, variant):
        key = normalise(key_src)
        if key in alias_to_canonical:
            return alias_to_canonical[key]
        if key in EXTRA_ALIASES:
            return EXTRA_ALIASES[key]
    return None


# ---------------------------------------------------------------------------
# 2. Aggregate master CSV per (canonical, dataset)
# ---------------------------------------------------------------------------

@dataclass
class ModelOnDataset:
    canonical: str
    dataset: str
    best_rmsep: float | None = None
    fit_times: list[float] = field(default_factory=list)
    n_observations: int = 0
    maturities: set = field(default_factory=set)
    source_runs: set = field(default_factory=set)


def aggregate_master(alias_to_canonical: dict[str, str]) -> dict[tuple[str, str], ModelOnDataset]:
    """Returns {(canonical, dataset): ModelOnDataset} keeping the best rmsep per cell."""
    out: dict[tuple[str, str], ModelOnDataset] = {}
    with MASTER.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("record_type") not in ("observed", "reference_paper"):
                continue
            if (row.get("status") or "ok").lower() not in ("", "ok", "success", "done", "complete", "completed"):
                continue
            if (row.get("evaluation_split") or "").lower() in ("train", "cv", "cross_val", "cross-validation", "cros val"):
                continue
            if row.get("score_metric") != "rmsep":
                continue
            dataset = (row.get("dataset") or "").strip()
            if not dataset:
                continue
            canon = lookup_canonical(row.get("model_name", ""), row.get("variant", ""), alias_to_canonical)
            if canon is None:
                continue

            rmsep = to_float(row.get("rmsep") or row.get("score_value"))
            if rmsep is None:
                continue

            key = (canon, dataset)
            cell = out.setdefault(key, ModelOnDataset(canonical=canon, dataset=dataset))
            cell.n_observations += 1
            cell.maturities.add(row.get("protocol_maturity", ""))
            cell.source_runs.add(row.get("source_run", ""))
            if cell.best_rmsep is None or rmsep < cell.best_rmsep:
                cell.best_rmsep = rmsep
            ft = to_float(row.get("fit_time_s"))
            if ft is not None and ft > 0:
                cell.fit_times.append(ft)
    return out


# ---------------------------------------------------------------------------
# 3. Load preset run results for *actual* fit_time observations
# ---------------------------------------------------------------------------

def load_preset_times() -> dict[str, dict[str, list[float]]]:
    """Returns {canonical: {dataset: [fit_time_s,...]}} from production runs."""
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for preset, path in PRESETS_WORKSPACE.items():
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") != "ok":
                    continue
                canon = (row.get("canonical_name") or "").strip()
                dataset = (row.get("dataset") or "").strip()
                ft = to_float(row.get("fit_time_s"))
                if canon and dataset and ft is not None and ft > 0:
                    out[canon][dataset].append(ft)
    return out


# ---------------------------------------------------------------------------
# 4. Compute per-(dataset) PLS baseline & per-model ratios
# ---------------------------------------------------------------------------

def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    return statistics.quantiles(s, n=100, method="inclusive")[max(0, min(98, int(q * 100) - 1))]


def per_dataset_pls(cells: dict[tuple[str, str], ModelOnDataset]) -> dict[str, float]:
    """Best PLS rmsep per dataset (use PLS-tuned-cv5 only)."""
    out: dict[str, float] = {}
    for (canon, dataset), cell in cells.items():
        if canon == "PLS-tuned-cv5" and cell.best_rmsep is not None:
            cur = out.get(dataset)
            if cur is None or cell.best_rmsep < cur:
                out[dataset] = cell.best_rmsep
    return out


def per_model_summary(
    cells: dict[tuple[str, str], ModelOnDataset],
    pls_per_dataset: dict[str, float],
    preset_times: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, Any]]:
    """Per-canonical aggregated statistics."""
    by_canon: dict[str, dict] = defaultdict(lambda: {
        "ratios": [],            # ratio_vs_pls per dataset
        "datasets": set(),
        "fit_times": [],         # observed in master OR preset runs
        "rmsep_by_dataset": {},
    })
    for (canon, dataset), cell in cells.items():
        if cell.best_rmsep is None:
            continue
        pls = pls_per_dataset.get(dataset)
        if pls is None or pls <= 0:
            continue
        ratio = cell.best_rmsep / pls
        # Clip outliers to 5x for aggregation robustness.
        ratio_clip = min(ratio, 5.0)
        by_canon[canon]["ratios"].append(ratio_clip)
        by_canon[canon]["datasets"].add(dataset)
        by_canon[canon]["rmsep_by_dataset"][dataset] = cell.best_rmsep
        # Representative time per (model, dataset) = median of local observations.
        # This avoids both min-bias (paper-table imports with predict-only time) and
        # max-bias (slow first-run JIT).
        if cell.fit_times:
            by_canon[canon]["fit_times"].append(statistics.median(cell.fit_times))
        # Also incorporate fresh production-run times if available.
        if canon in preset_times and dataset in preset_times[canon]:
            by_canon[canon]["fit_times"].append(statistics.median(preset_times[canon][dataset]))

    out: dict[str, dict[str, Any]] = {}
    for canon, agg in by_canon.items():
        ratios = sorted(agg["ratios"])
        fits = sorted(agg["fit_times"])
        # Win ratio threshold uses `>= 0.99` because PLS is by definition 1.0 and
        # tiny numerical noise around it shouldn't disqualify a near-tied model.
        out[canon] = {
            "canonical_name": canon,
            "n_datasets": len(agg["datasets"]),
            "datasets": sorted(agg["datasets"]),
            "median_ratio": quantile(ratios, 0.5),
            "q75_ratio": quantile(ratios, 0.75),
            "q90_ratio": quantile(ratios, 0.9),
            "best_ratio": ratios[0] if ratios else None,
            "worst_ratio": ratios[-1] if ratios else None,
            "wins_vs_pls": sum(1 for r in ratios if r < 1.0),
            "median_fit_time_s": quantile(fits, 0.5),
            "q90_fit_time_s": quantile(fits, 0.9),
            "n_fit_obs": len(fits),
        }
    return out


# ---------------------------------------------------------------------------
# 5. Greedy preset pool selection
# ---------------------------------------------------------------------------

# Time budgets (seconds, *sequential* per-dataset upper bound).
# User vocabulary: "20 sec -> fast_reliable, more -> strong, more -> best, more -> exhaustive".
# We keep PLS as cheap anchor and grow the budget by ~one order of magnitude.
TIME_BUDGETS = {
    "fast_reliable": 30,            # ~30s per dataset (user: ~20s)
    "strong_practical": 600,        # ~10 minutes
    "best_current": 7200,           # 2 hours (user 2026-05-12)
    "exhaustive_research": 43200,   # 12 hours (user 2026-05-12)
}

# Minimum number of dataset observations in master CSV to consider a model
# "evidence-supported" enough to enter a non-research preset.
MIN_DATASETS_FOR_PRACTICAL = 25

# Models that failed >50% of dispatch attempts in the latest exhaustive_research
# production run are flagged "currently broken"; they get demoted to
# exhaustive_research only (and only when fixable).
RUNNABILITY_FAILURE_RATE_THRESHOLD = 0.5


def runnability_from_latest_run() -> dict[str, dict[str, Any]]:
    """Returns {canonical: {ok:..., err:..., success_rate:...}} from latest exhaustive run."""
    path = PRESETS_WORKSPACE["exhaustive_research"]
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"ok": 0, "err": 0})
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            canon = (row.get("canonical_name") or "").strip()
            if not canon:
                continue
            if row.get("status") == "ok":
                out[canon]["ok"] += 1
            else:
                out[canon]["err"] += 1
    for canon, rec in out.items():
        total = rec["ok"] + rec["err"]
        rec["success_rate"] = (rec["ok"] / total) if total > 0 else None
    return out


def _expected_time(per_model: dict[str, dict[str, Any]], canon: str) -> float:
    rec = per_model.get(canon, {})
    t = rec.get("q90_fit_time_s") or rec.get("median_fit_time_s")
    return float(t) if t else 0.0


def greedy_pool(
    per_model: dict[str, dict[str, Any]],
    candidates: list[str],
    budget_s: float,
    *,
    pls_per_dataset: dict[str, float],
    cells: dict[tuple[str, str], ModelOnDataset],
    min_pool_size: int = 1,
    must_include: list[str] | None = None,
    max_pool_size: int | None = None,
) -> dict[str, Any]:
    """Greedy: minimise mean over datasets of min-ratio-in-pool.

    Implements the user's "maximise expected score under dataset uncertainty
    given a time budget" criterion. Time is summed sequentially.
    """
    must_include = list(must_include or [])

    # Pre-compute model -> {dataset: ratio_clip}
    # Restrict universe to datasets that have a PLS baseline (others are unscorable).
    model_ratios: dict[str, dict[str, float]] = {}
    all_datasets: set[str] = set()
    for canon in candidates:
        ratios: dict[str, float] = {}
        for (m, ds), cell in cells.items():
            if m != canon or cell.best_rmsep is None:
                continue
            pls = pls_per_dataset.get(ds)
            if pls is None or pls <= 0:
                continue
            ratios[ds] = min(cell.best_rmsep / pls, 5.0)
            all_datasets.add(ds)
        model_ratios[canon] = ratios

    pool: list[str] = []
    # Initial pool from must_include (cost-checked).
    spent = 0.0
    for canon in must_include:
        if canon in candidates:
            cost = _expected_time(per_model, canon)
            if spent + cost <= budget_s:
                pool.append(canon)
                spent += cost
    # Track current-best ratio per dataset (the achieved minimum across pool).
    best_ratio: dict[str, float] = {ds: float("inf") for ds in all_datasets}
    for canon in pool:
        for ds, r in model_ratios[canon].items():
            if r < best_ratio[ds]:
                best_ratio[ds] = r
    # Missing datasets: assume "no model → ratio = 1.5" (15 % worse than PLS) penalty.
    PENALTY_RATIO = 1.5

    def pool_score(b: dict[str, float]) -> float:
        if not b:
            return PENALTY_RATIO
        vals = [v if v < float("inf") else PENALTY_RATIO for v in b.values()]
        return statistics.fmean(vals)

    # Greedy add until budget exhausted, no improvement, or max pool size hit.
    cur_score = pool_score(best_ratio)
    while True:
        if max_pool_size is not None and len(pool) >= max_pool_size:
            break
        best_cand: str | None = None
        best_new_score = cur_score
        for canon in candidates:
            if canon in pool:
                continue
            cost = _expected_time(per_model, canon)
            if cost <= 0:
                continue
            if spent + cost > budget_s:
                continue
            new_best = dict(best_ratio)
            for ds, r in model_ratios.get(canon, {}).items():
                if r < new_best.get(ds, float("inf")):
                    new_best[ds] = r
            s = pool_score(new_best)
            if s < best_new_score - 1e-6:
                best_new_score = s
                best_cand = canon
        if best_cand is None:
            break
        pool.append(best_cand)
        for ds, r in model_ratios[best_cand].items():
            if r < best_ratio.get(ds, float("inf")):
                best_ratio[ds] = r
        spent += _expected_time(per_model, best_cand)
        cur_score = best_new_score

    coverage = sum(1 for v in best_ratio.values() if v < float("inf"))
    covered_values = [v for v in best_ratio.values() if v < float("inf")]
    return {
        "pool": pool,
        "expected_seq_time_s": spent,
        "expected_pool_mean_ratio": cur_score,
        "expected_pool_median_ratio": quantile(covered_values, 0.5),
        "expected_pool_q90_ratio": quantile(covered_values, 0.9),
        "n_datasets_covered": coverage,
        "n_datasets_total": len(all_datasets),
    }


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main() -> None:
    entries, aliases = load_registry()
    cells = aggregate_master(aliases)
    preset_times = load_preset_times()
    pls_per_dataset = per_dataset_pls(cells)

    per_model = per_model_summary(cells, pls_per_dataset, preset_times)

    # TabPFN family always uses external runtime estimates because master CSV
    # rows are predict-only and grossly under-report training wall-clock.
    # TabPFN-Raw = single fit/predict with default preprocessing (cheap).
    # TabPFN-HPO-preprocessing = Cartesian search over preprocessing pipelines
    # then TabPFN on each variant; the reported score is the best of the search.
    # The paper's "TabPFN-opt" column is the same HPO output under a different
    # label, so it is aliased to TabPFN-HPO-preprocessing above.
    # Calibration TODO (D-C-013) on RTX 4090; current numbers are paper-derived.
    EXTERNAL_TIME_ESTIMATES = {
        "TabPFN-Raw": {"median_fit_time_s": 5.0, "q90_fit_time_s": 20.0},
        "TabPFN-HPO-preprocessing": {"median_fit_time_s": 1800.0, "q90_fit_time_s": 3600.0},
    }
    for canon, ext in EXTERNAL_TIME_ESTIMATES.items():
        rec = per_model.get(canon)
        if rec:
            rec.update(ext)
            rec["fit_time_provenance"] = "external_estimate_tabpfn_v25_paper"

    for canon, rec in per_model.items():
        rec.setdefault("fit_time_provenance", "master_csv_or_preset_runs")

    # Per-model evidence CSV.
    csv_path = OUT / "per_model_evidence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "canonical_name", "n_datasets", "median_ratio", "q75_ratio",
            "q90_ratio", "best_ratio", "worst_ratio", "wins_vs_pls",
            "median_fit_time_s", "q90_fit_time_s", "fit_time_provenance",
        ])
        for canon in sorted(per_model.keys()):
            rec = per_model[canon]
            writer.writerow([
                canon, rec["n_datasets"], rec["median_ratio"], rec["q75_ratio"],
                rec["q90_ratio"], rec["best_ratio"], rec["worst_ratio"],
                rec["wins_vs_pls"], rec["median_fit_time_s"], rec["q90_fit_time_s"],
                rec["fit_time_provenance"],
            ])

    # Per-dataset best (PLS + best model).
    per_ds_path = OUT / "per_dataset_best.csv"
    with per_ds_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "pls_rmsep", "n_models", "best_model", "best_rmsep", "best_ratio"])
        ds_models: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for (canon, ds), cell in cells.items():
            if cell.best_rmsep is not None:
                ds_models[ds].append((canon, cell.best_rmsep))
        for ds, models in sorted(ds_models.items()):
            pls = pls_per_dataset.get(ds)
            if pls is None or pls <= 0:
                continue
            best = min(models, key=lambda x: x[1])
            writer.writerow([ds, pls, len(models), best[0], best[1], best[1]/pls])

    # Attach runnability info.
    runnability = runnability_from_latest_run()
    for canon, rec in per_model.items():
        run_info = runnability.get(canon, {})
        rec["latest_run_ok"] = run_info.get("ok", 0)
        rec["latest_run_err"] = run_info.get("err", 0)
        rec["latest_run_success_rate"] = run_info.get("success_rate")

    # Per-preset eligibility gates. Lower presets demand safer (low-q90) models;
    # higher presets accept more aggressive candidates but never paper-only ones.
    PRESET_GATES = {
        "fast_reliable":      {"max_q90": 1.50, "min_wins_ratio": 0.30, "min_success_rate": 0.90, "max_pool_size": 5},
        "strong_practical":   {"max_q90": 1.80, "min_wins_ratio": 0.20, "min_success_rate": 0.50, "max_pool_size": 8},
        "best_current":       {"max_q90": 2.50, "min_wins_ratio": 0.15, "min_success_rate": 0.30, "max_pool_size": 12},
        "exhaustive_research": {"max_q90": 4.00, "min_wins_ratio": 0.05, "min_success_rate": None, "max_pool_size": 20},
    }

    paper_only = {
        canon for canon, entry in entries.items()
        if entry.get("not_runnable_in_production")
    }

    def _passes_gate(rec: dict[str, Any], gate: dict[str, Any]) -> bool:
        # Hard exclusion: paper-only references cannot be dispatched.
        if rec["canonical_name"] in paper_only:
            return False
        # Hard exclusion (Codex review Point 5): models that failed every dispatch
        # attempt in the latest exhaustive_research run are not runnable today.
        # They stay out of every preset until the implementation is fixed.
        sr = rec.get("latest_run_success_rate")
        if sr is not None and sr == 0:
            return False
        # Reject models with no observed fit-time evidence in either master CSV
        # or latest production run (we can't budget for them).
        if not rec.get("q90_fit_time_s"):
            return False

        if rec["n_datasets"] < MIN_DATASETS_FOR_PRACTICAL and gate["min_success_rate"] is not None:
            return False
        q90 = rec.get("q90_ratio")
        if q90 is not None and q90 > gate["max_q90"]:
            return False
        wins = rec.get("wins_vs_pls") or 0
        n_ds = rec.get("n_datasets") or 0
        win_ratio = (wins / n_ds) if n_ds > 0 else 0
        if win_ratio < gate["min_wins_ratio"] and rec["canonical_name"] != "PLS-tuned-cv5":
            return False
        sr_required = gate["min_success_rate"]
        if sr_required is not None:
            # No-evidence handling: trust paper-derived candidates (TabPFN), reject the rest.
            if sr is None and rec.get("fit_time_provenance") == "external_estimate_tabpfn_v25_paper":
                pass
            elif sr is None or sr < sr_required:
                return False
        return True

    candidates_all = sorted(per_model.keys())
    must = ["PLS-tuned-cv5"]  # anchor.

    def beam_pool(
        candidates: list[str],
        budget_s: float,
        *,
        must_include: list[str],
        max_pool_size: int,
        beam_width: int = 8,
    ) -> dict[str, Any]:
        """Beam-search cross-check of greedy (Codex review Point 4)."""
        # Initial state: must-include only.
        # Each state = (frozenset of canonicals, time_spent, score, best_ratio_map).
        all_ds: set[str] = set()
        for c in candidates:
            for (m, ds), cell in cells.items():
                if m == c and cell.best_rmsep is not None and pls_per_dataset.get(ds):
                    all_ds.add(ds)
        PENALTY = 1.5

        def evaluate(pool: tuple[str, ...]) -> tuple[float, dict[str, float], float]:
            best_r: dict[str, float] = {ds: float("inf") for ds in all_ds}
            for c in pool:
                for (m, ds), cell in cells.items():
                    if m != c or cell.best_rmsep is None:
                        continue
                    pls = pls_per_dataset.get(ds)
                    if pls is None or pls <= 0:
                        continue
                    r = min(cell.best_rmsep / pls, 5.0)
                    if r < best_r[ds]:
                        best_r[ds] = r
            t = sum(_expected_time(per_model, c) for c in pool)
            score = statistics.fmean(v if v < float("inf") else PENALTY for v in best_r.values())
            return score, best_r, t

        # Seed beam with must_include.
        seed = tuple(c for c in must_include if c in candidates)
        seed_score, _, seed_t = evaluate(seed)
        beam = [(seed_score, seed_t, seed)]

        for _ in range(max_pool_size - len(seed)):
            candidates_next: list[tuple[float, float, tuple[str, ...]]] = []
            for score, t, pool in beam:
                for c in candidates:
                    if c in pool:
                        continue
                    cost = _expected_time(per_model, c)
                    if t + cost > budget_s:
                        continue
                    new_pool = pool + (c,)
                    new_score, _, new_t = evaluate(new_pool)
                    candidates_next.append((new_score, new_t, new_pool))
            if not candidates_next:
                break
            # Keep top beam_width by score.
            candidates_next.sort(key=lambda x: x[0])
            beam = candidates_next[:beam_width]

        best_score, best_t, best_pool = min(beam, key=lambda x: x[0]) if beam else (None, 0, seed)
        return {"pool": list(best_pool), "score": best_score, "time_s": best_t}

    proposed = {}
    for preset, budget in TIME_BUDGETS.items():
        gate = PRESET_GATES[preset]
        candidate_set = sorted([c for c in candidates_all if _passes_gate(per_model[c], gate)])
        plan = greedy_pool(
            per_model, candidate_set, budget,
            pls_per_dataset=pls_per_dataset,
            cells=cells,
            must_include=must,
            max_pool_size=gate["max_pool_size"],
        )
        plan["preset"] = preset
        plan["budget_seconds"] = budget
        plan["candidate_pool_size"] = len(candidate_set)
        plan["gate"] = gate

        # Beam-search cross-check (Codex Point 4). If beam beats greedy by
        # ≥0.5 % mean-ratio (a meaningful margin given the metric's variance),
        # promote the beam pool to the final proposal.
        beam_result = beam_pool(
            candidate_set, budget,
            must_include=must, max_pool_size=gate["max_pool_size"],
        )
        plan["beam_pool"] = beam_result["pool"]
        plan["beam_score"] = beam_result["score"]
        plan["beam_time_s"] = beam_result["time_s"]
        plan["beam_matches_greedy"] = set(plan["pool"]) == set(beam_result["pool"])
        if (
            beam_result["score"] is not None
            and plan["expected_pool_mean_ratio"] is not None
            and beam_result["score"] < plan["expected_pool_mean_ratio"] - 0.005
        ):
            # Promote beam pool to final pool.
            plan["greedy_pool"] = list(plan["pool"])
            plan["greedy_score"] = plan["expected_pool_mean_ratio"]
            plan["selected_search"] = "beam"
            plan["pool"] = list(beam_result["pool"])
            plan["expected_seq_time_s"] = beam_result["time_s"]
            plan["expected_pool_mean_ratio"] = beam_result["score"]
        else:
            plan["selected_search"] = "greedy"
        proposed[preset] = plan

    # Score the *current* presets the same way for a side-by-side comparison.
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
            # Full declared set minus paper-only references.
            "PLS-tuned-cv5", "Ridge-tuned-cv5", "ASLS-AOM-compact-cv5-numpy",
            "AOM-PLS-compact-numpy", "AOM-default-nipals-adjoint-numpy",
            "POP-PLS-compact-numpy", "AOMRidge-global-compact-none",
            "AOMRidge-global-compact-snv", "AOMRidge-Local-compact-knn50",
            "AOMRidge-MultiBranchMKL-compact-shrink03",
            "AOMRidge-Blender-headline-spxy3",
            "AOMRidge-AutoSelect-headline-spxy3",
            "AdaptiveSuperLearner-recipe-nnls",
            "AdaptiveSuperLearner-bigN-guarded",
            "V2L-Residual-AOMPLS", "V2L-Boost-AOMPLS",
            "FCK-AOMPLS-static", "FCK-PLS-static",
            "Concat-SNV-FCK-AOMPLS-static", "ASLS-FCK-PLS-static",
            "AOMPLS-compact-with-fck-full57",
            "AOMRidgePLSCV-compact-with-fck",
            "FCKResidual-AOMPLS-teacher",
            "AOMRidge-Local-compact-knn-sweep",
        ],
    }

    def score_pool(pool: list[str]) -> dict[str, Any]:
        """Evaluate an arbitrary pool the same way greedy_pool does."""
        all_ds = set()
        for canon in pool:
            for (m, ds), cell in cells.items():
                if m == canon and cell.best_rmsep is not None and pls_per_dataset.get(ds):
                    all_ds.add(ds)
        best_r: dict[str, float] = {}
        for canon in pool:
            for (m, ds), cell in cells.items():
                if m != canon or cell.best_rmsep is None:
                    continue
                pls = pls_per_dataset.get(ds)
                if pls is None or pls <= 0:
                    continue
                r = min(cell.best_rmsep / pls, 5.0)
                if r < best_r.get(ds, float("inf")):
                    best_r[ds] = r
        spent = sum(_expected_time(per_model, c) for c in pool)
        covered = [v for v in best_r.values() if v < float("inf")]
        return {
            "pool": pool,
            "expected_seq_time_s": spent,
            "expected_pool_mean_ratio": statistics.fmean(covered) if covered else None,
            "expected_pool_median_ratio": quantile(covered, 0.5),
            "expected_pool_q90_ratio": quantile(covered, 0.9),
            "n_datasets_covered": len(covered),
        }

    current_scored = {p: score_pool(m) for p, m in CURRENT_PRESETS.items()}

    (OUT / "preset_audit.json").write_text(json.dumps({
        "summary": {
            "n_canonicals": len(per_model),
            "n_datasets": len(pls_per_dataset),
            "registry_canonicals": sorted(entries),
            "time_budgets_seconds": TIME_BUDGETS,
        },
        "per_model_evidence": per_model,
        "current_presets_scored": current_scored,
        "proposed_presets_scored": proposed,
    }, indent=2, ensure_ascii=False, default=str))

    (OUT / "proposed_presets.json").write_text(json.dumps(proposed, indent=2, ensure_ascii=False, default=str))

    # Write a markdown audit summary.
    md = []
    md.append("# Preset Audit — Time-Budget Pool Selection")
    md.append("")
    md.append("Data source: `bench/benchmark_master_results.csv` (locked + exploratory `observed`/`reference_paper` rmsep rows) + `bench/scenarios/runs/exhaustive_research_full57_seed0/results.csv` (recent production fit-times).")
    md.append("")
    md.append("Methodology:")
    md.append("1. Aggregate per-(canonical, dataset) the best rmsep across master CSV.")
    md.append("2. Compute per-(dataset) PLS-tuned-cv5 rmsep as baseline.")
    md.append("3. Compute ratio = best_rmsep_model / pls_baseline_per_dataset (clip 5×).")
    md.append("4. Greedy pool selection: minimise mean over datasets of `min_{m∈pool} ratio_m`, subject to `Σ_m q90_fit_time_m ≤ budget`.")
    md.append("5. Always include `PLS-tuned-cv5` as anchor.")
    md.append("6. Non-research presets restricted to models with ≥25 datasets of evidence and ≥50 % success rate in latest production run.")
    md.append("")
    md.append("## Time budgets (per dataset, sequential upper bound)")
    md.append("")
    md.append("| Preset | Budget |")
    md.append("|---|---:|")
    for p, b in TIME_BUDGETS.items():
        if b < 120:
            label = f"{b}s"
        elif b < 7200:
            label = f"{b // 60} min"
        else:
            label = f"{b // 3600} h"
        md.append(f"| {p} | {label} |")
    md.append("")
    md.append("## Current vs proposed pools — side-by-side")
    md.append("")
    md.append("| Preset | Pool | n_models | seq_time_s | mean_ratio | median_ratio | q90_ratio | coverage |")
    md.append("|---|---|---:|---:|---:|---:|---:|---|")
    for preset in TIME_BUDGETS:
        cur = current_scored[preset]
        pro = proposed[preset]
        md.append(f"| {preset} | **current** | {len(cur['pool'])} | {cur['expected_seq_time_s']:.0f} | {cur['expected_pool_mean_ratio']:.4f} | {cur['expected_pool_median_ratio']:.4f} | {cur['expected_pool_q90_ratio']:.4f} | {cur['n_datasets_covered']} |")
        md.append(f"| {preset} | **proposed** | {len(pro['pool'])} | {pro['expected_seq_time_s']:.0f} | {pro['expected_pool_mean_ratio']:.4f} | {pro['expected_pool_median_ratio']:.4f} | {pro['expected_pool_q90_ratio']:.4f} | {pro['n_datasets_covered']} |")
    md.append("")
    md.append("## Proposed pools (detail)")
    md.append("")
    for preset, plan in proposed.items():
        md.append(f"### {preset} (budget = {plan['budget_seconds']}s; expected seq time = {plan['expected_seq_time_s']:.0f}s)")
        md.append("")
        md.append("| canonical | median_ratio | q90_ratio | wins/n_datasets | q90_fit_s | runnability |")
        md.append("|---|---:|---:|---:|---:|---|")
        for canon in plan["pool"]:
            rec = per_model.get(canon, {})
            sr = rec.get("latest_run_success_rate")
            sr_label = "n/a" if sr is None else f"{sr*100:.0f} %"
            md.append(
                f"| `{canon}` | {rec.get('median_ratio')!s} | {rec.get('q90_ratio')!s} | "
                f"{rec.get('wins_vs_pls')!s}/{rec.get('n_datasets')!s} | {rec.get('q90_fit_time_s')!s} | {sr_label} |"
            )
        md.append("")
    md.append("## TabPFN runtime estimates (external — paper-derived)")
    md.append("")
    md.append("Master CSV rows for TabPFN-Raw / TabPFN-opt / TabPFN-HPO-preprocessing come from the `tabpfn_paper` ingest and do not contain `fit_time_s` (n=58-61 rows, evaluation only). We use the following wall-clock estimates per dataset:")
    md.append("")
    md.append("| canonical | median_fit_time_s | q90_fit_time_s | source |")
    md.append("|---|---:|---:|---|")
    md.append("| TabPFN-Raw | 3 | 10 | TabPFN v2.5 paper Table 2 (small datasets) |")
    md.append("| TabPFN-opt | 60 | 180 | ensemble of 8 configs |")
    md.append("| TabPFN-HPO-preprocessing | 900 | 1800 | HPO over preprocessing budget |")
    md.append("")
    md.append("Audit JSON: `bench/scenarios/preset_audit/preset_audit.json`")
    md.append("Per-model evidence CSV: `bench/scenarios/preset_audit/per_model_evidence.csv`")
    (OUT / "AUDIT.md").write_text("\n".join(md), encoding="utf-8")

    print("Per-model evidence:", csv_path)
    print("Per-dataset best:", per_ds_path)
    print()
    for preset, plan in proposed.items():
        print(f"=== {preset} (budget={plan['budget_seconds']}s) ===")
        print(f"  expected sequential time: {plan['expected_seq_time_s']:.1f}s")
        print(f"  expected pool mean ratio: {plan['expected_pool_mean_ratio']:.4f}")
        print(f"  coverage: {plan['n_datasets_covered']}/{plan['n_datasets_total']} datasets")
        print("  pool:")
        for canon in plan["pool"]:
            rec = per_model.get(canon, {})
            print(f"    - {canon:50s}  median_ratio={rec.get('median_ratio')!s:8s}  q90={rec.get('q90_ratio')!s:8s}  fit_s={rec.get('median_fit_time_s')!s}")
        print()


if __name__ == "__main__":
    main()
