"""bench/export_benchmark_scenarios.py.

Reads:
    - bench/scenarios/model_registry.yaml
    - bench/benchmark_master_results.csv
    - bench/Subset_analysis/subset_transfer_summary.csv (optional)
    - bench/MASTER_CSV_HASH.txt (for provenance)

Writes:
    - bench/scenarios/fast_reliable.json
    - bench/scenarios/strong_practical.json
    - bench/scenarios/best_current.json
    - bench/scenarios/exhaustive_research.json
    - bench/scenarios/README.md

Each scenario JSON is a manifest the harness can consume directly:

    {
      "preset": "...",
      "registry_source": "...",
      "master_csv_sha256": "...",
      "candidates": [
          {"canonical_name": "...",
           "config_template": "...",
           "evidence": {...},
           "penalties": [...]},
          ...
      ],
      "ordering_hint": [...],
      "budget": {...}
    }

The exporter applies the C2 penalty system from
bench/PLAN_REPRISE_2026-05.md §8:

    - low coverage relative to cohort size,
    - non-nested selectors entering strong_practical / best_current,
    - q90 toxic (clipped score-ratio above 1.5),
    - canonical_name absent from master CSV (no evidence),
    - source-run-only models without `supports_predefined_test_split=true`.

DECISION_PENDING_CODEX_REVIEW (D-C-005).
The exact penalty thresholds and the resulting candidate ordering are
provisional. See bench/SYNC.md for the queued review.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

try:  # pragma: no cover - import guard
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required to load the model registry. "
        "Install it via `pip install pyyaml`."
    ) from exc


BENCH = Path(__file__).resolve().parent
REGISTRY_PATH = BENCH / "scenarios" / "model_registry.yaml"
MASTER_PATH = BENCH / "benchmark_master_results.csv"
HASH_PATH = BENCH / "MASTER_CSV_HASH.txt"
SUBSET_PATH = BENCH / "Subset_analysis" / "subset_transfer_summary.csv"
OUT_DIR = BENCH / "scenarios"
PRESETS = ("fast_reliable", "strong_practical", "best_current", "exhaustive_research")
SCHEMA_VERSION = "0.1.0"

RUNTIME_TIER_ORDER = ("fast", "medium", "slow", "very_slow")
MATURITY_RANK = {"oracle": -1, "local_not_master": -1, "legacy": -1, "exploratory": 0, "locked": 1}

LOW_COVERAGE_FRACTION = 0.40   # < 40% of cohort -> low_coverage penalty
TOXIC_Q90_THRESHOLD = 1.50     # q90 of clipped score-ratio above 1.5 -> q90_toxic
COHORT_SIZE_DEFAULT = 57


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def as_float(value: Any) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    text = text.replace("%", "").replace(",", ".")
    text = re.sub(r"[^0-9eE+\-\.]", "", text)
    if text in {"", ".", "+", "-"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def normalise_label(value: str) -> str:
    return clean_text(value).lower().replace("_", "-").replace(" ", "-")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_hash_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s+", line):
            continue
        key, _, value = line.partition(" ")
        out[key.strip()] = value.strip()
    return out


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegistryEntry:
    canonical_name: str
    aliases: tuple[str, ...]
    model_class: str
    module: str
    config_template: str
    task_types: tuple[str, ...]
    input_constraints: dict[str, Any]
    supports_predefined_test_split: bool
    inner_cv_nested: bool
    runtime_tier: str
    maturity: str
    notes: str
    not_runnable_in_production: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegistryEntry:
        return cls(
            canonical_name=clean_text(data["canonical_name"]),
            aliases=tuple(clean_text(a) for a in data.get("aliases", [])),
            model_class=clean_text(data.get("model_class", "")),
            module=clean_text(data.get("module", "")),
            config_template=clean_text(data.get("config_template", "")),
            task_types=tuple(clean_text(t) for t in data.get("task_types", [])),
            input_constraints=dict(data.get("input_constraints") or {}),
            supports_predefined_test_split=bool(data.get("supports_predefined_test_split", True)),
            inner_cv_nested=bool(data.get("inner_cv_nested", False)),
            runtime_tier=clean_text(data.get("runtime_tier", "medium")),
            maturity=clean_text(data.get("maturity", "exploratory")),
            notes=clean_text(data.get("notes", "")),
            not_runnable_in_production=bool(data.get("not_runnable_in_production", False)),
        )


@dataclass
class ModelEvidence:
    canonical_name: str
    n_rows_locked: int = 0
    n_rows_total: int = 0
    n_datasets: int = 0
    coverage_fraction: float = 0.0
    median_rel_source_pls: float | None = None
    q75_rel_source_pls: float | None = None
    q90_rel_source_pls: float | None = None
    median_rel_dataset_pls: float | None = None
    q75_rel_dataset_pls: float | None = None
    q90_rel_dataset_pls: float | None = None
    worst_clipped_ratio: float | None = None
    wins_vs_source_pls: int = 0
    wins_vs_dataset_pls: int = 0
    median_fit_time_s: float | None = None
    q90_fit_time_s: float | None = None
    source_families: list[str] = field(default_factory=list)
    source_runs: list[str] = field(default_factory=list)
    maturities_observed: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Registry root must be a mapping; got {type(data).__name__}")
    if "models" not in data or "presets" not in data:
        raise SystemExit("Registry must define `models` and `presets` keys.")
    return data


def build_alias_index(entries: list[RegistryEntry]) -> dict[str, RegistryEntry]:
    """Map every alias and canonical name (normalised) to its entry."""
    index: dict[str, RegistryEntry] = {}
    for entry in entries:
        for alias in (entry.canonical_name, *entry.aliases):
            key = normalise_label(alias)
            if not key:
                continue
            previous = index.get(key)
            if previous is not None and previous.canonical_name != entry.canonical_name:
                raise SystemExit(
                    f"Alias collision: '{alias}' is mapped to both "
                    f"'{previous.canonical_name}' and '{entry.canonical_name}'."
                )
            index[key] = entry
    return index


# ---------------------------------------------------------------------------
# Master CSV ingestion
# ---------------------------------------------------------------------------


def iter_master_rows(path: Path = MASTER_PATH) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def gather_evidence(
    rows: list[dict[str, str]],
    alias_index: dict[str, RegistryEntry],
) -> dict[str, ModelEvidence]:
    """Aggregate per-canonical-name statistics from the master CSV."""
    by_canonical: dict[str, ModelEvidence] = {}
    rel_source: dict[str, dict[str, float]] = defaultdict(dict)
    rel_dataset: dict[str, dict[str, float]] = defaultdict(dict)
    fit_times: dict[str, list[float]] = defaultdict(list)
    families: dict[str, set[str]] = defaultdict(set)
    runs: dict[str, set[str]] = defaultdict(set)
    maturities: dict[str, Counter] = defaultdict(Counter)
    datasets: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        record_type = row.get("record_type", "")
        if record_type not in {"observed", "reference_paper"}:
            continue
        if (row.get("status") or "ok").lower() not in {"", "ok", "success", "done", "complete", "completed"}:
            continue
        if (row.get("evaluation_split") or "").lower() in {"train", "cv", "cross_val", "cross-validation", "cros val"}:
            continue
        label = clean_text(row.get("model_name") or row.get("variant"))
        key = normalise_label(label)
        entry = alias_index.get(key)
        if entry is None:
            continue
        canonical = entry.canonical_name
        evidence = by_canonical.setdefault(canonical, ModelEvidence(canonical_name=canonical))

        evidence.n_rows_total += 1
        maturity = clean_text(row.get("protocol_maturity"))
        maturities[canonical][maturity] += 1
        if maturity == "locked":
            evidence.n_rows_locked += 1

        dataset = clean_text(row.get("dataset"))
        if dataset:
            datasets[canonical].add(dataset)

        rs = as_float(row.get("score_ratio_vs_source_run_pls"))
        rd = as_float(row.get("score_ratio_vs_dataset_pls"))
        if dataset:
            if rs is not None:
                previous = rel_source[canonical].get(dataset)
                if previous is None or rs < previous:
                    rel_source[canonical][dataset] = rs
            if rd is not None:
                previous = rel_dataset[canonical].get(dataset)
                if previous is None or rd < previous:
                    rel_dataset[canonical][dataset] = rd

        ft = as_float(row.get("fit_time_s"))
        if ft is not None and ft > 0:
            fit_times[canonical].append(ft)

        sf = clean_text(row.get("source_family"))
        if sf:
            families[canonical].add(sf)
        sr = clean_text(row.get("source_run"))
        if sr:
            runs[canonical].add(sr)

    for canonical, evidence in by_canonical.items():
        evidence.n_datasets = len(datasets[canonical])
        source_vals = sorted(rel_source[canonical].values())
        dataset_vals = sorted(rel_dataset[canonical].values())
        evidence.median_rel_source_pls = _safe_quantile(source_vals, 0.5)
        evidence.q75_rel_source_pls = _safe_quantile(source_vals, 0.75)
        evidence.q90_rel_source_pls = _safe_quantile(source_vals, 0.90)
        evidence.median_rel_dataset_pls = _safe_quantile(dataset_vals, 0.5)
        evidence.q75_rel_dataset_pls = _safe_quantile(dataset_vals, 0.75)
        evidence.q90_rel_dataset_pls = _safe_quantile(dataset_vals, 0.90)
        evidence.worst_clipped_ratio = _worst_clipped(source_vals or dataset_vals)
        evidence.wins_vs_source_pls = sum(1 for v in source_vals if v < 1.0)
        evidence.wins_vs_dataset_pls = sum(1 for v in dataset_vals if v < 1.0)
        if fit_times[canonical]:
            evidence.median_fit_time_s = _safe_quantile(sorted(fit_times[canonical]), 0.5)
            evidence.q90_fit_time_s = _safe_quantile(sorted(fit_times[canonical]), 0.9)
        evidence.source_families = sorted(families[canonical])
        evidence.source_runs = sorted(runs[canonical])
        evidence.maturities_observed = [
            f"{tag}:{count}" for tag, count in sorted(maturities[canonical].items(), key=lambda kv: -kv[1])
        ]
    return by_canonical


def _safe_quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    return statistics.quantiles(sorted_values, n=100, method="inclusive")[max(0, min(98, int(q * 100) - 1))]


def _worst_clipped(values: list[float]) -> float | None:
    if not values:
        return None
    return min(max(values), 5.0)


# ---------------------------------------------------------------------------
# Penalty system
# ---------------------------------------------------------------------------


def evaluate_penalties(
    entry: RegistryEntry,
    evidence: ModelEvidence | None,
    preset: str,
    cohort_size: int,
) -> list[str]:
    penalties: list[str] = []
    strict_presets = {"strong_practical", "best_current"}

    if evidence is None or evidence.n_rows_locked == 0:
        penalties.append("no_locked_evidence_in_master")
    if evidence is not None and cohort_size > 0:
        evidence.coverage_fraction = evidence.n_datasets / cohort_size
        if evidence.coverage_fraction < LOW_COVERAGE_FRACTION:
            penalties.append(
                f"low_coverage<{LOW_COVERAGE_FRACTION:.0%}"
                f"({evidence.n_datasets}/{cohort_size})"
            )

    if preset in strict_presets and not entry.inner_cv_nested:
        penalties.append("non_nested_selector")

    if (
        evidence is not None
        and evidence.q90_rel_source_pls is not None
        and evidence.q90_rel_source_pls > TOXIC_Q90_THRESHOLD
    ):
        penalties.append(f"q90_toxic({evidence.q90_rel_source_pls:.2f})")

    if not entry.supports_predefined_test_split:
        penalties.append("no_predefined_test_split")

    if entry.maturity == "exploratory" and preset != "exhaustive_research":
        penalties.append("exploratory_in_non_research_preset")

    if entry.not_runnable_in_production:
        penalties.append("not_runnable_in_production")

    return penalties


# ---------------------------------------------------------------------------
# Manifest assembly
# ---------------------------------------------------------------------------


def build_manifest(
    preset: str,
    preset_spec: dict[str, Any],
    entries_by_name: dict[str, RegistryEntry],
    evidence_by_name: dict[str, ModelEvidence],
    master_meta: dict[str, str],
    cohort_size: int,
) -> dict[str, Any]:
    members: list[str] = list(preset_spec.get("members", []))
    candidates: list[dict[str, Any]] = []
    missing: list[str] = []

    for name in members:
        entry = entries_by_name.get(name)
        if entry is None:
            missing.append(name)
            continue
        evidence = evidence_by_name.get(name)
        penalties = evaluate_penalties(entry, evidence, preset, cohort_size)
        candidates.append(
            {
                "canonical_name": entry.canonical_name,
                "aliases": list(entry.aliases),
                "model_class": entry.model_class,
                "module": entry.module,
                "config_template": entry.config_template,
                "task_types": list(entry.task_types),
                "input_constraints": entry.input_constraints,
                "supports_predefined_test_split": entry.supports_predefined_test_split,
                "inner_cv_nested": entry.inner_cv_nested,
                "runtime_tier": entry.runtime_tier,
                "maturity": entry.maturity,
                "not_runnable_in_production": entry.not_runnable_in_production,
                "notes": entry.notes,
                "evidence": _evidence_dict(evidence),
                "penalties": penalties,
            }
        )

    candidates.sort(key=lambda c: _ordering_key(c))
    ordering_hint = [c["canonical_name"] for c in candidates]

    return {
        "schema_version": SCHEMA_VERSION,
        "preset": preset,
        "description": preset_spec.get("description", ""),
        "generated_on": date.today().isoformat(),
        "generated_by": __file__,
        "registry_source": str(REGISTRY_PATH.relative_to(BENCH.parent)),
        "master_csv_source": str(MASTER_PATH.relative_to(BENCH.parent)),
        "master_csv_sha256": master_meta.get("sha256", ""),
        "runtime_tier_max": preset_spec.get("runtime_tier_max", "very_slow"),
        "maturity_min": preset_spec.get("maturity_min", "locked"),
        "cohort_size_assumed": cohort_size,
        "missing_registry_entries": missing,
        "candidates": candidates,
        "ordering_hint": ordering_hint,
        "budget": _preset_budget(preset),
        "expected_baselines": _expected_baselines(preset),
        "codex_review_status": "DECISION_PENDING_CODEX_REVIEW",
    }


def _evidence_dict(evidence: ModelEvidence | None) -> dict[str, Any]:
    if evidence is None:
        return {"present_in_master": False}
    return {
        "present_in_master": True,
        "n_rows_locked": evidence.n_rows_locked,
        "n_rows_total": evidence.n_rows_total,
        "n_datasets": evidence.n_datasets,
        "coverage_fraction_raw": round(evidence.coverage_fraction, 4),  # D-C-005a: renamed for clarity (raw, not clamped); add coverage_fraction_clamped if/when needed
        "median_rel_source_pls": _round(evidence.median_rel_source_pls),
        "q75_rel_source_pls": _round(evidence.q75_rel_source_pls),
        "q90_rel_source_pls": _round(evidence.q90_rel_source_pls),
        "median_rel_dataset_pls": _round(evidence.median_rel_dataset_pls),
        "q75_rel_dataset_pls": _round(evidence.q75_rel_dataset_pls),
        "q90_rel_dataset_pls": _round(evidence.q90_rel_dataset_pls),
        "worst_clipped_ratio": _round(evidence.worst_clipped_ratio),
        "wins_vs_source_pls": evidence.wins_vs_source_pls,
        "wins_vs_dataset_pls": evidence.wins_vs_dataset_pls,
        "median_fit_time_s": _round(evidence.median_fit_time_s, digits=3),
        "q90_fit_time_s": _round(evidence.q90_fit_time_s, digits=3),
        "source_families": evidence.source_families,
        "source_runs": evidence.source_runs,
        "maturities_observed": evidence.maturities_observed,
    }


def _round(value: float | None, *, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def _ordering_key(candidate: dict[str, Any]) -> tuple[int, int, float]:
    runtime = candidate.get("runtime_tier", "medium")
    runtime_idx = RUNTIME_TIER_ORDER.index(runtime) if runtime in RUNTIME_TIER_ORDER else len(RUNTIME_TIER_ORDER)
    has_no_evidence = 0 if candidate["evidence"].get("present_in_master") else 1
    median = candidate["evidence"].get("median_rel_source_pls")
    median = median if isinstance(median, (int, float)) else 9.99
    return (has_no_evidence, runtime_idx, median)


def _preset_budget(preset: str) -> dict[str, str]:
    return {
        "fast_reliable": {"target_runtime": "seconds .. few minutes per dataset", "guidance": "use full57 cohort if compute allows"},
        "strong_practical": {"target_runtime": "minutes .. <1h per dataset", "guidance": "TabPFN gated by (n<=5000, p<=1000)"},
        "best_current": {"target_runtime": "1-3h per dataset", "guidance": "include TabPFN-HPO + top AOM* + top AOMRidge*"},
        "exhaustive_research": {"target_runtime": "multi-hour / overnight per dataset", "guidance": "exploratory rows allowed"},
    }.get(preset, {"target_runtime": "unspecified"})


def _expected_baselines(preset: str) -> dict[str, str]:
    return {
        "fast_reliable": {"reference": "ASLS-AOM-compact-cv5-numpy median 0.96 vs PLS-std (AOM_v0/Summary §6)"},
        "strong_practical": {"reference": "TabPFN-opt and AOMRidge variants beat PLS on majority of cohort"},
        "best_current": {"reference": "TabPFN-HPO-preprocessing best paper baseline"},
        "exhaustive_research": {"reference": "Full multi-strategy comparison; see bench/benchmark_synthesis.md"},
    }.get(preset, {})


# ---------------------------------------------------------------------------
# Subset transfer report (informational only)
# ---------------------------------------------------------------------------


def subset_transfer_summary(path: Path = SUBSET_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out.append({k: row.get(k) for k in row})
    return out


# ---------------------------------------------------------------------------
# README writer
# ---------------------------------------------------------------------------


def write_readme(manifests: dict[str, dict[str, Any]], master_meta: dict[str, str]) -> None:
    lines: list[str] = []
    lines.append("# bench/scenarios/")
    lines.append("")
    lines.append(
        "Auto-generated manifests describing the four benchmark presets defined in "
        "`bench/PLAN_REPRISE_2026-05.md` §9. Each JSON file is consumed by "
        "`bench/harness/run_benchmark.py`; do not hand-edit them."
    )
    lines.append("")
    lines.append(f"Master CSV SHA256: `{master_meta.get('sha256', '<unknown>')}`")
    lines.append("Generator: `bench/export_benchmark_scenarios.py`")
    lines.append(f"Generated on: {date.today().isoformat()}")
    lines.append("")
    lines.append("## Presets")
    lines.append("")
    lines.append("| Preset | Description | Members | Penalised |")
    lines.append("|---|---|---:|---:|")
    for preset in PRESETS:
        manifest = manifests.get(preset, {})
        members = manifest.get("candidates", [])
        penalised = sum(1 for c in members if c.get("penalties"))
        lines.append(
            f"| `{preset}` | {manifest.get('description', '')} | {len(members)} | {penalised} |"
        )
    lines.append("")
    lines.append("## Codex review")
    lines.append("")
    lines.append(
        "Every manifest carries `codex_review_status: DECISION_PENDING_CODEX_REVIEW` "
        "until the registry, the penalty thresholds, and the candidate ordering are "
        "validated through `bench/SYNC.md`."
    )
    lines.append("")
    lines.append("## Schema")
    lines.append("")
    lines.append(
        "The exporter follows the schema documented at the top of "
        "`bench/scenarios/model_registry.yaml` and the manifest layout described in "
        "`bench/export_benchmark_scenarios.py`."
    )
    lines.append("")
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    entries = [RegistryEntry.from_dict(item) for item in registry["models"]]
    entries_by_name = {entry.canonical_name: entry for entry in entries}
    alias_index = build_alias_index(entries)

    rows = iter_master_rows()
    evidence_by_name = gather_evidence(rows, alias_index)

    master_meta: dict[str, str] = parse_hash_file(HASH_PATH)
    if "sha256" not in master_meta:
        master_meta["sha256"] = file_sha256(MASTER_PATH)

    cohort_size = COHORT_SIZE_DEFAULT
    cohorts = registry.get("cohorts", []) or []
    for cohort in cohorts:
        if cohort.get("name") == "full57":
            cohort_size = int(cohort.get("n_datasets", COHORT_SIZE_DEFAULT))
            break

    manifests: dict[str, dict[str, Any]] = {}
    for preset in PRESETS:
        spec = registry["presets"].get(preset)
        if spec is None:
            print(f"[warn] preset '{preset}' missing from registry; skipping.")
            continue
        manifest = build_manifest(
            preset=preset,
            preset_spec=spec,
            entries_by_name=entries_by_name,
            evidence_by_name=evidence_by_name,
            master_meta=master_meta,
            cohort_size=cohort_size,
        )
        manifests[preset] = manifest
        out_path = OUT_DIR / f"{preset}.json"
        out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"Wrote {out_path.relative_to(BENCH.parent)} "
            f"({len(manifest['candidates'])} candidates, "
            f"{sum(1 for c in manifest['candidates'] if c['penalties'])} penalised)"
        )

    write_readme(manifests, master_meta)
    print(f"Wrote {(OUT_DIR / 'README.md').relative_to(BENCH.parent)}")


if __name__ == "__main__":
    main()
