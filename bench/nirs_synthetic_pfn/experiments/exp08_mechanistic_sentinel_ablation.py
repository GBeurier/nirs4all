"""R2a sentinel mechanistic profile ablation (bench-only, report-only).

This experiment runs a non-gate, report-only ablation over fixed mechanistic
profiles defined in ``builder_adapter.R2A_MECHANISTIC_PROFILES``. Profiles are
applied post-generation as deterministic mechanistic approximations driven by
profile name + seed. They never read real spectra, labels, splits, targets, or
AUC; they never modify gate thresholds or metric definitions.

This experiment must NOT be interpreted as B2/B3 evidence. It is a separate
diagnostic lane that compares baseline synthetic spectra to per-profile
synthetic spectra against the same real cohort, in the ``uncalibrated_raw``
lane only.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from exp00_smoke_prior_dataset import PRESETS
from exp02_real_synthetic_scorecards import (
    _stable_dataset_seed,
    build_on_demand_synthetic_run_for_dataset,
    select_synthetic_preset_for_dataset,
)
from nirsyntheticpfn.adapters.builder_adapter import (
    R2A_MECHANISTIC_PROFILES,
    _apply_r2a_mechanistic_profile,
)
from nirsyntheticpfn.evaluation.realism import (
    PROVISIONAL_THRESHOLDS,
    RealDataset,
    align_to_real_grid,
    compare_real_synthetic,
    discover_local_real_datasets,
    is_index_fallback_grid,
    load_real_spectra,
    sanitize_finite_spectra,
)

PRIMARY_SENTINEL_TOKENS: tuple[str, ...] = ("BEER", "DIESEL", "CORN")
SECONDARY_MILK_SENTINEL_TOKENS: tuple[str, ...] = ("MILK",)
SECONDARY_SOIL_SENTINEL_TOKENS: tuple[str, ...] = (
    "LUCAS",
    "PHOSPHORUS",
    "MANURE",
    "SOIL",
)
SECONDARY_FRUIT_SENTINEL_TOKENS: tuple[str, ...] = (
    "BERRY",
    "PEACH",
    "PLUMS",
    "FRUIT",
)

# Priority groups (lower index = higher priority). Custom user-supplied tokens
# that do not belong to any of these groups fall into an implicit final group
# whose priority equals ``len(SENTINEL_PRIORITY_GROUPS)``.
SENTINEL_PRIORITY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("primary", PRIMARY_SENTINEL_TOKENS),
    ("secondary_milk", SECONDARY_MILK_SENTINEL_TOKENS),
    ("secondary_soil", SECONDARY_SOIL_SENTINEL_TOKENS),
    ("secondary_fruit", SECONDARY_FRUIT_SENTINEL_TOKENS),
)
DEFAULT_SENTINEL_TOKENS: tuple[str, ...] = tuple(
    token for _, group in SENTINEL_PRIORITY_GROUPS for token in group
)
# Default cap chosen to fit every named primary sentinel cohort row in the
# current AOM benchmarks (2 BEER + 3 DIESEL + 2 CORN = 7) plus a single slot
# for the highest-priority secondary cohort. Override via ``--max-sentinel-datasets``.
DEFAULT_MAX_SENTINEL_DATASETS = 8
DEFAULT_REPORT = Path("bench/nirs_synthetic_pfn/reports/r2a_mechanistic_sentinel_ablation.md")
DEFAULT_CSV = Path("bench/nirs_synthetic_pfn/reports/r2a_mechanistic_sentinel_ablation.csv")
R2A_AUDIT_SCOPE = "bench_only_r2a_sentinel_mechanistic_ablation"
R2A_AUDIT_FLAG_KEYS = (
    "oracle",
    "label_inputs_used",
    "target_inputs_used",
    "split_inputs_used",
    "source_oracle_used",
    "learned",
    "real_stat_capture",
    "thresholds_modified",
    "metrics_modified",
    "imputed",
    "replays_real_rows",
)


@dataclass(frozen=True)
class AblationRow:
    """One R2a ablation row for a (dataset, profile) pair in uncalibrated_raw lane."""

    status: str  # "compared" | "blocked"
    source: str
    task: str
    dataset: str
    synthetic_preset: str
    mechanistic_profile: str
    profile_enabled: bool
    profile_seed: int | None
    audit_oracle: bool
    audit_label_inputs_used: bool
    audit_target_inputs_used: bool
    audit_split_inputs_used: bool
    audit_source_oracle_used: bool
    audit_learned: bool
    audit_real_stat_capture: bool
    audit_thresholds_modified: bool
    audit_metrics_modified: bool
    audit_imputed: bool
    audit_replays_real_rows: bool
    profile_input_seed: int | None
    profile_scope: str
    profile_transform_params: str
    n_real_samples: int
    n_synthetic_samples: int
    n_wavelengths: int
    adversarial_auc: float | None
    pca_overlap: float | None
    nearest_neighbor_ratio: float | None
    derivative_log10_gap: float | None
    blocked_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-synthetic-samples", type=int, default=64)
    parser.add_argument("--max-real-samples", type=int, default=64)
    parser.add_argument(
        "--max-sentinel-datasets",
        type=int,
        default=DEFAULT_MAX_SENTINEL_DATASETS,
        help=(
            "Cap on sentinel datasets after priority-based selection. "
            "0 (or any non-positive value) means every runnable local cohort row, "
            "with no token filter or priority sort applied."
        ),
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default=",".join(R2A_MECHANISTIC_PROFILES),
        help="comma-separated list of R2a profile names.",
    )
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument(
        "--sentinel-tokens",
        type=str,
        default=",".join(DEFAULT_SENTINEL_TOKENS),
        help=(
            "Comma-separated case-insensitive tokens used to select sentinel datasets "
            "by matching against `source`, `task`, `database_name`, or `dataset`. "
            "Only applied when `--max-sentinel-datasets > 0`."
        ),
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    profiles = _parse_profiles(args.profiles)
    sentinel_tokens = _parse_sentinel_tokens(args.sentinel_tokens)
    root = _repo_root()
    result = run_ablation(
        root=root,
        profiles=profiles,
        n_synthetic_samples=args.n_synthetic_samples,
        max_real_samples=args.max_real_samples,
        max_sentinel_datasets=args.max_sentinel_datasets,
        seed=args.seed,
        sentinel_tokens=sentinel_tokens,
    )
    write_csv(result["rows"], args.csv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render_markdown(
            result=result,
            report_path=args.report,
            csv_path=args.csv,
            profiles=profiles,
            n_synthetic_samples=args.n_synthetic_samples,
            max_real_samples=args.max_real_samples,
            max_sentinel_datasets=args.max_sentinel_datasets,
            seed=args.seed,
            sentinel_tokens=sentinel_tokens,
        ),
        encoding="utf-8",
    )
    print(args.report)
    print(args.csv)


def run_ablation(
    *,
    root: Path,
    profiles: list[str],
    n_synthetic_samples: int,
    max_real_samples: int,
    max_sentinel_datasets: int,
    seed: int,
    sentinel_tokens: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the R2a ablation over (dataset, profile) pairs in uncalibrated_raw lane only.

    The ``"r2a_baseline"`` profile is always reported alongside requested profiles
    as the no-op control. The lane is uncalibrated_raw only; no marginal
    calibration, no SNV, no covariance calibration is applied.
    """
    requested = list(dict.fromkeys(profiles))
    if "r2a_baseline" not in requested:
        requested = ["r2a_baseline", *requested]
    for profile in requested:
        if profile not in R2A_MECHANISTIC_PROFILES:
            raise ValueError(
                f"unknown R2a profile {profile!r}; valid: {list(R2A_MECHANISTIC_PROFILES)}"
            )

    tokens = (
        tuple(DEFAULT_SENTINEL_TOKENS)
        if sentinel_tokens is None
        else tuple(sentinel_tokens)
    )
    real_datasets, _ = discover_local_real_datasets(root)
    if max_sentinel_datasets <= 0:
        # Documented behaviour: score every runnable cohort row, no token filter.
        sentinel_candidates = list(real_datasets)
        selected = list(real_datasets)
    else:
        sentinel_candidates = _select_sentinel_datasets(real_datasets, tokens)
        selected = sentinel_candidates[:max_sentinel_datasets]
    rows: list[AblationRow] = []
    if not selected:
        return {
            "status": "blocked_no_real_data",
            "rows": rows,
            "real_runnable_count": len(real_datasets),
            "real_sentinel_candidate_count": len(sentinel_candidates),
            "real_selected_count": 0,
            "profiles": requested,
            "sentinel_tokens": list(tokens),
        }

    for dataset in selected:
        preset, _, _ = select_synthetic_preset_for_dataset(dataset)
        try:
            real_X_raw, real_wavelengths_raw = load_real_spectra(dataset, root=root)
            if is_index_fallback_grid(real_wavelengths_raw):
                rows.extend(
                    _blocked_rows_for_all_profiles(
                        dataset=dataset,
                        preset=preset,
                        profiles=requested,
                        blocked_reason=(
                            "wavelength_grid_unknown: real wavelengths could not be parsed"
                        ),
                    )
                )
                continue
            sanitized_real, sanitized_wl, _, real_blocked = sanitize_finite_spectra(
                real_X_raw, real_wavelengths_raw, side="real"
            )
            if real_blocked is not None or sanitized_real is None or sanitized_wl is None:
                rows.extend(
                    _blocked_rows_for_all_profiles(
                        dataset=dataset,
                        preset=preset,
                        profiles=requested,
                        blocked_reason=f"non_finite_spectra: {real_blocked}",
                    )
                )
                continue
            real_X = _downsample_rows(
                sanitized_real,
                max_rows=max_real_samples,
                random_state=_stable_dataset_seed(seed, dataset, "r2a:real_downsample"),
            )
            real_wavelengths = sanitized_wl

            # Build synthetic baseline once per dataset; the profiles transform it.
            synthetic_run, _ = build_on_demand_synthetic_run_for_dataset(
                dataset=dataset,
                preset=preset,
                real_wavelengths=real_wavelengths,
                n_samples=n_synthetic_samples,
                seed=seed,
            )
            synth_X_baseline = np.asarray(synthetic_run.X, dtype=float)
            synth_wl = np.asarray(synthetic_run.wavelengths, dtype=float)
            base_seed = int(synthetic_run.builder_config["random_state"])

            for profile in requested:
                profile_X, profile_audit = _apply_r2a_mechanistic_profile(
                    synth_X_baseline,
                    synth_wl,
                    profile=profile,
                    seed=base_seed,
                )
                profile_seed = int(profile_audit["seed"])
                row_audit = _row_audit_fields(profile_audit)

                sanitized_syn, sanitized_syn_wl, _, syn_blocked = sanitize_finite_spectra(
                    profile_X, synth_wl, side="synthetic"
                )
                if syn_blocked is not None or sanitized_syn is None or sanitized_syn_wl is None:
                    rows.append(
                        _blocked_row(
                            dataset=dataset,
                            preset=preset,
                            profile=profile,
                            profile_enabled=profile != "r2a_baseline",
                            profile_seed=profile_seed,
                            profile_audit=profile_audit,
                            blocked_reason=f"non_finite_spectra_synthetic: {syn_blocked}",
                        )
                    )
                    continue
                profile_X = _downsample_rows(
                    sanitized_syn,
                    max_rows=max_real_samples,
                    random_state=_stable_dataset_seed(
                        seed, dataset, f"r2a:syn_downsample:{profile}"
                    ),
                )
                profile_wl = sanitized_syn_wl
                real_aligned, syn_aligned, aligned_wl = align_to_real_grid(
                    real_X, real_wavelengths, profile_X, profile_wl
                )
                scorecard = compare_real_synthetic(
                    real_X=real_aligned,
                    real_wavelengths=aligned_wl,
                    synthetic_X=syn_aligned,
                    synthetic_wavelengths=aligned_wl,
                    dataset=f"{dataset.database_name}/{dataset.dataset}",
                    source=dataset.source,
                    task=dataset.task,
                    synthetic_preset=preset,
                    comparison_space="uncalibrated_raw",
                    synthetic_mapping_strategy="r2a_sentinel_ablation",
                    synthetic_mapping_reason=(
                        f"r2a_mechanistic_profile={profile}; non_oracle=true; "
                        "report_only=true; not_a_b2_b3_b4_b5_gate=true"
                    ),
                    random_state=_stable_dataset_seed(
                        seed, dataset, f"r2a:metrics:{profile}"
                    ),
                )
                rows.append(
                    AblationRow(
                        status="compared",
                        source=dataset.source,
                        task=dataset.task,
                        dataset=f"{dataset.database_name}/{dataset.dataset}",
                        synthetic_preset=preset,
                        mechanistic_profile=profile,
                        profile_enabled=profile != "r2a_baseline",
                        profile_seed=profile_seed,
                        **row_audit,
                        n_real_samples=int(scorecard.n_real_samples),
                        n_synthetic_samples=int(scorecard.n_synthetic_samples),
                        n_wavelengths=int(scorecard.n_wavelengths),
                        adversarial_auc=scorecard.adversarial_auc,
                        pca_overlap=scorecard.pca_overlap,
                        nearest_neighbor_ratio=scorecard.nearest_neighbor_ratio,
                        derivative_log10_gap=scorecard.derivative_log10_gap,
                        blocked_reason="",
                    )
                )
        except Exception as exc:  # noqa: BLE001 - bench-only diagnostic surfacing
            rows.extend(
                _blocked_rows_for_all_profiles(
                    dataset=dataset,
                    preset=preset,
                    profiles=requested,
                    blocked_reason=f"{type(exc).__name__}: {exc}",
                )
            )

    compared = [row for row in rows if row.status == "compared"]
    status = "done" if compared else "blocked_no_successful_comparisons"
    return {
        "status": status,
        "rows": rows,
        "real_runnable_count": len(real_datasets),
        "real_sentinel_candidate_count": len(sentinel_candidates),
        "real_selected_count": len(selected),
        "profiles": requested,
        "sentinel_tokens": list(tokens),
    }


def render_markdown(
    *,
    result: dict[str, Any],
    report_path: Path,
    csv_path: Path,
    profiles: list[str],
    n_synthetic_samples: int,
    max_real_samples: int,
    max_sentinel_datasets: int,
    seed: int,
    sentinel_tokens: Sequence[str] | None = None,
) -> str:
    rows: list[AblationRow] = result["rows"]
    requested = result["profiles"]
    tokens = list(sentinel_tokens) if sentinel_tokens is not None else list(
        result.get("sentinel_tokens", DEFAULT_SENTINEL_TOKENS)
    )
    command = (
        "PYTHONPATH=bench/nirs_synthetic_pfn/src "
        "python bench/nirs_synthetic_pfn/experiments/exp08_mechanistic_sentinel_ablation.py "
        f"--profiles {','.join(profiles)} "
        f"--n-synthetic-samples {n_synthetic_samples} "
        f"--max-real-samples {max_real_samples} "
        f"--max-sentinel-datasets {max_sentinel_datasets} "
        f"--sentinel-tokens {','.join(tokens)} "
        f"--seed {seed}"
    )
    compared = [row for row in rows if row.status == "compared"]
    blocked = [row for row in rows if row.status == "blocked"]
    smoke_threshold = PROVISIONAL_THRESHOLDS["adversarial_auc_smoke"]
    by_profile = _aggregate_by_profile(compared)

    lines: list[str] = [
        "# R2a Sentinel Mechanistic Profile Ablation",
        "",
        "## Scope and Non-Gate Disclaimer",
        "",
        "- Bench-only, report-only diagnostic lane.",
        "- Does NOT modify or claim any B2/B3/B4/B5 gate outcome.",
        "- Profiles are fixed mechanistic approximations driven by profile name + seed.",
        "- No real-data calibration, no PCA/covariance/noise capture from real data, no ML/DL.",
        "- Comparison space is `uncalibrated_raw` only.",
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
        "## Outputs",
        "",
        f"- Markdown: `{report_path}`",
        f"- CSV: `{csv_path}`",
        "",
        "## Config",
        "",
        f"- Seed: {seed}",
        f"- Profiles: `{', '.join(requested)}`",
        f"- Synthetic samples per (dataset, profile): {n_synthetic_samples}",
        f"- Real samples per dataset cap: {max_real_samples}",
        f"- Sentinel dataset cap: {max_sentinel_datasets if max_sentinel_datasets > 0 else 'all runnable rows'}",
        f"- Sentinel tokens: `{', '.join(tokens)}`"
        + ("" if max_sentinel_datasets > 0 else " (ignored when cap <= 0)"),
        "",
        "## Summary",
        "",
        f"- Status: `{result['status']}`",
        f"- Real runnable rows discovered: {result['real_runnable_count']}",
        f"- Real sentinel candidates after token filter: {result.get('real_sentinel_candidate_count', 0)}",
        f"- Real rows selected: {result['real_selected_count']}",
        f"- Compared rows: {len(compared)}",
        f"- Blocked rows: {len(blocked)}",
        "",
        "## Per-Profile Aggregates (uncalibrated_raw)",
        "",
        f"| profile | n compared | mean adv AUC | median adv AUC | smoke failures (>{smoke_threshold:.2f}) |",
        "|---|---:|---:|---:|---:|",
    ]
    for profile in requested:
        agg = by_profile.get(profile, {"n": 0, "mean": None, "median": None, "fail": 0})
        lines.append(
            f"| `{profile}` | {agg['n']} | {_fmt(agg['mean'])} | {_fmt(agg['median'])} | {agg['fail']} |"
        )

    lines.extend([
        "",
        "## Audit Flags (every row)",
        "",
        "- `oracle=false`",
        "- `label_inputs_used=false`",
        "- `target_inputs_used=false`",
        "- `split_inputs_used=false`",
        "- `source_oracle_used=false`",
        "- `learned=false`",
        "- `real_stat_capture=false`",
        "- `thresholds_modified=false`",
        "- `metrics_modified=false`",
        "- `imputed=false`",
        "- `replays_real_rows=false`",
        "- `profile_input_seed`, `profile_scope`, and compact stable `profile_transform_params` are persisted in every CSV row.",
        "",
        "## Per-Row Metrics",
        "",
        "| dataset | preset | profile | n real | n syn | adv AUC | PCA overlap | NN ratio | deriv log10 gap | status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in rows:
        lines.append(
            f"| `{row.dataset}` | `{row.synthetic_preset}` | `{row.mechanistic_profile}` | "
            f"{row.n_real_samples} | {row.n_synthetic_samples} | "
            f"{_fmt(row.adversarial_auc)} | {_fmt(row.pca_overlap)} | "
            f"{_fmt(row.nearest_neighbor_ratio)} | {_fmt(row.derivative_log10_gap)} | "
            f"`{row.status}` |"
        )

    lines.extend([
        "",
        "## Decision",
        "",
        (
            "Report-only: profile aggregates above are diagnostic only; this lane does not "
            "establish any B2/B3/B4/B5 pass."
        ),
        "",
        "## Raw Summary JSON",
        "",
        "```json",
        json.dumps(
            {
                "status": result["status"],
                "real_runnable_count": result["real_runnable_count"],
                "real_sentinel_candidate_count": result.get(
                    "real_sentinel_candidate_count", 0
                ),
                "real_selected_count": result["real_selected_count"],
                "profiles": requested,
                "sentinel_tokens": tokens,
                "compared_row_count": len(compared),
                "blocked_row_count": len(blocked),
                "provisional_smoke_threshold": smoke_threshold,
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
    ])
    return "\n".join(lines)


def write_csv(rows: list[AblationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def _aggregate_by_profile(rows: list[AblationRow]) -> dict[str, dict[str, Any]]:
    smoke_threshold = PROVISIONAL_THRESHOLDS["adversarial_auc_smoke"]
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = out.setdefault(
            row.mechanistic_profile,
            {"aucs": [], "fail": 0},
        )
        if row.adversarial_auc is not None:
            bucket["aucs"].append(float(row.adversarial_auc))
            if row.adversarial_auc > smoke_threshold:
                bucket["fail"] += 1
    summary: dict[str, dict[str, Any]] = {}
    for profile, bucket in out.items():
        aucs = bucket["aucs"]
        summary[profile] = {
            "n": len(aucs),
            "mean": float(np.mean(aucs)) if aucs else None,
            "median": float(np.median(aucs)) if aucs else None,
            "fail": bucket["fail"],
        }
    return summary


def _blocked_rows_for_all_profiles(
    *,
    dataset: RealDataset,
    preset: str,
    profiles: list[str],
    blocked_reason: str,
) -> list[AblationRow]:
    return [
        _blocked_row(
            dataset=dataset,
            preset=preset,
            profile=profile,
            profile_enabled=profile != "r2a_baseline",
            profile_seed=None,
            profile_audit=_blocked_profile_audit(),
            blocked_reason=blocked_reason,
        )
        for profile in profiles
    ]


def _blocked_row(
    *,
    dataset: RealDataset,
    preset: str,
    profile: str,
    profile_enabled: bool,
    profile_seed: int | None,
    profile_audit: dict[str, Any],
    blocked_reason: str,
) -> AblationRow:
    return AblationRow(
        status="blocked",
        source=dataset.source,
        task=dataset.task,
        dataset=f"{dataset.database_name}/{dataset.dataset}",
        synthetic_preset=preset,
        mechanistic_profile=profile,
        profile_enabled=profile_enabled,
        profile_seed=profile_seed,
        **_row_audit_fields(profile_audit),
        n_real_samples=0,
        n_synthetic_samples=0,
        n_wavelengths=0,
        adversarial_auc=None,
        pca_overlap=None,
        nearest_neighbor_ratio=None,
        derivative_log10_gap=None,
        blocked_reason=blocked_reason,
    )


def _downsample_rows(X: np.ndarray, *, max_rows: int, random_state: int) -> np.ndarray:
    if max_rows <= 0 or X.shape[0] <= max_rows:
        return np.asarray(X, dtype=float)
    rng = np.random.default_rng(random_state)
    indices = rng.choice(X.shape[0], size=max_rows, replace=False)
    indices.sort()
    return np.asarray(X[indices], dtype=float)


def _parse_profiles(raw: str) -> list[str]:
    profiles = [token.strip() for token in raw.split(",") if token.strip()]
    if not profiles:
        raise ValueError("at least one profile must be provided")
    return profiles


def _parse_sentinel_tokens(raw: str) -> list[str]:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("at least one sentinel token must be provided")
    return tokens


def _token_priority_map(tokens: Sequence[str]) -> dict[str, int]:
    """Map each user-supplied token to its priority group index.

    Tokens belonging to one of ``SENTINEL_PRIORITY_GROUPS`` get that group's
    index (0 = primary, 1+ = secondary). Tokens not matching any default group
    fall into an implicit final group whose priority equals the number of
    default groups. Empty/whitespace tokens are dropped.
    """
    group_lookup: dict[str, int] = {}
    for prio, (_name, group) in enumerate(SENTINEL_PRIORITY_GROUPS):
        for token in group:
            group_lookup[token.casefold()] = prio
    custom_priority = len(SENTINEL_PRIORITY_GROUPS)
    out: dict[str, int] = {}
    for token in tokens:
        cf = token.casefold().strip()
        if not cf:
            continue
        out[cf] = group_lookup.get(cf, custom_priority)
    return out


def _select_sentinel_datasets(
    datasets: Iterable[RealDataset],
    tokens: Sequence[str],
) -> list[RealDataset]:
    """Filter discovered datasets to those matching any sentinel token.

    Match is a case-insensitive substring test against ``source``, ``task``,
    ``database_name``, and ``dataset``. Selection is primary-first: each matched
    dataset is assigned the highest priority (lowest priority index) of its
    matching tokens, then the result is sorted by ``(priority, original_index)``
    so primary sentinels precede secondaries and within each group the original
    cohort order is preserved. Empty ``tokens`` yields an empty list.
    """
    priority_map = _token_priority_map(tokens)
    if not priority_map:
        return []
    decorated: list[tuple[int, int, RealDataset]] = []
    for idx, dataset in enumerate(datasets):
        haystack = " ".join(
            (dataset.source, dataset.task, dataset.database_name, dataset.dataset)
        ).casefold()
        matched = [prio for token_cf, prio in priority_map.items() if token_cf in haystack]
        if matched:
            decorated.append((min(matched), idx, dataset))
    decorated.sort(key=lambda item: (item[0], item[1]))
    return [dataset for _, _, dataset in decorated]


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "bench" / "nirs_synthetic_pfn").is_dir():
            return candidate
    raise RuntimeError(f"could not locate repo root from {here}")


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}"


def _csv_fieldnames() -> list[str]:
    return [field.name for field in fields(AblationRow)]


def _row_audit_fields(profile_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "audit_oracle": _audit_bool(profile_audit, "oracle"),
        "audit_label_inputs_used": _audit_bool(profile_audit, "label_inputs_used"),
        "audit_target_inputs_used": _audit_bool(profile_audit, "target_inputs_used"),
        "audit_split_inputs_used": _audit_bool(profile_audit, "split_inputs_used"),
        "audit_source_oracle_used": _audit_bool(profile_audit, "source_oracle_used"),
        "audit_learned": _audit_bool(profile_audit, "learned"),
        "audit_real_stat_capture": _audit_bool(profile_audit, "real_stat_capture"),
        "audit_thresholds_modified": _audit_bool(profile_audit, "thresholds_modified"),
        "audit_metrics_modified": _audit_bool(profile_audit, "metrics_modified"),
        "audit_imputed": _audit_bool(profile_audit, "imputed"),
        "audit_replays_real_rows": _audit_bool(profile_audit, "replays_real_rows"),
        "profile_input_seed": _optional_int(profile_audit.get("input_seed")),
        "profile_scope": str(profile_audit.get("scope", R2A_AUDIT_SCOPE)),
        "profile_transform_params": _compact_json(profile_audit.get("transform_params", {})),
    }


def _blocked_profile_audit() -> dict[str, Any]:
    return {
        "scope": R2A_AUDIT_SCOPE,
        **dict.fromkeys(R2A_AUDIT_FLAG_KEYS, False),
        "input_seed": None,
        "transform_params": {},
    }


def _audit_bool(profile_audit: dict[str, Any], key: str) -> bool:
    return bool(profile_audit.get(key, False))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


# Keep PRESETS importable via this module for diagnostics.
__all__ = [
    "AblationRow",
    "DEFAULT_MAX_SENTINEL_DATASETS",
    "DEFAULT_SENTINEL_TOKENS",
    "PRESETS",
    "PRIMARY_SENTINEL_TOKENS",
    "SECONDARY_FRUIT_SENTINEL_TOKENS",
    "SECONDARY_MILK_SENTINEL_TOKENS",
    "SECONDARY_SOIL_SENTINEL_TOKENS",
    "SENTINEL_PRIORITY_GROUPS",
    "main",
    "render_markdown",
    "run_ablation",
    "write_csv",
]


if __name__ == "__main__":
    main()
