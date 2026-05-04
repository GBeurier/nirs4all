"""R9g DIESEL R4 component attribution diagnostic audit.

Bench-only diagnostic-only audit that compares the R4a/R4b/R4c family against
R3d and the clean R9e/R9f attenuation profiles. This audit does not introduce
or tune a profile. It only attributes why R4b/R4c reduce the DIESEL gap more
than R9e/R9f on the same repeated-seed cohort.

The audit intentionally remains comparative. Strict component-by-component
isolation of support CH centers, 1720 nm removal, damping windows, CH width /
gain, and the 975 nm continuum hump would require new builder variants and is
therefore out of scope for this phase.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import fields
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np


def _load_module(name: str, filename: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_exp09 = _load_module(
    "exp09_sentinel_morphology_audit",
    "exp09_sentinel_morphology_audit.py",
)
_exp10 = _load_module(
    "exp10_diesel_mean_shift_localization",
    "exp10_diesel_mean_shift_localization.py",
)
_exp15 = _load_module(
    "exp15_diesel_pathlength_reference_attenuation_audit",
    "exp15_diesel_pathlength_reference_attenuation_audit.py",
)
_exp16 = _load_module(
    "exp16_diesel_pre_offset_pathlength_reference_attenuation_audit",
    "exp16_diesel_pre_offset_pathlength_reference_attenuation_audit.py",
)

RealDataset = _exp15.RealDataset
R9gRow = _exp15.R9eRow
align_to_real_grid = _exp15.align_to_real_grid
discover_local_real_datasets = _exp15.discover_local_real_datasets
is_index_fallback_grid = _exp15.is_index_fallback_grid
load_real_spectra = _exp15.load_real_spectra
sanitize_finite_spectra = _exp15.sanitize_finite_spectra

R9G_AUDITED_PROFILES: tuple[str, ...] = (
    "r3d_diesel_matrix_v1",
    "r4a_diesel_basis_v1",
    "r4b_diesel_derivative_restore_v1",
    "r4c_diesel_balanced_derivative_v1",
    "r9e_diesel_pathlength_reference_attenuation_v1",
    "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1",
)
R9G_FOCUS_PROFILES: tuple[str, ...] = (
    "r4b_diesel_derivative_restore_v1",
    "r4c_diesel_balanced_derivative_v1",
)
R9G_REFERENCE_PROFILES: tuple[str, ...] = (
    "r3d_diesel_matrix_v1",
    "r4a_diesel_basis_v1",
    "r9e_diesel_pathlength_reference_attenuation_v1",
    "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1",
)
R9G_BASE_PROFILE = "r3d_diesel_matrix_v1"
R9G_AUDIT_SCOPE = "bench_only_r9g_diesel_r4_component_attribution_audit"
R9G_LIMITATION = (
    "strict_component_isolation_requires_new_builder_variants_not_this_phase"
)

DEFAULT_SEEDS: tuple[int, ...] = (20260501, 20260502, 20260503)
DEFAULT_N_SYNTHETIC_SAMPLES = 64
DEFAULT_MAX_REAL_SAMPLES = 64
DEFAULT_MAX_SENTINEL_DATASETS = 8
DEFAULT_SENTINEL_TOKENS: tuple[str, ...] = ("DIESEL",)
DEFAULT_REPORT = Path(
    "bench/nirs_synthetic_pfn/reports/"
    "r9g_diesel_r4_component_attribution_audit.md"
)
DEFAULT_CSV = Path(
    "bench/nirs_synthetic_pfn/reports/"
    "r9g_diesel_r4_component_attribution_audit.csv"
)
COMPARISON_SPACE = "uncalibrated_raw"
SUPPORT_LOW_NM: float = _exp10.SUPPORT_LOW_NM
SUPPORT_HIGH_NM: float = _exp10.SUPPORT_HIGH_NM

PAIRED_DELTA_ATTRS: tuple[str, ...] = (
    "global_mean_delta",
    "support_mean_delta",
    "support_weighted_delta",
    "off_support_weighted_delta",
    "morphology_gap_score",
    "log10_derivative_std_p50_ratio",
    "mean_curve_corr",
)


def _audit_fields() -> dict[str, Any]:
    fields_dict = cast(dict[str, Any], _exp15._audit_fields())
    fields_dict["audit_scope"] = R9G_AUDIT_SCOPE
    return fields_dict


def _validate_profiles(profiles: Sequence[str]) -> tuple[str, ...]:
    if not profiles:
        raise ValueError("at least one remediation profile must be provided")
    invalid = [profile for profile in profiles if profile not in R9G_AUDITED_PROFILES]
    if invalid:
        raise ValueError(
            f"unknown R9g profiles {invalid!r}; "
            f"valid profiles are {list(R9G_AUDITED_PROFILES)}"
        )
    return tuple(profiles)


def _effective_profile_for_dataset(dataset: Any, profile: str | None) -> str | None:
    return cast(str | None, _exp16._effective_profile_for_dataset(dataset, profile))


def _build_synthetic_run(
    *,
    dataset: Any,
    preset: str,
    n_samples: int,
    seed: int,
    remediation_profile: str,
) -> Any:
    return _exp16._build_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=n_samples,
        seed=seed,
        remediation_profile=remediation_profile,
    )


def _effective_matrix_route(metadata: dict[str, Any] | None) -> str | None:
    return cast(str | None, _exp15._effective_matrix_route(metadata))


def _guard_clip_fraction(profile: str, metadata: dict[str, Any] | None) -> float | None:
    if profile == _exp16.R9F_FOCUS_PROFILE:
        return cast(float | None, _exp16._guard_clip_fraction(profile, metadata))
    if profile == _exp15.R9E_FOCUS_PROFILE:
        return cast(float | None, _exp15._guard_clip_fraction(profile, metadata))
    return None


def _blocked_row(
    *,
    seed: int,
    requested_profile: str,
    effective_profile: str | None,
    dataset: Any,
    preset: str,
    blocked_reason: str,
    effective_matrix_route: str | None = None,
) -> Any:
    return R9gRow(
        status="blocked",
        seed=int(seed),
        remediation_profile=requested_profile,
        effective_remediation_profile=effective_profile,
        source=dataset.source,
        task=dataset.task,
        dataset=f"{dataset.database_name}/{dataset.dataset}",
        synthetic_preset=preset,
        effective_matrix_route=effective_matrix_route,
        comparison_space=COMPARISON_SPACE,
        n_real_samples=0,
        n_synthetic_samples=0,
        n_wavelengths=0,
        wavelength_min=None,
        wavelength_max=None,
        guard_clip_fraction=0.0
        if requested_profile
        in (
            _exp15.R9E_FOCUS_PROFILE,
            _exp16.R9F_FOCUS_PROFILE,
        )
        else None,
        **_exp15._empty_decomposition(),
        **_exp15._empty_morphology_subset(),
        **_audit_fields(),
        blocked_reason=blocked_reason,
    )


def run_audit(
    *,
    root: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_synthetic_samples: int = DEFAULT_N_SYNTHETIC_SAMPLES,
    max_real_samples: int = DEFAULT_MAX_REAL_SAMPLES,
    max_sentinel_datasets: int = DEFAULT_MAX_SENTINEL_DATASETS,
    sentinel_tokens: Sequence[str] | None = None,
    profiles: Sequence[str] | None = None,
    support_low_nm: float = SUPPORT_LOW_NM,
    support_high_nm: float = SUPPORT_HIGH_NM,
) -> dict[str, Any]:
    seeds = tuple(int(s) for s in seeds)
    if not seeds:
        raise ValueError("at least one seed must be provided")
    audited_profiles = _validate_profiles(
        profiles if profiles is not None else R9G_AUDITED_PROFILES
    )
    tokens = (
        tuple(DEFAULT_SENTINEL_TOKENS)
        if sentinel_tokens is None
        else tuple(sentinel_tokens)
    )
    real_datasets, _ = discover_local_real_datasets(root)
    sentinel_candidates = _exp09._select_sentinel_datasets(real_datasets, tokens)
    selected = (
        list(sentinel_candidates)
        if max_sentinel_datasets <= 0
        else sentinel_candidates[:max_sentinel_datasets]
    )
    rows: list[Any] = []
    if not selected:
        return {
            "status": "blocked_no_real_data",
            "rows": rows,
            "real_runnable_count": len(real_datasets),
            "real_sentinel_candidate_count": len(sentinel_candidates),
            "real_selected_count": 0,
            "sentinel_tokens": list(tokens),
            "seeds": list(seeds),
            "audited_profiles": list(audited_profiles),
            "support_low_nm": float(support_low_nm),
            "support_high_nm": float(support_high_nm),
        }

    for dataset in selected:
        preset = _exp09.select_synthetic_preset_for_dataset(dataset)
        try:
            real_X_raw, real_wl_raw = load_real_spectra(dataset, root=root)
        except Exception as exc:  # noqa: BLE001
            for seed in seeds:
                for profile in audited_profiles:
                    rows.append(
                        _blocked_row(
                            seed=seed,
                            requested_profile=profile,
                            effective_profile=_effective_profile_for_dataset(
                                dataset,
                                profile,
                            ),
                            dataset=dataset,
                            preset=preset,
                            blocked_reason=f"{type(exc).__name__}: {exc}",
                        )
                    )
            continue

        if is_index_fallback_grid(real_wl_raw):
            for seed in seeds:
                for profile in audited_profiles:
                    rows.append(
                        _blocked_row(
                            seed=seed,
                            requested_profile=profile,
                            effective_profile=_effective_profile_for_dataset(
                                dataset,
                                profile,
                            ),
                            dataset=dataset,
                            preset=preset,
                            blocked_reason="wavelength_grid_unknown",
                        )
                    )
            continue

        sanitized_real, sanitized_wl, _, real_blocked = sanitize_finite_spectra(
            real_X_raw,
            real_wl_raw,
            side="real",
        )
        if real_blocked is not None or sanitized_real is None or sanitized_wl is None:
            for seed in seeds:
                for profile in audited_profiles:
                    rows.append(
                        _blocked_row(
                            seed=seed,
                            requested_profile=profile,
                            effective_profile=_effective_profile_for_dataset(
                                dataset,
                                profile,
                            ),
                            dataset=dataset,
                            preset=preset,
                            blocked_reason=f"non_finite_spectra: {real_blocked}",
                        )
                    )
            continue

        for seed in seeds:
            real_X = _exp09._downsample_rows(
                sanitized_real,
                max_rows=max_real_samples,
                random_state=_exp09._stable_dataset_seed(
                    seed,
                    dataset,
                    "r9g:real_downsample",
                ),
            )
            for profile in audited_profiles:
                effective = _effective_profile_for_dataset(dataset, profile)
                effective_route: str | None = None
                try:
                    synthetic_run = _build_synthetic_run(
                        dataset=dataset,
                        preset=preset,
                        n_samples=n_synthetic_samples,
                        seed=seed,
                        remediation_profile=profile,
                    )
                    sanitized_syn, sanitized_syn_wl, _, syn_blocked = (
                        sanitize_finite_spectra(
                            np.asarray(synthetic_run.X, dtype=float),
                            np.asarray(synthetic_run.wavelengths, dtype=float),
                            side="synthetic",
                        )
                    )
                    effective_route = _effective_matrix_route(synthetic_run.metadata)
                    if (
                        syn_blocked is not None
                        or sanitized_syn is None
                        or sanitized_syn_wl is None
                    ):
                        raise ValueError(f"non_finite_spectra_synthetic: {syn_blocked}")
                    synth_downsampled = _exp09._downsample_rows(
                        sanitized_syn,
                        max_rows=max_real_samples,
                        random_state=_exp09._stable_dataset_seed(
                            seed,
                            dataset,
                            "r9g:syn_downsample",
                        ),
                    )
                    real_aligned, syn_aligned, aligned_wl = align_to_real_grid(
                        real_X,
                        sanitized_wl,
                        synth_downsampled,
                        sanitized_syn_wl,
                    )
                except Exception as exc:  # noqa: BLE001
                    rows.append(
                        _blocked_row(
                            seed=seed,
                            requested_profile=profile,
                            effective_profile=effective,
                            dataset=dataset,
                            preset=preset,
                            blocked_reason=f"{type(exc).__name__}: {exc}",
                            effective_matrix_route=effective_route,
                        )
                    )
                    continue

                metrics = _exp09.compute_morphology_metrics(
                    real_aligned,
                    syn_aligned,
                    aligned_wl,
                )
                decomposition = _exp10.compute_support_decomposition(
                    real_aligned,
                    syn_aligned,
                    aligned_wl,
                    support_low_nm=support_low_nm,
                    support_high_nm=support_high_nm,
                )
                rows.append(
                    R9gRow(
                        status="compared",
                        seed=int(seed),
                        remediation_profile=profile,
                        effective_remediation_profile=effective,
                        source=dataset.source,
                        task=dataset.task,
                        dataset=f"{dataset.database_name}/{dataset.dataset}",
                        synthetic_preset=preset,
                        effective_matrix_route=effective_route,
                        comparison_space=COMPARISON_SPACE,
                        n_real_samples=int(real_aligned.shape[0]),
                        n_synthetic_samples=int(syn_aligned.shape[0]),
                        n_wavelengths=metrics["n_wavelengths"],
                        wavelength_min=metrics["wavelength_min"],
                        wavelength_max=metrics["wavelength_max"],
                        support_low_nm=decomposition["support_low_nm"],
                        support_high_nm=decomposition["support_high_nm"],
                        support_count=decomposition["support_count"],
                        off_support_count=decomposition["off_support_count"],
                        support_weight=decomposition["support_weight"],
                        off_support_weight=decomposition["off_support_weight"],
                        support_mean_delta=decomposition["support_mean_delta"],
                        off_support_mean_delta=decomposition["off_support_mean_delta"],
                        support_weighted_delta=decomposition[
                            "support_weighted_delta"
                        ],
                        off_support_weighted_delta=decomposition[
                            "off_support_weighted_delta"
                        ],
                        global_mean_delta=decomposition["global_mean_delta"],
                        decomposition_residual=decomposition[
                            "decomposition_residual"
                        ],
                        **_exp15._morphology_subset(metrics),
                        guard_clip_fraction=_guard_clip_fraction(
                            profile,
                            synthetic_run.metadata,
                        ),
                        **_audit_fields(),
                        blocked_reason="",
                    )
                )

    compared = [row for row in rows if row.status == "compared"]
    return {
        "status": "done" if compared else "blocked_no_successful_comparisons",
        "rows": rows,
        "real_runnable_count": len(real_datasets),
        "real_sentinel_candidate_count": len(sentinel_candidates),
        "real_selected_count": len(selected),
        "sentinel_tokens": list(tokens),
        "seeds": list(seeds),
        "audited_profiles": list(audited_profiles),
        "support_low_nm": float(support_low_nm),
        "support_high_nm": float(support_high_nm),
    }


def _delta_col(focus: str, ref: str, attr: str) -> str:
    return f"delta_{focus}_minus_{ref}__{attr}"


def _dominant_gap_col(focus: str, ref: str) -> str:
    return f"dominant_gap_{focus}_minus_{ref}"


def _paired_delta_columns() -> list[str]:
    columns: list[str] = []
    for focus in R9G_FOCUS_PROFILES:
        for ref in R9G_REFERENCE_PROFILES:
            columns.extend(_delta_col(focus, ref, attr) for attr in PAIRED_DELTA_ATTRS)
            columns.append(_dominant_gap_col(focus, ref))
    return columns


def _csv_fieldnames() -> list[str]:
    return [field.name for field in fields(R9gRow)] + _paired_delta_columns()


def _row_key(row: Any) -> tuple[int, str]:
    return (int(row.seed), str(row.dataset))


def _paired_deltas_vs_references(rows: Sequence[Any]) -> list[dict[str, Any]]:
    compared = [row for row in rows if row.status == "compared"]
    by_profile: dict[str, dict[tuple[int, str], Any]] = {}
    for row in compared:
        by_profile.setdefault(row.remediation_profile, {})[_row_key(row)] = row
    entries: list[dict[str, Any]] = []
    for focus in R9G_FOCUS_PROFILES:
        for key, row in by_profile.get(focus, {}).items():
            entry: dict[str, Any] = {
                "seed": int(row.seed),
                "dataset": str(row.dataset),
                "focus_profile": focus,
                "focus_global_mean_delta": row.global_mean_delta,
                "focus_support_mean_delta": row.support_mean_delta,
                "focus_morphology_gap_score": row.morphology_gap_score,
                "focus_derivative": row.log10_derivative_std_p50_ratio,
                "focus_mean_curve_corr": row.mean_curve_corr,
                "focus_dominant_gap": row.dominant_morphology_gap,
            }
            for ref in R9G_REFERENCE_PROFILES:
                ref_row = by_profile.get(ref, {}).get(key)
                for attr in PAIRED_DELTA_ATTRS:
                    col = _delta_col(focus, ref, attr)
                    if ref_row is None:
                        entry[col] = None
                        continue
                    lhs = getattr(row, attr)
                    rhs = getattr(ref_row, attr)
                    entry[col] = None if lhs is None or rhs is None else float(lhs) - float(rhs)
                entry[_dominant_gap_col(focus, ref)] = (
                    None
                    if ref_row is None
                    else f"{row.dominant_morphology_gap}<-{ref_row.dominant_morphology_gap}"
                )
            entries.append(entry)
    return sorted(
        entries,
        key=lambda item: (
            str(item["dataset"]),
            int(item["seed"]),
            str(item["focus_profile"]),
        ),
    )


def write_csv(rows: list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    paired_by_key = {
        (int(entry["seed"]), str(entry["dataset"]), str(entry["focus_profile"])): entry
        for entry in _paired_deltas_vs_references(rows)
    }
    paired_cols = _paired_delta_columns()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_csv_fieldnames())
        writer.writeheader()
        for row in rows:
            record = row.to_dict()
            paired = (
                paired_by_key.get((*_row_key(row), row.remediation_profile))
                if row.remediation_profile in R9G_FOCUS_PROFILES
                and row.status == "compared"
                else None
            )
            for col in paired_cols:
                record[col] = paired.get(col) if paired is not None else None
            writer.writerow(record)


def _fmt(value: float | None) -> str:
    return cast(str, _exp15._fmt(value))


def _median(values: Iterable[float | None]) -> float | None:
    return cast(float | None, _exp15._median(values))


def _format_gap_distribution(rows: Sequence[Any]) -> str:
    counts = Counter(row.dominant_morphology_gap for row in rows)
    if not counts:
        return "n/a"
    return ", ".join(f"{label}={count}" for label, count in counts.most_common())


def _aggregate_by_profile(rows: Sequence[Any]) -> list[dict[str, Any]]:
    return cast(
        list[dict[str, Any]],
        _exp15._aggregate_by_profile(cast(Sequence[Any], rows)),
    )


def _median_delta(
    paired: Sequence[dict[str, Any]],
    *,
    focus: str,
    ref: str,
    attr: str,
) -> float | None:
    col = _delta_col(focus, ref, attr)
    return _median(
        cast(float | None, entry.get(col))
        for entry in paired
        if entry["focus_profile"] == focus
    )


def _paired_delta_summary(rows: Sequence[Any]) -> list[dict[str, Any]]:
    paired = _paired_deltas_vs_references(rows)
    out: list[dict[str, Any]] = []
    for focus in R9G_FOCUS_PROFILES:
        for ref in R9G_REFERENCE_PROFILES:
            bucket = [entry for entry in paired if entry["focus_profile"] == focus]
            gap_counts = Counter(
                entry.get(_dominant_gap_col(focus, ref))
                for entry in bucket
                if entry.get(_dominant_gap_col(focus, ref)) is not None
            )
            out.append(
                {
                    "focus": focus,
                    "reference": ref,
                    "n": len(bucket),
                    "median_global_mean_delta": _median_delta(
                        paired,
                        focus=focus,
                        ref=ref,
                        attr="global_mean_delta",
                    ),
                    "median_support_mean_delta": _median_delta(
                        paired,
                        focus=focus,
                        ref=ref,
                        attr="support_mean_delta",
                    ),
                    "median_morphology_gap_score": _median_delta(
                        paired,
                        focus=focus,
                        ref=ref,
                        attr="morphology_gap_score",
                    ),
                    "median_log10_derivative_std_p50_ratio": _median_delta(
                        paired,
                        focus=focus,
                        ref=ref,
                        attr="log10_derivative_std_p50_ratio",
                    ),
                    "median_mean_curve_corr": _median_delta(
                        paired,
                        focus=focus,
                        ref=ref,
                        attr="mean_curve_corr",
                    ),
                    "dominant_gap_pairing": ", ".join(
                        f"{label}={count}" for label, count in gap_counts.most_common()
                    )
                    or "n/a",
                }
            )
    return out


def _profile_median(
    summary: Sequence[dict[str, Any]],
    profile: str,
    key: str,
) -> float | None:
    for entry in summary:
        if entry["profile"] == profile:
            return cast(float | None, entry.get(key))
    return None


def _interpret_attribution(profile_summary: Sequence[dict[str, Any]]) -> str:
    r4b_gap = _profile_median(
        profile_summary,
        "r4b_diesel_derivative_restore_v1",
        "median_morphology_gap_score",
    )
    r4c_gap = _profile_median(
        profile_summary,
        "r4c_diesel_balanced_derivative_v1",
        "median_morphology_gap_score",
    )
    r9e_gap = _profile_median(
        profile_summary,
        "r9e_diesel_pathlength_reference_attenuation_v1",
        "median_morphology_gap_score",
    )
    r9f_gap = _profile_median(
        profile_summary,
        "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1",
        "median_morphology_gap_score",
    )
    r4a_derivative = _profile_median(
        profile_summary,
        "r4a_diesel_basis_v1",
        "median_log10_derivative_std_p50_ratio",
    )
    r4b_derivative = _profile_median(
        profile_summary,
        "r4b_diesel_derivative_restore_v1",
        "median_log10_derivative_std_p50_ratio",
    )
    derivative_restored = (
        r4a_derivative is not None
        and r4b_derivative is not None
        and r4b_derivative > r4a_derivative
    )
    beats_clean_attenuation = (
        r4b_gap is not None
        and r4c_gap is not None
        and r9e_gap is not None
        and r9f_gap is not None
        and min(r4b_gap, r4c_gap) < min(r9e_gap, r9f_gap)
    )
    if beats_clean_attenuation and derivative_restored:
        return (
            "The signal is primarily a coupled R4 hydrocarbon-basis effect: "
            "support CH centers/drop 1720 plus the 975 nm continuum hump set "
            "the support level, while R4b/R4c width/gain derivative restoration "
            "and weaker/narrower damping keep the gap improvement from becoming "
            "R4a-like derivative suppression. The damping windows and 975 nm "
            "hump cannot be isolated from the CH-center/drop-1720 decision "
            "without creating new builder variants, so the formal conclusion is "
            "coupling non-isolable in this phase."
        )
    return (
        "The attribution remains inconclusive on this cohort; strict component "
        "separation would require new builder variants, which this phase does "
        "not create."
    )


def render_markdown(
    *,
    result: dict[str, Any],
    report_path: Path,
    csv_path: Path,
    n_synthetic_samples: int,
    max_real_samples: int,
    max_sentinel_datasets: int,
    seeds: Sequence[int],
    sentinel_tokens: Sequence[str] | None = None,
    profiles: Sequence[str] | None = None,
    support_low_nm: float = SUPPORT_LOW_NM,
    support_high_nm: float = SUPPORT_HIGH_NM,
) -> str:
    rows: list[Any] = result["rows"]
    tokens = (
        list(sentinel_tokens)
        if sentinel_tokens is not None
        else list(result.get("sentinel_tokens", DEFAULT_SENTINEL_TOKENS))
    )
    audited = (
        list(profiles)
        if profiles is not None
        else list(result.get("audited_profiles", R9G_AUDITED_PROFILES))
    )
    compared = [row for row in rows if row.status == "compared"]
    blocked = [row for row in rows if row.status == "blocked"]
    profile_summary = _aggregate_by_profile(rows)
    paired_summary = _paired_delta_summary(rows)
    attribution = _interpret_attribution(profile_summary)
    command = (
        "PYTHONPATH=bench/nirs_synthetic_pfn/src "
        "python bench/nirs_synthetic_pfn/experiments/"
        "exp17_diesel_r4_component_attribution_audit.py "
        f"--n-synthetic-samples {n_synthetic_samples} "
        f"--max-real-samples {max_real_samples} "
        f"--max-sentinel-datasets {max_sentinel_datasets} "
        f"--sentinel-tokens {','.join(tokens)} "
        f"--seeds {','.join(str(int(s)) for s in seeds)} "
        f"--profiles {','.join(audited)}"
    )
    lines = [
        "# R9g DIESEL R4 Component Attribution Diagnostic Audit",
        "",
        "## Scope and Non-Gate Disclaimer",
        "",
        "- Bench-only, mechanistic, diagnostic-only repeated-seed audit. Comparison space is `uncalibrated_raw` only.",
        "- R3d remains the accepted baseline; R4b/R4c are not promoted, no gate.",
        "- No calibration, real-stat capture, PCA/covariance/noise capture, ML/DL, labels, targets, splits, downstream feedback, threshold mutation, or metric mutation.",
        "- No `nirs4all/` integration is authorized by this audit.",
        "- No R9e/R9f attenuation amplitude retuning is performed.",
        f"- Limitation: `{R9G_LIMITATION}`.",
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
        f"- Seeds: {', '.join(str(int(s)) for s in seeds)}",
        f"- Synthetic samples per dataset: {n_synthetic_samples}",
        f"- Real samples per dataset cap: {max_real_samples}",
        f"- Sentinel dataset cap: {max_sentinel_datasets if max_sentinel_datasets > 0 else 'all token-matched sentinel rows'}",
        f"- Sentinel tokens: `{', '.join(tokens)}`",
        f"- Audited profiles: `{', '.join(audited)}`",
        f"- Support window: `{support_low_nm:g}-{support_high_nm:g}` nm",
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
        "## Per-Profile Synthesis (compared rows only)",
        "",
    ]
    if not profile_summary:
        lines.append("No compared rows to aggregate.")
    else:
        lines.extend(
            [
                "| profile | n | median global mean delta | median support mean delta | median morphology gap | median derivative | median mean_curve_corr | median guard_clip_fraction | dominant gap |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for entry in profile_summary:
            lines.append(
                f"| `{entry['profile']}` | {entry['n']} | "
                f"{_fmt(entry['median_global_mean_delta'])} | "
                f"{_fmt(entry['median_support_mean_delta'])} | "
                f"{_fmt(entry['median_morphology_gap_score'])} | "
                f"{_fmt(entry['median_log10_derivative_std_p50_ratio'])} | "
                f"{_fmt(entry['median_mean_curve_corr'])} | "
                f"{_fmt(entry['median_guard_clip_fraction'])} | "
                f"{entry['dominant_gap_distribution']} |"
            )
    lines.extend(
        [
            "",
            "## Paired Attribution Deltas: R4b/R4c minus references",
            "",
            "Negative deltas on `global_mean_delta`, `support_mean_delta`, and `morphology_gap_score` mean the R4 focus profile is lower than the same `(seed, dataset)` reference. Positive derivative deltas mean more first-derivative energy than the reference.",
            "",
            "| focus | reference | n | median global | median support | median morphology gap | median derivative | median mean_curve_corr | dominant gap pairing |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    if not paired_summary:
        lines.append("| (no paired rows) | NA | 0 | NA | NA | NA | NA | NA | NA |")
    else:
        for entry in paired_summary:
            lines.append(
                f"| `{entry['focus']}` | `{entry['reference']}` | {entry['n']} | "
                f"{_fmt(entry['median_global_mean_delta'])} | "
                f"{_fmt(entry['median_support_mean_delta'])} | "
                f"{_fmt(entry['median_morphology_gap_score'])} | "
                f"{_fmt(entry['median_log10_derivative_std_p50_ratio'])} | "
                f"{_fmt(entry['median_mean_curve_corr'])} | "
                f"{entry['dominant_gap_pairing']} |"
            )
    lines.extend(
        [
            "",
            "## Mechanistic Attribution",
            "",
            attribution,
            "",
            "- Support CH centers/drop 1720: implicated as part of the R4-family support basis, but not isolated from damping and hump changes in this audit.",
            "- Width/gain derivative restoration: implicated by R4b/R4c recovering derivative relative to R4a while preserving more gap reduction than R9e/R9f.",
            "- Damping windows: implicated because R4b/R4c weaken/narrow the R4a damping, but the damping contribution is coupled to width/gain.",
            "- 975 nm continuum hump: plausible support-level contributor, but coupled to the CH-center/drop-1720 and damping package.",
            "- Coupling verdict: non-isolable without new builder variants; this phase does not modify `builder_adapter.py` and does not create those variants.",
            "",
            "## Mean-Shift Decomposition (uncalibrated_raw)",
            "",
            "| dataset | seed | profile | global mean delta | support mean delta | morphology gap | derivative | mean_curve_corr | dominant gap | guard_clip_fraction | status |",
            "|---|---:|---|---:|---:|---:|---:|---:|---|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row.dataset}` | {row.seed} | `{row.remediation_profile}` | "
            f"{_fmt(row.global_mean_delta)} | "
            f"{_fmt(row.support_mean_delta)} | "
            f"{_fmt(row.morphology_gap_score)} | "
            f"{_fmt(row.log10_derivative_std_p50_ratio)} | "
            f"{_fmt(row.mean_curve_corr)} | "
            f"`{row.dominant_morphology_gap}` | "
            f"{_fmt(row.guard_clip_fraction)} | `{row.status}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Diagnostic-only, not promoted, no gate, no `nirs4all/` integration. R3d remains the accepted DIESEL baseline; R4b/R4c remain explanatory evidence only.",
            "",
            "## Raw Summary JSON",
            "",
            "```json",
            json.dumps(
                {
                    "status": result["status"],
                    "comparison_space": COMPARISON_SPACE,
                    "real_runnable_count": result["real_runnable_count"],
                    "real_sentinel_candidate_count": result.get(
                        "real_sentinel_candidate_count",
                        0,
                    ),
                    "real_selected_count": result["real_selected_count"],
                    "sentinel_tokens": tokens,
                    "seeds": [int(s) for s in seeds],
                    "audited_profiles": audited,
                    "focus_profiles": list(R9G_FOCUS_PROFILES),
                    "reference_profiles": list(R9G_REFERENCE_PROFILES),
                    "support_low_nm": float(support_low_nm),
                    "support_high_nm": float(support_high_nm),
                    "compared_row_count": len(compared),
                    "blocked_row_count": len(blocked),
                    "limitation": R9G_LIMITATION,
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_csv_list(raw: str) -> list[str]:
    return cast(list[str], _exp15._parse_csv_list(raw))


def _parse_seeds(raw: str) -> list[int]:
    return [int(token) for token in _parse_csv_list(raw)]


def _repo_root() -> Path:
    return cast(Path, _exp15._repo_root())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-synthetic-samples", type=int, default=64)
    parser.add_argument("--max-real-samples", type=int, default=64)
    parser.add_argument("--max-sentinel-datasets", type=int, default=8)
    parser.add_argument("--seeds", type=_parse_seeds, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--sentinel-tokens",
        type=_parse_csv_list,
        default=list(DEFAULT_SENTINEL_TOKENS),
    )
    parser.add_argument(
        "--profiles",
        type=_parse_csv_list,
        default=list(R9G_AUDITED_PROFILES),
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    root = args.root if args.root is not None else _repo_root()
    result = run_audit(
        root=root,
        seeds=args.seeds,
        n_synthetic_samples=args.n_synthetic_samples,
        max_real_samples=args.max_real_samples,
        max_sentinel_datasets=args.max_sentinel_datasets,
        sentinel_tokens=args.sentinel_tokens,
        profiles=args.profiles,
    )
    rows: list[Any] = result["rows"]
    write_csv(rows, args.csv)
    markdown = render_markdown(
        result=result,
        report_path=args.report,
        csv_path=args.csv,
        n_synthetic_samples=args.n_synthetic_samples,
        max_real_samples=args.max_real_samples,
        max_sentinel_datasets=args.max_sentinel_datasets,
        seeds=args.seeds,
        sentinel_tokens=args.sentinel_tokens,
        profiles=args.profiles,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown, encoding="utf-8")
    print(f"wrote {args.report}")
    print(f"wrote {args.csv}")
    print(json.dumps({"status": result["status"], "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
