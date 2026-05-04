"""P2-02 DIESEL row-level pathlength/reference diagnostic audit.

Bench-only diagnostic audit for a predeclared row-level pathlength/reference
branch. The P2a profile inherits R3d and applies a positive row-wise factor to
the full generated wavelength row, not only to the 750-1550 nm DIESEL support.

This script adds no gate, promotion, threshold mutation, calibration, real-stat
capture, PCA/noise capture, ML/DL, labels, targets, splits, or production
integration.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
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


_exp09 = _load_module("exp09_sentinel_morphology_audit", "exp09_sentinel_morphology_audit.py")
_exp10 = _load_module("exp10_diesel_mean_shift_localization", "exp10_diesel_mean_shift_localization.py")
_exp15 = _load_module("exp15_diesel_pathlength_reference_attenuation_audit", "exp15_diesel_pathlength_reference_attenuation_audit.py")
_exp24 = _load_module("exp24_diesel_render_stage_failure_map", "exp24_diesel_render_stage_failure_map.py")

Exp25Row = _exp24.Exp24Row
align_to_real_grid = _exp24.align_to_real_grid
discover_local_real_datasets = _exp24.discover_local_real_datasets
is_index_fallback_grid = _exp24.is_index_fallback_grid
load_real_spectra = _exp24.load_real_spectra
sanitize_finite_spectra = _exp24.sanitize_finite_spectra

EXP25_AUDIT_SCOPE = "bench_only_p2_02_diesel_row_pathlength_reference_audit"
EXP25_DECISION = "diagnostic_only_no_gate_no_promotion_r3d_remains_baseline"
COMPARISON_SPACE = "uncalibrated_raw"
SUPPORT_LOW_NM: float = _exp10.SUPPORT_LOW_NM
SUPPORT_HIGH_NM: float = _exp10.SUPPORT_HIGH_NM

R3D_PROFILE = _exp24.R3D_PROFILE
R4B_PROFILE = _exp24.R4B_PROFILE
R4C_PROFILE = _exp24.R4C_PROFILE
R9E_PROFILE = _exp24.R9E_PROFILE
R9J_PROFILE = _exp24.R9J_PROFILE
R9L_PROFILE = _exp24.R9L_PROFILE
R9M_PROFILE = _exp24.R9M_PROFILE
P2A_PROFILE = "p2a_diesel_row_pathlength_reference_v1"

EXP25_AUDITED_PROFILES: tuple[str, ...] = (
    R3D_PROFILE,
    R9E_PROFILE,
    R9J_PROFILE,
    R9L_PROFILE,
    R9M_PROFILE,
    R4B_PROFILE,
    R4C_PROFILE,
    P2A_PROFILE,
)
EXP25_REFERENCE_PROFILES: tuple[str, ...] = (R3D_PROFILE, R9L_PROFILE, R4C_PROFILE)
EXP25_PAIRED_DELTA_ATTRS: tuple[str, ...] = _exp24.EXP24_PAIRED_DELTA_ATTRS

DEFAULT_SEEDS: tuple[int, ...] = (20260501, 20260502, 20260503)
DEFAULT_N_SYNTHETIC_SAMPLES = 64
DEFAULT_MAX_REAL_SAMPLES = 64
DEFAULT_MAX_SENTINEL_DATASETS = 8
DEFAULT_SENTINEL_TOKENS: tuple[str, ...] = ("DIESEL",)
DEFAULT_REPORT = Path("/tmp/exp25_diesel_row_pathlength_reference_audit.md")
DEFAULT_CSV = Path("/tmp/exp25_diesel_row_pathlength_reference_audit.csv")

PROFILE_STAGE_MAP: dict[str, dict[str, Any]] = {
    **_exp24.PROFILE_STAGE_MAP,
    P2A_PROFILE: {
        "render_stage_family": "row_level_pathlength_reference",
        "mechanism_family": "p2a_full_row_reference_attenuation",
        "stage_application": "after_r3d_output_clip_before_audit_alignment",
        "ch_width_gain_changed": False,
        "residual_damping_active": False,
        "clean_attenuation_active": False,
        "continuum_hump_active": False,
    },
}


def _audit_fields() -> dict[str, Any]:
    fields_dict = cast(dict[str, Any], _exp15._audit_fields())
    fields_dict["audit_scope"] = EXP25_AUDIT_SCOPE
    return fields_dict


def _validate_profiles(profiles: Sequence[str]) -> tuple[str, ...]:
    if not profiles:
        raise ValueError("at least one remediation profile must be provided")
    invalid = [profile for profile in profiles if profile not in EXP25_AUDITED_PROFILES]
    if invalid:
        raise ValueError(
            f"unknown exp25 profiles {invalid!r}; "
            f"valid profiles are {list(EXP25_AUDITED_PROFILES)}"
        )
    return tuple(profiles)


def _transform_params(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return cast(dict[str, Any], _exp24._transform_params(metadata))


def _metadata_stage_keys(transform_params: dict[str, Any]) -> str:
    keys = sorted(
        key
        for key in transform_params
        if any(
            token in key
            for token in (
                "application_stage",
                "route_key",
                "support_reference_attenuation",
                "row_pathlength_reference",
                "damping_windows_nm",
                "continuum_hump",
                "ch_overtone_width_nm",
                "ch_overtone_gain_range",
            )
        )
    )
    return ";".join(keys)


def _stage_support_nm(transform_params: dict[str, Any]) -> str:
    value = transform_params.get("row_pathlength_reference_applies_to")
    if value == "full_generated_wavelength_row":
        return "full_generated_row"
    return cast(str, _exp24._stage_support_nm(transform_params))


def _stage_n_support_bins(transform_params: dict[str, Any]) -> int | None:
    value = transform_params.get("row_pathlength_reference_n_wavelengths")
    if value is not None:
        return int(value)
    return cast(int | None, _exp24._stage_n_support_bins(transform_params))


def _stage_metadata(profile: str, metadata: dict[str, Any] | None) -> dict[str, Any]:
    transform_params = _transform_params(metadata)
    static = dict(PROFILE_STAGE_MAP[profile])
    stage_application = str(
        transform_params.get(
            "row_pathlength_reference_application_stage",
            transform_params.get(
                "support_reference_attenuation_application_stage",
                static["stage_application"],
            ),
        )
    )
    clean_attenuation_active = bool(
        static["clean_attenuation_active"]
        or "support_reference_attenuation_factor_range" in transform_params
    )
    row_pathlength_active = "row_pathlength_reference_factor_range" in transform_params
    residual_damping_active = bool(
        static["residual_damping_active"] or "damping_windows_nm" in transform_params
    )
    continuum_hump_active = bool(
        static["continuum_hump_active"] or "continuum_hump_center_nm" in transform_params
    )
    return {
        **static,
        "stage_application": stage_application,
        "stage_support_nm": _stage_support_nm(transform_params),
        "stage_n_support_bins": _stage_n_support_bins(transform_params),
        "ch_width_gain_changed": bool(static["ch_width_gain_changed"]),
        "residual_damping_active": residual_damping_active,
        "clean_attenuation_active": clean_attenuation_active,
        "continuum_hump_active": continuum_hump_active,
        "support_only_post_clip_active": (
            stage_application == "after_r3d_output_clip" and not row_pathlength_active
        ),
        "extra_guard_clip_active": bool(
            transform_params.get("support_intercept_guard_clip_fraction", 0.0)
            or transform_params.get("support_shape_guard_clip_fraction", 0.0)
            or transform_params.get(
                "support_centered_micro_path_modulation_guard_clip_fraction",
                0.0,
            )
        ),
        "metadata_stage_keys": _metadata_stage_keys(transform_params),
    }


def _row_stage_and_failure(
    *,
    profile: str,
    metadata: dict[str, Any] | None,
    status: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    axis, score, note = _exp24._failure_axis_and_score(
        status=status,
        dominant_morphology_gap=str(metrics["dominant_morphology_gap"]),
        support_count=int(metrics["support_count"]),
        off_support_count=int(metrics["off_support_count"]),
        support_weighted_delta=cast(float | None, metrics["support_weighted_delta"]),
        off_support_weighted_delta=cast(float | None, metrics["off_support_weighted_delta"]),
        support_mean_delta=cast(float | None, metrics["support_mean_delta"]),
        global_mean_delta=cast(float | None, metrics["global_mean_delta"]),
        log10_derivative_std_p50_ratio=cast(float | None, metrics["log10_derivative_std_p50_ratio"]),
        mean_curve_corr=cast(float | None, metrics["mean_curve_corr"]),
        morphology_gap_score=cast(float | None, metrics["morphology_gap_score"]),
    )
    return {
        **_stage_metadata(profile, metadata),
        "report_only_failure_axis": axis,
        "report_only_failure_score": score,
        "report_only_failure_note": note,
    }


def _p2a_token_source_overrides(dataset: Any) -> dict[str, object]:
    text, tokens = _exp09._dataset_text_and_tokens(dataset)
    if not _exp09._has_r2j_diesel_marker(text, tokens):
        return cast(dict[str, object], _exp09._r3d_token_source_overrides(dataset))
    overrides = cast(dict[str, object], _exp09._r2c_token_source_overrides(dataset))
    route = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp25_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    overrides["_r3d_diesel_readout_route"] = dict(route)
    overrides["_p2a_diesel_row_pathlength_reference_route"] = dict(route)
    return overrides


def _effective_profile_for_dataset(dataset: Any, profile: str | None) -> str | None:
    if profile == P2A_PROFILE:
        text, tokens = _exp09._dataset_text_and_tokens(dataset)
        if _exp09._has_r2j_diesel_marker(text, tokens):
            return profile
        return cast(str | None, _exp09._effective_remediation_profile_for_dataset(dataset, R3D_PROFILE))
    return cast(str | None, _exp24._effective_profile_for_dataset(dataset, profile))


def _build_synthetic_run(
    *,
    dataset: Any,
    preset: str,
    n_samples: int,
    seed: int,
    remediation_profile: str,
) -> Any:
    effective = _effective_profile_for_dataset(dataset, remediation_profile)
    if remediation_profile != P2A_PROFILE:
        return _exp24._build_synthetic_run(
            dataset=dataset,
            preset=preset,
            n_samples=n_samples,
            seed=seed,
            remediation_profile=remediation_profile,
        )
    if effective != P2A_PROFILE:
        return _exp09._build_baseline_synthetic_run(
            dataset=dataset,
            preset=preset,
            n_samples=n_samples,
            seed=seed,
            remediation_profile=effective,
        )
    run_seed = _exp09._stable_dataset_seed(seed, dataset, f"r2b:on_demand_synthetic:{preset}")
    source = _exp09._preset_source(
        preset,
        seed=run_seed,
        extra_overrides=_p2a_token_source_overrides(dataset),
    )
    record = _exp09.canonicalize_prior_config(source)
    return _exp09.build_synthetic_dataset_run(
        record,
        n_samples=n_samples,
        random_seed=run_seed,
        remediation_profile=effective,
    )


def _guard_clip_fraction(profile: str, metadata: dict[str, Any] | None) -> float | None:
    if profile == P2A_PROFILE:
        value = _transform_params(metadata).get("row_pathlength_reference_guard_clip_fraction")
        return None if value is None else float(value)
    return cast(float | None, _exp24._guard_clip_fraction(profile, metadata))


def _empty_metric_fields() -> dict[str, Any]:
    return cast(dict[str, Any], _exp24._empty_metric_fields())


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
    metrics = _empty_metric_fields()
    return Exp25Row(
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
        **_row_stage_and_failure(
            profile=requested_profile,
            metadata=None,
            status="blocked",
            metrics=metrics,
        ),
        n_real_samples=0,
        n_synthetic_samples=0,
        n_wavelengths=0,
        wavelength_min=None,
        wavelength_max=None,
        **metrics,
        guard_clip_fraction=None,
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
        profiles if profiles is not None else EXP25_AUDITED_PROFILES
    )
    tokens = tuple(DEFAULT_SENTINEL_TOKENS) if sentinel_tokens is None else tuple(sentinel_tokens)
    real_datasets, _ = discover_local_real_datasets(root)
    sentinel_candidates = _exp09._select_sentinel_datasets(real_datasets, tokens)
    selected = list(sentinel_candidates) if max_sentinel_datasets <= 0 else sentinel_candidates[:max_sentinel_datasets]
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
            real_x_raw, real_wl_raw = load_real_spectra(dataset, root=root)
        except Exception as exc:  # noqa: BLE001
            for seed in seeds:
                for profile in audited_profiles:
                    rows.append(
                        _blocked_row(
                            seed=seed,
                            requested_profile=profile,
                            effective_profile=_effective_profile_for_dataset(dataset, profile),
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
                            effective_profile=_effective_profile_for_dataset(dataset, profile),
                            dataset=dataset,
                            preset=preset,
                            blocked_reason="wavelength_grid_unknown",
                        )
                    )
            continue

        sanitized_real, sanitized_wl, _, real_blocked = sanitize_finite_spectra(
            real_x_raw,
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
                            effective_profile=_effective_profile_for_dataset(dataset, profile),
                            dataset=dataset,
                            preset=preset,
                            blocked_reason=f"non_finite_spectra: {real_blocked}",
                        )
                    )
            continue

        for seed in seeds:
            real_x = _exp09._downsample_rows(
                sanitized_real,
                max_rows=max_real_samples,
                random_state=_exp09._stable_dataset_seed(seed, dataset, "exp25:real_downsample"),
            )
            for profile in audited_profiles:
                effective = _effective_profile_for_dataset(dataset, profile)
                effective_route: str | None = None
                metadata: dict[str, Any] | None = None
                try:
                    synthetic_run = _build_synthetic_run(
                        dataset=dataset,
                        preset=preset,
                        n_samples=n_synthetic_samples,
                        seed=seed,
                        remediation_profile=profile,
                    )
                    metadata = synthetic_run.metadata
                    effective_route = cast(str | None, _exp24._effective_matrix_route(metadata))
                    sanitized_syn, sanitized_syn_wl, _, syn_blocked = sanitize_finite_spectra(
                        np.asarray(synthetic_run.X, dtype=float),
                        np.asarray(synthetic_run.wavelengths, dtype=float),
                        side="synthetic",
                    )
                    if syn_blocked is not None or sanitized_syn is None or sanitized_syn_wl is None:
                        raise ValueError(f"non_finite_spectra_synthetic: {syn_blocked}")
                    synth_downsampled = _exp09._downsample_rows(
                        sanitized_syn,
                        max_rows=max_real_samples,
                        random_state=_exp09._stable_dataset_seed(seed, dataset, "exp25:syn_downsample"),
                    )
                    real_aligned, syn_aligned, aligned_wl = align_to_real_grid(
                        real_x,
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

                morphology = _exp09.compute_morphology_metrics(real_aligned, syn_aligned, aligned_wl)
                decomposition = _exp10.compute_support_decomposition(
                    real_aligned,
                    syn_aligned,
                    aligned_wl,
                    support_low_nm=support_low_nm,
                    support_high_nm=support_high_nm,
                )
                metric_fields = {
                    "support_low_nm": decomposition["support_low_nm"],
                    "support_high_nm": decomposition["support_high_nm"],
                    "support_count": decomposition["support_count"],
                    "off_support_count": decomposition["off_support_count"],
                    "support_weight": decomposition["support_weight"],
                    "off_support_weight": decomposition["off_support_weight"],
                    "support_mean_delta": decomposition["support_mean_delta"],
                    "off_support_mean_delta": decomposition["off_support_mean_delta"],
                    "support_weighted_delta": decomposition["support_weighted_delta"],
                    "off_support_weighted_delta": decomposition["off_support_weighted_delta"],
                    "global_mean_delta": decomposition["global_mean_delta"],
                    "decomposition_residual": decomposition["decomposition_residual"],
                    **_exp15._morphology_subset(morphology),
                }
                rows.append(
                    Exp25Row(
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
                        **_row_stage_and_failure(
                            profile=profile,
                            metadata=metadata,
                            status="compared",
                            metrics=metric_fields,
                        ),
                        n_real_samples=int(real_aligned.shape[0]),
                        n_synthetic_samples=int(syn_aligned.shape[0]),
                        n_wavelengths=morphology["n_wavelengths"],
                        wavelength_min=morphology["wavelength_min"],
                        wavelength_max=morphology["wavelength_max"],
                        **metric_fields,
                        guard_clip_fraction=_guard_clip_fraction(profile, metadata),
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


def _row_key(row: Any) -> tuple[int, str]:
    return (int(row.seed), str(row.dataset))


def _delta_col(reference: str, attr: str) -> str:
    return f"delta_vs_{reference}__{attr}"


def _paired_delta_columns() -> list[str]:
    return [
        _delta_col(reference, attr)
        for reference in EXP25_REFERENCE_PROFILES
        for attr in EXP25_PAIRED_DELTA_ATTRS
    ]


def _csv_fieldnames() -> list[str]:
    return [field.name for field in fields(Exp25Row)] + _paired_delta_columns()


def _reference_rows(rows: Sequence[Any]) -> dict[str, dict[tuple[int, str], Any]]:
    references: dict[str, dict[tuple[int, str], Any]] = {
        profile: {} for profile in EXP25_REFERENCE_PROFILES
    }
    for row in rows:
        if row.status == "compared" and row.remediation_profile in references:
            references[row.remediation_profile][_row_key(row)] = row
    return references


def _row_reference_deltas(
    row: Any,
    references: dict[str, dict[tuple[int, str], Any]],
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for reference_profile, by_key in references.items():
        ref = by_key.get(_row_key(row))
        for attr in EXP25_PAIRED_DELTA_ATTRS:
            col = _delta_col(reference_profile, attr)
            if row.status != "compared" or ref is None:
                out[col] = None
                continue
            lhs = getattr(row, attr)
            rhs = getattr(ref, attr)
            out[col] = None if lhs is None or rhs is None else float(lhs) - float(rhs)
    return out


def write_csv(rows: list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    references = _reference_rows(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_csv_fieldnames(), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            record = row.to_dict()
            record.update(_row_reference_deltas(row, references))
            writer.writerow(record)


def _fmt(value: float | None) -> str:
    return cast(str, _exp15._fmt(value))


def _median(values: Iterable[float | None]) -> float | None:
    return cast(float | None, _exp15._median(values))


def _format_counter(values: Iterable[str]) -> str:
    counts = Counter(values)
    return ", ".join(f"{label}={count}" for label, count in counts.most_common()) or "n/a"


def _aggregate_by_profile(rows: Sequence[Any]) -> list[dict[str, Any]]:
    compared = [row for row in rows if row.status == "compared"]
    out: list[dict[str, Any]] = []
    for profile in EXP25_AUDITED_PROFILES:
        bucket = [row for row in compared if row.remediation_profile == profile]
        if not bucket:
            continue
        out.append(
            {
                "profile": profile,
                "n": len(bucket),
                "median_global_mean_delta": _median(row.global_mean_delta for row in bucket),
                "median_support_mean_delta": _median(row.support_mean_delta for row in bucket),
                "median_morphology_gap_score": _median(row.morphology_gap_score for row in bucket),
                "median_log10_derivative": _median(row.log10_derivative_std_p50_ratio for row in bucket),
                "median_mean_curve_corr": _median(row.mean_curve_corr for row in bucket),
                "dominant_gaps": _format_counter(row.dominant_morphology_gap for row in bucket),
                "failure_axes": _format_counter(row.report_only_failure_axis for row in bucket),
            }
        )
    return out


def render_markdown(
    *,
    result: dict[str, Any],
    report_path: Path,
    csv_path: Path,
    n_synthetic_samples: int,
    max_real_samples: int,
    max_sentinel_datasets: int,
    seeds: Sequence[int],
    sentinel_tokens: Sequence[str],
    profiles: Sequence[str],
) -> str:
    rows = cast(list[Any], result["rows"])
    compared = [row for row in rows if row.status == "compared"]
    blocked = [row for row in rows if row.status != "compared"]
    p2a_rows = [row for row in compared if row.remediation_profile == P2A_PROFILE]
    p2a_guard_max = _median(row.guard_clip_fraction for row in p2a_rows)
    lines = [
        "# P2-02 DIESEL Row-Level Pathlength/Reference Audit",
        "",
        f"- audit_scope: `{EXP25_AUDIT_SCOPE}`",
        f"- decision: `{EXP25_DECISION}`",
        f"- comparison_space: `{COMPARISON_SPACE}`",
        f"- report: `{report_path}`",
        f"- csv: `{csv_path}`",
        f"- rows: compared={len(compared)}, blocked={len(blocked)}",
        f"- seeds: `{','.join(str(seed) for seed in seeds)}`",
        f"- n_synthetic_samples: `{n_synthetic_samples}`",
        f"- max_real_samples: `{max_real_samples}`",
        f"- max_sentinel_datasets: `{max_sentinel_datasets}`",
        f"- sentinel_tokens: `{','.join(sentinel_tokens)}`",
        f"- profiles: `{','.join(profiles)}`",
        "",
        "## Contract",
        "",
        "- R3d remains the accepted DIESEL baseline.",
        "- P2a is diagnostic-only, bench-only, not promoted, and not a gate.",
        "- No R9n is created or referenced.",
        "- No calibration, real-stat capture, PCA/noise capture, ML/DL, labels, targets, splits, threshold mutation, metric mutation, or production integration is used.",
        "- P2a uses predeclared constants only and applies a full-row pathlength/reference factor before audit-side wavelength alignment.",
        "",
        "## Aggregate By Profile",
        "",
        "| profile | n | median gap | median global mean delta | median support mean delta | median derivative log10 | median mean curve corr | dominant gaps | failure axes |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in _aggregate_by_profile(rows):
        lines.append(
            "| {profile} | {n} | {gap} | {global_delta} | {support_delta} | {deriv} | {corr} | {gaps} | {axes} |".format(
                profile=row["profile"],
                n=row["n"],
                gap=_fmt(row["median_morphology_gap_score"]),
                global_delta=_fmt(row["median_global_mean_delta"]),
                support_delta=_fmt(row["median_support_mean_delta"]),
                deriv=_fmt(row["median_log10_derivative"]),
                corr=_fmt(row["median_mean_curve_corr"]),
                gaps=row["dominant_gaps"],
                axes=row["failure_axes"],
            )
        )
    lines.extend(
        [
            "",
            "## P2a Provenance",
            "",
            f"- p2a_rows_compared: `{len(p2a_rows)}`",
            f"- p2a_guard_clip_fraction_median: `{_fmt(p2a_guard_max)}`",
            "- mechanism: `positive_row_level_pathlength_reference_attenuation_full_wavelength_row`",
            "- factor_range: `[0.970, 0.985]`, inherited unchanged from the predeclared R9e blank/reference prior.",
            "- route_key: `_p2a_diesel_row_pathlength_reference_route`",
            "- stage: `after_r3d_output_clip_before_audit_alignment`",
            "- support_only: `False`",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python bench/nirs_synthetic_pfn/experiments/exp25_diesel_row_pathlength_reference_audit.py \\",
            f"  --report {report_path} \\",
            f"  --csv {csv_path}",
            "```",
        ]
    )
    if blocked:
        lines.extend(["", "## Blocked Rows", ""])
        for reason, count in Counter(row.blocked_reason for row in blocked).most_common():
            lines.append(f"- {reason}: {count}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-synthetic-samples", type=int, default=DEFAULT_N_SYNTHETIC_SAMPLES)
    parser.add_argument("--max-real-samples", type=int, default=DEFAULT_MAX_REAL_SAMPLES)
    parser.add_argument("--max-sentinel-datasets", type=int, default=DEFAULT_MAX_SENTINEL_DATASETS)
    parser.add_argument("--sentinel-token", dest="sentinel_tokens", action="append")
    parser.add_argument("--profile", dest="profiles", action="append")
    args = parser.parse_args()

    profiles = _validate_profiles(args.profiles or EXP25_AUDITED_PROFILES)
    sentinel_tokens = tuple(args.sentinel_tokens or DEFAULT_SENTINEL_TOKENS)
    result = run_audit(
        root=args.root,
        seeds=args.seeds,
        n_synthetic_samples=args.n_synthetic_samples,
        max_real_samples=args.max_real_samples,
        max_sentinel_datasets=args.max_sentinel_datasets,
        sentinel_tokens=sentinel_tokens,
        profiles=profiles,
    )
    write_csv(cast(list[Any], result["rows"]), args.csv)
    markdown = render_markdown(
        result=result,
        report_path=args.report,
        csv_path=args.csv,
        n_synthetic_samples=args.n_synthetic_samples,
        max_real_samples=args.max_real_samples,
        max_sentinel_datasets=args.max_sentinel_datasets,
        seeds=args.seeds,
        sentinel_tokens=sentinel_tokens,
        profiles=profiles,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
