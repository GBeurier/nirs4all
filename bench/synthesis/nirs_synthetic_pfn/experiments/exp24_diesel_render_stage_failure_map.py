"""P2-01 DIESEL render-stage failure-map benchmark.

Bench-only read-only audit for the Palier 2 failure map. This script compares
existing DIESEL profiles on identical rows and extracts already-emitted builder
metadata to classify which render-stage family owns each residual failure mode.

It introduces no generator mechanism, no profile, no retune, no gate, no
threshold or metric mutation, and no production integration.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
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
_exp23 = _load_module(
    "exp23_diesel_width_gain_damping_clean_attenuation_audit",
    "exp23_diesel_width_gain_damping_clean_attenuation_audit.py",
)

align_to_real_grid = _exp15.align_to_real_grid
discover_local_real_datasets = _exp15.discover_local_real_datasets
is_index_fallback_grid = _exp15.is_index_fallback_grid
load_real_spectra = _exp15.load_real_spectra
sanitize_finite_spectra = _exp15.sanitize_finite_spectra

EXP24_AUDIT_SCOPE = "bench_only_p2_01_diesel_render_stage_failure_map"
EXP24_DECISION = "diagnostic_read_only_failure_map_no_gate_no_promotion"
COMPARISON_SPACE = "uncalibrated_raw"
SUPPORT_LOW_NM: float = _exp10.SUPPORT_LOW_NM
SUPPORT_HIGH_NM: float = _exp10.SUPPORT_HIGH_NM

R3D_PROFILE = "r3d_diesel_matrix_v1"
R4B_PROFILE = "r4b_diesel_derivative_restore_v1"
R4C_PROFILE = "r4c_diesel_balanced_derivative_v1"
R9E_PROFILE = "r9e_diesel_pathlength_reference_attenuation_v1"
R9J_PROFILE = "r9j_diesel_residual_damping_isolation_v1"
R9L_PROFILE = "r9l_diesel_residual_damping_clean_attenuation_v1"
R9M_PROFILE = "r9m_diesel_width_gain_damping_clean_attenuation_v1"

EXP24_AUDITED_PROFILES: tuple[str, ...] = (
    R3D_PROFILE,
    R9E_PROFILE,
    R9J_PROFILE,
    R9L_PROFILE,
    R9M_PROFILE,
    R4B_PROFILE,
    R4C_PROFILE,
)
EXP24_REFERENCE_PROFILES: tuple[str, ...] = (R3D_PROFILE, R4C_PROFILE)
EXP24_PAIRED_DELTA_ATTRS: tuple[str, ...] = (
    "global_mean_delta",
    "support_mean_delta",
    "support_weighted_delta",
    "off_support_weighted_delta",
    "morphology_gap_score",
    "log10_derivative_std_p50_ratio",
    "mean_curve_corr",
)

DEFAULT_SEEDS: tuple[int, ...] = (20260501, 20260502, 20260503)
DEFAULT_N_SYNTHETIC_SAMPLES = 64
DEFAULT_MAX_REAL_SAMPLES = 64
DEFAULT_MAX_SENTINEL_DATASETS = 8
DEFAULT_SENTINEL_TOKENS: tuple[str, ...] = ("DIESEL",)
DEFAULT_REPORT = Path("/tmp/exp24_diesel_render_stage_failure_map.md")
DEFAULT_CSV = Path("/tmp/exp24_diesel_render_stage_failure_map.csv")

PROFILE_STAGE_MAP: dict[str, dict[str, Any]] = {
    R3D_PROFILE: {
        "render_stage_family": "r3d_absorbance_pipeline",
        "mechanism_family": "accepted_baseline_micro_path_ch_overtone_clip",
        "stage_application": "continuum_path_plus_feature_residual_plus_baseline_then_output_clip",
        "ch_width_gain_changed": False,
        "residual_damping_active": False,
        "clean_attenuation_active": False,
        "continuum_hump_active": False,
    },
    R9E_PROFILE: {
        "render_stage_family": "post_clip_support_reference_attenuation",
        "mechanism_family": "r9e_clean_support_attenuation_only",
        "stage_application": "after_r3d_output_clip",
        "ch_width_gain_changed": False,
        "residual_damping_active": False,
        "clean_attenuation_active": True,
        "continuum_hump_active": False,
    },
    R9J_PROFILE: {
        "render_stage_family": "pre_baseline_residual_damping",
        "mechanism_family": "r9j_residual_damping_only",
        "stage_application": "before_additive_baseline_and_output_clip",
        "ch_width_gain_changed": False,
        "residual_damping_active": True,
        "clean_attenuation_active": False,
        "continuum_hump_active": False,
    },
    R9L_PROFILE: {
        "render_stage_family": "pre_baseline_damping_plus_post_clip_attenuation",
        "mechanism_family": "r9j_residual_damping_plus_r9e_clean_attenuation",
        "stage_application": "before_baseline_damping_then_after_r3d_output_clip_attenuation",
        "ch_width_gain_changed": False,
        "residual_damping_active": True,
        "clean_attenuation_active": True,
        "continuum_hump_active": False,
    },
    R9M_PROFILE: {
        "render_stage_family": "width_gain_damping_plus_post_clip_attenuation",
        "mechanism_family": "r9i_width_gain_plus_r9j_damping_plus_r9e_attenuation",
        "stage_application": "ch_width_gain_before_baseline_damping_then_after_clip_attenuation",
        "ch_width_gain_changed": True,
        "residual_damping_active": True,
        "clean_attenuation_active": True,
        "continuum_hump_active": False,
    },
    R4B_PROFILE: {
        "render_stage_family": "r4_support_basis_bundle",
        "mechanism_family": "r4b_width_gain_damping_continuum_hump_bundle",
        "stage_application": "pre_baseline_basis_damping_hump_then_output_clip",
        "ch_width_gain_changed": True,
        "residual_damping_active": True,
        "clean_attenuation_active": False,
        "continuum_hump_active": True,
    },
    R4C_PROFILE: {
        "render_stage_family": "r4_support_basis_bundle",
        "mechanism_family": "r4c_balanced_width_gain_damping_continuum_hump_bundle",
        "stage_application": "pre_baseline_basis_damping_hump_then_output_clip",
        "ch_width_gain_changed": True,
        "residual_damping_active": True,
        "clean_attenuation_active": False,
        "continuum_hump_active": True,
    },
}


@dataclass
class Exp24Row:
    status: str
    seed: int
    remediation_profile: str
    effective_remediation_profile: str | None
    source: str
    task: str
    dataset: str
    synthetic_preset: str
    effective_matrix_route: str | None
    comparison_space: str
    render_stage_family: str
    mechanism_family: str
    stage_application: str
    stage_support_nm: str
    stage_n_support_bins: int | None
    ch_width_gain_changed: bool
    residual_damping_active: bool
    clean_attenuation_active: bool
    continuum_hump_active: bool
    support_only_post_clip_active: bool
    extra_guard_clip_active: bool
    metadata_stage_keys: str
    report_only_failure_axis: str
    report_only_failure_score: float | None
    report_only_failure_note: str
    n_real_samples: int
    n_synthetic_samples: int
    n_wavelengths: int
    wavelength_min: float | None
    wavelength_max: float | None
    support_low_nm: float
    support_high_nm: float
    support_count: int
    off_support_count: int
    support_weight: float
    off_support_weight: float
    support_mean_delta: float | None
    off_support_mean_delta: float | None
    support_weighted_delta: float | None
    off_support_weighted_delta: float | None
    global_mean_delta: float | None
    decomposition_residual: float | None
    real_global_mean: float | None
    synthetic_global_mean: float | None
    real_global_std: float | None
    synthetic_global_std: float | None
    log10_global_std_ratio: float | None
    log10_amplitude_p50_ratio: float | None
    log10_derivative_std_p50_ratio: float | None
    mean_curve_corr: float | None
    morphology_gap_score: float | None
    dominant_morphology_gap: str
    guard_clip_fraction: float | None
    audit_calibration: bool
    audit_real_stat_capture: bool
    audit_uses_pca: bool
    audit_captures_noise: bool
    audit_uses_ml: bool
    audit_uses_dl: bool
    audit_label_inputs_used: bool
    audit_target_inputs_used: bool
    audit_split_inputs_used: bool
    audit_thresholds_modified: bool
    audit_metrics_modified: bool
    audit_source_oracle_used: bool
    audit_scope: str
    blocked_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _audit_fields() -> dict[str, Any]:
    fields_dict = cast(dict[str, Any], _exp15._audit_fields())
    fields_dict["audit_scope"] = EXP24_AUDIT_SCOPE
    return fields_dict


def _validate_profiles(profiles: Sequence[str]) -> tuple[str, ...]:
    if not profiles:
        raise ValueError("at least one remediation profile must be provided")
    invalid = [profile for profile in profiles if profile not in EXP24_AUDITED_PROFILES]
    if invalid:
        raise ValueError(
            f"unknown exp24 profiles {invalid!r}; "
            f"valid profiles are {list(EXP24_AUDITED_PROFILES)}"
        )
    return tuple(profiles)


def _transform_params(metadata: dict[str, Any] | None) -> dict[str, Any]:
    audit = (metadata or {}).get("r2c_mechanistic_remediation") or {}
    return cast(dict[str, Any], audit.get("transform_params") or {})


def _metadata_stage_keys(transform_params: dict[str, Any]) -> str:
    stage_tokens = (
        "application_stage",
        "route_key",
        "support_reference_attenuation",
        "damping_windows_nm",
        "continuum_hump",
        "ch_overtone_width_nm",
        "ch_overtone_gain_range",
    )
    keys = sorted(
        key
        for key in transform_params
        if any(token in key for token in stage_tokens)
    )
    return ";".join(keys)


def _stage_support_nm(transform_params: dict[str, Any]) -> str:
    for key in (
        "support_reference_attenuation_support_nm",
        "continuum_hump_support_nm",
        "support_shape_support_nm",
    ):
        value = transform_params.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return f"{float(value[0]):g}-{float(value[1]):g}"
    return f"{SUPPORT_LOW_NM:g}-{SUPPORT_HIGH_NM:g}"


def _stage_n_support_bins(transform_params: dict[str, Any]) -> int | None:
    value = transform_params.get("support_reference_attenuation_n_support_bins")
    if value is not None:
        return int(value)
    value = transform_params.get("continuum_hump_n_support_bins")
    return None if value is None else int(value)


def _stage_metadata(
    profile: str,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    transform_params = _transform_params(metadata)
    static = dict(PROFILE_STAGE_MAP[profile])
    stage_application = str(
        transform_params.get(
            "support_reference_attenuation_application_stage",
            static["stage_application"],
        )
    )
    clean_attenuation_active = bool(
        static["clean_attenuation_active"]
        or "support_reference_attenuation_factor_range" in transform_params
    )
    residual_damping_active = bool(
        static["residual_damping_active"] or "damping_windows_nm" in transform_params
    )
    continuum_hump_active = bool(
        static["continuum_hump_active"] or "continuum_hump_center_nm" in transform_params
    )
    ch_width_gain_changed = bool(static["ch_width_gain_changed"])
    extra_guard_clip_active = bool(
        transform_params.get("support_intercept_guard_clip_fraction", 0.0)
        or transform_params.get("support_shape_guard_clip_fraction", 0.0)
        or transform_params.get("support_centered_micro_path_modulation_guard_clip_fraction", 0.0)
    )
    return {
        **static,
        "stage_application": stage_application,
        "stage_support_nm": _stage_support_nm(transform_params),
        "stage_n_support_bins": _stage_n_support_bins(transform_params),
        "ch_width_gain_changed": ch_width_gain_changed,
        "residual_damping_active": residual_damping_active,
        "clean_attenuation_active": clean_attenuation_active,
        "continuum_hump_active": continuum_hump_active,
        "support_only_post_clip_active": stage_application == "after_r3d_output_clip",
        "extra_guard_clip_active": extra_guard_clip_active,
        "metadata_stage_keys": _metadata_stage_keys(transform_params),
    }


def _failure_axis_and_score(
    *,
    status: str,
    dominant_morphology_gap: str,
    support_count: int,
    off_support_count: int,
    support_weighted_delta: float | None,
    off_support_weighted_delta: float | None,
    support_mean_delta: float | None,
    global_mean_delta: float | None,
    log10_derivative_std_p50_ratio: float | None,
    mean_curve_corr: float | None,
    morphology_gap_score: float | None,
) -> tuple[str, float | None, str]:
    if status != "compared":
        return "blocked", None, "row blocked before metric computation"
    if dominant_morphology_gap in {"derivative_under", "derivative_over"}:
        score = (
            None
            if log10_derivative_std_p50_ratio is None
            else abs(float(log10_derivative_std_p50_ratio))
        )
        return dominant_morphology_gap, score, "existing morphology dominant gap"
    if dominant_morphology_gap == "mean_curve_inversion":
        score = None if mean_curve_corr is None else 1.0 - float(mean_curve_corr)
        return "correlation", score, "existing morphology dominant gap"
    support_abs = (
        -1.0 if support_weighted_delta is None else abs(float(support_weighted_delta))
    )
    off_support_abs = (
        -1.0
        if off_support_weighted_delta is None
        else abs(float(off_support_weighted_delta))
    )
    if support_count > 0 and off_support_count == 0:
        score = None if support_mean_delta is None else abs(float(support_mean_delta))
        return "support_mean_drives_global_mean", score, "all compared wavelengths are inside the DIESEL support window"
    if support_abs >= off_support_abs and support_abs >= 0.0:
        score = None if support_mean_delta is None else abs(float(support_mean_delta))
        return "support_mean_delta", score, "support weighted mean delta dominates off-support delta"
    if off_support_abs >= 0.0:
        return "off_support_mean_delta", off_support_abs, "off-support weighted mean delta dominates support delta"
    score = None if global_mean_delta is None else abs(float(global_mean_delta))
    if score is not None:
        return "global_mean_delta", score, "global mean delta is the available level diagnostic"
    return "morphology_gap", morphology_gap_score, "morphology gap is the available diagnostic"


def _row_stage_and_failure(
    *,
    profile: str,
    metadata: dict[str, Any] | None,
    status: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    stage = _stage_metadata(profile, metadata)
    axis, score, note = _failure_axis_and_score(
        status=status,
        dominant_morphology_gap=str(metrics["dominant_morphology_gap"]),
        support_count=int(metrics["support_count"]),
        off_support_count=int(metrics["off_support_count"]),
        support_weighted_delta=cast(float | None, metrics["support_weighted_delta"]),
        off_support_weighted_delta=cast(
            float | None,
            metrics["off_support_weighted_delta"],
        ),
        support_mean_delta=cast(float | None, metrics["support_mean_delta"]),
        global_mean_delta=cast(float | None, metrics["global_mean_delta"]),
        log10_derivative_std_p50_ratio=cast(
            float | None,
            metrics["log10_derivative_std_p50_ratio"],
        ),
        mean_curve_corr=cast(float | None, metrics["mean_curve_corr"]),
        morphology_gap_score=cast(float | None, metrics["morphology_gap_score"]),
    )
    return {
        **stage,
        "report_only_failure_axis": axis,
        "report_only_failure_score": score,
        "report_only_failure_note": note,
    }


def _effective_profile_for_dataset(dataset: Any, profile: str | None) -> str | None:
    return cast(str | None, _exp23._effective_profile_for_dataset(dataset, profile))


def _build_synthetic_run(
    *,
    dataset: Any,
    preset: str,
    n_samples: int,
    seed: int,
    remediation_profile: str,
) -> Any:
    return _exp23._build_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=n_samples,
        seed=seed,
        remediation_profile=remediation_profile,
    )


def _guard_clip_fraction(
    profile: str,
    metadata: dict[str, Any] | None,
) -> float | None:
    transform_params = _transform_params(metadata)
    value = transform_params.get("support_reference_attenuation_guard_clip_fraction")
    if value is not None:
        return float(value)
    return cast(float | None, _exp23._guard_clip_fraction(profile, metadata))


def _effective_matrix_route(metadata: dict[str, Any] | None) -> str | None:
    return cast(str | None, _exp23._effective_matrix_route(metadata))


def _empty_metric_fields() -> dict[str, Any]:
    return {
        **_exp15._empty_decomposition(),
        **_exp15._empty_morphology_subset(),
    }


def _blocked_row(
    *,
    seed: int,
    requested_profile: str,
    effective_profile: str | None,
    dataset: Any,
    preset: str,
    blocked_reason: str,
    effective_matrix_route: str | None = None,
) -> Exp24Row:
    metrics = _empty_metric_fields()
    return Exp24Row(
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
        profiles if profiles is not None else EXP24_AUDITED_PROFILES
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
    rows: list[Exp24Row] = []
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
            real_x = _exp09._downsample_rows(
                sanitized_real,
                max_rows=max_real_samples,
                random_state=_exp09._stable_dataset_seed(
                    seed,
                    dataset,
                    "exp24:real_downsample",
                ),
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
                    effective_route = _effective_matrix_route(metadata)
                    sanitized_syn, sanitized_syn_wl, _, syn_blocked = (
                        sanitize_finite_spectra(
                            np.asarray(synthetic_run.X, dtype=float),
                            np.asarray(synthetic_run.wavelengths, dtype=float),
                            side="synthetic",
                        )
                    )
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
                            "exp24:syn_downsample",
                        ),
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

                morphology = _exp09.compute_morphology_metrics(
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
                    "off_support_weighted_delta": decomposition[
                        "off_support_weighted_delta"
                    ],
                    "global_mean_delta": decomposition["global_mean_delta"],
                    "decomposition_residual": decomposition["decomposition_residual"],
                    **_exp15._morphology_subset(morphology),
                }
                rows.append(
                    Exp24Row(
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


def _row_key(row: Exp24Row) -> tuple[int, str]:
    return (int(row.seed), str(row.dataset))


def _delta_col(reference: str, attr: str) -> str:
    return f"delta_vs_{reference}__{attr}"


def _paired_delta_columns() -> list[str]:
    return [
        _delta_col(reference, attr)
        for reference in EXP24_REFERENCE_PROFILES
        for attr in EXP24_PAIRED_DELTA_ATTRS
    ]


def _csv_fieldnames() -> list[str]:
    return [field.name for field in fields(Exp24Row)] + _paired_delta_columns()


def _reference_rows(rows: Sequence[Exp24Row]) -> dict[str, dict[tuple[int, str], Exp24Row]]:
    references: dict[str, dict[tuple[int, str], Exp24Row]] = {
        profile: {} for profile in EXP24_REFERENCE_PROFILES
    }
    for row in rows:
        if row.status == "compared" and row.remediation_profile in references:
            references[row.remediation_profile][_row_key(row)] = row
    return references


def _row_reference_deltas(
    row: Exp24Row,
    references: dict[str, dict[tuple[int, str], Exp24Row]],
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for reference_profile, by_key in references.items():
        ref = by_key.get(_row_key(row))
        for attr in EXP24_PAIRED_DELTA_ATTRS:
            col = _delta_col(reference_profile, attr)
            if row.status != "compared" or ref is None:
                out[col] = None
                continue
            lhs = getattr(row, attr)
            rhs = getattr(ref, attr)
            out[col] = None if lhs is None or rhs is None else float(lhs) - float(rhs)
    return out


def write_csv(rows: list[Exp24Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    references = _reference_rows(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=_csv_fieldnames(),
            lineterminator="\n",
        )
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


def _aggregate_by(rows: Sequence[Exp24Row], attr: str) -> list[dict[str, Any]]:
    compared = [row for row in rows if row.status == "compared"]
    buckets: dict[str, list[Exp24Row]] = {}
    for row in compared:
        buckets.setdefault(str(getattr(row, attr)), []).append(row)
    out: list[dict[str, Any]] = []
    for key, bucket in sorted(buckets.items()):
        out.append(
            {
                attr: key,
                "n": len(bucket),
                "median_global_mean_delta": _median(
                    row.global_mean_delta for row in bucket
                ),
                "median_support_mean_delta": _median(
                    row.support_mean_delta for row in bucket
                ),
                "median_morphology_gap_score": _median(
                    row.morphology_gap_score for row in bucket
                ),
                "median_log10_derivative_std_p50_ratio": _median(
                    row.log10_derivative_std_p50_ratio for row in bucket
                ),
                "median_mean_curve_corr": _median(row.mean_curve_corr for row in bucket),
                "median_guard_clip_fraction": _median(
                    row.guard_clip_fraction for row in bucket
                ),
                "failure_axis_distribution": _format_counter(
                    row.report_only_failure_axis for row in bucket
                ),
                "dominant_gap_distribution": _format_counter(
                    row.dominant_morphology_gap for row in bucket
                ),
            }
        )
    return out


def _paired_summary(rows: Sequence[Exp24Row]) -> list[dict[str, Any]]:
    references = _reference_rows(rows)
    out: list[dict[str, Any]] = []
    for profile in EXP24_AUDITED_PROFILES:
        bucket = [
            row
            for row in rows
            if row.status == "compared" and row.remediation_profile == profile
        ]
        for reference in EXP24_REFERENCE_PROFILES:
            deltas = [
                _row_reference_deltas(row, references)
                for row in bucket
                if _row_key(row) in references.get(reference, {})
            ]
            out.append(
                {
                    "profile": profile,
                    "reference": reference,
                    "n": len(deltas),
                    "median_global_mean_delta": _median(
                        entry[_delta_col(reference, "global_mean_delta")]
                        for entry in deltas
                    ),
                    "median_support_mean_delta": _median(
                        entry[_delta_col(reference, "support_mean_delta")]
                        for entry in deltas
                    ),
                    "median_morphology_gap_score": _median(
                        entry[_delta_col(reference, "morphology_gap_score")]
                        for entry in deltas
                    ),
                    "median_log10_derivative_std_p50_ratio": _median(
                        entry[
                            _delta_col(
                                reference,
                                "log10_derivative_std_p50_ratio",
                            )
                        ]
                        for entry in deltas
                    ),
                    "median_mean_curve_corr": _median(
                        entry[_delta_col(reference, "mean_curve_corr")]
                        for entry in deltas
                    ),
                }
            )
    return out


def _parse_csv_list(raw: str) -> list[str]:
    return cast(list[str], _exp15._parse_csv_list(raw))


def _parse_seeds(raw: str) -> list[int]:
    return [int(token) for token in _parse_csv_list(raw)]


def _repo_root() -> Path:
    return cast(Path, _exp15._repo_root())


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
    rows: list[Exp24Row] = result["rows"]
    compared = [row for row in rows if row.status == "compared"]
    blocked = [row for row in rows if row.status == "blocked"]
    tokens = list(sentinel_tokens) if sentinel_tokens is not None else list(result["sentinel_tokens"])
    audited = list(profiles) if profiles is not None else list(result["audited_profiles"])
    command = (
        "PYTHONPATH=bench/nirs_synthetic_pfn/src "
        "python bench/nirs_synthetic_pfn/experiments/"
        "exp24_diesel_render_stage_failure_map.py "
        f"--n-synthetic-samples {n_synthetic_samples} "
        f"--max-real-samples {max_real_samples} "
        f"--max-sentinel-datasets {max_sentinel_datasets} "
        f"--sentinel-tokens {','.join(tokens)} "
        f"--seeds {','.join(str(int(s)) for s in seeds)} "
        f"--profiles {','.join(audited)} "
        f"--report {report_path} "
        f"--csv {csv_path}"
    )
    lines = [
        "# P2-01 DIESEL Render-Stage Failure Map",
        "",
        "## Scope and Non-Gate Disclaimer",
        "",
        "- Bench-only, read-only, mechanistic diagnostic benchmark in `uncalibrated_raw`.",
        "- Compared profiles: R3d, R9e, R9j, R9l, R9m, R4b, and R4c on identical DIESEL rows.",
        "- R3d remains the accepted DIESEL baseline; R9m remains Palier 1 NO-GO evidence only.",
        "- No R9n, no new generator mechanism, no profile promotion, no gate, and no `nirs4all/` integration.",
        "- No calibration, real-stat capture, PCA/covariance/noise capture, ML/DL, labels, targets, splits, downstream feedback, threshold mutation, or metric mutation.",
        "- `report_only_failure_axis` and `report_only_failure_score` are diagnostic annotations only; they do not alter existing metrics or thresholds.",
        f"- Decision: `{EXP24_DECISION}`.",
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
        "## Stage-Family Map",
        "",
        "| stage family | n | median global | median support | median morphology gap | median derivative | median mean_curve_corr | failure axes | dominant gaps |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    stage_summary = _aggregate_by(rows, "render_stage_family")
    if not stage_summary:
        lines.append("| (no compared rows) | 0 | NA | NA | NA | NA | NA | NA | NA |")
    else:
        for entry in stage_summary:
            lines.append(
                f"| `{entry['render_stage_family']}` | {entry['n']} | "
                f"{_fmt(entry['median_global_mean_delta'])} | "
                f"{_fmt(entry['median_support_mean_delta'])} | "
                f"{_fmt(entry['median_morphology_gap_score'])} | "
                f"{_fmt(entry['median_log10_derivative_std_p50_ratio'])} | "
                f"{_fmt(entry['median_mean_curve_corr'])} | "
                f"{entry['failure_axis_distribution']} | "
                f"{entry['dominant_gap_distribution']} |"
            )
    lines.extend(
        [
            "",
            "## Profile Map",
            "",
            "| profile | stage family | mechanism family | n | median global | median support | median morphology gap | median derivative | median mean_curve_corr | failure axes |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    profile_summary = _aggregate_by(rows, "remediation_profile")
    if not profile_summary:
        lines.append("| (no compared rows) | NA | NA | 0 | NA | NA | NA | NA | NA | NA |")
    else:
        for entry in profile_summary:
            profile = str(entry["remediation_profile"])
            stage = PROFILE_STAGE_MAP[profile]["render_stage_family"]
            mechanism = PROFILE_STAGE_MAP[profile]["mechanism_family"]
            lines.append(
                f"| `{profile}` | `{stage}` | `{mechanism}` | {entry['n']} | "
                f"{_fmt(entry['median_global_mean_delta'])} | "
                f"{_fmt(entry['median_support_mean_delta'])} | "
                f"{_fmt(entry['median_morphology_gap_score'])} | "
                f"{_fmt(entry['median_log10_derivative_std_p50_ratio'])} | "
                f"{_fmt(entry['median_mean_curve_corr'])} | "
                f"{entry['failure_axis_distribution']} |"
            )
    lines.extend(
        [
            "",
            "## Paired Deltas",
            "",
            "Deltas are profile minus same `(seed, dataset)` reference. Negative deltas on mean/gap columns are lower than the reference; positive derivative deltas mean more first-derivative energy.",
            "",
            "| profile | reference | n | median global | median support | median morphology gap | median derivative | median mean_curve_corr |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for entry in _paired_summary(rows):
        lines.append(
            f"| `{entry['profile']}` | `{entry['reference']}` | {entry['n']} | "
            f"{_fmt(entry['median_global_mean_delta'])} | "
            f"{_fmt(entry['median_support_mean_delta'])} | "
            f"{_fmt(entry['median_morphology_gap_score'])} | "
            f"{_fmt(entry['median_log10_derivative_std_p50_ratio'])} | "
            f"{_fmt(entry['median_mean_curve_corr'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-Row Failure Map",
            "",
            "| dataset | seed | profile | stage family | global | support | morphology gap | derivative | mean_curve_corr | failure axis | dominant gap | guard clip | status |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row.dataset}` | {row.seed} | `{row.remediation_profile}` | "
            f"`{row.render_stage_family}` | {_fmt(row.global_mean_delta)} | "
            f"{_fmt(row.support_mean_delta)} | {_fmt(row.morphology_gap_score)} | "
            f"{_fmt(row.log10_derivative_std_p50_ratio)} | "
            f"{_fmt(row.mean_curve_corr)} | `{row.report_only_failure_axis}` | "
            f"`{row.dominant_morphology_gap}` | {_fmt(row.guard_clip_fraction)} | "
            f"`{row.status}` |"
        )
    lines.extend(
        [
            "",
            "## Initial Interpretation",
            "",
            "This audit is a map, not a remediation. It preserves the existing profile behavior and shows whether failures cluster in the R3d baseline pipeline, post-clip support attenuation, pre-baseline residual damping, combined damping/attenuation, or the R4 support-basis bundle.",
            "",
            "If all compared rows remain level/support dominated, P2.2 should target a predeclared row-level pathlength/reference branch rather than another support-only retune. If derivative or correlation failures dominate a stage family, the next hypothesis should be constrained to that rendering stage.",
            "",
            "## Raw Summary JSON",
            "",
            "```json",
            json.dumps(
                {
                    "status": result["status"],
                    "comparison_space": COMPARISON_SPACE,
                    "seeds": [int(s) for s in seeds],
                    "audited_profiles": audited,
                    "reference_profiles": list(EXP24_REFERENCE_PROFILES),
                    "compared_row_count": len(compared),
                    "blocked_row_count": len(blocked),
                    "decision": EXP24_DECISION,
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


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
        default=list(EXP24_AUDITED_PROFILES),
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
    rows: list[Exp24Row] = result["rows"]
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
