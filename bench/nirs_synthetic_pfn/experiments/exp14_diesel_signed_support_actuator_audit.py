"""R9e0 DIESEL signed support actuator diagnostic audit (bench-only, probe-only).

Diagnostic-only repeated-seed audit that compares synthetic DIESEL spectra
against real DIESEL sentinel cohorts using the ``uncalibrated_raw`` morphology
metrics from ``exp09_sentinel_morphology_audit`` plus the fixed-support
mean-shift decomposition from ``exp10_diesel_mean_shift_localization``. The
audit is diagnostic-only, non-gate, and never:

- fits any real-data calibration, marginal mapping, or covariance capture;
- runs PCA, adversarial AUC, or any ML/DL classifier;
- consumes labels, splits, targets, or downstream feedback;
- modifies B2/B3/B4/B5 thresholds or metric definitions;
- promotes any DIESEL profile or probe over the R3d baseline;
- registers any new builder profile (R9e0 is a post-hoc probe family only).

R9e0 is **probe-only**: there is no R9e0 builder profile, no
``builder_adapter.py`` change, and no nirs4all integration. The probes are
applied AFTER an R3d base render, AFTER alignment to the real wavelength grid,
and ONLY on the 750-1550 nm DIESEL real basis support. Off-support cells are
byte-identical to the R3d aligned synthetic by construction.

Audited builder profiles (DIESEL-only routes; non-DIESEL rows fall back
byte-identical to R3d via the existing exp09 routing):

- ``r3d_diesel_matrix_v1``
- ``r4a_diesel_basis_v1``
- ``r4b_diesel_derivative_restore_v1``
- ``r4c_diesel_balanced_derivative_v1``
- ``r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1``
- ``r9b_diesel_support_intercept_v1``
- ``r9c_diesel_selective_ch_bandwidth_damping_v1``
- ``r9d_diesel_energy_normalized_support_redistribution_v1``

R9e0 ephemeral support-only probes (post-hoc on the R3d aligned synthetic;
never registered as builder profiles):

1. ``r9e0_negative_blank_intercept_0p0010`` -- support signed actuator
   ``-0.0010``: ``X = max(X - 0.0010, 0)``.
2. ``r9e0_negative_blank_intercept_0p0020`` -- support signed actuator
   ``-0.0020``: ``X = max(X - 0.0020, 0)``.
3. ``r9e0_multiplicative_attenuation_0p985`` -- support ``X = 0.985 * X``.
4. ``r9e0_multiplicative_attenuation_0p970`` -- support ``X = 0.970 * X``.
5. ``r9e0_negative_intercept_0p0010_plus_r9d_shape_0p035`` -- support first
   apply signed actuator ``-0.0010`` with non-negative guard, then apply the same fixed
   mean-neutral CH shape family used by R9d (Gaussian CH overtone bands at
   1150/1210/1390/1460 nm with per-band widths 40/40/44/48 nm; mean-subtracted
   and max-abs-normalized on the support; clipped to [-1, 1]) at strength
   0.035, then multiplicatively renormalize the per-row support so the
   post-redistribution support mean equals the post-intercept support mean.

R9e0 audit constants are PRE-DECLARED MECHANISTIC CONSTANTS pulled from a
generic optical source: ``predeclared_generic_blank_reference_pathlength_and_
liquid_hydrocarbon_actuator_prior``. They are NOT computed from real spectra,
not from any R9b/R9c/R9d residual delta, not from labels/targets/splits, and
not tuned to close the morphology gap. R9e0 is GO/NO-GO evidence only; this
audit does not select a winning probe and does not authorize promotion of
any probe over R3d.
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
from nirsyntheticpfn.evaluation.realism import (
    RealDataset,
    align_to_real_grid,
    discover_local_real_datasets,
    is_index_fallback_grid,
    load_real_spectra,
    sanitize_finite_spectra,
)


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


# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------

R9E0_AUDITED_PROFILES: tuple[str, ...] = (
    "r3d_diesel_matrix_v1",
    "r4a_diesel_basis_v1",
    "r4b_diesel_derivative_restore_v1",
    "r4c_diesel_balanced_derivative_v1",
    "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1",
    "r9b_diesel_support_intercept_v1",
    "r9c_diesel_selective_ch_bandwidth_damping_v1",
    "r9d_diesel_energy_normalized_support_redistribution_v1",
)

R9E0_BASE_PROFILE: str = "r3d_diesel_matrix_v1"

R9E0_PAIRED_REFERENCE_PROFILES: tuple[str, ...] = (
    "r3d_diesel_matrix_v1",
    "r4c_diesel_balanced_derivative_v1",
)

R9E0_CONSTANTS_SOURCE: str = (
    "predeclared_generic_blank_reference_pathlength_and_"
    "liquid_hydrocarbon_actuator_prior"
)

# R9d CH overtone shape constants (PRE-DECLARED; reused unchanged from the
# R9d shape family for diagnostic continuity in probe 5).
R9D_SHAPE_CENTERS_NM: tuple[float, ...] = (1150.0, 1210.0, 1390.0, 1460.0)
R9D_SHAPE_WIDTHS_NM: tuple[float, ...] = (40.0, 40.0, 44.0, 48.0)


@dataclass(frozen=True)
class R9e0ProbeSpec:
    """Pre-declared post-hoc support-only probe definition.

    The probe is applied after the R3d base render and after alignment to the
    real wavelength grid, on the 750-1550 nm support only. Off-support cells
    are byte-identical to the R3d aligned synthetic.
    """

    name: str
    intercept: float = 0.0
    multiplicative: float = 1.0
    apply_r9d_shape: bool = False
    r9d_shape_strength: float = 0.0


R9E0_PROBES: tuple[R9e0ProbeSpec, ...] = (
    R9e0ProbeSpec(name="r9e0_negative_blank_intercept_0p0010", intercept=0.0010),
    R9e0ProbeSpec(name="r9e0_negative_blank_intercept_0p0020", intercept=0.0020),
    R9e0ProbeSpec(name="r9e0_multiplicative_attenuation_0p985", multiplicative=0.985),
    R9e0ProbeSpec(name="r9e0_multiplicative_attenuation_0p970", multiplicative=0.970),
    R9e0ProbeSpec(
        name="r9e0_negative_intercept_0p0010_plus_r9d_shape_0p035",
        intercept=0.0010,
        apply_r9d_shape=True,
        r9d_shape_strength=0.035,
    ),
)

R9E0_PROBE_NAMES: tuple[str, ...] = tuple(p.name for p in R9E0_PROBES)
R9E0_PROBES_BY_NAME: dict[str, R9e0ProbeSpec] = {p.name: p for p in R9E0_PROBES}

DEFAULT_SEEDS: tuple[int, ...] = (20260501, 20260502, 20260503)
DEFAULT_N_SYNTHETIC_SAMPLES: int = 64
DEFAULT_MAX_REAL_SAMPLES: int = 64
DEFAULT_MAX_SENTINEL_DATASETS: int = 8
DEFAULT_SENTINEL_TOKENS: tuple[str, ...] = ("DIESEL",)

DEFAULT_REPORT = Path(
    "bench/nirs_synthetic_pfn/reports/"
    "r9e0_diesel_signed_support_actuator_audit.md"
)
DEFAULT_CSV = Path(
    "bench/nirs_synthetic_pfn/reports/"
    "r9e0_diesel_signed_support_actuator_audit.csv"
)
R9E0_AUDIT_SCOPE: str = "bench_only_r9e0_diesel_signed_support_actuator_audit"
COMPARISON_SPACE: str = "uncalibrated_raw"

SUPPORT_LOW_NM: float = _exp10.SUPPORT_LOW_NM
SUPPORT_HIGH_NM: float = _exp10.SUPPORT_HIGH_NM


# ---------------------------------------------------------------------------
# Probe application.
# ---------------------------------------------------------------------------


def _build_r9d_support_shape(
    wavelengths: np.ndarray, support_mask: np.ndarray
) -> np.ndarray:
    """Build the fixed mean-neutral max-abs-normalized R9d CH shape on support.

    Returns an array shaped like ``wavelengths`` whose values are zero outside
    the support mask, mean-zero on the support, max-abs equal to 1 on the
    support (when the support is non-empty and the shape is non-constant),
    and clipped to [-1, 1]. Uses pre-declared centers/widths only.
    """
    wl = np.asarray(wavelengths, dtype=float).ravel()
    mask = np.asarray(support_mask, dtype=bool).ravel()
    raw = np.zeros_like(wl)
    for center, width in zip(
        R9D_SHAPE_CENTERS_NM, R9D_SHAPE_WIDTHS_NM, strict=True
    ):
        raw = raw + np.exp(-0.5 * ((wl - center) / width) ** 2)
    out = np.zeros_like(wl)
    if not mask.any():
        return out
    support_vals = raw[mask]
    support_vals = support_vals - float(support_vals.mean())
    max_abs = float(np.max(np.abs(support_vals)))
    if max_abs > 0.0:
        support_vals = support_vals / max_abs
    support_vals = np.clip(support_vals, -1.0, 1.0)
    out[mask] = support_vals
    return out


def apply_probe(
    base_X: np.ndarray,
    wavelengths: np.ndarray,
    *,
    spec: R9e0ProbeSpec,
    support_low_nm: float = SUPPORT_LOW_NM,
    support_high_nm: float = SUPPORT_HIGH_NM,
) -> tuple[np.ndarray, float]:
    """Apply an R9e0 post-hoc probe to a base synthetic on the support only.

    Returns the probed spectra and the guard-clip fraction (count of cells
    forced to zero by the non-negative guard divided by the count of cells
    that were eligible for clipping; ``0.0`` when the probe applies no
    intercept).
    """
    base = np.asarray(base_X)
    wl = np.asarray(wavelengths, dtype=float).ravel()
    if base.ndim != 2:
        raise ValueError("expected 2D synthetic spectra")
    if base.shape[1] != wl.size:
        raise ValueError("wavelengths must match the spectra column count")
    support_mask = (wl >= support_low_nm) & (wl <= support_high_nm)
    probed = base.copy()
    if not support_mask.any():
        return probed, 0.0

    support_block = probed[:, support_mask].astype(float, copy=True)
    clip_count = 0
    eligible_count = 0

    if spec.intercept != 0.0:
        eligible_count = int(support_block.size)
        candidate = support_block - float(spec.intercept)
        clip_count = int(np.sum(candidate < 0.0))
        support_block = np.maximum(candidate, 0.0)

    if spec.multiplicative != 1.0:
        support_block = float(spec.multiplicative) * support_block

    if spec.apply_r9d_shape:
        pre_shape_mean = support_block.mean(axis=1, keepdims=True)
        shape_full = _build_r9d_support_shape(wl, support_mask)
        shape_support = shape_full[support_mask]
        factor = np.exp(float(spec.r9d_shape_strength) * shape_support)
        support_block = support_block * factor[None, :]
        post_shape_mean = support_block.mean(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(
                post_shape_mean > 0.0,
                pre_shape_mean / post_shape_mean,
                np.ones_like(post_shape_mean),
            )
        support_block = support_block * scale

    probed[:, support_mask] = support_block
    guard_clip_fraction = (
        float(clip_count) / float(eligible_count) if eligible_count > 0 else 0.0
    )
    return probed, guard_clip_fraction


# ---------------------------------------------------------------------------
# Row dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class R9e0Row:
    """One R9e0 audit row for a (seed, profile-or-probe, dataset) triple."""

    status: str  # "compared" | "blocked"
    seed: int
    remediation_profile: str  # builder profile name OR probe name
    profile_kind: str  # "builder" | "probe"
    profile_registered: bool
    probe_only: bool
    base_profile: str | None  # for probes: "r3d_diesel_matrix_v1"; else None
    effective_remediation_profile: str | None
    source: str
    task: str
    dataset: str
    synthetic_preset: str
    effective_matrix_route: str | None
    comparison_space: str
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
    return {
        "audit_calibration": False,
        "audit_real_stat_capture": False,
        "audit_uses_pca": False,
        "audit_captures_noise": False,
        "audit_uses_ml": False,
        "audit_uses_dl": False,
        "audit_label_inputs_used": False,
        "audit_target_inputs_used": False,
        "audit_split_inputs_used": False,
        "audit_thresholds_modified": False,
        "audit_metrics_modified": False,
        "audit_source_oracle_used": False,
        "audit_scope": R9E0_AUDIT_SCOPE,
    }


def _empty_decomposition() -> dict[str, Any]:
    return {
        "support_low_nm": float(SUPPORT_LOW_NM),
        "support_high_nm": float(SUPPORT_HIGH_NM),
        "support_count": 0,
        "off_support_count": 0,
        "support_weight": 0.0,
        "off_support_weight": 0.0,
        "support_mean_delta": None,
        "off_support_mean_delta": None,
        "support_weighted_delta": None,
        "off_support_weighted_delta": None,
        "global_mean_delta": None,
        "decomposition_residual": None,
    }


def _empty_morphology_subset() -> dict[str, Any]:
    return {
        "real_global_mean": None,
        "synthetic_global_mean": None,
        "real_global_std": None,
        "synthetic_global_std": None,
        "log10_global_std_ratio": None,
        "log10_amplitude_p50_ratio": None,
        "log10_derivative_std_p50_ratio": None,
        "mean_curve_corr": None,
        "morphology_gap_score": None,
        "dominant_morphology_gap": "none",
    }


def _morphology_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "real_global_mean": metrics["real_global_mean"],
        "synthetic_global_mean": metrics["synthetic_global_mean"],
        "real_global_std": metrics["real_global_std"],
        "synthetic_global_std": metrics["synthetic_global_std"],
        "log10_global_std_ratio": metrics["log10_global_std_ratio"],
        "log10_amplitude_p50_ratio": metrics["log10_amplitude_p50_ratio"],
        "log10_derivative_std_p50_ratio": metrics["log10_derivative_std_p50_ratio"],
        "mean_curve_corr": metrics["mean_curve_corr"],
        "morphology_gap_score": metrics["morphology_gap_score"],
        "dominant_morphology_gap": metrics["dominant_morphology_gap"],
    }


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def is_probe(name: str) -> bool:
    """Return True iff ``name`` is one of the pre-declared R9e0 probe names."""
    return name in R9E0_PROBES_BY_NAME


def _validate_targets(profiles: Sequence[str], probes: Sequence[str]) -> None:
    if not profiles and not probes:
        raise ValueError(
            "at least one builder profile or one probe must be provided"
        )
    invalid_profiles = [p for p in profiles if p not in R9E0_AUDITED_PROFILES]
    if invalid_profiles:
        raise ValueError(
            f"unknown R9e0 builder profiles {invalid_profiles!r}; "
            f"valid profiles are {list(R9E0_AUDITED_PROFILES)}"
        )
    invalid_probes = [p for p in probes if p not in R9E0_PROBE_NAMES]
    if invalid_probes:
        raise ValueError(
            f"unknown R9e0 probes {invalid_probes!r}; "
            f"valid probes are {list(R9E0_PROBE_NAMES)}"
        )


def _blocked_row(
    *,
    seed: int,
    target_name: str,
    profile_kind: str,
    profile_registered: bool,
    probe_only: bool,
    base_profile: str | None,
    effective_profile: str | None,
    dataset: RealDataset,
    preset: str,
    blocked_reason: str,
    effective_matrix_route: str | None = None,
) -> R9e0Row:
    return R9e0Row(
        status="blocked",
        seed=int(seed),
        remediation_profile=target_name,
        profile_kind=profile_kind,
        profile_registered=profile_registered,
        probe_only=probe_only,
        base_profile=base_profile,
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
        guard_clip_fraction=None,
        **_empty_decomposition(),
        **_empty_morphology_subset(),
        **_audit_fields(),
        blocked_reason=blocked_reason,
    )


def _effective_matrix_route(
    *, metadata: dict[str, Any] | None
) -> str | None:
    audit = (metadata or {}).get("r2c_mechanistic_remediation") or {}
    transform_params = audit.get("transform_params") or {}
    return cast(
        "str | None",
        _exp09._effective_matrix_route_from_metadata(
            audit=audit, transform_params=transform_params
        ),
    )


def _compared_row(
    *,
    seed: int,
    target_name: str,
    profile_kind: str,
    profile_registered: bool,
    probe_only: bool,
    base_profile: str | None,
    effective_profile: str | None,
    dataset: RealDataset,
    preset: str,
    real_aligned: np.ndarray,
    syn_aligned: np.ndarray,
    aligned_wl: np.ndarray,
    effective_matrix_route: str | None,
    guard_clip_fraction: float | None,
    support_low_nm: float,
    support_high_nm: float,
) -> R9e0Row:
    metrics = _exp09.compute_morphology_metrics(
        real_aligned, syn_aligned, aligned_wl
    )
    decomposition = _exp10.compute_support_decomposition(
        real_aligned,
        syn_aligned,
        aligned_wl,
        support_low_nm=support_low_nm,
        support_high_nm=support_high_nm,
    )
    return R9e0Row(
        status="compared",
        seed=int(seed),
        remediation_profile=target_name,
        profile_kind=profile_kind,
        profile_registered=profile_registered,
        probe_only=probe_only,
        base_profile=base_profile,
        effective_remediation_profile=effective_profile,
        source=dataset.source,
        task=dataset.task,
        dataset=f"{dataset.database_name}/{dataset.dataset}",
        synthetic_preset=preset,
        effective_matrix_route=effective_matrix_route,
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
        support_weighted_delta=decomposition["support_weighted_delta"],
        off_support_weighted_delta=decomposition["off_support_weighted_delta"],
        global_mean_delta=decomposition["global_mean_delta"],
        decomposition_residual=decomposition["decomposition_residual"],
        guard_clip_fraction=guard_clip_fraction,
        **_morphology_subset(metrics),
        **_audit_fields(),
        blocked_reason="",
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
    probes: Sequence[str] | None = None,
    support_low_nm: float = SUPPORT_LOW_NM,
    support_high_nm: float = SUPPORT_HIGH_NM,
) -> dict[str, Any]:
    """Run the R9e0 DIESEL signed support actuator diagnostic audit."""
    seeds = tuple(int(s) for s in seeds)
    if not seeds:
        raise ValueError("at least one seed must be provided")

    audited_profiles = (
        tuple(R9E0_AUDITED_PROFILES) if profiles is None else tuple(profiles)
    )
    audited_probes = (
        tuple(R9E0_PROBE_NAMES) if probes is None else tuple(probes)
    )
    _validate_targets(audited_profiles, audited_probes)

    tokens = (
        tuple(DEFAULT_SENTINEL_TOKENS)
        if sentinel_tokens is None
        else tuple(sentinel_tokens)
    )
    real_datasets, _ = discover_local_real_datasets(root)
    sentinel_candidates = _exp09._select_sentinel_datasets(real_datasets, tokens)
    if max_sentinel_datasets <= 0:
        selected = list(sentinel_candidates)
    else:
        selected = sentinel_candidates[:max_sentinel_datasets]

    rows: list[R9e0Row] = []
    base_required_for_probes = bool(audited_probes) and (
        R9E0_BASE_PROFILE not in audited_profiles
    )

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
            "audited_probes": list(audited_probes),
            "support_low_nm": float(support_low_nm),
            "support_high_nm": float(support_high_nm),
        }

    for dataset in selected:
        preset = _exp09.select_synthetic_preset_for_dataset(dataset)
        try:
            real_X_raw, real_wl_raw = load_real_spectra(dataset, root=root)
        except Exception as exc:  # noqa: BLE001 - bench-only diagnostic surfacing
            for seed in seeds:
                _emit_blocked_for_targets(
                    rows=rows,
                    seed=seed,
                    dataset=dataset,
                    preset=preset,
                    profiles=audited_profiles,
                    probes=audited_probes,
                    blocked_reason=f"{type(exc).__name__}: {exc}",
                )
            continue

        if is_index_fallback_grid(real_wl_raw):
            for seed in seeds:
                _emit_blocked_for_targets(
                    rows=rows,
                    seed=seed,
                    dataset=dataset,
                    preset=preset,
                    profiles=audited_profiles,
                    probes=audited_probes,
                    blocked_reason=(
                        "wavelength_grid_unknown: real wavelengths "
                        "could not be parsed"
                    ),
                )
            continue

        sanitized_real, sanitized_wl, _, real_blocked = sanitize_finite_spectra(
            real_X_raw, real_wl_raw, side="real"
        )
        if (
            real_blocked is not None
            or sanitized_real is None
            or sanitized_wl is None
        ):
            for seed in seeds:
                _emit_blocked_for_targets(
                    rows=rows,
                    seed=seed,
                    dataset=dataset,
                    preset=preset,
                    profiles=audited_profiles,
                    probes=audited_probes,
                    blocked_reason=f"non_finite_spectra: {real_blocked}",
                )
            continue

        for seed in seeds:
            real_X = _exp09._downsample_rows(
                sanitized_real,
                max_rows=max_real_samples,
                random_state=_exp09._stable_dataset_seed(
                    seed, dataset, "r9e0:real_downsample"
                ),
            )
            real_wl = sanitized_wl

            r3d_aligned: np.ndarray | None = None
            r3d_aligned_wl: np.ndarray | None = None
            r3d_effective_route: str | None = None

            for profile in audited_profiles:
                effective = _exp09._effective_remediation_profile_for_dataset(
                    dataset, profile
                )
                aligned = _build_aligned_synthetic(
                    dataset=dataset,
                    preset=preset,
                    n_samples=n_synthetic_samples,
                    seed=seed,
                    profile=profile,
                    real_X=real_X,
                    real_wl=real_wl,
                    max_real_samples=max_real_samples,
                )
                if isinstance(aligned, str):
                    rows.append(
                        _blocked_row(
                            seed=seed,
                            target_name=profile,
                            profile_kind="builder",
                            profile_registered=True,
                            probe_only=False,
                            base_profile=None,
                            effective_profile=effective,
                            dataset=dataset,
                            preset=preset,
                            blocked_reason=aligned,
                        )
                    )
                    continue

                real_aligned, syn_aligned, aligned_wl, route = aligned
                rows.append(
                    _compared_row(
                        seed=seed,
                        target_name=profile,
                        profile_kind="builder",
                        profile_registered=True,
                        probe_only=False,
                        base_profile=None,
                        effective_profile=effective,
                        dataset=dataset,
                        preset=preset,
                        real_aligned=real_aligned,
                        syn_aligned=syn_aligned,
                        aligned_wl=aligned_wl,
                        effective_matrix_route=route,
                        guard_clip_fraction=None,
                        support_low_nm=support_low_nm,
                        support_high_nm=support_high_nm,
                    )
                )
                if profile == R9E0_BASE_PROFILE:
                    r3d_aligned = syn_aligned
                    r3d_aligned_wl = aligned_wl
                    r3d_effective_route = route

            if audited_probes and r3d_aligned is None:
                aligned = _build_aligned_synthetic(
                    dataset=dataset,
                    preset=preset,
                    n_samples=n_synthetic_samples,
                    seed=seed,
                    profile=R9E0_BASE_PROFILE,
                    real_X=real_X,
                    real_wl=real_wl,
                    max_real_samples=max_real_samples,
                )
                if isinstance(aligned, str):
                    for probe_name in audited_probes:
                        rows.append(
                            _blocked_row(
                                seed=seed,
                                target_name=probe_name,
                                profile_kind="probe",
                                profile_registered=False,
                                probe_only=True,
                                base_profile=R9E0_BASE_PROFILE,
                                effective_profile=None,
                                dataset=dataset,
                                preset=preset,
                                blocked_reason=(
                                    f"r3d_base_unavailable: {aligned}"
                                ),
                            )
                        )
                    continue
                _, r3d_aligned, r3d_aligned_wl, r3d_effective_route = aligned

            if not audited_probes or r3d_aligned is None or r3d_aligned_wl is None:
                continue
            if base_required_for_probes:
                # When the user explicitly excluded R3d from audited_profiles,
                # we still need an aligned R3d base for the probes; we built
                # it above. real_X above was downsampled but real_aligned was
                # not retained in the loop scope, so realign.
                pass

            _emit_probe_rows(
                rows=rows,
                seed=seed,
                dataset=dataset,
                preset=preset,
                probes=audited_probes,
                real_X=real_X,
                real_wl=real_wl,
                r3d_aligned=r3d_aligned,
                r3d_aligned_wl=r3d_aligned_wl,
                effective_matrix_route=r3d_effective_route,
                support_low_nm=support_low_nm,
                support_high_nm=support_high_nm,
            )

    compared = [row for row in rows if row.status == "compared"]
    status = "done" if compared else "blocked_no_successful_comparisons"
    return {
        "status": status,
        "rows": rows,
        "real_runnable_count": len(real_datasets),
        "real_sentinel_candidate_count": len(sentinel_candidates),
        "real_selected_count": len(selected),
        "sentinel_tokens": list(tokens),
        "seeds": list(seeds),
        "audited_profiles": list(audited_profiles),
        "audited_probes": list(audited_probes),
        "support_low_nm": float(support_low_nm),
        "support_high_nm": float(support_high_nm),
    }


def _emit_blocked_for_targets(
    *,
    rows: list[R9e0Row],
    seed: int,
    dataset: RealDataset,
    preset: str,
    profiles: Sequence[str],
    probes: Sequence[str],
    blocked_reason: str,
) -> None:
    for profile in profiles:
        effective = _exp09._effective_remediation_profile_for_dataset(
            dataset, profile
        )
        rows.append(
            _blocked_row(
                seed=seed,
                target_name=profile,
                profile_kind="builder",
                profile_registered=True,
                probe_only=False,
                base_profile=None,
                effective_profile=effective,
                dataset=dataset,
                preset=preset,
                blocked_reason=blocked_reason,
            )
        )
    for probe_name in probes:
        rows.append(
            _blocked_row(
                seed=seed,
                target_name=probe_name,
                profile_kind="probe",
                profile_registered=False,
                probe_only=True,
                base_profile=R9E0_BASE_PROFILE,
                effective_profile=None,
                dataset=dataset,
                preset=preset,
                blocked_reason=blocked_reason,
            )
        )


def _build_aligned_synthetic(
    *,
    dataset: RealDataset,
    preset: str,
    n_samples: int,
    seed: int,
    profile: str,
    real_X: np.ndarray,
    real_wl: np.ndarray,
    max_real_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None] | str:
    """Return ``(real_aligned, syn_aligned, aligned_wl, route)`` or a reason str."""
    try:
        synthetic_run = _exp09._build_baseline_synthetic_run(
            dataset=dataset,
            preset=preset,
            n_samples=n_samples,
            seed=seed,
            remediation_profile=profile,
        )
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"

    synth_X = np.asarray(synthetic_run.X, dtype=float)
    synth_wl = np.asarray(synthetic_run.wavelengths, dtype=float)
    sanitized_syn, sanitized_syn_wl, _, syn_blocked = sanitize_finite_spectra(
        synth_X, synth_wl, side="synthetic"
    )
    route = _effective_matrix_route(metadata=synthetic_run.metadata)
    if (
        syn_blocked is not None
        or sanitized_syn is None
        or sanitized_syn_wl is None
    ):
        return f"non_finite_spectra_synthetic: {syn_blocked}"

    synth_downsampled = _exp09._downsample_rows(
        sanitized_syn,
        max_rows=max_real_samples,
        random_state=_exp09._stable_dataset_seed(
            seed, dataset, "r9e0:syn_downsample"
        ),
    )

    try:
        real_aligned, syn_aligned, aligned_wl = align_to_real_grid(
            real_X, real_wl, synth_downsampled, sanitized_syn_wl
        )
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return real_aligned, syn_aligned, aligned_wl, route


def _emit_probe_rows(
    *,
    rows: list[R9e0Row],
    seed: int,
    dataset: RealDataset,
    preset: str,
    probes: Sequence[str],
    real_X: np.ndarray,
    real_wl: np.ndarray,
    r3d_aligned: np.ndarray,
    r3d_aligned_wl: np.ndarray,
    effective_matrix_route: str | None,
    support_low_nm: float,
    support_high_nm: float,
) -> None:
    # Real rows must be aligned to the same grid the R3d synthetic was aligned
    # to. The exp09 align_to_real_grid contract returns the real wavelengths
    # as the aligned wavelengths (real_aligned has the same shape as real_X
    # restricted to that grid); we re-run alignment against the R3d synthetic
    # to obtain the matched real array on the aligned grid.
    real_aligned, _ = _align_real_to_grid(
        real_X=real_X,
        real_wl=real_wl,
        target_X=r3d_aligned,
        target_wl=r3d_aligned_wl,
    )

    for probe_name in probes:
        spec = R9E0_PROBES_BY_NAME[probe_name]
        try:
            probed_X, guard_clip_fraction = apply_probe(
                r3d_aligned,
                r3d_aligned_wl,
                spec=spec,
                support_low_nm=support_low_nm,
                support_high_nm=support_high_nm,
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                _blocked_row(
                    seed=seed,
                    target_name=probe_name,
                    profile_kind="probe",
                    profile_registered=False,
                    probe_only=True,
                    base_profile=R9E0_BASE_PROFILE,
                    effective_profile=None,
                    dataset=dataset,
                    preset=preset,
                    blocked_reason=f"probe_apply_error: {type(exc).__name__}: {exc}",
                    effective_matrix_route=effective_matrix_route,
                )
            )
            continue
        rows.append(
            _compared_row(
                seed=seed,
                target_name=probe_name,
                profile_kind="probe",
                profile_registered=False,
                probe_only=True,
                base_profile=R9E0_BASE_PROFILE,
                effective_profile=None,
                dataset=dataset,
                preset=preset,
                real_aligned=real_aligned,
                syn_aligned=probed_X,
                aligned_wl=r3d_aligned_wl,
                effective_matrix_route=effective_matrix_route,
                guard_clip_fraction=float(guard_clip_fraction),
                support_low_nm=support_low_nm,
                support_high_nm=support_high_nm,
            )
        )


def _align_real_to_grid(
    *,
    real_X: np.ndarray,
    real_wl: np.ndarray,
    target_X: np.ndarray,
    target_wl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-align ``real_X`` onto ``target_wl`` using the exp09 helper."""
    real_aligned, _, aligned_wl = align_to_real_grid(
        real_X, real_wl, target_X, target_wl
    )
    return real_aligned, aligned_wl


# ---------------------------------------------------------------------------
# CSV / Markdown output.
# ---------------------------------------------------------------------------


PAIRED_DELTA_ATTRS: tuple[str, ...] = (
    "global_mean_delta",
    "support_mean_delta",
    "support_weighted_delta",
    "off_support_weighted_delta",
    "morphology_gap_score",
    "mean_curve_corr",
)


def _paired_delta_columns() -> list[str]:
    cols: list[str] = []
    for ref in R9E0_PAIRED_REFERENCE_PROFILES:
        for attr in PAIRED_DELTA_ATTRS:
            cols.append(f"delta_probe_minus_{ref}__{attr}")
    return cols


def _csv_fieldnames() -> list[str]:
    return [field.name for field in fields(R9e0Row)] + _paired_delta_columns()


def _row_key(row: R9e0Row) -> tuple[int, str]:
    return (int(row.seed), str(row.dataset))


def _paired_deltas_for_probes(
    rows: Sequence[R9e0Row],
) -> list[dict[str, Any]]:
    """Per-(seed, dataset, probe) paired deltas of probe minus reference profile.

    For each compared probe row, look up the matching compared rows for
    ``r3d_diesel_matrix_v1`` and ``r4c_diesel_balanced_derivative_v1`` on the
    same (seed, dataset) and emit arithmetic differences on the paired
    metrics. Missing/blocked references leave the matching delta cell at
    ``None``.
    """
    compared = [row for row in rows if row.status == "compared"]
    by_profile: dict[str, dict[tuple[int, str], R9e0Row]] = {}
    for row in compared:
        by_profile.setdefault(row.remediation_profile, {})[_row_key(row)] = row

    entries: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for row in compared:
        if row.profile_kind != "probe":
            continue
        triple = (int(row.seed), str(row.dataset), row.remediation_profile)
        if triple in seen:
            continue
        seen.add(triple)
        entry: dict[str, Any] = {
            "seed": int(row.seed),
            "dataset": str(row.dataset),
            "probe": row.remediation_profile,
            "guard_clip_fraction": row.guard_clip_fraction,
            "probe_global_mean_delta": row.global_mean_delta,
            "probe_support_mean_delta": row.support_mean_delta,
            "probe_off_support_weighted_delta": row.off_support_weighted_delta,
            "probe_morphology_gap_score": row.morphology_gap_score,
            "probe_mean_curve_corr": row.mean_curve_corr,
        }
        for ref in R9E0_PAIRED_REFERENCE_PROFILES:
            ref_row = by_profile.get(ref, {}).get(_row_key(row))
            for attr in PAIRED_DELTA_ATTRS:
                col = f"delta_probe_minus_{ref}__{attr}"
                if ref_row is None:
                    entry[col] = None
                    continue
                ref_value = getattr(ref_row, attr)
                probe_value = getattr(row, attr)
                if ref_value is None or probe_value is None:
                    entry[col] = None
                else:
                    entry[col] = float(probe_value) - float(ref_value)
        entries.append(entry)
    return entries


def write_csv(rows: list[R9e0Row], path: Path) -> None:
    """Write rows to ``path``; always emits a stable header even when empty.

    The CSV header includes the R9e0Row fields plus per-reference paired delta
    columns ``delta_probe_minus_<ref>__<attr>`` populated on probe rows that
    reached ``compared`` status (matched to the reference compared row by
    seed + dataset). Builder profile rows leave the matching delta cells
    blank.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames()
    paired_cols = _paired_delta_columns()
    paired_by_key: dict[tuple[int, str, str], dict[str, Any]] = {
        (int(entry["seed"]), str(entry["dataset"]), str(entry["probe"])): entry
        for entry in _paired_deltas_for_probes(rows)
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = row.to_dict()
            paired_entry: dict[str, Any] | None = None
            if row.profile_kind == "probe" and row.status == "compared":
                paired_entry = paired_by_key.get(
                    (int(row.seed), str(row.dataset), row.remediation_profile)
                )
            for col in paired_cols:
                record[col] = (
                    paired_entry.get(col) if paired_entry is not None else None
                )
            writer.writerow(record)


def _fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def _median(values: Iterable[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None]
    if not finite:
        return None
    return float(np.median(np.asarray(finite, dtype=float)))


def _format_gap_distribution(rows: Sequence[R9e0Row]) -> str:
    counts = Counter(row.dominant_morphology_gap for row in rows)
    if not counts:
        return "n/a"
    return ", ".join(
        f"{label}={count}" for label, count in counts.most_common()
    )


def _aggregate_by_profile(rows: Sequence[R9e0Row]) -> list[dict[str, Any]]:
    """Aggregate ``compared`` rows per ``remediation_profile``."""
    compared = [row for row in rows if row.status == "compared"]
    order: list[str] = []
    grouped: dict[str, list[R9e0Row]] = {}
    for row in compared:
        key = row.remediation_profile
        if key not in grouped:
            order.append(key)
            grouped[key] = []
        grouped[key].append(row)

    summary: list[dict[str, Any]] = []
    for profile in order:
        bucket = grouped[profile]
        kind = bucket[0].profile_kind
        registered = bucket[0].profile_registered
        summary.append(
            {
                "profile": profile,
                "profile_kind": kind,
                "profile_registered": registered,
                "n": len(bucket),
                "median_global_mean_delta": _median(
                    r.global_mean_delta for r in bucket
                ),
                "median_support_mean_delta": _median(
                    r.support_mean_delta for r in bucket
                ),
                "median_off_support_weighted_delta": _median(
                    r.off_support_weighted_delta for r in bucket
                ),
                "median_support_weight": _median(
                    r.support_weight for r in bucket
                ),
                "median_off_support_weight": _median(
                    r.off_support_weight for r in bucket
                ),
                "median_morphology_gap_score": _median(
                    r.morphology_gap_score for r in bucket
                ),
                "median_mean_curve_corr": _median(
                    r.mean_curve_corr for r in bucket
                ),
                "median_log10_derivative_std_p50_ratio": _median(
                    r.log10_derivative_std_p50_ratio for r in bucket
                ),
                "median_guard_clip_fraction": _median(
                    r.guard_clip_fraction for r in bucket
                ),
                "dominant_gap_distribution": _format_gap_distribution(bucket),
            }
        )
    return summary


def _diagnostic_outcome_lines(
    profile_summary: Sequence[dict[str, Any]],
) -> list[str]:
    """Render GO/NO-GO evidence per probe relative to R3d on the median global
    mean delta and median morphology gap score. Diagnostic only; no probe is
    promoted and no probe replaces R3d."""
    by_profile = {str(entry["profile"]): entry for entry in profile_summary}
    r3d = by_profile.get(R9E0_BASE_PROFILE)
    if r3d is None:
        return [
            "- R3d baseline not present in compared rows; no GO/NO-GO "
            "comparison available (diagnostic-only).",
        ]

    r3d_global = r3d.get("median_global_mean_delta")
    r3d_gap = r3d.get("median_morphology_gap_score")
    out: list[str] = [
        f"- R3d baseline medians: global mean delta {_fmt(cast(float | None, r3d_global))}; "
        f"morphology gap score {_fmt(cast(float | None, r3d_gap))}.",
    ]

    for probe_name in R9E0_PROBE_NAMES:
        entry = by_profile.get(probe_name)
        if entry is None:
            out.append(
                f"- Probe `{probe_name}`: no compared rows; no GO/NO-GO "
                "evidence emitted (diagnostic-only)."
            )
            continue
        probe_global = entry.get("median_global_mean_delta")
        probe_gap = entry.get("median_morphology_gap_score")
        improves_global = (
            probe_global is not None
            and r3d_global is not None
            and abs(float(probe_global)) < abs(float(r3d_global))
        )
        improves_gap = (
            probe_gap is not None
            and r3d_gap is not None
            and float(probe_gap) < float(r3d_gap)
        )
        verdict = "GO-evidence" if (improves_global and improves_gap) else "NO-GO-evidence"
        out.append(
            f"- Probe `{probe_name}` ({verdict}, diagnostic-only): "
            f"median global mean delta {_fmt(cast(float | None, probe_global))} "
            f"vs R3d {_fmt(cast(float | None, r3d_global))}; "
            f"median morphology gap score {_fmt(cast(float | None, probe_gap))} "
            f"vs R3d {_fmt(cast(float | None, r3d_gap))}; "
            f"median guard clip fraction {_fmt(entry.get('median_guard_clip_fraction'))}."
        )
    out.append(
        "- R9e0 is diagnostic-only and probe-only; no probe is promoted, "
        "no probe replaces R3d, and no probe is registered as a builder "
        "profile."
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
    sentinel_tokens: Sequence[str] | None = None,
    profiles: Sequence[str] | None = None,
    probes: Sequence[str] | None = None,
    support_low_nm: float = SUPPORT_LOW_NM,
    support_high_nm: float = SUPPORT_HIGH_NM,
) -> str:
    rows: list[R9e0Row] = result["rows"]
    tokens = (
        list(sentinel_tokens)
        if sentinel_tokens is not None
        else list(result.get("sentinel_tokens", DEFAULT_SENTINEL_TOKENS))
    )
    audited_profiles = (
        list(profiles)
        if profiles is not None
        else list(result.get("audited_profiles", R9E0_AUDITED_PROFILES))
    )
    audited_probes = (
        list(probes)
        if probes is not None
        else list(result.get("audited_probes", R9E0_PROBE_NAMES))
    )
    seeds_list = [int(s) for s in seeds]
    compared = [row for row in rows if row.status == "compared"]
    blocked = [row for row in rows if row.status == "blocked"]

    command = (
        "PYTHONPATH=bench/nirs_synthetic_pfn/src "
        "python bench/nirs_synthetic_pfn/experiments/"
        "exp14_diesel_signed_support_actuator_audit.py "
        f"--n-synthetic-samples {n_synthetic_samples} "
        f"--max-real-samples {max_real_samples} "
        f"--max-sentinel-datasets {max_sentinel_datasets} "
        f"--sentinel-tokens {','.join(tokens)} "
        f"--seeds {','.join(str(s) for s in seeds_list)} "
        f"--profiles {','.join(audited_profiles)} "
        f"--probes {','.join(audited_probes)}"
    )

    lines: list[str] = [
        "# R9e0 DIESEL Signed Support Actuator Diagnostic Audit",
        "",
        "## Scope and Non-Gate Disclaimer",
        "",
        "- Bench-only, mechanistic, diagnostic-only repeated-seed audit. Comparison space is `uncalibrated_raw` only.",
        "- This audit does NOT establish any B2/B3/B4/B5 pass and does not modify any gate threshold or metric.",
        "- no calibration fitted, captured, or applied (no marginal, covariance, or quantile mapping).",
        "- no PCA/covariance capture from real data; no adversarial AUC; no ML/DL.",
        "- no labels, splits, targets, or downstream feedback are consulted.",
        "- DIESEL routes only; non-DIESEL rows fall back byte-identical to R3d via the existing exp09 routing.",
        "- R9e0 is **probe-only** and **diagnostic-only**: no probe is promoted, no probe replaces R3d, no probe is registered as a builder profile, and there is no nirs4all integration. R3d remains the accepted DIESEL baseline.",
        "- The R9e0 base / fallback for every probe is R3d; off-support cells of every probe are byte-identical to the R3d aligned synthetic by construction.",
        "- The R9e0 probe constants are PRE-DECLARED MECHANISTIC CONSTANTS pulled from a generic optical source: `predeclared_generic_blank_reference_pathlength_and_liquid_hydrocarbon_actuator_prior`. They are NOT computed from real spectra, NOT from any R9b/R9c/R9d residual delta, NOT from labels/targets/splits, and NOT tuned to close the morphology gap.",
        "- Decomposes `synthetic_mean - real_mean` over the fixed "
        f"`{support_low_nm:g}-{support_high_nm:g}` nm support; "
        "the identity `global_mean_delta = support_weighted_delta + off_support_weighted_delta` "
        "holds to floating-point tolerance (reused unchanged from exp10).",
        "- This audit emits GO/NO-GO evidence per probe relative to R3d only; it does not select a winning probe.",
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
        f"- Seeds: {', '.join(str(s) for s in seeds_list)}",
        f"- Synthetic samples per dataset: {n_synthetic_samples}",
        f"- Real samples per dataset cap: {max_real_samples}",
        f"- Sentinel dataset cap: {max_sentinel_datasets if max_sentinel_datasets > 0 else 'all token-matched sentinel rows'}",
        f"- Sentinel tokens: `{', '.join(tokens)}`",
        f"- Audited builder profiles: `{', '.join(audited_profiles)}`",
        f"- Audited probes (post-hoc, never registered): `{', '.join(audited_probes)}`",
        f"- Support window: `{support_low_nm:g}-{support_high_nm:g}` nm",
        "",
        "## Summary",
        "",
        f"- Status: `{result['status']}`",
        f"- Real runnable rows discovered: {result['real_runnable_count']}",
        f"- Real sentinel candidates after token filter: "
        f"{result.get('real_sentinel_candidate_count', 0)}",
        f"- Real rows selected: {result['real_selected_count']}",
        f"- Compared rows: {len(compared)}",
        f"- Blocked rows: {len(blocked)}",
        "",
        "## Per-Profile Synthesis (compared rows only)",
        "",
    ]

    profile_summary = _aggregate_by_profile(rows)
    if not profile_summary:
        lines.append(
            "No compared rows to aggregate; per-profile synthesis is empty."
        )
    else:
        lines.extend(
            [
                "Aggregated medians per `remediation_profile` over rows with "
                "`status == compared`. `kind` is `builder` for registered "
                "builder profiles and `probe` for R9e0 ephemeral probes "
                "(`probe_only=true`, `profile_registered=false`).",
                "",
                (
                    "| profile | kind | n | median global mean delta | median "
                    "support mean delta | median off-support weighted delta | "
                    "median support weight | median morphology gap score | "
                    "median mean curve corr | median log10 deriv std p50 "
                    "ratio | median guard clip fraction | dominant gap dist. |"
                ),
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for entry in profile_summary:
            lines.append(
                f"| `{entry['profile']}` | `{entry['profile_kind']}` | "
                f"{entry['n']} | "
                f"{_fmt(entry['median_global_mean_delta'])} | "
                f"{_fmt(entry['median_support_mean_delta'])} | "
                f"{_fmt(entry['median_off_support_weighted_delta'])} | "
                f"{_fmt(entry['median_support_weight'])} | "
                f"{_fmt(entry['median_morphology_gap_score'])} | "
                f"{_fmt(entry['median_mean_curve_corr'])} | "
                f"{_fmt(entry['median_log10_derivative_std_p50_ratio'])} | "
                f"{_fmt(entry['median_guard_clip_fraction'])} | "
                f"{entry['dominant_gap_distribution']} |"
            )

    lines.extend(["", "## Diagnostic Outcome (GO/NO-GO evidence)", ""])
    lines.extend(_diagnostic_outcome_lines(profile_summary))

    lines.extend(
        [
            "",
            "## Paired Deltas: probe minus reference profile (per (seed, dataset, probe))",
            "",
            "Per-(seed, dataset, probe) arithmetic differences of probe row "
            "metrics minus the matching reference row metric (R3d and R4c) on "
            "the same morphology / decomposition field. These are not new "
            "metrics; they exist only to make the probe vs R3d/R4c direction "
            "directly readable. Reference rows that did not reach `compared` "
            "status leave the matching delta cell as NA.",
            "",
            (
                "| dataset | seed | probe | guard clip fraction | probe global "
                "mean delta | probe support mean delta | probe vs r3d global | "
                "probe vs r3d morphology gap | probe vs r4c global | probe vs "
                "r4c morphology gap |"
            ),
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    paired = _paired_deltas_for_probes(rows)
    if not paired:
        lines.append(
            "| (no probe compared rows) | NA | NA | NA | NA | NA | NA | NA | "
            "NA | NA |"
        )
    else:
        for entry in paired:
            lines.append(
                f"| `{entry['dataset']}` | {entry['seed']} | "
                f"`{entry['probe']}` | "
                f"{_fmt(entry.get('guard_clip_fraction'))} | "
                f"{_fmt(entry['probe_global_mean_delta'])} | "
                f"{_fmt(entry['probe_support_mean_delta'])} | "
                f"{_fmt(entry['delta_probe_minus_r3d_diesel_matrix_v1__global_mean_delta'])} | "
                f"{_fmt(entry['delta_probe_minus_r3d_diesel_matrix_v1__morphology_gap_score'])} | "
                f"{_fmt(entry['delta_probe_minus_r4c_diesel_balanced_derivative_v1__global_mean_delta'])} | "
                f"{_fmt(entry['delta_probe_minus_r4c_diesel_balanced_derivative_v1__morphology_gap_score'])} |"
            )

    lines.extend(
        [
            "",
            "## Mean-Shift Decomposition (uncalibrated_raw)",
            "",
            "| dataset | seed | profile | kind | n real | n syn | global mean delta | support weighted delta | off-support weighted delta | support mean delta | guard clip fraction | support weight | log10 std ratio | mean curve corr | dominant gap | status |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row.dataset}` | {row.seed} | `{row.remediation_profile}` | "
            f"`{row.profile_kind}` | "
            f"{row.n_real_samples} | {row.n_synthetic_samples} | "
            f"{_fmt(row.global_mean_delta)} | "
            f"{_fmt(row.support_weighted_delta)} | "
            f"{_fmt(row.off_support_weighted_delta)} | "
            f"{_fmt(row.support_mean_delta)} | "
            f"{_fmt(row.guard_clip_fraction)} | "
            f"{_fmt(row.support_weight)} | "
            f"{_fmt(row.log10_global_std_ratio)} | "
            f"{_fmt(row.mean_curve_corr)} | "
            f"`{row.dominant_morphology_gap}` | "
            f"`{row.status}` |"
        )

    lines.extend(
        [
            "",
            "## R9e0 Probe Provenance",
            "",
            (
                "- R9e0 is a post-hoc probe family. Every probe is applied "
                "AFTER the R3d base render, AFTER alignment to the real "
                "wavelength grid, and ONLY on the 750-1550 nm support. "
                "Off-support cells are byte-identical to the R3d aligned "
                "synthetic by construction."
            ),
            (
                "- No R9e0 probe is registered as a builder remediation "
                "profile. No `builder_adapter.py` change is required. R9e0 "
                "is bench-only diagnostic-only; this audit does not "
                "authorize a promotion over R3d and does not authorize any "
                "nirs4all integration."
            ),
            (
                "- Pre-declared probe definitions:\n"
                "    1. `r9e0_negative_blank_intercept_0p0010` -- support "
                "signed actuator `-0.0010`: `X = max(X - 0.0010, 0)`.\n"
                "    2. `r9e0_negative_blank_intercept_0p0020` -- support "
                "signed actuator `-0.0020`: `X = max(X - 0.0020, 0)`.\n"
                "    3. `r9e0_multiplicative_attenuation_0p985` -- support "
                "`X = 0.985 * X`.\n"
                "    4. `r9e0_multiplicative_attenuation_0p970` -- support "
                "`X = 0.970 * X`.\n"
                "    5. `r9e0_negative_intercept_0p0010_plus_r9d_shape_0p035` "
                "-- support first applies signed actuator `-0.0010` with "
                "non-negative guard, "
                "then multiplicatively apply the R9d CH shape family at "
                "strength 0.035 (centers 1150/1210/1390/1460 nm; per-band "
                "widths 40/40/44/48 nm; mean-subtracted and "
                "max-abs-normalized on the support; clipped to [-1, 1]) and "
                "multiplicatively renormalize the per-row support so the "
                "post-redistribution support mean equals the post-intercept "
                "support mean."
            ),
            (
                "- The R9e0 probe constants are PRE-DECLARED MECHANISTIC "
                "CONSTANTS from a generic optical source: "
                f"`{R9E0_CONSTANTS_SOURCE}`. They are NOT computed from any "
                "R9b/R9c/R9d mean-shift residual, real spectra, marginal "
                "statistic, PCA loading, quantile, ML/DL output, label, "
                "target, split, AUC, morphology gap score, threshold, "
                "calibration, or downstream feedback."
            ),
            (
                "- `guard_clip_fraction` accounts for the per-cell count of "
                "support cells forced to zero by the non-negative guard "
                "after the negative-blank intercept, divided by the number "
                "of support cells eligible for clipping. It is `0.0` for "
                "probes that do not apply an intercept and `NA` for builder "
                "profile rows."
            ),
            "",
            "## Audit Flags (every row)",
            "",
            "- `probe_only=true` for probe rows; `probe_only=false` for builder profile rows.",
            "- `profile_registered=false` for probe rows; `profile_registered=true` for builder profile rows.",
            "- `calibration=false`",
            "- `real_stat_capture=false`",
            "- `uses_pca=false`",
            "- `captures_noise=false`",
            "- `uses_ml=false`",
            "- `uses_dl=false`",
            "- `label_inputs_used=false`",
            "- `target_inputs_used=false`",
            "- `split_inputs_used=false`",
            "- `thresholds_modified=false`",
            "- `metrics_modified=false`",
            "- `source_oracle_used=false`",
            f"- `constants_source={R9E0_CONSTANTS_SOURCE}`",
            f"- `audit_scope={R9E0_AUDIT_SCOPE}`",
            "",
            "## Decision",
            "",
            (
                "Mechanistic, diagnostic-only post-hoc probe audit; this "
                "lane is non-gate and does not promote any probe over R3d. "
                "There is no integration into nirs4all from this audit. The "
                "GO/NO-GO evidence per probe is computed against R3d only "
                "and is descriptive: a probe that improves both medians "
                "relative to R3d earns GO-evidence, but no probe is "
                "selected, registered, or promoted by this audit. R3d "
                "remains the accepted DIESEL baseline."
            ),
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
                        "real_sentinel_candidate_count", 0
                    ),
                    "real_selected_count": result["real_selected_count"],
                    "sentinel_tokens": tokens,
                    "seeds": seeds_list,
                    "audited_profiles": audited_profiles,
                    "audited_probes": audited_probes,
                    "support_low_nm": float(support_low_nm),
                    "support_high_nm": float(support_high_nm),
                    "compared_row_count": len(compared),
                    "blocked_row_count": len(blocked),
                    "constants_source": R9E0_CONSTANTS_SOURCE,
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_csv_list(raw: str) -> list[str]:
    items = [token.strip() for token in raw.split(",") if token.strip()]
    if not items:
        raise ValueError("at least one comma-separated value must be provided")
    return items


def _parse_seeds(raw: str) -> list[int]:
    return [int(token) for token in _parse_csv_list(raw)]


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "bench" / "nirs_synthetic_pfn").is_dir():
            return candidate
    raise RuntimeError(f"could not locate repo root from {here}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-synthetic-samples",
        type=int,
        default=DEFAULT_N_SYNTHETIC_SAMPLES,
    )
    parser.add_argument(
        "--max-real-samples",
        type=int,
        default=DEFAULT_MAX_REAL_SAMPLES,
    )
    parser.add_argument(
        "--max-sentinel-datasets",
        type=int,
        default=DEFAULT_MAX_SENTINEL_DATASETS,
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--sentinel-tokens",
        type=str,
        default=",".join(DEFAULT_SENTINEL_TOKENS),
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default=",".join(R9E0_AUDITED_PROFILES),
        help=(
            "Comma-separated builder profiles to audit. Must be a subset of "
            f"{list(R9E0_AUDITED_PROFILES)}."
        ),
    )
    parser.add_argument(
        "--probes",
        type=str,
        default=",".join(R9E0_PROBE_NAMES),
        help=(
            "Comma-separated R9e0 probes (post-hoc, never registered). "
            f"Must be a subset of {list(R9E0_PROBE_NAMES)}."
        ),
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--support-low-nm", type=float, default=SUPPORT_LOW_NM)
    parser.add_argument("--support-high-nm", type=float, default=SUPPORT_HIGH_NM)
    args = parser.parse_args()

    sentinel_tokens = _parse_csv_list(args.sentinel_tokens)
    profiles = _parse_csv_list(args.profiles)
    probes = _parse_csv_list(args.probes)
    seeds = _parse_seeds(args.seeds)
    root = _repo_root()
    result = run_audit(
        root=root,
        seeds=seeds,
        n_synthetic_samples=args.n_synthetic_samples,
        max_real_samples=args.max_real_samples,
        max_sentinel_datasets=args.max_sentinel_datasets,
        sentinel_tokens=sentinel_tokens,
        profiles=profiles,
        probes=probes,
        support_low_nm=args.support_low_nm,
        support_high_nm=args.support_high_nm,
    )
    write_csv(result["rows"], args.csv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render_markdown(
            result=result,
            report_path=args.report,
            csv_path=args.csv,
            n_synthetic_samples=args.n_synthetic_samples,
            max_real_samples=args.max_real_samples,
            max_sentinel_datasets=args.max_sentinel_datasets,
            seeds=seeds,
            sentinel_tokens=sentinel_tokens,
            profiles=profiles,
            probes=probes,
            support_low_nm=args.support_low_nm,
            support_high_nm=args.support_high_nm,
        ),
        encoding="utf-8",
    )
    print(args.report)
    print(args.csv)


__all__ = [
    "COMPARISON_SPACE",
    "DEFAULT_CSV",
    "DEFAULT_MAX_REAL_SAMPLES",
    "DEFAULT_MAX_SENTINEL_DATASETS",
    "DEFAULT_N_SYNTHETIC_SAMPLES",
    "DEFAULT_REPORT",
    "DEFAULT_SEEDS",
    "DEFAULT_SENTINEL_TOKENS",
    "PAIRED_DELTA_ATTRS",
    "R9D_SHAPE_CENTERS_NM",
    "R9D_SHAPE_WIDTHS_NM",
    "R9E0_AUDITED_PROFILES",
    "R9E0_AUDIT_SCOPE",
    "R9E0_BASE_PROFILE",
    "R9E0_CONSTANTS_SOURCE",
    "R9E0_PAIRED_REFERENCE_PROFILES",
    "R9E0_PROBES",
    "R9E0_PROBES_BY_NAME",
    "R9E0_PROBE_NAMES",
    "R9e0ProbeSpec",
    "R9e0Row",
    "SUPPORT_HIGH_NM",
    "SUPPORT_LOW_NM",
    "_aggregate_by_profile",
    "_paired_deltas_for_probes",
    "apply_probe",
    "is_probe",
    "main",
    "render_markdown",
    "run_audit",
    "write_csv",
]


if __name__ == "__main__":
    main()
