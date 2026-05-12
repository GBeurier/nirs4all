"""R9e DIESEL pathlength/reference attenuation diagnostic audit.

Bench-only diagnostic-only repeated-seed audit that compares synthetic DIESEL
spectra against real DIESEL sentinel cohorts in ``uncalibrated_raw``. R9e is a
registered builder diagnostic profile, but this audit is non-gate and never:

- fits calibration, marginal mapping, covariance, or quantile transforms;
- runs PCA, adversarial AUC, ML, or DL;
- consumes labels, targets, splits, or downstream feedback;
- modifies B2/B3/B4/B5 thresholds or metric definitions;
- promotes R9e over the accepted R3d baseline;
- authorizes any ``nirs4all/`` integration.

R9e inherits the R3d micro-path / baseline / CH-overtone / non-negative output
clip byte-for-byte, then applies a positive row-wise multiplicative attenuation
factor in [0.970, 0.985] only on the fixed 750-1550 nm support. Off-support
cells are byte-identical to R3d. There is no additional clip, offset, R9d
shape, support-mean renormalization, or readout transform.
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


R9E_AUDITED_PROFILES: tuple[str, ...] = (
    "r3d_diesel_matrix_v1",
    "r4b_diesel_derivative_restore_v1",
    "r4c_diesel_balanced_derivative_v1",
    "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1",
    "r9d_diesel_energy_normalized_support_redistribution_v1",
    "r9e_diesel_pathlength_reference_attenuation_v1",
)

R9E_FOCUS_PROFILE = "r9e_diesel_pathlength_reference_attenuation_v1"
R9E_BASE_PROFILE = "r3d_diesel_matrix_v1"
R9E_PAIRED_REFERENCE_PROFILES: tuple[str, ...] = (
    "r3d_diesel_matrix_v1",
    "r4c_diesel_balanced_derivative_v1",
    "r9d_diesel_energy_normalized_support_redistribution_v1",
)
R9E_CONSTANTS_SOURCE = (
    "predeclared_generic_blank_reference_pathlength_attenuation_prior"
)

DEFAULT_SEEDS: tuple[int, ...] = (20260501, 20260502, 20260503)
DEFAULT_N_SYNTHETIC_SAMPLES = 64
DEFAULT_MAX_REAL_SAMPLES = 64
DEFAULT_MAX_SENTINEL_DATASETS = 8
DEFAULT_SENTINEL_TOKENS: tuple[str, ...] = ("DIESEL",)

DEFAULT_REPORT = Path(
    "bench/nirs_synthetic_pfn/reports/"
    "r9e_diesel_pathlength_reference_attenuation_audit.md"
)
DEFAULT_CSV = Path(
    "bench/nirs_synthetic_pfn/reports/"
    "r9e_diesel_pathlength_reference_attenuation_audit.csv"
)
R9E_AUDIT_SCOPE = (
    "bench_only_r9e_diesel_pathlength_reference_attenuation_audit"
)
COMPARISON_SPACE = "uncalibrated_raw"
SUPPORT_LOW_NM: float = _exp10.SUPPORT_LOW_NM
SUPPORT_HIGH_NM: float = _exp10.SUPPORT_HIGH_NM
R9E_GUARD_CLIP_FRACTION = 0.0


@dataclass(frozen=True)
class R9eRow:
    """One R9e audit row for a (seed, profile, dataset) triple."""

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
        "audit_scope": R9E_AUDIT_SCOPE,
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


def _validate_profiles(profiles: Sequence[str]) -> tuple[str, ...]:
    if not profiles:
        raise ValueError("at least one remediation profile must be provided")
    invalid = [
        profile for profile in profiles if profile not in R9E_AUDITED_PROFILES
    ]
    if invalid:
        raise ValueError(
            f"unknown R9e profiles {invalid!r}; "
            f"valid profiles are {list(R9E_AUDITED_PROFILES)}"
        )
    return tuple(profiles)


def _effective_matrix_route(metadata: dict[str, Any] | None) -> str | None:
    audit = (metadata or {}).get("r2c_mechanistic_remediation") or {}
    transform_params = audit.get("transform_params") or {}
    route = _exp09._effective_matrix_route_from_metadata(
        audit=audit,
        transform_params=transform_params,
    )
    return cast(str | None, route)


def _guard_clip_fraction(
    profile: str,
    metadata: dict[str, Any] | None,
) -> float | None:
    if profile != R9E_FOCUS_PROFILE:
        return None
    audit = (metadata or {}).get("r2c_mechanistic_remediation") or {}
    transform_params = audit.get("transform_params") or {}
    value = transform_params.get("support_reference_attenuation_guard_clip_fraction")
    if value is None:
        return R9E_GUARD_CLIP_FRACTION
    return float(value)


def _blocked_row(
    *,
    seed: int,
    requested_profile: str,
    effective_profile: str | None,
    dataset: RealDataset,
    preset: str,
    blocked_reason: str,
    effective_matrix_route: str | None = None,
) -> R9eRow:
    return R9eRow(
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
        guard_clip_fraction=(
            R9E_GUARD_CLIP_FRACTION
            if requested_profile == R9E_FOCUS_PROFILE
            else None
        ),
        **_empty_decomposition(),
        **_empty_morphology_subset(),
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
        profiles if profiles is not None else R9E_AUDITED_PROFILES
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

    rows: list[R9eRow] = []
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
                            effective_profile=(
                                _exp09._effective_remediation_profile_for_dataset(
                                    dataset, profile
                                )
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
                            effective_profile=(
                                _exp09._effective_remediation_profile_for_dataset(
                                    dataset, profile
                                )
                            ),
                            dataset=dataset,
                            preset=preset,
                            blocked_reason="wavelength_grid_unknown",
                        )
                    )
            continue

        sanitized_real, sanitized_wl, _, real_blocked = sanitize_finite_spectra(
            real_X_raw, real_wl_raw, side="real"
        )
        if real_blocked is not None or sanitized_real is None or sanitized_wl is None:
            for seed in seeds:
                for profile in audited_profiles:
                    rows.append(
                        _blocked_row(
                            seed=seed,
                            requested_profile=profile,
                            effective_profile=(
                                _exp09._effective_remediation_profile_for_dataset(
                                    dataset, profile
                                )
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
                    seed, dataset, "r9e:real_downsample"
                ),
            )
            real_wl = sanitized_wl

            for profile in audited_profiles:
                effective = _exp09._effective_remediation_profile_for_dataset(
                    dataset, profile
                )
                try:
                    synthetic_run = _exp09._build_baseline_synthetic_run(
                        dataset=dataset,
                        preset=preset,
                        n_samples=n_synthetic_samples,
                        seed=seed,
                        remediation_profile=profile,
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
                        )
                    )
                    continue

                synth_X = np.asarray(synthetic_run.X, dtype=float)
                synth_wl = np.asarray(synthetic_run.wavelengths, dtype=float)
                sanitized_syn, sanitized_syn_wl, _, syn_blocked = (
                    sanitize_finite_spectra(synth_X, synth_wl, side="synthetic")
                )
                effective_route = _effective_matrix_route(synthetic_run.metadata)
                if (
                    syn_blocked is not None
                    or sanitized_syn is None
                    or sanitized_syn_wl is None
                ):
                    rows.append(
                        _blocked_row(
                            seed=seed,
                            requested_profile=profile,
                            effective_profile=effective,
                            dataset=dataset,
                            preset=preset,
                            blocked_reason=(
                                f"non_finite_spectra_synthetic: {syn_blocked}"
                            ),
                            effective_matrix_route=effective_route,
                        )
                    )
                    continue

                synth_downsampled = _exp09._downsample_rows(
                    sanitized_syn,
                    max_rows=max_real_samples,
                    random_state=_exp09._stable_dataset_seed(
                        seed, dataset, "r9e:syn_downsample"
                    ),
                )
                try:
                    real_aligned, syn_aligned, aligned_wl = align_to_real_grid(
                        real_X, real_wl, synth_downsampled, sanitized_syn_wl
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
                    real_aligned, syn_aligned, aligned_wl
                )
                decomposition = _exp10.compute_support_decomposition(
                    real_aligned,
                    syn_aligned,
                    aligned_wl,
                    support_low_nm=support_low_nm,
                    support_high_nm=support_high_nm,
                )
                rows.append(
                    R9eRow(
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
                        off_support_mean_delta=decomposition[
                            "off_support_mean_delta"
                        ],
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
                        **_morphology_subset(metrics),
                        guard_clip_fraction=_guard_clip_fraction(
                            profile, synthetic_run.metadata
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


PAIRED_DELTA_ATTRS: tuple[str, ...] = (
    "global_mean_delta",
    "support_mean_delta",
    "support_weighted_delta",
    "off_support_weighted_delta",
    "morphology_gap_score",
    "log10_derivative_std_p50_ratio",
    "mean_curve_corr",
)


def _paired_delta_columns() -> list[str]:
    return [
        f"delta_r9e_minus_{ref}__{attr}"
        for ref in R9E_PAIRED_REFERENCE_PROFILES
        for attr in PAIRED_DELTA_ATTRS
    ]


def _csv_fieldnames() -> list[str]:
    return [field.name for field in fields(R9eRow)] + _paired_delta_columns()


def _row_key(row: R9eRow) -> tuple[int, str]:
    return (int(row.seed), str(row.dataset))


def _paired_deltas_vs_references(rows: Sequence[R9eRow]) -> list[dict[str, Any]]:
    compared = [row for row in rows if row.status == "compared"]
    by_profile: dict[str, dict[tuple[int, str], R9eRow]] = {}
    for row in compared:
        by_profile.setdefault(row.remediation_profile, {})[_row_key(row)] = row

    entries: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for row in compared:
        if row.remediation_profile != R9E_FOCUS_PROFILE:
            continue
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        entry: dict[str, Any] = {
            "seed": int(row.seed),
            "dataset": str(row.dataset),
            "r9e_global_mean_delta": row.global_mean_delta,
            "r9e_support_mean_delta": row.support_mean_delta,
            "r9e_guard_clip_fraction": row.guard_clip_fraction,
        }
        for ref in R9E_PAIRED_REFERENCE_PROFILES:
            ref_row = by_profile.get(ref, {}).get(key)
            for attr in PAIRED_DELTA_ATTRS:
                col = f"delta_r9e_minus_{ref}__{attr}"
                if ref_row is None:
                    entry[col] = None
                    continue
                lhs = getattr(row, attr)
                rhs = getattr(ref_row, attr)
                entry[col] = None if lhs is None or rhs is None else float(lhs) - float(rhs)
        entries.append(entry)
    return entries


def write_csv(rows: list[R9eRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    paired_by_key = {
        (int(entry["seed"]), str(entry["dataset"])): entry
        for entry in _paired_deltas_vs_references(rows)
    }
    paired_cols = _paired_delta_columns()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_csv_fieldnames())
        writer.writeheader()
        for row in rows:
            record = row.to_dict()
            paired = (
                paired_by_key.get(_row_key(row))
                if row.remediation_profile == R9E_FOCUS_PROFILE
                and row.status == "compared"
                else None
            )
            for col in paired_cols:
                record[col] = paired.get(col) if paired is not None else None
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


def _format_gap_distribution(rows: Sequence[R9eRow]) -> str:
    counts = Counter(row.dominant_morphology_gap for row in rows)
    if not counts:
        return "n/a"
    return ", ".join(f"{label}={count}" for label, count in counts.most_common())


def _aggregate_by_profile(rows: Sequence[R9eRow]) -> list[dict[str, Any]]:
    compared = [row for row in rows if row.status == "compared"]
    grouped: dict[str, list[R9eRow]] = {}
    order: list[str] = []
    for row in compared:
        if row.remediation_profile not in grouped:
            grouped[row.remediation_profile] = []
            order.append(row.remediation_profile)
        grouped[row.remediation_profile].append(row)

    out: list[dict[str, Any]] = []
    for profile in order:
        bucket = grouped[profile]
        out.append(
            {
                "profile": profile,
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
                "median_mean_curve_corr": _median(
                    row.mean_curve_corr for row in bucket
                ),
                "median_guard_clip_fraction": _median(
                    row.guard_clip_fraction for row in bucket
                ),
                "dominant_gap_distribution": _format_gap_distribution(bucket),
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
    sentinel_tokens: Sequence[str] | None = None,
    profiles: Sequence[str] | None = None,
    support_low_nm: float = SUPPORT_LOW_NM,
    support_high_nm: float = SUPPORT_HIGH_NM,
) -> str:
    rows: list[R9eRow] = result["rows"]
    tokens = (
        list(sentinel_tokens)
        if sentinel_tokens is not None
        else list(result.get("sentinel_tokens", DEFAULT_SENTINEL_TOKENS))
    )
    audited = (
        list(profiles)
        if profiles is not None
        else list(result.get("audited_profiles", R9E_AUDITED_PROFILES))
    )
    compared = [row for row in rows if row.status == "compared"]
    blocked = [row for row in rows if row.status == "blocked"]
    profile_summary = _aggregate_by_profile(rows)

    command = (
        "PYTHONPATH=bench/nirs_synthetic_pfn/src "
        "python bench/nirs_synthetic_pfn/experiments/"
        "exp15_diesel_pathlength_reference_attenuation_audit.py "
        f"--n-synthetic-samples {n_synthetic_samples} "
        f"--max-real-samples {max_real_samples} "
        f"--max-sentinel-datasets {max_sentinel_datasets} "
        f"--sentinel-tokens {','.join(tokens)} "
        f"--seeds {','.join(str(int(s)) for s in seeds)} "
        f"--profiles {','.join(audited)}"
    )

    lines: list[str] = [
        "# R9e DIESEL Pathlength/Reference Attenuation Diagnostic Audit",
        "",
        "## Scope and Non-Gate Disclaimer",
        "",
        "- Bench-only, mechanistic, diagnostic-only repeated-seed audit. Comparison space is `uncalibrated_raw` only.",
        "- R3d remains the accepted baseline; R9e is diagnostic-only, not promoted, no gate.",
        "- No calibration, real-stat capture, PCA/covariance/noise capture, ML/DL, labels, targets, splits, downstream feedback, threshold mutation, or metric mutation.",
        "- No `nirs4all/` integration is authorized by this audit.",
        "- R9e inherits R3d byte-for-byte through the non-negative output clip, then applies a positive support-only multiplicative factor in `[0.970, 0.985]` on `750-1550` nm.",
        "- Off-support cells are byte-identical to R3d; R9e adds no offset, no R9d shape, no support-mean renormalization, no readout transform, and no guard clip.",
        f"- Constants source: `{R9E_CONSTANTS_SOURCE}`.",
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
            "## Paired Deltas: R9e minus references",
            "",
            "Per-(seed, dataset) arithmetic deltas for R9e vs R3d/R4c/R9d on the same decomposition and morphology fields.",
            "",
            "| dataset | seed | R9e global | R9e support | guard_clip_fraction | vs R3d global | vs R4c global | vs R9d global | vs R3d gap | vs R4c derivative | vs R9d mean_curve_corr |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    paired = _paired_deltas_vs_references(rows)
    if not paired:
        lines.append("| (no R9e compared rows) | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
    else:
        for entry in paired:
            lines.append(
                f"| `{entry['dataset']}` | {entry['seed']} | "
                f"{_fmt(entry['r9e_global_mean_delta'])} | "
                f"{_fmt(entry['r9e_support_mean_delta'])} | "
                f"{_fmt(entry['r9e_guard_clip_fraction'])} | "
                f"{_fmt(entry['delta_r9e_minus_r3d_diesel_matrix_v1__global_mean_delta'])} | "
                f"{_fmt(entry['delta_r9e_minus_r4c_diesel_balanced_derivative_v1__global_mean_delta'])} | "
                f"{_fmt(entry['delta_r9e_minus_r9d_diesel_energy_normalized_support_redistribution_v1__global_mean_delta'])} | "
                f"{_fmt(entry['delta_r9e_minus_r3d_diesel_matrix_v1__morphology_gap_score'])} | "
                f"{_fmt(entry['delta_r9e_minus_r4c_diesel_balanced_derivative_v1__log10_derivative_std_p50_ratio'])} | "
                f"{_fmt(entry['delta_r9e_minus_r9d_diesel_energy_normalized_support_redistribution_v1__mean_curve_corr'])} |"
            )

    lines.extend(
        [
            "",
            "## Mean-Shift Decomposition (uncalibrated_raw)",
            "",
            "| dataset | seed | profile | effective profile | global mean delta | support weighted delta | off-support weighted delta | morphology gap | derivative | mean_curve_corr | dominant gap | guard_clip_fraction | status |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row.dataset}` | {row.seed} | `{row.remediation_profile}` | "
            f"`{row.effective_remediation_profile or 'none'}` | "
            f"{_fmt(row.global_mean_delta)} | "
            f"{_fmt(row.support_weighted_delta)} | "
            f"{_fmt(row.off_support_weighted_delta)} | "
            f"{_fmt(row.morphology_gap_score)} | "
            f"{_fmt(row.log10_derivative_std_p50_ratio)} | "
            f"{_fmt(row.mean_curve_corr)} | "
            f"`{row.dominant_morphology_gap}` | "
            f"{_fmt(row.guard_clip_fraction)} | `{row.status}` |"
        )

    lines.extend(
        [
            "",
            "## R9e Mechanism Provenance",
            "",
            "- Explicit route key: `_r9e_diesel_reference_attenuation_route`.",
            "- Application stage: `after_r3d_output_clip`.",
            "- Support: fixed `750-1550` nm.",
            "- Factor: positive row-wise uniform draw in `[0.970, 0.985]`.",
            "- Off-support unchanged: `true`.",
            "- Adds offset: `false`; calibration: `false`; uses real stats/PCA/noise/ML/DL/labels/targets/splits: `false`; mutates thresholds/metrics: `false`.",
            "- Guard clip fraction for R9e is `0` because no additional clip is applied.",
            "",
            "## Decision",
            "",
            "Diagnostic-only, not promoted, no gate, no `nirs4all/` integration. R3d remains the accepted DIESEL baseline; R9e is evidence for the pathlength/reference attenuation hypothesis only.",
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
                    "seeds": [int(s) for s in seeds],
                    "audited_profiles": audited,
                    "support_low_nm": float(support_low_nm),
                    "support_high_nm": float(support_high_nm),
                    "compared_row_count": len(compared),
                    "blocked_row_count": len(blocked),
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
    parser.add_argument("--n-synthetic-samples", type=int, default=64)
    parser.add_argument("--max-real-samples", type=int, default=64)
    parser.add_argument("--max-sentinel-datasets", type=int, default=8)
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--sentinel-tokens",
        type=_parse_csv_list,
        default=list(DEFAULT_SENTINEL_TOKENS),
    )
    parser.add_argument(
        "--profiles",
        type=_parse_csv_list,
        default=list(R9E_AUDITED_PROFILES),
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
    rows: list[R9eRow] = result["rows"]
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

