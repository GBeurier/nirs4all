"""R9f DIESEL pre-offset pathlength/reference attenuation diagnostic audit.

Bench-only diagnostic-only repeated-seed audit for the R9f hypothesis. R9f is
not a gate, not promoted over R3d, and does not authorize any ``nirs4all/``
integration. It applies a pre-declared positive row-wise attenuation in
``[0.970, 0.985]`` on fixed ``750-1550`` nm support only to the R3d
Beer-Lambert continuum/path component before additive detector offset and
before the existing output clip.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
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

RealDataset = _exp15.RealDataset
R9fRow = _exp15.R9eRow
align_to_real_grid = _exp15.align_to_real_grid
discover_local_real_datasets = _exp15.discover_local_real_datasets
is_index_fallback_grid = _exp15.is_index_fallback_grid
load_real_spectra = _exp15.load_real_spectra
sanitize_finite_spectra = _exp15.sanitize_finite_spectra

R9F_AUDITED_PROFILES: tuple[str, ...] = (
    "r3d_diesel_matrix_v1",
    "r4b_diesel_derivative_restore_v1",
    "r4c_diesel_balanced_derivative_v1",
    "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1",
    "r9d_diesel_energy_normalized_support_redistribution_v1",
    "r9e_diesel_pathlength_reference_attenuation_v1",
    "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1",
)
R9F_FOCUS_PROFILE = "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1"
R9F_BASE_PROFILE = "r3d_diesel_matrix_v1"
R9F_PAIRED_REFERENCE_PROFILES: tuple[str, ...] = (
    "r3d_diesel_matrix_v1",
    "r9e_diesel_pathlength_reference_attenuation_v1",
    "r4c_diesel_balanced_derivative_v1",
)
R9F_CONSTANTS_SOURCE = (
    "predeclared_generic_blank_reference_pathlength_attenuation_prior"
)

DEFAULT_SEEDS: tuple[int, ...] = (20260501, 20260502, 20260503)
DEFAULT_N_SYNTHETIC_SAMPLES = 64
DEFAULT_MAX_REAL_SAMPLES = 64
DEFAULT_MAX_SENTINEL_DATASETS = 8
DEFAULT_SENTINEL_TOKENS: tuple[str, ...] = ("DIESEL",)
DEFAULT_REPORT = Path(
    "bench/nirs_synthetic_pfn/reports/"
    "r9f_diesel_pre_offset_pathlength_reference_attenuation_audit.md"
)
DEFAULT_CSV = Path(
    "bench/nirs_synthetic_pfn/reports/"
    "r9f_diesel_pre_offset_pathlength_reference_attenuation_audit.csv"
)
R9F_AUDIT_SCOPE = (
    "bench_only_r9f_diesel_pre_offset_pathlength_reference_attenuation_audit"
)
COMPARISON_SPACE = "uncalibrated_raw"
SUPPORT_LOW_NM: float = _exp10.SUPPORT_LOW_NM
SUPPORT_HIGH_NM: float = _exp10.SUPPORT_HIGH_NM
R9F_GUARD_CLIP_FRACTION = 0.0


def _audit_fields() -> dict[str, Any]:
    fields_dict = cast(dict[str, Any], _exp15._audit_fields())
    fields_dict["audit_scope"] = R9F_AUDIT_SCOPE
    return fields_dict


def _validate_profiles(profiles: Sequence[str]) -> tuple[str, ...]:
    if not profiles:
        raise ValueError("at least one remediation profile must be provided")
    invalid = [profile for profile in profiles if profile not in R9F_AUDITED_PROFILES]
    if invalid:
        raise ValueError(
            f"unknown R9f profiles {invalid!r}; "
            f"valid profiles are {list(R9F_AUDITED_PROFILES)}"
        )
    return tuple(profiles)


def _effective_profile_for_dataset(dataset: Any, profile: str | None) -> str | None:
    if profile == R9F_FOCUS_PROFILE:
        text, tokens = _exp09._dataset_text_and_tokens(dataset)
        if _exp09._has_r2j_diesel_marker(text, tokens):
            return profile
        return cast(
            str | None,
            _exp09._effective_remediation_profile_for_dataset(
                dataset,
                R9F_BASE_PROFILE,
            ),
        )
    return cast(
        str | None,
        _exp09._effective_remediation_profile_for_dataset(dataset, profile),
    )


def _r9f_token_source_overrides(dataset: Any) -> dict[str, object]:
    text, tokens = _exp09._dataset_text_and_tokens(dataset)
    if not _exp09._has_r2j_diesel_marker(text, tokens):
        return cast(dict[str, object], _exp09._r3d_token_source_overrides(dataset))
    overrides = cast(dict[str, object], _exp09._r2c_token_source_overrides(dataset))
    route = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp16_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    overrides["_r3d_diesel_readout_route"] = dict(route)
    overrides["_r9f_diesel_pre_offset_reference_attenuation_route"] = dict(route)
    return overrides


def _build_synthetic_run(
    *,
    dataset: Any,
    preset: str,
    n_samples: int,
    seed: int,
    remediation_profile: str,
) -> Any:
    effective = _effective_profile_for_dataset(dataset, remediation_profile)
    if effective != R9F_FOCUS_PROFILE:
        return _exp09._build_baseline_synthetic_run(
            dataset=dataset,
            preset=preset,
            n_samples=n_samples,
            seed=seed,
            remediation_profile=effective,
        )
    run_seed = _exp09._stable_dataset_seed(
        seed,
        dataset,
        f"r2b:on_demand_synthetic:{preset}",
    )
    source = _exp09._preset_source(
        preset,
        seed=run_seed,
        extra_overrides=_r9f_token_source_overrides(dataset),
    )
    record = _exp09.canonicalize_prior_config(source)
    return _exp09.build_synthetic_dataset_run(
        record,
        n_samples=n_samples,
        random_seed=run_seed,
        remediation_profile=effective,
    )


def _effective_matrix_route(metadata: dict[str, Any] | None) -> str | None:
    return cast(str | None, _exp15._effective_matrix_route(metadata))


def _guard_clip_fraction(profile: str, metadata: dict[str, Any] | None) -> float | None:
    if profile != R9F_FOCUS_PROFILE:
        return None
    audit = (metadata or {}).get("r2c_mechanistic_remediation") or {}
    transform_params = audit.get("transform_params") or {}
    value = transform_params.get("support_reference_attenuation_guard_clip_fraction")
    return R9F_GUARD_CLIP_FRACTION if value is None else float(value)


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
    return R9fRow(
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
            R9F_GUARD_CLIP_FRACTION
            if requested_profile == R9F_FOCUS_PROFILE
            else None
        ),
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
        profiles if profiles is not None else R9F_AUDITED_PROFILES
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
                                dataset, profile
                            ),
                            dataset=dataset,
                            preset=preset,
                            blocked_reason=f"{type(exc).__name__}: {exc}",
                        )
                    )
            continue

        if is_index_fallback_grid(real_wl_raw):
            blocked_reason = "wavelength_grid_unknown"
            for seed in seeds:
                for profile in audited_profiles:
                    rows.append(
                        _blocked_row(
                            seed=seed,
                            requested_profile=profile,
                            effective_profile=_effective_profile_for_dataset(
                                dataset, profile
                            ),
                            dataset=dataset,
                            preset=preset,
                            blocked_reason=blocked_reason,
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
                                dataset, profile
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
                    "r9f:real_downsample",
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
                            "r9f:syn_downsample",
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
                    R9fRow(
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


PAIRED_DELTA_ATTRS: tuple[str, ...] = _exp15.PAIRED_DELTA_ATTRS


def _paired_delta_columns() -> list[str]:
    return [
        f"delta_r9f_minus_{ref}__{attr}"
        for ref in R9F_PAIRED_REFERENCE_PROFILES
        for attr in PAIRED_DELTA_ATTRS
    ]


def _csv_fieldnames() -> list[str]:
    return [field.name for field in fields(R9fRow)] + _paired_delta_columns()


def _row_key(row: Any) -> tuple[int, str]:
    return (int(row.seed), str(row.dataset))


def _paired_deltas_vs_references(rows: Sequence[Any]) -> list[dict[str, Any]]:
    compared = [row for row in rows if row.status == "compared"]
    by_profile: dict[str, dict[tuple[int, str], Any]] = {}
    for row in compared:
        by_profile.setdefault(row.remediation_profile, {})[_row_key(row)] = row
    entries: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for row in compared:
        if row.remediation_profile != R9F_FOCUS_PROFILE:
            continue
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        entry: dict[str, Any] = {
            "seed": int(row.seed),
            "dataset": str(row.dataset),
            "r9f_global_mean_delta": row.global_mean_delta,
            "r9f_support_mean_delta": row.support_mean_delta,
            "r9f_guard_clip_fraction": row.guard_clip_fraction,
        }
        for ref in R9F_PAIRED_REFERENCE_PROFILES:
            ref_row = by_profile.get(ref, {}).get(key)
            for attr in PAIRED_DELTA_ATTRS:
                col = f"delta_r9f_minus_{ref}__{attr}"
                if ref_row is None:
                    entry[col] = None
                    continue
                lhs = getattr(row, attr)
                rhs = getattr(ref_row, attr)
                entry[col] = None if lhs is None or rhs is None else float(lhs) - float(rhs)
        entries.append(entry)
    return entries


def write_csv(rows: list[Any], path: Path) -> None:
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
                if row.remediation_profile == R9F_FOCUS_PROFILE
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


def _aggregate_by_profile(rows: Sequence[Any]) -> list[dict[str, Any]]:
    return cast(
        list[dict[str, Any]],
        _exp15._aggregate_by_profile(cast(Sequence[Any], rows)),
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
        else list(result.get("audited_profiles", R9F_AUDITED_PROFILES))
    )
    compared = [row for row in rows if row.status == "compared"]
    blocked = [row for row in rows if row.status == "blocked"]
    profile_summary = _aggregate_by_profile(rows)
    command = (
        "PYTHONPATH=bench/nirs_synthetic_pfn/src "
        "python bench/nirs_synthetic_pfn/experiments/"
        "exp16_diesel_pre_offset_pathlength_reference_attenuation_audit.py "
        f"--n-synthetic-samples {n_synthetic_samples} "
        f"--max-real-samples {max_real_samples} "
        f"--max-sentinel-datasets {max_sentinel_datasets} "
        f"--sentinel-tokens {','.join(tokens)} "
        f"--seeds {','.join(str(int(s)) for s in seeds)} "
        f"--profiles {','.join(audited)}"
    )
    lines = [
        "# R9f DIESEL Pre-Offset Pathlength/Reference Attenuation Diagnostic Audit",
        "",
        "## Scope and Non-Gate Disclaimer",
        "",
        "- Bench-only, mechanistic, diagnostic-only repeated-seed audit. Comparison space is `uncalibrated_raw` only.",
        "- R3d remains the accepted baseline; R9f is diagnostic-only, not promoted, no gate.",
        "- No calibration, real-stat capture, PCA/covariance/noise capture, ML/DL, labels, targets, splits, downstream feedback, threshold mutation, or metric mutation.",
        "- No `nirs4all/` integration is authorized by this audit.",
        "- R9f attenuates only `continuum * path_factors[:, None] * path_profile` before additive baseline and output clip on fixed `750-1550` nm support.",
        "- Feature residuals, additive baseline/offsets, readout transform, R9d shape, support mean renormalization, negative intercept, and extra clipping are unchanged/not used.",
        f"- Constants source: `{R9F_CONSTANTS_SOURCE}`.",
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
        lines.extend([
            "| profile | n | median global mean delta | median support mean delta | median morphology gap | median derivative | median mean_curve_corr | median guard_clip_fraction | dominant gap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
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
    lines.extend([
        "",
        "## Paired Deltas: R9f minus references",
        "",
        "Per-(seed, dataset) arithmetic deltas for R9f vs R3d/R9e/R4c on support decomposition, derivative, mean_curve_corr, and guard clip context.",
        "",
        "| dataset | seed | R9f global | R9f support | guard_clip_fraction | vs R3d global | vs R9e global | vs R4c global | vs R3d gap | vs R9e derivative | vs R4c mean_curve_corr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    paired = _paired_deltas_vs_references(rows)
    if not paired:
        lines.append("| (no R9f compared rows) | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
    else:
        for entry in paired:
            lines.append(
                f"| `{entry['dataset']}` | {entry['seed']} | "
                f"{_fmt(entry['r9f_global_mean_delta'])} | "
                f"{_fmt(entry['r9f_support_mean_delta'])} | "
                f"{_fmt(entry['r9f_guard_clip_fraction'])} | "
                f"{_fmt(entry['delta_r9f_minus_r3d_diesel_matrix_v1__global_mean_delta'])} | "
                f"{_fmt(entry['delta_r9f_minus_r9e_diesel_pathlength_reference_attenuation_v1__global_mean_delta'])} | "
                f"{_fmt(entry['delta_r9f_minus_r4c_diesel_balanced_derivative_v1__global_mean_delta'])} | "
                f"{_fmt(entry['delta_r9f_minus_r3d_diesel_matrix_v1__morphology_gap_score'])} | "
                f"{_fmt(entry['delta_r9f_minus_r9e_diesel_pathlength_reference_attenuation_v1__log10_derivative_std_p50_ratio'])} | "
                f"{_fmt(entry['delta_r9f_minus_r4c_diesel_balanced_derivative_v1__mean_curve_corr'])} |"
            )
    lines.extend([
        "",
        "## Mean-Shift Decomposition (uncalibrated_raw)",
        "",
        "| dataset | seed | profile | effective profile | global mean delta | support weighted delta | off-support weighted delta | morphology gap | derivative | mean_curve_corr | dominant gap | guard_clip_fraction | status |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ])
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
    lines.extend([
        "",
        "## R9f Mechanism Provenance",
        "",
        "- Explicit route key: `_r9f_diesel_pre_offset_reference_attenuation_route`.",
        "- Application stage: `before_additive_baseline_and_output_clip_on_continuum_path_component`.",
        "- Support: fixed `750-1550` nm.",
        "- Factor: positive row-wise uniform draw in `[0.970, 0.985]`.",
        "- Component-only: `true`; offset unchanged: `true`; feature residual unchanged: `true`; no additional clip: `true`.",
        "- Forbidden flags remain `false`: calibration, real stats, PCA, covariance/noise capture, ML/DL, labels, targets, splits, thresholds, metrics, downstream feedback.",
        "",
        "## Decision",
        "",
        "Diagnostic-only, not promoted, no gate, no `nirs4all/` integration. R3d remains the accepted DIESEL baseline; R9f is evidence for the pre-offset continuum/path attenuation hypothesis only.",
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
    ])
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
        default=list(R9F_AUDITED_PROFILES),
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
