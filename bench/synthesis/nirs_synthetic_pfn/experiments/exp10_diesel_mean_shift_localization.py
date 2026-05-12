"""R9a DIESEL mean-shift localization audit (bench-only, diagnostic-only).

Repeated-seed audit lane that compares synthetic DIESEL spectra against real
DIESEL sentinel cohorts using the ``uncalibrated_raw`` morphology metrics from
``exp09_sentinel_morphology_audit`` plus a fixed-support mean-shift
decomposition over the 750-1550 nm DIESEL real basis. The audit is
diagnostic-only, non-gate, and never:

- fits any real-data calibration, marginal mapping, or covariance capture;
- runs PCA, adversarial AUC, or any ML/DL classifier;
- consumes labels, splits, targets, or downstream feedback;
- modifies B2/B3/B4/B5 thresholds or metric definitions;
- promotes any DIESEL profile over the R3d baseline.

The decomposition splits ``synthetic_mean - real_mean`` over a fixed
750-1550 nm support window into a support component and an off-support
component, and reports both their unweighted means and their
support-weight-weighted contributions to the global mean shift. The identity
``global_mean_delta = support_weighted_delta + off_support_weighted_delta``
holds exactly to floating-point tolerance; ``decomposition_residual`` reports
the floating-point gap.

Audited profiles (DIESEL-only routes; non-DIESEL rows fall back byte-identical
to R3d):

- ``r3d_diesel_matrix_v1``
- ``r4a_diesel_basis_v1``
- ``r4c_diesel_balanced_derivative_v1``
- ``r5a_diesel_absorbance_readout_v1``
- ``r5b_diesel_transmittance_readout_v1``
- ``r5c_diesel_blank_referenced_intensity_v1``
- ``r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1``

Re-uses ``exp09_sentinel_morphology_audit`` selection, dataset-token routing,
synthetic builder call, morphology metrics, and remediation provenance fields
without modification. R8b (and R8a upstream) only modulates the support of
the R4c base, so the support-mean of the R8b base is exactly the R4c support
mean by construction; this audit reports that decomposition row by row.
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


def _load_exp09_module() -> ModuleType:
    """Load ``exp09_sentinel_morphology_audit`` from this directory."""
    name = "exp09_sentinel_morphology_audit"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / "exp09_sentinel_morphology_audit.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_exp09 = _load_exp09_module()


# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------

R9A_AUDITED_PROFILES: tuple[str, ...] = (
    "r3d_diesel_matrix_v1",
    "r4a_diesel_basis_v1",
    "r4c_diesel_balanced_derivative_v1",
    "r5a_diesel_absorbance_readout_v1",
    "r5b_diesel_transmittance_readout_v1",
    "r5c_diesel_blank_referenced_intensity_v1",
    "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1",
)

DEFAULT_SEEDS: tuple[int, ...] = (20260430, 20260431, 20260432)
DEFAULT_N_SYNTHETIC_SAMPLES: int = 64
DEFAULT_MAX_REAL_SAMPLES: int = 64
DEFAULT_MAX_SENTINEL_DATASETS: int = 8
DEFAULT_SENTINEL_TOKENS: tuple[str, ...] = ("DIESEL",)

DEFAULT_REPORT = Path(
    "bench/nirs_synthetic_pfn/reports/r9a_diesel_mean_shift_localization_audit.md"
)
DEFAULT_CSV = Path(
    "bench/nirs_synthetic_pfn/reports/r9a_diesel_mean_shift_localization_audit.csv"
)
R9A_AUDIT_SCOPE = "bench_only_r9a_diesel_mean_shift_localization_audit"
COMPARISON_SPACE = "uncalibrated_raw"

SUPPORT_LOW_NM: float = 750.0
SUPPORT_HIGH_NM: float = 1550.0


# ---------------------------------------------------------------------------
# Support decomposition.
# ---------------------------------------------------------------------------


def compute_support_decomposition(
    real_X: np.ndarray,
    synthetic_X: np.ndarray,
    wavelengths: np.ndarray,
    *,
    support_low_nm: float = SUPPORT_LOW_NM,
    support_high_nm: float = SUPPORT_HIGH_NM,
) -> dict[str, Any]:
    """Decompose ``synthetic_mean - real_mean`` over a fixed support window.

    Splits the wavelength axis into ``[support_low_nm, support_high_nm]``
    (inclusive) and its complement, then returns:

    - ``support_mean_delta``: unweighted average of
      ``synthetic_mean - real_mean`` over the support indices.
    - ``off_support_mean_delta``: unweighted average over the off-support
      indices (``0.0`` when no off-support indices exist).
    - ``support_weight`` / ``off_support_weight``: index counts normalized by
      ``n_wavelengths``.
    - ``support_weighted_delta`` / ``off_support_weighted_delta``: weighted
      contributions to the global mean shift.
    - ``global_mean_delta``: unweighted average of ``synthetic_mean -
      real_mean`` over the full wavelength axis.
    - ``decomposition_residual``: ``global_mean_delta -
      (support_weighted_delta + off_support_weighted_delta)``; the identity
      holds exactly to floating-point tolerance.
    """
    real = np.asarray(real_X, dtype=float)
    synth = np.asarray(synthetic_X, dtype=float)
    wl = np.asarray(wavelengths, dtype=float).ravel()
    if real.ndim != 2 or synth.ndim != 2:
        raise ValueError("expected 2D real/synthetic spectra")
    if real.shape[1] != synth.shape[1]:
        raise ValueError("real and synthetic must share the wavelength axis")
    if real.shape[1] != wl.size:
        raise ValueError("wavelengths must match the spectra column count")

    n_wl = int(wl.size)
    if n_wl == 0:
        zero = 0.0
        return {
            "support_low_nm": float(support_low_nm),
            "support_high_nm": float(support_high_nm),
            "support_count": 0,
            "off_support_count": 0,
            "support_weight": zero,
            "off_support_weight": zero,
            "support_mean_delta": zero,
            "off_support_mean_delta": zero,
            "support_weighted_delta": zero,
            "off_support_weighted_delta": zero,
            "global_mean_delta": zero,
            "decomposition_residual": zero,
        }

    support_mask = (wl >= support_low_nm) & (wl <= support_high_nm)
    support_count = int(support_mask.sum())
    off_count = int(n_wl - support_count)
    support_weight = support_count / n_wl
    off_support_weight = off_count / n_wl

    real_mean_curve = (
        real.mean(axis=0) if real.shape[0] > 0 else np.zeros(n_wl, dtype=float)
    )
    synth_mean_curve = (
        synth.mean(axis=0) if synth.shape[0] > 0 else np.zeros(n_wl, dtype=float)
    )
    delta = synth_mean_curve - real_mean_curve

    if support_count > 0:
        support_mean_delta = float(delta[support_mask].mean())
    else:
        support_mean_delta = 0.0
    if off_count > 0:
        off_support_mean_delta = float(delta[~support_mask].mean())
    else:
        off_support_mean_delta = 0.0

    support_weighted_delta = float(support_weight * support_mean_delta)
    off_support_weighted_delta = float(off_support_weight * off_support_mean_delta)
    global_mean_delta = float(delta.mean())
    decomposition_residual = float(
        global_mean_delta - (support_weighted_delta + off_support_weighted_delta)
    )

    return {
        "support_low_nm": float(support_low_nm),
        "support_high_nm": float(support_high_nm),
        "support_count": support_count,
        "off_support_count": off_count,
        "support_weight": float(support_weight),
        "off_support_weight": float(off_support_weight),
        "support_mean_delta": support_mean_delta,
        "off_support_mean_delta": off_support_mean_delta,
        "support_weighted_delta": support_weighted_delta,
        "off_support_weighted_delta": off_support_weighted_delta,
        "global_mean_delta": global_mean_delta,
        "decomposition_residual": decomposition_residual,
    }


# ---------------------------------------------------------------------------
# Row dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class R9aRow:
    """One R9a row for a (seed, profile, dataset) triple."""

    status: str  # "compared" | "blocked"
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
    # Support decomposition fields.
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
    # Morphology metrics (subset of exp09 contract).
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
    # Audit flags (every row).
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
    audit_scope: str
    blocked_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _audit_fields() -> dict[str, Any]:
    return {
        "audit_oracle": False,
        "audit_label_inputs_used": False,
        "audit_target_inputs_used": False,
        "audit_split_inputs_used": False,
        "audit_source_oracle_used": False,
        "audit_learned": False,
        "audit_real_stat_capture": False,
        "audit_thresholds_modified": False,
        "audit_metrics_modified": False,
        "audit_imputed": False,
        "audit_replays_real_rows": False,
        "audit_scope": R9A_AUDIT_SCOPE,
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


def _validate_profiles(profiles: Sequence[str]) -> tuple[str, ...]:
    if not profiles:
        raise ValueError("at least one remediation profile must be provided")
    invalid = [
        profile for profile in profiles if profile not in R9A_AUDITED_PROFILES
    ]
    if invalid:
        raise ValueError(
            f"unknown R9a profiles {invalid!r}; "
            f"valid profiles are {list(R9A_AUDITED_PROFILES)}"
        )
    return tuple(profiles)


def _blocked_row(
    *,
    seed: int,
    requested_profile: str,
    effective_profile: str | None,
    dataset: RealDataset,
    preset: str,
    blocked_reason: str,
    effective_matrix_route: str | None = None,
) -> R9aRow:
    return R9aRow(
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
    """Run the R9a DIESEL mean-shift localization audit."""
    seeds = tuple(int(s) for s in seeds)
    if not seeds:
        raise ValueError("at least one seed must be provided")
    audited_profiles = _validate_profiles(
        profiles if profiles is not None else R9A_AUDITED_PROFILES
    )
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

    rows: list[R9aRow] = []
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
        except Exception as exc:  # noqa: BLE001 - bench-only diagnostic surfacing
            for seed in seeds:
                for profile in audited_profiles:
                    effective = _exp09._effective_remediation_profile_for_dataset(
                        dataset, profile
                    )
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

        if is_index_fallback_grid(real_wl_raw):
            for seed in seeds:
                for profile in audited_profiles:
                    effective = _exp09._effective_remediation_profile_for_dataset(
                        dataset, profile
                    )
                    rows.append(
                        _blocked_row(
                            seed=seed,
                            requested_profile=profile,
                            effective_profile=effective,
                            dataset=dataset,
                            preset=preset,
                            blocked_reason=(
                                "wavelength_grid_unknown: real wavelengths "
                                "could not be parsed"
                            ),
                        )
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
                for profile in audited_profiles:
                    effective = _exp09._effective_remediation_profile_for_dataset(
                        dataset, profile
                    )
                    rows.append(
                        _blocked_row(
                            seed=seed,
                            requested_profile=profile,
                            effective_profile=effective,
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
                    seed, dataset, "r9a:real_downsample"
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
                effective_route = _effective_matrix_route(
                    profile=profile,
                    metadata=synthetic_run.metadata,
                )
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
                        seed, dataset, "r9a:syn_downsample"
                    ),
                )
                synth_wl_clean = sanitized_syn_wl

                try:
                    real_aligned, syn_aligned, aligned_wl = align_to_real_grid(
                        real_X, real_wl, synth_downsampled, synth_wl_clean
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
                decomposition = compute_support_decomposition(
                    real_aligned,
                    syn_aligned,
                    aligned_wl,
                    support_low_nm=support_low_nm,
                    support_high_nm=support_high_nm,
                )

                rows.append(
                    R9aRow(
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
                        support_weighted_delta=decomposition["support_weighted_delta"],
                        off_support_weighted_delta=decomposition[
                            "off_support_weighted_delta"
                        ],
                        global_mean_delta=decomposition["global_mean_delta"],
                        decomposition_residual=decomposition["decomposition_residual"],
                        **_morphology_subset(metrics),
                        **_audit_fields(),
                        blocked_reason="",
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
        "sentinel_tokens": list(tokens),
        "seeds": list(seeds),
        "audited_profiles": list(audited_profiles),
        "support_low_nm": float(support_low_nm),
        "support_high_nm": float(support_high_nm),
    }


def _effective_matrix_route(
    *,
    profile: str,
    metadata: dict[str, Any] | None,
) -> str | None:
    audit = (metadata or {}).get("r2c_mechanistic_remediation") or {}
    transform_params = audit.get("transform_params") or {}
    route = _exp09._effective_matrix_route_from_metadata(
        audit=audit,
        transform_params=transform_params,
    )
    return cast(str | None, route)


# ---------------------------------------------------------------------------
# CSV / Markdown output.
# ---------------------------------------------------------------------------


def _csv_fieldnames() -> list[str]:
    return [field.name for field in fields(R9aRow)]


def write_csv(rows: list[R9aRow], path: Path) -> None:
    """Write rows to ``path``; always emits a stable header even when empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def _fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def _median(values: Iterable[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None]
    if not finite:
        return None
    return float(np.median(np.asarray(finite, dtype=float)))


def _max(values: Iterable[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None]
    if not finite:
        return None
    return float(np.max(np.asarray(finite, dtype=float)))


def _format_gap_distribution(rows: Sequence[R9aRow]) -> str:
    counts = Counter(row.dominant_morphology_gap for row in rows)
    if not counts:
        return "n/a"
    return ", ".join(
        f"{label}={count}" for label, count in counts.most_common()
    )


def _aggregate_by_profile(
    rows: Sequence[R9aRow],
) -> list[dict[str, Any]]:
    """Aggregate ``compared`` rows per ``remediation_profile``.

    Profile order follows first appearance in ``rows`` so the report is
    stable with respect to the audit driver iteration order.
    """
    compared = [row for row in rows if row.status == "compared"]
    order: list[str] = []
    grouped: dict[str, list[R9aRow]] = {}
    for row in compared:
        key = row.remediation_profile
        if key not in grouped:
            order.append(key)
            grouped[key] = []
        grouped[key].append(row)

    summary: list[dict[str, Any]] = []
    for profile in order:
        bucket = grouped[profile]
        summary.append(
            {
                "profile": profile,
                "n": len(bucket),
                "median_global_mean_delta": _median(
                    r.global_mean_delta for r in bucket
                ),
                "median_support_weight": _median(
                    r.support_weight for r in bucket
                ),
                "median_off_support_weight": _median(
                    r.off_support_weight for r in bucket
                ),
                "max_off_support_weight": _max(
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
                "dominant_gap_distribution": _format_gap_distribution(bucket),
            }
        )
    return summary


OFF_SUPPORT_NULL_NOTE: str = (
    "Note: `off_support_weight` median and max are 0.0 across every compared "
    "row, so the off-support diagnostic is structurally null because the "
    "aligned wavelength grid is entirely within the support window; the next "
    "mechanistic step should focus on the support-level hydrocarbon/readout "
    "mean, not on off-support."
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
    rows: list[R9aRow] = result["rows"]
    tokens = (
        list(sentinel_tokens)
        if sentinel_tokens is not None
        else list(result.get("sentinel_tokens", DEFAULT_SENTINEL_TOKENS))
    )
    audited = (
        list(profiles)
        if profiles is not None
        else list(result.get("audited_profiles", R9A_AUDITED_PROFILES))
    )
    seeds_list = [int(s) for s in seeds]
    compared = [row for row in rows if row.status == "compared"]
    blocked = [row for row in rows if row.status == "blocked"]

    command = (
        "PYTHONPATH=bench/nirs_synthetic_pfn/src "
        "python bench/nirs_synthetic_pfn/experiments/exp10_diesel_mean_shift_localization.py "
        f"--n-synthetic-samples {n_synthetic_samples} "
        f"--max-real-samples {max_real_samples} "
        f"--max-sentinel-datasets {max_sentinel_datasets} "
        f"--sentinel-tokens {','.join(tokens)} "
        f"--seeds {','.join(str(s) for s in seeds_list)} "
        f"--profiles {','.join(audited)}"
    )

    lines: list[str] = [
        "# R9a DIESEL Mean-Shift Localization Audit",
        "",
        "## Scope and Non-Gate Disclaimer",
        "",
        "- Bench-only, diagnostic-only repeated-seed audit. Comparison space is `uncalibrated_raw` only.",
        "- This audit does NOT establish any B2/B3/B4/B5 pass and does not modify any gate threshold or metric.",
        "- no calibration fitted, captured, or applied (no marginal, covariance, or quantile mapping).",
        "- no PCA/covariance capture from real data; no adversarial AUC; no ML/DL.",
        "- no labels, splits, targets, or downstream feedback are consulted.",
        "- DIESEL routes only; non-DIESEL rows fall back byte-identical to R3d via the existing exp09 routing.",
        "- Decomposes `synthetic_mean - real_mean` over the fixed "
        f"`{support_low_nm:g}-{support_high_nm:g}` nm support; "
        "the identity `global_mean_delta = support_weighted_delta + off_support_weighted_delta` "
        "holds to floating-point tolerance.",
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
        f"- Audited profiles: `{', '.join(audited)}`",
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
                "`status == compared`. `dominant gap dist.` is the count of "
                "`dominant_morphology_gap` labels in the bucket.",
                "",
                (
                    "| profile | n | median global mean delta | median support "
                    "weight | median off-support weight | median morphology "
                    "gap score | median mean curve corr | median log10 deriv "
                    "std p50 ratio | dominant gap dist. |"
                ),
                (
                    "|---|---:|---:|---:|---:|---:|---:|---:|---|"
                ),
            ]
        )
        for entry in profile_summary:
            lines.append(
                f"| `{entry['profile']}` | {entry['n']} | "
                f"{_fmt(entry['median_global_mean_delta'])} | "
                f"{_fmt(entry['median_support_weight'])} | "
                f"{_fmt(entry['median_off_support_weight'])} | "
                f"{_fmt(entry['median_morphology_gap_score'])} | "
                f"{_fmt(entry['median_mean_curve_corr'])} | "
                f"{_fmt(entry['median_log10_derivative_std_p50_ratio'])} | "
                f"{entry['dominant_gap_distribution']} |"
            )

        off_support_values = [
            entry["max_off_support_weight"] for entry in profile_summary
        ]
        all_zero = bool(off_support_values) and all(
            value is not None and float(value) == 0.0
            for value in off_support_values
        )
        if all_zero:
            lines.extend(["", OFF_SUPPORT_NULL_NOTE])

    lines.extend(
        [
            "",
            "## Mean-Shift Decomposition (uncalibrated_raw)",
            "",
            "| dataset | seed | profile | effective profile | n real | n syn | global mean delta | support weighted delta | off-support weighted delta | support mean delta | off-support mean delta | support weight | decomposition residual | log10 std ratio | mean curve corr | dominant gap | status |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        effective_profile_label = (
            row.effective_remediation_profile
            if row.effective_remediation_profile is not None
            else "none"
        )
        lines.append(
            f"| `{row.dataset}` | {row.seed} | `{row.remediation_profile}` | "
            f"`{effective_profile_label}` | "
            f"{row.n_real_samples} | {row.n_synthetic_samples} | "
            f"{_fmt(row.global_mean_delta)} | "
            f"{_fmt(row.support_weighted_delta)} | "
            f"{_fmt(row.off_support_weighted_delta)} | "
            f"{_fmt(row.support_mean_delta)} | "
            f"{_fmt(row.off_support_mean_delta)} | "
            f"{_fmt(row.support_weight)} | "
            f"{_fmt(row.decomposition_residual)} | "
            f"{_fmt(row.log10_global_std_ratio)} | "
            f"{_fmt(row.mean_curve_corr)} | "
            f"`{row.dominant_morphology_gap}` | "
            f"`{row.status}` |"
        )

    lines.extend(
        [
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
            f"- `audit_scope={R9A_AUDIT_SCOPE}`",
            "",
            "## Decision",
            "",
            (
                "Diagnostic-only mean-shift localization snapshot; this lane is non-gate "
                "and does not promote any DIESEL profile over R3d. The decomposition "
                "isolates how much of the global mean shift sits inside the fixed "
                f"{support_low_nm:g}-{support_high_nm:g} nm DIESEL real basis support and "
                "how much sits outside it. R8b is expected to share the R4c support "
                "mean exactly because R8b applies a multiplicative mean-preserving "
                "modulation only on the support, while off-support values are "
                "unchanged from the R4c base. Therefore R8b and R4c support, "
                "off-support, and global mean-shift decomposition fields should "
                "match to numerical tolerance; R8b/R4c differences belong to "
                "support shape/derivative diagnostics, not this mean-shift "
                "decomposition."
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
        help=(
            "Cap on sentinel datasets after priority-based selection. "
            "0 (or any non-positive value) means every token-matched sentinel row."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated repeated seeds.",
    )
    parser.add_argument(
        "--sentinel-tokens",
        type=str,
        default=",".join(DEFAULT_SENTINEL_TOKENS),
        help=(
            "Comma-separated case-insensitive tokens used to select sentinel "
            "datasets. Default DIESEL since this audit is DIESEL-only."
        ),
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default=",".join(R9A_AUDITED_PROFILES),
        help=(
            "Comma-separated remediation profiles to audit. Must be a subset of "
            f"{list(R9A_AUDITED_PROFILES)}."
        ),
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--support-low-nm", type=float, default=SUPPORT_LOW_NM)
    parser.add_argument("--support-high-nm", type=float, default=SUPPORT_HIGH_NM)
    args = parser.parse_args()

    sentinel_tokens = _parse_csv_list(args.sentinel_tokens)
    profiles = _parse_csv_list(args.profiles)
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
    "OFF_SUPPORT_NULL_NOTE",
    "R9A_AUDITED_PROFILES",
    "R9A_AUDIT_SCOPE",
    "R9aRow",
    "SUPPORT_HIGH_NM",
    "SUPPORT_LOW_NM",
    "_aggregate_by_profile",
    "compute_support_decomposition",
    "main",
    "render_markdown",
    "run_audit",
    "write_csv",
]


if __name__ == "__main__":
    main()
