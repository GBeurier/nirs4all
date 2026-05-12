"""P2-03 DIESEL support/off-support discriminability audit.

Bench-only read-only audit that checks when the existing R9e support-only
attenuation and P2a full-row attenuation can be distinguished. It introduces no
generator mechanism, profile, retune, gate, promotion, threshold mutation,
metric mutation, calibration, real-stat capture, PCA/noise capture, ML/DL,
labels, targets, splits, or production integration.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from collections import Counter
from collections.abc import Sequence
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


_exp25 = _load_module(
    "exp25_diesel_row_pathlength_reference_audit",
    "exp25_diesel_row_pathlength_reference_audit.py",
)

EXP26_AUDIT_SCOPE = "bench_only_p2_03_diesel_support_offsupport_discriminability_audit"
EXP26_DECISION = "diagnostic_read_only_no_gate_no_promotion"
COMPARISON_SPACE = "uncalibrated_raw"
SUPPORT_LOW_NM: float = _exp25.SUPPORT_LOW_NM
SUPPORT_HIGH_NM: float = _exp25.SUPPORT_HIGH_NM

R3D_PROFILE = _exp25.R3D_PROFILE
R9E_PROFILE = _exp25.R9E_PROFILE
P2A_PROFILE = _exp25.P2A_PROFILE

DEFAULT_SEEDS: tuple[int, ...] = (20260501, 20260502, 20260503)
DEFAULT_N_SYNTHETIC_SAMPLES = 64
DEFAULT_MAX_REAL_SAMPLES = 64
DEFAULT_MAX_SENTINEL_DATASETS = 8
DEFAULT_SENTINEL_TOKENS: tuple[str, ...] = ("DIESEL",)
DEFAULT_REPORT = Path("/tmp/exp26_diesel_support_offsupport_discriminability_audit.md")
DEFAULT_CSV = Path("/tmp/exp26_diesel_support_offsupport_discriminability_audit.csv")

REAL_ALIGNED_CASE = "real_aligned_current_cohort"
GENERATED_FULL_GRID_CASE = "generated_prior_full_grid_counterfactual"
EXP26_CASES: tuple[str, str] = (REAL_ALIGNED_CASE, GENERATED_FULL_GRID_CASE)


@dataclass
class Exp26Row:
    status: str
    case: str
    seed: int
    source: str
    task: str
    dataset: str
    synthetic_preset: str
    comparison_space: str
    grid_source: str
    grid_note: str
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
    r9e_stage_support_nm: str
    p2a_stage_support_nm: str
    r9e_support_only: bool
    p2a_support_only: bool
    r9e_off_support_unchanged_metadata: bool | None
    p2a_off_support_unchanged_metadata: bool | None
    r9e_factor_min: float | None
    r9e_factor_max: float | None
    p2a_factor_min: float | None
    p2a_factor_max: float | None
    r9e_vs_r3d_max_abs_delta_inside: float | None
    r9e_vs_r3d_max_abs_delta_outside: float | None
    p2a_vs_r3d_max_abs_delta_inside: float | None
    p2a_vs_r3d_max_abs_delta_outside: float | None
    r9e_vs_p2a_max_abs_delta_inside: float | None
    r9e_vs_p2a_max_abs_delta_outside: float | None
    r9e_vs_p2a_outside_minus_inside: float | None
    r9e_vs_p2a_outside_to_inside_ratio: float | None
    p2a_minus_r9e_delta_outside_vs_r3d: float | None
    distinguishable_by_off_support: bool
    current_cohort_can_distinguish: bool
    anti_leakage_flags_false: bool
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
    fields_dict = cast(dict[str, Any], _exp25._audit_fields())
    fields_dict["audit_scope"] = EXP26_AUDIT_SCOPE
    return fields_dict


def _anti_leakage_flags_false() -> bool:
    return all(value is False for key, value in _audit_fields().items() if key != "audit_scope")


def _transform_params(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return cast(dict[str, Any], _exp25._transform_params(metadata))


def _stage_support_nm(transform_params: dict[str, Any]) -> str:
    value = transform_params.get("row_pathlength_reference_applies_to")
    if value == "full_generated_wavelength_row":
        return "full_generated_row"
    support = transform_params.get("support_reference_attenuation_support_nm")
    if isinstance(support, (list, tuple)) and len(support) == 2:
        return f"{float(support[0]):g}-{float(support[1]):g}"
    return f"{SUPPORT_LOW_NM:g}-{SUPPORT_HIGH_NM:g}"


def _factor_min_max(transform_params: dict[str, Any], prefix: str) -> tuple[float | None, float | None]:
    min_value = transform_params.get(f"{prefix}_factor_min")
    max_value = transform_params.get(f"{prefix}_factor_max")
    return (
        None if min_value is None else float(min_value),
        None if max_value is None else float(max_value),
    )


def _support_mask(wavelengths: np.ndarray) -> np.ndarray:
    return (wavelengths >= SUPPORT_LOW_NM) & (wavelengths <= SUPPORT_HIGH_NM)


def _max_abs_delta(delta: np.ndarray, mask: np.ndarray) -> float | None:
    if delta.size == 0 or int(mask.sum()) == 0:
        return None
    return float(np.max(np.abs(delta[:, mask])))


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0.0:
        if numerator == 0.0:
            return 1.0
        return math.inf
    return float(numerator / denominator)


def _aligned_to_grid(
    *,
    real_x: np.ndarray,
    real_wl: np.ndarray,
    synthetic_x: np.ndarray,
    synthetic_wl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _, syn_aligned, aligned_wl = _exp25.align_to_real_grid(
        real_x,
        real_wl,
        synthetic_x,
        synthetic_wl,
    )
    return np.asarray(syn_aligned, dtype=float), np.asarray(aligned_wl, dtype=float)


def _row_factors_from_support(
    *,
    r3d_x: np.ndarray,
    r9e_x: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    support = _support_mask(wavelengths)
    if int(support.sum()) == 0:
        raise ValueError("cannot recover row factors without support wavelengths")
    base_block = r3d_x[:, support]
    r9e_block = r9e_x[:, support]
    ratio = np.divide(
        r9e_block,
        base_block,
        out=np.ones_like(base_block),
        where=base_block != 0.0,
    )
    factors = []
    for row_ratio, row_base in zip(ratio, base_block, strict=True):
        nonzero = row_ratio[row_base != 0.0]
        factors.append(float(np.median(nonzero)) if nonzero.size else 1.0)
    return np.asarray(factors, dtype=float)


def _project_existing_route_semantics(
    *,
    r3d_x: np.ndarray,
    wavelengths: np.ndarray,
    factors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if r3d_x.shape[0] != factors.size:
        raise ValueError("row factor count does not match spectra row count")
    support = _support_mask(wavelengths)
    r9e_x = r3d_x.copy()
    r9e_x[:, support] = r9e_x[:, support] * factors[:, None]
    p2a_x = r3d_x * factors[:, None]
    return r9e_x, p2a_x


def _row_from_arrays(
    *,
    case: str,
    seed: int,
    dataset: Any,
    preset: str,
    grid_source: str,
    grid_note: str,
    wavelengths: np.ndarray,
    r3d_x: np.ndarray,
    r9e_x: np.ndarray,
    p2a_x: np.ndarray,
    r9e_metadata: dict[str, Any] | None,
    p2a_metadata: dict[str, Any] | None,
    n_real_samples: int,
) -> Exp26Row:
    support = _support_mask(wavelengths)
    off_support = ~support
    support_count = int(support.sum())
    off_support_count = int(off_support.sum())
    n_wavelengths = int(wavelengths.size)
    support_weight = float(support_count / n_wavelengths) if n_wavelengths else 0.0
    off_support_weight = float(off_support_count / n_wavelengths) if n_wavelengths else 0.0

    r9e_delta = r9e_x - r3d_x
    p2a_delta = p2a_x - r3d_x
    r9e_p2a_delta = r9e_x - p2a_x
    r9e_p2a_inside = _max_abs_delta(r9e_p2a_delta, support)
    r9e_p2a_outside = _max_abs_delta(r9e_p2a_delta, off_support)
    r9e_outside = _max_abs_delta(r9e_delta, off_support)
    p2a_outside = _max_abs_delta(p2a_delta, off_support)

    r9e_params = _transform_params(r9e_metadata)
    p2a_params = _transform_params(p2a_metadata)
    r9e_factor_min, r9e_factor_max = _factor_min_max(
        r9e_params,
        "support_reference_attenuation",
    )
    p2a_factor_min, p2a_factor_max = _factor_min_max(
        p2a_params,
        "row_pathlength_reference",
    )

    outside_minus_inside = (
        None
        if r9e_p2a_outside is None or r9e_p2a_inside is None
        else float(r9e_p2a_outside - r9e_p2a_inside)
    )
    delta_outside_difference = (
        None
        if p2a_outside is None or r9e_outside is None
        else float(p2a_outside - r9e_outside)
    )
    distinguishable = bool(
        off_support_count > 0
        and r9e_p2a_outside is not None
        and r9e_p2a_outside > 0.0
        and (r9e_p2a_inside is None or r9e_p2a_inside <= 1e-12)
    )

    return Exp26Row(
        status="compared",
        case=case,
        seed=int(seed),
        source=dataset.source,
        task=dataset.task,
        dataset=f"{dataset.database_name}/{dataset.dataset}",
        synthetic_preset=preset,
        comparison_space=COMPARISON_SPACE,
        grid_source=grid_source,
        grid_note=grid_note,
        n_real_samples=int(n_real_samples),
        n_synthetic_samples=int(r3d_x.shape[0]),
        n_wavelengths=n_wavelengths,
        wavelength_min=None if n_wavelengths == 0 else float(wavelengths[0]),
        wavelength_max=None if n_wavelengths == 0 else float(wavelengths[-1]),
        support_low_nm=SUPPORT_LOW_NM,
        support_high_nm=SUPPORT_HIGH_NM,
        support_count=support_count,
        off_support_count=off_support_count,
        support_weight=support_weight,
        off_support_weight=off_support_weight,
        r9e_stage_support_nm=_stage_support_nm(r9e_params),
        p2a_stage_support_nm=_stage_support_nm(p2a_params),
        r9e_support_only=bool(r9e_params.get("support_reference_attenuation_only", True)),
        p2a_support_only=bool(p2a_params.get("row_pathlength_reference_support_only", False)),
        r9e_off_support_unchanged_metadata=cast(
            bool | None,
            r9e_params.get("support_reference_attenuation_off_support_unchanged"),
        ),
        p2a_off_support_unchanged_metadata=cast(
            bool | None,
            p2a_params.get("row_pathlength_reference_off_support_unchanged"),
        ),
        r9e_factor_min=r9e_factor_min,
        r9e_factor_max=r9e_factor_max,
        p2a_factor_min=p2a_factor_min,
        p2a_factor_max=p2a_factor_max,
        r9e_vs_r3d_max_abs_delta_inside=_max_abs_delta(r9e_delta, support),
        r9e_vs_r3d_max_abs_delta_outside=r9e_outside,
        p2a_vs_r3d_max_abs_delta_inside=_max_abs_delta(p2a_delta, support),
        p2a_vs_r3d_max_abs_delta_outside=p2a_outside,
        r9e_vs_p2a_max_abs_delta_inside=r9e_p2a_inside,
        r9e_vs_p2a_max_abs_delta_outside=r9e_p2a_outside,
        r9e_vs_p2a_outside_minus_inside=outside_minus_inside,
        r9e_vs_p2a_outside_to_inside_ratio=_ratio(r9e_p2a_outside, r9e_p2a_inside),
        p2a_minus_r9e_delta_outside_vs_r3d=delta_outside_difference,
        distinguishable_by_off_support=distinguishable,
        current_cohort_can_distinguish=case == REAL_ALIGNED_CASE and distinguishable,
        anti_leakage_flags_false=_anti_leakage_flags_false(),
        **_audit_fields(),
        blocked_reason="",
    )


def _blocked_rows(
    *,
    seed: int,
    dataset: Any,
    preset: str,
    blocked_reason: str,
) -> list[Exp26Row]:
    rows: list[Exp26Row] = []
    for case in EXP26_CASES:
        rows.append(
            Exp26Row(
                status="blocked",
                case=case,
                seed=int(seed),
                source=dataset.source,
                task=dataset.task,
                dataset=f"{dataset.database_name}/{dataset.dataset}",
                synthetic_preset=preset,
                comparison_space=COMPARISON_SPACE,
                grid_source="blocked",
                grid_note="blocked",
                n_real_samples=0,
                n_synthetic_samples=0,
                n_wavelengths=0,
                wavelength_min=None,
                wavelength_max=None,
                support_low_nm=SUPPORT_LOW_NM,
                support_high_nm=SUPPORT_HIGH_NM,
                support_count=0,
                off_support_count=0,
                support_weight=0.0,
                off_support_weight=0.0,
                r9e_stage_support_nm=f"{SUPPORT_LOW_NM:g}-{SUPPORT_HIGH_NM:g}",
                p2a_stage_support_nm="full_generated_row",
                r9e_support_only=True,
                p2a_support_only=False,
                r9e_off_support_unchanged_metadata=None,
                p2a_off_support_unchanged_metadata=None,
                r9e_factor_min=None,
                r9e_factor_max=None,
                p2a_factor_min=None,
                p2a_factor_max=None,
                r9e_vs_r3d_max_abs_delta_inside=None,
                r9e_vs_r3d_max_abs_delta_outside=None,
                p2a_vs_r3d_max_abs_delta_inside=None,
                p2a_vs_r3d_max_abs_delta_outside=None,
                r9e_vs_p2a_max_abs_delta_inside=None,
                r9e_vs_p2a_max_abs_delta_outside=None,
                r9e_vs_p2a_outside_minus_inside=None,
                r9e_vs_p2a_outside_to_inside_ratio=None,
                p2a_minus_r9e_delta_outside_vs_r3d=None,
                distinguishable_by_off_support=False,
                current_cohort_can_distinguish=False,
                anti_leakage_flags_false=_anti_leakage_flags_false(),
                **_audit_fields(),
                blocked_reason=blocked_reason,
            )
        )
    return rows


def run_audit(
    *,
    root: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_synthetic_samples: int = DEFAULT_N_SYNTHETIC_SAMPLES,
    max_real_samples: int = DEFAULT_MAX_REAL_SAMPLES,
    max_sentinel_datasets: int = DEFAULT_MAX_SENTINEL_DATASETS,
    sentinel_tokens: Sequence[str] | None = None,
) -> dict[str, Any]:
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("at least one seed must be provided")
    tokens = tuple(DEFAULT_SENTINEL_TOKENS) if sentinel_tokens is None else tuple(sentinel_tokens)
    real_datasets, _ = _exp25.discover_local_real_datasets(root)
    sentinel_candidates = _exp25._exp09._select_sentinel_datasets(real_datasets, tokens)
    selected = list(sentinel_candidates) if max_sentinel_datasets <= 0 else sentinel_candidates[:max_sentinel_datasets]
    rows: list[Exp26Row] = []
    if not selected:
        return {
            "status": "blocked_no_real_data",
            "rows": rows,
            "real_runnable_count": len(real_datasets),
            "real_sentinel_candidate_count": len(sentinel_candidates),
            "real_selected_count": 0,
            "sentinel_tokens": list(tokens),
            "seeds": list(seeds),
            "support_low_nm": SUPPORT_LOW_NM,
            "support_high_nm": SUPPORT_HIGH_NM,
        }

    for dataset in selected:
        preset = _exp25._exp09.select_synthetic_preset_for_dataset(dataset)
        try:
            real_x_raw, real_wl_raw = _exp25.load_real_spectra(dataset, root=root)
            if _exp25.is_index_fallback_grid(real_wl_raw):
                raise ValueError("wavelength_grid_unknown")
            sanitized_real, sanitized_wl, _, real_blocked = _exp25.sanitize_finite_spectra(
                real_x_raw,
                real_wl_raw,
                side="real",
            )
            if real_blocked is not None or sanitized_real is None or sanitized_wl is None:
                raise ValueError(f"non_finite_spectra: {real_blocked}")
        except Exception as exc:  # noqa: BLE001
            for seed in seeds:
                rows.extend(
                    _blocked_rows(
                        seed=seed,
                        dataset=dataset,
                        preset=preset,
                        blocked_reason=f"{type(exc).__name__}: {exc}",
                    )
                )
            continue

        for seed in seeds:
            try:
                real_x = _exp25._exp09._downsample_rows(
                    sanitized_real,
                    max_rows=max_real_samples,
                    random_state=_exp25._exp09._stable_dataset_seed(
                        seed,
                        dataset,
                        "exp26:real_downsample",
                    ),
                )
                runs = {
                    R3D_PROFILE: _exp25._build_synthetic_run(
                        dataset=dataset,
                        preset=preset,
                        n_samples=n_synthetic_samples,
                        seed=seed,
                        remediation_profile=R3D_PROFILE,
                    ),
                    R9E_PROFILE: _exp25._build_synthetic_run(
                        dataset=dataset,
                        preset=preset,
                        n_samples=n_synthetic_samples,
                        seed=seed,
                        remediation_profile=R9E_PROFILE,
                    ),
                    P2A_PROFILE: _exp25._build_synthetic_run(
                        dataset=dataset,
                        preset=preset,
                        n_samples=n_synthetic_samples,
                        seed=seed,
                        remediation_profile=P2A_PROFILE,
                    ),
                }
                sanitized_runs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                for profile, run in runs.items():
                    sanitized_x, sanitized_run_wl, _, syn_blocked = _exp25.sanitize_finite_spectra(
                        np.asarray(run.X, dtype=float),
                        np.asarray(run.wavelengths, dtype=float),
                        side=f"synthetic_{profile}",
                    )
                    if syn_blocked is not None or sanitized_x is None or sanitized_run_wl is None:
                        raise ValueError(f"non_finite_spectra_synthetic_{profile}: {syn_blocked}")
                    sanitized_runs[profile] = (sanitized_x, sanitized_run_wl)

                generated_wl = sanitized_runs[R3D_PROFILE][1]
                if not all(np.array_equal(generated_wl, sanitized_runs[profile][1]) for profile in (R9E_PROFILE, P2A_PROFILE)):
                    raise ValueError("profile_wavelength_grids_differ")

                downsampled: dict[str, np.ndarray] = {}
                for profile, (synthetic_x, _) in sanitized_runs.items():
                    downsampled[profile] = _exp25._exp09._downsample_rows(
                        synthetic_x,
                        max_rows=max_real_samples,
                        random_state=_exp25._exp09._stable_dataset_seed(
                            seed,
                            dataset,
                            "exp26:syn_downsample",
                        ),
                    )

                row_factors = _row_factors_from_support(
                    r3d_x=downsampled[R3D_PROFILE],
                    r9e_x=downsampled[R9E_PROFILE],
                    wavelengths=generated_wl,
                )
                aligned_r3d, aligned_wl = _aligned_to_grid(
                    real_x=real_x,
                    real_wl=np.asarray(sanitized_wl, dtype=float),
                    synthetic_x=downsampled[R3D_PROFILE],
                    synthetic_wl=generated_wl,
                )
                assert aligned_wl is not None
                aligned_r9e, aligned_p2a = _project_existing_route_semantics(
                    r3d_x=aligned_r3d,
                    wavelengths=aligned_wl,
                    factors=row_factors,
                )
                generated_r9e, generated_p2a = _project_existing_route_semantics(
                    r3d_x=downsampled[R3D_PROFILE],
                    wavelengths=generated_wl,
                    factors=row_factors,
                )

                rows.append(
                    _row_from_arrays(
                        case=REAL_ALIGNED_CASE,
                        seed=seed,
                        dataset=dataset,
                        preset=preset,
                        grid_source="real_wavelength_grid_after_alignment",
                        grid_note="current DIESEL real-aligned cohort grid with existing route factors projected on observed wavelengths",
                        wavelengths=aligned_wl,
                        r3d_x=aligned_r3d,
                        r9e_x=aligned_r9e,
                        p2a_x=aligned_p2a,
                        r9e_metadata=cast(dict[str, Any], runs[R9E_PROFILE].metadata),
                        p2a_metadata=cast(dict[str, Any], runs[P2A_PROFILE].metadata),
                        n_real_samples=int(real_x.shape[0]),
                    )
                )
                rows.append(
                    _row_from_arrays(
                        case=GENERATED_FULL_GRID_CASE,
                        seed=seed,
                        dataset=dataset,
                        preset=preset,
                        grid_source="synthetic_prior_generated_wavelength_grid",
                        grid_note="counterfactual audit uses the predeclared generated prior grid before real alignment with existing route factors",
                        wavelengths=generated_wl,
                        r3d_x=downsampled[R3D_PROFILE],
                        r9e_x=generated_r9e,
                        p2a_x=generated_p2a,
                        r9e_metadata=cast(dict[str, Any], runs[R9E_PROFILE].metadata),
                        p2a_metadata=cast(dict[str, Any], runs[P2A_PROFILE].metadata),
                        n_real_samples=0,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                rows.extend(
                    _blocked_rows(
                        seed=seed,
                        dataset=dataset,
                        preset=preset,
                        blocked_reason=f"{type(exc).__name__}: {exc}",
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
        "support_low_nm": SUPPORT_LOW_NM,
        "support_high_nm": SUPPORT_HIGH_NM,
    }


def _csv_fieldnames() -> list[str]:
    return [field.name for field in fields(Exp26Row)]


def write_csv(rows: list[Exp26Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_csv_fieldnames(), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.6g}"


def _case_summary(rows: Sequence[Exp26Row]) -> list[dict[str, Any]]:
    compared = [row for row in rows if row.status == "compared"]
    out: list[dict[str, Any]] = []
    for case in EXP26_CASES:
        bucket = [row for row in compared if row.case == case]
        if not bucket:
            continue
        out.append(
            {
                "case": case,
                "n": len(bucket),
                "min_off_support_count": min(row.off_support_count for row in bucket),
                "max_off_support_count": max(row.off_support_count for row in bucket),
                "max_inside": max(
                    (
                        row.r9e_vs_p2a_max_abs_delta_inside
                        for row in bucket
                        if row.r9e_vs_p2a_max_abs_delta_inside is not None
                    ),
                    default=None,
                ),
                "max_outside": max(
                    (
                        row.r9e_vs_p2a_max_abs_delta_outside
                        for row in bucket
                        if row.r9e_vs_p2a_max_abs_delta_outside is not None
                    ),
                    default=None,
                ),
                "distinguishable_count": sum(row.distinguishable_by_off_support for row in bucket),
                "current_can_distinguish_count": sum(row.current_cohort_can_distinguish for row in bucket),
                "anti_leakage_all_false": all(row.anti_leakage_flags_false for row in bucket),
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
) -> str:
    rows = cast(list[Exp26Row], result["rows"])
    compared = [row for row in rows if row.status == "compared"]
    blocked = [row for row in rows if row.status != "compared"]
    real_rows = [row for row in compared if row.case == REAL_ALIGNED_CASE]
    generated_rows = [row for row in compared if row.case == GENERATED_FULL_GRID_CASE]
    current_can_distinguish = any(row.current_cohort_can_distinguish for row in real_rows)
    generated_can_distinguish = any(row.distinguishable_by_off_support for row in generated_rows)
    lines = [
        "# P2-03 DIESEL Support/Off-Support Discriminability Audit",
        "",
        f"- audit_scope: `{EXP26_AUDIT_SCOPE}`",
        f"- decision: `{EXP26_DECISION}`",
        f"- comparison_space: `{COMPARISON_SPACE}`",
        f"- report: `{report_path}`",
        f"- csv: `{csv_path}`",
        f"- rows: compared={len(compared)}, blocked={len(blocked)}",
        f"- seeds: `{','.join(str(seed) for seed in seeds)}`",
        f"- n_synthetic_samples: `{n_synthetic_samples}`",
        f"- max_real_samples: `{max_real_samples}`",
        f"- max_sentinel_datasets: `{max_sentinel_datasets}`",
        f"- sentinel_tokens: `{','.join(sentinel_tokens)}`",
        "",
        "## Contract",
        "",
        "- R3d remains the accepted DIESEL baseline.",
        "- R9e and P2a are existing diagnostic-only profiles; this audit creates no R9n, new profile, mechanism, promotion, or gate.",
        "- No calibration, real-stat capture, PCA/noise capture, ML/DL, labels, targets, splits, threshold mutation, metric mutation, or production integration is used.",
        "- The counterfactual case uses the existing generated prior wavelength grid before audit-side real alignment.",
        "",
        "## Decision",
        "",
        f"- current_real_aligned_cohort_can_distinguish: `{current_can_distinguish}`",
        f"- generated_full_grid_can_distinguish: `{generated_can_distinguish}`",
        "- interpretation: current aligned DIESEL rows cannot distinguish R9e support-only from P2a full-row when off-support count is zero; the existing generated prior grid can distinguish them because P2a attenuates off-support wavelengths and R9e leaves them unchanged.",
        "",
        "## Case Summary",
        "",
        "| case | n | off-support count range | max abs R9e-P2a inside | max abs R9e-P2a outside | distinguishable | current can distinguish | anti-leakage false |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in _case_summary(rows):
        lines.append(
            "| {case} | {n} | {off_min}-{off_max} | {inside} | {outside} | {dist} | {current} | {leak} |".format(
                case=row["case"],
                n=row["n"],
                off_min=row["min_off_support_count"],
                off_max=row["max_off_support_count"],
                inside=_fmt(row["max_inside"]),
                outside=_fmt(row["max_outside"]),
                dist=row["distinguishable_count"],
                current=row["current_can_distinguish_count"],
                leak=row["anti_leakage_all_false"],
            )
        )
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "python bench/nirs_synthetic_pfn/experiments/exp26_diesel_support_offsupport_discriminability_audit.py \\",
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
    args = parser.parse_args()

    sentinel_tokens = tuple(args.sentinel_tokens or DEFAULT_SENTINEL_TOKENS)
    result = run_audit(
        root=args.root,
        seeds=args.seeds,
        n_synthetic_samples=args.n_synthetic_samples,
        max_real_samples=args.max_real_samples,
        max_sentinel_datasets=args.max_sentinel_datasets,
        sentinel_tokens=sentinel_tokens,
    )
    rows = cast(list[Exp26Row], result["rows"])
    write_csv(rows, args.csv)
    markdown = render_markdown(
        result=result,
        report_path=args.report,
        csv_path=args.csv,
        n_synthetic_samples=args.n_synthetic_samples,
        max_real_samples=args.max_real_samples,
        max_sentinel_datasets=args.max_sentinel_datasets,
        seeds=args.seeds,
        sentinel_tokens=sentinel_tokens,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
