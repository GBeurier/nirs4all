"""P2-05 DIESEL readout/geometry mechanistic audit.

Bench-only read-only audit for fixed analytical readout transforms and
available instrument/geometry metadata. The audit applies predeclared monotone
readout maps to existing R3d/P2a synthetic outputs; it does not add a profile,
change the builder, tune a factor, promote a route, or mutate any metric/gate.
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

EXP27_AUDIT_SCOPE = "bench_only_p2_05_diesel_readout_geometry_audit"
EXP27_DECISION = "report_only_no_profile_no_builder_change"
COMPARISON_SPACE = "uncalibrated_raw"
SUPPORT_LOW_NM: float = _exp25.SUPPORT_LOW_NM
SUPPORT_HIGH_NM: float = _exp25.SUPPORT_HIGH_NM

R3D_PROFILE = _exp25.R3D_PROFILE
P2A_PROFILE = _exp25.P2A_PROFILE

READOUT_IDENTITY = "absorbance_identity"
READOUT_TRANSMITTANCE = "transmittance_10_neg_absorbance"
READOUT_BLANK_INTENSITY = "blank_referenced_intensity_1_minus_10_neg_absorbance"
READOUT_TRANSFORMS: tuple[str, ...] = (
    READOUT_IDENTITY,
    READOUT_TRANSMITTANCE,
    READOUT_BLANK_INTENSITY,
)
SOURCE_PROFILES: tuple[str, ...] = (R3D_PROFILE, P2A_PROFILE)

DEFAULT_SEEDS: tuple[int, ...] = (20260501, 20260502, 20260503)
DEFAULT_N_SYNTHETIC_SAMPLES = 64
DEFAULT_MAX_REAL_SAMPLES = 64
DEFAULT_MAX_SENTINEL_DATASETS = 8
DEFAULT_SENTINEL_TOKENS: tuple[str, ...] = ("DIESEL",)
DEFAULT_REPORT = Path("/tmp/exp27_diesel_readout_geometry_audit.md")
DEFAULT_CSV = Path("/tmp/exp27_diesel_readout_geometry_audit.csv")


@dataclass
class Exp27Row:
    status: str
    seed: int
    source_profile: str
    readout_transform: str
    source: str
    task: str
    dataset: str
    synthetic_preset: str
    comparison_space: str
    transform_stage: str
    transform_parameters_source: str
    transform_monotone: bool
    transform_calibrated: bool
    geometry_metadata_present: bool
    geometry_audit_status: str
    geometry_metadata_keys: str
    instrument_key: str
    measurement_mode: str
    metadata_readout_space: str
    metadata_readout_transform: str
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
    readout_delta_vs_identity__global_mean_delta: float | None
    readout_delta_vs_identity__support_mean_delta: float | None
    readout_delta_vs_identity__morphology_gap_score: float | None
    readout_delta_vs_identity__log10_derivative_std_p50_ratio: float | None
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
    fields_dict["audit_scope"] = EXP27_AUDIT_SCOPE
    return fields_dict


def _anti_leakage_flags_false() -> bool:
    return all(value is False for key, value in _audit_fields().items() if key != "audit_scope")


def apply_readout_transform(x: np.ndarray, transform: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if transform == READOUT_IDENTITY:
        return x.copy()
    if transform == READOUT_TRANSMITTANCE:
        return cast(np.ndarray, np.clip(np.power(10.0, -x), 0.0, 1.0))
    if transform == READOUT_BLANK_INTENSITY:
        return cast(np.ndarray, np.clip(1.0 - np.power(10.0, -x), 0.0, 1.0))
    raise ValueError(f"unknown readout transform {transform!r}")


def _metadata_summary(metadata: dict[str, Any] | None) -> dict[str, Any]:
    metadata = {} if metadata is None else metadata
    instrument = cast(dict[str, Any], metadata.get("instrument", {}))
    transform_params = cast(dict[str, Any], _exp25._transform_params(metadata))
    builder_config = cast(dict[str, Any], metadata.get("builder_config", {}))
    features = cast(dict[str, Any], builder_config.get("features", {}))
    geometry_keys = sorted(
        key
        for key in transform_params
        if "geometry" in key or "source_detector" in key or "source-detector" in key
    )
    return {
        "geometry_metadata_present": bool(geometry_keys),
        "geometry_metadata_keys": ";".join(geometry_keys),
        "instrument_key": str(instrument.get("key", features.get("instrument", ""))),
        "measurement_mode": str(
            metadata.get("mode", instrument.get("measurement_mode", features.get("measurement_mode", "")))
        ),
        "metadata_readout_space": str(transform_params.get("readout_space", "")),
        "metadata_readout_transform": str(transform_params.get("readout_space_transform", "")),
    }


def _metric_fields(real_x: np.ndarray, synthetic_x: np.ndarray, wavelengths: np.ndarray) -> dict[str, Any]:
    morphology = _exp25._exp09.compute_morphology_metrics(real_x, synthetic_x, wavelengths)
    decomposition = _exp25._exp10.compute_support_decomposition(
        real_x,
        synthetic_x,
        wavelengths,
        support_low_nm=SUPPORT_LOW_NM,
        support_high_nm=SUPPORT_HIGH_NM,
    )
    return {
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
        **_exp25._exp15._morphology_subset(morphology),
    }


def _row_from_arrays(
    *,
    seed: int,
    source_profile: str,
    readout_transform: str,
    dataset: Any,
    preset: str,
    real_x: np.ndarray,
    synthetic_x: np.ndarray,
    wavelengths: np.ndarray,
    metadata: dict[str, Any] | None,
    identity_metrics: dict[str, Any] | None,
) -> Exp27Row:
    transformed = apply_readout_transform(synthetic_x, readout_transform)
    metrics = _metric_fields(real_x, transformed, wavelengths)
    metadata_fields = _metadata_summary(metadata)
    identity = metrics if identity_metrics is None else identity_metrics

    def delta(attr: str) -> float | None:
        lhs = metrics.get(attr)
        rhs = identity.get(attr)
        return None if lhs is None or rhs is None else float(lhs) - float(rhs)

    return Exp27Row(
        status="compared",
        seed=int(seed),
        source_profile=source_profile,
        readout_transform=readout_transform,
        source=dataset.source,
        task=dataset.task,
        dataset=f"{dataset.database_name}/{dataset.dataset}",
        synthetic_preset=preset,
        comparison_space=COMPARISON_SPACE,
        transform_stage="report_only_after_real_grid_alignment",
        transform_parameters_source="predeclared_beer_lambert_readout_maps_no_fit",
        transform_monotone=True,
        transform_calibrated=False,
        geometry_audit_status=(
            "available_metadata_only"
            if metadata_fields["geometry_metadata_present"]
            else "blocked_no_source_detector_geometry_metadata"
        ),
        n_real_samples=int(real_x.shape[0]),
        n_synthetic_samples=int(synthetic_x.shape[0]),
        n_wavelengths=int(wavelengths.size),
        wavelength_min=None if wavelengths.size == 0 else float(wavelengths[0]),
        wavelength_max=None if wavelengths.size == 0 else float(wavelengths[-1]),
        readout_delta_vs_identity__global_mean_delta=delta("global_mean_delta"),
        readout_delta_vs_identity__support_mean_delta=delta("support_mean_delta"),
        readout_delta_vs_identity__morphology_gap_score=delta("morphology_gap_score"),
        readout_delta_vs_identity__log10_derivative_std_p50_ratio=delta(
            "log10_derivative_std_p50_ratio"
        ),
        **metadata_fields,
        **metrics,
        **_audit_fields(),
        blocked_reason="",
    )


def _blocked_rows(
    *,
    seed: int,
    dataset: Any,
    preset: str,
    blocked_reason: str,
) -> list[Exp27Row]:
    rows: list[Exp27Row] = []
    empty_metrics = cast(dict[str, Any], _exp25._empty_metric_fields())
    for source_profile in SOURCE_PROFILES:
        for transform in READOUT_TRANSFORMS:
            rows.append(
                Exp27Row(
                    status="blocked",
                    seed=int(seed),
                    source_profile=source_profile,
                    readout_transform=transform,
                    source=dataset.source,
                    task=dataset.task,
                    dataset=f"{dataset.database_name}/{dataset.dataset}",
                    synthetic_preset=preset,
                    comparison_space=COMPARISON_SPACE,
                    transform_stage="report_only_after_real_grid_alignment",
                    transform_parameters_source="predeclared_beer_lambert_readout_maps_no_fit",
                    transform_monotone=True,
                    transform_calibrated=False,
                    geometry_metadata_present=False,
                    geometry_audit_status="blocked",
                    geometry_metadata_keys="",
                    instrument_key="",
                    measurement_mode="",
                    metadata_readout_space="",
                    metadata_readout_transform="",
                    n_real_samples=0,
                    n_synthetic_samples=0,
                    n_wavelengths=0,
                    wavelength_min=None,
                    wavelength_max=None,
                    **empty_metrics,
                    readout_delta_vs_identity__global_mean_delta=None,
                    readout_delta_vs_identity__support_mean_delta=None,
                    readout_delta_vs_identity__morphology_gap_score=None,
                    readout_delta_vs_identity__log10_derivative_std_p50_ratio=None,
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
    selected = (
        list(sentinel_candidates)
        if max_sentinel_datasets <= 0
        else sentinel_candidates[:max_sentinel_datasets]
    )
    rows: list[Exp27Row] = []
    if not selected:
        return {
            "status": "blocked_no_real_data",
            "rows": rows,
            "real_runnable_count": len(real_datasets),
            "real_sentinel_candidate_count": len(sentinel_candidates),
            "real_selected_count": 0,
            "sentinel_tokens": list(tokens),
            "seeds": list(seeds),
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
                        "exp27:real_downsample",
                    ),
                )
                identity_by_profile: dict[str, dict[str, Any]] = {}
                aligned_by_profile: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, Any]] = {}
                for source_profile in SOURCE_PROFILES:
                    run = _exp25._build_synthetic_run(
                        dataset=dataset,
                        preset=preset,
                        n_samples=n_synthetic_samples,
                        seed=seed,
                        remediation_profile=source_profile,
                    )
                    synthetic_x, synthetic_wl, _, syn_blocked = _exp25.sanitize_finite_spectra(
                        np.asarray(run.X, dtype=float),
                        np.asarray(run.wavelengths, dtype=float),
                        side=f"synthetic_{source_profile}",
                    )
                    if syn_blocked is not None or synthetic_x is None or synthetic_wl is None:
                        raise ValueError(f"non_finite_spectra_synthetic: {syn_blocked}")
                    synthetic_downsampled = _exp25._exp09._downsample_rows(
                        synthetic_x,
                        max_rows=max_real_samples,
                        random_state=_exp25._exp09._stable_dataset_seed(
                            seed,
                            dataset,
                            "exp27:syn_downsample",
                        ),
                    )
                    real_aligned, synthetic_aligned, aligned_wl = _exp25.align_to_real_grid(
                        real_x,
                        np.asarray(sanitized_wl, dtype=float),
                        synthetic_downsampled,
                        np.asarray(synthetic_wl, dtype=float),
                    )
                    aligned_by_profile[source_profile] = (
                        real_aligned,
                        synthetic_aligned,
                        aligned_wl,
                        run,
                    )
                    identity_by_profile[source_profile] = _metric_fields(
                        real_aligned,
                        synthetic_aligned,
                        aligned_wl,
                    )

                for source_profile, (real_aligned, synthetic_aligned, aligned_wl, run) in (
                    aligned_by_profile.items()
                ):
                    for transform in READOUT_TRANSFORMS:
                        rows.append(
                            _row_from_arrays(
                                seed=seed,
                                source_profile=source_profile,
                                readout_transform=transform,
                                dataset=dataset,
                                preset=preset,
                                real_x=real_aligned,
                                synthetic_x=synthetic_aligned,
                                wavelengths=aligned_wl,
                                metadata=cast(dict[str, Any], run.metadata),
                                identity_metrics=identity_by_profile[source_profile],
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
    }


def _csv_fieldnames() -> list[str]:
    return [field.name for field in fields(Exp27Row)]


def write_csv(rows: list[Exp27Row], path: Path) -> None:
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


def _summary(rows: Sequence[Exp27Row]) -> list[dict[str, Any]]:
    compared = [row for row in rows if row.status == "compared"]
    out: list[dict[str, Any]] = []
    for source_profile in SOURCE_PROFILES:
        for transform in READOUT_TRANSFORMS:
            bucket = [
                row
                for row in compared
                if row.source_profile == source_profile and row.readout_transform == transform
            ]
            if not bucket:
                continue
            out.append(
                {
                    "source_profile": source_profile,
                    "readout_transform": transform,
                    "n": len(bucket),
                    "median_gap": float(np.median([row.morphology_gap_score for row in bucket])),
                    "median_global": float(np.median([row.global_mean_delta for row in bucket])),
                    "median_support": float(np.median([row.support_mean_delta for row in bucket])),
                    "median_derivative": float(
                        np.median([row.log10_derivative_std_p50_ratio for row in bucket])
                    ),
                    "median_delta_gap": float(
                        np.median(
                            [
                                row.readout_delta_vs_identity__morphology_gap_score
                                for row in bucket
                            ]
                        )
                    ),
                    "geometry_present_count": sum(row.geometry_metadata_present for row in bucket),
                    "dominant_gaps": Counter(row.dominant_morphology_gap for row in bucket),
                    "anti_leakage_all_false": _anti_leakage_flags_false(),
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
    rows = cast(list[Exp27Row], result["rows"])
    compared = [row for row in rows if row.status == "compared"]
    blocked = [row for row in rows if row.status != "compared"]
    geometry_present = any(row.geometry_metadata_present for row in compared)
    lines = [
        "# P2-05 DIESEL Readout/Geometry Mechanistic Audit",
        "",
        f"- audit_scope: `{EXP27_AUDIT_SCOPE}`",
        f"- decision: `{EXP27_DECISION}`",
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
        "- This audit creates no profile, no R9n, no promotion, no gate, and no builder or `nirs4all/` change.",
        "- Readout transforms are fixed analytic maps: identity absorbance, `10**-A` transmittance, and `1 - 10**-A` blank-referenced intensity.",
        "- No geometry scalar is tested because that would duplicate R9e/P2a global/pathlength factors without source-detector metadata.",
        "- No calibration, real-stat capture, PCA/noise capture, ML/DL, labels, targets, splits, threshold mutation, metric mutation, or downstream feedback is used.",
        "",
        "## Decision",
        "",
        f"- source_detector_geometry_metadata_present: `{geometry_present}`",
        "- geometry_result: source-detector geometry is not auditable as a mechanistic transform in the current route; no predeclared distance/angle/collection metadata is available.",
        "- readout_result: fixed monotone readout maps are report-only counterfactual diagnostics in uncalibrated raw comparison space; deltas vs identity are not promotion evidence, and the observed non-identity maps degrade relative to identity.",
        "",
        "## Summary",
        "",
        "| source profile | readout transform | n | median gap | median global | median support | median derivative | median gap delta vs identity | geometry metadata rows | dominant gaps | anti-leakage false |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in _summary(rows):
        lines.append(
            "| {profile} | {transform} | {n} | {gap} | {global_} | {support} | {deriv} | {delta_gap} | {geom} | {gaps} | {leak} |".format(
                profile=row["source_profile"],
                transform=row["readout_transform"],
                n=row["n"],
                gap=_fmt(row["median_gap"]),
                global_=_fmt(row["median_global"]),
                support=_fmt(row["median_support"]),
                deriv=_fmt(row["median_derivative"]),
                delta_gap=_fmt(row["median_delta_gap"]),
                geom=row["geometry_present_count"],
                gaps=dict(row["dominant_gaps"]),
                leak=row["anti_leakage_all_false"],
            )
        )
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "python bench/nirs_synthetic_pfn/experiments/exp27_diesel_readout_geometry_audit.py \\",
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
    rows = cast(list[Exp27Row], result["rows"])
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
