"""Smoke experiment for Phase A3 real-fit adapter."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
from nirsyntheticpfn.adapters.builder_adapter import SyntheticDatasetRun, build_synthetic_dataset_run
from nirsyntheticpfn.adapters.fitted_config_adapter import (
    FittedConfigAdapterError,
    build_dataset_run_from_fitted_config,
)
from nirsyntheticpfn.adapters.fitted_residual_effects import fit_observable_residual_effects
from nirsyntheticpfn.adapters.prior_adapter import canonicalize_domain, canonicalize_prior_config

from nirs4all.synthesis.components import get_component
from nirs4all.synthesis.domains import get_domain_config
from nirs4all.synthesis.fitter import RealDataFitter
from nirs4all.synthesis.validation import (
    compute_baseline_curvature,
    compute_correlation_length,
    compute_derivative_statistics,
    compute_distribution_overlap,
    compute_peak_density,
    compute_spectral_realism_scorecard,
)

DEFAULT_OUTPUT = Path("bench/nirs_synthetic_pfn/reports/real_fit_adapter_smoke.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--source-seed", type=int, default=20260429)
    parser.add_argument("--regen-seed", type=int, default=20260430)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = run_smoke(
        n_samples=args.n_samples,
        source_seed=args.source_seed,
        regen_seed=args.regen_seed,
    )
    git_status = _git_status_summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_markdown(result, args.n_samples, args.source_seed, args.regen_seed, git_status),
        encoding="utf-8",
    )
    print(args.output)


def run_smoke(*, n_samples: int, source_seed: int, regen_seed: int) -> dict[str, Any]:
    source_record = canonicalize_prior_config(_source_prior(source_seed))
    source_run = build_synthetic_dataset_run(
        source_record,
        n_samples=n_samples,
        random_seed=source_seed,
    )
    fitted_config = RealDataFitter().fit(
        source_run.X,
        wavelengths=source_run.wavelengths,
        name="a3_synthetic_real_like_source",
    ).to_full_config()
    fitted_only_config = dict(fitted_config)
    fitted_only_config["fitted_residual_effects_config"] = fit_observable_residual_effects(
        source_run.X,
        source_run.wavelengths,
        fitted_only_config,
    )

    fitted_instrument = str(fitted_config.get("instrument", ""))
    variants = [
        _evaluate_variant(
            variant_name="fitted_instrument",
            source_run=source_run,
            fitted_config=fitted_only_config,
            n_samples=n_samples,
            regen_seed=regen_seed,
            instrument_override=None,
            instrument_override_reason=None,
            component_override=None,
            component_override_reason=None,
        )
    ]
    source_instrument = str(source_run.metadata["instrument"]["key"])
    source_components = [str(component) for component in source_run.latent_metadata["component_keys"]]
    fitted_components = [str(component) for component in fitted_config.get("components", [])]
    if source_components != fitted_components:
        component_reason = (
            "Synthetic smoke source provenance is known; this oracle diagnostic isolates "
            f"RealDataFitter component recall {fitted_components!r} vs source {source_components!r}. "
            "This is not a fitted-only executable contract."
        )
        variants.append(
            _evaluate_variant(
                variant_name="source_components_override",
                source_run=source_run,
                fitted_config=fitted_config,
                n_samples=n_samples,
                regen_seed=regen_seed,
                instrument_override=None,
                instrument_override_reason=None,
                component_override=source_components,
                component_override_reason=component_reason,
            )
        )
    if source_instrument and source_instrument != fitted_instrument:
        variants.append(
            _evaluate_variant(
                variant_name="source_instrument_override",
                source_run=source_run,
                fitted_config=fitted_config,
                n_samples=n_samples,
                regen_seed=regen_seed,
                instrument_override=source_instrument,
                instrument_override_reason=(
                    "Synthetic smoke source provenance is known; this diagnostic isolates "
                    f"RealDataFitter instrument inference {fitted_instrument!r} vs source {source_instrument!r}."
                ),
                component_override=None,
                component_override_reason=None,
            )
        )
        if source_components != fitted_components:
            variants.append(
                _evaluate_variant(
                    variant_name="source_instrument_and_components_override",
                    source_run=source_run,
                    fitted_config=fitted_config,
                    n_samples=n_samples,
                    regen_seed=regen_seed,
                    instrument_override=source_instrument,
                    instrument_override_reason=(
                        "Synthetic smoke source provenance is known; this diagnostic isolates "
                        f"RealDataFitter instrument inference {fitted_instrument!r} vs source {source_instrument!r}."
                    ),
                    component_override=source_components,
                    component_override_reason=component_reason,
                )
            )

    primary = _primary_variant(variants)
    if primary["status"] != "passed":
        return {
            "status": "failed",
            "contract_status": "failed",
            "scientific_status": "not_run",
            "source": _run_summary(source_run),
            "fitted_config": _to_builtin(fitted_only_config),
            "validation_summary": primary["validation_summary"],
            "ablations": variants,
            "correlation_length_sensitivity": _correlation_length_sensitivity(
                source_run=source_run,
                fitted_config=fitted_config,
                n_samples=n_samples,
                regen_seed=regen_seed,
            ),
            "candidate_mapping_diagnostics": _candidate_mapping_diagnostics(
                source_run=source_run,
                fitted_config=fitted_config,
                n_samples=n_samples,
                regen_seed=regen_seed,
            ),
        }

    return {
        **primary,
        "source": _run_summary(source_run),
        "fitted_config": _to_builtin(primary.get("fitted_config", fitted_only_config)),
        "ablations": variants,
        "correlation_length_sensitivity": _correlation_length_sensitivity(
            source_run=source_run,
            fitted_config=fitted_config,
            n_samples=n_samples,
            regen_seed=regen_seed,
        ),
        "candidate_mapping_diagnostics": _candidate_mapping_diagnostics(
            source_run=source_run,
            fitted_config=fitted_config,
            n_samples=n_samples,
            regen_seed=regen_seed,
        ),
    }


def _evaluate_variant(
    *,
    variant_name: str,
    source_run: SyntheticDatasetRun,
    fitted_config: dict[str, Any],
    n_samples: int,
    regen_seed: int,
    instrument_override: str | None,
    instrument_override_reason: str | None,
    component_override: list[str] | None,
    component_override_reason: str | None,
) -> dict[str, Any]:
    try:
        regenerated = build_dataset_run_from_fitted_config(
            fitted_config,
            n_samples=n_samples,
            random_seed=regen_seed,
            instrument_override=instrument_override,
            instrument_override_reason=instrument_override_reason,
            component_override=component_override,
            component_override_reason=component_override_reason,
        )
    except FittedConfigAdapterError as exc:
        return {
            "variant_name": variant_name,
            "status": "failed",
            "contract_status": "failed",
            "scientific_status": "not_run",
            "instrument_override": instrument_override,
            "instrument_override_reason": instrument_override_reason,
            "component_override": component_override,
            "component_override_reason": component_override_reason,
            "diagnostic_scope": _diagnostic_scope(instrument_override, component_override),
            "validation_summary": exc.validation_summary,
        }

    scorecard_inputs = _scorecard_inputs(source_run, regenerated)
    scorecard = compute_spectral_realism_scorecard(
        scorecard_inputs["source_X"],
        scorecard_inputs["regenerated_X"],
        scorecard_inputs["wavelengths"],
        include_adversarial=False,
        random_state=regen_seed,
    )
    scientific_status = "passed" if bool(scorecard.overall_pass) else "failed"
    return {
        "variant_name": variant_name,
        "status": "passed",
        "contract_status": "passed",
        "scientific_status": scientific_status,
        "regenerated": _run_summary(regenerated),
        "instrument_override": instrument_override,
        "instrument_override_reason": instrument_override_reason,
        "component_override": component_override,
        "component_override_reason": component_override_reason,
        "diagnostic_scope": _diagnostic_scope(instrument_override, component_override),
        "fitted_only_executable_contract": instrument_override is None and component_override is None,
        "fitted_config": _to_builtin(fitted_config),
        "unsupported_fields": regenerated.validation_summary["unsupported_fields"],
        "adapter_assumptions": regenerated.validation_summary["adapter_assumptions"],
        "effect_reconstruction": regenerated.builder_config.get(
            "fitted_config_mapping",
            {},
        ).get("effect_reconstruction", {}),
        "scorecard": _scorecard_to_dict(scorecard),
        "comparison": _comparison_summary(source_run, regenerated),
        "wavelength_grid_diagnostic": scorecard_inputs["wavelength_grid_diagnostic"],
        "correlation_length_diagnostic": _correlation_length_diagnostic(
            scorecard_inputs["source_X"],
            scorecard_inputs["regenerated_X"],
        ),
        "remaining_metric_diagnostics": _remaining_metric_diagnostics(
            scorecard_inputs["source_X"],
            scorecard_inputs["regenerated_X"],
            scorecard_inputs["wavelengths"],
            scorecard=_scorecard_to_dict(scorecard),
        ),
    }


def _primary_variant(variants: list[dict[str, Any]]) -> dict[str, Any]:
    for variant in variants:
        if variant["variant_name"] == "fitted_instrument":
            return variant
    return variants[0]


def _diagnostic_scope(instrument_override: str | None, component_override: list[str] | None) -> str:
    if instrument_override is None and component_override is None:
        return "fitted_only"
    if component_override is not None:
        return "oracle_source_provenance"
    return "source_provenance_instrument"


def render_markdown(
    result: dict[str, Any],
    n_samples: int,
    source_seed: int,
    regen_seed: int,
    git_status: dict[str, Any],
) -> str:
    command = (
        "PYTHONPATH=bench/nirs_synthetic_pfn/src "
        "python bench/nirs_synthetic_pfn/experiments/exp00_real_fit_adapter.py "
        f"--n-samples {n_samples} --source-seed {source_seed} --regen-seed {regen_seed}"
    )
    lines = [
        "# Real-Fit Adapter Smoke",
        "",
        "## Objective",
        "",
        "Fit a synthetic-real-like A2 dataset with `RealDataFitter.to_full_config()`, regenerate through the A3 fitted-config adapter, and compare spectra.",
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
        "## Config",
        "",
        f"- Samples: {n_samples}",
        f"- Source seed: {source_seed}",
        f"- Regeneration seed: {regen_seed}",
        "- Source preset: agriculture grain, FOSS XDS, reflectance, regression target",
        f"- Fitted-only gate variant: `{result.get('variant_name', 'fitted_instrument')}`",
        "",
        "## Git Status",
        "",
        _git_status_section(git_status),
        "",
        "## Dataset Summary",
        "",
        "| run | domain | instrument | mode | target | X shape | y shape | wavelength range | status |",
        "|---|---|---|---|---|---:|---:|---|---|",
        _summary_row("source", result["source"]),
    ]
    if result["status"] == "passed":
        lines.append(_summary_row("regenerated", result["regenerated"]))
    else:
        failures = result["validation_summary"].get("failures", [])
        lines.append(f"| regenerated | _failed_ | _failed_ | _failed_ | _failed_ | _failed_ | _failed_ | _failed_ | `{failures}` |")

    lines.extend([
        "",
        "## Component Diagnostic",
        "",
        f"- Source components: `{result['source'].get('components', [])}`",
        f"- Fitted components: `{result.get('fitted_config', {}).get('components', [])}`",
    ])
    if result["status"] == "passed":
        lines.append(f"- Regenerated components: `{result['regenerated'].get('components', [])}`")
    lines.append(
        "- Source components absent from fitted config: "
        f"`{_component_gap(result.get('source', {}), result.get('fitted_config', {}))}`"
    )

    lines.extend([
        "",
        "## Diagnostic Ablations",
        "",
    ])
    ablations = result.get("ablations", [])
    if ablations:
        lines.extend([
            "| variant | scope | effective instrument | effective components | contract | scientific | failed metrics | mean MAE | global gap | override reason |",
            "|---|---|---|---|---|---|---|---:|---:|---|",
        ])
        for variant in ablations:
            lines.append(_ablation_row(variant))
    else:
        lines.append("No ablations were run.")

    lines.extend([
        "",
        "## Correlation-Length Diagnostic",
        "",
        _correlation_length_section(result),
        "",
        "## Remaining Metric Diagnostics",
        "",
        _remaining_metric_section(result),
        "",
        "## Candidate Mapping Diagnostics",
        "",
        _candidate_mapping_section(result),
        "",
        "## Final Unblock Memo",
        "",
        _final_unblock_memo(result),
        "",
        "## Fitted Nuisance/Effect Reconstruction",
        "",
        _effect_reconstruction_section(result),
        "",
        "## Contract Checks",
        "",
        f"- Contract status: `{result.get('contract_status', result['status'])}`",
    ])
    if result["status"] == "passed":
        checks = result["regenerated"]["validation_summary"]["checks"]
        lines.extend([
            f"- Shape: `{checks['shape']}`",
            f"- Finite: `{checks['finite']}`",
            f"- Wavelengths monotonic: `{checks['wavelengths_monotonic']}`",
            f"- Target contract: `{checks['target_contract']}`",
        ])

    lines.extend([
        "",
        "## Unsupported Fields",
        "",
    ])
    unsupported_fields = result.get("unsupported_fields") or result.get("validation_summary", {}).get("unsupported_fields", [])
    if unsupported_fields:
        lines.extend([
            "| field | reason | value summary |",
            "|---|---|---|",
        ])
        for field in unsupported_fields:
            lines.append(
                f"| `{field['field']}` | {field['reason']} | `{field['value_summary']}` |"
            )
    else:
        lines.append("None.")

    lines.extend([
        "",
        "## Metrics",
        "",
    ])
    if result["status"] == "passed":
        comparison = result["comparison"]
        scorecard = result["scorecard"]
        lines.extend([
            f"- Scientific similarity status: `{result['scientific_status']}`",
            f"- Mean spectrum MAE: {comparison['mean_spectrum_mae']:.6g}",
            f"- Mean spectrum RMSE: {comparison['mean_spectrum_rmse']:.6g}",
            f"- Mean absolute global mean gap: {comparison['global_mean_abs_gap']:.6g}",
            f"- Realism overall pass: `{scorecard['overall_pass']}`",
            "",
            "| metric | value | threshold | passed |",
            "|---|---:|---:|---|",
        ])
        for metric in scorecard["metric_results"]:
            lines.append(
                f"| `{metric['metric']}` | {metric['value']:.6g} | "
                f"{metric['threshold']:.6g} | `{metric['passed']}` |"
            )
    else:
        lines.append("No metrics because regeneration failed.")

    lines.extend([
        "",
        "## Assumptions",
        "",
    ])
    for assumption in result.get("adapter_assumptions") or result.get("validation_summary", {}).get("adapter_assumptions", []):
        lines.append(f"- {assumption}")

    lines.extend([
        "",
        "## Failures",
        "",
        json.dumps(_failure_summary(result), indent=2, sort_keys=True),
        "",
        "## Gate Outcome",
        "",
        _gate_outcome(result),
        "",
        "## Raw Summary JSON",
        "",
        "```json",
        json.dumps(_to_builtin({"result": result, "git_status": git_status}), indent=2, sort_keys=True),
        "```",
        "",
        "## Decision",
        "",
        _decision(result),
        "",
    ])
    return "\n".join(lines)


def _source_prior(seed: int) -> dict[str, object]:
    domain_key = canonicalize_domain("grain")
    return {
        "domain": "grain",
        "domain_category": "research",
        "instrument": "foss_xds",
        "instrument_category": "benchtop",
        "wavelength_range": (400, 2500),
        "spectral_resolution": 4.0,
        "measurement_mode": "reflectance",
        "matrix_type": "solid",
        "temperature": 25.0,
        "particle_size": 150.0,
        "noise_level": 1.0,
        "components": _first_valid_domain_components(domain_key, 3),
        "n_samples": 100,
        "target_config": {"type": "regression", "n_targets": 1, "nonlinearity": "none"},
        "random_state": seed,
    }


def _first_valid_domain_components(domain_key: str, n_components: int) -> list[str]:
    components = []
    for component in get_domain_config(domain_key).typical_components:
        try:
            components.append(get_component(str(component)).name)
        except ValueError:
            continue
        if len(components) == n_components:
            return components
    raise ValueError(f"Not enough executable components for {domain_key}")


def _run_summary(run: SyntheticDatasetRun) -> dict[str, Any]:
    return {
        "domain": run.metadata["domain"]["key"],
        "instrument": run.metadata["instrument"]["key"],
        "mode": run.metadata["mode"],
        "target_type": run.metadata["target"]["type"],
        "components": run.latent_metadata.get("component_keys", []),
        "validation_summary": run.validation_summary,
    }


def _summary_row(name: str, summary: dict[str, Any]) -> str:
    run_summary = summary["validation_summary"]["summary"]
    return (
        f"| {name} | `{summary['domain']}` | `{summary['instrument']}` | "
        f"`{summary['mode']}` | `{summary['target_type']}` | "
        f"`{run_summary['X_shape']}` | `{run_summary['y_shape']}` | "
        f"`{run_summary['wavelength_range_nm']}` | `{summary['validation_summary']['status']}` |"
    )


def _comparison_summary(source: SyntheticDatasetRun, regenerated: SyntheticDatasetRun) -> dict[str, float]:
    source_mean = np.mean(source.X, axis=0)
    regenerated_mean = np.mean(regenerated.X, axis=0)
    diff = source_mean - regenerated_mean
    return {
        "mean_spectrum_mae": float(np.mean(np.abs(diff))),
        "mean_spectrum_rmse": float(np.sqrt(np.mean(diff**2))),
        "global_mean_abs_gap": float(abs(np.mean(source.X) - np.mean(regenerated.X))),
    }


def _scorecard_inputs(source: SyntheticDatasetRun, regenerated: SyntheticDatasetRun) -> dict[str, Any]:
    source_wavelengths = np.asarray(source.wavelengths, dtype=float)
    regenerated_wavelengths = np.asarray(regenerated.wavelengths, dtype=float)
    diagnostic = _wavelength_grid_diagnostic(source_wavelengths, regenerated_wavelengths)
    if diagnostic["exact_equal"]:
        diagnostic["scorecard_grid"] = "source_regenerated_exact"
        return {
            "source_X": np.asarray(source.X),
            "regenerated_X": np.asarray(regenerated.X),
            "wavelengths": source_wavelengths,
            "wavelength_grid_diagnostic": diagnostic,
        }

    common, source_idx, regenerated_idx = np.intersect1d(
        source_wavelengths,
        regenerated_wavelengths,
        return_indices=True,
    )
    if common.size >= 3:
        diagnostic["scorecard_grid"] = "exact_common_wavelengths"
        diagnostic["common_grid_size"] = int(common.size)
        return {
            "source_X": np.asarray(source.X)[:, source_idx],
            "regenerated_X": np.asarray(regenerated.X)[:, regenerated_idx],
            "wavelengths": common,
            "wavelength_grid_diagnostic": diagnostic,
        }

    diagnostic["scorecard_grid"] = "source_grid_no_exact_common"
    diagnostic["common_grid_size"] = int(common.size)
    diagnostic["warning"] = (
        "No exact common wavelength grid with at least three points; scorecard used "
        "the source grid to preserve legacy validation behavior."
    )
    return {
        "source_X": np.asarray(source.X),
        "regenerated_X": np.asarray(regenerated.X),
        "wavelengths": source_wavelengths,
        "wavelength_grid_diagnostic": diagnostic,
    }


def _wavelength_grid_diagnostic(
    source_wavelengths: np.ndarray,
    regenerated_wavelengths: np.ndarray,
) -> dict[str, Any]:
    same_shape = source_wavelengths.shape == regenerated_wavelengths.shape
    max_abs_diff = (
        float(np.max(np.abs(source_wavelengths - regenerated_wavelengths)))
        if same_shape and source_wavelengths.size
        else None
    )
    return {
        "source_size": int(source_wavelengths.size),
        "regenerated_size": int(regenerated_wavelengths.size),
        "same_shape": bool(same_shape),
        "exact_equal": bool(same_shape and np.array_equal(source_wavelengths, regenerated_wavelengths)),
        "max_abs_diff_nm": max_abs_diff,
        "source_monotonic": bool(np.all(np.diff(source_wavelengths) > 0)),
        "regenerated_monotonic": bool(np.all(np.diff(regenerated_wavelengths) > 0)),
        "source_range_nm": [float(source_wavelengths[0]), float(source_wavelengths[-1])],
        "regenerated_range_nm": [float(regenerated_wavelengths[0]), float(regenerated_wavelengths[-1])],
        "source_median_step_nm": float(np.median(np.diff(source_wavelengths))),
        "regenerated_median_step_nm": float(np.median(np.diff(regenerated_wavelengths))),
    }


def _correlation_length_diagnostic(source_X: np.ndarray, regenerated_X: np.ndarray) -> dict[str, Any]:
    source_lengths = compute_correlation_length(np.asarray(source_X))
    regenerated_lengths = compute_correlation_length(np.asarray(regenerated_X))
    max_lag = min(50, source_X.shape[1] // 4)
    return {
        "overlap": compute_distribution_overlap(source_lengths, regenerated_lengths),
        "threshold": 0.7,
        "passed": bool(compute_distribution_overlap(source_lengths, regenerated_lengths) >= 0.7),
        "source": _distribution_summary(source_lengths, max_lag=max_lag),
        "regenerated": _distribution_summary(regenerated_lengths, max_lag=max_lag),
        "interpretation": (
            "Correlation length is measured as the first autocorrelation lag below 1/e. "
            "Values at max_lag indicate spectra whose autocorrelation never crossed 1/e."
        ),
    }


def _distribution_summary(values: np.ndarray, *, max_lag: int) -> dict[str, Any]:
    unique, counts = np.unique(values, return_counts=True)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
        "max_lag": int(max_lag),
        "max_lag_count": int(np.sum(values >= max_lag)),
        "counts": {str(float(value)): int(count) for value, count in zip(unique, counts, strict=False)},
    }


def _continuous_distribution_summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
        "values": [float(value) for value in values.tolist()],
    }


def _metric_result(scorecard: dict[str, Any], metric_name: str) -> dict[str, Any]:
    for metric in scorecard.get("metric_results", []):
        if metric.get("metric") == metric_name:
            return cast("dict[str, Any]", metric)
    return {}


def _remaining_metric_diagnostics(
    source_X: np.ndarray,
    regenerated_X: np.ndarray,
    wavelengths: np.ndarray,
    *,
    scorecard: dict[str, Any],
) -> dict[str, Any]:
    source_deriv_means, source_deriv_stds = compute_derivative_statistics(source_X, wavelengths, order=1)
    regenerated_deriv_means, regenerated_deriv_stds = compute_derivative_statistics(
        regenerated_X,
        wavelengths,
        order=1,
    )
    source_peak_density = compute_peak_density(source_X, wavelengths)
    regenerated_peak_density = compute_peak_density(regenerated_X, wavelengths)
    source_curvature = compute_baseline_curvature(source_X)
    regenerated_curvature = compute_baseline_curvature(regenerated_X)

    derivative_metric = _metric_result(scorecard, "derivative_statistics")
    peak_metric = _metric_result(scorecard, "peak_density")
    curvature_metric = _metric_result(scorecard, "baseline_curvature")
    source_peak_mean = float(np.mean(source_peak_density))
    regenerated_peak_mean = float(np.mean(regenerated_peak_density))
    peak_ratio = (
        regenerated_peak_mean / source_peak_mean
        if source_peak_mean > 0
        else 1.0 if regenerated_peak_mean == 0 else float("inf")
    )
    curvature_overlap = compute_distribution_overlap(source_curvature, regenerated_curvature)
    return cast("dict[str, Any]", _to_builtin({
        "derivative_statistics": {
            "value": derivative_metric.get("value"),
            "threshold": derivative_metric.get("threshold"),
            "passed": derivative_metric.get("passed"),
            "ks_statistic": derivative_metric.get("details", {}).get("ks_statistic"),
            "source_derivative_mean": _continuous_distribution_summary(source_deriv_means),
            "regenerated_derivative_mean": _continuous_distribution_summary(regenerated_deriv_means),
            "source_derivative_std": _continuous_distribution_summary(source_deriv_stds),
            "regenerated_derivative_std": _continuous_distribution_summary(regenerated_deriv_stds),
            "interpretation": (
                "The scorecard KS test is applied to first-derivative standard deviations; "
                "means are reported to expose preprocessing or slope-state mismatches."
            ),
        },
        "peak_density": {
            "value": peak_metric.get("value", peak_ratio),
            "threshold": peak_metric.get("threshold"),
            "passed": peak_metric.get("passed"),
            "source_density": _continuous_distribution_summary(source_peak_density),
            "regenerated_density": _continuous_distribution_summary(regenerated_peak_density),
            "ratio": peak_ratio,
            "interpretation": "Peak density is counted as peaks per 100 nm using the scorecard prominence rule.",
        },
        "baseline_curvature": {
            "value": curvature_metric.get("value", curvature_overlap),
            "threshold": curvature_metric.get("threshold"),
            "passed": curvature_metric.get("passed"),
            "source_curvature": _continuous_distribution_summary(source_curvature),
            "regenerated_curvature": _continuous_distribution_summary(regenerated_curvature),
            "overlap": curvature_overlap,
            "interpretation": (
                "Baseline curvature is the residual standard deviation after a degree-3 polynomial fit."
            ),
        },
    }))


def _correlation_length_sensitivity(
    *,
    source_run: SyntheticDatasetRun,
    fitted_config: dict[str, Any],
    n_samples: int,
    regen_seed: int,
) -> dict[str, Any]:
    source_components = [str(component) for component in source_run.latent_metadata["component_keys"]]
    source_instrument = str(source_run.metadata["instrument"]["key"])
    diagnostic_config = _source_nuisance_diagnostic_config(fitted_config, source_run)
    try:
        regenerated = build_dataset_run_from_fitted_config(
            diagnostic_config,
            n_samples=n_samples,
            random_seed=regen_seed,
            instrument_override=source_instrument,
            instrument_override_reason=(
                "Correlation-length diagnostic uses source instrument provenance; "
                "this is not a fitted-only gate input."
            ),
            component_override=source_components,
            component_override_reason=(
                "Correlation-length diagnostic uses source component provenance; "
                "this is not a fitted-only gate input."
            ),
        )
    except FittedConfigAdapterError as exc:
        return {
            "status": "failed",
            "scope": "source_provenance_nuisance_diagnostic",
            "validation_summary": exc.validation_summary,
        }

    scorecard_inputs = _scorecard_inputs(source_run, regenerated)
    scorecard = compute_spectral_realism_scorecard(
        scorecard_inputs["source_X"],
        scorecard_inputs["regenerated_X"],
        scorecard_inputs["wavelengths"],
        include_adversarial=False,
        random_state=regen_seed,
    )
    return {
        "status": "passed",
        "scope": "source_provenance_nuisance_diagnostic",
        "gate_eligible": False,
        "reason": (
            "Uses source provenance for instrument, components, and nuisance/effect "
            "settings to isolate whether fitted nuisance inference is sufficient."
        ),
        "scorecard": _scorecard_to_dict(scorecard),
        "failed_metrics": _failed_metric_names({"scorecard": _scorecard_to_dict(scorecard)}),
        "wavelength_grid_diagnostic": scorecard_inputs["wavelength_grid_diagnostic"],
        "correlation_length_diagnostic": _correlation_length_diagnostic(
            scorecard_inputs["source_X"],
            scorecard_inputs["regenerated_X"],
        ),
        "applied_source_nuisance_fields": sorted(_source_nuisance_overrides(source_run).keys()),
    }


def _candidate_mapping_diagnostics(
    *,
    source_run: SyntheticDatasetRun,
    fitted_config: dict[str, Any],
    n_samples: int,
    regen_seed: int,
) -> dict[str, Any]:
    return {
        "n_components": _n_components_candidate_diagnostic(
            source_run=source_run,
            fitted_config=fitted_config,
            n_samples=n_samples,
            regen_seed=regen_seed,
        ),
        "boundary_components_config": _boundary_components_diagnostic(fitted_config),
        "baseline_curvature": _baseline_mapping_diagnostic(fitted_config),
        "preprocessing": _preprocessing_diagnostic(fitted_config),
    }


def _n_components_candidate_diagnostic(
    *,
    source_run: SyntheticDatasetRun,
    fitted_config: dict[str, Any],
    n_samples: int,
    regen_seed: int,
) -> dict[str, Any]:
    fitted_components = [str(component) for component in fitted_config.get("components", [])]
    target_count = int(fitted_config.get("n_components") or len(fitted_components))
    if target_count <= len(fitted_components):
        return {
            "status": "not_applicable",
            "gate_eligible": False,
            "reason": "Fitted n_components does not exceed the detected executable component list.",
            "fitted_components": fitted_components,
            "target_count": target_count,
        }

    expanded_components = _expanded_domain_components(fitted_config, target_count)
    if expanded_components == fitted_components:
        return {
            "status": "not_run",
            "gate_eligible": False,
            "reason": "No additional executable domain components were available to fill fitted n_components.",
            "fitted_components": fitted_components,
            "target_count": target_count,
        }

    candidate_config = dict(fitted_config)
    candidate_config["components"] = expanded_components
    try:
        regenerated = build_dataset_run_from_fitted_config(
            candidate_config,
            n_samples=n_samples,
            random_seed=regen_seed,
        )
    except FittedConfigAdapterError as exc:
        return {
            "status": "failed",
            "gate_eligible": False,
            "reason": "Domain-component fill failed the executable A3 contract.",
            "validation_summary": exc.validation_summary,
            "fitted_components": fitted_components,
            "candidate_components": expanded_components,
            "target_count": target_count,
        }

    scorecard_inputs = _scorecard_inputs(source_run, regenerated)
    scorecard = compute_spectral_realism_scorecard(
        scorecard_inputs["source_X"],
        scorecard_inputs["regenerated_X"],
        scorecard_inputs["wavelengths"],
        include_adversarial=False,
        random_state=regen_seed,
    )
    scorecard_dict = _scorecard_to_dict(scorecard)
    return {
        "status": "passed",
        "gate_eligible": False,
        "candidate_mapping": "fill_missing_component_names_from_inferred_domain_until_fitted_n_components",
        "reason": (
            "This uses only fitted_config.domain and fitted_config.n_components, but it imputes "
            "component identities that RealDataFitter did not detect, so it remains diagnostic."
        ),
        "fitted_components": fitted_components,
        "candidate_components": expanded_components,
        "target_count": target_count,
        "scientific_status": "passed" if bool(scorecard.overall_pass) else "failed",
        "scorecard": scorecard_dict,
        "failed_metrics": _failed_metric_names({"scorecard": scorecard_dict}),
        "remaining_metric_diagnostics": _remaining_metric_diagnostics(
            scorecard_inputs["source_X"],
            scorecard_inputs["regenerated_X"],
            scorecard_inputs["wavelengths"],
            scorecard=scorecard_dict,
        ),
    }


def _expanded_domain_components(fitted_config: dict[str, Any], target_count: int) -> list[str]:
    components = [str(component) for component in fitted_config.get("components", []) if str(component)]
    try:
        domain = get_domain_config(str(fitted_config.get("domain", "")))
    except Exception:
        return components
    for component in domain.typical_components:
        try:
            name = get_component(str(component)).name
        except ValueError:
            continue
        if name not in components:
            components.append(name)
        if len(components) >= target_count:
            break
    return components


def _boundary_components_diagnostic(fitted_config: dict[str, Any]) -> dict[str, Any]:
    config = fitted_config.get("boundary_components_config")
    components = config.get("components", []) if isinstance(config, dict) else []
    return {
        "status": "not_applicable" if not components else "unsupported",
        "gate_eligible": False,
        "raw": _to_builtin(config),
        "component_count": len(components) if isinstance(components, list) else 0,
        "reason": (
            "Actual fitted boundary_components_config is empty for this source; no fitted-only "
            "boundary component can be executed. Non-empty configs remain diagnostic until the "
            "bench adapter has an exact ComponentLibrary mapping for fitted band_center/bandwidth/amplitude."
        ),
    }


def _baseline_mapping_diagnostic(fitted_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "blocked",
        "gate_eligible": False,
        "raw_baseline_amplitude": _to_builtin(fitted_config.get("baseline_amplitude")),
        "raw_edge_artifacts_config": _to_builtin(fitted_config.get("edge_artifacts_config")),
        "reason": (
            "RealDataFitter exposes aggregate final-spectrum baseline and edge-curvature indicators, "
            "but not an inverse mapping to generator polynomial baseline coefficients. The fitted "
            "residual-effects contract now executes observed-spectrum baseline coefficient "
            "distributions separately; any remaining curvature gap needs a source-free residual "
            "basis improvement, not source provenance or threshold changes."
        ),
    }


def _preprocessing_diagnostic(fitted_config: dict[str, Any]) -> dict[str, Any]:
    preprocessing_type = str(fitted_config.get("preprocessing_type", ""))
    is_preprocessed = bool(fitted_config.get("is_preprocessed", False))
    status = "not_applicable" if preprocessing_type == "raw_absorbance" and not is_preprocessed else "unsupported"
    reason = (
        "The fitted source is raw_absorbance and is_preprocessed=False; derivative-statistic mismatch "
        "is therefore not explained by an executable preprocessing inverse."
        if status == "not_applicable"
        else "A3 has no executable inverse-preprocessing contract for fitted preprocessed spectra."
    )
    return {
        "status": status,
        "gate_eligible": False,
        "preprocessing_type": preprocessing_type,
        "is_preprocessed": is_preprocessed,
        "reason": reason,
    }


def _source_nuisance_diagnostic_config(
    fitted_config: dict[str, Any],
    source_run: SyntheticDatasetRun,
) -> dict[str, Any]:
    diagnostic_config = dict(fitted_config)
    diagnostic_config.update(_source_nuisance_overrides(source_run))
    return diagnostic_config


def _source_nuisance_overrides(source_run: SyntheticDatasetRun) -> dict[str, Any]:
    nuisance = source_run.builder_config["nuisance"]
    custom_params = nuisance["custom_params"]
    particle_size = float(nuisance["particle_size_um"])
    return {
        "complexity": source_run.builder_config["features"]["complexity"],
        "noise_base": float(custom_params["noise_base"]),
        "noise_signal_dep": float(custom_params["noise_signal_dep"]),
        "baseline_amplitude": float(custom_params["baseline_amplitude"]),
        "scatter_alpha_std": float(custom_params["scatter_alpha_std"]),
        "scatter_beta_std": float(custom_params["scatter_beta_std"]),
        "path_length_std": 0.05,
        "tilt_std": 0.01,
        "global_slope_mean": 0.05,
        "global_slope_std": 0.03,
        "particle_size_config": {
            "mean_size_um": particle_size,
            "std_size_um": max(1e-6, particle_size * 0.05),
            "size_effect_strength": 1.0,
        },
        "emsc_config": {
            "multiplicative_scatter_std": float(custom_params["scatter_alpha_std"]),
            "additive_scatter_std": float(custom_params["scatter_beta_std"]),
            "polynomial_order": 2,
            "include_wavelength_terms": True,
        },
        "temperature_config": {},
        "moisture_config": {},
        "edge_artifacts_config": {
            "detector_rolloff": {
                "enabled": bool(nuisance["edge_artifacts"]["detector_rolloff_enabled"]),
                "severity": float(nuisance["edge_artifacts"]["rolloff_severity"]),
            },
        },
    }


def _scorecard_to_dict(scorecard: Any) -> dict[str, Any]:
    result = scorecard.to_dict()
    result["metric_results"] = [
        {
            "metric": metric.metric.value,
            "value": metric.value,
            "threshold": metric.threshold,
            "passed": metric.passed,
            "details": metric.details,
        }
        for metric in scorecard.metric_results
    ]
    return cast("dict[str, Any]", _to_builtin(result))


def _correlation_length_section(result: dict[str, Any]) -> str:
    lines: list[str] = []
    if result["status"] != "passed":
        lines.append("No correlation-length diagnostics because regeneration failed.")
        return "\n".join(lines)

    grid = result.get("wavelength_grid_diagnostic", {})
    diagnostic = result.get("correlation_length_diagnostic", {})
    lines.extend([
        f"- Wavelength grids exact: `{grid.get('exact_equal')}`",
        f"- Scorecard grid: `{grid.get('scorecard_grid')}`",
        f"- Max wavelength difference: `{grid.get('max_abs_diff_nm')}` nm",
        (
            "- Fitted-only correlation-length overlap: "
            f"{diagnostic.get('overlap', 0.0):.6g} / {diagnostic.get('threshold', 0.7):.6g}"
        ),
    ])
    if diagnostic:
        source = diagnostic["source"]
        regenerated = diagnostic["regenerated"]
        lines.extend([
            (
                "- Source lag distribution: "
                f"mean={source['mean']:.6g}, median={source['median']:.6g}, "
                f"q25={source['q25']:.6g}, q75={source['q75']:.6g}, "
                f"max={source['max']:.6g}, max_lag_count={source['max_lag_count']}"
            ),
            (
                "- Regenerated lag distribution: "
                f"mean={regenerated['mean']:.6g}, median={regenerated['median']:.6g}, "
                f"q25={regenerated['q25']:.6g}, q75={regenerated['q75']:.6g}, "
                f"max={regenerated['max']:.6g}, max_lag_count={regenerated['max_lag_count']}"
            ),
        ])

    sensitivity = result.get("correlation_length_sensitivity", {})
    if sensitivity:
        if sensitivity.get("status") == "passed":
            sens_diag = sensitivity["correlation_length_diagnostic"]
            lines.extend([
                "",
                "Source-provenance nuisance sensitivity (diagnostic only, not gate-eligible):",
                (
                    f"- Correlation-length overlap: {sens_diag['overlap']:.6g} / "
                    f"{sens_diag['threshold']:.6g}; failed metrics: "
                    f"`{', '.join(sensitivity.get('failed_metrics', [])) or 'none'}`"
                ),
                f"- Applied source nuisance fields: `{sensitivity.get('applied_source_nuisance_fields', [])}`",
            ])
        else:
            lines.extend([
                "",
                "Source-provenance nuisance sensitivity failed before comparison.",
            ])
    lines.extend([
        "",
        _correlation_length_conclusion(result),
    ])
    return "\n".join(lines)


def _correlation_length_conclusion(result: dict[str, Any]) -> str:
    diagnostic = result.get("correlation_length_diagnostic", {})
    sensitivity = result.get("correlation_length_sensitivity", {})
    if not diagnostic:
        return "Conclusion: correlation-length source/regenerated values were not available."
    if diagnostic.get("passed"):
        if result.get("scientific_status") == "passed":
            return "Conclusion: correlation length is not blocking the fitted-only gate."
        return "Conclusion: correlation length is not blocking the fitted-only gate."
    grid = result.get("wavelength_grid_diagnostic", {})
    if grid.get("exact_equal") is False:
        return "Conclusion: wavelength-grid mismatch remains a possible adapter issue."
    if sensitivity.get("status") == "passed" and sensitivity.get("correlation_length_diagnostic", {}).get("passed"):
        return (
            "Conclusion: the stable blocker is not wavelength-grid alignment, validation invocation, "
            "instrument provenance, or component provenance. A source-provenance nuisance diagnostic "
            "can recover correlation length, so the fitted-only blocker is the fitter/generator "
            "nuisance/effect reconstruction contract rather than a local threshold issue."
        )
    return (
        "Conclusion: correlation length remains blocked after grid, instrument, and component "
        "diagnostics; no local fitted-only adapter correction was identified."
    )


def _remaining_metric_section(result: dict[str, Any]) -> str:
    diagnostics = result.get("remaining_metric_diagnostics")
    if result["status"] != "passed" or not diagnostics:
        return "No remaining-metric diagnostics because regeneration failed."

    lines = [
        "| metric | value | threshold | passed | source distribution | regenerated distribution |",
        "|---|---:|---:|---|---|---|",
    ]
    derivative = diagnostics["derivative_statistics"]
    lines.append(_diagnostic_metric_row(
        "derivative_statistics",
        derivative,
        source_key="source_derivative_std",
        regenerated_key="regenerated_derivative_std",
    ))
    peak = diagnostics["peak_density"]
    lines.append(_diagnostic_metric_row(
        "peak_density",
        peak,
        source_key="source_density",
        regenerated_key="regenerated_density",
    ))
    curvature = diagnostics["baseline_curvature"]
    lines.append(_diagnostic_metric_row(
        "baseline_curvature",
        curvature,
        source_key="source_curvature",
        regenerated_key="regenerated_curvature",
    ))
    lines.extend([
        "",
        "- Full per-sample source/regenerated values for these distributions are included in Raw Summary JSON.",
        _remaining_metric_conclusion(result),
    ])
    return "\n".join(lines)


def _diagnostic_metric_row(
    name: str,
    diagnostic: dict[str, Any],
    *,
    source_key: str,
    regenerated_key: str,
) -> str:
    return (
        f"| `{name}` | {_format_optional_float(diagnostic.get('value'))} | "
        f"{_format_optional_float(diagnostic.get('threshold'))} | `{diagnostic.get('passed')}` | "
        f"{_compact_distribution(diagnostic[source_key])} | "
        f"{_compact_distribution(diagnostic[regenerated_key])} |"
    )


def _compact_distribution(summary: dict[str, Any]) -> str:
    return (
        f"mean={summary['mean']:.6g}, std={summary['std']:.6g}, "
        f"q25={summary['q25']:.6g}, median={summary['median']:.6g}, "
        f"q75={summary['q75']:.6g}"
    )


def _format_optional_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"


def _remaining_metric_conclusion(result: dict[str, Any]) -> str:
    diagnostics = result.get("remaining_metric_diagnostics", {})
    failed = [
        name
        for name in ["derivative_statistics", "peak_density", "baseline_curvature"]
        if not diagnostics.get(name, {}).get("passed", False)
    ]
    if not failed:
        return "Conclusion: remaining fitted-only scorecard metrics are not blocking."
    candidate = result.get("candidate_mapping_diagnostics", {})
    preprocessing = candidate.get("preprocessing", {})
    boundary = candidate.get("boundary_components_config", {})
    residual = (
        result.get("effect_reconstruction", {})
        .get("fields", {})
        .get("fitted_residual_effects_config", {})
        .get("effective", {})
    )
    residual_note = (
        " The observed residual-effects contract was applied with "
        f"{residual.get('peak_templates', 0)} peak templates and "
        f"{residual.get('baseline_coefficients', 0)} baseline coefficients."
        if residual
        else ""
    )
    return (
        "Conclusion: remaining fitted-only blockers are "
        f"`{', '.join(failed)}`. "
        f"Preprocessing state is `{preprocessing.get('preprocessing_type')}`/"
        f"`is_preprocessed={preprocessing.get('is_preprocessed')}`, and boundary components are "
        f"`{boundary.get('status')}`.{residual_note} Next correction: improve the "
        "source-free residual peak-template basis and baseline coefficient model, then rerun "
        "the unchanged scorecard thresholds."
    )


def _candidate_mapping_section(result: dict[str, Any]) -> str:
    diagnostics = result.get("candidate_mapping_diagnostics", {})
    if not diagnostics:
        return "No candidate mapping diagnostics were run."

    n_components = diagnostics.get("n_components", {})
    boundary = diagnostics.get("boundary_components_config", {})
    baseline = diagnostics.get("baseline_curvature", {})
    preprocessing = diagnostics.get("preprocessing", {})
    lines = [
        "| area | status | gate eligible | finding |",
        "|---|---|---|---|",
        (
            f"| `n_components` | `{n_components.get('status')}` | "
            f"`{n_components.get('gate_eligible')}` | "
            f"{_n_components_finding(n_components)} |"
        ),
        (
            f"| `boundary_components_config` | `{boundary.get('status')}` | "
            f"`{boundary.get('gate_eligible')}` | {boundary.get('reason')} |"
        ),
        (
            f"| `baseline_curvature` | `{baseline.get('status')}` | "
            f"`{baseline.get('gate_eligible')}` | {baseline.get('reason')} |"
        ),
        (
            f"| `preprocessing` | `{preprocessing.get('status')}` | "
            f"`{preprocessing.get('gate_eligible')}` | {preprocessing.get('reason')} |"
        ),
    ]
    return "\n".join(lines)


def _final_unblock_memo(result: dict[str, Any]) -> str:
    if result["status"] != "passed":
        return "Blocked before scientific comparison: the fitted config did not regenerate an executable dataset."

    failed = _failed_metric_names(result)
    if not failed:
        return "No blocker remains: the fitted-only executable contract and scorecard both passed."

    diagnostics = result.get("candidate_mapping_diagnostics", {})
    boundary = diagnostics.get("boundary_components_config", {})
    preprocessing = diagnostics.get("preprocessing", {})
    baseline = diagnostics.get("baseline_curvature", {})
    n_components = diagnostics.get("n_components", {})
    ablations = result.get("ablations", [])
    source_provenance_effect = [
        variant
        for variant in ablations
        if variant.get("diagnostic_scope") != "fitted_only"
    ]
    source_provenance_peak_recovered = any(
        "peak_density" not in _failed_metric_names(variant)
        for variant in source_provenance_effect
        if variant.get("status") == "passed"
    )

    lines = [
        (
            "Final A3 unblock attempt is blocked, not failed contractually: the fitted-only "
            "variant regenerates finite spectra on the exact wavelength grid, but the configured "
            f"scorecard still fails `{', '.join(failed)}`."
        ),
        "",
        "Decision points:",
        (
            "- `peak_density` has no fitted-only repair selected: fitted components remain the "
            "only executable component identities, `n_components` fill is diagnostic only, and "
            f"source-provenance ablations recover peak density=`{source_provenance_peak_recovered}`."
        ),
        (
            "- `derivative_statistics` is not a preprocessing bug: "
            f"`preprocessing_type={preprocessing.get('preprocessing_type')}` and "
            f"`is_preprocessed={preprocessing.get('is_preprocessed')}`."
        ),
        (
            "- `baseline_curvature` has no defensible inverse: "
            f"{baseline.get('reason', 'no baseline diagnostic available')}"
        ),
        (
            "- The new `fitted_residual_effects_config` is executed from observed-spectrum "
            "coefficient distributions, but it did not close all fitted-only scorecard gaps; "
            "the next correction is a richer residual peak/baseline basis learned from `X` "
            "and `wavelengths` only."
        ),
        (
            "- Boundary components remain "
            f"`{boundary.get('status')}` with component_count="
            f"`{boundary.get('component_count')}`."
        ),
        (
            "- The fitted `n_components` diagnostic is "
            f"`{n_components.get('status')}` and gate_eligible="
            f"`{n_components.get('gate_eligible')}`."
        ),
        "",
        (
            "Conclusion: do not loosen thresholds, do not use source/oracle provenance in the "
            "fitted-only gate, and do not adopt component or nuisance imputations that the fitted "
            "config did not make executable. A3 remains blocked on fitter/generator information "
            "loss rather than a local bench adapter contract bug."
        ),
    ]
    return "\n".join(lines)


def _n_components_finding(diagnostic: dict[str, Any]) -> str:
    if diagnostic.get("status") != "passed":
        return str(diagnostic.get("reason"))
    failed = ", ".join(diagnostic.get("failed_metrics", [])) or "none"
    components = diagnostic.get("candidate_components", [])
    return (
        f"Fitted component fill to n_components={diagnostic.get('target_count')} "
        f"used `{components}` and scientific_status=`{diagnostic.get('scientific_status')}`; "
        f"failed metrics=`{failed}`. {diagnostic.get('reason')}"
    )


def _effect_reconstruction_section(result: dict[str, Any]) -> str:
    mapping = result.get("effect_reconstruction", {})
    if result["status"] != "passed" or not mapping:
        return "No fitted nuisance/effect reconstruction metadata because regeneration failed."

    fields = mapping.get("fields", {})
    lines = [
        f"- Mode: `{mapping.get('mode')}`",
        (
            "- Scope: fitted-only; raw `RealDataFitter.to_full_config()` values are "
            "reported separately from the effective generator controls."
        ),
        "",
        "| field | raw | effective | mapping |",
        "|---|---|---|---|",
    ]
    for field in [
        "complexity",
        "path_length_std",
        "baseline_amplitude",
        "scatter_alpha_std",
        "scatter_beta_std",
        "global_slope_mean",
        "global_slope_std",
        "particle_size_config",
        "emsc_config",
        "temperature_config",
        "moisture_config",
        "edge_artifacts_config",
        "fitted_residual_effects_config",
    ]:
        item = fields.get(field)
        if not item:
            continue
        lines.append(
            f"| `{field}` | `{_short_json(item.get('raw'))}` | "
            f"`{_short_json(item.get('effective'))}` | `{item.get('mapping')}` |"
        )

    failed = _failed_metric_names(result)
    if failed:
        lines.extend([
            "",
            "Remaining fitted-only scientific blockers after reconstruction:",
            f"- Failed metrics: `{', '.join(failed)}`",
        ])
    return "\n".join(lines)


def _short_json(value: Any, *, max_len: int = 120) -> str:
    text = json.dumps(_to_builtin(value), sort_keys=True)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _failed_metric_names(variant: dict[str, Any]) -> list[str]:
    scorecard = variant.get("scorecard") or {}
    return [
        str(metric["metric"])
        for metric in scorecard.get("metric_results", [])
        if not metric.get("passed", False)
    ]


def _component_gap(source: dict[str, Any], fitted_config: dict[str, Any]) -> list[str]:
    source_components = {str(component) for component in source.get("components", [])}
    fitted_components = {str(component) for component in fitted_config.get("components", [])}
    return sorted(source_components - fitted_components)


def _ablation_row(variant: dict[str, Any]) -> str:
    reason = " / ".join(
        item
        for item in [
            variant.get("instrument_override_reason"),
            variant.get("component_override_reason"),
        ]
        if item
    )
    if variant["status"] != "passed":
        failures = variant.get("validation_summary", {}).get("failures", [])
        return (
            f"| `{variant['variant_name']}` | `{variant.get('diagnostic_scope')}` | _failed_ | "
            f"_failed_ | `{variant.get('contract_status')}` | `not_run` | `{failures}` |  |  | `{reason}` |"
        )
    comparison = variant["comparison"]
    failed_metrics = ", ".join(_failed_metric_names(variant)) or "none"
    instrument = variant["regenerated"]["instrument"]
    components = variant["regenerated"].get("components", [])
    return (
        f"| `{variant['variant_name']}` | `{variant['diagnostic_scope']}` | `{instrument}` | "
        f"`{components}` | `{variant['contract_status']}` | `{variant['scientific_status']}` | `{failed_metrics}` | "
        f"{comparison['mean_spectrum_mae']:.6g} | {comparison['global_mean_abs_gap']:.6g} | "
        f"`{reason}` |"
    )


def _failure_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_failures": result.get("validation_summary", {}).get("failures", []),
        "scientific_failures": _failed_metric_names(result),
        "correlation_length_diagnostic": result.get("correlation_length_diagnostic"),
        "remaining_metric_diagnostics": result.get("remaining_metric_diagnostics"),
        "wavelength_grid_diagnostic": result.get("wavelength_grid_diagnostic"),
        "correlation_length_sensitivity": result.get("correlation_length_sensitivity"),
        "candidate_mapping_diagnostics": result.get("candidate_mapping_diagnostics"),
        "ablations": [
            {
                "variant": variant.get("variant_name"),
                "contract_status": variant.get("contract_status"),
                "scientific_status": variant.get("scientific_status"),
                "failed_metrics": _failed_metric_names(variant),
            }
            for variant in result.get("ablations", [])
        ],
    }


def _gate_outcome(result: dict[str, Any]) -> str:
    if result["status"] != "passed":
        return "A3 gate blocked: regeneration did not satisfy the executable dataset contract."
    if result.get("scientific_status") != "passed":
        suffix = ""
        if _oracle_variant_passed(result):
            suffix = " An oracle/source-provenance ablation passed, so the fitted-only blocker is diagnostic."
        return (
            "A3 gate blocked: the executable contract passed, but regenerated spectra "
            "did not pass the configured real/synthetic similarity scorecard."
            f"{suffix}"
        )
    return "A3 gate passed: executable contract and configured similarity scorecard both passed."


def _decision(result: dict[str, Any]) -> str:
    if result["status"] != "passed":
        return "Blocked: regeneration failed before scientific comparison."
    if result.get("scientific_status") != "passed":
        suffix = ""
        if _oracle_variant_passed(result):
            suffix = " Oracle/source-provenance success does not satisfy the fitted-only A3 gate."
        return (
            "Blocked: A3 contract is executable and unsupported fitted fields are explicit, "
            "but the fitted-only similarity scorecard failed, so the A3 gate is not satisfied."
            f"{suffix}"
        )
    return "Passed: A3 contract is executable and regenerated spectra passed the configured similarity scorecard."


def _oracle_variant_passed(result: dict[str, Any]) -> bool:
    return any(
        variant.get("diagnostic_scope") == "oracle_source_provenance"
        and variant.get("scientific_status") == "passed"
        for variant in result.get("ablations", [])
    )


def _git_status_summary() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--short"],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    by_status: dict[str, int] = {}
    for line in lines:
        status = line[:2].strip() or "unknown"
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "returncode": result.returncode,
        "entry_count": len(lines),
        "by_status": dict(sorted(by_status.items())),
        "sample": lines[:20],
    }


def _git_status_section(summary: dict[str, Any]) -> str:
    lines = [
        f"- Return code: {summary.get('returncode')}",
        f"- Entries: {summary.get('entry_count')}",
        f"- Status counts: `{summary.get('by_status', {})}`",
    ]
    sample = summary.get("sample") or []
    if sample:
        lines.append("- Sample:")
        lines.extend(f"  - `{line}`" for line in sample)
    return "\n".join(lines)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


if __name__ == "__main__":
    main()
