"""Observed-spectrum residual effects for the A3 fitted-config bench adapter."""

from __future__ import annotations

import importlib.util
from typing import Any, cast

import numpy as np

from nirs4all.synthesis import ComponentLibrary

MAX_BASELINE_ORDER = 9
BASE_FIT_BASELINE_ORDER = 3
MAX_DETAIL_TEMPLATES = 64
PEAK_PROMINENCE_STD_FRACTION = 0.04


def fit_observable_residual_effects(
    X: np.ndarray,
    wavelengths: np.ndarray,
    fitted_config: dict[str, Any],
) -> dict[str, Any]:
    """Fit a serializable residual-effects contract from observed fitted spectra only."""
    X_array = np.asarray(X, dtype=float)
    wl = np.asarray(wavelengths, dtype=float)
    base = _base_config(X_array, wl)
    if X_array.ndim != 2 or wl.ndim != 1 or X_array.shape[1] != wl.size or X_array.shape[0] < 2 or wl.size < 5:
        return base | {
            "enabled": False,
            "status": "insufficient_observed_spectra",
        }
    finite_mask = np.isfinite(X_array).all(axis=1)
    finite_wavelengths = np.isfinite(wl).all()
    if not finite_wavelengths or not finite_mask.any():
        return base | {
            "enabled": False,
            "status": "non_finite_observed_spectra",
        }
    X_fit = X_array[finite_mask]
    component_names, component_basis = _observable_component_basis(wl, fitted_config)
    base_basis = _stack_nonempty([
        component_basis,
        _chebyshev_basis(wl, BASE_FIT_BASELINE_ORDER),
    ])
    reconstructed = _least_squares_reconstruction(X_fit, base_basis)
    residual = X_fit - reconstructed

    observed_distributions = _observed_distributions(X_fit, wl)
    residual_baseline_basis = _chebyshev_basis(wl, MAX_BASELINE_ORDER)
    baseline_coefficients = _fit_coefficients(residual, residual_baseline_basis)
    local_residual = residual - baseline_coefficients @ residual_baseline_basis
    detail_model = _local_detail_model(
        X_fit,
        local_residual,
        wl,
        clip_abs=max(_quantile(np.abs(residual).ravel(), 0.995, default=0.0) * 3.0, 1e-8),
    )

    residual_abs = np.abs(residual)
    spectrum_range = np.ptp(X_fit, axis=1)
    residual_q995 = _quantile(residual_abs.ravel(), 0.995, default=0.0)
    range_q95 = _quantile(spectrum_range, 0.95, default=1.0)
    max_abs_effect = float(np.clip(max(residual_q995 * 3.0, 1e-8), 1e-8, max(range_q95 * 0.5, 1e-8)))

    return cast("dict[str, Any]", _to_builtin(base | {
        "enabled": bool(residual_q995 > 0.0 and (baseline_coefficients.size or detail_model["templates"])),
        "status": "fitted",
        "no_oracle": True,
        "fit": {
            "component_basis": {
                "components": component_names,
                "used_count": len(component_names),
                "basis_source": "fitted_config.components",
            },
            "base_baseline": {
                "kind": "chebyshev",
                "order": BASE_FIT_BASELINE_ORDER,
            },
            "residual_baseline": {
                "kind": "chebyshev",
                "order": MAX_BASELINE_ORDER,
                "model": "aggregate_distribution_not_rows",
                "coefficients": [
                    {"order": int(order), **_distribution(baseline_coefficients[:, order], clip_abs=max_abs_effect)}
                    for order in range(baseline_coefficients.shape[1])
                ],
            },
            "local_details": {
                "method": detail_model["method"],
                "model": "clustered_peak_ridge_distributions_not_rows",
                "max_templates": MAX_DETAIL_TEMPLATES,
                "active_count": observed_distributions["peak_count"],
                "templates": detail_model["templates"],
            },
        },
        "observed_distributions": observed_distributions,
        "residual_summary": {
            "mean_abs": float(np.mean(residual_abs)),
            "q95_abs": _quantile(residual_abs.ravel(), 0.95, default=0.0),
            "q995_abs": residual_q995,
            "max_abs_effect": max_abs_effect,
            "local_detail_template_count": len(detail_model["templates"]),
            "observed_peak_density_mean": observed_distributions["peak_density"]["mean"],
            "observed_baseline_curvature_mean": observed_distributions["baseline_curvature"]["mean"],
        },
        "application": {
            "mode": "sample_observed_distributions_and_local_detail_templates",
            "additive": True,
            "max_abs_effect": max_abs_effect,
            "match_observed_metric_distributions": True,
            "quantile_matching": {
                "scope": "fitted_observed_metric_distributions",
                "sampling": "inverse_cdf_quantile_sampling",
                "oracle_provenance": False,
                "row_level_values_serialized": False,
                "auditable": True,
            },
        },
    }))


def _base_config(X: np.ndarray, wavelengths: np.ndarray) -> dict[str, Any]:
    wavelength_range = [float(wavelengths[0]), float(wavelengths[-1])] if wavelengths.ndim == 1 and wavelengths.size else []
    return {
        "version": "A3.2-observable-local-residual-effects",
        "source": "observed_fitted_spectra",
        "oracle_provenance": False,
        "no_oracle": True,
        "source_fields": ["X", "wavelengths", "fitted_config.components"],
        "n_samples": int(X.shape[0]) if X.ndim == 2 else 0,
        "n_wavelengths": int(wavelengths.size) if wavelengths.ndim == 1 else 0,
        "wavelength_range_nm": wavelength_range,
    }


def _observable_component_basis(wavelengths: np.ndarray, fitted_config: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    raw_components = fitted_config.get("components", [])
    if not isinstance(raw_components, (list, tuple)):
        return [], np.empty((0, wavelengths.size), dtype=float)
    names: list[str] = []
    rows: list[np.ndarray] = []
    for raw_name in raw_components[:24]:
        name = str(raw_name)
        if not name:
            continue
        try:
            library = ComponentLibrary.from_predefined([name], random_state=0)
            spectrum = np.asarray(library.compute_all(wavelengths)[0], dtype=float)
        except Exception:
            continue
        normalized = _normalize_basis_row(spectrum)
        if normalized is None:
            continue
        names.append(name)
        rows.append(normalized)
    return names, np.vstack(rows) if rows else np.empty((0, wavelengths.size), dtype=float)


def _chebyshev_basis(wavelengths: np.ndarray, order: int) -> np.ndarray:
    if wavelengths.size == 0:
        return np.empty((0, 0), dtype=float)
    lower = float(wavelengths[0])
    upper = float(wavelengths[-1])
    if np.isclose(lower, upper):
        scaled = np.zeros_like(wavelengths, dtype=float)
    else:
        scaled = 2.0 * (wavelengths - lower) / (upper - lower) - 1.0
    return np.polynomial.chebyshev.chebvander(scaled, order).T.astype(float)


def _stack_nonempty(arrays: list[np.ndarray]) -> np.ndarray:
    rows = [array for array in arrays if array.size and array.shape[0] > 0]
    if not rows:
        return np.empty((0, 0), dtype=float)
    return np.vstack(rows)


def _least_squares_reconstruction(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if not basis.size:
        return np.zeros_like(X, dtype=float)
    coefficients = _fit_coefficients(X, basis)
    return np.asarray(coefficients @ basis, dtype=float)


def _fit_coefficients(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if not basis.size:
        return np.empty((X.shape[0], 0), dtype=float)
    coefficients, *_ = np.linalg.lstsq(basis.T, X.T, rcond=None)
    return np.asarray(coefficients.T, dtype=float)


def _observed_distributions(X: np.ndarray, wavelengths: np.ndarray) -> dict[str, Any]:
    wavelength_span = max(float(np.ptp(wavelengths)), 1e-8)
    try:
        from nirs4all.synthesis.validation import (
            compute_baseline_curvature,
            compute_derivative_statistics,
            compute_peak_density,
        )

        _, derivative_stds = compute_derivative_statistics(X, wavelengths, order=1)
        peak_density = compute_peak_density(X, wavelengths)
        curvature = compute_baseline_curvature(X)
        peak_counts = np.rint(peak_density * wavelength_span / 100.0).astype(int)
    except Exception:
        peak_counts = _peak_counts(X, wavelengths)
        derivative = np.diff(X, axis=1) / np.diff(wavelengths)[None, :]
        derivative_stds = np.std(derivative, axis=1)
        peak_density = peak_counts * 100.0 / wavelength_span
        curvature = _baseline_curvature(X)
    return {
        "derivative_std": _distribution(derivative_stds, clip_abs=max(_quantile(derivative_stds, 0.995, default=0.0) * 3.0, 1e-8)),
        "peak_count": _integer_distribution(peak_counts),
        "peak_density": _distribution(peak_density, clip_abs=max(float(np.max(peak_density)), 1e-8)),
        "baseline_curvature": _distribution(curvature, clip_abs=max(_quantile(curvature, 0.995, default=0.0) * 3.0, 1e-8)),
    }


def _peak_counts(X: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    return np.asarray([_peak_indices(row, wavelengths)[0].size for row in X], dtype=int)


def _baseline_curvature(X: np.ndarray) -> np.ndarray:
    positions = np.linspace(-1.0, 1.0, X.shape[1])
    basis = np.vander(positions, 4, increasing=True)
    coefficients, *_ = np.linalg.lstsq(basis, X.T, rcond=None)
    fitted = (basis @ coefficients).T
    return np.asarray(np.std(X - fitted, axis=1), dtype=float)


def _local_detail_model(
    X: np.ndarray,
    local_residual: np.ndarray,
    wavelengths: np.ndarray,
    *,
    clip_abs: float,
) -> dict[str, Any]:
    observations: list[dict[str, float]] = []
    spacing = _median_spacing(wavelengths)
    for row, detail in zip(X, local_residual, strict=False):
        peak_indices, widths = _peak_indices(row, wavelengths)
        for peak_idx in peak_indices:
            amplitude = float(detail[int(peak_idx)])
            if not np.isfinite(amplitude) or abs(amplitude) <= 0.0:
                continue
            width_nm = float(widths.get(int(peak_idx), max(2.0 * spacing, 8.0)))
            observations.append({
                "center_nm": float(wavelengths[int(peak_idx)]),
                "sigma_nm": float(np.clip(width_nm / 2.355, max(spacing * 0.5, 1.0), max(spacing * 6.0, 2.0))),
                "amplitude": amplitude,
            })
    if not observations:
        return {"method": _peak_method(), "templates": []}
    bin_width = max(spacing * 3.0, 24.0)
    lower = float(wavelengths[0])
    clusters: dict[int, list[dict[str, float]]] = {}
    for observation in observations:
        key = int(np.floor((observation["center_nm"] - lower) / bin_width))
        clusters.setdefault(key, []).append(observation)
    ordered_clusters = sorted(clusters.values(), key=len, reverse=True)[:MAX_DETAIL_TEMPLATES]
    total = float(sum(len(cluster) for cluster in ordered_clusters)) or 1.0
    templates = []
    for cluster in sorted(ordered_clusters, key=lambda items: float(np.mean([item["center_nm"] for item in items]))):
        centers = np.asarray([item["center_nm"] for item in cluster], dtype=float)
        sigmas = np.asarray([item["sigma_nm"] for item in cluster], dtype=float)
        amplitudes = np.asarray([item["amplitude"] for item in cluster], dtype=float)
        templates.append({
            "basis": "gaussian",
            "selection": "clustered_observed_local_peak_residual",
            "observed_count": len(cluster),
            "weight": float(len(cluster) / total),
            "center_nm": float(np.mean(centers)),
            "center_std_nm": float(np.clip(np.std(centers), 0.0, bin_width)),
            "sigma_nm": _distribution(sigmas, clip_abs=max(spacing * 8.0, 1.0)),
            "amplitude_abs": _distribution(np.abs(amplitudes), clip_abs=clip_abs),
            "positive_probability": float(np.mean(amplitudes >= 0.0)),
        })
    return {"method": _peak_method(), "templates": templates}


def _peak_indices(signal: np.ndarray, wavelengths: np.ndarray) -> tuple[np.ndarray, dict[int, float]]:
    spacing = _median_spacing(wavelengths)
    distance = 1
    prominence = max(float(np.std(signal)) * PEAK_PROMINENCE_STD_FRACTION, 1e-10)
    if _has_scipy_signal():
        try:
            from scipy.signal import find_peaks, peak_widths

            peaks, _ = find_peaks(signal, distance=distance, prominence=prominence)
            if peaks.size:
                widths_samples = peak_widths(signal, peaks, rel_height=0.5)[0]
                return np.asarray(peaks, dtype=int), {
                    int(peak): float(max(width, 1.0) * spacing)
                    for peak, width in zip(peaks, widths_samples, strict=False)
                }
        except Exception:
            pass
    local = np.flatnonzero((signal[1:-1] > signal[:-2]) & (signal[1:-1] >= signal[2:])) + 1
    local = local[signal[local] >= prominence]
    if not local.size:
        return np.empty(0, dtype=int), {}
    return _dedupe_by_distance(local, signal, distance), {}


def _peak_method() -> str:
    return "scipy.signal.find_peaks" if _has_scipy_signal() else "numpy_local_maxima"


def _dedupe_by_distance(peaks: np.ndarray, signal: np.ndarray, distance: int) -> np.ndarray:
    selected: list[int] = []
    for peak in sorted((int(value) for value in peaks), key=lambda idx: float(signal[idx]), reverse=True):
        if all(abs(peak - existing) >= distance for existing in selected):
            selected.append(peak)
    return np.asarray(sorted(selected), dtype=int)


def _peak_basis(wavelengths: np.ndarray, templates: list[dict[str, Any]]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for template in templates:
        center = float(template["center_nm"])
        sigma = max(float(template["sigma_nm"]), 1e-6)
        rows.append(np.exp(-0.5 * np.square((wavelengths - center) / sigma)))
    return np.vstack(rows) if rows else np.empty((0, wavelengths.size), dtype=float)


def _normalize_basis_row(row: np.ndarray) -> np.ndarray | None:
    if not np.isfinite(row).all():
        return None
    scale = float(np.max(np.abs(row)))
    if scale <= 0.0:
        return None
    return row / scale


def _distribution(values: np.ndarray, *, clip_abs: float) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"mean": 0.0, "std": 0.0, "q05": 0.0, "q95": 0.0, "clip_abs": float(clip_abs)}
    q01 = _quantile(finite, 0.01, default=0.0)
    q05 = _quantile(finite, 0.05, default=0.0)
    q25 = _quantile(finite, 0.25, default=0.0)
    q50 = _quantile(finite, 0.50, default=0.0)
    q75 = _quantile(finite, 0.75, default=0.0)
    q95 = _quantile(finite, 0.95, default=0.0)
    q99 = _quantile(finite, 0.99, default=0.0)
    quantile_probabilities = np.linspace(0.0, 1.0, 21)
    quantile_values = np.quantile(finite, quantile_probabilities)
    return {
        "mean": float(np.clip(np.mean(finite), -clip_abs, clip_abs)),
        "std": float(np.clip(np.std(finite), 0.0, clip_abs)),
        "q01": float(np.clip(q01, -clip_abs, clip_abs)),
        "q05": float(np.clip(q05, -clip_abs, clip_abs)),
        "q25": float(np.clip(q25, -clip_abs, clip_abs)),
        "median": float(np.clip(q50, -clip_abs, clip_abs)),
        "q75": float(np.clip(q75, -clip_abs, clip_abs)),
        "q95": float(np.clip(q95, -clip_abs, clip_abs)),
        "q99": float(np.clip(q99, -clip_abs, clip_abs)),
        "quantile_probabilities": [float(value) for value in quantile_probabilities.tolist()],
        "quantile_values": [
            float(np.clip(value, -clip_abs, clip_abs))
            for value in quantile_values.tolist()
        ],
        "clip_abs": float(clip_abs),
    }


def _integer_distribution(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=int)
    if finite.size == 0:
        return {"mean": 0.0, "std": 0.0, "values": [0], "probabilities": [1.0]}
    unique, counts = np.unique(finite, return_counts=True)
    probabilities = counts.astype(float) / float(np.sum(counts))
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": int(np.min(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "median": float(np.median(finite)),
        "q75": float(np.quantile(finite, 0.75)),
        "max": int(np.max(finite)),
        "values": [int(value) for value in unique.tolist()],
        "probabilities": [float(value) for value in probabilities.tolist()],
    }


def _quantile(values: np.ndarray, q: float, *, default: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(default)
    return float(np.quantile(finite, q))


def _median_spacing(wavelengths: np.ndarray) -> float:
    diffs = np.diff(wavelengths)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def _has_scipy_signal() -> bool:
    return importlib.util.find_spec("scipy.signal") is not None


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
