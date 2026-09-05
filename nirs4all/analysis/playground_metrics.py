"""Historical Playground descriptors owned by the scientific library.

These are per-spectrum exploration descriptors, not predictive-model scores.
The 0.9.1 definitions are retained; missing results and cleaning conventions are
reported explicitly. Chemometric calculations delegate to existing filters.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import linregress

from nirs4all.analysis.playground_types import PreviewBatch, positive_count
from nirs4all.operators.filters import HighLeverageFilter, XOutlierFilter

METRIC_CATEGORIES = {
    "amplitude": ("global_min", "global_max", "dynamic_range", "mean_intensity"),
    "energy": ("l2_norm", "rms_energy", "auc", "abs_auc"),
    "shape": ("baseline_slope", "baseline_offset", "peak_count", "peak_prominence_max"),
    "noise": ("hf_variance", "snr_estimate", "smoothness"),
    "quality": ("nan_count", "inf_count", "saturation_count", "zero_count"),
    "chemometric": ("hotelling_t2", "q_residual", "leverage", "distance_to_centroid", "lof_score"),
}
ALL_METRICS = tuple(name for names in METRIC_CATEGORIES.values() for name in names)
FAST_METRICS = (*METRIC_CATEGORIES["amplitude"], *METRIC_CATEGORIES["energy"],
                *METRIC_CATEGORIES["quality"], *METRIC_CATEGORIES["noise"])


def _chemometric(x: np.ndarray, metric: str, options: dict[str, Any]) -> np.ndarray:
    if len(x) < 2:
        raise ValueError("Chemometric descriptors require at least two observations")
    components = min(positive_count(options.get("n_pca_components", 5), "n_pca_components"), len(x) - 1, x.shape[1])
    if metric == "distance_to_centroid":
        maximum = positive_count(options.get("max_covariance_cells", 100_000_000), "max_covariance_cells")
        if x.shape[1] ** 2 > maximum:
            raise ValueError("Descriptor covariance exceeds host budget; raise max_covariance_cells explicitly")
    # Historical cleaning is preserved and exposed by compute_descriptors.
    clean = np.nan_to_num(x, nan=0)
    if metric == "leverage":
        owner = HighLeverageFilter(method="pca" if x.shape[1] > len(x) else "hat", n_components=min(components, 50))
        owner.fit(clean)
        return np.asarray(owner.get_leverages(clean))
    methods: dict[str, Literal["pca_leverage", "pca_residual", "mahalanobis", "lof"]] = {
        "hotelling_t2": "pca_leverage", "q_residual": "pca_residual", "distance_to_centroid": "mahalanobis", "lof_score": "lof",
    }
    filter_owner = XOutlierFilter(method=methods[metric], n_components=components,
                                  contamination=options.get("lof_contamination", 0.1))
    filter_owner.fit(clean)
    if filter_owner._distances_ is None:
        raise ValueError("Filter owner returned no descriptor distances")
    return np.asarray(filter_owner._distances_.copy())


def _one(batch: PreviewBatch, metric: str, options: dict[str, Any]) -> np.ndarray:
    x = batch.x
    if metric == "global_min":
        return np.asarray(np.nanmin(x, axis=1))
    if metric == "global_max":
        return np.asarray(np.nanmax(x, axis=1))
    if metric == "dynamic_range":
        return np.asarray(np.nanmax(x, axis=1) - np.nanmin(x, axis=1))
    if metric == "mean_intensity":
        return np.asarray(np.nanmean(x, axis=1))
    if metric == "l2_norm":
        return np.asarray(np.linalg.norm(x, axis=1))
    if metric == "rms_energy":
        return np.asarray(np.sqrt(np.nanmean(x**2, axis=1)))
    if metric in {"auc", "abs_auc"}:
        # NumPy>=2 is the package minimum. trapezoid is the non-deprecated
        # equivalent of the old np.trapz definition, removed in NumPy2.4.
        return np.asarray(np.trapezoid(np.abs(x) if metric == "abs_auc" else x, batch.wavelengths, axis=1))
    if metric in {"baseline_slope", "baseline_offset"}:
        coordinate = np.arange(x.shape[1])
        values = np.zeros(len(x))
        for index, row in enumerate(x):
            valid = ~np.isnan(row)
            if valid.sum() > 1:
                fit = linregress(coordinate[valid], row[valid])
                values[index] = fit.slope if metric == "baseline_slope" else fit.intercept
        return values
    if metric in {"peak_count", "peak_prominence_max"}:
        values = np.zeros(len(x))
        for index, row in enumerate(x):
            clean = np.nan_to_num(row, nan=0)
            kwargs = {"prominence": np.std(clean) * 0.5} if metric == "peak_count" else {}
            peaks, _ = find_peaks(clean, **kwargs)
            values[index] = len(peaks) if metric == "peak_count" else (float(np.max(peak_prominences(clean, peaks)[0])) if len(peaks) else 0)
        return values
    if metric == "hf_variance":
        return np.asarray(np.nanvar(np.diff(x, axis=1), axis=1))
    if metric == "snr_estimate":
        noise = np.nanstd(x, axis=1)
        noise[noise == 0] = 1e-10
        return np.asarray(np.abs(np.nanmean(x, axis=1)) / noise)
    if metric == "smoothness":
        variance = np.nanvar(np.diff(x, axis=1), axis=1)
        variance[variance == 0] = 1e-10
        return np.asarray(1 / variance)
    if metric == "nan_count":
        return np.asarray(np.isnan(x).sum(axis=1), dtype=float)
    if metric == "inf_count":
        return np.asarray(np.isinf(x).sum(axis=1), dtype=float)
    if metric == "saturation_count":
        threshold = options.get("saturation_threshold")
        if threshold is None:
            threshold = np.nanmax(x) * 0.99
        return np.asarray((x >= threshold).sum(axis=1), dtype=float)
    if metric == "zero_count":
        return np.asarray((x == 0).sum(axis=1), dtype=float)
    if metric in METRIC_CATEGORIES["chemometric"]:
        return _chemometric(x, metric, options)
    raise ValueError(f"Unknown spectral descriptor: {metric}")


def descriptor_statistics(values: np.ndarray) -> dict[str, Any]:
    """Historical nan-excluding summary with explicit observation counts."""
    valid = values[~np.isnan(values)]
    result: dict[str, Any] = {"count": len(values), "valid_count": len(valid), "nan_count": int(np.isnan(values).sum()),
                              "inf_count": int(np.isinf(values).sum())}
    names = ("min", "max", "mean", "std", "p5", "p25", "p50", "p75", "p95")
    if not len(valid):
        return {**result, **dict.fromkeys(names, 0.0)}
    result.update({"min": float(np.min(valid)), "max": float(np.max(valid)),
                   "mean": float(np.mean(valid)), "std": float(np.std(valid))})
    result.update({f"p{percentile}": float(np.percentile(valid, percentile)) for percentile in (5, 25, 50, 75, 95)})
    return result


def compute_descriptors(batch: PreviewBatch, *, pca: dict[str, Any] | None = None,
                        options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute requested descriptors, returning errors rather than dropping them.

    The historical filter calculations fit their own PCA; supplied display PCA
    is not silently reused as a different scientific definition. Baseline slope
    uses feature positions; AUC uses the actual current axis and its direction.
    """
    del pca
    options = dict(options or {})
    requested = options.get("metrics") or list(FAST_METRICS)
    if not isinstance(requested, (tuple, list)) or any(not isinstance(metric, str) for metric in requested):
        raise ValueError("metrics must be a list of descriptor names")
    values, statistics, errors = {}, {}, {}
    diagnostics: list[dict[str, Any]] = []
    for metric in dict.fromkeys(requested):
        try:
            calculated = _one(batch, metric, options)
            if calculated.ndim != 1 or len(calculated) != len(batch.x):
                raise ValueError("Descriptor does not match current sample rows")
            values[metric] = calculated.tolist()
            statistics[metric] = descriptor_statistics(calculated)
            if not np.isfinite(calculated).all():
                diagnostics.append({"metric": metric, "code": "non_finite_descriptor",
                                    "sample_indices": np.flatnonzero(~np.isfinite(calculated)).tolist()})
            if metric in (*METRIC_CATEGORIES["chemometric"], "peak_count", "peak_prominence_max") and not np.isfinite(batch.x).all():
                diagnostics.append({"metric": metric, "code": "historical_nan_to_num_input_policy"})
        except Exception as error:
            errors[metric] = str(error)
    return {"values": values, "statistics": statistics, "computed_metrics": list(values),
            "available_metrics": list(METRIC_CATEGORIES), "available_metric_names": list(ALL_METRICS),
            "n_samples": len(batch.x), "sample_origins": batch.origins, "errors": errors, "diagnostics": diagnostics,
            "definitions": {"baseline_axis": "feature_index", "auc_axis_kind": batch.axis_kind,
                            "auc_axis_unit": batch.header_unit, "zero_noise_denominator": 1e-10,
                            "nan_summary_policy": "exclude_nan_and_report_count", "chemometric_owner": "nirs4all.operators.filters"}}
