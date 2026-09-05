"""Paired-spectrum and repetition distances for exploratory analysis.

Definitions originate in Studio 0.9.1 ``shared/metrics_computer.py``. The
historical small-sample approximations remain available, but are explicitly
identified by ``effective_metric`` and diagnostics instead of mislabeled.
This module neither trains predictive models nor imports Studio.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nirs4all.analysis.playground_statistics import distance_statistics, spectral_matrix

DISTANCE_METRICS = frozenset({
    "euclidean", "manhattan", "cosine", "spectral_angle", "correlation", "mahalanobis", "pca_distance",
})


def _paired_values(left: NDArray[np.float64], right: NDArray[np.float64], metric: str) -> NDArray[np.float64]:
    """Evaluate the non-fitted historical paired definitions."""
    if metric == "euclidean":
        return np.asarray(np.linalg.norm(left - right, axis=1))
    if metric == "manhattan":
        return np.asarray(np.sum(np.abs(left - right), axis=1))
    if metric == "cosine":
        from scipy.spatial.distance import cdist

        return np.asarray([cdist(a[None], b[None], "cosine")[0, 0] for a, b in zip(left, right, strict=True)])
    if metric == "spectral_angle":
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1) + 1e-10
        return np.asarray(np.arccos(np.clip(np.sum(left * right, axis=1) / denominator, -1, 1)))
    if metric == "correlation":
        with np.errstate(invalid="ignore", divide="ignore"):
            correlations = [np.corrcoef(a, b)[0, 1] for a, b in zip(left, right, strict=True)]
        return np.asarray([1 - value if not np.isnan(value) else 1.0 for value in correlations])
    raise ValueError(f"Unknown unfitted metric: {metric}")


def paired_spectral_distances(
    reference: ArrayLike, final: ArrayLike, *, metric: str = "euclidean", scale: str = "linear",
) -> dict[str, Any]:
    """Compute row-aligned before/after distances and their actual provenance.

    ``mahalanobis`` uses Ledoit-Wolf on combined observations when at least
    ``n_features + 2`` pairs exist. The historical smaller-sample Euclidean
    approximation is explicit. PCA uses at most ten components of the combined
    data, capped at ``n_pairs - 1``; a single pair similarly reports Euclidean.
    Unexpected estimator failures propagate: they never trigger another metric.
    Cosine of a zero vector remains undefined, not a fabricated zero distance.
    """
    left = spectral_matrix(reference, name="reference", finite=True)
    right = spectral_matrix(final, name="final", finite=True)
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: reference {left.shape} != final {right.shape}")
    if metric not in DISTANCE_METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    if scale not in {"linear", "log"}:
        raise ValueError(f"Unknown distance scale: {scale}")
    effective = metric
    diagnostics: list[dict[str, Any]] = []
    if metric == "mahalanobis" and left.shape[0] < left.shape[1] + 2:
        effective = "euclidean"
        diagnostics.append({"code": "historical_small_sample_approximation", "required_pairs": left.shape[1] + 2})
    if metric == "pca_distance" and left.shape[0] == 1:
        effective = "euclidean"
        diagnostics.append({"code": "historical_single_pair_approximation"})
    if effective == "mahalanobis":
        from sklearn.covariance import LedoitWolf

        precision = LedoitWolf().fit(np.vstack([left, right])).precision_
        if not np.isfinite(precision).all():
            raise ValueError("Mahalanobis precision is non-finite")
        differences = left - right
        distances = np.sqrt(np.sum(differences @ precision * differences, axis=1))
        if not np.isfinite(distances).all():
            raise ValueError("Mahalanobis distance is non-finite")
    elif effective == "pca_distance":
        from sklearn.decomposition import PCA

        components = min(10, left.shape[1], left.shape[0] - 1)
        projection = PCA(n_components=components).fit(np.vstack([left, right]))
        distances = np.linalg.norm(projection.transform(left) - projection.transform(right), axis=1)
    else:
        distances = _paired_values(left, right, effective)
    if effective == "correlation":
        constant = (np.ptp(left, axis=1) == 0) | (np.ptp(right, axis=1) == 0)
        if constant.any():
            diagnostics.append({"code": "historical_constant_correlation_distance_one", "sample_indices": np.flatnonzero(constant).tolist()})
    if scale == "log":
        distances = np.log1p(distances)
    undefined = np.flatnonzero(~np.isfinite(distances)).tolist()
    if undefined:
        diagnostics.append({"code": "undefined_distance", "sample_indices": undefined})
    return {
        "metric": metric, "effective_metric": effective, "scale": scale,
        "distances": distances, "sample_indices": list(range(left.shape[0])),
        "statistics": distance_statistics(distances), "diagnostics": diagnostics,
    }


def _repetition_values(spectra: NDArray[np.float64], references: NDArray[np.float64], metric: str) -> NDArray[np.float64]:
    """Repetition mode has distinct historical zero-vector/angle conventions."""
    if metric not in {"cosine", "spectral_angle"}:
        return _paired_values(spectra, references, metric)
    norm_s = np.linalg.norm(spectra, axis=1)
    norm_r = np.linalg.norm(references, axis=1)
    valid = (norm_s >= 1e-10) & (norm_r >= 1e-10)
    values = np.zeros(len(spectra))
    cosine = np.sum(spectra[valid] * references[valid], axis=1) / (norm_s[valid] * norm_r[valid])
    values[valid] = 1 - cosine if metric == "cosine" else np.arccos(np.clip(cosine, -1, 1))
    return values


def repetition_variance(
    values: ArrayLike, group_ids: ArrayLike, *, reference: str = "group_mean", metric: str = "euclidean",
) -> dict[str, Any]:
    """Preserve historical sorted-group ordering and original sample indices.

    Singletons are excluded. ``selected`` retains the old group-mean alias,
    explicitly recorded. The historically unsupported repetition Mahalanobis
    and PCA metrics report the actual Euclidean calculation, not a false name.
    """
    spectra = spectral_matrix(values, finite=True)
    groups = np.asarray(group_ids)
    if groups.ndim != 1 or len(groups) != len(spectra):
        raise ValueError("group_ids must be one-dimensional and match the number of samples")
    if metric not in DISTANCE_METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    if reference not in {"group_mean", "first", "leave_one_out", "selected"}:
        raise ValueError(f"Unknown repetition reference: {reference}")
    effective_metric = "euclidean" if metric in {"mahalanobis", "pca_distance"} else metric
    effective_reference = "group_mean" if reference == "selected" else reference
    diagnostics: list[dict[str, Any]] = []
    if metric != effective_metric:
        diagnostics.append({"code": "historical_repetition_metric_approximation"})
    if reference != effective_reference:
        diagnostics.append({"code": "historical_selected_reference_alias"})
    distances: list[float] = []
    sample_indices: list[int] = []
    labels: list[str] = []
    per_group: dict[str, Any] = {}
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        if len(indices) < 2:
            continue
        selected = spectra[indices]
        if effective_reference == "leave_one_out":
            references = np.asarray([np.delete(selected, index, axis=0).mean(axis=0) for index in range(len(selected))])
        else:
            center = selected[0] if effective_reference == "first" else selected.mean(axis=0)
            references = np.broadcast_to(center, selected.shape)
        computed = _repetition_values(selected, references, effective_metric)
        if effective_metric in {"cosine", "spectral_angle"}:
            zeros = (np.linalg.norm(selected, axis=1) < 1e-10) | (np.linalg.norm(references, axis=1) < 1e-10)
            if zeros.any():
                diagnostics.append({"code": "historical_zero_vector_distance_zero", "sample_indices": indices[zeros].tolist()})
        if effective_metric == "correlation":
            constant = (np.ptp(selected, axis=1) == 0) | (np.ptp(references, axis=1) == 0)
            if constant.any():
                diagnostics.append({"code": "historical_constant_correlation_distance_one", "sample_indices": indices[constant].tolist()})
        distances.extend(computed.tolist())
        sample_indices.extend(indices.tolist())
        labels.extend([str(group)] * len(indices))
        stats = distance_statistics(computed)
        per_group[str(group)] = {key: stats[key] for key in ("mean", "std", "max", "count")}
    array = np.asarray(distances)
    return {
        "reference": reference, "effective_reference": effective_reference,
        "metric": metric, "effective_metric": effective_metric,
        "distances": array, "sample_indices": sample_indices, "group_ids": labels,
        "quantiles": distance_statistics(array)["quantiles"], "per_group": per_group,
        "n_groups": len(per_group), "diagnostics": diagnostics,
    }
