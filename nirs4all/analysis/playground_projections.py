"""Numerical projections and repetition presentations for spectral previews.

All charts have structured coordinates/statistics; decimation applies only to
returned drawing arrays and never to PCA, repetitions or full-axis statistics.
"""

from __future__ import annotations

import importlib.util
from collections import defaultdict
from typing import Any

import numpy as np

from nirs4all.analysis.playground_statistics import spectral_statistics
from nirs4all.analysis.playground_types import PreviewBatch, positive_count
from nirs4all.analysis.projections import compute_pca_projection
from nirs4all.data.repetition_detection import auto_detect_repetition_column, detect_repetition_groups


def pca_projection(batch: PreviewBatch, folds: dict[str, Any] | None = None) -> dict[str, Any]:
    """Owner PCA, preserving the historical variance and coloring contract."""
    owner = compute_pca_projection(batch.x, max_components=10, variance_threshold=0.999)
    result = {key: owner[key] for key in ("coordinates", "explained_variance_ratio", "explained_variance", "n_components")}
    result["n_components_999"] = owner["n_components_threshold"]
    if batch.y is not None:
        result["y"] = batch.y.tolist()
    if folds is not None:
        result["fold_labels"] = folds["fold_labels"]
    return result


def umap_projection(batch: PreviewBatch, folds: dict[str, Any] | None = None, *,
                    n_neighbors: int = 15, min_dist: float = 0.1, n_components: int = 2) -> dict[str, Any]:
    """Explicit optional UMAP owner, no alternate projection on failure."""
    if importlib.util.find_spec("umap") is None:
        return {"error": "UMAP not available. Install umap-learn.", "available": False}
    if len(batch.x) < 10:
        return {"error": f"UMAP requires at least 10 samples, got {len(batch.x)}", "available": True}
    positive_count(n_neighbors, "n_neighbors")
    positive_count(n_components, "n_components")
    neighbors = min(max(2, n_neighbors), len(batch.x) - 1)
    components = min(max(2, n_components), 3)
    import umap

    coordinates = umap.UMAP(n_components=components, n_neighbors=neighbors, min_dist=min_dist,
                            random_state=42, n_jobs=-1).fit_transform(batch.x)
    result = {"coordinates": coordinates.tolist(), "n_components": components,
              "params": {"n_neighbors": neighbors, "min_dist": min_dist}, "available": True}
    if batch.y is not None:
        result["y"] = batch.y.tolist()
    if folds is not None:
        result["fold_labels"] = folds["fold_labels"]
    return result


def _reference_distances(coordinates: np.ndarray, metric: str, max_covariance_cells: int) -> tuple[np.ndarray, str]:
    """Historical first reference for pairs, mean for groups of >=3."""
    reference = coordinates[0] if len(coordinates) == 2 else coordinates.mean(axis=0)
    if metric == "mahalanobis" and len(coordinates) > 2:
        positive_count(max_covariance_cells, "max_covariance_cells")
        if coordinates.shape[1] ** 2 > max_covariance_cells:
            raise ValueError("Repetition covariance exceeds host budget; raise max_covariance_cells explicitly")
        from scipy.spatial.distance import mahalanobis

        covariance = np.atleast_2d(np.cov(coordinates, rowvar=False))
        covariance += np.eye(covariance.shape[0]) * 1e-6
        inverse = np.linalg.inv(covariance)
        return np.asarray([mahalanobis(row, reference, inverse) for row in coordinates]), "mahalanobis"
    return np.asarray(np.linalg.norm(coordinates - reference, axis=1)), "euclidean"


def repetition_projection(batch: PreviewBatch, *, pca: dict[str, Any] | None = None,
                          umap: dict[str, Any] | None = None, options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Owner grouping plus historical variability with explicit effective space."""
    options = dict(options or {})
    column = options.get("bio_sample_column") or options.get("dataset_repetition")
    pattern = options.get("bio_sample_pattern")
    auto = options.get("auto_detect_repetitions", True)
    metric = options.get("distance_metric", "pca")
    if metric not in {"pca", "umap", "euclidean", "mahalanobis"}:
        raise ValueError(f"Unknown repetition distance metric: {metric}")
    ids = [f"Sample_{index}" if value is None else str(value) for index, value in enumerate(batch.sample_ids)] if batch.sample_ids is not None else [f"Sample_{index}" for index in range(len(batch.x))]
    groups: dict[str, list[int]] = defaultdict(list)
    if column and column in batch.metadata:
        for index, value in enumerate(batch.metadata[column]):
            groups[str(value)].append(index)
    elif auto and batch.metadata:
        column = auto_detect_repetition_column({key: values.tolist() for key, values in batch.metadata.items()})
        if column:
            for index, value in enumerate(batch.metadata[column]):
                groups[str(value)].append(index)
    elif pattern or auto:
        detection = detect_repetition_groups(ids, pattern=pattern) if pattern else detect_repetition_groups(ids)
        groups.update(detection.groups)
    repeated = {group: indices for group, indices in groups.items() if len(indices) >= 2}
    if not repeated:
        return {"has_repetitions": False, "n_bio_samples": len(groups) or len(batch.x), "n_with_reps": 0,
                "detected_pattern": pattern, "message": "No biological samples with repetitions found."}
    projection = pca if metric == "pca" else umap if metric == "umap" else None
    effective_space = metric if projection and "coordinates" in projection else "spectra"
    coordinates = np.asarray(projection["coordinates"]) if projection is not None and effective_space != "spectra" else batch.x
    if coordinates.ndim != 2 or len(coordinates) != len(batch.x):
        raise ValueError("Repetition projection coordinates do not match current sample rows")
    diagnostics: list[dict[str, Any]] = []
    if metric in {"pca", "umap"} and effective_space == "spectra":
        diagnostics.append({"code": "historical_unavailable_projection_uses_spectral_space", "requested_space": metric})
    points: list[dict[str, Any]] = []
    for group, indices in repeated.items():
        distances, effective_metric = _reference_distances(coordinates[indices], metric, options.get("max_covariance_cells", 100_000_000))
        if metric == "mahalanobis" and effective_metric != metric:
            diagnostics.append({"code": "historical_pair_uses_euclidean", "bio_sample": group})
        y_mean = float(np.mean(batch.y[indices])) if batch.y is not None else None
        for repetition, (index, distance) in enumerate(zip(indices, distances, strict=True)):
            points.append({"bio_sample": group, "rep_index": repetition, "sample_index": index,
                           "sample_id": ids[index], "distance": float(distance),
                           "y": float(batch.y[index]) if batch.y is not None else None, "y_mean": y_mean})
    distances = np.asarray([point["distance"] for point in points])
    if not np.isfinite(distances).all():
        diagnostics.append({"code": "undefined_repetition_distance",
                            "sample_indices": [point["sample_index"] for point in points if not np.isfinite(point["distance"])]})
    threshold = float(np.percentile(distances, 95))
    return {"has_repetitions": True, "n_bio_samples": len(groups), "n_with_reps": len(repeated),
            "n_singletons": len(groups) - len(repeated), "total_repetitions": len(points),
            "distance_metric": metric, "effective_space": effective_space, "detected_pattern": pattern,
            "data": points, "statistics": {"mean_distance": float(np.mean(distances)), "max_distance": float(np.max(distances)),
                                            "std_distance": float(np.std(distances)), "p95_distance": threshold},
            "high_variability_samples": [point for point in points if point["distance"] > threshold][:10],
            "bio_sample_groups": dict(list(repeated.items())[:50]), "diagnostics": diagnostics}


def display_indices(axis: np.ndarray, spectra: np.ndarray, target_points: int | None) -> np.ndarray:
    """Historical LTTB point selection; no numerical pipeline data is modified."""
    size = len(axis)
    if target_points is None or target_points == 0:
        return np.arange(size)
    positive_count(target_points, "max_wavelengths_returned")
    if size <= target_points or target_points < 3:
        return np.arange(size)
    mean = spectra.mean(axis=0)
    selected = np.empty(target_points, dtype=np.intp)
    selected[0], selected[-1] = 0, size - 1
    bucket = (size - 2) / (target_points - 2)
    previous = 0
    for index in range(1, target_points - 1):
        start = int(1 + (index - 1) * bucket)
        end = min(int(1 + index * bucket), size - 1)
        following = min(int(1 + (index + 1) * bucket), size - 1)
        if end >= following:
            following = min(end + 1, size)
        avg_x, avg_y = np.mean(axis[end:following]), np.mean(mean[end:following])
        area = np.abs((axis[previous] - avg_x) * (mean[start:end] - mean[previous])
                      - (axis[previous] - axis[start:end]) * (avg_y - mean[previous]))
        previous = start + int(np.argmax(area))
        selected[index] = previous
    return selected


def spectral_payload(batch: PreviewBatch, *, compute_statistics: bool = True,
                     max_wavelengths: int | None = None, indices: np.ndarray | None = None) -> dict[str, Any]:
    """Drawing arrays plus full-shape statistics and textual identity context."""
    selected = display_indices(batch.wavelengths, batch.x, max_wavelengths) if indices is None else indices
    stats = spectral_statistics(batch.x) if compute_statistics and len(batch.x) else None
    return {"spectra": batch.x[:, selected], "wavelengths": batch.wavelengths[selected].tolist(),
            "sample_indices": batch.origins, "shape": list(batch.x.shape), "statistics": stats,
            "header_unit": batch.header_unit, "axis_kind": batch.axis_kind, "sample_ids": batch.sample_ids,
            "metadata": batch.metadata or None, "y": batch.y, "sample_partitions": batch.partitions,
            "display_feature_indices": selected.tolist(), "statistics_scope": "full_processed_matrix"}
