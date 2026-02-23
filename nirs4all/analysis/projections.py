"""Projection utilities for visualization.

Functions for computing low-dimensional projections of spectral data
for visualization purposes (not pipeline transformers).
"""

from __future__ import annotations

import numpy as np


def compute_pca_projection(
    X: np.ndarray,
    max_components: int = 10,
    variance_threshold: float = 0.999,
) -> dict:
    """Compute PCA projection for visualization.

    Fits PCA with up to ``max_components`` components (capped by n_samples and
    n_features), then reports how many components are needed to reach the given
    cumulative variance threshold.

    Args:
        X: 2D array (n_samples, n_features). Must have at least 2 samples.
        max_components: Upper limit on components. Capped by min(n_samples, n_features).
        variance_threshold: Cumulative variance ratio target (default 0.999).

    Returns:
        Dict with keys:
            coordinates: list[list[float]] — PCA coordinates (n_samples x n_components).
            explained_variance_ratio: list[float] — per-component variance ratios.
            explained_variance: list[float] — per-component absolute variances.
            n_components: int — actual number of components computed.
            n_components_threshold: int — components needed for ``variance_threshold``.

    Raises:
        ValueError: If X has fewer than 2 samples or 0 features.
    """
    from sklearn.decomposition import PCA

    n_samples, n_features = X.shape
    if n_samples < 2:
        raise ValueError(f"PCA requires at least 2 samples, got {n_samples}")
    if n_features == 0:
        raise ValueError("PCA requires at least 1 feature")

    n_comp = min(max_components, n_samples, n_features)
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_thresh = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)
    n_components_thresh = min(max(n_components_thresh, 3), n_comp)

    return {
        "coordinates": X_pca.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance": pca.explained_variance_.tolist(),
        "n_components": n_comp,
        "n_components_threshold": n_components_thresh,
    }
