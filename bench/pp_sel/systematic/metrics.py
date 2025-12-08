"""Unsupervised metrics for preprocessing evaluation."""

from typing import Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


def compute_pca_metrics(X: np.ndarray, n_components: int = 10) -> Dict[str, float]:
    """Compute PCA-based metrics.

    Args:
        X: Input data matrix (n_samples, n_features).
        n_components: Number of PCA components to compute.

    Returns:
        Dictionary with variance_ratio, first_component_ratio, effective_dim.
    """
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    if n_comp < 1:
        return {"variance_ratio": 0.0, "first_component_ratio": 1.0, "effective_dim": 0.0}

    pca = PCA(n_components=n_comp)
    pca.fit(X)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    first_ratio = pca.explained_variance_ratio_[0]

    # Effective dimensionality (entropy-based)
    var_ratios = pca.explained_variance_ratio_
    var_ratios = var_ratios[var_ratios > 1e-10]
    entropy = -np.sum(var_ratios * np.log(var_ratios + 1e-10))
    effective_dim = np.exp(entropy)

    return {
        "variance_ratio": float(cumvar[-1]),
        "first_component_ratio": float(first_ratio),
        "effective_dim": float(effective_dim),
    }


def compute_snr(X: np.ndarray) -> float:
    """Compute signal-to-noise ratio.

    Args:
        X: Input data matrix (n_samples, n_features).

    Returns:
        Signal-to-noise ratio value.
    """
    signal_var = np.var(np.mean(X, axis=0))
    noise_var = np.mean(np.var(X, axis=0))
    if noise_var < 1e-10:
        return 100.0
    return float(signal_var / noise_var)


def compute_roughness(X: np.ndarray) -> float:
    """Compute spectral roughness (2nd derivative magnitude).

    Args:
        X: Input data matrix (n_samples, n_features).

    Returns:
        Mean absolute second derivative.
    """
    if X.shape[1] < 3:
        return 0.0
    d2 = np.diff(X, n=2, axis=1)
    return float(np.mean(np.abs(d2)))


def compute_separation(X: np.ndarray, n_samples: int = 100) -> float:
    """Compute inter-sample separation score.

    Args:
        X: Input data matrix (n_samples, n_features).
        n_samples: Maximum samples to use for distance computation.

    Returns:
        Normalized mean pairwise distance.
    """
    if X.shape[0] < 2:
        return 0.0

    # Subsample for speed
    if X.shape[0] > n_samples:
        idx = np.random.choice(X.shape[0], n_samples, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    # Compute pairwise distances
    dists = pairwise_distances(X_sub)

    # Mean distance (normalized)
    mean_dist = np.mean(dists[np.triu_indices(len(X_sub), k=1)])

    # Normalize by feature std
    std_norm = np.mean(np.std(X_sub, axis=0)) + 1e-10

    return float(mean_dist / std_norm)


def evaluate_unsupervised(X: np.ndarray) -> Dict[str, float]:
    """Compute all unsupervised metrics and total score.

    Args:
        X: Input data matrix (n_samples, n_features).

    Returns:
        Dictionary with all metrics and total_score.
    """
    pca_metrics = compute_pca_metrics(X)

    metrics = {
        "variance_ratio": pca_metrics["variance_ratio"],
        "effective_dim": pca_metrics["effective_dim"],
        "snr": compute_snr(X),
        "roughness": compute_roughness(X),
        "separation": compute_separation(X),
    }

    # Compute total score (normalized combination)
    # Higher is better for: variance_ratio, effective_dim, snr, separation
    # Lower is better for: roughness

    # Normalize roughness (invert and clip)
    roughness_score = 1.0 / (1.0 + metrics["roughness"])

    # Normalize SNR (log scale, clipped)
    snr_score = np.clip(np.log1p(metrics["snr"]) / 5.0, 0, 1)

    # Normalize effective_dim
    eff_dim_score = np.clip(metrics["effective_dim"] / 10.0, 0, 1)

    # Total score
    metrics["total_score"] = (
        0.25 * metrics["variance_ratio"]
        + 0.25 * eff_dim_score
        + 0.20 * snr_score
        + 0.15 * roughness_score
        + 0.15 * np.clip(metrics["separation"] / 10.0, 0, 1)
    )

    return metrics
