"""Distance metrics for diversity analysis.

This module provides multiple distance/similarity metrics for comparing
preprocessed data representations. Metrics are grouped into two categories:

1. **Subspace-based metrics**: Compare the principal subspaces of the data.
   - Grassmann distance: Angular distance between PCA subspaces.
   - CKA distance: Centered Kernel Alignment (representation similarity).

2. **Geometry-based metrics**: Compare the sample distributions.
   - RV coefficient distance: Correlation structure similarity.
   - Procrustes distance: Alignment-based shape similarity.
   - Trustworthiness distance: Neighborhood preservation.
   - Covariance distance: Distribution shape similarity.
"""

from typing import Dict, Tuple

import numpy as np
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def _center(X: np.ndarray) -> np.ndarray:
    """Center data by subtracting the mean."""
    return X - X.mean(axis=0, keepdims=True)


def _get_pca_components(X: np.ndarray, n_components: int = 5) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute PCA and return scores, loadings, and explained variance ratio.

    Args:
        X: Input data (n_samples, n_features).
        n_components: Number of components.

    Returns:
        Tuple of (Z scores, U loadings, explained variance ratio).
    """
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    if n_comp < 1:
        return np.zeros((X.shape[0], 1)), np.zeros((X.shape[1], 1)), 0.0

    Xc = _center(X)
    pca = PCA(n_components=n_comp, random_state=0)
    Z = pca.fit_transform(Xc)
    U = pca.components_.T
    evr = float(pca.explained_variance_ratio_.sum())
    return Z, U, evr


# =============================================================================
# SUBSPACE-BASED METRICS
# =============================================================================

def compute_grassmann_distance(
    X1: np.ndarray, X2: np.ndarray, n_components: int = 5
) -> float:
    """Compute Grassmann distance between PCA subspaces of two datasets.

    The Grassmann distance measures the angular distance between linear
    subspaces. Lower distance = more similar preprocessing effects on
    the principal subspace structure.

    Args:
        X1: First dataset (n_samples, n_features).
        X2: Second dataset (n_samples, n_features).
        n_components: Number of PCA components for subspace.

    Returns:
        Normalized Grassmann distance in [0, 1].
    """
    n_comp = min(
        n_components, X1.shape[0] - 1, X1.shape[1], X2.shape[0] - 1, X2.shape[1]
    )
    if n_comp < 1:
        return 1.0

    pca1 = PCA(n_components=n_comp)
    pca2 = PCA(n_components=n_comp)

    pca1.fit(X1)
    pca2.fit(X2)

    # Get principal components (column space)
    U1 = pca1.components_.T  # (features, n_comp)
    U2 = pca2.components_.T

    # Compute principal angles via SVD
    M = U1.T @ U2
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1, 1)

    # Grassmann distance = sqrt(sum of squared angles)
    angles = np.arccos(s)
    distance = np.sqrt(np.sum(angles**2))

    # Normalize to [0, 1]
    max_dist = np.sqrt(n_comp * (np.pi / 2) ** 2)
    return float(distance / max_dist)


def compute_cka_distance(X1: np.ndarray, X2: np.ndarray) -> float:
    """Compute CKA-based distance (1 - CKA similarity).

    CKA (Centered Kernel Alignment) measures the similarity of representations.
    It is invariant to orthogonal transformations and isotropic scaling.

    Args:
        X1: First dataset (n_samples, n_features).
        X2: Second dataset (n_samples, n_features).

    Returns:
        CKA distance (1 - similarity) in [0, 1].
    """

    def centering_matrix(n):
        return np.eye(n) - np.ones((n, n)) / n

    n = X1.shape[0]
    if n != X2.shape[0]:
        min_n = min(n, X2.shape[0])
        X1, X2 = X1[:min_n], X2[:min_n]
        n = min_n

    H = centering_matrix(n)

    # Linear kernels
    K1 = X1 @ X1.T
    K2 = X2 @ X2.T

    # Center kernels
    K1c = H @ K1 @ H
    K2c = H @ K2 @ H

    # HSIC
    hsic = np.sum(K1c * K2c) / (n - 1) ** 2

    # Normalize
    hsic1 = np.sum(K1c * K1c) / (n - 1) ** 2
    hsic2 = np.sum(K2c * K2c) / (n - 1) ** 2

    if hsic1 < 1e-10 or hsic2 < 1e-10:
        return 1.0

    cka = hsic / np.sqrt(hsic1 * hsic2)

    return float(1.0 - cka)


def compute_rv_distance(X1: np.ndarray, X2: np.ndarray) -> float:
    """Compute RV coefficient-based distance (1 - RV similarity).

    The RV coefficient is a multivariate generalization of the squared
    Pearson correlation. It measures the correlation between two
    configuration matrices.

    Args:
        X1: First dataset (n_samples, n_features).
        X2: Second dataset (n_samples, n_features).

    Returns:
        RV distance (1 - similarity) in [0, 1].
    """
    X1c = _center(X1)
    X2c = _center(X2)

    # Gram matrices
    A = X1c @ X1c.T
    B = X2c @ X2c.T

    num = np.trace(A @ B)
    den = np.sqrt(np.trace(A @ A) * np.trace(B @ B))

    if den < 1e-10:
        return 1.0

    rv = num / den
    return float(1.0 - rv)


# =============================================================================
# GEOMETRY-BASED METRICS
# =============================================================================

def compute_procrustes_distance(
    X1: np.ndarray, X2: np.ndarray, n_components: int = 5
) -> float:
    """Compute Procrustes distance between PCA projections.

    Procrustes analysis finds the optimal translation, rotation, and
    uniform scaling to align two point configurations. The Procrustes
    distance measures the residual after alignment.

    Args:
        X1: First dataset (n_samples, n_features).
        X2: Second dataset (n_samples, n_features).
        n_components: Number of PCA components to use.

    Returns:
        Procrustes disparity in [0, 1] (normalized).
    """
    Z1, _, _ = _get_pca_components(X1, n_components)
    Z2, _, _ = _get_pca_components(X2, n_components)

    # Ensure same number of components
    min_comp = min(Z1.shape[1], Z2.shape[1])
    Z1 = Z1[:, :min_comp]
    Z2 = Z2[:, :min_comp]

    try:
        _, _, disparity = procrustes(Z1, Z2)
        return float(np.clip(disparity, 0, 1))
    except Exception:
        return 1.0


def compute_trustworthiness_distance(
    X1: np.ndarray, X2: np.ndarray, n_components: int = 5, k: int = 10
) -> float:
    """Compute trustworthiness-based distance.

    Trustworthiness measures how well the neighborhood structure is
    preserved when projecting from one representation to another.
    Lower trustworthiness = more different neighborhood structures.

    Args:
        X1: Reference dataset (n_samples, n_features).
        X2: Comparison dataset (n_samples, n_features).
        n_components: Number of PCA components.
        k: Number of neighbors to consider.

    Returns:
        Trustworthiness distance (1 - trustworthiness) in [0, 1].
    """
    Z1, _, _ = _get_pca_components(X1, n_components)
    Z2, _, _ = _get_pca_components(X2, n_components)

    n = Z1.shape[0]
    k = max(2, min(k, n - 2))

    try:
        # Get all neighbors
        nn1 = NearestNeighbors(n_neighbors=n - 1).fit(Z1).kneighbors(return_distance=False)
        nn2 = NearestNeighbors(n_neighbors=n - 1).fit(Z2).kneighbors(return_distance=False)

        # Compute ranks in reference space
        ranks = np.zeros((n, n), dtype=int)
        for i in range(n):
            ranks[i, nn1[i]] = np.arange(n - 1)

        # Trustworthiness computation
        s = 0.0
        for i in range(n):
            Ui = set(nn2[i, 1:1 + k])  # k-neighbors in Z2
            Ki = set(nn1[i, 1:1 + k])  # k-neighbors in Z1
            for v in Ui - Ki:
                s += ranks[i, v] - (k - 1)

        Z_norm = n * k * (2 * n - 3 * k - 1) / 2
        trustworthiness = 1.0 - (2.0 / Z_norm) * s if Z_norm > 0 else 0.0

        return float(1.0 - np.clip(trustworthiness, 0, 1))
    except Exception:
        return 0.5  # Return neutral value on error


def compute_covariance_distance(X1: np.ndarray, X2: np.ndarray) -> float:
    """Compute covariance structure distance.

    Measures the difference in covariance matrices using the Frobenius norm.
    This captures differences in the shape and spread of distributions.

    Args:
        X1: First dataset (n_samples, n_features).
        X2: Second dataset (n_samples, n_features).

    Returns:
        Normalized Frobenius distance of covariance matrices.
    """
    X1c = _center(X1)
    X2c = _center(X2)

    # Compute covariances (on samples for efficiency if high-dimensional)
    if X1.shape[1] > X1.shape[0]:
        # High-dimensional: use Gram matrices
        cov1 = X1c @ X1c.T / (X1.shape[0] - 1)
        cov2 = X2c @ X2c.T / (X2.shape[0] - 1)
    else:
        cov1 = np.cov(X1c.T)
        cov2 = np.cov(X2c.T)

    # Handle 1D case
    if cov1.ndim == 0:
        cov1 = np.array([[cov1]])
    if cov2.ndim == 0:
        cov2 = np.array([[cov2]])

    # Frobenius norm of difference, normalized
    diff_norm = np.linalg.norm(cov1 - cov2, 'fro')
    scale = np.sqrt(np.linalg.norm(cov1, 'fro') * np.linalg.norm(cov2, 'fro'))

    if scale < 1e-10:
        return 1.0

    return float(np.clip(diff_norm / scale, 0, 1))


# =============================================================================
# COMBINED METRICS
# =============================================================================

def compute_all_distances(
    X1: np.ndarray, X2: np.ndarray, n_components: int = 5, k_neighbors: int = 10
) -> Dict[str, float]:
    """Compute all distance metrics between two datasets.

    Args:
        X1: First dataset (n_samples, n_features).
        X2: Second dataset (n_samples, n_features).
        n_components: Number of PCA components for subspace metrics.
        k_neighbors: Number of neighbors for trustworthiness.

    Returns:
        Dictionary with all distance metrics and combined scores.
    """
    # Subspace-based metrics
    grassmann = compute_grassmann_distance(X1, X2, n_components)
    cka = compute_cka_distance(X1, X2)
    rv = compute_rv_distance(X1, X2)

    # Geometry-based metrics
    procrustes_dist = compute_procrustes_distance(X1, X2, n_components)
    trustworthiness = compute_trustworthiness_distance(X1, X2, n_components, k_neighbors)
    covariance = compute_covariance_distance(X1, X2)

    # Combined scores
    # Subspace distance: emphasizes representation structure
    subspace_distance = 0.4 * grassmann + 0.4 * cka + 0.2 * rv

    # Geometry distance: emphasizes sample distribution
    geometry_distance = 0.4 * procrustes_dist + 0.3 * trustworthiness + 0.3 * covariance

    # Overall combined distance
    combined_distance = 0.5 * subspace_distance + 0.5 * geometry_distance

    return {
        # Individual metrics
        "grassmann": grassmann,
        "cka": cka,
        "rv": rv,
        "procrustes": procrustes_dist,
        "trustworthiness": trustworthiness,
        "covariance": covariance,
        # Combined scores
        "subspace_distance": subspace_distance,
        "geometry_distance": geometry_distance,
        "combined_distance": combined_distance,
    }
