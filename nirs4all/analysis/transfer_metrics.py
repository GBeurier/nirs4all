"""
Transfer Metrics Computation.

This module provides fast, optimized computation of transfer-focused metrics
between two datasets in PCA space. Metrics are designed to assess how well
preprocessing aligns datasets for transfer learning scenarios.

Metrics computed:
- Centroid Distance: Euclidean distance between dataset centroids in PCA space
- CKA (Centered Kernel Alignment): Representation similarity
- Grassmann Distance: Angular distance between PCA subspaces
- RV Coefficient: Multivariate correlation structure
- Procrustes Disparity: Shape alignment after optimal transformation
- Trustworthiness: Neighborhood preservation
- Spread Distance: Distribution overlap combining covariance and sample distances
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import subspace_angles
from scipy.spatial import procrustes
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


@dataclass
class TransferMetrics:
    """Container for transfer metrics between two datasets."""

    centroid_distance: float
    cka_similarity: float
    grassmann_distance: float
    rv_coefficient: float
    procrustes_disparity: float
    trustworthiness: float
    spread_distance: float
    evr_source: float
    evr_target: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "centroid_distance": self.centroid_distance,
            "cka_similarity": self.cka_similarity,
            "grassmann_distance": self.grassmann_distance,
            "rv_coefficient": self.rv_coefficient,
            "procrustes_disparity": self.procrustes_disparity,
            "trustworthiness": self.trustworthiness,
            "spread_distance": self.spread_distance,
            "evr_source": self.evr_source,
            "evr_target": self.evr_target,
        }


class TransferMetricsComputer:
    """
    Fast computation of transfer metrics between two datasets.

    Key optimization: Computes PCA once per dataset, then reuses
    for all metric computations.

    Args:
        n_components: Number of PCA components for projection.
        k_neighbors: Number of neighbors for trustworthiness computation.
        random_state: Random state for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 10,
        k_neighbors: int = 10,
        random_state: int = 0,
    ):
        self.n_components = n_components
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def compute(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        compute_trust: bool = True,
    ) -> TransferMetrics:
        """
        Compute all transfer metrics between two datasets.

        Args:
            X_source: Source dataset (n_samples_src, n_features).
            X_target: Target dataset (n_samples_tgt, n_features).
            compute_trust: Whether to compute trustworthiness (slower).

        Returns:
            TransferMetrics containing all computed metrics.
        """
        # Compute PCA for both datasets
        Z_src, U_src, evr_src = self._pca(X_source)
        Z_tgt, U_tgt, evr_tgt = self._pca(X_target)

        # Align component dimensions
        r_use = min(Z_src.shape[1], Z_tgt.shape[1])
        Z_src = Z_src[:, :r_use]
        Z_tgt = Z_tgt[:, :r_use]
        U_src = U_src[:, :r_use]
        U_tgt = U_tgt[:, :r_use]

        # Compute all metrics
        centroid_dist = self._centroid_distance(Z_src, Z_tgt)
        cka = self._cka(Z_src, Z_tgt)
        rv = self._rv(Z_src, Z_tgt)
        procrustes_disp = self._procrustes(Z_src, Z_tgt)
        spread_dist = self._spread_distance(Z_src, Z_tgt)

        # Grassmann requires same feature dimension
        if U_src.shape[0] == U_tgt.shape[0]:
            grassmann = self._grassmann(U_src, U_tgt)
        else:
            grassmann = np.nan

        # Trustworthiness is more expensive
        if compute_trust:
            trust = self._trustworthiness(Z_src, Z_tgt)
        else:
            trust = np.nan

        return TransferMetrics(
            centroid_distance=centroid_dist,
            cka_similarity=cka,
            grassmann_distance=grassmann,
            rv_coefficient=rv,
            procrustes_disparity=procrustes_disp,
            trustworthiness=trust,
            spread_distance=spread_dist,
            evr_source=evr_src,
            evr_target=evr_tgt,
        )

    def compute_raw_and_preprocessed(
        self,
        X_source_raw: np.ndarray,
        X_target_raw: np.ndarray,
        X_source_pp: np.ndarray,
        X_target_pp: np.ndarray,
        compute_trust: bool = True,
    ) -> Tuple[TransferMetrics, TransferMetrics, Dict[str, float]]:
        """
        Compute metrics for both raw and preprocessed data, plus improvement.

        Args:
            X_source_raw: Raw source dataset.
            X_target_raw: Raw target dataset.
            X_source_pp: Preprocessed source dataset.
            X_target_pp: Preprocessed target dataset.
            compute_trust: Whether to compute trustworthiness.

        Returns:
            Tuple of (raw_metrics, pp_metrics, improvements_dict)
        """
        raw_metrics = self.compute(X_source_raw, X_target_raw, compute_trust)
        pp_metrics = self.compute(X_source_pp, X_target_pp, compute_trust)

        improvements = self._compute_improvements(raw_metrics, pp_metrics)

        return raw_metrics, pp_metrics, improvements

    def _pca(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute PCA projection.

        Args:
            X: Input data (n_samples, n_features).

        Returns:
            Tuple of (scores, loadings, explained_variance_ratio).
        """
        X_centered = self._center(X)
        n_components = min(self.n_components, X_centered.shape[1], X_centered.shape[0])
        pca = PCA(n_components=n_components, random_state=self.random_state)
        Z = pca.fit_transform(X_centered)
        U = pca.components_.T
        evr = float(pca.explained_variance_ratio_.sum())
        return Z, U, evr

    @staticmethod
    def _center(X: np.ndarray) -> np.ndarray:
        """Center data by subtracting mean."""
        return X - X.mean(axis=0, keepdims=True)

    @staticmethod
    def _centroid_distance(Z1: np.ndarray, Z2: np.ndarray) -> float:
        """Compute Euclidean distance between centroids in PCA space."""
        centroid_1 = Z1.mean(axis=0)
        centroid_2 = Z2.mean(axis=0)
        return float(np.linalg.norm(centroid_1 - centroid_2))

    def _cka(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute Centered Kernel Alignment (CKA) similarity.

        CKA measures representation similarity via kernel alignment.
        For datasets with different sample sizes, we compute CKA on
        covariance structures rather than sample-level gram matrices.

        Higher values (closer to 1) indicate more similar representations.
        """
        Xc = self._center(X)
        Yc = self._center(Y)

        # Compute covariance matrices
        Cov_X = (Xc.T @ Xc) / (X.shape[0] - 1)
        Cov_Y = (Yc.T @ Yc) / (Y.shape[0] - 1)

        # CKA via covariance alignment
        numerator = np.linalg.norm(Cov_X @ Cov_Y, "fro") ** 2
        denom = np.linalg.norm(Cov_X @ Cov_X, "fro") * np.linalg.norm(Cov_Y @ Cov_Y, "fro")
        return float(numerator / denom) if denom > 0 else np.nan

    def _rv(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute RV coefficient (multivariate correlation).

        RV coefficient measures correlation structure similarity via
        covariance matrices. Works with different sample sizes.

        Higher values (closer to 1) indicate more similar structures.
        """
        Xc = self._center(X)
        Yc = self._center(Y)

        # Compute covariance matrices
        Cov_X = (Xc.T @ Xc) / (X.shape[0] - 1)
        Cov_Y = (Yc.T @ Yc) / (Y.shape[0] - 1)

        # RV coefficient on covariances
        numerator = np.trace(Cov_X @ Cov_Y)
        denominator = np.sqrt(np.trace(Cov_X @ Cov_X) * np.trace(Cov_Y @ Cov_Y))
        return float(numerator / denominator) if denominator > 0 else np.nan

    @staticmethod
    def _grassmann(U: np.ndarray, V: np.ndarray) -> float:
        """
        Compute Grassmann distance between subspaces.

        Grassmann distance is the sum of squared principal angles between subspaces.
        Lower values indicate more aligned subspaces.
        """
        theta = subspace_angles(U, V)
        return float(np.sqrt((theta**2).sum()))

    @staticmethod
    def _procrustes(Z1: np.ndarray, Z2: np.ndarray) -> float:
        """
        Compute Procrustes disparity after optimal alignment.

        For different sample sizes, we use the minimum sample count.
        Lower values indicate better geometric alignment.
        """
        # Use first 2 components for 2D Procrustes
        n_dims = min(2, Z1.shape[1], Z2.shape[1])
        # Take same number of samples from each
        n_samples = min(Z1.shape[0], Z2.shape[0])
        _, _, disparity = procrustes(Z1[:n_samples, :n_dims], Z2[:n_samples, :n_dims])
        return float(disparity)

    def _trustworthiness(
        self, Z_source: np.ndarray, Z_target: np.ndarray
    ) -> float:
        """
        Compute trustworthiness score.

        Measures how well neighborhood structure is preserved between
        source and target representations. Higher is better.

        Note: This computes how well neighborhoods in source are preserved
        when projecting into target space context.
        """
        n = min(Z_source.shape[0], Z_target.shape[0])
        k = max(2, min(self.k_neighbors, n - 2))

        # Take aligned samples (assumes same ordering)
        Z_src = Z_source[:n]
        Z_tgt = Z_target[:n]

        nn_src = NearestNeighbors(n_neighbors=n - 1).fit(Z_src)
        nn_tgt = NearestNeighbors(n_neighbors=n - 1).fit(Z_tgt)

        idx_src = nn_src.kneighbors(return_distance=False)
        idx_tgt = nn_tgt.kneighbors(return_distance=False)

        # Compute ranks in source space
        ranks = np.zeros((n, n), dtype=int)
        for i in range(n):
            ranks[i, idx_src[i]] = np.arange(n - 1)

        # Compute trustworthiness penalty
        penalty = 0.0
        for i in range(n):
            Ui = set(idx_tgt[i, 1:1 + k])  # k-neighbors in target
            Ki = set(idx_src[i, 1:1 + k])  # k-neighbors in source
            for v in Ui - Ki:
                penalty += ranks[i, v] - (k - 1)

        Z = n * k * (2 * n - 3 * k - 1) / 2
        return 1.0 - (2.0 / Z) * penalty if Z > 0 else np.nan

    def _spread_distance(
        self, Z1: np.ndarray, Z2: np.ndarray, n_samples: int = 100
    ) -> float:
        """
        Compute spread distance combining covariance and sample-wise distances.

        Measures how well the distributions overlap in PCA space.
        Lower values indicate better overlap.
        """
        # Covariance-based distance
        cov1 = np.cov(Z1.T)
        cov2 = np.cov(Z2.T)

        # Handle 1D case
        if cov1.ndim == 0:
            cov1 = np.array([[cov1]])
            cov2 = np.array([[cov2]])

        cov_dist = np.linalg.norm(cov1 - cov2, "fro")

        # Sample-wise distance (subsample for efficiency)
        n1 = min(n_samples, Z1.shape[0])
        n2 = min(n_samples, Z2.shape[0])

        rng = np.random.RandomState(self.random_state)
        idx1 = rng.choice(Z1.shape[0], n1, replace=False)
        idx2 = rng.choice(Z2.shape[0], n2, replace=False)

        Z1_sample = Z1[idx1]
        Z2_sample = Z2[idx2]

        # Compute pairwise distances and average minimum distance
        # For asymmetric sample sizes, compute mean of min distances separately
        dists = cdist(Z1_sample, Z2_sample, metric="euclidean")
        min_dist_1 = np.mean(dists.min(axis=1))  # Min dist from each Z1 sample to Z2
        min_dist_2 = np.mean(dists.min(axis=0))  # Min dist from each Z2 sample to Z1
        min_dist = (min_dist_1 + min_dist_2) / 2

        return float(cov_dist + min_dist)

    def _compute_improvements(
        self, raw: TransferMetrics, pp: TransferMetrics
    ) -> Dict[str, float]:
        """
        Compute improvement percentages for each metric.

        For distance metrics (lower is better): positive improvement means reduction.
        For similarity metrics (higher is better): positive improvement means increase.
        """
        eps = 1e-10
        improvements = {}

        # Distance metrics (lower is better) - positive improvement = reduction
        for metric in ["centroid_distance", "grassmann_distance", "procrustes_disparity", "spread_distance"]:
            raw_val = getattr(raw, metric)
            pp_val = getattr(pp, metric)
            if not np.isnan(raw_val) and not np.isnan(pp_val):
                improvements[f"{metric}_improvement"] = (raw_val - pp_val) / (raw_val + eps)
            else:
                improvements[f"{metric}_improvement"] = np.nan

        # Similarity metrics (higher is better) - positive improvement = increase
        for metric in ["cka_similarity", "rv_coefficient", "trustworthiness"]:
            raw_val = getattr(raw, metric)
            pp_val = getattr(pp, metric)
            if not np.isnan(raw_val) and not np.isnan(pp_val):
                improvements[f"{metric}_improvement"] = (pp_val - raw_val) / (abs(raw_val) + eps)
            else:
                improvements[f"{metric}_improvement"] = np.nan

        # EVR preservation (we want to maintain variance)
        raw_evr = raw.evr_source
        pp_evr = pp.evr_source
        if not np.isnan(raw_evr) and not np.isnan(pp_evr):
            improvements["evr_preservation"] = pp_evr / (raw_evr + eps)
        else:
            improvements["evr_preservation"] = np.nan

        return improvements


def compute_transfer_score(
    metrics: TransferMetrics,
    raw_metrics: Optional[TransferMetrics] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute a composite transfer score from metrics.

    Higher scores indicate better transfer potential.

    Args:
        metrics: TransferMetrics from preprocessed data.
        raw_metrics: Optional baseline metrics for computing improvements.
        weights: Optional custom weights for metric combination.

    Returns:
        Composite transfer score (0-1 scale, higher is better).
        Returns NaN if critical metrics are invalid.
    """
    if weights is None:
        # Default weights prioritizing centroid distance and CKA
        weights = {
            "centroid": 0.40,
            "cka": 0.30,
            "spread": 0.20,
            "evr": 0.10,
        }

    # Check for invalid critical metrics
    critical_metrics = [metrics.centroid_distance, metrics.spread_distance, metrics.evr_source]
    if any(np.isnan(m) or np.isinf(m) for m in critical_metrics):
        return float('nan')

    score = 0.0
    eps = 1e-10

    if raw_metrics is not None:
        # Use improvement-based scoring

        # Centroid improvement (reduction is good)
        # Handle case where raw centroid is very small (already well-aligned)
        raw_centroid = raw_metrics.centroid_distance
        if raw_centroid < eps:
            # Baseline already optimal, penalize any increase
            centroid_improv = -abs(metrics.centroid_distance) if metrics.centroid_distance > eps else 0.0
        else:
            centroid_improv = (raw_centroid - metrics.centroid_distance) / (raw_centroid + eps)
        score += weights.get("centroid", 0.4) * np.clip(centroid_improv, -1, 1)

        # CKA (higher is better, so we use raw value)
        cka_score = metrics.cka_similarity if not np.isnan(metrics.cka_similarity) else 0
        score += weights.get("cka", 0.3) * cka_score

        # Spread improvement (reduction is good)
        raw_spread = raw_metrics.spread_distance
        if raw_spread < eps:
            spread_improv = -abs(metrics.spread_distance) if metrics.spread_distance > eps else 0.0
        else:
            spread_improv = (raw_spread - metrics.spread_distance) / (raw_spread + eps)
        score += weights.get("spread", 0.2) * np.clip(spread_improv, -1, 1)

        # EVR preservation
        raw_evr = raw_metrics.evr_source
        if raw_evr < eps:
            evr_ratio = 1.0 if metrics.evr_source < eps else 0.0
        else:
            evr_ratio = metrics.evr_source / (raw_evr + eps)
        score += weights.get("evr", 0.1) * min(evr_ratio, 1.0)

    else:
        # Use absolute values (normalize to 0-1)
        # Lower distances and higher similarities are better
        cka_score = metrics.cka_similarity if not np.isnan(metrics.cka_similarity) else 0
        rv_score = metrics.rv_coefficient if not np.isnan(metrics.rv_coefficient) else 0

        score = 0.5 * cka_score + 0.3 * rv_score + 0.2 * metrics.evr_source

    return float(score)
