"""
X-based outlier filter for sample filtering.

This module provides the XOutlierFilter class for detecting and excluding
samples with outlier feature (X) values using various statistical and
machine learning methods commonly used in spectroscopy and chemometrics.
"""

from typing import Optional, Dict, Any, Literal
import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.decomposition import PCA

from .base import SampleFilter


class XOutlierFilter(SampleFilter):
    """
    Filter samples with outlier spectral features.

    This filter identifies samples whose X-values (spectra) are statistical
    outliers using various detection methods. It's commonly used to remove
    samples with corrupted, unusual, or non-representative spectra.

    Supported methods:
    - "mahalanobis": Mahalanobis distance from center (default)
    - "robust_mahalanobis": Robust Mahalanobis using MinCovDet (resistant to outliers)
    - "pca_residual": Q-statistic (residual) from PCA reconstruction
    - "pca_leverage": T² (Hotelling's T-squared) in PCA score space
    - "isolation_forest": Isolation Forest anomaly detection
    - "lof": Local Outlier Factor

    Attributes:
        method (str): Outlier detection method
        threshold (float): Detection threshold (method-specific)
        n_components (int): Number of PCA components for PCA-based methods
        contamination (float): Expected proportion of outliers for sklearn methods

    Example:
        >>> from nirs4all.operators.filters import XOutlierFilter
        >>>
        >>> # Mahalanobis distance (default)
        >>> filter_maha = XOutlierFilter(method="mahalanobis", threshold=3.0)
        >>>
        >>> # Robust Mahalanobis (better with outliers in training data)
        >>> filter_robust = XOutlierFilter(method="robust_mahalanobis", threshold=3.0)
        >>>
        >>> # PCA-based residual (Q-statistic)
        >>> filter_pca = XOutlierFilter(method="pca_residual", n_components=10)
        >>>
        >>> # Fit and get mask
        >>> filter_maha.fit(X_train)
        >>> mask = filter_maha.get_mask(X_train)  # True = keep

    In Pipeline:
        >>> pipeline = [
        ...     {
        ...         "sample_filter": {
        ...             "filters": [XOutlierFilter(method="mahalanobis", threshold=3.0)],
        ...         }
        ...     },
        ...     "snv",
        ...     "model:PLSRegression",
        ... ]
    """

    def __init__(
        self,
        method: Literal[
            "mahalanobis", "robust_mahalanobis", "pca_residual",
            "pca_leverage", "isolation_forest", "lof"
        ] = "mahalanobis",
        threshold: Optional[float] = None,
        n_components: Optional[int] = None,
        contamination: float = 0.1,
        random_state: Optional[int] = None,
        support_fraction: Optional[float] = None,
        reason: Optional[str] = None
    ):
        """
        Initialize the X outlier filter.

        Args:
            method: Outlier detection method:
                   - "mahalanobis": Mahalanobis distance. Uses threshold as
                                   number of standard deviations. Default threshold=3.0.
                   - "robust_mahalanobis": Robust version using MinCovDet estimator.
                                          More resistant to outliers in training data.
                   - "pca_residual": Q-statistic (squared reconstruction error).
                                    Uses n_components for PCA. Threshold is auto-computed
                                    or can be set manually.
                   - "pca_leverage": Hotelling's T² in PCA score space.
                                    High leverage samples in reduced space.
                   - "isolation_forest": Isolation Forest anomaly detection.
                                        Uses contamination parameter.
                   - "lof": Local Outlier Factor. Uses contamination parameter.
            threshold: Detection threshold. If None, uses method-specific defaults:
                      - mahalanobis/robust_mahalanobis: 3.0 (3 std deviations)
                      - pca_residual/pca_leverage: Auto-computed from chi-squared distribution
                      - isolation_forest/lof: Uses contamination parameter instead
            n_components: Number of PCA components for PCA-based methods and for
                         dimensionality reduction in Mahalanobis methods.
                         If None, uses min(n_samples, n_features, 10) for PCA methods
                         and min(n_samples - 1, n_features, 20) for Mahalanobis methods.
            contamination: Expected proportion of outliers for isolation_forest and lof.
                          Must be in (0, 0.5]. Default 0.1.
            random_state: Random state for reproducibility (isolation_forest, MinCovDet).
            support_fraction: Fraction of data to use for MinCovDet in robust_mahalanobis.
                             Higher values are faster but less robust. Default None uses
                             sklearn's default ((n_samples + n_features + 1) / 2n_samples).
                             Set to 0.9 for faster computation with slightly less robustness.
            reason: Custom exclusion reason. Defaults to filter description.

        Raises:
            ValueError: If method is not one of the supported methods.
            ValueError: If contamination is not in valid range.
        """
        super().__init__(reason=reason)
        self.method = method
        self.threshold = threshold
        self.n_components = n_components
        self.contamination = contamination
        self.random_state = random_state
        self.support_fraction = support_fraction

        # Validate parameters
        valid_methods = (
            "mahalanobis", "robust_mahalanobis", "pca_residual",
            "pca_leverage", "isolation_forest", "lof"
        )
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

        if not (0 < contamination <= 0.5):
            raise ValueError(
                f"contamination must be in (0, 0.5], got {contamination}"
            )

        # Fitted attributes (set during fit)
        self.threshold_: Optional[float] = None
        self.center_: Optional[np.ndarray] = None
        self.covariance_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None
        self.pca_: Optional[PCA] = None
        self.detector_: Optional[Any] = None  # For sklearn detectors
        self._distances_: Optional[np.ndarray] = None  # For stats

    @property
    def exclusion_reason(self) -> str:
        """Get descriptive exclusion reason."""
        if self.reason is not None:
            return self.reason
        return f"x_outlier_{self.method}"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "XOutlierFilter":
        """
        Compute outlier detection model from training data.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Target array (not used for X-based filtering, but kept for API consistency).

        Returns:
            self: The fitted filter instance.

        Raises:
            ValueError: If X has insufficient samples for the chosen method.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        # Handle empty or single sample case
        if n_samples == 0:
            # Set minimal state so get_mask returns empty array
            self.threshold_ = float('inf')
            self.center_ = np.zeros(n_features) if n_features > 0 else np.array([])
            self.precision_ = np.eye(n_features) if n_features > 0 else np.array([[1.0]])
            return self

        if n_samples == 1:
            # Single sample: cannot compute statistics, keep all samples
            self.threshold_ = float('inf')
            self.center_ = X[0].copy()
            self._distances_ = np.array([0.0])
            self.precision_ = np.eye(n_features) if n_features > 0 else np.array([[1.0]])
            return self

        if n_samples < 2:
            raise ValueError(f"Need at least 2 samples, got {n_samples}")

        if self.method in ("mahalanobis", "robust_mahalanobis"):
            self._fit_mahalanobis(X)
        elif self.method in ("pca_residual", "pca_leverage"):
            self._fit_pca(X)
        elif self.method == "isolation_forest":
            self._fit_isolation_forest(X)
        elif self.method == "lof":
            self._fit_lof(X)

        return self

    def _fit_mahalanobis(self, X: np.ndarray) -> None:
        """Fit Mahalanobis distance-based detection."""
        n_samples, n_features = X.shape

        # Handle high-dimensional case: reduce dimensions if needed
        # Use fewer components for robust_mahalanobis since MinCovDet is O(n³)
        if self.n_components is not None:
            max_components = self.n_components
        elif self.method == "robust_mahalanobis":
            max_components = 20  # Reduced for speed with MinCovDet
        else:
            max_components = 50

        if n_features > n_samples - 1 or n_features > max_components:
            n_comp = min(n_samples - 1, n_features, max_components)
            pca = PCA(n_components=n_comp)
            X_reduced = pca.fit_transform(X)
            self.pca_ = pca
        else:
            X_reduced = X
            self.pca_ = None

        if self.method == "robust_mahalanobis":
            # Use MinCovDet for robust estimation
            try:
                cov_estimator = MinCovDet(
                    random_state=self.random_state,
                    support_fraction=self.support_fraction
                )
                cov_estimator.fit(X_reduced)
            except ValueError:
                # Fall back to empirical if MinCovDet fails
                cov_estimator = EmpiricalCovariance()
                cov_estimator.fit(X_reduced)
        else:
            cov_estimator = EmpiricalCovariance()
            cov_estimator.fit(X_reduced)

        self.center_ = cov_estimator.location_
        self.covariance_ = cov_estimator.covariance_
        self.precision_ = cov_estimator.precision_

        # Compute distances for threshold calculation
        distances = cov_estimator.mahalanobis(X_reduced)
        self._distances_ = np.sqrt(distances)  # Convert to standard deviations

        # Set threshold
        if self.threshold is not None:
            self.threshold_ = self.threshold
        else:
            self.threshold_ = 3.0  # Default: 3 standard deviations

    def _fit_pca(self, X: np.ndarray) -> None:
        """Fit PCA-based detection (residual or leverage)."""
        n_samples, n_features = X.shape

        # Determine number of components
        if self.n_components is not None:
            n_comp = min(self.n_components, n_samples, n_features)
        else:
            n_comp = min(n_samples, n_features, 10)

        self.pca_ = PCA(n_components=n_comp)
        self.pca_.fit(X)
        self.center_ = self.pca_.mean_

        if self.method == "pca_residual":
            # Compute Q-statistic (squared reconstruction error)
            X_reconstructed = self.pca_.inverse_transform(self.pca_.transform(X))
            residuals = X - X_reconstructed
            self._distances_ = np.sum(residuals ** 2, axis=1)

            # Set threshold using percentile or provided value
            if self.threshold is not None:
                self.threshold_ = self.threshold
            else:
                # Use 95th percentile of training Q-statistics
                self.threshold_ = np.percentile(self._distances_, 95)

        else:  # pca_leverage (Hotelling's T²)
            scores = self.pca_.transform(X)
            # Normalize by explained variance
            variances = self.pca_.explained_variance_
            t_squared = np.sum((scores ** 2) / variances, axis=1)
            self._distances_ = t_squared

            # Set threshold using F-distribution approximation or percentile
            if self.threshold is not None:
                self.threshold_ = self.threshold
            else:
                # Use 95th percentile of training T² values
                self.threshold_ = float(np.percentile(t_squared, 95))

    def _fit_isolation_forest(self, X: np.ndarray) -> None:
        """Fit Isolation Forest detector."""
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise ImportError(
                "IsolationForest requires sklearn. Install with: pip install scikit-learn"
            )

        self.detector_ = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.detector_.fit(X)

        # Store scores for stats
        self._distances_ = -self.detector_.score_samples(X)  # Negative = higher anomaly

    def _fit_lof(self, X: np.ndarray) -> None:
        """Fit Local Outlier Factor detector."""
        try:
            from sklearn.neighbors import LocalOutlierFactor
        except ImportError:
            raise ImportError(
                "LocalOutlierFactor requires sklearn. Install with: pip install scikit-learn"
            )

        n_samples = X.shape[0]
        n_neighbors = min(20, n_samples - 1)

        self.detector_ = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            novelty=True,  # Enable predict for new data
            n_jobs=-1
        )
        self.detector_.fit(X)

        # Store scores for stats
        self._distances_ = -self.detector_.score_samples(X)

    def get_mask(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute boolean mask indicating which samples to KEEP.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Target array (not used, kept for API consistency).

        Returns:
            np.ndarray: Boolean array of shape (n_samples,) where:
                       - True means KEEP the sample (not an outlier)
                       - False means EXCLUDE the sample (outlier detected)

        Raises:
            ValueError: If filter has not been fitted.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.method in ("mahalanobis", "robust_mahalanobis"):
            return self._get_mask_mahalanobis(X)
        elif self.method in ("pca_residual", "pca_leverage"):
            return self._get_mask_pca(X)
        elif self.method in ("isolation_forest", "lof"):
            return self._get_mask_sklearn_detector(X)

        raise ValueError(f"Unknown method: {self.method}")

    def _get_mask_mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """Get mask using Mahalanobis distance."""
        if self.precision_ is None:
            raise ValueError("Filter has not been fitted. Call fit() first.")

        # Apply PCA if used during fitting
        if self.pca_ is not None:
            X_reduced = self.pca_.transform(X)
        else:
            X_reduced = X

        # Compute Mahalanobis distances
        diff = X_reduced - self.center_
        left = np.dot(diff, self.precision_)
        mahal_sq = np.sum(left * diff, axis=1)
        distances = np.sqrt(np.maximum(mahal_sq, 0))

        return distances <= self.threshold_

    def _get_mask_pca(self, X: np.ndarray) -> np.ndarray:
        """Get mask using PCA-based metrics."""
        if self.pca_ is None:
            raise ValueError("Filter has not been fitted. Call fit() first.")

        if self.method == "pca_residual":
            X_reconstructed = self.pca_.inverse_transform(self.pca_.transform(X))
            residuals = X - X_reconstructed
            q_stats = np.sum(residuals ** 2, axis=1)
            return q_stats <= self.threshold_

        else:  # pca_leverage
            scores = self.pca_.transform(X)
            variances = self.pca_.explained_variance_
            t_squared = np.sum((scores ** 2) / variances, axis=1)
            return t_squared <= self.threshold_

    def _get_mask_sklearn_detector(self, X: np.ndarray) -> np.ndarray:
        """Get mask using sklearn detector (IsolationForest, LOF)."""
        if self.detector_ is None:
            raise ValueError("Filter has not been fitted. Call fit() first.")

        # predict returns 1 for inliers, -1 for outliers
        predictions = self.detector_.predict(X)
        return predictions == 1

    def get_filter_stats(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Get statistics about filter application.

        Args:
            X: Feature array.
            y: Target array (unused).

        Returns:
            Dict containing:
                - Base stats (n_samples, n_excluded, n_kept, exclusion_rate)
                - method: Detection method used
                - threshold: Threshold value (if applicable)
                - n_components: PCA components (if applicable)
                - distance_stats: Statistics on computed distances/scores
        """
        base_stats = super().get_filter_stats(X, y)

        base_stats.update({
            "method": self.method,
            "threshold": self.threshold_,
            "n_components": self.pca_.n_components_ if self.pca_ is not None else None,
        })

        if self._distances_ is not None:
            base_stats["distance_stats"] = {
                "min": float(np.min(self._distances_)),
                "max": float(np.max(self._distances_)),
                "mean": float(np.mean(self._distances_)),
                "median": float(np.median(self._distances_)),
                "std": float(np.std(self._distances_)),
            }

        if self.method in ("isolation_forest", "lof"):
            base_stats["contamination"] = self.contamination

        return base_stats

    def __repr__(self) -> str:
        """Return string representation."""
        params = [f"method='{self.method}'"]

        if self.threshold is not None:
            params.append(f"threshold={self.threshold}")
        if self.n_components is not None:
            params.append(f"n_components={self.n_components}")
        if self.method in ("isolation_forest", "lof"):
            params.append(f"contamination={self.contamination}")

        return f"{self.__class__.__name__}({', '.join(params)})"
