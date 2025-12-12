"""
High leverage filter for sample filtering.

This module provides the HighLeverageFilter class for detecting and excluding
samples that have high leverage (influence) on model fitting.
"""

from typing import Optional, Dict, Any, Literal
import numpy as np
from sklearn.decomposition import PCA

from .base import SampleFilter


class HighLeverageFilter(SampleFilter):
    """
    Filter high-leverage samples that may unduly influence the model.

    High-leverage points are samples that are far from the center of the
    predictor space and can have a disproportionate effect on regression
    models. This filter identifies and excludes such samples.

    The leverage of a sample is computed from the hat matrix H = X(X'X)^(-1)X'.
    The diagonal elements h_ii represent the leverage of each sample.

    Supported methods:
    - "hat": Direct hat matrix diagonal computation
    - "pca": PCA-based leverage (for high-dimensional data)

    Common threshold guidelines:
    - 2 * p / n (where p = number of parameters, n = samples)
    - 3 * average leverage
    - Absolute threshold (e.g., 0.5)

    Attributes:
        method (str): Leverage computation method
        threshold_multiplier (float): Multiple of average leverage to use as threshold
        absolute_threshold (float): Absolute threshold (overrides multiplier if set)

    Example:
        >>> from nirs4all.operators.filters import HighLeverageFilter
        >>>
        >>> # Using multiplier of average leverage (default)
        >>> filter_obj = HighLeverageFilter(threshold_multiplier=2.0)
        >>>
        >>> # Using absolute threshold
        >>> filter_abs = HighLeverageFilter(absolute_threshold=0.5)
        >>>
        >>> # PCA-based for high-dimensional data
        >>> filter_pca = HighLeverageFilter(method="pca", n_components=10)
        >>>
        >>> # Fit and get mask
        >>> filter_obj.fit(X_train)
        >>> mask = filter_obj.get_mask(X_train)  # True = keep

    In Pipeline:
        >>> pipeline = [
        ...     {
        ...         "sample_filter": {
        ...             "filters": [HighLeverageFilter(threshold_multiplier=2.0)],
        ...         }
        ...     },
        ...     "snv",
        ...     "model:PLSRegression",
        ... ]
    """

    def __init__(
        self,
        method: Literal["hat", "pca"] = "hat",
        threshold_multiplier: float = 2.0,
        absolute_threshold: Optional[float] = None,
        n_components: Optional[int] = None,
        center: bool = True,
        reason: Optional[str] = None
    ):
        """
        Initialize the high leverage filter.

        Args:
            method: Leverage computation method:
                   - "hat": Direct hat matrix computation. Works well for
                           n_samples > n_features. For high-dimensional data,
                           automatically reduces via PCA.
                   - "pca": Compute leverage in PCA-reduced space. Useful for
                           high-dimensional spectral data.
            threshold_multiplier: Multiple of average leverage to use as threshold.
                                 Common values: 2.0 (lenient), 3.0 (strict).
                                 Average leverage = p/n where p=features, n=samples.
                                 Default 2.0.
            absolute_threshold: Absolute leverage threshold. If set, overrides
                              threshold_multiplier. Range (0, 1). Default None.
            n_components: Number of PCA components for "pca" method.
                         If None, uses min(n_samples-1, n_features, 50).
            center: Whether to center the data before computing leverage.
                   Default True (recommended).
            reason: Custom exclusion reason. Defaults to filter description.

        Raises:
            ValueError: If method is not valid.
            ValueError: If threshold_multiplier is not positive.
            ValueError: If absolute_threshold is not in (0, 1).
        """
        super().__init__(reason=reason)
        self.method = method
        self.threshold_multiplier = threshold_multiplier
        self.absolute_threshold = absolute_threshold
        self.n_components = n_components
        self.center = center

        # Validate parameters
        if method not in ("hat", "pca"):
            raise ValueError(f"method must be 'hat' or 'pca', got '{method}'")

        if threshold_multiplier <= 0:
            raise ValueError(
                f"threshold_multiplier must be positive, got {threshold_multiplier}"
            )

        if absolute_threshold is not None:
            if not (0 < absolute_threshold < 1):
                raise ValueError(
                    f"absolute_threshold must be in (0, 1), got {absolute_threshold}"
                )

        # Fitted attributes
        self.threshold_: Optional[float] = None
        self.mean_: Optional[np.ndarray] = None
        self.pca_: Optional[PCA] = None
        self.precision_: Optional[np.ndarray] = None
        self.n_effective_features_: Optional[int] = None
        self._leverages_: Optional[np.ndarray] = None

    @property
    def exclusion_reason(self) -> str:
        """Get descriptive exclusion reason."""
        if self.reason is not None:
            return self.reason
        return "high_leverage"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "HighLeverageFilter":
        """
        Compute leverage statistics from training data.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Target array (not used for leverage computation).

        Returns:
            self: The fitted filter instance.

        Raises:
            ValueError: If X has insufficient samples.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        # Handle edge cases
        if n_samples == 0:
            self.threshold_ = float('inf')
            self.mean_ = np.zeros(n_features) if n_features > 0 else np.array([])
            self.precision_ = np.eye(n_features) if n_features > 0 else np.array([[1.0]])
            return self

        if n_samples == 1:
            self.threshold_ = float('inf')
            self.mean_ = X[0].copy()
            self._leverages_ = np.array([1.0])  # Single sample has max leverage
            self.precision_ = np.eye(n_features) if n_features > 0 else np.array([[1.0]])
            return self

        if n_samples < 2:
            raise ValueError(f"Need at least 2 samples, got {n_samples}")

        # Center data if requested
        if self.center:
            self.mean_ = np.mean(X, axis=0)
            X_centered = X - self.mean_
        else:
            self.mean_ = np.zeros(n_features)
            X_centered = X

        if self.method == "pca" or n_features >= n_samples:
            # Use PCA for dimensionality reduction
            self._fit_pca(X_centered)
        else:
            # Direct hat matrix computation
            self._fit_hat(X_centered)

        # Compute leverages on training data
        self._leverages_ = self._compute_leverages(X_centered)

        # Set threshold
        if self.absolute_threshold is not None:
            self.threshold_ = self.absolute_threshold
        else:
            # Use multiplier of average leverage
            avg_leverage = float(np.mean(self._leverages_))
            self.threshold_ = self.threshold_multiplier * avg_leverage

        return self

    def _fit_hat(self, X: np.ndarray) -> None:
        """Fit for direct hat matrix computation."""
        # Compute (X'X)^(-1) using pseudo-inverse for numerical stability
        XtX = X.T @ X

        # Add small regularization for numerical stability
        reg = 1e-10 * np.trace(XtX) / XtX.shape[0]
        XtX_reg = XtX + reg * np.eye(XtX.shape[0])

        try:
            self.precision_ = np.linalg.inv(XtX_reg)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse
            self.precision_ = np.linalg.pinv(XtX)

        self.n_effective_features_ = X.shape[1]
        self.pca_ = None

    def _fit_pca(self, X: np.ndarray) -> None:
        """Fit using PCA-based leverage computation."""
        n_samples, n_features = X.shape

        # Determine number of components
        if self.n_components is not None:
            n_comp = min(self.n_components, n_samples - 1, n_features)
        else:
            n_comp = min(n_samples - 1, n_features, 50)

        self.pca_ = PCA(n_components=n_comp)
        scores = self.pca_.fit_transform(X)

        # Compute covariance in score space
        XtX = scores.T @ scores

        # Add regularization
        reg = 1e-10 * np.trace(XtX) / max(XtX.shape[0], 1)
        XtX_reg = XtX + reg * np.eye(XtX.shape[0])

        try:
            self.precision_ = np.linalg.inv(XtX_reg)
        except np.linalg.LinAlgError:
            self.precision_ = np.linalg.pinv(XtX)

        self.n_effective_features_ = n_comp

    def _compute_leverages(self, X: np.ndarray) -> np.ndarray:
        """Compute leverage values for samples."""
        # Transform to PCA space if using PCA
        if self.pca_ is not None:
            X_proj = self.pca_.transform(X)
        else:
            X_proj = X

        # Compute diagonal of hat matrix: h_ii = x_i' (X'X)^(-1) x_i
        # This is more efficient than computing the full hat matrix
        left = X_proj @ self.precision_
        leverages = np.sum(left * X_proj, axis=1)

        return leverages

    def get_mask(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute boolean mask indicating which samples to KEEP.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Target array (not used).

        Returns:
            np.ndarray: Boolean array of shape (n_samples,) where:
                       - True means KEEP the sample (low leverage)
                       - False means EXCLUDE the sample (high leverage)

        Raises:
            ValueError: If filter has not been fitted.
        """
        if self.precision_ is None:
            raise ValueError("Filter has not been fitted. Call fit() first.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Center data
        if self.center:
            X_centered = X - self.mean_
        else:
            X_centered = X

        # Compute leverages
        leverages = self._compute_leverages(X_centered)

        # Apply threshold
        return leverages <= self.threshold_

    def get_leverages(self, X: np.ndarray) -> np.ndarray:
        """
        Compute leverage values for samples.

        This method returns the raw leverage values for inspection or
        custom thresholding.

        Args:
            X: Feature array of shape (n_samples, n_features).

        Returns:
            np.ndarray: Array of leverage values for each sample.

        Raises:
            ValueError: If filter has not been fitted.
        """
        if self.precision_ is None:
            raise ValueError("Filter has not been fitted. Call fit() first.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.center:
            X_centered = X - self.mean_
        else:
            X_centered = X

        return self._compute_leverages(X_centered)

    def get_filter_stats(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Get statistics about filter application.

        Args:
            X: Feature array.
            y: Target array (unused).

        Returns:
            Dict containing:
                - Base stats (n_samples, n_excluded, n_kept, exclusion_rate)
                - method: Leverage computation method
                - threshold: Computed threshold
                - n_effective_features: Number of features/components used
                - leverage_stats: Statistics on leverage values
        """
        base_stats = super().get_filter_stats(X, y)

        base_stats.update({
            "method": self.method,
            "threshold": self.threshold_,
            "threshold_multiplier": self.threshold_multiplier,
            "absolute_threshold": self.absolute_threshold,
            "n_effective_features": self.n_effective_features_,
        })

        if self._leverages_ is not None:
            base_stats["leverage_stats"] = {
                "min": float(np.min(self._leverages_)),
                "max": float(np.max(self._leverages_)),
                "mean": float(np.mean(self._leverages_)),
                "median": float(np.median(self._leverages_)),
                "std": float(np.std(self._leverages_)),
            }

        return base_stats

    def __repr__(self) -> str:
        """Return string representation."""
        params = [f"method='{self.method}'"]

        if self.absolute_threshold is not None:
            params.append(f"absolute_threshold={self.absolute_threshold}")
        else:
            params.append(f"threshold_multiplier={self.threshold_multiplier}")

        if self.method == "pca" and self.n_components is not None:
            params.append(f"n_components={self.n_components}")

        return f"{self.__class__.__name__}({', '.join(params)})"
