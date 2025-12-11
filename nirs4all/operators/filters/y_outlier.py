"""
Y-based outlier filter for sample filtering.

This module provides the YOutlierFilter class for detecting and excluding
samples with outlier target (y) values using various statistical methods.
"""

from typing import Optional, Dict, Any, Literal
import numpy as np

from .base import SampleFilter


class YOutlierFilter(SampleFilter):
    """
    Filter samples with outlier target values.

    This filter identifies samples whose y-values are statistical outliers
    using one of several detection methods. It's commonly used to remove
    samples with extreme or erroneous target values before training.

    Supported methods:
    - "iqr": Interquartile Range method (default)
    - "zscore": Z-score (standard deviations from mean)
    - "percentile": Direct percentile cutoffs
    - "mad": Median Absolute Deviation (robust to outliers)

    Attributes:
        method (str): Outlier detection method
        threshold (float): Method-specific threshold
        lower_percentile (float): Lower cutoff for percentile method
        upper_percentile (float): Upper cutoff for percentile method

    Example:
        >>> from nirs4all.operators.filters import YOutlierFilter
        >>>
        >>> # IQR method (default, threshold=1.5 is standard)
        >>> filter_iqr = YOutlierFilter(method="iqr", threshold=1.5)
        >>>
        >>> # Z-score method (threshold=3.0 is common)
        >>> filter_zscore = YOutlierFilter(method="zscore", threshold=3.0)
        >>>
        >>> # Percentile method
        >>> filter_pct = YOutlierFilter(
        ...     method="percentile",
        ...     lower_percentile=1.0,
        ...     upper_percentile=99.0
        ... )
        >>>
        >>> # Fit and get mask
        >>> filter_iqr.fit(X_train, y_train)
        >>> mask = filter_iqr.get_mask(X_train, y_train)  # True = keep

    In Pipeline:
        >>> pipeline = [
        ...     {
        ...         "sample_filter": {
        ...             "filters": [YOutlierFilter(method="iqr", threshold=1.5)],
        ...         }
        ...     },
        ...     "snv",
        ...     "model:PLSRegression",
        ... ]
    """

    def __init__(
        self,
        method: Literal["iqr", "zscore", "percentile", "mad"] = "iqr",
        threshold: float = 1.5,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
        reason: Optional[str] = None
    ):
        """
        Initialize the Y outlier filter.

        Args:
            method: Outlier detection method:
                   - "iqr": Interquartile Range. Threshold multiplies the IQR.
                           threshold=1.5 is standard (mild outliers),
                           threshold=3.0 for extreme outliers.
                   - "zscore": Z-score. Threshold is number of standard deviations.
                              threshold=3.0 is common (3-sigma rule).
                   - "percentile": Direct percentile cutoffs.
                                  Uses lower_percentile and upper_percentile.
                   - "mad": Median Absolute Deviation. More robust to outliers.
                           threshold=3.5 is commonly used.
            threshold: Threshold value for iqr/zscore/mad methods. Default 1.5.
            lower_percentile: Lower percentile cutoff for percentile method.
                            Samples below this percentile are excluded. Default 1.0.
            upper_percentile: Upper percentile cutoff for percentile method.
                            Samples above this percentile are excluded. Default 99.0.
            reason: Custom exclusion reason. Defaults to filter description.

        Raises:
            ValueError: If method is not one of the supported methods.
            ValueError: If threshold is not positive.
            ValueError: If percentiles are not in valid range.
        """
        super().__init__(reason=reason)
        self.method = method
        self.threshold = threshold
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

        # Validate parameters
        valid_methods = ("iqr", "zscore", "percentile", "mad")
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")

        if not (0 <= lower_percentile < upper_percentile <= 100):
            raise ValueError(
                f"Percentiles must satisfy 0 <= lower < upper <= 100, "
                f"got lower={lower_percentile}, upper={upper_percentile}"
            )

        # Fitted attributes (set during fit)
        self.lower_bound_: Optional[float] = None
        self.upper_bound_: Optional[float] = None
        self.center_: Optional[float] = None
        self.scale_: Optional[float] = None

    @property
    def exclusion_reason(self) -> str:
        """Get descriptive exclusion reason."""
        if self.reason is not None:
            return self.reason
        return f"y_outlier_{self.method}"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "YOutlierFilter":
        """
        Compute outlier detection bounds from training data.

        Args:
            X: Feature array of shape (n_samples, n_features).
               Not used but required for sklearn compatibility.
            y: Target array of shape (n_samples,) or (n_samples, n_targets).
               Required for Y-based filtering.

        Returns:
            self: The fitted filter instance.

        Raises:
            ValueError: If y is None (required for Y-based filtering).
            ValueError: If y has no valid (non-NaN) values.
        """
        if y is None:
            raise ValueError("YOutlierFilter requires y values for fitting")

        # Flatten y if multi-dimensional
        y_flat = np.asarray(y).flatten()

        # Handle empty input
        if len(y_flat) == 0:
            # Set neutral bounds that keep everything
            self.lower_bound_ = float('-inf')
            self.upper_bound_ = float('inf')
            self.center_ = 0.0
            self.scale_ = 1.0
            return self

        # Remove NaN values for fitting
        y_valid = y_flat[~np.isnan(y_flat)]
        if len(y_valid) == 0:
            raise ValueError("y contains no valid (non-NaN) values")

        # Handle single sample case
        if len(y_valid) == 1:
            self.center_ = float(y_valid[0])
            self.scale_ = 0.0
            self.lower_bound_ = self.center_ - 1e-10
            self.upper_bound_ = self.center_ + 1e-10
            return self

        if self.method == "iqr":
            self._fit_iqr(y_valid)
        elif self.method == "zscore":
            self._fit_zscore(y_valid)
        elif self.method == "percentile":
            self._fit_percentile(y_valid)
        elif self.method == "mad":
            self._fit_mad(y_valid)

        return self

    def _fit_iqr(self, y: np.ndarray) -> None:
        """Fit using Interquartile Range method."""
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        iqr = q3 - q1

        self.lower_bound_ = q1 - self.threshold * iqr
        self.upper_bound_ = q3 + self.threshold * iqr
        self.center_ = np.median(y)
        self.scale_ = iqr

    def _fit_zscore(self, y: np.ndarray) -> None:
        """Fit using Z-score method."""
        self.center_ = np.mean(y)
        self.scale_ = np.std(y)

        if self.scale_ == 0:
            # All values are the same, no outliers possible
            self.lower_bound_ = self.center_ - 1e-10
            self.upper_bound_ = self.center_ + 1e-10
        else:
            self.lower_bound_ = self.center_ - self.threshold * self.scale_
            self.upper_bound_ = self.center_ + self.threshold * self.scale_

    def _fit_percentile(self, y: np.ndarray) -> None:
        """Fit using direct percentile cutoffs."""
        self.lower_bound_ = np.percentile(y, self.lower_percentile)
        self.upper_bound_ = np.percentile(y, self.upper_percentile)
        self.center_ = np.median(y)
        self.scale_ = self.upper_bound_ - self.lower_bound_

    def _fit_mad(self, y: np.ndarray) -> None:
        """Fit using Median Absolute Deviation method."""
        self.center_ = np.median(y)
        # MAD = median(|y - median(y)|)
        mad = np.median(np.abs(y - self.center_))

        # Scale factor for normal distribution consistency
        # MAD * 1.4826 ≈ standard deviation for normal data
        mad_scaled = mad * 1.4826

        if mad_scaled == 0:
            # All values are the same or very close
            self.lower_bound_ = self.center_ - 1e-10
            self.upper_bound_ = self.center_ + 1e-10
            self.scale_ = 1e-10
        else:
            self.lower_bound_ = self.center_ - self.threshold * mad_scaled
            self.upper_bound_ = self.center_ + self.threshold * mad_scaled
            self.scale_ = mad_scaled

    def get_mask(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute boolean mask indicating which samples to KEEP.

        Args:
            X: Feature array of shape (n_samples, n_features).
               Not used but required for API consistency.
            y: Target array of shape (n_samples,) or (n_samples, n_targets).
               Required for Y-based filtering.

        Returns:
            np.ndarray: Boolean array of shape (n_samples,) where:
                       - True means KEEP the sample (within bounds)
                       - False means EXCLUDE the sample (outside bounds)

        Raises:
            ValueError: If y is None.
            ValueError: If filter has not been fitted (bounds not set).
        """
        if y is None:
            raise ValueError("YOutlierFilter requires y values for filtering")

        if self.lower_bound_ is None or self.upper_bound_ is None:
            raise ValueError("Filter has not been fitted. Call fit() first.")

        # Flatten y if multi-dimensional
        y_flat = np.asarray(y).flatten()

        # Handle NaN values: mark as outliers (exclude them)
        is_nan = np.isnan(y_flat)

        # Check bounds
        within_bounds = (y_flat >= self.lower_bound_) & (y_flat <= self.upper_bound_)

        # Keep if within bounds AND not NaN
        return within_bounds & ~is_nan

    def get_filter_stats(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Get statistics about filter application including method-specific details.

        Args:
            X: Feature array.
            y: Target array.

        Returns:
            Dict containing:
                - Base stats (n_samples, n_excluded, n_kept, exclusion_rate)
                - method: Detection method used
                - threshold: Threshold value
                - lower_bound: Computed lower bound
                - upper_bound: Computed upper bound
                - center: Central value (mean/median)
                - scale: Scale measure (std/IQR/MAD)
                - y_range: (min, max) of input y values
        """
        base_stats = super().get_filter_stats(X, y)

        # Add method-specific stats
        y_flat = np.asarray(y).flatten() if y is not None else np.array([])
        y_valid = y_flat[~np.isnan(y_flat)] if len(y_flat) > 0 else np.array([])

        base_stats.update({
            "method": self.method,
            "threshold": self.threshold,
            "lower_bound": self.lower_bound_,
            "upper_bound": self.upper_bound_,
            "center": self.center_,
            "scale": self.scale_,
            "y_range": (float(np.min(y_valid)), float(np.max(y_valid))) if len(y_valid) > 0 else (None, None),
        })

        if self.method == "percentile":
            base_stats["lower_percentile"] = self.lower_percentile
            base_stats["upper_percentile"] = self.upper_percentile

        return base_stats

    def __repr__(self) -> str:
        """Return string representation."""
        if self.method == "percentile":
            return (
                f"{self.__class__.__name__}("
                f"method='{self.method}', "
                f"lower_percentile={self.lower_percentile}, "
                f"upper_percentile={self.upper_percentile})"
            )
        return f"{self.__class__.__name__}(method='{self.method}', threshold={self.threshold})"
