"""
Spectral quality filter for sample filtering.

This module provides the SpectralQualityFilter class for detecting and excluding
samples with poor spectral quality based on various quality metrics.
"""

from typing import Optional, Dict, Any
import numpy as np

from .base import SampleFilter


class SpectralQualityFilter(SampleFilter):
    """
    Filter samples with poor spectral quality.

    This filter identifies samples whose spectra exhibit quality issues such as:
    - High proportion of NaN or missing values
    - High proportion of zero values (potentially corrupted)
    - Very low variance (flat or constant spectra)
    - Values outside expected range (saturation)

    Attributes:
        max_nan_ratio (float): Maximum allowed NaN ratio per spectrum
        max_zero_ratio (float): Maximum allowed zero ratio
        min_variance (float): Minimum variance threshold
        max_value (float): Maximum allowed value (saturation detection)
        min_value (float): Minimum allowed value

    Example:
        >>> from nirs4all.operators.filters import SpectralQualityFilter
        >>>
        >>> # Default quality checks
        >>> filter_obj = SpectralQualityFilter()
        >>>
        >>> # Strict quality requirements
        >>> filter_strict = SpectralQualityFilter(
        ...     max_nan_ratio=0.01,
        ...     max_zero_ratio=0.1,
        ...     min_variance=1e-4
        ... )
        >>>
        >>> # Check for saturated spectra
        >>> filter_sat = SpectralQualityFilter(max_value=4.0, min_value=-0.5)
        >>>
        >>> # Get mask
        >>> mask = filter_obj.get_mask(X_train)  # True = keep

    In Pipeline:
        >>> pipeline = [
        ...     {
        ...         "sample_filter": {
        ...             "filters": [SpectralQualityFilter(max_nan_ratio=0.05)],
        ...         }
        ...     },
        ...     "snv",
        ...     "model:PLSRegression",
        ... ]
    """

    def __init__(
        self,
        max_nan_ratio: float = 0.1,
        max_zero_ratio: float = 0.5,
        min_variance: float = 1e-8,
        max_value: Optional[float] = None,
        min_value: Optional[float] = None,
        check_inf: bool = True,
        reason: Optional[str] = None
    ):
        """
        Initialize the spectral quality filter.

        Args:
            max_nan_ratio: Maximum allowed ratio of NaN values per spectrum.
                          Samples with higher NaN ratio are excluded.
                          Range [0, 1]. Default 0.1 (10%).
            max_zero_ratio: Maximum allowed ratio of zero values per spectrum.
                           High zero ratio may indicate corrupted or truncated data.
                           Range [0, 1]. Default 0.5 (50%).
            min_variance: Minimum variance threshold for spectra.
                         Samples with variance below this are excluded (flat spectra).
                         Default 1e-8.
            max_value: Maximum allowed value in spectrum. Samples with any value
                      above this threshold are excluded (saturation detection).
                      If None, no upper limit check. Default None.
            min_value: Minimum allowed value in spectrum. Samples with any value
                      below this threshold are excluded.
                      If None, no lower limit check. Default None.
            check_inf: Whether to check for infinite values. If True, samples
                      containing Inf are excluded. Default True.
            reason: Custom exclusion reason. Defaults to filter description.

        Raises:
            ValueError: If ratio parameters are not in valid range.
            ValueError: If min_variance is negative.
        """
        super().__init__(reason=reason)
        self.max_nan_ratio = max_nan_ratio
        self.max_zero_ratio = max_zero_ratio
        self.min_variance = min_variance
        self.max_value = max_value
        self.min_value = min_value
        self.check_inf = check_inf

        # Validate parameters
        if not (0 <= max_nan_ratio <= 1):
            raise ValueError(
                f"max_nan_ratio must be in [0, 1], got {max_nan_ratio}"
            )
        if not (0 <= max_zero_ratio <= 1):
            raise ValueError(
                f"max_zero_ratio must be in [0, 1], got {max_zero_ratio}"
            )
        if min_variance < 0:
            raise ValueError(
                f"min_variance must be non-negative, got {min_variance}"
            )

        # Quality statistics (set during get_mask)
        self._quality_stats_: Optional[Dict[str, np.ndarray]] = None

    @property
    def exclusion_reason(self) -> str:
        """Get descriptive exclusion reason."""
        if self.reason is not None:
            return self.reason
        return "spectral_quality"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SpectralQualityFilter":
        """
        Fit the filter (no-op for quality filter as thresholds are fixed).

        The SpectralQualityFilter uses fixed thresholds set at initialization,
        so no fitting is required. This method is provided for API consistency.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Target array (not used).

        Returns:
            self: The filter instance (unchanged).
        """
        # No fitting needed - thresholds are fixed
        return self

    def get_mask(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute boolean mask indicating which samples to KEEP based on quality.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Target array (not used for X-based quality checks).

        Returns:
            np.ndarray: Boolean array of shape (n_samples,) where:
                       - True means KEEP the sample (passes quality checks)
                       - False means EXCLUDE the sample (fails quality checks)
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples, n_features = X.shape

        # Initialize mask (all True = keep all)
        mask = np.ones(n_samples, dtype=bool)

        # Store quality statistics
        self._quality_stats_ = {}

        # Check NaN ratio
        nan_counts = np.sum(np.isnan(X), axis=1)
        nan_ratios = nan_counts / n_features
        self._quality_stats_["nan_ratio"] = nan_ratios
        mask &= (nan_ratios <= self.max_nan_ratio)

        # Check Inf values
        if self.check_inf:
            has_inf = np.any(np.isinf(X), axis=1)
            self._quality_stats_["has_inf"] = has_inf
            mask &= ~has_inf

        # Check zero ratio (treating NaN as non-zero for this check)
        X_no_nan = np.nan_to_num(X, nan=1.0)  # Replace NaN with non-zero
        zero_counts = np.sum(X_no_nan == 0, axis=1)
        zero_ratios = zero_counts / n_features
        self._quality_stats_["zero_ratio"] = zero_ratios
        mask &= (zero_ratios <= self.max_zero_ratio)

        # Check variance (ignoring NaN values)
        variances = np.nanvar(X, axis=1)
        self._quality_stats_["variance"] = variances
        mask &= (variances >= self.min_variance)

        # Check value range (if specified)
        if self.max_value is not None:
            max_values = np.nanmax(X, axis=1)
            self._quality_stats_["max_value"] = max_values
            mask &= (max_values <= self.max_value)

        if self.min_value is not None:
            min_values = np.nanmin(X, axis=1)
            self._quality_stats_["min_value"] = min_values
            mask &= (min_values >= self.min_value)

        return mask

    def get_quality_breakdown(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get detailed breakdown of which quality checks each sample fails.

        This method provides per-check masks to understand why specific
        samples were excluded.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Target array (not used).

        Returns:
            Dict with boolean arrays for each quality check:
                - "passes_nan": True if NaN ratio is acceptable
                - "passes_inf": True if no Inf values
                - "passes_zero": True if zero ratio is acceptable
                - "passes_variance": True if variance is sufficient
                - "passes_max_value": True if max value is within limit
                - "passes_min_value": True if min value is within limit
                - "passes_all": True if passes all checks
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples, n_features = X.shape

        breakdown = {}

        # NaN check
        nan_counts = np.sum(np.isnan(X), axis=1)
        nan_ratios = nan_counts / n_features
        breakdown["passes_nan"] = nan_ratios <= self.max_nan_ratio

        # Inf check
        if self.check_inf:
            has_inf = np.any(np.isinf(X), axis=1)
            breakdown["passes_inf"] = ~has_inf
        else:
            breakdown["passes_inf"] = np.ones(n_samples, dtype=bool)

        # Zero check
        X_no_nan = np.nan_to_num(X, nan=1.0)
        zero_counts = np.sum(X_no_nan == 0, axis=1)
        zero_ratios = zero_counts / n_features
        breakdown["passes_zero"] = zero_ratios <= self.max_zero_ratio

        # Variance check
        variances = np.nanvar(X, axis=1)
        breakdown["passes_variance"] = variances >= self.min_variance

        # Value range checks
        if self.max_value is not None:
            max_values = np.nanmax(X, axis=1)
            breakdown["passes_max_value"] = max_values <= self.max_value
        else:
            breakdown["passes_max_value"] = np.ones(n_samples, dtype=bool)

        if self.min_value is not None:
            min_values = np.nanmin(X, axis=1)
            breakdown["passes_min_value"] = min_values >= self.min_value
        else:
            breakdown["passes_min_value"] = np.ones(n_samples, dtype=bool)

        # Combined
        breakdown["passes_all"] = (
            breakdown["passes_nan"]
            & breakdown["passes_inf"]
            & breakdown["passes_zero"]
            & breakdown["passes_variance"]
            & breakdown["passes_max_value"]
            & breakdown["passes_min_value"]
        )

        return breakdown

    def get_filter_stats(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Get statistics about filter application including quality breakdown.

        Args:
            X: Feature array.
            y: Target array (unused).

        Returns:
            Dict containing:
                - Base stats (n_samples, n_excluded, n_kept, exclusion_rate)
                - Quality thresholds
                - Per-check failure counts
                - Quality metric distributions
        """
        # Ensure mask is computed for stats
        self.get_mask(X, y)

        base_stats = super().get_filter_stats(X, y)

        # Add quality thresholds
        base_stats.update({
            "max_nan_ratio": self.max_nan_ratio,
            "max_zero_ratio": self.max_zero_ratio,
            "min_variance": self.min_variance,
            "max_value": self.max_value,
            "min_value": self.min_value,
            "check_inf": self.check_inf,
        })

        # Add per-check breakdown
        breakdown = self.get_quality_breakdown(X, y)
        base_stats["failure_counts"] = {
            "nan_ratio": int(np.sum(~breakdown["passes_nan"])),
            "inf_values": int(np.sum(~breakdown["passes_inf"])),
            "zero_ratio": int(np.sum(~breakdown["passes_zero"])),
            "low_variance": int(np.sum(~breakdown["passes_variance"])),
            "max_value_exceeded": int(np.sum(~breakdown["passes_max_value"])),
            "min_value_exceeded": int(np.sum(~breakdown["passes_min_value"])),
        }

        # Add quality metric summaries
        if self._quality_stats_ is not None:
            base_stats["quality_metrics"] = {}
            for metric, values in self._quality_stats_.items():
                if values.dtype == bool:
                    base_stats["quality_metrics"][metric] = {
                        "count_true": int(np.sum(values))
                    }
                else:
                    base_stats["quality_metrics"][metric] = {
                        "min": float(np.nanmin(values)),
                        "max": float(np.nanmax(values)),
                        "mean": float(np.nanmean(values)),
                        "median": float(np.nanmedian(values)),
                    }

        return base_stats

    def __repr__(self) -> str:
        """Return string representation."""
        params = []

        if self.max_nan_ratio != 0.1:
            params.append(f"max_nan_ratio={self.max_nan_ratio}")
        if self.max_zero_ratio != 0.5:
            params.append(f"max_zero_ratio={self.max_zero_ratio}")
        if self.min_variance != 1e-8:
            params.append(f"min_variance={self.min_variance}")
        if self.max_value is not None:
            params.append(f"max_value={self.max_value}")
        if self.min_value is not None:
            params.append(f"min_value={self.min_value}")

        param_str = ", ".join(params) if params else ""
        return f"{self.__class__.__name__}({param_str})"
