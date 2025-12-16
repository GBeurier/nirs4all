"""
Base class for sample filtering operators.

Sample filters identify samples to be excluded from training based on various criteria.
They follow the sklearn TransformerMixin pattern for consistency with the existing
augmentation and transformation operators in nirs4all.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SampleFilter(TransformerMixin, BaseEstimator, ABC):
    """
    Base class for sample filtering operators.

    Sample filters identify samples that should be excluded from training datasets.
    Unlike transformers that modify data, filters mark samples for exclusion without
    altering the underlying data.

    The filtering pattern works as follows:
    1. `fit()`: Learn filter criteria from training data (e.g., compute thresholds)
    2. `get_mask()`: Return boolean mask indicating which samples to KEEP
    3. `transform()`: No-op (filtering happens at indexer level, not data level)

    All concrete filter implementations must override the `get_mask()` method.

    Attributes:
        reason (str): Identifier for this filter type, used to track exclusion reasons
                     in the indexer. Default is the class name.

    Example:
        >>> class MyFilter(SampleFilter):
        ...     def __init__(self, threshold: float = 1.0):
        ...         super().__init__()
        ...         self.threshold = threshold
        ...
        ...     def fit(self, X, y=None):
        ...         self.mean_ = np.mean(y)
        ...         self.std_ = np.std(y)
        ...         return self
        ...
        ...     def get_mask(self, X, y=None) -> np.ndarray:
        ...         z_scores = np.abs((y - self.mean_) / self.std_)
        ...         return z_scores <= self.threshold  # True = keep
    """

    def __init__(self, reason: Optional[str] = None):
        """
        Initialize the sample filter.

        Args:
            reason: String identifier for tracking exclusion reasons in the indexer.
                   If None, defaults to the class name (e.g., "YOutlierFilter").
        """
        self.reason = reason

    @property
    def exclusion_reason(self) -> str:
        """
        Get the exclusion reason identifier for this filter.

        Returns:
            str: Reason string to be stored in indexer's exclusion_reason column.
        """
        return self.reason if self.reason is not None else self.__class__.__name__

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SampleFilter":
        """
        Compute filter criteria from training data.

        This method should learn any thresholds, statistics, or models needed
        to identify outliers/bad samples. Override in subclasses for filters
        that need to learn from data.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Target array of shape (n_samples,) or (n_samples, n_targets).
               May be None for X-only filters.

        Returns:
            self: The fitted filter instance.
        """
        return self

    @abstractmethod
    def get_mask(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute boolean mask indicating which samples to KEEP.

        This is the core method that must be implemented by all concrete filters.
        Returns True for samples that should be kept, False for samples to exclude.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Target array of shape (n_samples,) or (n_samples, n_targets).
               May be None for X-only filters.

        Returns:
            np.ndarray: Boolean array of shape (n_samples,) where:
                       - True means KEEP the sample
                       - False means EXCLUDE the sample

        Raises:
            NotImplementedError: If the subclass doesn't implement this method.
        """
        raise NotImplementedError("Subclasses must implement get_mask()")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform is a no-op for filters.

        Filtering happens at the indexer level, not by modifying the data array.
        This method returns the input unchanged to maintain sklearn compatibility.

        Args:
            X: Feature array of shape (n_samples, n_features).

        Returns:
            np.ndarray: The unchanged input array.
        """
        return X

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params) -> np.ndarray:
        """
        Fit to data and return unchanged (transform is no-op).

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Target array of shape (n_samples,) or (n_samples, n_targets).
            **fit_params: Additional fitting parameters (unused).

        Returns:
            np.ndarray: The unchanged input array.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_excluded_indices(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get indices of samples to be excluded.

        Convenience method that inverts get_mask() to return indices of
        samples marked for exclusion.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Target array of shape (n_samples,) or (n_samples, n_targets).

        Returns:
            np.ndarray: Integer array of indices for samples to exclude.

        Example:
            >>> filter = YOutlierFilter(method="iqr")
            >>> filter.fit(X_train, y_train)
            >>> excluded_idx = filter.get_excluded_indices(X_train, y_train)
            >>> print(f"Excluding {len(excluded_idx)} samples")
        """
        X = np.asarray(X)
        if X.size == 0:
            return np.array([], dtype=np.intp)
        mask = self.get_mask(X, y)
        return np.where(~mask)[0]

    def get_kept_indices(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get indices of samples to be kept.

        Convenience method that returns indices of samples NOT marked for exclusion.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Target array of shape (n_samples,) or (n_samples, n_targets).

        Returns:
            np.ndarray: Integer array of indices for samples to keep.
        """
        X = np.asarray(X)
        if X.size == 0:
            return np.array([], dtype=np.intp)
        mask = self.get_mask(X, y)
        return np.where(mask)[0]

    def get_filter_stats(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Get statistics about filter application.

        Override in subclasses to provide filter-specific statistics
        (e.g., thresholds used, distribution of values, etc.).

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Target array of shape (n_samples,) or (n_samples, n_targets).

        Returns:
            Dict[str, Any]: Dictionary containing filter statistics:
                - n_samples: Total number of samples
                - n_excluded: Number of samples to exclude
                - n_kept: Number of samples to keep
                - exclusion_rate: Ratio of excluded to total
                - reason: Exclusion reason string
        """
        X = np.asarray(X)
        n_samples = len(X)

        # Handle empty dataset
        if n_samples == 0:
            return {
                "n_samples": 0,
                "n_excluded": 0,
                "n_kept": 0,
                "exclusion_rate": 0.0,
                "reason": self.exclusion_reason,
            }

        mask = self.get_mask(X, y)
        n_kept = int(np.sum(mask))
        n_excluded = n_samples - n_kept

        return {
            "n_samples": n_samples,
            "n_excluded": n_excluded,
            "n_kept": n_kept,
            "exclusion_rate": n_excluded / n_samples if n_samples > 0 else 0.0,
            "reason": self.exclusion_reason,
        }

    def _more_tags(self):
        """
        Provide additional tags for sklearn compatibility.

        Returns:
            dict: Additional estimator tags.
        """
        return {"allow_nan": False, "stateless": False}


class CompositeFilter(SampleFilter):
    """
    Combine multiple filters with AND/OR logic.

    This filter aggregates the results of multiple sub-filters using
    either "any" or "all" mode:
    - "any" (default): Exclude if ANY filter flags the sample
    - "all": Exclude only if ALL filters flag the sample

    Attributes:
        filters (List[SampleFilter]): List of filter instances to combine
        mode (str): Combination mode - "any" or "all"

    Example:
        >>> from nirs4all.operators.filters import YOutlierFilter, CompositeFilter
        >>>
        >>> # Exclude if either filter flags
        >>> combined = CompositeFilter(
        ...     filters=[
        ...         YOutlierFilter(method="iqr", threshold=1.5),
        ...         YOutlierFilter(method="zscore", threshold=3.0),
        ...     ],
        ...     mode="any"
        ... )
    """

    def __init__(
        self,
        filters: Optional[List[SampleFilter]] = None,
        mode: str = "any",
        reason: Optional[str] = None
    ):
        """
        Initialize the composite filter.

        Args:
            filters: List of SampleFilter instances to combine.
                    If None, defaults to empty list.
            mode: Combination mode for filter results:
                 - "any": Exclude if ANY filter flags (logical OR of exclusions)
                 - "all": Exclude only if ALL filters flag (logical AND of exclusions)
            reason: Custom reason string. If None, auto-generates from mode and filters.

        Raises:
            ValueError: If mode is not "any" or "all".
        """
        super().__init__(reason=reason)
        self.filters = filters if filters is not None else []
        self.mode = mode

        if mode not in ("any", "all"):
            raise ValueError(f"mode must be 'any' or 'all', got '{mode}'")

    @property
    def exclusion_reason(self) -> str:
        """Get combined exclusion reason from all filters."""
        if self.reason is not None:
            return self.reason

        if not self.filters:
            return "CompositeFilter"

        filter_names = [f.exclusion_reason for f in self.filters]
        return f"composite({self.mode}:{','.join(filter_names)})"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "CompositeFilter":
        """
        Fit all sub-filters to the training data.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Target array of shape (n_samples,) or (n_samples, n_targets).

        Returns:
            self: The fitted composite filter.
        """
        for f in self.filters:
            f.fit(X, y)
        return self

    def get_mask(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute combined mask from all sub-filters.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Target array of shape (n_samples,) or (n_samples, n_targets).

        Returns:
            np.ndarray: Boolean array where True = keep, False = exclude.
                       For "any" mode: keep if ALL filters say keep
                       For "all" mode: keep if ANY filter says keep
        """
        if not self.filters:
            # No filters = keep all samples
            return np.ones(len(X), dtype=bool)

        # Collect all masks (True = keep)
        masks = [f.get_mask(X, y) for f in self.filters]
        stacked = np.stack(masks, axis=0)  # Shape: (n_filters, n_samples)

        if self.mode == "any":
            # Exclude if ANY filter flags (keep only if ALL keep)
            # keep_mask = all filters say keep
            return np.all(stacked, axis=0)
        else:  # mode == "all"
            # Exclude only if ALL filters flag (keep if ANY keeps)
            # keep_mask = any filter says keep
            return np.any(stacked, axis=0)

    def get_filter_stats(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Get statistics including per-filter breakdown.

        Args:
            X: Feature array.
            y: Target array.

        Returns:
            Dict with overall stats and per-filter breakdown.
        """
        base_stats = super().get_filter_stats(X, y)

        # Add per-filter breakdown
        filter_stats = []
        for f in self.filters:
            filter_stats.append(f.get_filter_stats(X, y))

        base_stats["mode"] = self.mode
        base_stats["filter_breakdown"] = filter_stats
        return base_stats
