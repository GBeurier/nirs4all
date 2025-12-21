"""
Aggregator for sample data aggregation.

This module provides the Aggregator class for aggregating sample data
during loading, with support for various aggregation methods and
custom aggregation functions.

Phase 8 Implementation - Dataset Configuration Roadmap
Section 8.1: Sample Aggregation Enhancements
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)


class AggregationMethod(str, Enum):
    """Aggregation method for combining samples."""

    MEAN = "mean"
    MEDIAN = "median"
    VOTE = "vote"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    STD = "std"
    FIRST = "first"
    LAST = "last"
    COUNT = "count"


@dataclass
class AggregationConfig:
    """Configuration for sample aggregation.

    Attributes:
        column: Column name to group by for aggregation.
            If True, aggregate by y values.
            If None, no aggregation.
        method: Aggregation method or custom function.
        exclude_outliers: Whether to exclude outliers before aggregation.
        outlier_threshold: Z-score threshold for outlier detection.
        min_samples: Minimum number of samples per group (groups with fewer are dropped).
        custom_function: Optional custom aggregation function.
        feature_method: Aggregation method for features (X), if different from targets.
        target_method: Aggregation method for targets (Y), if different from features.
    """

    column: Optional[Union[str, bool]] = None
    method: Union[AggregationMethod, str] = AggregationMethod.MEAN
    exclude_outliers: bool = False
    outlier_threshold: float = 3.0
    min_samples: int = 1
    custom_function: Optional[Callable] = None
    feature_method: Optional[Union[AggregationMethod, str]] = None
    target_method: Optional[Union[AggregationMethod, str]] = None

    def __post_init__(self):
        """Normalize method values to enum."""
        if isinstance(self.method, str):
            try:
                self.method = AggregationMethod(self.method.lower())
            except ValueError:
                pass  # Keep as string for custom methods

        if isinstance(self.feature_method, str):
            try:
                self.feature_method = AggregationMethod(self.feature_method.lower())
            except ValueError:
                pass

        if isinstance(self.target_method, str):
            try:
                self.target_method = AggregationMethod(self.target_method.lower())
            except ValueError:
                pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AggregationConfig":
        """Create from configuration dictionary.

        Args:
            config: Configuration dictionary with aggregation settings.

        Returns:
            AggregationConfig instance.
        """
        aggregate = config.get("aggregate")
        if aggregate is None:
            return cls(column=None)

        return cls(
            column=aggregate,
            method=config.get("aggregate_method", AggregationMethod.MEAN),
            exclude_outliers=config.get("aggregate_exclude_outliers", False),
            outlier_threshold=config.get("aggregate_outlier_threshold", 3.0),
            min_samples=config.get("aggregate_min_samples", 1),
            feature_method=config.get("aggregate_feature_method"),
            target_method=config.get("aggregate_target_method"),
        )

    def is_enabled(self) -> bool:
        """Check if aggregation is enabled."""
        return self.column is not None


class AggregationError(Exception):
    """Exception raised when aggregation fails."""
    pass


class Aggregator:
    """Aggregates sample data during loading.

    Supports grouping by metadata columns, target values, or sample IDs,
    with configurable aggregation methods for features and targets.

    Example:
        ```python
        # Aggregate by sample_id column using mean
        config = AggregationConfig(column="sample_id", method="mean")
        aggregator = Aggregator(config)
        X_agg, y_agg, meta_agg = aggregator.aggregate(X, y, metadata)

        # Aggregate with outlier exclusion
        config = AggregationConfig(
            column="sample_id",
            method="mean",
            exclude_outliers=True,
            outlier_threshold=2.5
        )
        aggregator = Aggregator(config)
        result = aggregator.aggregate(X, y, metadata)

        # Custom aggregation function
        config = AggregationConfig(
            column="sample_id",
            custom_function=lambda x: np.percentile(x, 75, axis=0)
        )
        aggregator = Aggregator(config)
        result = aggregator.aggregate(X, y, metadata)
        ```
    """

    def __init__(self, config: AggregationConfig):
        """Initialize aggregator.

        Args:
            config: Aggregation configuration.
        """
        self.config = config
        self._custom_functions: Dict[str, Callable] = {}

    def register_function(self, name: str, func: Callable) -> None:
        """Register a custom aggregation function.

        Args:
            name: Name to reference the function.
            func: Aggregation function that takes array and returns aggregated value.
        """
        self._custom_functions[name] = func

    def aggregate(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        metadata: Optional[pd.DataFrame] = None,
        group_column: Optional[str] = None,
    ) -> tuple:
        """Aggregate data by groups.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Optional target array of shape (n_samples,) or (n_samples, n_targets).
            metadata: Optional metadata DataFrame.
            group_column: Override column to group by.

        Returns:
            Tuple of (X_aggregated, y_aggregated, metadata_aggregated).
            Elements are None if not provided in input.

        Raises:
            AggregationError: If aggregation fails.
        """
        if not self.config.is_enabled() and group_column is None:
            return X, y, metadata

        # Determine grouping column
        column = group_column or self.config.column

        # Get group labels
        group_labels = self._get_group_labels(column, X, y, metadata)

        if group_labels is None:
            logger.warning("Could not determine group labels for aggregation")
            return X, y, metadata

        # Validate group labels
        if len(group_labels) != len(X):
            raise AggregationError(
                f"Group labels length ({len(group_labels)}) does not match "
                f"data length ({len(X)})"
            )

        # Get unique groups
        unique_groups = np.unique(group_labels)
        n_groups = len(unique_groups)

        logger.debug(f"Aggregating {len(X)} samples into {n_groups} groups")

        # Aggregate features
        X_agg = self._aggregate_array(
            X,
            group_labels,
            unique_groups,
            self.config.feature_method or self.config.method
        )

        # Aggregate targets
        y_agg = None
        if y is not None:
            target_method = self.config.target_method or self.config.method
            # For classification (vote), use mode; for regression, use specified method
            if target_method == AggregationMethod.VOTE:
                y_agg = self._aggregate_vote(y, group_labels, unique_groups)
            else:
                y_agg = self._aggregate_array(
                    y.reshape(-1, 1) if y.ndim == 1 else y,
                    group_labels,
                    unique_groups,
                    target_method
                )
                if y.ndim == 1:
                    y_agg = y_agg.ravel()

        # Aggregate metadata
        meta_agg = None
        if metadata is not None:
            meta_agg = self._aggregate_metadata(metadata, group_labels, unique_groups)

        return X_agg, y_agg, meta_agg

    def _get_group_labels(
        self,
        column: Union[str, bool],
        X: np.ndarray,
        y: Optional[np.ndarray],
        metadata: Optional[pd.DataFrame],
    ) -> Optional[np.ndarray]:
        """Get group labels for aggregation.

        Args:
            column: Column name, True for y-based grouping, or False/None for no grouping.
            X: Feature array.
            y: Target array.
            metadata: Metadata DataFrame.

        Returns:
            Array of group labels, or None if cannot determine.
        """
        if column is True:
            # Group by target values
            if y is None:
                logger.warning("Cannot aggregate by y values: y is None")
                return None
            return y.astype(str) if y.dtype == object else y

        if isinstance(column, str):
            # Group by metadata column
            if metadata is None:
                logger.warning(f"Cannot aggregate by '{column}': metadata is None")
                return None

            if column not in metadata.columns:
                # Try common column name variations
                possible_names = [
                    column,
                    column.lower(),
                    column.upper(),
                    column.replace("_", ""),
                    column.replace("-", "_"),
                ]
                found_column = None
                for name in possible_names:
                    if name in metadata.columns:
                        found_column = name
                        break
                    # Case-insensitive search
                    for col in metadata.columns:
                        if col.lower() == name.lower():
                            found_column = col
                            break
                    if found_column:
                        break

                if found_column is None:
                    logger.warning(
                        f"Aggregation column '{column}' not found in metadata. "
                        f"Available columns: {list(metadata.columns)}"
                    )
                    return None
                column = found_column

            return metadata[column].values

        return None

    def _aggregate_array(
        self,
        data: np.ndarray,
        group_labels: np.ndarray,
        unique_groups: np.ndarray,
        method: Union[AggregationMethod, str, Callable],
    ) -> np.ndarray:
        """Aggregate an array by groups.

        Args:
            data: Array to aggregate, shape (n_samples, n_features).
            group_labels: Array of group labels.
            unique_groups: Unique group values.
            method: Aggregation method.

        Returns:
            Aggregated array of shape (n_groups, n_features).
        """
        n_groups = len(unique_groups)
        n_features = data.shape[1] if data.ndim > 1 else 1

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        result = np.zeros((n_groups, n_features), dtype=data.dtype)

        # Get the aggregation function
        agg_func = self._get_aggregation_function(method)

        for i, group in enumerate(unique_groups):
            mask = group_labels == group
            group_data = data[mask]

            # Exclude outliers if configured
            if self.config.exclude_outliers and len(group_data) > 2:
                group_data = self._remove_outliers(group_data)

            # Check minimum samples
            if len(group_data) < self.config.min_samples:
                logger.warning(
                    f"Group '{group}' has {len(group_data)} samples, "
                    f"less than minimum {self.config.min_samples}. Using available data."
                )

            if len(group_data) == 0:
                result[i] = np.nan
            else:
                result[i] = agg_func(group_data, axis=0)

        return result

    def _aggregate_vote(
        self,
        y: np.ndarray,
        group_labels: np.ndarray,
        unique_groups: np.ndarray,
    ) -> np.ndarray:
        """Aggregate targets using majority voting.

        Args:
            y: Target array.
            group_labels: Array of group labels.
            unique_groups: Unique group values.

        Returns:
            Aggregated target array.
        """
        result = np.zeros(len(unique_groups), dtype=y.dtype)

        for i, group in enumerate(unique_groups):
            mask = group_labels == group
            group_y = y[mask]

            if len(group_y) == 0:
                result[i] = np.nan if np.issubdtype(y.dtype, np.floating) else 0
            else:
                # Get mode (most common value)
                values, counts = np.unique(group_y, return_counts=True)
                result[i] = values[np.argmax(counts)]

        return result

    def _aggregate_metadata(
        self,
        metadata: pd.DataFrame,
        group_labels: np.ndarray,
        unique_groups: np.ndarray,
    ) -> pd.DataFrame:
        """Aggregate metadata by groups.

        For each group, takes the first row's metadata values.

        Args:
            metadata: Metadata DataFrame.
            group_labels: Array of group labels.
            unique_groups: Unique group values.

        Returns:
            Aggregated metadata DataFrame.
        """
        rows = []
        for group in unique_groups:
            mask = group_labels == group
            indices = np.where(mask)[0]
            if len(indices) > 0:
                # Take first row of each group
                rows.append(metadata.iloc[indices[0]])

        return pd.DataFrame(rows).reset_index(drop=True)

    def _get_aggregation_function(
        self,
        method: Union[AggregationMethod, str, Callable]
    ) -> Callable:
        """Get the aggregation function for a method.

        Args:
            method: Aggregation method name or callable.

        Returns:
            Callable that performs aggregation.
        """
        if callable(method):
            return method

        if self.config.custom_function is not None:
            return self.config.custom_function

        # Check custom registered functions
        method_str = method.value if isinstance(method, AggregationMethod) else str(method)
        if method_str in self._custom_functions:
            return self._custom_functions[method_str]

        # Built-in methods
        method_map = {
            AggregationMethod.MEAN: lambda x, axis=0: np.nanmean(x, axis=axis),
            AggregationMethod.MEDIAN: lambda x, axis=0: np.nanmedian(x, axis=axis),
            AggregationMethod.MIN: lambda x, axis=0: np.nanmin(x, axis=axis),
            AggregationMethod.MAX: lambda x, axis=0: np.nanmax(x, axis=axis),
            AggregationMethod.SUM: lambda x, axis=0: np.nansum(x, axis=axis),
            AggregationMethod.STD: lambda x, axis=0: np.nanstd(x, axis=axis),
            AggregationMethod.FIRST: lambda x, axis=0: x[0] if len(x) > 0 else np.nan,
            AggregationMethod.LAST: lambda x, axis=0: x[-1] if len(x) > 0 else np.nan,
            AggregationMethod.COUNT: lambda x, axis=0: np.sum(~np.isnan(x), axis=axis),
        }

        if isinstance(method, AggregationMethod):
            return method_map.get(method, method_map[AggregationMethod.MEAN])

        # Try to match string method
        try:
            method_enum = AggregationMethod(method_str.lower())
            return method_map.get(method_enum, method_map[AggregationMethod.MEAN])
        except ValueError:
            logger.warning(f"Unknown aggregation method '{method}', using mean")
            return method_map[AggregationMethod.MEAN]

    def _remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """Remove outlier rows from data using z-score.

        Args:
            data: Array of shape (n_samples, n_features).

        Returns:
            Array with outlier rows removed.
        """
        if len(data) <= 2:
            return data

        # Calculate z-scores across features
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)

        # Avoid division by zero
        std = np.where(std == 0, 1, std)

        z_scores = np.abs((data - mean) / std)

        # A row is an outlier if any feature exceeds threshold
        max_z = np.nanmax(z_scores, axis=1)
        mask = max_z <= self.config.outlier_threshold

        return data[mask]


def aggregate_data(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    metadata: Optional[pd.DataFrame] = None,
    column: Optional[Union[str, bool]] = None,
    method: Union[str, AggregationMethod] = "mean",
    exclude_outliers: bool = False,
    **kwargs,
) -> tuple:
    """Convenience function to aggregate data.

    Args:
        X: Feature array.
        y: Optional target array.
        metadata: Optional metadata DataFrame.
        column: Column to group by (str), or True for y-based grouping.
        method: Aggregation method.
        exclude_outliers: Whether to exclude outliers.
        **kwargs: Additional aggregation config options.

    Returns:
        Tuple of (X_aggregated, y_aggregated, metadata_aggregated).
    """
    if column is None:
        return X, y, metadata

    config = AggregationConfig(
        column=column,
        method=method,
        exclude_outliers=exclude_outliers,
        **kwargs,
    )
    aggregator = Aggregator(config)
    return aggregator.aggregate(X, y, metadata)
