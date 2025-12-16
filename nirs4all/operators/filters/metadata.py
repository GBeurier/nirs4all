"""
Metadata-based filter for sample filtering.

This module provides the MetadataFilter class for filtering samples based on
metadata column values using custom conditions.
"""

from typing import Optional, Dict, Any, Callable, Union, List
import numpy as np

from .base import SampleFilter


class MetadataFilter(SampleFilter):
    """
    Filter samples based on metadata column values.

    This filter allows excluding samples based on external metadata (not X or y)
    using custom condition functions. It's useful for filtering based on:
    - Sample quality flags
    - Acquisition conditions
    - Sample categories to exclude
    - Date/time-based filtering
    - Any other metadata criteria

    The filter works with metadata passed during get_mask() call, as metadata
    is not part of the standard sklearn X, y interface.

    Attributes:
        column (str): Metadata column name to filter on
        condition (Callable): Function returning True for samples to KEEP
        values_to_exclude (List): List of values that should be excluded
        values_to_keep (List): List of values that should be kept

    Example:
        >>> from nirs4all.operators.filters import MetadataFilter
        >>>
        >>> # Exclude specific values
        >>> filter_obj = MetadataFilter(
        ...     column="quality_flag",
        ...     values_to_exclude=["bad", "corrupted"]
        ... )
        >>>
        >>> # Keep only specific values
        >>> filter_keep = MetadataFilter(
        ...     column="sample_type",
        ...     values_to_keep=["control", "treatment"]
        ... )
        >>>
        >>> # Custom condition
        >>> filter_custom = MetadataFilter(
        ...     column="temperature",
        ...     condition=lambda x: 20 <= x <= 30  # Keep 20-30°C
        ... )
        >>>
        >>> # Get mask (metadata must be provided)
        >>> mask = filter_obj.get_mask(X, metadata=metadata_df)

    In Pipeline:
        >>> pipeline = [
        ...     {
        ...         "sample_filter": {
        ...             "filters": [
        ...                 MetadataFilter(
        ...                     column="quality",
        ...                     values_to_exclude=["bad"]
        ...                 )
        ...             ],
        ...         }
        ...     },
        ...     "snv",
        ...     "model:PLSRegression",
        ... ]
    """

    def __init__(
        self,
        column: str,
        condition: Optional[Callable[[Any], bool]] = None,
        values_to_exclude: Optional[List[Any]] = None,
        values_to_keep: Optional[List[Any]] = None,
        exclude_missing: bool = True,
        reason: Optional[str] = None
    ):
        """
        Initialize the metadata filter.

        Exactly one of `condition`, `values_to_exclude`, or `values_to_keep`
        must be provided.

        Args:
            column: Name of the metadata column to filter on.
            condition: Custom function that takes a value and returns True to KEEP
                      the sample, False to exclude. Applied element-wise.
            values_to_exclude: List of values that should be excluded.
                              Samples with these values are marked for exclusion.
            values_to_keep: List of values that should be kept.
                           Only samples with these values are kept.
            exclude_missing: Whether to exclude samples with missing/None values.
                            Default True.
            reason: Custom exclusion reason. Defaults to filter description.

        Raises:
            ValueError: If column is not provided.
            ValueError: If none or multiple of condition/values_to_exclude/values_to_keep are set.
        """
        super().__init__(reason=reason)
        self.column = column
        self.condition = condition
        self.values_to_exclude = values_to_exclude
        self.values_to_keep = values_to_keep
        self.exclude_missing = exclude_missing

        # Validate column
        if not column:
            raise ValueError("column must be provided")

        # Validate that exactly one filtering criterion is set
        criteria_set = sum([
            condition is not None,
            values_to_exclude is not None,
            values_to_keep is not None
        ])

        if criteria_set == 0:
            raise ValueError(
                "One of 'condition', 'values_to_exclude', or 'values_to_keep' must be provided"
            )
        if criteria_set > 1:
            raise ValueError(
                "Only one of 'condition', 'values_to_exclude', or 'values_to_keep' can be set"
            )

        # Convert lists to sets for faster lookup
        self._exclude_set = set(values_to_exclude) if values_to_exclude else None
        self._keep_set = set(values_to_keep) if values_to_keep else None

    @property
    def exclusion_reason(self) -> str:
        """Get descriptive exclusion reason."""
        if self.reason is not None:
            return self.reason
        return f"metadata_{self.column}"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MetadataFilter":
        """
        Fit the filter (no-op for metadata filter).

        Metadata filtering uses fixed criteria, so no fitting is required.

        Args:
            X: Feature array (not used).
            y: Target array (not used).

        Returns:
            self: The filter instance (unchanged).
        """
        return self

    def get_mask(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        metadata: Optional[Union[Dict[str, np.ndarray], Any]] = None
    ) -> np.ndarray:
        """
        Compute boolean mask indicating which samples to KEEP.

        Args:
            X: Feature array of shape (n_samples, n_features).
               Used only to determine number of samples if metadata is not provided.
            y: Target array (not used).
            metadata: Metadata dictionary, DataFrame, or object with column access.
                     Must contain the specified column. Can be:
                     - Dict[str, np.ndarray]: metadata[column] returns array
                     - pd.DataFrame: metadata[column] returns series
                     - Any object with __getitem__ that returns array-like

        Returns:
            np.ndarray: Boolean array of shape (n_samples,) where:
                       - True means KEEP the sample
                       - False means EXCLUDE the sample

        Raises:
            ValueError: If metadata is None and filtering requires it.
            KeyError: If the specified column is not in metadata.
        """
        X = np.asarray(X)
        n_samples = len(X)

        if metadata is None:
            raise ValueError(
                f"MetadataFilter requires metadata containing column '{self.column}'. "
                "Pass metadata to get_mask()."
            )

        # Extract column values
        try:
            column_values = metadata[self.column]
            # Convert to numpy array if needed (e.g., pandas Series)
            if hasattr(column_values, 'to_numpy'):
                column_values = column_values.to_numpy()
            elif hasattr(column_values, 'values') and not isinstance(column_values, np.ndarray):
                column_values = column_values.values
            column_values = np.asarray(column_values)
        except (KeyError, TypeError) as e:
            raise KeyError(
                f"Column '{self.column}' not found in metadata. "
                f"Available: {list(metadata.keys()) if hasattr(metadata, 'keys') else 'unknown'}"
            ) from e

        if len(column_values) != n_samples:
            raise ValueError(
                f"Metadata column length ({len(column_values)}) does not match "
                f"number of samples ({n_samples})"
            )

        # Apply filtering - use vectorized operations where possible
        mask = np.ones(n_samples, dtype=bool)

        if self.condition is not None:
            # Apply custom condition - try vectorized first, fallback to element-wise
            try:
                # Attempt vectorized application
                mask = np.array([self.condition(val) for val in column_values], dtype=bool)
            except Exception:
                # Fallback to element-wise with error handling
                for i, val in enumerate(column_values):
                    try:
                        mask[i] = self.condition(val)
                    except Exception:
                        mask[i] = False

        elif self._exclude_set is not None:
            # Vectorized: exclude samples with specified values
            mask = ~np.isin(column_values, list(self._exclude_set))

        elif self._keep_set is not None:
            # Vectorized: keep only samples with specified values
            mask = np.isin(column_values, list(self._keep_set))

        # Handle missing values (vectorized)
        if self.exclude_missing:
            # Handle None values (convert to object array first if needed)
            is_none = np.array([v is None for v in column_values], dtype=bool)
            # Handle NaN for numeric arrays
            try:
                is_nan = np.isnan(column_values.astype(float))
            except (ValueError, TypeError):
                is_nan = np.zeros(n_samples, dtype=bool)
            mask &= ~(is_none | is_nan)

        return mask

    def get_filter_stats(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about filter application.

        Args:
            X: Feature array.
            y: Target array (unused).
            metadata: Metadata dictionary.

        Returns:
            Dict containing:
                - Base stats (n_samples, n_excluded, n_kept, exclusion_rate)
                - column: Filtered column name
                - filtering_type: Type of filtering applied
                - value_counts: Count of unique values (if available)
        """
        if metadata is not None:
            mask = self.get_mask(X, y, metadata=metadata)
        else:
            # Return minimal stats if no metadata
            return {
                "n_samples": len(X),
                "n_excluded": 0,
                "n_kept": len(X),
                "exclusion_rate": 0.0,
                "reason": self.exclusion_reason,
                "column": self.column,
                "note": "Metadata not provided - no filtering applied",
            }

        n_samples = len(mask)
        n_kept = np.sum(mask)
        n_excluded = n_samples - n_kept

        stats = {
            "n_samples": n_samples,
            "n_excluded": n_excluded,
            "n_kept": n_kept,
            "exclusion_rate": n_excluded / n_samples if n_samples > 0 else 0.0,
            "reason": self.exclusion_reason,
            "column": self.column,
            "exclude_missing": self.exclude_missing,
        }

        # Add filtering type info
        if self.condition is not None:
            stats["filtering_type"] = "condition"
        elif self._exclude_set is not None:
            stats["filtering_type"] = "values_to_exclude"
            stats["excluded_values"] = list(self._exclude_set)
        elif self._keep_set is not None:
            stats["filtering_type"] = "values_to_keep"
            stats["kept_values"] = list(self._keep_set)

        # Add value distribution if metadata available
        if metadata is not None:
            try:
                column_values = metadata[self.column]
                if hasattr(column_values, 'to_numpy'):
                    column_values = column_values.to_numpy()
                elif hasattr(column_values, 'values') and not isinstance(column_values, np.ndarray):
                    column_values = column_values.values
                unique, counts = np.unique(column_values, return_counts=True)
                stats["value_distribution"] = dict(zip(
                    [str(u) for u in unique],
                    [int(c) for c in counts]
                ))
            except Exception:
                pass

        return stats

    def __repr__(self) -> str:
        """Return string representation."""
        params = [f"column='{self.column}'"]

        if self.condition is not None:
            params.append("condition=<function>")
        elif self.values_to_exclude is not None:
            if len(self.values_to_exclude) <= 3:
                params.append(f"values_to_exclude={self.values_to_exclude}")
            else:
                params.append(f"values_to_exclude=[{len(self.values_to_exclude)} values]")
        elif self.values_to_keep is not None:
            if len(self.values_to_keep) <= 3:
                params.append(f"values_to_keep={self.values_to_keep}")
            else:
                params.append(f"values_to_keep=[{len(self.values_to_keep)} values]")

        return f"{self.__class__.__name__}({', '.join(params)})"
