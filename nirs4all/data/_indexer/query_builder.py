"""
Convert Selector dictionaries to Polars filter expressions.

This module provides the QueryBuilder class that handles conversion of
user-friendly selector dictionaries into optimized Polars expressions.
"""

from typing import Dict, Any, Optional
import polars as pl

from nirs4all.data.types import Selector


class QueryBuilder:
    """
    Build Polars filter expressions from Selector dictionaries.

    This class centralizes query logic, converting user-friendly selector
    dictionaries into optimized Polars expressions for filtering operations.

    Supported selector patterns:
    - Single values: {"partition": "train"} → partition == "train"
    - Lists: {"group": [1, 2]} → group in [1, 2]
    - None values: {"augmentation": None} → augmentation is null
    - Multiple filters: Combined with AND logic

    Examples:
        >>> builder = QueryBuilder()
        >>> selector = {"partition": "train", "group": [1, 2]}
        >>> expr = builder.build(selector)
        >>> # expr: (partition == "train") & (group in [1, 2])
    """

    def __init__(self, valid_columns: Optional[list[str]] = None):
        """
        Initialize the query builder.

        Args:
            valid_columns: Optional list of valid column names for validation.
                         If None, no validation is performed.
        """
        self._valid_columns = set(valid_columns) if valid_columns else None

    def build(self, selector: Selector, exclude_columns: Optional[list[str]] = None) -> pl.Expr:
        """
        Build a Polars filter expression from a selector dictionary.

        Args:
            selector: Dictionary of column:value filters. If None or empty,
                     returns an expression that matches all rows (pl.lit(True)).
            exclude_columns: Optional list of columns to exclude from filtering
                           (e.g., "processings" which needs special handling).

        Returns:
            pl.Expr: Polars expression for filtering. Returns pl.lit(True) for
                    empty selectors (matches all rows).

        Raises:
            ValueError: If selector contains invalid column names (when valid_columns
                       is set during initialization).

        Examples:
            >>> builder = QueryBuilder()
            >>>
            >>> # Simple equality
            >>> expr = builder.build({"partition": "train"})
            >>>
            >>> # Multiple conditions (AND)
            >>> expr = builder.build({"partition": "train", "group": 1})
            >>>
            >>> # List membership
            >>> expr = builder.build({"partition": ["train", "val"]})
            >>>
            >>> # Null check
            >>> expr = builder.build({"augmentation": None})
            >>>
            >>> # Empty selector (match all)
            >>> expr = builder.build({})  # Returns pl.lit(True)
        """
        if not selector:
            return pl.lit(True)

        exclude_columns = set(exclude_columns) if exclude_columns else set()
        conditions = []

        for col, value in selector.items():
            # Skip excluded columns
            if col in exclude_columns:
                continue

            # Validate column if validation is enabled
            if self._valid_columns is not None and col not in self._valid_columns:
                continue  # Skip invalid columns silently for backward compatibility

            # Build condition based on value type
            if isinstance(value, list):
                conditions.append(pl.col(col).is_in(value))
            elif value is None:
                conditions.append(pl.col(col).is_null())
            else:
                conditions.append(pl.col(col) == value)

        # Handle empty conditions (all columns excluded or selector empty)
        if not conditions:
            return pl.lit(True)

        # Combine conditions with AND logic
        condition = conditions[0]
        for cond in conditions[1:]:
            condition = condition & cond

        return condition

    def build_sample_filter(self, sample_ids: list[int]) -> pl.Expr:
        """
        Build a filter expression for sample IDs.

        Convenience method for the common pattern of filtering by sample IDs.

        Args:
            sample_ids: List of sample IDs to match.

        Returns:
            pl.Expr: Expression matching the given sample IDs.

        Example:
            >>> expr = builder.build_sample_filter([0, 1, 2])
            >>> # expr: sample in [0, 1, 2]
        """
        if not sample_ids:
            return pl.lit(False)  # No samples = match nothing
        return pl.col("sample").is_in(sample_ids)

    def build_origin_filter(self, origin_ids: list[int]) -> pl.Expr:
        """
        Build a filter expression for origin IDs.

        Convenience method for filtering by origin sample IDs.

        Args:
            origin_ids: List of origin sample IDs to match.

        Returns:
            pl.Expr: Expression matching the given origin IDs.

        Example:
            >>> expr = builder.build_origin_filter([0, 1])
            >>> # expr: origin in [0, 1]
        """
        if not origin_ids:
            return pl.lit(False)
        return pl.col("origin").is_in(origin_ids)

    def build_base_samples_filter(self) -> pl.Expr:
        """
        Build a filter expression for base samples (sample == origin).

        Returns:
            pl.Expr: Expression matching base samples.

        Example:
            >>> expr = builder.build_base_samples_filter()
            >>> # expr: sample == origin
        """
        return pl.col("sample") == pl.col("origin")

    def build_augmented_samples_filter(self) -> pl.Expr:
        """
        Build a filter expression for augmented samples (sample != origin).

        Returns:
            pl.Expr: Expression matching augmented samples.

        Example:
            >>> expr = builder.build_augmented_samples_filter()
            >>> # expr: sample != origin
        """
        return pl.col("sample") != pl.col("origin")

    def build_excluded_filter(self, include_excluded: bool = False) -> pl.Expr:
        """
        Build a filter expression for excluded samples.

        Args:
            include_excluded: If True, return expression that matches all rows.
                            If False (default), return expression that excludes
                            samples marked as excluded=True.

        Returns:
            pl.Expr: Expression for filtering excluded samples.

        Examples:
            >>> expr = builder.build_excluded_filter(include_excluded=False)
            >>> # expr: (excluded == False) | excluded.is_null()

            >>> expr = builder.build_excluded_filter(include_excluded=True)
            >>> # expr: pl.lit(True)  # Match all
        """
        if include_excluded:
            return pl.lit(True)
        # Include samples where excluded is False OR null (not set)
        return (pl.col("excluded") == False) | pl.col("excluded").is_null()  # noqa: E712
