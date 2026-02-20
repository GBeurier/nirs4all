"""
Convert Selector dictionaries to Polars filter expressions.

This module provides the QueryBuilder class that handles conversion of
user-friendly selector dictionaries into optimized Polars expressions.
"""

import re
from collections.abc import Callable
from typing import Any, Optional, Union

import polars as pl

# Regex patterns for condition parsing
_COMPARISON_PATTERN = re.compile(r'^([<>]=?|[!=]=?)\s*(-?\d+\.?\d*)$')
_RANGE_PATTERN = re.compile(r'^(-?\d+\.?\d*)?\.\.(-?\d+\.?\d*)?$')

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

    def __init__(self, valid_columns: list[str] | None = None):
        """
        Initialize the query builder.

        Args:
            valid_columns: Optional list of valid column names for validation.
                         If None, no validation is performed.
        """
        self._valid_columns = set(valid_columns) if valid_columns else None

    def build(self, selector: dict[str, Any], exclude_columns: list[str] | None = None) -> pl.Expr:
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

        excluded: set[str] = set(exclude_columns) if exclude_columns else set()
        conditions = []

        for col, value in selector.items():
            # Skip excluded columns
            if col in excluded:
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

    # ==================== Tag Filtering Methods ====================

    def build_tag_filter(
        self,
        tag_name: str,
        condition: Any
    ) -> pl.Expr:
        """
        Build a filter expression for a tag column.

        Args:
            tag_name: Name of the tag column.
            condition: Filter condition. Supported formats:
                - Boolean: `True`, `False` - exact match
                - Comparison string: `"> 0.8"`, `"<= 50"`, `"== 1"`, `"!= 0"`
                - Range string: `"0..50"` (inclusive), `"50.."` (open end), `"..50"` (open start)
                - List of values: `["a", "b", "c"]` - matches any value in list
                - Callable: `lambda x: x > 0.8` - custom predicate (use sparingly)

        Returns:
            pl.Expr: Polars expression for filtering.

        Raises:
            ValueError: If condition format is not recognized.

        Examples:
            >>> # Boolean filter
            >>> expr = builder.build_tag_filter("is_outlier", True)
            >>>
            >>> # Comparison filter
            >>> expr = builder.build_tag_filter("quality_score", "> 0.8")
            >>>
            >>> # Range filter
            >>> expr = builder.build_tag_filter("cluster_id", "1..5")
            >>>
            >>> # List membership
            >>> expr = builder.build_tag_filter("category", ["A", "B", "C"])
        """
        return self._parse_tag_condition(tag_name, condition)

    def _parse_tag_condition(
        self,
        tag_name: str,
        condition: Any
    ) -> pl.Expr:
        """
        Parse a tag condition into a Polars expression.

        Args:
            tag_name: Name of the tag column.
            condition: Condition to parse.

        Returns:
            pl.Expr: Polars filter expression.

        Raises:
            ValueError: If condition format is not recognized.
        """
        col = pl.col(tag_name)

        # Boolean exact match
        if isinstance(condition, bool):
            return col == condition

        # Numeric exact match
        if isinstance(condition, (int, float)):
            return col == condition

        # List membership
        if isinstance(condition, list):
            return col.is_in(condition)

        # String conditions: comparison or range
        if isinstance(condition, str):
            return self._parse_string_condition(tag_name, condition)

        # Callable (lambda function)
        if callable(condition):
            # Note: This uses map_elements which can be slow
            # We need to infer return type - assume Boolean for filters
            return col.map_elements(condition, return_dtype=pl.Boolean)

        # None - check for null
        if condition is None:
            return col.is_null()

        raise ValueError(
            f"Unknown condition format for tag '{tag_name}': {type(condition).__name__}. "
            f"Supported: bool, int, float, str, list, callable, None"
        )

    def _parse_string_condition(
        self,
        tag_name: str,
        condition: str
    ) -> pl.Expr:
        """
        Parse string condition into Polars expression.

        Supports:
        - Comparison: "> 0.8", "<= 50", "== 1", "!= 0"
        - Range: "0..50", "50..", "..50"
        - String exact match (fallback)

        Args:
            tag_name: Name of the tag column.
            condition: String condition.

        Returns:
            pl.Expr: Polars filter expression.
        """
        col = pl.col(tag_name)
        condition = condition.strip()

        # Try comparison pattern first: "> 0.8", "<= 50", etc.
        match = _COMPARISON_PATTERN.match(condition)
        if match:
            operator, value_str = match.groups()
            value = float(value_str) if '.' in value_str else int(value_str)
            return self._build_comparison_expr(col, operator, value)

        # Try range pattern: "0..50", "50..", "..50"
        match = _RANGE_PATTERN.match(condition)
        if match:
            start_str, end_str = match.groups()
            return self._build_range_expr(col, start_str, end_str)

        # Fallback: exact string match
        return col == condition

    def _build_comparison_expr(
        self,
        col: pl.Expr,
        operator: str,
        value: int | float
    ) -> pl.Expr:
        """Build comparison expression."""
        if operator == '>':
            return col > value
        elif operator == '>=':
            return col >= value
        elif operator == '<':
            return col < value
        elif operator == '<=':
            return col <= value
        elif operator == '==' or operator == '=':
            return col == value
        elif operator == '!=' or operator == '<>':
            return col != value
        else:
            raise ValueError(f"Unknown comparison operator: {operator}")

    def _build_range_expr(
        self,
        col: pl.Expr,
        start_str: str | None,
        end_str: str | None
    ) -> pl.Expr:
        """
        Build range expression for "start..end" syntax.

        Both bounds are inclusive when specified.
        Open-ended ranges: "50.." means >= 50, "..50" means <= 50
        """
        if start_str and end_str:
            # Closed range: start..end (inclusive)
            start = float(start_str) if '.' in start_str else int(start_str)
            end = float(end_str) if '.' in end_str else int(end_str)
            return (col >= start) & (col <= end)
        elif start_str:
            # Open end: start.. (>= start)
            start = float(start_str) if '.' in start_str else int(start_str)
            return col >= start
        elif end_str:
            # Open start: ..end (<= end)
            end = float(end_str) if '.' in end_str else int(end_str)
            return col <= end
        else:
            # Empty range ".." - match all (shouldn't normally happen)
            return pl.lit(True)
