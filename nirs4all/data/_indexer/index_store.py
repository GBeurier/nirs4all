"""
Low-level DataFrame storage and query execution for the indexer.

This module provides the IndexStore class that encapsulates all DataFrame
operations, providing a clean abstraction over Polars-specific details.
"""

from typing import Any, Optional, Union

import polars as pl

# Map string dtype names to Polars types for tag columns
TAG_DTYPE_MAP: dict[str, pl.DataType] = {
    "bool": pl.Boolean(),
    "boolean": pl.Boolean(),
    "str": pl.Utf8(),
    "string": pl.Utf8(),
    "int": pl.Int32(),
    "int32": pl.Int32(),
    "int64": pl.Int64(),
    "float": pl.Float64(),
    "float32": pl.Float32(),
    "float64": pl.Float64(),
}

class IndexStore:
    """
    Low-level DataFrame storage and query execution.

    This class encapsulates all direct interactions with the Polars DataFrame,
    providing a clean interface for storage operations. It handles:
    - DataFrame initialization and schema management
    - Row insertion and updates
    - Filtered queries
    - Column access and statistics

    The IndexStore is backend-agnostic in design, though currently implemented
    with Polars for optimal performance.
    """

    # Core schema columns (not tag columns)
    CORE_COLUMNS = frozenset({
        "row", "sample", "origin", "partition", "group", "branch",
        "processings", "augmentation", "excluded", "exclusion_reason"
    })

    def __init__(self) -> None:
        """Initialize the index store with an empty DataFrame."""
        # Enable StringCache for consistent categorical encodings
        pl.enable_string_cache()

        # Track dynamic tag columns (name -> dtype)
        self._tag_columns: dict[str, pl.DataType] = {}

        # Initialize DataFrame with proper schema
        self._df = pl.DataFrame({
            "row": pl.Series([], dtype=pl.Int32),
            "sample": pl.Series([], dtype=pl.Int32),
            "origin": pl.Series([], dtype=pl.Int32),
            "partition": pl.Series([], dtype=pl.Categorical),
            "group": pl.Series([], dtype=pl.Int8),
            "branch": pl.Series([], dtype=pl.Int8),
            "processings": pl.Series([], dtype=pl.List(pl.Utf8)),  # Native list type!
            "augmentation": pl.Series([], dtype=pl.Categorical),
            "excluded": pl.Series([], dtype=pl.Boolean),  # Sample filtering flag
            "exclusion_reason": pl.Series([], dtype=pl.Utf8),  # Filtering reason
        })

    @property
    def df(self) -> pl.DataFrame:
        """
        Get the underlying DataFrame.

        Returns:
            pl.DataFrame: The complete index DataFrame.

        Note:
            Direct DataFrame access is provided for backward compatibility
            and advanced use cases. Prefer using query methods when possible.
        """
        df: pl.DataFrame = self._df
        return df

    @property
    def columns(self) -> list[str]:
        """
        Get list of column names.

        Returns:
            List[str]: Column names in the DataFrame.
        """
        return list(self._df.columns)

    @property
    def schema(self) -> dict[str, pl.DataType]:
        """
        Get the DataFrame schema.

        Returns:
            Dict[str, pl.DataType]: Mapping of column names to Polars data types.
        """
        return dict(self._df.schema)

    def __len__(self) -> int:
        """
        Get number of rows in the store.

        Returns:
            int: Number of rows.
        """
        return len(self._df)

    def query(self, condition: pl.Expr) -> pl.DataFrame:
        """
        Execute a filtered query.

        Args:
            condition: Polars expression for filtering.

        Returns:
            pl.DataFrame: Filtered DataFrame.

        Example:
            >>> condition = pl.col("partition") == "train"
            >>> train_data = store.query(condition)
        """
        result: pl.DataFrame = self._df.filter(condition)
        return result

    def append(self, data: dict[str, pl.Series]) -> None:
        """
        Append new rows to the DataFrame.

        Args:
            data: Dictionary mapping column names to Polars Series.

        Raises:
            ValueError: If data columns don't match schema.

        Example:
            >>> data = {
            ...     "row": pl.Series([0, 1], dtype=pl.Int32),
            ...     "sample": pl.Series([0, 1], dtype=pl.Int32),
            ...     # ... other columns
            ... }
            >>> store.append(data)
        """
        new_df = pl.DataFrame(data)
        self._df = pl.concat([self._df, new_df], how="vertical")

    def update_by_condition(self, condition: pl.Expr, updates: dict[str, Any]) -> None:
        """
        Update rows matching a condition.

        Args:
            condition: Polars expression identifying rows to update.
            updates: Dictionary of column:value pairs to update.

        Example:
            >>> condition = pl.col("partition") == "train"
            >>> store.update_by_condition(condition, {"group": 1})
        """
        for col, value in updates.items():
            # Cast the literal value to the expected column type
            cast_value = pl.lit(value).cast(self._df.schema[col])
            self._df = self._df.with_columns(
                pl.when(condition).then(cast_value).otherwise(pl.col(col)).alias(col)
            )

    def get_column(self, col: str, condition: pl.Expr | None = None) -> list[Any]:
        """
        Get column values, optionally filtered.

        Args:
            col: Column name to retrieve.
            condition: Optional filter condition.

        Returns:
            List[Any]: Column values.

        Raises:
            ValueError: If column doesn't exist.

        Example:
            >>> # Get all partitions
            >>> partitions = store.get_column("partition")
            >>> # Get train sample IDs
            >>> train_samples = store.get_column("sample", pl.col("partition") == "train")
        """
        if col not in self._df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

        df = self._df.filter(condition) if condition is not None else self._df
        return list(df.select(pl.col(col)).to_series().to_list())

    def get_unique(self, col: str) -> list[Any]:
        """
        Get unique values in a column.

        Args:
            col: Column name.

        Returns:
            List[Any]: Unique values in the column.

        Raises:
            ValueError: If column doesn't exist.

        Example:
            >>> unique_partitions = store.get_unique("partition")
        """
        if col not in self._df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
        return list(self._df.select(pl.col(col)).unique().to_series().to_list())

    def get_max(self, col: str) -> int | None:
        """
        Get maximum value in a column.

        Args:
            col: Column name.

        Returns:
            Optional[int]: Maximum value, or None if column is empty.

        Example:
            >>> max_sample_id = store.get_max("sample")
        """
        if len(self._df) == 0:
            return None
        max_val = self._df[col].max()
        if max_val is None:
            return None
        if isinstance(max_val, (int, float)):
            return int(max_val)
        return int(str(max_val))

    def __repr__(self) -> str:
        """String representation showing the DataFrame."""
        return str(self._df)

    # ==================== Tag Column Methods ====================

    def add_tag_column(self, name: str, dtype: str | pl.DataType | None = None) -> None:
        """
        Add a new tag column to the DataFrame.

        Tag columns are dynamic columns used for sample tagging and filtering
        (e.g., marking outliers, clusters, quality scores).

        Args:
            name: Column name for the tag. Must not conflict with core columns.
            dtype: Data type for the column. Can be:
                  - Polars type: pl.Boolean, pl.Utf8, pl.Int32, pl.Float64
                  - String: "bool", "str", "int", "float"

        Raises:
            ValueError: If name conflicts with core columns or tag already exists.

        Example:
            >>> store.add_tag_column("is_outlier", pl.Boolean)
            >>> store.add_tag_column("cluster_id", "int")
            >>> store.add_tag_column("quality_score", pl.Float64)
        """
        if dtype is None:
            dtype = pl.Boolean()

        # Resolve string dtype to Polars type
        if isinstance(dtype, str):
            dtype_lower = dtype.lower()
            if dtype_lower not in TAG_DTYPE_MAP:
                raise ValueError(
                    f"Unknown dtype '{dtype}'. Supported: {list(TAG_DTYPE_MAP.keys())}"
                )
            dtype = TAG_DTYPE_MAP[dtype_lower]

        # Validate name doesn't conflict with core columns
        if name in self.CORE_COLUMNS:
            raise ValueError(
                f"Tag column name '{name}' conflicts with core column. "
                f"Core columns: {sorted(self.CORE_COLUMNS)}"
            )

        # Check for duplicate tag column
        if name in self._tag_columns:
            raise ValueError(
                f"Tag column '{name}' already exists. "
                f"Use remove_tag_column() first if you want to replace it."
            )

        # Check if column already exists in DataFrame (shouldn't happen, but safety check)
        if name in self._df.columns:
            raise ValueError(
                f"Column '{name}' already exists in DataFrame."
            )

        # Add column to DataFrame with null values
        self._df = self._df.with_columns(
            pl.lit(None).cast(dtype).alias(name)
        )

        # Track in tag columns registry
        self._tag_columns[name] = dtype

    def set_tags(
        self,
        indices: list[int],
        tag_name: str,
        values: Any | list[Any]
    ) -> None:
        """
        Set tag values for specific sample indices.

        Args:
            indices: List of sample indices to update.
            tag_name: Name of the tag column.
            values: Value(s) to set. Can be:
                   - Single value: Applied to all indices
                   - List: Must match length of indices

        Raises:
            ValueError: If tag column doesn't exist or values length mismatch.

        Example:
            >>> store.set_tags([0, 1, 2], "is_outlier", True)
            >>> store.set_tags([0, 1], "cluster_id", [1, 2])
        """
        if tag_name not in self._tag_columns:
            raise ValueError(
                f"Tag column '{tag_name}' not found. "
                f"Available tags: {list(self._tag_columns.keys())}"
            )

        if not indices:
            return

        # Normalize values to list
        if not isinstance(values, list):
            values = [values] * len(indices)
        elif len(values) != len(indices):
            raise ValueError(
                f"Values length ({len(values)}) must match indices length ({len(indices)})"
            )

        # Build condition for sample indices
        condition = pl.col("sample").is_in(indices)

        # Create a mapping from sample index to value
        index_to_value = dict(zip(indices, values, strict=False))

        # Update using map_elements for efficient bulk update
        dtype = self._tag_columns[tag_name]
        self._df = self._df.with_columns(
            pl.when(condition)
            .then(
                pl.col("sample").map_elements(
                    lambda s: index_to_value.get(s),
                    return_dtype=dtype
                )
            )
            .otherwise(pl.col(tag_name))
            .alias(tag_name)
        )

    def get_tags(
        self,
        tag_name: str,
        condition: pl.Expr | None = None
    ) -> list[Any]:
        """
        Get tag values, optionally filtered by condition.

        Args:
            tag_name: Name of the tag column.
            condition: Optional Polars expression for filtering.

        Returns:
            List of tag values.

        Raises:
            ValueError: If tag column doesn't exist.

        Example:
            >>> outliers = store.get_tags("is_outlier")
            >>> train_outliers = store.get_tags("is_outlier", pl.col("partition") == "train")
        """
        if tag_name not in self._tag_columns:
            raise ValueError(
                f"Tag column '{tag_name}' not found. "
                f"Available tags: {list(self._tag_columns.keys())}"
            )

        return self.get_column(tag_name, condition)

    def get_tag_column_names(self) -> list[str]:
        """
        Get list of all tag column names.

        Returns:
            List of tag column names.

        Example:
            >>> tags = store.get_tag_column_names()
            >>> print(tags)  # ['is_outlier', 'cluster_id']
        """
        return list(self._tag_columns.keys())

    def remove_tag_column(self, name: str) -> None:
        """
        Remove a tag column from the DataFrame.

        Args:
            name: Name of the tag column to remove.

        Raises:
            ValueError: If tag column doesn't exist.

        Example:
            >>> store.remove_tag_column("is_outlier")
        """
        if name not in self._tag_columns:
            raise ValueError(
                f"Tag column '{name}' not found. "
                f"Available tags: {list(self._tag_columns.keys())}"
            )

        # Remove from DataFrame
        self._df = self._df.drop(name)

        # Remove from registry
        del self._tag_columns[name]

    def has_tag_column(self, name: str) -> bool:
        """
        Check if a tag column exists.

        Args:
            name: Name of the tag column.

        Returns:
            True if tag column exists.
        """
        return name in self._tag_columns

    def get_tag_dtype(self, name: str) -> pl.DataType:
        """
        Get the data type of a tag column.

        Args:
            name: Name of the tag column.

        Returns:
            Polars DataType of the tag column.

        Raises:
            ValueError: If tag column doesn't exist.
        """
        if name not in self._tag_columns:
            raise ValueError(
                f"Tag column '{name}' not found. "
                f"Available tags: {list(self._tag_columns.keys())}"
            )
        return self._tag_columns[name]

    # ==================== Serialization Methods ====================

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the IndexStore to a dictionary.

        Returns:
            Dictionary containing:
            - data: DataFrame data as dict of lists
            - tag_columns: Tag column metadata (name -> dtype string)
            - schema: Full schema for validation

        Example:
            >>> state = store.to_dict()
            >>> # ... save state ...
            >>> new_store = IndexStore.from_dict(state)
        """
        # Convert DataFrame to dict
        data = self._df.to_dict(as_series=False)

        # Serialize tag column metadata
        tag_columns = {}
        for name, dtype in self._tag_columns.items():
            # Convert Polars dtype to string representation
            tag_columns[name] = self._dtype_to_string(dtype)

        return {
            "data": data,
            "tag_columns": tag_columns,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "IndexStore":
        """
        Deserialize an IndexStore from a dictionary.

        Args:
            state: Dictionary from to_dict().

        Returns:
            New IndexStore instance with restored state.

        Example:
            >>> state = store.to_dict()
            >>> new_store = IndexStore.from_dict(state)
        """
        instance = cls()

        # Restore tag columns first (so schema is complete before loading data)
        tag_columns = state.get("tag_columns", {})
        for name, dtype_str in tag_columns.items():
            dtype = cls._string_to_dtype(dtype_str)
            instance._tag_columns[name] = dtype

        # Load data
        data = state.get("data", {})
        if data:
            # Build DataFrame from data
            # Need to handle processings (List type) specially
            series_dict = {}
            for col, values in data.items():
                if col == "processings":
                    # Native list type
                    series_dict[col] = pl.Series(values, dtype=pl.List(pl.Utf8))
                elif col in ("partition", "augmentation"):
                    # Categorical type
                    series_dict[col] = pl.Series(values, dtype=pl.Categorical)
                elif col in instance._tag_columns:
                    # Tag column with tracked dtype
                    series_dict[col] = pl.Series(values, dtype=instance._tag_columns[col])
                elif col in ("row", "sample", "origin"):
                    series_dict[col] = pl.Series(values, dtype=pl.Int32)
                elif col in ("group", "branch"):
                    series_dict[col] = pl.Series(values, dtype=pl.Int8)
                elif col == "excluded":
                    series_dict[col] = pl.Series(values, dtype=pl.Boolean)
                elif col == "exclusion_reason":
                    series_dict[col] = pl.Series(values, dtype=pl.Utf8)
                else:
                    # Unknown column, let Polars infer
                    series_dict[col] = pl.Series(values)

            instance._df = pl.DataFrame(series_dict)

        return instance

    @staticmethod
    def _dtype_to_string(dtype: pl.DataType) -> str:
        """Convert Polars dtype to string representation."""
        dtype_str = str(dtype)
        # Map common types to simple strings
        # Note: Polars may return "String" or "Utf8" depending on version
        mapping = {
            "Boolean": "bool",
            "Utf8": "str",
            "String": "str",  # Polars 1.0+ uses String instead of Utf8
            "Int32": "int32",
            "Int64": "int64",
            "Float32": "float32",
            "Float64": "float64",
        }
        return mapping.get(dtype_str, dtype_str)

    @staticmethod
    def _string_to_dtype(dtype_str: str) -> pl.DataType:
        """Convert string representation to Polars dtype."""
        if dtype_str in TAG_DTYPE_MAP:
            return TAG_DTYPE_MAP[dtype_str]
        # Try to handle other cases (including capitalized versions)
        reverse_mapping: dict[str, pl.DataType] = {
            "Boolean": pl.Boolean(),
            "Utf8": pl.Utf8(),
            "String": pl.Utf8(),  # Polars 1.0+ uses String instead of Utf8
            "Int32": pl.Int32(),
            "Int64": pl.Int64(),
            "Float32": pl.Float32(),
            "Float64": pl.Float64(),
        }
        if dtype_str in reverse_mapping:
            return reverse_mapping[dtype_str]
        raise ValueError(f"Unknown dtype string: {dtype_str}")
