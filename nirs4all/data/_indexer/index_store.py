"""
Low-level DataFrame storage and query execution for the indexer.

This module provides the IndexStore class that encapsulates all DataFrame
operations, providing a clean abstraction over Polars-specific details.
"""

from typing import Dict, List, Any, Optional
import polars as pl


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

    def __init__(self):
        """Initialize the index store with an empty DataFrame."""
        # Enable StringCache for consistent categorical encodings
        pl.enable_string_cache()

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
        return self._df

    @property
    def columns(self) -> List[str]:
        """
        Get list of column names.

        Returns:
            List[str]: Column names in the DataFrame.
        """
        return self._df.columns

    @property
    def schema(self) -> Dict[str, pl.DataType]:
        """
        Get the DataFrame schema.

        Returns:
            Dict[str, pl.DataType]: Mapping of column names to Polars data types.
        """
        return self._df.schema

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
        return self._df.filter(condition)

    def append(self, data: Dict[str, pl.Series]) -> None:
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

    def update_by_condition(self, condition: pl.Expr, updates: Dict[str, Any]) -> None:
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

    def get_column(self, col: str, condition: Optional[pl.Expr] = None) -> List[Any]:
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
        return df.select(pl.col(col)).to_series().to_list()

    def get_unique(self, col: str) -> List[Any]:
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
        return self._df.select(pl.col(col)).unique().to_series().to_list()

    def get_max(self, col: str) -> Optional[int]:
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
        return int(max_val) if max_val is not None else None

    def __repr__(self) -> str:
        """String representation showing the DataFrame."""
        return str(self._df)
