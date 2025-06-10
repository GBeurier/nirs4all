from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import polars as pl

from feature_source import FeatureSource


class Features:
    """Manages N aligned NumPy sources + a Polars index."""

    def __init__(self):
        """Initialize empty feature block."""
        self.sources: List[FeatureSource] = []
        self.index: Optional[pl.DataFrame] = None
        self._filtered_cache: Dict[str, pl.DataFrame] = {}  # Cache for frequent filters

    def add_features(self, x_list: List[np.ndarray]) -> None:
        """
        Add feature arrays as new sources.

        Args:
            x_list: List of 2D numpy arrays, all with the same number of rows
        """
        if not x_list:
            raise ValueError("x_list cannot be empty")

        n_rows = x_list[0].shape[0]
        for i, arr in enumerate(x_list):
            if arr.shape[0] != n_rows:
                raise ValueError(f"Array {i} has {arr.shape[0]} rows, expected {n_rows}")

        if self.sources and self.sources[0].num_samples != n_rows:
            raise ValueError(f"New arrays have {n_rows} rows, existing sources have {self.sources[0].num_samples}")

        for arr in x_list:
            self.sources.append(FeatureSource(arr))

        if self.index is None:
            self._build_initial_index(n_rows)

    def _build_initial_index(self, n_rows: int) -> None:
        """Build the initial index DataFrame with optimized data types."""
        self.index = pl.DataFrame({
            "row": pl.Series(range(n_rows), dtype=pl.Int32),
            "sample": pl.Series(range(n_rows), dtype=pl.Int32),
            "origin": pl.Series(range(n_rows), dtype=pl.Int32),
            "partition": pl.Series(["train"] * n_rows, dtype=pl.Categorical),
            "group": pl.Series([0] * n_rows, dtype=pl.Int8),
            "branch": pl.Series([0] * n_rows, dtype=pl.Int8),
            "processing": pl.Series(["raw"] * n_rows, dtype=pl.Categorical)
        })

    @property
    def num_samples(self) -> int:
        """Get the number of samples (rows) across all sources."""
        if not self.sources:
            return 0
        return self.sources[0].num_samples

    def __repr__(self):
        n_sources = len(self.sources)
        n_samples = self.num_samples
        return f"FeatureBlock(sources={n_sources}, samples={n_samples})"

    def _apply_filter(self, filter_dict: Dict[str, Any]) -> pl.DataFrame:
        """
        Apply filter to index DataFrame efficiently and return filtered view.

        Args:
            filter_dict: Dictionary of column: value pairs for exact matching

        Returns:
            Filtered DataFrame view
        """
        if self.index is None:
            raise ValueError("No index available - add features first")

        # Generate a cache key from filter_dict
        cache_key = str(sorted(filter_dict.items()))
        if cache_key in self._filtered_cache:
            return self._filtered_cache[cache_key]

        # Combine all conditions into a single expression
        conditions = []
        for column, value in filter_dict.items():
            if column not in self.index.columns:
                raise ValueError(f"Column '{column}' not found in index")
            if isinstance(value, (list, tuple, np.ndarray)):
                conditions.append(pl.col(column).is_in(value))
            else:
                conditions.append(pl.col(column) == value)

        # Combine conditions with logical AND
        if len(conditions) == 1:
            combined_condition = conditions[0]
        else:
            # Use fold to combine multiple conditions with AND
            combined_condition = pl.fold(
                acc=conditions[0],
                function=lambda acc, x: acc & x,
                exprs=conditions[1:]
            )

        # Apply all filters at once with lazy evaluation
        filtered_df = self.index.lazy().filter(combined_condition).select("row").collect()

        # Cache the result for reuse
        self._filtered_cache[cache_key] = filtered_df
        return filtered_df

    def _get_row_indices(self, filtered_df: pl.DataFrame) -> np.ndarray:
        """Get row indices from filtered DataFrame."""
        return filtered_df["row"].to_numpy()

    def x(self, filter_dict: Dict[str, Any], layout: str = "2d", src_concat: bool = False) -> np.ndarray | Tuple[np.ndarray, ...]:
        """
        Get feature arrays with specified layout and filtering.

        Args:
            filter_dict: Dictionary of column: value pairs for filtering
            layout: Layout type ("2d", "2d_interleaved", "3d", "3d_transpose")
            src_concat: Whether to concatenate sources along axis=1

        Returns:
            Tuple of numpy arrays (zero-copy views)
        """
        if not self.sources:
            raise ValueError("No features available")

        filtered_df = self._apply_filter(filter_dict)
        row_indices = self._get_row_indices(filtered_df)

        res = []
        for source in self.sources:
            if layout == "2d":
                res.append(source.layout_2d(row_indices))
            elif layout == "2d_interleaved":
                res.append(source.layout_2d_interleaved(row_indices))
            elif layout == "3d":
                res.append(source.layout_3d(row_indices))
            elif layout == "3d_transpose":
                res.append(source.layout_3d_transpose(row_indices))
            else:
                raise ValueError(f"Unknown layout: {layout}")

        if src_concat:
            return np.concatenate(res, axis=1)

        return tuple(res)

    def update_index(self, indices: np.ndarray | List[int], column: str, value: Any) -> None:
        """
        Update index DataFrame efficiently with new values for a specific column.

        Args:
            indices: Array of row indices to update
            column: Column name to update
            value: New value to set
        """
        if self.index is None:
            raise ValueError("No index available - add features first")

        if column not in self.index.columns:
            raise ValueError(f"Column '{column}' not found in index")

        indices = np.asarray(indices)
        if not np.all(np.isin(indices, self.index["row"].to_numpy())):
            raise ValueError("Some indices are out of bounds")

        # Perform update in a single vectorized operation
        self.index = self.index.with_columns(
            pl.when(pl.col("row").is_in(indices)).then(pl.lit(value)).otherwise(pl.col(column)).alias(column)
        )
        # Clear cache since index has changed
        self._filtered_cache.clear()