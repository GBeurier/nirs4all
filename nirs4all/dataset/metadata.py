"""
Metadata management for SpectroDataset.

This module contains Metadata class for managing sample-level auxiliary data.
Metadata has one row per sample and aligns with the indexer's row indices.
"""

from typing import Dict, List, Optional, Union, Literal, Any
import numpy as np
import polars as pl
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Metadata:
    """Lightweight metadata manager for sample-level auxiliary data."""

    def __init__(self):
        """Initialize empty metadata block."""
        self.df: Optional[pl.DataFrame] = None
        self._numeric_cache: Dict[str, tuple[np.ndarray, Dict]] = {}
        self._row_counter: int = 0  # Track next row ID

    def add_metadata(self,
                     data: Union[np.ndarray, pl.DataFrame, pd.DataFrame],
                     headers: Optional[List[str]] = None) -> None:
        """
        Add metadata rows.

        Args:
            data: 2D array (n_samples, n_cols) or DataFrame
            headers: Column names (required if data is ndarray)
        """
        if data is None:
            return

        # Convert input to DataFrame
        if isinstance(data, np.ndarray):
            if data.size == 0:
                return
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            if headers is None:
                headers = [f"meta_{i}" for i in range(data.shape[1])]
            new_df = pl.DataFrame({col: data[:, i] for i, col in enumerate(headers)})
        elif isinstance(data, pl.DataFrame):
            new_df = data.clone()
        elif isinstance(data, pd.DataFrame):
            new_df = pl.from_pandas(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        if len(new_df) == 0:
            return

        # Add row_id column
        n_rows = len(new_df)
        row_ids = list(range(self._row_counter, self._row_counter + n_rows))
        new_df = new_df.insert_column(0, pl.Series("row_id", row_ids, dtype=pl.Int32))

        # Append or initialize
        if self.df is None:
            self.df = new_df
        else:
            # Use diagonal strategy to handle different columns
            self.df = pl.concat([self.df, new_df], how="diagonal_relaxed")

        self._row_counter += n_rows

        # Clear numeric cache when data changes
        self._numeric_cache.clear()

    def get(self,
            indices: Optional[Union[List[int], np.ndarray]] = None,
            columns: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Get metadata as DataFrame.

        Args:
            indices: Row indices to select (None = all)
            columns: Columns to return (None = all except row_id)

        Returns:
            Polars DataFrame (without row_id column)
        """
        if self.df is None:
            return pl.DataFrame()

        result = self.df

        # Filter by row indices
        if indices is not None and len(indices) > 0:
            result = result.filter(pl.col("row_id").is_in(indices))

        # Select columns
        if columns is not None:
            missing = [c for c in columns if c not in result.columns]
            if missing:
                raise ValueError(f"Columns not found: {missing}")
            result = result.select(["row_id"] + columns)

        # Remove row_id from output
        return result.select([c for c in result.columns if c != "row_id"])

    def get_column(self,
                   column: str,
                   indices: Optional[Union[List[int], np.ndarray]] = None) -> np.ndarray:
        """
        Get single column as numpy array.

        Args:
            column: Column name
            indices: Row indices to select (None = all)

        Returns:
            Numpy array of column values
        """
        if self.df is None:
            raise ValueError("No metadata available")

        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found. Available: {self.columns}")

        result = self.df
        if indices is not None and len(indices) > 0:
            result = result.filter(pl.col("row_id").is_in(indices))

        return result[column].to_numpy()

    def to_numeric(self,
                   column: str,
                   indices: Optional[Union[List[int], np.ndarray]] = None,
                   method: Literal["label", "onehot"] = "label") -> tuple[np.ndarray, Dict]:
        """
        Convert categorical column to numeric encoding.

        Args:
            column: Column name
            indices: Row indices (None = all)
            method: "label" for label encoding, "onehot" for one-hot

        Returns:
            (numeric_array, encoding_info) tuple where encoding_info contains
            method details and class mappings
        """
        if self.df is None:
            raise ValueError("No metadata available")

        cache_key = f"{column}_{method}"

        # Get column data
        col_data = self.get_column(column, indices=None)  # Get full column for encoding

        # Check if already cached
        if cache_key in self._numeric_cache:
            full_numeric, encoding_info = self._numeric_cache[cache_key]
            # Filter to requested indices
            if indices is not None and len(indices) > 0:
                # Map indices to positions in full data
                all_row_ids = self.df["row_id"].to_numpy()
                positions = [np.where(all_row_ids == idx)[0][0] for idx in indices]
                return full_numeric[positions], encoding_info
            return full_numeric.copy(), encoding_info

        # Create encoding
        if method == "label":
            # Check if already numeric
            if np.issubdtype(col_data.dtype, np.number):
                numeric = col_data.astype(np.float32)
                encoding_info = {"method": "numeric", "dtype": str(col_data.dtype)}
            else:
                # Use LabelEncoder
                encoder = LabelEncoder()
                numeric = encoder.fit_transform(col_data).astype(np.float32)
                encoding_info = {
                    "method": "label",
                    "classes": encoder.classes_.tolist()
                }

        elif method == "onehot":
            # Get unique values
            unique_vals = np.unique(col_data)
            n_classes = len(unique_vals)
            n_samples = len(col_data)

            # Create one-hot matrix
            numeric = np.zeros((n_samples, n_classes), dtype=np.float32)
            val_to_idx = {val: i for i, val in enumerate(unique_vals)}
            for i, val in enumerate(col_data):
                numeric[i, val_to_idx[val]] = 1.0

            encoding_info = {
                "method": "onehot",
                "classes": unique_vals.tolist()
            }

        else:
            raise ValueError(f"Unknown method: {method}. Use 'label' or 'onehot'")

        # Cache for consistency
        self._numeric_cache[cache_key] = (numeric.copy(), encoding_info)

        # Filter to requested indices
        if indices is not None and len(indices) > 0:
            all_row_ids = self.df["row_id"].to_numpy()
            positions = [np.where(all_row_ids == idx)[0][0] for idx in indices]
            return numeric[positions], encoding_info

        return numeric, encoding_info

    def update_metadata(self,
                        indices: Union[List[int], np.ndarray],
                        column: str,
                        values: Union[List, np.ndarray]) -> None:
        """
        Update metadata values for specific rows.

        Args:
            indices: Row indices to update
            column: Column name
            values: New values (must match length of indices)
        """
        if self.df is None:
            raise ValueError("No metadata available")

        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")

        if len(indices) != len(values):
            raise ValueError(f"Length mismatch: {len(indices)} indices vs {len(values)} values")

        # Clear cache since data is changing
        self._numeric_cache.clear()

        # Update using Polars - more efficient approach
        # Create a mapping dict
        update_dict = dict(zip(indices, values))

        # Apply updates
        self.df = self.df.with_columns(
            pl.col("row_id").replace(update_dict, default=pl.col("row_id")).alias("_temp_update_key")
        )

        # Use the mapping to update values
        for idx, val in update_dict.items():
            self.df = self.df.with_columns(
                pl.when(pl.col("row_id") == idx)
                .then(pl.lit(val))
                .otherwise(pl.col(column))
                .alias(column)
            )

        # Remove temp column if it exists
        if "_temp_update_key" in self.df.columns:
            self.df = self.df.drop("_temp_update_key")

    def add_column(self,
                   column: str,
                   values: Union[List, np.ndarray]) -> None:
        """
        Add new metadata column.

        Args:
            column: Column name
            values: Column values (must match number of rows)
        """
        if self.df is None:
            raise ValueError("No metadata available. Add metadata first.")

        if len(values) != len(self.df):
            raise ValueError(f"Values length {len(values)} != metadata rows {len(self.df)}")

        if column in self.df.columns:
            raise ValueError(f"Column '{column}' already exists")

        # Add column
        self.df = self.df.with_columns(pl.Series(column, values))

        # Clear cache since structure changed
        self._numeric_cache.clear()

    @property
    def num_rows(self) -> int:
        """Number of metadata rows."""
        return 0 if self.df is None else len(self.df)

    @property
    def columns(self) -> List[str]:
        """List of metadata column names (excluding row_id)."""
        if self.df is None:
            return []
        return [c for c in self.df.columns if c != "row_id"]

    def __repr__(self) -> str:
        if self.df is None:
            return "Metadata(empty)"
        return f"Metadata(rows={self.num_rows}, columns={self.columns})"

    def __str__(self) -> str:
        return self.__repr__()
