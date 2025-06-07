"""
Target management for SpectroDataset.

This module contains TargetBlock for managing target values
with processing version tracking.
"""

import numpy as np
import polars as pl
from typing import Dict, Any, Optional, List, Tuple


class TargetBlock:
    """Polars table for target versions."""

    def __init__(self):
        """Initialize empty target block."""
        self.table: Optional[pl.DataFrame] = None

    def add_targets(self, y_df: pl.DataFrame, meta_df: Optional[pl.DataFrame] = None) -> None:
        """
        Add target data to the block.

        Args:
            y_df: DataFrame with columns 'sample', 'targets' (list), 'processing'
            meta_df: Optional metadata DataFrame (not used in this implementation)
        """
        # Validate required columns
        required_cols = ["sample", "targets", "processing"]
        for col in required_cols:
            if col not in y_df.columns:
                raise ValueError(f"Missing required column: {col}")

        if self.table is None:
            self.table = y_df.clone()
        else:
            # Append new data
            self.table = pl.concat([self.table, y_df], how="vertical")

    def y(self, filter_dict: Dict[str, Any], processed: bool = True) -> np.ndarray:
        """
        Get target arrays with filtering.

        Args:
            filter_dict: Dictionary of column: value pairs for filtering
            processed: Whether to return processed targets (not used yet)

        Returns:
            Stacked numpy array of targets
        """
        if self.table is None:
            raise ValueError("No targets available")

        # Apply filter
        filtered_df = self.table
        for column, value in filter_dict.items():
            if column not in self.table.columns:
                raise ValueError(f"Column '{column}' not found in targets")

            # Handle list values with 'is_in' for multiple matches
            if isinstance(value, (list, tuple, np.ndarray)):
                filtered_df = filtered_df.filter(pl.col(column).is_in(value))
            else:
                filtered_df = filtered_df.filter(pl.col(column) == value)

        # Extract and stack target arrays
        target_lists = filtered_df.select("targets").to_numpy().flatten()
        if len(target_lists) == 0:
            return np.array([])

        # Stack all targets into a single array
        targets = np.vstack(target_lists)
        return targets

    def get_indexed_targets(self, filter_dict: Dict[str, Any]) -> List[Tuple[np.ndarray, pl.DataFrame]]:
        """
        Get targets with their index information.

        Args:
            filter_dict: Dictionary of column: value pairs for filtering

        Returns:
            List of (target_array, index_dataframe) tuples
        """
        if self.table is None:
            raise ValueError("No targets available")

        # Apply filter
        filtered_df = self.table
        for column, value in filter_dict.items():
            if column not in self.table.columns:
                raise ValueError(f"Column '{column}' not found in targets")
            filtered_df = filtered_df.filter(pl.col(column) == value)

        # Extract targets
        target_lists = filtered_df.select("targets").to_numpy().flatten()
        if len(target_lists) == 0:
            return []

        targets = np.vstack(target_lists)
        return [(targets, filtered_df)]

    def update_y(self, new_values: np.ndarray, indexes: pl.DataFrame, processing_id: str) -> None:
        """
        Update target values with new processing version.

        Args:
            new_values: New target values array
            indexes: Index DataFrame indicating which samples to update
            processing_id: Processing identifier for this version
        """
        if self.table is None:
            self.table = pl.DataFrame()        # Convert new_values to list of arrays for Polars list column
        targets_list = [row.tolist() for row in new_values]

        # Create new rows with updated processing
        new_rows = indexes.select("sample").with_columns([
            pl.Series("targets", targets_list),
            pl.lit(processing_id).alias("processing")
        ])

        # Append to table
        self.table = pl.concat([self.table, new_rows], how="vertical")

    def __repr__(self):
        if self.table is None:
            return "TargetBlock(empty)"
        return f"TargetBlock(rows={len(self.table)})"
