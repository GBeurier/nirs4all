"""
Metadata management for SpectroDataset.

This module contains MetadataBlock for key-value metadata storage.
"""

from typing import Dict, Any, Optional

import polars as pl


class Metadata:
    """Key-value metadata store."""

    def __init__(self):
        """Initialize empty metadata block."""
        self.table: Optional[pl.DataFrame] = None

    # def add_meta(self, meta_df: pl.DataFrame) -> None:
    #     """
    #     Add metadata to the block.

    #     Args:
    #         meta_df: DataFrame with 'sample' column and arbitrary metadata fields
    #     """        # Validate required columns
    #     if "sample" not in meta_df.columns:
    #         raise ValueError("Missing required column: sample")

    #     if self.table is None:
    #         self.table = meta_df.clone()
    #     else:
    #         # Append new data (allow missing columns)
    #         self.table = pl.concat([self.table, meta_df], how="diagonal") ### TODO Check strategy validity here

    # def meta(self, filter_dict: Dict[str, Any]) -> pl.DataFrame:
    #     """
    #     Get metadata with filtering.

    #     Args:
    #         filter_dict: Dictionary of column: value pairs for filtering

    #     Returns:
    #         Filtered metadata DataFrame
    #     """
    #     if self.table is None:
    #         raise ValueError("No metadata available")        # Apply filter
    #     filtered_df = self.table
    #     for column, value in filter_dict.items():
    #         if column not in self.table.columns:
    #             raise ValueError(f"Column '{column}' not found in metadata")

    #         # Handle list values for filtering
    #         if isinstance(value, (list, tuple)):
    #             filtered_df = filtered_df.filter(pl.col(column).is_in(value))
    #         else:
    #             filtered_df = filtered_df.filter(pl.col(column) == value)

    #     return filtered_df

    # def __repr__(self):
    #     if self.table is None:
    #         return "MetadataBlock(empty)"
    #     return f"MetadataBlock(rows={len(self.table)})"
