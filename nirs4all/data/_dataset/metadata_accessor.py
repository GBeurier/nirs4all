"""
Metadata accessor for managing sample-level auxiliary data.

This module provides a dedicated interface for all metadata-related
operations, including retrieval, updates, and column management.
"""

from typing import Literal, Optional, Union

import numpy as np
import polars as pl

from nirs4all.data.indexer import Indexer
from nirs4all.data.metadata import Metadata
from nirs4all.data.types import Selector


class MetadataAccessor:
    """
    Accessor for metadata operations.

    Manages sample-level auxiliary information like sample IDs,
    batch numbers, quality scores, etc.

    Attributes:
        columns (List[str]): List of metadata column names
        num_rows (int): Number of metadata rows

    Examples:
        >>> # Used internally by SpectroDataset, accessible as:
        >>> # dataset.metadata.columns
        >>> # dataset.metadata.num_rows
    """

    def __init__(self, indexer: Indexer, metadata_block: Metadata):
        """
        Initialize metadata accessor.

        Args:
            indexer: Sample index manager for filtering
            metadata_block: Underlying metadata storage
        """
        self._indexer = indexer
        self._block = metadata_block

    def get(self,
            selector: Selector | None = None,
            columns: list[str] | None = None,
            include_augmented: bool = True) -> pl.DataFrame:
        """
        Get metadata as Polars DataFrame.

        Args:
            selector: Filter selector (e.g., {"partition": "train"}):
                - partition: "train", "test", "val"
                - group: group identifier
                - branch: branch identifier
                - fold: fold number
            columns: Specific columns to return (None = all)
            include_augmented: If True, include augmented versions of selected samples

        Returns:
            Polars DataFrame with metadata

        Examples:
            >>> # Get all train metadata
            >>> meta_df = dataset.metadata({"partition": "train"})
            >>> # Get specific columns for test set
            >>> meta_df = dataset.metadata(
            ...     {"partition": "test"},
            ...     columns=["sample_id", "quality"]
            ... )
        """
        indices = self._indexer.x_indices(selector, include_augmented) if selector else None
        return self._block.get(indices, columns)

    def column(self,
               column: str,
               selector: Selector | None = None,
               include_augmented: bool = True) -> np.ndarray:
        """
        Get single metadata column as array.

        Args:
            column: Column name
            selector: Filter selector (e.g., {"partition": "train"})
            include_augmented: If True, include augmented versions of selected samples

        Returns:
            Numpy array of column values

        Examples:
            >>> # Get batch info for train samples
            >>> batches = dataset.metadata_column("batch", {"partition": "train"})
        """
        indices = self._indexer.x_indices(selector, include_augmented) if selector else None
        return self._block.get_column(column, indices)

    def to_numeric(self,
                   column: str,
                   selector: Selector | None = None,
                   method: Literal["label", "onehot"] = "label",
                   include_augmented: bool = True) -> tuple[np.ndarray, dict]:
        """
        Get numeric encoding of metadata column.

        Args:
            column: Column name
            selector: Filter selector (e.g., {"partition": "train"})
            method: "label" for label encoding or "onehot" for one-hot encoding
            include_augmented: If True, include augmented versions of selected samples

        Returns:
            Tuple of (numeric_array, encoding_info)

        Examples:
            >>> # Get numeric encoding of categorical column
            >>> encoded, mapping = dataset.metadata_numeric(
            ...     "category",
            ...     {"partition": "train"},
            ...     method="label"
            ... )
        """
        indices = self._indexer.x_indices(selector, include_augmented) if selector else None
        return self._block.to_numeric(column, indices, method)

    def add_metadata(self,
                     data: np.ndarray | pl.DataFrame,
                     headers: list[str] | None = None) -> None:
        """
        Add metadata rows (aligns with add_samples call order).

        Args:
            data: Metadata as 2D array (n_samples, n_cols) or DataFrame
            headers: Column names (required if data is ndarray)

        Examples:
            >>> # Add metadata with numpy array
            >>> metadata = np.array([
            ...     ["batch_1", "high"],
            ...     ["batch_2", "medium"],
            ... ])
            >>> dataset.add_metadata(metadata, headers=["batch", "quality"])
        """
        self._block.add_metadata(data, headers)

    def update_metadata(self,
                        column: str,
                        values: list | np.ndarray,
                        selector: Selector | None = None,
                        include_augmented: bool = True) -> None:
        """
        Update metadata values for selected samples.

        Args:
            column: Column name
            values: New values
            selector: Filter selector (None = all samples)
            include_augmented: If True, include augmented versions of selected samples

        Examples:
            >>> # Update quality scores for train samples
            >>> dataset.update_metadata(
            ...     "quality",
            ...     new_quality_scores,
            ...     {"partition": "train"}
            ... )
        """
        indices = self._indexer.x_indices(selector, include_augmented) if selector else list(range(self._block.num_rows))
        self._block.update_metadata(indices, column, values)

    def add_column(self,
                   column: str,
                   values: list | np.ndarray) -> None:
        """
        Add new metadata column.

        Args:
            column: Column name
            values: Column values (must match number of samples)

        Examples:
            >>> # Add new column with computed values
            >>> new_scores = compute_quality_scores(X_data)
            >>> dataset.add_metadata_column("computed_quality", new_scores)
        """
        self._block.add_column(column, values)

    @property
    def columns(self) -> list[str]:
        """Get list of metadata column names."""
        return self._block.columns

    @property
    def num_rows(self) -> int:
        """Get number of metadata rows."""
        return self._block.num_rows
