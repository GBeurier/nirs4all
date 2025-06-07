"""
Feature management for SpectroDataset.

This module contains FeatureBlock and FeatureSource classes for managing
multi-source spectroscopy features with zero-copy semantics.
"""

import numpy as np
import polars as pl
from typing import List, Optional, Dict, Any, Tuple, Union


class FeatureSource:
    """Wrapper for a single (rows, dims) float array."""

    def __init__(self, array: np.ndarray):
        """
        Initialize a feature source with a 2D numpy array.

        Args:
            array: 2D numpy array of shape (n_rows, n_dims)
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a numpy ndarray")
        if array.ndim != 2:
            raise ValueError("array must be 2-dimensional")
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError("array must have a floating point dtype")

        self._array = array
        self._n_rows, self._n_dims = array.shape

    @property
    def array(self) -> np.ndarray:
        """Get the underlying numpy array (read-only view)."""
        return self._array

    @property
    def n_rows(self) -> int:
        """Get the number of rows."""
        return self._n_rows

    @property
    def n_dims(self) -> int:
        """Get the number of dimensions (features)."""
        return self._n_dims

    def __repr__(self):
        return f"FeatureSource(shape={self._array.shape}, dtype={self._array.dtype})"


class FeatureBlock:
    """Manages N aligned NumPy sources + a Polars index."""

    def __init__(self):
        """Initialize empty feature block."""
        self.sources: List[FeatureSource] = []
        self.index_df: Optional[pl.DataFrame] = None

    def add_features(self, x_list: List[np.ndarray]) -> None:
        """
        Add feature arrays as new sources.

        Args:
            x_list: List of 2D numpy arrays, all with the same number of rows
        """
        if not x_list:
            raise ValueError("x_list cannot be empty")

        # Validate all arrays have the same number of rows
        n_rows = x_list[0].shape[0]
        for i, arr in enumerate(x_list):
            if arr.shape[0] != n_rows:
                raise ValueError(f"Array {i} has {arr.shape[0]} rows, expected {n_rows}")

        # If we already have sources, ensure row count matches
        if self.sources and self.sources[0].n_rows != n_rows:
            raise ValueError(f"New arrays have {n_rows} rows, existing sources have {self.sources[0].n_rows}")

        # Add sources
        for arr in x_list:
            self.sources.append(FeatureSource(arr))

        # Build index if this is the first addition
        if self.index_df is None:
            self._build_initial_index(n_rows)

    def _build_initial_index(self, n_rows: int) -> None:
        """Build the initial index DataFrame."""
        self.index_df = pl.DataFrame({
            "row": list(range(n_rows)),
            "sample": list(range(n_rows)),  # sample == row for initial data
            "origin": list(range(n_rows)),  # origin == sample for pristine rows
            "partition": ["train"] * n_rows,
            "group": [0] * n_rows,
            "branch": [0] * n_rows,
            "processing": ["raw"] * n_rows
        })

    def n_samples(self) -> int:
        """Get the number of samples (rows) across all sources."""
        if not self.sources:
            return 0
        return self.sources[0].n_rows

    def __repr__(self):
        n_sources = len(self.sources)
        n_samples = self.n_samples()
        return f"FeatureBlock(sources={n_sources}, samples={n_samples})"

    def slice(self, row_indices: Union[List[int], slice]) -> "FeatureBlock":
        """
        Slice the feature block by row indices.

        Args:
            row_indices: List of row indices or a slice object

        Returns:
            A new FeatureBlock instance containing the sliced data
        """
        sliced_block = FeatureBlock()
        for source in self.sources:
            sliced_array = source.array[row_indices]
            sliced_block.add_features([sliced_array])
        if self.index_df is not None:
            sliced_block.index_df = self.index_df[row_indices]
        return sliced_block

    def to_numpy(self) -> np.ndarray:
        """Convert the feature block to a 3D numpy array (n_sources, n_rows, n_dims)."""
        return np.array([source.array for source in self.sources])

    def from_numpy(self, array: np.ndarray) -> None:
        """
        Populate the feature block from a 3D numpy array.

        Args:
            array: 3D numpy array of shape (n_sources, n_rows, n_dims)
        """
        if array.ndim != 3:
            raise ValueError("array must be a 3-dimensional numpy array")
        n_sources, n_rows, n_dims = array.shape
        for i in range(n_sources):
            self.add_features([array[i]])

    def _apply_filter(self, filter_dict: Dict[str, Any]) -> pl.DataFrame:
        """
        Apply filter to index DataFrame and return filtered view.

        Args:
            filter_dict: Dictionary of column: value pairs for exact matching

        Returns:
            Filtered DataFrame view
        """
        if self.index_df is None:
            raise ValueError("No index available - add features first")

        filtered_df = self.index_df
        for column, value in filter_dict.items():
            if column not in self.index_df.columns:
                raise ValueError(f"Column '{column}' not found in index")

            # Handle list values with 'is_in' for multiple matches
            if isinstance(value, (list, tuple, np.ndarray)):
                filtered_df = filtered_df.filter(pl.col(column).is_in(value))
            else:
                filtered_df = filtered_df.filter(pl.col(column) == value)
        return filtered_df

    def _get_row_indices(self, filtered_df: pl.DataFrame) -> np.ndarray:
        """Get row indices from filtered DataFrame."""
        return filtered_df.select("row").to_numpy().flatten()

    def _layout_2d(self, row_indices: np.ndarray, src_concat: bool = False) -> Tuple[np.ndarray, ...]:
        """
        Return 2D layout - concatenate processing variants along axis=1.

        Args:
            row_indices: Row indices to extract
            src_concat: Whether to concatenate sources along axis=1

        Returns:
            Tuple of 2D arrays (one per source, or single array if src_concat=True)
        """
        if len(self.sources) == 0:
            return ()

        arrays = []
        for source in self.sources:
            # Check if indices are contiguous for zero-copy slice
            if len(row_indices) > 1 and np.all(np.diff(row_indices) == 1):
                # Contiguous slice - true zero-copy
                start_idx = row_indices[0]
                end_idx = row_indices[-1] + 1
                array_view = source.array[start_idx:end_idx, :]
            else:
                # Non-contiguous indices - fancy indexing (creates copy)
                array_view = source.array[row_indices, :]
            arrays.append(array_view)

        if src_concat and len(arrays) > 1:
            # Concatenate all sources along axis=1
            return (np.concatenate(arrays, axis=1),)
        else:
            return tuple(arrays)

    def _layout_2d_interlaced(self, row_indices: np.ndarray, src_concat: bool = False) -> Tuple[np.ndarray, ...]:
        """
        Return 2D interlaced layout.
        For now, same as 2d since we don't have multiple processing variants yet.
        """
        return self._layout_2d(row_indices, src_concat)

    def _layout_3d(self, row_indices: np.ndarray, src_concat: bool = False) -> Tuple[np.ndarray, ...]:
        """
        Return 3D layout - stack variants along new axis.
        For now, same as 2d with extra dimension since we don't have multiple processing variants yet.
        """
        arrays_2d = self._layout_2d(row_indices, src_concat)
        arrays_3d = []

        for arr in arrays_2d:
            # Add dimension for variants (currently only 1 variant)
            arr_3d = arr[:, np.newaxis, :]
            arrays_3d.append(arr_3d)

        return tuple(arrays_3d)

    def _layout_3d_transpose(self, row_indices: np.ndarray, src_concat: bool = False) -> Tuple[np.ndarray, ...]:
        """
        Return 3D layout with transposed last two axes.
        """
        arrays_3d = self._layout_3d(row_indices, src_concat)
        arrays_transposed = []

        for arr in arrays_3d:
            # Swap last two axes
            arr_transposed = np.swapaxes(arr, -2, -1)
            arrays_transposed.append(arr_transposed)

        return tuple(arrays_transposed)

    def x(self, filter_dict: Dict[str, Any], layout: str = "2d", src_concat: bool = False) -> Tuple[np.ndarray, ...]:
        """
        Get feature arrays with specified layout and filtering.

        Args:
            filter_dict: Dictionary of column: value pairs for filtering
            layout: Layout type ("2d", "2d_interlaced", "3d", "3d_transpose")
            src_concat: Whether to concatenate sources along axis=1

        Returns:
            Tuple of numpy arrays (zero-copy views)
        """
        if not self.sources:
            raise ValueError("No features available")

        # Apply filter to get relevant rows
        filtered_df = self._apply_filter(filter_dict)
        row_indices = self._get_row_indices(filtered_df)

        # Apply layout
        if layout == "2d":
            return self._layout_2d(row_indices, src_concat)
        elif layout == "2d_interlaced":
            return self._layout_2d_interlaced(row_indices, src_concat)
        elif layout == "3d":
            return self._layout_3d(row_indices, src_concat)
        elif layout == "3d_transpose":
            return self._layout_3d_transpose(row_indices, src_concat)
        else:
            raise ValueError(f"Unknown layout: {layout}")

    def get_indexed_features(self, filter_dict: Dict[str, Any], layout: str = "2d",
                           src_concat: bool = False) -> Tuple[Tuple[np.ndarray, ...], pl.DataFrame]:
        """
        Get feature arrays and corresponding index DataFrame.

        Args:
            filter_dict: Dictionary of column: value pairs for filtering
            layout: Layout type ("2d", "2d_interlaced", "3d", "3d_transpose")
            src_concat: Whether to concatenate sources along axis=1

        Returns:
            Tuple of (feature arrays, filtered index DataFrame)
        """
        if not self.sources:
            raise ValueError("No features available")

        # Apply filter
        filtered_df = self._apply_filter(filter_dict)
        row_indices = self._get_row_indices(filtered_df)

        # Get feature arrays
        if layout == "2d":
            arrays = self._layout_2d(row_indices, src_concat)
        elif layout == "2d_interlaced":
            arrays = self._layout_2d_interlaced(row_indices, src_concat)
        elif layout == "3d":
            arrays = self._layout_3d(row_indices, src_concat)
        elif layout == "3d_transpose":
            arrays = self._layout_3d_transpose(row_indices, src_concat)
        else:
            raise ValueError(f"Unknown layout: {layout}")

        return arrays, filtered_df

    def dump_numpy(self, base_path: str) -> None:
        """
        Dump feature sources to numpy files.

        Args:
            base_path: Base path for saving files (features_src0.npy, features_src1.npy, etc.)
        """
        if not self.sources:
            return

        import os
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        for i, source in enumerate(self.sources):
            filepath = f"{base_path}_src{i}.npy"
            np.save(filepath, source.array, allow_pickle=False)

    def load_numpy(self, base_path: str, mmap_mode: str = "r") -> None:
        """
        Load feature sources from numpy files with zero-copy memory mapping.

        Args:
            base_path: Base path for loading files (features_src0.npy, features_src1.npy, etc.)
            mmap_mode: Memory map mode for np.load (default "r" for read-only)
        """
        import os
        import glob

        # Find all source files matching pattern
        pattern = f"{base_path}_src*.npy"
        files = sorted(glob.glob(pattern))

        if not files:
            return

        # Clear existing sources
        self.sources = []
        self.index_df = None

        # Load each source with memory mapping
        for filepath in files:
            data = np.load(filepath, mmap_mode=mmap_mode)
            source = FeatureSource(data)
            self.sources.append(source)

        # Rebuild index
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the index DataFrame from loaded sources."""
        if not self.sources:
            return        # Create index entries for all sources
        index_data = []
        for src_idx, source in enumerate(self.sources):
            n_samples = source.n_rows
            for row in range(n_samples):
                index_data.append({
                    "sample": len(index_data),  # Global sample index
                    "src": src_idx,
                    "row": row
                })

        self.index_df = pl.DataFrame(index_data)
