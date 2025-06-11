from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import polars as pl

from nirs4all.dataset.feature_source import FeatureSource


class Features:
    """Manages N aligned NumPy sources + a Polars index."""

    def __init__(self):
        """Initialize empty feature block."""
        self.sources: List[FeatureSource] = []
        self.index: Optional[pl.DataFrame] = None

    def add_features(self, filter_dict: Dict[str, Any], x_list: List[np.ndarray]) -> None:
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


    #####################################################################################

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

        row_ranges = self.indexer.get_contiguous_ranges(filter_dict)
        res = []
        for source in self.sources:
            if layout == "2d":
                res.append(source.layout_2d(row_ranges))
            elif layout == "2d_interleaved":
                res.append(source.layout_2d_interleaved(row_ranges))
            elif layout == "3d":
                res.append(source.layout_3d(row_ranges))
            elif layout == "3d_transpose":
                res.append(source.layout_3d_transpose(row_ranges))
            else:
                raise ValueError(f"Unknown layout: {layout}")

        if src_concat:
            return np.concatenate(res, axis=1)

        return tuple(res)


    # def update_processing(self, processing_id: str | int, new_data: np.ndarray, new_processing: str) -> None:
    #     pass

    # def augment_samples(self, count: Union[int, List[int]], indices: List[int] | None = None) -> None:
    #     pass

    # def add_samples(self, new_samples: np.ndarray) -> None:
    #     pass

    # def add_processings(self, processing_id: Union[str, List[str]]) -> None:
    #     pass
