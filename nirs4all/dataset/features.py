from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import polars as pl

from nirs4all.dataset.feature_source import FeatureSource
from nirs4all.dataset.indexer import SampleIndexManager


class Features:
    """Manages N aligned NumPy sources + a Polars index."""

    def __init__(self):
        """Initialize empty feature block."""
        self.sources: List[FeatureSource] = []
        self.init_index()

    def init_index(self) -> None:
        """Initialize the indexer with an empty DataFrame."""
        index_df = pl.DataFrame({
            "sample": pl.Series([], dtype=pl.Int32),
            "origin": pl.Series([], dtype=pl.Int32),
            "partition": pl.Series([], dtype=pl.Categorical),
            "group": pl.Series([], dtype=pl.Int8),
            "branch": pl.Series([], dtype=pl.Int8),
            "processing": pl.Series([], dtype=pl.Categorical),
        })

        INDEX_DEFAULT_VALUES = {
            "partition": "train",
            "group": 0,
            "branch": 0,
            "processing": "raw",
        }

        self.index = SampleIndexManager(index_df, default_values=INDEX_DEFAULT_VALUES)

    def add_features(self, filter_dict: Dict[str, Any], x_list: Union[np.ndarray, List[np.ndarray]], source: Union[int, List[int]] = -1) -> None:
        """
        Add new features to the block.
        Args:
            filter_dict: Dictionary of column: value pairs for filtering
            x_list: List of numpy arrays or a single 2D numpy array to add
        """
        if isinstance(x_list, np.ndarray):
            if isinstance(source, List):
                raise ValueError("Cannot specify multiple sources with a single numpy array")
        elif isinstance(x_list, list):
            if isinstance(source, int) and source > -1:
                raise ValueError("Cannot specify a single source with a list of numpy arrays")

        n_added_sources = 1 if isinstance(x_list, np.ndarray) else len(x_list)
        n_added_rows = x_list[0].shape[0] if isinstance(x_list, list) else x_list.shape[0]
        if len(self.sources) == 0:
            for _ in range(n_added_sources):
                self.sources.append(FeatureSource())

        kwargs = {}
        if "processing" in filter_dict:
            kwargs["processing"] = filter_dict["processing"]
            filter_dict["sample"] = self.index.get_indices({})

        for i in range(n_added_sources):
            src_index = i
            if isinstance(source, List):
                src_index = source[i]
            elif isinstance(source, int) and source > -1:
                src_index = source
            src = self.sources[src_index]

            new_x = x_list[i] if isinstance(x_list, list) else x_list
            src.add_samples(new_x, **kwargs)

        self.index.add_rows(n_added_rows, overrides=filter_dict)

    def validate_added_features(self, filter_dict: Dict[str, Any], x_list: np.ndarray | List[np.ndarray]) -> None:
        if not isinstance(filter_dict, dict):
            raise TypeError(
                f"filter_dict must be a dictionary, got {type(filter_dict)}"
            )

        if not x_list:
            raise ValueError("x_list cannot be empty")
        if not isinstance(x_list, (list, np.ndarray)):
            raise TypeError(
                f"x_list must be a list or a numpy array, got {type(x_list)}"
            )
        if isinstance(x_list, np.ndarray):
            if x_list.ndim != 2:
                raise ValueError(f"x_list must be a 2-D numpy array, got {x_list.ndim}D")
            x_list = [x_list]
        elif not isinstance(x_list, list):
            raise TypeError(
                f"x_list must be a list of numpy arrays, got {type(x_list)}"
            )

        if not all(isinstance(arr, np.ndarray) for arr in x_list):
            raise TypeError("All elements in x_list must be numpy arrays")

        if not all(arr.ndim == 2 for arr in x_list):
            raise ValueError("All arrays in x_list must be 2-D numpy arrays")

        n_rows = x_list[0].shape[0]
        for i, arr in enumerate(x_list):
            if arr.ndim != 2:
                raise ValueError(f"Array {i} must be 2-D, got shape {arr.shape}")
            if arr.shape[0] != n_rows:
                raise ValueError(
                    f"Array {i} has {arr.shape[0]} rows, expected {n_rows}"
                )

    def update_index(self, indices: List[int], filter_dict: Dict[str, Any]) -> None:
        """
        Update the index with new rows based on the provided filter_dict.
        Args:
            indices: List of indices to update
            filter_dict: Dictionary of column: value pairs for filtering
        """
        self.index.update_by_indices(indices, filter_dict)


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

    def __str__(self):
        n_sources = len(self.sources)
        n_samples = self.num_samples
        summary = f"FeatureBlock with {n_sources} sources and {n_samples} samples"
        for i, source in enumerate(self.sources):
            summary += f"\nSource {i}: {source}"
        if n_sources == 0:
            summary += "\nNo sources available"

        return summary


    #####################################################################################

    def x(self, filter_dict: Dict[str, Any], layout: str = "2d", source: Union[int, List[int]] = -1, src_concat: bool = False) -> np.ndarray | Tuple[np.ndarray, ...]:
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

        indices = self.index.get_contiguous_ranges(filter_dict)
        if isinstance(source, int):
            source = [source] if source != -1 else list(range(len(self.sources)))

        res = []
        for i, source_avai in enumerate(self.sources):
            if i not in source:
                continue
            if layout == "2d":
                res.append(source_avai.layout_2d(indices))
            elif layout == "2d_interleaved":
                res.append(source_avai.layout_2d_interleaved(indices))
            elif layout == "3d":
                res.append(source_avai.layout_3d(indices))
            elif layout == "3d_transpose":
                res.append(source_avai.layout_3d_transpose(indices))
            else:
                raise ValueError(f"Unknown layout: {layout}")

        if src_concat and len(res) > 1:
            return np.concatenate(res, axis=res[0].ndim - 1)

        if len(res) == 1:
            return res[0]

        return tuple(res)

    def set_x(self,
              filter_dict: Dict[str, Any],
              x: Union[np.ndarray, List[np.ndarray]],
              layout: str = "2d",
              filter_update: Optional[Dict[str, Any]] = None,
              src_concat: bool = False,
              source: Union[int, List[int]] = -1) -> None:
        """
        Set feature data for the dataset.

        Args:
            filter_dict: Dictionary to filter which samples to update
            x: Feature data as a numpy array or list of numpy arrays
            layout: Layout type ("2d", "2d_interleaved", "3d", "3d_transpose")
            src_concat: Whether to concatenate sources along axis=1
            source: Index of the source to update, -1 for all sources
        """
        if src_concat and isinstance(x, list):
            raise ValueError("Cannot split sources when x is a list")

        indices = self.index.get_indices(filter_dict)

        if src_concat:
            last_index = 0
            for i, source_avai in enumerate(self.sources):
                if isinstance(source, int) and source != -1 and i != source:
                    continue
                if layout == "2d" or layout == "2d_interleaved":
                    n_features = source_avai.num_2d_features
                elif layout == "3d" or layout == "3d_transpose":
                    n_features = source_avai.num_features
                else:
                    raise ValueError(f"Unknown layout: {layout}")

                # split vertically x
                source_x = x[last_index:last_index + n_features]
                source_avai.update_features(indices, source_x, layout=layout)
                last_index += n_features
        else:
            for i, source_x in enumerate(x):
                self.sources[i].update_features(indices, source_x, layout=layout)

        # if filter_update is None:
            # filter_update = {}
        # self.index.update_by_filter(filter_dict, filter_update)




    # def update_processing(self, processing_id: str | int, new_data: np.ndarray, new_processing: str) -> None:
    #     pass

    # def augment_samples(self, count: Union[int, List[int]], indices: List[int] | None = None) -> None:
    #     pass

    # def add_samples(self, new_samples: np.ndarray) -> None:
    #     pass

    # def add_processings(self, processing_id: Union[str, List[str]]) -> None:
    #     pass

