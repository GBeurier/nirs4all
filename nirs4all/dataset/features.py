from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import polars as pl

from nirs4all.dataset.feature_source import FeatureSource
from nirs4all.dataset.feature_indexer import FeatureIndex


class Features:
    """Manages N aligned NumPy sources + a Polars index."""

    def __init__(self):
        """Initialize empty feature block."""
        self.sources: List[FeatureSource] = []
        self.index = FeatureIndex()

    @property
    def num_samples(self) -> int:
        """Get the number of samples (rows) across all sources."""
        if not self.sources:
            return 0
        return self.sources[0].num_samples

    def add_features(self, overrides: Dict[str, Any], x_list: Union[np.ndarray, List[np.ndarray]], source: Union[int, List[int]] = -1) -> None:
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
        if "processing" in overrides:
            kwargs["processing"] = overrides["processing"]
            overrides["sample"], _ = self.index.get_indices({})
            overrides["origin"], _ = self.index.get_indices({})


        for i in range(n_added_sources):
            src_index = i
            if isinstance(source, List):
                src_index = source[i]
            elif isinstance(source, int) and source > -1:
                src_index = source
            src = self.sources[src_index]

            new_x = x_list[i] if isinstance(x_list, list) else x_list
            src.add_samples(new_x, **kwargs)

        self.index.add_rows(n_added_rows, overrides=overrides)

    def augment_samples(self, filter_dict: Dict[str, Any], count: Union[int, List[int]], augmentation_id: Optional[str] = None) -> List[int]:
        if augmentation_id is None:
            augmentation_id = filter_dict.get("augmentation", "unk_aug")
        if "augmentation" in filter_dict:
            del filter_dict["augmentation"]

        indices, _ = self.index.get_indices(filter_dict)

        for src in self.sources:
            src.augment_samples(indices, count)

        return self.index.augment_rows(indices, count, augmentation_id=augmentation_id)

    def x(self, filter_dict: Dict[str, Any], layout: str = "2d", source: Union[int, List[int]] = -1, src_concat: bool = False) -> np.ndarray | Tuple[np.ndarray, ...]:
        if not self.sources:
            raise ValueError("No features available")

        indices, processings = self.index.get_indices(filter_dict)

        if isinstance(source, int):
            source = [source] if source != -1 else list(range(len(self.sources)))

        res = []
        for i, source_avai in enumerate(self.sources):
            if i not in source:
                continue
            res.append(source_avai.get_features(indices, processings, layout))

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

        indices, processings = self.index.get_indices(filter_dict)

        print("=="*20)
        print(f"Setting features with filter_dict: {filter_dict}, layout: {layout}, source: {source}, src_concat: {src_concat}")
        print(f"Indices: {indices}, Processings: {processings}")
        print("=="*20)

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
                source_avai.update_features(indices, processings, source_x, layout=layout)
                last_index += n_features
        else:
            if isinstance(x, np.ndarray):
                x = [x]
            for i, source_x in enumerate(x):
                self.sources[i].update_features(indices, processings, source_x, layout=layout)

        if filter_update:
            if "processing" in filter_update and "processing" in filter_dict:
                for source_avai in self.sources:
                    source_avai.update_processing_ids(filter_dict["processing"], filter_update["processing"])

                if isinstance(filter_dict["processing"], list):
                    if isinstance(filter_update["processing"], list) and len(filter_dict["processing"]) == len(filter_update["processing"]):
                        for i, proc in enumerate(filter_dict["processing"]):
                            new_filter_dict = filter_dict.copy()
                            new_filter_dict["processing"] = proc
                            new_filter_update = filter_update.copy()
                            new_filter_update["processing"] = filter_update["processing"][i]
                            self.index.update_by_filter(new_filter_dict, new_filter_update)
                    elif isinstance(filter_update["processing"], str):
                        self.index.update_by_filter(filter_dict, filter_update)
                elif isinstance(filter_update["processing"], str):
                    if isinstance(filter_dict["processing"], list):
                        raise ValueError("Cannot update processing with a string when filter_dict has a list of processings")
                    else:
                        self.index.update_by_filter(filter_dict, filter_update)
            else:
                self.index.update_by_filter(filter_dict, filter_update)

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
        # unique augmentations
        summary += f"\nUnique augmentations: {self.index.uniques('augmentation')}"
        summary += f"\nIndex:\n{self.index.df}"
        return summary

    def groups(self, filter_dict: Dict[str, Any] = {}) -> np.ndarray:
        return self.index.get_column_values("group", filter_dict)

    # def update_index(self, indices: List[int], filter_dict: Dict[str, Any]) -> None:
    #     """
    #     Update the index with new rows based on the provided filter_dict.
    #     Args:
    #         indices: List of indices to update
    #         filter_dict: Dictionary of column: value pairs for filtering
    #     """
    #     self.index.update_by_indices(indices, filter_dict)

    # def validate_added_features(self, filter_dict: Dict[str, Any], x_list: np.ndarray | List[np.ndarray]) -> None:
    #     if not isinstance(filter_dict, dict):
    #         raise TypeError(
    #             f"filter_dict must be a dictionary, got {type(filter_dict)}"
    #         )

    #     if not x_list:
    #         raise ValueError("x_list cannot be empty")
    #     if not isinstance(x_list, (list, np.ndarray)):
    #         raise TypeError(
    #             f"x_list must be a list or a numpy array, got {type(x_list)}"
    #         )
    #     if isinstance(x_list, np.ndarray):
    #         if x_list.ndim != 2:
    #             raise ValueError(f"x_list must be a 2-D numpy array, got {x_list.ndim}D")
    #         x_list = [x_list]
    #     elif not isinstance(x_list, list):
    #         raise TypeError(
    #             f"x_list must be a list of numpy arrays, got {type(x_list)}"
    #         )

    #     if not all(isinstance(arr, np.ndarray) for arr in x_list):
    #         raise TypeError("All elements in x_list must be numpy arrays")

    #     if not all(arr.ndim == 2 for arr in x_list):
    #         raise ValueError("All arrays in x_list must be 2-D numpy arrays")

    #     n_rows = x_list[0].shape[0]
    #     for i, arr in enumerate(x_list):
    #         if arr.ndim != 2:
    #             raise ValueError(f"Array {i} must be 2-D, got shape {arr.shape}")
    #         if arr.shape[0] != n_rows:
    #             raise ValueError(
    #                 f"Array {i} has {arr.shape[0]} rows, expected {n_rows}"
    #             )