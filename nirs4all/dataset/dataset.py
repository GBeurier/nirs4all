"""
Main SpectroDataset orchestrator class.

This module contains the main facade that coordinates all dataset blocks
and provides the primary public API for users.
"""

from typing import Union, List, Dict, Any, Tuple, Optional

import numpy as np

from nirs4all.dataset.features import Features
# from nirs4all.dataset.targets import TargetBlock
# from nirs4all.dataset.metadata import MetadataBlock
# from nirs4all.dataset.folds import FoldsManager
# from nirs4all.dataset.predictions import PredictionBlock


class SpectroDataset:
    """
    Main dataset orchestrator that manages feature, target, metadata,
    fold, and prediction blocks.
    """
    def __init__(self):
        """Initialize an empty SpectroDataset."""
        self.features = Features()
        # self.targets = TargetBlock()
        # self.metadata = MetadataBlock()
        # self.predictions = PredictionBlock()
        # self.folds = FoldsManager()

    # FEATURES
    def x(self, filter_dict: Dict[str, Any] = {}, layout: str = "2d", source: Union[int, List[int]] = -1, src_concat: bool = True) -> np.ndarray | Tuple[np.ndarray, ...]:
        return self.features.x(filter_dict, layout, source, src_concat)

    def set_x(self,
              filter_dict: Dict[str, Any],
              x: np.ndarray | List[np.ndarray],
              layout: str = "2d",
              filter_update: Optional[Dict[str, Any]] = None,
              src_concat: bool = True,
              source: Union[int, List[int]] = -1) -> None:
        """
        Set feature data for the dataset.

        Args:
            filter_dict: Dictionary to filter which samples to update
            x: Feature data as a numpy array or list of numpy arrays
            source: Index of the source to update, -1 for all sources
        """
        self.features.set_x(filter_dict, x, layout=layout, filter_update=filter_update, src_concat=src_concat, source=source)

    def add_features(self, filter_dict: Dict[str, Any], x: np.ndarray | List[np.ndarray]) -> None:
        self.features.add_features(filter_dict, x)

    def is_multi_source(self) -> bool:
        """
        Check if the dataset has multiple feature sources.

        Returns:
            True if there are multiple sources, False otherwise.
        """
        return len(self.features.sources) > 1

    @property
    def n_sources(self) -> int:
        """
        Get the number of feature sources in the dataset.

        Returns:
            Number of feature sources.
        """
        return len(self.features.sources)


    # def sample_augmentation(self, count: Union[int, List[int]], indices: List[int] | None = None, filter_dict: Dict[str, Any] | None = None) -> np.ndarray:
    #     """
    #     Augment the dataset by duplicating samples.

    #     Args:
    #         count: Number of times to duplicate each sample, or a list of counts for each sample
    #         indices: Specific indices to augment, if None all samples are augmented
    #         filter_dict: Optional filters to apply before augmentation

    #     Returns:
    #         Augmented feature array
    #     """
    #     return self.features.sample_augmentation(count, indices, filter_dict)


    # def add_features(self, x_list: List[np.ndarray]) -> None:
    #     self.features.add_features(x_list)

    # def add_targets(self, y_list: List[np.ndarray]) -> None:
    #     self.targets.add_targets(y_list)

    # def add_meta(self, meta_df: pd.DataFrame) -> None:
    #     self.metadata.add_meta(meta_df)

    # def add_folds(self, folds_dict: Dict[str, List[int]]) -> None:
    #     self.folds.add_folds(folds_dict)

    # def add_predictions(self, filter_dict: Dict[str, Any], predictions: np.ndarray) -> None:
    #     self.predictions.add_predictions(filter_dict, predictions)


    # ## GETTERS AND DATA ACCESSORS ##


    # def x_indexed(self, filter_dict: Dict[str, Any], layout: str = "2d", src_concat: bool = False) -> Tuple[Tuple[np.ndarray, ...], Any]:
    #     return self.features.x_indexed(filter_dict, layout, src_concat)

    # def x_update(self, new_values: np.ndarray, indexes: Any, processing_id: str) -> None:
    #     self.features.x_update(new_values, indexes, processing_id)

    # def x_copy(self, indexes: Any, copy_count: Any ) -> np.ndarray:
    #     return self.features.x_copy(indexes, copy_count)

    # def y(self, filter_dict, processed=True, encoding="auto") -> np.ndarray:
    #     return self.targets.y(filter_dict, processed=processed, encoding=encoding) # auto, float, int, ohe, raw |

    # def y_indexed(self, filter_dict, processed=True, encoding="auto") -> Tuple[np.ndarray, Any]:
    #     return self.targets.y_indexed(filter_dict, processed=processed, encoding=encoding)

    # def y_update(self, new_values: np.ndarray, indexes: Any, processing_id: str) -> None:
    #     self.targets.y_update(new_values, indexes, processing_id)

    # def meta(self, filter_dict):
    #     return self.metadata.meta(filter_dict)



    ### io ###
    # def save(self, path: str) -> None:
    #     """
    #     Save the dataset to disk.

    #     Args:
    #         path: Directory path where to save the dataset
    #     """
    #     from . import io
    #     io.save(self, path)

    # def load(self, path: str) -> "SpectroDataset":
    #     """
    #     Load a dataset from disk.

    #     Args:
    #         path: Directory path containing the saved dataset

    #     Returns:
    #         Loaded SpectroDataset instance
    #     """
    #     from . import io
    #     return io.load(path)


    ### PRINTING AND SUMMARY ###
    def print_summary(self) -> None:
        """
        Print a comprehensive summary of the dataset.

        Shows counts, dimensions, number of sources, target versions, predictions, etc.
        """
        print("=== SpectroDataset Summary ===")
        print()        # Features summary
        if self.features.sources:
            total_samples = self.features.n_samples
            n_sources = len(self.features.sources)
            feature_dims = [src.n_dims for src in self.features.sources]
            print(f"📊 Features: {total_samples} samples, {n_sources} source(s)")
            for i, dims in enumerate(feature_dims):
                rows = self.features.sources[i].n_rows
                print(f"   Source {i}: {rows} rows x {dims} dims")
            print(self.features)
        else:
            print("📊 Features: No data")
        print()        # Targets summary
        # if self.targets.sources:
        #     n_targets = len(self.targets.sources)
        #     target_names = self.targets.get_target_names()
        #     processing_versions = []
        #     for name in target_names:
        #         versions = self.targets.get_processing_versions(name)
        #         processing_versions.extend(versions)
        #     processing_versions = list(set(processing_versions))
        #     print(f"🎯 Targets: {n_targets} source(s)")
        #     print(f"   Target names: {target_names}")
        #     print(f"   Processing versions: {processing_versions}")
        # else:
        #     print("🎯 Targets: No data")
        # print()

        # # Metadata summary
        # if self.metadata.table is not None:
        #     n_meta = len(self.metadata.table)
        #     meta_cols = self.metadata.table.columns
        #     print(f"📋 Metadata: {n_meta} entries")
        #     print(f"   Columns: {meta_cols}")
        # else:
        #     print("📋 Metadata: No data")
        # print()

        # # Folds summary
        # if self.folds.folds:
        #     n_folds = len(self.folds.folds)
        #     fold_sizes = [(len(f["train"]), len(f["val"])) for f in self.folds.folds]
        #     print(f"🔄 Folds: {n_folds} fold(s)")
        #     for i, (train_size, val_size) in enumerate(fold_sizes):
        #         print(f"   Fold {i}: {train_size} train, {val_size} val")
        # else:
        #     print("🔄 Folds: No data")
        # print()

        # # Predictions summary
        # if self.predictions.table is not None:
        #     n_preds = len(self.predictions.table)
        #     models = self.predictions.table.select("model").unique().to_series().to_list()
        #     partitions = self.predictions.table.select("partition").unique().to_series().to_list()
        #     print(f"🔮 Predictions: {n_preds} entries")
        #     print(f"   Models: {models}")
        #     print(f"   Partitions: {partitions}")
        # else:
        #     print("🔮 Predictions: No data")
        # print()

        # print("=" * 30)

    def __repr__(self):
        return self.features

    def __str__(self):
        return str(self.features)
        # return f"SpectroDataset(features={self.features}, targets={self.targets}, metadata={self.metadata}, folds={self.folds}, predictions={self.predictions})"
