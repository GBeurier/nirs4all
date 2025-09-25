"""
Main SpectroDataset orchestrator class.

This module contains the main facade that coordinates all dataset blocks
and provides the primary public API for users.
"""



import numpy as np

from nirs4all.dataset.helpers import Selector, SourceSelector, OutputData, InputData, Layout, IndexDict, get_num_samples, InputFeatures, ProcessingList
from nirs4all.dataset.features import Features
from nirs4all.dataset.targets import Targets
from nirs4all.dataset.indexer import Indexer
from nirs4all.dataset.metadata import Metadata
from nirs4all.dataset.predictions import Predictions
from sklearn.base import TransformerMixin
from typing import Optional, Union, List, Tuple, Dict, Any


class SpectroDataset:
    """
    Main dataset orchestrator that manages feature, target, metadata,
    fold, and prediction blocks.
    """
    def __init__(self, name: str = "Unknown_dataset"):
        self._indexer = Indexer()
        self._features = Features()
        self._targets = Targets()
        self._folds = []
        self._metadata = Metadata()
        self._predictions = Predictions()
        self.name = name

    def x(self, selector: Selector, layout: Layout = "2d", concat_source: bool = True) -> OutputData:
        indices = self._indexer.x_indices(selector)
        return self._features.x(indices, layout, concat_source)

    def y(self, selector: Selector) -> np.ndarray:
        indices = self._indexer.y_indices(selector)
        if selector and "y" in selector:
            processing = selector["y"]
        else:
            processing = "numeric"

        return self._targets.y(indices, processing)

    # FEATURES
    def add_samples(self,
                    data: InputData,
                    indexes: Optional[IndexDict] = None) -> None:
        num_samples = get_num_samples(data)
        self._indexer.add_samples_dict(num_samples, indexes)
        self._features.add_samples(data)

    def add_features(self,
                     features: InputFeatures,
                     processings: ProcessingList) -> None:
        self._features.update_features([], features, processings)
        # Update the indexer to add new processings to existing processing lists
        self._indexer.add_processings(processings)


    def replace_features(self,
                         source_processings: ProcessingList,
                         features: InputFeatures,
                         processings: ProcessingList) -> None:
        self._features.update_features(source_processings, features, processings)
        self._indexer.replace_processings(source_processings, processings)

    def update_features(self,
                        source_processings: ProcessingList,
                        features: InputFeatures,
                        processings: ProcessingList) -> None:
        self._features.update_features(source_processings, features, processings)

    def augment_samples(self,
                        data: InputData,
                        processings: ProcessingList,
                        augmentation_id: str,
                        selector: Optional[Selector] = None,
                        count: Union[int, List[int]] = 1) -> List[int]:
        # Get indices of samples to augment using selector
        if selector is None:
            # Augment all existing samples
            sample_indices = list(range(self._features.num_samples))
        else:
            sample_indices = self._indexer.x_indices(selector).tolist()

        if not sample_indices:
            return []

        # Add augmented samples to indexer first
        augmented_sample_ids = self._indexer.augment_rows(
            sample_indices, count, augmentation_id
        )

        # Add augmented data to features
        self._features.augment_samples(
            sample_indices, data, processings, count
        )

        return augmented_sample_ids

    def features_processings(self, src: int) -> List[str]:
        return self._features.preprocessing_str[src]

    def features_sources(self) -> int:
        return len(self._features.sources)


    def is_multi_source(self) -> bool:
        return len(self._features.sources) > 1



    # def targets(self, filter: Dict[str, Any] = {}, encoding: str = "auto") -> np.ndarray:
    #     indices = self._indexer.samples(filter)
    #     return self._targets.y(indices=indices, encoding=encoding)

    def add_targets(self, y: np.ndarray) -> None:
        self._targets.add_targets(y)

    # def set_targets(self, filter: Dict[str, Any], y: np.ndarray, transformer: TransformerMixin, new_processing: str) -> None:
    #     self._targets.set_y(filter, y, transformer, new_processing)

    # def metadata(self, filter: Dict[str, Any] = {}) -> pl.DataFrame:
    #     return self._metadata.meta(filter)

    # def add_metadata(self, meta_df: pl.DataFrame) -> None:
    #     self._metadata.add_meta(meta_df)

    # def predictions(self, filter: Dict[str, Any] = {}) -> pl.DataFrame:
    #     return self._predictions.prediction(filter)

    # def add_predictions(self, np_arr: np.ndarray, meta_dict: Dict[str, Any]) -> None:
    #     self._predictions.add_prediction(np_arr, meta_dict)

    @property
    def folds(self) -> List[Tuple[List[int], List[int]]]:
        return self._folds

    def set_folds(self, folds_iterable) -> None:
        """Set cross-validation folds from an iterable of (train_idx, val_idx) tuples."""
        self._folds = list(folds_iterable)

    def index_column(self, col: str, filter: Dict[str, Any] = {}) -> List[int]:
        return self._indexer.get_column_values(col, filter)


    @property
    def num_folds(self) -> int:
        """Return the number of folds."""
        return len(self._folds)

    @property
    def n_sources(self) -> int:
        return len(self._features.sources)

    def _fold_str(self) -> str:
        if not self._folds:
            return ""
        folds_count = [(len(train), len(val)) for train, val in self._folds]
        return str(folds_count)

    # def __repr__(self):
    #     txt = str(self._features)
    #     txt += "\n" + str(self._targets)
    #     txt += "\n" + str(self._indexer)
    #     return txt

    def __str__(self):
        txt = f"📊 Dataset: {self.name}"
        txt += "\n" + str(self._features)
        txt += "\n" + str(self._targets)
        txt += "\n" + str(self._indexer)
        if self._folds:
            txt += f"\nFolds: {self._fold_str()}"
        return txt
        # return f"SpectroDataset(features={self.features}, targets={self.targets}, metadata={self.metadata}, folds={self.folds}, predictions={self.predictions})"

    ### PRINTING AND SUMMARY ###
    def print_summary(self) -> None:
        """
        Print a comprehensive summary of the dataset.

        Shows counts, dimensions, number of sources, target versions, predictions, etc.
        """
        print("=== SpectroDataset Summary ===")
        print()        # Features summary
        if self._features.sources:
            total_samples = self._features.num_samples
            n_sources = len(self._features.sources)
            print(f"📊 Features: {total_samples} samples, {n_sources} source(s)")
            print(f"Features: {self._features.num_features}, processings: {self._features.num_processings}")
            print(f"Processing IDs: {self._features.preprocessing_str}")
            # print(self._features)
            # print(self._targets)
        else:
            print("📊 Features: No data")
        print()

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

    # FOLDS

