"""
Main SpectroDataset orchestrator class.

This module contains the main facade that coordinates all dataset blocks
and provides the primary public API for users.
"""



import numpy as np

from nirs4all.dataset.types import Selector, SourceSelector, OutputData, InputData, Layout
from nirs4all.dataset.features import Features
from nirs4all.dataset.targets import Targets
from nirs4all.dataset.indexer import Indexer
from nirs4all.dataset.metadata import Metadata
from nirs4all.dataset.predictions import Predictions
from sklearn.base import TransformerMixin
from typing import Optional


class SpectroDataset:
    """
    Main dataset orchestrator that manages feature, target, metadata,
    fold, and prediction blocks.
    """
    def __init__(self):
        self._indexer = Indexer()
        self._features = Features()
        self._targets = Targets()
        self._folds = []
        self._metadata = Metadata()
        self._predictions = Predictions()

    # FEATURES
    def get_features(self,
                     selector: Selector = {},
                     layout: Layout = "2d",
                     source: SourceSelector = -1,
                     concat_sources: bool = True) -> OutputData:

        indices, processings = self._indexer.get_samples_and_processings(selector)
        return self._features.data(selector, layout, source, concat_sources)


    def add_features(self,
                     data: InputData,
                     data_index: Selector = {}) -> None:

        self._features.add_features(data_index, data)

    def set_features(self, filter: Dict[str, Any], data: np.ndarray | List[np.ndarray], layout: str = "2d", filter_update: Optional[Dict[str, Any]] = None, concat_sources: bool = False, source: Union[int, List[int]] = -1) -> None:

        # x_indices, x_processings = self.indexer.get_indices(filter)
        self._features.set_x(filter, x, layout=layout, filter_update=filter_update, concat_sources=concat_sources, source=source)

    def augment_samples(self, filter: Dict[str, Any], count: Union[int, List[int]], augmentation_id: Optional[str] = None) -> List[int]:
        return self.features.augment_samples(filter, count, augmentation_id=augmentation_id)

    def targets(self, filter: Dict[str, Any] = {}, encoding: str = "auto") -> np.ndarray:
        indices = self._indexer.samples(filter)
        return self._targets.y(indices=indices, encoding=encoding)

    def add_targets(self, y: np.ndarray) -> None:
        self._targets.add_targets(y)

    def set_targets(self, filter: Dict[str, Any], y: np.ndarray, transformer: TransformerMixin, new_processing: str) -> None:
        self._targets.set_y(filter, y, transformer, new_processing)

    def metadata(self, filter: Dict[str, Any] = {}) -> pl.DataFrame:
        return self._metadata.meta(filter)

    def add_metadata(self, meta_df: pl.DataFrame) -> None:
        self._metadata.add_meta(meta_df)

    def predictions(self, filter: Dict[str, Any] = {}) -> pl.DataFrame:
        return self._predictions.prediction(filter)

    def add_predictions(self, np_arr: np.ndarray, meta_dict: Dict[str, Any]) -> None:
        self._predictions.add_prediction(np_arr, meta_dict)

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

    def is_multi_source(self) -> bool:
        return len(self._features.sources) > 1

    @property
    def n_sources(self) -> int:
        return len(self._features.sources)


    def __repr__(self):
        txt = str(self._features)
        txt += "\n" + str(self._targets)
        return txt

    def __str__(self):
        txt = str(self._features)
        txt += "\n" + str(self._targets)
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
            total_samples = self._features.n_samples
            n_sources = len(self._features.sources)
            feature_dims = [src.n_dims for src in self._features.sources]
            print(f"📊 Features: {total_samples} samples, {n_sources} source(s)")
            for i, dims in enumerate(feature_dims):
                rows = self._features.sources[i].n_rows
                print(f"   Source {i}: {rows} rows x {dims} dims")
            print(self._features)
            print(self._targets)
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

