"""
Main SpectroDataset orchestrator class.

This module contains the main facade that coordinates all dataset blocks
and provides the primary public API for users.
"""


import re
import numpy as np

from nirs4all.dataset.helpers import Selector, SourceSelector, OutputData, InputData, Layout, IndexDict, get_num_samples, InputFeatures, ProcessingList
from nirs4all.dataset.features import Features
from nirs4all.dataset.targets import Targets
from nirs4all.dataset.indexer import Indexer
from nirs4all.dataset.metadata import Metadata
from nirs4all.dataset.predictions import Predictions
from sklearn.base import TransformerMixin
from typing import Optional, Union, List, Tuple, Dict, Any, Literal


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
        # self._predictions = Predictions()
        self.name = name
        self._task_type: Optional[str] = None  # "regression", "binary_classification", "multiclass_classification"

    def x(self, selector: Selector, layout: Layout = "2d", concat_source: bool = True, include_augmented: bool = True) -> OutputData:
        """
        Get feature data with automatic augmented sample aggregation.

        Args:
            selector: Filter criteria (partition, group, branch, etc.)
            layout: Output layout ("2d" or "3d")
            concat_source: If True, concatenate multiple sources along feature axis
            include_augmented: If True, include augmented versions of selected samples.
                             If False, return only base samples (origin=null).
                             Default True for backward compatibility.

        Returns:
            Feature data array(s)

        Example:
            >>> # Get all train samples (base + augmented)
            >>> X_train = dataset.x({"partition": "train"})
            >>> # Get only base train samples (for splitting)
            >>> X_base = dataset.x({"partition": "train"}, include_augmented=False)
        """
        indices = self._indexer.x_indices(selector, include_augmented=include_augmented)
        return self._features.x(indices, layout, concat_source)

    # def x_train(self, layout: Layout = "2d", concat_source: bool = True) -> OutputData:
    #     selector = {"partition": "train"}
    #     return self.x(selector, layout, concat_source)

    # def x_test(self, layout: Layout = "2d", concat_source: bool = True) -> OutputData:
    #     selector = {"partition": "test"}
    #     return self.x(selector, layout, concat_source)

    def y(self, selector: Selector, include_augmented: bool = True) -> np.ndarray:
        """
        Get target data - automatically maps augmented samples to their origin for y values.

        Args:
            selector: Filter criteria (partition, group, branch, etc.)
            include_augmented: If True, include augmented versions of selected samples.
                             Augmented samples are automatically mapped to their origin's y value.
                             If False, return only base samples.
                             Default True for backward compatibility.

        Returns:
            Target values array

        Example:
            >>> # Get all train targets (base + augmented, with mapping)
            >>> y_train = dataset.y({"partition": "train"})
            >>> # Get only base train targets (for splitting)
            >>> y_base = dataset.y({"partition": "train"}, include_augmented=False)
        """
        if include_augmented:
            # Get all sample indices (base + augmented)
            x_indices = self._indexer.x_indices(selector, include_augmented=True)

            # Map each sample to its y index (augmented → origin)
            y_indices = np.array([
                self._indexer.get_origin_for_sample(int(sample_id))
                for sample_id in x_indices
            ], dtype=np.int32)
        else:
            # Get only base samples using x_indices with include_augmented=False
            y_indices = self._indexer.x_indices(selector, include_augmented=False)

        if selector and "y" in selector:
            processing = selector["y"]
        else:
            processing = "numeric"

        return self._targets.y(y_indices, processing)

    # FEATURES
    def add_samples(self,
                    data: InputData,
                    indexes: Optional[IndexDict] = None,
                    headers: Optional[Union[List[str], List[List[str]]]] = None) -> None:
        num_samples = get_num_samples(data)
        self._indexer.add_samples_dict(num_samples, indexes)
        self._features.add_samples(data, headers=headers)

    def add_features(self,
                     features: InputFeatures,
                     processings: ProcessingList,
                     source: int = -1) -> None:
        # print("Adding features with processings:", processings)
        self._features.update_features([], features, processings, source=source)
        # Update the indexer to add new processings to existing processing lists
        self._indexer.add_processings(processings)

    def replace_features(self,
                         source_processings: ProcessingList,
                         features: InputFeatures,
                         processings: ProcessingList,
                         source: int = -1) -> None:
        self._features.update_features(source_processings, features, processings, source=source)
        if source <= 0:  # Update all sources or single source 0
            self._indexer.replace_processings(source_processings, processings)

    def update_features(self,
                        source_processings: ProcessingList,
                        features: InputFeatures,
                        processings: ProcessingList,
                        source: int = -1) -> None:
        self._features.update_features(source_processings, features, processings, source=source)

    def augment_samples(self,
                        data: InputData,
                        processings: ProcessingList,
                        augmentation_id: str,
                        selector: Optional[Selector] = None,
                        count: Union[int, List[int]] = 1) -> List[int]:
        # Get indices of samples to augment using selector
        # IMPORTANT: Always use include_augmented=False to only augment base samples
        if selector is None:
            # Augment all base samples (exclude already augmented ones)
            sample_indices = self._indexer.x_indices({}, include_augmented=False).tolist()
        else:
            sample_indices = self._indexer.x_indices(selector, include_augmented=False).tolist()

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

    def headers(self, src: int) -> List[str]:
        return self._features.headers(src)

    def float_headers(self, src: int) -> np.ndarray:
        try:
            return np.array([float(header) for header in self._features.headers(src)])
        except ValueError as e:
            raise ValueError(f"Cannot convert headers to float: {e}")

    def short_preprocessings_str(self) -> str:
        processings_list = self._features.sources[0].processing_ids
        processings_list.pop(0)
        processings = "|".join(self.features_processings(0))
        replacements = [
            ("raw_", ""),
            ("SavitzkyGolay", "SG"),
            ("MultiplicativeScatterCorrection", "MSC"),
            ("StandardNormalVariate", "SNV"),
            ("FirstDerivative", "1stDer"),
            ("SecondDerivative", "2ndDer"),
            ("Detrend", "Detr"),
            ("Gaussian", "Gauss"),
            ("Haar", "Haar"),
            ("LogTransform", "Log"),
            ("MinMaxScaler", "MinMax"),
            ("RobustScaler", "Rbt"),
            ("StandardScaler", "Std"),
            ("QuantileTransformer", "Quant"),
            ("PowerTransformer", "Pow"),
            # ("_", ""),
        ]
        for long, short in replacements:
            processings = processings.replace(long, short)

        # replace expr _<digit>_ with | then remaining _<digits> with nothing
        processings = re.sub(r'_\d+_', '>', processings)
        processings = re.sub(r'_\d+', '', processings)
        return processings

    def features_sources(self) -> int:
        return len(self._features.sources)

    def is_multi_source(self) -> bool:
        return len(self._features.sources) > 1

    def is_regression(self) -> bool:
        return self._task_type == "regression"

    def is_classification(self) -> bool:
        return self._task_type in ["binary_classification", "multiclass_classification", "classification"]


    # def targets(self, filter: Dict[str, Any] = {}, encoding: str = "auto") -> np.ndarray:
    #     indices = self._indexer.samples(filter)
    #     return self._targets.y(indices=indices, encoding=encoding)

    #     return self._targets.y(indices=indices, encoding=encoding)

    #     return self._targets.y(indices=indices, encoding=encoding)

    def add_targets(self, y: np.ndarray) -> None:
        self._targets.add_targets(y)
        # Detect and set task type when targets are added
        self._task_type = self._detect_task_type(y)

    def _detect_task_type(self, y: np.ndarray) -> str:
        """
        Detect task type from target values.

        Returns:
            str: "regression", "binary_classification", or "multiclass_classification"
        """
        y_flat = np.array(y).flatten()
        y_clean = y_flat[~np.isnan(y_flat)]  # Remove NaN values

        if len(y_clean) == 0:
            return "regression"  # Default

        unique_values = np.unique(y_clean)
        n_unique = len(unique_values)

        # Check if values are integer-like (classification)
        is_integer_like = np.allclose(y_clean, np.round(y_clean), atol=1e-10)

        if is_integer_like and n_unique <= 50:  # Reasonable threshold for classification
            if n_unique == 2:
                return "binary_classification"
            elif n_unique > 2:
                return "classification"

        return "regression"

    def add_processed_targets(self,
                              processing_name: str,
                              targets: np.ndarray,
                              ancestor_processing: str = "numeric",
                              transformer: Optional[TransformerMixin] = None) -> None:
        new_task_type = self._detect_task_type(targets)
        if self._task_type != new_task_type:
            print(f"🔄 Task type updated from {self._task_type} to {new_task_type}")
            self._task_type = new_task_type

        self._targets.add_processed_targets(processing_name, targets, ancestor_processing, transformer)

    @property
    def task_type(self) -> Optional[str]:
        """Get the detected task type."""
        return self._task_type

    def set_task_type(self, task_type: str) -> None:
        """
        Manually set the task type.

        Args:
            task_type: One of "regression", "binary_classification", "multiclass_classification"
        """
        valid_types = ["regression", "binary_classification", "multiclass_classification", "classification"]
        if task_type not in valid_types:
            raise ValueError(f"Invalid task_type. Must be one of {valid_types}")
        self._task_type = task_type

    # METADATA
    def add_metadata(self,
                     data: Union[np.ndarray, Any],
                     headers: Optional[List[str]] = None) -> None:
        """
        Add metadata rows (aligns with add_samples call order).

        Args:
            data: Metadata as 2D array (n_samples, n_cols) or DataFrame
            headers: Column names (required if data is ndarray)
        """
        self._metadata.add_metadata(data, headers)

    def metadata(self,
                 selector: Optional[Selector] = None,
                 columns: Optional[List[str]] = None,
                 include_augmented: bool = True):
        """
        Get metadata as DataFrame.

        Args:
            selector: Filter selector (e.g., {"partition": "train"})
            columns: Specific columns to return (None = all)
            include_augmented: If True, include augmented versions of selected samples.
                             Default True for backward compatibility.

        Returns:
            Polars DataFrame with metadata
        """
        indices = self._indexer.x_indices(selector, include_augmented=include_augmented) if selector else None
        return self._metadata.get(indices, columns)

    def metadata_column(self,
                        column: str,
                        selector: Optional[Selector] = None,
                        include_augmented: bool = True) -> np.ndarray:
        """
        Get single metadata column as array.

        Args:
            column: Column name
            selector: Filter selector (e.g., {"partition": "train"})
            include_augmented: If True, include augmented versions of selected samples.
                             Default True for backward compatibility.

        Returns:
            Numpy array of column values
        """
        indices = self._indexer.x_indices(selector, include_augmented=include_augmented) if selector else None
        return self._metadata.get_column(column, indices)

    def metadata_numeric(self,
                         column: str,
                         selector: Optional[Selector] = None,
                         method: Literal["label", "onehot"] = "label",
                         include_augmented: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Get numeric encoding of metadata column.

        Args:
            column: Column name
            selector: Filter selector (e.g., {"partition": "train"})
            method: "label" for label encoding or "onehot" for one-hot encoding
            include_augmented: If True, include augmented versions of selected samples.
                             Default True for backward compatibility.

        Returns:
            Tuple of (numeric_array, encoding_info)
        """
        indices = self._indexer.x_indices(selector, include_augmented=include_augmented) if selector else None
        return self._metadata.to_numeric(column, indices, method)

    def update_metadata(self,
                        column: str,
                        values: Union[List, np.ndarray],
                        selector: Optional[Selector] = None,
                        include_augmented: bool = True) -> None:
        """
        Update metadata values for selected samples.

        Args:
            column: Column name
            values: New values
            selector: Filter selector (None = all samples)
            include_augmented: If True, include augmented versions of selected samples.
                             Default True for backward compatibility.
        """
        indices = self._indexer.x_indices(selector, include_augmented=include_augmented) if selector else list(range(self._metadata.num_rows))
        self._metadata.update_metadata(indices, column, values)

    def add_metadata_column(self,
                            column: str,
                            values: Union[List, np.ndarray]) -> None:
        """
        Add new metadata column.

        Args:
            column: Column name
            values: Column values (must match number of samples)
        """
        self._metadata.add_column(column, values)

    @property
    def metadata_columns(self) -> List[str]:
        """Get list of metadata column names."""
        return self._metadata.columns

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
    def num_features(self) -> Union[List[int], int]:
        return self._features.num_features

    @property
    def num_samples(self) -> int:
        return self._features.num_samples

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
        if self._task_type:
            txt += f" ({self._task_type})"
        txt += "\n" + str(self._features)
        txt += "\n" + str(self._targets)
        txt += "\n" + str(self._indexer)
        if self._metadata.num_rows > 0:
            txt += f"\n{str(self._metadata)}"
        if self._folds:
            txt += f"\nFolds: {self._fold_str()}"
        return txt

    # PRINTING AND SUMMARY
    def print_summary(self) -> None:
        """
        Print a comprehensive summary of the dataset.

        Shows counts, dimensions, number of sources, target versions, predictions, etc.
        """
        print("=== SpectroDataset Summary ===")
        print()

        # Task type
        if self._task_type:
            print(f"🎯 Task Type: {self._task_type}")
        else:
            print("🎯 Task Type: Not detected (no targets added yet)")
        print()

        # Features summary
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

        # Metadata summary
        if self._metadata.num_rows > 0:
            print(f"📋 Metadata: {self._metadata.num_rows} rows, {len(self._metadata.columns)} columns")
            print(f"Columns: {self._metadata.columns}")
            print()
        else:
            print("📋 Metadata: None")
            print()

    # IO methods (commented out)
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

