"""
Main SpectroDataset orchestrator class.

This module contains the main facade that coordinates all dataset blocks
and provides the primary public API for users.
"""


import re
import numpy as np

from nirs4all.data.types import Selector, SourceSelector, OutputData, InputData, Layout, IndexDict, InputFeatures, ProcessingList
from nirs4all.data.features import Features
from nirs4all.data.targets import Targets
from nirs4all.data.indexer import Indexer
from nirs4all.data.metadata import Metadata
from nirs4all.data.predictions import Predictions
from nirs4all.data.dataset_components import FeatureAccessor, TargetAccessor, MetadataAccessor
from nirs4all.utils.emoji import CHART, REFRESH, TARGET
from nirs4all.utils.model_utils import ModelUtils, TaskType
from sklearn.base import TransformerMixin
from typing import Optional, Union, List, Tuple, Dict, Any, Literal


class SpectroDataset:
    """
    Main dataset facade for spectroscopy and ML/DL pipelines.

    Coordinates feature, target, and metadata management through
    specialized accessor interfaces. The primary API uses direct methods
    like dataset.x() and dataset.y() for convenience.

    Attributes:
        name (str): Dataset identifier
        features (FeatureAccessor): Feature data accessor (internal use)
        targets (TargetAccessor): Target data accessor (internal use)
        metadata_accessor (MetadataAccessor): Metadata accessor (internal use)
        folds (List[Tuple]): Cross-validation fold splits

    Examples:
        >>> # Create dataset
        >>> dataset = SpectroDataset("my_dataset")
        >>> # Add samples
        >>> dataset.add_samples(X_train, {"partition": "train"})
        >>> dataset.add_targets(y_train)
        >>> # Get data
        >>> X = dataset.x({"partition": "train"})
        >>> y = dataset.y({"partition": "train"})
    """
    def __init__(self, name: str = "Unknown_dataset"):
        """
        Initialize a new SpectroDataset.

        Args:
            name: Dataset identifier (default: "Unknown_dataset")
        """
        self._indexer = Indexer()
        self._folds: List[Tuple[List[int], List[int]]] = []
        self.name = name

        # Initialize internal blocks
        _features_block = Features()
        _targets_block = Targets()
        _metadata_block = Metadata()

        # Create accessors (internal use, not primary API)
        self._feature_accessor = FeatureAccessor(self._indexer, _features_block)
        self._target_accessor = TargetAccessor(self._indexer, _targets_block)
        self._metadata_accessor = MetadataAccessor(self._indexer, _metadata_block)

        # Keep direct references for backward compatibility with internal code
        self._features = _features_block
        self._targets = _targets_block
        self._metadata = _metadata_block

    # ========== PRIMARY API: Feature Methods ==========

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
        return self._feature_accessor.x(selector, layout, concat_source, include_augmented)

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
        return self._target_accessor.y(selector, include_augmented)

    def add_samples(self,
                    data: InputData,
                    indexes: Optional[IndexDict] = None,
                    headers: Optional[Union[List[str], List[List[str]]]] = None,
                    header_unit: Optional[Union[str, List[str]]] = None) -> None:
        """
        Add feature samples to the dataset.

        Args:
            data: Feature data (single or multi-source)
            indexes: Optional index dictionary (partition, group, branch, fold)
            headers: Feature headers (wavelengths, feature names)
            header_unit: Unit type for headers ("cm-1", "nm", "none", "text", "index")
        """
        self._feature_accessor.add_samples(data, indexes, headers, header_unit)

    def add_features(self,
                     features: InputFeatures,
                     processings: ProcessingList,
                     source: int = -1) -> None:
        """Add processed feature versions to existing data."""
        self._feature_accessor.add_features(features, processings, source)

    def replace_features(self,
                         source_processings: ProcessingList,
                         features: InputFeatures,
                         processings: ProcessingList,
                         source: int = -1) -> None:
        """Replace existing processed features with new versions."""
        self._feature_accessor.replace_features(source_processings, features, processings, source)

    def update_features(self,
                        source_processings: ProcessingList,
                        features: InputFeatures,
                        processings: ProcessingList,
                        source: int = -1) -> None:
        """Update existing processed features."""
        self._feature_accessor.update_features(source_processings, features, processings, source)

    def augment_samples(self,
                        data: InputData,
                        processings: ProcessingList,
                        augmentation_id: str,
                        selector: Optional[Selector] = None,
                        count: Union[int, List[int]] = 1) -> List[int]:
        """Create augmented versions of existing samples."""
        return self._feature_accessor.augment_samples(data, processings, augmentation_id, selector, count)

    def features_processings(self, src: int) -> List[str]:
        """Get processing names for a source."""
        return self._feature_accessor.processing_names(src)

    def headers(self, src: int) -> List[str]:
        """Get feature headers for a source."""
        return self._feature_accessor.headers(src)

    def header_unit(self, src: int) -> str:
        """
        Get the unit type of headers for a data source.

        Args:
            src: Source index

        Returns:
            Unit string: "cm-1", "nm", "none", "text", "index"
        """
        return self._feature_accessor.header_unit(src)

    def float_headers(self, src: int) -> np.ndarray:
        """
        Get headers as float array (legacy method).

        WARNING: This method assumes headers are numeric and doesn't handle unit conversion.
        Use wavelengths_cm1() or wavelengths_nm() for wavelength data.

        Args:
            src: Source index

        Returns:
            Headers converted to float array

        Raises:
            ValueError: If headers cannot be converted to float
        """
        return self._feature_accessor.float_headers(src)

    def wavelengths_cm1(self, src: int) -> np.ndarray:
        """
        Get wavelengths in cm⁻¹ (wavenumber), converting from nm if needed.

        Args:
            src: Source index

        Returns:
            Wavelengths in cm⁻¹ as float array

        Raises:
            ValueError: If headers cannot be converted to wavelengths
        """
        return self._feature_accessor.wavelengths_cm1(src)

    def wavelengths_nm(self, src: int) -> np.ndarray:
        """
        Get wavelengths in nm, converting from cm⁻¹ if needed.

        Args:
            src: Source index

        Returns:
            Wavelengths in nm as float array

        Raises:
            ValueError: If headers cannot be converted to wavelengths
        """
        return self._feature_accessor.wavelengths_nm(src)

    def short_preprocessings_str(self) -> str:
        """Get shortened processing string for display."""
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
        ]
        for long, short in replacements:
            processings = processings.replace(long, short)

        # replace expr _<digit>_ with | then remaining _<digits> with nothing
        processings = re.sub(r'_\d+_', '>', processings)
        processings = re.sub(r'_\d+', '', processings)
        return processings

    def features_sources(self) -> int:
        """Get number of feature sources."""
        return self._feature_accessor.num_sources

    def is_multi_source(self) -> bool:
        """Check if dataset has multiple feature sources."""
        return self._feature_accessor.is_multi_source

    # ========== PRIMARY API: Target Methods ==========

    def add_targets(self, y: np.ndarray) -> None:
        """Add target samples to the dataset."""
        self._target_accessor.add_targets(y)

    def add_processed_targets(self,
                              processing_name: str,
                              targets: np.ndarray,
                              ancestor_processing: str = "numeric",
                              transformer: Optional[TransformerMixin] = None) -> None:
        """Add processed target version (e.g., scaled, encoded)."""
        self._target_accessor.add_processed_targets(processing_name, targets, ancestor_processing, transformer)

    @property
    def task_type(self) -> Optional[TaskType]:
        """Get the detected task type."""
        return self._target_accessor.task_type

    def set_task_type(self, task_type: Union[str, TaskType]) -> None:
        """Set the task type explicitly.

        Args:
            task_type: Task type as string ('regression', 'binary_classification', 'multiclass_classification') or TaskType enum
        """
        if isinstance(task_type, str):
            # Map common string values to TaskType enum
            task_map = {
                'regression': TaskType.REGRESSION,
                'binary': TaskType.BINARY_CLASSIFICATION,
                'binary_classification': TaskType.BINARY_CLASSIFICATION,
                'multiclass': TaskType.MULTICLASS_CLASSIFICATION,
                'multiclass_classification': TaskType.MULTICLASS_CLASSIFICATION,
            }
            task_type = task_map.get(task_type.lower(), TaskType.REGRESSION)
        self._targets._task_type = task_type

    @property
    def num_classes(self) -> int:
        """Get the number of unique classes for classification tasks."""
        return self._target_accessor.num_classes

    @property
    def is_regression(self) -> bool:
        """Check if dataset is for regression task."""
        task_type = self._target_accessor.task_type
        return task_type == TaskType.REGRESSION if task_type else False

    @property
    def is_classification(self) -> bool:
        """Check if dataset is for classification task."""
        task_type = self._target_accessor.task_type
        return task_type.is_classification if task_type else False

    # ========== PRIMARY API: Metadata Methods ==========

    def add_metadata(self,
                     data: Union[np.ndarray, Any],
                     headers: Optional[List[str]] = None) -> None:
        """
        Add metadata rows (aligns with add_samples call order).

        Args:
            data: Metadata as 2D array (n_samples, n_cols) or DataFrame
            headers: Column names (required if data is ndarray)
        """
        self._metadata_accessor.add_metadata(data, headers)

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
        return self._metadata_accessor.get(selector, columns, include_augmented)

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
        return self._metadata_accessor.column(column, selector, include_augmented)

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
        return self._metadata_accessor.to_numeric(column, selector, method, include_augmented)

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
        self._metadata_accessor.update_metadata(column, values, selector, include_augmented)

    def add_metadata_column(self,
                            column: str,
                            values: Union[List, np.ndarray]) -> None:
        """
        Add new metadata column.

        Args:
            column: Column name
            values: Column values (must match number of samples)
        """
        self._metadata_accessor.add_column(column, values)

    @property
    def metadata_columns(self) -> List[str]:
        """Get list of metadata column names."""
        return self._metadata_accessor.columns

    # ========== Cross-Validation Folds ==========

    # ========== Cross-Validation Folds ==========

    @property
    def folds(self) -> List[Tuple[List[int], List[int]]]:
        """Get cross-validation folds."""
        return self._folds

    def set_folds(self, folds_iterable) -> None:
        """Set cross-validation folds from an iterable of (train_idx, val_idx) tuples."""
        self._folds = list(folds_iterable)

    @property
    def num_folds(self) -> int:
        """Return the number of folds."""
        return len(self._folds)

    def _fold_str(self) -> str:
        """Get string representation of folds."""
        if not self._folds:
            return ""
        folds_count = [(len(train), len(val)) for train, val in self._folds]
        return str(folds_count)

    # ========== Index and Size Properties ==========

    def index_column(self, col: str, filter: Dict[str, Any] = {}) -> List[int]:
        """Get values from index column."""
        return self._indexer.get_column_values(col, filter)

    @property
    def num_features(self) -> Union[List[int], int]:
        """Get number of features per source."""
        return self._feature_accessor.num_features

    @property
    def num_samples(self) -> int:
        """Get total number of samples."""
        return self._feature_accessor.num_samples

    @property
    def n_sources(self) -> int:
        """Get number of feature sources."""
        return self._feature_accessor.num_sources

    # ========== String Representations ==========

    def __str__(self):
        """Return readable dataset summary."""
        txt = f"{CHART}Dataset: {self.name}"
        if self._target_accessor.task_type:
            txt += f" ({self._target_accessor.task_type})"
        txt += "\n" + str(self._features)
        txt += "\n" + str(self._targets)
        txt += "\n" + str(self._indexer)
        if self._metadata.num_rows > 0:
            txt += f"\n{str(self._metadata)}"
        if self._folds:
            txt += f"\nFolds: {self._fold_str()}"
        return txt

    def print_summary(self) -> None:
        """
        Print a comprehensive summary of the dataset.

        Shows counts, dimensions, number of sources, target versions, etc.
        """
        print("=== SpectroDataset Summary ===")
        print()

        # Task type
        task_type = self._target_accessor.task_type
        if task_type:
            print(f"{TARGET} Task Type: {task_type}")
        else:
            print(f"{TARGET} Task Type: Not detected (no targets added yet)")
        print()

        # Features summary
        if self._features.sources:
            total_samples = self._feature_accessor.num_samples
            n_sources = self._feature_accessor.num_sources
            print(f"{CHART}Features: {total_samples} samples, {n_sources} source(s)")
            print(f"Features: {self._feature_accessor.num_features}, processings: {self._features.num_processings}")
            print(f"Processing IDs: {self._features.preprocessing_str}")
        else:
            print(f"{CHART}Features: No data")
        print()

        # Metadata summary
        if self._metadata.num_rows > 0:
            print(f"📋 Metadata: {self._metadata.num_rows} rows, {len(self._metadata.columns)} columns")
            print(f"Columns: {self._metadata.columns}")
            print()
        else:
            print("📋 Metadata: None")
            print()
