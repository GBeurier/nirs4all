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
from nirs4all.data._dataset import FeatureAccessor, TargetAccessor, MetadataAccessor
from nirs4all.data.signal_type import (
    SignalType,
    SignalTypeInput,
    normalize_signal_type,
    detect_signal_type,
)
from nirs4all.core.logging import get_logger
from nirs4all.core.task_type import TaskType

logger = get_logger(__name__)
from sklearn.base import TransformerMixin
from typing import Optional, Union, List, Tuple, Dict, Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from nirs4all.operators.data.repetition import RepetitionConfig


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

        # Signal type per source (for multi-source support)
        self._signal_types: List[SignalType] = []
        self._signal_type_forced: List[bool] = []

        # Aggregation setting for sample-level prediction aggregation
        self._aggregate_column: Optional[str] = None
        self._aggregate_by_y: bool = False
        self._aggregate_method: Optional[str] = None  # 'mean', 'median', or 'vote' (None = auto)
        self._aggregate_exclude_outliers: bool = False
        self._aggregate_outlier_threshold: float = 0.95

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

    def x(self, selector: Selector, layout: Layout = "2d", concat_source: bool = True, include_augmented: bool = True, include_excluded: bool = False) -> OutputData:
        """
        Get feature data with automatic augmented sample aggregation.

        Args:
            selector: Filter criteria (partition, group, branch, etc.)
            layout: Output layout ("2d" or "3d")
            concat_source: If True, concatenate multiple sources along feature axis
            include_augmented: If True, include augmented versions of selected samples.
                             If False, return only base samples (origin=null).
                             Default True for backward compatibility.
            include_excluded: If True, include samples marked as excluded.
                            If False (default), exclude samples marked as excluded=True.
                            Use True when transforming ALL features (e.g., preprocessing).

        Returns:
            Feature data array(s)

        Example:
            >>> # Get all train samples (base + augmented)
            >>> X_train = dataset.x({"partition": "train"})
            >>> # Get only base train samples (for splitting)
            >>> X_base = dataset.x({"partition": "train"}, include_augmented=False)
            >>> # Get all features including excluded (for transformations)
            >>> X_all = dataset.x({"partition": "train"}, include_excluded=True)
        """
        return self._feature_accessor.x(selector, layout, concat_source, include_augmented, include_excluded)

    # def x_train(self, layout: Layout = "2d", concat_source: bool = True) -> OutputData:
    #     selector = {"partition": "train"}
    #     return self.x(selector, layout, concat_source)

    # def x_test(self, layout: Layout = "2d", concat_source: bool = True) -> OutputData:
    #     selector = {"partition": "test"}
    #     return self.x(selector, layout, concat_source)

    def y(self, selector: Selector, include_augmented: bool = True, include_excluded: bool = False) -> np.ndarray:
        """
        Get target data - automatically maps augmented samples to their origin for y values.

        Args:
            selector: Filter criteria (partition, group, branch, etc.)
            include_augmented: If True, include augmented versions of selected samples.
                             Augmented samples are automatically mapped to their origin's y value.
                             If False, return only base samples.
                             Default True for backward compatibility.
            include_excluded: If True, include samples marked as excluded.
                            If False (default), exclude samples marked as excluded=True.
                            Use True when transforming ALL targets (e.g., y_processing).

        Returns:
            Target values array

        Example:
            >>> # Get all train targets (base + augmented, with mapping)
            >>> y_train = dataset.y({"partition": "train"})
            >>> # Get only base train targets (for splitting)
            >>> y_base = dataset.y({"partition": "train"}, include_augmented=False)
            >>> # Get all targets including excluded (for y_processing)
            >>> y_all = dataset.y({"partition": "train"}, include_excluded=True)
        """
        return self._target_accessor.y(selector, include_augmented, include_excluded)

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

    def add_samples_batch(self,
                          data: Union[np.ndarray, List[np.ndarray]],
                          indexes_list: List[IndexDict]) -> None:
        """
        Add multiple samples in a single batch operation - O(N) instead of O(N²).

        This method is optimized for bulk insertion of augmented samples. It performs
        only one array concatenation and one indexer append, making it dramatically
        faster than calling add_samples() in a loop.

        Args:
            data: 3D array of shape (n_samples, n_processings, n_features) for single source,
                  or list of 3D arrays for multi-source datasets.
            indexes_list: List of index dictionaries, one per sample.

        Example:
            >>> # Batch add 100 augmented samples
            >>> data = np.random.rand(100, 2, 500)
            >>> indexes = [{"partition": "train", "origin": i, "augmentation": "noise"} for i in range(100)]
            >>> dataset.add_samples_batch(data, indexes)
        """
        self._feature_accessor.add_samples_batch(data, indexes_list)

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

    def add_merged_features(
        self,
        features: np.ndarray,
        processing_name: str = "merged",
        source: int = 0,
        processing_names: Optional[List[str]] = None
    ) -> None:
        """Add merged features from branch merge operations.

        This method is used by MergeController to store the output of
        branch merging operations. The merged features REPLACE all existing
        processings to become the new feature set for subsequent steps.

        Args:
            features: Feature array to store:
                - 2D array of shape (n_samples, n_features): flattened features
                - 3D array of shape (n_samples, n_processings, n_features):
                  features with preserved preprocessing dimension
            processing_name: Name for the merged processing (default: "merged").
                Used when features is 2D (single processing).
            source: Target source index (default: 0, first source).
            processing_names: Optional list of processing names for 3D features.
                If not provided, generates names like "merged_0", "merged_1", etc.

        Raises:
            ValueError: If features is not 2D or 3D, or sample count doesn't match.

        Example:
            >>> # 2D merged features (flattened)
            >>> merged = np.concatenate([branch0_features, branch1_features], axis=1)
            >>> dataset.add_merged_features(merged, "merged_snv_msc")
            >>>
            >>> # 3D merged features (preserved preprocessing dimension)
            >>> merged_3d = np.stack([snv_features, msc_features], axis=1)
            >>> dataset.add_merged_features(merged_3d, processing_names=["snv", "msc"])
        """
        if features.ndim not in (2, 3):
            raise ValueError(
                f"Merged features must be 2D or 3D array, got {features.ndim}D"
            )

        n_samples = features.shape[0]
        if n_samples != self.num_samples:
            raise ValueError(
                f"Sample count mismatch: got {n_samples} samples, "
                f"expected {self.num_samples}"
            )

        # Determine processing names based on array dimensionality
        if features.ndim == 2:
            processings = [processing_name]
        else:  # 3D array
            n_processings = features.shape[1]
            if processing_names is not None:
                if len(processing_names) != n_processings:
                    raise ValueError(
                        f"processing_names length ({len(processing_names)}) must match "
                        f"number of processings ({n_processings})"
                    )
                processings = processing_names
            else:
                # Generate default processing names
                processings = [f"{processing_name}_{i}" for i in range(n_processings)]

        # Replace ALL existing processings with the merged features
        # This effectively "resets" the feature storage to just the merged output
        self._feature_accessor.reset_features(
            features=features,
            processings=processings,
            source=source
        )

    def keep_sources(self, source_indices: Union[int, List[int]]) -> None:
        """Keep only specified sources, removing all others.

        Used after merge operations with output_as="features" to consolidate
        to a single source. This is called automatically by MergeController
        when output_as="features" is used.

        Args:
            source_indices: Single source index or list of source indices to keep.

        Raises:
            ValueError: If source indices are invalid.

        Example:
            >>> # After merge with output_as="features", keep only source 0
            >>> dataset.keep_sources(0)
        """
        self._feature_accessor.keep_sources(source_indices)

    def get_merged_features(
        self,
        processing_name: str = "merged",
        source: int = 0,
        selector: Optional[Selector] = None
    ) -> np.ndarray:
        """Get merged features by processing name.

        Retrieves features that were added via add_merged_features().
        Since merged features replace all existing processings, this
        returns the features for the single merged processing.

        Args:
            processing_name: Name of the merged processing (default: "merged").
            source: Source index to get features from (default: 0).
            selector: Optional sample filter.

        Returns:
            2D array of merged features (n_samples, n_merged_features).

        Raises:
            ValueError: If the processing name doesn't exist.

        Example:
            >>> X_merged = dataset.get_merged_features("merged_snv_msc")
            >>> print(X_merged.shape)  # (n_samples, n_merged_features)
        """
        # Verify the processing exists
        processings = self.features_processings(source)
        if processing_name not in processings:
            raise ValueError(
                f"Processing '{processing_name}' not found. "
                f"Available: {processings}"
            )

        # Get features with processing filter using selector
        if selector is None:
            selector = {}

        # Use the x() method to get features
        # Since add_merged_features replaces all processings, there's only
        # one processing now, so we can just return all features
        X = self.x(
            selector=selector,
            layout="2d",
            concat_source=True,
            include_augmented=False,
            include_excluded=False
        )

        return X

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

    # ========== Signal Type Management ==========

    def signal_type(self, src: int = 0) -> SignalType:
        """
        Get the signal type for a data source.

        If not set, attempts auto-detection based on value ranges and
        optionally wavelength band analysis.

        Args:
            src: Source index (default: 0)

        Returns:
            SignalType enum value

        Example:
            >>> signal = dataset.signal_type(0)
            >>> if signal == SignalType.REFLECTANCE:
            ...     dataset.convert_to_absorbance(0)
        """
        # Ensure signal type list is initialized for this source
        self._ensure_signal_type_initialized(src)

        # Return cached value if forced or already detected
        if self._signal_type_forced[src] or self._signal_types[src] != SignalType.AUTO:
            return self._signal_types[src]

        # Auto-detect
        detected, confidence, reason = self._detect_signal_type(src)
        if confidence >= 0.5:
            self._signal_types[src] = detected
        else:
            self._signal_types[src] = SignalType.UNKNOWN

        return self._signal_types[src]

    def set_signal_type(
        self,
        signal_type: SignalTypeInput,
        src: int = 0,
        forced: bool = True
    ) -> None:
        """
        Set the signal type for a data source.

        Args:
            signal_type: Signal type (string or SignalType enum)
            src: Source index (default: 0)
            forced: If True, prevents auto-detection from overriding (default: True)

        Example:
            >>> dataset.set_signal_type("absorbance", src=0)
            >>> dataset.set_signal_type(SignalType.REFLECTANCE_PERCENT, src=1)
        """
        self._ensure_signal_type_initialized(src)
        self._signal_types[src] = normalize_signal_type(signal_type)
        self._signal_type_forced[src] = forced

    def detect_signal_type(
        self,
        src: int = 0,
        force_redetect: bool = False
    ) -> Tuple[SignalType, float, str]:
        """
        Detect signal type using heuristics.

        Uses value range analysis and optionally wavelength band direction
        to determine the most likely signal type.

        Args:
            src: Source index (default: 0)
            force_redetect: If True, ignores cached/forced values and re-runs detection

        Returns:
            Tuple of (SignalType, confidence, reason_string)

        Example:
            >>> signal_type, confidence, reason = dataset.detect_signal_type()
            >>> print(f"Detected {signal_type.value} ({confidence:.0%}): {reason}")
        """
        if not force_redetect:
            self._ensure_signal_type_initialized(src)
            if self._signal_type_forced[src]:
                return self._signal_types[src], 1.0, "User-specified"

        return self._detect_signal_type(src)

    def _detect_signal_type(self, src: int) -> Tuple[SignalType, float, str]:
        """Internal detection logic."""
        if self._feature_accessor.num_samples == 0:
            return SignalType.UNKNOWN, 0.0, "No data available"

        # Get raw features for detection
        spectra = self.x({"partition": "train"}, layout="2d")
        if isinstance(spectra, list):
            spectra = spectra[src] if src < len(spectra) else spectra[0]

        # Get wavelengths if available
        wavelengths = None
        wavelength_unit = "nm"
        try:
            unit = self.header_unit(src)
            headers = self.headers(src)
            if unit in ("nm", "cm-1") and headers is not None and len(headers) > 0:
                wavelengths = self.wavelengths_nm(src) if unit == "nm" else self.wavelengths_cm1(src)
                wavelength_unit = unit
        except (ValueError, IndexError, TypeError):
            pass

        return detect_signal_type(spectra, wavelengths, wavelength_unit)

    def _ensure_signal_type_initialized(self, src: int) -> None:
        """Ensure signal type lists are large enough for the given source."""
        while len(self._signal_types) <= src:
            self._signal_types.append(SignalType.AUTO)
        while len(self._signal_type_forced) <= src:
            self._signal_type_forced.append(False)

    @property
    def signal_types(self) -> List[SignalType]:
        """
        Get signal types for all sources.

        Returns:
            List of SignalType values, one per source
        """
        # Ensure all sources are initialized
        for src in range(self._feature_accessor.num_sources):
            self._ensure_signal_type_initialized(src)
        return self._signal_types[:self._feature_accessor.num_sources]

    # ========== Aggregation Settings ==========

    @property
    def aggregate(self) -> Optional[str]:
        """
        Get the aggregation setting for sample-level prediction aggregation.

        Returns:
            - None: No aggregation
            - 'y': Aggregate by target values (y_true)
            - str: Aggregate by specified metadata column name

        Example:
            >>> dataset.aggregate
            'sample_id'  # Predictions will be aggregated by sample_id column
        """
        if self._aggregate_by_y:
            return 'y'
        return self._aggregate_column

    def set_aggregate(self, value: Union[str, bool, None]) -> None:
        """
        Set the aggregation behavior for sample-level prediction aggregation.

        When set, predictions from multiple spectra of the same biological sample
        (as identified by the aggregation key) will be aggregated automatically
        during scoring and reporting.

        Args:
            value: Aggregation setting
                - None: No aggregation (default behavior)
                - True: Aggregate by y_true values (target grouping)
                - str: Aggregate by specified metadata column (e.g., 'sample_id', 'ID')

        Example:
            >>> dataset.set_aggregate('sample_id')  # Aggregate by sample_id metadata column
            >>> dataset.set_aggregate(True)  # Aggregate by y values
            >>> dataset.set_aggregate(None)  # Disable aggregation
        """
        if value is True:
            self._aggregate_by_y = True
            self._aggregate_column = None
        elif isinstance(value, str):
            self._aggregate_by_y = False
            self._aggregate_column = value
        else:
            self._aggregate_by_y = False
            self._aggregate_column = None

    @property
    def aggregate_method(self) -> str:
        """
        Get the aggregation method for sample-level prediction aggregation.

        Returns:
            str: Aggregation method ('mean', 'median', or 'vote')

        Example:
            >>> dataset.aggregate_method
            'mean'  # Predictions will be averaged within groups
        """
        return self._aggregate_method

    def set_aggregate_method(self, value: Optional[str]) -> None:
        """
        Set the aggregation method for sample-level prediction aggregation.

        Args:
            value: Aggregation method
                - None: Use default method (mean for regression, vote for classification)
                - 'mean': Average predictions within each group
                - 'median': Median prediction within each group
                - 'vote': Majority voting for classification

        Example:
            >>> dataset.set_aggregate_method('median')
        """
        if value is not None:
            valid_methods = ('mean', 'median', 'vote')
            if value not in valid_methods:
                raise ValueError(f"Invalid aggregation method: {value}. Must be one of {valid_methods}")
        self._aggregate_method = value

    @property
    def aggregate_exclude_outliers(self) -> bool:
        """
        Get whether T² outlier exclusion is enabled for aggregation.

        Returns:
            bool: True if outliers should be excluded before aggregation
        """
        return self._aggregate_exclude_outliers

    def set_aggregate_exclude_outliers(self, value: bool, threshold: float = 0.95) -> None:
        """
        Enable/disable T² based outlier exclusion before aggregation.

        When enabled, uses Hotelling's T² statistic to identify and exclude
        outlier measurements within each sample group before averaging.

        Args:
            value: True to enable outlier exclusion, False to disable
            threshold: Confidence level for outlier detection (0-1, default 0.95)

        Example:
            >>> dataset.set_aggregate_exclude_outliers(True, threshold=0.95)
        """
        self._aggregate_exclude_outliers = value
        self._aggregate_outlier_threshold = threshold

    @property
    def aggregate_outlier_threshold(self) -> float:
        """
        Get the outlier detection threshold for T² exclusion.

        Returns:
            float: Confidence level (0-1) for chi-square critical value
        """
        return self._aggregate_outlier_threshold

    # ========== Repetition Transformation Methods ==========

    def _get_sample_groups(self, column: str) -> Dict[Any, List[int]]:
        """Get sample groups by a metadata column or target values.

        Groups samples by the unique values in the specified column, returning
        a mapping from each unique value to the list of row indices that share
        that value.

        Args:
            column: Column name to group by. Special values:
                - "y": Group by target values (y_true)
                - Any other string: Metadata column name

        Returns:
            Dict mapping group keys (sample IDs) to lists of row indices.

        Raises:
            ValueError: If column is "y" but no targets exist, or if column
                not found in metadata.

        Example:
            >>> groups = dataset._get_sample_groups("Sample_ID")
            >>> # {'sample_001': [0, 1, 2, 3], 'sample_002': [4, 5, 6, 7], ...}
        """
        from collections import defaultdict

        if column.lower() == "y":
            # Group by target values
            if self._targets.num_samples == 0:
                raise ValueError(
                    "Cannot group by 'y': no target values available. "
                    "Add targets first using add_targets(). [Error: REP-E003]"
                )
            y = self._targets.get_targets("numeric")
            if y is None or len(y) == 0:
                raise ValueError(
                    "Cannot group by 'y': no target values available. "
                    "Add targets first using add_targets(). [Error: REP-E003]"
                )
            groups: Dict[Any, List[int]] = defaultdict(list)
            for idx, val in enumerate(y):
                # Use tuple for array values (multi-output), else raw value
                key = tuple(val) if hasattr(val, '__iter__') and not isinstance(val, str) else val
                groups[key].append(idx)
            return dict(groups)
        else:
            # Group by metadata column
            if column not in self.metadata_columns:
                raise ValueError(
                    f"Column '{column}' not found in metadata. "
                    f"Available columns: {self.metadata_columns}. [Error: REP-E001]"
                )
            col_values = self.metadata_column(column)
            groups = defaultdict(list)
            for idx, val in enumerate(col_values):
                groups[val].append(idx)
            return dict(groups)

    def _validate_repetition_groups(
        self,
        groups: Dict[Any, List[int]],
        config: "RepetitionConfig"
    ) -> Tuple[Dict[Any, List[int]], int]:
        """Validate and optionally adjust sample groups for equal repetitions.

        Args:
            groups: Dict mapping sample IDs to row indices lists.
            config: RepetitionConfig with validation settings.

        Returns:
            Tuple of (validated_groups, n_reps) where n_reps is the number
            of repetitions per sample.

        Raises:
            ValueError: If groups have unequal sizes and on_unequal="error",
                or if expected_reps doesn't match actual counts.
        """
        from nirs4all.operators.data.repetition import UnequelRepsStrategy

        if not groups:
            raise ValueError(
                "No sample groups found. Check that the grouping column "
                "has valid values. [Error: REP-E002]"
            )

        # Get counts per group
        counts = {k: len(v) for k, v in groups.items()}
        unique_counts = set(counts.values())
        min_count = min(unique_counts)
        max_count = max(unique_counts)

        # Determine expected repetitions
        if config.expected_reps is not None:
            n_reps = config.expected_reps
        else:
            # Use mode (most common count) as expected
            from collections import Counter
            count_freq = Counter(counts.values())
            n_reps = count_freq.most_common(1)[0][0]

        # Check for unequal counts
        if len(unique_counts) > 1 or (config.expected_reps and n_reps not in unique_counts):
            strategy = config.get_unequal_strategy()

            if strategy == UnequelRepsStrategy.ERROR:
                count_summary = {c: sum(1 for v in counts.values() if v == c) for c in unique_counts}
                raise ValueError(
                    f"Unequal repetition counts detected: {count_summary}. "
                    f"Expected {n_reps} repetitions per sample. "
                    f"Use on_unequal='drop', 'pad', or 'truncate' to handle this. "
                    f"[Error: REP-E002]"
                )

            elif strategy == UnequelRepsStrategy.DROP:
                # Keep only groups with expected count
                groups = {k: v for k, v in groups.items() if len(v) == n_reps}
                if not groups:
                    raise ValueError(
                        f"After dropping, no samples remain with {n_reps} repetitions. "
                        f"[Error: REP-E002]"
                    )
                logger.warning(
                    f"Dropped {len(counts) - len(groups)} samples with != {n_reps} repetitions"
                )

            elif strategy == UnequelRepsStrategy.TRUNCATE:
                # Use minimum count
                n_reps = min_count
                groups = {k: v[:n_reps] for k, v in groups.items()}
                logger.warning(
                    f"Truncated all groups to {n_reps} repetitions (minimum)"
                )

            elif strategy == UnequelRepsStrategy.PAD:
                # Pad will be handled during reshaping (use NaN/repeat last)
                n_reps = max_count
                logger.warning(
                    f"Will pad shorter groups to {n_reps} repetitions with NaN"
                )

        # Sort indices within each group if preserve_order is True
        if config.preserve_order:
            groups = {k: sorted(v) for k, v in groups.items()}

        return groups, n_reps

    def reshape_reps_to_sources(self, config: "RepetitionConfig") -> None:
        """Transform repetitions into separate data sources.

        Each repetition index becomes a new source, reducing the number of
        samples but increasing the number of sources. This enables per-source
        branching and multi-source modeling strategies.

        Input: n_sources × (n_samples, n_pp, n_features)
        Output: (n_sources × n_reps) × (n_unique_samples, n_pp, n_features)

        Args:
            config: RepetitionConfig with column and options.

        Raises:
            ValueError: If grouping column not found, groups have unequal sizes
                and on_unequal="error", or no valid groups found.

        Example:
            >>> # With 120 samples (30 unique × 4 reps), 1 source, 500 features
            >>> config = RepetitionConfig(column="Sample_ID")
            >>> dataset.reshape_reps_to_sources(config)
            >>> # Result: 4 sources × (30 samples, 1 pp, 500 features)
        """
        # Resolve column
        column = config.resolve_column(self.aggregate)

        # Get and validate groups
        groups = self._get_sample_groups(column)
        groups, n_reps = self._validate_repetition_groups(groups, config)

        n_unique = len(groups)
        n_existing_sources = self.n_sources

        logger.info(
            f"Reshaping repetitions to sources: "
            f"{self.num_samples} samples → {n_unique} samples × {n_reps} reps"
        )
        logger.info(f"  Sources: {n_existing_sources} → {n_existing_sources * n_reps}")

        # Collect new source data
        # For each existing source, create n_reps new sources
        new_sources_data = []
        new_processing_ids = []

        for src_idx in range(n_existing_sources):
            # Get current source data in 3D layout
            X = self.x({}, layout="3d", concat_source=False, include_augmented=True, include_excluded=True)
            if isinstance(X, list):
                X_src = X[src_idx]
            else:
                X_src = X

            n_pp = X_src.shape[1]
            n_features = X_src.shape[2]
            processing_ids = self.features_processings(src_idx)

            # Create one new source per repetition index
            for rep_idx in range(n_reps):
                # Gather samples for this repetition
                rep_data = np.zeros((n_unique, n_pp, n_features), dtype=X_src.dtype)

                for sample_idx, (sample_id, row_indices) in enumerate(groups.items()):
                    if rep_idx < len(row_indices):
                        rep_data[sample_idx] = X_src[row_indices[rep_idx]]
                    else:
                        # Pad with NaN for missing repetitions
                        rep_data[sample_idx] = np.nan

                new_sources_data.append(rep_data)
                new_processing_ids.append(processing_ids)

        # Get sample keys in order
        sample_keys = list(groups.keys())

        # Rebuild metadata - take first row from each group
        first_indices = [groups[k][0] for k in sample_keys]
        old_metadata = self._metadata.df if self._metadata.num_rows > 0 else None

        # Clear and rebuild the dataset
        self._rebuild_dataset_from_sources(
            sources_data=new_sources_data,
            processing_ids=new_processing_ids,
            sample_keys=sample_keys,
            old_metadata=old_metadata,
            first_indices=first_indices,
            config=config,
            transformation_type="sources"
        )

        logger.success(
            f"Reshaped to {len(new_sources_data)} sources × {n_unique} samples"
        )

    def reshape_reps_to_preprocessings(self, config: "RepetitionConfig") -> None:
        """Transform repetitions into additional preprocessing slots.

        Each repetition becomes a new preprocessing dimension, reducing the
        number of samples but increasing the preprocessing count. This enables
        multi-preprocessing modeling strategies.

        Input: n_sources × (n_samples, n_pp, n_features)
        Output: n_sources × (n_unique_samples, n_pp × n_reps, n_features)

        Args:
            config: RepetitionConfig with column and options.

        Raises:
            ValueError: If grouping column not found, groups have unequal sizes
                and on_unequal="error", or no valid groups found.

        Example:
            >>> # With 120 samples (30 unique × 4 reps), 1 source, 1 pp, 500 features
            >>> config = RepetitionConfig(column="Sample_ID")
            >>> dataset.reshape_reps_to_preprocessings(config)
            >>> # Result: 1 source × (30 samples, 4 pp, 500 features)
        """
        # Resolve column
        column = config.resolve_column(self.aggregate)

        # Get and validate groups
        groups = self._get_sample_groups(column)
        groups, n_reps = self._validate_repetition_groups(groups, config)

        n_unique = len(groups)
        n_existing_sources = self.n_sources

        logger.info(
            f"Reshaping repetitions to preprocessings: "
            f"{self.num_samples} samples → {n_unique} samples × {n_reps} reps"
        )

        # Collect new source data
        new_sources_data = []
        new_processing_ids = []

        for src_idx in range(n_existing_sources):
            # Get current source data in 3D layout
            X = self.x({}, layout="3d", concat_source=False, include_augmented=True, include_excluded=True)
            if isinstance(X, list):
                X_src = X[src_idx]
            else:
                X_src = X

            n_existing_pp = X_src.shape[1]
            n_features = X_src.shape[2]
            existing_processing_ids = self.features_processings(src_idx)

            # New shape: (n_unique, n_existing_pp * n_reps, n_features)
            new_n_pp = n_existing_pp * n_reps
            new_data = np.zeros((n_unique, new_n_pp, n_features), dtype=X_src.dtype)

            # Build new processing names
            new_pp_names = []
            for rep_idx in range(n_reps):
                for pp_name in existing_processing_ids:
                    new_pp_names.append(config.get_pp_name(rep_idx, pp_name))

            # Fill data: for each unique sample, stack all repetitions
            for sample_idx, (sample_id, row_indices) in enumerate(groups.items()):
                for rep_idx in range(n_reps):
                    if rep_idx < len(row_indices):
                        row_idx = row_indices[rep_idx]
                        # Copy all existing preprocessings for this repetition
                        pp_start = rep_idx * n_existing_pp
                        pp_end = pp_start + n_existing_pp
                        new_data[sample_idx, pp_start:pp_end] = X_src[row_idx]
                    else:
                        # Pad with NaN
                        pp_start = rep_idx * n_existing_pp
                        pp_end = pp_start + n_existing_pp
                        new_data[sample_idx, pp_start:pp_end] = np.nan

            new_sources_data.append(new_data)
            new_processing_ids.append(new_pp_names)

            logger.info(
                f"  Source {src_idx}: {n_existing_pp} pp → {new_n_pp} pp"
            )

        # Get sample keys in order
        sample_keys = list(groups.keys())

        # Rebuild metadata - take first row from each group
        first_indices = [groups[k][0] for k in sample_keys]
        old_metadata = self._metadata.df if self._metadata.num_rows > 0 else None

        # Clear and rebuild the dataset
        self._rebuild_dataset_from_sources(
            sources_data=new_sources_data,
            processing_ids=new_processing_ids,
            sample_keys=sample_keys,
            old_metadata=old_metadata,
            first_indices=first_indices,
            config=config,
            transformation_type="preprocessings"
        )

        logger.success(
            f"Reshaped to {n_unique} samples × {new_sources_data[0].shape[1]} preprocessings"
        )

    def _rebuild_dataset_from_sources(
        self,
        sources_data: List[np.ndarray],
        processing_ids: List[List[str]],
        sample_keys: List[Any],
        old_metadata: Optional[Any],
        first_indices: List[int],
        config: "RepetitionConfig",
        transformation_type: str
    ) -> None:
        """Rebuild dataset with new sources/structure after repetition transformation.

        This internal method replaces the feature storage, indexer, and metadata
        with new data after rep_to_sources or rep_to_pp transformation.

        Args:
            sources_data: List of 3D arrays, one per new source.
            processing_ids: List of processing ID lists, one per new source.
            sample_keys: Sample identifiers (from grouping column).
            old_metadata: Original metadata DataFrame (or None).
            first_indices: Row indices to extract metadata from (first per group).
            config: RepetitionConfig for naming options.
            transformation_type: "sources" or "preprocessings" for logging.
        """
        n_samples = len(sample_keys)
        n_sources = len(sources_data)

        # Reset feature storage
        from nirs4all.data._features import FeatureSource
        from nirs4all.data.features import Features
        from nirs4all.data.indexer import Indexer

        # Create new features block
        new_features = Features()
        new_features.sources = []

        for src_idx, (data_3d, pp_ids) in enumerate(zip(sources_data, processing_ids)):
            source = FeatureSource()
            # Add samples as 2D (first pp), then reset with full 3D
            source.reset_features(data_3d, pp_ids)
            new_features.sources.append(source)

        # Create new indexer with reduced sample count
        new_indexer = Indexer()

        # Get partition info from old indexer if possible
        old_partitions = self._indexer.get_column_values("partition")
        new_partitions = [str(old_partitions[idx]) for idx in first_indices] if old_partitions else None

        # Get group info
        old_groups = self._indexer.get_column_values("group")
        new_groups = [old_groups[idx] for idx in first_indices] if old_groups else None

        # Add samples to new indexer by partition to respect the partition type constraint
        # Partition needs to be a single value, so we add samples per partition group
        if new_partitions:
            # Group samples by partition
            from collections import defaultdict
            partition_indices: Dict[str, List[int]] = defaultdict(list)
            for idx, part in enumerate(new_partitions):
                partition_indices[part].append(idx)

            # Add samples partition by partition
            for partition, idxs in partition_indices.items():
                groups_for_partition = [new_groups[i] for i in idxs] if new_groups else None
                new_indexer.add_samples(
                    count=len(idxs),
                    partition=partition,  # type: ignore
                    group=groups_for_partition,
                    processings=processing_ids[0] if processing_ids else ["raw"]
                )
        else:
            # All train by default
            new_indexer.add_samples(
                count=n_samples,
                partition="train",
                group=new_groups,
                processings=processing_ids[0] if processing_ids else ["raw"]
            )

        # Rebuild metadata if exists
        if old_metadata is not None and len(old_metadata) > 0:
            import polars as pl

            # Extract rows corresponding to first_indices
            new_metadata_rows = old_metadata[first_indices]

            # Reset metadata block using add_metadata (proper method)
            from nirs4all.data.metadata import Metadata
            new_metadata = Metadata()
            # Remove row_id if present before adding
            if "row_id" in new_metadata_rows.columns:
                new_metadata_rows = new_metadata_rows.drop("row_id")
            new_metadata.add_metadata(new_metadata_rows)

            self._metadata = new_metadata
            self._metadata_accessor = self._metadata_accessor.__class__(
                new_indexer, new_metadata
            )

        # Rebuild targets if exists (use first_indices)
        if self._targets.num_samples > 0:
            old_y = self._targets.get_targets("numeric")
            if old_y is not None and len(old_y) > 0:
                new_y = old_y[first_indices]
                # Reset targets block
                from nirs4all.data.targets import Targets
                new_targets = Targets()
                new_targets.add_targets(new_y)

                # Copy task type
                if self._targets.task_type:
                    new_targets.set_task_type(self._targets.task_type, forced=True)

                self._targets = new_targets
                self._target_accessor = self._target_accessor.__class__(
                    new_indexer, new_targets
                )

        # Replace internal storage
        self._features = new_features
        self._indexer = new_indexer
        self._feature_accessor = self._feature_accessor.__class__(
            new_indexer, new_features
        )

        # Reset signal types for new sources
        self._signal_types = [SignalType.AUTO] * n_sources
        self._signal_type_forced = [False] * n_sources

        # Clear folds (CV needs to be re-run on new sample count)
        self._folds = []

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

    def set_task_type(self, task_type: Union[str, TaskType], forced: bool = True) -> None:
        """Set the task type explicitly.

        Args:
            task_type: Task type as string ('regression', 'binary_classification', 'multiclass_classification') or TaskType enum
            forced: If True, prevents auto-detection from overriding this value
                   in subsequent y_processing steps (e.g., after MinMaxScaler). Default True.
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
        self._targets.set_task_type(task_type, forced)

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
        txt = f"[Dataset] {self.name}"
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
        logger.info("=== SpectroDataset Summary ===")

        # Task type
        task_type = self._target_accessor.task_type
        if task_type:
            logger.info(f"Task Type: {task_type}")
        else:
            logger.info("Task Type: Not detected (no targets added yet)")

        # Features summary
        if self._features.sources:
            total_samples = self._feature_accessor.num_samples
            n_sources = self._feature_accessor.num_sources
            logger.info(f"Features: {total_samples} samples, {n_sources} source(s)")
            logger.info(f"Features: {self._feature_accessor.num_features}, processings: {self._features.num_processings}")
            logger.info(f"Processing IDs: {self._features.preprocessing_str}")

            # Signal types per source
            signal_types_str = []
            for src in range(n_sources):
                sig_type = self.signal_type(src)
                forced_marker = "*" if self._signal_type_forced[src] else ""
                signal_types_str.append(f"{sig_type.value}{forced_marker}")
            logger.info(f"Signal types: [{', '.join(signal_types_str)}] (* = user-specified)")
        else:
            logger.info("Features: No data")

        # Metadata summary
        if self._metadata.num_rows > 0:
            logger.info(f"Metadata: {self._metadata.num_rows} rows, {len(self._metadata.columns)} columns")
            logger.info(f"Columns: {self._metadata.columns}")
        else:
            logger.info("Metadata: None")
