"""
Feature accessor for managing feature data operations.

This module provides a dedicated interface for all feature-related
operations, including data retrieval, augmentation, and wavelength conversions.
"""

import numpy as np
from typing import Optional, Union, List

from nirs4all.data.types import Selector, Layout, OutputData, InputData, IndexDict, InputFeatures, ProcessingList, get_num_samples
from nirs4all.data.indexer import Indexer
from nirs4all.data.features import Features


def _selector_to_dict(selector: Optional[Selector]) -> IndexDict:
    """
    Convert selector to dict format for internal use.

    Handles both legacy dict format and new DataSelector format.

    Args:
        selector: Selector in any format

    Returns:
        Dict representation of selector
    """
    if selector is None:
        return {}

    # Handle ExecutionContext (duck typing to avoid circular import)
    if hasattr(selector, "selector") and hasattr(selector, "state"):
        return dict(selector.selector)

    return dict(selector)


class FeatureAccessor:
    """
    Accessor for feature data operations.

    Provides methods for adding, retrieving, and manipulating feature data
    across multiple sources with different processing chains.

    Attributes:
        num_samples (int): Number of samples in the dataset
        num_features (Union[List[int], int]): Number of features per source
        num_sources (int): Number of feature sources

    Examples:
        >>> # Used internally by SpectroDataset, but can be accessed as:
        >>> # dataset.features.num_sources
        >>> # dataset.features.headers(0)
    """

    def __init__(self, indexer: Indexer, features_block: Features):
        """
        Initialize feature accessor.

        Args:
            indexer: Sample index manager for filtering
            features_block: Underlying feature storage
        """
        self._indexer = indexer
        self._block = features_block

    def x(self,
          selector: Optional[Selector] = None,
          layout: Layout = "2d",
          concat_source: bool = True,
          include_augmented: bool = True,
          include_excluded: bool = False) -> OutputData:
        """
        Get feature data with filtering and layout control.

        Args:
            selector: Filter criteria. Supported keys:
                - partition: "train", "test", "val"
                - group: group identifier
                - branch: branch identifier
                - fold: fold number
                - sample_ids: list of specific sample IDs
            layout: Output layout:
                - "2d": Shape (n_samples, n_features)
                - "3d": Shape (n_samples, n_processings, n_features)
            concat_source: If True, concatenate multiple sources along feature axis
            include_augmented: If True, include augmented versions of selected samples
            include_excluded: If True, include samples marked as excluded.
                            If False (default), exclude samples marked as excluded=True.
                            Use True when transforming ALL features (e.g., preprocessing).

        Returns:
            Feature array(s) matching the selector criteria.
            - Single source + 2D: np.ndarray shape (n_samples, n_features)
            - Single source + 3D: np.ndarray shape (n_samples, n_processings, n_features)
            - Multi-source + concat_source=True: np.ndarray (concatenated features)
            - Multi-source + concat_source=False: List[np.ndarray]

        Raises:
            ValueError: If no features available

        Examples:
            >>> # Get all train data
            >>> X_train = dataset.x({"partition": "train"})
            >>> # Get base train samples only (for splitting)
            >>> X_base = dataset.x({"partition": "train"}, include_augmented=False)
            >>> # Get 3D data for deep learning
            >>> X_3d = dataset.x(
            ...     {"partition": "train"},
            ...     layout="3d",
            ...     include_augmented=True
            ... )
            >>> # Get all features including excluded (for transformations)
            >>> X_all = dataset.x({"partition": "train"}, include_excluded=True)
        """
        # Convert selector to dict format for internal use
        selector_dict = _selector_to_dict(selector)
        indices = self._indexer.x_indices(selector_dict, include_augmented, include_excluded)
        return self._block.x(indices, layout, concat_source)

    def add_samples(self,
                    data: InputData,
                    indexes: Optional[IndexDict] = None,
                    headers: Optional[Union[List[str], List[List[str]]]] = None,
                    header_unit: Optional[Union[str, List[str]]] = None) -> None:
        """
        Add feature samples to the dataset.

        This is the primary method for loading spectral or feature data.
        Automatically registers samples in the indexer.

        Args:
            data: Feature data. Can be:
                - Single source: np.ndarray shape (n_samples, n_features)
                - Multi-source: List[np.ndarray] with compatible n_samples
            indexes: Optional index dictionary with keys:
                - partition: "train", "test", "val", etc.
                - group: group identifiers
                - branch: branch identifiers
                - fold: fold numbers
            headers: Feature headers (wavelengths, feature names):
                - Single source: List[str]
                - Multi-source: List[List[str]]
            header_unit: Unit type for headers:
                - Single source: "cm-1", "nm", "none", "text", "index"
                - Multi-source: List[str] with unit per source

        Raises:
            ValueError: If data dimensions don't match existing samples
            ValueError: If number of sources doesn't match

        Examples:
            >>> # Basic usage
            >>> data = np.random.rand(100, 200)
            >>> dataset.add_samples(data, {"partition": "train"})
            >>>
            >>> # With wavelength headers
            >>> headers = [f"{w}" for w in range(1000, 2500, 10)]
            >>> dataset.add_samples(
            ...     data,
            ...     {"partition": "train"},
            ...     headers=headers,
            ...     header_unit="nm"
            ... )
        """
        num_samples = get_num_samples(data)
        self._indexer.add_samples_dict(num_samples, indexes)
        self._block.add_samples(data, headers, header_unit)

    def add_samples_batch(
        self,
        data: Union[np.ndarray, List[np.ndarray]],
        indexes_list: List[IndexDict]
    ) -> None:
        """Add multiple samples in a single batch operation - O(N) instead of O(N²).

        This method is optimized for bulk insertion of augmented samples. It performs
        only one array concatenation and one indexer append, making it dramatically
        faster than calling add_samples() in a loop.

        Args:
            data: 3D array of shape (n_samples, n_processings, n_features) for single source,
                  or list of 3D arrays for multi-source datasets.
            indexes_list: List of index dictionaries, one per sample. Each dict can contain:
                - partition: "train", "test", "val"
                - origin: original sample ID (for augmented samples)
                - augmentation: augmentation type identifier
                - group, branch, etc.

        Raises:
            ValueError: If data and indexes_list lengths don't match.

        Example:
            >>> # Batch add 100 augmented samples
            >>> data = np.random.rand(100, 2, 500)  # 100 samples, 2 processings, 500 features
            >>> indexes = [{"partition": "train", "origin": i, "augmentation": "noise"} for i in range(100)]
            >>> dataset.add_samples_batch(data, indexes)
        """
        # Determine number of samples from data
        if isinstance(data, list):
            n_samples = data[0].shape[0]
        else:
            n_samples = data.shape[0]

        if n_samples != len(indexes_list):
            raise ValueError(
                f"Data has {n_samples} samples but got {len(indexes_list)} index entries"
            )

        if n_samples == 0:
            return

        # Extract common fields for batch indexer insert
        # Group indexes by field for efficient batch processing
        partitions = []
        origins = []
        augmentations = []
        groups = []
        branches = []

        for idx_dict in indexes_list:
            partitions.append(idx_dict.get("partition", "train"))
            origins.append(idx_dict.get("origin"))
            augmentations.append(idx_dict.get("augmentation"))
            groups.append(idx_dict.get("group"))
            branches.append(idx_dict.get("branch"))

        # Check if all partitions are the same (common case)
        unique_partitions = set(partitions)
        if len(unique_partitions) == 1:
            partition = partitions[0]
        else:
            partition = partitions  # Will be handled by indexer

        # Single batch add to indexer
        self._indexer.add_samples(
            count=n_samples,
            partition=partition,
            origin_indices=origins,
            augmentation=augmentations,
            group=groups if any(g is not None for g in groups) else None,
            branch=branches if any(b is not None for b in branches) else None
        )

        # Single batch add to feature storage
        self._block.add_samples_batch_3d(data)

    def add_features(self,
                     features: InputFeatures,
                     processings: ProcessingList,
                     source: int = -1) -> None:
        """
        Add processed feature versions to existing data.

        Use this to add features from preprocessing pipelines
        (e.g., after Savitzky-Golay filtering, MSC, derivatives).

        Args:
            features: Processed feature data matching sample count
            processings: Processing names/IDs for the new features
            source: Target source (-1 for all, or specific index)

        Examples:
            >>> # After preprocessing
            >>> from nirs4all.preprocessing import SavitzkyGolay
            >>> sg = SavitzkyGolay(window=11, polyorder=2)
            >>> X_sg = sg.fit_transform(dataset.x({}))
            >>> dataset.add_features(
            ...     X_sg, ["SavitzkyGolay_11_2"], source=0
            ... )
        """
        self._block.update_features([], features, processings, source)
        self._indexer.add_processings(processings)

    def replace_features(self,
                         source_processings: ProcessingList,
                         features: InputFeatures,
                         processings: ProcessingList,
                         source: int = -1) -> None:
        """
        Replace existing processed features with new versions.

        Args:
            source_processings: Existing processing names to replace
            features: New processed feature data
            processings: New processing names/IDs
            source: Target source (-1 for all, or specific index)

        Examples:
            >>> # Replace old preprocessing with improved version
            >>> dataset.replace_features(
            ...     ["SavitzkyGolay_5_2"],
            ...     X_new_sg,
            ...     ["SavitzkyGolay_11_2"],
            ...     source=0
            ... )
        """
        self._block.update_features(source_processings, features, processings, source)
        if source <= 0:
            self._indexer.replace_processings(source_processings, processings)

    def update_features(self,
                        source_processings: ProcessingList,
                        features: InputFeatures,
                        processings: ProcessingList,
                        source: int = -1) -> None:
        """
        Update existing processed features.

        Args:
            source_processings: Existing processing names to update
            features: Updated feature data
            processings: Processing names/IDs
            source: Target source (-1 for all, or specific index)
        """
        self._block.update_features(source_processings, features, processings, source)

    def reset_features(self,
                       features: np.ndarray,
                       processings: List[str],
                       source: int = 0) -> None:
        """
        Reset features and processings for a source.

        Replaces all features and processings with new data.
        Also updates the indexer to reflect the new processings.

        Args:
            features: New feature data (2D or 3D)
            processings: List of new processing names
            source: Target source index
        """
        # Reset features in storage
        self._block.sources[source].reset_features(features, processings)

        # Reset processings in indexer (only if source 0, as indexer tracks source 0 processings)
        if source == 0:
            self._indexer.reset_processings(processings)

    def keep_sources(self, source_indices: Union[int, List[int]]) -> None:
        """Keep only specified sources, removing all others.

        Used after merge operations with output_as="features" to consolidate
        to a single source.

        Args:
            source_indices: Single source index or list of source indices to keep.

        Raises:
            ValueError: If source indices are invalid.
        """
        self._block.keep_sources(source_indices)

    def augment_samples(self,
                        data: InputData,
                        processings: ProcessingList,
                        augmentation_id: str,
                        selector: Optional[Selector] = None,
                        count: Union[int, List[int]] = 1) -> List[int]:
        """
        Create augmented versions of existing samples.

        Augmentation creates new synthetic samples based on existing ones
        (e.g., noise addition, spectral shifts, mixup). Augmented samples
        are tracked and can be excluded during CV splitting.

        Args:
            data: Augmented feature data
            processings: Processing names for augmented data
            augmentation_id: Unique identifier for this augmentation batch
            selector: Filter for which samples to augment (None = all base samples)
            count: Number of augmented versions per sample:
                - int: Same count for all selected samples
                - List[int]: Specific count per sample

        Returns:
            List of newly created sample IDs

        Raises:
            ValueError: If selector matches no samples
            ValueError: If data shape doesn't match sample count

        Examples:
            >>> # Add noise to train samples
            >>> from nirs4all.augmentation import GaussianNoise
            >>> augmenter = GaussianNoise(std=0.01)
            >>> X_train_base = dataset.x(
            ...     {"partition": "train"},
            ...     include_augmented=False
            ... )
            >>> X_aug = augmenter.transform(X_train_base)
            >>> aug_ids = dataset.augment_samples(
            ...     X_aug,
            ...     ["raw", "GaussianNoise_0.01"],
            ...     "noise_aug_v1",
            ...     {"partition": "train"},
            ...     count=1
            ... )
        """
        # Always exclude already-augmented samples
        selector_dict = _selector_to_dict(selector)
        if selector_dict:
            sample_indices = self._indexer.x_indices(
                selector_dict, include_augmented=False
            ).tolist()
        else:
            sample_indices = self._indexer.x_indices(
                {}, include_augmented=False
            ).tolist()

        if not sample_indices:
            return []

        augmented_ids = self._indexer.augment_rows(
            sample_indices, count, augmentation_id
        )

        self._block.augment_samples(
            sample_indices, data, processings, count
        )

        return augmented_ids

    def headers(self, source: int = 0) -> List[str]:
        """
        Get feature headers (wavelengths, feature names) for a source.

        Args:
            source: Source index (default: 0)

        Returns:
            List of header strings

        Examples:
            >>> headers = dataset.headers(0)
            >>> # For spectroscopy data:
            >>> print(headers[:5])
            ['1000.0', '1001.5', '1003.0', '1004.5', '1006.0']
        """
        return self._block.headers(source)

    def header_unit(self, source: int = 0) -> str:
        """
        Get the unit type for headers.

        Args:
            source: Source index (default: 0)

        Returns:
            Unit string: "cm-1" | "nm" | "none" | "text" | "index"
        """
        return self._block.sources[source].header_unit

    def processing_names(self, source: int = 0) -> List[str]:
        """
        Get list of processing step names for a source.

        Args:
            source: Source index (default: 0)

        Returns:
            List of processing names (e.g., ["raw", "SavitzkyGolay_11_2", "MSC"])

        Examples:
            >>> processings = dataset.processing_names(0)
            >>> print(" -> ".join(processings))
            raw -> SavitzkyGolay_11_2 -> MSC -> StandardScaler
        """
        return self._block.preprocessing_str[source]

    @property
    def num_samples(self) -> int:
        """Number of samples in the dataset."""
        return self._block.num_samples

    @property
    def num_features(self) -> Union[List[int], int]:
        """Number of features per source (int if single source, list if multi)."""
        return self._block.num_features

    @property
    def num_sources(self) -> int:
        """Number of feature sources."""
        return len(self._block.sources)

    @property
    def is_multi_source(self) -> bool:
        """True if dataset has multiple feature sources."""
        return self.num_sources > 1

    # ========== Wavelength conversion methods (for spectroscopy datasets) ==========

    def wavelengths_cm1(self, source: int = 0) -> np.ndarray:
        """
        Get wavelengths in cm⁻¹ (wavenumber), converting from nm if needed.

        Note: Only applicable to spectroscopy datasets with wavelength headers.
        For non-spectroscopy datasets, returns feature indices.

        Args:
            source: Source index (default: 0)

        Returns:
            Wavelengths in cm⁻¹ as float array

        Raises:
            ValueError: If headers cannot be converted to wavelengths

        Examples:
            >>> wl_cm1 = dataset.wavelengths_cm1(0)
            >>> # For data in nm, automatically converts:
            >>> print(wl_cm1[:5])
            [10000.0, 9975.06, 9950.25, 9925.56, 9900.99]
        """
        headers = self.headers(source)
        unit = self.header_unit(source)

        if unit == "cm-1":
            return np.array([float(h) for h in headers])
        elif unit == "nm":
            nm_values = np.array([float(h) for h in headers])
            return 10_000_000.0 / nm_values
        elif unit in ["none", "index"]:
            return np.arange(len(headers), dtype=float)
        else:
            raise ValueError(
                f"Cannot convert unit '{unit}' to wavelengths (cm⁻¹). "
                f"Expected 'cm-1', 'nm', 'none', or 'index'."
            )

    def wavelengths_nm(self, source: int = 0) -> np.ndarray:
        """
        Get wavelengths in nm, converting from cm⁻¹ if needed.

        Note: Only applicable to spectroscopy datasets with wavelength headers.
        For non-spectroscopy datasets, returns feature indices.

        Args:
            source: Source index (default: 0)

        Returns:
            Wavelengths in nm as float array

        Raises:
            ValueError: If headers cannot be converted to wavelengths

        Examples:
            >>> wl_nm = dataset.wavelengths_nm(0)
            >>> # For data in cm-1, automatically converts:
            >>> print(wl_nm[:5])
            [780.0, 1000.0, 1500.0, 2000.0, 2500.0]
        """
        headers = self.headers(source)
        unit = self.header_unit(source)

        if unit == "nm":
            return np.array([float(h) for h in headers])
        elif unit == "cm-1":
            cm1_values = np.array([float(h) for h in headers])
            return 10_000_000.0 / cm1_values
        elif unit in ["none", "index"]:
            return np.arange(len(headers), dtype=float)
        else:
            raise ValueError(
                f"Cannot convert unit '{unit}' to wavelengths (nm). "
                f"Expected 'cm-1', 'nm', 'none', or 'index'."
            )

    def float_headers(self, source: int = 0) -> np.ndarray:
        """
        Get headers as float array (legacy method).

        WARNING: This method assumes headers are numeric and doesn't handle unit conversion.
        Use wavelengths_cm1() or wavelengths_nm() for wavelength data.

        Args:
            source: Source index (default: 0)

        Returns:
            Headers converted to float array

        Raises:
            ValueError: If headers cannot be converted to float
        """
        try:
            return np.array([float(header) for header in self._block.headers(source)])
        except ValueError as e:
            raise ValueError(f"Cannot convert headers to float: {e}")
