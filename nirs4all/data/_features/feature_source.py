"""Refactored FeatureSource using component-based architecture."""

import numpy as np
from typing import List, Optional

from nirs4all.data.types import InputFeatures, ProcessingList, SampleIndices
from nirs4all.data._features import (
    ArrayStorage,
    ProcessingManager,
    HeaderManager,
    LayoutTransformer,
    UpdateStrategy,
    AugmentationHandler,
    LayoutType,
)


class FeatureSource:
    """Manages a 3D numpy array of features using modular components.

    This class provides efficient storage and manipulation of feature data with multiple
    processing stages. Each sample can have multiple processing versions (e.g., raw, normalized,
    filtered), all stored in a single aligned 3D array.

    The implementation uses a component-based architecture for better modularity:
    - ArrayStorage: Manages the 3D numpy array
    - ProcessingManager: Tracks processing IDs and their indices
    - HeaderManager: Manages feature headers and units
    - LayoutTransformer: Transforms arrays to different layouts
    - UpdateStrategy: Handles update operation logic
    - AugmentationHandler: Manages sample augmentation

    Attributes:
        padding: Whether to allow padding when adding features with fewer dimensions.
        pad_value: Value to use for padding (default: 0.0).
    """

    def __init__(self, padding: bool = True, pad_value: float = 0.0):
        """Initialize an empty FeatureSource.

        Args:
            padding: If True, allow padding features to match existing dimensions.
            pad_value: Value to use for padding missing features.
        """
        self._storage = ArrayStorage(padding=padding, pad_value=pad_value)
        self._processing_mgr = ProcessingManager()
        self._header_mgr = HeaderManager()
        self._layout_transformer = LayoutTransformer()
        self._update_strategy = UpdateStrategy()
        self._augmentation_handler = AugmentationHandler()

    def __repr__(self):
        return (
            f"FeatureSource(shape={self._storage.shape}, "
            f"dtype={self._storage.dtype}, "
            f"processing_ids={self._processing_mgr.processing_ids})"
        )

    def __str__(self) -> str:
        array = self._storage.array
        if array.size > 0:
            mean_value = round(float(np.mean(array)), 3)
            variance_value = round(float(np.var(array)), 3)
            min_value = round(float(np.min(array)), 3)
            max_value = round(float(np.max(array)), 3)
        else:
            mean_value = variance_value = min_value = max_value = 0.0

        return (
            f"{self._storage.shape}, "
            f"processings={self._processing_mgr.processing_ids}, "
            f"min={min_value}, max={max_value}, "
            f"mean={mean_value}, var={variance_value}"
        )

    @property
    def headers(self) -> Optional[List[str]]:
        """Get the feature headers.

        Returns:
            List of header strings, or None if not set.
        """
        return self._header_mgr.headers

    @property
    def header_unit(self) -> str:
        """Get the unit type of the headers.

        Returns:
            Unit type string ("cm-1", "nm", "none", "text", "index").
        """
        return self._header_mgr.header_unit

    @property
    def num_samples(self) -> int:
        """Get the number of samples.

        Returns:
            Number of samples (first dimension of array).
        """
        return self._storage.num_samples

    @property
    def num_processings(self) -> int:
        """Get the number of processing stages.

        Returns:
            Number of unique processings (second dimension of array).
        """
        return self._processing_mgr.num_processings

    @property
    def num_features(self) -> int:
        """Get the number of features per processing.

        Returns:
            Number of features (third dimension of array).
        """
        return self._storage.num_features

    @property
    def num_2d_features(self) -> int:
        """Get total features when flattened to 2D.

        Returns:
            Product of processings and features dimensions.
        """
        return self._storage.num_processings * self._storage.num_features

    @property
    def processing_ids(self) -> List[str]:
        """Get a copy of the processing ID list.

        Returns:
            List of processing identifiers.
        """
        return self._processing_mgr.processing_ids

    def add_samples(
        self,
        new_samples: np.ndarray,
        headers: Optional[List[str]] = None
    ) -> None:
        """Add new samples to the feature source.

        Only allowed when there's only one processing (raw). Samples are added as
        a new row in the array with a single processing dimension.

        Args:
            new_samples: 2D array of shape (n_samples, n_features).
            headers: Optional list of feature header names.

        Raises:
            ValueError: If the dataset already has multiple processings, or if
                new_samples is not 2D.
        """
        if self.num_processings > 1:
            raise ValueError(
                "Cannot add new samples to a dataset that already has been processed."
            )

        if new_samples.ndim != 2:
            raise ValueError(
                f"new_samples must be a 2D array, got {new_samples.ndim} dimensions"
            )

        self._storage.add_samples(new_samples)

        # Only update headers if provided, and preserve existing unit
        if headers is not None:
            current_unit = self._header_mgr.header_unit
            self._header_mgr.set_headers(headers, unit=current_unit)

    def add_samples_batch_3d(self, data: np.ndarray) -> None:
        """Add multiple samples with 3D data in a single operation - O(N) instead of O(N²).

        This method is optimized for bulk insertion of augmented samples where
        each sample may have multiple processings.

        Args:
            data: 3D array of shape (n_samples, n_processings, n_features).

        Raises:
            ValueError: If data dimensions don't match existing processings/features.
        """
        if data.ndim != 3:
            raise ValueError(f"data must be a 3D array, got {data.ndim} dimensions")

        self._storage.add_samples_batch(data)

    def set_headers(self, headers: Optional[List[str]], unit: str = "cm-1") -> None:
        """Set feature headers with unit metadata.

        Args:
            headers: List of header strings (wavelengths, feature names, etc.).
            unit: Unit type - "cm-1" (wavenumber), "nm" (wavelength),
                  "none", "text", "index".
        """
        self._header_mgr.set_headers(headers, unit=unit)

    def update_features(
        self,
        source_processings: ProcessingList,
        features: InputFeatures,
        processings: ProcessingList
    ) -> None:
        """Add new features or replace existing ones.

        Args:
            source_processings: List of existing processing names to replace.
                Empty string "" means add new.
            features: List of feature arrays, each of shape (n_samples, n_features),
                or single array.
            processings: List of target processing names for the data.

        Example:
            # Add new 'savgol' and 'detrend', replace 'raw' with 'msc'
            update_features(["", "raw", ""],
                           [savgol_data, msc_data, detrend_data],
                           ["savgol", "msc", "detrend"])
        """
        # Normalize features to list of arrays
        feature_list = self._normalize_features_input(features)
        if not feature_list:
            return

        # Categorize operations
        replacements, additions = self._update_strategy.categorize_operations(
            feature_list,
            source_processings,
            processings,
            self._processing_mgr._processing_id_to_index
        )

        # Check if we should resize features
        should_resize, new_num_features = self._update_strategy.should_resize_features(
            replacements,
            additions,
            self._storage.num_features
        )

        if should_resize:
            self._storage.resize_features(new_num_features)
            self._header_mgr.clear_headers()

        # Apply operations
        self._apply_replacements(replacements)
        self._apply_additions(additions)

    def reset_features(
        self,
        features: np.ndarray,
        processings: List[str]
    ) -> None:
        """Reset features and processings.

        Replaces all features and processings with new data.

        Args:
            features: New feature data (2D or 3D).
            processings: List of new processing names.
        """
        # Reset storage
        self._storage.reset_data(features)

        # Reset processing manager
        self._processing_mgr.reset_processings(processings)

        # Clear headers as dimensions likely changed
        self._header_mgr.clear_headers()

    def _normalize_features_input(self, features: InputFeatures) -> List[np.ndarray]:
        """Normalize various feature input formats to list of arrays.

        Args:
            features: Input features in various formats.

        Returns:
            List of numpy arrays.
        """
        if isinstance(features, np.ndarray):
            return [features]

        if isinstance(features, list):
            if not features:
                return []

            # Check if it's list of lists (multi-source case)
            if isinstance(features[0], list):
                return list(features[0])  # Take first source

            # Check if it's list of arrays
            if isinstance(features[0], np.ndarray):
                return list(features)

        return []

    def _apply_replacements(self, replacements: List) -> None:
        """Apply replacement operations.

        Args:
            replacements: List of ReplacementOperation objects.
        """
        for replacement in replacements:
            self._storage.update_processing(replacement.proc_idx, replacement.new_data)

            # Update processing name if different
            old_name = self._processing_mgr.processing_ids[replacement.proc_idx]
            if replacement.new_proc_name != old_name:
                self._processing_mgr.rename_processing(
                    old_name,
                    replacement.new_proc_name
                )

    def _apply_additions(self, additions: List) -> None:
        """Apply addition operations.

        Args:
            additions: List of AdditionOperation objects.
        """
        for addition in additions:
            self._storage.add_processing(addition.new_data)
            self._processing_mgr.add_processing(addition.new_proc_name)

    def augment_samples(
        self,
        sample_indices: List[int],
        data: np.ndarray,
        processings: List[str],
        count_list: List[int]
    ) -> None:
        """Create augmented samples by duplicating existing samples.

        Args:
            sample_indices: List of sample indices to augment.
            data: Augmented feature data of shape (total_augmented_samples, n_features).
            processings: Processing names for the augmented data.
            count_list: Number of augmentations per sample.
        """
        # Validate inputs
        total_augmentations = self._augmentation_handler.validate_augmentation_inputs(
            sample_indices,
            data,
            count_list,
            self.num_samples
        )

        if total_augmentations == 0:
            return

        # Normalize processings
        proc_list = self._augmentation_handler.normalize_processings(processings)

        # Augment samples in storage (duplicates existing samples)
        self._storage.augment_samples(sample_indices, count_list, new_proc_data=None)

        # Add new processings for augmented samples
        for proc_name in proc_list:
            if not self._processing_mgr.has_processing(proc_name):
                self._add_new_processing_for_augmentation(
                    proc_name,
                    data,
                    total_augmentations
                )

    def _add_new_processing_for_augmentation(
        self,
        proc_name: str,
        data: np.ndarray,
        total_augmentations: int
    ) -> None:
        """Add a new processing for augmented samples only.

        Args:
            proc_name: Name for the new processing.
            data: Processing data for augmented samples.
            total_augmentations: Number of augmented samples.
        """
        # Add processing to storage (expands array and adds data for augmented samples)
        self._storage._add_processing_for_augmented(data, total_augmentations)

        # Register the new processing
        self._processing_mgr.add_processing(proc_name)

    def x(self, indices: SampleIndices, layout: str) -> np.ndarray:
        """Retrieve feature data in specified layout.

        Args:
            indices: Sample indices to retrieve.
            layout: Output format:
                - "2d": Flatten to (samples, processings * features)
                - "2d_interleaved": Transpose then flatten to (samples, features * processings)
                - "3d": Keep as (samples, processings, features)
                - "3d_transpose": Transpose to (samples, features, processings)

        Returns:
            Feature array in requested layout.

        Raises:
            ValueError: If layout is unknown.
        """
        if len(indices) == 0:
            return self._layout_transformer.get_empty_array(
                layout,
                self.num_processings,
                self.num_features,
                self._storage.dtype
            )

        # Get data from storage
        selected_data = self._storage.get_data(np.array(indices))

        # Transform to requested layout
        return self._layout_transformer.transform(
            selected_data,
            layout,
            self.num_processings,
            self.num_features
        )
