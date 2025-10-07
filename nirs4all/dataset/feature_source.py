"""
FeatureSource class for managing and manipulating a 2D numpy array of features.
This class provides methods to add new samples, processings, and augment data,
as well as to retrieve data in various layouts (2D, 3D).
"""

import numpy as np
from typing import List, Dict, Optional
from nirs4all.dataset.helpers import InputData, InputFeatures, ProcessingList, SampleIndices


class FeatureSource:

    def __init__(self, padding: bool = True, pad_value: float = 0.0):
        self.padding = padding
        self.pad_value = pad_value
        self._array = np.empty((0, 1, 0), dtype=np.float32)  # Initialize with empty shape (samples, processings, features)
        self._processing_ids: List[str] = ["raw"]  # Default processing ID
        self._processing_id_to_index: Dict[str, int] = {"raw": 0}  # Maps processing ID to index
        self._headers: Optional[List[str]] = None  # Optional feature headers

    def __repr__(self):
        return f"FeatureSource(shape={self._array.shape}, dtype={self._array.dtype}, processing_ids={self._processing_ids})"

    def __str__(self) -> str:
        mean_value = round(float(np.mean(self._array)), 3) if self._array.size > 0 else 0.0
        variance_value = round(float(np.var(self._array)), 3) if self._array.size > 0 else 0.0
        min_value = round(float(np.min(self._array)), 3) if self._array.size > 0 else 0.0
        max_value = round(float(np.max(self._array)), 3) if self._array.size > 0 else 0.0
        return f"{self._array.shape}, processings={self._processing_ids}, min={min_value}, max={max_value}, mean={mean_value}, var={variance_value})"

    @property
    def headers(self) -> Optional[List[str]]:
        return self._headers

    @property
    def num_samples(self) -> int:
        return self._array.shape[0]

    @property
    def num_processings(self) -> int:
        return len(self._processing_ids)

    @property
    def num_features(self) -> int:
        return self._array.shape[2]

    @property
    def num_2d_features(self) -> int:
        return self._array.shape[1] * self._array.shape[2]

    @property
    def processing_ids(self) -> List[str]:
        return self._processing_ids.copy()

    def add_samples(self, new_samples: np.ndarray, headers: Optional[List[str]] = None) -> None:
        if self.num_processings > 1:
            raise ValueError("Cannot add new samples to a dataset that already has been processed.")

        if new_samples.ndim != 2:
            raise ValueError(f"new_samples must be a 2D array, got {new_samples.ndim} dimensions")

        X = np.asarray(new_samples, dtype=self._array.dtype)

        # If this is the first data being added
        if self.num_samples == 0:
            self._array = X[:, None, :]
        else:
            prepared_data = self._prepare_data_for_storage(X)
            new_data_3d = prepared_data[:, None, :]
            self._array = np.concatenate((self._array, new_data_3d), axis=0)

        self._headers = headers

    def set_headers(self, headers: List[str]) -> None:
        """Set feature headers (wavelengths)."""
        self._headers = headers

    def update_features(self, source_processings: ProcessingList, features: InputFeatures, processings: ProcessingList) -> None:
        """
        Add new features or replace existing ones based on source_processings and processings.

        Args:
            source_processings: List of existing processing names to replace. Empty string "" means add new.
            data: List of feature arrays, each of shape (n_samples, n_features), or single array
            processings: List of target processing names for the data

        Example:
            # Add new 'savgol' and 'detrend', replace 'raw' with 'msc'
            update_features(["", "raw", ""],
                           [savgol_data, msc_data, detrend_data],
                           ["savgol", "msc", "detrend"])
        """
        # self._validate_update_inputs(features, source_processings, processings)
        # Normalize features to list of arrays
        feature_list: List[np.ndarray] = []
        if isinstance(features, np.ndarray):
            feature_list = [features]
        elif isinstance(features, list):
            if not features:
                return
            # Check if it's list of lists or list of arrays
            if isinstance(features[0], list):
                # Handle list of lists (multi-source case) - take first source
                feature_list = list(features[0])  # type: ignore
            elif isinstance(features[0], np.ndarray):
                feature_list = list(features)  # type: ignore
            else:
                return
        else:
            return

        # Separate operations: replacements and additions
        replacements, additions = self._categorize_operations(feature_list, source_processings, processings)

        # Check if all processings are being replaced with same new feature dimension
        if replacements and not additions:
            # Get new feature dimensions
            new_feature_dims = [new_data.shape[1] for _, new_data, _ in replacements]
            if len(set(new_feature_dims)) == 1 and new_feature_dims[0] != self.num_features:
                # All replacements have same new dimension - resize array
                self._resize_features(new_feature_dims[0])

        # Apply replacements first, then additions
        self._apply_replacements(replacements)
        self._apply_additions(additions)

    def _resize_features(self, new_num_features: int) -> None:
        """Resize the feature dimension of the array."""
        if self.num_samples == 0:
            return

        # Create new array with new feature dimension
        new_shape = (self.num_samples, self.num_processings, new_num_features)
        new_array = np.zeros(new_shape, dtype=self._array.dtype)

        # Copy existing data (will be replaced anyway, but initialize properly)
        min_features = min(self.num_features, new_num_features)
        new_array[:, :, :min_features] = self._array[:, :, :min_features]

        self._array = new_array

        # Clear headers since they won't match the new feature dimension
        # Headers will be set by the controller after replacement
        self._headers = None

    # def _validate_update_inputs(self, features: List[np.ndarray], source_processings: List[str], processings: List[str]) -> None:
    #     """Validate inputs for update_features."""
    #     if len(features) != len(source_processings) or len(features) != len(processings):
    #         raise ValueError("features, source_processings, and processings must have the same length")

    #     # Validate that all arrays have the same number of samples
    #     if self.num_samples > 0:
    #         for i, arr in enumerate(features):
    #             if arr.shape[0] != self.num_samples:
    #                 raise ValueError(f"Array {i} has {arr.shape[0]} samples, expected {self.num_samples}")

    def _categorize_operations(self, features: List[np.ndarray], source_processings: List[str], processings: List[str]):
        """Separate operations into replacements and additions."""
        replacements = []  # (processing_idx, new_data, new_processing_name)
        additions = []     # (new_data, new_processing_name)
        if len(source_processings) == 0:
            source_processings = [""] * len(processings)

        for arr, source_proc, target_proc in zip(features, source_processings, processings):
            if source_proc == "":
                # Add new processing
                if target_proc in self._processing_id_to_index:
                    raise ValueError(f"Processing '{target_proc}' already exists, cannot add")
                additions.append((arr, target_proc))
            else:
                # Replace existing processing
                if source_proc not in self._processing_id_to_index:
                    raise ValueError(f"Source processing '{source_proc}' does not exist")
                if target_proc != source_proc and target_proc in self._processing_id_to_index:
                    raise ValueError(f"Target processing '{target_proc}' already exists")

                source_idx = self._processing_id_to_index[source_proc]
                replacements.append((source_idx, arr, target_proc))

        return replacements, additions

    def _apply_replacements(self, replacements) -> None:
        """Apply replacement operations."""
        for proc_idx, new_data, new_proc_name in replacements:
            # Handle padding and feature dimension matching
            new_data = self._prepare_data_for_storage(new_data)

            # Update the array data
            self._array[:, proc_idx, :] = new_data

            # Update processing name if different
            if new_proc_name != self._processing_ids[proc_idx]:
                old_proc_name = self._processing_ids[proc_idx]
                self._processing_ids[proc_idx] = new_proc_name
                del self._processing_id_to_index[old_proc_name]
                self._processing_id_to_index[new_proc_name] = proc_idx

    def _apply_additions(self, additions) -> None:
        """Apply addition operations."""
        if not additions:
            return

        addition_data = []
        addition_names = []

        for new_data, new_proc_name in additions:
            # Handle padding and feature dimension matching
            new_data = self._prepare_data_for_storage(new_data)
            addition_data.append(new_data[:, None, :])
            addition_names.append(new_proc_name)

        # Concatenate new processings to existing array
        new_data_array = np.concatenate(addition_data, axis=1)

        if self.num_samples == 0:
            self._array = new_data_array
        else:
            self._array = np.concatenate((self._array, new_data_array), axis=1)

        # Update processing IDs and mapping
        start_idx = len(self._processing_ids)
        for i, proc_name in enumerate(addition_names):
            self._processing_ids.append(proc_name)
            self._processing_id_to_index[proc_name] = start_idx + i

    def _prepare_data_for_storage(self, new_data: np.ndarray) -> np.ndarray:
        """Prepare data for storage by handling padding and dimension matching."""
        if self.num_samples == 0:
            # First data being added - no preparation needed
            return new_data

        # Handle padding and feature dimension matching for existing data
        if self.padding and new_data.shape[1] < self.num_features:
            padded_data = np.full((new_data.shape[0], self.num_features), self.pad_value, dtype=new_data.dtype)
            padded_data[:, :new_data.shape[1]] = new_data
            return padded_data
        elif not self.padding and new_data.shape[1] != self.num_features:
            raise ValueError(f"Feature dimension mismatch: expected {self.num_features}, got {new_data.shape[1]}")

        return new_data

    def augment_samples(self,
                        sample_indices: List[int],
                        data: np.ndarray,
                        processings: List[str],
                        count_list: List[int]) -> None:
        """
        Create augmented samples by duplicating existing samples and adding new processing data.

        Args:
            sample_indices: List of sample indices to augment
            data: Augmented feature data of shape (total_augmented_samples, n_features)
            processings: Processing names for the augmented data
            count_list: Number of augmentations per sample (list with same length as sample_indices)
        """
        if not sample_indices:
            return

        total_augmentations = sum(count_list)
        if total_augmentations == 0:
            return

        # Validate input data shape
        if data.ndim != 2:
            raise ValueError(f"data must be a 2D array, got {data.ndim} dimensions")

        if data.shape[0] != total_augmentations:
            raise ValueError(f"data must have {total_augmentations} samples, got {data.shape[0]}")

        # Handle processings
        if isinstance(processings, str):
            processings = [processings]

        # Prepare the new processing data
        prep_data = self._prepare_data_for_storage(data)

        # First, expand the array to accommodate new samples
        new_num_samples = self.num_samples + total_augmentations
        # Keep the current number of processings for now, we'll add new ones separately
        current_processings = self._array.shape[1]  # Actual current processings
        new_shape = (new_num_samples, current_processings, self.num_features)

        # Create expanded array and copy existing data
        expanded_array = np.full(new_shape, self.pad_value, dtype=self._array.dtype)
        expanded_array[:self.num_samples, :current_processings, :] = self._array

        # Add augmented samples (copy from original samples first)
        sample_idx = 0
        for orig_idx, aug_count in zip(sample_indices, count_list):
            for _ in range(aug_count):
                # Copy all existing processings from the original sample
                expanded_array[self.num_samples + sample_idx, :current_processings, :] = self._array[orig_idx, :current_processings, :]
                sample_idx += 1

        # Update the array
        self._array = expanded_array

        # Now add the new processing(s)
        for proc_name in processings:
            if proc_name not in self._processing_id_to_index:
                # Add new processing dimension
                self._add_new_processing(proc_name, prep_data, total_augmentations)

    def _add_new_processing(self, proc_name: str, data: np.ndarray, total_augmentations: int) -> None:
        """Helper method to add a new processing to augmented samples."""
        # Get current processing count before adding new one
        current_processings = self._array.shape[1]

        # Expand array to include new processing
        new_shape = (self.num_samples, current_processings + 1, self.num_features)
        expanded_array = np.full(new_shape, self.pad_value, dtype=self._array.dtype)

        # Copy existing data
        expanded_array[:, :current_processings, :] = self._array

        # Add new processing data only to the last augmented samples
        augmented_start_idx = self.num_samples - total_augmentations

        for i in range(total_augmentations):
            augmented_sample_idx = augmented_start_idx + i
            expanded_array[augmented_sample_idx, current_processings, :] = data[i, :]

        # Update processing metadata
        self._processing_ids.append(proc_name)
        self._processing_id_to_index[proc_name] = current_processings

        # Update array
        self._array = expanded_array

    def x(self, indices: SampleIndices, layout: str) -> np.ndarray:
        if len(indices) == 0:
            return np.empty((0, self.num_2d_features), dtype=self._array.dtype) if layout in ["2d", "2d_interleaved"] else np.empty((0, self.num_processings, self.num_features), dtype=self._array.dtype)
        processings_indices = list(range(self.num_processings))
        selected_data = self._array[indices, :, :]
        selected_data = selected_data[:, processings_indices, :]

        if layout == "2d":
            selected_data = selected_data.reshape(len(indices), -1)
        elif layout == "2d_interleaved":
            selected_data = np.transpose(selected_data, (0, 2, 1)).reshape(len(indices), -1)
        elif layout == "3d":
            pass
        elif layout == "3d_transpose":
            selected_data = np.transpose(selected_data, (0, 2, 1))
        else:
            raise ValueError(f"Unknown layout: {layout}")

        return selected_data
