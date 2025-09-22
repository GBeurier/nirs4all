"""
FeatureSource class for managing and manipulating a 2D numpy array of features.
This class provides methods to add new samples, processings, and augment data,
as well as to retrieve data in various layouts (2D, 3D).
"""

import numpy as np
from typing import List, Dict, Optional
from nirs4all.dataset.helpers import InputData, InputFeatures, ProcessingList


class FeatureSource:

    def __init__(self, padding: bool = True, pad_value: float = 0.0):
        self.padding = padding
        self.pad_value = pad_value
        self._array = np.empty((0, 1, 0), dtype=np.float32)  # Initialize with empty shape (samples, processings, features)
        self._processing_ids: List[str] = ["raw"]  # Default processing ID
        self._processing_id_to_index: Dict[str, int] = {"raw": 0}  # Maps processing ID to index

    def __repr__(self):
        return f"FeatureSource(shape={self._array.shape}, dtype={self._array.dtype}, processing_ids={self._processing_ids})"

    def __str__(self) -> str:
        mean_value = np.mean(self._array) if self._array.size > 0 else 0.0
        variance_value = np.var(self._array) if self._array.size > 0 else 0.0
        return f"FeatureSource(shape={self._array.shape}, dtype={self._array.dtype}, processing_ids={self._processing_ids}, mean={mean_value}, variance={variance_value})"

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

    def add_samples(self, new_samples: np.ndarray) -> None:
        if self.num_samples > 0:
            raise ValueError("Cannot add samples to a non-empty FeatureSource. Use replace_features or augment_samples instead.")

        if new_samples.ndim != 2:
            raise ValueError(f"new_samples must be a 2D array, got {new_samples.ndim} dimensions")

        X = np.asarray(new_samples, dtype=self._array.dtype)
        self._array = X[:, None, :]

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
        self._validate_update_inputs(features, source_processings, processings)

        # Separate operations: replacements and additions
        replacements, additions = self._categorize_operations(features, source_processings, processings)

        # Apply replacements first, then additions
        self._apply_replacements(replacements)
        self._apply_additions(additions)

    def _validate_update_inputs(self, features: List[np.ndarray], source_processings: List[str], processings: List[str]) -> None:
        """Validate inputs for update_features."""
        if len(features) != len(source_processings) or len(features) != len(processings):
            raise ValueError("features, source_processings, and processings must have the same length")

        # Validate that all arrays have the same number of samples
        if self.num_samples > 0:
            for i, arr in enumerate(features):
                if arr.shape[0] != self.num_samples:
                    raise ValueError(f"Array {i} has {arr.shape[0]} samples, expected {self.num_samples}")

    def _categorize_operations(self, features: List[np.ndarray], source_processings: List[str], processings: List[str]):
        """Separate operations into replacements and additions."""
        replacements = []  # (processing_idx, new_data, new_processing_name)
        additions = []     # (new_data, new_processing_name)

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

    # def augment_samples(self, indices: List[int], count: Union[int, List[int]] | None = None) -> None:
    #     if count is None:
    #         count = 1

    #     if isinstance(count, int):
    #         count = [count] * len(indices)
    #     elif len(count) != len(indices):
    #         raise ValueError("count must be an int or a list with the same length as indices")

    #     new_shape = (
    #         self.num_samples + sum(count),
    #         self.num_processings,
    #         self.num_features
    #     )

    #     new_array = np.empty(new_shape, dtype=self._array.dtype)
    #     new_array[:self.num_samples, :, :] = self._array

    #     for i, (idx, c) in enumerate(zip(indices, count)):
    #         new_array[self.num_samples + sum(count[:i]), :, :] = self._array[idx, :, :].copy()
    #         for j in range(1, c):
    #             new_array[self.num_samples + sum(count[:i]) + j, :, :] = self._array[idx, :, :].copy()

    #     self._array = new_array

    # def get_features(self, indices: List[int], processings: List[str], layout: str) -> np.ndarray:

    #     processings_indices = sorted([self._processing_id_to_index[p] for p in processings if p in self._processing_id_to_index])

    #     selected_data = self._array[indices, :, :]
    #     selected_data = selected_data[:, processings_indices, :]
    #     if layout == "2d":
    #         selected_data = selected_data.reshape(len(indices), -1)
    #     elif layout == "2d_interleaved":
    #         selected_data = np.transpose(selected_data, (0, 2, 1)).reshape(len(indices), -1)
    #     elif layout == "3d":
    #         pass
    #     elif layout == "3d_transpose":
    #         selected_data = np.transpose(selected_data, (0, 2, 1))
    #     else:
    #         raise ValueError(f"Unknown layout: {layout}")
    #     return selected_data

    # def update_features(self, indices: np.ndarray, processings: List[str], new_data: np.ndarray, layout: str) -> None:
    #     processings_indices = sorted([self._processing_id_to_index[p] for p in processings if p in self._processing_id_to_index])
    #     print(f"Updating features at indices {indices} for processings {processings} with layout {layout}")
    #     print(f"Add new_data shape: {new_data.shape}, expected shape: ({len(indices)}, {len(processings_indices)}, {self.num_features})")
    #     if layout == "2d":
    #         new_data = new_data.reshape(len(indices), len(processings_indices), self.num_features)
    #         for i, p in enumerate(processings_indices):
    #             print(f"Updating processing {p}")
    #             self._array[indices, p, :] = new_data[:, i, :]
    #     elif layout == "2d_interleaved":
    #         new_data = new_data.reshape(len(indices), len(processings_indices), self.num_features).transpose(0, 2, 1)
    #         for i, p in enumerate(processings_indices):
    #             print(f"Updating processing {p}")
    #             self._array[indices, p, :] = new_data[:, i, :]
    #     elif layout == "3d":
    #         for i, p in enumerate(processings_indices):
    #             self._array[indices, p, :] = new_data[:, i, :]
    #     elif layout == "3d_transpose":
    #         new_data = np.transpose(new_data, (0, 2, 1))  # Ensure new_data is in the correct shape
    #         for i, p in enumerate(processings_indices):
    #             self._array[indices, processings_indices, :] = new_data[:, i, :]
    #     else:
    #         raise ValueError(f"Unknown layout: {layout}")

    # def update_processing_ids(self, old_processing: Union[List[str], str], new_processing: Union[List[str], str]) -> None:

    #     if isinstance(old_processing, list) and isinstance(new_processing, list):
    #         if len(old_processing) != len(new_processing):
    #             raise ValueError("old_processing and new_processing must have the same length")
    #         for old, new in zip(old_processing, new_processing):
    #             self.update_processing_ids(old, new)
    #         return
    #     elif isinstance(old_processing, str) and isinstance(new_processing, list):
    #         if len(new_processing) != 1:
    #             raise ValueError("new_processing must be a single string when old_processing is a string")
    #         new_processing = new_processing[0]
    #     elif isinstance(old_processing, list) and isinstance(new_processing, str):
    #         if len(old_processing) != 1:
    #             raise ValueError("old_processing must be a single string when new_processing is a string")
    #         old_processing = old_processing[0]
    #     elif not isinstance(old_processing, str) or not isinstance(new_processing, str):
    #         raise ValueError("old_processing and new_processing must be strings or lists of strings")

    #     if old_processing not in self._processing_id_to_index:
    #         raise ValueError(f"Processing ID '{old_processing}' does not exist")

    #     if new_processing in self._processing_id_to_index:
    #         raise ValueError(f"Processing ID '{new_processing}' already exists")

    #     idx = self._processing_id_to_index.pop(old_processing)
    #     self._processing_ids[idx] = new_processing
    #     self._processing_id_to_index[new_processing] = idx

    # def update_processing(self, processing_id: str | int, new_data: np.ndarray, new_processing: str) -> None:
    #     """
    #     Update the data for a specific processing ID.

    #     Args:
    #         processing_id: The processing ID to update.
    #         new_data: A 2D numpy array of shape (num_samples, num_features).
    #         new_processing: The new processing ID to assign to the updated data.
    #     Raises:
    #         ValueError: If the processing ID does not exist or if new_data has incorrect shape.
    #     """
    #     if new_processing is None or not isinstance(new_processing, str):
    #         raise ValueError("new_processing must be a non-empty string")

    #     if new_processing in self._processing_id_to_index:
    #         raise ValueError(f"Processing ID '{new_processing}' already exists") # TODO here warning and return current

    #     if processing_id not in self._processing_id_to_index:
    #         raise ValueError(f"Processing ID '{processing_id}' does not exist")

    #     if new_data.ndim != 2 or new_data.shape[1] != self.num_features:
    #         raise ValueError(f"new_data must be a 2D array with {self.num_features} features")

    #     if isinstance(processing_id, str):
    #         idx = self._processing_id_to_index[processing_id]
    #     else:
    #         idx = processing_id

    #     if self.padding:
    #         if new_data.shape[0] < self.num_samples:
    #             padded_data = np.full((self.num_samples, self.num_features), self.pad_value, dtype=new_data.dtype)
    #             padded_data[:new_data.shape[0], :] = new_data
    #             new_data = padded_data
    #         elif new_data.shape[0] > self.num_samples:
    #             raise ValueError("new_data has more samples than the source, padding is not allowed in this case")
    #     else:
    #         if new_data.shape[0] != self.num_samples:
    #             raise ValueError(f"new_data must have {self.num_samples} samples when padding is False")

    #     if new_processing not in self._processing_id_to_index:
    #         self.add_processings(new_processing)

    #     self._array[:, idx, :] = new_data
