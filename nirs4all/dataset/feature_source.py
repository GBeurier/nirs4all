"""
FeatureSource class for managing and manipulating a 2D numpy array of features.
This class provides methods to add new samples, processings, and augment data,
as well as to retrieve data in various layouts (2D, 3D).
"""

import numpy as np
from typing import List, Dict, Union


class FeatureSource:
    """Wrapper for a single (rows, dims) float array."""

    def __init__(self, padding: bool = True, pad_value: float = 0.0):
        """
        Initialize a FeatureSource with a 2D numpy array.
        Args:
            array: A 2D numpy array of shape (num_samples, num_features).
            padding: Whether to pad the array if necessary.
            pad_value: Value to use for padding if padding is enabled.
        """
        self.padding = padding
        self.pad_value = pad_value
        self._array = np.empty((0, 1, 0), dtype=np.float32)  # Initialize with empty shape
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

    def add_samples(self, new_samples: np.ndarray, processing: str = "raw") -> None:
        if new_samples.ndim != 2:
            raise ValueError(f"new_samples must be a 2D array, got {new_samples.ndim} dimensions")

        if self.num_samples > 0 and new_samples.shape[1] != self.num_features:
            if new_samples.shape[1] > self.num_features:
                raise ValueError(f"new_samples must have less than {self.num_features} features, got {new_samples.shape[1]}")

            if self.padding:
                padded_samples = np.full((new_samples.shape[0], self.num_features), self.pad_value, dtype=new_samples.dtype)
                padded_samples[:, :new_samples.shape[1]] = new_samples
                new_samples = padded_samples
            else:
                raise ValueError(f"new_samples must have {self.num_features} features, got {new_samples.shape[1]}")

        processing_index = -1
        if processing not in self._processing_id_to_index:
            processing_index = self.num_processings
            self._processing_id_to_index[processing] = processing_index
            self._processing_ids.append(processing)
        else:
            processing_index = self._processing_id_to_index[processing]

        if processing_index >= self._array.shape[1]:
            if self.num_samples != new_samples.shape[0]:
                raise ValueError(f"new_samples must have {self.num_samples} samples, got {new_samples.shape[0]}")

            new_shape = (self.num_samples, processing_index + 1, self.num_features)
            new_array = np.empty(new_shape, dtype=self._array.dtype)
            new_array[:, :self._array.shape[1], :] = self._array
            new_array[:, processing_index, :] = new_samples
            self._array = new_array
        else:
            if self.num_processings > 1:
                raise ValueError(
                    f"Processing '{processing}' already exists at index {processing_index}, cannot add new samples"
                )

            new_samples_count = self.num_samples + new_samples.shape[0]
            new_num_features = self.num_features if self.num_samples > 0 else new_samples.shape[1]
            new_shape = (new_samples_count, self._array.shape[1], new_num_features)

            new_array = np.empty(new_shape, dtype=self._array.dtype)
            new_array[:self.num_samples, :self._array.shape[1], :self._array.shape[2]] = self._array
            new_array[self.num_samples:, :, :] = new_samples[:, np.newaxis, :]

            self._array = new_array

    def augment_samples(self, indices: List[int], count: Union[int, List[int]] | None = None) -> None:
        if count is None:
            count = 1

        if isinstance(count, int):
            count = [count] * len(indices)
        elif len(count) != len(indices):
            raise ValueError("count must be an int or a list with the same length as indices")

        new_shape = (
            self.num_samples + sum(count),
            self.num_processings,
            self.num_features
        )

        new_array = np.empty(new_shape, dtype=self._array.dtype)
        new_array[:self.num_samples, :, :] = self._array

        for i, (idx, c) in enumerate(zip(indices, count)):
            new_array[self.num_samples + sum(count[:i]), :, :] = self._array[idx, :, :].copy()
            for j in range(1, c):
                new_array[self.num_samples + sum(count[:i]) + j, :, :] = self._array[idx, :, :].copy()

        self._array = new_array

    def get_features(self, indices: List[int], processings: List[str], layout: str) -> np.ndarray:

        processings_indices = sorted([self._processing_id_to_index[p] for p in processings if p in self._processing_id_to_index])

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

    def update_features(self, indices: np.ndarray, processings: List[str], new_data: np.ndarray, layout: str) -> None:
        if layout == "2d":
            self._array[indices, :, :] = new_data.reshape(len(indices), -1, self.num_features)
        elif layout == "2d_interleaved":
            self._array[indices, :, :] = np.transpose(new_data.reshape(len(indices), self.num_features, -1), (0, 2, 1))
        elif layout == "3d":
            self._array[indices, :, :] = new_data
        elif layout == "3d_transpose":
            self._array[indices, :, :] = np.transpose(new_data, (0, 2, 1))
        else:
            raise ValueError(f"Unknown layout: {layout}")

    def update_processing_ids(self, old_processing: Union[List[str], str], new_processing: Union[List[str], str]) -> None:

        if isinstance(old_processing, list) and isinstance(new_processing, list):
            if len(old_processing) != len(new_processing):
                raise ValueError("old_processing and new_processing must have the same length")
            for old, new in zip(old_processing, new_processing):
                self.update_processing_ids(old, new)
            return
        elif isinstance(old_processing, str) and isinstance(new_processing, list):
            if len(new_processing) != 1:
                raise ValueError("new_processing must be a single string when old_processing is a string")
            new_processing = new_processing[0]
        elif isinstance(old_processing, list) and isinstance(new_processing, str):
            if len(old_processing) != 1:
                raise ValueError("old_processing must be a single string when new_processing is a string")
            old_processing = old_processing[0]
        elif not isinstance(old_processing, str) or not isinstance(new_processing, str):
            raise ValueError("old_processing and new_processing must be strings or lists of strings")

        if old_processing not in self._processing_id_to_index:
            raise ValueError(f"Processing ID '{old_processing}' does not exist")

        if new_processing in self._processing_id_to_index:
            raise ValueError(f"Processing ID '{new_processing}' already exists")

        idx = self._processing_id_to_index.pop(old_processing)
        self._processing_ids[idx] = new_processing
        self._processing_id_to_index[new_processing] = idx

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