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

        # self._array = array.astype(np.float32)
        # self._array = self._array[:, np.newaxis, :]
        # self._processing_ids: List[str] = ["raw"]  # Default processing ID
        # self._processing_id_to_index: Dict[str, int] = {"raw": 0}  # Maps processing ID to index
        # self.padding = padding
        # self.pad_value = pad_value

    def __repr__(self):
        return f"FeatureSource(shape={self._array.shape}, dtype={self._array.dtype}, processing_ids={self._processing_ids})"

    def __str__(self) -> str:
        mean_value = np.mean(self._array) if self._array.size > 0 else 0.0
        variance_value = np.var(self._array) if self._array.size > 0 else 0.0
        return f"FeatureSource(shape={self._array.shape}, dtype={self._array.dtype}, processing_ids={self._processing_ids}, mean={mean_value}, variance={variance_value})"

    @property
    def num_samples(self) -> int:
        """Get the number of samples (rows) in the source."""
        return self._array.shape[0]

    @property
    def num_processings(self) -> int:
        """Get the number of processing IDs in the source."""
        return len(self._processing_ids)

    @property
    def num_features(self) -> int:
        """Get the number of features (dimensions) in the source."""
        return self._array.shape[2]

    @property
    def num_2d_features(self) -> int:
        """Get the number of 2D features (flattened dimensions) in the source."""
        return self._array.shape[1] * self._array.shape[2]

    def add_samples(self, new_samples: np.ndarray, processing: str = "raw") -> None:
        """
        Add new samples to the source.
        Args:
            new_samples: A 2D numpy array of shape (num_new_samples, num_features).
        Raises:
            ValueError: If new_samples is not a 2D array with the correct number of features.
        """
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




    def augment_samples(self, count: Union[int, List[int]], indices: List[int] | None = None) -> None:
        """
        Augment samples by repeating specified indices.

        Args:
            count: Number of times to repeat each index, or a list specifying counts for each index.
            indices: List of indices to repeat.
        """

        if indices is None:
            indices = list(range(self.num_samples))

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
            new_array[self.num_samples + sum(count[:i]), :, :] = self._array[idx, :, :]
            for j in range(1, c):
                new_array[self.num_samples + sum(count[:i]) + j, :, :] = self._array[idx, :, :]

        self._array = new_array

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

    def layout_2d(self, row_indices: np.ndarray) -> np.ndarray:
        selected_data = self._array[row_indices, :, :].reshape(len(row_indices), -1)
        return selected_data

    def layout_2d_interleaved(self, row_indices: np.ndarray) -> np.ndarray:
        selected_data = self._array[row_indices, :, :]  # Shape: (len(row_indices), num_processings, num_features)
        interleaved = np.transpose(selected_data, (0, 2, 1)).reshape(len(row_indices), -1)
        return interleaved

    def layout_3d(self, row_indices: np.ndarray) -> np.ndarray:
        selected_data = self._array[row_indices, :, :]
        return selected_data

    def layout_3d_transpose(self, row_indices: np.ndarray) -> np.ndarray:
        selected_data = self._array[row_indices, :, :]
        return np.transpose(selected_data, (0, 2, 1))

    def update_features(self, indices: np.ndarray, new_data: np.ndarray, layout: str) -> None:
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