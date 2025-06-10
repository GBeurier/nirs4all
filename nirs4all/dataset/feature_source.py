"""
FeatureSource class for managing and manipulating a 2D numpy array of features.
This class provides methods to add new samples, processings, and augment data,
as well as to retrieve data in various layouts (2D, 3D).
"""

import numpy as np
from typing import List, Dict, Union


class FeatureSource:
    """Wrapper for a single (rows, dims) float array."""

    def __init__(self, array: np.ndarray, padding: bool = True, pad_value: float = 0.0):
        """
        Initialize a FeatureSource with a 2D numpy array.
        Args:
            array: A 2D numpy array of shape (num_samples, num_features).
            padding: Whether to pad the array if necessary.
            pad_value: Value to use for padding if padding is enabled.
        """
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.floating):
            raise ValueError("FeatureSource requires a 2D numpy array of floats")

        self._array = array.astype(np.float32)
        self._array = self._array[:, np.newaxis, :]
        self._processing_ids: List[str] = ["raw"]  # Default processing ID
        self._processing_id_to_index: Dict[str, int] = {"raw": 0}  # Maps processing ID to index
        self.padding = padding
        self.pad_value = pad_value

    def __repr__(self):
        return f"FeatureSource(shape={self._array.shape}, dtype={self._array.dtype}, processing_ids={self._processing_ids})"

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

    def add_processings(self, processing_id: Union[str, List[str]]) -> None:
        """Add a new processing ID to the source.

        Args:
            processing_id: A new processing ID to add. If a list is provided, all IDs will be added.

        Raises:
            ValueError: If the processing ID already exists.
            TypeError: If processing_id is not a string or list of strings.
        """
        if isinstance(processing_id, str):
            processing_id = [processing_id]
        elif not isinstance(processing_id, list):
            raise TypeError("processing_id must be a string or a list of strings")

        for pid in processing_id:
            if not isinstance(pid, str):
                raise TypeError("All processing IDs must be strings")
            if pid in self._processing_id_to_index:
                raise ValueError(f"Processing ID '{pid}' already exists")

        current_num_processings = self._array.shape[1]
        new_num_processings = current_num_processings + len(processing_id)
        new_shape = (self.num_samples, new_num_processings, self.num_features)

        new_array = np.empty(new_shape, dtype=self._array.dtype)
        new_array[:, :current_num_processings, :] = self._array
        for i in range(len(processing_id)):
            new_array[:, current_num_processings + i, :] = self._array[:, 0, :]

        self._array = new_array

        for i, pid in enumerate(processing_id):
            self._processing_ids.append(pid)
            self._processing_id_to_index[pid] = current_num_processings + i

    def add_samples(self, new_samples: np.ndarray) -> None:
        """
        Add new samples to the source.
        Args:
            new_samples: A 2D numpy array of shape (num_new_samples, num_features).
        Raises:
            ValueError: If new_samples is not a 2D array with the correct number of features.
        """
        if new_samples.ndim == 1:
            new_samples = new_samples[:, np.newaxis]

        if new_samples.shape[1] > self.num_features:
            raise ValueError(f"new_samples must have {self.num_features} features, got {new_samples.shape[1]}")
        elif new_samples.shape[1] < self.num_features:
            if self.padding:
                # Pad new_samples to match num_features
                padded_samples = np.full((new_samples.shape[0], self.num_features), self.pad_value, dtype=new_samples.dtype)
                padded_samples[:, :new_samples.shape[1]] = new_samples
                new_samples = padded_samples
            else:
                raise ValueError(f"new_samples must have {self.num_features} features, got {new_samples.shape[1]}")

        new_num_samples = self.num_samples + new_samples.shape[0]
        new_shape = (new_num_samples, self._array.shape[1], self.num_features)

        new_array = np.empty(new_shape, dtype=self._array.dtype)
        new_array[:self.num_samples, :, :] = self._array
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

    def update_processing(self, processing_id: str | int, new_data: np.ndarray, new_processing: str) -> None:
        """
        Update the data for a specific processing ID.

        Args:
            processing_id: The processing ID to update.
            new_data: A 2D numpy array of shape (num_samples, num_features).
            new_processing: The new processing ID to assign to the updated data.
        Raises:
            ValueError: If the processing ID does not exist or if new_data has incorrect shape.
        """
        if new_processing is None or not isinstance(new_processing, str):
            raise ValueError("new_processing must be a non-empty string")

        if new_processing in self._processing_id_to_index:
            raise ValueError(f"Processing ID '{new_processing}' already exists") # TODO here warning and return current

        if processing_id not in self._processing_id_to_index:
            raise ValueError(f"Processing ID '{processing_id}' does not exist")

        if new_data.ndim != 2 or new_data.shape[1] != self.num_features:
            raise ValueError(f"new_data must be a 2D array with {self.num_features} features")

        if isinstance(processing_id, str):
            idx = self._processing_id_to_index[processing_id]
        else:
            idx = processing_id

        if self.padding:
            if new_data.shape[0] < self.num_samples:
                padded_data = np.full((self.num_samples, self.num_features), self.pad_value, dtype=new_data.dtype)
                padded_data[:new_data.shape[0], :] = new_data
                new_data = padded_data
            elif new_data.shape[0] > self.num_samples:
                raise ValueError("new_data has more samples than the source, padding is not allowed in this case")
        else:
            if new_data.shape[0] != self.num_samples:
                raise ValueError(f"new_data must have {self.num_samples} samples when padding is False")

        if new_processing not in self._processing_id_to_index:
            self.add_processings(new_processing)

        self._array[:, idx, :] = new_data

    def layout_2d(self, row_indices: np.ndarray) -> np.ndarray:
        """
        Get a 2D layout of the source data for specified row indices. Concatenates processing variants along axis=1.

        Args:
            row_indices: A 1D numpy array of row indices to extract.

        Returns:
            A 2D numpy array with shape (len(row_indices), num_processings * num_features).
        """
        if not isinstance(row_indices, np.ndarray) or row_indices.ndim != 1:
            raise ValueError("row_indices must be a 1D numpy array")

        if len(row_indices) == 0:
            return np.empty((0, self.num_processings * self.num_features), dtype=self._array.dtype)

        # Ensure row indices are within bounds
        if np.any(row_indices < 0) or np.any(row_indices >= self.num_samples):
            raise IndexError("Row indices out of bounds")

        # Extract the specified rows and concatenate processing variants
        selected_data = self._array[row_indices, :, :].reshape(len(row_indices), -1)
        return selected_data

    def layout_2d_interleaved(self, row_indices: np.ndarray) -> np.ndarray:
        """
        Get a 2D interleaved layout of the source data for specified row indices.
        Interleaves features from different processing variants along axis=1, so that
        features are ordered as [proc1_f1, proc2_f1, ..., procN_f1, proc1_f2, proc2_f2, ..., procN_f2, ...].

        Args:
            row_indices: A 1D numpy array of row indices to extract.

        Returns:
            A 2D numpy array with shape (len(row_indices), num_processings * num_features).
        """
        if not isinstance(row_indices, np.ndarray) or row_indices.ndim != 1:
            raise ValueError("row_indices must be a 1D numpy array")

        if len(row_indices) == 0:
            return np.empty((0, self.num_processings * self.num_features), dtype=self._array.dtype)

        # Ensure row indices are within bounds
        if np.any(row_indices < 0) or np.any(row_indices >= self.num_samples):
            raise IndexError("Row indices out of bounds")

        # Extract rows, transpose to interleave features, and reshape to 2D
        selected_data = self._array[row_indices, :, :]  # Shape: (len(row_indices), num_processings, num_features)
        interleaved = np.transpose(selected_data, (0, 2, 1)).reshape(len(row_indices), -1)
        return interleaved

    def layout_3d(self, row_indices: np.ndarray) -> np.ndarray:
        """
        Get a 3D layout of the source data for specified row indices.
        Stacks processing variants along a new axis, resulting in shape (len(row_indices), num_processings, num_features).

        Args:
            row_indices: A 1D numpy array of row indices to extract.

        Returns:
            A 3D numpy array with shape (len(row_indices), num_processings, num_features).
        """
        if not isinstance(row_indices, np.ndarray) or row_indices.ndim != 1:
            raise ValueError("row_indices must be a 1D numpy array")

        if len(row_indices) == 0:
            return np.empty((0, self.num_processings, self.num_features), dtype=self._array.dtype)

        # Ensure row indices are within bounds
        if np.any(row_indices < 0) or np.any(row_indices >= self.num_samples):
            raise IndexError("Row indices out of bounds")

        # Extract the specified rows
        selected_data = self._array[row_indices, :, :]
        return selected_data

    def layout_3d_transpose(self, row_indices: np.ndarray) -> np.ndarray:
        """
        Get a 3D layout of the source data for specified row indices with transposed last two axes.
        This results in shape (len(row_indices), num_features, num_processings).

        Args:
            row_indices: A 1D numpy array of row indices to extract.

        Returns:
            A 3D numpy array with shape (len(row_indices), num_features, num_processings).
        """
        if not isinstance(row_indices, np.ndarray) or row_indices.ndim != 1:
            raise ValueError("row_indices must be a 1D numpy array")

        if len(row_indices) == 0:
            return np.empty((0, self.num_features, self.num_processings), dtype=self._array.dtype)

        # Ensure row indices are within bounds
        if np.any(row_indices < 0) or np.any(row_indices >= self.num_samples):
            raise IndexError("Row indices out of bounds")

        # Extract the specified rows and transpose last two axes
        selected_data = self._array[row_indices, :, :]
        return np.transpose(selected_data, (0, 2, 1))