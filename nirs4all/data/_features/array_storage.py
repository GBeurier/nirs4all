"""Low-level 3D array storage with padding support."""

import numpy as np
from typing import Optional


class ArrayStorage:
    """Manages the 3D numpy array (samples, processings, features) with padding.

    This class handles the underlying storage and retrieval of feature data,
    including padding operations when feature dimensions don't match.

    Attributes:
        padding: Whether to allow padding when adding features with fewer dimensions.
        pad_value: Value to use for padding (default: 0.0).
        array: The underlying 3D numpy array.
    """

    def __init__(self, padding: bool = True, pad_value: float = 0.0):
        """Initialize empty array storage.

        Args:
            padding: If True, allow padding features to match existing dimensions.
            pad_value: Value to use for padding missing features.
        """
        self.padding = padding
        self.pad_value = pad_value
        self._array = np.empty((0, 1, 0), dtype=np.float32)

    @property
    def array(self) -> np.ndarray:
        """Get the underlying array.

        Returns:
            The 3D numpy array.
        """
        return self._array

    @property
    def shape(self) -> tuple:
        """Get array shape.

        Returns:
            Tuple of (samples, processings, features).
        """
        return self._array.shape

    @property
    def dtype(self) -> np.dtype:
        """Get array data type.

        Returns:
            Numpy dtype of the array.
        """
        return self._array.dtype

    @property
    def num_samples(self) -> int:
        """Get number of samples.

        Returns:
            Number of samples (first dimension).
        """
        return self._array.shape[0]

    @property
    def num_processings(self) -> int:
        """Get number of processings.

        Returns:
            Number of processings (second dimension).
        """
        return self._array.shape[1]

    @property
    def num_features(self) -> int:
        """Get number of features.

        Returns:
            Number of features (third dimension).
        """
        return self._array.shape[2]

    def initialize_with_data(self, data: np.ndarray) -> None:
        """Initialize array with first batch of data.

        Args:
            data: 2D array of shape (n_samples, n_features).
        """
        if data.ndim != 2:
            raise ValueError(f"data must be a 2D array, got {data.ndim} dimensions")
        self._array = data[:, None, :].astype(np.float32)

    def add_samples(self, data: np.ndarray) -> None:
        """Add new samples (rows) to the array.

        Args:
            data: 2D array of shape (n_samples, n_features).

        Raises:
            ValueError: If data is not 2D or dimensions don't match.
        """
        if data.ndim != 2:
            raise ValueError(f"data must be a 2D array, got {data.ndim} dimensions")

        if self.num_samples == 0:
            self.initialize_with_data(data)
        else:
            prepared_data = self._prepare_data_for_storage(data)
            new_data_3d = prepared_data[:, None, :]
            self._array = np.concatenate((self._array, new_data_3d), axis=0)

    def add_samples_batch(self, data: np.ndarray) -> None:
        """Add multiple new samples in a single operation - O(N) instead of O(N²).

        This is much more efficient than calling add_samples() in a loop because
        it performs only one array concatenation for all samples.

        Args:
            data: 3D array of shape (n_samples, n_processings, n_features).
                  For single processing, use shape (n_samples, 1, n_features).

        Raises:
            ValueError: If data dimensions don't match or are invalid.
        """
        if data.ndim != 3:
            raise ValueError(f"data must be a 3D array, got {data.ndim} dimensions")

        n_new_samples = data.shape[0]
        if n_new_samples == 0:
            return

        if self.num_samples == 0:
            # Initialize with the batch data
            self._array = data.astype(np.float32)
        else:
            # Validate dimensions
            if data.shape[1] != self.num_processings:
                raise ValueError(
                    f"Processing dimension mismatch: expected {self.num_processings}, "
                    f"got {data.shape[1]}"
                )

            # Handle feature padding if needed
            if self.padding and data.shape[2] < self.num_features:
                padded_data = np.full(
                    (n_new_samples, self.num_processings, self.num_features),
                    self.pad_value,
                    dtype=self._array.dtype
                )
                padded_data[:, :, :data.shape[2]] = data
                data = padded_data
            elif not self.padding and data.shape[2] != self.num_features:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self.num_features}, "
                    f"got {data.shape[2]}"
                )
            else:
                data = data.astype(self._array.dtype)

            # Single concatenation for all samples - O(N) instead of O(N²)
            self._array = np.concatenate((self._array, data), axis=0)

    def _prepare_data_for_storage(self, data: np.ndarray) -> np.ndarray:
        """Prepare data for storage by handling padding and dimension matching.

        Args:
            data: 2D array to prepare.

        Returns:
            Prepared array with matching feature dimension.

        Raises:
            ValueError: If padding is disabled and dimensions don't match.
        """
        if self.num_samples == 0:
            return data

        if self.padding and data.shape[1] < self.num_features:
            padded_data = np.full(
                (data.shape[0], self.num_features),
                self.pad_value,
                dtype=self._array.dtype
            )
            padded_data[:, :data.shape[1]] = data
            return padded_data
        elif not self.padding and data.shape[1] != self.num_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.num_features}, "
                f"got {data.shape[1]}"
            )

        return data.astype(self._array.dtype)

    def update_processing(self, proc_idx: int, data: np.ndarray) -> None:
        """Update data for a specific processing.

        Args:
            proc_idx: Processing index to update.
            data: 2D array of shape (n_samples, n_features).
        """
        prepared_data = self._prepare_data_for_storage(data)
        self._array[:, proc_idx, :] = prepared_data

    def add_processing(self, data: np.ndarray) -> int:
        """Add a new processing dimension.

        Args:
            data: 2D array of shape (n_samples, n_features).

        Returns:
            Index of the newly added processing.
        """
        prepared_data = self._prepare_data_for_storage(data)
        new_data_3d = prepared_data[:, None, :]

        if self.num_samples == 0:
            self._array = new_data_3d
        else:
            self._array = np.concatenate((self._array, new_data_3d), axis=1)

        return self.num_processings - 1

    def resize_features(self, new_num_features: int) -> None:
        """Resize the feature dimension.

        Args:
            new_num_features: New size for the feature dimension.
        """
        if self.num_samples == 0:
            return

        new_shape = (self.num_samples, self.num_processings, new_num_features)
        new_array = np.zeros(new_shape, dtype=self._array.dtype)

        min_features = min(self.num_features, new_num_features)
        new_array[:, :, :min_features] = self._array[:, :, :min_features]

        self._array = new_array

    def reset_data(self, data: np.ndarray) -> None:
        """Reset the array storage with new data.

        Replaces the entire 3D array with the provided data.
        Used when resetting feature storage (e.g. after merge).

        Args:
            data: 3D array of shape (n_samples, n_processings, n_features).
                  or 2D array of shape (n_samples, n_features) (will be reshaped to 3D).

        Raises:
            ValueError: If sample count doesn't match existing samples (unless empty).
        """
        if data.ndim == 2:
            data = data[:, None, :]

        if data.ndim != 3:
            raise ValueError(f"data must be 2D or 3D, got {data.ndim} dimensions")

        if self.num_samples > 0 and data.shape[0] != self.num_samples:
            raise ValueError(
                f"Sample count mismatch: expected {self.num_samples}, "
                f"got {data.shape[0]}"
            )

        self._array = data.astype(np.float32)

    def get_data(self, sample_indices: np.ndarray) -> np.ndarray:
        """Get data for specific sample indices.

        Args:
            sample_indices: Array of sample indices to retrieve.

        Returns:
            3D array of shape (len(sample_indices), processings, features).
        """
        if len(sample_indices) == 0:
            return np.empty(
                (0, self.num_processings, self.num_features),
                dtype=self._array.dtype
            )
        return self._array[sample_indices, :, :]

    def augment_samples(
        self,
        sample_indices: list,
        count_list: list,
        new_proc_data: Optional[np.ndarray] = None
    ) -> None:
        """Augment samples by duplicating them.

        Args:
            sample_indices: List of sample indices to augment.
            count_list: Number of augmentations per sample.
            new_proc_data: Optional data for new processing (if adding a new processing).
        """
        total_augmentations = sum(count_list)
        if total_augmentations == 0:
            return

        # Expand array to accommodate new samples
        new_num_samples = self.num_samples + total_augmentations
        current_processings = self._array.shape[1]
        new_shape = (new_num_samples, current_processings, self.num_features)

        expanded_array = np.full(new_shape, self.pad_value, dtype=self._array.dtype)
        expanded_array[:self.num_samples, :current_processings, :] = self._array

        # Copy augmented samples
        sample_idx = 0
        for orig_idx, aug_count in zip(sample_indices, count_list):
            for _ in range(aug_count):
                expanded_array[self.num_samples + sample_idx, :current_processings, :] = \
                    self._array[orig_idx, :current_processings, :]
                sample_idx += 1

        self._array = expanded_array

        # If new processing data is provided, add it
        if new_proc_data is not None:
            self._add_processing_for_augmented(new_proc_data, total_augmentations)

    def _add_processing_for_augmented(
        self,
        data: np.ndarray,
        total_augmentations: int
    ) -> None:
        """Add new processing dimension for augmented samples.

        Args:
            data: Processing data for augmented samples.
            total_augmentations: Number of augmented samples.
        """
        current_processings = self._array.shape[1]
        new_shape = (self.num_samples, current_processings + 1, self.num_features)
        expanded_array = np.full(new_shape, self.pad_value, dtype=self._array.dtype)

        # Copy existing data
        expanded_array[:, :current_processings, :] = self._array

        # Add new processing data only to augmented samples
        augmented_start_idx = self.num_samples - total_augmentations
        prepared_data = self._prepare_data_for_storage(data)

        for i in range(total_augmentations):
            augmented_sample_idx = augmented_start_idx + i
            expanded_array[augmented_sample_idx, current_processings, :] = prepared_data[i, :]

        self._array = expanded_array

    def __repr__(self) -> str:
        return f"ArrayStorage(shape={self.shape}, dtype={self.dtype})"
