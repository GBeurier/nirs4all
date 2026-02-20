"""Low-level block-based array storage with padding support and copy-on-write semantics.

Storage uses per-processing 2D blocks instead of a monolithic 3D array.
``add_processing()`` appends a block in O(1) without copying existing data.
A 3D view is computed lazily and cached for consumers that need it.
"""

from typing import Optional

import numpy as np


class SharedBlocks:
    """Reference-counted immutable block wrapper with copy-on-write semantics.

    Wraps a numpy array and tracks how many holders share it. When only one
    holder exists, mutations can proceed in-place. When shared, ``detach()``
    creates an independent copy so the original remains untouched.

    Used by ``ArrayStorage`` to avoid deep-copying feature arrays during
    branch snapshot/restore in the pipeline executor.
    """

    __slots__ = ("_array", "_refcount")

    def __init__(self, array: np.ndarray):
        self._array = array
        self._refcount = 1

    @property
    def array(self) -> np.ndarray:
        return self._array

    @property
    def refcount(self) -> int:
        return self._refcount

    def acquire(self) -> "SharedBlocks":
        """Increment refcount and return self for chaining."""
        self._refcount += 1
        return self

    def release(self) -> None:
        """Decrement refcount."""
        self._refcount = max(0, self._refcount - 1)

    def is_shared(self) -> bool:
        return self._refcount > 1

    def detach(self) -> "SharedBlocks":
        """Return an independent copy if shared, otherwise return self.

        This is the CoW trigger: only copies when someone else holds a
        reference.
        """
        if self._refcount > 1:
            self._refcount -= 1
            return SharedBlocks(self._array.copy())
        return self

    def __deepcopy__(self, memo: dict) -> "SharedBlocks":
        """Deep copy creates an independent block with refcount=1."""
        import copy
        return SharedBlocks(copy.deepcopy(self._array, memo))

    def __repr__(self) -> str:
        return f"SharedBlocks(shape={self._array.shape}, refcount={self._refcount})"

class ArrayStorage:
    """Block-based array storage with per-processing 2D blocks.

    Replaces the monolithic 3D array (samples x processings x features) with
    a list of 2D blocks (each samples x features, one per processing).

    Benefits:
    - ``add_processing()`` appends a block — O(1) allocation, no copy
    - ``remove_processing()`` drops a block — O(1) memory release
    - 3D view is computed lazily only when needed
    - Copy-on-Write (CoW) support for branch snapshots via ``SharedBlocks``

    Attributes:
        padding: Whether to allow padding when adding features with fewer dimensions.
        pad_value: Value to use for padding (default: 0.0).
    """

    def __init__(self, padding: bool = True, pad_value: float = 0.0):
        """Initialize empty array storage.

        Args:
            padding: If True, allow padding features to match existing dimensions.
            pad_value: Value to use for padding missing features.
        """
        self.padding = padding
        self.pad_value = pad_value
        self._blocks: list[np.ndarray] = []
        self._cached_3d: np.ndarray | None = None
        self._shared: SharedBlocks | None = None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _invalidate_cache(self) -> None:
        """Invalidate the cached 3D array after block mutation."""
        self._cached_3d = None

    def _materialize_blocks(self) -> None:
        """Decompose ``_shared`` 3D array into per-processing blocks.

        Handles CoW detach if the shared array has multiple holders.
        Called when transitioning from shared mode to block mode.
        """
        if self._shared is None:
            return
        self._shared = self._shared.detach()
        arr = self._shared._array
        self._blocks = [arr[:, i, :].copy() for i in range(arr.shape[1])]
        self._shared = None
        self._cached_3d = None

    def _prepare_for_mutation(self) -> None:
        """Ensure blocks are ready for in-place mutation.

        Handles two states:
        - Block mode (blocks exist): release shared ref, invalidate cache.
        - Shared mode (CoW restore): materialize blocks from shared 3D.
        """
        if self._blocks:
            if self._shared is not None:
                self._shared.release()
                self._shared = None
            self._cached_3d = None
            return
        if self._shared is not None:
            self._materialize_blocks()

    def _build_3d(self) -> np.ndarray:
        """Stack blocks into a 3D array (samples x processings x features)."""
        if not self._blocks:
            return np.empty((0, 0, 0), dtype=np.float32)
        return np.stack(self._blocks, axis=1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _array(self) -> np.ndarray:
        """Internal access to the 3D array (lazy)."""
        return self.array

    @property
    def array(self) -> np.ndarray:
        """Get the underlying 3D array (lazily computed from blocks).

        Returns:
            The 3D numpy array of shape (samples, processings, features).
        """
        if self._shared is not None:
            return self._shared._array
        if self._cached_3d is None:
            self._cached_3d = self._build_3d()
        return self._cached_3d

    @property
    def shape(self) -> tuple:
        """Get array shape.

        Returns:
            Tuple of (samples, processings, features).
        """
        if self._blocks:
            return (self._blocks[0].shape[0], len(self._blocks), self._blocks[0].shape[1])
        if self._shared is not None:
            return self._shared._array.shape
        return (0, 0, 0)

    @property
    def dtype(self) -> np.dtype:
        """Get array data type.

        Returns:
            Numpy dtype of the array.
        """
        if self._blocks:
            return self._blocks[0].dtype
        if self._shared is not None:
            return self._shared._array.dtype
        return np.dtype(np.float32)

    @property
    def num_samples(self) -> int:
        """Get number of samples.

        Returns:
            Number of samples (first dimension).
        """
        if self._blocks:
            return int(self._blocks[0].shape[0])
        if self._shared is not None:
            return int(self._shared._array.shape[0])
        return 0

    @property
    def num_processings(self) -> int:
        """Get number of processing blocks.

        Returns:
            Number of processing blocks.
        """
        if self._blocks:
            return len(self._blocks)
        if self._shared is not None:
            return int(self._shared._array.shape[1])
        return 0

    @property
    def num_features(self) -> int:
        """Get number of features per processing.

        Returns:
            Number of features (second dimension of each block).
        """
        if self._blocks:
            return int(self._blocks[0].shape[1])
        if self._shared is not None:
            return int(self._shared._array.shape[2])
        return 0

    @property
    def nbytes(self) -> int:
        """Total memory footprint of stored data.

        Returns:
            Sum of nbytes across all blocks, or shared array nbytes.
        """
        if self._blocks:
            return sum(b.nbytes for b in self._blocks)
        if self._shared is not None:
            return self._shared._array.nbytes
        return 0

    # ------------------------------------------------------------------
    # CoW support
    # ------------------------------------------------------------------

    def ensure_shared(self) -> SharedBlocks:
        """Build a SharedBlocks wrapper for CoW snapshot.

        Returns a SharedBlocks wrapping the current 3D array. The branch
        controller should call ``acquire()`` on the returned object.

        Returns:
            SharedBlocks wrapping the current data as a 3D array.
        """
        if self._shared is not None and not self._blocks:
            return self._shared
        arr_3d = self._build_3d()
        self._shared = SharedBlocks(arr_3d)
        return self._shared

    def restore_from_shared(self, shared: SharedBlocks) -> None:
        """Restore storage state from a CoW SharedBlocks reference.

        Used during branch restore. Clears blocks and cache, setting
        the shared 3D array as the source of truth until the next mutation.

        Args:
            shared: SharedBlocks (already acquired) to restore from.
        """
        if self._shared is not None:
            self._shared.release()
        self._shared = shared
        self._blocks = []
        self._cached_3d = None

    # ------------------------------------------------------------------
    # Mutation methods
    # ------------------------------------------------------------------

    def initialize_with_data(self, data: np.ndarray) -> None:
        """Initialize array with first batch of data.

        Args:
            data: 2D array of shape (n_samples, n_features).
        """
        if data.ndim != 2:
            raise ValueError(f"data must be a 2D array, got {data.ndim} dimensions")
        self._blocks = [data.astype(np.float32)]
        self._shared = None
        self._cached_3d = None

    def add_samples(self, data: np.ndarray) -> None:
        """Add new samples (rows) to the array.

        Only valid when there is a single processing block.

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
            self._prepare_for_mutation()
            prepared_data = self._prepare_data_for_storage(data)
            self._blocks[0] = np.concatenate(
                (self._blocks[0], prepared_data.astype(np.float32)), axis=0
            )

    def add_samples_batch(self, data: np.ndarray) -> None:
        """Add multiple new samples in a single operation — O(N) instead of O(N^2).

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
            self._blocks = [
                data[:, i, :].astype(np.float32).copy() for i in range(data.shape[1])
            ]
            self._shared = None
            self._cached_3d = None
        else:
            self._prepare_for_mutation()
            if data.shape[1] != self.num_processings:
                raise ValueError(
                    f"Processing dimension mismatch: expected {self.num_processings}, "
                    f"got {data.shape[1]}"
                )

            # Handle feature padding
            if self.padding and data.shape[2] < self.num_features:
                padded_data = np.full(
                    (n_new_samples, self.num_processings, self.num_features),
                    self.pad_value, dtype=np.float32,
                )
                padded_data[:, :, :data.shape[2]] = data
                data = padded_data
            elif not self.padding and data.shape[2] != self.num_features:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self.num_features}, "
                    f"got {data.shape[2]}"
                )
            else:
                data = data.astype(np.float32)

            for i in range(len(self._blocks)):
                self._blocks[i] = np.concatenate(
                    (self._blocks[i], data[:, i, :]), axis=0
                )

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
                dtype=self.dtype
            )
            padded_data[:, :data.shape[1]] = data
            return padded_data
        elif not self.padding and data.shape[1] != self.num_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.num_features}, "
                f"got {data.shape[1]}"
            )

        return data.astype(self.dtype)

    def update_processing(self, proc_idx: int, data: np.ndarray) -> None:
        """Update data for a specific processing.

        Args:
            proc_idx: Processing index to update.
            data: 2D array of shape (n_samples, n_features).
        """
        self._prepare_for_mutation()
        prepared_data = self._prepare_data_for_storage(data)
        self._blocks[proc_idx] = prepared_data.astype(self.dtype)

    def add_processing(self, data: np.ndarray) -> int:
        """Add a new processing block.

        O(1) allocation — appends a 2D block without copying existing data.

        Args:
            data: 2D array of shape (n_samples, n_features).

        Returns:
            Index of the newly added processing.
        """
        self._prepare_for_mutation()
        prepared_data = self._prepare_data_for_storage(data)
        self._blocks.append(prepared_data.astype(np.float32))
        return len(self._blocks) - 1

    def add_processings_batch(self, blocks: list[np.ndarray]) -> list[int]:
        """Add multiple processing blocks in one call.

        More efficient than calling ``add_processing()`` in a loop — invalidates
        the cached 3D view only once.

        Args:
            blocks: List of 2D arrays, each of shape (n_samples, n_features).

        Returns:
            List of indices for the newly added processings.
        """
        if not blocks:
            return []
        self._prepare_for_mutation()
        start_idx = len(self._blocks)
        for block in blocks:
            prepared = self._prepare_data_for_storage(block)
            self._blocks.append(prepared.astype(np.float32))
        return list(range(start_idx, len(self._blocks)))

    def remove_processing(self, proc_idx: int) -> None:
        """Remove a processing block by index.

        O(1) memory release — drops the block without copying.

        Args:
            proc_idx: Index of the processing to remove.
        """
        self._prepare_for_mutation()
        del self._blocks[proc_idx]

    def resize_features(self, new_num_features: int) -> None:
        """Resize the feature dimension across all processing blocks.

        Args:
            new_num_features: New size for the feature dimension.
        """
        if self.num_samples == 0:
            return

        self._prepare_for_mutation()
        min_features = min(self.num_features, new_num_features)
        for i in range(len(self._blocks)):
            old_block = self._blocks[i]
            new_block = np.zeros(
                (self.num_samples, new_num_features), dtype=old_block.dtype
            )
            new_block[:, :min_features] = old_block[:, :min_features]
            self._blocks[i] = new_block

    def reset_data(self, data: np.ndarray) -> None:
        """Reset the array storage with new data.

        Replaces the entire storage with the provided data.

        Args:
            data: 3D array of shape (n_samples, n_processings, n_features)
                  or 2D array of shape (n_samples, n_features) (single processing).

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

        self._blocks = [
            data[:, i, :].astype(np.float32).copy() for i in range(data.shape[1])
        ]
        self._shared = None
        self._cached_3d = None

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
                dtype=self.dtype
            )
        return self.array[sample_indices, :, :]

    def augment_samples(
        self,
        sample_indices: list,
        count_list: list,
        new_proc_data: np.ndarray | None = None
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

        self._prepare_for_mutation()

        # Build index array for augmented rows
        aug_indices = []
        for orig_idx, aug_count in zip(sample_indices, count_list, strict=False):
            aug_indices.extend([orig_idx] * aug_count)
        aug_indices_arr = np.array(aug_indices)

        # Expand each block with augmented rows
        for i in range(len(self._blocks)):
            aug_rows = self._blocks[i][aug_indices_arr, :]
            self._blocks[i] = np.concatenate((self._blocks[i], aug_rows), axis=0)

        self._invalidate_cache()

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
        self._prepare_for_mutation()
        augmented_start_idx = self.num_samples - total_augmentations
        prepared_data = self._prepare_data_for_storage(data)

        # New block: pad_value for all, then fill augmented rows
        new_block = np.full(
            (self.num_samples, self.num_features),
            self.pad_value,
            dtype=self.dtype,
        )
        new_block[augmented_start_idx:augmented_start_idx + total_augmentations, :] = (
            prepared_data[:total_augmentations, :]
        )

        self._blocks.append(new_block)

    def __deepcopy__(self, memo: dict) -> "ArrayStorage":
        """Deep copy produces an independent block-mode storage."""
        new = ArrayStorage.__new__(ArrayStorage)
        memo[id(self)] = new
        new.padding = self.padding
        new.pad_value = self.pad_value
        if self._blocks:
            new._blocks = [b.copy() for b in self._blocks]
        elif self._shared is not None:
            arr = self._shared._array
            new._blocks = [arr[:, i, :].copy() for i in range(arr.shape[1])]
        else:
            new._blocks = []
        new._cached_3d = None
        new._shared = None
        return new

    def __repr__(self) -> str:
        return f"ArrayStorage(shape={self.shape}, dtype={self.dtype}, blocks={len(self._blocks)})"
