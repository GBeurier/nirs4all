"""Tests for SharedBlocks copy-on-write wrapper and block-based ArrayStorage."""

import copy

import numpy as np
import pytest

from nirs4all.data._features.array_storage import ArrayStorage, SharedBlocks


class TestSharedBlocks:
    """Unit tests for the SharedBlocks reference-counted CoW wrapper."""

    def test_initial_refcount_is_one(self):
        arr = np.zeros((3, 2))
        sb = SharedBlocks(arr)
        assert sb.refcount == 1
        assert not sb.is_shared()

    def test_acquire_increments_refcount(self):
        sb = SharedBlocks(np.zeros((3, 2)))
        same = sb.acquire()
        assert same is sb
        assert sb.refcount == 2
        assert sb.is_shared()

    def test_release_decrements_refcount(self):
        sb = SharedBlocks(np.zeros((3, 2)))
        sb.acquire()
        assert sb.refcount == 2
        sb.release()
        assert sb.refcount == 1

    def test_release_does_not_go_below_zero(self):
        sb = SharedBlocks(np.zeros((3, 2)))
        sb.release()
        sb.release()
        assert sb.refcount == 0

    def test_detach_on_unshared_returns_self(self):
        arr = np.ones((3, 2))
        sb = SharedBlocks(arr)
        detached = sb.detach()
        assert detached is sb
        assert sb.refcount == 1

    def test_detach_on_shared_returns_independent_copy(self):
        arr = np.arange(6, dtype=np.float32).reshape(3, 2)
        sb = SharedBlocks(arr)
        sb.acquire()
        assert sb.refcount == 2

        detached = sb.detach()
        assert detached is not sb
        assert detached.refcount == 1
        assert sb.refcount == 1  # original decremented
        np.testing.assert_array_equal(detached.array, arr)

    def test_detach_produces_independent_memory(self):
        arr = np.arange(6, dtype=np.float32).reshape(3, 2)
        sb = SharedBlocks(arr)
        sb.acquire()

        detached = sb.detach()
        detached._array[0, 0] = 999.0

        # Original must be untouched
        assert sb.array[0, 0] != 999.0

    def test_array_property(self):
        arr = np.ones((2, 3))
        sb = SharedBlocks(arr)
        assert sb.array is arr

class TestArrayStorageBlockBased:
    """Tests for block-based ArrayStorage internals."""

    def _make_storage(self, n_samples=10, n_features=5) -> ArrayStorage:
        storage = ArrayStorage()
        storage.initialize_with_data(
            np.random.rand(n_samples, n_features).astype(np.float32)
        )
        return storage

    def test_initial_state_is_empty(self):
        storage = ArrayStorage()
        assert storage.num_samples == 0
        assert storage.num_processings == 0
        assert storage.num_features == 0
        assert storage.nbytes == 0
        assert len(storage._blocks) == 0
        assert storage._shared is None

    def test_initialize_creates_single_block(self):
        storage = self._make_storage(10, 5)
        assert len(storage._blocks) == 1
        assert storage._blocks[0].shape == (10, 5)
        assert storage._shared is None
        assert storage.shape == (10, 1, 5)

    def test_add_processing_appends_block_without_copy(self):
        storage = self._make_storage(10, 5)
        block0_id = id(storage._blocks[0])

        new_data = np.random.rand(10, 5).astype(np.float32)
        storage.add_processing(new_data)

        assert len(storage._blocks) == 2
        assert storage.num_processings == 2
        # Original block should be the same object (no copy needed)
        assert id(storage._blocks[0]) == block0_id

    def test_add_processings_batch(self):
        storage = self._make_storage(10, 5)
        blocks = [
            np.random.rand(10, 5).astype(np.float32),
            np.random.rand(10, 5).astype(np.float32),
        ]
        indices = storage.add_processings_batch(blocks)
        assert indices == [1, 2]
        assert storage.num_processings == 3

    def test_remove_processing(self):
        storage = self._make_storage(10, 5)
        proc1 = np.random.rand(10, 5).astype(np.float32)
        proc2 = np.random.rand(10, 5).astype(np.float32)
        storage.add_processing(proc1)
        storage.add_processing(proc2)
        assert storage.num_processings == 3

        storage.remove_processing(1)
        assert storage.num_processings == 2
        np.testing.assert_array_almost_equal(storage._blocks[1], proc2, decimal=5)

    def test_lazy_3d_build(self):
        storage = self._make_storage(10, 5)
        assert storage._cached_3d is None

        arr = storage.array
        assert arr.shape == (10, 1, 5)
        assert storage._cached_3d is not None
        assert storage._cached_3d is arr

    def test_3d_cache_invalidated_on_mutation(self):
        storage = self._make_storage(10, 5)
        _ = storage.array  # populate cache
        assert storage._cached_3d is not None

        storage.add_processing(np.random.rand(10, 5).astype(np.float32))
        assert storage._cached_3d is None

    def test_array_matches_stacked_blocks(self):
        storage = self._make_storage(10, 5)
        proc1 = np.random.rand(10, 5).astype(np.float32)
        storage.add_processing(proc1)

        arr = storage.array
        assert arr.shape == (10, 2, 5)
        np.testing.assert_array_equal(arr[:, 0, :], storage._blocks[0])
        np.testing.assert_array_equal(arr[:, 1, :], storage._blocks[1])

    def test_nbytes_sums_blocks(self):
        storage = self._make_storage(10, 5)
        expected = 10 * 5 * 4  # float32 = 4 bytes
        assert storage.nbytes == expected

        storage.add_processing(np.random.rand(10, 5).astype(np.float32))
        assert storage.nbytes == 2 * expected

    def test_update_processing_replaces_block(self):
        storage = self._make_storage(10, 5)
        new_data = np.ones((10, 5), dtype=np.float32) * 42.0
        storage.update_processing(0, new_data)
        np.testing.assert_array_equal(storage._blocks[0], new_data)

    def test_resize_features(self):
        storage = self._make_storage(10, 5)
        storage.add_processing(np.random.rand(10, 5).astype(np.float32))
        storage.resize_features(3)
        assert storage.num_features == 3
        for block in storage._blocks:
            assert block.shape == (10, 3)

    def test_add_samples(self):
        storage = self._make_storage(10, 5)
        new_samples = np.random.rand(3, 5).astype(np.float32)
        storage.add_samples(new_samples)
        assert storage.num_samples == 13

    def test_add_samples_batch_3d(self):
        storage = self._make_storage(10, 5)
        storage.add_processing(np.random.rand(10, 5).astype(np.float32))
        batch = np.random.rand(3, 2, 5).astype(np.float32)
        storage.add_samples_batch(batch)
        assert storage.num_samples == 13

    def test_reset_data(self):
        storage = self._make_storage(10, 5)
        new_data = np.random.rand(10, 2, 5).astype(np.float32)
        storage.reset_data(new_data)
        assert len(storage._blocks) == 2
        assert storage.num_processings == 2
        np.testing.assert_array_almost_equal(storage.array, new_data, decimal=5)

    def test_get_data(self):
        storage = self._make_storage(10, 5)
        result = storage.get_data(np.array([0, 3, 7]))
        assert result.shape == (3, 1, 5)

    def test_get_data_empty_indices(self):
        storage = self._make_storage(10, 5)
        result = storage.get_data(np.array([], dtype=int))
        assert result.shape == (0, 1, 5)

    def test_deepcopy_produces_independent_blocks(self):
        storage = self._make_storage(10, 5)
        storage.add_processing(np.random.rand(10, 5).astype(np.float32))
        copied = copy.deepcopy(storage)

        assert len(copied._blocks) == 2
        assert copied._shared is None
        # Modify copy - original unaffected
        copied._blocks[0][0, 0] = 999.0
        assert storage._blocks[0][0, 0] != 999.0

    def test_deepcopy_from_shared_mode(self):
        storage = self._make_storage(10, 5)
        shared = storage.ensure_shared()
        shared.acquire()  # simulate branch holding a ref

        # Force shared mode (no blocks, shared set)
        storage.restore_from_shared(shared.acquire())
        assert len(storage._blocks) == 0

        copied = copy.deepcopy(storage)
        assert len(copied._blocks) == 1  # decomposed from shared
        assert copied._shared is None

class TestArrayStorageCoW:
    """Tests for ArrayStorage CoW integration via ensure_shared/restore_from_shared."""

    def _make_storage(self, n_samples=10, n_features=5) -> ArrayStorage:
        storage = ArrayStorage()
        storage.initialize_with_data(
            np.random.rand(n_samples, n_features).astype(np.float32)
        )
        return storage

    def test_ensure_shared_creates_shared_blocks(self):
        storage = self._make_storage()
        shared = storage.ensure_shared()
        assert isinstance(shared, SharedBlocks)
        assert shared.refcount == 1
        assert shared.array.shape == (10, 1, 5)

    def test_ensure_shared_returns_same_in_shared_mode(self):
        storage = self._make_storage()
        # Put into shared mode
        shared = storage.ensure_shared()
        storage.restore_from_shared(shared.acquire())

        shared2 = storage.ensure_shared()
        assert shared2 is storage._shared

    def test_restore_from_shared_clears_blocks(self):
        storage = self._make_storage()
        shared = storage.ensure_shared()
        storage.restore_from_shared(shared.acquire())

        assert len(storage._blocks) == 0
        assert storage._shared is not None
        assert storage.num_samples == 10

    def test_mutation_after_restore_materializes_blocks(self):
        storage = self._make_storage(10, 5)
        original_data = storage.array.copy()

        shared = storage.ensure_shared()
        shared.acquire()  # branch holds a ref

        # Restore (simulating branch restore)
        storage.restore_from_shared(shared.acquire())
        assert len(storage._blocks) == 0

        # Mutation triggers block materialization
        new_proc = np.random.rand(10, 5).astype(np.float32)
        storage.add_processing(new_proc)

        assert len(storage._blocks) == 2
        assert storage._shared is None
        np.testing.assert_array_almost_equal(
            storage._blocks[0], original_data[:, 0, :], decimal=5
        )

    def test_add_processing_detaches_when_shared(self):
        storage = self._make_storage(10, 5)
        shared = storage.ensure_shared()
        snapshot = shared.acquire()
        assert snapshot.refcount == 2

        new_data = np.random.rand(10, 5).astype(np.float32)
        storage.add_processing(new_data)

        # Shared was released, blocks are independent
        assert storage._shared is None
        assert storage.num_processings == 2
        # Snapshot refcount dropped to 1 (we released)
        assert snapshot.refcount == 1

    def test_update_processing_preserves_snapshot(self):
        storage = self._make_storage(10, 5)
        original_data = storage.array.copy()
        shared = storage.ensure_shared()
        snapshot = shared.acquire()

        new_data = np.ones((10, 5), dtype=np.float32) * 42.0
        storage.update_processing(0, new_data)

        # Snapshot still has original data
        np.testing.assert_array_equal(
            snapshot.array[:, 0, :], original_data[:, 0, :]
        )
        # Storage has new data
        np.testing.assert_array_equal(storage._blocks[0], new_data)

    def test_cow_branch_simulation(self):
        """Simulate the full branch lifecycle: snapshot -> restore -> modify."""
        storage = self._make_storage(10, 5)
        original_data = storage.array.copy()

        # Snapshot (branch controller calls ensure_shared + acquire)
        shared = storage.ensure_shared()
        snapshot = shared.acquire()
        assert snapshot.refcount == 2

        # Branch 0: restore and mutate
        # restore_from_shared releases the old _shared (same object), then sets new
        storage.restore_from_shared(snapshot.acquire())
        # refcount: acquire() bumps to 3, then release() drops to 2
        assert snapshot.refcount == 2

        new_proc = np.random.rand(10, 5).astype(np.float32)
        storage.add_processing(new_proc)

        # Storage got independent blocks via materialize (detach copies when refcount>1)
        assert storage._shared is None
        assert storage.num_processings == 2

        # Snapshot still points to original data (refcount dropped by 1 during detach)
        assert snapshot.refcount == 1
        np.testing.assert_array_equal(
            snapshot.array[:, 0, :], original_data[:, 0, :]
        )

    def test_get_data_does_not_trigger_cow(self):
        storage = self._make_storage(10, 5)
        shared = storage.ensure_shared()
        shared.acquire()
        block0_id = id(storage._blocks[0])

        _ = storage.get_data(np.array([0, 1, 2]))

        # Blocks should not have been modified
        assert id(storage._blocks[0]) == block0_id
        assert len(storage._blocks) == 1

    def test_reset_data_clears_shared(self):
        storage = self._make_storage(10, 5)
        shared = storage.ensure_shared()
        shared.acquire()

        new_data = np.random.rand(10, 1, 5).astype(np.float32)
        storage.reset_data(new_data)

        assert storage._shared is None
        assert len(storage._blocks) == 1
        np.testing.assert_array_almost_equal(storage.array, new_data, decimal=5)

class TestArrayStorageAugmentation:
    """Tests for sample augmentation with block-based storage."""

    def _make_storage(self, n_samples=10, n_features=5) -> ArrayStorage:
        storage = ArrayStorage()
        storage.initialize_with_data(
            np.arange(n_samples * n_features, dtype=np.float32).reshape(n_samples, n_features)
        )
        return storage

    def test_augment_duplicates_rows(self):
        storage = self._make_storage(10, 5)
        storage.augment_samples(
            sample_indices=[0, 2],
            count_list=[1, 2],
        )
        assert storage.num_samples == 13
        # First augmented row should match sample 0
        np.testing.assert_array_equal(
            storage._blocks[0][10, :], storage._blocks[0][0, :]
        )
        # Second and third augmented rows should match sample 2
        np.testing.assert_array_equal(
            storage._blocks[0][11, :], storage._blocks[0][2, :]
        )
        np.testing.assert_array_equal(
            storage._blocks[0][12, :], storage._blocks[0][2, :]
        )

    def test_augment_with_new_processing(self):
        storage = self._make_storage(10, 5)
        new_proc_data = np.ones((3, 5), dtype=np.float32) * 99.0
        storage.augment_samples(
            sample_indices=[0, 1],
            count_list=[2, 1],
            new_proc_data=new_proc_data,
        )
        assert storage.num_samples == 13
        assert storage.num_processings == 2
        # New processing block: augmented rows should have 99.0, others pad_value (0.0)
        new_block = storage._blocks[1]
        np.testing.assert_array_equal(new_block[10, :], np.full(5, 99.0))
        np.testing.assert_array_equal(new_block[0, :], np.full(5, 0.0))

    def test_augment_preserves_multiple_processings(self):
        storage = self._make_storage(10, 5)
        proc1 = np.random.rand(10, 5).astype(np.float32)
        storage.add_processing(proc1)
        assert storage.num_processings == 2

        storage.augment_samples(
            sample_indices=[3],
            count_list=[2],
        )
        assert storage.num_samples == 12
        assert storage.num_processings == 2
        # Both blocks should have 12 samples
        for block in storage._blocks:
            assert block.shape[0] == 12
