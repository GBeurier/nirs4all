"""Tests for StepCache CoW (copy-on-write) semantics.

Verifies that the SharedBlocks-based caching mechanism:
- Produces byte-identical round-trip results
- Provides true CoW isolation (mutations don't affect cached data)
- Supports multiple concurrent restores
- Properly releases SharedBlocks on eviction and clear
- Preserves content hash consistency after restore
- Preserves processing names through snapshot/restore
- Reports timing statistics
"""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.data._features.array_storage import SharedBlocks
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.execution.step_cache import CachedStepState, StepCache


# =========================================================================
# Helpers
# =========================================================================

def _make_dataset(name: str = "test", n_samples: int = 50, n_features: int = 100) -> SpectroDataset:
    """Create a simple SpectroDataset for testing."""
    ds = SpectroDataset(name)
    X = np.random.rand(n_samples, n_features).astype(np.float64)
    ds.add_samples(X, indexes={"partition": "train"})
    y = np.random.rand(n_samples)
    ds.add_targets(y)
    return ds


# =========================================================================
# CoW round-trip correctness
# =========================================================================


class TestStepCacheCoWRoundTrip:
    """Verify snapshot → restore produces byte-identical feature data."""

    def test_round_trip_preserves_feature_data(self):
        """Features are identical after cache snapshot + restore round-trip."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("rt_test", n_samples=20, n_features=10)
        X_original = ds.x({"partition": "train"}, layout="2d").copy()

        cache.put("chain", "data", ds)
        state = cache.get("chain", "data")
        assert state is not None

        # Restore into same dataset
        cache.restore(state, ds)
        X_restored = ds.x({"partition": "train"}, layout="2d")
        np.testing.assert_array_equal(X_original, X_restored)

    def test_round_trip_preserves_content_hash(self):
        """Content hash is consistent after snapshot + restore."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("hash_test", n_samples=10, n_features=5)
        original_hash = ds.content_hash()

        cache.put("chain", "data", ds)
        state = cache.get("chain", "data")
        cache.restore(state, ds)

        assert ds.content_hash() == original_hash
        assert ds._content_hash_cache == original_hash


# =========================================================================
# CoW isolation
# =========================================================================


class TestStepCacheCoWIsolation:
    """Verify that mutations on a restored dataset don't affect cached data."""

    def test_mutation_does_not_affect_cached_state(self):
        """Mutating dataset after restore does not change cached snapshot."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("iso_test", n_samples=10, n_features=5)
        X_original = ds.x({"partition": "train"}, layout="2d").copy()

        cache.put("chain", "data", ds)

        # Restore and mutate
        state = cache.get("chain", "data")
        cache.restore(state, ds)

        # Force mutation by directly accessing and modifying the storage
        for source in ds._features.sources:
            source._storage._prepare_for_mutation()
            source._storage._blocks[0][:] = 999.0

        # Get state again and restore into a fresh dataset
        state2 = cache.get("chain", "data")
        ds2 = _make_dataset("fresh", n_samples=10, n_features=5)
        cache.restore(state2, ds2)

        X_from_cache = ds2.x({"partition": "train"}, layout="2d")
        np.testing.assert_array_equal(X_original, X_from_cache)

    def test_put_creates_independent_snapshot(self):
        """Mutating the original dataset after put does not affect cache."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("snap_test", n_samples=10, n_features=5)
        original_hash = ds.content_hash()

        cache.put("chain", "data", ds)

        # Mutate the dataset's feature storage
        for source in ds._features.sources:
            source._storage._prepare_for_mutation()
            source._storage._blocks[0][:] = -1.0

        # Cache should still hold original data
        state = cache.get("chain", "data")
        assert state is not None
        assert state.content_hash == original_hash


# =========================================================================
# Multiple restores
# =========================================================================


class TestStepCacheMultipleRestores:
    """Verify that a single cached state can be restored into multiple datasets."""

    def test_two_restores_produce_independent_copies(self):
        """Two restores from same state yield independent arrays."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("multi", n_samples=10, n_features=5)
        X_original = ds.x({"partition": "train"}, layout="2d").copy()

        cache.put("chain", "data", ds)
        state = cache.get("chain", "data")

        ds1 = _make_dataset("t1", n_samples=10, n_features=5)
        ds2 = _make_dataset("t2", n_samples=10, n_features=5)
        cache.restore(state, ds1)
        cache.restore(state, ds2)

        # Both should match original
        np.testing.assert_array_equal(
            ds1.x({"partition": "train"}, layout="2d"), X_original
        )
        np.testing.assert_array_equal(
            ds2.x({"partition": "train"}, layout="2d"), X_original
        )

        # Mutating ds1 should not affect ds2
        for source in ds1._features.sources:
            source._storage._prepare_for_mutation()
            source._storage._blocks[0][:] = 999.0

        X2 = ds2.x({"partition": "train"}, layout="2d")
        np.testing.assert_array_equal(X2, X_original)


# =========================================================================
# Eviction cleanup
# =========================================================================


class TestStepCacheEvictionCleanup:
    """Verify SharedBlocks references are released on eviction."""

    def test_eviction_releases_shared_blocks(self):
        """When an entry is evicted, its SharedBlocks refcounts drop."""
        # Very small cache: only fits one entry
        cache = StepCache(max_size_mb=0.01, max_entries=1)
        ds1 = _make_dataset("ds1", n_samples=5, n_features=5)
        ds2 = _make_dataset("ds2", n_samples=5, n_features=5)

        cache.put("chain_a", "data_1", ds1)
        state1 = cache.get("chain_a", "data_1")

        if state1 is not None:
            # Remember the SharedBlocks refs before eviction
            refs = [snap[0] for snap in state1.source_snapshots]
            initial_refcounts = [r.refcount for r in refs]

            # Putting a second entry should evict the first
            cache.put("chain_b", "data_2", ds2)

            # The evicted state's refs should have been released
            # (refcount decremented by 1 relative to what it was after get)
            for ref, initial_rc in zip(refs, initial_refcounts):
                # The eviction callback calls release(), so refcount should drop
                assert ref.refcount < initial_rc

    def test_clear_releases_all_refs(self):
        """clear() releases SharedBlocks for all entries."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("clear_test", n_samples=5, n_features=5)

        cache.put("chain", "data", ds)
        state = cache.get("chain", "data")
        assert state is not None

        refs = [snap[0] for snap in state.source_snapshots]
        pre_clear_rcs = [r.refcount for r in refs]

        cache.clear()

        for ref, pre_rc in zip(refs, pre_clear_rcs):
            assert ref.refcount < pre_rc


# =========================================================================
# Processing names
# =========================================================================


class TestStepCacheProcessingNames:
    """Verify processing names are preserved through snapshot/restore."""

    def test_processing_names_restored(self):
        """Processing names from cached state are applied on restore."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("proc_test", n_samples=10, n_features=5)

        # Record original processing names
        original_procs = [
            list(ds.features_processings(i))
            for i in range(ds.features_sources())
        ]

        cache.put("chain", "data", ds)
        state = cache.get("chain", "data")
        assert state is not None
        assert state.processing_names == original_procs

        # Restore into a different dataset
        ds2 = _make_dataset("target", n_samples=10, n_features=5)
        cache.restore(state, ds2)

        restored_procs = [
            list(ds2.features_processings(i))
            for i in range(ds2.features_sources())
        ]
        assert restored_procs == original_procs


# =========================================================================
# Timing stats
# =========================================================================


class TestStepCacheTimingStats:
    """Verify timing instrumentation in cache stats."""

    def test_stats_include_timing_fields(self):
        """stats() includes timing breakdown fields."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("timing", n_samples=10, n_features=5)

        cache.put("chain", "data", ds)
        cache.get("chain", "data")

        stats = cache.stats()
        assert "total_snapshot_ms" in stats
        assert "total_restore_ms" in stats
        assert stats["total_snapshot_ms"] >= 0
        assert stats["total_restore_ms"] >= 0

    def test_record_hash_time_accumulates(self):
        """record_hash_time() accumulates in stats."""
        cache = StepCache(max_size_mb=100)
        cache.record_hash_time(0.001)
        cache.record_hash_time(0.002)
        assert cache._total_hash_s == pytest.approx(0.003, abs=1e-9)


# =========================================================================
# CachedStepState uses SharedBlocks (not FeatureSource deep copies)
# =========================================================================


class TestCachedStepStateStructure:
    """Verify CachedStepState uses CoW SharedBlocks internally."""

    def test_snapshot_produces_shared_blocks(self):
        """Snapshot creates source_snapshots with SharedBlocks tuples."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("struct", n_samples=10, n_features=5)

        cache.put("chain", "data", ds)
        state = cache.get("chain", "data")
        assert state is not None

        # Verify structure
        assert len(state.source_snapshots) == ds.features_sources()
        for snapshot in state.source_snapshots:
            shared, proc_ids, headers, header_unit = snapshot
            assert isinstance(shared, SharedBlocks)
            assert isinstance(proc_ids, list)
            assert all(isinstance(p, str) for p in proc_ids)

    def test_bytes_estimate_reasonable(self):
        """CachedStepState.bytes_estimate is a reasonable approximation."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("size_test", n_samples=100, n_features=200)
        cache.put("chain", "data", ds)
        state = cache.get("chain", "data")
        assert state is not None
        # float32 storage: 100 * 200 * 4 = 80,000 bytes
        assert state.bytes_estimate >= 50_000
        assert state.bytes_estimate < 200_000
