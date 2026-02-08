"""Tests for StepCache: in-memory preprocessed data snapshots.

Covers:
- Basic get/put operations
- Cache miss returns None
- Deep copy semantics (no aliasing)
- LRU eviction under memory pressure
- Memory limit enforcement
- Thread safety
- Statistics / observability
- Clear operation
- Key composition
- Restore into dataset
- Selector fingerprint isolation
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

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
# Basic get/put
# =========================================================================


class TestStepCacheBasic:
    """Basic cache operations."""

    def test_put_and_get_returns_cached_state(self):
        """Put a dataset, then get returns a CachedStepState."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("ds1")
        cache.put("chain_a", "data_1", ds)
        result = cache.get("chain_a", "data_1")
        assert result is not None
        assert isinstance(result, CachedStepState)
        assert result.content_hash == ds.content_hash()

    def test_get_miss_returns_none(self):
        """Getting a non-existent key returns None."""
        cache = StepCache(max_size_mb=100)
        result = cache.get("missing_chain", "missing_data")
        assert result is None

    def test_put_overwrite(self):
        """Putting the same key twice overwrites the first entry."""
        cache = StepCache(max_size_mb=100)
        ds1 = _make_dataset("ds1", n_features=50)
        ds2 = _make_dataset("ds2", n_features=80)
        cache.put("chain_a", "data_1", ds1)
        cache.put("chain_a", "data_1", ds2)
        result = cache.get("chain_a", "data_1")
        assert result is not None
        assert result.content_hash == ds2.content_hash()

    def test_different_keys_independent(self):
        """Different keys store independent snapshots."""
        cache = StepCache(max_size_mb=100)
        ds1 = _make_dataset("ds1", n_features=50)
        ds2 = _make_dataset("ds2", n_features=80)
        cache.put("chain_a", "data_1", ds1)
        cache.put("chain_b", "data_2", ds2)
        r1 = cache.get("chain_a", "data_1")
        r2 = cache.get("chain_b", "data_2")
        assert r1 is not None and r1.content_hash == ds1.content_hash()
        assert r2 is not None and r2.content_hash == ds2.content_hash()

    def test_same_chain_different_data_hash(self):
        """Same chain hash but different data hash are separate entries."""
        cache = StepCache(max_size_mb=100)
        ds1 = _make_dataset("ds1", n_features=50)
        ds2 = _make_dataset("ds2", n_features=80)
        cache.put("chain_a", "data_1", ds1)
        cache.put("chain_a", "data_2", ds2)
        r1 = cache.get("chain_a", "data_1")
        r2 = cache.get("chain_a", "data_2")
        assert r1 is not None and r1.content_hash == ds1.content_hash()
        assert r2 is not None and r2.content_hash == ds2.content_hash()


# =========================================================================
# Deep copy semantics
# =========================================================================


class TestStepCacheDeepCopy:
    """Verify that get/put return deep copies (no aliasing)."""

    def test_put_creates_independent_copy(self):
        """Mutating the original dataset after put does not affect the cache."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("original", n_samples=10, n_features=5)
        original_hash = ds.content_hash()
        cache.put("chain", "data", ds)

        # Mutate the original dataset's feature data
        X = ds.x({"partition": "train"}, layout="2d")
        X[:] = 999.0

        # Cache should still have the original data
        result = cache.get("chain", "data")
        assert result is not None
        assert result.content_hash == original_hash

    def test_restore_produces_independent_data(self):
        """Two restores into different datasets produce independent arrays."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("test", n_samples=10, n_features=5)
        cache.put("chain", "data", ds)

        state = cache.get("chain", "data")
        assert state is not None

        ds1 = _make_dataset("target1", n_samples=10, n_features=5)
        ds2 = _make_dataset("target2", n_samples=10, n_features=5)
        cache.restore(state, ds1)
        cache.restore(state, ds2)

        # Mutating ds1 features should not affect ds2
        X1 = ds1.x({"partition": "train"}, layout="2d")
        X2 = ds2.x({"partition": "train"}, layout="2d")
        X1[:] = 999.0
        assert not np.allclose(X2, 999.0)


# =========================================================================
# Restore
# =========================================================================


class TestStepCacheRestore:
    """Verify that restore correctly applies cached state to a dataset."""

    def test_restore_feature_data(self):
        """Restoring a cached state replaces dataset feature data."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("test", n_samples=20, n_features=10)
        X_original = ds.x({"partition": "train"}, layout="2d").copy()

        cache.put("chain", "data", ds)

        # Create a new dataset with different data
        ds2 = _make_dataset("test2", n_samples=20, n_features=10)

        # Restore cached state into ds2
        state = cache.get("chain", "data")
        assert state is not None
        cache.restore(state, ds2)

        X_restored = ds2.x({"partition": "train"}, layout="2d")
        np.testing.assert_array_equal(X_original, X_restored)

    def test_restore_updates_content_hash(self):
        """Restoring updates the dataset's content hash cache."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("test", n_samples=10, n_features=5)
        original_hash = ds.content_hash()
        cache.put("chain", "data", ds)

        ds2 = _make_dataset("other", n_samples=10, n_features=5)
        state = cache.get("chain", "data")
        cache.restore(state, ds2)

        assert ds2._content_hash_cache == original_hash


# =========================================================================
# LRU eviction
# =========================================================================


class TestStepCacheEviction:
    """LRU eviction under memory pressure."""

    def test_eviction_when_over_limit(self):
        """Entries are evicted when total size exceeds max_size_mb."""
        # Use a very small cache (0.01 MB ~ 10 KB)
        cache = StepCache(max_size_mb=0.01)

        # Each dataset is ~40 KB (50*100*8 bytes for X + 50*8 for y)
        ds1 = _make_dataset("ds1", n_samples=50, n_features=100)
        ds2 = _make_dataset("ds2", n_samples=50, n_features=100)

        cache.put("chain_a", "data_1", ds1)
        cache.put("chain_b", "data_2", ds2)

        # Due to the very small cache, at least one should be evicted
        stats = cache.stats()
        assert stats["size_mb"] <= 0.05  # Allow some overhead

    def test_large_dataset_not_cached(self):
        """Datasets larger than max_size_mb are not cached at all."""
        cache = StepCache(max_size_mb=0.001)  # ~1 KB limit
        # This dataset is ~80 KB
        ds = _make_dataset("large", n_samples=100, n_features=100)
        cache.put("chain", "data", ds)
        # The backend's DataCache.set() skips items larger than max_size
        result = cache.get("chain", "data")
        # Should not be cached (too large)
        assert result is None


# =========================================================================
# Statistics and observability
# =========================================================================


class TestStepCacheStats:
    """Cache statistics tracking."""

    def test_hit_count_increments_on_hit(self):
        """hit_count increments on successful get."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("test", n_samples=10, n_features=5)
        cache.put("chain", "data", ds)
        assert cache.hit_count == 0
        cache.get("chain", "data")
        assert cache.hit_count == 1
        cache.get("chain", "data")
        assert cache.hit_count == 2

    def test_miss_count_increments_on_miss(self):
        """miss_count increments on failed get."""
        cache = StepCache(max_size_mb=100)
        assert cache.miss_count == 0
        cache.get("missing", "key")
        assert cache.miss_count == 1
        cache.get("another", "miss")
        assert cache.miss_count == 2

    def test_hit_count_not_incremented_on_miss(self):
        """hit_count should not change on a miss."""
        cache = StepCache(max_size_mb=100)
        cache.get("missing", "key")
        assert cache.hit_count == 0

    def test_stats_returns_complete_dict(self):
        """stats() returns expected keys."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("test", n_samples=10, n_features=5)
        cache.put("chain", "data", ds)
        cache.get("chain", "data")  # hit
        cache.get("missing", "key")  # miss

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert "entries" in stats
        assert "size_mb" in stats
        assert "evictions" in stats
        assert "peak_mb" in stats

    def test_stats_zero_requests(self):
        """stats() with no requests has 0.0 hit_rate."""
        cache = StepCache(max_size_mb=100)
        stats = cache.stats()
        assert stats["hit_rate"] == 0.0
        assert stats["hits"] == 0
        assert stats["misses"] == 0


# =========================================================================
# Clear
# =========================================================================


class TestStepCacheClear:
    """Cache clearing."""

    def test_clear_removes_all_entries(self):
        """clear() empties the cache."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("test", n_samples=10, n_features=5)
        cache.put("chain_a", "data_1", ds)
        cache.put("chain_b", "data_2", ds)
        cache.clear()
        assert cache.get("chain_a", "data_1") is None
        assert cache.get("chain_b", "data_2") is None

    def test_clear_resets_counters(self):
        """clear() resets hit and miss counters."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("test", n_samples=10, n_features=5)
        cache.put("chain", "data", ds)
        cache.get("chain", "data")  # hit
        cache.get("missing", "key")  # miss
        cache.clear()
        assert cache.hit_count == 0
        assert cache.miss_count == 0


# =========================================================================
# Key composition
# =========================================================================


class TestStepCacheKeyComposition:
    """Internal key building."""

    def test_make_key_format(self):
        """_make_key returns expected composite format."""
        key = StepCache._make_key("abc123", "def456")
        assert key == "step:abc123:def456"

    def test_make_key_different_inputs(self):
        """Different inputs produce different keys."""
        k1 = StepCache._make_key("chain_a", "data_1")
        k2 = StepCache._make_key("chain_a", "data_2")
        k3 = StepCache._make_key("chain_b", "data_1")
        assert k1 != k2
        assert k1 != k3
        assert k2 != k3

    def test_make_key_with_selector_includes_fingerprint(self):
        """_make_key with a selector appends the selector fingerprint."""
        from nirs4all.pipeline.config.context import DataSelector
        selector = DataSelector(partition="train", fold_id="fold_0")
        key_with = StepCache._make_key("chain_a", "data_1", selector)
        key_without = StepCache._make_key("chain_a", "data_1")
        assert key_with != key_without
        # Key with selector has 4 colon-separated parts
        assert key_with.count(":") == 3

    def test_different_selectors_different_keys(self):
        """Different selectors produce different keys."""
        from nirs4all.pipeline.config.context import DataSelector
        sel1 = DataSelector(partition="train", fold_id="fold_0")
        sel2 = DataSelector(partition="train", fold_id="fold_1")
        k1 = StepCache._make_key("chain_a", "data_1", sel1)
        k2 = StepCache._make_key("chain_a", "data_1", sel2)
        assert k1 != k2


# =========================================================================
# Thread safety
# =========================================================================


class TestStepCacheThreadSafety:
    """Concurrent access to the cache."""

    def test_concurrent_put_get(self):
        """Multiple threads can put and get without corruption."""
        cache = StepCache(max_size_mb=200)
        errors: list[str] = []
        n_threads = 8
        n_ops = 20

        def worker(thread_id: int) -> None:
            try:
                for i in range(n_ops):
                    ds = _make_dataset(f"t{thread_id}_i{i}", n_samples=10, n_features=5)
                    cache.put(f"chain_{thread_id}", f"data_{i}", ds)
                    result = cache.get(f"chain_{thread_id}", f"data_{i}")
                    if result is not None:
                        # Verify it's a valid CachedStepState
                        assert isinstance(result, CachedStepState)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"

    def test_concurrent_clear(self):
        """Clearing while other threads read/write doesn't crash."""
        cache = StepCache(max_size_mb=100)
        errors: list[str] = []

        def writer() -> None:
            try:
                for i in range(50):
                    ds = _make_dataset(f"w{i}", n_samples=5, n_features=5)
                    cache.put("chain", f"data_{i}", ds)
            except Exception as e:
                errors.append(f"writer: {e}")

        def reader() -> None:
            try:
                for i in range(50):
                    cache.get("chain", f"data_{i}")
            except Exception as e:
                errors.append(f"reader: {e}")

        def clearer() -> None:
            try:
                for _ in range(10):
                    cache.clear()
            except Exception as e:
                errors.append(f"clearer: {e}")

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=clearer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"


# =========================================================================
# Feature data integrity
# =========================================================================


class TestStepCacheDataIntegrity:
    """Verify that cached feature data is correctly preserved."""

    def test_feature_data_preserved_via_restore(self):
        """Feature array values are identical after cache round-trip via restore."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("integrity", n_samples=20, n_features=10)
        X_original = ds.x({"partition": "train"}, layout="2d").copy()

        cache.put("chain", "data", ds)

        # Restore into same dataset
        state = cache.get("chain", "data")
        assert state is not None
        cache.restore(state, ds)

        X_after = ds.x({"partition": "train"}, layout="2d")
        np.testing.assert_array_equal(X_original, X_after)

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
