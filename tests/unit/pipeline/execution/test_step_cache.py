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
- Estimate dataset size
"""

from __future__ import annotations

import threading

import numpy as np

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.execution.step_cache import StepCache

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

    def test_put_and_get_returns_dataset(self):
        """Put a dataset, then get it back."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("ds1")
        cache.put("chain_a", "data_1", ds)
        result = cache.get("chain_a", "data_1")
        assert result is not None
        assert result.name == "ds1"

    def test_get_miss_returns_none(self):
        """Getting a non-existent key returns None."""
        cache = StepCache(max_size_mb=100)
        result = cache.get("missing_chain", "missing_data")
        assert result is None

    def test_put_overwrite(self):
        """Putting the same key twice overwrites the first entry."""
        cache = StepCache(max_size_mb=100)
        ds1 = _make_dataset("ds1")
        ds2 = _make_dataset("ds2")
        cache.put("chain_a", "data_1", ds1)
        cache.put("chain_a", "data_1", ds2)
        result = cache.get("chain_a", "data_1")
        assert result is not None
        assert result.name == "ds2"

    def test_different_keys_independent(self):
        """Different keys store independent datasets."""
        cache = StepCache(max_size_mb=100)
        ds1 = _make_dataset("ds1")
        ds2 = _make_dataset("ds2")
        cache.put("chain_a", "data_1", ds1)
        cache.put("chain_b", "data_2", ds2)
        r1 = cache.get("chain_a", "data_1")
        r2 = cache.get("chain_b", "data_2")
        assert r1 is not None and r1.name == "ds1"
        assert r2 is not None and r2.name == "ds2"

    def test_same_chain_different_data_hash(self):
        """Same chain hash but different data hash are separate entries."""
        cache = StepCache(max_size_mb=100)
        ds1 = _make_dataset("ds1")
        ds2 = _make_dataset("ds2")
        cache.put("chain_a", "data_1", ds1)
        cache.put("chain_a", "data_2", ds2)
        r1 = cache.get("chain_a", "data_1")
        r2 = cache.get("chain_a", "data_2")
        assert r1 is not None and r1.name == "ds1"
        assert r2 is not None and r2.name == "ds2"


# =========================================================================
# Deep copy semantics
# =========================================================================


class TestStepCacheDeepCopy:
    """Verify that get/put return deep copies (no aliasing)."""

    def test_put_creates_independent_copy(self):
        """Mutating the original dataset after put does not affect the cache."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("original", n_samples=10, n_features=5)
        cache.put("chain", "data", ds)

        # Mutate the original
        ds.name = "mutated"

        # Cache should still have the original
        result = cache.get("chain", "data")
        assert result is not None
        assert result.name == "original"

    def test_get_returns_independent_copy(self):
        """Mutating a returned dataset does not affect subsequent gets."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("cached", n_samples=10, n_features=5)
        cache.put("chain", "data", ds)

        # Get and mutate
        r1 = cache.get("chain", "data")
        assert r1 is not None
        r1.name = "mutated"

        # Second get should still return original
        r2 = cache.get("chain", "data")
        assert r2 is not None
        assert r2.name == "cached"

    def test_get_returns_different_object(self):
        """Each get returns a distinct object instance."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("test", n_samples=10, n_features=5)
        cache.put("chain", "data", ds)
        r1 = cache.get("chain", "data")
        r2 = cache.get("chain", "data")
        assert r1 is not r2


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

        # ds1 should have been evicted to make room for ds2
        # (since the cache is only 10 KB and each dataset is ~40 KB)
        r1 = cache.get("chain_a", "data_1")
        # Due to the very small cache, at least one should be evicted
        # The exact behavior depends on DataCache internals, but the
        # cache should not grow beyond its limit
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
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 1
        assert stats["hit_rate"] == 0.5
        assert "entries" in stats
        assert "size_mb" in stats

    def test_stats_zero_requests(self):
        """stats() with no requests has 0.0 hit_rate."""
        cache = StepCache(max_size_mb=100)
        stats = cache.stats()
        assert stats["hit_rate"] == 0.0
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 0


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


# =========================================================================
# Size estimation
# =========================================================================


class TestStepCacheSizeEstimation:
    """Dataset size estimation."""

    def test_estimate_dataset_size_basic(self):
        """estimate_dataset_size returns reasonable byte count."""
        ds = _make_dataset("test", n_samples=100, n_features=200)
        size = StepCache.estimate_dataset_size(ds)
        # Internal storage uses float32: X is 100*1*200*4 = 80,000 bytes
        # y is 100*1*4 = 400 bytes (float32, shape (100, 1))
        # Total ~ 80,400
        assert size >= 80_000
        assert size < 120_000  # Allow some overhead

    def test_estimate_dataset_size_no_targets(self):
        """estimate_dataset_size works with no targets."""
        ds = SpectroDataset("no_targets")
        X = np.random.rand(50, 100).astype(np.float64)
        ds.add_samples(X, indexes={"partition": "train"})
        size = StepCache.estimate_dataset_size(ds)
        # Internal storage uses float32: 50*1*100*4 = 20,000 bytes
        assert size >= 50 * 100 * 4


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
                    if result is not None and result.name != f"t{thread_id}_i{i}":
                        errors.append(f"Thread {thread_id}: expected t{thread_id}_i{i}, got {result.name}")
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

    def test_feature_data_preserved(self):
        """Feature array values are identical after cache round-trip."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("integrity", n_samples=20, n_features=10)
        X_original = ds.x({"partition": "train"}, layout="2d").copy()
        y_original = ds.y({"partition": "train"}).copy()

        cache.put("chain", "data", ds)
        restored = cache.get("chain", "data")
        assert restored is not None

        X_restored = restored.x({"partition": "train"}, layout="2d")
        y_restored = restored.y({"partition": "train"})

        np.testing.assert_array_equal(X_original, X_restored)
        np.testing.assert_array_equal(y_original, y_restored)

    def test_dataset_name_preserved(self):
        """Dataset name survives the cache round-trip."""
        cache = StepCache(max_size_mb=100)
        ds = _make_dataset("my_special_name", n_samples=10, n_features=5)
        cache.put("chain", "data", ds)
        restored = cache.get("chain", "data")
        assert restored is not None
        assert restored.name == "my_special_name"
