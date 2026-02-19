"""
Unit tests for PredictionCache.
"""
import pytest

from nirs4all.visualization.prediction_cache import CacheKey, PredictionCache


class TestCacheKey:
    """Tests for CacheKey class."""

    def test_cache_key_creation(self):
        """Test basic cache key creation."""
        key = CacheKey(
            aggregate='ID',
            rank_metric='rmse',
            rank_partition='val',
            display_partition='test',
            group_by=('model_name',),
            filters=(('dataset_name', 'wheat'),)
        )
        assert key is not None

    def test_cache_key_equality(self):
        """Test cache key equality."""
        key1 = CacheKey(
            aggregate='ID',
            rank_metric='rmse',
            rank_partition='val',
            display_partition='test',
            group_by=('model_name',),
            filters=(('dataset_name', 'wheat'),)
        )
        key2 = CacheKey(
            aggregate='ID',
            rank_metric='rmse',
            rank_partition='val',
            display_partition='test',
            group_by=('model_name',),
            filters=(('dataset_name', 'wheat'),)
        )
        assert key1 == key2
        assert hash(key1) == hash(key2)

    def test_cache_key_inequality(self):
        """Test cache key inequality."""
        key1 = CacheKey(
            aggregate='ID',
            rank_metric='rmse',
            rank_partition='val',
            display_partition='test',
            group_by=('model_name',),
            filters=(('dataset_name', 'wheat'),)
        )
        key2 = CacheKey(
            aggregate='ID',
            rank_metric='r2',  # Different metric
            rank_partition='val',
            display_partition='test',
            group_by=('model_name',),
            filters=(('dataset_name', 'wheat'),)
        )
        assert key1 != key2

class TestPredictionCache:
    """Tests for PredictionCache class."""

    def test_cache_initialization(self):
        """Test cache initializes properly."""
        cache = PredictionCache(max_entries=10)
        assert len(cache) == 0
        stats = cache.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0

    def test_cache_put_get(self):
        """Test basic put and get operations."""
        cache = PredictionCache()
        key = PredictionCache.make_key(
            aggregate='ID',
            rank_metric='rmse'
        )
        cache.put(key, ['result1', 'result2'])
        result = cache.get(key)
        assert result == ['result1', 'result2']
        assert cache.get_stats()['hits'] == 1

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = PredictionCache()
        key = PredictionCache.make_key(
            aggregate='ID',
            rank_metric='rmse'
        )
        result = cache.get(key)
        assert result is None
        assert cache.get_stats()['misses'] == 1

    def test_get_or_compute(self):
        """Test get_or_compute caches on miss."""
        cache = PredictionCache()
        key = PredictionCache.make_key(
            aggregate='ID',
            rank_metric='rmse'
        )

        compute_count = 0

        def compute():
            nonlocal compute_count
            compute_count += 1
            return ['computed_result']

        # First call should compute
        result1 = cache.get_or_compute(key, compute)
        assert result1 == ['computed_result']
        assert compute_count == 1

        # Second call should use cache
        result2 = cache.get_or_compute(key, compute)
        assert result2 == ['computed_result']
        assert compute_count == 1  # Should not have computed again

        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1

    def test_cache_eviction(self):
        """Test LRU eviction when max size exceeded."""
        cache = PredictionCache(max_entries=2)

        key1 = PredictionCache.make_key(aggregate='ID', rank_metric='rmse')
        key2 = PredictionCache.make_key(aggregate='ID', rank_metric='r2')
        key3 = PredictionCache.make_key(aggregate='ID', rank_metric='mae')

        cache.put(key1, 'result1')
        cache.put(key2, 'result2')
        assert len(cache) == 2

        # This should evict key1 (oldest)
        cache.put(key3, 'result3')
        assert len(cache) == 2

        # key1 should be evicted
        assert cache.get(key1) is None
        # key2 and key3 should still be there
        assert cache.get(key2) == 'result2'
        assert cache.get(key3) == 'result3'

        assert cache.get_stats()['evictions'] == 1

    def test_make_key_normalizes_group_by(self):
        """Test make_key normalizes group_by parameter."""
        # String should become tuple
        key1 = PredictionCache.make_key(
            aggregate='ID',
            rank_metric='rmse',
            group_by='model_name'
        )
        key2 = PredictionCache.make_key(
            aggregate='ID',
            rank_metric='rmse',
            group_by=['model_name']
        )
        assert key1 == key2

    def test_make_key_sorts_filters(self):
        """Test make_key sorts filters for consistency."""
        key1 = PredictionCache.make_key(
            aggregate='ID',
            rank_metric='rmse',
            dataset_name='wheat',
            model_name='PLS'
        )
        key2 = PredictionCache.make_key(
            aggregate='ID',
            rank_metric='rmse',
            model_name='PLS',
            dataset_name='wheat'
        )
        assert key1 == key2

    def test_clear(self):
        """Test cache clear empties the cache."""
        cache = PredictionCache()
        key = PredictionCache.make_key(aggregate='ID', rank_metric='rmse')
        cache.put(key, 'result')
        assert len(cache) == 1

        cache.clear()
        assert len(cache) == 0
        assert cache.get(key) is None
