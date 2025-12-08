"""
PredictionCache - Caching layer for expensive prediction computations.

This module provides caching for aggregated predictions to avoid
redundant computations when multiple charts use the same data.
"""
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union


class CacheKey:
    """Immutable cache key for prediction queries."""

    __slots__ = ('_hash', '_repr')

    def __init__(
        self,
        aggregate: Optional[str],
        rank_metric: str,
        rank_partition: str,
        display_partition: str,
        group_by: Optional[Tuple[str, ...]],
        filters: Tuple[Tuple[str, Any], ...]
    ):
        """Create a cache key from query parameters.

        Args:
            aggregate: Aggregation column name (e.g., 'ID') or None.
            rank_metric: Metric used for ranking.
            rank_partition: Partition used for ranking.
            display_partition: Partition for display data.
            group_by: Tuple of column names for grouping.
            filters: Tuple of (key, value) filter pairs.
        """
        # Create a hashable representation
        self._repr = (
            aggregate,
            rank_metric,
            rank_partition,
            display_partition,
            group_by,
            filters
        )
        self._hash = hash(self._repr)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, CacheKey):
            return False
        return self._repr == other._repr

    def __repr__(self):
        return f"CacheKey({self._repr})"


class PredictionCache:
    """LRU-style cache for prediction query results.

    Caches expensive computations like aggregations to speed up
    repeated chart rendering with the same parameters.

    The cache stores:
    - Aggregated prediction results (keyed by aggregate + filters)
    - Top-k results (keyed by full query parameters)

    Cache invalidation happens when:
    - Max size is exceeded (LRU eviction)
    - clear() is called explicitly
    - Predictions data is modified (user must call clear())

    Example:
        >>> cache = PredictionCache(max_entries=100)
        >>> # First call computes and caches
        >>> result = cache.get_or_compute(key, compute_fn)
        >>> # Second call returns cached result
        >>> result = cache.get_or_compute(key, compute_fn)  # Fast!
    """

    def __init__(self, max_entries: int = 50):
        """Initialize the cache.

        Args:
            max_entries: Maximum number of cached entries before LRU eviction.
        """
        self._cache: Dict[CacheKey, Any] = {}
        self._access_order: List[CacheKey] = []
        self._max_entries = max_entries
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_compute_time': 0.0
        }

    @staticmethod
    def make_key(
        aggregate: Optional[str],
        rank_metric: str,
        rank_partition: str = 'val',
        display_partition: str = 'test',
        group_by: Optional[Union[str, List[str]]] = None,
        **filters
    ) -> CacheKey:
        """Create a cache key from query parameters.

        Args:
            aggregate: Aggregation column name or None.
            rank_metric: Metric for ranking.
            rank_partition: Partition for ranking.
            display_partition: Partition for display.
            group_by: Grouping column(s).
            **filters: Additional filter criteria.

        Returns:
            CacheKey for the query.
        """
        # Normalize group_by to tuple
        if group_by is None:
            group_by_tuple = None
        elif isinstance(group_by, str):
            group_by_tuple = (group_by,)
        else:
            group_by_tuple = tuple(sorted(group_by))

        # Normalize filters to sorted tuple of tuples
        # Convert values to strings for hashability
        filter_items = []
        for k, v in sorted(filters.items()):
            if v is None:
                continue
            # Convert lists to tuples for hashability
            if isinstance(v, list):
                v = tuple(v)
            filter_items.append((k, v))

        return CacheKey(
            aggregate=aggregate,
            rank_metric=rank_metric,
            rank_partition=rank_partition,
            display_partition=display_partition,
            group_by=group_by_tuple,
            filters=tuple(filter_items)
        )

    def get(self, key: CacheKey) -> Optional[Any]:
        """Get cached result if available.

        Args:
            key: Cache key to look up.

        Returns:
            Cached result or None if not found.
        """
        if key in self._cache:
            self._stats['hits'] += 1
            # Update access order (move to end)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]

        self._stats['misses'] += 1
        return None

    def put(self, key: CacheKey, value: Any) -> None:
        """Store result in cache.

        Args:
            key: Cache key.
            value: Result to cache.
        """
        # Evict oldest entries if at capacity
        while len(self._cache) >= self._max_entries:
            self._evict_oldest()

        self._cache[key] = value
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def get_or_compute(self, key: CacheKey, compute_fn: callable) -> Any:
        """Get cached result or compute and cache.

        This is the primary method for cache usage. It handles
        the cache lookup, computation, and storage in one call.

        Args:
            key: Cache key.
            compute_fn: Function to call if cache miss (no args).

        Returns:
            Result from cache or computation.
        """
        result = self.get(key)
        if result is not None:
            return result

        # Compute and cache
        t0 = time.time()
        result = compute_fn()
        compute_time = time.time() - t0
        self._stats['total_compute_time'] += compute_time

        self.put(key, result)
        return result

    def _evict_oldest(self) -> None:
        """Evict the least recently used entry."""
        if self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]
                self._stats['evictions'] += 1

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with hits, misses, evictions, hit_rate, etc.
        """
        total = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total if total > 0 else 0.0

        return {
            **self._stats,
            'total_requests': total,
            'hit_rate': hit_rate,
            'current_size': len(self._cache),
            'max_size': self._max_entries
        }

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"PredictionCache({stats['current_size']}/{stats['max_size']} entries, {stats['hit_rate']:.1%} hit rate)"
