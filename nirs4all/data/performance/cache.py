"""
Data caching for dataset loading.

This module provides caching functionality to avoid redundant file loading
and improve performance for repeated data access.

Phase 8 Implementation - Dataset Configuration Roadmap
Section 8.5: Performance Optimization - Caching
"""

import hashlib
import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

import numpy as np

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry:
    """A cached data entry.

    Attributes:
        data: The cached data.
        key: Cache key.
        timestamp: When the data was cached.
        size_bytes: Estimated size in bytes.
        source_path: Original file path (if applicable).
        source_mtime: Modification time of source file.
        hit_count: Number of times this entry was accessed.
    """
    data: Any
    key: str
    timestamp: float = field(default_factory=time.time)
    size_bytes: int = 0
    source_path: Optional[str] = None
    source_mtime: Optional[float] = None
    hit_count: int = 0

    def is_stale(self) -> bool:
        """Check if entry is stale (source file modified)."""
        if self.source_path is None or self.source_mtime is None:
            return False
        try:
            current_mtime = Path(self.source_path).stat().st_mtime
            return current_mtime > self.source_mtime
        except OSError:
            return True  # File doesn't exist, consider stale


class DataCache:
    """LRU cache for loaded data.

    Provides in-memory caching with:
    - Configurable size limits
    - LRU eviction policy
    - File modification detection
    - Thread-safe access
    - Cache statistics

    Example:
        ```python
        cache = DataCache(max_size_mb=500)

        # Store data
        cache.set("my_data", numpy_array, source_path="/path/to/file.csv")

        # Retrieve data
        data = cache.get("my_data")

        # With automatic loading
        data = cache.get_or_load("key", lambda: load_expensive_data())

        # Check stats
        print(cache.stats())
        ```
    """

    def __init__(
        self,
        max_size_mb: float = 500,
        max_entries: int = 100,
        ttl_seconds: Optional[float] = None,
    ):
        """Initialize cache.

        Args:
            max_size_mb: Maximum cache size in megabytes.
            max_entries: Maximum number of entries.
            ttl_seconds: Time-to-live for entries (None = no expiry).
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds

        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._total_size = 0
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get data from cache.

        Args:
            key: Cache key.

        Returns:
            Cached data or None if not found.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            # Check if stale
            if entry.is_stale():
                self._remove(key)
                self._misses += 1
                return None

            # Check TTL
            if self.ttl_seconds is not None:
                age = time.time() - entry.timestamp
                if age > self.ttl_seconds:
                    self._remove(key)
                    self._misses += 1
                    return None

            entry.hit_count += 1
            self._hits += 1
            return entry.data

    def set(
        self,
        key: str,
        data: Any,
        source_path: Optional[str] = None,
    ) -> None:
        """Store data in cache.

        Args:
            key: Cache key.
            data: Data to cache.
            source_path: Optional source file path for staleness detection.
        """
        size = self._estimate_size(data)

        # Don't cache if larger than max size
        if size > self.max_size_bytes:
            logger.debug(f"Data too large to cache: {size / 1024 / 1024:.1f} MB")
            return

        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._remove(key)

            # Evict if needed
            self._evict_if_needed(size)

            # Get source file mtime
            source_mtime = None
            if source_path:
                try:
                    source_mtime = Path(source_path).stat().st_mtime
                except OSError:
                    pass

            # Store entry
            entry = CacheEntry(
                data=data,
                key=key,
                size_bytes=size,
                source_path=source_path,
                source_mtime=source_mtime,
            )
            self._cache[key] = entry
            self._total_size += size

    def get_or_load(
        self,
        key: str,
        loader: Callable[[], T],
        source_path: Optional[str] = None,
    ) -> T:
        """Get from cache or load and cache.

        Args:
            key: Cache key.
            loader: Function to call if not cached.
            source_path: Optional source file path.

        Returns:
            Cached or newly loaded data.
        """
        data = self.get(key)
        if data is not None:
            return data

        # Load data
        data = loader()

        # Cache it
        self.set(key, data, source_path)

        return data

    def invalidate(self, key: str) -> bool:
        """Remove entry from cache.

        Args:
            key: Cache key.

        Returns:
            True if entry was removed.
        """
        with self._lock:
            if key in self._cache:
                self._remove(key)
                return True
            return False

    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._total_size = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0

            return {
                "entries": len(self._cache),
                "max_entries": self.max_entries,
                "size_mb": self._total_size / 1024 / 1024,
                "max_size_mb": self.max_size_bytes / 1024 / 1024,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }

    def _remove(self, key: str) -> None:
        """Remove entry from cache (internal, assumes lock held)."""
        entry = self._cache.pop(key, None)
        if entry:
            self._total_size -= entry.size_bytes

    def _evict_if_needed(self, new_size: int) -> None:
        """Evict entries if needed (internal, assumes lock held)."""
        # Evict based on entry count
        while len(self._cache) >= self.max_entries:
            self._evict_lru()

        # Evict based on size
        while self._total_size + new_size > self.max_size_bytes and self._cache:
            self._evict_lru()

    def _evict_lru(self) -> None:
        """Evict least recently used entry (internal, assumes lock held)."""
        if not self._cache:
            return

        # Find LRU entry (oldest timestamp, lowest hit count)
        lru_key = min(
            self._cache.keys(),
            key=lambda k: (self._cache[k].hit_count, self._cache[k].timestamp)
        )
        self._remove(lru_key)
        logger.debug(f"Evicted cache entry: {lru_key}")

    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data."""
        if isinstance(data, np.ndarray):
            return data.nbytes

        if isinstance(data, (list, tuple)):
            return sum(self._estimate_size(item) for item in data)

        if isinstance(data, dict):
            return sum(
                self._estimate_size(k) + self._estimate_size(v)
                for k, v in data.items()
            )

        try:
            return len(pickle.dumps(data))
        except Exception:
            return 1000  # Default estimate


def make_cache_key(
    path: Union[str, Path],
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a cache key from path and parameters.

    Args:
        path: File path.
        params: Loading parameters.

    Returns:
        Hash-based cache key.
    """
    key_data = str(path)
    if params:
        # Sort params for consistent hashing
        param_str = str(sorted(params.items()))
        key_data += param_str

    return hashlib.md5(key_data.encode()).hexdigest()


# Global cache instance
_cache: Optional[DataCache] = None


def cache_manager(max_size_mb: float = 500) -> DataCache:
    """Get or create the global cache instance.

    Args:
        max_size_mb: Maximum cache size (only used when creating).

    Returns:
        DataCache instance.
    """
    global _cache
    if _cache is None:
        _cache = DataCache(max_size_mb=max_size_mb)
    return _cache
