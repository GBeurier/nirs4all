"""
Performance optimization module for dataset loading.

This module provides lazy loading, caching, and memory-mapped file support.
"""

from .cache import (
    DataCache,
    CacheEntry,
    cache_manager,
)

from .lazy_loader import (
    LazyDataset,
    LazyArray,
)

__all__ = [
    "DataCache",
    "CacheEntry",
    "cache_manager",
    "LazyDataset",
    "LazyArray",
]
