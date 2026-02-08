"""
Performance optimization module for dataset loading.

This module provides caching and memory-mapped file support.
"""

from .cache import (
    DataCache,
    CacheEntry,
)

__all__ = [
    "DataCache",
    "CacheEntry",
]
