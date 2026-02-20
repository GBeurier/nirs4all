"""Memory estimation and RSS tracking utilities.

Provides fast, numpy-first memory estimation for datasets and cache entries,
plus process RSS tracking via /proc/self/statm (Linux) or psutil (macOS).

Phase 1.4 — Cache Implementation Backlog
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset

def estimate_dataset_bytes(dataset: SpectroDataset) -> int:
    """Estimate total memory footprint of a SpectroDataset.

    Sums numpy nbytes across all feature sources and target processings.
    O(1) per array — reads shape and dtype only, no serialization.

    Args:
        dataset: The SpectroDataset to measure.

    Returns:
        Estimated byte count (features + targets).
    """
    total = 0

    # Feature sources: block-based storage with per-processing 2D blocks
    for src in dataset._features.sources:
        total += src._storage.nbytes

    # Target processings: dict of 1D/2D numpy arrays keyed by processing name
    for arr in dataset._targets._data.values():
        total += arr.nbytes

    return total

def estimate_cache_entry_bytes(entry: Any) -> int:
    """Estimate memory size of a cache entry.

    Uses numpy nbytes for arrays and SpectroDatasets (O(1)).
    Falls back to sys.getsizeof for other types.

    Args:
        entry: The object to measure. Can be a numpy array, a SpectroDataset,
            a tuple/list of arrays, or any other object.

    Returns:
        Estimated byte count.
    """
    if isinstance(entry, np.ndarray):
        return entry.nbytes

    # Avoid circular import — check by class name
    cls_name = type(entry).__name__
    if cls_name == 'SpectroDataset':
        return estimate_dataset_bytes(entry)

    if isinstance(entry, (list, tuple)):
        total = 0
        for item in entry:
            total += estimate_cache_entry_bytes(item)
        return total

    if isinstance(entry, dict):
        total = 0
        for k, v in entry.items():
            total += estimate_cache_entry_bytes(k) + estimate_cache_entry_bytes(v)
        return total

    return sys.getsizeof(entry)

def get_process_rss_mb() -> float:
    """Get the current process Resident Set Size in megabytes.

    Reads /proc/self/statm on Linux (no dependencies).
    Falls back to psutil on other platforms.

    Returns:
        RSS in MB, or 0.0 if measurement is unavailable.
    """
    # Linux fast path: read /proc/self/statm
    try:
        with open('/proc/self/statm') as f:
            parts = f.read().split()
        # Field 1 (index 1) is RSS in pages
        rss_pages = int(parts[1])
        import resource
        page_size = resource.getpagesize()  # type: ignore[attr-defined]
        return float((rss_pages * page_size) / (1024 * 1024))
    except (OSError, IndexError, ValueError):
        pass

    # Fallback: psutil (macOS, Windows)
    try:
        import psutil
        process = psutil.Process()
        return float(process.memory_info().rss / (1024 * 1024))
    except (ImportError, Exception):
        return 0.0

def format_bytes(n: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        n: Number of bytes.

    Returns:
        Formatted string, e.g. "152.6 MB", "1.2 GB", "800 B".
    """
    if n < 0:
        return f"-{format_bytes(-n)}"

    if n < 1024:
        return f"{n} B"
    elif n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    elif n < 1024 ** 3:
        return f"{n / (1024 ** 2):.1f} MB"
    else:
        return f"{n / (1024 ** 3):.1f} GB"
