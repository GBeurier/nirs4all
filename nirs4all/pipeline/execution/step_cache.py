"""In-memory preprocessed data snapshots for pipeline step caching.

Stores the full ``SpectroDataset`` state after each preprocessing step,
keyed by ``(chain_path_hash, data_hash)``.  For generator sweeps with
shared preprocessing prefixes and for the refit pass replaying the same
chain, caching preprocessed dataset snapshots eliminates redundant
computation at the dataset level.

The ``StepCache`` wraps the existing :class:`DataCache` from
``nirs4all.data.performance.cache`` as the underlying LRU storage
backend, adding a step-specific composite key scheme and deep-copy
semantics to prevent aliasing.
"""

from __future__ import annotations

import copy
import threading

import numpy as np

from nirs4all.core.logging import get_logger
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.performance.cache import DataCache

logger = get_logger(__name__)


class StepCache:
    """LRU cache for preprocessed dataset snapshots.

    Each entry is a deep copy of a ``SpectroDataset`` stored after a
    preprocessing step completes.  Entries are keyed by a composite key
    of ``(chain_path_hash, data_hash)`` so that different pipeline
    variants sharing the same preprocessing prefix can reuse cached
    snapshots.

    The cache delegates to :class:`DataCache` for LRU eviction and
    thread-safe access, while adding dataset-specific size estimation
    and deep-copy semantics on get/put.

    Args:
        max_size_mb: Maximum memory budget in megabytes.  When the
            total estimated size of cached datasets exceeds this limit,
            the least-recently-used entries are evicted.

    Example:
        >>> cache = StepCache(max_size_mb=1024)
        >>> cache.put("step_hash_1", "data_hash_a", dataset)
        >>> restored = cache.get("step_hash_1", "data_hash_a")
        >>> assert restored is not None
    """

    def __init__(self, max_size_mb: int = 2048) -> None:
        self._backend = DataCache(
            max_size_mb=max_size_mb,
            max_entries=10_000,
            ttl_seconds=None,
        )
        self._lock = threading.Lock()
        self._hit_count = 0
        self._miss_count = 0

    @property
    def hit_count(self) -> int:
        """Total cache hits."""
        return self._hit_count

    @property
    def miss_count(self) -> int:
        """Total cache misses."""
        return self._miss_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, chain_path_hash: str, data_hash: str) -> SpectroDataset | None:
        """Retrieve a cached dataset snapshot.

        Returns a *deep copy* of the cached dataset so that subsequent
        pipeline mutations do not corrupt the cached state.

        Args:
            chain_path_hash: Hash identifying the chain of steps up to
                (and including) the cached step.
            data_hash: Hash of the input data to the cached step.

        Returns:
            A deep copy of the cached ``SpectroDataset``, or ``None`` on
            cache miss.
        """
        key = self._make_key(chain_path_hash, data_hash)
        with self._lock:
            dataset = self._backend.get(key)
            if dataset is None:
                self._miss_count += 1
                return None
            self._hit_count += 1
        # Deep copy outside the lock to avoid holding it during the
        # (potentially expensive) copy operation.
        return copy.deepcopy(dataset)

    def put(self, chain_path_hash: str, data_hash: str, dataset: SpectroDataset) -> None:
        """Store a dataset snapshot in the cache.

        A *deep copy* of the dataset is stored to prevent aliasing with
        the caller's live dataset object.

        Args:
            chain_path_hash: Hash identifying the chain of steps.
            data_hash: Hash of the input data.
            dataset: Dataset to cache (will be deep-copied).
        """
        key = self._make_key(chain_path_hash, data_hash)
        snapshot = copy.deepcopy(dataset)
        with self._lock:
            self._backend.set(key, snapshot)

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._backend.clear()
            self._hit_count = 0
            self._miss_count = 0

    def stats(self) -> dict[str, object]:
        """Return cache statistics for observability.

        Returns:
            Dict with ``hit_count``, ``miss_count``, ``hit_rate``,
            and the underlying ``DataCache`` statistics.
        """
        with self._lock:
            total = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total if total > 0 else 0.0
            backend_stats = self._backend.stats()
        return {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
            **backend_stats,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(chain_path_hash: str, data_hash: str) -> str:
        """Build a composite cache key from the two hash components."""
        return f"step:{chain_path_hash}:{data_hash}"

    @staticmethod
    def estimate_dataset_size(dataset: SpectroDataset) -> int:
        """Estimate the in-memory size of a dataset in bytes.

        Sums ``X.nbytes`` across all sources and processings.

        Args:
            dataset: Dataset to estimate.

        Returns:
            Estimated size in bytes.
        """
        total = 0
        for source in dataset._features.sources:
            arr = source._storage.array
            total += arr.nbytes
        # Add target array size
        y_arr = dataset._targets._data.get("numeric")
        if y_arr is not None and isinstance(y_arr, np.ndarray):
            total += y_arr.nbytes
        return total
