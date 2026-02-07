"""In-memory preprocessed data snapshots for pipeline step caching.

Stores lightweight feature snapshots after each preprocessing step,
keyed by ``(chain_path_hash, data_hash, selector_fingerprint)``.
For generator sweeps with shared preprocessing prefixes and for
the refit pass replaying the same chain, caching eliminates
redundant computation.

The ``StepCache`` wraps :class:`DataCache` from
``nirs4all.data.performance.cache`` as the LRU storage backend,
using ``CachedStepState`` (a targeted snapshot of feature arrays
only) instead of full dataset deep-copies.
"""

from __future__ import annotations

import copy
import hashlib
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

from nirs4all.core.logging import get_logger
from nirs4all.data.performance.cache import DataCache
from nirs4all.utils.memory import estimate_cache_entry_bytes

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import DataSelector

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CachedStepState — lightweight snapshot for step cache entries
# ---------------------------------------------------------------------------

@dataclass
class CachedStepState:
    """Lightweight snapshot of dataset feature state after a preprocessing step.

    Only the parts of SpectroDataset that change during preprocessing are
    captured: the feature source arrays and their processing names.  Metadata,
    targets, sample indices, fold assignments, and tags are NOT copied because
    preprocessing transforms do not modify them.

    Attributes:
        features_sources: Deep-copied list of FeatureSource objects.
        processing_names: Processing chain per source (list of lists of str).
        content_hash: Post-step content hash of the dataset.
        bytes_estimate: Estimated memory footprint (computed once at store time).
    """

    features_sources: List[Any] = field(default_factory=list)
    processing_names: List[List[str]] = field(default_factory=list)
    content_hash: str = ""
    bytes_estimate: int = 0


def _compute_selector_fingerprint(selector: DataSelector) -> str:
    """Hash the execution-context selector fields that affect transformer behaviour.

    The fingerprint includes: partition, fold_id, processing chain,
    include_augmented flag, tag_filters, and branch_path.  Two calls with the
    same chain and data hash but different selectors must produce different
    cache keys.

    Args:
        selector: The DataSelector whose relevant fields will be hashed.

    Returns:
        A 16-character hex digest.
    """
    parts = [
        str(selector.partition),
        str(selector.fold_id),
        str(selector.processing),
        str(selector.include_augmented),
        str(sorted(selector.tag_filters.items())) if selector.tag_filters else "",
        str(selector.branch_path),
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# StepCache — main API
# ---------------------------------------------------------------------------

class StepCache:
    """LRU cache for preprocessed feature snapshots.

    Each entry is a :class:`CachedStepState` — a targeted snapshot of the
    feature arrays only — keyed by ``(chain_path_hash, data_hash,
    selector_fingerprint)``.

    The cache delegates to :class:`DataCache` for LRU eviction and
    thread-safe access.

    Args:
        max_size_mb: Maximum memory budget in megabytes.
        max_entries: Maximum number of cached entries.
    """

    def __init__(self, max_size_mb: int = 2048, max_entries: int = 200) -> None:
        self._backend = DataCache(
            max_size_mb=max_size_mb,
            max_entries=max_entries,
            ttl_seconds=None,
            eviction_policy="lru",
        )
        self._lock = threading.Lock()
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        self._peak_bytes = 0

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def miss_count(self) -> int:
        return self._miss_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        chain_path_hash: str,
        data_hash: str,
        selector: Optional[DataSelector] = None,
    ) -> Optional[CachedStepState]:
        """Retrieve a cached step state.

        Args:
            chain_path_hash: Hash of the step chain.
            data_hash: Hash of the input data.
            selector: DataSelector for fingerprinting (ensures fold/partition
                correctness).

        Returns:
            A ``CachedStepState``, or ``None`` on cache miss.
        """
        key = self._make_key(chain_path_hash, data_hash, selector)
        with self._lock:
            state = self._backend.get(key)
            if state is None:
                self._miss_count += 1
                return None
            self._hit_count += 1
        return state

    def put(
        self,
        chain_path_hash: str,
        data_hash: str,
        dataset: SpectroDataset,
        selector: Optional[DataSelector] = None,
    ) -> None:
        """Store a feature snapshot in the cache.

        Only the feature sources and processing names are deep-copied.

        Args:
            chain_path_hash: Hash of the step chain.
            data_hash: Hash of the input data.
            dataset: Dataset whose feature state will be snapshotted.
            selector: DataSelector for fingerprinting.
        """
        state = self._snapshot(dataset)
        key = self._make_key(chain_path_hash, data_hash, selector)
        with self._lock:
            prev_evictions = self._backend.eviction_count
            self._backend.set(key, state)
            self._eviction_count += self._backend.eviction_count - prev_evictions
            # Track peak memory
            current = self._backend.total_size
            if current > self._peak_bytes:
                self._peak_bytes = current

    def restore(self, state: CachedStepState, dataset: SpectroDataset) -> None:
        """Restore cached feature state into a live dataset.

        Deep-copies the cached feature sources back into the dataset and
        updates the content hash.

        Args:
            state: Previously cached step state.
            dataset: Target dataset to restore into.
        """
        dataset._features.sources = copy.deepcopy(state.features_sources)
        dataset._content_hash_cache = state.content_hash

    def clear(self) -> None:
        with self._lock:
            self._backend.clear()
            self._hit_count = 0
            self._miss_count = 0
            self._eviction_count = 0
            self._peak_bytes = 0

    def stats(self) -> dict[str, object]:
        """Return cache statistics for observability."""
        with self._lock:
            total = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total if total > 0 else 0.0
            backend_stats = self._backend.stats()
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": hit_rate,
            "evictions": self._eviction_count,
            "peak_mb": self._peak_bytes / (1024 * 1024),
            "entries": backend_stats["entries"],
            "max_entries": backend_stats["max_entries"],
            "size_mb": backend_stats["size_mb"],
            "max_size_mb": backend_stats["max_size_mb"],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(
        chain_path_hash: str,
        data_hash: str,
        selector: Optional[DataSelector] = None,
    ) -> str:
        """Build a composite cache key."""
        if selector is not None:
            fingerprint = _compute_selector_fingerprint(selector)
            return f"step:{chain_path_hash}:{data_hash}:{fingerprint}"
        return f"step:{chain_path_hash}:{data_hash}"

    @staticmethod
    def _snapshot(dataset: SpectroDataset) -> CachedStepState:
        """Create a ``CachedStepState`` from the current dataset.

        Deep-copies only the feature sources (the only thing preprocessing
        modifies). Computing ``bytes_estimate`` via numpy nbytes is O(1).
        """
        sources = copy.deepcopy(dataset._features.sources)
        processing_names = [
            list(dataset.features_processings(i))
            for i in range(len(sources))
        ]

        # Estimate bytes via numpy nbytes (O(1) per array, no pickle)
        total_bytes = 0
        for src in sources:
            total_bytes += src._storage.nbytes

        return CachedStepState(
            features_sources=sources,
            processing_names=processing_names,
            content_hash=dataset.content_hash(),
            bytes_estimate=total_bytes,
        )
