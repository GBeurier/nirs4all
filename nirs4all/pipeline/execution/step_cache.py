"""In-memory preprocessed data snapshots for pipeline step caching.

Stores lightweight feature snapshots after each preprocessing step,
keyed by ``(chain_path_hash, data_hash, selector_fingerprint)``.
For generator sweeps with shared preprocessing prefixes and for
the refit pass replaying the same chain, caching eliminates
redundant computation.

Uses copy-on-write (CoW) ``SharedBlocks`` references instead of
``copy.deepcopy()`` for near-zero-cost cache hits.  The cached
``SharedBlocks`` are never mutated; when the next pipeline step
writes to the dataset, ``_prepare_for_mutation()`` triggers a CoW
detach that copies only once, on demand.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np

from nirs4all.core.logging import get_logger
from nirs4all.data._features.array_storage import SharedBlocks
from nirs4all.data.performance.cache import DataCache

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import DataSelector

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CachedStepState — lightweight CoW snapshot for step cache entries
# ---------------------------------------------------------------------------

# Per-source snapshot: (SharedBlocks ref, processing_ids, headers, header_unit)
SourceSnapshot = Tuple[SharedBlocks, List[str], Optional[List[str]], str]


@dataclass
class CachedStepState:
    """Lightweight CoW snapshot of dataset feature state after a preprocessing step.

    Uses ``SharedBlocks`` copy-on-write references instead of deep-copied
    ``FeatureSource`` objects.  Restore is O(n_sources) — near-free — because
    it only acquires an additional reference to each shared array.  The actual
    copy is deferred until the next pipeline step mutates the data.

    Attributes:
        source_snapshots: Per-source CoW snapshot tuples.
        processing_names: Processing chain per source (list of lists of str).
        content_hash: Post-step content hash of the dataset.
        bytes_estimate: Estimated memory footprint (computed once at store time).
    """

    source_snapshots: List[SourceSnapshot] = field(default_factory=list)
    processing_names: List[List[str]] = field(default_factory=list)
    content_hash: str = ""
    bytes_estimate: int = 0

    def release(self) -> None:
        """Release all SharedBlocks references held by this state.

        Called when the entry is evicted from the cache or on clear().
        """
        for snapshot in self.source_snapshots:
            shared = snapshot[0]
            shared.release()
        self.source_snapshots = []


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
    """LRU cache for preprocessed feature snapshots using copy-on-write.

    Each entry is a :class:`CachedStepState` containing ``SharedBlocks`` CoW
    references — keyed by ``(chain_path_hash, data_hash,
    selector_fingerprint)``.

    Cache hits are near-free (O(n_sources) reference acquire). The actual
    data copy is deferred to when the next pipeline step mutates the dataset,
    at which point ``SharedBlocks.detach()`` creates an independent copy.

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
            on_evict=self._on_evict,
        )
        self._lock = threading.Lock()
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        self._peak_bytes = 0

        # Timing instrumentation (cumulative, in seconds)
        self._total_hash_s = 0.0
        self._total_snapshot_s = 0.0
        self._total_restore_s = 0.0

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

        Creates a CoW snapshot using ``SharedBlocks.ensure_shared().acquire()``
        per source. This is O(n_sources * np.stack) for the initial snapshot.

        Args:
            chain_path_hash: Hash of the step chain.
            data_hash: Hash of the input data.
            dataset: Dataset whose feature state will be snapshotted.
            selector: DataSelector for fingerprinting.
        """
        t0 = time.monotonic()
        state = self._snapshot(dataset)
        self._total_snapshot_s += time.monotonic() - t0

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
        """Restore cached feature state into a live dataset via CoW.

        Acquires a new reference to each cached ``SharedBlocks`` and sets
        the dataset's storage to shared mode.  This is O(n_sources) — near
        free.  The actual data copy is deferred to the next mutation.

        Args:
            state: Previously cached step state.
            dataset: Target dataset to restore into.
        """
        t0 = time.monotonic()
        assert len(state.source_snapshots) == len(dataset._features.sources), (
            f"Source count mismatch: cached {len(state.source_snapshots)} "
            f"vs live {len(dataset._features.sources)}"
        )
        for source, (shared, proc_ids, headers, header_unit) in zip(
            dataset._features.sources, state.source_snapshots
        ):
            source._storage.restore_from_shared(shared.acquire())
            source._processing_mgr.reset_processings(proc_ids)
            source._header_mgr.set_headers(headers, unit=header_unit)
        dataset._content_hash_cache = state.content_hash
        self._total_restore_s += time.monotonic() - t0

    def clear(self) -> None:
        """Clear all cached entries, releasing SharedBlocks references."""
        with self._lock:
            # Release all SharedBlocks before clearing
            for key in list(self._backend._cache.keys()):
                entry = self._backend._cache.get(key)
                if entry and isinstance(entry.data, CachedStepState):
                    entry.data.release()
            self._backend.clear()
            self._hit_count = 0
            self._miss_count = 0
            self._eviction_count = 0
            self._peak_bytes = 0
            self._total_hash_s = 0.0
            self._total_snapshot_s = 0.0
            self._total_restore_s = 0.0

    def stats(self) -> dict[str, object]:
        """Return cache statistics for observability.

        Includes timing breakdown for hash, snapshot, and restore operations.
        """
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
            "total_snapshot_ms": self._total_snapshot_s * 1000,
            "total_restore_ms": self._total_restore_s * 1000,
        }

    def record_hash_time(self, elapsed_s: float) -> None:
        """Record time spent computing cache hashes (called by executor)."""
        self._total_hash_s += elapsed_s

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _on_evict(entry_data: Any) -> None:
        """Release SharedBlocks references when a cache entry is evicted."""
        if isinstance(entry_data, CachedStepState):
            entry_data.release()

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
        """Create a CoW snapshot from the current dataset.

        Uses ``ensure_shared().acquire()`` per source for O(np.stack) cost.
        The cached SharedBlocks reference is never mutated: subsequent
        transforms trigger CoW detach in the live dataset's storage.
        """
        source_snapshots: list[SourceSnapshot] = []
        processing_names: list[list[str]] = []
        total_bytes = 0

        for i, source in enumerate(dataset._features.sources):
            shared = source._storage.ensure_shared().acquire()
            proc_ids = source._processing_mgr.processing_ids  # returns a copy
            headers = list(source._header_mgr.headers) if source._header_mgr.headers else None
            header_unit = source._header_mgr.header_unit
            source_snapshots.append((shared, proc_ids, headers, header_unit))
            processing_names.append(list(dataset.features_processings(i)))
            total_bytes += source._storage.nbytes

        return CachedStepState(
            source_snapshots=source_snapshots,
            processing_names=processing_names,
            content_hash=dataset.content_hash(),
            bytes_estimate=total_bytes,
        )
