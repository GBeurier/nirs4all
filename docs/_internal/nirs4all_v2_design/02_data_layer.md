# NIRS4ALL v2.0: Data Layer Design

**Author**: GitHub Copilot (Claude Opus 4.5)
**Date**: December 25, 2025
**Status**: Design Proposal (Revised per Critical Review)
**Document**: 2 of 5

---

## Table of Contents

1. [Overview](#overview)
2. [FeatureBlock](#featureblock)
3. [FeatureBlockStore](#featureblockstore)
4. [SampleRegistry](#sampleregistry)
5. [TargetStore](#targetstore)
6. [ViewSpec and ViewResolver](#viewspec-and-viewresolver)
7. [DatasetContext](#datasetcontext)
8. [Aggregation and Repetition Handling](#aggregation-and-repetition-handling)
9. [Data Operations](#data-operations)
10. [Layout Strategies and Model Input Construction](#layout-strategies-and-model-input-construction)
11. [Backend Abstraction](#backend-abstraction)
12. [Performance Considerations](#performance-considerations)

---

## Overview

The Data Layer provides unified data storage with Copy-on-Write semantics and lazy evaluation. It replaces the current `SpectroDataset` with a more composable architecture while preserving NIRS-specific features.

### Design Goals

1. **Copy-on-Write (CoW)**: Blocks share memory until modified, avoiding full copies
2. **Lineage Tracking**: Every block knows its parent and transformation
3. **Lazy Evaluation**: Views are resolved only when data is needed
4. **Backend Flexibility**: NumPy by default, xarray optional
5. **Multi-Source Support**: Independent blocks aligned by sample ID
6. **Memory Efficiency**: Reference counting, garbage collection, CoW semantics
7. **Out-of-Core Support**: Memory-mapped arrays for datasets larger than RAM
8. **Domain-Aware**: Native support for sample repetitions and aggregation keys

### Immutability Clarification

**Important**: `FeatureBlock` is *logically immutable* from the API perspective, but uses *Copy-on-Write (CoW)* internally for memory efficiency.

- **Public API contract**: Blocks are never mutated after creation. Transformations always produce new blocks.
- **Internal optimization**: When a block has `_ref_count == 1` (single owner), the `derive()` method may reuse the underlying array memory to avoid unnecessary copies.
- **Thread safety**: The CoW mechanism is thread-safe through the FeatureBlockStore's locking.

```python
# From user perspective: blocks are immutable
block = store.get(block_id)
# block.data[0, 0] = 999  # ❌ NOT ALLOWED - data is read-only

# Transformations always create new blocks
new_block_id = store.register_transform(
    parent_id=block_id,
    data=transform_fn(block.data),  # New data, new block
    transform_info=info
)

# Internally: CoW may reuse memory when safe
# If ref_count == 1 and not shared, transform_fn may operate in-place
# This is transparent to users - the API always returns a new block ID
```

### Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DatasetContext                            │
│    (bundles everything needed for one pipeline execution)        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ FeatureBlockStore│  │  SampleRegistry │  │   TargetStore   │  │
│  │                 │  │                 │  │                 │  │
│  │ ┌─────────────┐ │  │  Polars DF:     │  │  versions:      │  │
│  │ │ Block "nir" │ │  │  - sample_id    │  │  - "raw"        │  │
│  │ │ 3D: S×P×F   │ │  │  - partition    │  │  - "scaled"     │  │
│  │ └─────────────┘ │  │  - fold_id      │  │  - "encoded"    │  │
│  │ ┌─────────────┐ │  │  - group        │  │                 │  │
│  │ │Block "mark" │ │  │  - excluded     │  │  transformers:  │  │
│  │ │ 3D: S×P×F   │ │  │  - origin       │  │  - fitted obj   │  │
│  │ └─────────────┘ │  │  - augmentation │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      ViewResolver                            ││
│  │   ViewSpec → lazy slice → materialized arrays                ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## FeatureBlock

A `FeatureBlock` uses Copy-on-Write semantics for memory efficiency with full lineage tracking.

### Structure

```python
@dataclass
class FeatureBlock:
    """Copy-on-Write feature data container with lineage tracking.

    Blocks use CoW semantics: the underlying array is shared until a
    mutation is requested, at which point a copy is made. This avoids
    the 15GB memory churn issue from naive immutable copies.

    Attributes:
        block_id: Unique identifier (hash-based)
        _data: 3D array (n_samples, n_processings, n_features) - internal
        headers: Feature names/wavelengths per processing slot
        header_unit: Unit type ("nm", "cm-1", "text", "index")
        processing_ids: Names of each processing slot
        metadata: Arbitrary block-level metadata
        parent_id: ID of parent block (None for source blocks)
        transform_info: Description of transformation that created this block
        lineage_hash: Deterministic hash of parent + transform
        _ref_count: Reference count for garbage collection
        _is_cow: Whether this block shares memory with another
    """
    block_id: str
    _data: np.ndarray  # Shape: (n_samples, n_processings, n_features)
    headers: List[str]
    header_unit: str
    processing_ids: List[str]
    metadata: Dict[str, Any]
    parent_id: Optional[str]
    transform_info: Optional[TransformInfo]
    lineage_hash: str
    created_at: datetime
    _ref_count: int = 1
    _is_cow: bool = False

    # Optional validity mask for spectroscopy-specific NaN handling
    _validity_mask: Optional[np.ndarray] = None  # Shape: same as _data, True=valid

    # Tracking for LRU eviction
    last_accessed: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate block data on construction.

        Raises:
            ValueError: If data shape is invalid or contains unexpected NaN
        """
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate data array shape and content.

        Checks:
        1. Data is 3D array (n_samples, n_processings, n_features)
        2. Dimensions are positive
        3. NaN handling is explicit (via validity_mask or documented)
        """
        # Shape validation
        if self._data.ndim != 3:
            raise ValueError(
                f"FeatureBlock data must be 3D (samples, processings, features), "
                f"got {self._data.ndim}D with shape {self._data.shape}"
            )

        n_samples, n_proc, n_feat = self._data.shape
        if n_samples == 0:
            raise ValueError("FeatureBlock cannot have 0 samples")
        if n_proc == 0:
            raise ValueError("FeatureBlock cannot have 0 processings")
        if n_feat == 0:
            raise ValueError("FeatureBlock cannot have 0 features")

        # NaN validation
        nan_count = np.isnan(self._data).sum()
        if nan_count > 0:
            if self._validity_mask is None:
                import warnings
                warnings.warn(
                    f"FeatureBlock contains {nan_count} NaN values but no "
                    f"validity_mask is set. Consider using with_validity_mask() "
                    f"to explicitly mark invalid regions, or preprocess to "
                    f"remove NaN values.",
                    UserWarning
                )
            else:
                # Verify mask covers NaN locations
                nan_mask = np.isnan(self._data)
                invalid_mask = ~self._validity_mask
                uncovered_nans = nan_mask & self._validity_mask
                if uncovered_nans.any():
                    raise ValueError(
                        f"validity_mask marks {uncovered_nans.sum()} NaN values "
                        f"as valid. All NaN locations must be marked invalid."
                    )

    @property
    def data(self) -> np.ndarray:
        """Read-only access to data. For modifications, use derive()."""
        return self._data

    @property
    def validity_mask(self) -> Optional[np.ndarray]:
        """Optional mask indicating valid data points.

        For spectroscopy data, some values may be invalid due to:
        - Detector saturation
        - Wavelength ranges outside detector sensitivity
        - Known instrument artifacts

        If set, True indicates valid data, False indicates invalid.
        Transforms should propagate or update this mask appropriately.
        """
        return self._validity_mask

    def with_validity_mask(self, mask: np.ndarray) -> "FeatureBlock":
        """Return new block with validity mask attached."""
        new_block = FeatureBlock(
            block_id=self.block_id,
            _data=self._data,
            headers=self.headers,
            header_unit=self.header_unit,
            processing_ids=self.processing_ids,
            metadata=self.metadata,
            parent_id=self.parent_id,
            transform_info=self.transform_info,
            lineage_hash=self.lineage_hash,
            created_at=self.created_at,
            _ref_count=self._ref_count,
            _is_cow=True,  # Share data
            _validity_mask=mask
        )
        return new_block

    def derive(self, transform_fn: Callable[[np.ndarray], np.ndarray],
               transform_info: TransformInfo) -> "FeatureBlock":
        """Create derived block with CoW semantics.

        If data is shared (CoW), copies before modification.
        Otherwise, transforms in-place and creates new block reference.
        """
        if self._is_cow:
            new_data = transform_fn(self._data.copy())
        else:
            new_data = transform_fn(self._data)

        return FeatureBlock(
            block_id=compute_block_id(self.lineage_hash, transform_info),
            _data=new_data,
            headers=self.headers,
            header_unit=self.header_unit,
            processing_ids=self.processing_ids + [transform_info.target_processing],
            metadata=self.metadata,
            parent_id=self.block_id,
            transform_info=transform_info,
            lineage_hash=compute_lineage_hash(self.lineage_hash, transform_info),
            created_at=datetime.now(),
            _is_cow=False
        )

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._data.shape

    @property
    def n_samples(self) -> int:
        return self._data.shape[0]

    @property
    def n_processings(self) -> int:
        return self._data.shape[1]

    @property
    def n_features(self) -> int:
        return self._data.shape[2]


@dataclass(frozen=True)
class TransformInfo:
    """Describes the transformation that created a block.

    Attributes:
        transform_class: Fully qualified class name
        transform_params: Constructor parameters
        fit_params: Parameters used during fit
        source_processing: Which processing slot was transformed
        target_processing: Name of the output processing slot
    """
    transform_class: str
    transform_params: Dict[str, Any]
    fit_params: Dict[str, Any] = field(default_factory=dict)
    source_processing: Optional[str] = None
    target_processing: Optional[str] = None
```

### Lineage Hashing

The lineage hash uniquely identifies a block based on its complete history:

```python
# Configuration for hash length
HASH_LENGTH = 32  # Characters from SHA-256 hex digest (32 = 128 bits)
# Note: Original design used 16 chars. Increased to 32 to reduce collision risk
# in large pipelines. Full 64-char hex is also valid for maximum safety.

def compute_lineage_hash(
    parent_hash: Optional[str],
    transform_info: Optional[TransformInfo],
    data_hash: Optional[str] = None,
    full_hash: bool = False,  # Set True for bundle exports
    include_version_info: bool = True  # Include library versions for reproducibility
) -> str:
    """Compute deterministic lineage hash.

    For source blocks (no parent): hash of data content
    For transformed blocks: hash of parent_hash + transform_info

    Args:
        parent_hash: Hash of parent block (None for source blocks)
        transform_info: Transformation that created this block
        data_hash: Hash of source data (for source blocks)
        full_hash: If True, return full 64-char hash (for exports/bundles)
        include_version_info: If True, include Python/NumPy version in hash

    Returns:
        Hash string (32 or 64 hex chars depending on full_hash)

    Hash Stability Guarantees:
        - Same inputs → same hash across runs on same machine
        - Same inputs → same hash across machines with same Python major.minor
        - Hash MAY differ across major NumPy versions (documented breaking change)
        - Hash MAY differ across platforms (x86 vs ARM) for floating-point edge cases

    Version Inclusion Rationale:
        Including Python/NumPy versions ensures that cache hits only occur when
        the environment is compatible. This prevents subtle bugs from library
        behavior changes between versions.

    Note on Hash Length:
        - 16 chars (64 bits): ~1 in 18 quintillion collision probability
          BUT with 10,000 blocks, birthday paradox gives ~1 in 340 million
        - 32 chars (128 bits): Negligible collision risk even for millions of blocks
        - 64 chars (256 bits): Full SHA-256, used for cryptographic guarantees

        We use 32 chars as default balance between readability and safety.
        Bundle exports use full 64 chars for reproducibility guarantees.
    """
    import sys
    import numpy as np

    hasher = hashlib.sha256()

    # Include version info for reproducibility
    if include_version_info:
        version_string = f"py{sys.version_info.major}.{sys.version_info.minor}"
        version_string += f"|np{np.__version__.split('.')[0]}"
        hasher.update(version_string.encode())

    if parent_hash is None:
        # Source block: hash the actual data
        if data_hash:
            hasher.update(data_hash.encode())
        length = 64 if full_hash else HASH_LENGTH
        return hasher.hexdigest()[:length]

    # Transformed block: hash parent + transform
    hasher.update(parent_hash.encode())
    hasher.update(transform_info.transform_class.encode())
    # Sort keys and use default=str for non-JSON-serializable params
    hasher.update(json.dumps(
        transform_info.transform_params,
        sort_keys=True,
        default=str
    ).encode())
    length = 64 if full_hash else HASH_LENGTH
    return hasher.hexdigest()[:length]


def compute_data_hash(data: np.ndarray) -> str:
    """Compute hash of array data for source block identification.

    Uses a combination of:
    - Array shape
    - Data type
    - Content hash (via tobytes() for small arrays, sampling for large)

    Args:
        data: NumPy array to hash

    Returns:
        64-character hex hash string
    """
    hasher = hashlib.sha256()

    # Shape and dtype are always included
    hasher.update(str(data.shape).encode())
    hasher.update(str(data.dtype).encode())

    # For large arrays, sample to avoid memory/time issues
    if data.size > 1_000_000:
        # Sample: first 1000, last 1000, and 1000 random elements
        rng = np.random.RandomState(42)  # Deterministic sampling
        indices = np.concatenate([
            np.arange(min(1000, data.size)),
            np.arange(max(0, data.size - 1000), data.size),
            rng.choice(data.size, min(1000, data.size), replace=False)
        ])
        flat = data.ravel()[np.unique(indices)]
        hasher.update(flat.tobytes())
    else:
        hasher.update(data.tobytes())

    return hasher.hexdigest()
```

### Block Creation

Blocks are created through the store, never directly:

```python
# ❌ Never create blocks directly
block = FeatureBlock(...)  # Forbidden

# ✅ Always use the store
block_id = store.register_source(
    data=X_raw,
    headers=wavelengths,
    header_unit="nm",
    metadata={"source": "NIR", "instrument": "FOSS"}
)

# ✅ Transformations create new blocks
scaled_id = store.register_transform(
    parent_id=block_id,
    data=X_scaled,
    transform_info=TransformInfo(
        transform_class="sklearn.preprocessing.MinMaxScaler",
        transform_params={"feature_range": (0, 1)},
        source_processing="raw",
        target_processing="scaled"
    )
)
```

---

## FeatureBlockStore

The `FeatureBlockStore` is the central registry for all feature blocks.

### Interface

```python
class FeatureBlockStore:
    """Central registry for feature blocks with lineage tracking.

    Provides:
    - Block registration (source and transformed)
    - Block retrieval by ID or lineage
    - Lineage traversal
    - Cache management
    - Persistence to disk
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self._blocks: Dict[str, FeatureBlock] = {}
        self._lineage_index: Dict[str, str] = {}  # lineage_hash → block_id
        self._cache_dir = cache_dir

    # === Registration Methods ===

    def register_source(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
        header_unit: str = "index",
        processing_id: str = "raw",
        metadata: Optional[Dict[str, Any]] = None,
        source_name: Optional[str] = None
    ) -> str:
        """Register a new source block (no parent).

        Args:
            data: Feature array (2D or 3D)
            headers: Feature names/wavelengths
            header_unit: Unit type for headers
            processing_id: Name for the processing slot
            metadata: Block-level metadata
            source_name: Optional human-readable name

        Returns:
            Block ID
        """
        ...

    def register_transform(
        self,
        parent_id: str,
        data: np.ndarray,
        transform_info: TransformInfo,
        processing_id: Optional[str] = None
    ) -> str:
        """Register a transformed block.

        Args:
            parent_id: ID of the parent block
            data: Transformed feature array
            transform_info: Description of the transformation
            processing_id: Override processing name

        Returns:
            Block ID (may return existing ID if lineage matches)
        """
        ...

    def register_augmented(
        self,
        parent_id: str,
        data: np.ndarray,
        augmentation_id: str,
        sample_indices: List[int]
    ) -> str:
        """Register augmented samples (new rows).

        Args:
            parent_id: ID of the block being augmented
            data: New samples to add
            augmentation_id: Name of augmentation method
            sample_indices: Original sample indices being augmented

        Returns:
            Block ID
        """
        ...

    # === Retrieval Methods ===

    def get(self, block_id: str) -> FeatureBlock:
        """Get block by ID."""
        ...

    def get_by_lineage(self, lineage_hash: str) -> Optional[FeatureBlock]:
        """Get block by lineage hash (for cache lookup)."""
        ...

    def get_data(
        self,
        block_id: str,
        sample_indices: Optional[List[int]] = None,
        processing_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """Get block data with optional slicing."""
        ...

    # === Lineage Methods ===

    def get_lineage_chain(self, block_id: str) -> List[str]:
        """Get ordered list of block IDs from source to current."""
        ...

    def get_children(self, block_id: str) -> List[str]:
        """Get all blocks derived from this block."""
        ...

    def get_processing_chain(self, block_id: str) -> List[str]:
        """Get list of processing names in lineage order."""
        ...

    # === Persistence ===

    def save_block(self, block_id: str, path: Path) -> None:
        """Save block to disk (Parquet + metadata JSON)."""
        ...

    def load_block(self, path: Path) -> str:
        """Load block from disk, return block_id."""
        ...

    def clear_cache(self, older_than: Optional[datetime] = None) -> int:
        """Clear cached blocks, return count removed."""
        ...

    def gc(self, keep_roots: bool = True) -> int:
        """Garbage collect unreachable blocks.

        Uses reference counting to remove blocks that are no longer
        reachable from any active context.

        Eviction follows the configured policy (default: LRU).

        Args:
            keep_roots: If True, never remove source blocks (no parent)

        Returns:
            Number of blocks removed
        """
        removed = 0
        to_remove = []

        for block_id, block in self._blocks.items():
            if block._ref_count <= 0:
                if keep_roots and block.parent_id is None:
                    continue
                to_remove.append(block_id)

        # Sort by eviction policy
        if self._eviction_policy == "lru":
            # LRU: evict least recently accessed first
            to_remove.sort(key=lambda bid: self._blocks[bid].last_accessed)
        elif self._eviction_policy == "fifo":
            # FIFO: evict oldest first
            to_remove.sort(key=lambda bid: self._blocks[bid].created_at)
        # Default is LRU

        for block_id in to_remove:
            # Persist to disk if cache_dir set
            if self._cache_dir:
                self.save_block(block_id, self._cache_dir / f"{block_id}.parquet")
            del self._blocks[block_id]
            removed += 1

        return removed


class EvictionPolicy(Enum):
    """Block eviction policy for garbage collection.

    LRU (Least Recently Used): Evicts blocks that haven't been accessed recently.
        Good for: Long-running sessions, iterative exploration.

    FIFO (First In First Out): Evicts oldest blocks first.
        Good for: Sequential pipeline execution, predictable behavior.

    SIZE: Evicts largest blocks first.
        Good for: Memory-constrained environments.
    """
    LRU = "lru"
    FIFO = "fifo"
    SIZE = "size"

    def inc_ref(self, block_id: str) -> None:
        """Increment reference count."""
        if block_id in self._blocks:
            self._blocks[block_id]._ref_count += 1

    def dec_ref(self, block_id: str) -> None:
        """Decrement reference count."""
        if block_id in self._blocks:
            self._blocks[block_id]._ref_count -= 1


class AsyncGarbageCollector:
    """Optional background garbage collection for feature blocks.

    For pipelines with many intermediate blocks, synchronous GC can cause
    noticeable pauses. This class provides background cleanup.

    Usage:
        store = FeatureBlockStore(async_gc=True)
        # GC happens automatically in background thread

        # Or configure explicitly:
        gc = AsyncGarbageCollector(store, interval_seconds=5.0)
        gc.start()
        # ... run pipeline ...
        gc.stop()

    Thread Safety:
        The async GC acquires locks before evicting blocks, ensuring
        thread safety with concurrent block access.
    """

    def __init__(
        self,
        store: "FeatureBlockStore",
        interval_seconds: float = 5.0,
        max_batch_size: int = 100
    ):
        """Initialize async garbage collector.

        Args:
            store: The FeatureBlockStore to collect from
            interval_seconds: How often to run GC (default: 5s)
            max_batch_size: Maximum blocks to evict per cycle
        """
        import threading
        from queue import Queue, Empty

        self._store = store
        self._interval = interval_seconds
        self._max_batch = max_batch_size
        self._queue: Queue[str] = Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start background GC thread."""
        import threading

        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._gc_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background GC thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def schedule_gc(self, block_id: str) -> None:
        """Schedule a block for garbage collection.

        The block will be checked and evicted if eligible on the next GC cycle.
        """
        self._queue.put(block_id)

    def _gc_loop(self) -> None:
        """Background GC loop."""
        import time
        from queue import Empty

        while self._running:
            try:
                # Collect scheduled blocks
                scheduled = []
                while len(scheduled) < self._max_batch:
                    try:
                        block_id = self._queue.get_nowait()
                        scheduled.append(block_id)
                    except Empty:
                        break

                # Evict eligible blocks
                for block_id in scheduled:
                    if self._store.can_evict(block_id):
                        self._store.evict(block_id)

                # Also run periodic full GC
                self._store.gc(keep_roots=True)

            except Exception as e:
                # Log but don't crash the GC thread
                import logging
                logging.getLogger("nirs4all.gc").warning(f"GC error: {e}")

            time.sleep(self._interval)
```

### Multi-Source Support

Multiple sources are simply multiple blocks with alignment via `SampleRegistry`:

```python
# Register multiple sources
nir_block_id = store.register_source(
    X_nir, headers=wavelengths, source_name="NIR"
)
marker_block_id = store.register_source(
    X_markers, headers=marker_names, source_name="markers"
)

# Alignment is handled by SampleRegistry
registry.add_samples(
    n_samples=100,
    partition="train",
    metadata={"site": ["A"]*50 + ["B"]*50}
)

# Access aligned data via ViewResolver
view = ViewSpec(
    block_ids=[nir_block_id, marker_block_id],
    partition="train"
)
X_nir, X_markers = resolver.materialize(view)
```

---

## Data Validation

The `DataValidator` validates input data before it enters the pipeline, providing clear error messages for common issues.

### Interface

```python
@dataclass
class DataValidationResult:
    """Result of data validation.

    Attributes:
        is_valid: Whether all validation checks passed
        errors: List of critical errors (pipeline will fail)
        warnings: List of non-critical warnings
        statistics: Optional data statistics for debugging
    """
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Optional[Dict[str, Any]] = None

    def raise_if_invalid(self) -> None:
        """Raise ValueError if validation failed."""
        if not self.is_valid:
            raise ValueError(
                "Data validation failed:\n" +
                "\n".join(f"  - {e}" for e in self.errors)
            )


class DataValidator:
    """Validates input data before pipeline execution.

    Checks for:
    - Shape consistency between X and y
    - NaN/Inf values
    - Data type compatibility
    - Memory constraints
    - Spectral data sanity (wavelength ordering, etc.)

    Usage:
        validator = DataValidator()
        result = validator.validate(X, y, wavelengths=wl)
        result.raise_if_invalid()  # Raises ValueError if invalid

        # Or with context (more comprehensive checks):
        result = validator.validate_context(dataset_context)
    """

    def __init__(
        self,
        strict: bool = False,
        max_nan_fraction: float = 0.1,
        check_memory: bool = True,
        max_memory_gb: Optional[float] = None
    ):
        """Initialize validator.

        Args:
            strict: If True, treat warnings as errors
            max_nan_fraction: Maximum allowed fraction of NaN values
            check_memory: If True, validate memory constraints
            max_memory_gb: Maximum allowed memory usage in GB
        """
        self.strict = strict
        self.max_nan_fraction = max_nan_fraction
        self.check_memory = check_memory
        self.max_memory_gb = max_memory_gb or 8.0  # Default 8GB

    def validate(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        wavelengths: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DataValidationResult:
        """Validate input arrays.

        Args:
            X: Feature matrix (n_samples, n_features) or (n_samples, n_processings, n_features)
            y: Target array (n_samples,) or (n_samples, n_targets)
            wavelengths: Wavelength array (n_features,)
            metadata: Additional metadata to validate

        Returns:
            DataValidationResult with validation outcome
        """
        errors = []
        warnings = []
        stats = {}

        # === Shape validation ===
        if X.ndim < 2 or X.ndim > 3:
            errors.append(f"X must be 2D or 3D, got {X.ndim}D")
        else:
            stats["X_shape"] = X.shape
            n_samples = X.shape[0]

            if y is not None:
                if y.shape[0] != n_samples:
                    errors.append(
                        f"X and y sample count mismatch: {n_samples} vs {y.shape[0]}"
                    )
                stats["y_shape"] = y.shape

            if wavelengths is not None:
                n_features = X.shape[-1]
                if len(wavelengths) != n_features:
                    errors.append(
                        f"Wavelength count ({len(wavelengths)}) != feature count ({n_features})"
                    )

        # === Numeric validation ===
        nan_count = np.isnan(X).sum()
        nan_fraction = nan_count / X.size if X.size > 0 else 0
        stats["nan_fraction"] = nan_fraction

        if nan_fraction > self.max_nan_fraction:
            errors.append(
                f"NaN fraction ({nan_fraction:.1%}) exceeds threshold ({self.max_nan_fraction:.1%})"
            )
        elif nan_count > 0:
            warnings.append(f"Data contains {nan_count} NaN values ({nan_fraction:.1%})")

        inf_count = np.isinf(X).sum()
        if inf_count > 0:
            errors.append(f"Data contains {inf_count} Inf values")

        # === Data type validation ===
        if not np.issubdtype(X.dtype, np.floating):
            warnings.append(f"X dtype is {X.dtype}, will be cast to float64")

        # === Memory validation ===
        if self.check_memory:
            memory_gb = X.nbytes / (1024 ** 3)
            stats["memory_gb"] = memory_gb

            if memory_gb > self.max_memory_gb:
                errors.append(
                    f"Data exceeds memory limit: {memory_gb:.2f}GB > {self.max_memory_gb}GB"
                )

        # === Spectral-specific checks ===
        if wavelengths is not None:
            if not np.all(np.diff(wavelengths) > 0):
                if np.all(np.diff(wavelengths) < 0):
                    warnings.append("Wavelengths are in descending order (will be reversed)")
                else:
                    warnings.append("Wavelengths are not monotonically ordered")

        # Convert warnings to errors in strict mode
        if self.strict:
            errors.extend(warnings)
            warnings = []

        return DataValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            statistics=stats
        )

    def validate_context(self, context: "DatasetContext") -> DataValidationResult:
        """Validate a complete DatasetContext.

        Performs additional checks:
        - Cross-block sample alignment
        - Registry-block consistency
        - Target-sample alignment
        """
        errors = []
        warnings = []
        stats = {}

        # Check all blocks have same sample count
        sample_counts = {}
        for block_id in context.active_block_ids:
            block = context.block_store.get(block_id)
            sample_counts[block_id] = block.data.shape[0]

        if len(set(sample_counts.values())) > 1:
            errors.append(
                f"Blocks have inconsistent sample counts: {sample_counts}"
            )

        # Check registry matches block samples
        registry_count = context.sample_registry.count()
        block_count = list(sample_counts.values())[0] if sample_counts else 0
        if registry_count != block_count:
            errors.append(
                f"Registry sample count ({registry_count}) != block count ({block_count})"
            )

        stats["block_counts"] = sample_counts
        stats["registry_count"] = registry_count

        return DataValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            statistics=stats
        )
```

---

## SampleRegistry

The `SampleRegistry` is a Polars DataFrame tracking sample identity, metadata, and **aggregation relationships** for NIRS data with repetitions.

### Schema

```python
SAMPLE_REGISTRY_SCHEMA = {
    # Identity columns
    "_sample_id": pl.Int64,       # Unique measurement ID (obs_id)
    "_row_index": pl.Int64,       # Position in block arrays
    "_bio_id": pl.Int64,          # Biological sample ID (for aggregation)

    # Repetition tracking
    "repetition_idx": pl.Int32,   # Repetition number within bio_id (0, 1, 2, ...)
    "repetition_source": pl.Utf8, # Source of repetition grouping

    # Partition columns
    "partition": pl.Utf8,         # "train", "val", "test"
    "fold_id": pl.Int32,          # CV fold assignment (nullable)

    # Grouping columns
    "group": pl.Utf8,             # Group key for GroupKFold

    # Augmentation tracking
    "origin": pl.Int64,           # Parent sample_id (nullable)
    "augmentation": pl.Utf8,      # Augmentation method (nullable)

    # Exclusion
    "excluded": pl.Boolean,       # Marked as outlier

    # User metadata (dynamic)
    # ... any additional columns from user data
}
```

### Interface

```python
class SampleRegistry:
    """Polars-backed sample identity and metadata registry.

    Provides efficient filtering, grouping, and fold management
    with lazy evaluation where possible.
    """

    def __init__(self):
        self._df: pl.DataFrame = pl.DataFrame(schema=SAMPLE_REGISTRY_SCHEMA)
        self._fold_indices: Optional[List[Tuple[List[int], List[int]]]] = None

    # === Sample Management ===

    def add_samples(
        self,
        n_samples: int,
        partition: str = "train",
        group: Optional[Union[str, List[str]]] = None,
        metadata: Optional[Dict[str, List[Any]]] = None
    ) -> List[int]:
        """Add samples to registry, return sample_ids."""
        ...

    def add_augmented_samples(
        self,
        origin_ids: List[int],
        augmentation: str,
        count_per_origin: int = 1
    ) -> List[int]:
        """Add augmented samples linked to origins."""
        ...

    def mark_excluded(
        self,
        sample_ids: List[int],
        reason: Optional[str] = None
    ) -> None:
        """Mark samples as excluded (outliers)."""
        ...

    # === Querying ===

    def get_indices(
        self,
        partition: Optional[str] = None,
        fold_id: Optional[int] = None,
        include_augmented: bool = True,
        include_excluded: bool = False,
        filter_expr: Optional[pl.Expr] = None
    ) -> np.ndarray:
        """Get row indices matching criteria."""
        ...

    def get_sample_ids(
        self,
        **filter_kwargs
    ) -> List[int]:
        """Get sample IDs matching criteria."""
        ...

    def get_metadata(
        self,
        columns: Optional[List[str]] = None,
        sample_ids: Optional[List[int]] = None
    ) -> pl.DataFrame:
        """Get metadata for samples."""
        ...

    # === Fold Management ===

    def assign_folds(
        self,
        splitter: Any,  # sklearn splitter
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> None:
        """Assign fold IDs using sklearn splitter."""
        ...

    @property
    def folds(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get (train_indices, val_indices) for each fold."""
        ...

    @property
    def n_folds(self) -> int:
        """Number of CV folds."""
        ...

    # === Aggregation Support ===

    def get_aggregation_groups(
        self,
        by: Union[str, List[str]]
    ) -> Dict[Any, List[int]]:
        """Get sample groups for aggregation."""
        ...


### Polars Filter Helpers

Using Polars expressions directly for `sample_filter` requires learning Polars syntax. To reduce the learning curve, NIRS4ALL provides helper functions for common filtering patterns:

```python
# nirs4all/data/filters.py
"""Convenience filter builders for SampleRegistry.

These functions generate Polars expressions without requiring
users to learn Polars syntax. The generated expressions can be
passed to get_indices() or ViewSpec.sample_filter.
"""

import polars as pl
from typing import Any, List, Union


def where_equal(column: str, value: Any) -> pl.Expr:
    """Filter where column equals value.

    Example:
        >>> indices = registry.get_indices(
        ...     filter_expr=where_equal("site", "A")
        ... )
    """
    return pl.col(column) == value


def where_in(column: str, values: List[Any]) -> pl.Expr:
    """Filter where column is in a list of values.

    Example:
        >>> indices = registry.get_indices(
        ...     filter_expr=where_in("site", ["A", "B"])
        ... )
    """
    return pl.col(column).is_in(values)


def where_not_in(column: str, values: List[Any]) -> pl.Expr:
    """Filter where column is NOT in a list of values."""
    return ~pl.col(column).is_in(values)


def where_between(
    column: str,
    min_val: Union[int, float],
    max_val: Union[int, float],
    inclusive: str = "both"
) -> pl.Expr:
    """Filter where column is between min and max.

    Args:
        column: Column name
        min_val: Minimum value
        max_val: Maximum value
        inclusive: "both", "left", "right", or "neither"

    Example:
        >>> indices = registry.get_indices(
        ...     filter_expr=where_between("temperature", 20, 30)
        ... )
    """
    if inclusive == "both":
        return (pl.col(column) >= min_val) & (pl.col(column) <= max_val)
    elif inclusive == "left":
        return (pl.col(column) >= min_val) & (pl.col(column) < max_val)
    elif inclusive == "right":
        return (pl.col(column) > min_val) & (pl.col(column) <= max_val)
    else:  # neither
        return (pl.col(column) > min_val) & (pl.col(column) < max_val)


def where_null(column: str) -> pl.Expr:
    """Filter where column is null."""
    return pl.col(column).is_null()


def where_not_null(column: str) -> pl.Expr:
    """Filter where column is not null."""
    return pl.col(column).is_not_null()


def where_contains(column: str, pattern: str) -> pl.Expr:
    """Filter where string column contains pattern.

    Example:
        >>> indices = registry.get_indices(
        ...     filter_expr=where_contains("sample_id", "batch_01")
        ... )
    """
    return pl.col(column).str.contains(pattern)


def where_regex(column: str, regex: str) -> pl.Expr:
    """Filter where string column matches regex pattern."""
    return pl.col(column).str.contains(regex)


def combine_and(*exprs: pl.Expr) -> pl.Expr:
    """Combine multiple expressions with AND.

    Example:
        >>> filter_expr = combine_and(
        ...     where_equal("site", "A"),
        ...     where_between("temperature", 20, 30)
        ... )
    """
    if not exprs:
        raise ValueError("At least one expression required")
    result = exprs[0]
    for expr in exprs[1:]:
        result = result & expr
    return result


def combine_or(*exprs: pl.Expr) -> pl.Expr:
    """Combine multiple expressions with OR.

    Example:
        >>> filter_expr = combine_or(
        ...     where_equal("site", "A"),
        ...     where_equal("site", "B")
        ... )
    """
    if not exprs:
        raise ValueError("At least one expression required")
    result = exprs[0]
    for expr in exprs[1:]:
        result = result | expr
    return result


# Convenience aliases matching pandas-like syntax
isin = where_in
notin = where_not_in
between = where_between
contains = where_contains


# === Usage Examples ===

# Simple equality filter
view = ViewSpec(
    block_ids=("nir",),
    sample_filter=where_equal("batch", "2024-01")
)

# Complex filter with multiple conditions
from nirs4all.data.filters import combine_and, where_in, where_between

filter_expr = combine_and(
    where_in("instrument", ["FOSS_1", "FOSS_2"]),
    where_between("protein", 10, 20),
    where_not_null("quality_score")
)
indices = registry.get_indices(filter_expr=filter_expr)

# Or use lambdas for those comfortable with Polars
indices = registry.get_indices(
    filter_expr=lambda: (pl.col("batch") == "2024-01") & (pl.col("protein") > 15)
)
```

### Integration with Blocks

The registry provides the mapping between sample identities and block row indices:

```python
# Sample indices are consistent across all blocks
indices = registry.get_indices(partition="train", fold_id=0)

# Use same indices for all blocks
X_nir = store.get_data(nir_block_id, sample_indices=indices)
X_markers = store.get_data(marker_block_id, sample_indices=indices)
y_train = target_store.get(sample_indices=indices)

# All arrays have same row alignment
assert X_nir.shape[0] == X_markers.shape[0] == y_train.shape[0]
```

### Mutation Semantics During Augmentation

The `SampleRegistry` follows a **controlled mutation** model. While blocks are immutable, the registry can grow during pipeline execution:

**Key Rules:**

1. **Original samples are never modified**: Existing rows in the registry are immutable
2. **New samples are appended**: `add_samples()` and `add_augmented_samples()` add new rows
3. **Augmented samples track origin**: `origin` column links to parent sample_id
4. **Partition inheritance**: Augmented samples inherit their parent's partition by default

```python
# Before augmentation
registry.n_samples  # 100 (original)

# Sample augmentation step adds synthetic samples
aug_ids = registry.add_augmented_samples(
    origin_ids=[1, 2, 3],      # Augment 3 samples
    augmentation="noise",       # Method identifier
    count_per_origin=5          # 5 augmented per original
)

# After augmentation
registry.n_samples  # 115 (100 original + 15 augmented)

# New samples are linked to origins
meta = registry.get_metadata(sample_ids=aug_ids)
assert meta["origin"].to_list() == [1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3]
```

**Branch-Specific Registries:**

During branching, each branch may have different augmentations. The system creates **shallow copies** of the registry per branch:

```python
# At ForkNode: branch isolation
branch_registries = []
for branch_id in range(n_branches):
    # Shallow copy: shares original rows, can append independently
    branch_reg = registry.branch_copy(branch_id=branch_id)
    branch_registries.append(branch_reg)

# Branch 0 adds augmented samples
branch_registries[0].add_augmented_samples(...)  # +15 samples

# Branch 1 has different augmentation
branch_registries[1].add_augmented_samples(...)  # +10 samples

# At JoinNode: merge strategies
merged_registry = SampleRegistry.merge(
    branch_registries,
    strategy="union"  # Combine all unique samples
)
# or strategy="intersection" for shared samples only
```

---

## TargetStore

The `TargetStore` manages target values with **bidirectional** transformation chains.

### Interface

```python
class TargetStore:
    """Target value storage with bidirectional transformation chain support.

    Maintains multiple versions of targets (raw, scaled, encoded)
    with fitted transformers for both forward and inverse transformation.
    Critical for returning predictions in original scale.
    """

    def __init__(self):
        self._versions: Dict[str, np.ndarray] = {}
        self._transformers: Dict[str, TransformerMixin] = {}
        self._transform_chain: List[str] = []  # Order of transforms
        self._task_type: Optional[TaskType] = None
        self._task_type_forced: bool = False

    # === Target Management ===

    def set_targets(
        self,
        y: np.ndarray,
        version: str = "raw"
    ) -> None:
        """Set target values for a version."""
        ...

    def add_transformed(
        self,
        version: str,
        y: np.ndarray,
        transformer: TransformerMixin,
        parent_version: str = "raw"
    ) -> None:
        """Add transformed target version.

        Stores the fitted transformer for inverse transformation.
        """
        self._versions[version] = y
        self._transformers[version] = transformer
        self._transform_chain.append(version)

    def get(
        self,
        version: str = "raw",
        sample_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Get target values."""
        ...

    def inverse_transform(
        self,
        y: np.ndarray,
        from_version: str
    ) -> np.ndarray:
        """Inverse transform predictions back to raw scale.

        Applies inverse transforms in reverse order from from_version
        back to "raw".

        Args:
            y: Predictions in transformed space
            from_version: Version the predictions are in

        Returns:
            Predictions in original (raw) scale
        """
        # Find position in transform chain
        if from_version not in self._transform_chain:
            if from_version == "raw":
                return y  # Already raw
            raise ValueError(f"Unknown version: {from_version}")

        idx = self._transform_chain.index(from_version)

        # Apply inverse transforms in reverse order
        result = y.copy()
        for version in reversed(self._transform_chain[:idx + 1]):
            transformer = self._transformers.get(version)
            if transformer and hasattr(transformer, 'inverse_transform'):
                result = transformer.inverse_transform(result.reshape(-1, 1)).ravel()

        return result

    def get_active_version(self) -> str:
        """Get the most recent transform version."""
        return self._transform_chain[-1] if self._transform_chain else "raw"

    # === Task Type ===

    @property
    def task_type(self) -> TaskType:
        """Detected or set task type."""
        ...

    def set_task_type(
        self,
        task_type: TaskType,
        forced: bool = True
    ) -> None:
        """Set task type explicitly."""
        ...

    @property
    def n_classes(self) -> int:
        """Number of classes for classification."""
        ...
```

---

## ViewSpec and ViewResolver

Views provide lazy, declarative access to data subsets.

### ViewSpec

```python
@dataclass(frozen=True)
class ViewSpec:
    """Declarative specification of a data view.

    A ViewSpec describes WHAT data to access, not HOW.
    The ViewResolver handles materialization.

    Attributes:
        block_ids: List of block IDs to include
        partition: Partition filter ("train", "val", "test", None for all)
        fold_id: Fold filter (None for all folds)
        fold_role: Role in fold ("train" or "val")
        include_augmented: Include augmented samples
        include_excluded: Include excluded samples
        processing_filter: Which processing slots to include
        sample_filter: Additional Polars filter expression
        target_version: Target version to use
    """
    block_ids: Tuple[str, ...]
    partition: Optional[str] = None
    fold_id: Optional[int] = None
    fold_role: Optional[str] = None  # "train" or "val" within fold
    include_augmented: bool = True
    include_excluded: bool = False
    processing_filter: Optional[Tuple[str, ...]] = None
    sample_filter: Optional[pl.Expr] = None
    target_version: str = "raw"

    def with_partition(self, partition: str) -> "ViewSpec":
        """Return new ViewSpec with different partition."""
        return replace(self, partition=partition)

    def with_fold(self, fold_id: int, role: str = "train") -> "ViewSpec":
        """Return new ViewSpec for specific fold."""
        return replace(self, fold_id=fold_id, fold_role=role)

    def for_transform(self) -> "ViewSpec":
        """Return ViewSpec suitable for transformation (include all)."""
        return replace(
            self,
            include_augmented=True,
            include_excluded=True
        )
```

### ViewResolver

```python
class ViewResolver:
    """Resolves ViewSpec to materialized data.

    Handles:
    - Sample index resolution from registry
    - Block data slicing
    - Multi-block concatenation
    - Layout formatting (2D/3D)
    """

    def __init__(
        self,
        block_store: FeatureBlockStore,
        sample_registry: SampleRegistry,
        target_store: TargetStore
    ):
        self._blocks = block_store
        self._registry = sample_registry
        self._targets = target_store

    def resolve_indices(self, view: ViewSpec) -> np.ndarray:
        """Resolve ViewSpec to sample indices."""
        return self._registry.get_indices(
            partition=view.partition,
            fold_id=view.fold_id,
            include_augmented=view.include_augmented,
            include_excluded=view.include_excluded,
            filter_expr=view.sample_filter
        )

    def materialize(
        self,
        view: ViewSpec,
        layout: str = "2d",
        concat_sources: bool = True,
        return_source_dict: bool = False
    ) -> Tuple[Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]], np.ndarray]:
        """Materialize ViewSpec to (X, y) arrays.

        Args:
            view: ViewSpec describing the data subset
            layout: "2d" (flatten processings) or "3d" (keep dim)
            concat_sources: Concatenate multiple blocks or return list.
                Ignored if return_source_dict=True.
            return_source_dict: If True, return Dict[source_name, array] instead
                of concatenated array or list. Useful for multi-head models
                that expect named inputs. Source names are taken from block
                metadata ("source" key) or block_id as fallback.

        Returns:
            Tuple of (X, y) where X is:
              - np.ndarray if single source or concat_sources=True
              - List[np.ndarray] if concat_sources=False and not return_source_dict
              - Dict[str, np.ndarray] if return_source_dict=True

        Example:
            >>> # Get sources as dict for multi-head model
            >>> X_dict, y = resolver.materialize(view, return_source_dict=True)
            >>> print(X_dict.keys())  # dict_keys(['NIR', 'markers'])
            >>> model.fit({"nir_input": X_dict["NIR"], "marker_input": X_dict["markers"]}, y)
        """
        indices = self.resolve_indices(view)

        # Get X from each block
        X_list = []
        for block_id in view.block_ids:
            block = self._blocks.get(block_id)
            X_block = block.data[indices]  # (n, p, f)

            if view.processing_filter:
                # Filter to specific processings
                p_indices = [
                    block.processing_ids.index(p)
                    for p in view.processing_filter
                    if p in block.processing_ids
                ]
                X_block = X_block[:, p_indices, :]

            if layout == "2d":
                # Flatten: (n, p, f) → (n, p*f)
                X_block = X_block.reshape(X_block.shape[0], -1)

            X_list.append(X_block)

        # Get y
        y = self._targets.get(
            version=view.target_version,
            sample_indices=indices
        )

        # Combine X based on options
        if return_source_dict:
            # Return dict mapping source_name -> array
            X_dict = {}
            for block_id, X_block in zip(view.block_ids, X_list):
                block = self._blocks.get(block_id)
                source_name = block.metadata.get("source", block_id)
                X_dict[source_name] = X_block
            return X_dict, y
        elif concat_sources and len(X_list) > 1:
            X = np.concatenate(X_list, axis=-1)
        elif len(X_list) == 1:
            X = X_list[0]
        else:
            X = X_list

        return X, y

    def materialize_X(
        self,
        view: ViewSpec,
        layout: str = "2d",
        concat_sources: bool = True,
        return_source_dict: bool = False
    ) -> Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]:
        """Materialize only X (no targets)."""
        X, _ = self.materialize(view, layout, concat_sources, return_source_dict)
        return X


class CachingViewResolver(ViewResolver):
    """ViewResolver with LRU caching for repeated materializations.

    In many workflows (e.g., hyperparameter tuning, cross-validation),
    the same ViewSpec is materialized many times. This class caches
    materialized arrays to avoid redundant computation.

    Usage:
        resolver = CachingViewResolver(
            block_store, sample_registry, target_store,
            max_cache_size_mb=500  # Cache up to 500MB
        )

        # First call materializes and caches
        X1, y1 = resolver.materialize(view)

        # Second call returns cached result (fast)
        X2, y2 = resolver.materialize(view)  # Cache hit!

    Cache Invalidation:
        Cache is automatically invalidated when:
        - Block data changes (detected via lineage hash)
        - Registry changes (new samples, fold changes)
        - Memory pressure triggers eviction

    Thread Safety:
        Uses threading.Lock for cache access, safe for concurrent use.
    """

    def __init__(
        self,
        block_store: FeatureBlockStore,
        sample_registry: SampleRegistry,
        target_store: TargetStore,
        max_cache_size_mb: float = 500.0,
        max_entries: int = 100
    ):
        super().__init__(block_store, sample_registry, target_store)

        import threading
        from collections import OrderedDict

        self._cache: OrderedDict[str, Tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._cache_sizes: Dict[str, float] = {}  # MB per entry
        self._max_cache_mb = max_cache_size_mb
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0

    def _cache_key(self, view: ViewSpec, layout: str, concat_sources: bool) -> str:
        """Generate cache key from ViewSpec and options."""
        import hashlib
        import json

        key_parts = [
            view.partition or "",
            str(view.fold_id),
            str(view.block_ids),
            str(view.include_augmented),
            str(view.include_excluded),
            str(view.target_version),
            layout,
            str(concat_sources)
        ]

        # Include block lineage hashes for invalidation
        for block_id in view.block_ids:
            block = self._blocks.get(block_id)
            key_parts.append(block.lineage_hash)

        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _current_cache_size_mb(self) -> float:
        """Get current cache size in MB."""
        return sum(self._cache_sizes.values())

    def _evict_oldest(self) -> None:
        """Evict oldest cache entries until under limit."""
        while (
            self._current_cache_size_mb() > self._max_cache_mb
            or len(self._cache) > self._max_entries
        ):
            if not self._cache:
                break
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            del self._cache_sizes[oldest_key]

    def materialize(
        self,
        view: ViewSpec,
        layout: str = "2d",
        concat_sources: bool = True,
        return_source_dict: bool = False
    ) -> Tuple[Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]], np.ndarray]:
        """Materialize with caching."""
        # Dict return mode is not cached (complex key)
        if return_source_dict:
            return super().materialize(view, layout, concat_sources, return_source_dict)

        cache_key = self._cache_key(view, layout, concat_sources)

        with self._lock:
            if cache_key in self._cache:
                self._cache_hits += 1
                # Move to end (LRU)
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key]

        # Cache miss - materialize
        self._cache_misses += 1
        result = super().materialize(view, layout, concat_sources, return_source_dict)

        # Cache the result
        X, y = result
        size_mb = (X.nbytes + y.nbytes) / (1024 ** 2)

        with self._lock:
            self._evict_oldest()
            self._cache[cache_key] = result
            self._cache_sizes[cache_key] = size_mb

        return result

    def clear_cache(self) -> None:
        """Clear all cached views."""
        with self._lock:
            self._cache.clear()
            self._cache_sizes.clear()

    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses),
            "entries": len(self._cache),
            "size_mb": self._current_cache_size_mb()
        }
```

---

## DatasetContext

The `DatasetContext` bundles everything needed for pipeline execution.

### Interface

```python
@dataclass
class DatasetContext:
    """Complete context for pipeline execution.

    Bundles all data components and provides convenience methods
    for common operations during training and prediction.
    """
    name: str
    block_store: FeatureBlockStore
    sample_registry: SampleRegistry
    target_store: TargetStore
    resolver: ViewResolver

    # Current state
    active_block_ids: List[str]
    current_view: ViewSpec

    # Artifacts (populated during training)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_arrays(
        cls,
        X: Union[np.ndarray, List[np.ndarray]],
        y: np.ndarray,
        name: str = "dataset",
        headers: Optional[List[str]] = None,
        metadata: Optional[Dict[str, List[Any]]] = None,
        partition: str = "train"
    ) -> "DatasetContext":
        """Create context from raw arrays (convenience factory)."""
        store = FeatureBlockStore()
        registry = SampleRegistry()
        targets = TargetStore()

        # Register sources
        if isinstance(X, list):
            block_ids = [
                store.register_source(x, source_name=f"source_{i}")
                for i, x in enumerate(X)
            ]
        else:
            block_ids = [store.register_source(X)]

        # Add samples
        registry.add_samples(
            n_samples=X[0].shape[0] if isinstance(X, list) else X.shape[0],
            partition=partition,
            metadata=metadata
        )

        # Set targets
        targets.set_targets(y)

        # Create resolver
        resolver = ViewResolver(store, registry, targets)

        # Initial view
        view = ViewSpec(
            block_ids=tuple(block_ids),
            partition=partition
        )

        return cls(
            name=name,
            block_store=store,
            sample_registry=registry,
            target_store=targets,
            resolver=resolver,
            active_block_ids=block_ids,
            current_view=view
        )

    # === Convenience Methods ===

    def x(
        self,
        partition: Optional[str] = None,
        fold_id: Optional[int] = None,
        fold_role: str = "train",
        layout: str = "2d"
    ) -> np.ndarray:
        """Get features for partition/fold."""
        view = self.current_view
        if partition:
            view = view.with_partition(partition)
        if fold_id is not None:
            view = view.with_fold(fold_id, fold_role)
        return self.resolver.materialize_X(view, layout)

    def y(
        self,
        partition: Optional[str] = None,
        fold_id: Optional[int] = None,
        fold_role: str = "train"
    ) -> np.ndarray:
        """Get targets for partition/fold."""
        view = self.current_view
        if partition:
            view = view.with_partition(partition)
        if fold_id is not None:
            view = view.with_fold(fold_id, fold_role)
        indices = self.resolver.resolve_indices(view)
        return self.target_store.get(
            version=view.target_version,
            sample_indices=indices
        )

    def xy(
        self,
        partition: Optional[str] = None,
        fold_id: Optional[int] = None,
        fold_role: str = "train",
        layout: str = "2d"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get (X, y) tuple."""
        view = self.current_view
        if partition:
            view = view.with_partition(partition)
        if fold_id is not None:
            view = view.with_fold(fold_id, fold_role)
        return self.resolver.materialize(view, layout)

    @property
    def folds(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """CV fold indices."""
        return self.sample_registry.folds

    @property
    def n_samples(self) -> int:
        """Total sample count."""
        return len(self.sample_registry._df)

    @property
    def n_features(self) -> int:
        """Feature count (first block, 2D)."""
        block = self.block_store.get(self.active_block_ids[0])
        return block.n_processings * block.n_features
```

---

## Aggregation and Repetition Handling

### Sample Repetitions in NIRS Data

NIRS data often contains multiple measurements per biological sample (repetitions). The data layer provides first-class support for this pattern:

```python
# Example: 100 biological samples, 3 repetitions each = 300 rows
# Registry tracks the relationship:
#   _sample_id | _bio_id | repetition_idx
#   0          | 0       | 0
#   1          | 0       | 1
#   2          | 0       | 2
#   3          | 1       | 0
#   ...
```

### RepetitionHandler

```python
class RepetitionHandler:
    """Handles sample repetitions in NIRS datasets.

    Supports multiple strategies for dealing with repetitions:
    - Keep as-is (each repetition is a separate sample)
    - Convert to separate sources (one source per repetition)
    - Convert to processings (one processing slot per repetition)
    - Aggregate (mean/median across repetitions)
    """

    def __init__(self, registry: SampleRegistry):
        self._registry = registry

    def detect_repetitions(
        self,
        column: str,
        expected_reps: Optional[int] = None
    ) -> RepetitionInfo:
        """Detect repetition structure from a grouping column.

        Args:
            column: Metadata column that identifies biological samples
            expected_reps: Expected number of repetitions (validates uniformity)

        Returns:
            RepetitionInfo with detected structure
        """
        df = self._registry._df

        # Group by the column and count
        groups = df.group_by(column).agg(pl.count().alias("n_reps"))
        rep_counts = groups["n_reps"].unique().to_list()

        if len(rep_counts) == 1:
            uniform = True
            n_reps = rep_counts[0]
        else:
            uniform = False
            n_reps = max(rep_counts)

        if expected_reps and expected_reps != n_reps:
            raise DataValidationError(
                f"Expected {expected_reps} repetitions, found {n_reps}",
                field="repetitions",
                expected=str(expected_reps),
                got=str(n_reps)
            )

        return RepetitionInfo(
            column=column,
            n_reps=n_reps,
            uniform=uniform,
            bio_ids=groups[column].to_list()
        )

    def assign_bio_ids(self, column: str) -> None:
        """Assign _bio_id and repetition_idx based on grouping column."""
        df = self._registry._df

        # Create bio_id mapping
        unique_values = df[column].unique().sort()
        bio_id_map = {v: i for i, v in enumerate(unique_values)}

        # Add bio_id column
        df = df.with_columns([
            pl.col(column).replace(bio_id_map).alias("_bio_id")
        ])

        # Add repetition_idx within each bio_id
        df = df.with_columns([
            pl.col("_sample_id").cum_count().over("_bio_id").alias("repetition_idx") - 1
        ])

        self._registry._df = df

    def to_sources(
        self,
        block_store: FeatureBlockStore,
        source_block_id: str,
        source_names: Optional[List[str]] = None
    ) -> List[str]:
        """Convert repetitions to separate source blocks.

        Each repetition becomes a separate FeatureBlock with aligned rows.

        Args:
            block_store: Block store to create new blocks in
            source_block_id: Original block with all repetitions
            source_names: Names for the new sources (default: "rep_0", "rep_1", ...)

        Returns:
            List of new block IDs, one per repetition
        """
        info = self._get_repetition_info()
        block = block_store.get(source_block_id)
        new_block_ids = []

        for rep_idx in range(info.n_reps):
            # Get sample indices for this repetition
            mask = self._registry._df["repetition_idx"] == rep_idx
            indices = self._registry._df.filter(mask)["_row_index"].to_numpy()

            # Extract data for these samples
            rep_data = block.data[indices]

            # Create new block
            name = source_names[rep_idx] if source_names else f"rep_{rep_idx}"
            new_id = block_store.register_source(
                data=rep_data,
                headers=block.headers,
                header_unit=block.header_unit,
                source_name=name,
                metadata={"repetition_idx": rep_idx, "parent_block": source_block_id}
            )
            new_block_ids.append(new_id)

        return new_block_ids


@dataclass
class RepetitionInfo:
    """Information about repetition structure."""
    column: str
    n_reps: int
    uniform: bool  # All bio_ids have same number of reps
    bio_ids: List[Any]


@dataclass
class UnequelRepsStrategy:
    """Strategy for handling unequal repetition counts."""
    method: Literal["error", "drop", "pad", "mask"]
    pad_value: float = 0.0
```

### Aggregation at Prediction Time

```python
@dataclass
class AggregationSpec:
    """Specification for sample-level prediction aggregation."""
    by: str = "_bio_id"          # Column to group by
    method: str = "mean"         # "mean", "median", "vote" (classification)
    exclude_outliers: bool = False
    outlier_threshold: float = 0.95  # IQR multiplier for outlier detection

    def aggregate_predictions(
        self,
        y_pred: np.ndarray,
        registry: SampleRegistry
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate predictions by bio_id.

        Args:
            y_pred: Predictions at measurement level
            registry: Sample registry with aggregation relationships

        Returns:
            Tuple of (aggregated predictions, bio_ids)
        """
        df = registry._df.with_columns([
            pl.lit(y_pred).alias("y_pred")
        ])

        if self.exclude_outliers:
            df = self._remove_outliers(df)

        if self.method == "mean":
            agg_expr = pl.col("y_pred").mean()
        elif self.method == "median":
            agg_expr = pl.col("y_pred").median()
        elif self.method == "vote":
            agg_expr = pl.col("y_pred").mode().first()
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")

        result = df.group_by(self.by).agg([
            agg_expr.alias("y_pred_agg")
        ]).sort(self.by)

        return result["y_pred_agg"].to_numpy(), result[self.by].to_numpy()

    def _remove_outliers(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove outlier predictions within each group."""
        return df.with_columns([
            (
                (pl.col("y_pred") >= pl.col("y_pred").quantile(0.25).over(self.by) -
                 self.outlier_threshold * (pl.col("y_pred").quantile(0.75).over(self.by) -
                                           pl.col("y_pred").quantile(0.25).over(self.by))) &
                (pl.col("y_pred") <= pl.col("y_pred").quantile(0.75).over(self.by) +
                 self.outlier_threshold * (pl.col("y_pred").quantile(0.75).over(self.by) -
                                           pl.col("y_pred").quantile(0.25).over(self.by)))
            ).alias("_is_valid")
        ]).filter(pl.col("_is_valid"))
```

### Multi-Source View with Branch Disambiguation

When selecting sources after branching, explicit disambiguation is required:

```python
@dataclass(frozen=True)
class ViewSpec:
    # ... existing fields ...
    source_filter: Optional[Dict[str, Any]] = None  # Source selection with branch context

    def select_source(
        self,
        source_name: str,
        branch_id: Optional[int] = None
    ) -> "ViewSpec":
        """Select a specific source, optionally within a branch context.

        Args:
            source_name: Base source name (e.g., "nir")
            branch_id: If set, select source from specific branch context

        Returns:
            New ViewSpec with source filter applied

        Example:
            # After branching, sources may be named "nir_branch_0", "nir_branch_1"
            view.select_source("nir", branch_id=0)  # Selects "nir_branch_0"
        """
        filter_spec = {"name": source_name}
        if branch_id is not None:
            filter_spec["branch_id"] = branch_id
        return replace(self, source_filter=filter_spec)
```

---

## Data Operations

Common data operations as standalone functions:

### Transform Operations

```python
def apply_transform(
    context: DatasetContext,
    transformer: TransformerMixin,
    source_block_id: str,
    processing_name: Optional[str] = None,
    fit_partition: str = "train"
) -> Tuple[DatasetContext, str]:
    """Apply sklearn transformer to a block.

    Args:
        context: Current dataset context
        transformer: sklearn-compatible transformer
        source_block_id: Block to transform
        processing_name: Name for new processing slot
        fit_partition: Partition to fit on

    Returns:
        Tuple of (updated context, new block_id)
    """
    block = context.block_store.get(source_block_id)

    # Get training data for fit
    train_indices = context.sample_registry.get_indices(
        partition=fit_partition
    )
    X_train = block.data[train_indices].reshape(len(train_indices), -1)

    # Fit and transform
    transformer.fit(X_train)

    # Transform all data
    X_all = block.data.reshape(block.n_samples, -1)
    X_transformed = transformer.transform(X_all)
    X_transformed = X_transformed.reshape(
        block.n_samples, 1, -1
    )  # Single processing

    # Register new block
    new_block_id = context.block_store.register_transform(
        parent_id=source_block_id,
        data=X_transformed,
        transform_info=TransformInfo(
            transform_class=f"{transformer.__class__.__module__}.{transformer.__class__.__name__}",
            transform_params=transformer.get_params(),
            source_processing=block.processing_ids[-1],
            target_processing=processing_name or transformer.__class__.__name__
        )
    )

    # Update context
    new_context = replace(
        context,
        active_block_ids=[new_block_id]  # Replace with new
    )

    return new_context, new_block_id
```

### Aggregation Operations

```python
def aggregate_view(
    context: DatasetContext,
    view: ViewSpec,
    by: Union[str, List[str]],
    method: str = "mean"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate features and targets by group.

    Args:
        context: Dataset context
        view: View to aggregate
        by: Column(s) to group by
        method: Aggregation method ("mean", "median")

    Returns:
        Tuple of (X_agg, y_agg, group_ids)
    """
    groups = context.sample_registry.get_aggregation_groups(by)
    X, y = context.resolver.materialize(view)

    group_ids = list(groups.keys())
    X_agg = np.zeros((len(groups), X.shape[1]))
    y_agg = np.zeros(len(groups))

    for i, (gid, indices) in enumerate(groups.items()):
        if method == "mean":
            X_agg[i] = X[indices].mean(axis=0)
            y_agg[i] = y[indices].mean()
        elif method == "median":
            X_agg[i] = np.median(X[indices], axis=0)
            y_agg[i] = np.median(y[indices])

    return X_agg, y_agg, np.array(group_ids)
```

---

## Layout Strategies and Model Input Construction

### Overview

A critical aspect of NIRS4ALL is transforming heterogeneous data structures into model-compatible inputs. Data in the system can vary across multiple dimensions:

- **Samples**: Observations (possibly with repetitions)
- **Sources**: Different data origins (NIR, markers, predictions, etc.)
- **Processings**: Different preprocessing variants per source
- **Branches**: Parallel pipeline execution paths
- **Features**: Variable feature counts per source/processing

This section defines the **LayoutStrategy** system for constructing model inputs from these heterogeneous structures.

### The Layout Problem

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Dimensions                              │
│                                                                  │
│  Sources:     [NIR: 500 features] [Markers: 50 features] [...]  │
│  Processings: [raw, SNV, MSC] per source                        │
│  Branches:    [branch_0, branch_1, ...]                         │
│  Repetitions: [rep_0, rep_1, rep_2, rep_3] per sample           │
│                                                                  │
│  How to feed this to a model?                                   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ sklearn model: expects (N, F) → flatten all             │    │
│  │ Conv1D:        expects (N, L, C) → stack processings    │    │
│  │ Multi-head NN: expects {src: array, ...} → dict format  │    │
│  │ Transformer:   expects (N, seq_len, d_model) → special  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### LayoutSpec: Declarative Layout Configuration

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Literal
from enum import Enum


class LayoutMode(Enum):
    """Target output layout modes."""
    FLAT_2D = "2d"              # (N, F) - flatten everything
    VOLUME_3D = "3d"            # (N, C, L) - channels-first (PyTorch)
    VOLUME_3D_TRANSPOSE = "3d_t" # (N, L, C) - channels-last (TensorFlow)
    MULTI_INPUT = "dict"        # Dict[str, ndarray] - multi-head models
    SEQUENCE = "seq"            # (N, seq_len, features) - transformers
    IMAGE_2D = "image_2d"       # (N, H, W, C) - 2D spatial data


class JoinStrategy(Enum):
    """Strategy for joining data across dimensions."""
    CONCAT = "concat"           # Horizontal concatenation
    STACK = "stack"             # Stack along new axis
    INTERLEAVE = "interleave"   # Interleave alternating elements
    DICT = "dict"               # Keep as dictionary
    DUPLICATE = "duplicate"     # Duplicate model per item (branches default)


class PaddingStrategy(Enum):
    """Strategy for handling dimension mismatches."""
    ERROR = "error"             # Raise error on mismatch
    PAD_ZERO = "pad_zero"       # Pad with zeros
    PAD_NAN = "pad_nan"         # Pad with NaN (for masking)
    PAD_MEAN = "pad_mean"       # Pad with column mean
    TRUNCATE = "truncate"       # Truncate to shortest
    INTERPOLATE = "interpolate" # Resample to target size


@dataclass
class LayoutSpec:
    """Specification for constructing model input from data blocks.

    Controls how data from multiple sources, processings, branches,
    and repetitions are combined into a model-compatible format.

    Attributes:
        mode: Target layout mode (2d, 3d, dict, etc.)
        source_join: How to join multiple sources
        processing_join: How to join multiple processings
        branch_join: How to join data from branches
        repetition_join: How to handle sample repetitions
        on_asymmetric: Strategy for mismatched dimensions
        target_shape: Optional explicit target shape (for interpolation)
        channel_order: Channel ordering for 3D layouts
        feature_axis: Which axis contains features
    """
    mode: LayoutMode = LayoutMode.FLAT_2D
    source_join: JoinStrategy = JoinStrategy.CONCAT
    processing_join: JoinStrategy = JoinStrategy.CONCAT
    branch_join: JoinStrategy = JoinStrategy.DUPLICATE
    repetition_join: JoinStrategy = JoinStrategy.CONCAT
    on_asymmetric: PaddingStrategy = PaddingStrategy.ERROR
    target_shape: Optional[Tuple[int, ...]] = None
    channel_order: str = "last"  # "first" for PyTorch, "last" for TensorFlow
    feature_axis: int = -1

    # Advanced options
    source_order: Optional[List[str]] = None  # Explicit source ordering
    processing_filter: Optional[List[str]] = None  # Select specific processings
    branch_filter: Optional[List[int]] = None  # Select specific branches
    mask_invalid: bool = False  # Return validity mask for padded values

    def validate(self) -> None:
        """Validate layout specification consistency."""
        if self.mode == LayoutMode.FLAT_2D:
            if self.source_join == JoinStrategy.STACK:
                raise ValueError(
                    "Cannot use STACK source_join with 2D mode. "
                    "Use CONCAT or switch to 3D mode."
                )
        if self.mode in (LayoutMode.VOLUME_3D, LayoutMode.VOLUME_3D_TRANSPOSE):
            if self.on_asymmetric == PaddingStrategy.ERROR:
                # Warn user about potential issues
                import warnings
                warnings.warn(
                    "3D layout with on_asymmetric='error' may fail if sources "
                    "have different feature counts. Consider 'pad_zero' or "
                    "'interpolate'.",
                    UserWarning
                )


# Common presets for convenience
LAYOUT_SKLEARN = LayoutSpec(
    mode=LayoutMode.FLAT_2D,
    source_join=JoinStrategy.CONCAT,
    processing_join=JoinStrategy.CONCAT
)

LAYOUT_TENSORFLOW_1D = LayoutSpec(
    mode=LayoutMode.VOLUME_3D_TRANSPOSE,
    source_join=JoinStrategy.STACK,
    processing_join=JoinStrategy.STACK,
    on_asymmetric=PaddingStrategy.PAD_ZERO,
    channel_order="last"
)

LAYOUT_PYTORCH_1D = LayoutSpec(
    mode=LayoutMode.VOLUME_3D,
    source_join=JoinStrategy.STACK,
    processing_join=JoinStrategy.STACK,
    on_asymmetric=PaddingStrategy.PAD_ZERO,
    channel_order="first"
)

LAYOUT_MULTI_HEAD = LayoutSpec(
    mode=LayoutMode.MULTI_INPUT,
    source_join=JoinStrategy.DICT,
    processing_join=JoinStrategy.CONCAT
)

LAYOUT_TRANSFORMER = LayoutSpec(
    mode=LayoutMode.SEQUENCE,
    source_join=JoinStrategy.CONCAT,
    processing_join=JoinStrategy.STACK,
    on_asymmetric=PaddingStrategy.PAD_ZERO
)
```

### LayoutResolver: Materializing Data with Layout

```python
class LayoutResolver:
    """Resolves LayoutSpec to model-compatible arrays.

    Handles all combinations of sources, processings, branches,
    and repetitions according to the specified layout strategy.
    """

    def __init__(
        self,
        block_store: FeatureBlockStore,
        sample_registry: SampleRegistry
    ):
        self._blocks = block_store
        self._registry = sample_registry

    def resolve(
        self,
        view: ViewSpec,
        layout: LayoutSpec
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Resolve view with layout specification.

        Args:
            view: ViewSpec defining data subset
            layout: LayoutSpec defining output format

        Returns:
            Model-compatible data (array or dict of arrays)

        Raises:
            LayoutError: If layout cannot be satisfied
        """
        layout.validate()

        # Collect raw data from all blocks
        block_data = self._collect_block_data(view)

        # Apply layout transformations
        if layout.mode == LayoutMode.MULTI_INPUT:
            return self._to_multi_input(block_data, layout)
        elif layout.mode == LayoutMode.FLAT_2D:
            return self._to_flat_2d(block_data, layout)
        elif layout.mode in (LayoutMode.VOLUME_3D, LayoutMode.VOLUME_3D_TRANSPOSE):
            return self._to_volume_3d(block_data, layout)
        elif layout.mode == LayoutMode.SEQUENCE:
            return self._to_sequence(block_data, layout)
        elif layout.mode == LayoutMode.IMAGE_2D:
            return self._to_image_2d(block_data, layout)
        else:
            raise ValueError(f"Unknown layout mode: {layout.mode}")

    def _collect_block_data(
        self,
        view: ViewSpec
    ) -> Dict[str, np.ndarray]:
        """Collect data from all blocks in view."""
        indices = self._registry.get_indices(
            partition=view.partition,
            fold_id=view.fold_id,
            include_augmented=view.include_augmented
        )

        data = {}
        for block_id in view.block_ids:
            block = self._blocks.get(block_id)
            block_data = block.data[indices]
            source_name = block.metadata.get("source", block_id)
            data[source_name] = block_data

        return data

    def _to_flat_2d(
        self,
        block_data: Dict[str, np.ndarray],
        layout: LayoutSpec
    ) -> np.ndarray:
        """Convert to flat 2D array (samples, features)."""
        arrays = []

        # Apply source ordering if specified
        sources = layout.source_order or list(block_data.keys())

        for source in sources:
            if source not in block_data:
                if layout.on_asymmetric == PaddingStrategy.ERROR:
                    raise LayoutError(f"Source '{source}' not found in data")
                continue

            arr = block_data[source]  # (N, P, F)

            # Flatten processings × features
            n_samples = arr.shape[0]
            arr_2d = arr.reshape(n_samples, -1)
            arrays.append(arr_2d)

        if layout.source_join == JoinStrategy.CONCAT:
            return np.concatenate(arrays, axis=1)
        elif layout.source_join == JoinStrategy.INTERLEAVE:
            return self._interleave_arrays(arrays, axis=1)
        else:
            raise ValueError(
                f"Unsupported source_join for 2D: {layout.source_join}"
            )

    def _to_volume_3d(
        self,
        block_data: Dict[str, np.ndarray],
        layout: LayoutSpec
    ) -> np.ndarray:
        """Convert to 3D volume (samples, channels, length) or (N, L, C)."""
        # Collect all processings from all sources
        channels = []
        sources = layout.source_order or list(block_data.keys())

        for source in sources:
            arr = block_data[source]  # (N, P, F)

            # Handle dimension mismatches
            arr = self._normalize_dimensions(arr, layout)

            if layout.processing_join == JoinStrategy.STACK:
                # Each processing becomes a channel
                for p in range(arr.shape[1]):
                    channels.append(arr[:, p, :])
            elif layout.processing_join == JoinStrategy.CONCAT:
                # Flatten processings, keep as single channel
                n_samples = arr.shape[0]
                channels.append(arr.reshape(n_samples, -1))

        # Stack channels
        if len(channels) == 0:
            raise LayoutError("No channels to stack")

        # Ensure all channels have same length
        max_len = max(c.shape[1] for c in channels)
        padded = []
        for c in channels:
            if c.shape[1] < max_len:
                c = self._pad_features(c, max_len, layout.on_asymmetric)
            padded.append(c)

        stacked = np.stack(padded, axis=1)  # (N, C, L)

        if layout.channel_order == "last":
            stacked = np.transpose(stacked, (0, 2, 1))  # (N, L, C)

        return stacked

    def _to_multi_input(
        self,
        block_data: Dict[str, np.ndarray],
        layout: LayoutSpec
    ) -> Dict[str, np.ndarray]:
        """Convert to multi-input dictionary format."""
        result = {}

        for source_name, arr in block_data.items():
            # Apply per-source layout
            if layout.processing_join == JoinStrategy.CONCAT:
                # Flatten to 2D per source
                n_samples = arr.shape[0]
                result[source_name] = arr.reshape(n_samples, -1)
            elif layout.processing_join == JoinStrategy.STACK:
                # Keep 3D structure per source
                result[source_name] = arr
            else:
                result[source_name] = arr

        return result

    def _normalize_dimensions(
        self,
        arr: np.ndarray,
        layout: LayoutSpec
    ) -> np.ndarray:
        """Normalize array dimensions according to layout strategy."""
        target = layout.target_shape

        if target is None:
            return arr

        current_features = arr.shape[-1]
        target_features = target[-1] if len(target) > 0 else current_features

        if current_features == target_features:
            return arr

        if layout.on_asymmetric == PaddingStrategy.INTERPOLATE:
            # Resample features to target size
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, current_features)
            x_new = np.linspace(0, 1, target_features)
            interpolator = interp1d(x_old, arr, axis=-1, kind='linear')
            return interpolator(x_new)
        elif layout.on_asymmetric in (PaddingStrategy.PAD_ZERO, PaddingStrategy.PAD_NAN):
            return self._pad_features(arr, target_features, layout.on_asymmetric)
        elif layout.on_asymmetric == PaddingStrategy.TRUNCATE:
            return arr[..., :target_features]
        elif layout.on_asymmetric == PaddingStrategy.ERROR:
            raise LayoutError(
                f"Feature dimension mismatch: {current_features} vs {target_features}. "
                f"Set on_asymmetric to 'pad_zero', 'interpolate', or 'truncate'."
            )

        return arr

    def _pad_features(
        self,
        arr: np.ndarray,
        target_len: int,
        strategy: PaddingStrategy
    ) -> np.ndarray:
        """Pad array to target feature length."""
        current_len = arr.shape[-1]
        if current_len >= target_len:
            return arr[..., :target_len]

        pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, target_len - current_len)]

        if strategy == PaddingStrategy.PAD_ZERO:
            return np.pad(arr, pad_width, mode='constant', constant_values=0)
        elif strategy == PaddingStrategy.PAD_NAN:
            return np.pad(arr, pad_width, mode='constant', constant_values=np.nan)
        elif strategy == PaddingStrategy.PAD_MEAN:
            mean_val = np.nanmean(arr)
            return np.pad(arr, pad_width, mode='constant', constant_values=mean_val)
        else:
            return np.pad(arr, pad_width, mode='constant', constant_values=0)

    def _interleave_arrays(
        self,
        arrays: List[np.ndarray],
        axis: int
    ) -> np.ndarray:
        """Interleave multiple arrays along an axis."""
        # All arrays must have same shape
        if len(set(arr.shape for arr in arrays)) > 1:
            raise LayoutError("Cannot interleave arrays with different shapes")

        # Stack and transpose to interleave
        stacked = np.stack(arrays, axis=axis + 1)
        shape = list(stacked.shape)
        # Merge interleave axis with target axis
        new_shape = shape[:axis] + [-1] + shape[axis + 2:]
        return stacked.reshape(new_shape)


class LayoutError(Exception):
    """Raised when layout resolution fails."""
    pass
```

### ModelInputSpec: Model-Declared Input Requirements

Models can declare their input requirements through `ModelInputSpec`. This allows the pipeline to automatically construct appropriate inputs.

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelWithInputSpec(Protocol):
    """Protocol for models that declare input requirements."""

    def get_input_spec(self) -> "ModelInputSpec":
        """Return the model's input specification."""
        ...


@dataclass
class ModelInputSpec:
    """Specification of a model's input requirements.

    Models can define this to communicate their expected input format
    to the pipeline, enabling automatic layout resolution.

    Attributes:
        layout: Required LayoutSpec (or LayoutMode shorthand)
        input_names: For multi-input models, expected input keys
        required_sources: Sources that must be present
        min_features: Minimum feature count per input
        accepts_3d: Whether model accepts 3D input
        accepts_variable_length: Whether model handles variable-length sequences
        preferred_dtype: Preferred data type
    """
    layout: Union[LayoutSpec, LayoutMode] = LayoutMode.FLAT_2D
    input_names: Optional[List[str]] = None
    required_sources: Optional[List[str]] = None
    min_features: Optional[int] = None
    accepts_3d: bool = False
    accepts_variable_length: bool = False
    preferred_dtype: str = "float32"

    def to_layout_spec(self) -> LayoutSpec:
        """Convert to full LayoutSpec."""
        if isinstance(self.layout, LayoutSpec):
            return self.layout
        elif self.layout == LayoutMode.FLAT_2D:
            return LAYOUT_SKLEARN
        elif self.layout == LayoutMode.VOLUME_3D:
            return LAYOUT_PYTORCH_1D
        elif self.layout == LayoutMode.VOLUME_3D_TRANSPOSE:
            return LAYOUT_TENSORFLOW_1D
        elif self.layout == LayoutMode.MULTI_INPUT:
            return LAYOUT_MULTI_HEAD
        else:
            return LayoutSpec(mode=self.layout)


# Example: Multi-head model with declared inputs
class MultiHeadNIRSModel:
    """Example multi-head model with input specification."""

    def __init__(self, hidden_dim: int = 64):
        self.hidden_dim = hidden_dim
        self._model = None

    def get_input_spec(self) -> ModelInputSpec:
        """Declare multi-input requirements."""
        return ModelInputSpec(
            layout=LayoutMode.MULTI_INPUT,
            input_names=["nir", "markers"],
            required_sources=["nir"],  # markers optional
            accepts_3d=True
        )

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray):
        """Fit with dictionary input."""
        # X = {"nir": array, "markers": array}
        ...

    def predict(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict with dictionary input."""
        ...


# Integration with pipeline execution
class ModelNode:
    def _resolve_input(
        self,
        context: DatasetContext,
        view: ViewSpec
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Resolve input based on model's specification."""
        model = self.operator

        # Check if model declares input spec
        if isinstance(model, ModelWithInputSpec):
            input_spec = model.get_input_spec()
            layout = input_spec.to_layout_spec()
        else:
            # Default to 2D for sklearn-style models
            layout = LAYOUT_SKLEARN

        # Resolve using LayoutResolver
        resolver = LayoutResolver(
            context.block_store,
            context.sample_registry
        )
        return resolver.resolve(view, layout)
```

### Fusion Strategies: Early, Mid, and Late

NIRS4ALL supports three fusion paradigms for combining multi-source or multi-branch data:

```
┌─────────────────────────────────────────────────────────────────┐
│                       Fusion Strategies                          │
│                                                                  │
│  ┌─────────────┐                                                │
│  │ EARLY FUSION│  Combine raw features before modeling          │
│  │             │  merge → model                                 │
│  │   [A|B|C]   │  Simple, but loses source-specific patterns   │
│  └─────────────┘                                                │
│                                                                  │
│  ┌─────────────┐                                                │
│  │ MID FUSION  │  Combine at intermediate representations      │
│  │             │  model_A.layers[:-2] ──┐                       │
│  │   [hA;hB]   │  model_B.layers[:-2] ──┼── concat → head      │
│  │             │  Preserves patterns, complex setup             │
│  └─────────────┘                                                │
│                                                                  │
│  ┌─────────────┐                                                │
│  │ LATE FUSION │  Combine predictions                          │
│  │             │  avg(pred_A, pred_B, pred_C)                   │
│  │   mean(ŷ)   │  Simple ensemble, loses feature interactions  │
│  └─────────────┘                                                │
└─────────────────────────────────────────────────────────────────┘
```

#### Early Fusion

Already supported via `merge` and `merge_sources`:

```python
# Early fusion: concatenate features before model
pipeline = [
    {"source_branch": {
        "NIR": [SNV(), PCA(n_components=50)],
        "markers": [VarianceThreshold()],
    }},
    {"merge_sources": "concat"},  # Early fusion
    {"model": PLSRegression(n_components=10)}
]
```

#### Late Fusion

Already supported via prediction merging:

```python
# Late fusion: average predictions
pipeline = [
    {"branch": [
        [SNV(), {"model": PLSRegression(10)}],
        [MSC(), {"model": RandomForest()}],
    ]},
    {"merge": "predictions"},  # Late fusion via OOF
    {"meta_model": LinearRegression()}
]
```

#### Mid Fusion

Mid fusion combines intermediate representations from neural networks. This requires:
1. Model layer extraction/truncation
2. Feature collection from truncated models
3. Concatenation and final head training

```python
@dataclass
class MidFusionConfig:
    """Configuration for mid-level neural network fusion.

    Allows truncating trained models at specific layers and
    concatenating their intermediate representations.

    Attributes:
        truncate_at: Layer index or name to truncate at (from end)
        freeze_base: Whether to freeze base model weights
        fusion_head: The model to train on concatenated representations
        fusion_strategy: How to combine representations
    """
    truncate_at: Union[int, str] = -2  # Second to last layer
    freeze_base: bool = True
    fusion_head: Optional[Any] = None
    fusion_strategy: Literal["concat", "add", "attention"] = "concat"


class MidFusionNode(DAGNode):
    """Node for mid-level fusion of neural network representations.

    Takes multiple trained models, extracts intermediate representations,
    and trains a fusion head on the concatenated features.
    """
    node_type: NodeType = NodeType.JOIN

    def __init__(self, config: MidFusionConfig):
        self.config = config
        self._base_models: List[Any] = []
        self._truncated_models: List[Any] = []
        self._fusion_head = None

    def execute(
        self,
        contexts: List[DatasetContext],
        model_artifacts: List[Dict[str, Any]],
        view: ViewSpec,
        mode: str = "train"
    ) -> Tuple[DatasetContext, Dict[str, Any]]:
        """Execute mid-fusion.

        Args:
            contexts: Contexts from each branch
            model_artifacts: Trained models from each branch
            view: Data view
            mode: "train" or "predict"

        Returns:
            Tuple of (merged context, fusion artifacts)
        """
        if mode == "train":
            return self._train_fusion(contexts, model_artifacts, view)
        else:
            return self._predict_fusion(contexts, view)

    def _train_fusion(
        self,
        contexts: List[DatasetContext],
        model_artifacts: List[Dict[str, Any]],
        view: ViewSpec
    ) -> Tuple[DatasetContext, Dict[str, Any]]:
        """Train the fusion head on intermediate representations."""
        # Extract and truncate models
        self._base_models = [
            art["virtual_model"].primary_model
            for art in model_artifacts
        ]

        self._truncated_models = [
            self._truncate_model(model, self.config.truncate_at)
            for model in self._base_models
        ]

        # Freeze base models if configured
        if self.config.freeze_base:
            for model in self._truncated_models:
                self._freeze_model(model)

        # Extract intermediate representations
        representations = []
        for ctx, truncated in zip(contexts, self._truncated_models):
            X = ctx.resolver.materialize_X(view)
            h = self._get_representation(truncated, X)
            representations.append(h)

        # Combine representations
        if self.config.fusion_strategy == "concat":
            H = np.concatenate(representations, axis=1)
        elif self.config.fusion_strategy == "add":
            H = np.sum(representations, axis=0)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.config.fusion_strategy}")

        # Train fusion head
        y = contexts[0].target_store.get()
        self._fusion_head = clone(self.config.fusion_head)
        self._fusion_head.fit(H, y)

        return contexts[0], {
            "base_models": self._base_models,
            "truncated_models": self._truncated_models,
            "fusion_head": self._fusion_head
        }

    def _truncate_model(self, model: Any, at: Union[int, str]) -> Any:
        """Truncate a model at the specified layer.

        Supports TensorFlow and PyTorch models.
        """
        # TensorFlow/Keras
        if hasattr(model, 'layers'):
            import tensorflow as tf
            if isinstance(at, int):
                layer = model.layers[at]
            else:
                layer = model.get_layer(at)
            return tf.keras.Model(
                inputs=model.input,
                outputs=layer.output
            )

        # PyTorch
        if hasattr(model, 'children'):
            import torch.nn as nn
            layers = list(model.children())
            if isinstance(at, int):
                layers = layers[:at] if at < 0 else layers[:at+1]
            return nn.Sequential(*layers)

        # sklearn-like: no truncation possible
        raise ValueError(
            f"Model type {type(model)} does not support layer truncation. "
            "Mid-fusion requires neural network models with accessible layers."
        )

    def _get_representation(
        self,
        truncated_model: Any,
        X: np.ndarray
    ) -> np.ndarray:
        """Get intermediate representation from truncated model."""
        # TensorFlow
        if hasattr(truncated_model, 'predict'):
            return truncated_model.predict(X, verbose=0)

        # PyTorch
        if hasattr(truncated_model, 'forward'):
            import torch
            with torch.no_grad():
                X_tensor = torch.from_numpy(X).float()
                return truncated_model(X_tensor).numpy()

        raise ValueError(f"Cannot extract representation from {type(truncated_model)}")

    def _freeze_model(self, model: Any) -> None:
        """Freeze model weights."""
        # TensorFlow
        if hasattr(model, 'trainable'):
            model.trainable = False

        # PyTorch
        if hasattr(model, 'parameters'):
            for param in model.parameters():
                param.requires_grad = False


# Pipeline syntax for mid-fusion
pipeline = [
    {"branch": [
        [{"source": "NIR"}, SNV(), {"model": Conv1DRegressor()}],
        [{"source": "markers"}, {"model": DenseRegressor()}],
    ]},
    {"mid_fusion": MidFusionConfig(
        truncate_at=-2,  # Remove last 2 layers
        freeze_base=True,
        fusion_head=DenseHead(hidden=64),
        fusion_strategy="concat"
    )},
]
```

### Handling 2D Spatial Data (RGB Images)

For 2D image data (e.g., hyperspectral images, RGB photos):

```python
@dataclass
class ImageLayoutSpec(LayoutSpec):
    """Layout specification for 2D image data.

    Extends LayoutSpec with image-specific options.

    Attributes:
        height: Target image height
        width: Target image width
        channels: Number of channels
        resize_method: How to resize images ("crop", "pad", "interpolate")
        normalize: Whether to normalize pixel values
    """
    mode: LayoutMode = LayoutMode.IMAGE_2D
    height: Optional[int] = None
    width: Optional[int] = None
    channels: int = 3
    resize_method: str = "interpolate"
    normalize: bool = True

    def to_shape(self) -> Tuple[int, int, int]:
        """Get target image shape (H, W, C)."""
        if self.height is None or self.width is None:
            raise ValueError("height and width must be specified for images")
        return (self.height, self.width, self.channels)


class ImageLayoutResolver(LayoutResolver):
    """Extended resolver for 2D image data."""

    def _to_image_2d(
        self,
        block_data: Dict[str, np.ndarray],
        layout: ImageLayoutSpec
    ) -> np.ndarray:
        """Convert to image format (N, H, W, C)."""
        images = []

        for source, arr in block_data.items():
            # Reshape 1D spectra to 2D if needed
            if arr.ndim == 2:
                # (N, F) → (N, H, W, 1)
                arr = self._spectra_to_image(arr, layout)
            elif arr.ndim == 3:
                # (N, P, F) → (N, H, W, P)
                n, p, f = arr.shape
                arr = arr.transpose(0, 2, 1)  # (N, F, P)
                arr = self._reshape_to_image(arr, layout)

            images.append(arr)

        # Combine images (stack as channels or batch)
        if layout.source_join == JoinStrategy.STACK:
            return np.concatenate(images, axis=-1)  # Along channel axis
        else:
            return images[0] if len(images) == 1 else images

    def _spectra_to_image(
        self,
        spectra: np.ndarray,
        layout: ImageLayoutSpec
    ) -> np.ndarray:
        """Convert 1D spectra to 2D image representation."""
        n_samples, n_features = spectra.shape
        h, w = layout.height, layout.width

        if h * w != n_features:
            # Resize spectra to match image dimensions
            target_features = h * w
            if layout.resize_method == "interpolate":
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, n_features)
                x_new = np.linspace(0, 1, target_features)
                interpolator = interp1d(x_old, spectra, axis=1, kind='linear')
                spectra = interpolator(x_new)
            elif layout.resize_method == "pad":
                pad_width = target_features - n_features
                spectra = np.pad(spectra, ((0, 0), (0, max(0, pad_width))),
                                 mode='constant')
                spectra = spectra[:, :target_features]

        # Reshape to (N, H, W, 1)
        return spectra.reshape(n_samples, h, w, 1)
```

### Integration with ViewResolver

The existing ViewResolver is extended to support LayoutSpec:

```python
class ViewResolver:
    """Extended ViewResolver with layout support."""

    def materialize(
        self,
        view: ViewSpec,
        layout: Union[str, LayoutSpec] = "2d",
        concat_sources: bool = True
    ) -> Tuple[Union[np.ndarray, Dict], np.ndarray]:
        """Materialize ViewSpec with layout.

        Args:
            view: ViewSpec describing data subset
            layout: Layout specification or shorthand ("2d", "3d", "dict")
            concat_sources: Legacy parameter (deprecated, use LayoutSpec)

        Returns:
            Tuple of (X, y)
        """
        # Convert shorthand to LayoutSpec
        if isinstance(layout, str):
            layout = self._parse_layout_shorthand(layout, concat_sources)

        # Use LayoutResolver for complex layouts
        if isinstance(layout, LayoutSpec) and layout.mode != LayoutMode.FLAT_2D:
            resolver = LayoutResolver(self._blocks, self._registry)
            X = resolver.resolve(view, layout)
        else:
            # Original simple materialization
            X = self._materialize_simple(view, layout)

        # Get y
        indices = self.resolve_indices(view)
        y = self._targets.get(
            version=view.target_version,
            sample_indices=indices
        )

        return X, y

    def _parse_layout_shorthand(
        self,
        shorthand: str,
        concat_sources: bool
    ) -> LayoutSpec:
        """Parse layout shorthand strings."""
        if shorthand == "2d":
            return LayoutSpec(
                mode=LayoutMode.FLAT_2D,
                source_join=JoinStrategy.CONCAT if concat_sources else JoinStrategy.DICT
            )
        elif shorthand == "3d":
            return LayoutSpec(
                mode=LayoutMode.VOLUME_3D,
                source_join=JoinStrategy.STACK,
                processing_join=JoinStrategy.STACK
            )
        elif shorthand == "dict":
            return LAYOUT_MULTI_HEAD
        else:
            raise ValueError(f"Unknown layout shorthand: {shorthand}")
```

### Summary: Layout Strategy Decision Tree

```
                    ┌─────────────────────┐
                    │  What is the model? │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
    ┌────▼────┐          ┌─────▼─────┐         ┌─────▼─────┐
    │ sklearn │          │ TF/PyTorch│         │ Multi-head │
    │ (flat)  │          │ (sequence)│         │ (dict)    │
    └────┬────┘          └─────┬─────┘         └─────┬─────┘
         │                     │                     │
    ┌────▼────┐          ┌─────▼─────┐         ┌─────▼─────┐
    │ 2D flat │          │ 3D volume │         │ Dict per  │
    │ concat  │          │ stack     │         │ source    │
    └─────────┘          └───────────┘         └───────────┘

    Sources: CONCAT        Sources: STACK        Sources: DICT
    Procs:   CONCAT        Procs:   STACK        Procs:   CONCAT
    Branches: DUPLICATE    Branches: STACK       Branches: DICT
```

---

## Backend Abstraction

The data layer supports multiple backends through an abstraction layer.

### Backend Protocol

```python
from typing import Protocol

class ArrayBackend(Protocol):
    """Protocol for array backends."""

    def create_array(
        self,
        data: np.ndarray,
        dims: Optional[Tuple[str, ...]] = None,
        coords: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Create array in backend format."""
        ...

    def slice_array(
        self,
        arr: Any,
        indices: np.ndarray,
        axis: int = 0
    ) -> Any:
        """Slice array along axis."""
        ...

    def concat_arrays(
        self,
        arrays: List[Any],
        axis: int = -1
    ) -> Any:
        """Concatenate arrays."""
        ...

    def to_numpy(self, arr: Any) -> np.ndarray:
        """Convert to numpy array."""
        ...


class NumpyBackend:
    """Default NumPy backend."""

    def create_array(self, data, dims=None, coords=None):
        return np.asarray(data)

    def slice_array(self, arr, indices, axis=0):
        return np.take(arr, indices, axis=axis)

    def concat_arrays(self, arrays, axis=-1):
        return np.concatenate(arrays, axis=axis)

    def to_numpy(self, arr):
        return arr


class XarrayBackend:
    """Optional xarray backend for named dimensions."""

    def create_array(self, data, dims=None, coords=None):
        import xarray as xr
        dims = dims or ("sample", "processing", "feature")
        return xr.DataArray(data, dims=dims, coords=coords)

    def slice_array(self, arr, indices, axis=0):
        dim = arr.dims[axis]
        return arr.isel({dim: indices})

    def concat_arrays(self, arrays, axis=-1):
        import xarray as xr
        dim = arrays[0].dims[axis]
        return xr.concat(arrays, dim=dim)

    def to_numpy(self, arr):
        return arr.values
```

### Backend Selection

```python
def get_backend(name: str = "numpy") -> ArrayBackend:
    """Get array backend by name."""
    if name == "numpy":
        return NumpyBackend()
    elif name == "xarray":
        return XarrayBackend()
    else:
        raise ValueError(f"Unknown backend: {name}")

# Usage in FeatureBlockStore
class FeatureBlockStore:
    def __init__(self, backend: str = "numpy"):
        self._backend = get_backend(backend)

    def register_source(self, data, ...):
        # Use backend for array creation
        arr = self._backend.create_array(data, dims=("sample", "processing", "feature"))
        ...
```

---

## Performance Considerations

### Memory Management

1. **Copy-on-Write**: Blocks share underlying arrays until modified
2. **Lazy Views**: ViewSpec resolution deferred until `materialize()`
3. **Index Caching**: Frequently-used index sets are cached
4. **Memory-Mapped Arrays**: Out-of-core support for large datasets

### Out-of-Core Support (Memory Maps)

For datasets larger than available RAM, FeatureBlocks can use NumPy memory maps:

```python
class FeatureBlockStore:
    """Store with optional memory-mapped backing for large datasets."""

    def register_source_mmap(
        self,
        file_path: Path,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float64,
        source_name: str = "default",
        headers: Optional[List[str]] = None
    ) -> str:
        """Register a memory-mapped source block.

        The data file should be in NumPy's native binary format (.npy)
        or a raw binary file. Data is NOT loaded into RAM.

        Args:
            file_path: Path to the data file
            shape: Expected shape of the array
            dtype: Data type
            source_name: Source identifier
            headers: Optional feature names

        Returns:
            Block ID

        Example:
            # For a 20GB dataset (100k samples × 2k features)
            block_id = store.register_source_mmap(
                Path("large_spectra.npy"),
                shape=(100000, 2000),
                source_name="nir"
            )

            # Only the requested slice is loaded into RAM
            X_train = store.get_data(block_id, sample_indices=train_idx)
        """
        # Create memory-mapped array
        mmap = np.memmap(file_path, dtype=dtype, mode='r', shape=shape)

        # Wrap in lazy-loading FeatureBlock
        block = MemoryMappedFeatureBlock(
            block_id=compute_source_block_id(source_name),
            _mmap=mmap,
            _file_path=file_path,
            headers=headers or [str(i) for i in range(shape[-1])],
            header_unit="index",
            processing_ids=["raw"],
            metadata={"source": source_name, "mmap": True},
            parent_id=None,
            transform_info=None,
            lineage_hash=compute_source_hash(source_name, file_path),
            created_at=datetime.now()
        )

        self._register(block)
        return block.block_id


class MemoryMappedFeatureBlock(FeatureBlock):
    """FeatureBlock backed by memory-mapped file.

    Only loads data chunks when accessed, enabling processing of
    datasets larger than available RAM.
    """

    _mmap: np.memmap
    _file_path: Path

    @property
    def data(self) -> np.ndarray:
        """Access underlying memory map. Slicing loads only needed chunks."""
        return self._mmap

    def get_slice(
        self,
        sample_indices: Optional[np.ndarray] = None,
        feature_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Get data slice, loading only required chunks from disk.

        Args:
            sample_indices: Row indices to load
            feature_indices: Column indices to load

        Returns:
            NumPy array (loaded into RAM)
        """
        if sample_indices is None and feature_indices is None:
            # Materialize full array (may cause OOM for large data)
            return np.array(self._mmap)

        # Efficient sliced access
        data = self._mmap
        if sample_indices is not None:
            data = data[sample_indices]
        if feature_indices is not None:
            data = data[:, feature_indices]

        # Convert from memmap to regular array for downstream processing
        return np.array(data)

    def derive(self, transform_fn, transform_info) -> FeatureBlock:
        """Derive creates a regular (in-memory) block.

        Memory-mapped blocks cannot be modified in-place. Derived blocks
        are regular NumPy arrays, so transformations should be applied
        to chunks or the full dataset must fit in RAM.
        """
        # Load full data (warning: may be large)
        data = np.array(self._mmap)
        new_data = transform_fn(data)

        return FeatureBlock(
            block_id=compute_block_id(self.lineage_hash, transform_info),
            _data=new_data,
            # ... rest of initialization
        )
```

**Chunked Processing for Large Datasets**:

```python
def process_large_dataset_chunked(
    store: FeatureBlockStore,
    block_id: str,
    transform_fn: Callable,
    chunk_size: int = 10000
) -> str:
    """Process large memory-mapped dataset in chunks.

    Args:
        store: Block store
        block_id: ID of memory-mapped block
        transform_fn: Transform to apply per chunk
        chunk_size: Number of samples per chunk

    Returns:
        New block ID with transformed data (memory-mapped output)
    """
    block = store.get(block_id)
    n_samples = block.data.shape[0]

    # Create output memory map
    output_path = store._disk_cache_dir / f"{block_id}_transformed.npy"
    output_shape = (n_samples,) + block.data.shape[1:]
    output_mmap = np.memmap(output_path, dtype=np.float64, mode='w+',
                            shape=output_shape)

    # Process in chunks
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = block.get_slice(sample_indices=np.arange(start, end))
        output_mmap[start:end] = transform_fn(chunk)

    output_mmap.flush()
    return store.register_source_mmap(output_path, output_shape)
```

### Caching Strategy

```python
class FeatureBlockStore:
    def __init__(self, cache_dir: Optional[Path] = None):
        self._memory_cache: Dict[str, FeatureBlock] = {}
        self._disk_cache_dir = cache_dir
        self._cache_policy = LRUPolicy(max_size_gb=4.0)

    def get(self, block_id: str) -> FeatureBlock:
        # Check memory cache
        if block_id in self._memory_cache:
            return self._memory_cache[block_id]

        # Check disk cache
        if self._disk_cache_dir:
            disk_path = self._disk_cache_dir / f"{block_id}.parquet"
            if disk_path.exists():
                block = self._load_from_disk(disk_path)
                self._memory_cache[block_id] = block
                return block

        raise KeyError(f"Block not found: {block_id}")

    def _maybe_evict(self):
        """Evict blocks to stay within memory budget."""
        while self._cache_policy.should_evict(self._memory_cache):
            victim = self._cache_policy.select_victim(self._memory_cache)
            if self._disk_cache_dir:
                self._save_to_disk(victim)
            del self._memory_cache[victim]
```

### Parallel Access

```python
from concurrent.futures import ThreadPoolExecutor

class ParallelViewResolver(ViewResolver):
    """Thread-safe view resolver with parallel materialization.

    Args:
        n_workers: Number of worker threads. Defaults to min(4, cpu_count()).
            Can be overridden via context.n_jobs or environment variable
            NIRS4ALL_WORKERS.
    """

    def __init__(
        self,
        *args,
        n_workers: Optional[int] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Determine worker count from multiple sources
        if n_workers is not None:
            self._n_workers = n_workers
        elif os.environ.get("NIRS4ALL_WORKERS"):
            self._n_workers = int(os.environ["NIRS4ALL_WORKERS"])
        else:
            import multiprocessing
            self._n_workers = min(4, multiprocessing.cpu_count())

        self._executor = ThreadPoolExecutor(max_workers=self._n_workers)

    def materialize_many(
        self,
        views: List[ViewSpec],
        layout: str = "2d"
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Materialize multiple views in parallel."""
        futures = [
            self._executor.submit(self.materialize, view, layout)
            for view in views
        ]
        return [f.result() for f in futures]
```

---

## Thread-Safety and Concurrency

### Design Principles

The Data Layer is designed for thread-safe access in parallel execution scenarios. Key concerns:

1. **Block Store Mutations**: Multiple threads may attempt to register the same transformation
2. **Reference Counting**: `inc_ref`/`dec_ref` must be atomic
3. **Branch Isolation**: CoW must prevent cross-branch data corruption
4. **SampleRegistry Updates**: Fold assignments and exclusions must be synchronized

### Thread-Safe FeatureBlockStore

```python
import threading
from typing import Dict, Optional


class FeatureBlockStore:
    """Thread-safe central registry for feature blocks.

    Uses fine-grained locking to allow maximum parallelism while
    ensuring correctness:
    - Block registration is atomic (no duplicate blocks)
    - Reference counting is atomic
    - Read operations are lock-free after registration
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self._blocks: Dict[str, FeatureBlock] = {}
        self._lineage_index: Dict[str, str] = {}

        # Locks for thread-safety
        self._registration_lock = threading.Lock()
        self._ref_count_lock = threading.Lock()

        self._cache_dir = cache_dir

    def register_transform(
        self,
        parent_id: str,
        data: np.ndarray,
        transform_info: TransformInfo,
        processing_id: Optional[str] = None
    ) -> str:
        """Register a transformed block (thread-safe).

        If a block with the same lineage already exists, returns
        the existing block_id instead of creating a duplicate.
        This provides automatic deduplication in parallel execution.
        """
        # Compute lineage hash outside the lock
        parent = self._blocks[parent_id]
        lineage_hash = compute_lineage_hash(parent.lineage_hash, transform_info)

        # Check if already exists (fast path, no lock)
        if lineage_hash in self._lineage_index:
            return self._lineage_index[lineage_hash]

        # Acquire lock for registration
        with self._registration_lock:
            # Double-check after acquiring lock (another thread may have registered)
            if lineage_hash in self._lineage_index:
                return self._lineage_index[lineage_hash]

            # Create and register new block
            block_id = f"block_{len(self._blocks):06d}"
            block = FeatureBlock(
                block_id=block_id,
                _data=data,
                headers=parent.headers,
                header_unit=parent.header_unit,
                processing_ids=parent.processing_ids + [processing_id or transform_info.target_processing],
                metadata=parent.metadata,
                parent_id=parent_id,
                transform_info=transform_info,
                lineage_hash=lineage_hash,
                created_at=datetime.now()
            )

            self._blocks[block_id] = block
            self._lineage_index[lineage_hash] = block_id

            # Mark parent as CoW (shared)
            parent._is_cow = True

            return block_id

    def inc_ref(self, block_id: str) -> None:
        """Increment reference count (thread-safe)."""
        with self._ref_count_lock:
            if block_id in self._blocks:
                self._blocks[block_id]._ref_count += 1

    def dec_ref(self, block_id: str) -> None:
        """Decrement reference count (thread-safe)."""
        with self._ref_count_lock:
            if block_id in self._blocks:
                self._blocks[block_id]._ref_count -= 1

    @contextmanager
    def lease(self, block_ids: List[str]) -> Iterator[List[FeatureBlock]]:
        """Context manager for safe block access with automatic cleanup.

        The Lease pattern ensures that blocks are not garbage collected
        while in use, and that references are properly decremented even
        if an exception occurs.

        This solves the risk of:
        - Memory leaks if dec_ref is never called (thread crash)
        - Premature block deletion during parallel access

        Args:
            block_ids: IDs of blocks to lease

        Yields:
            List of FeatureBlock objects

        Example:
            with store.lease([block_id1, block_id2]) as blocks:
                X1, X2 = blocks[0].data, blocks[1].data
                # Process data...
            # Blocks automatically released, even on exception
        """
        # Acquire references
        for block_id in block_ids:
            self.inc_ref(block_id)

        try:
            blocks = [self._blocks[bid] for bid in block_ids]
            yield blocks
        finally:
            # Always release, even on exception
            for block_id in block_ids:
                self.dec_ref(block_id)
            # Trigger GC to clean up if needed
            self.gc()

    def gc(self, keep_roots: bool = True) -> int:
        """Garbage collect unreachable blocks (thread-safe)."""
        with self._registration_lock:
            with self._ref_count_lock:
                # Implementation as before, but now atomic
                ...
```

### Branch Isolation with CoW

When branches are created (ForkNode), multiple branch contexts share references to the same blocks. The CoW mechanism ensures isolation:

```python
class FeatureBlock:
    """Copy-on-Write feature block with branch isolation."""

    def derive(
        self,
        transform_fn: Callable[[np.ndarray], np.ndarray],
        transform_info: TransformInfo
    ) -> "FeatureBlock":
        """Create derived block with CoW semantics.

        Branch isolation is ensured by checking BOTH:
        1. _is_cow flag (set when block is shared)
        2. _ref_count (number of contexts referencing this block)

        If either indicates sharing, a copy is made before transformation.
        """
        needs_copy = self._is_cow or self._ref_count > 1

        if needs_copy:
            # Make a copy to preserve original for other branches
            new_data = transform_fn(self._data.copy())
        else:
            # Safe to transform in-place (no other references)
            new_data = transform_fn(self._data)

        # Create new block with updated lineage
        return FeatureBlock(
            block_id=compute_block_id(self.lineage_hash, transform_info),
            _data=new_data,
            headers=self.headers,
            header_unit=self.header_unit,
            processing_ids=self.processing_ids + [transform_info.target_processing],
            metadata=self.metadata,
            parent_id=self.block_id,
            transform_info=transform_info,
            lineage_hash=compute_lineage_hash(self.lineage_hash, transform_info),
            created_at=datetime.now(),
            _is_cow=False,  # New block is not shared yet
            _ref_count=1
        )
```

### Branch Context Reference Management

When a ForkNode creates branches, reference counts are properly managed:

```python
class ForkNode:
    def execute(self, context: DatasetContext, ...):
        """Create branch contexts with proper reference management."""
        branch_contexts = []

        for branch_id in range(self.branch_count):
            # Create shallow copy of context
            branch_context = copy(context)

            # Increment ref counts for all active blocks
            for block_id in context.active_block_ids:
                context.block_store.inc_ref(block_id)

            branch_contexts.append(branch_context)

        return branch_contexts


class JoinNode:
    def execute(self, contexts: List[DatasetContext], ...):
        """Merge branches with proper cleanup."""
        # ... merge logic ...

        # Decrement ref counts for blocks no longer needed
        for ctx in contexts:
            for block_id in ctx.active_block_ids:
                if block_id not in merged_block_ids:
                    ctx.block_store.dec_ref(block_id)

        # Trigger GC to free unreferenced blocks
        contexts[0].block_store.gc()

        return merged_context
```

### SampleRegistry Thread-Safety

The SampleRegistry uses Polars which has its own thread-safety guarantees. However, mutation operations are synchronized:

```python
class SampleRegistry:
    """Thread-safe sample registry."""

    def __init__(self):
        self._df: pl.DataFrame = pl.DataFrame(schema=SAMPLE_REGISTRY_SCHEMA)
        self._mutation_lock = threading.Lock()

    def mark_excluded(
        self,
        sample_ids: List[int],
        reason: Optional[str] = None
    ) -> None:
        """Mark samples as excluded (thread-safe)."""
        with self._mutation_lock:
            self._df = self._df.with_columns([
                pl.when(pl.col("_sample_id").is_in(sample_ids))
                .then(True)
                .otherwise(pl.col("excluded"))
                .alias("excluded")
            ])

    def assign_folds(self, splitter, X, y, groups=None) -> None:
        """Assign CV folds (thread-safe)."""
        with self._mutation_lock:
            # ... fold assignment logic ...
```

### Concurrency Guidelines

| Operation | Thread-Safe | Notes |
|-----------|-------------|-------|
| Block read (`get`) | ✅ Yes | Lock-free after registration |
| Block registration | ✅ Yes | Double-checked locking |
| Ref count update | ✅ Yes | Atomic with lock |
| View materialization | ✅ Yes | Read-only operations |
| Sample exclusion | ✅ Yes | Mutation lock |
| Fold assignment | ✅ Yes | Mutation lock |
| Garbage collection | ✅ Yes | Full lock during GC |

---

## Next Document

**Document 3: DAG Execution Engine** covers:
- Node types and interfaces
- DAG compilation from pipeline syntax
- Execution scheduling and parallelism
- Branching and merging semantics
- Artifact and prediction management
