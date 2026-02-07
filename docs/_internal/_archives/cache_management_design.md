# Cache Management Design — nirs4all

> **Status**: Design proposal (v2 — revised after review)
> **Date**: 2025-02-07
> **Based on**: Codebase audit of all cache-related paths (see `spectrodataset_cache_memory_overflow_investigation.md`)
> **Revision**: Incorporates review feedback on keying correctness, size estimation, cross-run hydration gap, peak RAM tracking, deprecation strategy, and PR ordering.

---

## Table of Contents

1. [Current State Summary](#1-current-state-summary)
2. [Problems to Solve](#2-problems-to-solve)
3. [Design Principles](#3-design-principles)
4. [Architecture Overview](#4-architecture-overview)
5. [PR A — Safety, Hash Fix, and Observability](#5-pr-a--safety-hash-fix-and-observability)
6. [PR B — Step-Level Caching (OFF by default)](#6-pr-b--step-level-caching-off-by-default)
7. [PR C — Lightweight Branch Snapshots](#7-pr-c--lightweight-branch-snapshots)
8. [PR D — Memory-Bounded Feature Storage](#8-pr-d--memory-bounded-feature-storage)
9. [PR E — Cross-Run Cache Persistence](#9-pr-e--cross-run-cache-persistence)
10. [Configuration](#10-configuration)
11. [Observability](#11-observability)
12. [File-by-File Change Map](#12-file-by-file-change-map)
13. [Acceptance Criteria and Quality Governance](#13-acceptance-criteria-and-quality-governance)
14. [What NOT to Do](#14-what-not-to-do)

---

## 1. Current State Summary

### Active caching (actually used)

| Component | Location | What it caches | Scope |
|-----------|----------|----------------|-------|
| `_content_hash_cache` | `data/dataset.py:73` | xxhash128 of feature arrays | Per-dataset instance, invalidated on mutation |
| `DatasetConfigs.cache` | `data/config.py:178` | Raw loaded data tuples | Per-`DatasetConfigs` instance, **unbounded** |
| Transformer artifact cache | `controllers/transforms/transformer.py:789` | Fitted transformer objects | Per-dataset run via `ArtifactRegistry._by_chain_and_data` |

### Dead/unused infrastructure

| Component | Location | Problem |
|-----------|----------|---------|
| `StepCache` | `pipeline/execution/step_cache.py` | Instantiated in orchestrator (L216) but never passed to executor, step_runner, or controllers. Zero cache hits ever. Stats always zero. |
| `DataCache` | `data/performance/cache.py` | Only used by StepCache (which is unused). `cache_manager()` singleton never called. |
| `Features.cache` param | `data/features.py:21` | Stored in `__init__`, never read by any code. |
| `LazyArray` / `LazyDataset` | `data/performance/lazy_loader.py` | Fully implemented, exported from `performance.__init__`, never instantiated anywhere. Not re-exported by `data.__init__`. |
| `persist_cache_keys_to_store()` | `pipeline/storage/artifacts/artifact_registry.py:1225` | Implemented, tested, never called in orchestrator. |
| `load_cached_from_store()` | `pipeline/storage/artifacts/artifact_registry.py:1255` | Returns only an `artifact_id` string — does not hydrate a runtime `ArtifactRecord`. Gap with what `_try_cache_lookup()` expects. |

### Bugs / inconsistencies

1. **Dual hash attribute**: `dataset.py:1945` checks `hasattr(self, '_content_hash')` but the cache attribute is `_content_hash_cache`. `set_content_hash()` (L2032) sets `self._content_hash` — a separate, disconnected attribute.

2. **Transformer cache key ignores fit selector**: The artifact cache key (`transformer.py:837`) uses `dataset.content_hash()` (whole dataset), but the actual fit data depends on `fit_on_all`, partition, fold, and `include_excluded` flags (`transformer.py:163-175`). Two runs with the same data but different fit selectors can collide on the same cache key. Current risk is low (selector usually doesn't vary within a single run), but incorrect in principle.

3. **DataCache eviction is LFU, not LRU**: `DataCache.get()` (`cache.py:140`) increments `hit_count` but never updates `timestamp`. Eviction sorts by `(hit_count, timestamp)` — this is LFU with creation-time tiebreaker, not LRU.

4. **StepCache size estimation uses pickle**: When storing a `SpectroDataset` (not an ndarray/list/tuple/dict), `DataCache._estimate_size()` falls back to `len(pickle.dumps(data))` (`cache.py:303`). This is both slow and inaccurate for large datasets.

---

## 2. Problems to Solve

### P1 — Unbounded feature array growth

`ArrayStorage.add_processing()` concatenates along axis=1 each time a preprocessing variant is added. For a dataset with S samples, P processings, F features:
- Memory per source = `S × P × F × 4 bytes`
- P grows with every transform step in `add` mode
- Feature augmentation with M operations produces `P_base + P_base × M` processings

**Empirical**: 5000×1000 float32 goes from 19 MB (1 processing) to 248 MB (13 processings).

### P2 — Copy amplification during concatenation

Each `add_processing()` allocates a full new array via `np.concatenate`. The old array becomes garbage but coexists in RAM until GC collects it. Peak usage is roughly `2× steady-state` during every concatenation.

### P3 — Branch snapshot deep-copy explosion

`BranchController._snapshot_features()` does `copy.deepcopy(dataset._features.sources)`. For N branches:
- 1 initial snapshot + 2N copies (restore + result snapshot per branch)
- Executor does another `copy.deepcopy` per branch per post-branch step
- **Empirical**: 3000×8×700 float32 → 64 MB base, 6 branches → 385 MB additional

### P4 — No reuse of shared preprocessing across pipeline variants

When generators expand `[{_or_: [SNV, MSC]}, PLS({_range_: [1, 20]})]` into 40 variants, the first step (SNV or MSC) is re-executed from scratch for each variant because:
- Each variant gets a freshly loaded dataset
- `StepCache` exists but is not wired
- Only the transformer **fit artifact** is reused (via ArtifactRegistry), not the transformed data

### P5 — `DatasetConfigs.cache` is unbounded

The raw data cache in `DatasetConfigs` holds all loaded datasets for its entire lifetime with no size limit. Two large dataset entries can blow RAM on their own. Keying by name only (`config.py:368`) risks collisions if config parameters change.

### P6 — Dead code clutters the caching landscape

Six fully-implemented but unused components create false confidence and confusion about what is actually cached.

### P7 — Peak RAM not tracked

The dominant memory peaks are transient: `all_data` + `fit_data` + transform buffers in `transformer.py` (L162, L168, L328), repeated `np.concatenate` in `array_storage.py` (L220), and `copy.deepcopy` in branches (`branch.py:1512`, `executor.py:509`). Steady-state `nbytes` checks miss these spikes entirely.

---

## 3. Design Principles

1. **Observability first, architecture second.** Instrument memory and cache behavior before making structural changes. Metrics drive which phase to prioritize.

2. **Incremental, not revolutionary.** Fix the wiring and boundaries of existing mechanisms before introducing new abstractions. The "from-scratch" view-DAG proposal is architecturally elegant but too large a lift. Stay within the current mutable-dataset model.

3. **Budget-first.** Every cache must have a **byte** budget, not just an entry-count limit.

4. **Key correctness.** Separate `fit_key` (includes fit selector fingerprint) from `transform_key` (includes apply selector). Never allow two semantically different operations to share a cache key.

5. **Deep-copy is the enemy.** Reduce, defer, or replace `copy.deepcopy` on feature arrays wherever possible.

6. **Deprecate before removing.** Publicly exported symbols (`LazyDataset`, `LazyArray`, `cache_manager`, `Features(cache=...)`) get a deprecation warning period before deletion.

7. **New features OFF by default.** `StepCache` wiring ships disabled until proven stable via stress tests. Feature flags gate each phase.

---

## 4. Architecture Overview

The design is organized into five PRs that can be implemented, reviewed, and shipped independently. Ordering is strict — each PR builds on the previous:

```
PR A: Safety + hash fix + observability    (foundation — no behavioral change)
PR B: StepCache wiring (OFF by default)    (biggest perf win for generators)
PR C: Branch snapshot fix                  (fix branch memory scaling)
PR D: Feature storage (only if A-C metrics warrant it)
PR E: Cross-run cache persistence          (acceleration for iterative workflows)
```

| PR | Addresses | Risk |
|----|-----------|------|
| A | P5, P6, P7, hash bug, observability foundation | Very low — no functional change |
| B | P4, cache key correctness | Medium — new execution path, gated by flag |
| C | P3 | Medium — changes snapshot mechanics |
| D | P1, P2 | Higher — changes core storage layout |
| E | Future acceleration | Low — wires existing tested code |

**Go/no-go gates**: Each PR ships with stress tests. If metrics from PR A show that P1/P2 are not the bottleneck in practice, PR D can be deferred or dropped.

---

## 5. PR A — Safety, Hash Fix, and Observability

### 5.1 Deprecate dead code (not delete yet)

Publicly exported symbols get a deprecation warning for one minor version before removal:

| Symbol | Location | Action |
|--------|----------|--------|
| `LazyArray`, `LazyDataset` | `data/performance/__init__.py` | Add `DeprecationWarning` on import. Mark for removal in next minor. |
| `cache_manager()` | `data/performance/cache.py` | Add `DeprecationWarning` on call. |
| `Features(cache=...)` | `data/features.py` | Accept param silently (already a no-op), add `DeprecationWarning` if `cache=True` is passed. |

Remove StepCache premature stats logging in orchestrator (stats are always zero).

### 5.2 Fix the dual-hash-attribute bug

In `data/dataset.py`:
- `set_content_hash()` → set `_content_hash_cache` (the canonical attribute), not `_content_hash`
- `get_dataset_metadata()` → check `_content_hash_cache` instead of `_content_hash`
- Remove the stale `_content_hash` attribute entirely

**Priority: high.** This is a correctness fix.

### 5.3 Bound DatasetConfigs.cache by bytes

In `data/config.py`:
- Add `max_cache_bytes: int = 2 * 1024**3` (2 GB default)
- Track cumulative size of cached entries (estimated via numpy `nbytes` sum)
- When inserting would exceed budget, evict the oldest entry
- Key derivation: use `hash(name + str(sorted(config_params)))` instead of bare `name` to prevent collisions when the same dataset name is loaded with different parameters

### 5.4 Memory estimation and RSS tracking

Create `nirs4all/utils/memory.py`:

```python
import os

def estimate_dataset_bytes(dataset: SpectroDataset) -> int:
    """Estimate total feature array memory for all sources (steady-state)."""
    total = 0
    for source in dataset._features.sources:
        storage = source._storage
        total += storage._data.nbytes
    return total

def estimate_cache_entry_bytes(entry) -> int:
    """Fast size estimation using numpy nbytes. No pickle."""
    if isinstance(entry, np.ndarray):
        return entry.nbytes
    if hasattr(entry, '_features'):  # SpectroDataset-like
        return estimate_dataset_bytes(entry)
    # Fallback: sys.getsizeof (shallow but fast)
    return sys.getsizeof(entry)

def get_process_rss_mb() -> float:
    """Current process RSS in MB. Linux/macOS only."""
    try:
        # /proc/self/statm: pages, resident, shared, text, lib, data, dirty
        with open("/proc/self/statm") as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1e6
    except (FileNotFoundError, OSError):
        return 0.0

def format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
```

### 5.5 Per-step memory and RSS logging

In `PipelineExecutor._execute_single_step()`, after step execution:

```
[Step 3/8] SNV | shape: 5000×4×2000 | steady: 152.6 MB | RSS: 1.2 GB
```

At `verbose >= 2`.

Emit a warning when RSS exceeds `memory_warning_threshold_mb`:

```
⚠ Process RSS at 3.8 GB (threshold: 3.0 GB) after step SNV.
```

### 5.6 Stress test suite

Add `tests/integration/test_memory_stress.py`:
- Large dataset + 100 generated preprocessing variants → measure peak RSS
- Branch-heavy pipeline (6+ branches) → measure peak RSS
- Feature augmentation with 20 operations → measure peak RSS
- These tests establish baselines and serve as go/no-go gates for subsequent PRs

---

## 6. PR B — Step-Level Caching (OFF by default)

### 6.1 What to cache

Cache the **dataset feature state after a preprocessing step** as a lightweight snapshot. This lets variant `[SNV, PLS(5)]` reuse the post-SNV result from variant `[SNV, PLS(3)]`.

What NOT to cache at step level:
- Model steps (variant-specific)
- Splitter steps (cheap, stateful with fold indices)
- Branch entry/exit (complex state, handled in PR C)

### 6.2 Cache key — with fit selector fingerprint

The current `StepCache` key `(chain_path_hash, data_hash)` is insufficient. The transformer's fit behavior depends on the selector (partition, fold, `fit_on_all`, `include_excluded`). Two calls with the same chain + data but different fit selectors must not collide.

**Revised key**:

```
key = f"step:{chain_path_hash}:{data_hash}:{selector_fingerprint}"
```

Where `selector_fingerprint` is computed from:

```python
def compute_selector_fingerprint(context: ExecutionContext) -> str:
    """Fingerprint the aspects of the selector that affect fit/transform behavior."""
    s = context.selector
    components = [
        s.partition or "all",
        str(s.fold_id),
        str(s.processing),
        str(s.include_augmented),
        str(sorted(s.tag_filters.items())) if s.tag_filters else "",
        str(s.branch_path),
    ]
    return hashlib.md5("|".join(components).encode()).hexdigest()[:12]
```

This ensures:
- Same operator sequence + same input data + same selector → cache hit
- Different fold → cache miss
- Different partition → cache miss

### 6.3 Which steps are cacheable

Add a class method to `OperatorController`:

```python
@classmethod
def supports_step_cache(cls) -> bool:
    """Whether this step's output should be cached for cross-variant reuse."""
    return False
```

Enable for `TransformerMixinController` → `True`.

Leave disabled for `ModelController`, `SplitterController`, `BranchController`, `MergeController`, `FeatureAugmentationController`.

### 6.4 CachedStepState — lightweight, no pickle

Replace `StepCache`'s current `copy.deepcopy(dataset)` with a dedicated lightweight state object:

```python
@dataclass
class CachedStepState:
    features_sources: List[FeatureSource]  # deep-copied on store
    processing_names: List[List[str]]       # processing chain per source
    content_hash: str                       # post-step content hash
    bytes_estimate: int                     # computed via numpy nbytes, NOT pickle
```

**Size estimation**: Use `estimate_cache_entry_bytes()` (numpy `nbytes` sum over all source arrays). Never use `pickle.dumps()` for size estimation — it's too slow and inaccurate for large arrays.

**On store**: Deep-copy only `features_sources` and `processing_names`. Skip metadata, y-values, sample indices, headers (unchanged by preprocessing).

**On restore**: Deep-copy the cached `features_sources` back into the dataset, update context processing names.

### 6.5 Wiring into execution

**Step 1: Pass StepCache through RuntimeContext**

Add `step_cache: Optional[Any] = None` to `RuntimeContext`.

In `orchestrator.py`, pass `step_cache` to `RuntimeContext` only when `cache_config.step_cache_enabled` is `True`.

**Step 2: Cache lookup/store in executor**

In `PipelineExecutor._execute_single_step()`:

```python
def _execute_single_step(self, step, dataset, context, runtime_context, ...):
    controller = self._router.route(parsed_step, step)
    cache = runtime_context.step_cache
    cacheable = (
        cache is not None
        and controller.supports_step_cache()
        and not runtime_context.in_branch_mode  # don't cache mid-branch
    )

    if cacheable:
        chain = runtime_context.trace_recorder.current_chain()
        chain_hash = compute_chain_hash(chain.to_path())
        data_hash = dataset.content_hash()
        selector_fp = compute_selector_fingerprint(context)
        cached = cache.get(chain_hash, data_hash, selector_fp)
        if cached is not None:
            # Cache hit — restore feature state
            _restore_from_cached_state(dataset, context, cached)
            return cached_step_result

    result = step_runner.execute(step, dataset, context, runtime_context, ...)

    if cacheable and result is not None:
        state = _build_cached_state(dataset, context)
        cache.put(chain_hash, data_hash, selector_fp, state)

    return result
```

### 6.6 StepCache budget and eviction

- Default `max_size_mb = 2048` (2 GB), adjustable via `CacheConfig`
- Fix `DataCache.get()` to also update `timestamp` on access (currently only `hit_count` is updated, making eviction LFU not LRU). With both fields refreshed, eviction becomes a proper LRU+LFU hybrid.
- Size estimation: override `DataCache._estimate_size()` to use `CachedStepState.bytes_estimate` directly instead of falling through to the pickle path.

### 6.7 Feature flag: OFF by default

```python
@dataclass
class CacheConfig:
    step_cache_enabled: bool = False  # OFF by default until stress-tested
```

Activation requires explicit opt-in: `nirs4all.run(..., cache=CacheConfig(step_cache_enabled=True))`.

After stress tests confirm no regressions, flip default to `True`.

### 6.8 Correctness test: cache on vs off

Add `tests/integration/test_step_cache_correctness.py`:
- Run the same generator pipeline with `step_cache_enabled=True` and `step_cache_enabled=False`
- Assert identical results (best score, predictions) within floating-point tolerance
- This is a **mandatory** gate for enabling the cache by default

### 6.9 Impact estimate

For a pipeline with 40 variants from `[{_or_: [SNV, MSC]}, PLS({_range_: [1, 20]})]`:
- Without cache: 40 executions of preprocessing + 40 model fits
- With cache: 2 preprocessing executions (SNV once, MSC once) + 40 model fits
- **~50% wall-time reduction** for preprocessing-dominated pipelines

---

## 7. PR C — Lightweight Branch Snapshots

### 7.1 Problem recap

`BranchController._snapshot_features()` does `copy.deepcopy(dataset._features.sources)`. For N branches, total copies = 1 + 2N of the full feature storage. The executor adds more copies per post-branch step.

### 7.2 Copy-on-write feature sources

Introduce a lightweight reference-counting wrapper:

```python
class SharedBlocks:
    """Reference-counted immutable block list. Copy-on-write semantics."""

    def __init__(self, blocks: List[np.ndarray], block_ids: List[str]):
        self._blocks = blocks
        self._block_ids = block_ids
        self._refcount = 1

    def acquire(self) -> 'SharedBlocks':
        """Increment refcount. Returns self."""
        self._refcount += 1
        return self

    def release(self):
        """Decrement refcount."""
        self._refcount -= 1

    def is_shared(self) -> bool:
        return self._refcount > 1

    def detach(self) -> 'SharedBlocks':
        """Create an independent copy if shared, otherwise return self."""
        if self._refcount > 1:
            self._refcount -= 1
            return SharedBlocks(
                [b.copy() for b in self._blocks],
                list(self._block_ids)
            )
        return self
```

**Note**: `SharedBlocks` wraps the existing `ArrayStorage._data` (or `_blocks` after PR D). It's introduced at the `ArrayStorage` level so that branch snapshots can share underlying arrays without deep-copying.

**Usage in ArrayStorage — mutation triggers detach:**

```python
class ArrayStorage:
    _shared: SharedBlocks

    def add_processing(self, data_2d, processing_id):
        self._shared = self._shared.detach()  # CoW: copy only if shared
        self._shared._blocks.append(data_2d)
        ...
```

**Usage in BranchController:**

```python
def _snapshot_features(self, dataset):
    """Lightweight snapshot: acquire shared references instead of deep-copying."""
    return [
        source._storage._shared.acquire()
        for source in dataset._features.sources
    ]

def _restore_features(self, dataset, snapshot):
    """Restore from snapshot. No deep-copy unless mutation follows."""
    for source, shared in zip(dataset._features.sources, snapshot):
        source._storage._shared = shared.acquire()
        source._storage._cached_3d = None
```

### 7.3 Branch context cleanup

After a branch's post-branch steps complete:
- Release all `SharedBlocks` references in that branch's snapshot
- Clear the snapshot from `branch_contexts`
- This ensures memory from completed branches is freed promptly

### 7.4 Executor post-branch optimization

Instead of `copy.deepcopy(features_snapshot)` on every post-branch step, use `shared.acquire()` and let copy-on-write handle mutations. Most post-branch steps (model training) don't modify features, so no copy occurs.

### 7.5 Interaction with PR D

If PR D (block-based storage) is implemented, `SharedBlocks` wraps the block list directly. If PR D is deferred, `SharedBlocks` can wrap the monolithic 3D array with the same CoW semantics (detach copies the 3D array on mutation). The CoW benefit holds either way.

---

## 8. PR D — Memory-Bounded Feature Storage

> **Gate**: Only implement if metrics from PR A stress tests show that P1/P2 (array growth / concatenation) are a significant bottleneck. If PR B + PR C resolve the dominant memory issues, PR D can be deferred.

### 8.1 Block-based storage

Replace the monolithic 3D array with a list of 2D blocks:

```python
class ArrayStorage:
    _blocks: List[np.ndarray]           # Each block is (samples, features), one per processing
    _block_ids: List[str]               # Processing ID per block
    _cached_3d: Optional[np.ndarray]    # Lazily computed concatenated view, invalidated on mutation
```

**Benefits**:
- `add_processing()` appends a 2D block — no full-array copy
- `remove_processing(id)` drops a block — O(1) memory release
- 3D view is computed lazily only when needed and cached until next mutation
- Individual processings can be evicted independently

**Migration**:
- `add_processing(data_2d)` → `self._blocks.append(data_2d); self._cached_3d = None`
- `x(indices, layout="3d")` → if `_cached_3d is None: _cached_3d = np.stack(self._blocks, axis=1); return _cached_3d[indices]`
- `x(indices, layout="2d")` → `np.concatenate([b[indices] for b in self._blocks], axis=1)` (no full 3D needed)

### 8.2 Processing eviction policy

Add to `FeatureSource`:

```python
max_processings: int = 50              # configurable hard limit
eviction_policy: str = "oldest_first"  # or "error"
```

When `len(self._blocks) >= max_processings`:
- `"oldest_first"`: drop the oldest non-"raw" processing block, log warning
- `"error"`: raise `MemoryError` with a clear message

### 8.3 Batch add for feature augmentation

For `FeatureAugmentationController` which calls `add_processing()` M×P times:

```python
def add_processings_batch(self, blocks: List[np.ndarray], ids: List[str]):
    """Add multiple processings at once without intermediate copies."""
    self._blocks.extend(blocks)
    self._block_ids.extend(ids)
    self._cached_3d = None
```

### 8.4 Global memory watchdog

In `PipelineExecutor`, before each step:

```python
current_mb = estimate_dataset_bytes(dataset) / 1e6
if current_mb > config.max_feature_ram_mb:
    if config.overflow_policy == "error":
        raise MemoryError(...)
    elif config.overflow_policy == "warn":
        logger.warning(...)
    elif config.overflow_policy == "evict":
        for source in dataset._features.sources:
            source.evict_cold_processings(target_mb=config.max_feature_ram_mb)
```

Default `max_feature_ram_mb = 4096` (4 GB).

### 8.5 Backward compatibility

Public API of `FeatureSource` and `ArrayStorage` does not change for callers:
- `x(indices, layout)` returns the same shapes
- `add_processing()`, `replace_features()`, `reset_features()` have the same signatures
- Internal change from `_data: np.ndarray` to `_blocks: List[np.ndarray]` is private

### 8.6 Benchmark gate

**Mandatory before merge**: Run the stress test suite from PR A (same scenarios) and demonstrate:
- Peak RSS reduction ≥ 30% on the feature-augmentation-heavy scenario
- No performance regression > 5% on standard pipelines
- Exact numerical equivalence of results

---

## 9. PR E — Cross-Run Cache Persistence

### 9.1 The hydration gap

`load_cached_from_store()` (`artifact_registry.py:1255`) queries DuckDB and returns an `artifact_id` string. But `_try_cache_lookup()` (`transformer.py:840`) expects a full `ArtifactRecord` with loadable binary data — it calls `registry.get_by_chain_and_data()` which returns an `ArtifactRecord`, then `registry.load_artifact(record)` to deserialize the binary.

Simply calling `load_cached_from_store()` at startup is **not enough**. The runtime needs a hydrated `ArtifactRecord` that maps back to a physical binary file on disk.

### 9.2 Required hydration bridge

Add an `ArtifactRegistry` method:

```python
def hydrate_cached_artifacts(self, store) -> int:
    """Load cross-run cache keys from store, validate binary files exist,
    and populate in-memory indexes with full ArtifactRecords.

    Returns: number of successfully hydrated entries.
    """
    cached_entries = store.find_all_cached_artifacts(self._dataset_name)
    hydrated = 0
    for entry in cached_entries:
        artifact_id = entry["artifact_id"]
        chain_path_hash = entry["chain_path_hash"]
        input_data_hash = entry["input_data_hash"]

        # 1. Check if binary file still exists on disk
        binary_path = self._resolve_binary_path(artifact_id)
        if not binary_path or not binary_path.exists():
            continue  # Graceful miss — file was cleaned up

        # 2. Build a full ArtifactRecord from store metadata + binary path
        record = self._build_record_from_store(entry, binary_path)
        if record is None:
            continue

        # 3. Register in runtime indexes
        self._artifacts[artifact_id] = record
        self._by_chain_and_data[(chain_path_hash, input_data_hash)] = artifact_id
        hydrated += 1

    return hydrated
```

### 9.3 Wiring

In `PipelineOrchestrator.execute()`:

**Before dataset loop** (after creating `ArtifactRegistry`):
```python
if store is not None and cache_config.persist_cache_keys:
    n = artifact_registry.hydrate_cached_artifacts(store)
    if verbose >= 1 and n > 0:
        logger.info(f"Loaded {n} cached transformer artifacts from previous runs")
```

**After dataset loop** (after `artifact_registry.end_run()`):
```python
if store is not None and cache_config.persist_cache_keys:
    artifact_registry.persist_cache_keys_to_store(store)
```

### 9.4 Safety

- Verify binary file exists on disk before accepting a cache hit
- Verify `record.class_name` matches the expected operator class
- On any mismatch or missing file: silently return cache miss, log at `verbose >= 2`
- Strict compatibility: cache schema version embedded in keys (already the case via `operator_fingerprint` in chain path)

### 9.5 Scope

Cross-run persistence applies only to **fitted transformer artifacts**. It does NOT persist transformed feature arrays — those are too large and variant-specific. The step-level cache (PR B) remains in-memory only.

---

## 10. Configuration

### 10.1 CacheConfig dataclass

Create `nirs4all/config/cache_config.py`:

```python
@dataclass
class CacheConfig:
    """Cache and memory management configuration."""

    # Step cache (PR B) — OFF by default until stress-tested
    step_cache_enabled: bool = False
    step_cache_max_mb: int = 2048
    step_cache_max_entries: int = 200

    # Feature storage (PR D)
    max_processings_per_source: int = 50
    max_feature_ram_mb: int = 4096
    overflow_policy: str = "warn"  # "warn", "error", "evict"

    # Branch snapshots (PR C)
    use_cow_snapshots: bool = True

    # Cross-run persistence (PR E)
    persist_cache_keys: bool = True

    # Observability
    log_cache_stats: bool = True
    log_step_memory: bool = True
    memory_warning_threshold_mb: int = 3072
```

### 10.2 Integration with nirs4all.run()

```python
nirs4all.run(
    pipeline=[...],
    dataset="...",
    cache=CacheConfig(step_cache_enabled=True),  # explicit opt-in
)
```

Default `CacheConfig()` is used when not specified. `step_cache_enabled` defaults to `False`.

### 10.3 Flow through execution

`PipelineRunner` → `PipelineOrchestrator` → `RuntimeContext` carries `cache_config`.

Each component reads only the fields it needs:
- `StepCache`: `step_cache_max_mb`, `step_cache_max_entries`
- `ArrayStorage` / `FeatureSource`: `max_processings_per_source`
- `PipelineExecutor`: `max_feature_ram_mb`, `overflow_policy`, `memory_warning_threshold_mb`
- `Orchestrator`: `persist_cache_keys`, `step_cache_enabled`

### 10.4 Rollback flags

Each feature is independently toggleable. If a cache-related regression is discovered in production:
- `step_cache_enabled=False` disables step caching
- `use_cow_snapshots=False` reverts to deep-copy branch snapshots
- `persist_cache_keys=False` disables cross-run persistence

No code removal required for rollback.

---

## 11. Observability

### 11.1 Per-step logging (verbose ≥ 2)

```
[Step 3/8] SNV | shape: 5000×4×2000 | steady: 152.6 MB | RSS: 1.2 GB
[Step 3/8] SNV | shape: 5000×4×2000 | steady: 152.6 MB | RSS: 1.2 GB | cache: hit
```

### 11.2 End-of-dataset cache summary (verbose ≥ 1)

```
Step cache: 45 hits / 12 misses (78.9% hit rate) | 312.4 MB peak | 8 evictions
Transformer cache: 22 reuses / 6 fits
Peak RSS: 2.8 GB
```

### 11.3 StepCache.stats()

```python
def stats(self) -> dict:
    return {
        "hits": self._hits,
        "misses": self._misses,
        "evictions": self._evictions,
        "current_entries": len(self._cache),
        "current_mb": self._current_mb,
        "peak_mb": self._peak_mb,
        "hit_rate": self._hits / max(1, self._hits + self._misses),
    }
```

### 11.4 Memory warnings

RSS-based threshold (not just steady-state nbytes):

```
⚠ Process RSS at 3.8 GB (threshold: 3.0 GB) after step SNV.
  Largest source: source[0] 5000×12×2000 = 457.8 MB
  Active branches: 4
```

### 11.5 Inflight byte tracking

For the transformer controller specifically (where transient copies cause the worst peaks), add lightweight tracking:

```python
# In TransformerMixinController.execute():
inflight_bytes = all_data.nbytes + fit_data.nbytes  # transient buffers
if inflight_bytes > INFLIGHT_WARNING_MB * 1e6:
    logger.warning(f"Transform inflight buffers: {inflight_bytes / 1e6:.0f} MB")
```

This captures the `all_data` + `fit_data` + transform buffer peak that steady-state metrics miss.

---

## 12. File-by-File Change Map

### PR A — Safety, Hash Fix, Observability

| File | Changes |
|------|---------|
| `data/performance/lazy_loader.py` | Add `DeprecationWarning` to `LazyArray.__init__`, `LazyDataset.__init__` |
| `data/performance/__init__.py` | Keep exports, add deprecation note in docstring |
| `data/performance/cache.py` | Add `DeprecationWarning` to `cache_manager()` |
| `data/features.py` | Add `DeprecationWarning` when `cache=True` is passed |
| `data/dataset.py` | Fix: consolidate `_content_hash` / `_content_hash_cache` into single attribute |
| `data/config.py` | Add byte-budget to `DatasetConfigs.cache`, hash-based key derivation |
| `utils/memory.py` | **New file** — `estimate_dataset_bytes()`, `estimate_cache_entry_bytes()`, `get_process_rss_mb()`, `format_bytes()` |
| `pipeline/execution/executor.py` | Add per-step memory + RSS logging |
| `pipeline/execution/orchestrator.py` | Remove premature StepCache stats logging |
| `tests/integration/test_memory_stress.py` | **New file** — baseline stress tests |

### PR B — Step-Level Caching

| File | Changes |
|------|---------|
| `config/cache_config.py` | **New file** — `CacheConfig` dataclass |
| `pipeline/config/context.py` | Add `step_cache` and `cache_config` fields to `RuntimeContext` |
| `pipeline/execution/orchestrator.py` | Pass `step_cache` to `RuntimeContext` when enabled |
| `pipeline/execution/step_cache.py` | Replace `SpectroDataset` deep-copy with `CachedStepState`; add `selector_fingerprint` to key; dedicated size estimator |
| `pipeline/execution/executor.py` | Add cache lookup/store around step execution |
| `controllers/controller.py` | Add `supports_step_cache()` class method (default `False`) |
| `controllers/transforms/transformer.py` | Override `supports_step_cache()` → `True` |
| `data/performance/cache.py` | Fix `get()` to update `timestamp` on access |
| `tests/integration/test_step_cache_correctness.py` | **New file** — cache on/off equivalence test |

### PR C — Branch Snapshots

| File | Changes |
|------|---------|
| `data/_features/array_storage.py` | Add `SharedBlocks` class with CoW semantics |
| `data/_features/feature_source.py` | Integrate `SharedBlocks` for snapshot/restore |
| `controllers/data/branch.py` | Replace `copy.deepcopy` with `SharedBlocks.acquire()` / `detach()` |
| `pipeline/execution/executor.py` | Replace `copy.deepcopy(features_snapshot)` with CoW acquire; add cleanup after branch completion |

### PR D — Feature Storage (conditional)

| File | Changes |
|------|---------|
| `data/_features/array_storage.py` | Replace `_data` (3D array) with `_blocks` (list of 2D) + lazy `_cached_3d` |
| `data/_features/feature_source.py` | Add `max_processings`, `eviction_policy`, `add_processings_batch()` |
| `controllers/data/feature_augmentation.py` | Use `add_processings_batch()` |

### PR E — Cross-Run Persistence

| File | Changes |
|------|---------|
| `pipeline/storage/artifacts/artifact_registry.py` | Add `hydrate_cached_artifacts()` method |
| `pipeline/execution/orchestrator.py` | Call `hydrate_cached_artifacts()` before dataset loop, `persist_cache_keys_to_store()` after |

---

## 13. Acceptance Criteria and Quality Governance

### PR A
- [ ] `_content_hash` / `_content_hash_cache` consolidated — single attribute
- [ ] `DatasetConfigs.cache` bounded by bytes (2 GB default)
- [ ] Deprecation warnings emitted for `LazyArray`, `LazyDataset`, `cache_manager`, `Features(cache=True)`
- [ ] Per-step memory + RSS logging visible at `verbose=2`
- [ ] Stress test baselines established (peak RSS for 3 scenarios)
- [ ] All existing tests pass unchanged

### PR B
- [ ] `StepCache` wired: non-zero hit/miss stats on generator pipelines when enabled
- [ ] Cache key includes `selector_fingerprint` (partition, fold, processing, branch_path)
- [ ] Size estimation uses numpy `nbytes`, not pickle
- [ ] `DataCache.get()` updates `timestamp` on access
- [ ] Pipeline results identical with cache ON vs OFF (within float tolerance)
- [ ] Budget respected: memory under `step_cache_max_mb`
- [ ] `examples/run.sh -q` passes with cache ON
- [ ] Feature flag: OFF by default

### PR C
- [ ] Branch snapshots use CoW references
- [ ] 6-branch pipeline memory stays < 2× single-branch (vs current 7×)
- [ ] Post-branch steps that don't modify features don't trigger copies
- [ ] Cleanup after merge releases all branch snapshot references
- [ ] All branch/merge tests pass unchanged

### PR D (conditional)
- [ ] Benchmark gate: peak RSS reduction ≥ 30% on augmentation-heavy scenario
- [ ] No performance regression > 5% on standard pipelines
- [ ] `add_processing()` does not copy the full existing array
- [ ] `max_processings_per_source` enforcement works
- [ ] Exact numerical equivalence of results

### PR E
- [ ] `hydrate_cached_artifacts()` fully hydrates `ArtifactRecord` from store + validates binary existence
- [ ] Missing binary files → graceful cache miss, not error
- [ ] `record.class_name` validation prevents type mismatch
- [ ] Second run on same dataset + pipeline shows cache hits for fitted transformers
- [ ] Cross-run test: run pipeline twice, verify fit count drops on second run

### Quality governance (all PRs)

- **Stress tests in CI**: The 3 scenarios from PR A run on every PR to detect regressions
- **Go/no-go criteria**: Each PR must pass stress tests before merge; RSS must not increase vs baseline
- **Rollback flags**: Each feature independently disableable via `CacheConfig` field
- **No silent behavioral changes**: Any default that changes pipeline behavior (e.g., feature augmentation action) requires a deprecation warning period + version-gated migration (see [§14.4](#144-feature-augmentation-default))

---

## 14. What NOT to Do

### 14.1 Don't build a view DAG / lineage graph

The "from-scratch" investigation proposes an immutable-snapshot + virtual-view architecture. While theoretically cleaner, it requires rewriting the execution model, controller interface, and dataset mutation semantics. The current mutable-dataset model works. Fix its boundaries, don't replace it.

**What to keep from the from-scratch proposal**: the distinction between `fit_key` and `transform_key` (integrated into §6.2), and multi-level budgets covering both steady-state and inflight bytes (integrated into §11.5).

### 14.2 Don't add disk spill for intermediate feature arrays

`np.memmap` or zarr for feature blocks adds I/O complexity and platform-specific behavior. The simpler path is evicting cold processings from RAM entirely and recomputing on demand (which the step cache makes cheap).

### 14.3 Don't cache model training results

Model steps are the variant-specific part of a pipeline. Caching their outputs defeats the purpose of trying multiple configurations.

### 14.4 Feature augmentation default — don't change silently

The current default `action="add"` in `FeatureAugmentationController` (`feature_augmentation.py:138`) encourages multiplicative processing growth. Changing this to `"extend"` would be safer for memory but is a **functional change** — existing tests explicitly verify the `"add"` default behavior.

**Correct approach**: Do NOT change the default in this work. Instead:
1. Log a warning when `action="add"` with a large number of processings (e.g., ≥ 10) would produce > 50 processing slots
2. Document the memory implications of `"add"` vs `"extend"` in docstrings
3. If a default change is desired, do it as a separate versioned migration: deprecation warning in version N, default change in version N+1

### 14.5 Don't attempt to parallelize step execution

Parallelism across branches or variants would require thread-safe dataset access. Caching shared prefixes gives most of the benefit without concurrency complexity.

### 14.6 Don't delete publicly exported symbols without deprecation

`LazyDataset`, `LazyArray`, `cache_manager` are exported from `data.performance.__init__`. `Features(cache=...)` is a public constructor parameter. Even though none are re-exported by user-facing modules, external code could import them directly. Deprecate first (PR A), remove later.
