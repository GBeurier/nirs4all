# Caching and Memory Optimization

This document describes the runtime caching subsystem introduced to reduce memory usage and wall-clock time during pipeline execution. It covers block-based feature storage, copy-on-write branch snapshots, step-level caching, and observability tooling.

## Overview

The caching subsystem addresses three performance problems in pipeline execution:

1. **Generator variant redundancy**: When `_or_` / `_range_` generators expand a pipeline into many variants, shared preprocessing prefixes are re-executed from scratch for every variant.
2. **Branch deep-copy explosion**: Each branch in a `{"branch": [...]}` step deep-copies the entire feature set, even when most branches only read features without modifying them.
3. **Array concatenation overhead**: Adding a new processing via `add_processing()` previously reallocated and copied the entire 3D array, causing transient 2x memory spikes.

The subsystem has four layers, each independently toggleable:

| Layer | What it does | Default | Config flag |
|-------|-------------|---------|-------------|
| **Block-based storage** | Per-processing 2D blocks instead of monolithic 3D array | Always on | N/A (structural) |
| **Copy-on-Write snapshots** | Branch snapshots share arrays via refcounting | ON | `use_cow_snapshots` |
| **Step-level caching** | Reuse preprocessing results across generator variants | OFF | `step_cache_enabled` |
| **Observability** | Per-step RSS logging, cache stats, memory warnings | ON | `log_step_memory`, `log_cache_stats` |

## Configuration

All cache settings live in a single dataclass:

```python
from nirs4all.config.cache_config import CacheConfig

cache = CacheConfig(
    # Step cache (off by default)
    step_cache_enabled=True,
    step_cache_max_mb=2048,       # 2 GB budget
    step_cache_max_entries=200,

    # Branch snapshots (on by default)
    use_cow_snapshots=True,

    # Observability
    log_cache_stats=True,
    log_step_memory=True,
    memory_warning_threshold_mb=3072,  # 3 GB
)

result = nirs4all.run(
    pipeline=pipeline,
    dataset=dataset,
    cache=cache,
)
```

When no `cache=` argument is provided, `CacheConfig()` defaults are used: CoW snapshots are on, step caching is off, and observability logging is on.

### Configuration fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `step_cache_enabled` | `bool` | `False` | Enable step-level caching for cross-variant reuse |
| `step_cache_max_mb` | `int` | `2048` | Maximum memory budget for the step cache (MB) |
| `step_cache_max_entries` | `int` | `200` | Maximum number of cached step entries |
| `use_cow_snapshots` | `bool` | `True` | Use copy-on-write snapshots for branch features |
| `log_cache_stats` | `bool` | `True` | Log end-of-run cache statistics at `verbose >= 1` |
| `log_step_memory` | `bool` | `True` | Log per-step memory stats at `verbose >= 2` |
| `memory_warning_threshold_mb` | `int` | `3072` | RSS threshold (MB) for memory warnings |

### Config flow

```
nirs4all.run(cache=CacheConfig(...))
  -> PipelineRunner.cache_config
    -> PipelineOrchestrator.cache_config
      -> RuntimeContext(cache_config=, step_cache=)
        -> PipelineExecutor (reads from runtime_context)
        -> BranchController (reads use_cow_snapshots)
```

---

## Block-Based Feature Storage

**Location**: `nirs4all/data/_features/array_storage.py`

### Problem

The previous implementation stored all processings in a single 3D numpy array of shape `(samples, processings, features)`. Every call to `add_processing()` allocated a new 3D array and copied all existing data via `np.concatenate`. Peak memory during the concatenation was ~2x steady-state.

### Solution

`ArrayStorage` now maintains a list of 2D blocks, one per processing:

```
Before (monolithic):
  _array: np.ndarray  # shape (N, P, F) — one 3D array

After (block-based):
  _blocks: List[np.ndarray]          # P blocks, each (N, F)
  _cached_3d: Optional[np.ndarray]   # lazily computed
  _shared: Optional[SharedBlocks]    # CoW state (Phase 3)
```

### Key operations

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `add_processing(data)` | O(1) | Appends a 2D block, no copy of existing data |
| `add_processings_batch(blocks)` | O(k) | Appends k blocks, invalidates cache once |
| `remove_processing(idx)` | O(1) | Drops a block from the list |
| `.array` (property) | O(N*P*F) lazy | Builds 3D via `np.stack`, cached until next mutation |
| `.nbytes` (property) | O(P) | Sum of block `.nbytes` values |
| `get_data(indices)` | O(N*P*F) | Indexes into the 3D view |

### Lazy 3D cache

The `.array` property builds the 3D array only when accessed and caches it in `_cached_3d`. Any mutation (`add_processing`, `remove_processing`, `update_processing`, `resize_features`) invalidates the cache by calling `_prepare_for_mutation()`.

Code paths that only need 2D access (e.g., per-processing iteration) can avoid the 3D build entirely by iterating `_blocks` directly.

### Processing eviction

`FeatureSource.evict_oldest_processings(max_processings)` removes the oldest non-raw processings (by insertion order, preserving the raw processing at index 0). This prevents unbounded growth during feature augmentation.

```python
# In FeatureSource:
evicted = source.evict_oldest_processings(max_processings=50)
```

The eviction works with `ProcessingManager.remove_processing()` to drop the processing ID and reindex, and `ArrayStorage.remove_processing()` to drop the corresponding 2D block.

---

## Copy-on-Write Branch Snapshots

**Location**: `nirs4all/data/_features/array_storage.py` (SharedBlocks), `nirs4all/controllers/data/branch.py`

### Problem

Branch execution deep-copies the entire feature set for each branch. For N duplication branches with a 64 MB feature set, total copies are ~N*128 MB (snapshot + restore copies per branch). Most post-branch steps (model training) only read features without modifying them — the deep-copy is wasted.

### Solution: SharedBlocks

`SharedBlocks` is a reference-counted wrapper around a numpy array:

```python
class SharedBlocks:
    __slots__ = ("_array", "_refcount")

    def acquire(self) -> "SharedBlocks"   # refcount += 1, return self
    def release(self) -> None             # refcount -= 1
    def is_shared(self) -> bool           # refcount > 1
    def detach(self) -> "SharedBlocks"    # copy only if shared
```

The key insight is `detach()`: if the array is shared by multiple branches (`refcount > 1`), it creates an independent copy and decrements the original's refcount. If only one holder exists, it returns self — no copy.

### ArrayStorage CoW integration

`ArrayStorage` provides three methods for CoW lifecycle:

```python
# 1. Before branching: wrap current data in SharedBlocks
shared = storage.ensure_shared()

# 2. Branch restore: enter shared mode (zero-copy)
storage.restore_from_shared(shared.acquire())

# 3. On mutation: CoW trigger (copy only if shared)
storage._prepare_for_mutation()  # calls _materialize_blocks() if needed
```

### Branch lifecycle

```
BranchController.execute()
  |
  | 1. _snapshot_features(dataset, use_cow=True)
  |    -> For each source: source._storage.ensure_shared().acquire()
  |    -> Returns lightweight snapshot (SharedBlocks refs + metadata)
  |
  | 2. Per branch:
  |    _restore_features(dataset, snapshot, use_cow=True)
  |      -> source._storage.restore_from_shared(shared.acquire())
  |    Execute branch steps...
  |      -> If step mutates features (transformer):
  |         _prepare_for_mutation() -> _materialize_blocks()
  |           -> detach() copies array only if refcount > 1
  |
  | 3. _release_snapshot(snapshot, use_cow=True)
  |    -> Releases shared references for prompt GC
```

### Memory improvement

For 6 branches with a 64 MB feature set:

| Approach | Additional memory |
|----------|------------------|
| Deep-copy (old) | ~768 MB (12 full copies) |
| CoW (new) | ~0 MB if read-only, ~64 MB per branch that mutates |

### Rollback

Set `use_cow_snapshots=False` in `CacheConfig` to revert to deep-copy behavior.

---

## Step-Level Caching

**Location**: `nirs4all/pipeline/execution/step_cache.py`, `nirs4all/pipeline/execution/executor.py`

### Problem

Generator sweeps like `[{_or_: [SNV, MSC]}, PLS({_range_: [1, 20]})]` expand into many variants. Each variant re-executes all preprocessing from scratch, even when variants share a common prefix. For 2 scalers x 20 PLS components = 40 variants, the SNV transform runs 20 times with identical input.

### Solution

`StepCache` stores lightweight feature snapshots (`CachedStepState`) after each preprocessing step, keyed by `(chain_path_hash, data_hash, selector_fingerprint)`.

When a variant encounters a step that has already been computed with the same input:
1. **Cache hit**: Restore the feature state from the snapshot, skip execution.
2. **Cache miss**: Execute normally, store the result.

### Cache key composition

```
key = "step:{chain_path_hash}:{data_hash}:{selector_fingerprint}"
```

| Component | What it captures | Why |
|-----------|-----------------|-----|
| `chain_path_hash` | MD5 of the step's serialized config | Identifies the operator and its parameters |
| `data_hash` | Content hash of the dataset before the step | Ensures input data matches |
| `selector_fingerprint` | SHA-256 of partition, fold_id, processing, include_augmented, tag_filters, branch_path | Prevents cross-contamination between folds/partitions |

### CachedStepState

```python
@dataclass
class CachedStepState:
    features_sources: List[Any]       # Deep-copied FeatureSources
    processing_names: List[List[str]] # Processing chain per source
    content_hash: str                 # Post-step hash
    bytes_estimate: int               # numpy nbytes (no pickle)
```

Only feature sources are deep-copied — targets, metadata, fold assignments, and tags are immutable during preprocessing and are not included. This makes snapshots significantly smaller than full dataset copies.

### Which steps are cached

Only controllers that return `True` from `supports_step_cache()` are cached:

| Controller | Cached | Reason |
|-----------|--------|--------|
| `TransformerMixinController` | Yes | Preprocessing transforms benefit from cross-variant reuse |
| `ModelController` | No | Variant-specific — caching defeats the purpose |
| `SplitterController` | No | Cheap to execute, stateful with fold indices |
| `BranchController` | No | Complex state, handled by CoW snapshots |
| `MergeController` | No | Depends on branch results |
| `FeatureAugmentationController` | No | Produces many processings, full result is large |

### Eviction

The step cache uses **LRU** (Least Recently Used) eviction, which is appropriate because early pipeline steps are accessed often during variant expansion, then become cold. The underlying `DataCache` backend updates timestamps on every access and evicts by oldest timestamp when the budget is exceeded.

Budget is enforced by both entry count (`step_cache_max_entries`) and total bytes (`step_cache_max_mb`).

### Execution flow

```
PipelineExecutor._execute_single_step():
  |
  | step_cache = runtime_context.step_cache
  | if step_cache and mode == "train" and _is_step_cacheable(step):
  |   |
  |   | 1. Compute cache key
  |   | 2. step_cache.get(key) -> CachedStepState or None
  |   |
  |   | HIT:
  |   |   step_cache.restore(state, dataset)
  |   |   return  # Skip execution
  |   |
  |   | MISS:
  |   |   Execute step normally
  |   |   step_cache.put(key, dataset)
```

---

## Observability

### Per-step memory logging

At `verbose >= 2`, each step logs its steady-state memory footprint and process RSS:

```
[Step 3/8] SNV | shape: 5000x4x2000 | steady: 152.6 MB | RSS: 1.2 GB
```

At any verbosity level, a warning fires when RSS exceeds `memory_warning_threshold_mb`:

```
Warning: Process RSS at 3.8 GB (threshold: 3.0 GB) after step SNV
```

### Cache statistics

At `verbose >= 1`, after each dataset execution completes (when `log_cache_stats=True`):

```
Step cache: 45 hits / 12 misses (78.9% hit rate) | 312.4 MB peak | 8 evictions
```

Available via `step_cache.stats()`:

```python
{
    "hits": 45,
    "misses": 12,
    "hit_rate": 0.789,
    "evictions": 8,
    "peak_mb": 312.4,
    "entries": 10,
    "max_entries": 200,
    "size_mb": 280.5,
    "max_size_mb": 2048.0,
}
```

### Memory estimation utilities

**Location**: `nirs4all/utils/memory.py`

| Function | Description |
|----------|-------------|
| `estimate_dataset_bytes(dataset)` | Sum of numpy `nbytes` across all feature sources and targets |
| `estimate_cache_entry_bytes(entry)` | numpy `nbytes` if available, `sys.getsizeof` fallback (no pickle) |
| `get_process_rss_mb()` | Process RSS via `/proc/self/statm` (Linux) or `psutil` (macOS) |
| `format_bytes(n)` | Human-readable: `"152.6 MB"`, `"1.2 GB"` |

All estimation functions are O(1) per array (read shape and dtype only). No serialization or pickle calls.

---

## Architecture Diagram

```
nirs4all.run(cache=CacheConfig(...))
  |
  v
PipelineRunner -------- cache_config ---------> PipelineOrchestrator
                                                  |
                                                  | Creates StepCache (if enabled)
                                                  | Creates RuntimeContext
                                                  v
                                              PipelineExecutor
                                                  |
                               +------------------+------------------+
                               |                                     |
                         Step cacheable?                        Not cacheable
                               |                                     |
                        cache.get(key)                         Execute normally
                          /        \
                       HIT          MISS
                        |             |
                   Restore from    Execute step
                   CachedStepState    |
                        |          cache.put(key)
                        v             v
                     Next step     Next step


Branch execution with CoW:

  BranchController.execute()
    |
    |  ensure_shared() + acquire()     # O(1) snapshot
    |
    +-- Branch 1: restore_from_shared()
    |     |-> Step modifies features? -> detach() -> copy
    |     |-> Step reads only?        -> no copy
    |
    +-- Branch 2: restore_from_shared()
    |     |-> ...
    |
    |  release_snapshot()              # Free shared refs
```

---

## Performance Tuning

### Step cache budget

The default 2 GB / 200 entries is suitable for most workloads. For large datasets with many generator variants, increase the budget:

```python
cache = CacheConfig(
    step_cache_enabled=True,
    step_cache_max_mb=4096,       # 4 GB
    step_cache_max_entries=500,
)
```

Monitor eviction counts via the end-of-run stats. If evictions are high, the budget is too small and cache effectiveness is reduced.

### Memory warning threshold

The default 3 GB threshold may need adjustment for machines with more or less RAM:

```python
cache = CacheConfig(
    memory_warning_threshold_mb=8192,  # 8 GB for large-memory machines
)
```

### Disabling features for debugging

Each layer can be independently disabled:

```python
# Disable step caching (default)
CacheConfig(step_cache_enabled=False)

# Disable CoW snapshots (fall back to deep-copy)
CacheConfig(use_cow_snapshots=False)

# Disable observability logging
CacheConfig(log_cache_stats=False, log_step_memory=False)
```

---

## Example

See `examples/developer/06_internals/D03_cache_performance.py` for a runnable side-by-side comparison of cache configurations measuring wall-clock time, RSS delta, and cache hit statistics.

```bash
cd examples
./run.sh -n "D03*"
```

---

## Source Files

| File | Contents |
|------|----------|
| `nirs4all/config/cache_config.py` | `CacheConfig` dataclass |
| `nirs4all/data/_features/array_storage.py` | `ArrayStorage` (block-based), `SharedBlocks` (CoW) |
| `nirs4all/data/_features/feature_source.py` | `FeatureSource.evict_oldest_processings()` |
| `nirs4all/data/_features/processing_manager.py` | `ProcessingManager.remove_processing()` |
| `nirs4all/data/performance/cache.py` | `DataCache` (LRU/LFU backend), `CacheEntry` |
| `nirs4all/pipeline/execution/step_cache.py` | `StepCache`, `CachedStepState` |
| `nirs4all/pipeline/execution/executor.py` | Cache wiring in `_execute_single_step()` |
| `nirs4all/pipeline/execution/orchestrator.py` | `StepCache` creation and stats logging |
| `nirs4all/pipeline/config/context.py` | `RuntimeContext` carries cache state |
| `nirs4all/controllers/controller.py` | `supports_step_cache()` base method |
| `nirs4all/controllers/transforms/transformer.py` | `supports_step_cache() -> True` |
| `nirs4all/controllers/data/branch.py` | CoW snapshot/restore methods |
| `nirs4all/utils/memory.py` | Memory estimation utilities |
