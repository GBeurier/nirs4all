# Cache Implementation Backlog — nirs4all

> **Status**: Implementation backlog (ready to execute)
> **Date**: 2026-02-07
> **Derived from**: `cache_management_design.md` (design proposal v2)
> **Scope**: Runtime-only caching. No cross-run disk persistence. No over-engineering.

---

## Scope Decisions

This backlog is a practical subset of the full design document (`cache_management_design.md`). The following scoping decisions were made upfront:

| Decision | Rationale |
|----------|-----------|
| **No cross-run disk persistence** (design doc PR E dropped) | The `persist_cache_keys_to_store()` / `load_cached_from_store()` / `hydrate_cached_artifacts()` path adds DuckDB coupling and binary-file validation for a marginal speedup on iterative workflows. Runtime caching covers the dominant use case (generator variant reuse within a single `nirs4all.run()` call). |
| **Delete dead code immediately** (no deprecation cycle) | Project convention in CLAUDE.md: *"Never keep dead code, obsolete code or deprecated code. I want a clean repository (no backward compatibility)."* The design doc's deprecation-first approach contradicts this. |
| **Block-based feature storage is conditional** (Phase 4) | Only implement if Phase 1 stress tests show array concatenation is a significant bottleneck after step caching and CoW branches are in place. |
| **No view DAG / immutable lineage graph** | The design doc explicitly rejects this (§14.1). Fix boundaries of the current mutable-dataset model, don't replace it. |
| **No disk spill** | No `np.memmap` or zarr for intermediate arrays. Evict cold processings and recompute via step cache instead. |
| **No model result caching** | Model steps are variant-specific. Caching their output defeats the purpose of trying multiple configurations. |
| **No parallel step execution** | Would require thread-safe dataset access. Step caching gives most of the benefit without concurrency complexity. |

---

## Phase 1 — Cleanup, Bug Fixes, and Observability

*Foundation phase. No behavioral change to pipeline execution. Establishes correctness, removes confusion from dead code, and adds the measurement infrastructure needed to make informed decisions about Phases 2–4.*

### 1.1 Delete dead caching infrastructure

**Problem**: Six fully-implemented but completely unused components create false confidence about what is actually cached and clutter the codebase. A developer reading the code would reasonably assume these are wired in — they are not.

**What to delete and why**:

| Symbol | Location | Evidence it's dead |
|--------|----------|--------------------|
| `LazyArray` class | `data/performance/lazy_loader.py:21-146` | Fully implemented (load-on-demand arrays with `load()`/`unload()` methods). Exported from `data/performance/__init__.py:13-16`. Never instantiated anywhere in the codebase — no imports outside the module itself. Not re-exported by `data/__init__.py`. |
| `LazyDataset` class | `data/performance/lazy_loader.py:149-295` | Same situation. Wraps X, y, metadata as `LazyArray`s. Never instantiated. |
| `create_lazy_dataset()` | `data/performance/lazy_loader.py:298-348` | Helper for `LazyDataset`. Never called. |
| `cache_manager()` singleton | `data/performance/cache.py:335-347` | Returns a global `DataCache` instance. Only consumer was `StepCache` which creates its own `DataCache` directly (L55-59). No other call sites exist. |
| `Features(cache=...)` parameter | `data/features.py:21` | Constructor accepts `cache: bool = False` and stores `self.cache = cache`. The attribute is **never read** by any code — the docstring itself says "not yet implemented". |
| `persist_cache_keys_to_store()` | `pipeline/storage/artifacts/artifact_registry.py` | Implemented, tested, never called from orchestrator or anywhere in the execution path. Part of the cross-run persistence path we're explicitly excluding. |
| `load_cached_from_store()` | `pipeline/storage/artifacts/artifact_registry.py` | Returns only an `artifact_id` string — doesn't hydrate a runtime `ArtifactRecord`. Has a design gap (the hydration bridge described in the design doc §9.2 was never built). Never called. |

**Also clean up**:
- Remove the `lazy_loader.py` file entirely
- Update `data/performance/__init__.py` exports: remove `LazyArray`, `LazyDataset`, `cache_manager`
- Remove premature StepCache stats logging in orchestrator (stats are always zero because StepCache is instantiated but never passed to executor/step_runner/controllers)
- Remove any dangling imports of these symbols across the codebase

**Acceptance criteria**:
- All existing tests pass unchanged
- `ruff check nirs4all/data nirs4all/pipeline/execution nirs4all/pipeline/storage` passes
- No references to deleted symbols remain (grep confirms)
- `data/performance/__init__.py` exports only `DataCache` and `CacheEntry`

---

### 1.2 Fix dual-hash-attribute bug

**Problem**: `SpectroDataset` has two disconnected hash attributes that can diverge silently.

**Current state** (in `data/dataset.py`):
- **L73**: `self._content_hash_cache: Optional[str] = None` — the canonical cached hash, used by `content_hash()` (L1485-1522) and invalidated by `_invalidate_content_hash()` (L1481-1483) on every feature mutation
- **L2032-2039**: `set_content_hash()` sets `self._content_hash` — a **completely separate attribute** that is never declared in `__init__`, never invalidated, and bypasses the cache system entirely
- **L1945**: `get_dataset_metadata()` checks `hasattr(self, '_content_hash') and self._content_hash` — reading the wrong attribute. If `set_content_hash()` was called, it reads the manually-set value. If not, it falls through to computing a sample-based hash (first/last 100 rows) which differs from the full `content_hash()` method

**The bug**: After calling `set_content_hash("abc123")`, `content_hash()` returns a freshly computed full-dataset hash (ignoring the manual value), while `get_dataset_metadata()` returns `"abc123"` (the manual value). These can diverge. Furthermore, any feature mutation invalidates `_content_hash_cache` but leaves `_content_hash` untouched, so `get_dataset_metadata()` returns a stale hash.

**Fix**:
1. `set_content_hash()` → set `self._content_hash_cache` (the canonical attribute)
2. `get_dataset_metadata()` → check `self._content_hash_cache` instead of `self._content_hash`
3. Remove all traces of the standalone `_content_hash` attribute
4. Existing `_invalidate_content_hash()` already clears `_content_hash_cache` — no change needed there

**Acceptance criteria**:
- Single hash attribute (`_content_hash_cache`) throughout the class
- `set_content_hash(v)` followed by `content_hash()` (default call, no `source_index`) returns `v`
- `set_content_hash(v)` followed by a feature mutation followed by `content_hash()` returns a freshly computed hash (not `v`)
- `get_dataset_metadata()["hash"]` agrees with `content_hash()`
- Existing unit tests pass (add a test for the round-trip if none exists)

---

### 1.3 Bound `DatasetConfigs.cache` by bytes

**Problem**: The raw data cache in `DatasetConfigs` (`data/config.py:178`) is a plain `Dict[str, Any]` with no size limit. It stores full loaded data tuples `(x_train, y_train, m_train, headers, ...)` keyed by dataset name (`config.py:368-393`). Two large dataset entries can blow RAM on their own.

**Secondary issue**: Keying by bare name (`config.py:368`: `if name in self.cache`) risks collisions if the same dataset name is loaded with different configuration parameters (e.g., different `test_size`, different `signal_type` filtering).

**Fix**:
1. Add `max_cache_bytes: int = 2 * 1024**3` (2 GB default) to `DatasetConfigs`
2. Track cumulative size of cached entries using numpy `nbytes` sums (via the `estimate_cache_entry_bytes()` utility from task 1.4)
3. When inserting would exceed budget, evict the oldest entry (FIFO — simple and sufficient since this cache rarely has more than a handful of entries)
4. Improve key derivation: include config parameters in the key, not just the name. Use canonical JSON (`sort_keys=True`) with a stable serializer for paths/enums and hash `(name, normalized_config)` to prevent collisions.
5. Explicitly exclude non-serializable runtime objects from the key (e.g., `_preloaded_dataset`). For these, use a separate key component based on object identity/version.

**Why FIFO over LRU**: The `DatasetConfigs` cache typically holds 1–3 entries (one per dataset in a multi-dataset run). LRU tracking adds complexity for no practical benefit at this scale.

**Acceptance criteria**:
- Cache stays within byte budget
- Loading the same dataset name with different parameters produces separate cache entries
- Loading datasets that would exceed the budget evicts older entries
- Existing tests pass unchanged

---

### 1.4 Memory estimation utilities

**Problem**: The codebase has no consistent way to estimate memory usage. The existing `DataCache._estimate_size()` (`cache.py:289-306`) falls back to `len(pickle.dumps(data))` for non-numpy types (`cache.py:304`). This is both slow (serializes the entire object) and inaccurate (pickle representation size ≠ in-memory size). It's also called on every cache insertion.

**Solution**: Create `nirs4all/utils/memory.py` with fast, numpy-first estimation:

```python
estimate_dataset_bytes(dataset)       # Sum of numpy nbytes across all feature sources
estimate_cache_entry_bytes(entry)     # numpy nbytes if array/dataset, sys.getsizeof fallback
get_process_rss_mb()                  # Read /proc/self/statm (Linux), psutil fallback (macOS)
format_bytes(n)                       # Human-readable: "152.6 MB"
```

**Design rationale**:
- `nbytes` is O(1) — just reads the array's dtype and shape. No serialization.
- `sys.getsizeof` is shallow but fast — acceptable as a last-resort fallback for small objects (metadata dicts, string lists).
- RSS tracking via `/proc/self/statm` is the lightest possible approach on Linux. No dependency on `psutil` (optional import for macOS).
- These utilities are used by every subsequent phase: bounded `DatasetConfigs.cache` (1.3), per-step memory logging (1.5), `CachedStepState` size estimation (2.4), stress tests (1.6).

**Acceptance criteria**:
- `estimate_dataset_bytes()` returns correct value for a known dataset (cross-check with manual `nbytes` sum)
- No pickle calls anywhere in size estimation
- `get_process_rss_mb()` returns > 0 on Linux
- Unit tests cover all functions

---

### 1.5 Per-step memory and RSS logging

**Problem**: The dominant memory peaks are transient — `np.concatenate` in `add_processing()` (`array_storage.py`), `copy.deepcopy` in branches (`branch.py`, `executor.py`), and `all_data + fit_data` buffers in the transformer controller. Steady-state `nbytes` checks miss these spikes entirely. There is currently no visibility into per-step memory behavior.

**Solution**: In `PipelineExecutor._execute_single_step()` (`pipeline/execution/executor.py`), after step execution, at `verbose >= 2`:

```
[Step 3/8] SNV | shape: 5000x4x2000 | steady: 152.6 MB | RSS: 1.2 GB
```

Emit a warning (always, regardless of verbose level) when RSS exceeds `memory_warning_threshold_mb` (default 3072 MB):

```
Warning: Process RSS at 3.8 GB (threshold: 3.0 GB) after step SNV
```

Also add lightweight inflight tracking in `TransformerMixinController.execute()` — log a warning when `all_data.nbytes + fit_data.nbytes` exceeds a threshold. This captures the transient peak that steady-state metrics miss.

**Acceptance criteria**:
- Memory info visible at `verbose=2` for each step
- RSS warning fires when threshold is exceeded
- No measurable performance overhead (the `nbytes` check and `/proc` read are both O(1))
- The threshold is configurable (later via `CacheConfig`, for now a module-level constant)

---

### 1.6 Memory stress test baselines

**Problem**: There are no tests that measure or track memory behavior. Without baselines, we can't evaluate whether Phases 2–4 actually help, and we can't detect memory regressions.

**Solution**: Add `tests/integration/test_memory_stress.py` with three scenarios:

1. **Generator variant explosion**: Large dataset (5000 samples x 1000 features) + 100 generated preprocessing variants via `_or_` / `_range_`. Measures peak RSS to establish the baseline that step caching (Phase 2) should improve.

2. **Branch-heavy pipeline**: 6+ duplication branches with post-branch model steps. Measures peak RSS to establish the baseline that CoW snapshots (Phase 3) should improve.

3. **Feature augmentation growth**: Feature augmentation with 20 operations in `add` mode. Measures peak RSS to evaluate whether block-based storage (Phase 4) is needed.

These tests:
- Record peak RSS values as test metadata (printed, not asserted — baselines shift with hardware)
- Assert no OOM / no crash
- Serve as go/no-go gates: Phase 4 is only warranted if scenario 3 shows a significant bottleneck after Phases 2+3

**Acceptance criteria**:
- Three stress test scenarios run successfully
- Peak RSS values are printed to stdout for each scenario
- Tests are tagged (e.g., `@pytest.mark.stress`) so they can be excluded from fast CI runs

---

## Phase 2 — Step-Level Caching (OFF by default)

*The biggest single performance win. When generators expand `[{_or_: [SNV, MSC]}, PLS({_range_: [1, 20]})]` into 40 variants, each variant currently re-executes all preprocessing from scratch. Step caching lets variant `[SNV, PLS(5)]` reuse the post-SNV result from variant `[SNV, PLS(3)]`.*

*Estimated impact: ~50% wall-time reduction for preprocessing-dominated generator pipelines (40 variants → 2 preprocessing executions instead of 40).*

### 2.1 `CacheConfig` dataclass

**Rationale**: All cache-related settings need a single, typed entry point. Each feature must be independently toggleable for rollback without code changes. The config flows through the existing `PipelineRunner` → `PipelineOrchestrator` → `RuntimeContext` chain.

**Create** `nirs4all/config/cache_config.py`:

```python
@dataclass
class CacheConfig:
    # Step cache (Phase 2) — OFF by default until stress-tested
    step_cache_enabled: bool = False
    step_cache_max_mb: int = 2048        # 2 GB budget
    step_cache_max_entries: int = 200

    # Branch snapshots (Phase 3)
    use_cow_snapshots: bool = True

    # Observability (Phase 1.5)
    log_cache_stats: bool = True
    log_step_memory: bool = True
    memory_warning_threshold_mb: int = 3072
```

**Integration**: Expose a `cache` parameter on the main execution entrypoint (runner + API facade), e.g. `nirs4all.run(..., cache=CacheConfig(step_cache_enabled=True))`. Default `CacheConfig()` is used when not specified.

**Why OFF by default**: Step caching introduces a new execution path (cache lookup → restore → skip step execution). Until the correctness test (2.7) and stress tests confirm no regressions, it must be opt-in. After validation, the default flips to `True`.

**Acceptance criteria**:
- `CacheConfig` is importable from `nirs4all.config`
- Config flows through `PipelineRunner` → `Orchestrator` → `RuntimeContext`
- `nirs4all.run()` accepts `cache=` parameter
- Default behavior (no `cache=` argument) is identical to current behavior

---

### 2.2 Fix cache key to include selector fingerprint

**Problem**: The current `StepCache` key (`step_cache.py:150-152`) is `f"step:{chain_path_hash}:{data_hash}"`. This is insufficient because the transformer's fit behavior depends on the execution context selector — specifically the partition, fold, `fit_on_all` flag, and `include_excluded` setting (see `transformer.py:141-147` where these are extracted from step config).

Two calls with the same chain + data hash but different selectors (e.g., different folds in cross-validation) would collide on the same cache key, returning incorrect cached results.

**Current risk**: Low in practice because the selector usually doesn't vary within a single pipeline variant run. But it's incorrect in principle, and step caching makes collisions more likely by caching across variants.

**Fix**: Add a `selector_fingerprint` component to the key:

```
key = f"step:{chain_path_hash}:{data_hash}:{selector_fingerprint}"
```

The `selector_fingerprint` hashes: `partition`, `fold_id`, `processing` index, `include_augmented`, `tag_filters`, `branch_path`. These are the fields from `DataSelector` (`pipeline/config/context.py`) that affect what data the transformer sees during fit and transform.

**Acceptance criteria**:
- Same chain + same data + same selector → cache hit
- Same chain + same data + different fold → cache miss
- Same chain + same data + different partition → cache miss
- Unit test verifies key uniqueness for different selectors

---

### 2.3 Add `supports_step_cache()` to controller hierarchy

**Rationale**: Not all pipeline steps produce cacheable output. Models are variant-specific (caching defeats the purpose). Splitters are cheap and stateful (fold indices). Branches have complex state handled separately (Phase 3). Only preprocessing transforms benefit from cross-variant caching.

**Change**: Add to `OperatorController` base class (`controllers/controller.py:14-69`):

```python
@classmethod
def supports_step_cache(cls) -> bool:
    """Whether this step's output should be cached for cross-variant reuse.

    Only preprocessing transforms benefit from step caching.
    Models, splitters, and branch/merge steps should not be cached.
    """
    return False
```

Override to `True` in `TransformerMixinController` only.

Explicitly leave `False` (inheriting default) for:
- `ModelController` — variant-specific, caching defeats the purpose
- `SplitterController` — cheap to execute, stateful with fold indices
- `BranchController` — complex state, handled by CoW snapshots (Phase 3)
- `MergeController` — depends on branch results
- `FeatureAugmentationController` — produces many processings, caching the full result is expensive

**Acceptance criteria**:
- `TransformerMixinController.supports_step_cache()` returns `True`
- All other controllers return `False`
- No change to existing behavior (the method is only consulted when step cache is enabled)

---

### 2.4 Lightweight `CachedStepState` (replace deep-copy with targeted snapshot)

**Problem**: The current `StepCache` deep-copies the entire `SpectroDataset` on both `put()` (`step_cache.py:116`) and `get()` (`step_cache.py:102`). A `SpectroDataset` contains feature arrays, target arrays, metadata, sample indices, headers, fold assignments, tags, and more. Preprocessing transforms only modify the feature arrays — everything else is unchanged.

Deep-copying 500 MB of features plus all the metadata twice (on store and on restore) is wasteful. It also uses `pickle.dumps()` for size estimation (`cache.py:304`), which serializes the entire object just to measure its size.

**Fix**: Replace the full dataset deep-copy with a targeted snapshot:

```python
@dataclass
class CachedStepState:
    features_sources: List[FeatureSource]  # deep-copied on store (the only thing that changes)
    processing_names: List[List[str]]       # processing chain per source
    content_hash: str                       # post-step content hash
    bytes_estimate: int                     # computed via numpy nbytes, NOT pickle
```

**On store**: Deep-copy only `features_sources` and `processing_names`. Skip metadata, y-values, sample indices, headers, fold assignments, tags — these are not modified by preprocessing transforms.

**On restore**: Deep-copy the cached `features_sources` back into the dataset, update the context processing names, set the content hash.

**Size estimation**: Use `estimate_cache_entry_bytes()` from Phase 1.4 (numpy `nbytes` sum over all source arrays). The `bytes_estimate` is computed once at store time and reused for cache budget enforcement — no per-access overhead.

**Acceptance criteria**:
- `CachedStepState` is smaller than a full `SpectroDataset` deep-copy
- Size estimation uses `nbytes`, never pickle
- Restoring from cache produces a dataset identical to what the step would have produced (verified by correctness test in 2.7)

---

### 2.5 Make StepCache eviction policy truly LRU

**Problem**: `DataCache.get()` (`cache.py:140`) increments `hit_count` on access but **never updates `timestamp`**. The eviction logic (`cache.py:282-285`) sorts by `(hit_count, timestamp)` — this is LFU (least-frequently-used) with creation-time tiebreaker, not LRU (least-recently-used).

For step caching, LRU is more appropriate: early pipeline steps are accessed often during variant expansion, then become cold. LFU would keep them around even after they're no longer useful.

**Fix**:
1. Add an explicit `eviction_policy` setting to `DataCache` (`"lfu"` default for backward compatibility, `"lru"` for StepCache).
2. In `DataCache.get()`, update `entry.timestamp = time.time()` on access.
3. For `"lru"` eviction mode, evict by oldest `timestamp` only (do not include `hit_count` in ordering).
4. Override `_estimate_size()` to use `CachedStepState.bytes_estimate` directly when the cached value is a `CachedStepState`, avoiding the pickle fallback path entirely.

**Acceptance criteria**:
- `timestamp` is updated on every `get()` call
- StepCache entries evict strictly by recency of access
- No pickle calls during size estimation for `CachedStepState` entries

---

### 2.6 Wire StepCache into execution pipeline

**Problem**: `StepCache` is instantiated in the orchestrator (`orchestrator.py`) but never passed to the executor, step_runner, or controllers. It has zero cache hits ever. Stats are always zero.

**Fix** (three wiring points):

**A. Pass through RuntimeContext**:
Add `step_cache: Optional[StepCache] = None` to `RuntimeContext` (in `pipeline/config/context.py`). The orchestrator creates the `StepCache` and passes it to `RuntimeContext` only when `cache_config.step_cache_enabled is True`.

**B. Cache lookup/store in executor**:
In `PipelineExecutor._execute_single_step()`:
1. Check if caching applies: `cache is not None` AND `controller.supports_step_cache()` AND not currently in a branch (branch state is complex, handled by Phase 3)
2. Before executing: compute cache key, look up in cache. On hit → restore feature state from `CachedStepState`, skip step execution.
3. After executing: build `CachedStepState` from current dataset state, store in cache.

**C. Budget enforcement**:
The `StepCache` wraps `DataCache` which already handles eviction. Configure it with `step_cache_max_mb` and `step_cache_max_entries` from `CacheConfig`.

**What to cache**: Only the dataset feature state after a preprocessing (transformer) step. This lets variant `[SNV, PLS(5)]` reuse the post-SNV result from variant `[SNV, PLS(3)]`.

**What NOT to cache at step level**:
- Model steps (variant-specific)
- Splitter steps (cheap, stateful with fold indices)
- Branch entry/exit (complex state, handled in Phase 3)
- Steps inside branches (branch state complicates key correctness)

**Acceptance criteria**:
- With `step_cache_enabled=True`, generator pipelines show non-zero hit/miss stats
- Cache budget (`step_cache_max_mb`) is respected — eviction fires when budget exceeded
- With `step_cache_enabled=False` (default), behavior is identical to current behavior — zero overhead
- The StepCache is scoped to a single `nirs4all.run()` call — created fresh each time, not persisted

---

### 2.7 Cache correctness test (mandatory gate)

**Rationale**: Step caching introduces a new execution path where step execution is skipped and results are restored from a snapshot. Any subtle difference in the restored state (e.g., missing processing metadata, slightly different array layout) would produce different pipeline results. This must be caught before enabling the cache by default.

**Add** `tests/integration/test_step_cache_correctness.py`:

1. Run the same generator pipeline (e.g., `[{_or_: [SNV, MSC]}, PLS({_range_: [1, 20]})]`) with `step_cache_enabled=True` and `step_cache_enabled=False`
2. Assert identical results: best score, best RMSE, all per-variant predictions within `atol=1e-10`
3. Run `examples/run.sh -q` with cache ON and verify all examples pass

**This test is a mandatory gate**: the step cache default cannot be flipped to `True` until this test passes on the full example suite.

**Acceptance criteria**:
- Results are numerically equivalent with cache ON vs OFF (`np.allclose(..., atol=1e-10, rtol=0)`)
- All examples pass with cache ON
- Test is included in the integration test suite

---

### 2.8 End-of-run cache summary logging

**Rationale**: Users and developers need visibility into whether the cache is working and whether the budget is appropriate.

At `verbose >= 1`, after each dataset execution completes:

```
Step cache: 45 hits / 12 misses (78.9% hit rate) | 312.4 MB peak | 8 evictions
```

Uses `StepCache.stats()` which tracks: hits, misses, evictions, current entries, current MB, peak MB, hit rate.

**Acceptance criteria**:
- Stats are accurate and non-zero for generator pipelines
- Stats are suppressed when `log_cache_stats=False` in `CacheConfig`
- Stats print nothing when step cache is disabled (no misleading zeros)

---

## Phase 3 — Copy-on-Write Branch Snapshots

*Fixes the deep-copy explosion during branching. Can be developed in parallel with Phase 2 — no dependency between them.*

### 3.1 `SharedBlocks` CoW wrapper

**Problem recap**: `BranchController._snapshot_features()` does `copy.deepcopy(dataset._features.sources)` (in `branch.py`). For N branches, total copies = 1 initial snapshot + 2N copies (restore + result snapshot per branch). The executor adds more copies per post-branch step.

Empirical: 3000 samples x 8 processings x 700 features (float32) → 64 MB base. 6 branches → 385 MB of additional copies.

Most post-branch steps (model training) **don't modify features** — they only read them. The deep-copy is wasteful because the copy is never mutated.

**Solution**: Introduce a reference-counted copy-on-write wrapper at the `ArrayStorage` level.

Add `SharedBlocks` class to `data/_features/array_storage.py`:

```python
class SharedBlocks:
    """Reference-counted immutable block wrapper. Copy-on-write semantics."""

    def __init__(self, array: np.ndarray):
        self._array = array
        self._refcount = 1

    def acquire(self) -> 'SharedBlocks':
        """Increment refcount. Returns self for chaining."""
        self._refcount += 1
        return self

    def release(self):
        """Decrement refcount."""
        self._refcount -= 1

    def is_shared(self) -> bool:
        return self._refcount > 1

    def detach(self) -> 'SharedBlocks':
        """Create an independent copy if shared, otherwise return self.

        This is the CoW trigger: only copies when someone else holds a reference.
        """
        if self._refcount > 1:
            self._refcount -= 1
            return SharedBlocks(self._array.copy())
        return self
```

**Why reference counting instead of `weakref`**: Weakrefs don't prevent garbage collection — we need the opposite: keep the shared array alive as long as any branch holds a reference. Manual refcounting is simple (3 methods), well-understood, and deterministic.

**Why at the `ArrayStorage` level**: This is where the actual numpy arrays live (`array_storage.py:28`: `self._array`). Wrapping at this level means branch snapshots share the underlying array without knowing about each other. Mutations trigger `detach()` which copies only when needed.

**Acceptance criteria**:
- `acquire()` increments refcount, `release()` decrements
- `detach()` on unshared block returns self (no copy)
- `detach()` on shared block returns new independent block, decrements original refcount
- Unit tests cover all three scenarios

---

### 3.2 Integrate `SharedBlocks` into `ArrayStorage`

**Change**: Wrap `ArrayStorage._array` (`array_storage.py:28`) in `SharedBlocks`.

On mutation (`add_processing`, `replace_features`, `reset_features`): call `self._shared = self._shared.detach()` before modifying. This is the CoW trigger — if the array is shared with a branch snapshot, `detach()` copies it. If not shared, it's a no-op.

**Public API of `ArrayStorage` does not change**: `x(indices, layout)`, `add_processing()`, `replace_features()`, `reset_features()` have the same signatures and return the same shapes. The change from `self._array` to `self._shared._array` is internal.

**Acceptance criteria**:
- All existing `ArrayStorage` tests pass unchanged
- `add_processing()` on a shared storage triggers a copy (detach)
- `add_processing()` on an unshared storage does NOT trigger a copy
- Array values are numerically identical before and after the refactoring

---

### 3.3 Replace deep-copy in `BranchController`

**Current code** (in `branch.py`):
```python
def _snapshot_features(self, dataset):
    return copy.deepcopy(dataset._features.sources)

def _restore_features(self, dataset, snapshot):
    dataset._features.sources = copy.deepcopy(snapshot)
```

Two full deep-copies per branch: one to snapshot, one to restore. For 6 branches with a 64 MB feature set, that's ~768 MB of copies.

**New code**:
```python
def _snapshot_features(self, dataset):
    """Lightweight snapshot: acquire shared references instead of deep-copying."""
    return [
        source._storage._shared.acquire()
        for source in dataset._features.sources
    ]

def _restore_features(self, dataset, snapshot):
    """Restore from snapshot. No copy unless mutation follows."""
    for source, shared in zip(dataset._features.sources, snapshot):
        source._storage._shared = shared.acquire()
```

**Memory improvement**: Branches now hold references to shared arrays instead of full copies. Additional branch memory is near-constant for read-only branches, and copies happen only on mutation.

Actual copies only happen when a branch modifies the feature arrays (e.g., a preprocessing step inside a branch calls `add_processing()`, triggering `detach()`).

**After branch completion**: Release all `SharedBlocks` references in that branch's snapshot. This ensures memory from completed branches is freed promptly (refcount drops, and if no other branch holds a reference, the array becomes eligible for GC).

Gate with `cache_config.use_cow_snapshots` flag: if `False`, fall back to the current `copy.deepcopy` behavior.

**Acceptance criteria**:
- 6-branch pipeline memory stays < 2x single-branch (vs current ~7x)
- All branch/merge integration tests pass unchanged
- Snapshot references are released after branch completion (no memory leak)
- With `use_cow_snapshots=False`, behavior reverts to deep-copy (rollback path)

---

### 3.4 Optimize executor post-branch steps

**Problem**: The executor (`executor.py`) does `copy.deepcopy(features_snapshot)` on every post-branch step. Most post-branch steps are model training — they read features but don't modify them. The deep-copy is unnecessary.

**Fix**: Replace `copy.deepcopy(features_snapshot)` with `shared.acquire()` and let copy-on-write handle mutations. Only steps that actually modify features (rare after branching) trigger a copy.

**Acceptance criteria**:
- Post-branch steps that don't modify features trigger zero copies
- Post-branch steps that DO modify features still work correctly (detach triggers copy)
- Integration tests pass unchanged

---

## Phase 4 — Block-Based Feature Storage (Conditional)

> **Gate**: Only implement if Phase 1 stress test scenario 3 (feature augmentation) shows that array concatenation is a significant bottleneck AFTER Phases 2+3 are in place. If step caching + CoW branches resolve the dominant memory issues, Phase 4 can be deferred indefinitely.

### 4.1 Replace monolithic 3D array with block list

**Problem**: `ArrayStorage.add_processing()` (`array_storage.py`) calls `np.concatenate((self._array, new_data_3d), axis=1)` on every new processing. This allocates a full new 3D array and copies all existing data. The old array coexists in RAM until GC collects it, so peak usage is ~2x steady-state during every concatenation.

For a dataset with S=5000 samples, F=1000 features, going from 1 to 13 processings: memory grows from 19 MB to 248 MB, with each concatenation briefly requiring 2x the current array size.

**Fix**: Replace `_array` (one 3D array: samples x processings x features) with `_blocks` (list of 2D arrays: each is samples x features, one per processing):

```python
class ArrayStorage:
    _blocks: List[np.ndarray]           # Each block is (samples, features)
    _block_ids: List[str]               # Processing ID per block
    _cached_3d: Optional[np.ndarray]    # Lazily computed, invalidated on mutation
```

**Benefits**:
- `add_processing()` appends a 2D block — O(1) allocation, no copy of existing data
- `remove_processing(id)` drops a block — O(1) memory release
- 3D view is computed lazily only when needed (some code paths only need 2D access)
- Individual processings can be evicted independently (useful for feature augmentation)
- Peak RAM during `add_processing()` drops from 2x to ~1x steady-state

**Migration**: Public API unchanged. Internal change from `_array: np.ndarray` to `_blocks: List[np.ndarray]` is private. `x(indices, layout)` returns the same shapes.

**Acceptance criteria**:
- Peak RSS reduction >= 30% on the feature-augmentation-heavy stress test scenario
- No performance regression > 5% on standard pipelines
- Exact numerical equivalence of results with the monolithic array implementation
- All existing tests pass unchanged

---

### 4.2 Processing eviction and batch add

**Processing eviction**: Add `max_processings` limit per source (default 50). When exceeded, evict the oldest non-raw processing block and log a warning. This prevents unbounded growth during feature augmentation with many operations.

**Batch add**: Add `add_processings_batch(blocks, ids)` for `FeatureAugmentationController`, which currently calls `add_processing()` M x P times. Batch add extends the block list in one call, invalidating `_cached_3d` only once.

**Acceptance criteria**:
- Processing count stays within `max_processings` limit
- Batch add produces identical results to sequential `add_processing()` calls
- Feature augmentation stress test shows improved performance

---

## Ordering and Dependencies

```
Phase 1 (Foundation) ──────────────────────────────────────────
  1.1 Delete dead code
  1.2 Fix dual-hash bug
  1.3 Bound DatasetConfigs.cache ──────── depends on 1.4
  1.4 Memory estimation utilities
  1.5 Per-step memory logging ─────────── depends on 1.4
  1.6 Stress test baselines ───────────── depends on 1.4, 1.5

Phase 2 (Step Cache) ──────── depends on Phase 1 ──────────────
  2.1 CacheConfig dataclass
  2.2 Fix cache key (selector fingerprint)
  2.3 supports_step_cache() method
  2.4 CachedStepState (lightweight snapshot)
  2.5 Fix DataCache eviction (LRU)
  2.6 Wire into execution ─────────────── depends on 2.1–2.5
  2.7 Correctness test ────────────────── depends on 2.6
  2.8 Cache summary logging ───────────── depends on 2.6

Phase 3 (CoW Branches) ──── independent of Phase 2 ───────────
  3.1 SharedBlocks class
  3.2 Integrate into ArrayStorage ─────── depends on 3.1
  3.3 Replace deep-copy in branches ───── depends on 3.2
  3.4 Optimize executor post-branch ───── depends on 3.2

Phase 4 (Block Storage) ─── conditional on Phase 1 metrics ───
  4.1 Block-based storage
  4.2 Eviction + batch add ────────────── depends on 4.1
```

**Parallelism**: Phases 2 and 3 can be developed in parallel after Phase 1 ships — they touch different files and solve different problems (cross-variant reuse vs branch memory).

**Go/no-go gates**:
- Phase 2 default flip (`step_cache_enabled=True`): blocked on correctness test (2.7) + example suite passing
- Phase 4: blocked on Phase 1 stress test metrics showing array concatenation is still a bottleneck after Phases 2+3

---

## What Is Explicitly Excluded

| Excluded | Rationale |
|----------|-----------|
| Cross-run disk persistence (design doc PR E) | Runtime caching covers the dominant use case. Disk persistence adds DuckDB coupling, binary-file validation, and a hydration bridge for marginal benefit. |
| View DAG / immutable lineage graph | Design doc §14.1: "too large a lift". Fix boundaries of mutable-dataset model instead. |
| Disk spill (`np.memmap`, zarr) | Design doc §14.2: adds I/O complexity. Evicting cold processings + recomputing via step cache is simpler. |
| Model result caching | Design doc §14.3: model steps are variant-specific. Caching defeats the purpose. |
| Parallel step execution | Design doc §14.5: requires thread-safe dataset access. Step caching gives most of the benefit. |
| Deprecation warning cycles | Project convention: delete dead code immediately, no backward compatibility. |
| Feature augmentation default change (`add` → `extend`) | Design doc §14.4: functional change. Separate versioned migration, not part of caching work. |

---

## Phase 5 — Documentation and Developer Guidance

### 5.1 Update developer documentation for caching subsystem

**Scope**: Document the caching subsystem for developers who need to understand or extend it.

**Deliverables**:

1. **Developer guide** (Sphinx / docs/source):
   - Architecture overview: block-based `ArrayStorage`, `SharedBlocks` CoW, `StepCache`, `DataCache`
   - How step caching works: cache keys, selector fingerprinting, snapshot/restore lifecycle
   - How CoW branch snapshots work: `ensure_shared()` / `restore_from_shared()` / `_materialize_blocks()`
   - How to enable/configure caching via `CacheConfig`
   - Performance tuning: `max_size_mb`, `max_entries`, `max_processings`

2. **API reference updates**:
   - Document `CacheConfig` dataclass fields and defaults
   - Document `ArrayStorage.ensure_shared()`, `restore_from_shared()`, `add_processings_batch()`, `remove_processing()`, `nbytes`
   - Document `FeatureSource.evict_oldest_processings()`
   - Document `ProcessingManager.remove_processing()`

3. **Update existing references**:
   - Update CLAUDE.md cache section if needed
   - Update examples README to reference `D03_cache_performance.py`

**Acceptance criteria**:
- All new public methods have Google-style docstrings (already done in code)
- Developer guide explains the block-based storage architecture with a state diagram
- `D03_cache_performance.py` example is referenced from docs
