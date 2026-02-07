# Step Cache Optimization: Diagnosis and Roadmap

**Date**: 2026-02-07
**Status**: Approved direction — Phase A (CoW Step Cache)
**Scope**: `nirs4all/pipeline/execution/step_cache.py`, `executor.py`
**Related**: `D03_cache_performance.py`, Codex analysis (`functional_data_cache_reanalysis_2026-02-07.md`)

---

## 1. Problem Statement

The D03 benchmark runs 192 pipeline variants (4 scatter × 4 smooth × 3 deriv × 4 PLS components) with three cache modes. Result: **CoW snapshots help memory; step cache shows no execution speedup — sometimes slower than baseline.**

The cache *is hitting correctly* (keys match, shared prefixes are detected). The problem is that the hit path is as expensive as recomputation.

---

## 2. Root Cause: Deep-Copy Overhead Exceeds Transform Cost

### 2.1 Cost model (200 samples × 1000 features, float32)

| Operation | Cost |
|-----------|------|
| SNV transform | ~0.1ms |
| SavitzkyGolay | ~0.5ms |
| EMSC | ~2ms |
| `content_hash()` (xxhash, 0.8MB) | ~0.15ms |
| `_index_state_hash()` (polars) | ~0.2ms |
| `copy.deepcopy(features_sources)` **snapshot** | ~1-3ms |
| `copy.deepcopy(features_sources)` **restore** | ~1-3ms |

**Cache interaction overhead: 2-7ms per step. Most NIRS transforms: <2ms.**

### 2.2 The four compounding costs

1. **Deep-copy on put** (`StepCache._snapshot()` → `copy.deepcopy(sources)`): O(data_size) per cache miss.
2. **Deep-copy on get** (`StepCache.restore()` → `copy.deepcopy(cached_sources)`): O(data_size) per cache hit.
3. **Data hashing per step**: `content_hash()` + `_index_state_hash()` on every cacheable step.
4. **Model training dominates**: PLS fit ≈ 50-200ms per variant, dwarfing any preprocessing savings.

### 2.3 D03 worked example

48 unique preprocessing chains × 4 PLS components = 192 variants.

| Mode | Preprocessing cost | Model cost | Total | Speedup |
|------|--------------------|------------|-------|---------|
| No cache | 192 × ~2ms = 384ms | 192 × 100ms = 19.2s | 19.6s | 1.0× |
| Current cache | 192 × ~5ms overhead = 960ms | 19.2s | 20.2s | **0.97× (slower)** |

The cache is a **net negative** because every cache hit still costs ~5ms (hash + deepcopy restore), while the transform it skips costs ~1ms.

### 2.4 Confirmation: the cartesian cache IS hitting

I traced the full key computation path. For each fresh dataset per variant:

- `content_hash()` → xxhash over raw arrays → **deterministic across fresh datasets** (same raw bytes from I/O cache, same float32 cast in `initialize_with_data()`).
- `_index_state_hash()` → polars hash_rows → **deterministic** (same samples added in same order).
- `selector_fingerprint` → SHA256 of (partition=None, fold_id=None, processing=["raw"], tags={}, branch_path=[]) → **identical** at step 1 for all variants.

Cache keys match. Hits occur. The step cache stats would show a healthy hit rate. **The problem is purely that hits are expensive, not that they're missing.**

### 2.5 Verdict on the Codex analysis

The Codex document is mostly accurate in its diagnosis. Correct observations:
- Deep-copy overhead is non-trivial
- Branch substeps bypass the step cache
- Model training dominates
- Fit-context identity is critical for correctness

**Where it goes wrong**: it proposes a DataNode/TransformNode graph abstraction with a two-tier hybrid architecture. This doesn't address the root cause (deep-copy cost) and adds complexity without benefit. The current keying model is already correct — the problem is implementation cost, not cache identity.

---

## 3. Solution: CoW Step Cache (Phase A)

### 3.1 Core idea

Replace `copy.deepcopy()` with `SharedBlocks` CoW references — the same mechanism already used for branch snapshots. The hit path becomes O(1) instead of O(data_size).

### 3.2 Current flow vs. proposed flow

**Current** (in `step_cache.py`):
```
PUT (_snapshot):  copy.deepcopy(features.sources)         → O(data_size)
GET (restore):    copy.deepcopy(cached.features_sources)   → O(data_size)
```

**Proposed**:
```
PUT (_snapshot):  ensure_shared().acquire() per source      → O(n_sources × np.stack)
GET (restore):    restore_from_shared(ref.acquire())        → O(n_sources)  ← near-free
                  next mutation triggers CoW detach()        → O(data_size), but deferred
```

### 3.3 Why this is safe

**CoW isolation guarantee**: `SharedBlocks.detach()` copies the array if and only if `refcount > 1`. After restore, the cache entry holds one reference, the live storage holds another. When the next transform mutates, `_prepare_for_mutation()` → `_materialize_blocks()` → `detach()` creates an independent copy. The cached data is never modified.

**Fold safety**: Unchanged. The cache key still includes `selector_fingerprint` which encodes `fold_id`. Different folds produce different keys.

**Index-state safety**: Unchanged. The cache key still includes `_index_state_hash()` which hashes exclusions, partitions, groups, tags.

**Content hash consistency**: After restore, `_content_hash_cache` is set from the cached state (O(1) for subsequent lookups). This is the same behavior as today.

### 3.4 Where the real cost goes

After Phase A:
- **Cache miss** (48 in D03): `ensure_shared()` does `np.stack(blocks)` ≈ O(data_size). Similar to current deep-copy. Unavoidable — we need to capture the state.
- **Cache hit** (144 in D03): `restore_from_shared(ref.acquire())` ≈ O(1). **This is the win.**
- **First mutation after hit**: `_materialize_blocks()` does CoW detach + slice ≈ O(data_size). But this happens inside the transform step which allocates new data anyway.

For consecutive cache hits (e.g., steps 1 AND 2 of a 3-step prefix are both hits), only ONE materialization happens (when the first non-cached step mutates). Current approach does TWO deep-copy restores.

### 3.5 Expected impact

| Mode | Preprocessing cost | Total | Speedup |
|------|--------------------|-------|---------|
| No cache | 192 × 2ms = 384ms | 19.6s | 1.0× |
| Current cache | 192 × 5ms = 960ms | 20.2s | 0.97× |
| **CoW cache** | 48 × 3ms + 144 × 0.1ms = 158ms | 19.4s | **1.01×** |

For expensive transforms (50ms/step, e.g. Wavelet on high-dimensional data):

| Mode | Preprocessing cost | Total | Speedup |
|------|--------------------|-------|---------|
| No cache | 192 × 100ms = 19.2s | 38.4s | 1.0× |
| Current cache | 48 × 100ms + 144 × 5ms = 5.5s | 24.7s | 1.55× |
| **CoW cache** | 48 × 100ms + 144 × 0.1ms = 4.8s | 24.0s | **1.60×** |

The main win is not a dramatic speedup — it's making the cache **stop being a net negative** so it can be enabled by default.

---

## 4. Implementation Roadmap

### Step 1: Instrument current cache to confirm diagnosis

**Goal**: Get hard numbers on where time goes, before changing anything.

**Changes in `executor.py`**:
- Time the `_step_cache_data_hash()` call (hash overhead).
- Time the `step_cache.get()` call (lookup overhead).
- Time the `step_cache.restore()` call (restore overhead).
- Time the `step_cache.put()` call (snapshot overhead).
- Time the actual step execution (transform compute time).
- Report per-step breakdown in `StepCache.stats()`.

**New fields in `StepCache.stats()`**:
```python
{
    "total_hash_ms": ...,       # Sum of all hash computations
    "total_snapshot_ms": ...,   # Sum of all snapshot (put) operations
    "total_restore_ms": ...,    # Sum of all restore (get) operations
    "total_transform_ms": ...,  # Sum of all transform executions (misses only)
    "net_impact_ms": ...,       # transform_ms_saved - (hash_ms + snapshot_ms + restore_ms)
}
```

**Success criterion**: `net_impact_ms` is negative on D03, confirming that the cache costs more than it saves.

### Step 2: CoW snapshot in `StepCache._snapshot()`

**Goal**: Replace `copy.deepcopy(sources)` with `ensure_shared().acquire()`.

**Changes in `step_cache.py`**:

```python
@dataclass
class CachedStepState:
    shared_refs: List[SharedBlocks]       # CoW refs per source (replaces features_sources)
    processing_names: List[List[str]]
    content_hash: str
    bytes_estimate: int
```

```python
@staticmethod
def _snapshot(dataset: SpectroDataset) -> CachedStepState:
    shared_refs = []
    processing_names = []
    total_bytes = 0
    for i, src in enumerate(dataset._features.sources):
        shared = src._storage.ensure_shared()
        shared_refs.append(shared.acquire())
        processing_names.append(list(dataset.features_processings(i)))
        total_bytes += src._storage.nbytes
    return CachedStepState(
        shared_refs=shared_refs,
        processing_names=processing_names,
        content_hash=dataset.content_hash(),
        bytes_estimate=total_bytes,
    )
```

**Safety**: `ensure_shared()` builds a 3D array from blocks and wraps it. `acquire()` increments refcount to 2 (storage + cache). When the next step calls `_prepare_for_mutation()`, the storage releases its ref (refcount → 1, cache still holds it). Blocks are then mutated freely.

### Step 3: CoW restore in `StepCache.restore()`

**Goal**: Replace `copy.deepcopy(cached_sources)` with `restore_from_shared()`.

```python
def restore(self, state: CachedStepState, dataset: SpectroDataset) -> None:
    for src, cached_ref in zip(dataset._features.sources, state.shared_refs):
        src._storage.restore_from_shared(cached_ref.acquire())
    dataset._content_hash_cache = state.content_hash
```

**Safety**: `acquire()` increments the cached ref's refcount. The storage takes ownership of one ref. When the next step mutates, `_materialize_blocks()` → `detach()` copies because refcount > 1. The cache's reference is never modified.

**Edge case — source count mismatch**: If the cached state has a different number of sources than the live dataset, this is a cache key mismatch that should never happen (content_hash would differ). Add an assertion:

```python
assert len(state.shared_refs) == len(dataset._features.sources), \
    "Source count mismatch between cached state and live dataset"
```

### Step 4: Release tracking on eviction

**Goal**: When the LRU cache evicts an entry, release the SharedBlocks references to avoid memory leaks.

**Changes**: Add a cleanup callback to `DataCache` or wrap the eviction logic in `StepCache`:

```python
def _release_state(state: CachedStepState) -> None:
    """Release CoW references when a cache entry is evicted."""
    for ref in state.shared_refs:
        ref.release()
```

The `DataCache` backend needs to call this on eviction and on `clear()`. Options:
- Add an `on_evict` callback parameter to `DataCache`.
- Override in `StepCache` by wrapping put/clear.

**Note on `DataCache` compatibility**: Check whether `DataCache` supports eviction callbacks. If not, add one. This is a small change.

### Step 5: Verify content_hash consistency after restore

**Goal**: Ensure that after CoW restore, the content_hash is consistent so the next step's cache lookup uses the correct key.

Current code already does:
```python
dataset._content_hash_cache = state.content_hash
```

After CoW restore, `restore_from_shared()` sets the storage to shared mode (no blocks, shared ref). When `content_hash()` is called for the next step's lookup, it returns `_content_hash_cache` (O(1)).

**Test**: After snapshot → restore round-trip, verify `dataset.content_hash()` matches the original post-step hash.

### Step 6: Update `bytes_estimate` for LRU eviction accuracy

**Goal**: The LRU cache uses `bytes_estimate` to track total memory. With CoW, the actual memory cost depends on sharing.

**Decision**: Keep `bytes_estimate` as the underlying array size (conservative). This over-estimates memory when data is shared (the same array counted once per cache entry), but it's simple and prevents the LRU from growing unbounded. Correct accounting would require tracking shared refcounts across entries, which adds complexity for little benefit.

### Step 7: Tests

**Unit tests** (new file: `tests/unit/pipeline/execution/test_step_cache_cow.py`):

1. **Round-trip correctness**: Snapshot → restore → verify features are byte-identical to original.

2. **CoW isolation**: Snapshot → restore → mutate dataset → verify cached state is unchanged. This is the critical safety test.

3. **Multiple restores**: Snapshot once → restore into two different datasets → mutate both → verify both get independent copies and cached state is unchanged.

4. **Eviction cleanup**: Fill cache to max → trigger eviction → verify evicted SharedBlocks refcounts reach 0.

5. **Content hash consistency**: Snapshot → restore → verify `content_hash()` returns correct value without recomputation.

6. **Processing names preserved**: Snapshot after multi-processing step → restore → verify `features_processings()` matches.

**Integration tests**:

7. **D03 benchmark correctness**: Run D03 with CoW cache and verify identical best scores vs. no-cache baseline.

8. **Complex pipeline**: Run a pipeline with branches + generators + stacking, verify predictions match no-cache mode.

### Step 8: Re-measure with instrumentation

**Goal**: Run D03 with the instrumented CoW cache and verify that `net_impact_ms` is now positive (or at least not negative).

**Expected results**:
- `total_restore_ms` drops from hundreds of ms to < 1ms.
- `total_snapshot_ms` stays similar (np.stack ≈ deepcopy).
- `net_impact_ms` should be positive for large datasets, approximately zero for small ones.

### Step 9: Change `step_cache_enabled` default to `True`

**Goal**: After validation, enable caching by default since it no longer has a negative cost.

**Change in `cache_config.py`**:
```python
step_cache_enabled: bool = True  # was False
```

**Change in `CacheConfig` docstring**: Update to reflect that caching is now the default.

### Step 10: Update D03 benchmark

**Goal**: The benchmark should report cache stats including the new timing breakdown, so users can see the net impact.

---

## 5. Additional Improvements (Post Phase A)

These are independent improvements that can be done after or alongside Phase A.

### 5.1 Reduce `_index_state_hash()` cost

`_index_state_hash()` calls `polars.hash_rows()` on the entire index DataFrame. For a 200-row dataset this is ~0.2ms, but for 5000+ rows it becomes noticeable.

**Optimization**: Cache the index state hash on the dataset (similar to `_content_hash_cache`). Invalidate it when the index is mutated (same invalidation points as content hash). This makes repeated lookups O(1).

**Implementation**: Add `_index_state_hash_cache` to `SpectroDataset.__init__()`. Invalidate in all methods that modify the indexer (`add_samples`, `exclude_samples`, `set_tag`, etc.).

### 5.2 Cache observability: time-breakdown stats

Even after Phase A, the `StepCache.stats()` output should include timing breakdown so that performance regressions can be detected.

**Fields** (from Step 1 instrumentation):
```python
"total_hash_ms": ...,
"total_snapshot_ms": ...,
"total_restore_ms": ...,
"total_transform_ms_saved": ...,
"net_impact_ms": ...,
```

Logged at `verbose >= 1` when `log_cache_stats=True` (existing setting).

### 5.3 Branch substep caching

The Codex document correctly notes that branch substeps bypass `_execute_single_step()` and miss the step cache. This is in `_execute_step_on_branches()` (executor.py line ~594) which calls `self.step_runner.execute()` directly.

**Fix**: Route branch substep execution through `_execute_single_step()` instead. This is a small change but requires care:
- Branch substeps already have correct `context` with `branch_path` set.
- The `selector_fingerprint` already includes `branch_path`, so cache keys would be branch-specific.
- Model steps within branches would still be non-cacheable (correct).

**Priority**: Low. Branch substep sharing is rare in practice — branches typically have different input data per branch, so cache hits would be unlikely anyway.

### 5.4 Skip `_is_step_cacheable()` parse overhead

`_is_step_cacheable()` calls `self.step_runner.parser.parse(step)` and `self.step_runner.router.route(parsed, step)` to check if the controller supports caching. This parse+route happens on every cacheable step, even though the result is deterministic for a given step config.

**Optimization**: Cache the cacheability result per step config hash. Since the step config doesn't change across variants, this lookup can be done once per unique step.

---

## 6. Correctness Guarantees (Data Leak Prevention)

### 6.1 Three-part cache key prevents fold leakage

The cache key is `(step_hash, data_hash, selector_fingerprint)`.

The `selector_fingerprint` is a SHA256 of: `partition`, `fold_id`, `processing`, `include_augmented`, `tag_filters`, `branch_path`.

**Guarantee**: Two steps with the same config and same input data but different folds will have different `selector_fingerprint` values → different cache keys → no cross-fold reuse → no data leakage.

This is unchanged by Phase A. CoW only changes the storage mechanism, not the keying.

### 6.2 Index-state hash prevents exclusion leakage

The `data_hash` includes `_index_state_hash()` which hashes the entire polars index DataFrame (partitions, groups, exclusions, tags, branches).

**Guarantee**: If step X is applied to 200 samples in variant A and 195 samples in variant B (5 excluded), the index hashes differ → cache miss → no reuse of incorrectly-fit transforms.

### 6.3 CoW isolation prevents cross-variant contamination

After restore, the live dataset shares array data with the cache entry via `SharedBlocks`. When the next step mutates the data:

1. `_prepare_for_mutation()` is called on the storage.
2. Storage is in shared mode → `_materialize_blocks()` is called.
3. `_shared.detach()` checks `refcount > 1` → creates independent copy.
4. Blocks are extracted from the copy.
5. The cache entry's SharedBlocks reference is unmodified.

**Guarantee**: Mutations on any variant never affect cached data or other variants.

### 6.4 What could go wrong (and why it won't)

| Risk | Why it's safe |
|------|---------------|
| Transform modifies cached array in-place | CoW detach() copies before mutation |
| Two variants restore same cache entry simultaneously | Each acquire() increments refcount; each detach() creates independent copy |
| Cache entry evicted while variant still uses restored data | The variant's storage holds its own acquired ref (refcount >= 1); eviction releases the cache's ref only |
| content_hash stale after restore | Explicitly set from cached state |
| Stochastic transform cached | `TransformerMixinController.supports_step_cache()` is the only one returning True; augmentation controllers return False |

---

## 7. Summary

| Aspect | Current | After Phase A |
|--------|---------|---------------|
| Cache hit cost | ~3-6ms (deepcopy) | ~0.01ms (ref acquire) |
| Cache miss cost | transform + ~3-6ms (deepcopy) | transform + ~3ms (np.stack) |
| Net impact (D03, cheap transforms) | **Negative** (cache is slower) | **Neutral to positive** |
| Net impact (expensive transforms) | Positive but reduced by overhead | **Positive, near-optimal** |
| Default enabled | No (`step_cache_enabled=False`) | **Yes** (`step_cache_enabled=True`) |
| Memory model | 2× per cache entry (independent copy) | Shared until mutation (CoW) |
| Correctness | Key includes fold, index, tags, branch | **Unchanged** |
| Infrastructure reuse | None | Reuses `SharedBlocks` from branch CoW |

The current step cache architecture (keying model, scope, observability, controller gating) is sound. The single change needed is replacing `copy.deepcopy()` with the existing `SharedBlocks` CoW mechanism in `_snapshot()` and `restore()`. This makes the cache a net positive instead of a net negative, enabling it by default.
