# SpectroDataset Cache and Memory Overflow Investigation

## Overview of the Caching Mechanism in the Dataset

This review looked at the current `SpectroDataset`/pipeline code path to identify what is cached today and what is not.

### 1. Caches that currently exist

1. `SpectroDataset` content-hash cache (small metadata cache)
- `nirs4all/data/dataset.py:1481`
- `SpectroDataset` caches one string in `_content_hash_cache`.
- It is invalidated on feature mutation and reused in `content_hash()`.
- This cache does not cache arrays or transformed features.

2. Dataset loading cache (`DatasetConfigs.cache`)
- `nirs4all/data/config.py:178`
- Raw loaded tuples `(x_train, y_train, m_train, ...)` are cached by dataset name.
- This avoids re-reading/parsing files per pipeline variant.
- This is an input-loading cache, not a preprocessing-result cache.

3. Transformer fit cache via artifact registry (chain + data hash)
- Lookup path: `nirs4all/controllers/transforms/transformer.py:293`, `nirs4all/controllers/transforms/transformer.py:768`
- Keyed by chain path + `dataset.content_hash()`.
- Reuses already-fitted transformer artifacts and can skip repeated `fit()`.
- This is a cache for fitted transformer objects, not for transformed feature tensors.

4. Generic `DataCache` implementation
- `nirs4all/data/performance/cache.py:60`
- Provides size-limited in-memory cache infrastructure.
- In practice, it is currently used by `StepCache` backend, but not integrated into dataset loading or dataset tensor retrieval paths.

5. `StepCache` implementation (snapshot cache abstraction exists)
- `nirs4all/pipeline/execution/step_cache.py:54`
- Deep-copies and stores full dataset snapshots.
- However, execution wiring is currently missing: orchestrator instantiates it but does not pass/use it in executor step flow (`nirs4all/pipeline/execution/orchestrator.py:216`, `nirs4all/pipeline/execution/orchestrator.py:334`).

### 2. Memory-relevant behavior that is not cache-protected

1. Feature processings are eagerly materialized and retained in-memory
- New processings are added by concatenating the 3D array along processing axis (`nirs4all/data/_features/array_storage.py:205`).
- No eviction policy exists for old processings in `SpectroDataset`.

2. `Features(cache=True)` is declared but not implemented
- `nirs4all/data/features.py:21`
- Code comment explicitly says caching is "not yet implemented".

3. Branch execution snapshots full feature arrays
- `copy.deepcopy(dataset._features.sources)` is used for branch snapshots (`nirs4all/controllers/data/branch.py:1510`).
- Branch contexts keep these snapshots until merge; executor preserves branch metadata between steps (`nirs4all/pipeline/execution/executor.py:618`).

## Review of This Mechanism

### Strengths

1. Correctness guard exists for transformer cache reuse
- Cache keys use chain path + data hash, which is the right direction for correctness.

2. Raw data reload overhead is reduced
- `DatasetConfigs.cache` avoids repeated file parsing for the same dataset during variant sweeps.

3. Storage footprint for features is at least float32-based
- Core feature storage uses `float32` in `ArrayStorage`.

### Weaknesses, flaws, and anti-patterns

1. Main memory pressure is not managed by cache
- The dominant memory object is the in-memory feature tensor `(samples, processings, features)`.
- This tensor grows with every added preprocessing variant and is never evicted.

2. Repeated `np.concatenate` on growing tensors
- `add_processing()` allocates a new larger array each time (`nirs4all/data/_features/array_storage.py:220`).
- This creates avoidable copy amplification and high peak memory during growth.

3. Transform execution materializes multiple full copies in the same step
- Transformer controller requests both `all_data` and `fit_data` (`nirs4all/controllers/transforms/transformer.py:162`, `nirs4all/controllers/transforms/transformer.py:168`).
- Then it accumulates transformed arrays before writing (`nirs4all/controllers/transforms/transformer.py:327`).
- On large datasets, peak RAM can exceed steady-state array size by a large factor.

4. Branch snapshots are expensive
- Branch mode deep-copies full feature sources per branch (`nirs4all/controllers/data/branch.py:1512`).
- With many branches, memory scales approximately with `num_branches * dataset_feature_size`.

5. `feature_augmentation` default encourages multiplicative growth
- Code default is `action="add"` (`nirs4all/controllers/data/feature_augmentation.py:137`), which chains across existing processings and can explode processings count.
- This is high risk for large datasets and generator-heavy pipelines.

6. `StepCache` exists but is not wired
- Orchestrator creates `StepCache` (`nirs4all/pipeline/execution/orchestrator.py:216`) and logs stats (`nirs4all/pipeline/execution/orchestrator.py:334`), but execution does not read/write it.
- Current impact on runtime memory/compute is effectively zero.

7. `DataCache` LRU behavior is imperfect
- Eviction uses `(hit_count, timestamp)` (`nirs4all/data/performance/cache.py:281`), while access does not refresh timestamp.
- Behavior is closer to LFU/age hybrid than true LRU.

### Quick empirical probe (local)

Using project code in `.venv`:

1. Processing growth probe (`5000 x 1000`, float32)
- Base (`1` processing): ~`19.07 MB`
- After `13` processings total: ~`247.96 MB`
- Linear memory growth with processings count was confirmed.

2. Branch snapshot probe (`3000 x 8 x 700`, float32)
- Dataset feature tensor: ~`64.09 MB`
- `6` deep-copied branch snapshots: ~`384.52 MB` additional
- Combined resident size from base + snapshots: ~`448.61 MB`

These numbers are only small synthetic probes, but they confirm the current scaling pattern and overflow risk.

## Proposal to Manage the Problem of Memory Overflow

### Target outcome

Keep pipeline behavior functionally equivalent while bounding memory usage under large datasets + many preprocessing variants.

### Phase 1 (low-risk, immediate)

1. Wire `StepCache` into actual step execution or remove it temporarily
- If kept: add explicit pre-step lookup and post-step store at executor level for preprocessing steps only.
- If not kept yet: remove dead instantiation to avoid false confidence.

2. Add hard memory guardrails
- Add config knobs:
  - `max_feature_ram_mb`
  - `max_processings_per_source`
  - `overflow_policy` in `{error, warn, drop_oldest}`
- Estimate projected memory before adding processings:
  - `bytes = n_samples * n_processings * n_features * 4` per source.

3. Safer default for feature augmentation
- Change default action from `add` to `extend`, or keep `add` but emit a strong warning when dataset is large.

4. Add observability
- Log per step:
  - shape per source
  - processings count
  - estimated feature bytes
  - peak warning when above threshold.

### Phase 2 (core fix)

1. Introduce bounded processing cache with eviction
- New runtime component: `FeatureProcessingCache`.
- Key: `(dataset_hash, source_idx, processing_id_or_chain_hash, selector_scope)`.
- Value: feature array or reference to materialized block.
- LRU + byte-budget eviction.

2. Decouple "available processing" from "materialized in RAM"
- Keep processing metadata/index in dataset, but allow tensor payload to be:
  - in memory (hot),
  - memory-mapped on disk (warm),
  - unloaded (cold, recompute or load on demand).

3. Avoid repeated full-array concatenation
- Replace append-by-concatenate with chunk/block storage or preallocated growth strategy.
- At minimum, batch multiple new processings into one concatenation to reduce copy amplification.

4. Reduce branch snapshot size
- Replace full `deepcopy` snapshots with:
  - copy-on-write references, or
  - source/processings-level delta snapshots.
- Keep only latest needed snapshot per branch and release aggressively after merge.

### Phase 3 (optional, high leverage)

1. Add spill-to-disk policy for large intermediate arrays
- `np.memmap` or parquet/arrow blocks for intermediate preprocessings.
- Keep RAM for active branch/step only.

2. Cross-run cache key persistence activation
- Existing methods (`persist_cache_keys_to_store`, `load_cached_from_store`) are present but currently unused.
- Activate them only after memory-bounded local cache is stable.

### Proposed acceptance criteria

1. Stress scenario should not exceed configured memory budget
- Example: large dataset + generator producing many preprocessing variants.

2. No behavioral regression
- Same selected best pipeline and same predictions within tolerance.

3. Clear cache metrics in logs
- hit/miss/eviction + current cache MB.

4. Branch-heavy pipelines stay bounded
- Memory should scale sub-linearly with branch count after snapshot redesign.

### Recommended implementation order

1. Add guardrails + observability.
2. Make feature augmentation default safer (or warning gate).
3. Wire step cache properly for preprocessing-only checkpoints.
4. Replace deep-copy branch snapshots with lightweight snapshots.
5. Introduce bounded materialization cache and optional spill-to-disk.

## From-Scratch Proposition (Global Redesign)

This section assumes we remove current cache mechanisms and redesign caching/data execution from first principles.

### 1. Core idea

Do not cache by duplicating mutable `SpectroDataset` objects.

Instead:

1. Keep one immutable base dataset snapshot.
2. Represent each preprocessing result as a virtual view node (lineage metadata only).
3. Materialize arrays only on demand into a bounded multi-level cache (RAM first, disk second).
4. Reconstruct or reload evicted views using lineage when needed.

This gives:
- no recomputation for shared prefixes,
- bounded memory,
- no branch deep-copy explosion,
- deterministic reuse across many generated pipelines.

### 2. Design goals

1. Reuse computation aggressively for shared preprocessing prefixes.
2. Keep memory usage under a strict byte budget.
3. Avoid dataset-wide deep copies and repeated full-array concatenations.
4. Support branch/generator combinatorics without linear memory blow-up per branch.
5. Keep correctness strict: cache reuse only when input data and operator semantics are identical.
6. Allow transparent eviction and later reconstruction from lineage.

### 3. Data model from scratch

#### 3.1 Immutable dataset snapshot

`DatasetSnapshot` is the only "real" dataset payload owner.

Fields:
- `snapshot_id`: stable fingerprint of raw data payload + metadata schema.
- `sources`: immutable raw source arrays (memory-map friendly).
- `y`, `metadata`, `headers`, `units`.
- `sample_index_table`: immutable sample index with partition/flags.

No preprocessing variants are stored in this object.

#### 3.2 Sample-set identity

Selectors become stable objects:
- `SampleSet(selector_expr, include_augmented, include_excluded)`
- `sample_set_id = hash(sorted_sample_ids + selector options + snapshot_id)`

This makes transform fit/apply keys unambiguous across branches and generated pipelines.

#### 3.3 View graph (not dataset copies)

Each transformed dataset state is a `ViewNode`:
- `view_id`
- `parent_view_ids` (usually one, many for merge)
- `operator_fingerprint`
- `source_index`
- `sample_set_id` for apply phase
- `fit_set_id` for fit phase
- `output_schema` (shape/dtype/header lineage)
- `materialized_block_id` (optional, null if evicted/not materialized)

The full pipeline run is a DAG of view nodes, not a sequence of mutable dataset mutations.

#### 3.4 Fit state node

Separate fit state from transformed output:
- `FitNode` key:
  - operator fingerprint
  - parent view input identity
  - fit sample-set identity
  - source index
- value: serialized fitted operator artifact.

A transform output can reuse an existing `FitNode` and avoid re-fitting.

### 4. Cache layers

#### 4.1 L1: RAM block cache (strict budget)

Purpose:
- very fast reuse of recent arrays.

Properties:
- budget in bytes (for example 2 GB, configurable).
- weighted LRU admission/eviction by:
  - block size,
  - estimated recompute cost,
  - observed reuse count.
- no object graphs, only array blocks and compact metadata.

#### 4.2 L2: local disk block cache (spill layer)

Purpose:
- keep large intermediate results off RAM but quickly reloadable.

Properties:
- local workspace cache directory.
- chunked array format (zarr/ndarray+metadata) with compression.
- content-addressed by `block_id`.
- separate disk budget and GC policy.

#### 4.3 L3: persistent run cache (optional)

Purpose:
- cross-run reuse for identical dataset snapshot and lineage keys.

Properties:
- keyed by same deterministic IDs.
- references to persisted artifacts/blocks.
- strict compatibility checks using schema and operator ABI version.

### 5. Keying strategy (strict correctness)

#### 5.1 Operator fingerprint

`operator_fingerprint = hash(`  
`  operator_class_path,`  
`  canonical_params,`  
`  operator_code_version,`  
`  nirs4all_cache_schema_version`  
`)`

#### 5.2 Fit key

`fit_key = hash(`  
`  snapshot_id,`  
`  source_index,`  
`  parent_view_id,`  
`  fit_sample_set_id,`  
`  operator_fingerprint`  
`)`

#### 5.3 Transform output key

`transform_key = hash(`  
`  snapshot_id,`  
`  source_index,`  
`  parent_view_id,`  
`  apply_sample_set_id,`  
`  fit_key,`  
`  output_layout`  
`)`

Reuse is allowed only for exact key match.

### 6. Execution flow

#### 6.1 Compile once to execution DAG

Before execution:
1. Expand generator/branch steps into an explicit DAG.
2. Deduplicate equivalent subgraphs by key (common-prefix folding).
3. Topologically schedule nodes.

Important: two generated pipelines that share `raw -> SNV -> D1` point to the same view node IDs.

#### 6.2 Resolve each node with cache lookup

For each transform node:
1. Resolve/compute `fit_key`.
2. Lookup `FitNode` in L1, then L2/L3; fit only on miss.
3. Resolve `transform_key`.
4. Lookup transformed block in L1, then L2/L3; transform only on miss.
5. Register resulting block into cache manager with admission policy.

#### 6.3 No dataset deep copies for branching

Branch contexts carry:
- `branch_path`
- current `view_id` map per source
- selector state

No branch stores full copied arrays.
Only view references differ.

### 7. Memory strategy (why this solves overflow)

#### 7.1 Processing variants are virtual by default

Current behavior stores all processings in one growing 3D tensor.
New behavior stores:
- lineage metadata for each processing variant,
- materialized blocks only for hot/recent variants.

Old/cold variants are evicted from RAM and optionally kept on disk.

#### 7.2 Hard budgets

Introduce explicit budgets:
- `cache.ram_budget_mb`
- `cache.disk_budget_mb`
- `cache.max_entry_mb`
- `cache.max_inflight_compute_mb`
- `cache.max_materialized_views_per_source`

If a candidate block exceeds `max_entry_mb`, it can be "stream-only" (never admitted to L1).

#### 7.3 Copy amplification control

Replace repeated `np.concatenate` growth path with block-based writes:
- one processing variant equals one block.
- query-time concatenation for only requested variants.
- optional temporary concat cache for repeated consumer calls.

### 8. Policies per operator type

Not all steps should be cached equally.

1. Deterministic stateless transforms (for example fixed math ops)
- cache output blocks aggressively.

2. Deterministic fitted transforms (for example scalers)
- cache fit state + output blocks.

3. Stochastic transforms
- cache only if seeded and deterministic seed is in key.
- otherwise mark non-cacheable by default.

4. Sample augmentation
- usually high-volume and low-reuse.
- default: cache fit state only, avoid caching expanded output unless explicitly enabled.

5. Splitters and model steps
- keep model artifact cache separate from feature block cache.

### 9. API and runtime integration

#### 9.1 Replace mutable "add/replace features" semantics

Current dataset mutation methods should become view registration APIs:
- `register_view(parent_view, transform_spec, fit_set, apply_set) -> view_id`
- `materialize(view_id, layout, selector) -> array`

This is the key architectural shift.

#### 9.2 Context object changes

Execution context should carry:
- `current_view_ids_by_source`
- `current_sample_set_id`
- `branch_path`

not mutable processing name arrays tied to a monolithic tensor.

#### 9.3 Query path

`dataset.x(selector, layout, processings=...)`:
1. resolve requested view IDs,
2. ensure materialization (lookup/recompute),
3. concatenate only requested materialized blocks,
4. optionally cache the final query block with a short TTL.

### 10. Invalidation model

Invalidate by identity versioning, not ad-hoc flushes.

Keys include:
- `snapshot_id` (raw data identity),
- operator fingerprint with code/schema version.

Invalidation events:
1. Data file change -> new `snapshot_id`.
2. Operator code/params schema change -> new operator fingerprint.
3. Layout/selector changes -> distinct transform/query keys automatically.

No global "clear everything" required for correctness.

### 11. Observability and safeguards

Required metrics:
- `cache_l1_hit_rate`, `cache_l2_hit_rate`
- `bytes_ram_current`, `bytes_disk_current`
- `evictions_ram`, `evictions_disk`
- `recompute_count`
- `fit_reuse_count`
- `materialization_latency_ms`
- `peak_inflight_mb`

Required logs per step:
- input view ids,
- cache hit/miss outcome,
- bytes admitted/evicted,
- recompute reason if miss.

Runtime safeguards:
- high-watermark actions (`warn`, `throttle`, `hard_fail`).
- deny admission for huge low-value blocks.

### 12. Practical rollout plan

#### Phase A: introduce view graph without changing user API

1. Internally map current APIs to view registration/materialization.
2. Keep existing public signatures stable.
3. Add L1 RAM cache only.

#### Phase B: add disk spill and eviction pressure tests

1. Add L2 block store.
2. Add stress suites:
  - large dataset + 100s of generated preprocessing variants,
  - branch-heavy pipelines,
  - augmentation-heavy pipelines.

#### Phase C: enable persistent L3 and cross-run reuse

1. Persist fit nodes + selected transform blocks.
2. Enforce strict compatibility gates via versioned fingerprints.

### 13. Why this is better than step-level dataset snapshot caching

Step-level full-dataset snapshots are logically simple but expensive:
- they duplicate entire dataset state,
- they interact poorly with branch multiplicity,
- they still keep mutation semantics.

View-graph + block materialization gives equivalent logical reuse while:
- minimizing duplication,
- making eviction cheap and safe,
- scaling to high combinatorics.

### 14. Minimal "from scratch" MVP recommendation

If we implement only one robust system:

1. Immutable `DatasetSnapshot`.
2. View DAG with deterministic keys.
3. L1 RAM block cache with strict byte budget.
4. L2 disk spill cache.
5. No dataset deep-copy in branch path.

This MVP already solves the dominant overflow failure mode and gives a clean base for future cross-run caching.
