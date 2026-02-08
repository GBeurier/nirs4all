# Nirs4all Cache Reanalysis and Functional Data Cache Proposal (2026-02-07)

## 1. Current Caching Mechanism in nirs4all (High Level)

### 1.1 Runtime cache and memory layers

nirs4all currently uses several different mechanisms that are often conflated as "the cache":

1. Dataset load cache (`DatasetConfigs.cache` in `nirs4all/data/config.py`)
- Scope: raw loaded arrays per dataset config key.
- Goal: avoid repeated file I/O and parsing.

2. Block-based feature storage (`ArrayStorage`)
- Scope: inside each `FeatureSource`.
- Goal: avoid repeated 3D concatenation costs when adding processings.

3. CoW branch snapshots (`BranchController` + shared blocks)
- Scope: branch execution snapshots.
- Goal: reduce deep copies when branch paths are read-only.

4. Step cache (`StepCache` in `nirs4all/pipeline/execution/step_cache.py`)
- Scope: one in-memory LRU per dataset execution in `PipelineOrchestrator`.
- Goal: reuse post-step feature state across variant pipelines.

5. Transformer artifact check-before-fit (`TransformerMixinController`)
- Scope: fitted transformer artifacts in `ArtifactRegistry`.
- Goal: avoid refitting same transformer chain+data conditions.

These layers have different keying models and lifecycle. They are not one unified functional data cache today.

### 1.2 Configuration and lifecycle

`CacheConfig` fields (`nirs4all/config/cache_config.py`):

- `step_cache_enabled`
- `step_cache_max_mb`
- `step_cache_max_entries`
- `use_cow_snapshots`
- `log_cache_stats`
- `log_step_memory`
- `memory_warning_threshold_mb`

Lifecycle in current executor flow:

- `PipelineOrchestrator` creates one `StepCache` when `step_cache_enabled=True`.
- That same `StepCache` is reused across all generated pipeline variants for the current dataset.
- The cache is not persisted across datasets or separate `run()` calls.

So your expectation "shared across 200+ pipelines on same dataset" is partially true already, but only for paths that actually go through the step-cache wrapper and are declared cacheable.

### 1.3 What exactly is keyed today

Step cache lookup/store happens in `PipelineExecutor._execute_single_step`.

Current key is effectively:

`(step_hash, pre_step_data_hash, selector_fingerprint)`

Where:

- `step_hash`: hash of step config (`_step_cache_key_hash`).
- `pre_step_data_hash`: now `feature_content_hash + index_state_hash` (`_step_cache_data_hash`).
- `selector_fingerprint`: partition/fold/processing/include_augmented/tag_filters/branch_path.

Important consequence:

- This is a context-aware prefix-step cache, not a pure "dataset hash + full chain hash" cache.

### 1.4 Why some chains are not cached (exceptions)

Not security. Mostly correctness and execution-path boundaries.

1. Cacheability policy is controller-gated
- Base controller default: `supports_step_cache=False`.
- Only transform controller returns `True`.
- Models/splitters/branch/merge are intentionally not step-cached.

2. Branch internals bypass the executor step-cache wrapper
- Branch substeps execute via `runtime_context.step_runner.execute(...)` inside `BranchController`.
- Post-branch per-branch step execution (`_execute_step_on_branches`) also calls `step_runner.execute(...)` directly.
- Since step cache hooks live in `_execute_single_step`, these code paths do not hit step-cache logic.

3. Mode restriction
- Step cache currently applies in training mode only.

4. Subpipeline handling
- Previously weak for nested cases; now recursive cacheability check is in place (subpipeline cacheable only when all effective nested substeps are cacheable).

### 1.5 Why your measured speedup is low

Your logs are consistent with current design and workload shape.

1. Model training still runs for every variant
- Cache mostly helps preprocessing; model fit cost can dominate total wall time.
- If model dominates, global speedup stays near 1.0x.

2. Step-cache overhead is non-trivial
- On cache miss: deep-copy snapshot of feature sources.
- On cache hit: deep-copy restore into live dataset.
- These memory copies can be expensive relative to cheap transforms.

3. Hashing overhead per step
- `content_hash()` can re-hash feature arrays after mutations.
- `_index_state_hash()` hashes index rows to protect correctness.
- Safer keying improves correctness, but adds CPU overhead.

4. Branch/substep reuse is currently under-captured
- Because branch substeps bypass `_execute_single_step`, expected reuse opportunities are missed in branch-heavy graphs.

5. Benchmark shape can hide gains
- Running baseline, CoW, full-cache sequentially in same process affects memory state.
- `regression2` is invalid in your run (score `nan`), so that segment is not informative.

### 1.6 Pros and cons of current mechanism

Pros:

- Good correctness guardrails (selector + index-aware keying).
- Useful memory reduction via CoW snapshots.
- Practical reuse for shared transform prefixes in generator sweeps.
- Bounded memory via LRU limits.

Cons:

- Not a canonical functional data cache.
- Branch execution paths bypass the main step-cache wrapper.
- Deep-copy snapshot model reduces net speed gains.
- Cache layers (step cache vs transformer artifact cache) overlap but are not unified.

## 2. Functional Data Cache Concept

### 2.1 Principle

Your model is:

- If raw data identity is the same,
- and transform chain identity is the same,
- then transformed features are the same,
- regardless of sweep/branch/source context.

This is correct only if fit context is included when transforms are fit-dependent.

Functional cache identity should be:

`Key = H(raw_data_identity, transform_chain_identity, fit_context_identity)`

### 2.2 Fit context is the critical third dimension

For correctness, `fit_context_identity` must encode at least:

- partition/fold scope used for `fit()`
- include/exclude sample state
- tag-filter state
- branch subset semantics (if branch changes fit population)
- `fit_on_all` and similar options
- random_state/seed policy for stochastic transforms

Without this, functional cache can silently leak information across folds or branches.

### 2.3 Why functional cache is attractive for nirs4all

Pros:

- Matches your intended mental model for transformation-space exploration.
- Naturally enables reuse across sweeps and branch topologies.
- Can reduce repeated transform work dramatically on large transform search spaces.

Cons:

- Identity contract is stricter and more complex.
- Requires explicit treatment of fit-dependent and stochastic operators.
- Needs strong memory governance and observability.

### 2.4 How it can fit nirs4all architecture

High-level mapping:

- `DataNode`: immutable representation of a feature state.
- `TransformNode`: deterministic application of one transform to a parent `DataNode` under a given `fit_context`.
- Cache store maps `TransformNodeID -> DataNodeID`.

Branching then becomes graph composition over data nodes, not special-case snapshot logic.

## 3. Detailed Proposal for Functional Data Cache

### 3.1 Design goals

1. Preserve correctness first (no fold leakage).
2. Capture reuse across variant pipelines and branch paths.
3. Reduce copy-heavy restore overhead.
4. Keep bounded memory and predictable runtime.

### 3.2 Recommended architecture (hybrid)

Use a two-tier approach:

1. Tier A: keep current step cache as hot-path accelerator.
2. Tier B: add canonical functional data cache (data-node graph) as the source of truth.

In this model:

- Step-cache entries can store references/handles to canonical data nodes.
- Branch/subpipeline paths can resolve to same data node when identity matches.
- Existing infra can be migrated incrementally.

### 3.3 Key model (proposed)

`RawDataID = H(raw_feature_bytes, source_layout, metadata_needed_for_fit)`

`FitContextID = H(partition, fold, index_state, tag_filters, fit_on_all, random_state_policy)`

`ChainID = H(serialized_transform_chain, operator_params, operator_impl_version)`

`FunctionalKey = H(RawDataID, ChainID, FitContextID)`

Notes:

- `index_state` should remain explicit (current tightening is aligned with this).
- `operator_impl_version` can be lightweight (module+class+version hash) to avoid stale compatibility issues.

### 3.4 Variants

Variant 1: Strict functional cache (recommended baseline)
- Full fit-context identity always in key.
- Safe default for training and CV.

Variant 2: Relaxed transform-only mode (opt-in)
- Allow broader reuse for declared stateless or fit-on-all transforms.
- Faster but easier to misuse.

Variant 3: Branch-aware canonical nodes
- Same as Variant 1, plus branch execution refactor so branch substeps route through cache-aware execution path.

### 3.5 Branch-cache direction (what I would do after key tightening)

Short answer:

- Do not key by branch id itself.
- Key by effective fit/data context.

Why:

- Branch id is just topology metadata.
- If two branches have identical effective input and fit context, they should collide intentionally.
- If branch subsets differ, `index_state` and selector/fold context should force cache miss.

Practical step:

- Route branch substeps through the same cache-aware executor wrapper (or equivalent hook).
- Keep strict fit-context keying initially.

### 3.6 Expected gains and limits

Expected strong gains when:

- Large transform search spaces (many shared prefixes/chains).
- Expensive transforms relative to model fit.
- Repeated branch/sweep combinations on same data.

Expected weak gains when:

- Model fit dominates runtime.
- Transforms are cheap compared with hash/copy overhead.
- Pipeline has little transform overlap.

### 3.7 Bottlenecks, deadlocks, and runtime risks

Bottlenecks:

- Frequent full-array hashing and index hashing.
- Copy-heavy snapshot/restore path.
- Registry metadata growth in very large sweeps.

Deadlock risk (future parallel execution):

- Competing locks between cache registry and dataset mutation paths.

Mitigation:

- Immutable data-node handles + short critical sections.
- Fixed lock acquisition order (cache registry -> dataset state).
- Prefer lock-free reads and append-only metadata where possible.

Runtime risks:

- False hits if fit context omitted.
- Memory blow-up if node eviction policy is weak.
- Debugging complexity if cache observability is insufficient.

### 3.8 Edge cases (illustrated)

Case A: same raw data, same chain, different fold
- Must miss unless fit policy explicitly fold-agnostic.

Case B: same raw data, same chain, different excluded samples
- Must miss (index-state change).

Case C: stochastic transform without fixed seed
- Must be non-cacheable or key must include stochastic policy.

Case D: same branch topology, different subset definitions
- Must miss when effective fit population changes.

Case E: same transform chain after source merge vs before merge
- Must miss unless source-layout identity is equivalent.

### 3.9 Open questions

1. Should strict functional cache become default, with relaxed mode explicit only?
2. Which operators are formally declared deterministic/stateless in nirs4all?
3. Do we want optional on-disk spill for long runs, or strict in-memory only?
4. Can we unify step cache and transformer artifact cache into one node registry API?
5. Should prediction/explain mode use the same functional cache semantics?

### 3.10 Concrete next implementation steps

1. Keep current tightened keying as baseline (already aligned with strict context identity).
2. Add instrumentation to quantify cache overhead split:
- hash time
- snapshot/restore copy time
- transform compute time
3. Refactor branch substep execution to pass through cache-aware wrapper.
4. Introduce `FitContextFingerprint` utility and use one canonical implementation in both step cache and transformer artifact lookup.
5. Prototype functional node registry behind a feature flag and compare against current step cache on large transform sweeps.

---

## 4. Difference Between Your Model and Current Implementation

Your model:

- `same dataset hash + same transform chain hash => same transformed features`

Current implementation:

- `same single-step hash + same pre-step data/index hash + same selector fingerprint => reuse that step state`
- and only on specific execution paths.

So the main difference is granularity and scope:

- You propose canonical full-chain functional caching across all contexts.
- Current system is step-prefix, context-aware, and partially path-dependent.

Your direction is valid and can be implemented, but only with explicit fit-context identity and a branch-path integration refactor.
