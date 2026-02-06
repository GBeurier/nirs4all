# Caching Analysis for nirs4all

## Context

This document analyzes the current state of caching in the nirs4all library and identifies opportunities for a caching architecture that would benefit the upcoming "refit-after-CV" feature. In that feature, after cross-validation selects the best pipeline configuration, the winning configuration is retrained on ALL training data. Caching preprocessing transforms and predictions avoids redundant computation during this second pass.

---

## 1. Current State of Caching

### 1.1 Chain-Based Preprocessing Identification (Partial Caching Infrastructure)

**Location**: `nirs4all/pipeline/storage/artifacts/artifact_registry.py`, `nirs4all/pipeline/storage/artifacts/operator_chain.py`

The OperatorChain system and the ArtifactRegistry already provide most of the infrastructure needed for computation-level caching of preprocessing steps. This is the most important existing foundation.

**How it works**:

1. Every preprocessing step produces an `OperatorChain` path like `s1.MinMaxScaler>s3.SNV[br=0]`, deterministically identifying the chain of operations that produced a given state.
2. The `ArtifactRegistry` is a **single instance shared across all pipeline variants within a dataset** (created in `PipelineOrchestrator._execute_dataset()`, reused across all pipeline variants in the generator sweep loop).
3. The registry maintains an in-memory `_by_chain_path` dict that maps chain paths to artifact IDs. As pipeline variant #1 registers fitted transformers, their chain paths are indexed.
4. The `get_by_chain()` and `get_chain_prefix()` methods allow lookup of previously registered artifacts by chain path.

**What this means for caching**: When 100 pipeline variants share the same first 3 preprocessing steps (e.g., a generator sweep varying only model parameters), the chain paths for those shared steps are identical across all variants. The fitted artifacts from variant #1 are indexed in `_by_chain_path` and remain accessible to variant #100 during the same run. **The infrastructure to detect that a preprocessing chain has already been computed exists today.**

**What is missing**: The transformer controller (`TransformerMixinController.execute()`) does **not** consult the registry before fitting. It always clones the operator and fits from scratch. The registry is only consulted during **prediction mode** (via `artifact_provider`) to load pre-fitted artifacts. Adding a cache check before fitting — "has this exact chain path already been fitted on the same data?" — would turn the existing infrastructure into a computation cache with relatively low implementation effort.

**Key insight**: If all chain-indexed artifacts are retained in memory until the dataset changes (which is already the case — the registry lives for the duration of one dataset's pipeline sweep), then any pipeline variant can benefit from preprocessing already computed by a prior variant. This is not hypothetical infrastructure — the registry, the chain indexing, and the artifact loading mechanisms are all in place. The missing piece is the check-before-fit logic in the controller, plus data identity verification (see section 1.10).

### 1.2 Artifact Content-Addressed Storage (Binary Deduplication)

**Location**: `nirs4all/pipeline/storage/artifacts/artifact_registry.py`, `nirs4all/pipeline/storage/workspace_store.py`

This is the most mature caching/deduplication mechanism in the codebase. It operates at the binary artifact level.

**How it works**:

1. When a fitted model or transformer is persisted, the binary content (joblib or pickle serialization) is SHA256-hashed.
2. The `ArtifactRegistry._find_existing_by_hash()` method checks both an in-memory dict (`_by_content_hash`) and the filesystem (shard directories) for an existing file with the same hash.
3. If found, the existing file path is reused -- no new file is written. A log message "Deduplication: reusing {path} for {artifact_id}" is emitted.
4. In the `WorkspaceStore.save_artifact()` method, the same pattern applies: the `GET_ARTIFACT_BY_HASH` query checks the DuckDB `artifacts` table, and if a match is found, the `ref_count` is incremented instead of writing a new file.

**What this deduplicates**: Identical binary objects across different pipelines, folds, or runs. If two PLS models happen to produce byte-identical serialized output, only one file is stored. In practice this is uncommon for models (trained on different folds) but can occur for transformers fitted on the same training data.

**What this does NOT do**:
- It does not skip the fitting/training step. The transformer is still cloned and fitted. Only the persistence is deduplicated.
- It does not detect that the same operator with the same parameters was fitted on the same data. The deduplication is purely content-based (post-serialization).
- It is not a computation cache -- it is a storage deduplication layer.

**Relevant code paths**:
- `ArtifactRegistry.register()` (line ~360-365 in `artifact_registry.py`): content hash check
- `ArtifactRegistry.register_with_chain()` (line ~358-366): same pattern
- `WorkspaceStore.save_artifact()` (line ~660-682): DuckDB-level deduplication with `ref_count`
- `artifacts` table schema (in `store_schema.py`): has `content_hash VARCHAR NOT NULL` and `ref_count INTEGER DEFAULT 1`
- Index: `CREATE INDEX IF NOT EXISTS idx_artifacts_content_hash ON artifacts(content_hash)`

### 1.3 Dataset Loading Cache (`DataCache`)

**Location**: `nirs4all/data/performance/cache.py`

A general-purpose LRU cache for loaded data, designed for dataset file loading.

**How it works**:
- `DataCache` is a thread-safe in-memory LRU cache with configurable size limits (default 500 MB), max entries (100), and optional TTL.
- `CacheEntry` tracks source file mtime for staleness detection.
- `make_cache_key()` creates an MD5 hash from file path + loading parameters.
- A global singleton is available via `cache_manager()`.
- `get_or_load(key, loader, source_path)` provides the get-or-compute pattern.

**Current usage**: The `DataCache` class is defined and exported via `nirs4all.data.performance`, but **no call sites were found in the actual data loaders** (csv_loader.py, excel_loader.py, etc.). The `cache_manager()` singleton exists but is not invoked during pipeline execution. This appears to be infrastructure built for Phase 8 (Performance Optimization) but not yet integrated.

**What it could cache**: Raw file I/O results (loaded DataFrames, numpy arrays from CSV/Excel/Parquet). Not useful for the refit scenario directly since data is already in memory.

### 1.4 Lazy Loading (`LazyArray`, `LazyDataset`)

**Location**: `nirs4all/data/performance/lazy_loader.py`

Provides deferred loading for large arrays. `LazyArray` wraps a loader function and only materializes data on first access. `LazyDataset` wraps X, y, and metadata as lazy components.

**Current usage**: Like `DataCache`, this is defined but not widely integrated into the main pipeline execution path. It is infrastructure for future optimization.

### 1.5 Visualization Prediction Cache (`PredictionCache`)

**Location**: `nirs4all/visualization/prediction_cache.py`, used by `nirs4all/visualization/predictions.py`

An LRU cache specifically for aggregated prediction query results in the visualization layer.

**How it works**:
- `CacheKey` encodes query parameters (aggregate, rank_metric, partitions, group_by, filters).
- `PredictionCache.get_or_compute(key, compute_fn)` returns cached aggregation results.
- Used by `PredictionAnalyzer` (in `visualization/predictions.py`) to avoid recomputing expensive aggregations when multiple charts use the same query.

**Relevance to refit**: None. This is a UI/visualization layer cache, not a computation cache.

### 1.6 Backend Availability Cache

**Location**: `nirs4all/utils/backend.py`

A simple dict-based cache (`_availability_cache`, `_gpu_cache`) for backend availability checks (TensorFlow, PyTorch, JAX). Avoids repeated import probes.

**Relevance to refit**: None.

### 1.7 DatasetConfigs Loading Cache

**Location**: `nirs4all/data/config.py` (line 178, 368-390)

`DatasetConfigs` has a `self.cache: Dict[str, Any]` that stores loaded data tuples `(x_train, y_train, m_train, ..., x_test, y_test, m_test, ...)` keyed by dataset name. This avoids re-parsing files when the same dataset is used by multiple pipelines in a single run.

**How it works**: On first load, the parsed data tuple is stored in `self.cache[name]`. On subsequent pipeline executions with the same dataset name, the cached tuple is returned directly.

**Relevance to refit**: Moderate. During refit, the dataset is already loaded and available. This cache prevents re-parsing but does not prevent re-fitting of preprocessing.

### 1.8 Dataset Content Hash

**Location**: `nirs4all/data/dataset.py` (lines 1866-1895, 1972-1979)

`SpectroDataset` has `set_content_hash()` and `metadata()` that computes or stores a content hash. The `metadata()` method computes a quick MD5 from a sample of feature data (first 100 rows, flattened). The `dataset_hash` is passed to `store.begin_pipeline()` for run-compatibility tracking.

**How it works**: A 12-char MD5 truncated hash is computed from a data sample. This is used for provenance tracking (detecting when a dataset has changed between runs), not for caching.

**Relevance to refit**: The dataset hash could be a component of a cache key for step-level caching. If the dataset hash has not changed, preprocessing results computed during the CV pass are still valid for the refit pass.

### 1.9 Pipeline Configuration Hash

**Location**: `nirs4all/pipeline/config/pipeline_config.py` (line 294-303), `nirs4all/pipeline/execution/executor.py` (line 1073-1083)

Pipeline steps are hashed using MD5 for identification:
- `PipelineConfigs.get_hash(steps)`: serializes steps to JSON and MD5-hashes them, producing an 8-char hash used in pipeline names.
- `PipelineExecutor._compute_pipeline_hash()`: similar 6-char MD5 hash used as `dataset_hash` parameter (misleadingly named -- it hashes the pipeline config, not the dataset).

**Relevance to refit**: The pipeline config hash identifies WHICH pipeline configuration was used. Combined with a dataset hash, it could form a cache key for "was this exact pipeline already executed on this exact data?"

### 1.10 Data Identity in Artifact and Chain IDs (Gap)

**This is a critical architectural gap that must be addressed before implementing computation-level caching.**

The V3 artifact ID format is `{pipeline_id}${chain_hash}:{fold_id}`. The chain hash is derived from the `OperatorChain` path (e.g., `s1.MinMaxScaler>s3.SNV[br=0]`), which encodes **only the operator sequence and structural context** (branch indices, source indices, fold ID). It does **not** include any data identity component.

**What is in a chain path**: operator class names, step indices, branch paths, source indices, substep indices.

**What is NOT in a chain path**: dataset name, dataset content hash, sample indices, partition identity (train/test/val), fold split configuration.

**Current safety within a single run**: Within one run on one dataset, all pipeline variants share the same `ArtifactRegistry` instance and operate on the same dataset. Different `pipeline_id` values (UUIDs from `WorkspaceStore.begin_pipeline()`) prevent ID collisions between variants. The registry is recreated per dataset in the orchestrator loop, so artifacts from dataset A cannot leak into dataset B's registry. This means the current system is **safe by construction within a single run** — but only because data isolation is implicit (same registry = same dataset), not explicit (no data hash in IDs).

**Risk for multi-folder datasets**: When a dataset is configured from multiple folders, each folder may contribute different sample partitions. The `source_index` in `OperatorNode` tracks multi-source processing, but this refers to the multi-source index within a single `SpectroDataset`, not to the folder origin. Two datasets loaded from different folder configurations but sharing the same operator chain would produce **identical chain paths** despite being fitted on different data.

**Risk for caching**: If computation-level caching is implemented using chain paths as cache keys (as proposed in section 5), a cache hit based solely on chain path would be incorrect when:
- Two different datasets happen to use the same preprocessing chain
- The same dataset is reloaded with different samples (e.g., updated folder content)
- Different fold configurations produce different training subsets for shared (pre-splitter) steps

**Requirement**: Any cache key used for computation-level caching **must** include a data identity component alongside the chain path. The existing `SpectroDataset.metadata()` content hash (section 1.8) provides this — it hashes the actual feature data. The cache key should be `(chain_path_hash, input_data_content_hash)` to guarantee that a cached result is only reused when both the operator chain AND the input data match.

**Recommendation for future cross-run caching**: If artifacts are ever reused across runs (e.g., DuckDB-backed persistent cache), the data hash must be stored alongside the artifact record. The current `pipelines` table already stores `dataset_hash` per pipeline, but this is not propagated to individual artifact records. Extending `ArtifactRecord` with an `input_data_hash` field would close this gap.

---

## 2. What Is NOT Cached (Gaps)

### 2.1 Transform Fitting

**No caching exists.** Every time a pipeline step runs a transformer (e.g., `MinMaxScaler`, `SNV`, `SavitzkyGolay`), the transformer is:
1. Cloned from the template via `sklearn.base.clone(op)`
2. Fitted on the training data via `transformer.fit(fit_2d)` (or `fit(fit_2d, wavelengths=...)`)
3. Applied to all data via `transformer.transform(all_2d)`

(Source: `TransformerMixinController.execute()` in `controllers/transforms/transformer.py`, lines 291-301)

During refit-after-CV, the preprocessing chain (e.g., MinMaxScaler -> SNV -> SavitzkyGolay) for the winning pipeline has already been fitted during the CV pass. The fitted transformer artifacts are persisted to disk. But the current execution path has no mechanism to say "I already have a fitted MinMaxScaler for this exact data configuration -- reuse it instead of fitting again."

The retrainer (`pipeline/retrainer.py`) does support a `TRANSFER` mode where preprocessing artifacts from a previous run are reused in predict mode (transform-only, no fit). But this is designed for cross-dataset transfer, not for refit-on-same-data optimization.

### 2.2 Model Training

**No caching exists.** Models are trained from scratch on every execution. The `BaseModelController` trains K fold models during CV, and these fold model artifacts are persisted. But there is no mechanism to detect that the refit is requesting the same model class with the same parameters on a superset of the same data.

This is expected -- the refit model trains on ALL training data (not fold subsets), so it genuinely is a new computation. However, there may be opportunities for warm-starting (using a fold model's weights as initialization for the full-data refit).

### 2.3 Transform Output Caching (Intermediate Data)

**No caching exists.** When a pipeline has 5 preprocessing steps, the intermediate data arrays at each step are not saved anywhere. During refit, steps 1-4 (preprocessing) would need to re-execute even though they produce the same output (fitted on the same training data, transforming the same input).

The `SpectroDataset` stores transformed features via `dataset.replace_features()` or `dataset.add_features()`, but these are in-memory and ephemeral -- they exist only during one pipeline execution and are not persisted between the CV pass and the refit pass.

### 2.4 Step-Level Result Caching

**No caching exists.** There is no mechanism to say "step 3 with input hash X and operator config Y has already been computed; skip it and use the cached output." The pipeline executor (`PipelineExecutor._execute_steps()`) always executes every step sequentially without any skip-if-cached logic.

The `StepExecutionMode.SKIP` enum exists in the trace system, but it is only used for generator-produced None steps (e.g., `_or_: [None, SNV()]`), not for cache-based skipping.

### 2.5 Prediction Deduplication

**No deduplication exists** in the `Predictions` class. Every model evaluation produces a new prediction record. If the same model with the same parameters is evaluated on the same data twice, two identical prediction records are stored. The DuckDB `predictions` table has no unique constraint that would prevent this.

### 2.6 Cross-Pipeline Transform Reuse

When multiple pipeline configurations share a common prefix (e.g., both start with MinMaxScaler -> SNV but differ in the model), the shared preprocessing steps are executed independently for each pipeline. The `ArtifactRegistry` is shared across variants and indexes fitted artifacts by chain path (see section 1.1), so the infrastructure to detect shared prefixes exists. However, the transformer controller always fits from scratch — it does not consult the registry before fitting. The missing piece is a check-before-fit path in the controller that loads the existing artifact and skips fitting when the chain path and data hash match.

---

## 3. Opportunities for the Refit-After-CV Feature

### 3.1 The Refit Scenario

After CV completes:
1. The best pipeline configuration is identified (e.g., MinMaxScaler -> SNV -> PLS(n_components=10))
2. That exact configuration needs to be retrained on ALL training data (no fold splitting)
3. The preprocessing steps (MinMaxScaler, SNV) need to be refitted on the full training set
4. The model (PLS) needs to be retrained on the full preprocessed training data
5. The final model is evaluated on the held-out test set

### 3.2 What Can Be Cached for Refit

#### 3.2.1 Fitted Transformers (High Impact)

During CV, transformers are fitted on (K-1)/K of the training data per fold. During refit, they must be fitted on ALL training data. **These are NOT the same fits** -- the training data subset is different (full set vs. fold subset). Therefore, fitted transformer caching from CV is not directly applicable to refit.

However, there is a subtlety: if the pipeline uses `fit_on_all: True` for unsupervised preprocessing steps (which fit on ALL data regardless of fold), then the fitted state IS identical between CV and refit. These transformers could be reused.

**Opportunity**: For `fit_on_all: True` steps, detect that the transformer has already been fitted on the same data and reuse the artifact directly. The cache key would be: `(operator_class, operator_params_hash, data_hash, "fit_on_all")`.

#### 3.2.2 Transform Outputs (Medium Impact)

Even when transformers must be refitted (on full training data), if the refit uses the same preprocessor chain, the **transform output for test data** could potentially be cached. During CV, the test set is transformed using fold-fitted transformers. During refit, it is transformed using the full-train-fitted transformer. These produce different outputs unless the transformer is stateless (e.g., Detrend, first derivative with fixed window).

**Opportunity**: For stateless transformers (those that don't learn from data, like fixed-parameter spectral derivatives), the transform output is identical regardless of what data was used for fitting. These transforms could be cached.

#### 3.2.3 Cross-Pipeline Preprocessing (High Impact for Generator Sweeps)

When a generator sweep produces 100 pipeline variants that share the same first 3 preprocessing steps but vary in model parameters, the preprocessing is currently repeated 100 times. Caching the preprocessed dataset at the branch point would eliminate this redundancy.

**Opportunity**: Cache the preprocessed dataset state after each step, keyed by `(step_chain_hash, input_data_hash)`. When a subsequent pipeline encounters the same step chain with the same input, skip preprocessing and use the cached output.

This is the highest-impact optimization for both the current architecture (generator sweeps) and the refit feature (refit replays the same preprocessing chain).

### 3.3 What Should NOT Be Cached for Refit

- **Model training**: The refit model trains on different data (full training set vs. fold subsets), so fold models cannot be reused. The refit produces a genuinely new model.
- **CV predictions**: These are the output of fold models on fold validation sets. They are not relevant to the refitted model's evaluation.
- **Fold split indices**: The refit does not use fold splitting, so splitter artifacts are not reusable.

---

## 4. Existing Infrastructure That Enables Caching

### 4.1 Operator Chain (V3 Artifact System)

The `OperatorChain` and `OperatorNode` system in `pipeline/storage/artifacts/operator_chain.py` provides deterministic identification of execution paths. Every artifact's identity is a chain path like `s1.MinMaxScaler>s3.SNV[br=0]>s4.PLS[br=0]`, which is SHA256-hashed to produce an artifact ID.

**Why this matters for caching**: The chain path is a natural cache key for step-level caching. If step N's chain path is the same as a previously computed step, and the input data hash matches, the output can be reused. As described in section 1.1, the `ArtifactRegistry` already indexes fitted artifacts by chain path and is shared across pipeline variants within a dataset run. The chain path alone is not sufficient as a cache key — it must be combined with a data content hash (see section 1.10) — but it provides the operator-identity half of a complete cache key.

### 4.2 Execution Trace

The `ExecutionTrace` system (`pipeline/trace/execution_trace.py`) records every step's execution with:
- `step_index`, `operator_class`, `operator_config`
- Input/output shapes
- Artifact IDs produced
- Branch and source context

**Why this matters for caching**: The trace from the CV pass contains the complete record of what was computed. A refit pass could consult the CV trace to identify which steps produced identical results and which need recomputation.

### 4.3 Chain Builder / Chain Replay

The `ChainBuilder` (`pipeline/storage/chain_builder.py`) converts traces to chain records. The `WorkspaceStore.replay_chain()` method loads artifacts from a stored chain and applies them to new data. This is the prediction-mode path.

**Why this matters for caching**: The replay infrastructure already implements "load fitted artifacts and apply them." This could be extended to a "load fitted artifacts, check if data matches, skip fitting" pattern.

### 4.4 Dataset Content Hash

`SpectroDataset.metadata()` computes a content hash from feature data. This could serve as the data component of a step-level cache key.

---

## 5. Proposed Caching Architecture

### 5.1 Step-Level Transform Cache

**Goal**: Skip refitting of transformers when the same operator with the same parameters was already fitted on the same data.

**Leverages existing infrastructure**: The `ArtifactRegistry._by_chain_path` dict (section 1.1) already indexes fitted artifacts by chain path and is shared across pipeline variants within a dataset run. The proposed cache adds a data hash verification layer on top of this existing index rather than building a parallel cache.

**Cache key**: `(chain_path_hash, input_data_content_hash)` — the chain path hash encodes the full operator identity (class, parameters, position in the chain, branch/source context), so `operator_class` and `operator_params_hash` are redundant. The `input_data_content_hash` is **mandatory** for correctness, especially when multiple folders produce different sample partitions (section 1.10).

**Cache value**: `(fitted_transformer_artifact_id, output_data_hash, output_shape)`

**Storage**: In-memory dict during a single run session. The `_by_chain_path` dict already provides the chain-to-artifact mapping; extending it (or adding a parallel dict) with data hash verification is straightforward. Optionally backed by the DuckDB store for cross-run caching (requires adding `input_data_hash` to the `ArtifactRecord`).

**Integration point**: `TransformerMixinController.execute()` — before `transformer.fit(fit_2d)`, compute the input data hash, then look up `(chain_path, data_hash)` in the cache. If hit, load the fitted transformer artifact via `artifact_registry.get_by_chain()` and skip to `transformer.transform(all_2d)`. If miss, fit as usual and register in the cache.

**Cache invalidation**: Cache entries are valid only when:
- The input data hash matches (same training data subset)
- The operator class and parameters are identical
- The chain path up to this point is identical (same preprocessing history)

**Hashing cost consideration**: Computing MD5 or xxhash of numpy arrays is fast (gigabytes/second on modern CPUs), much cheaper than fitting transformers. For a 1000x2000 float64 matrix (~16MB), hashing takes ~5ms while fitting a scaler or SNV takes 10-100ms. For expensive transforms (CARS, MCUVE, SavitzkyGolay with optimization), the savings are much larger.

### 5.2 Preprocessed Data Snapshot Cache

**Goal**: For generator sweeps where multiple pipelines share a preprocessing prefix, cache the entire preprocessed dataset state after each step.

**Cache key**: `(step_chain_hash, dataset_content_hash)` -- where `step_chain_hash` is the hash of the chain path up to and including this step.

**Cache value**: A snapshot of the `SpectroDataset` features state (numpy arrays per source/processing), stored in memory.

**Integration point**: `PipelineExecutor._execute_single_step()` -- after step execution, store the dataset snapshot. Before step execution, check if a snapshot exists for this cache key.

**Memory consideration**: Spectral datasets are typically 16-64 MB per snapshot. With 5 preprocessing steps and 2 sources, that is 160-640 MB of cache. This is manageable with an LRU eviction policy (reuse the `DataCache` class).

### 5.3 Refit-Specific Optimizations

For the refit-after-CV two-pass architecture specifically:

1. **Pass 1 (CV)**: Execute normally. Record all step traces, artifact IDs, and data hashes.
2. **Between passes**: Identify the winning pipeline configuration. Compute which steps need full-data refitting vs. which can be skipped.
3. **Pass 2 (Refit)**: For each step:
   - If the step uses `fit_on_all: True`: load the CV artifact directly (same fit).
   - If the step is stateless (no learned state): reuse the CV artifact.
   - If the step fits on training data only: refit on the full training set (cannot skip).
   - For the model step: always train from scratch on full training data.

This requires extending the `StepExecutionMode` enum or the retrainer's `StepMode` to include a "reuse_if_same_data" mode.

### 5.4 Implementation Priorities

| Priority | Component | Impact | Complexity | Description |
|----------|-----------|--------|------------|-------------|
| P0 | Check-before-fit in transformer controller | Critical | Low | Consult `ArtifactRegistry._by_chain_path` + data hash before fitting; load existing artifact if match. Most infrastructure already exists (section 1.1). |
| P0 | Data identity in cache keys | Critical | Low | Add `input_data_content_hash` to all cache lookups. Required for correctness with multi-folder datasets (section 1.10). |
| P1 | Refit step-mode logic | Critical | Low | Determine which steps need refitting vs. reuse during refit |
| P2 | Preprocessed data snapshot cache | High | Medium | Cache dataset state between pipelines sharing a prefix |
| P3 | Transform artifact reuse for `fit_on_all` | Medium | Low | Skip refitting for unsupervised preprocessing steps |
| P4 | Step-level transform cache | Medium | Medium | General-purpose transform output caching |
| P5 | Cross-run cache (DuckDB-backed) | Low | High | Persist step caches across runs for repeated experiments |

---

## 6. Implementation Considerations

### 6.1 Cache Invalidation

Cache entries must be invalidated when:
- The input data changes (different samples, different preprocessing upstream)
- The operator parameters change
- The operator version changes (library updates)
- The pipeline structure changes (different step order)

The `OperatorChain` hash naturally captures most of these: if any upstream step changes, the chain hash changes, invalidating all downstream cache entries.

### 6.2 Memory vs. Disk Trade-offs

| Storage | Latency | Capacity | Lifetime | Use Case |
|---------|---------|----------|----------|----------|
| In-memory dict | ~0ms | ~2 GB | Single run | Step-level caching within a run |
| `DataCache` (existing) | ~1ms | Configurable (500 MB default) | Single session | Dataset loading, preprocessed snapshots |
| DuckDB store | ~5ms | Unlimited | Permanent | Cross-run artifact deduplication |
| Filesystem | ~10ms | Unlimited | Permanent | Binary artifact content-addressed storage |

For the refit feature, in-memory caching (within the same `PipelineOrchestrator.execute()` call) is sufficient and simplest to implement. The CV pass and refit pass happen within the same session.

### 6.3 Hash Computation Cost

| Data Size | MD5 Time | xxhash Time | Transform Fit Time (typical) |
|-----------|----------|-------------|------------------------------|
| 1000x500 (4 MB) | ~1ms | ~0.3ms | 5-50ms |
| 1000x2000 (16 MB) | ~5ms | ~1ms | 10-200ms |
| 10000x2000 (160 MB) | ~50ms | ~10ms | 100-2000ms |

Hashing overhead is negligible compared to transform fitting for all but the smallest/simplest transforms.

### 6.4 Correctness Risks

- **Stale cache**: If the data changes between CV and refit (should not happen in normal flow, but defensive checks are needed).
- **Parameter mutation**: Some sklearn transformers mutate internal state. Cache keys must be computed from the original step config, not the fitted object.
- **Non-deterministic transforms**: Some transforms (e.g., with random initialization) produce different results across fits. These should not be cached unless deterministic seeding is enforced.
- **Feature augmentation**: Steps with `add_feature=True` mode modify the dataset's feature set. Cached snapshots must capture the full feature state, including processing names and headers.

### 6.5 Interaction with Existing Artifact System

The proposed caching layer is complementary to the existing artifact system, not a replacement:

| System | Purpose | Scope |
|--------|---------|-------|
| `ArtifactRegistry._by_chain_path` (existing) | Index fitted artifacts by chain path | Cross-pipeline within a dataset run |
| `ArtifactRegistry` (content-addressed) | Deduplicate identical binary files | Cross-pipeline within a run |
| `WorkspaceStore.save_artifact()` (ref counting) | Persistent artifact storage with GC | Cross-run |
| **Proposed: Check-before-fit** | Skip redundant fitting using existing chain index + data hash | Within a dataset run |
| **Proposed: Snapshot Cache** | Skip redundant preprocessing chains | Within a run session |

The check-before-fit approach builds directly on the existing `_by_chain_path` index, adding only data hash verification and a controller-level cache check. The snapshot cache is a higher-level optimization that caches the full `SpectroDataset` state. Both reduce the number of artifacts produced (fewer fits = fewer artifacts to persist), naturally reducing storage I/O.

---

## 7. Summary

### What Exists Today

| Mechanism | Type | Scope | Used? |
|-----------|------|-------|-------|
| Chain-based preprocessing identification | Partial caching infra | Cross-pipeline within a dataset | Yes (indexing), No (computation skipping) |
| Artifact content-addressed dedup | Binary storage dedup | Cross-pipeline | Yes, actively |
| `DataCache` (LRU, in-memory) | Data loading cache | Session | No (defined but not integrated) |
| `LazyArray`/`LazyDataset` | Deferred loading | Session | No (defined but not integrated) |
| `PredictionCache` | Aggregation cache | Visualization | Yes, in `PredictionAnalyzer` |
| Backend availability cache | Import probe cache | Process | Yes, in `utils/backend.py` |
| `DatasetConfigs.cache` | Parsed data tuple cache | Single run | Yes, prevents re-parsing |
| Dataset content hash | Provenance tracking | Metadata | Yes, for run compatibility |
| Pipeline config hash | Pipeline identification | Naming | Yes, for pipeline naming |

### What Is Missing for Refit

| Gap | Impact for Refit | Difficulty |
|-----|------------------|------------|
| No check-before-fit in transformer controller | Shared preprocessing infra exists but is not consulted before fitting | Low |
| No data identity in chain/artifact IDs | Cache keys lack data hash; unsafe for multi-folder/multi-dataset reuse | Low |
| No transform fitting cache | Refit re-fits all preprocessing from scratch | Medium |
| No preprocessed data snapshots | Shared preprocessing prefix re-executed per pipeline | Medium |
| No step-level skip logic | Cannot skip steps that produce identical results | Low |
| No `fit_on_all` artifact reuse detection | Unsupervised steps unnecessarily refitted | Low |
| No stateless transform detection | Fixed-parameter transforms unnecessarily refitted | Low |
| `DataCache` not integrated into loaders | Separate concern, not blocking refit | Low |

### Key Recommendations

1. **Leverage the existing chain infrastructure for computation skipping**: The `ArtifactRegistry` already indexes fitted artifacts by chain path and is shared across pipeline variants within a dataset (section 1.1). The lowest-hanging fruit is adding a check-before-fit path in `TransformerMixinController.execute()`: before cloning and fitting, look up the chain path in the registry; if a matching artifact exists and the input data hash matches, load the fitted artifact and skip to `transform()`. This turns the existing storage-level deduplication into computation-level caching with minimal new code.

2. **Always include data identity in cache keys**: Chain paths alone are not sufficient for safe caching (section 1.10). Every cache key must combine `(chain_path_hash, input_data_content_hash)`. This is critical when datasets are loaded from multiple folders that may produce different sample partitions. The existing `SpectroDataset.metadata()` content hash provides the data identity component. For future cross-run caching, extend `ArtifactRecord` with an `input_data_hash` field.

3. **Start with step-mode logic in the refit pass**: The retrainer already has `StepMode` with train/predict/skip modes. Extend this to determine which steps can reuse CV artifacts during refit.

4. **Add in-memory preprocessed data snapshot cache**: Within `PipelineOrchestrator.execute()`, cache the dataset state after each preprocessing step, keyed by `(chain_path_hash, dataset_content_hash)`. This benefits both generator sweeps (current architecture) and the refit pass.

5. **Leverage the existing `DataCache` class**: It is well-designed (thread-safe, LRU, size-limited, staleness detection) and should be repurposed for preprocessed snapshots instead of building a new cache.

6. **Do not cache model training**: Models must be retrained on full data during refit. Focus caching efforts on preprocessing only.
