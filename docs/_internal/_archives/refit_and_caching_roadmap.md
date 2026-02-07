# Refit-After-CV + Caching: Unified Implementation Roadmap

> **Source documents**: `refit_after_cv_analysis.md`, `caching_analysis.md`
> **Date**: 2026-02-06

---

## 0. Design Merge Analysis: Potential Problems

Before defining phases, this section identifies conflicts, shared dependencies, and risks that arise from implementing the refit and caching designs together.

### 0.1 Shared Foundational Requirements

Both designs depend on infrastructure that does not yet exist. Building either feature without the other's requirements in mind would create rework.

| Requirement | Needed by Refit | Needed by Caching | Current State |
|---|---|---|---|
| **Data content hashing** (input data identity in cache keys) | Yes — refit must verify that cached preprocessing matches the refit's full training data | Yes — cache keys must include `(chain_path_hash, input_data_hash)` for correctness | Partial: `SpectroDataset.metadata()` computes an MD5 from a data sample, but it is not integrated into any cache lookup |
| **Execution phase concept** (CV vs refit) | Yes — the executor must distinguish Pass 1 (CV) from Pass 2 (refit) to apply different logic (splitter replacement, refit_params, fold artifact cleanup) | Yes — the cache must know whether a `fit_on_all` artifact from CV can be reused in refit, or whether the training data changed | Does not exist. The executor runs steps linearly with no phase awareness |
| **Topology analyzer** | Yes — dispatches to simple/stacking/nested/separation refit strategies | Partially — the snapshot cache benefits from knowing shared prefixes, but doesn't strictly require topology analysis | Does not exist |
| **`OperatorChain` as cache key** | Yes — identifies which preprocessing chain the refit is replaying | Yes — chain path is the operator-identity half of every cache key | Exists but is only used for artifact naming, not for cache lookups |

**Risk if built separately**: If caching is implemented first without considering the refit execution phase, the cache will not distinguish CV-fitted artifacts from refit-fitted artifacts. If refit is implemented first without caching, the refit pass will redundantly refit preprocessing steps that could be skipped.

**Recommendation**: Build the shared foundation (data hashing, phase concept) as Phase 0 before either feature.

### 0.2 Artifact Lifecycle Conflicts

The two designs modify artifact lifecycle in potentially conflicting ways:

| Aspect | Caching Design | Refit Design | Conflict? |
|---|---|---|---|
| **Fold model artifacts** | Not affected (caching targets preprocessing, not models) | Transient: persist during CV, delete after refit succeeds (Section 8.2.2) | No direct conflict, but the cache must not hold references to fold model artifacts that get deleted |
| **ArtifactRegistry._by_chain_path** | Cache reads this index to detect reusable preprocessing | Refit reads this index to identify the winning chain's artifacts | No conflict — both are read operations. But writes from the refit pass must update the index correctly |
| **Content-addressed dedup** | Cache skip-before-fit reduces the number of artifacts produced (fewer fits = fewer to dedup) | Refit-only persistence reduces the number of artifacts stored (fold artifacts cleaned up) | Synergy — both reduce artifact count |

**Risk**: The `ArtifactRegistry` is recreated per dataset in the orchestrator loop. If the refit pass creates a new registry (because it's a "new pipeline execution"), all cached chain-path mappings from Pass 1 are lost. The refit pass would not be able to look up CV artifacts for `fit_on_all` reuse.

**Solution**: The refit pass must share the same `ArtifactRegistry` instance as Pass 1, or explicitly carry over the chain-path index. This is a design constraint that must be established in Phase 0.

### 0.3 StepCache Scope and Data Identity

The StepCache (preprocessed data snapshots) operates within a single `PipelineOrchestrator.execute()` call. Both CV and refit happen within that call, so the cache naturally spans both phases. However:

- **During CV**: Steps are fitted on fold training subsets (e.g., 80% of training data). The cache stores the preprocessed state for that subset.
- **During refit**: The same steps are fitted on ALL training data (100%). The preprocessed state is different.

The cache key `(chain_path_hash, input_data_hash)` handles this correctly — the data hash differs between fold-subset and full-training data. But there are subtleties:

| Step Type | CV Input Data | Refit Input Data | Same Hash? | Cache Reusable? |
|---|---|---|---|---|
| `fit_on_all: True` | All data (train+test) | All data (train+test) | Yes | Yes — skip fitting in refit |
| Stateless transform | Any data | Any data | Depends on input | Yes, if input data hash matches |
| Normal transform | Fold training subset | Full training data | No | No — must refit |
| Pre-splitter step | All training data | All training data | Yes (if no upstream change) | Yes — skip fitting in refit |

**Key insight**: Pre-splitter preprocessing steps (steps before the CV splitter in the pipeline) receive the SAME input data in both CV and refit (the full training set, before fold splitting). Their cached results from CV are directly reusable in refit. This is a significant optimization that neither document explicitly identifies.

**Risk**: If the cache does not correctly distinguish pre-splitter from post-splitter steps, it might incorrectly serve a fold-subset result for a refit request (or vice versa). The data hash prevents this, but the implementation must be careful.

### 0.4 Memory Pressure Compounding

Both features add memory consumption within a single run:

| Component | Memory Cost | Lifetime |
|---|---|---|
| StepCache (preprocessed snapshots) | 16-64 MB per step × N steps × M sources | Duration of run |
| Fold model artifacts on disk | N models × K folds × model_size | Until cleanup after refit |
| Refit model artifacts | N models × model_size | Persisted permanently |
| ArtifactRegistry chain index | Negligible (dict of strings) | Duration of run |

For a typical pipeline (5 preprocessing steps, 2 sources, 5-fold CV, 3 models): StepCache ≈ 640 MB, fold artifacts on disk ≈ model-dependent. With the LRU eviction on StepCache (`max_size_mb=2048`), this is manageable.

**Risk**: For very large datasets (10,000+ samples × 2,000+ wavelengths) or many generator variants, the StepCache could consume significant memory. The existing `DataCache` class has LRU eviction and size limits — reusing it mitigates this.

### 0.5 Execution Flow Interaction

Both features modify the pipeline execution flow at different levels:

```
PipelineOrchestrator.execute()
  ├─ Pass 1: variant loop (existing)
  │   ├─ PipelineExecutor._execute_steps()
  │   │   ├─ _execute_single_step()
  │   │   │   ├─ [CACHING] StepCache.get() — check before executing
  │   │   │   ├─ Controller.execute()
  │   │   │   │   └─ TransformerMixinController.execute()
  │   │   │   │       └─ [CACHING] check-before-fit via ArtifactRegistry
  │   │   │   └─ [CACHING] StepCache.put() — store after executing
  │   │   └─ ...
  │   └─ ...
  ├─ [REFIT] Selection: identify winning config per model
  ├─ [REFIT] Pass 2: execute_refit()
  │   ├─ [REFIT] Replace splitter with dummy fold
  │   ├─ PipelineExecutor._execute_steps()  (reuses same flow)
  │   │   └─ [CACHING] Same cache checks apply (pre-splitter steps may hit cache)
  │   ├─ [REFIT] Persist as fold_id="final"
  │   └─ [REFIT] Cleanup fold artifacts
  └─ Return results
```

The refit pass reuses the same `_execute_steps()` flow, which means caching hooks in `_execute_single_step()` and `TransformerMixinController` automatically apply to both CV and refit. This is a design synergy — caching accelerates both phases without duplicating logic.

**Risk**: If the refit pass uses a different execution path (e.g., a custom replay function), the caching hooks won't apply. The refit should reuse the standard execution path, not create a parallel one.

### 0.6 Check-Before-Fit and Refit Correctness

The caching design's P0 proposal (check-before-fit in `TransformerMixinController`) must interact correctly with the refit:

- **During CV**: If pipeline variant #2 shares preprocessing with variant #1, the check-before-fit loads variant #1's fitted artifact instead of refitting. Correct behavior.
- **During refit**: The refit pass may share preprocessing with CV (pre-splitter steps, `fit_on_all` steps). Check-before-fit should load the CV artifact. Correct behavior.
- **During refit (post-splitter steps)**: The refit's training data differs from any CV fold's data. Check-before-fit will find no match (data hash differs). Correct behavior — the step refits on full data.

**Conclusion**: Check-before-fit works correctly for both CV and refit, provided the data hash is always included in the cache key. No special handling needed.

### 0.7 Schema and Store Changes

Both features require changes to the persistence layer:

| Change | Source | Impact |
|---|---|---|
| `refit_context` column in predictions table | Refit | Schema migration |
| `fold_id="final"` prediction entries | Refit | New entry type |
| `input_data_hash` in ArtifactRecord (optional, for cross-run cache) | Caching | Schema extension (deferred to later phase) |
| `refit_params` keyword in parser | Refit | Step parser change |
| `_stateless` class attribute on operators | Caching | Operator base class change |

These changes are independent and can be implemented in any order. No conflict.

### 0.8 Summary of Merge Risks

| Risk | Severity | Mitigation |
|---|---|---|
| ArtifactRegistry not shared between CV and refit passes | High | Refit pass reuses the same registry instance |
| Cache serves fold-subset result for refit request | Medium | Data hash in cache key prevents this by construction |
| Cache references deleted fold artifacts | Low | Cache stores dataset snapshots, not artifact references; fold artifacts are model-level, not preprocessing-level |
| Memory pressure from cache + fold artifacts | Medium | LRU eviction on StepCache; fold artifacts on disk, not memory |
| Refit uses different execution path, bypassing cache | High | Refit must reuse standard `_execute_steps()` flow |
| Phase concept not established before building either feature | High | Build Phase 0 (foundation) first |

---

## 1. High-Level Phase Overview

```
Phase 0: Shared Foundation
    │
    ├──────────────────┐
    ▼                  ▼
Phase 1: Core         Phase 2: Simple Refit
Caching               (non-stacking)
    │                  │
    ├──────────────────┤
    ▼                  ▼
Phase 3: Advanced Caching + Stacking Refit
    │
    ▼
Phase 4: Advanced Refit (nested, separation, lazy, warm-start)
    │
    ▼
Phase 5: Webapp Integration + Cross-Run Caching
```

| Phase | Name | Depends On | Coverage |
|---|---|---|---|
| **0** | Shared Foundation | — | Data identity hashing, execution phase concept, topology analyzer, ArtifactRegistry cache-key support |
| **1** | Core Caching | Phase 0 | Check-before-fit, `fit_on_all` reuse, stateless transform skip, cross-pipeline preprocessing reuse |
| **2** | Simple Refit | Phase 0 | Two-pass orchestration, simple refit for non-stacking pipelines, `fold_id="final"`, result object, `refit_params`, bundle changes |
| **3** | Advanced Caching + Stacking Refit | Phases 1, 2 | Preprocessed data snapshot cache (StepCache), stacking refit (two-step), mixed merge, per-model selection |
| **4** | Advanced Refit | Phase 3 | Nested stacking (recursive), separation branch refit, lazy per-model standalone refits, warm-start for deep learning |
| **5** | Webapp Integration + Cross-Run Caching | Phase 4 | WebSocket events, UI phases, cross-run DuckDB-backed cache, `input_data_hash` in ArtifactRecord |

**Phases 1 and 2 are parallelizable** — they share Phase 0 as a dependency but are otherwise independent. Phase 3 requires both because the stacking refit benefits from the snapshot cache, and the snapshot cache is validated by the refit pass.

---

## 2. Phase 0 — Shared Foundation

**Goal**: Build the infrastructure that both caching and refit depend on, without yet implementing either feature.

### Task 0.1 — Data Content Hashing Utility

**What**: Create a fast, deterministic content hash function for numpy arrays / SpectroDataset feature data.

**Why**: Both caching (cache keys) and refit (`fit_on_all` reuse, pre-splitter step detection) require verifying data identity. The existing `SpectroDataset.metadata()` computes an MD5 from a data sample (first 100 rows), which is too coarse — identical hashes for different datasets with the same first 100 rows.

**Details**:
- Create `nirs4all/utils/hashing.py` with `compute_data_hash(X: np.ndarray) -> str` using xxhash (fast) or SHA-256 (secure). xxhash preferred for in-process caching (performance), SHA-256 for cross-run persistence.
- Hash the full array data (not a sample). For typical NIRS datasets (1000×2000 float64 = 16 MB), xxhash takes ~1ms.
- Add `SpectroDataset.content_hash(source_index: int | None = None) -> str` that hashes the feature data for a given source (or all sources).
- Cache the hash within the `SpectroDataset` instance (invalidated on `replace_features()`, `add_features()`, or any mutation).

**Touches**: `nirs4all/utils/hashing.py` (new), `nirs4all/data/dataset.py`

**Acceptance**: `dataset.content_hash()` returns a deterministic hash. Two datasets with identical features produce the same hash. Mutating features invalidates the cached hash. Performance: <5ms for 16 MB arrays.

### Task 0.2 — Execution Phase Enum and Context

**What**: Add an `ExecutionPhase` enum (`CV`, `REFIT`) to the execution context so that controllers and cache logic can distinguish between the two passes.

**Why**: The transformer controller needs to know if it's in the refit phase to decide whether to reuse `fit_on_all` artifacts. The cache needs to know the phase for logging/debugging. The model controller needs the phase for `refit_params` resolution and fold artifact cleanup.

**Details**:
- Add `ExecutionPhase` enum in `nirs4all/pipeline/config/execution_context.py` (or a new `nirs4all/pipeline/execution/phase.py` if cleaner).
- Add `phase: ExecutionPhase` field to the runtime context (or `ExecutionContext`) that is passed through the execution flow.
- Default to `ExecutionPhase.CV`. The refit orchestration (Phase 2) will set it to `ExecutionPhase.REFIT`.
- Controllers access it via `runtime_context.phase`.

**Touches**: `nirs4all/pipeline/config/execution_context.py` or `nirs4all/pipeline/execution/`, `RuntimeContext` propagation

**Acceptance**: `runtime_context.phase` is accessible in all controllers. Default is `CV`. Setting it to `REFIT` does not break any existing test.

### Task 0.3 — Topology Analyzer

**What**: Create a `PipelineTopology` descriptor and an `analyze_topology(expanded_steps)` function that inspects an expanded pipeline configuration and identifies its structural properties.

**Why**: The refit mechanism dispatches to different strategies (simple, stacking, nested, separation) based on topology. The caching layer uses topology info for shared-prefix detection. Both need this analysis.

**Details**:
- Create `nirs4all/pipeline/analysis/topology.py` (or `nirs4all/pipeline/config/topology.py`).
- `PipelineTopology` dataclass with fields:
  - `has_stacking: bool` — `merge: "predictions"` present
  - `has_feature_merge: bool` — `merge: "features"` present
  - `has_mixed_merge: bool` — `merge: {"features": ..., "predictions": ...}` present
  - `has_concat_merge: bool` — `merge: "concat"` present
  - `has_separation_branch: bool` — `by_metadata`, `by_tag`, `by_source` branches
  - `has_branches_without_merge: bool` — competing branches (no merge step after)
  - `max_stacking_depth: int` — nesting depth of `merge: "predictions"` (1 for flat stacking, 2+ for nested)
  - `model_nodes: list[ModelNodeInfo]` — all model nodes with their branch path, step index, and containing merge type
  - `splitter_step_index: int | None` — index of the CV splitter step (or None if no splitter)
  - `has_sequential_models: bool` — multiple model steps without explicit branching/merging (implicit stacking, Section 6.8)
  - `has_multi_source: bool` — `by_source` branches (structurally equivalent to stacking when inner models produce predictions, Section 4.14)
- `analyze_topology(steps: list) -> PipelineTopology` walks the step list recursively through branches.
- Handle the edge case from Section 8.2.6: `separation_branch → inner_model → merge("concat") → outer_model` should flag `has_stacking = True` (leakage concern).
- Detect sequential model steps (Section 6.8): `[SNV → PLS → Ridge]` without explicit branch/merge is implicit stacking. If Model 2 receives features derived from Model 1's predictions, the OOF dependency applies. Flag `has_sequential_models = True` and `has_stacking = True`.
- Distinguish `by_source` branches (Section 4.14) from `by_metadata`/`by_tag` separation: `by_source` with inner models producing predictions merged via `{"sources": "concat"}` is structurally stacking (each source's model predictions become meta-features).

**Touches**: new file `nirs4all/pipeline/analysis/topology.py`

**Acceptance**: Correct topology detection for all 12+ cases in the refit analysis Section 4.17 edge case matrix. Unit tests for each topology.

### Task 0.4 — ArtifactRegistry Cache-Key Support

**What**: Extend `ArtifactRegistry` to support cache lookups by `(chain_path, data_hash)` pairs, not just by `chain_path` alone.

**Why**: The caching analysis Section 1.10 identifies that chain paths alone are insufficient for safe caching — they must be combined with input data identity. This is the foundation for check-before-fit (Phase 1) and `fit_on_all` reuse (Phase 1/2).

**Details**:
- Add a parallel index `_by_chain_and_data: dict[tuple[str, str], str]` in `ArtifactRegistry`, mapping `(chain_path_hash, input_data_hash)` → `artifact_id`.
- Extend `register_with_chain()` to accept an optional `input_data_hash: str` parameter and populate the new index.
- Add `get_by_chain_and_data(chain_path: str, data_hash: str) -> ArtifactRecord | None` lookup method.
- The existing `_by_chain_path` index remains unchanged (backward compatible).

**Touches**: `nirs4all/pipeline/storage/artifacts/artifact_registry.py`

**Acceptance**: `get_by_chain_and_data()` returns the correct artifact when both chain path and data hash match. Returns `None` when either differs. Existing `get_by_chain()` behavior unchanged.

### Task 0.5 — Ensure ArtifactRegistry Lifespan Spans Both Passes

**What**: Verify and enforce that the `ArtifactRegistry` instance created in `PipelineOrchestrator._execute_dataset()` survives through the refit pass, so that chain-indexed artifacts from CV are accessible during refit.

**Why**: Section 0.2 of this document identifies that if the refit pass creates a new registry, all cached chain-path mappings from Pass 1 are lost.

**Details**:
- Audit `PipelineOrchestrator._execute_dataset()` to confirm the registry is created once and passed to all variant executions.
- Ensure the refit execution (Phase 2) receives the same registry instance. If it currently creates a new `PipelineExecutor` per variant, the registry must be threaded through.
- Document the lifecycle: registry is created at dataset start, used by all CV variants, used by the refit pass, destroyed at dataset end.

**Touches**: `nirs4all/pipeline/execution/orchestrator.py`

**Acceptance**: A test that registers an artifact in a CV variant and retrieves it during the refit phase via `get_by_chain_and_data()`.

### Task 0.6 — RepeatedKFold OOF Accumulation Bug Fix (Pre-Existing)

**What**: Fix the pre-existing bug in `TrainingSetReconstructor._collect_oof_predictions()` where multiple OOF predictions per sample (from RepeatedKFold) overwrite each other instead of being accumulated and averaged.

**Why**: Section 8.13 of the refit analysis identifies this as issue #22. When `RepeatedKFold(n_splits=5, n_repeats=3)` is used, each training sample appears in 3 different validation folds. The current code uses simple assignment `oof_preds[pos] = y_vals[i]`, silently discarding earlier predictions. This affects OOF prediction quality for stacking and for the optional OOF-based meta-model training mode.

**Details**:
- In `TrainingSetReconstructor._collect_oof_predictions()` (around `reconstructor.py:771-775`):
  - Replace simple assignment with accumulation: track `oof_preds_sum` and `oof_preds_count` per position.
  - Average where multiple predictions exist: `oof_preds[mask] = oof_preds_sum[mask] / oof_preds_count[mask]`.
- This is a correctness fix independent of the refit feature, but it surfaces during refit when `stacking_meta_training="oof"` is used with a generator-selected `RepeatedKFold`.

**Touches**: `nirs4all/controllers/models/stacking/reconstructor.py`

**Acceptance**: A pipeline using `RepeatedKFold(n_splits=5, n_repeats=3)` produces averaged OOF predictions where each sample's OOF value is the mean across all repeats, not just the last one.

---

## 3. Phase 1 — Core Caching

**Goal**: Eliminate redundant preprocessing computation within a single run. Standalone value for generator sweeps; prepares the infrastructure for refit acceleration.

**Depends on**: Phase 0 (data hashing, ArtifactRegistry cache-key support)

### Task 1.1 — Check-Before-Fit in TransformerMixinController

**What**: Before cloning and fitting a transformer, consult the `ArtifactRegistry` to check if an identical preprocessing step has already been fitted on the same data. If so, load the existing fitted artifact and skip fitting.

**Why**: This is the P0 item from the caching analysis. When 100 pipeline variants share the same first 3 preprocessing steps (common in generator sweeps), fitting is repeated 100 times. With check-before-fit, it is fitted once and reused 99 times.

**Details**:
- In `TransformerMixinController.execute()` (around line 291):
  1. Compute the input data hash: `data_hash = dataset.content_hash()`.
  2. Compute the current chain path (already available via the chain context).
  3. Call `artifact_registry.get_by_chain_and_data(chain_path, data_hash)`.
  4. If hit: load the fitted transformer artifact. Set it as the operator. Skip to `transform()`.
  5. If miss: proceed with clone → fit → transform as usual. After fitting, register with data hash: `artifact_registry.register_with_chain(..., input_data_hash=data_hash)`.
- Log cache hits: `"Cache hit: reusing fitted {operator_class} from chain {chain_path} (data hash match)"`.
- Ensure the loaded artifact is a compatible fitted transformer (same class, same params).

**Touches**: `nirs4all/controllers/transforms/transformer.py`, `nirs4all/pipeline/storage/artifacts/artifact_registry.py`

**Acceptance**: A generator sweep pipeline `[{"_or_": [PLS(5), PLS(10), PLS(15)]}, SNV()]` where SNV precedes the generator split: SNV is fitted once, reused for all 3 variants. Validate via log messages or a counter.

### Task 1.2 — `fit_on_all` Artifact Reuse Validation

**What**: Validate and test that the check-before-fit mechanism from Task 1.1 correctly handles `fit_on_all: True` transformers across folds and between CV and refit phases.

**Why**: `fit_on_all` transformers are fitted on all data (train+test) regardless of fold structure. Their fitted state is identical across folds and between CV and refit. Task 1.1's check-before-fit handles this automatically (same data → same hash → cache hit), but specific validation is needed to ensure the data hash at the `fit_on_all` call point reflects the full data (train+test), not just the fold training subset.

**Details**:
- Verify that `TransformerMixinController.execute()` correctly passes the full data (train+test) to `fit()` when `fit_on_all=True`, and that the `dataset.content_hash()` computed at that point reflects the full data (not the fold subset).
- Verify that Task 1.1's cache lookup produces a hit when the same `fit_on_all` step is encountered in a different fold or during the refit phase.
- Write integration tests covering:
  - Same `fit_on_all` step across multiple CV folds: fitted once, reused K-1 times.
  - Same `fit_on_all` step between CV and refit phase: fitted once in CV, reused in refit.
  - Different `fit_on_all` steps (different operator class/params): no incorrect cache hit.
- This task is primarily validation/testing, not new code. If the data hash at the `fit_on_all` call point is incorrect (e.g., includes only training data), fix the hash computation.

**Touches**: `nirs4all/controllers/transforms/transformer.py` (validation + potential fix), integration tests

**Acceptance**: A pipeline with `{"fit_on_all": True, "step": StandardScaler()}` — the scaler is fitted once during CV and reused (not re-fitted) during refit. Verified by log or counter.

### Task 1.3 — Stateless Transform Detection and Skip

**What**: Add a `_stateless` class attribute to operators whose output depends only on input data and fixed parameters (not on learned state). Skip fitting entirely for these operators when the input data and params match a previous execution.

**Why**: Transforms like fixed-parameter SavitzkyGolay derivatives, `CropTransformer`, `ResampleTransformer`, and `Detrend` with fixed polynomial order do not learn from data. Their `fit()` is a no-op (or stores trivial state). Skipping them saves time and simplifies caching.

**Details**:
- Add `_stateless: ClassVar[bool] = False` to `SpectraTransformerMixin` (base class).
- Set `_stateless = True` on: `CropTransformer`, `ResampleTransformer`, `Detrend` (fixed order), `ToAbsorbance`, `FromAbsorbance`, `KubelkaMunk`, `SignalTypeConverter`, first/second derivative with fixed parameters.
- In `TransformerMixinController.execute()`, after the check-before-fit from Task 1.1: if the operator has `_stateless = True`, the fit step can be further optimized — the fitted "state" is always the same regardless of training data. Use a simplified cache key: `(chain_path_hash, operator_params_hash)` without the data hash (since data doesn't affect the fit).
- For stateless operators, the transform output still depends on input data, so the StepCache (Phase 3) still keys on data hash.

**Touches**: `nirs4all/operators/base/`, `nirs4all/operators/transforms/` (multiple files), `nirs4all/controllers/transforms/transformer.py`

**Acceptance**: `CropTransformer._stateless == True`. A pipeline running the same stateless transform on different data subsets (fold vs full) correctly reuses the fitted operator without re-calling fit().

### Task 1.4 — Cross-Pipeline Preprocessing Reuse (Generator Sweeps)

**What**: Validate that the check-before-fit from Task 1.1 correctly handles the generator sweep case where multiple pipeline variants share a common preprocessing prefix.

**Why**: This is the highest-impact optimization for current usage. Generator sweeps like `[SNV(), {"_or_": [PLS(5), PLS(10)]}]` should compute SNV only once.

**Details**:
- This is primarily a validation/integration task. Task 1.1 provides the mechanism.
- Write integration tests for:
  - Generator varying only model params: preprocessing shared across all variants.
  - Generator varying preprocessing choice: first preprocessing differs, but shared steps after divergence should still be detected.
  - Generator varying splitter: preprocessing shared (different splitters don't affect preprocessing).
  - Cartesian generator: all prefix combinations.
- Verify that the `ArtifactRegistry` chain-path index correctly handles multi-variant scenarios.
- Measure and document performance improvement: run a 100-variant generator sweep with and without check-before-fit. Report time savings.

**Touches**: Integration tests, benchmarks

**Acceptance**: Demonstrated time savings on a generator sweep benchmark. No correctness regressions in existing tests.

---

## 4. Phase 2 — Simple Refit (Non-Stacking)

**Goal**: Implement the two-pass architecture for pipelines without `merge: "predictions"`. Covers 80% of use cases: simple pipelines, generators, finetuning, generators+finetuning, branch+features merge.

**Depends on**: Phase 0 (execution phase, topology analyzer)

### Task 2.1 — Prediction Store: `refit_context` Column and `fold_id="final"`

**What**: Extend the prediction store schema to support refit entries.

**Why**: The refit produces new prediction entries with `fold_id="final"` that must coexist with CV entries. The `refit_context` field distinguishes standalone refits from stacking-context refits (Phase 3), and prevents uniqueness constraint violations (Section 8.6.2).

**Details**:
- Add `refit_context VARCHAR DEFAULT NULL` column to the DuckDB `predictions` table schema in `store_schema.py`.
- Update the uniqueness key to include `refit_context` (or ensure no constraint conflicts).
- Update `Predictions` class to accept and store `refit_context` in prediction records.
- Define constants: `REFIT_CONTEXT_STANDALONE = "standalone"`, `REFIT_CONTEXT_STACKING = "stacking"`, `None` for CV entries.
- Update `WorkspaceStore` schema migration to add the column to existing databases.

**Touches**: `nirs4all/pipeline/storage/store_schema.py`, `nirs4all/pipeline/storage/workspace_store.py`, `nirs4all/pipeline/execution/predictions.py`

**Acceptance**: A prediction entry with `fold_id="final"` and `refit_context="standalone"` can be created, stored, and queried without conflicting with CV entries.

### Task 2.2 — `refit_params` Keyword Support

**What**: Add `refit_params` as a recognized keyword in the pipeline step parser, enabling refit-specific training parameter overrides.

**Why**: Section 8.11 specifies that training parameters optimal for CV may not be optimal for refit (more epochs, lower LR, warm-start control). This is a parser-level change needed before the refit execution can resolve parameters.

**Details**:
- Add `"refit_params"` to `RESERVED_KEYWORDS` in `nirs4all/pipeline/steps/parser.py`.
- In `PipelineConfigs._preprocess_steps()`, handle `refit_params` the same way `train_params` and `finetune_params` are handled (key normalization to `{"model": {..., "refit_params": {...}}}`).
- Add `resolve_refit_params(model_config: dict) -> dict` utility: merges `refit_params` on top of `train_params` (refit overrides win on conflicts). Unspecified parameters inherit from `train_params`.
- Support special `refit_params` keys: `warm_start` (bool), `warm_start_fold` (str: `"best"`, `"last"`, `"fold_N"`). These are stored but not acted on until Phase 4.

**Touches**: `nirs4all/pipeline/steps/parser.py`, `nirs4all/pipeline/config/pipeline_config.py`

**Acceptance**: A pipeline step `{"model": PLS(), "train_params": {"verbose": 0}, "refit_params": {"verbose": 1}}` parses without error. `resolve_refit_params()` returns `{"verbose": 1}`.

### Task 2.3 — Winning Configuration Extraction

**What**: After Pass 1 completes, extract the winning pipeline configuration (expanded steps + finetuned params) for each model node.

**Why**: The refit must replay the exact configuration that won the CV selection. This includes the expanded generator choices, the finetuned params from Optuna, and the full preprocessing chain.

**Details**:
- Create `nirs4all/pipeline/execution/refit/config_extractor.py` (or similar).
- `extract_winning_config(predictions, variant_configs, model_node_info) -> RefitConfig` that:
  1. Queries predictions for the best val_score entry.
  2. Maps back to the pipeline variant that produced it (via `pipeline_id` or variant index).
  3. Retrieves the expanded step list for that variant.
  4. Retrieves `best_params` from the prediction record (Section 8.1.3: already stored).
  5. For `individual` Optuna mode: selects params from the best-performing fold (Section 8.2.3).
  6. Returns a `RefitConfig` with: `expanded_steps`, `best_params`, `variant_index`, `generator_choices`.
- Handle per-model selection: each model node independently identifies its best variant (Section 3.10, 5.6). For Phase 2 (non-stacking), there is only one model, so global selection = per-model selection.

**Touches**: new file in `nirs4all/pipeline/execution/refit/`

**Acceptance**: For a generator pipeline with finetuning, `extract_winning_config()` returns the correct variant's steps with the correct finetuned params.

### Task 2.4 — Simple Refit Execution

**What**: Implement `execute_simple_refit()` that re-executes the winning configuration on all training data (no CV folds).

**Why**: This is the core refit mechanism for 80% of use cases.

**Details**:
- Create `nirs4all/pipeline/execution/refit/executor.py`.
- **Design note on Retrainer relationship** (Section 7.4): The existing `Retrainer` class has FULL/TRANSFER/FINETUNE modes. The refit mechanism is conceptually a new mode (REFIT) but is architecturally different: it is automatic (within `nirs4all.run()`), topology-aware (dispatches to simple/stacking), and tightly coupled to the CV selection result. Rather than extending the Retrainer class (which is designed for user-initiated, cross-dataset retraining), implement the refit as a standalone module in `nirs4all/pipeline/execution/refit/`. The two systems share the chain replay concept but serve different purposes. If significant code duplication emerges during implementation, refactor shared logic into a common utility.
- `execute_simple_refit(refit_config, dataset, context, artifact_registry) -> RefitResult`:
  1. Set `context.phase = ExecutionPhase.REFIT`.
  2. Take the winning variant's expanded steps.
  3. **Replace the splitter step**: detect the splitter step via `CrossValidatorController.matches()` and replace it with a dummy single "fold" `[(all_train_indices, [])]`, or skip it entirely and pass all training data.
  4. **Inject finetuned params**: modify the model step to use `best_params`. Apply `resolve_refit_params()` to merge `refit_params` on top.
  5. **Execute the pipeline** using the standard `PipelineExecutor._execute_steps()` flow (reuses caching hooks from Phase 1).
  6. The single "fold" trains on all training data. No validation set.
  7. Evaluate on the test set (if exists). If no test set, log warning and set `test_score=None` (Section 8.2.4).
  8. Create prediction entry with `fold_id="final"`, `refit_context="standalone"`.
  9. Persist the refit model artifact.
- Handle edge cases:
  - No CV splitter in pipeline (Section 8.4.3): the model from Pass 1 IS the final model. Skip refit, relabel the existing prediction entry. Log: "No cross-validation detected. The trained model is already the final model."
  - No test set (Section 8.2.4): refit trains on all data (entire dataset), `final_score=None`, warn. The refit model is still useful for deployment.
  - `y_processing` (Section 8.4.2): clone the target scaler, refit on all training targets (`fit(y_train_all)`). Use for `transform(y_train_all)` during training and `inverse_transform(y_pred)` during evaluation.
  - Classification (Section 8.4.1): no special handling needed. Train on all data, evaluate test accuracy/F1. For stacking (Phase 3): OOF predictions carry class probability columns.
  - Augmentation (Section 6.4): runs normally during refit. The augmentation controller already handles the all-data case — the refit operates like a single fold with all training data. Augmentation is stochastic and will produce different samples than any CV fold, which is acceptable (the refit model should generalize, not reproduce exact CV results).
  - Filters/exclusions (Section 6.5): re-run filters during refit (fit filter on all training data). This ensures the refit model sees the same quality of data, adapted to the full training set. Outlier detection on the full set may slightly differ from fold-level detection — this is acceptable and more representative of the deployment scenario.

**Touches**: new file `nirs4all/pipeline/execution/refit/executor.py`, `nirs4all/pipeline/execution/orchestrator.py`

**Acceptance**: A simple pipeline `[SNV(), KFold(5), PLS(10)]` produces a `fold_id="final"` prediction entry with the correct test score. The refit model is persisted as a single artifact. A pipeline with `y_processing`, augmentation, and exclusion steps produces correct refit results.

### Task 2.5 — Orchestrator Two-Pass Flow

**What**: Modify `PipelineOrchestrator.execute()` to add the refit pass after the variant loop.

**Why**: The refit must be triggered automatically after Pass 1 completes, within the same `nirs4all.run()` call.

**Details**:
- In `PipelineOrchestrator.execute()`, after the variant loop (line ~268):
  1. Check `refit_enabled` (from the `refit` parameter of `nirs4all.run()`).
  2. Call `analyze_topology(expanded_steps)` to determine the refit strategy.
  3. If `topology.has_stacking`: defer to Phase 3 (stacking refit). For now, log "Stacking refit not yet implemented" and skip.
  4. Otherwise: call `extract_winning_config()` then `execute_simple_refit()`.
  5. Add the refit result to `run_predictions`.
- Add `refit` parameter to `nirs4all.run()` API:
  - `refit=True` (default): enable refit.
  - `refit=False`: disable refit (legacy behavior).
  - `refit={...}`: dict with options (`enabled`, `default_refit_params`, `stacking_meta_training`).
- Pass the `refit` config through: `api/run.py` → `PipelineRunner` → `PipelineOrchestrator`.
- **Critical**: Pass the same `ArtifactRegistry` instance from Pass 1 to `execute_simple_refit()` (per Task 0.5). The refit pass must not create a new registry.

**Touches**: `nirs4all/pipeline/execution/orchestrator.py`, `nirs4all/api/run.py`, `nirs4all/pipeline/runner.py` (or equivalent)

**Acceptance**: `nirs4all.run(pipeline, dataset, refit=True)` produces both CV entries and a `fold_id="final"` refit entry. `refit=False` produces only CV entries (legacy behavior).

### Task 2.6 — Fold Artifact Lifecycle: Persist Then Cleanup

**What**: During Pass 1, persist fold model artifacts to disk as before (avoiding memory pressure). After a successful refit, clean up transient fold artifacts.

**Why**: Section 8.2.2 identifies that keeping fold models in memory would cause OOM for large models. The persist-then-cleanup strategy provides the same memory footprint as today during Pass 1, with clean final state after refit.

**Details**:
- During Pass 1: no change to artifact persistence behavior. Fold models and fold transformers are saved to disk as transient artifacts.
- **After successful refit — full artifact cleanup** at the dataset level (when all pipelines for a dataset are done):
  1. **Delete fold model artifacts**: all artifacts where `fold_id` is a CV fold (not "final", not "avg", not "w_avg") for every pipeline in the dataset run.
  2. **Delete fold transformer artifacts**: fitted preprocessor artifacts from CV folds. These are no longer needed — the refit chain has its own freshly fitted transformer artifacts.
  3. **Delete losing variant artifacts**: for generator sweeps, all artifacts from non-winning variants (their transformer and model artifacts). Only the winning variant's refit chain survives.
  4. **Keep the complete refit chain per model**: for each model node with `fold_id="final"`, the full chain of artifacts is preserved: all refit-fitted transformer artifacts upstream of the model + the refit model artifact itself. This is the deployable chain.
  5. **Keep ALL fold prediction records** (the DuckDB `predictions` + `prediction_arrays` rows): these are lightweight metadata and y_pred/y_true arrays. They are NOT deleted — they remain available for result browsing, analysis, and potential future retraining. This includes:
     - Per-fold predictions (fold_0, fold_1, ...) for every variant, branch, and model
     - Aggregated predictions (avg, w_avg)
     - Refit predictions (fold_id="final")
     - Base model OOF predictions from stacking (stored as fold-level predictions for each base model)
     - All scores, best_params, preprocessings metadata, branch_id, branch_name
  6. **Keep chain records**: the `chains` table entries remain (they describe the pipeline structure), only the binary artifacts on disk are cleaned up.
- If refit fails:
  - Do NOT delete any artifacts (they are the only model artifacts available as fallback).
  - Set `result.final = None`, `result.final_score = None`.
  - Log error with details.
- Add `cleanup_transient_artifacts(run_id, dataset_name, winning_pipeline_ids: list[str])` method to `WorkspaceStore`:
  1. Identify all artifact records for the dataset run.
  2. Mark artifacts belonging to the refit chains (winning pipelines, `fold_id="final"`) as permanent.
  3. Delete all other binary artifact files from the `artifacts/` directory.
  4. Update `ref_count` and remove orphaned artifact records from the `artifacts` table.
  5. Use content-addressed dedup awareness: if a binary file is referenced by both a transient and a permanent artifact record, do NOT delete it (decrement ref_count only).
- When `refit=False`: fold artifacts persist permanently (legacy behavior). No cleanup step runs. This is the current behavior — nothing changes.
- **Cleanup timing**: runs after ALL refit passes for a dataset complete (not per-pipeline). This ensures that cross-pipeline artifact dedup (shared preprocessors via content-addressed storage) is respected — a binary file shared between the winning chain and a losing variant is not deleted.

**Touches**: `nirs4all/pipeline/storage/workspace_store.py`, `nirs4all/pipeline/execution/refit/executor.py`, `nirs4all/pipeline/execution/orchestrator.py`

**Acceptance**:
- After successful refit: the workspace `artifacts/` directory contains only the refit chain binaries. No fold model or fold transformer files remain. No losing variant artifacts remain.
- The DuckDB `predictions` table still contains ALL prediction records (all folds, all variants, all branches) — queryable for analysis.
- After a failed refit: all artifact files remain on disk.
- With `refit=False`: all fold artifacts persist as before.

### Task 2.7 — Result Object Changes

**What**: Extend `RunResult` to expose refit results alongside CV results.

**Why**: Users need access to both the CV selection metric and the refit deployment metric (Section 8.1.4).

**Details**:
- Add to `RunResult`:
  - `final: PredictionEntry | None` — the outermost refit model's prediction entry.
  - `final_score: float | None` — the refit model's test score.
  - `cv_best: PredictionEntry` — the best CV entry (for comparison).
  - `cv_best_score: float` — CV-estimated performance.
  - `models: dict[str, ModelRefitResult]` — per-model refit results (populated lazily in Phase 4; for Phase 2, contains one entry for the single model).
- Keep `best_score` as the CV selection metric (unchanged, Section 8.1.4 recommendation (b)).
- `result.export("model.n4a")` exports the refit model (single model, not fold ensemble).
- `result.top(n)` includes the refit entry alongside CV entries, sorted by score. The refit entry is clearly labeled.

**Prediction browsability at all levels** — all CV prediction records are preserved in the DuckDB store (per Task 2.6), enabling multi-level result inspection:

| Level | Access | What's available |
|---|---|---|
| **Run** | `result.top(n)`, `store.query_predictions(run_id=...)` | All predictions across all variants, branches, folds, models |
| **Pipeline/variant** | `store.query_predictions(pipeline_id=...)` | All folds and branches within one generator variant |
| **Model** | `result.models["PLS"]`, `store.query_predictions(model_class=...)` | All variants × folds × partitions for one model type |
| **Chain** | `store.query_aggregated_predictions(chain_id=...)` | Aggregated scores (min/max/avg per partition) + list of all fold predictions |
| **Fold** | `store.get_chain_predictions(chain_id, fold_id="fold_0")` | One fold's y_true, y_pred, scores per partition |
| **Branch** | `store.query_predictions(branch_id=...)` | All predictions for one branch within stacking |
| **Refit** | `result.final`, `store.query_predictions(fold_id="final")` | The deployment model's test predictions |

For stacking pipelines specifically, the user can navigate from the final stacking result down to individual base model results:
- `result.final` → Ridge stacking refit test score
- `store.query_predictions(pipeline_id=winning_pipeline, branch_id=0)` → PLS base model's per-fold val scores, OOF predictions
- `store.query_predictions(pipeline_id=winning_pipeline, branch_id=1)` → RF base model's per-fold val scores, OOF predictions
- `store.query_predictions(pipeline_id=winning_pipeline, fold_id="avg")` → Aggregated CV scores per model
- All `y_true`/`y_pred` arrays are preserved in `prediction_arrays` for replotting, residual analysis, etc.

No existing prediction records are deleted by the refit cleanup — only binary artifact files are removed.

**Touches**: `nirs4all/api/result.py`

**Acceptance**: `result.final_score` returns the refit model's test score. `result.cv_best_score` returns the best CV val score. `result.export()` produces a bundle with a single model artifact. All CV fold/branch/model predictions remain queryable via the workspace store after refit cleanup.

### Task 2.8 — Bundle Export for Single Refit Model

**What**: Update `BundleGenerator` to export the single refit model instead of the fold ensemble.

**Why**: The refit model is the deployment model. Bundles should contain one model artifact, not K fold artifacts.

**Details**:
- In `BundleGenerator`, when `fold_id="final"` artifacts exist:
  - Export the single refit model artifact.
  - Export the refit preprocessing chain (transformers fitted on all training data).
  - Do NOT include fold model artifacts.
- In `BundleLoader`, support loading both:
  - New format: single refit model (direct prediction, no ensemble).
  - Legacy format: K fold models (ensemble prediction, for backward compatibility).
  - Detection: check if `fold_id="final"` exists in the bundle metadata.
- Update `MinimalPipelineExtractor` to extract the refit chain (the chain with `fold_id="final"` artifacts).

**Touches**: `nirs4all/pipeline/bundle/generator.py`, `nirs4all/pipeline/bundle/loader.py`, `nirs4all/pipeline/trace/extractor.py`

**Acceptance**: A bundle exported after refit contains 1 model artifact. Loading and predicting with it produces correct predictions. Legacy bundles (fold ensembles) still load and predict correctly.

### Task 2.9 — Prediction Mode: Single Refit Model Dispatch

**What**: Update the prediction mode in `BaseModelController` to detect refit models and use single-model prediction instead of fold ensemble averaging.

**Why**: Currently, prediction mode (`BaseModelController.train()` lines 787-811) hardcodes the pattern of loading per-fold model artifacts and ensembling them. With refit, prediction mode should load the single `fold_id="final"` artifact and predict directly.

**Details**:
- In `BaseModelController.train()` prediction mode path:
  1. Check if a `fold_id="final"` artifact exists for this model step.
  2. If yes: load the single refit model, predict directly (no ensemble, no averaging).
  3. If no (legacy bundles): fall back to the current fold ensemble behavior.
- Update `BundleLoader`'s prediction replay to use the same dispatch logic.
- This naturally simplifies the prediction flow: one forward pass instead of K forward passes + averaging.

**Touches**: `nirs4all/controllers/models/base_model.py`, `nirs4all/pipeline/bundle/loader.py`

**Acceptance**: `nirs4all.predict("model.n4a", new_data)` on a refit bundle uses single-model prediction. On a legacy bundle, it uses fold ensemble prediction. Both produce correct results.

### Task 2.10 — Refit Metadata Enrichment

**What**: Add metadata about the refit to the prediction record and the result object, including CV strategy information and generator choices.

**Why**: Section 8.13 recommends that when a generator varies the CV splitter, the result should include metadata about which splitter was selected. More generally, users need to understand what configuration was selected and refitted.

**Details**:
- Add to the `fold_id="final"` prediction record:
  - `cv_strategy: str` — description of the CV splitter used during selection (e.g., `"KennardStoneSplit(5)"`) from `generator_choices`.
  - `cv_n_folds: int` — number of CV folds.
  - `generator_choices: dict` — the full generator choice record for the winning variant.
  - `best_params: dict` — the finetuned params (if any) used for the refit.
- Expose on `result.final`:
  - `result.final.cv_strategy` — the CV strategy used for selection.
  - `result.final.generator_choices` — which generator options were selected.

**Touches**: `nirs4all/pipeline/execution/refit/executor.py`, `nirs4all/api/result.py`, `nirs4all/pipeline/execution/predictions.py`

**Acceptance**: `result.final.cv_strategy` returns `"KFold(5)"` for a pipeline that used KFold. `result.final.generator_choices` contains the selected options.

### Task 2.11 — `refit=True` as Default + Backward Compatibility

**What**: Make `refit=True` the default for `nirs4all.run()`, with `refit=False` available for the legacy behavior.

**Why**: The refit is the scientifically correct pattern and should be the default (Section 1.4).

**Details**:
- Set `refit=True` as default in `nirs4all.run()`.
- When `refit=False`:
  - Pass 1 runs as before (fold models persisted, fold ensemble for prediction).
  - No Pass 2.
  - `result.final = None`, `result.final_score = None`.
  - `result.export()` exports the fold ensemble (legacy behavior).
- Update all examples in `examples/` to work with refit (most should work without changes since refit adds results without removing any).
- Run the full example suite (`./run.sh -q`) to verify backward compatibility.

**Touches**: `nirs4all/api/run.py`, examples, integration tests

**Acceptance**: All existing examples pass with `refit=True` (default). Examples that explicitly set `refit=False` produce legacy behavior.

---

## 5. Phase 3 — Advanced Caching + Stacking Refit

**Goal**: Add the preprocessed data snapshot cache and implement the stacking refit. These are combined because the stacking refit is the primary consumer of advanced caching, and the snapshot cache needs the stacking execution flow to be validated.

**Depends on**: Phases 1 (check-before-fit) and 2 (simple refit, orchestrator two-pass flow)

### Task 3.1 — StepCache: In-Memory Preprocessed Data Snapshots

**What**: Implement the `StepCache` that stores the full `SpectroDataset` state after each preprocessing step, keyed by `(chain_path_hash, data_hash)`.

**Why**: For generator sweeps with shared preprocessing prefixes and for the refit pass replaying the same chain, caching preprocessed dataset snapshots eliminates redundant computation at the dataset level (not just the operator level).

**Details**:
- Create `nirs4all/pipeline/execution/step_cache.py`.
- `StepCache` class with:
  - `__init__(max_size_mb: int = 2048)` — configurable memory limit.
  - `get(chain_path_hash: str, data_hash: str) -> SpectroDataset | None` — returns a copy of the cached dataset state, or None.
  - `put(chain_path_hash: str, data_hash: str, dataset: SpectroDataset)` — stores a copy. LRU eviction if over memory limit.
  - Memory tracking: estimate dataset size from `X.nbytes` for each source/processing.
- Integration point: `PipelineExecutor._execute_single_step()`:
  - Before executing: check `step_cache.get(chain_path, data_hash)`.
  - If hit: restore dataset state from cache, skip step execution entirely.
  - If miss: execute step, then `step_cache.put(chain_path, data_hash, dataset)`.
- The `StepCache` instance lives in the `PipelineOrchestrator` scope (same lifetime as `ArtifactRegistry`), shared across all variants and the refit pass.
- **Reuse the existing `DataCache` class** (`nirs4all/data/performance/cache.py`) as the underlying storage backend. It is already thread-safe, has LRU eviction, configurable size limits, and staleness detection. Wrapping it with a step-specific key scheme avoids duplicating cache infrastructure (per caching analysis Section 6.5 recommendation).

**Touches**: new file `nirs4all/pipeline/execution/step_cache.py`, `nirs4all/pipeline/execution/executor.py`, `nirs4all/pipeline/execution/orchestrator.py`

**Acceptance**: A generator sweep `[SNV(), {"_or_": [PLS(5), PLS(10), PLS(15)]}]` — SNV step is cached after variant 0 and reused for variants 1 and 2 (verified by log or counter). Memory stays under the configured limit.

### Task 3.2 — Per-Model Selection Logic

**What**: After Pass 1, for each model node in the pipeline, identify the variant where that model achieved the best val score (independently of other models).

**Why**: Section 3.10 requires per-model independent refit. In stacking with generators, different base models may have their best variant differ from the meta-model's best variant.

**Details**:
- Create `nirs4all/pipeline/execution/refit/model_selector.py`.
- `select_best_per_model(predictions, topology, variant_configs) -> dict[str, PerModelSelection]`:
  1. Enumerate all model nodes from `topology.model_nodes`.
  2. For each model node, filter prediction entries by model name/branch path.
  3. For each model, find the variant with the best val_score for that model.
  4. Return a mapping: `model_name → PerModelSelection(variant_index, best_params, expanded_steps, branch_path)`.
- Handle edge cases:
  - Single variant (no generators): all models share the same variant.
  - Branches without merge: models in different branches are alternatives, not cooperating.
  - Branches with merge: all branch models are cooperating; the meta-model's winning variant determines the stacking context.

**Touches**: new file `nirs4all/pipeline/execution/refit/model_selector.py`

**Acceptance**: For the canonical example (SNV/MSC × PLS/RF × Ridge), per-model selection correctly identifies PLS's best variant independently from RF's best variant.

### Task 3.3 — Stacking Refit: Two-Step Revised Design

**What**: Implement the revised stacking refit from Section 8.8: (Step 1) retrain base models on all data, (Step 2) train meta-model on base model predictions.

**Why**: The stacking refit handles pipelines with `merge: "predictions"`, covering ~15% of use cases.

**Details**:
- Create `nirs4all/pipeline/execution/refit/stacking_refit.py`.
- `execute_stacking_refit(refit_config, dataset, context, artifact_registry, topology) -> RefitResult`:
  - **Step 1 — Base model refit**:
    1. For each base model branch in the winning variant:
       a. Clone the preprocessing chain.
       b. Fit preprocessing on all training data (via standard `_execute_steps()` — benefits from Phase 1 caching).
       c. Clone the model with winning params + `refit_params` overrides.
       d. Train model on all preprocessed training data.
       e. Generate predictions on the training data (in-sample predictions become meta-features).
       f. Persist model as `fold_id="final"`, `refit_context="stacking"`.
    2. GPU-aware serialization (Section 8.4.7): if any base model is GPU-backed, run sequentially. Otherwise, parallelize.
  - **Step 2 — Meta-model refit**:
    1. Collect base model in-sample predictions into a feature matrix.
    2. For classification: probability columns (N_train × N_base_models × N_classes).
    3. Train meta-model on prediction features + training targets.
    4. Evaluate: base models predict on test → meta-model predicts → test score.
    5. Persist meta-model as `fold_id="final"`, `refit_context="stacking"`.
- Optional OOF meta-training: if `refit_config.stacking_meta_training == "oof"`, use Pass 1's OOF predictions for meta-model training (default: in-sample).
- Integrate with the orchestrator: in Task 2.5's stacking dispatch path, call `execute_stacking_refit()` instead of logging "not implemented".

**Touches**: new file `nirs4all/pipeline/execution/refit/stacking_refit.py`, `nirs4all/pipeline/execution/orchestrator.py`

**Acceptance**: A stacking pipeline `[KFold(5), branch([SNV+PLS, MSC+RF]), merge("predictions"), Ridge()]` produces a `fold_id="final"` Ridge entry with correct test score. Deployment prediction flow uses refit base models + refit meta-model.

### Task 3.4 — Mixed Merge Refit (Hybrid)

**What**: Handle pipelines with `merge: {"features": [...], "predictions": [...]}` where some branches contribute features (no OOF dependency) and others contribute predictions (OOF dependency).

**Why**: Section 4.9 describes the hybrid case. Feature branches use simple refit, prediction branches use stacking refit.

**Details**:
- In `execute_stacking_refit()`, distinguish between branch types:
  - Branches in the `"features"` list: simple refit (fit preprocessing on all data, compute features).
  - Branches in the `"predictions"` list: base model refit (Step 1 of stacking refit).
- Merge the two outputs: feature branches contribute feature columns, prediction branches contribute prediction columns. Train meta-model on the combined features.

**Touches**: `nirs4all/pipeline/execution/refit/stacking_refit.py`

**Acceptance**: A mixed merge pipeline produces correct refit results where feature branches are not treated as stacking branches.

### Task 3.5 — GPU-Aware Serialization for Parallel Refits

**What**: During the stacking refit's base model refit step, detect GPU-backed models and serialize their training instead of running in parallel.

**Why**: Section 8.4.7 identifies that concurrent GPU model training causes OOM crashes.

**Details**:
- Add `_is_gpu_model(model_config) -> bool` check that detects TensorFlow, PyTorch, JAX model types.
- In the base model refit loop: if any model is GPU-backed, run all refits sequentially. If all are CPU-only, parallelize with joblib.
- After each GPU model refit, call the framework's memory cleanup (`torch.cuda.empty_cache()`, etc.).

**Touches**: `nirs4all/pipeline/execution/refit/stacking_refit.py`, `nirs4all/utils/backend.py` (if needed for detection)

**Acceptance**: A stacking refit with one GPU model and one CPU model runs sequentially without GPU OOM.

---

## 6. Phase 4 — Advanced Refit

**Goal**: Handle the remaining 5% of refit cases and add optimization features.

**Depends on**: Phase 3 (stacking refit, per-model selection)

### Task 4.1 — Nested Stacking Refit (Recursive)

**What**: Handle stacking pipelines that contain stacking sub-pipelines (Section 4.11 "The Monster").

**Why**: Recursive stacking requires the refit to be applied at each level of nesting.

**Details**:
- Refactor `execute_stacking_refit()` to be recursive:
  - Before refitting a base model branch, check if that branch itself contains stacking (i.e., a `merge: "predictions"` within the branch).
  - If yes: recursively call `execute_stacking_refit()` for the inner stacking pipeline.
  - If no: simple base model refit.
- Add a depth limit (default 3) with a warning if exceeded.
- Track recursion depth for logging and progress reporting.

**Touches**: `nirs4all/pipeline/execution/refit/stacking_refit.py`

**Acceptance**: A nested stacking pipeline (stacking within a stacking branch) produces correct refit results. Depth limit warning emitted at depth 3.

### Task 4.2 — Separation Branch and Multi-Source Refit (Per-Branch)

**What**: Handle pipelines with `by_metadata`, `by_tag`, or `by_source` separation branches where each branch has its own data subset and/or model.

**Why**: Section 4.13 (separation) and Section 4.14 (multi-source) require per-branch refit — each branch is refitted independently on its own full training subset. Multi-source `by_source` pipelines with inner models producing predictions are structurally equivalent to stacking and need the stacking refit treatment.

**Details**:
- In the topology analyzer (Task 0.3), detect separation branches.
- For separation branches:
  1. Split the full training data by the separation criterion (metadata column, tag value, source).
  2. For each branch: refit the branch's model on that branch's full training subset.
  3. Persist per-branch refit artifacts with `fold_id="final"` and branch identification.
- Handle `merge: "concat"` after separation: reassemble predictions in original sample order for test evaluation.
- Handle the leakage case (Section 8.2.6): if separation + concat + outer model, treat as stacking topology.
- **Multi-source `by_source` pipelines** (Section 4.14): when `by_source` branches have inner models whose predictions are merged (e.g., `{"merge": {"sources": "concat"}}` followed by an outer model), this is structurally stacking. Apply the stacking refit: retrain per-source base models on all data → generate in-sample predictions → train outer model.
- When `by_source` branches perform preprocessing only (no inner model), this is simple feature merge — apply simple refit.

**Touches**: `nirs4all/pipeline/execution/refit/executor.py`, `nirs4all/pipeline/execution/refit/stacking_refit.py`

**Acceptance**: A separation pipeline `[KFold(5), branch(by_metadata="site"), PLS(10), merge("concat")]` produces per-site refit models. A multi-source pipeline `[branch(by_source=True, steps={"NIR": [SNV, PLS], "markers": [MinMax, Ridge]}), merge(sources="concat"), GBR()]` uses stacking refit for the outer model. Prediction routes new samples to the correct branch/source model.

### Task 4.3 — Lazy Per-Model Standalone Refits

**What**: Make standalone per-model refits lazy (computed on demand) instead of executing all of them eagerly.

**Why**: Section 8.5.1 identifies that eagerly refitting every model node is expensive. For a stacking pipeline with 5 base models, the default should refit only the outermost model. Individual base model refits are triggered on access.

**Details**:
- During the refit phase, store the `PerModelSelection` metadata (winning variant, best_params, chain config) for each model node without executing the refit.
- `result.models["PLS"]` returns a `LazyModelRefitResult` that triggers `execute_simple_refit()` on first access to `.score`, `.export()`, or `.chain`.
- The lazy refit uses the same `ArtifactRegistry` and `StepCache` (which should still be in scope if the result object holds a reference).
- Edge case: if the user accesses `result.models["PLS"]` after the run has completed and the registry/cache are destroyed, the lazy refit must re-execute from scratch (slower but correct).

**Touches**: `nirs4all/api/result.py`, `nirs4all/pipeline/execution/refit/model_selector.py`

**Acceptance**: `result.models["PLS"].score` triggers a refit on first access. `result.final_score` does NOT trigger standalone refits for base models.

### Task 4.4 — Warm-Start Refit for Deep Learning

**What**: Support initializing the refit model from the best fold model's weights instead of training from random initialization.

**Why**: Section 8.5.3 describes that warm-starting from a fold model converges 3-5x faster for neural networks.

**Details**:
- In the deep learning model controllers (`TensorFlowModelController`, `PyTorchModelController`, `JAXModelController`):
  1. During refit, check `resolved_params.get("warm_start", False)`.
  2. If True: load the best fold model's weights from disk (fold artifacts are still on disk at this point — cleanup happens after refit).
  3. Determine which fold: `warm_start_fold` param (`"best"` = best val score fold, `"last"` = last fold, `"fold_N"` = specific fold).
  4. Inject weights into the refit model instance before training.
  5. Train on all training data with refit-specific params (typically fewer epochs, lower LR).
- For sklearn models: check for `warm_start` support (`hasattr(model, 'warm_start')`) and set `model.set_params(warm_start=True)` if the `refit_params` request it.

**Touches**: deep learning model controllers in `nirs4all/controllers/models/`, `nirs4all/controllers/models/base_model.py`

**Acceptance**: A PyTorch model with `refit_params={"warm_start": True, "epochs": 30}` initializes from the best fold model's weights and trains for 30 epochs on all data. Training converges faster than cold-start (verified by loss curves).

### Task 4.5 — Branches Without Merge: Winner Selection

**What**: Handle the case where branches have no merge step (competing alternatives) — select and refit only the winning branch.

**Why**: Section 4.12 specifies that branches without merge are alternatives. Only the best-performing branch is refitted.

**Details**:
- In the topology analyzer, detect branches without a following merge step (Task 0.3 already captures this).
- During refit config extraction: identify the winning branch by comparing per-branch prediction scores.
- Refit only the winning branch's pipeline on all training data.

**Touches**: `nirs4all/pipeline/execution/refit/config_extractor.py`, `nirs4all/pipeline/execution/refit/executor.py`

**Acceptance**: A pipeline with two competing branches (no merge) refits only the winning branch. The losing branch's model is not persisted.

---

## 7. Phase 5 — Webapp Integration + Cross-Run Caching

**Goal**: Integrate the refit and caching features into the webapp UI, and extend caching for cross-run persistence.

**Depends on**: Phase 4 (all refit features complete)

### Task 5.1 — WebSocket Events for Refit Phase

**What**: Emit distinct WebSocket events during the refit phase so the webapp can show progress.

**Why**: Section 8.4.5 specifies that the RunProgress page should show a "Refit" phase indicator.

**Details**:
- Define new WebSocket message types: `REFIT_STARTED`, `REFIT_PROGRESS`, `REFIT_STEP`, `REFIT_COMPLETED`, `REFIT_FAILED`.
- Emit `REFIT_STARTED` when the refit phase begins (after Pass 1 completes).
- Emit `REFIT_STEP` for each refit step (preprocessing, model training, evaluation).
- Emit `REFIT_COMPLETED` with the refit score, or `REFIT_FAILED` with error details.
- The job manager tracks refit as a sub-phase of the training job (not a separate job).

**Touches**: `nirs4all-webapp/api/training.py`, `nirs4all-webapp/websocket/manager.py`

**Acceptance**: The webapp's WebSocket connection receives refit events during a training job.

### Task 5.2 — RunProgress Page: Refit Phase Indicator

**What**: Add a "Refit" phase indicator to the RunProgress page in the webapp.

**Why**: Users need visual feedback that the refit is running and its progress.

**Details**:
- In `RunProgress.tsx`, add a phase indicator after the CV phase completes.
- Show: "Refitting best model on all training data..." with progress bar.
- For stacking: show sub-steps ("Refitting base model 1/3...", "Training meta-model...").
- Show refit score when complete.

**Touches**: `nirs4all-webapp/src/pages/RunProgress.tsx`

**Acceptance**: The RunProgress page shows the refit phase with progress during a training run.

### Task 5.3 — Results Page: Dual Scoring Display

**What**: Update the Results page to display both CV scores and refit scores, clearly labeled.

**Why**: Users need to see both the selection metric (CV) and the deployment metric (refit test score).

**Details**:
- In `Results.tsx` / `Analysis.tsx`, add columns/sections for:
  - "CV Score" (val_score from best CV entry).
  - "Final Score" (test_score from refit entry).
  - "Final Model" indicator (which model is the deployment model).
- Default export target: the refit model.
- Per-model access: show standalone refit scores for each model node (from `result.models`).

**Touches**: `nirs4all-webapp/src/pages/Results.tsx`, `nirs4all-webapp/src/pages/Analysis.tsx`

**Acceptance**: The Results page shows both CV and refit scores for a completed training run with `refit=True`.

### Task 5.4 — Cross-Run Caching (DuckDB-Backed)

**What**: Extend the step-level cache to persist across runs using the DuckDB workspace store.

**Why**: Users often run similar pipelines on the same dataset. Cross-run caching avoids refitting preprocessing that was already computed in a previous run.

**Details**:
- Extend `ArtifactRecord` with an `input_data_hash VARCHAR` field.
- Store the cache key `(chain_path_hash, input_data_hash)` alongside each artifact record.
- At the start of a run, query the workspace store for matching artifacts before fitting.
- Add cache invalidation logic: if the dataset content hash has changed (file modified), invalidate all cached artifacts for that dataset.
- This is a low-priority optimization (P5 in the caching analysis). Implement only after in-memory caching is proven.

**Touches**: `nirs4all/pipeline/storage/store_schema.py`, `nirs4all/pipeline/storage/workspace_store.py`, `nirs4all/pipeline/storage/artifacts/artifact_registry.py`

**Acceptance**: A preprocessing step fitted in run #1 is reused (not re-fitted) in run #2 when the dataset and chain are identical. Cache miss when the dataset changes.

### Task 5.5 — Backend API Updates for Refit Configuration

**What**: Expose the `refit` configuration in the webapp backend API.

**Why**: Users need to configure refit behavior (enable/disable, `refit_params`, `stacking_meta_training`) from the webapp UI.

**Details**:
- Add `refit` field to the training configuration API endpoints.
- Validate the `refit` config in the backend (Pydantic schema).
- Pass through to `nirs4all.run()`.
- In the pipeline editor: add refit-specific configuration options to model step renderers (for `refit_params`).

**Touches**: `nirs4all-webapp/api/training.py`, `nirs4all-webapp/api/pipelines.py`, `nirs4all-webapp/src/components/pipeline-editor/`

**Acceptance**: The webapp can start a training job with `refit=True` and custom `refit_params`.

---

## 8. Dependency Graph

```
Phase 0: Shared Foundation
  Task 0.1 (data hashing)
  Task 0.2 (execution phase)
  Task 0.3 (topology analyzer)
  Task 0.4 (ArtifactRegistry cache-key support)
  Task 0.5 (registry lifespan)
  Task 0.6 (RepeatedKFold OOF bug fix)
         │
    ┌────┴────────────────────────┐
    ▼                             ▼
Phase 1: Core Caching          Phase 2: Simple Refit
  Task 1.1 (check-before-fit)    Task 2.1 (prediction schema)
    ← needs 0.1, 0.4             Task 2.2 (refit_params parser)
  Task 1.2 (fit_on_all valid.)   Task 2.3 (winning config extraction)
    ← needs 1.1                  Task 2.4 (simple refit execution)
  Task 1.3 (stateless skip)        ← needs 0.2, 0.3
    ← needs 1.1                  Task 2.5 (orchestrator two-pass)
  Task 1.4 (cross-pipeline)        ← needs 2.3, 2.4, 0.5
    ← needs 1.1                  Task 2.6 (fold artifact lifecycle)
                                    ← needs 2.4
                                 Task 2.7 (result object)
                                    ← needs 2.5
                                 Task 2.8 (bundle export)
                                    ← needs 2.5
                                 Task 2.9 (prediction mode dispatch)
                                    ← needs 2.8
                                 Task 2.10 (refit metadata)
                                    ← needs 2.4
                                 Task 2.11 (default + compat)
                                    ← needs 2.7, 2.8, 2.9
    │                             │
    └──────────┬──────────────────┘
               ▼
Phase 3: Advanced Caching + Stacking Refit
  Task 3.1 (StepCache)
    ← needs 1.1, 2.5
  Task 3.2 (per-model selection)
    ← needs 0.3, 2.3
  Task 3.3 (stacking refit)
    ← needs 2.4, 2.5, 3.2
  Task 3.4 (mixed merge refit)
    ← needs 3.3
  Task 3.5 (GPU-aware serialization)
    ← needs 3.3
               │
               ▼
Phase 4: Advanced Refit
  Task 4.1 (nested stacking)
    ← needs 3.3
  Task 4.2 (separation + multi-source refit)
    ← needs 2.4, 0.3, 3.3
  Task 4.3 (lazy per-model refits)
    ← needs 3.2, 2.7
  Task 4.4 (warm-start DL)
    ← needs 2.4, 2.2
  Task 4.5 (branches without merge)
    ← needs 2.3, 0.3
               │
               ▼
Phase 5: Webapp + Cross-Run
  Task 5.1 (WebSocket events)
    ← needs 2.5
  Task 5.2 (RunProgress UI)
    ← needs 5.1
  Task 5.3 (Results dual scoring)
    ← needs 2.7
  Task 5.4 (cross-run cache)
    ← needs 1.1, 0.4
  Task 5.5 (backend API)
    ← needs 2.5
```

---

## 9. Summary

| Phase | Tasks | Key Deliverable | User Impact |
|---|---|---|---|
| **0** | 6 | Foundation: data hashing, execution phase, topology analyzer, registry extensions, OOF bug fix | None (internal infrastructure) |
| **1** | 4 | Core caching: check-before-fit, stateless skip, cross-pipeline reuse | Generator sweeps run significantly faster |
| **2** | 11 | Simple refit: two-pass, fold_id="final", result object, bundle export, prediction mode, refit_params, metadata | 80% of pipelines produce scientifically correct deployment models |
| **3** | 5 | StepCache + stacking refit: snapshot cache, per-model selection, stacking two-step | 95% of pipelines covered; generator sweeps + stacking fully optimized |
| **4** | 5 | Advanced refit: nested, separation/multi-source, lazy, warm-start, no-merge branches | 100% of pipeline topologies covered; DL models refit efficiently |
| **5** | 5 | Webapp integration + cross-run caching | Full UI support; repeated experiments benefit from persistent cache |
| **Total** | **36** | | |

### Critical Path

The minimum path to deliver value is:
1. **Phase 0** (foundation) → **Phase 2** (simple refit): delivers the core refit feature for 80% of pipelines.
2. **Phase 0** → **Phase 1** (core caching): delivers standalone performance improvements for generator sweeps.
3. Phases 1 + 2 → **Phase 3** (stacking refit + advanced cache): completes 95% coverage.

Phases 4 and 5 are incremental and can be deferred or parallelized.
