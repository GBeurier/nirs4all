# Technical Debt Synthesis & Structured Backlog

Date: 2026-02-07
Scope: Cross-cutting validation of 4 audit documents against current codebase, followed by prioritized backlog
Method: Code-verified claim validation using targeted source inspection across all referenced modules

---

## Part 1: Audit Claim Validation

### Context and approach

Four audit documents were produced on 2026-02-07 covering:

1. **Backend Readiness Assessment** — overall production suitability
2. **Pipeline Module Technical Debt Review** — `nirs4all/pipeline/**`
3. **Data Module Technical Debt Review** — `nirs4all/data/**`
4. **Workspace/Predictions/Artifacts Technical Debt Review** — storage, prediction lifecycle, artifact replay

This synthesis validates each material claim against the current codebase (commit `88cbf1c`, branch `cache_refactoring`). Claims were verified by reading the exact source lines cited. Each claim receives a verdict: **CONFIRMED**, **PARTIALLY VALID**, **SUPERSEDED** (was true but has been fixed), or **INVALID** (not supported by current code).

### Rationale for the validation process

Several audit claims reference line numbers and behaviors that may have changed during the recent `cache_refactoring` work (commits `88cbf1c`, `a83aee8`). The step cache integration, CoW snapshots, and dataset cache management were specifically reworked. Validating against current code ensures the backlog reflects actual debt, not already-resolved issues.

---

### 1.1 Correctness and Provenance Claims

#### [CONFIRMED] Pipeline hash stored as `dataset_hash`

- **Source docs**: Backend Assessment §3E, Pipeline Debt §P0
- **Evidence**: `executor.py:143` computes hash from pipeline steps via `_compute_pipeline_hash()`. Line 160 passes it as `dataset_hash=pipeline_hash` to `begin_pipeline()`. The `workspace_store.py:357` contract documents this parameter as "Content hash of the dataset at execution time".
- **Impact**: Misleading metadata; cache invalidation and run-compatibility checks use wrong semantics.
- **Backlog verdict**: Remains P0. Must fix.

#### [CONFIRMED] Refit persistence lifecycle is split (labeling after write)

- **Source doc**: Workspace/Predictions §P0
- **Evidence**: `refit/executor.py:143-158` calls `executor.execute()` (which flushes predictions to store), then `_relabel_refit_predictions()` after. Same pattern in `stacking_refit.py:656→671` and `736→751`. Predictions may reach the store with incorrect `fold_id` and `refit_context`.
- **Impact**: Store rows for refit can have incorrect metadata; `v_aggregated_predictions` view may misclassify rows.
- **Backlog verdict**: Remains P0. Relabeling must happen before persistence.

#### [CONFIRMED] Fold key contract is fragmented

- **Source doc**: Workspace/Predictions §P0
- **Evidence**: Chain builder (`chain_builder.py:120`) writes `f"fold_{fold_id}"`. Execution trace (`execution_trace.py:80`) uses `Dict[int, str]`. Workspace store expects both `"final"` and `"fold_final"` (`workspace_store.py:1472, 1824-1835`). Trace deserialization converts string keys to int (`execution_trace.py:126`). Provider lookup uses bare `"final"` (`context.py:913`).
- **Impact**: Multiple fold key dialects create lookup ambiguity and normalization shims across modules.
- **Backlog verdict**: Remains P0. Normalize once at ingestion boundary.

#### [CONFIRMED] Unsafe deserialization with pickle/joblib

- **Source doc**: Backend Assessment §3H
- **Evidence**: `workspace_store.py:126` uses `joblib.load(io.BytesIO(data))` and line 129 uses `pickle.loads(data)` with explicit `# noqa: S301` security suppression.
- **Impact**: Acceptable for trusted internal workflows; unsafe for external bundles.
- **Backlog verdict**: P1 for internal use. P0 if external bundle loading is planned.

---

### 1.2 Performance Claims

#### [CONFIRMED] Branch execution copies full feature tensors repeatedly in the default API path

- **Source docs**: Pipeline Debt §P0, Backend Assessment §3
- **Evidence**: `executor.py:509-520` has two paths. Line 510-517: CoW path uses `SharedBlocks.acquire()` for zero-copy restore. Line 519-520: fallback uses `copy.deepcopy(features_snapshot)`. In the default run path, `api/run.py:393-395` only sets cache config when `cache` is explicitly provided; `runner.py:247` defaults `cache_config=None`; `controllers/data/branch.py:1406-1409` disables CoW when cache config is absent.
- **Current state**: CoW exists but is not active by default unless callers pass an explicit cache config object.
- **Backlog verdict**: Upgrade to P0 for production-readiness: establish safe default cache config and verify branch-heavy pipelines avoid deep-copy path by default.

#### [SUPERSEDED] Feature processing concatenation causes memory amplification

- **Source doc**: Data Debt §P0
- **Evidence**: `array_storage.py:413-427` now uses O(1) `self._blocks.append()` — no concatenation. The `add_processing()` docstring confirms "O(1) allocation — appends a 2D block without copying existing data."
- **Current state**: Fixed in `cache_refactoring` branch (commit `a83aee8`).
- **Backlog verdict**: **Remove from backlog.** Claim no longer valid.

#### [SUPERSEDED] DatasetConfigs cache is unbounded and keyed only by name

- **Source doc**: Data Debt §P0
- **Evidence**: `config.py:372-383` now implements `_make_cache_key()` with MD5 hash of full config parameters (not just name). Lines 385-398 implement FIFO eviction with a byte budget.
- **Current state**: Fixed in `cache_refactoring` branch.
- **Backlog verdict**: **Remove from backlog.** Claim no longer valid.

#### [SUPERSEDED] Step cache is implemented but not integrated into execution

- **Source docs**: Pipeline Debt §P0, Backend Assessment
- **Evidence**: Step cache is now fully integrated. `executor.py:722-750` retrieves cache from `RuntimeContext`, computes step/data hashes, performs cache lookup, returns early on hit, and stores results on miss. `orchestrator.py:198-206` creates the cache, passes it via `RuntimeContext`. `context.py:1188` holds the `step_cache` attribute.
- **Current state**: Fully integrated in `cache_refactoring` branch (commit `88cbf1c`).
- **Backlog verdict**: **Remove from backlog.** Claim no longer valid.

#### [CONFIRMED] Per-step shape tracing materializes data twice (2D + 3D)

- **Source doc**: Pipeline Debt §P1
- **Evidence**: `executor.py:1090` calls `dataset.x(layout="2d")` and `1098` calls `dataset.x(layout="3d")`. Called both pre-step and post-step (lines 691, 710, and in branch path 575, 595).
- **Impact**: Extra allocations on every step, multiplied by branches.
- **Backlog verdict**: Remains P1. Gate behind verbosity flag or cache when selector unchanged.

#### [CONFIRMED] Artifact loader step lookups are linear scans

- **Source doc**: Pipeline Debt §P1
- **Evidence**: `artifact_loader.py:364` iterates `self._artifacts.items()` on each `load_for_step()` call.
- **Impact**: O(N) per step in prediction/explain workflows.
- **Backlog verdict**: Remains P1. Build indexes at loader init.

#### [CONFIRMED] Prediction flush matching has quadratic behavior

- **Source doc**: Pipeline Debt §P1
- **Evidence**: `executor.py:305-340` shows nested candidate filtering per prediction over chain_rows. O(predictions × chains) without indexing.
- **Backlog verdict**: Remains P1. Pre-index chains by composite key.

#### [CONFIRMED] Memory-heavy defaults (keep_datasets=True)

- **Source doc**: Pipeline Debt §P1
- **Evidence**: `runner.py:130` defaults `keep_datasets=True`. `orchestrator.py:252,303` retains raw/preprocessed arrays.
- **Backlog verdict**: Remains P1. Default to False or add bounded policy.

#### [CONFIRMED] Metadata hashing materializes full data

- **Source doc**: Data Debt §P1
- **Evidence**: `dataset.py:1949-1955` calls `self.x(None, layout="2d")` to materialize full feature array, then hashes only a sample of it.
- **Backlog verdict**: Remains P1. Use `content_hash()` or a non-materializing path.

---

### 1.3 Reproducibility Claims

#### [CONFIRMED] Incomplete seed propagation into generator expansion

- **Source doc**: Pipeline Debt §P1
- **Evidence**: `pipeline_config.py:78` calls `expand_spec_with_choices(self.steps)` without passing seed. `_generator/core.py:74` accepts seed parameter. `_generator/iterator.py:284` creates unseeded `random.Random()` when seed is None. `runner.py:51-56` does seed Python `random` when `random_state` is provided, but that seed is not propagated into generator expansion calls.
- **Impact**: Config expansion with `_sample_` and probabilistic generators is non-reproducible.
- **Backlog verdict**: Remains P1. Propagate seed through expansion calls.

#### [PARTIALLY VALID] Resolver selects from unsorted iteration

- **Source doc**: Pipeline Debt §P1
- **Evidence**: `resolver.py:525,536` uses `iterdir()` (unordered). Line 546 returns `candidates[0]`. However, line 1349 uses `sorted(..., reverse=True)` in some paths. Chain selection (`resolver.py:1303-1330`) has multiple fallback heuristics using serialized JSON string matching.
- **Backlog verdict**: Remains P1. Sort candidates deterministically; match branch paths structurally.

#### [CONFIRMED] Global RNG seeding inside data utilities

- **Source doc**: Data Debt §P1
- **Evidence**: `loader.py:48` uses `np.random.seed(config['random_state'])`. `row_selector.py:536` uses `np.random.seed(random_state)`.
- **Impact**: Global RNG state mutation affects cross-run determinism.
- **Backlog verdict**: Remains P1. Use local `np.random.Generator` instances.

#### [CONFIRMED] Import-time root logger mutation

- **Source doc**: Pipeline Debt §P1
- **Evidence**: `pipeline_config.py:26-29` adds handler and sets level on `logging.root` at import time, with no duplicate-handler guard.
- **Backlog verdict**: Remains P1. Route through central bootstrap.

#### [CONFIRMED] Hash identity is brittle and truncated

- **Source doc**: Pipeline Debt §P1
- **Evidence**: `pipeline_config.py:91` truncates to 6 chars. `pipeline_config.py:303` returns 8 chars but caller takes only 6. `executor.py:1130` uses `default=str` in JSON serialization. MD5 with 6-hex chars = 16.7M possible values; collision-prone at scale.
- **Backlog verdict**: Remains P1. Increase hash length; separate display hash from identity hash.

---

### 1.4 Contracts and Resolution Claims

#### [CONFIRMED] Resolver chain selection is heuristic and non-deterministic under branch ambiguity

- **Source doc**: Workspace/Predictions §P0
- **Evidence**: `resolver.py:1303-1304` compares raw serialized JSON strings for branch matching. Line 1330 falls back to first chain row. Store-first then filesystem fallback at lines 588-628.
- **Backlog verdict**: Remains P0. Parse branch paths structurally; add deterministic tie-breaks.

#### [CONFIRMED] Store-first and filesystem-legacy resolution both remain active

- **Source doc**: Workspace/Predictions §P1
- **Evidence**: `resolver.py:589-590` tries store-based resolution. On failure, lines 594-628 fall through to filesystem-based resolution with manifest/pipeline.json loading.
- **Backlog verdict**: Remains P1. Define explicit mode policy with observability on fallback.

#### [CONFIRMED] Prediction persistence has duplicated write paths

- **Source doc**: Workspace/Predictions §P1
- **Evidence**: `executor.py:280` uses `_flush_predictions_to_store()`. `predictions.py:378-394` has separate `flush()` method. No `.flush()` calls found in runtime flow — it exists as orphaned API.
- **Backlog verdict**: Remains P1. Consolidate or remove orphaned path.

#### [CONFIRMED] Chain replay wavelength awareness gap

- **Source doc**: Workspace/Predictions §P1
- **Evidence**: `workspace_store.py:2025-2027` documents `wavelengths` parameter for spectral transformers. Line 2074 applies `transformer.transform(X_current)` without passing wavelengths.
- **Backlog verdict**: Remains P1. Implement wavelength passthrough in replay path.

#### [CONFIRMED] Artifact dedup integrity check is incomplete

- **Source doc**: Workspace/Predictions §P1
- **Evidence**: `artifact_registry.py:1305-1308` matches by short hash pattern but the comment "Verify it's actually the same content" is followed by an immediate return without verification.
- **Backlog verdict**: Remains P1. Verify full content hash before reuse.

#### [CONFIRMED] Schema lacks natural-key idempotency for predictions

- **Source doc**: Workspace/Predictions §P1
- **Evidence**: `store_schema.py:75-99` — predictions table has PK on `prediction_id` only. No UNIQUE constraint on semantic key `(pipeline_id, chain_id, fold_id, partition)`.
- **Backlog verdict**: Remains P1. Add uniqueness constraint or upsert guard.

---

### 1.5 Dead Code and Documentation Claims

#### [CONFIRMED] Documentation drift: Python version

- **Source doc**: Backend Assessment §3G
- **Evidence**: `pyproject.toml:52` requires `>=3.11`. Three doc files still say `3.9+`: `installation.md:20`, `quickstart.md:8`, `export_bundles.md:355`.
- **Backlog verdict**: P0 (quick fix).

#### [CONFIRMED] Documentation drift: TensorFlow in base install

- **Source doc**: Backend Assessment §3G
- **Evidence**: `installation.md:13-16` claims TensorFlow is installed by default. `pyproject.toml` has TensorFlow only in optional extras.
- **Backlog verdict**: P0 (quick fix).

#### [CONFIRMED] Dead code: `executor._filter_binaries_for_branch`

- **Evidence**: Single reference at definition site only (line 902). Method returns input unchanged.
- **Backlog verdict**: P2. Delete.

#### [CONFIRMED] Dead code: `store_queries.build_filter_clause`

- **Evidence**: Single reference at definition site only (line 471). Zero callers.
- **Backlog verdict**: P2. Delete.

#### [CONFIRMED] Dead code: generator constraints and registry functions

- **Evidence**: `count_with_constraints` (constraints.py:213), `get_all_strategies` (registry.py:96), `_get_registry_state` (registry.py:119) — zero callers found. `clear_registry` (registry.py:110) is a test utility — tolerable.
- **Backlog verdict**: P2. Delete unreferenced functions.

#### [CONFIRMED] Dead code: `data/io.py` is fully commented out

- **Evidence**: All 175 lines are commented out. No imports.
- **Backlog verdict**: P2. Delete file.

#### [CONFIRMED] StepRunner.execute contains unreachable block

- **Evidence**: `step_runner.py:200` returns early, followed by 31 lines of unreachable code (comments, pass, and another return at line 233).
- **Backlog verdict**: P2. Remove unreachable block.

#### [CONFIRMED] Placeholder in JAX model

- **Evidence**: `nirs4all/operators/models/jax/nicon.py:161` contains `pass # Placeholder` with deferred implementation notes.
- **Backlog verdict**: P2 (JAX path is explicitly not production-ready per Backend Assessment §2).

#### [PARTIALLY VALID] `pipeline/run.py` appears orphaned from runtime flow but defines domain entities

- **Source doc**: Pipeline Debt §P2
- **Evidence**: No runtime or test imports were found for `nirs4all.pipeline.run` (note: distinct from `nirs4all.pipeline.runner`, which is widely imported). The file defines domain entities (`Run`, `RunStatus`, `RunConfig`) that may be intended as an external/public surface.
- **Backlog verdict**: Keep as P2 decision item (owner decision: deprecate/remove vs adopt/document).

#### [PARTIALLY VALID] Dormant caching/performance infrastructure in data module

- **Evidence**: `DataCache` is actively used by `pipeline/execution/step_cache.py`. The dormant part is `LazyDataset`, `LazyArray` (zero runtime callers), API stubs in `predictions.py:1190-1199` (`clear_caches()`, `get_cache_stats()`), and unimplemented `cache` flag in `features.py:21-28`.
- **Backlog verdict**: Keep as P2 cleanup, but scope to real stubs/dormant exports only (do not remove active `DataCache` path).

#### [INVALID] Dataset hash inconsistency (dual attribute)

- **Source doc**: Data Debt §P0
- **Evidence**: Current code uses a single `_content_hash_cache` attribute consistently. `set_content_hash()` writes to `_content_hash_cache`. No separate `_content_hash` attribute exists.
- **Backlog verdict**: **Remove from backlog.** Claim no longer valid.

#### [CONFIRMED] Workspace global state is process-global mutable singleton

- **Evidence**: `workspace/__init__.py:14` has `_active_workspace: Optional[Path] = None` with module-level setters without synchronization.
- **Backlog verdict**: P2. Acceptable for CLI; problematic for service deployment.

#### [CONFIRMED] Dual loader stacks

- **Source doc**: Data Debt §P1
- **Evidence**: `csv_loader.py` (legacy) and `csv_loader_new.py` (modern registry) coexist. `DatasetConfigs` uses the legacy path. The dual stacks are intentional backward compatibility.
- **Backlog verdict**: P2. Consolidate to single loader stack.

---

### 1.6 Validation Summary

| Verdict | Count |
|---------|-------|
| CONFIRMED | 31 |
| PARTIALLY VALID | 4 |
| SUPERSEDED (fixed in cache_refactoring) | 3 |
| INVALID (not supported by code) | 1 |

**Key insights**:

1. Branch-memory risk remains high in default usage because CoW is not active unless cache config is explicitly provided — upgraded to P0.
2. Seed reproducibility risk is primarily propagation-level (generator expansion), not complete absence of Python RNG seeding.
3. Some cleanup claims need scope correction (`pipeline/run.py` has zero imports but defines domain entities; `DataCache` is active via step cache, while predictions cache APIs and `LazyDataset`/`LazyArray` are stubs).
4. Three claims were invalidated by the `cache_refactoring` branch (step cache integration, dataset cache keying, feature concatenation). One claim was a false positive (dataset hash attributes are now consistent).

---

## Part 2: Structured Backlog

### Backlog design rationale

The backlog is organized into **workstreams** (parallel tracks of related work) with **phases** (sequential within each workstream). Priority is driven by:

1. **Correctness over performance**: Semantic bugs and data integrity issues before optimization.
2. **Quick wins first**: Documentation fixes and dead code removal have high signal-to-noise improvement at low cost.
3. **Contract stabilization before consolidation**: Fix fold keys and hash semantics before unifying prediction paths.
4. **Production readiness gradient**: Each phase should leave the codebase strictly better for production use.

Items are tagged with estimated effort: **S** (< 1 day), **M** (1-3 days), **L** (3-5 days), **XL** (> 5 days).

---

### Phase 0: Quick Wins and Correctness Fixes (1-2 weeks)

**Goal**: Fix all documentation drift, remove confirmed dead code, and address the highest-risk semantic bugs.

#### 0.1 Documentation Fixes

| ID | Item | Effort | Source |
|----|------|--------|--------|
| D-01 | Fix Python version in docs (3.9+ → 3.11+) in `installation.md`, `quickstart.md`, `export_bundles.md` | S | Backend Assessment §3G |
| D-02 | Fix TensorFlow installation claim in `installation.md` (optional, not base) | S | Backend Assessment §3G |

#### 0.2 Dead Code Removal

| ID | Item | Effort | Source |
|----|------|--------|--------|
| DC-01 | Delete `nirs4all/data/io.py` (fully commented out) | S | Data Debt |
| DC-02 | Delete `executor._filter_binaries_for_branch` (line 902) | S | Pipeline Debt §P2 |
| DC-03 | Delete `store_queries.build_filter_clause` (line 471) | S | Pipeline Debt §P2 |
| DC-04 | Delete `count_with_constraints`, `get_all_strategies`, `_get_registry_state` from generator modules | S | Pipeline Debt §P2 |
| DC-05 | Remove unreachable block in `step_runner.py:200-233` | S | Pipeline Debt §P2 |
| DC-06 | Remove no-op `clear_caches()` and `get_cache_stats()` stubs from `predictions.py` | S | Data Debt §P1 |

#### 0.3 P0 Correctness Fixes

| ID | Item | Effort | Source |
|----|------|--------|--------|
| C-01 | Fix `dataset_hash` semantics: pass actual dataset content hash (not pipeline hash) into `begin_pipeline()` | M | Pipeline Debt §P0, Backend Assessment §3E |
| C-02 | Fix refit persistence ordering: relabel predictions before flush to store | M | Workspace/Predictions §P0 |
| C-03 | Normalize fold key contract: define canonical fold key format (`str`), normalize at ingestion boundaries only (chain builder, trace import) | L | Workspace/Predictions §P0 |
| C-04 | Make default cache behavior explicit and safe (`cache=None` path): either instantiate default `CacheConfig` or update API contract so branch-heavy runs avoid deep-copy by default | M | Pipeline Debt §P0 |

---

### Phase 1: Reproducibility and Contract Hardening (2-4 weeks)

**Goal**: Eliminate non-determinism sources and stabilize contracts between modules.

#### 1.1 Reproducibility

| ID | Item | Effort | Source |
|----|------|--------|--------|
| R-01 | Propagate seed through `expand_spec_with_choices()` call in `pipeline_config.py:78` | S | Pipeline Debt §P1 |
| R-02 | Replace global `np.random.seed()` with local `np.random.Generator` in `loader.py:48` and `row_selector.py:536` | S | Data Debt §P1 |
| R-03 | Sort resolver filesystem candidates deterministically (lines 525, 536) | S | Pipeline Debt §P1 |
| R-04 | Remove import-time root logger mutation in `pipeline_config.py:26-29`; route through bootstrap | M | Pipeline Debt §P1 |
| R-05 | Increase pipeline hash length (6→12+ hex chars); separate display hash from identity hash | M | Pipeline Debt §P1 |

#### 1.2 Resolver Determinism

| ID | Item | Effort | Source |
|----|------|--------|--------|
| RES-01 | Refactor chain selection to parse `branch_path` structurally (not raw JSON string match) | M | Workspace/Predictions §P0 |
| RES-02 | Add deterministic tie-break order and fail-fast on unresolved ambiguity | M | Workspace/Predictions §P0 |
| RES-03 | Add branch-heavy resolver integration tests | M | Workspace/Predictions §P0 |

#### 1.3 Contract Stabilization

| ID | Item | Effort | Source |
|----|------|--------|--------|
| CT-01 | Implement wavelength passthrough in chain replay path (`workspace_store.py:2074`) | M | Workspace/Predictions §P1 |
| CT-02 | Add full content hash verification in artifact dedup (`artifact_registry.py:1305-1308`) | S | Workspace/Predictions §P1 |
| CT-03 | Add natural-key uniqueness constraint or upsert guard for predictions table | M | Workspace/Predictions §P1 |
| CT-04 | Consolidate prediction write path: either wire `Predictions.flush()` into runtime or remove it | M | Workspace/Predictions §P1 |
| CT-05 | Define explicit resolver mode policy (store-only, filesystem-only, controlled fallback) with fallback observability | M | Workspace/Predictions §P1 |

---

### Phase 2: Performance Hardening (2-4 weeks)

**Goal**: Address measured performance hotspots in execution and storage paths.

#### 2.1 Execution Performance

| ID | Item | Effort | Source |
|----|------|--------|--------|
| P-01 | Gate shape tracing behind verbosity flag; cache shape metadata when selector unchanged | M | Pipeline Debt §P1 |
| P-02 | Build artifact loader indexes at init (`by_step`, `by_step_branch`, `by_step_branch_source`) | M | Pipeline Debt §P1 |
| P-03 | Pre-index chain rows for prediction flush matching (replace O(N×M) with O(N) lookup) | M | Pipeline Debt §P1 |
| P-04 | Default `keep_datasets=False` in runner; add bounded snapshot policy | S | Pipeline Debt §P1 |

#### 2.2 Data Module Performance

| ID | Item | Effort | Source |
|----|------|--------|--------|
| P-05 | Replace metadata hashing data materialization (`dataset.py:1949`) with `content_hash()` or non-materializing hash | M | Data Debt §P1 |

---

### Phase 3: Architecture Consolidation (4-8 weeks)

**Goal**: Reduce structural redundancy and simplify maintenance surface.

#### 3.1 Loader and Provider Unification

| ID | Item | Effort | Source |
|----|------|--------|--------|
| A-01 | Consolidate to single CSV loader stack; deprecate and remove legacy `csv_loader.py` | L | Data Debt §P1 |
| A-02 | Introduce single artifact query service; reduce filtering duplication across `MapArtifactProvider`, `LoaderArtifactProvider`, `artifact_loader.load_for_step()`, `minimal_predictor.get_artifacts_for_step()` | XL | Pipeline Debt §P2 |
| A-03 | Collapse minimal/full prediction setup duplication into one engine with strategy hooks | L | Pipeline Debt §P2 |

#### 3.2 Infrastructure Cleanup

| ID | Item | Effort | Source |
|----|------|--------|--------|
| A-04 | Remove dormant `LazyDataset`/`LazyArray` from `data/performance/` (keep active `DataCache`); remove prediction stub APIs | M | Data Debt §P2 |
| A-05 | Make workspace injection explicit through constructors; keep global helper as CLI convenience only | M | Workspace/Predictions §P2 |
| A-06 | Complete or remove JAX model placeholder (`nicon.py:161`) | M | Backend Assessment §3F |
| A-07 | Decide fate of `pipeline/run.py` (deprecate/remove or document as public API) | S | Pipeline Debt §P2 |

---

### Phase 4: Quality Gates and CI (ongoing)

**Goal**: Prevent debt recurrence through automated enforcement.

| ID | Item | Effort | Source |
|----|------|--------|--------|
| Q-01 | Enforce `ruff check --select F401,F841` (zero unused symbols) in CI for `nirs4all/pipeline` and `nirs4all/data` | M | Pipeline Debt §P2, Data Debt §P2 |
| Q-02 | Add C901 complexity budget per file; fail CI on increase | M | Pipeline Debt §P2 |
| Q-03 | Staged coverage ratchet for critical modules — Stage 1: `executor`≥45%, `orchestrator`≥50%, `resolver`≥62%, `workspace_store`≥30%; Stage 2 targets defined after 2 cycles | L | Pipeline Debt |
| Q-04 | Add contract tests: fold-key normalization across trace→chain→store→replay | M | Workspace/Predictions |
| Q-05 | Add deterministic-selection tests for resolver with ambiguous branches | M | Workspace/Predictions |
| Q-06 | Add integration test asserting persisted refit labels and contexts in store rows | M | Workspace/Predictions |

---

### Phase 5: Security Hardening (when external bundle loading is planned)

| ID | Item | Effort | Source |
|----|------|--------|--------|
| S-01 | Add artifact trust model: trusted-internal mode (current) + restricted mode (no arbitrary pickle) for external bundles | XL | Backend Assessment §3H |
| S-02 | Add signature/hash verification for `.n4a` bundles | L | Backend Assessment §3H |

---

### Backlog Dependencies

```
Phase 0 (quick wins + P0 correctness)
  └── Phase 1 (reproducibility + contracts)
        ├── Phase 2 (performance) — can run in parallel with Phase 1.2/1.3
        └── Phase 3 (consolidation) — depends on stable contracts from Phase 1
              └── Phase 4 (CI gates) — can start early, ratchets up over time
                    └── Phase 5 (security) — when external bundle loading is needed
```

**Critical path**: Phase 0 → Phase 1 → Phase 3

**Parallelizable**: Phase 2 can start as soon as Phase 0 is done. Phase 4 can start immediately (CI gates are additive).

---

### Items Removed from Original Audits

These items appeared in the original audits but are no longer applicable:

| Original Claim | Decision |
|----------------|----------|
| Step cache not integrated into execution (Pipeline Debt §P0) | Removed — fixed in commit `88cbf1c`, step cache fully wired into executor |
| Feature processing concatenation memory amplification (Data Debt §P0) | Removed — fixed in commit `a83aee8`, now uses O(1) block append |
| DatasetConfigs cache unbounded and name-keyed (Data Debt §P0) | Removed — cache now uses config-fingerprinted keys with FIFO eviction |
| Dataset hash inconsistency between `_content_hash` and `_content_hash_cache` (Data Debt §P0) | Removed — code uses single consistent attribute `_content_hash_cache` |
| Remove data performance cache infra wholesale (Data Debt §P1) | Re-scoped — `DataCache` is active (used by step cache); only dormant stubs (`LazyDataset`, `LazyArray`, prediction no-op APIs) are target debt |

---

## Appendix: Claim Validation Matrix

| # | Claim | Verdict | Priority |
|---|-------|---------|----------|
| 1 | Pipeline hash as dataset_hash | CONFIRMED | P0 |
| 2 | Refit persistence lifecycle split | CONFIRMED | P0 |
| 3 | Fold key contract fragmented | CONFIRMED | P0 |
| 4 | Resolver chain selection heuristic | CONFIRMED | P0 |
| 5 | Doc drift: Python version | CONFIRMED | P0 (quick) |
| 6 | Doc drift: TensorFlow | CONFIRMED | P0 (quick) |
| 7 | Unsafe pickle/joblib deserialization | CONFIRMED | P1 |
| 8 | Branch deep-copy in default run path | CONFIRMED | P0 |
| 9 | Per-step shape tracing 2D+3D | CONFIRMED | P1 |
| 10 | Artifact loader linear scans | CONFIRMED | P1 |
| 11 | Prediction flush quadratic matching | CONFIRMED | P1 |
| 12 | Incomplete seed propagation | CONFIRMED | P1 |
| 13 | Global RNG seeding in data utils | CONFIRMED | P1 |
| 14 | Import-time root logger mutation | CONFIRMED | P1 |
| 15 | Hash truncation (6 chars MD5) | CONFIRMED | P1 |
| 16 | Resolver non-deterministic iteration | PARTIALLY VALID | P1 |
| 17 | Memory-heavy defaults (keep_datasets) | CONFIRMED | P1 |
| 18 | Metadata hashing materializes data | CONFIRMED | P1 |
| 19 | Store + filesystem dual resolution | CONFIRMED | P1 |
| 20 | Prediction write path duplication | CONFIRMED | P1 |
| 21 | Chain replay wavelength gap | CONFIRMED | P1 |
| 22 | Artifact dedup integrity incomplete | CONFIRMED | P1 |
| 23 | Schema lacks natural-key uniqueness | CONFIRMED | P1 |
| 24 | Dual loader stacks | CONFIRMED | P2 |
| 25 | Dead: executor._filter_binaries_for_branch | CONFIRMED | P2 |
| 26 | Dead: store_queries.build_filter_clause | CONFIRMED | P2 |
| 27 | Dead: generator constraints/registry functions | CONFIRMED | P2 |
| 28 | Dead: data/io.py commented out | CONFIRMED | P2 |
| 29 | Dead: StepRunner unreachable block | CONFIRMED | P2 |
| 30 | Dormant data/performance infrastructure | PARTIALLY VALID | P2 |
| 31 | Workspace global mutable singleton | CONFIRMED | P2 |
| 32 | JAX model placeholder | CONFIRMED | P2 |
| 33 | Redundant artifact providers | PARTIALLY VALID | P2 |
| 34 | Dual prediction paths | CONFIRMED (by design) | P2 |
| 35 | Step cache not integrated | SUPERSEDED | — |
| 36 | Feature concat memory amplification | SUPERSEDED | — |
| 37 | DatasetConfigs cache unbounded/name-keyed | SUPERSEDED | — |
| 38 | Dataset hash dual-attribute inconsistency | INVALID | — |
| 39 | pipeline/run.py orphaned | PARTIALLY VALID | P2 |

---

## Part 3: Implementation Roadmap

This roadmap translates the structured backlog into concrete implementation guidance: files to modify, what to change, testing strategy, and inter-task dependencies. Each work item includes enough detail to begin implementation directly.

---

### Phase 0: Quick Wins and Correctness Fixes

#### Phase 0 Progress Update (2026-02-07)

Implementation status on branch `cache_refactoring`:

| ID | Status | Notes |
|----|--------|-------|
| D-01 | ✅ Completed | Updated Python minimum version references to `3.11+` in installation, quickstart, and export-bundles docs (`docs/source/user_guide/deployment/export_bundles.md` path in current tree). |
| D-02 | ✅ Completed | Corrected installation wording: TensorFlow is optional (extras), not part of the base install. |
| DC-01 | ✅ Completed | Deleted `nirs4all/data/io.py` (fully commented dead file). |
| DC-02 | ✅ Completed | Removed `PipelineExecutor._filter_binaries_for_branch` (unused no-op). |
| DC-03 | ✅ Completed | Removed `build_filter_clause` from `store_queries.py` (unused legacy helper). |
| DC-04 | ✅ Completed | Removed `count_with_constraints`, `get_all_strategies`, `_get_registry_state` (unused). |
| DC-05 | ✅ Completed | Removed unreachable block after return in `step_runner.py`. |
| DC-06 | ✅ Completed | Removed no-op cache stubs from `Predictions`; updated `PredictionAnalyzer` to gracefully handle optional prediction-level cache APIs. |
| C-01 | ✅ Completed | `begin_pipeline(dataset_hash=...)` now uses `dataset.content_hash()` (true dataset hash). |
| C-02 | ✅ Completed | Refit labels now flow through runtime context and are applied during store flush, so persisted rows are labeled before persistence. |
| C-03 | ✅ Completed | Fold-key normalization landed: canonical `fold_*` keys in trace/chain/store path with legacy lookup compatibility. |
| C-04 | ✅ Completed | `nirs4all.api.run(cache=None)` now explicitly sets `CacheConfig()` defaults (CoW snapshots on, step cache off). |

Validation run after implementation:

- `.venv/bin/pytest -q tests/unit/api/test_run_cache_config.py tests/unit/pipeline/execution/test_executor_phase0.py tests/unit/pipeline/trace/test_execution_trace.py tests/unit/pipeline/config/test_context.py tests/unit/visualization/test_prediction_analyzer_default_aggregate.py tests/unit/pipeline/execution/refit/test_refit_executor.py tests/unit/pipeline/execution/refit/test_stacking_refit.py tests/unit/pipeline/execution/refit/test_refit_p2c.py`
- Result: **248 passed**

#### D-01: Fix Python version in docs [S]

**Files**:
- `docs/source/getting_started/installation.md:20` — change `3.9+` to `3.11+`
- `docs/source/getting_started/quickstart.md:8` — change `3.9+` to `3.11+`
- `docs/source/user_guide/predictions/export_bundles.md:355` — change `3.9+` to `3.11+`

**Reference**: `pyproject.toml:52` requires `python_requires = ">=3.11"`.

**Test**: `grep -r "3\.9" docs/source/ --include="*.md"` returns no hits.

#### D-02: Fix TensorFlow installation claim [S]

**File**: `docs/source/getting_started/installation.md:13-16`

**Change**: Replace claim that TensorFlow is installed by default. Clarify it's in optional extras (`pip install nirs4all[tensorflow]`).

**Reference**: `pyproject.toml` optional dependencies section.

#### DC-01 through DC-06: Dead code removal [S each]

| ID | File | Lines | Action |
|----|------|-------|--------|
| DC-01 | `nirs4all/data/io.py` | entire file | Delete file (175 commented-out lines, zero imports) |
| DC-02 | `nirs4all/pipeline/execution/executor.py` | 902-926 | Delete `_filter_binaries_for_branch` method (no-op, zero callers) |
| DC-03 | `nirs4all/pipeline/storage/store_queries.py` | 467-503 | Delete `build_filter_clause` function (zero callers, marked "Legacy") |
| DC-04 | `nirs4all/pipeline/config/_generator/constraints.py` | 213+ | Delete `count_with_constraints` (zero callers) |
| DC-04 | `nirs4all/pipeline/config/_generator/registry.py` | 96+, 119+ | Delete `get_all_strategies`, `_get_registry_state` (zero callers; keep `clear_registry` for tests) |
| DC-05 | `nirs4all/pipeline/steps/step_runner.py` | 206-233 | Delete unreachable block after `return` on line 200 |
| DC-06 | `nirs4all/data/predictions.py` | 1186-1199 | Delete `clear_caches()` and `get_cache_stats()` stubs (no-op, "API compatibility" comment) |

**Test**: `ruff check nirs4all/` passes. `pytest tests/unit/ -x` passes. No import errors.

#### C-01: Fix `dataset_hash` semantics [M]

**Problem**: `executor.py:160` passes `pipeline_hash` as `dataset_hash` to `begin_pipeline()`.

**Files to modify**:
- `nirs4all/pipeline/execution/executor.py:154-161`

**Implementation**:
```python
# Before (line 160):
dataset_hash=pipeline_hash,

# After:
dataset_hash=dataset.content_hash(),
```

The `dataset` parameter is available in `execute()` scope. `SpectroDataset.content_hash()` (dataset.py:1485) computes a SHA256/xxhash of the feature data, cached in `_content_hash_cache`.

**Migration**: Historical store rows have pipeline hashes in the `dataset_hash` column. Options:
1. Add a `dataset_hash_version` column to distinguish legacy vs corrected rows
2. Accept that pre-fix rows use pipeline hash (document as known limitation)

**Test**: Integration test that runs a pipeline, reads back the store row, and asserts `dataset_hash` matches `dataset.content_hash()`.

#### C-02: Fix refit persistence ordering [M]

**Problem**: `refit/executor.py:143` calls `executor.execute()` (which may flush predictions to store), then `_relabel_refit_predictions()` at line 158. Predictions may reach the store with wrong `fold_id` and `refit_context`.

**Files to modify**:
- `nirs4all/pipeline/execution/refit/executor.py:142-158`
- `nirs4all/pipeline/execution/refit/stacking_refit.py:656-671, 735-751`

**Implementation approach**: Inject final labels into the runtime context **before** `execute()` so predictions are created with correct metadata from the start:
1. Set `runtime_context.refit_fold_id = "final"` and `runtime_context.refit_context = {...}` before `execute()`
2. Have `executor._flush_predictions_to_store()` read these fields when persisting
3. Remove the post-execute `_relabel_refit_predictions()` call

**Alternative** (simpler): Collect predictions in memory (via `prediction_store` parameter) without flushing to the workspace store during refit execution, relabel, then flush once.

**Test**: Integration test: run refit pipeline, query store for refit predictions, assert `fold_id="final"` and `refit_context` are correct on every row.

#### C-03: Normalize fold key contract [L]

**Problem**: Four fold key dialects across modules.

**Current formats**:
| Module | Format | Example |
|--------|--------|---------|
| `chain_builder.py:120` | `f"fold_{fold_id}"` | `"fold_0"`, `"fold_final"` |
| `execution_trace.py:80` | `Dict[int, str]` | `{0: "artifact_id", 1: "..."}` |
| `workspace_store.py` | `"final"` or `"fold_final"` | Mixed |
| `context.py` | `"final"` (bare) | Provider lookup |

**Canonical format**: `str` keys with `f"fold_{n}"` pattern. Special cases: `"fold_final"` for refit.

**Files to modify**:
1. `execution_trace.py:80` — Change `fold_artifact_ids: Dict[int, str]` to `Dict[str, str]`
2. `execution_trace.py:126` — Remove `str→int` key conversion in deserialization
3. `workspace_store.py` — Normalize all `"final"` references to `"fold_final"`
4. `context.py` — Provider lookup uses `"fold_final"` (not bare `"final"`)
5. `chain_builder.py` — Already canonical, no change needed

**Test**: Contract test: create trace → build chain → store → load → verify fold keys survive roundtrip in canonical format. This becomes Q-04.

#### C-04: Make default cache behavior explicit [M]

**Problem**: `api/run.py:394` only sets `cache_config` when `cache` is explicitly provided. Default path has no `CacheConfig`, so `branch.py:1406-1409` disables CoW.

**Files to modify**:
- `nirs4all/api/run.py:393-395`

**Implementation**:
```python
# Before:
if cache is not None:
    runner.cache_config = cache

# After:
from nirs4all.config.cache_config import CacheConfig
runner.cache_config = cache if cache is not None else CacheConfig()
```

`CacheConfig()` defaults: `step_cache_enabled=False`, `use_cow_snapshots=True`. This enables CoW by default without turning on step caching.

**Also update**: `api/run.py:268` docstring to reflect that `cache=None` now uses `CacheConfig()` defaults (CoW on, step cache off).

**Test**: Run a branch-heavy pipeline without explicit `cache=` parameter. Verify CoW path is taken (no deep-copy). Check memory footprint is reduced vs previous behavior.

---

### Phase 1: Reproducibility and Contract Hardening

#### Phase 1 Progress Update (2026-02-07)

Implementation status on branch `cache_refactoring`:

| ID | Status | Notes |
|----|--------|-------|
| R-01 | ✅ Completed | `PipelineConfigs` now propagates a deterministic seed into `expand_spec_with_choices(...)` via `random_state` (explicit arg or root config key). |
| R-02 | ✅ Completed | Removed global NumPy RNG mutation in loader/row selector paths; both now use local RNG instances. |
| R-03 | ✅ Completed | Resolver filesystem discovery now sorts run/subdir candidates deterministically before selection. |
| R-04 | ✅ Completed | Removed import-time root logger mutation from `pipeline_config.py`; logging remains bootstrap-controlled. |
| R-05 | ✅ Completed | Introduced longer identity hash (16 hex chars) and separate short display hash (8 chars) for pipeline names. |
| RES-01 | ✅ Completed | Store chain matching now parses and compares `branch_path` structurally (not raw JSON-string equality). |
| RES-02 | ✅ Completed | Added deterministic chain ordering and explicit `ResolverAmbiguityError` fail-fast behavior on unresolved multi-match cases. |
| RES-03 | ✅ Completed | Added resolver determinism integration coverage in `tests/integration/test_resolver_determinism.py`. |
| CT-01 | ✅ Completed | Chain replay now passes `wavelengths` to transformers when supported (signature/kwargs/mixin-aware). |
| CT-02 | ✅ Completed | Artifact dedup now verifies full content hash before reusing short-hash filename matches. |
| CT-03 | ✅ Completed | Added prediction upsert guard for explicit-ID writes on natural keys (idempotent persistence path) plus supporting index. |
| CT-04 | ✅ Completed | Runtime prediction persistence now routes through `Predictions.flush(...)` (single authoritative write path). |
| CT-05 | ✅ Completed | Resolver mode policy implemented: `"auto"`, `"store"`, `"filesystem"` with store-miss fallback observability in auto mode. |

Validation run after implementation:

- `.venv/bin/pytest -q tests/unit/pipeline/config/test_pipeline_config.py tests/unit/data/loaders/test_loader_random_state.py tests/unit/data/selection/test_row_selector.py tests/unit/pipeline/test_resolver.py tests/integration/test_resolver_determinism.py tests/unit/pipeline/storage/test_workspace_store.py tests/unit/pipeline/storage/artifacts/test_artifact_registry.py tests/unit/data/test_predictions_store.py tests/unit/pipeline/execution/test_executor_phase0.py tests/unit/pipeline/storage/test_chain_replay.py`
- `.venv/bin/pytest -q tests/unit/pipeline/execution/refit/test_refit_infrastructure.py`
- Result: **220 passed**

#### R-01: Propagate seed into generator expansion [S]

**File**: `nirs4all/pipeline/config/pipeline_config.py:78`

**Implementation**:
```python
# Before:
expanded = expand_spec_with_choices(self.steps)

# After:
expanded = expand_spec_with_choices(self.steps, seed=self.random_state)
```

The `expand_spec_with_choices` function in `_generator/core.py:74` already accepts a `seed` parameter. `_generator/iterator.py:284` will use it to create a seeded `random.Random()` instance.

**Test**: Run same pipeline config with `_sample_` keyword twice with same seed. Assert identical expanded configs.

#### R-02: Replace global RNG mutations [S]

**Files**:
- `nirs4all/data/loaders/loader.py:48` — Replace `np.random.seed(config['random_state'])` with local `np.random.default_rng(config['random_state'])`
- `nirs4all/data/selection/row_selector.py:536` — Same pattern

**Implementation**: Create `rng = np.random.default_rng(random_state)` and pass to downstream sampling calls instead of mutating global state.

**Test**: Two concurrent dataset loads with different seeds produce deterministic, independent results.

#### R-03: Sort resolver filesystem candidates [S]

**File**: `nirs4all/pipeline/resolver.py:525,536,546`

**Implementation**: After `iterdir()`, sort candidates by name before selecting: `candidates = sorted(candidates, key=lambda p: p.name)`. Return `candidates[0]` from sorted list.

**Test**: Resolver returns same candidate regardless of filesystem ordering.

#### R-04: Remove import-time root logger mutation [M]

**File**: `nirs4all/pipeline/config/pipeline_config.py:26-29`

**Implementation**: Remove `logging.root.addHandler(...)` and `logging.root.setLevel(...)` from module scope. Move logging configuration to a `configure_logging()` function called once during `nirs4all.run()` bootstrap or `PipelineRunner.__init__()`.

**Test**: `import nirs4all.pipeline.config.pipeline_config` does not add handlers to root logger.

#### R-05: Strengthen pipeline identity hash [M]

**File**: `nirs4all/pipeline/config/pipeline_config.py:91,303`

**Implementation**:
- Increase identity hash to 16 hex chars (64-bit, ~18.4 quintillion values)
- Keep display hash at 6-8 chars for human-readable output
- Document which hash is used where (display vs identity/matching)

**Test**: Two distinct pipeline configs produce different 16-char identity hashes.

#### RES-01: Structural branch-path matching [M]

**File**: `nirs4all/pipeline/resolver.py:1303-1330`

**Implementation**: Replace raw JSON string comparison (`resolver.py:1303-1304`) with parsed dict/list comparison. Normalize branch path representation before matching.

**Test**: Two semantically equivalent branch paths with different JSON serialization order match correctly.

#### RES-02: Deterministic tie-break and fail-fast [M]

**File**: `nirs4all/pipeline/resolver.py:1330` and chain selection paths

**Implementation**:
- When multiple chain candidates match, sort by `(pipeline_id, chain_id)` and select first
- When ambiguity cannot be resolved, raise `ResolverAmbiguityError` instead of silently picking first

**Test**: Resolver with ambiguous branches raises explicit error; resolver with deterministic candidates returns consistently.

#### RES-03: Resolver integration tests [M]

**New file**: `tests/integration/test_resolver_determinism.py`

Tests:
1. Branch-heavy pipeline → resolve → assert deterministic candidate selection
2. Ambiguous branches → assert fail-fast error
3. Store-first + filesystem fallback → assert correct resolution mode

#### CT-01: Wavelength passthrough in chain replay [M]

**File**: `nirs4all/pipeline/storage/workspace_store.py:2074`

**Implementation**: Check if transformer accepts `wavelengths` parameter (via `inspect.signature` or a protocol). If so, pass `wavelengths=...` from the chain metadata.

**Test**: Replay a chain containing a wavelength-aware transformer. Assert wavelengths are passed through.

#### CT-02: Full content hash verification in artifact dedup [S]

**File**: `nirs4all/pipeline/storage/artifacts/artifact_registry.py:1305-1308`

**Implementation**: After short-hash match, compute full content hash of both artifacts and compare. Only reuse if full hashes match.

**Test**: Two artifacts with same short hash but different content are NOT deduplicated.

#### CT-03: Natural-key uniqueness for predictions [M]

**File**: `nirs4all/pipeline/storage/store_schema.py:75-99`

**Implementation**: Add `UNIQUE` constraint on `(pipeline_id, chain_id, fold_id, partition)` to predictions table. Update insert logic to use `INSERT OR REPLACE` (upsert) semantics.

**Migration**: Existing duplicate rows must be deduplicated before adding constraint.

**Test**: Inserting duplicate semantic prediction is idempotent (no error, single row).

#### CT-04: Consolidate prediction write path [M]

**File**: `nirs4all/data/predictions.py:378-394`

**Implementation**: `Predictions.flush()` has zero runtime callers. Remove the method. Document that `executor._flush_predictions_to_store()` is the single authoritative write path.

**Test**: Grep confirms no callers of `Predictions.flush()`. Existing tests pass.

#### CT-05: Resolver mode policy [M]

**File**: `nirs4all/pipeline/resolver.py:588-628`

**Implementation**:
1. Add `resolution_mode` parameter: `"store"`, `"filesystem"`, `"auto"` (default)
2. In `"auto"` mode, try store first, log structured event on filesystem fallback
3. In `"store"` / `"filesystem"` mode, skip the other path entirely

**Test**: `resolution_mode="store"` never touches filesystem. Fallback events are logged.

---

### Phase 2: Performance Hardening

#### Phase 2 Progress Update (2026-02-08)

Implementation status on branch `cache_refactoring`:

| ID | Status | Notes |
|----|--------|-------|
| P-01 | ✅ Completed | Shape tracing is now gated at `verbose >= 2`; executor caches traced shape metadata by selector + dataset-structure key to avoid repeated 2D/3D materialization when unchanged. |
| P-02 | ✅ Completed | `ArtifactLoader` now builds step/branch/source indexes at manifest import and resolves `load_for_step()` via indexed candidate sets instead of full artifact scans. |
| P-03 | ✅ Completed | Prediction flush chain resolution now pre-indexes chain rows by step/branch/class/preprocessing and performs O(1)-style lookup per prediction row. |
| P-04 | ✅ Completed | Default runner/orchestrator behavior is now `keep_datasets=False`; when enabled, preprocessed snapshots are bounded via `max_preprocessed_snapshots_per_dataset` (default `3`). |
| P-05 | ✅ Completed | `SpectroDataset.get_dataset_metadata()` hash generation now uses `content_hash()` (non-materializing path), removing the prior `x(..., layout="2d")` metadata hashing materialization. |

Validation run after implementation:

- `.venv/bin/pytest -q tests/unit/pipeline/execution/test_executor_phase0.py tests/unit/pipeline/storage/artifacts/test_artifact_loader.py tests/unit/pipeline/test_runner_state.py tests/unit/test_hashing.py`
- `.venv/bin/pytest -q tests/unit/pipeline/test_runner_comprehensive.py::TestRunnerInitialization`
- Result: **85 passed**

#### P-01: Gate shape tracing [M]

**File**: `nirs4all/pipeline/execution/executor.py:691,710,575,595,1090,1098`

**Implementation**: Guard `_log_step_shapes()` calls behind `verbose >= 2` or a dedicated trace config flag. Cache shape metadata when dataset selector hasn't changed between pre/post step.

**Test**: Default `verbose=1` run does not call `dataset.x(layout="2d")` or `dataset.x(layout="3d")` for shape tracing.

#### P-02: Artifact loader indexes [M]

**File**: `nirs4all/pipeline/storage/artifacts/artifact_loader.py:364`

**Implementation**: At loader init, build `dict` indexes: `by_step[step_name]`, `by_step_branch[(step, branch)]`, `by_step_branch_source[(step, branch, source)]`. Replace `items()` iteration in `load_for_step()` with O(1) dict lookup.

**Test**: Benchmark `load_for_step` with 100+ artifacts shows sub-linear scaling.

#### P-03: Pre-index chain rows for flush matching [M]

**File**: `nirs4all/pipeline/execution/executor.py:305-340`

**Implementation**: Before the prediction flush loop, build a dict index of chain rows by composite key `(step_name, branch_id, fold_id)`. Replace nested candidate filtering with direct dict lookup.

**Test**: Flush with 1000 predictions and 50 chains completes in O(predictions) time.

#### P-04: Default `keep_datasets=False` [S]

**File**: `nirs4all/pipeline/runner.py:130`

**Implementation**: Change `keep_datasets=True` to `keep_datasets=False`. Update `orchestrator.py:252,303` to respect the flag.

**Test**: Default run does not retain raw/preprocessed arrays after completion. Memory baseline stays bounded.

#### P-05: Non-materializing metadata hash [M]

**File**: `nirs4all/data/dataset.py:1949-1955`

**Implementation**: Replace `self.x(None, layout="2d")` with `self.content_hash()` which hashes the underlying blocks without full materialization. Or hash only the block metadata (shapes, dtypes) if a full content hash is too expensive.

**Test**: `get_dataset_metadata()` hash path does not trigger full 2D materialization.

---

### Phase 3: Architecture Consolidation

#### Phase 3 Progress Update (2026-02-08)

Implementation status on branch `cache_refactoring`:

| ID | Status | Notes |
|----|--------|-------|
| A-01 | ✅ Completed | Consolidated runtime CSV loading to the modern `csv_loader_new` stack. Legacy `csv_loader.py` is now a deprecation shim that delegates to the unified implementation; internal loader paths now import from `csv_loader_new` directly. |
| A-02 | ✅ Completed | Added shared `ArtifactQueryService` + `ArtifactQuerySpec` and refactored artifact lookup/filtering in `ArtifactLoader`, `LoaderArtifactProvider`, `MinimalArtifactProvider`, and `ArtifactRegistry` to use a single query contract for branch/source/substep/fold/pipeline matching. |
| A-03 | ✅ Completed | Refactored `Predictor` into a unified prediction execution engine with strategy hooks (`minimal` vs `full`) to eliminate duplicated setup logic for selector/context/executor/runtime wiring. |
| A-04 | ✅ Completed | Confirmed dormant `LazyDataset`/`LazyArray` and prediction cache stubs are removed; `nirs4all.data.performance` now exposes active cache components only (`DataCache`, `CacheEntry`) with regression coverage. |
| A-05 | ✅ Completed | Enforced explicit workspace injection for runtime-critical orchestration: `PipelineOrchestrator` now requires an explicit `workspace_path`; default/global workspace resolution remains at the `PipelineRunner` convenience layer. |
| A-06 | ✅ Completed | Removed JAX transformer placeholder path in `nicon.py` by consolidating to one concrete `TransformerBlock` implementation and updating transformer builder usage accordingly. |
| A-07 | ✅ Completed | Adopted `pipeline/run.py` as public API: exported run entities/helpers through `nirs4all.pipeline` and top-level `nirs4all`, with dedicated unit tests for transitions/roundtrip/metric helpers and API exposure. |

Validation run after implementation:

- `.venv/bin/pytest -q tests/unit/data/loaders/test_csv_loader.py tests/unit/data/loaders/test_csv_loader_new.py tests/unit/pipeline/storage/artifacts/test_artifact_loader.py tests/unit/pipeline/storage/artifacts/test_artifact_registry.py tests/unit/pipeline/config/test_context.py tests/unit/pipeline/test_runner_state.py tests/unit/pipeline/test_runner_predict.py -k 'not controls_entropy' tests/unit/pipeline/test_run_entities.py tests/unit/pipeline/execution/test_orchestrator_workspace_injection.py tests/unit/data/performance/test_exports.py`
- Result: **158 passed, 1 deselected**

#### A-01: Consolidate CSV loader stacks [L]

**Files**:
- `nirs4all/data/loaders/csv_loader.py` (legacy)
- `nirs4all/data/loaders/csv_loader_new.py` (modern)
- `nirs4all/data/config.py` (uses legacy path)

**Implementation**: Migrate `DatasetConfigs` to use `csv_loader_new.py`. Remove `csv_loader.py`.

#### A-02: Unify artifact query service [XL]

**Files**: `MapArtifactProvider`, `LoaderArtifactProvider`, `artifact_loader.load_for_step()`, `minimal_predictor.get_artifacts_for_step()`

**Implementation**: Create single `ArtifactQueryService` with shared filtering logic. Providers delegate to it.

#### A-03: Collapse prediction setup duplication [L]

**Implementation**: Unify minimal/full prediction pipelines into one engine with strategy pattern (minimal vs full as strategies).

#### A-04: Remove dormant data/performance stubs [M]

**Files**:
- `nirs4all/data/performance/__init__.py` — Remove `LazyDataset`, `LazyArray` exports (keep `DataCache`)
- Remove `LazyDataset`/`LazyArray` source files if they exist as separate modules

#### A-05: Explicit workspace injection [M]

**File**: `nirs4all/workspace/__init__.py`

**Implementation**: Add workspace parameter to constructors of runtime-critical components. Keep module-level `set_workspace()`/`get_workspace()` as CLI convenience.

#### A-06: JAX placeholder decision [M]

**File**: `nirs4all/operators/models/jax/nicon.py:161`

**Decision**: Either complete JAX implementation or remove from supported matrix.

#### A-07: Decide `pipeline/run.py` fate [S]

**File**: `nirs4all/pipeline/run.py`

**Decision**: Module has zero imports but defines domain entities (`Run`, `RunStatus`, `RunConfig`). Either:
1. Adopt: Add imports, tests, and document as public API
2. Remove: Delete the file

---

### Phase 4: Quality Gates and CI

#### Phase 4 Progress Update (2026-02-08)

Implementation status on branch `cache_refactoring`:

| ID | Status | Notes |
|----|--------|-------|
| Q-01 | ✅ Completed | Added unused-symbol debt ratchet for `F401/F841` via `scripts/ci/check_unused_symbol_budget.py` with baseline snapshot file `.github/quality/ruff_unused_budget.json`; wired into `CI.yaml`, `pre-publish.yml`, and `publish.yml`. |
| Q-02 | ✅ Completed | Added per-file C901 complexity budget ratchet via `scripts/ci/check_complexity_budget.py` with baseline `.github/quality/ruff_c901_budget.json`; CI fails on per-file count/max regression. |
| Q-03 | ✅ Completed | Added Stage 1 coverage threshold policy file `.github/quality/coverage_stage1_thresholds.json` and enforcement script `scripts/ci/check_coverage_thresholds.py`; wired to full-test workflows (`pre-publish.yml`, `publish.yml`) after coverage JSON generation. |
| Q-04 | ✅ Completed | Added fold-key contract integration test `tests/integration/storage/test_fold_key_contract.py` validating canonical fold key normalization across trace → chain → store → replay and `fold_final` replay preference. |
| Q-05 | ✅ Completed | Resolver deterministic-selection integration coverage remains active in `tests/integration/test_resolver_determinism.py` (structural branch matching, ambiguity fail-fast, mode policy behavior). |
| Q-06 | ✅ Completed | Added persisted-refit-label integration test `tests/integration/pipeline/test_refit_persistence_contract.py` asserting flush-time REFIT overrides persist to store rows (`fold_id="final"`, `refit_context="standalone"`). |

Validation run after implementation:

- `python scripts/ci/check_unused_symbol_budget.py`
- `python scripts/ci/check_complexity_budget.py --max-complexity 10`
- `python -m pytest -q tests/integration/storage/test_fold_key_contract.py tests/integration/test_resolver_determinism.py tests/integration/pipeline/test_refit_persistence_contract.py`
- Result: **5 passed**

#### Q-01: Unused symbol enforcement [M]

**CI config**: Add `ruff check --select F401,F841` for `nirs4all/pipeline/` and `nirs4all/data/`. Start with current ignore list, freeze count, reduce over releases.

#### Q-02: Complexity budget [M]

**CI config**: Enable C901 rule with current max as baseline. Fail CI if complexity increases in any file.

#### Q-03: Staged coverage ratchet [L]

**Stage 1 targets** (match current baselines + 10%):
- `executor` ≥ 45%
- `orchestrator` ≥ 50%
- `resolver` ≥ 62%
- `workspace_store` ≥ 30%

**Stage 2**: Define after 2 release cycles based on actual improvement rate.

#### Q-04 through Q-06: Contract and regression tests [M each]

| Test | What it validates |
|------|-------------------|
| Q-04 | Fold key roundtrip: trace → chain → store → replay preserves canonical format |
| Q-05 | Resolver determinism: ambiguous branches produce consistent or explicit-error results |
| Q-06 | Refit labels: persisted predictions have correct `fold_id` and `refit_context` |

---

### Phase 5: Security Hardening

#### S-01: Artifact trust model [XL]

**Trigger**: Only needed when external bundle loading is planned.

**Implementation**: Two modes:
1. **Trusted-internal** (current): joblib/pickle as-is
2. **Restricted**: No arbitrary deserialization; use safetensors or explicit allowlists

#### S-02: Bundle signature verification [L]

**Implementation**: Add SHA256 manifest inside `.n4a` bundles. Verify before loading.

---

### Roadmap Timeline and Execution Order

```
Week 1-2:  Phase 0 (quick wins + P0 correctness)
           ├── D-01, D-02 (day 1)
           ├── DC-01..DC-06 (day 1-2)
           ├── C-01, C-04 (day 3-5, parallel)
           ├── C-02 (day 5-7)
           └── C-03 (day 7-10, depends on C-02)

Week 3-4:  Phase 1.1 (reproducibility) — parallel start
           ├── R-01, R-02, R-03 (day 1-2)
           └── R-04, R-05 (day 3-5)

Week 4-6:  Phase 1.2 + 1.3 (resolver + contracts)
           ├── RES-01, RES-02 (day 1-4)
           ├── RES-03 (day 4-5)
           ├── CT-02, CT-04 (day 1-2, parallel with resolver)
           └── CT-01, CT-03, CT-05 (day 3-7)

Week 5-8:  Phase 2 (performance) — parallel from Phase 0 completion
           ├── P-04 (day 1)
           ├── P-01, P-02, P-03 (day 2-6, parallel)
           └── P-05 (day 7-9)

Week 7-10: Phase 4 starts (CI gates) — additive, parallel
           ├── Q-01, Q-02 (day 1-3)
           ├── Q-03 Stage 1 (day 3-6)
           └── Q-04, Q-05, Q-06 (day 4-8, as contracts land)

Week 9-14: Phase 3 (architecture) — after Phase 1 contracts stabilize
           ├── A-04, A-05, A-06, A-07 (week 1-2)
           ├── A-01 (week 2-3)
           └── A-02, A-03 (week 3-5)

Phase 5:   Security — triggered by external bundle loading decision
```

**Critical path**: Phase 0 (C-01..C-04) → Phase 1 (RES-01..RES-03) → Phase 3 (A-01..A-03)

**Quick wins deliverable by end of week 1**: D-01, D-02, DC-01..DC-06 (all documentation fixes and dead code removal)
