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

#### [PARTIALLY VALID] `pipeline/run.py` appears orphaned from runtime flow but may be a dormant public surface

- **Source doc**: Pipeline Debt §P2
- **Evidence**: No runtime or test imports were found for `nirs4all.pipeline.run` outside its own module in this branch; however, the file defines full domain entities (`Run`, `RunStatus`, `RunConfig`) that can still be intended as an external/public surface.
- **Backlog verdict**: Keep as P2 decision item (owner decision: deprecate/remove vs adopt/document).

#### [PARTIALLY VALID] Dormant caching/performance infrastructure in data module

- **Evidence**: `DataCache` is actively used by `pipeline/execution/step_cache.py`. The dormant part is mainly API stubs in `predictions.py:1190-1199` (`clear_caches()`, `get_cache_stats()`), plus stale compatibility placeholders.
- **Backlog verdict**: Keep as P2 cleanup, but scope to real stubs/compatibility debt only (do not remove active `DataCache` path).

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

**Key insight**: The audits remain substantially accurate, but backlog priority shifts are required for production readiness:

1. Branch-memory risk remains high in default usage because CoW is not active unless cache config is explicitly provided.
2. Seed reproducibility risk is primarily propagation-level (generator expansion), not complete absence of Python RNG seeding.
3. Some cleanup claims need scope correction (`pipeline/run.py` is ambiguous rather than clearly dead; `DataCache` is active, while predictions cache APIs are stubs).
4. Previously fixed items stay removed (step cache wiring, dataset cache keying/eviction, feature append concatenation).

---

## Part 2: Production-Ready Backlog (Revised)

### Context for this revision

The previous backlog was directionally good, but it mixed low-risk cleanup with production blockers and did not define release gates.
This revision prioritizes **integrity and determinism first**, then performance and consolidation, with explicit acceptance criteria for each work package.

Effort tags: **S** (<1 day), **M** (1-3 days), **L** (3-5 days), **XL** (>5 days).

### Backlog updates and rationale (vs previous synthesis)

| Update | What changed | Rationale |
|---|---|---|
| U-01 | Branch snapshot CoW moved from mitigated/P1 to explicit P0 | Default API flow does not set cache config, so branch path falls back to deep-copy in normal usage. This is a production memory risk, not an edge case. |
| U-02 | Added `CacheConfig` defaulting work item | Documentation says `cache=None` uses default cache config, but code only sets cache config when user passes one. This mismatch must be resolved to make behavior predictable. |
| U-03 | Split fold-key fix into contract + migration + compatibility tasks | Fold key inconsistencies affect trace, chain, store, export, and replay. A one-step refactor is risky without migration and compatibility windows. |
| U-04 | Added schema/data migration tasks for provenance fields | Fixing `dataset_hash` semantics and prediction idempotency requires migration/backfill strategy, not only code edits. |
| U-05 | Reordered dead-code cleanup behind integrity blockers | Cleanup improves maintainability but should not delay correctness/integrity fixes that affect persisted data. |
| U-06 | Re-scoped data performance cleanup | `DataCache` is active (used by step cache); only real stubs/no-op APIs should be removed. |
| U-07 | Changed `pipeline/run.py` handling from remove to owner decision | Module looks orphaned in runtime/tests, but can still be intended external surface. Production backlog should force an explicit decision, not silent deletion. |
| U-08 | Added staged quality ratchet with current baselines | Current coverage for critical modules is low (`executor` 35%, `orchestrator` 41%, `resolver` 57%, `workspace_store` 20%). A single jump to high thresholds is not credible. |
| U-09 | Added release gates with go/no-go criteria | “Production-ready” needs binary gates (integrity, determinism, operational confidence), not only task completion. |
| U-10 | Added rollback/flag strategy for high-risk changes | Contract and persistence changes need safe rollout and rollback to avoid corrupting long-running experiments. |

### Production release gates

| Gate | Objective | Must pass before next gate |
|---|---|---|
| G0: Integrity | No known P0 correctness/provenance defects in default execution path | P0 items completed + migration scripts + integration tests green |
| G1: Determinism | Repeated runs with same seed produce stable selection and replay | Determinism test suite green + resolver ambiguity fail-fast in place |
| G2: Operational Confidence | Service-like runs are observable, bounded, and resilient | Memory/perf budgets enforced + CI ratchet thresholds met + rollback path tested |

### Hierarchical backlog

### G0 - Integrity and Provenance (Critical Path, 1-2 weeks)

| ID | Priority | Item | Depends on | Definition of done |
|---|---|---|---|---|
| G0-01 | P0 | Fix `dataset_hash` semantics (`begin_pipeline` must store dataset content hash, not pipeline hash) | None | New writes use dataset content hash; migration updates historical rows or marks legacy rows; cache invalidation uses correct field |
| G0-02 | P0 | Fix refit labeling order so fold/context labels are finalized before store persistence | None | Refit predictions persist directly with final labels; no post-persist relabel pass; integration test covers standalone + stacking refit |
| G0-03 | P0 | Canonical fold key contract (`str` keys, one dialect) | G0-02 | Canonical key spec documented; trace/chain/store/export/replay adapters implemented; compatibility shim with deprecation warning |
| G0-04 | P0 | Make default cache behavior explicit and safe (`cache=None` path) | None | Either instantiate default `CacheConfig` (with documented defaults) or update API contract/docs; branch-heavy default run avoids deep-copy restore path |
| G0-05 | P0 | Fix docs drift (Python version + TensorFlow base install claim) | None | `installation.md`, `quickstart.md`, `export_bundles.md` aligned with `pyproject.toml`; docs CI passes |

### G1 - Determinism and Resolution Hardening (2-3 weeks)

| ID | Priority | Item | Depends on | Definition of done |
|---|---|---|---|---|
| G1-01 | P1 | Propagate seed into generator expansion calls | None | `PipelineConfigs` passes seed through expansion path; reproducibility tests cover `_sample_` and probabilistic strategies |
| G1-02 | P1 | Replace global RNG mutation (`np.random.seed`) with local generators in data utilities | None | No global RNG reseeding in data loaders/selectors; deterministic behavior preserved via explicit generator injection |
| G1-03 | P1 | Resolver determinism rewrite (sorted candidates + structural branch-path matching + tie-break rules) | G0-03 | No `candidates[0]` fallback without deterministic ordering; ambiguity raises explicit error; resolver regression tests added |
| G1-04 | P1 | Remove import-time root logger mutation | None | Logging config initialized once via bootstrap; import side effects removed |
| G1-05 | P1 | Strengthen pipeline identity hash (separate display hash from identity hash) | None | Identity hash length and algorithm documented; callers use identity hash for matching, short hash for display only |

### G2 - Persistence and Replay Contracts (2-3 weeks)

| ID | Priority | Item | Depends on | Definition of done |
|---|---|---|---|---|
| G2-01 | P1 | Add natural-key idempotency for predictions (unique index and/or upsert) | G0-03 | Duplicate semantic rows blocked; retry writes are idempotent |
| G2-02 | P1 | Consolidate prediction persistence path (`executor` vs `Predictions.flush`) | G2-01 | Single authoritative runtime write path; alternative path removed or explicitly delegated |
| G2-03 | P1 | Complete artifact dedup integrity check (verify full content hash before reuse) | None | Short-hash match no longer accepted without full-hash verification |
| G2-04 | P1 | Implement wavelength-aware replay passthrough | None | Replay path passes wavelengths to compatible transformers; unit tests cover wavelength-aware operators |
| G2-05 | P1 | Define resolver mode policy (store-only/filesystem-only/controlled fallback) + telemetry | G1-03 | Fallback is intentional and logged/metriced; mode selectable per run |

### G3 - Performance and Memory Hardening (parallel after G0, 2-4 weeks)

| ID | Priority | Item | Depends on | Definition of done |
|---|---|---|---|---|
| G3-01 | P1 | Gate per-step shape tracing behind config/verbosity and avoid duplicate materialization | None | No unconditional 2D+3D materialization for each step; trace fidelity preserved when enabled |
| G3-02 | P1 | Build artifact loader indexes (`by_step`, `by_step+branch`, `by_step+branch+source`) | None | `load_for_step` avoids O(N) scans; benchmark confirms improved lookup complexity |
| G3-03 | P1 | Pre-index chain rows for prediction flush matching | None | Flush path scales ~O(predictions) instead of O(predictions×chains) |
| G3-04 | P1 | Change memory-heavy defaults (`keep_datasets`) to bounded/off policy | None | Default run does not retain unbounded snapshots; opt-in path documented |
| G3-05 | P1 | Remove metadata hash full materialization path | None | `get_dataset_metadata()` hash path avoids full `x(layout="2d")` materialization |
| G3-06 | P1 | Step-cache graduation plan (still optional, correctness-validated) | G0-04 | Cache remains feature-flagged until stress tests pass; rollout checklist defined |

### G4 - Architecture and Cleanup (after G0/G1, 3-6 weeks)

| ID | Priority | Item | Depends on | Definition of done |
|---|---|---|---|---|
| G4-01 | P2 | Remove confirmed dead code (`data/io.py`, unused helper functions, unreachable StepRunner block) | None | Symbols removed with tests/docs updated |
| G4-02 | P2 | Decide fate of `pipeline/run.py` (deprecate/remove or document as public API) | None | ADR decision recorded; either published API docs/tests or deprecation + removal plan |
| G4-03 | P2 | Consolidate dual CSV loader stacks | G1-02 | One loader path retained; compatibility shim + migration notes provided |
| G4-04 | P2 | Reduce global workspace singleton usage in service-facing paths | None | Constructors accept explicit workspace dependency in runtime-critical components |
| G4-05 | P2 | JAX placeholder decision (complete or remove from supported matrix) | None | Support matrix and code align |

### G5 - Quality Platform, Observability, and Security (start early, ratchet over time)

| ID | Priority | Item | Depends on | Definition of done |
|---|---|---|---|---|
| G5-01 | P1 | CI static checks ratchet (start by enforcing unused-symbol debt freeze, then reduce ignore set) | None | CI blocks new unused imports/locals in touched files; ignore list shrinks over releases |
| G5-02 | P1 | Coverage ratchet for critical modules (stage 1 then stage 2) | None | Stage 1: `executor>=45%`, `orchestrator>=50%`, `resolver>=62%`, `workspace_store>=30%`; Stage 2 targets defined after 2 cycles |
| G5-03 | P1 | Contract tests (fold-key roundtrip, resolver ambiguity, refit persisted labels, replay parity) | G0-03, G1-03 | New tests fail on contract regressions |
| G5-04 | P1 | Observability for fallback/ambiguity/cache/memory paths | G2-05 | Structured counters/log events present; dashboards/alerts defined for production runs |
| G5-05 | P1/P0* | Artifact trust model and restricted deserialization mode | None | Trusted-internal mode explicit; restricted mode available for untrusted bundles (`*P0 if external bundles are accepted`) |
| G5-06 | P1/P0* | Bundle signature/hash verification | G5-05 | Bundle integrity verified before load (`*P0 if external bundles are accepted`) |

### Sequencing and critical path

```
G0 (integrity/provenance)
  -> G1 (determinism/resolution)
     -> G2 (persistence/replay contracts)
        -> G5 contract tests + CI ratchet

Parallel from G0:
  G3 (performance/memory)

After G0/G1:
  G4 (architecture/cleanup)
```

Critical path: `G0 -> G1 -> G2 -> G5(contract tests)`.

### Removed or re-scoped items from earlier backlog

| Item | Decision |
|---|---|
| Step cache not integrated | Removed (already fixed and integrated) |
| Feature concatenation memory amplification | Removed (already fixed by block append path) |
| DatasetConfigs unbounded/name-only cache | Removed (already fixed with fingerprinted keys + eviction) |
| Dataset dual hash attributes | Removed (single `_content_hash_cache` in current code) |
| Remove `features.py` cache flag | Removed (claim not applicable in current file) |
| Remove data performance cache infra | Re-scoped (`DataCache` is active; only no-op/stub APIs are target debt) |

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
