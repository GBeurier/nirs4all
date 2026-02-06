# Refit & Caching Roadmap — Implementation Status

> **Source**: `refit_and_caching_roadmap.md`
> **Last updated**: 2026-02-06

## Status Legend
- `TODO` — Not started
- `IN_PROGRESS` — Agent working on it
- `REVIEW` — Implementation done, under review
- `DONE` — Merged, verified, tests pass
- `BLOCKED` — Waiting on dependency
- `SKIPPED` — Excluded from scope

---

## Task Packets & Execution Order

### Wave 1 — Phase 0: Shared Foundation (all independent, launch in parallel)

| Packet | Tasks | Scope | Status | Dependencies |
|--------|-------|-------|--------|-------------|
| **P0-A** | 0.1, 0.2 | Data hashing utility + Execution phase enum | DONE | None |
| **P0-B** | 0.3 | Topology analyzer | DONE | None |
| **P0-C** | 0.4, 0.5 | ArtifactRegistry cache-key support + lifespan | DONE | None |
| **P0-D** | 0.6 | RepeatedKFold OOF accumulation bug fix | DONE | None |

### Wave 2 — Phase 1 (Core Caching) + Phase 2A (Simple Refit Prep) — parallelizable

| Packet | Tasks | Scope | Status | Dependencies |
|--------|-------|-------|--------|-------------|
| **P1-A** | 1.1 | Check-before-fit in TransformerMixinController | DONE | P0-A (0.1), P0-C (0.4) |
| **P2-A** | 2.1, 2.2, 2.3 | Prediction store schema + refit_params parser + winning config extraction | DONE | P0-A (0.2) for 2.3 |

### Wave 3 — Phase 1B + Phase 2B

| Packet | Tasks | Scope | Status | Dependencies |
|--------|-------|-------|--------|-------------|
| **P1-B** | 1.2, 1.3, 1.4 | fit_on_all validation + stateless detection + cross-pipeline reuse | DONE | P1-A (1.1) |
| **P2-B** | 2.4, 2.5 | Simple refit execution + orchestrator two-pass | DONE | P0-A (0.2), P0-B (0.3), P0-C (0.5), P2-A (2.3) |

### Wave 4 — Phase 2C (Refit Integration)

| Packet | Tasks | Scope | Status | Dependencies |
|--------|-------|-------|--------|-------------|
| **P2-C** | 2.6, 2.7, 2.8, 2.9, 2.10 | Fold lifecycle + result object + bundle + prediction mode + metadata | DONE | P2-B (2.4, 2.5) |

### Wave 5 — Phase 2D (Default + Compat)

| Packet | Tasks | Scope | Status | Dependencies |
|--------|-------|-------|--------|-------------|
| **P2-D** | 2.11 | refit=True default + backward compatibility | DONE | P2-C (2.7, 2.8, 2.9) |

### Wave 6 — Phase 3 (Advanced Caching + Stacking Refit)

| Packet | Tasks | Scope | Status | Dependencies |
|--------|-------|-------|--------|-------------|
| **P3-A** | 3.1, 3.2 | StepCache + per-model selection logic | DONE | P1-A (1.1), P2-B (2.5), P0-B (0.3), P2-A (2.3) |
| **P3-B** | 3.3, 3.4, 3.5 | Stacking refit + mixed merge + GPU-aware serialization | DONE | P3-A (3.2), P2-B (2.4, 2.5) |

### Wave 7 — Phase 4 (Advanced Refit)

| Packet | Tasks | Scope | Status | Dependencies |
|--------|-------|-------|--------|-------------|
| **P4-A** | 4.1, 4.2, 4.5 | Nested stacking + separation/multi-source + branches w/o merge | DONE | P3-B (3.3), P0-B (0.3) |
| **P4-B** | 4.3, 4.4 | Lazy per-model refits + warm-start DL | DONE | P3-A (3.2), P2-C (2.7), P2-B (2.4), P2-A (2.2) |

### Wave 8 — Phase 5 (Webapp + Cross-Run)

| Packet | Tasks | Scope | Status | Dependencies |
|--------|-------|-------|--------|-------------|
| **P5-A** | 5.1, 5.2 | WebSocket events + RunProgress UI | DONE | P2-B (2.5) |
| **P5-B** | 5.3, 5.5 | Results dual scoring + backend API | DONE | P2-C (2.7), P2-B (2.5) |
| **P5-C** | 5.4 | Cross-run DuckDB caching | DONE | P1-A (1.1), P0-C (0.4) |

---

## Detailed Task Status

### Phase 0 — Shared Foundation

| Task | Name | Status | Assignee | Notes |
|------|------|--------|----------|-------|
| 0.1 | Data Content Hashing Utility | DONE | P0-A | `utils/hashing.py` (new), `data/dataset.py` — 15 tests |
| 0.2 | Execution Phase Enum and Context | DONE | P0-A | `pipeline/config/context.py` — 10 tests |
| 0.3 | Topology Analyzer | DONE | P0-B | `pipeline/analysis/topology.py` (new) — 29 tests |
| 0.4 | ArtifactRegistry Cache-Key Support | DONE | P0-C | `artifact_registry.py` — 12 tests |
| 0.5 | ArtifactRegistry Lifespan (Both Passes) | DONE | P0-C | `orchestrator.py` lifecycle comment — verified |
| 0.6 | RepeatedKFold OOF Accumulation Bug Fix | DONE | P0-D | `reconstructor.py` — 8 tests + 31 existing |

### Phase 1 — Core Caching

| Task | Name | Status | Assignee | Notes |
|------|------|--------|----------|-------|
| 1.1 | Check-Before-Fit in TransformerMixinController | DONE | P1-A | `controllers/transforms/transformer.py` — 12 tests + 22 existing |
| 1.2 | fit_on_all Artifact Reuse Validation | DONE | P1-B | 3 integration tests — fit once, reuse across folds + refit |
| 1.3 | Stateless Transform Detection and Skip | DONE | P1-B | `_stateless=True` on ~20 operators, params-hash cache key — 16 tests |
| 1.4 | Cross-Pipeline Preprocessing Reuse | DONE | P1-B | 4 integration tests — generator sweep reuse validated |

### Phase 2 — Simple Refit

| Task | Name | Status | Assignee | Notes |
|------|------|--------|----------|-------|
| 2.1 | Prediction Store: refit_context + fold_id="final" | DONE | P2-A | `store_schema.py`, `store_queries.py`, `workspace_store.py`, `predictions.py` — 10 tests |
| 2.2 | refit_params Keyword Support | DONE | P2-A | `parser.py` + new `config/refit_params.py` — 8 tests |
| 2.3 | Winning Configuration Extraction | DONE | P2-A | New `refit/config_extractor.py` — 11 tests |
| 2.4 | Simple Refit Execution | DONE | P2-B | New `refit/executor.py` — RefitResult, FullTrainFoldSplitter, param injection — 32 tests |
| 2.5 | Orchestrator Two-Pass Flow | DONE | P2-B | `orchestrator.py` + `runner.py` + `api/run.py` — refit param threaded — 23 tests |
| 2.6 | Fold Artifact Lifecycle | DONE | P2-C | `workspace_store.py` cleanup_transient_artifacts + orchestrator integration — 8 tests |
| 2.7 | Result Object Changes | DONE | P2-C | `api/result.py` ModelRefitResult + RunResult.final/.cv_best/.models — 10 tests |
| 2.8 | Bundle Export for Single Refit Model | DONE | P2-C | `bundle/loader.py` has_refit manifest + replay_chain final model — 3 tests |
| 2.9 | Prediction Mode: Single Model Dispatch | DONE | P2-C | `base_model.py` + ArtifactProvider.get_refit_artifact — 9 tests |
| 2.10 | Refit Metadata Enrichment | DONE | P2-C | `refit/executor.py` metadata enrichment + _extract_cv_strategy — 16 tests |
| 2.11 | refit=True Default + Backward Compat | DONE | P2-D | `api/run.py`, `runner.py`, `orchestrator.py` default changed to True — 11 tests |

### Phase 3 — Advanced Caching + Stacking Refit

| Task | Name | Status | Assignee | Notes |
|------|------|--------|----------|-------|
| 3.1 | StepCache: In-Memory Preprocessed Data Snapshots | DONE | P3-A | New `step_cache.py` wrapping DataCache, LRU eviction, thread-safe — 25 tests |
| 3.2 | Per-Model Selection Logic | DONE | P3-A | New `model_selector.py` — PerModelSelection, per-variant aggregation — 26 tests |
| 3.3 | Stacking Refit: Two-Step | DONE | P3-B | New `stacking_refit.py` — execute_stacking_refit, base model + meta-model — 67 tests |
| 3.4 | Mixed Merge Refit (Hybrid) | DONE | P3-B | `_classify_branch_type()` for features vs predictions branches — in stacking_refit.py |
| 3.5 | GPU-Aware Serialization | DONE | P3-B | `_is_gpu_model()`, sequential dispatch, `_cleanup_gpu_memory()` — in stacking_refit.py |

### Phase 4 — Advanced Refit

| Task | Name | Status | Assignee | Notes |
|------|------|--------|----------|-------|
| 4.1 | Nested Stacking Refit (Recursive) | DONE | P4-A | `_branch_contains_stacking()` + recursive `execute_stacking_refit()` with depth limit — 13 tests |
| 4.2 | Separation Branch + Multi-Source Refit | DONE | P4-A | `execute_separation_refit()` delegates to stacking or simple refit — 5 tests |
| 4.3 | Lazy Per-Model Standalone Refits | DONE | P4-B | `LazyModelRefitResult` in `api/result.py` — 16 tests |
| 4.4 | Warm-Start Refit for Deep Learning | DONE | P4-B | `_resolve_warm_start_fold` + `_apply_warm_start` in `base_model.py` — 16 tests |
| 4.5 | Branches Without Merge: Winner Selection | DONE | P4-A | `_select_winning_branch()` + `execute_competing_branches_refit()` — 10 tests |

### Phase 5 — Webapp + Cross-Run

| Task | Name | Status | Assignee | Notes |
|------|------|--------|----------|-------|
| 5.1 | WebSocket Events for Refit Phase | DONE | P5-A | `websocket/manager.py` REFIT_STARTED/PROGRESS/COMPLETED/FAILED — 29 webapp tests |
| 5.2 | RunProgress Page: Refit Phase Indicator | DONE | P5-A | `RunProgress.tsx` RefitPhaseIndicator component with status tracking |
| 5.3 | Results Page: Dual Scoring Display | DONE | P5-B | `Results.tsx` has_refit, final score display alongside CV score |
| 5.4 | Cross-Run Caching (DuckDB-Backed) | DONE | P5-C | `store_schema.py` input_data_hash + chain_path_hash columns, `artifact_registry.py` cross-run lookup |
| 5.5 | Backend API Updates for Refit Config | DONE | P5-B | `training.py` refit config passthrough to nirs4all.run() |

---

## Execution Log

| Date | Action | Details |
|------|--------|---------|
| 2026-02-06 | Roadmap parsed | 36 tasks, 6 phases, 14 packets identified |
| 2026-02-06 | Wave 1 launched | Packets P0-A, P0-B, P0-C, P0-D (Phase 0) |
| 2026-02-06 | Wave 1 complete | All 4 packets DONE — 104 tests pass |
| 2026-02-06 | Wave 2 launched | Packets P1-A, P2-A (Phase 1 + Phase 2A) |
| 2026-02-06 | Wave 2 complete | Both packets DONE — P1-A: 12 new tests, P2-A: 29 new tests |
| 2026-02-06 | Wave 3 launched | Packets P1-B, P2-B (Phase 1B + Phase 2B) |
| 2026-02-06 | Wave 3 complete | Both packets DONE — P1-B: 23 new tests (830 ctrl pass), P2-B: 55 new tests (84 refit pass) |
| 2026-02-06 | Wave 4 launched | Packet P2-C (Phase 2C — Refit Integration) |
| 2026-02-06 | Wave 4 complete | P2-C DONE — 46 new tests, 130 refit tests pass total |
| 2026-02-06 | Wave 5+6 launched | Packets P2-D + P3-A in parallel |
| 2026-02-06 | Wave 5+6 complete | P2-D DONE — 11 new tests (141 refit pass), P3-A DONE — 51 new tests |
| 2026-02-06 | Status audit | Discovered P3-B, P4-B, P5-A, P5-B, P5-C were implemented but STATUS not updated |
| 2026-02-06 | P3-B verified DONE | `stacking_refit.py` (24KB) — 67 tests pass |
| 2026-02-06 | P4-B verified DONE | `test_lazy_refit.py` + `test_warm_start.py` — 32 tests pass |
| 2026-02-06 | P5-A verified DONE | WebSocket events + RunProgress RefitPhaseIndicator — 29 webapp tests pass |
| 2026-02-06 | P5-B verified DONE | Results.tsx dual scoring + training.py refit config passthrough |
| 2026-02-06 | P5-C verified DONE | store_schema.py cross-run columns + artifact_registry cross-run lookup |
| 2026-02-06 | Test fix | Fixed test_schema_migration_adds_refit_context (missing artifacts table in test DDL) |
| 2026-02-06 | Remaining: P4-A | Tasks 4.1, 4.2, 4.5 — nested stacking, separation refit, branches w/o merge |
| 2026-02-06 | P4-A complete | `stacking_refit.py` extended + `orchestrator.py` dispatch updated — 34 new tests, 297 refit tests pass |
| 2026-02-06 | Roadmap complete | All 36/36 tasks DONE across 6 phases |

---

## Summary

| Phase | Total Tasks | Done | TODO | Completion |
|-------|------------|------|------|------------|
| 0 — Foundation | 6 | 6 | 0 | 100% |
| 1 — Core Caching | 4 | 4 | 0 | 100% |
| 2 — Simple Refit | 11 | 11 | 0 | 100% |
| 3 — Stacking Refit | 5 | 5 | 0 | 100% |
| 4 — Advanced Refit | 5 | 5 | 0 | 100% |
| 5 — Webapp + Cross-Run | 5 | 5 | 0 | 100% |
| **Total** | **36** | **36** | **0** | **100%** |

**All tasks complete.**
