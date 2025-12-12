# Artifacts System Refactoring - Roadmap

## Overview

This roadmap defines the implementation phases for the artifacts system refactoring described in [artifacts_specifications.md](./artifacts_specifications.md).

**Estimated Duration:** 3-4 weeks
**Priority:** High (blocks stacking and transfer features)

---

## Phase 1: Foundation (Week 1)

### 1.1 Type Definitions and Data Structures

| Task | Description | Files |
|------|-------------|-------|
| Create `ArtifactRecord` dataclass | Full artifact metadata with all new fields | `storage/artifacts/types.py` |
| Create `ArtifactType` enum | `model`, `transformer`, `splitter`, `encoder`, `meta_model` | `storage/artifacts/types.py` |
| Create execution path utilities | `generate_artifact_id()`, `parse_artifact_id()` | `storage/artifacts/utils.py` |

### 1.2 ArtifactRegistry Implementation

| Task | Description | Files |
|------|-------------|-------|
| Core registry class | `generate_id()`, `register()`, `resolve()` | `storage/artifacts/artifact_registry.py` |
| Deduplication logic | Content-hash based file reuse | `storage/artifacts/artifact_registry.py` |
| Dependency graph | `add_dependency()`, `get_dependencies()`, `resolve_dependencies()` | `storage/artifacts/artifact_registry.py` |

### 1.3 Manifest Schema v2.0

| Task | Description | Files |
|------|-------------|-------|
| Define new artifacts section schema | YAML structure with `schema_version`, `items` list | `storage/manifest_manager.py` |
| Add schema version detection | Auto-detect v1 vs v2 manifests | `storage/manifest_manager.py` |
| Implement v2 serialization | Write new format | `storage/manifest_manager.py` |

**Deliverable:** New types and registry can generate IDs and store artifacts (parallel to existing system).

---

## Phase 2: Loader and Legacy Removal (Week 1-2)

### 2.1 ArtifactLoader Implementation

| Task | Description | Files |
|------|-------------|-------|
| Core loader class | New loader with centralized binaries support | `storage/artifacts/artifact_loader.py` |
| `load_by_id()` | Single artifact by ID | `storage/artifacts/artifact_loader.py` |
| `load_for_step()` | All artifacts for step/branch/fold context | `storage/artifacts/artifact_loader.py` |
| `load_with_dependencies()` | Transitive dependency resolution | `storage/artifacts/artifact_loader.py` |
| `load_fold_models()` | Load all fold models for CV averaging | `storage/artifacts/artifact_loader.py` |

### 2.2 Delete Legacy Code (No Backward Compatibility)

| Task | Description | Files |
|------|-------------|-------|
| Delete `BinaryLoader` | Remove entirely | `storage/artifacts/binary_loader.py` |
| Delete old `ArtifactManager` | Remove or rewrite | `storage/artifacts/manager.py` |
| Remove `legacy_pickle` format | No old format support | `storage/artifacts/artifact_persistence.py` |
| Remove `metadata.json` support | No old manifest support | All relevant files |
| Remove `_binaries/` handling | Old per-run storage | `storage/io.py` |
| Remove `artifacts/objects/` | Old content-addressed structure | `storage/manifest_manager.py` |

**Deliverable:** Single, clean loader with centralized binaries. No legacy code.

---

## Phase 3: Controller Integration (Week 2)

### 3.1 Context Updates

| Task | Description | Files |
|------|-------------|-------|
| Add `branch_path: List[int]` to `DataSelector` | Replace single `branch_id` | `pipeline/config/context.py` |
| Add branch path tracking in orchestrator | Maintain path stack during branching | `pipeline/execution/orchestrator.py` |
| Update `RuntimeContext` | Add `artifact_registry` reference | `pipeline/config/context.py` |

### 3.2 Controller Updates

| Controller | Changes | Files |
|------------|---------|-------|
| `ModelController` | Use registry, track dependencies to preprocessors | `pipeline/steps/model_controller.py` |
| `TransformController` | Use registry for fitted transformers | `pipeline/steps/transform_controller.py` |
| `SplitterController` | Use registry for splitter persistence | `pipeline/steps/splitter_controller.py` |
| `YProcessingController` | Use registry for encoders/scalers | `pipeline/steps/y_processing_controller.py` |
| `BranchController` | Pass branch path to sub-pipelines | `pipeline/steps/branch_controller.py` |

### 3.3 Runner Integration

| Task | Description | Files |
|------|-------------|-------|
| Initialize registry in runner | Create `ArtifactRegistry` at run start | `pipeline/runner.py` |
| Pass registry to controllers | Via `RuntimeContext` | `pipeline/runner.py` |
| Update predict mode | Use `ArtifactLoader` instead of `BinaryLoader` | `pipeline/predictor.py` |

**Deliverable:** All controllers use new artifact system for training and prediction.

---

## Phase 4: Stacking Support (Week 2-3) ✅ COMPLETED

### 4.1 MetaModel Artifact Handling ✅

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| `meta_model` artifact type | Special handling in registry | `storage/artifacts/types.py` | ✅ Done |
| Source model references in `ArtifactRecord` | `meta_config` field with ordered sources | `storage/artifacts/types.py` | ✅ Done |
| Dependency validation | Ensure source models exist at registration | `storage/artifacts/artifact_registry.py` | ✅ Done |
| `register_meta_model()` convenience method | Simplified meta-model registration | `storage/artifacts/artifact_registry.py` | ✅ Done |

### 4.2 MetaModel Loading ✅

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| `load_meta_model_with_sources()` | Auto-load source models with feature columns | `storage/artifacts/artifact_loader.py` | ✅ Done |
| `load_meta_model_for_prediction()` | Convenience method for prediction mode | `storage/artifacts/artifact_loader.py` | ✅ Done |
| Feature order preservation | Match `feature_columns` from config | `storage/artifacts/artifact_loader.py` | ✅ Done |
| Branch context validation | Ensure correct branch during prediction | `storage/artifacts/artifact_loader.py` | ✅ Done |

**Deliverable:** Stacking pipelines save and load correctly. ✅

---

## Phase 5: Cleanup Utilities (Week 3) ✅ COMPLETED

### 5.1 Orphan Detection and Deletion ✅

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| `find_orphaned_artifacts()` | Scan manifests vs binaries directory | `storage/artifacts/artifact_registry.py` | ✅ Done |
| `delete_orphaned_artifacts()` | Delete with dry-run support | `storage/artifacts/artifact_registry.py` | ✅ Done |
| `delete_pipeline_artifacts()` | Delete artifacts for specific pipeline | `storage/artifacts/artifact_registry.py` | ✅ Done |
| `cleanup_failed_run()` | Auto-cleanup on exception | `storage/artifacts/artifact_registry.py` | ✅ Done |
| `purge_dataset_artifacts()` | Delete ALL artifacts for dataset | `storage/artifacts/artifact_registry.py` | ✅ Done |

### 5.2 CLI Tools ✅

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| `nirs4all artifacts` CLI | Entry point for artifact management | `nirs4all/cli/commands/artifacts.py` | ✅ Done |
| `list-orphaned` command | Show unreferenced artifacts | `nirs4all/cli/commands/artifacts.py` | ✅ Done |
| `cleanup` command | Delete orphaned artifacts | `nirs4all/cli/commands/artifacts.py` | ✅ Done |
| `stats` command | Show storage statistics | `nirs4all/cli/commands/artifacts.py` | ✅ Done |
| `purge` command | Delete all artifacts for dataset | `nirs4all/cli/commands/artifacts.py` | ✅ Done |

### 5.3 Runner Integration ✅

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| Auto-cleanup on failure | Call `cleanup_failed_run()` in exception handler | `pipeline/execution/orchestrator.py` | ✅ Done |
| Cleanup logging | Log what was deleted | `storage/artifacts/artifact_registry.py` | ✅ Done |

### 5.4 Optimization ✅

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| Lazy loading | Don't load until needed | `storage/artifacts/artifact_loader.py` | ✅ Done |
| LRU cache | Cache by artifact ID with eviction | `storage/artifacts/artifact_loader.py` | ✅ Done |
| `preload_artifacts()` | Warm cache before prediction | `storage/artifacts/artifact_loader.py` | ✅ Done |
| `set_cache_size()` | Dynamic cache resizing | `storage/artifacts/artifact_loader.py` | ✅ Done |

**Deliverable:** Complete cleanup toolkit with CLI and auto-cleanup on failure. ✅

---

## Phase 6: Future Enhancements

### 6.1 Remote Storage (Deferred)

| Task | Description |
|------|-------------|
| Storage backend abstraction | Interface for local/S3/Azure |
| S3 backend implementation | Upload/download artifacts |

> **Note:** Remote storage is out of scope. All storage is local for now.

---

## Testing Strategy

### Unit Tests

| Component | Test Focus | Location | Status |
|-----------|------------|----------|--------|
| `ArtifactRegistry` | ID generation, deduplication, dependency graph | `tests/unit/pipeline/storage/artifacts/test_artifact_registry.py` | ✅ |
| `ArtifactLoader` | Loading by ID, step, fold models, dependencies | `tests/unit/pipeline/storage/artifacts/test_artifact_loader.py` | ✅ |
| `ArtifactRecord` | Serialization, validation | `tests/unit/pipeline/storage/artifacts/test_types.py` | ✅ |
| Cleanup utilities | Orphan detection, deletion, failed run cleanup, purge | `tests/unit/pipeline/storage/artifacts/test_artifact_registry.py` | ✅ |
| Deduplication | Cross-run deduplication, hash collision | `tests/unit/pipeline/storage/artifacts/test_artifact_registry.py` | ✅ |
| MetaModel handling | Meta-model registration, dependency validation | `tests/unit/pipeline/storage/artifacts/test_artifact_registry.py::TestMetaModelHandling` | ✅ |
| MetaModel loading | Load with sources, branch validation | `tests/unit/pipeline/storage/artifacts/test_artifact_loader.py::TestMetaModelLoading` | ✅ |
| LRU Cache | Eviction, hit/miss stats, preloading | `tests/unit/pipeline/storage/artifacts/test_artifact_loader.py::TestLRUCache` | ✅ |
| Manifest v2 | Schema writing and reading | `tests/unit/pipeline/storage/test_manifest_v2.py` | |

### Integration Tests

| Scenario | Description | Location |
|----------|-------------|----------|
| Training + Predict | Full pipeline with centralized artifacts | `tests/integration/artifacts/test_artifact_flow.py` |
| Branching | Branch-specific artifact isolation | `tests/integration/artifacts/test_branching_artifacts.py` |
| Stacking | Meta-model dependencies | `tests/integration/artifacts/test_stacking_artifacts.py` |
| Cross-run deduplication | Same model reused across runs | `tests/integration/artifacts/test_deduplication.py` |
| CV Fold Models | Per-fold model saving and ensemble loading | `tests/integration/artifacts/test_fold_models.py` |
| Cleanup | Orphan detection and deletion | `tests/integration/artifacts/test_cleanup.py` |

### Example Validation

| Example | Purpose | Status |
|---------|---------|--------|
| `Q5_predict.py` | Verify predict mode with new loader | |
| `Q30_branching.py` | Verify branch artifacts | |
| `Q18_stacking.py` | Verify meta-model persistence | ✅ Training works |

---

## Documentation

### User Documentation

| Document | Update |
|----------|--------|
| `docs/reference/outputs_vs_artifacts.md` | Update artifact structure description |
| `docs/specifications/manifest.md` | Add v2 schema documentation |
| `docs/user_guide/predict_mode.md` | Update loading explanation |

### Developer Documentation

| Document | Update |
|----------|--------|
| `docs/api/storage.md` | New API reference |
| `CONTRIBUTING.md` | How to add new artifact types |
| Cleanup utilities guide | CLI tools usage |

---

## Success Criteria

### Functional

- [ ] All existing examples pass (`run.ps1`)
- [ ] Branching pipelines produce correct artifacts
- [ ] Stacking pipelines save/load correctly
- [ ] Per-fold models saved and loadable for CV averaging
- [ ] Cross-run deduplication works (same hash = one file)
- [ ] Cleanup utilities work (orphan detection, deletion)
- [ ] Failed runs clean up their artifacts automatically

### Performance

- [ ] No regression in training time
- [ ] Artifact loading ≤ 10% overhead vs current
- [ ] Deduplication ratio maintained or improved

### Code Quality

- [ ] No `step_number` in artifact identification
- [ ] No legacy code (`BinaryLoader`, `legacy_pickle`, `metadata.json`, `_binaries/`)
- [ ] Centralized binaries at `workspace/binaries/<dataset>/`
- [ ] All new code has type hints
- [ ] Test coverage > 80% for new modules
- [ ] No pylint errors in new code

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Performance regression | Medium | Benchmark before/after; lazy loading |
| Scope creep | Medium | Strict phase boundaries |
| Complex dependency graphs | Medium | Cycle detection; depth limits |
| Artifact accumulation | Medium | Cleanup utilities; auto-cleanup on failure |

---

## Dependencies

- Requires: Branching feature complete (Phase 2+)
- Blocks: Stacking feature, Transfer learning
- Related: Workspace v2, Remote storage
