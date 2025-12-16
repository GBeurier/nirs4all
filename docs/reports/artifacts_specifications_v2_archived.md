# ⚠️ SUPERSEDED BY V3 DESIGN

> **This document is obsolete.** See [ARTIFACT_SYSTEM_V3_DESIGN.md](./ARTIFACT_SYSTEM_V3_DESIGN.md) for the current artifact system design.
>
> This document is preserved for historical reference only. It describes the V2 implementation which has been superseded.

---

# Artifacts System V2 Specifications (ARCHIVED)

**Status**: Superseded by V3 (December 2025)
**Reason**: V2 system failed to handle branching reload, multi-source branching, nested branches, and meta-model stacking correctly. See [ARTIFACT_SYSTEM_CURRENT_STATE.md](./ARTIFACT_SYSTEM_CURRENT_STATE.md) for the analysis of V2 limitations.

---

## Original Document

*The following is the original V2 specification document, preserved for historical reference.*

---

## 1. Refactoring Objectives

### 1.1 Primary Goals

1. **Deterministic Identification**: Replace iterator-based naming with stable, content-derived artifact IDs that are reproducible across runs and replay operations.

2. **Branch-Aware Storage**: Native support for pipeline branching where artifacts can be shared (pre-branch) or branch-specific.

3. **Stacking Compatibility**: Enable meta-models to reference source model artifacts with clear dependency graphs.

4. **Replay & Transfer**: Reliable artifact loading during predict mode, transfer learning, and pipeline replay—without relying on run-order reconstruction.

5. **Maintenance Simplification**: Eliminate legacy backward-compatibility code paths and reduce architectural complexity.

### 1.2 Secondary Goals

- **Global deduplication**: Same binary content stored once per dataset, across all runs
- **Cleanup utilities**: Functions to delete orphaned artifacts not referenced by any manifest
- Improve debugging with meaningful artifact metadata
- Enable future features: remote storage, artifact search

---

## 2. Known Limitations (Why V3 Was Needed)

The V2 system as implemented has these critical limitations:

### 2.1 Branch Substeps Not Recorded

The execution trace records the branch controller step but **not individual substeps within branches**. This means:
- All branch artifacts are lumped under the parent step
- Artifact filtering by branch relies on parsing the artifact ID
- Reload fails when operation counters diverge between training and prediction

### 2.2 Multi-Source + Branching Fails

When a pipeline has multiple X sources and branching:
- Step 1 creates N transformers (one per source)
- Step 3 (branch) creates M×N transformers (M branches × N sources)
- Reload cannot correctly match transformers to sources

### 2.3 No Operator Chain Concept

V2 identifies artifacts by `{pipeline}:{branch_path}:{step}:{fold}` but this doesn't capture:
- The full chain of preprocessing that led to this artifact
- Dependencies between transformers at the same step
- Cross-branch relationships for meta-models

---

## 3. What V2 Got Right

These concepts from V2 are preserved in V3:

- ✅ Content-addressed deduplication (SHA256 hash)
- ✅ Centralized binaries at `workspace/binaries/<dataset>/`
- ✅ `ArtifactRecord` dataclass for metadata
- ✅ `ArtifactRegistry` for training-time registration
- ✅ `ArtifactLoader` for prediction-time loading
- ✅ Cleanup utilities for orphaned artifacts
- ✅ LRU caching in loader
- ✅ Dependency tracking for meta-models

---

## 4. Migration to V3

V3 introduces the **Operator Chain** concept to address V2 limitations. Migration steps:

1. Run manifest migration script to convert V2 manifests to V3 format
2. Retrain pipelines that use branching, multi-source, or stacking
3. Update custom controllers to use chain-based registration

See [ARTIFACT_SYSTEM_V3_DESIGN.md](./ARTIFACT_SYSTEM_V3_DESIGN.md) for the complete V3 specification.
