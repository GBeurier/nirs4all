# Branching, Concat-Transform, and Merge: Unified Design Document

**Version**: 1.0.0
**Status**: Design Proposal
**Date**: December 2025
**Author**: Architecture Review

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Current State Analysis](#current-state-analysis)
   - [3.1 BranchController](#31-branchcontroller)
   - [3.2 ConcatAugmentationController](#32-concataugmentationcontroller)
   - [3.3 MetaModel / Stacking](#33-metamodel--stacking)
   - [3.4 Feature Augmentation Mode](#34-feature-augmentation-mode)
   - [3.5 Identified Redundancies and Conflicts](#35-identified-redundancies-and-conflicts)
4. [Conceptual Framework](#conceptual-framework)
   - [4.1 Fundamental Operations Taxonomy](#41-fundamental-operations-taxonomy)
   - [4.2 When Each Operation Applies](#42-when-each-operation-applies)
   - [4.3 Data Leakage Considerations](#43-data-leakage-considerations)
5. [Design Proposal: Unified Branch-Merge Architecture](#design-proposal-unified-branch-merge-architecture)
   - [5.1 Design Principles](#51-design-principles)
   - [5.2 Proposed Syntax](#52-proposed-syntax)
   - [5.3 Merge Strategies](#53-merge-strategies)
   - [5.4 Fold-Aware Merge](#54-fold-aware-merge)
6. [Design Proposal: Concat-Transform Clarification](#design-proposal-concat-transform-clarification)
7. [Design Proposal: Stacking Integration](#design-proposal-stacking-integration)
8. [Implementation Roadmap](#implementation-roadmap)
9. [Migration Guide](#migration-guide)
10. [Appendix: Decision Log](#appendix-decision-log)

---

## Executive Summary

This document addresses critical redundancies and conceptual overlaps in nirs4all's branching, concatenation, and merging systems. After exhaustive analysis of the codebase, we identified several overlapping concepts that create user confusion and potential data leakage bugs:

| Concept | Current Implementation | Issue |
|---------|------------------------|-------|
| **Preprocessing branching** | `BranchController` | Works well, but no formal "merge" step |
| **Feature concatenation** | `ConcatAugmentationController` | Overlaps with branch merge use case |
| **Prediction merging** | `MergeController` | **Empty file** - not implemented |
| **Stacking** | `MetaModel` + `TrainingSetReconstructor` | Has OOF safeguards but separate syntax |

**Key Finding**: Merging branch predictions without OOF safeguards is equivalent to stacking without fold awareness, causing data leakage. This is a critical design flaw that must be addressed.

**Recommendation**: Unify the conceptual model with clear separation:
1. **`concat_transform`** → Feature-level concatenation within a single execution path
2. **`branch` + implicit merge** → Parallel preprocessing, results used independently per branch
3. **`MetaModel`** → Prediction-level stacking with OOF safeguards (the ONLY safe way to merge predictions)

---

## Problem Statement

### User Confusion Scenarios

**Scenario 1: "I want to try 3 preprocessings and combine their features"**

User might attempt:
```python
# Option A: concat_transform (CORRECT for this use case)
{"concat_transform": [SNV(), MSC(), FirstDerivative()]}

# Option B: branch + merge (INTUITIVE but problematic)
{"branch": [[SNV()], [MSC()], [FirstDerivative()]]},
{"merge": "features"}  # Does not exist, would duplicate concat_transform

# Option C: feature_augmentation (INCORRECT - creates processings, not concat)
{"feature_augmentation": [SNV(), MSC(), FirstDerivative()]}
```

**Scenario 2: "I want 3 branches with different models, then combine predictions"**

```python
# DANGEROUS (if it existed):
{"branch": [
    [SNV(), PLSRegression(n_components=10)],
    [MSC(), RandomForestRegressor()],
]},
{"merge": "predictions"}  # Would cause data leakage!

# CORRECT:
{"branch": [
    [SNV(), PLSRegression(n_components=10)],
    [MSC(), RandomForestRegressor()],
]},
{"model": MetaModel(model=Ridge(), source_models="all")}  # OOF-safe
```

**Scenario 3: "I want branch-specific preprocessings, then one shared model"**

```python
# CURRENT (works correctly):
{"branch": [[SNV(), PCA(50)], [MSC(), PCA(50)]]},
PLSRegression(n_components=10)  # Runs on EACH branch independently
# Result: 2 independent models, 2 sets of predictions
```

### Core Questions

1. **When to use `concat_transform` vs `branch`?**
2. **How to merge branch outputs safely?**
3. **What happens to the "merge.py" empty file?**
4. **How do we prevent users from accidentally creating data leakage?**

---

## Current State Analysis

### 3.1 BranchController

**Location**: `nirs4all/controllers/data/branch.py`

**What it does**:
- Creates N independent execution contexts (one per branch)
- Each branch gets its own feature snapshot
- Post-branch steps iterate over ALL branch contexts independently
- Nested branching creates Cartesian product of contexts

**Key implementation details**:
```python
# Branch context storage
context.custom["branch_contexts"] = [
    {
        "branch_id": 0,
        "name": "snv_pca",
        "context": ExecutionContext(...),
        "features_snapshot": [...]  # Snapshot of features AFTER branch processing
    },
    # ...
]
```

**Post-branch execution** (in `executor.py`):
```python
for branch_info in branch_contexts:
    # Restore features from snapshot
    dataset._features.sources = copy.deepcopy(features_snapshot)
    # Execute step on this branch
    step_result = self.step_runner.execute(...)
```

**Pros**:
- Clean isolation between branches
- Proper snapshot/restore mechanism
- Supports nested branching
- Works with prediction mode

**Cons**:
- No formal "merge" step - branches just continue independently
- No way to combine branch outputs into a single path
- Every post-branch step runs N times (once per branch)

**Hidden Flaw**: There's no mechanism to exit branch mode and return to single-path execution. Once you branch, you stay branched forever.

### 3.2 ConcatAugmentationController

**Location**: `nirs4all/controllers/data/concat_transform.py`

**What it does**:
- Applies multiple transformers and concatenates results horizontally
- Two modes: REPLACE (top-level) or ADD (inside feature_augmentation)
- Supports nested concat_transform and chains

**Key implementation**:
```python
# Apply each operation and collect results
for operation in operations:
    transformed = operation.fit_transform(data)
    concat_blocks.append(transformed)

# Horizontal concatenation
concatenated = np.hstack(concat_blocks)
```

**Pros**:
- Clean horizontal concatenation
- Handles chains of transformers
- Supports artifact persistence for prediction mode

**Cons**:
- Only operates on features, not predictions
- No alignment checking for different feature dimensions
- Naming can get complex with nested concats

**Relationship to branching**:
- `concat_transform` is conceptually equivalent to "branch + merge features" in a single step
- But it's more efficient (no context multiplication) and explicit about intent

### 3.3 MetaModel / Stacking

**Location**:
- `nirs4all/operators/models/meta.py` (MetaModel operator)
- `nirs4all/controllers/models/stacking/reconstructor.py` (OOF reconstruction)
- `nirs4all/controllers/models/stacking/crossbranch.py` (cross-branch validation)

**What it does**:
- Collects out-of-fold (OOF) predictions from source models
- Builds meta-feature matrix with OOF safeguards
- Trains meta-learner on these features
- Supports cross-branch stacking (BranchScope.ALL_BRANCHES)

**Key OOF safeguard**:
```python
# For each training sample, use ONLY predictions from folds
# where that sample was NOT used for training
for fold_id, fold_predictions in enumerate(val_predictions):
    for sample_idx in fold_predictions.sample_indices:
        X_train_meta[sample_idx] = fold_predictions.y_pred[sample_idx]
```

**Critical insight**: This OOF reconstruction is what prevents data leakage. Without it, stacking would overfit because the meta-learner would see predictions from models that already trained on those samples.

**Cross-branch validation**:
```python
class CrossBranchCompatibility(Enum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE_SAMPLES = "incompatible_samples"  # sample_partitioner
    INCOMPATIBLE_PARTITIONS = "incompatible_partitions"
```

**Pros**:
- Proper OOF handling prevents leakage
- Supports multi-level stacking
- Cross-branch validation catches incompatible branches

**Cons**:
- Syntax is different from a hypothetical "merge predictions"
- User must understand MetaModel is the only safe way to combine predictions

### 3.4 Feature Augmentation Mode

**Mechanism**: `context.metadata.add_feature = True`

When active, transformers ADD new processings instead of REPLACING:
```python
# Normal mode: replace "raw" with "raw_MinMaxScaler_0"
# Add mode: keep "raw", add "minmax" as new processing
```

This creates a 3D structure: `(samples, processings, features)`

**Relationship to concat_transform**:
- `feature_augmentation` creates multiple PROCESSINGS (axis=1)
- `concat_transform` creates one processing with MORE FEATURES (axis=2)
- These are orthogonal operations

### 3.5 Identified Redundancies and Conflicts

#### Conflict 1: Branch Merge vs Concat Transform

| Aspect | `branch` + hypothetical merge | `concat_transform` |
|--------|-------------------------------|---------------------|
| Intent | "Try different preprocessings" | "Combine different views" |
| Implementation | N contexts → merge | Single context, horizontal concat |
| Efficiency | N × overhead | 1 × overhead |
| Feature result | Same | Same |

**Verdict**: For feature-level concatenation, `concat_transform` is strictly better. A "merge features" operation would be redundant.

#### Conflict 2: Merge Predictions vs MetaModel

| Aspect | Hypothetical "merge predictions" | MetaModel stacking |
|--------|----------------------------------|---------------------|
| Intent | "Combine branch model outputs" | "Stack models with OOF" |
| Data leakage | YES (fatal flaw) | NO (OOF safeguards) |
| Implementation | Simple concat | Complex reconstruction |

**Verdict**: "Merge predictions" without OOF safeguards is DANGEROUS and should not be implemented. MetaModel is the correct solution.

#### Conflict 3: Concat Transform vs Feature Augmentation

| Aspect | `concat_transform` | `feature_augmentation` |
|--------|---------------------|------------------------|
| Result axis | Features (axis=2) | Processings (axis=1) |
| Shape change | (N, P, F) → (N, P, F×M) | (N, P, F) → (N, P×M, F) |
| Use case | Sensor fusion, multi-view | Preprocessing variants |

**Verdict**: These are NOT redundant - they operate on different axes. Documentation should clarify this.

#### Bug: Empty merge.py

The file exists but is empty. This suggests an incomplete implementation that should either be:
1. Removed entirely (if merge is not needed)
2. Implemented properly (if merge is needed)

**Recommendation**: Delete merge.py and document that merging is handled by:
- `concat_transform` for features
- `MetaModel` for predictions

---

## Conceptual Framework

### 4.1 Fundamental Operations Taxonomy

After analysis, we identify four fundamental operations:

| Operation | Input | Output | Axis | Safe? |
|-----------|-------|--------|------|-------|
| **Transform** | Features | Features | Same shape or reduced | ✅ |
| **Augment** | Features | Features + new processings | Adds processing axis | ✅ |
| **Concat** | Multiple feature views | Single concatenated view | Expands feature axis | ✅ |
| **Stack** | Multiple predictions | Meta-features → prediction | New feature space | ✅ with OOF |

### 4.2 When Each Operation Applies

```
Single preprocessing path:
    X → Transform → Transform → Model → Predictions

Multiple preprocessing variants (axis=1):
    X → feature_augmentation([T1, T2, T3]) → Model (sees 3D data)
        Creates: (N, 3, F)

Feature concatenation (axis=2):
    X → concat_transform([T1, T2, T3]) → Model
        Creates: (N, 1, F*3)

Parallel pipelines with independent models:
    X → branch([
            [T1, Model1],
            [T2, Model2],
        ])
        Creates: 2 independent prediction sets

Prediction stacking (safe):
    X → branch([...]) → Model → MetaModel(source_models="all")
        Uses OOF predictions from previous models
```

### 4.3 Data Leakage Considerations

**Why merging predictions is dangerous**:

```
Fold structure:
    Fold 1: Train on samples [1-80], Validate on [81-100]
    Fold 2: Train on samples [1-20, 41-100], Validate on [21-40]
    ...

Safe stacking (OOF):
    For sample 50:
        Use prediction from Fold 1 (sample 50 was in validation)
        NOT from Fold 2 (sample 50 was in training)

Unsafe merge:
    For sample 50:
        Use ALL fold predictions (includes leak from training folds)
```

**The leak happens because**: Models trained on sample 50 will predict sample 50 too well (memorization). Using these predictions to train a meta-model overfits.

---

## Design Proposal: Unified Branch-Merge Architecture

### Initial Proposal (Step 1)

I propose the following unified design:

1. **Keep `branch` as-is** for parallel preprocessing
2. **Remove `merge.py`** entirely - it's a conceptual trap
3. **Document that branches stay independent** unless explicitly stacked
4. **Enhance `MetaModel`** to be the canonical "merge predictions" mechanism
5. **Keep `concat_transform`** for feature-level concatenation

New syntax for branch + stack:
```python
# Clear intent: branches run independently, then stack with OOF
pipeline = [
    {"branch": [[SNV(), PLS()], [MSC(), RF()]]},
    {"model": MetaModel(
        model=Ridge(),
        branch_scope=BranchScope.ALL_BRANCHES
    )}
]
```

### Verification Questions (Step 2)

**Q1**: What if a user wants to merge features from branches (not predictions)?

**Q2**: How does `BranchScope.ALL_BRANCHES` interact with sample_partitioner branches that have disjoint samples?

**Q3**: What happens if a user wants to train a SINGLE model on concatenated features from multiple branches?

### Answers to Verification Questions (Step 3)

**A1: Merging features from branches**

This is a valid use case: "I want SNV features + MSC features as input to one model."

Current workarounds:
1. Use `concat_transform([SNV(), MSC()])` - achieves same result more efficiently
2. Use multi-source with source-specific preprocessing

The branch → merge_features path is redundant with `concat_transform`. However, if the user has already created branches, they're stuck.

**Proposed solution**: Add a `merge_features` step that exits branch mode and concatenates:
```python
{"branch": [[SNV(), PCA(50)], [MSC(), PCA(50)]]},
{"merge": "features"},  # Exit branch mode, concat features
PLSRegression()  # Single model on concatenated features
```

But this requires careful design to avoid confusion with prediction merge.

**A2: sample_partitioner + ALL_BRANCHES**

The `CrossBranchValidator` already handles this:
```python
if has_sample_partitioner:
    result.compatibility = CrossBranchCompatibility.INCOMPATIBLE_PARTITIONS
    result.add_error("Cross-branch stacking not supported with sample_partitioner")
```

This is correct. Sample partitioner creates disjoint sample sets that cannot be stacked.

**A3: Single model on concatenated branch features**

This requires exiting branch mode. Current workaround is clunky:
```python
# Current: concat_transform does this in one step
{"concat_transform": [SNV(), MSC()]}

# If user already branched, no clean exit
{"branch": [[SNV()], [MSC()]]},
# ??? No way to merge and continue with single model
```

**Insight**: We need a "branch exit" mechanism that either:
- Merges features (for preprocessing-only branches)
- Requires MetaModel (for model-containing branches)

### Final Proposal (Step 4)

Based on verification, here's the refined design:

#### 5.1 Design Principles

1. **Explicit intent**: Syntax must clearly indicate merge type
2. **No unsafe merges**: Prediction merges ONLY through MetaModel
3. **Clean exit**: Provide mechanism to exit branch mode
4. **Efficiency**: Prefer concat_transform when possible

#### 5.2 Proposed Syntax

**Option A: Merge as explicit step (RECOMMENDED)**

```python
# Feature merge: exits branch mode, concatenates features
{"branch": [[SNV()], [MSC()]]},
{"merge": "features"},  # Returns to single execution path
PLSRegression()  # Single model

# Prediction stack: uses MetaModel (unchanged)
{"branch": [[SNV(), PLS()], [MSC(), RF()]]},
{"model": MetaModel(model=Ridge(), branch_scope=BranchScope.ALL_BRANCHES)}
```

**Option B: Merge as branch option**

```python
# Auto-merge features at end of branch
{"branch": [[SNV()], [MSC()]], "merge": "features"},
PLSRegression()

# No merge = stay in branch mode (current behavior)
{"branch": [[SNV()], [MSC()]]},
PLSRegression()  # Runs on EACH branch
```

**Recommendation**: Option A is clearer about when merge happens.

#### 5.3 Merge Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| `"features"` | Horizontal concat of transformed features | Preprocessing-only branches |
| `"predictions"` | **NOT ALLOWED** - use MetaModel | - |
| `"passthrough"` | Keep branches separate (default) | When you want N independent models |

#### 5.4 Fold-Aware Merge

The `"features"` merge is safe because:
- It happens BEFORE any model training
- Features are deterministic transforms of input
- No prediction leakage is possible

The key invariant: **Never merge predictions without OOF safeguards**.

---

## Design Proposal: Concat-Transform Clarification

### Initial Proposal (Step 1)

Keep `concat_transform` as-is with documentation improvements:

```python
# concat_transform: horizontal concatenation of transformed features
{"concat_transform": [
    PCA(n_components=50),
    SVD(n_components=50),
    None,  # Pass through original features
]}
# Result: [PCA_features | SVD_features | original_features]
```

Document the relationship to branching:
- `concat_transform` = efficient way to concat feature views
- `branch` + `merge: "features"` = equivalent but less efficient
- Use `concat_transform` when you don't need intermediate steps between transforms

### Verification Questions (Step 2)

**Q1**: How does `concat_transform` handle different output shapes from transforms?

**Q2**: Should `concat_transform` support generator syntax for combinatorial exploration?

**Q3**: How does `concat_transform` interact with multi-source datasets?

### Answers to Verification Questions (Step 3)

**A1: Different output shapes**

Current implementation uses `np.hstack()` which handles different widths:
```python
# PCA(50) → 50 features
# SVD(30) → 30 features
# np.hstack → 80 features
```

This works correctly. No issue.

**A2: Generator syntax**

Looking at the code, `concat_transform` already supports generator syntax through `normalize_generator_spec()`:
```python
{"concat_transform": {"_or_": [PCA, SVD, ICA], "pick": 2}}
# Generates combinations: [PCA, SVD], [PCA, ICA], [SVD, ICA]
```

This is already implemented. No change needed.

**A3: Multi-source datasets**

Current implementation iterates over sources:
```python
for sd_idx in range(n_sources):
    processing_ids = list(dataset.features_processings(sd_idx))
    # Apply concat_transform per source
```

This is correct - each source is processed independently.

### Final Proposal (Step 4)

**No code changes needed for concat_transform**. Only documentation improvements:

1. Add clear comparison table: concat_transform vs branch+merge vs feature_augmentation
2. Add examples showing when to use each
3. Document generator syntax support

---

## Design Proposal: Stacking Integration

### Initial Proposal (Step 1)

Enhance MetaModel to be the canonical prediction-merge mechanism:

1. Add `branch_scope=BranchScope.ALL_BRANCHES` as prominent option
2. Improve error messages when user attempts unsafe merge
3. Add validation to detect branch-model combinations that need MetaModel

### Verification Questions (Step 2)

**Q1**: What if a user wants to merge predictions from branches WITHOUT training a meta-model (e.g., just average them)?

**Q2**: How do we handle the case where branches have different numbers of folds?

**Q3**: Should MetaModel automatically detect when it needs cross-branch scope?

### Answers to Verification Questions (Step 3)

**A1: Simple prediction aggregation**

Use cases:
- Ensemble averaging: `(pred_branch_1 + pred_branch_2) / 2`
- Weighted averaging: based on validation scores

This is a legitimate use case that doesn't require a meta-learner. However, it still has leakage concerns unless we use OOF predictions.

**Proposed solution**: Add aggregation mode to MetaModel:
```python
MetaModel(
    model=None,  # No meta-learner
    aggregation="mean",  # or "weighted_mean", "vote"
    branch_scope=BranchScope.ALL_BRANCHES
)
```

This uses the OOF reconstruction machinery but outputs aggregated predictions instead of training a model.

**A2: Different fold counts**

The `FoldAlignmentValidator` already handles this:
```python
fold_counts = {name: info['n_folds'] for name, info in model_fold_info.items()}
if len(unique_counts) > 1:
    result.add_error("FOLD_COUNT_MISMATCH", f"Source models have different fold counts")
```

This is a validation error that must be fixed by the user.

**A3: Auto-detect cross-branch scope**

This is risky because:
- User might not realize they're doing cross-branch stacking
- Could mask sample_partitioner incompatibility issues

**Recommendation**: Keep explicit `branch_scope` parameter. Add helpful error message when branches exist but scope is CURRENT_ONLY.

### Final Proposal (Step 4)

1. **Add aggregation mode** to MetaModel:
```python
MetaModel(
    model=Ridge(),  # or None for pure aggregation
    aggregation="mean",  # "mean", "weighted_mean", "vote", None
    branch_scope=BranchScope.ALL_BRANCHES
)
```

2. **Add detection/warning** when branches exist:
```python
# In MetaModelController.execute():
if branch_contexts and self.stacking_config.branch_scope == BranchScope.CURRENT_ONLY:
    logger.warning(
        "Pipeline has branches but MetaModel uses CURRENT_ONLY scope. "
        "Consider branch_scope=BranchScope.ALL_BRANCHES to stack across branches."
    )
```

3. **Document the relationship**:
- `MetaModel` is the ONLY safe way to combine predictions from branches
- Simple aggregation is available through `model=None, aggregation="mean"`

---

## Implementation Roadmap

### Phase 1: Cleanup (Immediate)

1. **Delete `merge.py`** - empty file serves no purpose
2. **Add documentation** explaining branch/concat/stack relationships
3. **Add validation** to warn about unsafe prediction combinations

### Phase 2: Feature Merge (Short-term)

1. **Implement `{"merge": "features"}`** to exit branch mode
2. **Add tests** for feature merge scenarios
3. **Update examples** to show proper merge usage

### Phase 3: MetaModel Enhancement (Medium-term)

1. **Add aggregation mode** (`model=None, aggregation="mean"`)
2. **Add branch detection warning**
3. **Improve cross-branch validation error messages**

### Phase 4: Documentation (Ongoing)

1. Create "Branching and Merging Guide" user documentation
2. Add decision flowchart for choosing concat vs branch
3. Add troubleshooting section for common leakage mistakes

---

## Migration Guide

### For Users

**If you were planning to use `merge_predictions`**:
Use `MetaModel` instead:
```python
# Instead of (hypothetical):
{"merge": "predictions"}

# Use:
{"model": MetaModel(model=Ridge(), branch_scope=BranchScope.ALL_BRANCHES)}

# Or for simple averaging:
{"model": MetaModel(model=None, aggregation="mean")}
```

**If you want to combine features from different transforms**:
Use `concat_transform`:
```python
# Preferred:
{"concat_transform": [SNV(), MSC(), FirstDerivative()]}

# Equivalent but less efficient:
{"branch": [[SNV()], [MSC()], [FirstDerivative()]]},
{"merge": "features"}
```

### For Developers

**Removing merge.py**:
- File is empty, no code changes needed
- Update imports if any exist (none found)
- Update `__init__.py` if exported (not exported)

**Adding feature merge**:
- Implement in `nirs4all/controllers/data/merge_features.py`
- Register with `@register_controller`
- Add tests in `tests/unit/controllers/data/`

---

## Appendix: Decision Log

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Delete merge.py | Empty file, concept trap | Implement full merge (rejected: redundant) |
| Feature merge via explicit step | Clear intent, composable | Branch option (rejected: less visible) |
| No prediction merge | Data leakage risk | OOF merge (rejected: duplicates MetaModel) |
| Keep concat_transform | Efficient, well-tested | Merge into branch (rejected: different use case) |
| MetaModel aggregation mode | Enables safe simple ensembles | Separate operator (rejected: code duplication) |

---

## Summary

This design document establishes a clean conceptual model for nirs4all's branching and merging operations:

| Operation | Keyword | Purpose | Safe? |
|-----------|---------|---------|-------|
| Parallel preprocessing | `branch` | Independent contexts | ✅ |
| Feature concatenation | `concat_transform` | Horizontal feature concat | ✅ |
| Exit branch (features) | `merge: "features"` | Combine branch features | ✅ |
| Prediction stacking | `MetaModel` | OOF-safe prediction ensemble | ✅ |
| Simple prediction merge | (not allowed) | - | ❌ Leakage |

The key insight is that **prediction merging without OOF safeguards is fundamentally unsafe** and should never be allowed. MetaModel provides the correct mechanism for all prediction-combining use cases.
