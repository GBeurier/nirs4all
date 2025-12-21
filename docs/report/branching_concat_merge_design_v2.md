# Branching, Concat-Transform, and Merge: Unified Design Document v2

**Version**: 2.0.0
**Status**: Complete Design Proposal
**Date**: December 2025
**Author**: Architecture Review

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Analysis - Gaps in Current Design](#2-problem-analysis---gaps-in-current-design)
3. [Fundamental Concepts](#3-fundamental-concepts)
4. [Current State Analysis](#4-current-state-analysis)
5. [Design: MergeController (NEW)](#5-design-mergecontroller-new)
6. [Design: ConcatTransformController (UNCHANGED)](#6-design-concattransformcontroller-unchanged)
7. [Design: MetaModel Refactoring](#7-design-metamodel-refactoring)
8. [Complete Use Case Matrix](#8-complete-use-case-matrix)
9. [Implementation Specification](#9-implementation-specification)
10. [Migration Guide](#10-migration-guide)

---

## 1. Executive Summary

This document provides a complete redesign of nirs4all's branching and merging system to address critical gaps:

| Gap Identified | Current Status | Solution |
|----------------|----------------|----------|
| Can't mix features + predictions | ❌ Impossible | New `merge` controller |
| Can't select specific models from branches | ⚠️ Partial (source_models) | Enhanced `merge` syntax |
| No explicit branch exit point | ❌ Branches continue forever | `merge` exits branch mode |
| Asymmetric branches (features vs models) | ❌ Undefined behavior | `merge` handles explicitly |
| merge.py is empty | ❌ Dead code | Implement full MergeController |

**Key Innovation**: The `merge` controller becomes the **single unification point** for all branch outputs, handling features, predictions, or both, with mandatory OOF safeguards for predictions.

**Three Controllers - Clear Responsibilities**:

| Controller | Scope | Input | Output | Branch-Aware? |
|------------|-------|-------|--------|---------------|
| `concat_transform` | Single path | Transformers | Concatenated features | No |
| `merge` | Branch exit | Branch contexts | Unified features/predictions | Yes (exits) |
| `MetaModel` | Convenience | Predictions | Stacked predictions | Delegates to merge |

---

## 2. Problem Analysis - Gaps in Current Design

### 2.1 Scenario 1: Asymmetric Branches

```python
{"branch": [[SNV(), PLS()], [PCA(10)]]},
{"model": MetaModel(model=Ridge(), branch_scope=BranchScope.ALL_BRANCHES)}
```

**User Intent**: Train Ridge on [PLS_predictions | PCA_features] (11 dimensions)

**Current Behavior**:
- Branch 0: SNV → PLS → stores predictions in prediction_store
- Branch 1: PCA(10) → no model, no predictions stored
- MetaModel looks for predictions from ALL_BRANCHES
- MetaModel finds: only PLS predictions from branch 0
- MetaModel fails or produces incorrect result (only 1D input)

**Root Cause**: MetaModel only collects predictions, not features. Branch 1 has no predictions.

### 2.2 Scenario 2: Selective Model Stacking

```python
{"branch": [[SNV(), PLS(), CNN(), RF()], [PCA(10)]]},
{"model": MetaModel(model=Ridge(), source_models=["RF"])}
```

**User Intent**: Stack only RF predictions, not PLS/CNN

**Current Behavior**:
- Branch 0: Runs 3 models sequentially: PLS → CNN → RF
- All 3 models store predictions
- MetaModel with `source_models=["RF"]` correctly selects only RF
- BUT: Branch 1 still has no predictions → undefined behavior

**Root Cause**: No way to say "RF predictions from branch 0 + features from branch 1"

### 2.3 Scenario 3: Post-Branch Model + MetaModel

```python
{"branch": [[SNV(), PLS(), CNN(), RF()], [PCA(10)]]},
{"model": CNN()},  # Post-branch model
{"model": MetaModel(model=Ridge(), branch_scope=BranchScope.ALL_BRANCHES)}
```

**Current Behavior**:
1. Branch 0: SNV → PLS → CNN → RF (3 models, 3 prediction sets)
2. Branch 1: PCA(10) (no predictions)
3. Post-branch CNN: Runs on **EACH** branch:
   - On Branch 0: CNN trained on SNV features (RF didn't transform features)
   - On Branch 1: CNN trained on PCA features
   - Creates 2 more prediction sets
4. MetaModel with ALL_BRANCHES sees: PLS, CNN(in-branch), RF, CNN(branch-0-post), CNN(branch-1-post)
5. **Total: 5 prediction sets!** User probably wanted 2-3.

**Root Cause**: No clarity on which predictions to include. Post-branch models multiply across branches.

### 2.4 The Core Problem

**There is no explicit merge/exit point for branches.**

Current behavior:
- `branch` creates N parallel contexts
- All subsequent steps run N times (once per branch)
- No way to "exit" branch mode and return to single-path execution
- No way to combine branch outputs into a single input for the next step

---

## 3. Fundamental Concepts

### 3.1 Data Types in Pipeline

| Data Type | Storage | Safety for Merge |
|-----------|---------|------------------|
| **Features** | `dataset._features.sources` | ✅ Safe (deterministic transforms) |
| **Predictions** | `prediction_store` | ⚠️ Requires OOF reconstruction |
| **Targets** | `dataset.y()` | N/A (not merged) |

### 3.2 Why Prediction Merge Requires OOF

Without OOF (Out-of-Fold) reconstruction:

```
Training sample 50:
  - Model trained on fold containing sample 50 → overfits to sample 50
  - Model predicts sample 50 → prediction is too optimistic
  - Meta-model trains on this prediction → learns to trust overfit predictions
  → DATA LEAKAGE
```

With OOF reconstruction:

```
Training sample 50:
  - Use prediction from fold where sample 50 was in VALIDATION set
  - This prediction came from a model that never saw sample 50
  → NO LEAKAGE
```

**Invariant**: Any operation that uses predictions for training MUST use OOF reconstruction.

### 3.3 Operations Taxonomy

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE FLOW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Features ──► Transform ──► Features (same/reduced dims)        │
│                                                                   │
│  Features ──► feature_augmentation ──► Features + processings   │
│               (adds processing axis)                              │
│                                                                   │
│  Features ──► concat_transform ──► Features (expanded dims)     │
│               (horizontal concat)                                 │
│                                                                   │
│  Features ──► branch ──► N parallel contexts                    │
│                          (creates branch mode)                    │
│                                                                   │
│  N contexts ──► merge ──► 1 context                             │
│                 (exits branch mode, combines outputs)             │
│                                                                   │
│  Features ──► Model ──► Predictions (stored)                    │
│                                                                   │
│  Predictions ──► MetaModel ──► Stacked Predictions              │
│                  (OOF-safe)                                       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Current State Analysis

### 4.1 BranchController ✅ Keep As-Is

**Location**: `nirs4all/controllers/data/branch.py`

**Behavior**:
- Creates N independent execution contexts
- Stores in `context.custom["branch_contexts"]`
- Each branch has `features_snapshot` after processing
- Post-branch steps iterate over all contexts

**No changes needed** - branch creation is well-designed.

### 4.2 ConcatAugmentationController ✅ Keep As-Is

**Location**: `nirs4all/controllers/data/concat_transform.py`

**Behavior**:
- Applies multiple transformers, concatenates results horizontally
- Works within single execution path
- No branch awareness

**No changes needed** - serves its purpose for single-path feature concat.

### 4.3 MergeController ❌ Empty - Must Implement

**Location**: `nirs4all/controllers/data/merge.py`

**Current State**: Empty file

**Required**: Full implementation (see Section 5)

### 4.4 MetaModelController ⚠️ Refactor

**Location**: `nirs4all/controllers/models/meta_model.py`

**Current Issues**:
- Complex parameter surface (source_models, branch_scope, stacking_config)
- Mixes OOF reconstruction with model training
- Doesn't handle asymmetric branches (features + predictions)

**Required**: Refactor to delegate branch handling to MergeController (see Section 7)

---

## 5. Design: MergeController (NEW)

### 5.1 Initial Proposal

The `merge` controller is the **explicit exit point** from branch mode that:
1. Collects outputs from specified branches
2. Combines features and/or predictions
3. Creates a single unified context
4. Exits branch mode (clears branch_contexts)

### 5.2 Verification Questions

**Q1**: What if a branch has BOTH preprocessing and a model? Which is used?

**Q2**: How does merge handle nested branches?

**Q3**: What happens to branches not included in merge?

### 5.3 Answers

**A1**: Merge type determines collection:
- `"features"` → Uses feature snapshot (after all transforms, ignores models)
- `"predictions"` → Uses model OOF predictions (requires models in branch)
- Both can be combined with explicit branch selection

**A2**: Nested branches are flattened. Merge sees all leaf-level branches.
- Branch 0_0, 0_1 (from nested) are individual merge sources
- User can select by flattened index or name

**A3**: Merge exits ALL branches:
- Specified branches contribute to merged output
- Non-specified branches are dropped (their outputs not used)
- All branch mode is exited regardless

### 5.4 Final Syntax Specification

```python
# =============================================================================
# SIMPLE FORMS
# =============================================================================

# Merge features from all branches (horizontal concat)
{"merge": "features"}

# Merge predictions from all branches (OOF reconstruction)
{"merge": "predictions"}

# Merge both features and predictions from all branches
{"merge": "all"}

# =============================================================================
# SELECTIVE FORMS
# =============================================================================

# Merge features from specific branches
{"merge": {"features": [0, 2]}}  # Branch indices

# Merge predictions from specific branches
{"merge": {"predictions": [1, 3]}}

# Mix features from some branches, predictions from others
{"merge": {
    "features": [1],      # Features from branch 1
    "predictions": [0]    # Predictions from branch 0
}}

# =============================================================================
# FULL CONTROL
# =============================================================================

{"merge": {
    "features": {
        "branches": [1, 2],           # Which branches
        "processings": ["raw", "snv"], # Optional: which processings (future)
    },
    "predictions": {
        "branches": [0],              # Which branches
        "models": ["RF", "XGB"],      # Optional: filter by model name
        "aggregation": "oof",         # "oof" (train) or "mean"/"weighted" (test)
    },
    "include_original": False,        # Include pre-branch features?
    "on_missing": "error",            # "error" or "ignore" if branch has no output
}}
```

### 5.5 Behavior Specification

| Scenario | `merge: "features"` | `merge: "predictions"` | `merge: "all"` |
|----------|---------------------|------------------------|----------------|
| Branch with only transforms | ✅ Concat features | ❌ Error (no model) | ✅ Features only |
| Branch with transform + model | ✅ Concat features (ignore model) | ✅ OOF predictions | ✅ Both |
| Branch with only model | ✅ Original features | ✅ OOF predictions | ✅ Both |
| Post-branch model | ✅ Features after model* | ✅ OOF predictions | ✅ Both |

*Models don't transform features, so "features after model" = same as before model

### 5.6 Output Handling

Merge produces a **new processing** called `"merged"` containing:
- For features: horizontal concat of branch feature snapshots
- For predictions: OOF-reconstructed prediction matrix
- For both: [features | predictions] concatenated

**Example**:
```python
{"branch": [[SNV(), PLS()], [PCA(10)]]},
{"merge": {"features": [1], "predictions": [0]}}
```

Output shape: `(n_samples, 1 + 10)` = [PLS_oof | PCA_features]

### 5.7 Prediction Mode Support

In prediction/explain mode:
- Features: Apply saved transformers, concat as during training
- Predictions: Load models, get test predictions, aggregate (mean/weighted)
- Merge config is saved in manifest for reproducibility

---

## 6. Design: ConcatTransformController (UNCHANGED)

### 6.1 Current Behavior (Keep)

```python
{"concat_transform": [PCA(50), SVD(30), None]}
```

- Applies each transformer to current features
- Concatenates results horizontally: `[PCA_out | SVD_out | original]`
- Works within single execution path
- No branch awareness

### 6.2 Relationship to Merge

| Aspect | `concat_transform` | `merge: "features"` |
|--------|-------------------|---------------------|
| Scope | Single path | Exits branch mode |
| Input | Transformers | Branch contexts |
| Efficiency | 1× execution | N× branch execution + merge |
| Use when | Same data, different transforms | Different preprocessing pipelines |

**Equivalence**:
```python
# These produce the same result:
{"concat_transform": [SNV(), MSC(), D1()]}

# vs
{"branch": [[SNV()], [MSC()], [D1()]]},
{"merge": "features"}
```

The first is more efficient; use `concat_transform` when you don't need intermediate branch steps.

### 6.3 No Changes Needed

`concat_transform` is well-designed for its purpose. Keep as-is.

---

## 7. Design: MetaModel Refactoring

### 7.1 Initial Proposal

Refactor MetaModel to delegate branch/merge logic:

**Current**: MetaModel handles everything
- Source model selection
- Branch scope handling
- OOF reconstruction
- Meta-learner training

**Proposed**: MetaModel = merge + train
- Merge handles: source selection, branch scope, OOF reconstruction
- MetaModel handles: just the meta-learner training

### 7.2 Verification Questions

**Q1**: Should MetaModel continue to exist or be deprecated?

**Q2**: How do existing pipelines migrate?

**Q3**: What about MetaModel-specific features (use_proba, aggregation)?

### 7.3 Answers

**A1**: Keep MetaModel for convenience and backward compatibility:
- Common case (stack all + train) should remain simple
- MetaModel internally translates to merge + train
- Power users can use explicit merge for complex cases

**A2**: Existing pipelines work unchanged:
- MetaModel API is preserved
- Implementation changes are internal
- New capabilities available via explicit merge

**A3**: MetaModel-specific features move to merge config:
- `use_proba` → `merge: {predictions: {proba: true}}`
- `aggregation` → `merge: {predictions: {aggregation: "weighted"}}`
- MetaModel keeps these params, passes to internal merge

### 7.4 Final Design

**MetaModel Behavior**:
1. If explicit merge step exists before MetaModel → use merged features
2. If no merge step → MetaModel internally creates merge config from its params
3. Train meta-learner on merged output

**MetaModel Parameter Mapping**:

| MetaModel Param | Internal Merge Config |
|-----------------|----------------------|
| `source_models="all"` | `{"predictions": True}` |
| `source_models=["RF"]` | `{"predictions": {"models": ["RF"]}}` |
| `branch_scope=ALL_BRANCHES` | `{"predictions": {"branches": "all"}}` |
| `branch_scope=CURRENT_ONLY` | `{"predictions": {"branches": [current]}}` |
| `use_proba=True` | `{"predictions": {"proba": True}}` |

**New Capability via Explicit Merge**:

```python
# BEFORE (impossible):
# Mix features and predictions from different branches

# AFTER (possible):
{"branch": [[SNV(), PLS()], [PCA(10)]]},
{"merge": {"features": [1], "predictions": [0]}},
{"model": Ridge()}  # Trains on [PLS_oof | PCA_features]
```

---

## 8. Complete Use Case Matrix

### 8.1 Feature-Only Pipelines

| Use Case | Syntax |
|----------|--------|
| Single transform | `SNV()` |
| Transform chain | `SNV(), PCA(50)` |
| Concat transforms (same path) | `{"concat_transform": [SNV(), MSC()]}` |
| Multiple processings | `{"feature_augmentation": [SNV(), MSC()]}` |
| Branch + merge features | `{"branch": [[SNV()], [MSC()]]}, {"merge": "features"}` |

### 8.2 Model Pipelines

| Use Case | Syntax |
|----------|--------|
| Single model | `{"model": PLSRegression()}` |
| Multiple models (all predictions kept) | `PLS(), RF(), XGB()` |
| Branch preprocessing + single model | `{"branch": [[SNV()], [MSC()]]}, PLS()` → runs 2× |
| Branch preprocessing + merge + single model | `{"branch": [[SNV()], [MSC()]]}, {"merge": "features"}, PLS()` → runs 1× |

### 8.3 Stacking Pipelines

| Use Case | Syntax |
|----------|--------|
| Stack all previous models | `PLS(), RF(), {"model": MetaModel(Ridge())}` |
| Stack selected models | `PLS(), RF(), XGB(), {"model": MetaModel(Ridge(), source_models=["RF", "XGB"])}` |
| Stack across branches | `{"branch": [[SNV(), PLS()], [MSC(), RF()]]}, {"merge": "predictions"}, Ridge()` |
| **NEW**: Mix features + predictions | `{"branch": [[SNV(), PLS()], [PCA(10)]]}, {"merge": {"features": [1], "predictions": [0]}}, Ridge()` |

### 8.4 Complex Pipelines

| Use Case | Syntax |
|----------|--------|
| Nested branches + merge | `{"branch": [[{"branch": [...]}, PLS()], [...]]}, {"merge": "predictions"}` |
| Post-branch model + merge | `{"branch": [[SNV()], [MSC()]]}, PLS(), {"merge": "predictions"}` |
| Selective model from multi-model branch | `{"branch": [[SNV(), PLS(), RF(), XGB()]]}, {"merge": {"predictions": {"models": ["XGB"]}}}` |

---

## 9. Implementation Specification

### 9.1 MergeController Implementation

```python
@register_controller
class MergeController(OperatorController):
    """Controller for merging branch outputs.

    Handles:
    - Feature concatenation from branches
    - OOF prediction reconstruction from branches
    - Mixed feature+prediction merging
    - Branch mode exit
    """

    priority = 5  # High priority to catch merge keyword

    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        return keyword == "merge"

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return True

    def execute(
        self,
        step_info,
        dataset,
        context,
        runtime_context,
        source=-1,
        mode="train",
        loaded_binaries=None,
        prediction_store=None
    ):
        config = self._parse_merge_config(step_info.original_step["merge"])
        branch_contexts = context.custom.get("branch_contexts", [])

        if not branch_contexts:
            raise ValueError("merge requires active branch contexts")

        merged_features = []

        # Collect features from specified branches
        if config.collect_features:
            for branch_id in config.feature_branches:
                branch_info = self._get_branch(branch_contexts, branch_id)
                features = self._extract_features(dataset, branch_info)
                merged_features.append(features)

        # Collect OOF predictions from specified branches
        if config.collect_predictions:
            reconstructor = TrainingSetReconstructor(
                prediction_store=prediction_store,
                source_model_names=config.model_names,
                stacking_config=config.stacking_config
            )
            oof_result = reconstructor.reconstruct(
                dataset=dataset,
                context=context,
                branch_ids=config.prediction_branches
            )
            merged_features.append(oof_result.X_train_meta)

        # Horizontal concat all collected outputs
        final_features = np.hstack(merged_features)

        # Store as new processing "merged"
        dataset.add_merged_features(final_features, processing_name="merged")

        # Exit branch mode
        context.custom["branch_contexts"] = []
        context.custom["in_branch_mode"] = False

        # Update context to select merged processing
        context = context.with_processing([["merged"]])

        return context, []  # No artifacts (merge is logic-only)
```

### 9.2 TrainingSetReconstructor Enhancement

Add method for branch-specific reconstruction:

```python
def reconstruct_for_branches(
    self,
    dataset: SpectroDataset,
    context: ExecutionContext,
    branch_ids: Optional[List[int]] = None,
    model_names: Optional[List[str]] = None
) -> ReconstructionResult:
    """Reconstruct OOF predictions for specific branches and models."""

    # Filter source models by branch and name
    filtered_sources = self._filter_sources(
        branch_ids=branch_ids,
        model_names=model_names,
        context=context
    )

    # Existing reconstruction logic with filtered sources
    return self._reconstruct_with_sources(
        dataset=dataset,
        context=context,
        source_model_names=filtered_sources
    )
```

### 9.3 MetaModel Refactoring

```python
class MetaModel(BaseModelOperator):
    """Wrapper for meta-model stacking.

    Internally delegates to MergeController for OOF reconstruction,
    then trains the meta-learner.
    """

    def get_controller_type(self) -> str:
        return "meta"

    def to_merge_config(self) -> Dict:
        """Convert MetaModel params to merge config."""
        return {
            "predictions": {
                "branches": "all" if self.branch_scope == BranchScope.ALL_BRANCHES else "current",
                "models": self.source_models if self.source_models != "all" else None,
                "proba": self.use_proba,
                "aggregation": self.stacking_config.test_aggregation.value,
            }
        }
```

### 9.4 File Changes Summary

| File | Action |
|------|--------|
| `nirs4all/controllers/data/merge.py` | **Implement** full MergeController |
| `nirs4all/controllers/models/meta_model.py` | **Refactor** to use merge internally |
| `nirs4all/controllers/models/stacking/reconstructor.py` | **Enhance** for branch filtering |
| `nirs4all/operators/data/merge.py` | **Create** MergeConfig dataclass |
| `tests/unit/controllers/data/test_merge.py` | **Create** comprehensive tests |

---

## 10. Migration Guide

### 10.1 For Users

**Existing pipelines work unchanged**. New capabilities are additive.

**To use new features**:

```python
# OLD (MetaModel only):
{"model": MetaModel(model=Ridge(), branch_scope=BranchScope.ALL_BRANCHES)}

# NEW (explicit merge for more control):
{"merge": "predictions"},
{"model": Ridge()}

# NEW (mix features and predictions - previously impossible):
{"branch": [[SNV(), PLS()], [PCA(10)]]},
{"merge": {"features": [1], "predictions": [0]}},
{"model": Ridge()}
```

### 10.2 For Developers

**Phase 1**: Implement MergeController (2-3 days)
- Parse merge config
- Collect branch features
- Delegate to TrainingSetReconstructor for predictions
- Exit branch mode

**Phase 2**: Enhance TrainingSetReconstructor (1 day)
- Add branch filtering
- Add model name filtering
- Expose via public method

**Phase 3**: Refactor MetaModel (1 day)
- Convert params to merge config
- Delegate to merge internally
- Keep public API unchanged

**Phase 4**: Tests and Documentation (2 days)
- Unit tests for all merge scenarios
- Integration tests with branches
- User documentation

---

## Appendix A: Decision Log

| Decision | Rationale | Alternatives Rejected |
|----------|-----------|----------------------|
| Merge as explicit step | Clear intent, composable | Branch option (less visible) |
| OOF mandatory for predictions | Data leakage prevention | Optional OOF (dangerous) |
| Keep concat_transform | Efficient for single-path | Merge into merge (slower) |
| Keep MetaModel | Backward compatibility | Deprecate (breaking change) |
| Merge exits ALL branches | Simpler mental model | Selective branch exit (complex) |

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Branch Mode** | Pipeline state with multiple parallel execution contexts |
| **OOF** | Out-of-Fold: predictions from folds where sample was in validation |
| **Feature Snapshot** | Copy of dataset features at a point in time (stored per branch) |
| **Merge** | Exit branch mode by combining branch outputs |
| **Stacking** | Using predictions as features for a meta-learner |
