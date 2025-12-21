# Branching, Concat-Transform, and Merge: Design Review v4

**Version**: 4.1.0
**Status**: Revised Design - Merge as Core Primitive with Per-Branch Control
**Date**: December 2025
**Author**: Design Review

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Verified Current State](#2-verified-current-state)
3. [Problem Analysis](#3-problem-analysis)
4. [Design: MergeController](#4-design-mergecontroller)
5. [Design: MetaModel Integration](#5-design-metamodel-integration)
6. [Design: ConcatTransformController (No Changes)](#6-design-concattransformcontroller-no-changes)
7. [Complete Use Case Matrix](#7-complete-use-case-matrix)
8. [Implementation Specification](#8-implementation-specification)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [Appendix: Relationship with Asymmetric Sources Design](#appendix-relationship-with-asymmetric-sources-design)

---

## 1. Executive Summary

### Design Philosophy: Overlapping Responsibilities by Choice

This design intentionally allows **overlapping responsibilities** between controllers, giving users the freedom to choose their preferred logic and level of control. Rather than enforcing a single "correct" way to accomplish a task, we provide:

- **Multiple equivalent syntaxes** for common operations
- **Composable primitives** that can be combined flexibly
- **Convenience wrappers** for frequent patterns without hiding the underlying mechanics

This approach prioritizes user expressiveness over API minimalism. Users who want simple, safe defaults can use high-level operators like `MetaModel`. Users who need fine-grained control can compose lower-level primitives like `merge` + model training.

### Objectives (Detailed)

#### 1. Establish Merge as the Core Primitive for Branch Combination

**The Problem**: Currently, once a pipeline enters "branch mode", all subsequent steps execute N times (once per branch) with no way to return to single-path execution. There is no explicit mechanism to "exit" branch mode and recombine outputs.

**The Solution**: Implement `MergeController` as the **foundational primitive** for all branch combination operations. Merge handles:
- Exiting branch mode (always, unconditionally)
- Collecting features and/or predictions from branches
- Enforcing OOF (out-of-fold) safety when predictions are involved
- Creating a unified dataset for subsequent steps

**Key Insight**: Stacking is conceptually "merge predictions + train a model". By making merge the primitive:
- Stacking becomes a composition: `merge(predictions) → train(model)`
- Users can use merge alone for data preparation
- Users can combine merge + any model (not just MetaModel)
- The OOF logic lives in one place (merge), reused everywhere

#### 2. Guarantee Data Leakage Prevention with Explicit Opt-Out

**The Problem**: Merging predictions to train a subsequent model without OOF reconstruction causes **data leakage**—the model sees its own predictions on training data, leading to overly optimistic performance estimates.

**The Solution**: When merge includes predictions as inputs, OOF reconstruction is **mandatory by default**. This means:
- Training predictions are reconstructed from validation fold outputs
- Each sample's prediction comes from a model that never saw that sample during training
- The resulting features are "safe" for training any downstream model

**Explicit Unsafe Mode**: For advanced users who understand the implications (e.g., very large datasets where leakage impact is minimal, or rapid prototyping), an `unsafe=True` option disables OOF reconstruction. This option:
- Requires explicit acknowledgment (cannot be set accidentally)
- Generates prominent warnings in logs and outputs
- Is documented with clear explanations of risks
- Should NOT be used for final model evaluation

```python
# Safe (default): OOF reconstruction
{"merge": "predictions"}

# Unsafe (explicit opt-in): Direct predictions (DATA LEAKAGE WARNING)
{"merge": {"predictions": "all", "unsafe": True}}
```

#### 3. Unify Feature and Prediction Merging in One Controller

**The Problem**: The current `MetaModel` can only collect predictions—it cannot mix features from some branches with predictions from others. This limits expressiveness for asymmetric pipelines.

**The Solution**: `MergeController` handles **both** features and predictions uniformly:
- `{"merge": "features"}` — collect and concat features from all branches
- `{"merge": "predictions"}` — collect OOF predictions from all branches
- `{"merge": {"features": [1], "predictions": [0]}}` — mix: features from branch 1, predictions from branch 0

This flexibility enables use cases like:
- Branch 0: SNV → PLS → predictions (1 feature)
- Branch 1: PCA(10) → features only (10 features)
- Merge: `[PLS_oof_predictions | PCA_features]` → 11 features for downstream model

#### 4. Refactor MetaModel as a Convenience Wrapper over Merge

**The Problem**: Currently, `MetaModel` implements its own OOF reconstruction, branch handling, and source selection—duplicating logic that should be centralized in merge.

**The Solution**: Refactor `MetaModel` to be a **thin wrapper** that:
1. Internally creates a merge configuration based on its parameters
2. Delegates to `MergeController` for data preparation
3. Trains the meta-learner on the merged output

**User-Facing Behavior**: Unchanged. Users can still write:
```python
PLS(), RF(), {"model": MetaModel(Ridge())}
```

**Internal Implementation**: MetaModel now does:
```python
# Conceptually:
merged_X = MergeController.merge(predictions=source_models, ...)
meta_model.fit(merged_X, y)
```

**Benefits**:
- Single source of truth for OOF logic
- MetaModel automatically gains new merge capabilities
- Users can achieve the same result with explicit merge + regular model
- Easier testing and maintenance

#### 5. Maintain ConcatTransformController Independence

**The Problem**: `ConcatTransformController` and `MergeController` might seem similar (both concatenate features) but serve different purposes.

**The Solution**: Keep them as **independent, complementary controllers**:

| Aspect | `concat_transform` | `merge` |
|--------|-------------------|---------|
| **Scope** | Single execution path | Exits branch mode |
| **Input** | List of transformers | Branch contexts |
| **Execution** | 1× (parallel transforms on same data) | After N× branch execution |
| **Purpose** | Feature engineering variants | Pipeline variant combination |

**Equivalence for simple cases** (both produce similar results):
```python
# concat_transform: more efficient, single path
{"concat_transform": [SNV(), MSC(), D1()]}

# branch + merge: less efficient but allows intermediate branch steps
{"branch": [[SNV()], [MSC()], [D1()]]}, {"merge": "features"}
```

Use `concat_transform` when you just need parallel transforms on the same data. Use `branch + merge` when you need different intermediate steps in each branch.

### Summary of Key Changes

| Component | Current Status | Proposed Action |
|-----------|----------------|-----------------|
| `BranchController` | ✅ Fully implemented | No changes needed |
| `ConcatAugmentationController` | ✅ Fully implemented | No changes needed |
| `merge.py` | ❌ Empty file | **Full implementation as core primitive** |
| `MetaModelController` | ✅ Feature-complete | **Refactor to use MergeController internally** |
| `TrainingSetReconstructor` | ✅ Fully implemented | Move to shared utility, expose for merge |

### New Capabilities in This Design

| Capability | Description |
|------------|-------------|
| **Merge as primitive** | Core controller for all branch combination operations |
| **OOF safety by default** | Prediction merging uses OOF reconstruction automatically |
| **Unsafe mode** | Explicit `unsafe=True` for advanced users (with warnings) |
| **Per-branch selection** | Choose which models per branch: `all`, `best`, `top_k`, explicit list |
| **Per-branch aggregation** | Combine predictions: `separate`, `mean`, `weighted_mean`, `proba_mean` |
| **Different strategies per branch** | Branch 0 can use `best`, Branch 1 can use `top_k` with `mean` |
| **Mixed merging** | Features from some branches, predictions from others |
| **MetaModel as wrapper** | MetaModel delegates to merge, gains all new capabilities |

### Architectural Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Pipeline                                │
├─────────────────────────────────────────────────────────────────────┤
│  {"branch": [[A], [B]]} → step1 → {"merge": ...} → model            │
└────────────────────────────────────┬────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │   Branch     │  │    Merge     │  │   Model      │
            │  Controller  │  │  Controller  │  │  Controller  │
            └──────────────┘  └──────┬───────┘  └──────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │  Collect     │  │  OOF Recon   │  │   Concat     │
            │  Features    │  │  (if preds)  │  │   Features   │
            └──────────────┘  └──────────────┘  └──────────────┘
                                     │
                    Uses: TrainingSetReconstructor
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
            ┌──────────────┐                  ┌──────────────┐
            │  MetaModel   │  ◄── delegates ──│   Explicit   │
            │  (wrapper)   │                  │    Merge     │
            └──────────────┘                  └──────────────┘
```

---

## 2. Verified Current State

### 2.1 BranchController ✅ Verified Correct

**Location**: [nirs4all/controllers/data/branch.py](nirs4all/controllers/data/branch.py)

**Verified Behaviors**:
- Creates N independent execution contexts (one per branch definition)
- Stores branch contexts in `context.custom["branch_contexts"]`
- Each context contains:
  - `branch_id`: Numeric identifier
  - `name`: Branch name
  - `context`: Independent ExecutionContext copy
  - `features_snapshot`: Deep copy of dataset features after branch steps
  - `generator_choice`: For generator syntax tracking
- Sets `context.custom["in_branch_mode"] = True`
- Supports nested branching via `_multiply_branch_contexts()`
- Supports prediction mode with target branch filtering

**Post-Branch Iteration** (verified in [executor.py](nirs4all/pipeline/execution/executor.py#L281-L348)):
- Executor checks for `branch_contexts` in context
- If present and step is NOT a branch step, executes step on each branch
- Restores features from `features_snapshot` before each branch execution
- Updates branch contexts after execution

**No changes needed** - the implementation is correct and complete.

### 2.2 ConcatAugmentationController ✅ Verified Correct

**Location**: [nirs4all/controllers/data/concat_transform.py](nirs4all/controllers/data/concat_transform.py)

**Verified Behaviors**:
- Applies multiple transformers to current features
- Concatenates results horizontally: `[transform1_output | transform2_output | ...]`
- Supports `None` to include original features
- Works within single execution path (no branch awareness)
- Two modes:
  - Replace mode (top-level): Replaces each processing with concatenated version
  - Add mode (inside feature_augmentation): Adds new processing

**No changes needed** - well-designed for its purpose.

### 2.3 MergeController ❌ Empty - Needs Implementation

**Location**: [nirs4all/controllers/data/merge.py](nirs4all/controllers/data/merge.py)

**Current State**: Empty file (0 bytes)

**Required**: Full implementation as specified in Section 4.

### 2.4 MetaModelController ✅ Feature-Complete, Needs Integration

**Location**: [nirs4all/controllers/models/meta_model.py](nirs4all/controllers/models/meta_model.py)

**Verified Capabilities**:
- Collects OOF predictions from source models via `TrainingSetReconstructor`
- Supports `source_models` parameter for model selection
- Supports `BranchScope.CURRENT_ONLY` and `BranchScope.ALL_BRANCHES`
- Comprehensive branch validation via `BranchValidator`
- Multi-level stacking validation via `MultiLevelValidator`
- Cross-branch stacking validation via `CrossBranchValidator`
- Coverage strategies: STRICT, DROP_INCOMPLETE, IMPUTE_ZERO, IMPUTE_MEAN
- Test aggregation: MEAN, WEIGHTED_MEAN, BEST_FOLD

**Current Limitations**:
- Cannot mix features and predictions (only collects predictions)
- Does not exit branch mode (branches continue after MetaModel)
- Complex parameter surface for what could be merge + train

**Proposed**: Refactor to optionally delegate to MergeController for complex cases.

### 2.5 TrainingSetReconstructor ✅ Complete

**Location**: [nirs4all/controllers/models/stacking/reconstructor.py](nirs4all/controllers/models/stacking/reconstructor.py)

**Verified Capabilities**:
- OOF reconstruction from prediction store
- Fold alignment validation
- Coverage handling per stacking config
- Classification support with probability features
- Branch-aware prediction filtering

This is the core engine that MergeController will use for prediction merging.

---

## 3. Problem Analysis

### 3.1 Gap 1: No Branch Exit Mechanism

**Current Behavior**:
```python
pipeline = [
    {"branch": [[SNV()], [MSC()]]},
    PLS(10),     # Runs on BOTH branches (2× execution)
    Ridge(),     # STILL runs on BOTH branches (2× execution)
    # ... forever
]
```

**Problem**: Once in branch mode, there's no way to "exit" and return to single-path execution. All subsequent steps run N times.

**Required**: A `merge` step that:
1. Collects outputs from branches
2. Combines them into a single dataset
3. Clears `branch_contexts` and sets `in_branch_mode=False`

### 3.2 Gap 2: Cannot Mix Features and Predictions

**Current Behavior**:
```python
{"branch": [[SNV(), PLS()], [PCA(10)]]},
{"model": MetaModel(model=Ridge(), branch_scope=BranchScope.ALL_BRANCHES)}
```

Branch 0: Has PLS predictions (1 feature)
Branch 1: Has PCA features (10 features), NO predictions

MetaModel only looks for predictions → Branch 1 contributes nothing.

**User Intent**: Train Ridge on `[PLS_predictions | PCA_features]` (11 dimensions)

**Required**: A merge syntax that can specify:
```python
{"merge": {"predictions": [0], "features": [1]}}
```

### 3.3 Gap 3: Asymmetric Branch Handling

**Scenario**: Multiple models in one branch, only features in another:
```python
{"branch": [[SNV(), PLS(), RF(), XGB()], [PCA(10)]]},
{"merge": {"predictions": {"branches": [0], "models": ["XGB"]}, "features": [1]}}
```

**Required**: Merge controller must handle:
- Selective model prediction extraction
- Feature extraction from branches without models
- Proper OOF reconstruction for selected predictions

### 3.4 Gap 4: Multi-Model Branch Aggregation

**Scenario**: A branch contains multiple models, and the user needs control over how their predictions are combined:

```python
# Branch 0 has 3 models: PLS, RF, XGB
# Branch 1 has 2 models: PLS, SVR
{"branch": [
    [SNV(), PLS(10), RF(), XGB()],
    [MSC(), PLS(5), SVR()]
]}
```

**Questions that arise**:
1. Should predictions from all models be kept separate (3 + 2 = 5 features)?
2. Should they be aggregated within each branch (2 features)?
3. Should only the best model per branch be kept (2 features)?
4. Can different strategies apply to different branches?

**Current State**: No per-branch control. MetaModel's `source_models` applies globally, not per-branch.

**Required**: Per-branch configuration for:
- **Selection strategy**: Which models to include (`all`, `top_k`, `best`, explicit list)
- **Aggregation strategy**: How to combine selected predictions (`separate`, `mean`, `weighted_mean`, `proba_mean`)

**Use Case Examples**:
```python
# Best model from branch 0, top 2 from branch 1
{"merge": {
    "predictions": [
        {"branch": 0, "select": "best", "metric": "rmse"},
        {"branch": 1, "select": {"top_k": 2, "metric": "r2"}, "aggregate": "mean"}
    ]
}}

# All models separate from branch 0, averaged from branch 1
{"merge": {
    "predictions": [
        {"branch": 0, "aggregate": "separate"},  # 3 features (PLS, RF, XGB)
        {"branch": 1, "aggregate": "mean"}        # 1 feature (mean of PLS, SVR)
    ]
}}
```

---

## 4. Design: MergeController

### 4.1 Core Principle: Merge as Foundation

`MergeController` is the **core primitive** for all branch combination operations. It is designed to:

1. **Always exit branch mode** — Merge unconditionally clears branch contexts and returns to single-path execution
2. **Handle both features and predictions** — Unified interface for all branch output types
3. **Enforce OOF safety by default** — Predictions are always reconstructed using out-of-fold strategy unless explicitly disabled
4. **Be composable** — Can be used alone or as the foundation for higher-level operators like `MetaModel`

### 4.2 Controller Specification

```python
@register_controller
class MergeController(OperatorController):
    """Controller for merging branch outputs and exiting branch mode.

    This controller is the CORE PRIMITIVE for branch combination. It:
    1. Collects features and/or predictions from specified branches
    2. Performs horizontal concatenation of features
    3. Performs OOF reconstruction for predictions (mandatory unless unsafe=True)
    4. Creates a unified "merged" processing in the dataset
    5. ALWAYS clears branch contexts and exits branch mode

    Keywords: "merge"
    Priority: 5 (same as BranchController)

    OOF Safety:
        When predictions are merged, OOF reconstruction is MANDATORY by default.
        This prevents data leakage when the merged output is used for training.
        Set `unsafe=True` to disable OOF (generates prominent warnings).

    Relationship to MetaModel:
        MetaModel internally uses MergeController for data preparation, then
        trains the meta-learner. Users can achieve the same result with:
            {"merge": "predictions"}, {"model": Ridge()}
        which is equivalent to:
            {"model": MetaModel(Ridge())}
    """

    priority = 5

    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        return keyword == "merge"

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return True
```

### 4.3 Syntax Specification

```python
# =============================================================================
# SIMPLE FORMS (Safe by default)
# =============================================================================

# Merge features from all branches (horizontal concat)
{"merge": "features"}

# Merge predictions from all branches (OOF reconstruction - SAFE)
{"merge": "predictions"}

# Merge both features and predictions from all branches
{"merge": "all"}

# =============================================================================
# SELECTIVE FORMS
# =============================================================================

# Merge features from specific branches
{"merge": {"features": [0, 2]}}  # Branch indices

# Merge predictions from specific branches (OOF reconstruction)
{"merge": {"predictions": [1, 3]}}

# Mix features from some branches, predictions from others
{"merge": {
    "features": [1],      # Features from branch 1
    "predictions": [0]    # OOF predictions from branch 0
}}

# =============================================================================
# FULL CONTROL
# =============================================================================

{"merge": {
    "features": {
        "branches": [1, 2],           # Which branches
    },
    "predictions": {
        "branches": [0],              # Which branches
        "models": ["RF", "XGB"],      # Filter by model name
        "proba": False,               # Use probabilities (classification)
    },
    "include_original": False,        # Include pre-branch features?
    "on_missing": "error",            # "error" | "warn" | "skip"
}}

# =============================================================================
# PER-BRANCH MODEL SELECTION & AGGREGATION
# =============================================================================

# When a branch has multiple models, control how they're selected and combined

# Keep all models' predictions separate (default)
{"merge": {
    "predictions": [
        {"branch": 0, "aggregate": "separate"},  # N features (one per model)
        {"branch": 1, "aggregate": "separate"}
    ]
}}

# Average predictions within each branch
{"merge": {
    "predictions": [
        {"branch": 0, "aggregate": "mean"},        # 1 feature (mean of all models)
        {"branch": 1, "aggregate": "weighted_mean", "weight_metric": "r2"}  # Weighted by R²
    ]
}}

# Select specific models or top-K per branch
{"merge": {
    "predictions": [
        {"branch": 0, "select": "best", "metric": "rmse"},           # 1 feature (best RMSE)
        {"branch": 1, "select": {"top_k": 2, "metric": "r2"}}        # 2 features (top 2 R²)
    ]
}}

# Combine selection and aggregation
{"merge": {
    "predictions": [
        {"branch": 0, "select": {"top_k": 3}, "aggregate": "mean"},   # Mean of top 3
        {"branch": 1, "select": ["PLS", "RF"], "aggregate": "separate"}  # Explicit models
    ]
}}

# Classification: average probabilities
{"merge": {
    "predictions": [
        {"branch": 0, "proba": True, "aggregate": "proba_mean"}  # Mean probabilities
    ]
}}

# =============================================================================
# UNSAFE MODE (Explicit opt-in for advanced users)
# =============================================================================

# ⚠️ WARNING: Disables OOF reconstruction - causes DATA LEAKAGE
# Only use for rapid prototyping or when you understand the implications
{"merge": {
    "predictions": "all",
    "unsafe": True    # DISABLES OOF - training predictions used directly
}}
```

### 4.4 OOF Safety Model

#### Default Behavior (Safe)

When `predictions` is specified without `unsafe=True`:

1. **Training phase**: Predictions are reconstructed using OOF strategy
   - For each sample, use the prediction from the fold where that sample was in validation
   - Ensures no sample sees its own training prediction
   - Uses `TrainingSetReconstructor` internally

2. **Prediction phase**: Predictions are aggregated across folds
   - Multiple fold models produce multiple predictions
   - Aggregation strategy (mean, weighted, best) is configurable

#### Unsafe Mode (Explicit Opt-In)

When `unsafe=True`:

```python
{"merge": {"predictions": "all", "unsafe": True}}
```

1. **Training phase**: Uses direct training predictions (DATA LEAKAGE)
   - Fast, no fold reconstruction needed
   - Results in overly optimistic metrics
   - Appropriate for: rapid prototyping, shape verification, very large datasets

2. **Warning behavior**:
   - Logger emits `WARNING` level message at merge time
   - Predictions metadata includes `"unsafe_merge": True`
   - Manifest records unsafe flag for reproducibility audit

```python
# Warning message example:
logger.warning(
    "⚠️ UNSAFE MERGE: OOF reconstruction disabled for predictions. "
    "Training predictions are used directly, causing DATA LEAKAGE. "
    "Do NOT use for final model evaluation. "
    "Set unsafe=False (default) for production pipelines."
)
```

3. **Production considerations**:
   - When `unsafe=True`, the merge step adds a metadata flag
   - Downstream analysis tools can filter/warn on unsafe predictions
   - Export/bundle operations can optionally reject unsafe pipelines

### 4.5 Behavior Specification

| Merge Type | Branch Has Only Transforms | Branch Has Transform + Model | Branch Has Only Model |
|------------|---------------------------|------------------------------|----------------------|
| `"features"` | ✅ Use `features_snapshot` | ✅ Use `features_snapshot` (ignores model) | ✅ Use pre-model features |
| `"predictions"` | ❌ Error (no model) | ✅ OOF predictions | ✅ OOF predictions |
| `"all"` | ✅ Features only | ✅ Features + Predictions | ✅ Features + Predictions |

### 4.6 Per-Branch Model Selection & Aggregation

When a branch contains **multiple models**, the user can control:

1. **Selection**: Which models' predictions to include
2. **Aggregation**: How to combine selected predictions into features

#### Selection Strategies

| Strategy | Syntax | Description |
|----------|--------|-------------|
| All (default) | `"select": "all"` | Include all models in the branch |
| Best | `"select": "best"` | Single best model by metric |
| Top-K | `"select": {"top_k": N}` | Top N models by metric |
| Explicit | `"select": ["ModelA", "ModelB"]` | Specific model names |

Optional `metric` parameter: `"rmse"`, `"mae"`, `"r2"`, `"accuracy"`, `"f1"` (default: task-appropriate)

#### Aggregation Strategies

| Strategy | Syntax | Output Dimension | Description |
|----------|--------|------------------|-------------|
| Separate (default) | `"aggregate": "separate"` | N (one per model) | Keep predictions as separate features |
| Mean | `"aggregate": "mean"` | 1 | Simple average |
| Weighted Mean | `"aggregate": "weighted_mean"` | 1 | Weight by validation score |
| Proba Mean | `"aggregate": "proba_mean"` | K (classes) | Average class probabilities |

#### Examples

**Scenario**: Branch 0 has 4 models (PLS, RF, XGB, SVR), Branch 1 has 2 models (PLS, RF)

```python
# All separate (default): 4 + 2 = 6 features
{"merge": "predictions"}

# Best per branch: 1 + 1 = 2 features
{"merge": {"predictions": [
    {"branch": 0, "select": "best", "metric": "rmse"},
    {"branch": 1, "select": "best", "metric": "rmse"}
]}}

# Top 2 from branch 0, average from branch 1: 2 + 1 = 3 features
{"merge": {"predictions": [
    {"branch": 0, "select": {"top_k": 2, "metric": "r2"}},
    {"branch": 1, "aggregate": "mean"}
]}}

# All from branch 0 averaged, explicit models from branch 1 separate: 1 + 2 = 3 features
{"merge": {"predictions": [
    {"branch": 0, "aggregate": "weighted_mean", "weight_metric": "r2"},
    {"branch": 1, "select": ["PLS", "RF"], "aggregate": "separate"}
]}}
```

#### Selection + Aggregation Interaction

| Select | Aggregate | Result |
|--------|-----------|--------|
| `"all"` | `"separate"` | N features (all models) |
| `"all"` | `"mean"` | 1 feature (mean of all) |
| `"best"` | (ignored) | 1 feature (best model) |
| `{"top_k": 3}` | `"separate"` | 3 features |
| `{"top_k": 3}` | `"mean"` | 1 feature (mean of top 3) |
| `["A", "B"]` | `"separate"` | 2 features |
| `["A", "B"]` | `"weighted_mean"` | 1 feature |

### 4.7 Output Processing

Merge creates a new processing called `"merged"`:
- Features are horizontal-concatenated: `np.hstack([branch0_features, branch1_features, ...])`
- Predictions are OOF-reconstructed via `TrainingSetReconstructor`
- Combined output: `np.hstack([merged_features, oof_predictions])`

The merged processing replaces all branch processings and becomes the active processing for subsequent steps.

### 4.5 Prediction Mode Support

In prediction/explain mode:
1. **Features**: Load and apply saved transformers from each branch, concatenate
2. **Predictions**: Load source models, get test predictions, aggregate per `TestAggregation` config
3. **Merge config**: Saved in manifest for reproducibility

---

## 5. Design: MetaModel as Convenience Wrapper

### 5.1 Architectural Change: MetaModel Uses Merge Internally

**Previous Architecture** (current implementation):
```
MetaModel → TrainingSetReconstructor → OOF predictions → Train
            ↑ internal, duplicated logic
```

**New Architecture** (proposed):
```
MetaModel → MergeController → Merged features → Train
            ↑ delegates to core primitive
```

MetaModel becomes a **thin convenience wrapper** that:
1. Translates its parameters to a merge configuration
2. Calls `MergeController` to prepare the training data
3. Trains the meta-learner on the merged output

### 5.2 User-Facing Equivalences

The following are now **semantically equivalent**:

```python
# Pattern A: MetaModel (convenience)
pipeline = [
    KFold(n_splits=5),
    PLS(10), RF(),
    {"model": MetaModel(Ridge(), source_models="all")}
]

# Pattern B: Explicit merge + model (compositional)
pipeline = [
    KFold(n_splits=5),
    PLS(10), RF(),
    {"merge": "predictions"},
    {"model": Ridge()}
]
```

Both produce the same result: Ridge trained on OOF predictions from PLS and RF.

**Benefits of Pattern B**:
- More explicit about what's happening
- Can add steps between merge and model training
- Can use any model, not just those wrapped in MetaModel

**Benefits of Pattern A**:
- Concise for common stacking use case
- Familiar API for sklearn StackingClassifier/Regressor users

### 5.3 MetaModel Can Also Stack Features

With the new architecture, MetaModel gains the ability to include features alongside predictions:

```python
# Stack predictions + original features
{"model": MetaModel(
    model=Ridge(),
    source_models="all",
    include_features=True  # NEW: include current features in meta-training
)}
```

This is equivalent to:
```python
{"merge": {"predictions": "all", "include_original": True}},
{"model": Ridge()}
```

### 5.4 Internal Refactoring

MetaModel's execute method now delegates to merge:

```python
class MetaModelController(OperatorController):
    def execute(self, step_info, dataset, context, ...):
        meta_model = step_info.original_step["model"]

        # Step 1: Build merge config from MetaModel params
        merge_config = self._build_merge_config(meta_model, context)

        # Step 2: Delegate to MergeController for data preparation
        merged_dataset, merge_output = MergeController.merge_branches(
            dataset=dataset,
            context=context,
            config=merge_config,
            prediction_store=prediction_store,
            mode=mode,
        )

        # Step 3: Train meta-learner on merged output
        X_train, y_train = self._get_training_data(merged_dataset, context)
        meta_model.model.fit(X_train, y_train)

        # ... rest of training/prediction logic
```

### 5.5 Parameter Mapping

| MetaModel Parameter | Translated Merge Config |
|---------------------|------------------------|
| `source_models="all"` | `{"predictions": "all"}` |
| `source_models=["RF", "XGB"]` | `{"predictions": {"models": ["RF", "XGB"]}}` |
| `branch_scope=ALL_BRANCHES` | `{"predictions": {"branches": "all"}}` |
| `branch_scope=CURRENT_ONLY` | `{"predictions": {"branches": "current"}}` |
| `use_proba=True` | `{"predictions": {"proba": True}}` |
| `include_features=True` (new) | `{"predictions": ..., "include_original": True}` |
| `stacking_config.coverage_strategy` | Passed through to merge |
| `stacking_config.test_aggregation` | Passed through to merge |

### 5.6 Backward Compatibility

All existing MetaModel pipelines continue to work unchanged:

```python
# These all work exactly as before:
{"model": MetaModel(Ridge())}
{"model": MetaModel(Ridge(), source_models=["PLS", "RF"])}
{"model": MetaModel(Ridge(), branch_scope=BranchScope.ALL_BRANCHES)}
```

The refactoring is purely internal—the API surface remains the same.

---

## 6. Design: ConcatTransformController (No Changes)

### 6.1 Current Behavior (Keep)

```python
{"concat_transform": [PCA(50), SVD(30), None]}
```

- Applies each transformer to current features
- Concatenates horizontally: `[PCA_output | SVD_output | original]`
- Works within single execution path
- No branch awareness required

### 6.2 Relationship to Merge

| Aspect | `concat_transform` | `merge: "features"` |
|--------|-------------------|---------------------|
| Scope | Single execution path | Exits branch mode |
| Input | List of transformers | Branch contexts |
| Execution | 1× (parallel transforms) | N× (branch execution) + merge |
| Use when | Same data, different transforms | Different preprocessing pipelines |

**Equivalence** (for feature-only merging):
```python
# These produce similar results:
{"concat_transform": [SNV(), MSC(), D1()]}

# vs
{"branch": [[SNV()], [MSC()], [D1()]]},
{"merge": "features"}
```

The first is more efficient; use `concat_transform` when you don't need intermediate branch steps.

---

## 7. Complete Use Case Matrix

### 7.1 Feature-Only Pipelines

| Use Case | Syntax |
|----------|--------|
| Single transform | `SNV()` |
| Transform chain | `SNV(), PCA(50)` |
| Concat transforms (single path) | `{"concat_transform": [SNV(), MSC()]}` |
| Multiple processings | `{"feature_augmentation": [SNV(), MSC()]}` |
| Branch + merge features | `{"branch": [[SNV()], [MSC()]]}, {"merge": "features"}` |

### 7.2 Model Pipelines

| Use Case | Syntax |
|----------|--------|
| Single model | `{"model": PLSRegression()}` |
| Sequential models | `PLS(), RF(), XGB()` |
| Branch preprocessing + single model | `{"branch": [[SNV()], [MSC()]]}, PLS()` → runs 2× |
| Branch preprocessing + merge + single model | `{"branch": [[SNV()], [MSC()]]}, {"merge": "features"}, PLS()` → runs 1× |

### 7.3 Stacking Pipelines (Safe - OOF)

| Use Case | Syntax |
|----------|--------|
| Stack all previous models (MetaModel) | `PLS(), RF(), {"model": MetaModel(Ridge())}` |
| Stack all previous models (explicit) | `PLS(), RF(), {"merge": "predictions"}, {"model": Ridge()}` |
| Stack selected models | `PLS(), RF(), XGB(), {"model": MetaModel(Ridge(), source_models=["RF", "XGB"])}` |
| Stack across branches | `{"branch": [[SNV(), PLS()], [MSC(), RF()]]}, {"merge": "predictions"}, Ridge()` |
| Mix features + predictions | `{"branch": [[SNV(), PLS()], [PCA(10)]]}, {"merge": {"features": [1], "predictions": [0]}}, Ridge()` |
| Stack + include original features | `{"merge": {"predictions": "all", "include_original": True}}, Ridge()` |

### 7.4 Unsafe Mode (Rapid Prototyping Only)

| Use Case | Syntax | Warning |
|----------|--------|---------|
| Quick shape check | `{"merge": {"predictions": "all", "unsafe": True}}` | ⚠️ DATA LEAKAGE |
| Fast iteration | `{"merge": {"predictions": [0], "unsafe": True}}, Ridge()` | ⚠️ DATA LEAKAGE |

**⚠️ Important**: Unsafe mode disables OOF reconstruction. Training predictions are used directly, which causes data leakage and overly optimistic metrics. **Do NOT use for final model evaluation.**

### 7.5 Per-Branch Model Selection & Aggregation

| Use Case | Syntax | Output Features |
|----------|--------|-----------------|
| All models separate (default) | `{"merge": "predictions"}` | N (one per model) |
| Best per branch | `{"merge": {"predictions": [{"branch": 0, "select": "best"}, {"branch": 1, "select": "best"}]}}` | 2 |
| Top-K per branch | `{"merge": {"predictions": [{"branch": 0, "select": {"top_k": 2}}]}}` | 2 |
| Average within branch | `{"merge": {"predictions": [{"branch": 0, "aggregate": "mean"}]}}` | 1 |
| Weighted average | `{"merge": {"predictions": [{"branch": 0, "aggregate": "weighted_mean", "weight_metric": "r2"}]}}` | 1 |
| Different strategies per branch | `{"merge": {"predictions": [{"branch": 0, "select": "best"}, {"branch": 1, "aggregate": "mean"}]}}` | 2 |
| Top-K averaged | `{"merge": {"predictions": [{"branch": 0, "select": {"top_k": 3}, "aggregate": "mean"}]}}` | 1 |
| Explicit models averaged | `{"merge": {"predictions": [{"branch": 0, "select": ["RF", "XGB"], "aggregate": "mean"}]}}` | 1 |
| Proba mean (classification) | `{"merge": {"predictions": [{"branch": 0, "proba": True, "aggregate": "proba_mean"}]}}` | K (classes) |

### 7.6 Complex Pipelines

| Use Case | Syntax |
|----------|--------|
| Nested branches + merge | `{"branch": [[{"branch": [...]}, PLS()], [...]]}, {"merge": "predictions"}` |
| Selective model from multi-model branch | `{"branch": [[SNV(), PLS(), RF(), XGB()]]}, {"merge": {"predictions": {"models": ["XGB"]}}}` |
| MetaModel with feature inclusion | `PLS(), RF(), {"model": MetaModel(Ridge(), include_features=True)}` |

### 7.6 Equivalence Table

These pairs are semantically equivalent (produce same results):

| MetaModel Syntax | Explicit Merge Syntax |
|------------------|----------------------|
| `{"model": MetaModel(Ridge())}` | `{"merge": "predictions"}, {"model": Ridge()}` |
| `{"model": MetaModel(Ridge(), source_models=["RF"])}` | `{"merge": {"predictions": {"models": ["RF"]}}}, {"model": Ridge()}` |
| `{"model": MetaModel(Ridge(), branch_scope=ALL_BRANCHES)}` | `{"merge": {"predictions": {"branches": "all"}}}, {"model": Ridge()}` |
| `{"model": MetaModel(Ridge(), include_features=True)}` | `{"merge": {"predictions": "all", "include_original": True}}, {"model": Ridge()}` |

---

## 8. Implementation Specification

### 8.1 MergeConfig Dataclass

**File**: `nirs4all/operators/data/merge.py`

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class MergeMode(Enum):
    """What to merge from branches."""
    FEATURES = "features"
    PREDICTIONS = "predictions"
    ALL = "all"


class SelectionStrategy(Enum):
    """How to select models within a branch."""
    ALL = "all"           # Include all models
    BEST = "best"         # Single best by metric
    TOP_K = "top_k"       # Top K by metric
    EXPLICIT = "explicit" # Explicit model list


class AggregationStrategy(Enum):
    """How to aggregate predictions within a branch."""
    SEPARATE = "separate"         # Keep as separate features (default)
    MEAN = "mean"                 # Simple average
    WEIGHTED_MEAN = "weighted_mean"  # Weighted by validation score
    PROBA_MEAN = "proba_mean"     # Average class probabilities


@dataclass
class BranchPredictionConfig:
    """Configuration for prediction collection from a single branch.

    Attributes:
        branch: Branch index to collect from.
        select: Model selection strategy.
            - "all" (default): All models in branch
            - "best": Single best model by metric
            - {"top_k": N}: Top N models by metric
            - ["ModelA", "ModelB"]: Explicit model names
        metric: Metric for selection (rmse, mae, r2, accuracy, f1).
            Default is task-appropriate.
        aggregate: How to combine predictions.
            - "separate" (default): Each model is a separate feature
            - "mean": Average predictions
            - "weighted_mean": Weight by validation score
            - "proba_mean": Average class probabilities
        weight_metric: Metric for weighted aggregation (default: same as `metric`).
        proba: Use class probabilities instead of predictions (classification).
    """
    branch: int
    select: Union[str, Dict, List[str]] = "all"
    metric: Optional[str] = None  # rmse, mae, r2, accuracy, f1
    aggregate: str = "separate"   # separate, mean, weighted_mean, proba_mean
    weight_metric: Optional[str] = None
    proba: bool = False

    def __post_init__(self):
        # Validate aggregate
        valid_aggregates = ("separate", "mean", "weighted_mean", "proba_mean")
        if self.aggregate not in valid_aggregates:
            raise ValueError(f"aggregate must be one of {valid_aggregates}, got {self.aggregate}")

        # Validate select format
        if isinstance(self.select, dict):
            if "top_k" not in self.select:
                raise ValueError("dict select must contain 'top_k' key")
        elif isinstance(self.select, str):
            if self.select not in ("all", "best"):
                raise ValueError(f"string select must be 'all' or 'best', got {self.select}")


@dataclass
class MergeConfig:
    """Configuration for branch merging.

    Attributes:
        collect_features: Whether to collect features from branches.
        feature_branches: Which branches to collect features from.
            "all" or list of branch indices.
        collect_predictions: Whether to collect predictions from branches.
        prediction_branches: Legacy: which branches (simple mode).
            For advanced control, use `prediction_configs`.
        prediction_configs: List of BranchPredictionConfig for per-branch control.
            Takes precedence over prediction_branches.
        model_filter: Legacy: global model filter (simple mode).
        use_proba: Legacy: global proba setting (simple mode).
        include_original: Include pre-branch features in merged output.
        on_missing: How to handle missing branches/predictions.
            "error" (default), "warn", or "skip".
        unsafe: If True, DISABLE OOF reconstruction for predictions.
            ⚠️ CAUSES DATA LEAKAGE - only for rapid prototyping.
    """
    collect_features: bool = False
    feature_branches: Union[str, List[int]] = "all"
    collect_predictions: bool = False
    prediction_branches: Union[str, List[int]] = "all"  # Legacy simple mode
    prediction_configs: Optional[List[BranchPredictionConfig]] = None  # Advanced per-branch
    model_filter: Optional[List[str]] = None  # Legacy
    use_proba: bool = False  # Legacy
    include_original: bool = False
    on_missing: str = "error"
    unsafe: bool = False

    def __post_init__(self):
        # Validate on_missing
        if self.on_missing not in ("error", "warn", "skip"):
            raise ValueError(f"on_missing must be 'error', 'warn', or 'skip', got {self.on_missing}")

        # Validate unsafe usage
        if self.unsafe and self.collect_predictions:
            import warnings
            warnings.warn(
                "⚠️ MergeConfig: unsafe=True disables OOF reconstruction. "
                "Training predictions will be used directly, causing DATA LEAKAGE. "
                "Do NOT use for final model evaluation.",
                UserWarning,
                stacklevel=2
            )

    def has_per_branch_config(self) -> bool:
        """Check if using advanced per-branch prediction config."""
        return self.prediction_configs is not None and len(self.prediction_configs) > 0
```

### 8.2 MergeController Implementation

**File**: `nirs4all/controllers/data/merge.py`

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.controllers.models.stacking import TrainingSetReconstructor
from nirs4all.operators.data.merge import MergeConfig
from nirs4all.pipeline.execution.result import StepOutput
from nirs4all.core.logging import get_logger

logger = get_logger(__name__)


@register_controller
class MergeController(OperatorController):
    """Controller for merging branch outputs - CORE PRIMITIVE.

    This is the foundational controller for all branch combination operations.
    MetaModel and other stacking operators delegate to this controller.
    """

    priority = 5

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
    ) -> Tuple:
        # Parse configuration
        config = self._parse_config(step_info.original_step["merge"])
        branch_contexts = context.custom.get("branch_contexts", [])

        if not branch_contexts:
            raise ValueError(
                "merge requires active branch contexts. "
                "Use merge only after a branch step."
            )

        # Validate branch indices
        self._validate_branches(config, branch_contexts)

        merged_parts = []

        # Collect features from specified branches
        if config.collect_features:
            features = self._collect_features(
                dataset, branch_contexts, config.feature_branches, config.on_missing
            )
            merged_parts.extend(features)

        # Collect predictions from specified branches
        if config.collect_predictions:
            predictions = self._collect_predictions(
                dataset, context, branch_contexts,
                config, prediction_store, mode
            )
            merged_parts.append(predictions)

        # Include original pre-branch features if requested
        if config.include_original:
            original = self._get_original_features(dataset, context)
            merged_parts.insert(0, original)

        if not merged_parts:
            raise ValueError("merge resulted in empty output - check configuration")

        # Horizontal concat all parts
        final_features = np.hstack(merged_parts)

        # Add merged features to dataset
        dataset.add_merged_features(final_features, processing_name="merged")

        # ALWAYS exit branch mode
        context.custom["branch_contexts"] = []
        context.custom["in_branch_mode"] = False

        # Update context to use merged processing
        context = context.with_processing([["merged"]])

        # Build metadata
        metadata = {
            "merged_shape": final_features.shape,
            "feature_branches": config.feature_branches if config.collect_features else [],
            "prediction_branches": config.prediction_branches if config.collect_predictions else [],
            "include_original": config.include_original,
        }

        # Add unsafe warning to metadata if applicable
        if config.unsafe:
            metadata["unsafe_merge"] = True
            logger.warning(
                "⚠️ UNSAFE MERGE: OOF reconstruction disabled for predictions. "
                "Training predictions are used directly, causing DATA LEAKAGE. "
                "Do NOT use for final model evaluation. "
                "Set unsafe=False (default) for production pipelines."
            )

        logger.success(
            f"Merged {len(config.feature_branches) if config.collect_features else 0} feature branches, "
            f"{len(config.prediction_branches) if config.collect_predictions else 0} prediction branches → "
            f"shape {final_features.shape}"
            f"{' [UNSAFE]' if config.unsafe else ''}"
        )

        return context, StepOutput(metadata=metadata)

    def _parse_config(self, raw_config) -> MergeConfig:
        """Parse merge configuration from various formats."""
        if isinstance(raw_config, str):
            if raw_config == "features":
                return MergeConfig(collect_features=True)
            elif raw_config == "predictions":
                return MergeConfig(collect_predictions=True)
            elif raw_config == "all":
                return MergeConfig(collect_features=True, collect_predictions=True)
            else:
                raise ValueError(f"Unknown merge mode: {raw_config}")

        elif isinstance(raw_config, dict):
            config = MergeConfig()

            if "features" in raw_config:
                config.collect_features = True
                feat_spec = raw_config["features"]
                if isinstance(feat_spec, list):
                    config.feature_branches = feat_spec
                elif isinstance(feat_spec, dict):
                    config.feature_branches = feat_spec.get("branches", "all")
                # else keep default "all"

            if "predictions" in raw_config:
                config.collect_predictions = True
                pred_spec = raw_config["predictions"]
                if isinstance(pred_spec, list):
                    config.prediction_branches = pred_spec
                elif isinstance(pred_spec, dict):
                    config.prediction_branches = pred_spec.get("branches", "all")
                    config.model_filter = pred_spec.get("models")
                    config.use_proba = pred_spec.get("proba", False)
                elif pred_spec == "all":
                    pass  # keep default "all"

            config.include_original = raw_config.get("include_original", False)
            config.on_missing = raw_config.get("on_missing", "error")
            config.unsafe = raw_config.get("unsafe", False)

            return config

        raise ValueError(f"Invalid merge config type: {type(raw_config)}")

    def _collect_predictions(
        self,
        dataset,
        context,
        branch_contexts,
        config: MergeConfig,
        prediction_store,
        mode
    ) -> np.ndarray:
        """Collect predictions from specified branches.

        Uses OOF reconstruction by default (safe).
        If config.unsafe=True, uses direct predictions (DATA LEAKAGE).
        """
        if prediction_store is None:
            raise ValueError(
                "prediction_store required for prediction merge. "
                "Ensure models were trained in the specified branches."
            )

        # Resolve branch indices
        prediction_branches = config.prediction_branches
        if prediction_branches == "all":
            prediction_branches = list(range(len(branch_contexts)))

        # Get source model names from prediction store
        source_models = []
        for idx in prediction_branches:
            branch_info = branch_contexts[idx]
            branch_id = branch_info["branch_id"]

            # Find models that ran in this branch
            branch_preds = prediction_store.filter_predictions(
                branch_id=branch_id, load_arrays=False
            )
            model_names = set(p.get("model_name") for p in branch_preds)

            # Apply model filter if specified
            if config.model_filter:
                model_names = model_names.intersection(config.model_filter)

            source_models.extend(model_names)

        if not source_models:
            raise ValueError(
                f"No model predictions found in branches {prediction_branches}. "
                f"Model filter: {config.model_filter}"
            )

        if config.unsafe:
            # UNSAFE: Use direct training predictions (DATA LEAKAGE)
            return self._collect_predictions_unsafe(
                dataset, prediction_store, source_models, mode
            )
        else:
            # SAFE: Use OOF reconstruction
            return self._collect_predictions_oof(
                dataset, context, prediction_store, source_models, mode, config
            )

    def _collect_predictions_oof(
        self, dataset, context, prediction_store, source_models, mode, config
    ) -> np.ndarray:
        """Safe OOF reconstruction for predictions."""
        from nirs4all.operators.models.meta import StackingConfig
        from nirs4all.controllers.models.stacking.config import ReconstructorConfig

        reconstructor = TrainingSetReconstructor(
            prediction_store=prediction_store,
            source_model_names=list(source_models),
            stacking_config=StackingConfig(),
            reconstructor_config=ReconstructorConfig(log_warnings=True),
        )

        result = reconstructor.reconstruct(
            dataset=dataset,
            context=context,
        )

        return result.X_train_meta if mode == "train" else result.X_test_meta

    def _collect_predictions_unsafe(
        self, dataset, prediction_store, source_models, mode
    ) -> np.ndarray:
        """UNSAFE: Direct prediction collection without OOF.

        ⚠️ This causes DATA LEAKAGE when used for training.
        Only use for rapid prototyping or when you understand the implications.
        """
        # Collect train predictions directly (leaks information)
        partition = "train" if mode == "train" else "test"

        all_preds = []
        for model_name in source_models:
            preds = prediction_store.filter_predictions(
                model_name=model_name,
                partition=partition,
                load_arrays=True
            )
            if preds:
                # Aggregate across folds if multiple
                pred_arrays = [p["y_pred"] for p in preds if "y_pred" in p]
                if pred_arrays:
                    # Mean across folds
                    mean_pred = np.mean(pred_arrays, axis=0)
                    all_preds.append(mean_pred.reshape(-1, 1))

        if not all_preds:
            raise ValueError(f"No predictions found for partition '{partition}'")

        return np.hstack(all_preds)

    # ... rest of helper methods (_validate_branches, _collect_features, etc.)

    @classmethod
    def merge_branches(
        cls,
        dataset,
        context,
        config: MergeConfig,
        prediction_store=None,
        mode="train"
    ):
        """Static method for programmatic merge (used by MetaModel).

        This allows MetaModelController to delegate to merge logic
        without going through the full step execution machinery.
        """
        controller = cls()
        # Create minimal step_info
        step_info = type('StepInfo', (), {'original_step': {'merge': config}})()

        return controller.execute(
            step_info=step_info,
            dataset=dataset,
            context=context,
            runtime_context=None,
            mode=mode,
            prediction_store=prediction_store
        )
```

### 8.3 SpectroDataset Extension

Add method to `SpectroDataset`:

```python
def add_merged_features(
    self,
    features: np.ndarray,
    processing_name: str = "merged"
) -> None:
    """Add merged features as a new processing.

    Args:
        features: 2D array (n_samples, n_features)
        processing_name: Name for the merged processing
    """
    # Reshape to 3D: (n_samples, 1, n_features)
    features_3d = features.reshape(features.shape[0], 1, -1)

    # Create new FeatureSource or update existing
    self._features.add_processing(
        features=features_3d,
        processing_name=processing_name,
        source=0  # Add to first source
    )
```

### 8.4 MetaModelController Refactoring

**File**: `nirs4all/controllers/models/meta_model.py`

Key changes to delegate to MergeController:

```python
from nirs4all.controllers.data.merge import MergeController
from nirs4all.operators.data.merge import MergeConfig

class MetaModelController(OperatorController):

    def _build_merge_config(self, meta_model, context) -> MergeConfig:
        """Translate MetaModel parameters to MergeConfig."""
        config = MergeConfig(collect_predictions=True)

        # Source model selection
        if meta_model.source_models == "all":
            config.prediction_branches = "all"
        elif isinstance(meta_model.source_models, list):
            config.model_filter = meta_model.source_models

        # Branch scope
        if hasattr(meta_model, 'branch_scope'):
            from nirs4all.operators.models.meta import BranchScope
            if meta_model.branch_scope == BranchScope.ALL_BRANCHES:
                config.prediction_branches = "all"
            elif meta_model.branch_scope == BranchScope.CURRENT_ONLY:
                config.prediction_branches = "current"

        # Classification probabilities
        if hasattr(meta_model, 'use_proba'):
            config.use_proba = meta_model.use_proba

        # Include features (new capability)
        if getattr(meta_model, 'include_features', False):
            config.collect_features = True
            config.include_original = True

        # Coverage strategy from stacking config
        # (passed through to TrainingSetReconstructor)

        return config

    def execute(self, step_info, dataset, context, ...):
        meta_model = step_info.original_step["model"]

        # Check if already in merged state (explicit merge before MetaModel)
        if context.custom.get("has_merged_features"):
            # Use existing merged features
            X_train = dataset.get_merged_features()
        else:
            # Build merge config and delegate
            merge_config = self._build_merge_config(meta_model, context)

            # Use MergeController's logic
            context, merge_output = MergeController.merge_branches(
                dataset=dataset,
                context=context,
                config=merge_config,
                prediction_store=prediction_store,
                mode=mode,
            )
            X_train = dataset.get_merged_features()

        # Train meta-learner
        y_train = dataset.y
        meta_model.model.fit(X_train, y_train)

        # ... rest of execution
```

### 8.5 File Changes Summary

| File | Action | Priority | Phase |
|------|--------|----------|-------|
| `nirs4all/operators/data/merge.py` | **Create** MergeMode, SelectionStrategy, AggregationStrategy enums | P0 | 1 |
| `nirs4all/operators/data/merge.py` | **Create** BranchPredictionConfig dataclass | P0 | 1 |
| `nirs4all/operators/data/merge.py` | **Create** MergeConfig dataclass | P0 | 1 |
| `nirs4all/controllers/data/merge.py` | **Implement** MergeController skeleton | P0 | 2 |
| `nirs4all/controllers/data/merge.py` | **Implement** config parsing | P0 | 1 |
| `nirs4all/controllers/data/merge.py` | **Implement** feature collection | P0 | 3 |
| `nirs4all/controllers/data/merge.py` | **Implement** OOF prediction collection | P0 | 4 |
| `nirs4all/controllers/data/merge.py` | **Implement** per-branch selection/aggregation | P0 | 5 |
| `nirs4all/data/dataset.py` | **Add** `add_merged_features()`, `get_merged_features()` | P0 | 2 |
| `nirs4all/controllers/models/meta_model.py` | **Refactor** to use MergeController | P1 | 7 |
| `tests/unit/operators/data/test_merge_config.py` | **Create** config parsing tests | P0 | 1 |
| `tests/unit/controllers/data/test_merge_controller.py` | **Create** controller unit tests | P0 | 2-5 |
| `tests/integration/pipeline/test_merge_features.py` | **Create** feature merge tests | P0 | 3 |
| `tests/integration/pipeline/test_merge_predictions.py` | **Create** prediction merge tests | P0 | 4 |
| `tests/integration/pipeline/test_merge_per_branch.py` | **Create** per-branch strategy tests | P0 | 5 |
| `tests/integration/pipeline/test_merge_mixed.py` | **Create** mixed merge tests | P0 | 6 |
| `tests/integration/pipeline/test_meta_model_backward_compat.py` | **Create** backward compat tests | P0 | 7 |
| `tests/integration/pipeline/test_merge_prediction_mode.py` | **Create** prediction mode tests | P1 | 8 |
| `examples/Q_merge_branches.py` | **Create** comprehensive examples | P2 | 9 |
| `docs/reference/branching.md` | **Update** with merge documentation | P2 | 9 |
| `docs/user_guide/stacking.md` | **Update** merge relationship | P2 | 9 |
| `docs/specifications/pipeline_syntax.md` | **Update** with merge syntax | P2 | 9 |

---

## 9. Implementation Roadmap

### Guiding Principle

**Merge is the core primitive**. Implement it first with full OOF support, then refactor MetaModel to use it. This ensures:
- Single source of truth for OOF logic
- MetaModel gets improvements automatically
- Users can use either approach (merge+model or MetaModel)

---

### Phase 1: Data Structures & Config Parsing (3-4 days)

**Goal**: Establish all data structures and config parsing logic

#### Task 1.1: Create Enum Types (0.5 day)
- **File**: `nirs4all/operators/data/merge.py`
- **Actions**:
  - [ ] Create `MergeMode` enum: `FEATURES`, `PREDICTIONS`, `ALL`
  - [ ] Create `SelectionStrategy` enum: `ALL`, `BEST`, `TOP_K`, `EXPLICIT`
  - [ ] Create `AggregationStrategy` enum: `SEPARATE`, `MEAN`, `WEIGHTED_MEAN`, `PROBA_MEAN`
- **Tests**: Unit tests for enum values and string conversion

#### Task 1.2: Create BranchPredictionConfig (0.5 day)
- **File**: `nirs4all/operators/data/merge.py`
- **Actions**:
  - [ ] Implement `BranchPredictionConfig` dataclass
  - [ ] Add `branch`, `select`, `metric`, `aggregate`, `weight_metric`, `proba` fields
  - [ ] Implement `__post_init__` validation for `aggregate` and `select`
  - [ ] Add helper method `get_selection_strategy() -> SelectionStrategy`
- **Tests**:
  - Valid configurations (all combinations)
  - Invalid aggregate value → ValueError
  - Invalid select dict (missing top_k) → ValueError

#### Task 1.3: Create MergeConfig (1 day)
- **File**: `nirs4all/operators/data/merge.py`
- **Actions**:
  - [ ] Implement `MergeConfig` dataclass with all fields
  - [ ] Implement `__post_init__` validation for `on_missing`, `unsafe`
  - [ ] Add `has_per_branch_config()` helper method
  - [ ] Add `get_prediction_configs()` method that normalizes legacy → per-branch
- **Tests**:
  - Simple mode with `prediction_branches`
  - Advanced mode with `prediction_configs`
  - `unsafe=True` warning emission
  - Invalid `on_missing` value → ValueError

#### Task 1.4: Implement Config Parser (1-1.5 days)
- **File**: `nirs4all/controllers/data/merge.py`
- **Actions**:
  - [ ] Implement `_parse_config(raw_config) -> MergeConfig`
  - [ ] Handle simple string forms: `"features"`, `"predictions"`, `"all"`
  - [ ] Handle dict with `features` and `predictions` keys
  - [ ] Handle legacy prediction format: `{"predictions": [0, 1]}`
  - [ ] Handle per-branch format: `{"predictions": [{"branch": 0, ...}]}`
  - [ ] Handle mixed features + predictions
  - [ ] Create `_parse_branch_prediction_config(item) -> BranchPredictionConfig`
- **Tests**:
  - All simple forms parse correctly
  - Legacy format backward compatible
  - Per-branch format with all options
  - Mixed formats
  - Invalid formats → clear error messages

#### Deliverable Phase 1
- All data structures implemented and tested
- Config parsing handles all syntax variants
- Clear error messages for invalid configurations

---

### Phase 2: MergeController Skeleton & Branch Exit (2-3 days)

**Goal**: Basic controller that exits branch mode

#### Task 2.1: Controller Registration (0.5 day)
- **File**: `nirs4all/controllers/data/merge.py`
- **Actions**:
  - [ ] Create `MergeController` class with `@register_controller`
  - [ ] Set `priority = 5`
  - [ ] Implement `matches()` for `keyword == "merge"`
  - [ ] Implement `supports_prediction_mode() -> True`
- **Tests**: Controller registered and matches correctly

#### Task 2.2: Branch Validation (0.5 day)
- **File**: `nirs4all/controllers/data/merge.py`
- **Actions**:
  - [ ] Implement `_validate_branches(config, branch_contexts)`
  - [ ] Resolve `"all"` to actual branch indices
  - [ ] Validate feature branch indices exist
  - [ ] Validate prediction branch indices exist
  - [ ] Clear error message for invalid indices
- **Tests**:
  - Valid indices pass
  - Invalid index → ValueError with available indices
  - `"all"` resolves correctly

#### Task 2.3: Branch Mode Exit Logic (0.5 day)
- **File**: `nirs4all/controllers/data/merge.py`
- **Actions**:
  - [ ] Implement branch context clearing: `context.custom["branch_contexts"] = []`
  - [ ] Set `context.custom["in_branch_mode"] = False`
  - [ ] Create new context with merged processing
  - [ ] Return proper `StepOutput` with metadata
- **Tests**:
  - After merge, branch mode is False
  - Branch contexts are empty
  - Context has merged processing

#### Task 2.4: SpectroDataset Extension (0.5 day)
- **File**: `nirs4all/data/dataset.py`
- **Actions**:
  - [ ] Add `add_merged_features(features, processing_name="merged")`
  - [ ] Reshape 2D → 3D: `(n_samples, 1, n_features)`
  - [ ] Add to first source via `_features.add_processing()`
  - [ ] Add `get_merged_features() -> np.ndarray` helper
- **Tests**:
  - Features added correctly
  - Retrieval works
  - Shape validation

#### Task 2.5: Execute Method Skeleton (1 day)
- **File**: `nirs4all/controllers/data/merge.py`
- **Actions**:
  - [ ] Implement `execute()` main flow
  - [ ] Call `_parse_config()`
  - [ ] Validate branch contexts exist (error if not)
  - [ ] Call `_validate_branches()`
  - [ ] Placeholder for feature/prediction collection
  - [ ] Add merged features to dataset
  - [ ] Exit branch mode
  - [ ] Build and return metadata
- **Tests**:
  - Integration test: empty merge exits branch mode
  - Error when no branch contexts

#### Deliverable Phase 2
- Controller skeleton complete
- Branch mode exit works
- SpectroDataset can store merged features

---

### Phase 3: Feature Merging (2-3 days)

**Goal**: Collect and concatenate features from branches

#### Task 3.1: Feature Snapshot Extraction (1 day)
- **File**: `nirs4all/controllers/data/merge.py`
- **Actions**:
  - [ ] Implement `_extract_features_from_snapshot(snapshot) -> np.ndarray`
  - [ ] Handle `FeatureSource` objects in snapshot
  - [ ] Extract 3D `_storage` array
  - [ ] Flatten to 2D: `(n_samples, processings * features)`
  - [ ] Concatenate multiple sources horizontally
- **Tests**:
  - Single source extraction
  - Multiple sources concatenation
  - Empty snapshot handling

#### Task 3.2: Feature Collection (1 day)
- **File**: `nirs4all/controllers/data/merge.py`
- **Actions**:
  - [ ] Implement `_collect_features(dataset, branch_contexts, branch_indices, on_missing)`
  - [ ] Iterate specified branch indices
  - [ ] Get `features_snapshot` from each branch context
  - [ ] Handle missing snapshot based on `on_missing`:
    - `"error"`: raise ValueError
    - `"warn"`: log warning, skip
    - `"skip"`: silent skip
  - [ ] Return list of 2D arrays
- **Tests**:
  - Collect from single branch
  - Collect from multiple branches
  - Missing branch with `error` → ValueError
  - Missing branch with `warn` → warning logged
  - Missing branch with `skip` → silent

#### Task 3.3: Include Original Features (0.5 day)
- **File**: `nirs4all/controllers/data/merge.py`
- **Actions**:
  - [ ] Implement `_get_original_features(dataset, context) -> np.ndarray`
  - [ ] Get pre-branch features from dataset
  - [ ] Insert at beginning of merged parts when `include_original=True`
- **Tests**:
  - Original features prepended correctly
  - Shape validation

#### Task 3.4: Integration Testing (0.5 day)
- **File**: `tests/integration/pipeline/test_merge_features.py`
- **Actions**:
  - [ ] Test `{"merge": "features"}` with 2 branches
  - [ ] Test `{"merge": {"features": [0]}}` (single branch)
  - [ ] Test feature merge → model training
  - [ ] Verify shapes throughout pipeline
- **Tests**: Full pipeline execution with feature merging

#### Deliverable Phase 3
- `{"merge": "features"}` fully working
- All `on_missing` strategies implemented
- Integration tests passing

---

### Phase 4: Prediction Merging - Simple Mode (3-4 days)

**Goal**: Collect OOF predictions from branches (legacy simple syntax)

#### Task 4.1: Model Discovery (1 day)
- **File**: `nirs4all/controllers/data/merge.py`
- **Actions**:
  - [ ] Implement `_discover_branch_models(branch_contexts, prediction_store, branch_idx)`
  - [ ] Query prediction store for branch's predictions
  - [ ] Extract unique model names
  - [ ] Apply `model_filter` if specified
  - [ ] Return list of model names with their metrics
- **Tests**:
  - Discover all models in branch
  - Model filter applied correctly
  - No models found → clear error

#### Task 4.2: OOF Reconstruction Integration (1.5 days)
- **File**: `nirs4all/controllers/data/merge.py`
- **Actions**:
  - [ ] Implement `_collect_predictions_oof(dataset, context, prediction_store, source_models, mode, config)`
  - [ ] Create `TrainingSetReconstructor` instance
  - [ ] Pass through coverage strategy from config
  - [ ] Call `reconstruct()` and extract features
  - [ ] Return appropriate array for train/test mode
- **Tests**:
  - OOF reconstruction produces correct shape
  - Coverage strategies work
  - Test mode uses aggregated predictions

#### Task 4.3: Unsafe Mode Implementation (0.5 day)
- **File**: `nirs4all/controllers/data/merge.py`
- **Actions**:
  - [ ] Implement `_collect_predictions_unsafe(dataset, prediction_store, source_models, mode)`
  - [ ] Collect training predictions directly (NO OOF)
  - [ ] Aggregate across folds with mean
  - [ ] Emit prominent warning
  - [ ] Tag metadata with `unsafe_merge=True`
- **Tests**:
  - Unsafe mode works
  - Warning emitted
  - Metadata tagged

#### Task 4.4: Simple Prediction Merge (1 day)
- **File**: `nirs4all/controllers/data/merge.py`
- **Actions**:
  - [ ] Implement `_collect_predictions(dataset, context, branch_contexts, config, prediction_store, mode)`
  - [ ] Handle simple mode: all branches, all models, separate
  - [ ] Iterate branches → discover models → collect OOF
  - [ ] Concatenate all predictions horizontally
  - [ ] Route to OOF or unsafe based on config
- **Tests**:
  - Simple `{"merge": "predictions"}` works
  - Multiple branches combined
  - Multiple models per branch → separate features

#### Deliverable Phase 4
- `{"merge": "predictions"}` working with OOF
- Unsafe mode with warnings
- Simple syntax fully supported

---

### Phase 5: Prediction Merging - Per-Branch Control (3-4 days)

**Goal**: Per-branch selection and aggregation strategies

#### Task 5.1: Model Ranking & Selection (1.5 days)
- **File**: `nirs4all/controllers/data/merge.py`
- **Actions**:
  - [ ] Implement `_rank_models(model_names, prediction_store, branch_id, metric)`
  - [ ] Query validation predictions for each model
  - [ ] Compute metric (rmse, mae, r2, accuracy, f1)
  - [ ] Return sorted list of (model_name, score)
  - [ ] Implement `_select_models(branch_config, ranked_models) -> List[str]`
  - [ ] Handle `"all"`: return all
  - [ ] Handle `"best"`: return top 1
  - [ ] Handle `{"top_k": N}`: return top N
  - [ ] Handle explicit list: return filtered
- **Tests**:
  - Ranking by each metric
  - All selection strategies
  - Top-K with K > available → all available
  - Explicit model not found → error

#### Task 5.2: Prediction Aggregation (1.5 days)
- **File**: `nirs4all/controllers/data/merge.py`
- **Actions**:
  - [ ] Implement `_aggregate_predictions(predictions_list, strategy, weights=None)`
  - [ ] `"separate"`: stack horizontally
  - [ ] `"mean"`: np.mean(axis=0)
  - [ ] `"weighted_mean"`: weighted average with normalized weights
  - [ ] `"proba_mean"`: average class probabilities
  - [ ] Implement `_get_aggregation_weights(model_names, prediction_store, branch_id, metric)`
- **Tests**:
  - Each aggregation strategy produces correct shape
  - Weighted mean normalization
  - Proba mean for multi-class

#### Task 5.3: Per-Branch Collection (1 day)
- **File**: `nirs4all/controllers/data/merge.py`
- **Actions**:
  - [ ] Implement `_collect_predictions_per_branch(dataset, context, branch_contexts, prediction_configs, prediction_store, mode)`
  - [ ] Iterate `BranchPredictionConfig` list
  - [ ] For each branch:
    1. Discover models
    2. Rank models
    3. Select based on strategy
    4. Collect OOF for selected
    5. Aggregate based on strategy
  - [ ] Concatenate all branch outputs
- **Tests**:
  - Single branch with selection + aggregation
  - Multiple branches with different strategies
  - Mixed: some branches select, some aggregate

#### Task 5.4: Integration Testing (0.5 day)
- **File**: `tests/integration/pipeline/test_merge_per_branch.py`
- **Actions**:
  - [ ] Test best per branch
  - [ ] Test top-K per branch
  - [ ] Test mean aggregation
  - [ ] Test weighted mean
  - [ ] Test different strategies per branch
  - [ ] Test explicit model selection
- **Tests**: All per-branch scenarios

#### Deliverable Phase 5
- Full per-branch control working
- All selection strategies
- All aggregation strategies
- Integration tests passing

---

### Phase 6: Mixed Merging (2 days)

**Goal**: Combine features and predictions from different branches

#### Task 6.1: Mixed Collection Logic (1 day)
- **File**: `nirs4all/controllers/data/merge.py`
- **Actions**:
  - [ ] Update `execute()` to handle mixed config
  - [ ] Collect features from feature branches
  - [ ] Collect predictions from prediction branches
  - [ ] Handle overlap (branch in both → features + predictions)
  - [ ] Concatenate: `[original? | features | predictions]`
- **Tests**:
  - Features from branch 0, predictions from branch 1
  - Same branch for both
  - With include_original

#### Task 6.2: Asymmetric Branch Handling (0.5 day)
- **File**: `nirs4all/controllers/data/merge.py`
- **Actions**:
  - [ ] Handle branch with only transforms → features only
  - [ ] Handle branch with only models → predictions only
  - [ ] Handle branch with both → based on config
  - [ ] Clear error when requesting predictions from transform-only branch
- **Tests**:
  - Asymmetric pipeline scenarios
  - Error cases

#### Task 6.3: Integration Testing (0.5 day)
- **File**: `tests/integration/pipeline/test_merge_mixed.py`
- **Actions**:
  - [ ] Test `{"merge": {"features": [1], "predictions": [0]}}`
  - [ ] Test `{"merge": "all"}` with multi-model branches
  - [ ] Test include_original with mixed
- **Tests**: Complex scenarios

#### Deliverable Phase 6
- Mixed feature + prediction merging
- Asymmetric branch handling
- All combinations tested

---

### Phase 7: MetaModel Refactoring (3-4 days)

**Goal**: Refactor MetaModel to use MergeController internally

#### Task 7.1: Static Merge Method (0.5 day)
- **File**: `nirs4all/controllers/data/merge.py`
- **Actions**:
  - [ ] Add `@classmethod merge_branches(cls, dataset, context, config, prediction_store, mode)`
  - [ ] Create minimal step_info
  - [ ] Call `execute()` internally
  - [ ] Return (context, StepOutput)
- **Tests**: Static method works

#### Task 7.2: MetaModel Config Translation (1 day)
- **File**: `nirs4all/controllers/models/meta_model.py`
- **Actions**:
  - [ ] Implement `_build_merge_config(meta_model, context) -> MergeConfig`
  - [ ] Map `source_models` to `model_filter` or `prediction_configs`
  - [ ] Map `branch_scope` to `prediction_branches`
  - [ ] Map `use_proba` to per-branch `proba`
  - [ ] Add new `include_features` → `collect_features + include_original`
  - [ ] Pass through `stacking_config` coverage/aggregation
- **Tests**: All mappings correct

#### Task 7.3: MetaModel Execute Refactoring (1.5 days)
- **File**: `nirs4all/controllers/models/meta_model.py`
- **Actions**:
  - [ ] Check if merge already applied (`context.custom.get("has_merged_features")`)
  - [ ] If yes: use existing merged features
  - [ ] If no: build merge config, call `MergeController.merge_branches()`
  - [ ] Get merged features from dataset
  - [ ] Train meta-learner
  - [ ] Keep rest of logic (prediction, storage, etc.)
- **Tests**:
  - MetaModel without prior merge → works
  - MetaModel after explicit merge → uses existing
  - Backward compatibility

#### Task 7.4: Backward Compatibility Testing (1 day)
- **File**: `tests/integration/pipeline/test_meta_model_backward_compat.py`
- **Actions**:
  - [ ] Run all existing MetaModel examples
  - [ ] Verify identical results
  - [ ] Test Q_meta_stacking.py
  - [ ] Test Q18_stacking.py
- **Tests**: No regressions

#### Deliverable Phase 7
- MetaModel uses MergeController internally
- Full backward compatibility
- New `include_features` capability

---

### Phase 8: Prediction Mode & Artifacts (2-3 days)

**Goal**: MergeController works in prediction/explain mode

#### Task 8.1: Manifest Storage (0.5 day)
- **File**: `nirs4all/controllers/data/merge.py`
- **Actions**:
  - [ ] Save `MergeConfig` to manifest during training
  - [ ] Include: feature_branches, prediction_configs, unsafe flag
  - [ ] Serialize `BranchPredictionConfig` list
- **Tests**: Config saved and retrievable

#### Task 8.2: Prediction Mode Execution (1.5 days)
- **File**: `nirs4all/controllers/data/merge.py`
- **Actions**:
  - [ ] Detect `mode == "predict"` or `mode == "explain"`
  - [ ] Load merge config from manifest
  - [ ] Apply saved transformers for feature branches
  - [ ] Load models and get test predictions
  - [ ] Aggregate test predictions per branch config
  - [ ] Create merged features for test data
- **Tests**:
  - Train → predict cycle
  - Feature branches in prediction mode
  - Prediction branches with aggregation

#### Task 8.3: Bundle Export Support (0.5 day)
- **File**: `nirs4all/pipeline/bundle/export.py`
- **Actions**:
  - [ ] Include merge artifacts in bundle
  - [ ] Export per-branch transformers
  - [ ] Export model references for prediction branches
- **Tests**: Bundle export/import with merge

#### Task 8.4: Full Cycle Testing (0.5 day)
- **File**: `tests/integration/pipeline/test_merge_prediction_mode.py`
- **Actions**:
  - [ ] Train pipeline with merge → save → load → predict
  - [ ] Verify prediction shapes
  - [ ] Verify feature reconstruction
- **Tests**: Complete cycle

#### Deliverable Phase 8
- Full prediction mode support
- Bundle export working
- Train/predict cycle tested

---

### Phase 9: Documentation & Examples (2 days)

**Goal**: Complete documentation and working examples

#### Task 9.1: Example File (1 day)
- **File**: `examples/Q_merge_branches.py`
- **Actions**:
  - [ ] Example 1: Feature merging
  - [ ] Example 2: Prediction merging (OOF safe)
  - [ ] Example 3: Mixed merging
  - [ ] Example 4: Per-branch selection (best, top-k)
  - [ ] Example 5: Per-branch aggregation (mean, weighted)
  - [ ] Example 6: Different strategies per branch
  - [ ] Example 7: Comparison with MetaModel
  - [ ] Example 8: Unsafe mode (with warnings)
- **Tests**: Example runs without error

#### Task 9.2: Reference Documentation (0.5 day)
- **File**: `docs/reference/branching.md`
- **Actions**:
  - [ ] Add "Merge" section
  - [ ] Document all syntax forms
  - [ ] Document selection strategies
  - [ ] Document aggregation strategies
  - [ ] Add troubleshooting section
- **Output**: Complete merge documentation

#### Task 9.3: Update Related Docs (0.5 day)
- **Files**: Multiple
- **Actions**:
  - [ ] Update `docs/user_guide/stacking.md` - explain merge relationship
  - [ ] Update `docs/specifications/pipeline_syntax.md` - add merge syntax
  - [ ] Update `.github/copilot-instructions.md` - mention merge
- **Output**: Consistent documentation

#### Deliverable Phase 9
- Comprehensive example file
- Complete reference documentation
- All docs updated

---

### Timeline Summary

| Phase | Duration | Depends On | Key Deliverable |
|-------|----------|------------|-----------------|
| Phase 1: Data Structures | 3-4 days | - | Enums, configs, parsing |
| Phase 2: Controller Skeleton | 2-3 days | Phase 1 | Branch exit, basic execute |
| Phase 3: Feature Merge | 2-3 days | Phase 2 | `{"merge": "features"}` |
| Phase 4: Prediction Simple | 3-4 days | Phase 2 | OOF, unsafe, simple syntax |
| Phase 5: Prediction Per-Branch | 3-4 days | Phase 4 | Selection + aggregation |
| Phase 6: Mixed Merge | 2 days | Phase 3, 5 | Features + predictions |
| Phase 7: MetaModel Refactor | 3-4 days | Phase 5 | Backward compat |
| Phase 8: Prediction Mode | 2-3 days | Phase 6, 7 | Train/predict cycle |
| Phase 9: Documentation | 2 days | Phase 8 | Examples and docs |

**Total Estimated Time**: 22-29 days

---

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| MetaModel backward compat | Phase 7 dedicated to testing all existing examples |
| OOF edge cases | Reuse proven TrainingSetReconstructor |
| Unsafe mode misuse | Prominent warnings at config, parse, and execute |
| Performance regression | Benchmark on Q18, Q_meta_stacking before/after |
| Complex config parsing | Extensive unit tests in Phase 1 |
| Per-branch complexity | Build on simple mode first (Phase 4 → 5) |

---

## Appendix: Relationship with Asymmetric Sources Design

The Codex review noted potential overlap between this design and the [asymmetric_sources_design.md](docs/specifications/asymmetric_sources_design.md). Here's the clarification:

### Different Scope

| Aspect | Branching Merge (this doc) | Source Branching (asymmetric sources) |
|--------|---------------------------|--------------------------------------|
| **Dimension** | Pipeline execution paths | Data sources (sensor modalities) |
| **Branch type** | Preprocessing/model variants | Different input sources |
| **Merge target** | Feature matrices + predictions | Multi-source features |
| **Keywords** | `branch`, `merge` | `source_branch`, `merge_sources` |
| **Controller** | `MergeController` | `SourceMergeController` (future) |

### Complementary, Not Overlapping

- **Branching Merge** (`merge`): Exits pipeline branch mode, combines outputs from different preprocessing paths
- **Source Merge** (`merge_sources`): Combines features from different data sources (NIR, markers, etc.)

Both can coexist:

```python
pipeline = [
    # Source-level branching (different sensors)
    {"source_branch": {
        "NIR": [SNV()],
        "markers": [VarianceThreshold()],
    }},
    {"merge_sources": "concat"},  # Combine sensor data

    # Pipeline-level branching (different models)
    {"branch": [[PLS(10)], [RandomForest()]]},
    {"merge": "predictions"},  # Stack predictions

    Ridge()  # Final model
]
```

### Implementation Note

When implementing `MergeController`, use a different keyword (`merge`) from the source merge (`merge_sources`) to maintain clarity. The controllers should be separate but may share internal utilities like `TrainingSetReconstructor`.

---

## Appendix: Decision Log

| Decision | Rationale | Alternatives Rejected |
|----------|-----------|----------------------|
| Merge as core primitive | Single source of truth for OOF; MetaModel uses it | MetaModel owns OOF (duplication) |
| OOF mandatory with explicit opt-out | Prevents accidental data leakage | Optional OOF default (dangerous) |
| `unsafe=True` for production opt-out | Advanced users may accept risk knowingly | No opt-out (too restrictive) |
| Prominent warnings for unsafe | Users must acknowledge leakage risk | Silent unsafe (leads to errors) |
| Merge ALWAYS exits branch mode | Simpler mental model, predictable | Partial merge (confusing) |
| MetaModel as thin wrapper | Backward compat, composition over inheritance | Deprecate MetaModel (breaking) |
| Keep concat_transform independent | Different purpose (single path transforms) | Merge into merge (slower) |
| Overlapping responsibilities by choice | User flexibility, multiple valid approaches | Single "correct" way (limiting) |
| Separate from source_merge | Different concerns (pipeline vs data sources) | Combined controller (confusing) |
