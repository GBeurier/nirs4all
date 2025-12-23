# Branching, Concat-Transform, and Merge: Design Review v6

**Version**: 6.0.0
**Status**: Comprehensive Design - Branch/Merge Distinction, Prediction Selection, Output Targets
**Date**: December 2025
**Author**: Design Review

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Branch vs Source: Fundamental Distinction](#2-branch-vs-source-fundamental-distinction)
3. [Verified Current State](#3-verified-current-state)
4. [Problem Analysis](#4-problem-analysis)
5. [Design: MergeController](#5-design-mergecontroller)
6. [Design: MetaModel Integration](#6-design-metamodel-integration)
7. [Design: ConcatTransformController (No Changes)](#7-design-concattransformcontroller-no-changes)
8. [Complete Use Case Matrix](#8-complete-use-case-matrix)
9. [Prediction Selection Specification](#9-prediction-selection-specification)
10. [Implementation Specification](#10-implementation-specification)
11. [Multi-Source Dataset Considerations](#11-multi-source-dataset-considerations)
12. [Asymmetric Branch Design](#12-asymmetric-branch-design)
13. [Error Catalog and Resolution](#13-error-catalog-and-resolution)
14. [Unified Merge Controller](#14-unified-merge-controller)
15. [Implementation Roadmap](#15-implementation-roadmap)
16. [Appendix: Relationship with Asymmetric Sources Design](#appendix-relationship-with-asymmetric-sources-design)

---

## 1. Executive Summary

### Core Design Principle: Branches and Sources are Distinct Concepts

**Branches** and **Sources** represent fundamentally different dimensions in nirs4all:

| Concept | Dimension | Purpose | Created By |
|---------|-----------|---------|------------|
| **Branch** | Execution parallelism | Run N independent pipeline paths | `{"branch": [...]}` |
| **Source** | Data provenance | Data from different origins/modalities | Dataset config or `merge` with `output_as: "sources"` |

**Critical Insight**: Branches represent *computation* (parallel execution), while sources represent *data origin* (where features came from). They should NOT automatically convert because:

1. **Training per-branch models** requires staying in branch mode
2. **Source semantics** implies "independent data," not "alternative processing"
3. **Debug/traceability** is clearer with explicit conversion points

The `merge` step is the **explicit transition point** that can:
- Exit branch mode and return to single-path execution
- Optionally convert branch outputs to sources (via `output_as: "sources"`)
- Combine features and/or predictions with full control

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

#### 4. Restore MetaModel as Standalone Operator

**The Problem**: Previous designs proposed refactoring `MetaModel` to delegate to `MergeController`. This was a misunderstanding of their distinct roles. Stacking (MetaModel) works on the *history* of predictions, while Merge works on *parallel* branch outputs.

**The Solution**: Restore `MetaModel` as a **standalone operator** that:
1. Works without any branch context (flat pipelines)
2. Does NOT modify execution context (unlike merge)
3. Uses shared utilities (`TrainingSetReconstructor`) for OOF logic
4. Can be used *alongside* merge but is not *dependent* on it

**User-Facing Behavior**: Unchanged. Users can still write:
```python
PLS(), RF(), {"model": MetaModel(Ridge())}
```

**Internal Implementation**: MetaModel uses `TrainingSetReconstructor` directly to fetch predictions from the store, without invoking `MergeController` or exiting branch mode.

**Benefits**:
- Clear separation of concerns: Stacking vs Branch Merging
- MetaModel works in all contexts (flat, inside branch, after merge)
- No forced branch exit when stacking
- Simpler implementation and debugging

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
| **MetaModel standalone** | MetaModel works independently of branches, using shared OOF logic |

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
            └──────────────┘  └──────┬───────┘  └──────┬───────┘
                                     │                 │
                    ┌────────────────┼────────────────┐│
                    ▼                ▼                ▼▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │  Collect     │  │  OOF Recon   │  │  MetaModel   │
            │  Features    │  │  (Shared)    │  │  Controller  │
            └──────────────┘  └──────────────┘  └──────────────┘
                                     ▲                 │
                                     │                 │
                                     └──── uses ───────┘
```

---

## 2. Branch vs Source: Fundamental Distinction

This section explains why branches should NOT automatically become sources, and why merge is the explicit transition point.

### 2.1 The Two Dimensions

nirs4all operates on two orthogonal dimensions that must remain conceptually separate:

```
                    Sources (Data Provenance)
                    ┌─────┬─────┬─────┐
                    │ S0  │ S1  │ S2  │  ← Different sensors/modalities
                    ├─────┼─────┼─────┤
    Branches        │     │     │     │  Branch 0 (SNV path)
    (Execution      ├─────┼─────┼─────┤
     Paths)         │     │     │     │  Branch 1 (MSC path)
                    └─────┴─────┴─────┘
```

| Dimension | Represents | Lifecycle | Shared State |
|-----------|------------|-----------|--------------|
| **Sources** | Where data comes from | Persists throughout pipeline | Sample indices, y values |
| **Branches** | Alternative processing paths | Temporary (until merge) | Context, prediction store |

### 2.2 Why Branches Cannot Automatically Become Sources

The user proposed: "What if merge just adds each branch output as a source?"

This seems elegant but **breaks fundamental capabilities**:

#### Problem 1: Cannot Train Models Per Branch

With multi-source, models see **concatenated features** by default:

```python
# If branches auto-converted to sources:
{"branch": [[SNV()], [MSC()]]},  # → Creates source_snv, source_msc
PLSRegression()  # Sees concatenated features → ONE model trained

# But user wants:
{"branch": [[SNV(), PLSRegression()], [MSC(), PLSRegression()]]},
# → TWO models trained, one per branch (requires branch mode)
```

**Branch mode keeps steps running N times**, enabling per-branch model training.

#### Problem 2: Prediction Store Loses Branch Association

Predictions are stored in `prediction_store` by model name, not by source. If branches become sources:

```python
{"branch": [[SNV(), PLS()], [MSC(), RF()]]},
# Predictions stored as: {"PLS": ..., "RF": ...}
# No information about which prediction came from which branch!
```

**Stacking requires branch→prediction association** for proper selection.

#### Problem 3: Source Semantics Conflict

Sources semantically mean "data from different origins" (sensors, instruments). Branches mean "alternative processing of the same data."

```python
# This is confusing:
dataset = load("nir.csv")  # One source
{"branch": [[SNV()], [MSC()]]}  # Two branches
# If branches → sources: Now we have 3 sources?
# The SNV/MSC "sources" don't represent different instruments!
```

**Sources should represent data provenance**, not processing alternatives.

### 2.3 The Right Model: Merge as Transition Point

`merge` is the **explicit transition point** between branch mode and single-path execution:

```python
{"branch": [[SNV()], [MSC()]]},  # Enter branch mode (N=2)
SavitzkyGolay(),                 # Runs 2× (once per branch)
PCA(50),                         # Runs 2× (once per branch)
{"merge": "features"},           # EXIT branch mode → single path
PLSRegression()                  # Runs 1× on merged features
```

### 2.4 Merge Output Options

Merge can output to different targets based on user needs:

```python
# Default: Concatenated features (2D matrix)
{"merge": "features"}
# Result: Single processing with horizontally concatenated features

# Output as sources (preserve branch identity)
{"merge": {"features": "all", "output_as": "sources"}}
# Result: Each branch becomes a source (branch_0, branch_1)
# Use case: Different downstream processing per branch output

# Output as predictions (for stacking)
{"merge": {"predictions": "all", "output_as": "features"}}
# Result: OOF predictions become feature columns

# Keep as structured dict (for multi-head models)
{"merge": {"features": "all", "output_as": "dict"}}
# Result: {"branch_0": array, "branch_1": array}
```

### 2.5 When to Use `output_as: "sources"`

Use `output_as: "sources"` when:

1. **Different downstream processing per branch output**:
   ```python
   {"branch": [[PCA(50)], [AutoEncoder(20)]]},
   {"merge": {"features": "all", "output_as": "sources"}},
   # Now branch outputs are sources: pca_branch, ae_branch
   {"source_transform": {
       "pca_branch": [MinMaxScaler()],
       "ae_branch": [StandardScaler()]
   }}
   ```

2. **Multi-head model with branch-specific inputs**:
   ```python
   {"branch": [[SNV(), PCA(50)], [MSC(), PLS(10)]]},
   {"merge": {"features": "all", "output_as": "sources"}},
   MultiHeadNN(input_sources=["snv_branch", "msc_branch"])
   ```

3. **Late fusion with branch-aware selection**:
   ```python
   {"merge": {"features": "all", "output_as": "sources"}},
   {"select_sources": ["snv_branch"]},  # Use only one branch's output
   PLSRegression()
   ```

### 2.6 Summary: The Design Principle

```
┌────────────────────────────────────────────────────────────────┐
│  Branch Mode                              │  Single-Path Mode  │
│  ─────────────                            │  ────────────────  │
│                                           │                    │
│  Steps run N times                        │  Steps run 1×      │
│  Per-branch models                        │  Single model      │
│  Branch contexts tracked                  │  No branch state   │
│                                           │                    │
│  {"branch": [...]} ──────► {"merge": ...} ──────►  [steps]     │
│                      ▲                    │                    │
│                      │                    │                    │
│         EXPLICIT TRANSITION POINT         │                    │
│         User controls output_as           │                    │
└────────────────────────────────────────────────────────────────┘
```

**Key Decisions**:
- Branches **do not** automatically become sources
- `merge` is **required** to exit branch mode
- `merge` offers **output_as** for flexibility: features, sources, dict
- Predictions require **explicit selection** from branches/sources

---

## 3. Verified Current State

### 3.1 BranchController ✅ Verified Correct

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

### 3.2 ConcatAugmentationController ✅ Verified Correct

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

### 3.3 MergeController ❌ Empty - Needs Implementation

**Location**: [nirs4all/controllers/data/merge.py](nirs4all/controllers/data/merge.py)

**Current State**: Empty file (0 bytes)

**Required**: Full implementation as specified in Section 4.

### 3.4 MetaModelController ✅ Feature-Complete, Needs Integration

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

**Proposed**: Keep as standalone controller. Share OOF reconstruction logic with MergeController via `TrainingSetReconstructor`.

### 3.5 TrainingSetReconstructor ✅ Complete

**Location**: [nirs4all/controllers/models/stacking/reconstructor.py](nirs4all/controllers/models/stacking/reconstructor.py)

**Verified Capabilities**:
- OOF reconstruction from prediction store
- Fold alignment validation
- Coverage handling per stacking config
- Classification support with probability features
- Branch-aware prediction filtering

This is the core engine that MergeController will use for prediction merging.

---

## 4. Problem Analysis

### 4.1 Gap 1: No Branch Exit Mechanism

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

### 4.2 Gap 2: Cannot Mix Features and Predictions

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

### 4.3 Gap 3: Asymmetric Branch Handling

**Scenario**: Multiple models in one branch, only features in another:
```python
{"branch": [[SNV(), PLS(), RF(), XGB()], [PCA(10)]]},
{"merge": {"predictions": {"branches": [0], "models": ["XGB"]}, "features": [1]}}
```

**Required**: Merge controller must handle:
- Selective model prediction extraction
- Feature extraction from branches without models
- Proper OOF reconstruction for selected predictions

### 4.4 Gap 4: Multi-Model Branch Aggregation

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

## 5. Design: MergeController

### 5.1 Core Principle: Merge as Foundation

`MergeController` is the **core primitive** for all branch combination operations. It is designed to:

1. **Always exit branch mode** — Merge unconditionally clears branch contexts and returns to single-path execution
2. **Handle both features and predictions** — Unified interface for all branch output types
3. **Enforce OOF safety by default** — Predictions are always reconstructed using out-of-fold strategy unless explicitly disabled
4. **Be composable** — Can be used alone or as the foundation for higher-level operators like `MetaModel`

### 5.2 Controller Specification

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

### 5.3 Syntax Specification

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

# =============================================================================
# OUTPUT TARGET (Where merged data goes)
# =============================================================================

# Default: Merged output becomes a single feature matrix
{"merge": "features"}  # output_as: "features" (default)

# As sources: Each branch becomes a separate source
{"merge": {
    "features": "all",
    "output_as": "sources"  # Creates branch_0, branch_1, ... sources
}}
# Use case: Different downstream processing per branch output

# As dict: Keep structured for multi-head models
{"merge": {
    "features": "all",
    "output_as": "dict"  # Returns {"branch_0": array, "branch_1": array}
}}
# Use case: Multi-head neural networks

# Predictions as sources (for source-aware downstream)
{"merge": {
    "predictions": "all",
    "output_as": "sources"  # Each branch's predictions become a source
}}

# Named sources from named branches
{"merge": {
    "features": "all",
    "output_as": "sources",
    "source_names": ["snv_features", "msc_features"]  # Custom names
}}
```

### 5.4 OOF Safety Model

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

### 5.5 Behavior Specification

| Merge Type | Branch Has Only Transforms | Branch Has Transform + Model | Branch Has Only Model |
|------------|---------------------------|------------------------------|----------------------|
| `"features"` | ✅ Use `features_snapshot` | ✅ Use `features_snapshot` (ignores model) | ✅ Use pre-model features |
| `"predictions"` | ❌ Error (no model) | ✅ OOF predictions | ✅ OOF predictions |
| `"all"` | ✅ Features only | ✅ Features + Predictions | ✅ Features + Predictions |

### 5.6 Per-Branch Model Selection & Aggregation

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

### 5.7 Output Processing

Merge creates a new processing called `"merged"`:
- Features are horizontal-concatenated: `np.hstack([branch0_features, branch1_features, ...])`
- Predictions are OOF-reconstructed via `TrainingSetReconstructor`
- Combined output: `np.hstack([merged_features, oof_predictions])`

The merged processing replaces all branch processings and becomes the active processing for subsequent steps.

### 5.8 Prediction Mode Support

In prediction/explain mode:
1. **Features**: Load and apply saved transformers from each branch, concatenate
2. **Predictions**: Load source models, get test predictions, aggregate per `TestAggregation` config
3. **Merge config**: Saved in manifest for reproducibility

---

## 6. Design: MetaModel Integration

### 6.1 Architectural Decision: MetaModel Remains Standalone

**Previous Proposal (Rejected)**:
```
MetaModel → MergeController → Merged features → Train
```

**Final Architecture**:
```
MetaModel → TrainingSetReconstructor → OOF predictions → Train
MergeController → TrainingSetReconstructor → OOF predictions → Merge
```

**Rationale**:
1. **Separation of Concerns**: Stacking (MetaModel) is about *model history*. Merging is about *branch combination*. While they share the need for OOF predictions, they are distinct operations.
2. **Context Preservation**: MetaModel should not modify the execution context (it just adds a model). Merge *must* modify the context (exit branch mode).
3. **Flexibility**: Users can stack within a branch without exiting it.

### 6.2 Shared Utilities

Both `MetaModelController` and `MergeController` use shared components located in `nirs4all/controllers/shared/`:

1. **TrainingSetReconstructor**: Handles OOF prediction reconstruction from the prediction store.
2. **ModelSelector**: Handles selection of models (best, top_k, etc.) based on validation metrics.
3. **PredictionAggregator**: Handles aggregation of predictions (mean, weighted_mean, etc.).

### 6.3 User-Facing Equivalences

While the implementations are separate, users can achieve similar results with different patterns:

```python
# Pattern A: MetaModel (Standard Stacking)
pipeline = [
    KFold(n_splits=5),
    PLS(10), RF(),
    {"model": MetaModel(Ridge(), source_models="all")}
]

# Pattern B: Explicit Merge + Model (Branch Combination)
pipeline = [
    KFold(n_splits=5),
    {"branch": [[PLS(10)], [RF()]]},  # Explicit branches
    {"merge": "predictions"},         # Exit branches, combine OOF
    {"model": Ridge()}                # Train on combined
]
```

**When to use which?**
- Use **MetaModel** for standard stacking (ensemble learning) where you want to combine previous models.
- Use **Merge + Model** when you have explicit parallel branches that need to be combined and then processed further.

### 6.4 MetaModel Capabilities

MetaModel retains its full feature set:
- **Source Selection**: `source_models=["PLS", "RF"]` or `source_models="all"`
- **Branch Scope**: `BranchScope.CURRENT_ONLY` (default) or `BranchScope.ALL_BRANCHES`
- **Feature Passthrough**: `include_features=True` (new)
- **Coverage Strategies**: Strict, Drop Incomplete, Impute

### 6.5 Internal Implementation

MetaModel's execute method uses the shared reconstructor:

```python
class MetaModelController(OperatorController):
    def execute(self, step_info, dataset, context, ...):
        # ... setup ...

        # Use shared reconstructor directly
        reconstructor = TrainingSetReconstructor(...)
        result = reconstructor.reconstruct(dataset, context)

        X_train = result.X_train_meta
        # ... train meta-model ...
```

It does **NOT** delegate to `MergeController`.

### 6.6 Backward Compatibility

All existing MetaModel pipelines continue to work unchanged. The refactoring is purely internal to share code with MergeController without coupling them.

---

## 7. Design: ConcatTransformController (No Changes)

### 7.1 Current Behavior (Keep)

```python
{"concat_transform": [PCA(50), SVD(30), None]}
```

- Applies each transformer to current features
- Concatenates horizontally: `[PCA_output | SVD_output | original]`
- Works within single execution path
- No branch awareness required

### 7.2 Relationship to Merge

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

## 8. Complete Use Case Matrix

### 8.1 Feature-Only Pipelines

| Use Case | Syntax |
|----------|--------|
| Single transform | `SNV()` |
| Transform chain | `SNV(), PCA(50)` |
| Concat transforms (single path) | `{"concat_transform": [SNV(), MSC()]}` |
| Multiple processings | `{"feature_augmentation": [SNV(), MSC()]}` |
| Branch + merge features | `{"branch": [[SNV()], [MSC()]]}, {"merge": "features"}` |

### 8.2 Model Pipelines

| Use Case | Syntax |
|----------|--------|
| Single model | `{"model": PLSRegression()}` |
| Sequential models | `PLS(), RF(), XGB()` |
| Branch preprocessing + single model | `{"branch": [[SNV()], [MSC()]]}, PLS()` → runs 2× |
| Branch preprocessing + merge + single model | `{"branch": [[SNV()], [MSC()]]}, {"merge": "features"}, PLS()` → runs 1× |

### 8.3 Stacking Pipelines (Safe - OOF)

| Use Case | Syntax |
|----------|--------|
| Stack all previous models (MetaModel) | `PLS(), RF(), {"model": MetaModel(Ridge())}` |
| Stack all previous models (explicit) | `PLS(), RF(), {"merge": "predictions"}, {"model": Ridge()}` |
| Stack selected models | `PLS(), RF(), XGB(), {"model": MetaModel(Ridge(), source_models=["RF", "XGB"])}` |
| Stack across branches | `{"branch": [[SNV(), PLS()], [MSC(), RF()]]}, {"merge": "predictions"}, Ridge()` |
| Mix features + predictions | `{"branch": [[SNV(), PLS()], [PCA(10)]]}, {"merge": {"features": [1], "predictions": [0]}}, Ridge()` |
| Stack + include original features | `{"merge": {"predictions": "all", "include_original": True}}, Ridge()` |

### 8.4 Unsafe Mode (Rapid Prototyping Only)

| Use Case | Syntax | Warning |
|----------|--------|---------|
| Quick shape check | `{"merge": {"predictions": "all", "unsafe": True}}` | ⚠️ DATA LEAKAGE |
| Fast iteration | `{"merge": {"predictions": [0], "unsafe": True}}, Ridge()` | ⚠️ DATA LEAKAGE |

**⚠️ Important**: Unsafe mode disables OOF reconstruction. Training predictions are used directly, which causes data leakage and overly optimistic metrics. **Do NOT use for final model evaluation.**

### 8.5 Per-Branch Model Selection & Aggregation

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

### 8.6 Complex Pipelines

| Use Case | Syntax |
|----------|--------|
| Nested branches + merge | `{"branch": [[{"branch": [...]}, PLS()], [...]]}, {"merge": "predictions"}` |
| Selective model from multi-model branch | `{"branch": [[SNV(), PLS(), RF(), XGB()]]}, {"merge": {"predictions": {"models": ["XGB"]}}}` |
| MetaModel with feature inclusion | `PLS(), RF(), {"model": MetaModel(Ridge(), include_features=True)}` |

### 8.7 Equivalence Table

These pairs are semantically equivalent (produce same results):

| MetaModel Syntax | Explicit Merge Syntax |
|------------------|----------------------|
| `{"model": MetaModel(Ridge())}` | `{"merge": "predictions"}, {"model": Ridge()}` |
| `{"model": MetaModel(Ridge(), source_models=["RF"])}` | `{"merge": {"predictions": {"models": ["RF"]}}}, {"model": Ridge()}` |
| `{"model": MetaModel(Ridge(), branch_scope=ALL_BRANCHES)}` | `{"merge": {"predictions": {"branches": "all"}}}, {"model": Ridge()}` |
| `{"model": MetaModel(Ridge(), include_features=True)}` | `{"merge": {"predictions": "all", "include_original": True}}, {"model": Ridge()}` |

---

## 9. Prediction Selection Specification

This section provides comprehensive documentation for selecting predictions from branches and sources during stacking and merging operations.

### 9.1 The Prediction Selection Problem

When merging predictions from multiple branches/sources, users need fine-grained control:

1. **Which branches** to include
2. **Which models** within each branch
3. **How to aggregate** multiple models' predictions
4. **From which sources** (in multi-source scenarios)
5. **Which prediction partition** (train OOF, validation, test)

### 9.2 Prediction Selection Syntax

```python
# =============================================================================
# SIMPLE SELECTION
# =============================================================================

# All predictions from all branches
{"merge": "predictions"}

# Predictions from specific branches
{"merge": {"predictions": [0, 2]}}  # Branch indices

# =============================================================================
# PER-BRANCH SELECTION
# =============================================================================

# Full per-branch control
{"merge": {
    "predictions": [
        {
            "branch": 0,                           # Required: branch index or name
            "models": "all",                       # "all" | "best" | ["model1", "model2"] | {"top_k": 3}
            "metric": "rmse",                      # Metric for selection (rmse, mae, r2, accuracy, f1)
            "aggregate": "separate",               # "separate" | "mean" | "weighted_mean" | "proba_mean"
            "proba": False,                        # Use class probabilities (classification)
            "sources": "all"                       # Source filter: "all" | [0, 1] | ["NIR", "markers"]
        },
        {
            "branch": 1,
            "models": "best",
            "metric": "r2"
        }
    ]
}}

# =============================================================================
# MODEL SELECTION STRATEGIES
# =============================================================================

# All models (default): Each model = one feature
{"merge": {"predictions": [{"branch": 0, "models": "all"}]}}

# Best model only: One feature (highest metric)
{"merge": {"predictions": [{"branch": 0, "models": "best", "metric": "rmse"}]}}

# Top K models: K features
{"merge": {"predictions": [{"branch": 0, "models": {"top_k": 3}, "metric": "r2"}]}}

# Explicit model names: Specific models only
{"merge": {"predictions": [{"branch": 0, "models": ["PLS", "RF"]}]}}

# By model type: All instances of a model class
{"merge": {"predictions": [{"branch": 0, "models": {"type": "PLSRegression"}}]}}

# =============================================================================
# AGGREGATION STRATEGIES
# =============================================================================

# Separate (default): Each model = one feature column
{"merge": {"predictions": [{"branch": 0, "aggregate": "separate"}]}}
# Result: N features (one per selected model)

# Mean: Average all selected models
{"merge": {"predictions": [{"branch": 0, "aggregate": "mean"}]}}
# Result: 1 feature

# Weighted Mean: Weight by validation performance
{"merge": {"predictions": [{"branch": 0, "aggregate": "weighted_mean", "weight_metric": "r2"}]}}
# Result: 1 feature

# Proba Mean: Average class probabilities (classification)
{"merge": {"predictions": [{"branch": 0, "proba": True, "aggregate": "proba_mean"}]}}
# Result: K features (one per class)

# =============================================================================
# SOURCE-AWARE SELECTION (Multi-Source Datasets)
# =============================================================================

# Predictions only from models trained on specific sources
{"merge": {"predictions": [{"branch": 0, "sources": ["NIR"]}]}}

# Cross-source: Combine predictions from different source-specific models
{"merge": {
    "predictions": [
        {"branch": 0, "sources": ["NIR"], "aggregate": "mean"},
        {"branch": 0, "sources": ["markers"], "aggregate": "mean"}
    ]
}}
```

### 9.3 Selection by Branch Name

Branches can be named for clearer selection:

```python
# Named branches
{"branch": {
    "spectral_path": [SNV(), PLS(10)],
    "feature_path": [PCA(50), RF()]
}},

# Select by name
{"merge": {
    "predictions": [
        {"branch": "spectral_path", "models": "best"},
        {"branch": "feature_path", "models": "all"}
    ],
    "features": ["spectral_path"]  # Features from one branch
}}
```

### 9.4 Combining Selection and Aggregation

The `models` (selection) and `aggregate` parameters work together:

| Selection | Aggregation | Result |
|-----------|-------------|--------|
| `"all"` (3 models) | `"separate"` | 3 features |
| `"all"` (3 models) | `"mean"` | 1 feature (mean of 3) |
| `"best"` | (ignored) | 1 feature |
| `{"top_k": 2}` | `"separate"` | 2 features |
| `{"top_k": 2}` | `"weighted_mean"` | 1 feature |
| `["RF", "XGB"]` | `"separate"` | 2 features |
| `["RF", "XGB"]` | `"mean"` | 1 feature |

### 9.5 Cross-Branch Selection

Select predictions across branches with different strategies:

```python
# Different strategies per branch
{"merge": {
    "predictions": [
        {"branch": 0, "models": "best", "metric": "rmse"},           # 1 feature
        {"branch": 1, "models": {"top_k": 2}, "aggregate": "mean"},  # 1 feature
        {"branch": 2, "models": "all"}                                # N features
    ]
}}

# Mixed: Features from some branches, predictions from others
{"merge": {
    "features": [1],           # Features from branch 1
    "predictions": [
        {"branch": 0, "models": "best"},    # Best prediction from branch 0
        {"branch": 2, "models": "all"}      # All predictions from branch 2
    ]
}}
```

### 9.6 Stacking Configuration Integration

For `MetaModel`, prediction selection maps to merge configuration:

```python
# MetaModel with selection (high-level API)
{"model": MetaModel(
    model=Ridge(),
    source_models="best",           # → models: "best"
    selection_metric="rmse",        # → metric: "rmse"
    branch_scope=BranchScope.ALL,   # → branches: "all"
    aggregate="mean"                # → aggregate: "mean"
)}

# Equivalent explicit merge (low-level API)
{"merge": {
    "predictions": [
        {"branch": "all", "models": "best", "metric": "rmse", "aggregate": "mean"}
    ]
}},
{"model": Ridge()}
```

### 9.7 Output Targets for Predictions

Control where merged predictions go:

```python
# As features (default): Predictions become feature columns
{"merge": {"predictions": "all", "output_as": "features"}}

# As sources: Each branch's predictions become a separate source
{"merge": {"predictions": "all", "output_as": "sources"}}

# As dict: Keep as structured dict for multi-head models
{"merge": {"predictions": "all", "output_as": "dict"}}
```

### 9.8 Validation Rules

The merge controller validates prediction selections:

| Rule | Validation | Error Code |
|------|------------|------------|
| Branch exists | Branch index/name in contexts | MERGE-E021 |
| Model exists | Named models exist in prediction store | MERGE-E013 |
| Has predictions | Branch has models (for prediction merge) | MERGE-E010 |
| Metric available | Validation scores exist for ranking | MERGE-E015 |
| Source exists | Named source in dataset | MERGE-E031 |

### 9.9 Complete Stacking Example

```python
pipeline = [
    KFold(n_splits=5),

    # Create 3 branches with different preprocessing + models
    {"branch": {
        "snv_path": [SNV(), Detrend(), PLSRegression(10), RandomForestRegressor()],
        "msc_path": [MSC(), SavitzkyGolay(), PLSRegression(15), SVR()],
        "pca_path": [PCA(50), PLSRegression(20)]
    }},

    # Selective merge: Best from first two branches, all from third
    {"merge": {
        "predictions": [
            {"branch": "snv_path", "models": "best", "metric": "rmse"},   # 1 feature
            {"branch": "msc_path", "models": "best", "metric": "rmse"},   # 1 feature
            {"branch": "pca_path", "models": "all", "aggregate": "mean"}  # 1 feature
        ],
        "include_original": True  # Also include pre-branch features
    }},

    # Stack with Ridge
    {"model": Ridge()}
]
```

---

## 10. Implementation Specification

### 10.1 MergeConfig Dataclass

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

### 10.2 MergeController Implementation

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

### 10.3 SpectroDataset Extension

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

### 10.4 MetaModelController Refactoring

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

### 10.5 File Changes Summary

| File | Action | Status | Phase |
|------|--------|--------|-------|
| `nirs4all/operators/data/merge.py` | **Create** MergeMode, SelectionStrategy, AggregationStrategy enums | ✅ Done | 1 |
| `nirs4all/operators/data/merge.py` | **Create** BranchPredictionConfig dataclass | ✅ Done | 1 |
| `nirs4all/operators/data/merge.py` | **Create** MergeConfig dataclass | ✅ Done | 1 |
| `nirs4all/controllers/data/merge.py` | **Implement** MergeController skeleton | ✅ Done | 2 |
| `nirs4all/controllers/data/merge.py` | **Implement** config parsing | ✅ Done | 1 |
| `nirs4all/controllers/data/merge.py` | **Implement** feature collection | ✅ Done | 3 |
| `nirs4all/controllers/data/merge.py` | **Implement** OOF prediction collection | ✅ Done | 4 |
| `nirs4all/controllers/data/merge.py` | **Implement** per-branch selection/aggregation | ✅ Done | 5 |
| `nirs4all/data/dataset.py` | **Add** `add_merged_features()`, `get_merged_features()` | ✅ Done | 2 |
| `nirs4all/controllers/models/meta_model.py` | **Refactor** to use MergeController | ⏳ Pending | 7 |
| `tests/unit/operators/data/test_merge_config.py` | **Create** config parsing tests | ✅ Done | 1 |
| `tests/unit/controllers/data/test_merge_controller.py` | **Create** controller unit tests | ✅ Done | 2-5 |
| `tests/integration/pipeline/test_merge_features.py` | **Create** feature merge tests | ✅ Done | 3 |
| `tests/integration/pipeline/test_merge_predictions.py` | **Create** prediction merge tests | ✅ Done | 4 |
| `tests/integration/pipeline/test_merge_per_branch.py` | **Create** per-branch strategy tests | ✅ Done | 5 |
| `tests/integration/pipeline/test_merge_mixed.py` | **Create** mixed merge tests | ✅ Done | 6 |
| `tests/integration/pipeline/test_meta_model_backward_compat.py` | **Create** backward compat tests | ⏳ Pending | 7 |
| `tests/integration/pipeline/test_merge_prediction_mode.py` | **Create** prediction mode tests | ⏳ Pending | 8 |
| `examples/Q_merge_branches.py` | **Create** comprehensive examples | ⏳ Pending | 11 |
| `docs/reference/branching.md` | **Update** with merge documentation | ⏳ Pending | 11 |
| `docs/user_guide/stacking.md` | **Update** merge relationship | ⏳ Pending | 11 |
| `docs/specifications/pipeline_syntax.md` | **Update** with merge syntax | ⏳ Pending | 11 |

---

## 11. Multi-Source Dataset Considerations

This section addresses the interaction between branching, merging, and multi-source datasets—a critical gap in the original design.

### 11.1 Background: Multi-Source Datasets

nirs4all supports datasets with multiple feature sources:

```python
# Example: NIR spectra + genetic markers
dataset_config = {
    "sources": [
        {"name": "NIR", "train_x": "nir.csv"},       # 500 features
        {"name": "markers", "train_x": "snps.csv"},  # 50,000 features
    ],
    "train_y": "phenotypes.csv"
}
```

Each source has shape `(samples × processings × features)`, and sources can have different feature counts and processing histories.

### 11.2 The Branches × Sources Interaction

When branching with multi-source datasets, a **cross-product** relationship emerges:

```python
pipeline = [
    {"branch": [[SNV()], [MSC()]]},  # 2 branches
    # With 3 sources, we have: 2 branches × 3 sources = 6 "sub-contexts"?
]
```

**Current Behavior**: Each branch contains **all sources**. Transformers in a branch are applied to all sources independently. This means:

- Branch 0 (SNV): SNV applied to NIR, SNV applied to markers, ...
- Branch 1 (MSC): MSC applied to NIR, MSC applied to markers, ...

The branching dimension and source dimension are **orthogonal**.

### 11.3 Feature Merge with Multi-Source

When merging features from branches:

```python
{"branch": [[SNV()], [MSC()]]},
{"merge": "features"}
```

**Question**: How are sources handled during feature merge?

**Answer**: Merge operates on the **2D flattened view** of features. Each branch's `features_snapshot` contains all sources, already concatenated to 2D by the dataset accessor.

```
Branch 0 (SNV): [NIR_snv | markers_snv] = (samples, NIR_features + marker_features)
Branch 1 (MSC): [NIR_msc | markers_msc] = (samples, NIR_features + marker_features)

After merge: [Branch0_features | Branch1_features]
           = (samples, 2 × (NIR_features + marker_features))
```

**Important**: This assumes all branches have the same sources. If branches differ in source selection, a **shape mismatch error** occurs (see Section 12).

### 11.4 Prediction Merge with Multi-Source

Prediction merge is **source-agnostic**:

```python
{"branch": [[SNV(), PLS(10)], [MSC(), RF()]]},
{"merge": "predictions"}
```

When models are trained, they receive the concatenated feature view (all sources merged). Predictions are per-sample scalars (regression) or class probabilities (classification).

**Result**: Predictions don't carry source information—they represent the model's output on the full feature space.

### 11.5 Source Branching vs Pipeline Branching

There are **two types of branching** with distinct purposes:

| Aspect | Pipeline Branching (`branch`) | Source Branching (`source_branch`) |
|--------|-------------------------------|-----------------------------------|
| **Dimension** | Pipeline execution paths | Data source modalities |
| **Scope** | All sources processed per branch | Each source gets its own branch |
| **Use case** | Compare preprocessing strategies | Per-modality pipelines |
| **Exit via** | `{"merge": ...}` | `{"merge_sources": ...}` |

**Pipeline Branching Example**:
```python
{"branch": [[SNV(), PCA(50)], [MSC(), PCA(50)]]}  # Compare SNV vs MSC
```

**Source Branching Example** (proposed):
```python
{"source_branch": {
    "NIR": [SNV(), SavitzkyGolay()],
    "markers": [VarianceThreshold()]
}}
```

### 11.6 Merge Modes for Multi-Source

The unified `MergeController` supports different modes for multi-source handling:

```python
# Standard: Sources already flattened, merge operates on 2D view
{"merge": "features"}

# Source-aware: Preserve source structure during merge
{"merge": {
    "features": "all",
    "source_handling": "preserve"  # "concat" (default) | "preserve" | "dict"
}}

# Source merge (separate operation): Combine sources into single feature space
{"merge_sources": "concat"}  # After source_branch or when sources need unification
```

### 11.7 Multi-Source Error Scenarios

| Scenario | Error Type | Resolution |
|----------|------------|------------|
| Branch drops a source | MERGE-E003 | Ensure all branches process all sources |
| Different source order per branch | Shape mismatch | Not possible (sources are ordered by index) |
| Source branch without merge_sources | Infinite branch mode | Add `{"merge_sources": ...}` |
| merge_sources on single-source dataset | Warning (no-op) | Remove unnecessary merge_sources |

---

## 12. Asymmetric Branch Design

Branches can be **asymmetric**—having different internal structures. This section defines what asymmetry means and how to handle it.

### 12.1 Dimensions of Asymmetry

Branches can differ along multiple dimensions:

| Dimension | Example | Impact on Merge |
|-----------|---------|-----------------|
| **Step count** | Branch 0: 3 steps, Branch 1: 1 step | No impact (final output matters) |
| **Operator types** | Branch 0: transforms only, Branch 1: transform + model | Mixed merge required |
| **Output dimensions** | Branch 0: PCA(50), Branch 1: PCA(10) | Feature dimension mismatch |
| **Model count** | Branch 0: 3 models, Branch 1: 1 model | Per-branch aggregation needed |
| **Source handling** | Branch 0: uses 2 sources, Branch 1: uses 1 source | Shape mismatch |
| **Processing count** | Branch 0: generates 3 processings, Branch 1: generates 1 | 3D layout fails |

### 12.2 Asymmetric Feature Dimensions

When branches produce different feature dimensions:

```python
{"branch": [
    [PCA(n_components=50)],   # Branch 0: 50 features
    [PCA(n_components=10)]    # Branch 1: 10 features
]},
{"merge": "features"}  # 60 features total (horizontal concat)
```

**This is allowed**. Feature merge performs horizontal concatenation regardless of per-branch feature counts. The final feature count is the sum of all branch feature counts.

**When is this problematic?**
- When you expect uniform feature dimensions for downstream processing
- When using 3D layout that requires aligned dimensions

### 12.3 Asymmetric Model Presence

When some branches have models and others don't:

```python
{"branch": [
    [SNV(), PLS(10)],         # Branch 0: has model → predictions
    [MSC(), PCA(20)]          # Branch 1: no model → features only
]},
{"merge": "predictions"}  # ERROR: Branch 1 has no predictions
```

**Resolution options**:

1. **Selective merge**: Only request predictions from branches that have them
   ```python
   {"merge": {"predictions": [0], "features": [1]}}
   ```

2. **Mixed merge**: Combine features from some branches, predictions from others
   ```python
   {"merge": {
       "predictions": {"branches": [0]},
       "features": {"branches": [1]}
   }}
   ```

3. **Add models to all branches**: Restructure pipeline
   ```python
   {"branch": [
       [SNV(), PLS(10)],
       [MSC(), PCA(20), PLS(10)]  # Add model to Branch 1
   ]}
   ```

### 12.4 Asymmetric Model Counts

When branches have different numbers of models:

```python
{"branch": [
    [SNV(), PLS(10), RF(), XGB()],  # Branch 0: 3 models
    [MSC(), PLS(5)]                  # Branch 1: 1 model
]},
{"merge": "predictions"}  # 4 prediction features total
```

**Default behavior**: Each model contributes one prediction feature.

**Per-branch control** (see Section 9):
```python
{"merge": {
    "predictions": [
        {"branch": 0, "select": "best"},      # 1 feature (best of 3)
        {"branch": 1, "aggregate": "separate"} # 1 feature (only model)
    ]
}}
```

### 12.5 Asymmetric Output Scenarios

This section provides comprehensive examples for handling asymmetric branch outputs.

#### Scenario 1: Mixed Features + Predictions

**Setup**: Branch 0 has models, Branch 1 has only transforms.

```python
{"branch": [
    [SNV(), Detrend(), PLSRegression(10), RandomForestRegressor()],  # 2 models
    [MSC(), PCA(30)]                                                   # 0 models
]},
{"merge": {
    "predictions": [{"branch": 0, "models": "best", "metric": "rmse"}],  # 1 feature
    "features": [1]                                                       # 30 features
}}
# Result: 31 features (1 prediction + 30 PCA components)
```

#### Scenario 2: Selective Model Extraction

**Setup**: Multiple branches with varying model counts.

```python
{"branch": {
    "fast_path": [SNV(), PLSRegression(10)],           # 1 model
    "ensemble_path": [MSC(), PLSRegression(15), RF(), XGB(), SVR()],  # 4 models
    "feature_path": [PCA(50)]                           # 0 models
}},
{"merge": {
    "predictions": [
        {"branch": "fast_path", "models": "all"},                    # 1 feature
        {"branch": "ensemble_path", "models": {"top_k": 2}, "aggregate": "weighted_mean"}  # 1 feature
    ],
    "features": ["feature_path"]  # 50 features
}}
# Result: 52 features
```

#### Scenario 3: Predictions as Sources

**Setup**: Want to keep branch predictions as separate sources for multi-head downstream processing.

```python
{"branch": [
    [SNV(), PLSRegression(10)],
    [MSC(), RF()]
]},
{"merge": {
    "predictions": "all",
    "output_as": "sources"  # Creates source_branch_0, source_branch_1
}},
# Now subsequent steps see 2 sources with 1 feature each
MultiHeadNN(heads_per_source=True)
```

#### Scenario 4: Cross-Branch Stacking with Feature Augmentation

**Setup**: Some branches provide predictions, some provide features, combine for final model.

```python
{"branch": {
    "spectral_models": [SNV(), PLSRegression(5), PLSRegression(10), PLSRegression(15)],
    "ml_models": [MSC(), RF(), XGB()],
    "feature_extraction": [PCA(20), SelectKBest(10)]
}},
{"merge": {
    "predictions": [
        {"branch": "spectral_models", "models": "all", "aggregate": "mean"},  # 1 feature
        {"branch": "ml_models", "models": "best", "metric": "r2"}             # 1 feature
    ],
    "features": ["feature_extraction"],  # 10 features (SelectKBest output)
    "include_original": True             # + original features
}},
{"model": Ridge()}
# Result: Ridge trained on [mean_PLS + best_ML + 10_features + original]
```

#### Scenario 5: Asymmetric with Source Awareness

**Setup**: Multi-source dataset with branches that treat sources differently.

```python
# Dataset has 2 sources: NIR (500 features), markers (1000 features)
{"branch": [
    [SNV(), PLSRegression(10)],  # Trained on concatenated sources
    [SourceSelect("NIR"), PCA(20)]  # Only uses NIR source
]},
{"merge": {
    "predictions": [{"branch": 0}],
    "features": [{"branch": 1, "source_handling": "preserve"}]
}}
# Note: Branch 1 only has NIR features - this is intentional
```

#### Scenario 6: Hierarchical Stacking

**Setup**: First level stacking, then second level with mixed inputs.

```python
# First level: Create base predictions
{"branch": [
    [SNV(), PLSRegression(10), RF()],
    [MSC(), XGBRegressor()]
]},
{"merge": {
    "predictions": "all",
    "output_as": "features"  # 3 prediction features
}},

# Second level: Branch again for meta-model variants
{"branch": [
    [StandardScaler(), Ridge()],
    [MinMaxScaler(), ElasticNet()]
]},
{"merge": {"predictions": "all"}},

# Third level: Final meta-model
{"model": LassoCV()}
```

### 12.6 Shape Compatibility Rules

| Merge Type | Sample Count | Feature Count | Source Count | Processing Count |
|------------|--------------|---------------|--------------|------------------|
| Feature merge (2D) | Must match | Any (concatenated) | Must match | Any (flattened) |
| Feature merge (3D) | Must match | Any | Must match | Must match (or use strategy) |
| Prediction merge | Must match | N/A (scalars) | N/A | N/A |
| Mixed merge | Must match | Features: any | Must match (for features) | 2D: Any |

**Key Insight**: In 2D layout (the default), features are flattened and concatenated horizontally. Different feature dimensions across branches is **expected and normal** - each branch can have different preprocessing (e.g., different PCA components). No shape mismatch handling is needed.

### 12.7 Handling Shape Mismatches (3D Layout Only)

The `on_shape_mismatch` parameter only applies when using **3D layout** for features, where the number of processings must align across branches. In 2D layout (the default), this parameter has no effect.

**Example scenario**:
- Branch 0: (200 samples, 500 features) from MinMaxScaler (1 processing)
- Branch 1: (200 samples, 4 processings, 20 features) from multi-processing

| Layout | Result | Shape Mismatch? |
|--------|--------|------------------|
| 2D (default) | (200, 500 + 4×20 = 580) | No - just concatenate |
| 3D | Cannot align processings | Yes - needs strategy |

```python
# 2D layout (default) - no on_shape_mismatch needed
{"merge": "features"}  # Works with any feature dimensions

# 3D layout - may need on_shape_mismatch
{"merge": {
    "features": "all",
    "layout": "3d",  # Future feature
    "on_shape_mismatch": "error"  # Default: strict validation
}}
```

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| `"error"` (default) | Raise clear error with resolution options | 3D layout: strict validation |
| `"pad"` | Pad shorter branches with zeros to match longest | 3D layout: align processings |
| `"truncate"` | Truncate longer branches to match shortest | 3D layout: rare, use with caution |
| `"allow"` | Force 2D flattening and concatenate | 3D layout: fallback to 2D |

**Note**: In 2D layout (default), these strategies have no effect - features are simply concatenated regardless of dimensions. The strategies only apply to future 3D layout support.

### 12.8 Validation During Merge

`MergeController` performs these validations:

1. **Sample count validation**: All branches must have same sample count (always required)
2. **Source count validation**: All branches must have same source count (for feature merge)
3. **Processing count validation**: Only for 3D layout requests (not checked in 2D)
4. **Model availability validation**: For prediction merge, validate models exist
5. **Shape compatibility validation**: 3D layout only, based on `on_shape_mismatch` setting

**Important**: In 2D layout (default), feature dimensions are NOT validated because horizontal concatenation of different dimensions is the intended behavior.

---

## 13. Error Catalog and Resolution

This section provides a comprehensive catalog of errors that can occur during merge operations.

### 13.1 Shape Incompatibility Errors

| Error Code | Message | Cause | Resolution |
|------------|---------|-------|------------|
| **MERGE-E001** | Processing count mismatch in 3D | 3D layout with different processings | Use 2D layout (default) or `on_shape_mismatch` |
| **MERGE-E002** | Sample count mismatch | Branches have different sample counts (BUG) | Report as bug—branches share sample set |
| **MERGE-E003** | Source count mismatch | Branch dropped/added sources | Ensure all branches process all sources |
| **MERGE-E004** | Processing count mismatch in 3D | Different processing counts per branch | Use `layout="2d"` (flatten processings) |
| **MERGE-E005** | 3D concat incompatibility | Asymmetric shapes prevent 3D concat | Force 2D or use separate sources |

**Note**: In 2D layout (the default), different feature dimensions across branches is **normal and expected**. Features are simply concatenated horizontally. MERGE-E001 only applies to future 3D layout support.

**Example Error Message (MERGE-E001 - 3D layout only)**:
```
MergeError [MERGE-E001]: Cannot merge features in 3D layout - processing count mismatch.

Branches have different processing counts:
  - Branch 0 (SNV): 500 samples × 1 processing × 1500 features
  - Branch 1 (Multi-processing): 500 samples × 4 processings × 375 features

Cause: 3D layout requires aligned processing dimensions.

To resolve:
  1. Use 2D layout (default): {"merge": "features"} - flattens and concatenates
  2. Use {"merge": {"features": "all", "on_shape_mismatch": "pad"}} to pad processings
  3. Use {"merge": {"features": "all", "on_shape_mismatch": "allow"}} to force 2D
  4. Use selective merge: {"merge": {"features": [0]}} to use only one branch
```

### 13.2 Missing Data Errors

| Error Code | Message | Cause | Resolution |
|------------|---------|-------|------------|
| **MERGE-E010** | No predictions in branch | Requested predictions but branch has no model | Use feature merge or add model |
| **MERGE-E011** | Partial model coverage | Some branches have models, others don't | Use selective/mixed merge |
| **MERGE-E012** | OOF reconstruction failure | Fold misalignment in prediction store | Check splitter configuration |
| **MERGE-E013** | Model not found | Explicit model name doesn't exist | Verify model names |
| **MERGE-E014** | No features snapshot | Branch context missing features (BUG) | Report as internal error |

**Example Error Message (MERGE-E010)**:
```
MergeError [MERGE-E010]: Cannot merge predictions - no model found in branch.

Requested prediction merge from branch 1, but it contains no trained models.

Branch 1 pipeline: [MSC(), PCA(n_components=20)]
  → Contains only transformers, no model step.

To resolve:
  1. Add a model to branch 1: [MSC(), PCA(20), {"model": PLSRegression(10)}]
  2. Use feature merge instead: {"merge": "features"}
  3. Use mixed merge: {"merge": {"predictions": [0], "features": [1]}}
  4. Exclude branch 1 from prediction merge: {"merge": {"predictions": [0]}}
```

### 13.3 Configuration Errors

| Error Code | Message | Cause | Resolution |
|------------|---------|-------|------------|
| **MERGE-E020** | Merge outside branch mode | `{"merge": ...}` without prior branch | Add branch step or remove merge |
| **MERGE-E021** | Invalid branch index | Branch index exceeds count | Use valid indices (0 to N-1) |
| **MERGE-E022** | Invalid merge mode | Unknown merge type string | Use "features", "predictions", or "all" |
| **MERGE-E023** | Conflicting selection/aggregation | Incompatible per-branch config | Review configuration |
| **MERGE-E024** | merge_sources on single-source | Only one source in dataset | Remove merge_sources (not needed) |
| **MERGE-E025** | Unsafe without predictions | `unsafe=True` set but no predictions | Remove unsafe flag or add predictions |

**Example Error Message (MERGE-E020)**:
```
MergeError [MERGE-E020]: Cannot merge - not in branch mode.

A merge step was encountered without a prior branch step.

Pipeline context:
  - in_branch_mode: False
  - branch_contexts: []

To resolve:
  1. Add a branch step before merge:
     {"branch": [[SNV()], [MSC()]]},
     {"merge": "features"}
  2. Remove the merge step if branching is not intended
```

### 13.4 Multi-Source Specific Errors

| Error Code | Message | Cause | Resolution |
|------------|---------|-------|------------|
| **MERGE-E030** | Source processing incompatibility | Sources have different processing counts | Use `on_incompatible="flatten"` |
| **MERGE-E031** | Unknown source name | Source name not found in dataset | Check `dataset.source_names` |
| **MERGE-E032** | Source branch + branch conflict | Complex nesting | Simplify or use sequential |
| **MERGE-E033** | Source branch partial coverage | Not all sources specified | Use `"*": []` for unspecified |
| **MERGE-E034** | Merge sources 3D shape conflict | 3D concat with asymmetric processings | Force 2D layout |

**Example Error Message (MERGE-E030)**:
```
SourceMergeError [MERGE-E030]: Cannot merge sources in 3D layout.

Sources have different processing counts:
  - Source 0 (NIR): shape (500, 3, 500) - 3 processings
  - Source 1 (markers): shape (500, 1, 50000) - 1 processing

numpy cannot concatenate arrays with different axis 1 dimensions.

To resolve:
  1. Use flatten mode: {"merge_sources": {"strategy": "concat", "on_incompatible": "flatten"}}
  2. Apply preprocessing to equalize processing counts
  3. Use source_branch for per-source processing, then merge

Example:
  {"merge_sources": {"strategy": "concat", "on_incompatible": "flatten"}}
  # Results in 2D concat: (500, 3×500 + 1×50000) = (500, 51500)
```

### 13.5 Deadlock and Infinite Mode Scenarios

| Code | Scenario | Detection | Prevention |
|------|----------|-----------|------------|
| **DL-001** | Branch without merge | Pipeline ends in branch mode | Warn at pipeline end |
| **DL-002** | Nested branch without nested merge | Multiple branch levels, one merge | Require merge per branch level |
| **DL-003** | source_branch + branch interleaved | Complex nesting | Recommend sequential pattern |

**Detection**: The pipeline executor should check at pipeline end:
```python
if context.custom.get("in_branch_mode"):
    logger.warning(
        "Pipeline ended while still in branch mode. "
        "All subsequent runs will execute on all branches. "
        "Add {'merge': ...} to exit branch mode explicitly."
    )
```

---

## 14. Unified Merge Controller

This section describes the unified `MergeController` that handles all merge operations.

### 14.1 Controller Keywords

The unified controller handles multiple keywords:

| Keyword | Purpose | Exit Branch Mode? |
|---------|---------|-------------------|
| `merge` | Combine branch outputs (features/predictions) | ✅ Yes |
| `merge_sources` | Combine multi-source features | ❌ No |
| `merge_predictions` | Late fusion of predictions | Depends on context |

### 14.2 Controller Specification

```python
@register_controller
class MergeController(OperatorController):
    """Unified controller for all merge operations.

    This controller handles:
    1. Branch merging (`merge`): Exit branch mode, combine branch outputs
    2. Source merging (`merge_sources`): Combine multi-source features
    3. Prediction merging (`merge_predictions`): Late fusion for ensembles

    Multi-Source Behavior:
        When operating on multi-source datasets, branch merge operates on
        the 2D flattened view (sources already concatenated per branch).
        Use `merge_sources` to explicitly control source combination.

    Asymmetric Branch Handling:
        Validates shape compatibility before merging.
        Provides clear errors with resolution options.
        Supports `on_shape_mismatch` parameter for flexible handling.

    OOF Safety:
        Prediction merging uses OOF reconstruction by default.
        Set `unsafe=True` to disable (with prominent warnings).
    """

    priority = 5

    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        return keyword in ("merge", "merge_sources", "merge_predictions")

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return True

    def execute(self, step_info, dataset, context, runtime_context, ...):
        keyword = step_info.keyword

        if keyword == "merge":
            return self._execute_branch_merge(...)
        elif keyword == "merge_sources":
            return self._execute_source_merge(...)
        elif keyword == "merge_predictions":
            return self._execute_prediction_merge(...)
```

### 14.3 Extended Syntax

**Branch Merge** (`merge`):
```python
# Simple forms
{"merge": "features"}      # Merge features from all branches
{"merge": "predictions"}   # Merge OOF predictions from all branches
{"merge": "all"}           # Merge both features and predictions

# With multi-source awareness
{"merge": {
    "features": "all",
    "source_handling": "concat",      # "concat" (default) | "per_source"
    "on_shape_mismatch": "error"      # 3D layout only: "error" | "pad" | "truncate" | "allow"
}}

# Per-branch control (same as before)
{"merge": {
    "predictions": [
        {"branch": 0, "select": "best", "metric": "rmse"},
        {"branch": 1, "aggregate": "mean"}
    ]
}}
```

**Source Merge** (`merge_sources`):
```python
# Simple forms
{"merge_sources": "concat"}   # Horizontal concatenation
{"merge_sources": "stack"}    # Stack as 3D (with padding if needed)
{"merge_sources": "dict"}     # Keep as dict for multi-head models

# With selection and control
{"merge_sources": {
    "sources": ["NIR", "markers"],     # Which sources (by name or index)
    "strategy": "concat",
    "on_incompatible": "flatten"       # "error" | "flatten" | "pad"
}}
```

**Prediction Merge** (`merge_predictions`):
```python
# For late fusion (source or branch agnostic)
{"merge_predictions": "average"}         # Mean of all predictions
{"merge_predictions": "weighted_average"} # Weight by validation score
{"merge_predictions": "vote"}            # Majority vote (classification)
{"merge_predictions": "stack"}           # Stack as features for meta-model
```

### 14.4 Chaining Merge Operations

Complex pipelines may require multiple merge operations:

```python
pipeline = [
    # Source-level branching
    {"source_branch": {
        "NIR": [SNV(), SavitzkyGolay()],
        "markers": [VarianceThreshold()]
    }},
    {"merge_sources": "concat"},        # Step 1: Combine sources

    # Pipeline-level branching
    {"branch": [[PLS(10)], [RF()]]},
    {"merge": "predictions"},           # Step 2: Combine branch predictions

    # Meta-model
    {"model": Ridge()}
]
```

### 14.5 Validation Flow

The controller validates in this order:

1. **Context validation**: Check branch mode for `merge`, sources exist for `merge_sources`
2. **Configuration parsing**: Validate merge config structure
3. **Target validation**: Check specified branches/sources exist
4. **Shape validation**: Check compatibility based on merge type
5. **Data availability**: Check features/predictions exist
6. **Execute merge**: Perform the actual merge operation
7. **State update**: Update context (exit branch mode if applicable)

### 14.6 Relationship to Existing Controllers

| Controller | Responsibility | Interaction with MergeController |
|------------|----------------|----------------------------------|
| `BranchController` | Creates branch contexts, enters branch mode | MergeController exits branch mode |
| `TransformerController` | Applies transforms per (branch, source, processing) | Produces features that MergeController collects |
| `ModelController` | Trains models per branch | Produces predictions that MergeController collects |
| `MetaModelController` | Convenience for stacking | **Delegates to MergeController** for data prep |
| `ConcatTransformController` | Concat transforms in single path | Independent (no branch awareness) |

---

## 15. Implementation Roadmap

### 15.1 Guiding Principle

**Merge is the core primitive**. Implement it first with full OOF support, then refactor MetaModel to use it. Multi-source support is integrated throughout, with source merge as a separate phase.

This comprehensive roadmap supersedes the original Phase 1-9 plan by incorporating:
- Multi-source dataset support (Phases 9-10)
- Asymmetric branch handling (integrated into Phase 6)
- Source branching (Phase 10)
- Comprehensive error catalog (integrated throughout)

### 15.2 Phase Overview

| Phase | Status | Duration | Key Deliverable |
|-------|--------|----------|-----------------|
| **Phase 1**: Data Structures | ✅ Done | 3-4 days | Enums, configs, parsing (including multi-source options) |
| **Phase 2**: Controller Skeleton | ✅ Done | 2-3 days | Branch exit, unified controller matching |
| **Phase 3**: Feature Merge | ✅ Done | 2-3 days | `{"merge": "features"}` with shape validation |
| **Phase 4**: Prediction Simple | ✅ Done | 3-4 days | OOF, unsafe, simple syntax |
| **Phase 5**: Prediction Per-Branch | ✅ Done | 3-4 days | Selection + aggregation strategies |
| **Phase 6**: Mixed Merge + Asymmetric | ✅ Done | 2-3 days | Features + predictions, asymmetric handling |
| **Phase 7**: MetaModel Refactor | ✅ Done | 3-4 days | Backward compatibility, merge delegation |
| **Phase 8**: Prediction Mode | ✅ Done | 2-3 days | Train/predict cycle |
| **Phase 9**: Source Merge | ✅ Done | 3-4 days | `{"merge_sources": ...}` |
| **Phase 10**: Source Branching | ✅ Done | 4-5 days | `{"source_branch": ...}` |
| **Phase 11**: Documentation | ✅ Done | 2-3 days | Examples and comprehensive docs |

**Total Estimated Time**: 29-40 days
**Current Progress**: All 11 phases complete ✅

### 15.3 Phase Details

#### Phase 1: Data Structures & Config Parsing (3-4 days) ✅

**Goal**: Establish all data structures and config parsing logic

**Key Tasks**:
- [x] Create enum types: `MergeMode`, `SelectionStrategy`, `AggregationStrategy`, `ShapeMismatchStrategy`
- [x] Create `BranchPredictionConfig` dataclass with validation
- [x] Create `MergeConfig` with `source_handling` and `on_shape_mismatch` fields
- [x] Implement config parser for all syntax variants

**Multi-Source Additions**:
- [x] Add `source_handling` field: `"concat"` | `"per_source"` | `"dict"`
- [x] Add `on_shape_mismatch` field: `"error"` | `"pad"` | `"truncate"` | `"allow"`
- [x] Add `SourceMergeConfig` dataclass for `merge_sources` keyword

#### Phase 2: Controller Skeleton & Branch Exit (2-3 days) ✅

**Goal**: Basic unified controller that handles all merge keywords

**Key Tasks**:
- [x] Register controller for keywords: `"merge"`, `"merge_sources"`, `"merge_predictions"`
- [x] Implement keyword dispatch in `execute()` method
- [x] Implement branch validation with error codes (MERGE-E020, MERGE-E021)
- [x] Implement branch mode exit logic
- [x] Add `add_merged_features()` to SpectroDataset

#### Phase 3: Feature Merging with Shape Validation (2-3 days) ✅

**Goal**: Feature merge with comprehensive shape handling

**Key Tasks**:
- [x] Extract features from snapshot with source handling options
- [x] Validate shape compatibility (sample count always, feature count based on strategy)
- [x] Implement `on_shape_mismatch` strategies: error, pad, truncate, allow
- [x] Generate clear error messages (MERGE-E001, MERGE-E003)

#### Phase 4: Prediction Merging - Simple Mode (3-4 days) ✅

**Goal**: OOF prediction collection with safety guarantees

**Key Tasks**:
- [x] Model discovery from prediction store
- [x] OOF reconstruction via `TrainingSetReconstructor`
- [x] Unsafe mode with prominent warnings
- [x] Error handling for missing models (MERGE-E010, MERGE-E011)

#### Phase 5: Prediction Merging - Per-Branch Control (3-4 days) ✅

**Goal**: Advanced per-branch selection and aggregation

**Key Tasks**:
- [x] Model ranking by metric (rmse, mae, r2, accuracy, f1)
- [x] Selection strategies: all, best, top_k, explicit
- [x] Aggregation strategies: separate, mean, weighted_mean, proba_mean
- [x] Per-branch configuration parsing

#### Phase 6: Mixed Merge + Asymmetric Handling (2-3 days) ✅

**Goal**: Handle complex asymmetric branch scenarios

**Key Tasks**:
- [x] Mixed feature + prediction collection
- [x] Asymmetric branch detection and validation
- [x] Error handling for partial model coverage (MERGE-E010)
- [x] Different strategies per branch

**Asymmetric Scenarios Covered**:
- [x] Models in some branches, not others
- [x] Different feature dimensions per branch
- [x] Different model counts per branch

#### Phase 7: MetaModel Refactoring (3-4 days) ✅

**Goal**: Restore MetaModel as standalone operator using shared utilities

**Key Tasks**:
- [x] Extract `TrainingSetReconstructor` to shared module
- [x] Extract `ModelSelector` and `PredictionAggregator` to shared module
- [x] Update `MetaModelController` to use shared utilities
- [x] Update `MergeController` to use shared utilities
- [x] Remove any merge delegation logic from MetaModel
- [x] Ensure backward compatibility

**Integration Pattern**:
MetaModel and MergeController are siblings that share the same underlying utilities for OOF reconstruction and model selection.

**User-Facing Equivalences**:
MetaModel remains the high-level API for stacking, while Merge + Model provides a compositional alternative for branch combination.

#### Phase 8: Prediction Mode & Artifacts (2-3 days) ✅

**Goal**: Full train/predict cycle support

**Key Tasks**:
- [x] Save merge config to manifest
- [x] Load and apply in prediction mode
- [x] Bundle export support
- [x] Full cycle integration tests

#### Phase 9: Source Merge Implementation (3-4 days) ✅

**Goal**: Multi-source feature combination

**Key Tasks**:
- [x] Parse source selection (by name or index)
- [x] Implement merge strategies: concat, stack, dict
- [x] Handle incompatible shapes: error, flatten, pad, truncate
- [x] Source validation utilities on SpectroDataset
- [x] Implement merge_predictions for late fusion without branch context

**Implementation Details**:
- Added `SourceMergeConfig`, `SourceMergeStrategy`, `SourceIncompatibleStrategy` enums/dataclasses
- Implemented `_execute_source_merge()` with three strategies:
  - `concat`: 2D horizontal concatenation of all source features
  - `stack`: 3D stacking along source axis (requires uniform shapes)
  - `dict`: Keep sources as structured dictionary for multi-input models
- Implemented `_execute_prediction_merge()` for late fusion of predictions
- Single-source dataset handling: returns no-op with warning
- Error codes: MERGE-E024 (single source), MERGE-E030 (shape mismatch), MERGE-E031 (unknown source)
- 19 new unit tests covering config parsing, strategies, and error handling

#### Phase 10: Source Branching (4-5 days) ✅

**Goal**: Per-source pipeline execution

**Key Tasks**:
- [x] Create `SourceBranchConfig` dataclass in `operators/data/merge.py`
- [x] Create `SourceBranchController` in `controllers/data/source_branch.py`
- [x] Implement `SourceBranchConfigParser` for syntax parsing
- [x] Source isolation via processing chain filtering per source
- [x] Auto-merge option with concat/stack/dict strategies
- [x] Integration with `merge_sources` for manual merge
- [x] Prediction mode support with artifact provider
- [x] Unit tests in `tests/unit/controllers/data/test_source_branch.py` (27 tests)
- [x] Integration tests in `tests/integration/pipeline/test_source_branch.py` (9 tests)

**Implementation Notes**:
- Uses processing chain filtering to isolate sources during sub-pipeline execution
- Auto-merge collects features from all sources after processing
- Compatible with MetaModel stacking (requires DROP_INCOMPLETE coverage strategy)
- Error codes: SOURCEBRANCH-E001 (no sources), SOURCEBRANCH-E002 (single source warning)

#### Phase 11: Documentation & Examples (2-3 days) ✅

**Goal**: Comprehensive documentation and examples

**Key Tasks**:
- [x] Example files: Q_merge_branches.py, Q_merge_sources.py
- [x] Reference documentation for all merge syntax
- [x] Error catalog documentation
- [x] Update related docs and copilot instructions

**Implementation Details**:
- Created `examples/Q_merge_branches.py` with 8 comprehensive examples:
  - Basic feature merge
  - Prediction merge (stacking)
  - Mixed merge (features + predictions)
  - Per-branch model selection
  - Per-branch aggregation
  - Asymmetric branches
  - Merge vs MetaModel equivalence
  - Output targets
- Created `examples/Q_merge_sources.py` with 7 examples for multi-source datasets:
  - Basic source merge
  - Source branching (per-source pipelines)
  - Source branch with auto-merge
  - Source merge strategies (concat, stack, dict)
  - Combined source + pipeline branching
  - Source-aware stacking
  - Shape mismatch handling
- Created `docs/specifications/merge_syntax.md` with full syntax reference
- Updated `.github/copilot-instructions.md` with branching and merging section
- Added key file references for merge.py and source_branch.py

### 15.4 Dependency Graph

```
Phase 1 (Data Structures)
    ↓
Phase 2 (Skeleton) ──────────────────────────────────┐
    ├──────────────────┐                             │
    ↓                  ↓                             ↓
Phase 3 (Features)  Phase 4 (Predictions)        Phase 9 (Source Merge)
    │                  ↓                             ↓
    │              Phase 5 (Per-Branch)          Phase 10 (Source Branch)
    │                  │                             │
    └─────────┬────────┘                             │
              ↓                                      │
        Phase 6 (Mixed + Asymmetric)                 │
              │                                      │
              ├──────────────────────────────────────┤
              ↓                                      │
        Phase 7 (MetaModel)                          │
              ↓                                      │
        Phase 8 (Prediction Mode)                    │
              │                                      │
              └──────────────────────────────────────┤
                                                     ↓
                                              Phase 11 (Documentation)
```

### 15.5 Risk Mitigation

| Risk | Mitigation Strategy |
|------|---------------------|
| MetaModel backward compatibility | Dedicated Phase 7 testing with all existing examples |
| OOF edge cases | Reuse proven `TrainingSetReconstructor` |
| Unsafe mode misuse | Prominent warnings at config, parse, and execute time |
| Performance regression | Benchmark on Q18, Q_meta_stacking before/after |
| Complex config parsing | Extensive unit tests in Phase 1 |
| Per-branch complexity | Build on simple mode first (Phase 4 → 5) |
| Multi-source shape mismatches | Comprehensive error catalog with clear resolutions |
| Source branch complexity | Implement after merge_sources is proven stable |
| Asymmetric branch edge cases | Dedicated validation and testing in Phase 6 |

### 15.6 Testing Strategy

| Test Category | Phase | Coverage |
|---------------|-------|----------|
| Unit: Config parsing | 1 | All syntax variants, error cases |
| Unit: Validation | 2, 3 | Branch/source validation, shape checks |
| Integration: Feature merge | 3 | Single/multi source, shape mismatch handling |
| Integration: Prediction merge | 4, 5 | OOF, unsafe, per-branch strategies |
| Integration: Mixed merge | 6 | Asymmetric branches |
| Integration: MetaModel | 7 | Backward compatibility |
| E2E: Train/Predict | 8 | Full cycle with bundles |
| Integration: Source merge | 9 | Multi-source datasets |
| Integration: Source branch | 10 | Per-source pipelines |
| E2E: Examples | 11 | All example files run successfully |

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

**Design Integration (v6.0.0)**: This design now fully integrates the asymmetric sources concepts:
- `merge_sources` keyword is part of the unified MergeController
- `source_branch` is planned for Phase 10
- Multi-source shape handling is documented in Section 11
- Error catalog covers multi-source errors (MERGE-E030 to MERGE-E034)
- Branch vs Source distinction is explicitly documented in Section 2
- Prediction selection syntax is comprehensively documented in Section 9

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
| **Unified controller with keyword dispatch** | DRY principle, shared utilities | Separate controllers (code duplication) |
| **on_shape_mismatch parameter** | Flexible handling for advanced users | Strict error only (too rigid) |
| **Error catalog with codes** | Clear debugging, searchable resolutions | Generic error messages (confusing) |
| **Branches × Sources orthogonal** | Simpler mental model, matches current behavior | Cross-product (exponential complexity) |
| **Source merge operates post-flattening** | Consistent with 2D concat behavior | Per-source 3D merge (complex shapes) |
| **Source branch after merge_sources** | Build on stable foundation | Simultaneous implementation (risky) |
| **Branches NOT auto-sources** | Enable per-branch model training; sources = data origin | Auto-convert (loses training capability) |
| **`output_as` parameter for merge** | User controls output destination (features/sources/dict) | Single output format (inflexible) |
| **Explicit prediction selection syntax** | Full control over which predictions from which branches/models | Global selection only (limited stacking) |
| **Named branches for selection** | Clearer than numeric indices for complex pipelines | Index-only (error-prone) |
| **Model selection strategies** | all, best, top_k, explicit - covers all use cases | Single strategy (insufficient) |
| **Aggregation strategies per branch** | Different branches may need different aggregation | Global aggregation (inflexible) |
| **Hierarchical stacking support** | Enable multi-level stacking with merge → branch → merge | Single-level only (limiting) |
