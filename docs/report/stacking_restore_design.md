# Stacking Feature Restoration Design Document

**Version**: 1.4.0
**Status**: Phase 1, 2, 3, 4 & 5 Complete
**Date**: December 22, 2025
**Author**: Design Review

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Current State Analysis](#4-current-state-analysis)
5. [Original Stacking Behavior (To Restore)](#5-original-stacking-behavior-to-restore)
6. [Feature Input Modes for Stacking](#6-feature-input-modes-for-stacking)
7. [Model Selection Strategies](#7-model-selection-strategies)
8. [Proposition: Unified Architecture](#8-proposition-unified-architecture)
9. [Shared Components](#9-shared-components)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [Code Examples](#11-code-examples)
12. [Appendix: Key Files Reference](#appendix-key-files-reference)

---

## 1. Executive Summary

### The Misunderstanding

During the branching/concat/merge refactoring, the stacking functionality was incorrectly conflated with branch merging. The design documents (v3) positioned `MergeController` as the "core primitive" for stacking, suggesting that `MetaModel` should delegate to it.

**This was a misunderstanding.**

Stacking (via `MetaModel`) is fundamentally different from branch merging:

| Aspect | Stacking (MetaModel) | Branch Merge |
|--------|---------------------|--------------|
| **Scope** | All models in pipeline history | Models within specific branches |
| **Requires Branches?** | **NO** - works on flat pipelines | YES - requires active branch mode |
| **Context Modification** | **None** - reads from prediction_store | Exits branch mode, clears contexts |
| **Model Selection** | Any model from any previous step | Per-branch model selection |
| **Primary Use Case** | Ensemble stacking without branching | Combining parallel execution paths |

### The Core Issue

The Phase 7 additions to `MetaModelController` added logic to optionally delegate to `MergeController`:

```python
# Phase 7: Enable MergeController delegation for branch mode
use_merge_controller: bool = False  # Currently disabled but represents wrong direction
```

This implies stacking needs branch awareness, but **the original and correct behavior is that stacking works completely independently of branches**.

### What Needs to Be Restored

1. **MetaModel as standalone operator** - Works without any branch context
2. **Pipeline-wide model discovery** - Finds ALL models in prediction_store before current step
3. **No context modification** - MetaModel is just a model that uses different features (OOF predictions instead of raw X)
4. **Merge as optional enhancement** - When IN branch mode, stacking can use merge semantics, but this is NOT the default

---

## 2. Problem Statement

### Current Issues

1. **Q_complex_all_keywords.py fails** because it expects merge+model to work for stacking after branches, but the fundamental stacking behavior (without branches) was not preserved.

2. **Design confusion** between:
   - `MetaModel` (stacking: uses OOF predictions from previous models)
   - `merge` (branch exit: combines features/predictions from parallel branches)

3. **The current MetaModelController** has complex branch validation logic that should only apply when branches ARE involved, not as the default path.

### Root Cause

The design documents incorrectly stated:
> "Stacking is conceptually 'merge predictions + train a model'"

This is **only true when branches are involved**. For the common case (flat pipeline with multiple models), stacking is simply:
> "Collect OOF predictions from all previous models + train a model on those predictions"

No merging, no branch contexts, no context modification.

---

## 3. Objectives

### Primary Objectives

1. **Restore standalone stacking behavior**: `MetaModel` must work in pipelines without any branches, collecting OOF predictions from ALL models that ran before it.

2. **No context modification**: MetaModel should NOT modify the execution context. It reads from the prediction_store, builds meta-features, trains its model, and stores its own predictions. That's it.

3. **Preserve all old stacking features**:
   - Explicit source model selection (`source_models=["PLS", "RF"]`)
   - Automatic selection (`source_models="all"`)
   - OOF reconstruction with coverage strategies
   - Test aggregation strategies
   - Multi-level stacking support
   - Branch-aware stacking (CURRENT_ONLY, ALL_BRANCHES scopes)

### Secondary Objectives

4. **Avoid code redundancy**: Identify shared utilities between stacking and merging:
   - OOF reconstruction logic (`TrainingSetReconstructor`)
   - Model selection utilities
   - Prediction aggregation strategies

5. **Clear separation of concerns**:
   - `MetaModelController`: Stacking without branch mode assumptions
   - `MergeController`: Branch combination and exit
   - When BOTH are needed: User explicitly writes `{"merge": ...}` then `{"model": ...}`

---

## 4. Current State Analysis

### 4.1 MetaModelController (Current)

**Location**: [nirs4all/controllers/models/meta_model.py](nirs4all/controllers/models/meta_model.py)

**What's Working**:
- OOF reconstruction via `TrainingSetReconstructor` ✅
- Source model selection via selectors ✅
- Coverage strategies (STRICT, DROP_INCOMPLETE, IMPUTE_*) ✅
- Test aggregation (MEAN, WEIGHTED_MEAN, BEST_FOLD) ✅
- Multi-level stacking validation ✅
- Prediction mode support ✅

**What's Problematic**:
- `use_merge_controller` flag implies dependency on branches ❌
- `_should_use_merge_controller()` adds unnecessary complexity ❌
- `_reconstruct_with_merge_controller()` conflates two distinct operations ❌
- Branch validation runs even when no branches exist ⚠️

### 4.2 MergeController (Current)

**Location**: [nirs4all/controllers/data/merge.py](nirs4all/controllers/data/merge.py)

**What's Working**:
- Configuration parsing for all syntax variants ✅
- Branch validation ✅
- Feature collection from branch snapshots ✅
- Prediction collection with per-branch config ✅
- Model selection and aggregation utilities ✅
- OOF reconstruction delegation ✅
- Branch mode exit ✅

**What's Its Purpose**:
- **ONLY for branch combination** - requires active branch mode
- Exits branch mode and combines outputs
- Per-branch model selection (best, top_k, explicit)
- Per-branch aggregation (separate, mean, weighted_mean)

### 4.3 TrainingSetReconstructor (Shared Utility)

**Location**: [nirs4all/controllers/models/stacking/reconstructor.py](nirs4all/controllers/models/stacking/reconstructor.py)

**Purpose**: Core OOF reconstruction logic used by BOTH stacking and merge.

**Correctly Implemented**: This is a shared utility that should remain shared.

### 4.4 Key Distinction Lost

The design documents conflated:

```
WRONG THINKING:
  MetaModel = merge + model = branch merging + model training

CORRECT THINKING:
  MetaModel = collect OOF from prediction_store + train model
  merge = exit branch mode + combine branch outputs

  These CAN overlap (stacking after branches) but MetaModel
  does NOT require merge.
```

---

## 5. Original Stacking Behavior (To Restore)

### 5.1 Basic Stacking (No Branches)

The PRIMARY use case that must work:

```python
pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5),

    # Base models (all run, all produce OOF predictions)
    PLSRegression(n_components=5),
    RandomForestRegressor(n_estimators=50),
    XGBRegressor(n_estimators=100),

    # Meta-model: Uses OOF predictions from ALL 3 models above
    {"model": MetaModel(model=Ridge())},
]
```

**Behavior**:
1. Each base model runs on each fold, producing validation predictions
2. Predictions are stored in `prediction_store`
3. MetaModel step:
   - Queries prediction_store for all previous models
   - Reconstructs OOF training matrix (N_samples × N_models)
   - Aggregates test predictions for test matrix
   - Trains Ridge on OOF matrix
   - Predicts on test matrix
   - Stores its own predictions
4. Pipeline continues (or ends)

**NO branch contexts, NO context modification, NO merge step.**

### 5.2 Stacking with Explicit Selection

```python
pipeline = [
    # ... preprocessing ...
    KFold(n_splits=5),

    PLSRegression(n_components=5),
    RandomForestRegressor(n_estimators=50),
    XGBRegressor(n_estimators=100),
    SVR(kernel='rbf'),  # Will NOT be used

    # Only use PLS and RF, ignore XGB and SVR
    {"model": MetaModel(
        model=Ridge(),
        source_models=["PLSRegression", "RandomForestRegressor"]
    )},
]
```

### 5.3 Multi-Level Stacking (Without Branches)

```python
pipeline = [
    KFold(n_splits=5),

    # Level 0: Base models
    PLSRegression(n_components=5),
    RandomForestRegressor(n_estimators=50),

    # Level 1: First meta-model
    {"model": MetaModel(model=Ridge()), "name": "Meta_L1"},

    # Level 2: Stacks on Level 0 AND Level 1
    {"model": MetaModel(
        model=ElasticNet(),
        stacking_config=StackingConfig(allow_meta_sources=True)
    ), "name": "Meta_L2"},
]
```

### 5.4 Stacking WITH Branches (Uses Branch Context)

This is where `BranchScope` matters:

```python
pipeline = [
    KFold(n_splits=5),

    {"branch": [
        [SNV(), PLSRegression(n_components=5)],   # Branch 0
        [MSC(), RandomForestRegressor()],          # Branch 1
    ]},

    # Option A: Stack only from CURRENT branch
    {"model": MetaModel(
        model=Ridge(),
        stacking_config=StackingConfig(branch_scope=BranchScope.CURRENT_ONLY)
    )},  # Runs twice, once per branch

    # Option B: Stack from ALL branches (requires merge semantics)
    {"merge": "predictions"},  # Exit branch mode, combine OOF predictions
    {"model": Ridge()},        # Train on merged predictions
]
```

**Key Insight**: When branches are involved:
- `CURRENT_ONLY`: MetaModel runs per-branch (no merge needed)
- `ALL_BRANCHES`: User explicitly uses merge then model (or merge handles it)

---

## 6. Feature Input Modes for Stacking

### 6.1 Overview

MetaModel can construct its training features (X) from multiple sources, not just OOF predictions. This provides flexibility for different stacking strategies:

| Input Mode | Description | Use Case |
|------------|-------------|----------|
| `predictions_only` | Only OOF predictions (default) | Pure stacking ensemble |
| `predictions_and_features` | OOF predictions + current features | Stacking with passthrough |
| `features_only` | Only current context features | Regular model on transformed features |
| `branch_features` | Features from specific branches | Selective branch feature combination |
| `mixed` | Any combination of above | Advanced custom configurations |

### 6.2 Predictions Only (Default)

This is the classic stacking behavior:

```python
{"model": MetaModel(
    model=Ridge(),
    input_mode="predictions",  # Default, can be omitted
)}
```

The meta-features are:
```
X_meta = [OOF_model1 | OOF_model2 | ... | OOF_modelN]
# Shape: (n_samples, n_models)
```

### 6.3 Predictions and Features (Passthrough)

Include original features alongside predictions. This is similar to sklearn's `StackingRegressor(passthrough=True)`:

```python
{"model": MetaModel(
    model=Ridge(),
    input_mode="predictions_and_features",
    # Or explicit:
    include_features=True,  # Include current X context
)}
```

The meta-features are:
```
X_meta = [OOF_model1 | OOF_model2 | ... | OOF_modelN | X_current]
# Shape: (n_samples, n_models + n_features)
```

### 6.4 Branch Features Selection

When branches are in the pipeline, MetaModel can select features from specific branches:

```python
pipeline = [
    {"branch": [
        [SNV(), PCA(n_components=20)],      # Branch 0: 20 features
        [MSC(), PCA(n_components=30)],       # Branch 1: 30 features
        [FirstDerivative(), PLS(n_components=5)],  # Branch 2: Has model
    ]},

    # Stack predictions from Branch 2, add features from Branches 0 and 1
    {"model": MetaModel(
        model=Ridge(),
        source_predictions={
            "branches": [2],
            "select": "all",
        },
        source_features={
            "branches": [0, 1],  # Get features from these branches
        },
    )},
]
```

The meta-features are:
```
X_meta = [OOF_from_branch2 | Features_branch0 | Features_branch1]
# Shape: (n_samples, n_models_branch2 + 20 + 30)
```

### 6.5 Feature Input Configuration

The `source_features` parameter controls feature inclusion:

```python
@dataclass
class FeatureInputConfig:
    """Configuration for feature inputs in MetaModel.

    Attributes:
        include: Whether to include features at all.
        source: Where to get features from:
            - "current": Current feature context (post-preprocessing)
            - "original": Original X before any transforms
            - "branches": Specific branch feature snapshots
        branches: When source="branches", which branch indices to use.
            - "all": All branches
            - [0, 2]: Specific branch indices
            - "current": Only current branch (when in branch mode)
        on_missing: How to handle missing branch features:
            - "error": Raise ValueError
            - "warn": Log warning and skip
            - "skip": Silent skip
    """
    include: bool = False
    source: str = "current"  # "current", "original", "branches"
    branches: Union[str, List[int]] = "all"
    on_missing: str = "error"
```

### 6.6 Complete Input Mode Examples

```python
# Example 1: Classic stacking (predictions only)
{"model": MetaModel(model=Ridge())}

# Example 2: Stacking with passthrough (current features)
{"model": MetaModel(
    model=Ridge(),
    source_features={"include": True, "source": "current"}
)}

# Example 3: Stacking with branch features (no current features)
{"model": MetaModel(
    model=Ridge(),
    source_features={"include": True, "source": "branches", "branches": [0, 1]}
)}

# Example 4: Predictions from specific branches + features from others
{"model": MetaModel(
    model=Ridge(),
    source_predictions={"branches": [2], "select": "best"},
    source_features={"include": True, "source": "branches", "branches": [0, 1]}
)}

# Example 5: All predictions + all branch features
{"model": MetaModel(
    model=Ridge(),
    source_predictions="all",
    source_features={"include": True, "source": "branches", "branches": "all"}
)}
```

---

## 7. Model Selection Strategies

### 7.1 Overview

Model selection determines WHICH models' predictions are used as meta-features. This is crucial when:
- There are many base models and you want only the best ones
- Different branches have different models and you want fine-grained control
- You want to compare different selection strategies

### 7.2 Selection Strategies

| Strategy | Description | Parameter |
|----------|-------------|-----------|
| `all` | Use all available models | `select="all"` |
| `best` | Use single best model by metric | `select="best"` |
| `top_k` | Use top K models by metric | `select={"top_k": 3}` |
| `explicit` | Use explicitly named models | `select=["PLS", "RF"]` |
| `regex` | Use models matching pattern | `select={"regex": "PLS_.*"}` |
| `threshold` | Use models above/below metric threshold | `select={"threshold": 0.9, "metric": "r2"}` |

### 7.3 Global Selection (No Branches)

When no branches are involved, selection applies to ALL models in the prediction store:

```python
# Use all models (default)
{"model": MetaModel(model=Ridge(), source_models="all")}

# Use explicit list
{"model": MetaModel(model=Ridge(), source_models=["PLS_5", "RF", "XGB"])}

# Use best 3 models by RMSE
{"model": MetaModel(
    model=Ridge(),
    source_models={"select": {"top_k": 3}, "metric": "rmse"}
)}

# Use best single model
{"model": MetaModel(
    model=Ridge(),
    source_models={"select": "best", "metric": "r2"}
)}
```

### 7.4 Per-Branch Selection (With Branches)

When branches are involved, selection can be configured PER BRANCH:

```python
pipeline = [
    KFold(n_splits=5),

    {"branch": {
        "pls_variants": [
            SNV(),
            {"model": PLSRegression(n_components=3), "name": "PLS_3"},
            {"model": PLSRegression(n_components=5), "name": "PLS_5"},
            {"model": PLSRegression(n_components=10), "name": "PLS_10"},
        ],
        "tree_models": [
            MSC(),
            {"model": RandomForestRegressor(), "name": "RF"},
            {"model": GradientBoostingRegressor(), "name": "GBR"},
            {"model": XGBRegressor(), "name": "XGB"},
        ],
    }},

    # Per-branch selection strategy
    {"model": MetaModel(
        model=Ridge(),
        source_predictions=[
            # Branch 0 (pls_variants): Use best PLS by R²
            {"branch": 0, "select": "best", "metric": "r2"},
            # Branch 1 (tree_models): Use top 2 by RMSE
            {"branch": 1, "select": {"top_k": 2}, "metric": "rmse"},
        ],
    )},
]
```

### 7.5 Per-Branch Selection Configuration

```python
@dataclass
class BranchModelSelectionConfig:
    """Per-branch model selection configuration.

    Attributes:
        branch: Branch index (int) or name (str).
        select: Selection strategy:
            - "all": All models in this branch
            - "best": Single best model
            - {"top_k": N}: Top N models
            - ["name1", "name2"]: Explicit model names
            - {"regex": "pattern"}: Regex matching
            - {"threshold": value, "op": ">="}: Threshold filtering
        metric: Metric for ranking (rmse, r2, mae, accuracy, etc.).
            Default: "rmse" for regression, "accuracy" for classification.
        partition: Which partition's score to use for ranking.
            Default: "val"
        aggregate: How to combine selected predictions:
            - "separate": Each model is a separate feature (default)
            - "mean": Average predictions into single feature
            - "weighted_mean": Weighted average by metric score
        weight_metric: Metric for weighted_mean (defaults to same as metric).
    """
    branch: Union[int, str]
    select: Union[str, Dict, List[str]] = "all"
    metric: Optional[str] = None
    partition: str = "val"
    aggregate: str = "separate"
    weight_metric: Optional[str] = None
```

### 7.6 Selection Strategy Details

#### 7.6.1 `all` Strategy

Uses all models from the specified scope:

```python
# Global: All models in prediction_store
source_models="all"

# Per-branch: All models from branch 0
source_predictions=[{"branch": 0, "select": "all"}]
```

#### 7.6.2 `best` Strategy

Uses single best model by validation metric:

```python
# Global: Best model overall
source_models={"select": "best", "metric": "rmse"}

# Per-branch: Best from each specified branch
source_predictions=[
    {"branch": 0, "select": "best", "metric": "rmse"},
    {"branch": 1, "select": "best", "metric": "r2"},
]
```

**Metric direction is automatic**: RMSE/MAE lower is better, R²/accuracy higher is better.

#### 7.6.3 `top_k` Strategy

Uses top K models by validation metric:

```python
# Top 3 overall
source_models={"select": {"top_k": 3}, "metric": "rmse"}

# Top 2 from each branch
source_predictions=[
    {"branch": 0, "select": {"top_k": 2}, "metric": "rmse"},
    {"branch": 1, "select": {"top_k": 2}, "metric": "rmse"},
]
```

#### 7.6.4 `explicit` Strategy

Uses explicitly named models:

```python
# Global explicit selection
source_models=["PLS_5", "RF", "XGB"]

# Per-branch explicit selection
source_predictions=[
    {"branch": 0, "select": ["PLS_5", "PLS_10"]},
    {"branch": 1, "select": ["RF", "XGB"]},  # Exclude GBR
]
```

#### 7.6.5 `regex` Strategy

Uses models matching a regex pattern:

```python
# All PLS variants
source_models={"select": {"regex": "PLS_.*"}}

# Per-branch regex
source_predictions=[
    {"branch": 0, "select": {"regex": "PLS_[0-9]+"}},  # PLS_3, PLS_5, PLS_10
    {"branch": 1, "select": {"regex": ".*Boost.*"}},   # GradientBoosting, XGBoost
]
```

#### 7.6.6 `threshold` Strategy

Uses models meeting a metric threshold:

```python
# Models with R² > 0.8
source_models={"select": {"threshold": 0.8, "metric": "r2", "op": ">="}}

# Models with RMSE < 5.0
source_models={"select": {"threshold": 5.0, "metric": "rmse", "op": "<"}}
```

### 7.7 Aggregation Within Selection

When multiple models are selected, they can be aggregated:

```python
# Default: Each model = 1 feature (3 models → 3 features)
source_predictions=[{"branch": 0, "select": {"top_k": 3}, "aggregate": "separate"}]

# Mean: Averaged into 1 feature (3 models → 1 feature)
source_predictions=[{"branch": 0, "select": {"top_k": 3}, "aggregate": "mean"}]

# Weighted mean: Weighted by metric score (3 models → 1 feature)
source_predictions=[{
    "branch": 0,
    "select": {"top_k": 3},
    "aggregate": "weighted_mean",
    "weight_metric": "r2"  # Higher R² = higher weight
}]
```

### 7.8 Complete Selection Examples

```python
# Example 1: Simple - all models, separate features
{"model": MetaModel(model=Ridge())}

# Example 2: Best from each of 3 branches
{"model": MetaModel(
    model=Ridge(),
    source_predictions=[
        {"branch": 0, "select": "best", "metric": "rmse"},
        {"branch": 1, "select": "best", "metric": "rmse"},
        {"branch": 2, "select": "best", "metric": "rmse"},
    ]
)}

# Example 3: Top 2 from branch 0, all from branch 1, with aggregation
{"model": MetaModel(
    model=Ridge(),
    source_predictions=[
        {"branch": 0, "select": {"top_k": 2}, "aggregate": "mean"},
        {"branch": 1, "select": "all", "aggregate": "separate"},
    ]
)}

# Example 4: Threshold selection globally
{"model": MetaModel(
    model=Ridge(),
    source_models={"select": {"threshold": 0.7, "metric": "r2", "op": ">="}}
)}

# Example 5: Mixed - explicit from some, best from others
{"model": MetaModel(
    model=Ridge(),
    source_predictions=[
        {"branch": "pls_branch", "select": ["PLS_5"]},  # Explicit by name
        {"branch": "tree_branch", "select": "best", "metric": "rmse"},
    ]
)}
```

---

## 8. Proposition: Unified Architecture

### 8.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Pipeline Execution                                    │
│                                                                              │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐        │
│  │  Base Model 1   │────▶│  Base Model 2   │────▶│  Base Model 3   │        │
│  │  (stores OOF)   │     │  (stores OOF)   │     │  (stores OOF)   │        │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘        │
│           │                      │                      │                    │
│           ▼                      ▼                      ▼                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      PREDICTION STORE                                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │   │
│  │  │ Model1 OOF   │  │ Model2 OOF   │  │ Model3 OOF   │                │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    │ query                                   │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    MetaModelController                                │   │
│  │                                                                       │   │
│  │   1. Query prediction_store for previous models                      │   │
│  │   2. Use TrainingSetReconstructor for OOF reconstruction             │   │
│  │   3. Train meta-learner on OOF features                              │   │
│  │   4. Store predictions (does NOT modify context)                     │   │
│  │                                                                       │   │
│  │   [NO BRANCH AWARENESS REQUIRED FOR BASIC CASE]                      │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────┐                                                        │
│  │  Next Step...   │                                                        │
│  └─────────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────────┘


SEPARATE FLOW FOR BRANCHES:

┌─────────────────────────────────────────────────────────────────────────────┐
│                     Branch Mode Pipeline                                     │
│                                                                              │
│  ┌─────────────────┐                                                        │
│  │  {"branch": ..} │  ─────▶  Enters branch mode (N contexts)               │
│  └─────────────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Per-Branch Execution                              │    │
│  │   Branch 0: SNV() → PLS() → (predictions stored with branch_id=0)   │    │
│  │   Branch 1: MSC() → RF()  → (predictions stored with branch_id=1)   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │  {"merge": ..}  │  ─────▶  MergeController                               │
│  └─────────────────┘           │                                            │
│                                │  1. Collect features/predictions           │
│                                │  2. Perform per-branch selection           │
│                                │  3. Handle OOF for predictions              │
│                                │  4. EXIT branch mode                        │
│                                │  5. Store merged features in dataset        │
│                                ▼                                             │
│  ┌─────────────────┐                                                        │
│  │  {"model": ..}  │  ─────▶  Regular model training on merged features     │
│  └─────────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Key Design Decisions

1. **MetaModel is NOT a merge operation**
   - It reads from prediction_store
   - It does NOT modify execution context
   - It does NOT require branch awareness for basic operation
   - It is simply a model that uses different features (OOF predictions)

2. **MergeController is for branch exit**
   - REQUIRES active branch mode
   - ALWAYS exits branch mode
   - Combines features/predictions from branches
   - Stores merged result for subsequent steps

3. **When Both Are Needed**
   - User explicitly writes both:
     ```python
     {"merge": "predictions"},  # Exit branches, combine
     {"model": Ridge()}         # Train on combined
     ```
   - OR uses MetaModel with ALL_BRANCHES scope (handled internally)

4. **Branch-Aware Stacking (Optional)**
   - `CURRENT_ONLY`: MetaModel runs per-branch, no special handling
   - `ALL_BRANCHES`: MetaModel internally collects from all branches
   - This does NOT mean MetaModel becomes a merge operation

---

## 9. Shared Components

### 9.1 Components That SHOULD Be Shared

| Component | Purpose | Used By |
|-----------|---------|---------|
| `TrainingSetReconstructor` | OOF prediction reconstruction | MetaModel, Merge |
| `ModelSelector` | Model ranking by validation score | MetaModel, Merge |
| `PredictionAggregator` | Aggregation strategies (mean, weighted_mean) | MetaModel, Merge |
| `StackingConfig` | Coverage and aggregation settings | MetaModel |
| `BranchPredictionConfig` | Per-branch selection config | Merge |

### 9.2 Component Organization

```
nirs4all/controllers/
├── shared/                     # Shared utilities (Phase 2)
│   ├── __init__.py             # Exports ModelSelector, PredictionAggregator
│   ├── model_selector.py       # ModelSelector (SHARED)
│   └── prediction_aggregator.py # PredictionAggregator (SHARED)
│
├── models/
│   ├── meta_model.py           # MetaModelController (stacking)
│   └── stacking/
│       ├── __init__.py         # Exports
│       ├── reconstructor.py    # TrainingSetReconstructor (SHARED)
│       ├── config.py           # ReconstructorConfig
│       ├── multilevel.py       # MultiLevelValidator
│       ├── crossbranch.py      # CrossBranchValidator
│       ├── serialization.py    # MetaModelSerializer
│       └── exceptions.py       # Stacking-specific exceptions
│
├── data/
│   ├── merge.py                # MergeController (imports from shared/)
│   └── merge/
│       ├── __init__.py
│       ├── config.py           # MergeConfigParser
│       └── asymmetric.py       # AsymmetricBranchAnalyzer (merge-specific)
```

### 9.3 Shared Utilities ✅ EXTRACTED

`ModelSelector` and `PredictionAggregator` have been extracted from `merge.py` to the shared location:

**Location**: `nirs4all/controllers/shared/`

**Components**:
1. **ModelSelector** (`model_selector.py`): Model ranking and selection by validation metrics
   - `select_models()`: Per-branch model selection with BranchPredictionConfig
   - `select_models_global()`: Global model selection for non-branch pipelines
   - `get_model_scores()`: Get validation scores for weighted aggregation
   - Supports: ALL, BEST, TOP_K, EXPLICIT selection strategies

2. **PredictionAggregator** (`prediction_aggregator.py`): Prediction aggregation strategies
   - `aggregate()`: Aggregate predictions with various strategies
   - `aggregate_folds()`: Aggregate predictions across CV folds
   - Supports: SEPARATE, MEAN, WEIGHTED_MEAN, PROBA_MEAN strategies

**Usage**:
```python
from nirs4all.controllers.shared import ModelSelector, PredictionAggregator

# Model selection
selector = ModelSelector(prediction_store, context)
models = selector.select_models(available, config, branch_id=0)

# Prediction aggregation
aggregated = PredictionAggregator.aggregate(
    predictions={"PLS": pls_preds, "RF": rf_preds},
    strategy=AggregationStrategy.MEAN,
)
```

**Backward Compatibility**: Imports from `merge.py` still work via re-export.

---

## 10. Implementation Roadmap

### Phase 1: Restore Standalone Stacking (Priority: HIGH) ✅ COMPLETED

**Goal**: Make MetaModel work without any branch awareness in the default case.

**Status**: ✅ Completed on December 22, 2025

**Completed Tasks**:
1. ✅ Removed `use_merge_controller` class attribute from MetaModelController
2. ✅ Removed `_should_use_merge_controller()` method
3. ✅ Removed `_reconstruct_with_merge_controller()` method
4. ✅ Updated docstrings to reflect stacking restoration design principles
5. ✅ Verified `_reconstruct_with_reconstructor()` works without branch context
6. ✅ Tested with Q_meta_stacking.py examples - all pass
7. ✅ All unit tests pass (196 tests in stacking modules)
8. ✅ All integration tests pass (45 tests for stacking/branching)

**Acceptance Criteria (all met)**:
- ✅ Basic stacking pipeline works without branches
- ✅ No dependency on MergeController for default path
- ✅ All existing stacking tests pass

### Phase 2: Refactor Shared Utilities (Priority: MEDIUM) ✅ COMPLETED

**Goal**: Extract and share utilities between MetaModel and Merge.

**Status**: ✅ Completed on December 22, 2025

**Completed Tasks**:
1. ✅ Created `nirs4all/controllers/shared/` module with `__init__.py`
2. ✅ Moved `ModelSelector` to `nirs4all/controllers/shared/model_selector.py`
3. ✅ Moved `PredictionAggregator` to `nirs4all/controllers/shared/prediction_aggregator.py`
4. ✅ Updated `merge.py` to import from shared module (backward compatible)
5. ✅ Added `select_models_global()` method to ModelSelector for non-branch stacking
6. ✅ Added `aggregate_folds()` method to PredictionAggregator for CV fold aggregation
7. ✅ All 169 stacking unit tests pass
8. ✅ All 632 merge/stacking related tests pass
9. ✅ Q_meta_stacking.py example runs successfully

**Acceptance Criteria (all met)**:
- ✅ Single implementation of model selection in shared module
- ✅ Single implementation of prediction aggregation in shared module
- ✅ Both controllers import from shared location
- ✅ Backward compatibility maintained via re-export in merge.py

### Phase 3: Branch-Aware Stacking (Priority: MEDIUM) ✅ COMPLETED

**Goal**: Ensure MetaModel works correctly when branches ARE involved.

**Status**: ✅ Completed on December 22, 2025

**Completed Tasks**:
1. ✅ `CURRENT_ONLY` scope: Filter predictions by current branch_id (already working via selector)
2. ✅ `ALL_BRANCHES` scope: Collect from all branches with proper alignment
   - Modified `_get_source_models()` to respect BranchScope configuration
   - Added `_create_all_branches_context()` helper for cross-branch selection
   - Updated `TrainingSetReconstructor.reconstruct()` to not filter by branch_id when ALL_BRANCHES
   - Updated `FoldAlignmentValidator.validate()` with `branch_id_override` parameter
3. ✅ Branch validation only runs when in branch mode (existing guard for BranchType.NONE)
4. ✅ Fixed cross-branch error handling (`_raise_cross_branch_error()` method)
5. ✅ Fixed CrossBranchValidator to handle empty sample_indices gracefully
6. ✅ All 169 stacking unit tests pass
7. ✅ All 15 meta stacking integration tests pass
8. ✅ Q_meta_stacking.py example runs successfully

**Acceptance Criteria (all met)**:
- ✅ Stacking inside branches works (CURRENT_ONLY scope)
- ✅ Cross-branch stacking works (ALL_BRANCHES scope)
- ✅ No regression in stacking-related tests

### Phase 4: Fix Q_complex_all_keywords.py (Priority: HIGH) ✅ COMPLETE

**Goal**: Make the complex example work correctly.

**Summary**: The issue was that sample augmentation creates additional samples (e.g., 390 from 130 original) but OOF predictions only exist for original sample IDs. The merge controller and reconstructor were using `include_augmented=True` when collecting sample indices, which returned 390 IDs but only 130-189 had predictions.

**Fixes Applied**:
1. **reconstructor.py**: Changed `_get_sample_indices()` to use `include_augmented=False` for training samples
2. **reconstructor.py**: Added logic in `reconstruct()` to re-fetch y values when lengths don't match
3. **reconstructor.py**: Added `_get_y_values_for_samples()` helper method
4. **meta_model.py**: Added logic to match y_train_unscaled with result.y_train length
5. **meta_model.py**: Added `_get_unscaled_y_for_samples()` helper method
6. **merge.py**: In 2 locations for prediction collection, changed to use `include_augmented=False` for training indices
7. **merge.py**: Added augmentation propagation logic to copy predictions from base samples to their augmented versions

**Tasks**:
1. ✅ Verify merge+model pattern works for branch stacking
2. ✅ Verify MetaModel standalone works for non-branch stacking
3. ✅ Update example comments to clarify intent (MetaModel before merge tests branch-aware stacking)
4. ✅ Example serves as integration test

**Acceptance Criteria**:
- ✅ Q_complex_all_keywords.py runs without errors
- ✅ Output matches expected behavior (all keywords exercised successfully)

### Phase 5: Documentation Update (Priority: MEDIUM) ✅ COMPLETED

**Goal**: Correct the design documents to reflect proper architecture.

**Status**: ✅ Completed on December 22, 2025

**Completed Tasks**:
1. ✅ Updated branching_concat_merge_design_v3.md
2. ✅ Clarified MetaModel vs Merge distinction (standalone vs branch exit)
3. ✅ Updated architecture diagrams
4. ✅ Updated implementation roadmap to reflect shared utilities approach

**Acceptance Criteria**:
- ✅ Design documents accurately reflect the implemented architecture
- ✅ No confusion between MetaModel and Merge roles

---

## 11. Code Examples

### 11.1 Basic Stacking (No Branches) - MUST WORK

```python
from nirs4all.pipeline import PipelineRunner
from nirs4all.data import DatasetConfigs
from nirs4all.operators.models import MetaModel
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5, shuffle=True, random_state=42),

    # Base models
    PLSRegression(n_components=5),
    RandomForestRegressor(n_estimators=50, random_state=42),

    # Meta-learner: uses OOF from both models above
    {"model": MetaModel(model=Ridge())},
]

runner = PipelineRunner()
predictions, _ = runner.run(pipeline, DatasetConfigs("sample_data/regression"))

# Should have predictions for:
# - PLSRegression (5 folds)
# - RandomForestRegressor (5 folds)
# - MetaModel_Ridge (5 folds)
```

### 11.2 Stacking After Branches - With Merge

```python
from nirs4all.operators.transforms import SNV, MSC

pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5),

    # Create parallel branches
    {"branch": [
        [SNV(), PLSRegression(n_components=5)],
        [MSC(), RandomForestRegressor(n_estimators=50)],
    ]},

    # Merge predictions from branches and train meta-model
    {"merge": "predictions"},  # Collect OOF, exit branch mode
    {"model": Ridge()},        # Train on merged predictions
]
```

### 11.3 Stacking Inside Branches - No Merge

```python
pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5),

    {"branch": [
        [
            SNV(),
            PLSRegression(n_components=5),
            RandomForestRegressor(n_estimators=50),
            {"model": MetaModel(model=Ridge())},  # Stacks within Branch 0
        ],
        [
            MSC(),
            PLSRegression(n_components=10),
            {"model": MetaModel(model=ElasticNet())},  # Stacks within Branch 1
        ],
    ]},

    # Now merge the BRANCH outputs (features or predictions)
    {"merge": "predictions"},
    {"model": Ridge()},  # Final meta-model
]
```

### 11.4 Stacking with Feature Passthrough

```python
pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5),

    PLSRegression(n_components=5),
    RandomForestRegressor(n_estimators=50),

    # Meta-model with OOF predictions + original features
    {"model": MetaModel(
        model=Ridge(),
        source_features={"include": True, "source": "current"}
    )},
]
# X_meta = [OOF_PLS | OOF_RF | X_current]
```

### 11.5 Per-Branch Model Selection

```python
pipeline = [
    KFold(n_splits=5),

    {"branch": {
        "pls_variants": [
            SNV(),
            {"model": PLSRegression(n_components=3), "name": "PLS_3"},
            {"model": PLSRegression(n_components=5), "name": "PLS_5"},
            {"model": PLSRegression(n_components=10), "name": "PLS_10"},
            {"model": PLSRegression(n_components=15), "name": "PLS_15"},
        ],
        "tree_ensemble": [
            MSC(),
            {"model": RandomForestRegressor(), "name": "RF"},
            {"model": GradientBoostingRegressor(), "name": "GBR"},
        ],
    }},

    # Selective stacking:
    # - Branch 0: Use only best PLS by R²
    # - Branch 1: Use top 2 trees by RMSE, aggregated as mean
    {"model": MetaModel(
        model=Ridge(),
        source_predictions=[
            {"branch": 0, "select": "best", "metric": "r2"},
            {"branch": 1, "select": {"top_k": 2}, "metric": "rmse", "aggregate": "mean"},
        ],
    )},
]
# X_meta = [OOF_best_PLS | mean(OOF_top2_trees)]
# Shape: (n_samples, 2)
```

### 11.6 Mixed Features and Predictions from Branches

```python
pipeline = [
    KFold(n_splits=5),

    {"branch": [
        # Branch 0: Feature extraction only (no model)
        [SNV(), PCA(n_components=20)],

        # Branch 1: Feature extraction only
        [MSC(), PCA(n_components=30)],

        # Branch 2: Models for predictions
        [FirstDerivative(), PLSRegression(n_components=5), RandomForestRegressor()],
    ]},

    # Mixed: Features from branches 0,1 + Predictions from branch 2
    {"model": MetaModel(
        model=Ridge(),
        source_predictions=[{"branch": 2, "select": "all"}],
        source_features={
            "include": True,
            "source": "branches",
            "branches": [0, 1]
        },
    )},
]
# X_meta = [OOF_PLS | OOF_RF | PCA_features_branch0 | PCA_features_branch1]
# Shape: (n_samples, 2 + 20 + 30) = (n_samples, 52)
```

### 11.7 Threshold-Based Model Selection

```python
pipeline = [
    KFold(n_splits=5),

    # Many models with varying quality
    {"model": PLSRegression(n_components=3), "name": "PLS_3"},
    {"model": PLSRegression(n_components=5), "name": "PLS_5"},
    {"model": PLSRegression(n_components=10), "name": "PLS_10"},
    {"model": RandomForestRegressor(n_estimators=10), "name": "RF_small"},
    {"model": RandomForestRegressor(n_estimators=100), "name": "RF_large"},

    # Only use models with R² > 0.7
    {"model": MetaModel(
        model=Ridge(),
        source_models={
            "select": {"threshold": 0.7, "metric": "r2", "op": ">="}
        }
    )},
]
# Only models meeting the threshold contribute to X_meta
```

---

## Appendix: Key Files Reference

| File | Purpose | Action Needed |
|------|---------|---------------|
| [nirs4all/controllers/models/meta_model.py](nirs4all/controllers/models/meta_model.py) | MetaModelController | Remove merge delegation, restore standalone behavior |
| [nirs4all/controllers/data/merge.py](nirs4all/controllers/data/merge.py) | MergeController | Keep as-is for branch merging |
| [nirs4all/controllers/models/stacking/reconstructor.py](nirs4all/controllers/models/stacking/reconstructor.py) | OOF reconstruction | Shared utility - keep |
| [nirs4all/operators/models/meta.py](nirs4all/operators/models/meta.py) | MetaModel operator | No changes needed |
| [nirs4all/operators/data/merge.py](nirs4all/operators/data/merge.py) | Merge config classes | No changes needed |
| [examples/Q_meta_stacking.py](examples/Q_meta_stacking.py) | Stacking examples | Use as test reference |
| [examples/Q_complex_all_keywords.py](examples/Q_complex_all_keywords.py) | Complex pipeline | Fix to work correctly |

---

## Summary

The core insight is:

> **MetaModel and Merge are orthogonal operations that can be composed but should not be conflated.**

- **MetaModel** = Use OOF predictions (and optionally features) as inputs for training
- **Merge** = Exit branch mode and combine branch outputs

### Key Capabilities to Restore/Add

| Capability | Description |
|------------|-------------|
| **Standalone stacking** | Works without branches, uses all previous models |
| **Feature passthrough** | Include current features alongside OOF predictions |
| **Branch feature selection** | Select features from specific branches |
| **Global model selection** | all, best, top_k, explicit, regex, threshold |
| **Per-branch model selection** | Different selection strategy per branch |
| **Per-branch aggregation** | separate, mean, weighted_mean per branch |
| **Mixed inputs** | Predictions from some branches, features from others |

### When to Use What

| Scenario | Use |
|----------|-----|
| Flat pipeline, stack all models | `MetaModel(model=Ridge())` |
| Flat pipeline, stack specific models | `MetaModel(model=Ridge(), source_models=["PLS", "RF"])` |
| Flat pipeline, include original features | `MetaModel(model=Ridge(), source_features={"include": True})` |
| Branches, stack within each | `MetaModel(...)` inside each branch |
| Branches, combine all predictions | `{"merge": "predictions"}, {"model": Ridge()}` |
| Branches, selective per-branch | `MetaModel(source_predictions=[{...}, {...}])` |
| Branches, mixed features/predictions | `MetaModel(source_predictions=[...], source_features={...})` |

When you need both (stacking after branches), you compose them explicitly:
```python
{"merge": "predictions"},
{"model": Ridge()}
```

Or use MetaModel with per-branch configuration for automatic handling.

The refactoring should REMOVE the Phase 7 merge delegation code from MetaModelController and restore the original standalone stacking behavior, while ADDING the new feature input and per-branch selection capabilities.
