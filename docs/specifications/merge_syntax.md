# Merge Syntax Reference

**Version**: 1.0.0
**Status**: Production Ready
**Date**: December 2025

This document provides a comprehensive reference for the merge syntax in nirs4all pipelines.

---

## Table of Contents

1. [Overview](#overview)
2. [Keywords](#keywords)
3. [Branch Merge Syntax](#branch-merge-syntax)
4. [Source Merge Syntax](#source-merge-syntax)
5. [Source Branch Syntax](#source-branch-syntax)
6. [Disjoint Sample Branch Merging](#disjoint-sample-branch-merging)
7. [Prediction Selection](#prediction-selection)
8. [Aggregation Strategies](#aggregation-strategies)
9. [Error Codes](#error-codes)
10. [Examples](#examples)

---

## Overview

nirs4all provides three related keywords for combining data:

| Keyword | Purpose | Exits Branch Mode? | Use Case |
|---------|---------|-------------------|----------|
| `merge` | Combine outputs from pipeline branches | ✅ Yes | Feature/prediction stacking |
| `merge_sources` | Combine features from data sources | ❌ No | Multi-modal fusion |
| `source_branch` | Per-source pipeline execution | ❌ No | Source-specific preprocessing |

### Key Concepts

**Branches vs Sources**:
- **Branches** represent *execution paths* (parallel processing strategies)
- **Sources** represent *data provenance* (different sensors/modalities)

These are orthogonal dimensions. A pipeline can have:
- Multiple sources, each processed by multiple branches
- A single source with multiple processing branches
- Multiple sources with source-specific branching

---

## Keywords

### `merge`

Combines outputs from pipeline branches and exits branch mode.

```python
# Simple syntax
{"merge": "features"}      # Collect features from all branches
{"merge": "predictions"}   # Collect OOF predictions from all branches
{"merge": "all"}           # Collect both features and predictions

# Dict syntax
{"merge": {
    "features": "all",              # or [0, 1, 2] for specific branches
    "predictions": "all",           # or [0, 1] for specific branches
    "include_original": False,      # Include pre-branch features?
    "on_missing": "error",          # "error" | "warn" | "skip"
    "unsafe": False,                # Disable OOF safety? (NOT RECOMMENDED)
    "output_as": "features",        # "features" | "sources" | "dict"
}}
```

### `merge_sources`

Combines features from multiple data sources.

```python
# Simple syntax
{"merge_sources": "concat"}    # Horizontal concatenation
{"merge_sources": "stack"}     # 3D stacking (requires compatible shapes)
{"merge_sources": "dict"}      # Keep as dictionary

# Dict syntax
{"merge_sources": {
    "strategy": "concat",              # "concat" | "stack" | "dict"
    "sources": "all",                  # or ["NIR", "markers"] for specific sources
    "on_incompatible": "flatten",      # "error" | "flatten" | "pad" | "truncate"
}}
```

### `source_branch`

Creates per-source processing pipelines.

```python
# Auto mode (each source processed independently)
{"source_branch": "auto"}

# Dict syntax (source-specific pipelines)
{"source_branch": {
    "NIR": [SNV(), FirstDerivative()],
    "markers": [VarianceThreshold(), MinMaxScaler()],
    "_default_": [MinMaxScaler()],     # Default for unlisted sources
    "_merge_after_": True,             # Auto-merge after? (default: True)
    "_merge_strategy_": "concat",      # Merge strategy (default: "concat")
}}
```

---

## Branch Merge Syntax

### Feature Merging

Collect and concatenate features from branches:

```python
# All branches
{"merge": "features"}

# Specific branches
{"merge": {"features": [0, 2]}}      # Branches 0 and 2 only

# With options
{"merge": {
    "features": "all",
    "include_original": True,         # Prepend pre-branch features
}}
```

**Result**: 2D array `[branch0_features | branch1_features | ...]`

### Prediction Merging

Collect OOF predictions from branch models:

```python
# All predictions from all branches
{"merge": "predictions"}

# Specific branches
{"merge": {"predictions": [0, 1]}}

# With unsafe mode (NOT RECOMMENDED)
{"merge": {"predictions": "all", "unsafe": True}}
```

**OOF Safety**: By default, predictions are reconstructed using out-of-fold (OOF) values. This prevents data leakage when merged predictions are used for training downstream models.

**Unsafe Mode**: Setting `unsafe=True` disables OOF reconstruction. Only use this if:
- You understand the implications of data leakage
- You're doing rapid prototyping
- The downstream model won't be evaluated on this data

### Mixed Merging

Combine features from some branches with predictions from others:

```python
{"merge": {
    "features": [1],        # Features from branch 1
    "predictions": [0],     # OOF predictions from branch 0
}}
```

**Use case**: Branch 0 has a strong model (predictions valuable), Branch 1 has good features (e.g., PCA without model).

---

## Source Merge Syntax

### Strategies

| Strategy | Output Shape | Requirement | Use Case |
|----------|--------------|-------------|----------|
| `concat` | 2D (samples, total_features) | None | Default fusion |
| `stack` | 3D (samples, sources, features) | Same feature count | CNNs, attention |
| `dict` | Dict[str, ndarray] | None | Multi-input models |

```python
# Concatenation (default)
{"merge_sources": "concat"}
# Shape: (n_samples, src1_features + src2_features + ...)

# 3D stacking
{"merge_sources": "stack"}
# Shape: (n_samples, n_sources, n_features)
# Requires all sources to have same feature dimension

# Dictionary
{"merge_sources": "dict"}
# Result: {"NIR": array, "markers": array}
# Stored in context for multi-input models
```

### Incompatibility Handling

When sources have different feature dimensions and you want to stack:

```python
{"merge_sources": {
    "strategy": "stack",
    "on_incompatible": "pad",      # Zero-pad shorter sources
}}

# Options:
# - "error": Raise ValueError (default for stack)
# - "flatten": Fall back to 2D concat
# - "pad": Zero-pad shorter to match longest
# - "truncate": Truncate longer to match shortest
```

---

## Source Branch Syntax

### Auto Mode

Process each source independently with default pipeline:

```python
{"source_branch": "auto"}
```

### Source-Specific Pipelines

```python
{"source_branch": {
    "NIR": [
        SNV(),
        FirstDerivative(),
        SavitzkyGolay(window_length=11, polyorder=2),
    ],
    "markers": [
        VarianceThreshold(threshold=0.01),
        MinMaxScaler(),
    ],
    "Raman": [
        BaselineCorrection(),
        SNV(),
    ],
}}
```

### Special Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `_default_` | List[step] | None | Pipeline for unlisted sources |
| `_merge_after_` | bool | True | Auto-merge sources after processing |
| `_merge_strategy_` | str | "concat" | Strategy for auto-merge |

```python
{"source_branch": {
    "NIR": [SNV()],
    "_default_": [MinMaxScaler()],    # Other sources use this
    "_merge_after_": False,           # Don't auto-merge
}}
# Followed by explicit merge_sources step
{"merge_sources": "concat"}
```

---

## Disjoint Sample Branch Merging

Disjoint sample branches are created by partitioners that divide samples into non-overlapping subsets. Unlike copy branches (where all branches see all samples), disjoint branches ensure each sample exists in exactly ONE branch.

### Partitioner Types

| Partitioner | Description | Use Case |
|-------------|-------------|----------|
| `metadata_partitioner` | Partition by metadata column value | Site/variety/instrument-specific models |
| `sample_partitioner` | Partition by outlier status | Separate outlier vs. inlier models |

### Metadata Partitioner Syntax

```python
{
    "branch": [PLS(5), RF(100)],        # Steps to run in each partition
    "by": "metadata_partitioner",        # Partitioner type
    "column": "site",                    # Metadata column name
    "cv": ShuffleSplit(n_splits=3),      # Optional: per-branch CV
    "min_samples": 20,                   # Optional: skip small partitions
    "group_values": {                    # Optional: combine rare values
        "others": ["C", "D", "E"],
    },
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `branch` | `list` | Yes | - | Models/steps to run in each branch |
| `by` | `str` | Yes | - | `"metadata_partitioner"` or `"sample_partitioner"` |
| `column` | `str` | Yes* | - | Metadata column (*required for metadata_partitioner) |
| `cv` | Splitter | No | `None` | Per-branch cross-validation strategy |
| `min_samples` | `int` | No | `1` | Minimum samples per branch; branches with fewer are skipped |
| `group_values` | `dict` | No | `None` | Map of branch_name → list of values to group |

### Disjoint Merge Syntax

When merging disjoint branches, additional options control model selection:

```python
# Default: auto-detect N columns from branches
{"merge": "predictions"}

# Override: force specific column count
{"merge": "predictions", "n_columns": 2}

# Override: selection criterion for top-N
{"merge": "predictions", "select_by": "mse"}   # default: lowest MSE
{"merge": "predictions", "select_by": "r2"}    # highest R²
{"merge": "predictions", "select_by": "mae"}   # lowest MAE
{"merge": "predictions", "select_by": "order"} # first N in definition order
```

### Disjoint Merge Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_columns` | `int` | `None` (auto) | Force specific output column count |
| `select_by` | `str` | `"mse"` | Criterion for selecting top N models |

### Feature Merge Behavior

For feature merge with disjoint branches:

```
IF all branches produce the SAME feature dimension:
    → Concatenate rows by sample_id
    → Result: (n_total_samples, n_features)

ELSE (asymmetric feature dimensions):
    → ERROR: "Cannot merge features from disjoint branches with different
       feature dimensions"
```

### Prediction Merge Behavior

For prediction merge with disjoint branches:

1. **Determine N** (output column count):
   - If `n_columns` specified: use that value
   - Otherwise: N = min(model_count across all branches)

2. **Select top N models per branch** by `select_by` criterion

3. **Reconstruct OOF matrix** by sample_id:
   - Each row populated from its branch's predictions
   - Result: (n_total_samples, N)

### Example: Site-Specific Stacking

```python
pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3),

    # Partition by site
    {
        "branch": [PLS(5), RF(100), XGB()],
        "by": "metadata_partitioner",
        "column": "site",
        "cv": ShuffleSplit(n_splits=3),
    },

    # Merge with model selection
    {"merge": "predictions", "n_columns": 2, "select_by": "mse"},

    # Meta-learner
    Ridge(),
]
```

### Example: Group Rare Values

```python
pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3),

    {
        "branch": [PLS(10)],
        "by": "metadata_partitioner",
        "column": "percentage",
        "group_values": {
            "zero": [0],
            "low": [5, 10, 15],
            "high": [20, 25, 30, 35, 40],
        },
    },

    {"merge": "predictions"},
]
# Creates 3 partitions: "zero", "low", "high"
```

### Disjoint vs Copy Branch Comparison

| Aspect | Copy Branches | Disjoint Branches |
|--------|---------------|-------------------|
| Samples | All branches see all samples | Each sample in exactly one branch |
| OOF shape | (n_samples, n_models) per branch | (n_branch_samples, n_models) per branch |
| Feature merge | Horizontal concat (n, f1+f2+...) | Row concat (n1+n2+..., f) if symmetric |
| Prediction merge | Stack for meta-model | Reconstruct by sample_id |
| Cross-validation | Global folds, shared | Per-branch folds, independent |
| Use case | Compare preprocessing variants | Per-site/variety/instrument models |

---

## Prediction Selection

### Selection Strategies

Control which models from each branch contribute to the merge:

| Strategy | Syntax | Description |
|----------|--------|-------------|
| `all` | `"all"` | All models in the branch |
| `best` | `"best"` | Single best model by metric |
| `top_k` | `{"top_k": N}` | Top N models by metric |
| `explicit` | `["model1", "model2"]` | Specific model names |

### Per-Branch Configuration

```python
{"merge": {
    "predictions": [
        {"branch": 0, "select": "best", "metric": "rmse"},
        {"branch": 1, "select": {"top_k": 2}, "metric": "r2"},
        {"branch": 2, "select": ["PLS", "RF"]},  # Explicit names
    ]
}}
```

### Metrics for Selection

| Metric | Lower is Better? | Task |
|--------|-----------------|------|
| `rmse` | ✅ | Regression |
| `mse` | ✅ | Regression |
| `mae` | ✅ | Regression |
| `r2` | ❌ | Regression |
| `accuracy` | ❌ | Classification |
| `f1` | ❌ | Classification |

---

## Aggregation Strategies

Control how predictions from multiple models are combined:

| Strategy | Result Shape | Description |
|----------|-------------|-------------|
| `separate` | (samples, n_models) | Each model as separate feature |
| `mean` | (samples, 1) | Simple average |
| `weighted_mean` | (samples, 1) | Weighted by validation score |
| `proba_mean` | (samples, n_classes) | Average probabilities (classification) |

### Per-Branch Aggregation

```python
{"merge": {
    "predictions": [
        {
            "branch": 0,
            "select": "all",
            "aggregate": "weighted_mean",
            "metric": "rmse",  # Used for weighting
        },
        {
            "branch": 1,
            "select": "all",
            "aggregate": "mean",
        },
    ]
}}
```

**Example**: Branch 0 has 3 PLS variants, Branch 1 has 2 RF variants.
- With `separate`: 5 features (3 + 2)
- With `mean` per branch: 2 features (1 + 1)

---

## Error Codes

### Branch Merge Errors

| Code | Message | Resolution |
|------|---------|------------|
| MERGE-E001 | No feature snapshot | Branch has no features to merge |
| MERGE-E002 | Feature dimension mismatch | Use `on_shape_mismatch` parameter |
| MERGE-E003 | Sample count mismatch | All branches must have same samples |
| MERGE-E010 | No models found | Branch has no trained models |
| MERGE-E011 | Missing model | Explicit model name not found |
| MERGE-E020 | Not in branch mode | Use merge only after branch step |
| MERGE-E021 | Invalid branch index | Branch index out of range |
| MERGE-E025 | Unsafe merge warning | Data leakage risk acknowledged |

### Source Merge Errors

| Code | Message | Resolution |
|------|---------|------------|
| MERGE-E024 | Single source | merge_sources is no-op for single source |
| MERGE-E030 | Shape mismatch | Use `on_incompatible` parameter |
| MERGE-E031 | Unknown source | Source name not in dataset |

### Source Branch Errors

| Code | Message | Resolution |
|------|---------|------------|
| SOURCEBRANCH-E001 | No sources | Dataset has no feature sources |
| SOURCEBRANCH-E002 | Single source | source_branch is no-op for single source |

### Disjoint Merge Errors

| Code | Message | Resolution |
|------|---------|------------|
| DISJOINT-E001 | Feature dimension mismatch | Ensure all partitions produce same feature count |
| DISJOINT-E002 | n_columns exceeds minimum | Reduce n_columns or add more models |
| DISJOINT-E003 | No predictions in branch | Ensure models are trained in each partition |
| DISJOINT-E004 | Too few samples | Merged predictions have < 10 samples |
| DISJOINT-E005 | Non-finite values | Merged predictions contain NaN/Inf |
| DISJOINT-E006 | Missing metadata column | Required column not in prediction data |

---

## Examples

### Basic Stacking Pipeline

```python
pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5),

    # Branch: different preprocessing + models
    {"branch": [
        [SNV(), PLSRegression(n_components=10)],
        [MSC(), RandomForestRegressor()],
    ]},

    # Merge predictions (OOF-safe)
    {"merge": "predictions"},

    # Meta-learner
    Ridge(alpha=1.0),
]
```

### Multi-Source Fusion

```python
dataset = DatasetConfigs([
    {"path": "nir.csv", "source_name": "NIR"},
    {"path": "markers.csv", "source_name": "markers"},
])

pipeline = [
    # Source-specific preprocessing
    {"source_branch": {
        "NIR": [SNV(), FirstDerivative()],
        "markers": [MinMaxScaler()],
    }},

    # Combine sources
    # (auto-merged by default, or explicit:)
    # {"merge_sources": "concat"},

    KFold(n_splits=5),
    PLSRegression(n_components=15),
]
```

### Asymmetric Branches with Mixed Merge

```python
pipeline = [
    KFold(n_splits=5),

    # Branch 0: Model (produces predictions)
    # Branch 1: Features only (no model)
    {"branch": [
        [SNV(), PLSRegression(n_components=10)],
        [PCA(n_components=30)],  # No model!
    ]},

    # Mixed merge: predictions from 0, features from 1
    {"merge": {
        "predictions": [0],
        "features": [1],
    }},

    Ridge(alpha=1.0),
]
```

### Per-Branch Selection and Aggregation

```python
pipeline = [
    KFold(n_splits=5),

    {"branch": [
        [SNV(), PLS(5), PLS(10), PLS(15)],  # 3 PLS variants
        [MSC(), RF(), GBR()],                # 2 ensemble models
    ]},

    {"merge": {
        "predictions": [
            # Best PLS from branch 0
            {"branch": 0, "select": "best", "metric": "rmse"},
            # Weighted average of ensemble models from branch 1
            {"branch": 1, "select": "all", "aggregate": "weighted_mean", "metric": "rmse"},
        ]
    }},

    Ridge(alpha=1.0),
]
```

### Disjoint Sample Branch (Metadata Partitioner)

```python
pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5),

    # Partition samples by site metadata
    {
        "branch": [PLS(5), RF(100), XGB()],
        "by": "metadata_partitioner",
        "column": "site",
        "cv": ShuffleSplit(n_splits=3),  # Per-site CV
        "min_samples": 20,                # Skip small sites
    },

    # Merge disjoint predictions
    # Each site: select top 2 models by MSE
    {"merge": "predictions", "n_columns": 2, "select_by": "mse"},

    # Meta-learner combines per-site predictions
    Ridge(alpha=1.0),
]
```

### Disjoint Feature Merge

```python
pipeline = [
    KFold(n_splits=5),

    # Partition by site, apply site-specific preprocessing
    {
        "branch": [SNV(), PCA(n_components=20)],
        "by": "metadata_partitioner",
        "column": "site",
    },

    # Merge features (requires same dimensions from all sites)
    {"merge": "features"},

    # Global model on merged features
    PLSRegression(n_components=10),
]
```

---

## See Also

- [Pipeline Syntax](pipeline_syntax.md) - General pipeline syntax reference
- [Branching Specification](concat_augmentation_specification.md) - Concat augmentation
- [Design Document](../report/branching_concat_merge_design_v3.md) - Full design rationale
- [Disjoint Merge Specification](../reports/disjoint_sample_branch_merging.md) - Full disjoint merge design
