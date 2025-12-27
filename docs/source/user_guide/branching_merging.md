# Branching and Merging

This guide covers creating parallel processing paths in nirs4all pipelines using branching, and combining their outputs with merge operations.

## Overview

Branching enables evaluating multiple preprocessing strategies or models within a single pipeline execution. The results can be merged to:

- **Compare** different approaches and select the best
- **Combine** features from different preprocessing chains
- **Stack** predictions from multiple models (ensemble learning)

### Key Concepts

| Keyword | Purpose | Exits Branch Mode? |
|---------|---------|-------------------|
| `branch` | Create parallel execution paths | ❌ No |
| `merge` | Combine branch outputs | ✅ Yes |
| `source_branch` | Per-source preprocessing | ❌ No |
| `merge_sources` | Combine multi-source features | ❌ No |

## Creating Branches

### Basic Syntax

Use the `branch` keyword to create parallel processing paths:

```python
from nirs4all.operators.transforms import SNV, MSC, FirstDerivative, SavitzkyGolay

pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5),

    # Create 3 branches with different preprocessing
    {"branch": [
        [SNV(), SavitzkyGolay(window_length=11, polyorder=2)],
        [MSC(), FirstDerivative()],
        [FirstDerivative(), SecondDerivative()],
    ]},

    # This step runs on ALL branches
    PLSRegression(n_components=10),
]
```

Each branch receives a copy of the data and processes it independently. Steps after the branch block execute on **every branch**.

### Generator Syntax

For exploring many variants, use generator syntax:

```python
# Generates 3 branches automatically
{"branch": {"_or_": [SNV(), MSC(), FirstDerivative()]}}

# Multiple steps with generator
{"branch": {
    "_or_": [
        [SNV(), PCA(n_components=10)],
        [MSC(), PCA(n_components=20)],
        [FirstDerivative(), PCA(n_components=15)],
    ]
}}
```

### Post-Branch Steps

Steps after a branch block run independently on each branch:

```python
pipeline = [
    KFold(n_splits=5),
    {"branch": [
        [SNV()],
        [MSC()],
    ]},
    SavitzkyGolay(window_length=11),  # Runs 2x (once per branch)
    PLSRegression(n_components=10),    # Runs 2x (once per branch)
]
# Result: 2 separate models, one per branch
```

## Merging Branches

The `merge` step combines branch outputs and **always exits branch mode**.

### Feature Merging

Concatenate features from all branches horizontally:

```python
pipeline = [
    KFold(n_splits=5),
    {"branch": [
        [SNV(), PCA(n_components=10)],   # 10 features
        [MSC(), PCA(n_components=20)],   # 20 features
    ]},
    {"merge": "features"},               # Now 30 features
    PLSRegression(n_components=15),      # Single model on merged features
]
```

**Result shape**: `(n_samples, branch0_features + branch1_features + ...)`

### Prediction Merging (Stacking)

Collect out-of-fold (OOF) predictions for stacking:

```python
pipeline = [
    KFold(n_splits=5),
    {"branch": [
        [SNV(), PLSRegression(n_components=10)],
        [MSC(), RandomForestRegressor()],
    ]},
    {"merge": "predictions"},            # OOF predictions as features
    Ridge(alpha=1.0),                    # Meta-learner
]
```

:::{important}
**OOF Safety**: Predictions are reconstructed using out-of-fold values by default. This prevents data leakage when merged predictions are used for training downstream models.
:::

### Mixed Merging

Combine features from some branches with predictions from others:

```python
pipeline = [
    KFold(n_splits=5),
    {"branch": [
        [SNV(), PLSRegression(n_components=10)],  # Good model
        [MSC(), PCA(n_components=20)],            # Good features (no model)
    ]},
    {"merge": {
        "predictions": [0],  # OOF predictions from branch 0
        "features": [1],     # PCA features from branch 1
    }},
    Ridge(alpha=1.0),
]
```

### Merge Options

Full configuration syntax:

```python
{"merge": {
    "features": "all",              # or [0, 1, 2] for specific branches
    "predictions": "all",           # or [0, 1] for specific branches
    "include_original": False,      # Include pre-branch features?
    "on_missing": "error",          # "error" | "warn" | "skip"
    "unsafe": False,                # Disable OOF safety? (NOT RECOMMENDED)
}}
```

## Multi-Source Pipelines

For datasets with multiple feature sources (e.g., NIR + markers), use source-specific processing.

### Source Branching

Apply different preprocessing to each data source:

```python
from nirs4all.data import DatasetConfigs

# Multi-source dataset
dataset = DatasetConfigs([
    {"path": "nir_spectra.csv", "source_name": "NIR"},
    {"path": "genetic_markers.csv", "source_name": "markers"},
])

pipeline = [
    # Source-specific preprocessing
    {"source_branch": {
        "NIR": [SNV(), FirstDerivative(), SavitzkyGolay(window_length=11)],
        "markers": [VarianceThreshold(), MinMaxScaler()],
    }},

    # Sources auto-merged after source_branch by default
    KFold(n_splits=5),
    PLSRegression(n_components=15),
]
```

### Source Branch Options

```python
{"source_branch": {
    "NIR": [SNV()],
    "markers": [MinMaxScaler()],
    "_default_": [MinMaxScaler()],    # For unlisted sources
    "_merge_after_": True,            # Auto-merge after? (default: True)
    "_merge_strategy_": "concat",     # Merge strategy (default: "concat")
}}
```

### Explicit Source Merging

Control how sources are combined:

```python
{"source_branch": {
    "NIR": [SNV()],
    "markers": [MinMaxScaler()],
    "_merge_after_": False,           # Disable auto-merge
}},
{"merge_sources": "concat"},          # Explicit merge step
```

### Source Merge Strategies

| Strategy | Output Shape | Requirement |
|----------|--------------|-------------|
| `concat` | 2D (samples, total_features) | None |
| `stack` | 3D (samples, sources, features) | Same feature count |
| `dict` | Dict\[str, ndarray\] | None |

```python
# Horizontal concatenation (default)
{"merge_sources": "concat"}

# 3D stacking (for CNNs, attention models)
{"merge_sources": "stack"}

# Keep as dictionary (for multi-input models)
{"merge_sources": "dict"}
```

## Advanced: Prediction Selection

Control which models contribute to the merge:

```python
{"merge": {
    "predictions": [
        # Best model from branch 0
        {"branch": 0, "select": "best", "metric": "rmse"},
        # Top 2 models from branch 1
        {"branch": 1, "select": {"top_k": 2}, "metric": "r2"},
    ]
}}
```

### Selection Options

| Strategy | Syntax | Description |
|----------|--------|-------------|
| `all` | `"all"` | All models in the branch |
| `best` | `"best"` | Single best model by metric |
| `top_k` | `{"top_k": N}` | Top N models by metric |
| `explicit` | `["PLS", "RF"]` | Specific model names |

### Aggregation Options

Control how predictions are combined per branch:

```python
{"merge": {
    "predictions": [
        {
            "branch": 0,
            "select": "all",
            "aggregate": "weighted_mean",  # Weight by validation score
            "metric": "rmse",
        },
    ]
}}
```

| Strategy | Result Shape | Description |
|----------|-------------|-------------|
| `separate` | (samples, n_models) | Each model as separate feature |
| `mean` | (samples, 1) | Simple average |
| `weighted_mean` | (samples, 1) | Weighted by validation score |

## Advanced: Metadata Partitioning

Create branches based on metadata values (e.g., site, variety, instrument):

```python
pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5),

    # Partition samples by site
    {
        "branch": [PLSRegression(n_components=5), RandomForestRegressor()],
        "by": "metadata_partitioner",
        "column": "site",
        "cv": ShuffleSplit(n_splits=3),  # Per-site CV
        "min_samples": 20,               # Skip small partitions
    },

    # Merge disjoint predictions
    {"merge": "predictions", "n_columns": 2, "select_by": "mse"},

    # Meta-learner
    Ridge(alpha=1.0),
]
```

:::{note}
**Disjoint vs Copy Branches**: Metadata partitioners create disjoint branches where each sample exists in exactly one branch. Regular branches create copies where all branches see all samples.
:::

### Partition Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `by` | `str` | `"metadata_partitioner"` or `"sample_partitioner"` |
| `column` | `str` | Metadata column name |
| `cv` | Splitter | Per-partition cross-validation |
| `min_samples` | `int` | Minimum samples per partition |
| `group_values` | `dict` | Group rare values together |

### Grouping Rare Values

```python
{
    "branch": [PLSRegression(n_components=10)],
    "by": "metadata_partitioner",
    "column": "percentage",
    "group_values": {
        "zero": [0],
        "low": [5, 10, 15],
        "high": [20, 25, 30, 35, 40],
    },
}
# Creates 3 partitions: "zero", "low", "high"
```

## Complete Examples

### Basic Stacking Pipeline

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from nirs4all.operators.transforms import SNV, MSC
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.data import DatasetConfigs

pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5),

    # Branch: different preprocessing + models
    {"branch": [
        [SNV(), PLSRegression(n_components=10)],
        [MSC(), RandomForestRegressor(n_estimators=100)],
    ]},

    # Merge OOF predictions
    {"merge": "predictions"},

    # Meta-learner
    {"model": Ridge(alpha=1.0)},
]

runner = PipelineRunner(verbose=1)
predictions, _ = runner.run(
    PipelineConfigs(pipeline, "Stacking Example"),
    DatasetConfigs("data/wheat.csv", y_column="protein")
)
```

### Multi-Source Fusion

```python
from nirs4all.operators.transforms import SNV, FirstDerivative
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

dataset = DatasetConfigs([
    {"path": "nir.csv", "source_name": "NIR"},
    {"path": "markers.csv", "source_name": "markers"},
])

pipeline = [
    # Source-specific preprocessing
    {"source_branch": {
        "NIR": [SNV(), FirstDerivative()],
        "markers": [VarianceThreshold(), MinMaxScaler()],
    }},

    # Merged automatically, then train
    KFold(n_splits=5),
    PLSRegression(n_components=15),
]

runner = PipelineRunner(verbose=1)
predictions, _ = runner.run(
    PipelineConfigs(pipeline, "Multi-Source Fusion"),
    dataset
)
```

### Feature Engineering + Model Stacking

```python
pipeline = [
    KFold(n_splits=5),

    # Branch 0: Model (produces predictions)
    # Branch 1: Features only (no model)
    {"branch": [
        [SNV(), PLSRegression(n_components=10)],
        [MSC(), PCA(n_components=30)],  # Feature extraction only
    ]},

    # Mixed merge: predictions from 0, features from 1
    {"merge": {
        "predictions": [0],
        "features": [1],
    }},

    # Meta-learner on combined features
    {"model": Ridge(alpha=1.0)},
]
```

## Error Handling

### Common Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| "Not in branch mode" | `merge` without prior `branch` | Add a `branch` step first |
| "Feature dimension mismatch" | Branches have different feature counts | Use `on_missing: "skip"` |
| "No models found" | Branch has no trained models | Ensure models in branch |

### Debugging Tips

1. **Enable verbose logging**: `PipelineRunner(verbose=2)`
2. **Check branch contexts**: Inspect `context.custom["branch_contexts"]`
3. **Validate merge config**: Print parsed `MergeConfig` before execution

## See Also

- {doc}`stacking` - Stacking with MetaModel
- {doc}`preprocessing` - Preprocessing techniques
- {doc}`/reference/pipeline_syntax` - Full pipeline syntax reference
- {doc}`/examples/index` - Working examples including branching

**Example files:**
- `examples/developer/01_advanced_pipelines/D01_branching_basics.py` - Basic branching
- `examples/developer/01_advanced_pipelines/D03_merge_basics.py` - Merge examples
- `examples/user/03_preprocessing/U10_feature_augmentation.py` - Feature augmentation
