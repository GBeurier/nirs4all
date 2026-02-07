# Branch Merging

After creating parallel branches with `branch` or `source_branch`, use **merge operators** to combine outputs and continue with a single pipeline path.

## Overview

nirs4all provides three merge-related keywords:

| Keyword | Purpose | Use Case |
|---------|---------|----------|
| `merge` | Combine outputs from pipeline branches | Feature fusion, stacking, ensemble building |
| `merge_sources` | Combine features from data sources | Multi-instrument fusion, multi-modal data |
| `source_branch` | Apply per-source preprocessing (auto-merges) | Source-specific preprocessing before fusion |

## Branch Merging (`merge`)

The `merge` keyword combines outputs from branches created with `{"branch": [...]}`.

### Basic Syntax

```python
# Simple string syntax
{"merge": "features"}      # Concatenate X matrices from all branches
{"merge": "predictions"}   # Collect OOF predictions (for stacking)
{"merge": "all"}           # Collect both features and predictions
```

### Feature Merging

Concatenate feature matrices (X) horizontally from all branches:

```python
pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=5, random_state=42),
    {"branch": {
        "snv": [SNV()],
        "msc": [MSC()],
        "derivative": [FirstDerivative()],
    }},
    {"merge": "features"},  # Concatenate X from all branches
    PLSRegression(n_components=10),
]
```

**Before merge:**
```
Branch 0 (snv):       shape (n, p)
Branch 1 (msc):       shape (n, p)
Branch 2 (derivative): shape (n, p)
```

**After merge:**
```
Merged X: shape (n, 3*p)  # Horizontal concatenation
```

### Prediction Merging (Stacking)

Collect out-of-fold (OOF) predictions from branches for meta-model training:

```python
pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=5, random_state=42),
    {"branch": {
        "pls": [SNV(), PLSRegression(n_components=5)],
        "ridge": [MSC(), Ridge(alpha=1.0)],
        "rf": [FirstDerivative(), RandomForestRegressor(n_estimators=50)],
    }},
    {"merge": "predictions"},  # Collect OOF predictions
    Ridge(alpha=0.1),          # Meta-model (Level 2)
]
```

**Before merge:**
```
Branch 0 (pls):   predictions shape (n,)
Branch 1 (ridge): predictions shape (n,)
Branch 2 (rf):    predictions shape (n,)
```

**After merge:**
```
Meta X: shape (n, 3)  # Stacked predictions as features
```

OOF predictions prevent data leakage by using validation set predictions during training.

### Mixed Merging

Select features from some branches and predictions from others:

```python
pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=5, random_state=42),
    {"branch": [
        [SNV()],                                      # Branch 0: preprocessing only
        [MSC()],                                      # Branch 1: preprocessing only
        [FirstDerivative(), PLSRegression(5)],       # Branch 2: has model
    ]},
    {"merge": {"features": [0, 1], "predictions": [2]}},
    Ridge(alpha=0.1),
]
```

This combines:
- Features from branches 0 and 1 (preprocessed X)
- Predictions from branch 2 (model outputs)

### Dict Syntax Options

The dict syntax provides fine-grained control over merging:

```python
{"merge": {
    # What to collect
    "features": "all",           # or [0, 2] for specific branches
    "predictions": "all",        # or [1, 3] for specific branches

    # Feature options
    "include_original": True,    # Include pre-branch features
    "aggregation": "mean",       # mean | median | std | min | max

    # Prediction options
    "unsafe": False,             # Default: use OOF reconstruction

    # Output format
    "output_as": "features",     # features | sources | dict
    "source_names": ["snv", "msc"],  # Names for output_as="sources"

    # Error handling
    "on_missing": "error",       # error | warn | skip
    "on_shape_mismatch": "error", # error | allow | pad | truncate
}}
```

### Branch Selection

Select specific branches by index or name:

```python
# By index
{"merge": {"features": [0, 2]}}     # Only branches 0 and 2

# By name (when using named branches)
{"merge": {"features": ["snv", "derivative"]}}

# All branches (default)
{"merge": {"features": "all"}}
```

### Feature Aggregation

Instead of concatenating, aggregate features across branches:

```python
pipeline = [
    {"branch": {
        "snv": [SNV()],
        "msc": [MSC()],
        "derivative": [FirstDerivative()],
    }},
    {"merge": {"features": "all", "aggregation": "mean"}},
    PLSRegression(n_components=5),
]
```

**Aggregation options:**
- `"mean"` - Average features across branches
- `"median"` - Median of features
- `"std"` - Standard deviation
- `"min"` - Minimum value
- `"max"` - Maximum value

Result shape: `(n, p)` instead of `(n, branches*p)`

### Include Original Features

Preserve pre-branch features alongside merged branch features:

```python
pipeline = [
    MinMaxScaler(),  # Original features after this
    {"branch": {
        "snv": [SNV()],
        "derivative": [FirstDerivative()],
    }},
    {"merge": {"features": "all", "include_original": True}},
    PLSRegression(n_components=10),
]
```

Result: `[original_X | snv_X | derivative_X]`

### Output Formats

Control how merged features are structured:

```python
# Default: single feature matrix
{"merge": {"features": "all", "output_as": "features"}}
# Result: shape (n, total_features)

# As separate sources (for multi-source models)
{"merge": {"features": "all", "output_as": "sources", "source_names": ["snv", "msc"]}}
# Result: SpectroDataset with named sources

# As dictionary (for multi-input models)
{"merge": {"features": "all", "output_as": "dict"}}
# Result: {"branch_0": X0, "branch_1": X1, ...}
```

### Per-Branch Prediction Configuration

Fine control over which models and how predictions are collected:

```python
{"merge": {
    "predictions": [
        # Branch 0: Use best model by RMSE
        {"branch": 0, "select": "best", "metric": "rmse"},

        # Branch 1: Top 2 models by R²
        {"branch": 1, "select": {"top_k": 2}, "metric": "r2"},

        # Branch 2: Specific models by name
        {"branch": 2, "select": ["PLS", "Ridge"]},

        # Branch 3: Average predictions instead of separate columns
        {"branch": 3, "select": "all", "aggregate": "mean"},
    ]
}}
```

**Selection strategies:**
- `"all"` - All models in branch (default)
- `"best"` - Single best model by metric
- `{"top_k": N}` - Top N models by metric
- `["model1", "model2"]` - Explicit model names

**Aggregation strategies:**
- `"separate"` - Each model as a separate feature column (default)
- `"mean"` - Simple average of predictions
- `"weighted_mean"` - Weighted by validation scores
- `"proba_mean"` - For classification: average class probabilities

### Unsafe Mode

By default, prediction merging uses OOF reconstruction to prevent data leakage. Disable this for special cases:

```python
# Safe (default): OOF predictions reconstructed per fold
{"merge": "predictions"}  # or {"merge": {"predictions": "all", "unsafe": False}}

# Unsafe: Direct predictions (data leakage risk!)
{"merge": {"predictions": "all", "unsafe": True}}
```

Use `unsafe: True` only when you understand the implications (e.g., for final model predictions on new data).

### Nested Branching

Multiple branch-merge cycles enable hierarchical architectures:

```python
pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=5, random_state=42),

    # Level 1: Preprocessing exploration
    {"branch": {
        "snv": [SNV()],
        "msc": [MSC()],
    }},
    {"merge": "features"},  # Exit first branch level

    # Level 2: Model comparison
    {"branch": {
        "pls": [PLSRegression(n_components=5)],
        "ridge": [Ridge(alpha=1.0)],
    }},
    {"merge": "predictions"},  # Stack predictions

    # Final meta-model
    Ridge(alpha=0.1),
]
```

Each `merge` exits its branch level, enabling sequential branch-merge cycles.

## Source Merging (`merge_sources`)

The `merge_sources` keyword combines features from different data sources (sensors, instruments, modalities).

### Basic Syntax

```python
# Simple string syntax
{"merge_sources": "concat"}   # Horizontal concatenation
{"merge_sources": "stack"}    # 3D stacking (for CNNs)
{"merge_sources": "average"}  # Element-wise average
```

### Concatenation (Default)

Horizontally concatenate all sources:

```python
pipeline = [
    {"source_branch": {
        "NIR": [SNV(), FirstDerivative()],
        "Raman": [MSC(), SavitzkyGolay()],
    }},
    {"merge_sources": "concat"},
    PLSRegression(n_components=15),
]
```

**Result:** `shape (n, nir_features + raman_features)`

### 3D Stacking

Create 3D array for CNN models:

```python
{"merge_sources": "stack"}
```

**Result:** `shape (n, n_sources, n_features)`

Requires sources to have the same number of features.

### Averaging

Element-wise average of sources:

```python
{"merge_sources": "average"}
```

**Result:** `shape (n, n_features)`

Requires identical dimensions across sources.

### Dict Syntax Options

```python
{"merge_sources": {
    "strategy": "concat",           # concat | stack | dict
    "sources": "all",               # or ["NIR", "markers"] for specific
    "on_incompatible": "error",     # error | flatten | pad | truncate
    "output_name": "merged",        # Name for merged source
    "preserve_source_info": True,   # Keep source metadata
}}
```

### Weighted Merging

Scale source contributions before combining:

```python
{"merge_sources": {
    "mode": "concat",
    "weights": {"NIR": 1.0, "Raman": 0.5}  # Scale Raman features by 0.5
}}
```

### Selective Merging

Include only specific sources:

```python
{"merge_sources": {
    "sources": ["NIR", "markers"],  # Exclude Raman
    "mode": "concat"
}}
```

### Handling Shape Mismatches

When sources have different feature dimensions:

```python
{"merge_sources": {
    "strategy": "stack",
    "on_incompatible": "error",     # Default: raise error
}}

# Or fallback strategies:
{"merge_sources": {
    "strategy": "stack",
    "on_incompatible": "flatten",   # Fall back to 2D concat
}}

{"merge_sources": {
    "strategy": "stack",
    "on_incompatible": "pad",       # Zero-pad shorter sources
}}

{"merge_sources": {
    "strategy": "stack",
    "on_incompatible": "truncate",  # Truncate longer sources
}}
```

## Source Branching (`source_branch`)

Apply source-specific preprocessing pipelines. By default, sources are automatically merged after processing.

### Basic Syntax

```python
{"source_branch": {
    "NIR": [SNV(), FirstDerivative()],
    "Raman": [MSC(), SavitzkyGolay()],
    "markers": [StandardScaler()],
}}
```

### Auto Mode

Process each source independently with empty pipeline:

```python
{"source_branch": "auto"}
```

### Default Pipeline

Apply default preprocessing to unlisted sources:

```python
{"source_branch": {
    "NIR": [SNV()],
    "_default_": [MinMaxScaler()],  # Applied to other sources
}}
```

### Controlling Auto-Merge

By default, sources merge after `source_branch`. Disable this:

```python
{"source_branch": {
    "NIR": [SNV()],
    "Raman": [MSC()],
    "_merge_after_": False,         # Keep sources separate
    "_merge_strategy_": "concat",   # Merge strategy if merging (default)
}}
```

### Indexed Sources

Reference sources by index instead of name:

```python
{"source_branch": {
    0: [SNV(), FirstDerivative()],
    1: [MinMaxScaler()],
}}
```

## Complete Examples

### Example 1: Multi-Preprocessing Fusion

Combine multiple preprocessing strategies:

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from nirs4all.operators.transforms import SNV, MSC, FirstDerivative
import nirs4all

pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5, shuffle=True, random_state=42),
    {"branch": {
        "snv": [SNV()],
        "msc": [MSC()],
        "derivative": [FirstDerivative()],
    }},
    {"merge": "features"},
    PLSRegression(n_components=15),
]

result = nirs4all.run(pipeline=pipeline, dataset="path/to/data")
```

### Example 2: Two-Level Stacking

Build a meta-model from diverse base models:

```python
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5, shuffle=True, random_state=42),
    {"branch": {
        "pls": [SNV(), PLSRegression(n_components=10)],
        "rf": [MSC(), RandomForestRegressor(n_estimators=100)],
        "ridge": [FirstDerivative(), Ridge(alpha=1.0)],
    }},
    {"merge": "predictions"},
    Ridge(alpha=0.1),  # Meta-model
]
```

### Example 3: Multi-Instrument Fusion

Combine data from different spectrometers:

```python
pipeline = [
    KFold(n_splits=5, shuffle=True, random_state=42),
    {"source_branch": {
        "portable": [
            SNV(),
            SavitzkyGolay(window_length=15, polyorder=2),
            FirstDerivative(),
        ],
        "benchtop": [
            SNV(),
            FirstDerivative(),
        ],
    }},
    {"merge_sources": {
        "mode": "concat",
        "weights": {"portable": 0.7, "benchtop": 1.0}
    }},
    PLSRegression(n_components=10),
]
```

### Example 4: Hybrid Source and Pipeline Branching

Combine source-level and algorithm-level branching:

```python
pipeline = [
    KFold(n_splits=5, shuffle=True, random_state=42),

    # Step 1: Per-source preprocessing
    {"source_branch": {
        "NIR": [SNV()],
        "Raman": [MSC()],
    }},
    {"merge_sources": "concat"},

    # Step 2: Feature scaling
    MinMaxScaler(),

    # Step 3: Model comparison
    {"branch": {
        "pls": [PLSRegression(n_components=10)],
        "rf": [RandomForestRegressor(n_estimators=100)],
    }},
]
```

### Example 5: Advanced Prediction Selection

Fine-grained control over stacking:

```python
pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5, shuffle=True, random_state=42),
    {"branch": {
        "diverse_pls": [SNV(),
            {"model": PLSRegression(3), "name": "PLS_3"},
            {"model": PLSRegression(5), "name": "PLS_5"},
            {"model": PLSRegression(10), "name": "PLS_10"},
        ],
        "ensemble": [MSC(), RandomForestRegressor(n_estimators=100)],
    }},
    {"merge": {
        "predictions": [
            # Top 2 PLS models by R²
            {"branch": "diverse_pls", "select": {"top_k": 2}, "metric": "r2"},
            # All from ensemble branch, averaged
            {"branch": "ensemble", "select": "all", "aggregate": "mean"},
        ]
    }},
    Ridge(alpha=0.1),
]
```

## Key Behaviors

1. **merge ALWAYS exits branch mode** - After merge, the pipeline returns to a single path
2. **OOF safety by default** - Prediction merging uses out-of-fold reconstruction
3. **source_branch auto-merges** - Unless `_merge_after_: False`
4. **Feature snapshots** - Pre-branch features are preserved for `include_original`
5. **Prediction mode support** - Merge configurations are saved and restored for inference

## Troubleshooting

### "No feature snapshot found"
The merge requires features from before the branch. Ensure preprocessing steps exist before branching.

### "Feature dimension mismatch"
Branches produced different feature counts. Use `aggregation` or check preprocessing consistency.

### "No models found in branch"
For prediction merge, at least one branch must contain a trained model.

### "Incomplete OOF coverage"
Some samples lack validation predictions. Check cross-validation setup or use `on_missing: "warn"`.

### "merge_sources requires multiple sources"
Your dataset has only one source. Use `branch` instead for single-source data.

## See Also

- {doc}`branching` - Creating parallel pipeline branches
- {doc}`stacking` - MetaModel stacking patterns
- {doc}`multi_source` - Loading and working with multi-source data
- {doc}`/reference/pipeline_syntax` - Complete pipeline syntax reference
- [D03_merge_basics.py](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D03_merge_basics.py) - Feature and prediction merge examples
- [D04_merge_sources.py](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D04_merge_sources.py) - Multi-source merge examples

```{seealso}
**Related Examples:**
- [D03: Merge Basics](../../../examples/developer/01_advanced_pipelines/D03_merge_basics.py) - Feature and prediction merging strategies
- [D04: Merge Sources](../../../examples/developer/01_advanced_pipelines/D04_merge_sources.py) - Multi-source branching and merging
- [D01: Branching Basics](../../../examples/developer/01_advanced_pipelines/D01_branching_basics.py) - Branching that leads to merging
```
