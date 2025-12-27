# Pipeline Branching

Pipeline branching enables splitting a pipeline into multiple parallel sub-pipelines ("branches"), each with its own preprocessing context while sharing common upstream state (splits, initial preprocessing).

## Overview

Branching is useful when you want to:

- **Compare preprocessing strategies**: Test SNV vs MSC vs derivatives with the same model
- **Explore model-preprocessing combinations**: Different models with different preprocessing
- **Efficient experimentation**: Single dataset load, shared CV splits
- **Independent Y processing**: Different target transformations per branch

## Basic Syntax

### List Syntax (Anonymous Branches)

The simplest form uses a list of lists:

```python
pipeline = [
    ShuffleSplit(n_splits=5),
    MinMaxScaler(),  # Shared - applied once
    {"branch": [
        [SNV()],           # Branch 0
        [MSC()],           # Branch 1
        [FirstDerivative()],  # Branch 2
    ]},
    PLSRegression(n_components=5),  # Runs on EACH branch
]
```

Result: 15 predictions (3 branches × 5 folds)

### Named Branches (Dictionary Syntax)

For better tracking, use named branches:

```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"branch": {
        "snv_pca": [SNV(), PCA(n_components=10)],
        "msc_detrend": [MSC(), Detrend()],
        "derivative": [FirstDerivative()],
    }},
    PLSRegression(n_components=5),
]
```

Named branches appear in predictions and visualizations with their given names.

### Generator Syntax

Use `_or_` generators for dynamic branch creation:

```python
pipeline = [
    ShuffleSplit(n_splits=3),
    {"branch": {"_or_": [SNV(), MSC(), FirstDerivative()]}},
    PLSRegression(n_components=5),
]
```

This expands to 3 branches automatically named based on the operator class.

## Multi-Step Branches

Branches can contain multiple steps:

```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"branch": {
        "snv_pca": [
            SNV(),
            PCA(n_components=10),
        ],
        "msc_savgol": [
            MSC(),
            SavitzkyGolay(window_length=11, polyorder=2),
        ],
    }},
    PLSRegression(n_components=5),
]
```

## Branch-Specific Y Processing

Each branch can have its own Y transformation:

```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"branch": {
        "scaled_y": [
            SNV(),
            {"y_processing": StandardScaler()},  # Y scaling in this branch only
        ],
        "raw_y": [
            MSC(),
            # No Y processing - uses numeric targets
        ],
    }},
    PLSRegression(n_components=5),
]
```

## In-Branch Model Training

Models can be placed inside branches:

```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"branch": {
        "snv_pls": [SNV(), PLSRegression(n_components=5)],
        "msc_pls": [MSC(), PLSRegression(n_components=10)],
        "derivative_rf": [FirstDerivative(), RandomForestRegressor()],
    }},
]
```

Each branch trains its own model independently.

## Post-Branch Steps

Steps after the branch block execute on **each branch**:

```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"branch": [
        [SNV(), PCA(n_components=10)],
        [MSC(), Detrend()],
    ]},
    PLSRegression(n_components=5),  # Trains twice: once per branch
]
```

The PLSRegression trains on:
- Branch 0: SNV→PCA features
- Branch 1: MSC→Detrend features

## Visualization

### Branch Summary

```python
from nirs4all.visualization.predictions import PredictionAnalyzer

analyzer = PredictionAnalyzer(predictions)

# Get summary statistics
summary = analyzer.branch_summary(metrics=['rmse', 'r2'])
print(summary.to_markdown())
```

Output:

| branch_name | branch_id | count | rmse_mean | rmse_std | r2_mean | r2_std |
|-------------|-----------|-------|-----------|----------|---------|--------|
| snv_pca | 0 | 5 | 0.123 | 0.008 | 0.945 | 0.012 |
| msc_detrend | 1 | 5 | 0.145 | 0.011 | 0.932 | 0.015 |
| derivative | 2 | 5 | 0.167 | 0.015 | 0.918 | 0.019 |

### Branch Comparison Bar Chart

```python
fig = analyzer.plot_branch_comparison(
    display_metric='rmse',
    display_partition='test',
    show_ci=True,  # Show confidence intervals
    ci_level=0.95
)
```

### Branch Boxplot

```python
fig = analyzer.plot_branch_boxplot(
    display_metric='rmse',
    display_partition='test'
)
```

### Branch × Fold Heatmap

```python
fig = analyzer.plot_branch_heatmap(
    y_var='fold_id',
    display_metric='rmse'
)
```

### Using Standard Heatmap

```python
fig = analyzer.plot_heatmap(
    x_var='branch_name',
    y_var='model_name',
    display_metric='rmse'
)
```

## Filtering by Branch

```python
# By branch name
snv_preds = predictions.filter_predictions(branch_name='snv_pca')

# By branch ID
branch_0 = predictions.filter_predictions(branch_id=0)

# Top model per branch
top = predictions.top(n=1, rank_metric='rmse', branch_name='snv_pca')
```

## Helper Methods

```python
# Get all branch names
branches = analyzer.get_branches()
# ['snv_pca', 'msc_detrend', 'derivative']

# Get all branch IDs
branch_ids = analyzer.get_branch_ids()
# [0, 1, 2]
```

## Key Behaviors

1. **Shared state before branch**: Splits and upstream preprocessing are computed once
2. **Independent contexts**: Each branch has its own X and Y processing state
3. **Single dataset load**: No redundant I/O
4. **Post-branch iteration**: Steps after branch execute on all branches
5. **Branch metadata**: Predictions include `branch_id` and `branch_name`

## Combining with Generators

Branches work with `_or_` and `_range_` generators:

```python
pipeline = [
    ShuffleSplit(n_splits=3),
    {"branch": {"_or_": [SNV(), MSC(), FirstDerivative()]}},  # 3 branches
    {"_range_": [5, 15, 5], "param": "n_components", "model": PLSRegression},  # 3 PLS variants
]
# Result: 27 predictions (3 branches × 3 n_components × 3 folds)
```

## Example Use Cases

### Use Case 1: Preprocessing Comparison

```python
pipeline = [
    GroupKFold(n_splits=5),
    {"y_processing": StandardScaler()},
    {"branch": [
        [SNV()],
        [MSC()],
        [FirstDerivative()],
        [SecondDerivative()],
    ]},
    PLSRegression(n_components=10),
]
```

### Use Case 2: Model-Preprocessing Exploration

```python
pipeline = [
    ShuffleSplit(n_splits=3),
    {"branch": {
        "snv_pls5": [SNV(), PLSRegression(n_components=5)],
        "snv_pls10": [SNV(), PLSRegression(n_components=10)],
        "msc_rf": [MSC(), RandomForestRegressor()],
    }},
]
```

### Use Case 3: Spectral Preprocessing Comparison

```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"branch": {
        "raw": [],  # No preprocessing
        "snv": [SNV()],
        "msc": [MSC()],
        "d1": [FirstDerivative()],
        "d2": [SecondDerivative()],
        "savgol_d1": [SavitzkyGolay(deriv=1)],
        "savgol_d2": [SavitzkyGolay(deriv=2)],
    }},
    PLSRegression(n_components=10),
]
```

## See Also

- {doc}`writing_pipelines` - Pipeline configuration guide
- {doc}`/reference/generator_keywords` - Generator syntax reference
- {doc}`/reference/predictions_api` - Predictions API reference
- {doc}`stacking` - Meta-model stacking guide
