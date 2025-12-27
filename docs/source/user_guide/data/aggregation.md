# Sample Aggregation

## Overview

In NIRS applications, it's common to have multiple spectral measurements (repetitions) for the same physical sample. For example:

- 4 scans per soil sample to reduce measurement noise
- Multiple measurements at different positions on a grain sample
- Repeated measurements for quality control

The **aggregation** feature allows you to:

1. Train models on all individual spectra (to maximize data)
2. Evaluate and report performance on **aggregated predictions** (one prediction per physical sample)

When aggregation is enabled, predictions from multiple spectra of the same biological sample are automatically combined, and both raw and aggregated metrics are reported.

## Quick Start

### Define Aggregation at Dataset Level

```python
from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.visualization.predictions import PredictionAnalyzer
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# Define dataset with aggregation column
dataset = DatasetConfigs(
    "path/to/spectra",
    aggregate="sample_id"  # Aggregate by sample_id column in metadata
)

# Define pipeline
pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.25),
    {"model": PLSRegression(n_components=10)}
]

# Run pipeline
runner = PipelineRunner(verbose=1)
predictions, _ = runner.run(PipelineConfigs(pipeline, "PLS"), dataset)

# Create analyzer with same aggregate setting
analyzer = PredictionAnalyzer(predictions, default_aggregate=runner.last_aggregate)

# All plots now use aggregation by default
fig = analyzer.plot_top_k(k=5)  # Automatically aggregated by sample_id
```

## Aggregation Methods

### By Metadata Column (Recommended)

Use a column from your metadata file to group samples:

```python
# Metadata file should contain a column like 'sample_id', 'ID', 'batch', etc.
dataset = DatasetConfigs("path/to/data", aggregate="sample_id")
```

### By Target Values

For classification tasks, aggregate by target class:

```python
# Aggregate spectra sharing the same target value
dataset = DatasetConfigs("path/to/data", aggregate=True)
```

### Via Config Dictionary

When using configuration dictionaries:

```python
config = {
    "train_x": "data/spectra.csv",
    "train_y": "data/targets.csv",
    "train_m": "data/metadata.csv",  # Contains 'sample_id' column
    "aggregate": "sample_id"
}

dataset = DatasetConfigs(config)
```

## Pipeline Output

When aggregation is enabled, the TabReport shows both raw and aggregated metrics:

```
|-----------|--------|----------|------|-------|
| Partition | Nsamp  | Nfeat    | R2   | RMSE  |
|-----------|--------|----------|------|-------|
| Cros Val  | 400    | 200      | 0.87 | 0.712 |
| Cros Val* | 100    | 200      | 0.92 | 0.598 |  <- Aggregated
| Test      | 100    | 200      | 0.85 | 0.756 |
| Test*     | 25     | 200      | 0.90 | 0.632 |  <- Aggregated
|-----------|--------|----------|------|-------|
* Aggregated by sample_id
```

The asterisk (`*`) rows show performance when predictions for repeated measurements are averaged before computing metrics.

## Visualization with Aggregation

### Automatic Aggregation via `default_aggregate`

When you set `default_aggregate` on the analyzer, all visualization methods use it automatically:

```python
# Get aggregate setting from last run
analyzer = PredictionAnalyzer(predictions, default_aggregate=runner.last_aggregate)

# All these plots use aggregation automatically
fig1 = analyzer.plot_top_k(k=5)
fig2 = analyzer.plot_histogram()
fig3 = analyzer.plot_heatmap('model_name', 'preprocessings')
fig4 = analyzer.plot_candlestick('model_name')
```

### Overriding the Default

You can override the default for specific plots:

```python
# Use default aggregation
fig1 = analyzer.plot_top_k(k=5)

# Override: disable aggregation for this plot
fig2 = analyzer.plot_top_k(k=5, aggregate='')

# Override: use different aggregation column
fig3 = analyzer.plot_top_k(k=5, aggregate='batch_id')
```

### Manual Aggregation per Plot

Without setting a default, specify aggregation per method call:

```python
analyzer = PredictionAnalyzer(predictions)

# Explicit aggregation
fig = analyzer.plot_top_k(k=5, aggregate='sample_id')
fig = analyzer.plot_heatmap('model', 'preprocessing', aggregate='sample_id')
```

## Multi-Dataset Aggregation

Different datasets can have different aggregation columns:

```python
config1 = {
    "train_x": "dataset1/spectra.csv",
    "train_y": "dataset1/targets.csv",
    "train_m": "dataset1/metadata.csv",
    "aggregate": "sample_id"  # Dataset 1 uses sample_id
}

config2 = {
    "train_x": "dataset2/spectra.csv",
    "train_y": "dataset2/targets.csv",
    "train_m": "dataset2/metadata.csv",
    "aggregate": "batch_number"  # Dataset 2 uses batch_number
}

dataset = DatasetConfigs([config1, config2])
```

Alternatively, use a list of aggregate values:

```python
dataset = DatasetConfigs(
    [config1, config2],
    aggregate=["sample_id", "batch_number"]
)
```

## Priority Resolution

When aggregation is specified in multiple places, the priority order is:

1. **Constructor parameter** (highest priority)
2. **Config dictionary** (lower priority)

```python
config = {
    "train_x": "...",
    "aggregate": "sample_id"  # Config-level setting
}

# Constructor parameter overrides config dict
dataset = DatasetConfigs(config, aggregate="batch_id")  # Uses "batch_id"
```

## Aggregation Algorithm

For **regression** tasks:
- Predictions for samples in the same group are averaged
- y_true values are also averaged (for consistent comparison)

For **classification** tasks:
- Probabilities (if available) are averaged, then argmax is applied
- Without probabilities, majority voting is used

## Complete Example

```python
"""
Example: Soil Analysis with Multiple Scans per Sample
Each soil sample has 4 spectral scans to reduce measurement noise.
"""

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.visualization.predictions import PredictionAnalyzer
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Dataset config with aggregation
dataset = DatasetConfigs(
    {
        "train_x": "soil_data/spectra_train.csv",
        "train_y": "soil_data/targets_train.csv",
        "train_m": "soil_data/metadata_train.csv",  # Has 'sample_id' column
        "test_x": "soil_data/spectra_test.csv",
        "test_y": "soil_data/targets_test.csv",
        "test_m": "soil_data/metadata_test.csv",
    },
    aggregate="sample_id"
)

# Pipeline with hyperparameter search
pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
]

# Add models with different n_components
for n in [5, 10, 15, 20]:
    pipeline.append({"model": PLSRegression(n_components=n)})

# Run
runner = PipelineRunner(verbose=1)
predictions, _ = runner.run(PipelineConfigs(pipeline, "SoilPLS"), dataset)

# Analyze with aggregation
analyzer = PredictionAnalyzer(predictions, default_aggregate=runner.last_aggregate)

# All visualizations use aggregated metrics
fig1 = analyzer.plot_top_k(k=3, rank_metric='rmse')
fig2 = analyzer.plot_heatmap('model_name', 'preprocessings')

plt.show()
```

## See Also

- {doc}`/reference/predictions_api` - Predictions API reference
- {doc}`/user_guide/visualization/prediction_charts` - Visualization methods
- {doc}`/getting_started/index` - Quick start guide
