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

## Repetition Handling

### Setting Repetition Column

The repetition column identifies which spectra belong to the same physical sample. This is fundamental for proper cross-validation (preventing data leakage) and score aggregation:

```python
from nirs4all.data import DatasetConfigs

# Create dataset
dataset = DatasetConfigs("path/to/data")

# Set repetition column from metadata
dataset.set_repetition('Sample_ID')

# Check current setting
print(f"Repetition column: {dataset.repetition}")
```

The repetition column must exist in your metadata. You can also set it at dataset creation time:

```python
dataset = DatasetConfigs(
    "path/to/data",
    repetition="Sample_ID"
)
```

### Repetition Statistics

Query statistics about how repetitions are distributed across samples:

```python
# Get detailed statistics
stats = dataset.repetition_stats

print(f"Number of unique samples: {stats['n_groups']}")
print(f"Repetitions per sample: {stats['min']}-{stats['max']} (avg: {stats['mean']:.1f})")
print(f"Variable counts: {stats['is_variable']}")
```

The statistics dictionary includes:
- `n_groups`: Number of unique samples
- `min`: Minimum repetitions per sample
- `max`: Maximum repetitions per sample
- `mean`: Average repetitions per sample
- `std`: Standard deviation of repetition counts
- `is_variable`: True if samples have different numbers of repetitions

### Repetition Groups

Access the raw grouping information:

```python
# Get mapping of sample IDs to row indices
groups = dataset.repetition_groups
# Example: {'sample_001': [0, 1, 2, 3], 'sample_002': [4, 5, 6, 7], ...}

# Count samples
n_samples = len(groups)

# Find which samples have most repetitions
max_reps = max(len(v) for v in groups.values())
print(f"Maximum repetitions: {max_reps}")
```

## Configuring Aggregation

### Enable Aggregation

Aggregation determines how predictions are combined when you have multiple spectra per sample:

```python
# Aggregate by metadata column (recommended)
dataset.set_aggregate('Sample_ID')

# Aggregate by target values (y_true)
dataset.set_aggregate(True)

# Disable aggregation
dataset.set_aggregate(None)

# Check current setting
agg_setting = dataset.aggregate
print(f"Aggregation: {agg_setting}")
```

When aggregation is enabled, both raw (per-spectrum) and aggregated (per-sample) metrics are computed and displayed.

### Aggregation Methods

Different aggregation methods are available depending on your task:

| Method | Use Case | Description |
|--------|----------|-------------|
| `mean` | Regression (default) | Average predictions within each sample group |
| `median` | Regression (robust) | Median prediction within each group (outlier-resistant) |
| `robust_mean` | Regression (robust) | Mean after excluding outliers using T² statistic |
| `vote` | Classification | Majority voting or probability averaging |

Set the aggregation method:

```python
# Use median instead of mean
dataset.set_aggregate_method('median')

# Use robust mean (with automatic outlier exclusion)
dataset.set_aggregate_method('robust_mean')

# Check current method
method = dataset.aggregate_method
print(f"Aggregation method: {method}")
```

### Outlier Exclusion

For robust aggregation, you can enable Hotelling's T² based outlier exclusion:

```python
# Enable outlier exclusion with 95% confidence threshold
dataset.set_aggregate_exclude_outliers(True, threshold=0.95)

# Use stricter threshold (99%)
dataset.set_aggregate_exclude_outliers(True, threshold=0.99)

# Disable outlier exclusion
dataset.set_aggregate_exclude_outliers(False)

# Check settings
is_enabled = dataset.aggregate_exclude_outliers
threshold = dataset.aggregate_outlier_threshold
print(f"Outlier exclusion: {is_enabled} (threshold: {threshold})")
```

When enabled, outlier repetitions are identified using the T² statistic and excluded before averaging. This is useful when you have occasional bad measurements within an otherwise good set of repetitions.

## Complete Example

Here's a complete workflow showing repetition handling and aggregation:

```python
import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit

# Load data and configure repetition
dataset = nirs4all.data.DatasetConfigs("soil_samples/")
dataset.set_repetition('Sample_ID')

# Inspect repetition statistics
stats = dataset.repetition_stats
print(f"{stats['n_groups']} samples with {stats['mean']:.1f} reps each")
if stats['is_variable']:
    print(f"Warning: variable repetition counts ({stats['min']}-{stats['max']})")

# Configure aggregation
dataset.set_aggregate('Sample_ID')
dataset.set_aggregate_method('robust_mean')
dataset.set_aggregate_exclude_outliers(True, threshold=0.95)

# Check configuration
print(f"Aggregation: {dataset.aggregate}")
print(f"Method: {dataset.aggregate_method}")
print(f"Outlier exclusion: {dataset.aggregate_exclude_outliers}")

# Run pipeline
result = nirs4all.run(
    pipeline=[
        MinMaxScaler(),
        ShuffleSplit(n_splits=5, test_size=0.2),
        {"model": PLSRegression(n_components=10)}
    ],
    dataset=dataset,
    verbose=1
)

# Results will show both raw and aggregated metrics
print(f"Best RMSE (raw): {result.best_rmse:.4f}")

# Get predictions with aggregation
from nirs4all.visualization.predictions import PredictionAnalyzer
analyzer = PredictionAnalyzer(result.predictions)

# Top models ranked by aggregated metrics
top_models = result.predictions.top(5, rank_metric='rmse', by_repetition='Sample_ID')
for i, model in enumerate(top_models, 1):
    print(f"{i}. {model['model_name']}: RMSE={model['val_score']:.4f}")
```

## Properties and Methods Reference

### Properties

- `dataset.repetition` → Optional[str]

  Get the column name identifying sample repetitions.

- `dataset.repetition_groups` → Dict[Any, List[int]]

  Get sample groups by repetition column (mapping of sample IDs to row indices).

- `dataset.repetition_stats` → Dict[str, Any]

  Get statistics about repetition counts (n_groups, min, max, mean, std, is_variable).

- `dataset.aggregate` → Optional[str]

  Get the aggregation setting ('y' for target-based, column name, or None).

- `dataset.aggregate_method` → str

  Get the aggregation method ('mean', 'median', 'robust_mean', or 'vote').

- `dataset.aggregate_exclude_outliers` → bool

  Get whether T² outlier exclusion is enabled.

- `dataset.aggregate_outlier_threshold` → float

  Get the outlier detection threshold (0-1).

### Methods

- `dataset.set_repetition(column: Optional[str])` → None

  Set the column name identifying sample repetitions. Pass None to disable.

- `dataset.set_aggregate(value: Union[str, bool, None])` → None

  Enable sample-level aggregation. Pass True for y-based aggregation, column name for metadata-based, or None to disable.

- `dataset.set_aggregate_method(value: Optional[str])` → None

  Set aggregation method ('mean', 'median', 'robust_mean', or 'vote').

- `dataset.set_aggregate_exclude_outliers(value: bool, threshold: float = 0.95)` → None

  Enable/disable T² based outlier exclusion before aggregation.

## See Also

- {doc}`/reference/predictions_api` - Predictions API reference
- {doc}`/user_guide/visualization/prediction_charts` - Visualization methods
- {doc}`/getting_started/index` - Quick start guide
- {doc}`signal_types` - Signal type detection and conversion

```{seealso}
**Related Examples:**
- [U04: Repetition Aggregation](../../../examples/user/05_cross_validation/U04_aggregation.py) - Handle repeated measurements with aggregation
- [U02: Group Splitting](../../../examples/user/05_cross_validation/U02_group_splitting.py) - Group-aware cross-validation
- [D03: Repetition Transform](../../../examples/developer/05_advanced_features/D03_repetition_transform.py) - Advanced repetition handling strategies
```
