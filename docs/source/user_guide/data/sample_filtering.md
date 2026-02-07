# Sample Filtering User Guide

## Overview

Sample filtering in nirs4all provides a non-destructive mechanism for identifying and excluding problematic samples from training datasets. Unlike traditional data deletion, sample filtering marks samples as "excluded" in the indexer while preserving the original data, allowing for:

- **Reversibility**: Exclusions can be undone at any time
- **Auditability**: Full tracking of what was excluded and why
- **Non-destructive operations**: Original data remains intact

## Quick Start

### Basic Usage in Pipeline

```python
from nirs4all.operators.filters import YOutlierFilter
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.data import DatasetConfigs
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression

pipeline = [
    "chart_y",  # Visualize y distribution before filtering
    {
        "sample_filter": {
            "filters": [YOutlierFilter(method="iqr", threshold=1.5)],
            "report": True,  # Print filtering report
        }
    },
    "chart_y",  # Visualize y distribution after filtering
    "snv",
    {"split": KFold(n_splits=5)},
    {"model": PLSRegression(n_components=5)},
]

config = PipelineConfigs(pipeline, name="filtered_pipeline")
runner = PipelineRunner()
runner.run(config, DatasetConfigs("my_dataset"))
```

### Programmatic Usage

```python
from nirs4all.data import SpectroDataset
from nirs4all.operators.filters import YOutlierFilter

# Create or load dataset
dataset = SpectroDataset("my_dataset")
# ... add samples and targets ...

# Create filter
filter_obj = YOutlierFilter(method="iqr", threshold=1.5)

# Get training data
selector = {"partition": "train"}
X = dataset.x(selector, layout="2d", include_augmented=False)
y = dataset.y(selector, include_augmented=False)
sample_indices = dataset._indexer.x_indices(selector, include_augmented=False)

# Fit and get mask
filter_obj.fit(X, y)
mask = filter_obj.get_mask(X, y)

# Mark excluded samples
exclude_indices = sample_indices[~mask].tolist()
n_excluded = dataset._indexer.mark_excluded(exclude_indices, reason="iqr_outlier")
print(f"Excluded {n_excluded} samples")
```

## Available Filters

### Y-based Outlier Filters (YOutlierFilter)

Detect outliers based on target (y) values.

| Method | Description | Recommended Threshold |
|--------|-------------|----------------------|
| `iqr` | Interquartile Range | 1.5 (mild) to 3.0 (extreme) |
| `zscore` | Standard deviations from mean | 2.0 to 3.0 |
| `percentile` | Direct percentile cutoffs | 1-99 or 5-95 |
| `mad` | Median Absolute Deviation (robust) | 3.0 to 3.5 |

**Examples:**

```python
# IQR method (default, robust to outliers)
filter_iqr = YOutlierFilter(method="iqr", threshold=1.5)

# Z-score method (assumes normal distribution)
filter_zscore = YOutlierFilter(method="zscore", threshold=3.0)

# Percentile method (exclude extreme 2%)
filter_pct = YOutlierFilter(method="percentile", lower_percentile=1, upper_percentile=99)

# MAD method (most robust to outliers)
filter_mad = YOutlierFilter(method="mad", threshold=3.5)
```

### X-based Outlier Filters (XOutlierFilter)

Detect outliers based on feature (spectral) patterns.

| Method | Description | Best For |
|--------|-------------|----------|
| `mahalanobis` | Distance from center in feature space | General use |
| `robust_mahalanobis` | Robust Mahalanobis (MinCovDet) | Data with existing outliers |
| `pca_residual` | Q-statistic from PCA reconstruction | High-dimensional data |
| `pca_leverage` | Hotelling's T² in PCA space | High leverage samples |
| `isolation_forest` | Ensemble anomaly detection | Complex patterns |
| `lof` | Local Outlier Factor | Local density anomalies |

**Examples:**

```python
from nirs4all.operators.filters import XOutlierFilter

# Mahalanobis distance
filter_maha = XOutlierFilter(method="mahalanobis", threshold=3.0)

# PCA residual (for high-dimensional spectra)
filter_pca = XOutlierFilter(method="pca_residual", n_components=10)

# Isolation Forest (ML-based)
filter_iso = XOutlierFilter(method="isolation_forest", contamination=0.05)
```

### Spectral Quality Filter (SpectralQualityFilter)

Filter samples based on data quality metrics.

```python
from nirs4all.operators.filters import SpectralQualityFilter

# Default quality checks
filter_quality = SpectralQualityFilter()

# Strict quality requirements
filter_strict = SpectralQualityFilter(
    max_nan_ratio=0.01,      # Max 1% NaN allowed
    max_zero_ratio=0.1,      # Max 10% zeros allowed
    min_variance=1e-4,       # Minimum spectral variance
    max_value=4.0,           # Maximum absorbance (saturation)
    min_value=-0.5,          # Minimum absorbance
)
```

### High Leverage Filter (HighLeverageFilter)

Filter samples with high influence on model fitting.

```python
from nirs4all.operators.filters import HighLeverageFilter

# Using multiplier of average leverage
filter_leverage = HighLeverageFilter(threshold_multiplier=2.0)

# Using absolute threshold
filter_leverage_abs = HighLeverageFilter(absolute_threshold=0.5)

# PCA-based for high-dimensional data
filter_leverage_pca = HighLeverageFilter(method="pca", n_components=10)
```

### Metadata Filter (MetadataFilter)

Filter samples based on metadata column values.

```python
from nirs4all.operators.filters import MetadataFilter

# Exclude specific values
filter_meta = MetadataFilter(
    column="quality_flag",
    values_to_exclude=["bad", "corrupted", "uncertain"]
)

# Keep only specific values
filter_meta_keep = MetadataFilter(
    column="sample_type",
    values_to_keep=["control", "treatment"]
)

# Custom condition
filter_custom = MetadataFilter(
    column="temperature",
    condition=lambda x: 20 <= x <= 30  # Keep samples at 20-30°C
)
```

## Combining Filters

### Using CompositeFilter

```python
from nirs4all.operators.filters import YOutlierFilter, CompositeFilter

# Combine multiple filters
composite = CompositeFilter(
    filters=[
        YOutlierFilter(method="iqr", threshold=1.5),
        YOutlierFilter(method="zscore", threshold=3.0),
    ],
    mode="any"  # Exclude if ANY filter flags the sample
)

# Mode options:
# - "any": Exclude if ANY filter flags (stricter)
# - "all": Exclude only if ALL filters flag (more lenient)
```

### In Pipeline

```python
pipeline = [
    {
        "sample_filter": {
            "filters": [
                YOutlierFilter(method="iqr", threshold=1.5),
                SpectralQualityFilter(max_nan_ratio=0.05),
            ],
            "mode": "any",  # Combination mode
            "report": True,
            "cascade_to_augmented": True,  # Also exclude augmented versions
        }
    },
    # ... rest of pipeline
]
```

## Pipeline Integration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `filters` | list | required | List of SampleFilter instances |
| `mode` | str | "any" | How to combine filters: "any" or "all" |
| `report` | bool | False | Print filtering report |
| `cascade_to_augmented` | bool | True | Exclude augmented samples from excluded bases |

## Managing Exclusions

### Viewing Exclusion Status

```python
# Get exclusion summary
summary = dataset._indexer.get_exclusion_summary()
print(f"Total excluded: {summary['total_excluded']}")
print(f"Exclusion rate: {summary['exclusion_rate']:.1%}")
print(f"By reason: {summary['by_reason']}")
print(f"By partition: {summary['by_partition']}")

# Get excluded samples as DataFrame
excluded_df = dataset._indexer.get_excluded_samples()
print(excluded_df)

# Get excluded samples for specific partition
train_excluded = dataset._indexer.get_excluded_samples({"partition": "train"})
```

### Reverting Exclusions

```python
# Re-include specific samples
n_included = dataset._indexer.mark_included([0, 1, 2])

# Re-include all excluded samples
n_reset = dataset._indexer.reset_exclusions()

# Reset only train partition
n_reset = dataset._indexer.reset_exclusions({"partition": "train"})
```

### Including Excluded Samples in Queries

```python
# By default, excluded samples are filtered out
indices = dataset._indexer.x_indices({"partition": "train"})  # Excludes marked samples

# Explicitly include excluded samples
all_indices = dataset._indexer.x_indices(
    {"partition": "train"},
    include_excluded=True
)
```

## Filtering Reports

### Using FilteringReportGenerator

```python
from nirs4all.operators.filters import FilteringReportGenerator

# Create report generator
report_gen = FilteringReportGenerator(dataset)

# Generate report for filters
report = report_gen.create_report(
    filters=[YOutlierFilter(method="iqr"), YOutlierFilter(method="zscore")],
    X=X_train,
    y=y_train,
    sample_indices=sample_indices,
    mode="any",
    partition="train",
    dry_run=True,  # Don't actually mark samples
)

# Print report
report.print_report(verbose=2)

# Export as JSON
json_report = report.to_json()
with open("filtering_report.json", "w") as f:
    f.write(json_report)
```

### Comparing Filters

```python
# Compare how different filters would affect the data
comparison = report_gen.compare_filters(
    filters=[
        YOutlierFilter(method="iqr", threshold=1.5),
        YOutlierFilter(method="zscore", threshold=3.0),
        YOutlierFilter(method="mad", threshold=3.5),
    ],
    X=X_train,
    y=y_train,
)

print(f"Individual results: {comparison['individual']}")
print(f"Overlap analysis: {comparison['overlap']}")
print(f"Unique exclusions: {comparison['unique_exclusions']}")
```

## Visualization

### Exclusion Charts

```python
pipeline = [
    {
        "sample_filter": {
            "filters": [YOutlierFilter(method="iqr")],
            "report": True,
        }
    },
    # Color by inclusion status
    {"exclusion_chart": {"color_by": "status"}},

    # Color by target value
    {"exclusion_chart": {"color_by": "y"}},

    # Color by exclusion reason
    {"exclusion_chart": {"color_by": "reason"}},
]
```

### Including Excluded Samples in Existing Charts

```python
pipeline = [
    {
        "sample_filter": {
            "filters": [YOutlierFilter(method="iqr")],
        }
    },
    # Show excluded samples with highlighting
    {"chart_y": {"include_excluded": True, "highlight_excluded": True}},
    {"chart_2d": {"include_excluded": True, "highlight_excluded": True}},
]
```

## Best Practices

### 1. Start Conservative

Begin with lenient thresholds and tighten as needed:

```python
# Start with mild outlier detection
filter_obj = YOutlierFilter(method="iqr", threshold=3.0)  # Very lenient

# After inspection, tighten if needed
filter_obj = YOutlierFilter(method="iqr", threshold=1.5)  # Standard
```

### 2. Use Dry Runs

Preview filtering effects before applying:

```python
report = report_gen.create_report(
    filters=[...],
    dry_run=True  # Don't actually exclude
)
report.print_report()
```

### 3. Document Exclusion Reasons

Always provide clear reasons:

```python
dataset._indexer.mark_excluded(
    sample_indices,
    reason="y_outlier_iqr_1.5"  # Clear, searchable reason
)
```

### 4. Consider Augmented Samples

When excluding base samples, their augmented versions should typically also be excluded to prevent data leakage:

```python
{
    "sample_filter": {
        "filters": [...],
        "cascade_to_augmented": True,  # Default and recommended
    }
}
```

### 5. Filter Before Splitting

Apply filtering before cross-validation to ensure consistent treatment:

```python
pipeline = [
    {"sample_filter": {...}},  # Filter first
    {"split": KFold(n_splits=5)},  # Then split
    {"model": PLSRegression()},
]
```

### 6. Combine Multiple Criteria

Use composite filters for robust outlier detection:

```python
{
    "sample_filter": {
        "filters": [
            YOutlierFilter(method="iqr"),
            SpectralQualityFilter(max_nan_ratio=0.05),
            XOutlierFilter(method="mahalanobis"),
        ],
        "mode": "any",
    }
}
```

## Edge Cases and Warnings

### Empty Datasets

Filters gracefully handle empty datasets:
```python
# Returns empty mask without errors
mask = filter_obj.get_mask(np.array([]).reshape(0, 100), np.array([]))
```

### All Samples Excluded

If filtering would exclude all samples, a warning is issued and at least one sample is preserved:
```
UserWarning: Sample filtering would exclude ALL 50 samples.
Consider adjusting filter thresholds. Keeping at least one sample.
```

### Single Sample

Filters handle single samples gracefully, typically keeping them:
```python
filter_obj.fit(X_single, y_single)  # Works without error
mask = filter_obj.get_mask(X_single, y_single)  # Returns [True]
```

## Troubleshooting

### Filter Not Detecting Outliers

- Check that y values have sufficient variance
- Verify filter is fitted before calling get_mask()
- Try different detection methods or thresholds
- Inspect filter statistics: `filter_obj.get_filter_stats(X, y)`

### Too Many Samples Excluded

- Increase threshold values
- Change from "any" to "all" mode in composite filters
- Review exclusion summary to identify aggressive filters

### Metadata Filter Not Working

- Ensure metadata is passed to get_mask(): `filter.get_mask(X, metadata=metadata)`
- Verify column name exists in metadata
- Check that metadata length matches sample count

## API Reference

### SampleFilter Base Class

All filters inherit from `SampleFilter` and provide:

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Learn filter parameters from training data |
| `get_mask(X, y)` | Return boolean mask (True = keep) |
| `get_excluded_indices(X, y)` | Get indices of samples to exclude |
| `get_kept_indices(X, y)` | Get indices of samples to keep |
| `get_filter_stats(X, y)` | Get filtering statistics |
| `transform(X)` | No-op (filtering happens at indexer level) |

### Indexer Methods

| Method | Description |
|--------|-------------|
| `mark_excluded(indices, reason, cascade)` | Mark samples as excluded |
| `mark_included(indices, cascade)` | Remove exclusion flag |
| `get_excluded_samples(selector)` | Get excluded samples DataFrame |
| `get_exclusion_summary()` | Get summary statistics |
| `reset_exclusions(selector)` | Reset all exclusions |

## See Also

- {doc}`/user_guide/augmentation/sample_augmentation_guide` - Adding synthetic samples
- {doc}`/user_guide/preprocessing/handbook` - Data preprocessing options
- {doc}`/reference/operator_catalog` - Complete operator reference

```{seealso}
**Related Examples:**
- [U03: Sample Exclusion](../../../examples/user/05_cross_validation/U03_sample_filtering.py) - Outlier detection with exclude keyword
- [U05: Tagging Analysis](../../../examples/user/05_cross_validation/U05_tagging_analysis.py) - Mark samples with tag for analysis
- [U06: Exclusion Strategies](../../../examples/user/05_cross_validation/U06_exclusion_strategies.py) - Advanced exclusion strategies
```
