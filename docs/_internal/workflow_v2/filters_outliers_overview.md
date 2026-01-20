# Filters and Outliers in nirs4all (v2)

## Overview

nirs4all provides a **non-destructive filtering system** for identifying and handling outlier samples. The v2 system introduces two distinct operations:

- **`tag`** - Mark samples without removal (for analysis/branching)
- **`exclude`** - Mark samples AND exclude from training

Samples are marked in the dataset's indexer rather than removed, preserving an audit trail and allowing re-inclusion if needed.

## Core Architecture

### SampleFilter Base Class

All filters inherit from `SampleFilter` (`operators/filters/base.py`):

- `fit(X, y)` - Learn thresholds from training data
- `get_mask(X, y)` - Returns boolean array: `True` = keep, `False` = exclude
- `transform(X)` - No-op (filtering happens at indexer level)
- `tag_name` - Optional custom name for the tag column
- Stores `exclusion_reason` string for audit trail

### Non-Destructive Design

Filters don't remove data. Instead, controllers mark samples in `dataset._indexer`:

```
Raw samples -> Filter computes mask -> Samples marked in indexer -> Downstream steps respect marks
```

This design enables:
- Easy re-inclusion via `dataset._indexer.mark_included(indices)`
- Audit trail with per-sample exclusion reasons
- Cascading exclusion to augmented samples
- Branching by tag values

## Filter Types

### Y-Based Outliers (`YOutlierFilter`)

Identifies samples with extreme target values.

| Method | Logic | Default Threshold |
|--------|-------|-------------------|
| `iqr` | Outside Q1 - 1.5xIQR to Q3 + 1.5xIQR | 1.5 |
| `zscore` | |z| > threshold | 3.0 |
| `mad` | |y - median| / MAD > threshold | 3.5 |
| `percentile` | Outside fixed percentile bounds | 1, 99 |

### X-Based Outliers (`XOutlierFilter`)

Identifies samples with unusual spectral patterns.

| Method | Logic |
|--------|-------|
| `mahalanobis` | Distance in covariance space > threshold x std |
| `robust_mahalanobis` | Uses MinCovDet, resistant to outliers in training |
| `pca_residual` | Q-statistic (reconstruction error) exceeds limit |
| `pca_leverage` | Hotelling's T^2 in reduced space |
| `isolation_forest` | Anomaly score via random partitioning |
| `lof` | Local Outlier Factor (density-based) |

Auto-reduces dimensionality via PCA when n_features > n_samples.

### Spectral Quality (`SpectralQualityFilter`)

Fixed-threshold checks for data quality issues:

- **NaN ratio** - max percentage of missing values
- **Zero ratio** - max percentage of zeros (dead spectra)
- **Variance** - minimum variance (flat spectra)
- **Value range** - optional min/max bounds (saturation)

No fitting required - uses fixed parameters.

### High Leverage (`HighLeverageFilter`)

Identifies samples with disproportionate model influence.

| Method | Computation |
|--------|-------------|
| `hat` | Hat matrix diagonal: h_ii = x_i'(X'X)^-1 x_i |
| `pca` | Leverage in PCA-reduced space |

Threshold: typically 2-3x average leverage (p/n).

### Metadata Filter (`MetadataFilter`)

Filters based on non-spectral columns:
- Custom condition lambda
- Explicit values to exclude
- Explicit values to keep

### Composite Filter (`CompositeFilter`)

Combines multiple filters:
- `mode="any"` - exclude if ANY filter flags (stricter)
- `mode="all"` - exclude only if ALL filters flag (lenient)

## Pipeline Integration (v2)

### TagController

Marks samples with tags for analysis or downstream branching:

```python
from nirs4all.operators.filters import YOutlierFilter, XOutlierFilter

# Single filter - creates tag named "y_outlier_iqr"
{"tag": YOutlierFilter(method="iqr")},

# Multiple filters - each creates its own tag
{"tag": [YOutlierFilter(method="iqr"), XOutlierFilter(method="mahalanobis")]},

# Named tags
{"tag": {
    "extreme_y": YOutlierFilter(method="zscore", threshold=3),
    "anomalous_x": XOutlierFilter(method="isolation_forest"),
}},
```

Key behaviors:
- Runs in both train and prediction mode (tags computed fresh on prediction data)
- Does NOT exclude samples - only marks them
- Tags stored in dataset indexer for querying
- Priority 5 (runs early)

### ExcludeController

Excludes samples from training:

```python
from nirs4all.operators.filters import YOutlierFilter, XOutlierFilter

# Single filter
{"exclude": YOutlierFilter(method="iqr")},

# Multiple filters with mode
{"exclude": [YOutlierFilter(), XOutlierFilter(method="mahalanobis")], "mode": "any"},
```

Key behaviors:
- Runs only in train mode (NOT during prediction)
- Gets base training samples only (excludes augmented)
- Marks excluded samples in indexer with reason (tag with `excluded_` prefix)
- Cascades to augmented samples by default
- Priority 5 (runs early, before augmentation)

### Separation Branches with Tags

Branch by tag values to create disjoint sample subsets:

```python
# First tag samples
{"tag": YOutlierFilter(method="iqr", tag_name="y_outlier")},

# Then branch by tag
{"branch": {
    "by_tag": "y_outlier",
    "values": {
        "clean": False,
        "outliers": True,
    }
}},

# Process each branch
PLSRegression(10),

# Reassemble samples
{"merge": "concat"},
```

This replaces the old `OutlierExcluderController` with a more flexible pattern.

### Metadata-Based Branching

Branch by metadata column values:

```python
{"branch": {"by_metadata": "site"}},

# With value mapping
{"branch": {
    "by_metadata": "temperature",
    "values": {
        "cold": "< 20",
        "warm": ">= 20",
    }
}},
```

## Indexer Integration

The `SpectroDataset._indexer` provides the backend:

```python
# Mark samples excluded
dataset._indexer.mark_excluded(indices, reason="y_outlier_iqr", cascade_to_augmented=True)

# Query exclusions
excluded_df = dataset._indexer.get_excluded_samples({"partition": "train"})
summary = dataset._indexer.get_exclusion_summary()
# {"total_excluded": 12, "by_reason": {"y_outlier_iqr": 8, ...}}

# Tag operations (v2)
dataset.add_tag("my_tag", dtype="bool")
dataset.set_tag("my_tag", indices=[0, 1, 2], values=[True, True, False])
dataset.get_tag("my_tag")  # Returns numpy array
```

## Key Design Decisions

1. **Non-destructive** - Mark, don't delete. Enables undo and audit.

2. **Tag vs Exclude** - `tag` for analysis/branching, `exclude` for removal from training.

3. **Train-only exclusion** - Exclude never runs during prediction. Tags are computed fresh.

4. **Sklearn-compatible** - Implements `TransformerMixin` + `BaseEstimator` for interoperability.

5. **Early execution** - Both controllers have priority 5, ensuring filtering before augmentation.

6. **Learned vs fixed thresholds** - Y/X filters learn from data; quality filter uses fixed thresholds.

## Relevant Files

- `nirs4all/operators/filters/` - Filter implementations
- `nirs4all/controllers/data/tag.py` - TagController
- `nirs4all/controllers/data/exclude.py` - ExcludeController
- `nirs4all/controllers/data/branch.py` - BranchController (separation modes)
- `nirs4all/data/indexer.py` - Indexer mark_excluded/get_excluded_samples/tag operations
- `tests/unit/operators/filters/` - Unit tests
- `tests/unit/controllers/data/test_tag_controller.py` - TagController tests
- `tests/unit/controllers/data/test_exclude_controller.py` - ExcludeController tests

## Migration from v1

See `migration_guide_v2.md` for detailed migration instructions.

Quick reference:
| Old (v1) | New (v2) |
|----------|----------|
| `{"sample_filter": {...}}` | `{"exclude": ...}` |
| `OutlierExcluderController` | `{"tag": ...}` + `{"branch": {"by_tag": ...}}` |
| `SamplePartitionerController` | `{"branch": {"by_tag": ...}}` |
| `MetadataPartitionerController` | `{"branch": {"by_metadata": ...}}` |
