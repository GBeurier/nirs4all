# Filters and Outliers in nirs4all

## Overview

nirs4all provides a **non-destructive filtering system** for identifying and excluding outlier samples. Samples are marked in the dataset's indexer rather than removed, preserving an audit trail and allowing re-inclusion if needed.

## Core Architecture

### SampleFilter Base Class

All filters inherit from `SampleFilter` (`operators/filters/base.py`):

- `fit(X, y)` — Learn thresholds from training data
- `get_mask(X, y)` — Returns boolean array: `True` = keep, `False` = exclude
- `transform(X)` — No-op (filtering happens at indexer level)
- Stores `exclusion_reason` string for audit trail

### Non-Destructive Design

Filters don't remove data. Instead, the `SampleFilterController` marks samples in `dataset._indexer`:

```
Raw samples → Filter computes mask → Excluded indices marked in indexer → Downstream steps skip marked samples
```

This design enables:
- Easy re-inclusion via `dataset._indexer.mark_included(indices)`
- Audit trail with per-sample exclusion reasons
- Cascading exclusion to augmented samples

## Filter Types

### Y-Based Outliers (`YOutlierFilter`)

Identifies samples with extreme target values.

| Method | Logic | Default Threshold |
|--------|-------|-------------------|
| `iqr` | Outside Q1 - 1.5×IQR to Q3 + 1.5×IQR | 1.5 |
| `zscore` | |z| > threshold | 3.0 |
| `mad` | |y - median| / MAD > threshold | 3.5 |
| `percentile` | Outside fixed percentile bounds | 1, 99 |

### X-Based Outliers (`XOutlierFilter`)

Identifies samples with unusual spectral patterns.

| Method | Logic |
|--------|-------|
| `mahalanobis` | Distance in covariance space > threshold × std |
| `robust_mahalanobis` | Uses MinCovDet, resistant to outliers in training |
| `pca_residual` | Q-statistic (reconstruction error) exceeds limit |
| `pca_leverage` | Hotelling's T² in reduced space |
| `isolation_forest` | Anomaly score via random partitioning |
| `lof` | Local Outlier Factor (density-based) |

Auto-reduces dimensionality via PCA when n_features > n_samples.

### Spectral Quality (`SpectralQualityFilter`)

Fixed-threshold checks for data quality issues:

- **NaN ratio** — max percentage of missing values
- **Zero ratio** — max percentage of zeros (dead spectra)
- **Variance** — minimum variance (flat spectra)
- **Value range** — optional min/max bounds (saturation)

No fitting required — uses fixed parameters.

### High Leverage (`HighLeverageFilter`)

Identifies samples with disproportionate model influence.

| Method | Computation |
|--------|-------------|
| `hat` | Hat matrix diagonal: h_ii = x_i'(X'X)⁻¹x_i |
| `pca` | Leverage in PCA-reduced space |

Threshold: typically 2-3× average leverage (p/n).

### Metadata Filter (`MetadataFilter`)

Filters based on non-spectral columns:
- Custom condition lambda
- Explicit values to exclude
- Explicit values to keep

### Composite Filter (`CompositeFilter`)

Combines multiple filters:
- `mode="any"` — exclude if ANY filter flags (stricter)
- `mode="all"` — exclude only if ALL filters flag (lenient)

## Pipeline Integration

### SampleFilterController

Executes filtering within the pipeline:

```python
{
    "sample_filter": {
        "filters": [YOutlierFilter(...), XOutlierFilter(...)],
        "mode": "any",
        "report": True
    }
}
```

Key behaviors:
- Runs only in train mode (not prediction)
- Gets base training samples only (excludes augmented)
- Marks excluded samples in indexer with reason
- Optional: cascades to augmented samples
- Priority 5 (runs early, before augmentation)

### OutlierExcluderController

Creates parallel branches with different outlier strategies:

```python
{
    "branch": {
        "by": "outlier_excluder",
        "strategies": [None, {"method": "mahalanobis"}, {"method": "isolation_forest"}]
    }
}
```

Useful for comparing outlier handling approaches in a single run.

## Indexer Integration

The `SpectroDataset._indexer` provides the backend:

```python
# Mark samples excluded
dataset._indexer.mark_excluded(indices, reason="y_outlier_iqr", cascade_to_augmented=True)

# Query exclusions
excluded_df = dataset._indexer.get_excluded_samples({"partition": "train"})
summary = dataset._indexer.get_exclusion_summary()
# {"total_excluded": 12, "by_reason": {"y_outlier_iqr": 8, ...}}
```

## Key Design Decisions

1. **Non-destructive** — Mark, don't delete. Enables undo and audit.

2. **Train-only** — Filters never exclude during prediction. Prediction samples always processed.

3. **Sklearn-compatible** — Implements `TransformerMixin` + `BaseEstimator` for interoperability.

4. **Early execution** — Controller priority 5 ensures filtering before augmentation.

5. **Learned vs fixed thresholds** — Y/X filters learn from data; quality filter uses fixed thresholds.

## Relevant Files

- `nirs4all/operators/filters/` — Filter implementations
- `nirs4all/controllers/data/sample_filter.py` — SampleFilterController
- `nirs4all/controllers/data/outlier_excluder.py` — OutlierExcluderController
- `nirs4all/data/indexer.py` — Indexer mark_excluded/get_excluded_samples
- `tests/unit/operators/filters/` — Unit tests
