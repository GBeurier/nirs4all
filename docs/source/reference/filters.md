# Filters Reference

Filters identify and mark samples for exclusion or tagging based on statistical criteria. They are non-destructive -- they mark samples in the dataset indexer rather than removing data.

## Usage in Pipeline

Filters are used with the `tag` (mark without removing) or `exclude` (remove from training) keywords:

```python
from nirs4all.operators.filters import YOutlierFilter, XOutlierFilter

# Tag samples (mark for analysis, do not remove)
pipeline = [
    {"tag": YOutlierFilter(method="iqr", threshold=1.5)},
    SNV(),
    {"model": PLSRegression(n_components=10)},
]

# Exclude samples from training
pipeline = [
    {"exclude": YOutlierFilter(method="iqr", threshold=1.5)},
    SNV(),
    {"model": PLSRegression(n_components=10)},
]

# Multiple filters with mode
pipeline = [
    {"exclude": [
        YOutlierFilter(method="iqr"),
        XOutlierFilter(method="mahalanobis", threshold=3.0),
    ], "mode": "any"},  # Exclude if ANY filter flags the sample
    SNV(),
    {"model": PLSRegression(n_components=10)},
]
```

All filters below are imported from `nirs4all.operators.filters`.

---

## YOutlierFilter

Detects samples with outlier target (y) values.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `"iqr"` | Detection method: `"iqr"`, `"zscore"`, `"percentile"`, `"mad"` |
| `threshold` | `1.5` | Method-specific threshold (see below) |
| `lower_percentile` | `1.0` | Lower cutoff for percentile method |
| `upper_percentile` | `99.0` | Upper cutoff for percentile method |
| `reason` | `None` | Custom reason string for reports |
| `tag_name` | `None` | Custom tag name (defaults to auto-generated) |

### Method Details

| Method | Threshold Meaning | Typical Values |
|--------|-------------------|----------------|
| `"iqr"` | Multiplier of the interquartile range | 1.5 (mild), 3.0 (extreme) |
| `"zscore"` | Number of standard deviations from mean | 2.0-3.0 |
| `"percentile"` | Uses `lower_percentile` and `upper_percentile` instead | 1-99, 5-95 |
| `"mad"` | Multiplier of Median Absolute Deviation | 3.0-3.5 |

---

## XOutlierFilter

Detects samples with outlier spectral features (X values).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `"mahalanobis"` | Detection method (see below) |
| `threshold` | `None` | Detection threshold (auto-computed if None for some methods) |
| `n_components` | `None` | Number of PCA components (for PCA-based methods) |
| `contamination` | `0.1` | Expected outlier proportion (for sklearn methods) |
| `random_state` | `None` | Random seed for reproducibility |
| `support_fraction` | `None` | Support fraction for robust covariance estimation |
| `reason` | `None` | Custom reason string for reports |
| `tag_name` | `None` | Custom tag name |

### Method Details

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| `"mahalanobis"` | Mahalanobis distance from center | `threshold` (default 3.0) |
| `"robust_mahalanobis"` | Robust Mahalanobis using MinCovDet | `threshold`, `support_fraction` |
| `"pca_residual"` | Q-statistic (squared reconstruction error) | `n_components`, `threshold` |
| `"pca_leverage"` | Hotelling's T-squared in PCA score space | `n_components`, `threshold` |
| `"isolation_forest"` | Isolation Forest anomaly detection | `contamination`, `random_state` |
| `"lof"` | Local Outlier Factor | `contamination` |

---

## SpectralQualityFilter

Detects samples with poor spectral quality.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_nan_ratio` | `0.1` | Maximum allowed NaN ratio per spectrum (0-1) |
| `max_zero_ratio` | `0.5` | Maximum allowed zero-value ratio |
| `min_variance` | `1e-8` | Minimum variance threshold (flags flat spectra) |
| `max_value` | `None` | Maximum allowed value (saturation detection) |
| `min_value` | `None` | Minimum allowed value |
| `check_inf` | `True` | Whether to check for infinite values |
| `reason` | `None` | Custom reason string |
| `tag_name` | `None` | Custom tag name |

---

## HighLeverageFilter

Detects high-leverage samples that may unduly influence model fitting.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `"hat"` | Leverage computation: `"hat"` (direct hat matrix) or `"pca"` (PCA-based) |
| `threshold_multiplier` | `2.0` | Multiple of average leverage used as threshold |
| `absolute_threshold` | `None` | Absolute threshold (overrides multiplier if set) |
| `n_components` | `None` | Number of PCA components (for `"pca"` method) |

Common threshold guidelines:
- `threshold_multiplier=2.0` -- 2x average leverage (standard rule)
- `threshold_multiplier=3.0` -- 3x average leverage (conservative)
- `absolute_threshold=0.5` -- Fixed absolute threshold

---

## MetadataFilter

Filters samples based on metadata column values.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `column` | *(required)* | Metadata column name to filter on |
| `condition` | `None` | Callable returning True for samples to KEEP |
| `values_to_exclude` | `None` | List of values that should be excluded |
| `values_to_keep` | `None` | List of values that should be kept |

Usage examples:

```python
# Exclude specific values
MetadataFilter(column="quality_flag", values_to_exclude=["bad", "corrupted"])

# Keep only specific values
MetadataFilter(column="sample_type", values_to_keep=["control", "treatment"])

# Custom condition
MetadataFilter(column="temperature", condition=lambda x: 20 <= x <= 30)
```

---

## CompositeFilter

Combines multiple filters with AND/OR logic.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `filters` | *(required)* | List of `SampleFilter` instances |
| `mode` | `"any"` | Combination logic: `"any"` (OR) or `"all"` (AND) |

```python
from nirs4all.operators.filters import CompositeFilter, YOutlierFilter, XOutlierFilter

composite = CompositeFilter(
    filters=[
        YOutlierFilter(method="iqr"),
        XOutlierFilter(method="mahalanobis"),
    ],
    mode="any",  # Exclude if ANY filter flags the sample
)
```

---

## Tag vs Exclude Keywords

| Keyword | Behavior |
|---------|----------|
| `{"tag": filter}` | Marks samples with a tag for downstream analysis or branching. Samples are NOT removed from training. |
| `{"exclude": filter}` | Removes flagged samples from training data. Test samples are never excluded. |
| `{"exclude": [f1, f2], "mode": "any"}` | Multiple filters combined: `"any"` excludes if any filter flags, `"all"` excludes only if all filters flag. |

---

## See Also

- {doc}`../reference/pipeline_keywords` -- Full pipeline keyword reference (including `tag` and `exclude`)
- {doc}`../reference/splitters` -- Cross-validation splitters
- {doc}`../reference/transforms` -- Preprocessing transforms
