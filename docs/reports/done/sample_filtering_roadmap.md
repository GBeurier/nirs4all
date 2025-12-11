# Sample Filtering Implementation Roadmap

## Executive Summary

This document outlines the analysis, recommendations, and implementation roadmap for adding **sample filtering** capabilities to nirs4all. Sample filtering complements the existing feature selection (resampler, CARS, MC-UVE), sample augmentation, and feature augmentation mechanisms by enabling the removal of unwanted samples (outliers, corrupted data, etc.) from training datasets.

---

## 1. Current Architecture Analysis

### 1.1 Existing Data Manipulation Patterns

nirs4all currently supports three types of data manipulation:

| Capability | Axis | Direction | Implementation |
|------------|------|-----------|----------------|
| **Feature Selection** | Columns (features) | Reduction | `Resampler`, `CARS`, `MCUVE`, `CropTransformer`, `ResampleTransformer` |
| **Sample Augmentation** | Rows (samples) | Addition | `SampleAugmentationController`, `Augmenter` classes |
| **Feature Augmentation** | Columns (features) | Addition | `feature_augmentation` keyword in pipeline |

**Missing**: Sample filtering (row reduction).

### 1.2 Key Components Understanding

#### 1.2.1 Indexer Architecture

The `Indexer` class ([indexer.py](nirs4all/data/indexer.py)) is the central component for sample management. It uses:

- **Polars DataFrame** for efficient index storage
- **Component-based architecture**: `IndexStore`, `QueryBuilder`, `SampleManager`, `AugmentationTracker`, `ProcessingManager`
- **Sample tracking via**: `sample`, `origin`, `partition`, `group`, `branch`, `augmentation`, `processings`

Key observation: The indexer currently has no concept of "excluded" or "filtered" samples.

#### 1.2.2 Data Flow in Pipeline

```
Dataset.add_samples()
    → Indexer.add_samples()
    → Features.add_samples()
    → Targets.add_targets()

Pipeline execution:
    → context.selector (partition, fold, etc.)
    → Indexer.x_indices() / y_indices()
    → Dataset.x() / y()
```

The selector mechanism already filters by `partition`, `group`, `branch`, `augmentation`. This is the natural extension point.

#### 1.2.3 Sample Augmentation Pattern

The `SampleAugmentationController` provides a good template:
- Uses `TransformerMixin` pattern for augmentation operators
- Modifies dataset by **adding** samples via `dataset.add_samples_batch()`
- Tracks augmented samples via `origin` field in indexer
- Operates only on `partition="train"` to prevent data leakage

---

## 2. Design Options Analysis

### 2.1 Option A: Indexer Flag Approach (Tag-based)

**Concept**: Add an `excluded` boolean column to the indexer. Samples marked as `excluded=True` are automatically filtered from `x_indices()` and `y_indices()` calls.

**Pros**:
- Non-destructive: original data remains intact
- Reversible: samples can be "un-filtered" by toggling the flag
- Auditable: easy to inspect which samples were filtered and why
- Compatible with existing architecture

**Cons**:
- Increases indexer complexity
- Memory overhead (excluded samples still in memory)
- Requires propagation through all indexer methods

**Implementation sketch**:
```python
# In IndexStore schema
"excluded": pl.Boolean  # or "status": pl.Categorical for more states

# In QueryBuilder.build()
# Auto-inject: pl.col("excluded") == False

# In Indexer
def mark_excluded(self, sample_indices: List[int], reason: str = None):
    """Mark samples as excluded."""

def get_excluded(self, selector: Selector) -> np.ndarray:
    """Get excluded sample indices."""
```

### 2.2 Option B: Physical Removal Approach

**Concept**: Actually remove samples from the underlying data structures (Features, Targets, Indexer).

**Pros**:
- Clean data structures
- Lower memory footprint
- Simpler querying (no filter overhead)

**Cons**:
- Destructive: cannot recover removed samples
- Index renumbering required (breaks references)
- Breaks `origin` references for augmented samples
- More complex implementation (cascade deletion)

**Not recommended** due to complexity and data loss risks.

### 2.3 Option C: Virtual Partition Approach

**Concept**: Introduce a special partition value (e.g., `"excluded"`) where filtered samples are moved.

**Pros**:
- Uses existing partition mechanism
- Non-destructive
- Simple implementation

**Cons**:
- Mixes filtering semantics with data splitting semantics
- May conflict with future partition requirements
- Less explicit than a dedicated flag

### 2.4 Option D: Mask Array Approach

**Concept**: Store a separate mask array at the dataset level, applied during `x()` and `y()` retrieval.

**Pros**:
- Completely decoupled from indexer
- Very simple implementation

**Cons**:
- External to indexer (inconsistency)
- Doesn't integrate with existing selector mechanism
- Harder to track multiple filtering reasons

---

## 3. Recommended Approach: Hybrid Flag + Metadata

**Primary mechanism**: Add an `excluded` boolean column to the indexer (Option A)
**Secondary mechanism**: Add exclusion metadata for auditability

### 3.1 Core Design Decisions

1. **Column addition**: Add `excluded: bool` (default `False`) to IndexStore schema
2. **Auto-filtering**: `x_indices()` and `y_indices()` automatically exclude samples unless `include_excluded=True`
3. **Metadata tracking**: Store exclusion reasons in a separate structure or metadata column
4. **Pipeline operator**: Create `sample_filter` keyword for pipeline-based filtering

### 3.2 Filtering Operators Design

Following the sklearn `TransformerMixin` pattern used by augmenters:

```python
class SampleFilter(TransformerMixin, BaseEstimator):
    """Base class for sample filtering operators."""

    def fit(self, X, y=None):
        """Compute filter criteria from training data."""
        return self

    def get_mask(self, X, y=None) -> np.ndarray:
        """Return boolean mask: True = keep, False = exclude."""
        raise NotImplementedError

    def transform(self, X):
        """Transform is no-op for filters (filtering happens at indexer level)."""
        return X
```

### 3.3 Proposed Filter Operators

#### 3.3.1 Y-based Outlier Filters

```python
class YOutlierFilter(SampleFilter):
    """Filter samples with outlier target values."""

    def __init__(self, method: str = "iqr", threshold: float = 1.5):
        """
        Args:
            method: "iqr" (Interquartile Range), "zscore", "percentile"
            threshold: 1.5 for IQR (standard), 3.0 for zscore
        """
        self.method = method
        self.threshold = threshold
```

#### 3.3.2 X-based Outlier Filters

```python
class XOutlierFilter(SampleFilter):
    """Filter samples with outlier spectral features."""

    def __init__(self, method: str = "mahalanobis", threshold: float = None):
        """
        Args:
            method: "mahalanobis", "pca_residual", "isolation_forest", "lof"
            threshold: Detection threshold (method-specific)
        """
```

#### 3.3.3 Spectral Quality Filters

```python
class SpectralQualityFilter(SampleFilter):
    """Filter samples with poor spectral quality."""

    def __init__(self,
                 max_nan_ratio: float = 0.1,
                 max_zero_ratio: float = 0.5,
                 min_variance: float = 1e-6):
        """
        Args:
            max_nan_ratio: Maximum allowed NaN ratio per spectrum
            max_zero_ratio: Maximum allowed zero ratio (flat spectra)
            min_variance: Minimum variance (removes constant spectra)
        """
```

#### 3.3.4 Leverage/Influence Filters

```python
class HighLeverageFilter(SampleFilter):
    """Filter high-leverage samples that may unduly influence the model."""

    def __init__(self, threshold_multiplier: float = 2.0):
        """
        Args:
            threshold_multiplier: Multiple of average leverage to flag
        """
```

#### 3.3.5 Metadata-based Filters

```python
class MetadataFilter(SampleFilter):
    """Filter samples based on metadata column values."""

    def __init__(self, column: str, condition: Callable[[Any], bool]):
        """
        Args:
            column: Metadata column name
            condition: Lambda returning True to keep, False to exclude
        """
```

---

## 4. Pipeline Integration Design

### 4.1 Keyword Syntax

```python
pipeline = [
    "chart_2d",
    {
        "sample_filter": {
            "filters": [
                YOutlierFilter(method="iqr", threshold=1.5),
                XOutlierFilter(method="mahalanobis"),
            ],
            "mode": "any",  # "any" = exclude if ANY filter flags, "all" = only if ALL flag
            "report": True,  # Generate filtering report
        }
    },
    "snv",
    "model:PLSRegression",
]
```

### 4.2 Controller Implementation

```python
@register_controller
class SampleFilterController(OperatorController):
    """Controller for sample filtering operations."""

    priority = 5  # Execute early, before augmentation

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return keyword == "sample_filter"

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return False  # Filtering only during training
```

### 4.3 Execution Flow

1. Retrieve train samples (base only, no augmented)
2. Get X and y data
3. Apply each filter's `get_mask()` method
4. Combine masks according to `mode`
5. Mark excluded samples in indexer
6. Generate filtering report (optional)
7. Return updated context

---

## 5. Indexer Modifications

### 5.1 Schema Changes

```python
# In IndexStore
SCHEMA = {
    "row": pl.Int32,
    "sample": pl.Int32,
    "origin": pl.Int32,
    "partition": pl.Categorical,
    "group": pl.Int8,
    "branch": pl.Int8,
    "processings": pl.List(pl.Utf8),
    "augmentation": pl.Categorical,
    "excluded": pl.Boolean,  # NEW
    "exclusion_reason": pl.Utf8,  # NEW (optional)
}
```

### 5.2 API Additions

```python
class Indexer:
    def mark_excluded(
        self,
        sample_indices: List[int],
        reason: str = None
    ) -> None:
        """Mark samples as excluded from training."""

    def mark_included(
        self,
        sample_indices: List[int]
    ) -> None:
        """Remove exclusion flag from samples."""

    def x_indices(
        self,
        selector: Selector,
        include_augmented: bool = True,
        include_excluded: bool = False  # NEW parameter
    ) -> np.ndarray:
        """Get sample indices with filtering."""

    def get_excluded_samples(
        self,
        selector: Selector = None
    ) -> pl.DataFrame:
        """Get DataFrame of excluded samples with reasons."""

    def get_exclusion_summary(self) -> Dict[str, int]:
        """Get summary of exclusions by reason."""
```

### 5.3 QueryBuilder Modifications

```python
class QueryBuilder:
    def build(
        self,
        selector: Selector,
        exclude_columns: List[str] = None,
        include_excluded: bool = False  # NEW
    ) -> pl.Expr:
        """Build filter expression with exclusion handling."""
        condition = self._build_base_condition(selector, exclude_columns)

        if not include_excluded:
            condition = condition & (pl.col("excluded") == False)

        return condition
```

---

## 6. Dataset API Changes

### 6.1 New Methods

```python
class SpectroDataset:
    def filter_samples(
        self,
        filter_obj: SampleFilter,
        selector: Selector = {"partition": "train"},
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Apply sample filter to dataset.

        Args:
            filter_obj: SampleFilter instance
            selector: Which samples to consider for filtering
            dry_run: If True, return report without modifying data

        Returns:
            Dict with filtering statistics and excluded sample info
        """

    def get_excluded(self, selector: Selector = None) -> pl.DataFrame:
        """Get excluded samples as DataFrame."""

    def reset_exclusions(self, selector: Selector = None) -> int:
        """Remove all exclusion flags. Returns count of samples reset."""
```

---

## 7. Visualization Support

### 7.1 New Chart Types

```python
# Chart showing excluded vs included samples
"exclusion_chart"  # 2D scatter with color coding

# Chart showing outlier detection boundaries
"outlier_detection_chart"  # With thresholds visualized
```

### 7.2 Integration with Existing Charts

Existing charts should support an `include_excluded` parameter:

```python
pipeline = [
    {"chart_2d": {"include_excluded": True, "highlight_excluded": True}},
]
```

---

## 8. Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1-2)

**Goal**: Add exclusion tracking to indexer without breaking existing functionality.

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Add `excluded` column to IndexStore schema | High | 2h | None |
| Update QueryBuilder to handle exclusion | High | 3h | Schema change |
| Add `mark_excluded()` / `mark_included()` to Indexer | High | 2h | Schema change |
| Update `x_indices()` / `y_indices()` with `include_excluded` param | High | 2h | QueryBuilder |
| Add migration for existing datasets | Medium | 2h | Schema change |
| Unit tests for indexer changes | High | 4h | All above |

### Phase 2: Base Filter Framework (Week 2-3)

**Goal**: Create the base filter classes and one concrete implementation.

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Create `SampleFilter` base class | High | 2h | None |
| Implement `YOutlierFilter` (IQR, zscore) | High | 4h | Base class |
| Create `SampleFilterController` | High | 4h | Base class, Indexer changes |
| Pipeline integration (`sample_filter` keyword) | High | 3h | Controller |
| Integration tests | High | 4h | All above |

### Phase 3: Additional Filters (Week 3-4)

**Goal**: Implement the full suite of filtering operators.

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Implement `XOutlierFilter` (Mahalanobis, LOF) | Medium | 4h | Base class |
| Implement `SpectralQualityFilter` | Medium | 3h | Base class |
| Implement `HighLeverageFilter` | Medium | 3h | Base class |
| Implement `MetadataFilter` | Medium | 2h | Base class |
| Composite filter (AND/OR logic) | Medium | 2h | All filters |
| Unit tests for each filter | High | 6h | All above |

### Phase 4: Visualization & Reporting (Week 4-5)

**Goal**: Add visualization and reporting capabilities.

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Create `exclusion_chart` controller | Medium | 3h | Indexer changes |
| Create filtering report generator | Medium | 3h | Controller |
| Update existing charts for `include_excluded` | Low | 4h | Indexer changes |
| Documentation and examples | High | 4h | All above |

### Phase 5: Polish & Documentation (Week 5-6)

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Edge case handling (empty datasets, all excluded) | High | 3h | All above |
| Performance optimization | Medium | 4h | All above |
| User guide documentation | High | 4h | All above |
| Example scripts (`Q_XX_sample_filtering.py`) | High | 4h | All above |

---

## 9. Example Usage

### 9.1 Basic Pipeline Usage

```python
from nirs4all.operators.filters import YOutlierFilter, XOutlierFilter
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.data import DatasetConfigs

pipeline = [
    "chart_2d",  # Show data before filtering
    {
        "sample_filter": {
            "filters": [YOutlierFilter(method="iqr", threshold=1.5)],
            "report": True,
        }
    },
    "chart_2d",  # Show data after filtering
    "snv",
    {"split": KFold(n_splits=5)},
    "model:PLSRegression",
]

config = PipelineConfigs(pipeline, name="with_filtering")
runner = PipelineRunner()
runner.run(config, DatasetConfigs("my_data"))
```

### 9.2 Programmatic Usage

```python
from nirs4all.data import SpectroDataset
from nirs4all.operators.filters import YOutlierFilter

dataset = SpectroDataset("my_data")
# ... load data ...

# Dry run to see what would be excluded
report = dataset.filter_samples(
    YOutlierFilter(method="zscore", threshold=3.0),
    dry_run=True
)
print(f"Would exclude {report['n_excluded']} samples")

# Actually apply filter
dataset.filter_samples(YOutlierFilter(method="zscore", threshold=3.0))

# Check exclusions
print(dataset.get_excluded())
```

### 9.3 Multiple Filters

```python
pipeline = [
    {
        "sample_filter": {
            "filters": [
                YOutlierFilter(method="iqr"),
                SpectralQualityFilter(max_nan_ratio=0.05),
                MetadataFilter(column="quality", condition=lambda x: x != "bad"),
            ],
            "mode": "any",  # Exclude if ANY filter flags
        }
    },
    # ... rest of pipeline
]
```

---

## 10. Considerations & Edge Cases

### 10.1 Augmented Samples

When a base sample is excluded:
- **Option A**: Automatically exclude all its augmented samples (recommended)
- **Option B**: Keep augmented samples (risky, may cause data leakage)

Recommendation: Cascade exclusion to augmented samples via `origin` tracking.

### 10.2 Cross-Validation Interaction

Filters should be applied **before** cross-validation splitting to ensure:
- Consistent exclusion across folds
- No data leakage from test fold affecting train filter decisions

The pipeline ordering should enforce: `sample_filter` → `split` → `model`

### 10.3 Prediction Mode

During prediction:
- Filters should **NOT** exclude any samples
- Warning should be logged if predicting on data that would have been filtered
- Optional: Store filter thresholds to flag suspicious predictions

### 10.4 Multi-Source Datasets

For multi-source datasets:
- `XOutlierFilter` should consider all sources or allow source-specific filtering
- Configuration: `sources="all"` or `sources=[0, 2]`

---

## 11. Backward Compatibility

All changes should be backward compatible:
- Default `excluded=False` for existing datasets
- Default `include_excluded=False` maintains current behavior
- No changes to existing pipeline syntax required

---

## 12. Success Metrics

The implementation will be considered successful when:

1. **Functional**: All filter types work correctly on test datasets
2. **Non-destructive**: Exclusions are reversible without data loss
3. **Auditable**: Clear reporting of what was excluded and why
4. **Performant**: Filtering adds < 5% overhead to pipeline execution
5. **Documented**: User guide and examples available
6. **Tested**: > 90% test coverage on new code

---

## 13. References

- sklearn `TransformerMixin` pattern: https://scikit-learn.org/stable/developers/develop.html
- Outlier detection methods: https://scikit-learn.org/stable/modules/outlier_detection.html
- Mahalanobis distance: Used in chemometrics for spectral outlier detection
- Current nirs4all architecture: See [data/indexer.py](nirs4all/data/indexer.py), [controllers/data/sample_augmentation.py](nirs4all/controllers/data/sample_augmentation.py)

---

*Document prepared by Senior ML/Python Developer analysis - December 2024*
