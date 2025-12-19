# Feature Analysis: Dataset-Level Aggregation by Sample ID

**Author:** GitHub Copilot
**Date:** December 19, 2025
**Status:** Design Proposal
**Priority:** High

---

## 1. Objective

Enable users to define a **sample aggregation column** at the dataset level, allowing predictions from multiple spectra of the same biological sample to be aggregated automatically. When this setting is defined:

1. **Scores in logs** should include both raw and aggregated metrics
2. **TabReportManager** should display aggregated scores alongside raw scores
3. The setting should propagate through the entire pipeline, eliminating the need to specify `aggregate='column_name'` repeatedly in visualization calls

### Use Case

In NIRS applications, it's common to have multiple spectral measurements (repetitions) for the same physical sample. For example:
- 4 scans per soil sample to reduce measurement noise
- Multiple measurements at different positions on a grain sample
- Repeated measurements for quality control

Users want to:
1. Train models on all individual spectra (to maximize data)
2. Evaluate and report performance on **aggregated predictions** (one prediction per physical sample)

---

## 2. Current State

### 2.1 Aggregation in Predictions (Implemented ✅)

The `Predictions` class already supports aggregation via a static `aggregate()` method:

```python
# nirs4all/data/predictions.py
@staticmethod
def aggregate(
    y_pred: np.ndarray,
    group_ids: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    y_true: Optional[np.ndarray] = None,
    method: str = 'mean'
) -> Dict[str, Any]:
    """
    Aggregate predictions by group (e.g., same sample ID with multiple measurements).

    For regression: averages y_pred values within each group.
    For classification: averages y_proba then takes argmax, or uses majority voting.
    """
```

### 2.2 Aggregation in Ranker (Implemented ✅)

The `PredictionRanker.top()` method accepts an `aggregate` parameter:

```python
# nirs4all/data/_predictions/ranker.py
def top(
    self,
    n: int,
    rank_metric: str = "",
    ...
    aggregate: Optional[str] = None,  # <-- Aggregation column
    ...
) -> PredictionResultsList:
    """
    Get top n models ranked by a metric.

    Args:
        aggregate: If provided, aggregate predictions by this metadata column or 'y'.
                  When 'y', groups by y_true values.
                  When a column name (e.g., 'ID'), groups by that metadata column.
                  Aggregated predictions have recalculated metrics.
    """
```

The ranker uses caching (`AggregationCache`, `ScoreCache`) to avoid redundant computations.

### 2.3 Aggregation in Visualization (Implemented ✅)

All visualization methods in `PredictionAnalyzer` support aggregation:

```python
# nirs4all/visualization/predictions.py
def plot_top_k(
    self,
    k: int = 5,
    ...
    aggregate: Optional[str] = None,  # <-- Passed through to ranker
    ...
) -> Figure:
    """
    Plot top K model comparison.

    Args:
        aggregate: If provided, aggregate predictions by this metadata column or 'y'.
    """
```

### 2.4 TabReportManager (Aggregation-aware ✅)

The `TabReportManager.generate_best_score_tab_report()` method now supports aggregation via the `aggregate` parameter:

```python
# nirs4all/visualization/reports.py
@staticmethod
def generate_best_score_tab_report(
    best_by_partition: Dict[str, Dict[str, Any]],
    aggregate: Optional[Union[str, bool]] = None  # <-- NEW PARAMETER
) -> Tuple[str, Optional[str]]:
    """Generate best score tab report from partition data.

    Args:
        best_by_partition: Dict mapping partition names to prediction entries
        aggregate: Sample aggregation setting.
            - None (default): No aggregation, only raw scores
            - True: Aggregate by y_true values
            - str: Aggregate by specified metadata column (e.g., 'sample_id')
            When set, both raw and aggregated scores are included.
    """
```

### 2.5 DatasetConfigs (NOT aggregation-aware ❌)

The `DatasetConfigs` class supports `task_type` and `signal_type` parameters but has **no aggregation setting**:

```python
# nirs4all/data/config.py
class DatasetConfigs:
    def __init__(
        self,
        configurations: Union[Dict[str, Any], List[...], str, List[str]],
        task_type: Union[str, List[str]] = "auto",
        signal_type: Union[SignalTypeInput, List[SignalTypeInput], None] = None
        # No aggregate/sample_id parameter
    ):
```

### 2.6 PipelineOrchestrator Logging (NOT aggregation-aware ❌)

When displaying best scores, the orchestrator does not consider aggregation:

```python
# nirs4all/pipeline/execution/orchestrator.py (line ~467)
best_by_partition = run_dataset_predictions.get_entry_partitions(best)
tab_report, _ = TabReportManager.generate_best_score_tab_report(best_by_partition)
# No aggregation applied before reporting
```

### 2.7 Roadmap Note (Already Identified ✅)

The Roadmap.md file already identifies this feature:

```markdown
> [Aggregation] add aggregation as property of the dataset
>   (if True it's on y val, if "str" it's on metadata, if nothing, no agg)
> [TabReport] include > aggregated
```

---

## 3. Suggestions

### 3.1 Add `aggregate` Parameter (Two Entry Points)

Users should be able to define the `aggregate` setting in **two ways**:

#### Option A: As a DatasetConfigs Constructor Parameter

Extend `DatasetConfigs.__init__()` to accept an aggregation specification:

```python
class DatasetConfigs:
    def __init__(
        self,
        configurations: ...,
        task_type: ... = "auto",
        signal_type: ... = None,
        aggregate: Union[str, bool, None] = None  # NEW PARAMETER
    ):
        """
        Initialize dataset configurations.

        Args:
            aggregate: Sample aggregation setting (applies to all datasets).
                - None (default): No aggregation
                - True: Aggregate by y_true values (unique target groups)
                - str: Aggregate by specified metadata column (e.g., 'sample_id', 'ID')
        """
```

#### Option B: Inside the Configuration Dictionary

Allow `aggregate` to be specified in the config dict itself (per-dataset control):

```python
# In config dict - useful for per-dataset settings or YAML/JSON configs
config = {
    "train_x": "/path/to/train_spectra.csv",
    "train_y": "/path/to/train_targets.csv",
    "train_m": "/path/to/train_metadata.csv",
    "test_x": "/path/to/test_spectra.csv",
    "test_y": "/path/to/test_targets.csv",
    "test_m": "/path/to/test_metadata.csv",
    "aggregate": "sample_id"  # <-- NEW: Aggregate by this metadata column
}

dataset = DatasetConfigs(config)
```

#### Priority Resolution

When both are specified, the **constructor parameter takes precedence** (allows override):

```python
config = {
    "train_x": "...",
    "aggregate": "sample_id"  # Config-level setting
}

# Constructor parameter overrides config dict
dataset = DatasetConfigs(config, aggregate="batch_id")  # Uses "batch_id"
```

**Semantics:**
- `None`: No aggregation (current behavior)
- `True`: Aggregate by `y` (target values) - useful for classification where multiple spectra share the same class
- `"column_name"`: Aggregate by the specified metadata column (e.g., `"sample_id"`, `"ID"`)

### 3.2 Store Aggregation Setting in SpectroDataset

Propagate the setting to `SpectroDataset`:

```python
class SpectroDataset:
    def __init__(self, name: str = "Unknown_dataset"):
        ...
        self._aggregate_column: Optional[str] = None  # NEW
        self._aggregate_by_y: bool = False  # NEW

    @property
    def aggregate(self) -> Optional[str]:
        """Get aggregation column name, or 'y' if aggregating by target."""
        if self._aggregate_by_y:
            return 'y'
        return self._aggregate_column

    def set_aggregate(self, value: Union[str, bool, None]) -> None:
        """Set aggregation behavior."""
        if value is True:
            self._aggregate_by_y = True
            self._aggregate_column = None
        elif isinstance(value, str):
            self._aggregate_by_y = False
            self._aggregate_column = value
        else:
            self._aggregate_by_y = False
            self._aggregate_column = None
```

### 3.3 Update TabReportManager

Add aggregation support to the report generator:

```python
@staticmethod
def generate_best_score_tab_report(
    best_by_partition: Dict[str, Dict[str, Any]],
    aggregate: Optional[str] = None  # NEW PARAMETER
) -> Tuple[str, Optional[str]]:
    """
    Generate best score tab report from partition data.

    Args:
        best_by_partition: Dict mapping partition names to prediction entries
        aggregate: If provided, also compute and display aggregated metrics.
                  When 'y', groups by y_true values.
                  When a column name, groups by that metadata column.

    Returns:
        Tuple of (formatted_string, csv_string_content)
        If aggregate is set, both raw and aggregated scores are included.
    """
```

The output format could show both:

```
|----------|--------|----------|-------|-------|-----|-----|------|------|------|-------|-------|-------|------|------|------|-------------|
|          | Nsample| Nfeature | Mean  | Median| Min | Max | SD   | CV   | R2   | RMSE  | MSE   | SEP   | MAE  | RPD  | Bias | Consistency |
|----------|--------|----------|-------|-------|-----|-----|------|------|------|-------|-------|-------|------|------|------|-------------|
| Cros Val | 150    | 100      | 12.5  | 12.3  | 8.1 | 18.2| 2.3  | 0.18 | 0.89 | 0.782 | 0.612 | 0.756 | 0.612| 4.32 | 0.02 | 85.3        |
| Cros Val*| 50     | 100      |       |       |     |     |      |      | 0.92 | 0.654 | 0.428 | 0.632 | 0.512| 5.12 | 0.01 | 92.1        |
| Train    | 120    | 100      | 12.4  | 12.2  | 8.0 | 18.0| 2.2  | 0.18 | 0.95 | 0.512 | 0.262 | 0.498 | 0.398| 6.10 | 0.01 | 91.2        |
| Train*   | 40     | 100      |       |       |     |     |      |      | 0.97 | 0.412 | 0.170 | 0.398 | 0.312| 7.45 | 0.00 | 96.5        |
| Test     | 30     | 100      | 12.6  | 12.4  | 8.2 | 18.3| 2.4  | 0.19 | 0.87 | 0.823 | 0.678 | 0.801 | 0.678| 4.01 | 0.03 | 82.1        |
| Test*    | 10     | 100      |       |       |     |     |      |      | 0.91 | 0.698 | 0.487 | 0.675 | 0.562| 4.89 | 0.02 | 90.0        |
|----------|--------|----------|-------|-------|-----|-----|------|------|------|-------|-------|-------|------|------|------|-------------|

* Aggregated by sample_id
```

### 3.4 Update PipelineOrchestrator

Pass the aggregation setting from the dataset to the reporting:

```python
# In orchestrator.py, when printing best predictions:
if self.enable_tab_reports:
    best_by_partition = run_dataset_predictions.get_entry_partitions(best)

    # Get aggregate setting from dataset
    aggregate_column = dataset.aggregate  # Could be None, 'y', or column name

    tab_report, tab_report_csv = TabReportManager.generate_best_score_tab_report(
        best_by_partition,
        aggregate=aggregate_column  # Pass aggregation setting
    )
    logger.info(tab_report)
```

### 3.5 Propagate to Visualization Default

When `DatasetConfigs` has an `aggregate` setting, it could be passed as a default to the `PredictionAnalyzer`:

```python
class PredictionAnalyzer:
    def __init__(
        self,
        predictions_obj: Predictions,
        ...,
        default_aggregate: Optional[str] = None  # NEW: Default from dataset
    ):
        self.default_aggregate = default_aggregate

    def plot_top_k(
        self,
        ...,
        aggregate: Optional[str] = None,  # Can override default
        ...
    ):
        # Use provided aggregate or fall back to default
        effective_aggregate = aggregate if aggregate is not None else self.default_aggregate
        ...
```

---

## 4. Roadmap

### Phase 1: Core Infrastructure (Priority: High)

| Task | Module | Description | Effort |
|------|--------|-------------|--------|
| 1.1 | `DatasetConfigs` | Add `aggregate` parameter to `__init__()` | S |
| 1.2 | `config_parser` | Parse `aggregate` key from config dict in `parse_config()` | S |
| 1.3 | `SpectroDataset` | Add `aggregate` property and `set_aggregate()` method | S |
| 1.4 | `DatasetConfigs` | Propagate aggregate setting to `SpectroDataset` during loading | M |
| 1.5 | `DatasetConfigs` | Implement priority resolution (constructor > config dict) | S |
| 1.6 | Tests | Unit tests for aggregate setting (both entry points) | M |

**Deliverable:** Users can specify `aggregate='sample_id'` either in `DatasetConfigs` constructor or in config dict.

### Phase 2: TabReportManager Enhancement (Priority: High) ✅ IMPLEMENTED

| Task | Module | Description | Effort | Status |
|------|--------|-------------|--------|--------|
| 2.1 | `TabReportManager` | Add `aggregate` parameter to `generate_best_score_tab_report()` | M | ✅ |
| 2.2 | `TabReportManager` | Compute aggregated metrics using `Predictions.aggregate()` | M | ✅ |
| 2.3 | `TabReportManager` | Format output with both raw and aggregated rows | M | ✅ |
| 2.4 | `TabReportManager` | Update CSV format for aggregated reports | S | ✅ |
| 2.5 | Tests | Unit tests for aggregated TabReportManager output | M | ✅ |

**Deliverable:** TabReports show both raw and aggregated metrics when aggregate is set.

**Implementation Notes (Dec 2025):**
- Added `_aggregate_predictions()` helper method that uses `Predictions.aggregate()` internally
- Table rows marked with `*` suffix indicate aggregated data (e.g., `Cros Val*`)
- Aggregated rows have blank descriptive stats (Mean, Median, etc.) since they don't apply after averaging
- CSV format includes new `Aggregated` column showing the aggregation column name
- Footer note explains which column was used for aggregation
- 17 unit tests added in `tests/unit/visualization/test_tab_report_aggregation.py`

### Phase 3: Pipeline Integration (Priority: Medium) ✅ IMPLEMENTED

| Task | Module | Description | Effort | Status |
|------|--------|-------------|--------|--------|
| 3.1 | `PipelineOrchestrator` | Pass aggregate from dataset to TabReportManager | S | ✅ |
| 3.2 | `ExecutionContext` | Store and propagate aggregate setting through pipeline | M | ✅ |
| 3.3 | Logging | Update log messages to indicate when aggregated scores are shown | S | ✅ |
| 3.4 | Integration Tests | End-to-end test with aggregate setting | L | ✅ |

**Deliverable:** Pipeline runs automatically use dataset's aggregate setting for reporting.

**Implementation Notes (Dec 2025):**
- `PipelineOrchestrator._print_best_predictions()` now extracts `aggregate` from `dataset.aggregate` and passes it to `TabReportManager.generate_best_score_tab_report()`
- `ExecutionContext` class extended with `aggregate_column` property that is initialized from `dataset.aggregate` in `PipelineExecutor.initialize_context()`
- Added info-level log message when aggregated scores are included: "Including aggregated scores (by {column}) in report"
- 13 integration tests added in `tests/integration/pipeline/test_aggregation_integration.py`

### Phase 4: Visualization Defaults (Priority: Low) ✅ IMPLEMENTED

| Task | Module | Description | Effort | Status |
|------|--------|-------------|--------|--------|
| 4.1 | `PipelineRunner` | Return aggregate setting along with predictions | S | ✅ |
| 4.2 | `PredictionAnalyzer` | Add `default_aggregate` constructor parameter | S | ✅ |
| 4.3 | Documentation | Update user guide with aggregation examples | M | ✅ |
| 4.4 | Examples | Add example demonstrating aggregate feature | M | ✅ |

**Deliverable:** Visualization automatically uses dataset's aggregate setting by default.

**Implementation Notes (Dec 2025):**
- Added `runner.last_aggregate` property to access aggregate setting from last executed dataset
- Added `default_aggregate` constructor parameter to `PredictionAnalyzer`
- Added `_resolve_aggregate()` helper method for unified aggregate resolution
- All visualization methods (`plot_top_k`, `plot_confusion_matrix`, `plot_histogram`, `plot_heatmap`, `plot_candlestick`, branch methods) now use default aggregate
- User guide documentation added: `docs/user_guide/aggregation.md`
- Example script added: `examples/Q34_aggregation.py`
- 18 unit tests added in `tests/unit/visualization/test_prediction_analyzer_default_aggregate.py`

### Phase 5: Advanced Features (Priority: Low, Future)

| Task | Module | Description | Effort |
|------|--------|-------------|--------|
| 5.1 | Config Files | Support aggregate in YAML/JSON config files | M |
| 5.2 | Multi-dataset | Per-dataset aggregate settings in multi-dataset runs | M |
| 5.3 | Aggregation Methods | Support different methods (median, vote) configurable even in analyzer and charts | M |
| 5.4 | Outlier Exclusion | T² based outlier exclusion before aggregation (from Roadmap) | L |

---

## 5. API Examples

### After Implementation

#### Example 1: Using Constructor Parameter (Simple)

```python
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs
from sklearn.cross_decomposition import PLSRegression

# Define dataset with aggregation via constructor parameter
dataset = DatasetConfigs(
    "path/to/soil_spectra",
    aggregate="sample_id"  # <-- Aggregate by sample_id column
)

pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=5),
    {"model": PLSRegression(n_components=10)}
]

runner = PipelineRunner(verbose=1)
predictions, per_dataset = runner.run(
    PipelineConfigs(pipeline, "PLS_Soil"),
    dataset
)
```

#### Example 2: Using Config Dict (Detailed Control)

```python
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs

# Define aggregation inside config dict
config = {
    "train_x": "data/soil_train_spectra.csv",
    "train_y": "data/soil_train_targets.csv",
    "train_m": "data/soil_train_metadata.csv",  # Contains 'sample_id' column
    "test_x": "data/soil_test_spectra.csv",
    "test_y": "data/soil_test_targets.csv",
    "test_m": "data/soil_test_metadata.csv",
    "aggregate": "sample_id"  # <-- Aggregate by this metadata column
}

dataset = DatasetConfigs(config)

# Or with global_params style:
config = {
    "train_x": "data/spectra.csv",
    "train_y": "data/targets.csv",
    "train_m": "data/metadata.csv",
    "global_params": {
        "aggregate": "sample_id",
        "header_unit": "nm"
    }
}
```

#### Example 3: Multi-Dataset with Per-Dataset Aggregation

```python
# Different aggregation columns per dataset
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

datasets = DatasetConfigs([config1, config2])
```

#### Log Output

```
# |----------|--------|----------|------|-------|-------|
# | Cros Val | 400    | 200      | 0.87 | 0.712 | 0.612 |
# | Cros Val*| 100    | 200      | 0.92 | 0.598 | 0.512 |  <-- Aggregated
# | Test     | 100    | 200      | 0.85 | 0.756 | 0.678 |
# | Test*    | 25     | 200      | 0.90 | 0.632 | 0.562 |  <-- Aggregated
# * Aggregated by sample_id
```

### JSON Configuration

```json
{
    "train_x": "data/spectra.csv",
    "train_y": "data/targets.csv",
    "train_m": "data/metadata.csv",
    "aggregate": "sample_id"
}
```

### YAML Configuration

```yaml
train_x: "data/spectra.csv"
train_y: "data/targets.csv"
train_m: "data/metadata.csv"
aggregate: "sample_id"  # Aggregate predictions by this column

# Or for classification:
# aggregate: true  # Aggregate by y values
```

---

## 6. Impact Analysis

### Affected Files

| File | Change Type | Impact |
|------|-------------|--------|
| `nirs4all/data/config.py` | Modify | Add `aggregate` parameter to `DatasetConfigs.__init__()` |
| `nirs4all/data/config_parser.py` | Modify | Parse `aggregate` key from config dict |
| `nirs4all/data/dataset.py` | Modify | Add `aggregate` property and setter |
| `nirs4all/visualization/reports.py` | Modify | Support aggregated reports |
| `nirs4all/pipeline/execution/orchestrator.py` | Modify | Pass aggregate to reporting |
| `tests/unit/data/test_config.py` | Add/Modify | New test cases (both entry points) |
| `tests/unit/visualization/test_reports.py` | Add/Modify | Aggregation tests |
| `docs/user_guide/*.md` | Modify | Document new feature |
| `examples/Q*.py` | Add | New example with aggregation |

### Backward Compatibility

- **100% backward compatible**: All changes are additive
- Default behavior (`aggregate=None`) preserves current functionality
- Existing scripts work unchanged

### Dependencies

- Leverages existing `Predictions.aggregate()` method
- Uses existing `AggregationCache` and `ScoreCache` for performance
- No new external dependencies

---

## 7. Success Criteria

1. ✅ `DatasetConfigs(path, aggregate='sample_id')` accepted without error
2. ✅ `DatasetConfigs({"train_x": "...", "aggregate": "sample_id"})` accepted without error
3. ✅ Constructor parameter overrides config dict value when both specified
4. ✅ Pipeline log shows both raw and aggregated scores when aggregate is set
5. ✅ TabReportManager CSV includes aggregated rows
6. ✅ All existing tests pass (backward compatibility)
7. ✅ New unit tests for aggregation feature pass (both entry points)
8. ✅ Example script demonstrates the feature

---

## 8. References

- [Roadmap.md](../../Roadmap.md) - Original feature request
- [predictions.py](../../nirs4all/data/predictions.py) - `Predictions.aggregate()` implementation
- [ranker.py](../../nirs4all/data/_predictions/ranker.py) - `PredictionRanker.top()` with aggregate support
- [reports.py](../../nirs4all/visualization/reports.py) - Current `TabReportManager` implementation
- [config.py](../../nirs4all/data/config.py) - Current `DatasetConfigs` implementation
