# Multi-Target Support Design Document

**Version**: 1.0.0
**Status**: Draft / Proposal
**Date**: January 2026

This document describes the design for enabling datasets with multiple targets in nirs4all and nirs4all_webapp, allowing users to define a default target for backward compatibility or select specific targets at training time for single-target or multivariate analysis.

---

## Table of Contents

1. [Objectives](#objectives)
2. [Current State Analysis](#current-state-analysis)
   - [nirs4all Library](#nirs4all-library-current-state)
   - [nirs4all_webapp](#nirs4all_webapp-current-state)
3. [Components to Change](#components-to-change)
4. [Design Propositions](#design-propositions)
   - [Proposition A: Target Selection at Runtime](#proposition-a-target-selection-at-runtime)
   - [Proposition B: Multi-Target as First-Class Citizen](#proposition-b-multi-target-as-first-class-citizen)
   - [Proposition C: Hybrid Approach with Target Registry](#proposition-c-hybrid-approach-with-target-registry)
5. [Recommendation](#recommendation)
6. [Migration Path](#migration-path)

---

## Objectives

### Primary Goals

1. **Multi-Target Loading**: Datasets should support loading multiple target columns simultaneously (e.g., protein, moisture, fat content)

2. **Backward Compatibility**: Existing single-target workflows must continue to work without modification by using a default target

3. **Target Selection at Training Time**: Users should be able to specify which target(s) to use when calling `nirs4all.run()`:
   - Single target: `target=0` or `target="protein"`
   - Multiple targets: `targets=[0, 3]` or `targets=["protein", "fat"]`

4. **Independent Y Processing**: `TransformerMixin` operators (scalers) should apply independently to each target column

5. **Multivariate Support**: Enable training models that predict multiple targets simultaneously (multi-output regression/classification)

6. **Per-Target Metrics**: Compute and display metrics for each target separately, with optional aggregation

7. **Branch-per-Target**: Allow pipeline branches to operate on different targets and re-aggregate results

### Secondary Goals

1. **Webapp Integration**: Full support in the webapp for target selection, visualization, and metric display

2. **Mixed Task Types**: Support datasets where different targets have different task types (e.g., target 0 is regression, target 1 is classification)

3. **Target Metadata**: Store and display units, descriptions, and value ranges per target

---

## Current State Analysis

### nirs4all Library Current State

#### Data Storage (`nirs4all/data/targets.py`)

The `Targets` class already supports multi-target storage:

```python
# Internal storage supports 2D arrays
_data: Dict[str, np.ndarray]  # shape: (n_samples, n_targets)

# Properties expose multi-target info
@property
def num_targets(self) -> int:
    """Number of target columns (1 for single-target, >1 for multi-target)"""

@property
def task_type(self) -> TaskType:
    """Returns task type - currently single value for all targets"""
```

**Current Limitations**:
- `task_type` is a single value, not per-target
- Most downstream code assumes `y.shape[1] == 1` or treats y as 1D
- No target selection mechanism at retrieval time

#### Schema Configuration (`nirs4all/data/schema/config.py`)

The schema supports loading multiple columns via `train_y_filter`:

```python
class DatasetConfigSchema:
    train_y: Union[str, np.ndarray, List]  # Path or data
    train_y_filter: Optional[List[int]]     # Column indices to select
    test_y_filter: Optional[List[int]]      # Column indices to select
```

**Current Limitations**:
- No target naming/metadata in schema
- No default target specification
- `y_filter` applies at load time, not runtime

#### SpectroDataset Access (`nirs4all/data/_dataset/target_accessor.py`)

The `TargetAccessor` provides y access:

```python
def y(self, selector=None, include_augmented=True, include_excluded=False):
    """Get targets with filtering by partition/fold"""
    # Returns full (n_samples, n_targets) array
```

**Current Limitations**:
- No target column selection in selector
- Returns all targets, assumes caller handles multi-target

#### Y Processing (`nirs4all/controllers/transforms/y_transformer.py`)

The `YTransformerMixinController` applies transformers:

```python
# Current: applies transformer to entire y array
y_train = dataset.y(train_context)  # (n_samples, n_targets)
transformer.fit(y_train)             # Fits on all columns
y_transformed = transformer.transform(y_all)
```

**Current Behavior**: sklearn transformers like `StandardScaler` apply column-wise by default, so multi-target already works. However, the processing name and task type detection don't account for per-column differences.

#### Predictions and Metrics (`nirs4all/data/_predictions/result.py`)

```python
class PredictionResult:
    y_true: np.ndarray   # (n_samples,) or (n_samples, n_targets)
    y_pred: np.ndarray   # Same shape as y_true
```

**Current Limitations**:
- Metrics computed on flattened arrays or first column
- No per-target metric breakdown
- Visualization assumes single target

#### Pipeline Context (`nirs4all/pipeline/config/context.py`)

```python
class DataSelector:
    y: str = "numeric"  # Processing version name

class PipelineState:
    y_processing: str = "numeric"
```

**Current Limitations**:
- No target selection in context
- No mechanism to propagate selected targets through pipeline

### nirs4all_webapp Current State

#### Backend - Dataset API (`api/datasets.py`)

Datasets store target configuration:

```python
class TargetConfig:
    column: str           # Column name
    type: str            # "regression", "binary_classification", "multiclass_classification"
    unit: Optional[str]   # e.g., "%", "mg/L"
    classes: Optional[List[str]]
    is_default: bool      # Which is default

class Dataset:
    targets: List[TargetConfig]
    default_target: Optional[str]
```

**Current State**: Backend supports multi-target metadata, auto-detection from files, and default target selection.

#### Frontend - Target Selector (`src/components/datasets/TargetSelector.tsx`)

Reusable component for target selection:

```typescript
interface TargetSelectorProps {
  datasetId: string;
  value?: string;           // Selected target column
  onChange?: (column, config?) => void;
  targets?: TargetConfig[];
  defaultTarget?: string;
}
```

**Current State**: UI exists for single-target selection from multi-target datasets.

#### Frontend - Dataset Binding (`src/components/pipeline-editor/DatasetBinding.tsx`)

```typescript
interface BoundDataset {
  selectedTarget?: string;   // Single selected target
  taskType?: "regression" | "classification";
}
```

**Current State**: Pipeline editor allows selecting ONE target. No multi-target selection.

#### Frontend - Predictions Display

**Current Limitations**:
- Displays single metric set (R², RMSE, MAE)
- No per-target breakdown
- No multivariate visualization

---

## Components to Change

### nirs4all Library

| Component | File | Changes Required |
|-----------|------|------------------|
| Targets class | `data/targets.py` | Add per-target task type, target metadata |
| TargetAccessor | `data/_dataset/target_accessor.py` | Add target selection to `y()` method |
| DataSelector | `pipeline/config/context.py` | Add `target_indices` or `target_names` field |
| PipelineState | `pipeline/config/context.py` | Track active targets through pipeline |
| YTransformerController | `controllers/transforms/y_transformer.py` | Per-target processing tracking |
| PredictionResult | `data/_predictions/result.py` | Per-target metrics computation |
| Metrics | `data/_predictions/metrics.py` | Multi-target metric aggregation options |
| API run() | `api/run.py` | Add `target` parameter |
| Export/Predict | `pipeline/prediction/` | Store/load target selection in bundle |

### nirs4all_webapp

| Component | File | Changes Required |
|-----------|------|------------------|
| Dataset types | `src/types/datasets.ts` | Multi-target selection interface |
| TargetSelector | `src/components/datasets/TargetSelector.tsx` | Multi-select mode |
| DatasetBinding | `src/components/pipeline-editor/DatasetBinding.tsx` | Multi-target binding |
| Run config | `api/training.py` | Pass target selection to nirs4all |
| Predictions display | `src/components/runs/PredictDialog.tsx` | Per-target metrics |
| Target viz | `src/components/datasets/detail/DatasetTargetsTab.tsx` | Multi-target comparison |

---

## Design Propositions

### Proposition A: Target Selection at Runtime

**Philosophy**: Keep multi-target storage as-is, add target selection at API call time.

#### Design

**1. API Changes**

```python
# nirs4all.run() gains target parameter
result = nirs4all.run(
    pipeline=[...],
    dataset="path/to/dataset",
    target=0,                    # Single target by index
    # OR
    target="protein",            # Single target by name
    # OR
    targets=[0, 2],              # Multi-target by indices
    # OR
    targets=["protein", "fat"],  # Multi-target by names
)
```

**2. Context Propagation**

```python
@dataclass
class DataSelector:
    partition: str = "train"
    y: str = "numeric"
    target_indices: Optional[List[int]] = None  # NEW: which columns to use
```

**3. Dataset y() Method**

```python
def y(self, selector=None, include_augmented=True, include_excluded=False):
    y_full = self._get_targets(selector)

    # Apply target selection
    if selector and selector.target_indices is not None:
        return y_full[:, selector.target_indices]
    return y_full
```

**4. Metrics Per Target**

```python
class MultiTargetMetrics:
    per_target: Dict[str, Metrics]  # {target_name: {rmse, r2, mae}}
    aggregated: Metrics             # Combined metrics (mean, weighted)

    def display(self, mode="per_target"):  # or "aggregated" or "both"
        ...
```

#### Advantages

- Minimal changes to existing code
- Full backward compatibility (no target param = use all)
- Target selection is explicit and traceable
- Easy to implement incrementally

#### Disadvantages

- Target selection happens late (at training), not at dataset definition
- No compile-time validation of target existence
- Multi-output models need special handling

---

### Proposition B: Multi-Target as First-Class Citizen

**Philosophy**: Make multi-target a core concept throughout the system.

#### Design

**1. Target Registry in Dataset**

```python
@dataclass
class TargetInfo:
    index: int
    name: str
    task_type: TaskType
    unit: Optional[str] = None
    description: Optional[str] = None
    is_default: bool = False

class Targets:
    _registry: Dict[str, TargetInfo]  # name -> info

    def get_target(self, name_or_index) -> np.ndarray:
        """Get single target column by name or index"""

    def get_targets(self, selection=None) -> Tuple[np.ndarray, List[TargetInfo]]:
        """Get multiple targets with metadata"""
```

**2. Task Type Per Target**

```python
class Targets:
    @property
    def task_types(self) -> Dict[str, TaskType]:
        """Map of target_name -> task_type"""
        return {name: info.task_type for name, info in self._registry.items()}
```

**3. Per-Target Y Processing Chain**

```python
class ProcessingChain:
    # Current: single chain for all targets
    # New: optionally per-target chains
    _chains: Dict[str, List[ProcessingStep]]  # target_name -> chain

    def add_processing(self, name, ancestor, transformer, target=None):
        """Add processing step, optionally per-target"""
```

**4. Branch-per-Target Syntax**

```python
pipeline = [
    MinMaxScaler(),

    # Different models per target
    {"target_branch": {
        "protein": [PLSRegression(10)],
        "moisture": [PLSRegression(8)],
        "fat": [RandomForestRegressor()],
    }},

    # Merge predictions back
    {"merge_targets": "concat"},
]
```

**5. Schema Enhancement**

```yaml
# dataset.yaml
features:
  path: spectra.csv

targets:
  path: constituents.csv
  columns:
    - name: protein
      type: regression
      unit: "%"
      default: true
    - name: moisture
      type: regression
      unit: "%"
    - name: quality_grade
      type: multiclass_classification
      classes: [A, B, C, D]
```

#### Advantages

- Rich target metadata throughout system
- Supports mixed task types naturally
- Branch-per-target enables specialized pipelines
- Schema validation catches errors early

#### Disadvantages

- Significant refactoring required
- More complex internal model
- Potential performance overhead for single-target usage
- Longer implementation timeline

---

### Proposition C: Hybrid Approach with Target Registry

**Philosophy**: Add target registry for metadata and selection, but keep execution simple.

#### Design

**1. Target Registry (Metadata Only)**

```python
@dataclass
class TargetColumn:
    index: int
    name: Optional[str]
    task_type: Optional[TaskType] = None
    unit: Optional[str] = None
    is_default: bool = False

class TargetRegistry:
    """Lightweight registry for target metadata - does not affect storage"""
    columns: List[TargetColumn]
    default_index: int = 0

    def by_name(self, name: str) -> int:
        """Get column index by name"""

    def by_index(self, index: int) -> TargetColumn:
        """Get column info by index"""

    def resolve(self, selection) -> List[int]:
        """
        Resolve various selection formats to indices:
        - None -> [default_index]
        - int -> [index]
        - str -> [by_name(str)]
        - List[int|str] -> resolved indices
        - "all" -> all indices
        """
```

**2. Targets Class Enhancement**

```python
class Targets:
    _data: Dict[str, np.ndarray]      # Unchanged
    _registry: TargetRegistry          # NEW: metadata

    def get_targets(self, selection=None, processing="numeric") -> np.ndarray:
        """Get target data with optional column selection"""
        indices = self._registry.resolve(selection)
        return self._data[processing][:, indices]
```

**3. API Surface**

```python
# Default behavior (backward compatible)
result = nirs4all.run(pipeline, dataset)  # Uses default target

# Explicit single target
result = nirs4all.run(pipeline, dataset, target="protein")
result = nirs4all.run(pipeline, dataset, target=0)

# Multi-target (multivariate regression)
result = nirs4all.run(pipeline, dataset, targets=["protein", "moisture"])
result = nirs4all.run(pipeline, dataset, targets="all")

# Mixed: different models per target (via branch syntax)
pipeline = [
    MinMaxScaler(),
    {"target_branch": {
        "protein": PLSRegression(10),
        "moisture": PLSRegression(8),
    }},
]
```

**4. Context Changes**

```python
@dataclass
class DataSelector:
    partition: str = "train"
    y: str = "numeric"
    targets: Optional[Union[str, int, List]] = None  # NEW

@dataclass
class PipelineState:
    y_processing: str = "numeric"
    active_targets: Optional[List[int]] = None  # NEW: resolved indices
```

**5. Metrics Enhancement**

```python
@dataclass
class TargetMetrics:
    target_name: str
    target_index: int
    task_type: TaskType
    metrics: Dict[str, float]  # rmse, r2, mae, accuracy, etc.

@dataclass
class MultiTargetResult:
    targets: List[TargetMetrics]

    @property
    def aggregated_rmse(self) -> float:
        """Mean RMSE across targets"""

    @property
    def aggregated_r2(self) -> float:
        """Mean R² across targets"""

    def to_dataframe(self) -> pd.DataFrame:
        """Targets as rows, metrics as columns"""
```

**6. Webapp Changes**

```typescript
// Target selection mode
interface TargetSelection {
  mode: "single" | "multi" | "all";
  targets: string[];  // Column names
}

// In pipeline binding
interface BoundDataset {
  // ... existing fields
  targetSelection: TargetSelection;
}

// In run results
interface RunMetrics {
  overall: MetricSet;
  perTarget: Record<string, MetricSet>;
}
```

#### Advantages

- Clean separation: registry for metadata, selection for execution
- Backward compatible: `target=None` uses default
- Flexible: supports all selection patterns
- Incremental: can implement in phases
- Minimal storage overhead

#### Disadvantages

- Two concepts to understand (registry vs selection)
- Need to keep registry in sync with actual data
- Per-target y_processing requires additional tracking

---

## Recommendation

**Recommended: Proposition C (Hybrid Approach)**

Rationale:

1. **Lowest Risk**: Changes are additive, existing code paths remain functional

2. **Best Balance**: Rich enough for real use cases, simple enough for implementation

3. **Incremental Delivery**: Can ship in phases:
   - Phase 1: Target registry + single target selection
   - Phase 2: Multi-target selection + per-target metrics
   - Phase 3: Branch-per-target syntax (if needed)

4. **Webapp Alignment**: The webapp already has `TargetConfig` with similar structure to `TargetColumn`

5. **Backward Compatible**: All existing code works (selection=None uses default)

### Implementation Priority

| Phase | Scope | Components |
|-------|-------|------------|
| 1 | nirs4all core | `TargetRegistry`, `Targets.get_targets()`, `target` param in `run()` |
| 2 | nirs4all metrics | `MultiTargetResult`, per-target metric display |
| 3 | webapp binding | Multi-target selector, run config |
| 4 | webapp display | Per-target metrics in results |
| 5 | advanced | `target_branch` syntax for per-target pipelines |

---

## Migration Path

### Phase 1: Foundation

1. Add `TargetRegistry` and `TargetColumn` classes
2. Integrate registry into `Targets` class
3. Add `target` parameter to `nirs4all.run()`
4. Propagate selection through `DataSelector`
5. Update bundle export/import to store target selection

### Phase 2: Metrics

1. Create `MultiTargetResult` class
2. Update `PredictionResult` to compute per-target metrics
3. Update `RunResult.best_rmse`, etc. to handle multi-target
4. Add `result.target_metrics` property

### Phase 3: Webapp Integration

1. Update `TargetSelector` to support multi-select mode
2. Update `DatasetBinding` to store `TargetSelection`
3. Update training API to pass target selection to nirs4all
4. Update run manifest to store selected targets

### Phase 4: Visualization

1. Add per-target metric display in run results
2. Add multi-target comparison charts
3. Update predictions table to show per-target columns

### Breaking Changes

None anticipated if:
- `target=None` defaults to all targets (current behavior) or default target
- Existing bundle format remains readable
- Metric properties fallback to first target if single-target

### Deprecation Plan

No deprecations required. New functionality is additive.

---

## Appendix: Alternative Designs Considered

### A1: Target as Separate Dataset Dimension

Treat targets like sources with a `target_branch` paralleling `source_branch`. Rejected: over-complicates the common case.

### A2: Targets in Pipeline Steps

Allow `PLSRegression(target="protein")` in pipeline steps. Rejected: mixes concerns, hard to serialize.

### A3: Multiple y Properties

Add `y1`, `y2`, `y3` properties to SpectroDataset. Rejected: not scalable, ugly API.

---

## References

- Current target implementation: `nirs4all/data/targets.py`
- Processing chain: `nirs4all/data/_targets/processing_chain.py`
- Webapp target types: `nirs4all_webapp/src/types/datasets.ts`
- Merge syntax spec: `docs/_internal/specifications/merge_syntax.md`
