# Multivariate Target Support Analysis

This document analyzes multivariate target handling in nirs4all and proposes a design for comprehensive support.

**Status**: Design Document
**Created**: 2026-01-20
**Scope**: nirs4all library + webapp

---

## Table of Contents

1. [Current Target Handling](#1-current-target-handling)
2. [Current Multivariate Capacity](#2-current-multivariate-capacity)
3. [High-Level Design: nirs4all Library](#3-high-level-design-nirs4all-library)
4. [High-Level Design: Webapp](#4-high-level-design-webapp)
5. [High-Level Roadmap](#5-high-level-roadmap)
6. [Additional Enhancements](#6-additional-enhancements)

---

## 1. Current Target Handling

This section documents how target variables (Y) flow through the system from loading to use.

### 1.1 Loading Pipeline

#### Configuration Schema (`nirs4all/data/schema/config.py`)

The `DatasetConfigSchema` defines target specification:

```python
# Legacy format (primary)
train_y: Optional[str]              # Path to CSV or None (extract from X)
train_y_filter: Optional[List[int]] # Column indices to select
train_y_params: Optional[LoadingParams]  # Loading parameters

# Multi-source format
shared_targets:
  path: "data/targets.csv"
  columns: [0, 1, 2]                # Target column indices
  link_by: "sample_id"              # Linking key
```

**Key loading parameters for Y:**
- `categorical_mode`: "auto" (factorize strings), "preserve" (to_numeric), "none"
- `na_policy`: "remove" (filter rows) or "abort" (raise error)

#### Data Loading Flow

```
DatasetConfigs
    │
    ▼
handle_data() ─────────► load_XY()
    │                        │
    │                        ▼
    │                   load_csv() with data_type='y'
    │                        │
    │                        ▼
    │                   Categorical conversion
    │                   NA handling
    │                        │
    ▼                        ▼
SpectroDataset.add_targets(y_array)
```

**File**: `nirs4all/data/loaders/loader.py` (lines 185-349)

Three Y extraction scenarios:
1. **Separate file**: `train_y` points to CSV
2. **From X**: `train_y=None`, `train_y_filter=[0, 2]` extracts columns from X
3. **No Y**: Both None → empty Y array

### 1.2 Internal Storage

#### Targets Class (`nirs4all/data/targets.py`)

Y is stored in a `Targets` object with processing chain:

```python
class Targets:
    _data: Dict[str, np.ndarray]  # processing_name -> array
    # Always 2D: (n_samples, n_targets)
```

**Processing versions:**
- `"raw"`: Original data type preserved (string, int, float)
- `"numeric"`: Converted to float32, label-encoded for categorical
- Custom: "scaled", "normalized", etc. (added via pipeline)

**Key methods:**
| Method | Purpose |
|--------|---------|
| `add_targets(y)` | Add samples, creates raw + numeric |
| `add_processed_targets(name, y, ...)` | Add transformed version |
| `get_targets(processing, indices)` | Retrieve by processing name |
| `transform_predictions(y_pred, from, to)` | Inverse transform predictions |

**Shape enforcement** (lines 308-312):
```python
if targets.ndim == 1:
    targets = targets.reshape(-1, 1)  # Force 2D
```

### 1.3 Access Patterns

#### SpectroDataset API (`nirs4all/data/dataset.py`)

```python
# Get targets
y = dataset.y(selector={"partition": "train"}, include_augmented=False)

# Task type (global)
dataset.task_type  # "regression" | "classification" | "auto"
dataset.is_regression  # bool
dataset.num_classes  # int for classification
```

**Selector options:**
- `{"partition": "train"}` or `{"partition": "test"}`
- `{"fold": 0}` for cross-validation
- Indices: `[0, 1, 5, 10]`

#### TargetAccessor (`nirs4all/data/_dataset/target_accessor.py`)

Internal layer providing filtered access:
```python
class TargetAccessor:
    def y(self, selector, include_augmented=True, include_excluded=False):
        # Returns filtered targets based on selector
```

### 1.4 Usage in Pipeline

#### Model Training

Controllers receive Y from context:
```python
# BaseModelController
y_train = context.y_train  # From RuntimeContext
model.fit(X_train, y_train)
```

#### Y Transformations

The `YTransformerMixinController` applies sklearn transformers:
```python
{"y_processing": StandardScaler()}  # Pipeline syntax

# Controller fits on train, transforms both
y_scaled = scaler.fit_transform(y_train)
dataset.add_processed_targets("scaled", y_scaled, ...)
```

### 1.5 Current Limitations

| Limitation | Impact |
|-----------|--------|
| No target selection at runtime | Cannot choose which target(s) to train on |
| Global task_type | Cannot mix regression + classification targets |
| No target metadata | No names, units, descriptions stored |
| No missing target policy | Rows with NaN in any target affect all |

---

## 2. Current Multivariate Capacity

This section assesses which components support multivariate Y and which break.

### 2.1 Infrastructure Support

The core data infrastructure **does support** multivariate targets:

| Component | Support | Notes |
|-----------|---------|-------|
| `Targets` class | ✅ Full | 2D arrays `(n_samples, n_targets)` |
| `SpectroDataset.y()` | ✅ Full | Returns all target columns |
| `TargetAccessor` | ✅ Full | Transparent pass-through |
| `NumericConverter` | ✅ Full | Column-wise type conversion |
| `ProcessingChain` | ✅ Full | Preserves 2D structure |

**Evidence** (`targets.py:171-185`):
```python
def test_multidimensional_targets(self):
    data = np.array([[1, 2], [3, 4], [5, 6]])  # 3 samples, 2 targets
    targets = Targets()
    targets.add_targets(data)
    assert targets.get_targets("scaled").shape == (3, 2)  # Works
```

### 2.2 Operators Assessment

#### Splitters

| Operator | File | Multivariate | Issue |
|----------|------|--------------|-------|
| `KennardStoneSplitter` | splitters.py:403 | ✅ Works | X-only algorithm |
| `SPXYSplitter` | splitters.py:459 | ⚠️ Partial | Reshapes 1D→2D, uses PCA on Y |
| `SPXYGFold` | splitters.py:561 | ✅ Works | Explicit multi-output logic |
| `SystematicCircularSplitter` | splitters.py:119 | ❌ **Breaks** | Uses `y[:, 0]` only |
| `KBinsStratifiedSplitter` | splitters.py:154 | ⚠️ Partial | sklearn handles 2D |
| `BinnedStratifiedGroupKFold` | splitters.py:189 | ❌ **Breaks** | `.ravel()` loses structure |

**SystematicCircularSplitter issue** (line 140):
```python
ordered_idx = np.argsort(y[:, 0], axis=0)  # Only uses first target
```

**BinnedStratifiedGroupKFold issue** (line 323):
```python
y_binned = discretizer.fit_transform(y).ravel().astype(int)  # Flattens
```

#### Filters

| Operator | File | Multivariate | Issue |
|----------|------|--------------|-------|
| `HighLeverageFilter` | high_leverage.py | ✅ Works | X-only |
| `XOutlierFilter` | x_outlier.py | ✅ Works | X-only |
| `YOutlierFilter` | y_outlier.py | ❌ **Breaks** | `.flatten()` all dimensions |

**YOutlierFilter issue** (line 157):
```python
y_flat = np.asarray(y).flatten()  # Loses multi-output structure
# All methods (IQR, ZScore, MAD) operate on flattened 1D
```

#### Controllers

| Controller | File | Multivariate | Issue |
|------------|------|--------------|-------|
| `YTransformerMixinController` | y_transformer.py | ✅ Works | sklearn handles 2D |
| `TagController` | tag.py | ❌ **Breaks** | Explicit flatten |
| `ExcludeController` | exclude.py | ⚠️ Depends | Delegates to filters |
| `BaseModelController` | base_model.py | ⚠️ Depends | Depends on model |
| `CrossValidatorController` | split.py | ❌ **Breaks** | `_bin_y_for_groups` ravels |

**TagController issue** (line 148-149):
```python
if y is not None and y.ndim > 1:
    y = y.flatten()  # Intentional but breaks multivariate
```

**CrossValidatorController issue** (line 516):
```python
y = np.asarray(y).ravel()  # For force_group="y" option
```

### 2.3 Model Support Matrix

Model multivariate support depends on the underlying framework:

| Model Type | Multi-Output | Notes |
|------------|-------------|-------|
| `PLSRegression` | ✅ Native | Designed for multi-output |
| `RandomForestRegressor` | ✅ Native | sklearn multi-output |
| `LinearRegression` | ✅ Native | sklearn multi-output |
| `Ridge`, `Lasso` | ✅ Native | sklearn multi-output |
| `SVR`, `SVC` | ❌ Single | Requires MultiOutputRegressor wrapper |
| `XGBRegressor` | ⚠️ Limited | `multi_output_regressor=True` param |
| TensorFlow/PyTorch | ✅ Native | Output layer configurable |

### 2.4 Summary: Breaking Points

**Critical failures with multivariate Y:**

1. **YOutlierFilter** - Flattens Y before analysis
   - Location: `operators/filters/y_outlier.py:157`
   - Impact: Outlier detection treats all values as one pool

2. **TagController** - Explicitly flattens Y
   - Location: `controllers/data/tag.py:148-149`
   - Impact: Tags computed on wrong data shape

3. **SystematicCircularSplitter** - Uses first column only
   - Location: `operators/splitters/splitters.py:140`
   - Impact: Sorting ignores other targets

4. **BinnedStratifiedGroupKFold** - Ravel after discretization
   - Location: `operators/splitters/splitters.py:323`
   - Impact: Stratification meaningless for multi-output

5. **CrossValidatorController._bin_y_for_groups** - Ravel Y
   - Location: `controllers/splitters/split.py:516`
   - Impact: `force_group="y"` fails

### 2.5 Adaptability Assessment

| Category | Adaptability | Effort |
|----------|-------------|--------|
| Storage layer | Already works | None |
| Loading | Already works | None |
| Splitters (Y-dependent) | Moderate | Per-operator changes |
| Filters (Y-dependent) | Moderate | Per-operator changes |
| Y transformers | Already works | None |
| Model controllers | Low | Mostly works |
| Metrics | High | New result structure |
| Pipeline DSL | High | New keywords |

---

## 3. High-Level Design: nirs4all Library

This section outlines the proposed design for comprehensive multivariate target support.

### 3.1 Design Decisions

Based on analysis and requirements:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Target selection scope | Run-level + Pipeline DSL | Maximum flexibility |
| Task type handling | Auto-detect per target | Supports mixed regression/classification |
| Operator fallback | Error if multivariate + no selection | Explicit over implicit |
| Missing target policy | Configurable (default: exclude) | User control over behavior |
| Metrics reporting | Per-target + aggregate | Full visibility |
| Target branching | Reuse `branch` syntax | Consistent DSL |
| Multi-output models | Both modes supported | Model-dependent default |

### 3.2 Core Components

#### 3.2.1 TargetRegistry

New class to store target metadata in `SpectroDataset`:

```python
# nirs4all/data/target_registry.py

@dataclass
class TargetInfo:
    name: str                          # "pH", "quality", etc.
    index: int                         # Column index in Y array
    task_type: str = "auto"            # "regression", "classification", "auto"
    unit: Optional[str] = None         # "mg/L", etc.
    description: Optional[str] = None

class TargetRegistry:
    _targets: Dict[str, TargetInfo]    # name -> info

    def register(self, name: str, index: int, **kwargs) -> None
    def get(self, name: str) -> TargetInfo
    def get_by_index(self, index: int) -> TargetInfo
    def names(self) -> List[str]
    def task_type(self, name: str) -> str  # Auto-detect if "auto"
```

**Auto-Detection Behavior (Default)**:

Target names and task types are **auto-detected by default** and can be overridden:

| Property | Auto-Detection | Override |
|----------|---------------|----------|
| **Name** | CSV column header, or `target_0`, `target_1`, etc. if no header | `target_columns[].name` |
| **Task Type** | Inferred from data: continuous → regression, discrete/string → classification | `target_columns[].task_type` |
| **Unit** | None (not auto-detected) | `target_columns[].unit` |

```python
# Auto-detection example
dataset = nirs4all.load("data/")  # Y columns named from CSV headers
print(dataset.target_names)       # ["pH", "quality"] (from headers)
print(dataset.target_registry.get("pH").task_type)  # "regression" (auto-detected)

# Override example
dataset = nirs4all.load("data/", target_columns=[
    {"column": 0, "name": "acidity", "task_type": "regression"},  # Override name
    {"column": 1, "task_type": "classification"},                 # Keep auto name, force type
])
```

**Integration with SpectroDataset:**
```python
class SpectroDataset:
    _target_registry: TargetRegistry

    @property
    def target_names(self) -> List[str]

    def y(self, selector=None, targets=None, ...):
        # targets: str, List[str], or None (all)
        # Returns subset of columns if targets specified
```

#### 3.2.2 Target Selection in API

**Run-level selection:**
```python
# nirs4all/api/runner.py

def run(
    pipeline,
    dataset,
    target: Optional[Union[str, int, List[Union[str, int]]]] = None,  # NEW
    missing_target_policy: str = "exclude",           # NEW: "exclude", "keep"
    ...
):
    """
    Args:
        target: Target name(s) or index(es) to train on. None = all targets.
        missing_target_policy: How to handle rows with NaN in selected target(s).
    """
```

#### 3.2.3 Multi-Dataset Runs

Runs can operate on **multiple datasets** simultaneously, each with its own target configuration:

```python
# nirs4all/api/runner.py

def run(
    pipeline,
    dataset: Union[SpectroDataset, List[SpectroDataset], str, List[str]],
    target: Optional[Union[
        str, int, List[Union[str, int]],           # Single dataset target(s)
        List[List[Union[str, int]]]                # Per-dataset targets
    ]] = None,
    ...
):
    """
    Args:
        dataset: Single dataset or list of datasets (paths or SpectroDataset objects).
        target: Target selection. For multi-dataset:
                - Single value/list: applied to ALL datasets
                - List of lists: per-dataset target configuration
    """
```

**Multi-Dataset Target Configuration:**

```python
# Example 1: Same targets for all datasets
result = nirs4all.run(
    pipeline=[SNV(), PLSRegression(10)],
    dataset=["dataset1/", "dataset2/"],
    target="starch",                        # "starch" target from both datasets
)

# Example 2: Per-dataset target configuration
result = nirs4all.run(
    pipeline=[SNV(), PLSRegression(10)],
    dataset=["corn_data/", "wheat_data/"],
    target=[[0, 2], "starch"],              # indices [0,2] for corn, "starch" for wheat
)

# Example 3: Mixed indices and names
result = nirs4all.run(
    pipeline=[SNV(), PLSRegression(10)],
    dataset=[dataset1, dataset2, dataset3],
    target=[
        [0, 2],                              # Dataset 1: target indices 0 and 2
        "starch",                            # Dataset 2: target named "starch"
        ["protein", "moisture"],             # Dataset 3: multiple named targets
    ],
)
```

**Target Resolution for Multi-Dataset:**

| Scenario | `target` Parameter | Behavior |
|----------|-------------------|----------|
| Single dataset | `"starch"` | Select "starch" target |
| Multi-dataset, uniform | `"starch"` | Select "starch" from each dataset |
| Multi-dataset, per-dataset | `[[0,2], "starch"]` | First gets indices, second gets name |
| Multi-dataset, length mismatch | `[[0,2]]` with 3 datasets | Error: target list length must match |

**Validation Rules:**
- If `target` is a list of lists, its length must equal the number of datasets
- Each dataset must have the specified target(s)
- Target types (regression/classification) should be compatible across datasets

**Pipeline-level selection (overrides run-level):**
```python
pipeline = [
    SNV(),
    {"target": "pH"},           # Select single target for subsequent steps
    PLSRegression(10),
]
```

#### 3.2.4 Model-Specific Target Selection

Models can specify which target(s) they operate on via the `target` parameter in the model step dictionary. This enables **fine-grained control** within multivariate pipelines:

```python
# Single model targeting a specific column
pipeline = [
    SNV(),
    {"model": PLSRegression(n_components=10), "target": 2},       # Index
    {"model": PLSRegression(n_components=15), "target": "starch"}, # Name
]

# Different models for different targets in same pipeline
pipeline = [
    SNV(),
    {"model": PLSRegression(n_components=10), "target": "pH"},
    {"model": RandomForestRegressor(), "target": "quality"},
]

# Model targeting multiple specific targets (subset)
pipeline = [
    SNV(),
    {"model": PLSRegression(n_components=10), "target": [0, 2]},  # Indices
    {"model": Ridge(), "target": ["protein", "moisture"]},        # Names
]
```

**Model Step Target Syntax:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | Estimator | The sklearn-compatible model |
| `target` | `int`, `str`, `List[int]`, `List[str]` | Target(s) for this model |
| Other params | Any | Model hyperparameters (e.g., `n_components`) |

```python
# Full syntax example
{
    "model": PLSRegression(),
    "target": 2,              # or "my_target" or [0, 2] or ["a", "b"]
    "n_components": 10,       # Model hyperparameter
}
```

**Behavior:**
- If `target` specified: model trains only on those target(s)
- If `target` not specified: model uses context's selected targets (from pipeline `{"target": ...}` step or run-level)
- If no target selection anywhere: model uses all targets

**Use Cases:**
1. **Specialized models per target** - PLS for continuous, RF for categorical
2. **Hyperparameter tuning per target** - Different `n_components` per property
3. **Partial model training** - Train model on subset of available targets
4. **Sequential multi-target** - Chain models that each handle different targets

#### 3.2.5 Context Propagation

Target selection flows through `RuntimeContext`:

```python
# nirs4all/pipeline/context.py

class RuntimeContext:
    selected_targets: Optional[List[str]] = None  # Currently active targets
    missing_target_policy: str = "exclude"

    @property
    def y_train(self) -> np.ndarray:
        # Returns filtered Y based on selected_targets

    @property
    def effective_task_type(self) -> str:
        # Returns task type for selected targets (error if mixed)
```

#### 3.2.6 Target-Aware Operators

**Controller protocol:**
```python
class OperatorController:
    @classmethod
    def supports_multivariate(cls) -> bool:
        """Whether operator handles multiple targets."""
        return False  # Default conservative

    @classmethod
    def per_target_mode(cls) -> bool:
        """If True, apply operator separately to each target."""
        return True  # Default for transforms
```

**Operator behavior matrix:**

| Operator Type | `supports_multivariate` | `per_target_mode` | Behavior |
|--------------|------------------------|-------------------|----------|
| X-only (KS, XOutlier) | True | N/A | Works unchanged |
| Y-transform (Scaler) | True | True | Apply per column |
| YOutlierFilter | False | True | Filter per target |
| Y-splitters | False | False | Require target selection |
| Models | Model-dependent | N/A | Wrap if needed |

**Error handling:**
```python
def execute(self, step_info, dataset, context, ...):
    n_targets = dataset.num_targets
    if n_targets > 1 and not self.supports_multivariate():
        if context.selected_targets is None:
            raise MultiTargetError(
                f"{self.__class__.__name__} does not support multivariate Y. "
                "Specify target selection via run(target=...) or pipeline step."
            )
```

#### 3.2.7 Target Branching

Extend existing `branch` syntax with `by_target`:

```python
# Per-target model branches
pipeline = [
    SNV(),
    {"branch": {"by_target": True}},  # Creates branch per target
    PLSRegression(10),                 # Applied to each target branch
    {"merge": "concat"},               # Reassemble predictions
]

# Custom per-target pipelines
pipeline = [
    SNV(),
    {"branch": {"by_target": {
        "pH": [PLSRegression(10)],
        "quality": [RandomForestClassifier()],
    }}},
    {"merge": "concat"},
]
```

**Implementation in BranchController:**
```python
# Detect by_target branch
if "by_target" in branch_config:
    targets = dataset.target_names if branch_config["by_target"] is True \
              else list(branch_config["by_target"].keys())

    for target_name in targets:
        # Create sub-context with single target selected
        sub_context = context.with_target_selection([target_name])
        # Execute branch steps
        branch_results.append(execute_branch(steps, sub_context))
```

#### 3.2.8 Model Controller Updates

**Multi-output detection:**
```python
# nirs4all/controllers/models/sklearn_model.py

def _prepare_model(self, model, context):
    n_outputs = len(context.selected_targets or dataset.target_names)

    if n_outputs > 1 and not self._supports_multi_output(model):
        # Wrap in MultiOutputRegressor/Classifier
        if context.effective_task_type == "regression":
            return MultiOutputRegressor(model)
        else:
            return MultiOutputClassifier(model)
    return model

def _supports_multi_output(self, model) -> bool:
    # Check sklearn tags or known list
    return hasattr(model, '_get_tags') and model._get_tags().get('multioutput', False)
```

#### 3.2.9 Results Structure

**Per-target metrics:**
```python
# nirs4all/pipeline/result.py

class MultiTargetMetrics:
    per_target: Dict[str, Dict[str, float]]  # {target: {metric: value}}
    aggregate: Dict[str, float]               # Combined scores

    def __getattr__(self, name):
        # Backward compat: return aggregate for single-target
        if len(self.per_target) == 1:
            return self.per_target[next(iter(self.per_target))].get(name)
        return self.aggregate.get(name)

class RunResult:
    metrics: MultiTargetMetrics

    @property
    def best_rmse(self) -> float:
        return self.metrics.aggregate['rmse']

    def rmse(self, target: str = None) -> float:
        if target:
            return self.metrics.per_target[target]['rmse']
        return self.metrics.aggregate['rmse']
```

### 3.3 Loading Enhancements

#### DatasetConfigSchema updates

```python
# nirs4all/data/schema/config.py

class TargetColumnConfig(BaseModel):
    column: Union[int, str]           # Index or name
    name: Optional[str] = None        # Override name
    task_type: Optional[str] = None   # Override task type
    unit: Optional[str] = None

class DatasetConfigSchema(BaseModel):
    # Existing
    train_y: Optional[str]
    train_y_filter: Optional[List[int]]

    # New
    target_columns: Optional[List[Union[int, TargetColumnConfig]]] = None
    # Detailed target specification with metadata
```

**Example configuration:**
```yaml
train_x: data/spectra.csv
train_y: data/targets.csv
target_columns:
  - column: 0
    name: pH
    task_type: regression
    unit: pH
  - column: 1
    name: quality
    task_type: classification
```

### 3.4 Operator Fixes Required

#### YOutlierFilter

```python
# Current (broken)
y_flat = np.asarray(y).flatten()

# Fixed
def fit(self, X, y):
    y = np.atleast_2d(y)
    if y.shape[1] > 1:
        if self.target_index is not None:
            y = y[:, self.target_index:self.target_index+1]
        elif self.per_target:
            # Apply per-column, combine masks with OR
            masks = [self._compute_mask(y[:, i]) for i in range(y.shape[1])]
            return np.any(masks, axis=0)
        else:
            raise MultiTargetError("YOutlierFilter requires target selection")
```

#### TagController

```python
# Current (broken)
if y is not None and y.ndim > 1:
    y = y.flatten()

# Fixed: Remove flatten, pass 2D to filter
# Filter implementation handles multivariate
```

#### CrossValidatorController

```python
# Current (broken)
y = np.asarray(y).ravel()

# Fixed
if y.ndim > 1:
    if y.shape[1] > 1:
        # Use first target for binning, or aggregate
        y = y[:, 0]  # Or: y.mean(axis=1)
    else:
        y = y.ravel()
```

### 3.5 Backward Compatibility

**No breaking changes** for existing single-target workflows:

1. `target=None` (default) uses all targets
2. Single-target datasets work unchanged
3. `RunResult.best_rmse` returns aggregate (same as before for n_targets=1)
4. Operators that break on multivariate will error early with clear message

---

## 4. High-Level Design: Webapp

This section outlines multivariate target support for the React + FastAPI webapp.

### 4.1 Dataset Configuration UI

#### Progressive Disclosure Approach

**Basic Mode (Default):**
Simple column selection, similar to current implementation:

```
┌─────────────────────────────────────────────────────────┐
│ Target Columns                                          │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ ☑ Column 0 (pH)                                     │ │
│ │ ☑ Column 1 (quality)                                │ │
│ │ ☐ Column 2 (temperature)                            │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ [▼ Advanced target settings]                            │
└─────────────────────────────────────────────────────────┘
```

**Advanced Mode (Expanded):**
Full metadata configuration per target:

```
┌─────────────────────────────────────────────────────────┐
│ Target Columns                           [▲ Collapse]   │
├─────────────────────────────────────────────────────────┤
│ ☑ pH                                                    │
│   ├─ Name: [pH____________]                             │
│   ├─ Task: [Regression ▼]  Auto-detected: regression    │
│   └─ Unit: [pH____________]                             │
│                                                         │
│ ☑ quality                                               │
│   ├─ Name: [quality_______]                             │
│   ├─ Task: [Classification ▼]  Auto-detected: class.    │
│   └─ Unit: [_____________]                              │
│                                                         │
│ ☐ temperature (not selected as target)                  │
└─────────────────────────────────────────────────────────┘
```

#### Component Structure

```
src/components/datasets/
├── TargetColumnSelector.tsx      # Main component
├── TargetColumnBasic.tsx         # Simple checkbox list
├── TargetColumnAdvanced.tsx      # Full metadata editor
└── TargetColumnItem.tsx          # Single target config
```

**React component:**
```typescript
// src/components/datasets/TargetColumnSelector.tsx

interface TargetConfig {
  column: number | string;
  name?: string;
  taskType?: 'regression' | 'classification' | 'auto';
  unit?: string;
}

interface Props {
  columns: Array<{ index: number; name: string; inferredType: string }>;
  selected: TargetConfig[];
  onChange: (targets: TargetConfig[]) => void;
}

function TargetColumnSelector({ columns, selected, onChange }: Props) {
  const [advanced, setAdvanced] = useState(false);

  return (
    <div className="target-selector">
      {advanced ? (
        <TargetColumnAdvanced ... />
      ) : (
        <TargetColumnBasic ... />
      )}
      <button onClick={() => setAdvanced(!advanced)}>
        {advanced ? 'Collapse' : 'Advanced target settings'}
      </button>
    </div>
  );
}
```

### 4.2 Run Configuration UI

#### Target Selection in Pipeline Editor

Add target selection step in pipeline editor:

```
┌─────────────────────────────────────────────────────────┐
│ Pipeline: My Model                                      │
├─────────────────────────────────────────────────────────┤
│ [SNV]                                                   │
│   ↓                                                     │
│ [Target Selection]  ← NEW node type                     │
│   Target: [pH ▼] or [All targets]                       │
│   ↓                                                     │
│ [PLS Regression]                                        │
│   Components: 10                                        │
└─────────────────────────────────────────────────────────┘
```

**Or as run-level configuration:**

```
┌─────────────────────────────────────────────────────────┐
│ Run Configuration                                       │
├─────────────────────────────────────────────────────────┤
│ Dataset: [Corn Dataset ▼]                               │
│ Pipeline: [PLS Pipeline ▼]                              │
│                                                         │
│ Target Selection:                                       │
│ ○ All targets (pH, quality)                             │
│ ● Select targets:                                       │
│   ☑ pH                                                  │
│   ☐ quality                                             │
│                                                         │
│ Missing Target Policy:                                  │
│ ● Exclude rows with missing values                      │
│ ○ Keep rows (handle per-model)                          │
│                                                         │
│ [Run Training]                                          │
└─────────────────────────────────────────────────────────┘
```

#### Node Registry Addition

```typescript
// src/data/nodes/registry.ts

{
  type: 'target_selection',
  category: 'data',
  label: 'Target Selection',
  description: 'Select which target(s) to use for subsequent steps',
  config: {
    targets: {
      type: 'target_select',  // New config type
      label: 'Targets',
      multi: true,
    }
  }
}
```

### 4.3 Results Display

#### Per-Target Metrics Dashboard

```
┌─────────────────────────────────────────────────────────┐
│ Run Results: Corn Dataset - PLS Pipeline                │
├─────────────────────────────────────────────────────────┤
│ Overall Score: 0.92                                     │
│                                                         │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Per-Target Metrics                                  │ │
│ ├─────────────────────────────────────────────────────┤ │
│ │ Target    │ RMSE    │ R²      │ Task               │ │
│ ├───────────┼─────────┼─────────┼────────────────────┤ │
│ │ pH        │ 0.12    │ 0.95    │ Regression         │ │
│ │ quality   │ 0.89*   │ -       │ Classification     │ │
│ └─────────────────────────────────────────────────────┘ │
│ * Accuracy for classification tasks                     │
│                                                         │
│ [View Predictions] [Export Results] [Compare Runs]      │
└─────────────────────────────────────────────────────────┘
```

**Visualization per target:**
- Separate prediction vs actual plots per target
- Target switcher in visualization component
- Aggregated view option

### 4.4 Backend API Changes

#### Dataset Endpoints

```python
# api/datasets.py

class DatasetTargetConfig(BaseModel):
    column: Union[int, str]
    name: Optional[str] = None
    task_type: Optional[str] = None
    unit: Optional[str] = None

class DatasetCreateRequest(BaseModel):
    # Existing fields...
    target_columns: Optional[List[Union[int, DatasetTargetConfig]]] = None

@router.get("/datasets/{id}/targets")
async def get_dataset_targets(id: str) -> List[TargetInfo]:
    """Get target metadata for a dataset."""
    dataset = load_dataset(id)
    return [
        TargetInfo(
            name=t.name,
            index=t.index,
            task_type=t.task_type,
            unit=t.unit,
            inferred_type=infer_task_type(dataset.y()[:, t.index])
        )
        for t in dataset.target_registry.all()
    ]
```

#### Training Endpoints

```python
# api/training.py

class TrainingRequest(BaseModel):
    dataset_id: str
    pipeline_id: str
    # New fields
    targets: Optional[List[str]] = None          # Target names
    missing_target_policy: str = "exclude"

@router.post("/training/run")
async def run_training(request: TrainingRequest):
    result = nirs4all.run(
        pipeline=load_pipeline(request.pipeline_id),
        dataset=load_dataset(request.dataset_id),
        target=request.targets,
        missing_target_policy=request.missing_target_policy,
    )
    return format_result(result)
```

#### Results Endpoints

```python
# api/results.py

class TargetMetrics(BaseModel):
    target: str
    task_type: str
    metrics: Dict[str, float]  # rmse, r2, accuracy, etc.

class RunResultResponse(BaseModel):
    run_id: str
    aggregate_metrics: Dict[str, float]
    per_target_metrics: List[TargetMetrics]
    selected_targets: List[str]

@router.get("/runs/{id}/results")
async def get_run_results(id: str) -> RunResultResponse:
    run = load_run(id)
    return RunResultResponse(
        run_id=id,
        aggregate_metrics=run.metrics.aggregate,
        per_target_metrics=[
            TargetMetrics(target=t, task_type=..., metrics=m)
            for t, m in run.metrics.per_target.items()
        ],
        selected_targets=run.selected_targets,
    )
```

### 4.5 Workflow Integration

#### Target-Aware Pipeline Editor

Branch node updates to support `by_target`:

```typescript
// src/components/pipeline-editor/nodes/BranchNode.tsx

interface BranchConfig {
  mode: 'duplication' | 'separation';
  // Existing separation modes
  by_metadata?: string;
  by_tag?: { tag: string; values: Record<string, boolean> };
  // New
  by_target?: boolean | Record<string, PipelineStep[]>;
}
```

**Visual representation:**

```
     ┌──────────┐
     │ Branch   │
     │ by_target│
     └────┬─────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌───────┐   ┌───────┐
│ pH    │   │quality│
│ Branch│   │Branch │
└───────┘   └───────┘
```

### 4.6 Validation

#### Multi-Target Pipeline Validation

Add validation rules for multivariate scenarios:

```typescript
// src/components/pipeline-editor/validation/rules.ts

const multiTargetRules: ValidationRule[] = [
  {
    id: 'y-operator-needs-target-selection',
    severity: 'error',
    check: (pipeline, dataset) => {
      if (dataset.targets.length <= 1) return [];

      const yDependentOps = findYDependentOps(pipeline);
      const hasTargetSelection = hasTargetSelectionStep(pipeline);

      if (yDependentOps.length > 0 && !hasTargetSelection) {
        return [{
          nodeId: yDependentOps[0].id,
          message: `${yDependentOps[0].label} requires target selection for multi-target datasets`,
          suggestion: 'Add a Target Selection step before this operator',
        }];
      }
      return [];
    }
  },
  {
    id: 'mixed-task-type-warning',
    severity: 'warning',
    check: (pipeline, dataset) => {
      const targets = getSelectedTargets(pipeline, dataset);
      const taskTypes = new Set(targets.map(t => t.task_type));

      if (taskTypes.size > 1) {
        return [{
          message: 'Selected targets have mixed task types (regression + classification)',
          suggestion: 'Consider using target branching for different model types',
        }];
      }
      return [];
    }
  }
];
```

---

## 5. High-Level Roadmap

This section outlines the implementation phases without specific code details.

### 5.1 Phase Overview

```
Phase 1: Foundation          Phase 2: Operators      Phase 3: Pipeline DSL
├─ TargetRegistry            ├─ Fix Y-dependent      ├─ Target selection step
├─ Target selection in API   │   operators           ├─ by_target branching
├─ Context propagation       ├─ Per-target mode      └─ Merge strategies
└─ Basic validation          └─ Error handling

Phase 4: Results             Phase 5: Webapp         Phase 6: Documentation
├─ MultiTargetMetrics        ├─ Dataset config UI    ├─ User guide updates
├─ Per-target scores         ├─ Run config UI        ├─ API reference
├─ Aggregation methods       ├─ Results dashboard    └─ Migration guide
└─ Export format updates     └─ Validation rules
```

### 5.2 Phase 1: Foundation

**Goal**: Establish core infrastructure for multivariate target handling.

**Components**:
1. **TargetRegistry class** - Store target metadata (name, task_type, unit)
2. **SpectroDataset integration** - `target_names` property, `y(targets=...)` parameter
3. **API parameter** - `nirs4all.run(target=...)` and `missing_target_policy`
4. **RuntimeContext updates** - `selected_targets`, `effective_task_type`
5. **DatasetConfigSchema** - `target_columns` configuration

**Deliverables**:
- Users can declare target metadata at load time
- Users can select targets at run time
- Single-target workflows unchanged

**Dependencies**: None

---

### 5.3 Phase 2: Operator Fixes

**Goal**: Make Y-dependent operators multivariate-aware.

**Components**:
1. **YOutlierFilter** - Per-target mode, remove flatten
2. **TagController** - Remove explicit flatten
3. **SystematicCircularSplitter** - Target selection or aggregate
4. **BinnedStratifiedGroupKFold** - Handle 2D Y properly
5. **CrossValidatorController** - Fix `_bin_y_for_groups`
6. **Controller protocol** - `supports_multivariate()`, `per_target_mode()`

**Deliverables**:
- All Y-dependent operators handle multivariate gracefully
- Clear error messages when target selection required
- Per-target application option where applicable

**Dependencies**: Phase 1

---

### 5.4 Phase 3: Pipeline DSL

**Goal**: Enable target-specific pipeline configurations.

**Components**:
1. **Target selection step** - `{"target": "pH"}` in pipeline
2. **Branch by_target** - `{"branch": {"by_target": True}}`
3. **Custom per-target pipelines** - `{"by_target": {"pH": [...], "quality": [...]}}`
4. **Merge strategies** - `{"merge": "concat"}` for target branches
5. **TargetSelectionController** - New controller for target step

**Deliverables**:
- Flexible per-target model architectures
- Stacking across targets
- Mixed task type pipelines

**Dependencies**: Phase 1, Phase 2

---

### 5.5 Phase 4: Results Structure

**Goal**: Comprehensive per-target and aggregate metrics.

**Components**:
1. **MultiTargetMetrics class** - Per-target + aggregate storage
2. **RunResult updates** - `rmse(target=...)`, `metrics.per_target`
3. **Aggregation methods** - Weighted average, task-specific
4. **Export updates** - CSV/JSON with per-target breakdown
5. **PredictResult updates** - Per-target predictions

**Deliverables**:
- Per-target metric access
- Backward-compatible aggregate access
- Rich export formats

**Dependencies**: Phase 1

---

### 5.6 Phase 5: Webapp Integration

**Goal**: Full multivariate support in the webapp UI.

**Components**:
1. **Dataset configuration** - Progressive disclosure target UI
2. **Run configuration** - Target selection, missing policy
3. **Pipeline editor** - Target selection node, by_target branch
4. **Results dashboard** - Per-target metrics, visualizations
5. **Validation rules** - Multi-target specific checks
6. **Backend endpoints** - Targets API, updated training/results

**Deliverables**:
- Intuitive multi-target dataset setup
- Visual target selection in pipelines
- Rich per-target results display

**Dependencies**: Phase 1, Phase 3, Phase 4

---

### 5.7 Phase 6: Documentation & Polish

**Goal**: Complete documentation and migration support.

**Components**:
1. **User guide** - Multi-target workflows, examples
2. **API reference** - New parameters, classes
3. **Migration guide** - Upgrading existing pipelines
4. **Examples** - Multi-target regression, mixed task types
5. **Webapp tutorials** - Dataset setup, target branching

**Deliverables**:
- Comprehensive multi-target documentation
- Example notebooks
- Video tutorials (optional)

**Dependencies**: Phase 1-5

---

### 5.8 Dependency Graph

```
                    ┌─────────────┐
                    │  Phase 1    │
                    │ Foundation  │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  Phase 2    │ │  Phase 3    │ │  Phase 4    │
    │  Operators  │ │ Pipeline DSL│ │  Results    │
    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
           │               │               │
           └───────────────┼───────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Phase 5    │
                    │   Webapp    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Phase 6    │
                    │Documentation│
                    └─────────────┘
```

### 5.9 Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing workflows | High | Comprehensive backward compat tests |
| Performance with many targets | Medium | Lazy evaluation, caching |
| Complex validation logic | Medium | Incremental rollout, feature flags |
| Mixed task type edge cases | Medium | Clear error messages, documentation |
| Webapp complexity increase | Low | Progressive disclosure, good UX |

### 5.10 Success Criteria

**Phase 1 Complete When**:
- [ ] Can load dataset with multiple named targets
- [ ] Can run pipeline on selected target subset
- [ ] Existing single-target tests pass

**Phase 2 Complete When**:
- [ ] All Y-dependent operators work with multivariate Y
- [ ] Clear error messages for unsupported scenarios
- [ ] New operator tests for multi-target cases

**Phase 3 Complete When**:
- [ ] `{"target": "pH"}` step works in pipeline
- [ ] `{"branch": {"by_target": True}}` creates per-target branches
- [ ] Can train different models per target

**Phase 4 Complete When**:
- [ ] Per-target metrics accessible in results
- [ ] Aggregate scores computed correctly
- [ ] Export includes per-target breakdown

**Phase 5 Complete When**:
- [ ] Webapp dataset config supports target metadata
- [ ] Webapp run config supports target selection
- [ ] Webapp results show per-target metrics

**Phase 6 Complete When**:
- [ ] User guide covers multi-target workflows
- [ ] Examples demonstrate key scenarios
- [ ] Migration guide helps existing users

---

## 6. Additional Enhancements

This section covers additional features identified during design review.

### 6.1 Target Correlation Analysis

**Scope**: Optional feature for analyzing relationships between targets.

**Library (nirs4all)**:
- Parameter: `analyze_correlations=True` in `run()` or separate `nirs4all.analyze_targets(dataset)`
- Returns correlation matrix without polluting main API
- Stored in result metadata, accessible via `result.target_analysis.correlations`

```python
# Option 1: During run
result = nirs4all.run(
    pipeline=[SNV(), PLSRegression(10)],
    dataset="data/",
    analyze_correlations=True,  # Optional
)
print(result.target_analysis.correlations)  # Returns correlation matrix

# Option 2: Standalone analysis
analysis = nirs4all.analyze_targets("data/")
print(analysis.correlations)
```

**Webapp**:
- "Analyze Targets" button in dataset view
- Displays correlation matrix heatmap
- Provides suggestions: "pH and moisture (r=0.85) - consider joint PLS modeling"

### 6.2 Target Weighting for Aggregate Metrics

**Default**: Equal weighting (current behavior preserved).

**Configurable weights**:
```python
result = nirs4all.run(
    pipeline=[SNV(), PLSRegression(10)],
    dataset="data/",
    target_weights={"pH": 3.0, "moisture": 1.0},  # pH matters 3x more
)
# Or with presets
result = nirs4all.run(..., target_weights="by_variance")  # Weight by inverse variance
```

**Preset options**:
| Preset | Description |
|--------|-------------|
| `"equal"` | All targets weighted equally (default) |
| `"by_variance"` | Inverse variance weighting |
| `"by_range"` | Normalize by target range |

**Per-target leaderboard**: Extend existing `Predictions.top()` mechanism:
```python
# Existing behavior (works with multivariate)
result.top(5)                    # Top 5 by aggregate score

# Extended for multivariate
result.top(5, target="pH")       # Top 5 for pH specifically
result.top(5, target="all")      # Top 5 for each target (returns dict)
```

### 6.3 Per-Target Y Processing

Different transformations can be applied to different targets:

```python
# Per-target y_processing with dict syntax
pipeline = [
    SNV(),
    {"y_processing": {
        "pH": StandardScaler(),
        "concentration": LogTransform(),
        0: MinMaxScaler(),           # Index-based also supported
    }},
    PLSRegression(10),
]

# Sequential transforms (existing behavior, applies to ALL targets)
pipeline = [
    {"y_processing": [StandardScaler(), PowerTransformer()]},  # Applied sequentially
]
```

**Parsing Rules** (to avoid ambiguity):
| Syntax | Interpretation |
|--------|---------------|
| `{"y_processing": StandardScaler()}` | Single transform, all targets |
| `{"y_processing": [A(), B()]}` | Sequential transforms, all targets |
| `{"y_processing": {"pH": A()}}` | Per-target transform (dict with str/int keys) |
| `{"y_processing": {0: A(), 1: B()}}` | Per-target transform by index |

**Validation**: When parsing pipeline, check if dict keys are strings/ints (per-target) vs other types (error).

### 6.4 Multi-Output Model Handling

**Policy**: Require explicit declaration when model doesn't support multi-output natively.

#### Option 1: Manual sklearn wrapping (verbose)

```python
from sklearn.multioutput import MultiOutputRegressor

pipeline = [
    SNV(),
    {"model": MultiOutputRegressor(SVR()), "target": ["pH", "moisture"]},
]
```

#### Option 2: `multi_output` keyword (recommended)

New pipeline keyword `multi_output` provides explicit, declarative multi-output handling:

**Same model for all targets:**
```python
pipeline = [
    SNV(),
    {"multi_output": SVR()},  # Internally wraps in MultiOutputRegressor
]

# Equivalent to branching by_target with same model, but explicit intent
```

**Different models per target (dict syntax):**
```python
pipeline = [
    SNV(),
    {"multi_output": {
        "pH": PLSRegression(10),
        "quality": RandomForestRegressor(),
        0: SVR(),                          # Index-based also supported
    }},
]
```

**List syntax with target/model pairs:**
```python
pipeline = [
    SNV(),
    {"multi_output": [
        {"target": "pH", "model": PLSRegression(10)},
        {"target": ["protein", "moisture"], "model": Ridge()},  # One model for multiple targets
        {"target": 2, "model": SVR()},
    ]},
]
```

**Comparison with `branch` + `by_target`:**

| Feature | `multi_output` | `branch` + `by_target` |
|---------|---------------|------------------------|
| Intent | Explicit multi-output modeling | General-purpose branching |
| Output | Single merged prediction array | Requires explicit `merge` step |
| Use case | One-liner multi-output setup | Complex per-target pipelines |
| Preprocessing | Shared (before step) | Can differ per branch |

```python
# multi_output: simple, explicit
pipeline = [
    SNV(),
    {"multi_output": {"pH": PLS(10), "quality": RF()}},
]

# branch + by_target: flexible, verbose
pipeline = [
    SNV(),
    {"branch": {"by_target": {
        "pH": [PLSRegression(10)],
        "quality": [RandomForestRegressor()],
    }}},
    {"merge": "concat"},
]
```

**Error handling**: If a model doesn't support multi-output and is used in a multi-target context without `multi_output` keyword, raise clear error with suggestion:

```
MultiTargetError: SVR() does not support multi-output natively.
Use {"multi_output": SVR()} or wrap manually with MultiOutputRegressor(SVR()).
```

### 6.5 Multi-Dataset Independence

Multi-dataset runs form an **independent matrix** (pipeline × datasets):

```
Datasets:     [corn_data, wheat_data, soy_data]
                   ↓           ↓          ↓
Pipeline:     ─────┬───────────┬──────────┬─────
                   │           │          │
Results:      [result_1]  [result_2]  [result_3]
```

**Target specification for multi-dataset**:
- Single value/list: applied uniformly to all datasets
- List of lists (same length as datasets): per-dataset configuration

```python
# Uniform targets across datasets
nirs4all.run(pipeline, [ds1, ds2, ds3], target="starch")

# Per-dataset targets (list of lists)
nirs4all.run(pipeline, [ds1, ds2, ds3], target=[
    [0, 2],        # ds1: indices 0 and 2
    "starch",      # ds2: named target
    ["a", "b"],    # ds3: multiple named targets
])
```

**No automatic target alignment**: Datasets are independent; each must have its specified targets.

---

## Appendix: Design Questions Resolved

| Question | Decision |
|----------|----------|
| Target selection scope | Both run-level and pipeline DSL |
| Mixed task types | Auto-detect per target, allow override |
| Operator fallback on multivariate | Error if no target selection; per-target mode optional |
| Missing target handling | Configurable (default: exclude rows) |
| Metrics reporting | Both per-target and aggregate |
| Target branching syntax | Reuse existing `branch` with `by_target` |
| Multi-output models | Explicit via `multi_output` keyword or manual wrapping |
| Webapp UI | Progressive disclosure |
| Target names/types | Auto-detected by default, overridable |
| Multi-dataset runs | Independent matrix (pipeline × datasets) |
| Target correlation analysis | Optional param in library, button in webapp |
| Target weighting | Configurable + presets, equal by default |
| Hierarchical targets | Not supported (keep flat) |
| Per-target y_processing | Supported via dict syntax |
| Cross-target QC | Not supported |
| Prediction chaining | Not supported |
| Per-target uncertainty | Not supported |
| Target aliases | Not supported (auto-names sufficient) |
| Per-target leaderboard | Extend existing `Predictions.top()` mechanism |
