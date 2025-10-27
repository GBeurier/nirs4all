## Executive Summary

This specification proposes a centralized metrics and loss management system for NIRS4ALL to replace the current fragmented approach where loss/metrics are defined locally across different controllers and frameworks.

**Core Principle**: Define loss and metrics once (globally or per-pipeline), propagate automatically to all components, compute consistently across all frameworks.

---

## Current State Analysis

### Current Tipping Points

#### 1. **Loss/Metrics Definition Locations**

| Location | Current Behavior | Issues |
|----------|------------------|--------|
| model_utils.py | Hardcoded defaults per TaskType | No user control, framework-agnostic |
| op_model.py (L122-129) | Reads `train_params['loss']`, defaults to 'MSELoss' | Local logic, string-based |
| base_model_controller.py | No explicit loss management | Relies on model defaults |
| evaluator.py | Provides metric calculation functions | Good API but not integrated |
| Pipeline config | No loss/metrics fields | Users can't specify preferences |

#### 2. **Metric Calculation Points**

| Location | Framework | Current Behavior |
|----------|-----------|------------------|
| `PipelineRunner.run()` | All | Implicit via controllers |
| `Predictions.calculate_scores()` (L1189) | Post-hoc | Hardcoded metric lists |
| `TabReportManager` (L186) | Reporting | Hardcoded based on task_type |
| Model controllers | Framework-specific | Each implements own logic |

#### 3. **Framework-Specific Loss Handling**

| Framework | Loss Definition | Metric Calculation |
|-----------|----------------|-------------------|
| **scikit-learn** | Implicit in model | `model_utils.calculate_scores()` |
| **TensorFlow/Keras** | `model.compile(loss=...)` | Built-in or custom |
| **PyTorch** | `torch.nn.MSELoss()` etc. | Manual in training loop |
| **Functions (nicon)** | [`train_params['loss']`](nirs4all/controllers/torch/op_model.py) | Framework-specific |

#### 4. **Default Logic Issues**

```python
# Current scattered defaults:
ModelUtils.DEFAULT_LOSSES = {
    TaskType.REGRESSION: "mse",
    TaskType.BINARY_CLASSIFICATION: "binary_crossentropy",
    TaskType.MULTICLASS_CLASSIFICATION: "categorical_crossentropy"
}

ModelUtils.DEFAULT_METRICS = {
    TaskType.REGRESSION: ["mae", "mse"],
    TaskType.BINARY_CLASSIFICATION: ["accuracy", "auc"],
    TaskType.MULTICLASS_CLASSIFICATION: ["accuracy", "categorical_accuracy"]
}

# But also:
# - torch/op_model.py: loss_fn_config = train_params.get('loss', 'MSELoss')
# - predictions.py: metrics = ['mse', 'r2', 'mae']
# - tab_report_manager.py: hardcoded metric lists per task_type
```

---

## Proposed Architecture

### Design Principles

1. **Single Source of Truth**: One class manages all loss/metrics logic
2. **Hierarchical Configuration**: Global defaults → Pipeline config → Step-level overrides
3. **Framework Abstraction**: Unified API that translates to framework-specific implementations
4. **Extensibility**: Easy to add custom losses/metrics
5. **Validation**: Check compatibility (loss vs task_type, framework support)

---

## Core Components

### 1. **MetricsConfig Class** (NEW)

**File**: config.py (new module)

```python
@dataclass
class MetricsConfig:
    """
    Central configuration for loss functions and metrics.

    Hierarchy:
    1. Step-level config (highest priority)
    2. Pipeline-level config
    3. Global defaults (lowest priority)
    """

    # Loss configuration
    loss: Optional[str] = None
    loss_params: Dict[str, Any] = field(default_factory=dict)

    # Metrics configuration
    metrics: Optional[List[str]] = None
    primary_metric: Optional[str] = None  # For optimization/ranking

    # Task context
    task_type: Optional[TaskType] = None
    framework: Optional[str] = None

    # Behavior
    auto_infer_task: bool = True
    validate_compatibility: bool = True

    def __post_init__(self):
        """Apply defaults and validate."""
        if self.loss is None and self.task_type:
            self.loss = self._get_default_loss()
        if self.metrics is None and self.task_type:
            self.metrics = self._get_default_metrics()
        if self.primary_metric is None:
            self.primary_metric = self._infer_primary_metric()

    def merge(self, other: 'MetricsConfig') -> 'MetricsConfig':
        """Merge with another config (other takes precedence)."""
        pass

    def to_framework_format(self, framework: str) -> Dict[str, Any]:
        """Convert to framework-specific loss/metrics objects."""
        pass
```

### 2. **MetricsRegistry Class** (NEW)

**File**: `nirs4all/metrics/registry.py`

```python
class MetricsRegistry:
    """
    Registry for available losses and metrics with framework mappings.

    Responsibilities:
    - Validate loss/metric names
    - Map unified names to framework-specific implementations
    - Check framework compatibility
    - Provide metric calculation functions
    """

    # Unified name → framework implementations
    LOSS_MAPPINGS = {
        'mse': {
            'sklearn': 'squared_error',
            'tensorflow': 'mse',
            'torch': 'MSELoss',
            'function': 'mean_squared_error'
        },
        'binary_crossentropy': {
            'sklearn': 'log_loss',
            'tensorflow': 'binary_crossentropy',
            'torch': 'BCELoss',
        },
        # ... more losses
    }

    METRIC_MAPPINGS = {
        'mae': {
            'sklearn': 'mean_absolute_error',
            'tensorflow': 'mae',
            'torch': 'manual',  # compute via evaluator
            'function': evaluator.eval
        },
        # ... more metrics
    }

    @classmethod
    def get_loss(cls, loss_name: str, framework: str, **params) -> Any:
        """Get framework-specific loss function/class."""
        pass

    @classmethod
    def get_metric(cls, metric_name: str, framework: str) -> Callable:
        """Get framework-specific metric calculator."""
        pass

    @classmethod
    def validate_loss(cls, loss_name: str, task_type: TaskType, framework: str) -> bool:
        """Check if loss is valid for task/framework."""
        pass

    @classmethod
    def compute_metrics(cls, y_true, y_pred, metrics: List[str], task_type: TaskType) -> Dict[str, float]:
        """Compute all metrics using evaluator."""
        return {m: evaluator.eval(y_true, y_pred, m) for m in metrics}
```

### 3. **Modified PipelineConfigs** (UPDATED)

**File**: config.py

```python
class PipelineConfigs:
    def __init__(
        self,
        steps: Any,
        name: str = "",
        # NEW FIELDS
        metrics_config: Optional[MetricsConfig] = None,
        loss: Optional[str] = None,  # Convenience shorthand
        metrics: Optional[List[str]] = None,  # Convenience shorthand
        primary_metric: Optional[str] = None,
        **kwargs
    ):
        """
        Added metrics configuration at pipeline level.

        Args:
            metrics_config: Full MetricsConfig object
            loss: Shorthand for metrics_config.loss
            metrics: Shorthand for metrics_config.metrics
            primary_metric: Shorthand for metrics_config.primary_metric
        """
        self.metrics_config = metrics_config or MetricsConfig(
            loss=loss,
            metrics=metrics,
            primary_metric=primary_metric
        )
```

### 4. **Modified PipelineRunner** (UPDATED)

**File**: runner.py

```python
class PipelineRunner:
    def run(
        self,
        config: Union[PipelineConfigs, List],
        dataset: Union[DatasetConfigs, str],
        # NEW: Optional override
        metrics_config: Optional[MetricsConfig] = None,
        **kwargs
    ):
        """
        Run pipeline with metrics configuration propagation.

        Priority:
        1. metrics_config parameter (highest)
        2. config.metrics_config
        3. Auto-infer from dataset.task_type
        """

        # Establish metrics config
        effective_config = self._resolve_metrics_config(
            override=metrics_config,
            pipeline=config.metrics_config if hasattr(config, 'metrics_config') else None,
            dataset=dataset
        )

        # Store in execution context
        self.execution_context.metrics_config = effective_config

        # Propagate to controllers during step execution
        # ...
```

### 5. **Modified Controllers** (UPDATED)

**File**: All controller base classes

```python
class BaseModelController:
    def train(self, X, y, step_config, execution_context, **kwargs):
        """
        Training now receives metrics_config from execution context.
        """
        metrics_config = execution_context.metrics_config

        # Get step-level override if present
        if 'loss' in step_config or 'metrics' in step_config:
            step_metrics = MetricsConfig(
                loss=step_config.get('loss'),
                metrics=step_config.get('metrics'),
                task_type=metrics_config.task_type,
                framework=self.get_framework_name()
            )
            metrics_config = metrics_config.merge(step_metrics)

        # Convert to framework format
        framework_config = metrics_config.to_framework_format(self.get_framework_name())

        # Apply to model training
        self._train_with_config(X, y, framework_config, **kwargs)
```

**PyTorch Example** (`nirs4all/controllers/torch/op_model.py`):

```python
def train(self, X, y, step_config, execution_context, **kwargs):
    metrics_config = execution_context.metrics_config

    # Get loss function from registry
    loss_fn = MetricsRegistry.get_loss(
        metrics_config.loss,
        framework='torch',
        **metrics_config.loss_params
    )

    # Training loop uses loss_fn
    for epoch in range(epochs):
        loss = loss_fn(y_pred, y_true)
        # ...

    # Compute metrics using registry
    scores = MetricsRegistry.compute_metrics(
        y_true, y_pred,
        metrics=metrics_config.metrics,
        task_type=metrics_config.task_type
    )

    return model, scores
```

### 6. **Modified Predictions** (UPDATED)

**File**: predictions.py

```python
class Predictions:
    def calculate_scores(self, metrics: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Calculate scores using MetricsRegistry instead of hardcoded lists.
        """
        if metrics is None:
            # Use pipeline-level metrics if available
            metrics = self._get_default_metrics()

        scores_data = []
        for row in self._df.to_dicts():
            y_true = np.array(json.loads(row["y_true"]))
            y_pred = np.array(json.loads(row["y_pred"]))

            # Use MetricsRegistry
            scores = MetricsRegistry.compute_metrics(
                y_true, y_pred,
                metrics=metrics,
                task_type=row["task_type"]
            )
            # ...
```

### 7. **Pipeline JSON/YAML Format** (EXTENDED)

**File**: Pipeline config files

```json
{
  "pipeline": [
    {"class": "sklearn.preprocessing.MinMaxScaler"},
    {"class": "sklearn.model_selection.ShuffleSplit", "params": {"n_splits": 3}},
    {
      "name": "PLS-10",
      "model": {"class": "sklearn.cross_decomposition.PLSRegression", "params": {"n_components": 10}}
    }
  ],

  "metrics": {
    "loss": "mse",
    "metrics": ["mae", "rmse", "r2", "bias"],
    "primary_metric": "rmse"
  }
}
```

**Step-level override**:

```json
{
  "name": "Custom-NN",
  "model": {"function": "nirs4all.operators.models.cirad_tf.customizable_nicon"},
  "loss": "huber",
  "loss_params": {"delta": 0.5},
  "metrics": ["mae", "mse"],
  "train_params": {"epochs": 100}
}
```

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)

**Files to Create**:
- __init__.py
- config.py (MetricsConfig)
- `nirs4all/metrics/registry.py` (MetricsRegistry)

**Changes**:
1. Create MetricsConfig dataclass
2. Implement MetricsRegistry with mappings for common losses/metrics
3. Write unit tests for registry lookup

**Deliverables**:
- [ ] MetricsConfig with merge logic
- [ ] MetricsRegistry with sklearn/tensorflow/torch mappings
- [ ] Validation functions (loss vs task_type)
- [ ] Tests for framework translation

### Phase 2: Pipeline Integration (Week 2)

**Files to Modify**:
- config.py
- runner.py

**Changes**:
1. Add `metrics_config` parameter to PipelineConfigs
2. Implement `_resolve_metrics_config()` in PipelineRunner
3. Store MetricsConfig in execution_context
4. Update JSON/YAML serialization/deserialization

**Deliverables**:
- [ ] PipelineConfigs accepts loss/metrics parameters
- [ ] PipelineRunner propagates config to context
- [ ] Pipeline JSON format supports "metrics" field
- [ ] Backward compatibility (existing pipelines work without metrics field)

### Phase 3: Controller Updates (Week 3)

**Files to Modify**:
- base_model_controller.py
- op_model.py
- TensorFlow/Keras controllers
- Sklearn controllers

**Changes**:
1. Modify controller train() signatures to accept execution_context
2. Extract metrics_config from context
3. Convert to framework-specific format
4. Replace hardcoded loss/metrics logic

**Deliverables**:
- [ ] PyTorch controllers use MetricsRegistry.get_loss()
- [ ] TensorFlow controllers use MetricsRegistry.get_loss()
- [ ] All controllers compute metrics via MetricsRegistry.compute_metrics()
- [ ] Remove local loss defaults

### Phase 4: Predictions & Reporting (Week 4)

**Files to Modify**:
- predictions.py
- tab_report_manager.py
- model_utils.py

**Changes**:
1. Update Predictions.calculate_scores() to use MetricsRegistry
2. Deprecate hardcoded metric lists in TabReportManager
3. Refactor ModelUtils to delegate to MetricsRegistry
4. Store metrics_config in prediction metadata

**Deliverables**:
- [ ] Predictions use pipeline-defined metrics
- [ ] Reports show all computed metrics
- [ ] Metadata includes metrics_config used
- [ ] Deprecation warnings for old logic

### Phase 5: Documentation & Migration (Week 5)

**Files to Create/Update**:
- `docs/METRICS_CONFIGURATION.md` (new)
- WRITING_A_PIPELINE.md
- README.md

**Changes**:
1. Document new metrics configuration API
2. Provide migration guide for existing code
3. Add examples for custom losses/metrics
4. Update all pipeline examples

**Deliverables**:
- [ ] Comprehensive metrics docs
- [ ] Migration guide with code examples
- [ ] Updated pipeline examples
- [ ] Deprecation timeline for old APIs

---

## API Examples

### Example 1: Pipeline-Level Configuration

```python
from nirs4all import PipelineRunner, PipelineConfigs
from nirs4all.metrics import MetricsConfig

pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=5),
    {"model": PLSRegression(n_components=10), "name": "PLS-10"}
]

config = PipelineConfigs(
    pipeline,
    name="my_pipeline",
    loss="mse",
    metrics=["mae", "rmse", "r2", "bias", "sep"],
    primary_metric="rmse"
)

runner = PipelineRunner()
predictions = runner.run(config, dataset)
# All models use MSE loss, all predictions compute the 5 metrics
```

### Example 2: Step-Level Override

```python
pipeline = [
    MinMaxScaler(),
    {
        "model": customizable_nicon,
        "name": "NN-Huber",
        "loss": "huber",
        "loss_params": {"delta": 0.5},
        "metrics": ["mae", "mse"],  # Fewer metrics for this model
        "train_params": {"epochs": 100}
    }
]

config = PipelineConfigs(
    pipeline,
    name="mixed_pipeline",
    loss="mse",  # Default for other models
    metrics=["mae", "rmse", "r2"]
)
```

### Example 3: Runtime Override

```python
from nirs4all.metrics import MetricsConfig

# Override at run time
custom_metrics = MetricsConfig(
    loss="huber",
    metrics=["mae", "mse", "max_error"],
    primary_metric="mae"
)

runner.run(config, dataset, metrics_config=custom_metrics)
```

### Example 4: Custom Metric Registration

```python
from nirs4all.metrics import MetricsRegistry

def my_custom_metric(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / y_true)

MetricsRegistry.register_metric(
    name="mape_custom",
    function=my_custom_metric,
    task_types=["regression"]
)

# Now use it in pipeline
config = PipelineConfigs(
    pipeline,
    metrics=["mae", "rmse", "mape_custom"]
)
```

---

## Backward Compatibility Strategy

### Migration Path

1. **Deprecation Phase** (3 months):
   - Old APIs still work but emit warnings
   - Documentation updated with migration examples

2. **Dual Support** (6 months):
   - Both old and new APIs functional
   - Internal code refactored to use new system

3. **Removal** (v1.0):
   - Old APIs removed
   - Only MetricsConfig approach supported

### Compatibility Code

```python
# In PipelineRunner.run()
def run(self, config, dataset, **kwargs):
    # Support old 'metric' parameter
    if 'metric' in kwargs:
        warnings.warn(
            "Parameter 'metric' is deprecated. Use 'metrics_config' instead.",
            DeprecationWarning
        )
        # Convert old format to new
        metrics_config = MetricsConfig(
            primary_metric=kwargs.pop('metric'),
            metrics=self._infer_metrics_from_primary(kwargs['metric'])
        )
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/test_metrics_config.py`

```python
def test_metrics_config_defaults():
    """Test default loss/metrics for each task type."""
    config = MetricsConfig(task_type=TaskType.REGRESSION)
    assert config.loss == "mse"
    assert "mae" in config.metrics

def test_metrics_config_merge():
    """Test hierarchical config merging."""
    global_config = MetricsConfig(loss="mse", metrics=["mae", "r2"])
    step_config = MetricsConfig(loss="huber")
    merged = global_config.merge(step_config)
    assert merged.loss == "huber"
    assert merged.metrics == ["mae", "r2"]

def test_registry_framework_mapping():
    """Test loss/metric name translation."""
    sklearn_loss = MetricsRegistry.get_loss("mse", "sklearn")
    assert sklearn_loss == "squared_error"

    torch_loss = MetricsRegistry.get_loss("mse", "torch")
    assert torch_loss.__name__ == "MSELoss"
```

### Integration Tests

**File**: `tests/test_pipeline_metrics_integration.py`

```python
def test_pipeline_with_custom_metrics():
    """Test end-to-end pipeline with custom metrics."""
    config = PipelineConfigs(
        [MinMaxScaler(), PLSRegression()],
        loss="mae",
        metrics=["mae", "rmse", "r2", "bias"]
    )

    predictions = runner.run(config, dataset)

    # Check all metrics computed
    assert "mae" in predictions.get_best().keys()
    assert "bias" in predictions.get_best().keys()

def test_step_level_override():
    """Test step-level loss override."""
    pipeline = [
        {"model": PLSRegression(), "loss": "mae"},
        {"model": RandomForest(), "loss": "mse"}
    ]
    config = PipelineConfigs(pipeline, loss="rmse")

    # Both models should use their specific losses, not global
    # ... verify via mock controllers
```

---

## Benefits

### 1. **User Experience**
- ✅ One place to define loss/metrics (pipeline config)
- ✅ No need to understand framework-specific naming
- ✅ Consistent results across frameworks
- ✅ Easy to compare models with same metrics

### 2. **Code Quality**
- ✅ No duplicated metric logic
- ✅ Single source of truth (MetricsRegistry)
- ✅ Easy to extend (register new losses/metrics)
- ✅ Better testability

### 3. **Flexibility**
- ✅ Global defaults + step-level overrides
- ✅ Runtime configuration changes
- ✅ Custom metrics support
- ✅ Framework-agnostic API

### 4. **Maintainability**
- ✅ Clear separation of concerns
- ✅ Easier debugging (one place to check)
- ✅ Simpler controller code
- ✅ Less framework-specific knowledge required

---

## Open Questions

1. **Should we support callable losses/metrics directly?**
   ```python
   config = PipelineConfigs(pipeline, loss=my_custom_loss_fn)
   ```
   → **Proposal**: Yes, via MetricsRegistry.register()

2. **How to handle framework-specific parameters?**
   ```python
   # PyTorch BCEWithLogitsLoss has pos_weight
   config = MetricsConfig(loss="bce_with_logits", loss_params={"pos_weight": [1.2]})
   ```
   → **Proposal**: Pass through via `loss_params` dict

3. **Should metrics be computed during training or post-hoc?**
   → **Proposal**: Both (training metrics for monitoring, post-hoc via Predictions for comparison)

4. **How to handle framework incompatibilities?**
   (e.g., scikit-learn doesn't support custom losses)
   → **Proposal**: Raise informative error, suggest alternatives

---

## Success Criteria

✅ Users can define loss/metrics once in pipeline config
✅ All frameworks (sklearn, TF, PyTorch) respect the configuration
✅ No hardcoded metric lists anywhere in the codebase
✅ Backward compatibility maintained for 6 months
✅ Full test coverage (unit + integration)
✅ Documentation complete with migration guide
✅ Zero regression in existing functionality

---

## File Tree Changes

```
nirs4all/
├── metrics/                    # NEW MODULE
│   ├── __init__.py
│   ├── config.py               # MetricsConfig
│   └── registry.py             # MetricsRegistry
├── pipeline/
│   ├── config.py               # MODIFIED: add metrics_config
│   └── runner.py               # MODIFIED: propagate metrics_config
├── controllers/
│   ├── models/
│   │   └── base_model_controller.py  # MODIFIED: use metrics_config
│   └── torch/
│       └── op_model.py         # MODIFIED: use MetricsRegistry
├── dataset/
│   ├── predictions.py          # MODIFIED: use MetricsRegistry
│   └── evaluator.py            # UNCHANGED (used by registry)
├── utils/
│   ├── model_utils.py          # MODIFIED: delegate to MetricsRegistry
│   └── tab_report_manager.py  # MODIFIED: use metrics from config
└── docs/
    ├── METRICS_CONFIGURATION.md  # NEW
    └── WRITING_A_PIPELINE.md   # UPDATED
```

---

**Ready for review and roadmap planning! 🚀**

GBeurier: Ok now create a METRIC_SPECIFICATIONS.MD in docs. I'm ok with all your proposal.
I'm not sure to understand primary metric. Thus In the doc I want a better current state analysis, then an explanation of the choices, then the design itself. Transfer your work to the markdown.

GitHub Copilot: I'll create a comprehensive METRICS_SPECIFICATIONS.md document with enhanced analysis and explanations. Let me first gather more context about the current state, then create the document.

Searched text for `primary_metric|primary metric|best_model|get_best` (`**/*.py`), 20 results

Read [](file:///d%3A/Workspace/ML/NIRS/nirs4all/nirs4all/dataset/predictions.py#1180-1180), lines 1180 to 1230

Read [](file:///d%3A/Workspace/ML/NIRS/nirs4all/nirs4all/utils/tab_report_manager.py#180-180), lines 180 to 230

Read [](file:///d%3A/Workspace/ML/NIRS/nirs4all/nirs4all/utils/model_utils.py#335-335), lines 335 to 365

Read [](file:///d%3A/Workspace/ML/NIRS/nirs4all/nirs4all/dataset/predictions.py#1665-1665), lines 1665 to 1715
