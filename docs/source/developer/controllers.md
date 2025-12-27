# Controller System

The controller system is the core dispatch mechanism in nirs4all. It routes pipeline steps to appropriate handlers based on keywords and operator types.

## Overview

Every pipeline step is processed by a **controller**. The system uses a **priority-based registry** to match steps to controllers:

1. **StepParser** normalizes the step into a `ParsedStep`
2. **ControllerRouter** queries all registered controllers
3. The highest-priority matching controller executes the step

```python
# These all get routed to appropriate controllers
pipeline = [
    MinMaxScaler(),                         # → TransformerMixinController
    {"y_processing": MinMaxScaler()},       # → YProcessingController
    KFold(n_splits=5),                       # → SplitterController
    {"branch": [[A], [B]]},                  # → BranchController
    {"merge": "predictions"},                # → MergeController
    PLSRegression(n_components=10),          # → ModelController
]
```

## Controller Base Class

All controllers inherit from `OperatorController`:

```python
from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
    from nirs4all.pipeline.steps.parser import ParsedStep
    from nirs4all.data.dataset import SpectroDataset

class OperatorController(ABC):
    """Base class for pipeline operators."""

    priority: int = 100  # Lower = higher priority

    @classmethod
    @abstractmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Check if this controller should handle the step."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def use_multi_source(cls) -> bool:
        """Check if controller supports multi-source datasets."""
        return False

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Check if controller should execute during prediction."""
        return False

    @abstractmethod
    def execute(
        self,
        step_info: "ParsedStep",
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple["ExecutionContext", Any]:
        """Execute the step."""
        raise NotImplementedError
```

## Controller Registry

Controllers are registered globally using the `@register_controller` decorator:

```python
from nirs4all.controllers.registry import register_controller, CONTROLLER_REGISTRY

@register_controller
class MyController(OperatorController):
    priority = 50

    @classmethod
    def matches(cls, step, operator, keyword):
        return keyword == "my_keyword"

    # ... rest of implementation
```

The registry automatically sorts controllers by priority and prevents duplicate registration.

## Keyword System

### Keyword Types

The parser recognizes three categories of keywords:

1. **Reserved Keywords** (not treated as operators):
   - `params`, `metadata`, `steps`, `name`, `finetune_params`, `train_params`, `model_params`

2. **Serialization Operators** (checked first):
   - `class`, `function`, `module`, `object`, `pipeline`, `instance`

3. **Workflow Keywords**:
   - **Priority keywords** (checked in order):
     1. `model`
     2. `preprocessing`
     3. `feature_augmentation`
     4. `y_processing`
     5. `sample_augmentation`
   - **Custom keywords**: Any other non-reserved key

### Keyword Prioritization

When multiple potential keywords exist in a step:

```python
# "model" wins (priority keyword)
{"model": SVC(), "my_custom": lambda x: x}

# "my_custom" is used (no priority keyword present)
{"my_custom": lambda x: x, "params": {...}}

# "class" wins (serialization operator)
{"class": "sklearn.svm.SVC", "model": SVC()}
```

## Built-in Controllers

### Transform Controllers

| Controller | Priority | Keywords/Matches |
|------------|----------|------------------|
| `TransformerMixinController` | 100 | sklearn TransformerMixin objects |
| `YProcessingController` | 50 | `y_processing` keyword |
| `FeatureAugmentationController` | 50 | `feature_augmentation` keyword |

### Model Controllers

| Controller | Priority | Keywords/Matches |
|------------|----------|------------------|
| `ModelController` | 100 | sklearn estimators, `model` keyword |
| `TensorFlowModelController` | 30 | TensorFlow/Keras models |
| `PyTorchModelController` | 30 | PyTorch models |
| `JAXModelController` | 30 | JAX/Flax models |
| `MetaModelController` | 20 | `MetaModel` instances |

### Data Controllers

| Controller | Priority | Keywords/Matches |
|------------|----------|------------------|
| `BranchController` | 5 | `branch` keyword |
| `MergeController` | 5 | `merge`, `merge_sources`, `merge_predictions` keywords |
| `SourceBranchController` | 5 | `source_branch` keyword |

### Splitter Controllers

| Controller | Priority | Keywords/Matches |
|------------|----------|------------------|
| `SplitterController` | 100 | sklearn splitters (KFold, etc.) |

## Writing Custom Controllers

### Step-by-Step Guide

1. **Create the Controller Class**

```python
from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.pipeline.execution.result import StepOutput

@register_controller
class SmoothingController(OperatorController):
    """Custom controller for spectral smoothing."""

    priority = 45  # Between data (5) and generic (100)

    @classmethod
    def matches(cls, step, operator, keyword):
        """Match on 'smoothing' or 'smooth' keywords."""
        if isinstance(step, dict):
            return 'smoothing' in step or 'smooth' in step
        return keyword in ['smoothing', 'smooth']

    @classmethod
    def use_multi_source(cls):
        """Support multi-source datasets."""
        return True

    @classmethod
    def supports_prediction_mode(cls):
        """Execute during prediction to transform new data."""
        return True

    def execute(self, step_info, dataset, context, runtime_context,
                source=-1, mode="train", loaded_binaries=None,
                prediction_store=None):
        """Apply smoothing to spectral data."""
        smoother = step_info.operator
        params = step_info.metadata.get('params', {})

        # Get current features
        X = dataset.X

        # Apply smoothing
        if callable(smoother):
            X_smoothed = smoother(X, **params)
        else:
            X_smoothed = smoother.fit_transform(X)

        # Update dataset
        dataset.X = X_smoothed

        # Return updated context and empty artifacts
        return context, StepOutput()
```

2. **Import the Module**

The controller is registered when imported:

```python
# In your script or __init__.py
import my_custom_controllers  # Registers all controllers
```

3. **Use in Pipeline**

```python
from scipy.signal import savgol_filter

pipeline = [
    {"smoothing": savgol_filter, "params": {"window": 5, "polyorder": 2}},
    {"preprocessing": StandardScaler()},
    {"model": PLSRegression()},
]
```

### Important Considerations

#### Keyword Rules

- **Avoid reserved keywords**: `params`, `metadata`, `steps`, `name`, `finetune_params`, `train_params`
- **Don't use serialization keywords**: `class`, `function`, `module`, `object`, `pipeline`, `instance`
- **Priority keywords take precedence**: If your step has `model`, that wins

#### Priority Guidelines

| Priority Range | Use Case |
|----------------|----------|
| 1-10 | Critical operations (branch, merge) |
| 20-50 | Specific operator types (framework-specific models) |
| 50-80 | Custom business logic |
| 80-100 | Generic fallbacks |
| 1000+ | Catch-all (DummyController) |

#### Prediction Mode

If your controller modifies features (like preprocessing), set `supports_prediction_mode() = True` so it executes when loading bundles or running predictions.

### Real-World Examples

#### Example 1: Baseline Correction

```python
@register_controller
class BaselineCorrectionController(OperatorController):
    priority = 40

    @classmethod
    def matches(cls, step, operator, keyword):
        return keyword in ["baseline_correction", "baseline"]

    @classmethod
    def use_multi_source(cls):
        return True

    @classmethod
    def supports_prediction_mode(cls):
        return True

    def execute(self, step_info, dataset, context, runtime_context,
                source=-1, mode="train", loaded_binaries=None,
                prediction_store=None):
        method = step_info.metadata.get('params', {}).get('method', 'als')
        # Apply baseline correction...
        return context, StepOutput()
```

Usage:
```python
pipeline = [
    {"baseline_correction": my_baseline_fn, "params": {"method": "als"}}
]
```

#### Example 2: Outlier Detection

```python
@register_controller
class OutlierDetectionController(OperatorController):
    priority = 35

    @classmethod
    def matches(cls, step, operator, keyword):
        return keyword in ["outlier_detection", "outliers"]

    @classmethod
    def use_multi_source(cls):
        return True

    @classmethod
    def supports_prediction_mode(cls):
        return False  # Only during training

    def execute(self, step_info, dataset, context, runtime_context,
                source=-1, mode="train", loaded_binaries=None,
                prediction_store=None):
        detector = step_info.operator
        # Mark outliers in metadata...
        return context, StepOutput()
```

Usage:
```python
from sklearn.ensemble import IsolationForest

pipeline = [
    {"outlier_detection": IsolationForest(), "params": {"contamination": 0.1}}
]
```

## Controller Methods Reference

### `matches(cls, step, operator, keyword) -> bool`

Determines if this controller should handle the step.

**Parameters:**
- `step`: Original step configuration (dict, string, or object)
- `operator`: Deserialized operator (if any)
- `keyword`: Extracted keyword (e.g., "model", "preprocessing")

**Returns:** `True` if this controller should handle the step

### `use_multi_source(cls) -> bool`

Indicates if the controller supports multi-source datasets.

**Returns:** `True` if the controller can process datasets with multiple feature sources

### `supports_prediction_mode(cls) -> bool`

Indicates if the controller should execute during prediction mode.

**Returns:** `True` if the controller should run when loading bundles or making predictions on new data

### `execute(...) -> Tuple[ExecutionContext, StepOutput]`

Executes the step logic.

**Parameters:**
- `step_info`: Parsed step containing operator, keyword, and metadata
- `dataset`: SpectroDataset to operate on
- `context`: ExecutionContext with pipeline state
- `runtime_context`: RuntimeContext with infrastructure (workspace, logging)
- `source`: Data source index (-1 for all sources)
- `mode`: "train" or "predict"
- `loaded_binaries`: Pre-loaded artifacts for prediction mode
- `prediction_store`: External store for model predictions

**Returns:** Tuple of (updated context, StepOutput with artifacts)

## Testing Custom Controllers

Test your controller with:

```python
def test_my_controller_matches():
    """Verify keyword matching."""
    assert MyController.matches(
        {"my_keyword": lambda x: x},
        None,
        "my_keyword"
    )
    assert not MyController.matches(
        {"other": lambda x: x},
        None,
        "other"
    )

def test_my_controller_registration():
    """Verify registration in registry."""
    from nirs4all.controllers.registry import CONTROLLER_REGISTRY
    assert MyController in CONTROLLER_REGISTRY

def test_my_controller_priority():
    """Verify priority ordering."""
    from nirs4all.controllers.registry import CONTROLLER_REGISTRY
    names = [c.__name__ for c in CONTROLLER_REGISTRY]
    # Verify position relative to other controllers
```

## See Also

- {doc}`architecture` - High-level architecture overview
- {doc}`/reference/pipeline_syntax` - Complete pipeline syntax reference
- {doc}`/user_guide/pipelines/branching` - User-facing branching documentation
- {doc}`/examples/index` - Working examples

**Source files:**
- `nirs4all/controllers/registry.py` - Controller registration
- `nirs4all/controllers/controller.py` - Base controller class
- `nirs4all/controllers/transforms/` - Transform controllers
- `nirs4all/controllers/models/` - Model controllers
