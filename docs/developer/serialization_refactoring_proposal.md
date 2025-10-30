# nirs4all Serialization Refactoring Proposal

**Date**: October 30, 2025
**Version**: 0.4.1
**Status**: Final Architecture - Ready for Implementation

---

## Executive Summary

This document proposes refactoring the serialization/instantiation system to improve **readability**, **maintainability**, **extensibility**, and **logical consistency** while preserving user experience. The refactoring eliminates `_runtime_instance` from serialized configs and establishes **controller-delegated responsibilities** for model lifecycle operations.

**Architectural Decisions**:
- ✅ **Controllers own model lifecycle**: instantiation, cloning, optional save/load
- ✅ **ModelFactory provides helpers**: reusable utilities for common operations
- ✅ **artifact_serialization as fallback**: handles saving/loading when controllers don't
- ✅ **Fully serializable configs**: No live Python objects in JSON/YAML
- ✅ **Clean deprecated code**: Remove obsolete functions entirely

**Goals**:
- ✅ **Extensibility**: Users can register custom controllers with custom model handling
- ✅ **Clear responsibilities**: Controllers decide how their models work
- ✅ **Framework-specific formats**: TF→.h5/.keras, PyTorch→.ckpt, sklearn→.joblib
- ✅ **Backward compatible**: No changes to user-facing API
- ✅ **Clean codebase**: Remove all deprecated functions (no backward compatibility cruft)

**Non-goals**:
- ❌ Changing JSON/YAML format visible to users (both remain supported)
- ❌ Breaking public methods used in examples
- ❌ Centralizing all logic in one place (defeats extensibility)

---

## 1. Core Problems

### 1.1 `_runtime_instance` Creates Non-Serializable Configs

**Current**:
```python
# After serialize_component():
{
    "function": "nirs4all.operators.models.cirad_tf.nicon",
    "_runtime_instance": <function nicon>  # ← Live Python object
}

# json.dumps() fails:
TypeError: Object of type function is not JSON serializable
```

**Impact**:
- Configs cannot be exported to clean JSON files
- `get_hash()` uses `default=str` hack: `<function nicon at 0x...>` → inconsistent hashes
- Tests must special-case `_runtime_instance` extraction

**Proposed Solution**:
- **Remove `_runtime_instance` from serialized output**
- **Re-import functions during deserialization** using string path
- **Controllers receive clean dicts** without live objects

---

### 1.2 Unclear Instantiation Responsibilities

**Current**: Model lifecycle operations scattered across multiple layers:

1. **`deserialize_component()`**: Sometimes instantiates classes directly
2. **`ModelFactory.build_single_model()`**: Instantiates models with introspection
3. **`Controller._get_model_instance()`**: Abstract method, unclear what it should do
4. **Direct calls**: `sklearn.base.clone()`, `joblib.dump()` scattered throughout codebase

**Impact**:
- Signature inspection done multiple times
- Parameter filtering logic duplicated
- Unclear which layer owns instantiation, cloning, saving, loading
- Hard to extend with custom frameworks (where should custom logic go?)

**Proposed Solution**:
- **Controllers own full lifecycle**: instantiate, clone, save, load
- **ModelFactory provides helpers**: filter_params, compute_input_shape, detect_framework
- **`deserialize_component()` only imports**—never instantiates
- **Clear ownership**: Controllers decide everything, use helpers as needed

---

### 1.3 Duplicate Framework Detection

**Current**: Two implementations:

1. **`ModelFactory.detect_framework(model)`** (line 494 in factory.py)
2. **`artifact_serialization._detect_framework(obj)`** (line 65 in artifact_serialization.py)

Both inspect `obj.__module__` for `'sklearn'`, `'tensorflow'`, `'torch'`, etc.

**Impact**:
- Duplicate code maintained in two places
- Inconsistent results if only one is updated

**Proposed Solution**:
- **Consolidate in `artifact_serialization`** (already used for saving)
- **ModelFactory helper delegates**: `ModelFactory.detect_framework(model)` → calls `artifact_serialization.detect_framework(model)`
- **Controllers use helper**: Call `ModelFactory.detect_framework()` when needed

---

### 1.4 Framework-Specific Formats Ignored

**Current**: 15+ files import `joblib` or `pickle` directly. All models saved as `.joblib` regardless of framework.

**Impact**:
- TensorFlow models lose native format benefits (.h5 inspection tools, portability)
- PyTorch models can't use `.ckpt` format (standard in community)
- Inconsistent with framework best practices

**Proposed Solution**:
- **Controllers enforce format**: `TensorFlowModelController.save_model()` uses `.h5`
- **artifact_serialization detects and adapts**: Fallback uses framework-appropriate format
- **Audit direct imports**: Replace `joblib.dump()` calls with `controller.save_model()`

---

## 2. Proposed Architecture

### 2.1 Layer Responsibilities

| Layer | Responsibility | Input | Output | Instantiates? |
|-------|---------------|-------|--------|---------------|
| **Serialize** | Python → JSON dict | Objects | Serializable dicts | **No** |
| **Deserialize** | JSON dict → imports | Dicts | Classes/functions (not instances) | **No** |
| **Controllers** | Model lifecycle | Config + dataset | Model instances | **Yes** |
| **ModelFactory** | Helper utilities | Classes/functions + metadata | Filtered params, shapes | **No** |
| **Artifact** | Trained → Disk (fallback) | Fitted models | Files + metadata | **No** |

**Key Change**: **Controllers own instantiation, cloning, save/load**. ModelFactory provides reusable helpers. artifact_serialization is fallback when controllers don't implement save/load.

---

### 2.2 Serialization Changes

**Remove `_runtime_instance` embedding**:

```python
# BEFORE
def serialize_component(obj: Any) -> Any:
    if inspect.isfunction(obj):
        func_serialized = {"function": f"{obj.__module__}.{obj.__name__}"}
        if hasattr(obj, 'framework'):
            func_serialized["_runtime_instance"] = obj  # ← REMOVE THIS
        return func_serialized

# AFTER
def serialize_component(obj: Any) -> Any:
    if inspect.isfunction(obj):
        func_serialized = {"function": f"{obj.__module__}.{obj.__name__}"}
        if hasattr(obj, 'framework'):
            func_serialized["framework"] = obj.framework  # ← Store framework string
        if params:
            func_serialized["params"] = serialize_component(params)
        return func_serialized
```

**Result**: Fully JSON-serializable output:
```json
{
    "function": "nirs4all.operators.models.cirad_tf.nicon",
    "framework": "tensorflow"
}
```

---

### 2.3 Deserialization Changes

**Return classes/functions, not instances**:

```python
# BEFORE
def deserialize_component(blob: Any, infer_type: Any = None) -> Any:
    if "class" in blob:
        cls = import_class(blob["class"])
        params = {k: deserialize_component(v) for k, v in blob["params"].items()}
        return cls(**params)  # ← Instantiates here

# AFTER
def deserialize_component(blob: Any, infer_type: Any = None) -> Any:
    if "class" in blob:
        cls = import_class(blob["class"])
        params = {k: deserialize_component(v) for k, v in blob.get("params", {}).items()}
        # Return dict for factory to instantiate
        return {
            "type": "class",
            "cls": cls,
            "params": params
        }

    if "function" in blob:
        func = import_function(blob["function"])
        params = {k: deserialize_component(v) for k, v in blob.get("params", {}).items()}
        # Return dict for factory to instantiate
        return {
            "type": "function",
            "func": func,
            "framework": blob.get("framework"),
            "params": params
        }
```

**Alternative (simpler)**: Return raw class/function + separate metadata dict:
```python
def deserialize_component(blob: Any) -> Tuple[Any, Dict[str, Any]]:
    if "class" in blob:
        cls = import_class(blob["class"])
        metadata = {"type": "class", "params": blob.get("params", {})}
        return cls, metadata
```

---

### 2.4 ModelFactory Helper Methods

**ModelFactory becomes a utility class** providing reusable helpers (not instantiation):

```python
class ModelFactory:
    """Helper utilities for model operations (used by controllers)."""

    @staticmethod
    def filter_params(callable_obj, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter parameters to only those accepted by callable's signature.

        Args:
            callable_obj: Class or function to check signature
            params: Dictionary of parameters

        Returns:
            Filtered parameters matching signature
        """
        if inspect.isclass(callable_obj):
            sig = inspect.signature(callable_obj.__init__)
        else:
            sig = inspect.signature(callable_obj)

        return {k: v for k, v in params.items()
                if k in sig.parameters or 'kwargs' in sig.parameters}

    @staticmethod
    def compute_input_shape(dataset, framework: str) -> Union[int, Tuple[int, ...]]:
        """Compute input shape from dataset based on framework.

        Args:
            dataset: Dataset with X array
            framework: Framework name ('sklearn', 'tensorflow', 'pytorch', etc.)

        Returns:
            Input shape appropriate for framework
        """
        if framework in ['sklearn', 'xgboost', 'lightgbm']:
            return dataset.X.shape[1]  # Flat features
        elif framework in ['tensorflow', 'pytorch']:
            # Return tuple shape for neural networks
            return (dataset.X.shape[1],)
        else:
            return dataset.X.shape[1]

    @staticmethod
    def detect_framework(model) -> str:
        """Detect framework from model instance.

        Args:
            model: Model instance

        Returns:
            Framework name string
        """
        module = model.__class__.__module__
        if 'sklearn' in module:
            return 'sklearn'
        elif 'tensorflow' in module or 'keras' in module:
            return 'tensorflow'
        elif 'torch' in module:
            return 'pytorch'
        elif 'xgboost' in module:
            return 'xgboost'
        elif 'lightgbm' in module:
            return 'lightgbm'
        return 'unknown'

    @staticmethod
    def get_num_classes(dataset) -> Optional[int]:
        """Extract number of classes for classification tasks.

        Args:
            dataset: Dataset with y labels

        Returns:
            Number of unique classes or None
        """
        if hasattr(dataset, 'is_classification') and dataset.is_classification:
            import numpy as np
            return len(np.unique(dataset.y))
        return None
```

**Key Point**: These are **helpers** that controllers can use. Controllers decide when and how to use them.

---

### 2.5 Controller Responsibilities (Primary)

**Controllers own the full model lifecycle**:

```python
class BaseModelController(ABC):
    """Base controller with model lifecycle methods."""

    @abstractmethod
    def instantiate_model(self, model_config: Dict, dataset, force_params=None):
        """Create model instance from config.

        Controllers decide how to handle:
        - Classes vs functions
        - Framework-specific parameter injection
        - Special initialization logic

        Args:
            model_config: Deserialized config dict or callable
            dataset: Dataset for shape inference
            force_params: Parameters to override

        Returns:
            Instantiated model
        """
        pass

    @abstractmethod
    def clone_model(self, model):
        """Clone an existing model instance.

        Framework-specific cloning:
        - sklearn: clone(model)
        - TensorFlow: tf.keras.models.clone_model()
        - PyTorch: deepcopy or manual state_dict copy

        Args:
            model: Model to clone

        Returns:
            Cloned model
        """
        pass

    def save_model(self, model, filepath: str) -> None:
        """Optional: Save model in framework-specific format.

        If not implemented, falls back to artifact_serialization.persist().
        Implementations should use:
        - TensorFlow: .h5 or .keras format
        - PyTorch: .ckpt or .pt format
        - sklearn: .joblib format

        Args:
            model: Trained model
            filepath: Path to save (without extension)
        """
        # Default fallback
        from nirs4all.pipeline.artifact_serialization import persist
        persist(model, filepath)

    def load_model(self, filepath: str):
        """Optional: Load model from framework-specific format.

        If not implemented, falls back to artifact_serialization.load().

        Args:
            filepath: Path to load from

        Returns:
            Loaded model
        """
        # Default fallback
        from nirs4all.pipeline.artifact_serialization import load
        return load(filepath)
```

**Example: SklearnModelController**:
```python
class SklearnModelController(BaseModelController):
    def instantiate_model(self, model_config, dataset, force_params=None):
        """Instantiate sklearn model."""
        # Handle already-instantiated
        if not isinstance(model_config, dict) and not inspect.isclass(model_config):
            return model_config

        # Handle dict from deserialization
        if isinstance(model_config, dict) and 'cls' in model_config:
            cls = model_config['cls']
            params = {**model_config.get('params', {}), **(force_params or {})}

            # Use ModelFactory helper to filter params
            filtered = ModelFactory.filter_params(cls, params)
            return cls(**filtered)

        # Handle raw class
        elif inspect.isclass(model_config):
            params = force_params or {}
            filtered = ModelFactory.filter_params(model_config, params)
            return model_config(**filtered)

        else:
            return model_config

    def clone_model(self, model):
        """Clone sklearn model."""
        from sklearn.base import clone
        return clone(model)

    def save_model(self, model, filepath: str):
        """Save as .joblib."""
        import joblib
        if not filepath.endswith('.joblib'):
            filepath += '.joblib'
        joblib.dump(model, filepath)

    def load_model(self, filepath: str):
        """Load from .joblib."""
        import joblib
        return joblib.load(filepath)
```

**Example: TensorFlowModelController**:
```python
class TensorFlowModelController(BaseModelController):
    def instantiate_model(self, model_config, dataset, force_params=None):
        """Instantiate TensorFlow model (classes or factory functions)."""

        # Handle factory functions (decorated with @framework('tensorflow'))
        if isinstance(model_config, dict) and model_config.get('type') == 'function':
            func = model_config['func']
            params = {**model_config.get('params', {}), **(force_params or {})}

            # Compute dataset-dependent parameters using helpers
            input_shape = ModelFactory.compute_input_shape(dataset, 'tensorflow')
            num_classes = ModelFactory.get_num_classes(dataset)

            # Merge and filter
            all_params = {'input_shape': input_shape, 'num_classes': num_classes, **params}
            filtered = ModelFactory.filter_params(func, all_params)
            return func(**filtered)

        # Handle Keras model classes
        elif isinstance(model_config, dict) and 'cls' in model_config:
            cls = model_config['cls']
            params = {**model_config.get('params', {}), **(force_params or {})}
            filtered = ModelFactory.filter_params(cls, params)
            return cls(**filtered)

        # Handle callable functions directly
        elif callable(model_config) and not inspect.isclass(model_config):
            input_shape = ModelFactory.compute_input_shape(dataset, 'tensorflow')
            num_classes = ModelFactory.get_num_classes(dataset)
            all_params = {'input_shape': input_shape, 'num_classes': num_classes, **(force_params or {})}
            filtered = ModelFactory.filter_params(model_config, all_params)
            return model_config(**filtered)

        else:
            return model_config  # Already instantiated

    def clone_model(self, model):
        """Clone Keras model."""
        import tensorflow as tf
        return tf.keras.models.clone_model(model)

    def save_model(self, model, filepath: str):
        """Save as .h5 or .keras."""
        if not filepath.endswith(('.h5', '.keras')):
            filepath += '.h5'
        model.save(filepath)

    def load_model(self, filepath: str):
        """Load from .h5 or .keras."""
        import tensorflow as tf
        return tf.keras.models.load_model(filepath)
```

**Example: Custom Controller (User Extension)**:
```python
class PyTorchLightningController(BaseModelController):
    """Custom controller for PyTorch Lightning models."""

    @staticmethod
    def matches(config: Dict) -> bool:
        """Match if config has 'lightning' key."""
        return config.get('model', {}).get('framework') == 'lightning'

    def instantiate_model(self, model_config, dataset, force_params=None):
        """Instantiate PyTorch Lightning module."""
        if isinstance(model_config, dict) and 'cls' in model_config:
            cls = model_config['cls']
            params = {**model_config.get('params', {}), **(force_params or {})}

            # Lightning-specific: add dataset metadata
            input_dim = ModelFactory.compute_input_shape(dataset, 'pytorch')
            num_classes = ModelFactory.get_num_classes(dataset)
            params['input_dim'] = input_dim
            params['num_classes'] = num_classes

            filtered = ModelFactory.filter_params(cls, params)
            return cls(**filtered)
        else:
            return model_config

    def clone_model(self, model):
        """Clone Lightning module."""
        import copy
        return copy.deepcopy(model)

    def save_model(self, model, filepath: str):
        """Save Lightning checkpoint."""
        import torch
        if not filepath.endswith('.ckpt'):
            filepath += '.ckpt'
        torch.save(model.state_dict(), filepath)

    def load_model(self, filepath: str):
        """Load Lightning checkpoint."""
        import torch
        # Note: Needs model class to instantiate
        # Real implementation would need to store model class info
        raise NotImplementedError("Lightning load requires model class")
```

**Result**:
- ✅ Each controller handles its framework's specifics
- ✅ ModelFactory provides common helpers (not instantiation)
- ✅ Extensible: users register custom controllers with custom logic
- ✅ Clear ownership: controller decides everything about its models
- ✅ Framework-specific formats enforced by controllers

---

### 2.6 Save/Load Fallback Mechanism

**Design**: Controllers can **optionally** implement `save_model()` and `load_model()`. If not implemented, the base class provides fallback to `artifact_serialization`.

**Base Implementation**:
```python
class BaseModelController(ABC):
    def save_model(self, model, filepath: str) -> None:
        """Default fallback to artifact_serialization.persist()."""
        from nirs4all.pipeline.artifact_serialization import persist
        persist(model, filepath)

    def load_model(self, filepath: str):
        """Default fallback to artifact_serialization.load()."""
        from nirs4all.pipeline.artifact_serialization import load
        return load(filepath)
```

**Controller Can Override**:
```python
class TensorFlowModelController(BaseModelController):
    def save_model(self, model, filepath: str) -> None:
        """Override: use TensorFlow native format."""
        if not filepath.endswith(('.h5', '.keras')):
            filepath += '.h5'
        model.save(filepath)  # Native TF save
```

**artifact_serialization Updates**:
```python
# In artifact_serialization.py
def persist(model, filepath: str) -> None:
    """Generic fallback for models without controller support.

    Detects framework and uses appropriate format:
    - TensorFlow → .h5 (via model.save if available)
    - PyTorch → .ckpt (via torch.save if available)
    - sklearn → .joblib
    - Unknown → .pkl (pickle fallback)
    """
    framework = detect_framework(model)

    if framework == 'tensorflow':
        if not filepath.endswith(('.h5', '.keras')):
            filepath += '.h5'
        model.save(filepath)

    elif framework == 'pytorch':
        import torch
        if not filepath.endswith(('.ckpt', '.pt')):
            filepath += '.ckpt'
        torch.save(model.state_dict(), filepath)

    elif framework in ['sklearn', 'xgboost', 'lightgbm']:
        import joblib
        if not filepath.endswith('.joblib'):
            filepath += '.joblib'
        joblib.dump(model, filepath)

    else:
        # Generic fallback
        import pickle
        if not filepath.endswith('.pkl'):
            filepath += '.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
```

**Result**:
- ✅ Controllers can use framework-specific APIs for optimal saving
- ✅ Fallback handles unknown frameworks gracefully
- ✅ Consistent file extensions across codebase
- ✅ artifact_serialization becomes smarter, not dumber

---

## 3. Implementation Plan

### Phase 1: Remove `_runtime_instance` from Serialization (1 day)

**Tasks**:
1. Modify `serialize_component()`:
   - Remove `func_serialized["_runtime_instance"] = obj`
   - Add `func_serialized["framework"] = obj.framework` (store string only)

2. Modify `deserialize_component()`:
   - Import function and return dict with `{"type": "function", "func": ..., "framework": ...}`
   - Never instantiate—controllers will handle that

3. Update `PipelineConfigs.get_hash()`:
   - Remove `default=str` hack (no longer needed with fully serializable dicts)

4. Update tests:
   - Remove `_runtime_instance` extraction logic
   - Verify JSON serialization produces valid JSON (no repr strings)

**Validation**: Run `pytest tests/unit/pipeline/test_serialization.py`

---

### Phase 2: Refactor ModelFactory to Helpers (1 day)

**Tasks**:
1. Remove `ModelFactory.build_single_model()`:
   - Delete this method entirely (controllers will instantiate)
   - Remove `_instantiate_class()` and `_instantiate_function()` private methods

2. Add helper methods:
   - `filter_params(callable_obj, params)` - parameter filtering with signature inspection
   - `compute_input_shape(dataset, framework)` - shape computation logic
   - `detect_framework(model)` - consolidate with artifact_serialization
   - `get_num_classes(dataset)` - extract class count

3. Update imports in controllers:
   - Controllers will use `ModelFactory.filter_params()` etc as needed
   - No more calls to `ModelFactory.build_single_model()`

**Validation**: Run `pytest tests/unit/controllers/models/test_factory.py` (update tests)

---

### Phase 3: Implement Controller Lifecycle Methods (2 days)

**Tasks**:
1. Update `BaseModelController`:
   - Add abstract `instantiate_model(model_config, dataset, force_params)` method
   - Add abstract `clone_model(model)` method
   - Add optional `save_model(model, filepath)` with fallback to `artifact_serialization.persist()`
   - Add optional `load_model(filepath)` with fallback to `artifact_serialization.load()`
   - Remove deprecated `_get_model_instance()` method

2. Implement in `SklearnModelController`:
   - `instantiate_model()`: Handle classes from deserialized dicts, use `ModelFactory.filter_params()`
   - `clone_model()`: Use `sklearn.base.clone()`
   - `save_model()`: Save as `.joblib`
   - `load_model()`: Load from `.joblib`

3. Implement in `TensorFlowModelController`:
   - `instantiate_model()`: Handle factory functions (with `input_shape`, `num_classes`), use helpers
   - `clone_model()`: Use `tf.keras.models.clone_model()`
   - `save_model()`: Save as `.h5` or `.keras`
   - `load_model()`: Load with `tf.keras.models.load_model()`
   - **Delete** `_extract_model_config()` method entirely

4. Implement in `PyTorchModelController`:
   - `instantiate_model()`: Handle PyTorch modules
   - `clone_model()`: Use `deepcopy`
   - `save_model()`: Save as `.ckpt` or `.pt` with `torch.save()`
   - `load_model()`: Load with `torch.load()`

**Validation**: Run `pytest tests/unit/controllers/` and examples `Q1_*.py`

---

### Phase 4: Update Callers to Use New Controller Methods (1 day)

**Tasks**:
1. Find all calls to old methods:
   - Search for `_get_model_instance()` calls → replace with `instantiate_model()`
   - Search for `sklearn.base.clone()` direct calls → replace with `controller.clone_model()`
   - Search for `joblib.dump()` direct calls → replace with `controller.save_model()`

2. Update pipeline execution code:
   - Use `controller.instantiate_model()` instead of factory
   - Use `controller.clone_model()` for CV folds
   - Use `controller.save_model()` for persistence

**Validation**: Run full test suite `pytest tests/`

---

### Phase 5: Remove Deprecated Functions (1 hour)

**Tasks**:
1. Delete deprecated methods:
   - `BaseModelController._get_model_instance()` → delete
   - `TensorFlowModelController._extract_model_config()` → delete
   - `ModelFactory.build_single_model()` → delete (if no other callers)

2. Search codebase for:
   - `grep -r "_get_model_instance" nirs4all/`
   - `grep -r "_extract_model_config" nirs4all/`
   - `grep -r "build_single_model" nirs4all/`

3. Remove without backward compatibility:
   - No migration scripts
   - No graceful degradation
   - Clean removal per user requirements

**Validation**: Ensure no references remain, run `pytest tests/`

---

### Phase 6: Framework-Specific Format Enforcement (2 days)

**Tasks**:
1. Audit all 15+ files with direct `joblib`/`pickle` imports:
   - Replace with controller save/load methods where applicable
   - Keep `artifact_serialization` as fallback for generic objects

2. Update `artifact_serialization.persist()`:
   - Detect framework and use appropriate format
   - TensorFlow → `.h5`
   - PyTorch → `.ckpt`
   - sklearn → `.joblib`

3. Add file extension validation:
   - Controllers check/add extensions in `save_model()`
   - Raise clear error if wrong format detected

**Validation**: Run examples and verify output formats, check `exports/` directory

---

### Phase 7: Documentation & Testing (1 day)

**Tasks**:
1. Update docstrings:
   - `serialize_component()`: Document JSON-serializable output
   - `deserialize_component()`: Document returns classes/functions, not instances
   - `BaseModelController`: Document lifecycle method responsibilities
   - `ModelFactory`: Document as helper utility class

2. Add integration test:
   ```python
   def test_controller_lifecycle():
       """Test full controller lifecycle: instantiate → train → clone → save → load."""
       controller = SklearnModelController()

       # Instantiate
       config = {'cls': RandomForestClassifier, 'params': {'n_estimators': 10}}
       model = controller.instantiate_model(config, dataset)

       # Clone
       model_clone = controller.clone_model(model)

       # Save
       controller.save_model(model, 'test_model')

       # Load
       loaded = controller.load_model('test_model.joblib')
   ```

3. Update user documentation:
   - Explain how to create custom controllers with custom lifecycle logic
   - Document ModelFactory helpers available for reuse

**Validation**: Run `pytest tests/` and `.\run.ps1 -l` in examples/
       serialized = serialize_component(pipeline)
       json_str = json.dumps(serialized)  # Should not fail

       # Deserialize
       loaded = json.loads(json_str)
       deserialized = deserialize_component(loaded)

       # Instantiate via factory
       dataset = create_test_dataset()
       model = ModelFactory.build_single_model(deserialized[1], dataset)

       # Verify it's a TensorFlow model
       assert isinstance(model, tf.keras.Model)
   ```

3. Update user documentation:
   - `WRITING_A_PIPELINE.md`: Explain that functions are serialized as string paths
   - `SERIALIZATION.md`: Document new architecture

---

## 4. Edge Cases & Considerations

### 4.1 User-Provided Instances

**Current**: User provides `MinMaxScaler(feature_range=(0, 2))`

**Flow**:
```
Instance → serialize_component() → {"class": "...", "params": {"feature_range": [0, 2]}}
→ deserialize_component() → {"type": "class", "cls": MinMaxScaler, "params": {...}}
→ ModelFactory → MinMaxScaler(feature_range=(0, 2))  # Fresh instance
```

**Consideration**: Should we preserve the original instance instead of recreating?

**Answer**: No—cross-validation requires fresh clones anyway.

---

### 4.2 Function Signature Variations

**Case 1**: Simple function
```python
@framework('tensorflow')
def simple_model(input_shape):
    return Sequential([Dense(10, input_shape=input_shape)])
```

**Case 2**: Function with params bundle
```python
@framework('tensorflow')
def configurable_model(input_shape, params={}):
    layers = params.get('layers', [64, 32])
    return Sequential([Dense(n, input_shape=input_shape) for n in layers])
```

**Case 3**: Function with explicit params
```python
@framework('tensorflow')
def explicit_model(input_shape, num_classes, dropout=0.5):
    return Sequential([
        Dense(64, input_shape=input_shape),
        Dropout(dropout),
        Dense(num_classes)
    ])
```

**Solution**: `ModelFactory._instantiate_function()` inspects signature and adapts:
- If `params` in signature: bundle extras
- If explicit params: pass directly
- If only `input_shape`: call with just that

---

### 4.3 Backward Compatibility

**Old pipelines with `_runtime_instance` in JSON**:

**Decision**: **No backward compatibility** - remove deprecated functions entirely per user requirements.

```python
# In deserialize_component() - NO graceful degradation
def deserialize_component(blob: Any, infer_type: Any = None) -> Any:
    if "_runtime_instance" in blob:
        # Do NOT handle it - user must regenerate config
        raise ValueError(
            "Found deprecated _runtime_instance in config. "
            "This format is no longer supported. "
            "Please re-run your pipeline to generate a clean config."
        )
    # ... rest of deserialization
```

**Impact**:
- ✅ Clean codebase - no dead code
- ✅ Clear error messages direct users to solution
- ❌ Old saved configs will break - users must regenerate

**Migration**: Document in CHANGELOG.md with clear instructions to re-run pipelines.

---

### 4.4 Hash Stability

**Current**: `get_hash()` uses `default=str` which converts functions to `<function...>` → unstable hashes

**After refactoring**:
```python
def get_hash(steps) -> str:
    # No need for default=str—all objects are JSON-serializable
    serializable = json.dumps(steps, sort_keys=True).encode('utf-8')
    return hashlib.md5(serializable).hexdigest()[0:8]
```

**Impact**: Hashes will **change** for pipelines with model functions.

**Mitigation**: Document in changelog as **expected breaking change**—users must re-run pipelines.

---

### 4.4 Hash Stability

**Current**: `get_hash()` uses `default=str` which converts functions to `<function...>` → unstable hashes

**After refactoring**:
```python
def get_hash(steps) -> str:
    # No need for default=str—all objects are JSON-serializable
    serializable = json.dumps(steps, sort_keys=True).encode('utf-8')
    return hashlib.md5(serializable).hexdigest()[0:8]
```

**Impact**: Hashes will **change** for pipelines with model functions.

**Mitigation**: Document in changelog as **expected breaking change**—users must re-run pipelines.

---

## 5. Optimization Opportunities

### 5.1 Cache Signature Introspection

**Current**: `_filter_params()` inspects signature every time a model is instantiated (every fold).

**Proposed**:
```python
class ModelFactory:
    _signature_cache = {}  # Class-level cache

    @staticmethod
    def _get_signature(cls_or_func):
        key = id(cls_or_func)
        if key not in ModelFactory._signature_cache:
            ModelFactory._signature_cache[key] = inspect.signature(cls_or_func)
        return ModelFactory._signature_cache[key]
```

**Impact**: Reduce overhead for pipelines with many folds.

---

### 5.2 Lazy Import for Framework Detection

**Current**: `_check_framework()` uses `importlib.util.find_spec()` (cached).

**Already optimal**—no change needed.

---

### 5.3 Deduplicate Parameter Filtering

**Current**: Parameter filtering logic in 3 places:
1. `_changed_kwargs()` in `serialization.py`
2. `_filter_params()` in `factory.py`
3. `prepare_and_call()` in `factory.py`

**Proposed**: Single method in factory:
```python
@staticmethod
def _filter_to_signature(callable_obj, params):
    """Filter params to only those accepted by callable."""
    sig = ModelFactory._get_signature(callable_obj)

    # Check for **kwargs
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values())

    if has_kwargs:
        return params  # Accept all

    valid_params = {name for name in sig.parameters if name != 'self'}
    return {k: v for k, v in params.items() if k in valid_params}
```

Use this in both `_instantiate_class()` and `_instantiate_function()`.

---

## 6. Centralized vs Delegated Model Lifecycle: Architectural Analysis

### 6.1 The Core Question

**Where should model creation, cloning, saving, and loading logic reside?**

Two viable approaches exist, each with distinct trade-offs:

**Approach A: Centralized in ModelFactory**
- All model operations go through `ModelFactory`
- Controllers call factory methods
- Single source of truth

**Approach B: Delegated to Controllers**
- Each controller implements its own model operations
- `ModelFactory` provides utilities/helpers
- Framework-specific logic in controllers

---

### 6.2 Comparison Matrix

| Aspect | Centralized (ModelFactory) | Delegated (Controllers) |
|--------|---------------------------|-------------------------|
| **Maintainability** | ✅ Single place to update | ❌ Changes in N controllers |
| **Extensibility** | ❌ Users must modify factory | ✅ Users register custom controllers |
| **Framework-specific logic** | ⚠️ Giant if/else tree | ✅ Isolated per controller |
| **Code duplication** | ✅ Minimal | ⚠️ Some duplication acceptable |
| **Testing** | ✅ Test one place | ⚠️ Test each controller |
| **Discovery** | ❌ Implicit (factory decides) | ✅ Explicit (controller.matches()) |
| **Custom models** | ❌ Modify core code | ✅ Add new controller |
| **Consistency** | ✅ Enforced by factory | ⚠️ Each controller can differ |

---

### 6.3 Current Hybrid State (Problem)

**Problem**: We have a confusing hybrid:
- `ModelFactory` handles instantiation
- Controllers handle cloning (`_clone_model()` abstract method)
- `artifact_serialization` handles saving
- `ModelFactory._load_model_from_file()` handles loading (but controllers don't use it)

**Result**: Unclear who is responsible for what.

---

### 6.4 Detailed Analysis by Operation

#### 6.4.1 Model Creation (Instantiation)

**Centralized Approach**:
```python
# ModelFactory
class ModelFactory:
    @staticmethod
    def create_model(config, dataset, framework=None):
        if framework == 'sklearn':
            return ModelFactory._create_sklearn(config, dataset)
        elif framework == 'tensorflow':
            return ModelFactory._create_tensorflow(config, dataset)
        # ... more frameworks
```

**Pros**:
- ✅ Single method to call from anywhere
- ✅ Consistent parameter handling
- ✅ Easy to add framework-wide features (e.g., caching)

**Cons**:
- ❌ ModelFactory becomes huge with all framework logic
- ❌ Users can't add custom frameworks without modifying core
- ❌ Tight coupling between factory and all frameworks

---

**Delegated Approach**:
```python
# In each controller
class TensorFlowModelController(BaseModelController):
    def _get_model_instance(self, dataset, model_config, force_params):
        # TensorFlow-specific instantiation logic here
        # Can use ModelFactory helpers if needed
        return self._instantiate_tensorflow_model(model_config, dataset)
```

**Pros**:
- ✅ Framework logic stays in framework controller
- ✅ Users can register custom controllers with custom instantiation
- ✅ No giant factory class with all framework code
- ✅ Controllers discovered via `matches()` pattern

**Cons**:
- ❌ Potential duplication across controllers
- ❌ Harder to enforce consistency
- ❌ Each controller must implement correctly

---

#### 6.4.2 Model Cloning

**Centralized Approach**:
```python
# ModelFactory
@staticmethod
def clone_model(model):
    framework = detect_framework(model)
    if framework == 'sklearn':
        from sklearn.base import clone
        return clone(model)
    elif framework == 'tensorflow':
        from tensorflow.keras.models import clone_model
        return clone_model(model)
    # ...
```

**Pros**:
- ✅ Consistent cloning logic
- ✅ Framework detection automatic
- ✅ Easy to test centrally

**Cons**:
- ❌ Generic cloning may not handle edge cases
- ❌ TensorFlow model functions need special handling (don't clone, return as-is)
- ❌ Custom models with special cloning requirements need factory modification

---

**Delegated Approach** (current):
```python
# In each controller
@abstractmethod
def _clone_model(self, model: Any) -> Any:
    """Clone using framework-specific method."""
    pass

# TensorFlowModelController
def _clone_model(self, model):
    if callable(model) and hasattr(model, 'framework'):
        return model  # Don't clone functions
    return keras.models.clone_model(model)
```

**Pros**:
- ✅ Controllers know framework-specific quirks (e.g., don't clone functions)
- ✅ Custom controllers can implement custom cloning
- ✅ Framework logic stays isolated

**Cons**:
- ❌ Each controller must implement (risk of inconsistency)
- ❌ Potential duplication of framework detection

---

#### 6.4.3 Model Saving/Loading

**Current State**: Handled by `artifact_serialization.persist()` and `load()` (centralized).

**Analysis**: This is **correctly centralized** because:
- ✅ Saving uses content-addressed storage (framework-agnostic wrapper)
- ✅ Framework-specific formats inside (keras, joblib, torch.save)
- ✅ Deduplication requires central coordination
- ✅ No reason for controllers to have different saving logic

**Comparison to Transformers/Splitters**:

| Component Type | Saving Complexity | Custom Logic? | Recommendation |
|----------------|-------------------|---------------|----------------|
| **Models** | High (framework-specific) | Yes (TF, PyTorch, sklearn) | ✅ Centralized in `artifact_serialization` |
| **Transformers** | Low (sklearn) | No (all sklearn-compatible) | ✅ Centralized in `artifact_serialization` |
| **Splitters** | None (stateless) | N/A (not saved) | N/A |
| **Custom objects** | Unknown | Potentially | ✅ Centralized with fallback to pickle |

**Conclusion**: Saving/loading should remain centralized in `artifact_serialization`.

---

### 6.5 Hybrid Recommendation: Pragmatic Split

**Proposal**: Split responsibilities pragmatically based on extensibility needs:

| Operation | Location | Reason |
|-----------|----------|--------|
| **Instantiation (creation)** | **Delegated to controllers** | Users need to customize for custom frameworks |
| **Cloning** | **Delegated to controllers** | Framework-specific quirks (e.g., TF model functions) |
| **Saving** | **Centralized in artifact_serialization** | Content-addressed storage requires coordination |
| **Loading** | **Centralized in artifact_serialization** | Same as saving |
| **Framework detection** | **Centralized in artifact_serialization** | Reusable utility |
| **Parameter filtering** | **Centralized in ModelFactory helpers** | Common logic, but controllers call it |

---

### 6.6 Detailed Recommendation

#### Controllers Should:
```python
class BaseModelController(OperatorController, ABC):
    @abstractmethod
    def _get_model_instance(self, dataset, model_config, force_params):
        """Create model instance (framework-specific)."""
        pass

    @abstractmethod
    def _clone_model(self, model):
        """Clone model (framework-specific)."""
        pass
```

**Reason**: Users registering custom controllers need full control over instantiation and cloning.

#### ModelFactory Should Provide Helpers:
```python
class ModelFactory:
    @staticmethod
    def filter_params(cls_or_func, params):
        """Helper: filter params to signature (reusable by controllers)."""
        sig = inspect.signature(cls_or_func)
        # ... filtering logic

    @staticmethod
    def compute_input_shape(dataset, framework):
        """Helper: compute input shape from dataset (reusable)."""
        # ... shape computation

    @staticmethod
    def build_single_model(model_config, dataset, force_params):
        """Convenience method for standard cases (used by base controllers)."""
        # Handles common patterns: string paths, dicts with 'class'/'function'
        # But controllers can override _get_model_instance() for custom logic
```

**Reason**: Provide utilities without forcing all logic through factory.

#### artifact_serialization Remains Central:
```python
def persist(obj, artifacts_dir, name, format_hint):
    """Save any object with framework detection."""
    # Centralized because content-addressed storage requires coordination

def load(artifact_meta, results_dir):
    """Load any object from artifact metadata."""
    # Centralized for consistency
```

**Reason**: Storage coordination and deduplication require centralization.

---

### 6.7 Implementation for Custom Controllers

**Example: User adds PyTorch Lightning support**

```python
from nirs4all.controllers.models.base_model import BaseModelController
from nirs4all.controllers.registry import register_controller

@register_controller
class PyTorchLightningController(BaseModelController):
    priority = 20

    @classmethod
    def matches(cls, step, operator, keyword):
        # Check if it's a Lightning model
        return isinstance(operator, pl.LightningModule)

    def _get_model_instance(self, dataset, model_config, force_params):
        # Custom instantiation for Lightning
        if callable(model_config):
            # Use ModelFactory helpers
            input_shape = ModelFactory.compute_input_shape(dataset, 'pytorch')
            params = ModelFactory.filter_params(model_config, force_params)
            return model_config(input_shape=input_shape, **params)
        # ... handle other cases

    def _clone_model(self, model):
        # Lightning-specific cloning
        config = model.hparams
        return model.__class__(**config)

    def _train_model(self, model, X_train, y_train, **kwargs):
        # Lightning-specific training
        trainer = pl.Trainer(**kwargs)
        trainer.fit(model, train_loader)
        return model
```

**Key Point**: User **never modifies core code**—just registers a new controller.

**If instantiation were centralized in ModelFactory**:
- ❌ User must modify `ModelFactory.build_single_model()`
- ❌ Core code bloats with framework-specific if/else
- ❌ Harder to maintain separation of concerns

---

### 6.8 What About Transformers and Other Components?

**Analysis**:

| Component | Instantiation | Cloning | Saving | Complexity |
|-----------|--------------|---------|--------|------------|
| **Models** | Controller | Controller | Central | High (framework-specific) |
| **Transformers** | Deserialize* | sklearn.base.clone | Central | Low (all sklearn) |
| **Splitters** | Deserialize* | sklearn.base.clone | N/A | Low (stateless) |
| **Custom operators** | Controller | Controller | Central | Varies |

*Transformers and splitters can be instantiated during deserialization because:
- They're always sklearn-compatible (no dataset dependency)
- No framework-specific quirks
- Parameters known at config time (no `input_shape`)

**Models are different** because:
- ❌ Framework-specific (sklearn, TF, PyTorch, custom)
- ❌ Dataset-dependent (`input_shape`, `num_classes`)
- ❌ Need deferred instantiation
- ✅ Controllers provide the right abstraction layer

---

### 6.9 Final Recommendation

**Adopt Delegated Approach for Models, Centralized for Artifacts**:

1. **Controllers own instantiation and cloning**:
   - `_get_model_instance()` - creates models
   - `_clone_model()` - clones models
   - Enables custom controller registration

2. **ModelFactory provides reusable helpers**:
   - `filter_params()` - parameter filtering
   - `compute_input_shape()` - shape computation
   - `build_single_model()` - convenience for standard cases
   - Controllers can use or override

3. **artifact_serialization owns saving/loading**:
   - `persist()` - framework-aware saving
   - `load()` - framework-aware loading
   - Content-addressed storage

4. **deserialize_component() remains simple**:
   - Only imports and validates
   - Never instantiates models (delegates to controllers)
   - Can instantiate simple sklearn transformers/splitters

**Rationale**: Balances maintainability (centralized utilities) with extensibility (controller delegation).

---

## 7. Questions for Author (Resolved)

### Q1: Should we preserve user instances?

**Scenario**: User provides `scaler = MinMaxScaler(feature_range=(0,2))` and later modifies `scaler.feature_range = (0, 3)`.

**Current behavior**: Extract params at serialization time → recreate later → user modification lost.

**Alternative**: Store reference to user instance, use it directly (no cloning).

**Trade-off**: Convenience vs. purity (cross-validation expects fresh copies).

**✅ DECISION**: Keep current behavior (extract params, recreate). It's easier to manage saving/loading, and cross-validation requires fresh copies anyway.

---

### Q2: ModelFactory location

**Current**: `controllers/models/factory.py` (relocated from `utils/model_builder.py`)

**Consideration**: Should it move back to `utils/`? It's used by all controllers, not just models.

**✅ DECISION**: Keep in `controllers/models/`—responsibilities are controller-specific. The factory provides helpers that controllers use, so co-location makes sense for understanding the architecture.

---

### Q3: Framework decorator enforcement

**Scenario**: User forgets `@framework` decorator on model function.

**Current behavior**: `framework = None` → `ValueError` in `ModelFactory._from_callable()`.

**Alternative**: Try to infer framework from function body (inspect for `tf.keras`, `torch.nn`, etc.).

**✅ DECISION**: Keep strict enforcement but improve error message. Better to be explicit than implicit. If users extend nirs4all with custom controllers, they may need custom instantiation logic, so the decorator helps route to the right controller.

**Improved error message**:
```python
if framework is None:
    raise ValueError(
        f"Model function '{func.__name__}' must be decorated with @framework('tensorflow'|'pytorch'). "
        f"Example: @framework('tensorflow')\ndef {func.__name__}(input_shape, params): ..."
    )
```

---

### Q4: Should deserialization return tuples?

**Option A**: Return dict with type info
```python
{"type": "class", "cls": StandardScaler, "params": {}}
```

**Option B**: Return tuple
```python
(StandardScaler, {"type": "class", "params": {}})
```

**Option C**: Return class/function directly, caller checks type
```python
StandardScaler  # Factory detects it's a class
```

**✅ DECISION**: Move most instantiation to controllers. `deserialize_component()` can return imported classes/functions (not instances), and controllers handle instantiation via `_get_model_instance()`. For simple sklearn transformers/splitters, deserialization can instantiate directly since they have no dataset dependencies.

---

### Q5: Should model instantiation live in one place? (NEW - Core Architecture Question)

**Current**: `deserialize_component()` instantiates some, `ModelFactory` instantiates others.

**Option A - Centralized**: All model creation in `ModelFactory.build_single_model()`
- ✅ Pros: Single place to maintain, consistent behavior
- ❌ Cons: Users extending nirs4all must modify core factory code

**Option B - Delegated**: Controllers own instantiation via `_get_model_instance()`
- ✅ Pros: Users can register custom controllers with custom instantiation
- ❌ Cons: Potential duplication, harder to enforce consistency

**✅ DECISION**: **Delegated approach** (Option B). While centralization helps maintainability, **extensibility is more important** for a library that users extend with custom controllers. Users should be able to register custom controllers without modifying core code. `ModelFactory` provides helpers (`filter_params()`, `compute_input_shape()`) that controllers can use, achieving some centralization benefits without forcing all logic through the factory.

**Rationale**: Custom controllers (e.g., PyTorch Lightning, JAX, custom hardware) may need specialized instantiation. Controllers discovered via `matches()` pattern provide the right abstraction—each controller knows how to create its models.

See Section 6 above for detailed analysis of centralized vs delegated approaches.

---

## 8. Testing Strategy

### 8.1 Unit Tests

```python
# Test serialize/deserialize round-trip
def test_serialize_deserialize_function():
    from nirs4all.operators.models.cirad_tf import nicon

    serialized = serialize_component(nicon)
    assert "function" in serialized
    assert "framework" in serialized
    assert "_runtime_instance" not in serialized

    # Should be JSON-serializable
    json_str = json.dumps(serialized)
    loaded = json.loads(json_str)

    deserialized = deserialize_component(loaded)
    assert deserialized["type"] == "function"
    assert deserialized["framework"] == "tensorflow"

# Test factory instantiation
def test_factory_instantiate_function():
    from nirs4all.operators.models.cirad_tf import nicon

    config = {
        "type": "function",
        "func": nicon,
        "framework": "tensorflow",
        "params": {}
    }

    dataset = create_test_dataset(is_classification=True, num_classes=3)
    model = ModelFactory.build_single_model(config, dataset)

    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, dataset.x('train').shape[1], ...)
```

---

### 8.2 Integration Tests

```python
def test_full_pipeline_with_function():
    """Test complete pipeline with model factory function."""
    from nirs4all.operators.models.cirad_tf import nicon

    pipeline = [
        StandardScaler(),
        {"model": nicon}
    ]

    dataset = load_test_dataset()
    runner = PipelineRunner()
    predictions = runner.train(pipeline, dataset)

    # Verify predictions exist
    assert len(predictions) > 0

    # Verify config is JSON-serializable
    config_path = runner.saver.base_path / "pipeline.json"
    with open(config_path) as f:
        config = json.load(f)  # Should not fail
```

---

### 8.3 Performance Tests

```python
def test_signature_caching_performance():
    """Verify signature caching reduces overhead."""
    from sklearn.ensemble import RandomForestRegressor

    # Cold run
    start = time.time()
    for _ in range(1000):
        sig = inspect.signature(RandomForestRegressor.__init__)
    cold_time = time.time() - start

    # Warm run with cache
    start = time.time()
    for _ in range(1000):
        sig = ModelFactory._get_signature(RandomForestRegressor)
    warm_time = time.time() - start

    assert warm_time < cold_time * 0.1  # 10x faster
```

---

## 9. Migration Guide for Users

### 9.1 No Changes Needed for Most Users

**Existing code continues to work**:
```python
# All these still work identically
pipeline = [
    StandardScaler(),                    # Instance
    PlsRegression(n_components=10),     # Instance
    {"model": "sklearn.svm.SVR"},       # String
    {"model": nicon}                     # Function
]
```

---

### 9.2 Breaking Changes (Internal Only)

**If you were directly accessing `_runtime_instance`** (unlikely):
```python
# OLD (internal code)
if "_runtime_instance" in model_dict:
    func = model_dict["_runtime_instance"]

# NEW
# No longer exists—use factory to instantiate
```

**If you were serializing configs to JSON manually**:
```python
# OLD
config = serialize_component(pipeline)
json.dumps(config, default=str)  # Required default=str hack

# NEW
config = serialize_component(pipeline)
json.dumps(config)  # Works directly
```

---

### 9.3 Hash Changes

**Pipeline UIDs will change** for pipelines with model functions.

**Action required**: Re-run training to generate new pipeline files.

**Example**:
```
Old UID: pipeline_a3f2e1
New UID: pipeline_b4e3d2  # Different hash after removing _runtime_instance
```

---

## 10. Summary & Recommendation

### 10.1 Key Improvements

| Aspect | Before | After |
|--------|--------|-------|
| **JSON serialization** | Broken (needs `default=str`) | ✅ Works natively |
| **Hash stability** | Unstable (function repr) | ✅ Stable (string paths) |
| **Instantiation responsibility** | Split (deserialize + factory) | ✅ Clear (controllers own it) |
| **Extensibility** | Modify core factory | ✅ Register custom controllers |
| **Framework detection** | 2 implementations | ✅ 1 implementation |
| **Code clarity** | `_runtime_instance` confusion | ✅ Clear layer boundaries |
| **Performance** | Redundant introspection | ✅ Cached signatures |
| **Custom frameworks** | Fork core code | ✅ Add controller + register |

---

### 10.2 Implementation Effort

**Total**: ~5 days of development + testing

| Phase | Effort | Risk |
|-------|--------|------|
| Remove `_runtime_instance` | 1 day | Low |
| Consolidate factory | 2 days | Medium |
| Remove controller special cases | 1 day | Low |
| Consolidate framework detection | 1 hour | Low |
| Documentation & testing | 1 day | Low |

---

### 10.3 Final Architectural Decision

**Architecture**: **Controller-Delegated Model Lifecycle** (Decision made: Oct 30, 2025)

**Responsibilities**:
1. **Controllers own model lifecycle** (Primary):
   - `instantiate_model()` - Create instances from configs
   - `clone_model()` - Clone existing instances
   - `save_model()` - Optional: Framework-specific format (fallback to artifact_serialization)
   - `load_model()` - Optional: Framework-specific loading (fallback to artifact_serialization)

2. **ModelFactory provides helpers** (Secondary):
   - `filter_params(callable, params)` - Signature-based filtering
   - `compute_input_shape(dataset, framework)` - Shape computation
   - `detect_framework(model)` - Framework detection (delegates to artifact_serialization)
   - `get_num_classes(dataset)` - Class count extraction

3. **artifact_serialization as fallback**:
   - `persist(model, filepath)` - Generic saving with framework detection
   - `load(filepath)` - Generic loading
   - Used when controller doesn't override save/load

**Key Decisions**:
- ✅ **No backward compatibility**: Remove `_runtime_instance` and deprecated functions entirely
- ✅ **Framework-specific formats**: TensorFlow→.h5/.keras, PyTorch→.ckpt, sklearn→.joblib
- ✅ **Extensibility priority**: Users can register custom controllers with custom logic
- ✅ **Clean codebase**: Delete deprecated methods without graceful degradation

**Rationale**:
1. **Extensibility**: Custom controllers can have completely custom model handling (e.g., PyTorch Lightning with special initialization)
2. **Clear ownership**: Each controller decides how its models work (no ambiguity)
3. **Maintainability**: Helpers reduce duplication without forcing centralization
4. **Correctness**: Fully JSON-serializable configs enable validation and hashing
5. **Framework best practices**: Each framework uses its native format

**Breaking changes are acceptable** because:
- User-facing API unchanged (examples still work)
- Internal architecture only (no public method changes)
- Hash changes expected (documented in CHANGELOG)
- Clean code > backward compatibility with cruft

---

**Document prepared by**: GitHub Copilot
**Version**: 0.4.1 (Final Architecture)
**Review completed**: Oct 30, 2025
**Architectural decision**: Controller-delegated lifecycle with ModelFactory helpers
**Related documents**: [`SERIALIZATION_ANALYSIS.md`](./SERIALIZATION_ANALYSIS.md)
**Status**: ✅ Ready for implementation