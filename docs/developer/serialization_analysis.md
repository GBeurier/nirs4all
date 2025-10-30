# nirs4all Serialization & Instantiation Analysis

**Date**: October 30, 2025
**Version**: 0.4.1
**Status**: Technical Survey

---

## Executive Summary

This document analyzes the serialization, deserialization, and instantiation mechanisms across nirs4all pipelines. The analysis reveals **3 distinct serialization layers** with overlapping responsibilities and **inconsistent handling of `_runtime_instance`** for model factory functions.

**Key Findings**:
- **Pipeline config serialization** (user→JSON/YAML) handles `_runtime_instance` for `@framework` functions
- **Artifact serialization** (trained models→disk) uses framework-aware serialization
- **Model factory** provides a third instantiation path with parameter introspection
- `_runtime_instance` serves a legitimate purpose but creates non-serializable JSON
- Responsibilities overlap between `serialization.py`, `factory.py`, and TensorFlow controller

---

## 1. Serialization Layers

### 1.1 Pipeline Configuration Serialization (`pipeline/serialization.py`)

**Purpose**: Convert user-provided Python objects to JSON/YAML-serializable dictionaries.

**Entry**: `serialize_component(obj: Any) -> Any`

**Flow**:
```
User Input: PlsRegression(n_components=15)
    ↓
serialize_component()
    ↓
Introspect __init__ signature → extract non-default params
    ↓
Output: {"class": "sklearn.cross_decomposition._pls.PLSRegression", "params": {"n_components": 15}}
```

**Key Logic**:
```python
def serialize_component(obj: Any) -> Any:
    # Trivial types pass through
    if obj is None or isinstance(obj, (bool, int, float)): return obj

    # Normalize string paths to internal module paths
    if isinstance(obj, str):
        # "sklearn.preprocessing.StandardScaler" → "sklearn.preprocessing._data.StandardScaler"
        return f"{cls.__module__}.{cls.__qualname__}"

    # Recursively serialize containers
    if isinstance(obj, (dict, list, tuple)): # ...

    # Classes → string path
    if inspect.isclass(obj):
        return f"{obj.__module__}.{obj.__qualname__}"

    # Functions → special handling
    if inspect.isfunction(obj):
        func_serialized = {"function": f"{obj.__module__}.{obj.__name__}"}
        if params: func_serialized["params"] = serialize_component(params)

        # CRITICAL: Keep runtime instance for @framework decorated functions
        if hasattr(obj, 'framework'):
            func_serialized["_runtime_instance"] = obj  # ← NON-SERIALIZABLE!

        return func_serialized

    # Instances → introspect and extract changed params
    params = _changed_kwargs(obj)  # Compare current vs default values
    return {"class": "...", "params": {...}} if params else "..."
```

**`_changed_kwargs()` Mechanism**:
```python
def _changed_kwargs(obj):
    """Extract non-default parameters from an instance."""
    sig = inspect.signature(obj.__class__.__init__)
    out = {}

    for name, param in sig.parameters.items():
        if name == "self": continue

        default = param.default if param.default is not inspect._empty else None
        current = getattr(obj, name, default)

        if current != default:
            out[name] = current

    return out
```

**Purpose of `_runtime_instance`**:
- User provides: `nicon(input_shape, params={})` (a function decorated with `@framework('tensorflow')`)
- Serialization needs the **function reference** for later instantiation with dataset-dependent shape
- String path alone insufficient because function needs to be **called**, not instantiated
- `_runtime_instance` keeps the callable for later use during pipeline execution

**Problem**: The resulting JSON contains live Python objects (not JSON-serializable).

---

### 1.2 Deserialization (`pipeline/serialization.py`)

**Entry**: `deserialize_component(blob: Any, infer_type: Any = None) -> Any`

**Flow for Regular Classes**:
```
{"class": "sklearn.preprocessing._data.StandardScaler", "params": {"feature_range": [0, 1]}}
    ↓
Import module: sklearn.preprocessing._data
    ↓
Get class: StandardScaler
    ↓
Instantiate: StandardScaler(feature_range=(0, 1))
    ↓
Return instance
```

**Flow for `@framework` Functions**:
```
{"function": "nirs4all.operators.models.cirad_tf.nicon", "params": {...}, "_runtime_instance": <function>}
    ↓
Detect hasattr(cls_or_func, 'framework')
    ↓
Return: {"function": "...", "_runtime_instance": <function>}  # ← Special dict for controller
    ↓
TensorFlow controller handles instantiation with input_shape
```

**Key Logic**:
```python
def deserialize_component(blob: Any, infer_type: Any = None) -> Any:
    # ... trivial cases ...

    if isinstance(blob, dict) and ("class" in blob or "function" in blob):
        key = "class" if "class" in blob else "function"

        # Import the class/function
        mod_name, _, cls_or_func_name = blob[key].rpartition(".")
        mod = importlib.import_module(mod_name)
        cls_or_func = getattr(mod, cls_or_func_name)

        # Deserialize params
        params = {k: deserialize_component(v) for k, v in blob.get("params", {}).items()}

        # SPECIAL HANDLING for model factory functions
        if key == "function" and hasattr(cls_or_func, 'framework'):
            return {
                "function": blob[key],
                "_runtime_instance": cls_or_func  # ← Return dict with callable
            }

        # Regular instantiation
        return cls_or_func(**params)
```

**Observation**: Model factory functions are **never instantiated** during deserialization—they're returned as a dict for controller handling.

---

### 1.3 Model Factory (`controllers/models/factory.py`)

**Purpose**: Instantiate models from various config formats with dataset-dependent parameters.

**Entry**: `ModelFactory.build_single_model(model_config, dataset, force_params)`

**Supports**:
1. **String**: `"sklearn.ensemble.RandomForestRegressor"` or `"/path/to/model.pkl"`
2. **Dict with `class`**: `{"class": "...", "params": {...}}`
3. **Dict with `function`**: `{"function": "...", "params": {...}, "framework": "tensorflow"}`
4. **Instance**: `RandomForestRegressor(n_estimators=100)`
5. **Callable**: `nicon` (function) or `RandomForestRegressor` (class)

**Flow for `@framework` Function**:
```
Input: {"function": "nirs4all.operators.models.cirad_tf.nicon", "_runtime_instance": <func>}
    ↓
ModelFactory._from_dict()
    ↓
Detect 'function' key → extract callable (either from string or _runtime_instance)
    ↓
Get framework from function.framework attribute
    ↓
Determine input_shape from dataset: dataset.x(context, layout=...)
    ↓
Call: nicon(input_shape=(features, processings), params={...})
    ↓
Return: TensorFlow Keras Model instance
```

**Key Logic**:
```python
def _from_dict(model_dict, dataset, force_params=None):
    # ...

    elif 'function' in model_dict:
        callable_model = model_dict['function']

        # Import if string path
        if isinstance(callable_model, str):
            mod_name, _, func_name = callable_model.rpartition(".")
            mod = importlib.import_module(mod_name)
            callable_model = getattr(mod, func_name)

        # Get framework and compute input_shape
        framework = getattr(callable_model, 'framework', None)
        input_dim = ModelFactory._get_input_dim(framework, dataset)

        params['input_dim'] = input_dim
        params['input_shape'] = input_dim

        # For classification, add num_classes
        if framework == 'tensorflow' and dataset.is_classification:
            params['num_classes'] = dataset.num_classes

        # Call the function
        model = ModelFactory.prepare_and_call(callable_model, params, force_params)
        return model
```

**`prepare_and_call()` Mechanism**:
- Inspects function signature
- Handles `params` bundle argument (for functions expecting `def model(input_shape, params={})`)
- Handles `**kwargs` (for flexible signatures)
- Filters parameters to match signature

**Observation**: This is where `_runtime_instance` is actually **used**—the factory calls it with computed `input_shape`.

---

### 1.4 Artifact Serialization (`pipeline/artifact_serialization.py`)

**Purpose**: Save/load **trained** models and transformers to/from disk.

**Entry**: `persist(obj, artifacts_dir, name, format_hint) -> ArtifactMeta`

**Flow**:
```
Trained Model: StandardScaler(fitted with data)
    ↓
Detect framework: sklearn
    ↓
Serialize with joblib (or pickle): bytes
    ↓
Compute SHA256 hash: "a3f2e1..."
    ↓
Save to: _binaries/StandardScaler_a3f2e1.joblib
    ↓
Return metadata: {"hash": "sha256:a3f2e1...", "name": "scaler", "path": "...", "format": "joblib", ...}
```

**Framework Detection**:
```python
def _detect_framework(obj: Any) -> str:
    obj_module = type(obj).__module__

    if 'sklearn' in obj_module: return 'sklearn_pickle'
    if 'tensorflow' in obj_module or 'keras' in obj_module: return 'tensorflow_keras'
    if 'torch' in obj_module: return 'pytorch_state_dict'
    # ...
    return 'pickle'  # Fallback
```

**Serialization Formats**:
| Framework | Format | Extension | Method |
|-----------|--------|-----------|--------|
| sklearn | joblib | `.joblib` | `joblib.dump(obj, buffer, compress=3)` |
| TensorFlow | Keras | `.keras` | `model.save(path)` |
| PyTorch | State dict | `.pt` | `torch.save(model.state_dict(), buffer)` |
| Generic | Pickle | `.pkl` | `pickle.dumps(obj, protocol=HIGHEST_PROTOCOL)` |

**Observation**: This layer is **independent** of pipeline config serialization—it only handles **trained** artifacts.

---

## 2. `_runtime_instance` Problem

### 2.1 Why It Exists

**User Intent**: Provide instantiated objects for convenience:
```python
# User writes this:
pipeline = [
    MinMaxScaler(feature_range=(0, 2)),  # Instance
    PlsRegression(n_components=15),       # Instance
    nicon                                 # Function (TensorFlow model)
]

# Instead of this:
pipeline = [
    {"class": "sklearn.preprocessing.MinMaxScaler", "params": {"feature_range": [0, 2]}},
    {"class": "sklearn.cross_decomposition.PLSRegression", "params": {"n_components": 15}},
    {"function": "nirs4all.operators.models.cirad_tf.nicon", "framework": "tensorflow"}
]
```

**Challenge for Model Functions**:
- Regular classes: Extract params via `_changed_kwargs()` → serialize → deserialize → instantiate ✅
- Model functions: Need `input_shape` which **depends on dataset** (not known until runtime) ❌
- Solution: Keep the function reference (`_runtime_instance`) for later call with `input_shape`

### 2.2 Where It's Created

**Location**: `pipeline/serialization.py:serialize_component()`

```python
if inspect.isfunction(obj):
    func_serialized = {"function": f"{obj.__module__}.{obj.__name__}"}

    # Keep runtime instance for @framework decorated functions
    if hasattr(obj, 'framework'):
        func_serialized["_runtime_instance"] = obj  # ← Created here

    return func_serialized
```

### 2.3 Where It's Used

**Location 1**: `pipeline/serialization.py:deserialize_component()`
```python
if key == "function" and hasattr(cls_or_func, 'framework'):
    return {
        "function": blob[key],
        "_runtime_instance": cls_or_func  # ← Returned to controller
    }
```

**Location 2**: `controllers/models/factory.py:_from_dict()`
```python
elif 'function' in model_dict:
    callable_model = model_dict['function']

    # If it's a string, import it (no _runtime_instance)
    # If it's already callable, use it (_runtime_instance case)
```

**Location 3**: Tests (`tests/unit/pipeline/test_serialization.py`)
```python
if isinstance(deserialized, dict) and "_runtime_instance" in deserialized:
    func = deserialized["_runtime_instance"]
    # Test can now call func(input_shape, params)
```

### 2.4 The Cycle

```
User provides: nicon (function)
    ↓
[serialize_component] → {"function": "...", "_runtime_instance": <func>}  # Non-serializable dict
    ↓
[JSON dump] → ❌ TypeError: Object of type function is not JSON serializable
    ↓
[Solution in config.py] → PipelineConfigs.get_hash() ignores _runtime_instance (uses json.dumps with default=str)
    ↓
[deserialize_component] → {"function": "...", "_runtime_instance": <func>}  # Keeps reference
    ↓
[PipelineRunner.run_step] → Passes to controller
    ↓
[BaseModelController._get_model_instance] → Calls ModelFactory.build_single_model()
    ↓
[ModelFactory._from_dict] → Calls function with input_shape from dataset
    ↓
Returns: Keras Model instance (ready for training)
```

### 2.5 Problems

1. **Non-serializable JSON**: `_runtime_instance` prevents clean JSON/YAML export
2. **Hash inconsistency**: `get_hash()` uses `default=str` which converts functions to `<function...>` strings
3. **Redundant work**: Function is imported twice (once in deserialize, once in factory if `_runtime_instance` missing)
4. **Unclear lifecycle**: When is `_runtime_instance` present? Only for `@framework` functions deserialized from serialized config

---

## 3. Responsibility Analysis

### 3.1 Serialization Responsibilities

| Component | Responsibility | Input | Output |
|-----------|---------------|-------|--------|
| `serialize_component()` | Config → JSON-serializable dict | Python objects | Dict with strings/primitives (+ `_runtime_instance`) |
| `deserialize_component()` | Dict → Python objects | Serialized dict | Instantiated objects or dicts with `_runtime_instance` |
| `ModelFactory` | Config → Model instance | Dict/string/callable | Trained-ready model |
| `persist()` | Trained model → Disk | Trained object | ArtifactMeta + file on disk |
| `load()` | Disk → Trained model | ArtifactMeta + path | Loaded object |

### 3.2 Overlaps & Redundancies

**Overlap 1: Model instantiation logic**
- `deserialize_component()` instantiates regular classes
- `ModelFactory.build_single_model()` also instantiates classes
- **Redundancy**: Both inspect signatures and filter parameters

**Overlap 2: Framework detection**
- `ModelFactory.detect_framework(model)` detects by module path
- `artifact_serialization._detect_framework(obj)` detects by module path
- **Redundancy**: Same logic, different locations

**Overlap 3: Parameter introspection**
- `_changed_kwargs(obj)` extracts non-default params
- `ModelFactory._filter_params(cls, params)` filters valid params for signature
- **Overlap**: Both inspect `__init__` signatures

**Overlap 4: String path normalization**
- `serialize_component()` normalizes `"sklearn.preprocessing.StandardScaler"` → internal path
- `ModelFactory.import_class()` imports from string path
- **Mild overlap**: Both handle string → class resolution

---

## 4. Cloning & Re-instantiation

### 4.1 Why Cloning Happens

**Cross-validation**: Each fold needs a fresh model with same architecture but untrained weights.

```python
# Pseudo-code from BaseModelController
for fold in folds:
    model_clone = self._clone_model(base_model)  # Fresh copy
    trained_model = self._train_model(model_clone, X_train, y_train)
    predictions[fold] = self._predict_model(trained_model, X_test)
```

### 4.2 Cloning Mechanisms

**Location 1**: `BaseModelController._clone_model()` (abstract method)

```python
@abstractmethod
def _clone_model(self, model: Any) -> Any:
    """Clone model using framework-specific method."""
    pass
```

**Location 2**: `ModelFactory._clone_model()` (static method)

```python
@staticmethod
def _clone_model(model, framework):
    if framework == 'sklearn':
        from sklearn.base import clone
        return clone(model)
    elif framework == 'tensorflow':
        from tensorflow.keras.models import clone_model
        return clone_model(model)
    else:
        from copy import deepcopy
        return deepcopy(model)
```

**Observation**: Two cloning implementations, but ModelFactory's is not used by controllers.

### 4.3 Special Case: Model Factory Functions

**Problem**: How to "clone" a function?

**Solution in TensorFlow controller**:
```python
def _clone_model(self, model: Any) -> Any:
    if callable(model) and hasattr(model, 'framework'):
        # Don't clone functions—return as-is for later instantiation
        return model
    else:
        # Clone actual model instances
        return super()._clone_model(model)
```

**Observation**: Functions are **not cloned**—they're called fresh each time to create new model instances.

---

## 5. Execution Flow Example

### 5.1 Training Pipeline with `@framework` Function

```python
# User code
from nirs4all.operators.models.cirad_tf import nicon
from sklearn.preprocessing import StandardScaler

pipeline = [
    StandardScaler(),           # Instance
    nicon                       # Function (@framework('tensorflow'))
]

dataset = SpectroDataset(...)
runner = PipelineRunner()
runner.train(pipeline, dataset)
```

**Step-by-step**:

1. **Pipeline Initialization** (`PipelineConfigs.__init__`)
   ```python
   steps = [StandardScaler(), nicon]
   steps = _preprocess_steps(steps)        # No change
   steps = serialize_component(steps)      # Serialize
   ```

   **After serialization**:
   ```python
   [
       "sklearn.preprocessing._data.StandardScaler",  # No params → string only
       {
           "function": "nirs4all.operators.models.cirad_tf.nicon",
           "_runtime_instance": <function nicon>  # ← Non-serializable
       }
   ]
   ```

2. **Deserialization in Runner** (`PipelineRunner.run_step`)
   ```python
   step = deserialize_component(step)
   ```

   **After deserialization**:
   ```python
   [
       StandardScaler(),  # Instantiated
       {
           "function": "nirs4all.operators.models.cirad_tf.nicon",
           "_runtime_instance": <function nicon>  # ← Still present
       }
   ]
   ```

3. **Controller Matching** (`TensorFlowModelController.matches`)
   ```python
   if isinstance(step, dict) and 'model' in step:
       model = step['model']
       return cls._is_tensorflow_model_or_function(model)
   ```

   **Result**: Matches because `model` dict has `_runtime_instance` with `framework` attribute

4. **Model Instantiation** (`TensorFlowModelController._get_model_instance`)
   ```python
   model_config = self._extract_model_config(step, operator)
   model = ModelFactory.build_single_model(model_config, dataset, force_params)
   ```

   **`_extract_model_config()` output**:
   ```python
   {
       'model_instance': <function nicon>  # From _runtime_instance
   }
   ```

   **ModelFactory flow**:
   ```python
   # Detects 'model_instance' key
   model_obj = model_config['model_instance']
   # Calls build_single_model recursively → _from_callable()
   framework = getattr(model_obj, 'framework')  # 'tensorflow'
   input_dim = _get_input_dim('tensorflow', dataset)  # (features, processings)
   model = nicon(input_shape=input_dim, params={})  # ← FUNCTION CALLED HERE
   ```

   **Result**: Keras Model instance ready for training

5. **Training** (`TensorFlowModelController._train_model`)
   ```python
   model.compile(optimizer='adam', loss='mse')
   model.fit(X_train, y_train, epochs=100, ...)
   ```

6. **Artifact Saving** (`persist()`)
   ```python
   artifact = persist(trained_model, artifacts_dir, "nicon_model")
   # Saves to: _binaries/Model_abc123.keras
   ```

---

## 6. Issues & Observations

### 6.1 Strengths

✅ **User-friendly syntax**: Users can provide instances directly
✅ **Framework flexibility**: `@framework` decorator cleanly tags functions
✅ **Lazy instantiation**: Models only built when dataset shape is known
✅ **Artifact deduplication**: Hash-based storage prevents redundant saves
✅ **Framework-aware serialization**: Optimal formats for each framework

### 6.2 Weaknesses

❌ **Non-serializable configs**: `_runtime_instance` breaks JSON export
❌ **Responsibility overlap**: 3 layers with similar instantiation logic
❌ **Unclear lifecycle**: When does `_runtime_instance` exist vs not?
❌ **Redundant introspection**: Signature inspection in multiple places
❌ **Two cloning methods**: ModelFactory and controller implementations
❌ **Inconsistent handling**: TensorFlow controller has special `_extract_model_config`

### 6.3 Root Cause

**Design conflict**:
- **Config serialization** wants JSON/YAML-compatible structures
- **Model functions** need runtime context (input_shape) not available during serialization
- **Current solution**: Embed live object (`_runtime_instance`) as escape hatch
- **Result**: Configs are not truly serializable, only hash-able via `default=str` hack

---

## 7. Questions for Clarification

### 7.1 Confirmed Understanding

✅ `_runtime_instance` allows user convenience (provide instances instead of dicts)
✅ Model functions need dataset-dependent `input_shape` at runtime
✅ `@framework` decorator tags functions for special handling
✅ Artifact serialization is separate from config serialization

### 7.2 Open Questions & Answers

1. **Should configs be fully JSON-serializable?**
   - Current: No (`_runtime_instance` breaks JSON)
   - Alternative: Serialize function as string path only, re-import during deserialization
   - **Decision**: Yes, proceed with fully serializable configs (see refactoring proposal)

2. **Is ModelFactory's cloning method dead code?**
   - Controllers implement `_clone_model()` themselves
   - Factory's `_clone_model()` never called
   - **Decision**: Keep separate—controller responsibilities may differ, especially for custom controllers

3. **Should model instantiation live in one place?**
   - Current: `deserialize_component()` instantiates some, `ModelFactory` instantiates others
   - Alternative: `deserialize_component()` only validates, `ModelFactory` always instantiates
   - **Decision**: See detailed analysis in refactoring proposal—two viable approaches exist

4. **Why does TensorFlow controller have `_extract_model_config()`?**
   - It wraps `_runtime_instance` in `model_instance` key
   - Could this be handled generically in `BaseModelController`?
   - **Decision**: Move to base controller if possible, but preserve controller-specific flexibility

---

## 8. Summary

**Current State**: 3-layer serialization system with `_runtime_instance` as bridge mechanism for model factory functions.

**Layers**:
1. **Config → JSON** (`serialization.py`): User objects → dicts (with `_runtime_instance` escape hatch)
2. **Model Config → Instance** (`factory.py`): Dicts/strings/callables → trained-ready models
3. **Trained → Disk** (`artifact_serialization.py`): Fitted models → framework-specific files

**Key Insight**: `_runtime_instance` serves a legitimate purpose (deferred instantiation for dataset-dependent functions) but creates architectural tension between serializability and convenience.

**Architectural Tension**: The central question is whether model creation/cloning/saving should be:
- **Centralized** in ModelFactory (maintainability)
- **Delegated** to controllers (extensibility for custom controllers)

See refactoring proposal for detailed comparison and recommendation.

**Next Steps**: See refactoring proposal document for solutions.

---

**Document prepared by**: GitHub Copilot
**Review required**: Completed
**Related documents**: `serialization_refactoring_proposal.md`