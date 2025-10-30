# TensorFlow Model Controller Refactoring Proposal

**Date:** October 30, 2025
**Author:** GitHub Copilot
**Status:** Draft for Review

---

## Executive Summary

The `TensorFlowModelController` (760 lines) has grown complex with mixed responsibilities for:
- Model creation/configuration
- Training orchestration
- Compilation configuration
- Callback management
- Data preparation

This proposal recommends **splitting responsibilities into focused components** while **eliminating redundancy** with `ModelUtils` and `ModelBuilderFactory`.

**Expected Outcome:**
- 60% reduction in main controller size (~300 lines)
- Clear separation of concerns
- Elimination of duplicated logic
- Better testability and maintainability

---

## Problems Identified

### 1. **Overlapping Responsibilities with ModelBuilderFactory**

**Current Issues:**
- `_get_model_instance()` calls `ModelBuilderFactory.build_single_model()` but then has fallback logic
- `_create_model_from_function()` duplicates model creation logic
- `_extract_model_config()` has 60+ lines of config extraction that could be in ModelBuilderFactory

**Impact:** Maintenance burden when model configuration formats change

### 2. **Mixed Compilation & Training Configuration**

**Current Issues:**
- `_prepare_compilation_config()` (40 lines)
- `_configure_optimizer()` (30 lines)
- `_prepare_fit_config()` (35 lines)
- `_configure_callbacks()` (90 lines)

These 195 lines (25% of file) are all TensorFlow-specific configuration management.

**Impact:** Hard to test, hard to understand, hard to modify

### 3. **Redundancy with ModelUtils**

**Current Issues:**
- Task type detection duplicated: `_detect_task_type()` calls `ModelUtils.detect_task_type()`
- Loss/metric configuration: Uses `ModelUtils.get_default_loss()` but has custom validation
- Score calculation: Delegates to `ModelUtils.calculate_scores()` via parent class

**Impact:** Unnecessary abstraction layer

### 4. **Data Preparation Logic Mixed with Model Logic**

**Current Issues:**
- `_prepare_data()` (30 lines) handles tensor reshaping
- Layout preference (`get_preferred_layout()`) is in controller
- Shape validation logic scattered

**Impact:** Coupling between model and data concerns

### 5. **Verbose Configuration Logging**

**Current Issues:**
- `_log_training_config()` (30 lines) for verbose output
- Scattered `if verbose > X:` checks throughout training
- Mix of business logic and logging

**Impact:** Cluttered code, hard to maintain verbose levels

---

## Proposed Architecture

### Phase 1: Extract Configuration Management (Priority: HIGH)

**Create:** `nirs4all/controllers/models/tensorflow/config.py`

```python
class TensorFlowCompilationConfig:
    """Manages TensorFlow model compilation configuration."""

    @staticmethod
    def prepare(train_params: Dict, task_type: TaskType) -> Dict:
        """Prepare compilation config from train_params."""
        # Consolidates _prepare_compilation_config + _configure_optimizer
        pass

    @staticmethod
    def create_optimizer(optimizer_name: str, learning_rate: float) -> keras.optimizers.Optimizer:
        """Create optimizer instance with learning rate."""
        pass


class TensorFlowFitConfig:
    """Manages TensorFlow model fit configuration."""

    @staticmethod
    def prepare(train_params: Dict, X_val, y_val, verbose: int) -> Dict:
        """Prepare fit config including validation setup."""
        pass


class TensorFlowCallbackFactory:
    """Factory for creating TensorFlow callbacks."""

    @staticmethod
    def create_early_stopping(params: Dict, verbose: int) -> keras.callbacks.EarlyStopping:
        pass

    @staticmethod
    def create_cyclic_lr(params: Dict, verbose: int) -> keras.callbacks.LearningRateScheduler:
        pass

    @staticmethod
    def create_reduce_lr_on_plateau(params: Dict, verbose: int) -> keras.callbacks.ReduceLROnPlateau:
        pass

    @staticmethod
    def create_best_model_memory(verbose: bool) -> keras.callbacks.Callback:
        pass

    @staticmethod
    def build_callbacks(train_params: Dict, existing: List, verbose: int) -> List:
        """Orchestrate callback creation from train_params."""
        pass
```

**Benefits:**
- 195 lines moved out of main controller
- Each class has single responsibility
- Easy to test independently
- Clear API for configuration

---

### Phase 2: Simplify Model Instance Creation (Priority: HIGH)

**Modify:** `ModelBuilderFactory` to handle TensorFlow model functions better

**Current Problem:**
```python
def _get_model_instance(...):
    try:
        model = ModelBuilderFactory.build_single_model(...)
        return model
    except Exception as e:
        # Fallback for legacy formats (15 lines)
        if 'model_instance' in model_config:
            ...
        if 'model_factory' in model_config:
            ...
```

**Proposed:**
```python
def _get_model_instance(self, dataset, model_config, force_params=None):
    """Delegate entirely to ModelBuilderFactory."""
    return ModelBuilderFactory.build_single_model(
        model_config,
        dataset,
        force_params or {}
    )
```

**Remove:**
- `_create_model_from_function()` (30 lines) - move to ModelBuilderFactory
- Fallback logic in `_get_model_instance()` (15 lines)

**Update ModelBuilderFactory:**
- Add `_from_tensorflow_function()` method to handle TensorFlow-specific model creation
- Consolidate all model creation paths

**Benefits:**
- 45 lines removed from controller
- Single source of truth for model creation
- Better error messages from centralized location

---

### Phase 3: Extract Data Preparation (Priority: MEDIUM)

**Create:** `nirs4all/controllers/models/tensorflow/data_prep.py`

```python
class TensorFlowDataPreparation:
    """Handles TensorFlow-specific data preparation and reshaping."""

    @staticmethod
    def prepare_features(X: np.ndarray) -> np.ndarray:
        """Convert features to proper tensor format for TensorFlow."""
        # Logic from _prepare_data for X transformation
        pass

    @staticmethod
    def prepare_targets(y: np.ndarray) -> np.ndarray:
        """Prepare targets for TensorFlow training."""
        # Logic from _prepare_data for y transformation
        pass

    @staticmethod
    def prepare(X: np.ndarray, y: Optional[np.ndarray], context: Dict) -> Tuple:
        """Complete data preparation pipeline."""
        pass

    @staticmethod
    def get_preferred_layout() -> str:
        """Return layout preference for TensorFlow."""
        return "3d_transpose"
```

**Benefits:**
- 30 lines moved out
- Reusable across TensorFlow operations
- Easier to test data transformations
- Clear naming of what's happening

---

### Phase 4: Simplify Training Method (Priority: HIGH)

**Current `_train_model()` is 100+ lines doing:**
1. Parameter extraction (5 lines)
2. Model instantiation if callable (5 lines)
3. Task type detection (2 lines)
4. Loss/metric configuration (15 lines)
5. Compilation (10 lines)
6. Fit configuration (10 lines)
7. Training execution (5 lines)
8. Score calculation and logging (40 lines)

**Proposed Refactored Version:**

```python
def _train_model(
    self,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    **kwargs
) -> Any:
    """Train TensorFlow/Keras model."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is not available")

    train_params = kwargs
    verbose = train_params.get('verbose', 0)

    # 1. Ensure we have a model instance (not a callable)
    if callable(model) and not self._is_tensorflow_model(model):
        model = ModelBuilderFactory.build_model_from_callable(
            model,
            input_shape=X_train.shape[1:],
            params=train_params.get('model_params', {})
        )

    # 2. Detect task and configure compilation
    task_type = ModelUtils.detect_task_type(y_train)
    compile_config = TensorFlowCompilationConfig.prepare(train_params, task_type)
    model.compile(**compile_config)

    # 3. Configure training
    fit_config = TensorFlowFitConfig.prepare(train_params, X_val, y_val, verbose)
    validation_data = fit_config.pop('validation_data', None)

    # 4. Train
    history = model.fit(X_train, y_train, validation_data=validation_data, **fit_config)
    model.history = history

    # 5. Log results if verbose
    if verbose > 1:
        self._log_training_results(model, X_train, y_train, X_val, y_val, task_type)

    return model
```

**Extract to separate method:**
```python
def _log_training_results(self, model, X_train, y_train, X_val, y_val, task_type):
    """Log training and validation scores."""
    # Current 40 lines of score logging moved here
    pass
```

**Benefits:**
- Main training logic: 100+ lines → ~35 lines
- Clear 5-step process
- Easy to understand flow
- Logging separated from training logic

---

### Phase 5: Consolidate Configuration Extraction (Priority: MEDIUM)

**Current Issue:**
`_extract_model_config()` is 60+ lines handling:
- Function serialization
- Class serialization
- Runtime instances
- Legacy formats
- Nested model dictionaries

**Proposed:**
Move this logic to `ModelBuilderFactory` or a dedicated `ConfigExtractor` class.

```python
class ModelConfigExtractor:
    """Extract and normalize model configurations from various formats."""

    @staticmethod
    def extract(step: Any, operator: Any = None) -> Dict[str, Any]:
        """Extract unified model config from step and operator."""
        pass

    @staticmethod
    def normalize_tensorflow_config(config: Dict) -> Dict:
        """Normalize TensorFlow-specific configurations."""
        pass
```

**Benefits:**
- 60 lines removed from controller
- Reusable across different model types
- Single place to handle config format changes
- Easier to add new config formats

---

### Phase 6: Remove Redundant Wrapper Methods (Priority: LOW)

**Current Redundant Methods:**

1. **`_detect_task_type()`** - Just calls `ModelUtils.detect_task_type()`
   - **Action:** Remove, call `ModelUtils` directly

2. **`_calculate_and_print_scores()`** - Delegates to parent class
   - **Action:** Remove wrapper, use parent directly

3. **`_is_tensorflow_model()`** - Could be in ModelBuilderFactory
   - **Action:** Move to `ModelBuilderFactory.is_tensorflow_model()`

**Benefits:**
- 40 lines removed
- Less indirection
- Clearer where logic lives

---

## Proposed File Structure

```
nirs4all/controllers/models/
├── base_model.py                    (unchanged)
├── sklearn_model.py                 (unchanged)
├── tensorflow_model.py              (~300 lines, down from 760)
└── tensorflow/
    ├── __init__.py
    ├── config.py                    (195 lines - compilation, fit, callbacks)
    ├── data_prep.py                 (30 lines - data preparation)
    └── model_factory.py             (optional - model creation helpers)

nirs4all/utils/
├── model_builder.py                 (enhanced with TF-specific logic)
└── model_utils.py                   (unchanged)
```

---

## Migration Strategy

### Step 1: Create New Components (Non-breaking)
1. Create `tensorflow/config.py` with configuration classes
2. Create `tensorflow/data_prep.py` with data preparation
3. Add tests for new components

### Step 2: Refactor Controller (Breaking changes isolated)
1. Update `_train_model()` to use new components
2. Replace `_prepare_data()` with `TensorFlowDataPreparation`
3. Remove redundant methods

### Step 3: Update ModelBuilderFactory (Isolated changes)
1. Move `_create_model_from_function()` logic to ModelBuilderFactory
2. Add TensorFlow-specific model creation methods
3. Update `_get_model_instance()` to pure delegation

### Step 4: Cleanup and Documentation
1. Remove old code
2. Update docstrings
3. Add migration guide for custom model functions

---

## Backward Compatibility

### Breaking Changes (Minimal)
- Internal methods become private/removed (already prefixed with `_`)
- No public API changes

### Safe Changes
- All public methods maintain same signatures
- Existing pipelines continue to work
- New components are internal implementation details

---

## Testing Strategy

### New Component Tests
```python
# test_tensorflow_config.py
def test_compilation_config_with_task_type()
def test_optimizer_creation_with_learning_rate()
def test_callback_factory_early_stopping()
def test_callback_factory_cyclic_lr()

# test_tensorflow_data_prep.py
def test_prepare_features_2d_to_3d()
def test_prepare_features_3d_transpose()
def test_prepare_targets_flatten()
```

### Integration Tests (Existing)
- All existing tests should pass without modification
- Add new tests for refactored `_train_model()` flow

---

## Expected Benefits Summary

| Metric | Current | Proposed | Improvement |
|--------|---------|----------|-------------|
| Controller LOC | 760 | ~300 | 60% reduction |
| Methods in Controller | 20 | ~12 | 40% reduction |
| Configuration Logic (isolated) | Mixed | 195 lines in `config.py` | Separated |
| Data Prep Logic (isolated) | Mixed | 30 lines in `data_prep.py` | Separated |
| Code Duplication | High (ModelBuilder overlap) | Low | Eliminated |
| Testability | Medium (tightly coupled) | High (focused components) | Improved |
| Maintainability | Low (find logic scattered) | High (clear responsibilities) | Improved |

---

## Estimated Effort

- **Phase 1 (Config extraction):** 4-6 hours
- **Phase 2 (Model creation):** 3-4 hours
- **Phase 3 (Data prep):** 2-3 hours
- **Phase 4 (Training refactor):** 3-4 hours
- **Phase 5 (Config extraction):** 2-3 hours
- **Phase 6 (Cleanup):** 1-2 hours
- **Testing & Documentation:** 4-6 hours

**Total: 19-28 hours** (2-3.5 days of focused work)

---

## Additional Recommendations

### 1. Consider Strategy Pattern for Callbacks
Instead of if/else chains in `_configure_callbacks()`, use a registry:

```python
CALLBACK_STRATEGIES = {
    'early_stopping': TensorFlowCallbackFactory.create_early_stopping,
    'cyclic_lr': TensorFlowCallbackFactory.create_cyclic_lr,
    'reduce_lr_on_plateau': TensorFlowCallbackFactory.create_reduce_lr_on_plateau,
}

for callback_name, enabled in train_params.items():
    if enabled and callback_name in CALLBACK_STRATEGIES:
        callbacks.append(CALLBACK_STRATEGIES[callback_name](train_params, verbose))
```

### 2. Separate Verbose Logging Concerns
Consider a `TensorFlowTrainingLogger` class to handle all verbose output:

```python
class TensorFlowTrainingLogger:
    def __init__(self, verbose: int):
        self.verbose = verbose

    def log_compilation(self, config: Dict):
        if self.verbose > 2:
            print(f" Model compiled with: {config}")

    def log_training_config(self, fit_config: Dict, train_params: Dict):
        if self.verbose > 2:
            # Current _log_training_config logic
            pass

    def log_training_results(self, model, scores: Dict, task_type: TaskType):
        if self.verbose > 1:
            # Current score logging logic
            pass
```

### 3. Type Hints and Protocols
Add Protocol for TensorFlow model interface:

```python
from typing import Protocol

class TensorFlowModel(Protocol):
    """Protocol defining TensorFlow model interface."""
    def compile(self, optimizer, loss, metrics) -> None: ...
    def fit(self, X, y, **kwargs) -> Any: ...
    def predict(self, X, **kwargs) -> np.ndarray: ...
```

---

## Questions for Review

1. **Priority:** Should we tackle all phases or focus on high-priority items first?
2. **Backward Compatibility:** Are there any custom model functions in use that might break?
3. **Testing:** Do we have good integration test coverage for TensorFlow models?
4. **Timeline:** Is 2-3.5 days acceptable for this refactoring?
5. **ModelBuilderFactory:** Should we refactor it at the same time or separately?

---

## Conclusion

This refactoring will transform the TensorFlow controller from a monolithic 760-line file into a clean, maintainable architecture with:

- **Clear separation of concerns** (config, data prep, training, logging)
- **Elimination of redundancy** with ModelUtils and ModelBuilderFactory
- **Improved testability** through focused components
- **Better maintainability** with each component having a single responsibility
- **60% reduction in controller complexity**

The proposed changes are **mostly non-breaking** and can be implemented **incrementally** with comprehensive tests at each phase.

**Next Steps:**
1. Review and approve proposal
2. Prioritize phases
3. Create implementation tasks
4. Begin Phase 1 (highest impact)

---

**Ready to proceed? Please review and provide feedback.**
