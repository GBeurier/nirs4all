# Model Refactoring Notes

**Date:** October 30, 2025
**Status:** In Progress - base_model.py refactored, additional improvements identified

---

## Completed Refactoring (base_model.py)

### Summary

Successfully implemented Phase 1 of the base_model refactoring proposal:

- **Reduced `launch_training()` from 278 lines to ~140 lines** (with comprehensive docstring)
- **Created 6 modular components** in `nirs4all/controllers/models/components/`:
  1. `identifier_generator.py` - Model naming and ID generation
  2. `prediction_transformer.py` - Scaling/unscaling predictions
  3. `prediction_assembler.py` - Assembling prediction records
  4. `model_loader.py` - Loading models from binaries
  5. `score_calculator.py` - Score calculation with ModelUtils/Evaluator
  6. `index_normalizer.py` - Index normalization and validation

### Benefits Achieved

✅ **Single Responsibility**: Each component has one clear purpose
✅ **Testability**: Components can be unit tested independently
✅ **Readability**: launch_training() now has clear sequential steps
✅ **Maintainability**: Changes to naming, scoring, etc. are isolated
✅ **Reusability**: Components can be used by tensorflow_model, sklearn_model

### Key Design Decisions

1. **Components as Services**: Each component is stateless and can be reused
2. **Dataclasses for Structure**: Used `@dataclass` for ModelIdentifiers, PartitionScores, etc.
3. **Preserved Public API**: `execute()` signature unchanged, only internal refactoring
4. **Google Style Docstrings**: All components have comprehensive documentation

---

## Future Improvements

### 1. TensorFlow Model Controller (`tensorflow_model.py`)

**Current Issues:**
- `_train_model()` method is 308 lines (lines 171-478) - too large
- Complex compilation and callback configuration logic embedded
- Duplicate parameter handling between compile and fit configs
- Score calculation duplicated from base class

**Proposed Improvements:**

#### A. Extract Compilation Configuration Component
```python
# components/tensorflow_compiler.py
class TensorFlowCompiler:
    """Handles TensorFlow model compilation configuration."""

    def prepare_compile_config(self, train_params, task_type):
        """Extract compilation parameters and auto-configure for task type."""

    def configure_optimizer(self, compile_config):
        """Handle optimizer instantiation and learning rate."""
```

#### B. Extract Callback Management Component
```python
# components/tensorflow_callbacks.py
class TensorFlowCallbackManager:
    """Manages TensorFlow training callbacks."""

    def configure_callbacks(self, train_params, verbose=0):
        """Build list of callbacks from configuration."""

    def create_early_stopping(self, params):
        """Create EarlyStopping callback."""

    def create_best_model_memory_callback(self, verbose=False):
        """Create custom callback to store best model in memory."""
```

#### C. Extract Fit Configuration Component
```python
# components/tensorflow_fit_config.py
class TensorFlowFitConfig:
    """Prepares fit() parameters for TensorFlow models."""

    def prepare_fit_config(self, train_params, X_val, y_val, callbacks, verbose=0):
        """Extract and validate fit parameters."""
```

#### D. Refactor `_train_model()` Structure
```python
def _train_model(self, model, X_train, y_train, X_val=None, y_val=None, **kwargs):
    """Train TensorFlow model - refactored to use components."""

    # 1. Configuration
    compiler = TensorFlowCompiler()
    callback_mgr = TensorFlowCallbackManager()
    fit_config_builder = TensorFlowFitConfig()

    # 2. Compile model
    compile_config = compiler.prepare_compile_config(kwargs, task_type)
    model.compile(**compile_config)

    # 3. Configure callbacks
    callbacks = callback_mgr.configure_callbacks(kwargs, verbose)

    # 4. Prepare fit config
    fit_config = fit_config_builder.prepare_fit_config(kwargs, X_val, y_val, callbacks, verbose)

    # 5. Train
    history = model.fit(X_train, y_train, **fit_config)
    model.history = history

    # 6. Score calculation (reuse base class component)
    return model
```

**Expected Impact:**
- Reduce `_train_model()` from 308 lines to ~40 lines
- Improve testability of TensorFlow-specific logic
- Eliminate duplicate parameter extraction
- Easier to add new compilation/callback strategies

---

### 2. Scikit-Learn Model Controller (`sklearn_model.py`)

**Current Issues:**
- `_train_model()` is simpler (162 lines) but still has embedded concerns
- Parameter filtering logic is inline
- Score display logic duplicated from base class

**Proposed Improvements:**

#### A. Extract Parameter Filtering Component
```python
# components/sklearn_params_filter.py
class SklearnParamsFilter:
    """Filters training parameters to match model signature."""

    def filter_valid_params(self, model, train_params):
        """Return only parameters that exist in model's signature."""
```

#### B. Simplify `_train_model()`
```python
def _train_model(self, model, X_train, y_train, X_val=None, y_val=None, train_params=None):
    """Train sklearn model - simplified."""

    # 1. Filter parameters
    param_filter = SklearnParamsFilter()
    valid_params = param_filter.filter_valid_params(model, train_params)

    # 2. Set parameters
    if valid_params:
        model.set_params(**valid_params)

    # 3. Fit
    model.fit(X_train, y_train.ravel())

    # 4. Optional score display (reuse base class component)
    if self.verbose > 1:
        self._display_training_scores(model, X_train, y_train, X_val, y_val)

    return model
```

**Expected Impact:**
- Reduce `_train_model()` from 162 lines to ~25 lines
- Centralize parameter validation logic
- Improve consistency with TensorFlow controller

---

### 3. Model Builder (`model_builder.py`)

**Current Issues:**
- Large `ModelBuilderFactory` class with many responsibilities
- `build_single_model()` has complex branching (lines 33-72)
- Framework detection logic embedded
- Input dimension calculation mixed with model building

**Proposed Improvements:**

#### A. Split into Multiple Builders (Strategy Pattern)
```python
# utils/model_builder/base_builder.py
class ModelBuilder(ABC):
    """Base builder for all model types."""

    @abstractmethod
    def can_build(self, model_config):
        """Check if this builder can handle the config."""

    @abstractmethod
    def build(self, model_config, dataset, force_params=None):
        """Build model from configuration."""

# utils/model_builder/string_builder.py
class StringModelBuilder(ModelBuilder):
    """Builds models from string paths (file paths or class paths)."""

# utils/model_builder/dict_builder.py
class DictModelBuilder(ModelBuilder):
    """Builds models from dictionary configurations."""

# utils/model_builder/instance_builder.py
class InstanceModelBuilder(ModelBuilder):
    """Handles pre-built model instances."""

# utils/model_builder/callable_builder.py
class CallableModelBuilder(ModelBuilder):
    """Builds models from callable functions/classes."""
```

#### B. Extract Framework Detection
```python
# utils/model_builder/framework_detector.py
class FrameworkDetector:
    """Detects ML framework from model or configuration."""

    def detect(self, model_or_config):
        """Return 'sklearn', 'tensorflow', 'pytorch', etc."""
```

#### C. Extract Input Dimension Calculator
```python
# utils/model_builder/input_dim_calculator.py
class InputDimCalculator:
    """Calculates input dimensions for model instantiation."""

    def calculate(self, dataset, framework, layout):
        """Return input_dim tuple for model constructor."""
```

#### D. Refactor ModelBuilderFactory
```python
class ModelBuilderFactory:
    """Factory coordinating multiple model builders."""

    def __init__(self):
        self.builders = [
            StringModelBuilder(),
            DictModelBuilder(),
            InstanceModelBuilder(),
            CallableModelBuilder()
        ]
        self.framework_detector = FrameworkDetector()
        self.input_calculator = InputDimCalculator()

    def build_single_model(self, model_config, dataset, force_params=None):
        """Build model by delegating to appropriate builder."""

        # Find appropriate builder
        for builder in self.builders:
            if builder.can_build(model_config):
                return builder.build(model_config, dataset, force_params)

        raise ValueError(f"No builder found for config: {model_config}")
```

**Expected Impact:**
- Reduce complexity through Strategy pattern
- Easier to add new model sources (e.g., HuggingFace, remote URLs)
- Better separation of framework-specific logic
- Improved testability of each builder type

---

### 4. Helper.py Improvements

**Current Issues:**
- `ModelControllerHelper` has mixed responsibilities
- Some methods duplicate logic in new components
- Model info/validation methods underutilized

**Proposed Improvements:**

#### A. Deprecate Duplicate Methods
Methods now handled by components:
- `create_model_identifiers()` → Use `ModelIdentifierGenerator`
- Scoring methods → Use `ScoreCalculator`

#### B. Focus on Core Helper Utilities
Keep and enhance:
- `extract_core_name()` - Still needed by identifier generator
- `extract_classname_from_config()` - Core utility
- `clone_model()` - Framework-agnostic cloning
- `sanitize_model_name()` - File path safety
- `get_model_info()` - Model introspection
- `validate_model()` - Interface validation

#### C. Move Serialization to Dedicated Module
```python
# utils/serialization/model_serializer.py
class ModelSerializer:
    """Handles model serialization across frameworks."""

    def is_serializable(self, model):
        """Check if model can be serialized."""

    def serialize(self, model):
        """Serialize model to bytes."""

    def deserialize(self, binary_data, framework):
        """Deserialize model from bytes."""
```

---

### 5. Fold Averaging Improvements (Phase 2 from Proposal)

**Current Issues:**
- `_create_fold_averages()` is 270 lines (lines 663-933)
- Re-predicts all data instead of caching predictions
- Duplicate transformation and assembly logic

**Proposed Strategy Components:**

#### A. Prediction Cache
```python
# components/strategies/prediction_cache.py
@dataclass
class CachedPrediction:
    """Cached predictions from fold training."""
    model: Any
    predictions_scaled: dict
    predictions_unscaled: dict
    scores: PartitionScores

class PredictionCache:
    """Caches predictions during fold training to avoid re-prediction."""

    def store(self, fold_idx, cached_prediction):
        """Store fold predictions."""

    def get_all(self):
        """Get all cached fold predictions."""
```

#### B. Averaging Strategy
```python
# components/strategies/averaging.py
class AveragingStrategy(ABC):
    """Base strategy for fold prediction averaging."""

    @abstractmethod
    def average(self, cached_predictions):
        """Average predictions from multiple folds."""

class SimpleAverageStrategy(AveragingStrategy):
    """Arithmetic mean of fold predictions."""

class WeightedAverageStrategy(AveragingStrategy):
    """Score-weighted average of fold predictions."""
```

#### C. Refactor `_create_fold_averages()`
```python
def _create_fold_averages(self, fold_models, fold_predictions_cache, ...):
    """Create fold-averaged predictions - refactored to use cache and strategy."""

    # 1. Select averaging strategy
    strategy = WeightedAverageStrategy() if use_weighted else SimpleAverageStrategy()

    # 2. Average predictions (no re-prediction needed!)
    averaged = strategy.average(fold_predictions_cache)

    # 3. Calculate scores
    scores = self.score_calculator.calculate(true_values, averaged.predictions, task_type)

    # 4. Assemble prediction record
    prediction_data = self.prediction_assembler.assemble_fold_average(
        base_prediction,
        averaged.predictions,
        scores,
        is_weighted=use_weighted
    )

    return prediction_data
```

**Expected Impact:**
- Reduce `_create_fold_averages()` from 270 lines to ~40 lines
- **Performance improvement: 20-40% faster** (no re-prediction)
- Eliminate duplicate transformation logic
- Easier to add new averaging strategies (e.g., rank-weighted, median)

---

## Testing Strategy

### Priority 1: Component Unit Tests
Create tests for each new component:
```
tests/unit/controllers/models/components/
├── test_identifier_generator.py
├── test_prediction_transformer.py
├── test_prediction_assembler.py
├── test_model_loader.py
├── test_score_calculator.py
└── test_index_normalizer.py
```

### Priority 2: Integration Tests
Test components working together:
```
tests/integration/controllers/models/
├── test_base_model_refactored.py
├── test_tensorflow_model.py
└── test_sklearn_model.py
```

### Priority 3: Regression Tests
Ensure refactoring didn't break existing behavior:
- Run all existing examples (Q1-Q14)
- Compare prediction outputs before/after refactoring
- Verify model persistence/loading still works

---

## Implementation Roadmap

### Phase 1: ✅ COMPLETED (base_model.py)
- [x] Create component infrastructure
- [x] Implement 6 core components
- [x] Refactor launch_training()
- [x] Update imports and __init__.py

### Phase 2: TensorFlow Controller (Week 2)
- [ ] Create TensorFlowCompiler component
- [ ] Create TensorFlowCallbackManager component
- [ ] Create TensorFlowFitConfig component
- [ ] Refactor _train_model()
- [ ] Unit tests for TensorFlow components

### Phase 3: Sklearn Controller (Week 3)
- [ ] Create SklearnParamsFilter component
- [ ] Refactor _train_model()
- [ ] Unit tests for Sklearn components
- [ ] Integration tests

### Phase 4: Model Builder (Week 4)
- [ ] Design builder strategy pattern
- [ ] Implement StringModelBuilder
- [ ] Implement DictModelBuilder
- [ ] Implement InstanceModelBuilder
- [ ] Implement CallableModelBuilder
- [ ] Extract FrameworkDetector
- [ ] Extract InputDimCalculator
- [ ] Refactor ModelBuilderFactory

### Phase 5: Fold Averaging (Week 5)
- [ ] Create PredictionCache
- [ ] Create AveragingStrategy hierarchy
- [ ] Refactor _create_fold_averages()
- [ ] Performance benchmarks
- [ ] Integration tests

### Phase 6: Testing & Documentation (Week 6)
- [ ] Complete unit test coverage (>90%)
- [ ] Integration test suite
- [ ] Regression test all examples
- [ ] Update developer documentation
- [ ] Performance benchmark report

---

## Metrics & Success Criteria

### Code Quality Metrics
- ✅ **base_model.py**: Reduced launch_training from 278 → ~140 lines (49% reduction)
- **Target for tensorflow_model.py**: Reduce _train_model from 308 → ~40 lines (87% reduction)
- **Target for sklearn_model.py**: Reduce _train_model from 162 → ~25 lines (85% reduction)
- **Target for model_builder.py**: Reduce build_single_model from ~70 → ~20 lines (71% reduction)

### Test Coverage Metrics
- **Current**: ~0% for model controllers
- **Target**: >90% for all components and controllers

### Performance Metrics
- **Fold Averaging**: Target 20-40% speedup with prediction caching
- **No Regression**: All examples must run with same results

---

## Breaking Changes & Migration

### Version Bump
- **From**: 0.4.1
- **To**: 0.5.0 (minor version for internal API changes)

### Breaking Changes
- **Internal only**: launch_training() signature changed internally but execute() preserved
- **Components**: New dependencies in nirs4all.controllers.models.components
- **No user-facing changes**: Public API (execute, fit, predict) unchanged

### Migration Guide for Custom Controllers
If users extended BaseModelController:

```python
# OLD (0.4.1)
class MyController(BaseModelController):
    def launch_training(self, ...):
        # Custom implementation
        pass

# NEW (0.5.0)
class MyController(BaseModelController):
    # Use components for common tasks
    def launch_training(self, ...):
        identifiers = self.identifier_generator.generate(...)
        predictions = self.prediction_transformer.transform_to_unscaled(...)
        # etc.
```

---

## Notes & Considerations

### Design Philosophy
- **Pragmatic over Perfect**: Prioritize working code over theoretical purity
- **Testability First**: Every component must be easily testable
- **Backwards Compatibility**: Public APIs remain stable
- **Progressive Enhancement**: Can be done incrementally, phase by phase

### Risk Mitigation
- **Comprehensive Testing**: Every refactored method gets tests
- **Gradual Rollout**: Phase-by-phase implementation
- **Rollback Plan**: Keep old methods commented during transition
- **User Communication**: Clear changelog and migration guide

### Future Opportunities
- **PyTorch Support**: New components make adding PyTorch controller easier
- **JAX Support**: Same modular structure applies
- **Remote Models**: Builder pattern enables HuggingFace integration
- **Model Registry**: Components support future model versioning/tracking

---

**Last Updated:** October 30, 2025
**Next Review:** After Phase 2 completion (TensorFlow Controller)
