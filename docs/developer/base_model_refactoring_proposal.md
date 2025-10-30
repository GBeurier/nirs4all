# Base Model Controller Refactoring Proposal

**Version:** 0.5.0 (Breaking changes in internal APIs only)
**Date:** October 30, 2025
**Status:** Proposal - Awaiting Approval

---

## Executive Summary

The base_model controller suffers from severe maintainability issues stemming from a 278-line "god method" (`launch_training()`), zero test coverage, scattered responsibilities, and performance inefficiencies. This proposal outlines a comprehensive refactoring that will:

- **Improve Maintainability**: Break down monolithic methods into 6 testable components
- **Enhance Performance**: Eliminate re-prediction overhead (20-40% speedup in fold-based training)
- **Increase Test Coverage**: From 0% to >90% for model controllers
- **Preserve Public API**: Keep `execute()` signature intact, only change internal methods

**Key Metrics:**
- Current `launch_training()`: 278 lines, 12+ responsibilities
- Target: 6 focused components, each <50 lines
- Current test coverage: 0%
- Target test coverage: >90%

---

## Current Architecture Issues

### Critical Problems

1. **God Method Antipattern** - `launch_training()` (278 lines)
   - Name extraction, model loading, instantiation, training, prediction, transformation, scoring, assembly
   - Impossible to test individual pieces
   - High cognitive load and modification risk

2. **Zero Test Coverage**
   - No tests found for model controllers
   - Changes are high-risk without safety net

3. **Performance Inefficiencies**
   - Re-predicts all data for fold averaging (should cache)
   - Keeps all fold models in memory (10 folds × 100MB = 1GB)
   - Duplicates predictions during assembly

4. **Scattered Concerns**
   - Mode branching across 3 methods (execute, train, launch_training)
   - Config extraction with 40+ lines of branching
   - Duplicate scoring logic in sklearn and tensorflow controllers

5. **Poor Documentation**
   - Missing docstrings: `train()`, `launch_training()`, `_create_fold_averages()`, `_add_all_predictions()`
   - TensorFlow callback methods (75 lines) undocumented

### Code Evidence

```python
# Current launch_training() structure - TOO MANY RESPONSIBILITIES
def launch_training(self, dataset, model_config, context, runner, ...):
    # 1. Extract model classname (10 lines)
    # 2. Load model from binaries for predict mode (30 lines)
    # 3. Generate identifiers (7 lines)
    # 4. Instantiate/clone model (20 lines)
    # 5. Handle explain mode capture (10 lines)
    # 6. Prepare data (3 lines)
    # 7. Train model (2 lines)
    # 8. Predict on train/val/test (9 lines)
    # 9. Unscale predictions (20 lines)
    # 10. Calculate scores (10 lines)
    # 11. Convert indices (10 lines)
    # 12. Assemble prediction data dict (35 lines)
    return trained_model, model_id, score, name, prediction_data
```

---

## Refactoring Strategy

### Design Principles

1. **Single Responsibility**: Each component does ONE thing well
2. **Open/Closed**: Easy to extend without modifying core logic
3. **Dependency Inversion**: Depend on abstractions, not concrete implementations
4. **Interface Segregation**: Small, focused interfaces
5. **Preserve Public API**: No breaking changes to `execute()` signature

### Version Strategy

- **Version Bump**: 0.4.1 → 0.5.0 (minor version bump for internal API changes)
- **Breaking Changes**: Limited to `BaseModelController` and its children (sklearn, tensorflow)
- **Public API Stability**: `execute()` signature unchanged
- **Internal API Changes**: `launch_training()`, `train()`, helper methods can change signatures

---

## Refactoring Roadmap

### Phase 1: Extract Core Components (Week 1)

**Goal**: Break down `launch_training()` into 6 testable components

#### 1.1 Create Component Infrastructure

**Directory Structure:**
```
nirs4all/controllers/models/
├── components/
│   ├── __init__.py
│   ├── identifier_generator.py    # NEW
│   ├── prediction_transformer.py  # NEW
│   ├── prediction_assembler.py    # NEW
│   ├── model_loader.py             # NEW
│   ├── score_calculator.py         # NEW
│   └── index_normalizer.py         # NEW
├── base_model.py                   # REFACTOR
├── helper.py                       # REFACTOR
├── sklearn_model.py                # UPDATE
└── tensorflow_model.py             # UPDATE
```

#### 1.2 Component: ModelIdentifierGenerator

**Purpose**: Generate all model identifiers in one place

**Location**: `nirs4all/controllers/models/components/identifier_generator.py`

**Signature:**
```python
class ModelIdentifierGenerator:
    """Generates consistent model identifiers for training and persistence."""

    def generate(
        self,
        model_config: Dict[str, Any],
        runner: 'PipelineRunner',
        context: Dict[str, Any],
        fold_idx: Optional[int] = None
    ) -> ModelIdentifiers:
        """Generate all identifiers for a model.

        Returns:
            ModelIdentifiers: dataclass with fields:
                - classname: str
                - name: str
                - model_id: str (name + operation_counter)
                - model_uuid: str (unique in predictions DB)
                - display_name: str (for console output)
        """
```

**Tasks:**
- [ ] Create `ModelIdentifiers` dataclass
- [ ] Move name extraction from `launch_training()` lines 329-345
- [ ] Move operation counter logic
- [ ] Unit tests with various config formats

#### 1.3 Component: PredictionTransformer

**Purpose**: Handle scaling/unscaling of predictions

**Location**: `nirs4all/controllers/models/components/prediction_transformer.py`

**Signature:**
```python
class PredictionTransformer:
    """Transforms predictions between scaled and unscaled spaces."""

    def transform(
        self,
        predictions: np.ndarray,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        from_space: str = "scaled",
        to_space: str = "unscaled"
    ) -> np.ndarray:
        """Transform predictions between spaces.

        Args:
            predictions: Array to transform
            dataset: Dataset with task_type and _targets
            context: Processing context with y processing mode
            from_space: Source space ("scaled" or "unscaled")
            to_space: Target space ("scaled" or "unscaled")
        """
```

**Tasks:**
- [ ] Extract unscaling logic from `launch_training()` lines 427-447
- [ ] Extract duplicate logic from `_create_fold_averages()` lines 534-549
- [ ] Handle classification vs regression automatically
- [ ] Unit tests with mock dataset and targets

#### 1.4 Component: PredictionDataAssembler

**Purpose**: Assemble prediction data dictionaries consistently

**Location**: `nirs4all/controllers/models/components/prediction_assembler.py`

**Signature:**
```python
@dataclass
class PartitionPrediction:
    """Single partition prediction data."""
    partition_name: str
    indices: List[int]
    y_true: np.ndarray
    y_pred: np.ndarray
    score: float

@dataclass
class PredictionRecord:
    """Complete prediction record for storage."""
    model_identifiers: ModelIdentifiers
    dataset_info: Dict[str, Any]
    partitions: List[PartitionPrediction]
    metadata: Dict[str, Any]

class PredictionDataAssembler:
    """Assembles prediction data for storage."""

    def assemble(
        self,
        model_identifiers: ModelIdentifiers,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        partitions: List[PartitionPrediction],
        metadata: Optional[Dict[str, Any]] = None
    ) -> PredictionRecord:
        """Assemble complete prediction record."""

    def to_storage_dict(self, record: PredictionRecord) -> Dict[str, Any]:
        """Convert to dict format expected by Predictions.add_prediction()."""
```

**Tasks:**
- [ ] Create dataclasses for structured data
- [ ] Extract assembly logic from `launch_training()` lines 462-494
- [ ] Extract duplicate logic from `_create_fold_averages()` lines 567-618
- [ ] Unit tests for dict format compliance

#### 1.5 Component: ModelLoader

**Purpose**: Load models from binaries (predict/explain modes)

**Location**: `nirs4all/controllers/models/components/model_loader.py`

**Signature:**
```python
class ModelLoader:
    """Loads models from serialized binaries."""

    def load_from_binaries(
        self,
        loaded_binaries: List[Tuple[str, bytes]],
        model_name: str,
        operation_counter: int,
        fold_idx: Optional[int] = None
    ) -> Tuple[Any, str]:
        """Load model from binary artifacts.

        Returns:
            Tuple of (loaded_model, matched_artifact_name)

        Raises:
            ValueError: If no matching model found in binaries
        """
```

**Tasks:**
- [ ] Extract loading logic from `launch_training()` lines 359-390
- [ ] Move `_load_model_from_binaries()` logic here
- [ ] Handle fold-specific and non-fold naming patterns
- [ ] Unit tests with mock binaries

#### 1.6 Component: ScoreCalculator

**Purpose**: Calculate and format scores consistently

**Location**: `nirs4all/controllers/models/components/score_calculator.py`

**Signature:**
```python
@dataclass
class PartitionScores:
    """Scores for a single partition."""
    partition_name: str
    metric: str
    score: float
    higher_is_better: bool
    detailed_scores: Optional[Dict[str, float]] = None

class ScoreCalculator:
    """Calculates evaluation scores for models."""

    def calculate_partition_scores(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: TaskType,
        partition_name: str = "test",
        calculate_detailed: bool = False
    ) -> PartitionScores:
        """Calculate scores for a partition."""

    def format_score_summary(
        self,
        scores: List[PartitionScores],
        model_name: str,
        verbose: int = 1
    ) -> str:
        """Format scores for console output."""
```

**Tasks:**
- [ ] Extract score calculation from `launch_training()` lines 449-461
- [ ] Move common logic from sklearn/tensorflow controllers
- [ ] Integrate with ModelUtils for metric selection
- [ ] Unit tests for all task types

#### 1.7 Component: IndexNormalizer

**Purpose**: Convert and validate sample indices

**Location**: `nirs4all/controllers/models/components/index_normalizer.py`

**Signature:**
```python
class IndexNormalizer:
    """Normalizes sample indices to consistent format."""

    def normalize(
        self,
        indices: Optional[Union[np.ndarray, List[int]]],
        data_length: int
    ) -> List[int]:
        """Convert indices to normalized list format.

        Args:
            indices: Raw indices (None, numpy array, or list)
            data_length: Expected data length for validation

        Returns:
            List of integer indices (0-based)
        """
```

**Tasks:**
- [ ] Extract from `launch_training()` lines 448-454
- [ ] Add validation for out-of-bounds indices
- [ ] Unit tests with edge cases

#### 1.8 Refactor launch_training()

**New Signature:**
```python
def launch_training(
    self,
    dataset: 'SpectroDataset',
    model_config: Dict[str, Any],
    context: Dict[str, Any],
    runner: 'PipelineRunner',
    data: TrainingData,  # NEW: encapsulates X/y/indices
    mode: ExecutionMode,  # NEW: enum instead of string
    best_params: Optional[Dict[str, Any]] = None,
    loaded_binaries: Optional[List[Tuple[str, bytes]]] = None
) -> TrainingResult:  # NEW: structured result
    """Execute single model training or prediction.

    Orchestrates components to:
    1. Generate identifiers
    2. Load or instantiate model
    3. Train model (if needed)
    4. Generate predictions
    5. Transform predictions
    6. Calculate scores
    7. Assemble prediction record

    Returns:
        TrainingResult: dataclass with trained_model, model_id, scores, predictions
    """
```

**Tasks:**
- [ ] Create `TrainingData` and `TrainingResult` dataclasses
- [ ] Replace inline logic with component calls
- [ ] Reduce from 278 lines to <80 lines
- [ ] Update docstring (Google style)

---

### Phase 2: Refactor Fold Averaging (Week 2)

**Goal**: Separate fold averaging into testable strategies

#### 2.1 Create Averaging Strategy Infrastructure

**Directory Structure:**
```
nirs4all/controllers/models/strategies/
├── __init__.py
├── averaging.py           # NEW
└── execution_modes.py     # NEW (Phase 3)
```

#### 2.2 Strategy: AveragingStrategy

**Location**: `nirs4all/controllers/models/strategies/averaging.py`

**Signature:**
```python
class AveragingStrategy(ABC):
    """Base strategy for fold prediction averaging."""

    @abstractmethod
    def average(
        self,
        fold_predictions: List[FoldPredictionCache],
        fold_scores: List[float],
        task_type: TaskType
    ) -> AveragedPredictions:
        """Average predictions from multiple folds."""

class SimpleAverageStrategy(AveragingStrategy):
    """Arithmetic mean of fold predictions."""

    def average(self, ...) -> AveragedPredictions:
        """Compute arithmetic mean across folds."""

class WeightedAverageStrategy(AveragingStrategy):
    """Score-weighted average of fold predictions."""

    def average(self, ...) -> AveragedPredictions:
        """Compute weighted mean using score-based weights."""
```

**Tasks:**
- [ ] Create `FoldPredictionCache` dataclass (stores predictions from training)
- [ ] Create `AveragedPredictions` dataclass
- [ ] Extract averaging logic from `_create_fold_averages()` lines 534-618
- [ ] **Cache predictions** during `launch_training()` to avoid re-prediction
- [ ] Unit tests for both strategies

#### 2.3 Refactor _create_fold_averages()

**New Signature:**
```python
def _create_fold_averages(
    self,
    dataset: 'SpectroDataset',
    fold_caches: List[FoldPredictionCache],  # NEW: cached predictions
    averaging_strategies: List[AveragingStrategy],  # NEW: injectable
    context: Dict[str, Any],
    runner: 'PipelineRunner'
) -> List[PredictionRecord]:
    """Create averaged predictions using strategies.

    Returns:
        List of PredictionRecords (one per strategy)
    """
```

**Tasks:**
- [ ] Replace inline averaging with strategy pattern
- [ ] Remove re-prediction code (use cached predictions)
- [ ] Reduce from 108 lines to <40 lines
- [ ] Update docstring

---

### Phase 3: Execution Mode Strategies (Week 2-3)

**Goal**: Eliminate scattered mode branching

#### 3.1 Create ExecutionMode Enum

**Location**: `nirs4all/controllers/models/strategies/execution_modes.py`

**Signature:**
```python
class ExecutionMode(Enum):
    """Execution modes for model controller."""
    TRAIN = "train"
    PREDICT = "predict"
    EXPLAIN = "explain"
    FINETUNE = "finetune"

@dataclass
class ModeContext:
    """Context for mode-specific execution."""
    mode: ExecutionMode
    dataset: 'SpectroDataset'
    runner: 'PipelineRunner'
    loaded_binaries: Optional[List[Tuple[str, bytes]]] = None

class ExecutionModeStrategy(ABC):
    """Strategy for handling execution mode."""

    @abstractmethod
    def should_train(self) -> bool:
        """Whether this mode trains a model."""

    @abstractmethod
    def get_model(
        self,
        controller: 'BaseModelController',
        model_config: Dict[str, Any],
        context: ModeContext,
        best_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Get model instance for this mode."""

    @abstractmethod
    def post_prediction_hook(
        self,
        model: Any,
        controller: 'BaseModelController',
        context: ModeContext
    ) -> None:
        """Execute mode-specific post-prediction logic."""

class TrainModeStrategy(ExecutionModeStrategy):
    """Normal training mode."""

class PredictModeStrategy(ExecutionModeStrategy):
    """Prediction-only mode (load from binaries)."""

class ExplainModeStrategy(ExecutionModeStrategy):
    """Explanation mode (SHAP integration)."""

class FinetuneModeStrategy(ExecutionModeStrategy):
    """Finetuning mode (with best_params)."""
```

**Tasks:**
- [ ] Create mode enum and strategies
- [ ] Replace mode string checks with strategy pattern
- [ ] Move explain mode capture from `launch_training()` lines 374-379
- [ ] Unit tests for each strategy

#### 3.2 Update launch_training() to Use Strategies

**Tasks:**
- [ ] Replace mode branching with strategy calls
- [ ] Inject strategy based on ExecutionMode enum
- [ ] Remove scattered `if mode == "predict"` checks

---

### Phase 4: Config Extraction Refactor (Week 3)

**Goal**: Clean up complex config extraction logic

#### 4.1 Create ModelConfigParser

**Location**: `nirs4all/controllers/models/components/config_parser.py`

**Signature:**
```python
@dataclass
class CanonicalModelConfig:
    """Canonical model configuration format."""
    model_instance: Any  # Model class, function, or instance
    name: Optional[str] = None
    model_params: Dict[str, Any] = field(default_factory=dict)
    train_params: Dict[str, Any] = field(default_factory=dict)
    finetune_params: Optional[Dict[str, Any]] = None

class ConfigParser(ABC):
    """Base parser for model configurations."""

    @abstractmethod
    def can_parse(self, step: Any, operator: Any) -> bool:
        """Check if this parser can handle the config."""

    @abstractmethod
    def parse(self, step: Any, operator: Any) -> CanonicalModelConfig:
        """Parse config into canonical format."""

class OperatorConfigParser(ConfigParser):
    """Parses configs with operator parameter."""

class DictConfigParser(ConfigParser):
    """Parses dict-based configs."""

class SerializedConfigParser(ConfigParser):
    """Parses serialized configs (function/class/import keys)."""

class InstanceConfigParser(ConfigParser):
    """Parses direct model instances."""

class ModelConfigParserChain:
    """Chain of responsibility for config parsing."""

    def __init__(self, parsers: List[ConfigParser]):
        self.parsers = parsers

    def parse(self, step: Any, operator: Any) -> CanonicalModelConfig:
        """Parse using first matching parser."""
```

**Tasks:**
- [ ] Create parser chain with 4 concrete parsers
- [ ] Replace `_extract_model_config()` logic (46 lines → chain)
- [ ] Remove commented DEBUG code
- [ ] Add validation layer
- [ ] Unit tests for all config formats

#### 4.2 Update Base Model to Use Parser

**Tasks:**
- [ ] Replace `_extract_model_config()` with parser chain
- [ ] Update tensorflow_model.py to use same parser
- [ ] Reduce code duplication between controllers

---

### Phase 5: Extract Common Framework Patterns (Week 4)

**Goal**: Reduce duplication between sklearn and tensorflow controllers

#### 5.1 Extract Common Scoring Logic

**New Base Class Method:**
```python
class BaseModelController:
    def _calculate_and_log_training_scores(
        self,
        model: Any,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any],
        y_val: Optional[Any],
        task_type: TaskType,
        verbose: int = 1
    ) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
        """Calculate and log training/validation scores.

        Common implementation used by sklearn and tensorflow controllers.

        Returns:
            Tuple of (train_scores, val_scores)
        """
```

**Tasks:**
- [ ] Extract common logic from sklearn_model.py lines 135-147
- [ ] Extract common logic from tensorflow_model.py lines 263-306
- [ ] Update both controllers to call base method
- [ ] Unit tests in base controller tests

#### 5.2 Create TensorFlow CallbackConfigBuilder

**Location**: `nirs4all/controllers/models/components/tf_callbacks.py`

**Signature:**
```python
class CallbackConfigBuilder:
    """Builds TensorFlow callback configurations."""

    def build_callbacks(
        self,
        train_params: Dict[str, Any],
        existing_callbacks: List[Any],
        verbose: int = 0
    ) -> List[Any]:
        """Build complete callback list.

        Handles:
        - Early stopping
        - Cyclic learning rate
        - Reduce LR on plateau
        - Best model memory
        """

    def build_early_stopping(self, params: Dict[str, Any]) -> Any:
        """Build early stopping callback."""

    def build_cyclic_lr(self, params: Dict[str, Any]) -> Any:
        """Build cyclic learning rate callback."""

    def build_reduce_lr_on_plateau(self, params: Dict[str, Any]) -> Any:
        """Build reduce LR callback."""

    def build_best_model_memory(self, verbose: bool) -> Any:
        """Build best model memory callback."""
```

**Tasks:**
- [ ] Extract callback logic from `_configure_callbacks()` (75 lines)
- [ ] Create builder with 4 focused methods
- [ ] Unit tests for each callback type
- [ ] Update tensorflow_model.py to use builder

---

### Phase 6: Test Suite Implementation (Week 4-5)

**Goal**: Achieve >90% test coverage

#### 6.1 Create Test Structure

**Directory Structure:**
```
tests/controllers/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures
│   ├── test_base_model.py             # NEW
│   ├── test_sklearn_model.py          # NEW
│   ├── test_tensorflow_model.py       # NEW
│   ├── components/
│   │   ├── test_identifier_generator.py   # NEW
│   │   ├── test_prediction_transformer.py # NEW
│   │   ├── test_prediction_assembler.py   # NEW
│   │   ├── test_model_loader.py           # NEW
│   │   ├── test_score_calculator.py       # NEW
│   │   └── test_index_normalizer.py       # NEW
│   └── strategies/
│       ├── test_averaging.py              # NEW
│       └── test_execution_modes.py        # NEW
```

#### 6.2 Core Test Fixtures

**Location**: `tests/controllers/models/conftest.py`

**Fixtures:**
```python
@pytest.fixture
def mock_dataset():
    """Mock SpectroDataset for testing."""

@pytest.fixture
def mock_runner():
    """Mock PipelineRunner for testing."""

@pytest.fixture
def mock_predictions():
    """Mock Predictions storage for testing."""

@pytest.fixture
def sample_model_config():
    """Sample model configuration dict."""

@pytest.fixture
def sample_training_data():
    """Sample X/y training data."""
```

**Tasks:**
- [ ] Create shared fixtures
- [ ] Parametrize for regression/classification tasks
- [ ] Parametrize for sklearn/tensorflow models

#### 6.3 Unit Tests - Components

**Test Coverage per Component:**

1. **ModelIdentifierGenerator** (test_identifier_generator.py)
   - [ ] Test name extraction from dict configs
   - [ ] Test name extraction from instances
   - [ ] Test name extraction from functions
   - [ ] Test operation counter integration
   - [ ] Test fold-specific naming
   - [ ] Test UUID generation uniqueness

2. **PredictionTransformer** (test_prediction_transformer.py)
   - [ ] Test classification prediction unscaling
   - [ ] Test regression prediction unscaling
   - [ ] Test scaled → unscaled transformation
   - [ ] Test unscaled → scaled transformation
   - [ ] Test handling of None values
   - [ ] Test shape preservation

3. **PredictionDataAssembler** (test_prediction_assembler.py)
   - [ ] Test record assembly with all partitions
   - [ ] Test record assembly with missing partitions
   - [ ] Test storage dict format compliance
   - [ ] Test metadata attachment
   - [ ] Test dataclass validation

4. **ModelLoader** (test_model_loader.py)
   - [ ] Test loading from fold-specific binaries
   - [ ] Test loading from non-fold binaries
   - [ ] Test error handling for missing models
   - [ ] Test multiple matching models
   - [ ] Test sklearn model deserialization
   - [ ] Test tensorflow model deserialization

5. **ScoreCalculator** (test_score_calculator.py)
   - [ ] Test regression metric calculation (MSE, MAE, R²)
   - [ ] Test binary classification metrics (accuracy, F1)
   - [ ] Test multiclass classification metrics
   - [ ] Test score formatting for console
   - [ ] Test detailed vs summary scores
   - [ ] Test metric selection per task type

6. **IndexNormalizer** (test_index_normalizer.py)
   - [ ] Test None → full range conversion
   - [ ] Test numpy array → list conversion
   - [ ] Test list → list passthrough
   - [ ] Test out-of-bounds detection
   - [ ] Test negative index handling
   - [ ] Test validation errors

**Total Component Tests:** ~40 tests

#### 6.4 Unit Tests - Strategies

**Averaging Strategies** (test_averaging.py):
- [ ] Test SimpleAverageStrategy with 3 folds
- [ ] Test SimpleAverageStrategy with 10 folds
- [ ] Test WeightedAverageStrategy with varying scores
- [ ] Test weight normalization (sum to 1.0)
- [ ] Test edge case: single fold
- [ ] Test edge case: identical predictions

**Execution Mode Strategies** (test_execution_modes.py):
- [ ] Test TrainModeStrategy.get_model()
- [ ] Test PredictModeStrategy.get_model() with binaries
- [ ] Test ExplainModeStrategy post-prediction hook
- [ ] Test FinetuneModeStrategy with best_params
- [ ] Test mode-specific training decisions

**Total Strategy Tests:** ~15 tests

#### 6.5 Unit Tests - Controllers

**BaseModelController** (test_base_model.py):
- [ ] Test execute() with train mode
- [ ] Test execute() with predict mode
- [ ] Test execute() with explain mode
- [ ] Test execute() with finetune mode
- [ ] Test train() without folds
- [ ] Test train() with 5 folds
- [ ] Test finetune() delegation to OptunaManager
- [ ] Test launch_training() orchestration
- [ ] Test _persist_model() serialization
- [ ] Test get_xy() data extraction
- [ ] Test abstract method requirements

**SklearnModelController** (test_sklearn_model.py):
- [ ] Test matches() with BaseEstimator
- [ ] Test _get_model_instance() with dict config
- [ ] Test _get_model_instance() with force_params
- [ ] Test _train_model() parameter filtering
- [ ] Test _predict_model() shape handling
- [ ] Test _prepare_data() 2D formatting
- [ ] Test _evaluate_model() cross-validation
- [ ] Test end-to-end training with sklearn model

**TensorFlowModelController** (test_tensorflow_model.py):
- [ ] Test matches() with keras.Model
- [ ] Test matches() with @framework('tensorflow')
- [ ] Test _get_model_instance() with function
- [ ] Test _get_model_instance() with class
- [ ] Test _create_model_from_function() signature handling
- [ ] Test _train_model() callback configuration
- [ ] Test _prepare_data() 3D tensor reshaping
- [ ] Test _configure_callbacks() early stopping
- [ ] Test _configure_callbacks() cyclic LR
- [ ] Test _configure_optimizer() learning rate
- [ ] Test end-to-end training with keras model

**Total Controller Tests:** ~30 tests

#### 6.6 Integration Tests

**End-to-End Workflows** (test_integration.py):
- [ ] Test complete training pipeline with sklearn model
- [ ] Test complete training pipeline with tensorflow model
- [ ] Test fold-based training with averaging
- [ ] Test hyperparameter optimization with Optuna
- [ ] Test prediction mode with loaded binaries
- [ ] Test explain mode with SHAP integration

**Total Integration Tests:** ~10 tests

#### 6.7 Property-Based Tests

**Invariants** (test_properties.py):
- [ ] Config extraction is idempotent
- [ ] Prediction shapes match sample counts
- [ ] Fold weights sum to 1.0
- [ ] Score calculations are deterministic
- [ ] Index normalization preserves order

**Total Property Tests:** ~8 tests

**Grand Total: ~100+ tests**

---

### Phase 7: Documentation Update (Week 5)

**Goal**: Complete Google-style docstrings for all methods

#### 7.1 Base Model Controller Documentation

**Methods Needing Docstrings:**
- [ ] `train()` - Add Args, Returns, Raises sections
- [ ] `launch_training()` - Complete rewrite for new signature
- [ ] `_create_fold_averages()` - Add full documentation
- [ ] `_add_all_predictions()` - Add full documentation
- [ ] `_extract_model_config()` - Document all supported formats

**Example Template:**
```python
def train(
    self,
    dataset: 'SpectroDataset',
    model_config: Dict[str, Any],
    context: Dict[str, Any],
    runner: 'PipelineRunner',
    prediction_store: 'Predictions',
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_train_unscaled: np.ndarray,
    y_test_unscaled: np.ndarray,
    folds: List[Tuple],
    best_params: Optional[Dict[str, Any]] = None,
    loaded_binaries: Optional[List[Tuple[str, bytes]]] = None,
    mode: str = "train"
) -> List['ArtifactMeta']:
    """Train model(s) and manage fold-based training.

    Orchestrates single-fold or multi-fold training, manages model
    persistence, and coordinates fold averaging strategies.

    Args:
        dataset: Dataset containing features and targets
        model_config: Model configuration dict with model instance and params
        context: Pipeline execution context with step_id and processing info
        runner: Pipeline runner for artifact persistence
        prediction_store: External prediction storage manager
        X_train: Training features (all samples)
        y_train: Training targets (all samples, scaled)
        X_test: Test features
        y_test: Test targets (scaled)
        y_train_unscaled: Training targets (original scale)
        y_test_unscaled: Test targets (original scale)
        folds: List of (train_indices, val_indices) tuples for CV
        best_params: Optional hyperparameters from finetuning
        loaded_binaries: Optional pre-loaded models for prediction mode
        mode: Execution mode ("train", "predict", "explain", "finetune")

    Returns:
        List of ArtifactMeta objects for persisted model binaries

    Raises:
        ValueError: If mode is invalid or required data is missing
        RuntimeError: If model training or persistence fails

    Examples:
        >>> # Single-fold training
        >>> binaries = controller.train(
        ...     dataset, model_config, context, runner, predictions,
        ...     X_train, y_train, X_test, y_test,
        ...     y_train_unscaled, y_test_unscaled,
        ...     folds=[]
        ... )

        >>> # Multi-fold training with 5 folds
        >>> binaries = controller.train(
        ...     dataset, model_config, context, runner, predictions,
        ...     X_train, y_train, X_test, y_test,
        ...     y_train_unscaled, y_test_unscaled,
        ...     folds=[(train_idx, val_idx) for fold in range(5)]
        ... )
    """
```

#### 7.2 TensorFlow Controller Documentation

**Methods Needing Docstrings:**
- [ ] `_prepare_compilation_config()`
- [ ] `_configure_optimizer()`
- [ ] `_prepare_fit_config()`
- [ ] `_configure_callbacks()` - Document all callback types
- [ ] `_log_training_config()`

#### 7.3 Helper Documentation

**Methods Needing Examples:**
- [ ] `extract_core_name()` - Add examples for each config format
- [ ] `clone_model()` - Document framework-specific behavior
- [ ] `create_model_identifiers()` - Add full docstring

#### 7.4 Module-Level Documentation

**Tasks:**
- [ ] Add architecture overview to base_model.py module docstring
- [ ] Add usage examples to base_model.py
- [ ] Document execution flow (execute → train → launch_training)
- [ ] Document mode system

---

### Phase 8: Performance Optimizations (Week 6)

**Goal**: Eliminate performance bottlenecks

#### 8.1 Cache Predictions During Training

**Current Problem:**
```python
# base_model.py:534-545 - Re-predicts for averaging!
for fold_model_tuple in folds_models:
    _, fold_model, _ = fold_model_tuple
    fold_train_preds = self._predict_model(fold_model, X_train)  # SLOW
```

**Solution:**
```python
@dataclass
class TrainingResult:
    trained_model: Any
    model_id: str
    scores: Dict[str, float]
    cached_predictions: CachedPredictions  # NEW: cache predictions here

@dataclass
class CachedPredictions:
    train_pred: np.ndarray
    val_pred: np.ndarray
    test_pred: np.ndarray
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
```

**Tasks:**
- [ ] Modify `launch_training()` to return cached predictions
- [ ] Update `_create_fold_averages()` to use cached predictions
- [ ] Measure speedup (expect 20-40% improvement for 10 folds)

#### 8.2 Stream Predictions to Storage

**Current Problem:**
```python
# Keeps all predictions in memory until the end
all_predictions = [pred1, pred2, ..., pred10]
self._add_all_predictions(prediction_store, all_predictions, weights)
```

**Solution:**
```python
# Stream predictions incrementally
for fold_idx, result in enumerate(fold_results):
    prediction_store.add_prediction(result.prediction_record)
```

**Tasks:**
- [ ] Modify `train()` to stream predictions during fold loop
- [ ] Remove batch storage at end
- [ ] Reduce memory footprint

#### 8.3 Avoid Keeping Fold Models in Memory

**Current Problem:**
```python
folds_models = []  # List of (model_id, model, score)
for fold in folds:
    model, model_id, score, ... = launch_training(...)
    folds_models.append((model_id, model, score))  # MEMORY LEAK
```

**Solution:**
```python
# Don't keep models, only predictions and scores
fold_results = []  # List of TrainingResult (with cached predictions)
for fold in folds:
    result = launch_training(...)
    fold_results.append(result)
    # Model can be garbage collected after persistence
```

**Tasks:**
- [ ] Modify train() to not accumulate models
- [ ] Use TrainingResult with cached predictions instead
- [ ] Measure memory reduction

---

### Phase 9: Code Cleanup (Week 6)

**Goal**: Remove technical debt

#### 9.1 Remove Commented DEBUG Code

**Files to Clean:**
- [ ] base_model.py lines 128-132, 392, 554-557, 563-592
- [ ] tensorflow_model.py lines 114-119
- [ ] sklearn_model.py (check for commented code)

#### 9.2 Remove TODO Comments

**TODOs to Address:**
- [ ] base_model.py:392 - "RETRIEVE THE HOLD METHOD" → Implement or remove
- [ ] base_model.py:554 - "prediction CSV" → Implement or remove

#### 9.3 Standardize Error Handling

**Tasks:**
- [ ] Consistent exception types across controllers
- [ ] Clear error messages with context
- [ ] Proper error propagation to pipeline level

---

## API Stability Guarantees

### Public API (NO BREAKING CHANGES)

**Preserved Signatures:**
```python
# nirs4all/controllers/models/base_model.py
class BaseModelController(OperatorController, ABC):
    def execute(
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, bytes]]] = None,
        prediction_store: 'Predictions' = None
    ) -> Tuple[Dict[str, Any], List['ArtifactMeta']]:
        """UNCHANGED - Public API"""
```

**Rationale**: Pipeline depends on this signature

### Internal API (BREAKING CHANGES ALLOWED)

**Changed Signatures:**
```python
# nirs4all/controllers/models/base_model.py
class BaseModelController(OperatorController, ABC):
    def train(...) -> List['ArtifactMeta']:
        """CHANGED - Internal only, called by execute()"""

    def launch_training(...) -> TrainingResult:
        """CHANGED - Internal only, called by train()"""

    def _create_fold_averages(...) -> List[PredictionRecord]:
        """CHANGED - Internal only, called by train()"""
```

**Rationale**: These are internal methods only called within controller hierarchy

### Framework Controller API (BREAKING CHANGES ALLOWED)

**Changed Abstract Methods:**
```python
# Subclasses (sklearn_model.py, tensorflow_model.py) can handle signature changes
class SklearnModelController(BaseModelController):
    def _train_model(...) -> Any:
        """CHANGED - Framework-specific implementation"""
```

**Rationale**: Only 2 active implementations (sklearn, tensorflow), easy to update

---

## Migration Guide

### For Users (No Changes Required)

Pipeline code remains unchanged:
```python
# This continues to work as-is
pipeline.run(dataset)
```

### For Framework Controller Developers

If implementing new controller (e.g., PyTorchModelController):

**Before (0.4.1):**
```python
class MyController(BaseModelController):
    def _train_model(self, model, X_train, y_train, X_val, y_val, **kwargs):
        # Implementation
```

**After (0.5.0):**
```python
class MyController(BaseModelController):
    def _train_model(self, model, X_train, y_train, X_val, y_val, train_params):
        # Implementation - no **kwargs, explicit train_params dict
```

**Changes:**
1. Use explicit `train_params` dict instead of `**kwargs`
2. Import new dataclasses: `TrainingData`, `TrainingResult`
3. Return structured `TrainingResult` instead of model

---

## Success Criteria

### Code Quality
- [ ] `launch_training()` reduced from 278 lines to <80 lines
- [ ] All methods have Google-style docstrings
- [ ] No commented DEBUG code
- [ ] No unresolved TODO comments
- [ ] Pylint score >9.0

### Test Coverage
- [ ] >90% line coverage for base_model.py
- [ ] >90% line coverage for sklearn_model.py
- [ ] >90% line coverage for tensorflow_model.py
- [ ] >85% line coverage for new components
- [ ] All integration tests passing

### Performance
- [ ] Fold-based training 20-40% faster (cached predictions)
- [ ] Memory usage reduced by 50% (no fold model retention)
- [ ] No performance regression in single-fold training

### Documentation
- [ ] All public methods documented with examples
- [ ] Architecture diagram added to module docstring
- [ ] Migration guide complete
- [ ] CHANGELOG.md updated with breaking changes

---

## Risk Assessment

### High Risk
- **Signature changes in launch_training()**: Mitigated by comprehensive tests
- **Fold averaging refactor**: Mitigated by property-based tests (weights sum to 1.0)

### Medium Risk
- **Config extraction refactor**: Mitigated by extensive format testing
- **Mode strategy pattern**: Mitigated by integration tests covering all modes

### Low Risk
- **Component extraction**: Low risk, isolated changes
- **Documentation**: Zero risk, additive only

---

## Timeline

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| Phase 1: Component Extraction | 1 week | 6 components + refactored launch_training() |
| Phase 2: Fold Averaging | 1 week | AveragingStrategy + cached predictions |
| Phase 3: Mode Strategies | 1 week | ExecutionMode enum + 4 strategies |
| Phase 4: Config Parser | 1 week | ModelConfigParserChain |
| Phase 5: Framework Patterns | 1 week | Common scoring + CallbackConfigBuilder |
| Phase 6: Test Suite | 1-2 weeks | 100+ tests, >90% coverage |
| Phase 7: Documentation | 1 week | Complete docstrings |
| Phase 8: Performance | 1 week | Caching optimizations |
| Phase 9: Cleanup | 1 week | Remove technical debt |

**Total: 9-10 weeks**

---

## Approval Checklist

- [ ] Architecture approved
- [ ] Component structure approved
- [ ] Test strategy approved
- [ ] Performance goals approved
- [ ] Documentation standards approved
- [ ] Timeline approved
- [ ] Version bump (0.5.0) approved

---

## Next Steps

1. **Review this proposal** - Provide feedback on approach
2. **Approve timeline** - Confirm 9-10 week timeline acceptable
3. **Assign priority** - Determine if this blocks other work
4. **Create branch** - `feature/base-model-refactor-0.5.0`
5. **Start Phase 1** - Begin component extraction

---

## References

- [Refactoring Catalog - Martin Fowler](https://refactoring.com/catalog/)
- [Clean Architecture - Robert C. Martin](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Strategy Pattern - Gang of Four](https://en.wikipedia.org/wiki/Strategy_pattern)
- [Chain of Responsibility Pattern](https://refactoring.guru/design-patterns/chain-of-responsibility)

---

**Document Version:** 1.0
**Last Updated:** October 30, 2025
**Author:** GitHub Copilot
**Status:** Awaiting Approval
