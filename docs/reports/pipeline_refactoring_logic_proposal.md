# Pipeline Refactoring Logic Proposal
**Date**: October 30, 2025
**Author**: GitHub Copilot Analysis
**Version**: 1.0 - Initial Proposal

---

## Executive Summary

This document proposes a comprehensive refactoring of nirs4all's `PipelineRunner` to achieve three interconnected goals:

1. **Separation of Concerns**: Split orchestration (many pipelines) from execution (single pipeline)
2. **ML-Standard APIs**: Expose `fit`, `transform`, `predict` interfaces at both levels
3. **Optuna Integration**: Enable hyperparameter optimization at both meta-pipeline and sub-pipeline levels

The proposed architecture preserves all existing functionality while establishing a cleaner foundation that aligns with scikit-learn conventions and anticipates future needs (single-file pipelines, stacking, transfer learning).

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Core Problem Statement](#2-core-problem-statement)
3. [Proposed Architecture](#3-proposed-architecture)
4. [Design Decisions and Alternatives](#4-design-decisions-and-alternatives)
5. [API Evolution Strategy](#5-api-evolution-strategy)
6. [Optuna Integration Strategy](#6-optuna-integration-strategy)
7. [Implementation Phases](#7-implementation-phases)
8. [Open Questions](#8-open-questions)

---

## 1. Current State Analysis

### 1.1 What Works Well ✅

**Strengths to preserve:**
- Powerful configuration system (JSON/YAML pipelines)
- Multi-source dataset support (unique to NIRS domain)
- Binary serialization with manifest system
- Comprehensive prediction tracking
- Controller-based architecture (extensible)
- Cross-framework support (sklearn, TF, PyTorch)
- Rich preprocessing operators for spectroscopy

**Key observation**: The *internal logic* of step execution, controller selection, and data flow is fundamentally sound. The refactoring should preserve this.

### 1.2 Core Issues ❌

**Problem 1: Conflated Responsibilities**
```python
# PipelineRunner does EVERYTHING:
runner = PipelineRunner(verbose=1)
predictions, datasets = runner.run(pipeline_config, dataset_config)
```

This single class handles:
- Input normalization (arrays, configs, datasets)
- Pipeline generation (expanding `_or_` combinations)
- Multi-dataset iteration
- Multi-pipeline orchestration
- Single pipeline execution
- Result aggregation
- File I/O and serialization
- Workspace management
- Prediction/explain modes

**Consequence**: Cannot treat a single pipeline as a reusable object.

**Problem 2: No ML-Standard API**
```python
# Cannot do this today:
pipe = Pipeline([...])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Or compose:
ensemble = VotingRegressor([('nirs1', pipe1), ('nirs2', pipe2)])
```

**Consequence**: Isolated from sklearn ecosystem, cannot use with:
- sklearn's cross-validation tools
- Model selection utilities
- SHAP/eli5 explainability
- Ensemble methods
- Hyperparameter search (GridSearch, etc.)

**Problem 3: Optuna Integration Limitations**

Currently:
```python
# Can optimize a single model's hyperparameters
{"model": PLSRegression(), "finetune_params": {"n_components": [5, 10, 15]}}

# CANNOT optimize across:
# - Multiple preprocessing strategies
# - Model selection (PLS vs RF vs MLP)
# - Pipeline structure
# - Meta-parameters (CV strategy, augmentation count)
```

**Consequence**: Limited to per-model hyperparameter tuning, cannot explore broader search spaces.

---

## 2. Core Problem Statement

### 2.1 Two Levels of Abstraction Needed

**Level 1: Single Pipeline** (like sklearn's `Pipeline`)
- One sequence of steps: `[scaler, preprocessor, cv_splitter, model]`
- Operates on one dataset
- Has standard ML API: `fit(X, y)`, `predict(X)`, `transform(X)`
- Serializable as single object
- Can be optimized with Optuna

**Level 2: Pipeline Orchestrator** (like sklearn's `ParameterGrid` + cross-validation)
- Manages many Level 1 pipelines
- Handles multiple datasets
- Expands generation operators (`_or_`, `_range_`)
- Aggregates results across runs
- Provides analysis and ranking

### 2.2 Desired User Interfaces

**Scenario A: Simple single pipeline (sklearn-like)**
```python
from nirs4all.pipeline import Pipeline

pipe = Pipeline([
    MinMaxScaler(),
    SavitzkyGolay(),
    PLSRegression(n_components=10)
])

# Standard sklearn API
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
score = pipe.score(X_test, y_test)

# Save/load
pipe.save('model.pkl')
loaded = Pipeline.load('model.pkl')
```

**Scenario B: Multi-pipeline orchestration (current nirs4all style)**
```python
from nirs4all.pipeline import PipelineRunner, PipelineConfigs

runner = PipelineRunner()
predictions, datasets = runner.run(
    pipeline=[
        MinMaxScaler(),
        {"feature_augmentation": {"_or_": [SNV, SG, Detrend], "count": 2}},
        ShuffleSplit(n_splits=3),
        {"model": {"_or_": [PLSRegression(10), RandomForest(50)]}}
    ],
    dataset='sample_data/regression'
)

# Analysis
top_5 = predictions.top(5, rank_metric='rmse')
```

**Scenario C: Optuna optimization of single pipeline**
```python
from nirs4all.pipeline import Pipeline
from nirs4all.optimization import OptunaSearch

pipe = Pipeline([
    MinMaxScaler(),
    SavitzkyGolay(),
    PLSRegression()  # No params set
])

search = OptunaSearch(
    pipe,
    param_space={
        'PLSRegression__n_components': [5, 10, 15, 20],
        'SavitzkyGolay__window_length': (5, 21),
        'SavitzkyGolay__polyorder': [2, 3, 4]
    },
    cv=ShuffleSplit(n_splits=5),
    n_trials=100
)

search.fit(X_train, y_train)
best_pipe = search.best_estimator_
```

**Scenario D: Optuna optimization of meta-pipeline**
```python
from nirs4all.pipeline import PipelineRunner
from nirs4all.optimization import MetaOptunaSearch

runner = PipelineRunner()
search = MetaOptunaSearch(
    runner,
    pipeline_space={
        'preprocessing': {'_or_': [SNV, MSC, Detrend, SG]},
        'augmentation_count': [1, 2, 3],
        'model': {'_or_': [PLSRegression, RandomForest, MLPRegressor]},
        'model_params': {
            'PLSRegression__n_components': [5, 10, 15],
            'RandomForest__n_estimators': [50, 100, 200],
            'MLPRegressor__hidden_layer_sizes': [(20,), (50,), (20, 20)]
        }
    },
    objective='best_test_rmse',  # or 'mean_test_rmse', 'best_val_r2'
    n_trials=200
)

search.fit('sample_data/regression')
best_config = search.best_pipeline_config_
best_predictions = search.best_predictions_
```

---

## 3. Proposed Architecture

### 3.1 High-Level Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                      PipelineRunner                             │
│               (Level 2: Orchestrator)                           │
│                                                                 │
│  • Manages multiple pipelines and datasets                     │
│  • Expands generation operators (_or_, _range_)                │
│  • Aggregates results                                           │
│  • Provides ranking and analysis                               │
│  • Workspace and file management                               │
│                                                                 │
│  Public API:                                                    │
│    - run(pipeline, dataset, **kwargs)                          │
│    - predict(model_id, dataset)                                │
│    - explain(model_id, dataset)                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ creates and manages
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Pipeline                                 │
│               (Level 1: Single Pipeline)                        │
│                                                                 │
│  • One sequence of steps                                        │
│  • Operates on one dataset                                      │
│  • Stateful (fitted components)                                │
│  • Serializable                                                │
│                                                                 │
│  Public API (sklearn-compatible):                              │
│    - fit(X, y)                                                 │
│    - predict(X)                                                │
│    - transform(X)                                              │
│    - score(X, y)                                               │
│    - get_params() / set_params()                               │
│    - save() / load()                                           │
│                                                                 │
│  Internal (delegates to):                                      │
│    - PipelineExecutor (step-by-step execution)                 │
│    - BinaryManager (fitted component storage)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ uses
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     PipelineExecutor                            │
│              (Internal: Step Execution Logic)                   │
│                                                                 │
│  • Current runner logic (run_steps, run_step)                  │
│  • Controller selection and execution                          │
│  • Context management                                          │
│  • Binary loading/saving                                       │
│                                                                 │
│  NOT exposed as public API                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Class Responsibilities

#### **Pipeline** (NEW)
**Role**: Single, reusable, sklearn-compatible pipeline object

**Responsibilities**:
- Store step definitions
- Implement sklearn BaseEstimator interface
- Manage internal dataset state
- Delegate execution to PipelineExecutor
- Handle fitted component persistence
- Provide serialization (save/load)

**Key attributes**:
```python
class Pipeline(BaseEstimator):
    steps: List[Any]              # Step definitions
    _fitted: bool                 # Fit state
    _dataset: SpectroDataset      # Internal dataset (created from X, y)
    _context: Dict[str, Any]      # Execution context
    _executor: PipelineExecutor   # Execution engine
    _binaries: BinaryManager      # Fitted components
    name: str                     # Optional identifier
```

#### **PipelineRunner** (REFACTORED)
**Role**: Orchestrate multiple pipelines and datasets

**Responsibilities**:
- Expand PipelineConfigs (generation operators)
- Iterate over datasets
- Create and manage Pipeline instances
- Aggregate Predictions
- Workspace management
- Provide analysis/ranking methods

**Key changes**:
```python
class PipelineRunner:
    # KEEP existing __init__ parameters
    # KEEP workspace management
    # KEEP prediction/explain modes

    # CHANGE: run() creates Pipeline instances internally
    def run(self, pipeline, dataset, **kwargs):
        # 1. Normalize inputs
        pipeline_configs = self._normalize_pipeline(pipeline)
        dataset_configs = self._normalize_dataset(dataset)

        # 2. For each combination:
        for steps in pipeline_configs.steps:
            for config, name in dataset_configs.configs:
                # Create Pipeline instance
                pipe = Pipeline(steps, name=f"{config_name}_{name}")

                # Get dataset
                ds = dataset_configs.get_dataset(config, name)

                # Fit pipeline (delegates to executor)
                pipe.fit(ds.x({'partition': 'train'}), ds.y({'partition': 'train'}))

                # Evaluate and store predictions
                # ...
```

#### **PipelineExecutor** (NEW, extracted from runner)
**Role**: Internal execution engine

**Responsibilities**:
- Execute steps sequentially/parallel
- Select controllers
- Manage context
- Handle binary loading/saving
- Mode switching (train/predict/explain)

**Key methods**:
```python
class PipelineExecutor:
    def execute_steps(
        self,
        steps: List[Any],
        dataset: SpectroDataset,
        context: Dict[str, Any],
        mode: str = "train",
        binaries: Optional[BinaryManager] = None
    ) -> Tuple[SpectroDataset, Dict[str, Any]]:
        # Current run_steps logic
        pass

    def execute_step(self, step, dataset, context, mode, binaries):
        # Current run_step logic
        pass
```

---

## 4. Design Decisions and Alternatives

### Decision 1: Where to split the abstraction?

**Option A: Two classes (Pipeline + PipelineRunner)** ⭐ RECOMMENDED
- **Pro**: Clear separation, sklearn compatibility, incremental migration
- **Pro**: PipelineRunner can still use current API
- **Pro**: Pipeline can be used standalone
- **Con**: Some duplication (both can execute)

**Option B: Three classes (Pipeline + Orchestrator + Runner)**
- **Pro**: Maximum separation
- **Con**: More complex, harder to migrate
- **Con**: Unclear responsibilities

**Option C: Single class with modes**
- **Pro**: Minimal changes
- **Con**: Continues current problem
- **Con**: Difficult to make sklearn-compatible

**Decision**: **Option A**. Provides best balance of clarity, compatibility, and migration path.

---

### Decision 2: How to handle dataset normalization in Pipeline.fit()?

**Context**: Pipeline.fit(X, y) receives numpy arrays, but internal execution needs SpectroDataset.

**Option A: Create SpectroDataset internally** ⭐ RECOMMENDED
```python
def fit(self, X, y):
    # Create internal dataset from arrays
    self._dataset = SpectroDataset(name=self.name)
    self._dataset.add_samples(X, {"partition": "train"})
    self._dataset.add_targets(y)

    # Execute steps
    self._executor.execute_steps(self.steps, self._dataset, self._context)
```
- **Pro**: Transparent to user, sklearn-compatible
- **Pro**: Internal state encapsulated
- **Con**: May lose metadata (but can be provided separately)

**Option B: Require SpectroDataset input**
```python
def fit(self, dataset: SpectroDataset):
    self._dataset = dataset
    # ...
```
- **Pro**: No conversion needed
- **Con**: Not sklearn-compatible
- **Con**: Defeats purpose of refactoring

**Option C: Support both**
```python
def fit(self, X, y=None):
    if isinstance(X, SpectroDataset):
        self._dataset = X
    else:
        self._dataset = self._create_dataset(X, y)
    # ...
```
- **Pro**: Maximum flexibility
- **Con**: API confusion

**Decision**: **Option A** for Pipeline, **current flexible approach** for PipelineRunner.

---

### Decision 3: How to handle multi-source data in sklearn API?

**Context**: SpectroDataset supports multi-source 3D data, sklearn expects 2D arrays.

**Option A: Flatten multi-source in fit()** ⭐ RECOMMENDED
```python
def fit(self, X, y):
    # If X is 3D, treat as multi-source
    if X.ndim == 3:
        self._dataset.add_samples(X, {"partition": "train"})  # Multi-source
    elif isinstance(X, list):
        # Multiple 2D arrays = multi-source
        self._dataset.add_samples(X, {"partition": "train"})
    else:
        # Single 2D array
        self._dataset.add_samples(X, {"partition": "train"})
```
- **Pro**: Natural extension of sklearn API
- **Pro**: Preserves multi-source capability
- **Con**: Not standard sklearn (but acceptable for domain-specific lib)

**Option B: Separate MultiSourcePipeline class**
- **Pro**: Explicit about multi-source
- **Con**: More classes, API complexity

**Decision**: **Option A**. Document clearly that X can be 3D/list for multi-source.

---

### Decision 4: How to handle prediction storage in Pipeline?

**Context**: Currently, PipelineRunner manages Predictions object externally.

**Option A: Pipeline doesn't store predictions** ⭐ RECOMMENDED
```python
# Pipeline just does fit/predict
pipe.fit(X, y)
y_pred = pipe.predict(X_test)

# PipelineRunner manages prediction storage
runner = PipelineRunner()
predictions, _ = runner.run(pipeline, dataset)
```
- **Pro**: Pipeline stays simple and stateless (except fitted components)
- **Pro**: Matches sklearn behavior
- **Pro**: PipelineRunner handles aggregation

**Option B: Pipeline has internal prediction history**
```python
pipe.fit(X, y)
pipe.predict(X_test)
predictions = pipe.predictions_  # Access history
```
- **Pro**: Self-contained
- **Con**: Stateful, memory intensive
- **Con**: Not sklearn-like

**Decision**: **Option A**. Pipeline is stateless except for fitted components. PipelineRunner aggregates predictions.

---

### Decision 5: How to serialize Pipeline?

**Context**: Need to save/load pipelines as single files.

**Option A: Pickle with metadata** ⭐ RECOMMENDED FOR MVP
```python
# Save
pipe.save('model.pkl')

# Internally:
{
    'steps': self.steps,
    'binaries': self._binaries.to_dict(),
    'context': self._context,
    'metadata': {
        'version': '0.5.0',
        'created_at': ...,
        'task_type': ...
    }
}
```
- **Pro**: Simple, works immediately
- **Pro**: Can load anywhere with nirs4all installed
- **Con**: Python-specific

**Option B: ONNX format**
- **Pro**: Cross-platform, production-ready
- **Con**: Complex, may not support all operators
- **Con**: Significant implementation effort

**Option C: Joblib + manifest (current approach)**
- **Pro**: Handles large arrays efficiently
- **Con**: Multiple files

**Decision**: **Option A** for initial implementation. Can add ONNX export later as separate feature.

---

## 5. API Evolution Strategy

### 5.1 Backward Compatibility Plan

**Phase 1: Coexistence (v0.5.0)**
```python
# NEW API (recommended)
from nirs4all.pipeline import Pipeline

pipe = Pipeline([...])
pipe.fit(X, y)

# OLD API (still works, no warnings)
from nirs4all.pipeline import PipelineRunner, PipelineConfigs

runner = PipelineRunner()
runner.run(pipeline_config, dataset_config)
```

**Phase 2: Deprecation (v0.6.0)**
```python
# OLD API shows deprecation warnings
runner.run(...)  # DeprecationWarning: Use Pipeline for single pipelines
```

**Phase 3: Removal (v1.0.0)**
```python
# OLD API removed, migration guide provided
```

### 5.2 Naming Strategy

**Option A: New class name** ⭐ RECOMMENDED
- **OLD**: `PipelineRunner` (orchestrator)
- **NEW**: `Pipeline` (single pipeline)

**Option B: Keep PipelineRunner, rename old**
- **OLD**: `PipelineOrchestrator` (renamed)
- **NEW**: `PipelineRunner` (single pipeline)
- **Con**: Breaking change

**Option C: Different naming entirely**
- `NirsPipeline` (single)
- `PipelineRunner` (orchestrator)

**Decision**: **Option A**. "Pipeline" is the natural sklearn-like name.

### 5.3 Import Structure

```python
# Recommended imports
from nirs4all.pipeline import Pipeline              # Single pipeline
from nirs4all.pipeline import PipelineRunner        # Orchestrator
from nirs4all.pipeline import PipelineConfigs       # Config object
from nirs4all.optimization import OptunaSearch      # Single-pipe optimization
from nirs4all.optimization import MetaOptunaSearch  # Multi-pipe optimization
```

---

## 6. Optuna Integration Strategy

### 6.1 Current Limitation

Today:
```python
# Can only optimize within a single model step
{
    "model": PLSRegression(),
    "finetune_params": {
        "model_params": {"n_components": [5, 10, 15]},
        "n_trials": 50
    }
}
```

**Cannot optimize**:
- Preprocessing choices (SNV vs MSC vs Detrend)
- Model selection (PLS vs RandomForest vs MLP)
- Pipeline structure
- Cross-validation strategy

### 6.2 Proposed Two-Level Optimization

#### Level 1: Single Pipeline Optimization (NEW)

```python
from nirs4all.pipeline import Pipeline
from nirs4all.optimization import OptunaSearch

pipe = Pipeline([
    MinMaxScaler(),
    SavitzkyGolay(),     # Will be optimized
    PLSRegression()      # Will be optimized
])

search = OptunaSearch(
    pipe,
    param_space={
        # Preprocessing params
        'SavitzkyGolay__window_length': (5, 21),
        'SavitzkyGolay__polyorder': [2, 3, 4],

        # Model params
        'PLSRegression__n_components': [5, 10, 15, 20, 25]
    },
    cv=ShuffleSplit(n_splits=5),
    scoring='neg_root_mean_squared_error',
    n_trials=100,
    verbose=1
)

search.fit(X_train, y_train)
best_pipe = search.best_estimator_
best_score = search.best_score_
best_params = search.best_params_
```

**Implementation approach**:
- Similar to sklearn's `GridSearchCV` / `RandomizedSearchCV`
- Wraps Pipeline with parameter sampling
- Returns fitted Pipeline with best parameters

#### Level 2: Meta-Pipeline Optimization (NEW)

```python
from nirs4all.pipeline import PipelineRunner
from nirs4all.optimization import MetaOptunaSearch

runner = PipelineRunner()

search = MetaOptunaSearch(
    runner,
    pipeline_space={
        'preprocessing': {
            '_or_': [StandardNormalVariate, MultiplicativeScatterCorrection, Detrend]
        },
        'smoothing': {
            '_or_': [
                None,  # No smoothing
                SavitzkyGolay(window_length=5, polyorder=2),
                SavitzkyGolay(window_length=11, polyorder=3),
                Gaussian(sigma=2)
            ]
        },
        'cv': {
            '_or_': [
                ShuffleSplit(n_splits=3),
                KFold(n_splits=5),
                RepeatedKFold(n_splits=3, n_repeats=2)
            ]
        },
        'model': {
            '_or_': [
                {'class': PLSRegression, 'params': {'n_components': [5, 10, 15, 20]}},
                {'class': RandomForestRegressor, 'params': {'n_estimators': [50, 100, 200]}},
                {'class': MLPRegressor, 'params': {'hidden_layer_sizes': [(20,), (50,), (20,20)]}}
            ]
        }
    },
    objective='minimize',  # or 'maximize'
    metric='test_rmse',    # or 'val_r2', 'mean_test_rmse'
    aggregation='best',    # or 'mean', 'median', 'robust_mean'
    n_trials=200,
    dataset='sample_data/regression',
    verbose=1
)

search.optimize()
best_config = search.best_pipeline_config_
best_predictions = search.best_predictions_
best_score = search.best_value_

# Access trial history
trials_df = search.trials_dataframe()  # Optuna study as DataFrame
```

**Implementation approach**:
- Treats entire pipeline structure as search space
- Each trial generates a concrete pipeline configuration
- Runs pipeline with PipelineRunner
- Aggregates results based on metric/aggregation strategy
- Returns best configuration

### 6.3 Key Design: Objective Function Interface

**For single pipeline**:
```python
# OptunaSearch internally does:
def objective(trial):
    # Sample parameters
    params = sample_params(trial, param_space)

    # Create pipeline with params
    pipe_copy = clone(pipeline)
    pipe_copy.set_params(**params)

    # Cross-validate
    scores = cross_val_score(pipe_copy, X, y, cv=cv, scoring=scoring)

    # Return metric
    return scores.mean()
```

**For meta-pipeline**:
```python
# MetaOptunaSearch internally does:
def objective(trial):
    # Sample pipeline structure
    pipeline_config = sample_pipeline_structure(trial, pipeline_space)

    # Create and run pipeline
    runner = PipelineRunner()
    predictions, _ = runner.run(pipeline_config, dataset)

    # Aggregate results
    if aggregation == 'best':
        score = predictions.get_best()[metric]
    elif aggregation == 'mean':
        score = predictions[metric].mean()

    return score
```

### 6.4 Challenges and Solutions

**Challenge 1: Pipeline structure in trial parameters**

Optuna expects scalar parameters, but we have pipeline structures.

**Solution**: Encode structure as categorical indices
```python
# Internally in MetaOptunaSearch
def sample_pipeline_structure(trial, pipeline_space):
    structure = []

    # Sample preprocessing
    preprocessing_idx = trial.suggest_categorical('preprocessing', [0, 1, 2])
    preprocessing_options = [SNV, MSC, Detrend]
    structure.append(preprocessing_options[preprocessing_idx])

    # Sample model
    model_idx = trial.suggest_categorical('model', [0, 1, 2])
    model_options = [PLSRegression, RandomForest, MLPRegressor]
    model_class = model_options[model_idx]

    # Sample model params based on model choice
    if model_idx == 0:  # PLS
        n_components = trial.suggest_categorical('n_components', [5, 10, 15, 20])
        structure.append({'model': model_class(n_components=n_components)})
    # ...

    return structure
```

**Challenge 2: Variable hyperparameters per model**

Different models have different hyperparameters.

**Solution**: Conditional parameter sampling
```python
model_type = trial.suggest_categorical('model_type', ['pls', 'rf', 'mlp'])

if model_type == 'pls':
    n_components = trial.suggest_int('pls_n_components', 5, 20)
    model = PLSRegression(n_components=n_components)
elif model_type == 'rf':
    n_estimators = trial.suggest_int('rf_n_estimators', 50, 200)
    max_depth = trial.suggest_int('rf_max_depth', 3, 10)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
# ...
```

**Challenge 3: Slow trials (some pipelines expensive)**

**Solutions**:
1. Use pruning (Optuna's `trial.report()` + `trial.should_prune()`)
2. Parallel trials (`n_jobs` parameter)
3. Early stopping based on validation performance
4. Caching of intermediate results

---

## 7. Implementation Phases

### Phase 1: Foundation (Weeks 1-3)

**Goal**: Create Pipeline class without breaking existing code

**Tasks**:
1. Extract execution logic to PipelineExecutor
   - Move `run_steps`, `run_step`, controller selection
   - Keep same interfaces
   - Add tests

2. Create Pipeline class skeleton
   ```python
   class Pipeline(BaseEstimator):
       def __init__(self, steps, name="", **kwargs):
           self.steps = steps
           self.name = name
           self._fitted = False

       def fit(self, X, y=None):
           # Create internal dataset
           # Delegate to executor
           # Store binaries
           return self

       def predict(self, X):
           # Check fitted
           # Create dataset
           # Execute with loaded binaries
           return y_pred

       def get_params(self, deep=True):
           # sklearn compatibility
           pass

       def set_params(self, **params):
           # sklearn compatibility
           pass
   ```

3. Update PipelineRunner to use PipelineExecutor
   - No API changes
   - Internal refactoring only

4. Add comprehensive tests
   - Test Pipeline.fit/predict with numpy arrays
   - Test sklearn compatibility (get_params, set_params)
   - Test with sklearn tools (cross_val_score)

**Deliverable**: Working Pipeline class, existing code unchanged

---

### Phase 2: Serialization (Weeks 4-5)

**Goal**: Enable single-file save/load for Pipeline

**Tasks**:
1. Implement Pipeline.save()
   ```python
   def save(self, path):
       data = {
           'steps': self.steps,
           'binaries': self._executor.binaries.to_dict(),
           'context': self._context,
           'metadata': {...}
       }
       with open(path, 'wb') as f:
           pickle.dump(data, f)
   ```

2. Implement Pipeline.load()
   ```python
   @classmethod
   def load(cls, path):
       with open(path, 'rb') as f:
           data = pickle.load(f)
       pipe = cls(data['steps'])
       pipe._fitted = True
       pipe._executor.binaries = BinaryManager.from_dict(data['binaries'])
       return pipe
   ```

3. Test predict/explain with saved pipelines
   ```python
   # Save
   pipe.fit(X_train, y_train)
   pipe.save('model.pkl')

   # Load and predict
   loaded_pipe = Pipeline.load('model.pkl')
   y_pred = loaded_pipe.predict(X_test)
   ```

4. Update PipelineRunner.predict() to work with Pipeline
   - Accept Pipeline objects or model IDs
   - Backward compatible

**Deliverable**: Single-file pipeline persistence

---

### Phase 3: Optuna Integration - Level 1 (Weeks 6-8)

**Goal**: Enable hyperparameter optimization for single Pipeline

**Tasks**:
1. Create OptunaSearch class
   ```python
   class OptunaSearch:
       def __init__(self, pipeline, param_space, cv, scoring, n_trials, **kwargs):
           self.pipeline = pipeline
           self.param_space = param_space
           self.cv = cv
           self.scoring = scoring
           self.n_trials = n_trials

       def fit(self, X, y):
           study = optuna.create_study(direction='minimize')
           study.optimize(self._objective, n_trials=self.n_trials)

           self.best_params_ = study.best_params
           self.best_score_ = study.best_value

           # Refit with best params
           self.best_estimator_ = clone(self.pipeline)
           self.best_estimator_.set_params(**self.best_params_)
           self.best_estimator_.fit(X, y)
   ```

2. Implement parameter sampling
   - Support sklearn parameter naming (double underscore)
   - Handle nested parameters
   - Support ranges, categoricals, distributions

3. Add cross-validation
   - Use sklearn's cross_val_score
   - Support custom CV strategies
   - Handle folds properly

4. Write examples
   ```python
   # examples/Q1_optuna_single.py
   ```

**Deliverable**: Working single-pipeline optimization

---

### Phase 4: Optuna Integration - Level 2 (Weeks 9-11)

**Goal**: Enable meta-pipeline optimization

**Tasks**:
1. Create MetaOptunaSearch class
   ```python
   class MetaOptunaSearch:
       def __init__(self, runner, pipeline_space, objective, metric, aggregation, n_trials, dataset, **kwargs):
           # ...

       def optimize(self):
           study = optuna.create_study()
           study.optimize(self._objective, n_trials=self.n_trials)

       def _objective(self, trial):
           # Sample pipeline structure
           pipeline_config = self._sample_structure(trial)

           # Run pipeline
           predictions, _ = self.runner.run(pipeline_config, self.dataset)

           # Aggregate metric
           score = self._aggregate_predictions(predictions)
           return score
   ```

2. Implement structure sampling
   - Encode pipeline steps as categorical choices
   - Handle conditional parameters
   - Support nested structures

3. Add aggregation strategies
   - best, mean, median, robust_mean
   - Per-partition metrics
   - Multi-objective (Pareto front)

4. Write examples
   ```python
   # examples/Q1_optuna_meta.py
   ```

**Deliverable**: Working meta-pipeline optimization

---

### Phase 5: Documentation and Migration (Weeks 12-13)

**Goal**: Complete documentation and migration guide

**Tasks**:
1. Write comprehensive documentation
   - User guide: Pipeline vs PipelineRunner
   - API reference
   - Migration guide
   - Optuna integration tutorials

2. Update all examples
   - Add new Pipeline examples
   - Keep PipelineRunner examples
   - Add Optuna examples

3. Performance benchmarks
   - Compare old vs new execution speed
   - Optimize bottlenecks

4. Deprecation warnings (if removing old API)

**Deliverable**: Production-ready refactored codebase

---

## 8. Open Questions

### Q1: Should Pipeline support multi-dataset by default?

**Context**: PipelineRunner handles multiple datasets. Should Pipeline?

**Option A**: Pipeline is single-dataset only
- **Pro**: Simpler, matches sklearn
- **Con**: Less powerful

**Option B**: Pipeline can handle multiple datasets
```python
pipe.fit([X1, X2], [y1, y2], dataset_names=['ds1', 'ds2'])
```
- **Pro**: More flexible
- **Con**: Complicates API

**Recommendation**: **Option A** for initial version. Add multi-dataset support later if needed.

---

### Q2: How to handle predictions as first-class objects for stacking?

**Context**: Want to enable:
```python
{"model": PLSRegression(), "use_predictions": ["step_2", "step_5"]}
```

**Current approach**: Predictions stored in external Predictions object.

**Proposed approach**:
- Pipeline can output predictions to context
- Subsequent steps can access via context
- PipelineRunner aggregates for analysis

**Need to design**: API for referencing previous predictions within pipeline.

**Recommendation**: Defer to Phase 2+, not critical for MVP.

---

### Q3: How to handle real-time/streaming execution?

**Context**: Future feature for online learning or real-time prediction.

**Current approach**: Batch processing only.

**Proposed approach**:
- Pipeline.predict() already supports single-sample prediction
- Could add Pipeline.partial_fit() for incremental learning
- Would need streaming-compatible controllers

**Recommendation**: Document as future feature, design architecture to accommodate.

---

### Q4: Should we support pipeline composition/nesting?

**Context**: Enable:
```python
sub_pipe = Pipeline([scaler, smoother])
main_pipe = Pipeline([sub_pipe, cv, model])
```

**Option A**: Support nested pipelines
- **Pro**: Very flexible
- **Con**: Complex implementation

**Option B**: Flatten during initialization
- **Pro**: Simpler
- **Con**: Less intuitive

**Recommendation**: **Option B** for MVP, **Option A** as future enhancement.

---

### Q5: How to handle workspace/run management in Pipeline?

**Context**: PipelineRunner manages workspace, runs, exports. Should Pipeline?

**Option A**: Pipeline has no workspace concept
- **Pro**: Simple, portable
- **Con**: No automatic saving of intermediates

**Option B**: Pipeline can register with workspace
```python
pipe = Pipeline([...], workspace='path/to/workspace')
pipe.fit(X, y)  # Automatically saves to runs/
```
- **Pro**: Integrates with existing workspace system
- **Con**: Couples Pipeline to workspace

**Recommendation**: **Option A**. Workspace is PipelineRunner responsibility. Pipeline is self-contained.

---

### Q6: How to handle explain mode (SHAP) with Pipeline?

**Current approach**: PipelineRunner.explain() is separate method.

**Proposed approach**:
```python
# Option A: Separate method
pipe.explain(X, shap_params={...})

# Option B: Use SHAP directly (pipe is compatible)
import shap
explainer = shap.Explainer(pipe, X_train)
shap_values = explainer(X_test)
```

**Recommendation**: **Option B** (ensure sklearn compatibility). Can provide convenience method later.

---

### Q7: Should Pipeline support different execution modes (train/predict/explain)?

**Current approach**: Mode is PipelineRunner parameter.

**Proposed approach**:
- Pipeline automatically determines mode from method called
  - `fit()` → train mode
  - `predict()` → predict mode
  - `explain()` → explain mode (if we add it)
- Controllers check mode from context

**Recommendation**: Yes, use method-based mode detection. Simpler and more intuitive.

---

## 9. Summary and Next Steps

### 9.1 Core Proposal

**Split PipelineRunner into**:
1. **Pipeline** (NEW): Single, reusable, sklearn-compatible pipeline
2. **PipelineRunner** (REFACTORED): Orchestrator for multiple pipelines/datasets
3. **PipelineExecutor** (EXTRACTED): Internal execution engine

**Enable**:
- sklearn-compatible API for single pipelines
- Single-file persistence
- Optuna optimization at two levels (pipeline and meta-pipeline)
- Composability with sklearn tools

**Preserve**:
- All existing functionality
- Current API (backward compatible during transition)
- Controller architecture
- Multi-source dataset support

### 9.2 Implementation Strategy

**Incremental, non-breaking**:
1. Extract executor logic (internal)
2. Add Pipeline class (new feature)
3. Refactor PipelineRunner to use Pipeline internally (transparent)
4. Add serialization
5. Add Optuna integration
6. Document and migrate examples
7. Deprecate old patterns (v0.6.0)
8. Remove deprecated code (v1.0.0)

### 9.3 Expected Timeline

- **Weeks 1-3**: Foundation (Pipeline class + PipelineExecutor)
- **Weeks 4-5**: Serialization
- **Weeks 6-8**: Optuna Level 1
- **Weeks 9-11**: Optuna Level 2
- **Weeks 12-13**: Documentation and migration

**Total**: ~3 months for complete implementation

### 9.4 Risks and Mitigations

**Risk 1**: Breaking existing code
- **Mitigation**: Maintain backward compatibility for at least 2 minor versions

**Risk 2**: Performance regression
- **Mitigation**: Benchmark throughout, optimize bottlenecks

**Risk 3**: Complexity creep
- **Mitigation**: Start with MVP, add features incrementally

**Risk 4**: Incomplete sklearn compatibility
- **Mitigation**: Follow sklearn's estimator tests, validate with sklearn tools

### 9.5 Success Criteria

**MVP Success**:
1. Pipeline works with sklearn cross_val_score
2. Pipeline can save/load as single file
3. Pipeline.predict() matches runner.predict() results
4. OptunaSearch optimizes pipeline hyperparameters
5. All existing tests pass
6. Documentation complete

**Full Success** (after all phases):
1. MetaOptunaSearch enables structure search
2. Examples migrated
3. Performance benchmarks show no regression
4. Community feedback positive

---

## 10. Request for Feedback

### Key Questions for Discussion

1. **Architecture**: Is the Pipeline/PipelineRunner/PipelineExecutor split clear and appropriate?

2. **API**: Does the proposed sklearn-compatible API meet the stated goals?

3. **Optuna**: Are the two-level optimization strategies (single-pipeline vs meta-pipeline) well-designed?

4. **Backward Compatibility**: Is the migration strategy (coexistence → deprecation → removal) acceptable?

5. **Priorities**: Should any phases be reordered or split differently?

6. **Open Questions**: Which of the open questions (Q1-Q7) are most critical to resolve before starting?

### Next Steps

After validation of this proposal:

1. Create detailed roadmap documents for each phase
2. Set up project tracking (GitHub issues/milestones)
3. Begin Phase 1 implementation
4. Regular progress reviews and adjustments

---

**Document Status**: Draft for Review
**Awaiting**: Maintainer feedback and approval to proceed with detailed roadmaps
