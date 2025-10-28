# Pipeline as Single Estimator/Transformer - Complete Analysis

**Date**: October 14, 2025
**Author**: Analysis based on nirs4all codebase structure
**Goal**: Transform nirs4all pipelines into sklearn-compatible single estimators/transformers

---

## Executive Summary

This document provides a comprehensive analysis and roadmap for transforming the nirs4all pipeline system into a **unified, distributable, sklearn-compatible component** that can:

1. **Act as a single estimator/transformer** (sklearn API compliant)
2. **Be packaged and shipped as a binary** (cross-platform)
3. **Be analyzed as a whole** (SHAP, feature importance, etc.)
4. **Be composed into larger pipelines** (nestable)

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Problem Statement](#2-problem-statement)
3. [Solution Architecture](#3-solution-architecture)
4. [Implementation Roadmap](#4-implementation-roadmap)
5. [Technical Details](#5-technical-details)
6. [Distribution Strategy](#6-distribution-strategy)
7. [Examples and Use Cases](#7-examples-and-use-cases)
8. [References and Best Practices](#8-references-and-best-practices)

---

## 1. Current State Analysis

### 1.1 Current Architecture

The nirs4all pipeline system consists of:

```
Pipeline Logic (nirs4all/pipeline/)
├── runner.py          # Orchestrates execution
├── config.py          # Configuration management
├── operation.py       # Operation abstraction
├── serialization.py   # Save/load components
├── io.py             # File I/O management
└── binary_loader.py  # Load saved binaries

Controllers (nirs4all/controllers/)
├── controller.py                    # Base controller
├── sklearn/op_model.py             # Sklearn models
├── sklearn/op_transformermixin.py  # Sklearn transformers
├── models/base_model_controller.py # Model base
└── [other controllers...]

Dataset (nirs4all/dataset/)
├── dataset.py        # Core dataset orchestrator
├── features.py       # Feature management
├── targets.py        # Target management
├── predictions.py    # Prediction storage
└── [other components...]
```

### 1.2 Key Characteristics

**Strengths:**
- ✅ Powerful and flexible configuration system (JSON/YAML)
- ✅ Support for multi-source datasets (unique feature)
- ✅ Advanced preprocessing pipeline (NIRS-specific)
- ✅ Comprehensive prediction tracking and analysis
- ✅ Cross-validation integration with fold management
- ✅ Hyperparameter tuning with Optuna
- ✅ Binary serialization of fitted components
- ✅ Support for sklearn, TensorFlow, and PyTorch models

**Limitations:**
- ❌ Not sklearn-compatible (no `fit`, `predict`, `transform` API)
- ❌ Cannot be used as a component in sklearn pipelines
- ❌ Not directly analyzable with standard tools (SHAP, eli5, etc.)
- ❌ Requires `PipelineRunner` + `DatasetConfigs` for execution
- ❌ Cannot be serialized as a single object (multiple files)
- ❌ Stateful execution with external context management

### 1.3 Current Execution Flow

```python
# Current usage (verbose)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.data import DatasetConfigs

pipeline = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    ShuffleSplit(n_splits=3),
    {"model": PLSRegression(n_components=5)}
]

config = PipelineConfigs(pipeline)
dataset_config = DatasetConfigs('data/')
runner = PipelineRunner()
predictions, datasets = runner.run(config, dataset_config)
```

**Problem**: This is not compatible with sklearn's API or tooling ecosystem.

---

## 2. Problem Statement

### 2.1 Three Main Requirements

#### Requirement 1: Single Estimator/Transformer Interface
**Goal**: Use the pipeline like any sklearn estimator

```python
# Desired usage
from nirs4all.pipeline import Nirs4allPipeline

pipe = Nirs4allPipeline([
    MinMaxScaler(),
    StandardNormalVariate(),
    PLSRegression(n_components=5)
])

# Standard sklearn API
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
score = pipe.score(X_test, y_test)

# Works with sklearn tools
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X, y, cv=5)

# Works with SHAP
import shap
explainer = shap.Explainer(pipe, X_train)
shap_values = explainer(X_test)
```

#### Requirement 2: Binary Distribution
**Goal**: Package and distribute complete pipelines

```python
# Save as single file
pipe.save('my_pipeline.nirs4all')

# Load and use anywhere
pipe = Nirs4allPipeline.load('my_pipeline.nirs4all')
y_pred = pipe.predict(X_new)
```

#### Requirement 3: Composability
**Goal**: Use nirs4all pipelines as components

```python
# Nest in sklearn pipelines
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor

nirs_pipe1 = Nirs4allPipeline([...])
nirs_pipe2 = Nirs4allPipeline([...])

meta_pipe = Pipeline([
    ('feature_eng', nirs_pipe1),
    ('model', nirs_pipe2)
])

# Or use in ensemble
ensemble = VotingRegressor([
    ('nirs1', nirs_pipe1),
    ('nirs2', nirs_pipe2)
])
```

### 2.2 Technical Challenges

1. **State Management**: Current system uses external context and dataset objects
2. **Multi-Source Data**: Sklearn expects 2D arrays, nirs4all can handle multi-source 3D
3. **Fold Management**: Built-in CV logic conflicts with sklearn's CV tools
4. **Binary Format**: Multiple files vs. single binary
5. **Y-Processing**: Target transformation needs to be reversible
6. **Cross-Framework**: sklearn, TensorFlow, PyTorch support

---

## 3. Solution Architecture

### 3.1 Core Design: Wrapper + Internal State

Create a new `Nirs4allPipeline` class that:
1. **Wraps** the existing pipeline system
2. **Implements** sklearn's `BaseEstimator` interface
3. **Manages** internal state (dataset, context, fitted components)
4. **Exposes** standard `fit`, `predict`, `transform` methods

### 3.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Nirs4allPipeline                         │
│                 (sklearn-compatible wrapper)                 │
├─────────────────────────────────────────────────────────────┤
│  Public API:                                                │
│    - fit(X, y)                                              │
│    - predict(X)                                             │
│    - transform(X)                                           │
│    - fit_predict(X, y)                                      │
│    - score(X, y)                                            │
│    - get_params() / set_params()                            │
│    - save() / load()                                        │
├─────────────────────────────────────────────────────────────┤
│  Internal Components:                                        │
│    - _pipeline_config: PipelineConfigs                      │
│    - _dataset: SpectroDataset (internal state)             │
│    - _context: Dict (execution context)                     │
│    - _fitted_components: List[Tuple] (binaries)            │
│    - _runner: PipelineRunner (execution engine)            │
│    - _y_transformer: Optional[TransformerMixin]            │
├─────────────────────────────────────────────────────────────┤
│  Delegates to:                                              │
│    ↓                                                         │
│    Existing nirs4all pipeline system                        │
│    (minimal changes required)                               │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Key Design Decisions

#### Decision 1: Wrapper Pattern (Not Rewrite)
- **Why**: Preserve existing functionality and tests
- **How**: Create new `Nirs4allPipeline` class that uses existing code
- **Benefit**: Minimal risk, incremental development

#### Decision 2: Internal Dataset Management
- **Why**: Sklearn expects `fit(X, y)`, not `DatasetConfigs`
- **How**: Create `SpectroDataset` internally from numpy arrays
- **Benefit**: Transparent to users

#### Decision 3: Stateful Fitted Components
- **Why**: Need to store fitted transformers and models
- **How**: Store binaries internally, load during predict
- **Benefit**: Single-object serialization

#### Decision 4: Optional CV Delegation
- **Why**: Can use sklearn's CV or built-in fold management
- **How**: Detect CV in pipeline, use appropriately
- **Benefit**: Flexibility for users

---

## 4. Implementation Roadmap

### Phase 1: Core Wrapper (Week 1-2)

**Goal**: Basic sklearn-compatible wrapper

```python
# File: nirs4all/pipeline/sklearn_wrapper.py

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import pickle
import numpy as np
from typing import List, Any, Optional, Union
from pathlib import Path

class Nirs4allPipeline(BaseEstimator):
    """
    Sklearn-compatible wrapper for nirs4all pipelines.

    Parameters
    ----------
    steps : list
        Pipeline steps in nirs4all format
    name : str, optional
        Pipeline name for tracking
    cv : cross-validator, optional
        If None, no cross-validation. If provided, overrides fold management.
    task_type : str, optional
        'regression' or 'classification'. Auto-detected if None.
    verbose : int, default=0
        Verbosity level
    """

    def __init__(
        self,
        steps: List[Any],
        name: str = "nirs4all_pipeline",
        cv: Optional[Any] = None,
        task_type: Optional[str] = None,
        verbose: int = 0
    ):
        self.steps = steps
        self.name = name
        self.cv = cv
        self.task_type = task_type
        self.verbose = verbose

        # Internal state (set during fit)
        self._is_fitted = False
        self._pipeline_config = None
        self._dataset = None
        self._context = None
        self._fitted_components = {}
        self._y_transformer = None
        self._feature_names_in = None

    def fit(self, X, y=None):
        """
        Fit the pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,) or (n_samples, n_targets), optional
            Target values

        Returns
        -------
        self : object
            Fitted pipeline
        """
        # Validate input
        if y is not None:
            X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        else:
            X = check_array(X)

        # Store feature names if available
        if hasattr(X, 'columns'):
            self._feature_names_in = list(X.columns)

        # Convert to numpy
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)
            if y.ndim == 1:
                y = y.reshape(-1, 1)

        # Create internal dataset
        from nirs4all.data import SpectroDataset
        self._dataset = SpectroDataset(name="sklearn_dataset")
        self._dataset.add_samples(X)
        if y is not None:
            self._dataset.add_targets(y)

        # Add CV folds if provided
        if self.cv is not None:
            from sklearn.model_selection import check_cv
            cv = check_cv(self.cv, y, classifier=False)
            folds = list(cv.split(X, y))
            self._dataset.set_folds(folds)

        # Create pipeline config
        from nirs4all.pipeline import PipelineConfigs
        self._pipeline_config = PipelineConfigs(
            self.steps,
            name=self.name
        )

        # Execute pipeline in training mode
        from nirs4all.pipeline import PipelineRunner
        runner = PipelineRunner(
            save_files=False,  # Keep in memory
            verbose=self.verbose,
            mode="train"
        )

        # Create temporary dataset config
        from nirs4all.data import DatasetConfigs
        # Use in-memory dataset
        dataset_config = DatasetConfigs.from_dataset(self._dataset)

        # Run pipeline and capture fitted components
        predictions, _ = runner.run(self._pipeline_config, dataset_config)

        # Store fitted components (binaries)
        self._fitted_components = runner.step_binaries
        self._context = runner.history.final_context
        self._is_fitted = True

        return self

    def predict(self, X):
        """
        Predict using the fitted pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values
        """
        check_is_fitted(self, '_is_fitted')
        X = check_array(X)

        # Create dataset for prediction
        pred_dataset = SpectroDataset(name="pred_dataset")
        pred_dataset.add_samples(X)

        # Load fitted components
        from nirs4all.pipeline.binary_loader import BinaryLoader
        loader = BinaryLoader(
            simulation_path=None,  # In-memory
            step_binaries=self._fitted_components
        )

        # Execute in prediction mode
        runner = PipelineRunner(
            save_files=False,
            verbose=self.verbose,
            mode="predict"
        )

        dataset_config = DatasetConfigs.from_dataset(pred_dataset)

        # Run prediction
        y_pred, _ = runner.predict(
            prediction_obj=self._get_best_model_id(),
            dataset_config=dataset_config
        )

        return y_pred.ravel() if y_pred.shape[1] == 1 else y_pred

    def transform(self, X):
        """
        Transform data using fitted preprocessing steps.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples

        Returns
        -------
        X_transformed : ndarray
            Transformed data
        """
        check_is_fitted(self, '_is_fitted')
        X = check_array(X)

        # Similar to predict but return features instead
        # Implementation depends on pipeline structure
        raise NotImplementedError(
            "transform() requires pipeline to end with a transformer, "
            "not a predictor"
        )

    def score(self, X, y):
        """
        Score the pipeline using appropriate metric.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            True values

        Returns
        -------
        score : float
            R² for regression, accuracy for classification
        """
        from sklearn.metrics import r2_score, accuracy_score

        y_pred = self.predict(X)

        if self._is_regression():
            return r2_score(y, y_pred)
        else:
            return accuracy_score(y, y_pred)

    def _is_regression(self):
        """Check if pipeline is for regression."""
        if self.task_type is not None:
            return self.task_type == 'regression'
        return self._dataset.task_type == 'regression'

    def _get_best_model_id(self):
        """Get the ID of the best model from training."""
        # Implementation depends on prediction storage
        # Return best model's identifier
        pass

    def save(self, path: Union[str, Path]):
        """
        Save fitted pipeline to file.

        Parameters
        ----------
        path : str or Path
            Output file path
        """
        check_is_fitted(self, '_is_fitted')

        state = {
            'steps': self.steps,
            'name': self.name,
            'cv': self.cv,
            'task_type': self.task_type,
            'verbose': self.verbose,
            'fitted_components': self._fitted_components,
            'context': self._context,
            'feature_names_in': self._feature_names_in,
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Union[str, Path]):
        """
        Load fitted pipeline from file.

        Parameters
        ----------
        path : str or Path
            Input file path

        Returns
        -------
        pipeline : Nirs4allPipeline
            Loaded fitted pipeline
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)

        pipe = cls(
            steps=state['steps'],
            name=state['name'],
            cv=state['cv'],
            task_type=state['task_type'],
            verbose=state['verbose']
        )

        pipe._fitted_components = state['fitted_components']
        pipe._context = state['context']
        pipe._feature_names_in = state['feature_names_in']
        pipe._is_fitted = True

        return pipe
```

**Implementation Tasks:**
- [ ] Create `Nirs4allPipeline` class with sklearn base classes
- [ ] Implement `fit()` method with internal dataset creation
- [ ] Implement `predict()` method with binary loading
- [ ] Add `get_params()` / `set_params()` for sklearn compatibility
- [ ] Add basic `save()` / `load()` methods
- [ ] Write unit tests for basic functionality

### Phase 2: Advanced Features (Week 3-4)

**Goal**: Full sklearn compatibility and SHAP support

**Tasks:**
- [ ] Implement `transform()` for transformer-ending pipelines
- [ ] Add `fit_transform()` optimization
- [ ] Support for `Pipeline` nesting (recursion)
- [ ] SHAP integration via `__sklearn_is_fitted__` and `__call__`
- [ ] Feature name tracking (`get_feature_names_out()`)
- [ ] Parallel prediction support
- [ ] Memory optimization for large pipelines

**Example: SHAP Integration**

```python
class Nirs4allPipeline(BaseEstimator):
    # ...existing code...

    def __call__(self, X):
        """Make pipeline callable for SHAP compatibility."""
        return self.predict(X)

    def __sklearn_is_fitted__(self):
        """Check if fitted (sklearn convention)."""
        return self._is_fitted

    # Enable SHAP
    def enable_shap_mode(self):
        """
        Configure pipeline for SHAP analysis.
        Returns the final model for explanation.
        """
        check_is_fitted(self, '_is_fitted')

        # Extract final fitted model
        final_model = self._get_final_model()

        # Create wrapper that includes preprocessing
        class ShapWrapper:
            def __init__(self, pipeline, model):
                self.pipeline = pipeline
                self.model = model

            def predict(self, X):
                # Apply preprocessing
                X_prep = self.pipeline._apply_preprocessing(X)
                # Predict with final model
                return self.model.predict(X_prep)

        return ShapWrapper(self, final_model)
```

### Phase 3: Distribution Format (Week 5-6)

**Goal**: Cross-platform binary distribution

**Options Analysis:**

#### Option A: Enhanced Pickle (Simplest)
```python
# Single file, includes everything
pipe.save('model.nirs4all')  # Actually a pickle
```
**Pros**: Simple, works immediately
**Cons**: Not cross-language, Python version dependent

#### Option B: ONNX Format (Best for deployment)
```python
from skl2onnx import to_onnx

# Convert to ONNX
onnx_model = pipe.to_onnx(X_sample)
with open('model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

# Use anywhere
import onnxruntime as rt
sess = rt.InferenceSession('model.onnx')
y_pred = sess.run(None, {'input': X_test})[0]
```
**Pros**: Cross-platform, fast inference, production-ready
**Cons**: Limited operator support, may not support custom ops

#### Option C: Joblib + Manifest (Hybrid)
```python
# Directory-based format
pipe.save('model_bundle/')
# Creates:
#   model_bundle/
#   ├── manifest.json        # Metadata
#   ├── pipeline.joblib      # Fitted pipeline
#   ├── config.yaml          # Original config
#   └── metadata/            # Additional info

# Load
pipe = Nirs4allPipeline.load('model_bundle/')
```
**Pros**: Flexible, debuggable, supports complex pipelines
**Cons**: Multiple files, requires zip for distribution

**Recommendation**: Implement all three, make it configurable:

```python
pipe.save('model.pkl', format='pickle')      # Default
pipe.save('model.onnx', format='onnx')       # Production
pipe.save('model_dir/', format='bundle')     # Development
```

### Phase 4: Testing & Documentation (Week 7-8)

**Goal**: Production-ready with comprehensive tests

**Testing Strategy:**
```python
# tests/test_sklearn_wrapper.py

class TestNirs4allPipeline:
    def test_basic_fit_predict(self):
        """Test basic sklearn API compatibility."""
        X, y = load_sample_data()
        pipe = Nirs4allPipeline([MinMaxScaler(), PLSRegression()])
        pipe.fit(X, y)
        y_pred = pipe.predict(X)
        assert y_pred.shape == y.shape

    def test_sklearn_cross_val(self):
        """Test integration with sklearn CV."""
        X, y = load_sample_data()
        pipe = Nirs4allPipeline([...])
        scores = cross_val_score(pipe, X, y, cv=5)
        assert len(scores) == 5

    def test_shap_compatibility(self):
        """Test SHAP analysis works."""
        X, y = load_sample_data()
        pipe = Nirs4allPipeline([...])
        pipe.fit(X, y)

        explainer = shap.Explainer(pipe.enable_shap_mode(), X)
        shap_values = explainer(X)
        assert shap_values.shape == X.shape

    def test_save_load_roundtrip(self):
        """Test serialization works."""
        X, y = load_sample_data()
        pipe = Nirs4allPipeline([...])
        pipe.fit(X, y)

        pipe.save('test.pkl')
        pipe2 = Nirs4allPipeline.load('test.pkl')

        np.testing.assert_array_equal(
            pipe.predict(X),
            pipe2.predict(X)
        )

    def test_nested_pipelines(self):
        """Test using as component."""
        X, y = load_sample_data()

        inner_pipe = Nirs4allPipeline([MinMaxScaler(), ...])
        outer_pipe = Pipeline([
            ('nirs', inner_pipe),
            ('model', RandomForestRegressor())
        ])

        outer_pipe.fit(X, y)
        assert hasattr(outer_pipe, 'predict')
```

---

## 5. Technical Details

### 5.1 Handling Multi-Source Data

**Problem**: Sklearn expects 2D `(n_samples, n_features)`, but nirs4all supports multi-source 3D data.

**Solution**: Flatten internally, reconstruct during processing

```python
class Nirs4allPipeline(BaseEstimator):
    def __init__(self, steps, multi_source_mode='auto'):
        """
        multi_source_mode : 'auto', 'flatten', 'separate'
            - 'auto': Detect from input shape
            - 'flatten': Concatenate sources along feature axis
            - 'separate': Keep sources separate (return list)
        """
        self.multi_source_mode = multi_source_mode
        self._source_splits = None

    def fit(self, X, y=None):
        # Detect if X is multi-source
        if isinstance(X, list) or (isinstance(X, np.ndarray) and X.ndim == 3):
            self._is_multi_source = True

            if isinstance(X, list):
                # List of 2D arrays
                self._source_splits = [arr.shape[1] for arr in X]
                X_flat = np.hstack(X)  # Concatenate horizontally
            else:
                # 3D array (n_samples, n_sources, n_features)
                self._source_splits = [X.shape[2]] * X.shape[1]
                X_flat = X.reshape(X.shape[0], -1)

            X = X_flat
        else:
            self._is_multi_source = False

        # Continue with normal fit...
        return super().fit(X, y)

    def predict(self, X):
        # Flatten if needed
        if self._is_multi_source:
            if isinstance(X, list):
                X = np.hstack(X)
            elif X.ndim == 3:
                X = X.reshape(X.shape[0], -1)

        return super().predict(X)
```

### 5.2 Y-Transformation Management

**Problem**: Target transformations need to be reversed for predictions.

**Solution**: Track y-transformations and apply inverse automatically

```python
class Nirs4allPipeline(BaseEstimator):
    def fit(self, X, y=None):
        # Detect y-processing steps
        self._y_steps = []
        for step in self.steps:
            if isinstance(step, dict) and 'y_processing' in step:
                self._y_steps.append(step['y_processing'])

        # Fit y-transformers
        if self._y_steps:
            from sklearn.compose import TransformedTargetRegressor
            # Wrap model with automatic inverse transform
            self._y_transformer = Pipeline(self._y_steps)
            self._y_transformer.fit(y.reshape(-1, 1))

        # Continue with transformed y
        y_transformed = self._y_transformer.transform(y.reshape(-1, 1))
        # ... fit pipeline with y_transformed ...

    def predict(self, X):
        y_pred = self._predict_raw(X)

        # Apply inverse transform
        if self._y_transformer is not None:
            y_pred = self._y_transformer.inverse_transform(y_pred)

        return y_pred
```

### 5.3 Cross-Validation Strategy

**Problem**: nirs4all has built-in CV, sklearn also has CV. Avoid conflicts.

**Solution**: Make CV optional, delegate to sklearn when appropriate

```python
class Nirs4allPipeline(BaseEstimator):
    def __init__(self, steps, cv=None, use_internal_cv=True):
        """
        cv : sklearn cross-validator or None
            If provided, overrides any CV in pipeline steps
        use_internal_cv : bool
            If True, execute CV steps in pipeline
            If False, remove CV steps and let sklearn handle it
        """
        self.cv = cv
        self.use_internal_cv = use_internal_cv
        self.steps = steps

    def fit(self, X, y=None):
        steps = self.steps.copy()

        # Remove CV steps if not using internal CV
        if not self.use_internal_cv:
            steps = [s for s in steps if not self._is_cv_step(s)]

        # Add external CV if provided
        if self.cv is not None and self.use_internal_cv:
            # Inject CV into pipeline
            steps.insert(-1, self.cv)  # Before model

        self._processed_steps = steps
        # ... continue fit ...

    @staticmethod
    def _is_cv_step(step):
        """Check if step is a CV splitter."""
        from sklearn.model_selection import BaseCrossValidator
        if isinstance(step, BaseCrossValidator):
            return True
        if isinstance(step, dict):
            for v in step.values():
                if isinstance(v, BaseCrossValidator):
                    return True
        return False
```

### 5.4 Binary Serialization Strategy

**Key Insight**: Store everything needed for prediction in a single object

```python
class Nirs4allPipeline(BaseEstimator):
    def save(self, path, format='auto', compress=True):
        """
        Save fitted pipeline.

        Parameters
        ----------
        path : str or Path
            Output path
        format : str
            'pickle', 'joblib', 'onnx', 'bundle', 'auto'
        compress : bool
            Whether to compress (for pickle/joblib)
        """
        check_is_fitted(self, '_is_fitted')

        # Collect all state
        state = {
            # Constructor params (for get_params)
            'constructor_params': {
                'steps': self.steps,
                'name': self.name,
                'cv': self.cv,
                # ... all __init__ params
            },

            # Fitted state
            'fitted_components': self._fitted_components,
            'context': self._context,
            'y_transformer': self._y_transformer,
            'source_splits': self._source_splits,
            'feature_names_in': self._feature_names_in,
            'is_multi_source': self._is_multi_source,

            # Metadata
            'nirs4all_version': nirs4all.__version__,
            'sklearn_version': sklearn.__version__,
            'creation_date': datetime.now().isoformat(),
        }

        path = Path(path)

        if format == 'auto':
            format = self._detect_format(path)

        if format == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        elif format == 'joblib':
            import joblib
            joblib.dump(state, path, compress=compress)

        elif format == 'onnx':
            self._save_onnx(path, state)

        elif format == 'bundle':
            self._save_bundle(path, state)

    def _save_onnx(self, path, state):
        """Convert to ONNX format."""
        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import FloatTensorType

        # Create input type
        n_features = len(self._feature_names_in) if self._feature_names_in else 1
        initial_type = [('float_input', FloatTensorType([None, n_features]))]

        try:
            # Convert sklearn-compatible pipeline to ONNX
            onnx_model = to_onnx(self, initial_types=initial_type)

            # Save
            with open(path, 'wb') as f:
                f.write(onnx_model.SerializeToString())

        except Exception as e:
            raise RuntimeError(
                f"ONNX conversion failed: {e}\n"
                "Note: Not all pipeline components may be ONNX-compatible. "
                "Try format='pickle' or format='bundle' instead."
            )
```

---

## 6. Distribution Strategy

### 6.1 Format Comparison Table

| Format | Size | Speed | Cross-platform | Cross-language | Production-ready | Debuggable |
|--------|------|-------|----------------|----------------|------------------|------------|
| Pickle | Medium | Fast | Python only | No | ❌ | ❌ |
| Joblib | Small | Fast | Python only | No | ⚠️ | ❌ |
| ONNX | Small | Very Fast | ✅ | ✅ | ✅ | ❌ |
| Bundle | Large | Fast | Python only | No | ⚠️ | ✅ |

### 6.2 Recommended Workflow

```python
# Development: Use bundle for debugging
pipe.save('dev_model/', format='bundle')

# Staging: Use joblib for Python deployment
pipe.save('staging_model.joblib', format='joblib', compress=3)

# Production: Use ONNX for maximum compatibility
pipe.save('prod_model.onnx', format='onnx')
```

### 6.3 Version Compatibility

**Problem**: Pipelines trained with one version may not load with another.

**Solution**: Version metadata + fallback strategies

```python
class Nirs4allPipeline(BaseEstimator):
    @classmethod
    def load(cls, path, strict=True):
        """
        Load pipeline with version checking.

        Parameters
        ----------
        strict : bool
            If True, raise error on version mismatch
            If False, attempt to load anyway with warnings
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)

        # Check version compatibility
        saved_version = state.get('nirs4all_version', 'unknown')
        current_version = nirs4all.__version__

        if saved_version != current_version:
            msg = (
                f"Pipeline was saved with nirs4all=={saved_version}, "
                f"but you are using version {current_version}. "
            )

            if strict:
                raise ValueError(msg + "Set strict=False to attempt loading anyway.")
            else:
                warnings.warn(msg + "Loading may fail or produce incorrect results.")

        # Reconstruct pipeline
        pipe = cls(**state['constructor_params'])

        # Restore fitted state
        for key, value in state.items():
            if key != 'constructor_params':
                setattr(pipe, f'_{key}', value)

        pipe._is_fitted = True
        return pipe
```

---

## 7. Examples and Use Cases

### 7.1 Basic Usage

```python
from nirs4all.pipeline import Nirs4allPipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
import numpy as np

# Create pipeline (nirs4all format)
pipeline = Nirs4allPipeline([
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    ShuffleSplit(n_splits=3, test_size=0.25),
    {"model": PLSRegression(n_components=5), "name": "PLS_5"}
], name="my_pipeline")

# Use like any sklearn estimator
X_train, X_test = ...  # Your data
y_train, y_test = ...

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
score = pipeline.score(X_test, y_test)

print(f"R² score: {score:.3f}")
```

### 7.2 SHAP Analysis

```python
import shap
from nirs4all.pipeline import Nirs4allPipeline

# Train pipeline
pipe = Nirs4allPipeline([...])
pipe.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.Explainer(pipe, X_train)
shap_values = explainer(X_test)

# Visualize
shap.plots.waterfall(shap_values[0])
shap.plots.beeswarm(shap_values)
```

### 7.3 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'steps__model__n_components': [3, 5, 10, 15],
    'steps__preprocessing__window_length': [5, 9, 11]
}

# Grid search
pipe = Nirs4allPipeline([...])
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.3f}")
```

### 7.4 Ensemble Methods

```python
from sklearn.ensemble import VotingRegressor

# Create multiple pipelines
pipe1 = Nirs4allPipeline([...], name="pls_pipeline")
pipe2 = Nirs4allPipeline([...], name="rf_pipeline")
pipe3 = Nirs4allPipeline([...], name="nn_pipeline")

# Combine in ensemble
ensemble = VotingRegressor([
    ('pls', pipe1),
    ('rf', pipe2),
    ('nn', pipe3)
], weights=[2, 1, 1])

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
```

### 7.5 Nested Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Use nirs4all pipeline as preprocessing step
nirs_preproc = Nirs4allPipeline([
    MinMaxScaler(),
    StandardNormalVariate(),
    SavitzkyGolay()
], name="nirs_preprocessing")

# Combine with sklearn steps
full_pipeline = Pipeline([
    ('nirs', nirs_preproc),
    ('pca', PCA(n_components=20)),
    ('model', PLSRegression(n_components=5))
])

full_pipeline.fit(X_train, y_train)
```

### 7.6 Model Distribution

```python
# Train and save
pipe = Nirs4allPipeline([...])
pipe.fit(X_train, y_train)
pipe.save('model_v1.0.pkl')

# --- On another machine ---
from nirs4all.pipeline import Nirs4allPipeline

# Load and predict
pipe = Nirs4allPipeline.load('model_v1.0.pkl')
y_pred = pipe.predict(X_new)
```

---

## 8. References and Best Practices

### 8.1 Sklearn API Compliance

**Must implement:**
- `fit(X, y)` - Train the estimator
- `predict(X)` - Make predictions
- `get_params(deep=True)` - Get parameters as dict
- `set_params(**params)` - Set parameters

**Should implement:**
- `score(X, y)` - Calculate performance metric
- `fit_predict(X, y)` - Fit and predict in one call
- `transform(X)` - Transform data (if transformer)
- `fit_transform(X, y)` - Fit and transform (if transformer)

**Optional but recommended:**
- `get_feature_names_out()` - Feature names after transformation
- `__sklearn_is_fitted__()` - Check if fitted
- `_more_tags()` - Provide metadata about estimator

### 8.2 Serialization Best Practices

1. **Always include version metadata**
```python
state = {
    'nirs4all_version': nirs4all.__version__,
    'sklearn_version': sklearn.__version__,
    'python_version': sys.version,
    'numpy_version': np.__version__,
    'creation_date': datetime.now().isoformat(),
    # ... actual state ...
}
```

2. **Use protocol version explicitly**
```python
pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
```

3. **Test load/save roundtrip**
```python
assert np.allclose(
    pipe.predict(X),
    pipe.load('temp.pkl').predict(X)
)
```

4. **Document format stability**
```markdown
## Serialization Format

Version 1.0 (stable): pickle protocol 4
- Compatible with Python 3.8+
- Forward compatible guarantee: files from 1.0 will load in 2.0

Version 2.0 (planned): Add ONNX export
```

### 8.3 Testing Strategy

**Test coverage matrix:**
```python
# 1. Basic functionality
- fit + predict
- score calculation
- get/set params

# 2. sklearn integration
- Pipeline nesting
- cross_val_score
- GridSearchCV
- VotingRegressor/Classifier

# 3. Serialization
- pickle roundtrip
- joblib roundtrip
- ONNX export (if supported)
- Version compatibility

# 4. SHAP integration
- Explainer creation
- SHAP value computation
- Visualization

# 5. Edge cases
- Empty pipeline
- Single-step pipeline
- Multi-source data
- Missing values
- Large datasets
```

### 8.4 Performance Optimization

```python
# Use caching for repeated fits
class Nirs4allPipeline(BaseEstimator):
    def __init__(self, steps, memory=None):
        """
        memory : str or joblib.Memory, optional
            Used to cache fitted transformers
        """
        self.memory = memory

    def fit(self, X, y=None):
        if self.memory is not None:
            # Use caching
            from sklearn.pipeline import Pipeline
            cache_pipe = Pipeline(self.steps, memory=self.memory)
            # ... use cached version ...
```

### 8.5 Documentation Template

```python
class Nirs4allPipeline(BaseEstimator):
    """
    Sklearn-compatible wrapper for nirs4all pipelines.

    This class allows using nirs4all's powerful preprocessing and modeling
    capabilities within the sklearn ecosystem, including:

    - Cross-validation with `cross_val_score`
    - Hyperparameter tuning with `GridSearchCV`
    - Ensemble methods like `VotingRegressor`
    - SHAP analysis for model interpretation
    - Standard serialization with pickle/joblib
    - ONNX export for production deployment

    Parameters
    ----------
    steps : list
        Pipeline steps in nirs4all format. Can include:
        - Sklearn transformers (e.g., MinMaxScaler())
        - NIRS-specific operators (e.g., StandardNormalVariate())
        - Models (e.g., {"model": PLSRegression()})
        - Y-processing (e.g., {"y_processing": MinMaxScaler()})
        - Cross-validation (e.g., ShuffleSplit())

    name : str, default="nirs4all_pipeline"
        Pipeline name for tracking and logging

    cv : cross-validator, optional
        If provided, overrides CV steps in pipeline

    task_type : {"regression", "classification"}, optional
        Task type. Auto-detected from targets if None

    verbose : int, default=0
        Verbosity level (0=silent, 1=progress, 2=debug)

    Attributes
    ----------
    _is_fitted : bool
        Whether the pipeline has been fitted

    _feature_names_in : list of str
        Feature names from training data

    Examples
    --------
    Basic usage:

    >>> from nirs4all.pipeline import Nirs4allPipeline
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from sklearn.cross_decomposition import PLSRegression
    >>>
    >>> pipe = Nirs4allPipeline([
    ...     MinMaxScaler(),
    ...     PLSRegression(n_components=5)
    ... ])
    >>> pipe.fit(X_train, y_train)
    >>> y_pred = pipe.predict(X_test)

    With SHAP:

    >>> import shap
    >>> explainer = shap.Explainer(pipe, X_train)
    >>> shap_values = explainer(X_test)

    Save and load:

    >>> pipe.save('my_model.pkl')
    >>> pipe2 = Nirs4allPipeline.load('my_model.pkl')

    See Also
    --------
    sklearn.pipeline.Pipeline : Sklearn's native Pipeline
    nirs4all.pipeline.PipelineRunner : Original nirs4all runner

    Notes
    -----
    This wrapper maintains full compatibility with the original nirs4all
    pipeline system while providing sklearn API compliance. All nirs4all
    features (multi-source data, NIRS operators, fold management) are
    preserved.

    References
    ----------
    .. [1] Scikit-learn Pipeline documentation:
           https://scikit-learn.org/stable/modules/compose.html
    """
```

---

## 9. Conclusion and Recommendations

### 9.1 Summary of Approach

**Core Strategy**: Wrapper pattern that delegates to existing nirs4all infrastructure

**Key Benefits**:
1. ✅ Minimal changes to existing codebase (low risk)
2. ✅ Full sklearn compatibility (interoperability)
3. ✅ Maintains nirs4all's unique features (multi-source, NIRS ops)
4. ✅ Single-file distribution (easy sharing)
5. ✅ SHAP-compatible (explainability)
6. ✅ Composable (nesting, ensembles)

### 9.2 Implementation Priority

**Phase 1 (Critical)**: Basic wrapper with fit/predict
- Enables sklearn integration immediately
- Low complexity, high value

**Phase 2 (Important)**: Save/load and SHAP
- Enables distribution and explainability
- Medium complexity, high value

**Phase 3 (Nice-to-have)**: ONNX export
- Enables production deployment
- High complexity, medium value (niche use case)

### 9.3 Next Steps

1. **Prototype** the basic `Nirs4allPipeline` wrapper (1-2 days)
2. **Test** with existing examples (Q1, Q2, etc.)
3. **Iterate** based on feedback
4. **Document** with examples
5. **Release** as experimental feature
6. **Gather feedback** from users
7. **Stabilize** API and release v1.0

### 9.4 Long-term Vision

```python
# Future state: Seamless sklearn integration

from nirs4all import Nirs4allPipeline

# Define pipeline
pipe = Nirs4allPipeline([...])

# Use anywhere sklearn works
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X, y, cv=5)

# Explain with SHAP
import shap
shap.plots.waterfall(shap_values[0])

# Deploy to production
pipe.save('model.onnx', format='onnx')

# Share with colleagues
pipe.save('model.pkl')
# They can load and use immediately
```

**This document provides a complete roadmap for transforming nirs4all into a sklearn-compatible, distributable, and analyzable pipeline system while preserving all its unique capabilities.**
