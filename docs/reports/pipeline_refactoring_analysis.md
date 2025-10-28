# Pipeline Runner Refactoring - sklearn API Native Implementation

**Date**: October 14, 2025
**Goal**: Refactor PipelineRunner to directly implement sklearn's BaseEstimator API without wrapper

---

## Executive Summary

This document provides a complete analysis and roadmap for **refactoring the PipelineRunner** to natively implement sklearn's API (`fit`, `predict`, `transform`, etc.) instead of using a wrapper. This is a more invasive but architecturally cleaner approach that transforms nirs4all into a first-class sklearn-compatible library.

**Key Difference from Wrapper Approach:**
- **Wrapper**: Add a new class that delegates to existing code (low risk, quick)
- **Refactoring**: Transform existing classes to be sklearn-compatible (higher risk, cleaner architecture)

---

## Table of Contents

1. [Current Architecture Analysis](#1-current-architecture-analysis)
2. [Target Architecture](#2-target-architecture)
3. [Refactoring Strategy](#3-refactoring-strategy)
4. [Detailed Implementation Plan](#4-detailed-implementation-plan)
5. [Migration Path](#5-migration-path)
6. [Code Examples](#6-code-examples)
7. [Testing Strategy](#7-testing-strategy)
8. [Risk Assessment](#8-risk-assessment)

---

## 1. Current Architecture Analysis

### 1.1 Current Class Structure

```
Current Design (Non-sklearn):
┌─────────────────────────────────────┐
│         PipelineConfigs             │
│  - Load/parse configuration         │
│  - Expand combinations              │
│  - Serialize/deserialize            │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│         DatasetConfigs              │
│  - Load datasets from files         │
│  - Cache datasets                   │
│  - Create SpectroDataset            │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│         PipelineRunner              │
│  - run(configs, datasets)           │
│  - predict(model_id, datasets)      │
│  - explain(model_id, datasets)      │
│  - Internal state management        │
└─────────────────────────────────────┘
```

**Current Usage:**
```python
# Configuration-centric approach
pipeline_config = PipelineConfigs([...])
dataset_config = DatasetConfigs('data/')
runner = PipelineRunner(verbose=1)
predictions, _ = runner.run(pipeline_config, dataset_config)
```

### 1.2 Issues with Current Design

1. **Not sklearn-compatible**: No `fit(X, y)` / `predict(X)` API
2. **Requires external configuration objects**: Can't use directly with numpy arrays
3. **File-centric**: Assumes data comes from files (DatasetConfigs)
4. **No parameter management**: Missing `get_params()` / `set_params()`
5. **Stateful in wrong way**: State tied to runner, not to a fitted model object
6. **No standard serialization**: Custom save/load mechanism

### 1.3 Dependencies to Refactor

**Core classes that need changes:**
- `PipelineRunner` → Becomes `Pipeline` (sklearn-compatible)
- `PipelineConfigs` → Becomes constructor parameter / internal
- `DatasetConfigs` → No longer needed (data from numpy arrays)
- `SpectroDataset` → Becomes internal state

**Classes that DON'T need changes:**
- Controllers (sklearn, tensorflow, torch)
- Operators (transformations, models)
- Predictions storage
- Serialization utilities

---

## 2. Target Architecture

### 2.1 New Class Structure

```
Target Design (sklearn-compatible):
┌─────────────────────────────────────┐
│           Pipeline                  │
│  (replaces PipelineRunner)          │
│                                     │
│  sklearn BaseEstimator API:         │
│    - __init__(steps, **params)      │
│    - fit(X, y)                      │
│    - predict(X)                     │
│    - transform(X)                   │
│    - get_params() / set_params()    │
│                                     │
│  Internal state:                    │
│    - _dataset: SpectroDataset       │
│    - _fitted_components: dict       │
│    - _context: dict                 │
└─────────────────────────────────────┘
```

### 2.2 Target Usage

```python
# sklearn-style API
from nirs4all.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

# Create pipeline
pipe = Pipeline([
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    PLSRegression(n_components=5)
], name="my_pipeline", cv=5, verbose=1)

# Fit with numpy arrays (sklearn API)
pipe.fit(X_train, y_train)

# Predict
y_pred = pipe.predict(X_test)

# Score
score = pipe.score(X_test, y_test)

# Works with sklearn tools
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X, y, cv=5)
```

### 2.3 Backward Compatibility Strategy

**Option A: Complete replacement (Breaking change)**
```python
# Old code breaks
from nirs4all.pipeline import PipelineRunner  # ImportError

# Must update to new API
from nirs4all.pipeline import Pipeline
```

**Option B: Deprecation path (Recommended)**
```python
# Old code still works but shows warning
from nirs4all.pipeline import PipelineRunner  # DeprecationWarning
runner = PipelineRunner()
runner.run(config, dataset_config)  # Still works

# New code uses new API
from nirs4all.pipeline import Pipeline
pipe = Pipeline([...])
pipe.fit(X, y)
```

**Option C: Dual API (Most compatible, most complex)**
```python
# Both APIs coexist
from nirs4all.pipeline import Pipeline

# New API
pipe = Pipeline([...])
pipe.fit(X, y)

# Legacy API (via adapter)
pipe = Pipeline([...])
pipe.run(pipeline_config, dataset_config)  # Internally converts
```

---

## 3. Refactoring Strategy

### 3.1 Phased Approach

**Phase 1: Foundation (Week 1-2)**
- Rename `PipelineRunner` → `Pipeline`
- Add sklearn base classes
- Implement basic `fit()` and `predict()`
- Keep internal logic unchanged

**Phase 2: API Completion (Week 3-4)**
- Implement `get_params()` / `set_params()`
- Add `transform()` for transformer pipelines
- Implement `score()`
- Add proper parameter validation

**Phase 3: Internal Cleanup (Week 5-6)**
- Remove `DatasetConfigs` dependency
- Streamline dataset creation from arrays
- Optimize state management
- Improve serialization

**Phase 4: Migration & Testing (Week 7-8)**
- Update all examples
- Update documentation
- Comprehensive testing
- Deprecation warnings for old API

### 3.2 Key Design Principles

1. **Minimal Breaking Changes**: Preserve internal logic as much as possible
2. **Clear Separation**: Public sklearn API vs internal implementation
3. **Backward Compatibility**: Support old code during transition
4. **Progressive Enhancement**: Can be done incrementally
5. **Test-Driven**: Every change backed by tests

---

## 4. Detailed Implementation Plan

### 4.1 Class Renaming and Inheritance

**Current:**
```python
class PipelineRunner:
    def __init__(self, max_workers=None, verbose=0, ...):
        # Many implementation-specific params

    def run(self, pipeline_configs, dataset_configs):
        # Configuration-centric
```

**Refactored:**
```python
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class Pipeline(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible pipeline for NIRS data analysis.

    Parameters
    ----------
    steps : list
        Pipeline steps (transformers, models, etc.)
    name : str, default="pipeline"
        Pipeline identifier
    cv : int or cross-validator, optional
        Cross-validation strategy
    verbose : int, default=0
        Verbosity level
    random_state : int, optional
        Random seed for reproducibility
    n_jobs : int, default=None
        Number of parallel jobs (-1 for all cores)
    """

    def __init__(
        self,
        steps,
        *,  # Force keyword-only arguments
        name="pipeline",
        cv=None,
        verbose=0,
        random_state=None,
        n_jobs=None,
        # Implementation details (prefixed with _)
        _save_files=False,
        _results_path=None,
    ):
        # Store all constructor parameters (required for get_params)
        self.steps = steps
        self.name = name
        self.cv = cv
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._save_files = _save_files
        self._results_path = _results_path

        # Internal state (not parameters, prefixed with _)
        self._is_fitted = False
        self._dataset = None
        self._fitted_components = {}
        self._context = {}
        self._predictions = None
```

### 4.2 Implementing fit()

**Core transformation: `run()` → `fit()`**

```python
class Pipeline(BaseEstimator, RegressorMixin):
    def fit(self, X, y=None, **fit_params):
        """
        Fit the pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can be:
            - 2D numpy array (standard)
            - 3D numpy array (multi-source: n_samples, n_sources, n_features)
            - List of 2D arrays (multi-source)
            - pandas DataFrame

        y : array-like of shape (n_samples,) or (n_samples, n_targets), optional
            Target values

        **fit_params : dict
            Additional fit parameters:
            - sample_weight : array-like of shape (n_samples,)
            - groups : array-like of shape (n_samples,)

        Returns
        -------
        self : object
            Fitted pipeline
        """
        # 1. Validate input
        if y is not None:
            X, y = check_X_y(
                X, y,
                accept_sparse=False,
                multi_output=True,
                y_numeric=True,
                dtype=np.float64
            )
        else:
            X = check_array(X, accept_sparse=False, dtype=np.float64)

        # 2. Initialize random state
        if self.random_state is not None:
            self._init_random_state(self.random_state)

        # 3. Store feature information
        if hasattr(X, 'columns'):
            self._feature_names_in = list(X.columns)
            X = X.values
        else:
            self._feature_names_in = None

        self.n_features_in_ = X.shape[1] if X.ndim == 2 else X.shape[2]

        # 4. Create internal dataset (replaces DatasetConfigs)
        self._dataset = self._create_dataset_from_arrays(X, y)

        # 5. Add cross-validation folds if specified
        if self.cv is not None:
            self._add_cv_folds(self._dataset, X, y)

        # 6. Create internal pipeline config (replaces PipelineConfigs)
        self._pipeline_config = self._create_pipeline_config(self.steps)

        # 7. Execute pipeline (internal logic, mostly unchanged)
        self._execute_fit()

        # 8. Mark as fitted
        self._is_fitted = True

        return self

    def _create_dataset_from_arrays(self, X, y):
        """Create SpectroDataset from numpy arrays."""
        from nirs4all.data import SpectroDataset

        dataset = SpectroDataset(name=self.name)

        # Handle different input formats
        if isinstance(X, list):
            # Multi-source: list of 2D arrays
            dataset.add_samples(X, headers=None)
        elif X.ndim == 3:
            # Multi-source: 3D array (n_samples, n_sources, n_features)
            X_list = [X[:, i, :] for i in range(X.shape[1])]
            dataset.add_samples(X_list, headers=None)
        else:
            # Single source: 2D array (n_samples, n_features)
            dataset.add_samples(X, headers=None)

        # Add targets if provided
        if y is not None:
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            dataset.add_targets(y)

        return dataset

    def _add_cv_folds(self, dataset, X, y):
        """Add cross-validation folds to dataset."""
        from sklearn.model_selection import check_cv

        # Get appropriate cross-validator
        cv = check_cv(self.cv, y, classifier=self._is_classifier())

        # Generate folds
        folds = list(cv.split(X, y))
        dataset.set_folds(folds)

    def _create_pipeline_config(self, steps):
        """Create internal PipelineConfigs from steps."""
        from nirs4all.pipeline.config import PipelineConfigs
        return PipelineConfigs(steps, name=self.name)

    def _execute_fit(self):
        """Execute the pipeline fitting (internal logic)."""
        # This is essentially the current run() logic
        # but without needing external config objects

        from nirs4all.pipeline.runner import PipelineRunner

        # Create internal runner (temporary during refactoring)
        runner = PipelineRunner(
            verbose=self.verbose,
            save_files=self._save_files,
            results_path=self._results_path,
            mode="train",
            random_state=self.random_state,
            max_workers=self.n_jobs if self.n_jobs else None
        )

        # Execute with internal dataset
        # This would be refactored to not need DatasetConfigs
        predictions, _ = runner._run_internal(
            self._pipeline_config,
            self._dataset
        )

        # Store results
        self._predictions = predictions
        self._fitted_components = runner.step_binaries
        self._context = runner.history.final_context if hasattr(runner.history, 'final_context') else {}
```

### 4.3 Implementing predict()

```python
class Pipeline(BaseEstimator, RegressorMixin):
    def predict(self, X):
        """
        Predict using the fitted pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values
        """
        # 1. Check if fitted
        check_is_fitted(self, '_is_fitted')

        # 2. Validate input
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        # 3. Check feature count matches
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but Pipeline was fitted with "
                f"{self.n_features_in_} features"
            )

        # 4. Create prediction dataset
        pred_dataset = self._create_dataset_from_arrays(X, y=None)

        # 5. Load fitted components
        from nirs4all.pipeline.binary_loader import BinaryLoader
        loader = BinaryLoader(
            simulation_path=None,
            step_binaries=self._fitted_components
        )

        # 6. Execute prediction
        runner = PipelineRunner(
            verbose=self.verbose,
            mode="predict"
        )

        # This needs refactoring to not require external configs
        y_pred = runner._predict_internal(
            dataset=pred_dataset,
            fitted_components=self._fitted_components,
            context=self._context
        )

        # 7. Return predictions
        return y_pred.ravel() if y_pred.shape[1] == 1 else y_pred
```

### 4.4 Implementing get_params() and set_params()

```python
class Pipeline(BaseEstimator, RegressorMixin):
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters for sub-objects (steps)

        Returns
        -------
        params : dict
            Parameter names mapped to their values
        """
        # Get constructor parameters
        params = {
            'steps': self.steps,
            'name': self.name,
            'cv': self.cv,
            'verbose': self.verbose,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
        }

        if deep:
            # Add parameters from pipeline steps
            # Format: steps__<step_idx>__<param_name>
            for idx, step in enumerate(self.steps):
                if isinstance(step, dict):
                    for key, value in step.items():
                        if hasattr(value, 'get_params'):
                            # Step is an estimator
                            step_params = value.get_params(deep=True)
                            for param_key, param_value in step_params.items():
                                params[f'steps__{idx}__{key}__{param_key}'] = param_value
                elif hasattr(step, 'get_params'):
                    # Step is an estimator
                    step_params = step.get_params(deep=True)
                    for param_key, param_value in step_params.items():
                        params[f'steps__{idx}__{param_key}'] = param_value

        return params

    def set_params(self, **params):
        """
        Set parameters for this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters

        Returns
        -------
        self : object
            Estimator instance
        """
        if not params:
            return self

        # Separate direct params from nested step params
        direct_params = {}
        step_params = {}

        for key, value in params.items():
            if '__' in key:
                # Nested parameter (e.g., steps__0__n_components)
                step_params[key] = value
            else:
                # Direct parameter
                direct_params[key] = value

        # Set direct parameters
        for key, value in direct_params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")

        # Set nested step parameters
        for key, value in step_params.items():
            parts = key.split('__')
            if parts[0] != 'steps':
                raise ValueError(f"Invalid nested parameter {key}")

            step_idx = int(parts[1])
            if step_idx >= len(self.steps):
                raise ValueError(f"Step index {step_idx} out of range")

            step = self.steps[step_idx]

            if isinstance(step, dict):
                # Dict step: {"model": EstimatorClass(), "name": "..."}
                step_key = parts[2]  # e.g., "model"
                param_key = '__'.join(parts[3:])  # remaining parts

                if step_key in step and hasattr(step[step_key], 'set_params'):
                    step[step_key].set_params(**{param_key: value})
            elif hasattr(step, 'set_params'):
                # Direct estimator step
                param_key = '__'.join(parts[2:])
                step.set_params(**{param_key: value})

        return self
```

### 4.5 Implementing transform() and fit_transform()

```python
class Pipeline(BaseEstimator, RegressorMixin):
    def transform(self, X):
        """
        Transform data using fitted pipeline.

        Only works if pipeline ends with a transformer, not a predictor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        Returns
        -------
        X_transformed : ndarray
            Transformed data
        """
        check_is_fitted(self, '_is_fitted')

        # Check if last step is a transformer
        if not self._is_transformer_pipeline():
            raise AttributeError(
                "Pipeline ending with predictor cannot transform. "
                "Use predict() instead."
            )

        X = check_array(X, accept_sparse=False, dtype=np.float64)

        # Execute transformation
        # This requires extracting features from the internal dataset
        # after applying all preprocessing steps

        pred_dataset = self._create_dataset_from_arrays(X, y=None)

        # Apply fitted transformers
        X_transformed = self._apply_transformers(pred_dataset)

        return X_transformed

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit and transform in one step.

        More efficient than calling fit() then transform() separately.
        """
        self.fit(X, y, **fit_params)

        # For fit_transform, we can directly extract transformed data
        # from the internal dataset

        if self._is_transformer_pipeline():
            return self._extract_transformed_features()
        else:
            raise AttributeError(
                "Pipeline ending with predictor cannot transform"
            )

    def _is_transformer_pipeline(self):
        """Check if pipeline ends with transformer."""
        # Check last step
        last_step = self.steps[-1]

        # Skip non-model steps (charts, CV, etc.)
        model_steps = [s for s in self.steps if self._is_model_step(s)]
        if not model_steps:
            return True  # Only transformers

        # Check if last model step is transformer
        last_model = model_steps[-1]
        if isinstance(last_model, dict):
            if 'model' in last_model:
                return hasattr(last_model['model'], 'transform')
        return hasattr(last_model, 'transform')
```

### 4.6 Implementing score()

```python
class Pipeline(BaseEstimator, RegressorMixin):
    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R² (or accuracy for classifiers).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            True values
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights

        Returns
        -------
        score : float
            R² of predictions for regression,
            Accuracy for classification
        """
        from sklearn.metrics import r2_score, accuracy_score

        y_pred = self.predict(X)

        if self._is_classifier():
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        else:
            return r2_score(y, y_pred, sample_weight=sample_weight, multioutput='uniform_average')

    def _is_classifier(self):
        """Check if pipeline is for classification."""
        if self._dataset is not None:
            return self._dataset.task_type in ['binary_classification', 'multiclass_classification', 'classification']
        return False
```

### 4.7 Standard Serialization

```python
class Pipeline(BaseEstimator, RegressorMixin):
    def __getstate__(self):
        """
        Prepare pipeline for pickling.

        This is called by pickle.dump() and must return a dict
        representing the pipeline state.
        """
        state = self.__dict__.copy()

        # Remove unpicklable objects if any
        # (e.g., file handles, locks)

        return state

    def __setstate__(self, state):
        """
        Restore pipeline from pickled state.

        This is called by pickle.load().
        """
        self.__dict__.update(state)

    # Standard pickle-based save/load
    def save(self, path):
        """
        Save fitted pipeline to file using pickle.

        Parameters
        ----------
        path : str or Path
            Output file path
        """
        import pickle
        check_is_fitted(self, '_is_fitted')

        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        """
        Load fitted pipeline from file.

        Parameters
        ----------
        path : str or Path
            Input file path

        Returns
        -------
        pipeline : Pipeline
            Loaded fitted pipeline
        """
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
```

---

## 5. Migration Path

### 5.1 Phase 1: New API Addition (Weeks 1-2)

**Goal**: Add new API without breaking existing code

```python
# File: nirs4all/pipeline/__init__.py

# New sklearn-compatible API
from .sklearn_pipeline import Pipeline

# Old API (still works)
from .runner import PipelineRunner  # Add deprecation warning
from .config import PipelineConfigs

__all__ = ['Pipeline', 'PipelineRunner', 'PipelineConfigs']
```

```python
# File: nirs4all/pipeline/sklearn_pipeline.py

class Pipeline(BaseEstimator, RegressorMixin):
    """New sklearn-compatible pipeline."""

    def __init__(self, steps, **kwargs):
        # Implementation as shown above
        pass

    def fit(self, X, y=None):
        # Implementation as shown above
        pass

    def predict(self, X):
        # Implementation as shown above
        pass

    # ... rest of implementation
```

**Testing:**
```python
# Old code still works
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs

config = PipelineConfigs([...])
dataset = DatasetConfigs('data/')
runner = PipelineRunner()
runner.run(config, dataset)

# New code works alongside
from nirs4all.pipeline import Pipeline

pipe = Pipeline([...])
pipe.fit(X, y)
```

### 5.2 Phase 2: Example Migration (Weeks 3-4)

**Goal**: Update all examples to use new API

```python
# Before (examples/Q1_regression.py)
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs

pipeline = [...]
config = PipelineConfigs(pipeline)
dataset = DatasetConfigs('sample_data/regression')
runner = PipelineRunner(verbose=1)
predictions, _ = runner.run(config, dataset)
```

```python
# After (examples/Q1_regression.py)
from nirs4all.pipeline import Pipeline
from nirs4all.data import load_dataset  # New helper

# Load data as arrays
X_train, X_test, y_train, y_test = load_dataset('sample_data/regression')

# Create and fit pipeline
pipeline = Pipeline([...], verbose=1)
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
score = pipeline.score(X_test, y_test)
print(f"R² score: {score:.3f}")
```

### 5.3 Phase 3: Deprecation (Weeks 5-6)

**Goal**: Mark old API as deprecated

```python
# File: nirs4all/pipeline/runner.py

import warnings

class PipelineRunner:
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "PipelineRunner is deprecated and will be removed in version 3.0. "
            "Use nirs4all.pipeline.Pipeline instead:\n"
            "  from nirs4all.pipeline import Pipeline\n"
            "  pipe = Pipeline(steps=[...])\n"
            "  pipe.fit(X, y)\n"
            "See migration guide: https://docs.nirs4all.org/migration",
            DeprecationWarning,
            stacklevel=2
        )
        # ... existing implementation
```

### 5.4 Phase 4: Removal (Version 3.0)

**Goal**: Remove old API completely

```python
# File: nirs4all/pipeline/__init__.py

from .sklearn_pipeline import Pipeline

# Old API removed
# from .runner import PipelineRunner  # REMOVED
# from .config import PipelineConfigs  # REMOVED (or made internal)

__all__ = ['Pipeline']
```

---

## 6. Code Examples

### 6.1 Basic Regression Pipeline

```python
from nirs4all.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from nirs4all.operators.transforms import StandardNormalVariate

# Create pipeline (nirs4all format still supported)
pipe = Pipeline([
    MinMaxScaler(),
    StandardNormalVariate(),
    PLSRegression(n_components=5)
], name="pls_pipeline", verbose=1)

# Fit (sklearn API)
pipe.fit(X_train, y_train)

# Predict
y_pred = pipe.predict(X_test)

# Score
r2 = pipe.score(X_test, y_test)
print(f"R² = {r2:.3f}")

# Save
pipe.save('pls_model.pkl')

# Load
pipe2 = Pipeline.load('pls_model.pkl')
```

### 6.2 Cross-Validation

```python
from sklearn.model_selection import cross_val_score

pipe = Pipeline([...], cv=5)

# Fit with internal CV
pipe.fit(X, y)

# Or use sklearn's CV
scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
print(f"CV scores: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### 6.3 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    MinMaxScaler(),
    StandardNormalVariate(),
    PLSRegression()
])

param_grid = {
    'steps__2__n_components': [3, 5, 10, 15, 20],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='r2')
grid.fit(X, y)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.3f}")

best_pipe = grid.best_estimator_
```

### 6.4 SHAP Analysis

```python
import shap

pipe = Pipeline([...])
pipe.fit(X_train, y_train)

# SHAP works directly
explainer = shap.Explainer(pipe, X_train)
shap_values = explainer(X_test)

shap.plots.waterfall(shap_values[0])
shap.plots.beeswarm(shap_values)
```

### 6.5 Pipeline Composition

```python
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.decomposition import PCA
from nirs4all.pipeline import Pipeline as NirsPipeline

# Create nirs4all preprocessing pipeline
nirs_prep = NirsPipeline([
    MinMaxScaler(),
    StandardNormalVariate(),
    SavitzkyGolay()
], name="preprocessing")

# Compose with sklearn steps
full_pipe = SklearnPipeline([
    ('nirs', nirs_prep),
    ('pca', PCA(n_components=20)),
    ('model', PLSRegression(n_components=5))
])

full_pipe.fit(X_train, y_train)
```

### 6.6 Ensemble Methods

```python
from sklearn.ensemble import VotingRegressor

pipe1 = Pipeline([...], name="pls")
pipe2 = Pipeline([...], name="rf")
pipe3 = Pipeline([...], name="nn")

ensemble = VotingRegressor([
    ('pls', pipe1),
    ('rf', pipe2),
    ('nn', pipe3)
])

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
```

---

## 7. Testing Strategy

### 7.1 Test Structure

```
tests/
├── unit/
│   ├── test_pipeline_api.py          # Test sklearn API compliance
│   ├── test_pipeline_params.py       # Test get_params/set_params
│   ├── test_pipeline_serialization.py # Test save/load
│   └── test_pipeline_validation.py   # Test input validation
├── integration/
│   ├── test_sklearn_compatibility.py # Test with sklearn tools
│   ├── test_shap_integration.py      # Test SHAP analysis
│   └── test_examples.py              # Test all examples still work
└── migration/
    ├── test_backward_compat.py       # Test old API still works
    └── test_deprecation_warnings.py  # Test warnings shown
```

### 7.2 Key Test Cases

```python
# tests/unit/test_pipeline_api.py

class TestPipelineAPI:
    def test_init_stores_params(self):
        """Test that __init__ stores all parameters."""
        pipe = Pipeline(
            steps=[MinMaxScaler()],
            name="test",
            cv=5,
            verbose=1
        )
        assert pipe.steps == [MinMaxScaler()]
        assert pipe.name == "test"
        assert pipe.cv == 5
        assert pipe.verbose == 1

    def test_fit_returns_self(self):
        """Test that fit() returns self (sklearn convention)."""
        pipe = Pipeline([MinMaxScaler()])
        result = pipe.fit(X, y)
        assert result is pipe

    def test_fit_sets_attributes(self):
        """Test that fit() sets required attributes."""
        pipe = Pipeline([MinMaxScaler()])
        pipe.fit(X, y)

        assert hasattr(pipe, 'n_features_in_')
        assert pipe.n_features_in_ == X.shape[1]

    def test_predict_checks_fitted(self):
        """Test that predict() checks if fitted."""
        pipe = Pipeline([MinMaxScaler()])

        with pytest.raises(NotFittedError):
            pipe.predict(X)

    def test_predict_checks_n_features(self):
        """Test that predict() checks feature count."""
        pipe = Pipeline([MinMaxScaler()])
        pipe.fit(X_train, y_train)

        X_wrong = np.random.randn(10, X_train.shape[1] + 1)
        with pytest.raises(ValueError, match="X has .* features"):
            pipe.predict(X_wrong)

    def test_get_params_shallow(self):
        """Test get_params(deep=False)."""
        pipe = Pipeline(
            steps=[MinMaxScaler()],
            name="test",
            cv=5
        )
        params = pipe.get_params(deep=False)

        assert params['name'] == "test"
        assert params['cv'] == 5

    def test_get_params_deep(self):
        """Test get_params(deep=True)."""
        pipe = Pipeline([
            MinMaxScaler(),
            PLSRegression(n_components=5)
        ])
        params = pipe.get_params(deep=True)

        assert 'steps__1__n_components' in params
        assert params['steps__1__n_components'] == 5

    def test_set_params(self):
        """Test set_params()."""
        pipe = Pipeline([PLSRegression(n_components=5)])
        pipe.set_params(steps__0__n_components=10)

        assert pipe.steps[0].n_components == 10

    def test_clone(self):
        """Test that pipeline can be cloned."""
        from sklearn.base import clone

        pipe = Pipeline([MinMaxScaler()], name="original")
        pipe_clone = clone(pipe)

        assert pipe_clone.name == "original"
        assert pipe_clone is not pipe
```

```python
# tests/integration/test_sklearn_compatibility.py

class TestSklearnCompatibility:
    def test_cross_val_score(self):
        """Test with cross_val_score."""
        from sklearn.model_selection import cross_val_score

        pipe = Pipeline([MinMaxScaler(), PLSRegression()])
        scores = cross_val_score(pipe, X, y, cv=5)

        assert len(scores) == 5
        assert all(np.isfinite(scores))

    def test_grid_search(self):
        """Test with GridSearchCV."""
        from sklearn.model_selection import GridSearchCV

        pipe = Pipeline([PLSRegression()])
        param_grid = {'steps__0__n_components': [3, 5, 10]}

        grid = GridSearchCV(pipe, param_grid, cv=3)
        grid.fit(X, y)

        assert hasattr(grid, 'best_estimator_')
        assert hasattr(grid, 'best_score_')

    def test_voting_regressor(self):
        """Test with VotingRegressor."""
        from sklearn.ensemble import VotingRegressor

        pipe1 = Pipeline([PLSRegression(n_components=5)])
        pipe2 = Pipeline([PLSRegression(n_components=10)])

        ensemble = VotingRegressor([('p1', pipe1), ('p2', pipe2)])
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)

        assert y_pred.shape == y_test.shape

    def test_pipeline_nesting(self):
        """Test nesting in sklearn Pipeline."""
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.decomposition import PCA

        nirs_pipe = Pipeline([MinMaxScaler()])
        full_pipe = SklearnPipeline([
            ('nirs', nirs_pipe),
            ('pca', PCA()),
            ('model', PLSRegression())
        ])

        full_pipe.fit(X_train, y_train)
        y_pred = full_pipe.predict(X_test)

        assert y_pred.shape == y_test.shape
```

---

## 8. Risk Assessment

### 8.1 Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing code | HIGH | HIGH | Phased migration with deprecation warnings |
| Performance regression | MEDIUM | MEDIUM | Benchmark before/after, optimize if needed |
| Feature loss | LOW | HIGH | Comprehensive testing, feature parity checklist |
| User confusion | MEDIUM | MEDIUM | Clear migration guide, dual documentation |
| Incomplete migration | MEDIUM | HIGH | Automated migration tools, clear timeline |

### 8.2 Mitigation Strategies

**Risk 1: Breaking Existing Code**
- Solution: Maintain old API for 2 major versions
- Timeline:
  - v2.0: New API added, old API works with warnings
  - v2.x: Both APIs coexist
  - v3.0: Old API removed

**Risk 2: Performance Regression**
- Solution: Benchmark critical paths
- Metrics: Fit time, predict time, memory usage
- Target: <5% regression acceptable

**Risk 3: Feature Loss**
- Solution: Feature parity checklist
- Requirements:
  - ✅ All preprocessing operators work
  - ✅ All model types supported (sklearn, tf, torch)
  - ✅ Multi-source data handling
  - ✅ Cross-validation
  - ✅ Prediction tracking
  - ✅ SHAP analysis

**Risk 4: User Confusion**
- Solution: Comprehensive documentation
- Deliverables:
  - Migration guide
  - Side-by-side comparison
  - Updated examples
  - Video tutorials

**Risk 5: Incomplete Migration**
- Solution: Automated tools + clear timeline
- Tools:
  - Migration script (updates imports)
  - API compatibility checker
  - Deprecation warnings with fix suggestions

### 8.3 Rollback Plan

If refactoring causes major issues:

```python
# Emergency rollback: Revert to old API as primary

# File: nirs4all/pipeline/__init__.py

# OLD (rollback state)
from .runner import PipelineRunner
from .config import PipelineConfigs

# NEW (marked as experimental)
from .sklearn_pipeline import Pipeline as ExperimentalPipeline

__all__ = ['PipelineRunner', 'PipelineConfigs', 'ExperimentalPipeline']
```

---

## 9. Comparison: Refactoring vs Wrapper

### 9.1 Side-by-Side Comparison

| Aspect | Refactoring Approach | Wrapper Approach |
|--------|---------------------|------------------|
| **Implementation** | Transform existing classes | Add new class that delegates |
| **Risk** | HIGH - Changes core code | LOW - Existing code unchanged |
| **Effort** | HIGH - 2-3 months | LOW - 1-2 weeks |
| **Architecture** | Clean - Native sklearn | Mixed - Dual layer |
| **Performance** | Optimal - Direct execution | Slight overhead from delegation |
| **Backward Compat** | Requires migration | Can coexist forever |
| **Maintenance** | Single codebase | Two codebases (old + new) |
| **Learning Curve** | Single API to learn | Two APIs to understand |
| **Testing** | Comprehensive rewrite needed | Focused on wrapper only |
| **Documentation** | Complete rewrite | Supplement existing docs |

### 9.2 Recommendation Matrix

**Use Refactoring If:**
- ✅ Long-term vision is pure sklearn compatibility
- ✅ Team has bandwidth for major project
- ✅ Users can tolerate migration period
- ✅ Architecture cleanup is high priority
- ✅ Planning major version bump (e.g., 2.0 → 3.0)

**Use Wrapper If:**
- ✅ Need quick sklearn compatibility
- ✅ Cannot break existing code
- ✅ Limited development resources
- ✅ Risk-averse environment
- ✅ Want incremental adoption

### 9.3 Hybrid Approach (Recommended)

**Best of both worlds:**

1. **Phase 1** (Months 1-2): Implement wrapper
   - Quick win: sklearn compatibility NOW
   - Low risk: Existing code unaffected
   - User feedback: Learn what works/doesn't

2. **Phase 2** (Months 3-6): Evaluate feedback
   - Is wrapper sufficient?
   - What patterns emerge?
   - Where are pain points?

3. **Phase 3** (Months 7-12): Decide on refactoring
   - If wrapper works: Keep it, maybe optimize
   - If issues found: Plan strategic refactoring
   - If highly successful: Consider full refactoring for v3.0

**This approach minimizes risk while keeping options open.**

---

## 10. Conclusion

### 10.1 Summary

The refactoring approach transforms nirs4all into a first-class sklearn citizen by:

1. **Renaming**: `PipelineRunner` → `Pipeline`
2. **Inheriting**: From `BaseEstimator`, `RegressorMixin`
3. **Implementing**: `fit()`, `predict()`, `score()`, `get_params()`, `set_params()`
4. **Simplifying**: Remove external config objects, work directly with numpy arrays
5. **Standardizing**: Use pickle for serialization, follow sklearn conventions

### 10.2 Key Benefits

✅ **Native sklearn integration** - Not a wrapper, IS sklearn-compatible
✅ **Cleaner architecture** - Single unified API
✅ **Better performance** - No delegation overhead
✅ **Easier maintenance** - One codebase, not two
✅ **Future-proof** - Aligned with sklearn ecosystem

### 10.3 Key Challenges

⚠️ **Breaking changes** - Requires user migration
⚠️ **High effort** - 2-3 months of development
⚠️ **Testing complexity** - Must maintain feature parity
⚠️ **Documentation burden** - Complete rewrite needed
⚠️ **Risk** - Core code changes can introduce bugs

### 10.4 Final Recommendation

**Start with the wrapper approach** (see `PIPELINE_AS_ESTIMATOR_ANALYSIS.md`), then **evaluate refactoring for a future major version** based on:

- User adoption of wrapper API
- Identified performance bottlenecks
- Community feedback
- Team resources

The wrapper gives you 90% of the benefits with 10% of the risk. Refactoring can wait until you're confident in the API design and have validated it with real users.

---

## Appendices

### A. Complete File Structure After Refactoring

```
nirs4all/pipeline/
├── __init__.py           # Exports Pipeline (new main API)
├── pipeline.py           # NEW: sklearn-compatible Pipeline class
├── runner.py             # DEPRECATED: Old PipelineRunner (with warnings)
├── config.py             # Internal: PipelineConfigs (no longer user-facing)
├── operation.py          # Internal: No changes needed
├── serialization.py      # Internal: No changes needed
├── io.py                 # Internal: No changes needed
├── binary_loader.py      # Internal: No changes needed
└── history.py            # Internal: No changes needed
```

### B. Migration Checklist

- [ ] Implement `Pipeline` class with sklearn bases
- [ ] Implement `fit(X, y)` method
- [ ] Implement `predict(X)` method
- [ ] Implement `score(X, y)` method
- [ ] Implement `get_params(deep)` method
- [ ] Implement `set_params(**params)` method
- [ ] Implement `transform(X)` for transformer pipelines
- [ ] Implement `fit_transform(X, y)` optimization
- [ ] Add proper input validation
- [ ] Add deprecation warnings to old API
- [ ] Update all examples
- [ ] Update documentation
- [ ] Create migration guide
- [ ] Create automated migration script
- [ ] Comprehensive testing (100+ tests)
- [ ] Performance benchmarking
- [ ] Release v2.0 with both APIs
- [ ] Monitor feedback
- [ ] Plan v3.0 (remove old API)

### C. Resources

- sklearn Estimator development guide: https://scikit-learn.org/stable/developers/develop.html
- sklearn Pipeline source: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/pipeline.py
- Rolling your own estimator: https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
- Parameter routing: https://scikit-learn.org/stable/metadata_routing.html

---

**End of Document**
