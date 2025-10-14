# Pipeline Refactoring - Quick Start Guide

**TL;DR**: Transform `PipelineRunner` into a sklearn-compatible `Pipeline` class by implementing the sklearn estimator API directly in the core classes.

---

## The Transformation

### Before (Current):
```python
# Configuration-centric, not sklearn-compatible
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.dataset import DatasetConfigs

config = PipelineConfigs([MinMaxScaler(), PLSRegression()])
dataset = DatasetConfigs('data/')
runner = PipelineRunner(verbose=1)
predictions, _ = runner.run(config, dataset)
```

### After (Refactored):
```python
# sklearn-compatible, works directly with arrays
from nirs4all.pipeline import Pipeline

pipe = Pipeline([MinMaxScaler(), PLSRegression()], verbose=1)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
score = pipe.score(X_test, y_test)
```

---

## Key Changes

### 1. Rename and Inherit

**Before:**
```python
class PipelineRunner:
    def __init__(self, max_workers=None, verbose=0, ...):
        pass
```

**After:**
```python
from sklearn.base import BaseEstimator, RegressorMixin

class Pipeline(BaseEstimator, RegressorMixin):
    def __init__(self, steps, *, name="pipeline", cv=None, verbose=0,
                 random_state=None, n_jobs=None):
        # Store ALL constructor params (required for get_params)
        self.steps = steps
        self.name = name
        self.cv = cv
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Internal state (prefixed with _)
        self._is_fitted = False
        self._dataset = None
        self._fitted_components = {}
```

### 2. Transform run() → fit()

**Before:**
```python
def run(self, pipeline_configs: PipelineConfigs,
        dataset_configs: DatasetConfigs) -> Any:
    # Load datasets from configs
    # Execute pipeline
    # Return predictions
```

**After:**
```python
def fit(self, X, y=None, **fit_params):
    """Fit the pipeline (sklearn API)."""
    # 1. Validate input
    X, y = check_X_y(X, y, multi_output=True)

    # 2. Create internal dataset from arrays
    self._dataset = self._create_dataset_from_arrays(X, y)

    # 3. Add CV folds if specified
    if self.cv is not None:
        self._add_cv_folds(self._dataset, X, y)

    # 4. Execute pipeline (internal logic mostly unchanged)
    self._execute_fit()

    # 5. Mark as fitted
    self._is_fitted = True
    self.n_features_in_ = X.shape[1]

    return self
```

### 3. Add predict()

**New method:**
```python
def predict(self, X):
    """Predict using fitted pipeline (sklearn API)."""
    # 1. Check if fitted
    check_is_fitted(self, '_is_fitted')

    # 2. Validate input
    X = check_array(X)

    # 3. Create prediction dataset
    pred_dataset = self._create_dataset_from_arrays(X, y=None)

    # 4. Load fitted components and execute
    y_pred = self._execute_predict(pred_dataset)

    return y_pred.ravel() if y_pred.shape[1] == 1 else y_pred
```

### 4. Implement Parameter Management

**New methods:**
```python
def get_params(self, deep=True):
    """Get parameters (sklearn API)."""
    params = {
        'steps': self.steps,
        'name': self.name,
        'cv': self.cv,
        'verbose': self.verbose,
        'random_state': self.random_state,
        'n_jobs': self.n_jobs,
    }

    if deep:
        # Add nested step parameters
        # Format: steps__<idx>__<param>
        for idx, step in enumerate(self.steps):
            if hasattr(step, 'get_params'):
                step_params = step.get_params(deep=True)
                for key, value in step_params.items():
                    params[f'steps__{idx}__{key}'] = value

    return params

def set_params(self, **params):
    """Set parameters (sklearn API)."""
    # Separate direct params from nested params
    for key, value in params.items():
        if '__' in key:
            # Nested: steps__0__n_components
            self._set_nested_param(key, value)
        else:
            # Direct: verbose, cv, etc.
            setattr(self, key, value)

    return self
```

### 5. Add score()

**New method:**
```python
def score(self, X, y, sample_weight=None):
    """Score predictions (sklearn API)."""
    from sklearn.metrics import r2_score, accuracy_score

    y_pred = self.predict(X)

    if self._is_classifier():
        return accuracy_score(y, y_pred, sample_weight=sample_weight)
    else:
        return r2_score(y, y_pred, sample_weight=sample_weight)
```

---

## Internal Helper Methods

### Create Dataset from Arrays

```python
def _create_dataset_from_arrays(self, X, y):
    """Convert numpy arrays to internal SpectroDataset."""
    from nirs4all.dataset import SpectroDataset

    dataset = SpectroDataset(name=self.name)

    # Handle different formats
    if isinstance(X, list):
        # Multi-source: list of 2D arrays
        dataset.add_samples(X)
    elif X.ndim == 3:
        # Multi-source: 3D array
        X_list = [X[:, i, :] for i in range(X.shape[1])]
        dataset.add_samples(X_list)
    else:
        # Single source: 2D array
        dataset.add_samples(X)

    if y is not None:
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        dataset.add_targets(y)

    return dataset
```

### Add Cross-Validation Folds

```python
def _add_cv_folds(self, dataset, X, y):
    """Add CV folds to dataset."""
    from sklearn.model_selection import check_cv

    cv = check_cv(self.cv, y, classifier=self._is_classifier())
    folds = list(cv.split(X, y))
    dataset.set_folds(folds)
```

---

## Migration Strategy

### Phase 1: Add New API (Weeks 1-2)

Create new `Pipeline` class alongside existing `PipelineRunner`:

```python
# nirs4all/pipeline/__init__.py

# New API
from .sklearn_pipeline import Pipeline

# Old API (with deprecation warning)
from .runner import PipelineRunner
from .config import PipelineConfigs

__all__ = ['Pipeline', 'PipelineRunner', 'PipelineConfigs']
```

Both APIs work simultaneously:

```python
# Old code still works
runner = PipelineRunner()
runner.run(config, dataset)

# New code works alongside
pipe = Pipeline([...])
pipe.fit(X, y)
```

### Phase 2: Deprecate Old API (Weeks 3-4)

Add deprecation warnings:

```python
# nirs4all/pipeline/runner.py

class PipelineRunner:
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "PipelineRunner is deprecated. Use Pipeline instead:\n"
            "  from nirs4all.pipeline import Pipeline\n"
            "  pipe = Pipeline(steps=[...])\n"
            "  pipe.fit(X, y)",
            DeprecationWarning,
            stacklevel=2
        )
        # ... existing code
```

### Phase 3: Update Examples (Weeks 5-6)

Convert all examples to new API:

```python
# Before: examples/Q1_regression.py
config = PipelineConfigs([...])
dataset = DatasetConfigs('data/')
runner = PipelineRunner()
predictions, _ = runner.run(config, dataset)

# After: examples/Q1_regression.py
from nirs4all.dataset import load_dataset

X_train, X_test, y_train, y_test = load_dataset('data/')
pipe = Pipeline([...])
pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
```

### Phase 4: Remove Old API (Version 3.0)

In next major version, remove old API completely:

```python
# nirs4all/pipeline/__init__.py

from .sklearn_pipeline import Pipeline

# PipelineRunner REMOVED
# PipelineConfigs REMOVED

__all__ = ['Pipeline']
```

---

## Benefits of Refactoring

### 1. Native sklearn Compatibility ✅

```python
# Works with all sklearn tools
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import VotingRegressor

# Cross-validation
scores = cross_val_score(pipe, X, y, cv=5)

# Hyperparameter tuning
grid = GridSearchCV(pipe, param_grid, cv=5)

# Ensembles
ensemble = VotingRegressor([('p1', pipe1), ('p2', pipe2)])
```

### 2. Direct SHAP Analysis ✅

```python
import shap

pipe = Pipeline([...])
pipe.fit(X_train, y_train)

explainer = shap.Explainer(pipe, X_train)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)
```

### 3. Standard Serialization ✅

```python
import pickle

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(pipe, f)

# Load
with open('model.pkl', 'rb') as f:
    pipe = pickle.load(f)
```

### 4. Cleaner Architecture ✅

- Single API instead of multiple config objects
- Consistent with sklearn conventions
- Easier to learn and use
- Better documentation alignment

### 5. Composability ✅

```python
# Nest in sklearn pipelines
from sklearn.pipeline import Pipeline as SklearnPipeline

full_pipe = SklearnPipeline([
    ('nirs', Pipeline([...])),  # nirs4all pipeline
    ('pca', PCA()),              # sklearn transformer
    ('model', RandomForest())    # sklearn model
])
```

---

## Comparison: Wrapper vs Refactoring

| Aspect | Wrapper | Refactoring |
|--------|---------|-------------|
| **Time to implement** | 1-2 weeks | 2-3 months |
| **Risk** | LOW | HIGH |
| **Backward compatibility** | Perfect | Requires migration |
| **Architecture** | Two layers | Single clean layer |
| **Performance** | Slight overhead | Optimal |
| **Maintenance** | Two codebases | One codebase |
| **User migration** | Optional | Required |

---

## Recommended Approach: Hybrid

**Best strategy:** Start with wrapper, migrate to refactoring

### Step 1: Implement Wrapper (Month 1)
- Quick win: sklearn compatibility NOW
- Low risk: existing code untouched
- Get user feedback

### Step 2: Gather Feedback (Months 2-3)
- How is wrapper used?
- Any performance issues?
- What features are most valuable?

### Step 3: Decide (Month 4)
- **If wrapper sufficient**: Keep it, optimize as needed
- **If issues found**: Plan strategic refactoring
- **If highly successful**: Full refactoring for v3.0

---

## Implementation Checklist

### Core API
- [ ] Create `Pipeline` class inheriting from `BaseEstimator`
- [ ] Implement `__init__` storing all parameters
- [ ] Implement `fit(X, y)` method
- [ ] Implement `predict(X)` method
- [ ] Implement `score(X, y)` method
- [ ] Implement `get_params(deep)` method
- [ ] Implement `set_params(**params)` method
- [ ] Implement `transform(X)` for transformer pipelines
- [ ] Add input validation (`check_X_y`, `check_array`)
- [ ] Add fitted check (`check_is_fitted`)

### Internal Methods
- [ ] `_create_dataset_from_arrays(X, y)`
- [ ] `_add_cv_folds(dataset, X, y)`
- [ ] `_execute_fit()` (refactor from `run()`)
- [ ] `_execute_predict(dataset)` (refactor from `predict()`)
- [ ] `_is_classifier()` helper
- [ ] `_is_transformer_pipeline()` helper

### Serialization
- [ ] Implement `__getstate__()` for pickle
- [ ] Implement `__setstate__()` for pickle
- [ ] Add `save(path)` convenience method
- [ ] Add `load(path)` class method

### Backward Compatibility
- [ ] Keep `PipelineRunner` with deprecation warning
- [ ] Create migration guide document
- [ ] Update all examples to new API
- [ ] Create automated migration script

### Testing
- [ ] Test sklearn API compliance (100+ tests)
- [ ] Test with `cross_val_score`
- [ ] Test with `GridSearchCV`
- [ ] Test with `VotingRegressor`
- [ ] Test SHAP integration
- [ ] Test save/load roundtrip
- [ ] Test parameter management
- [ ] Performance benchmarks

### Documentation
- [ ] Complete API reference
- [ ] Migration guide
- [ ] Updated examples
- [ ] Updated tutorials
- [ ] Comparison guide (old vs new API)

---

## Quick Implementation Template

```python
# File: nirs4all/pipeline/sklearn_pipeline.py

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np

class Pipeline(BaseEstimator, RegressorMixin):
    def __init__(self, steps, *, name="pipeline", cv=None, verbose=0,
                 random_state=None, n_jobs=None):
        # Store parameters
        self.steps = steps
        self.name = name
        self.cv = cv
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Internal state
        self._is_fitted = False
        self._dataset = None
        self._fitted_components = {}

    def fit(self, X, y=None, **fit_params):
        # Validate
        if y is not None:
            X, y = check_X_y(X, y, multi_output=True)
        else:
            X = check_array(X)

        # Create dataset
        self._dataset = self._create_dataset_from_arrays(X, y)

        # Add CV folds
        if self.cv is not None:
            self._add_cv_folds(self._dataset, X, y)

        # Execute pipeline
        self._execute_fit()

        # Mark fitted
        self._is_fitted = True
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        check_is_fitted(self, '_is_fitted')
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features")

        pred_dataset = self._create_dataset_from_arrays(X, y=None)
        y_pred = self._execute_predict(pred_dataset)

        return y_pred.ravel() if y_pred.shape[1] == 1 else y_pred

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def _create_dataset_from_arrays(self, X, y):
        from nirs4all.dataset import SpectroDataset
        dataset = SpectroDataset(name=self.name)
        dataset.add_samples(X)
        if y is not None:
            dataset.add_targets(y.reshape(-1, 1) if y.ndim == 1 else y)
        return dataset

    def _add_cv_folds(self, dataset, X, y):
        from sklearn.model_selection import check_cv
        cv = check_cv(self.cv, y)
        folds = list(cv.split(X, y))
        dataset.set_folds(folds)

    def _execute_fit(self):
        # TODO: Refactor PipelineRunner logic here
        pass

    def _execute_predict(self, dataset):
        # TODO: Refactor prediction logic here
        pass
```

---

## Next Steps

1. **Read full analysis**: `PIPELINE_REFACTORING_ANALYSIS.md`
2. **Compare with wrapper**: `PIPELINE_AS_ESTIMATOR_ANALYSIS.md`
3. **Decide approach**: Wrapper first (recommended) or direct refactoring
4. **Start prototyping**: Implement `Pipeline` class
5. **Test thoroughly**: Ensure feature parity
6. **Gather feedback**: Before committing to approach
7. **Plan migration**: Clear timeline and communication

---

## Summary

**Refactoring transforms nirs4all from a configuration-driven system to a sklearn-native library.**

- ✅ More Pythonic: `pipe.fit(X, y)` vs `runner.run(config, dataset)`
- ✅ Better integration: Works seamlessly with sklearn ecosystem
- ✅ Cleaner code: Single API, not multiple config objects
- ✅ Future-proof: Aligned with ML community standards

**But it requires:**
- ⚠️ User migration effort
- ⚠️ Extensive testing
- ⚠️ Documentation rewrite
- ⚠️ Several months of development

**Recommendation**: Start with wrapper (quick win), evaluate, then consider full refactoring for v3.0 if validated by users.
