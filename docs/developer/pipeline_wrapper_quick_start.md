# Quick Start: Making nirs4all sklearn-compatible

**TL;DR**: Create a `Nirs4allPipeline` wrapper class that implements sklearn's API while delegating to the existing nirs4all infrastructure.

## The Problem

Currently, nirs4all pipelines are NOT sklearn-compatible:

```python
# Current (verbose, not sklearn-compatible)
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs

config = PipelineConfigs([...steps...])
dataset_config = DatasetConfigs('data/')
runner = PipelineRunner()
predictions, _ = runner.run(config, dataset_config)
```

This means:
- ❌ Can't use with `cross_val_score`
- ❌ Can't use with `GridSearchCV`
- ❌ Can't use with SHAP directly
- ❌ Can't nest in sklearn Pipelines
- ❌ Can't save as single file

## The Solution

Create a wrapper that works like this:

```python
# Desired (sklearn-compatible)
from nirs4all.pipeline import Nirs4allPipeline

pipe = Nirs4allPipeline([...steps...])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Now works with sklearn tools!
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X, y, cv=5)

# And with SHAP!
import shap
explainer = shap.Explainer(pipe, X_train)
```

## Implementation Overview

### Core Class Structure

```python
from sklearn.base import BaseEstimator, RegressorMixin

class Nirs4allPipeline(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper for nirs4all pipelines."""

    def __init__(self, steps, name="pipeline", cv=None, verbose=0):
        # Store constructor params (required for get_params/set_params)
        self.steps = steps
        self.name = name
        self.cv = cv
        self.verbose = verbose

        # Internal state (set during fit)
        self._is_fitted = False
        self._dataset = None
        self._fitted_components = {}

    def fit(self, X, y=None):
        """Fit the pipeline using sklearn API."""
        # 1. Convert X, y to internal SpectroDataset
        from nirs4all.data import SpectroDataset
        self._dataset = SpectroDataset(name="sklearn_data")
        self._dataset.add_samples(X)
        if y is not None:
            self._dataset.add_targets(y)

        # 2. Create PipelineConfigs from steps
        from nirs4all.pipeline import PipelineConfigs
        config = PipelineConfigs(self.steps, name=self.name)

        # 3. Execute with PipelineRunner
        from nirs4all.pipeline import PipelineRunner
        from nirs4all.data import DatasetConfigs

        runner = PipelineRunner(save_files=False, verbose=self.verbose)
        dataset_config = DatasetConfigs.from_dataset(self._dataset)

        predictions, _ = runner.run(config, dataset_config)

        # 4. Store fitted components
        self._fitted_components = runner.step_binaries
        self._is_fitted = True

        return self

    def predict(self, X):
        """Predict using fitted pipeline."""
        from sklearn.utils.validation import check_is_fitted, check_array
        check_is_fitted(self, '_is_fitted')
        X = check_array(X)

        # Create prediction dataset
        pred_dataset = SpectroDataset(name="pred_data")
        pred_dataset.add_samples(X)

        # Load binaries and predict
        from nirs4all.pipeline import PipelineRunner
        runner = PipelineRunner(mode="predict", verbose=self.verbose)
        # ... execute prediction with loaded binaries ...

        return y_pred

    def save(self, path):
        """Save fitted pipeline to file."""
        import pickle
        state = {
            'steps': self.steps,
            'name': self.name,
            'fitted_components': self._fitted_components,
            # ... other state ...
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path):
        """Load fitted pipeline from file."""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
        pipe = cls(steps=state['steps'], name=state['name'])
        pipe._fitted_components = state['fitted_components']
        pipe._is_fitted = True
        return pipe
```

## Key Design Decisions

### 1. Wrapper Pattern (Not Rewrite)
✅ **Use existing code** - Minimal changes to current system
✅ **Low risk** - Existing tests still pass
✅ **Incremental** - Can be developed step-by-step

### 2. Internal Dataset Management
✅ **Create `SpectroDataset` automatically** from numpy arrays
✅ **Hide complexity** from users
✅ **Preserve features** like multi-source data handling

### 3. Stateful Binary Storage
✅ **Store fitted components** in memory during fit
✅ **Load for prediction** without file I/O
✅ **Single file save/load** using pickle

## Benefits Achieved

### 1. Single Estimator Interface ✅
```python
pipe = Nirs4allPipeline([...])
pipe.fit(X, y)        # Standard sklearn
y_pred = pipe.predict(X)
```

### 2. Works with sklearn Tools ✅
```python
# Cross-validation
scores = cross_val_score(pipe, X, y, cv=5)

# Hyperparameter tuning
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X, y)

# Ensembles
ensemble = VotingRegressor([('pipe1', pipe1), ('pipe2', pipe2)])
```

### 3. SHAP Compatible ✅
```python
import shap
explainer = shap.Explainer(pipe, X_train)
shap_values = explainer(X_test)
shap.plots.waterfall(shap_values[0])
```

### 4. Single Binary Distribution ✅
```python
# Save
pipe.save('model.pkl')

# Load anywhere
pipe = Nirs4allPipeline.load('model.pkl')
y_pred = pipe.predict(X_new)
```

### 5. Composable ✅
```python
# Use as component in larger pipelines
from sklearn.pipeline import Pipeline

meta_pipe = Pipeline([
    ('nirs_prep', Nirs4allPipeline([...preprocessing...])),
    ('model', RandomForestRegressor())
])
```

## Implementation Phases

### Phase 1: Core Wrapper (Week 1-2)
- [ ] Create `Nirs4allPipeline` class
- [ ] Implement `fit()` and `predict()`
- [ ] Basic serialization
- [ ] Unit tests

### Phase 2: sklearn Integration (Week 3-4)
- [ ] Full `get_params/set_params` support
- [ ] `transform()` for transformer pipelines
- [ ] Test with `cross_val_score`, `GridSearchCV`
- [ ] Integration tests

### Phase 3: Advanced Features (Week 5-6)
- [ ] SHAP integration
- [ ] ONNX export (optional)
- [ ] Multi-source data handling
- [ ] Performance optimization

### Phase 4: Polish & Release (Week 7-8)
- [ ] Documentation
- [ ] Examples
- [ ] API stabilization
- [ ] Release v1.0

## Quick Wins

You can achieve 80% of the value with just Phase 1:

```python
# File: nirs4all/pipeline/sklearn_wrapper.py
# ~200 lines of code

class Nirs4allPipeline(BaseEstimator, RegressorMixin):
    def __init__(self, steps, **kwargs):
        self.steps = steps
        # ... store all params ...

    def fit(self, X, y):
        # Create internal dataset
        # Run existing pipeline
        # Store binaries
        return self

    def predict(self, X):
        # Load binaries
        # Run prediction
        return y_pred

    def save/load(self):
        # Pickle entire state
```

This unlocks:
- ✅ sklearn cross-validation
- ✅ sklearn hyperparameter tuning
- ✅ Basic SHAP analysis
- ✅ Single-file distribution

## Example Use Cases

### Use Case 1: SHAP Analysis
```python
from nirs4all.pipeline import Nirs4allPipeline
import shap

# Train model
pipe = Nirs4allPipeline([
    MinMaxScaler(),
    StandardNormalVariate(),
    PLSRegression(n_components=10)
])
pipe.fit(X_train, y_train)

# Explain predictions
explainer = shap.Explainer(pipe, X_train)
shap_values = explainer(X_test)

# Visualize
shap.plots.waterfall(shap_values[0])
shap.plots.beeswarm(shap_values)
```

### Use Case 2: Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

pipe = Nirs4allPipeline([...])

param_grid = {
    'steps__model__n_components': [3, 5, 10, 15],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X, y)

print(f"Best params: {grid.best_params_}")
```

### Use Case 3: Model Distribution
```python
# Team member 1: Train and share
pipe = Nirs4allPipeline([...])
pipe.fit(X_train, y_train)
pipe.save('best_model.pkl')

# Team member 2: Load and use
pipe = Nirs4allPipeline.load('best_model.pkl')
y_pred = pipe.predict(X_new)
```

## Next Steps

1. **Review** the detailed analysis in `PIPELINE_AS_ESTIMATOR_ANALYSIS.md`
2. **Prototype** the basic wrapper class
3. **Test** with existing nirs4all examples (Q1, Q2, etc.)
4. **Iterate** based on what works/doesn't work
5. **Release** as experimental feature

## References

- Full analysis: `docs/PIPELINE_AS_ESTIMATOR_ANALYSIS.md`
- Sklearn Pipeline docs: https://scikit-learn.org/stable/modules/compose.html
- SHAP docs: https://shap.readthedocs.io/
- ONNX sklearn: https://onnx.ai/sklearn-onnx/
