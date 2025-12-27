# NIRS4ALL API Design v2: Module-Level Functions & sklearn Meta-Estimator

## Document Information

- **Author**: GitHub Copilot (Claude Opus 4.5)
- **Date**: December 24, 2025
- **Status**: Draft Proposal (Reviewed)
- **Scope**: Public API redesign for nirs4all 0.6+
- **Companion Document**: [api_v2_migration_roadmap.md](api_v2_migration_roadmap.md) - Critical review and implementation roadmap

> **Note**: This document has been updated based on the critical review in the migration roadmap.
> Key changes: thin wrapper pattern for `run()`, `NIRSPipeline.from_result()` class method,
> `NIRSPipelineSearch` dropped (generator syntax provides this).

---

## 1. Reformulation of Objectives

### 1.1 Problem Statement

The current nirs4all API requires users to understand and instantiate multiple objects:

```python
# Current API (verbose)
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs

pipeline_config = PipelineConfigs(pipeline, "my_pipeline")
dataset_config = DatasetConfigs("path/to/data")
runner = PipelineRunner(verbose=1, save_artifacts=True, ...)
predictions, _ = runner.run(pipeline_config, dataset_config)

# Prediction
y_pred, _ = runner.predict(best_model, new_data)

# Explanation
shap_results, _ = runner.explain(best_model, data, shap_params=...)
```

This approach has several friction points:

1. **Cognitive overhead**: Users must understand `PipelineConfigs`, `DatasetConfigs`, and `PipelineRunner` as separate concepts
2. **Boilerplate**: Every script requires the same import/instantiation pattern
3. **Documentation fragmentation**: Features are spread across multiple class docstrings
4. **sklearn incompatibility**: Cannot use the pipeline directly with sklearn tools (GridSearchCV, SHAP, etc.)

### 1.2 Goals

**Goal 1: Module-Level "Service" API**
- Provide `nirs4all.run()`, `nirs4all.predict()`, `nirs4all.explain()`, `nirs4all.retrain()` as the primary public interface
- Support both simple (config-first) and advanced (session-based) usage patterns
- Maintain full power of current API while reducing surface complexity

**Goal 2: sklearn-Compatible Meta-Estimator**
- Create a `NIRSPipeline` class that implements sklearn's `BaseEstimator` interface
- Enable direct usage with SHAP, Optuna, cross_validate, GridSearchCV
- Support `fit()`, `predict()`, `transform()`, and optionally `partial_fit()`
- Provide proper model exposure for gradient-based SHAP on neural networks

### 1.3 Non-Goals

- Breaking backward compatibility with `PipelineRunner` (it remains available)
- Replacing the controller registry architecture
- Modifying the internal execution flow

---

## 2. Design Analysis: Module-Level API

### 2.1 What We Gain

| Benefit | Explanation |
|---------|-------------|
| **Single entry point** | `nirs4all.run(...)` is discoverable and self-documenting |
| **Reduced boilerplate** | No need to import/instantiate 3 classes |
| **Config-first design** | Configs become the contract, functions are verbs |
| **Stable public surface** | Internal refactoring doesn't break callers |
| **CLI/UI friendly** | Maps naturally to command-line verbs |

### 2.2 What We Risk

| Risk | Mitigation |
|------|------------|
| **Hidden global state** | Explicit defaults via `get_default_config()`, no silent singletons |
| **Repeated setup cost** | Session pattern for repeated calls |
| **Advanced features harder** | Accept `session=` parameter for power users |
| **Testing complexity** | Provide `nirs4all.reset_defaults()` for test isolation |

### 2.3 Recommended Design: Facade Pattern

The module-level functions are thin facades over the existing `PipelineRunner`:

```python
# nirs4all/__init__.py (additions)

def run(
    pipeline: PipelineSpec,
    dataset: DatasetSpec,
    *,
    config: Optional[RunConfig] = None,
    session: Optional[Session] = None,
    **overrides
) -> RunResult:
    """Execute a training pipeline.

    This is the primary entry point for training ML pipelines on NIRS data.

    Args:
        pipeline: Pipeline definition. Can be:
            - List of pipeline steps (operators, models, etc.)
            - Path to pipeline config file (.yaml, .json)
            - Dict with 'pipeline' key
            - PipelineConfigs instance (backward compat)

        dataset: Dataset definition. Can be:
            - Path to data directory or file
            - Tuple of (X, y) arrays
            - Dict with 'train_x', 'train_y' etc.
            - DatasetConfigs instance (backward compat)

        config: Optional RunConfig with execution settings.
            If None, uses sensible defaults.

        session: Optional Session for resource reuse across calls.
            If None, creates an ephemeral session.

        **overrides: Override specific config values:
            verbose, save_artifacts, save_charts, random_state, etc.

    Returns:
        RunResult containing:
            - predictions: Predictions object with all results
            - best: Best prediction entry (shortcut)
            - artifacts: Path to saved artifacts
            - metrics: Summary metrics dict

    Examples:
        >>> # Simple usage
        >>> result = nirs4all.run(
        ...     pipeline=[MinMaxScaler(), PLSRegression(10)],
        ...     dataset="sample_data/wheat"
        ... )
        >>> print(result.best['rmse'])

        >>> # With config
        >>> result = nirs4all.run(
        ...     pipeline=[SNV(), KFold(5), {"model": PLSRegression(10)}],
        ...     dataset={"train_x": X, "train_y": y},
        ...     config=RunConfig(verbose=2, save_artifacts=True)
        ... )

        >>> # With session for repeated calls
        >>> with nirs4all.session(verbose=1) as s:
        ...     r1 = nirs4all.run(pipeline1, data1, session=s)
        ...     r2 = nirs4all.run(pipeline2, data2, session=s)
    """
    ...
```

---

## 3. Design Analysis: sklearn Meta-Estimator

### 3.1 Two-Layer Architecture

Following sklearn conventions, we separate concerns into two layers:

**Layer 1: Single Pipeline Estimator (`NIRSPipeline`)**
- Wraps a single pipeline configuration
- Implements `fit()`, `predict()`, `transform()`
- Exposes fitted model for SHAP
- Used for deployment, prediction, explanation

**Layer 2: Multi-Pipeline Search (`NIRSPipelineSearch`)**
- Evaluates multiple pipeline configurations
- Implements cross-validation and ranking
- Exposes `best_estimator_`, `cv_results_`
- Similar to `GridSearchCV` pattern

### 3.2 SHAP Compatibility Requirements

SHAP has different requirements depending on the model type:

| Model Type | SHAP Method | Requirement |
|------------|-------------|-------------|
| sklearn | `Explainer(auto)` | `predict()` callable |
| Tree models | `TreeExplainer` | Direct model access |
| Neural Networks | `DeepExplainer` | Differentiable graph from X to y |
| Black-box | `KernelExplainer` | `predict()` callable |

For NN + high-dimensional NIRS:
- Need differentiable preprocessing OR
- Use partition/grouping strategy with black-box SHAP
- Expose `model_` attribute for direct access

### 3.3 Stacking Meta-Learner SHAP

For stacking, SHAP can explain at multiple levels:

1. **Meta-level**: Which base learner contributed? (inputs = base predictions)
2. **Base-level**: Which wavelengths matter per base learner?
3. **End-to-end**: Combined attribution to original features

If meta-learner is linear: `φ_i(stack) = Σ_j w_j * φ_i^(j)`

For non-linear meta-learners, use local approximation (linear surrogate).

---

## 4. Complete Design Proposal

### 4.1 Package Structure

```
nirs4all/
├── __init__.py              # Module-level API (run, predict, explain, session)
├── api/
│   ├── __init__.py
│   ├── run.py               # run() implementation
│   ├── predict.py           # predict() implementation
│   ├── explain.py           # explain() implementation
│   ├── retrain.py           # retrain() implementation
│   ├── session.py           # Session context manager
│   ├── config.py            # RunConfig, PredictConfig, ExplainConfig
│   └── result.py            # RunResult, PredictResult, ExplainResult
├── sklearn/
│   ├── __init__.py
│   ├── pipeline.py          # NIRSPipeline (BaseEstimator)
│   ├── search.py            # NIRSPipelineSearch (meta-estimator)
│   └── shap_bridge.py       # SHAP integration helpers
└── ... (existing modules)
```

### 4.2 Configuration Classes

```python
# nirs4all/api/config.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

@dataclass
class RunConfig:
    """Configuration for pipeline execution.

    All fields are optional with sensible defaults.
    This class is JSON/YAML serializable for reproducibility.
    """
    # Execution control
    verbose: int = 1
    random_state: Optional[int] = None
    continue_on_error: bool = False

    # Output control
    workspace_path: Optional[Path] = None
    save_artifacts: bool = True
    save_charts: bool = True
    enable_reports: bool = True

    # Display control
    plots_visible: bool = False
    show_spinner: bool = True
    show_progress_bar: bool = True

    # Logging control
    log_file: bool = True
    log_format: str = "pretty"  # "pretty", "minimal", "json"
    json_output: bool = False  # Output predictions as JSON
    use_unicode: bool = True
    use_colors: bool = True

    # Advanced
    max_generation_count: int = 10000
    keep_datasets: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PredictConfig:
    """Configuration for prediction."""
    verbose: int = 0
    all_predictions: bool = False
    batch_size: Optional[int] = None


@dataclass
class ExplainConfig:
    """Configuration for SHAP explanation."""
    n_samples: int = 200
    explainer_type: str = "auto"  # "auto", "tree", "kernel", "deep", "linear"
    visualizations: List[str] = field(default_factory=lambda: ["spectral", "summary"])

    # Binning for spectral SHAP
    bin_size: Union[int, Dict[str, int]] = 20
    bin_stride: Union[int, Dict[str, int]] = 10
    bin_aggregation: Union[str, Dict[str, str]] = "mean"

    # Output control
    plots_visible: bool = True
    output_dir: Optional[Path] = None

    # Advanced: explanation space
    space: str = "preprocessed"  # "raw" or "preprocessed"
    feature_groups: Optional[List[List[int]]] = None  # For grouped SHAP
```

### 4.3 Result Classes

```python
# nirs4all/api/result.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from pathlib import Path
import numpy as np

if TYPE_CHECKING:
    from nirs4all.pipeline import PipelineRunner

from nirs4all.data.predictions import Predictions

@dataclass
class RunResult:
    """Result from nirs4all.run().

    Provides convenient access to predictions, best model, and artifacts.
    """
    predictions: Predictions
    per_dataset: Dict[str, Any]
    _runner: Optional["PipelineRunner"] = field(default=None, repr=False)

    # Convenience accessors
    @property
    def best(self) -> Dict[str, Any]:
        """Get best prediction entry."""
        top = self.predictions.top(n=1)
        return top[0] if top else {}

    @property
    def best_score(self) -> float:
        """Get best model's primary score."""
        return self.best.get('test_score', float('nan'))

    @property
    def artifacts_path(self) -> Optional[Path]:
        """Get path to saved artifacts."""
        return self.per_dataset.get('artifacts_path')

    def top(self, n: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Shortcut to predictions.top()."""
        return self.predictions.top(n=n, **kwargs)

    def export(self, output_path: str, format: str = "n4a") -> Path:
        """Export best model to bundle."""
        if self._runner is None:
            raise RuntimeError("Cannot export: runner reference not available")
        return self._runner.export(
            source=self.best,
            output_path=output_path,
            format=format
        )


@dataclass
class PredictResult:
    """Result from nirs4all.predict()."""
    y_pred: np.ndarray
    predictions: Predictions

    @property
    def y_pred_flat(self) -> np.ndarray:
        """Flattened predictions."""
        return self.y_pred.flatten()


@dataclass
class ExplainResult:
    """Result from nirs4all.explain()."""
    shap_values: np.ndarray
    feature_importance: np.ndarray
    output_dir: Path
    model_name: str

    # Per-visualization results
    visualizations: Dict[str, Path]
```

### 4.4 Session Context Manager

```python
# nirs4all/api/session.py

from contextlib import contextmanager
from typing import Optional, Generator
from pathlib import Path

from nirs4all.pipeline import PipelineRunner

class Session:
    """Execution session for resource reuse.

    A session maintains state across multiple run/predict/explain calls:
    - Shared workspace
    - Cached models and transformers
    - Consistent logging configuration
    - Device placement (GPU/CPU)

    Use sessions when:
    - Making multiple calls in sequence
    - Need to share artifacts between calls
    - Want consistent configuration

    Example:
        >>> with nirs4all.session(verbose=2, save_artifacts=True) as s:
        ...     r1 = nirs4all.run(pipeline1, data1, session=s)
        ...     r2 = nirs4all.run(pipeline2, data2, session=s)
        ...     # Both runs share the same workspace
        >>> # Session cleanup happens here
    """

    def __init__(
        self,
        workspace_path: Optional[Path] = None,
        verbose: int = 1,
        random_state: Optional[int] = None,
        save_artifacts: bool = True,
        save_charts: bool = True,
        **kwargs
    ):
        self.workspace_path = workspace_path or Path.cwd() / "workspace"
        self.verbose = verbose
        self.random_state = random_state
        self.save_artifacts = save_artifacts
        self.save_charts = save_charts
        self._kwargs = kwargs

        self._runner: Optional[PipelineRunner] = None
        self._is_active = False

    @property
    def runner(self) -> PipelineRunner:
        """Get or create the underlying PipelineRunner."""
        if self._runner is None:
            self._runner = PipelineRunner(
                workspace_path=self.workspace_path,
                verbose=self.verbose,
                random_state=self.random_state,
                save_artifacts=self.save_artifacts,
                save_charts=self.save_charts,
                **self._kwargs
            )
        return self._runner

    def __enter__(self) -> "Session":
        self._is_active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._is_active = False
        self._cleanup()
        return False

    def _cleanup(self):
        """Release session resources."""
        # Close any open files, clear caches, etc.
        self._runner = None


@contextmanager
def session(**kwargs) -> Generator[Session, None, None]:
    """Create an execution session.

    Context manager for grouped operations with shared resources.

    Args:
        **kwargs: Session configuration (verbose, workspace_path, etc.)

    Yields:
        Session instance for use with run(), predict(), explain()

    Example:
        >>> with nirs4all.session(verbose=2) as s:
        ...     result = nirs4all.run(pipeline, data, session=s)
    """
    s = Session(**kwargs)
    try:
        yield s.__enter__()
    finally:
        s.__exit__(None, None, None)
```

### 4.5 Module-Level Functions

```python
# nirs4all/api/run.py

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np

from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions

from .config import RunConfig
from .session import Session
from .result import RunResult

# Type aliases for flexibility
PipelineSpec = Union[
    List[Any],                    # List of steps
    Dict[str, Any],               # Dict with 'pipeline' key
    str,                          # Path to config file
    Path,                         # Path object
    PipelineConfigs               # Backward compat
]

DatasetSpec = Union[
    str,                          # Path to data
    Path,                         # Path object
    Tuple[np.ndarray, ...],       # (X,) or (X, y) or (X, y, metadata)
    Dict[str, Any],               # Config dict
    List[str],                    # Multiple paths
    DatasetConfigs                # Backward compat
]


def run(
    pipeline: PipelineSpec,
    dataset: DatasetSpec,
    *,
    name: str = "",
    config: Optional[RunConfig] = None,
    session: Optional[Session] = None,
    **overrides
) -> RunResult:
    """Execute a training pipeline on a dataset.

    This is the primary entry point for training ML pipelines on NIRS data.
    It's a thin wrapper around PipelineRunner - all normalization of pipeline
    and dataset formats is handled by the existing PipelineRunner.run() method.

    Args:
        pipeline: Pipeline definition (steps list, path, or PipelineConfigs).
            PipelineRunner already handles normalization.
        dataset: Dataset definition (path, arrays, dict, or DatasetConfigs).
            PipelineRunner already handles normalization.
        name: Optional pipeline name for identification
        config: Optional RunConfig for execution settings
        session: Optional Session for resource reuse
        **overrides: Override specific config values (verbose, save_artifacts, etc.)

    Returns:
        RunResult with predictions, best model, and artifacts

    Examples:
        >>> # Minimal usage
        >>> result = nirs4all.run(
        ...     [MinMaxScaler(), PLSRegression(10)],
        ...     "sample_data/wheat"
        ... )

        >>> # With cross-validation
        >>> result = nirs4all.run(
        ...     [SNV(), KFold(5), {"model": PLSRegression(10)}],
        ...     (X_train, y_train),
        ...     verbose=2
        ... )

        >>> # With full config
        >>> result = nirs4all.run(
        ...     pipeline="configs/best_pipeline.yaml",
        ...     dataset={"train_x": "data/X.csv", "train_y": "data/y.csv"},
        ...     config=RunConfig(verbose=1, save_artifacts=True, random_state=42)
        ... )
    """
    # Build runner kwargs from config and overrides
    runner_kwargs = _build_runner_kwargs(config, overrides)

    # Get runner from session or create new one
    if session is not None:
        runner = session.runner
    else:
        runner = PipelineRunner(**runner_kwargs)

    # Execute - PipelineRunner.run() handles all format normalization
    predictions, per_dataset = runner.run(
        pipeline=pipeline,
        dataset=dataset,
        pipeline_name=name
    )

    return RunResult(
        predictions=predictions,
        per_dataset=per_dataset,
        _runner=runner  # Store for export() support
    )


def _build_runner_kwargs(
    config: Optional[RunConfig],
    overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Build PipelineRunner kwargs from config and overrides.

    This uses a thin wrapper approach - we pass kwargs directly to
    PipelineRunner rather than duplicating normalization logic.
    """
    # Start with config values if provided
    if config is not None:
        kwargs = config.to_dict()
    else:
        kwargs = {}

    # Apply overrides (these take precedence)
    kwargs.update(overrides)

    return kwargs


# NOTE: Pipeline and dataset normalization is handled by PipelineRunner.run()
# which already accepts flexible input types:
#   - pipeline: Union[PipelineConfigs, List[Any], Dict, str]
#   - dataset: Union[DatasetConfigs, SpectroDataset, np.ndarray, Tuple, Dict, List, str]
#
# We don't duplicate this logic here (thin wrapper pattern).
```

### 4.6 sklearn Meta-Estimator

```python
# nirs4all/sklearn/pipeline.py

from typing import Any, Dict, List, Optional, Union
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import nirs4all
from nirs4all.pipeline import PipelineRunner, PipelineConfigs, MinimalPipeline
from nirs4all.data import DatasetConfigs


class NIRSPipeline(BaseEstimator, RegressorMixin):
    """sklearn-compatible NIRS pipeline estimator.

    Wraps a nirs4all pipeline to provide sklearn's BaseEstimator interface.
    This enables:
    - SHAP compatibility for model explanation
    - Optuna integration for hyperparameter tuning
    - joblib serialization for deployment

    **IMPORTANT**: This class can be used in two modes:

    1. **Prediction wrapper** (recommended): Wrap an already-trained pipeline
       using class methods `from_result()` or `from_bundle()`. In this mode,
       `fit()` raises NotImplementedError.

    2. **Training mode**: Pass `steps` to __init__ and call `fit()`. This
       delegates to `nirs4all.run()` internally. Note that cross-validation
       creates multiple models; the primary model (fold 0) is exposed via
       `model_` property.

    After fitting or loading, the pipeline exposes:
    - model_: The primary fitted model (fold 0 for CV, for SHAP access)
    - preprocessor_: Fitted preprocessing chain (if available)
    - pipeline_: MinimalPipeline for prediction replay

    Parameters
    ----------
    steps : list
        List of pipeline steps (transformers, splitters, model).
        Same format as nirs4all.run() pipeline argument.

    cv : cross-validator, optional
        Cross-validation splitter. If provided, used during fit().
        If None, uses train/test split from data.

    random_state : int, optional
        Random seed for reproducibility.

    verbose : int, default=0
        Verbosity level.

    save_artifacts : bool, default=False
        Whether to save artifacts to disk during fit.

    Attributes
    ----------
    model_ : estimator
        Fitted final model (available after fit).

    preprocessor_ : transformer
        Fitted preprocessing chain (available after fit).

    n_features_in_ : int
        Number of input features (available after fit).

    feature_names_in_ : ndarray, optional
        Feature names (if provided during fit).

    classes_ : ndarray
        Class labels (for classifiers, available after fit).

    Examples
    --------
    >>> from nirs4all.sklearn import NIRSPipeline
    >>> from sklearn.model_selection import cross_validate
    >>>
    >>> pipe = NIRSPipeline(
    ...     steps=[MinMaxScaler(), SNV(), PLSRegression(10)]
    ... )
    >>>
    >>> # Use with sklearn cross_validate
    >>> scores = cross_validate(pipe, X, y, cv=5, scoring='r2')
    >>>
    >>> # Use with SHAP
    >>> import shap
    >>> pipe.fit(X_train, y_train)
    >>> explainer = shap.Explainer(pipe.predict, X_train)
    >>> shap_values = explainer(X_test)
    >>>
    >>> # Use with Optuna
    >>> def objective(trial):
    ...     n_comp = trial.suggest_int('n_components', 1, 30)
    ...     pipe = NIRSPipeline(steps=[MinMaxScaler(), PLSRegression(n_comp)])
    ...     return cross_val_score(pipe, X, y, cv=5).mean()
    """

    def __init__(
        self,
        steps: List[Any],
        *,
        cv=None,
        random_state: Optional[int] = None,
        verbose: int = 0,
        save_artifacts: bool = False,
        workspace_path: Optional[str] = None
    ):
        # sklearn requirement: store all params in __init__, no heavy work
        self.steps = steps
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose
        self.save_artifacts = save_artifacts
        self.workspace_path = workspace_path

    def fit(self, X, y, **fit_params):
        """Fit the pipeline.

        Delegates to nirs4all.run() internally. If cross-validation is used,
        multiple models are created (one per fold). The `model_` property
        returns the primary model (fold 0) for SHAP compatibility.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        **fit_params : dict
            Additional parameters passed to the pipeline.

        Returns
        -------
        self : NIRSPipeline
            Fitted estimator.

        Raises
        ------
        NotImplementedError
            If this instance was created via from_result() or from_bundle()
            (prediction wrapper mode).
        """
        # Check if this is a prediction wrapper
        if hasattr(self, '_minimal_pipeline') and self._minimal_pipeline is not None:
            raise NotImplementedError(
                "This NIRSPipeline was created from a trained model and cannot be refit.\n"
                "Create a new NIRSPipeline with steps to train a new model."
            )

        # Validate input
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        # Store input metadata
        self.n_features_in_ = X.shape[1]

        # Build pipeline with optional CV
        pipeline_steps = list(self.steps)
        if self.cv is not None:
            # Insert CV before model
            pipeline_steps = self._insert_cv(pipeline_steps, self.cv)

        # Run training via nirs4all
        result = nirs4all.run(
            pipeline=pipeline_steps,
            dataset=(X, y),
            config=nirs4all.RunConfig(
                verbose=self.verbose,
                save_artifacts=self.save_artifacts,
                random_state=self.random_state,
                workspace_path=self.workspace_path
            )
        )

        # Extract fitted components
        self._extract_fitted_components(result)

        return self

    def predict(self, X):
        """Predict using the fitted pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values.
        """
        check_is_fitted(self)
        X = check_array(X)

        # Use minimal pipeline for prediction
        return self.pipeline_.predict(X)

    def transform(self, X):
        """Apply preprocessing transformation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        check_is_fitted(self)
        X = check_array(X)

        return self.preprocessor_.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform in one step."""
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def score(self, X, y):
        """Return R² score (for regressors) or accuracy (for classifiers)."""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    @property
    def shap_model(self):
        """Get model object for SHAP.

        Returns the underlying model for direct SHAP analysis.
        For gradient-based SHAP on NNs, this provides access to
        the differentiable model.
        """
        check_is_fitted(self)
        return self.model_

    def explain(
        self,
        X,
        *,
        method: str = "auto",
        background: Optional[np.ndarray] = None,
        **kwargs
    ):
        """Generate SHAP explanations.

        Convenience method for SHAP analysis that automatically
        selects the appropriate explainer based on model type.

        Parameters
        ----------
        X : array-like
            Samples to explain.

        method : str, default="auto"
            SHAP method: "auto", "kernel", "tree", "deep", "linear"

        background : array-like, optional
            Background samples for SHAP. If None, uses kmeans on X.

        **kwargs : dict
            Additional arguments for ShapAnalyzer.

        Returns
        -------
        shap_values : shap.Explanation
            SHAP explanation object.
        """
        check_is_fitted(self)

        import shap

        if method == "auto":
            # Auto-select based on model type
            model = self.model_
            if hasattr(model, 'tree_') or 'Tree' in type(model).__name__:
                explainer = shap.TreeExplainer(model)
            elif hasattr(model, 'coef_'):
                explainer = shap.LinearExplainer(model, background or X[:100])
            else:
                # Black-box fallback
                explainer = shap.KernelExplainer(self.predict, background or shap.kmeans(X, 50))

        return explainer(X)

    def _insert_cv(self, steps, cv):
        """Insert CV splitter before model in steps."""
        # Find model step (last step or step with 'model' key)
        model_idx = None
        for i, step in enumerate(steps):
            if isinstance(step, dict) and 'model' in step:
                model_idx = i
                break

        if model_idx is None:
            # Model is last step
            model_idx = len(steps) - 1

        # Insert CV before model
        return steps[:model_idx] + [cv] + steps[model_idx:]

    def _extract_fitted_components(self, result):
        """Extract fitted model and preprocessor from result."""
        # Get best prediction
        best = result.best

        # Load fitted pipeline from artifacts
        # This uses the MinimalPipeline extraction
        from nirs4all.pipeline import TraceBasedExtractor

        extractor = TraceBasedExtractor()
        self.pipeline_ = extractor.extract(
            result.per_dataset.get('execution_trace'),
            result.per_dataset.get('artifact_provider')
        )

        # Extract model
        self.model_ = self.pipeline_.get_model()

        # Extract preprocessor chain
        self.preprocessor_ = self.pipeline_.get_preprocessor()

        # Store predictions metadata
        self._predictions = result.predictions
        self._best_params = best.get('best_params', {})

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'steps': self.steps,
            'cv': self.cv,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'save_artifacts': self.save_artifacts,
            'workspace_path': self.workspace_path
        }

    def set_params(self, **params):
        """Set parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # =========================================================================
    # Class methods for prediction wrapper mode
    # =========================================================================

    @classmethod
    def from_result(cls, result: "RunResult", model_index: int = 0) -> "NIRSPipeline":
        """Create wrapper from nirs4all RunResult.

        This creates a prediction-only wrapper. The `fit()` method will
        raise NotImplementedError.

        Parameters
        ----------
        result : RunResult
            Result from nirs4all.run()

        model_index : int, default=0
            Which model to wrap (0 = best by ranking)

        Returns
        -------
        NIRSPipeline
            Prediction wrapper ready for predict() and SHAP analysis
        """
        pipe = cls(steps=[])  # Empty steps for wrapper mode
        pipe._load_from_result(result, model_index)
        return pipe

    @classmethod
    def from_bundle(cls, bundle_path: str) -> "NIRSPipeline":
        """Create wrapper from exported .n4a bundle.

        Parameters
        ----------
        bundle_path : str
            Path to .n4a bundle file

        Returns
        -------
        NIRSPipeline
            Prediction wrapper ready for predict()
        """
        from nirs4all.pipeline.bundle import BundleLoader

        pipe = cls(steps=[])  # Empty steps for wrapper mode
        loader = BundleLoader(bundle_path)
        pipe._minimal_pipeline = loader.minimal_pipeline
        pipe._artifact_provider = loader.artifact_provider
        return pipe

    def _load_from_result(self, result, model_index):
        """Internal: load minimal pipeline from RunResult."""
        from nirs4all.pipeline.trace import TraceBasedExtractor

        predictions = result.predictions.top(n=model_index + 1)
        if model_index >= len(predictions):
            raise ValueError(f"model_index {model_index} >= available models {len(predictions)}")

        target = predictions[model_index]
        extractor = TraceBasedExtractor()
        self._minimal_pipeline = extractor.extract(
            result.per_dataset.get('execution_trace'),
            result.per_dataset.get('artifact_provider')
        )
        self.pipeline_ = self._minimal_pipeline
        self.model_ = self._minimal_pipeline.get_model() if self._minimal_pipeline else None


class NIRSPipelineClassifier(NIRSPipeline, ClassifierMixin):
    """Classification variant of NIRSPipeline.

    Same as NIRSPipeline but for classification tasks.
    Provides predict_proba() and exposes classes_.
    """

    def fit(self, X, y, **fit_params):
        """Fit the classifier."""
        super().fit(X, y, **fit_params)

        # Store classes
        self.classes_ = np.unique(y)

        return self

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like
            Samples to predict.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X = check_array(X)

        if hasattr(self.model_, 'predict_proba'):
            return self.model_.predict_proba(X)
        else:
            # Fallback: use predictions as pseudo-probabilities
            y_pred = self.predict(X)
            # One-hot encode
            proba = np.zeros((len(y_pred), len(self.classes_)))
            for i, pred in enumerate(y_pred):
                idx = np.where(self.classes_ == pred)[0]
                if len(idx) > 0:
                    proba[i, idx[0]] = 1.0
            return proba

    def score(self, X, y):
        """Return accuracy score."""
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
```

### 4.7 Multi-Pipeline Search (Meta-Estimator)

```python
# nirs4all/sklearn/search.py

from typing import Any, Dict, List, Optional, Union
import numpy as np

from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.model_selection import cross_val_score


class NIRSPipelineSearch(BaseEstimator, MetaEstimatorMixin):
    """Search over multiple pipeline configurations.

    Similar to GridSearchCV but for nirs4all pipelines with generator
    syntax (_or_, _range_, feature_augmentation).

    After fitting, provides:
    - best_estimator_: Fitted NIRSPipeline with best configuration
    - best_params_: Best generator choices
    - best_score_: Best cross-validation score
    - cv_results_: Dictionary with all results

    Parameters
    ----------
    pipeline : list
        Pipeline with generator syntax (_or_, _range_, etc.)

    cv : cross-validator
        Cross-validation strategy.

    scoring : str or callable
        Scoring metric for ranking.

    refit : bool, default=True
        If True, refit best pipeline on full data.

    n_jobs : int, default=1
        Number of parallel jobs.

    Examples
    --------
    >>> from nirs4all.sklearn import NIRSPipelineSearch
    >>>
    >>> # Pipeline with generator syntax
    >>> pipeline = [
    ...     MinMaxScaler(),
    ...     {"_or_": [SNV(), MSC(), Gaussian()]},  # Try each
    ...     {"model": PLSRegression(), "_range_": [1, 30, 5], "param": "n_components"}
    ... ]
    >>>
    >>> search = NIRSPipelineSearch(pipeline, cv=5, scoring='neg_root_mean_squared_error')
    >>> search.fit(X, y)
    >>>
    >>> print(search.best_params_)
    >>> print(search.best_score_)
    >>>
    >>> # Predict with best model
    >>> y_pred = search.predict(X_test)
    """

    def __init__(
        self,
        pipeline: List[Any],
        cv=5,
        scoring: str = "neg_root_mean_squared_error",
        refit: bool = True,
        n_jobs: int = 1,
        verbose: int = 0,
        random_state: Optional[int] = None
    ):
        self.pipeline = pipeline
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y, **fit_params):
        """Fit and find best pipeline configuration.

        Parameters
        ----------
        X : array-like
            Training data.

        y : array-like
            Target values.

        **fit_params : dict
            Additional parameters.

        Returns
        -------
        self : NIRSPipelineSearch
            Fitted searcher.
        """
        # Run nirs4all with generator expansion
        result = nirs4all.run(
            pipeline=self.pipeline,
            dataset=(X, y),
            config=nirs4all.RunConfig(
                verbose=self.verbose,
                random_state=self.random_state,
                save_artifacts=True  # Need artifacts for refit
            )
        )

        # Extract cv_results from predictions
        self.cv_results_ = self._build_cv_results(result)

        # Get best configuration
        best = result.best
        self.best_score_ = best.get('test_score', float('nan'))
        self.best_params_ = best.get('best_params', {})
        self.best_index_ = 0  # TODO: actual index

        # Refit if requested
        if self.refit:
            # Create NIRSPipeline with best configuration
            best_steps = self._reconstruct_steps(best)
            self.best_estimator_ = NIRSPipeline(
                steps=best_steps,
                random_state=self.random_state,
                verbose=self.verbose
            )
            self.best_estimator_.fit(X, y)

        return self

    def predict(self, X):
        """Predict using best estimator."""
        if not self.refit:
            raise ValueError("predict requires refit=True")
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        """Score using best estimator."""
        if not self.refit:
            raise ValueError("score requires refit=True")
        return self.best_estimator_.score(X, y)

    def _build_cv_results(self, result):
        """Build cv_results_ dict from nirs4all result."""
        # Similar to GridSearchCV's cv_results_ format
        return {
            'mean_test_score': [],
            'std_test_score': [],
            'params': [],
            'rank_test_score': [],
            # ... additional fields
        }

    def _reconstruct_steps(self, best_pred):
        """Reconstruct pipeline steps from best prediction."""
        # Use generator_choices from PipelineConfigs to reconstruct
        # the exact step configuration
        ...
```

### 4.8 Updated Package Exports

```python
# nirs4all/__init__.py (updated)

"""
NIRS4All - Near-Infrared Spectroscopy Analysis Library

Public API:
    nirs4all.run(pipeline, dataset, **config)    - Train a pipeline
    nirs4all.predict(source, dataset, **config)  - Predict with trained model
    nirs4all.explain(source, dataset, **config)  - Generate SHAP explanations
    nirs4all.retrain(source, dataset, **config)  - Retrain with new data
    nirs4all.session(**config)                   - Create execution session

Classes (for advanced usage):
    nirs4all.PipelineRunner    - Direct runner access
    nirs4all.PipelineConfigs   - Pipeline configuration
    nirs4all.DatasetConfigs    - Dataset configuration (from nirs4all.data)

sklearn Integration:
    nirs4all.sklearn.NIRSPipeline        - sklearn-compatible estimator
    nirs4all.sklearn.NIRSPipelineSearch  - Multi-pipeline search
"""

__version__ = "0.6.0"

# Module-level API
from .api import (
    run,
    predict,
    explain,
    retrain,
    session,
    RunConfig,
    PredictConfig,
    ExplainConfig,
    RunResult,
    PredictResult,
    ExplainResult,
    Session,
)

# Backward compatibility - existing imports still work
from .pipeline import PipelineRunner, PipelineConfigs
from .data import DatasetConfigs
from .controllers import register_controller, CONTROLLER_REGISTRY

# Utility functions
from .utils import (
    is_tensorflow_available,
    is_gpu_available,
    framework
)

# sklearn integration (lazy import to avoid sklearn dependency)
def __getattr__(name):
    if name == "sklearn":
        from . import sklearn as sklearn_module
        return sklearn_module
    raise AttributeError(f"module 'nirs4all' has no attribute {name}")

__all__ = [
    # Version
    "__version__",

    # Module-level API (recommended)
    "run",
    "predict",
    "explain",
    "retrain",
    "session",

    # Config classes
    "RunConfig",
    "PredictConfig",
    "ExplainConfig",

    # Result classes
    "RunResult",
    "PredictResult",
    "ExplainResult",

    # Session
    "Session",

    # Pipeline components (backward compat)
    "PipelineRunner",
    "PipelineConfigs",
    "DatasetConfigs",

    # Controller system
    "register_controller",
    "CONTROLLER_REGISTRY",

    # Utilities
    "is_tensorflow_available",
    "is_gpu_available",
    "framework",

    # sklearn submodule
    "sklearn",
]
```

---

## 5. Usage Examples

### 5.1 Simple Training (New API)

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

# One-liner training
result = nirs4all.run(
    pipeline=[MinMaxScaler(), PLSRegression(10)],
    dataset="sample_data/wheat",
    verbose=1
)

print(f"Best RMSE: {result.best['rmse']:.4f}")
```

### 5.2 Full Pipeline with Config

```python
import nirs4all
from nirs4all.operators.transforms import SNV, SavitzkyGolay

pipeline = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    {"feature_augmentation": [SNV(), SavitzkyGolay()]},
    KFold(n_splits=5),
    {"model": PLSRegression(10)}
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset="data/spectra",
    config=nirs4all.RunConfig(
        verbose=2,
        save_artifacts=True,
        random_state=42
    )
)

# Export best model
result.export("exports/best_model.n4a")
```

### 5.3 Session for Multiple Runs

```python
import nirs4all

with nirs4all.session(verbose=1, save_artifacts=True) as s:
    # Multiple runs share workspace
    r1 = nirs4all.run(pipeline_pls, data, session=s)
    r2 = nirs4all.run(pipeline_rf, data, session=s)
    r3 = nirs4all.run(pipeline_nn, data, session=s)

    # Compare results
    print(f"PLS: {r1.best_score:.4f}")
    print(f"RF:  {r2.best_score:.4f}")
    print(f"NN:  {r3.best_score:.4f}")
```

### 5.4 sklearn Integration

```python
from nirs4all.sklearn import NIRSPipeline
from sklearn.model_selection import cross_validate, GridSearchCV
import shap

# Create sklearn-compatible pipeline
pipe = NIRSPipeline(
    steps=[MinMaxScaler(), SNV(), PLSRegression(10)]
)

# Use with sklearn cross_validate
cv_results = cross_validate(pipe, X, y, cv=5, scoring='r2', return_estimator=True)
print(f"Mean R²: {cv_results['test_score'].mean():.4f}")

# Use with SHAP
pipe.fit(X_train, y_train)
explainer = shap.Explainer(pipe.predict, X_train[:100])
shap_values = explainer(X_test)
shap.summary_plot(shap_values)

# Access underlying model for advanced SHAP
model = pipe.shap_model
tree_explainer = shap.TreeExplainer(model)  # If tree-based
```

### 5.5 Multi-Pipeline Search

```python
from nirs4all.sklearn import NIRSPipelineSearch

# Pipeline with generator syntax
pipeline = [
    MinMaxScaler(),
    {"_or_": [SNV(), MSC(), Gaussian(), None]},  # Try each preprocessing
    {"_range_": [1, 30, 5], "param": "n_components", "model": PLSRegression()}
]

search = NIRSPipelineSearch(
    pipeline=pipeline,
    cv=5,
    scoring='neg_root_mean_squared_error'
)

search.fit(X, y)

print(f"Best params: {search.best_params_}")
print(f"Best score: {-search.best_score_:.4f} RMSE")

# Predict with best configuration
y_pred = search.predict(X_test)
```

### 5.6 Stacking with SHAP

```python
from nirs4all.sklearn import NIRSPipeline
import shap

# Train stacking pipeline
stacking_pipeline = [
    MinMaxScaler(),
    {"branch": [
        [SNV(), PLSRegression(10)],
        [SavitzkyGolay(), RandomForestRegressor()],
    ]},
    {"merge": "predictions"},
    {"model": RidgeCV()}  # Linear meta-learner
]

pipe = NIRSPipeline(steps=stacking_pipeline)
pipe.fit(X_train, y_train)

# SHAP on meta-learner (which base model contributed)
meta_model = pipe.model_
base_predictions = pipe.transform(X_test)  # Base model predictions
explainer = shap.LinearExplainer(meta_model, base_predictions)
meta_shap = explainer(base_predictions)

# Get feature-level attributions (if meta is linear)
# φ_i(stack) = Σ_j w_j * φ_i^(j)
# This aggregates base-level SHAP weighted by meta-learner coefficients
```

---

## 6. Migration Guide

### 6.1 From Current API to New API

| Current API | New API |
|-------------|---------|
| `PipelineConfigs(steps, name)` | Pass `steps` directly to `run()` |
| `DatasetConfigs(path)` | Pass `path` directly to `run()` |
| `PipelineRunner(verbose=1)` | `run(..., verbose=1)` or `RunConfig(verbose=1)` |
| `runner.run(pipeline, dataset)` | `nirs4all.run(pipeline, dataset)` |
| `runner.predict(model, data)` | `nirs4all.predict(model, data)` |
| `runner.explain(model, data)` | `nirs4all.explain(model, data)` |

### 6.2 Backward Compatibility

The current API remains fully functional:

```python
# This still works exactly as before
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs

runner = PipelineRunner(verbose=1)
predictions, _ = runner.run(
    PipelineConfigs(pipeline, "name"),
    DatasetConfigs("path")
)
```

---

## 7. Implementation Roadmap

### Phase 1: API Module (v0.6.0)
- [ ] Create `nirs4all/api/` module structure
- [ ] Implement `RunConfig`, `PredictConfig`, `ExplainConfig`
- [ ] Implement `RunResult`, `PredictResult`, `ExplainResult`
- [ ] Implement `Session` context manager
- [ ] Implement `run()`, `predict()`, `explain()`, `retrain()` facades
- [ ] Update `nirs4all/__init__.py` exports

### Phase 2: sklearn Integration (v0.6.0)
- [ ] Create `nirs4all/sklearn/` module
- [ ] Implement `NIRSPipeline` BaseEstimator
- [ ] Implement `NIRSPipelineClassifier`
- [ ] Add SHAP integration helpers
- [ ] Test with sklearn cross_validate, GridSearchCV

### Phase 3: Advanced sklearn (v0.7.0)
- [ ] ~~Implement `NIRSPipelineSearch` meta-estimator~~ (DROPPED - generator syntax provides this)
- [ ] Add Optuna integration example
- [ ] Document generator syntax for hyperparameter search
- [ ] Support `partial_fit()` for streaming (if demand)
- [ ] Differentiable preprocessing for gradient SHAP (if demand)

### Phase 4: Documentation (v0.6.0)
- [ ] Update README with new API examples
- [ ] Create migration guide
- [ ] Update RTD documentation
- [ ] Update examples/Q*.py files

---

## 8. Appendix: Design Decisions

### A1: Why Facade Pattern (Not Replacing PipelineRunner)

- **Risk mitigation**: PipelineRunner is battle-tested; facades add UX without core changes
- **Flexibility**: Power users can still use PipelineRunner directly
- **Testing**: All existing tests remain valid
- **Incremental adoption**: Users can migrate gradually

### A2: Why Separate sklearn Module

- **Optional dependency**: sklearn is not strictly required for nirs4all core
- **Clean separation**: sklearn conventions don't pollute core API
- **Lazy loading**: Module only loaded when accessed

### A3: Why Session Instead of Global State

- **Explicit resource management**: User controls lifecycle
- **Test isolation**: Each test can use fresh session
- **Parallel execution**: Multiple sessions can run concurrently
- **IDE support**: Type hints work properly

### A4: SHAP Strategy for High-Dimensional NIRS

For 2k+ wavelength features:
1. **Default**: Use `space="preprocessed"` (explain after normalization)
2. **Grouped**: Provide `feature_groups` for wavelength regions
3. **Gradient**: For NNs, implement differentiable preprocessing in torch/tf

---

## 9. Open Questions

1. **Should `nirs4all.run()` return a future/handle for async execution?**
   - Current: synchronous, blocks until complete
   - Alternative: return `RunHandle` with `.result()`, `.cancel()`, `.progress()`

2. **How to handle multi-dataset in sklearn estimator?**
   - Option A: Separate `MultiDatasetPipeline` class
   - Option B: Accept dataset name in `fit(X, y, dataset_name=...)`

3. **Should `NIRSPipeline` support generator syntax?**
   - Current: No, use `NIRSPipelineSearch` for multi-config
   - Alternative: Expand in `fit()` and use best

4. **Differentiable preprocessing priority?**
   - Which transforms to implement first in torch/tf
   - Trade-off: coverage vs. complexity
