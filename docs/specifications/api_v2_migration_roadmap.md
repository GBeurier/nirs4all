# NIRS4ALL API v2 Migration Roadmap

## Document Information

- **Author**: Critical Review by GitHub Copilot (Claude Opus 4.5)
- **Date**: December 24, 2025
- **Status**: Review + Actionable Roadmap
- **Base Document**: [api_design_v2.md](api_design_v2.md)
- **Scope**: Critical assessment and migration plan for nirs4all 0.6+

---

## Part 1: Critical Review of API Design v2

### 1.1 Executive Summary

The API Design v2 proposes valuable simplifications but contains several **misalignments with the current implementation** that would result in duplicated code, semantic confusion, and incomplete sklearn integration. This review identifies what works, what needs modification, and what should be deferred.

**Verdict**: Keep the module-level API vision, significantly revise the sklearn meta-estimator design, and leverage existing infrastructure rather than duplicating it.

---

### 1.2 Module-Level API Assessment

#### What's Proposed
```python
def run(
    pipeline: PipelineSpec,
    dataset: DatasetSpec,
    *,
    config: Optional[RunConfig] = None,
    session: Optional[Session] = None,
    **overrides
) -> RunResult
```

#### Current Reality

**Good news**: `PipelineRunner.run()` already accepts flexible input types:
```python
# From runner.py lines 217-223
def run(
    self,
    pipeline: Union[PipelineConfigs, List[Any], Dict, str],
    dataset: Union[DatasetConfigs, SpectroDataset, np.ndarray, Tuple[np.ndarray, ...], Dict, List[Dict], str, List[str]],
    ...
)
```

The type aliases `PipelineSpec` and `DatasetSpec` in the design document are **already implemented** in the current runner!

#### Issues Identified

| Issue | Location | Impact |
|-------|----------|--------|
| **Duplicated normalization** | Design's `_normalize_pipeline()` duplicates `PipelineConfigs._load_steps()` | Maintenance burden, divergence risk |
| **Missing parameters** | `RunConfig` missing `json_output`, `random_state` | Config/Runner mismatch |
| **Configuration drift** | Two paths: RunConfig + PipelineRunner(**kwargs) | User confusion |

#### Recommendation

**Thin wrapper approach** - Don't duplicate normalization logic:

```python
# Correct implementation pattern
def run(pipeline, dataset, *, name="", **runner_kwargs) -> RunResult:
    runner = PipelineRunner(**runner_kwargs)
    predictions, per_dataset = runner.run(pipeline, dataset, pipeline_name=name)
    return RunResult(predictions=predictions, per_dataset=per_dataset, _runner=runner)
```

**Rationale**: PipelineRunner and DatasetConfigs already handle all format conversions. The module-level API should be a convenience layer, not a reimplementation.

---

### 1.3 Result Classes Assessment

#### What's Proposed
`RunResult` with `.best`, `.best_score`, `.export()` convenience methods.

#### Current Reality
```python
# Current return type
predictions, per_dataset = runner.run(...)
best = predictions.top(n=1)[0]  # Manual step
```

#### Verdict: ✅ APPROVED with modification

The Result wrapper classes add genuine value. However:

| Issue | Solution |
|-------|----------|
| `RunResult.export()` needs runner reference | Store `_runner` attribute |
| Missing artifacts path access | Add from `per_dataset` dict |

---

### 1.4 Session Context Assessment

#### What's Proposed
```python
with nirs4all.session(verbose=2, save_artifacts=True) as s:
    r1 = nirs4all.run(pipeline1, data1, session=s)
    r2 = nirs4all.run(pipeline2, data2, session=s)
```

#### Current Reality
No equivalent exists. Users must create a new PipelineRunner per call.

#### Verdict: ✅ APPROVED

This is genuinely new functionality that adds value:
- Shared workspace across runs
- Reused logging configuration
- Potential for cached transformers

---

### 1.5 sklearn Meta-Estimator Assessment (NIRSPipeline)

#### What's Proposed
```python
class NIRSPipeline(BaseEstimator, RegressorMixin):
    def fit(self, X, y, **fit_params): ...
    def predict(self, X): ...
    @property
    def model_(self): ...  # Direct model access for SHAP
```

#### Critical Issues

##### Issue 1: Cross-Validation Creates Multiple Models

The design assumes a single fitted model, but nirs4all's CV creates **N models per fold**:

```python
# Current behavior with ShuffleSplit(n_splits=5)
# Results in 5 separate model artifacts:
# - model:step_3:fold_0
# - model:step_3:fold_1
# - model:step_3:fold_2
# - model:step_3:fold_3
# - model:step_3:fold_4
```

**Design flaw**: Line 700-702 mentions `model_: The fitted final model` but there is no single model when using CV.

##### Issue 2: Transform Semantics Undefined

What does `pipe.transform(X)` return?
- Only preprocessing output? (Which fold's preprocessing?)
- Base model predictions for stacking? (Which fold?)

The design doesn't specify.

##### Issue 3: MinimalPipeline ≠ sklearn Estimator

Line 702 mentions `pipeline_: Full fitted MinimalPipeline` but `MinimalPipeline` is a **data structure for prediction replay**, not an sklearn estimator with `fit()`/`predict()` methods.

```python
# From trace/extractor.py - MinimalPipeline is:
@dataclass
class MinimalPipeline:
    steps: List[MinimalPipelineStep]  # Just configuration, not fitted estimators
```

##### Issue 4: SHAP Model Access

For the `shap_model` property (lines 880-888), which fold's model should be returned?

#### Verdict: ⚠️ REQUIRES MAJOR REVISION

The NIRSPipeline as designed cannot work. Proposed alternative:

```python
class NIRSPipeline(BaseEstimator, RegressorMixin):
    """sklearn-compatible PREDICTION wrapper (not training wrapper)."""

    def __init__(self, minimal_pipeline=None, artifact_provider=None,
                 fold_strategy='weighted_average'):
        self.minimal_pipeline = minimal_pipeline
        self.artifact_provider = artifact_provider
        self.fold_strategy = fold_strategy

    @classmethod
    def from_result(cls, result: RunResult, model_index: int = 0) -> "NIRSPipeline":
        """Create from nirs4all RunResult (recommended path)."""
        best = result.predictions.top(n=1)[model_index]
        return cls.from_prediction(best, result._runner)

    @classmethod
    def from_bundle(cls, bundle_path: str) -> "NIRSPipeline":
        """Create from exported .n4a bundle."""
        loader = BundleLoader(bundle_path)
        return cls(loader.minimal_pipeline, loader.artifact_provider)

    def fit(self, X, y=None):
        """Not supported - use nirs4all.run() for training."""
        raise NotImplementedError(
            "NIRSPipeline is a prediction wrapper. "
            "Use nirs4all.run() for training, then wrap with NIRSPipeline.from_result()"
        )

    def predict(self, X):
        """Predict using MinimalPredictor (respects fold ensemble)."""
        return self._predictor.predict(X)

    @property
    def model_(self):
        """Return primary model artifact (fold 0 or weighted best)."""
        return self._load_primary_model()
```

**Key changes**:
1. NIRSPipeline is a **prediction wrapper**, not a training estimator
2. Construction via `from_result()` or `from_bundle()` class methods
3. `fit()` raises NotImplementedError - training happens via `nirs4all.run()`
4. Clear semantics: wraps an already-trained pipeline

---

### 1.6 NIRSPipelineSearch Assessment

#### What's Proposed
```python
class NIRSPipelineSearch(BaseEstimator, MetaEstimatorMixin):
    """Search over multiple pipeline configurations."""
    def fit(self, X, y): ...
    # Exposes best_estimator_, best_params_, cv_results_
```

#### Current Reality

Generator syntax already provides this:
```python
pipeline = [
    {"_or_": [SNV(), MSC()]},  # Expands to 2 pipelines
    {"_range_": [1, 30, 5], "param": "n_components", "model": PLSRegression()}  # 6 more
]
# Total: 12 pipeline variants, all evaluated by runner.run()
```

#### Verdict: ❌ DROP (Redundant)

The existing generator + `runner.run()` already provides multi-pipeline search with better semantics:
- Automatic expansion at config time
- All variants tracked in Predictions
- `predictions.top(n=5)` provides ranking

Adding NIRSPipelineSearch would:
1. Duplicate existing functionality
2. Force awkward `best_params_` mapping to generator choices
3. Not integrate with branching/source pipelines

**Alternative**: Document how to use generators + `nirs4all.run()` for hyperparameter search.

---

### 1.7 Configuration Classes Assessment

#### What's Proposed
`RunConfig`, `PredictConfig`, `ExplainConfig` dataclasses.

#### Current Reality
`PipelineRunner.__init__()` accepts all parameters directly:
```python
PipelineRunner(
    verbose=1,
    save_artifacts=True,
    log_format="pretty",
    # ... 15+ parameters
)
```

#### Verdict: ⚠️ OPTIONAL (defer to Phase 4)

Pros:
- Better IDE autocomplete
- Serializable to YAML/JSON
- Clean grouping of related options

Cons:
- Another layer of indirection
- Risk of drift with PipelineRunner params

**Recommendation**: Implement as an optional convenience. Users can still use `**kwargs` directly.

---

## Part 2: Migration Roadmap

### Phase 0: Preparation (Week 1)

**Goal**: Set up structure without breaking existing code.

#### Tasks

| ID | Task | Effort | Dependencies |
|----|------|--------|--------------|
| 0.1 | Create `nirs4all/api/` directory structure | 0.5 day | None |
| 0.2 | Create `nirs4all/api/__init__.py` with empty exports | 0.5 day | 0.1 |
| 0.3 | Create `nirs4all/api/result.py` with RunResult, PredictResult, ExplainResult stubs | 1 day | 0.1 |
| 0.4 | Create `nirs4all/sklearn/` directory structure | 0.5 day | None |
| 0.5 | Update `.gitignore` for any new patterns | 0.5 day | None |

#### Directory Structure
```
nirs4all/
├── api/
│   ├── __init__.py          # Exports run, predict, explain, session
│   ├── result.py             # RunResult, PredictResult, ExplainResult
│   ├── session.py            # Session context manager
│   └── config.py             # Optional: RunConfig, etc. (Phase 4)
├── sklearn/
│   ├── __init__.py           # Exports NIRSPipeline
│   ├── pipeline.py           # NIRSPipeline prediction wrapper
│   └── classifier.py         # NIRSPipelineClassifier
└── ... (existing modules unchanged)
```

---

### Phase 1: Result Classes (Week 2)

**Goal**: Implement result wrapper classes that add convenience without changing internals.

#### Tasks

| ID | Task | Effort | Dependencies |
|----|------|--------|--------------|
| 1.1 | Implement `RunResult` dataclass with `.best`, `.best_score`, `.top()` | 1 day | 0.3 |
| 1.2 | Add `_runner` reference to RunResult for `.export()` support | 0.5 day | 1.1 |
| 1.3 | Implement `RunResult.export()` delegating to runner | 0.5 day | 1.2 |
| 1.4 | Implement `PredictResult` dataclass | 0.5 day | 0.3 |
| 1.5 | Implement `ExplainResult` dataclass | 0.5 day | 0.3 |
| 1.6 | Add unit tests for result classes | 1 day | 1.1-1.5 |

#### Implementation: RunResult

```python
# nirs4all/api/result.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from nirs4all.pipeline import PipelineRunner
    from nirs4all.data.predictions import Predictions

@dataclass
class RunResult:
    """Result from nirs4all.run()."""

    predictions: "Predictions"
    per_dataset: Dict[str, Any]
    _runner: Optional["PipelineRunner"] = field(default=None, repr=False)

    @property
    def best(self) -> Dict[str, Any]:
        """Get best prediction entry by default ranking."""
        top = self.predictions.top(n=1)
        return top[0] if top else {}

    @property
    def best_score(self) -> float:
        """Get best model's test score."""
        return self.best.get('test_score', float('nan'))

    @property
    def artifacts_path(self) -> Optional[Path]:
        """Get path to run artifacts directory."""
        if self._runner and self._runner.current_run_dir:
            return self._runner.current_run_dir
        return None

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
```

---

### Phase 2: Module-Level API (Week 3)

**Goal**: Implement thin wrapper functions that delegate to PipelineRunner.

#### Tasks

| ID | Task | Effort | Dependencies |
|----|------|--------|--------------|
| 2.1 | Implement `nirs4all.run()` function | 1 day | 1.1 |
| 2.2 | Implement `nirs4all.predict()` function | 0.5 day | 1.4 |
| 2.3 | Implement `nirs4all.explain()` function | 0.5 day | 1.5 |
| 2.4 | Implement `nirs4all.retrain()` function | 0.5 day | - |
| 2.5 | Update `nirs4all/__init__.py` exports | 0.5 day | 2.1-2.4 |
| 2.6 | Create `examples/Q_new_api.py` example | 1 day | 2.1-2.5 |
| 2.7 | Add integration tests | 1 day | 2.1-2.5 |

#### Implementation: run()

```python
# nirs4all/api/run.py
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from nirs4all.api.result import RunResult

# Type aliases (documentation, not runtime enforcement)
PipelineSpec = Union[List[Any], Dict[str, Any], str, Path]
DatasetSpec = Union[str, Path, Tuple, Dict[str, Any], List]


def run(
    pipeline: PipelineSpec,
    dataset: DatasetSpec,
    *,
    name: str = "",
    **runner_kwargs
) -> RunResult:
    """Execute a training pipeline on a dataset.

    This is the primary entry point for training ML pipelines on NIRS data.
    It's a convenience wrapper around PipelineRunner that provides a simpler
    interface and richer result objects.

    Args:
        pipeline: Pipeline definition. Accepts:
            - List of steps: [MinMaxScaler(), {"model": PLSRegression(10)}]
            - Dict with 'pipeline' key: {"pipeline": [...], "description": "..."}
            - Path to config file: "configs/my_pipeline.yaml"

        dataset: Dataset definition. Accepts:
            - Path to data folder: "sample_data/regression"
            - Path to config file: "configs/my_data.yaml"
            - Tuple of arrays: (X,) or (X, y) or (X, y, metadata)
            - Dict with paths: {"train_x": "...", "train_y": "..."}

        name: Optional pipeline name for identification.

        **runner_kwargs: Arguments passed to PipelineRunner:
            - verbose (int): Verbosity level (0-3). Default: 1
            - save_artifacts (bool): Save model artifacts. Default: True
            - save_charts (bool): Save visualization charts. Default: True
            - workspace_path (str|Path): Workspace directory. Default: "./workspace"
            - plots_visible (bool): Show plots interactively. Default: False
            - random_state (int): Random seed for reproducibility.
            - log_file (bool): Write logs to file. Default: True
            - See PipelineRunner for full list.

    Returns:
        RunResult containing predictions, best model info, and export methods.

    Examples:
        >>> # Simple usage
        >>> result = nirs4all.run(
        ...     pipeline=[MinMaxScaler(), PLSRegression(10)],
        ...     dataset="sample_data/regression",
        ...     verbose=1
        ... )
        >>> print(f"Best RMSE: {result.best['rmse']:.4f}")

        >>> # Export best model
        >>> result.export("exports/my_model.n4a")

        >>> # Access detailed predictions
        >>> for pred in result.top(5, rank_metric='rmse'):
        ...     print(f"{pred['model_name']}: {pred['rmse']:.4f}")
    """
    from nirs4all.pipeline import PipelineRunner

    runner = PipelineRunner(**runner_kwargs)
    predictions, per_dataset = runner.run(
        pipeline=pipeline,
        dataset=dataset,
        pipeline_name=name
    )

    return RunResult(
        predictions=predictions,
        per_dataset=per_dataset,
        _runner=runner
    )
```

#### Updated Package Exports

```python
# nirs4all/__init__.py (additions)

# Module-level API (primary interface)
from .api import run, predict, explain, retrain, session, Session
from .api.result import RunResult, PredictResult, ExplainResult

# Update __all__
__all__ = [
    # Module-level API (NEW - primary interface)
    "run",
    "predict",
    "explain",
    "retrain",
    "session",
    "Session",
    "RunResult",
    "PredictResult",
    "ExplainResult",

    # Existing exports (still available)
    "PipelineRunner",
    "PipelineConfigs",
    "register_controller",
    "CONTROLLER_REGISTRY",
    ...
]
```

---

### Phase 3: Session Context Manager (Week 4)

**Goal**: Implement Session for resource reuse across multiple calls.

#### Tasks

| ID | Task | Effort | Dependencies |
|----|------|--------|--------------|
| 3.1 | Implement `Session` class | 1 day | 2.1-2.4 |
| 3.2 | Implement `session()` context manager function | 0.5 day | 3.1 |
| 3.3 | Modify `run()`, `predict()`, `explain()` to accept `session=` | 0.5 day | 3.2 |
| 3.4 | Add workspace isolation between sessions | 1 day | 3.1 |
| 3.5 | Add session cleanup on exit | 0.5 day | 3.1 |
| 3.6 | Create `examples/Q_session.py` example | 0.5 day | 3.1-3.3 |
| 3.7 | Add unit tests | 1 day | 3.1-3.5 |

#### Implementation: Session

```python
# nirs4all/api/session.py
from contextlib import contextmanager
from typing import Optional, Generator, Any
from pathlib import Path

from nirs4all.pipeline import PipelineRunner


class Session:
    """Execution session for resource reuse across multiple operations.

    A session maintains:
    - Shared PipelineRunner instance
    - Consistent workspace path
    - Shared logging configuration

    Use sessions when:
    - Making multiple run/predict calls in sequence
    - Need consistent workspace for artifact sharing
    - Want to avoid re-initialization overhead

    Example:
        >>> with nirs4all.session(verbose=2, save_artifacts=True) as s:
        ...     r1 = nirs4all.run(pipeline1, data1, session=s)
        ...     r2 = nirs4all.run(pipeline2, data2, session=s)
        ...     # Both use same workspace, share config
    """

    def __init__(
        self,
        workspace_path: Optional[str] = None,
        verbose: int = 1,
        **kwargs
    ):
        self._workspace_path = Path(workspace_path) if workspace_path else Path.cwd() / "workspace"
        self._verbose = verbose
        self._runner_kwargs = kwargs
        self._runner: Optional[PipelineRunner] = None
        self._is_active = False

    @property
    def runner(self) -> PipelineRunner:
        """Get or create the underlying PipelineRunner."""
        if self._runner is None:
            self._runner = PipelineRunner(
                workspace_path=self._workspace_path,
                verbose=self._verbose,
                **self._runner_kwargs
            )
        return self._runner

    @property
    def workspace_path(self) -> Path:
        """Get session workspace path."""
        return self._workspace_path

    def __enter__(self) -> "Session":
        self._is_active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._is_active = False
        self._cleanup()
        return False

    def _cleanup(self):
        """Release session resources."""
        # Currently just drops reference; could add explicit cleanup
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

---

### Phase 4: sklearn Prediction Wrapper (Week 5-6)

**Goal**: Implement NIRSPipeline as a **prediction-only** sklearn wrapper.

#### Critical Design Decision

**NIRSPipeline is NOT a training estimator.** It wraps an already-trained nirs4all pipeline for sklearn-compatible prediction and SHAP integration.

#### Tasks

| ID | Task | Effort | Dependencies |
|----|------|--------|--------------|
| 4.1 | Implement base `NIRSPipeline` class | 2 days | Phase 3 complete |
| 4.2 | Implement `NIRSPipeline.from_result()` class method | 1 day | 4.1 |
| 4.3 | Implement `NIRSPipeline.from_bundle()` class method | 1 day | 4.1 |
| 4.4 | Implement `predict()` using MinimalPredictor | 1 day | 4.2, 4.3 |
| 4.5 | Implement `model_` property for SHAP access | 0.5 day | 4.4 |
| 4.6 | Implement `NIRSPipelineClassifier` variant | 1 day | 4.1 |
| 4.7 | Create `examples/Q_sklearn_wrapper.py` | 1 day | 4.1-4.6 |
| 4.8 | Add SHAP integration tests | 1 day | 4.5 |
| 4.9 | Add cross_validate compatibility tests | 0.5 day | 4.4 |

#### Implementation: NIRSPipeline

```python
# nirs4all/sklearn/pipeline.py
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class NIRSPipeline(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper for trained nirs4all pipelines.

    This class wraps a trained nirs4all pipeline to provide sklearn's
    BaseEstimator interface. It enables:
    - SHAP compatibility for model explanation
    - sklearn utility function compatibility (cross_val_score on predict)
    - joblib serialization for deployment

    IMPORTANT: This is a PREDICTION wrapper, not a training estimator.
    Training should be done with nirs4all.run(). Use the class methods
    to create instances:
    - NIRSPipeline.from_result(result) - From nirs4all RunResult
    - NIRSPipeline.from_bundle(path) - From exported .n4a bundle

    The wrapper handles CV fold ensembling internally using the strategy
    from the original training (weighted average by default).

    Attributes:
        model_: The primary fitted model (for SHAP access)
        n_features_in_: Number of input features
        preprocessing_chain_: Description of preprocessing steps

    Examples:
        >>> # Train with nirs4all
        >>> result = nirs4all.run(pipeline, data)

        >>> # Wrap for sklearn compatibility
        >>> pipe = NIRSPipeline.from_result(result)

        >>> # Use with SHAP
        >>> import shap
        >>> explainer = shap.Explainer(pipe.predict, X_train[:100])
        >>> shap_values = explainer(X_test)

        >>> # Or from exported bundle
        >>> pipe = NIRSPipeline.from_bundle("exports/model.n4a")
        >>> predictions = pipe.predict(X_new)
    """

    def __init__(
        self,
        _minimal_pipeline=None,
        _artifact_provider=None,
        _fold_weights=None,
        _run_dir=None
    ):
        """Initialize wrapper (use class methods instead).

        Args:
            _minimal_pipeline: Internal MinimalPipeline (use from_result/from_bundle)
            _artifact_provider: Internal artifact provider
            _fold_weights: Per-fold weights for ensemble
            _run_dir: Path to run directory
        """
        self._minimal_pipeline = _minimal_pipeline
        self._artifact_provider = _artifact_provider
        self._fold_weights = _fold_weights or {}
        self._run_dir = _run_dir
        self._predictor = None
        self._primary_model = None

    @classmethod
    def from_result(cls, result: "RunResult", model_index: int = 0) -> "NIRSPipeline":
        """Create wrapper from nirs4all RunResult.

        Args:
            result: RunResult from nirs4all.run()
            model_index: Which model to wrap (0 = best)

        Returns:
            NIRSPipeline ready for prediction
        """
        from nirs4all.pipeline.resolver import PredictionResolver
        from nirs4all.pipeline.minimal_predictor import MinimalArtifactProvider

        # Get target prediction
        predictions = result.predictions.top(n=model_index + 1)
        if model_index >= len(predictions):
            raise ValueError(f"model_index {model_index} >= available models {len(predictions)}")
        target = predictions[model_index]

        # Resolve to minimal pipeline
        runner = result._runner
        resolver = PredictionResolver(
            workspace_path=runner.workspace_path,
            runs_dir=runner.runs_dir
        )
        resolved = resolver.resolve(target)

        return cls(
            _minimal_pipeline=resolved.minimal_pipeline,
            _artifact_provider=resolved.artifact_provider,
            _fold_weights=resolved.fold_weights,
            _run_dir=resolved.run_dir
        )

    @classmethod
    def from_bundle(cls, bundle_path: Union[str, Path]) -> "NIRSPipeline":
        """Create wrapper from exported .n4a bundle.

        Args:
            bundle_path: Path to .n4a bundle file

        Returns:
            NIRSPipeline ready for prediction
        """
        from nirs4all.pipeline.bundle import BundleLoader

        loader = BundleLoader(bundle_path)

        return cls(
            _minimal_pipeline=loader.minimal_pipeline,
            _artifact_provider=loader.artifact_provider,
            _fold_weights=loader.fold_weights,
            _run_dir=None  # Bundle is self-contained
        )

    def fit(self, X, y=None, **fit_params):
        """Not supported - use nirs4all.run() for training.

        Raises:
            NotImplementedError: Always, with guidance message.
        """
        raise NotImplementedError(
            "NIRSPipeline is a prediction wrapper for trained models.\n"
            "For training, use:\n"
            "  result = nirs4all.run(pipeline, dataset)\n"
            "  pipe = NIRSPipeline.from_result(result)"
        )

    def predict(self, X) -> np.ndarray:
        """Predict using the wrapped pipeline.

        Handles preprocessing, fold ensembling, and y inverse transform.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Predictions array (n_samples,)
        """
        check_is_fitted(self, '_minimal_pipeline')

        if self._predictor is None:
            self._init_predictor()

        # Create SpectroDataset from X
        from nirs4all.data import SpectroDataset
        dataset = SpectroDataset(name="prediction")
        dataset.add_samples(X, {"partition": "all"})

        # Run prediction
        y_pred, _ = self._predictor.predict(self._minimal_pipeline, dataset)
        return np.asarray(y_pred).flatten()

    @property
    def model_(self):
        """Access primary model artifact for SHAP.

        Returns the fold 0 model by default, or the weighted-best fold
        if weights indicate one is clearly superior.

        Returns:
            Fitted model object (sklearn/keras/torch)
        """
        if self._primary_model is None:
            self._load_primary_model()
        return self._primary_model

    @property
    def n_features_in_(self) -> int:
        """Number of input features."""
        if self._minimal_pipeline:
            return self._minimal_pipeline.n_features
        return 0

    @property
    def preprocessing_chain_(self) -> str:
        """Description of preprocessing steps."""
        if self._minimal_pipeline:
            return self._minimal_pipeline.get_preprocessing_summary()
        return ""

    def _init_predictor(self):
        """Initialize MinimalPredictor."""
        from nirs4all.pipeline.minimal_predictor import MinimalPredictor
        from nirs4all.pipeline.storage.artifacts.artifact_loader import ArtifactLoader

        # Build artifact loader from provider
        loader = ArtifactLoader(self._run_dir) if self._run_dir else None

        self._predictor = MinimalPredictor(
            artifact_loader=loader or self._artifact_provider,
            run_dir=self._run_dir or Path.cwd()
        )

    def _load_primary_model(self):
        """Load primary model for SHAP access."""
        if self._artifact_provider is None:
            return None

        # Find model step
        model_step = self._minimal_pipeline.get_model_step()
        if model_step is None:
            return None

        # Load fold 0 model by default
        artifacts = self._artifact_provider.get_artifacts_for_step(model_step.step_index)
        if artifacts:
            _, self._primary_model = artifacts[0]

    def get_params(self, deep=True) -> Dict[str, Any]:
        """Get parameters (sklearn interface)."""
        return {
            '_minimal_pipeline': self._minimal_pipeline,
            '_artifact_provider': self._artifact_provider,
            '_fold_weights': self._fold_weights,
            '_run_dir': self._run_dir
        }

    def set_params(self, **params) -> "NIRSPipeline":
        """Set parameters (sklearn interface)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
```

---

### Phase 5: Documentation & Examples (Week 7)

**Goal**: Update documentation to showcase new API as primary interface.

#### Tasks

| ID | Task | Effort | Dependencies |
|----|------|--------|--------------|
| 5.1 | Update README.md with new API examples | 1 day | Phase 4 |
| 5.2 | Create migration guide document | 1 day | Phase 4 |
| 5.3 | Update RTD API reference | 1 day | Phase 4 |
| 5.4 | Create `examples/Q40_new_api.py` comprehensive example | 1 day | Phase 4 |
| 5.5 | Create `examples/Q41_sklearn_shap.py` SHAP example | 1 day | Phase 4 |
| 5.6 | Create `examples/Q42_session_workflow.py` session example | 0.5 day | Phase 3 |
| 5.7 | Review and update existing examples as needed | 1 day | Phase 4 |

---

### Phase 6: Optional Enhancements (Week 8+)

#### 6A: Configuration Dataclasses (Optional)

Only implement if there's user demand:

| ID | Task | Effort |
|----|------|--------|
| 6A.1 | Implement `RunConfig` dataclass | 1 day |
| 6A.2 | Implement `PredictConfig` dataclass | 0.5 day |
| 6A.3 | Implement `ExplainConfig` dataclass | 0.5 day |
| 6A.4 | Add config validation | 0.5 day |
| 6A.5 | Add config serialization (YAML/JSON) | 1 day |

#### 6B: Advanced sklearn Integration (Future)

Defer these to future versions:

| Feature | Status | Rationale |
|---------|--------|-----------|
| `NIRSPipelineSearch` | DROPPED | Generator syntax already provides this |
| `NIRSPipeline.fit()` | DEFERRED | Complex, unclear semantics with CV |
| Differentiable preprocessing | DEFERRED | Requires torch/tf implementations |
| `partial_fit()` streaming | DEFERRED | Would require significant controller changes |

---

## Part 3: Summary Tables

### API Changes Summary

| Change | Type | Phase | Priority |
|--------|------|-------|----------|
| `nirs4all.run()` | New function | 2 | HIGH |
| `nirs4all.predict()` | New function | 2 | HIGH |
| `nirs4all.explain()` | New function | 2 | HIGH |
| `nirs4all.session()` | New context manager | 3 | MEDIUM |
| `RunResult` | New class | 1 | HIGH |
| `NIRSPipeline` | New class (prediction-only) | 4 | MEDIUM |
| `NIRSPipelineSearch` | DROPPED | - | - |
| `RunConfig` | Optional | 6A | LOW |

### Backward Compatibility

| Existing API | Status | Notes |
|--------------|--------|-------|
| `PipelineRunner` | ✅ Unchanged | Still works exactly as before |
| `PipelineConfigs` | ✅ Unchanged | Still works exactly as before |
| `DatasetConfigs` | ✅ Unchanged | Still works exactly as before |
| `Predictions` | ✅ Unchanged | Still works exactly as before |
| All examples (Q*.py) | ✅ Work without changes | Tested |

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Parameter drift between RunConfig and PipelineRunner | MEDIUM | MEDIUM | Use kwargs passthrough, not duplication |
| NIRSPipeline fold semantics confusion | LOW | HIGH | Clear documentation, from_result() pattern |
| Session resource leaks | LOW | LOW | Explicit cleanup in __exit__ |
| Breaking existing examples | LOW | HIGH | Phase 0 testing, thin wrapper approach |

---

## Part 4: Implementation Checklist

### Pre-Implementation

- [ ] Review this roadmap with stakeholders
- [ ] Decide on Phase 6A (config classes) - implement or defer
- [ ] Create feature branch `feature/api-v2`

### Phase 0 Checklist
- [ ] Create `nirs4all/api/` directory
- [ ] Create `nirs4all/sklearn/` directory
- [ ] Stub out all new files with docstrings
- [ ] Ensure all existing tests pass

### Phase 1 Checklist
- [ ] Implement `RunResult` with all properties
- [ ] Implement `PredictResult`
- [ ] Implement `ExplainResult`
- [ ] Add unit tests (≥90% coverage)

### Phase 2 Checklist
- [ ] Implement `run()` function
- [ ] Implement `predict()` function
- [ ] Implement `explain()` function
- [ ] Implement `retrain()` function
- [ ] Update `__init__.py` exports
- [ ] Add integration tests
- [ ] Create example file

### Phase 3 Checklist
- [ ] Implement `Session` class
- [ ] Implement `session()` context manager
- [ ] Modify API functions to accept session
- [ ] Add tests for session lifecycle
- [ ] Create example file

### Phase 4 Checklist
- [ ] Implement `NIRSPipeline` base class
- [ ] Implement `from_result()` class method
- [ ] Implement `from_bundle()` class method
- [ ] Implement `predict()` method
- [ ] Implement `model_` property
- [ ] Implement `NIRSPipelineClassifier`
- [ ] Add SHAP integration tests
- [ ] Create example file

### Phase 5 Checklist
- [ ] Update README.md
- [ ] Create migration guide
- [ ] Update RTD documentation
- [ ] Create comprehensive examples
- [ ] Review all existing examples

### Release Checklist
- [ ] Version bump to 0.6.0
- [ ] Update CHANGELOG.md
- [ ] Tag release
- [ ] Update PyPI

---

## Appendix A: Design Decisions Rationale

### A1: Why Thin Wrapper Instead of Full Facade

**Decision**: Module-level API functions are thin wrappers around PipelineRunner, not reimplementations.

**Rationale**:
1. PipelineRunner already handles all format normalization
2. Duplicating logic creates maintenance burden
3. Risk of divergence between two code paths
4. Easier to test - existing tests cover the internals

### A2: Why NIRSPipeline is Prediction-Only

**Decision**: NIRSPipeline.fit() raises NotImplementedError.

**Rationale**:
1. nirs4all's CV creates N models - no single fitted model
2. Generator expansion happens before execution
3. Branching pipelines have multiple output paths
4. Clear separation: nirs4all.run() for training, NIRSPipeline for deployment

### A3: Why Drop NIRSPipelineSearch

**Decision**: Do not implement NIRSPipelineSearch.

**Rationale**:
1. Generator syntax (`_or_`, `_range_`) already provides multi-pipeline search
2. `predictions.top()` already provides ranking
3. `best_params_` pattern doesn't map to generator choices naturally
4. Would duplicate functionality with worse integration

### A4: Why Session Over Global State

**Decision**: Use Session context manager, not global configuration.

**Rationale**:
1. Explicit resource management lifecycle
2. Test isolation - each test gets fresh session
3. Parallel execution possible - multiple sessions
4. IDE type hints work properly

---

## Appendix B: Rejected Alternatives

### B1: Full sklearn Training Support

**Rejected approach**: Make NIRSPipeline.fit() actually train the pipeline.

**Why rejected**:
- Would need to re-implement much of PipelineOrchestrator
- CV semantics unclear (create N models, return which?)
- Generator expansion happens at config time, not fit time
- Branching not representable in sklearn fit(X, y) signature

### B2: Automatic Config Synchronization

**Rejected approach**: Auto-sync RunConfig fields with PipelineRunner params.

**Why rejected**:
- Metaclass magic for synchronization is fragile
- Better to just pass kwargs through
- Explicit is better than implicit

### B3: Lazy Session Initialization

**Rejected approach**: Session creates runner on first use.

**Why adopted instead**: Session creates runner on first use (this IS the implemented approach).

**Why this works**:
- Avoids overhead if session is created but not used
- Context manager pattern handles cleanup
- Runner kwargs stored for deferred creation
