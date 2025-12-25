# NIRS4ALL v2.0: API Layer Design

**Author**: GitHub Copilot (Claude Opus 4.5)
**Date**: December 25, 2025
**Status**: Design Proposal (Revised per Critical Review)
**Document**: 4 of 5

---

## Table of Contents

1. [Overview](#overview)
2. [Static Module API](#static-module-api)
3. [Result Objects](#result-objects)
4. [sklearn Estimators](#sklearn-estimators)
5. [Session Management](#session-management)
6. [Explainer Abstraction](#explainer-abstraction)
7. [UX Syntax Flexibility](#ux-syntax-flexibility)
8. [CLI Interface](#cli-interface)
9. [Configuration Files](#configuration-files)

---

## Overview

The API Layer provides three levels of access to nirs4all:

1. **Static Functions**: `nirs4all.run()`, `nirs4all.predict()`, `nirs4all.explain()`
2. **sklearn Estimators**: `NIRSRegressor`, `NIRSClassifier`, `NIRSSearchCV`
3. **CLI/Config**: Command-line and YAML-based pipeline definition

### Design Goals

1. **Simple Default Case**: One function call for common workflows
2. **sklearn Compatibility**: Works with GridSearchCV, SHAP, cross_validate
3. **Progressive Complexity**: Simple → Advanced without API breaks
4. **Discoverable**: Good IDE autocomplete and documentation
5. **Preserve v1 UX**: All syntax forms from v1 remain valid
6. **Lazy by Default**: Only load/compute what's accessed

### API Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Code                                 │
│                                                                  │
│  Simple:     nirs4all.run(pipeline, data)                       │
│  sklearn:    NIRSRegressor(pipeline).fit(X, y).predict(X_new)   │
│  Advanced:   with nirs4all.session() as s: ...                  │
│  CLI:        nirs4all run --config pipeline.yaml data/          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DAG Execution Engine                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Static Module API

### Module Interface

```python
# nirs4all/__init__.py

from .api import run, predict, explain, retrain
from .api import session, RunResult, PredictResult, ExplainResult
from .sklearn import NIRSRegressor, NIRSClassifier, NIRSSearchCV

__all__ = [
    # Functions
    "run", "predict", "explain", "retrain",
    # Session
    "session",
    # Results
    "RunResult", "PredictResult", "ExplainResult",
    # sklearn estimators
    "NIRSRegressor", "NIRSClassifier", "NIRSSearchCV",
]
```

### run() - Training

```python
def run(
    pipeline: PipelineSpec,
    data: DataSpec,
    *,
    name: str = "",
    cv: Optional[CVSpec] = None,
    verbose: int = 1,
    save_artifacts: bool = True,
    workspace: Optional[str] = None,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    **kwargs
) -> RunResult:
    """Execute a training pipeline on data.

    This is the primary entry point for nirs4all. It compiles the pipeline
    to a DAG, executes it on the provided data, and returns ranked results.

    Args:
        pipeline: Pipeline specification. Can be:
            - List of steps: [MinMaxScaler(), PLSRegression()]
            - Path to YAML: "pipeline.yaml"
            - Dict with steps: {"steps": [...], "name": "my_pipe"}
        data: Data specification. Can be:
            - Path to file: "data.csv" or "data.mat"
            - Tuple of arrays: (X, y)
            - Dict: {"X": X, "y": y, "metadata": {...}}
            - DatasetContext object
        name: Pipeline name for tracking (auto-generated if empty)
        cv: Cross-validation specification. Can be:
            - int: Number of folds (uses KFold)
            - sklearn splitter: ShuffleSplit(n_splits=5)
            - None: Uses default 5-fold
        verbose: Verbosity level (0=silent, 1=progress, 2=debug)
        save_artifacts: If True, save trained models and artifacts
        workspace: Output directory (default: ./workspace)
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (1=sequential)
        **kwargs: Additional options passed to execution engine

    Returns:
        RunResult with predictions, artifacts, and convenience methods

    Examples:
        >>> # Simple usage
        >>> result = nirs4all.run(
        ...     [MinMaxScaler(), PLSRegression(n_components=10)],
        ...     "path/to/data.csv"
        ... )
        >>> print(result.best)

        >>> # With cross-validation
        >>> result = nirs4all.run(
        ...     [SNV(), {"model": PLSRegression()}],
        ...     (X, y),
        ...     cv=ShuffleSplit(n_splits=5, random_state=42)
        ... )

        >>> # With generators
        >>> result = nirs4all.run(
        ...     [
        ...         {"_or_": [SNV(), MSC(), Detrend()]},
        ...         {"model": PLSRegression(),
        ...          "n_components": {"_range_": [1, 30, 1]}}
        ...     ],
        ...     data
        ... )
        >>> result.top(5)  # Best 5 configurations
    """
    # Normalize inputs
    context = _normalize_data(data)
    dag = _build_dag(pipeline, cv)

    # Configure engine
    engine = ExecutionEngine(
        parallel=n_jobs > 1,
        n_workers=n_jobs,
        verbose=verbose
    )

    # Execute
    exec_result = engine.execute(dag, context, mode="train")

    # Save if requested
    if save_artifacts:
        artifact_manager = ArtifactManager(Path(workspace or "./workspace"))
        for node_id, artifacts in exec_result.artifacts.items():
            for name, obj in artifacts.items():
                artifact_manager.save(node_id, name, obj, dag.nodes[node_id].lineage_hash)

    return RunResult(
        predictions=exec_result.prediction_store,
        artifacts=exec_result.artifacts,
        context=exec_result.context,
        dag=dag
    )


# Type aliases for flexibility
PipelineSpec = Union[
    List[Any],              # Direct step list
    str,                    # Path to YAML
    Path,                   # Path object
    Dict[str, Any],         # Dict with "steps" key
    ExecutableDAG           # Pre-built DAG
]

DataSpec = Union[
    str,                    # Path to data file
    Path,                   # Path object
    Tuple[np.ndarray, np.ndarray],  # (X, y)
    Dict[str, Any],         # {"X": X, "y": y, ...}
    DatasetContext          # Pre-built context
]

CVSpec = Union[
    int,                    # Number of folds
    BaseCrossValidator,     # sklearn splitter
    None                    # Use default
]
```

### predict() - Inference

```python
def predict(
    model: ModelSpec,
    data: DataSpec,
    *,
    verbose: int = 0,
    return_all_folds: bool = False,
    aggregation: str = "weighted_mean",
    fold: Optional[Union[int, str]] = None,
    lazy_load: bool = True,
    **kwargs
) -> PredictResult:
    """Apply trained model to new data.

    Args:
        model: Trained model specification. Can be:
            - RunResult: Uses best model from training run
            - str: Path to exported bundle (.n4a)
            - Dict: {"artifact_path": "...", "model_name": "..."}
            - VirtualModel: Direct model object
        data: Data specification (same as run())
        verbose: Verbosity level
        return_all_folds: If True, return predictions from all fold models
        aggregation: How to aggregate fold predictions
            ("mean", "weighted_mean", "median")
        fold: Which fold to use for prediction:
            - None: Use all folds with aggregation (default)
            - int: Use specific fold index (0, 1, 2, ...)
            - "best": Use the best-scoring fold model only
            - "random": Use a random fold (for diversity)
        lazy_load: If True, only load required fold models (default).
            Set to False to load all models upfront.
        **kwargs: Additional options

    Returns:
        PredictResult with predictions and metadata

    Examples:
        >>> # From training result
        >>> train_result = nirs4all.run(pipeline, train_data)
        >>> pred_result = nirs4all.predict(train_result, test_data)
        >>> y_pred = pred_result.y_pred

        >>> # From exported bundle (fast: loads only best fold)
        >>> pred_result = nirs4all.predict(
        ...     "model.n4a", new_data, fold="best"
        ... )

        >>> # Get all fold predictions
        >>> pred_result = nirs4all.predict(
        ...     model, data, return_all_folds=True
        ... )
        >>> fold_preds = pred_result.fold_predictions  # (n_folds, n_samples)
    """
    # Load model with optional lazy loading
    virtual_model, dag, artifacts = _load_model(
        model,
        lazy=lazy_load,
        fold=fold
    )

    # Normalize data
    context = _normalize_data(data)

    # Build minimal DAG for prediction (only required nodes)
    minimal_dag = _build_minimal_dag(dag, artifacts)

    # Execute in predict mode
    engine = ExecutionEngine(verbose=verbose)
    exec_result = engine.execute(
        minimal_dag, context, mode="predict", artifacts=artifacts
    )

    return PredictResult(
        y_pred=exec_result.prediction_store.get_latest()["y_pred"],
        fold_predictions=_get_fold_predictions(exec_result) if return_all_folds else None,
        model=virtual_model,
        context=context
    )


def _load_model(
    model: ModelSpec,
    lazy: bool = True,
    fold: Optional[Union[int, str]] = None
) -> Tuple[VirtualModel, ExecutableDAG, Dict]:
    """Load model with optional lazy/partial loading.

    Args:
        model: Model specification
        lazy: If True, only load what's needed
        fold: Specific fold to load (None = all)

    Returns:
        Tuple of (VirtualModel, DAG, artifacts)
    """
    if isinstance(model, VirtualModel):
        return model, None, {}

    if isinstance(model, RunResult):
        # From RunResult: artifacts may already be cached
        return model.best_model, model.dag, model.artifacts

    if isinstance(model, (str, Path)):
        # From bundle file
        from .bundle import BundleLoader

        loader = BundleLoader(model)

        if fold is not None and lazy:
            # Lazy: only load specific fold
            virtual_model = loader.load_fold_model(fold)
        else:
            # Load all folds
            virtual_model = loader.load_virtual_model()

        dag = loader.load_dag() if not lazy else None
        artifacts = loader.load_artifacts() if not lazy else {}

        return virtual_model, dag, artifacts

    raise TypeError(f"Cannot load model from {type(model)}")


ModelSpec = Union[
    RunResult,              # Training result
    str,                    # Path to bundle
    Path,                   # Path object
    Dict[str, Any],         # Artifact specification
    VirtualModel            # Direct model
]
```

### explain() - Explanations

```python
def explain(
    model: ModelSpec,
    data: DataSpec,
    *,
    method: str = "auto",
    n_samples: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    verbose: int = 0,
    **kwargs
) -> ExplainResult:
    """Generate SHAP explanations for model predictions.

    Args:
        model: Trained model (same as predict())
        data: Data to explain
        method: SHAP method to use:
            - "auto": Automatically select based on model type
            - "tree": TreeExplainer for tree-based models
            - "deep": DeepExplainer for neural networks
            - "kernel": KernelExplainer (slow, model-agnostic)
            - "linear": LinearExplainer for linear models
        n_samples: Number of samples for background (kernel method)
        feature_names: Names for features in explanations
        verbose: Verbosity level
        **kwargs: Passed to SHAP explainer

    Returns:
        ExplainResult with SHAP values and visualization helpers

    Examples:
        >>> # Auto-detect method
        >>> result = nirs4all.explain(trained_model, test_data)
        >>> result.summary_plot()

        >>> # Force kernel method
        >>> result = nirs4all.explain(
        ...     model, data, method="kernel", n_samples=100
        ... )

        >>> # With feature names (wavelengths)
        >>> result = nirs4all.explain(
        ...     model, data,
        ...     feature_names=[f"{w} nm" for w in wavelengths]
        ... )
        >>> result.waterfall_plot(sample_idx=0)
    """
    import shap

    # Load model
    virtual_model, dag, artifacts = _load_model(model)

    # Normalize data
    context = _normalize_data(data)
    X = context.x(layout="2d")

    # Select explainer
    explainer_cls = _select_shap_explainer(virtual_model, method)

    # Build explainer
    if method == "kernel" or explainer_cls == shap.KernelExplainer:
        background = shap.sample(X, n_samples or 100)
        explainer = explainer_cls(virtual_model.predict, background)
    else:
        explainer = explainer_cls(virtual_model.primary_model)

    # Compute SHAP values
    shap_values = explainer.shap_values(X)

    return ExplainResult(
        shap_values=shap_values,
        expected_value=explainer.expected_value,
        X=X,
        feature_names=feature_names or context.feature_names,
        model=virtual_model
    )
```

### retrain() - Model Update

```python
def retrain(
    model: ModelSpec,
    new_data: DataSpec,
    *,
    mode: str = "finetune",
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    freeze_layers: Optional[List[str]] = None,
    verbose: int = 1,
    **kwargs
) -> RunResult:
    """Retrain or fine-tune a model with new data.

    Args:
        model: Trained model to update
        new_data: New data for retraining
        mode: Retraining mode:
            - "finetune": Continue training from current weights
            - "transfer": Freeze early layers, train later layers
            - "full": Retrain all parameters from current state
        epochs: Number of training epochs (for NN models)
        learning_rate: Learning rate (typically lower for finetuning)
        freeze_layers: Layer names to freeze (for transfer mode)
        verbose: Verbosity level
        **kwargs: Additional training options

    Returns:
        RunResult with updated model

    Examples:
        >>> # Fine-tune with new data
        >>> original = nirs4all.run(pipeline, original_data)
        >>> updated = nirs4all.retrain(
        ...     original, new_data, mode="finetune", epochs=10
        ... )

        >>> # Transfer learning
        >>> updated = nirs4all.retrain(
        ...     model, new_data,
        ...     mode="transfer",
        ...     freeze_layers=["conv1", "conv2"]
        ... )
    """
    ...
```

---

## Result Objects

### Design Principles

Per the critical review, result objects follow these principles:

1. **Lazy by Default**: Heavy objects (artifacts, DAG) load on first access
2. **Simple for Simple Cases**: `result.best` and `result.y_pred` work immediately
3. **Full Access Available**: All data accessible when needed
4. **Memory Efficient**: Dropped references are garbage collected

### RunResult

```python
from functools import cached_property


class RunResult:
    """Result from nirs4all.run().

    Provides convenient access to predictions, best models,
    and export functionality.

    This class uses lazy loading for heavy objects. Accessing
    `.artifacts` or `.dag` loads them from disk on first access.
    For simple use cases, only `.best` and `.y_pred` are needed.

    Attributes:
        predictions: Lightweight prediction metadata (always loaded)
        _artifacts_path: Path to artifacts (loaded lazily via .artifacts)
        _dag_path: Path to DAG (loaded lazily via .dag)
        context: Dataset context reference

    Examples:
        >>> result = nirs4all.run(pipeline, data)
        >>> # Simple access (no heavy loading)
        >>> print(result.best_score)
        >>> y_pred = result.y_pred
        >>>
        >>> # Full access (loads artifacts on first access)
        >>> model = result.best_model  # Loads VirtualModel
        >>> result.export("model.n4a")  # Uses loaded artifacts
    """

    def __init__(
        self,
        predictions: PredictionStore,
        artifacts_path: Optional[Path],
        dag_path: Optional[Path],
        context: DatasetContext,
        # For in-memory results (no lazy loading)
        _artifacts: Optional[Dict[str, Dict[str, Any]]] = None,
        _dag: Optional[ExecutableDAG] = None
    ):
        self.predictions = predictions
        self._artifacts_path = artifacts_path
        self._dag_path = dag_path
        self.context = context
        self.__artifacts = _artifacts
        self.__dag = _dag

    # ===== Lightweight Properties (No Lazy Loading) =====

    @property
    def best(self) -> "PredictionResult":
        """Get the best prediction (lightweight metadata)."""
        return PredictionResult.from_dict(
            self.predictions.get_best()
        )

    @property
    def best_score(self) -> float:
        """Get the best validation score."""
        return self.best.val_score

    @property
    def y_pred(self) -> np.ndarray:
        """Get predictions from best model (OOF reconstructed)."""
        return self.best.y_pred

    @property
    def y_true(self) -> np.ndarray:
        """Get true values aligned with y_pred."""
        return self.best.y_true

    # ===== Lazy-Loaded Properties =====

    @cached_property
    def artifacts(self) -> Dict[str, Dict[str, Any]]:
        """Load artifacts from disk (cached after first access).

        Returns:
            Dict mapping node_id -> artifact_name -> artifact
        """
        if self.__artifacts is not None:
            return self.__artifacts
        if self._artifacts_path is None:
            return {}
        return self._load_artifacts(self._artifacts_path)

    @cached_property
    def dag(self) -> ExecutableDAG:
        """Load DAG from disk (cached after first access)."""
        if self.__dag is not None:
            return self.__dag
        if self._dag_path is None:
            raise ValueError("DAG not available (run with save_artifacts=True)")
        return self._load_dag(self._dag_path)

    @cached_property
    def best_model(self) -> VirtualModel:
        """Get the best virtual model (triggers artifact loading)."""
        best = self.best
        return self._get_model_for_prediction(best)

    def _load_artifacts(self, path: Path) -> Dict:
        """Load artifacts from disk."""
        import joblib
        return joblib.load(path)

    def _load_dag(self, path: Path) -> ExecutableDAG:
        """Load DAG from disk."""
        import joblib
        return joblib.load(path)

    def _get_model_for_prediction(self, pred: "PredictionResult") -> VirtualModel:
        """Retrieve VirtualModel for a prediction."""
        node_id = pred.metadata.get("node_id")
        if node_id and node_id in self.artifacts:
            return self.artifacts[node_id].get("virtual_model")
        raise ValueError(f"Model not found for prediction {pred.id}")

    # ===== Utility Methods =====

    def top(
        self,
        n: int = 5,
        metric: str = "val_score",
        ascending: Optional[bool] = None
    ) -> List["PredictionResult"]:
        """Get top n predictions by metric.

        Args:
            n: Number of predictions to return
            metric: Metric to rank by
            ascending: Sort order (None = infer from metric)

        Returns:
            List of PredictionResult objects
        """
        return [
            PredictionResult.from_dict(p)
            for p in self.predictions.top(n, rank_metric=metric, ascending=ascending)
        ]

    def export(
        self,
        path: Union[str, Path],
        model: Optional["PredictionResult"] = None,
        include_data: bool = False
    ) -> Path:
        """Export model as bundle (.n4a).

        Args:
            path: Output path
            model: Which model to export (default: best)
            include_data: Include training data in bundle

        Returns:
            Path to exported bundle
        """
        from .bundle import BundleExporter

        model = model or self.best
        exporter = BundleExporter(self.artifacts, self.dag)
        return exporter.export(path, model, include_data)

    def compare(
        self,
        metrics: Optional[List[str]] = None,
        partition: str = "test"
    ) -> pl.DataFrame:
        """Compare all pipeline variants.

        Returns DataFrame with metrics for each configuration.
        """
        return self.predictions.compare_table(
            metrics=metrics or ["mse", "rmse", "r2"],
            partition=partition
        )

    def __repr__(self) -> str:
        n_preds = len(self.predictions)
        best = self.best
        return (
            f"RunResult({n_preds} predictions, "
            f"best: {best.model_name} @ {best.val_score:.4f})"
        )


@dataclass
class PredictionResult:
    """Single prediction entry with metadata."""
    id: str
    model_name: str
    model_class: str
    partition: str
    fold_id: Optional[int]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]
    val_score: Optional[float]
    test_score: Optional[float]
    train_score: Optional[float]
    metric: str
    task_type: str
    preprocessings: str
    branch_id: Optional[int]
    branch_path: List[int]
    sample_indices: List[int]
    metadata: Dict[str, Any]

    @classmethod
    def from_dict(cls, d: Dict) -> "PredictionResult":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})

    @property
    def scores(self) -> Dict[str, float]:
        """All computed scores."""
        return {
            "val": self.val_score,
            "test": self.test_score,
            "train": self.train_score
        }

    def save_to_csv(self, path: Union[str, Path]) -> None:
        """Save predictions to CSV."""
        df = pl.DataFrame({
            "y_true": self.y_true,
            "y_pred": self.y_pred
        })
        df.write_csv(path)


@dataclass
class PredictResult:
    """Result from nirs4all.predict()."""
    y_pred: np.ndarray
    fold_predictions: Optional[np.ndarray]  # (n_folds, n_samples)
    model: VirtualModel
    context: DatasetContext

    def to_dataframe(self) -> pl.DataFrame:
        """Convert to DataFrame."""
        return pl.DataFrame({"y_pred": self.y_pred})

    def save(self, path: Union[str, Path]) -> None:
        """Save predictions."""
        self.to_dataframe().write_csv(path)


@dataclass
class ExplainResult:
    """Result from nirs4all.explain()."""
    shap_values: np.ndarray
    expected_value: float
    X: np.ndarray
    feature_names: List[str]
    model: VirtualModel

    def summary_plot(self, **kwargs) -> None:
        """Generate SHAP summary plot."""
        import shap
        shap.summary_plot(
            self.shap_values, self.X,
            feature_names=self.feature_names,
            **kwargs
        )

    def waterfall_plot(self, sample_idx: int = 0, **kwargs) -> None:
        """Generate waterfall plot for single sample."""
        import shap
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=self.expected_value,
                data=self.X[sample_idx],
                feature_names=self.feature_names
            ),
            **kwargs
        )

    def force_plot(self, sample_idx: int = 0, **kwargs):
        """Generate force plot for single sample."""
        import shap
        return shap.force_plot(
            self.expected_value,
            self.shap_values[sample_idx],
            self.X[sample_idx],
            feature_names=self.feature_names,
            **kwargs
        )

    def bar_plot(self, **kwargs) -> None:
        """Generate bar plot of mean absolute SHAP values."""
        import shap
        shap.bar_plot(
            shap.Explanation(
                values=self.shap_values,
                data=self.X,
                feature_names=self.feature_names
            ),
            **kwargs
        )

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top n features by mean absolute SHAP value."""
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:n]
        return [
            (self.feature_names[i], mean_abs[i])
            for i in top_indices
        ]
```

---

## sklearn Estimators

### Step Naming Convention

For sklearn compatibility (GridSearchCV, set_params), pipeline steps need consistent names. NIRS4ALL supports two conventions:

1. **Explicit Names** (recommended):
   ```python
   pipeline = [
       {"name": "scaler", "op": MinMaxScaler()},
       {"name": "pls", "op": PLSRegression(n_components=10)}
   ]
   # Use: grid.set_params(pls__n_components=15)
   ```

2. **Positional Names** (auto-generated):
   ```python
   pipeline = [MinMaxScaler(), PLSRegression()]
   # Auto-named: step0, step1
   # Use: grid.set_params(step1__n_components=15)
   ```

3. **Class-Based Names** (fallback):
   ```python
   pipeline = [MinMaxScaler(), PLSRegression()]
   # Named: minmaxscaler, plsregression
   # Use: grid.set_params(plsregression__n_components=15)
   ```

The naming strategy is selected by the `step_naming` parameter.

### NIRSRegressor

```python
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from typing import Literal


StepNaming = Literal["explicit", "positional", "class"]


class NIRSRegressor(BaseEstimator, RegressorMixin):
    """sklearn-compatible regressor wrapping nirs4all pipeline.

    Enables use with:
    - GridSearchCV, RandomizedSearchCV
    - cross_validate, cross_val_predict
    - SHAP explainers
    - sklearn pipelines

    Parameters:
        pipeline: Pipeline steps. Can be:
            - List of operators: [MinMaxScaler(), PLSRegression()]
            - List of named dicts: [{"name": "scaler", "op": MinMaxScaler()}]
            - Callable returning list
        cv: Cross-validation specification (int or sklearn splitter)
        step_naming: How to name steps for set_params:
            - "explicit": Require named dicts (error if missing)
            - "positional": Auto-name as step0, step1, ... (default)
            - "class": Use lowercase class names
        verbose: Verbosity level
        random_state: Random seed
        n_jobs: Number of parallel jobs

    Attributes:
        dag_: Compiled DAG after fit
        result_: Full RunResult after fit
        best_model_: Best VirtualModel after fit
        best_score_: Best validation score
        step_names_: Resolved step names after fit

    Examples:
        >>> # Basic usage with positional naming
        >>> reg = NIRSRegressor([MinMaxScaler(), PLSRegression()])
        >>> reg.fit(X_train, y_train)
        >>> reg.set_params(step1__n_components=15)  # Positional name

        >>> # With explicit naming (recommended for GridSearchCV)
        >>> reg = NIRSRegressor([
        ...     {"name": "scaler", "op": MinMaxScaler()},
        ...     {"name": "pls", "op": PLSRegression()}
        ... ], step_naming="explicit")
        >>>
        >>> param_grid = {"pls__n_components": [5, 10, 15, 20]}
        >>> grid = GridSearchCV(reg, param_grid, cv=5)
        >>> grid.fit(X, y)

        >>> # With SHAP
        >>> import shap
        >>> explainer = shap.Explainer(reg.predict, X_train)
        >>> shap_values = explainer(X_test)
    """

    def __init__(
        self,
        pipeline: Union[List[Any], Callable],
        cv: CVSpec = 5,
        step_naming: StepNaming = "positional",
        verbose: int = 0,
        random_state: Optional[int] = None,
        n_jobs: int = 1
    ):
        self.pipeline = pipeline
        self.cv = cv
        self.step_naming = step_naming
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, **fit_params):
        """Fit the pipeline to data.

        Args:
            X: Features array
            y: Target array
            **fit_params: Additional fit parameters

        Returns:
            self
        """
        # Get pipeline (may be callable)
        pipeline = self.pipeline() if callable(self.pipeline) else self.pipeline

        # Resolve step names for sklearn compatibility
        self.step_names_, self._step_map = self._resolve_step_names(pipeline)

        # Add CV if not in pipeline
        if not self._has_splitter(pipeline):
            pipeline = self._add_cv(pipeline)

        # Run training
        self.result_ = run(
            pipeline=pipeline,
            data=(X, y),
            cv=self.cv,
            verbose=self.verbose,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            save_artifacts=False,
            **fit_params
        )

        self.dag_ = self.result_.dag
        self.best_model_ = self.result_.best_model
        self.best_score_ = self.result_.best_score

        return self

    def _resolve_step_names(
        self,
        pipeline: List
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Resolve step names based on naming strategy.

        Returns:
            Tuple of (names list, name->step mapping)
        """
        names = []
        step_map = {}

        for i, step in enumerate(pipeline):
            if isinstance(step, dict) and "name" in step:
                # Explicit name
                name = step["name"]
                op = step.get("op") or step.get("operator")
            elif self.step_naming == "explicit":
                raise ValueError(
                    f"Step {i} missing 'name' key. With step_naming='explicit', "
                    f"all steps must have explicit names: {{'name': 'x', 'op': Op()}}"
                )
            elif self.step_naming == "positional":
                name = f"step{i}"
                op = step
            else:  # class
                name = step.__class__.__name__.lower()
                op = step

            names.append(name)
            step_map[name] = op

        return names, step_map

    def predict(self, X):
        """Generate predictions.

        Args:
            X: Features array

        Returns:
            Predictions array
        """
        check_is_fitted(self, ["best_model_"])
        return self.best_model_.predict(X)

    def score(self, X, y, sample_weight=None):
        """Return R² score.

        Args:
            X: Features array
            y: True targets
            sample_weight: Sample weights

        Returns:
            R² score
        """
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def get_params(self, deep=True):
        """Get estimator parameters.

        When deep=True, returns nested parameters using step names.
        """
        params = {
            "pipeline": self.pipeline,
            "cv": self.cv,
            "step_naming": self.step_naming,
            "verbose": self.verbose,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs
        }

        if deep and hasattr(self, "_step_map"):
            for name, step in self._step_map.items():
                if hasattr(step, "get_params"):
                    for key, value in step.get_params(deep=True).items():
                        params[f"{name}__{key}"] = value

        return params

    def set_params(self, **params):
        """Set estimator parameters.

        Supports nested parameters using step names:
            reg.set_params(pls__n_components=15)
        """
        nested_params = {}

        for key, value in params.items():
            if "__" in key:
                # Nested parameter (e.g., pls__n_components)
                step_name, param_name = key.split("__", 1)
                if step_name not in nested_params:
                    nested_params[step_name] = {}
                nested_params[step_name][param_name] = value
            else:
                setattr(self, key, value)

        # Apply nested parameters
        if hasattr(self, "_step_map"):
            for step_name, step_params in nested_params.items():
                if step_name in self._step_map:
                    step = self._step_map[step_name]
                    if hasattr(step, "set_params"):
                        step.set_params(**step_params)
                else:
                    raise ValueError(
                        f"Unknown step '{step_name}'. "
                        f"Available: {list(self._step_map.keys())}"
                    )

        return self

    def _set_nested_param(self, key: str, value: Any):
        """Set parameter in nested pipeline step."""
        parts = key.split("__")
        step_name = parts[0]
        param_name = "__".join(parts[1:])

        # Find step in pipeline
        pipeline = self.pipeline() if callable(self.pipeline) else self.pipeline
        for i, step in enumerate(pipeline):
            if self._get_step_name(step) == step_name:
                if hasattr(step, "set_params"):
                    step.set_params(**{param_name: value})
                break

    def _has_splitter(self, pipeline: List) -> bool:
        """Check if pipeline has a splitter step."""
        from sklearn.model_selection import BaseCrossValidator
        for step in pipeline:
            if isinstance(step, BaseCrossValidator):
                return True
            if isinstance(step, dict) and "splitter" in step:
                return True
        return False

    def _add_cv(self, pipeline: List) -> List:
        """Add cross-validation to pipeline."""
        from sklearn.model_selection import KFold

        cv = self.cv
        if isinstance(cv, int):
            cv = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        # Insert before first model step
        new_pipeline = []
        cv_inserted = False
        for step in pipeline:
            if not cv_inserted and self._is_model_step(step):
                new_pipeline.append(cv)
                cv_inserted = True
            new_pipeline.append(step)

        if not cv_inserted:
            new_pipeline.insert(0, cv)

        return new_pipeline


class NIRSClassifier(NIRSRegressor):
    """sklearn-compatible classifier wrapping nirs4all pipeline.

    Same interface as NIRSRegressor, but with classification-specific
    methods (predict_proba, score uses accuracy).
    """

    def predict_proba(self, X):
        """Predict class probabilities.

        Args:
            X: Features array

        Returns:
            Probability array (n_samples, n_classes)
        """
        check_is_fitted(self, ["best_model_"])
        if hasattr(self.best_model_, "predict_proba"):
            return self.best_model_.predict_proba(X)
        else:
            raise AttributeError("Model does not support predict_proba")

    def score(self, X, y, sample_weight=None):
        """Return accuracy score."""
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)
```

### NIRSSearchCV

```python
class NIRSSearchCV(BaseEstimator):
    """Grid search over nirs4all pipeline configurations.

    Similar to sklearn's GridSearchCV but designed for nirs4all
    pipelines with generator syntax.

    Parameters:
        pipeline: Base pipeline with generators
        param_grid: Additional parameters to search (optional)
        cv: Cross-validation specification
        scoring: Scoring metric
        refit: Whether to refit best model
        verbose: Verbosity level
        n_jobs: Number of parallel jobs

    Attributes:
        cv_results_: DataFrame with all results
        best_estimator_: Best fitted estimator
        best_params_: Parameters of best estimator
        best_score_: Best validation score

    Examples:
        >>> # Using generator syntax
        >>> search = NIRSSearchCV(
        ...     pipeline=[
        ...         {"_or_": [SNV(), MSC()]},
        ...         {"model": PLSRegression(),
        ...          "n_components": {"_range_": [1, 20, 1]}}
        ...     ]
        ... )
        >>> search.fit(X, y)
        >>> print(search.best_params_)

        >>> # With param_grid
        >>> search = NIRSSearchCV(
        ...     pipeline=[MinMaxScaler(), PLSRegression()],
        ...     param_grid={"pls__n_components": [5, 10, 15]}
        ... )
    """

    def __init__(
        self,
        pipeline: List[Any],
        param_grid: Optional[Dict] = None,
        cv: CVSpec = 5,
        scoring: str = "neg_mean_squared_error",
        refit: bool = True,
        verbose: int = 0,
        n_jobs: int = 1,
        random_state: Optional[int] = None
    ):
        self.pipeline = pipeline
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y, **fit_params):
        """Fit search to data.

        Expands generators and param_grid, evaluates all combinations,
        and optionally refits the best model.
        """
        # Combine generators and param_grid
        expanded_pipelines = self._expand_configurations()

        # Train all configurations
        results = []
        for i, (pipeline, config) in enumerate(expanded_pipelines):
            if self.verbose:
                print(f"Configuration {i+1}/{len(expanded_pipelines)}")

            result = run(
                pipeline=pipeline,
                data=(X, y),
                cv=self.cv,
                verbose=max(0, self.verbose - 1),
                save_artifacts=False,
                n_jobs=1  # Parallelism at configuration level
            )

            results.append({
                "config_id": i,
                "pipeline": pipeline,
                "config": config,
                "best_val_score": result.best_score,
                "result": result
            })

        # Build results DataFrame
        self.cv_results_ = pl.DataFrame([
            {k: v for k, v in r.items() if k != "result"}
            for r in results
        ])

        # Find best
        best_idx = np.argmin([
            r["best_val_score"] if "neg" in self.scoring else -r["best_val_score"]
            for r in results
        ])

        best_result = results[best_idx]
        self.best_score_ = best_result["best_val_score"]
        self.best_params_ = best_result["config"]

        # Refit best
        if self.refit:
            self.best_estimator_ = NIRSRegressor(
                pipeline=best_result["pipeline"],
                cv=self.cv,
                verbose=self.verbose,
                random_state=self.random_state
            )
            self.best_estimator_.fit(X, y)

        return self

    def predict(self, X):
        """Predict using best estimator."""
        check_is_fitted(self, ["best_estimator_"])
        return self.best_estimator_.predict(X)

    def _expand_configurations(self) -> List[Tuple[List, Dict]]:
        """Expand all pipeline configurations."""
        # Use DAG builder's generator expansion
        builder = DAGBuilder()
        expanded = builder._expand_generators(self.pipeline)

        # If param_grid provided, expand those too
        if self.param_grid:
            from sklearn.model_selection import ParameterGrid
            param_combinations = list(ParameterGrid(self.param_grid))

            configurations = []
            for pipeline in expanded:
                for params in param_combinations:
                    config_pipeline = self._apply_params(pipeline, params)
                    configurations.append((config_pipeline, params))
        else:
            configurations = [(p, {}) for p in expanded]

        return configurations
```

### sklearn Interoperability Edge Cases

The sklearn estimators (`NIRSRegressor`, `NIRSClassifier`) are designed for full scikit-learn ecosystem compatibility, but certain edge cases require special handling.

#### Nested Cross-Validation

When using `NIRSRegressor` inside `cross_val_score` or nested in `GridSearchCV`, the pipeline's internal CV interacts with sklearn's outer CV:

```python
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

# Scenario 1: cross_val_score with internal CV
# The pipeline has its own CV for model selection; sklearn adds another layer
reg = NIRSRegressor([
    MinMaxScaler(),
    KFold(n_splits=3),  # Internal CV for validation
    PLSRegression()
])

# Outer CV evaluates the whole estimator on held-out data
scores = cross_val_score(reg, X, y, cv=5)  # 5 outer folds

# Total fits: 5 outer * 3 inner = 15 model fits (plus final refit per outer fold)
```

**Behavior clarification:**
- **Inner CV** (inside pipeline): Used for model selection/hyperparameter tuning
- **Outer CV** (sklearn): Used for unbiased performance estimation
- The `fit()` method runs the full internal CV, then refits on all provided data
- Outer CV sees only the final refitted model's predictions

#### Automatic Double CV Detection

`NIRSRegressor` automatically detects when it's being used inside an external CV loop and issues a warning:

```python
class NIRSRegressor(BaseEstimator, RegressorMixin):
    """sklearn-compatible regressor with automatic nested CV detection."""

    _cv_nesting_level = 0  # Class-level nesting tracker

    def fit(self, X, y, **fit_params):
        # Detect if we're inside an external CV loop
        NIRSRegressor._cv_nesting_level += 1

        try:
            if NIRSRegressor._cv_nesting_level > 1:
                # We're inside nested CV!
                self._handle_nested_cv_warning()

            # Normal fit logic...
            return self._fit_internal(X, y, **fit_params)

        finally:
            NIRSRegressor._cv_nesting_level -= 1

    def _handle_nested_cv_warning(self):
        """Handle detection of nested CV usage."""
        if self._has_internal_cv():
            warnings.warn(
                "NIRSRegressor with internal CV is being used inside an external "
                "cross-validation loop (e.g., GridSearchCV, cross_val_score). "
                "This causes nested CV which may be computationally expensive.\n\n"
                "Options:\n"
                "  1. Set cv=None to disable internal CV (recommended)\n"
                "  2. Use nirs4all.run() directly without sklearn wrapper\n"
                "  3. Remove the external CV if internal CV is sufficient\n\n"
                f"Current configuration: internal_cv={self.cv}, "
                f"estimated total fits: {self._estimate_total_fits()}",
                UserWarning,
                stacklevel=4
            )

            if self.auto_disable_nested_cv:
                # Automatically disable internal CV to prevent explosion
                self._original_cv = self.cv
                self.cv = None
                warnings.warn(
                    "Internal CV automatically disabled (auto_disable_nested_cv=True). "
                    "Set auto_disable_nested_cv=False to override.",
                    UserWarning,
                    stacklevel=4
                )

    def _has_internal_cv(self) -> bool:
        """Check if pipeline has internal CV."""
        return self.cv is not None and self.cv > 1

    def _estimate_total_fits(self) -> str:
        """Estimate total model fits for user warning."""
        internal = self.cv if self.cv else 1
        generators = self._count_generator_variants()
        # External CV unknown, estimate as 5
        return f"~{5 * internal * generators} (assuming 5-fold external CV)"
```

**Usage with auto-disable:**

```python
# With auto_disable_nested_cv=True (default), internal CV is disabled automatically
reg = NIRSRegressor(
    pipeline=[MinMaxScaler(), KFold(5), {"model": PLSRegression(10)}],
    auto_disable_nested_cv=True  # Default
)

# Using in GridSearchCV triggers warning and auto-disables internal CV
grid = GridSearchCV(reg, param_grid, cv=5)
grid.fit(X, y)  # Warning issued, internal CV disabled

# To keep internal CV (expert mode), set auto_disable_nested_cv=False
reg = NIRSRegressor(
    pipeline=[...],
    auto_disable_nested_cv=False  # Keep nested CV (slow!)
)
```

#### Nested GridSearchCV

Avoid nested hyperparameter search when the pipeline already contains generators:

```python
# PROBLEMATIC: Double search over same parameters
reg = NIRSRegressor([
    MinMaxScaler(),
    {"model": PLSRegression(), "n_components": {"_range_": [1, 15, 1]}}  # Internal search
])

# This adds ANOTHER search layer - computationally wasteful
grid = GridSearchCV(reg, {"step1__n_components": [5, 10, 15]}, cv=5)  # DON'T

# RECOMMENDED: Use ONE of:
# Option A: Use generators only (nirs4all-native)
reg = NIRSRegressor([
    MinMaxScaler(),
    {"model": PLSRegression(), "n_components": {"_range_": [1, 15, 1]}}
])
reg.fit(X, y)

# Option B: Use GridSearchCV only (sklearn-native)
reg = NIRSRegressor([MinMaxScaler(), PLSRegression()], step_naming="positional")
grid = GridSearchCV(reg, {"step1__n_components": range(1, 16)}, cv=5)
grid.fit(X, y)
```

#### Clone Behavior

sklearn's `clone()` function requires proper `get_params`/`set_params` implementation:

```python
from sklearn.base import clone

reg = NIRSRegressor([MinMaxScaler(), PLSRegression(n_components=10)])
reg_clone = clone(reg)  # Works: creates fresh estimator with same parameters

# Important: Fitted state is NOT cloned
reg.fit(X, y)
reg_clone = clone(reg)
assert not hasattr(reg_clone, "best_model_")  # Clone is unfitted
```

#### Pipeline Steps with Mutable State

Operators with mutable default arguments or class-level state can cause issues:

```python
# PROBLEMATIC: Mutable default
class BadTransform:
    def __init__(self, cache={}):  # Shared across instances!
        self.cache = cache

# CORRECT: Use None default
class GoodTransform:
    def __init__(self, cache=None):
        self.cache = cache if cache is not None else {}
```

#### Memory During Parallel CV

When using `n_jobs > 1`, each worker gets a copy of the data:

```python
# Memory usage: ~n_jobs × data_size during fit
reg = NIRSRegressor([...], n_jobs=4)

# For large datasets, consider:
# 1. Reduce n_jobs
# 2. Use memory-mapped arrays (X = np.load('data.npy', mmap_mode='r'))
# 3. Reduce internal CV folds
```

#### Compatibility Matrix

| sklearn Function | Fully Supported | Notes |
|-----------------|-----------------|-------|
| `cross_val_score` | ✅ | Uses final refitted model |
| `cross_val_predict` | ✅ | OOF predictions from outer CV |
| `GridSearchCV` | ✅ | Use `step_naming="positional"` or explicit |
| `RandomizedSearchCV` | ✅ | Same as GridSearchCV |
| `Pipeline` | ⚠️ | Use with caution; nesting pipelines adds complexity |
| `FeatureUnion` | ⚠️ | Prefer nirs4all's native branching |
| `clone` | ✅ | Clones unfitted estimator |
| `set_params` | ✅ | Nested parameters supported |
| `pickle` | ✅ | Via VirtualModel serialization |

---

## Error Reporting and Recovery

When pipeline execution fails, nirs4all provides detailed error information for debugging and potential recovery.

### Exception Hierarchy

```python
from nirs4all.exceptions import NIRSError

# Base exceptions
class NIRSError(Exception):
    """Base exception for all nirs4all errors."""
    pass

class PipelineExecutionError(NIRSError):
    """Raised when pipeline execution fails.

    Attributes:
        node_id: ID of the failed node
        node_type: Type of the failed node (transform, model, etc.)
        step_index: Position in original pipeline list
        partial_context: DatasetContext up to failure point
        partial_predictions: Any predictions collected before failure
        traceback: Original exception traceback
    """

    def __init__(
        self,
        message: str,
        node_id: str,
        node_type: str,
        step_index: int,
        original_exception: Exception,
        partial_context: Optional[DatasetContext] = None,
        partial_predictions: Optional[Dict] = None
    ):
        super().__init__(message)
        self.node_id = node_id
        self.node_type = node_type
        self.step_index = step_index
        self.original_exception = original_exception
        self.partial_context = partial_context
        self.partial_predictions = partial_predictions
        self.__cause__ = original_exception


class BranchExecutionError(PipelineExecutionError):
    """Raised when one or more branches fail during fork/join.

    Attributes:
        failed_branches: Dict mapping branch_id to exception
        successful_branches: Dict mapping branch_id to context
    """

    def __init__(
        self,
        failed_branches: Dict[int, Exception],
        successful_branches: Optional[Dict[int, DatasetContext]] = None
    ):
        branch_ids = list(failed_branches.keys())
        message = f"Branches {branch_ids} failed during fork/join execution"
        super().__init__(
            message=message,
            node_id="join",
            node_type="join",
            step_index=-1,
            original_exception=list(failed_branches.values())[0]
        )
        self.failed_branches = failed_branches
        self.successful_branches = successful_branches or {}
```

### Error Handling in run()

```python
from nirs4all import run

try:
    result = run(
        pipeline=[MinMaxScaler(), FaultyTransform(), PLSRegression()],
        data=(X, y)
    )
except PipelineExecutionError as e:
    print(f"Pipeline failed at step {e.step_index} ({e.node_type})")
    print(f"Node ID: {e.node_id}")
    print(f"Original error: {e.original_exception}")

    # Access partial state for debugging
    if e.partial_context is not None:
        print(f"Data shape at failure: {e.partial_context.x.shape}")

    # Re-raise original exception for standard debugging
    raise e.original_exception from e
```

### Branch Failure Handling

```python
try:
    result = run(
        pipeline=[
            {"branch": [
                [SNV(), PLSRegression()],
                [FaultyTransform(), PLSRegression()]  # This fails
            ]},
            {"merge": "predictions"}
        ],
        data=(X, y),
        continue_on_error=False  # Default: fail fast
    )
except BranchExecutionError as e:
    print(f"Failed branches: {list(e.failed_branches.keys())}")

    # Inspect successful branches
    for branch_id, ctx in e.successful_branches.items():
        print(f"Branch {branch_id} completed with shape {ctx.x.shape}")

    # Optionally retry without failed branches
    # ... custom recovery logic
```

### Partial Recovery Mode

For long-running pipelines, enable partial recovery to save intermediate state:

```python
result = run(
    pipeline=[...],
    data=(X, y),
    checkpoint_interval=5,  # Save checkpoint every 5 steps
    on_error="checkpoint"   # Save state on failure
)

# On failure, resume from checkpoint
result = run(
    pipeline=[...],
    data=(X, y),
    resume_from="./workspace/runs/run_123/checkpoints/step_15.pkl"
)
```

### Error Context in Logs

When verbose > 0, errors include additional context:

```
[ERROR] Pipeline execution failed at step 3 (ModelNode)
        Node: model_pls_01
        Operator: PLSRegression(n_components=50)
        Error: ValueError: n_components=50 is greater than max(n_samples, n_features)=30

        Context at failure:
          - X shape: (25, 30)
          - y shape: (25,)
          - Active partition: train
          - Current fold: 2/5

        Pipeline state saved to: ./workspace/runs/run_123/error_state.pkl
```

### Validation Errors (Pre-Execution)

Some errors are caught before execution begins:

```python
from nirs4all.exceptions import PipelineValidationError

try:
    result = run(
        pipeline=[MinMaxScaler()],  # No model!
        data=(X, y)
    )
except PipelineValidationError as e:
    print(f"Validation failed: {e.message}")
    print(f"Suggestions: {e.suggestions}")
    # e.suggestions might be: ["Add a model step using {'model': ...}"]
```

### Debuggability for Lazy Views and DAG Execution

With lazy views, hashed blocks, and complex DAGs, debugging "dimension mismatch" errors can be challenging. The system provides rich context to aid debugging:

#### Dimension Mismatch Errors

```python
class DimensionMismatchError(PipelineExecutionError):
    """Raised when array dimensions don't match between pipeline steps.

    Provides detailed context about where the mismatch occurred and
    what the expected vs actual dimensions were.
    """

    def __init__(
        self,
        message: str,
        expected_shape: Tuple[int, ...],
        actual_shape: Tuple[int, ...],
        node_id: str,
        input_block_ids: List[str],
        output_block_id: Optional[str] = None,
        sample_registry_count: Optional[int] = None
    ):
        super().__init__(message, node_id, ...)
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        self.input_block_ids = input_block_ids
        self.sample_registry_count = sample_registry_count

    def __str__(self):
        return f"""
Dimension Mismatch at node '{self.node_id}'

  Expected shape: {self.expected_shape}
  Actual shape:   {self.actual_shape}

  Input blocks:
    {self._format_input_blocks()}

  Sample registry has {self.sample_registry_count} samples

  Possible causes:
    - Sample augmentation changed row count without updating registry
    - Feature selection reduced columns but next step expects original count
    - Branch merge combined incompatible shapes

  To debug:
    1. Check the pipeline step before '{self.node_id}'
    2. Verify sample counts match between transforms
    3. Use verbose=2 to see intermediate shapes
"""
```

#### Lazy View Resolution Errors

```python
class ViewResolutionError(NIRSError):
    """Raised when a lazy view cannot be materialized.

    Includes full context about what the view was trying to resolve
    and why it failed.
    """

    def __init__(
        self,
        message: str,
        view_spec: ViewSpec,
        missing_blocks: List[str],
        available_blocks: List[str]
    ):
        super().__init__(message)
        self.view_spec = view_spec
        self.missing_blocks = missing_blocks
        self.available_blocks = available_blocks

    def __str__(self):
        return f"""
View Resolution Failed

  Requested view:
    - Block IDs: {self.view_spec.block_ids}
    - Partition: {self.view_spec.partition}
    - Fold: {self.view_spec.fold_id}
    - Sample filter: {self.view_spec.sample_filter}

  Missing blocks: {self.missing_blocks}

  Available blocks in store:
    {self._format_available_blocks()}

  This typically happens when:
    - A block was garbage collected prematurely
    - The block ID was computed incorrectly (lineage hash mismatch)
    - A branch failed and its blocks were not created

  To debug:
    1. Enable checkpoint_interval to save intermediate state
    2. Use store.list_blocks() to see what's available
    3. Check the lineage of requested blocks with store.get_lineage()
"""
```

#### DAG Path Tracing

When an error occurs deep in a branched pipeline, the system traces the execution path:

```python
# Error output includes execution path
"""
[ERROR] Pipeline execution failed

  Execution path to failure:
    → SOURCE (data_loader)
    → TRANSFORM (snv_transform)
    → FORK (branch_point) → Branch 1
        → TRANSFORM (first_derivative)
        → MODEL (pls_regression)  ← FAILED HERE

  Branch context:
    - Branch 1 of 3
    - This branch applies: FirstDerivative → PLSRegression

  Full DAG visualization saved to: ./workspace/runs/run_123/dag_error.html
"""
```

#### Interactive Debugging Mode

```python
# Enable interactive debugging
result = run(
    pipeline=[...],
    data=(X, y),
    debug=True  # Drops into debugger on error
)

# Or use post-mortem debugging
try:
    result = run(pipeline, data)
except PipelineExecutionError as e:
    # Inspect the partial state
    print(e.partial_context.block_store.list_blocks())
    print(e.partial_context.sample_registry.get_metadata())

    # Get intermediate data at failure point
    X_at_failure = e.partial_context.x
    print(f"Shape at failure: {X_at_failure.shape}")
```

---

## Session Management

### Session Context

```python
from contextlib import contextmanager


@contextmanager
def session(
    workspace: Optional[str] = None,
    verbose: int = 1,
    save_artifacts: bool = True,
    **defaults
):
    """Create execution session for resource reuse.

    A session maintains shared state across multiple run/predict calls:
    - Shared workspace directory
    - Consistent logging configuration
    - Cached transformers (optional)

    Args:
        workspace: Shared workspace directory
        verbose: Default verbosity level
        save_artifacts: Default artifact saving
        **defaults: Other default options

    Yields:
        Session object with run/predict/explain methods

    Examples:
        >>> with nirs4all.session(workspace="./experiment1") as s:
        ...     r1 = s.run(pipeline1, data1)
        ...     r2 = s.run(pipeline2, data2)
        ...     comparison = s.compare([r1, r2])
    """
    sess = Session(
        workspace=workspace,
        verbose=verbose,
        save_artifacts=save_artifacts,
        **defaults
    )

    try:
        yield sess
    finally:
        sess.close()


class Session:
    """Execution session for resource reuse."""

    def __init__(
        self,
        workspace: Optional[str] = None,
        verbose: int = 1,
        save_artifacts: bool = True,
        **defaults
    ):
        self.workspace = Path(workspace or "./workspace")
        self.verbose = verbose
        self.save_artifacts = save_artifacts
        self.defaults = defaults

        self._results: List[RunResult] = []
        self._artifact_manager = ArtifactManager(self.workspace / "artifacts")

    def run(self, pipeline, data, **kwargs) -> RunResult:
        """Run pipeline with session defaults."""
        merged = {
            "workspace": str(self.workspace),
            "verbose": self.verbose,
            "save_artifacts": self.save_artifacts,
            **self.defaults,
            **kwargs
        }
        result = run(pipeline, data, **merged)
        self._results.append(result)
        return result

    def predict(self, model, data, **kwargs) -> PredictResult:
        """Predict with session defaults."""
        merged = {"verbose": self.verbose, **self.defaults, **kwargs}
        return predict(model, data, **merged)

    def explain(self, model, data, **kwargs) -> ExplainResult:
        """Explain with session defaults."""
        merged = {"verbose": self.verbose, **self.defaults, **kwargs}
        return explain(model, data, **merged)

    def compare(
        self,
        results: Optional[List[RunResult]] = None,
        metrics: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """Compare results from multiple runs."""
        results = results or self._results

        comparisons = []
        for i, result in enumerate(results):
            best = result.best
            comparisons.append({
                "run_id": i,
                "model": best.model_name,
                "preprocessing": best.preprocessings,
                "val_score": best.val_score,
                "test_score": best.test_score
            })

        return pl.DataFrame(comparisons)

    def close(self):
        """Clean up session resources."""
        pass  # Future: close connections, flush caches
```

---

## Explainer Abstraction

Per the critical review, ExplainResult is decoupled from SHAP to support multiple explainer backends.

### Explainer Protocol

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class Explainer(Protocol):
    """Protocol for model explanation backends.

    Implementations:
    - SHAPExplainer (default)
    - LIMEExplainer
    - IntegratedGradientsExplainer (for neural networks)
    - PermutationExplainer
    """

    name: str

    def explain(
        self,
        model: VirtualModel,
        X: np.ndarray,
        **kwargs
    ) -> "ExplainResult":
        """Generate explanations for predictions.

        Args:
            model: The model to explain
            X: Input data
            **kwargs: Backend-specific options

        Returns:
            ExplainResult with feature attributions
        """
        ...

    def supports(self, model: VirtualModel) -> bool:
        """Check if this explainer supports the model type."""
        ...


class SHAPExplainer:
    """SHAP-based explainer (default implementation)."""

    name = "shap"

    def __init__(self, method: str = "auto"):
        self.method = method

    def explain(
        self,
        model: VirtualModel,
        X: np.ndarray,
        n_samples: int = 100,
        **kwargs
    ) -> "ExplainResult":
        import shap

        explainer_cls = self._select_explainer(model)

        if explainer_cls == shap.KernelExplainer:
            background = shap.sample(X, n_samples)
            explainer = explainer_cls(model.predict, background)
        else:
            explainer = explainer_cls(model.primary_model)

        shap_values = explainer.shap_values(X)

        return ExplainResult(
            values=shap_values,
            expected_value=explainer.expected_value,
            X=X,
            method="shap",
            explainer_name=explainer_cls.__name__
        )

    def _select_explainer(self, model: VirtualModel):
        import shap

        if self.method != "auto":
            return getattr(shap, f"{self.method.title()}Explainer")

        # Auto-detect
        primary = model.primary_model

        if hasattr(primary, "tree_"):
            return shap.TreeExplainer
        elif hasattr(primary, "coef_"):
            return shap.LinearExplainer
        else:
            return shap.KernelExplainer

    def supports(self, model: VirtualModel) -> bool:
        return True  # SHAP supports all models


class LIMEExplainer:
    """LIME-based explainer."""

    name = "lime"

    def explain(
        self,
        model: VirtualModel,
        X: np.ndarray,
        n_features: int = 10,
        **kwargs
    ) -> "ExplainResult":
        import lime.lime_tabular

        explainer = lime.lime_tabular.LimeTabularExplainer(
            X, mode="regression", **kwargs
        )

        # Explain each sample
        all_values = []
        for i in range(len(X)):
            exp = explainer.explain_instance(X[i], model.predict)
            values = np.zeros(X.shape[1])
            for feat, weight in exp.as_list():
                # Parse feature name to get index
                idx = int(feat.split()[0])
                values[idx] = weight
            all_values.append(values)

        return ExplainResult(
            values=np.array(all_values),
            expected_value=np.mean(model.predict(X)),
            X=X,
            method="lime",
            explainer_name="LimeTabularExplainer"
        )

    def supports(self, model: VirtualModel) -> bool:
        return hasattr(model, "predict")
```

### ExplainResult (Backend-Agnostic)

```python
@dataclass
class ExplainResult:
    """Result from explanation, independent of backend.

    This class stores raw attribution values. Plotting methods
    use pluggable backends selected based on available libraries.
    """
    values: np.ndarray          # Feature attributions (n_samples, n_features)
    expected_value: float       # Base value / expected output
    X: np.ndarray               # Input data
    method: str                 # "shap", "lime", "integrated_gradients"
    explainer_name: str         # Specific explainer class name
    feature_names: Optional[List[str]] = None

    # ===== Backend-Agnostic Analysis =====

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top n features by mean absolute attribution."""
        mean_abs = np.abs(self.values).mean(axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:n]
        names = self.feature_names or [f"feature_{i}" for i in range(len(mean_abs))]
        return [(names[i], mean_abs[i]) for i in top_indices]

    def feature_importance(self) -> pl.DataFrame:
        """Get feature importance as DataFrame."""
        mean_abs = np.abs(self.values).mean(axis=0)
        names = self.feature_names or [f"feature_{i}" for i in range(len(mean_abs))]
        return pl.DataFrame({
            "feature": names,
            "importance": mean_abs
        }).sort("importance", descending=True)

    # ===== Pluggable Plotting =====

    def summary_plot(self, backend: str = "auto", **kwargs) -> None:
        """Generate summary plot using available backend.

        Args:
            backend: "shap", "matplotlib", or "auto" (detect best)
            **kwargs: Backend-specific options
        """
        plotter = self._get_plotter(backend)
        plotter.summary_plot(self, **kwargs)

    def waterfall_plot(
        self,
        sample_idx: int = 0,
        backend: str = "auto",
        **kwargs
    ) -> None:
        """Generate waterfall plot for single sample."""
        plotter = self._get_plotter(backend)
        plotter.waterfall_plot(self, sample_idx, **kwargs)

    def bar_plot(self, backend: str = "auto", **kwargs) -> None:
        """Generate bar plot of feature importance."""
        plotter = self._get_plotter(backend)
        plotter.bar_plot(self, **kwargs)

    def _get_plotter(self, backend: str) -> "ExplainPlotter":
        """Get plotting backend."""
        if backend == "auto":
            try:
                import shap
                return SHAPPlotter()
            except ImportError:
                return MatplotlibPlotter()
        elif backend == "shap":
            return SHAPPlotter()
        elif backend == "matplotlib":
            return MatplotlibPlotter()
        else:
            raise ValueError(f"Unknown backend: {backend}")


class SHAPPlotter:
    """SHAP-based plotting backend."""

    def summary_plot(self, result: ExplainResult, **kwargs):
        import shap
        shap.summary_plot(
            result.values, result.X,
            feature_names=result.feature_names,
            **kwargs
        )

    def waterfall_plot(self, result: ExplainResult, sample_idx: int, **kwargs):
        import shap
        shap.waterfall_plot(
            shap.Explanation(
                values=result.values[sample_idx],
                base_values=result.expected_value,
                data=result.X[sample_idx],
                feature_names=result.feature_names
            ),
            **kwargs
        )

    def bar_plot(self, result: ExplainResult, **kwargs):
        import shap
        shap.bar_plot(
            shap.Explanation(
                values=result.values,
                data=result.X,
                feature_names=result.feature_names
            ),
            **kwargs
        )


class MatplotlibPlotter:
    """Pure matplotlib plotting backend (no SHAP dependency)."""

    def summary_plot(self, result: ExplainResult, **kwargs):
        import matplotlib.pyplot as plt

        mean_abs = np.abs(result.values).mean(axis=0)
        names = result.feature_names or [f"f{i}" for i in range(len(mean_abs))]

        sorted_idx = np.argsort(mean_abs)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(mean_abs)), mean_abs[sorted_idx])
        ax.set_yticks(range(len(mean_abs)))
        ax.set_yticklabels([names[i] for i in sorted_idx])
        ax.set_xlabel("Mean |Attribution|")
        ax.set_title("Feature Importance")
        plt.tight_layout()
        plt.show()

    def waterfall_plot(self, result: ExplainResult, sample_idx: int, **kwargs):
        import matplotlib.pyplot as plt

        values = result.values[sample_idx]
        names = result.feature_names or [f"f{i}" for i in range(len(values))]

        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(values))[::-1][:10]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if v < 0 else 'blue' for v in values[sorted_idx]]
        ax.barh(range(len(sorted_idx)), values[sorted_idx], color=colors)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([names[i] for i in sorted_idx])
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel("Attribution")
        ax.set_title(f"Sample {sample_idx} Explanation")
        plt.tight_layout()
        plt.show()

    def bar_plot(self, result: ExplainResult, **kwargs):
        self.summary_plot(result, **kwargs)
```

### Registering Custom Explainers

```python
class ExplainerRegistry:
    """Registry for explanation backends."""

    _explainers: Dict[str, Type[Explainer]] = {
        "shap": SHAPExplainer,
        "lime": LIMEExplainer,
    }

    @classmethod
    def register(cls, name: str, explainer_cls: Type[Explainer]):
        """Register a custom explainer."""
        cls._explainers[name] = explainer_cls

    @classmethod
    def get(cls, name: str, **kwargs) -> Explainer:
        """Get explainer by name."""
        if name not in cls._explainers:
            raise ValueError(f"Unknown explainer: {name}. Available: {list(cls._explainers)}")
        return cls._explainers[name](**kwargs)

    @classmethod
    def auto_select(cls, model: VirtualModel) -> Explainer:
        """Auto-select best explainer for model."""
        for name, explainer_cls in cls._explainers.items():
            explainer = explainer_cls()
            if explainer.supports(model):
                return explainer
        return SHAPExplainer()  # Fallback
```

---

## Explainability Limitations for Complex Pipelines

Multi-stage pipelines with branching, stacking, or ensembles present unique challenges for model explanations. This section documents known limitations and recommended strategies.

### Challenge: VirtualModel Opacity

When a pipeline includes multiple models (stacking, voting, branching with different models), the `VirtualModel` abstraction wraps them into a single predict interface. SHAP/LIME explanations operate on this composite model:

```
                   ┌─────────────────────────────────────┐
                   │         VirtualModel                │
                   │  ┌─────────┐    ┌─────────┐        │
Input Features ───►│  │ Model A │ +  │ Model B │  ───►  │ ───► Prediction
                   │  └─────────┘    └─────────┘        │
                   └─────────────────────────────────────┘
                              ▲
                              │
                   SHAP sees this as a black box
```

**Limitations:**

1. **Feature attribution aggregation**: SHAP values reflect the composite model's behavior, not individual base models
2. **Loss of per-model interpretability**: Cannot distinguish which base model uses which features
3. **Stacking meta-learner confusion**: For stacked models, SHAP on VirtualModel explains the meta-learner's use of base model predictions, not original features

### Recommended Strategies

#### Strategy 1: Per-Branch Explanations

For pipelines with branches, explain each branch independently before merging:

```python
# Explain each branch's model separately
for branch_id in range(num_branches):
    branch_result = nirs4all.explain(
        run_dir,
        test_data,
        branch_filter=branch_id,  # Explain only this branch
        method="shap"
    )

# Combine attributions with branch weights if needed
combined_shap = weighted_average(branch_shap_values, branch_weights)
```

#### Strategy 2: Explicit Model Access

Use `VirtualModel.get_models()` to access individual models for targeted explanations:

```python
from nirs4all.explain import SHAPExplainer

virtual_model = result.load_model()
individual_models = virtual_model.get_models()

explainer = SHAPExplainer(max_samples=100)
for model_name, model in individual_models.items():
    # Create single-model VirtualModel wrapper
    single_model = VirtualModel({model_name: model}, aggregation="none")
    explanation = explainer.explain(single_model, X_test)
    print(f"{model_name}: {explanation.feature_names[:5]}")
```

#### Strategy 3: Stacking-Aware Explanations

For stacking pipelines, the system provides a two-level explanation mode:

```python
# Level 1: Explain meta-learner (which base models contribute to prediction)
meta_explanation = nirs4all.explain(
    run_dir,
    test_data,
    level="meta"  # Explain meta-learner only
)

# Level 2: Explain each base model (which original features matter)
base_explanations = nirs4all.explain(
    run_dir,
    test_data,
    level="base"  # Explain each base model individually
)

# Chain explanations (experimental): propagate importance through stack
chained = nirs4all.explain(
    run_dir,
    test_data,
    level="chained"  # Multiply meta-importance by base-importance
)
```

### Limitations Documentation

| Pipeline Type | SHAP/LIME Behavior | Recommendation |
|--------------|-------------------|----------------|
| Single model | Full feature attribution | Default usage |
| Parallel branches (same features) | Aggregated attribution | Use `branch_filter` |
| Parallel branches (different features) | Union of feature spaces | Explain separately |
| Stacking (meta-learner) | Explains meta-model inputs (base predictions) | Use `level="base"` |
| VirtualModel ensemble | Black-box composite | Use `get_models()` |
| Mixed GPU/CPU models | May require model-by-model | Separate explanations |

### Known Edge Cases

1. **TreeSHAP incompatibility**: TreeSHAP requires consistent feature order; branched pipelines may reorder features. Fallback to KernelSHAP if TreeSHAP fails.

2. **Memory with large ensembles**: SHAP background samples multiply with ensemble size. Reduce `max_samples` for large ensembles:
   ```python
   nirs4all.explain(run_dir, test_data, max_samples=50)  # Default is 100
   ```

3. **Categorical features in stacking**: If base models produce categorical predictions, SHAP may fail. Use regression-mode base models or one-hot encode outputs.

---

## UX Syntax Flexibility

Preserving v1's flexible syntax is critical for user adoption. All input forms are normalized during parsing.

### Pipeline Step Formats

```python
# All of these are equivalent and valid:

# 1. Class reference (simplest)
MinMaxScaler

# 2. Instance (with parameters)
MinMaxScaler(feature_range=(0, 1))

# 3. Dict with keyword wrapper
{"preprocessing": MinMaxScaler()}

# 4. Dict with explicit class path
{"class": "sklearn.preprocessing.MinMaxScaler"}

# 5. Dict with parameters
{
    "class": "sklearn.preprocessing.MinMaxScaler",
    "params": {"feature_range": [0, 1]}
}

# 6. String class path
"sklearn.preprocessing.MinMaxScaler"

# 7. Named step (for sklearn compatibility)
{"name": "scaler", "op": MinMaxScaler()}
```

### YAML Pipeline Definition

```yaml
# pipeline.yaml - All formats work in YAML too

steps:
  # Class path (string)
  - class: sklearn.preprocessing.MinMaxScaler

  # With parameters
  - class: nirs4all.transforms.SNV
    params:
      copy: true

  # Keyword wrapper
  - keyword: preprocessing
    class: sklearn.preprocessing.StandardScaler

  # Named step
  - name: pls
    class: sklearn.cross_decomposition.PLSRegression
    params:
      n_components: 10

  # Generator syntax
  - _or_:
      - class: nirs4all.transforms.SNV
      - class: nirs4all.transforms.MSC
      - class: nirs4all.transforms.Detrend
```

### Data Input Formats

```python
# All valid data specifications:

# 1. File path (string)
nirs4all.run(pipeline, "path/to/data.csv")

# 2. Path object
nirs4all.run(pipeline, Path("data/spectra.mat"))

# 3. Tuple of arrays
nirs4all.run(pipeline, (X, y))

# 4. Dict with arrays
nirs4all.run(pipeline, {"X": X, "y": y})

# 5. Dict with metadata
nirs4all.run(pipeline, {
    "X": X,
    "y": y,
    "metadata": {"sample_id": ids},
    "wavelengths": [1000, 1001, ..., 2500]
})

# 6. Multiple sources
nirs4all.run(pipeline, {
    "sources": {
        "NIR": X_nir,
        "SWIR": X_swir,
    },
    "y": y
})
```

### Normalization Pipeline

```python
class StepNormalizer:
    """Normalizes all step formats to canonical form."""

    def normalize(self, step: Any) -> ParsedStep:
        """Convert any step format to ParsedStep."""

        # Class reference: MinMaxScaler
        if isinstance(step, type):
            return ParsedStep(
                operator_class=self._get_class_path(step),
                operator_params={},
                keyword=None,
                operator_instance=None  # Instantiate lazily
            )

        # Instance: MinMaxScaler()
        if hasattr(step, "fit") or hasattr(step, "transform"):
            return ParsedStep(
                operator_class=self._get_class_path(type(step)),
                operator_params=self._get_params(step),
                keyword=None,
                operator_instance=step
            )

        # String class path: "sklearn.preprocessing.MinMaxScaler"
        if isinstance(step, str):
            return ParsedStep(
                operator_class=step,
                operator_params={},
                keyword=None,
                operator_instance=None
            )

        # Dict formats
        if isinstance(step, dict):
            return self._normalize_dict(step)

        raise ValueError(f"Cannot normalize step: {step}")

    def _normalize_dict(self, step: Dict) -> ParsedStep:
        """Normalize dict step formats."""

        # Named step: {"name": "x", "op": Op()}
        if "name" in step and ("op" in step or "operator" in step):
            op = step.get("op") or step["operator"]
            normalized = self.normalize(op)
            normalized.name = step["name"]
            return normalized

        # Keyword wrapper: {"preprocessing": Op()}
        keyword_keys = {"preprocessing", "model", "splitter", "y_processing",
                       "feature_augmentation", "sample_augmentation"}
        for kw in keyword_keys:
            if kw in step:
                normalized = self.normalize(step[kw])
                normalized.keyword = kw
                return normalized

        # Explicit class: {"class": "...", "params": {...}}
        if "class" in step:
            return ParsedStep(
                operator_class=step["class"],
                operator_params=step.get("params", {}),
                keyword=step.get("keyword"),
                operator_instance=None
            )

        raise ValueError(f"Cannot normalize dict step: {step}")

    def _get_class_path(self, cls: type) -> str:
        """Get fully qualified class path."""
        return f"{cls.__module__}.{cls.__name__}"

    def _get_params(self, instance: Any) -> Dict:
        """Extract parameters from instance."""
        if hasattr(instance, "get_params"):
            return instance.get_params(deep=True)
        return {}
```

---

## CLI Interface

The CLI provides command-line access to all major nirs4all functionality. Designed for:
- Batch processing in production environments
- Integration with shell scripts and CI/CD pipelines
- Quick experimentation without writing Python code

### Command Structure

```bash
# Core commands
nirs4all run [OPTIONS] PIPELINE DATA        # Train a pipeline
nirs4all predict [OPTIONS] MODEL DATA       # Generate predictions
nirs4all explain [OPTIONS] MODEL DATA       # Explain predictions

# Model management
nirs4all export [OPTIONS] RESULT OUTPUT     # Export trained model
nirs4all info MODEL                         # Display model information
nirs4all compare RESULT1 RESULT2 ...        # Compare multiple runs

# Utilities
nirs4all validate PIPELINE                  # Validate pipeline syntax
nirs4all list-transforms                    # List available transforms
nirs4all list-models                        # List available models
nirs4all test-install                       # Verify installation

# Help
nirs4all --help                             # General help
nirs4all run --help                         # Command-specific help
```

### Detailed Command Reference

#### `run` - Train Pipeline

```bash
nirs4all run [OPTIONS] PIPELINE DATA

Options:
  -n, --name TEXT           Pipeline name for tracking
  -c, --cv INTEGER          Cross-validation folds [default: 5]
  -v, --verbose INTEGER     Verbosity level (0-3) [default: 1]
  -o, --output PATH         Output directory [default: ./workspace]
  -e, --export PATH         Export best model after training
  -j, --n-jobs INTEGER      Number of parallel jobs [default: 1]
  -s, --seed INTEGER        Random seed for reproducibility
  --config FILE             Load options from YAML config
  --no-save                 Don't save artifacts
  --format [table|json|csv] Output format [default: table]

Arguments:
  PIPELINE  Path to pipeline YAML file or Python module
  DATA      Path to data file (CSV, MAT, Excel) or directory

Examples:
  nirs4all run pipeline.yaml data.csv
  nirs4all run pipeline.yaml data.csv --cv 10 --seed 42
  nirs4all run pipeline.yaml ./data_folder --export model.n4a
  nirs4all run --config experiment.yaml
```

#### `predict` - Generate Predictions

```bash
nirs4all predict [OPTIONS] MODEL DATA

Options:
  -o, --output PATH         Output file path [default: predictions.csv]
  -f, --format [csv|json|npz] Output format [default: csv]
  --include-uncertainty     Include prediction intervals
  --batch-size INTEGER      Batch size for large datasets

Arguments:
  MODEL  Path to model bundle (.n4a) or run directory
  DATA   Path to new data for prediction

Examples:
  nirs4all predict model.n4a new_samples.csv
  nirs4all predict ./runs/run_123 new_samples.csv --format json
  nirs4all predict model.n4a large_data.csv --batch-size 1000
```

#### `explain` - Explain Predictions

```bash
nirs4all explain [OPTIONS] MODEL DATA

Options:
  -o, --output PATH         Output file path
  -m, --method [shap|lime]  Explanation method [default: shap]
  --max-samples INTEGER     Max background samples [default: 100]
  --plot                    Generate visualization
  --plot-output PATH        Path for visualization output

Arguments:
  MODEL  Path to model bundle (.n4a)
  DATA   Path to data for explanation

Examples:
  nirs4all explain model.n4a test.csv --output shap_values.csv
  nirs4all explain model.n4a test.csv --plot --plot-output shap.png
```

#### `info` - Model Information

```bash
nirs4all info [OPTIONS] MODEL

Options:
  --format [text|json]      Output format [default: text]
  --show-pipeline           Show full pipeline definition
  --show-metrics            Show training metrics
  --show-lineage            Show data transformation lineage

Arguments:
  MODEL  Path to model bundle (.n4a) or run directory

Output includes:
  - Model type and parameters
  - Training metrics (R², RMSE, etc.)
  - Cross-validation scores
  - Pipeline structure
  - Feature dimensions
  - Training date and version
```

#### `validate` - Validate Pipeline

```bash
nirs4all validate [OPTIONS] PIPELINE

Options:
  --strict                  Fail on warnings
  --show-dag                Print DAG structure

Arguments:
  PIPELINE  Path to pipeline YAML file

Validates:
  - Syntax correctness
  - Operator availability
  - Parameter validity
  - DAG structure (no cycles)
  - Generator expansion limits
```

### Implementation

```python
import click


@click.group()
@click.version_option()
def cli():
    """NIRS4ALL - Near-Infrared Spectroscopy Analysis Library.

    A comprehensive toolkit for NIRS data analysis with ML/DL pipelines.
    See https://nirs4all.readthedocs.io for documentation.
    """
    pass


@cli.command()
@click.argument("pipeline", type=click.Path(exists=True))
@click.argument("data", type=click.Path(exists=True))
@click.option("--name", "-n", default="", help="Pipeline name")
@click.option("--cv", "-c", default=5, help="Cross-validation folds")
@click.option("--verbose", "-v", default=1, help="Verbosity level")
@click.option("--output", "-o", default="./workspace", help="Output directory")
@click.option("--export", "-e", default=None, help="Export best model to path")
@click.option("--n-jobs", "-j", default=1, help="Number of parallel jobs")
@click.option("--seed", "-s", default=None, type=int, help="Random seed")
@click.option("--config", type=click.Path(exists=True), help="Config file")
@click.option("--no-save", is_flag=True, help="Don't save artifacts")
@click.option("--format", type=click.Choice(["table", "json", "csv"]),
              default="table", help="Output format")
def run(pipeline, data, name, cv, verbose, output, export, n_jobs, seed,
        config, no_save, format):
    """Train a pipeline on data.

    PIPELINE: Path to pipeline YAML file
    DATA: Path to data file (CSV, MAT, etc.)
    """
    import nirs4all

    result = nirs4all.run(
        pipeline=pipeline,
        data=data,
        name=name,
        cv=cv,
        verbose=verbose,
        workspace=output,
        n_jobs=n_jobs,
        random_state=seed,
        save_artifacts=not no_save
    )

    _print_result(result, format)

    if export:
        result.export(export)
        click.echo(f"Exported to {export}")


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.argument("data", type=click.Path(exists=True))
@click.option("--output", "-o", default="predictions.csv", help="Output file")
@click.option("--format", "-f", type=click.Choice(["csv", "json", "npz"]),
              default="csv", help="Output format")
@click.option("--include-uncertainty", is_flag=True,
              help="Include prediction intervals")
def predict(model, data, output, format, include_uncertainty):
    """Generate predictions with trained model.

    MODEL: Path to model bundle (.n4a)
    DATA: Path to new data
    """
    import nirs4all

    result = nirs4all.predict(model, data, include_uncertainty=include_uncertainty)
    result.save(output, format=format)

    click.echo(f"Predictions saved to {output}")


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.argument("data", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output file")
@click.option("--method", "-m", type=click.Choice(["shap", "lime"]),
              default="shap", help="Explanation method")
@click.option("--max-samples", default=100, help="Max background samples")
@click.option("--plot", is_flag=True, help="Generate visualization")
@click.option("--plot-output", default=None, help="Path for plot output")
def explain(model, data, output, method, max_samples, plot, plot_output):
    """Explain predictions with SHAP or LIME.

    MODEL: Path to model bundle (.n4a)
    DATA: Path to data for explanation
    """
    import nirs4all

    result = nirs4all.explain(
        model, data,
        method=method,
        max_samples=max_samples
    )

    if output:
        result.save(output)
        click.echo(f"Explanations saved to {output}")

    if plot:
        fig = result.plot()
        if plot_output:
            fig.savefig(plot_output)
            click.echo(f"Plot saved to {plot_output}")
        else:
            fig.show()


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--format", type=click.Choice(["text", "json"]),
              default="text", help="Output format")
@click.option("--show-pipeline", is_flag=True, help="Show pipeline definition")
@click.option("--show-metrics", is_flag=True, help="Show training metrics")
def info(model, format, show_pipeline, show_metrics):
    """Display model information.

    MODEL: Path to model bundle (.n4a) or run directory
    """
    import nirs4all

    bundle = nirs4all.load(model)
    info_dict = bundle.info()

    if format == "json":
        import json
        click.echo(json.dumps(info_dict, indent=2, default=str))
    else:
        _print_model_info(info_dict, show_pipeline, show_metrics)


@cli.command()
@click.argument("pipeline", type=click.Path(exists=True))
@click.option("--strict", is_flag=True, help="Fail on warnings")
@click.option("--show-dag", is_flag=True, help="Print DAG structure")
def validate(pipeline, strict, show_dag):
    """Validate pipeline syntax and structure.

    PIPELINE: Path to pipeline YAML file
    """
    import nirs4all

    try:
        result = nirs4all.validate_pipeline(pipeline, strict=strict)

        if result.is_valid:
            click.echo(click.style("✓ Pipeline is valid", fg="green"))
        else:
            click.echo(click.style("✗ Pipeline is invalid", fg="red"))
            for error in result.errors:
                click.echo(f"  - {error}")

        for warning in result.warnings:
            click.echo(click.style(f"  ⚠ {warning}", fg="yellow"))

        if show_dag:
            click.echo("\nDAG Structure:")
            click.echo(result.dag_repr)

        if not result.is_valid and strict:
            raise SystemExit(1)

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))
        raise SystemExit(1)


@cli.command("test-install")
def test_install():
    """Verify nirs4all installation and dependencies."""
    import nirs4all

    click.echo("Checking nirs4all installation...")

    checks = nirs4all.check_installation()
    all_ok = True

    for component, status in checks.items():
        if status["ok"]:
            click.echo(click.style(f"  ✓ {component}: {status['version']}", fg="green"))
        else:
            click.echo(click.style(f"  ✗ {component}: {status['error']}", fg="red"))
            all_ok = False

    if all_ok:
        click.echo(click.style("\nAll checks passed!", fg="green"))
    else:
        click.echo(click.style("\nSome checks failed.", fg="yellow"))
        raise SystemExit(1)


def _print_result(result, format):
    """Print run result in specified format."""
    if format == "json":
        import json
        click.echo(json.dumps(result.summary(), indent=2, default=str))
    elif format == "csv":
        import pandas as pd
        df = pd.DataFrame(result.all_scores())
        click.echo(df.to_csv(index=False))
    else:  # table
        click.echo(f"\nBest: {result.best.model_name}")
        click.echo(f"  R²: {result.best.val_r2:.4f}")
        click.echo(f"  RMSE: {result.best.val_rmse:.4f}")


def _print_model_info(info_dict, show_pipeline, show_metrics):
    """Print model info in text format."""
    click.echo(f"Model: {info_dict['model_type']}")
    click.echo(f"Created: {info_dict['created_at']}")
    click.echo(f"Version: nirs4all {info_dict['library_version']}")
    click.echo(f"Features: {info_dict['n_features']}")

    if show_metrics and "metrics" in info_dict:
        click.echo("\nMetrics:")
        for name, value in info_dict["metrics"].items():
            click.echo(f"  {name}: {value:.4f}")

    if show_pipeline and "pipeline" in info_dict:
        click.echo("\nPipeline:")
        for i, step in enumerate(info_dict["pipeline"]):
            click.echo(f"  {i+1}. {step}")


if __name__ == "__main__":
    cli()
```

---

## Pipeline Templates and Patterns

This section provides canonical patterns for common NIRS analysis workflows. Use these as starting points for your own pipelines.

### Template 1: Basic Regression

The simplest pipeline for NIRS regression:

```python
from nirs4all import run
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression

pipeline = [
    MinMaxScaler(),                           # Scale to [0, 1]
    KFold(n_splits=5, shuffle=True),          # 5-fold CV
    {"model": PLSRegression(n_components=10)} # PLS regression
]

result = run(pipeline, data="data.csv")
```

### Template 2: Preprocessing Comparison

Compare multiple preprocessing methods:

```python
from nirs4all import run
from nirs4all.transforms import SNV, MSC, Detrend, FirstDerivative

pipeline = [
    # Try 4 preprocessing methods, pick best
    {"_or_": [SNV(), MSC(), Detrend(), FirstDerivative()]},
    MinMaxScaler(),
    KFold(n_splits=5),
    {"model": PLSRegression(n_components=10)}
]

result = run(pipeline, data="data.csv")
print(f"Best preprocessing: {result.best.pipeline[0]}")
```

### Template 3: Hyperparameter Sweep

Grid search over PLS components:

```python
pipeline = [
    SNV(),
    MinMaxScaler(),
    KFold(n_splits=5),
    {
        "model": PLSRegression(),
        "n_components": {"_range_": [1, 30, 1]}  # 1 to 30
    }
]

result = run(pipeline, data="data.csv")
print(f"Optimal n_components: {result.best.model.n_components}")
```

### Template 4: Stacking Ensemble

Two-level stacking with meta-learner:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

pipeline = [
    SNV(),
    MinMaxScaler(),
    KFold(n_splits=5),

    # Level 1: Parallel base models
    {"branch": [
        [{"model": PLSRegression(n_components=10)}],
        [{"model": RandomForestRegressor(n_estimators=100)}],
        [{"model": Ridge(alpha=1.0)}]
    ]},

    # Merge OOF predictions as features for meta-learner
    {"merge": "predictions"},

    # Level 2: Meta-learner
    {"model": Ridge(alpha=0.1)}
]

result = run(pipeline, data="data.csv")
```

### Template 5: Multi-Source Fusion

Combine NIR spectra with auxiliary features:

```python
pipeline = [
    # Source-specific preprocessing
    {"source_branch": {
        "nir": [SNV(), FirstDerivative(), MinMaxScaler()],
        "markers": [VarianceThreshold(), StandardScaler()]
    }},

    # Merge all sources
    {"merge_sources": "concat"},

    # Train on combined features
    KFold(n_splits=5),
    {"model": PLSRegression(n_components=15)}
]

# Data with multiple sources
data = DatasetConfigs([
    {"path": "nir_spectra.csv", "source": "nir"},
    {"path": "biomarkers.csv", "source": "markers"}
])

result = run(pipeline, data)
```

### Template 6: Feature Selection Pipeline

Select optimal wavelengths then model:

```python
from sklearn.feature_selection import SelectKBest, mutual_info_regression

pipeline = [
    SNV(),
    MinMaxScaler(),

    # Feature selection
    SelectKBest(mutual_info_regression, k=50),

    KFold(n_splits=5),
    {"model": PLSRegression(n_components=10)}
]

result = run(pipeline, data="data.csv")
```

### Template 7: Classification with Deep Learning

Neural network for classification:

```python
from nirs4all.models import Nicon  # 1D CNN

pipeline = [
    SNV(),
    MinMaxScaler(),
    KFold(n_splits=5),
    {
        "model": Nicon(
            n_classes=3,
            epochs=50,
            batch_size=32,
            learning_rate=0.001
        )
    }
]

result = run(pipeline, data="data.csv", task="classification")
```

### Template 8: Transfer Learning

Train on one dataset, fine-tune on another:

```python
# Step 1: Train base model
base_result = run(
    pipeline=[SNV(), MinMaxScaler(), KFold(5), {"model": PLSRegression(20)}],
    data="large_dataset.csv"
)
base_result.export("base_model.n4a")

# Step 2: Transfer to new domain
from nirs4all.transfer import TransferPipeline

transfer_pipeline = TransferPipeline(
    base_model="base_model.n4a",
    strategy="fine_tune",     # or "freeze", "adapt"
    fine_tune_layers=-2       # Fine-tune last 2 layers (for NN)
)

result = run(
    pipeline=[transfer_pipeline],
    data="small_target_dataset.csv",
    cv=3
)
```

### Template 9: Outlier Detection + Modeling

Remove outliers before training:

```python
from nirs4all.outliers import MahalanobisOutlier

pipeline = [
    SNV(),
    MinMaxScaler(),

    # Remove outliers (creates branch for exclusion tracking)
    {"outlier_detection": MahalanobisOutlier(threshold=3.0)},

    KFold(n_splits=5),
    {"model": PLSRegression(n_components=10)}
]

result = run(pipeline, data="data.csv")
print(f"Removed {result.n_outliers} outliers")
```

### Template 10: Full Production Pipeline

Complete pipeline with all best practices:

```yaml
# production_pipeline.yaml
name: production_model
version: "2.0"

steps:
  # Validate input data
  - class: nirs4all.validators.DataValidator
    params:
      check_nan: true
      check_range: [0, 1000000]

  # Preprocessing
  - class: nirs4all.transforms.SNV
  - class: sklearn.preprocessing.MinMaxScaler

  # Feature augmentation
  - keyword: feature_augmentation
    operators:
      - class: nirs4all.transforms.FirstDerivative
      - class: nirs4all.transforms.SecondDerivative
    action: extend

  # Cross-validation
  - class: sklearn.model_selection.RepeatedKFold
    params:
      n_splits: 5
      n_repeats: 3
      random_state: 42

  # Model with hyperparameter search
  - keyword: model
    class: sklearn.cross_decomposition.PLSRegression
    search:
      n_components:
        _range_: [1, 20, 1]

options:
  save_artifacts: true
  checkpoint_interval: 10
  random_state: 42
  n_jobs: 4
```

---

## Configuration Files

### Pipeline YAML

```yaml
# pipeline.yaml
name: my_pipeline
version: "2.0"

# Pipeline steps
steps:
  - class: sklearn.preprocessing.MinMaxScaler
    params:
      feature_range: [0, 1]

  - keyword: feature_augmentation
    operators:
      - class: nirs4all.transforms.SNV
      - class: nirs4all.transforms.FirstDerivative
    action: extend

  - class: sklearn.model_selection.ShuffleSplit
    params:
      n_splits: 5
      test_size: 0.2
      random_state: 42

  - keyword: model
    class: sklearn.cross_decomposition.PLSRegression
    params:
      n_components: 10

# Optional: generators
# _or_ syntax expands to multiple pipelines
variants:
  preprocessing:
    _or_:
      - class: nirs4all.transforms.SNV
      - class: nirs4all.transforms.MSC
      - class: nirs4all.transforms.Detrend
```

### Config YAML

```yaml
# config.yaml
pipeline: pipeline.yaml
data: data/train.csv

options:
  name: experiment_001
  cv: 5
  verbose: 1
  workspace: ./results
  random_state: 42
  n_jobs: 4

export:
  path: models/best_model.n4a
  include_data: false

report:
  format: html
  output: reports/experiment_001.html
```

### Loading Configuration

```python
def load_config(path: Union[str, Path]) -> Dict:
    """Load configuration from YAML file."""
    import yaml

    path = Path(path)
    with open(path) as f:
        config = yaml.safe_load(f)

    # Resolve relative paths
    base_dir = path.parent
    if "pipeline" in config and not Path(config["pipeline"]).is_absolute():
        config["pipeline"] = str(base_dir / config["pipeline"])
    if "data" in config and not Path(config["data"]).is_absolute():
        config["data"] = str(base_dir / config["data"])

    return config


def run_from_config(config_path: Union[str, Path]) -> RunResult:
    """Run pipeline from configuration file."""
    config = load_config(config_path)

    result = run(
        pipeline=config["pipeline"],
        data=config["data"],
        **config.get("options", {})
    )

    # Export if configured
    if "export" in config:
        result.export(**config["export"])

    return result
```

---

## Next Document

**Document 5: Implementation Plan** covers:
- Phase breakdown and timeline
- Dependencies and risks
- Testing strategy
- Migration from v1.x
