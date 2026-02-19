"""
NIRS4All API Module - High-level functional interface.

This module provides the primary public API for nirs4all, offering
simple function-based entry points that wrap the underlying PipelineRunner.

Public API:
    run(pipeline, dataset, **kwargs) -> RunResult
        Execute a training pipeline on a dataset.

    predict(model, data, **kwargs) -> PredictResult
        Make predictions with a trained model.

    explain(model, data, **kwargs) -> ExplainResult
        Generate SHAP explanations for model predictions.

    retrain(source, data, **kwargs) -> RunResult
        Retrain a pipeline on new data.

    session(**kwargs) -> Session
        Create an execution session for resource reuse.

    generate(n_samples, **kwargs) -> SpectroDataset | (X, y)
        Generate synthetic NIRS data for testing and research.

Example:
    >>> import nirs4all
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from sklearn.cross_decomposition import PLSRegression
    >>>
    >>> result = nirs4all.run(
    ...     pipeline=[MinMaxScaler(), PLSRegression(10)],
    ...     dataset="sample_data/regression",
    ...     verbose=1
    ... )
    >>> print(f"Best RMSE: {result.best_rmse:.4f}")

For more examples, see the examples/Q40_new_api.py file.
"""

# Result classes (Phase 1)
from .explain import explain

# Synthetic data generation
from .generate import generate_namespace as generate
from .predict import predict
from .result import ExplainResult, LazyModelRefitResult, ModelRefitResult, PredictResult, RunResult
from .retrain import retrain

# Module-level functions (Phase 2)
from .run import run

# Session (Phase 3 - full implementation)
from .session import Session, load_session, session

__all__ = [
    # Module-level API functions
    "run",
    "predict",
    "explain",
    "retrain",
    # Session
    "Session",
    "session",
    "load_session",
    # Synthetic data generation
    "generate",
    # Result classes
    "RunResult",
    "PredictResult",
    "ExplainResult",
    "ModelRefitResult",
    "LazyModelRefitResult",
]
