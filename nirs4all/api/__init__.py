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
from .result import RunResult, PredictResult, ExplainResult

# Session (Phase 3 - full implementation)
from .session import Session, session, load_session

# Module-level functions (Phase 2)
from .run import run
from .predict import predict
from .explain import explain
from .retrain import retrain

# Synthetic data generation
from .generate import generate_namespace as generate

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
]
