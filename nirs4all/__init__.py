"""
NIRS4All - A comprehensive package for Near-Infrared Spectroscopy data processing and analysis.

This package provides tools for spectroscopy data handling, preprocessing, model building,
and pipeline management with support for multiple ML backends.

Public API (recommended)::

    nirs4all.run(pipeline, dataset, ...)         - Train a pipeline
    nirs4all.predict(model, data, ...)           - Make predictions
    nirs4all.explain(model, data, ...)           - Generate SHAP explanations
    nirs4all.retrain(source, data, ...)          - Retrain a pipeline
    nirs4all.session(...)                        - Create execution session
    nirs4all.load_session(path)                  - Load saved session
    nirs4all.generate(n_samples, ...)            - Generate synthetic NIRS data

Classes (for advanced usage):
    nirs4all.PipelineRunner    - Direct runner access
    nirs4all.PipelineConfigs   - Pipeline configuration
    nirs4all.DatasetConfigs    - Dataset configuration (from nirs4all.data)

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
    >>> result.export("exports/best_model.n4a")

Synthetic Data Generation:
    >>> # Generate synthetic data for testing
    >>> dataset = nirs4all.generate(n_samples=1000, random_state=42)
    >>>
    >>> # Use convenience functions
    >>> dataset = nirs4all.generate.regression(n_samples=500)
    >>> dataset = nirs4all.generate.classification(n_samples=300, n_classes=3)

See examples/ for more usage examples.
"""
__version__ = "0.7.1"

# Module-level API (primary interface) - Phase 2
from .api import (
    ExplainResult,
    PredictResult,
    RunResult,
    Session,
    explain,
    generate,
    load_session,
    predict,
    retrain,
    run,
    session,
)
from .controllers import CONTROLLER_REGISTRY, register_controller

# Core pipeline components - for advanced usage
from .pipeline import (
    PipelineConfigs,
    PipelineRunner,
    Run,
    RunConfig,
    RunStatus,
    generate_run_id,
)

# Utility functions for backend detection
from .utils import (
    framework,
    # is_torch_available,
    is_gpu_available,
    is_tensorflow_available,
)

# Make commonly used classes available at package level
__all__ = [
    # Module-level API (primary interface)
    "run",
    "predict",
    "explain",
    "retrain",
    "session",
    "load_session",
    "Session",
    "RunResult",
    "PredictResult",
    "ExplainResult",
    # Synthetic data generation
    "generate",

    # Pipeline components (advanced usage)
    "PipelineRunner",
    "PipelineConfigs",
    "Run",
    "RunStatus",
    "RunConfig",
    "generate_run_id",

    # Controller system
    "register_controller",
    "CONTROLLER_REGISTRY",

    # Utilities
    "is_tensorflow_available",
    # "is_torch_available",
    "is_gpu_available",
    "framework"
]
