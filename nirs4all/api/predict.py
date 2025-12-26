"""
Module-level predict() function for nirs4all.

This module provides a simple interface for making predictions with trained
nirs4all models. It wraps PipelineRunner.predict() with ergonomic defaults.

Example:
    >>> import nirs4all
    >>> result = nirs4all.predict(
    ...     model="exports/best_model.n4a",
    ...     data=X_new,
    ...     verbose=1
    ... )
    >>> print(f"Predictions shape: {result.shape}")
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np

from nirs4all.pipeline import PipelineRunner
from nirs4all.data import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset

from .result import PredictResult
from .session import Session


# Type aliases for clarity
ModelSpec = Union[
    Dict[str, Any],               # Prediction dict from previous run
    str,                          # Path to bundle (.n4a) or config
    Path                          # Path to bundle or config
]

DataSpec = Union[
    str,                          # Path to data folder
    Path,                         # Path to data folder
    np.ndarray,                   # X array
    Tuple[np.ndarray, ...],       # (X,) or (X, y)
    Dict[str, Any],               # Dict with X key
    SpectroDataset,               # Direct SpectroDataset instance
    DatasetConfigs                # Backward compat
]


def predict(
    model: ModelSpec,
    data: DataSpec,
    *,
    name: str = "prediction_dataset",
    all_predictions: bool = False,
    session: Optional[Session] = None,
    verbose: int = 0,
    **runner_kwargs: Any
) -> PredictResult:
    """Make predictions with a trained model on new data.

    This function provides a simple interface for running inference with
    trained nirs4all pipelines. The model can be specified as a prediction
    dict from a previous run, or as a path to an exported bundle.

    Args:
        model: Trained model specification. Can be:
            - Prediction dict from ``result.best`` or ``result.top()``
            - Path to exported bundle: ``"exports/model.n4a"``
            - Path to pipeline config directory

        data: Data to predict on. Can be:
            - Path to data folder: ``"new_data/"``
            - Numpy array: ``X_new`` (n_samples, n_features)
            - Tuple: ``(X,)`` or ``(X, y)`` for evaluation
            - Dict: ``{"X": X, "metadata": meta}``
            - SpectroDataset instance

        name: Name for the prediction dataset (for logging).
            Default: "prediction_dataset"

        all_predictions: If True, return predictions from all folds.
            If False (default), return single aggregated prediction.

        session: Optional Session for resource reuse.
            If provided, uses the session's runner.

        verbose: Verbosity level (0=quiet, 1=info, 2=debug).
            Default: 0

        **runner_kwargs: Additional PipelineRunner parameters.
            Common options: workspace_path, plots_visible

    Returns:
        PredictResult containing:
            - y_pred: Predicted values array (n_samples,)
            - metadata: Additional prediction metadata
            - model_name: Name of the model used
            - preprocessing_steps: List of preprocessing steps applied

        Use ``result.to_dataframe()`` for pandas DataFrame output.

    Raises:
        ValueError: If model specification is invalid.
        FileNotFoundError: If model bundle or data path doesn't exist.

    Examples:
        Predict from an exported bundle:

        >>> import nirs4all
        >>>
        >>> result = nirs4all.predict(
        ...     model="exports/wheat_model.n4a",
        ...     data=X_new
        ... )
        >>> print(f"Predictions: {result.values[:5]}")

        Predict using a result from a previous run:

        >>> # Training
        >>> train_result = nirs4all.run(pipeline, train_data)
        >>>
        >>> # Prediction with best model
        >>> pred_result = nirs4all.predict(
        ...     model=train_result.best,
        ...     data=X_test
        ... )

        Get all fold predictions:

        >>> result = nirs4all.predict(
        ...     model="exports/model.n4a",
        ...     data=X_new,
        ...     all_predictions=True
        ... )
        >>> print(f"Shape: {result.shape}")

        Convert to DataFrame:

        >>> result = nirs4all.predict(model, data)
        >>> df = result.to_dataframe()
        >>> df.to_csv("predictions.csv")

    See Also:
        - :func:`nirs4all.run`: Train a pipeline
        - :func:`nirs4all.explain`: Generate SHAP explanations
        - :class:`nirs4all.api.result.PredictResult`: Result class
    """
    # Use session runner if provided, otherwise create new
    if session is not None:
        runner = session.runner
    else:
        all_kwargs = {
            "mode": "predict",
            "verbose": verbose,
            **runner_kwargs
        }
        runner = PipelineRunner(**all_kwargs)

    # Convert Path to str for compatibility with type hints
    model_arg = str(model) if isinstance(model, Path) else model
    data_arg = str(data) if isinstance(data, Path) else data

    # Call the runner's predict method
    y_pred, predictions = runner.predict(
        prediction_obj=model_arg,
        dataset=data_arg,
        dataset_name=name,
        all_predictions=all_predictions,
        verbose=verbose
    )

    # Extract model info for the result
    model_name = ""
    preprocessing_steps = []

    if isinstance(model, dict):
        model_name = model.get("model_name", "")
        preprocessing_steps = model.get("preprocessings", [])
        if isinstance(preprocessing_steps, str):
            preprocessing_steps = [preprocessing_steps]

    # Handle array output
    if isinstance(y_pred, dict):
        # all_predictions=True returns dict
        # Extract first fold's predictions as primary
        first_key = next(iter(y_pred.keys()), None)
        if first_key and isinstance(y_pred[first_key], np.ndarray):
            y_array = y_pred[first_key]
        else:
            y_array = np.array([])
        metadata = {"all_folds": y_pred}
    else:
        y_array = y_pred if isinstance(y_pred, np.ndarray) else np.asarray(y_pred)
        metadata = {}

    return PredictResult(
        y_pred=y_array,
        metadata=metadata,
        model_name=model_name,
        preprocessing_steps=preprocessing_steps
    )
