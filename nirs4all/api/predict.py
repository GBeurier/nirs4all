"""
Module-level predict() function for nirs4all.

This module provides a simple interface for making predictions with trained
nirs4all models. It wraps PipelineRunner.predict() with ergonomic defaults.

Two prediction paths are supported:

1. **Store-based** (preferred): ``nirs4all.predict(chain_id="abc", data=X)``
   replays a stored chain directly from the DuckDB workspace.

2. **Model-based** (legacy): ``nirs4all.predict(model="model.n4a", data=X)``
   resolves via PredictionResolver / BundleLoader.

Example:
    >>> import nirs4all
    >>> result = nirs4all.predict(
    ...     model="exports/best_model.n4a",
    ...     data=X_new,
    ...     verbose=1
    ... )
    >>> print(f"Predictions shape: {result.shape}")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nirs4all.data import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline import PipelineRunner

from .result import PredictResult
from .session import Session

# Type aliases for clarity
ModelSpec = (
    dict[str, Any]               # Prediction dict from previous run
    | str                          # Path to bundle (.n4a) or config
    | Path                          # Path to bundle or config
)

DataSpec = (
    str                          # Path to data folder
    | Path                         # Path to data folder
    | np.ndarray                   # X array
    | tuple[np.ndarray, ...]       # (X,) or (X, y)
    | dict[str, Any]               # Dict with X key
    | SpectroDataset               # Direct SpectroDataset instance
    | DatasetConfigs                # Backward compat
)

def predict(
    model: ModelSpec | None = None,
    data: DataSpec | None = None,
    *,
    chain_id: str | None = None,
    workspace_path: str | Path | None = None,
    name: str = "prediction_dataset",
    all_predictions: bool = False,
    session: Session | None = None,
    verbose: int = 0,
    **runner_kwargs: Any,
) -> PredictResult:
    """Make predictions with a trained model on new data.

    This function provides a simple interface for running inference with
    trained nirs4all pipelines.

    Two prediction paths are supported:

    **Store-based** (preferred) -- pass ``chain_id`` together with a
    raw numpy array for ``data``:

    >>> result = nirs4all.predict(chain_id="abc123", data=X_new)

    **Model-based** (legacy) -- pass ``model`` together with ``data``:

    >>> result = nirs4all.predict(model="exports/model.n4a", data=X_new)

    Args:
        model: Trained model specification. Can be:
            - Prediction dict from ``result.best`` or ``result.top()``
            - Path to exported bundle: ``"exports/model.n4a"``
            - Path to pipeline config directory
            Mutually exclusive with ``chain_id``.

        data: Data to predict on. Can be:
            - Path to data folder: ``"new_data/"``
            - Numpy array: ``X_new`` (n_samples, n_features)
            - Tuple: ``(X,)`` or ``(X, y)`` for evaluation
            - Dict: ``{"X": X, "metadata": meta}``
            - SpectroDataset instance

        chain_id: Chain identifier in the workspace DuckDB store.
            When provided, uses the fast store-based replay path.
            Mutually exclusive with ``model``.

        workspace_path: Workspace root directory.  Required when using
            ``chain_id`` outside a session.  Ignored when a ``session``
            is provided (the session's workspace is used instead).

        name: Name for the prediction dataset (for logging).
            Default: "prediction_dataset"

        all_predictions: If True, return predictions from all folds.
            If False (default), return single aggregated prediction.

        session: Optional Session for resource reuse.
            If provided, uses the session's runner.

        verbose: Verbosity level (0=quiet, 1=info, 2=debug).
            Default: 0

        **runner_kwargs: Additional PipelineRunner parameters.
            Common options: plots_visible

    Returns:
        PredictResult containing:
            - y_pred: Predicted values array (n_samples,)
            - metadata: Additional prediction metadata
            - model_name: Name of the model used
            - preprocessing_steps: List of preprocessing steps applied

        Use ``result.to_dataframe()`` for pandas DataFrame output.

    Raises:
        ValueError: If neither ``model`` nor ``chain_id`` is provided,
            or if both are provided.
        FileNotFoundError: If model bundle or data path doesn't exist.

    Examples:
        Predict from a stored chain (preferred):

        >>> import nirs4all
        >>> result = nirs4all.predict(chain_id="abc123", data=X_new)

        Predict from an exported bundle:

        >>> result = nirs4all.predict(
        ...     model="exports/wheat_model.n4a",
        ...     data=X_new
        ... )

        Predict using a result from a previous run:

        >>> train_result = nirs4all.run(pipeline, train_data)
        >>> pred_result = nirs4all.predict(
        ...     model=train_result.best,
        ...     data=X_test
        ... )

    See Also:
        - :func:`nirs4all.run`: Train a pipeline
        - :func:`nirs4all.explain`: Generate SHAP explanations
        - :class:`nirs4all.api.result.PredictResult`: Result class
    """
    # ---- Validate mutually exclusive arguments ----
    if model is not None and chain_id is not None:
        raise ValueError("Provide either 'model' or 'chain_id', not both.")
    if model is None and chain_id is None:
        raise ValueError("Provide either 'model' or 'chain_id'.")
    if data is None:
        raise ValueError("'data' is required.")

    # ---- Store-based path (chain_id) ----
    if chain_id is not None:
        return _predict_from_chain(
            chain_id=chain_id,
            data=data,
            workspace_path=workspace_path,
            session=session,
            verbose=verbose,
            **runner_kwargs,
        )

    # ---- Model-based path (legacy) ----
    return _predict_from_model(
        model=model,
        data=data,
        name=name,
        all_predictions=all_predictions,
        session=session,
        verbose=verbose,
        workspace_path=workspace_path,
        **runner_kwargs,
    )

# -----------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------

def _predict_from_chain(
    chain_id: str,
    data: DataSpec,
    workspace_path: str | Path | None,
    session: Session | None,
    verbose: int,
    **runner_kwargs: Any,
) -> PredictResult:
    """Replay a stored chain on new data via WorkspaceStore."""
    from nirs4all.pipeline.storage.chain_replay import replay_chain
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    # Resolve workspace path
    if session is not None:
        ws_path = session.runner.workspace_path
    elif workspace_path is not None:
        ws_path = Path(workspace_path)
    else:
        ws_path = Path("workspace")

    # Get X from data
    X = _extract_X(data)

    # Open store, replay, and fetch chain metadata in one session
    store = WorkspaceStore(ws_path)
    try:
        y_pred = replay_chain(store, chain_id=chain_id, X=X)
        chain_info = store.get_chain(chain_id)
        model_name = chain_info.get("model_class", "") if chain_info else ""
    finally:
        store.close()

    return PredictResult(
        y_pred=y_pred,
        metadata={"chain_id": chain_id},
        model_name=model_name,
        preprocessing_steps=[],
    )

def _predict_from_model(
    model: ModelSpec,
    data: DataSpec,
    name: str,
    all_predictions: bool,
    session: Session | None,
    verbose: int,
    workspace_path: str | Path | None = None,
    **runner_kwargs: Any,
) -> PredictResult:
    """Predict via the legacy model/resolver path."""
    # Use session runner if provided, otherwise create new
    if session is not None:
        runner = session.runner
    else:
        all_kwargs: dict[str, Any] = {
            "mode": "predict",
            "verbose": verbose,
            **runner_kwargs,
        }
        if workspace_path is not None:
            all_kwargs["workspace_path"] = workspace_path
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
        verbose=verbose,
    )

    # Extract model info for the result
    model_name = ""
    preprocessing_steps: list[str] = []

    if isinstance(model, dict):
        model_name = model.get("model_name", "")
        raw_pp = model.get("preprocessings", [])
        preprocessing_steps = [raw_pp] if isinstance(raw_pp, str) else list(raw_pp)

    # Handle array output
    if isinstance(y_pred, dict):
        first_key = next(iter(y_pred.keys()), None)
        y_array = y_pred[first_key] if first_key and isinstance(y_pred[first_key], np.ndarray) else np.array([])
        metadata: dict[str, Any] = {"all_folds": y_pred}
    else:
        y_array = y_pred if isinstance(y_pred, np.ndarray) else np.asarray(y_pred)
        metadata = {}

    return PredictResult(
        y_pred=y_array,
        metadata=metadata,
        model_name=model_name,
        preprocessing_steps=preprocessing_steps,
    )

def _extract_X(data: DataSpec) -> np.ndarray:
    """Extract feature matrix X from various data formats.

    Args:
        data: Data in any supported format.

    Returns:
        Feature matrix as numpy array.

    Raises:
        TypeError: If data format is not supported for chain replay.
    """
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, tuple):
        return np.asarray(data[0])
    if isinstance(data, dict):
        if "X" in data:
            return np.asarray(data["X"])
        raise TypeError("Dict data must contain an 'X' key for chain replay.")
    if isinstance(data, SpectroDataset):
        return data.x({})
    raise TypeError(
        f"Unsupported data type for chain replay: {type(data).__name__}. "
        "Pass a numpy array, tuple, dict with 'X' key, or SpectroDataset."
    )
