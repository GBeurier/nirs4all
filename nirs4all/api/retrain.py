"""
Module-level retrain() function for nirs4all.

This module provides a simple interface for retraining nirs4all pipelines
on new data. It wraps PipelineRunner.retrain() with ergonomic defaults.

Example:
    >>> import nirs4all
    >>> # Full retrain on new data
    >>> result = nirs4all.retrain(
    ...     source="exports/model.n4a",
    ...     data=new_data,
    ...     mode="full"
    ... )
    >>> print(f"New RMSE: {result.best_rmse:.4f}")
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np

from nirs4all.pipeline import PipelineRunner
from nirs4all.data import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset

from .result import RunResult
from .session import Session


# Type aliases for clarity
SourceSpec = Union[
    Dict[str, Any],               # Prediction dict from previous run
    str,                          # Path to bundle (.n4a) or config
    Path                          # Path to bundle or config
]

DataSpec = Union[
    str,                          # Path to data folder
    Path,                         # Path to data folder
    np.ndarray,                   # X array
    Tuple[np.ndarray, ...],       # (X,) or (X, y)
    Dict[str, Any],               # Dict with X, y keys
    SpectroDataset,               # Direct SpectroDataset instance
    DatasetConfigs                # Backward compat
]


def retrain(
    source: SourceSpec,
    data: DataSpec,
    *,
    mode: str = "full",
    name: str = "retrain_dataset",
    new_model: Optional[Any] = None,
    epochs: Optional[int] = None,
    session: Optional[Session] = None,
    verbose: int = 1,
    save_artifacts: bool = True,
    **kwargs: Any
) -> RunResult:
    """Retrain a pipeline on new data.

    This function enables retraining trained pipelines with various modes,
    allowing for full retraining, transfer learning, or fine-tuning.

    Args:
        source: Pipeline source to retrain from. Can be:
            - Prediction dict from ``result.best`` or ``result.top()``
            - Path to exported bundle: ``"exports/model.n4a"``
            - Path to pipeline config directory

        data: New dataset to train on. Can be:
            - Path to data folder: ``"new_data/"``
            - Numpy arrays: ``(X, y)``
            - Dict: ``{"X": X, "y": y}``
            - SpectroDataset instance

        mode: Retrain mode. Options:
            - "full": Train everything from scratch (same pipeline structure)
            - "transfer": Use existing preprocessing, train new model
            - "finetune": Continue training existing model
            Default: "full"

        name: Name for the retrain dataset (for logging).
            Default: "retrain_dataset"

        new_model: Optional new model for transfer mode.
            Replaces the original model while keeping preprocessing.

        epochs: Optional number of epochs for fine-tuning neural networks.

        session: Optional Session for resource reuse.
            If provided, uses the session's runner.

        verbose: Verbosity level (0=quiet, 1=info, 2=debug).
            Default: 1

        save_artifacts: Whether to save retrained artifacts.
            Default: True

        **kwargs: Additional retraining parameters:
            - learning_rate: Learning rate for fine-tuning
            - freeze_layers: List of layers to freeze during fine-tuning
            - step_modes: Per-step mode overrides (advanced)

    Returns:
        RunResult containing:
            - predictions: Predictions from the retrained pipeline
            - per_dataset: Per-dataset execution details
            - best: Best prediction entry
            - best_score: Best model's primary test score

    Raises:
        ValueError: If mode is invalid or source cannot be resolved.
        FileNotFoundError: If source references files that don't exist.

    Examples:
        Full retrain on new data:

        >>> import nirs4all
        >>>
        >>> # Original training
        >>> original = nirs4all.run(pipeline, train_data)
        >>>
        >>> # Retrain on new data with same pipeline
        >>> retrained = nirs4all.retrain(
        ...     source=original.best,
        ...     data=new_train_data,
        ...     mode="full"
        ... )
        >>> print(f"Original: {original.best_rmse:.4f}")
        >>> print(f"Retrained: {retrained.best_rmse:.4f}")

        Transfer learning with new model:

        >>> from sklearn.ensemble import RandomForestRegressor
        >>>
        >>> result = nirs4all.retrain(
        ...     source="exports/pls_model.n4a",
        ...     data=new_data,
        ...     mode="transfer",
        ...     new_model=RandomForestRegressor(n_estimators=100)
        ... )

        Fine-tune a neural network:

        >>> result = nirs4all.retrain(
        ...     source="exports/nn_model.n4a",
        ...     data=new_data,
        ...     mode="finetune",
        ...     epochs=10,
        ...     learning_rate=0.0001
        ... )

        Retrain from an exported bundle:

        >>> result = nirs4all.retrain(
        ...     source="exports/wheat_model.n4a",
        ...     data="new_wheat_data/",
        ...     mode="full",
        ...     verbose=2
        ... )
        >>> result.export("exports/retrained_model.n4a")

    See Also:
        - :func:`nirs4all.run`: Train a pipeline from scratch
        - :func:`nirs4all.predict`: Make predictions
        - :class:`nirs4all.pipeline.RetrainMode`: Retrain mode enum
    """
    # Validate mode
    valid_modes = {"full", "transfer", "finetune"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

    # Use session runner if provided, otherwise create new
    if session is not None:
        runner = session.runner
    else:
        runner = PipelineRunner(
            verbose=verbose,
            save_artifacts=save_artifacts
        )

    # Convert Path to str for compatibility with type hints
    source_arg = str(source) if isinstance(source, Path) else source
    data_arg = str(data) if isinstance(data, Path) else data

    # Call the runner's retrain method
    predictions, per_dataset = runner.retrain(
        source=source_arg,
        dataset=data_arg,
        mode=mode,
        dataset_name=name,
        new_model=new_model,
        epochs=epochs,
        verbose=verbose,
        **kwargs
    )

    return RunResult(
        predictions=predictions,
        per_dataset=per_dataset,
        _runner=runner
    )
