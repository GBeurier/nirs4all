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

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from nirs4all.data import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline import PipelineRunner
from nirs4all.pipeline.engine import require_legacy_engine

from .native_result import NativeMethodsRunResult
from .native_training import run_native_methods
from .result import RunResult
from .session import Session

# Type aliases for clarity
SourceSpec: TypeAlias = (
    NativeMethodsRunResult  # Native in-memory Methods result
    | dict[str, Any]  # Prediction dict from previous run
    | str  # Path to bundle (.n4a) or config
    | Path  # Path to bundle or config
)

DataSpec: TypeAlias = (
    str  # Path to data folder
    | Path  # Path to data folder
    | np.ndarray  # X array
    | tuple[np.ndarray, ...]  # (X,) or (X, y)
    | dict[str, Any]  # Dict with X, y keys
    | SpectroDataset  # Direct SpectroDataset instance
    | DatasetConfigs  # Backward compat
)


def retrain(
    source: SourceSpec,
    data: DataSpec,
    *,
    mode: str = "full",
    name: str = "retrain_dataset",
    new_model: Any | None = None,
    epochs: int | None = None,
    session: Session | None = None,
    verbose: int = 1,
    save_artifacts: bool = True,
    **kwargs: Any,
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
            - Dict: ``{"X": X, "y": y}``; the native path additionally
              requires explicit ``sample_ids``.
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
            - engine: ``"native"`` enables the strict in-memory Methods full
              retrain subset. It requires a ``NativeMethodsRunResult`` source,
              a raw ``{"X", "y", "sample_ids"}`` dataset and preserves the
              attested selected PLS variant. Archive sources, transfer and
              finetune remain explicit refusals; all other modes use legacy.

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

    engine = kwargs.pop("engine", None)
    if engine == "native":
        return _retrain_native_methods_full(
            source,
            data,
            mode=mode,
            name=name,
            new_model=new_model,
            epochs=epochs,
            session=session,
            verbose=verbose,
            save_artifacts=save_artifacts,
            extra_kwargs=kwargs,
        )
    require_legacy_engine("retrain", engine)

    # Use session runner if provided, otherwise create new
    runner = session.runner if session is not None else PipelineRunner(verbose=verbose, save_artifacts=save_artifacts)

    # Convert Path to str for compatibility with type hints
    source_arg = str(source) if isinstance(source, Path) else source
    data_arg = str(data) if isinstance(data, Path) else data

    # Call the runner's retrain method
    predictions, per_dataset = runner.retrain(source=source_arg, dataset=data_arg, mode=mode, dataset_name=name, new_model=new_model, epochs=epochs, verbose=verbose, **kwargs)

    return RunResult(
        predictions=predictions,
        per_dataset=per_dataset,
        _runner=runner,
        _owns_runner=session is None,
    )


def _retrain_native_methods_full(
    source: SourceSpec,
    data: DataSpec,
    *,
    mode: str,
    name: str,
    new_model: Any | None,
    epochs: int | None,
    session: Session | None,
    verbose: int,
    save_artifacts: bool,
    extra_kwargs: Mapping[str, Any],
) -> NativeMethodsRunResult:
    """Refit one selected in-memory Methods variant without legacy orchestration.

    This is intentionally the first native retrain capability, not a silent
    reinterpretation of every historical retrain mode. A durable Archive V2
    source, transfer learning and finetuning need their own capability and
    lineage contracts, so they are refused before data reaches the runtime.
    """

    if mode != "full":
        raise NotImplementedError("engine='native' retrain currently supports only mode='full'")
    if not isinstance(source, NativeMethodsRunResult):
        raise TypeError("engine='native' retrain requires an in-memory NativeMethodsRunResult source")
    if not isinstance(data, Mapping):
        raise TypeError("engine='native' retrain requires data={'X', 'y', 'sample_ids'}")
    if new_model is not None or epochs is not None:
        raise NotImplementedError("engine='native' full retrain does not accept new_model or epochs")
    if session is not None:
        raise NotImplementedError("engine='native' retrain is stateless; do not pass a session")
    if not save_artifacts:
        raise ValueError("engine='native' retrain requires save_artifacts=True")
    if verbose not in (0, 1, 2):
        raise ValueError("engine='native' retrain verbose must be 0, 1, or 2")
    if extra_kwargs:
        raise NotImplementedError(f"engine='native' retrain does not accept legacy kwargs: {sorted(extra_kwargs)}")

    return run_native_methods(
        _selected_native_methods_pipeline(source),
        data,
        name=name,
        save_charts=False,
        random_state=None,
    )


def _selected_native_methods_pipeline(source: NativeMethodsRunResult) -> list[Any]:
    """Clone the source recipe and apply its attested selected PLS patch only."""

    original = getattr(source.native_estimator, "pipeline", None)
    if not isinstance(original, list):
        raise ValueError("native retrain source does not retain a portable list pipeline")
    pipeline = copy.deepcopy(original)
    models = [step["model"] for step in pipeline if isinstance(step, Mapping) and set(step) == {"model"}]
    if len(models) != 1:
        raise ValueError("native retrain source does not retain exactly one portable Methods model")

    outcome = getattr(source.native_estimator, "training_outcome_", None)
    document = outcome.to_dict() if hasattr(outcome, "to_dict") else outcome
    if not isinstance(document, Mapping):
        raise ValueError("native retrain source does not retain a structured native outcome")
    patches = document.get("parameter_patches", [])
    if not isinstance(patches, list):
        raise ValueError("native retrain source parameter patches are malformed")
    saw_components_patch = False
    for patch in patches:
        if not isinstance(patch, Mapping):
            raise ValueError("native retrain source parameter patch is malformed")
        if (
            patch.get("schema_version") != 1
            or patch.get("node_id") != "model:compat.0"
            or patch.get("namespace") != "operator"
            or patch.get("path") != ["n_components"]
            or isinstance(patch.get("value"), bool)
            or not isinstance(patch.get("value"), int)
            or patch["value"] < 1
            or saw_components_patch
        ):
            raise ValueError("native retrain source carries an unsupported selected parameter patch")
        models[0].n_components = patch["value"]
        saw_components_patch = True
    return pipeline
