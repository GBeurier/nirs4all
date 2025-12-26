"""
Module-level run() function for nirs4all.

This module provides the primary entry point for training ML pipelines on NIRS data.
It wraps PipelineRunner.run() with a simpler, more ergonomic interface.

Example:
    >>> import nirs4all
    >>> result = nirs4all.run(
    ...     pipeline=[MinMaxScaler(), PLSRegression(10)],
    ...     dataset="sample_data/regression",
    ...     verbose=1
    ... )
    >>> print(f"Best RMSE: {result.best_rmse:.4f}")
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np

from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset

from .result import RunResult
from .session import Session


# Type aliases for flexibility (documentation, not runtime enforcement)
PipelineSpec = Union[
    List[Any],                    # List of steps (most common)
    Dict[str, Any],               # Dict configuration
    str,                          # Path to YAML/JSON config
    Path,                         # Path to config file
    PipelineConfigs               # Backward compat: existing PipelineConfigs
]

DatasetSpec = Union[
    str,                          # Path to data folder
    Path,                         # Path to data folder
    np.ndarray,                   # X array (y inferred or None)
    Tuple[np.ndarray, ...],       # (X,) or (X, y) or (X, y, metadata)
    Dict[str, Any],               # Dict with X, y, metadata keys
    List[Dict[str, Any]],         # List of dataset dicts (multi-dataset)
    SpectroDataset,               # Direct SpectroDataset instance
    DatasetConfigs                # Backward compat: existing DatasetConfigs
]


def run(
    pipeline: PipelineSpec,
    dataset: DatasetSpec,
    *,
    name: str = "",
    session: Optional[Session] = None,
    # Common runner options (shortcuts for most-used parameters)
    verbose: int = 1,
    save_artifacts: bool = True,
    save_charts: bool = True,
    plots_visible: bool = False,
    random_state: Optional[int] = None,
    # All other PipelineRunner options
    **runner_kwargs: Any
) -> RunResult:
    """Execute a training pipeline on a dataset.

    This is the primary entry point for training ML pipelines on NIRS data.
    It provides a simpler interface than creating PipelineRunner and config
    objects directly.

    Args:
        pipeline: Pipeline definition. Can be:
            - List of steps (most common): ``[MinMaxScaler(), PLSRegression(10)]``
            - Dict with steps: ``{"steps": [...], "name": "my_pipeline"}``
            - Path to YAML/JSON config file: ``"configs/my_pipeline.yaml"``
            - PipelineConfigs object (backward compatibility)

        dataset: Dataset definition. Can be:
            - Path to data folder: ``"sample_data/regression"``
            - Numpy arrays: ``(X, y)`` or ``X`` alone
            - Dict with arrays: ``{"X": X, "y": y, "metadata": meta}``
            - SpectroDataset instance
            - DatasetConfigs object (backward compatibility)

        name: Optional pipeline name for identification and logging.
            If not provided, a name will be generated.

        session: Optional Session object for resource reuse across multiple
            runs. When provided, shares workspace and configuration.

        verbose: Verbosity level (0=quiet, 1=info, 2=debug, 3=trace).
            Default: 1

        save_artifacts: Whether to save binary artifacts (models, transformers).
            Default: True

        save_charts: Whether to save charts and visual outputs.
            Default: True

        plots_visible: Whether to display plots interactively.
            Default: False

        random_state: Random seed for reproducibility.
            Default: None (no seeding)

        **runner_kwargs: Additional PipelineRunner parameters. See
            PipelineRunner.__init__ for full list. Common options:
            - workspace_path: Workspace root directory
            - continue_on_error: Whether to continue on step failures
            - show_spinner: Whether to show progress spinners
            - log_file: Whether to write logs to disk
            - log_format: Output format ("pretty", "minimal", "json")
            - show_progress_bar: Whether to show progress bars
            - max_generation_count: Max pipeline combinations (for generators)

    Returns:
        RunResult containing:
            - predictions: Predictions object with all pipeline results
            - per_dataset: Dictionary with per-dataset execution details
            - best: Best prediction entry (convenience accessor)
            - best_score: Best model's primary test score
            - best_rmse, best_r2, best_accuracy: Score shortcuts

        Use ``result.top(n=5)`` to get top N predictions, or
        ``result.export("path.n4a")`` to export the best model.

    Raises:
        ValueError: If pipeline or dataset format is invalid.
        FileNotFoundError: If pipeline config or dataset path doesn't exist.

    Examples:
        Simple usage with list of steps:

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

        With cross-validation and multiple models:

        >>> from sklearn.model_selection import ShuffleSplit
        >>>
        >>> result = nirs4all.run(
        ...     pipeline=[
        ...         MinMaxScaler(),
        ...         ShuffleSplit(n_splits=3),
        ...         {"model": PLSRegression(10)}
        ...     ],
        ...     dataset="sample_data/regression",
        ...     name="PLS_experiment",
        ...     verbose=2,
        ...     save_artifacts=True
        ... )

        Using a session for multiple runs:

        >>> with nirs4all.session(verbose=1) as s:
        ...     r1 = nirs4all.run(pipeline1, data, session=s)
        ...     r2 = nirs4all.run(pipeline2, data, session=s)
        ...     print(f"Pipeline 1: {r1.best_score:.4f}")
        ...     print(f"Pipeline 2: {r2.best_score:.4f}")

        Export the best model:

        >>> result = nirs4all.run(pipeline, dataset)
        >>> result.export("exports/best_model.n4a")

    See Also:
        - :func:`nirs4all.predict`: Make predictions with a trained model
        - :func:`nirs4all.explain`: Generate SHAP explanations
        - :func:`nirs4all.session`: Create execution session for resource reuse
        - :class:`nirs4all.PipelineRunner`: Direct runner access for advanced use
    """
    # If session provided, use its runner
    if session is not None:
        runner = session.runner
        # Update runner settings if explicitly provided
        if verbose != 1:  # Not the default
            runner.verbose = verbose
    else:
        # Build runner kwargs from explicit params + extras
        all_kwargs = {
            "verbose": verbose,
            "save_artifacts": save_artifacts,
            "save_charts": save_charts,
            "plots_visible": plots_visible,
            **runner_kwargs
        }
        if random_state is not None:
            all_kwargs["random_state"] = random_state

        runner = PipelineRunner(**all_kwargs)

    # Execute the pipeline
    # PipelineRunner.run() already handles all format normalization for both
    # pipeline and dataset (PipelineConfigs, lists, dicts, paths, arrays, etc.)
    # Convert Path to str for compatibility with type hints
    pipeline_arg = str(pipeline) if isinstance(pipeline, Path) else pipeline
    dataset_arg = str(dataset) if isinstance(dataset, Path) else dataset

    predictions, per_dataset = runner.run(
        pipeline=pipeline_arg,
        dataset=dataset_arg,
        pipeline_name=name
    )

    return RunResult(
        predictions=predictions,
        per_dataset=per_dataset,
        _runner=runner
    )
