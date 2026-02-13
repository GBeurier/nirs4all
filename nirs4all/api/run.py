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
from nirs4all.config.cache_config import CacheConfig
from nirs4all.data import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions

from .result import RunResult
from .session import Session


# Type aliases for a single pipeline or dataset (not lists)
SinglePipelineSpec = Union[
    List[Any],                    # List of steps (most common)
    Dict[str, Any],               # Dict configuration
    str,                          # Path to YAML/JSON config
    Path,                         # Path to config file
    PipelineConfigs               # Backward compat: existing PipelineConfigs
]

SingleDatasetSpec = Union[
    str,                          # Path to data folder
    Path,                         # Path to data folder
    np.ndarray,                   # X array (y inferred or None)
    Tuple[np.ndarray, ...],       # (X,) or (X, y) or (X, y, metadata)
    Dict[str, Any],               # Dict with X, y, metadata keys
    SpectroDataset,               # Direct SpectroDataset instance
    DatasetConfigs,               # Backward compat: existing DatasetConfigs
]

# Type aliases that also support lists for batch execution
PipelineSpec = Union[
    SinglePipelineSpec,
    List[SinglePipelineSpec],     # List of pipelines for batch execution
]

DatasetSpec = Union[
    SingleDatasetSpec,
    List[SingleDatasetSpec],      # List of datasets for batch execution
]


def _is_single_pipeline(pipeline: Any) -> bool:
    """Check if pipeline is a single pipeline definition (not a list of pipelines).

    A single pipeline can be:
    - A PipelineConfigs object
    - A dict (configuration)
    - A str/Path (config file path)
    - A list of steps where steps are NOT themselves lists of steps

    A list of pipelines would be:
    - A list where elements are themselves lists of steps (each inner list is a pipeline)
    """
    if isinstance(pipeline, (PipelineConfigs, dict, str, Path)):
        return True

    if isinstance(pipeline, list):
        if len(pipeline) == 0:
            return True  # Empty list treated as single empty pipeline

        # Check the first element to determine if this is a list of pipelines
        # or a single pipeline (list of steps)
        first = pipeline[0]

        # If the first element is a list, this could be:
        # 1. A list of pipelines (each is a list of steps)
        # 2. A single pipeline with a sub-pipeline as first step (rare)
        #
        # Heuristic: if the first element is a list AND contains typical step objects
        # (dicts with known keys like "model", "preprocessing", etc., or class instances),
        # it's likely a list of pipelines.

        if isinstance(first, list):
            # Check if the inner list looks like a pipeline (list of steps)
            if len(first) > 0:
                inner_first = first[0]
                # If inner elements are dicts, classes, instances, etc., it's likely
                # that the outer list is a list of pipelines
                if isinstance(inner_first, (dict, str)) or _looks_like_step(inner_first):
                    return False  # It's a list of pipelines

        # Otherwise, treat as a single pipeline
        return True

    return True


def _looks_like_step(obj: Any) -> bool:
    """Check if an object looks like a pipeline step.

    Steps can be:
    - sklearn-like objects (have fit/transform/predict methods)
    - Class objects (types)
    - Dicts with step configuration keys
    - None (no-op step)
    """
    if obj is None:
        return True
    if isinstance(obj, type):
        return True  # It's a class
    if isinstance(obj, dict):
        return True
    # Check if it's an instance with sklearn-like interface
    if hasattr(obj, 'fit') or hasattr(obj, 'transform') or hasattr(obj, 'predict'):
        return True
    # Check for nirs4all transforms
    if hasattr(obj, '__class__') and obj.__class__.__module__.startswith('nirs4all'):
        return True
    return False


def _is_single_dataset(dataset: Any) -> bool:
    """Check if dataset is a single dataset definition (not a list of datasets).

    A single dataset can be:
    - A DatasetConfigs object
    - A SpectroDataset instance
    - A str/Path (data folder path)
    - A numpy array
    - A tuple of arrays (X, y, ...)
    - A dict with X, y keys

    A list of datasets would be:
    - A list where each element is a dataset spec (str, SpectroDataset, dict, array, tuple)
    """
    if isinstance(dataset, (DatasetConfigs, SpectroDataset, str, Path, np.ndarray, tuple)):
        return True

    if isinstance(dataset, dict):
        # A dict could be a dataset config or data dict
        # If it has 'X' or 'features' key, it's a data dict (single dataset)
        # Otherwise it could be a DatasetConfigs-like dict
        return True

    if isinstance(dataset, list):
        if len(dataset) == 0:
            return True  # Empty list treated as single empty dataset

        first = dataset[0]

        # List of SpectroDataset is treated as multi-dataset (handled specially by orchestrator)
        if isinstance(first, SpectroDataset):
            return False

        # List of paths/strings -> multi-dataset
        if isinstance(first, (str, Path)):
            return False

        # List of dicts where each dict is a dataset config -> multi-dataset
        if isinstance(first, dict):
            # Check if it looks like a dataset config (has path, X, etc.)
            if 'path' in first or 'X' in first or 'features' in first:
                return False

        # List of arrays or tuples -> multi-dataset
        if isinstance(first, (np.ndarray, tuple)):
            return False

        # Otherwise, it might be something else, treat as single
        return True

    return True


def _normalize_to_list(spec: Any, is_single_fn) -> List[Any]:
    """Normalize a spec (pipeline or dataset) to a list of specs.

    If it's a single spec, wrap it in a list.
    If it's already a list of specs, return as-is.
    """
    if is_single_fn(spec):
        return [spec]
    else:
        return spec


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
    refit: Union[bool, Dict[str, Any], List[Dict[str, Any]], None] = True,
    cache: Optional[Any] = None,
    project: Optional[str] = None,
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
            - **List of pipelines**: ``[pipeline1, pipeline2, ...]`` - each
              pipeline is executed independently (cartesian product with datasets)

        dataset: Dataset definition. Can be:
            - Path to data folder: ``"sample_data/regression"``
            - Numpy arrays: ``(X, y)`` or ``X`` alone
            - Dict with arrays: ``{"X": X, "y": y, "metadata": meta}``
            - SpectroDataset instance
            - List of SpectroDataset instances (multi-dataset)
            - DatasetConfigs object (backward compatibility)
            - **List of datasets**: ``[dataset1, dataset2, ...]`` - each
              dataset is used with each pipeline (cartesian product)

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

        refit: Refit configuration. After cross-validation selects the
            winning pipeline variant(s), retrain on the full training set.
            - ``True``: Refit top 1 by RMSECV (default).
            - ``False`` or ``None``: Disable refit.
            - ``dict``: Single criterion, e.g. ``{"top_k": 3, "ranking": "mean_val"}``.
            - ``list[dict]``: Multiple criteria for union selection, e.g.
              ``[{"top_k": 3, "ranking": "rmsecv"}, {"top_k": 1, "ranking": "mean_val"}]``.
            Ranking methods: ``"rmsecv"`` (OOF concatenated val score),
            ``"mean_val"`` (mean of individual fold val scores).

        cache: Optional CacheConfig for step-level caching.
            - ``None``: Use default CacheConfig (step cache OFF, CoW snapshots ON).
            - ``CacheConfig(step_cache_enabled=True)``: Enable step caching.

        project: Optional project name to tag the run with.  If the project
            does not exist yet it will be created automatically.

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

        Multiple pipelines executed independently:

        >>> pipeline_pls = [MinMaxScaler(), PLSRegression(10)]
        >>> pipeline_rf = [StandardScaler(), RandomForestRegressor()]
        >>>
        >>> result = nirs4all.run(
        ...     pipeline=[pipeline_pls, pipeline_rf],  # Two independent pipelines
        ...     dataset="sample_data/regression",
        ...     verbose=1
        ... )
        >>> print(f"Total configs: {result.num_predictions}")

        Cartesian product of pipelines × datasets:

        >>> pipelines = [pipeline1, pipeline2, pipeline3]
        >>> datasets = [dataset_a, dataset_b]
        >>>
        >>> # Runs 6 combinations: p1×da, p1×db, p2×da, p2×db, p3×da, p3×db
        >>> result = nirs4all.run(
        ...     pipeline=pipelines,
        ...     dataset=datasets,
        ...     verbose=1
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
    # Normalize pipelines and datasets to lists
    pipelines = _normalize_to_list(pipeline, _is_single_pipeline)
    datasets = _normalize_to_list(dataset, _is_single_dataset)

    # Extract store_run_id before passing runner_kwargs to PipelineRunner
    caller_store_run_id: Optional[str] = runner_kwargs.pop("store_run_id", None)

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

    # Set cache config on runner (flows to orchestrator -> runtime_context).
    # Default path keeps step cache disabled while enabling CoW snapshots.
    runner.cache_config = cache if cache is not None else CacheConfig()

    # Execute the cartesian product: each pipeline × each dataset
    all_predictions = Predictions()
    all_per_dataset: Dict[str, Any] = {}
    total_combos = len(pipelines) * len(datasets)

    # When multiple combos or caller provided a store_run_id, group all
    # pipeline×dataset executions under a single store run.
    shared_run_id: Optional[str] = caller_store_run_id
    caller_owns_run = caller_store_run_id is not None  # Caller manages lifecycle
    multi_run = total_combos > 1 or shared_run_id is not None

    if multi_run and shared_run_id is None:
        # Pre-create a single store run for the whole batch
        store = getattr(runner, 'orchestrator', runner).store if hasattr(runner, 'orchestrator') else None
        if store is None:
            store = getattr(getattr(runner, 'orchestrator', None), 'store', None)
        if store is not None and hasattr(store, 'begin_run'):
            import json as _json
            dataset_meta = []
            for ds in datasets:
                if isinstance(ds, str):
                    dataset_meta.append({"name": Path(ds).stem})
                elif hasattr(ds, 'name'):
                    dataset_meta.append({"name": ds.name})
                else:
                    dataset_meta.append({"name": "dataset"})
            shared_run_id = store.begin_run(
                name=name or "run",
                config={"n_pipelines": len(pipelines), "n_datasets": len(datasets)},
                datasets=dataset_meta,
            )

    try:
        for pipeline_idx, single_pipeline in enumerate(pipelines):
            for dataset_idx, single_dataset in enumerate(datasets):
                # Generate name with index if multiple pipelines
                if len(pipelines) > 1:
                    pipeline_name = f"{name}_p{pipeline_idx}" if name else f"pipeline_{pipeline_idx}"
                else:
                    pipeline_name = name

                # Convert Path to str for compatibility with type hints
                pipeline_arg = str(single_pipeline) if isinstance(single_pipeline, Path) else single_pipeline
                dataset_arg = str(single_dataset) if isinstance(single_dataset, Path) else single_dataset

                predictions, per_dataset = runner.run(
                    pipeline=pipeline_arg,
                    dataset=dataset_arg,
                    pipeline_name=pipeline_name,
                    refit=refit,
                    store_run_id=shared_run_id,
                    manage_store_run=not multi_run,
                )

                # Merge predictions from this run
                all_predictions.merge_predictions(predictions)

                # Merge per_dataset info (datasets with same name will be combined)
                for ds_name, ds_info in per_dataset.items():
                    if ds_name not in all_per_dataset:
                        all_per_dataset[ds_name] = ds_info
                    else:
                        # Merge run_predictions from multiple runs on same dataset
                        existing_run_preds = all_per_dataset[ds_name].get("run_predictions")
                        new_run_preds = ds_info.get("run_predictions")
                        if existing_run_preds is not None and new_run_preds is not None:
                            existing_run_preds.merge_predictions(new_run_preds)

        # Complete the shared store run (only if we created it, not if caller owns it)
        if multi_run and shared_run_id is not None and not caller_owns_run:
            store = getattr(getattr(runner, 'orchestrator', None), 'store', None)
            if store is not None and hasattr(store, 'complete_run'):
                summary: Dict[str, Any] = {"total_pipelines": total_combos}
                if all_predictions.num_predictions > 0:
                    best = all_predictions.get_best(ascending=None)
                    if best:
                        summary["best_score"] = best.get("test_score")
                        summary["best_metric"] = best.get("metric")
                store.complete_run(shared_run_id, summary)

    except Exception as e:
        # Fail the shared store run (only if we created it)
        if multi_run and shared_run_id is not None and not caller_owns_run:
            store = getattr(getattr(runner, 'orchestrator', None), 'store', None)
            if store is not None and hasattr(store, 'fail_run'):
                store.fail_run(shared_run_id, str(e))
        raise

    # Extract per-model selections from the orchestrator (if available)
    orchestrator = getattr(runner, 'orchestrator', None)
    per_model_selections = getattr(orchestrator, '_per_model_selections', None) if orchestrator else None

    # Tag the run with a project if requested
    if project is not None and orchestrator is not None:
        run_id = shared_run_id or getattr(orchestrator, 'last_run_id', None)
        store = getattr(orchestrator, 'store', None)
        if run_id and store and hasattr(store, 'get_or_create_project'):
            project_id = store.get_or_create_project(project)
            store.set_run_project(run_id, project_id)

    return RunResult(
        predictions=all_predictions,
        per_dataset=all_per_dataset,
        _runner=runner,
        _per_model_selections=per_model_selections,
    )
