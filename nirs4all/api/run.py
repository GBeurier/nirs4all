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

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np

from nirs4all.config.cache_config import CacheConfig
from nirs4all.data import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.pipeline.engine import resolve_engine

from .result import RunResult
from .session import Session

if TYPE_CHECKING:
    from .tuning import TunedSingleEstimatorConformalResult

# Type aliases for a single pipeline or dataset (not lists)
SinglePipelineSpec: TypeAlias = (
    list[Any]  # List of steps (most common)
    | dict[str, Any]  # Dict configuration
    | str  # Path to YAML/JSON config
    | Path  # Path to config file
    | PipelineConfigs  # Backward compat: existing PipelineConfigs
)

SingleDatasetSpec: TypeAlias = (
    str  # Path to data folder
    | Path  # Path to data folder
    | np.ndarray  # X array (y inferred or None)
    | tuple[np.ndarray, ...]  # (X,) or (X, y) or (X, y, metadata)
    | dict[str, Any]  # Dict with X, y, metadata keys
    | SpectroDataset  # Direct SpectroDataset instance
    | DatasetConfigs  # Backward compat: existing DatasetConfigs
)

# Type aliases that also support lists for batch execution
PipelineSpec: TypeAlias = (
    SinglePipelineSpec | list[SinglePipelineSpec]  # List of pipelines for batch execution
)

DatasetSpec: TypeAlias = (
    SingleDatasetSpec | list[SingleDatasetSpec]  # List of datasets for batch execution
)


def _is_pipeline_wrapper_dict(obj: Any) -> bool:
    """Return True for dicts that wrap a full pipeline definition."""
    return isinstance(obj, dict) and any(key in obj for key in ("pipeline", "steps"))


def _is_batch_pipeline_list(pipeline: list[Any]) -> bool:
    """Return True only when every outer item is clearly a full pipeline spec.

    This prevents nested canonical steps such as ``[[...], {...}]`` from being
    misread as a batch of pipelines when they are actually one pipeline with a
    nested list step followed by ordinary steps.
    """
    if not pipeline:
        return False

    saw_pipeline_spec = False
    for item in pipeline:
        if isinstance(item, (PipelineConfigs, str, Path)):
            saw_pipeline_spec = True
            continue
        if isinstance(item, list):
            saw_pipeline_spec = True
            continue
        if _is_pipeline_wrapper_dict(item):
            saw_pipeline_spec = True
            continue
        if isinstance(item, dict) or _looks_like_step(item):
            return False
        return False

    return saw_pipeline_spec


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
        return not _is_batch_pipeline_list(pipeline)

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
    if hasattr(obj, "fit") or hasattr(obj, "transform") or hasattr(obj, "predict"):
        return True
    # Check for nirs4all transforms
    return bool(hasattr(obj, "__class__") and obj.__class__.__module__.startswith("nirs4all"))


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
        if isinstance(first, dict) and ("path" in first or "X" in first or "features" in first):
            return False

        # List of arrays or tuples -> multi-dataset
        return not isinstance(first, (np.ndarray, tuple))

    return True


def _normalize_to_list(spec: Any, is_single_fn) -> list[Any]:
    """Normalize a spec (pipeline or dataset) to a list of specs.

    If it's a single spec, wrap it in a list.
    If it's already a list of specs, return as-is.
    """
    if is_single_fn(spec):
        return [spec]
    else:
        return list(spec)


def run(
    pipeline: PipelineSpec,
    dataset: DatasetSpec,
    *,
    name: str = "",
    session: Session | None = None,
    # Common runner options (shortcuts for most-used parameters)
    verbose: int = 1,
    save_artifacts: bool = True,
    save_charts: bool = True,
    plots_visible: bool = False,
    random_state: int | None = None,
    refit: bool | dict[str, Any] | list[dict[str, Any]] | None = True,
    cache: Any | None = None,
    project: str | None = None,
    report_naming: str = "nirs",
    engine: str | None = None,
    tuning: Any | None = None,
    calibration: Any | None = None,
    results_path: str | Path | None = None,
    training_losses: tuple[Mapping[str, Any], ...] = (),
    local_implementations: Any | None = None,
    # All other PipelineRunner options
    **runner_kwargs: Any,
) -> "RunResult | TunedSingleEstimatorConformalResult":
    """Execute a training pipeline on a dataset.

    This is the primary entry point for training ML pipelines on NIRS data.
    It provides a simpler interface than creating PipelineRunner and config
    objects directly.

    Args:
        pipeline: Pipeline definition. Can be:
            - List of steps (most common): ``[MinMaxScaler(), PLSRegression(10)]``
            - Dict with steps: ``{"steps": [...], "name": "my_pipeline"}``
              or the public wrapper alias ``{"pipeline": [...], "name": "my_pipeline"}``
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

        report_naming: Naming convention for metrics in reports and summaries.
            - ``"nirs"`` (default): Chemometrics terminology (RMSECV, RMSEP, etc.)
            - ``"ml"``: Machine learning terminology (CV_Score, Test_Score, etc.)
            - ``"auto"``: Auto-detect based on context (defaults to "nirs")
            Affects column headers in final summary tables. Internal variable names
            use ML conventions regardless of this setting.

        engine: Execution backend selector. ``None`` (default) resolves to ``"legacy"``: the
            public-maintained nirs4all stays pure-Python by default and runs on the in-process
            orchestrator (interim posture until the planned global refactoring lands). ``"dag-ml"``
            runs the pipeline natively on the dag-ml backend (Rust, in-process by default), with a
            TRANSPARENT fallback to the legacy engine (a warning is emitted) when a pipeline shape is
            not yet covered or the dag-ml backend is not installed. ``"dual"`` (side-by-side
            comparison) is reserved and raises ``NotImplementedError``. Override the default
            per-process with ``$N4A_ENGINE`` (e.g. ``$N4A_ENGINE=dag-ml``).

        tuning: Typed native tuning specification for the currently supported
            DAG-ML subset. With ``engine="dag-ml"``, this supports explicit
            array datasets, a single estimator or linear sklearn-like
            transformer→estimator chain, Optuna/n4m-compatible ``space``,
            required ``score_data`` for objective scoring, optional
            ``winner`` projection, workspace persistence/resume, and optional
            conformal ``calibration``. Broader DAG branch/merge graphs,
            dataset loaders, arbitrary structural model selection and legacy
            execution remain fail-closed. Deterministic model-local
            ``finetune_params`` grids are still the native path for the older
            graph-generation subset.

        calibration: Optional top-level conformal calibration payload for the
            currently supported native tuning subset. This is equivalent to
            ``tuning["calibration"]`` and requires ``run(tuning=..., engine="dag-ml")``.
            Calibration evidence is derived from ``tuning["winner"]``; arbitrary
            legacy runs and payloads that supply their own ``calibration_data`` remain
            fail-closed.

        results_path: Native results output root (dag-ml engine only, P3 Slice 2b-i; OFF by default).
            When given, the dag-ml run ADDITIONALLY writes a native results directory
            ``<results_path>/<run_id>/`` (``manifest.json`` + the verbatim ``score_set.json`` +
            ``predictions.parquet``); ``$N4A_NATIVE_RESULTS`` enables it env-only, defaulting to
            ``./nirs4all_results/<run_id>/``. ``None`` + unset env → nothing is written and the run is
            behaviorally identical to today. The legacy SQLite+Parquet+joblib workspace is untouched;
            an explicit ``results_path`` is threaded as a named ``run()`` parameter (not a runner_kwarg),
            so it bypasses the dag-ml runner_kwarg allowlist. It has no effect on ``engine="legacy"``.

        training_losses: DAG-ML-native training loss role references for
            process-local custom training losses. This is honored only with
            ``engine="dag-ml"`` and must be paired with ``local_implementations``.

        local_implementations: Process-local DAG-ML implementation registry
            (for Python, typically ``dag_ml.LocalImplementationRegistry``) used
            to bind ``training_losses`` to callable custom losses. It cannot be
            serialized or forwarded to the legacy backend.

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
        When ``run(tuning=..., calibration=...)`` or nested
        ``tuning["calibration"]`` is supplied, returns a
        ``TunedSingleEstimatorConformalResult`` containing both the tuned
        ``run`` result and the calibrated conformal ``calibrated`` result.

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
    custom_training_loss_requested = bool(training_losses) or local_implementations is not None
    if custom_training_loss_requested and resolve_engine(engine) != "dag-ml":
        raise ValueError("training_losses/local_implementations require engine='dag-ml'")

    if calibration is not None:
        if tuning is None:
            raise NotImplementedError("run(calibration=...) currently requires run(tuning=..., engine='dag-ml') with an explicit tuning.winner")
        tuning = _coerce_public_tuning_payload(tuning)
        if not isinstance(tuning, Mapping):
            raise TypeError("run(tuning=...) must be a mapping when run(calibration=...) is supplied")
        if tuning.get("calibration") is not None:
            raise ValueError("provide conformal calibration either as run(calibration=...) or tuning.calibration, not both")
        tuning = dict(tuning)
        tuning["calibration"] = _coerce_public_runtime_payload(calibration)

    if tuning is not None:
        if custom_training_loss_requested:
            raise NotImplementedError(
                "run(tuning=...) does not yet thread DAG-ML process-local training losses; "
                "use a concrete engine='dag-ml' pipeline until the native tuning adapter binds "
                "training_losses and local_implementations."
            )
        tuning = _coerce_public_tuning_payload(tuning)
        if resolve_engine(engine) == "dag-ml":
            return _run_single_estimator_tuning_subset(
                pipeline,
                dataset,
                tuning,
                name=name,
                runner_kwargs=runner_kwargs,
            )
        from nirs4all.pipeline.dagml.tuning_contracts import DagMLTuningNotImplementedError, parse_tuning_spec

        tuning_spec = parse_tuning_spec(_tuning_spec_payload(tuning))
        raise DagMLTuningNotImplementedError(tuning_spec)

    def _run_legacy() -> RunResult:
        """Run the in-process legacy orchestrator path (the engine='legacy' behaviour).

        Defined as a closure over ``run()``'s arguments so both the default path and the
        dag-ml→legacy cutover fallback re-enter the SAME code without re-passing every parameter.
        """
        # Normalize pipelines and datasets to lists
        pipelines = _normalize_to_list(pipeline, _is_single_pipeline)
        datasets = _normalize_to_list(dataset, _is_single_dataset)

        # Extract store_run_id before passing runner_kwargs to PipelineRunner
        caller_store_run_id: str | None = runner_kwargs.pop("store_run_id", None)

        # If session provided, use its runner
        if session is not None:
            runner = session.runner
            # Update runner settings if explicitly provided
            if verbose != 1:  # Not the default
                runner.verbose = verbose
        else:
            # Build runner kwargs from explicit params + extras
            all_kwargs = {"verbose": verbose, "save_artifacts": save_artifacts, "save_charts": save_charts, "plots_visible": plots_visible, "report_naming": report_naming, **runner_kwargs}
            if random_state is not None:
                all_kwargs["random_state"] = random_state

            runner = PipelineRunner(**all_kwargs)

        # Set cache config on runner (flows to orchestrator -> runtime_context).
        # Default path keeps step cache disabled while enabling CoW snapshots.
        runner.cache_config = cache if cache is not None else CacheConfig()

        # Execute the cartesian product: each pipeline × each dataset
        all_predictions = Predictions()
        all_per_dataset: dict[str, Any] = {}
        total_combos = len(pipelines) * len(datasets)

        # When multiple combos or caller provided a store_run_id, group all
        # pipeline×dataset executions under a single store run.
        shared_run_id: str | None = caller_store_run_id
        caller_owns_run = caller_store_run_id is not None  # Caller manages lifecycle
        multi_run = total_combos > 1 or shared_run_id is not None

        if multi_run and shared_run_id is None:
            # Pre-create a single store run for the whole batch
            store = getattr(runner, "orchestrator", runner).store if hasattr(runner, "orchestrator") else None
            if store is None:
                store = getattr(getattr(runner, "orchestrator", None), "store", None)
            if store is not None and hasattr(store, "begin_run"):
                dataset_meta = []
                for ds in datasets:
                    if isinstance(ds, str):
                        dataset_meta.append({"name": Path(ds).stem})
                    elif hasattr(ds, "name"):
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
                for _dataset_idx, single_dataset in enumerate(datasets):
                    # Generate name with index if multiple pipelines
                    pipeline_name = f"{name}_p{pipeline_idx}" if name else f"pipeline_{pipeline_idx}" if len(pipelines) > 1 else name

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
                store = getattr(getattr(runner, "orchestrator", None), "store", None)
                if store is not None and hasattr(store, "complete_run"):
                    summary: dict[str, Any] = {"total_pipelines": total_combos}
                    if all_predictions.num_predictions > 0:
                        best = all_predictions.get_best(ascending=None, score_scope="all")
                        if best:
                            summary["best_score"] = best.get("test_score")
                            summary["best_metric"] = best.get("metric")
                    store.complete_run(shared_run_id, summary)

        except Exception as e:
            # Fail the shared store run (only if we created it)
            if multi_run and shared_run_id is not None and not caller_owns_run:
                store = getattr(getattr(runner, "orchestrator", None), "store", None)
                if store is not None and hasattr(store, "fail_run"):
                    store.fail_run(shared_run_id, str(e))
            raise

        # Extract per-model selections from the orchestrator (if available)
        orchestrator = getattr(runner, "orchestrator", None)
        per_model_selections = getattr(orchestrator, "_per_model_selections", None) if orchestrator else None

        # Tag the run with a project if requested
        if project is not None and orchestrator is not None:
            run_id = shared_run_id or getattr(orchestrator, "last_run_id", None)
            store = getattr(orchestrator, "store", None)
            if run_id and store and hasattr(store, "get_or_create_project"):
                project_id = store.get_or_create_project(project)
                store.set_run_project(run_id, project_id)

        result = RunResult(
            predictions=all_predictions,
            per_dataset=all_per_dataset,
            _runner=runner,
            _owns_runner=session is None,
            _workspace_path=runner.workspace_path,
            _per_model_selections=per_model_selections,
        )

        # For non-session runs, detach from the runner immediately to release
        # the DB connection.  Export operations will re-open the store on demand.
        if session is None:
            result.detach()

        return result

    # ADR-17 backend selector (nirs4all-core -> dag-ml migration). The DEFAULT engine is LEGACY again
    # (interim posture: the public-maintained nirs4all stays pure-Python by default until the planned
    # global refactoring; the legacy-DROP cutover flips it back to dag-ml). dag-ml stays FULLY SELECTABLE
    # via `engine="dag-ml"` / `$N4A_ENGINE=dag-ml`: it dispatches to the dag-ml backend, which runs the
    # pipeline natively (Rust, IN-PROCESS by default via the PyO3 extension) and returns a RunResult of
    # dag-ml's native scores. A plain `run()` (the legacy default) runs the in-process legacy orchestrator
    # (`_run_legacy`).
    #
    # TRANSPARENT LEGACY FALLBACK. The default is 100% safe via two catchable signals, both warned:
    #   * SHAPE not yet covered — the catchable coverage-boundary shapes (no splitter, .n4a export,
    #     rich stacking, non-default refit/session/cache/project/workspace, …) raise DagMlUnsupported,
    #     a NotImplementedError subclass.
    #   * BACKEND not installed — the dag-ml preflight raises DagMlUnavailable when NEITHER mechanism
    #     is present (no in-process extension AND no dag-ml-cli). dag-ml is a HARD dependency, but a
    #     wheel install missing the native backend still degrades gracefully rather than crashing.
    # In either case we warn and re-run on the legacy engine (run_via_dagml cleans its own temp dir in
    # a finally). ONLY DagMlUnsupported/NotImplementedError/DagMlUnavailable are caught — a GENUINE
    # dag-ml runtime/operator bug propagates untouched (never silently swallowed into legacy).
    if resolve_engine(engine) == "dag-ml":
        from nirs4all.pipeline.dagml.errors import DagMlUnavailable, DagMlUnsupported
        from nirs4all.pipeline.dagml.run_backend import run_via_dagml

        try:
            # Forward EVERY run() kwarg that affects the dag-ml run so engine='dag-ml' honors the same
            # options as legacy (P1b). Two regimes:
            #   HONORED natively — `random_state` (global seeding, like legacy) and `name` (DERIVED into
            #     the canonical legacy `config_name` via PipelineConfigs — `config_{hash}` unnamed /
            #     `{name}_p0_{hash}` named, `_refit` on the refit rows — and carried on the dag-ml
            #     RunResult predictions; a generator pipeline's winner-only projection carries no
            #     config_name rather than a wrong one, #55).
            #   VALIDATED, fall back if un-honorable — `refit`, `session`, `cache`, `project`, and the
            #     workspace/persistence runner_kwargs are checked against what the scores-only in-memory
            #     dag-ml path can deliver; a non-default value it cannot satisfy raises DagMlUnsupported
            #     (caught below → legacy fallback), so no user option is ever silently dropped.
            # Defaults are honored natively and never trigger a fallback, so a plain engine='dag-ml' run
            # runs natively. The remaining kwargs are presentation/logging-only for the current
            # score-only path: `save_artifacts=True` (the default) runs natively — the dag-ml run keeps no
            # on-disk artifacts, but .n4a export is now bridged (P1c): RunResult.export() re-fits the same
            # pipeline on the legacy engine on demand (a documented best-effort for unseeded-stochastic
            # shapes; exact for deterministic ones via engine parity); `save_charts=True` is accepted only
            # because any chart-producing pipeline step is itself unsupported→fallback;
            # `verbose`/`plots_visible`/`report_naming` affect only logging/display, never the scores.
            return run_via_dagml(
                pipeline,
                dataset,
                name=name,
                random_state=random_state,
                refit=refit,
                project=project,
                session=session,
                cache=cache,
                runner_kwargs=runner_kwargs,
                results_path=results_path,
                training_losses=training_losses,
                local_implementations=local_implementations,
            )
        except DagMlUnavailable as e:
            if custom_training_loss_requested:
                raise
            warnings.warn(
                f"the dag-ml backend is not available ({e}); falling back to the legacy engine",
                stacklevel=2,
            )
            return _run_legacy()
        except (DagMlUnsupported, NotImplementedError) as e:
            if custom_training_loss_requested:
                raise
            warnings.warn(
                f"engine='dag-ml' does not support this pipeline shape ({e}); falling back to the legacy engine",
                stacklevel=2,
            )
            return _run_legacy()

    return _run_legacy()


def _run_single_estimator_tuning_subset(
    pipeline: Any,
    dataset: Any,
    tuning: Any,
    *,
    name: str,
    runner_kwargs: dict[str, Any],
) -> RunResult:
    """Run the currently supported public ``run(tuning=...)`` subset.

    The supported subset is intentionally narrow: single-estimator pipeline,
    explicit array dataset, and explicit ``tuning.score_data``. Broader pipeline
    syntax remains fail-closed through ``DagMLTuningNotImplementedError``.
    """

    from nirs4all.api.tuning import tune_single_estimator
    from nirs4all.pipeline.dagml.tuning_contracts import DagMLTuningNotImplementedError, parse_tuning_spec

    tuning_spec_payload = _tuning_spec_payload(tuning)
    tuning_spec = parse_tuning_spec(tuning_spec_payload)
    execution_tuning_spec = tuning_spec
    (
        X_score,
        y_score,
        score_metric,
        score_sample_ids,
        score_groups,
        score_metadata,
        score_extractor,
    ) = _score_data_for_run(tuning)

    X, y, sample_ids, groups, metadata = _raw_array_tuning_dataset(dataset)
    winner_payload = _optional_mapping_value(tuning, "winner")
    (
        winner_x,
        winner_y_true,
        winner_score,
        winner_metric,
        winner_sample_ids,
        winner_dataset_name,
        winner_model_name,
        winner_task_type,
        winner_metadata,
    ) = _winner_payload_for_run(winner_payload)
    workspace_path = runner_kwargs.get("workspace_path")
    workspace_tuning_id = _optional_mapping_single_alias(
        tuning,
        "run(tuning=...).workspace_tuning_id",
        "workspace_tuning_id",
        "tuning_id",
    )
    resume_tuning_result = None
    if tuning_spec.resume:
        if workspace_path is None or not workspace_tuning_id:
            raise ValueError("run(tuning.resume=True) requires workspace_path and tuning.workspace_tuning_id")
        from nirs4all.api.tuning import load_workspace_tuning_result

        resume_tuning_result = load_workspace_tuning_result(workspace_path, workspace_tuning_id)
        execution_tuning_spec = parse_tuning_spec({**tuning_spec_payload, "resume": False})
        if resume_tuning_result.tuning.fingerprint != execution_tuning_spec.fingerprint:
            raise ValueError("workspace tuning result does not match the requested run(tuning=...) contract")
    try:
        result = tune_single_estimator(
            pipeline,
            X,
            y,
            execution_tuning_spec,
            X_score=None if score_extractor is not None else X_score,
            y_score=None if score_extractor is not None else y_score,
            score_extractor=score_extractor,
            score_metric=None if score_extractor is not None else score_metric,
            score_sample_ids=None if score_extractor is not None else score_sample_ids,
            score_groups=None if score_extractor is not None else score_groups,
            score_metadata=None if score_extractor is not None else score_metadata,
            sample_ids=sample_ids,
            groups=groups,
            metadata=metadata,
            workspace_path=workspace_path,
            workspace_name=name,
            workspace_tuning_id=workspace_tuning_id,
            workspace_metadata=_optional_mapping_value(tuning, "workspace_metadata"),
            resume_tuning_result=resume_tuning_result,
            resume_tuning_id=workspace_tuning_id,
            winner_x=winner_x,
            winner_y_true=winner_y_true,
            winner_score=winner_score,
            winner_metric=winner_metric,
            winner_sample_ids=winner_sample_ids,
            winner_dataset_name=winner_dataset_name,
            winner_model_name=winner_model_name,
            winner_task_type=winner_task_type,
            winner_metadata=winner_metadata,
            calibration=_calibration_payload_for_run(tuning, workspace_path=workspace_path, workspace_name=name),
        )
    except (NotImplementedError, TypeError) as exc:
        raise DagMLTuningNotImplementedError(tuning_spec) from exc
    return cast(RunResult, result)


def _tuning_spec_payload(tuning: Any) -> dict[str, Any]:
    from nirs4all.pipeline.dagml.tuning_contracts import SUPPORTED_TUNING_KEYS

    if not isinstance(tuning, dict):
        if not isinstance(tuning, Mapping):
            raise TypeError("run(tuning=...) must be a mapping")
        tuning = dict(tuning)
    allowed_runtime_keys = {
        "calibration",
        "score_data",
        "tuning_id",
        "winner",
        "workspace_metadata",
        "workspace_tuning_id",
    }
    unknown = sorted(set(tuning) - set(SUPPORTED_TUNING_KEYS) - allowed_runtime_keys)
    if unknown:
        raise ValueError(f"run(tuning=...) does not support keys {unknown}")
    return {key: value for key, value in tuning.items() if key in SUPPORTED_TUNING_KEYS}


def _coerce_public_tuning_payload(tuning: Any) -> Any:
    to_dict = getattr(tuning, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return tuning


def _coerce_public_runtime_payload(payload: Any) -> Any:
    to_dict = getattr(payload, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return payload


def _mapping_value(payload: Any, key: str, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("run(tuning=...) must be a mapping")
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _score_data_for_run(tuning: Any) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    if not isinstance(tuning, Mapping):
        raise TypeError("run(tuning=...) must be a mapping")
    if "score_data" not in tuning:
        raise ValueError("run(tuning=...).score_data must be supplied")
    value = tuning.get("score_data")
    if isinstance(value, tuple | list):
        if len(value) < 2:
            raise ValueError("run(tuning=...).score_data tuple/list must contain (X_score, y_score)")
        if len(value) > 5:
            raise ValueError("run(tuning=...).score_data tuple/list supports at most (X_score, y_score, sample_ids, groups, metadata)")
        return (
            value[0],
            value[1],
            None,
            value[2] if len(value) > 2 else None,
            value[3] if len(value) > 3 else None,
            value[4] if len(value) > 4 else None,
            None,
        )
    if not isinstance(value, Mapping):
        raise ValueError("run(tuning=...).score_data must be a mapping or tuple/list")
    _validate_score_data_aliases(value)
    if "dataset" in value or "spectro_dataset" in value:
        X_score, y_score, sample_ids, groups, metadata = _spectro_dataset_arrays(
            value,
            label="run(tuning=...).score_data",
            forbid_xy=True,
            identity_label="score",
        )
        score_metric = _optional_mapping_single_alias(value, "run(tuning=...).score_data metric", "metric", "score_metric")
        score_extractor = _score_extractor_for_run(
            value,
            X_score,
            y_score,
            score_metric,
            _optional_mapping_first(tuning, "metric"),
            sample_ids,
            groups,
            metadata,
        )
        return (
            X_score,
            y_score,
            score_metric,
            sample_ids,
            groups,
            metadata,
            score_extractor,
        )
    X_score = _optional_mapping_single_alias(value, "run(tuning=...).score_data features", "X", "X_score")
    y_score = _optional_mapping_single_alias(value, "run(tuning=...).score_data target", "y", "y_score")
    if X_score is None or y_score is None:
        raise ValueError("run(tuning=...).score_data requires X/y or X_score/y_score")
    score_metric = _optional_mapping_single_alias(value, "run(tuning=...).score_data metric", "metric", "score_metric")
    score_sample_ids = _optional_mapping_single_alias(
        value,
        "run(tuning=...).score_data sample_ids",
        "sample_ids",
        "score_sample_ids",
        "prediction_sample_ids",
        "physical_sample_ids",
    )
    score_groups = _optional_mapping_single_alias(value, "run(tuning=...).score_data groups", "groups", "score_groups")
    score_metadata = _optional_mapping_single_alias(value, "run(tuning=...).score_data metadata", "metadata", "score_metadata")
    score_extractor = _score_extractor_for_run(
        value,
        X_score,
        y_score,
        score_metric,
        _optional_mapping_first(tuning, "metric"),
        score_sample_ids,
        score_groups,
        score_metadata,
    )
    return (
        X_score,
        y_score,
        score_metric,
        score_sample_ids,
        score_groups,
        score_metadata,
        score_extractor,
    )


def _score_extractor_for_run(
    score_data: Mapping[str, Any],
    X_score: Any,
    y_score: Any,
    score_metric: Any,
    tuning_metric: Any,
    score_sample_ids: Any,
    score_groups: Any,
    score_metadata: Any,
) -> Any:
    conformal_calibration = _optional_mapping_single_alias(
        score_data,
        "run(tuning=...).score_data conformal_calibration",
        "conformal_calibration",
        "conformal_score_calibration",
    )
    if conformal_calibration is None:
        return None
    _validate_conformal_score_calibration_aliases(conformal_calibration)
    from nirs4all.pipeline.dagml.pipeline_objective import make_conformal_prediction_score_extractor

    coverage = _optional_mapping_single_alias(
        score_data,
        "run(tuning=...).score_data conformal_coverage",
        "conformal_coverage",
        "coverage",
    )
    return make_conformal_prediction_score_extractor(
        score_metric or tuning_metric or "rmse",
        X_score,
        y_score,
        conformal_calibration,
        coverage=0.9 if coverage is None else coverage,
        sample_ids=score_sample_ids,
        groups=score_groups,
        metadata=score_metadata,
    )


def _validate_score_data_aliases(score_data: Mapping[str, Any]) -> None:
    _optional_mapping_single_alias(score_data, "run(tuning=...).score_data dataset", "dataset", "spectro_dataset")
    _optional_mapping_single_alias(score_data, "run(tuning=...).score_data features", "X", "X_score")
    _optional_mapping_single_alias(score_data, "run(tuning=...).score_data target", "y", "y_score")
    _optional_mapping_single_alias(score_data, "run(tuning=...).score_data metric", "metric", "score_metric")
    _optional_mapping_single_alias(
        score_data,
        "run(tuning=...).score_data sample_ids",
        "sample_ids",
        "score_sample_ids",
        "prediction_sample_ids",
        "physical_sample_ids",
    )
    _optional_mapping_single_alias(score_data, "run(tuning=...).score_data groups", "groups", "score_groups")
    _optional_mapping_single_alias(score_data, "run(tuning=...).score_data metadata", "metadata", "score_metadata")
    _optional_mapping_single_alias(
        score_data,
        "run(tuning=...).score_data conformal_calibration",
        "conformal_calibration",
        "conformal_score_calibration",
    )
    _optional_mapping_single_alias(
        score_data,
        "run(tuning=...).score_data conformal_coverage",
        "conformal_coverage",
        "coverage",
    )
    if ("dataset" in score_data or "spectro_dataset" in score_data) and any(key in score_data for key in ("X", "X_score", "y", "y_score")):
        raise ValueError("run(tuning=...).score_data dataset-backed mappings must not also provide X/y arrays")


def _validate_conformal_score_calibration_aliases(calibration_data: Any) -> None:
    if not isinstance(calibration_data, Mapping):
        raise TypeError("run(tuning=...).score_data.conformal_calibration must be a mapping")
    _optional_mapping_single_alias(
        calibration_data,
        "run(tuning=...).score_data.conformal_calibration features",
        "X",
        "X_calibration",
        "features",
    )
    _optional_mapping_single_alias(
        calibration_data,
        "run(tuning=...).score_data.conformal_calibration target",
        "y",
        "y_true",
        "y_calibration",
        "target",
        "targets",
    )
    _optional_mapping_single_alias(
        calibration_data,
        "run(tuning=...).score_data.conformal_calibration sample_ids",
        "sample_ids",
        "calibration_sample_ids",
        "physical_sample_ids",
    )
    _optional_mapping_single_alias(
        calibration_data,
        "run(tuning=...).score_data.conformal_calibration groups",
        "groups",
        "calibration_groups",
    )
    _optional_mapping_single_alias(
        calibration_data,
        "run(tuning=...).score_data.conformal_calibration metadata",
        "metadata",
        "calibration_metadata",
    )


def _winner_payload_for_run(
    winner_payload: Mapping[str, Any] | None,
) -> tuple[Any, Any, Any, Any, Any, str, Any, str, Mapping[str, Any] | None]:
    if winner_payload is None:
        return None, None, None, None, None, "tuning_winner", None, "regression", None
    _validate_winner_payload_aliases(winner_payload)
    if "dataset" in winner_payload or "spectro_dataset" in winner_payload:
        winner_x, winner_y_true, winner_sample_ids, winner_groups, winner_metadata = _spectro_dataset_arrays(
            winner_payload,
            label="run(tuning=...).winner",
            forbid_xy=True,
            identity_label="winner",
        )
        return (
            winner_x,
            winner_y_true,
            _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner score", "score", "winner_score"),
            _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner metric", "metric", "winner_metric"),
            winner_sample_ids,
            _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner dataset_name", "dataset_name", "winner_dataset_name") or "tuning_winner",
            _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner model_name", "model_name", "winner_model_name"),
            _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner task_type", "task_type", "winner_task_type") or "regression",
            _winner_metadata_mapping(winner_metadata, winner_groups),
        )
    return (
        _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner features", "X", "x", "winner_x"),
        _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner target", "y_true", "winner_y_true"),
        _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner score", "score", "winner_score"),
        _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner metric", "metric", "winner_metric"),
        _optional_mapping_single_alias(
            winner_payload,
            "run(tuning=...).winner sample_ids",
            "sample_ids",
            "winner_sample_ids",
            "prediction_sample_ids",
            "physical_sample_ids",
        ),
        _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner dataset_name", "dataset_name", "winner_dataset_name") or "tuning_winner",
        _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner model_name", "model_name", "winner_model_name"),
        _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner task_type", "task_type", "winner_task_type") or "regression",
        _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner metadata", "metadata", "winner_metadata"),
    )


def _validate_winner_payload_aliases(winner_payload: Mapping[str, Any]) -> None:
    _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner dataset", "dataset", "spectro_dataset")
    _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner features", "X", "x", "winner_x")
    _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner target", "y_true", "winner_y_true")
    _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner score", "score", "winner_score")
    _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner metric", "metric", "winner_metric")
    _optional_mapping_single_alias(
        winner_payload,
        "run(tuning=...).winner sample_ids",
        "sample_ids",
        "winner_sample_ids",
        "prediction_sample_ids",
        "physical_sample_ids",
    )
    _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner dataset_name", "dataset_name", "winner_dataset_name")
    _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner model_name", "model_name", "winner_model_name")
    _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner task_type", "task_type", "winner_task_type")
    _optional_mapping_single_alias(winner_payload, "run(tuning=...).winner metadata", "metadata", "winner_metadata")
    if ("dataset" in winner_payload or "spectro_dataset" in winner_payload) and any(key in winner_payload for key in ("X", "x", "winner_x", "y_true", "winner_y_true")):
        raise ValueError("run(tuning=...).winner dataset-backed mappings must not also provide X/y_true arrays")


def _winner_metadata_mapping(metadata: Any, groups: Any) -> Mapping[str, Any] | None:
    result: dict[str, Any] = {}
    if metadata is not None:
        if isinstance(metadata, Mapping):
            result.update(dict(metadata))
        else:
            rows = list(metadata)
            if rows and all(isinstance(row, Mapping) for row in rows):
                keys = sorted({str(key) for row in rows for key in row})
                for key in keys:
                    result[key] = [row.get(key) for row in rows]
            elif rows:
                raise ValueError("run(tuning=...).winner metadata must be a mapping or row mappings")
    if groups is not None and "group" not in result:
        result["group"] = list(groups)
    return result or None


def _optional_mapping_value(payload: Any, key: str) -> Mapping[str, Any] | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise TypeError("expected a mapping")
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _optional_mapping_first(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    if mapping is None:
        return None
    return _first_present(mapping, *keys)


def _optional_mapping_single_alias(mapping: Mapping[str, Any] | None, label: str, *keys: str) -> Any:
    if mapping is None:
        return None
    provided = [(key, mapping[key]) for key in keys if key in mapping and mapping[key] is not None]
    if not provided:
        return None
    if len(provided) > 1:
        names = ", ".join(key for key, _value in provided)
        raise ValueError(f"{label} received multiple aliases ({names}); provide exactly one")
    return provided[0][1]


def _calibration_payload_for_run(
    tuning: Any,
    *,
    workspace_path: Any,
    workspace_name: str,
) -> dict[str, Any] | None:
    payload = _optional_mapping_value(tuning, "calibration")
    if payload is None:
        return None
    if "calibration_data" in payload:
        raise ValueError("run(tuning=...).calibration must not include calibration_data; calibration evidence is derived from tuning.winner")
    result = dict(payload)
    if workspace_path is not None:
        result.setdefault("workspace_path", workspace_path)
        result.setdefault("workspace_name", workspace_name)
    return result


def _raw_array_tuning_dataset(dataset: Any) -> tuple[Any, Any, Any, Any, Any]:
    if isinstance(dataset, SpectroDataset):
        raise ValueError("run(tuning=...) SpectroDataset input must use an explicit mapping {'dataset': spectro_dataset, 'selector': {...}}")
    if isinstance(dataset, tuple | list):
        if len(dataset) < 2:
            raise ValueError("run(tuning=...) array dataset tuple/list must contain (X, y)")
        if len(dataset) > 5:
            raise ValueError("run(tuning=...) array dataset tuple/list supports at most (X, y, sample_ids, groups, metadata)")
        X, y = dataset[0], dataset[1]
        sample_ids = dataset[2] if len(dataset) > 2 else None
        groups = dataset[3] if len(dataset) > 3 else None
        metadata = dataset[4] if len(dataset) > 4 else None
        _validate_tuning_dataset_identity(sample_ids, y=y, name="dataset sample_ids")
        _validate_tuning_dataset_identity(groups, y=y, name="dataset groups")
        _validate_tuning_dataset_identity(metadata, y=y, name="dataset metadata")
        return X, y, sample_ids, groups, metadata
    if isinstance(dataset, Mapping):
        if "dataset" in dataset or "spectro_dataset" in dataset:
            return _spectro_dataset_tuning_dataset(dataset)
        X = _first_present(dataset, "X", "features")
        y = _first_present(dataset, "y", "target", "targets")
        if X is None or y is None:
            raise ValueError("run(tuning=...) dataset mapping requires X/y")
        sample_ids = _first_present(dataset, "sample_ids", "physical_sample_ids")
        groups = dataset.get("groups")
        metadata = dataset.get("metadata")
        _validate_tuning_dataset_identity(sample_ids, y=y, name="dataset sample_ids")
        _validate_tuning_dataset_identity(groups, y=y, name="dataset groups")
        _validate_tuning_dataset_identity(metadata, y=y, name="dataset metadata")
        return (
            X,
            y,
            sample_ids,
            groups,
            metadata,
        )
    raise ValueError("run(tuning=...) currently supports only explicit array datasets: (X, y), [X, y], or {'X': ..., 'y': ...}")


def _spectro_dataset_tuning_dataset(dataset: Mapping[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    return _spectro_dataset_arrays(
        dataset,
        label="run(tuning=...) SpectroDataset",
        forbid_xy=True,
        identity_label="dataset",
    )


def _spectro_dataset_arrays(
    dataset: Mapping[str, Any],
    *,
    label: str,
    forbid_xy: bool,
    identity_label: str,
) -> tuple[Any, Any, Any, Any, Any]:
    spectro = _coerce_tuning_dataset_source(
        _first_present(dataset, "dataset", "spectro_dataset"),
        label=label,
    )
    if forbid_xy and any(key in dataset for key in ("X", "features", "X_score", "y", "target", "targets", "y_score")):
        raise ValueError(f"{label} mappings must not also provide X/y arrays")

    selector = dataset.get("selector")
    if selector is None:
        raise ValueError(f"{label} mapping requires an explicit selector")
    if not isinstance(selector, Mapping):
        raise ValueError(f"{label} selector must be a mapping")
    selector_dict = dict(selector)
    include_augmented = bool(dataset.get("include_augmented", False))

    X = spectro.x(
        selector_dict,
        layout="2d",
        concat_source=True,
        include_augmented=include_augmented,
    )
    if isinstance(X, list):
        raise ValueError(f"{label} extraction requires a single 2D feature matrix")
    y = spectro.y(selector_dict, include_augmented=include_augmented)
    y_array = np.asarray(y)
    if y_array.ndim == 2 and y_array.shape[1] == 1:
        y = y_array[:, 0]

    sample_ids = _dataset_identity_alias(dataset, identity_label)
    if sample_ids is None:
        sample_id_column = _first_present(dataset, "sample_id_column", "physical_sample_id_column")
        if sample_id_column is not None:
            sample_ids = spectro.metadata_column(
                str(sample_id_column),
                selector_dict,
                include_augmented=include_augmented,
            )

    groups = dataset.get("groups")
    if groups is None:
        group_column = dataset.get("group_column")
        if group_column is not None:
            groups = spectro.metadata_column(
                str(group_column),
                selector_dict,
                include_augmented=include_augmented,
            )

    metadata = dataset.get("metadata")
    if metadata is None:
        metadata_columns = dataset.get("metadata_columns")
        if metadata_columns is not None:
            if isinstance(metadata_columns, str):
                metadata_columns = [metadata_columns]
            metadata_frame = spectro.metadata(
                selector_dict,
                columns=list(metadata_columns),
                include_augmented=include_augmented,
            )
            if hasattr(metadata_frame, "to_dicts"):
                metadata = metadata_frame.to_dicts()
            else:
                metadata = metadata_frame

    _validate_tuning_dataset_identity(sample_ids, y=y, name=f"{identity_label} sample_ids")
    _validate_tuning_dataset_identity(groups, y=y, name=f"{identity_label} groups")
    _validate_tuning_dataset_identity(metadata, y=y, name=f"{identity_label} metadata")
    return X, y, sample_ids, groups, metadata


def _dataset_identity_alias(dataset: Mapping[str, Any], identity_label: str) -> Any:
    if identity_label == "score":
        return _optional_mapping_single_alias(dataset, "run(tuning=...).score_data sample_ids", "sample_ids", "score_sample_ids", "prediction_sample_ids", "physical_sample_ids")
    if identity_label == "winner":
        return _optional_mapping_single_alias(dataset, "run(tuning=...).winner sample_ids", "sample_ids", "winner_sample_ids", "prediction_sample_ids", "physical_sample_ids")
    return _optional_mapping_single_alias(dataset, f"{identity_label} sample_ids", "sample_ids", "physical_sample_ids")


def _coerce_tuning_dataset_source(source: Any, *, label: str) -> SpectroDataset:
    """Resolve an explicit tuning dataset source through existing nirs4all loaders."""

    if isinstance(source, SpectroDataset):
        return source
    if isinstance(source, DatasetConfigs):
        configs = source
    elif isinstance(source, (str, Path, Mapping)):
        configs = DatasetConfigs(cast(dict[str, Any] | list[dict[str, Any]] | str | list[str], source))
    else:
        raise ValueError(f"{label} mapping key 'dataset' must contain a SpectroDataset, DatasetConfigs, dataset config mapping, or config/path string")
    if len(configs.configs) != 1:
        raise ValueError(f"{label} mapping key 'dataset' must resolve to exactly one dataset")
    return configs.get_dataset_at(0)


def _validate_tuning_dataset_identity(value: Any, *, y: Any, name: str) -> None:
    if value is None:
        return
    expected = _n_tuning_rows(y)
    actual = _n_tuning_rows(value)
    if actual and expected and actual != expected:
        raise ValueError(f"{name} contains {actual} rows but y contains {expected} values")


def _n_tuning_rows(value: Any) -> int:
    if isinstance(value, Mapping):
        lengths = [_n_tuning_rows(column) for column in value.values()]
        non_zero_lengths = [length for length in lengths if length]
        if not non_zero_lengths:
            return 0
        if len(set(non_zero_lengths)) != 1:
            raise ValueError("dataset metadata contains inconsistent row lengths")
        return non_zero_lengths[0]
    if isinstance(value, (str, bytes, bytearray)):
        return 0
    shape = getattr(value, "shape", None)
    if shape:
        try:
            return int(shape[0])
        except (TypeError, ValueError, IndexError):
            pass
    try:
        return len(value)
    except TypeError:
        return 0
