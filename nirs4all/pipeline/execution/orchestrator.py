"""Pipeline orchestrator for coordinating multiple pipeline executions."""
from pathlib import Path
from typing import Any

import numpy as np

from nirs4all.core.logging import get_logger
from nirs4all.data.config import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.execution.builder import ExecutorBuilder
from nirs4all.pipeline.execution.step_cache import StepCache
from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore
from nirs4all.visualization.reports import TabReportManager

logger = get_logger(__name__)


def _get_default_workspace_path() -> Path:
    """Get the default workspace path.

    Uses the workspace module's get_active_workspace() which checks:
    1. Explicitly set workspace via set_active_workspace()
    2. NIRS4ALL_WORKSPACE environment variable
    3. ./workspace in current working directory

    Returns:
        Default workspace path.
    """
    from nirs4all.workspace import get_active_workspace
    return get_active_workspace()


class PipelineOrchestrator:
    """Orchestrates execution of multiple pipelines across multiple datasets.

    High-level coordinator that manages:
    - Workspace initialization via WorkspaceStore
    - Global predictions aggregation
    - Best results reporting
    - Dataset/pipeline normalization

    Attributes:
        workspace_path: Root workspace directory
        store: WorkspaceStore for DuckDB-backed persistence
        verbose: Verbosity level
        mode: Execution mode (train/predict/explain)
        save_artifacts: Whether to save binary artifacts
        save_charts: Whether to save charts and visual outputs
        enable_tab_reports: Whether to generate tab reports
        keep_datasets: Whether to keep dataset snapshots
        plots_visible: Whether to display plots
    """

    def __init__(
        self,
        workspace_path: str | Path | None = None,
        verbose: int = 0,
        mode: str = "train",
        save_artifacts: bool = True,
        save_charts: bool = True,
        enable_tab_reports: bool = True,
        continue_on_error: bool = False,
        show_spinner: bool = True,
        keep_datasets: bool = True,
        plots_visible: bool = False
    ) -> None:
        """Initialize pipeline orchestrator.

        Args:
            workspace_path: Workspace root directory
            verbose: Verbosity level
            mode: Execution mode (train/predict/explain)
            save_artifacts: Whether to save binary artifacts
            save_charts: Whether to save charts and visual outputs
            enable_tab_reports: Whether to generate tab reports
            continue_on_error: Whether to continue on errors
            show_spinner: Whether to show spinners
            keep_datasets: Whether to keep dataset snapshots
            plots_visible: Whether to display plots
        """
        # Workspace configuration
        if workspace_path is None:
            workspace_path = _get_default_workspace_path()
        self.workspace_path = Path(workspace_path)

        # Create WorkspaceStore for DuckDB-backed persistence
        self.store = WorkspaceStore(self.workspace_path)

        # Create exports directory for on-demand exports
        (self.workspace_path / "exports").mkdir(parents=True, exist_ok=True)

        # Configuration
        self.verbose = verbose
        self.mode = mode
        self.save_artifacts = save_artifacts
        self.save_charts = save_charts
        self.enable_tab_reports = enable_tab_reports
        self.continue_on_error = continue_on_error
        self.show_spinner = show_spinner
        self.keep_datasets = keep_datasets
        self.plots_visible = plots_visible

        # Dataset snapshots (if keep_datasets is True)
        self.raw_data: dict[str, np.ndarray] = {}
        self.pp_data: dict[str, dict[str, np.ndarray]] = {}

        # Figure references to prevent garbage collection
        self._figure_refs: list[Any] = []

        # Legacy compatibility: runs_dir for predictor/explainer modes
        self.runs_dir = self.workspace_path

        # Store last executed pipeline info for post-run operations and syncing
        self.last_pipeline_uid: str | None = None
        self.last_executor: Any = None  # For syncing step_number, substep_number, operation_count
        self.last_aggregate_column: str | None = None  # Last dataset's aggregate setting
        self.last_aggregate_method: str | None = None  # Last dataset's aggregate method
        self.last_aggregate_exclude_outliers: bool = False  # Last dataset's exclude outliers setting
        self.last_execution_trace: Any = None  # ExecutionTrace from last run for post-run visualization
        self.last_run_id: str | None = None  # Last WorkspaceStore run UUID

    def execute(
        self,
        pipeline: PipelineConfigs | list[Any] | dict | str,
        dataset: DatasetConfigs | SpectroDataset | list[SpectroDataset] | np.ndarray | tuple[np.ndarray, ...] | dict | list[dict] | str | list[str],
        pipeline_name: str = "",
        dataset_name: str = "dataset",
        max_generation_count: int = 10000,
        artifact_loader: Any = None,
        target_model: dict[str, Any] | None = None,
        explainer: Any = None,
        refit: bool | dict[str, Any] | None = True,
    ) -> tuple[Predictions, dict[str, Any]]:
        """Execute pipeline configurations on dataset configurations.

        Args:
            pipeline: Pipeline definition (PipelineConfigs, List[steps], Dict, or file path)
            dataset: Dataset definition (DatasetConfigs, SpectroDataset, numpy arrays, Dict, or file path)
            pipeline_name: Optional name for the pipeline
            dataset_name: Optional name for array-based datasets
            max_generation_count: Maximum number of pipeline combinations to generate
            artifact_loader: ArtifactLoader for predict/explain modes
            target_model: Target model for predict/explain modes
            explainer: Explainer instance for explain mode
            refit: Refit configuration.
                - ``True``: Enable refit (retrain winning model on full training set, default).
                - ``False`` or ``None``: Disable refit (legacy behavior).
                - ``dict``: Refit options (reserved for future use).

        Returns:
            Tuple of (run_predictions, dataset_predictions)
        """
        from nirs4all.pipeline.config.context import RuntimeContext

        # Normalize inputs
        pipeline_configs = self._normalize_pipeline(
            pipeline,
            name=pipeline_name,
            max_generation_count=max_generation_count
        )
        dataset_configs = self._normalize_dataset(dataset, dataset_name=dataset_name)

        # Clear previous figure references
        self._figure_refs.clear()

        n_pipelines = len(pipeline_configs.steps)
        n_datasets = len(dataset_configs.configs)
        total_runs = n_pipelines * n_datasets
        logger.info("=" * 120)
        logger.starting(
            f"Starting Nirs4all run(s) with {n_pipelines} "
            f"pipeline(s) on {n_datasets} dataset(s) ({total_runs} total runs)."
        )
        logger.info("=" * 120)

        datasets_predictions = {}
        run_predictions = Predictions()
        current_run = 0

        # Begin run in store
        run_id = None
        if self.mode == "train":
            dataset_meta = []
            for _config, name in dataset_configs.configs:
                dataset_meta.append({"name": name})
            run_id = self.store.begin_run(
                name=pipeline_name or "run",
                config={"n_pipelines": n_pipelines, "n_datasets": n_datasets},
                datasets=dataset_meta,
            )
            self.last_run_id = run_id

        # Execute for each dataset
        try:
            for _dataset_idx, (config, name) in enumerate(dataset_configs.configs):
                # Create artifact registry for this dataset.
                # Lifecycle: Registry is created once per dataset, shared by all
                # pipeline variant (CV) executions, and will be available to the
                # refit pass.  It is destroyed (end_run) at the end of the dataset
                # loop iteration.
                artifact_registry = None
                if self.mode == "train":
                    artifact_registry = ArtifactRegistry(
                        workspace=self.workspace_path,
                        dataset=name,
                    )
                    artifact_registry.start_run()

                # Create step cache for this dataset.
                # Lifecycle: same as artifact_registry -- created once per
                # dataset, shared across all pipeline variants and the refit
                # pass, then discarded at the end of the dataset loop.
                step_cache = StepCache()

                # Build executor using ExecutorBuilder
                executor = (ExecutorBuilder()
                    .with_workspace(self.workspace_path)
                    .with_verbose(self.verbose)
                    .with_mode(self.mode)
                    .with_save_artifacts(self.save_artifacts)
                    .with_save_charts(self.save_charts)
                    .with_continue_on_error(self.continue_on_error)
                    .with_show_spinner(self.show_spinner)
                    .with_plots_visible(self.plots_visible)
                    .with_artifact_loader(artifact_loader)
                    .with_artifact_registry(artifact_registry)
                    .with_store(self.store)
                    .build())

                self.last_executor = executor

                # Predictions accumulate in-memory; store-backed via flush()
                run_dataset_predictions = Predictions()

                # Execute each pipeline configuration on this dataset
                for _i, (steps, config_name, gen_choices) in enumerate(zip(
                    pipeline_configs.steps,
                    pipeline_configs.names,
                    pipeline_configs.generator_choices,
                    strict=False,
                )):
                    current_run += 1
                    logger.info(f"Run {current_run}/{total_runs}: pipeline '{config_name}' on dataset '{name}'")

                    dataset = dataset_configs.get_dataset(config, name)

                    # Capture raw data BEFORE any preprocessing happens
                    if self.keep_datasets and name not in self.raw_data:
                        self.raw_data[name] = dataset.x({}, layout="2d")

                    if self.verbose > 0:
                        print(dataset)

                    # Initialize execution context via executor
                    context = executor.initialize_context(dataset)

                    # Create RuntimeContext with store
                    runtime_context = RuntimeContext(
                        store=self.store,
                        artifact_loader=artifact_loader,
                        artifact_registry=artifact_registry,
                        step_runner=executor.step_runner,
                        target_model=target_model,
                        explainer=explainer,
                        run_id=run_id,
                    )

                    # Execute pipeline with cleanup on failure
                    config_predictions = Predictions()
                    # Pass dataset repetition context for by_repetition=True resolution
                    if dataset.repetition:
                        config_predictions.set_repetition_column(dataset.repetition)
                    try:
                        executor.execute(
                            steps=steps,
                            config_name=config_name,
                            dataset=dataset,
                            context=context,
                            runtime_context=runtime_context,
                            prediction_store=config_predictions,
                            generator_choices=gen_choices
                        )
                    except Exception:
                        # Cleanup artifacts from failed run
                        if artifact_registry is not None:
                            artifact_registry.cleanup_failed_run()
                        raise

                    # Capture last pipeline_uid for syncing back to runner
                    if runtime_context.pipeline_uid:
                        self.last_pipeline_uid = runtime_context.pipeline_uid

                    # Capture execution trace for post-run visualization
                    self.last_execution_trace = runtime_context.get_execution_trace()

                    # Capture preprocessed data AFTER preprocessing
                    if self.keep_datasets:
                        if name not in self.pp_data:
                            self.pp_data[name] = {}
                        self.pp_data[name][dataset.short_preprocessings_str()] = dataset.x({}, layout="2d")

                    # Merge new predictions into run-level stores
                    if config_predictions.num_predictions > 0:
                        run_dataset_predictions.merge_predictions(config_predictions)
                        run_predictions.merge_predictions(config_predictions)

                # --- Pass 2: Refit (optional) ---
                # After all variants have been executed for this dataset,
                # retrain the winning model on the full training set.
                refit_enabled = refit is True or (isinstance(refit, dict) and refit)
                if refit_enabled and self.mode == "train" and run_id and run_dataset_predictions.num_predictions > 0:
                    self._execute_refit_pass(
                        run_id=run_id,
                        dataset=dataset,
                        executor=executor,
                        artifact_registry=artifact_registry,
                        run_dataset_predictions=run_dataset_predictions,
                        run_predictions=run_predictions,
                        artifact_loader=artifact_loader,
                        target_model=target_model,
                        explainer=explainer,
                    )

                # Log step cache statistics
                cache_stats = step_cache.stats()
                if cache_stats["hit_count"] > 0 or cache_stats["miss_count"] > 0:
                    logger.info(
                        f"Step cache: {cache_stats['hit_count']} hits, "
                        f"{cache_stats['miss_count']} misses "
                        f"({cache_stats['hit_rate']:.0%} hit rate), "
                        f"{cache_stats['size_mb']:.1f} MB used"
                    )

                # Mark run as completed successfully
                if artifact_registry is not None:
                    artifact_registry.end_run()

                # Store last aggregate column for visualization integration
                self.last_aggregate_column = dataset.aggregate
                self.last_aggregate_method = dataset.aggregate_method
                self.last_aggregate_exclude_outliers = dataset.aggregate_exclude_outliers

                # Print best results for this dataset
                self._print_best_predictions(
                    run_dataset_predictions,
                    dataset,
                    name,
                )

                # Store dataset prediction info
                datasets_predictions[name] = {
                    "run_predictions": run_dataset_predictions,
                    "dataset": dataset,
                    "dataset_name": name
                }

            # Complete run in store
            if run_id and self.mode == "train":
                summary = {"total_pipelines": total_runs}
                if run_predictions.num_predictions > 0:
                    best = run_predictions.get_best(ascending=None)
                    if best:
                        summary["best_score"] = best.get("test_score")
                        summary["best_metric"] = best.get("metric")
                self.store.complete_run(run_id, summary)

        except Exception as e:
            # Fail run in store
            if run_id and self.mode == "train":
                self.store.fail_run(run_id, str(e))
            raise

        return run_predictions, datasets_predictions

    def _normalize_pipeline(
        self,
        pipeline: PipelineConfigs | list[Any] | dict | str,
        name: str = "",
        max_generation_count: int = 10000
    ) -> PipelineConfigs:
        """Normalize pipeline input to PipelineConfigs."""
        if isinstance(pipeline, PipelineConfigs):
            return pipeline

        if isinstance(pipeline, list):
            pipeline_dict = {"pipeline": pipeline}
            return PipelineConfigs(pipeline_dict, name=name, max_generation_count=max_generation_count)

        return PipelineConfigs(pipeline, name=name, max_generation_count=max_generation_count)

    def _normalize_dataset(
        self,
        dataset: DatasetConfigs | SpectroDataset | list[SpectroDataset] | np.ndarray | tuple[np.ndarray, ...] | dict | list[dict] | str | list[str],
        dataset_name: str = "array_dataset"
    ) -> DatasetConfigs:
        """Normalize dataset input to DatasetConfigs."""
        if isinstance(dataset, DatasetConfigs):
            return dataset

        # Handle list of SpectroDataset instances
        if isinstance(dataset, list) and len(dataset) > 0 and isinstance(dataset[0], SpectroDataset):
            return self._wrap_dataset_list(dataset)

        # Simplified normalization - delegate to DatasetConfigs
        return DatasetConfigs(dataset) if not isinstance(dataset, (SpectroDataset, np.ndarray, tuple)) else self._wrap_dataset(dataset, dataset_name)

    def _wrap_dataset(self, dataset: SpectroDataset | np.ndarray | tuple, dataset_name: str) -> DatasetConfigs:
        """Wrap SpectroDataset or arrays in DatasetConfigs."""
        if isinstance(dataset, SpectroDataset):
            configs = DatasetConfigs.__new__(DatasetConfigs)
            configs.configs = [({"_preloaded_dataset": dataset}, dataset.name)]
            configs.cache = {dataset.name: self._extract_dataset_cache(dataset)}
            configs._task_types = ["auto"]  # Default task type for wrapped datasets
            configs._signal_type_overrides = [None]  # No override for wrapped datasets
            configs._aggregates = [None]  # No aggregation for wrapped datasets
            configs._aggregate_methods = [None]  # No aggregate method for wrapped datasets
            configs._aggregate_exclude_outliers = [False]  # No outlier exclusion for wrapped datasets
            configs._config_task_types = [None]  # No config-level task type
            configs._config_aggregates = [None]  # No config-level aggregate
            configs._config_aggregate_methods = [None]  # No config-level aggregate method
            configs._config_aggregate_exclude_outliers = [None]  # No config-level exclude outliers
            configs._config_repetitions = [None]  # No config-level repetition
            configs._repetitions = [None]  # No repetition for wrapped datasets
            return configs

        # Handle numpy arrays and tuples
        spectro_dataset = SpectroDataset(name=dataset_name)

        if isinstance(dataset, np.ndarray):
            # Single array X - for prediction mode, add to test partition only
            # For training mode, this would have y provided as tuple
            spectro_dataset.add_samples(dataset, indexes={"partition": "test"})
        elif isinstance(dataset, tuple):
            X = dataset[0]
            y = dataset[1] if len(dataset) > 1 else None
            partition_info = dataset[2] if len(dataset) > 2 else None

            if partition_info is None:
                # No partition info - add all to train with y if provided
                spectro_dataset.add_samples(X, indexes={"partition": "train"})
                if y is not None:
                    spectro_dataset.add_targets(y)
            else:
                # Split data based on partition_info
                self._split_and_add_data(spectro_dataset, X, y, partition_info)

        configs = DatasetConfigs.__new__(DatasetConfigs)
        configs.configs = [({"_preloaded_dataset": spectro_dataset}, dataset_name)]
        configs.cache = {dataset_name: self._extract_dataset_cache(spectro_dataset)}
        configs._task_types = ["auto"]  # Default task type for wrapped datasets
        configs._signal_type_overrides = [None]  # No override for wrapped datasets
        configs._aggregates = [None]  # No aggregation for wrapped datasets
        configs._aggregate_methods = [None]  # No aggregate method for wrapped datasets
        configs._aggregate_exclude_outliers = [False]  # No outlier exclusion for wrapped datasets
        configs._config_task_types = [None]  # No config-level task type
        configs._config_aggregates = [None]  # No config-level aggregate
        configs._config_aggregate_methods = [None]  # No config-level aggregate method
        configs._config_aggregate_exclude_outliers = [None]  # No config-level exclude outliers
        configs._config_repetitions = [None]  # No config-level repetition
        configs._repetitions = [None]  # No repetition for wrapped datasets
        return configs

    def _wrap_dataset_list(self, datasets: list[SpectroDataset]) -> DatasetConfigs:
        """Wrap a list of SpectroDataset instances in DatasetConfigs."""
        configs = DatasetConfigs.__new__(DatasetConfigs)
        configs.configs = []
        configs.cache = {}
        configs._task_types = []
        configs._signal_type_overrides = []
        configs._aggregates = []
        configs._aggregate_methods = []
        configs._aggregate_exclude_outliers = []
        configs._config_task_types = []
        configs._config_aggregates = []
        configs._config_aggregate_methods = []
        configs._config_aggregate_exclude_outliers = []
        configs._config_repetitions = []
        configs._repetitions = []

        for ds in datasets:
            configs.configs.append(({"_preloaded_dataset": ds}, ds.name))
            configs.cache[ds.name] = self._extract_dataset_cache(ds)
            configs._task_types.append("auto")
            configs._signal_type_overrides.append(None)
            configs._aggregates.append(None)
            configs._aggregate_methods.append(None)
            configs._aggregate_exclude_outliers.append(False)
            configs._config_task_types.append(None)
            configs._config_aggregates.append(None)
            configs._config_aggregate_methods.append(None)
            configs._config_aggregate_exclude_outliers.append(None)
            configs._config_repetitions.append(None)
            configs._repetitions.append(None)

        return configs

    def _split_and_add_data(self, dataset: SpectroDataset, X: np.ndarray, y: np.ndarray | None, partition_info: dict) -> None:
        """Split data according to partition_info and add to dataset.

        partition_info can be:
        - {"train": 80} - first 80 samples for train, rest for test
        - {"train": slice(0, 70), "test": slice(70, 100)} - explicit slices
        - {"train": [0,1,2,...], "test": [80,81,...]} - explicit indices
        """
        n_samples = X.shape[0]

        # Process partition_info to get indices for each partition
        partition_indices = {}

        for partition_name, partition_spec in partition_info.items():
            if isinstance(partition_spec, int):
                # Integer means "first N samples"
                partition_indices[partition_name] = slice(0, partition_spec)
            elif isinstance(partition_spec, (slice, list, np.ndarray)):
                partition_indices[partition_name] = partition_spec
            else:
                raise ValueError(f"Invalid partition spec for '{partition_name}': {partition_spec}")

        # If only train is specified, create test from remaining samples
        if "train" in partition_indices and "test" not in partition_indices:
            train_spec = partition_indices["train"]
            if isinstance(train_spec, slice):
                train_end = train_spec.stop if train_spec.stop is not None else train_spec.start
            elif isinstance(train_spec, int):
                train_end = train_spec
            else:
                # list of indices - find max + 1
                train_indices_array = np.array(train_spec)
                train_end = train_indices_array.max() + 1 if len(train_indices_array) > 0 else 0

            # Test partition is remaining samples
            if train_end < n_samples:
                partition_indices["test"] = slice(train_end, n_samples)

        # Add samples for each partition
        for partition_name, indices_spec in partition_indices.items():
            # Get the actual data slice
            if isinstance(indices_spec, (slice, list, np.ndarray)):
                X_partition = X[indices_spec]
                y_partition = y[indices_spec] if y is not None else None
            else:
                raise ValueError(f"Unexpected indices spec type: {type(indices_spec)}")

            # Add to dataset
            if len(X_partition) > 0:
                dataset.add_samples(X_partition, indexes={"partition": partition_name})
                if y_partition is not None and len(y_partition) > 0:
                    dataset.add_targets(y_partition)

    def _extract_dataset_cache(self, dataset: SpectroDataset) -> tuple:
        """Extract cache tuple from a SpectroDataset.

        Returns a 14-tuple matching the format expected by DatasetConfigs:
        (x_train, y_train, m_train, train_headers, m_train_headers, train_unit, train_signal_type,
         x_test, y_test, m_test, test_headers, m_test_headers, test_unit, test_signal_type)
        """
        try:
            x_train = dataset.x({"partition": "train"}, layout="2d")
            y_train = dataset.y({"partition": "train"})
            m_train = None
            train_signal_type = dataset.signal_type(0) if dataset.n_sources > 0 else None
        except Exception:
            x_train = y_train = m_train = None
            train_signal_type = None

        try:
            x_test = dataset.x({"partition": "test"}, layout="2d")
            y_test = dataset.y({"partition": "test"})
            m_test = None
            test_signal_type = dataset.signal_type(0) if dataset.n_sources > 0 else None
        except Exception:
            x_test = y_test = m_test = None
            test_signal_type = None

        # Return 14-tuple with signal_type included
        return (x_train, y_train, m_train, None, None, None, train_signal_type,
                x_test, y_test, m_test, None, None, None, test_signal_type)

    def _execute_refit_pass(
        self,
        run_id: str,
        dataset: SpectroDataset,
        executor: Any,
        artifact_registry: ArtifactRegistry | None,
        run_dataset_predictions: Predictions,
        run_predictions: Predictions,
        artifact_loader: Any = None,
        target_model: dict[str, Any] | None = None,
        explainer: Any = None,
    ) -> None:
        """Execute the refit pass after all CV variants have completed.

        Extracts the winning configuration from the store, analyzes its
        topology, and dispatches to the appropriate refit strategy.

        For non-stacking pipelines, uses ``execute_simple_refit`` which
        replaces the CV splitter with a single full-training-data fold
        and re-executes the winning pipeline.

        Args:
            run_id: Store run identifier.
            dataset: The dataset used for CV (will be deep-copied by refit).
            executor: PipelineExecutor instance.
            artifact_registry: Shared artifact registry from CV pass.
            run_dataset_predictions: Per-dataset prediction accumulator.
            run_predictions: Global prediction accumulator.
            artifact_loader: Optional artifact loader.
            target_model: Optional target model dict.
            explainer: Optional explainer instance.
        """
        from nirs4all.pipeline.analysis.topology import analyze_topology
        from nirs4all.pipeline.config.context import RuntimeContext
        from nirs4all.pipeline.execution.refit import execute_simple_refit, extract_winning_config
        from nirs4all.pipeline.execution.refit.stacking_refit import (
            execute_competing_branches_refit,
            execute_separation_refit,
            execute_stacking_refit,
        )

        try:
            refit_config = extract_winning_config(self.store, run_id)
        except ValueError as e:
            logger.warning(f"Cannot perform refit: {e}")
            return

        # Analyze topology to determine refit strategy
        topology = analyze_topology(refit_config.expanded_steps)

        # Create a RuntimeContext for the refit pass
        runtime_context = RuntimeContext(
            store=self.store,
            artifact_loader=artifact_loader,
            artifact_registry=artifact_registry,
            step_runner=executor.step_runner,
            target_model=target_model,
            explainer=explainer,
            run_id=run_id,
        )

        if topology.has_stacking or topology.has_mixed_merge:
            refit_result = execute_stacking_refit(
                refit_config=refit_config,
                dataset=dataset,
                context=None,
                runtime_context=runtime_context,
                artifact_registry=artifact_registry,
                executor=executor,
                prediction_store=run_dataset_predictions,
                topology=topology,
            )
        elif topology.has_branches_without_merge:
            refit_result = execute_competing_branches_refit(
                refit_config=refit_config,
                dataset=dataset,
                context=None,
                runtime_context=runtime_context,
                artifact_registry=artifact_registry,
                executor=executor,
                prediction_store=run_dataset_predictions,
                topology=topology,
            )
        elif topology.has_separation_branch:
            refit_result = execute_separation_refit(
                refit_config=refit_config,
                dataset=dataset,
                context=None,
                runtime_context=runtime_context,
                artifact_registry=artifact_registry,
                executor=executor,
                prediction_store=run_dataset_predictions,
                topology=topology,
            )
        else:
            refit_result = execute_simple_refit(
                refit_config=refit_config,
                dataset=dataset,
                context=None,  # executor.initialize_context is called inside
                runtime_context=runtime_context,
                artifact_registry=artifact_registry,
                executor=executor,
                prediction_store=run_dataset_predictions,
            )

        if refit_result.success:
            # Also merge refit predictions into the global run predictions
            if refit_result.predictions_count > 0:
                logger.info(
                    f"Refit completed: {refit_result.predictions_count} "
                    f"prediction(s) added with fold_id='final'"
                )

            # Clean up transient CV fold artifacts now that the refit model exists
            try:
                removed = self.store.cleanup_transient_artifacts(
                    run_id=run_id,
                    dataset_name=dataset.name,
                    winning_pipeline_ids=[refit_config.pipeline_id],
                )
                if removed > 0:
                    logger.info(f"Cleaned up {removed} transient artifact file(s)")
            except Exception as e:
                logger.warning(f"Transient artifact cleanup failed (non-fatal): {e}")
        else:
            logger.warning("Refit pass did not complete successfully")

    def _print_best_predictions(
        self,
        run_dataset_predictions: Predictions,
        dataset: SpectroDataset,
        name: str,
    ) -> None:
        """Print best predictions for a dataset.

        Reports best predictions to the logger.  Persistence is handled
        by :meth:`Predictions.flush` / :class:`WorkspaceStore`.
        """
        if run_dataset_predictions.num_predictions > 0:
            # Use None for ascending to let ranker infer from metric
            best = run_dataset_predictions.get_best(
                ascending=None
            )
            logger.success(f"Best prediction in run for dataset '{name}': {Predictions.pred_long_string(best)}")

            if self.enable_tab_reports:
                best_by_partition = run_dataset_predictions.get_entry_partitions(best)

                # Get aggregation setting from dataset for reporting
                aggregate_column = dataset.aggregate  # Could be None, 'y', or column name
                aggregate_method = dataset.aggregate_method  # Could be None, 'mean', 'median', 'vote'
                aggregate_exclude_outliers = dataset.aggregate_exclude_outliers

                # Log aggregation info if enabled
                if aggregate_column:
                    agg_label = "y (target values)" if aggregate_column == 'y' else f"'{aggregate_column}'"
                    method_label = f", method='{aggregate_method}'" if aggregate_method else ""
                    outlier_label = ", exclude_outliers=True" if aggregate_exclude_outliers else ""
                    logger.info(f"Including aggregated scores (by {agg_label}{method_label}{outlier_label}) in report")

                tab_report, tab_report_csv_file = TabReportManager.generate_best_score_tab_report(
                    best_by_partition,
                    aggregate=aggregate_column,
                    aggregate_method=aggregate_method,
                    aggregate_exclude_outliers=aggregate_exclude_outliers
                )
                logger.info(tab_report)

        logger.info("=" * 120)
