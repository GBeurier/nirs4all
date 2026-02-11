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
from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore
from nirs4all.visualization.reports import TabReportManager

logger = get_logger(__name__)


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
        keep_datasets: bool = False,
        max_preprocessed_snapshots_per_dataset: int = 3,
        plots_visible: bool = False,
        random_state: int | None = None,
    ) -> None:
        """Initialize pipeline orchestrator.

        Args:
            workspace_path: Workspace root directory (required).
            verbose: Verbosity level
            mode: Execution mode (train/predict/explain)
            save_artifacts: Whether to save binary artifacts
            save_charts: Whether to save charts and visual outputs
            enable_tab_reports: Whether to generate tab reports
            continue_on_error: Whether to continue on errors
            show_spinner: Whether to show spinners
            keep_datasets: Whether to keep dataset snapshots
            max_preprocessed_snapshots_per_dataset: Maximum number of
                preprocessed snapshots to retain per dataset
            plots_visible: Whether to display plots
            random_state: Random seed for reproducibility propagation
        """
        # Workspace configuration
        if workspace_path is None:
            raise ValueError(
                "workspace_path must be provided explicitly to PipelineOrchestrator. "
                "Use PipelineRunner for CLI/default workspace resolution."
            )
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
        self.max_preprocessed_snapshots_per_dataset = max(0, int(max_preprocessed_snapshots_per_dataset))
        self.plots_visible = plots_visible
        self.random_state = random_state

        # Dataset snapshots (if keep_datasets is True)
        self.raw_data: dict[str, np.ndarray] = {}
        self.pp_data: dict[str, dict[str, np.ndarray]] = {}

        # Figure references to prevent garbage collection
        self._figure_refs: list[Any] = []

        # Legacy compatibility: runs_dir for predictor/explainer modes
        self.runs_dir = self.workspace_path

        # Cache configuration (set by PipelineRunner before execute())
        self.cache_config: Any = None

        # Per-model selections from refit (populated by _execute_per_model_refit)
        self._per_model_selections: dict[str, Any] | None = None

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
        store_run_id: str | None = None,
        manage_store_run: bool = True,
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
            store_run_id: Optional existing store run ID to join. When provided,
                skips ``begin_run()`` and uses this ID directly.
            manage_store_run: Whether to manage the store run lifecycle
                (begin/complete/fail). Set to ``False`` when the caller manages
                the run lifecycle externally (e.g. multi-pipeline batch).

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

        # Begin run in store (or join existing run)
        run_id = store_run_id
        if self.mode == "train" and run_id is None and manage_store_run:
            dataset_meta = []
            for _config, name in dataset_configs.configs:
                dataset_meta.append({"name": name})
            run_id = self.store.begin_run(
                name=pipeline_name or "run",
                config={"n_pipelines": n_pipelines, "n_datasets": n_datasets},
                datasets=dataset_meta,
            )
        if run_id:
            self.last_run_id = run_id

        # Create StepCache if step caching is enabled
        step_cache = None
        if self.cache_config is not None and getattr(self.cache_config, 'step_cache_enabled', False):
            from nirs4all.pipeline.execution.step_cache import StepCache
            step_cache = StepCache(
                max_size_mb=self.cache_config.step_cache_max_mb,
                max_entries=self.cache_config.step_cache_max_entries,
            )

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

                # Accumulator for best preprocessing chain per model across all variants.
                # Shared across pipeline variants via RuntimeContext (which returns
                # self on deepcopy).
                best_refit_chains: dict = {}

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
                        cache_config=self.cache_config,
                        step_cache=step_cache,
                        best_refit_chains=best_refit_chains,
                        random_state=self.random_state,
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
                        snapshot_key = dataset.short_preprocessings_str()
                        dataset_snapshots = self.pp_data[name]
                        can_store = (
                            snapshot_key in dataset_snapshots
                            or self.max_preprocessed_snapshots_per_dataset == 0
                            or len(dataset_snapshots) < self.max_preprocessed_snapshots_per_dataset
                        )
                        if can_store:
                            dataset_snapshots[snapshot_key] = dataset.x({}, layout="2d")
                        else:
                            logger.debug(
                                "Skipping preprocessed snapshot for dataset '%s' "
                                "(limit=%d, key='%s')",
                                name,
                                self.max_preprocessed_snapshots_per_dataset,
                                snapshot_key,
                            )

                    # Merge new predictions into run-level stores
                    if config_predictions.num_predictions > 0:
                        run_dataset_predictions.merge_predictions(config_predictions)
                        run_predictions.merge_predictions(config_predictions)

                # --- Pass 2: Refit (optional) ---
                # After all variants have been executed for this dataset,
                # retrain the winning model on the full training set.
                refit_enabled = refit is True or (isinstance(refit, dict) and refit)
                if refit_enabled and self.mode == "train" and run_id and run_dataset_predictions.num_predictions > 0:
                    # Re-load a pristine dataset for refit.
                    # The per-variant training runs mutate dataset state
                    # (added processings/folds/exclusions), so passing the
                    # last trained instance into refit can cause duplicate
                    # processing IDs and stale fold/index mappings.
                    refit_dataset = dataset_configs.get_dataset(config, name)
                    self._execute_refit_pass(
                        run_id=run_id,
                        dataset=refit_dataset,
                        executor=executor,
                        artifact_registry=artifact_registry,
                        run_dataset_predictions=run_dataset_predictions,
                        run_predictions=run_predictions,
                        artifact_loader=artifact_loader,
                        target_model=target_model,
                        explainer=explainer,
                        best_refit_chains=best_refit_chains,
                    )

                # Mark run as completed successfully
                if artifact_registry is not None:
                    artifact_registry.end_run()

                # Log step cache statistics at end of dataset
                if step_cache is not None and self.cache_config and self.cache_config.log_cache_stats:
                    cache_stats = step_cache.stats()
                    if cache_stats["hits"] + cache_stats["misses"] > 0:
                        logger.info(
                            f"Step cache: {cache_stats['hits']} hits, "
                            f"{cache_stats['misses']} misses "
                            f"({cache_stats['hit_rate']:.0%} hit rate), "
                            f"{cache_stats['evictions']} evictions, "
                            f"peak {cache_stats['peak_mb']:.1f} MB"
                        )

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

            # Complete run in store (only if we manage the lifecycle)
            if run_id and self.mode == "train" and manage_store_run:
                summary = {"total_pipelines": total_runs}
                if run_predictions.num_predictions > 0:
                    best = run_predictions.get_best(ascending=None)
                    if best:
                        summary["best_score"] = best.get("test_score")
                        summary["best_metric"] = best.get("metric")
                self.store.complete_run(run_id, summary)

        except Exception as e:
            # Fail run in store (only if we manage the lifecycle)
            if run_id and self.mode == "train" and manage_store_run:
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
        best_refit_chains: dict | None = None,
    ) -> None:
        """Execute the refit pass after all CV variants have completed.

        When *best_refit_chains* is populated (accumulated during CV by
        ``BranchController``), each model is refit directly on its best
        preprocessing chain — no store queries or topology dispatch needed.

        Falls back to topology-based dispatch for stacking, separation,
        or when no accumulated chains are available.

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
            best_refit_chains: Accumulated best chains per model from CV.
        """
        from nirs4all.pipeline.analysis.topology import analyze_topology
        from nirs4all.pipeline.config.context import BestChainEntry, RuntimeContext
        from nirs4all.pipeline.execution.refit import execute_simple_refit, extract_winning_config
        from nirs4all.pipeline.execution.refit.config_extractor import RefitConfig, extract_per_model_configs
        from nirs4all.pipeline.execution.refit.executor import RefitResult
        from nirs4all.pipeline.execution.refit.stacking_refit import (
            _find_branch_step,
            execute_competing_branches_refit,
            execute_separation_refit,
            execute_stacking_refit,
        )

        try:
            refit_config = extract_winning_config(
                self.store, run_id, dataset_name=dataset.name,
            )
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
            cache_config=self.cache_config,
            random_state=self.random_state,
        )

        # Fast path: use accumulated best chains when available and topology
        # is not stacking/separation (those have their own refit strategies).
        if (
            best_refit_chains
            and not topology.has_stacking
            and not topology.has_mixed_merge
            and not topology.has_separation_branch
        ):
            refit_result = self._execute_accumulated_refit(
                refit_config=refit_config,
                best_refit_chains=best_refit_chains,
                dataset=dataset,
                runtime_context=runtime_context,
                artifact_registry=artifact_registry,
                executor=executor,
                run_dataset_predictions=run_dataset_predictions,
            )
        elif topology.has_stacking or topology.has_mixed_merge:
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
            # Simple pipeline — check for multi-model variants
            refit_result = self._execute_per_model_refit(
                run_id=run_id,
                refit_config=refit_config,
                dataset=dataset,
                runtime_context=runtime_context,
                artifact_registry=artifact_registry,
                executor=executor,
                run_dataset_predictions=run_dataset_predictions,
            )

        if refit_result.success:
            if refit_result.predictions_count > 0:
                logger.info(
                    f"Refit completed: {refit_result.predictions_count} "
                    f"prediction(s) added with fold_id='final'"
                )

            # Clean up transient CV fold artifacts now that the refit model exists
            winning_pids = [refit_config.pipeline_id]
            if self._per_model_selections:
                # Include all per-model winning pipeline IDs
                per_model_configs = extract_per_model_configs(
                    self.store, run_id, metric=refit_config.metric,
                    dataset_name=dataset.name,
                )
                winning_pids = list({
                    cfg.pipeline_id
                    for _, cfg in per_model_configs.values()
                    if cfg.pipeline_id
                })
                if not winning_pids:
                    winning_pids = [refit_config.pipeline_id]
            try:
                removed = self.store.cleanup_transient_artifacts(
                    run_id=run_id,
                    dataset_name=dataset.name,
                    winning_pipeline_ids=winning_pids,
                )
                if removed > 0:
                    logger.info(f"Cleaned up {removed} transient artifact file(s)")
            except Exception as e:
                logger.warning(f"Transient artifact cleanup failed (non-fatal): {e}")
        else:
            logger.warning("Refit pass did not complete successfully")

    def _execute_accumulated_refit(
        self,
        refit_config: Any,
        best_refit_chains: dict,
        dataset: Any,
        runtime_context: Any,
        artifact_registry: Any,
        executor: Any,
        run_dataset_predictions: Any,
    ) -> Any:
        """Refit each model using accumulated best chains from CV.

        Uses the ``BestChainEntry`` objects accumulated during CV execution
        by ``BranchController``.  Each model is refit on its best
        preprocessing chain — no store queries needed.

        Args:
            refit_config: Global winning RefitConfig (for pre-branch steps).
            best_refit_chains: Accumulated best chains per model from CV.
            dataset: Original dataset for refit.
            runtime_context: Shared runtime context.
            artifact_registry: Artifact registry from CV pass.
            executor: PipelineExecutor instance.
            run_dataset_predictions: Per-dataset prediction accumulator.

        Returns:
            A :class:`RefitResult` summarizing the outcome.
        """
        import copy

        from nirs4all.pipeline.execution.refit import execute_simple_refit
        from nirs4all.pipeline.execution.refit.config_extractor import RefitConfig
        from nirs4all.pipeline.execution.refit.executor import RefitResult
        from nirs4all.pipeline.execution.refit.stacking_refit import _find_branch_step

        # Extract pre-branch steps from the original expanded pipeline
        steps = refit_config.expanded_steps
        branch_info = _find_branch_step(steps)
        if branch_info is not None:
            pre_branch_steps = steps[:branch_info[0]]
        else:
            pre_branch_steps = []

        logger.info(
            f"Accumulated refit: {len(best_refit_chains)} model(s) "
            f"({', '.join(best_refit_chains.keys())})"
        )

        total_predictions = 0
        all_success = True

        for model_name, entry in best_refit_chains.items():
            flat_steps = copy.deepcopy(pre_branch_steps) + copy.deepcopy(entry.branch_steps)
            # Filter out None steps (from _or_ generators expanding to None)
            flat_steps = [s for s in flat_steps if s is not None]

            flat_config = RefitConfig(
                expanded_steps=flat_steps,
                best_params=entry.best_params,
                variant_index=refit_config.variant_index,
                generator_choices=refit_config.generator_choices,
                pipeline_id=refit_config.pipeline_id,
                metric=entry.metric,
                best_score=entry.avg_val_score,
            )

            logger.info(
                f"  Refitting '{model_name}' "
                f"(cv_score={entry.avg_val_score:.4f})"
            )

            result = execute_simple_refit(
                refit_config=flat_config,
                dataset=dataset,
                context=None,
                runtime_context=runtime_context,
                artifact_registry=artifact_registry,
                executor=executor,
                prediction_store=run_dataset_predictions,
            )

            if result.success:
                total_predictions += result.predictions_count
            else:
                all_success = False
                logger.warning(f"  Refit failed for model '{model_name}'")

        return RefitResult(
            success=all_success,
            predictions_count=total_predictions,
            metric=refit_config.metric,
        )

    def _execute_per_model_refit(
        self,
        run_id: str,
        refit_config: Any,
        dataset: Any,
        runtime_context: Any,
        artifact_registry: Any,
        executor: Any,
        run_dataset_predictions: Any,
    ) -> Any:
        """Refit each unique model independently on its best variant.

        If the pipeline has multiple model classes (e.g. from ``_or_``
        generators), each model gets its own refit on the variant where
        it performed best.  Falls back to standard single-model refit
        when only one model class is found.

        Args:
            run_id: Store run identifier.
            refit_config: Global winning RefitConfig (fallback).
            dataset: Original dataset for refit.
            runtime_context: Shared runtime context.
            artifact_registry: Artifact registry from CV pass.
            executor: PipelineExecutor instance.
            run_dataset_predictions: Per-dataset prediction accumulator.

        Returns:
            A :class:`RefitResult` summarizing the outcome.
        """
        from nirs4all.pipeline.execution.refit import execute_simple_refit
        from nirs4all.pipeline.execution.refit.config_extractor import extract_per_model_configs
        from nirs4all.pipeline.execution.refit.executor import RefitResult

        # Try per-model extraction
        per_model_configs = extract_per_model_configs(
            self.store, run_id, metric=refit_config.metric,
            dataset_name=dataset.name,
        )

        if not per_model_configs:
            # Single model or single variant — standard refit
            return execute_simple_refit(
                refit_config=refit_config,
                dataset=dataset,
                context=None,
                runtime_context=runtime_context,
                artifact_registry=artifact_registry,
                executor=executor,
                prediction_store=run_dataset_predictions,
            )

        # Multi-model: refit each model independently
        logger.info(
            f"Per-model refit: {len(per_model_configs)} unique model(s) "
            f"({', '.join(per_model_configs.keys())})"
        )

        total_predictions = 0
        all_success = True
        selections = {}

        for model_name, (selection, model_config) in per_model_configs.items():
            logger.info(
                f"Refitting model '{model_name}' from variant "
                f"{selection.variant_index} (cv_score={selection.best_score:.4f})"
            )
            result = execute_simple_refit(
                refit_config=model_config,
                dataset=dataset,
                context=None,
                runtime_context=runtime_context,
                artifact_registry=artifact_registry,
                executor=executor,
                prediction_store=run_dataset_predictions,
            )
            if result.success:
                total_predictions += result.predictions_count
            else:
                all_success = False
                logger.warning(f"Refit failed for model '{model_name}'")

            selections[model_name] = selection

        # Store per-model selections for RunResult.models
        self._per_model_selections = selections

        return RefitResult(
            success=all_success,
            predictions_count=total_predictions,
            metric=refit_config.metric,
        )

    def _print_best_predictions(
        self,
        run_dataset_predictions: Predictions,
        dataset: SpectroDataset,
        name: str,
    ) -> None:
        """Print best predictions for a dataset.

        Reports best CV predictions and refit final scores to the logger.
        Persistence is handled by :meth:`Predictions.flush` /
        :class:`WorkspaceStore`.
        """
        if run_dataset_predictions.num_predictions > 0:
            # Get aggregation setting from dataset for reporting
            aggregate_column = dataset.aggregate  # Could be None, 'y', or column name
            aggregate_method = dataset.aggregate_method  # Could be None, 'mean', 'median', 'vote'
            aggregate_exclude_outliers = dataset.aggregate_exclude_outliers

            # Check for final (refit) entries
            refit_entries = [
                e for e in run_dataset_predictions._buffer
                if str(e.get("fold_id")) == "final"
            ]

            # Deduplicate: keep one entry per model (prefer "test" partition).
            # The model controller creates separate entries per partition
            # (train/test), but the report should show one row per model.
            if refit_entries:
                seen: dict[tuple, dict] = {}
                for e in refit_entries:
                    key = (e.get("model_name"), e.get("step_idx"), e.get("config_name"))
                    existing = seen.get(key)
                    if existing is None or e.get("partition") == "test":
                        seen[key] = e
                refit_entries = list(seen.values())

            if refit_entries:
                self._print_refit_report(
                    run_dataset_predictions, refit_entries, name,
                    aggregate_column, aggregate_method, aggregate_exclude_outliers,
                )
            else:
                self._print_cv_only_report(
                    run_dataset_predictions, name,
                    aggregate_column, aggregate_method, aggregate_exclude_outliers,
                )

        logger.info("=" * 120)

    def _print_refit_report(
        self,
        predictions: Predictions,
        refit_entries: list,
        name: str,
        aggregate_column: str | None,
        aggregate_method: str | None,
        aggregate_exclude_outliers: bool,
    ) -> None:
        """Print structured report when final (refit) entries exist.

        Headline: Final model performance with best final score.
        Detail: Per-model table (when multiple models), then CV selection summary.
        """
        from nirs4all.data.predictions import _infer_ascending

        metric = refit_entries[0].get("metric", "rmse")
        asc = _infer_ascending(metric)
        rankable = [e for e in refit_entries if e.get("test_score") is not None]
        if not rankable:
            return

        rankable.sort(key=lambda e: e["test_score"], reverse=not asc)
        best_refit = rankable[0]

        # --- Headline: Final model performance ---
        logger.success(
            f"Final model performance for dataset '{name}': "
            f"{Predictions.pred_long_string(best_refit)}"
        )

        if self.enable_tab_reports:
            refit_partitions = predictions.get_entry_partitions(best_refit)
            if aggregate_column:
                agg_label = "y (target values)" if aggregate_column == "y" else f"'{aggregate_column}'"
                method_label = f", method='{aggregate_method}'" if aggregate_method else ""
                outlier_label = ", exclude_outliers=True" if aggregate_exclude_outliers else ""
                logger.info(f"Including aggregated scores (by {agg_label}{method_label}{outlier_label}) in report")
            refit_tab, _ = TabReportManager.generate_best_score_tab_report(
                refit_partitions,
                aggregate=aggregate_column,
                aggregate_method=aggregate_method,
                aggregate_exclude_outliers=aggregate_exclude_outliers,
            )
            logger.info(refit_tab)

            # --- Tab reports for all refitted models ---
            if len(rankable) > 1:
                for rank_idx, entry in enumerate(rankable):
                    if entry is best_refit:
                        continue
                    model_name = entry.get("model_name", "unknown")
                    entry_partitions = predictions.get_entry_partitions(entry)
                    entry_tab, _ = TabReportManager.generate_best_score_tab_report(
                        entry_partitions,
                        aggregate=aggregate_column,
                        aggregate_method=aggregate_method,
                        aggregate_exclude_outliers=aggregate_exclude_outliers,
                    )
                    logger.info(
                        f"Refit scores for #{rank_idx + 1} '{model_name}' "
                        f"({Predictions.pred_long_string(entry)}):\n{entry_tab}"
                    )

        # --- Per-model table when multiple models refit ---
        if len(rankable) > 1:
            summary = TabReportManager.generate_per_model_summary(rankable, ascending=asc, metric=metric)
            logger.info(f"Per-model final scores ({len(rankable)} models):\n{summary}")

        # --- Detail: CV selection summary ---
        cv_best = predictions.get_best(ascending=None, score_scope="cv")
        if cv_best:
            logger.info(
                f"CV selection summary for dataset '{name}': "
                f"{Predictions.pred_long_string(cv_best)}"
            )
            if self.enable_tab_reports:
                cv_partitions = predictions.get_entry_partitions(cv_best)
                cv_tab, _ = TabReportManager.generate_best_score_tab_report(
                    cv_partitions,
                    aggregate=aggregate_column,
                    aggregate_method=aggregate_method,
                    aggregate_exclude_outliers=aggregate_exclude_outliers,
                )
                logger.info(cv_tab)

    def _print_cv_only_report(
        self,
        predictions: Predictions,
        name: str,
        aggregate_column: str | None,
        aggregate_method: str | None,
        aggregate_exclude_outliers: bool,
    ) -> None:
        """Print report when no final (refit) entries exist — CV only."""
        best = predictions.get_best(ascending=None)
        logger.success(f"Best prediction in run for dataset '{name}': {Predictions.pred_long_string(best)}")

        if self.enable_tab_reports:
            best_by_partition = predictions.get_entry_partitions(best)
            if aggregate_column:
                agg_label = "y (target values)" if aggregate_column == "y" else f"'{aggregate_column}'"
                method_label = f", method='{aggregate_method}'" if aggregate_method else ""
                outlier_label = ", exclude_outliers=True" if aggregate_exclude_outliers else ""
                logger.info(f"Including aggregated scores (by {agg_label}{method_label}{outlier_label}) in report")
            tab_report, _ = TabReportManager.generate_best_score_tab_report(
                best_by_partition,
                aggregate=aggregate_column,
                aggregate_method=aggregate_method,
                aggregate_exclude_outliers=aggregate_exclude_outliers,
            )
            logger.info(tab_report)
