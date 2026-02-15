"""Pipeline orchestrator for coordinating multiple pipeline executions."""
from pathlib import Path
from typing import Any

import numpy as np

from nirs4all.core.logging import get_logger
from nirs4all.data.config import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions, _infer_ascending
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
        n_jobs: int = 1,
        report_naming: str = "nirs",
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
            n_jobs: Number of parallel workers for pipeline variants (1=sequential, -1=all cores)
            report_naming: Naming convention for metrics in reports ("nirs", "ml", or "auto")
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
        self.report_naming = report_naming
        self.n_jobs = n_jobs

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
        refit: bool | dict[str, Any] | list[dict[str, Any]] | None = True,
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

                # Deferred artifact persistence: only write artifacts to
                # disk for configs whose validation score beats the
                # previous best.  Activated when there are multiple
                # pipeline configs to compare.
                use_deferred = (
                    len(pipeline_configs.steps) > 1
                    and self.mode == "train"
                    and self.save_artifacts
                )
                best_deferred_val: float | None = None
                best_deferred_ascending: bool | None = None

                # Determine if parallel execution should be used
                use_parallel = (
                    self.n_jobs != 1
                    and len(pipeline_configs.steps) > 1
                    and self.mode == "train"  # Only train mode for now
                )

                # Parallel execution disables deferred artifacts (can't compare across workers)
                if use_parallel:
                    use_deferred = False
                    logger.info(f"Executing {len(pipeline_configs.steps)} variants in parallel (n_jobs={self.n_jobs})")

                # ===================================================================
                # PARALLEL EXECUTION PATH
                # ===================================================================
                if use_parallel:
                    from joblib import Parallel, delayed
                    import multiprocessing

                    # Determine number of workers
                    n_workers = self.n_jobs
                    if n_workers == -1:
                        n_workers = multiprocessing.cpu_count()
                    n_workers = min(n_workers, len(pipeline_configs.steps))

                    # Create a pickle-safe copy of the executor for parallel workers.
                    # PipelineExecutor.store holds a WorkspaceStore with a threading.RLock
                    # which cannot be pickled by loky.  Workers never use executor.store
                    # (they use runtime_context.store which is set to None), so we null it.
                    import copy as _copy
                    parallel_executor = _copy.copy(executor)
                    parallel_executor.store = None

                    # Prepare variant data for parallel execution
                    variant_data_list = []
                    for _i, (steps, config_name, gen_choices) in enumerate(zip(
                        pipeline_configs.steps,
                        pipeline_configs.names,
                        pipeline_configs.generator_choices,
                        strict=False,
                    )):
                        current_run += 1
                        dataset_copy = dataset_configs.get_dataset(config, name)

                        # Capture raw data (first variant only)
                        if self.keep_datasets and name not in self.raw_data and _i == 0:
                            self.raw_data[name] = dataset_copy.x({}, layout="2d")

                        # Initialize context
                        context = executor.initialize_context(dataset_copy)

                        # Create runtime context for parallel workers.
                        # Objects containing threading locks are set to None since they
                        # can't be pickled by loky: store (WorkspaceStore._lock),
                        # step_cache (StepCache._lock), and explainer (holds a reference
                        # chain back to the runner/orchestrator/store).
                        runtime_context_copy = RuntimeContext(
                            store=None,
                            artifact_loader=artifact_loader,
                            artifact_registry=artifact_registry,
                            step_runner=parallel_executor.step_runner,
                            target_model=target_model,
                            explainer=None,
                            run_id=run_id,
                            cache_config=self.cache_config,
                            step_cache=None,
                            best_refit_chains=best_refit_chains,
                            random_state=self.random_state,
                        )

                        variant_data_list.append({
                            "steps": steps,
                            "config_name": config_name,
                            "gen_choices": gen_choices,
                            "dataset": dataset_copy,
                            "executor": parallel_executor,
                            "context": context,
                            "runtime_context": runtime_context_copy,
                            "run_number": current_run,
                            "total_runs": total_runs,
                            "verbose": self.verbose,
                            "keep_datasets": self.keep_datasets,
                        })

                    # Execute variants in parallel using the module-level function
                    # (not a bound method) to avoid pickling the orchestrator.
                    results = Parallel(n_jobs=n_workers, backend='loky', verbose=0)(
                        delayed(_execute_single_variant)(variant_data)
                        for variant_data in variant_data_list
                    )

                    # CRITICAL FIX: Reconstruct store state after parallel execution
                    # In parallel mode, store=None in workers to avoid concurrent writes.
                    # This means pipeline records and predictions were never saved to the
                    # store. We must reconstruct them here so refit pass and tab reports work.

                    # Filter out failed variants
                    failed_variants = [r for r in results if r.get("failed", False)]
                    successful_results = [r for r in results if not r.get("failed", False)]

                    if failed_variants:
                        logger.warning(f"Skipped {len(failed_variants)} variant(s) due to incompatible hyperparameters:")
                        for failed in failed_variants:
                            logger.warning(f"  - {failed.get('config_name', 'unknown')}: {failed.get('failure_reason', 'unknown error')}")

                    logger.info(f"Processing {len(successful_results)} successful parallel execution results ({len(failed_variants)} failed)")

                    for idx, result in enumerate(successful_results):
                        config_predictions = result["predictions"]
                        config_name = result.get("config_name", "unknown")

                        logger.debug(
                            f"Result {idx+1}/{len(successful_results)}: config '{config_name}', "
                            f"{config_predictions.num_predictions} predictions"
                        )

                        # Store reconstruction (non-fatal if it fails).
                        # Skip entirely if the store is in degraded mode (persistent lock).
                        store_available = (
                            config_predictions.num_predictions > 0
                            and self.store
                            and run_id
                            and not self.store.degraded
                        )
                        if store_available:
                            try:
                                # Extract metadata from result
                                steps = result.get("steps", [])
                                gen_choices = result.get("generator_choices", [])
                                dataset_obj = result.get("dataset")

                                # Create pipeline record
                                pipeline_id = self.store.begin_pipeline(
                                    run_id=run_id,
                                    name=config_name,
                                    expanded_config=steps,
                                    generator_choices=gen_choices,
                                    dataset_name=name,
                                    dataset_hash=dataset_obj.content_hash() if dataset_obj else "",
                                )

                                # Sync artifact records from parallel worker
                                for art_record in result.get("artifact_records", []):
                                    self.store.register_existing_artifact(**art_record)

                                # Save chains from parallel worker (must precede
                                # prediction flush due to foreign key constraint)
                                for chain_data in result.get("chain_data_list", []):
                                    self.store.save_chain(pipeline_id=pipeline_id, **chain_data)

                                # Flush predictions using chain-aware resolver
                                executor._flush_predictions_to_store(
                                    self.store,
                                    pipeline_id,
                                    config_predictions,
                                    runtime_context=None,
                                )

                                # Compute metrics for complete_pipeline
                                best = config_predictions.get_best(ascending=None, score_scope="cv")
                                best_val = best.get("val_score") if best else None
                                best_test = best.get("test_score") if best else None
                                metric = best.get("metric", "rmse") if best else "rmse"
                                duration_ms = int(result.get("duration_ms", 0))

                                # Complete pipeline record
                                self.store.complete_pipeline(
                                    pipeline_id=pipeline_id,
                                    best_val=best_val,
                                    best_test=best_test,
                                    metric=metric,
                                    duration_ms=duration_ms,
                                )

                                # Capture last pipeline_uid for syncing
                                self.last_pipeline_uid = pipeline_id

                                logger.debug(f"  -> Store reconstruction completed (pipeline_id={pipeline_id})")
                            except Exception as e:
                                logger.warning(
                                    f"[X] Store reconstruction failed for config '{config_name}': {e}. "
                                    f"Predictions will still be merged in-memory for reporting."
                                )

                        # ALWAYS merge predictions into run-level stores (even if store ops failed)
                        if config_predictions.num_predictions > 0:
                            run_dataset_predictions.merge_predictions(config_predictions)
                            run_predictions.merge_predictions(config_predictions)
                            logger.debug(f"  -> Merged {config_predictions.num_predictions} predictions")

                        # Merge best_refit_chains from this worker
                        worker_refit_chains = result.get("best_refit_chains")
                        if worker_refit_chains and best_refit_chains is not None:
                            self._merge_refit_chains(best_refit_chains, worker_refit_chains)

                        # Capture execution trace
                        if result.get("execution_trace"):
                            self.last_execution_trace = result["execution_trace"]

                        # Capture preprocessed data
                        if self.keep_datasets and "preprocessed_data" in result:
                            if name not in self.pp_data:
                                self.pp_data[name] = {}
                            snapshot_key = result["preprocessing_key"]
                            self.pp_data[name][snapshot_key] = result["preprocessed_data"]

                    logger.info(
                        f"Parallel results processed: {run_dataset_predictions.num_predictions} "
                        f"total predictions merged"
                    )

                # ===================================================================
                # SEQUENTIAL EXECUTION PATH (original code)
                # ===================================================================
                else:
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

                        # Enter deferred mode before execution
                        if use_deferred and artifact_registry is not None:
                            artifact_registry.begin_deferred()

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
                            # Rollback deferred artifacts before cleanup
                            if use_deferred and artifact_registry is not None and artifact_registry._deferred_mode:
                                artifact_registry.rollback_deferred()
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

                        # Deferred artifact persistence: commit if this config's
                        # validation score beats the previous best, rollback otherwise.
                        if use_deferred and artifact_registry is not None and artifact_registry._deferred_mode:
                            should_commit = True
                            if config_predictions.num_predictions > 0:
                                config_best = config_predictions.get_best(ascending=None)
                                if config_best:
                                    val_score = config_best.get("val_score")
                                    if val_score is not None:
                                        # Has folds — compare with best seen so far
                                        if best_deferred_val is None:
                                            best_deferred_val = val_score
                                            best_deferred_ascending = _infer_ascending(
                                                config_best.get("metric", "rmse")
                                            )
                                        else:
                                            if best_deferred_ascending:
                                                is_better = val_score < best_deferred_val
                                            else:
                                                is_better = val_score > best_deferred_val
                                            if is_better:
                                                best_deferred_val = val_score
                                            else:
                                                should_commit = False
                                    # val_score is None → no folds → always commit
                            if should_commit:
                                artifact_registry.commit_deferred()
                            else:
                                artifact_registry.rollback_deferred()

                        # Merge new predictions into run-level stores
                        if config_predictions.num_predictions > 0:
                            run_dataset_predictions.merge_predictions(config_predictions)
                            run_predictions.merge_predictions(config_predictions)

                # --- Pass 2: Refit (optional) ---
                # After all variants have been executed for this dataset,
                # retrain the winning model on the full training set.
                refit_enabled = refit is True or (isinstance(refit, dict) and refit) or (isinstance(refit, list) and refit)
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
                        refit=refit,
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

                # In parallel mode, 'dataset' still holds the raw input parameter;
                # load a SpectroDataset for metadata access in common post-execution code.
                if use_parallel:
                    dataset = dataset_configs.get_dataset(config, name)

                # Store last aggregate column for visualization integration
                self.last_aggregate_column = dataset.aggregate
                self.last_aggregate_method = dataset.aggregate_method
                self.last_aggregate_exclude_outliers = dataset.aggregate_exclude_outliers

                # Print best results for this dataset
                logger.info(
                    f"Preparing to print results for dataset '{name}': "
                    f"{run_dataset_predictions.num_predictions} predictions"
                )
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

                # Cleanup between datasets to release memory (GPU, step cache, GC)
                if n_datasets > 1:
                    self._cleanup_between_datasets(step_cache, name)

            # Print global summary of all final models across all datasets
            if self.mode == "train":
                from nirs4all.visualization.reports import TabReportManager
                pred_index = TabReportManager._build_prediction_index(run_predictions)

                self._print_global_final_summary(
                    run_predictions,
                    datasets_predictions,
                    pred_index=pred_index,
                )

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
            # Fail run in store (only if we manage the lifecycle).
            # Use try/except to prevent store errors from masking the
            # original pipeline error (e.g. DuckDB lock conflicts).
            if run_id and self.mode == "train" and manage_store_run:
                try:
                    self.store.fail_run(run_id, str(e))
                except Exception:
                    pass  # fail_run already logs internally
            raise

        return run_predictions, datasets_predictions

    def _cleanup_between_datasets(self, step_cache: Any, dataset_name: str) -> None:
        """Release memory between dataset iterations.

        Clears the step cache (entries from the previous dataset are not
        reusable), releases GPU memory held by models like TabPFN, and
        runs garbage collection.

        Args:
            step_cache: StepCache instance (or None).
            dataset_name: Name of the dataset that just completed (for logging).
        """
        import gc

        # Clear step cache -- entries are keyed by content hash so entries
        # from the previous dataset will never hit, but still consume memory.
        if step_cache is not None:
            step_cache.clear()

        # Release GPU memory (PyTorch/TensorFlow/JAX).
        # This is critical for models like TabPFN that allocate CUDA tensors
        # and don't release them until the Python objects are garbage-collected.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except ImportError:
            pass

        # Force garbage collection to release model objects and their tensors.
        gc.collect()

        logger.debug(f"Cleanup completed after dataset '{dataset_name}'")

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
        refit: bool | dict[str, Any] | list[dict[str, Any]] | None = True,
    ) -> None:
        """Execute the refit pass after all CV variants have completed.

        When *best_refit_chains* is populated (accumulated during CV by
        ``BranchController``), each model is refit directly on its best
        preprocessing chain — no store queries or topology dispatch needed.

        Falls back to topology-based dispatch for stacking, separation,
        or when no accumulated chains are available.

        Supports multi-config refit when ``refit`` is a dict or list of
        dicts specifying :class:`RefitCriterion` options (top_k, ranking).

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
            refit: Refit configuration from the user API.
        """
        from nirs4all.pipeline.analysis.topology import analyze_topology
        from nirs4all.pipeline.config.context import BestChainEntry, RuntimeContext
        from nirs4all.pipeline.execution.refit import execute_simple_refit, extract_winning_config
        from nirs4all.pipeline.execution.refit.config_extractor import RefitConfig, extract_per_model_configs, extract_top_configs, parse_refit_param
        from nirs4all.pipeline.execution.refit.executor import RefitResult
        from nirs4all.pipeline.execution.refit.stacking_refit import (
            _find_branch_step,
            execute_competing_branches_refit,
            execute_separation_refit,
            execute_stacking_refit,
        )

        # Parse refit criteria
        criteria = parse_refit_param(refit)

        # Determine if we have multi-config criteria (non-default selection)
        is_multi_config = len(criteria) > 1 or (len(criteria) == 1 and (criteria[0].top_k > 1 or criteria[0].ranking != "rmsecv"))

        # Always show refit mode for transparency
        if criteria:
            print(f"\n{'=' * 80}")
            print(f"REFIT MODE: {'Multi-criteria' if is_multi_config else 'Single criterion'}")
            print(f"Criteria: {len(criteria)} criterion/criteria")
            print(f"{'=' * 80}\n", flush=True)

        # Snapshot buffer size before refit so we can sync new entries
        # to the global run_predictions after refit completes (P2.3).
        pre_refit_count = run_dataset_predictions.num_predictions

        if is_multi_config:
            # Multi-config refit: extract top configs based on criteria
            # Log each criterion's selection for transparency
            logger.info("=" * 80)
            logger.info("MULTI-CRITERIA REFIT SELECTION")
            logger.info("=" * 80)

            for crit_idx, criterion in enumerate(criteria, 1):
                logger.info(
                    f"Criterion #{crit_idx}: ranking={criterion.ranking}, "
                    f"top_k={criterion.top_k}, metric={criterion.metric or 'default'}"
                )

            try:
                refit_configs = extract_top_configs(
                    self.store, run_id, criteria,
                    predictions=run_dataset_predictions,
                    dataset_name=dataset.name,
                )
            except ValueError as e:
                logger.warning(f"Cannot perform refit: {e}")
                return

            total_selections = sum(c.top_k for c in criteria)
            num_duplicates = total_selections - len(refit_configs)
            logger.info(
                f"Selection result: {len(refit_configs)} unique config(s) "
                f"(from {total_selections} total selections, {num_duplicates} duplicates)"
            )
            logger.info("-" * 80)

            all_winning_pids: list[str] = []
            total_predictions = 0
            any_success = False

            for config_idx, refit_config in enumerate(refit_configs, 1):
                criteria_str = ", ".join(refit_config.selected_by_criteria) if refit_config.selected_by_criteria else "unknown"
                logger.info(
                    f"Refitting #{config_idx}/{len(refit_configs)}: "
                    f"'{refit_config.config_name}' [{criteria_str}] (best_val={refit_config.selection_score:.4f})"
                )
                refit_result = self._execute_single_refit(
                    refit_config=refit_config,
                    run_id=run_id,
                    dataset=dataset,
                    executor=executor,
                    artifact_registry=artifact_registry,
                    run_dataset_predictions=run_dataset_predictions,
                    artifact_loader=artifact_loader,
                    target_model=target_model,
                    explainer=explainer,
                    best_refit_chains=best_refit_chains,
                )
                if refit_result.success:
                    any_success = True
                    total_predictions += refit_result.predictions_count
                    if refit_config.pipeline_id:
                        all_winning_pids.append(refit_config.pipeline_id)
                    logger.info(f"  ✓ Refit #{config_idx} completed successfully")
                else:
                    logger.warning(f"  ✗ Refit #{config_idx} failed")

            logger.info("=" * 80)
            if any_success and total_predictions > 0:
                logger.info(
                    f"REFIT SUMMARY: {total_predictions} prediction(s) added "
                    f"across {len(refit_configs)} config(s)"
                )
            logger.info("=" * 80)

            # Clean up transient artifacts for all winning pipelines
            if all_winning_pids:
                try:
                    removed = self.store.cleanup_transient_artifacts(
                        run_id=run_id,
                        dataset_name=dataset.name,
                        winning_pipeline_ids=all_winning_pids,
                    )
                    if removed > 0:
                        logger.info(f"Cleaned up {removed} transient artifact file(s)")
                except Exception as e:
                    logger.warning(f"Transient artifact cleanup failed (non-fatal): {e}")

            if not any_success:
                logger.warning("Multi-config refit pass did not complete successfully")

            # Sync refit predictions to global buffer
            self._sync_refit_to_global(run_dataset_predictions, run_predictions, pre_refit_count)
            return

        # Single-config refit (default path: refit=True)
        try:
            refit_config = extract_winning_config(
                self.store, run_id, dataset_name=dataset.name,
            )
        except ValueError as e:
            logger.warning(f"Cannot perform refit: {e}")
            return

        refit_result = self._execute_single_refit(
            refit_config=refit_config,
            run_id=run_id,
            dataset=dataset,
            executor=executor,
            artifact_registry=artifact_registry,
            run_dataset_predictions=run_dataset_predictions,
            artifact_loader=artifact_loader,
            target_model=target_model,
            explainer=explainer,
            best_refit_chains=best_refit_chains,
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

        # Sync refit predictions to global buffer
        self._sync_refit_to_global(run_dataset_predictions, run_predictions, pre_refit_count)

    def _sync_refit_to_global(
        self,
        run_dataset_predictions: Predictions,
        run_predictions: Predictions,
        pre_refit_count: int,
    ) -> None:
        """Sync refit predictions from per-dataset to global buffer.

        After refit, new predictions (fold_id='final') are in
        ``run_dataset_predictions`` but not in ``run_predictions``.
        This copies the newly added entries to the global buffer.
        """
        new_count = run_dataset_predictions.num_predictions - pre_refit_count
        if new_count > 0:
            refit_preds = Predictions()
            refit_preds._buffer = run_dataset_predictions._buffer[pre_refit_count:]
            run_predictions.merge_predictions(refit_preds)
            logger.debug(f"Synced {new_count} refit prediction(s) to global buffer")

    def _execute_single_refit(
        self,
        refit_config: Any,
        run_id: str,
        dataset: SpectroDataset,
        executor: Any,
        artifact_registry: ArtifactRegistry | None,
        run_dataset_predictions: Predictions,
        artifact_loader: Any = None,
        target_model: dict[str, Any] | None = None,
        explainer: Any = None,
        best_refit_chains: dict | None = None,
    ) -> Any:
        """Execute refit for a single config, dispatching by topology.

        Args:
            refit_config: :class:`RefitConfig` for the pipeline variant.
            run_id: Store run identifier.
            dataset: Original dataset (will be deep-copied internally).
            executor: PipelineExecutor instance.
            artifact_registry: Shared artifact registry.
            run_dataset_predictions: Per-dataset prediction accumulator.
            artifact_loader: Optional artifact loader.
            target_model: Optional target model dict.
            explainer: Optional explainer instance.
            best_refit_chains: Accumulated best chains per model from CV.

        Returns:
            A :class:`RefitResult` summarizing the outcome.
        """
        from nirs4all.pipeline.analysis.topology import analyze_topology
        from nirs4all.pipeline.config.context import RuntimeContext
        from nirs4all.pipeline.execution.refit.executor import RefitResult
        from nirs4all.pipeline.execution.refit.stacking_refit import (
            execute_competing_branches_refit,
            execute_separation_refit,
            execute_stacking_refit,
        )

        topology = analyze_topology(refit_config.expanded_steps)

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
            return self._execute_accumulated_refit(
                refit_config=refit_config,
                best_refit_chains=best_refit_chains,
                dataset=dataset,
                runtime_context=runtime_context,
                artifact_registry=artifact_registry,
                executor=executor,
                run_dataset_predictions=run_dataset_predictions,
            )
        elif topology.has_stacking or topology.has_mixed_merge:
            return execute_stacking_refit(
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
            return execute_competing_branches_refit(
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
            return execute_separation_refit(
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
            return self._execute_per_model_refit(
                run_id=run_id,
                refit_config=refit_config,
                dataset=dataset,
                runtime_context=runtime_context,
                artifact_registry=artifact_registry,
                executor=executor,
                run_dataset_predictions=run_dataset_predictions,
            )

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
                config_name=refit_config.config_name,
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
        if run_dataset_predictions.num_predictions == 0:
            logger.warning(
                f"No predictions to report for dataset '{name}'. "
                f"This usually indicates a problem during pipeline execution."
            )
            return

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
                from nirs4all.visualization.reports import TabReportManager
                dataset_pred_index = TabReportManager._build_prediction_index(run_dataset_predictions)
                metric = refit_entries[0].get("metric", "rmse")
                TabReportManager.enrich_refit_entries(refit_entries, dataset_pred_index, metric)

                self._print_refit_report(
                    run_dataset_predictions, refit_entries, name,
                    aggregate_column, aggregate_method, aggregate_exclude_outliers,
                    pred_index=dataset_pred_index,
                )
            else:
                self._print_cv_only_report(
                    run_dataset_predictions, name,
                    aggregate_column, aggregate_method, aggregate_exclude_outliers,
                )

        logger.info("=" * 120)

    @staticmethod
    def _get_entry_partitions_indexed(
        predictions: Predictions,
        entry: dict,
        pred_index: dict | None,
    ) -> dict[str, dict]:
        """Get train/val/test partitions for an entry using the index when available."""
        if pred_index is not None:
            key = (
                entry.get("dataset_name"),
                entry.get("config_name", ""),
                entry.get("model_name"),
                entry.get("fold_id", ""),
                entry.get("step_idx", 0),
            )
            result = pred_index["partitions"].get(key)
            if result is not None:
                return result
        return predictions.get_entry_partitions(entry)

    def _print_refit_report(
        self,
        predictions: Predictions,
        refit_entries: list,
        name: str,
        aggregate_column: str | None,
        aggregate_method: str | None,
        aggregate_exclude_outliers: bool,
        pred_index: dict | None = None,
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

        # Enrich each refit entry with the CV test score from its w_avg fold.
        # This is the test score computed from the weighted-average of fold
        # predictions (before refit) — useful for comparing CV vs refit performance.
        if pred_index is not None:
            w_avg_index = pred_index.get("w_avg", {})
            for entry in rankable:
                w_avg_key = (entry["dataset_name"], entry.get("config_name"), entry["model_name"], entry.get("step_idx", 0))
                w_avg_parts = w_avg_index.get(w_avg_key, {})
                test_entry = w_avg_parts.get("test")
                if test_entry is not None:
                    entry["cv_test_score"] = test_entry.get("test_score")
        else:
            for entry in rankable:
                w_avg_matches = predictions.filter_predictions(
                    dataset_name=entry["dataset_name"],
                    config_name=entry.get("config_name"),
                    model_name=entry["model_name"],
                    step_idx=entry.get("step_idx"),
                    fold_id="w_avg",
                    partition="test",
                    load_arrays=False,
                )
                if w_avg_matches:
                    entry["cv_test_score"] = w_avg_matches[0].get("test_score")

        best_refit = rankable[0]

        # --- Headline: Final model performance ---
        cv_test_str = ""
        if best_refit.get("cv_test_score") is not None:
            cv_test_str = f", [cv_test: {best_refit['cv_test_score']:.4f}]"
        final_msg = (
            f"Final model performance for dataset '{name}': "
            f"{Predictions.pred_long_string(best_refit)}{cv_test_str}"
        )
        # Always show best model (even with verbose=0)
        print(f"\n{final_msg}")
        # Also log for file output if enabled
        if self.verbose > 0:
            logger.success(final_msg)

        if self.enable_tab_reports:
            refit_partitions = self._get_entry_partitions_indexed(predictions, best_refit, pred_index)
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
                    entry_partitions = self._get_entry_partitions_indexed(predictions, entry, pred_index)
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

        # --- Per-model table for this dataset (always show, even with 1 model) ---
        if len(rankable) > 0:
            summary = TabReportManager.generate_per_model_summary(
                rankable, ascending=asc, metric=metric,
                aggregate=aggregate_column,
                aggregate_method=aggregate_method,
                aggregate_exclude_outliers=aggregate_exclude_outliers,
                predictions=predictions,
                report_naming=self.report_naming,
                pred_index=pred_index,
                verbose=self.verbose,
            )
            model_word = "model" if len(rankable) == 1 else "models"

            # Detect multi-criteria refit from pipeline names
            refit_suffix = self._detect_refit_criteria(rankable)
            header = f"Dataset '{name}' - Final scores ({len(rankable)} {model_word})"
            if refit_suffix:
                header += f" - {refit_suffix}"

            print(f"\n{header}:\n{summary}", flush=True)

        # --- Top 30 CV chains (averaged across folds) ---
        avg_entries = predictions.top(
            n=30,
            ascending=asc,
            score_scope="cv",
            rank_partition="val",
            fold_id="avg",
        )

        if avg_entries:
            import numpy as np
            from nirs4all.visualization.naming import get_metric_names

            # Determine task type and metric for naming
            first_entry = avg_entries[0]
            cv_task_type = first_entry.get("task_type", "regression")
            cv_metric = first_entry.get("metric", "rmse")
            cv_task_str = "regression" if cv_task_type == "regression" else "classification"
            names = get_metric_names(self.report_naming, cv_task_str, cv_metric)

            rows: list[dict] = []
            all_fold_ids: set[str] = set()
            for entry in avg_entries:
                e_model = entry.get("model_name", "unknown")
                e_config = entry.get("config_name", "")
                e_step = entry.get("step_idx", 0)
                e_preproc = entry.get("preprocessings", "")

                # Build display chain including y_processing when present
                target_proc = entry.get("target_processing", "")
                y_label = ""
                if target_proc and target_proc not in ("numeric", "raw", ""):
                    # Extract class name(s) from processing chain like "numeric_StandardScaler42"
                    import re
                    parts = re.findall(r'_([A-Z][A-Za-z]+)\d+', target_proc)
                    if parts:
                        y_label = "y:" + "|".join(parts)
                chain_display = f"{y_label}>{e_preproc}" if y_label else e_preproc

                # Look up sibling entries (w_avg + individual folds)
                siblings = predictions.filter_predictions(
                    model_name=e_model, config_name=e_config, step_idx=e_step,
                    preprocessings=e_preproc, partition="val", load_arrays=False,
                )

                w_ens_test: float | None = None
                fold_scores: dict[str, float | None] = {}
                for sib in siblings:
                    fid = str(sib.get("fold_id", ""))
                    if fid == "w_avg":
                        w_ens_test = sib.get("test_score")
                    elif fid not in ("avg", "w_avg", "final") and fid:
                        fold_scores[fid] = sib.get("val_score")
                        all_fold_ids.add(fid)

                # Compute MF_Val: mean of per-fold val scores
                valid_vals = [v for v in fold_scores.values() if v is not None]
                mf_val = float(np.mean(valid_vals)) if valid_vals else None

                rows.append({
                    "model": e_model,
                    "rmsecv": entry.get("val_score"),
                    "mf_val": mf_val,
                    "ens_test": entry.get("test_score"),
                    "w_ens_test": w_ens_test,
                    "folds": fold_scores,
                    "chain": chain_display,
                })

            sorted_fold_ids = sorted(all_fold_ids)

            # Build header with proper naming convention
            cv_col = names["cv_score"]
            mfv_col = names["mean_fold_cv"]
            ens_col = names["ens_test"]
            wens_col = names["w_ens_test"]

            fold_cols = "".join(f" | {'f' + fid:>7s}" for fid in sorted_fold_ids)
            header = (
                f"| {'#':>3s} | {'Model':10s} | {cv_col:>9s} | {mfv_col:>9s}"
                f" | {ens_col:>9s} | {wens_col:>10s}{fold_cols} | {'Chain':<60s} |"
            )
            sep = "-" * len(header)

            print(f"\nTop 30 CV chains (ranked by {cv_col}) for dataset '{name}':")
            print(sep)
            print(header)
            print(sep)

            def _fmt(v: float | None) -> str:
                return f"{v:.4f}" if v is not None else "N/A"

            for idx, row in enumerate(rows, 1):
                fold_vals = "".join(f" | {_fmt(row['folds'].get(fid)):>7s}" for fid in sorted_fold_ids)
                print(
                    f"| {idx:3d} | {row['model'][:10]:10s} | {_fmt(row['rmsecv']):>9s} | {_fmt(row['mf_val']):>9s}"
                    f" | {_fmt(row['ens_test']):>9s} | {_fmt(row['w_ens_test']):>10s}{fold_vals} | {row['chain'][:60]:<60s} |"
                )
            print(sep)
            print()

        # --- Detail: CV selection summary (always visible) ---
        cv_best = predictions.get_best(ascending=None, score_scope="cv")
        if cv_best:
            cv_summary_msg = (
                f"CV selection summary for dataset '{name}': "
                f"{Predictions.pred_long_string(cv_best)}"
            )
            print(f"\n{cv_summary_msg}")
            if self.verbose > 0:
                logger.success(cv_summary_msg)
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

    def _detect_refit_criteria(self, entries: list[dict]) -> str:
        """Detect and format multi-criteria refit info from pipeline names.

        Args:
            entries: List of prediction entries (refit models)

        Returns:
            Formatted string like "Multi-criteria refit [rmsecv(top3), mean_val(top3)]"
            or empty string if not multi-criteria refit.
        """
        if not entries:
            return ""

        # Extract unique criterion sets from pipeline names
        all_criteria: set[str] = set()
        for entry in entries:
            config_name = entry.get("config_name", "")
            # Check if this is a multi-criteria refit (has both rmsecv and mean_val in name)
            if "_refit_" in config_name:
                # Extract the suffix after _refit_
                parts = config_name.split("_refit_")
                if len(parts) > 1:
                    suffix = parts[1]
                    # Parse criterion labels like "rmsecvt3_mean_valt3"
                    # Convert back to readable format
                    criteria = []
                    if "rmsecvt" in suffix:
                        import re
                        match = re.search(r"rmsecvt(\d+)", suffix)
                        if match:
                            criteria.append(f"rmsecv(top{match.group(1)})")
                    if "mean_valt" in suffix:
                        import re
                        match = re.search(r"mean_valt(\d+)", suffix)
                        if match:
                            criteria.append(f"mean_val(top{match.group(1)})")

                    if criteria:
                        all_criteria.update(criteria)

        if len(all_criteria) > 1:
            sorted_criteria = sorted(all_criteria)
            return f"Multi-criteria refit [{', '.join(sorted_criteria)}]"
        return ""

    def _merge_refit_chains(
        self,
        target_chains: dict,
        source_chains: dict,
    ) -> None:
        """Merge refit chains from a parallel worker into the main accumulator.

        For each model in source_chains, compare its avg_val_score with the
        existing entry in target_chains (if any). Keep the better one.

        Args:
            target_chains: Main accumulator dict (model_name -> BestChainEntry)
            source_chains: Worker's refit chains to merge in
        """
        from nirs4all.data.predictions import _infer_ascending

        for model_name, source_entry in source_chains.items():
            existing_entry = target_chains.get(model_name)

            if existing_entry is None:
                # No existing entry - use worker's entry
                target_chains[model_name] = source_entry
                continue

            # Compare scores - keep the better one
            metric = source_entry.metric
            ascending = _infer_ascending(metric)

            source_score = source_entry.avg_val_score
            existing_score = existing_entry.avg_val_score

            is_better = (
                (ascending and source_score < existing_score)
                or (not ascending and source_score > existing_score)
            )

            if is_better:
                target_chains[model_name] = source_entry

    def _print_global_final_summary(
        self,
        run_predictions: Predictions,
        datasets_predictions: dict,
        pred_index: dict | None = None,
    ) -> None:
        """Print a global summary table of ALL final (refit) models across all datasets.

        This shows a unified view of all models that were refit, regardless of which dataset
        they belong to. Always displayed (even with verbose=0).

        Args:
            run_predictions: Global Predictions object with all predictions.
            datasets_predictions: Dict mapping dataset names to their prediction info.
        """
        from nirs4all.visualization.reports import TabReportManager

        # Collect all final (refit) entries across all datasets
        all_refit_entries = run_predictions.filter_predictions(
            fold_id="final",
            load_arrays=False
        )

        if not all_refit_entries:
            return  # No refit entries to show

        # Filter to entries with valid test scores
        rankable = [e for e in all_refit_entries if e.get("test_score") is not None]
        if not rankable:
            return

        # Deduplicate: keep one entry per (model, step, config) - prefer "test" partition
        seen: dict[tuple, dict] = {}
        for e in rankable:
            key = (e.get("dataset_name"), e.get("model_name"), e.get("step_idx"), e.get("config_name"))
            existing = seen.get(key)
            if existing is None or e.get("partition") == "test":
                seen[key] = e
        rankable = list(seen.values())
        if not rankable:
            return

        # Enrich each entry with cv_test_score from w_avg fold (if available)
        if pred_index is not None:
            w_avg_index = pred_index.get("w_avg", {})
            for entry in rankable:
                w_avg_key = (entry["dataset_name"], entry.get("config_name"), entry["model_name"], entry.get("step_idx", 0))
                w_avg_parts = w_avg_index.get(w_avg_key, {})
                test_entry = w_avg_parts.get("test")
                if test_entry is not None:
                    entry["cv_test_score"] = test_entry.get("test_score")
        else:
            for entry in rankable:
                w_avg_matches = run_predictions.filter_predictions(
                    dataset_name=entry["dataset_name"],
                    config_name=entry.get("config_name"),
                    model_name=entry["model_name"],
                    step_idx=entry.get("step_idx"),
                    fold_id="w_avg",
                    partition="test",
                    load_arrays=False,
                )
                if w_avg_matches:
                    entry["cv_test_score"] = w_avg_matches[0].get("test_score")

        # Get metric and aggregation settings from first entry
        metric = rankable[0].get("metric", "rmse")

        # Enrich refit entries with CV metrics (RMSECV, mean/wmean fold test)
        if pred_index is not None:
            TabReportManager.enrich_refit_entries(rankable, pred_index, metric)

        # Try to get aggregate settings from last dataset processed
        aggregate_column = self.last_aggregate_column
        aggregate_method = self.last_aggregate_method
        aggregate_exclude_outliers = self.last_aggregate_exclude_outliers

        # Generate the global summary table (using pre-built index for efficiency)
        summary = TabReportManager.generate_per_model_summary(
            rankable,
            ascending=True,  # Will be adjusted based on metric
            metric=metric,
            aggregate=aggregate_column,
            aggregate_method=aggregate_method,
            report_naming=self.report_naming,
            aggregate_exclude_outliers=aggregate_exclude_outliers,
            predictions=run_predictions,
            pred_index=pred_index,
            verbose=self.verbose,
        )

        if summary:
            # Always print (even with verbose=0)
            print(f"\n{'=' * 120}")
            print(f"GLOBAL SUMMARY: All final models ({len(rankable)} models across {len(datasets_predictions)} dataset(s))")
            print(f"{'=' * 120}")
            print(summary)
            print(f"{'=' * 120}\n")

    # _execute_single_variant is a module-level function (see below)
    # to avoid pickling the orchestrator (which contains WorkspaceStore with
    # threading.RLock) when dispatched to loky workers.


def _execute_single_variant(variant_data: dict[str, Any]) -> dict[str, Any]:
    """Execute a single pipeline variant in a parallel worker.

    This is a module-level function (not a method) so that joblib/loky only
    pickles the variant_data dict — not the PipelineOrchestrator instance
    which contains a WorkspaceStore with an unpicklable threading.RLock.

    Args:
        variant_data: Dictionary containing all data needed for variant execution.

    Returns:
        Dictionary with predictions, traces, and metadata for post-processing.
    """
    import copy
    import time

    steps = variant_data["steps"]
    config_name = variant_data["config_name"]
    gen_choices = variant_data["gen_choices"]
    dataset = variant_data["dataset"]
    executor = variant_data["executor"]
    context = variant_data["context"]
    runtime_context = variant_data["runtime_context"]
    run_number = variant_data["run_number"]
    total_runs = variant_data["total_runs"]
    verbose = variant_data["verbose"]
    keep_datasets = variant_data["keep_datasets"]

    # Deep copy mutable objects to avoid cross-worker interference
    dataset = copy.deepcopy(dataset)
    context = copy.deepcopy(context)
    runtime_context = copy.deepcopy(runtime_context)

    # Disable store operations in parallel workers to avoid concurrent writes
    if runtime_context:
        runtime_context.store = None

    logger.info(f"Run {run_number}/{total_runs}: pipeline '{config_name}' on dataset")

    if verbose > 0:
        print(dataset)

    # Create prediction store for this variant
    config_predictions = Predictions()
    if dataset.repetition:
        config_predictions.set_repetition_column(dataset.repetition)

    # Track execution time
    start_time = time.monotonic()

    # Execute variant
    variant_failed = False
    failure_reason = None
    try:
        executor.execute(
            steps=steps,
            config_name=config_name,
            dataset=dataset,
            context=context,
            runtime_context=runtime_context,
            prediction_store=config_predictions,
            generator_choices=gen_choices,
        )
        logger.info(
            f"Run {run_number}/{total_runs} completed: '{config_name}', "
            f"{config_predictions.num_predictions} predictions generated"
        )
    except Exception as e:
        error_msg = str(e)
        # Check if this is a recoverable error (invalid hyperparameters for this dataset)
        if 'n_components' in error_msg or 'upper bound' in error_msg or 'Invalid hyperparameters' in error_msg:
            logger.warning(f"Variant '{config_name}' skipped due to incompatible hyperparameters: {error_msg}")
            variant_failed = True
            failure_reason = error_msg
            # Don't raise - mark as failed and continue
        else:
            # Unknown error - log and re-raise
            logger.error(f"Variant '{config_name}' failed with unexpected error: {e}")
            import traceback
            traceback.print_exc()
            raise

    # Compute duration
    duration_ms = int((time.monotonic() - start_time) * 1000)

    # Build chain data for store reconstruction.
    # In parallel workers store=None, so executor.execute() skips chain
    # building and artifact syncing.  We do it here so the orchestrator
    # can reconstruct store state after collecting results.
    chain_data_list: list[dict[str, Any]] = []
    artifact_records: list[dict[str, Any]] = []

    tr = getattr(runtime_context, 'trace_recorder', None) if runtime_context else None
    if tr is not None:
        from nirs4all.pipeline.storage.chain_builder import ChainBuilder
        trace = tr.finalize(
            preprocessing_chain=dataset.short_preprocessings_str(),
            metadata={"n_steps": len(steps)}
        )
        chain_builder = ChainBuilder(trace, executor.artifact_registry)
        chain_data_list = chain_builder.build_all()

    if executor.artifact_registry is not None:
        for record in executor.artifact_registry.get_all_records():
            artifact_records.append({
                "artifact_id": record.artifact_id,
                "path": record.path,
                "content_hash": record.content_hash,
                "operator_class": record.class_name,
                "artifact_type": record.artifact_type.value,
                "format": record.format,
                "size_bytes": record.size_bytes,
            })

    # Capture results
    result = {
        "predictions": config_predictions,
        "pipeline_uid": runtime_context.pipeline_uid if runtime_context else None,
        "execution_trace": runtime_context.get_execution_trace() if runtime_context else None,
        "config_name": config_name,
        "best_refit_chains": runtime_context.best_refit_chains if runtime_context else None,
        "steps": steps,
        "generator_choices": gen_choices,
        "dataset": dataset,
        "duration_ms": duration_ms,
        "chain_data_list": chain_data_list,
        "artifact_records": artifact_records,
        "failed": variant_failed,
        "failure_reason": failure_reason,
    }

    # Capture preprocessed data snapshot if requested
    if keep_datasets:
        result["preprocessed_data"] = dataset.x({}, layout="2d")
        result["preprocessing_key"] = dataset.short_preprocessings_str()

    logger.debug(f"Worker returning result: {len(result['predictions']._buffer)} predictions in buffer")

    return result
