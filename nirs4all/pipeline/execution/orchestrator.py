"""Pipeline orchestrator for coordinating multiple pipeline executions."""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from nirs4all.data.config import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.execution.builder import ExecutorBuilder
from nirs4all.pipeline.execution.executor import PipelineExecutor
from nirs4all.pipeline.storage.io import SimulationSaver
from nirs4all.pipeline.storage.manifest_manager import ManifestManager
from nirs4all.core.logging import get_logger
from nirs4all.visualization.reports import TabReportManager

logger = get_logger(__name__)


def _get_default_workspace_path() -> Path:
    """Get the default workspace path.

    Checks NIRS4ALL_WORKSPACE environment variable first, then falls back
    to ./workspace in the current working directory.

    Returns:
        Default workspace path.
    """
    env_workspace = os.environ.get("NIRS4ALL_WORKSPACE")
    if env_workspace:
        return Path(env_workspace)
    return Path.cwd() / "workspace"


class PipelineOrchestrator:
    """Orchestrates execution of multiple pipelines across multiple datasets.

    High-level coordinator that manages:
    - Workspace initialization
    - Global predictions aggregation
    - Best results reporting
    - Dataset/pipeline normalization

    Attributes:
        workspace_path: Root workspace directory
        runs_dir: Directory for storing runs
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
        workspace_path: Optional[Union[str, Path]] = None,
        verbose: int = 0,
        mode: str = "train",
        save_artifacts: bool = True,
        save_charts: bool = True,
        enable_tab_reports: bool = True,
        continue_on_error: bool = False,
        show_spinner: bool = True,
        keep_datasets: bool = True,
        plots_visible: bool = False
    ):
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
        self.runs_dir = self.workspace_path / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        # Create other workspace directories
        (self.workspace_path / "exports").mkdir(exist_ok=True)
        (self.workspace_path / "library").mkdir(exist_ok=True)

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
        self.raw_data: Dict[str, np.ndarray] = {}
        self.pp_data: Dict[str, Dict[str, np.ndarray]] = {}

        # Figure references to prevent garbage collection
        self._figure_refs: List[Any] = []

        # Store last executed pipeline info for post-run operations and syncing
        self.last_saver: Any = None
        self.last_pipeline_uid: Optional[str] = None
        self.last_manifest_manager: Any = None
        self.last_executor: Any = None  # For syncing step_number, substep_number, operation_count
        self.last_aggregate_column: Optional[str] = None  # Last dataset's aggregate setting
        self.last_aggregate_method: Optional[str] = None  # Last dataset's aggregate method
        self.last_aggregate_exclude_outliers: bool = False  # Last dataset's exclude outliers setting
        self.last_execution_trace: Any = None  # ExecutionTrace from last run for post-run visualization

    def execute(
        self,
        pipeline: Union[PipelineConfigs, List[Any], Dict, str],
        dataset: Union[DatasetConfigs, SpectroDataset, np.ndarray, Tuple[np.ndarray, ...], Dict, List[Dict], str, List[str]],
        pipeline_name: str = "",
        dataset_name: str = "dataset",
        max_generation_count: int = 10000,
        artifact_loader: Any = None,
        target_model: Optional[Dict[str, Any]] = None,
        explainer: Any = None
    ) -> Tuple[Predictions, Dict[str, Any]]:
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

        nb_combinations = len(pipeline_configs.steps) * len(dataset_configs.configs)
        logger.info("=" * 120)
        logger.starting(
            f"Starting Nirs4all run(s) with {len(pipeline_configs.steps)} "
            f"pipeline on {len(dataset_configs.configs)} dataset ({nb_combinations} total runs)."
        )
        logger.info("=" * 120)

        datasets_predictions = {}
        run_predictions = Predictions()

        # Execute for each dataset
        for config, name in dataset_configs.configs:
            # Create run directory: workspace/runs/<dataset>/
            # All pipelines for a dataset go in the same folder regardless of date
            current_run_dir = self.runs_dir / name
            current_run_dir.mkdir(parents=True, exist_ok=True)

            # Create artifact registry for this dataset (v2 artifact system)
            artifact_registry = None
            if self.mode == "train":
                artifact_registry = ArtifactRegistry(
                    workspace=self.workspace_path,
                    dataset=name,
                    manifest_manager=None  # Set per-pipeline below
                )
                artifact_registry.start_run()

            # Build executor using ExecutorBuilder
            executor = (ExecutorBuilder()
                .with_run_directory(current_run_dir)
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
                .build())

            # Get components from executor for compatibility
            saver = executor.saver
            manifest_manager = executor.manifest_manager

            # Update artifact registry with manifest manager
            if artifact_registry is not None:
                artifact_registry.manifest_manager = manifest_manager

            # Store saver for post-run operations (e.g., export_best_for_dataset)
            self.last_saver = saver
            self.last_executor = executor

            # Load global predictions from workspace root (dataset_name.meta.parquet)
            dataset_prediction_path = self.workspace_path / f"{name}.meta.parquet"
            global_dataset_predictions = Predictions.load_from_file_cls(dataset_prediction_path)
            run_dataset_predictions = Predictions()

            # Execute each pipeline configuration on this dataset
            for i, (steps, config_name, gen_choices) in enumerate(zip(
                pipeline_configs.steps,
                pipeline_configs.names,
                pipeline_configs.generator_choices
            )):
                dataset = dataset_configs.get_dataset(config, name)

                # Capture raw data BEFORE any preprocessing happens
                if self.keep_datasets and name not in self.raw_data:
                    self.raw_data[name] = dataset.x({}, layout="2d")

                if self.verbose > 0:
                    print(dataset)

                # Initialize execution context via executor
                context = executor.initialize_context(dataset)

                # Create RuntimeContext with artifact_registry
                runtime_context = RuntimeContext(
                    saver=saver,
                    manifest_manager=manifest_manager,
                    artifact_loader=artifact_loader,
                    artifact_registry=artifact_registry,
                    step_runner=executor.step_runner,
                    target_model=target_model,
                    explainer=explainer
                )

                # Execute pipeline with cleanup on failure
                config_predictions = Predictions()
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
                except Exception as e:
                    # Cleanup artifacts from failed run
                    if artifact_registry is not None:
                        artifact_registry.cleanup_failed_run()
                    raise

                # Capture last pipeline_uid and manifest_manager for syncing back to runner
                if runtime_context.pipeline_uid:
                    self.last_pipeline_uid = runtime_context.pipeline_uid
                    self.last_manifest_manager = manifest_manager

                # Capture execution trace for post-run visualization
                self.last_execution_trace = runtime_context.get_execution_trace()

                # Capture preprocessed data AFTER preprocessing
                if self.keep_datasets:
                    if name not in self.pp_data:
                        self.pp_data[name] = {}
                    self.pp_data[name][dataset.short_preprocessings_str()] = dataset.x({}, layout="2d")

                # Merge new predictions into stores
                if config_predictions.num_predictions > 0:
                    global_dataset_predictions.merge_predictions(config_predictions)
                    run_dataset_predictions.merge_predictions(config_predictions)
                    run_predictions.merge_predictions(config_predictions)

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
                global_dataset_predictions,
                dataset,
                name,
                dataset_prediction_path,
                saver
            )

            # Store dataset prediction info
            datasets_predictions[name] = {
                "global_predictions": global_dataset_predictions,
                "run_predictions": run_dataset_predictions,
                "dataset": dataset,
                "dataset_name": name
            }

        return run_predictions, datasets_predictions

    def _normalize_pipeline(
        self,
        pipeline: Union[PipelineConfigs, List[Any], Dict, str],
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
        dataset: Union[DatasetConfigs, SpectroDataset, np.ndarray, Tuple[np.ndarray, ...], Dict, List[Dict], str, List[str]],
        dataset_name: str = "array_dataset"
    ) -> DatasetConfigs:
        """Normalize dataset input to DatasetConfigs."""
        if isinstance(dataset, DatasetConfigs):
            return dataset

        # Simplified normalization - delegate to DatasetConfigs
        return DatasetConfigs(dataset) if not isinstance(dataset, (SpectroDataset, np.ndarray, tuple)) else self._wrap_dataset(dataset, dataset_name)

    def _wrap_dataset(self, dataset: Union[SpectroDataset, np.ndarray, Tuple], dataset_name: str) -> DatasetConfigs:
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
        return configs

    def _split_and_add_data(self, dataset: SpectroDataset, X: np.ndarray, y: Optional[np.ndarray], partition_info: Dict) -> None:
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
            elif isinstance(partition_spec, slice):
                partition_indices[partition_name] = partition_spec
            elif isinstance(partition_spec, (list, np.ndarray)):
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
            if isinstance(indices_spec, slice):
                X_partition = X[indices_spec]
                y_partition = y[indices_spec] if y is not None else None
            elif isinstance(indices_spec, (list, np.ndarray)):
                X_partition = X[indices_spec]
                y_partition = y[indices_spec] if y is not None else None
            else:
                raise ValueError(f"Unexpected indices spec type: {type(indices_spec)}")

            # Add to dataset
            if len(X_partition) > 0:
                dataset.add_samples(X_partition, indexes={"partition": partition_name})
                if y_partition is not None and len(y_partition) > 0:
                    dataset.add_targets(y_partition)

    def _extract_dataset_cache(self, dataset: SpectroDataset) -> Tuple:
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
        except:
            x_train = y_train = m_train = None
            train_signal_type = None

        try:
            x_test = dataset.x({"partition": "test"}, layout="2d")
            y_test = dataset.y({"partition": "test"})
            m_test = None
            test_signal_type = dataset.signal_type(0) if dataset.n_sources > 0 else None
        except:
            x_test = y_test = m_test = None
            test_signal_type = None

        # Return 14-tuple with signal_type included
        return (x_train, y_train, m_train, None, None, None, train_signal_type,
                x_test, y_test, m_test, None, None, None, test_signal_type)

    def _print_best_predictions(
        self,
        run_dataset_predictions: Predictions,
        global_dataset_predictions: Predictions,
        dataset: SpectroDataset,
        name: str,
        dataset_prediction_path: Path,
        saver: SimulationSaver
    ):
        """Print and save best predictions for a dataset.

        Saves best prediction as 'best_<pipeline_folder_name>.csv' in the run directory.
        Replaces existing best if the new score is better.
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
                if tab_report_csv_file:
                    filename = f"Report_best_{best['config_name']}_{best['model_name']}_{best['id']}.csv"
                    saver.save_file(filename, tab_report_csv_file)

            if self.save_artifacts:
                # Only save predictions if there's actual prediction data
                if best.get("y_pred") is not None and len(best["y_pred"]) > 0:
                    # Get the pipeline folder name from config_name (e.g., "0001_pls_baseline_abc123")
                    pipeline_folder = best.get('config_name', 'unknown')
                    prediction_name = f"best_{pipeline_folder}.csv"
                    prediction_path = saver.base_path / prediction_name

                    # Check if we should replace existing best prediction
                    should_save = True
                    existing_best_files = list(saver.base_path.glob("best_*.csv"))

                    if existing_best_files:
                        # There's already a best prediction - check if new one is better
                        # Compare using test_score (lower is better for regression metrics like RMSE)
                        current_score = best.get('test_score')
                        if current_score is not None:
                            # Get best from global predictions to compare properly
                            global_best = global_dataset_predictions.get_best(ascending=None)
                            global_best_score = global_best.get('test_score') if global_best else None

                            if global_best_score is not None:
                                # Determine if lower is better based on task type
                                is_regression = best.get('task', 'regression') == 'regression'
                                if is_regression:
                                    # Lower score is better (RMSE, MAE, etc.)
                                    should_save = current_score <= global_best_score
                                else:
                                    # Higher score is better (accuracy, F1, etc.)
                                    should_save = current_score >= global_best_score

                        if should_save:
                            # Remove old best prediction files
                            for old_file in existing_best_files:
                                old_file.unlink()

                    if should_save:
                        Predictions.save_predictions_to_csv(best["y_true"], best["y_pred"], prediction_path)

        if global_dataset_predictions.num_predictions > 0:
            global_dataset_predictions.save_to_file(dataset_prediction_path)

        logger.info("=" * 120)
