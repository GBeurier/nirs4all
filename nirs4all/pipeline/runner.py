"""Pipeline runner - Main entry point for pipeline execution.

This module provides the PipelineRunner class, which serves as the main interface
for executing ML pipelines on spectroscopic datasets. It delegates execution to
PipelineOrchestrator and provides prediction/explanation capabilities via
Predictor and Explainer classes.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import os

import numpy as np

from nirs4all.core.logging import configure_logging
from nirs4all.data.config import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.config.context import ExecutionContext
from nirs4all.pipeline.execution.orchestrator import PipelineOrchestrator
from nirs4all.pipeline.predictor import Predictor
from nirs4all.pipeline.explainer import Explainer
from nirs4all.pipeline.retrainer import Retrainer, RetrainMode, StepMode, ExtractedPipeline


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


def init_global_random_state(seed: Optional[int] = None):
    """Initialize global random state for reproducibility.

    Sets random seeds for numpy, Python's random module, TensorFlow, PyTorch, and sklearn
    to ensure reproducible results across runs.

    Args:
        seed: Random seed value. If None, uses default seed of 42 for TensorFlow and PyTorch.
    """
    import numpy as np
    import random
    import os

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed if seed is not None else 42)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed if seed is not None else 42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed if seed is not None else 42)
    except ImportError:
        pass

    try:
        from sklearn.utils import check_random_state
        _ = check_random_state(seed)
    except ImportError:
        pass


class PipelineRunner:
    """Main pipeline execution interface.

    Orchestrates pipeline execution on datasets, providing a simplified interface for
    training, prediction, and explanation workflows. Delegates actual execution to
    PipelineOrchestrator, Predictor, and Explainer.

    Attributes:
        workspace_path (Path): Root workspace directory
        verbose (int): Verbosity level (0=quiet, 1=info, 2=debug, 3=trace)
        mode (str): Execution mode ('train', 'predict', 'explain')
        save_artifacts (bool): Whether to save binary artifacts (models, transformers)
        save_charts (bool): Whether to save charts and visual outputs
        enable_tab_reports (bool): Whether to generate tabular reports
        continue_on_error (bool): Whether to continue on step failures
        show_spinner (bool): Whether to show progress spinners
        keep_datasets (bool): Whether to keep raw/preprocessed data snapshots
        plots_visible (bool): Whether to display plots interactively
        orchestrator (PipelineOrchestrator): Underlying orchestrator for execution
        predictor (Predictor): Handler for prediction mode
        explainer (Explainer): Handler for explanation mode
        raw_data (Dict[str, np.ndarray]): Raw dataset snapshots (if keep_datasets=True)
        pp_data (Dict[str, Dict[str, np.ndarray]]): Preprocessed data snapshots

    Example:
        >>> # Training workflow
        >>> runner = PipelineRunner(workspace_path="./workspace", verbose=1)
        >>> pipeline = [{"preprocessing": StandardScaler()}, {"model": SVC()}]
        >>> X, y = load_data()
        >>> predictions, dataset_preds = runner.run(pipeline, (X, y))

        >>> # Prediction workflow
        >>> runner = PipelineRunner(mode="predict")
        >>> y_pred, preds = runner.predict(best_model, X_new)

        >>> # Explanation workflow
        >>> runner = PipelineRunner(mode="explain")
        >>> shap_results, out_dir = runner.explain(best_model, X_test)
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
        plots_visible: bool = False,
        random_state: Optional[int] = None,
        # Logging configuration
        log_file: bool = True,
        log_format: str = "pretty",
        use_unicode: Optional[bool] = None,
        use_colors: Optional[bool] = None,
        show_progress_bar: bool = True,
        json_output: bool = False,
    ):
        """Initialize pipeline runner.

        Args:
            workspace_path: Workspace root directory. Defaults to './workspace'
            verbose: Verbosity level (0=quiet, 1=info, 2=debug, 3=trace)
            mode: Execution mode ('train', 'predict', 'explain')
            save_artifacts: Whether to save binary artifacts (models, transformers)
            save_charts: Whether to save charts and visual outputs
            enable_tab_reports: Whether to generate tabular reports
            continue_on_error: Whether to continue on step failures
            show_spinner: Whether to show progress spinners
            keep_datasets: Whether to keep data snapshots (raw/preprocessed)
            plots_visible: Whether to display plots interactively
            random_state: Random seed for reproducibility
            log_file: Whether to write logs to workspace/logs/ directory
            log_format: Output format: "pretty" (default), "minimal", or "json"
            use_unicode: Use Unicode symbols (auto-detected if None). Set to False
                for HPC/cluster environments without Unicode support.
            use_colors: Use ANSI colors (auto-detected if None). Set to False to
                disable colored output.
            show_progress_bar: Whether to show TTY-aware progress bars
            json_output: Also write JSON Lines log file for machine parsing
        """
        if random_state is not None:
            init_global_random_state(random_state)

        if workspace_path is None:
            workspace_path = _get_default_workspace_path()
        self.workspace_path = Path(workspace_path)

        self.verbose = verbose
        self.mode = mode
        self.save_artifacts = save_artifacts
        self.save_charts = save_charts
        self.enable_tab_reports = enable_tab_reports
        self.continue_on_error = continue_on_error
        self.show_spinner = show_spinner
        self.keep_datasets = keep_datasets
        self.plots_visible = plots_visible

        # Store logging configuration
        self.log_file = log_file
        self.log_format = log_format
        self.use_unicode = use_unicode
        self.use_colors = use_colors
        self.show_progress_bar = show_progress_bar
        self.json_output = json_output

        # Configure logging system
        log_dir = self.workspace_path / "logs" if log_file else None
        configure_logging(
            verbose=verbose,
            log_file=log_file,
            log_dir=log_dir,
            log_format=log_format,
            use_unicode=use_unicode,
            use_colors=use_colors,
            show_progress=True,
            show_progress_bar=show_progress_bar,
            json_output=json_output,
        )

        # Create orchestrator
        self.orchestrator = PipelineOrchestrator(
            workspace_path=self.workspace_path,
            verbose=verbose,
            mode=mode,
            save_artifacts=save_artifacts,
            save_charts=save_charts,
            enable_tab_reports=enable_tab_reports,
            continue_on_error=continue_on_error,
            show_spinner=show_spinner,
            keep_datasets=keep_datasets,
            plots_visible=plots_visible
        )

        # Create predictor and explainer
        self.predictor = Predictor(self)
        self.explainer = Explainer(self)
        self.retrainer = Retrainer(self)

        # Expose orchestrator state for convenience
        self.raw_data = self.orchestrator.raw_data
        self.pp_data = self.orchestrator.pp_data
        self._figure_refs = self.orchestrator._figure_refs

        # Model capture support for explainer
        self._capture_model: bool = False

        # Last run aggregate settings (for visualization integration)
        self._last_aggregate_column: Optional[str] = None
        self._last_aggregate_method: Optional[str] = None
        self._last_aggregate_exclude_outliers: bool = False

        # Execution state (synchronized from executor during execution)
        self.step_number: int = 0
        self.substep_number: int = -1
        self.operation_count: int = 0

        # Runtime components (set by executor during execution)
        self.saver: Any = None  # SimulationSaver
        self.manifest_manager: Any = None  # ManifestManager
        self.artifact_loader: Any = None  # ArtifactLoader for predict/explain modes
        self.pipeline_uid: Optional[str] = None  # Current pipeline UID
        self.target_model: Optional[Dict] = None  # Target model for predict/explain modes
        self.last_execution_trace: Any = None  # ExecutionTrace from last run

        # Library for template management
        self._library: Any = None  # PipelineLibrary (lazy)

    def run(
        self,
        pipeline: Union[PipelineConfigs, List[Any], Dict, str],
        dataset: Union[DatasetConfigs, SpectroDataset, np.ndarray, Tuple[np.ndarray, ...], Dict, List[Dict], str, List[str]],
        pipeline_name: str = "",
        dataset_name: str = "dataset",
        max_generation_count: int = 10000
    ) -> Tuple[Predictions, Dict[str, Any]]:
        """Execute pipeline on dataset(s).

        Main entry point for training workflows. Executes one or more pipeline
        configurations on one or more datasets, tracking predictions and artifacts.

        Args:
            pipeline: Pipeline definition (PipelineConfigs, list of steps, dict, or path)
            dataset: Dataset definition (see DatasetConfigs for supported formats)
            pipeline_name: Optional pipeline name for identification
            dataset_name: Name for array-based datasets
            max_generation_count: Max pipeline combinations to generate

        Returns:
            Tuple of (run_predictions, datasets_predictions)
        """
        run_predictions, dataset_predictions = self.orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset,
            pipeline_name=pipeline_name,
            dataset_name=dataset_name,
            max_generation_count=max_generation_count,
            artifact_loader=self.artifact_loader,
            target_model=self.target_model,
            explainer=self.explainer
        )

        # Sync state
        if self.keep_datasets:
            self.raw_data = self.orchestrator.raw_data
            self.pp_data = self.orchestrator.pp_data
        self._figure_refs = self.orchestrator._figure_refs

        # Sync runtime components from last executed pipeline
        if self.orchestrator.last_saver is not None:
            self.saver = self.orchestrator.last_saver
        if self.orchestrator.last_pipeline_uid is not None:
            self.pipeline_uid = self.orchestrator.last_pipeline_uid
        if self.orchestrator.last_manifest_manager is not None:
            self.manifest_manager = self.orchestrator.last_manifest_manager

        # Sync execution state from last executor (via orchestrator)
        # Note: These values come from the last executed pipeline
        if hasattr(self.orchestrator, 'last_executor'):
            if self.orchestrator.last_executor:
                self.step_number = self.orchestrator.last_executor.step_number
                self.substep_number = self.orchestrator.last_executor.substep_number
                self.operation_count = self.orchestrator.last_executor.operation_count

        # Sync aggregate column from last dataset for visualization integration
        if hasattr(self.orchestrator, 'last_aggregate_column'):
            self._last_aggregate_column = self.orchestrator.last_aggregate_column
        if hasattr(self.orchestrator, 'last_aggregate_method'):
            self._last_aggregate_method = self.orchestrator.last_aggregate_method
        if hasattr(self.orchestrator, 'last_aggregate_exclude_outliers'):
            self._last_aggregate_exclude_outliers = self.orchestrator.last_aggregate_exclude_outliers

        # Sync execution trace for post-run visualization
        if hasattr(self.orchestrator, 'last_execution_trace'):
            self.last_execution_trace = self.orchestrator.last_execution_trace

        return run_predictions, dataset_predictions

    def predict(
        self,
        prediction_obj: Union[Dict[str, Any], str],
        dataset: Union[DatasetConfigs, SpectroDataset, np.ndarray, Tuple[np.ndarray, ...], Dict, List[Dict], str, List[str]],
        dataset_name: str = "prediction_dataset",
        all_predictions: bool = False,
        verbose: int = 0
    ) -> Union[Tuple[np.ndarray, Predictions], Tuple[Dict[str, Any], Predictions]]:
        """Run prediction using a saved model on new dataset.

        Delegates to Predictor class for actual execution.

        Args:
            prediction_obj: Model identifier (dict with config_path or prediction ID)
            dataset: New dataset to predict on
            dataset_name: Name for the dataset
            all_predictions: If True, return all predictions; if False, return single best
            verbose: Verbosity level

        Returns:
            If all_predictions=False: (y_pred, predictions)
            If all_predictions=True: (predictions_dict, predictions)
        """
        return self.predictor.predict(
            prediction_obj=prediction_obj,
            dataset=dataset,
            dataset_name=dataset_name,
            all_predictions=all_predictions,
            verbose=verbose
        )

    def explain(
        self,
        prediction_obj: Union[Dict[str, Any], str],
        dataset: Union[DatasetConfigs, SpectroDataset, np.ndarray, Tuple[np.ndarray, ...], Dict, List[Dict], str, List[str]],
        dataset_name: str = "explain_dataset",
        shap_params: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        plots_visible: bool = True
    ) -> Tuple[Dict[str, Any], str]:
        """Generate SHAP explanations for a saved model.

        Delegates to Explainer class for actual execution.

        Args:
            prediction_obj: Model identifier (dict with config_path or prediction ID)
            dataset: Dataset to explain on
            dataset_name: Name for the dataset
            shap_params: SHAP configuration parameters
            verbose: Verbosity level
            plots_visible: Whether to display plots interactively

        Returns:
            Tuple of (shap_results_dict, output_directory_path)
        """
        return self.explainer.explain(
            prediction_obj=prediction_obj,
            dataset=dataset,
            dataset_name=dataset_name,
            shap_params=shap_params,
            verbose=verbose,
            plots_visible=plots_visible
        )

    def export_best_for_dataset(
        self,
        dataset_name: str,
        mode: str = "predictions"
    ) -> Optional[Path]:
        """Export best results for a dataset to exports/ folder.

        Args:
            dataset_name: Name of the dataset to export
            mode: Export mode ('predictions' or other)

        Returns:
            Path to exported file, or None if export failed
        """
        saver = self.saver or getattr(self.predictor, 'saver', None) or getattr(self.explainer, 'saver', None)
        if saver is None:
            raise ValueError("No saver configured. Run a pipeline first.")

        return saver.export_best_for_dataset(
            dataset_name,
            self.workspace_path,
            self.orchestrator.runs_dir,
            mode
        )

    @property
    def current_run_dir(self) -> Optional[Path]:
        """Get current run directory.

        Returns:
            Path to current run directory, or None if not set
        """
        if self.saver and hasattr(self.saver, 'base_path'):
            return self.saver.base_path
        # Fallback for predictor/explainer modes
        saver = getattr(self.predictor, 'saver', None) or getattr(self.explainer, 'saver', None)
        return getattr(saver, 'base_path', None) if saver else None

    @property
    def runs_dir(self) -> Path:
        """Get runs directory.

        Returns:
            Path to runs directory in workspace
        """
        return self.orchestrator.runs_dir

    @property
    def last_aggregate(self) -> Optional[str]:
        """Get aggregate column from the last executed dataset.

        Returns the aggregation setting from the last dataset processed by run().
        This can be used to create a PredictionAnalyzer with matching defaults.

        Returns:
            Aggregate column name ('y' for y-based aggregation, column name for
            metadata-based aggregation, or None if no aggregation was set).

        Example:
            >>> runner = PipelineRunner()
            >>> predictions, _ = runner.run(pipeline, DatasetConfigs(path, aggregate='sample_id'))
            >>> # Create analyzer with same aggregate setting
            >>> analyzer = PredictionAnalyzer(predictions, default_aggregate=runner.last_aggregate)
        """
        return self._last_aggregate_column

    @property
    def last_aggregate_method(self) -> Optional[str]:
        """Get aggregate method from the last executed dataset.

        Returns:
            Aggregate method ('mean', 'median', 'vote') or None for default.
        """
        return self._last_aggregate_method

    @property
    def last_aggregate_exclude_outliers(self) -> bool:
        """Get aggregate exclude_outliers setting from the last executed dataset.

        Returns:
            True if T² outlier exclusion was enabled, False otherwise.
        """
        return self._last_aggregate_exclude_outliers

    @property
    def library(self) -> "PipelineLibrary":
        """Get pipeline library for template management.

        Returns:
            PipelineLibrary instance for managing pipeline templates
        """
        if self._library is None:
            from nirs4all.pipeline.storage.library import PipelineLibrary
            self._library = PipelineLibrary(self.workspace_path)
        return self._library

    def next_op(self) -> int:
        """Get the next operation ID (for controller compatibility).

        Returns:
            Next operation counter value
        """
        self.operation_count += 1
        return self.operation_count

    def export(
        self,
        source: Union[Dict[str, Any], str, Path],
        output_path: Union[str, Path],
        format: str = "n4a",
        include_metadata: bool = True,
        compress: bool = True
    ) -> Path:
        """Export a trained pipeline to a standalone bundle.

        Creates a self-contained prediction bundle that can be used for
        deployment, sharing, or archival without requiring the original
        workspace or full nirs4all installation.

        Supported formats:
            - 'n4a': Full bundle (ZIP archive with artifacts and metadata)
            - 'n4a.py': Portable Python script with embedded artifacts

        Phase 6 Feature:
            This method enables exporting trained pipelines as standalone
            bundles that can be loaded and used for prediction without
            the original workspace structure.

        Args:
            source: Prediction source to export. Can be:
                - prediction dict: From a previous run's Predictions object
                - folder path: Path to a pipeline directory
                - Run object: Best prediction from a Run
            output_path: Path for the output bundle file
            format: Bundle format ('n4a' or 'n4a.py')
            include_metadata: Whether to include full metadata in bundle
            compress: Whether to compress artifacts (for .n4a format)

        Returns:
            Path to the created bundle file

        Raises:
            ValueError: If format is not supported
            FileNotFoundError: If source cannot be resolved

        Example:
            >>> runner = PipelineRunner()
            >>> predictions, _ = runner.run(pipeline, dataset)
            >>> best_pred = predictions.top(n=1)[0]
            >>>
            >>> # Export to .n4a bundle
            >>> runner.export(best_pred, "exports/wheat_model.n4a")
            >>>
            >>> # Export to portable Python script
            >>> runner.export(best_pred, "exports/wheat_model.n4a.py", format='n4a.py')
            >>>
            >>> # Later, predict from bundle
            >>> y_pred, _ = runner.predict("exports/wheat_model.n4a", X_new)
        """
        from nirs4all.pipeline.bundle import BundleGenerator

        generator = BundleGenerator(
            workspace_path=self.workspace_path,
            verbose=self.verbose
        )

        return generator.export(
            source=source,
            output_path=output_path,
            format=format,
            include_metadata=include_metadata,
            compress=compress
        )

    def export_model(
        self,
        source: Union[Dict[str, Any], str, Path],
        output_path: Union[str, Path],
        format: Optional[str] = None,
        fold: Optional[int] = None
    ) -> Path:
        """Export only the model artifact from a trained pipeline.

        Unlike `export()` which creates a full bundle with all preprocessing
        artifacts and metadata, this method exports just the model binary.
        This is useful when you want a lightweight model file that can be
        loaded directly into other pipelines or used with external tools.

        The output format is determined by the file extension or can be
        specified explicitly. The model can then be reloaded using:
        - Direct path in pipeline config: {"model": "path/to/model.joblib"}
        - As prediction source: runner.predict("path/to/model.joblib", data)

        Args:
            source: Prediction source to export from. Can be:
                - prediction dict: From a previous run's Predictions object
                - folder path: Path to a pipeline directory
                - bundle path: Path to a .n4a bundle
            output_path: Path for the output model file. Extension determines
                format: .joblib, .pkl, .h5, .keras, .pt
            format: Optional explicit format ('joblib', 'pickle', 'keras_h5').
                If None, determined from output_path extension.
            fold: Optional fold index to export. If None, exports fold 0 or
                the primary model artifact.

        Returns:
            Path to the created model file

        Raises:
            ValueError: If no model artifact found
            FileNotFoundError: If source cannot be resolved

        Example:
            >>> runner = PipelineRunner()
            >>> predictions, _ = runner.run(pipeline, dataset)
            >>> best_pred = predictions.top(n=1)[0]
            >>>
            >>> # Export just the model
            >>> runner.export_model(best_pred, "exports/pls_model.joblib")
            >>>
            >>> # Later, use in new pipeline
            >>> new_pipeline = [
            ...     MinMaxScaler(),
            ...     {"model": "exports/pls_model.joblib", "name": "pretrained"}
            ... ]
        """
        from nirs4all.pipeline.resolver import PredictionResolver
        from nirs4all.pipeline.storage.artifacts.artifact_persistence import to_bytes

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resolve the prediction source
        resolver = PredictionResolver(
            workspace_path=self.workspace_path,
            runs_dir=self.runs_dir
        )
        resolved = resolver.resolve(source, verbose=self.verbose)

        if resolved.model_step_index is None:
            raise ValueError("No model step found in the resolved prediction")

        # Get the model artifact
        if resolved.artifact_provider is None:
            raise ValueError("No artifact provider available for this source")

        artifacts = resolved.artifact_provider.get_artifacts_for_step(
            resolved.model_step_index
        )

        if not artifacts:
            raise ValueError(
                f"No model artifacts found at step {resolved.model_step_index}"
            )

        # Select the fold
        if fold is not None:
            # Find artifact for specific fold
            model = None
            for artifact_id, artifact in artifacts:
                if f":{fold}" in str(artifact_id) or artifact_id.endswith(f"_{fold}"):
                    model = artifact
                    break
            if model is None:
                raise ValueError(f"No artifact found for fold {fold}")
        else:
            # Use first artifact (fold 0 or primary)
            _, model = artifacts[0]

        # Determine format from extension if not specified
        if format is None:
            ext = output_path.suffix.lower()
            format_map = {
                '.joblib': 'joblib',
                '.pkl': 'cloudpickle',
                '.pickle': 'cloudpickle',
                '.h5': 'keras_h5',
                '.hdf5': 'keras_h5',
                '.keras': 'tensorflow_keras',
                '.pt': 'pytorch_state_dict',
                '.pth': 'pytorch_state_dict',
            }
            format = format_map.get(ext, 'joblib')

        # Serialize and write
        data, actual_format = to_bytes(model, format)
        with open(output_path, 'wb') as f:
            f.write(data)

        if self.verbose > 0:
            print(f"Exported model to {output_path} (format: {actual_format})")

        return output_path

    def retrain(
        self,
        source: Union[Dict[str, Any], str, Path],
        dataset: Union[DatasetConfigs, SpectroDataset, np.ndarray, Tuple[np.ndarray, ...], Dict, List[Dict], str, List[str]],
        mode: str = "full",
        dataset_name: str = "retrain_dataset",
        new_model: Optional[Any] = None,
        epochs: Optional[int] = None,
        step_modes: Optional[List[StepMode]] = None,
        verbose: int = 0,
        **kwargs
    ) -> Tuple[Predictions, Dict[str, Any]]:
        """Retrain a pipeline on new data.

        Enables retraining trained pipelines with various modes:
        - full: Train from scratch with same pipeline structure
        - transfer: Use existing preprocessing artifacts, train new model
        - finetune: Continue training existing model with new data

        Phase 7 Feature:
            This method enables retraining pipelines without having to
            reconstruct the pipeline configuration manually. It uses the
            resolved prediction source (from Phase 3/4) to extract the
            pipeline structure and optionally reuse preprocessing artifacts.

        Args:
            source: Prediction source to retrain from. Can be:
                - prediction dict: From a previous run's Predictions object
                - folder path: Path to a pipeline directory
                - Run object: Best prediction from a Run
                - artifact_id: Direct artifact reference
                - bundle: Exported prediction bundle (.n4a)
            dataset: New dataset to train on. Supports same formats as run()
            mode: Retrain mode:
                - 'full': Train everything from scratch (same pipeline structure)
                - 'transfer': Use existing preprocessing, train new model
                - 'finetune': Continue training existing model
            dataset_name: Name for the dataset if array-based
            new_model: Optional new model for transfer mode (replaces original)
            epochs: Optional epochs for fine-tuning
            step_modes: Optional per-step mode overrides for fine-grained control
            verbose: Verbosity level
            **kwargs: Additional parameters:
                - learning_rate: Learning rate for fine-tuning
                - freeze_layers: List of layers to freeze during fine-tuning

        Returns:
            Tuple of (run_predictions, datasets_predictions)

        Raises:
            ValueError: If mode is invalid or source cannot be resolved
            FileNotFoundError: If source references files that don't exist

        Example:
            >>> runner = PipelineRunner()
            >>> predictions, _ = runner.run(pipeline, dataset)
            >>> best_pred = predictions.top(n=1)[0]
            >>>
            >>> # Full retrain on new data
            >>> new_preds, _ = runner.retrain(best_pred, new_data, mode='full')
            >>>
            >>> # Transfer: use preprocessing from old model, train new one
            >>> new_preds, _ = runner.retrain(
            ...     best_pred, new_data, mode='transfer',
            ...     new_model=XGBRegressor()
            ... )
            >>>
            >>> # Finetune: continue training existing model
            >>> new_preds, _ = runner.retrain(
            ...     best_pred, new_data, mode='finetune', epochs=10
            ... )
            >>>
            >>> # Fine-grained control: specify per-step modes
            >>> from nirs4all.pipeline import StepMode
            >>> step_modes = [
            ...     StepMode(step_index=1, mode='predict'),  # Use existing
            ...     StepMode(step_index=2, mode='train'),    # Retrain
            ... ]
            >>> new_preds, _ = runner.retrain(
            ...     best_pred, new_data, mode='full', step_modes=step_modes
            ... )
        """
        return self.retrainer.retrain(
            source=source,
            dataset=dataset,
            mode=mode,
            dataset_name=dataset_name,
            new_model=new_model,
            epochs=epochs,
            step_modes=step_modes,
            verbose=verbose,
            **kwargs
        )

    def extract(
        self,
        source: Union[Dict[str, Any], str, Path]
    ) -> ExtractedPipeline:
        """Extract a trained pipeline for inspection or modification.

        Loads a trained pipeline from a prediction source and returns an
        ExtractedPipeline object that can be inspected, modified, and then
        executed with runner.run().

        Phase 7 Feature:
            This method enables extracting and modifying trained pipelines
            without retraining from scratch.

        Args:
            source: Prediction source to extract. Can be:
                - prediction dict: From a previous run's Predictions object
                - folder path: Path to a pipeline directory
                - Run object: Best prediction from a Run
                - artifact_id: Direct artifact reference
                - bundle: Exported prediction bundle (.n4a)

        Returns:
            ExtractedPipeline object with:
                - steps: List of pipeline steps (can be modified)
                - trace: Original execution trace (read-only)
                - artifact_provider: Provider for original artifacts
                - model_step_index: Index of the model step
                - preprocessing_chain: Summary of preprocessing

        Example:
            >>> runner = PipelineRunner()
            >>> predictions, _ = runner.run(pipeline, dataset)
            >>> best_pred = predictions.top(n=1)[0]
            >>>
            >>> # Extract for inspection
            >>> extracted = runner.extract(best_pred)
            >>> print(f"Steps: {len(extracted.steps)}")
            >>> print(f"Preprocessing: {extracted.preprocessing_chain}")
            >>>
            >>> # Modify and run
            >>> from sklearn.ensemble import RandomForestRegressor
            >>> extracted.set_model(RandomForestRegressor())
            >>> new_preds, _ = runner.run(extracted.steps, new_data)
        """
        return self.retrainer.extract(source, verbose=self.verbose)


