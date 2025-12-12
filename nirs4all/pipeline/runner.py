"""Pipeline runner - Main entry point for pipeline execution.

This module provides the PipelineRunner class, which serves as the main interface
for executing ML pipelines on spectroscopic datasets. It delegates execution to
PipelineOrchestrator and provides prediction/explanation capabilities via
Predictor and Explainer classes.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from nirs4all.data.config import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.config.context import ExecutionContext
from nirs4all.pipeline.execution.orchestrator import PipelineOrchestrator
from nirs4all.pipeline.predictor import Predictor
from nirs4all.pipeline.explainer import Explainer


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
        verbose (int): Verbosity level (0=quiet, 1=info, 2=debug)
        mode (str): Execution mode ('train', 'predict', 'explain')
        save_files (bool): Whether to save output files
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
        save_files: bool = True,
        enable_tab_reports: bool = True,
        continue_on_error: bool = False,
        show_spinner: bool = True,
        keep_datasets: bool = True,
        plots_visible: bool = False,
        random_state: Optional[int] = None
    ):
        """Initialize pipeline runner.

        Args:
            workspace_path: Workspace root directory. Defaults to './workspace'
            verbose: Verbosity level (0=quiet, 1=info, 2=debug)
            mode: Execution mode ('train', 'predict', 'explain')
            save_files: Whether to save output files
            enable_tab_reports: Whether to generate tabular reports
            continue_on_error: Whether to continue on step failures
            show_spinner: Whether to show progress spinners
            keep_datasets: Whether to keep data snapshots (raw/preprocessed)
            plots_visible: Whether to display plots interactively
            random_state: Random seed for reproducibility
        """
        if random_state is not None:
            init_global_random_state(random_state)

        if workspace_path is None:
            workspace_path = Path.cwd() / "workspace"
        self.workspace_path = Path(workspace_path)

        self.verbose = verbose
        self.mode = mode
        self.save_files = save_files
        self.enable_tab_reports = enable_tab_reports
        self.continue_on_error = continue_on_error
        self.show_spinner = show_spinner
        self.keep_datasets = keep_datasets
        self.plots_visible = plots_visible

        # Create orchestrator
        self.orchestrator = PipelineOrchestrator(
            workspace_path=self.workspace_path,
            verbose=verbose,
            mode=mode,
            save_files=save_files,
            enable_tab_reports=enable_tab_reports,
            continue_on_error=continue_on_error,
            show_spinner=show_spinner,
            keep_datasets=keep_datasets,
            plots_visible=plots_visible
        )

        # Create predictor and explainer
        self.predictor = Predictor(self)
        self.explainer = Explainer(self)

        # Expose orchestrator state for convenience
        self.raw_data = self.orchestrator.raw_data
        self.pp_data = self.orchestrator.pp_data
        self._figure_refs = self.orchestrator._figure_refs

        # Model capture support for explainer
        self._capture_model: bool = False

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

