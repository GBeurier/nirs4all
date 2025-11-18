"""Generate clean runner.py without backward compatibility."""
from pathlib import Path

CLEAN_RUNNER_CONTENT = '''"""Pipeline runner - Main entry point for pipeline execution.

This module provides the PipelineRunner class, which serves as the main interface
for executing ML pipelines on spectroscopic datasets. It delegates execution to
PipelineOrchestrator while managing runner-specific state for predict/explain modes.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from nirs4all.data.config import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.binary_loader import BinaryLoader
from nirs4all.pipeline.config import PipelineConfigs
from nirs4all.pipeline.execution.orchestrator import PipelineOrchestrator


def init_global_random_state(seed: Optional[int] = None):
    """Initialize global random state for reproducibility.

    Sets random seeds for numpy, Python's random module, TensorFlow, and sklearn
    to ensure reproducible results across runs.

    Args:
        seed: Random seed value. If None, uses default seed of 42 for TensorFlow.
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
        from sklearn.utils import check_random_state
        _ = check_random_state(seed)
    except ImportError:
        pass


class PipelineRunner:
    """Main pipeline execution interface.

    Orchestrates pipeline execution on datasets, providing a simplified interface for
    training, prediction, and explanation workflows. Delegates actual execution to
    PipelineOrchestrator while managing runner-specific state.

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
        raw_data (Dict[str, np.ndarray]): Raw dataset snapshots (if keep_datasets=True)
        pp_data (Dict[str, Dict[str, np.ndarray]]): Preprocessed data snapshots

    Predict/Explain State:
        saver: File saver instance (set during predict/explain)
        manifest_manager: Manifest manager instance
        pipeline_uid: Current pipeline unique identifier
        binary_loader: Binary artifact loader for predict mode
        config_path: Path to loaded configuration
        target_model: Target model metadata for predictions
        _capture_model: Flag to capture model for SHAP analysis
        _captured_model: Captured model instance

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

        # State for predict/explain modes
        self.saver = None
        self.manifest_manager = None
        self.pipeline_uid: Optional[str] = None
        self.binary_loader: Optional[BinaryLoader] = None
        self.config_path: Optional[str] = None
        self.target_model: Optional[Dict[str, Any]] = None
        self._capture_model: bool = False
        self._captured_model: Optional[Any] = None

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

        # Expose orchestrator state for convenience
        self.raw_data = self.orchestrator.raw_data
        self.pp_data = self.orchestrator.pp_data
        self._figure_refs = self.orchestrator._figure_refs

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
        result = self.orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset,
            pipeline_name=pipeline_name,
            dataset_name=dataset_name,
            max_generation_count=max_generation_count,
            runner=self
        )

        # Sync state
        if self.keep_datasets:
            self.raw_data = self.orchestrator.raw_data
            self.pp_data = self.orchestrator.pp_data
        self._figure_refs = self.orchestrator._figure_refs

        return result.run_predictions, result.dataset_predictions

    def predict(
        self,
        prediction_obj: Union[Dict[str, Any], str],
        dataset: Union[DatasetConfigs, SpectroDataset, np.ndarray, Tuple[np.ndarray, ...], Dict, List[Dict], str, List[str]],
        dataset_name: str = "prediction_dataset",
        all_predictions: bool = False,
        verbose: int = 0
    ) -> Union[Tuple[np.ndarray, Predictions], Tuple[Dict[str, Any], Predictions]]:
        """Run prediction using a saved model on new dataset."""
        from nirs4all.data.config import DatasetConfigs
        from nirs4all.pipeline.manifest_manager import ManifestManager
        from nirs4all.utils.emoji import ROCKET, CHECK, SEARCH
        from nirs4all.pipeline.context import ExecutionContext, DataSelector, PipelineState, StepMetadata
        import json

        print("=" * 120)
        print(f"\\033[94m{ROCKET}Starting Nirs4all prediction(s)\\033[0m")
        print("=" * 120)

        # Normalize dataset
        dataset_config = self.orchestrator._normalize_dataset(dataset, dataset_name, runner=self)

        self.mode = "predict"
        self.verbose = verbose

        # Initialize saver for prediction mode
        run_dir = self._get_run_dir_from_prediction(prediction_obj)
        from nirs4all.pipeline.io import SimulationSaver
        self.saver = SimulationSaver(run_dir, save_files=self.save_files)
        self.manifest_manager = ManifestManager(run_dir)

        # Load pipeline steps
        steps = self._prepare_replay(prediction_obj, dataset_config, verbose)

        # Execute pipeline on dataset
        run_predictions = Predictions()
        for config, name in dataset_config.configs:
            dataset_obj = dataset_config.get_dataset(config, name)
            config_predictions = Predictions()

            # Initialize context
            context = ExecutionContext(
                selector=DataSelector(
                    partition=None,
                    processing=[["raw"]] * dataset_obj.features_sources(),
                    layout="2d",
                    concat_source=True
                ),
                state=PipelineState(y_processing="numeric", step_number=0, mode="predict"),
                metadata=StepMetadata()
            )

            # Create executor and run
            from nirs4all.pipeline.execution.executor import PipelineExecutor
            from nirs4all.pipeline.steps.runner import StepRunner
            from nirs4all.pipeline.steps.parser import StepParser
            from nirs4all.pipeline.steps.router import ControllerRouter

            step_runner = StepRunner(
                parser=StepParser(),
                router=ControllerRouter(),
                verbose=verbose,
                mode="predict",
                show_spinner=self.show_spinner
            )

            executor = PipelineExecutor(
                step_runner=step_runner,
                artifact_manager=None,
                manifest_manager=self.manifest_manager,
                verbose=verbose,
                mode="predict",
                continue_on_error=self.continue_on_error,
                saver=self.saver
            )

            executor.execute(steps, "prediction", dataset_obj, context, self, config_predictions)
            run_predictions.merge_predictions(config_predictions)

        if all_predictions:
            res = {}
            for pred in run_predictions.to_dicts():
                if pred['dataset_name'] not in res:
                    res[pred['dataset_name']] = {}
                res[pred['dataset_name']][pred['id']] = pred['y_pred']
            return res, run_predictions

        # Get single prediction matching target model
        single_pred = run_predictions.get_similar(
            model_name=self.target_model.get('model_name', None),
            step_idx=self.target_model.get('step_idx', None),
            fold_id=self.target_model.get('fold_id', None),
            partition='test'
        )

        if single_pred is None:
            raise ValueError("No matching prediction found for the specified model criteria.")

        print(f"{CHECK}Predicted with: {single_pred['model_name']} [{single_pred['id']}]")
        filename = f"Predict_[{single_pred['id']}].csv"
        y_pred = single_pred["y_pred"]
        prediction_path = self.saver.base_path / filename
        Predictions.save_predictions_to_csv(y_pred=y_pred, filepath=prediction_path)

        return single_pred["y_pred"], run_predictions

    def explain(
        self,
        prediction_obj: Union[Dict[str, Any], str],
        dataset: Union[DatasetConfigs, SpectroDataset, np.ndarray, Tuple[np.ndarray, ...], Dict, List[Dict], str, List[str]],
        dataset_name: str = "explain_dataset",
        shap_params: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        plots_visible: bool = True
    ) -> Tuple[Dict[str, Any], str]:
        """Generate SHAP explanations for a saved model."""
        from nirs4all.data.config import DatasetConfigs
        from nirs4all.utils.emoji import SEARCH, CHECK
        from nirs4all.visualization.analysis.shap import ShapAnalyzer
        from nirs4all.pipeline.manifest_manager import ManifestManager
        from nirs4all.pipeline.context import ExecutionContext, DataSelector, PipelineState, StepMetadata
        from nirs4all.pipeline.io import SimulationSaver
        from nirs4all.pipeline.execution.executor import PipelineExecutor
        from nirs4all.pipeline.steps.runner import StepRunner
        from nirs4all.pipeline.steps.parser import StepParser
        from nirs4all.pipeline.steps.router import ControllerRouter

        print("=" * 120)
        print(f"\\033[94m{SEARCH}Starting SHAP Explanation Analysis\\033[0m")
        print("=" * 120)

        # Setup SHAP parameters
        if shap_params is None:
            shap_params = {}
        shap_params.setdefault('n_samples', 200)
        shap_params.setdefault('visualizations', ['spectral', 'summary'])
        shap_params.setdefault('explainer_type', 'auto')
        shap_params.setdefault('bin_size', 20)
        shap_params.setdefault('bin_stride', 10)
        shap_params.setdefault('bin_aggregation', 'sum')

        # Normalize dataset
        dataset_config = self.orchestrator._normalize_dataset(dataset, dataset_name, runner=self)

        self.mode = "explain"
        self._capture_model = True
        self._captured_model = None

        try:
            # Setup saver and manifest
            config, name = dataset_config.configs[0]
            run_dir = self._get_run_dir_from_prediction(prediction_obj)
            self.saver = SimulationSaver(run_dir, save_files=self.save_files)
            self.manifest_manager = ManifestManager(run_dir)

            # Load pipeline
            steps = self._prepare_replay(prediction_obj, dataset_config, verbose)
            dataset_obj = dataset_config.get_dataset(config, name)

            # Execute pipeline to capture model
            context = ExecutionContext(
                selector=DataSelector(
                    partition=None,
                    processing=[["raw"]] * dataset_obj.features_sources(),
                    layout="2d",
                    concat_source=True
                ),
                state=PipelineState(y_processing="numeric", step_number=0, mode="explain"),
                metadata=StepMetadata()
            )

            config_predictions = Predictions()

            step_runner = StepRunner(
                parser=StepParser(),
                router=ControllerRouter(),
                verbose=verbose,
                mode="explain",
                show_spinner=self.show_spinner
            )

            executor = PipelineExecutor(
                step_runner=step_runner,
                artifact_manager=None,
                manifest_manager=self.manifest_manager,
                verbose=verbose,
                mode="explain",
                continue_on_error=self.continue_on_error,
                saver=self.saver
            )

            executor.execute(steps, "explanation", dataset_obj, context, self, config_predictions)

            # Extract captured model
            if self._captured_model is None:
                raise ValueError("Failed to capture model. Model controller may not support capture.")

            model, controller = self._captured_model

            # Get test data
            test_context = context.with_partition('test')
            X_test = dataset_obj.x(test_context, layout=controller.get_preferred_layout())
            y_test = dataset_obj.y(test_context)

            # Get feature names
            feature_names = None
            if hasattr(dataset_obj, 'wavelengths') and dataset_obj.wavelengths is not None:
                feature_names = [f"λ{w:.1f}" for w in dataset_obj.wavelengths]

            task_type = 'classification' if dataset_obj.task_type and dataset_obj.task_type.is_classification else 'regression'

            # Create output directory
            model_id = self.target_model.get('id', 'unknown')
            output_dir = self.saver.base_path / dataset_obj.name / self.config_path / "explanations" / model_id
            output_dir.mkdir(parents=True, exist_ok=True)

            if verbose > 0:
                print(f"📁 Output directory: {output_dir}")

            # Run SHAP analysis
            analyzer = ShapAnalyzer()
            shap_results = analyzer.explain_model(
                model=model,
                X=X_test,
                y=y_test,
                feature_names=feature_names,
                task_type=task_type,
                n_background=shap_params['n_samples'],
                explainer_type=shap_params['explainer_type'],
                output_dir=str(output_dir),
                visualizations=shap_params['visualizations'],
                bin_size=shap_params['bin_size'],
                bin_stride=shap_params['bin_stride'],
                bin_aggregation=shap_params['bin_aggregation'],
                plots_visible=plots_visible
            )

            shap_results['model_name'] = self.target_model.get('model_name', 'unknown')
            shap_results['model_id'] = model_id
            shap_results['dataset_name'] = dataset_obj.name

            if verbose > 0:
                print(f"\\n{CHECK}SHAP explanation completed!")
                print(f"📁 Visualizations saved to: {output_dir}")
                for viz in shap_params['visualizations']:
                    print(f"   • {viz}.png")
                print("=" * 120)

            return shap_results, str(output_dir)

        finally:
            self._capture_model = False
            self._captured_model = None

    def export_best_for_dataset(
        self,
        dataset_name: str,
        mode: str = "predictions"
    ) -> Optional[Path]:
        """Export best results for a dataset to exports/ folder."""
        if self.saver is None:
            raise ValueError("No saver configured. Run a pipeline first.")

        return self.saver.export_best_for_dataset(
            dataset_name,
            self.workspace_path,
            self.orchestrator.runs_dir,
            mode
        )

    def _get_run_dir_from_prediction(self, prediction_obj: Union[Dict[str, Any], str]) -> Path:
        """Get run directory from prediction object."""
        if isinstance(prediction_obj, dict):
            if 'run_dir' in prediction_obj:
                return Path(prediction_obj['run_dir'])
            elif 'config_path' in prediction_obj:
                config_path = prediction_obj['config_path']
                dataset_name = Path(config_path).parts[0]
                matching_dirs = sorted(
                    self.orchestrator.runs_dir.glob(f"*_{dataset_name}"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                if matching_dirs:
                    return matching_dirs[0]
                raise ValueError(f"No run directory found for dataset: {dataset_name}")

        # Fallback: use most recent run
        run_dirs = sorted(
            self.orchestrator.runs_dir.glob("*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if run_dirs:
            return run_dirs[0]
        raise ValueError("No run directories found")

    def _prepare_replay(
        self,
        selection_obj: Union[Dict[str, Any], str],
        dataset_config: DatasetConfigs,
        verbose: int = 0
    ) -> List[Any]:
        """Prepare pipeline replay from saved configuration."""
        from nirs4all.utils.emoji import SEARCH
        import json

        # Get configuration path and target model
        config_path, target_model = self.saver.get_predict_targets(selection_obj)
        target_model.pop("y_pred", None)
        target_model.pop("y_true", None)

        self.config_path = config_path
        self.target_model = target_model

        pipeline_uid = target_model.get('pipeline_uid')
        if not pipeline_uid:
            raise ValueError(
                "No pipeline_uid found in prediction metadata. "
                "This prediction was created with an older version of nirs4all. "
                "Please retrain the model."
            )

        # Load pipeline configuration
        pipeline_dir_name = Path(config_path).parts[-1] if '/' in config_path or '\\\\' in config_path else config_path
        config_dir = self.saver.base_path / pipeline_dir_name
        pipeline_json = config_dir / "pipeline.json"

        if verbose > 0:
            print(f"{SEARCH}Loading {pipeline_json}")

        if not pipeline_json.exists():
            raise FileNotFoundError(f"Pipeline not found: {pipeline_json}")

        with open(pipeline_json, 'r', encoding='utf-8') as f:
            pipeline_data = json.load(f)

        steps = pipeline_data["steps"] if isinstance(pipeline_data, dict) and "steps" in pipeline_data else pipeline_data

        # Load binaries from manifest
        manifest_path = self.saver.base_path / pipeline_uid / "manifest.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\\n"
                f"Pipeline UID: {pipeline_uid}\\n"
                f"The model artifacts may have been deleted or moved."
            )

        print(f"{SEARCH}Loading from manifest: {pipeline_uid}")
        manifest = self.manifest_manager.load_manifest(pipeline_uid)
        self.binary_loader = BinaryLoader.from_manifest(manifest, self.saver.base_path)

        return steps

    @property
    def current_run_dir(self) -> Optional[Path]:
        """Get current run directory."""
        return getattr(self.saver, 'base_path', None) if self.saver else None

    @property
    def runs_dir(self) -> Path:
        """Get runs directory."""
        return self.orchestrator.runs_dir
'''

if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "nirs4all" / "pipeline" / "runner.py"
    print(f"Writing clean runner to: {output_path}")
    output_path.write_text(CLEAN_RUNNER_CONTENT, encoding='utf-8')
    print(f"✓ Clean runner written ({len(CLEAN_RUNNER_CONTENT)} characters)")
    print(f"  Lines: {CLEAN_RUNNER_CONTENT.count(chr(10))}")
