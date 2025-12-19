"""Pipeline explainer - Handles SHAP explanation generation.

This module provides the Explainer class for generating model explanations
using SHAP (SHapley Additive exPlanations) on trained pipelines.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from nirs4all.data.config import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.storage.artifacts.artifact_loader import ArtifactLoader
from nirs4all.pipeline.config.context import ExecutionContext, DataSelector, PipelineState, StepMetadata, LoaderArtifactProvider
from nirs4all.pipeline.execution.builder import ExecutorBuilder
from nirs4all.pipeline.storage.io import SimulationSaver
from nirs4all.pipeline.storage.manifest_manager import ManifestManager

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)


class Explainer:
    """Handles SHAP explanation generation for trained models.

    This class manages the explanation workflow: loading saved models,
    replaying pipelines to capture the trained model, and generating
    SHAP explanations with visualizations.

    Attributes:
        runner: Parent PipelineRunner instance
        saver: File saver for managing outputs
        manifest_manager: Manager for pipeline manifests
        pipeline_uid: Unique identifier for the pipeline
        artifact_loader: Loader for trained model artifacts
        config_path: Path to the pipeline configuration
        target_model: Metadata for the target model
        captured_model: Tuple of (model, controller) captured during replay
    """

    def __init__(self, runner: 'PipelineRunner'):
        """Initialize explainer.

        Args:
            runner: Parent PipelineRunner instance
        """
        self.runner = runner
        self.saver: Optional[SimulationSaver] = None
        self.manifest_manager: Optional[ManifestManager] = None
        self.pipeline_uid: Optional[str] = None
        self.artifact_loader: Optional[ArtifactLoader] = None
        self.config_path: Optional[str] = None
        self.target_model: Optional[Dict[str, Any]] = None
        self.captured_model: Optional[Tuple[Any, Any]] = None

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

        Args:
            prediction_obj: Model identifier (dict with config_path or prediction ID)
            dataset: Dataset to explain on
            dataset_name: Name for the dataset
            shap_params: SHAP configuration parameters
            verbose: Verbosity level
            plots_visible: Whether to display plots interactively

        Returns:
            Tuple of (shap_results_dict, output_directory_path)

        Example:
            >>> explainer = Explainer(runner)
            >>> shap_results, out_dir = explainer.explain(
            ...     {"config_path": "0001_abc123"},
            ...     X_test,
            ...     shap_params={"n_samples": 200, "visualizations": ["spectral", "summary"]}
            ... )
        """
        from nirs4all.visualization.analysis.shap import ShapAnalyzer

        logger.starting("Starting SHAP Explanation Analysis")

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
        dataset_config = self.runner.orchestrator._normalize_dataset(
            dataset, dataset_name
        )

        # Enable model capture mode
        self.runner.mode = "explain"
        self.runner._capture_model = True
        self.captured_model = None

        try:
            # Setup saver and manifest
            config, name = dataset_config.configs[0]
            run_dir = self._get_run_dir_from_prediction(prediction_obj)
            self.saver = SimulationSaver(run_dir, save_artifacts=self.runner.save_artifacts, save_charts=self.runner.save_charts)
            self.manifest_manager = ManifestManager(run_dir)

            # Load pipeline
            steps = self._prepare_replay(prediction_obj, dataset_config, verbose)
            dataset_obj = dataset_config.get_dataset(config, name)

            # Register with saver to allow artifact persistence
            self.saver.register(self.pipeline_uid)

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

            # Build executor using ExecutorBuilder
            executor = (ExecutorBuilder()
                .with_run_directory(run_dir)
                .with_verbose(verbose)
                .with_mode("explain")
                .with_save_artifacts(self.runner.save_artifacts)
                .with_save_charts(self.runner.save_charts)
                .with_continue_on_error(self.runner.continue_on_error)
                .with_show_spinner(self.runner.show_spinner)
                .with_plots_visible(plots_visible)
                .with_artifact_loader(self.artifact_loader)
                .with_saver(self.saver)
                .with_manifest_manager(self.manifest_manager)
                .build())

            # Create RuntimeContext with artifact_provider for V3 loading
            from nirs4all.pipeline.config.context import RuntimeContext

            # Create artifact_provider from artifact_loader for V3 artifact loading
            artifact_provider = None
            if self.artifact_loader:
                artifact_provider = LoaderArtifactProvider(loader=self.artifact_loader)

            runtime_context = RuntimeContext(
                saver=self.saver,
                manifest_manager=self.manifest_manager,
                artifact_loader=self.artifact_loader,
                artifact_provider=artifact_provider,
                step_runner=executor.step_runner,
                target_model=self.target_model,
                explainer=self.runner.explainer
            )

            executor.execute(steps, "explanation", dataset_obj, context, runtime_context, config_predictions)

            # Extract captured model
            if self.captured_model is None:
                raise ValueError("Failed to capture model. Model controller may not support capture.")

            model, controller = self.captured_model

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

            logger.debug(f"Output directory: {output_dir}")

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

            logger.success("SHAP explanation completed!")
            logger.artifact("visualization", path=output_dir)
            for viz in shap_params['visualizations']:
                logger.debug(f"  • {viz}.png")

            return shap_results, str(output_dir)

        finally:
            self.runner._capture_model = False

    def capture_model(self, model: Any, controller: Any):
        """Capture a model during pipeline execution for SHAP analysis.

        This method is called by the model controller during explain mode
        to capture the trained model instance.

        Args:
            model: Trained model instance
            controller: Controller that trained the model
        """
        self.captured_model = (model, controller)

    def _get_run_dir_from_prediction(self, prediction_obj: Union[Dict[str, Any], str]) -> Path:
        """Get run directory from prediction object.

        Args:
            prediction_obj: Model identifier

        Returns:
            Path to run directory

        Raises:
            ValueError: If no run directory can be found
        """
        if isinstance(prediction_obj, dict):
            if 'run_dir' in prediction_obj:
                return Path(prediction_obj['run_dir'])
            elif 'config_path' in prediction_obj:
                config_path = prediction_obj['config_path']
                dataset_name = Path(config_path).parts[0]
                # First try exact match
                exact_match = self.runner.orchestrator.runs_dir / dataset_name
                if exact_match.exists() and exact_match.is_dir():
                    return exact_match
                # Then try pattern match (for legacy directories with date prefix)
                matching_dirs = sorted(
                    self.runner.orchestrator.runs_dir.glob(f"*_{dataset_name}"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                if matching_dirs:
                    return matching_dirs[0]
                raise ValueError(f"No run directory found for dataset: {dataset_name}")

        # Fallback: use most recent run
        run_dirs = sorted(
            self.runner.orchestrator.runs_dir.glob("*"),
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
        """Prepare pipeline replay from saved configuration.

        Args:
            selection_obj: Model selection criteria
            dataset_config: Dataset configuration
            verbose: Verbosity level

        Returns:
            List of pipeline steps to execute

        Raises:
            ValueError: If pipeline_uid is missing or invalid
            FileNotFoundError: If pipeline configuration or manifest not found
        """
        import json

        # Get configuration path and target model
        config_path, target_model = self.saver.get_predict_targets(selection_obj)
        target_model.pop("y_pred", None)
        target_model.pop("y_true", None)

        self.config_path = config_path
        self.target_model = target_model
        self.runner.target_model = target_model  # Set on runner for controller access

        pipeline_uid = target_model.get('pipeline_uid')
        if not pipeline_uid:
            raise ValueError(
                "No pipeline_uid found in prediction metadata. "
                "This prediction was created with an older version of nirs4all. "
                "Please retrain the model."
            )

        self.pipeline_uid = pipeline_uid

        # Load pipeline configuration
        pipeline_dir_name = Path(config_path).parts[-1] if '/' in config_path or '\\' in config_path else config_path
        config_dir = self.saver.base_path / pipeline_dir_name
        pipeline_json = config_dir / "pipeline.json"

        logger.debug(f"Loading {pipeline_json}")

        if not pipeline_json.exists():
            raise FileNotFoundError(f"Pipeline not found: {pipeline_json}")

        with open(pipeline_json, 'r', encoding='utf-8') as f:
            pipeline_data = json.load(f)

        steps = pipeline_data["steps"] if isinstance(pipeline_data, dict) and "steps" in pipeline_data else pipeline_data

        # Load binaries from manifest
        manifest_path = self.saver.base_path / pipeline_uid / "manifest.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                f"Pipeline UID: {pipeline_uid}\n"
                f"The model artifacts may have been deleted or moved."
            )

        logger.info(f"Loading from manifest: {pipeline_uid}")
        manifest = self.manifest_manager.load_manifest(pipeline_uid)
        self.artifact_loader = ArtifactLoader.from_manifest(manifest, self.saver.base_path)

        return steps
