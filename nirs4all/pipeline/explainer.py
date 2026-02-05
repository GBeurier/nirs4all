"""Pipeline explainer - Handles SHAP explanation generation.

This module provides the Explainer class for generating model explanations
using SHAP (SHapley Additive exPlanations) on trained pipelines.

Supports the same prediction sources as the Predictor: prediction dicts,
folder paths, artifact IDs, bundles, and trace IDs are resolved via
PredictionResolver.
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

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)


class Explainer:
    """Handles SHAP explanation generation for trained models.

    This class manages the explanation workflow: resolving a prediction source
    via PredictionResolver, replaying the pipeline to capture the trained model,
    and generating SHAP explanations with visualizations.

    Attributes:
        runner: Parent PipelineRunner instance.
        pipeline_uid: Unique identifier for the pipeline.
        artifact_loader: Loader for trained model artifacts.
        config_path: Path to the pipeline configuration.
        target_model: Metadata for the target model.
        captured_model: Tuple of (model, controller) captured during replay.
    """

    def __init__(self, runner: "PipelineRunner"):
        """Initialize explainer.

        Args:
            runner: Parent PipelineRunner instance.
        """
        self.runner = runner
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
        plots_visible: bool = True,
    ) -> Tuple[Dict[str, Any], str]:
        """Generate SHAP explanations for a saved model.

        Args:
            prediction_obj: Model identifier (dict with config_path or prediction ID).
            dataset: Dataset to explain on.
            dataset_name: Name for the dataset.
            shap_params: SHAP configuration parameters.
            verbose: Verbosity level.
            plots_visible: Whether to display plots interactively.

        Returns:
            Tuple of (shap_results_dict, output_directory_path).

        Example:
            >>> explainer = Explainer(runner)
            >>> shap_results, out_dir = explainer.explain(
            ...     {"config_path": "0001_abc123"},
            ...     X_test,
            ...     shap_params={"n_samples": 200, "visualizations": ["spectral", "summary"]},
            ... )
        """
        from nirs4all.visualization.analysis.shap import ShapAnalyzer

        logger.starting("Starting SHAP Explanation Analysis")

        # Setup SHAP parameters
        if shap_params is None:
            shap_params = {}
        shap_params.setdefault("n_samples", 200)
        shap_params.setdefault("visualizations", ["spectral", "summary"])
        shap_params.setdefault("explainer_type", "auto")
        shap_params.setdefault("bin_size", 20)
        shap_params.setdefault("bin_stride", 10)
        shap_params.setdefault("bin_aggregation", "sum")

        # Normalize dataset
        dataset_config = self.runner.orchestrator._normalize_dataset(dataset, dataset_name)

        # Enable model capture mode
        self.runner.mode = "explain"
        self.runner._capture_model = True
        self.captured_model = None

        try:
            # Resolve prediction source via PredictionResolver
            steps = self._prepare_replay(prediction_obj, dataset_config, verbose)

            config, name = dataset_config.configs[0]
            dataset_obj = dataset_config.get_dataset(config, name)

            # Execute pipeline to capture model
            context = ExecutionContext(
                selector=DataSelector(
                    partition=None,
                    processing=[["raw"]] * dataset_obj.features_sources(),
                    layout="2d",
                    concat_source=True,
                ),
                state=PipelineState(y_processing="numeric", step_number=0, mode="explain"),
                metadata=StepMetadata(),
            )

            config_predictions = Predictions()

            # Build executor using ExecutorBuilder
            executor = (
                ExecutorBuilder()
                .with_workspace(self.runner.workspace_path)
                .with_verbose(verbose)
                .with_mode("explain")
                .with_save_artifacts(self.runner.save_artifacts)
                .with_save_charts(self.runner.save_charts)
                .with_continue_on_error(self.runner.continue_on_error)
                .with_show_spinner(self.runner.show_spinner)
                .with_plots_visible(plots_visible)
                .with_artifact_loader(self.artifact_loader)
                .build()
            )

            # Create RuntimeContext with artifact_provider for V3 loading
            from nirs4all.pipeline.config.context import RuntimeContext

            artifact_provider = self._resolved_artifact_provider
            if artifact_provider is None and self.artifact_loader:
                artifact_provider = LoaderArtifactProvider(loader=self.artifact_loader)

            runtime_context = RuntimeContext(
                artifact_loader=self.artifact_loader,
                artifact_provider=artifact_provider,
                step_runner=executor.step_runner,
                target_model=self.target_model,
                explainer=self.runner.explainer,
            )

            executor.execute(steps, "explanation", dataset_obj, context, runtime_context, config_predictions)

            # Extract captured model
            if self.captured_model is None:
                raise ValueError("Failed to capture model. Model controller may not support capture.")

            model, controller = self.captured_model

            # Get test data
            test_context = context.with_partition("test")
            X_test = dataset_obj.x(test_context, layout=controller.get_preferred_layout())
            y_test = dataset_obj.y(test_context)

            # Get feature names
            feature_names = None
            if hasattr(dataset_obj, "wavelengths") and dataset_obj.wavelengths is not None:
                feature_names = [f"\u03bb{w:.1f}" for w in dataset_obj.wavelengths]

            task_type = "classification" if dataset_obj.task_type and dataset_obj.task_type.is_classification else "regression"

            # Create output directory under workspace
            model_id = self.target_model.get("id", "unknown")
            output_dir = self.runner.workspace_path / "explanations" / (self.pipeline_uid or "unknown") / model_id
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
                n_background=shap_params["n_samples"],
                explainer_type=shap_params["explainer_type"],
                output_dir=str(output_dir),
                visualizations=shap_params["visualizations"],
                bin_size=shap_params["bin_size"],
                bin_stride=shap_params["bin_stride"],
                bin_aggregation=shap_params["bin_aggregation"],
                plots_visible=plots_visible,
            )

            shap_results["model_name"] = self.target_model.get("model_name", "unknown")
            shap_results["model_id"] = model_id
            shap_results["dataset_name"] = dataset_obj.name

            logger.success("SHAP explanation completed!")
            logger.artifact("visualization", path=output_dir)
            for viz in shap_params["visualizations"]:
                logger.debug(f"  - {viz}.png")

            return shap_results, str(output_dir)

        finally:
            self.runner._capture_model = False

    def capture_model(self, model: Any, controller: Any) -> None:
        """Capture a model during pipeline execution for SHAP analysis.

        This method is called by the model controller during explain mode
        to capture the trained model instance.

        Args:
            model: Trained model instance.
            controller: Controller that trained the model.
        """
        self.captured_model = (model, controller)

    def _prepare_replay(
        self,
        selection_obj: Union[Dict[str, Any], str],
        dataset_config: DatasetConfigs,
        verbose: int = 0,
    ) -> List[Any]:
        """Prepare pipeline replay from saved configuration using PredictionResolver.

        Uses PredictionResolver to resolve the prediction source, then extracts
        the pipeline steps, artifact loader, and target model metadata.

        Args:
            selection_obj: Model selection criteria (dict, string, etc.).
            dataset_config: Dataset configuration.
            verbose: Verbosity level.

        Returns:
            List of pipeline steps to execute.

        Raises:
            ValueError: If pipeline_uid is missing or invalid.
            FileNotFoundError: If pipeline configuration or manifest not found.
        """
        from nirs4all.pipeline.resolver import PredictionResolver

        resolver = PredictionResolver(
            workspace_path=self.runner.workspace_path,
            runs_dir=self.runner.workspace_path,
            store=self.runner.store,
        )
        resolved = resolver.resolve(selection_obj, verbose=verbose)

        # Extract target model metadata
        if isinstance(selection_obj, dict):
            target_model = {k: v for k, v in selection_obj.items() if k not in ("y_pred", "y_true", "X")}
        else:
            target_model = dict(resolved.target_model) if resolved.target_model else {}

        self.target_model = target_model
        self.runner.target_model = target_model

        pipeline_uid = resolved.pipeline_uid
        if not pipeline_uid:
            raise ValueError(
                "No pipeline_uid found in prediction metadata. "
                "This prediction was created with an older version of nirs4all. "
                "Please retrain the model."
            )

        self.pipeline_uid = pipeline_uid
        self.config_path = pipeline_uid

        # Store the resolved artifact provider for use in explain
        self._resolved_artifact_provider = resolved.artifact_provider

        # Create artifact loader from resolved artifact provider (for LoaderArtifactProvider)
        self.artifact_loader = None
        if resolved.artifact_provider and hasattr(resolved.artifact_provider, "artifact_loader"):
            self.artifact_loader = resolved.artifact_provider.artifact_loader

        return list(resolved.minimal_pipeline)
