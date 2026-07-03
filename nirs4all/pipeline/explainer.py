"""Pipeline explainer - Handles SHAP explanation generation.

This module provides the Explainer class for generating model explanations
using SHAP (SHapley Additive exPlanations) on trained pipelines.

Supports the same prediction sources as the Predictor: prediction dicts,
folder paths, artifact IDs, bundles, and trace IDs are resolved via
PredictionResolver.
"""
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import numpy as np

from nirs4all.core.logging import get_logger

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.data.config import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.data.types import Layout
from nirs4all.pipeline.config.context import DataSelector, ExecutionContext, LoaderArtifactProvider, PipelineState, StepMetadata
from nirs4all.pipeline.execution.builder import ExecutorBuilder
from nirs4all.pipeline.explain_lineage import derive_relation_explain_lineage
from nirs4all.pipeline.storage.artifacts.artifact_loader import ArtifactLoader

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
        self.pipeline_uid: str | None = None
        self.artifact_loader: ArtifactLoader | None = None
        self.config_path: str | None = None
        self.target_model: dict[str, Any] | None = None
        self.captured_model: tuple[Any, Any] | None = None
        self._resolved_artifact_provider: Any | None = None
        self._direct_explain_model: Any | None = None
        self._relation_replay_manifest: dict[str, Any] | None = None

    def explain(
        self,
        prediction_obj: dict[str, Any] | str,
        dataset: DatasetConfigs | SpectroDataset | np.ndarray | tuple[np.ndarray, ...] | dict | list[dict] | str | list[str],
        dataset_name: str = "explain_dataset",
        shap_params: dict[str, Any] | None = None,
        verbose: int = 0,
        plots_visible: bool = True,
    ) -> tuple[dict[str, Any], str]:
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
        shap_options: dict[str, Any] = {} if shap_params is None else dict(shap_params)
        shap_options.setdefault("n_samples", 200)
        shap_options.setdefault("visualizations", ["spectral", "summary"])
        shap_options.setdefault("explainer_type", "auto")
        shap_options.setdefault("bin_size", 20)
        shap_options.setdefault("bin_stride", 10)
        shap_options.setdefault("bin_aggregation", "sum")

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

            if self._direct_explain_model is not None:
                model = self._direct_explain_model
                preferred_layout: Layout = "2d"
            else:
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
                    .with_figure_refs(self.runner._figure_refs)
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
                preferred_layout = cast(Layout, controller.get_preferred_layout())

            # Get test data
            test_context = context.with_partition("test")
            X_test = cast(np.ndarray, dataset_obj.x(test_context, layout=preferred_layout))
            y_test = dataset_obj.y(test_context)

            # Get feature names
            feature_names = None
            if hasattr(dataset_obj, "wavelengths") and dataset_obj.wavelengths is not None:
                feature_names = [f"\u03bb{w:.1f}" for w in dataset_obj.wavelengths]
            relation_manifest = self._relation_explain_manifest(dataset_obj)
            relation_lineage = derive_relation_explain_lineage(
                relation_manifest,
                feature_names=feature_names,
                n_features=self._feature_count(X_test),
            )
            if feature_names is None and relation_lineage is not None and relation_lineage.feature_names is not None:
                feature_names = relation_lineage.feature_names

            task_type = "classification" if dataset_obj.task_type and dataset_obj.task_type.is_classification else "regression"

            # Create output directory under workspace
            target_model = self.target_model or {}
            model_id = target_model.get("id", "unknown")
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
                n_background=shap_options["n_samples"],
                explainer_type=shap_options["explainer_type"],
                output_dir=str(output_dir),
                visualizations=shap_options["visualizations"],
                bin_size=shap_options["bin_size"],
                bin_stride=shap_options["bin_stride"],
                bin_aggregation=shap_options["bin_aggregation"],
                plots_visible=plots_visible,
            )

            shap_results["model_name"] = target_model.get("model_name", "unknown")
            shap_results["model_id"] = model_id
            shap_results["dataset_name"] = dataset_obj.name
            self._attach_relation_explain_lineage(shap_results, relation_manifest, X_test, feature_names)

            logger.success("SHAP explanation completed!")
            logger.artifact("visualization", path=output_dir)
            for viz in shap_options["visualizations"]:
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

    def _relation_explain_manifest(self, dataset_obj: SpectroDataset) -> dict[str, Any] | None:
        """Return relation materialization/replay metadata available for explanations."""
        materialization_manifest = getattr(dataset_obj, "_relation_materialization_manifest", None)
        if isinstance(materialization_manifest, dict):
            return materialization_manifest
        if isinstance(self._relation_replay_manifest, dict):
            return self._relation_replay_manifest
        return None

    @staticmethod
    def _feature_count(X: Any) -> int | None:
        """Return the feature width for 2D SHAP inputs when it can be inferred."""
        shape = getattr(X, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[-1])
        return None

    @staticmethod
    def _attach_relation_explain_lineage(
        shap_results: dict[str, Any],
        manifest: dict[str, Any] | None,
        X: Any,
        feature_names: list[str] | None,
    ) -> None:
        """Attach relation explanation metadata to SHAP results in-place."""
        result_feature_names = shap_results.get("feature_names")
        lineage_feature_names = result_feature_names if result_feature_names is not None else feature_names
        relation_lineage = derive_relation_explain_lineage(
            manifest,
            feature_names=lineage_feature_names,
            n_features=Explainer._feature_count(X),
        )
        if relation_lineage is None:
            return
        if result_feature_names is None and relation_lineage.feature_names is not None:
            shap_results["feature_names"] = relation_lineage.feature_names
        if relation_lineage.explanation_level is not None:
            shap_results["explanation_level"] = relation_lineage.explanation_level
        if relation_lineage.feature_lineage:
            shap_results["feature_lineage"] = relation_lineage.feature_lineage
        if relation_lineage.lineage_warning is not None:
            shap_results["lineage_warning"] = relation_lineage.lineage_warning

    def _prepare_replay(
        self,
        selection_obj: dict[str, Any] | str,
        dataset_config: DatasetConfigs,
        verbose: int = 0,
    ) -> list[Any]:
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
        target_model = {k: v for k, v in selection_obj.items() if k not in ("y_pred", "y_true", "X")} if isinstance(selection_obj, dict) else dict(resolved.target_model) if resolved.target_model else {}

        # Bundle/store sources expose the model step on the resolved prediction
        # but may not carry it inside target_model. Surface it so the model
        # controller can identify which step to capture for SHAP.
        if "step_idx" not in target_model and resolved.model_step_index is not None:
            target_model["step_idx"] = resolved.model_step_index

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
        self._direct_explain_model = None
        self._relation_replay_manifest = self._extract_relation_replay_manifest(resolved)

        self._direct_explain_model = self._resolve_direct_explain_model(resolved)

        # Create artifact loader from resolved artifact provider (for LoaderArtifactProvider)
        self.artifact_loader = None
        if resolved.artifact_provider and hasattr(resolved.artifact_provider, "artifact_loader"):
            self.artifact_loader = resolved.artifact_provider.artifact_loader

        return list(resolved.minimal_pipeline)

    @staticmethod
    def _resolve_direct_explain_model(resolved: Any) -> Any | None:
        """Return the native bundle composite model when replay is cosmetic only."""
        if not getattr(resolved, "train_pipeline", None):
            return None

        provider = getattr(resolved, "artifact_provider", None)
        step_index = getattr(resolved, "model_step_index", None)
        if provider is None or step_index is None or not hasattr(provider, "get_artifacts_for_step"):
            return None

        artifacts = provider.get_artifacts_for_step(step_index)
        if not artifacts:
            return None

        for artifact_id, model in artifacts:
            if "foldfinal" in str(artifact_id).lower() and hasattr(model, "predict"):
                return model

        for _artifact_id, model in artifacts:
            if hasattr(model, "predict"):
                return model

        return None

    @staticmethod
    def _extract_relation_replay_manifest(resolved: Any) -> dict[str, Any] | None:
        """Extract relation replay metadata from resolver outputs when available."""
        manifest = getattr(resolved, "manifest", None)
        if isinstance(manifest, dict):
            relation = manifest.get("relation_replay_manifest")
            if isinstance(relation, dict):
                return relation
            chain = manifest.get("chain")
            if isinstance(chain, dict):
                relation = chain.get("relation_replay_manifest")
                if isinstance(relation, dict):
                    return relation

        provider = getattr(resolved, "artifact_provider", None)
        loader = getattr(provider, "artifact_loader", None)
        for candidate in (loader, provider):
            relation = getattr(candidate, "relation_replay_manifest", None)
            if isinstance(relation, dict):
                return relation
        return None
