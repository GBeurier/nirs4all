"""
Minimal Pipeline Predictor - Execute minimal pipeline for prediction.

This module provides the MinimalPredictor class which executes a minimal
pipeline extracted from an execution trace. It reuses existing controllers
in predict mode with artifact injection.

The MinimalPredictor is the key component of Phase 5: it ensures that
prediction only runs the required steps, not the entire original pipeline.

Design Principles:
    1. Controller-Agnostic: Uses existing controllers without hardcoding types
    2. Minimal Execution: Only runs steps needed for the specific prediction
    3. Artifact Injection: Provides pre-loaded artifacts to controllers
    4. Deterministic: Same minimal pipeline -> same prediction

Usage:
    >>> from nirs4all.pipeline.minimal_predictor import MinimalPredictor
    >>> from nirs4all.pipeline.trace import TraceBasedExtractor
    >>>
    >>> # Extract minimal pipeline
    >>> extractor = TraceBasedExtractor()
    >>> minimal = extractor.extract(trace, full_pipeline_steps)
    >>>
    >>> # Predict using minimal pipeline
    >>> predictor = MinimalPredictor(artifact_loader, run_dir)
    >>> y_pred, predictions = predictor.predict(minimal, dataset)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.config.context import (
    DataSelector,
    ExecutionContext,
    PipelineState,
    RuntimeContext,
    StepMetadata,
    ArtifactProvider,
    LoaderArtifactProvider,
)
from nirs4all.pipeline.trace import MinimalPipeline, MinimalPipelineStep


logger = logging.getLogger(__name__)


class MinimalArtifactProvider(ArtifactProvider):
    """Artifact provider backed by a MinimalPipeline.

    Provides artifacts from the minimal pipeline's artifact map, which
    contains StepArtifacts extracted from the execution trace.

    This provider is used during minimal pipeline prediction to inject
    the correct artifacts into controllers by step index.

    Attributes:
        minimal_pipeline: The source MinimalPipeline
        artifact_loader: ArtifactLoader for loading actual artifact objects
    """

    def __init__(
        self,
        minimal_pipeline: MinimalPipeline,
        artifact_loader: Any  # ArtifactLoader
    ):
        """Initialize minimal artifact provider.

        Args:
            minimal_pipeline: MinimalPipeline with artifact mappings
            artifact_loader: ArtifactLoader for loading artifact objects
        """
        self.minimal_pipeline = minimal_pipeline
        self.artifact_loader = artifact_loader
        self._cache: Dict[str, Any] = {}

    def get_artifact(
        self,
        step_index: int,
        fold_id: Optional[int] = None
    ) -> Optional[Any]:
        """Get a single artifact for a step.

        Args:
            step_index: 1-based step index
            fold_id: Optional fold ID for fold-specific artifacts

        Returns:
            Artifact object or None if not found
        """
        step_artifacts = self.minimal_pipeline.get_artifacts_for_step(step_index)
        if not step_artifacts:
            return None

        # Try fold-specific artifact first
        if fold_id is not None and step_artifacts.fold_artifact_ids:
            artifact_id = step_artifacts.fold_artifact_ids.get(fold_id)
            if artifact_id:
                return self._load_artifact(artifact_id)

        # Try primary artifact
        if step_artifacts.primary_artifact_id:
            return self._load_artifact(step_artifacts.primary_artifact_id)

        # Try first artifact
        if step_artifacts.artifact_ids:
            return self._load_artifact(step_artifacts.artifact_ids[0])

        return None

    def get_artifacts_for_step(
        self,
        step_index: int,
        branch_path: Optional[List[int]] = None,
        branch_id: Optional[int] = None
    ) -> List[Tuple[str, Any]]:
        """Get all artifacts for a step.

        Args:
            step_index: 1-based step index
            branch_path: Optional branch path filter
            branch_id: Optional branch ID filter (used when branch_path not available)

        Returns:
            List of (artifact_id, artifact_object) tuples
        """
        step_artifacts = self.minimal_pipeline.get_artifacts_for_step(step_index)
        if not step_artifacts:
            return []

        # Get the step to check branch matching
        step = self.minimal_pipeline.get_step(step_index)

        results = []
        for artifact_id in step_artifacts.artifact_ids:
            # Filter by branch path if specified
            if branch_path is not None and step:
                # Check if artifact belongs to this branch
                if step.branch_path and step.branch_path != branch_path:
                    continue
            elif branch_id is not None and step:
                # Check by branch_id (first element of branch_path)
                if step.branch_path and step.branch_path[0] != branch_id:
                    continue

            obj = self._load_artifact(artifact_id)
            if obj is not None:
                results.append((artifact_id, obj))

        return results

    def get_fold_artifacts(
        self,
        step_index: int,
        branch_path: Optional[List[int]] = None
    ) -> List[Tuple[int, Any]]:
        """Get all fold-specific artifacts for a step.

        Args:
            step_index: 1-based step index
            branch_path: Optional branch path filter

        Returns:
            List of (fold_id, artifact_object) tuples, sorted by fold_id
        """
        step_artifacts = self.minimal_pipeline.get_artifacts_for_step(step_index)
        if not step_artifacts or not step_artifacts.fold_artifact_ids:
            return []

        results = []
        for fold_id, artifact_id in step_artifacts.fold_artifact_ids.items():
            obj = self._load_artifact(artifact_id)
            if obj is not None:
                results.append((fold_id, obj))

        return sorted(results, key=lambda x: x[0])

    def has_artifacts_for_step(self, step_index: int) -> bool:
        """Check if artifacts exist for a step.

        Args:
            step_index: 1-based step index

        Returns:
            True if artifacts are available for this step
        """
        step_artifacts = self.minimal_pipeline.get_artifacts_for_step(step_index)
        return step_artifacts is not None and len(step_artifacts.artifact_ids) > 0

    def get_fold_weights(self) -> Dict[int, float]:
        """Get fold weights for CV ensemble averaging.

        Returns:
            Dictionary mapping fold_id to weight
        """
        return dict(self.minimal_pipeline.fold_weights or {})

    def _load_artifact(self, artifact_id: str) -> Optional[Any]:
        """Load an artifact by ID with caching.

        Args:
            artifact_id: Artifact ID to load

        Returns:
            Loaded artifact object or None on error
        """
        if artifact_id in self._cache:
            return self._cache[artifact_id]

        try:
            obj = self.artifact_loader.load_by_id(artifact_id)
            self._cache[artifact_id] = obj
            return obj
        except (KeyError, FileNotFoundError) as e:
            logger.warning(f"Failed to load artifact {artifact_id}: {e}")
            return None


class MinimalPredictor:
    """Execute minimal pipeline for prediction.

    This class takes a MinimalPipeline (extracted from an ExecutionTrace)
    and executes only the required steps using existing controllers with
    artifact injection.

    The MinimalPredictor achieves the Phase 5 goal of "execute only needed
    steps" by:
    1. Using the minimal pipeline's step list (not the full original pipeline)
    2. Injecting pre-loaded artifacts via ArtifactProvider
    3. Running controllers in predict mode

    Attributes:
        artifact_loader: ArtifactLoader for loading artifacts
        run_dir: Path to run directory
        saver: Optional SimulationSaver for outputs
        manifest_manager: Optional ManifestManager
        verbose: Verbosity level

    Example:
        >>> predictor = MinimalPredictor(artifact_loader, run_dir)
        >>> y_pred, predictions = predictor.predict(minimal_pipeline, dataset)
    """

    def __init__(
        self,
        artifact_loader: Any,  # ArtifactLoader
        run_dir: Union[str, Path],
        saver: Any = None,
        manifest_manager: Any = None,
        verbose: int = 0
    ):
        """Initialize minimal predictor.

        Args:
            artifact_loader: ArtifactLoader for loading artifacts
            run_dir: Path to run directory
            saver: Optional SimulationSaver for outputs
            manifest_manager: Optional ManifestManager
            verbose: Verbosity level
        """
        self.artifact_loader = artifact_loader
        self.run_dir = Path(run_dir)
        self.saver = saver
        self.manifest_manager = manifest_manager
        self.verbose = verbose

    def predict(
        self,
        minimal_pipeline: MinimalPipeline,
        dataset: SpectroDataset,
        target_model: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Predictions]:
        """Execute minimal pipeline and return predictions.

        Runs only the steps in the minimal pipeline, using pre-loaded
        artifacts from the execution trace.

        Args:
            minimal_pipeline: MinimalPipeline to execute
            dataset: Dataset to predict on
            target_model: Optional target model metadata for filtering

        Returns:
            Tuple of (y_pred array, Predictions object)
        """
        from nirs4all.pipeline.execution.builder import ExecutorBuilder
        from nirs4all.utils.emoji import ROCKET, CHECK

        if self.verbose > 0:
            print(f"{ROCKET} Minimal prediction: {minimal_pipeline.get_step_count()} steps")

        # Create artifact provider from minimal pipeline
        artifact_provider = MinimalArtifactProvider(
            minimal_pipeline=minimal_pipeline,
            artifact_loader=self.artifact_loader
        )

        # Initialize context for prediction
        context = ExecutionContext(
            selector=DataSelector(
                partition="all",
                processing=[["raw"]] * dataset.features_sources(),
                layout="2d",
                concat_source=True
            ),
            state=PipelineState(
                y_processing="numeric",
                step_number=0,
                mode="predict"
            ),
            metadata=StepMetadata()
        )

        # Build executor
        executor = (ExecutorBuilder()
            .with_run_directory(self.run_dir)
            .with_verbose(self.verbose)
            .with_mode("predict")
            .with_save_files(False)
            .with_continue_on_error(False)
            .with_show_spinner(False)
            .with_plots_visible(False)
            .with_artifact_loader(self.artifact_loader)
            .with_saver(self.saver)
            .with_manifest_manager(self.manifest_manager)
            .build())

        # Create RuntimeContext with artifact_provider
        runtime_context = RuntimeContext(
            saver=self.saver,
            manifest_manager=self.manifest_manager,
            artifact_loader=self.artifact_loader,
            artifact_provider=artifact_provider,
            step_runner=executor.step_runner,
            target_model=target_model
        )

        # Extract step configs from minimal pipeline
        steps = [step.step_config for step in minimal_pipeline.steps]

        # Execute minimal pipeline
        predictions = Predictions()

        executor.execute_minimal(
            steps=steps,
            minimal_pipeline=minimal_pipeline,
            dataset=dataset,
            context=context,
            runtime_context=runtime_context,
            prediction_store=predictions
        )

        # Get y_pred from predictions
        if predictions.num_predictions > 0:
            # Filter by target model if specified
            if target_model:
                candidates = predictions.filter_predictions(**{
                    k: v for k, v in target_model.items()
                    if k in ("model_name", "step_idx", "fold_id", "branch_id")
                })
            else:
                candidates = predictions.to_dicts()

            # Get non-empty predictions
            non_empty = [p for p in candidates if len(p.get("y_pred", [])) > 0]
            if non_empty:
                y_pred = non_empty[0]["y_pred"]
                if self.verbose > 0:
                    print(f"{CHECK} Prediction complete: {len(y_pred)} samples")
                return np.array(y_pred), predictions

        # Return empty if no predictions
        return np.array([]), predictions

    def predict_with_fold_ensemble(
        self,
        minimal_pipeline: MinimalPipeline,
        dataset: SpectroDataset,
        fold_strategy: str = "weighted_average"
    ) -> Tuple[np.ndarray, Predictions]:
        """Execute minimal pipeline with fold ensemble averaging.

        For cross-validation models, runs prediction with each fold model
        and combines results according to fold_strategy.

        Args:
            minimal_pipeline: MinimalPipeline to execute
            dataset: Dataset to predict on
            fold_strategy: How to combine folds ("average", "weighted_average")

        Returns:
            Tuple of (y_pred array, Predictions object)
        """
        fold_weights = minimal_pipeline.fold_weights or {}

        if not fold_weights:
            # No folds, regular prediction
            return self.predict(minimal_pipeline, dataset)

        # Get predictions for each fold
        fold_predictions: Dict[int, np.ndarray] = {}

        for fold_id in sorted(fold_weights.keys()):
            target_model = {"fold_id": fold_id}
            y_pred, _ = self.predict(minimal_pipeline, dataset, target_model)
            if len(y_pred) > 0:
                fold_predictions[fold_id] = y_pred

        if not fold_predictions:
            return np.array([]), Predictions()

        # Combine fold predictions
        fold_arrays = list(fold_predictions.values())
        fold_ids = list(fold_predictions.keys())

        if fold_strategy == "weighted_average" and fold_weights:
            # Weighted average
            weights = np.array([fold_weights.get(fid, 1.0) for fid in fold_ids])
            weights = weights / weights.sum()  # Normalize
            y_pred_combined = np.average(fold_arrays, axis=0, weights=weights)
        else:
            # Simple average
            y_pred_combined = np.mean(fold_arrays, axis=0)

        # Create combined prediction record
        predictions = Predictions()
        predictions.add_prediction(
            dataset_name="prediction",
            model_name=minimal_pipeline.preprocessing_chain,
            step_idx=minimal_pipeline.model_step_index or 0,
            fold_id="ensemble",
            y_pred=y_pred_combined
        )

        return y_pred_combined, predictions

    def validate_minimal_pipeline(
        self,
        minimal_pipeline: MinimalPipeline
    ) -> Tuple[bool, List[str]]:
        """Validate that minimal pipeline can be executed.

        Checks that:
        - All step configs are present
        - All required artifacts are loadable
        - Model step is included

        Args:
            minimal_pipeline: MinimalPipeline to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check step configs
        for step in minimal_pipeline.steps:
            if step.step_config is None:
                issues.append(f"Step {step.step_index} has no config")

        # Check model step
        if minimal_pipeline.model_step_index is None:
            issues.append("No model step in minimal pipeline")
        elif not minimal_pipeline.has_step(minimal_pipeline.model_step_index):
            issues.append(
                f"Model step {minimal_pipeline.model_step_index} not in pipeline"
            )

        # Check artifacts are loadable
        for step_index, step_artifacts in minimal_pipeline.artifact_map.items():
            for artifact_id in step_artifacts.artifact_ids:
                try:
                    self.artifact_loader.load_by_id(artifact_id)
                except (KeyError, FileNotFoundError):
                    issues.append(
                        f"Artifact {artifact_id} for step {step_index} not loadable"
                    )

        is_valid = len(issues) == 0
        return is_valid, issues
