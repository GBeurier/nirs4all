"""
Minimal Pipeline Predictor - Execute minimal pipeline for prediction (V3).

This module provides the MinimalPredictor class which executes a minimal
pipeline extracted from an execution trace. It reuses existing controllers
in predict mode with artifact injection.

The MinimalPredictor is the key component of Phase 5: it ensures that
prediction only runs the required steps, not the entire original pipeline.

V3 Features:
    - Chain-based artifact identification using chain_path
    - ArtifactRecord metadata for branch/substep info (no ID parsing)
    - Support for multi-source pipelines via source_index

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
from nirs4all.pipeline.storage.artifacts.types import ArtifactRecord, ArtifactType
from nirs4all.pipeline.trace import MinimalPipeline, MinimalPipelineStep


logger = logging.getLogger(__name__)


class MinimalArtifactProvider(ArtifactProvider):
    """Artifact provider backed by a MinimalPipeline (V3).

    Provides artifacts from the minimal pipeline's artifact map, which
    contains StepArtifacts extracted from the execution trace.

    This provider uses V3 ArtifactRecord metadata (chain_path, branch_path,
    substep_index) instead of parsing V2-style artifact IDs.

    Attributes:
        minimal_pipeline: The source MinimalPipeline
        artifact_loader: ArtifactLoader for loading actual artifact objects
        target_sub_index: Filter artifacts by substep_index
        target_model_name: Filter artifacts by custom_name
    """

    def __init__(
        self,
        minimal_pipeline: MinimalPipeline,
        artifact_loader: Any,  # ArtifactLoader
        target_sub_index: Optional[int] = None,
        target_model_name: Optional[str] = None
    ):
        """Initialize minimal artifact provider.

        Args:
            minimal_pipeline: MinimalPipeline with artifact mappings
            artifact_loader: ArtifactLoader for loading artifact objects
            target_sub_index: Optional substep_index to filter model artifacts.
                              Used when a subpipeline contains multiple models
                              (e.g., [JaxMLPRegressor, nicon]) and we need to
                              load artifacts for a specific one.
            target_model_name: Optional model name to filter artifacts by.
                               Used as fallback when sub_index is not available
                               (e.g., for avg/w_avg predictions).
        """
        self.minimal_pipeline = minimal_pipeline
        self.artifact_loader = artifact_loader
        self.target_sub_index = target_sub_index
        self.target_model_name = target_model_name
        self._cache: Dict[str, Any] = {}

    def _get_record(self, artifact_id: str) -> Optional[ArtifactRecord]:
        """Get artifact record from loader.

        Args:
            artifact_id: Artifact ID to look up

        Returns:
            ArtifactRecord or None if not found
        """
        try:
            return self.artifact_loader.get_record(artifact_id)
        except (KeyError, AttributeError):
            return None

    def _get_branch_from_record(self, artifact_id: str) -> Optional[int]:
        """Get branch index from artifact record.

        Uses ArtifactRecord.branch_path for V3 artifacts.

        Args:
            artifact_id: Artifact ID to look up

        Returns:
            First branch index (0, 1, ...) if artifact is from a branch,
            None if artifact is shared (pre-branch).
        """
        record = self._get_record(artifact_id)
        if record is None:
            return None

        if record.branch_path and len(record.branch_path) > 0:
            return record.branch_path[0]
        return None

    def _get_substep_from_record(self, artifact_id: str) -> Optional[int]:
        """Get substep_index from artifact record.

        Uses ArtifactRecord.substep_index for V3 artifacts.

        Args:
            artifact_id: Artifact ID to look up

        Returns:
            Substep index or None if not present.
        """
        record = self._get_record(artifact_id)
        if record is None:
            return None
        return record.substep_index

    def _derive_operator_name(
        self, obj: Any, artifact_id: str, step_index: Optional[int] = None
    ) -> str:
        """Derive operator name from object class and artifact metadata.

        Reconstructs names like "MinMaxScaler_1" from the object's class
        and the substep_index from the artifact record. For y_processing
        steps, adds "y_" prefix to match the naming convention used
        during training (e.g., "y_MinMaxScaler_1").

        Args:
            obj: The loaded artifact object.
            artifact_id: The artifact ID.
            step_index: Optional step index for checking if y_processing.

        Returns:
            Operator name in format "{ClassName}_{substep_index}" or class name.
            For y_processing steps: "y_{ClassName}_{substep_index}".
        """
        class_name = obj.__class__.__name__

        # Get substep_index from artifact record (V3 approach)
        sub_index = self._get_substep_from_record(artifact_id)

        # Check if this is a y_processing step
        is_y_processing = False
        if step_index is not None and self.minimal_pipeline:
            minimal_step = self.minimal_pipeline.get_step(step_index)
            if minimal_step and minimal_step.operator_type == "y_processing":
                is_y_processing = True

        # Build name with optional y_ prefix
        prefix = "y_" if is_y_processing else ""
        if sub_index is not None:
            return f"{prefix}{class_name}_{sub_index}"
        return f"{prefix}{class_name}"

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
        branch_id: Optional[int] = None,
        source_index: Optional[int] = None,
        substep_index: Optional[int] = None
    ) -> List[Tuple[str, Any]]:
        """Get all artifacts for a step (V3).

        Filters artifacts by branch using the branch_path from ArtifactRecord.
        This is critical for multisource + branching reload, where branch
        substep artifacts are lumped together in the execution trace but can
        be distinguished by their artifact records.

        Returns tuples of (operator_name, artifact_object) where operator_name
        is derived from the object class and substep_index (e.g., "MinMaxScaler_1").
        This allows transformer controllers to look up artifacts by name.

        Args:
            step_index: 1-based step index
            branch_path: Optional branch path filter (e.g., [0] for branch 0)
            branch_id: Optional branch ID filter (used when branch_path not available)
            source_index: Optional source/dataset index filter for multi-source
            substep_index: Optional substep index filter for branch substeps

        Returns:
            List of (operator_name, artifact_object) tuples
        """
        step_artifacts = self.minimal_pipeline.get_artifacts_for_step(step_index)
        if not step_artifacts:
            return []

        # Determine target branch to filter by
        target_branch: Optional[int] = None
        if branch_path is not None and len(branch_path) > 0:
            target_branch = branch_path[0]
        elif branch_id is not None:
            target_branch = branch_id

        # Debug: log filtering params
        logger.debug(
            f"get_artifacts_for_step({step_index}): target_sub_index={self.target_sub_index}, "
            f"substep_index={substep_index}, model_step_index={self.minimal_pipeline.model_step_index}"
        )

        results = []
        for artifact_id in step_artifacts.artifact_ids:
            # Get artifact record for V3 metadata
            record = self._get_record(artifact_id)

            # Filter by branch if specified - use record.branch_path
            if target_branch is not None:
                artifact_branch = None
                if record and record.branch_path:
                    artifact_branch = record.branch_path[0] if len(record.branch_path) > 0 else None

                # Include artifact if:
                # - It has no branch (shared/pre-branch artifact)
                # - Its branch matches the target branch
                if artifact_branch is not None and artifact_branch != target_branch:
                    logger.debug(
                        f"Filtering artifact {artifact_id} (branch={artifact_branch}) "
                        f"- target branch is {target_branch}"
                    )
                    continue

            # Filter by substep_index if specified (for branch substeps)
            # This ensures each transformer controller gets only its own artifact
            if substep_index is not None and record is not None:
                artifact_substep = record.substep_index
                if artifact_substep is not None and artifact_substep != substep_index:
                    logger.debug(
                        f"Filtering artifact {artifact_id} (substep_index={artifact_substep}) "
                        f"- target substep is {substep_index}"
                    )
                    continue

            # Filter by source_index if specified (for multi-source datasets)
            # Each source gets its own transformer artifact, and we need to select
            # the correct one for the current source being processed
            if source_index is not None and record is not None:
                artifact_source = record.source_index
                if artifact_source is not None and artifact_source != source_index:
                    logger.debug(
                        f"Filtering artifact {artifact_id} (source_index={artifact_source}) "
                        f"- target source is {source_index}"
                    )
                    continue

            # Filter model artifacts to only load the correct model in subpipelines
            # This is critical when a list like [JaxMLPRegressor, nicon] creates
            # artifacts with different substep_index values
            should_filter = self.target_sub_index is not None or self.target_model_name is not None

            if should_filter and record is not None:
                # Check if this artifact is a model by querying its type from the record
                is_model_artifact = record.artifact_type in (
                    ArtifactType.MODEL, ArtifactType.META_MODEL
                )

                if is_model_artifact:
                    # Strategy 1: Filter by substep_index if available
                    if self.target_sub_index is not None:
                        artifact_sub_index = record.substep_index
                        if artifact_sub_index is not None and artifact_sub_index != self.target_sub_index:
                            logger.debug(
                                f"Filtering artifact {artifact_id} (substep_index={artifact_sub_index}) "
                                f"- target sub_index is {self.target_sub_index}"
                            )
                            continue
                    # Strategy 2: Filter by model name (fallback for avg/w_avg predictions)
                    elif self.target_model_name is not None:
                        # Check if artifact's custom_name matches target model name
                        if record.custom_name and record.custom_name != self.target_model_name:
                            logger.debug(
                                f"Filtering artifact {artifact_id} (custom_name={record.custom_name}) "
                                f"- target model is {self.target_model_name}"
                            )
                            continue

            obj = self._load_artifact(artifact_id)
            if obj is not None:
                # Derive operator name from object class and artifact sub_index
                # This allows transformer controllers to look up by name
                # Pass step_index to check if y_processing (needs y_ prefix)
                operator_name = self._derive_operator_name(obj, artifact_id, step_index)
                # Get substep_index for sorting
                artifact_substep = record.substep_index if record else None
                results.append((operator_name, obj, artifact_substep))

        # Sort by substep_index to ensure artifacts are returned in the same order
        # they were created during training. This is critical for multi-source
        # pipelines with feature_augmentation where transformers are loaded
        # by index position.
        results.sort(key=lambda x: (x[2] if x[2] is not None else float('inf')))

        # Remove substep_index from results (keep only operator_name, obj tuples)
        results = [(name, obj) for name, obj, _ in results]

        logger.debug(
            f"get_artifacts_for_step({step_index}, branch_path={branch_path}) "
            f"-> {len(results)} artifacts from {len(step_artifacts.artifact_ids)} total"
        )
        return results

    def get_fold_artifacts(
        self,
        step_index: int,
        branch_path: Optional[List[int]] = None
    ) -> List[Tuple[int, Any]]:
        """Get all fold-specific artifacts for a step.

        Filters by target_sub_index when set (for subpipelines with multiple models).
        When target_sub_index is set, looks through all artifact_ids instead of
        fold_artifact_ids because fold_artifact_ids only stores the last model's
        artifacts when multiple models exist in a subpipeline.

        Args:
            step_index: 1-based step index
            branch_path: Optional branch path filter

        Returns:
            List of (fold_id, artifact_object) tuples, sorted by fold_id
        """
        step_artifacts = self.minimal_pipeline.get_artifacts_for_step(step_index)
        if not step_artifacts:
            return []

        results = []

        # When target_sub_index is set, we need to search through all artifact_ids
        # because fold_artifact_ids gets overwritten when multiple models exist in a subpipeline
        if self.target_sub_index is not None:
            for artifact_id in step_artifacts.artifact_ids:
                record = self._get_record(artifact_id)
                if record is None:
                    continue

                # Check if this is a model artifact with matching substep_index
                is_model_artifact = record.artifact_type in (
                    ArtifactType.MODEL, ArtifactType.META_MODEL
                )
                if not is_model_artifact:
                    continue

                artifact_sub_index = record.substep_index
                if artifact_sub_index is not None and artifact_sub_index != self.target_sub_index:
                    logger.debug(
                        f"Filtering artifact {artifact_id} (substep_index={artifact_sub_index}) "
                        f"- target sub_index is {self.target_sub_index}"
                    )
                    continue

                # Extract fold_id from artifact_id (format: pipeline_uid$hash:fold_id)
                fold_id = None
                if ':' in artifact_id:
                    parts = artifact_id.rsplit(':', 1)
                    if len(parts) == 2:
                        try:
                            fold_id = int(parts[1])
                        except ValueError:
                            pass

                if fold_id is None:
                    continue

                obj = self._load_artifact(artifact_id)
                if obj is not None:
                    results.append((fold_id, obj))
        else:
            # Standard case: use fold_artifact_ids
            if not step_artifacts.fold_artifact_ids:
                return []

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

    def _get_substep_from_artifact(self, artifact_id: str) -> Optional[int]:
        """Get substep_index from artifact record (V3).

        Args:
            artifact_id: Artifact ID to look up.

        Returns:
            substep_index or None if not found.
        """
        try:
            record = self.artifact_loader.get_record(artifact_id)
            if record is not None:
                return record.substep_index
        except (KeyError, AttributeError):
            pass
        return None

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

        logger.info(f"Minimal prediction: {minimal_pipeline.get_step_count()} steps")

        # Extract target_sub_index from model_artifact_id if present
        # This is critical for subpipelines with multiple models
        target_sub_index = None
        if target_model:
            model_artifact_id = target_model.get('model_artifact_id')
            if model_artifact_id:
                target_sub_index = self._get_substep_from_artifact(model_artifact_id)

        # Create artifact provider from minimal pipeline
        artifact_provider = MinimalArtifactProvider(
            minimal_pipeline=minimal_pipeline,
            artifact_loader=self.artifact_loader,
            target_sub_index=target_sub_index
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
            .with_save_artifacts(False)
            .with_save_charts(False)
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
                logger.success(f"Prediction complete: {len(y_pred)} samples")
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
