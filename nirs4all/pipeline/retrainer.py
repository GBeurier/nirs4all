"""Pipeline retrainer - Handles retraining, transfer learning, and fine-tuning.

This module provides the Retrainer class for retraining pipelines with different
modes: full retrain, transfer learning (reuse preprocessing), and fine-tuning.

Phase 7 Implementation:
    This module enables retraining trained pipelines on new data with various modes:
    - full: Train everything from scratch (same pipeline structure)
    - transfer: Use existing preprocessing artifacts, train new model
    - finetune: Continue training existing model with new data

Design Principles:
    1. Controller-Agnostic: Works with any controller type via per-step mode control
    2. Reuses Existing Infrastructure: Leverages resolver, artifact provider, executor
    3. Composable: Same infrastructure for all retrain modes
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from nirs4all.core.logging import get_logger
from nirs4all.data.config import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.config.context import (
    ExecutionContext,
    DataSelector,
    PipelineState,
    StepMetadata,
    RuntimeContext,
    ArtifactProvider,
    LoaderArtifactProvider,
)
from nirs4all.pipeline.trace import (
    ExecutionTrace,
    TraceBasedExtractor,
    MinimalPipeline,
    StepExecutionMode,
)
from nirs4all.pipeline.storage.artifacts.artifact_loader import ArtifactLoader
from nirs4all.pipeline.storage.manifest_manager import ManifestManager
from nirs4all.pipeline.resolver import PredictionResolver, ResolvedPrediction


logger = get_logger(__name__)


class RetrainMode(str, Enum):
    """Mode of retraining operation.

    Attributes:
        FULL: Train everything from scratch (same pipeline structure)
        TRANSFER: Use existing preprocessing artifacts, train new model
        FINETUNE: Continue training existing model with new data
    """

    FULL = "full"
    TRANSFER = "transfer"
    FINETUNE = "finetune"

    def __str__(self) -> str:
        return self.value


@dataclass
class StepMode:
    """Mode override for a specific step during retraining.

    Enables fine-grained control over which steps train vs. use existing artifacts.

    Attributes:
        step_index: 1-based step index to apply this mode to
        mode: How to execute this step ('train', 'predict', 'skip')
        artifact_id: For 'predict' mode, specific artifact to use
        kwargs: Additional step-specific parameters (e.g., epochs for finetune)
    """

    step_index: int
    mode: str = "train"  # 'train', 'predict', 'skip'
    artifact_id: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def is_train(self) -> bool:
        """Check if this step should train.

        Returns:
            True if step should be trained
        """
        return self.mode == "train"

    def is_predict(self) -> bool:
        """Check if this step should use existing artifacts.

        Returns:
            True if step should use existing artifacts (predict mode)
        """
        return self.mode == "predict"


@dataclass
class RetrainConfig:
    """Configuration for retraining operation.

    Attributes:
        mode: Overall retrain mode (full, transfer, finetune)
        step_modes: Per-step mode overrides (optional, for fine-grained control)
        new_model: Optional new model to use instead of original (for transfer)
        epochs: Optional epochs for fine-tuning
        learning_rate: Optional learning rate for fine-tuning
        freeze_layers: Optional list of layers to freeze during fine-tuning
        metadata: Additional metadata for the retrain operation
    """

    mode: RetrainMode = RetrainMode.FULL
    step_modes: List[StepMode] = field(default_factory=list)
    new_model: Optional[Any] = None
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    freeze_layers: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_step_mode(self, step_index: int) -> Optional[StepMode]:
        """Get mode override for a specific step.

        Args:
            step_index: 1-based step index

        Returns:
            StepMode if override exists, None otherwise
        """
        for sm in self.step_modes:
            if sm.step_index == step_index:
                return sm
        return None

    def should_train_step(self, step_index: int, is_model: bool = False) -> bool:
        """Determine if a step should train based on mode and overrides.

        Args:
            step_index: 1-based step index
            is_model: Whether this is the model step

        Returns:
            True if step should train
        """
        # Check for explicit override
        override = self.get_step_mode(step_index)
        if override:
            return override.is_train()

        # Apply mode defaults
        if self.mode == RetrainMode.FULL:
            return True
        elif self.mode == RetrainMode.TRANSFER:
            # Transfer: only train model, use existing preprocessing
            return is_model
        elif self.mode == RetrainMode.FINETUNE:
            # Finetune: continue training model only
            return is_model

        return True


class RetrainArtifactProvider(ArtifactProvider):
    """Artifact provider for retraining that respects step modes.

    Provides artifacts only for steps that should use existing artifacts
    (i.e., mode='predict'), while returning None for steps that should train.

    Attributes:
        base_provider: Underlying artifact provider
        retrain_config: Configuration determining which steps use artifacts
        trace: Execution trace for step type detection
    """

    def __init__(
        self,
        base_provider: ArtifactProvider,
        retrain_config: RetrainConfig,
        trace: Optional[ExecutionTrace] = None
    ):
        """Initialize retrain artifact provider.

        Args:
            base_provider: The underlying artifact provider
            retrain_config: Configuration for the retrain operation
            trace: Optional execution trace for step type detection
        """
        self.base_provider = base_provider
        self.retrain_config = retrain_config
        self.trace = trace

    def _should_provide_artifact(self, step_index: int) -> bool:
        """Determine if artifacts should be provided for this step.

        Returns True only if the step should use existing artifacts (predict mode).

        Args:
            step_index: 1-based step index

        Returns:
            True if artifacts should be provided
        """
        # Check for explicit override
        override = self.retrain_config.get_step_mode(step_index)
        if override:
            return override.is_predict()

        # Determine if this is a model step from trace
        is_model = False
        if self.trace:
            step = self.trace.get_step(step_index)
            if step:
                is_model = step.operator_type in ("model", "meta_model")

        # Apply mode defaults
        if self.retrain_config.mode == RetrainMode.FULL:
            # Full retrain: don't provide artifacts (train everything)
            return False
        elif self.retrain_config.mode == RetrainMode.TRANSFER:
            # Transfer: provide artifacts for preprocessing, not for model
            return not is_model
        elif self.retrain_config.mode == RetrainMode.FINETUNE:
            # Finetune: provide model artifact for continuation
            return is_model

        return False

    def get_artifact(
        self,
        step_index: int,
        fold_id: Optional[int] = None
    ) -> Optional[Any]:
        """Get a single artifact for a step if applicable.

        Args:
            step_index: 1-based step index
            fold_id: Optional fold ID for fold-specific artifacts

        Returns:
            Artifact object or None if step should train
        """
        if not self._should_provide_artifact(step_index):
            return None
        return self.base_provider.get_artifact(step_index, fold_id)

    def get_artifacts_for_step(
        self,
        step_index: int,
        branch_path: Optional[List[int]] = None
    ) -> List[Tuple[str, Any]]:
        """Get all artifacts for a step if applicable.

        Args:
            step_index: 1-based step index
            branch_path: Optional branch path filter

        Returns:
            List of (artifact_id, artifact_object) tuples, or empty if should train
        """
        if not self._should_provide_artifact(step_index):
            return []
        return self.base_provider.get_artifacts_for_step(step_index, branch_path)

    def get_fold_artifacts(
        self,
        step_index: int,
        branch_path: Optional[List[int]] = None
    ) -> List[Tuple[int, Any]]:
        """Get all fold-specific artifacts for a step if applicable.

        Args:
            step_index: 1-based step index
            branch_path: Optional branch path filter

        Returns:
            List of (fold_id, artifact_object) tuples, or empty if should train
        """
        if not self._should_provide_artifact(step_index):
            return []
        return self.base_provider.get_fold_artifacts(step_index, branch_path)

    def has_artifacts_for_step(self, step_index: int) -> bool:
        """Check if artifacts should be used for this step.

        Args:
            step_index: 1-based step index

        Returns:
            True if artifacts are available and should be used
        """
        if not self._should_provide_artifact(step_index):
            return False
        return self.base_provider.has_artifacts_for_step(step_index)


class Retrainer:
    """Handles retraining pipelines with various modes.

    This class manages the retrain workflow: loading saved pipelines,
    determining which steps to retrain vs. reuse, and executing the
    modified pipeline on new data.

    Phase 7 Implementation:
        The Retrainer enables three modes:
        - full: Train from scratch with same pipeline structure
        - transfer: Use existing preprocessing, train new model
        - finetune: Continue training existing model

    Attributes:
        runner: Parent PipelineRunner instance
        resolver: Prediction resolver for loading sources
    """

    def __init__(self, runner: 'PipelineRunner'):
        """Initialize retrainer.

        Args:
            runner: Parent PipelineRunner instance
        """
        self.runner = runner
        self.resolver = PredictionResolver(
            workspace_path=runner.workspace_path,
            runs_dir=runner.runs_dir
        )
        self._resolved: Optional[ResolvedPrediction] = None

    def retrain(
        self,
        source: Union[Dict[str, Any], str, Path, Any],
        dataset: Union[DatasetConfigs, SpectroDataset, np.ndarray, Tuple[np.ndarray, ...], Dict, List[Dict], str, List[str]],
        mode: Union[str, RetrainMode] = "full",
        dataset_name: str = "retrain_dataset",
        new_model: Optional[Any] = None,
        epochs: Optional[int] = None,
        step_modes: Optional[List[StepMode]] = None,
        verbose: int = 0,
        **kwargs
    ) -> Tuple[Predictions, Dict[str, Any]]:
        """Retrain a pipeline on new data.

        Args:
            source: Prediction source (dict, folder, Run, artifact_id, bundle)
            dataset: New dataset to train on
            mode: Retrain mode ('full', 'transfer', 'finetune')
            dataset_name: Name for the dataset
            new_model: Optional new model for transfer mode
            epochs: Optional epochs for fine-tuning
            step_modes: Optional per-step mode overrides
            verbose: Verbosity level
            **kwargs: Additional parameters (learning_rate, freeze_layers, etc.)

        Returns:
            Tuple of (predictions, dataset_predictions_dict)

        Example:
            >>> retrainer = Retrainer(runner)
            >>>
            >>> # Full retrain
            >>> preds, _ = retrainer.retrain(best_pred, new_data, mode='full')
            >>>
            >>> # Transfer: use preprocessing, new model
            >>> preds, _ = retrainer.retrain(best_pred, new_data, mode='transfer')
            >>>
            >>> # Finetune: continue training
            >>> preds, _ = retrainer.retrain(best_pred, new_data, mode='finetune', epochs=10)
        """
        logger.info("=" * 120)
        logger.starting(f"Starting Nirs4all retrain ({mode} mode)")
        logger.info("=" * 120)

        # Convert mode string to enum
        if isinstance(mode, str):
            mode = RetrainMode(mode)

        # Create retrain configuration
        config = RetrainConfig(
            mode=mode,
            step_modes=step_modes or [],
            new_model=new_model,
            epochs=epochs,
            learning_rate=kwargs.get('learning_rate'),
            freeze_layers=kwargs.get('freeze_layers'),
            metadata=kwargs
        )

        # Resolve source
        self._resolved = self.resolver.resolve(source, verbose=verbose)

        if verbose > 0:
            logger.info(f"  Source type: {self._resolved.source_type}")
            logger.info(f"  Pipeline UID: {self._resolved.pipeline_uid}")
            if self._resolved.has_trace():
                logger.info("  Has execution trace: yes")

        # Normalize dataset
        dataset_configs = self.runner.orchestrator._normalize_dataset(
            dataset, dataset_name
        )

        # Execute retrain based on mode
        if mode == RetrainMode.FULL:
            return self._retrain_full(config, dataset_configs, verbose)
        elif mode == RetrainMode.TRANSFER:
            return self._retrain_transfer(config, dataset_configs, verbose)
        elif mode == RetrainMode.FINETUNE:
            return self._retrain_finetune(config, dataset_configs, verbose)
        else:
            raise ValueError(f"Unknown retrain mode: {mode}")

    def _retrain_full(
        self,
        config: RetrainConfig,
        dataset_configs: DatasetConfigs,
        verbose: int
    ) -> Tuple[Predictions, Dict[str, Any]]:
        """Full retrain: train everything from scratch.

        Uses the same pipeline structure but trains all steps anew.

        Args:
            config: Retrain configuration
            dataset_configs: Dataset configuration
            verbose: Verbosity level

        Returns:
            Tuple of (predictions, dataset_predictions_dict)
        """
        if verbose > 0:
            logger.starting("Full retrain: training all steps from scratch")

        # Get pipeline steps from resolved source
        steps = self._resolved.minimal_pipeline

        # For full retrain, we just run the pipeline normally
        # The runner will train everything from scratch
        return self.runner.run(
            pipeline=steps,
            dataset=dataset_configs,
            pipeline_name=f"retrain_full_{self._resolved.pipeline_uid[:8]}"
        )

    def _retrain_transfer(
        self,
        config: RetrainConfig,
        dataset_configs: DatasetConfigs,
        verbose: int
    ) -> Tuple[Predictions, Dict[str, Any]]:
        """Transfer retrain: use existing preprocessing, train new model.

        Reuses preprocessing artifacts from the source prediction and trains
        a new (or the same type of) model on the new data.

        Args:
            config: Retrain configuration (may include new_model)
            dataset_configs: Dataset configuration
            verbose: Verbosity level

        Returns:
            Tuple of (predictions, dataset_predictions_dict)
        """
        from nirs4all.pipeline.execution.builder import ExecutorBuilder
        from nirs4all.pipeline.storage.io import SimulationSaver
        from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry

        if verbose > 0:
            logger.starting("Transfer mode: reusing preprocessing, training new model")

        # We need the execution trace or manifest to know which steps are preprocessing
        if not self._resolved.has_trace():
            # Fallback: assume all steps before model step are preprocessing
            logger.warning(
                "No execution trace available. Assuming all steps before "
                "model step are preprocessing."
            )

        # Get pipeline steps
        steps = list(self._resolved.minimal_pipeline)

        # Optionally replace model step with new model
        if config.new_model is not None and self._resolved.model_step_index:
            model_step_index = self._resolved.model_step_index
            # Find the step in the list by its step_index (1-based)
            from nirs4all.pipeline.trace.extractor import MinimalPipelineStep
            for i, step in enumerate(steps):
                if isinstance(step, MinimalPipelineStep) and step.step_index == model_step_index:
                    # MinimalPipelineStep - modify the step_config
                    old_config = step.step_config
                    if isinstance(old_config, dict):
                        new_config = dict(old_config)
                        new_config['model'] = config.new_model
                        # Update model name to reflect new model
                        new_config['name'] = type(config.new_model).__name__
                        step.step_config = new_config
                    else:
                        step.step_config = {'model': config.new_model}
                    if verbose > 0:
                        logger.info(f"  Replaced model with: {type(config.new_model).__name__}")
                    break
            else:
                # Fallback: try list-based indexing (for non-MinimalPipelineStep cases)
                model_idx = model_step_index - 1  # Convert to 0-based
                if 0 <= model_idx < len(steps):
                    old_step = steps[model_idx]
                    if isinstance(old_step, dict) and 'model' in old_step:
                        new_step = dict(old_step)
                        new_step['model'] = config.new_model
                        # Update model name to reflect new model
                        new_step['name'] = type(config.new_model).__name__
                        steps[model_idx] = new_step
                    else:
                        steps[model_idx] = {'model': config.new_model}
                    if verbose > 0:
                        logger.info(f"  Replaced model with: {type(config.new_model).__name__}")

        # Execute with transfer artifact provider
        return self._execute_with_retrain_config(
            steps=steps,
            config=config,
            dataset_configs=dataset_configs,
            verbose=verbose,
            pipeline_name=f"retrain_transfer_{self._resolved.pipeline_uid[:8]}"
        )

    def _retrain_finetune(
        self,
        config: RetrainConfig,
        dataset_configs: DatasetConfigs,
        verbose: int
    ) -> Tuple[Predictions, Dict[str, Any]]:
        """Finetune retrain: continue training existing model.

        Loads the existing model and continues training on new data.

        Args:
            config: Retrain configuration (may include epochs, learning_rate)
            dataset_configs: Dataset configuration
            verbose: Verbosity level

        Returns:
            Tuple of (predictions, dataset_predictions_dict)
        """
        if verbose > 0:
            logger.starting("Finetune mode: continuing training on new data")
            if config.epochs:
                logger.info(f"  Additional epochs: {config.epochs}")

        # Get pipeline steps
        steps = list(self._resolved.minimal_pipeline)

        # Inject finetune parameters into model step
        if self._resolved.model_step_index:
            model_idx = self._resolved.model_step_index - 1  # Convert to 0-based
            if 0 <= model_idx < len(steps):
                old_step = steps[model_idx]
                if isinstance(old_step, dict):
                    new_step = dict(old_step)
                    # Add finetune metadata
                    new_step['_finetune'] = True
                    if config.epochs:
                        new_step['_finetune_epochs'] = config.epochs
                    if config.learning_rate:
                        new_step['_finetune_lr'] = config.learning_rate
                    if config.freeze_layers:
                        new_step['_finetune_freeze'] = config.freeze_layers
                    steps[model_idx] = new_step

        # Execute with finetune artifact provider
        return self._execute_with_retrain_config(
            steps=steps,
            config=config,
            dataset_configs=dataset_configs,
            verbose=verbose,
            pipeline_name=f"retrain_finetune_{self._resolved.pipeline_uid[:8]}"
        )

    def _execute_with_retrain_config(
        self,
        steps: List[Any],
        config: RetrainConfig,
        dataset_configs: DatasetConfigs,
        verbose: int,
        pipeline_name: str
    ) -> Tuple[Predictions, Dict[str, Any]]:
        """Execute pipeline with retrain-aware artifact provider.

        This is the core execution method that respects the retrain configuration
        by injecting a RetrainArtifactProvider that provides artifacts only for
        steps that should use existing artifacts.

        Args:
            steps: Pipeline steps to execute
            config: Retrain configuration
            dataset_configs: Dataset configuration
            verbose: Verbosity level
            pipeline_name: Name for the retrain pipeline

        Returns:
            Tuple of (predictions, dataset_predictions_dict)
        """
        from datetime import datetime
        from nirs4all.pipeline.execution.builder import ExecutorBuilder
        from nirs4all.pipeline.storage.io import SimulationSaver
        from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry
        from nirs4all.pipeline.trace import TraceRecorder

        run_predictions = Predictions()
        datasets_predictions = {}

        for data_config, name in dataset_configs.configs:
            # Create run directory
            date_str = datetime.now().strftime("%Y-%m-%d")
            current_run_dir = self.runner.runs_dir / f"{date_str}_{name}"
            current_run_dir.mkdir(parents=True, exist_ok=True)

            # Create components
            saver = SimulationSaver(current_run_dir, save_artifacts=self.runner.save_artifacts, save_charts=self.runner.save_charts)
            manifest_manager = ManifestManager(current_run_dir)

            # Create pipeline in manifest system
            pipeline_config = {"steps": steps}
            pipeline_hash = hash(str(steps)) % (2**32)  # Simple hash for identification
            pipeline_uid, pipeline_dir = manifest_manager.create_pipeline(
                name=pipeline_name,
                dataset=name,
                pipeline_config=pipeline_config,
                pipeline_hash=f"{pipeline_hash:08x}"
            )

            # Register with saver
            saver.register(pipeline_uid)

            # Create artifact registry for new artifacts
            artifact_registry = ArtifactRegistry(
                workspace=self.runner.workspace_path,
                dataset=name,
                manifest_manager=manifest_manager
            )
            artifact_registry.start_run()

            # Build executor
            executor = (ExecutorBuilder()
                .with_run_directory(current_run_dir)
                .with_workspace(self.runner.workspace_path)
                .with_verbose(verbose)
                .with_mode("train")  # Retrain is a train mode
                .with_save_artifacts(self.runner.save_artifacts)
                .with_save_charts(self.runner.save_charts)
                .with_continue_on_error(self.runner.continue_on_error)
                .with_show_spinner(self.runner.show_spinner)
                .with_plots_visible(self.runner.plots_visible)
                .with_artifact_registry(artifact_registry)
                .build())

            # Get dataset
            dataset = dataset_configs.get_dataset(data_config, name)

            # Initialize context
            context = executor.initialize_context(dataset)

            # Set mode based on retrain config
            # For transfer/finetune, we use a hybrid mode
            if config.mode in (RetrainMode.TRANSFER, RetrainMode.FINETUNE):
                context.state.mode = "retrain"

            # Create retrain artifact provider
            base_provider = self._resolved.artifact_provider
            if base_provider is None and self._resolved.run_dir:
                # Create provider from manifest
                manifest = self._resolved.manifest
                loader = ArtifactLoader.from_manifest(manifest, self._resolved.run_dir)
                base_provider = LoaderArtifactProvider(loader=loader, trace=self._resolved.trace)

            retrain_provider = None
            if base_provider:
                retrain_provider = RetrainArtifactProvider(
                    base_provider=base_provider,
                    retrain_config=config,
                    trace=self._resolved.trace
                )

            # Create runtime context with retrain provider
            runtime_context = RuntimeContext(
                saver=saver,
                manifest_manager=manifest_manager,
                artifact_loader=None,  # Not needed, using provider
                artifact_provider=retrain_provider,
                artifact_registry=artifact_registry,
                step_runner=executor.step_runner
            )

            # Store retrain config in runtime context for controllers
            runtime_context.retrain_config = config  # type: ignore

            # Execute pipeline
            config_predictions = Predictions()

            try:
                executor.execute(
                    steps=steps,
                    config_name=pipeline_name,
                    dataset=dataset,
                    context=context,
                    runtime_context=runtime_context,
                    prediction_store=config_predictions
                )

                artifact_registry.end_run()

            except Exception as e:
                artifact_registry.cleanup_failed_run()
                raise

            run_predictions.merge_predictions(config_predictions)
            datasets_predictions[name] = {
                "run_predictions": config_predictions,
                "dataset": dataset,
                "dataset_name": name
            }

        return run_predictions, datasets_predictions

    def extract(
        self,
        source: Union[Dict[str, Any], str, Path, Any],
        verbose: int = 0
    ) -> 'ExtractedPipeline':
        """Extract a pipeline for inspection or modification.

        Returns an ExtractedPipeline object that can be inspected, modified,
        and then executed with runner.run().

        Args:
            source: Prediction source (dict, folder, Run, artifact_id, bundle)
            verbose: Verbosity level

        Returns:
            ExtractedPipeline for inspection/modification

        Example:
            >>> extracted = retrainer.extract(best_pred)
            >>> print(extracted.steps)
            >>> extracted.steps[-1] = {"model": RandomForestRegressor()}
            >>> preds, _ = runner.run(extracted.steps, new_data)
        """
        resolved = self.resolver.resolve(source, verbose=verbose)

        return ExtractedPipeline(
            steps=list(resolved.minimal_pipeline),
            trace=resolved.trace,
            artifact_provider=resolved.artifact_provider,
            model_step_index=resolved.model_step_index,
            preprocessing_chain=resolved.get_preprocessing_chain(),
            source_pipeline_uid=resolved.pipeline_uid,
            metadata={
                "source_type": str(resolved.source_type),
                "run_dir": str(resolved.run_dir) if resolved.run_dir else None
            }
        )


@dataclass
class ExtractedPipeline:
    """Extracted pipeline for inspection and modification.

    Represents a pipeline extracted from a trained prediction, ready for
    inspection, modification, or re-execution.

    Attributes:
        steps: List of pipeline steps (can be modified)
        trace: Original execution trace (read-only)
        artifact_provider: Provider for original artifacts
        model_step_index: Index of the model step
        preprocessing_chain: Summary of preprocessing
        source_pipeline_uid: UID of the source pipeline
        metadata: Additional metadata
    """

    steps: List[Any] = field(default_factory=list)
    trace: Optional[ExecutionTrace] = None
    artifact_provider: Optional[ArtifactProvider] = None
    model_step_index: Optional[int] = None
    preprocessing_chain: str = ""
    source_pipeline_uid: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_step(self, index: int) -> Any:
        """Get a step by 0-based index.

        Args:
            index: 0-based step index

        Returns:
            Step configuration
        """
        return self.steps[index]

    def set_step(self, index: int, step: Any) -> None:
        """Set a step by 0-based index.

        Args:
            index: 0-based step index
            step: New step configuration
        """
        self.steps[index] = step

    def get_model_step(self) -> Optional[Any]:
        """Get the model step.

        Returns:
            Model step configuration or None
        """
        if self.model_step_index is None:
            return None
        idx = self.model_step_index - 1  # Convert to 0-based
        if 0 <= idx < len(self.steps):
            return self.steps[idx]
        return None

    def set_model(self, model: Any) -> None:
        """Replace the model in the model step.

        Args:
            model: New model to use
        """
        if self.model_step_index is None:
            raise ValueError("No model step identified in pipeline")

        idx = self.model_step_index - 1  # Convert to 0-based
        if idx < 0 or idx >= len(self.steps):
            raise ValueError(f"Invalid model step index: {self.model_step_index}")

        old_step = self.steps[idx]
        if isinstance(old_step, dict) and 'model' in old_step:
            new_step = dict(old_step)
            new_step['model'] = model
            self.steps[idx] = new_step
        else:
            self.steps[idx] = {'model': model}

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return (
            f"ExtractedPipeline(steps={len(self.steps)}, "
            f"model_step={self.model_step_index}, "
            f"preprocessing='{self.preprocessing_chain}')"
        )
