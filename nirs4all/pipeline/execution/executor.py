"""Pipeline executor for executing a single pipeline on a single dataset."""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.storage.artifacts.manager import ArtifactManager
from nirs4all.pipeline.config.context import ExecutionContext
from nirs4all.pipeline.storage.manifest_manager import ManifestManager
from nirs4all.pipeline.steps.step_runner import StepRunner
from nirs4all.utils.emoji import ROCKET, MEDAL_GOLD, FLAG, CROSS


class PipelineExecutor:
    """Executes a single pipeline configuration on a single dataset.

    Handles:
    - Step-by-step execution
    - Context propagation
    - Artifact management for one pipeline run
    - Predictions accumulation for this pipeline

    Attributes:
        step_runner: Executes individual steps
        artifact_manager: Manages artifact persistence
        manifest_manager: Manages pipeline manifests
        verbose: Verbosity level
        mode: Execution mode (train/predict/explain)
        continue_on_error: Whether to continue on step failures
    """

    def __init__(
        self,
        step_runner: StepRunner,
        artifact_manager: Optional[ArtifactManager] = None,
        manifest_manager: Optional[ManifestManager] = None,
        verbose: int = 0,
        mode: str = "train",
        continue_on_error: bool = False,
        saver: Any = None,
        binary_loader: Any = None
    ):
        """Initialize pipeline executor.

        Args:
            step_runner: Step runner for executing individual steps
            artifact_manager: Optional artifact manager
            manifest_manager: Optional manifest manager
            verbose: Verbosity level
            mode: Execution mode (train/predict/explain)
            continue_on_error: Whether to continue on step failures
            saver: Simulation saver for file operations
            binary_loader: Binary loader for predict/explain modes
        """
        self.step_runner = step_runner
        self.artifact_manager = artifact_manager
        self.manifest_manager = manifest_manager
        self.verbose = verbose
        self.mode = mode
        self.continue_on_error = continue_on_error
        self.saver = saver
        self.binary_loader = binary_loader

        # Execution state
        self.step_number = 0
        self.substep_number = -1
        self.operation_count = 0

    def initialize_context(self, dataset: SpectroDataset) -> ExecutionContext:
        """Initialize ExecutionContext for pipeline execution.

        Args:
            dataset: Dataset to create context for

        Returns:
            Initialized ExecutionContext
        """
        from nirs4all.pipeline.config.context import DataSelector, PipelineState, StepMetadata

        selector = DataSelector(
            partition=None,
            processing=[["raw"]] * dataset.features_sources(),
            layout="2d",
            concat_source=True
        )

        state = PipelineState(
            y_processing="numeric",
            step_number=0,
            mode=self.mode
        )

        metadata = StepMetadata()

        return ExecutionContext(
            selector=selector,
            state=state,
            metadata=metadata
        )

    def execute(
        self,
        steps: List[Any],
        config_name: str,
        dataset: SpectroDataset,
        context: ExecutionContext,
        runner: Any,  # PipelineRunner reference for compatibility
        prediction_store: Optional[Predictions] = None
    ) -> None:
        """Execute pipeline steps sequentially on dataset.

        Args:
            steps: List of pipeline steps to execute
            config_name: Pipeline configuration name
            dataset: Dataset to process
            context: Initial execution context
            runner: Runner instance (for compatibility with controllers)
            prediction_store: Prediction store for accumulating results

        Raises:
            RuntimeError: If pipeline execution fails
        """
        # Reset state for this execution
        self.step_number = 0
        self.substep_number = -1
        self.operation_count = 0

        print(f"\033[94m{ROCKET}Starting pipeline {config_name} on dataset {dataset.name}\033[0m")
        print("-" * 120)

        # Compute pipeline hash for identification
        pipeline_hash = self._compute_pipeline_hash(steps)

        # Create pipeline in manifest system (if in train mode)
        pipeline_uid = None
        if self.mode == "train" and self.manifest_manager:
            pipeline_config = {"steps": steps}
            pipeline_uid, pipeline_dir = self.manifest_manager.create_pipeline(
                name=config_name,
                dataset=dataset.name,
                pipeline_config=pipeline_config,
                pipeline_hash=pipeline_hash
            )

            # Register with saver
            if self.saver:
                self.saver.register(pipeline_uid)

            # Set pipeline_uid on runner for compatibility with controllers
            if runner:
                runner.pipeline_uid = pipeline_uid
        else:
            # For predict/explain modes, use temporary UID
            pipeline_uid = f"temp_{pipeline_hash}"

        # Save pipeline configuration
        if self.mode != "predict" and self.mode != "explain" and self.saver:
            self.saver.save_json("pipeline.json", steps)

        # Initialize prediction store if not provided
        if prediction_store is None:
            prediction_store = Predictions()

        # Execute all steps
        all_artifacts = []
        try:
            context = self._execute_steps(
                steps,
                dataset,
                context,
                runner,
                prediction_store,
                all_artifacts
            )

            # Save final pipeline configuration
            if self.mode != "predict" and self.mode != "explain" and self.saver:
                self.saver.save_json("pipeline.json", steps)

            # Print best result if predictions were generated
            if prediction_store.num_predictions > 0:
                pipeline_best = prediction_store.get_best(
                    ascending=True if dataset.is_regression else False
                )
                if pipeline_best:
                    print(f"{MEDAL_GOLD}Pipeline Best: {Predictions.pred_short_string(pipeline_best)}")

                if self.verbose > 0:
                    print(
                        f"\033[94m{FLAG}Pipeline {config_name} completed successfully "
                        f"on dataset {dataset.name}\033[0m"
                    )

            print("=" * 120)

        except Exception as e:
            print(
                f"\033[91m{CROSS}Pipeline {config_name} on dataset {dataset.name} "
                f"failed: \n{str(e)}\033[0m"
            )
            import traceback
            traceback.print_exc()
            raise

    def _execute_steps(
        self,
        steps: List[Any],
        dataset: SpectroDataset,
        context: ExecutionContext,
        runner: Any,
        prediction_store: Predictions,
        all_artifacts: List[Any]
    ) -> ExecutionContext:
        """Execute all steps in sequence.

        Args:
            steps: List of steps to execute
            dataset: Dataset to process
            context: Current execution context
            runner: Runner instance for compatibility
            prediction_store: Prediction store
            all_artifacts: List to accumulate artifacts

        Returns:
            Updated execution context
        """
        for step in steps:
            self.step_number += 1
            self.substep_number = 0
            self.operation_count = 0

            # Sync step number to runner for controller compatibility
            if runner:
                runner.step_number = self.step_number
                runner.substep_number = self.substep_number
                runner.operation_count = self.operation_count

            # Update context with current step number
            if isinstance(context, ExecutionContext):
                context = context.with_step_number(self.step_number)

            # Load binaries if in prediction/explain mode
            loaded_binaries = None
            if self.mode in ("predict", "explain") and self.binary_loader:
                loaded_binaries = self.binary_loader.get_step_binaries(self.step_number)
                if self.verbose > 1 and loaded_binaries:
                    print(f"🔍 Loaded {', '.join(b[0] for b in loaded_binaries)} binaries for step {self.step_number}")

            try:
                # Execute step via step runner
                step_result = self.step_runner.execute(
                    step=step,
                    dataset=dataset,
                    context=context,
                    runner=runner,
                    loaded_binaries=loaded_binaries,
                    prediction_store=prediction_store
                )

                # Update context and accumulate artifacts
                context = step_result.updated_context
                all_artifacts.extend(step_result.artifacts)

                # Sync operation_count back from runner (controllers may have incremented it)
                if runner:
                    self.operation_count = runner.operation_count

                # Append artifacts to manifest if in train mode
                if (self.mode == "train" and
                    self.manifest_manager and
                    runner.pipeline_uid and
                    step_result.artifacts):
                    self.manifest_manager.append_artifacts(
                        runner.pipeline_uid,
                        step_result.artifacts
                    )
                    if self.verbose > 1:
                        print(f"📦 Appended {len(step_result.artifacts)} artifacts to manifest")

            except Exception as e:
                if self.continue_on_error:
                    print(f"⚠️  Step {self.step_number} failed but continuing: {str(e)}")
                else:
                    raise RuntimeError(f"Pipeline step {self.step_number} failed: {str(e)}") from e

        return context

    def _compute_pipeline_hash(self, steps: List[Any]) -> str:
        """Compute MD5 hash of pipeline configuration.

        Args:
            steps: Pipeline steps

        Returns:
            6-character hash string
        """
        pipeline_json = json.dumps(steps, sort_keys=True, default=str).encode('utf-8')
        return hashlib.md5(pipeline_json).hexdigest()[:6]

    def next_op(self) -> int:
        """Get the next operation ID (for compatibility)."""
        self.operation_count += 1
        return self.operation_count
