"""Step runner for executing individual pipeline steps."""
from typing import Any, List, Optional, Tuple

from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.config.context import ExecutionContext
from nirs4all.pipeline.execution.result import ArtifactMeta, StepResult
from nirs4all.pipeline.steps.parser import ParsedStep, StepParser, StepType
from nirs4all.pipeline.steps.router import ControllerRouter
from nirs4all.utils.emoji import WARNING, SMALL_DIAMOND


class StepRunner:
    """Executes a single pipeline step.

    Handles:
    - Step parsing (delegates to StepParser)
    - Controller selection (delegates to ControllerRouter)
    - Controller execution
    - Binary loading/saving for this step

    Attributes:
        parser: Parses step configuration
        router: Routes to appropriate controller
        verbose: Verbosity level
        mode: Execution mode (train/predict/explain)
    """

    def __init__(
        self,
        parser: Optional[StepParser] = None,
        router: Optional[ControllerRouter] = None,
        verbose: int = 0,
        mode: str = "train",
        show_spinner: bool = True,
        plots_visible: bool = False
    ):
        """Initialize step runner.

        Args:
            parser: Step parser (creates new if None)
            router: Controller router (creates new if None)
            verbose: Verbosity level
            mode: Execution mode (train/predict/explain)
            show_spinner: Whether to show spinner for long operations
            plots_visible: Whether to display plots
        """
        self.parser = parser or StepParser()
        self.router = router or ControllerRouter()
        self.verbose = verbose
        self.mode = mode
        self.show_spinner = show_spinner
        self.plots_visible = plots_visible
        self._figure_refs = []

    def execute(
        self,
        step: Any,
        dataset: SpectroDataset,
        context: ExecutionContext,
        runtime_context: Any,  # RuntimeContext
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Predictions] = None
    ) -> StepResult:
        """Execute a single pipeline step.

        Args:
            step: Raw step configuration
            dataset: Dataset to process
            context: Execution context
            runtime_context: Runtime infrastructure context
            loaded_binaries: Pre-loaded artifacts for this step
            prediction_store: Prediction store for accumulating results

        Returns:
            StepResult with updated context and artifacts

        Raises:
            RuntimeError: If step execution fails
        """
        # Parse the step
        parsed_step = self.parser.parse(step)

        # Handle None/skip steps
        if parsed_step.metadata.get("skip", False):
            if self.verbose > 0:
                print(f"{WARNING}No operation defined for this step, skipping.")
            return StepResult(updated_context=context, artifacts=[])

        # Handle subpipelines (nested lists)
        if parsed_step.step_type == StepType.SUBPIPELINE:
            substeps = parsed_step.metadata["steps"]
            current_context = context
            all_artifacts = []

            # Track substep index for artifact ID uniqueness
            for substep_idx, substep in enumerate(substeps):
                # Update runtime_context substep_number for each substep
                if runtime_context:
                    runtime_context.substep_number = substep_idx

                result = self.execute(
                    step=substep,
                    dataset=dataset,
                    context=current_context,
                    runtime_context=runtime_context,
                    loaded_binaries=loaded_binaries,
                    prediction_store=prediction_store
                )
                current_context = result.updated_context
                all_artifacts.extend(result.artifacts)

            # Reset substep_number after processing subpipeline
            if runtime_context:
                runtime_context.substep_number = -1

            return StepResult(updated_context=current_context, artifacts=all_artifacts)

        # Route to controller
        controller = self.router.route(parsed_step, step)

        if self.verbose > 1:
            operator_name = (
                parsed_step.operator.__class__.__name__
                if parsed_step.operator is not None
                else ""
            )
            controller_name = controller.__class__.__name__

            if parsed_step.operator is not None:
                print(f"{SMALL_DIAMOND}Executing controller {controller_name} with operator {operator_name}")
            else:
                print(f"{SMALL_DIAMOND}Executing controller {controller_name} without operator")

        # Check if controller supports prediction mode
        if (self.mode == "predict" or self.mode == "explain") and not controller.supports_prediction_mode():
            if self.verbose > 0:
                print(
                    f"{WARNING}Controller {controller.__class__.__name__} "
                    f"does not support prediction mode, skipping step"
                )
            return StepResult(updated_context=context, artifacts=[])

        # Update context with step metadata
        if parsed_step.keyword:
            context = context.with_metadata(keyword=parsed_step.keyword)

        # Execute controller
        try:
            result = controller.execute(
                step_info=parsed_step,
                dataset=dataset,
                context=context,
                runtime_context=runtime_context,
                source=-1,
                mode=self.mode,
                loaded_binaries=loaded_binaries,
                prediction_store=prediction_store
            )

            # Handle both legacy (context, artifacts) and new (context, StepOutput) returns
            if isinstance(result, tuple):
                updated_context, output_data = result

                # Check if output_data is StepOutput or list of artifacts
                from nirs4all.pipeline.execution.result import StepOutput

                if isinstance(output_data, StepOutput):
                    return StepResult(
                        updated_context=updated_context,
                        artifacts=output_data.artifacts,
                        outputs=output_data.outputs
                    )
                else:
                    # Legacy format: output_data is list of artifacts
                    return StepResult(
                        updated_context=updated_context,
                        artifacts=output_data or [],
                        outputs=[]
                    )

                    # In legacy, artifacts were what?
                    # In BaseModelController:
                    # artifact = self._persist_model(...) -> returns ArtifactMeta
                    # binaries.append(artifact)
                    # So legacy controllers DID persistence and returned ArtifactMeta.

                    # The new proposal says controllers return raw objects.
                    # So if I get StepOutput, it has raw objects.
                    # I need to wrap them in something that StepResult accepts, OR change StepResult.

                    # If I change StepResult to accept raw objects, I break compatibility with legacy controllers that return ArtifactMeta.
                    # Unless I handle both.

                    # Let's make StepResult generic or capable of holding both.
                    # Actually, the Executor will handle persistence.
                    # So StepRunner should just pass the raw objects to Executor.
                    # But StepResult.artifacts is typed as List[ArtifactMeta].

                    # I should probably update StepResult to allow Any for artifacts, or create a new field.
                    # Or, I can persist here in StepRunner if I have access to artifact_manager?
                    # StepRunner doesn't have artifact_manager. Executor has it.

                    # So StepRunner should return the raw StepOutput to Executor.
                    # I will modify StepResult to carry the StepOutput object.

                    pass

            return StepResult(
                updated_context=updated_context,
                artifacts=output_data.artifacts if isinstance(output_data, StepOutput) else (output_data or []),
                outputs=output_data.outputs if isinstance(output_data, StepOutput) else []
            )

        except Exception as e:
            raise RuntimeError(f"Step execution failed: {str(e)}") from e
        finally:
            # Reset ephemeral metadata flags to prevent leakage between steps
            context.metadata.reset_ephemeral_flags()
