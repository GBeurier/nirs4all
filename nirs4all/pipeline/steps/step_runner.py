"""Step runner for executing individual pipeline steps."""
from typing import Any, List, Optional, Tuple

from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.config.context import ExecutionContext
from nirs4all.core.logging import get_logger
from nirs4all.pipeline.execution.result import ArtifactMeta, StepResult
from nirs4all.pipeline.steps.parser import ParsedStep, StepParser, StepType
from nirs4all.pipeline.steps.router import ControllerRouter

logger = get_logger(__name__)


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
            logger.warning("No operation defined for this step, skipping.")
            return StepResult(updated_context=context, artifacts=[])

        # Handle subpipelines (nested lists)
        if parsed_step.step_type == StepType.SUBPIPELINE:
            substeps = parsed_step.metadata["steps"]
            current_context = context
            all_artifacts = []

            # In predict mode, check if we should only execute a specific substep
            # This is critical for subpipelines with multiple models like [JaxMLPRegressor, nicon]
            # where we want to run only the model that was selected as best during training
            #
            # IMPORTANT: Only apply target_sub_index filtering for MODEL subpipelines.
            # For TRANSFORMER subpipelines (e.g., from feature_augmentation like [SNV, SG]),
            # all substeps must execute because they represent parallel feature channels.
            target_sub_index = None
            if (self.mode in ("predict", "explain") and
                runtime_context and
                hasattr(runtime_context, 'artifact_provider') and
                runtime_context.artifact_provider is not None and
                hasattr(runtime_context.artifact_provider, 'target_sub_index')):
                # Only use target_sub_index filtering if this subpipeline contains models
                # Check if any substep is a model by parsing and routing
                has_model_substep = self._subpipeline_contains_model(substeps)
                if has_model_substep:
                    target_sub_index = runtime_context.artifact_provider.target_sub_index

            # Track substep index for artifact ID uniqueness
            for substep_idx, substep in enumerate(substeps):
                # In predict mode with target_sub_index, skip substeps that don't match
                if target_sub_index is not None and substep_idx != target_sub_index:
                    logger.debug(f"Skipping substep {substep_idx} (target is {target_sub_index})")
                    continue

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

        operator_name = (
            parsed_step.operator.__class__.__name__
            if parsed_step.operator is not None
            else ""
        )
        controller_name = controller.__class__.__name__

        if parsed_step.operator is not None:
            logger.debug(f"Executing controller {controller_name} with operator {operator_name}")
        else:
            logger.debug(f"Executing controller {controller_name} without operator")

        # Check if controller supports prediction mode
        if (self.mode == "predict" or self.mode == "explain") and not controller.supports_prediction_mode():
            logger.warning(
                f"Controller {controller.__class__.__name__} "
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

    def _subpipeline_contains_model(self, substeps: list) -> bool:
        """Check if a subpipeline contains any model substeps.

        This is used to determine whether target_sub_index filtering should
        be applied during prediction. Model subpipelines (e.g., [model1, model2])
        need filtering to run only the best model. Transformer subpipelines
        (e.g., [SNV, SavGol] from feature_augmentation) need to run all substeps.

        Args:
            substeps: List of substep configurations

        Returns:
            True if any substep is a model, False otherwise
        """
        from nirs4all.controllers.models.base_model import BaseModelController

        for substep in substeps:
            parsed = self.parser.parse(substep)
            if parsed.metadata.get("skip", False):
                continue

            # Check if this substep would route to a model controller
            try:
                controller = self.router.route(parsed, substep)
                if isinstance(controller, BaseModelController):
                    return True
            except Exception:
                # If routing fails, assume it's not a model
                pass

        return False
