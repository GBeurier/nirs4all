# pipeline/runners/base.py
"""Base class for pipeline operator controllers."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.execution.result import StepOutput

if TYPE_CHECKING:
    from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
    from nirs4all.pipeline.steps.parser import ParsedStep

class OperatorController(ABC):
    """Base class for pipeline operators."""
    priority: int = 100

    @classmethod
    @abstractmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Check if the operator matches the step and keyword."""
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    @abstractmethod
    def use_multi_source(cls) -> bool:
        """Check if the operator supports multi-source datasets."""
        return False

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """
        Check if the controller should execute during prediction mode.

        Returns:
            True if the controller should execute in prediction mode,
            False if it should be skipped (e.g., chart controllers)
        """
        return False

    @classmethod
    def supports_step_cache(cls) -> bool:
        """Whether this step's output should be cached for cross-variant reuse.

        Only preprocessing transforms benefit from step caching.
        Models, splitters, branch/merge, and feature augmentation steps
        should not be cached.

        Returns:
            True if step results can be safely cached and restored.
        """
        return False

    @abstractmethod
    def execute(
        self,
        step_info: "ParsedStep",
        dataset: SpectroDataset,
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: list[tuple[str, Any]] | None = None,
        prediction_store: Any | None = None
    ) -> tuple["ExecutionContext", Any]:
        """
        Run the operator with the given parameters and context.

        Args:
            step_info: Parsed step containing operator, keyword, and metadata
            dataset: Dataset to operate on
            context: Pipeline execution context
            runtime_context: Runtime infrastructure context
            source: Data source index
            mode: Execution mode ("train" or "predict")
            loaded_binaries: Pre-loaded binary objects for prediction mode
            prediction_store: External prediction store for model predictions

        Returns:
            Tuple of (updated_context, StepOutput)
        """
        raise NotImplementedError("Subclasses must implement this method.")

