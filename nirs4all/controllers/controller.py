# pipeline/runners/base.py
"""Base class for pipeline operator controllers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING

from nirs4all.dataset.dataset import SpectroDataset

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner

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

    @abstractmethod
    def execute(
        self,
        step: Any,
        operator: Any,
        dataset: SpectroDataset,
        context: Dict[str, Any],
        runner: "PipelineRunner",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None  # NEW: External prediction store
    ):
        """
        Run the operator with the given parameters and context.

        Args:
            step: Pipeline step configuration
            operator: The operator instance
            dataset: Dataset to operate on
            context: Pipeline execution context
            runner: Pipeline runner instance
            source: Data source index
            mode: Execution mode ("train" or "predict")
            loaded_binaries: Pre-loaded binary objects for prediction mode
            prediction_store: External prediction store for model predictions
        """
        raise NotImplementedError("Subclasses must implement this method.")



