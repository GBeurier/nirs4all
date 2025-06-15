# pipeline/runners/base.py
"""Base class for pipeline operator controllers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING

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

    @abstractmethod
    def execute(
        self,
        step: Any,
        operator: Any,
        dataset: SpectroDataset,
        context: Dict[str, Any],
        runner: "PipelineRunner"
    ):
        """Run the operator with the given parameters and context."""
        raise NotImplementedError("Subclasses must implement this method.")


