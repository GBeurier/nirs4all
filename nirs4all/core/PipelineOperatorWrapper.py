# pipeline/runners/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class PipelineOperatorWrapper(ABC):
    """Base class for pipeline operators."""
    priority: int = 100

    @classmethod
    @abstractmethod
    def matches(cls, step) -> bool:
        """ Check if the operator matches the given keyword or operator criteria."""

    @abstractmethod
    def execute(
        self,
        step: Any | None,
        dataset: Any,
        context: Dict[str, Any]
    ):
        """Run the operator with the given parameters and context."""
