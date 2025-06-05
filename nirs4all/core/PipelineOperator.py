# pipeline/runners/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class PipelineOperator(ABC):
    """Base class for pipeline operators."""
    priority: int = 100

    @classmethod
    @abstractmethod
    def matches(cls, op: Any, keyword: str | None) -> bool:
        """ Check if the operator matches the given keyword or operator criteria."""

    @abstractmethod
    def run(
        self,
        op: Any,
        params: Dict[str, Any],
        context: Dict[str, Any],
        dataset: Any,
    ):
        """Run the operator with the given parameters and context."""
