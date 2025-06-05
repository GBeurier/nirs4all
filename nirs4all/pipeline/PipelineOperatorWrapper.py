# pipeline/runners/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING

from nirs4all.spectra.SpectraDataset import SpectraDataset

if TYPE_CHECKING:
    from .PipelineRunner import PipelineRunner

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
        step: Any,
        dataset: SpectraDataset,
        context: Dict[str, Any],
        runner: "PipelineRunner"
    ):
        """Run the operator with the given parameters and context."""
