
from abc import ABC, abstractmethod
from typing import Dict, Any

from SpectraDataset import SpectraDataset

class PipelineOperation(ABC):
    """Base class for pipeline operations."""

    @abstractmethod
    def execute(self, dataset: SpectraDataset, context: Dict[str, Any]) -> None:
        """Execute operation with simplified context dict containing only branch info"""
        print(f"Executing {self.get_name()} operation")

    @abstractmethod
    def get_name(self) -> str:
        pass


class GenericOperation(PipelineOperation):
    """Generic wrapper for operators that don't fit standard categories"""

    def __init__(self, operator: Any):
        self.operator = operator

    def execute(self, dataset, context=None):
        """Generic execution - try common patterns"""
        print(f"  ⚙️ Executing {self.get_name()}")

        if self.operator is None:
            print("    ❗ No operator provided, nothing to execute")
            return

        if hasattr(self.operator, 'fit_transform'):
            print(f"    📊 Would fit_transform on training data")
        elif hasattr(self.operator, 'transform'):
            print(f"    🔄 Would transform all data")
        elif hasattr(self.operator, 'fit'):
            print(f"    🎯 Would fit on training data")
        else:
            print(f"    💡 Would execute {type(self.operator)}")

    def get_name(self) -> str:
        if self.operator is None:
            return "GenericOperation(None)"
        return f"Generic({self.operator.__class__.__name__})"