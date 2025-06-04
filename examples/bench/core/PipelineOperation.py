
from abc import ABC, abstractmethod
from sklearn.base import TransformerMixin, ClusterMixin, BaseEstimator
import hashlib
from typing import Optional, List

from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext

class PipelineOperation(ABC):
    """Base class for pipeline operations."""

    @abstractmethod
    def execute(self, dataset: SpectraDataset, context: 'PipelineContext') -> None:
        print(f"Executing {self.get_name()} operation")
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
