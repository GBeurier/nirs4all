from abc import ABC, abstractmethod
from sklearn.base import TransformerMixin, ClusterMixin, BaseEstimator
import hashlib
from typing import Optional, List

try:
    from SpectraDataset import SpectraDataset
    from PipelineContext import PipelineContext
    from DatasetView import DatasetView
except ImportError:
    from SpectraDataset import SpectraDataset
    from PipelineContext import PipelineContext
    from DatasetView import DatasetView

class PipelineOperation(ABC):
    """Base class for pipeline operations."""

    @abstractmethod
    def execute(self, dataset: SpectraDataset, context: 'PipelineContext') -> None:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
