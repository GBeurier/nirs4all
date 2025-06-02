"""
New4All - Next generation pipeline system
"""

# Core classes
from SpectraFeatures import SpectraFeatures
from TargetManager import TargetManager
from SpectraDataset import SpectraDataset
from DatasetView import DatasetView

# Pipeline components
from PipelineOperation import PipelineOperation
from PipelineContext import PipelineContext
from ModelOperation import ModelOperation
from TransformationOperation import TransformationOperation
from MergeSourcesOperation import MergeSourcesOperation
from ClusteringOperation import ClusteringOperation
from SplitOperation import SplitOperation
from DispatchOperation import DispatchOperation

# Pipeline management
from Pipeline import Pipeline
from OperationFactory import OperationFactory

__all__ = [
    'SpectraFeatures',
    'TargetManager',
    'SpectraDataset',
    'DatasetView',
    'PipelineOperation',
    'PipelineContext',
    'ModelOperation',
    'TransformationOperation',
    'MergeSourcesOperation',
    'ClusteringOperation',
    'SplitOperation',
    'DispatchOperation',
    'Pipeline',
    'OperationFactory'
]