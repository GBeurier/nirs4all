"""
Operations module for nirs4all package.

This module contains all operation classes for pipeline processing.
"""

# Import main operation classes that actually exist
from .OperationCentroidPropagation import OperationCentroidPropagation
from .OperationCluster import OperationCluster
from .OperationFolds import OperationFolds
from .OperationSplit import OperationSplit
from .OperationSubpipeline import OperationSubpipeline
from .OperationTranformation import OperationTransformation
from .OpTransformerMixin import OpTransformerMixin

__all__ = [
    'OperationCentroidPropagation',
    'OperationCluster',
    'OperationFolds',
    'OperationSplit',
    'OperationSubpipeline',
    'OperationTransformation',
    'OpTransformerMixin',
]
