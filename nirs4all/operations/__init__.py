"""
Operations module for nirs4all package.

This module contains all operation classes for pipeline processing.
"""

# Import main operation classes that actually exist
# Note: Archive operations are commented out as they may have compatibility issues
# from .archives.operation_centroid_propagation import OperationCentroidPropagation
# from .archives.operation_cluster import OperationCluster
# from .archives.operation_folds import OperationFolds
# from .archives.operation_split import OperationSplit
# from .archives.operation_subpipeline import OperationSubpipeline
# from .archives.operation_tranformation import OperationTransformation
# from .archives.op_transformer_mixin import OpTransformerMixin

# Import working modules
from .operation_presets import *
from .operator_controller import OperatorController
from .operator_registry import register_controller, CONTROLLER_REGISTRY
from .dummy_controller import DummyController

__all__ = [
    'OperatorController',
    'register_controller',
    'CONTROLLER_REGISTRY',
    'DummyController',
    # Archived operations not included
]
