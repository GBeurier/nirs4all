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
from .controller import OperatorController
from .registry import register_controller, CONTROLLER_REGISTRY

# Import actions FIRST to ensure controllers get registered before anything else uses the registry
# from . import actions

from .log.op_dummy import DummyController
from .sklearn.op_transformermixin import TransformerMixinController
from .dataset.op_feature_augmentation import FeatureAugmentationController
from .dataset.op_sample_augmentation import SampleAugmentationController
from .sklearn.op_split import CrossValidatorController
from .chart.op_spectra_charts import SpectraChartController
from .chart.op_spectra_charts3d import SpectraChartController3D
__all__ = [
    'OperatorController',
    'register_controller',
    'CONTROLLER_REGISTRY',
    'DummyController',
    'TransformerMixinController',
    'FeatureAugmentationController',
    'SampleAugmentationController',
    'CrossValidatorController',
    'SpectraChartController',
    'SpectraChartController3D',
    # Archived operations not included
]
