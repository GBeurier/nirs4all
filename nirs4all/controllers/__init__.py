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

# Import model controllers FIRST (higher priority for supervised models)
from .sklearn.op_model import SklearnModelController
from .tensorflow.op_model import TensorFlowModelController
# from .torch.op_model import PyTorchModelController

# Then import transformers (lower priority)
from .sklearn.op_transformermixin import TransformerMixinController
from .sklearn.op_y_transformermixin import YTransformerMixinController
from .dataset.op_feature_augmentation import FeatureAugmentationController
from .dataset.op_sample_augmentation import SampleAugmentationController
from .dataset.op_resampler import ResamplerController
from .sklearn.op_split import CrossValidatorController
from .chart.op_spectra_charts import SpectraChartController
from .chart.op_fold_charts import FoldChartController
from .chart.op_y_chart import YChartController
__all__ = [
    'OperatorController',
    'register_controller',
    'CONTROLLER_REGISTRY',
    'DummyController',
    'TransformerMixinController',
    'YTransformerMixinController',
    'FeatureAugmentationController',
    'SampleAugmentationController',
    'ResamplerController',
    'CrossValidatorController',
    'SpectraChartController',
    'FoldChartController',
    'YChartController',
    'SklearnModelController',
    'TensorFlowModelController',
    # 'PyTorchModelController',
    # Archived operations not included
]
