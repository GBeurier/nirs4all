"""
Controllers module for nirs4all package.

This module contains all controller classes for pipeline operator execution.
Controllers implement the execution logic for different operator types following
the operator-controller pattern.
"""

# Import base controller class
from .base import BaseController

# Import core controller infrastructure
from .controller import OperatorController
from .registry import register_controller, CONTROLLER_REGISTRY

# Import flow control controllers
from .flow.dummy import DummyController

# Import model controllers (higher priority for supervised models)
from .models.sklearn_model import SklearnModelController
from .models.tensorflow_model import TensorFlowModelController
from .models.torch_model import PyTorchModelController
from .models.jax_model import JaxModelController

# Import transform controllers
from .transforms.transformer import TransformerMixinController
from .transforms.y_transformer import YTransformerMixinController

# Import data manipulation controllers
from .data.feature_augmentation import FeatureAugmentationController
from .data.sample_augmentation import SampleAugmentationController
from .data.resampler import ResamplerController
from .data.concat_transform import ConcatAugmentationController
from .data.auto_transfer_preproc import AutoTransferPreprocessingController
from .data.sample_filter import SampleFilterController
from .data.branch import BranchController

# Import splitter controllers
from .splitters.split import CrossValidatorController

# Import chart controllers
from .charts.spectra import SpectraChartController
from .charts.folds import FoldChartController
from .charts.targets import YChartController
from .charts.augmentation import AugmentationChartController
from .charts.spectral_distribution import SpectralDistributionController

__all__ = [
    'BaseController',
    'OperatorController',
    'register_controller',
    'CONTROLLER_REGISTRY',
    'DummyController',
    'TransformerMixinController',
    'YTransformerMixinController',
    'FeatureAugmentationController',
    'SampleAugmentationController',
    'SampleFilterController',
    'ResamplerController',
    'ConcatAugmentationController',
    'AutoTransferPreprocessingController',
    'BranchController',
    'CrossValidatorController',
    'SpectraChartController',
    'FoldChartController',
    'YChartController',
    'AugmentationChartController',
    'SpectralDistributionController',
    'SklearnModelController',
    'TensorFlowModelController',
    'PyTorchModelController',
    'JaxModelController',
]
