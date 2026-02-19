"""
Controllers module for nirs4all package.

This module contains all controller classes for pipeline operator execution.
Controllers implement the execution logic for different operator types following
the operator-controller pattern.
"""

# Import base controller class
from .base import BaseController
from .charts.augmentation import AugmentationChartController
from .charts.folds import FoldChartController

# Import chart controllers
from .charts.spectra import SpectraChartController
from .charts.spectral_distribution import SpectralDistributionController
from .charts.targets import YChartController

# Import core controller infrastructure
from .controller import OperatorController
from .data.auto_transfer_preproc import AutoTransferPreprocessingController
from .data.branch import BranchController
from .data.concat_transform import ConcatAugmentationController
from .data.exclude import ExcludeController

# Import data manipulation controllers
from .data.feature_augmentation import FeatureAugmentationController
from .data.resampler import ResamplerController
from .data.sample_augmentation import SampleAugmentationController
from .data.tag import TagController

# Import flow control controllers
from .flow.dummy import DummyController
from .models.jax_model import JaxModelController

# Import model controllers (higher priority for supervised models)
from .models.sklearn_model import SklearnModelController
from .models.tensorflow_model import TensorFlowModelController
from .models.torch_model import PyTorchModelController
from .registry import CONTROLLER_REGISTRY, register_controller

# Import splitter controllers
from .splitters.split import CrossValidatorController

# Import transform controllers
from .transforms.transformer import TransformerMixinController
from .transforms.y_transformer import YTransformerMixinController

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
    'ExcludeController',
    'TagController',
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
