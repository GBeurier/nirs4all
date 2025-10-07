from ..augmentation.random import (Random_X_Operation,
                                   Rotate_Translate)
from ..augmentation.splines import (Spline_Curve_Simplification,
                                   Spline_X_Simplification,
                                   Spline_Y_Perturbations,
                                   Spline_X_Perturbations,
                                   Spline_Smoothing)
from ..augmentation.abc_augmenter import Augmenter, IdentityAugmenter

from sklearn.preprocessing import FunctionTransformer as IdentityTransformer
from sklearn.preprocessing import RobustScaler as RobustNormalVariate
from sklearn.preprocessing import StandardScaler as StandardNormalVariate

from .nirs import (Haar, MultiplicativeScatterCorrection, SavitzkyGolay, Wavelet, msc, savgol, wavelet_transform, LogTransform, FirstDerivative, SecondDerivative, log_transform, first_derivative, second_derivative)
from .scalers import (Derivate, Normalize, SimpleScale, derivate, norml, spl_norml)
from .signal import Baseline, Detrend, Gaussian, baseline, detrend, gaussian
from .features import CropTransformer, ResampleTransformer
from .resampler import Resampler


__all__ = [
    # Data augmentation
    "Spline_Smoothing",
    "Spline_X_Perturbations",
    "Spline_Y_Perturbations",
    "Spline_X_Simplification",
    "Spline_Curve_Simplification",
    "Rotate_Translate",
    "Random_X_Operation",
    "Augmenter",
    "IdentityAugmenter",

    # Sklearn aliases
    "IdentityTransformer",  # sklearn.preprocessing.FunctionTransformer alias
    "StandardNormalVariate",  # sklearn.preprocessing.StandardScaler alias
    "RobustNormalVariate",  # sklearn.preprocessing.RobustScaler alias

    # NIRS transformations
    "SavitzkyGolay",
    "Haar",
    "MultiplicativeScatterCorrection",
    "Wavelet",
    "savgol",
    "msc",
    "wavelet_transform",
    "LogTransform",
    "FirstDerivative",
    "SecondDerivative",
    "log_transform",
    "first_derivative",
    "second_derivative",

    # Scalers
    "Normalize",
    "Derivate",
    "SimpleScale",
    "norml",
    "derivate",
    "spl_norml",

    # Signal processing
    "Baseline",
    "Detrend",
    "Gaussian",
    "baseline",
    "detrend",
    "gaussian",

    # Features
    "CropTransformer",
    "ResampleTransformer",

    # Wavelength resampling
    "Resampler"
]
