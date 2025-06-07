from ._random_augmentation import (Random_X_Operation,
                                   Rotate_Translate)
from ._spline_augmentation import (Spline_Curve_Simplification,
                                   Spline_X_Simplification,
                                   Spline_Y_Perturbations,
                                   Spline_X_Perturbations,
                                   Spline_Smoothing)
from .augmenter import Augmenter, IdentityAugmenter

from sklearn.preprocessing import FunctionTransformer as IdentityTransformer
from sklearn.preprocessing import RobustScaler as RobustNormalVariate
from sklearn.preprocessing import StandardScaler as StandardNormalVariate

from ._nirs import (Haar, MultiplicativeScatterCorrection, SavitzkyGolay, Wavelet, msc, savgol, wavelet_transform)
from ._scaler import (Derivate, Normalize, SimpleScale, derivate, norml, spl_norml)
from ._standard import Baseline, Detrend, Gaussian, baseline, detrend, gaussian
from ._preparation import CropTransformer, ResampleTransformer
from .augmenter import Augmenter, IdentityAugmenter


__all__ = [
    "Spline_Smoothing",
    "Spline_X_Perturbations",
    "Spline_Y_Perturbations",
    "Spline_X_Simplification",
    "Spline_Curve_Simplification",
    "Rotate_Translate",
    "Random_X_Operation",
    "Augmenter",
    "IdentityAugmenter",
    "IdentityTransformer",  # sklearn.preprocessing.FunctionTransformer alias
    "Baseline",
    "StandardNormalVariate",  # sklearn.preprocessing.StandardScaler alias
    "RobustNormalVariate",  # sklearn.preprocessing.RobusScaler alias
    "SavitzkyGolay",
    "Haar",
    "Normalize",
    "Detrend",
    "MultiplicativeScatterCorrection",
    "Derivate",
    "Gaussian",
    "Wavelet",
    "SimpleScale",
    "baseline",
    "savgol",
    "norml",
    "detrend",
    "msc",
    "wavelet_transform",
    "derivate",
    "spl_norml",
    "gaussian",
    "CropTransformer",
    "ResampleTransformer",
    "Augmenter",
    "IdentityAugmenter",
]
