from .base import SpectraTransformerMixin

from .augmentation.random import (
    Random_X_Operation,
    Rotate_Translate
)
from .augmentation.splines import (
    Spline_Curve_Simplification,
    Spline_X_Simplification,
    Spline_Y_Perturbations,
    Spline_X_Perturbations,
    Spline_Smoothing
)
from .augmentation.abc_augmenter import Augmenter, IdentityAugmenter
from .augmentation.environmental import (
    TemperatureAugmenter,
    MoistureAugmenter,
)
from .augmentation.scattering import (
    ParticleSizeAugmenter,
    EMSCDistortionAugmenter,
)
from .augmentation.edge_artifacts import (
    DetectorRollOffAugmenter,
    StrayLightAugmenter,
    EdgeCurvatureAugmenter,
    TruncatedPeakAugmenter,
    EdgeArtifactsAugmenter,
    DETECTOR_MODELS,
)

from .filters import (
    SampleFilter,
    YOutlierFilter,
)

from .transforms import (
    # NIRS transformations
    Haar,
    MultiplicativeScatterCorrection,
    SavitzkyGolay,
    Wavelet,
    msc,
    savgol,
    wavelet_transform,

    # Scalers
    Derivate,
    Normalize,
    SimpleScale,
    derivate,
    norml,
    spl_norml,

    # Signal processing
    Baseline,
    Detrend,
    Gaussian,
    baseline,
    detrend,
    gaussian,

    # Features
    CropTransformer,
    ResampleTransformer,

    # Sklearn aliases
    IdentityTransformer,
    StandardNormalVariate,
    RobustStandardNormalVariate,
    LocalStandardNormalVariate
)

__all__ = [
    # Base classes
    "SpectraTransformerMixin",

    # Data augmentation
    "Random_X_Operation",
    "Rotate_Translate",
    "Spline_Curve_Simplification",
    "Spline_X_Simplification",
    "Spline_Y_Perturbations",
    "Spline_X_Perturbations",
    "Spline_Smoothing",
    "Augmenter",
    "IdentityAugmenter",
    # Environmental effects augmentation
    "TemperatureAugmenter",
    "MoistureAugmenter",
    # Scattering effects augmentation
    "ParticleSizeAugmenter",
    "EMSCDistortionAugmenter",
    # Edge artifacts augmentation
    "DetectorRollOffAugmenter",
    "StrayLightAugmenter",
    "EdgeCurvatureAugmenter",
    "TruncatedPeakAugmenter",
    "EdgeArtifactsAugmenter",
    "DETECTOR_MODELS",

    # Sample filtering
    "SampleFilter",
    "YOutlierFilter",

    # NIRS transformations
    "Haar",
    "MultiplicativeScatterCorrection",
    "SavitzkyGolay",
    "Wavelet",
    "msc",
    "savgol",
    "wavelet_transform",

    # Scalers
    "Derivate",
    "Normalize",
    "SimpleScale",
    "derivate",
    "norml",
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

    # Sklearn aliases
    "IdentityTransformer",
    "StandardNormalVariate",
    "RobustStandardNormalVariate",
    "LocalStandardNormalVariate"
]
