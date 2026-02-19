from .augmentation.edge_artifacts import (
    DETECTOR_MODELS,
    DetectorRollOffAugmenter,
    EdgeArtifactsAugmenter,
    EdgeCurvatureAugmenter,
    StrayLightAugmenter,
    TruncatedPeakAugmenter,
)
from .augmentation.environmental import (
    MoistureAugmenter,
    TemperatureAugmenter,
)
from .augmentation.random import Random_X_Operation, Rotate_Translate
from .augmentation.scattering import (
    EMSCDistortionAugmenter,
    ParticleSizeAugmenter,
)
from .augmentation.splines import Spline_Curve_Simplification, Spline_Smoothing, Spline_X_Perturbations, Spline_X_Simplification, Spline_Y_Perturbations
from .augmentation.synthesis import (
    BatchEffectAugmenter,
    DeadBandAugmenter,
    HeteroscedasticNoiseAugmenter,
    InstrumentalBroadeningAugmenter,
    PathLengthAugmenter,
)
from .base import SpectraTransformerMixin
from .filters import (
    SampleFilter,
    YOutlierFilter,
)
from .transforms import (
    # Signal processing
    Baseline,
    # Features
    CropTransformer,
    # Scalers
    Derivate,
    Detrend,
    Gaussian,
    # NIRS transformations
    Haar,
    # Sklearn aliases
    IdentityTransformer,
    LocalStandardNormalVariate,
    MultiplicativeScatterCorrection,
    Normalize,
    ResampleTransformer,
    RobustStandardNormalVariate,
    SavitzkyGolay,
    SimpleScale,
    StandardNormalVariate,
    Wavelet,
    baseline,
    derivate,
    detrend,
    gaussian,
    msc,
    norml,
    savgol,
    spl_norml,
    wavelet_transform,
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
    # Synthesis-derived augmentations
    "PathLengthAugmenter",
    "BatchEffectAugmenter",
    "InstrumentalBroadeningAugmenter",
    "HeteroscedasticNoiseAugmenter",
    "DeadBandAugmenter",

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
