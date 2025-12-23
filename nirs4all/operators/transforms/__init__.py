from ..augmentation.random import (
    Random_X_Operation,
    Rotate_Translate,
)
from ..augmentation.splines import (
    Spline_Curve_Simplification,
    Spline_X_Simplification,
    Spline_Y_Perturbations,
    Spline_X_Perturbations,
    Spline_Smoothing,
)
from ..augmentation.spectral import (
    GaussianAdditiveNoise,
    MultiplicativeNoise,
    LinearBaselineDrift,
    PolynomialBaselineDrift,
    WavelengthShift,
    WavelengthStretch,
    LocalWavelengthWarp,
    SmoothMagnitudeWarp,
    BandPerturbation,
    GaussianSmoothingJitter,
    UnsharpSpectralMask,
    BandMasking,
    ChannelDropout,
    SpikeNoise,
    LocalClipping,
    MixupAugmenter,
    LocalMixupAugmenter,
    ScatterSimulationMSC,
)
from ..augmentation.abc_augmenter import Augmenter, IdentityAugmenter

from .nirs import (
    Haar,
    MultiplicativeScatterCorrection,
    SavitzkyGolay,
    Wavelet,
    WaveletFeatures,
    WaveletPCA,
    WaveletSVD,
    msc,
    savgol,
    wavelet_transform,
    LogTransform,
    FirstDerivative,
    SecondDerivative,
    log_transform,
    first_derivative,
    second_derivative,
    ReflectanceToAbsorbance,
    reflectance_to_absorbance,
    # Baseline correction
    PYBASELINES_METHODS,
    pybaseline_correction,
    PyBaselineCorrection,
    ASLSBaseline,
    asls_baseline,
    AirPLS,
    ArPLS,
    IModPoly,
    ModPoly,
    SNIP,
    RollingBall,
    IASLS,
    BEADS,
)

# Import scalers (including local aliases such as IdentityTransformer and
# RobustNormalVariate which are defined in the scalers module)
from .scalers import (
    IdentityTransformer,
    Derivate,
    Normalize,
    SimpleScale,
    derivate,
    norml,
    spl_norml,
    StandardNormalVariate,
    LocalStandardNormalVariate,
    RobustStandardNormalVariate,
)
from .signal import Baseline, Detrend, Gaussian, baseline, detrend, gaussian
from .signal_conversion import (
    ToAbsorbance,
    FromAbsorbance,
    PercentToFraction,
    FractionToPercent,
    KubelkaMunk,
    SignalTypeConverter,
)
from .features import CropTransformer, ResampleTransformer, FlattenPreprocessing
from .resampler import Resampler
from .feature_selection import CARS, MCUVE
from .presets import (
    id_preprocessing,
    savgol_only,
    haar_only,
    nicon_set,
    decon_set,
    senseen_set,
    transf_set,
    special_set,
    small_set,
    dumb_set,
    dumb_and_dumber_set,
    dumb_set_2D,
    list_of_2D_sets,
    optimal_set_2D,
    preprocessing_list,
    fat_set,
)
from .targets import IntegerKBinsDiscretizer, RangeDiscretizer


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
    "GaussianAdditiveNoise",
    "MultiplicativeNoise",
    "LinearBaselineDrift",
    "PolynomialBaselineDrift",
    "WavelengthShift",
    "WavelengthStretch",
    "LocalWavelengthWarp",
    "SmoothMagnitudeWarp",
    "BandPerturbation",
    "GaussianSmoothingJitter",
    "UnsharpSpectralMask",
    "BandMasking",
    "ChannelDropout",
    "SpikeNoise",
    "LocalClipping",
    "MixupAugmenter",
    "LocalMixupAugmenter",
    "ScatterSimulationMSC",

    # Sklearn aliases
    "IdentityTransformer",  # sklearn.preprocessing.FunctionTransformer alias
    "StandardNormalVariate",
    "LocalStandardNormalVariate",
    "RobustStandardNormalVariate",

    # NIRS transformations
    "SavitzkyGolay",
    "Haar",
    "MultiplicativeScatterCorrection",
    "Wavelet",
    "WaveletFeatures",
    "WaveletPCA",
    "WaveletSVD",
    "savgol",
    "msc",
    "wavelet_transform",
    "LogTransform",
    "FirstDerivative",
    "SecondDerivative",
    "log_transform",
    "first_derivative",
    "second_derivative",
    "ReflectanceToAbsorbance",
    "reflectance_to_absorbance",
    # Baseline correction (pybaselines wrapper)
    "PYBASELINES_METHODS",
    "pybaseline_correction",
    "PyBaselineCorrection",
    "ASLSBaseline",
    "asls_baseline",
    "AirPLS",
    "ArPLS",
    "IModPoly",
    "ModPoly",
    "SNIP",
    "RollingBall",
    "IASLS",
    "BEADS",

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

    # Signal type conversion
    "ToAbsorbance",
    "FromAbsorbance",
    "PercentToFraction",
    "FractionToPercent",
    "KubelkaMunk",
    "SignalTypeConverter",

    # Features
    "CropTransformer",
    "ResampleTransformer",
    "FlattenPreprocessing",

    # Wavelength resampling
    "Resampler",

    # Feature selection
    "CARS",
    "MCUVE",

    # Targets / discretizers
    "IntegerKBinsDiscretizer",
    "RangeDiscretizer",

    # Preset sets and helpers
    "id_preprocessing",
    "savgol_only",
    "haar_only",
    "nicon_set",
    "decon_set",
    "senseen_set",
    "transf_set",
    "special_set",
    "small_set",
    "dumb_set",
    "dumb_and_dumber_set",
    "dumb_set_2D",
    "list_of_2D_sets",
    "optimal_set_2D",
    "preprocessing_list",
    "fat_set",
]
