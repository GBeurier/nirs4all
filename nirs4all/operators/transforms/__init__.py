from ..augmentation.random import (
    Random_X_Operation,
    Rotate_Translate,
)
from ..augmentation.spectral import (
    BandMasking,
    BandPerturbation,
    ChannelDropout,
    GaussianAdditiveNoise,
    GaussianSmoothingJitter,
    LinearBaselineDrift,
    LocalClipping,
    LocalMixupAugmenter,
    LocalWavelengthWarp,
    MixupAugmenter,
    MultiplicativeNoise,
    PolynomialBaselineDrift,
    ScatterSimulationMSC,
    SmoothMagnitudeWarp,
    SpikeNoise,
    UnsharpSpectralMask,
    WavelengthShift,
    WavelengthStretch,
)
from ..augmentation.splines import (
    Spline_Curve_Simplification,
    Spline_Smoothing,
    Spline_X_Perturbations,
    Spline_X_Simplification,
    Spline_Y_Perturbations,
)
from ..augmentation.synthesis import (
    BatchEffectAugmenter,
    DeadBandAugmenter,
    HeteroscedasticNoiseAugmenter,
    InstrumentalBroadeningAugmenter,
    PathLengthAugmenter,
)
from .feature_selection import CARS, MCUVE, FlexiblePCA, FlexibleSVD
from .features import CropTransformer, FlattenPreprocessing, ResampleTransformer
from .nirs import (
    BEADS,
    IASLS,
    # Baseline correction
    PYBASELINES_METHODS,
    SNIP,
    AirPLS,
    AreaNormalization,
    ArPLS,
    ASLSBaseline,
    ExtendedMultiplicativeScatterCorrection,
    FirstDerivative,
    Haar,
    IModPoly,
    LogTransform,
    ModPoly,
    MultiplicativeScatterCorrection,
    PyBaselineCorrection,
    ReflectanceToAbsorbance,
    RollingBall,
    SavitzkyGolay,
    SecondDerivative,
    Wavelet,
    WaveletFeatures,
    WaveletPCA,
    WaveletSVD,
    asls_baseline,
    first_derivative,
    log_transform,
    msc,
    pybaseline_correction,
    reflectance_to_absorbance,
    savgol,
    second_derivative,
    wavelet_transform,
)
from .norris_williams import NorrisWilliams, norris_williams
from .orthogonalization import EPO, OSC
from .presets import (
    decon_set,
    dumb_and_dumber_set,
    dumb_set,
    dumb_set_2D,
    fat_set,
    haar_only,
    id_preprocessing,
    list_of_2D_sets,
    nicon_set,
    optimal_set_2D,
    preprocessing_list,
    savgol_only,
    senseen_set,
    small_set,
    special_set,
    transf_set,
)
from .resampler import Resampler

# Import scalers (including local aliases such as IdentityTransformer and
# RobustNormalVariate which are defined in the scalers module)
from .scalers import (
    Derivate,
    IdentityTransformer,
    LocalStandardNormalVariate,
    Normalize,
    RobustStandardNormalVariate,
    SimpleScale,
    StandardNormalVariate,
    derivate,
    norml,
    spl_norml,
)
from .signal import Baseline, Detrend, Gaussian, baseline, detrend, gaussian
from .signal_conversion import (
    FractionToPercent,
    FromAbsorbance,
    KubelkaMunk,
    PercentToFraction,
    SignalTypeConverter,
    ToAbsorbance,
)
from .targets import IntegerKBinsDiscretizer, RangeDiscretizer
from .wavelet_denoise import WaveletDenoise, wavelet_denoise

__all__ = [
    # Data augmentation
    "Spline_Smoothing",
    "Spline_X_Perturbations",
    "Spline_Y_Perturbations",
    "Spline_X_Simplification",
    "Spline_Curve_Simplification",
    "Rotate_Translate",
    "Random_X_Operation",
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
    # Synthesis-derived augmentations
    "PathLengthAugmenter",
    "BatchEffectAugmenter",
    "InstrumentalBroadeningAugmenter",
    "HeteroscedasticNoiseAugmenter",
    "DeadBandAugmenter",

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
    "WaveletDenoise",
    "wavelet_denoise",
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
    "AreaNormalization",
    "ExtendedMultiplicativeScatterCorrection",

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

    # Dimensionality reduction
    "FlexiblePCA",
    "FlexibleSVD",

    # Norris-Williams gap derivative
    "NorrisWilliams",
    "norris_williams",

    # Orthogonalization
    "OSC",
    "EPO",

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
