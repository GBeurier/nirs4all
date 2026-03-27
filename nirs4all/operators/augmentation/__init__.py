"""
Augmentation operators for spectral data.

This module provides data augmentation operators for NIRS spectra, including:

Noise and Distortion:
    - GaussianAdditiveNoise: Add Gaussian noise
    - MultiplicativeNoise: Apply random gain factors
    - HeteroscedasticNoiseAugmenter: Signal-dependent noise
    - SpikeNoise: Add spike artifacts

Baseline Effects:
    - LinearBaselineDrift: Add linear baseline
    - PolynomialBaselineDrift: Add polynomial baseline

Wavelength Distortions:
    - WavelengthShift: Shift spectra along wavelength axis
    - WavelengthStretch: Stretch/compress wavelength axis
    - LocalWavelengthWarp: Local wavelength distortions via spline control points
    - SmoothMagnitudeWarp: Smooth multiplicative warping

Spectral Distortions:
    - BandPerturbation: Perturb random wavelength bands
    - GaussianSmoothingJitter: Random Gaussian kernel broadening
    - UnsharpSpectralMask: High-pass spectral enhancement
    - BandMasking: Multiplicative masking of random bands
    - ChannelDropout: Random channel zeroing
    - LocalClipping: Local spectrum clipping

Mixing:
    - MixupAugmenter: Blend samples with KNN neighbors
    - LocalMixupAugmenter: Band-local mixup with KNN neighbors
    - ScatterSimulationMSC: Simple MSC-style scatter

Spline-based:
    - Spline_Smoothing: Smoothing spline augmentation
    - Spline_X_Perturbations: Wavelength axis perturbations via B-spline
    - Spline_Y_Perturbations: Intensity axis perturbations via B-spline
    - Spline_X_Simplification: Spectrum simplification on x-axis
    - Spline_Curve_Simplification: Spectrum simplification along curve length

Environmental Effects (require wavelengths):
    - TemperatureAugmenter: Simulate temperature-induced spectral changes
    - MoistureAugmenter: Simulate moisture/water activity effects

Scattering Effects (require wavelengths):
    - ParticleSizeAugmenter: Simulate particle size scattering
    - EMSCDistortionAugmenter: Apply EMSC-style distortions

Edge Artifacts (require wavelengths):
    - DetectorRollOffAugmenter: Simulate detector sensitivity roll-off at edges
    - StrayLightAugmenter: Simulate stray light effects (peak truncation)
    - EdgeCurvatureAugmenter: Simulate edge curvature/baseline bending
    - TruncatedPeakAugmenter: Add truncated peaks at spectral boundaries
    - EdgeArtifactsAugmenter: Combined edge artifacts augmenter

Synthesis-derived:
    - PathLengthAugmenter: Optical path length scaling
    - BatchEffectAugmenter: Batch/session/instrument effects
    - InstrumentalBroadeningAugmenter: Spectral resolution broadening
    - DeadBandAugmenter: Detector saturation simulation

Random/Geometric:
    - Rotate_Translate: Rotation and translation augmentation
    - Random_X_Operation: Random multiplicative/additive operations
"""

from .edge_artifacts import (
    DETECTOR_MODELS,
    DetectorRollOffAugmenter,
    EdgeArtifactsAugmenter,
    EdgeCurvatureAugmenter,
    StrayLightAugmenter,
    TruncatedPeakAugmenter,
)
from .environmental import (
    TEMPERATURE_REGION_PARAMS,
    MoistureAugmenter,
    TemperatureAugmenter,
)
from .random import Random_X_Operation, Rotate_Translate
from .scattering import (
    EMSCDistortionAugmenter,
    ParticleSizeAugmenter,
)
from .spectral import (
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
from .splines import (
    Spline_Curve_Simplification,
    Spline_Smoothing,
    Spline_X_Perturbations,
    Spline_X_Simplification,
    Spline_Y_Perturbations,
)
from .synthesis import (
    BatchEffectAugmenter,
    DeadBandAugmenter,
    HeteroscedasticNoiseAugmenter,
    InstrumentalBroadeningAugmenter,
    PathLengthAugmenter,
)

__all__ = [
    # Spectral noise and distortion
    "GaussianAdditiveNoise",
    "MultiplicativeNoise",
    "SpikeNoise",
    # Baseline effects
    "LinearBaselineDrift",
    "PolynomialBaselineDrift",
    # Wavelength distortions
    "WavelengthShift",
    "WavelengthStretch",
    "LocalWavelengthWarp",
    "SmoothMagnitudeWarp",
    # Spectral distortions
    "BandPerturbation",
    "GaussianSmoothingJitter",
    "UnsharpSpectralMask",
    "BandMasking",
    "ChannelDropout",
    "LocalClipping",
    # Mixing
    "MixupAugmenter",
    "LocalMixupAugmenter",
    "ScatterSimulationMSC",
    # Random augmentations
    "Random_X_Operation",
    "Rotate_Translate",
    # Spline-based augmentations
    "Spline_Curve_Simplification",
    "Spline_X_Simplification",
    "Spline_Y_Perturbations",
    "Spline_X_Perturbations",
    "Spline_Smoothing",
    # Environmental effects (require wavelengths)
    "TemperatureAugmenter",
    "MoistureAugmenter",
    "TEMPERATURE_REGION_PARAMS",
    # Scattering effects (require wavelengths)
    "ParticleSizeAugmenter",
    "EMSCDistortionAugmenter",
    # Edge artifacts (require wavelengths)
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
]
