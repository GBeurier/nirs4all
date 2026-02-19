"""
Augmentation operators for spectral data.

This module provides data augmentation operators for NIRS spectra, including:

Noise and Distortion:
    - GaussianAdditiveNoise: Add Gaussian noise
    - MultiplicativeNoise: Apply random gain factors

Baseline Effects:
    - LinearBaselineDrift: Add linear baseline
    - PolynomialBaselineDrift: Add polynomial baseline

Wavelength Distortions:
    - WavelengthShift: Shift spectra along wavelength axis
    - WavelengthStretch: Stretch/compress wavelength axis

Environmental Effects (require wavelengths):
    - TemperatureAugmenter: Simulate temperature-induced spectral changes
    - MoistureAugmenter: Simulate moisture/water activity effects

Scattering Effects (require wavelengths):
    - ParticleSizeAugmenter: Simulate particle size scattering
    - EMSCDistortionAugmenter: Apply EMSC-style distortions
    - ScatterSimulationMSC: Simple MSC-style scatter (legacy)

Edge Artifacts (require wavelengths):
    - DetectorRollOffAugmenter: Simulate detector sensitivity roll-off at edges
    - StrayLightAugmenter: Simulate stray light effects (peak truncation)
    - EdgeCurvatureAugmenter: Simulate edge curvature/baseline bending
    - TruncatedPeakAugmenter: Add truncated peaks at spectral boundaries
    - EdgeArtifactsAugmenter: Combined edge artifacts augmenter
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
