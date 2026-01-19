"""
Physical signal-chain reconstruction and variance modeling for NIR spectra.

This module implements a physically realistic "full signal-chain" reconstruction
workflow that:
1. Reconstructs spectra using a physical forward model (Beer-Lambert + instrument chain)
2. Learns distributions of physical parameters for variance modeling
3. Generates realistic synthetic datasets by sampling from learned distributions

Key Components:
    - CanonicalForwardModel: Physical model on canonical grid
    - InstrumentModel: Wavelength warp, ILS convolution, gain/offset
    - EnvironmentalEffectsModel: Temperature, moisture, and scattering effects
    - DomainModel: Absorbance/reflectance transformation
    - PreprocessingOperator: Match dataset preprocessing (SG derivatives, SNV, etc.)
    - VariableProjectionSolver: NNLS inner solve + nonlinear outer optimization
    - GlobalCalibrator: Prototype-based instrument parameter estimation
    - ParameterDistributionFitter: Learn distributions in parameter space
    - ReconstructionGenerator: Generate synthetic data from learned distributions

Example:
    >>> from nirs4all.data.synthetic.reconstruction import (
    ...     ReconstructionPipeline,
    ...     DatasetConfig,
    ... )
    >>>
    >>> # Configure for a dataset
    >>> config = DatasetConfig(
    ...     wavelengths=wavelengths,
    ...     signal_type="absorbance",
    ...     preprocessing="first_derivative",
    ...     domain="food_dairy",
    ... )
    >>>
    >>> # Run full reconstruction pipeline
    >>> pipeline = ReconstructionPipeline(config)
    >>> result = pipeline.fit(X_real)
    >>>
    >>> # Generate synthetic data
    >>> X_synth = pipeline.generate(n_samples=1000)

References:
    - Burns, D. A., & Ciurczak, E. W. (2007). Handbook of Near-Infrared Analysis.
    - Workman Jr, J., & Weyer, L. (2012). Practical Guide and Spectral Atlas for
      Interpretive Near-Infrared Spectroscopy.
"""

from .forward import (
    CanonicalForwardModel,
    InstrumentModel,
    DomainTransform,
    PreprocessingOperator,
    ForwardChain,
)
from .environmental import (
    EnvironmentalEffectsModel,
    EnvironmentalParameterConfig,
)
from .calibration import (
    PrototypeSelector,
    GlobalCalibrator,
    CalibrationResult,
)
from .inversion import (
    VariableProjectionSolver,
    InversionResult,
    MultiscaleSchedule,
)
from .distributions import (
    ParameterDistributionFitter,
    ParameterSampler,
    DistributionResult,
)
from .generator import (
    ReconstructionGenerator,
    GenerationResult,
)
from .validation import (
    ReconstructionValidator,
    ValidationResult,
)
from .pipeline import (
    DatasetConfig,
    ReconstructionPipeline,
    PipelineResult,
    reconstruct_and_generate,
)

__all__ = [
    # Forward model
    "CanonicalForwardModel",
    "InstrumentModel",
    "DomainTransform",
    "PreprocessingOperator",
    "ForwardChain",
    # Environmental effects
    "EnvironmentalEffectsModel",
    "EnvironmentalParameterConfig",
    # Calibration
    "PrototypeSelector",
    "GlobalCalibrator",
    "CalibrationResult",
    # Inversion
    "VariableProjectionSolver",
    "InversionResult",
    "MultiscaleSchedule",
    # Distributions
    "ParameterDistributionFitter",
    "ParameterSampler",
    "DistributionResult",
    # Generator
    "ReconstructionGenerator",
    "GenerationResult",
    # Validation
    "ReconstructionValidator",
    "ValidationResult",
    # Pipeline
    "DatasetConfig",
    "ReconstructionPipeline",
    "PipelineResult",
    "reconstruct_and_generate",
]
