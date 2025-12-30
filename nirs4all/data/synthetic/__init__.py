"""
Synthetic NIRS Data Generation Module.

This module provides tools for generating realistic synthetic NIRS spectra
for testing, examples, benchmarking, and ML research.

Key Features:
    - Physically-motivated generation based on Beer-Lambert law
    - Voigt profile peak shapes (Gaussian + Lorentzian convolution)
    - Realistic NIR band positions from known spectroscopic databases
    - Configurable complexity levels (simple, realistic, complex)
    - Batch/session effects for domain adaptation research
    - Direct SpectroDataset creation for pipeline integration

Quick Start:
    >>> from nirs4all.data.synthetic import SyntheticNIRSGenerator
    >>>
    >>> # Simple generation
    >>> generator = SyntheticNIRSGenerator(random_state=42)
    >>> X, Y, E = generator.generate(n_samples=1000)
    >>>
    >>> # Create a SpectroDataset
    >>> dataset = generator.create_dataset(n_train=800, n_test=200)

    >>> # Use predefined components
    >>> from nirs4all.data.synthetic import ComponentLibrary
    >>> library = ComponentLibrary.from_predefined(["water", "protein", "lipid"])
    >>> generator = SyntheticNIRSGenerator(component_library=library)

See Also:
    - nirs4all.generate: Top-level generation API
    - SyntheticDatasetBuilder: Fluent dataset construction

References:
    - Workman Jr, J., & Weyer, L. (2012). Practical Guide and Spectral Atlas
      for Interpretive Near-Infrared Spectroscopy. CRC Press.
    - Burns, D. A., & Ciurczak, E. W. (2007). Handbook of Near-Infrared
      Analysis. CRC Press.
"""

from __future__ import annotations

# Core generator
from .generator import SyntheticNIRSGenerator

# Builder for fluent construction
from .builder import SyntheticDatasetBuilder

# Spectral components
from .components import (
    NIRBand,
    SpectralComponent,
    ComponentLibrary,
)

# Predefined components constant
from ._constants import (
    get_predefined_components,
    COMPLEXITY_PARAMS,
    DEFAULT_WAVELENGTH_START,
    DEFAULT_WAVELENGTH_END,
    DEFAULT_WAVELENGTH_STEP,
    DEFAULT_NIR_ZONES,
    DEFAULT_REALISTIC_COMPONENTS,
)

# Configuration classes
from .config import (
    SyntheticDatasetConfig,
    FeatureConfig,
    TargetConfig,
    MetadataConfig,
    PartitionConfig,
    BatchEffectConfig,
    OutputConfig,
    ComplexityLevel,
    ConcentrationMethod,
)

# Validation utilities
from .validation import (
    ValidationError,
    validate_spectra,
    validate_concentrations,
    validate_wavelengths,
    validate_synthetic_output,
)

# Metadata generation (Phase 3)
from .metadata import (
    MetadataGenerator,
    MetadataGenerationResult,
    generate_sample_metadata,
)

# Target generation (Phase 3)
from .targets import (
    TargetGenerator,
    ClassSeparationConfig,
    generate_regression_targets,
    generate_classification_targets,
)

# Multi-source generation (Phase 3)
from .sources import (
    MultiSourceGenerator,
    SourceConfig,
    MultiSourceResult,
    generate_multi_source,
)

# Export capabilities (Phase 4)
from .exporter import (
    DatasetExporter,
    CSVVariationGenerator,
    ExportConfig,
    export_to_folder,
    export_to_csv,
)

# Real data fitting (Phase 4)
from .fitter import (
    RealDataFitter,
    FittedParameters,
    SpectralProperties,
    compute_spectral_properties,
    fit_to_real_data,
    compare_datasets,
)


# Backward-compatible alias for predefined components
# Note: This is a function call, not a constant, to avoid circular imports
def _get_predefined_components():
    """Get predefined components (lazy loading to avoid circular imports)."""
    return get_predefined_components()


# Make PREDEFINED_COMPONENTS available as a module-level name for backward compat
# Users should prefer get_predefined_components() for explicit behavior
class _PredefinedComponentsProxy:
    """Proxy object for lazy loading of predefined components."""

    def __getitem__(self, key):
        return get_predefined_components()[key]

    def __iter__(self):
        return iter(get_predefined_components())

    def __len__(self):
        return len(get_predefined_components())

    def keys(self):
        return get_predefined_components().keys()

    def values(self):
        return get_predefined_components().values()

    def items(self):
        return get_predefined_components().items()

    def __contains__(self, key):
        return key in get_predefined_components()

    def __repr__(self):
        return repr(get_predefined_components())


PREDEFINED_COMPONENTS = _PredefinedComponentsProxy()


__all__ = [
    # Core generator
    "SyntheticNIRSGenerator",
    # Builder
    "SyntheticDatasetBuilder",
    # Components
    "NIRBand",
    "SpectralComponent",
    "ComponentLibrary",
    "PREDEFINED_COMPONENTS",
    "get_predefined_components",
    # Configuration
    "SyntheticDatasetConfig",
    "FeatureConfig",
    "TargetConfig",
    "MetadataConfig",
    "PartitionConfig",
    "BatchEffectConfig",
    "OutputConfig",
    "ComplexityLevel",
    "ConcentrationMethod",
    # Constants
    "COMPLEXITY_PARAMS",
    "DEFAULT_WAVELENGTH_START",
    "DEFAULT_WAVELENGTH_END",
    "DEFAULT_WAVELENGTH_STEP",
    "DEFAULT_NIR_ZONES",
    "DEFAULT_REALISTIC_COMPONENTS",
    # Validation
    "ValidationError",
    "validate_spectra",
    "validate_concentrations",
    "validate_wavelengths",
    "validate_synthetic_output",
    # Metadata (Phase 3)
    "MetadataGenerator",
    "MetadataGenerationResult",
    "generate_sample_metadata",
    # Targets (Phase 3)
    "TargetGenerator",
    "ClassSeparationConfig",
    "generate_regression_targets",
    "generate_classification_targets",
    # Multi-source (Phase 3)
    "MultiSourceGenerator",
    "SourceConfig",
    "MultiSourceResult",
    "generate_multi_source",
    # Export (Phase 4)
    "DatasetExporter",
    "CSVVariationGenerator",
    "ExportConfig",
    "export_to_folder",
    "export_to_csv",
    # Fitting (Phase 4)
    "RealDataFitter",
    "FittedParameters",
    "SpectralProperties",
    "compute_spectral_properties",
    "fit_to_real_data",
    "compare_datasets",
]
