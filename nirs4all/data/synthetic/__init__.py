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
    # Discovery API (Phase 1 enhancement)
    available_components,
    get_component,
    search_components,
    list_categories,
    component_info,
    validate_predefined_components,
    validate_component_coverage,
    normalize_component_amplitudes,
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
    # Phase 1-4 enhanced inference classes
    InstrumentInference,
    DomainInference,
    EnvironmentalInference,
    ScatteringInference,
    MeasurementModeInference,
    # Phase 5: Spectral fitting tools
    ComponentFitter,
    ComponentFitResult,
    fit_components,
    # Phase 5: Preprocessing detection
    PreprocessingType,
    PreprocessingInference,
)

# ================================================================
# Phase 1: Enhanced Component Generation
# ================================================================

# Wavenumber-based band placement utilities (Phase 1.1)
from .wavenumber import (
    # Core conversion functions
    wavenumber_to_wavelength,
    wavelength_to_wavenumber,
    convert_bandwidth_to_wavelength,
    # NIR zones and regions
    NIR_ZONES_WAVENUMBER,
    classify_wavelength_zone,
    classify_wavelength_zone as get_nir_zone,  # Alias for backward compatibility
    get_zone_wavelength_range,
    get_all_zones_wavelength,
    # Visible-NIR extended zones (Phase 2)
    EXTENDED_SPECTRAL_ZONES,
    VISIBLE_ZONES_WAVENUMBER,
    classify_wavelength_extended,
    get_all_zones_extended,
    is_visible_region,
    is_nir_region,
    # Overtone and combination band calculations
    FUNDAMENTAL_VIBRATIONS,
    calculate_overtone_position,
    calculate_combination_band,
    apply_hydrogen_bonding_shift,
    # Result dataclasses
    OvertoneResult,
    CombinationBandResult,
)

# Procedural component generator (Phase 1.2)
from .procedural import (
    # Functional group types and properties
    FunctionalGroupType,
    FUNCTIONAL_GROUP_PROPERTIES,
    # Configuration
    ProceduralComponentConfig,
    # Generator class
    ProceduralComponentGenerator,
)

# Application domain priors (Phase 1.3)
from .domains import (
    # Domain categories
    DomainCategory,
    # Configuration classes
    ConcentrationPrior,
    DomainConfig,
    # Domain registry
    APPLICATION_DOMAINS,
    # Utility functions
    get_domain_config,
    list_domains,
    get_domain_components,
    get_domains_for_component,
    create_domain_aware_library,
)

# Aggregate components (Phase 4 - Roadmap Phase 4)
from ._aggregates import (
    AggregateComponent,
    AGGREGATE_COMPONENTS,
    get_aggregate,
    list_aggregates,
    expand_aggregate,
    aggregate_info,
    list_domains as list_aggregate_domains,
    list_categories as list_aggregate_categories,
    validate_aggregates,
)

# Band assignments dictionary (comprehensive NIR band reference)
from ._bands import (
    # Core dataclass
    BandAssignment,
    # Band dictionary
    NIR_BANDS,
    # API functions
    get_band,
    list_functional_groups,
    list_bands,
    get_bands_in_range,
    get_bands_by_tag,
    get_bands_by_overtone,
    get_bands_by_compound,
    generate_band_spectrum,
    band_info,
    list_all_tags,
    validate_bands,
    summary as band_summary,
)

# Product-level generation (Phase 7 - Roadmap Phase 7)
from .products import (
    # Enums
    VariationType,
    # Dataclasses
    ComponentVariation,
    ProductTemplate,
    # Registry
    PRODUCT_TEMPLATES,
    # Generator classes
    ProductGenerator,
    CategoryGenerator,
    # Convenience functions
    list_product_templates,
    get_product_template,
    generate_product_samples,
    product_template_info,
    list_product_categories,
    list_product_domains,
)


# ================================================================
# Phase 2: Instrument Simulation Enhancement
# ================================================================

# Instrument archetypes and simulation (Phase 2.1)
from .instruments import (
    # Enums
    InstrumentCategory,
    DetectorType,
    MonochromatorType,
    # Dataclasses
    SensorConfig,
    MultiSensorConfig,
    MultiScanConfig,
    EdgeArtifactsConfig,
    InstrumentArchetype,
    # Registry and utilities
    INSTRUMENT_ARCHETYPES,
    get_instrument_archetype,
    list_instrument_archetypes,
    get_instruments_by_category,
    # Simulator
    InstrumentSimulator,
    # Phase 6: Instrument wavelength grids
    INSTRUMENT_WAVELENGTHS,
    get_instrument_wavelengths,
    list_instrument_wavelength_grids,
    get_instrument_wavelength_info,
)

# Measurement modes and physics (Phase 2.2)
from .measurement_modes import (
    # Enums
    MeasurementMode,
    # Configurations
    TransmittanceConfig,
    ReflectanceConfig,
    TransflectanceConfig,
    ATRConfig,
    ScatteringConfig,
    # Simulator
    MeasurementModeSimulator,
    # Factory functions
    create_transmittance_simulator,
    create_reflectance_simulator,
    create_atr_simulator,
)

# Detector models and noise (Phase 2.3)
from .detectors import (
    # Response curves
    DetectorSpectralResponse,
    DETECTOR_RESPONSES,
    get_detector_response,
    # Noise configuration
    NoiseModelConfig,
    DetectorConfig,
    DETECTOR_NOISE_DEFAULTS,
    get_default_noise_config,
    # Simulator
    DetectorSimulator,
    # Convenience functions
    simulate_detector_effects,
    get_detector_wavelength_range,
    list_detector_types,
)

# ================================================================
# Phase 3: Matrix and Environmental Effects
# ================================================================

# Environmental effects (Phase 3.1, 3.4) - Configuration classes only
# NOTE: Use nirs4all.operators.augmentation for applying effects:
#   - TemperatureAugmenter for temperature effects
#   - MoistureAugmenter for moisture effects
from .environmental import (
    # Enums
    SpectralRegion,
    # Dataclasses
    TemperatureEffectParams,
    TemperatureConfig,
    MoistureConfig,
    EnvironmentalEffectsConfig,
    # Constants
    TEMPERATURE_EFFECT_PARAMS,
    # Utility functions
    get_temperature_effect_regions,
)

# Scattering effects (Phase 3.2, 3.3) - Configuration classes only
# NOTE: Use nirs4all.operators.augmentation for applying effects:
#   - ParticleSizeAugmenter for particle size effects
#   - EMSCDistortionAugmenter for EMSC-style distortions
from .scattering import (
    # Enums
    ScatteringModel,
    # Dataclasses
    ParticleSizeDistribution,
    ParticleSizeConfig,
    EMSCConfig,
    ScatteringCoefficientConfig,
    ScatteringEffectsConfig,
)

# ================================================================
# Phase 4: Validation and Infrastructure
# ================================================================

# Spectral realism scorecard (Phase 4.1)
from .validation import (
    # Enums
    RealismMetric,
    # Dataclasses
    MetricResult,
    SpectralRealismScore,
    DatasetComparisonResult,
    # Core metric functions
    compute_correlation_length,
    compute_derivative_statistics,
    compute_peak_density,
    compute_baseline_curvature,
    compute_snr,
    compute_distribution_overlap,
    compute_adversarial_validation_auc,
    compute_spectral_realism_scorecard,
    validate_against_benchmark,
    quick_realism_check,
)

# Benchmark datasets (Phase 4.2)
from .benchmarks import (
    # Enums
    BenchmarkDomain,
    # Dataclasses
    BenchmarkDatasetInfo,
    LoadedBenchmarkDataset,
    # Registry
    BENCHMARK_DATASETS,
    # Functions
    list_benchmark_datasets,
    get_benchmark_info,
    get_datasets_by_domain,
    load_benchmark_dataset,
    get_benchmark_spectral_properties,
    create_synthetic_matching_benchmark,
)

# Conditional prior sampling (Phase 4.3)
from .prior import (
    # Enums
    MatrixType,
    # Dataclasses
    NIRSPriorConfig,
    # Classes
    PriorSampler,
    # Convenience functions
    sample_prior,
    sample_prior_batch,
    get_domain_compatible_instruments,
    get_instrument_typical_modes,
)

# GPU acceleration (Phase 4.4)
from .accelerated import (
    # Enums
    AcceleratorBackend,
    # Dataclasses
    AcceleratedArrays,
    # Classes
    AcceleratedGenerator,
    # Functions
    detect_best_backend,
    get_backend_info,
    is_gpu_available,
    get_acceleration_speedup_estimate,
    benchmark_backends,
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
    # Discovery API (Phase 1 enhancement)
    "available_components",
    "get_component",
    "search_components",
    "list_categories",
    "component_info",
    "validate_predefined_components",
    "validate_component_coverage",
    "normalize_component_amplitudes",
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
    # Phase 1-4 enhanced inference classes
    "InstrumentInference",
    "DomainInference",
    "EnvironmentalInference",
    "ScatteringInference",
    "MeasurementModeInference",
    # Phase 5: Spectral fitting tools
    "ComponentFitter",
    "ComponentFitResult",
    "fit_components",
    # Phase 5: Preprocessing detection
    "PreprocessingType",
    "PreprocessingInference",
    # ================================================================
    # Phase 1: Enhanced Component Generation
    # ================================================================
    # Wavenumber utilities (Phase 1.1)
    "wavenumber_to_wavelength",
    "wavelength_to_wavenumber",
    "convert_bandwidth_to_wavelength",
    "NIR_ZONES_WAVENUMBER",
    "classify_wavelength_zone",
    "get_nir_zone",  # Alias for classify_wavelength_zone
    "get_zone_wavelength_range",
    "get_all_zones_wavelength",
    # Visible-NIR extended zones (Phase 2)
    "EXTENDED_SPECTRAL_ZONES",
    "VISIBLE_ZONES_WAVENUMBER",
    "classify_wavelength_extended",
    "get_all_zones_extended",
    "is_visible_region",
    "is_nir_region",
    # Overtone/combination calculations
    "FUNDAMENTAL_VIBRATIONS",
    "calculate_overtone_position",
    "calculate_combination_band",
    "apply_hydrogen_bonding_shift",
    "OvertoneResult",
    "CombinationBandResult",
    # Procedural generator (Phase 1.2)
    "FunctionalGroupType",
    "FUNCTIONAL_GROUP_PROPERTIES",
    "ProceduralComponentConfig",
    "ProceduralComponentGenerator",
    # Domain priors (Phase 1.3)
    "DomainCategory",
    "ConcentrationPrior",
    "DomainConfig",
    "APPLICATION_DOMAINS",
    "get_domain_config",
    "list_domains",
    "get_domain_components",
    "get_domains_for_component",
    "create_domain_aware_library",
    # Aggregate components (Roadmap Phase 4)
    "AggregateComponent",
    "AGGREGATE_COMPONENTS",
    "get_aggregate",
    "list_aggregates",
    "expand_aggregate",
    "aggregate_info",
    "list_aggregate_domains",
    "list_aggregate_categories",
    "validate_aggregates",
    # Band assignments dictionary
    "BandAssignment",
    "NIR_BANDS",
    "get_band",
    "list_functional_groups",
    "list_bands",
    "get_bands_in_range",
    "get_bands_by_tag",
    "get_bands_by_overtone",
    "get_bands_by_compound",
    "generate_band_spectrum",
    "band_info",
    "list_all_tags",
    "validate_bands",
    "band_summary",
    # Product-level generation (Roadmap Phase 7)
    "VariationType",
    "ComponentVariation",
    "ProductTemplate",
    "PRODUCT_TEMPLATES",
    "ProductGenerator",
    "CategoryGenerator",
    "list_product_templates",
    "get_product_template",
    "generate_product_samples",
    "product_template_info",
    "list_product_categories",
    "list_product_domains",
    # ================================================================
    # Phase 2: Instrument Simulation Enhancement
    # ================================================================
    # Instruments (Phase 2.1)
    "InstrumentCategory",
    "DetectorType",
    "MonochromatorType",
    "SensorConfig",
    "MultiSensorConfig",
    "MultiScanConfig",
    "EdgeArtifactsConfig",
    "InstrumentArchetype",
    "INSTRUMENT_ARCHETYPES",
    "get_instrument_archetype",
    "list_instrument_archetypes",
    "get_instruments_by_category",
    "InstrumentSimulator",
    # Phase 6: Instrument wavelength grids
    "INSTRUMENT_WAVELENGTHS",
    "get_instrument_wavelengths",
    "list_instrument_wavelength_grids",
    "get_instrument_wavelength_info",
    # Measurement modes (Phase 2.2)
    "MeasurementMode",
    "TransmittanceConfig",
    "ReflectanceConfig",
    "TransflectanceConfig",
    "ATRConfig",
    "ScatteringConfig",
    "MeasurementModeSimulator",
    "create_transmittance_simulator",
    "create_reflectance_simulator",
    "create_atr_simulator",
    # Detectors (Phase 2.3)
    "DetectorSpectralResponse",
    "DETECTOR_RESPONSES",
    "get_detector_response",
    "NoiseModelConfig",
    "DetectorConfig",
    "DETECTOR_NOISE_DEFAULTS",
    "get_default_noise_config",
    "DetectorSimulator",
    "simulate_detector_effects",
    "get_detector_wavelength_range",
    "list_detector_types",
    # ================================================================
    # Phase 3: Matrix and Environmental Effects
    # ================================================================
    # Environmental effects (Phase 3.1, 3.4) - Configuration classes
    # Use nirs4all.operators.augmentation for applying effects
    "SpectralRegion",
    "TemperatureEffectParams",
    "TemperatureConfig",
    "MoistureConfig",
    "EnvironmentalEffectsConfig",
    "TEMPERATURE_EFFECT_PARAMS",
    "get_temperature_effect_regions",
    # Scattering effects (Phase 3.2, 3.3) - Configuration classes
    # Use nirs4all.operators.augmentation for applying effects
    "ScatteringModel",
    "ParticleSizeDistribution",
    "ParticleSizeConfig",
    "EMSCConfig",
    "ScatteringCoefficientConfig",
    "ScatteringEffectsConfig",
    # ================================================================
    # Phase 4: Validation and Infrastructure
    # ================================================================
    # Spectral realism scorecard (Phase 4.1)
    "RealismMetric",
    "MetricResult",
    "SpectralRealismScore",
    "compute_correlation_length",
    "compute_derivative_statistics",
    "compute_peak_density",
    "compute_baseline_curvature",
    "compute_snr",
    "compute_distribution_overlap",
    "compute_adversarial_validation_auc",
    "compute_spectral_realism_scorecard",
    "DatasetComparisonResult",
    "validate_against_benchmark",
    "quick_realism_check",
    # Benchmark datasets (Phase 4.2)
    "BenchmarkDomain",
    "BenchmarkDatasetInfo",
    "BENCHMARK_DATASETS",
    "list_benchmark_datasets",
    "get_benchmark_info",
    "get_datasets_by_domain",
    "LoadedBenchmarkDataset",
    "load_benchmark_dataset",
    "get_benchmark_spectral_properties",
    "create_synthetic_matching_benchmark",
    # Conditional prior sampling (Phase 4.3)
    "MatrixType",
    "NIRSPriorConfig",
    "PriorSampler",
    "sample_prior",
    "sample_prior_batch",
    "get_domain_compatible_instruments",
    "get_instrument_typical_modes",
    # GPU acceleration (Phase 4.4)
    "AcceleratorBackend",
    "AcceleratedArrays",
    "AcceleratedGenerator",
    "detect_best_backend",
    "get_backend_info",
    "is_gpu_available",
    "get_acceleration_speedup_estimate",
    "benchmark_backends",
]
