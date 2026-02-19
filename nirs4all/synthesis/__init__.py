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
    >>> from nirs4all.synthesis import SyntheticNIRSGenerator
    >>>
    >>> # Simple generation
    >>> generator = SyntheticNIRSGenerator(random_state=42)
    >>> X, Y, E = generator.generate(n_samples=1000)
    >>>
    >>> # Create a SpectroDataset
    >>> dataset = generator.create_dataset(n_train=800, n_test=200)

    >>> # Use predefined components
    >>> from nirs4all.synthesis import ComponentLibrary
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

# Aggregate components (Phase 4 - Roadmap Phase 4)
from ._aggregates import (
    AGGREGATE_COMPONENTS,
    AggregateComponent,
    aggregate_info,
    expand_aggregate,
    get_aggregate,
    list_aggregates,
    validate_aggregates,
)
from ._aggregates import (
    list_categories as list_aggregate_categories,
)
from ._aggregates import (
    list_domains as list_aggregate_domains,
)

# Band assignments dictionary (comprehensive NIR band reference)
from ._bands import (
    # Band dictionary
    NIR_BANDS,
    # Core dataclass
    BandAssignment,
    band_info,
    generate_band_spectrum,
    # API functions
    get_band,
    get_bands_by_compound,
    get_bands_by_overtone,
    get_bands_by_tag,
    get_bands_in_range,
    list_all_tags,
    list_bands,
    list_functional_groups,
    validate_bands,
)
from ._bands import (
    summary as band_summary,
)

# Predefined components constant
from ._constants import (
    COMPLEXITY_PARAMS,
    DEFAULT_NIR_ZONES,
    DEFAULT_REALISTIC_COMPONENTS,
    DEFAULT_WAVELENGTH_END,
    DEFAULT_WAVELENGTH_START,
    DEFAULT_WAVELENGTH_STEP,
    get_predefined_components,
)

# GPU acceleration (Phase 4.4)
from .accelerated import (
    # Dataclasses
    AcceleratedArrays,
    # Classes
    AcceleratedGenerator,
    # Enums
    AcceleratorBackend,
    benchmark_backends,
    # Functions
    detect_best_backend,
    get_acceleration_speedup_estimate,
    get_backend_info,
    is_gpu_available,
)

# Benchmark datasets (Phase 4.2)
from .benchmarks import (
    # Registry
    BENCHMARK_DATASETS,
    # Dataclasses
    BenchmarkDatasetInfo,
    # Enums
    BenchmarkDomain,
    LoadedBenchmarkDataset,
    create_synthetic_matching_benchmark,
    get_benchmark_info,
    get_benchmark_spectral_properties,
    get_datasets_by_domain,
    # Functions
    list_benchmark_datasets,
    load_benchmark_dataset,
)

# Builder for fluent construction
from .builder import SyntheticDatasetBuilder

# Spectral components
from .components import (
    ComponentLibrary,
    NIRBand,
    SpectralComponent,
    # Discovery API (Phase 1 enhancement)
    available_components,
    component_info,
    get_component,
    list_categories,
    normalize_component_amplitudes,
    search_components,
    validate_component_coverage,
    validate_predefined_components,
)

# Configuration classes
from .config import (
    BatchEffectConfig,
    ComplexityLevel,
    ConcentrationMethod,
    FeatureConfig,
    MetadataConfig,
    OutputConfig,
    PartitionConfig,
    SyntheticDatasetConfig,
    TargetConfig,
)

# Detector models and noise (Phase 2.3)
from .detectors import (
    DETECTOR_NOISE_DEFAULTS,
    DETECTOR_RESPONSES,
    DetectorConfig,
    # Simulator
    DetectorSimulator,
    # Response curves
    DetectorSpectralResponse,
    # Noise configuration
    NoiseModelConfig,
    get_default_noise_config,
    get_detector_response,
    get_detector_wavelength_range,
    list_detector_types,
    # Convenience functions
    simulate_detector_effects,
)

# Application domain priors (Phase 1.3)
from .domains import (
    # Domain registry
    APPLICATION_DOMAINS,
    # Configuration classes
    ConcentrationPrior,
    # Domain categories
    DomainCategory,
    DomainConfig,
    create_domain_aware_library,
    get_domain_components,
    # Utility functions
    get_domain_config,
    get_domains_for_component,
    list_domains,
)

# ================================================================
# Phase 3: Matrix and Environmental Effects
# ================================================================
# Environmental effects (Phase 3.1, 3.4) - Configuration classes only
# NOTE: Use nirs4all.operators.augmentation for applying effects:
#   - TemperatureAugmenter for temperature effects
#   - MoistureAugmenter for moisture effects
from .environmental import (
    # Constants
    TEMPERATURE_EFFECT_PARAMS,
    EnvironmentalEffectsConfig,
    MoistureConfig,
    # Enums
    SpectralRegion,
    TemperatureConfig,
    # Dataclasses
    TemperatureEffectParams,
    # Utility functions
    get_temperature_effect_regions,
)

# Export capabilities (Phase 4)
from .exporter import (
    CSVVariationGenerator,
    DatasetExporter,
    ExportConfig,
    export_to_csv,
    export_to_folder,
)

# Real data fitting (Phase 4)
from .fitter import (
    COMPONENT_CATEGORIES,
    EXCLUDED_COMPONENTS,
    UNIVERSAL_COMPONENTS,
    ComponentFitResult,
    # Phase 5: Spectral fitting tools
    ComponentFitter,
    DerivativeAwareForwardModelFitter,
    DomainInference,
    EnvironmentalInference,
    FittedParameters,
    ForwardModelFitter,
    # Physical forward model fitting
    InstrumentChain,
    # Phase 1-4 enhanced inference classes
    InstrumentInference,
    MeasurementModeInference,
    OperatorVarianceParams,
    # Phase 5: Optimized spectral fitting (greedy selection)
    OptimizedComponentFitter,
    OptimizedFitResult,
    PCAVarianceParams,
    PreprocessingInference,
    # Phase 5: Preprocessing detection
    PreprocessingType,
    RealBandFitResult,
    # Phase 5: Real band fitting (NIR_BANDS)
    RealBandFitter,
    RealDataFitter,
    ScatteringInference,
    SpectralProperties,
    VarianceFitResult,
    # Phase 5: Variance fitting
    VarianceFitter,
    compare_datasets,
    compute_spectral_properties,
    fit_components,
    fit_components_optimized,
    fit_real_bands,
    fit_to_real_data,
    fit_variance,
    multiscale_derivative_fit,
    multiscale_fit,
)

# Core generator
from .generator import SyntheticNIRSGenerator

# ================================================================
# Phase 2: Instrument Simulation Enhancement
# ================================================================
# Instrument archetypes and simulation (Phase 2.1)
from .instruments import (
    # Registry and utilities
    INSTRUMENT_ARCHETYPES,
    # Phase 6: Instrument wavelength grids
    INSTRUMENT_WAVELENGTHS,
    DetectorType,
    EdgeArtifactsConfig,
    InstrumentArchetype,
    # Enums
    InstrumentCategory,
    # Simulator
    InstrumentSimulator,
    MonochromatorType,
    MultiScanConfig,
    MultiSensorConfig,
    # Dataclasses
    SensorConfig,
    get_instrument_archetype,
    get_instrument_wavelength_info,
    get_instrument_wavelengths,
    get_instruments_by_category,
    list_instrument_archetypes,
    list_instrument_wavelength_grids,
)

# Measurement modes and physics (Phase 2.2)
from .measurement_modes import (
    ATRConfig,
    # Enums
    MeasurementMode,
    # Simulator
    MeasurementModeSimulator,
    ReflectanceConfig,
    ScatteringConfig,
    TransflectanceConfig,
    # Configurations
    TransmittanceConfig,
    create_atr_simulator,
    create_reflectance_simulator,
    # Factory functions
    create_transmittance_simulator,
)

# Metadata generation (Phase 3)
from .metadata import (
    MetadataGenerationResult,
    MetadataGenerator,
    generate_sample_metadata,
)

# Conditional prior sampling (Phase 4.3)
from .prior import (
    # Enums
    MatrixType,
    # Dataclasses
    NIRSPriorConfig,
    # Classes
    PriorSampler,
    get_domain_compatible_instruments,
    get_instrument_typical_modes,
    # Convenience functions
    sample_prior,
    sample_prior_batch,
)

# Procedural component generator (Phase 1.2)
from .procedural import (
    FUNCTIONAL_GROUP_PROPERTIES,
    # Functional group types and properties
    FunctionalGroupType,
    # Configuration
    ProceduralComponentConfig,
    # Generator class
    ProceduralComponentGenerator,
)

# Product-level generation (Phase 7 - Roadmap Phase 7)
from .products import (
    # Registry
    PRODUCT_TEMPLATES,
    CategoryGenerator,
    # Dataclasses
    ComponentVariation,
    # Generator classes
    ProductGenerator,
    ProductTemplate,
    # Enums
    VariationType,
    generate_product_samples,
    get_product_template,
    list_product_categories,
    list_product_domains,
    # Convenience functions
    list_product_templates,
    product_template_info,
)

# Scattering effects (Phase 3.2, 3.3) - Configuration classes only
# NOTE: Use nirs4all.operators.augmentation for applying effects:
#   - ParticleSizeAugmenter for particle size effects
#   - EMSCDistortionAugmenter for EMSC-style distortions
from .scattering import (
    EMSCConfig,
    ParticleSizeConfig,
    # Dataclasses
    ParticleSizeDistribution,
    ScatteringCoefficientConfig,
    ScatteringEffectsConfig,
    # Enums
    ScatteringModel,
)

# Multi-source generation (Phase 3)
from .sources import (
    MultiSourceGenerator,
    MultiSourceResult,
    SourceConfig,
    generate_multi_source,
)

# Target generation (Phase 3)
from .targets import (
    ClassSeparationConfig,
    TargetGenerator,
    generate_classification_targets,
    generate_regression_targets,
)

# Validation utilities
# ================================================================
# Phase 4: Validation and Infrastructure
# ================================================================
# Spectral realism scorecard (Phase 4.1)
from .validation import (
    DatasetComparisonResult,
    # Dataclasses
    MetricResult,
    # Enums
    RealismMetric,
    SpectralRealismScore,
    ValidationError,
    compute_adversarial_validation_auc,
    compute_baseline_curvature,
    # Core metric functions
    compute_correlation_length,
    compute_derivative_statistics,
    compute_distribution_overlap,
    compute_peak_density,
    compute_snr,
    compute_spectral_realism_scorecard,
    quick_realism_check,
    validate_against_benchmark,
    validate_concentrations,
    validate_spectra,
    validate_synthetic_output,
    validate_wavelengths,
)

# ================================================================
# Phase 1: Enhanced Component Generation
# ================================================================
# Wavenumber-based band placement utilities (Phase 1.1)
from .wavenumber import (
    # Visible-NIR extended zones (Phase 2)
    EXTENDED_SPECTRAL_ZONES,
    # Overtone and combination band calculations
    FUNDAMENTAL_VIBRATIONS,
    # NIR zones and regions
    NIR_ZONES_WAVENUMBER,
    VISIBLE_ZONES_WAVENUMBER,
    CombinationBandResult,
    # Result dataclasses
    OvertoneResult,
    apply_hydrogen_bonding_shift,
    calculate_combination_band,
    calculate_overtone_position,
    classify_wavelength_extended,
    classify_wavelength_zone,
    convert_bandwidth_to_wavelength,
    get_all_zones_extended,
    get_all_zones_wavelength,
    get_zone_wavelength_range,
    is_nir_region,
    is_visible_region,
    wavelength_to_wavenumber,
    # Core conversion functions
    wavenumber_to_wavelength,
)
from .wavenumber import (
    classify_wavelength_zone as get_nir_zone,  # Alias for backward compatibility
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
    # Phase 5: Optimized spectral fitting (greedy selection)
    "OptimizedComponentFitter",
    "OptimizedFitResult",
    "fit_components_optimized",
    "COMPONENT_CATEGORIES",
    "EXCLUDED_COMPONENTS",
    "UNIVERSAL_COMPONENTS",
    # Phase 5: Preprocessing detection
    "PreprocessingType",
    "PreprocessingInference",
    # Phase 5: Real band fitting (NIR_BANDS)
    "RealBandFitter",
    "RealBandFitResult",
    "fit_real_bands",
    # Phase 5: Variance fitting
    "VarianceFitter",
    "VarianceFitResult",
    "OperatorVarianceParams",
    "PCAVarianceParams",
    "fit_variance",
    # Physical forward model fitting
    "InstrumentChain",
    "ForwardModelFitter",
    "DerivativeAwareForwardModelFitter",
    "multiscale_fit",
    "multiscale_derivative_fit",
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
