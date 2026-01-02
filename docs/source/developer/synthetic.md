# Synthetic Data Generator

This guide covers the internals of the synthetic NIRS data generator for developers who want to extend or customize it.

## Architecture Overview

```
nirs4all/data/synthetic/
├── __init__.py           # Public API exports
├── generator.py          # SyntheticNIRSGenerator (core engine)
├── builder.py            # SyntheticDatasetBuilder (fluent API)
├── components.py         # NIRBand, SpectralComponent, ComponentLibrary
├── config.py             # Configuration dataclasses
├── _constants.py         # Predefined components (111 total), defaults
├── metadata.py           # MetadataGenerator
├── targets.py            # TargetGenerator (regression/classification)
├── sources.py            # MultiSourceGenerator
├── exporter.py           # DatasetExporter, CSVVariationGenerator
├── fitter.py             # RealDataFitter
├── validation.py         # Data validation utilities
│
├── # === Phase 1: Enhanced Component Generation ===
├── wavenumber.py         # Wavenumber utilities, overtone/combination bands
├── procedural.py         # ProceduralComponentGenerator
├── domains.py            # Application domain priors (20 domains)
│
├── # === Phase 2: Realistic Instrument Simulation ===
├── instruments.py        # InstrumentArchetype, InstrumentSimulator, multi-sensor
├── measurement_modes.py  # MeasurementMode, measurement mode simulators
├── detectors.py          # DetectorSimulator, detector response curves
│
├── # === Phase 3: Environmental and Matrix Effects ===
├── environmental.py      # Temperature and moisture/water activity effects
└── scattering.py         # Particle size effects, EMSC-style scattering
```

## Physical Model

The generator implements a **Beer-Lambert law** based model:

```
A_i(λ) = L_i * Σ_k c_ik * ε_k(λ) + baseline_i(λ) + scatter_i(λ) + noise_i(λ)
```

Where:
- `c_ik`: Concentration of component k in sample i
- `ε_k(λ)`: Molar absorptivity of component k (Voigt profiles)
- `L_i`: Optical path length factor
- `baseline`: Polynomial baseline drift
- `scatter`: Multiplicative/additive scattering effects
- `noise`: Wavelength-dependent Gaussian noise

### Peak Shapes

Absorption bands are modeled using **Voigt profiles** (convolution of Gaussian and Lorentzian):

```python
from nirs4all.data.synthetic import NIRBand

band = NIRBand(
    center=1450,    # Peak center in nm
    sigma=25,       # Gaussian width (FWHM)
    gamma=3,        # Lorentzian width
    amplitude=0.8,  # Peak height
    name="O-H 1st overtone"
)
```

## Core Components

### SyntheticNIRSGenerator

The main generation engine:

```python
from nirs4all.data.synthetic import SyntheticNIRSGenerator

generator = SyntheticNIRSGenerator(
    wavelength_start=1000,
    wavelength_end=2500,
    wavelength_step=2,
    component_library=library,    # Optional custom components
    complexity="realistic",
    random_state=42,
)

# Generate raw data
X, C, E = generator.generate(n_samples=1000)
# X: spectra (n_samples, n_wavelengths)
# C: concentrations (n_samples, n_components)
# E: pure component spectra (n_components, n_wavelengths)

# Generate with metadata
X, C, E, metadata = generator.generate(
    n_samples=1000,
    return_metadata=True,
    include_batch_effects=True,
    n_batches=3
)
```

### ComponentLibrary

Manages collections of spectral components:

```python
from nirs4all.data.synthetic import (
    ComponentLibrary,
    SpectralComponent,
    NIRBand
)

# Create from predefined components
library = ComponentLibrary.from_predefined(
    ["water", "protein", "lipid"],
    random_state=42
)

# Create custom library
library = ComponentLibrary(random_state=42)

# Add custom component
my_component = SpectralComponent(
    name="my_compound",
    bands=[
        NIRBand(center=1500, sigma=20, gamma=2, amplitude=0.6),
        NIRBand(center=2100, sigma=30, gamma=3, amplitude=0.8),
    ],
    correlation_group=1  # Components with same group are correlated
)
library.add_component(my_component)

# Add random component
library.add_random_component(
    name="unknown",
    n_bands=4,
    wavelength_range=(1000, 2500)
)

# Compute spectra
wavelengths = np.linspace(1000, 2500, 751)
spectra_matrix = library.compute_all(wavelengths)
# Shape: (n_components, n_wavelengths)
```

### Concentration Methods

Different distributions for component concentrations:

```python
# Dirichlet: Sum-to-one constraint (natural compositions)
C = generator.generate_concentrations(
    n_samples=1000,
    method="dirichlet",
    alpha=np.array([1.0, 2.0, 1.5])  # Concentration weights
)

# Uniform: Independent uniform distributions
C = generator.generate_concentrations(
    n_samples=1000,
    method="uniform"
)

# Lognormal: Skewed distributions (biological data)
C = generator.generate_concentrations(
    n_samples=1000,
    method="lognormal"
)

# Correlated: With inter-component correlations
correlation_matrix = np.array([
    [1.0, 0.7, -0.3],
    [0.7, 1.0, 0.2],
    [-0.3, 0.2, 1.0]
])
C = generator.generate_concentrations(
    n_samples=1000,
    method="correlated",
    correlation_matrix=correlation_matrix
)
```

## Extending the Generator

### Creating Custom Components

```python
from nirs4all.data.synthetic import SpectralComponent, NIRBand

# Define a pharmaceutical compound
aspirin = SpectralComponent(
    name="aspirin",
    bands=[
        NIRBand(center=1520, sigma=15, gamma=1.5, amplitude=0.4, name="C-H aromatic"),
        NIRBand(center=1680, sigma=20, gamma=2.0, amplitude=0.5, name="C=O"),
        NIRBand(center=2020, sigma=25, gamma=2.5, amplitude=0.35, name="C-H combination"),
    ],
    correlation_group=10  # Unique group for independent variation
)

# Use in generator
library = ComponentLibrary.from_predefined(["starch", "water"])
library.add_component(aspirin)

generator = SyntheticNIRSGenerator(
    component_library=library,
    complexity="realistic",
    random_state=42
)
```

### Custom Noise Models

The generator supports configurable noise through complexity parameters:

```python
from nirs4all.data.synthetic._constants import COMPLEXITY_PARAMS

# Default complexity parameters
print(COMPLEXITY_PARAMS["realistic"])
# {
#     'noise_scale': 0.005,
#     'baseline_degree': 2,
#     'baseline_amplitude': 0.1,
#     'scatter_intensity': 0.02,
#     ...
# }

# For custom noise, subclass or modify generator parameters
class CustomNoiseGenerator(SyntheticNIRSGenerator):
    def _set_complexity_params(self):
        super()._set_complexity_params()
        # Override specific parameters
        self.params['noise_scale'] = 0.01
        self.params['noise_wavelength_dependence'] = True
```

### Custom Target Transformations

```python
from nirs4all.data.synthetic import TargetGenerator

class CustomTargetGenerator(TargetGenerator):
    def custom_transform(self, concentrations, **kwargs):
        """Apply custom transformation to concentrations."""
        # Example: ratio of two components
        ratio = concentrations[:, 0] / (concentrations[:, 1] + 1e-6)
        return np.clip(ratio, 0, 10)
```

## Wavenumber Utilities (Phase 1)

The `wavenumber` module provides utilities for physically-correct band placement based on wavenumber (cm⁻¹) instead of wavelength (nm). This is critical because harmonic relationships (overtones, combinations) are **linear in wavenumber space**, not wavelength space.

### Mathematical Background

```
λ (nm) = 10⁷ / ν̃ (cm⁻¹)
ν̃ (cm⁻¹) = 10⁷ / λ (nm)
```

For overtones with anharmonicity correction:

```
ν̃ₙ = n × ν̃₀ × (1 - n × χ)
```

Where:
- `n` is the quantum number (1 = fundamental, 2 = 1st overtone, etc.)
- `ν̃₀` is the fundamental frequency
- `χ` is the anharmonicity constant (typically 0.01-0.03)

### NIR Spectral Zones

The module defines 7 NIR spectral zones in wavenumber space:

| Zone | Wavenumber (cm⁻¹) | Wavelength (nm) | Description |
|------|-------------------|-----------------|-------------|
| 3rd overtones | 9000-12500 | 800-1111 | Electronic transitions |
| 2nd overtones | 7500-9000 | 1111-1333 | C-H, N-H 2nd overtones |
| 1st O-H/N-H | 6000-7500 | 1333-1667 | O-H, N-H 1st overtones |
| 1st C-H | 5500-6250 | 1600-1818 | C-H 1st overtones |
| O-H comb | 5000-5500 | 1818-2000 | O-H combination bands |
| N-H comb | 4500-5200 | 1923-2222 | N-H, C-H combinations |
| C-H comb | 4000-4545 | 2200-2500 | C-H, C-O combinations |

### Fundamental Vibrations

22 fundamental vibrations are defined (in cm⁻¹):

```python
from nirs4all.data.synthetic import FUNDAMENTAL_VIBRATIONS

# O-H vibrations
print(FUNDAMENTAL_VIBRATIONS["O-H_stretch_free"])    # 3650 cm⁻¹
print(FUNDAMENTAL_VIBRATIONS["O-H_stretch_hbond"])   # 3400 cm⁻¹

# C-H vibrations
print(FUNDAMENTAL_VIBRATIONS["C-H_stretch_CH3_asym"])  # 2960 cm⁻¹
print(FUNDAMENTAL_VIBRATIONS["C-H_stretch_aromatic"])  # 3060 cm⁻¹

# N-H vibrations
print(FUNDAMENTAL_VIBRATIONS["N-H_stretch_primary"])   # 3400 cm⁻¹
```

### Overtone Calculation

Calculate overtone positions with anharmonicity correction:

```python
from nirs4all.data.synthetic import calculate_overtone_position

# Calculate O-H 1st overtone (n=2 in spectroscopic convention)
result = calculate_overtone_position("O-H_stretch_free", 2)
print(f"O-H 1st overtone: {result.wavelength_nm:.1f} nm")  # ~1433 nm
print(f"Wavenumber: {result.wavenumber_cm:.1f} cm⁻¹")
print(f"Amplitude factor: {result.amplitude_factor:.3f}")

# C-H overtone series
for n in [2, 3, 4]:  # 1st, 2nd, 3rd overtones
    result = calculate_overtone_position("C-H_stretch_CH3_asym", n)
    print(f"C-H {n-1}st overtone: {result.wavelength_nm:.1f} nm")

# Use numeric frequency directly
result = calculate_overtone_position(3400, 2, anharmonicity=0.022)
```

### Combination Band Calculation

Calculate combination band positions:

```python
from nirs4all.data.synthetic import calculate_combination_band

# O-H stretch + bend combination
result = calculate_combination_band(["O-H_stretch_free", "O-H_bend"])
print(f"O-H combination: {result.wavelength_nm:.1f} nm")  # ~1890 nm

# Multiple modes
result = calculate_combination_band(["C-H_stretch_CH3_asym", "C-H_bend"])
print(f"C-H combination: {result.wavelength_nm:.1f} nm")
```

### Hydrogen Bonding Shift

Model hydrogen bonding effects on band positions:

```python
from nirs4all.data.synthetic import apply_hydrogen_bonding_shift

# Free O-H stretch
free_oh = 3650  # cm⁻¹

# Apply H-bonding (shifts to lower wavenumber = longer wavelength)
bonded_oh = apply_hydrogen_bonding_shift(free_oh, strength=0.5)
print(f"H-bonded O-H: {bonded_oh:.0f} cm⁻¹")  # ~3400 cm⁻¹
```

### Zone Classification

Classify wavelengths into NIR zones:

```python
from nirs4all.data.synthetic import classify_wavelength_zone

zone = classify_wavelength_zone(1450)  # O-H 1st overtone region
print(f"1450 nm is in: {zone}")

zone = classify_wavelength_zone(2100)  # Combination region
print(f"2100 nm is in: {zone}")
```

## Procedural Component Generator (Phase 1)

The `procedural` module enables programmatic generation of chemically-plausible spectral components based on functional group properties.

### Functional Group Types

10 functional group types are defined:

```python
from nirs4all.data.synthetic import FunctionalGroupType

print(list(FunctionalGroupType))
# [HYDROXYL, AMINE, METHYL, METHYLENE, AROMATIC_CH,
#  VINYL, CARBONYL, CARBOXYL, THIOL, WATER]
```

Each functional group has physical properties defined:

```python
from nirs4all.data.synthetic import FUNCTIONAL_GROUP_PROPERTIES

props = FUNCTIONAL_GROUP_PROPERTIES[FunctionalGroupType.HYDROXYL]
print(f"Fundamental: {props['fundamental_cm']} cm⁻¹")
print(f"Bandwidth: {props['bandwidth_cm']} cm⁻¹")
print(f"H-bond susceptibility: {props['h_bond_susceptibility']}")
print(f"Typical amplitude: {props['typical_amplitude']}")
```

### Procedural Configuration

```python
from nirs4all.data.synthetic import ProceduralComponentConfig

config = ProceduralComponentConfig(
    # Band generation
    n_fundamental_bands=3,           # Number of fundamental vibrations
    include_overtones=True,          # Generate overtone bands
    max_overtone_order=3,            # Up to 2nd overtone (n=3)
    include_combinations=True,       # Generate combination bands
    max_combinations=3,              # Max combination bands

    # Environmental effects
    h_bond_strength=0.5,             # Average H-bonding (0-1)
    h_bond_variability=0.2,          # Random variation

    # Anharmonicity
    anharmonicity=0.02,              # Default anharmonicity constant
    anharmonicity_variability=0.005, # Random variation

    # Band property variation
    amplitude_variability=0.3,
    bandwidth_variability=0.2,

    # Wavelength range
    wavelength_range=(900, 2500),    # NIR region

    # Specific functional groups (optional)
    functional_groups=[
        FunctionalGroupType.HYDROXYL,
        FunctionalGroupType.METHYL,
    ],
)
```

### Generating Components

```python
from nirs4all.data.synthetic import ProceduralComponentGenerator

generator = ProceduralComponentGenerator(random_state=42)

# Generate a single component
component = generator.generate_component(
    name="ethanol_like",
    config=config,
)

print(f"Generated: {component.name}")
print(f"Bands: {len(component.bands)}")
for band in component.bands:
    print(f"  - {band.name}: {band.center:.1f} nm")
```

### Specifying Functional Groups

```python
# Generate from specific functional groups
component = generator.generate_component(
    name="amine_compound",
    functional_groups=[
        FunctionalGroupType.AMINE,
        FunctionalGroupType.METHYLENE,
    ],
)

# Generate sugar-like compound
sugar = generator.generate_component(
    name="carbohydrate",
    functional_groups=[
        FunctionalGroupType.HYDROXYL,  # Multiple O-H groups
        FunctionalGroupType.HYDROXYL,
        FunctionalGroupType.METHYLENE,
    ],
    config=ProceduralComponentConfig(
        h_bond_strength=0.7,  # Strong H-bonding
    ),
)
```

### Generating Variants

Create perturbed versions of existing components:

```python
# Generate base component
base = generator.generate_component("base_compound")

# Create variants with small perturbations
for i in range(5):
    variant = generator.generate_variant(
        base,
        perturbation_scale=0.1,  # 10% variation
    )
    print(f"Variant {i+1}: {len(variant.bands)} bands")
```

### Integration with ComponentLibrary

```python
from nirs4all.data.synthetic import ComponentLibrary

# Create library with procedural + predefined components
library = ComponentLibrary(random_state=42)

# Add predefined components
predefined = ComponentLibrary.from_predefined(["water", "starch"])
for comp in predefined.components.values():
    library.add_component(comp)

# Add procedurally generated components
proc_gen = ProceduralComponentGenerator(random_state=42)
for i in range(5):
    component = proc_gen.generate_component(
        name=f"novel_compound_{i}",
        config=ProceduralComponentConfig(n_fundamental_bands=3),
    )
    library.add_component(component)

print(f"Library has {len(library.component_names)} components")
```

## Application Domains (Phase 1)

The `domains` module provides domain-aware configuration for synthetic data generation, with 20 predefined application domains across 8 categories.

### Domain Categories

```python
from nirs4all.data.synthetic import DomainCategory

print(list(DomainCategory))
# [AGRICULTURE, FOOD, PHARMACEUTICAL, PETROCHEMICAL,
#  ENVIRONMENTAL, BIOMEDICAL, INDUSTRIAL, RESEARCH]
```

### Available Domains

| Category | Domains |
|----------|---------|
| Agriculture | `agriculture_grain`, `agriculture_forage`, `agriculture_oilseeds`, `agriculture_fruit` |
| Food | `food_dairy`, `food_meat`, `food_beverages`, `food_baking` |
| Pharmaceutical | `pharma_tablets`, `pharma_powders`, `pharma_liquids` |
| Petrochemical | `petrochem_fuel`, `petrochem_polymers`, `petrochem_lubricants` |
| Environmental | `environ_water`, `environ_soil` |
| Biomedical | `biomed_tissue`, `biomed_blood` |
| Industrial | `industrial_textiles`, `industrial_coatings` |

### Getting Domain Configuration

```python
from nirs4all.data.synthetic import get_domain_config, list_domains

# List all domains
all_domains = list_domains()
print(f"Available domains: {len(all_domains)}")

# List domains by category
ag_domains = list_domains(category=DomainCategory.AGRICULTURE)
print(f"Agriculture domains: {ag_domains}")

# Get specific domain configuration
grain = get_domain_config("agriculture_grain")
print(f"Domain: {grain.name}")
print(f"Category: {grain.category}")
print(f"Components: {grain.typical_components}")
print(f"Wavelength range: {grain.wavelength_range}")
print(f"Measurement mode: {grain.measurement_mode}")
```

### Domain Configuration Details

Each domain includes:

```python
from nirs4all.data.synthetic import DomainConfig

# DomainConfig fields:
# - name: Display name
# - category: DomainCategory enum
# - description: Detailed description
# - typical_components: List of component names
# - component_weights: Relative importance weights
# - concentration_priors: Statistical priors for concentrations
# - wavelength_range: (start, end) in nm
# - n_components_range: (min, max) components per sample
# - noise_level: "low", "medium", "high"
# - measurement_mode: "reflectance", "transmission", etc.
# - typical_sample_types: Example sample types
# - complexity: "simple", "realistic", "complex"
```

### Concentration Priors

Domains include realistic concentration priors:

```python
from nirs4all.data.synthetic import ConcentrationPrior
import numpy as np

# Get grain domain priors
grain = get_domain_config("agriculture_grain")

# Starch typically 50-80% in grains
starch_prior = grain.concentration_priors.get("starch")
if starch_prior:
    print(f"Starch prior: {starch_prior.distribution}")
    print(f"  Mean: ~65%, Range: 30-80%")

# Sample from priors
rng = np.random.default_rng(42)
concentrations = grain.sample_concentrations(n_samples=100, rng=rng)
print(f"Sampled starch mean: {concentrations.get('starch', []).mean():.2f}")
```

### Domain-Aware Library Creation

The `create_domain_aware_library` function samples components and concentrations based on domain priors:

```python
from nirs4all.data.synthetic import create_domain_aware_library

# Returns (component_names, concentration_matrix)
components, concentrations = create_domain_aware_library(
    "agriculture_grain",
    n_samples=100,
    random_state=42
)
print(f"Components: {components}")
print(f"Concentrations shape: {concentrations.shape}")
```

### Using Domains with Generator

To use domain configurations with `SyntheticNIRSGenerator`, create a `ComponentLibrary` from the domain's typical components:

```python
from nirs4all.data.synthetic import (
    SyntheticNIRSGenerator,
    ComponentLibrary,
    get_domain_config,
)

# Get domain config
domain = get_domain_config("agriculture_grain")

# Create ComponentLibrary from domain's typical components
library = ComponentLibrary.from_predefined(domain.typical_components[:5])

# Configure generator with domain parameters
generator = SyntheticNIRSGenerator(
    component_library=library,
    wavelength_start=domain.wavelength_range[0],
    wavelength_end=domain.wavelength_range[1],
    complexity="realistic",
    random_state=42,
)

# Generate domain-appropriate data
X, C, *rest = generator.generate(n_samples=500)
print(f"Generated {X.shape[0]} spectra for grain analysis")
```

### Getting Domain Components

```python
from nirs4all.data.synthetic import get_domain_components, get_domains_for_component

# Get typical components for a domain
components = get_domain_components("food_dairy")
print(f"Dairy components: {components}")

# Find which domains use a specific component
domains = get_domains_for_component("protein")
print(f"Domains using protein: {domains}")
```

## Extended Component Library (Phase 1)

The predefined component library has been extended from 31 to **111 components** covering diverse application areas.

### Component Categories

| Category | Count | Examples |
|----------|-------|----------|
| Carbohydrates | 15 | starch, cellulose, glucose, maltose, raffinose, inulin |
| Proteins | 10 | protein, casein, gluten, albumin, collagen, keratin |
| Lipids | 12 | lipid, oil, oleic_acid, linoleic_acid, phospholipid |
| Alcohols | 8 | ethanol, methanol, glycerol, sorbitol, mannitol |
| Organic Acids | 10 | acetic_acid, citric_acid, ascorbic_acid, formic_acid |
| Pharmaceuticals | 10 | aspirin, paracetamol, ibuprofen, metformin |
| Polymers | 8 | polyethylene, polystyrene, pmma, pet, abs |
| Minerals | 5 | kaolinite, montmorillonite, silica, talc |
| Pigments | 7 | chlorophyll, carotenoid, anthocyanin, lycopene |
| Solvents | 7 | acetone, dmso, ethyl_acetate, toluene |
| Other | 19 | water, aromatic, alkane, etc. |

### Listing Components

```python
from nirs4all.data.synthetic._constants import get_predefined_components

components = get_predefined_components()
print(f"Total components: {len(components)}")

# List all component names
for name in sorted(components.keys()):
    print(f"  - {name}")
```

### Using Extended Components

```python
from nirs4all.data.synthetic import ComponentLibrary

# Pharmaceutical formulation
pharma = ComponentLibrary.from_predefined([
    "ibuprofen", "microcrystalline_cellulose", "starch", "water"
])

# Food analysis
food = ComponentLibrary.from_predefined([
    "protein", "lipid", "starch", "glucose", "fructose", "moisture"
])

# Polymer analysis
polymers = ComponentLibrary.from_predefined([
    "polyethylene", "polypropylene", "pet", "abs", "pmma"
])

# Soil analysis
soil = ComponentLibrary.from_predefined([
    "kaolinite", "montmorillonite", "silica", "cellulose", "lignin"
])
```

## Instrument Simulation (Phase 2)

Phase 2 introduces realistic instrument simulation with 19 predefined instrument archetypes, multi-sensor stitching, multi-scan averaging, and detector response modeling.

### Instrument Archetypes

The module provides pre-configured models of real NIR instruments:

```python
from nirs4all.data.synthetic import (
    get_instrument_archetype,
    list_instrument_archetypes,
    get_instruments_by_category,
    InstrumentCategory,
)

# List all available instruments (19 total)
instruments = list_instrument_archetypes()
print(f"Available: {instruments}")

# Get by category
by_category = get_instruments_by_category()
for cat, names in by_category.items():
    print(f"{cat}: {names}")

# Get specific instrument
foss_xds = get_instrument_archetype("foss_xds")
print(f"FOSS XDS range: {foss_xds.wavelength_range}")
print(f"Resolution: {foss_xds.spectral_resolution} nm")
print(f"SNR: {foss_xds.snr}")
```

#### Available Instruments

| Category | Instruments |
|----------|-------------|
| Benchtop | `foss_xds`, `unity_spectrastar`, `metrohm_ds2500` |
| FT-NIR | `bruker_mpa`, `perkin_spectrum_two`, `thermo_antaris`, `abb_mb3600` |
| Handheld | `viavi_micronir`, `scio`, `tellspec`, `linksquare`, `siware_neoscanner` |
| Process | `nir_o_process`, `asd_fieldspec`, `buchi_nirmaster` |
| Embedded | `neospectra_micro`, `innospectra` |
| Filter | `foss_infratec` |
| Diode Array | `perten_da7200` |

### InstrumentArchetype Properties

Each archetype includes complete optical and electronic specifications:

```python
from nirs4all.data.synthetic import (
    InstrumentArchetype,
    InstrumentCategory,
    DetectorType,
    MonochromatorType,
)

# Create custom instrument
custom = InstrumentArchetype(
    name="my_instrument",
    category=InstrumentCategory.BENCHTOP,
    detector_type=DetectorType.INGAAS,
    monochromator_type=MonochromatorType.GRATING,
    wavelength_range=(900, 1700),
    spectral_resolution=4.0,           # nm FWHM
    wavelength_accuracy=0.1,           # nm
    photometric_noise=0.0001,          # AU
    photometric_range=(0.0, 3.0),      # AU min/max
    snr=20000,                         # Signal-to-noise ratio
    stray_light=0.0001,                # Fraction
    warm_up_drift=0.1,                 # %/hour
    temperature_sensitivity=0.01,      # nm/°C
    scan_speed=5.0,                    # Scans/second
    integration_time_ms=50.0,          # ms
    description="Custom research instrument",
)
```

### Multi-Sensor Configuration

Many NIR instruments use multiple detectors to cover wide wavelength ranges, then stitch the signals together:

```python
from nirs4all.data.synthetic import (
    MultiSensorConfig,
    SensorConfig,
    DetectorType,
)

# Configure multi-sensor system (e.g., FOSS XDS with Si + PbS)
multi_sensor = MultiSensorConfig(
    enabled=True,
    sensors=[
        SensorConfig(
            detector_type=DetectorType.SI,
            wavelength_range=(400, 1100),
            spectral_resolution=0.5,
            noise_level=0.8,
            gain=1.0,
            overlap_range=20.0,  # nm overlap for stitching
        ),
        SensorConfig(
            detector_type=DetectorType.PBS,
            wavelength_range=(1100, 2500),
            spectral_resolution=0.5,
            noise_level=1.2,
        ),
    ],
    stitch_method="weighted",      # weighted, average, first, last, optimal
    stitch_smoothing=10.0,         # nm smoothing at boundaries
    add_stitch_artifacts=True,     # Simulate stitching artifacts
    artifact_intensity=0.01,       # Artifact strength (0-1)
)
```

#### Stitch Methods

| Method | Description |
|--------|-------------|
| `weighted` | Linear blend based on distance from boundary |
| `average` | Simple average in overlap region |
| `first` | Use first sensor's data in overlap |
| `last` | Use second sensor's data in overlap |
| `optimal` | SNR-weighted combination |

### Multi-Scan Averaging

Real instruments take multiple scans and average them to reduce noise:

```python
from nirs4all.data.synthetic import MultiScanConfig

multi_scan = MultiScanConfig(
    enabled=True,
    n_scans=32,                     # Number of scans to average
    averaging_method="mean",        # mean, median, weighted, savgol
    scan_to_scan_noise=0.001,       # Additional noise between scans
    wavelength_jitter=0.05,         # nm shift between scans
    discard_outliers=True,          # Remove outlier scans
    outlier_threshold=2.5,          # Z-score threshold
)
```

#### Averaging Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `mean` | Arithmetic mean | Standard averaging |
| `median` | Median filter | Robust to outliers |
| `weighted` | SNR-weighted average | Variable quality scans |
| `savgol` | Savitzky-Golay smoothing | Spectral smoothing |

### Using Instruments with the Generator

```python
from nirs4all.data.synthetic import SyntheticNIRSGenerator

# Generate with a predefined instrument
gen = SyntheticNIRSGenerator(
    wavelength_start=1000,
    wavelength_end=2500,
    instrument="foss_xds",  # Apply FOSS XDS effects
    random_state=42,
)

X, C, pure, metadata = gen.generate(n_samples=100, return_metadata=True)
print(f"Multi-sensor applied: {metadata.get('multi_sensor')}")
print(f"Multi-scan applied: {metadata.get('multi_scan')}")

# Generate with custom multi-sensor/multi-scan config
gen = SyntheticNIRSGenerator(
    wavelength_start=400,
    wavelength_end=2500,
    multi_sensor_config=multi_sensor,
    multi_scan_config=multi_scan,
    random_state=42,
)
```

### InstrumentSimulator

For fine-grained control, use the simulator directly:

```python
from nirs4all.data.synthetic import InstrumentSimulator, get_instrument_archetype
import numpy as np

# Get archetype and create simulator
archetype = get_instrument_archetype("bruker_mpa")
simulator = InstrumentSimulator(archetype, random_state=42)

# Apply to existing spectra
spectra = np.random.default_rng(42).normal(0.5, 0.1, (100, 500))
wavelengths = np.linspace(1000, 2500, 500)

result, output_wl = simulator.apply(
    spectra,
    wavelengths,
    temperature_offset=2.0,  # °C deviation from calibration
)
print(f"Output shape: {result.shape}")
print(f"Output range: {output_wl.min():.0f}-{output_wl.max():.0f} nm")
```

### Detector Response Curves

The module includes realistic detector spectral response curves:

```python
from nirs4all.data.synthetic import (
    get_detector_response,
    list_detector_types,
    DetectorType,
    DetectorSimulator,
)
import numpy as np

# Available detectors
print(f"Detector types: {list_detector_types()}")
# ['si', 'ingaas', 'ingaas_ext', 'pbs', 'pbse', 'mems', 'mct']

# Get response curve
wavelengths = np.linspace(800, 2500, 1000)
ingaas = get_detector_response(DetectorType.INGAAS)

print(f"InGaAs sensitivity: {ingaas.short_cutoff}-{ingaas.cutoff_wavelength} nm")
print(f"Peak wavelength: {ingaas.peak_wavelength} nm")
print(f"Peak QE: {ingaas.peak_qe}")

# Get responsivity at specific wavelengths
responsivity = ingaas.get_response_at(wavelengths)
```

#### Detector Specifications

| Type | Range (nm) | Peak (nm) | Peak QE | Use Case |
|------|------------|-----------|---------|----------|
| Si | 350-1100 | 850 | 0.85 | Visible/short-NIR |
| InGaAs | 850-1700 | 1300 | 0.92 | Standard NIR |
| InGaAs Ext | 850-2500 | 1700 | 0.85 | Extended NIR |
| PbS | 1000-3000 | 2200 | 0.85 | Long-NIR |
| PbSe | 1500-5000 | 3500 | 0.75 | MIR |
| MEMS | 900-1700 | 1300 | 0.60 | Miniature/embedded |
| MCT | 1000-14000 | 5000 | 0.70 | Cooled high-perf |

### Detector Noise Models

Each detector type has characteristic noise properties:

```python
from nirs4all.data.synthetic import (
    DetectorSimulator,
    DetectorConfig,
    NoiseModelConfig,
    DetectorType,
)

# Configure detector noise
noise_config = NoiseModelConfig(
    shot_noise_factor=1.0,       # Signal-dependent (√signal)
    thermal_noise_factor=0.5,    # Temperature-dependent
    read_noise_factor=0.3,       # Constant per-pixel
    flicker_noise_factor=0.1,    # 1/f noise
    dark_current=0.001,          # Counts/ms
    integration_time_ms=50.0,
)

detector_config = DetectorConfig(
    detector_type=DetectorType.INGAAS,
    gain=1.0,
    nonlinearity=0.01,           # Nonlinearity coefficient
    noise=noise_config,
)

# Apply detector effects
simulator = DetectorSimulator(detector_config, random_state=42)
processed = simulator.apply(spectra, wavelengths, add_noise=True)
```

### Measurement Modes

Different sampling geometries produce different spectral characteristics:

```python
from nirs4all.data.synthetic import (
    MeasurementModeSimulator,
    MeasurementModeConfig,
    MeasurementMode,
    TransmittanceConfig,
    ReflectanceConfig,
    ATRConfig,
)

# Transmittance (liquid samples)
trans_config = MeasurementModeConfig(
    mode=MeasurementMode.TRANSMITTANCE,
    transmittance=TransmittanceConfig(
        path_length_mm=10.0,
        path_length_variation=0.02,
        cuvette_material="quartz",
    ),
)
sim = MeasurementModeSimulator(config=trans_config, random_state=42)
processed = sim.apply(spectra, wavelengths)

# Diffuse reflectance (powder samples)
refl_config = MeasurementModeConfig(
    mode=MeasurementMode.REFLECTANCE,
    reflectance=ReflectanceConfig(
        geometry="integrating_sphere",
        sample_presentation="powder",
        illumination_angle=0.0,
        collection_angle=45.0,
    ),
)

# ATR (solids, pastes)
atr_config = MeasurementModeConfig(
    mode=MeasurementMode.ATR,
    atr=ATRConfig(
        crystal_material="diamond",
        crystal_refractive_index=2.4,
        incidence_angle=45.0,
        n_reflections=1,
    ),
)
```

#### Available Measurement Modes

| Mode | Description | Typical Samples |
|------|-------------|-----------------|
| `transmittance` | Light passes through sample | Liquids, thin films |
| `reflectance` | Diffuse or specular reflection | Powders, solids |
| `transflectance` | Transmission + reflection | Liquids with mirror |
| `atr` | Attenuated total reflectance | Pastes, thick liquids |
| `interactance` | Fiber probe immersed | Suspensions |
| `fiber_optic` | Remote fiber measurement | Process monitoring |

## Environmental Effects (Phase 3)

Phase 3 introduces simulation of environmental and matrix effects that affect NIR spectra in real-world measurements. These effects are critical for generating realistic synthetic data that captures the variability seen in practical applications.

### Temperature Effects

Temperature affects NIR spectra through peak shifts, intensity changes, and band broadening. The `environmental.py` module provides comprehensive temperature simulation:

```python
from nirs4all.data.synthetic import (
    TemperatureEffectSimulator,
    TemperatureConfig,
    TemperatureEffectParams,
    SpectralRegion,
)
import numpy as np

# Configure temperature effects
temp_config = TemperatureConfig(
    sample_temperature=35.0,           # Sample temperature (°C)
    reference_temperature=25.0,        # Reference/calibration temperature
    temperature_variation=2.0,         # Sample-to-sample variation (σ)
    enable_wavelength_shift=True,      # Enable peak position shifts
    enable_intensity_change=True,      # Enable intensity changes
    enable_broadening=True,            # Enable thermal broadening
)

# Create simulator
simulator = TemperatureEffectSimulator(temp_config, random_state=42)

# Apply to spectra
wavelengths = np.linspace(1000, 2500, 751)
spectra = np.random.randn(100, 751) * 0.1 + 0.5  # Example spectra

# Generate per-sample temperatures
sample_temps = simulator.sample_temperatures(n_samples=100)

# Apply effects
modified_spectra = simulator.apply(spectra, wavelengths, sample_temperatures=sample_temps)
```

#### Spectral Regions

The module defines 8 spectral regions with distinct temperature responses:

| Region | Wavelength (nm) | Shift (nm/°C) | Intensity (%/°C) | Reference |
|--------|-----------------|---------------|------------------|-----------|
| O-H 1st overtone | 1400-1520 | -0.30 | -0.20 | Maeda et al. (1995) |
| O-H combination | 1900-2000 | -0.40 | -0.30 | Segtnan et al. (2001) |
| C-H 1st overtone | 1650-1780 | -0.05 | -0.05 | Literature estimate |
| C-H combination | 2200-2400 | -0.08 | -0.08 | Literature estimate |
| N-H 1st overtone | 1500-1550 | -0.20 | -0.15 | Literature estimate |
| N-H combination | 2000-2100 | -0.25 | -0.20 | Literature estimate |
| Free water | N/A | -0.35 | -0.25 | Luck (1998) |
| Bound water | N/A | -0.20 | -0.35 | Büning-Pfaue (2003) |

#### Custom Temperature Parameters

```python
from nirs4all.data.synthetic import TemperatureEffectParams, SpectralRegion

# Define custom parameters for a specific region
custom_params = TemperatureEffectParams(
    wavelength_range=(1400, 1520),
    shift_per_degree=-0.30,           # nm/°C (negative = blue shift)
    intensity_change_per_degree=-0.002,  # fraction/°C
    broadening_per_degree=0.001,      # fraction/°C
    reference="Custom application"
)

# Use in configuration
temp_config = TemperatureConfig(
    sample_temperature=40.0,
    custom_params={SpectralRegion.OH_FIRST_OVERTONE: custom_params}
)
```

### Moisture/Water Activity Effects

Water content and activity affect hydrogen bonding, which in turn modifies water-related bands:

```python
from nirs4all.data.synthetic import (
    MoistureEffectSimulator,
    MoistureConfig,
)

# Configure moisture effects
moisture_config = MoistureConfig(
    water_activity=0.6,              # Water activity (0-1)
    moisture_content=0.15,           # Fractional moisture content
    free_water_fraction=0.3,         # Fraction of water that is free
    bound_water_shift=20.0,          # nm shift for bound water
    enable_band_shift=True,
    enable_intensity_modulation=True,
)

# Create simulator
moisture_sim = MoistureEffectSimulator(moisture_config, random_state=42)

# Apply effects
modified_spectra = moisture_sim.apply(spectra, wavelengths)
```

#### Water Activity Effects

| Water Activity | Effect on Spectra |
|----------------|-------------------|
| Low (< 0.3) | Bound water dominates, shifted O-H bands |
| Medium (0.3-0.7) | Mixed free/bound water, broadened bands |
| High (> 0.7) | Free water dominates, sharp O-H bands |

### Combined Environmental Effects

The `EnvironmentalEffectsSimulator` combines all environmental effects:

```python
from nirs4all.data.synthetic import (
    EnvironmentalEffectsSimulator,
    EnvironmentalEffectsConfig,
    TemperatureConfig,
    MoistureConfig,
)

# Combine temperature and moisture
env_config = EnvironmentalEffectsConfig(
    temperature=TemperatureConfig(
        sample_temperature=35.0,
        temperature_variation=3.0,
    ),
    moisture=MoistureConfig(
        water_activity=0.65,
        moisture_content=0.12,
    ),
)

# Create combined simulator
env_simulator = EnvironmentalEffectsSimulator(env_config, random_state=42)

# Apply all effects
modified_spectra = env_simulator.apply(
    spectra,
    wavelengths,
    sample_temperatures=temps,  # Optional: explicit temperatures
)
```

## Scattering Effects (Phase 3)

The `scattering.py` module simulates light scattering effects, which are critical for diffuse reflectance spectra. Rather than implementing full Mie theory, this module uses empirical EMSC-style models that approximate real-world scattering distortions.

### Particle Size Effects

Particle size strongly affects NIR spectra through scattering:

```python
from nirs4all.data.synthetic import (
    ParticleSizeSimulator,
    ParticleSizeConfig,
    ParticleSizeDistribution,
)

# Configure particle size distribution
psd = ParticleSizeDistribution(
    mean_size_um=50.0,       # Mean size in micrometers
    std_size_um=15.0,        # Standard deviation
    min_size_um=5.0,         # Minimum (truncation)
    max_size_um=200.0,       # Maximum (truncation)
    distribution="lognormal", # lognormal, normal, or uniform
)

# Configure particle size effects
ps_config = ParticleSizeConfig(
    distribution=psd,
    reference_size_um=50.0,  # Reference size for calibration
    enable_baseline_shift=True,
    enable_multiplicative_scatter=True,
    enable_wavelength_dependence=True,
)

# Create simulator
ps_simulator = ParticleSizeSimulator(ps_config, random_state=42)

# Sample particle sizes for each sample
sizes = psd.sample(n_samples=100, rng=np.random.default_rng(42))

# Apply effects
modified_spectra = ps_simulator.apply(spectra, wavelengths, particle_sizes=sizes)
```

#### Particle Size Regimes

| Size Regime | Particle Size | Scattering Behavior | λ Dependence |
|-------------|---------------|---------------------|--------------|
| Small | < 10 μm | Rayleigh-like | λ⁻⁴ |
| Medium | 10-100 μm | Mixed Mie | λ⁻¹ to λ⁻² |
| Large | > 100 μm | Geometrical | Weak |

### EMSC-Style Transformations

Extended Multiplicative Scatter Correction (EMSC) models are used to simulate realistic scattering distortions:

```python
from nirs4all.data.synthetic import (
    EMSCTransformSimulator,
    EMSCConfig,
)

# Configure EMSC-style effects
emsc_config = EMSCConfig(
    baseline_offset_range=(-0.1, 0.1),      # Additive offset
    multiplicative_range=(0.8, 1.2),         # Multiplicative scaling
    linear_slope_range=(-0.0001, 0.0001),   # Linear baseline
    quadratic_range=(-1e-7, 1e-7),          # Quadratic baseline
    inverse_lambda_range=(0.0, 0.01),       # 1/λ scattering term
    sample_variation=0.3,                    # Sample-to-sample variation
)

# Create simulator
emsc_simulator = EMSCTransformSimulator(emsc_config, random_state=42)

# Apply EMSC-style distortions
distorted_spectra = emsc_simulator.apply(spectra, wavelengths)
```

#### EMSC Model

The EMSC model transforms spectra as:

```
A_observed = a + b × A_pure + c × λ + d × λ² + e × (1/λ)
```

Where:
- `a`: Baseline offset (scattering-induced)
- `b`: Multiplicative scaling (path length variation)
- `c`: Linear wavelength dependence
- `d`: Quadratic term (curvature)
- `e`: Inverse wavelength term (Rayleigh-like scattering)

### Scattering Coefficient Generation

For Kubelka-Munk reflectance simulation, generate realistic scattering coefficients:

```python
from nirs4all.data.synthetic import (
    ScatteringCoefficientGenerator,
    ScatteringCoefficientConfig,
)

# Configure scattering coefficient generation
scat_config = ScatteringCoefficientConfig(
    baseline_scattering=1.0,         # S₀ reference value
    wavelength_reference_nm=1700.0,  # Reference wavelength
    wavelength_exponent=1.5,         # λ dependence exponent
    particle_size_um=50.0,           # Affects wavelength exponent
)

# Generate scattering coefficients
generator = ScatteringCoefficientGenerator(scat_config, random_state=42)
S_lambda = generator.generate(wavelengths)

# Use with Kubelka-Munk
# R∞ = 1 + K/S - √((K/S)² + 2K/S)
```

#### Wavelength Exponent vs Particle Size

| Particle Size | Wavelength Exponent b | Model |
|---------------|----------------------|-------|
| < 10 μm | 2-4 | Rayleigh (λ⁻⁴) |
| 10-50 μm | 1-2 | Mixed |
| 50-100 μm | 0.5-1 | Mie |
| > 100 μm | 0-0.5 | Geometrical |

### Combined Scattering Effects

The `ScatteringEffectsSimulator` combines all scattering-related effects:

```python
from nirs4all.data.synthetic import (
    ScatteringEffectsSimulator,
    ScatteringEffectsConfig,
    ParticleSizeConfig,
    EMSCConfig,
    ScatteringCoefficientConfig,
)

# Combine all scattering effects
scatter_config = ScatteringEffectsConfig(
    particle_size=ParticleSizeConfig(
        distribution=ParticleSizeDistribution(mean_size_um=50.0),
    ),
    emsc=EMSCConfig(
        multiplicative_range=(0.85, 1.15),
    ),
    scattering_coefficient=ScatteringCoefficientConfig(
        baseline_scattering=1.0,
    ),
)

# Create combined simulator
scatter_simulator = ScatteringEffectsSimulator(scatter_config, random_state=42)

# Apply all scattering effects
modified_spectra = scatter_simulator.apply(
    spectra,
    wavelengths,
    particle_sizes=sizes,  # Optional: explicit sizes
)
```

### Integration with Generator

Phase 3 effects are integrated into the main `SyntheticNIRSGenerator`:

```python
from nirs4all.data.synthetic import (
    SyntheticNIRSGenerator,
    EnvironmentalEffectsConfig,
    ScatteringEffectsConfig,
    TemperatureConfig,
    ParticleSizeConfig,
)

# Configure environmental and scattering effects
env_config = EnvironmentalEffectsConfig(
    temperature=TemperatureConfig(
        sample_temperature=30.0,
        temperature_variation=5.0,
    ),
)

scatter_config = ScatteringEffectsConfig(
    particle_size=ParticleSizeConfig(
        distribution=ParticleSizeDistribution(mean_size_um=75.0),
    ),
)

# Create generator with Phase 3 effects
generator = SyntheticNIRSGenerator(
    wavelength_start=1000,
    wavelength_end=2500,
    complexity="realistic",
    environmental_config=env_config,
    scattering_effects_config=scatter_config,
    random_state=42,
)

# Generate with effects
X, C, E = generator.generate(
    n_samples=500,
    include_environmental_effects=True,
    include_scattering_effects=True,
)
```

## Metadata System

The `MetadataGenerator` creates realistic sample metadata:

```python
from nirs4all.data.synthetic import MetadataGenerator

metadata_gen = MetadataGenerator(random_state=42)

result = metadata_gen.generate(
    n_samples=100,
    sample_id_prefix="wheat",
    n_groups=5,                    # For GroupKFold
    group_names=["Field_A", "Field_B", "Field_C", "Field_D", "Field_E"],
    n_repetitions=(2, 4),          # Variable repetitions per sample
)

# Access generated metadata
print(result.sample_ids)          # ['wheat_001', 'wheat_002', ...]
print(result.groups)              # ['Field_A', 'Field_A', 'Field_B', ...]
print(result.biological_samples)  # Sample indices for repetitions
print(result.repetition_counts)   # [3, 2, 4, ...]
```

## Classification Generation

The `TargetGenerator` creates separable classes:

```python
from nirs4all.data.synthetic import TargetGenerator, ClassSeparationConfig

target_gen = TargetGenerator(random_state=42)

# Generate classification targets
y = target_gen.classification(
    n_samples=1000,
    concentrations=C,              # From generator
    n_classes=3,
    class_weights=[0.4, 0.4, 0.2], # Class proportions
    separation=2.0,                # Higher = more separable
    separation_method="component"  # How to create class differences
)

# Separation methods:
# - "component": Different concentration profiles per class
# - "threshold": Classes based on concentration thresholds
# - "cluster": K-means-like assignment
```

## Non-Linear Target Complexity

The `NonLinearTargetProcessor` creates challenging, realistic targets that require
non-linear models to predict well. It implements three strategies:

### Architecture

```python
from nirs4all.data.synthetic.targets import (
    NonLinearTargetProcessor,
    NonLinearTargetConfig
)

config = NonLinearTargetConfig(
    # Proposition 1: Non-linear interactions
    nonlinear_interactions="polynomial",  # none, polynomial, synergistic, antagonistic
    interaction_strength=0.5,              # 0=linear, 1=fully non-linear
    hidden_factors=2,                      # Latent variables not in spectra
    polynomial_degree=2,                   # 2 or 3

    # Proposition 2: Confounders
    signal_to_confound_ratio=0.7,         # 1.0=fully predictable
    n_confounders=2,                       # Confounding variables
    spectral_masking=0.0,                  # Future: hide signal in noisy regions
    temporal_drift=True,                   # Relationship changes over samples

    # Proposition 3: Multi-regime
    n_regimes=3,                           # Different relationship regimes
    regime_method="concentration",         # concentration, spectral, random
    regime_overlap=0.2,                    # Transition zone smoothness
    noise_heteroscedasticity=0.5,          # Noise varies by regime
)

processor = NonLinearTargetProcessor(config, random_state=42)
y_complex = processor.process(
    concentrations=C,
    y_base=y_linear,
    spectra=X  # Optional, for spectral-based regime assignment
)
```

### Interaction Types

| Type | Formula | Use Case |
|------|---------|----------|
| `polynomial` | y = f(C₁², C₁×C₂, ...) | General non-linearity |
| `synergistic` | y = f(√(C₁×C₂) × (C₁+C₂)) | Chemical synergies |
| `antagonistic` | y = Vmax×C/(Km+C) × inhibition | Michaelis-Menten kinetics |

### Hidden Factors

Hidden factors are latent variables that affect the target but have **no spectral signature**.
This creates irreducible prediction error, testing whether models overfit:

```python
# With hidden_factors=3, about 30% of target variance is unexplainable
config = NonLinearTargetConfig(
    nonlinear_interactions="none",
    hidden_factors=3
)
```

### Multi-Regime Landscapes

Different regions of the feature space have different target-spectra relationships:

```python
# Regime 0: y ∝ C₁ (linear)
# Regime 1: y ∝ C₁² (quadratic)
# Regime 2: y ∝ max(y) - y (inverse)
# Regime 3+: y ∝ C₁/C₂ (ratio)

config = NonLinearTargetConfig(
    n_regimes=4,
    regime_method="concentration",  # Partition by concentration space
    regime_overlap=0.2,             # Smooth transitions
    noise_heteroscedasticity=0.5    # Different noise per regime
)
```

### Extending the Processor

```python
class CustomTargetProcessor(NonLinearTargetProcessor):
    def _apply_nonlinear_interactions(self, C, y):
        # Custom non-linear transformation
        custom_term = np.sin(C[:, 0] * np.pi) * np.exp(-C[:, 1])
        strength = self.config.interaction_strength
        return (1 - strength) * y + strength * custom_term.reshape(-1, 1)
```

## Multi-Source Generation

Generate datasets with multiple data types:

```python
from nirs4all.data.synthetic import MultiSourceGenerator, SourceConfig

generator = MultiSourceGenerator(random_state=42)

result = generator.generate(
    n_samples=500,
    sources=[
        SourceConfig(
            name="NIR",
            source_type="nir",
            wavelength_range=(1000, 2500),
            complexity="realistic",
            components=["water", "protein", "lipid"]
        ),
        SourceConfig(
            name="VIS",
            source_type="nir",
            wavelength_range=(400, 1000),
            complexity="simple"
        ),
        SourceConfig(
            name="markers",
            source_type="aux",
            n_features=10
        )
    ]
)

# Access individual sources
X_nir = result.sources["NIR"]
X_vis = result.sources["VIS"]
X_markers = result.sources["markers"]

# Get combined features
X_all = result.get_combined_features()
```

## Exporter System

Export synthetic data to various formats:

```python
from nirs4all.data.synthetic import DatasetExporter, CSVVariationGenerator

exporter = DatasetExporter()

# Standard format (Xcal, Ycal, Xval, Yval)
path = exporter.to_folder(
    "output/dataset",
    X, y,
    train_ratio=0.8,
    wavelengths=wavelengths,
    format="standard"
)

# Single file format
path = exporter.to_folder(
    "output/dataset",
    X, y,
    format="single"
)

# For testing CSV loaders with format variations
csv_gen = CSVVariationGenerator()
variations = csv_gen.generate_variations(
    X, y, wavelengths,
    output_dir="test_data",
    variations=[
        "comma_delimiter",
        "tab_delimiter",
        "no_headers",
        "european_decimals"
    ]
)
```

## Real Data Fitting

Analyze real data and generate similar synthetic data:

```python
from nirs4all.data.synthetic import RealDataFitter, compute_spectral_properties

# Compute properties of real data
props = compute_spectral_properties(
    X_real,
    wavelengths=wavelengths,
    name="real_wheat"
)

print(f"Global mean: {props.global_mean:.4f}")
print(f"Noise estimate: {props.noise_estimate:.6f}")
print(f"PCA components for 95% var: {props.pca_n_components_95}")

# Fit generator parameters
fitter = RealDataFitter()
params = fitter.fit(X_real, wavelengths=wavelengths)

print(f"Fitted wavelength range: {params.wavelength_start}-{params.wavelength_end}")
print(f"Recommended complexity: {params.complexity}")

# Generate similar data
X_synth = fitter.generate(n_samples=1000, random_state=42)
```

## Validation System

Validate generated data quality:

```python
from nirs4all.data.synthetic import (
    validate_spectra,
    validate_wavelengths,
    validate_concentrations,
    ValidationError
)

try:
    validate_spectra(X, wavelengths)
    validate_concentrations(C)
    validate_wavelengths(wavelengths)
except ValidationError as e:
    print(f"Validation failed: {e}")
```

## Integration with Test Fixtures

The synthetic module provides pytest fixtures for testing:

```python
# In tests/conftest.py (already configured)

def test_my_feature(standard_regression_dataset):
    """Use session-scoped synthetic dataset."""
    X = standard_regression_dataset.x({"partition": "train"})
    y = standard_regression_dataset.y({"partition": "train"})
    # ... test logic

def test_with_custom_data(synthetic_builder_factory):
    """Use factory to create custom dataset."""
    dataset = (
        synthetic_builder_factory(n_samples=100)
        .with_features(complexity="simple")
        .with_classification(n_classes=3)
        .build()
    )
    # ... test logic

def test_loader(synthetic_dataset_folder):
    """Test with temporary CSV files."""
    from nirs4all.data import DatasetConfigs
    dataset = DatasetConfigs(str(synthetic_dataset_folder))
    # ... test logic
```

## Performance Considerations

| Operation | Samples | Time |
|-----------|---------|------|
| Generation (simple) | 10,000 | ~0.3s |
| Generation (realistic) | 10,000 | ~0.5s |
| Generation (complex) | 10,000 | ~1.0s |
| Metadata generation | 10,000 | ~0.05s |
| Export to CSV | 10,000 | ~0.5s |
| Real data fitting | N/A | ~0.2s |

Tips:
- Use `complexity="simple"` for unit tests
- Reuse generators with same random_state
- Use session-scoped fixtures for expensive datasets
- Generate once, clone for modifications

## Predefined Spectral Components

The synthetic generator includes **48 predefined spectral components** with band assignments based on published NIR spectroscopy literature. Each component's absorption bands are modeled using Voigt profiles with accurate wavelength positions, widths, and relative intensities.

### Available Components

#### Water and Moisture

| Component | Description | Key Bands (nm) | Reference |
|-----------|-------------|----------------|-----------|
| `water` | Free water (H₂O) | 1450, 1940, 2500 | [1] pp. 34-36 |
| `moisture` | Bound water in matrices | 1460, 1930 | [2] pp. 358-362 |

#### Proteins and Nitrogen Compounds

| Component | Description | Key Bands (nm) | Reference |
|-----------|-------------|----------------|-----------|
| `protein` | General protein (amide, N-H) | 1510, 1680, 2050, 2180, 2300 | [1] pp. 48-52 |
| `nitrogen_compound` | Primary/secondary amines | 1500, 2060, 2150 | [1] pp. 52-54 |
| `urea` | CO(NH₂)₂ | 1480, 1530, 2010, 2170 | [9] p. 1125 |
| `amino_acid` | Free amino acids | 1520, 2040, 2260 | [3] pp. 215-220 |
| `casein` | Milk protein | 1510, 1680, 2050, 2180 | [4] pp. 85-88 |
| `gluten` | Wheat protein complex | 1505, 1680, 2050, 2180, 2290 | [5] pp. 155-160 |

#### Lipids and Hydrocarbons

| Component | Description | Key Bands (nm) | Reference |
|-----------|-------------|----------------|-----------|
| `lipid` | Triglycerides (C-H) | 1210, 1390, 1720, 2310, 2350 | [1] pp. 44-48 |
| `oil` | Vegetable/mineral oils | 1165, 1215, 1410, 1725, 2140, 2305 | [4] pp. 67-72 |
| `saturated_fat` | Saturated fatty acids | 1195, 1395, 1730, 2315, 2355 | [7] pp. 15-20 |
| `unsaturated_fat` | Mono/polyunsaturated fats | 1160, 1400, 1720, 2145, 2175 | [7] pp. 20-25 |
| `waxes` | Cuticular waxes | 1190, 1720, 2310, 2350 | [7] pp. 15-20 |
| `aromatic` | Benzene derivatives | 1145, 1685, 2150, 2440 | [1] pp. 56-58 |
| `alkane` | Saturated hydrocarbons | 1190, 1715, 2310, 2360 | [7] pp. 10-15 |

#### Carbohydrates

| Component | Description | Key Bands (nm) | Reference |
|-----------|-------------|----------------|-----------|
| `starch` | Amylose/amylopectin | 1460, 1580, 2100, 2270 | [5] pp. 155-160 |
| `cellulose` | β-1,4-glucan chains | 1490, 1780, 2090, 2280, 2340 | [6] pp. 295-300 |
| `glucose` | D-glucose | 1440, 1690, 2080, 2270 | [2] pp. 368-370 |
| `fructose` | D-fructose | 1430, 1695, 2070, 2260 | [2] pp. 368-370 |
| `sucrose` | Disaccharide | 1435, 1685, 2075, 2265 | [2] pp. 370-372 |
| `lactose` | Milk sugar | 1450, 1690, 1940, 2100, 2270 | [12] |
| `hemicellulose` | Xylan/glucomannan | 1470, 1760, 2085, 2250 | [6] pp. 300-303 |
| `lignin` | Aromatic polymer | 1140, 1420, 1670, 2130, 2270 | [6] pp. 303-305 |
| `dietary_fiber` | Plant cell wall material | 1490, 1770, 2090, 2275, 2340 | [6], [5] |

#### Alcohols and Polyols

| Component | Description | Key Bands (nm) | Reference |
|-----------|-------------|----------------|-----------|
| `ethanol` | C₂H₅OH | 1410, 1580, 1695, 2050, 2290 | [1] pp. 38-40 |
| `methanol` | CH₃OH | 1400, 1545, 1705, 2040 | [1] pp. 38-40 |
| `glycerol` | Polyol (fermentation) | 1450, 1580, 1700, 2060, 2280 | [11] |

#### Organic Acids

| Component | Description | Key Bands (nm) | Reference |
|-----------|-------------|----------------|-----------|
| `acetic_acid` | CH₃COOH | 1420, 1700, 1940, 2240 | [8] pp. 8-10 |
| `citric_acid` | C₆H₈O₇ | 1440, 1920, 2060, 2260 | [4] pp. 78-80 |
| `lactic_acid` | CH₃CH(OH)COOH | 1430, 1485, 1700, 2020, 2255 | [9] pp. 1128-1130 |
| `malic_acid` | Fruit acid (apples) | 1440, 1920, 2050, 2255 | [4] pp. 78-80 |
| `tartaric_acid` | Grape/wine acid | 1435, 1910, 2040, 2260 | [11] |

#### Plant Pigments and Phenolics

| Component | Description | Key Bands (nm) | Reference |
|-----------|-------------|----------------|-----------|
| `chlorophyll` | Chlorophyll a/b | 1070, 1400, 1730, 2270 | [2] pp. 375-378 |
| `carotenoid` | β-carotene, xanthophylls | 1050, 1680, 2135, 2280 | [2] pp. 378-380 |
| `tannins` | Phenolic compounds | 1420, 1670, 2056, 2270 | [6], [11] |

#### Pharmaceutical Compounds

| Component | Description | Key Bands (nm) | Reference |
|-----------|-------------|----------------|-----------|
| `caffeine` | C₈H₁₀N₄O₂ | 1130, 1665, 1695, 2010, 2280 | [9] pp. 1130-1132 |
| `aspirin` | Acetylsalicylic acid | 1145, 1435, 1680, 2020, 2140 | [9] pp. 1125-1128 |
| `paracetamol` | Acetaminophen | 1140, 1390, 1510, 1670, 2055, 2260 | [9] pp. 1132-1135 |

#### Fibers and Textiles

| Component | Description | Key Bands (nm) | Reference |
|-----------|-------------|----------------|-----------|
| `cotton` | Cotton cellulose | 1200, 1494, 1780, 2100, 2280, 2345 | [6] pp. 295-298 |
| `polyester` | PET fiber | 1140, 1660, 1720, 2015, 2130, 2255 | [1] pp. 60-62 |

#### Polymers and Plastics

| Component | Description | Key Bands (nm) | Reference |
|-----------|-------------|----------------|-----------|
| `polyethylene` | HDPE/LDPE plastic | 1190, 1720, 2310, 2355 | [15] |
| `polystyrene` | Aromatic polymer | 1145, 1680, 1720, 2170, 2300 | [15] |
| `natural_rubber` | cis-1,4-polyisoprene | 1160, 1720, 2130, 2250, 2350 | [15] |
| `nylon` | Polyamide fiber | 1500, 1720, 2050, 2295 | [1] pp. 60-62 |

#### Solvents

| Component | Description | Key Bands (nm) | Reference |
|-----------|-------------|----------------|-----------|
| `acetone` | Ketone (propan-2-one) | 1690, 1710, 2100, 2300 | [1] pp. 42-44 |

#### Soil Minerals

| Component | Description | Key Bands (nm) | Reference |
|-----------|-------------|----------------|-----------|
| `carbonates` | CaCO₃, MgCO₃ (calcite) | 2330, 2525 | [13] |
| `gypsum` | CaSO₄·2H₂O | 1740, 1900, 2200 | [13] |
| `kaolinite` | Clay mineral | 1400, 2160, 2200 | [13] |

### Component Correlation Groups

Components within the same correlation group can be configured to have correlated concentrations, reflecting real-world relationships:

| Group | Components | Application Domain |
|-------|------------|-----------|
| 1 | water, moisture | Moisture analysis |
| 2 | protein, nitrogen_compound, urea, amino_acid, casein, gluten | Nitrogen/protein analysis |
| 3 | lipid, oil, saturated_fat, unsaturated_fat, waxes | Fat/oil analysis |
| 4 | starch, cellulose, glucose, fructose, sucrose, hemicellulose, lactose, dietary_fiber, cotton | Carbohydrate analysis |
| 5 | chlorophyll, carotenoid, lignin, tannins | Plant constituents |
| 6 | aromatic, alkane | Petrochemical |
| 7 | ethanol, methanol, glycerol | Alcohols/polyols |
| 8 | acetic_acid, citric_acid, lactic_acid, malic_acid, tartaric_acid | Organic acids |
| 9 | caffeine, aspirin, paracetamol | Pharmaceuticals |
| 10 | polyester | Synthetic fibers |
| 11 | polyethylene, polystyrene, natural_rubber, nylon | Polymers |
| 12 | acetone | Solvents |
| 13 | carbonates, gypsum, kaolinite | Soil minerals |

### Usage Example

```python
from nirs4all.data.synthetic import ComponentLibrary

# List all available components (48 total)
library = ComponentLibrary.from_predefined()
print(f"Available components: {library.component_names}")

# Create library with agricultural components
agri_library = ComponentLibrary.from_predefined([
    "water", "protein", "starch", "cellulose", "oil", "chlorophyll"
])

# Create library with dairy components
dairy_library = ComponentLibrary.from_predefined([
    "water", "lactose", "casein", "lipid"
])

# Create library with wine/beverage components
wine_library = ComponentLibrary.from_predefined([
    "water", "ethanol", "glycerol", "glucose", "fructose",
    "malic_acid", "tartaric_acid", "tannins"
])

# Create library with soil analysis components
soil_library = ComponentLibrary.from_predefined([
    "water", "carbonates", "gypsum", "kaolinite", "cellulose", "lignin"
])

# Create library with polymer/plastic components
polymer_library = ComponentLibrary.from_predefined([
    "polyethylene", "polystyrene", "nylon", "polyester"
])
```

## References

The predefined spectral components are based on established NIR spectroscopy literature:

1. Workman Jr, J., & Weyer, L. (2012). *Practical Guide and Spectral Atlas for Interpretive Near-Infrared Spectroscopy* (2nd ed.). CRC Press. — Comprehensive NIR band assignments and spectral atlas.

2. Burns, D. A., & Ciurczak, E. W. (Eds.). (2007). *Handbook of Near-Infrared Analysis* (3rd ed.). CRC Press. — Theory and applications of NIR spectroscopy.

3. Siesler, H. W., Ozaki, Y., Kawata, S., & Heise, H. M. (Eds.). (2002). *Near-Infrared Spectroscopy: Principles, Instruments, Applications*. Wiley-VCH. — Detailed overtone/combination band theory.

4. Osborne, B. G., Fearn, T., & Hindle, P. H. (1993). *Practical NIR Spectroscopy with Applications in Food and Beverage Analysis* (2nd ed.). Longman Scientific & Technical. — Food and agricultural applications.

5. Williams, P. C., & Norris, K. H. (Eds.). (2001). *Near-Infrared Technology in the Agricultural and Food Industries* (2nd ed.). AACC International. — Band assignments for agricultural commodities.

6. Schwanninger, M., Rodrigues, J. C., & Fackler, K. (2011). A Review of Band Assignments in Near Infrared Spectra of Wood and Wood Components. *Journal of Near Infrared Spectroscopy*, 19(5), 287-308. — Comprehensive wood/cellulose band assignments.

7. Murray, I. (1986). The NIR Spectra of Homologous Series of Organic Compounds. In P. C. Williams & K. H. Norris (Eds.), *Near-Infrared Technology in the Agricultural and Food Industries*. AACC. — Fundamental hydrocarbon and alcohol band positions.

8. Bokobza, L. (1998). Near Infrared Spectroscopy. *Journal of Near Infrared Spectroscopy*, 6(1), 3-17. — Review of NIR fundamentals and band assignments.

9. Reich, G. (2005). Near-Infrared Spectroscopy and Imaging: Basic Principles and Pharmaceutical Applications. *Advanced Drug Delivery Reviews*, 57(8), 1109-1143. — Pharmaceutical compound band assignments.

10. Blanco, M., & Villarroya, I. (2002). NIR Spectroscopy: A Rapid-Response Analytical Tool. *TrAC Trends in Analytical Chemistry*, 21(4), 240-250. — Review of NIR applications and typical band positions.

11. Martelo-Vidal, M. J., & Vázquez, M. (2014). Application of Artificial Neural Networks Coupled to UV–VIS–NIR Spectroscopy for the Rapid Quantification of Wine Compounds. *CyTA – Journal of Food*, 12(1), 32-39. — Wine fermentation compounds (glycerol, organic acids, tannins).

12. Luypaert, J., Zhang, M. H., & Massart, D. L. (2003). Feasibility Study for the Use of Near Infrared Spectroscopy in the Qualitative and Quantitative Analysis of Green Tea. *Analytica Chimica Acta*, 478(2), 303-312. — Tea and beverage NIR assignments.

13. Khayamim, F., et al. (2015). Visible and Near-Infrared Spectroscopy for the Prediction of Soil Properties. *Spectroscopy Letters*, 48(6), 405-418. — Soil mineral NIR characterization.

14. Shenk, J. S., Workman Jr, J. J., & Westerhaus, M. O. (2008). Application of NIR Spectroscopy to Agricultural Products. In D. A. Burns & E. W. Ciurczak (Eds.), *Handbook of Near-Infrared Analysis* (3rd ed., pp. 347-386). CRC Press. — Comprehensive agricultural NIR band assignments.

15. Lachenal, G. (1995). Dispersive and Fourier Transform Near-Infrared Spectroscopy of Polymeric Materials. *Vibrational Spectroscopy*, 9(1), 93-100. — Polymer NIR band assignments.

## See Also

- {doc}`/user_guide/data/synthetic_data` - User guide
- {doc}`/api/nirs4all.data.synthetic` - API reference
- {doc}`testing` - Testing guide with synthetic fixtures
