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
├── _constants.py         # Predefined components, defaults
├── metadata.py           # MetadataGenerator
├── targets.py            # TargetGenerator (regression/classification)
├── sources.py            # MultiSourceGenerator
├── exporter.py           # DatasetExporter, CSVVariationGenerator
├── fitter.py             # RealDataFitter
└── validation.py         # Data validation utilities
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

## References

1. Workman Jr, J., & Weyer, L. (2012). *Practical Guide and Spectral Atlas for Interpretive Near-Infrared Spectroscopy*. CRC Press.

2. Burns, D. A., & Ciurczak, E. W. (2007). *Handbook of Near-Infrared Analysis*. CRC Press.

3. NIR band assignments based on spectroscopic databases and published literature.

## See Also

- {doc}`/user_guide/data/synthetic_data` - User guide
- {doc}`/api/nirs4all.data.synthetic` - API reference
- {doc}`testing` - Testing guide with synthetic fixtures
