# Synthetic Data Generation

Generate realistic synthetic NIRS spectra for testing, prototyping, and research.

## Overview

NIRS4ALL includes a powerful synthetic data generator based on **Beer-Lambert law** physics with realistic instrumental effects. This enables:

- **Reproducible testing**: Generate deterministic datasets for CI/CD pipelines
- **Prototyping**: Quickly explore pipelines without needing real data
- **Research**: Create controlled experiments with known ground truth
- **Teaching**: Demonstrate NIRS concepts with configurable examples

```{note}
The synthetic generator produces physically-motivated spectra with Voigt profile peak shapes and realistic noise models, making them suitable for algorithm development and validation.
```

## Quick Start

### Simple Generation

```python
import nirs4all

# Generate a dataset with 1000 samples
dataset = nirs4all.generate(n_samples=1000, random_state=42)

# Use immediately in a pipeline
result = nirs4all.run(
    pipeline=[MinMaxScaler(), PLSRegression(10)],
    dataset=dataset
)
```

### Get Raw Arrays

```python
# Get numpy arrays for quick experiments
X, y = nirs4all.generate(n_samples=500, as_dataset=False, random_state=42)

print(f"Features shape: {X.shape}")  # (500, 751)
print(f"Targets shape: {y.shape}")    # (500,) or (500, n_components)
```

## Convenience Functions

### Regression Datasets

```python
# Basic regression dataset
dataset = nirs4all.generate.regression(n_samples=500)

# With specific target range and distribution
dataset = nirs4all.generate.regression(
    n_samples=1000,
    target_range=(0, 100),           # Scale targets to 0-100
    target_component="protein",       # Use protein concentration as target
    distribution="lognormal",         # Lognormal concentration distribution
    random_state=42
)
```

### Classification Datasets

```python
# Binary classification
dataset = nirs4all.generate.classification(
    n_samples=500,
    n_classes=2,
    random_state=42
)

# Multiclass with imbalanced classes
dataset = nirs4all.generate.classification(
    n_samples=1000,
    n_classes=3,
    class_weights=[0.5, 0.3, 0.2],    # 50%, 30%, 20% class distribution
    class_separation=2.0,              # Higher = more separable classes
    random_state=42
)
```

### Multi-Source Datasets

Combine multiple data types (e.g., NIR spectra + chemical markers):

```python
dataset = nirs4all.generate.multi_source(
    n_samples=500,
    sources=[
        {"name": "NIR", "type": "nir", "wavelength_range": (1000, 2500)},
        {"name": "markers", "type": "aux", "n_features": 15}
    ],
    random_state=42
)
```

## Builder API

For full control, use the fluent builder interface:

```python
from nirs4all.data.synthetic import SyntheticDatasetBuilder

dataset = (
    SyntheticDatasetBuilder(n_samples=1000, random_state=42)
    # Configure spectral features
    .with_features(
        wavelength_range=(1000, 2500),
        complexity="realistic",              # simple, realistic, or complex
        components=["water", "protein", "lipid", "starch"]
    )
    # Configure target generation
    .with_targets(
        distribution="lognormal",
        range=(5, 50),
        component="protein"                  # Use protein as primary target
    )
    # Add metadata
    .with_metadata(
        n_groups=5,                          # 5 sample groups
        n_repetitions=(2, 4)                 # 2-4 measurements per sample
    )
    # Configure partitioning
    .with_partitions(train_ratio=0.8)
    # Add batch effects (for domain adaptation research)
    .with_batch_effects(n_batches=3)
    .build()
)
```

## Configuration Options

### Complexity Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `"simple"` | Minimal noise, no scatter | Unit tests, quick prototyping |
| `"realistic"` | Typical NIR noise and effects | Algorithm development, validation |
| `"complex"` | High noise, artifacts, outliers | Robustness testing |

```python
# Simple (fast, clean spectra)
dataset = nirs4all.generate(complexity="simple", n_samples=1000)

# Realistic (recommended for most use cases)
dataset = nirs4all.generate(complexity="realistic", n_samples=1000)

# Complex (challenging scenarios)
dataset = nirs4all.generate(complexity="complex", n_samples=1000)
```

### Predefined Components

Available spectral components based on known NIR band assignments:

| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"water"` | 1450, 1940, 2500 | O-H overtones and combinations |
| `"protein"` | 1510, 1680, 2050, 2180 | N-H and C-H bonds |
| `"lipid"` | 1210, 1390, 1720, 2310 | C-H stretching |
| `"starch"` | 1460, 1580, 2100, 2270 | O-H and C-O combinations |
| `"cellulose"` | 1490, 1780, 2090, 2280 | Cellulose fingerprint |
| `"chlorophyll"` | 1070, 1400, 2270 | Plant pigments |
| `"oil"` | 1165, 1725, 2305 | Unsaturated C-H |
| `"nitrogen_compound"` | 1500, 2060, 2150 | N-H combinations |

```python
# Use specific components
dataset = nirs4all.generate(
    components=["water", "protein", "lipid"],
    n_samples=1000
)
```

### Concentration Distributions

| Distribution | Description | Use Case |
|--------------|-------------|----------|
| `"dirichlet"` | Sum-to-one constraint | Natural composition data |
| `"uniform"` | Independent uniform | Wide concentration range |
| `"lognormal"` | Skewed, realistic | Agricultural/biological data |
| `"correlated"` | With inter-component correlations | Complex mixtures |

### Target Configuration

```python
# Single component target with scaling
dataset = nirs4all.generate.builder(n_samples=1000)
    .with_targets(
        component="protein",       # Use one component
        range=(0, 100),            # Scale to percentage
        distribution="lognormal"
    )
    .build()

# Multi-output regression (all components as targets)
dataset = nirs4all.generate.builder(n_samples=1000)
    .with_targets(
        component=None,            # Use all components
        range=(0, 1),              # Normalize
    )
    .build()
```

## Exporting Synthetic Data

### To Folder (DatasetConfigs compatible)

```python
# Generate and save to folder
path = nirs4all.generate.to_folder(
    "data/synthetic",
    n_samples=1000,
    train_ratio=0.8,
    format="standard",          # Creates Xcal, Ycal, Xval, Yval files
    random_state=42
)

# Later, load with DatasetConfigs
from nirs4all.data import DatasetConfigs
dataset = DatasetConfigs(path)
```

### Export Formats

| Format | Description | Files Created |
|--------|-------------|---------------|
| `"standard"` | Separate train/test files | Xcal.csv, Ycal.csv, Xval.csv, Yval.csv |
| `"single"` | All data in one file | data.csv (with partition column) |
| `"fragmented"` | Multiple small files | Useful for loader testing |

### To Single CSV

```python
path = nirs4all.generate.to_csv(
    "data/synthetic.csv",
    n_samples=500,
    random_state=42
)
```

## Matching Real Data

Generate synthetic data that resembles a real dataset:

```python
# From a dataset path
dataset = nirs4all.generate.from_template(
    "sample_data/regression",
    n_samples=1000,
    random_state=42
)

# From numpy arrays
dataset = nirs4all.generate.from_template(
    X_real,
    n_samples=500,
    wavelengths=wavelengths,
    random_state=42
)
```

The fitter analyzes:
- Statistical properties (mean, std, range)
- Spectral shape (slope, curvature)
- Noise characteristics
- PCA structure

## Advanced: Custom Component Library

Create custom spectral components for specific applications:

```python
from nirs4all.data.synthetic import (
    SyntheticNIRSGenerator,
    ComponentLibrary,
    SpectralComponent,
    NIRBand
)

# Define custom component
my_component = SpectralComponent(
    name="my_compound",
    bands=[
        NIRBand(center=1500, sigma=20, gamma=2, amplitude=0.6, name="C-H stretch"),
        NIRBand(center=2100, sigma=30, gamma=3, amplitude=0.8, name="O-H combination"),
    ]
)

# Create library with custom and predefined components
library = ComponentLibrary()
library.add_component(my_component)
library.add_from_predefined(["water", "protein"])

# Generate with custom library
generator = SyntheticNIRSGenerator(
    component_library=library,
    random_state=42
)
X, Y, E = generator.generate(n_samples=1000)
```

## Integration with Pipelines

### Direct Usage

```python
import nirs4all
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

# Generate and train in one workflow
result = nirs4all.run(
    pipeline=[
        StandardScaler(),
        KFold(n_splits=5),
        {"model": PLSRegression(n_components=10)}
    ],
    dataset=nirs4all.generate(n_samples=1000, random_state=42),
    name="synthetic_test",
    verbose=1
)

print(f"Best RMSE: {result.best_rmse:.4f}")
```

### Comparing Preprocessing Methods

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from nirs4all.operators.transforms import SNV, MSC, FirstDerivative

# Generate consistent dataset
dataset = nirs4all.generate(n_samples=500, complexity="realistic", random_state=42)

# Test different preprocessing
for preproc in [MinMaxScaler(), StandardScaler(), SNV(), MSC(), FirstDerivative()]:
    result = nirs4all.run(
        pipeline=[preproc, KFold(3), PLSRegression(10)],
        dataset=dataset,
        verbose=0
    )
    print(f"{preproc.__class__.__name__}: RMSE={result.best_rmse:.4f}")
```

## Performance

| Samples | Complexity | Approximate Time |
|---------|------------|-----------------|
| 1,000 | simple | ~0.05s |
| 1,000 | realistic | ~0.1s |
| 10,000 | realistic | ~0.5s |
| 100,000 | complex | ~5s |

## API Reference

### Top-Level Functions

| Function | Description |
|----------|-------------|
| `nirs4all.generate()` | Main generation function |
| `nirs4all.generate.regression()` | Regression dataset |
| `nirs4all.generate.classification()` | Classification dataset |
| `nirs4all.generate.multi_source()` | Multi-source dataset |
| `nirs4all.generate.builder()` | Get builder for full control |
| `nirs4all.generate.to_folder()` | Generate and export to folder |
| `nirs4all.generate.to_csv()` | Generate and export to CSV |
| `nirs4all.generate.from_template()` | Generate matching real data |

### Core Classes

| Class | Description |
|-------|-------------|
| `SyntheticNIRSGenerator` | Core generation engine |
| `SyntheticDatasetBuilder` | Fluent builder interface |
| `ComponentLibrary` | Collection of spectral components |
| `SpectralComponent` | Single chemical component definition |
| `NIRBand` | Single absorption band (Voigt profile) |

## See Also

- {doc}`/developer/synthetic` - Developer guide for extending the generator
- {doc}`/api/nirs4all.api.generate` - API reference
- {doc}`/api/nirs4all.data.synthetic` - Low-level classes reference
- {doc}`loading_data` - Loading real datasets
- {doc}`/getting_started/concepts` - Understanding SpectroDataset
