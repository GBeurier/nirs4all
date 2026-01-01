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

## Non-Linear Target Complexity (NEW!)

By default, synthetic targets have a simple linear relationship with spectral features,
making them too easy to predict. Use these methods to create more realistic, challenging datasets:

### Non-Linear Interactions

Add polynomial, synergistic, or antagonistic relationships between concentrations and targets:

```python
dataset = (
    SyntheticDatasetBuilder(n_samples=1000, random_state=42)
    .with_features(complexity="realistic")
    .with_targets(component=0, range=(0, 100))
    .with_nonlinear_targets(
        interactions="polynomial",     # polynomial, synergistic, antagonistic
        interaction_strength=0.6,      # 0=linear, 1=fully non-linear
        hidden_factors=2,              # Latent variables not in spectra
        polynomial_degree=2            # Quadratic terms
    )
    .build()
)
```

| Interaction Type | Description | Use Case |
|------------------|-------------|----------|
| `"polynomial"` | C₁², C₁×C₂, C₁×C₂×C₃ terms | General non-linearity |
| `"synergistic"` | Combinations enhance effect | Chemical synergies |
| `"antagonistic"` | Michaelis-Menten saturation | Enzyme kinetics, inhibition |

### Confounders and Partial Predictability

Introduce factors that make the target only partially predictable from spectra:

```python
dataset = (
    SyntheticDatasetBuilder(n_samples=1000, random_state=42)
    .with_features(complexity="realistic")
    .with_targets(component=0, range=(0, 100))
    .with_target_complexity(
        signal_to_confound_ratio=0.7,  # 70% predictable, 30% irreducible error
        n_confounders=2,               # Variables affecting both spectra and target
        temporal_drift=True            # Relationship changes over samples
    )
    .build()
)
```

### Multi-Regime Target Landscapes

Create regions in feature space with different target-spectra relationships:

```python
dataset = (
    SyntheticDatasetBuilder(n_samples=1000, random_state=42)
    .with_features(complexity="realistic")
    .with_targets(component=0, range=(0, 100))
    .with_complex_target_landscape(
        n_regimes=3,                      # 3 different relationship regimes
        regime_method="concentration",    # concentration, spectral, or random
        regime_overlap=0.2,               # Smooth transitions between regimes
        noise_heteroscedasticity=0.5      # Noise varies by regime
    )
    .build()
)
```

### Combining All Complexity Features

For realistic benchmarking, combine all complexity features:

```python
# Create a challenging benchmark dataset
dataset = (
    SyntheticDatasetBuilder(n_samples=1000, random_state=42)
    .with_features(complexity="realistic")
    .with_targets(component=0, range=(0, 100))
    # Non-linear interactions
    .with_nonlinear_targets(
        interactions="polynomial",
        interaction_strength=0.5,
        hidden_factors=2
    )
    # Confounders
    .with_target_complexity(
        signal_to_confound_ratio=0.7,
        n_confounders=2
    )
    # Multi-regime
    .with_complex_target_landscape(
        n_regimes=3,
        noise_heteroscedasticity=0.3
    )
    .build()
)
```

```{note}
These features help test whether your model can handle:
- Non-linear relationships (try tree-based models, neural networks)
- Irreducible error (avoid overfitting)
- Subpopulations with different behaviors (local models, mixture models)
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

The generator includes **48 predefined spectral components** with physically-accurate NIR band assignments based on published spectroscopy literature. Use these directly or as building blocks for custom scenarios.

**Water & Moisture:**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"water"` | 1450, 1940, 2500 | Free water O-H overtones |
| `"moisture"` | 1460, 1930 | Bound water in organic matrices |

**Proteins & Nitrogen Compounds:**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"protein"` | 1510, 1680, 2050, 2180 | Amide N-H and aromatic C-H |
| `"nitrogen_compound"` | 1500, 2060, 2150 | Primary/secondary amines |
| `"urea"` | 1480, 1530, 2010, 2170 | Urea CO(NH₂)₂ |
| `"amino_acid"` | 1520, 2040, 2260 | Free amino acids |
| `"casein"` | 1510, 1680, 2050, 2180 | Milk protein |
| `"gluten"` | 1505, 1680, 2050, 2180 | Wheat protein complex |

**Lipids & Hydrocarbons:**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"lipid"` | 1210, 1390, 1720, 2310 | Triglyceride C-H stretching |
| `"oil"` | 1165, 1725, 2305 | Vegetable/mineral oils |
| `"saturated_fat"` | 1195, 1730, 2315 | Saturated fatty acids |
| `"unsaturated_fat"` | 1160, 1720, 2145 | Mono/polyunsaturated fats (=C-H) |
| `"waxes"` | 1190, 1720, 2310 | Cuticular waxes |
| `"aromatic"` | 1145, 1685, 2150 | Benzene derivatives |
| `"alkane"` | 1190, 1715, 2310 | Saturated hydrocarbons |

**Carbohydrates:**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"starch"` | 1460, 1580, 2100, 2270 | Amylose/amylopectin |
| `"cellulose"` | 1490, 1780, 2090, 2280 | β-1,4-glucan chains |
| `"glucose"` | 1440, 1690, 2080, 2270 | D-glucose monosaccharide |
| `"fructose"` | 1430, 1695, 2070 | D-fructose (fruit sugar) |
| `"sucrose"` | 1435, 1685, 2075 | Disaccharide (table sugar) |
| `"lactose"` | 1450, 1690, 1940, 2100 | Milk sugar |
| `"hemicellulose"` | 1470, 1760, 2085 | Xylan/glucomannan |
| `"lignin"` | 1140, 1420, 1670, 2130 | Aromatic plant polymer |
| `"dietary_fiber"` | 1490, 1770, 2090, 2275 | Plant cell wall material |

**Alcohols & Polyols:**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"ethanol"` | 1410, 1580, 1695, 2050 | Ethanol C₂H₅OH |
| `"methanol"` | 1400, 1545, 1705, 2040 | Methanol CH₃OH |
| `"glycerol"` | 1450, 1580, 1700, 2060 | Polyol (fermentation) |

**Organic Acids:**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"acetic_acid"` | 1420, 1700, 1940 | Acetic acid CH₃COOH |
| `"citric_acid"` | 1440, 1920, 2060 | Citric acid (fruit acids) |
| `"lactic_acid"` | 1430, 1485, 1700, 2020 | Lactic acid |
| `"malic_acid"` | 1440, 1920, 2050, 2255 | Fruit acid (apples) |
| `"tartaric_acid"` | 1435, 1910, 2040, 2260 | Grape/wine acid |

**Plant Pigments & Phenolics:**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"chlorophyll"` | 1070, 1400, 1730, 2270 | Chlorophyll a/b |
| `"carotenoid"` | 1050, 1680, 2135 | β-carotene, xanthophylls |
| `"tannins"` | 1420, 1670, 2056, 2270 | Phenolic compounds |

**Pharmaceuticals:**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"caffeine"` | 1130, 1665, 1695, 2010 | Caffeine C₈H₁₀N₄O₂ |
| `"aspirin"` | 1145, 1435, 1680, 2020 | Acetylsalicylic acid |
| `"paracetamol"` | 1140, 1390, 1510, 1670 | Acetaminophen |

**Fibers & Textiles:**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"cotton"` | 1200, 1494, 1780, 2100 | Cotton cellulose fiber |
| `"polyester"` | 1140, 1660, 1720, 2015 | PET synthetic fiber |
| `"nylon"` | 1500, 1720, 2050, 2295 | Polyamide fiber |

**Polymers & Plastics:**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"polyethylene"` | 1190, 1720, 2310, 2355 | HDPE/LDPE plastic |
| `"polystyrene"` | 1145, 1680, 1720, 2170 | Aromatic polymer |
| `"natural_rubber"` | 1160, 1720, 2130, 2250 | cis-1,4-polyisoprene |

**Solvents:**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"acetone"` | 1690, 1710, 2100, 2300 | Ketone (propan-2-one) |

**Soil Minerals:**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"carbonates"` | 2330, 2525 | CaCO₃, MgCO₃ (calcite) |
| `"gypsum"` | 1740, 1900, 2200 | CaSO₄·2H₂O |
| `"kaolinite"` | 1400, 2160, 2200 | Clay mineral |

```python
# Use specific components
dataset = nirs4all.generate(
    components=["water", "protein", "lipid"],
    n_samples=1000
)

# List all available components
from nirs4all.data.synthetic import ComponentLibrary
library = ComponentLibrary.from_predefined()
print(library.component_names)  # All 48 component names
```

```{seealso}
For detailed band assignments with literature references, see {doc}`/developer/synthetic`.
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
| `NonLinearTargetProcessor` | Non-linear target complexity (NEW!) |
| `NonLinearTargetConfig` | Configuration for target complexity |

### Builder Methods for Target Complexity

| Method | Description |
|--------|-------------|
| `.with_nonlinear_targets()` | Add polynomial, synergistic, or antagonistic interactions |
| `.with_target_complexity()` | Add confounders and partial predictability |
| `.with_complex_target_landscape()` | Create multi-regime target landscapes |

## See Also

- {doc}`/developer/synthetic` - Developer guide for extending the generator
- {doc}`/api/nirs4all.api.generate` - API reference
- {doc}`/api/nirs4all.data.synthetic` - Low-level classes reference
- {doc}`loading_data` - Loading real datasets
- {doc}`/getting_started/concepts` - Understanding SpectroDataset
