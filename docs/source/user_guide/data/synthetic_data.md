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
from nirs4all.synthesis import SyntheticDatasetBuilder

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

## Non-Linear Target Complexity

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

## Environmental and Matrix Effects

Real NIR spectra are affected by environmental conditions and sample matrix properties. Use these features to generate more realistic synthetic data.

### Temperature Effects

Temperature variations cause peak shifts, intensity changes, and band broadening:

```python
from nirs4all.synthesis import (
    SyntheticDatasetBuilder,
    EnvironmentalEffectsConfig,
    TemperatureConfig,
)

dataset = (
    SyntheticDatasetBuilder(n_samples=1000, random_state=42)
    .with_features(complexity="realistic")
    .with_targets(component=0, range=(0, 100))
    .with_environmental_effects(
        temperature=TemperatureConfig(
            sample_temperature=35.0,      # °C above reference (25°C)
            temperature_variation=5.0,    # Sample-to-sample variation
        ),
    )
    .build()
)
```

| Effect | Description | Impact on Spectra |
|--------|-------------|-------------------|
| Peak shift | O-H bands shift with temperature | ~0.3 nm/°C blue shift |
| Intensity change | H-bonding decreases with temperature | ~0.2%/°C intensity decrease |
| Broadening | Thermal motion widens peaks | ~0.1%/°C width increase |

### Moisture/Water Activity Effects

Water content and activity affect hydrogen bonding and water band shapes:

```python
from nirs4all.synthesis import MoistureConfig

dataset = (
    SyntheticDatasetBuilder(n_samples=1000, random_state=42)
    .with_features(complexity="realistic")
    .with_environmental_effects(
        moisture=MoistureConfig(
            water_activity=0.65,          # Water activity (0-1)
            moisture_content=0.12,        # Fractional moisture
            free_water_fraction=0.4,      # Fraction of free water
        ),
    )
    .build()
)
```

### Particle Size Effects (Scattering)

Particle size affects scattering in diffuse reflectance measurements:

```python
from nirs4all.synthesis import (
    ScatteringEffectsConfig,
    ParticleSizeConfig,
    ParticleSizeDistribution,
)

dataset = (
    SyntheticDatasetBuilder(n_samples=1000, random_state=42)
    .with_features(complexity="realistic")
    .with_scattering_effects(
        particle_size=ParticleSizeConfig(
            distribution=ParticleSizeDistribution(
                mean_size_um=50.0,        # Mean particle size (μm)
                std_size_um=15.0,         # Size variation
                distribution="lognormal", # Distribution type
            ),
        ),
    )
    .build()
)
```

### Combined Effects

Combine environmental and scattering effects for maximum realism:

```python
from nirs4all.synthesis import (
    SyntheticNIRSGenerator,
    EnvironmentalEffectsConfig,
    ScatteringEffectsConfig,
    TemperatureConfig,
    MoistureConfig,
    ParticleSizeConfig,
    EMSCConfig,
)

# Create generator with all Phase 3 effects
generator = SyntheticNIRSGenerator(
    wavelength_start=1000,
    wavelength_end=2500,
    complexity="realistic",
    environmental_config=EnvironmentalEffectsConfig(
        temperature=TemperatureConfig(sample_temperature=30.0),
        moisture=MoistureConfig(water_activity=0.6),
    ),
    scattering_effects_config=ScatteringEffectsConfig(
        particle_size=ParticleSizeConfig(
            distribution=ParticleSizeDistribution(mean_size_um=60.0)
        ),
        emsc=EMSCConfig(multiplicative_range=(0.85, 1.15)),
    ),
    random_state=42,
)

# Generate with effects enabled
X, C, E = generator.generate(
    n_samples=500,
    include_environmental_effects=True,
    include_scattering_effects=True,
)
```

```{note}
Environmental and scattering effects are correctable by standard preprocessing (SNV, MSC, derivatives). Use them to test robustness of your preprocessing pipeline.
```

## Validation and Benchmarking

Phase 4 introduces tools to validate synthetic data quality and benchmark against standard datasets.

### Spectral Realism Scorecard

Evaluate how realistic your synthetic data is compared to real reference data using 6 quantitative metrics:

```python
from nirs4all.synthesis import compute_spectral_realism_scorecard

# Compare synthetic data to real reference data
score = compute_spectral_realism_scorecard(
    real_spectra=X_real,
    synthetic_spectra=X_synth,
    wavelengths=wavelengths,
    include_adversarial=True
)

print(f"Overall Pass: {score.overall_pass}")
print(f"Correlation Length Overlap: {score.correlation_length_overlap:.3f}")
print(f"Adversarial AUC: {score.adversarial_auc:.3f}")  # Lower is better (harder to distinguish)
```

### Benchmark Datasets

Access metadata and properties for standard NIR benchmark datasets to create matching synthetic data:

```python
from nirs4all.synthesis import (
    list_benchmark_datasets,
    get_benchmark_info,
    create_synthetic_matching_benchmark
)

# List available benchmarks
print(list_benchmark_datasets())
# ['corn', 'tecator', 'shootout2002', 'wheat_kernels', ...]

# Get info about a dataset
info = get_benchmark_info("corn")
print(f"{info.full_name}: {info.n_samples} samples, {info.n_wavelengths} wavelengths")

# Create synthetic data matching the benchmark properties
X, C, E = create_synthetic_matching_benchmark("corn", n_samples=1000)
```

### Prior Sampling

Generate realistic configurations based on domain knowledge (hierarchical sampling):

```python
from nirs4all.synthesis import sample_prior

# Sample a random realistic configuration
config = sample_prior(random_state=42)
print(f"Domain: {config['domain']}")
print(f"Instrument: {config['instrument']}")

# Sample for a specific domain
food_config = sample_prior(domain="food")
```

### GPU Acceleration

Accelerate generation of large datasets using JAX or CuPy (automatically detected):

```python
from nirs4all.synthesis import AcceleratedGenerator

# Automatically uses GPU if available (JAX/CuPy)
gen = AcceleratedGenerator(random_state=42)

# Generate large batch efficiently
X = gen.generate_batch(
    n_samples=100000,
    wavelengths=wavelengths,
    component_spectra=E,
    concentrations=C
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

The generator includes **116 predefined spectral components** with physically-accurate NIR band assignments based on published spectroscopy literature. Use these directly or as building blocks for custom scenarios.

**Water & Moisture (2):**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"water"` | 1450, 1940, 2500 | Free water O-H overtones |
| `"moisture"` | 1460, 1930 | Bound water in organic matrices |

**Proteins & Nitrogen Compounds (12):**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"protein"` | 1510, 1680, 2050, 2180 | Amide N-H and aromatic C-H |
| `"nitrogen_compound"` | 1500, 2060, 2150 | Primary/secondary amines |
| `"urea"` | 1480, 1530, 2010, 2170 | Urea CO(NH₂)₂ |
| `"amino_acid"` | 1520, 2040, 2260 | Free amino acids |
| `"casein"` | 1510, 1680, 2050, 2180 | Milk protein |
| `"gluten"` | 1505, 1680, 2050, 2180 | Wheat protein complex |
| `"albumin"` | 1512, 1682, 2055, 2182 | Globular protein (egg white, serum) |
| `"collagen"` | 1508, 1560, 2048, 2175 | Fibrous structural protein |
| `"keratin"` | 1520, 1685, 2060, 2185 | Structural protein (hair, nails) |
| `"zein"` | 1515, 1676, 2052, 2178 | Corn protein (prolamin) |
| `"gelatin"` | 1505, 2045, 2172 | Denatured collagen |
| `"whey"` | 1514, 1680, 2052, 2180 | Milk serum proteins |

**Lipids & Hydrocarbons (15):**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"lipid"` | 1210, 1390, 1720, 2310 | Triglyceride C-H stretching |
| `"oil"` | 1165, 1725, 2305 | Vegetable/mineral oils |
| `"saturated_fat"` | 1195, 1730, 2315 | Saturated fatty acids |
| `"unsaturated_fat"` | 1160, 1720, 2145 | Mono/polyunsaturated fats (=C-H) |
| `"waxes"` | 1190, 1720, 2310 | Cuticular waxes |
| `"aromatic"` | 1145, 1685, 2150 | Benzene derivatives |
| `"alkane"` | 1190, 1715, 2310 | Saturated hydrocarbons |
| `"oleic_acid"` | 1162, 1722, 2142, 2308 | Monounsaturated fatty acid (C18:1) |
| `"linoleic_acid"` | 1158, 1718, 2138, 2305 | Polyunsaturated fatty acid (C18:2) |
| `"linolenic_acid"` | 1155, 1715, 2135, 2302 | Polyunsaturated fatty acid (C18:3) |
| `"palmitic_acid"` | 1196, 1732, 2318, 2358 | Saturated fatty acid (C16:0) |
| `"stearic_acid"` | 1194, 1728, 2315, 2355 | Saturated fatty acid (C18:0) |
| `"phospholipid"` | 1205, 1725, 2165, 2305 | Lecithin-like membrane lipids |
| `"cholesterol"` | 1390, 1708, 2298 | Sterol lipid |
| `"cocoa_butter"` | 1210, 1728, 2312, 2352 | Triglyceride mix |

**Petroleum & Hydrocarbons (5):**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"crude_oil"` | 1647, 1712, 1759, 2310, 2348 | Petroleum crude oil (TPH, PAH) |
| `"diesel"` | 1675, 1725, 1762, 2310, 2355 | Diesel fuel |
| `"gasoline"` | 1140, 1665, 1720, 2140, 2305 | Gasoline/petrol (high aromatics) |
| `"kerosene"` | 1138, 1670, 1722, 1758, 2308 | Kerosene/jet fuel |
| `"pah"` | 1140, 1647, 2150, 2450 | Polycyclic aromatic hydrocarbons |

**Carbohydrates (18):**
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
| `"cotton"` | 1200, 1494, 1780, 2100 | Cotton cellulose fiber |
| `"dietary_fiber"` | 1490, 1770, 2090, 2275 | Plant cell wall material |
| `"maltose"` | 1445, 1688, 2078, 2268 | Malt sugar (glucose disaccharide) |
| `"raffinose"` | 1448, 1692, 2082, 2272 | Trisaccharide |
| `"inulin"` | 1432, 1698, 2068, 2258 | Fructose polymer |
| `"xylose"` | 1438, 1686, 2072, 2262 | Pentose monosaccharide |
| `"arabinose"` | 1442, 1684, 2076, 2266 | Pentose monosaccharide |
| `"galactose"` | 1442, 1692, 2082, 2272 | Hexose monosaccharide |
| `"mannose"` | 1444, 1688, 2078, 2268 | Hexose monosaccharide |
| `"trehalose"` | 1440, 1686, 2076, 2266 | Non-reducing disaccharide |

**Alcohols & Polyols (9):**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"ethanol"` | 1410, 1580, 1695, 2050 | Ethanol C₂H₅OH |
| `"methanol"` | 1400, 1545, 1705, 2040 | Methanol CH₃OH |
| `"glycerol"` | 1450, 1580, 1700, 2060 | Polyol (fermentation) |
| `"propanol"` | 1415, 1575, 1698, 2055 | Propyl alcohol |
| `"butanol"` | 1418, 1572, 1702, 2058 | Butyl alcohol |
| `"sorbitol"` | 1445, 1585, 1695, 2065 | Sugar alcohol |
| `"mannitol"` | 1448, 1582, 1698, 2068 | Sugar alcohol |
| `"xylitol"` | 1442, 1578, 1692, 2062 | Sugar alcohol |
| `"isopropanol"` | 1412, 1568, 1690, 2048 | Isopropyl alcohol |

**Organic Acids (12):**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"acetic_acid"` | 1420, 1700, 1940 | Acetic acid CH₃COOH |
| `"citric_acid"` | 1440, 1920, 2060 | Citric acid (fruit acids) |
| `"lactic_acid"` | 1430, 1485, 1700, 2020 | Lactic acid |
| `"malic_acid"` | 1440, 1920, 2050, 2255 | Fruit acid (apples) |
| `"tartaric_acid"` | 1435, 1910, 2040, 2260 | Grape/wine acid |
| `"formic_acid"` | 1425, 1695, 1935 | Formic acid HCOOH |
| `"oxalic_acid"` | 1435, 1705, 1945 | Oxalic acid (COOH)₂ |
| `"succinic_acid"` | 1432, 1705, 1942, 2245 | Dicarboxylic acid |
| `"fumaric_acid"` | 1428, 1700, 1938, 2135 | Unsaturated dicarboxylic |
| `"propionic_acid"` | 1422, 1698, 1938, 2242 | Propionic acid |
| `"butyric_acid"` | 1420, 1702, 1940, 2248 | Short-chain fatty acid |
| `"ascorbic_acid"` | 1445, 1565, 1918, 2062 | Vitamin C |

**Plant Pigments & Phenolics (8):**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"chlorophyll"` | 1070, 1400, 1730, 2270 | Chlorophyll a/b |
| `"carotenoid"` | 1050, 1680, 2135 | β-carotene, xanthophylls |
| `"tannins"` | 1420, 1670, 2056, 2270 | Phenolic compounds |
| `"anthocyanin"` | 1040, 1425, 1672, 2055 | Red-purple plant pigment |
| `"lycopene"` | 1055, 1685, 2138, 2282 | Red carotenoid (tomatoes) |
| `"lutein"` | 1048, 1415, 1678, 2132 | Yellow carotenoid |
| `"xanthophyll"` | 1052, 1418, 1682, 2135 | General yellow pigments |
| `"melanin"` | 1100, 1510, 1680, 2055 | Brown-black pigment |

**Pharmaceuticals (10):**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"caffeine"` | 1130, 1665, 1695, 2010 | Caffeine C₈H₁₀N₄O₂ |
| `"aspirin"` | 1145, 1435, 1680, 2020 | Acetylsalicylic acid |
| `"paracetamol"` | 1140, 1390, 1510, 1670 | Acetaminophen |
| `"ibuprofen"` | 1148, 1432, 1682, 1722 | Anti-inflammatory drug |
| `"naproxen"` | 1145, 1430, 1678, 2028 | NSAID drug |
| `"diclofenac"` | 1142, 1505, 1675, 2032 | NSAID drug |
| `"metformin"` | 1485, 1545, 2045, 2168 | Diabetes drug |
| `"omeprazole"` | 1148, 1498, 1672, 2038 | Proton pump inhibitor |
| `"amoxicillin"` | 1148, 1390, 1512, 1675 | Antibiotic |
| `"microcrystalline_cellulose"` | 1492, 1782, 2092, 2282 | Pharmaceutical excipient |

**Fibers & Textiles (2):**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"polyester"` | 1140, 1660, 1720, 2015 | PET synthetic fiber |
| `"nylon"` | 1500, 1720, 2050, 2295 | Polyamide fiber |

**Polymers & Plastics (10):**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"polyethylene"` | 1190, 1720, 2310, 2355 | HDPE/LDPE plastic |
| `"polystyrene"` | 1145, 1680, 1720, 2170 | Aromatic polymer |
| `"natural_rubber"` | 1160, 1720, 2130, 2250 | cis-1,4-polyisoprene |
| `"pmma"` | 1190, 1718, 2015, 2298 | Polymethyl methacrylate (acrylic) |
| `"pvc"` | 1188, 1715, 2295 | Polyvinyl chloride |
| `"polypropylene"` | 1185, 1392, 1718, 2305 | PP plastic |
| `"pet"` | 1138, 1658, 1722, 2018 | Polyethylene terephthalate |
| `"ptfe"` | 2180, 2365 | Polytetrafluoroethylene (Teflon) |
| `"abs"` | 1145, 1158, 1682, 1718 | Acrylonitrile butadiene styrene |

**Solvents (6):**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"acetone"` | 1690, 1710, 2100, 2300 | Ketone (propan-2-one) |
| `"dmso"` | 1700, 2020, 2290 | Dimethyl sulfoxide |
| `"ethyl_acetate"` | 1695, 1720, 2010, 2285 | Ester solvent |
| `"toluene"` | 1142, 1678, 1705, 2145 | Aromatic solvent |
| `"chloroform"` | 1695, 2250 | Halogenated solvent |
| `"hexane"` | 1192, 1718, 2308, 2358 | Alkane solvent |

**Soil Minerals (8):**
| Component | Key Bands (nm) | Description |
|-----------|----------------|-------------|
| `"carbonates"` | 2330, 2525 | CaCO₃, MgCO₃ (calcite) |
| `"gypsum"` | 1740, 1900, 2200 | CaSO₄·2H₂O |
| `"kaolinite"` | 1400, 2160, 2200 | Clay mineral |
| `"montmorillonite"` | 1410, 1910, 2210 | Smectite clay |
| `"illite"` | 1405, 2205, 2345 | Mica-like clay |
| `"goethite"` | 1420, 1920, 2260 | Iron oxyhydroxide |
| `"talc"` | 1395, 2315, 2390 | Magnesium silicate |
| `"silica"` | 1380, 1900, 2220 | Silicon dioxide |

```python
# Use specific components
dataset = nirs4all.generate(
    components=["water", "protein", "lipid"],
    n_samples=1000
)

# List all available components
from nirs4all.synthesis import ComponentLibrary
library = ComponentLibrary.from_predefined()
print(library.component_names)  # All 116 component names
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
    "synthesis",
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
    "synthesis.csv",
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
from nirs4all.synthesis import (
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
| `NonLinearTargetProcessor` | Non-linear target complexity |
| `NonLinearTargetConfig` | Configuration for target complexity |

### Environmental & Scattering Classes (Phase 3)

| Class | Description |
|-------|-------------|
| `TemperatureConfig` | Temperature effect configuration |
| `MoistureConfig` | Moisture/water activity configuration |
| `EnvironmentalEffectsConfig` | Combined environmental effects |
| `EnvironmentalEffectsSimulator` | Apply temperature and moisture effects |
| `ParticleSizeConfig` | Particle size distribution configuration |
| `ParticleSizeDistribution` | Sample particle size distributions |
| `EMSCConfig` | EMSC-style scattering configuration |
| `ScatteringCoefficientConfig` | Scattering coefficient generation |
| `ScatteringEffectsConfig` | Combined scattering effects |
| `ScatteringEffectsSimulator` | Apply particle size and scattering effects |

### Validation & Benchmarking Classes (Phase 4)

| Class | Description |
|-------|-------------|
| `SpectralRealismScore` | Scorecard results container |
| `RealismMetric` | Enum of validation metrics |
| `BenchmarkDatasetInfo` | Metadata for benchmark datasets |
| `NIRSPriorConfig` | Configuration for prior sampling |
| `PriorSampler` | Hierarchical configuration sampler |
| `AcceleratedGenerator` | GPU-accelerated generation engine |
| `AcceleratorBackend` | Enum of acceleration backends (JAX, CuPy, NumPy) |

### Builder Methods for Target Complexity

| Method | Description |
|--------|-------------|
| `.with_nonlinear_targets()` | Add polynomial, synergistic, or antagonistic interactions |
| `.with_target_complexity()` | Add confounders and partial predictability |
| `.with_complex_target_landscape()` | Create multi-regime target landscapes |

## See Also

- {doc}`/developer/synthetic` - Developer guide for extending the generator
- {doc}`/api/nirs4all.api.generate` - API reference
- {doc}`/api/nirs4all.synthesis` - Low-level classes reference
- {doc}`loading_data` - Loading real datasets
- {doc}`/getting_started/concepts` - Understanding SpectroDataset

```{seealso}
**Related Examples:**
- [U05: Synthetic Data](../../../examples/user/02_data_handling/U05_synthetic_data.py) - Quick synthetic data generation
- [U06: Advanced Synthetic Data](../../../examples/user/02_data_handling/U06_synthetic_advanced.py) - SyntheticDatasetBuilder for full control
- [D05: Custom Components](../../../examples/developer/02_generators/D05_synthetic_custom_components.py) - Custom spectral component libraries
```
