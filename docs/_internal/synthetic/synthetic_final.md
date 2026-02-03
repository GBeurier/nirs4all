# Synthetic NIRS Generator: Notebook vs nirs4all API

A comprehensive analysis comparing the `generator_final.ipynb` approach with the `nirs4all.synthesis` module capabilities.

---

## 1. What is Done in generator_final.ipynb

### Overview

The notebook presents a **systematic evaluation framework** for fitting 8 different spectral decomposition approaches to real NIRS datasets, then generating synthetic data and evaluating its quality.

### Datasets Analyzed

9 real-world NIRS datasets covering diverse applications:
- Beer (extract content)
- Biscuit (fat)
- Diesel (boiling point)
- Grapevine (chloride)
- LUCAS soil (organic carbon)
- Milk (fat)
- Poultry manure (CaO)
- Rice (amylose)
- Tablet (escitalopram)

### Spectral Fitting Approaches

#### A. Pure Band Fitting (Unconstrained Gaussians)
- Iteratively adds Gaussian profiles to minimize residuals
- No prior knowledge of NIR chemistry
- Parameters: `max_bands=50`, `min_sigma=1.0`, `max_sigma=300.0`, `target_r2=0.99`

#### B. Real Band Fitting (Literature-Constrained)
- Uses known NIR band assignments from `_bands.py`
- Band centers **fixed** based on published literature
- Only amplitudes optimized via NNLS (Non-Negative Least Squares)

#### C. Constrained Component Fitting
- Fits predefined chemical component spectra from `_constants.py`
- Components selected based on dataset domain (e.g., milk → water, fat, protein, lactose)
- Regularization parameter: 0.01

#### D. Optimized Component Selection
- Greedy search + swap refinement for best component combination
- Two-phase: forward selection then local optimization via swaps

#### E-H. Combined Approaches
4 hybrid methods combining component fitting with band fitting on residuals:
- Components + Pure Bands
- Optimized Components + Pure Bands
- Components + Real Bands
- Optimized Components + Real Bands

### Variance Modeling

Two variance fitting methods:
1. **Operator-based**: noise, scatter, baseline operators
2. **PCA-based**: principal component score distributions

### Synthetic Generation Pipeline

```
For each dataset:
  For each of 8 fitting approaches:
    For each of 2 variance methods:
      1. Fit mean spectrum (or spectra)
      2. Characterize noise (high-freq from 2nd derivative)
      3. Characterize baseline variation (polynomial coefficients)
      4. Characterize sample-to-sample variation
      5. Generate synthetic spectra
      6. Evaluate quality (KS test, Wasserstein, discriminator)
```

### Quality Evaluation Metrics

- **R² coefficient**: Fitting quality (target > 0.95)
- **KS statistic**: PCA score distribution similarity
- **Wasserstein distance**: Distribution distance
- **Discriminator accuracy**: RandomForest classifier (real vs. synthetic)
  - Lower accuracy = harder to distinguish = better synthetic quality

### External Libraries Used

| Library | Purpose |
|---------|---------|
| NumPy | Numerical operations |
| Pandas | Data manipulation |
| Matplotlib | Visualization |
| SciPy | Savitzky-Golay filtering, NNLS, KS test, Wasserstein |
| scikit-learn | PCA, RandomForest discriminator |

### nirs4all Usage in Notebook

**Used:**
- `nirs4all.data.DatasetConfigs` for loading datasets
- `nirs4all.synthesis._constants.py` for predefined components
- `nirs4all.synthesis._bands.py` for band assignments

**NOT Used:**
- `SyntheticNIRSGenerator` class
- `SyntheticDatasetBuilder` fluent interface
- Physical operators (Temperature, Particle Size, EMSC, etc.)
- Instrument simulation
- Environmental effects
- Validation/benchmarking utilities

---

## 2. What is Possible with nirs4all API

### Core Generation Engine

The `SyntheticNIRSGenerator` implements physics-based spectral generation:

```python
from nirs4all.synthesis import SyntheticNIRSGenerator

generator = SyntheticNIRSGenerator(
    wavelength_range=(1000, 2500),
    wavelength_step=2,
    complexity="realistic",
    random_state=42
)
X, y, metadata = generator.generate(n_samples=1000)
```

**Physics Model (Beer-Lambert):**
```
A_i(λ) = L_i × Σ_k c_ik × ε_k(λ) + baseline_i(λ) + scatter_i(λ) + noise_i(λ)
```

### Fluent Builder Pattern

```python
from nirs4all.synthesis import SyntheticDatasetBuilder

dataset = (
    SyntheticDatasetBuilder(n_samples=1000)
    .with_features(complexity="realistic")
    .with_targets(distribution="lognormal", range=(0, 100))
    .with_components(["water", "protein", "lipid", "starch"])
    .with_instrument("bruker_matrix_f")
    .with_measurement_mode("reflectance")
    .with_batch_effects(n_batches=3)
    .build()
)
```

### Predefined Components (126+)

Organized by category with literature-based band positions:

| Category | Examples |
|----------|----------|
| Water & Moisture | water, moisture |
| Proteins (12) | protein, casein, gluten, albumin, collagen, keratin |
| Lipids (20) | lipid, oleic_acid, linoleic_acid, fish_oil, palm_oil |
| Carbohydrates (15) | starch, glucose, cellulose, lactose, pectin |
| Alcohols (9) | ethanol, methanol, glycerol |
| Organic Acids (12) | acetic_acid, citric_acid, lactic_acid |
| Pharmaceuticals (10) | caffeine, aspirin, paracetamol |
| Polymers (10) | polyethylene, polystyrene, PVC |
| Soil Minerals (8) | carbonite, gypsum, kaolinite |

### Physical-Based Operators

#### Temperature Effects
```python
from nirs4all.operators.augmentation import TemperatureAugmenter

temp_aug = TemperatureAugmenter(
    temperature_delta=10.0,           # °C change
    reference_temperature=25.0,
    enable_shift=True,                # Peak position shifts
    enable_intensity=True,            # Amplitude changes
    enable_broadening=True,           # Band width changes
    region_specific=True              # Per-region physics
)
```

**Region-specific effects:**
| Region | Wavelength | Shift/°C | Intensity/°C |
|--------|-----------|----------|--------------|
| O-H 1st overtone | 1400-1520 nm | -0.30 nm | -0.2% |
| O-H combination | 1900-2000 nm | -0.40 nm | -0.3% |
| C-H 1st overtone | 1650-1780 nm | -0.05 nm | -0.05% |
| N-H 1st overtone | 1490-1560 nm | -0.20 nm | -0.15% |
| Free water | 1380-1420 nm | -0.10 nm | +0.3% |
| Bound water | 1440-1500 nm | -0.35 nm | -0.4% |

#### Scattering Effects
```python
from nirs4all.operators.augmentation import ParticleSizeAugmenter, EMSCDistortionAugmenter

# Particle size scattering
particle_aug = ParticleSizeAugmenter(
    mean_size_um=50.0,
    size_variation_um=15.0,
    wavelength_exponent=1.5,          # Rayleigh-Mie regime
    include_path_length=True
)

# EMSC-style distortion
emsc_aug = EMSCDistortionAugmenter(
    multiplicative_range=(0.9, 1.1),
    additive_range=(-0.05, 0.05),
    polynomial_order=2
)
```

#### Moisture Effects
```python
from nirs4all.operators.augmentation import MoistureAugmenter

moisture_aug = MoistureAugmenter(
    water_activity_delta=0.2,
    free_water_fraction=0.6,
    bound_water_shift=25.0            # nm shift between states
)
```

#### Edge/Detector Artifacts
```python
from nirs4all.operators.augmentation import (
    DetectorRollOffAugmenter,
    StrayLightAugmenter,
    EdgeCurvatureAugmenter,
    TruncatedPeakAugmenter
)

# Detector sensitivity curves
detector_aug = DetectorRollOffAugmenter(
    detector_model="ingaas_extended",
    effect_strength=1.0,
    noise_amplification=0.5
)

# Stray light peak truncation
stray_aug = StrayLightAugmenter(
    stray_light_fraction=0.001,
    edge_enhancement=2.0
)
```

### Instrument Simulation (20+ Archetypes)

```python
from nirs4all.synthesis import get_instrument_config

# Available instruments
instruments = [
    "bruker_matrix_f",       # FT-NIR benchtop
    "foss_nir_ds2500",       # Grating benchtop
    "si_ware_neospectra",    # MEMS handheld
    "thermo_antaris_ii",     # FT-NIR process
    "ocean_nirquest",        # Diode array
    # ... 15+ more
]

config = get_instrument_config("bruker_matrix_f")
```

### Measurement Modes

- **Transmittance**: Beer-Lambert direct transmission
- **Reflectance**: Kubelka-Munk diffuse reflectance
- **Transflectance**: Double-pass with mirror
- **ATR**: Attenuated total reflectance
- **Interactance**: Partial transmission/reflection

### Wavenumber-Based Band Placement

```python
from nirs4all.synthesis import (
    calculate_overtone_position,
    calculate_combination_band,
    apply_hydrogen_bonding_shift,
    FUNDAMENTAL_VIBRATIONS
)

# Calculate 2nd overtone position with anharmonicity
position_nm = calculate_overtone_position(
    fundamental_wavenumber=3400,      # O-H stretch
    overtone_order=2,
    anharmonicity=0.02
)
```

### Procedural Component Generation

```python
from nirs4all.synthesis import (
    ProceduralComponentGenerator,
    ProceduralComponentConfig
)

config = ProceduralComponentConfig(
    n_fundamental_bands=3,
    include_overtones=True,
    include_combinations=True,
    anharmonicity_factor=0.98,
    target_nir_coverage=True
)

generator = ProceduralComponentGenerator(random_state=42)
library = generator.generate_library(n_components=50, config=config)
```

### Domain-Aware Generation

```python
from nirs4all.synthesis import (
    get_domain_config,
    create_domain_aware_library
)

# 15+ domains available
domains = [
    "agriculture_grain", "agriculture_soil",
    "food_dairy", "food_meat", "food_beverage",
    "pharmaceutical_tablets", "pharmaceutical_powders",
    "petrochemical_fuels", "petrochemical_polymers",
    "textile", "environmental", "biomedical"
]

config = get_domain_config("food_dairy")
library = create_domain_aware_library("food_dairy")
```

### Validation & Benchmarking

```python
from nirs4all.synthesis import (
    compute_spectral_realism_scorecard,
    validate_against_benchmark,
    RealDataFitter
)

# Realism scoring
scores = compute_spectral_realism_scorecard(
    X_synthetic, X_real, wavelengths
)

# Real data fitting
fitter = RealDataFitter(X_real, wavelengths)
params = fitter.infer_all()
```

---

## 3. What is Missing in nirs4all API

### Critical Gaps

#### 3.1 No Spectral Decomposition/Fitting to Real Data

The notebook performs **spectral fitting** (decomposing real spectra into components):

```python
# Notebook approach
def fit_spectrum_bands(spectrum, wavelengths, max_bands=50):
    """Fit unconstrained Gaussians to spectrum"""
    bands = []
    residual = spectrum.copy()
    while r2 < target_r2 and n_bands < max_bands:
        # Find next band to add
        ...
    return bands, amplitudes
```

**nirs4all has:**
- `RealDataFitter` - but it infers **generator parameters**, not spectral decomposition
- `ComponentFitter` - exists but is under-documented and limited

**Missing:**
- Pure Gaussian band fitting (unconstrained)
- NNLS amplitude fitting with fixed band positions
- Iterative band refinement algorithms
- Greedy component selection with optimization

#### 3.2 No Variance/Noise Model Fitting

The notebook characterizes variance in real data:

```python
# Notebook approach
def characterize_noise(X):
    """Extract noise characteristics from real spectra"""
    # High-frequency noise from 2nd derivative
    d2 = np.diff(X, n=2, axis=1)
    noise_std = np.std(d2, axis=1)

    # Baseline variation via polynomial
    baseline_coeffs = np.polyfit(...)

    return noise_model
```

**nirs4all has:**
- Noise operators (`GaussianAdditiveNoise`, etc.) - but they don't **learn** from data
- `ScatteringConfig`, `NoiseModelConfig` - configuration, not inference

**Missing:**
- `NoiseModelFitter.from_real_data(X)`
- Baseline variation inference
- Sample-to-sample correlation modeling

#### 3.3 No Comparison Framework for Fitting Approaches

The notebook systematically compares 8 fitting methods:

```python
# Notebook evaluates
approaches = [
    "pure_bands", "real_bands",
    "components", "optimized_components",
    "components+pure_bands", "components+real_bands",
    "optimized+pure_bands", "optimized+real_bands"
]

for approach in approaches:
    result = fit_and_generate(approach, dataset)
    evaluate_quality(result)
```

**Missing in nirs4all:**
- Standardized fitting approach comparison framework
- Fitting quality metrics (R² of decomposition)
- Automatic approach selection

#### 3.4 Limited Component Amplitude Optimization

The notebook uses NNLS with constraints:

```python
# Notebook approach
from scipy.optimize import nnls

# Fit component amplitudes (non-negative)
amplitudes, residual = nnls(component_matrix.T, spectrum)
```

**nirs4all has:**
- Fixed concentration distributions (uniform, normal, lognormal)
- No fitting of concentrations to match real spectra

**Missing:**
- `fit_concentrations(X_real, component_library)` → concentration distributions
- Per-sample concentration estimation

#### 3.5 No Preprocessing Type Detection

The notebook auto-detects preprocessing:

```python
# Notebook approach
def detect_preprocessing(X):
    if np.all(X > 0) and np.max(X) < 5:
        return "absorbance"
    elif np.any(X < 0):
        return "derivative"
    else:
        return "reflectance"
```

**nirs4all has:**
- `PreprocessingInference` class exists in `fitter.py`
- But limited documentation and integration

#### 3.6 No Discriminator-Based Evaluation

The notebook trains a classifier to evaluate synthetic quality:

```python
# Notebook approach
from sklearn.ensemble import RandomForestClassifier

def evaluate_synthetic_quality(X_real, X_synthetic):
    X = np.vstack([X_real, X_synthetic])
    y = np.array([0]*len(X_real) + [1]*len(X_synthetic))

    clf = RandomForestClassifier()
    accuracy = cross_val_score(clf, X, y).mean()

    # Lower accuracy = better synthetic quality
    return 1 - accuracy
```

**nirs4all has:**
- `compute_spectral_realism_scorecard` - but doesn't include discriminator

**Missing:**
- `adversarial_validation_score(X_real, X_synthetic)` built-in metric

---

## 4. What nirs4all Can Do That Is Not in the Notebook

### 4.1 Physics-Based Environmental Effects

The notebook generates noise/variance statistically. nirs4all can model **physical causes**:

```python
# Temperature variation (physically-motivated)
temp_aug = TemperatureAugmenter(
    temperature_range=(-5, 35),       # Lab temperature variation
    region_specific=True              # O-H, C-H, N-H respond differently
)

# Moisture/water activity variation
moisture_aug = MoistureAugmenter(
    water_activity_range=(0.3, 0.9),
    free_water_fraction=0.6
)
```

**Benefits:**
- Physically interpretable variation
- Correct wavelength-dependent effects
- Literature-validated shift/intensity coefficients

### 4.2 Instrument-Specific Simulation

The notebook doesn't consider instrument effects. nirs4all simulates:

```python
# Detector-specific response curves
detector_aug = DetectorRollOffAugmenter(detector_model="ingaas_extended")

# Instrument noise characteristics
generator = SyntheticNIRSGenerator(
    instrument_archetype="bruker_matrix_f",
    measurement_mode="reflectance"
)
```

**Available instrument effects:**
- Detector sensitivity roll-off at edges
- Stray light peak truncation
- Multi-sensor stitching artifacts
- Wavelength calibration drift

### 4.3 Scattering Physics

The notebook uses simple polynomial baselines. nirs4all models scattering physics:

```python
# Particle size-dependent scattering
particle_aug = ParticleSizeAugmenter(
    size_range_um=(20, 100),
    wavelength_exponent=1.5,          # Rayleigh-Mie regime
    include_path_length=True
)

# Kubelka-Munk diffuse reflectance
generator = SyntheticNIRSGenerator(
    measurement_mode="reflectance",
    scattering_model="kubelka_munk"
)
```

### 4.4 Voigt Peak Profiles

The notebook uses simple Gaussians. nirs4all uses Voigt profiles:

```python
# Voigt = Gaussian (thermal) * Lorentzian (pressure)
from nirs4all.synthesis import NIRBand

band = NIRBand(
    center=1450,
    sigma=15.0,      # Gaussian width (thermal broadening)
    gamma=3.0,       # Lorentzian width (pressure broadening)
    amplitude=0.5
)
```

**Benefits:**
- More accurate peak shapes
- Realistic peak tails
- Physically meaningful width parameters

### 4.5 Wavenumber-Based Overtone/Combination Bands

The notebook uses ad-hoc band positions. nirs4all calculates from fundamentals:

```python
from nirs4all.synthesis import (
    calculate_overtone_position,
    calculate_combination_band,
    FUNDAMENTAL_VIBRATIONS
)

# O-H stretch fundamental at 3400 cm⁻¹
# Calculate 1st overtone with anharmonicity
overtone_1_nm = calculate_overtone_position(
    fundamental_wavenumber=3400,
    overtone_order=1,
    anharmonicity=0.02
)  # → ~1450 nm

# Calculate combination band (stretch + bend)
combination_nm = calculate_combination_band(
    wavenumber1=3400,  # O-H stretch
    wavenumber2=1650,  # O-H bend
    anharmonicity=0.015
)  # → ~1940 nm
```

### 4.6 Domain-Aware Component Selection

The notebook manually assigns components per dataset. nirs4all has domain knowledge:

```python
from nirs4all.synthesis import get_domain_config

# Automatic component selection for dairy
config = get_domain_config("food_dairy")
print(config.typical_components)
# ['water', 'lipid', 'protein', 'lactose', 'casein']

print(config.concentration_ranges)
# {'water': (0.85, 0.90), 'lipid': (0.01, 0.05), ...}
```

### 4.7 Large-Scale Generation with GPU Acceleration

The notebook generates hundreds of samples. nirs4all scales to millions:

```python
from nirs4all.synthesis import SyntheticNIRSGenerator

# Generate 100k samples efficiently
generator = SyntheticNIRSGenerator(use_gpu=True)
X, y = generator.generate(n_samples=100000)
# Time: ~5 seconds (complex) vs. hours with notebook approach
```

### 4.8 Multi-Source Generation

The notebook handles single-source spectra. nirs4all generates correlated multi-source:

```python
dataset = (
    SyntheticDatasetBuilder(n_samples=1000)
    .with_multi_source({
        "NIR": {"wavelength_range": (1000, 2500)},
        "markers": {"n_features": 10}
    })
    .with_correlated_sources(correlation=0.7)
    .build()
)
```

### 4.9 Batch Effects for Domain Adaptation

```python
dataset = (
    SyntheticDatasetBuilder(n_samples=1000)
    .with_batch_effects(
        n_batches=5,
        batch_shift_range=(-0.02, 0.02),
        batch_scale_range=(0.95, 1.05)
    )
    .build()
)
```

### 4.10 Complex Target Relationships

The notebook uses simple linear targets. nirs4all can model:

```python
dataset = (
    SyntheticDatasetBuilder(n_samples=1000)
    .with_nonlinear_targets(
        interactions="polynomial",
        hidden_factors=2,              # Latent variables
        noise_ratio=0.1
    )
    .with_target_complexity(
        confounders=True,
        temporal_drift=True
    )
    .build()
)
```

---

## 5. Target Generation with nirs4all API

### Current Target Generation Approaches

#### 5.1 Component-Based Targets (Implemented)

```python
# Target from component concentration
dataset = (
    SyntheticDatasetBuilder(n_samples=1000)
    .with_targets(
        target_component="protein",     # Use protein concentration as y
        distribution="lognormal",
        range=(5, 25)                   # % protein content
    )
    .build()
)
```

#### 5.2 Multi-Target Regression (Implemented)

```python
# Multiple targets from different components
dataset = (
    SyntheticDatasetBuilder(n_samples=1000)
    .with_targets(
        target_components=["protein", "lipid", "moisture"],
        n_targets=3
    )
    .build()
)
```

#### 5.3 Classification Targets (Implemented)

```python
# Discrete classes
dataset = nirs4all.generate.classification(
    n_samples=500,
    n_classes=3,
    class_separation=2.0
)
```

### Advanced Target Generation (Partially Implemented)

#### 5.4 Polynomial Interactions

```python
# y = β₀ + β₁c₁ + β₂c₂ + β₃c₁c₂ + β₄c₁² + ε
dataset = (
    SyntheticDatasetBuilder(n_samples=1000)
    .with_nonlinear_targets(interactions="polynomial")
    .build()
)
```

#### 5.5 Hidden Factors (Latent Variables)

```python
# Target depends on factors NOT in the spectra
dataset = (
    SyntheticDatasetBuilder(n_samples=1000)
    .with_nonlinear_targets(
        hidden_factors=2,
        hidden_factor_strength=0.3
    )
    .build()
)
```

### What Could Be Added for Realistic Target Generation

#### 5.6 Property-Based Targets (Not Just Concentration)

Real NIRS predicts **properties**, not just concentrations:
- Grain hardness (from protein + starch interaction)
- Oil oxidation state (from unsaturated lipid degradation)
- Tissue hydration (from free/bound water ratio)

**Proposed API:**
```python
# Physical property as target
dataset = (
    SyntheticDatasetBuilder(n_samples=1000)
    .with_property_target(
        property="hardness",
        model="protein_starch_interaction",
        base_components=["protein", "starch"]
    )
    .build()
)
```

#### 5.7 Temporal/Aging Effects

```python
# Target changes with sample age
dataset = (
    SyntheticDatasetBuilder(n_samples=1000)
    .with_temporal_target(
        degradation_component="unsaturated_fat",
        degradation_rate=0.01,         # per day
        age_range=(0, 90)              # days
    )
    .build()
)
```

#### 5.8 Environmental-Dependent Targets

```python
# Target affected by measurement conditions
dataset = (
    SyntheticDatasetBuilder(n_samples=1000)
    .with_environment_dependent_target(
        base_target="moisture",
        temperature_sensitivity=-0.1,   # %/°C
        humidity_sensitivity=0.05
    )
    .build()
)
```

#### 5.9 Multi-Population Targets

```python
# Different subpopulations with different relationships
dataset = (
    SyntheticDatasetBuilder(n_samples=1000)
    .with_multi_population_target(
        n_populations=3,
        population_slopes=[1.0, 1.5, 0.8],
        population_intercepts=[0, 5, -3]
    )
    .build()
)
```

---

## 6. High-Level Plan for New Notebook

### Objective

Create a new notebook `nirs4all_synthetic_study.ipynb` that:
1. Uses **only** nirs4all API for spectral generation
2. Demonstrates physical-based operators
3. Fits generator parameters to real datasets
4. Generates synthetic data with realistic variance
5. Evaluates quality using nirs4all validation tools

### Notebook Structure

#### Part 1: Real Dataset Analysis (30%)

```python
# Cell 1.1: Load real datasets
import nirs4all
from nirs4all.data import DatasetConfigs

datasets = {
    "milk": DatasetConfigs("/path/to/milk"),
    "grain": DatasetConfigs("/path/to/grain"),
    # ...
}

# Cell 1.2: Spectral characterization
from nirs4all.synthesis import (
    compute_spectral_realism_scorecard,
    compute_derivative_statistics,
    compute_peak_density,
    compute_snr
)

for name, ds in datasets.items():
    stats = {
        "derivative": compute_derivative_statistics(ds.X, ds.wavelengths_nm),
        "peaks": compute_peak_density(ds.X, ds.wavelengths_nm),
        "snr": compute_snr(ds.X),
    }

# Cell 1.3: Domain inference
from nirs4all.synthesis import RealDataFitter

fitter = RealDataFitter(ds.X, ds.wavelengths_nm)
inferred_domain = fitter.infer_domain()
inferred_instrument = fitter.infer_instrument_type()
inferred_components = fitter.infer_components()
```

#### Part 2: Component-Based Generation (25%)

```python
# Cell 2.1: Domain-aware library
from nirs4all.synthesis import create_domain_aware_library

library = create_domain_aware_library("food_dairy")

# Cell 2.2: Generate with domain knowledge
generator = SyntheticNIRSGenerator(
    component_library=library,
    wavelength_range=(1100, 2500),
    complexity="realistic"
)
X_synthetic, y_synthetic = generator.generate(n_samples=1000)

# Cell 2.3: Compare component decomposition
# Use nirs4all's component fitter on real data
from nirs4all.synthesis import ComponentFitter

comp_fitter = ComponentFitter(ds.X, ds.wavelengths_nm, library)
component_contributions = comp_fitter.fit()
```

#### Part 3: Physical Effects Layer (25%)

```python
# Cell 3.1: Temperature variation
from nirs4all.operators.augmentation import TemperatureAugmenter

temp_aug = TemperatureAugmenter(
    temperature_range=(-5, 35),
    region_specific=True
)
X_temp_varied = temp_aug.transform(X_synthetic, ds.wavelengths_nm)

# Cell 3.2: Particle size / scattering
from nirs4all.operators.augmentation import (
    ParticleSizeAugmenter,
    EMSCDistortionAugmenter
)

particle_aug = ParticleSizeAugmenter(size_range_um=(20, 100))
X_scattered = particle_aug.transform(X_temp_varied, ds.wavelengths_nm)

# Cell 3.3: Detector effects
from nirs4all.operators.augmentation import (
    DetectorRollOffAugmenter,
    StrayLightAugmenter
)

detector_aug = DetectorRollOffAugmenter(detector_model="ingaas_extended")
X_with_detector = detector_aug.transform(X_scattered, ds.wavelengths_nm)

# Cell 3.4: Combined pipeline
from nirs4all.operators.augmentation import Compose

physical_augmentations = Compose([
    TemperatureAugmenter(temperature_range=(-5, 35)),
    MoistureAugmenter(water_activity_range=(0.3, 0.9)),
    ParticleSizeAugmenter(size_range_um=(20, 100)),
    DetectorRollOffAugmenter(detector_model="ingaas_extended"),
])

X_realistic = physical_augmentations.transform(X_synthetic, ds.wavelengths_nm)
```

#### Part 4: Instrument Simulation (10%)

```python
# Cell 4.1: Different instrument archetypes
from nirs4all.synthesis import get_instrument_config

instruments = [
    "bruker_matrix_f",      # FT-NIR
    "foss_nir_ds2500",      # Grating
    "si_ware_neospectra",   # MEMS handheld
]

for instr in instruments:
    config = get_instrument_config(instr)
    generator = SyntheticNIRSGenerator(instrument_archetype=instr)
    X_instr, _ = generator.generate(n_samples=500)

    # Plot and compare spectral characteristics
```

#### Part 5: Validation & Quality Assessment (10%)

```python
# Cell 5.1: Realism scorecard
from nirs4all.synthesis import compute_spectral_realism_scorecard

scores = compute_spectral_realism_scorecard(
    X_synthetic=X_realistic,
    X_real=ds.X,
    wavelengths=ds.wavelengths_nm
)

print(f"Correlation length: {scores['correlation_length']:.3f}")
print(f"Derivative stats: {scores['derivative_stats']:.3f}")
print(f"Distribution overlap: {scores['distribution_overlap']:.3f}")

# Cell 5.2: Adversarial validation (to be added)
# This needs to be implemented - see Section 3.6
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

X_combined = np.vstack([ds.X, X_realistic])
y_binary = np.array([0]*len(ds.X) + [1]*len(X_realistic))
clf = RandomForestClassifier(n_estimators=100)
accuracy = cross_val_score(clf, X_combined, y_binary, cv=5).mean()
print(f"Discriminator accuracy: {accuracy:.3f} (lower is better)")

# Cell 5.3: PCA comparison
from sklearn.decomposition import PCA

pca = PCA(n_components=5)
scores_real = pca.fit_transform(ds.X)
scores_synthetic = pca.transform(X_realistic)

# KS test per component
from scipy.stats import ks_2samp
for i in range(5):
    stat, pval = ks_2samp(scores_real[:, i], scores_synthetic[:, i])
    print(f"PC{i+1}: KS={stat:.3f}, p={pval:.3f}")
```

#### Part 6: Recommendations & Summary

```python
# Cell 6.1: Generate quality comparison table
# Compare: base synthetic vs. with physical effects vs. notebook approach

results_df = pd.DataFrame({
    "Method": ["Base", "+Temperature", "+Scattering", "+Detector", "All Physical"],
    "KS_PC1": [...],
    "Discriminator_Acc": [...],
    "Wasserstein": [...]
})

# Cell 6.2: Best practices recommendations
# - Domain-aware component selection
# - Physical effects improve realism
# - Instrument simulation for cross-device robustness
```

### Implementation Priorities

1. **High Priority (Required for notebook):**
   - Verify all operators work with wavelength arrays
   - Test `RealDataFitter` on actual datasets
   - Implement discriminator evaluation wrapper

2. **Medium Priority (Improves notebook):**
   - Add `ComponentFitter.fit_amplitudes()` method
   - Add `NoiseModelFitter.from_data()` method
   - Better documentation for `validation.py`

3. **Low Priority (Nice to have):**
   - Pure Gaussian band fitting algorithm
   - NNLS amplitude optimization
   - Comparison framework for fitting approaches

---

## Summary

| Aspect | generator_final.ipynb | nirs4all API |
|--------|----------------------|--------------|
| **Spectral decomposition** | ✅ Multiple fitting methods | ❌ Limited fitting tools |
| **Variance modeling** | ✅ Noise/baseline from data | ❌ No variance inference |
| **Physical effects** | ❌ Not used | ✅ Temperature, scattering, moisture |
| **Instrument simulation** | ❌ Not used | ✅ 20+ archetypes |
| **Component library** | ✅ Uses _constants.py | ✅ 126+ components |
| **Band assignments** | ✅ Uses _bands.py | ✅ Literature-based |
| **Quality evaluation** | ✅ Discriminator + KS | ⚠️ Scorecard only |
| **Target generation** | ❌ Not focus | ✅ Multiple distributions |
| **Scalability** | ❌ Hundreds of samples | ✅ GPU for millions |

**Recommendation:** The new notebook should leverage nirs4all's physical operators while adding the missing spectral fitting capabilities. The combination of component-based generation plus physical augmentation should produce more realistic synthetic data than either approach alone.
