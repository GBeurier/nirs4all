# Physical Signal-Chain Reconstruction Workflow

This document describes the physically realistic signal-chain reconstruction and variance modeling workflow for NIR spectra in the `nirs4all.data.synthetic.reconstruction` module.

## Overview

The reconstruction workflow enables:
1. **Reconstruction** of spectra using a physical forward model
2. **Learning** distributions of physical parameters for variance modeling
3. **Generation** of realistic synthetic datasets matching real data statistics

## Workflow Steps

### Step 1: Dataset Configuration

Detect or specify dataset properties:

```python
from nirs4all.data.synthetic.reconstruction import DatasetConfig

# Auto-detect from data
config = DatasetConfig.from_data(X, wavelengths, name="my_dataset")

# Or specify manually
config = DatasetConfig(
    wavelengths=wavelengths,
    signal_type="absorbance",  # or "reflectance"
    preprocessing="first_derivative",  # or "none", "snv", etc.
    domain="food_dairy",  # for component selection
)
```

### Step 2: Global Calibration

Calibrate global instrument parameters using prototype spectra:

```python
from nirs4all.data.synthetic.reconstruction import (
    ForwardChain,
    PrototypeSelector,
    GlobalCalibrator,
)

# Create forward chain with environmental model
chain = ForwardChain.create(
    canonical_grid=np.linspace(900, 2600, 340),
    target_grid=wavelengths,
    component_names=["water", "protein", "lipid"],
    domain="absorbance",
    preprocessing_type="none",
    include_environmental=True,  # Enable environmental effects
)

# Select prototypes
selector = PrototypeSelector(n_prototypes=5)
prototypes, indices = selector.select(X)

# Calibrate
calibrator = GlobalCalibrator()
calib_result = calibrator.calibrate(prototypes, chain)

print(f"Wavelength shift: {calib_result.wl_shift:.2f} nm")
print(f"ILS sigma: {calib_result.ils_sigma:.2f} nm")
```

### Step 3: Per-Sample Inversion

Fit physical parameters for each sample using variable projection:

```python
from nirs4all.data.synthetic.reconstruction import (
    VariableProjectionSolver,
    MultiscaleSchedule,
)

solver = VariableProjectionSolver(
    fit_environmental=True,  # Fit temperature, water activity, scattering
)
schedule = MultiscaleSchedule.quick()  # or MultiscaleSchedule() for thorough

# Fit single sample
result = solver.fit(X[0], chain, schedule)
print(f"R²: {result.r_squared:.4f}")
print(f"Concentrations: {result.concentrations}")
print(f"Temperature delta: {result.temperature_delta:.2f}°C")
print(f"Scattering amplitude: {result.scattering_amplitude:.4f}")

# Fit batch
results = solver.fit_batch(X, chain, schedule)
```

### Step 4: Learn Parameter Distributions

Model variance in parameter space:

```python
from nirs4all.data.synthetic.reconstruction import (
    ParameterDistributionFitter,
    ParameterSampler,
)

# Collect parameters (including environmental)
params = {
    "concentrations": np.array([r.concentrations for r in results]),
    "baseline_coeffs": np.array([r.baseline_coeffs for r in results]),
    "path_lengths": np.array([r.path_length for r in results]),
    "wl_shifts": np.array([r.wl_shift_residual for r in results]),
    "temperature_deltas": np.array([r.temperature_delta for r in results]),
    "water_activities": np.array([r.water_activity for r in results]),
    "scattering_powers": np.array([r.scattering_power for r in results]),
    "scattering_amplitudes": np.array([r.scattering_amplitude for r in results]),
}

# Fit distributions
fitter = ParameterDistributionFitter(
    positive_params=["concentrations", "path_lengths", "scattering_amplitudes"],
    bounded_params={
        "wl_shifts": (-5.0, 5.0),
        "water_activities": (0.1, 0.9),
        "scattering_powers": (0.5, 3.0),
    },
)
dist_result = fitter.fit(params)

# Create sampler
sampler = ParameterSampler(dist_result, use_correlations=True)
```

### Step 5: Generate Synthetic Data

Sample parameters and run forward model:

```python
from nirs4all.data.synthetic.reconstruction import ReconstructionGenerator

generator = ReconstructionGenerator(
    noise_level=0.001,
    multiplicative_noise=0.01,
)

gen_result = generator.generate(
    n_samples=500,
    forward_chain=chain,
    sampler=sampler,
    random_state=42,
)

X_synthetic = gen_result.X
# Environmental parameters are also available:
print(f"Temperature range: [{gen_result.temperature_deltas.min():.1f}, {gen_result.temperature_deltas.max():.1f}]°C")
```

### Step 6: Validate

Compare synthetic vs real data:

```python
from nirs4all.data.synthetic.reconstruction import ReconstructionValidator

validator = ReconstructionValidator()
validation = validator.validate(results, X_real, X_synthetic)

print(validation.summary())
```

## Complete Pipeline

The `ReconstructionPipeline` class orchestrates all steps:

```python
from nirs4all.data.synthetic.reconstruction import (
    DatasetConfig,
    ReconstructionPipeline,
)

# Configure
config = DatasetConfig.from_data(X, wavelengths)
config.domain = "food_dairy"

# Create pipeline with environmental fitting
pipeline = ReconstructionPipeline(
    config=config,
    n_prototypes=5,
    fit_environmental=True,  # Fit temperature, water activity, scattering
    verbose=True,
)

# Fit (runs all steps)
result = pipeline.fit(X, max_samples=100)

# Generate more synthetic data
X_synthetic = pipeline.generate(n_samples=500, result=result)

# Access results
print(result.summary())
print(f"Validation score: {result.validation.overall_score}/100")
```

## Convenience Function

For quick usage:

```python
from nirs4all.data.synthetic.reconstruction import reconstruct_and_generate

X_synthetic, result = reconstruct_and_generate(
    X=X_real,
    wavelengths=wavelengths,
    domain="food_dairy",
    fit_environmental=True,
    verbose=True,
)
```

## Environmental Effects Model

The `EnvironmentalEffectsModel` captures physical effects that cause spectral variations:

```python
from nirs4all.data.synthetic.reconstruction import EnvironmentalEffectsModel

env_model = EnvironmentalEffectsModel(
    temperature_delta=5.0,      # °C deviation from 25°C reference
    water_activity=0.7,         # Free vs bound water ratio (0-1)
    scattering_power=1.5,       # Wavelength exponent (λ^-n)
    scattering_amplitude=0.02,  # Baseline magnitude
)

# Apply to absorption spectrum
modified = env_model.apply(absorption, wavelengths)
```

### Environmental Parameters

| Parameter | Bounds | Physical Meaning |
|-----------|--------|------------------|
| `temperature_delta` | -15 to +15°C | Temperature deviation from 25°C reference |
| `water_activity` | 0.1 to 0.9 | Free vs bound water ratio |
| `scattering_power` | 0.5 to 3.0 | Wavelength exponent (λ^-n) |
| `scattering_amplitude` | 0.0 to 0.2 | Scattering baseline magnitude |

## Module Architecture

```
nirs4all/data/synthetic/reconstruction/
├── __init__.py          # Module exports
├── forward.py           # Forward model components
│   ├── CanonicalForwardModel    # Physical model on canonical grid
│   ├── InstrumentModel          # Wavelength warp, ILS, gain/offset
│   ├── DomainTransform          # Absorbance/reflectance conversion
│   ├── PreprocessingOperator    # Match dataset preprocessing
│   └── ForwardChain             # Complete forward chain
├── environmental.py     # Environmental effects
│   ├── EnvironmentalEffectsModel    # Temperature, moisture, scattering
│   └── EnvironmentalParameterConfig # Parameter bounds and priors
├── calibration.py       # Global calibration
│   ├── PrototypeSelector        # Select representative spectra
│   ├── GlobalCalibrator         # Fit instrument parameters
│   └── CalibrationResult        # Calibration output
├── inversion.py         # Per-sample fitting
│   ├── VariableProjectionSolver # Nonlinear/linear separation
│   ├── MultiscaleSchedule       # Coarse-to-fine fitting
│   └── InversionResult          # Per-sample fit output
├── distributions.py     # Parameter distributions
│   ├── ParameterDistributionFitter  # Fit marginals + correlations
│   ├── ParameterSampler             # Sample with Gaussian copula
│   └── DistributionResult           # Distribution parameters
├── generator.py         # Synthetic generation
│   ├── ReconstructionGenerator  # Sample params + forward model
│   └── GenerationResult         # Generated data
├── validation.py        # Quality validation
│   ├── ReconstructionValidator  # Compare synthetic vs real
│   └── ValidationResult         # Validation metrics
└── pipeline.py          # End-to-end workflow
    ├── DatasetConfig            # Dataset configuration
    ├── ReconstructionPipeline   # Orchestrates all steps
    └── PipelineResult           # Full pipeline output
```

## Key Principles

### 1. Physical Interpretability

Every modeled term is physically interpretable:
- **Component concentrations**: Chemical constituents
- **Path length**: Sample thickness / optical depth
- **Baseline**: Instrumental drift
- **Continuum**: Smooth background absorption
- **Temperature**: Region-specific shifts and intensity changes
- **Water activity**: Free vs bound water affecting band positions
- **Scattering**: Wavelength-dependent baseline (λ^-n)

### 2. Canonical Grid

A single latent physical model on high-resolution canonical grid:
- Apply environmental effects (temperature, moisture, scattering)
- Apply dataset-specific instrument transforms
- Apply resampling to dataset grid
- Apply exact preprocessing (SG derivatives, SNV, etc.)

### 3. Variable Projection

Separate parameters into:
- **Linear** (concentrations, baseline): Solved via NNLS/QP
- **Nonlinear** (path_length, wl_shift, environmental): Optimized in outer loop

### 4. Multiscale Fitting

Avoid local minima by progressive refinement:
1. Fit coarse features on smoothed spectra
2. Progressively reduce smoothing
3. Final fit on full resolution

### 5. Variance in Parameter Space

Model variance as distributions of physical parameters, not directly on spectra:
- Log-normal for positive params (concentrations, path_length, scattering_amplitude)
- Gaussian for shifts and temperature
- Beta for bounded params (water_activity)
- Gaussian copula for correlations

## Configuration Options

### Forward Chain

| Parameter | Description | Default |
|-----------|-------------|---------|
| `canonical_resolution` | Resolution of canonical grid (nm) | 0.5 |
| `baseline_order` | Chebyshev baseline polynomial order | 5 |
| `continuum_order` | Continuum absorption polynomial order | 3 |
| `ils_sigma` | Instrument line shape width (nm) | 4.0 |
| `include_environmental` | Include environmental effects model | False |

### Calibration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_prototypes` | Number of prototype spectra | 5 |
| `wl_shift_bounds` | Wavelength shift bounds (nm) | (-10, 10) |
| `ils_sigma_bounds` | ILS sigma bounds (nm) | (2, 20) |

### Inversion

| Parameter | Description | Default |
|-----------|-------------|---------|
| `path_length_bounds` | Path length bounds | (0.5, 2.0) |
| `wl_shift_bounds` | Per-sample shift bounds (nm) | (-2, 2) |
| `baseline_smoothness_penalty` | Regularization strength | 1e-4 |
| `fit_environmental` | Fit environmental parameters | False |
| `temperature_bounds` | Temperature delta bounds (°C) | (-15, 15) |
| `water_activity_bounds` | Water activity bounds | (0.1, 0.9) |
| `scattering_power_bounds` | Scattering power bounds | (0.5, 3.0) |
| `scattering_amplitude_bounds` | Scattering amplitude bounds | (0.0, 0.2) |

### Generation

| Parameter | Description | Default |
|-----------|-------------|---------|
| `noise_level` | Additive noise std | 0.001 |
| `multiplicative_noise` | Multiplicative noise std | 0.01 |
| `noise_type` | Type ("additive", "multiplicative", "both") | "both" |

## Validation Metrics

- **Mean R²**: Average reconstruction quality
- **Residual autocorrelation**: Should be low (no systematic patterns)
- **Mean spectrum correlation**: Synthetic vs real mean
- **Discriminator accuracy**: How distinguishable (lower = better)
- **PCA score distributions**: KS test on PC scores

## References

- Burns, D. A., & Ciurczak, E. W. (2007). Handbook of Near-Infrared Analysis.
- Workman Jr, J., & Weyer, L. (2012). Practical Guide and Spectral Atlas for Interpretive Near-Infrared Spectroscopy.
- Segtnan, V. H., et al. (2001). Studies on the Structure of Water Using Two-Dimensional Near-Infrared Correlation Spectroscopy and Principal Component Analysis. Analytical Chemistry.
