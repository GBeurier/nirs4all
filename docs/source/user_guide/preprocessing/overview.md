# Preprocessing Overview

This guide covers spectral preprocessing techniques for NIRS data. Preprocessing is critical for NIRS analysis as it removes artifacts, reduces noise, and enhances spectral features relevant to the target property.

## Why Preprocessing Matters

Near-infrared spectroscopy (NIRS) data requires specialized preprocessing to:
- Remove scatter effects from particle size, surface roughness, or path length variations (MSC, SNV)
- Eliminate baseline drift and offset (detrending, derivatives)
- Reduce noise while preserving spectral features (Savitzky–Golay, Gaussian smoothing)
- Normalize intensity variations across instruments or sessions
- Enhance absorption bands through derivatives or wavelet transforms

## Common NIRS Preprocessing Workflow

A typical preprocessing workflow follows this order:

```
1. SCATTER CORRECTION (MSC or SNV)
   ↓
2. SMOOTHING (Savitzky–Golay or Gaussian)
   ↓
3. BASELINE CORRECTION (detrending or derivatives)
   ↓
4. SCALING (StandardScaler or normalization)
```

:::{tip}
The order matters! Apply scatter correction before derivatives, and smoothing before derivatives to reduce noise amplification.
:::

## Quick Example

```python
from nirs4all.operators.transforms import SNV, SavitzkyGolay
from sklearn.preprocessing import StandardScaler

pipeline = [
    SNV(),                                    # Scatter correction
    SavitzkyGolay(window_length=15, deriv=1), # First derivative with smoothing
    StandardScaler(),                          # Feature scaling
    {"model": PLSRegression(n_components=10)}
]
```

## Available Operators

### NIRS4ALL Transformers

All NIRS4ALL transformers follow the sklearn TransformerMixin pattern and can be used in pipelines.

#### Scatter Correction

| Operator | Description | Typical Use |
|----------|-------------|-------------|
| `SNV` (StandardNormalVariate) | Row-wise normalization: subtract mean, divide by std | General scatter correction |
| `MSC` (MultiplicativeScatterCorrection) | Reference-based correction for multiplicative effects | When reference spectrum available |
| `RSNV` (RobustStandardNormalVariate) | Outlier-resistant SNV variant | Noisy/heterogeneous samples |
| `LSNV` (LocalStandardNormalVariate) | Window-based local SNV | Heterogeneous materials |

#### Derivatives

| Operator | Description | Typical Use |
|----------|-------------|-------------|
| `FirstDerivative` | Numerical 1st derivative along wavelengths | Remove baseline offset |
| `SecondDerivative` | Numerical 2nd derivative along wavelengths | Resolve overlapping peaks |
| `SavitzkyGolay` | Smoothing + optional derivative | Most common derivative method |

:::{warning}
**Axis convention**: `FirstDerivative` and `SecondDerivative` operate along **axis=1** (wavelengths), which is correct for NIRS. The legacy `Derivate` class uses axis=0 (samples) and should be avoided.
:::

#### Smoothing

| Operator | Description | Parameters |
|----------|-------------|------------|
| `SavitzkyGolay` | Polynomial smoothing filter | `window_length=11, polyorder=3, deriv=0` |
| `Gaussian` | Gaussian filter | `sigma=1, order=2` |

#### Baseline & Normalization

| Operator | Description | Typical Use |
|----------|-------------|-------------|
| `Detrend` | Remove linear/polynomial baseline | Baseline slope removal |
| `Baseline` | Subtract per-feature mean | Simple offset removal |
| `Normalize` | Scale to range or L2 norm | Intensity standardization |
| `SimpleScale` | Min-max scaling per feature | Bounded [0,1] range |
| `LogTransform` | Logarithmic transformation | Convert reflectance to pseudo-absorbance |

#### Wavelets & Advanced

| Operator | Description | Typical Use |
|----------|-------------|-------------|
| `Wavelet` | Discrete wavelet transform | Denoising, multi-resolution features |
| `Haar` | Haar wavelet (shortcut) | Quick wavelet denoising |
| `CropTransformer` | Select wavelength range | Region selection |
| `ResampleTransformer` | Resample to fixed size | Standardize different instruments |

### Sklearn-Compatible Transformers

These standard sklearn transformers work seamlessly in NIRS4ALL pipelines:

| Transformer | Purpose | NIRS Context |
|-------------|---------|--------------|
| `StandardScaler` | Zero mean, unit variance | Use after scatter correction |
| `RobustScaler` | Median/IQR scaling | Robust to outliers |
| `MinMaxScaler` | Scale to [0, 1] | Bounded models |
| `PCA` | Dimensionality reduction | NIRS spectra are highly collinear |
| `FunctionTransformer` | Wrap custom functions | Quick custom preprocessing |

## Recommended Preprocessing by Model Type

### Classical ML (sklearn)

| Model | Recommended Preprocessing | Avoid |
|-------|--------------------------|-------|
| **PLS** | Mean-centering, SNV/MSC, 1st derivative | Over-aggressive 2nd derivative |
| **PCR** | Center + autoscale, SG smoothing | Raw uncorrected scatter |
| **SVM/SVR** | Standardization, SNV/MSC, SG + 1st deriv | No scaling |
| **Random Forest** | SG smoothing, SNV/MSC | Per-feature standardization |
| **k-NN** | Per-feature scaling or SNV, band selection | Raw unscaled spectra |

### Neural Networks

| Model | Recommended Preprocessing | Avoid |
|-------|--------------------------|-------|
| **MLP** | Standardization, mean-centering/SNV, SG smoothing | Raw unscaled spectra |
| **1D CNN** | Input scaling, SNV/MSC, 1st derivative optional | Over-smoothed spectra |
| **Transformers** | Standardization, SNV/MSC, patch/bin tokens | Very long sequences without reduction |

## Preprocessing Order Rules

### Correct Order

```python
# ✅ CORRECT: scatter → smooth → derivative
pipeline = [
    MSC(),                           # 1. Scatter correction first
    SavitzkyGolay(window_length=15), # 2. Smoothing
    FirstDerivative(),               # 3. Derivative last
]

# ✅ CORRECT: SavGol with built-in derivative
pipeline = [
    SNV(),
    SavitzkyGolay(window_length=15, deriv=1),  # Smoothing + derivative
]

# ✅ CORRECT: Detrend before scatter correction
pipeline = [
    Detrend(),
    SNV(),
    SavitzkyGolay(window_length=15),
]
```

### Incorrect Orders (Avoid)

```python
# ❌ WRONG: Don't combine two scatter corrections
pipeline = [SNV(), MSC()]  # Redundant

# ❌ WRONG: Don't smooth after SavGol derivative
pipeline = [
    SavitzkyGolay(deriv=1),
    SavitzkyGolay(),  # SG already includes smoothing
]

# ❌ WRONG: Derivative before scatter correction
pipeline = [
    FirstDerivative(),  # Amplifies scatter artifacts
    MSC(),
]
```

## Task-Specific Preprocessing

### Protein/Nitrogen Prediction (N-H bonds, 2000-2200nm)

```python
pipeline = [
    MSC(),
    SavitzkyGolay(window_length=17, deriv=1),  # Critical for protein
    StandardScaler(),
]
```

### Moisture Prediction (O-H bonds, 1400-1500nm, 1900-2000nm)

```python
pipeline = [
    MSC(),
    SavitzkyGolay(window_length=15),  # Smooth, avoid derivatives in high absorption
    RSNV(),  # Local scatter for heterogeneous moisture
]
```

### Fat/Oil Prediction (C-H bonds, 1700-1800nm, 2300-2400nm)

```python
pipeline = [
    MSC(),  # Important for fat scatter
    SNV(),
    SavitzkyGolay(window_length=21, deriv=1),
]
```

## Multi-Layer Preprocessing for Deep Learning

When training neural networks, multiple preprocessing "views" can be stacked as channels:

```python
# Create multiple preprocessing branches
pipeline = [
    {"branch": [
        [SNV(), SavitzkyGolay()],           # Channel 1: SNV + smooth
        [MSC(), FirstDerivative()],          # Channel 2: MSC + 1st deriv
        [SNV(), SecondDerivative()],         # Channel 3: SNV + 2nd deriv
        [Wavelet('db4')],                    # Channel 4: Wavelet features
    ]},
    {"merge": "features"},  # Stack as multi-channel input
    {"model": CNN1D()}
]
```

:::{note}
After concatenating multi-layer preprocessing, apply **per-channel** scaling, not a single global scaler.
:::

## SciPy Functions Reference

These SciPy functions are used internally by NIRS4ALL transformers:

| Function | Purpose | Used By |
|----------|---------|---------|
| `scipy.signal.savgol_filter` | Smoothing + derivatives | `SavitzkyGolay` |
| `scipy.signal.detrend` | Linear trend removal | `Detrend` |
| `scipy.ndimage.gaussian_filter1d` | Gaussian smoothing | `Gaussian` |
| `scipy.interpolate.interp1d` | 1D interpolation | `ResampleTransformer` |
| `scipy.interpolate.UnivariateSpline` | Spline smoothing | Spline augmenters |

## Best Practices

:::{tip}
**Golden Rules for NIRS Preprocessing**

1. **Always fit preprocessing inside CV** to avoid data leakage
2. **Start simple**: SNV + SavGol(deriv=1) is a strong baseline
3. **Don't over-preprocess**: More steps ≠ better results
4. **Validate your choices**: Compare RMSE across preprocessing variants
5. **Match preprocessing to model**: Trees don't need scaling; SVMs do
:::

## See Also

- {doc}`cheatsheet` - Quick reference by model type
- {doc}`handbook` - In-depth theory and advanced techniques
- {doc}`/reference/operator_catalog` - Complete operator reference
- {doc}`snv` - Detailed SNV documentation
