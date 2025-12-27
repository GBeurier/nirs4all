# Preprocessing Handbook

In-depth guide to NIRS preprocessing methods with theory, best practices, and advanced techniques.

## Philosophy: Multi-Layer Preprocessing

Modern NIRS analysis benefits from providing multiple "views" of the spectral data. Each preprocessing layer should provide a **different view** of the chemical information:

- **Minimize redundancy** between layers
- **Maximize complementary information**
- **Order matters** in sequential preprocessing

## Preprocessing Categories

### 1. Scatter Correction

Scatter effects arise from particle size variations, surface roughness, and path length differences. These multiplicative effects must be corrected before most analyses.

#### Standard Normal Variate (SNV)

```python
from nirs4all.operators.transforms import SNV

# Row-wise normalization
snv = SNV(copy=True)
X_corrected = snv.fit_transform(X)
```

**How it works**: For each spectrum, subtract mean and divide by standard deviation.

$$X_{SNV}^{(i)} = \frac{X^{(i)} - \bar{X}^{(i)}}{\sigma^{(i)}}$$

**Pros**: Simple, no reference needed, fast
**Cons**: Sensitive to outliers, may distort relative peak heights

#### Multiplicative Scatter Correction (MSC)

```python
from nirs4all.operators.transforms import MSC

msc = MSC(scale=True, copy=True)
X_corrected = msc.fit_transform(X)
```

**How it works**: Regresses each spectrum against a reference (typically the mean spectrum).

**Pros**: Better preserves absolute intensities, reference-based
**Cons**: Requires representative reference, computationally heavier

#### Robust/Local Variants

| Variant | Use Case | Advantage |
|---------|----------|-----------|
| **RSNV** | Noisy data | Outlier-resistant (uses robust statistics) |
| **LSNV** | Heterogeneous samples | Window-based local correction |
| **EMSC** | Complex scatter | Extended MSC with polynomial terms |

### 2. Smoothing

Smoothing reduces high-frequency noise while preserving spectral features.

#### Savitzky-Golay Filter

The workhorse of NIRS preprocessing:

```python
from nirs4all.operators.transforms import SavitzkyGolay

# Smoothing only
sg_smooth = SavitzkyGolay(window_length=15, polyorder=3, deriv=0)

# First derivative with smoothing
sg_d1 = SavitzkyGolay(window_length=15, polyorder=3, deriv=1)

# Second derivative
sg_d2 = SavitzkyGolay(window_length=21, polyorder=3, deriv=2)
```

**Parameter Guidelines**:

| Parameter | Typical Range | Effect of Increase |
|-----------|---------------|-------------------|
| `window_length` | 11-25 | More smoothing, less noise |
| `polyorder` | 2-4 | Better peak preservation |
| `deriv` | 0-2 | Higher order derivatives |

:::{tip}
For derivatives, use longer windows to reduce noise amplification:
- 1st derivative: window 11-17
- 2nd derivative: window 21-31
:::

#### Gaussian Smoothing

```python
from nirs4all.operators.transforms import Gaussian

gauss = Gaussian(sigma=2, order=0)
X_smooth = gauss.fit_transform(X)
```

Simpler than SG, good for quick smoothing without derivatives.

### 3. Baseline Correction

#### Detrending

Removes linear or polynomial baselines:

```python
from nirs4all.operators.transforms import Detrend

# Linear detrend
detrend = Detrend(bp=0)

# Piecewise with breakpoints
detrend_pw = Detrend(bp=[100, 200])  # Breakpoints at indices
```

#### First and Second Derivatives

Derivatives inherently remove baseline components:

- **1st derivative**: Removes constant offset and linear slope
- **2nd derivative**: Removes linear and quadratic baselines

```python
from nirs4all.operators.transforms import FirstDerivative, SecondDerivative

d1 = FirstDerivative(delta=1.0, edge_order=2)
d2 = SecondDerivative(delta=1.0, edge_order=2)
```

### 4. Wavelets

Wavelet transforms provide multi-resolution analysis:

```python
from nirs4all.operators.transforms import Wavelet

# Common wavelets for NIRS
wavelet_db4 = Wavelet(wavelet='db4')    # Good for peaks
wavelet_sym5 = Wavelet(wavelet='sym5')  # Good for baselines
wavelet_haar = Wavelet(wavelet='haar')  # Sharp features
```

**When to use wavelets**:
- Denoising with thresholding
- Multi-resolution feature extraction
- Capturing both local and global features

### 5. Scaling

#### Feature-wise Scaling (per wavelength)

```python
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# Zero mean, unit variance
scaler = StandardScaler()

# Robust to outliers
scaler = RobustScaler()

# Bounded range
scaler = MinMaxScaler(feature_range=(0, 1))
```

#### Sample-wise Scaling

Already handled by SNV, but can also use:

```python
from nirs4all.operators.transforms import Normalize

# Vector normalization (L2 norm)
norm = Normalize(feature_range=(-1, 1))
```

## Preprocessing Order

### General Order

```
1. Convert to absorbance (if reflectance)
2. Detrend (if needed before scatter correction)
3. MSC/SNV (scatter correction)
4. Smoothing (SavGol, Gaussian)
5. Derivatives (if using)
6. Scaling (StandardScaler, RobustScaler)
```

### Common Correct Chains

```python
# Chain 1: Standard NIRS preprocessing
[MSC(), SavitzkyGolay(deriv=1), StandardScaler()]

# Chain 2: SNV-Detrend combo
[SNV(), Detrend()]  # Detrend AFTER SNV

# Chain 3: Smooth then derivative
[SavitzkyGolay(deriv=0), FirstDerivative()]

# Chain 4: Full pipeline
[Detrend(), MSC(), SavitzkyGolay(deriv=1), RobustScaler()]
```

### Order Violations to Avoid

```python
# ❌ Two scatter corrections
[SNV(), MSC()]  # Redundant!

# ❌ Smooth after SG derivative
[SavitzkyGolay(deriv=1), SavitzkyGolay(deriv=0)]  # SG already smooths

# ❌ Derivative before scatter correction
[FirstDerivative(), MSC()]  # Amplifies scatter artifacts

# ❌ Detrend after SG with derivative
[SavitzkyGolay(deriv=1), Detrend()]  # Derivative already removes linear baseline
```

## Multi-Layer Preprocessing for Deep Learning

### Optimal 8-10 Channels

For neural networks, stack multiple preprocessing views as channels:

```python
preprocessing_layers = [
    # Layer 1: Raw baseline information
    None,  # Pass-through

    # Layer 2: Scatter-corrected baseline
    [MSC()],

    # Layer 3: Normalized + smoothed
    [SNV(), SavitzkyGolay()],

    # Layer 4: First derivative (critical!)
    [MSC(), FirstDerivative()],

    # Layer 5: Second derivative
    [SNV(), SecondDerivative()],

    # Layer 6: SNV-detrend combo
    [SNV(), Detrend()],

    # Layer 7: Wavelet high-frequency
    [Wavelet('db4')],

    # Layer 8: Wavelet low-frequency
    [Wavelet('sym5')],
]
```

### Minimal 5 Channels

```python
minimal_layers = [
    None,                               # Raw
    [MSC(), SavitzkyGolay()],           # Standard
    [SNV(), FirstDerivative()],         # 1st derivative
    [MSC(), SecondDerivative()],        # 2nd derivative
    [Wavelet('db6')],                   # Wavelet
]
```

### Per-Channel Scaling

:::{warning}
After concatenating multi-layer preprocessing, apply **per-channel** scaling:

```python
# ❌ WRONG: Global scaler on all channels
all_channels = np.concatenate([ch1, ch2, ch3], axis=1)
scaler.fit_transform(all_channels)

# ✅ CORRECT: Scale each channel separately
for i, channel in enumerate(channels):
    channels[i] = StandardScaler().fit_transform(channel)
final = np.stack(channels, axis=-1)  # (samples, wavelengths, channels)
```
:::

## Task-Specific Preprocessing

### Protein/Nitrogen (N-H bonds: 2000-2200nm)

```python
protein_preprocessing = [
    MSC(),
    SavitzkyGolay(window_length=17, deriv=1),  # Critical for protein
    StandardScaler(),
]
```

Key considerations:
- 1st derivative enhances amide bands
- 2nd derivative resolves overlapping peaks
- Consider Wavelet('coif3') for additional features

### Moisture (O-H bonds: 1400-1500nm, 1900-2000nm)

```python
moisture_preprocessing = [
    MSC(),
    SavitzkyGolay(window_length=15),  # Smooth only in high absorption
    RSNV(),  # Local scatter for heterogeneous moisture
]
```

Key considerations:
- Water peaks are strong, derivatives may not be needed
- Use caution near saturation regions
- Haar wavelet captures sharp water absorption edges

### Fat/Oil (C-H bonds: 1700-1800nm, 2300-2400nm)

```python
fat_preprocessing = [
    MSC(),  # Important for fat scatter
    SNV(),
    SavitzkyGolay(window_length=21, deriv=1),
]
```

Key considerations:
- Fat scatter is significant, MSC helps
- Area normalization can be useful
- Wavelet('db8') captures smooth fat peaks

### Cellulose/Lignin (Plant matrices)

```python
cellulose_preprocessing = [
    Detrend(),
    MSC(),
    SavitzkyGolay(window_length=17, deriv=1),
    # Consider region-specific: 1600-1800nm, 2100-2350nm
]
```

## Advanced Techniques

### Region-Specific Preprocessing

Different spectral regions may benefit from different preprocessing:

```python
def region_specific_preprocessing(X, wavelengths):
    """Apply different preprocessing to different regions."""

    # Region 1: 1100-1400nm (C-H, good SNR)
    mask1 = wavelengths < 1400
    X1 = SNV().fit_transform(X[:, mask1])
    X1 = FirstDerivative().fit_transform(X1)

    # Region 2: 1400-1600nm (water, high absorption)
    mask2 = (wavelengths >= 1400) & (wavelengths < 1600)
    X2 = MSC().fit_transform(X[:, mask2])
    X2 = SavitzkyGolay().fit_transform(X2)  # Smooth only

    # Region 3: 1600-2400nm (protein, fat, lower SNR)
    mask3 = wavelengths >= 1600
    X3 = RSNV().fit_transform(X[:, mask3])
    X3 = SecondDerivative().fit_transform(X3)

    return np.concatenate([X1, X2, X3], axis=1)
```

### Instrument Transfer

When transferring calibrations between instruments:

```python
transfer_preprocessing = [
    # EMSC with reference from new instrument
    EMSC(reference_spectrum=new_instrument_reference),
    # Or PDS (Piecewise Direct Standardization)
    PDS(transfer_samples=transfer_set),
    # Standard preprocessing
    SavitzkyGolay(deriv=1),
    StandardScaler(),
]
```

## Validation & Quality Control

### Check Your Preprocessing

Use this function to validate preprocessing choices:

```python
def validate_preprocessing_layers(layers):
    """Check for common preprocessing mistakes."""
    issues = []

    # Check for scatter correction redundancy
    scatter_methods = ['SNV', 'MSC', 'LSNV', 'RSNV']
    for layer in layers:
        if isinstance(layer, list):
            scatter_count = sum(
                any(s in str(p) for s in scatter_methods)
                for p in layer
            )
            if scatter_count > 1:
                issues.append("⚠️ Multiple scatter corrections in same pipeline")

    # Check for derivative coverage
    all_layers_str = str(layers)
    if 'FirstDeriv' not in all_layers_str and 'deriv=1' not in all_layers_str:
        issues.append("❌ Missing first derivative (critical for NIRS)")

    return issues
```

### Metrics to Monitor

| Metric | What it Tells You |
|--------|-------------------|
| **RMSECV** | Cross-validation error (primary metric) |
| **R²** | Explained variance |
| **Bias** | Systematic offset |
| **RPD** | Ratio of Performance to Deviation |

## Summary

### Key Takeaways

1. **Order matters**: Scatter → Smooth → Derivative → Scale
2. **Don't over-process**: Start simple, add complexity only if needed
3. **Match to model**: Trees don't need scaling; neural networks need careful normalization
4. **Validate inside CV**: Never fit preprocessing on full dataset
5. **First derivative is critical**: Almost always improves PLS/SVM/CNN performance

### Quick Decision Tree

```
Is there scatter/baseline?
  └─ Yes → Apply SNV or MSC
  └─ No → Skip

Is data noisy?
  └─ Yes → Apply SavitzkyGolay smoothing
  └─ No → Skip or light smoothing

Need baseline removal?
  └─ Yes → Apply 1st derivative (removes offset + slope)
  └─ Need peak resolution? → Apply 2nd derivative (with caution)

Using distance-based model (SVM, k-NN, MLP)?
  └─ Yes → Apply StandardScaler or RobustScaler
  └─ No (trees) → Skip scaling
```

## See Also

- {doc}`overview` - Quick introduction to preprocessing
- {doc}`cheatsheet` - Quick reference by model type
- {doc}`/reference/operator_catalog` - Complete operator reference
