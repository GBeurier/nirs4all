# Spectral Latent Features Transformer

A scientifically-grounded transformer for converting high-dimensional NIRS spectra into TabPFN-optimized latent features.

## Overview

### The Problem

TabPFN 2.5 was trained on synthetic tabular data with specific characteristics:
- **Independent features**: Low autocorrelation between columns
- **Mixed types**: Both continuous and categorical-like features
- **Moderate dimensionality**: ~100-500 features
- **Simple correlations**: Tree-like or polynomial relationships

NIRS (Near-Infrared Spectroscopy) spectra are fundamentally different:
- **Highly autocorrelated**: Neighboring wavelengths are nearly identical
- **High dimensionality**: Often 1000+ wavelengths
- **Smooth signals**: Continuous absorption curves
- **Multicollinear**: Most wavelengths are redundant

This transformer bridges the gap by extracting diverse, decorrelated features that capture spectral information in a format TabPFN can leverage effectively.

---

## Feature Extraction Modules

### 1. PCA Module (~60 features)

**Purpose**: Extract decorrelated global patterns that explain maximum variance.

**Scientific Basis**:
Principal Component Analysis decomposes the spectrum into orthogonal components (scores) ordered by variance explained. This is the gold standard for:
- Removing multicollinearity that confuses ML models
- Compressing spectral information into fewer dimensions
- Separating signal from noise (later PCs often capture noise)

**Features**:
- First N principal component scores (decorrelated, unit variance if whitened)

**References**:
- Wold, S., Esbensen, K., & Geladi, P. (1987). Principal Component Analysis. *Chemometrics and Intelligent Laboratory Systems*, 2(1-3), 37-52.
- Næs, T., Isaksson, T., Fearn, T., & Davies, T. (2002). *A User-Friendly Guide to Multivariate Calibration and Classification*. NIR Publications.

---

### 2. Wavelet Module (~60 features)

**Purpose**: Multi-scale decomposition capturing both smooth trends and sharp features.

**Scientific Basis**:
The Discrete Wavelet Transform (DWT) decomposes the spectrum into:
- **Approximation coefficients**: Smooth, low-frequency baseline trends
- **Detail coefficients**: Sharp, high-frequency peaks and variations

This multi-resolution analysis captures information at different spatial scales, similar to how human experts analyze spectra.

**Features**:
- Per-level statistics: mean, std, energy (L2 norm), entropy
- Top N coefficients (by magnitude) at each level
- Supports wavelets: db4 (Daubechies-4), sym4 (Symlet-4), haar, coif3

**Key Insight**:
The energy distribution across decomposition levels reveals spectral complexity. Spectra dominated by smooth baselines have more energy in approximation; those with sharp peaks have more in details.

**References**:
- Mallat, S. G. (1989). A theory for multiresolution signal decomposition: the wavelet representation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 11(7), 674-693.
- Trygg, J., & Wold, S. (1998). PLS regression on wavelet compressed NIR spectra. *Chemometrics and Intelligent Laboratory Systems*, 42(1-2), 209-220.

---

### 3. FFT Module (~34 features)

**Purpose**: Frequency domain representation capturing periodic patterns.

**Scientific Basis**:
The Fast Fourier Transform converts spectra from the wavelength domain to the frequency domain:
- **Low frequencies**: Smooth baseline variations, broad absorption bands
- **High frequencies**: Sharp peaks, noise

Log-spaced frequency bands ensure multi-scale representation similar to how the human visual system perceives frequency.

**Features**:
- Band energy: Sum of squared magnitudes in log-spaced frequency bands
- Top N frequencies: Magnitudes of the most prominent frequency components
- Spectral centroid: "Center of mass" of the frequency distribution
- Spectral spread: Standard deviation around the centroid

**References**:
- Cooley, J. W., & Tukey, J. W. (1965). An algorithm for the machine calculation of complex Fourier series. *Mathematics of Computation*, 19(90), 297-301.
- Geladi, P., MacDougall, D., & Martens, H. (1985). Linearization and scatter-correction for near-infrared reflectance spectra of meat. *Applied Spectroscopy*, 39(3), 491-500.

---

### 4. Local Statistics Module (~99 features)

**Purpose**: Band-wise statistics that simulate independent measurements.

**Scientific Basis**:
Dividing the spectrum into bands and computing statistics per band simulates having multiple independent sensors measuring different spectral regions. This creates features that resemble typical tabular data (e.g., sensor readings, measurements).

**Features per band**:
- mean, std, min, max
- Quantiles: Q25, Q75 (interquartile range indicators)
- Skewness: Asymmetry of value distribution
- Kurtosis: "Peakedness" of distribution

**Inter-band features**:
- Slope: Linear trend across band means
- Curvature: Second derivative of band means
- Range ratio: (max - min) / mean

**References**:
- Næs, T., et al. (2002). *A User-Friendly Guide to Multivariate Calibration and Classification*.
- Brown, S. D., Tauler, R., & Walczak, B. (Eds.). (2020). *Comprehensive Chemometrics*. Elsevier.

---

### 5. Derivative Module (~20 features)

**Purpose**: Shape characterization emphasizing spectral changes.

**Scientific Basis**:
Derivatives are fundamental in spectroscopy:
- **1st derivative**: Rate of change; removes constant baseline offset
- **2nd derivative**: Curvature; sharpens peaks, removes linear baseline

Savitzky-Golay filtering provides smooth derivatives while reducing noise amplification.

**Features**:
For each derivative order (1st, 2nd):
- mean, std, min, max, range
- Zero-crossing count: Number of sign changes (indicates spectral complexity)
- Positive/negative area: Separate integral contributions
- Max absolute position: Location of largest change (normalized 0-1)
- Energy: Sum of squared values

**References**:
- Savitzky, A., & Golay, M. J. (1964). Smoothing and differentiation of data by simplified least squares procedures. *Analytical Chemistry*, 36(8), 1627-1639.
- Rinnan, Å., Van Den Berg, F., & Engelsen, S. B. (2009). Review of the most common pre-processing techniques for near-infrared spectra. *TrAC Trends in Analytical Chemistry*, 28(10), 1201-1222.

---

### 6. Peak Module (~40 features)

**Purpose**: Absorption band characterization based on Beer-Lambert law.

**Scientific Basis**:
In NIRS, peaks correspond to molecular absorption bands:
- **Peak positions**: Relate to specific molecular vibrations (O-H, C-H, N-H)
- **Peak heights**: Proportional to concentration (Beer-Lambert law: A = εlc)
- **Peak widths**: Relate to molecular environment and overlapping bands
- **Peak prominence**: Distinguishes true peaks from baseline fluctuations

**Features**:
- n_peaks: Total number of detected peaks
- peak_density: Peaks per wavelength (spectral complexity)
- Per-peak (top N): position, height, prominence, width
- Aggregate: mean/std of heights, widths, prominences, spacing

**References**:
- Workman Jr, J., & Weyer, L. (2012). *Practical Guide and Spectral Atlas for Interpretive Near-Infrared Spectroscopy*. CRC Press.
- Siesler, H. W., Ozaki, Y., Kawata, S., & Heise, H. M. (Eds.). (2008). *Near-Infrared Spectroscopy: Principles, Instruments, Applications*. John Wiley & Sons.

---

### 7. Scatter Module (~14 features)

**Purpose**: Baseline and light scattering characterization.

**Scientific Basis**:
Light scattering in NIRS depends on:
- **Particle size**: Affects baseline slope and offset
- **Sample texture**: Causes multiplicative effects
- **Path length**: Varies with sample heterogeneity

These effects, while not directly related to chemical composition, can carry predictive information (e.g., grain hardness relates to light scattering).

**Features**:
- Polynomial baseline coefficients (degree 0-3): Intercept, slope, curvature, higher-order
- Residual statistics: std, skew, kurtosis of baseline-corrected spectrum
- Total/normalized area under curve
- Spectral centroid and spread
- Asymmetry: Left vs right half imbalance
- Flatness: Geometric mean / arithmetic mean ratio
- Crest factor: Peak / RMS ratio

**References**:
- Martens, H., & Næs, T. (1989). *Multivariate Calibration*. John Wiley & Sons.
- Barnes, R. J., Dhanoa, M. S., & Lister, S. J. (1989). Standard normal variate transformation and de-trending of near-infrared diffuse reflectance spectra. *Applied Spectroscopy*, 43(5), 772-777.

---

### 8. Discretization Module (~19 features)

**Purpose**: Create categorical-like features to match TabPFN's training data distribution.

**Scientific Basis**:
TabPFN was trained on data with mixed continuous and categorical features. By discretizing spectral values into bins, we create features that resemble categorical measurements.

**Features**:
- Global histogram: Proportion in each of N bins (learned from training data)
- Histogram entropy: Information content of the distribution
- Mode bin: Most frequent value category
- Uniformity: Sum of squared proportions (1 = single bin, 1/N = uniform)
- Per-band mode bins: Dominant value category in each spectral region

**References**:
- Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and unsupervised discretization of continuous features. *Machine Learning Proceedings*, 194-202.
- Hollmann, N., et al. (2023). TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second. *ICLR 2023*.

---

### 9. Wavelet-PCA Module (~15 features, disabled by default)

**Purpose**: Multi-scale PCA on wavelet coefficients for compact, decorrelated multi-resolution features.

**Scientific Basis**:
Applies PCA separately to each wavelet decomposition level:
- Each scale captures different frequency information
- PCA per scale reduces redundancy within each frequency band
- Results in a compact, interpretable feature set
- Combines benefits of wavelets (multi-resolution) with PCA (decorrelation)

**Features**:
- 2-5 principal components per wavelet level (approximation + details)
- Whitened for unit variance

**References**:
- Trygg, J., & Wold, S. (1998). PLS regression on wavelet compressed NIR spectra. *Chemometrics and Intelligent Laboratory Systems*, 42(1-2), 209-220.

---

### 10. PLS Module (~20 features, disabled by default)

**Purpose**: Supervised latent features aligned with the target variable for regression tasks.

**Scientific Basis**:
Partial Least Squares finds latent variables (scores) that maximize covariance with Y:
- Optimal for prediction (scores are Y-aligned)
- Naturally handles multicollinearity
- Widely used in chemometrics (NIRS, Raman, etc.)
- Decorrelated scores capture predictive variance

**Features**:
- N PLS scores (latent variables aligned with Y)

**Note**: Requires `y` during `fit()`. If `y` is not provided, this module is skipped.

**References**:
- Wold, S., Sjöström, M., & Eriksson, L. (2001). PLS-regression: a basic tool of chemometrics. *Chemometrics and Intelligent Laboratory Systems*, 58(2), 109-130.

---

### 11. NMF Module (~15 features, disabled by default)

**Purpose**: Non-negative spectral decomposition for interpretable component analysis.

**Scientific Basis**:
Non-negative Matrix Factorization decomposes spectra into:
- Non-negative basis spectra (can represent pure chemical components)
- Non-negative mixing coefficients (interpretable as concentrations)
- Additive decomposition matches physical reality of absorbance

**Features**:
- N mixing coefficients representing contributions of latent spectral components

**Note**: Works best with non-negative spectra. Negative values are shifted automatically.

**References**:
- Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, 401(6755), 788-791.

---

### 12. Band Area Module (~25-35 features, disabled by default)

**Purpose**: Integrated intensity features based on Beer-Lambert law.

**Scientific Basis**:
Area under the curve (AUC) for spectral bands provides:
- Robust integration reducing noise sensitivity
- Total absorption in regions (relates to concentration)
- Normalized areas for relative composition
- Band ratios commonly used in spectroscopic analysis

**Features**:
- Absolute area per band (trapezoidal integration)
- Normalized areas (area_band / total_area)
- Total area under spectrum
- Adjacent band ratios
- First-half vs second-half ratio

**References**:
- Mark, H., & Workman Jr, J. (2007). *Chemometrics in Spectroscopy*. Academic Press.

---

## Output Normalization

The transformer applies output normalization to make features more "tabular-like":

### Quantile Transform (default)
Transforms features to a uniform distribution (0 to 1). This:
- Removes outliers' influence
- Makes all features comparable
- Matches the bounded, smooth distributions in TabPFN's training data

### Power Transform (alternative)
Yeo-Johnson transformation to approximate Gaussian distributions:
- Handles both positive and negative values
- Reduces skewness
- Stabilizes variance

### Standard Scaling
Simple z-score normalization (mean=0, std=1):
- Fastest option
- Preserves relative relationships
- May leave outliers with extreme values

---

## Usage Examples

### Basic Usage

```python
from spectral_latent_features import SpectralLatentFeatures

# Create transformer with defaults (~300 features)
transformer = SpectralLatentFeatures()

# Fit and transform training data
X_train_latent = transformer.fit_transform(X_train)

# Transform test data
X_test_latent = transformer.transform(X_test)

print(f"Original: {X_train.shape} -> Latent: {X_train_latent.shape}")
```

### With TabPFN

```python
from spectral_latent_features import SpectralLatentFeatures
from tabpfn import TabPFNRegressor

# Create feature transformer
feature_transformer = SpectralLatentFeatures(
    n_pca=60,
    use_wavelets=True,
    output_normalization='quantile'
)

# Transform spectra to latent features
X_train_latent = feature_transformer.fit_transform(X_train)
X_test_latent = feature_transformer.transform(X_test)

# Train TabPFN on latent features
model = TabPFNRegressor(device='cuda')
model.fit(X_train_latent, y_train)
y_pred = model.predict(X_test_latent)
```

### Lightweight Version

```python
from spectral_latent_features import SpectralLatentFeaturesLite

# Faster transformer with fewer features (~150)
transformer = SpectralLatentFeaturesLite(n_pca=40, n_local_bands=10)
X_latent = transformer.fit_transform(X)
```

### Convenience Function

```python
from spectral_latent_features import create_tabpfn_features

# Auto-configure for target feature count
X_latent, transformer = create_tabpfn_features(X_train, n_features=300)
X_test_latent = transformer.transform(X_test)
```

### Integration with NIRS4All Pipeline

```python
from spectral_latent_features import SpectralLatentFeatures

pipeline = [
    {"split": SPXYGFold, "split_params": {...}, "group": "ID"},
    SpectralLatentFeatures(n_pca=80, use_wavelets=True),
    {"model": TabPFNRegressor(device='cuda'), "name": "TabPFN"}
]
```

---

## Parameter Tuning Guide

### Module Activation Flags and Associated Parameters

Each module can be enabled/disabled with a `use_*` flag. When disabled (`False`), the associated parameters have no effect.

| Activation Flag | Default | Associated Parameters | Description |
|-----------------|---------|----------------------|-------------|
| `use_pca` | `True` | `n_pca`, `pca_whiten`, `pca_variance_threshold` | Global decorrelated patterns via PCA |
| `use_wavelets` | `True` | `wavelet`, `wavelet_levels`, `wavelet_coeffs_per_level` | Multi-scale decomposition (requires `pywt`) |
| `use_wavelet_pca` | `False` | `wavelet`, `wavelet_levels`, `wavelet_pca_components_per_level` | PCA per wavelet scale (requires `pywt`) |
| `use_fft` | `True` | `n_fft_bands`, `n_fft_top` | Frequency domain features |
| `use_local_stats` | `True` | `n_local_bands` | Band-wise statistics |
| `use_derivatives` | `True` | `n_deriv_window` | Savitzky-Golay derivative features |
| `use_peaks` | `True` | `n_peaks` | Absorption band detection |
| `use_scatter` | `True` | `poly_degree` | Baseline/scattering indices |
| `use_discretization` | `True` | `n_bins`, `n_disc_bands` | Categorical-like histogram features |
| `use_pls` | `False` | `n_pls` | Supervised PLS latent scores (requires `y`) |
| `use_nmf` | `False` | `n_nmf`, `nmf_max_iter` | Non-negative matrix factorization |
| `use_band_areas` | `False` | `n_area_bands`, `area_include_ratios` | Band area (AUC) features |

**Note**: `output_normalization` and `random_state` are always used regardless of module activation.

### Parameter Reference by Module

| Module | Parameter | Default | Description | Impact |
|--------|-----------|---------|-------------|--------|
| **PCA** | `n_pca` | 60 | Number of PCA components | More = more global variance captured |
| | `pca_whiten` | True | Unit variance PCA components | Improves TabPFN compatibility |
| | `pca_variance_threshold` | 0.999 | Variance threshold for component selection | Higher = more components retained |
| **Wavelet** | `wavelet` | 'db4' | Wavelet type ('db4', 'sym4', 'haar', 'coif3') | db4 good for smooth signals |
| | `wavelet_levels` | 4 | Decomposition depth | More levels = more scales |
| | `wavelet_coeffs_per_level` | 8 | Top coefficients per level | More = finer detail |
| **Wavelet-PCA** | `wavelet_pca_components_per_level` | 3 | PCA components per wavelet level | More = more multi-scale info |
| **FFT** | `n_fft_bands` | 15 | Log-spaced frequency bands | More = finer frequency resolution |
| | `n_fft_top` | 15 | Top frequency magnitudes | More = more frequency info |
| **Local Stats** | `n_local_bands` | 12 | Spectral bands for statistics | More = finer spatial resolution |
| **Derivatives** | `n_deriv_window` | 11 | Savitzky-Golay window length | Larger = smoother derivatives |
| **Peaks** | `n_peaks` | 8 | Top peaks to extract | More if many absorption bands |
| **Scatter** | `poly_degree` | 3 | Polynomial degree for baseline | Higher = more flexible baseline |
| **Discretization** | `n_bins` | 10 | Histogram bins | More = finer discretization |
| | `n_disc_bands` | 6 | Bands for per-band mode | More = finer spatial resolution |
| **PLS** | `n_pls` | 20 | Number of PLS components | More = captures more Y-related variance |
| **NMF** | `n_nmf` | 15 | Number of NMF components | More = finer spectral decomposition |
| | `nmf_max_iter` | 200 | Max iterations for NMF | Higher = better convergence |
| **Band Areas** | `n_area_bands` | 12 | Number of spectral bands for AUC | More = finer spatial resolution |
| | `area_include_ratios` | True | Include band area ratios | Adds relative composition info |
| **Output** | `output_normalization` | 'quantile' | Normalization method | 'quantile' most TabPFN-friendly |
| | `random_state` | None | Random seed | For reproducibility |

### Recommended Configurations

**For small datasets (< 100 samples)**:
```python
SpectralLatentFeaturesLite(n_pca=30, n_local_bands=8)
# ~120 features - reduces overfitting risk
```

**For medium datasets (100-1000 samples)**:
```python
SpectralLatentFeatures()  # defaults
# ~300-350 features - good balance
```

**For large datasets (> 1000 samples)**:
```python
SpectralLatentFeatures(n_pca=100, n_local_bands=20, n_peaks=15)
# ~450 features - captures more information
```

---

## Feature Importance Interpretation

After training, you can analyze which feature types are most predictive:

```python
# Get feature names
feature_names = transformer.get_feature_names_out()

# With tree-based model
importances = model.feature_importances_
feature_importance = dict(zip(feature_names, importances))

# Group by module
pca_imp = sum(v for k, v in feature_importance.items() if k.startswith('pca_'))
wavelet_imp = sum(v for k, v in feature_importance.items() if k.startswith('wavelet_'))
# ... etc
```

Typical patterns:
- **PCA dominant**: Global spectral patterns most predictive
- **Peak dominant**: Specific absorption bands important
- **Derivative dominant**: Spectral shape/changes matter more than absolute values

---

## Computational Complexity

| Module | Time Complexity | Memory |
|--------|----------------|--------|
| PCA | O(n × d²) | O(d²) |
| Wavelet | O(n × d) | O(d) |
| FFT | O(n × d log d) | O(d) |
| Local Stats | O(n × d) | O(1) |
| Derivatives | O(n × d) | O(d) |
| Peaks | O(n × d) | O(d) |
| Scatter | O(n × d) | O(1) |
| Discretization | O(n × d) | O(1) |

Where n = samples, d = wavelengths

**Typical performance**:
- 1000 samples × 1000 wavelengths: < 5 seconds
- Dominated by PCA fitting

---

## References

### Core Methods

1. Wold, S., Esbensen, K., & Geladi, P. (1987). Principal Component Analysis. *Chemometrics and Intelligent Laboratory Systems*, 2(1-3), 37-52.

2. Mallat, S. G. (1989). A theory for multiresolution signal decomposition: the wavelet representation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 11(7), 674-693.

3. Savitzky, A., & Golay, M. J. (1964). Smoothing and differentiation of data by simplified least squares procedures. *Analytical Chemistry*, 36(8), 1627-1639.

### Spectroscopy Applications

4. Rinnan, Å., Van Den Berg, F., & Engelsen, S. B. (2009). Review of the most common pre-processing techniques for near-infrared spectra. *TrAC Trends in Analytical Chemistry*, 28(10), 1201-1222.

5. Næs, T., Isaksson, T., Fearn, T., & Davies, T. (2002). *A User-Friendly Guide to Multivariate Calibration and Classification*. NIR Publications.

6. Workman Jr, J., & Weyer, L. (2012). *Practical Guide and Spectral Atlas for Interpretive Near-Infrared Spectroscopy*. CRC Press.

### TabPFN

7. Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F. (2023). TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second. *ICLR 2023*.

### Chemometrics

8. Martens, H., & Næs, T. (1989). *Multivariate Calibration*. John Wiley & Sons.

9. Brown, S. D., Tauler, R., & Walczak, B. (Eds.). (2020). *Comprehensive Chemometrics*. Elsevier.

---

## License

This module is part of the NIRS4All project.
