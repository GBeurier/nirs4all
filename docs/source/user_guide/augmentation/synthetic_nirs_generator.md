# Synthetic NIRS Spectra Generator - Scientific Documentation

## Overview

The `SyntheticNIRSGenerator` provides a physically-motivated model for generating realistic synthetic Near-Infrared (NIR) spectra. It implements the Beer-Lambert law with additional effects for instrumental artifacts, noise, and inter-sample variability commonly observed in real-world spectroscopic measurements.

This generator is designed for:
- **Autoencoder training**: Generate unlimited labeled data for unsupervised feature learning
- **Algorithm benchmarking**: Test preprocessing and modeling algorithms under controlled conditions
- **Domain adaptation research**: Simulate multi-instrument/multi-session variability
- **Data augmentation**: Supplement real datasets with physically plausible synthetic samples

---

## Theoretical Foundation

### Beer-Lambert Law

The fundamental spectroscopic relationship underlying the generator is the **Beer-Lambert-Bouguer Law**:

$$A(\lambda) = \varepsilon(\lambda) \cdot c \cdot L$$

Where:
- $A(\lambda)$ = Absorbance at wavelength $\lambda$
- $\varepsilon(\lambda)$ = Molar absorptivity (extinction coefficient)
- $c$ = Concentration of the absorbing species
- $L$ = Optical path length

For mixtures with $K$ components, the total absorbance follows the **additivity principle**:

$$A(\lambda) = \sum_{k=1}^{K} \varepsilon_k(\lambda) \cdot c_k \cdot L$$

In matrix notation: $\mathbf{A} = \mathbf{C} \cdot \mathbf{E}$

where $\mathbf{C}$ is the concentration matrix $(N \times K)$ and $\mathbf{E}$ is the pure component spectra matrix $(K \times P)$.

**References:**
- Beer, A. (1852). Bestimmung der Absorption des rothen Lichts in farbigen Flüssigkeiten. *Annalen der Physik*, 162(5), 78-88.
- Swinehart, D. F. (1962). The Beer-Lambert Law. *Journal of Chemical Education*, 39(7), 333.

---

## Peak Shape: Voigt Profile

### Scientific Justification

Real absorption peaks in NIR spectroscopy are neither purely Gaussian nor purely Lorentzian. The observed peak shape is a **Voigt profile** - the convolution of Gaussian (Doppler/thermal) and Lorentzian (collision/pressure) broadening mechanisms.

$$V(x; \sigma, \gamma) = \int_{-\infty}^{\infty} G(x'; \sigma) L(x - x'; \gamma) dx'$$

Where:
- $G(x; \sigma)$ = Gaussian profile with width $\sigma$
- $L(x; \gamma)$ = Lorentzian profile with width $\gamma$

The Voigt function is computed using `scipy.special.voigt_profile`.

### Parameters

| Parameter | Type | Effect | Typical Range |
|-----------|------|--------|---------------|
| `center` | float | Peak position in nm | 1000-2500 nm |
| `sigma` | float | Gaussian width (HWHM) in nm | 10-50 nm |
| `gamma` | float | Lorentzian width (HWHM) in nm | 0-10 nm |
| `amplitude` | float | Peak height in absorbance units | 0.1-1.0 AU |

**Effect of parameters:**
- **sigma**: Controls the overall peak width. Larger values = broader peaks
- **gamma**: Controls "tails" of the peak. γ=0 gives pure Gaussian; increasing γ adds Lorentzian character with heavier tails
- **amplitude**: Directly scales peak height; related to absorptivity × concentration

**References:**
- Olivero, J. J., & Longbothum, R. L. (1977). Empirical fits to the Voigt line width. *Journal of Quantitative Spectroscopy and Radiative Transfer*, 17(2), 233-236.
- Whiting, E. E. (1968). An empirical approximation to the Voigt profile. *Journal of Quantitative Spectroscopy and Radiative Transfer*, 8(6), 1379-1384.

---

## NIR Band Assignments

### Scientific Basis

NIR absorption bands arise from **molecular overtones and combination bands** of fundamental vibrational modes. The predefined components are based on established NIR band assignments from spectroscopic literature.

| Functional Group | Band Type | Wavelength (nm) | Spectral Region |
|-----------------|-----------|-----------------|-----------------|
| O-H (water) | 1st overtone | 1400-1500 | ν + δ |
| O-H (water) | combination | 1900-2000 | ν₁ + ν₃ |
| N-H (protein) | 1st overtone | 1500-1560 | 2ν |
| N-H (protein) | combination | 2000-2100 | ν + δ |
| C-H (aliphatic) | 2nd overtone | 1100-1250 | 3ν |
| C-H (aliphatic) | 1st overtone | 1650-1800 | 2ν |
| C-H (aromatic) | combination | 2200-2400 | ν + δ |

Where ν = stretching, δ = bending modes.

### Predefined Components

The generator includes **111 predefined spectral components** covering diverse application areas. Example components (showing key band assignments):

```
# Water & Moisture (2)
water:        O-H bands at 1450, 1940, 2500 nm
moisture:     Bound water at 1460, 1930 nm

# Proteins (12): protein, casein, gluten, albumin, collagen, keratin, whey...
protein:      N-H bands at 1510, 1680, 2050, 2180, 2300 nm

# Lipids (15): lipid, oil, oleic_acid, palmitic_acid, phospholipid, cholesterol...
lipid:        C-H bands at 1210, 1390, 1720, 2310, 2350 nm

# Carbohydrates (18): starch, cellulose, glucose, maltose, raffinose, trehalose...
starch:       O-H/C-O bands at 1460, 1580, 2100, 2270 nm
cellulose:    O-H/C-O bands at 1490, 1780, 2090, 2280, 2340 nm

# Pigments (8): chlorophyll, carotenoid, anthocyanin, lycopene, lutein...
chlorophyll:  absorption at 1070, 1400, 2270 nm

# Also includes: Alcohols (9), Organic Acids (12), Pharmaceuticals (10),
#                Polymers (10), Solvents (6), Minerals (8), Fibers (2)
```

See {doc}`/user_guide/data/synthetic_data` for the complete list of 111 components.

**References:**
- Workman Jr, J., & Weyer, L. (2012). *Practical Guide and Spectral Atlas for Interpretive Near-Infrared Spectroscopy*. CRC Press.
- Burns, D. A., & Ciurczak, E. W. (2007). *Handbook of Near-Infrared Analysis* (3rd ed.). CRC Press.
- Shenk, J. S., Workman Jr, J. J., & Westerhaus, M. O. (2008). Application of NIR spectroscopy to agricultural products. In *Handbook of Near-Infrared Analysis* (3rd ed., pp. 347-386). CRC Press.

---

## Concentration Generation Methods

### Dirichlet Distribution (default)

The Dirichlet distribution generates compositional data that sums to 1.0, appropriate for relative proportions:

$$\mathbf{c} \sim \text{Dir}(\alpha_1, \alpha_2, ..., \alpha_K)$$

| Parameter | Effect |
|-----------|--------|
| α = [1,1,...,1] | Uniform over simplex |
| α = [2,2,...,2] | Concentrated toward center |
| α < 1 | Sparse (extreme values) |
| α > 1 | Dense (moderate values) |

### Other Methods

| Method | Distribution | Use Case |
|--------|--------------|----------|
| `uniform` | U(0,1) independent | When components are independent |
| `lognormal` | LogN(0, 0.5) normalized | For positively skewed concentrations |
| `correlated` | Multivariate normal + Cholesky | When components have known correlations |

**References:**
- Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*. Chapman & Hall.

---

## Instrumental and Physical Effects

### 1. Path Length Variation

**Scientific Basis:** In diffuse reflectance/transmittance, the effective optical path length varies due to:
- Sample packing density
- Particle size distribution
- Probe-sample contact pressure

$$A_i(\lambda) = L_i \cdot A_0(\lambda), \quad L_i \sim \mathcal{N}(1, \sigma_L)$$

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| `path_length_std` | Sample-to-sample variation | 0.02-0.08 |

**Reference:** Martens, H., & Næs, T. (1989). *Multivariate Calibration*. John Wiley & Sons.

---

### 2. Baseline Drift

**Scientific Basis:** Baseline variations arise from:
- Detector drift over time
- Temperature effects on optical components
- Sample holder variations
- Reference spectrum mismatches

Modeled as a polynomial:

$$\text{baseline}_i(\lambda) = b_0 + b_1 \tilde{\lambda} + b_2 \tilde{\lambda}^2 + b_3 \tilde{\lambda}^3$$

where $\tilde{\lambda}$ is the centered, scaled wavelength.

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| `baseline_amplitude` | Maximum drift magnitude (AU) | 0.01-0.05 |

---

### 3. Global Slope

**Scientific Basis:** NIR spectra commonly exhibit a global upward or downward trend caused by:
- **Rayleigh scattering**: $I \propto \lambda^{-4}$ (small particles)
- **Mie scattering**: wavelength-dependent for particles comparable to λ
- **Sample surface roughness**: affects baseline slope
- **Detector sensitivity curve**: not perfectly flat

$$A_i(\lambda) \leftarrow A_i(\lambda) + s_i \cdot \frac{\lambda - \lambda_{\min}}{\lambda_{\max} - \lambda_{\min}}$$

where $s_i \sim \mathcal{N}(\mu_s, \sigma_s)$ is the slope (absorbance per 1000nm).

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| `global_slope_mean` | Average slope direction (AU/1000nm) | -0.1 to +0.15 |
| `global_slope_std` | Sample-to-sample slope variation | 0.02-0.05 |

**Positive slope:** Common in diffuse reflectance (scattering increases with λ)
**Negative slope:** Can occur with certain sample types or instrument configurations

**References:**
- Rinnan, Å., Van Den Berg, F., & Engelsen, S. B. (2009). Review of the most common pre-processing techniques for near-infrared spectra. *TrAC Trends in Analytical Chemistry*, 28(10), 1201-1222.

---

### 4. Scattering Effects (MSC/SNV-like)

**Scientific Basis:** Multiplicative Scatter Correction (MSC) and Standard Normal Variate (SNV) are designed to remove scatter-induced baseline effects. The generator simulates these effects *before* correction:

$$A_{\text{scatter}}(\lambda) = \alpha \cdot A(\lambda) + \beta + \gamma \cdot \tilde{\lambda}$$

Where:
- $\alpha$ = multiplicative scatter effect (gain)
- $\beta$ = additive offset
- $\gamma$ = wavelength-dependent tilt

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| `scatter_alpha_std` | Multiplicative variation | 0.02-0.08 |
| `scatter_beta_std` | Additive offset variation | 0.005-0.02 |
| `tilt_std` | Linear tilt variation | 0.005-0.02 |

**References:**
- Geladi, P., MacDougall, D., & Martens, H. (1985). Linearization and scatter-correction for near-infrared reflectance spectra of meat. *Applied Spectroscopy*, 39(3), 491-500.
- Barnes, R. J., Dhanoa, M. S., & Lister, S. J. (1989). Standard normal variate transformation and de-trending of near-infrared diffuse reflectance spectra. *Applied Spectroscopy*, 43(5), 772-777.

---

### 5. Wavelength Calibration Errors

**Scientific Basis:** Wavelength calibration errors occur due to:
- Spectrometer temperature changes (thermal expansion of grating)
- Aging of optical components
- Mechanical drift in monochromator
- Different instruments having slightly different calibrations

Modeled as a shift and stretch:

$$\lambda_{\text{measured}} = s \cdot \lambda_{\text{true}} + \Delta\lambda$$

Where $s \sim \mathcal{N}(1, \sigma_s)$ and $\Delta\lambda \sim \mathcal{N}(0, \sigma_\Delta)$.

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| `shift_std` | Wavelength shift (nm) | 0.2-1.0 nm |
| `stretch_std` | Wavelength scale factor std | 0.0005-0.002 |

**Reference:** Feudale, R. N., et al. (2002). Transfer of multivariate calibration models: a review. *Chemometrics and Intelligent Laboratory Systems*, 64(2), 181-192.

---

### 6. Instrumental Broadening

**Scientific Basis:** Finite spectral resolution of the instrument causes peak broadening. The instrument's slit function is approximated as Gaussian.

$$A_{\text{meas}}(\lambda) = A_{\text{true}}(\lambda) * G(\lambda; \text{FWHM})$$

Where * denotes convolution and FWHM is the Full Width at Half Maximum of the instrumental line shape.

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| `instrumental_fwhm` | Spectral resolution (nm) | 4-12 nm |

**Lower FWHM:** Higher resolution, sharper peaks, typically research-grade instruments
**Higher FWHM:** Lower resolution, broader peaks, typical of industrial/portable instruments

**Reference:** Griffiths, P. R., & De Haseth, J. A. (2007). *Fourier Transform Infrared Spectrometry* (2nd ed.). John Wiley & Sons.

---

### 7. Noise Model

**Scientific Basis:** NIR detector noise has multiple components:
- **Shot noise**: Poisson-distributed, signal-dependent
- **Thermal noise**: Johnson-Nyquist, independent of signal
- **Readout noise**: Electronics, independent of signal

Approximated as heteroscedastic Gaussian:

$$\sigma(\lambda) = \sigma_{\text{base}} + \sigma_{\text{signal}} \cdot |A(\lambda)|$$

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| `noise_base` | Signal-independent noise floor | 0.002-0.008 |
| `noise_signal_dep` | Signal-dependent noise factor | 0.005-0.015 |

**Signal-to-Noise Ratio (SNR)** approximately: SNR ≈ A / σ(A)

**Reference:** Workman Jr, J. (2007). NIR spectroscopy instrumentation. In *Handbook of Near-Infrared Analysis* (3rd ed., pp. 91-112). CRC Press.

---

### 8. Artifacts

**Scientific Basis:** Real-world spectra may contain artifacts:

| Artifact Type | Cause | Model |
|---------------|-------|-------|
| **Spike** | Cosmic rays, electrical interference | Random point additions |
| **Dead band** | Detector defects, atmospheric absorption | Localized noise increase |
| **Saturation** | Detector/ADC overflow | Clipping at high absorbance |

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| `artifact_prob` | Probability of artifact per sample | 0.0-0.05 |

---

### 9. Batch Effects

**Scientific Basis:** Multi-session/multi-instrument data exhibits systematic differences:
- Lamp aging → intensity drift
- Environmental changes → baseline shift
- Recalibration → scale changes

$$A_{\text{batch}_j} = g_j \cdot A + \mathbf{o}_j$$

Where $g_j$ is batch-specific gain and $\mathbf{o}_j$ is batch-specific offset.

Used for:
- Domain adaptation research
- Transfer learning studies
- Calibration maintenance testing

**Reference:** Feudale, R. N., et al. (2002). Transfer of multivariate calibration models: a review. *Chemometrics and Intelligent Laboratory Systems*, 64(2), 181-192.

---

## Complexity Levels

The generator provides three preset complexity levels optimizing parameters for different use cases:

### Simple (Testing/Debugging)
```python
complexity = "simple"
```
- Low noise, minimal artifacts
- Small path length and scatter variation
- No global slope (flat baseline trend)
- Suitable for: algorithm debugging, unit testing

### Realistic (Training/Benchmarking)
```python
complexity = "realistic"  # Default
```
- Moderate noise levels
- Typical instrument resolution (8 nm FWHM)
- ~2% artifact rate
- Positive global slope (typical NIR behavior)
- Suitable for: model training, algorithm comparison

### Complex (Robustness Testing)
```python
complexity = "complex"
```
- High noise levels
- Large inter-sample variability
- ~5% artifact rate
- Strong global slope variation
- Lower resolution (12 nm FWHM)
- Suitable for: stress testing, robustness evaluation

---

## Complete Parameter Reference

| Parameter | Simple | Realistic | Complex | Unit |
|-----------|--------|-----------|---------|------|
| `path_length_std` | 0.02 | 0.05 | 0.08 | fraction |
| `baseline_amplitude` | 0.01 | 0.02 | 0.05 | AU |
| `scatter_alpha_std` | 0.02 | 0.05 | 0.08 | fraction |
| `scatter_beta_std` | 0.005 | 0.01 | 0.02 | AU |
| `tilt_std` | 0.005 | 0.01 | 0.02 | AU |
| `global_slope_mean` | 0.0 | 0.05 | 0.08 | AU/1000nm |
| `global_slope_std` | 0.02 | 0.03 | 0.05 | AU/1000nm |
| `shift_std` | 0.2 | 0.5 | 1.0 | nm |
| `stretch_std` | 0.0005 | 0.001 | 0.002 | fraction |
| `instrumental_fwhm` | 4 | 8 | 12 | nm |
| `noise_base` | 0.002 | 0.005 | 0.008 | AU |
| `noise_signal_dep` | 0.005 | 0.01 | 0.015 | fraction |
| `artifact_prob` | 0.0 | 0.02 | 0.05 | probability |

---

## Usage Examples

### Basic Usage
```python
from examples.synthetic import SyntheticNIRSGenerator

generator = SyntheticNIRSGenerator(
    wavelength_start=1000,
    wavelength_end=2500,
    complexity="realistic",
    random_state=42
)

X, Y, E = generator.generate(n_samples=1000)
# X: (1000, 751) spectra
# Y: (1000, 5) concentrations
# E: (5, 751) pure component spectra
```

### Custom Parameters
```python
generator = SyntheticNIRSGenerator(complexity="realistic")
# Override specific parameters
generator.params["global_slope_mean"] = 0.02
generator.params["noise_base"] = 0.003

X, Y, E = generator.generate(n_samples=500)
```

### With Batch Effects
```python
X, Y, E, metadata = generator.generate(
    n_samples=600,
    include_batch_effects=True,
    n_batches=3,
    return_metadata=True
)

batch_ids = metadata["batch_ids"]  # Sample-to-batch mapping
```

---

## References

### Core Spectroscopy
1. **Beer, A.** (1852). Bestimmung der Absorption des rothen Lichts in farbigen Flüssigkeiten. *Annalen der Physik*, 162(5), 78-88.

2. **Workman Jr, J., & Weyer, L.** (2012). *Practical Guide and Spectral Atlas for Interpretive Near-Infrared Spectroscopy*. CRC Press. ISBN: 978-1439875254

3. **Burns, D. A., & Ciurczak, E. W.** (2007). *Handbook of Near-Infrared Analysis* (3rd ed.). CRC Press. ISBN: 978-0849373930

### Peak Shapes
4. **Olivero, J. J., & Longbothum, R. L.** (1977). Empirical fits to the Voigt line width. *Journal of Quantitative Spectroscopy and Radiative Transfer*, 17(2), 233-236.

### Preprocessing and Scatter Correction
5. **Rinnan, Å., Van Den Berg, F., & Engelsen, S. B.** (2009). Review of the most common pre-processing techniques for near-infrared spectra. *TrAC Trends in Analytical Chemistry*, 28(10), 1201-1222.

6. **Barnes, R. J., Dhanoa, M. S., & Lister, S. J.** (1989). Standard normal variate transformation and de-trending of near-infrared diffuse reflectance spectra. *Applied Spectroscopy*, 43(5), 772-777.

7. **Geladi, P., MacDougall, D., & Martens, H.** (1985). Linearization and scatter-correction for near-infrared reflectance spectra of meat. *Applied Spectroscopy*, 39(3), 491-500.

### Calibration Transfer
8. **Feudale, R. N., Woody, N. A., Tan, H., Myles, A. J., Brown, S. D., & Ferré, J.** (2002). Transfer of multivariate calibration models: a review. *Chemometrics and Intelligent Laboratory Systems*, 64(2), 181-192.

### Multivariate Calibration
9. **Martens, H., & Næs, T.** (1989). *Multivariate Calibration*. John Wiley & Sons. ISBN: 978-0471930471

### Compositional Data
10. **Aitchison, J.** (1986). *The Statistical Analysis of Compositional Data*. Chapman & Hall. ISBN: 978-0412280603

---

## Version History

- **v1.0.0** (2024): Initial implementation with Beer-Lambert model, Voigt profiles, complexity levels
- **v1.1.0** (2024): Added global slope effect, SyntheticRealComparator for real data comparison
