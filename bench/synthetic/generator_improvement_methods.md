# Synthetic NIRS Generator Improvement: Methods and Theory

This document provides a scientific explanation of the fitting methods used in the `generator_improvement.ipynb` notebook for matching synthetic NIRS spectra to real data.

## 1. Overview of the Fitting Pipeline

The fitting pipeline consists of five complementary approaches:

1. **Component-based fitting** (Step 1): Linear combination of predefined spectral components
2. **Envelope fitting** (Step 2): Component quantities for min/max variation bounds
3. **Band-based fitting** (Step 3): Direct optimization of Voigt profile parameters
4. **Noise/variance fitting** (Step 4): Environmental noise operator estimation
5. **PCA-based modeling** (Step 5): Data-driven variance decomposition

---

## 2. Component-Based Fitting (Steps 1-2)

### 2.1 Mathematical Formulation

The component-based approach models a spectrum as a **linear combination** of predefined spectral components:

$$
\mathbf{y} = \mathbf{E} \cdot \mathbf{c} + \mathbf{b} + \boldsymbol{\epsilon}
$$

Where:
- $\mathbf{y} \in \mathbb{R}^{n_\lambda}$ is the observed spectrum
- $\mathbf{E} \in \mathbb{R}^{n_\lambda \times n_c}$ is the design matrix (columns = component spectra)
- $\mathbf{c} \in \mathbb{R}^{n_c}$ is the concentration vector (unknown)
- $\mathbf{b} \in \mathbb{R}^{n_\lambda}$ is a polynomial baseline
- $\boldsymbol{\epsilon}$ is residual noise

### 2.2 Solving via Non-Negative Least Squares (NNLS)

Since concentrations must be non-negative (physical constraint), we solve:

$$
\min_{\mathbf{c} \geq 0} \| \mathbf{y} - \mathbf{E}\mathbf{c} - \mathbf{b} \|_2^2
$$

The **NNLS algorithm** (Lawson & Hanson, 1974) solves this efficiently:

1. Start with an active set of zero-constrained variables
2. Iteratively move variables between active and free sets
3. Solve unconstrained least squares on the free set
4. Check for constraint violations and update active set

**Why is it fast?**
- The design matrix $\mathbf{E}$ is precomputed from component library spectra
- NNLS for $m$ wavelengths and $n$ components is $O(mn^2)$
- With ~100 components and ~1000 wavelengths, this is <1ms per spectrum
- No iterative optimization of nonlinear parameters - just linear algebra

### 2.3 Predefined Components

Each component $e_i(\lambda)$ is a sum of Voigt profile bands based on spectroscopy literature:

$$
e_i(\lambda) = \sum_{j=1}^{n_b} A_{ij} \cdot V(\lambda - \lambda_{ij}^0, \sigma_{ij}, \gamma_{ij})
$$

The Voigt profile $V(x, \sigma, \gamma)$ is the convolution of Gaussian and Lorentzian:

$$
V(x, \sigma, \gamma) = \frac{\text{Re}[w(z)]}{\sigma\sqrt{2\pi}}, \quad z = \frac{x + i\gamma}{\sigma\sqrt{2}}
$$

Where $w(z)$ is the Faddeeva function.

### 2.4 Limitations

**Fixed peak shapes**: Predefined components have fixed band widths and positions from literature values. Real samples may have:
- Shifted peaks due to matrix effects
- Broader/narrower peaks from different physical states
- Different peak shapes (more/less Lorentzian character)

---

## 3. Band-Based Fitting (Step 3)

### 3.1 Direct Parameter Optimization

Instead of using predefined components with fixed parameters, we optimize band parameters directly:

$$
\hat{y}(\lambda) = \sum_{k=1}^{K} A_k \cdot G(\lambda - \lambda_k^0, \sigma_k) + \text{baseline}(\lambda)
$$

Where each band has optimizable parameters:
- $\lambda_k^0$: Center wavelength
- $\sigma_k$: Gaussian width (standard deviation)
- $A_k$: Amplitude

### 3.2 Optimization Method

We use **L-BFGS-B** (Limited-memory BFGS with Bounds):

$$
\min_{\boldsymbol{\theta}} \sum_{\lambda} \left( y(\lambda) - \hat{y}(\lambda; \boldsymbol{\theta}) \right)^2
$$

With box constraints:
- $\lambda_{\min} \leq \lambda_k^0 \leq \lambda_{\max}$
- $5 \text{ nm} \leq \sigma_k \leq 200 \text{ nm}$
- $A_k \geq 0$

### 3.3 Peak Detection Initialization

Initial band parameters are estimated using scipy's `find_peaks`:

1. Apply Savitzky-Golay smoothing to reduce noise
2. Detect peaks with prominence threshold
3. Estimate width from peak half-height width
4. Convert FWHM to Gaussian $\sigma$: $\sigma = \text{FWHM} / 2.355$

### 3.4 Current Limitations

The current implementation has several simplifications that limit fitting quality:

1. **Too few bands**: Only 3-12 bands are used, but real spectra may have 20-50 overlapping absorptions
2. **No Voigt profiles**: Using pure Gaussians, missing Lorentzian broadening
3. **Fixed band count during optimization**: Cannot add/remove bands dynamically
4. **Independent optimization**: Bands are optimized independently of physical constraints
5. **No negative bands**: Cannot model derivative-like features

---

## 4. PCA-Based Variance Modeling (Step 5)

### 4.1 Method

PCA decomposes the spectral variance into orthogonal components:

$$
\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T + \mathbf{\bar{x}}
$$

For generation, we:
1. Fit PCA on real data $\mathbf{X}_{\text{real}}$
2. Sample scores from $\mathcal{N}(0, \sigma_k^2)$ for $k = 1, ..., K$ PCs
3. Reconstruct: $\mathbf{x}_{\text{synth}} = \sum_{k=1}^{K} s_k \mathbf{v}_k + \mathbf{\bar{x}}$

### 4.2 Critical Point: PCA is Fit on Real Data Only

**The PCA approach does NOT use component-based generation.**

It is a purely data-driven method:
- **Input**: Real spectral data matrix $\mathbf{X}_{\text{real}}$
- **Output**: Principal components and variance structure
- **No connection** to the synthetic component library

This is why PCA achieves higher variance correlation (>0.97) - it directly captures the exact covariance structure of the real data, including:
- Complex multi-component mixtures
- Scattering effects
- Instrument-specific artifacts
- Any preprocessing artifacts

### 4.3 PCA vs Component-Based: Tradeoffs

| Aspect | Component-Based | PCA-Based |
|--------|----------------|-----------|
| Interpretability | High (known chemistry) | Low (abstract components) |
| Variance matching | Moderate | Excellent |
| Novel samples | Generates plausible chemistry | May generate artifacts |
| Extrapolation | Physics-guided | Unreliable outside data range |
| Parameter control | Direct (concentrations) | Indirect (scores) |

---

## 5. Noise/Variance Fitting (Step 4)

### 5.1 Noise Model

The noise model decomposes sample-to-sample variation into:

$$
x_{ij} = \bar{x}_j + \epsilon_{ij}^{\text{base}} + \epsilon_{ij}^{\text{signal}} + \alpha_i \cdot \lambda_j + \beta_i
$$

Where:
- $\epsilon_{ij}^{\text{base}} \sim \mathcal{N}(0, \sigma_{\text{base}}^2)$: Wavelength-independent noise
- $\epsilon_{ij}^{\text{signal}} \sim \mathcal{N}(0, (\sigma_{\text{dep}} \cdot \bar{x}_j)^2)$: Signal-dependent noise
- $\alpha_i$: Sample-specific slope
- $\beta_i$: Sample-specific offset

### 5.2 Parameter Estimation

1. **Baseline noise**: Estimated from first differences (high-frequency component)
   $$\sigma_{\text{base}} = \frac{\text{std}(\Delta x)}{\sqrt{2}}$$

2. **Signal-dependent noise**: Linear regression of local std vs. signal level

3. **Slope/offset variation**: Linear fit to each sample, then variance of slopes/offsets

### 5.3 Why Noise-Based Performs Worse Than PCA

The noise model assumes **additive, uncorrelated** variations. Real spectral variation includes:
- Correlated component concentration changes
- Scattering effects (multiplicative, wavelength-dependent)
- Temperature/humidity effects on specific bands
- Path length variations (multiplicative)

These are captured by PCA's covariance structure but not by simple noise operators.

---

## 6. Recommendations for Improvement

### 6.1 Band Fitting Improvements

1. **Increase band count**: Use 20-50 bands instead of 3-12
2. **Multi-resolution initialization**: Start with coarse peaks, refine with fine
3. **Voigt profiles**: Include Lorentzian broadening parameter
4. **Derivative constraints**: Allow negative amplitudes for derivative data
5. **Global optimization**: Use basin-hopping or differential evolution
6. **Hierarchical fitting**: Fit major peaks first, then residual peaks

### 6.2 Hybrid Approach

Combine component-based structure with PCA variance:

1. Generate base spectra using component library
2. Apply PCA-learned transformations for realistic variation
3. Add noise operators for fine-scale variation

$$
\mathbf{x}_{\text{synth}} = \mathbf{T}_{\text{PCA}} \cdot f(\mathbf{c}) + \boldsymbol{\epsilon}
$$

Where $f(\mathbf{c})$ generates component-based spectra and $\mathbf{T}_{\text{PCA}}$ applies PCA-learned variation patterns.

---

## References

1. Lawson, C.L., Hanson, R.J. (1974). Solving Least Squares Problems. Prentice-Hall.
2. Savitzky, A., Golay, M.J.E. (1964). Smoothing and Differentiation of Data by Simplified Least Squares Procedures. Analytical Chemistry, 36(8), 1627-1639.
3. Burns, D.A., Ciurczak, E.W. (2007). Handbook of Near-Infrared Analysis, Third Edition. CRC Press.
4. Workman, J., Weyer, L. (2012). Practical Guide and Spectral Atlas for Interpretive Near-Infrared Spectroscopy. CRC Press.
