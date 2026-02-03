# Hierarchical Synthetic Spectra Fitting - Implementation Plan

## Problem Statement

Current approach (differential evolution on all parameters simultaneously) is flawed because:
1. Component spectral shapes don't match real data's absorption features
2. Optimizes secondary effects (noise, scatter) before primary signal (components)
3. Results in trivially distinguishable spectra (99.7% discriminator accuracy)
4. Global mean difference: -100% despite 98% correlation - shapes correlate but magnitudes are wrong

## Proposed Solution: Three-Stage Hierarchical Fitting

### Stage 1: Component Selection & Mixing (Baseline Shape)

**Goal**: Match the median/mean spectral shape using optimal component mixture from the full library

**Input**:
- Real dataset median spectrum (smooth reference)
- Component library (100+ predefined NIRS components)

**Method**: Sparse Non-Negative Least Squares (NNLS) or LASSO

```
minimize ||median_real - Σ_k w_k * ε_k(λ)||² + λ * ||w||₁
subject to: w_k >= 0 (concentrations must be positive)
```

**Implementation Steps**:

1. **Extract target signal**:
   - Compute median spectrum (robust to outliers)
   - Apply heavy smoothing (Savitzky-Golay, window=51) to remove noise
   - This is the "pure signal" we want to reproduce

2. **Build component matrix E**:
   - Load ALL available predefined components from ComponentLibrary
   - Interpolate each to match target wavelength grid
   - Normalize each component spectrum (unit area or peak normalization)

3. **Solve sparse mixture problem**:
   - Use `scipy.optimize.nnls` for basic NNLS
   - Or `sklearn.linear_model.Lasso(positive=True)`
   - Or `scipy.optimize.minimize` with L1 penalty and bounds

4. **Select top-K contributing components**:
   - Rank by weight magnitude
   - Keep components with w_k > threshold (e.g., 1% of max)
   - Typically 3-8 components should suffice

5. **Fine-tune weights**:
   - Re-fit NNLS with only selected components
   - Optionally add small wavelength shifts per component (±5nm)

**Output**:
- `selected_components: List[str]` - names of contributing components
- `base_weights: np.ndarray` - relative concentrations
- `fitted_median: np.ndarray` - reconstructed baseline spectrum

**Validation metric**:
- R² between smoothed real median and fitted median > 0.95
- RMSE < 5% of signal range

---

### Stage 2: Scatter & Multiplicative Effects

**Goal**: Match sample-to-sample variation structure (first-order effects)

**Input**:
- Real dataset X_real
- Fitted median from Stage 1
- SNV-corrected real data (to isolate scatter effects)

**Method**: Match statistics of scatter parameters

**Key observations from real NIRS data**:
- Scatter causes multiplicative + additive distortion: `X_i = α_i * X_true + β_i`
- Can be estimated by SNV decomposition
- Tilt varies with wavelength (wavelength-dependent scatter)

**Implementation Steps**:

1. **Estimate scatter parameters from real data**:
   ```python
   # For each sample, estimate alpha (scale) and beta (offset)
   for i in range(n_samples):
       # Linear fit: X_real[i] = alpha * median_real + beta + residual
       alpha_i = X_real[i].std() / median_real.std()  # scale ratio
       beta_i = X_real[i].mean() - alpha_i * median_real.mean()  # offset
   ```

2. **Compute scatter statistics**:
   - `alpha_mean, alpha_std` - multiplicative scatter distribution
   - `beta_mean, beta_std` - additive offset distribution
   - Correlation between alpha and beta

3. **Estimate wavelength-dependent tilt**:
   - Fit linear trend to each sample's deviation from median
   - Extract tilt coefficient distribution

4. **Match generator parameters**:
   - `scatter_alpha_std` ← std(alpha)
   - `scatter_beta_std` ← std(beta)
   - `tilt_std` ← std(tilt_coefficients)
   - `path_length_std` ← related to alpha variation

5. **Validate via PCA structure**:
   - Real data PCA loadings (PC1, PC2) should match synthetic
   - Explained variance ratios should align

**Output**:
- `scatter_alpha_std, scatter_beta_std, tilt_std, path_length_std`
- `baseline_amplitude` (if polynomial baseline needed)

**Validation metric**:
- PCA variance ratio correlation > 0.9
- Slope distribution KS-test p-value > 0.05

---

### Stage 3: High-Frequency Noise & Fine Details

**Goal**: Match noise characteristics at all frequency scales

**Input**:
- Real dataset
- Synthetic dataset from Stages 1-2

**Method**: Multi-scale noise analysis

**Implementation Steps**:

1. **Extract noise by subtracting smooth baseline**:
   ```python
   # Apply heavy smoothing to isolate signal
   smooth_signal = savgol_filter(spectrum, window=31, polyorder=2)
   noise_residual = spectrum - smooth_signal
   ```

2. **Analyze noise at multiple scales**:
   - **High-frequency** (1st derivative std): detector/shot noise
   - **Mid-frequency** (2nd derivative std): instrumental artifacts
   - **Wavelength-dependent**: heteroscedastic noise profile

3. **Estimate noise model parameters**:
   ```python
   # First-difference noise estimate
   diff_std = np.diff(X_real, axis=1).std(axis=0) / sqrt(2)

   # Signal-dependent vs constant noise
   # Model: noise_std = noise_base + noise_signal_dep * signal
   # Fit linear regression: diff_std vs signal level
   ```

4. **Match generator noise parameters**:
   - `noise_base` ← intercept of noise vs signal fit
   - `noise_signal_dep` ← slope of noise vs signal fit
   - `instrumental_fwhm` ← peak width in 2nd derivative analysis

5. **Artifact frequency**:
   - Detect spikes in real data
   - Count artifact rate per sample
   - Set `artifact_prob` accordingly

**Output**:
- `noise_base, noise_signal_dep, instrumental_fwhm, artifact_prob`

**Validation metric**:
- Noise level ratio (synthetic/real) in range [0.9, 1.1]
- First-difference distribution KS-test p-value > 0.05

---

## Implementation Architecture

### New Module Structure

```
nirs4all/synthesis/
├── fitting/
│   ├── __init__.py
│   ├── component_fitter.py    # Stage 1: sparse mixture fitting
│   ├── scatter_fitter.py      # Stage 2: scatter parameter estimation
│   ├── noise_fitter.py        # Stage 3: noise characterization
│   └── hierarchical_fitter.py # Orchestrates all stages
```

### API Design

```python
from nirs4all.synthesis.fitting import HierarchicalFitter

# Fit synthetic generator to real dataset
fitter = HierarchicalFitter(component_library="all")  # Use all 100+ components

result = fitter.fit(
    X_real=X_real,
    wavelengths=wavelengths,
    verbose=True
)

# Result contains fitted parameters at each stage
print(result.selected_components)  # ['water', 'protein', 'lipid', 'starch']
print(result.base_weights)         # [0.45, 0.25, 0.20, 0.10]
print(result.scatter_params)       # {'scatter_alpha_std': 0.048, ...}
print(result.noise_params)         # {'noise_base': 0.003, ...}

# Generate synthetic data with fitted parameters
dataset = result.generate(n_samples=1000)

# Or export fitted configuration
config = result.to_config()
dataset = nirs4all.generate.builder().from_config(config).build()
```

### Validation Framework

```python
# Built-in validation at each stage
report = fitter.validate(X_real, X_synthetic)

print(report.stage1_r2)              # Component fit quality
print(report.stage2_pca_correlation) # Scatter match
print(report.stage3_noise_ratio)     # Noise calibration
print(report.discriminator_accuracy) # Final distinguishability (target: <0.6)
print(report.overall_score)          # Weighted aggregate
```

---

## Key Technical Considerations

### 1. Component Library Expansion

Current library has ~30 predefined components. For robust fitting:
- Add more NIR-active compounds (100+ total)
- Include common matrix effects (particle scattering profiles)
- Allow user-provided custom components

### 2. Handling Different Spectral Ranges

Real datasets may use different wavelength ranges:
- Stage 1 fitting must interpolate components to match
- Some components may not have absorption in the range
- Auto-filter components with no signal in target range

### 3. Concentration Distribution Fitting

After Stage 1 determines base composition:
- Analyze concentration variation in real data via PCA
- Fit Dirichlet alpha parameters to match variation
- Or fit correlation matrix for correlated method

### 4. Iterative Refinement

Stages may need iteration:
- After Stage 3, recheck Stage 1 fit (noise may have biased estimate)
- Allow optional 2nd pass with noise-corrected median

---

## Success Criteria

1. **Discriminator accuracy < 60%** (ideally ~50%)
2. **Mean spectrum correlation > 0.98**
3. **PCA structure match (first 3 PCs correlation > 0.95)**
4. **Noise level within 10% of real data**
5. **Chemometric model performance similar on real vs synthetic**

---

## Additional Notes

- Add generation on uneven sampling (as real spectra)
