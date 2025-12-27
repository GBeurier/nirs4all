# Preprocessing Cheatsheet

Quick reference for NIR preprocessing selection by model type.

## Classical ML (scikit-learn + libraries)

| Model | Task | Works Well | Avoid |
|-------|------|-----------|-------|
| **PLS (PLS-R, PLS-DA)** | R/C | Mean-centering; SNV/MSC/EMSC; mild SG smoothing; **1st derivative**; detrend; wavelength masking | Over-aggressive **2nd deriv** with short windows; no scatter correction; leaking preprocessing |
| **OPLS / Kernel PLS / Local PLS** | R/C | Same as PLS; for Local/LW: **distance-aware scaling** (SNV); for Kernel: global **standardization** | Using raw unscaled spectra for neighbor search; noisy high-order derivatives |
| **PCR** | R | Center + (often) autoscale; SG smoothing; baseline removal before PCA; retain PCs via CV | PCA on raw uncorrected scatter; keeping too many PCs; PCA fitted outside CV |
| **Linear / Ridge / Lasso / ElasticNet** | R/C | Center + scale; SNV/MSC; mild SG; band selection to reduce collinearity | No scaling; strong noise amplification via derivatives |
| **LDA / QDA** | C | Dimension reduction first (PCA/PLS scores); center + scale; SNV/MSC; outlier control | Training directly on thousands of wavelengths; uncorrected batch effects |
| **k-NN** | R/C | **Per-feature scaling** or SNV; baseline/scatter correction; smoothed **1st deriv**; band selection | Raw unscaled spectra; high-order noisy derivatives; very high-D |
| **SVM / SVR** | R/C | **Standardization**; SNV/MSC; SG + **1st deriv**; band selection or PCA/PLS scores | No scaling; feeding entire noisy spectrum; aggressive 2nd deriv |
| **Decision Tree** | R/C | SG smoothing; SNV/MSC if strong scatter; **band/bin selection** | Per-feature standardization; high-order noisy derivatives |
| **Random Forest** | R/C | SG smoothing; SNV/MSC helpful; **band/bin selection**; remove obvious noise regions | Standardization per wavelength; over-derivation amplifying noise |
| **Gradient Boosting** | R/C | As RF; plus outlier trimming; modest feature reduction; early stopping | Per-feature standardization; noisy 2nd deriv; no denoising |
| **XGBoost / LightGBM / CatBoost** | R/C | SG smoothing; SNV/MSC; band/bin selection; remove artifacts; tune regularization | Standardization per wavelength; noisy derivatives |
| **TabPFN** | R/C | Minimal scaling needed (internal z-score); **mask noisy/irrelevant bands**; band/bin reduction | Manual re-scaling; feeding artifact regions; extreme dimensionality |

**Legend**: R = Regression, C = Classification

## Neural Networks

| Model | Task | Works Well | Avoid |
|-------|------|-----------|-------|
| **MLP** | R/C | **Standardization** or min-max; mean-centering/SNV; baseline removal; SG smoothing; band/PCA/PLS scores | Raw unscaled spectra; high-D collinearity with small N; noisy derivatives |
| **1D CNN** | R/C | Input scaling to stable range; SNV/MSC; SG smoothing; **1st deriv** optional; data augmentation | No normalization; over-smoothed spectra; pure 2nd deriv without smoothing |
| **RNN (LSTM/GRU)** | R/C | Standardization; mean-centering; baseline removal; moderate smoothing; **downsampling/binning** | Very long raw sequences with noise; unscaled inputs that saturate gates |
| **Transformers** | R/C | Standardization; positional encoding; SNV/MSC; denoising or **patch/bin tokens** | Raw baselines/scatter; very long token sequences; no normalization |
| **Vision backbones (transfer)** | C/R | Match pretrained **input normalization**; consistent encoding; SNV/MSC before encoding | Mismatch of expected scale; noisy encodings |

## Quick Rules of Thumb

::::{grid} 2
:gutter: 3

:::{grid-item}
### When to Apply What

| Condition | Action |
|-----------|--------|
| Scatter/baseline present | **SNV/MSC/EMSC + detrend** |
| Noisy spectra | **SG smoothing**; conservative derivatives |
| High-D with small N | **Band/bin selection** or **PCA/PLS scores** |
| Model needs scaling | SVM, k-NN, linear, MLP, RNN, Transformers |
| Model dislikes scaling | Trees, RF, Boosting |
:::

:::{grid-item}
### Derivative Guidelines

| Derivative | Best For | Caution |
|------------|----------|---------|
| **1st derivative** | PLS, SVM, k-NN, CNN | Use with smoothing |
| **2nd derivative** | Overlapping peaks | Only with adequate SNR |
| **SG derivative** | Most applications | Window 11-21, polyorder 2-3 |
:::
::::

## Preprocessing Chains

### Minimal Robust (3 steps)
```python
[SNV(), SavitzkyGolay(window_length=15, deriv=1), StandardScaler()]
```

### Standard (4 steps)
```python
[MSC(), SavitzkyGolay(window_length=17), FirstDerivative(), RobustScaler()]
```

### Multi-view (for deep learning)
```python
[
    {"branch": [
        [SNV(), SavitzkyGolay()],
        [MSC(), FirstDerivative()],
        [SNV(), SecondDerivative()],
    ]},
    {"merge": "features"}
]
```

## Parameter Recommendations

| Operator | Parameter | Default Range | Notes |
|----------|-----------|---------------|-------|
| **SavitzkyGolay** | `window_length` | 11-21 | Must be odd |
| | `polyorder` | 2-3 | Higher = less smoothing |
| | `deriv` | 0-2 | 0=smooth, 1=1st deriv |
| **FirstDerivative** | `delta` | 1.0 | Wavelength spacing |
| **Gaussian** | `sigma` | 1-3 | Higher = more smoothing |
| **RSNV/LSNV** | window | 25-75 points | Depends on resolution |
| **Wavelet** | level | 3-5 | Decomposition levels |

## Warning Signs

:::{warning}
**🚨 Preprocessing Anti-patterns**

- ❌ **SNV + MSC together** - Redundant scatter correction
- ❌ **SavGol then Detrend** - SG already removes trends
- ❌ **Derivative before scatter correction** - Amplifies artifacts
- ❌ **Very short SG window (< 7)** - Insufficient smoothing for derivatives
- ❌ **Global scaler on SNV-normalized channels** - Double normalization
- ❌ **Fitting preprocessing on full dataset** - Data leakage!
:::

## See Also

- {doc}`overview` - Comprehensive preprocessing guide
- {doc}`handbook` - In-depth theory and advanced techniques
- {doc}`/reference/operator_catalog` - Complete operator reference
