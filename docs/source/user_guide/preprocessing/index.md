# Preprocessing

This section covers spectral preprocessing techniques for NIRS data.

```{toctree}
:maxdepth: 2

overview
cheatsheet
handbook
snv
resampler
transfer_preprocessing_selector_cheatsheet
```

## Overview

Preprocessing is a critical step in NIRS data analysis. NIRS4ALL provides a comprehensive set of preprocessing operators that are sklearn-compatible and can be chained in pipelines.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 📖 Preprocessing Overview
:link: overview
:link-type: doc

Comprehensive guide to available preprocessing techniques, operators, and when to use them.

+++
{bdg-primary}`Start Here`
:::

:::{grid-item-card} 📋 Cheatsheet
:link: cheatsheet
:link-type: doc

Quick reference for preprocessing selection by model type.

+++
{bdg-info}`Quick Ref`
:::

:::{grid-item-card} 📚 Handbook
:link: handbook
:link-type: doc

In-depth guide with theory, advanced techniques, and multi-layer preprocessing.

+++
{bdg-success}`Complete`
:::


:::{grid-item-card} 📐 Resampling
:link: resampler
:link-type: doc

Wavelength resampling and interpolation techniques.

+++
{bdg-warning}`Advanced`
:::

::::

## Available Preprocessing Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **SNV** | Standard Normal Variate | Scatter correction |
| **MSC** / **EMSC** | Multiplicative Scatter Correction | Reference-based scatter correction |
| **Derivatives** | Savitzky-Golay, First/Second | Baseline removal, peak enhancement |
| **NorrisWilliams** | Gap derivative with segment smoothing | Noise-robust derivatives |
| **WaveletDenoise** | Multi-level wavelet denoising | Noise reduction with frequency selectivity |
| **OSC** | Orthogonal Signal Correction | Remove Y-orthogonal variation |
| **EPO** | External Parameter Orthogonalization | Remove external parameter influence |
| **Detrend** | Polynomial detrending | Linear/quadratic baseline removal |
| **Baseline** | ALS, AirPLS, ArPLS, SNIP, etc. | Advanced baseline correction |
| **Normalization** | Min-max, area, vector | Scale standardization |
| **Smoothing** | Gaussian, moving average | Noise reduction |

## Quick Example

```python
from nirs4all.operators.transforms import SNV, SavitzkyGolay, NorrisWilliams, WaveletDenoise

pipeline = [
    SNV(),                                    # Scatter correction
    SavitzkyGolay(window_length=15, deriv=1), # First derivative
    # ... rest of pipeline
]

# Or use newer transforms
pipeline = [
    WaveletDenoise(wavelet='db4', level=3),    # Wavelet denoising
    NorrisWilliams(gap=5, segment=5, deriv=1), # Norris-Williams gap derivative
    # ... rest of pipeline
]
```

## See Also

- {doc}`/reference/operator_catalog` - Complete operator reference
- {doc}`snv` - Detailed SNV documentation
- {doc}`transfer_preprocessing_selector_cheatsheet` - Transfer learning preprocessing
