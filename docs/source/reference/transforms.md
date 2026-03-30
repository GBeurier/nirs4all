# Transforms Reference

All transforms listed here are sklearn-compatible `TransformerMixin` classes that can be used as pipeline steps directly.

```{note}
Any sklearn `TransformerMixin` class (e.g., `StandardScaler`, `PCA`, `MinMaxScaler`) can also be used directly as a pipeline step.
```

## Usage in Pipeline

```python
from nirs4all.operators.transforms import SNV, SavitzkyGolay, FirstDerivative
from sklearn.cross_decomposition import PLSRegression

pipeline = [
    SNV(),
    SavitzkyGolay(window_length=11, polyorder=3, deriv=1),
    {"model": PLSRegression(n_components=10)},
]
```

All transforms below are imported from `nirs4all.operators.transforms` unless otherwise noted.

---

## Scatter Correction and Normalization

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `StandardNormalVariate` (alias: `SNV`) | `axis=1`, `with_mean=True`, `with_std=True`, `ddof=0` | Row-wise centering and scaling to remove scatter effects |
| `LocalStandardNormalVariate` | `window=11`, `pad_mode="reflect"` | Per-sample local normalization with a sliding window along features |
| `RobustStandardNormalVariate` (alias: `RNV`) | `axis=1`, `with_center=True`, `with_scale=True`, `k=1.4826` | Robust centering (median) and scaling (MAD) per sample |
| `MultiplicativeScatterCorrection` (alias: `MSC`) | `scale=True` | Corrects scatter by regressing each spectrum against the mean reference |
| `ExtendedMultiplicativeScatterCorrection` (alias: `EMSC`) | `degree=2`, `scale=True` | MSC extended with polynomial terms to model chemical and physical scatter |
| `AreaNormalization` | `method="sum"` | Normalizes each spectrum by its total area; method: `"sum"`, `"abs_sum"`, `"trapz"` |
| `Normalize` | `feature_range=(-1, 1)` | Range normalization or linalg normalization when range is `(-1, 1)` |
| `SimpleScale` | *(none)* | Column-wise min-max scaling to [0, 1] |

---

## Smoothing

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `SavitzkyGolay` | `window_length=11`, `polyorder=3`, `deriv=0`, `delta=1.0` | Savitzky-Golay polynomial smoothing and optional derivative computation |
| `Gaussian` | `order=2`, `sigma=1` | 1D Gaussian filter using `scipy.ndimage.gaussian_filter1d` |
| `WaveletDenoise` | `wavelet="db4"`, `level=5`, `mode="periodization"`, `threshold_mode="soft"`, `noise_estimator="median"` | Multi-level wavelet decomposition with thresholding for denoising |

---

## Derivatives

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `FirstDerivative` | `delta=1.0`, `edge_order=2` | First numerical derivative using `numpy.gradient` along the feature axis |
| `SecondDerivative` | `delta=1.0`, `edge_order=2` | Second numerical derivative using `numpy.gradient` applied twice |
| `NorrisWilliams` | `gap=5`, `segment=5`, `deriv=1`, `delta=1.0` | Gap derivative with segment smoothing (Norris-Williams method) |
| `Derivate` | `order=1`, `delta=1` | Nth-order derivative using `numpy.gradient` along axis 0 |

---

## Baseline Correction

### Simple

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `Baseline` | *(none)* | Removes the column-wise mean baseline from each spectrum |
| `Detrend` | `bp=0` | Removes linear trend using `scipy.signal.detrend`; `bp` sets breakpoints |

### pybaselines Wrappers

All baseline correction classes below wrap the [pybaselines](https://pybaselines.readthedocs.io/) library. They share a common interface via `PyBaselineCorrection`.

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `PyBaselineCorrection` | `method="asls"`, `**method_params` | General wrapper for any pybaselines method |
| `ASLSBaseline` | `lam=1e6`, `p=0.01`, `max_iter=50`, `tol=1e-3` | Asymmetric Least Squares baseline correction |
| `AirPLS` | `lam=1e6`, `max_iter=50`, `tol=1e-3` | Adaptive Iteratively Reweighted Penalized Least Squares |
| `ArPLS` | `lam=1e6`, `max_iter=50`, `tol=1e-3` | Asymmetrically Reweighted Penalized Least Squares |
| `IModPoly` | `poly_order=5`, `max_iter=250`, `tol=1e-3` | Improved Modified Polynomial baseline correction |
| `ModPoly` | `poly_order=5`, `max_iter=250`, `tol=1e-3` | Modified Polynomial baseline correction |
| `SNIP` | `max_half_window=40`, `decreasing=True`, `smooth_half_window=None` | Statistics-sensitive Non-linear Iterative Peak-clipping |
| `RollingBall` | `half_window=50`, `smooth_half_window=None` | Morphological rolling ball baseline estimation |
| `IASLS` | `lam=1e6`, `p=0.01`, `lam_1=1e-4`, `max_iter=50`, `tol=1e-3` | Improved Asymmetric Least Squares |
| `BEADS` | `lam_0=1.0`, `lam_1=1.0`, `lam_2=1.0`, `max_iter=50`, `tol=1e-2` | Baseline Estimation And Denoising with Sparsity |

---

## Orthogonalization

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `OSC` | `n_components=1`, `scale=True`, `method="dosc"` | Orthogonal Signal Correction -- removes Y-orthogonal variation from X (supervised, requires y) |
| `EPO` | `scale=True` | External Parameter Orthogonalization -- removes variation correlated with external parameters (e.g., temperature) |

```{note}
`OSC` requires `y` during `fit()`. `EPO` requires external parameters `d` during `fit()`, not `y`.
```

---

## Signal Conversion

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `ReflectanceToAbsorbance` | `min_value=1e-8`, `percent=False` | Converts reflectance to absorbance via Beer-Lambert law: A = -log10(R) |
| `ToAbsorbance` | `source_type="reflectance"`, `epsilon=1e-10`, `clip_negative=True` | Converts reflectance or transmittance to absorbance; supports percent inputs |
| `FromAbsorbance` | `target_type="reflectance"` | Converts absorbance back to reflectance or transmittance via 10^(-A) |
| `SignalTypeConverter` | `source_type="reflectance"`, `target_type="absorbance"`, `epsilon=1e-10` | General-purpose converter that auto-determines the conversion path |
| `KubelkaMunk` | `source_type="reflectance"`, `epsilon=1e-10` | Kubelka-Munk transformation for diffuse reflectance: F(R) = (1-R)^2 / (2R) |
| `LogTransform` | `base=e`, `offset=0.0`, `auto_offset=True`, `min_value=1e-8` | Elementwise logarithm with automatic handling of zeros/negatives |
| `PercentToFraction` | *(none)* | Divides by 100 to convert percentage values to fractional [0, 1] range |
| `FractionToPercent` | *(none)* | Multiplies by 100 to convert fractional values to percentage range |

---

## Wavelet Transforms and Feature Extraction

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `Wavelet` | `wavelet="haar"`, `mode="periodization"` | Single-level Discrete Wavelet Transform |
| `Haar` | *(none)* | Shortcut for `Wavelet(wavelet="haar")` |
| `WaveletFeatures` | `wavelet="db4"`, `max_level=5`, `n_coeffs_per_level=10` | Extracts statistical features from wavelet decomposition at multiple scales |
| `WaveletPCA` | `wavelet="db4"`, `max_level=4`, `n_components_per_level=3`, `whiten=True` | Multi-scale PCA on wavelet coefficients for compact multi-resolution representation |
| `WaveletSVD` | `wavelet="db4"`, `max_level=4`, `n_components_per_level=3` | Multi-scale SVD on wavelet coefficients (no centering, works for sparse data) |

---

## Feature Selection

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `CARS` | `n_components=10`, `n_sampling_runs=50`, `n_variables_ratio_start=1.0`, `n_variables_ratio_end=0.1`, `cv_folds=5`, `subset_ratio=0.8` | Competitive Adaptive Reweighted Sampling for wavelength selection (requires y) |
| `MCUVE` | `n_components=10`, `n_iterations=100`, `subset_ratio=0.8`, `threshold_method="auto"`, `threshold_percentile=99` | Monte-Carlo Uninformative Variable Elimination (requires y) |
| `FlexiblePCA` | `n_components=0.95`, `whiten=False`, `svd_solver="auto"` | PCA with flexible specification: int for count, float in (0,1) for variance ratio |
| `FlexibleSVD` | `n_components=0.95`, `algorithm="randomized"`, `n_iter=5` | Truncated SVD with flexible specification: int for count, float in (0,1) for variance ratio |

---

## Resampling and Cropping

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `Resampler` | `target_wavelengths`, `method="linear"`, `crop_range=None`, `fill_value=0.0` | Resamples spectral data to a new wavelength grid using scipy interpolation |
| `CropTransformer` | `start=0`, `end=None` | Crops features by index range [start:end] |
| `ResampleTransformer` | `num_samples` | Resamples each spectrum to a fixed number of points via linear interpolation |
| `FlattenPreprocessing` | `sources="all"` | Flattens 3D (samples, preprocessings, features) to 2D by concatenating preprocessing views |

---

## Target Transforms

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `IntegerKBinsDiscretizer` | *(imported from targets)* | Discretizes continuous targets into integer bins |
| `RangeDiscretizer` | *(imported from targets)* | Discretizes targets based on value ranges |

---

## See Also

- {doc}`../reference/augmentations` -- Data augmentation operators
- {doc}`../reference/pipeline_keywords` -- Pipeline keyword syntax reference
- {doc}`../reference/operator_catalog` -- Full operator catalog
