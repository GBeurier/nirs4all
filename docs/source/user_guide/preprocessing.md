# Preprocessing and TransformerMixin Reference

This page lists transformer-like components shipped in `nirs4all` under
`nirs4all.operators.augmentation` and `nirs4all.operators.transforms`,
and a short list of commonly used scikit-learn and scipy `TransformerMixin` classes that
are useful for NIRS preprocessing and machine learning pipelines.

**Context for NIRS analysis:**
Near-infrared spectroscopy (NIRS) data often requires specialized preprocessing to:
- Remove scatter effects from particle size, surface roughness, or path length variations (MSC, SNV)
- Eliminate baseline drift and offset (detrending, derivatives)
- Reduce noise while preserving spectral features (Savitzky–Golay, Gaussian smoothing)
- Normalize intensity variations across instruments or sessions
- Enhance absorption bands through derivatives or wavelet transforms

Notes:
- Each entry shows the class name, location, a 1-line purpose, the main
  init/usage signature and a 1-2 line usage note.
- Entries are intentionally concise for quick lookup by beginners.
- NIRS-specific recommendations and typical use cases are highlighted.

## nirs4all Augmentation (Data Augmenters)

**Why augmentation in NIRS?**
Data augmentation creates realistic spectral variations to improve model robustness and generalization,
especially useful when training data is limited. Augmenters simulate natural variability such as
instrument drift, temperature effects, slight wavelength shifts, or baseline fluctuations.

### Augmenter
`nirs4all.operators.augmentation.abc_augmenter.Augmenter`

Base class for augmentation; inherits `TransformerMixin, BaseEstimator`.

**Purpose**: Common API for augmenters (apply_on, random_state, copy).

**Signature**: `Augmenter(apply_on='samples', random_state=None, *, copy=True)`

**Usage**: Subclass and implement `augment(self, X, apply_on)`; used by other
augmentation operators. `apply_on` controls whether augmentation is per-sample or global.

### IdentityAugmenter
`nirs4all.operators.augmentation.abc_augmenter.IdentityAugmenter`

**Purpose**: No-op augmenter that returns input unchanged.

**Signature**: `IdentityAugmenter()`

**Usage**: Handy placeholder in augmentation pipelines or tests.

### Rotate_Translate
`nirs4all.operators.augmentation.random.Rotate_Translate`

**Purpose**: Apply a randomized rotation/translation deformation to spectra.

**Signature**: `Rotate_Translate(apply_on='samples', random_state=None, *, copy=True, p_range=2, y_factor=3)`

**Usage**: Augment samples or global spectrum; simulates baseline shifts and small
intensity variations typical of instrument drift or sample positioning effects.

**NIRS context**: Useful for training models robust to baseline instability.

### Random_X_Operation
`nirs4all.operators.augmentation.random.Random_X_Operation`

**Purpose**: Apply an elementwise random multiplicative (or custom) operation.

**Signature**: `Random_X_Operation(apply_on='global', random_state=None, *, copy=True, operator_func=operator.mul, operator_range=(0.97, 1.03))`

**Usage**: Simulate small random gain/attenuation across wavelengths or samples.

### Spline_Smoothing
`nirs4all.operators.augmentation.splines.Spline_Smoothing`

**Purpose**: Smooth each spectrum with a smoothing spline (UnivariateSpline).

**Signature**: `Spline_Smoothing(apply_on='samples', random_state=None, *, copy=True)`

**Usage**: Gentle smoothing alternative to Savitzky–Golay for augmentation. Uses default parameters in `augment` method.

### Spline_X_Perturbations
`nirs4all.operators.augmentation.splines.Spline_X_Perturbations`

**Purpose**: Perturb the x-axis via B-spline warping (wavelength shifts).

**Signature**: `Spline_X_Perturbations(apply_on='samples', random_state=None, *, copy=True, spline_degree=3, perturbation_density=0.05, perturbation_range=(-10, 10))`

**Usage**: Simulate non-linear wavelength distortions or small calibration shifts.

### Spline_Y_Perturbations
`nirs4all.operators.augmentation.splines.Spline_Y_Perturbations`

**Purpose**: Perturb the y-axis (intensity) using B-spline added distortions.

**Signature**: `Spline_Y_Perturbations(apply_on='samples', random_state=None, *, copy=True, spline_points=None, perturbation_intensity=0.005)`

**Usage**: Simulate low-frequency additive distortions or detector artifacts.

### Spline_X_Simplification
`nirs4all.operators.augmentation.splines.Spline_X_Simplification`

**Purpose**: Approximate a spectrum by fewer x-control points (downsampling).

**Signature**: `Spline_X_Simplification(apply_on='samples', random_state=None, *, copy=True, spline_points=None, uniform=False)`

**Usage**: Create simplified/resampled variants for augmentation or compression.

### Spline_Curve_Simplification
`nirs4all.operators.augmentation.splines.Spline_Curve_Simplification`

**Purpose**: Simplify the curve (y) via spline fitting on selected points.

**Signature**: `Spline_Curve_Simplification(apply_on='samples', random_state=None, *, copy=True, spline_points=None, uniform=False)`

**Usage**: Generate smoothed or lower-complexity spectra for augmentation.


## nirs4all Transformations (Preprocessing Transformers)

**Common NIRS preprocessing workflow:**
1. **Scatter correction** (MSC or SNV) — removes multiplicative effects from particle size
2. **Smoothing** (Savitzky–Golay or Gaussian) — reduces high-frequency noise
3. **Baseline correction** (detrending or derivatives) — removes offset and slope
4. **Scaling** (StandardScaler or normalization) — standardizes feature magnitudes

### StandardNormalVariate (SNV)
`nirs4all.operators.transforms.scalers.StandardNormalVariate`

Row-wise normalization technique commonly used in spectroscopy to remove scatter effects.
Each sample (row) is centered and scaled independently.

**Signature**: `StandardNormalVariate(axis=1, with_mean=True, with_std=True, ddof=0, copy=True)`

**Usage**: Standard SNV with `axis=1` (row-wise). For each sample: SNV = (X - mean(X)) / std(X).

**NIRS context**: Essential preprocessing step that removes multiplicative scatter effects.

### LocalStandardNormalVariate (LSNV)
`nirs4all.operators.transforms.scalers.LocalStandardNormalVariate`

Per-sample local normalization with a sliding window along features.

**Signature**: `LocalStandardNormalVariate(window=11, pad_mode='reflect', constant_values=0.0, copy=True)`

**Usage**: Local normalization for spectra with varying baseline along wavelengths.

### RobustStandardNormalVariate (RSNV)
`nirs4all.operators.transforms.scalers.RobustStandardNormalVariate`

Per-sample robust centering and scaling using median and MAD.

**Signature**: `RobustStandardNormalVariate(axis=1, with_center=True, with_scale=True, k=1.4826, copy=True)`

**Usage**: Robust alternative to SNV when outliers are present.

### Normalize
`nirs4all.operators.transforms.scalers.Normalize`

Normalize spectra either by linear feature range or by linalg norm.

**Signature**: `Normalize(feature_range=(-1, 1), *, copy=True)`

**Usage**: Choose user-defined min/max OR set `(-1, 1)` to use linalg normalization.

**NIRS context**: Useful after scatter correction to standardize intensity ranges.

### SimpleScale
`nirs4all.operators.transforms.scalers.SimpleScale`

Min-max scaling per-feature (columns) to `[0, 1]`.

**Signature**: `SimpleScale(copy=True)`

**Usage**: Fit computes min_ and max_; inverse_transform available. Use for simple per-wavelength scaling.

### Derivate
`nirs4all.operators.transforms.scalers.Derivate`

Compute Nth order derivative along samples axis using np.gradient.

**Signature**: `Derivate(order=1, delta=1, copy=True)`

**Usage**: Applies gradient along axis 0 (rows/samples, NOT features).

:::{warning}
**Axis warning**: This transformer operates on **axis=0**. For typical NIRS workflows
where you want derivatives along wavelengths, use `FirstDerivative` or `SecondDerivative` instead.
:::

### Wavelet / Haar
`nirs4all.operators.transforms.nirs.Wavelet`, `nirs4all.operators.transforms.nirs.Haar`

Single-level Discrete Wavelet Transform (DWT) on spectra.

**Signature**: `Wavelet(wavelet='haar', mode='periodization', *, copy=True)`; `Haar()` is shortcut.

**Usage**: Useful for denoising or multi-resolution features; returns resampled coeffs.

### SavitzkyGolay
`nirs4all.operators.transforms.nirs.SavitzkyGolay`

Smoothing and derivative calculation via Savitzky–Golay filter.

**Signature**: `SavitzkyGolay(window_length=11, polyorder=3, deriv=0, delta=1.0, *, copy=True)`

**Usage**: Common for smoothing and computing spectral derivatives while preserving peaks.

**NIRS context**: `deriv=1` or `deriv=2` computes 1st/2nd derivative — effective for baseline
correction and enhancing absorption bands. Window size should be odd; typical values 5–21.

### MultiplicativeScatterCorrection (MSC)
`nirs4all.operators.transforms.nirs.MultiplicativeScatterCorrection`

Correct additive/multiplicative scatter effects using a reference spectrum.

**Signature**: `MultiplicativeScatterCorrection(scale=True, *, copy=True)`

**Usage**: Call `fit`/`partial_fit` to compute parameters; then `transform` to correct spectra.

**NIRS context**: Essential for reflectance spectroscopy; reduces particle size effects and
path length variations. Often applied before smoothing or derivatives.

### ExtendedMultiplicativeScatterCorrection (EMSC)
`nirs4all.operators.transforms.nirs.ExtendedMultiplicativeScatterCorrection`

Extended MSC including polynomial terms to model chemical and physical light scattering effects.

**Signature**: `ExtendedMultiplicativeScatterCorrection(degree=2, scale=True, *, copy=True)`

**Usage**: Extends MSC by including polynomial terms for more complex scatter patterns.

### AreaNormalization
`nirs4all.operators.transforms.nirs.AreaNormalization`

Normalizes each spectrum by dividing by its total area.

**Signature**: `AreaNormalization(method='sum', *, copy=True)`

**Methods**: 'sum' (sum of values), 'abs_sum' (sum of absolute values), or 'trapz' (trapezoidal integration).

**Usage**: Removes intensity variations while preserving spectral shape.

### LogTransform
`nirs4all.operators.transforms.nirs.LogTransform`

Stable elementwise log transform with auto-offset handling and inverse.

**Signature**: `LogTransform(base=np.e, offset=0.0, auto_offset=True, min_value=1e-8, *, copy=True)`

**Usage**: Handles zeros/negatives safely; `fit` computes fitted offset for inverse.

### ReflectanceToAbsorbance
`nirs4all.operators.transforms.nirs.ReflectanceToAbsorbance`

Convert reflectance spectra to absorbance using Beer-Lambert law.

**Signature**: `ReflectanceToAbsorbance(min_value=1e-8, percent=False, *, copy=True)`

**Usage**: Applies A = -log10(R). Set `percent=True` if reflectance is in 0-100 range.

**NIRS context**: Fundamental transformation as absorbance is linearly related to concentration.

### FirstDerivative / SecondDerivative
`nirs4all.operators.transforms.nirs.FirstDerivative`, `nirs4all.operators.transforms.nirs.SecondDerivative`

Numerical derivatives along feature axis (wavelengths) using np.gradient.

**Signature**: `FirstDerivative(delta=1.0, edge_order=2, *, copy=True)` and `SecondDerivative(...)`

**Usage**: Common alternative to Savitzky–Golay derivatives; operates on **axis=1** (wavelengths).

**NIRS context**: 1st derivative removes baseline offset and slope; 2nd derivative removes
linear baseline and enhances narrow peaks. Recommended for NIRS spectral preprocessing.

### WaveletFeatures
`nirs4all.operators.transforms.nirs.WaveletFeatures`

Discrete Wavelet Transform feature extractor for spectral data.

**Signature**: `WaveletFeatures(wavelet='db4', max_level=5, n_coeffs_per_level=10, *, copy=True)`

**Usage**: Extracts statistical features from wavelet decomposition at multiple scales.

### WaveletPCA
`nirs4all.operators.transforms.nirs.WaveletPCA`

Multi-scale PCA on wavelet coefficients.

**Signature**: `WaveletPCA(wavelet='db4', max_level=4, n_components_per_level=3, whiten=True, *, copy=True)`

**Usage**: Combines multi-resolution analysis with decorrelation for compact representation.

### WaveletSVD
`nirs4all.operators.transforms.nirs.WaveletSVD`

Multi-scale SVD on wavelet coefficients.

**Signature**: `WaveletSVD(wavelet='db4', max_level=4, n_components_per_level=3, *, copy=True)`

**Usage**: Similar to WaveletPCA but uses SVD which works better for sparse data.

### Baseline
`nirs4all.operators.transforms.signal.Baseline`

Subtract per-feature mean (remove baseline) with inverse available.

**Signature**: `Baseline(*, copy=True)`

**Usage**: Call `partial_fit`/`fit` to compute `mean_`, then `transform` to subtract.

### Detrend
`nirs4all.operators.transforms.signal.Detrend`

Remove linear trend (optionally piecewise via breakpoints).

**Signature**: `Detrend(bp=0, *, copy=True)`

**Usage**: Uses `scipy.signal.detrend`; useful to remove baseline slope.

### Gaussian
`nirs4all.operators.transforms.signal.Gaussian`

1D Gaussian smoothing (ndimage gaussian_filter1d wrapper).

**Signature**: `Gaussian(order=2, sigma=1, *, copy=True)`

**Usage**: Low-pass smoothing; `order` parameter maps to derivative order in filter.

### CropTransformer
`nirs4all.operators.transforms.features.CropTransformer`

Select a wavelength (column) subrange from spectra.

**Signature**: `CropTransformer(start=0, end=None)`

**Usage**: `transform(X)` returns `X[:, start:end]`; end defaults to number of columns.

### ResampleTransformer
`nirs4all.operators.transforms.features.ResampleTransformer`

Resample each spectrum to a fixed number of samples (linear interpolation).

**Signature**: `ResampleTransformer(num_samples: int)`

**Usage**: Useful to standardize spectra with different lengths or to down/up-sample.

### FlattenPreprocessing
`nirs4all.operators.transforms.features.FlattenPreprocessing`

Flatten the preprocessing dimension of a 3D feature array.

**Signature**: `FlattenPreprocessing(sources='all')`

**Usage**: Transforms (samples, preprocessings, features) to (samples, preprocessings * features).
Useful after feature_augmentation when flattening multiple preprocessing views.

### PyBaselineCorrection
`nirs4all.operators.transforms.nirs.PyBaselineCorrection`

General baseline correction using the pybaselines library.

**Signature**: `PyBaselineCorrection(method='asls', *, copy=True, **method_params)`

**Available methods by category:**

- **Whittaker-based**: 'asls', 'iasls', 'airpls', 'arpls', 'drpls', 'iarpls', 'aspls', 'psalsa', 'derpsalsa'
- **Polynomial**: 'poly', 'modpoly', 'imodpoly', 'penalized_poly', 'loess', 'quant_reg'
- **Morphological**: 'mor', 'imor', 'mormol', 'amormol', 'rolling_ball', 'mwmv', 'tophat', 'mpspline', 'jbcd'
- **Spline**: 'mixture_model', 'irsqr', 'corner_cutting', 'pspline_asls', etc.
- **Smooth**: 'noise_median', 'snip', 'swima', 'ipsa'

**Usage**: `PyBaselineCorrection(method='airpls', lam=1e5)` for airPLS baseline correction.

**Convenience aliases**: `AirPLS`, `ArPLS`, `IModPoly`, `ModPoly`, `SNIP`, `RollingBall`


## Useful scikit-learn TransformerMixin Classes for NIRS

Below are common `sklearn` transformers you can use in pipelines with nirs4all
outputs. Each entry is short — their docstrings and parameters are in scikit-learn docs.

### StandardScaler
`sklearn.preprocessing.StandardScaler`

Zero mean, unit variance scaling per feature (column).

**Usage**: `StandardScaler(with_mean=True, with_std=True)`; apply before
models that assume standardized inputs (SVM, linear models, NN).

**NIRS context**: Use after scatter correction and smoothing; helps models when wavelengths
have different variance. Note: operates column-wise (per wavelength), complementing row-wise SNV.

### RobustScaler
`sklearn.preprocessing.RobustScaler`

Scale using median and IQR; robust to outliers.

**Usage**: `RobustScaler(with_centering=True, with_scaling=True)`; good for
real-world noisy spectral datasets with occasional spikes or detector artifacts.

### MinMaxScaler
`sklearn.preprocessing.MinMaxScaler`

Rescale features to a fixed range (common: `[0, 1]`).

**Usage**: `MinMaxScaler(feature_range=(0, 1))`; useful for bounded models.

### FunctionTransformer
`sklearn.preprocessing.FunctionTransformer`

Wrap arbitrary numpy functions as transformers.

**Usage**: `FunctionTransformer(np.log1p)` or custom lambda for quick operations.

### PCA
`sklearn.decomposition.PCA`

Dimensionality reduction preserving variance; common for denoising.

**Usage**: `PCA(n_components=10)`; often used after smoothing/MSC.

**NIRS context**: NIRS spectra are highly collinear (neighboring wavelengths correlate);
PCA captures most variance in 5–20 components, reducing overfitting and computation time.

### TruncatedSVD
`sklearn.decomposition.TruncatedSVD`

Similar to PCA but works on sparse matrices; useful for large features.

### PolynomialFeatures
`sklearn.preprocessing.PolynomialFeatures`

Create interaction / polynomial basis features; rarely used directly on raw spectra but useful in classical chemometrics pipelines.

### QuantileTransformer
`sklearn.preprocessing.QuantileTransformer`

Map features to a uniform or normal distribution; useful for robust scaling across samples.

### PowerTransformer
`sklearn.preprocessing.PowerTransformer`

Stabilize variance and make data more Gaussian-like (Yeo-Johnson/Box-Cox).


## Useful SciPy Functions for NIRS

These common SciPy helpers are used across nirs4all transformers or are handy
when building custom preprocessing steps.

### scipy.signal.savgol_filter
Smoothing and derivative calculation (Savitzky–Golay).

**Signature**: `savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0)`

**Usage**: Low-noise smoothing with preservation of peak shapes; used in `SavitzkyGolay` class.

### scipy.signal.detrend
Remove linear (or piecewise linear) trend from data.

**Signature**: `detrend(data, bp=0)`

**Usage**: Quick baseline slope removal; used in `Detrend` transformer.

### scipy.ndimage.gaussian_filter1d
Apply 1D Gaussian smoothing along the feature axis.

**Signature**: `gaussian_filter1d(input, sigma, order=0)`

**Usage**: Fast separable Gaussian smoothing; used in `Gaussian` transformer.

### scipy.signal.resample
Fourier-based resampling to a new number of samples.

**Signature**: `resample(x, num, axis=0)`

**Usage**: Useful for changing spectral sampling rate (aliasing caveats apply).

### scipy.interpolate.UnivariateSpline
Smooth spline fitting and evaluation for 1D signals.

**Signature**: `UnivariateSpline(x, y, s=0, k=3)`

**Usage**: Smoothing or interpolation; used by spline augmenters and smoothing.

### scipy.interpolate.interp1d
Linear or higher-order 1D interpolation between points.

**Signature**: `interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')`

**Usage**: Used by `ResampleTransformer` for simple resampling between wavelengths.

### scipy.signal.medfilt
Apply sliding-window median filter for impulse noise removal.

**Signature**: `medfilt(x, kernel_size=3)`

**Usage**: Robust to spikes; occasionally used as pre-step before smoothing.

### scipy.signal.find_peaks
Detect peaks in 1D signals (prominence, height, width options).

**Signature**: `find_peaks(x, height=None, prominence=None, width=None)`

**Usage**: Feature extraction (peak locations/heights) from spectra.


## Quick Recipes (Beginners)

**Typical simple pipeline**: smoothing → scatter correction → scaling → PCA → model
- e.g. `SavitzkyGolay → MultiplicativeScatterCorrection → StandardScaler → PCA`

**Augmentation examples**: `Rotate_Translate`, `Spline_Y_Perturbations`, and `Random_X_Operation`
are useful for building robust training sets.


## Notes and Caveats

### Axis Conventions — IMPORTANT

- Most transformers assume `X.shape == (n_samples, n_features)` where features = wavelengths.
- **Exception**: `Derivate` (in `scalers.py`) computes gradients along **axis=0** (samples),
  which is unusual for NIRS. It is kept for legacy support.

For standard spectral derivatives along wavelengths, use:
- `FirstDerivative` or `SecondDerivative` (in `nirs.py`) — these use **axis=1** (features/wavelengths)
- `SavitzkyGolay` with `deriv=1` or `deriv=2` — also operates along wavelengths

:::{warning}
**Verify before chaining derivatives**: Mixing `Derivate` with other derivative methods will
produce incorrect results due to different axis conventions.
:::

### Other Notes
- Most nirs4all transformers raise on sparse inputs and provide `inverse_transform`
  only when it is meaningful (check class methods).
- For advanced usage, consult the specific source file in `nirs4all/operators/` for
  parameter details and behavior.
- When using augmenters, set `random_state` for reproducibility in experiments.

---

## See Also

- {doc}`/reference/operator_catalog` - Complete listing of all operators
- {doc}`/reference/pipeline_syntax` - Full pipeline syntax reference
- {doc}`branching_merging` - Using preprocessors in branched pipelines
- {doc}`/examples/index` - Working examples including preprocessing

**Example files:**
- `examples/user/03_preprocessing/U09_preprocessing_basics.py` - Preprocessing example
- `examples/user/03_preprocessing/U10_feature_augmentation.py` - Feature augmentation

**External documentation:**
- [scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [SciPy Signal Processing](https://docs.scipy.org/doc/scipy/reference/signal.html)
