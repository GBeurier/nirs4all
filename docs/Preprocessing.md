# Preprocessing and TransformerMixin reference (nirs4all)

This page lists transformer-like components shipped in `nirs4all` under
`nirs4all.operators.augmentation` and `nirs4all.operators.transformations`,
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

## nirs4all augmentation (data augmenters)

**Why augmentation in NIRS?**
Data augmentation creates realistic spectral variations to improve model robustness and generalization,
especially useful when training data is limited. Augmenters simulate natural variability such as
instrument drift, temperature effects, slight wavelength shifts, or baseline fluctuations.

- `Augmenter` — `nirs4all.operators.augmentation.abc_augmenter.Augmenter`
  - Base class for augmentation; inherits `TransformerMixin, BaseEstimator`.
  - Purpose: common API for augmenters (apply_on, random_state, copy).
  - Key signature: `Augmenter(apply_on='samples', random_state=None, *, copy=True)`
  - Usage: subclass and implement `augment(self, X, apply_on)`; used by other
    augmentation operators. `apply_on` controls whether augmentation is per-sample or global.

- `IdentityAugmenter` — `...abc_augmenter.IdentityAugmenter`
  - Purpose: no-op augmenter that returns input unchanged.
  - Signature: `IdentityAugmenter()`
  - Usage: handy placeholder in augmentation pipelines or tests.

- `Rotate_Translate` — `...augmentation.random.Rotate_Translate`
  - Purpose: apply a randomized rotation/translation deformation to spectra.
  - Signature: `Rotate_Translate(apply_on='samples', random_state=None, *, copy=True, p_range=2, y_factor=3)`
  - Usage: augment samples or global spectrum; simulates baseline shifts and small
    intensity variations typical of instrument drift or sample positioning effects.
  - **NIRS context**: useful for training models robust to baseline instability.

- `Random_X_Operation` — `...augmentation.random.Random_X_Operation`
  - Purpose: apply an elementwise random multiplicative (or custom) op.
  - Signature: `Random_X_Operation(apply_on='global', random_state=None, *, copy=True, operator_func=operator.mul, operator_range=(0.97,1.03))`
  - Usage: simulate small random gain/attenuation across wavelengths or samples.

- `Spline_Smoothing` — `...augmentation.splines.Spline_Smoothing`
  - Purpose: smooth each spectrum with a smoothing spline (UnivariateSpline).
  - Signature: `Spline_Smoothing()` (uses default parameters in `augment`).
  - Usage: gentle smoothing alternative to Savitzky–Golay for augmentation.

- `Spline_X_Perturbations` — `...augmentation.splines.Spline_X_Perturbations`
  - Purpose: perturb the x-axis via B‑spline warping (wavelength shifts).
  - Signature: `Spline_X_Perturbations(apply_on='samples', random_state=None, *, copy=True, spline_degree=3, perturbation_density=0.05, perturbation_range=(-10,10))`
  - Usage: simulate non-linear wavelength distortions or small calibration shifts.

- `Spline_Y_Perturbations` — `...augmentation.splines.Spline_Y_Perturbations`
  - Purpose: perturb the y-axis (intensity) using B‑spline added distortions.
  - Signature: `Spline_Y_Perturbations(apply_on='samples', random_state=None, *, copy=True, spline_points=None, perturbation_intensity=0.005)`
  - Usage: simulate low-frequency additive distortions or detector artifacts.

- `Spline_X_Simplification` — `...augmentation.splines.Spline_X_Simplification`
  - Purpose: approximate a spectrum by fewer x-control points (downsampling).
  - Signature: `Spline_X_Simplification(apply_on='samples', random_state=None, *, copy=True, spline_points=None, uniform=False)`
  - Usage: create simplified/resampled variants for augmentation or compression.

- `Spline_Curve_Simplification` — `...augmentation.splines.Spline_Curve_Simplification`
  - Purpose: simplify the curve (y) via spline fitting on selected points.
  - Signature: `Spline_Curve_Simplification(apply_on='samples', random_state=None, *, copy=True, spline_points=None, uniform=False)`
  - Usage: generate smoothed or lower-complexity spectra for augmentation.


## nirs4all transformations (preprocessing transformers)

**Common NIRS preprocessing workflow:**
1. **Scatter correction** (MSC or SNV) — removes multiplicative effects from particle size
2. **Smoothing** (Savitzky–Golay or Gaussian) — reduces high-frequency noise
3. **Baseline correction** (detrending or derivatives) — removes offset and slope
4. **Scaling** (StandardScaler or normalization) — standardizes feature magnitudes
5. **Dimensionality reduction** (PCA) — reduces correlated wavelength information

- `Normalize` — `nirs4all.operators.transformations.scalers.Normalize`
  - Purpose: normalize spectra either by linear feature range or by linalg norm.
  - Signature: `Normalize(feature_range=(-1,1), *, copy=True)`
  - Usage: choose user-defined min/max OR set `(-1,1)` to use linalg normalization.
  - **NIRS context**: useful after scatter correction to standardize intensity ranges.

-- `SimpleScale` — `...transformations.scalers.SimpleScale`
  - Purpose: min-max scaling per-feature (columns) to ` [0,1]`.
  - Signature: `SimpleScale(copy=True)` (fit computes min_ and max_)
  - Usage: inverse_transform available; use for simple per-wavelength scaling.

- `Derivate` — `...transformations.scalers.Derivate`
  - Purpose: compute Nth order derivative along samples axis using np.gradient.
  - Signature: `Derivate(order=1, delta=1, copy=True)`
  - Usage: applies gradient along axis 0 (rows/samples, NOT features).
  - **⚠️ Axis warning**: this transformer operates on axis=0; for typical NIRS workflows
    where you want derivatives along wavelengths, use `FirstDerivative` or `SecondDerivative` instead.

- `Wavelet` / `Haar` — `...transformations.nirs.Wavelet`, `Haar`
  - Purpose: single-level discrete wavelet transform (DWT) on spectra.
  - Signature: `Wavelet(wavelet='haar', mode='periodization', *, copy=True)`; `Haar()` is shortcut.
  - Usage: useful for denoising or multi-resolution features; returns resampled coeffs.

- `SavitzkyGolay` — `...transformations.nirs.SavitzkyGolay`
  - Purpose: smoothing and derivative calculation via Savitzky–Golay filter.
  - Signature: `SavitzkyGolay(window_length=11, polyorder=3, deriv=0, delta=1.0, *, copy=True)`
  - Usage: common for smoothing and computing spectral derivatives while preserving peaks.
  - **NIRS context**: `deriv=1` or `deriv=2` computes 1st/2nd derivative — effective for baseline
    correction and enhancing absorption bands. Window size should be odd; typical values 5–21.

- `MultiplicativeScatterCorrection` (MSC) — `...transformations.nirs.MultiplicativeScatterCorrection`
  - Purpose: correct additive/multiplicative scatter effects using a reference spectrum.
  - Signature: `MultiplicativeScatterCorrection(scale=True, *, copy=True)`
  - Usage: call `fit`/`partial_fit` to compute parameters; then `transform` to correct spectra.
  - **NIRS context**: essential for reflectance spectroscopy; reduces particle size effects and
    path length variations. Often applied before smoothing or derivatives.

- `LogTransform` — `...transformations.nirs.LogTransform`
  - Purpose: stable elementwise log transform with auto-offset handling and inverse.
  - Signature: `LogTransform(base=np.e, offset=0.0, auto_offset=True, min_value=1e-8, *, copy=True)`
  - Usage: handles zeros/negatives safely; `fit` computes fitted offset for inverse.

- `FirstDerivative` / `SecondDerivative` — `...transformations.nirs.FirstDerivative`, `SecondDerivative`
  - Purpose: numerical derivatives along feature axis (wavelengths) using np.gradient.
  - Signature: `FirstDerivative(delta=1.0, edge_order=2, *, copy=True)` and `SecondDerivative(...)`
  - Usage: common alternative to Savitzky–Golay derivatives; operates on axis=1 (wavelengths).
  - **NIRS context**: 1st derivative removes baseline offset and slope; 2nd derivative removes
    linear baseline and enhances narrow peaks. Recommended for NIRS spectral preprocessing.

- `Baseline` — `...transformations.signal.Baseline`
  - Purpose: subtract per-feature mean (remove baseline) with inverse available.
  - Signature: `Baseline(*, copy=True)`
  - Usage: call `partial_fit`/`fit` to compute `mean_`, then `transform` to subtract.

- `Detrend` — `...transformations.signal.Detrend`
  - Purpose: remove linear trend (optionally piecewise via breakpoints).
  - Signature: `Detrend(bp=0, *, copy=True)`
  - Usage: uses `scipy.signal.detrend`; useful to remove baseline slope.

- `Gaussian` — `...transformations.signal.Gaussian`
  - Purpose: 1D gaussian smoothing (ndimage gaussian_filter1d wrapper).
  - Signature: `Gaussian(order=2, sigma=1, *, copy=True)`
  - Usage: low-pass smoothing; `order` parameter maps to derivative order in helper.

- `CropTransformer` — `...transformations.features.CropTransformer`
  - Purpose: select a wavelength (column) subrange from spectra.
  - Signature: `CropTransformer(start=0, end=None)`
  - Usage: `transform(X)` returns `X[:, start:end]`; end defaults to number of columns.

- `ResampleTransformer` — `...transformations.features.ResampleTransformer`
  - Purpose: resample each spectrum to a fixed number of samples (linear interp).
  - Signature: `ResampleTransformer(num_samples: int)`
  - Usage: useful to standardize spectra with different lengths or to down/up-sample.

- `StandardNormalVariate` - `...transformations.features.ResampleTransformer`
  - Purpose: normalizes each spectrum (sample) individually to remove multiplicative scatter effects.
  - Signature: `StandardNormalVariate(axis=1, with_mean=True, with_std=True, ddof=0)`
  - Usage: By default, SNV operates row-wise (axis=1), each spectrum (row) is centered and scaled independently.

## Useful scikit-learn TransformerMixin classes for NIRS (quick reference)

Below are common `sklearn` transformers you can use in pipelines with nirs4all
outputs. Each entry is short — their docstrings and parameters are in scikit-learn docs.

- `StandardScaler` (`sklearn.preprocessing.StandardScaler`)
  - Purpose: zero mean, unit variance scaling per feature (column).
  - Typical usage: `StandardScaler(with_mean=True, with_std=True)`; apply before
    models that assume standardized inputs (SVM, linear models, NN).
  - **NIRS context**: use after scatter correction and smoothing; helps models when wavelengths
    have different variance. Note: operates column-wise (per wavelength), complementing row-wise SNV.

- `RobustScaler` (`sklearn.preprocessing.RobustScaler`)
  - Purpose: scale using median and IQR; robust to outliers.
  - Typical usage: `RobustScaler(with_centering=True, with_scaling=True)`; good for
    real-world noisy spectral datasets with occasional spikes or detector artifacts.

-- `MinMaxScaler` (`sklearn.preprocessing.MinMaxScaler`)
  - Purpose: rescale features to a fixed range (common: ` [0,1]`).
  - Typical usage: `MinMaxScaler(feature_range=(0,1))`; useful for bounded models.

- `FunctionTransformer` (`sklearn.preprocessing.FunctionTransformer`)
  - Purpose: wrap arbitrary numpy functions as transformers.
  - Typical usage: `FunctionTransformer(np.log1p)` or custom lambda for quick ops.

- `PCA` (`sklearn.decomposition.PCA`)
  - Purpose: dimensionality reduction preserving variance; common for denoising.
  - Typical usage: `PCA(n_components=10)`; often used after smoothing/MSC.
  - **NIRS context**: NIRS spectra are highly collinear (neighboring wavelengths correlate);
    PCA captures most variance in 5–20 components, reducing overfitting and computation time.

- `TruncatedSVD` (`sklearn.decomposition.TruncatedSVD`)
  - Purpose: similar to PCA but works on sparse matrices; useful for large features.

- `PolynomialFeatures` (`sklearn.preprocessing.PolynomialFeatures`)
  - Purpose: create interaction / polynomial basis features; rarely used directly on raw spectra but useful in classical chemometrics pipelines.

- `QuantileTransformer` (`sklearn.preprocessing.QuantileTransformer`)
  - Purpose: map features to a uniform or normal distribution; useful for robust scaling across samples.

- `PowerTransformer` (`sklearn.preprocessing.PowerTransformer`)
  - Purpose: stabilize variance and make data more Gaussian-like (Yeo-Johnson/Box-Cox).


## Useful SciPy functions for NIRS

These common SciPy helpers are used across nirs4all transformers or are handy
when building custom preprocessing steps. Each entry is short and shows the
function, purpose and typical usage note.

- `scipy.signal.savgol_filter`
  - Purpose: smoothing and derivative calculation (Savitzky–Golay).
  - Signature: `savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0)`
  - Usage: low-noise smoothing with preservation of peak shapes; used in
    `SavitzkyGolay` class.

- `scipy.signal.detrend`
  - Purpose: remove linear (or piecewise linear) trend from data.
  - Signature: `detrend(data, bp=0)`
  - Usage: quick baseline slope removal; used in `Detrend` transformer.

- `scipy.ndimage.gaussian_filter1d`
  - Purpose: apply 1D Gaussian smoothing along the feature axis.
  - Signature: `gaussian_filter1d(input, sigma, order=0)`
  - Usage: fast separable Gaussian smoothing; used in `Gaussian` transformer.

- `scipy.signal.resample`
  - Purpose: Fourier-based resampling to a new number of samples.
  - Signature: `resample(x, num, axis=0)`
  - Usage: useful for changing spectral sampling rate (aliasing caveats apply).

- `scipy.interpolate.UnivariateSpline`
  - Purpose: smooth spline fitting and evaluation for 1D signals.
  - Signature: `UnivariateSpline(x, y, s=0, k=3)`
  - Usage: smoothing or interpolation; used by spline augmenters and smoothing.

- `scipy.interpolate.interp1d`
  - Purpose: linear or higher-order 1D interpolation between points.
  - Signature: `interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')`
  - Usage: used by `ResampleTransformer` for simple resampling between wavelengths.

- `scipy.signal.medfilt` (median filter)
  - Purpose: apply sliding-window median filter for impulse noise removal.
  - Signature: `medfilt(x, kernel_size=3)`
  - Usage: robust to spikes; occasionally used as pre-step before smoothing.

- `scipy.signal.find_peaks`
  - Purpose: detect peaks in 1D signals (prominence, height, width options).
  - Signature: `find_peaks(x, height=None, prominence=None, width=None)`
  - Usage: feature extraction (peak locations/heights) from spectra.


## Quick recipes (beginners)

- Typical simple pipeline: smoothing → scatter correction → scaling → PCA → model
  - e.g. `SavitzkyGolay → MultiplicativeScatterCorrection → StandardScaler → PCA`

- Augmentation examples: `Rotate_Translate`, `Spline_Y_Perturbations`, and `Random_X_Operation`
  are useful for building robust training sets.


## Notes and caveats

**Axis conventions — IMPORTANT:**
- Most transformers assume `X.shape == (n_samples, n_features)` where features = wavelengths.
- **⚠️ Exception**: `Derivate` (in `scalers.py`) computes gradients along **axis=0** (samples),
  which is unusual for NIRS. It is kept for legacy support.
  For standard spectral derivatives along wavelengths, use:
  - `FirstDerivative` or `SecondDerivative` (in `nirs.py`) — these use **axis=1** (features/wavelengths)
  - `SavitzkyGolay` with `deriv=1` or `deriv=2` — also operates along wavelengths
- **Verify before chaining derivatives**: mixing `Derivate` with other derivative methods will
  produce incorrect results due to different axis conventions.

**Other notes:**
- Most nirs4all transformers raise on sparse inputs and provide `inverse_transform`
  only when it is meaningful (check class methods).
- For advanced usage, consult the specific source file in `nirs4all/operators/` for
  parameter details and behavior.
- When using tranformers, set `random_state` for reproducibility in experiments.

---

**Further reading:**
- See `examples/Q1_regression.py` for a complete NIRS pipeline with preprocessing combinations
- See `examples/Tutorial_2_Advanced_Analysis.ipynb` for interactive preprocessing workflows
- Consult scikit-learn and SciPy documentation for detailed parameter descriptions


