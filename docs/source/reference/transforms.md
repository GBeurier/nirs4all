# Transforms Reference

A transform changes the features presented to a model. Choose it from a measurement question, then verify that useful signal survives. For example, scatter correction removes gain/offset differences; smoothing suppresses rapid oscillations; derivatives emphasize slopes; resampling changes where spectra are evaluated.

## Choose a transform by the effect you want to remove

Read {doc}`nodes/preprocessing` for the worked result figure, expected dimensions and exercises. Choose a family below to compare the enumerated operators.

## Same worked recipe in JSON, YAML and Python

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "pipeline": [
    {
      "class": "nirs4all.operators.transforms.StandardNormalVariate"
    }
  ]
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
pipeline:
- class: nirs4all.operators.transforms.StandardNormalVariate
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.operators.transforms import SNV

pipeline = [SNV()]
```
:::

::::

These are Python SDK operators. Native R/Octave/WASM recipes use their own method IDs and facade; see {doc}`/guide/languages`.

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
| `Derivate` | `order=1`, `delta=1`, `axis=1` | Nth-order wavelength derivative; `axis=0` explicitly differentiates along samples |

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

## How to interpret and combine the families

```{figure} /assets/guide/preprocessing_node.svg
:alt: Spectra with offset and gain differences become the same row-normalized shape after SNV.

Educational SNV example: `[1, 2, 3]` and `[3, 5, 7]` both become approximately
`[-1.225, 0, 1.225]`. The mechanism removes gain/offset, not sample identity.
```

**Scatter correction.** `StandardNormalVariate` uses each row's mean and
standard deviation; it does not estimate the population mean spectrum.
`LocalStandardNormalVariate` uses local windows and can alter peak contrast
at window boundaries. `RobustStandardNormalVariate` substitutes median and
MAD, reducing the influence of spikes on normalization. MSC learns the mean
training spectrum and corrects each row using a fitted offset and slope.
EMSC extends that regression with polynomial nuisance terms. The current
MSC `scale` argument is retained for API compatibility and does not change
the raw-spectrum reference calculation. Fit its reference on training rows.
Area normalization compares relative spectral contributions; it can discard
absolute intensity that was useful for predicting concentration.

**Smoothing and derivatives.** Savitzky–Golay fits a local polynomial to a
window. Increase `window_length` to suppress more noise at the cost of
blurring narrow peaks; `polyorder` must be smaller than the window length.
Set `deriv=0` for smoothing and `deriv=1` or `2` for smoothed derivatives.
`delta` is the grid spacing and controls derivative units. The usual choice
is an odd window no longer than the spectrum. Gaussian filtering spreads
each point over its neighbors; `sigma` controls that scale. First/second
derivatives differentiate along the spectral axis, while Norris–Williams
adds segment averaging and a gap. Wavelet denoising thresholds detail
coefficients; its level, boundary mode and threshold can change edge peaks.
Inspect those boundaries before comparing validation scores.

**Baseline correction.** Detrend removes a linear trend. AsLS and IAsLS
balance smoothness (`lam`) against asymmetric treatment of peaks (`p`);
AirPLS and ArPLS adapt weights iteratively. ModPoly and IModPoly use
polynomial baselines; SNIP iteratively clips peaks, RollingBall uses a
morphological envelope, and BEADS combines sparse signal and baseline
estimation. These methods encode different beliefs about the baseline.
Do not choose a method solely because it makes a spectrum visually flat:
broad chemical peaks may also be removed. Optional `pybaselines` is needed
for its wrappers.

**Orthogonalization.** OSC learns directions in X that are orthogonal to y.
It is supervised even though it appears before the model. EPO learns a
spectral interference subspace from external parameters `d`, such as
temperature; `d` is not the prediction target. EPO's fitted projection can
then transform future spectra without requiring `d`. Both learn state
inside training folds. See `U05_orthogonalization.py` for external-parameter
binding in a full workflow.

**Physical conversions.** Reflectance-to-absorbance computes `-log10(R)` for
fractional reflectance; percent reflectance requires dividing by 100 first
or declaring the appropriate source type. `FromAbsorbance` applies the
inverse exponential. `SignalTypeConverter` selects the conversion from its
declared source and target types; it does not establish those types from
chemistry. Kubelka–Munk computes `(1-R)^2/(2R)` for diffuse reflectance.
LogTransform is a general numerical logarithm with offset rules, which need
not have a physical absorbance interpretation. Keep units and signal type
with the source schema, especially across language boundaries.

**Representation and selection.** Wavelet/Haar return coefficients;
WaveletFeatures summarizes coefficients at multiple scales. WaveletPCA and
WaveletSVD learn reductions at each scale, so their fitted bases must be
reused at prediction. FlexiblePCA centers its input and FlexibleSVD offers
a truncated SVD representation. CARS and MCUVE use y to select wavelengths;
their internal selection must not see outer validation targets. Cropping
removes indexed channels, resampling changes the grid, and flattening joins
preprocessing views. Each changes how downstream feature positions should
be interpreted. An index crop does not imply a particular physical
wavelength range unless the schema establishes that mapping.

## A complete, small transformation experiment

This standalone Python example exposes fitted state explicitly. It creates
training and future rows, fits MSC on training only, then reuses its reference.
It is a numerical experiment rather than a cross-validation pipeline.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
import numpy as np
from nirs4all.operators.transforms import MSC, SNV, SavitzkyGolay

wavelengths = np.linspace(1000, 2200, 241)  # 5 nm spacing
peak = np.exp(-((wavelengths - 1450) / 100) ** 2)
X_train = np.stack([peak, 1.2 * peak + 0.1, 0.8 * peak - 0.05])
X_future = np.stack([1.1 * peak + 0.08])

msc = MSC().fit(X_train)
corrected_train = msc.transform(X_train)
corrected_future = msc.transform(X_future)
assert corrected_future.shape == (1, 241)

snv = SNV().fit_transform(X_train)
assert np.allclose(snv.mean(axis=1), 0, atol=1e-12)
derivative = SavitzkyGolay(
    window_length=11, polyorder=3, deriv=1, delta=5.0
).fit_transform(X_train)
assert derivative.shape == X_train.shape
```
:::
::::

For workflow-level fit scope use {doc}`nodes/preprocessing`, not manual
prefitting on the whole dataset. A Python SDK serialized preprocessing node
can be written as follows; this import-path form is not a portable native ID:

::::{tab-set}
:sync-group: language

:::{tab-item} YAML
:sync: yaml

```yaml
preprocessing:
  class: nirs4all.operators.transforms.SavitzkyGolay
  params:
    window_length: 11
    polyorder: 3
    deriv: 1
    delta: 5.0
```
:::
::::

## Learning exercises and exact APIs

Run `U01_preprocessing_basics.py` to compare transformations, then
`U04_signal_conversion.py` to inspect declared units and
`U06_wavelet_denoise.py` to inspect scale and boundary behavior. Compare one
factor at a time on the same folds. Preserve the fitted transform and feature
schema with the final model.

The {doc}`/api/modules` tree contains per-class constructor and method
documentation under `nirs4all.operators.transforms`. The native operator IDs
and binding availability are separate contracts: consult
{doc}`/guide/interfaces` before translating an import path to another host.

## Ragged temporal encoding: SequenceSummary

```{figure} /assets/guide/ragged_series.svg
:alt: Variable-length sequences become five fixed summary columns: mean, population standard deviation, minimum, maximum and observation count.

Educational one-channel example. Sequences of length 2, 3 and 4 keep their sample identity while becoming a matrix with three rows and five columns. Values are described below.
```

| Observed sequence | Mean | Population SD | Minimum | Maximum | Length |
|---|---|---|---|---|---|
| `[1, 3]` | 2 | 1 | 1 | 3 | 2 |
| `[2, 4, 6]` | 4 | 1.633 | 2 | 6 | 3 |
| `[0, 2, 4, 6]` | 3 | 2.236 | 0 | 6 | 4 |

The summary uses equally weighted observations. It ignores time-coordinate spacing and loses the ordering of measurements. With several channels, summaries are constructed per channel and channel identity must remain fixed. The observation-count feature is optional; it is not elapsed time.


`SequenceSummary` accepts an IO `RaggedSeriesBatch`: one variable-length
multichannel series per sample. It computes an ordered subset of `mean`,
`std`, `min` and `max` for each channel, optionally appending observation count.
For two channels, four statistics and `include_length=True`, output width
is `2 × 4 + 1 = 9`, regardless of individual series lengths. Columns are
ordered by channel, then statistic; length is last. Standard deviation uses
`ddof=0`.


Each sample is summarized independently into the same column contract.
Sequence lengths are allowed to differ; channel identities must stay fixed.
The optional length feature describes measurement count, not elapsed time.

The encoder deliberately ignores time coordinates and weights observations
equally. Therefore an arithmetic mean is not a time-weighted mean for
irregular sampling; order and dynamics are lost. Use it when aggregate
behavior is a plausible representation, and compare against an appropriate
temporal representation when dynamics matter.

`channel_names` supplies stable names. `min_observations=1` accepts singleton
series; a larger value makes an explicit applicability rule.
`channel_bounds` supplies inclusive finite domain limits per channel and is
not inferred from held-out observations. Empty series and nonfinite observed
values are refused. A missing modality must be handled by its presence policy
before reaching the encoder; do not pad it with fake measurements.

Fit records the channel contract, not training values or lengths. Run
`U12_multimodal_ragged_series.py`, `U13_multimodal_late_missing_sources.py`
and {doc}`/guide/datasets` for workflow, missingness and domain
examples. Availability of this Python encoder does not imply an identical
ragged runtime profile in every binding.

## Fixed convolution features: FCKStaticTransformer

`FCKStaticTransformer` applies a constructor-defined bank of normalized
fractional convolution filters along wavelengths. The defaults combine four
orders, two scales and two kernel sizes into 16 filters. With flattening,
`(n, p)` becomes `(n, 16p)`; `flatten=False` retains `(n, 16, p)`.
`alphas`, `scales`, `kernel_sizes` and `sigma` define the bank;
`mode` defines boundary handling. It does not learn filters from y or the
training population, but selecting its hyperparameters still requires
training/validation separation. Large banks increase memory and model
complexity. Compare its added representations against a simple derivative
baseline rather than treating more columns as additional independent evidence.
