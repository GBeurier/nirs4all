# NIRS Data Augmentation in `nirs4all`
Developer guidelines (no-learning augmentations)

This document describes how to implement and integrate **purely algorithmic** NIRS augmentations into `nirs4all`.
All operators are designed as **sklearn-style transformers** and must be safe to use in NIRS pipelines (no model training inside the transform).

---

## 1. General design rules

### 1.1 Input / output conventions

- Input spectra: `X` as `np.ndarray` or `ArrayLike` of shape `(n_samples, n_wavelengths)`.
- Optional wavelength axis: `lambda_axis` (1D array of shape `(n_wavelengths,)`), passed via:
  - `__init__(..., lambda_axis: Optional[np.ndarray] = None)`, or
  - a config object / dataset wrapper (depending on `nirs4all`’s existing patterns).
- Output: **same shape** as input, `(n_samples, n_wavelengths)`.
- No change of sample order; no change of dtype beyond standard float conversions.

### 1.2 Base class and randomness

All augmenters should:

- Inherit from:
  ```python
  class BaseAugmenter(BaseEstimator, TransformerMixin):
      def __init__(self, random_state: Optional[int | np.random.Generator] = None, ...):
          ...
    ```

* Use a local RNG:

  ```python
  self._rng = np.random.default_rng(self.random_state)
  ```
* Be **stateless** w.r.t. the data (no data-dependent fit, except for mixup where we just cache `n_samples`).

### 1.3 Training-only behavior

Most augmentations should be used **during training only**.
We recommend a common wrapper (if not already present):

```python
class TrainOnlyWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, augmenter, apply_during: str = "fit"):
        # apply_during in {"fit", "always"}
        self.augmenter = augmenter
        self.apply_during = apply_during

    def fit(self, X, y=None):
        self.augmenter.fit(X, y)
        return self

    def transform(self, X, y=None, *, is_training: bool = False):
        if self.apply_during == "fit" and not is_training:
            return X
        return self.augmenter.transform(X, y)
```

Pipelines can then pass `is_training=True` from the training loop.

---

## 2. Augmentation families and suggested classes

Below: one class per family, with a **minimal required behavior** and **recommended parameters**.

### 2.1 Additive / multiplicative noise

#### 2.1.1 Gaussian Additive Noise (correlated)

**Class name:** `GaussianAdditiveNoise`

**Effect:**
`X_aug = X + noise`, where `noise` is Gaussian, lissé le long de l’axe spectral.

**Constructor:**

```python
class GaussianAdditiveNoise(BaseAugmenter):
    def __init__(
        self,
        sigma: float = 0.01,
        smoothing_kernel_width: int = 5,
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

**Notes:**

* `sigma` is relative to the global standard deviation of `X` (or per-sample).
* `smoothing_kernel_width` must be odd; use 1D conv with a normalized Gaussian kernel.

#### 2.1.2 Multiplicative Noise (gain jitter)

**Class name:** `MultiplicativeNoise`

**Effect:**
`X_aug = (1 + ε) * X` with ε ~ N(0, `sigma_gain`²).

**Constructor:**

```python
class MultiplicativeNoise(BaseAugmenter):
    def __init__(
        self,
        sigma_gain: float = 0.01,
        per_sample: bool = True,
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

**Notes:**

* `per_sample=True`: draw one gain per sample.
* Optionally add `per_wavelength` (gain per (sample, wavelength) with small σ).

---

### 2.2 Baseline shifts and drifts

#### 2.2.1 Linear Baseline Drift

**Class name:** `LinearBaselineDrift`

**Effect:**
`X_aug[i, :] = X[i, :] + a_i + b_i * (λ - λ₀)`

**Constructor:**

```python
class LinearBaselineDrift(BaseAugmenter):
    def __init__(
        self,
        offset_range: tuple[float, float] = (-0.02, 0.02),
        slope_range: tuple[float, float] = (-0.0005, 0.0005),
        lambda_axis: Optional[np.ndarray] = None,
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

**Notes:**

* If `lambda_axis` is `None`, use indices [0..n_wavelengths-1] as surrogate λ.
* `a_i` and `b_i` drawn independently per sample.

#### 2.2.2 Polynomial Baseline Drift

**Class name:** `PolynomialBaselineDrift`

**Effect:**
Add a low-frequency polynomial to each spectrum.

```python
class PolynomialBaselineDrift(BaseAugmenter):
    def __init__(
        self,
        degree: int = 2,
        coeff_ranges: dict[int, tuple[float, float]] | None = None,
        lambda_axis: Optional[np.ndarray] = None,
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

**Notes:**

* `coeff_ranges` maps polynomial degree → `(min, max)` coefficient.
* If `coeff_ranges` is `None`, define default ranges internally, scaled to typical NIRS amplitude.

---

### 2.3 Wavelength axis distortions

All these transforms **warp λ** then **resample** to the original λ grid.

Use a robust interpolation method (e.g. `np.interp` for 1D).

#### 2.3.1 Global Wavelength Shift

**Class name:** `WavelengthShift`

**Effect:**
Shift λ by `δλ`, then interpolate back on original axis.

```python
class WavelengthShift(BaseAugmenter):
    def __init__(
        self,
        shift_range: tuple[float, float] = (-2.0, 2.0),  # nm
        lambda_axis: Optional[np.ndarray] = None,
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

#### 2.3.2 Global Stretch / Compression

**Class name:** `WavelengthStretch`

**Effect:**
λ' = λ₀ + (1 + α) * (λ - λ₀), α small; then interpolation.

```python
class WavelengthStretch(BaseAugmenter):
    def __init__(
        self,
        stretch_range: tuple[float, float] = (-0.005, 0.005),  # relative
        lambda_axis: Optional[np.ndarray] = None,
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

#### 2.3.3 Local Nonlinear Warp

**Class name:** `LocalWavelengthWarp`

**Effect:**
Apply a smooth monotone warp using random control points.

```python
class LocalWavelengthWarp(BaseAugmenter):
    def __init__(
        self,
        n_control_points: int = 5,
        max_shift_nm: float = 1.0,
        lambda_axis: Optional[np.ndarray] = None,
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

**Implementation idea:**

* Define control points on λ-axis.
* Sample small shifts for each control point in `[-max_shift_nm, max_shift_nm]`.
* Fit a monotone spline and apply warp per sample.

---

### 2.4 Magnitude warping

#### 2.4.1 Smooth Gain Function

**Class name:** `SmoothMagnitudeWarp`

**Effect:**
Multiply by a smooth gain curve `f(λ)` with `f(λ) ≈ 1`.

```python
class SmoothMagnitudeWarp(BaseAugmenter):
    def __init__(
        self,
        n_control_points: int = 5,
        gain_range: tuple[float, float] = (0.95, 1.05),
        lambda_axis: Optional[np.ndarray] = None,
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

**Implementation idea:**

* Sample gain values at control points uniformly in `gain_range`.
* Interpolate by spline or linear interpolation across λ.
* `X_aug = X * f(λ)`.

#### 2.4.2 Band-Specific Perturbation

**Class name:** `BandPerturbation`

**Effect:**
Multiply or offset intensity within specific λ bands (e.g. water bands).

```python
class BandPerturbation(BaseAugmenter):
    def __init__(
        self,
        bands: list[tuple[float, float]],  # list of (λ_min, λ_max)
        gain_range: tuple[float, float] = (0.9, 1.1),
        offset_range: tuple[float, float] = (-0.01, 0.01),
        lambda_axis: Optional[np.ndarray] = None,
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

**Notes:**

* For each sample, select a subset of bands and apply random gain/offset.

---

### 2.5 Resolution / smoothing jitter

#### 2.5.1 Gaussian Smoothing Jitter

**Class name:** `GaussianSmoothingJitter`

**Effect:**
Convolve each spectrum with a Gaussian of random σ within a range.

```python
class GaussianSmoothingJitter(BaseAugmenter):
    def __init__(
        self,
        sigma_range: tuple[float, float] = (0.5, 1.5),
        kernel_size: int = 7,
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

**Notes:**

* Use a normalized 1D Gaussian kernel; handle boundary with reflection or nearest padding.

#### 2.5.2 Unsharp Mask (Mild Sharpening)

**Class name:** `UnsharpSpectralMask`

**Effect:**
`X_aug = X + k * (X - smooth(X))`, small `k`.

```python
class UnsharpSpectralMask(BaseAugmenter):
    def __init__(
        self,
        amount_range: tuple[float, float] = (0.0, 0.2),
        smoothing_sigma: float = 1.0,
        kernel_size: int = 7,
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

---

### 2.6 Spectral masking and dropout

#### 2.6.1 Band Masking

**Class name:** `BandMasking`

**Effect:**
Randomly mask short contiguous bands (set to 0 or interpolate).

```python
class BandMasking(BaseAugmenter):
    def __init__(
        self,
        max_mask_width: int = 10,
        n_bands_range: tuple[int, int] = (0, 2),
        mode: str = "zero",  # or "interp"
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

**Notes:**

* Draw `k` bands per sample in `n_bands_range`.
* For `mode="interp"`, linearly interpolate between boundaries of each band.

#### 2.6.2 Channel Dropout

**Class name:** `ChannelDropout`

**Effect:**
Drop individual wavelengths.

```python
class ChannelDropout(BaseAugmenter):
    def __init__(
        self,
        dropout_prob: float = 0.01,
        mode: str = "zero",  # or "interp"
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

---

### 2.7 Rare structured artefacts

#### 2.7.1 Spike Noise

**Class name:** `SpikeNoise`

**Effect:**
Add a small number of narrow spikes.

```python
class SpikeNoise(BaseAugmenter):
    def __init__(
        self,
        n_spikes_range: tuple[int, int] = (0, 3),
        amplitude_range: tuple[float, float] = (0.05, 0.2),
        width_range: tuple[int, int] = (1, 3),
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

**Notes:**

* Spikes can be positive or negative; consider symmetric amplitude range.
* Optionally smooth a bit to avoid delta-like artefacts.

#### 2.7.2 Local Saturation / Clipping

**Class name:** `LocalClipping`

**Effect:**
Clip segments locally to mimic saturation.

```python
class LocalClipping(BaseAugmenter):
    def __init__(
        self,
        clip_prob: float = 0.1,
        n_segments_range: tuple[int, int] = (0, 2),
        segment_width_range: tuple[int, int] = (1, 5),
        clip_mode: str = "max",  # "max", "min", or "both"
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

---

### 2.8 Sample combinations (mixup-style, no learning)

#### 2.8.1 Global Mixup

**Class name:** `MixupAugmenter`

**Effect:**
Combine pairs of samples `(x_i, y_i)` and `(x_j, y_j)`:

* `x_aug = λ_mix * x_i + (1 - λ_mix) * x_j`
* `y_aug = λ_mix * y_i + (1 - λ_mix) * y_j` (regression)

```python
class MixupAugmenter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        alpha: float = 0.2,
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

**Notes:**

* Use Beta(α, α) to sample `λ_mix`.
* This transformer modifies **both X and y**; `transform` must return `(X_aug, y_aug)` or use `fit_resample`-style API if you already have one.

#### 2.8.2 Local Mixup (nearest-neighbor)

**Class name:** `LocalMixupAugmenter`

Same as `MixupAugmenter` but restricts pairs to neighbors in spectral space.

```python
class LocalMixupAugmenter(MixupAugmenter):
    def __init__(
        self,
        alpha: float = 0.2,
        n_neighbors: int = 10,
        distance_metric: str = "euclidean",
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

**Implementation note:**
Precompute nearest neighbors in `fit`, then sample j among neighbors of i in `transform`.

---

### 2.9 Scattering-based simulation

#### 2.9.1 MSC-Style Scatter Simulation

**Class name:** `ScatterSimulationMSC`

**Effect:**
Simulate scatter variation by perturbing `a, b` in `x ≈ a + b * x_ref`.

```python
class ScatterSimulationMSC(BaseAugmenter):
    def __init__(
        self,
        offset_range: tuple[float, float] = (-0.02, 0.02),
        scale_range: tuple[float, float] = (0.95, 1.05),
        reference_mode: str = "self",  # "self", "global_mean", or "provided"
        reference_spectrum: Optional[np.ndarray] = None,
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

**Notes:**

* For `reference_mode="self"`: treat `x` itself as reference and apply `x_aug = a + b * x`.
* For `reference_mode="global_mean"`: compute mean spectrum in `fit`.
* For `reference_mode="provided"`: use external reference.

---

## 3. Implementation details and utilities

### 3.1 Utility functions

Create a small internal module for common operations:

* 1D convolution with reflection padding:

  ```python
  def conv1d_reflect(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
      ...
  ```
* Gaussian kernel factory:

  ```python
  def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
      ...
  ```
* Safe interpolation:

  ```python
  def interpolate_to_axis(
      x: np.ndarray,
      lambda_src: np.ndarray,
      lambda_dst: np.ndarray,
      fill_value: float = "edge",
  ) -> np.ndarray:
      ...
  ```

### 3.2 Parameter validation

Each augmenter should validate ranges in `__init__`:

* Ensure min ≤ max for ranges.
* Ensure widths and kernel sizes are positive integers.
* Raise clear `ValueError` with short messages (for config debugging).

---

## 4. Integration into `nirs4all` pipelines

* Expose all classes in a dedicated module, e.g. `nirs4all.augmentation.spectral`.

* Provide **factory functions** or configuration keys to instantiate from JSON/YAML:

  ```yaml
  - type: GaussianAdditiveNoise
    params:
      sigma: 0.02
      smoothing_kernel_width: 7
  - type: WavelengthShift
    params:
      shift_range: [-1.0, 1.0]
  ```

* Ensure compatibility with:

  * Existing dataset abstraction (`SpectraDataset` / `SpectroDataset`).
  * Any `TrainOnlyWrapper` or pipeline-level mechanism that passes `is_training=True` only during training.

---

## 5. Testing guidelines

For each augmenter:

1. **Shape invariance:** `X_aug.shape == X.shape`.
2. **Determinism:** fixed `random_state` ⇒ identical outputs.
3. **Amplitude sanity:** output values stay in reasonable bounds for typical NIRS ranges.
4. **No NaN / inf:** unless explicitly requested (should not be).
5. **Lambda-axis usage:** when `lambda_axis` is provided, warps must be consistent with nm values.

Once implemented, add small end-to-end tests combining several augmenters in a pipeline to ensure they compose correctly.

