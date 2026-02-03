# NIRS Data Augmentation in `nirs4all`
Developer guidelines (no-learning augmentations)

This document describes how to implement and integrate **purely algorithmic** NIRS augmentations into `nirs4all`.
All operators are designed as **sklearn-style transformers** and must be safe to use in NIRS pipelines (no model training inside the transform).

---

## 1. General design rules

### 1.1 Input / output conventions

- Input: `X` (2D array of shape `(n_samples, n_wavelengths)`)
- Output: `X_transformed` (same shape)
- Wavelength handling: Wavelength-aware operators use `SpectraTransformerMixin` and receive wavelengths automatically from the controller
- Random state: `random_state` (int or None) for reproducibility, using `numpy.random.default_rng`

### 1.2 Base class and randomness

Base class patterns:

```python
# For wavelength-aware operators:
from nirs4all.operators.base import SpectraTransformerMixin

class MyAugmenter(SpectraTransformerMixin):
    _requires_wavelengths = "optional"  # or True

    def _transform_impl(self, X, wavelengths):
        # wavelengths may be None if _requires_wavelengths = "optional"
        return X_transformed

# For plain operators:
from sklearn.base import TransformerMixin, BaseEstimator

class MyAugmenter(TransformerMixin, BaseEstimator):
    def transform(self, X, **kwargs):
        return X_transformed
```

All augmenters should:

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
class GaussianAdditiveNoise(TransformerMixin, BaseEstimator):
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
class MultiplicativeNoise(TransformerMixin, BaseEstimator):
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
class LinearBaselineDrift(SpectraTransformerMixin):
    _requires_wavelengths = "optional"

    def __init__(
        self,
        offset_range: tuple[float, float] = (-0.02, 0.02),
        slope_range: tuple[float, float] = (-0.0005, 0.0005),
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

**Notes:**

* Wavelengths are received automatically from the controller. If unavailable, indices [0..n_wavelengths-1] are used as surrogate λ.
* `a_i` and `b_i` drawn independently per sample.

#### 2.2.2 Polynomial Baseline Drift

**Class name:** `PolynomialBaselineDrift`

**Effect:**
Add a low-frequency polynomial to each spectrum.

```python
class PolynomialBaselineDrift(SpectraTransformerMixin):
    _requires_wavelengths = "optional"

    def __init__(
        self,
        degree: int = 2,
        coeff_ranges: dict[int, tuple[float, float]] | None = None,
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
class WavelengthShift(SpectraTransformerMixin):
    _requires_wavelengths = "optional"

    def __init__(
        self,
        shift_range: tuple[float, float] = (-2.0, 2.0),  # nm
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

#### 2.3.2 Global Stretch / Compression

**Class name:** `WavelengthStretch`

**Effect:**
λ' = λ₀ + (1 + α) * (λ - λ₀), α small; then interpolation.

```python
class WavelengthStretch(SpectraTransformerMixin):
    _requires_wavelengths = "optional"

    def __init__(
        self,
        stretch_range: tuple[float, float] = (-0.005, 0.005),  # relative
        random_state: Optional[int | np.random.Generator] = None,
    ):
        ...
```

#### 2.3.3 Local Nonlinear Warp

**Class name:** `LocalWavelengthWarp`

**Effect:**
Apply a smooth monotone warp using random control points.

```python
class LocalWavelengthWarp(SpectraTransformerMixin):
    _requires_wavelengths = "optional"

    def __init__(
        self,
        n_control_points: int = 5,
        max_shift_nm: float = 1.0,
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
class SmoothMagnitudeWarp(SpectraTransformerMixin):
    _requires_wavelengths = "optional"

    def __init__(
        self,
        n_control_points: int = 5,
        gain_range: tuple[float, float] = (0.95, 1.05),
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
class BandPerturbation(SpectraTransformerMixin):
    _requires_wavelengths = "optional"

    def __init__(
        self,
        bands: list[tuple[float, float]],  # list of (λ_min, λ_max)
        gain_range: tuple[float, float] = (0.9, 1.1),
        offset_range: tuple[float, float] = (-0.01, 0.01),
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
class GaussianSmoothingJitter(TransformerMixin, BaseEstimator):
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
class UnsharpSpectralMask(TransformerMixin, BaseEstimator):
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
class BandMasking(TransformerMixin, BaseEstimator):
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
class ChannelDropout(TransformerMixin, BaseEstimator):
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
class SpikeNoise(TransformerMixin, BaseEstimator):
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
class LocalClipping(TransformerMixin, BaseEstimator):
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
class ScatterSimulationMSC(SpectraTransformerMixin):
    _requires_wavelengths = "optional"

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

## 3. Wavelength-Aware Augmentation (Physics-Based)

Some augmentation operators require wavelength information to apply physically realistic effects. These operators inherit from `SpectraTransformerMixin` and automatically receive wavelengths from the dataset when used in a pipeline.

### 3.1 Environmental Effects

#### 3.1.1 TemperatureAugmenter

**Class name:** `TemperatureAugmenter`

**Effect:** Simulates temperature-induced spectral changes based on literature values for NIR spectroscopy.

Temperature affects NIR spectra through:
- Peak position shifts (especially O-H, N-H bands)
- Intensity changes (hydrogen bonding disruption)
- Band broadening (thermal motion)

```python
from nirs4all.operators.augmentation import TemperatureAugmenter

class TemperatureAugmenter(SpectraTransformerMixin):
    def __init__(
        self,
        temperature_delta: float = 5.0,
        temperature_range: Optional[Tuple[float, float]] = None,
        reference_temperature: float = 25.0,
        enable_shift: bool = True,
        enable_intensity: bool = True,
        enable_broadening: bool = True,
        region_specific: bool = True,
        random_state: Optional[int] = None,
    ):
        ...
```

**Notes:**
* Region-specific effects for O-H (1400-1520nm, 1900-2000nm), N-H (1490-1560nm), and C-H (1650-1780nm) bands
* Literature-based parameters from Maeda et al. (1995), Segtnan et al. (2001)
* Use `temperature_range` for per-sample random variation

**Example:**

```python
from nirs4all.operators.augmentation import TemperatureAugmenter
from sklearn.cross_decomposition import PLSRegression
import nirs4all

# Fixed temperature shift
pipeline = [
    TemperatureAugmenter(temperature_delta=10.0),
    PLSRegression(n_components=10),
]

# Random temperature variation for robustness training
pipeline = [
    TemperatureAugmenter(temperature_range=(-5, 15)),
    PLSRegression(n_components=10),
]

result = nirs4all.run(pipeline=pipeline, dataset="my_dataset")
```

#### 3.1.2 MoistureAugmenter

**Class name:** `MoistureAugmenter`

**Effect:** Simulates moisture/water activity effects on spectra.

Water activity affects NIR spectra through shifts in water bands between free and bound states.

```python
from nirs4all.operators.augmentation import MoistureAugmenter

class MoistureAugmenter(SpectraTransformerMixin):
    def __init__(
        self,
        water_activity_delta: float = 0.1,
        water_activity_range: Optional[Tuple[float, float]] = None,
        free_water_fraction: float = 0.3,
        bound_water_shift: float = 15.0,
        random_state: Optional[int] = None,
    ):
        ...
```

**Notes:**
* Affects 1st overtone (1400-1500nm) and combination (1900-2000nm) water bands
* Models free vs. bound water state transitions
* `free_water_fraction` controls the ratio of free to bound water

### 3.2 Scattering Effects

#### 3.2.1 ParticleSizeAugmenter

**Class name:** `ParticleSizeAugmenter`

**Effect:** Simulates particle size effects on light scattering.

Particle size affects NIR spectra through wavelength-dependent baseline scattering, typically following a lambda^(-n) relationship.

```python
from nirs4all.operators.augmentation import ParticleSizeAugmenter

class ParticleSizeAugmenter(SpectraTransformerMixin):
    def __init__(
        self,
        mean_size_um: float = 50.0,
        size_variation_um: float = 15.0,
        size_range_um: Optional[Tuple[float, float]] = None,
        wavelength_exponent: float = 1.5,
        size_effect_strength: float = 0.1,
        include_path_length: bool = True,
        random_state: Optional[int] = None,
    ):
        ...
```

**Notes:**
* Higher `wavelength_exponent` = finer particles (Rayleigh scattering ~4, Mie scattering ~1-2)
* Use `size_range_um` for per-sample random variation
* Path length effect simulates longer optical paths for smaller particles

#### 3.2.2 EMSCDistortionAugmenter

**Class name:** `EMSCDistortionAugmenter`

**Effect:** Applies EMSC-style scatter distortions.

Simulates the spectral distortions that Extended Multiplicative Scatter Correction (EMSC) is designed to correct:

`x_distorted = a + b*x + c1*lambda + c2*lambda^2 + ...`

```python
from nirs4all.operators.augmentation import EMSCDistortionAugmenter

class EMSCDistortionAugmenter(SpectraTransformerMixin):
    def __init__(
        self,
        multiplicative_range: Tuple[float, float] = (0.9, 1.1),
        additive_range: Tuple[float, float] = (-0.05, 0.05),
        polynomial_order: int = 2,
        polynomial_strength: float = 0.02,
        correlation: float = 0.0,
        random_state: Optional[int] = None,
    ):
        ...
```

**Notes:**
* `multiplicative_range` controls the gain factor (b)
* `additive_range` controls the offset (a)
* `polynomial_order` adds wavelength-dependent baseline curvature
* `correlation` links additive and multiplicative effects (typical in real scatter)

### 3.3 Combining Environmental and Scattering Augmentation

For maximum robustness in field applications (e.g., handheld NIRS), combine multiple augmentation types:

```python
from nirs4all.operators.augmentation import (
    TemperatureAugmenter,
    MoistureAugmenter,
    ParticleSizeAugmenter,
)
from sklearn.cross_decomposition import PLSRegression
import nirs4all

# Robust field deployment pipeline
pipeline = [
    TemperatureAugmenter(temperature_range=(-10, 20)),
    MoistureAugmenter(water_activity_range=(0.3, 0.9)),
    ParticleSizeAugmenter(size_range_um=(20, 100)),
    PLSRegression(n_components=10),
]

result = nirs4all.run(pipeline=pipeline, dataset="field_samples")
```

### 3.4 Edge Artifacts

Edge artifacts are common instrumental and physical phenomena that cause spectral distortions at the boundaries (start and end) of the measured wavelength range. These effects are well-documented in NIR spectroscopy literature and can significantly impact model performance if not accounted for.

**Scientific Background:**

Edge artifacts arise from several sources:

1. **Detector sensitivity roll-off**: NIR detectors (InGaAs, PbS, Silicon CCD) have wavelength-dependent sensitivity curves that typically decrease at the edges of their spectral range, causing increased noise and reduced signal quality.

2. **Stray light contamination**: Scattered light within the spectrometer that reaches the detector without passing through the sample. This effect is often wavelength-dependent and more pronounced at spectral edges.

3. **Truncated absorption peaks**: Real absorption bands whose centers lie outside the measured wavelength range, appearing as rising/falling baselines at the spectral edges.

4. **Baseline curvature**: Instrumental effects causing systematic baseline bending near measurement boundaries.

**References:**
- Workman Jr, J., & Weyer, L. (2012). *Practical Guide and Spectral Atlas for Interpretive Near-Infrared Spectroscopy*. CRC Press. Chapters 4-5.
- Burns, D. A., & Ciurczak, E. W. (2007). *Handbook of Near-Infrared Analysis* (3rd ed.). CRC Press.
- Siesler, H. W., Ozaki, Y., Kawata, S., & Heise, H. M. (2002). *Near-Infrared Spectroscopy: Principles, Instruments, Applications*. Wiley-VCH.
- ASTM E1944-98(2017): Standard Practice for Describing and Measuring Performance of NIR Instruments.

#### 3.4.1 DetectorRollOffAugmenter

**Class name:** `DetectorRollOffAugmenter`

**Effect:** Simulates detector sensitivity roll-off at spectral edges, causing increased noise and baseline distortion at the boundaries.

```python
from nirs4all.operators.augmentation import DetectorRollOffAugmenter

class DetectorRollOffAugmenter(SpectraTransformerMixin):
    def __init__(
        self,
        detector_model: str = "generic_nir",
        effect_strength: float = 1.0,
        noise_amplification: float = 0.02,
        include_baseline_distortion: bool = True,
        random_state: Optional[int] = None,
    ):
        ...
```

**Detector Models:**
- `"ingaas_standard"`: Standard InGaAs (1000-1600 nm optimal)
- `"ingaas_extended"`: Extended InGaAs (1100-2200 nm optimal)
- `"pbs"`: Lead sulfide (1000-2800 nm optimal)
- `"silicon_ccd"`: Silicon CCD (400-900 nm optimal)
- `"generic_nir"`: Generic NIR detector

**Notes:**
* `effect_strength` scales the overall roll-off effect (0-2)
* `noise_amplification` adds extra noise at low-sensitivity wavelengths
* Detector response curves based on manufacturer specifications

**Example:**
```python
from nirs4all.operators.augmentation import DetectorRollOffAugmenter
from sklearn.cross_decomposition import PLSRegression
import nirs4all

# Simulate InGaAs detector edge effects
pipeline = [
    DetectorRollOffAugmenter(detector_model="ingaas_standard", effect_strength=1.2),
    PLSRegression(n_components=10),
]

result = nirs4all.run(pipeline=pipeline, dataset="my_dataset")
```

#### 3.4.2 StrayLightAugmenter

**Class name:** `StrayLightAugmenter`

**Effect:** Simulates stray light contamination following the physics: T_observed = (T_true + s) / (1 + s)

Stray light causes non-linear compression of high-absorbance regions and affects the edges where detector sensitivity is lower.

```python
from nirs4all.operators.augmentation import StrayLightAugmenter

class StrayLightAugmenter(SpectraTransformerMixin):
    def __init__(
        self,
        stray_light_fraction: float = 0.001,
        edge_enhancement: float = 2.0,
        edge_width: float = 0.1,
        include_peak_truncation: bool = True,
        random_state: Optional[int] = None,
    ):
        ...
```

**Notes:**
* `stray_light_fraction`: Base stray light level (typical range: 0.0001-0.02)
* `edge_enhancement`: Factor by which stray light increases at edges
* Physics-based implementation following Beer-Lambert law deviations
* Reference: Workman & Weyer (2012), Chapter 5: Stray Light Effects

#### 3.4.3 EdgeCurvatureAugmenter

**Class name:** `EdgeCurvatureAugmenter`

**Effect:** Adds baseline curvature/bending at spectral edges, mimicking instrumental effects and optical path variations.

```python
from nirs4all.operators.augmentation import EdgeCurvatureAugmenter

class EdgeCurvatureAugmenter(SpectraTransformerMixin):
    def __init__(
        self,
        curvature_strength: float = 0.02,
        curvature_type: str = "random",
        asymmetry: float = 0.0,
        edge_focus: float = 0.7,
        random_state: Optional[int] = None,
    ):
        ...
```

**Curvature Types:**
- `"concave"`: Upward curving at edges
- `"convex"`: Downward curving at edges
- `"asymmetric"`: Different curvature at left and right edges
- `"random"`: Randomly selected per sample

**Notes:**
* `curvature_strength` controls the magnitude of baseline bending (0.01-0.1 typical)
* `asymmetry` parameter allows different effects at left vs right edges
* `edge_focus` controls how concentrated the effect is at edges (higher = more edge-focused)

#### 3.4.4 TruncatedPeakAugmenter

**Class name:** `TruncatedPeakAugmenter`

**Effect:** Adds truncated absorption peaks at spectral boundaries, simulating absorption bands whose centers lie outside the measured range.

```python
from nirs4all.operators.augmentation import TruncatedPeakAugmenter

class TruncatedPeakAugmenter(SpectraTransformerMixin):
    def __init__(
        self,
        peak_probability: float = 0.3,
        amplitude_range: Tuple[float, float] = (0.01, 0.1),
        width_range: Tuple[float, float] = (50, 200),
        left_edge: bool = True,
        right_edge: bool = True,
        random_state: Optional[int] = None,
    ):
        ...
```

**Notes:**
* Models half-Gaussian/half-Voigt peaks entering from outside the wavelength range
* Common in NIR where O-H and C-H overtone bands may extend beyond measurement limits
* `amplitude_range` in absorbance units (AU)
* `width_range` in nm for the peak half-width

#### 3.4.5 EdgeArtifactsAugmenter (Combined)

**Class name:** `EdgeArtifactsAugmenter`

**Effect:** Combines all edge artifact effects in a single augmenter for convenience.

```python
from nirs4all.operators.augmentation import EdgeArtifactsAugmenter

class EdgeArtifactsAugmenter(SpectraTransformerMixin):
    def __init__(
        self,
        detector_roll_off: bool = True,
        stray_light: bool = True,
        edge_curvature: bool = True,
        truncated_peaks: bool = True,
        overall_strength: float = 1.0,
        detector_model: str = "generic_nir",
        random_state: Optional[int] = None,
    ):
        ...
```

**Example - Robust Pipeline with Edge Artifacts:**
```python
from nirs4all.operators.augmentation import (
    TemperatureAugmenter,
    ParticleSizeAugmenter,
    EdgeArtifactsAugmenter,
)
from sklearn.cross_decomposition import PLSRegression
import nirs4all

# Comprehensive augmentation for field robustness
pipeline = [
    TemperatureAugmenter(temperature_range=(-5, 15)),
    ParticleSizeAugmenter(size_range_um=(30, 80)),
    EdgeArtifactsAugmenter(
        detector_model="ingaas_standard",
        overall_strength=0.8,
    ),
    PLSRegression(n_components=10),
]

result = nirs4all.run(pipeline=pipeline, dataset="field_samples")
```

### 3.5 Edge Artifacts in Synthetic Data Generation

The `SyntheticNIRSGenerator` supports edge artifacts through the `EdgeArtifactsConfig`:

```python
from nirs4all.synthesis import SyntheticNIRSGenerator, EdgeArtifactsConfig

# Configure edge artifacts for synthetic data
edge_config = EdgeArtifactsConfig(
    enable_detector_rolloff=True,
    enable_stray_light=True,
    enable_truncated_peaks=True,
    enable_edge_curvature=False,
    detector_model="ingaas_standard",
    rolloff_severity=0.5,
    stray_fraction=0.002,
    left_peak_amplitude=0.05,
    right_peak_amplitude=0.03,
)

generator = SyntheticNIRSGenerator(
    complexity="realistic",
    edge_artifacts_config=edge_config,
    random_state=42,
)

X, Y, E = generator.generate(n_samples=1000)
```

### 3.6 Fitting Edge Artifacts from Real Data

The `RealDataFitter` can automatically detect and characterize edge artifacts in real spectra:

```python
from nirs4all.synthesis import RealDataFitter

# Fit edge artifacts from real data
fitter = RealDataFitter()
params = fitter.fit(
    X_real,
    wavelengths=wavelengths,
    infer_edge_artifacts=True,
)

# Access inferred edge artifact characteristics
print(params.edge_artifact_inference.has_edge_artifacts)
print(params.edge_artifact_inference.detector_model)
print(params.edge_artifact_inference.has_truncated_peaks)

# Create generator matching real data edge characteristics
generator = fitter.create_matched_generator()
X_synth, Y_synth, E = generator.generate(n_samples=500)
```

---

## 4. Implementation details and utilities

### 4.1 Utility functions

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

### 4.2 Parameter validation

Each augmenter should validate ranges in `__init__`:

* Ensure min ≤ max for ranges.
* Ensure widths and kernel sizes are positive integers.
* Raise clear `ValueError` with short messages (for config debugging).

---

## 5. Integration into `nirs4all` pipelines

* Expose all classes in a dedicated module, e.g. `nirs4all.operators.augmentation`.

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

## 6. Testing guidelines

For each augmenter:

1. **Shape invariance:** `X_aug.shape == X.shape`.
2. **Determinism:** fixed `random_state` ⇒ identical outputs.
3. **Amplitude sanity:** output values stay in reasonable bounds for typical NIRS ranges.
4. **No NaN / inf:** unless explicitly requested (should not be).
5. **Wavelength handling:** wavelength-aware operators receive wavelengths automatically from the controller via kwargs.

Once implemented, add small end-to-end tests combining several augmenters in a pipeline to ensure they compose correctly.

