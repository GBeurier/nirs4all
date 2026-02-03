# Operator Base Class Design v2: Eliminating the Augmenter / Transformer Split

**Author:** Design Analysis (revision)
**Date:** 2026-02-03 (reviewed and updated)
**Status:** Proposal (supersedes previous version)
**Scope:** `nirs4all/operators/` base class hierarchy, controller integration

---

## Table of Contents

1. [Why the Previous Proposal Was Wrong](#section-1-why-the-previous-proposal-was-wrong)
2. [The Core Insight](#section-2-the-core-insight)
3. [Current Architecture: Where It Breaks](#section-3-current-architecture-where-it-breaks)
4. [The New Design](#section-4-the-new-design)
5. [Controller System: What Changes](#section-5-controller-system-what-changes)
6. [Operator-by-Operator Migration](#section-6-operator-by-operator-migration)
7. [Edge Cases and Risks](#section-7-edge-cases-and-risks)
8. [Point of View](#section-8-point-of-view)
9. [Migration Guide for Users](#section-9-migration-guide-for-users)

---

## Section 1: Why the Previous Proposal Was Wrong

The previous version of this document analyzed the same problem and recommended **Option B: `WavelengthAwareAugmenter`** -- a new third base class combining `Augmenter` and `SpectraTransformerMixin`. That was wrong. Here is why.

### 1.1 The Previous Proposal Made Things Worse

| Metric | Before (current) | Previous Option B | Delta |
|--------|-------------------|-------------------|-------|
| Base classes | 2 (`Augmenter`, `SpectraTransformerMixin`) | 3 (`Augmenter`, `SpectraTransformerMixin`, `WavelengthAwareAugmenter`) | +1 |
| Abstract method names | 2 (`augment`, `transform_with_wavelengths`) | 3 (`augment`, `transform_with_wavelengths`, `augment_with_wavelengths`) | +1 |
| Decision tree for new operators | 2 choices | 4 choices | +2 |
| Lines of framework code | ~290 | ~340+ | +50 |

The previous proposal treated the symptom (wavelength-aware augmenters can't have `apply_on`) by adding a third class that combines both capabilities. But this doesn't solve the root cause: **there should not be a separate `Augmenter` class in the first place**.

### 1.2 Three Method Names for the Same Operation

The fundamental design flaw is having three different method names for the same operation: "apply a transformation to a spectral matrix":

| Base class | Abstract method | Call chain |
|-----------|----------------|------------|
| `Augmenter` | `augment(X, apply_on)` | `transform(X)` -> `augment(X, self.apply_on)` |
| `SpectraTransformerMixin` | `transform_with_wavelengths(X, wl)` | `transform(X, wl)` -> `transform_with_wavelengths(X, wl)` |
| Previous Option B | `augment_with_wavelengths(X, wl, apply_on)` | `transform(X, wl)` -> `augment_with_wavelengths(X, wl, self.apply_on)` |

All three ultimately do the same thing: receive a 2D array `(n_samples, n_features)`, apply some transformation, return a 2D array of the same shape. The indirection (`transform` -> `augment`, `transform` -> `transform_with_wavelengths`) adds complexity without value.

### 1.3 What the Previous Analysis Got Right

The previous document correctly identified:
- The parallel augmentation wavelength bug in `SampleAugmentationController._emit_augmentation_steps_parallel()`
- The ad-hoc `lambda_axis` pattern in 6 augmenters
- The inconsistent RNG management (per-call vs per-instance)
- The need for protocol-based wavelength detection (`_requires_wavelengths` flag, not isinstance)

These are real problems. The solution just doesn't need a new base class.

---

## Section 2: The Core Insight

### 2.1 "Augmenter" vs "Transformer" Is a Controller Concern, Not an Operator Concern

A `SavitzkyGolay` can be used as an augmentation operator:
```python
{"sample_augmentation": {"transformers": [SavitzkyGolay(window_length=11)], "count": 3}}
```

A `GaussianAdditiveNoise` can be used as a transform step:
```python
pipeline = [GaussianAdditiveNoise(sigma=0.01), PLSRegression(10)]
```

The same operator class can serve both roles. Whether it acts as "augmentation" (creating sample copies) or "transformation" (modifying features in-place) depends on **which controller handles it**, not on what base class it inherits from:

| Role | Controller | What it does |
|------|-----------|-------------|
| Transform | `TransformerMixinController` | `fit(X_train)`, `transform(X_all)`, replace features |
| Sample augmentation | `SampleAugmentationController` | For each target sample: clone operator, `transform(sample)`, add augmented copy |
| Feature augmentation | `FeatureAugmentationController` | Run operator as transform, add result as new processing channel |

The **operator** doesn't need to know which role it plays. It just implements `transform(X)`. The **controller** decides how to use it.

### 2.2 `apply_on` Is a Regular Parameter

Looking at the actual usage of `apply_on` in augmenters, it controls the math:

```python
# GaussianAdditiveNoise.augment()
if apply_on == "global":
    scale = np.std(X) * self.sigma       # one std for entire matrix
else:
    stds = np.std(X, axis=1, keepdims=True)  # per-sample std
```

This is a parameter that affects the transformation behavior, like `sigma` or `n_components`. It does not need a base class. `apply_on` is never referenced in any controller -- it's purely internal to the operator's math. Confirmed by grep: zero occurrences of `apply_on` in `nirs4all/controllers/`.

### 2.3 The Only Structural Difference Is Wavelength Dependency

Some operators need wavelengths to do their math:
- `TemperatureAugmenter`: Applies region-specific effects based on wavelength ranges
- `LinearBaselineDrift`: Generates drift proportional to wavelength position
- `CropTransformer`: Crops spectral range by wavelength bounds

Others don't:
- `GaussianAdditiveNoise`: Adds noise proportional to signal magnitude
- `StandardScaler`: Centers and scales features
- `SavitzkyGolay`: Smooths by polynomial fitting over feature indices

This is the ONLY distinction that needs framework support (wavelength injection by the controller). Everything else (`apply_on`, `random_state`, `copy`) is just regular constructor parameters.

### 2.4 Wavelengths Are Just a Transform Parameter

The `SpectraTransformerMixin` already has the right signature: `transform(X, wavelengths=None)`. This is a clean extension of sklearn's `transform(X)` -- it adds an optional keyword argument with a default of `None`. Calling `transform(X)` still works. The controller injects `wavelengths` when the operator declares it needs them.

The `transform_with_wavelengths` indirection adds nothing. The validation (`if wavelengths is None and self._requires_wavelengths: raise ValueError`) can stay in `transform()` itself, or better yet, in the controller.

---

## Section 3: Current Architecture: Where It Breaks

### 3.1 Complete Operator Inventory

**26 operators inheriting `Augmenter`** (`operators/augmentation/`):

| Operator | Needs wavelengths? | Uses `apply_on`? | Uses `random_state`? |
|----------|-------------------|-----------------|---------------------|
| `IdentityAugmenter` | No | No (ignores) | No |
| `GaussianAdditiveNoise` | No | Yes (`global` vs `samples`) | Yes |
| `MultiplicativeNoise` | No | Yes (`global` vs `samples`) | Yes |
| `LinearBaselineDrift` | Optional (has `lambda_axis`) | No (ignores) | Yes |
| `PolynomialBaselineDrift` | Optional (has `lambda_axis`) | No (ignores) | Yes |
| `WavelengthShift` | Optional (has `lambda_axis`) | No (ignores) | Yes |
| `WavelengthStretch` | Optional (has `lambda_axis`) | No (ignores) | Yes |
| `LocalWavelengthWarp` | Optional (has `lambda_axis`) | No (ignores) | Yes |
| `SmoothMagnitudeWarp` | Optional (has `lambda_axis`) | No (ignores) | Yes |
| `BandPerturbation` | No | No (ignores) | Yes |
| `GaussianSmoothingJitter` | No | No (ignores) | Yes |
| `UnsharpSpectralMask` | No | No (ignores) | No |
| `BandMasking` | No | No (ignores) | Yes |
| `ChannelDropout` | No | No (ignores) | Yes |
| `SpikeNoise` | No | No (ignores) | Yes |
| `LocalClipping` | No | No (ignores) | Yes |
| `MixupAugmenter` | No | No (ignores) | Yes |
| `LocalMixupAugmenter` | No | No (ignores) | Yes |
| `ScatterSimulationMSC` | No | No (ignores) | Yes |
| `Spline_Smoothing` | No | No (ignores) | Yes |
| `Spline_X_Perturbations` | No | No (ignores) | Yes |
| `Spline_Y_Perturbations` | No | No (ignores) | Yes |
| `Spline_X_Simplification` | No | No (ignores) | Yes |
| `Spline_Curve_Simplification` | No | No (ignores) | Yes |
| `Rotate_Translate` | No | No (ignores) | Yes |
| `Random_X_Operation` | No | No (ignores) | Yes |

Key finding: **only 2 out of 26 augmenters actually use `apply_on` to change behavior** (`GaussianAdditiveNoise`, `MultiplicativeNoise`). The other 24 accept it in the signature but ignore it. This confirms that `apply_on` is not a core augmenter concept -- it's a parameter for a specific subset of operators.

**9 operators inheriting `SpectraTransformerMixin`** (`operators/augmentation/`):

| Operator | `_requires_wavelengths` | Has `random_state`? | Has `apply_on`? |
|----------|------------------------|--------------------|--------------------|
| `TemperatureAugmenter` | `True` | Yes (manual) | No |
| `MoistureAugmenter` | `True` | Yes (manual) | No |
| `ParticleSizeAugmenter` | `True` | Yes (manual) | No |
| `EMSCDistortionAugmenter` | `True` | Yes (manual) | No |
| `DetectorRollOffAugmenter` | `True` | Yes (manual) | No |
| `StrayLightAugmenter` | `True` | Yes (manual) | No |
| `EdgeCurvatureAugmenter` | `True` | Yes (manual) | No |
| `TruncatedPeakAugmenter` | `True` | Yes (manual) | No |
| `EdgeArtifactsAugmenter` | `True` | Yes (manual) | No |

"Manual" means each operator creates its own `np.random.default_rng(self.random_state)` inside `transform_with_wavelengths()` on every call instead of using a shared instance.

**~30+ operators using plain `TransformerMixin + BaseEstimator`** (`operators/transforms/`):

All NIRS transforms (SNV, MSC, SavitzkyGolay, derivatives, baseline corrections, signal conversions, feature selection, etc.) use plain sklearn base classes. None use `SpectraTransformerMixin` or `Augmenter`.

### 3.2 The Three Method Signatures in the Controller

`TransformerMixinController.execute()` must handle all operator types. Currently it branches:

```python
# Fitting
if needs_wavelengths:
    transformer.fit(fit_2d, wavelengths=wavelengths)
else:
    transformer.fit(fit_2d)

# Transforming
if needs_wavelengths:
    transformed_2d = transformer.transform(all_2d, wavelengths=wavelengths)
else:
    transformed_2d = transformer.transform(all_2d)
```

This branching appears 8+ times across the controller (main execute, sample augmentation batch, sample augmentation sequential, etc.). It's the direct consequence of having two different `transform()` signatures: `transform(X)` and `transform(X, wavelengths=None)`.

### 3.3 The Parallel Augmentation Bug

`SampleAugmentationController._emit_augmentation_steps_parallel()` bypasses `TransformerMixinController` and calls `fit()`/`transform()` directly. It does NOT pass wavelengths:

```python
# Line 518: cloned.fit(train_proc)  -- no wavelengths
# Line 540: transformed = fitted.transform(proc_data)  -- no wavelengths
```

Any `SpectraTransformerMixin`-based augmenter used in parallel sample augmentation will fail with `ValueError: requires wavelengths but none were provided`.

### 3.4 The `lambda_axis` Workaround

Six augmenters (`LinearBaselineDrift`, `PolynomialBaselineDrift`, `WavelengthShift`, `WavelengthStretch`, `LocalWavelengthWarp`, `SmoothMagnitudeWarp`) need wavelength positions for their math but inherit from `Augmenter` (no wavelength injection). They work around this with a `lambda_axis` constructor parameter:

```python
class LinearBaselineDrift(Augmenter):
    def __init__(self, ..., lambda_axis=None):
        self.lambda_axis = lambda_axis

    def augment(self, X, apply_on="samples"):
        lambdas = self.lambda_axis if self.lambda_axis is not None else np.arange(n_features)
```

This means:
- Users must manually provide `lambda_axis` at construction time
- The controller's wavelength injection doesn't reach these operators
- If `lambda_axis` is not provided, they fall back to integer indices, which gives wrong results for non-uniformly-spaced wavelengths

---

## Section 4: The New Design

### 4.1 Principle: One Interface, One Method Name

**All operators use `transform(X)` or `transform(X, wavelengths=None)`.**

No `augment()`. No `transform_with_wavelengths()`. No `augment_with_wavelengths()`.

The `SampleAugmentationController` and `FeatureAugmentationController` handle how the transformation is applied (as augmentation, as feature channel, etc.). The operator just transforms.

### 4.2 Class Hierarchy

```
sklearn.base.TransformerMixin + sklearn.base.BaseEstimator
    |
    +-- [External sklearn operators -- unchanged]
    |       StandardScaler, MinMaxScaler, PCA, PLSRegression, etc.
    |       transform(X) -- standard sklearn signature
    |
    +-- [nirs4all transforms -- unchanged]
    |       SavitzkyGolay, Wavelet, MSC, SNV, PyBaselineCorrection, etc.
    |       transform(X) -- standard sklearn signature
    |
    +-- SpectraTransformerMixin (SIMPLIFIED)
    |       _requires_wavelengths flag
    |       transform(X, wavelengths=None) -- single override point
    |       |
    |       +-- TemperatureAugmenter
    |       +-- MoistureAugmenter
    |       +-- ParticleSizeAugmenter
    |       +-- EMSCDistortionAugmenter
    |       +-- DetectorRollOffAugmenter
    |       +-- StrayLightAugmenter
    |       +-- EdgeCurvatureAugmenter
    |       +-- TruncatedPeakAugmenter
    |       +-- EdgeArtifactsAugmenter
    |       +-- LinearBaselineDrift (migrated from Augmenter)
    |       +-- PolynomialBaselineDrift (migrated)
    |       +-- WavelengthShift (migrated)
    |       +-- WavelengthStretch (migrated)
    |       +-- LocalWavelengthWarp (migrated)
    |       +-- SmoothMagnitudeWarp (migrated)
    |
    +-- [Former Augmenter subclasses -- now plain TransformerMixin]
            GaussianAdditiveNoise, MultiplicativeNoise, BandPerturbation,
            GaussianSmoothingJitter, SpikeNoise, MixupAugmenter, etc.
            transform(X) -- standard sklearn signature
```

**Result: 2 base classes in play** (`TransformerMixin`, `SpectraTransformerMixin`), **1 abstract override point** (`transform`), **0 nirs4all-specific abstract methods**.

### 4.3 Simplified `SpectraTransformerMixin`

```python
# nirs4all/operators/base/spectra_mixin.py

from typing import Optional, Union
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SpectraTransformerMixin(TransformerMixin, BaseEstimator):
    """
    Base class for wavelength-aware operators.

    Subclasses override transform(X, wavelengths=None) directly.
    The controller detects _requires_wavelengths and injects wavelengths automatically.

    Parameters
    ----------
    None -- this is a mixin class. Subclasses define their own parameters.

    Attributes
    ----------
    _requires_wavelengths : bool or str
        Class-level flag:
        - True: wavelengths required, controller raises ValueError if unavailable
        - "optional": wavelengths used if available, controller passes None if unavailable
        - False: wavelengths not needed (unusual for this class, but supported)
    """

    _requires_wavelengths: Union[bool, str] = True

    def fit(self, X, y=None, **fit_params):
        """No-op fit. Override if stateful fitting is needed.

        fit_params may include 'wavelengths' for wavelength-aware fitting.
        Subclasses needing wavelengths during fit should extract them:
            wavelengths = fit_params.get('wavelengths')
        """
        return self

    def transform(self, X, wavelengths: Optional[np.ndarray] = None):
        """Transform spectra, optionally using wavelength information.

        Subclasses override this method directly.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input spectra.
        wavelengths : ndarray of shape (n_features,) or None
            Wavelength array in nm. Injected by the controller when available.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed spectra.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement transform(X, wavelengths)"
        )

    def _more_tags(self):
        return {"allow_nan": False, "requires_wavelengths": self._requires_wavelengths}
```

**What changed from the current `SpectraTransformerMixin`:**

| Aspect | Before | After |
|--------|--------|-------|
| Abstract method | `transform_with_wavelengths(X, wavelengths)` | None -- subclasses override `transform()` directly |
| `transform()` body | Validates wavelengths, delegates to `transform_with_wavelengths` | `raise NotImplementedError` (subclasses override entirely) |
| Wavelength validation | In `transform()` | Removed -- controller already validates, or operator validates if needed |

The key change: **removing the `transform_with_wavelengths` indirection**. Subclasses override `transform(X, wavelengths=None)` directly. This eliminates one level of abstraction and one unique method name.

### 4.4 `Augmenter` Base Class: Deleted

The `Augmenter` class is deleted. Its former subclasses become plain `TransformerMixin + BaseEstimator` with `apply_on`, `random_state`, and `copy` as regular constructor parameters.

**What `Augmenter` provided and where it goes:**

| `Augmenter` feature | Where it goes |
|---------------------|--------------|
| `apply_on` constructor param | Regular constructor param on operators that use it |
| `random_state` constructor param | Regular constructor param (like many sklearn estimators: `RandomForestRegressor`, `KMeans`, etc.) |
| `random_gen = np.random.default_rng(random_state)` | Operators create local RNGs per `transform()` (or persist in `fit()` if stateful randomness is desired) |
| `copy` constructor param | Regular constructor param on operators that use it |
| `augment(X, apply_on)` abstract method | Eliminated -- operators implement `transform(X)` directly |
| `transform(X)` routing to `augment()` | Eliminated -- `transform(X)` IS the implementation |
| Global RNG seeding (`random.seed`, `np.random.seed`) | Eliminated -- this was a bad practice |
| `fit(X, y)` no-op | No longer needed -- `TransformerMixin` provides `fit_transform`, subclasses define `fit` if needed |
| `fit_transform(X, y)` bypass | Eliminated -- sklearn's default `fit_transform` (which calls `fit` then `transform`) is correct |
| `_more_tags()` | Operators define this if needed |

### 4.5 Concrete Examples: What Operators Look Like After Migration

#### Example 1: `GaussianAdditiveNoise` (former Augmenter, no wavelengths)

**Before:**
```python
class GaussianAdditiveNoise(Augmenter):
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 sigma=0.01, smoothing_kernel_width=1):
        super().__init__(apply_on, random_state, copy=copy)
        self.sigma = sigma
        self.smoothing_kernel_width = smoothing_kernel_width

    def augment(self, X, apply_on="samples"):
        if apply_on == "global":
            scale = np.std(X) * self.sigma
            noise = self.random_gen.normal(0, scale, size=X.shape)
        else:
            stds = np.std(X, axis=1, keepdims=True)
            noise = self.random_gen.normal(0, 1, size=X.shape) * stds * self.sigma
        ...
        return X + noise
```

**After:**
```python
class GaussianAdditiveNoise(TransformerMixin, BaseEstimator):
    def __init__(self, sigma=0.01, smoothing_kernel_width=1,
                 apply_on="samples", random_state=None):
        self.sigma = sigma
        self.smoothing_kernel_width = smoothing_kernel_width
        self.apply_on = apply_on
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rng = np.random.default_rng(self.random_state)
        if self.apply_on == "global":
            scale = np.std(X) * self.sigma
            noise = rng.normal(0, scale, size=X.shape)
        else:
            stds = np.std(X, axis=1, keepdims=True)
            noise = rng.normal(0, 1, size=X.shape) * stds * self.sigma
        ...
        return X + noise
```

Changes:
- Base class: `Augmenter` -> `TransformerMixin, BaseEstimator`
- No `super().__init__()` call needed
- `augment(X, apply_on)` -> `transform(X)` using `self.apply_on`
- `self.random_gen` -> local `rng = np.random.default_rng(self.random_state)`
- Global RNG seeding eliminated

#### Example 2: `TemperatureAugmenter` (former SpectraTransformerMixin, needs wavelengths)

**Before:**
```python
class TemperatureAugmenter(SpectraTransformerMixin):
    _requires_wavelengths = True

    def __init__(self, temperature_delta=5.0, ..., random_state=None):
        self.temperature_delta = temperature_delta
        ...
        self.random_state = random_state

    def transform_with_wavelengths(self, X, wavelengths):
        rng = np.random.default_rng(self.random_state)
        ...
        return result
```

**After:**
```python
class TemperatureAugmenter(SpectraTransformerMixin):
    _requires_wavelengths = True

    def __init__(self, temperature_delta=5.0, ..., random_state=None):
        self.temperature_delta = temperature_delta
        ...
        self.random_state = random_state

    def transform(self, X, wavelengths=None):
        rng = np.random.default_rng(self.random_state)
        ...
        return result
```

Changes:
- Base class: stays `SpectraTransformerMixin`
- `transform_with_wavelengths(X, wavelengths)` -> `transform(X, wavelengths=None)`
- Everything else identical

This is a minimal rename. The operator's logic doesn't change at all.

Note: if a future wavelength-aware operator needs stateful fitting (e.g., computing wavelength-dependent statistics during `fit()`), it would override `fit()` and extract wavelengths from `**fit_params`:
```python
def fit(self, X, y=None, **fit_params):
    wavelengths = fit_params.get('wavelengths')
    # ... compute stateful parameters using wavelengths ...
    return self
```
The controller already passes `wavelengths` to `fit()` as a keyword argument when the operator requires them. Currently, all 9 `SpectraTransformerMixin` operators have stateless `fit()` (no-op), so this is forward-looking.

#### Example 3: `LinearBaselineDrift` (former Augmenter with `lambda_axis`, migrates to SpectraTransformerMixin)

**Before:**
```python
class LinearBaselineDrift(Augmenter):
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 offset_range=(-0.1, 0.1), slope_range=(-0.001, 0.001),
                 lambda_axis=None):
        super().__init__(apply_on, random_state, copy=copy)
        self.offset_range = offset_range
        self.slope_range = slope_range
        self.lambda_axis = lambda_axis

    def augment(self, X, apply_on="samples"):
        n_samples, n_features = X.shape
        lambdas = self.lambda_axis if self.lambda_axis is not None else np.arange(n_features)
        lambdas_centered = lambdas - np.mean(lambdas)
        offsets = self.random_gen.uniform(self.offset_range[0], self.offset_range[1], size=(n_samples, 1))
        slopes = self.random_gen.uniform(self.slope_range[0], self.slope_range[1], size=(n_samples, 1))
        drift = offsets + slopes * lambdas_centered.reshape(1, -1)
        return X + drift
```

**After:**
```python
class LinearBaselineDrift(SpectraTransformerMixin):
    _requires_wavelengths = "optional"

    def __init__(self, offset_range=(-0.1, 0.1), slope_range=(-0.001, 0.001),
                 random_state=None):
        self.offset_range = offset_range
        self.slope_range = slope_range
        self.random_state = random_state

    def transform(self, X, wavelengths=None):
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        lambdas = wavelengths if wavelengths is not None else np.arange(n_features)
        lambdas_centered = lambdas - np.mean(lambdas)
        offsets = rng.uniform(self.offset_range[0], self.offset_range[1], size=(n_samples, 1))
        slopes = rng.uniform(self.slope_range[0], self.slope_range[1], size=(n_samples, 1))
        drift = offsets + slopes * lambdas_centered.reshape(1, -1)
        return X + drift
```

Changes:
- Base class: `Augmenter` -> `SpectraTransformerMixin`
- `_requires_wavelengths = "optional"` -- works with or without wavelengths
- `lambda_axis` constructor param: **deleted** -- wavelengths now injected by controller
- `apply_on` param: **deleted** -- this operator never used it (always ignored the argument)
- `augment(X, apply_on)` -> `transform(X, wavelengths=None)`
- `self.lambda_axis` -> `wavelengths` parameter (with fallback to `np.arange`)

#### Example 4: `BandPerturbation` (former Augmenter, no wavelengths, no apply_on)

**Before:**
```python
class BandPerturbation(Augmenter):
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 n_bands=3, band_width_range=(10, 50), perturbation_range=(-0.05, 0.05)):
        super().__init__(apply_on, random_state, copy=copy)
        self.n_bands = n_bands
        self.band_width_range = band_width_range
        self.perturbation_range = perturbation_range

    def augment(self, X, apply_on="samples"):
        ...  # never references apply_on
```

**After:**
```python
class BandPerturbation(TransformerMixin, BaseEstimator):
    def __init__(self, n_bands=3, band_width_range=(10, 50), perturbation_range=(-0.05, 0.05),
                 random_state=None):
        self.n_bands = n_bands
        self.band_width_range = band_width_range
        self.perturbation_range = perturbation_range
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ...
```

Changes:
- `apply_on` removed from constructor (was never used in the actual logic)
- Base class simplified
- `augment` -> `transform`

### 4.6 Decision Tree for New Operators

After migration, the decision is simple:

```
Does the operator need wavelengths?
  YES -> inherit SpectraTransformerMixin, set _requires_wavelengths
         override transform(X, wavelengths=None)
  NO  -> inherit TransformerMixin + BaseEstimator (or just use a plain sklearn transformer)
         override transform(X)
```

That's it. No more choosing between `Augmenter`, `SpectraTransformerMixin`, or both.

---

## Section 5: Controller System: What Changes

### 5.1 `TransformerMixinController._needs_wavelengths()`

**Before:**
```python
@staticmethod
def _needs_wavelengths(operator: Any) -> bool:
    return (
        isinstance(operator, SpectraTransformerMixin) and
        getattr(operator, '_requires_wavelengths', False)
    )
```

**After:**
```python
@staticmethod
def _wavelength_requirement(operator: Any) -> str:
    """Check if operator requires wavelengths via the _requires_wavelengths protocol.

    Returns:
        "required": operator needs wavelengths, raise error if unavailable
        "optional": operator uses wavelengths if available, works without them
        "none": operator doesn't need wavelengths
    """
    req = getattr(operator, '_requires_wavelengths', False)
    if req is True:
        return "required"
    if req == "optional":
        return "optional"
    if req is False or req is None:
        return "none"
    raise ValueError(
        f"Invalid _requires_wavelengths={req!r} on {operator.__class__.__name__}; "
        "use True, False, or 'optional'."
    )
```

**Valid `_requires_wavelengths` values:**

| Value | Meaning | Controller behavior |
|-------|---------|--------------------|
| `True` | Required | Raise `ValueError` if wavelengths unavailable |
| `"optional"` | Best-effort | Pass `None` if wavelengths unavailable |
| `False` / not set | Not needed | Never pass wavelengths |

Helper for extracting wavelengths based on requirement level:
```python
def _get_wavelengths(self, dataset, source_index, operator_name, requirement):
    """Extract wavelengths for the given requirement level."""
    if requirement == "none":
        return None
    elif requirement == "required":
        return self._extract_wavelengths(dataset, source_index, operator_name)
    else:  # "optional"
        try:
            return self._extract_wavelengths(dataset, source_index, operator_name)
        except ValueError:
            return None
```

This change makes wavelength detection **protocol-based** instead of **isinstance-based**. Any operator -- even a plain sklearn `TransformerMixin` subclass -- can opt into wavelength injection by setting `_requires_wavelengths = True` as a class attribute and accepting `wavelengths` in its `transform()`. No need to inherit from `SpectraTransformerMixin`.

It also distinguishes between `True` (required -- raise on failure) and `"optional"` (best-effort -- silently pass `None` if unavailable). This matters for the 6 `lambda_axis` operators that have their own `np.arange` fallback: they set `_requires_wavelengths = "optional"` and the controller won't crash when the dataset lacks wavelength headers.

`SpectraTransformerMixin` remains useful as a convenience base class (provides the `_requires_wavelengths` default and a no-op `fit`), but it's no longer the ONLY way to get wavelength injection.

### 5.2 The Wavelength if/else Branching Stays (And That's OK)

The controller still branches:
```python
wl_req = self._wavelength_requirement(op)
wavelengths = self._get_wavelengths(dataset, sd_idx, operator_name, wl_req)

if wavelengths is not None:
    transformer.transform(all_2d, wavelengths=wavelengths)
else:
    transformer.transform(all_2d)
```

This might seem like it could be simplified to always pass `wavelengths`:
```python
# DON'T do this
transformer.transform(all_2d, wavelengths=wavelengths)  # breaks sklearn operators!
```

But this would break external sklearn operators whose `transform(X)` doesn't accept `wavelengths`. The branching is necessary to maintain sklearn compatibility. It's not a design flaw -- it's the minimal cost of supporting two APIs in one controller.

### 5.3 Fix: `SampleAugmentationController._emit_augmentation_steps_parallel()`

The parallel path currently bypasses `TransformerMixinController` and calls `fit()`/`transform()` directly without wavelength injection. This must be fixed.

Note: `TransformerMixinController._execute_for_sample_augmentation()` and `_execute_for_sample_augmentation_sequential()` already handle wavelengths correctly. The bug is specifically in the parallel path inside `SampleAugmentationController` (`controllers/data/sample_augmentation.py`), which operates independently.

**Before (buggy, line ~518 and ~540 of `sample_augmentation.py`):**
```python
# Pre-fit transformers
for trans_idx, _ in active_transformers:
    ...
    for source_idx in range(n_sources):
        for proc_idx in range(n_processings):
            cloned = clone(transformer)
            train_proc = train_data[source_idx][:, proc_idx, :]
            cloned.fit(train_proc)  # BUG: no wavelengths
            all_fitted[(trans_idx, source_idx, proc_idx)] = cloned

# Transform in parallel
def process_transformer(args):
    ...
    for source_idx in range(n_sources):
        ...
        for proc_idx in range(n_processings):
            ...
            transformed = fitted.transform(proc_data)  # BUG: no wavelengths
```

**After (fixed):**

The fix uses the same `_wavelength_requirement` / `_get_wavelengths` pattern as `TransformerMixinController`. Since `SampleAugmentationController` is a separate class, it should either delegate to the controller's helpers or duplicate the protocol check. Delegating is cleaner. The simplest approach: import and use the same static method, or inline the protocol check.

> **Note on parallelism**: The code below works because `ThreadPoolExecutor` shares memory with the main thread. The `wavelengths_cache` dict and `all_fitted` dict are accessible inside `process_transformer` via closure capture. This is safe for read-only access from worker threads.

```python
# Pre-fit transformers
wavelengths_cache = {}  # source_idx -> wavelengths or _MISSING sentinel
_MISSING = object()

for trans_idx, _ in active_transformers:
    transformer = transformers[trans_idx]
    wl_req = TransformerMixinController._wavelength_requirement(transformer)

    for source_idx in range(n_sources):
        # Resolve wavelengths once per source
        if source_idx not in wavelengths_cache:
            try:
                wavelengths_cache[source_idx] = TransformerMixinController._extract_wavelengths(
                    dataset, source_idx, transformer.__class__.__name__
                )
            except ValueError:
                wavelengths_cache[source_idx] = _MISSING

        cached = wavelengths_cache[source_idx]
        if wl_req == "required" and cached is _MISSING:
            raise ValueError(
                f"{transformer.__class__.__name__} requires wavelengths, but none were found."
            )
        wavelengths = None if cached is _MISSING else cached

        for proc_idx in range(n_processings):
            cloned = clone(transformer)
            train_proc = train_data[source_idx][:, proc_idx, :]
            if wavelengths is not None:
                cloned.fit(train_proc, wavelengths=wavelengths)
            else:
                cloned.fit(train_proc)
            all_fitted[(trans_idx, source_idx, proc_idx)] = cloned

# Transform in parallel
# wavelengths_cache and all_fitted are captured by closure (read-only access is thread-safe)
def process_transformer(args):
    trans_idx, sample_ids = args
    transformer = transformers[trans_idx]
    wl_req = TransformerMixinController._wavelength_requirement(transformer)

    ...
    for source_idx in range(n_sources):
        cached = wavelengths_cache.get(source_idx, _MISSING)
        if wl_req == "required" and cached is _MISSING:
            raise ValueError(
                f"{transformer.__class__.__name__} requires wavelengths, but none were found."
            )
        wavelengths = None if cached is _MISSING else cached
        ...
        for proc_idx in range(n_processings):
            ...
            fitted = all_fitted[(trans_idx, source_idx, proc_idx)]
            if wavelengths is not None:
                transformed = fitted.transform(proc_data, wavelengths=wavelengths)
            else:
                transformed = fitted.transform(proc_data)
```

### 5.4 `FeatureAugmentationController`: No Changes

`FeatureAugmentationController` delegates to `step_runner.execute()` which routes to `TransformerMixinController`. The wavelength injection happens there. No changes needed.

### 5.5 Controller Detection Summary

| Check | Before | After |
|-------|--------|-------|
| Needs wavelengths? | `isinstance(op, SpectraTransformerMixin) and getattr(op, '_requires_wavelengths', False)` → always `bool` | `getattr(op, '_requires_wavelengths', False)` → returns `True`, `"optional"`, or `False`. Controller uses `_wavelength_requirement()` to get `"required"` / `"optional"` / `"none"`. |
| Get wavelengths | `_extract_wavelengths()` → raises on failure | `_get_wavelengths(requirement)` → raises for `"required"`, returns `None` for `"optional"` if unavailable |
| Is a transformer? | `isinstance(op, TransformerMixin)` | `isinstance(op, TransformerMixin)` (unchanged) |
| Is an augmenter? | Never actually checked in production code | Still not checked -- controller dispatch is by keyword |

---

## Section 6: Operator-by-Operator Migration

### 6.1 Execution Order (Important)

The phases should be executed in a different order than they are numbered:

| Step | Phase | Rationale |
|------|-------|----------|
| 1 | Phase 1 | Foundation: update `SpectraTransformerMixin`, fix controller, fix parallel bug |
| 2 | Phase 3 | Immediate win: 6 `lambda_axis` operators get automatic wavelength injection |
| 3 | Phase 2 | Bulk migration: 20 operators from `Augmenter` to plain `TransformerMixin` |
| 4 | Phase 4 | Cleanup: delete dead code, update exports, run full test suite |

Phase 3 comes before Phase 2 because the `lambda_axis` operators are more fragile (they have the workaround) and benefit most from the controller's wavelength injection. Phase 2 is mechanical and can be done in batches.

### 6.2 Phase 1: SpectraTransformerMixin + Controller (Revised)

Initially planned as infrastructure-only, but if we change `SpectraTransformerMixin.transform()` to raise `NotImplementedError`, existing subclasses that override `transform_with_wavelengths()` instead of `transform()` will break. So Phase 1 must be combined with the migration of those 9 subclasses.

**Atomic changes (all in one commit):

1. Simplify `SpectraTransformerMixin`:
   - Remove `transform_with_wavelengths()` abstract method
   - Change `transform()` to `raise NotImplementedError` default

2. Migrate 9 augmenters:

| Operator | File | Change |
|----------|------|--------|
| `TemperatureAugmenter` | `environmental.py` | Rename `transform_with_wavelengths` -> `transform` |
| `MoistureAugmenter` | `environmental.py` | Rename `transform_with_wavelengths` -> `transform` |
| `ParticleSizeAugmenter` | `scattering.py` | Rename `transform_with_wavelengths` -> `transform` |
| `EMSCDistortionAugmenter` | `scattering.py` | Rename `transform_with_wavelengths` -> `transform` |
| `DetectorRollOffAugmenter` | `edge_artifacts.py` | Rename `transform_with_wavelengths` -> `transform` |
| `StrayLightAugmenter` | `edge_artifacts.py` | Rename `transform_with_wavelengths` -> `transform` |
| `EdgeCurvatureAugmenter` | `edge_artifacts.py` | Rename `transform_with_wavelengths` -> `transform` |
| `TruncatedPeakAugmenter` | `edge_artifacts.py` | Rename `transform_with_wavelengths` -> `transform` |
| `EdgeArtifactsAugmenter` | `edge_artifacts.py` | Rename `transform_with_wavelengths` -> `transform` |

For each: rename `transform_with_wavelengths(self, X, wavelengths)` to `transform(self, X, wavelengths=None)`.

3. Update controller: Replace `_needs_wavelengths()` with `_wavelength_requirement()` and `_get_wavelengths()` (Section 5.1).

4. Fix parallel augmentation bug.

5. Update tests: Replace `assert isinstance(aug, SpectraTransformerMixin)` with functional tests where needed. Replace calls to `aug.transform_with_wavelengths(X, wl)` with `aug.transform(X, wavelengths=wl)`.

### 6.3 Phase 2: Delete `Augmenter`, migrate 20 non-wavelength augmenters

**Delete `abc_augmenter.py`** (or keep only `IdentityAugmenter` as a plain transformer).

Migrate 20 operators to `TransformerMixin + BaseEstimator`:

| Operator | File | Key changes |
|----------|------|-------------|
| `GaussianAdditiveNoise` | `spectral.py` | Keep `apply_on`, `random_state`. Rename `augment` -> `transform`. |
| `MultiplicativeNoise` | `spectral.py` | Keep `apply_on`, `random_state`. Rename `augment` -> `transform`. |
| `BandPerturbation` | `spectral.py` | Drop `apply_on` (unused). Rename `augment` -> `transform`. |
| `GaussianSmoothingJitter` | `spectral.py` | Drop `apply_on` (unused). Rename `augment` -> `transform`. |
| `UnsharpSpectralMask` | `spectral.py` | Drop `apply_on` (unused). Rename `augment` -> `transform`. |
| `BandMasking` | `spectral.py` | Drop `apply_on` (unused). Rename `augment` -> `transform`. |
| `ChannelDropout` | `spectral.py` | Drop `apply_on` (unused). Rename `augment` -> `transform`. |
| `SpikeNoise` | `spectral.py` | Drop `apply_on` (unused). Rename `augment` -> `transform`. |
| `LocalClipping` | `spectral.py` | Drop `apply_on` (unused). Rename `augment` -> `transform`. |
| `MixupAugmenter` | `spectral.py` | Drop `apply_on` (unused). Rename `augment` -> `transform`. |
| `LocalMixupAugmenter` | `spectral.py` | Drop `apply_on` (unused). Rename `augment` -> `transform`. |
| `ScatterSimulationMSC` | `spectral.py` | Drop `apply_on` (unused). Rename `augment` -> `transform`. |
| `Spline_Smoothing` | `splines.py` | Drop `apply_on` (unused). Rename `augment` -> `transform`. |
| `Spline_X_Perturbations` | `splines.py` | Drop `apply_on` (unused). Rename `augment` -> `transform`. |
| `Spline_Y_Perturbations` | `splines.py` | Drop `apply_on` (unused). Rename `augment` -> `transform`. |
| `Spline_X_Simplification` | `splines.py` | Drop `apply_on` (unused). Rename `augment` -> `transform`. |
| `Spline_Curve_Simplification` | `splines.py` | Drop `apply_on` (unused). Rename `augment` -> `transform`. |
| `Rotate_Translate` | `random.py` | Drop `apply_on` (unused). Rename `augment` -> `transform`. |
| `Random_X_Operation` | `random.py` | Drop `apply_on` (unused). Rename `augment` -> `transform`. |
| `IdentityAugmenter` | `abc_augmenter.py` | Becomes trivial `TransformerMixin` with `transform(X): return X` |

**Pattern for each migration** (operators that DON'T use `apply_on`):
```python
# Before
class BandPerturbation(Augmenter):
    def __init__(self, apply_on="samples", random_state=None, *, copy=True, ...):
        super().__init__(apply_on, random_state, copy=copy)
        ...

    def augment(self, X, apply_on="samples"):
        ...  # never uses apply_on

# After
class BandPerturbation(TransformerMixin, BaseEstimator):
    def __init__(self, ..., random_state=None):
        ...
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rng = np.random.default_rng(self.random_state)
        ...
```

**Pattern for operators that USE `apply_on`** (only `GaussianAdditiveNoise` and `MultiplicativeNoise`):
```python
class GaussianAdditiveNoise(TransformerMixin, BaseEstimator):
    def __init__(self, sigma=0.01, ..., apply_on="samples", random_state=None):
        self.sigma = sigma
        ...
        self.apply_on = apply_on
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rng = np.random.default_rng(self.random_state)
        if self.apply_on == "global":
            ...
        else:
            ...
```

### 6.4 Phase 3: Migrate 6 `lambda_axis` augmenters to `SpectraTransformerMixin`

| Operator | File | Key changes |
|----------|------|-------------|
| `LinearBaselineDrift` | `spectral.py` | `Augmenter` -> `SpectraTransformerMixin`, `_requires_wavelengths="optional"`, drop `lambda_axis` and `apply_on`, rename `augment` -> `transform` |
| `PolynomialBaselineDrift` | `spectral.py` | Same pattern |
| `WavelengthShift` | `spectral.py` | Same pattern |
| `WavelengthStretch` | `spectral.py` | Same pattern |
| `LocalWavelengthWarp` | `spectral.py` | Same pattern |
| `SmoothMagnitudeWarp` | `spectral.py` | Same pattern |

These 6 operators:
1. Change base to `SpectraTransformerMixin`
2. Set `_requires_wavelengths = "optional"` (they work without wavelengths via `np.arange()` fallback)
3. Remove `lambda_axis` constructor param (wavelengths now injected by controller)
4. Remove `apply_on` (none of them use it)
5. Rename `augment(X, apply_on)` -> `transform(X, wavelengths=None)`
6. Replace `self.lambda_axis` -> `wavelengths` parameter

### 6.5 Phase 4: Cleanup

1. Delete `abc_augmenter.py` (or reduce to just `IdentityAugmenter` as utility)
2. Remove `Augmenter` from `augmentation/__init__.py` exports
3. Update `augmentation/__init__.py` imports
4. Grep for `import.*Augmenter`, `isinstance.*Augmenter` -- fix all references
5. Grep for `lambda_axis` -- confirm complete removal
6. Grep for `transform_with_wavelengths` -- confirm complete removal
7. Grep for `\.augment(` -- confirm no remaining calls to the old abstract method
8. Update webapp node definitions (remove `lambda_axis`, `apply_on` where dropped, add `apply_on` where kept)
9. Run examples: `cd nirs4all/examples && ./run.sh -q`

### 6.6 Breaking Changes Summary

| Breaking change | Scope | Why it's OK |
|----------------|-------|-------------|
| `Augmenter` class deleted | Anyone importing `Augmenter` | No backward compat per CLAUDE.md. User code should inherit from `TransformerMixin` directly. |
| `augment()` method removed | Anyone calling `.augment()` directly | Should use `.transform()` instead. Controller already uses `.transform()`. |
| `transform_with_wavelengths()` removed | Anyone calling it directly | Should use `.transform(X, wavelengths=wl)` instead. |
| `lambda_axis` parameter removed from 6 operators | Anyone constructing with `lambda_axis=...` | Remove the parameter -- controller now injects wavelengths automatically. |
| `apply_on` removed from ~24 operators that ignored it | Anyone passing `apply_on=` to these operators | Remove the parameter -- it had no effect anyway. |
| `IdentityAugmenter` changes base | Anyone using `isinstance(..., Augmenter)` | Check for `TransformerMixin` instead, or use duck typing. |

### 6.7 Testing Strategy

| Phase | Tests |
|-------|-------|
| Phase 1 | For each of 9 migrated SpectraTransformerMixin operators: same output given same input + wavelengths. `transform(X, wavelengths=wl)` works. `clone()` works. `get_params()`/`set_params()` works. Parallel sample augmentation with wavelength-aware operator works. |
| Phase 2 | For each of 20 migrated Augmenter operators: same output given same input + random_state. `transform(X)` works. `clone()` works. `get_params()`/`set_params()` works. Used in `sample_augmentation` pipeline step: works. Used as plain transform step: works. |
| Phase 3 | For each of 6 migrated lambda_axis operators: same output when wavelengths match old lambda_axis. Controller automatically injects wavelengths. Fallback to `np.arange` works when wavelengths unavailable. |
| Phase 4 | Full integration: `pytest tests/`. Examples: `./run.sh -q`. No remaining references to deleted symbols. |

---

## Section 7: Edge Cases and Risks

### 7.1 RNG Behavior Change

**Issue:** The current `Augmenter.transform()` seeds the global RNG:
```python
if self.random_state is not None:
    random.seed(self.random_state)
    np.random.seed(self.random_state)
```

After migration, operators use `np.random.default_rng(self.random_state)` locally. This changes behavior:
- Global RNG state no longer polluted (good)
- Other code that relied on global seeding may get different random sequences (unlikely but possible)
- The `random` module (stdlib) is no longer seeded (no operator uses `random.random()` directly anyway)

**Mitigation:** This is a deliberate improvement. The global seeding was a bad practice. Document it in changelog.

### 7.2 Pickle Compatibility for Saved Bundles

**Issue:** `.n4a` bundles contain pickled fitted transformers. If a user has a bundle containing a fitted `TemperatureAugmenter` saved before migration, the pickle includes the class path `nirs4all.operators.augmentation.environmental.TemperatureAugmenter`.

Since the class stays in the same file and the same module, the pickle will load correctly. The base class change doesn't affect pickle loading -- pickle stores the class name and module, not the inheritance chain.

However, if the class was an `Augmenter` subclass and the `Augmenter` class is deleted, unpickling will fail IF the pickle tries to instantiate `Augmenter` directly (which it doesn't -- it instantiates the concrete subclass).

There is a subtlety with `clone()` and `get_params()`: sklearn's `clone()` inspects `__init__` parameters via `get_params()`. If an old pickle has `self.apply_on` in its `__dict__` but the new class no longer accepts `apply_on` in `__init__`, then:
- `unpickle()` works (restores `__dict__` directly, no constructor call)
- `transform()` works (the attribute still exists)
- `clone()` **may break** (inspects `__init__` signature, finds `apply_on` has no matching parameter)
- `get_params()` returns only params matching `__init__` -- old params silently dropped

For `.n4a` bundles, the typical flow is `unpickle -> transform` (prediction), not `unpickle -> clone`. So this is safe for the primary use case. Retraining from old bundles that `clone()` fitted operators would fail for operators whose constructor changed. This is acceptable per CLAUDE.md's "no backward compatibility" policy.

**Mitigation:** Since pickles reference the concrete class, not the base class, this should be safe for prediction. Test with actual `.n4a` bundles after migration. Document the breaking change for retraining from pre-migration bundles.

### 7.3 `copy` Parameter

**Issue:** The current `Augmenter` has `copy=True` in the constructor. After migration, this parameter is removed from most operators (they weren't using it -- the commented-out `_validate_data` call was the only consumer).

**Mitigation:** Grep for `self.copy` in augmenter implementations. If any operator actually uses it, keep the parameter. From the current code, `Augmenter.transform()` has a commented-out `_validate_data` call with `copy=self.copy` -- this was never active. No operator uses `self.copy` in its `augment()` method.

### 7.4 `fit_transform` Behavior Change

**Issue:** The current `Augmenter.fit_transform()` calls `self.transform(X)` directly, bypassing `fit()`. This means `fit()` is never called when `fit_transform()` is used. After migration, operators inherit sklearn's default `fit_transform()` which calls `self.fit(X, y).transform(X)`.

**Mitigation:** Since `fit()` is a no-op for all augmenters, this change is functionally identical. But it's worth noting.

### 7.5 Webapp Node Definitions

**Issue:** The webapp's node registry (`nirs4all-webapp/src/data/nodes/definitions/`) defines parameter schemas for each operator. Changes include:
- `lambda_axis` removed from 6 operators
- `apply_on` removed from ~24 operators that ignored it
- `apply_on` kept for `GaussianAdditiveNoise` and `MultiplicativeNoise`
- `copy` parameter removed from all former `Augmenter` subclasses

**Mitigation:** Update node definitions as part of Phase 4. The webapp should be able to run without nirs4all installed (per CLAUDE.md), so node definitions are independent of the library code.

### 7.6 Synthesis Generator Compatibility

**Issue:** The synthesis generator (`synthesis/generator.py`) imports and instantiates augmenters directly:
```python
from nirs4all.operators.augmentation import (
    TemperatureAugmenter, MoistureAugmenter,
    ParticleSizeAugmenter, EMSCDistortionAugmenter, ...
)
```

It calls their `transform()` methods, passing wavelengths. Currently, `operator.transform(X, wavelengths=wl)` on a `SpectraTransformerMixin` delegates to `transform_with_wavelengths()`. After migration, `transform(X, wavelengths=wl)` goes directly to the operator's implementation. The call site `operator.transform(X, wavelengths=wl)` is identical before and after -- only the internal dispatch changes.

**Mitigation:** No changes needed in the generator for the SpectraTransformerMixin operators. For the former `Augmenter` operators that the generator uses directly (if any), ensure they're called via `transform()` (which they should already be, since `Augmenter.transform()` is the public API).

### 7.7 Wavelength Flag / Signature Mismatch

**Issue:** The controller decides to pass `wavelengths=...` based on `_requires_wavelengths`. If an operator sets `_requires_wavelengths = True` but its `transform()` signature does not accept `wavelengths`, a `TypeError` will surface at runtime.

**Mitigation:** Treat this as a contract: any operator that sets `_requires_wavelengths` must accept `wavelengths` in `transform()` and `fit()` (if it overrides `fit`). Add a validation check in the controller (e.g., `inspect.signature`) in debug/testing builds, or a small unit test that enforces this contract for all registered operators.

### 7.8 Wavelength Cache Cross-Contamination

**Issue:** A cache that stores `None` for a source because an *optional* operator had missing wavelengths can incorrectly suppress the error for a *required* operator later in the same run. The first optional lookup "poisons" the cache for that source.

**Mitigation:** Cache the raw extraction result with a sentinel for "missing", then apply the requirement check per operator (as in Section 5.3). This ensures required operators still raise even if an optional operator previously saw missing wavelengths.

---

## Section 8: Point of View

### 8.1 This Is the Right Design

The previous proposal added complexity to solve a symptom. This design removes complexity by addressing the root cause: **the `Augmenter` base class and `transform_with_wavelengths` abstraction have no reason to exist**.

The evidence:
- `apply_on` is used by 2 out of 26 `Augmenter` subclasses. It's a parameter for 2 operators, not a framework concept.
- `augment()` is just `transform()` with an extra argument that 24 out of 26 implementations ignore.
- `transform_with_wavelengths()` is just `transform()` with a different name.
- No controller ever checks `isinstance(op, Augmenter)`. The framework doesn't distinguish augmenters from transformers at the type level.
- The `SampleAugmentationController` accepts any `TransformerMixin` -- it doesn't require `Augmenter` subclasses.

When a base class provides functionality that its subclasses almost universally ignore, and the framework doesn't use the base class for dispatch, that base class is dead weight.

### 8.2 The Pragmatic Benefits

**For operator authors**: One decision (need wavelengths or not?), one method to implement (`transform`), done.

**For the controller system**: One protocol check (`_requires_wavelengths`), two levels (`"required"` vs `"optional"`), one injection path (`wavelengths` kwarg), done.

**For the synthesis module**: The generator calls `transform(X, wavelengths=wl)` on all wavelength-aware operators. After this refactoring, that's the same signature for all of them -- no need to check base class types.

**For users**: `SavitzkyGolay` in a `sample_augmentation` step works the same as `GaussianAdditiveNoise` in a `sample_augmentation` step. Both are just transformers. Users don't need to understand the difference between "augmenter" and "transformer" to build pipelines.

### 8.3 What I'd Watch Out For

**1. The `random_state` and reproducibility contract.** After removing `Augmenter`, each stochastic operator manages its own RNG. There are two *conflicting* expectations in the codebase today:

- **Reproducible-per-call**: calling `transform()` multiple times with the same inputs yields the same output (current `Augmenter` behavior via global seeding).
- **Stochastic-per-call**: repeated calls yield different outputs unless the operator is re-seeded (expected by many augmentation users).

This proposal must resolve the contract explicitly, otherwise reproducibility will remain inconsistent across operators and controllers.

**Recommended contract (explicit):**

| Scenario | Behavior |
|----------|----------|
| `random_state=42`, single `transform()` call | Deterministic |
| `random_state=42`, multiple `transform()` calls on same instance | Different results (RNG advances) |
| `random_state=42`, `clone()` then `transform()` | Same as first call (RNG resets) |
| `random_state=None` | Non-deterministic |

**Implementation pattern:**

```python
class SomeAugmenter(TransformerMixin, BaseEstimator):
    def __init__(self, ..., random_state=None):
        self.random_state = random_state

    def fit(self, X, y=None):
        # Seed once per fitted instance
        self._rng = np.random.default_rng(self.random_state)
        return self

    def transform(self, X):
        # Use and advance RNG on each call
        rng = self._rng if hasattr(self, "_rng") else np.random.default_rng(self.random_state)
        ...
```

This preserves reproducibility while avoiding the "same output every call" trap. It also keeps `clone()` semantics consistent with scikit-learn.

**Controller impact:** The `SampleAugmentationController` should continue cloning per augmented sample; this keeps each clone reproducible if `random_state` is fixed, while still producing different samples within a batch.

**Migration note:** If some operators remain stateless (no `fit()` override), they should still use a per-instance RNG stored on first call to `transform()` to avoid identical outputs on repeated calls.

**2. The `apply_on` orphans.** Only `GaussianAdditiveNoise` and `MultiplicativeNoise` keep `apply_on`. This is a one-off parameter on 2 operators out of 35+. It's fine -- many sklearn operators have unique parameters. But if more noise operators are added later, consider whether `apply_on` should become a convention documented in a guide rather than a parameter name.

**3. Scope of change.** This migration touches 35 operator files + 2 controller files + tests. It should be done in focused phases with test passes between each. Don't try to do it in one mega-commit.

### 8.4 Timeline: Before Synthesis Refactoring

This migration MUST be completed **before** the synthesis externalization described in [synthesis_augmentation_refactoring.md](synthesis_augmentation_refactoring.md). The synthesis refactoring creates new augmenters and reorganizes the generator. Doing that on top of the old `Augmenter` hierarchy and then migrating everything again would be wasted effort.

**Dependency chain:**
1. **This refactoring** → Clean operator hierarchy
2. **Synthesis refactoring** → Uses clean operators
3. **Synthesis as pipeline** → Depends on both

### 8.5 Summary Table

| Metric | Before | After |
|--------|--------|-------|
| nirs4all base classes | 2 (`Augmenter`, `SpectraTransformerMixin`) | 1 (`SpectraTransformerMixin`) |
| Abstract methods | 2 (`augment`, `transform_with_wavelengths`) | 0 (subclasses override `transform` directly) |
| Method names to learn | 3 (`transform`, `augment`, `transform_with_wavelengths`) | 1 (`transform`) |
| Lines in base classes | ~160 (`abc_augmenter.py`) + ~160 (`spectra_mixin.py`) = ~320 | ~50 (`spectra_mixin.py`) |
| Decision tree for new operators | "Need wavelengths? Need apply_on? Both?" | "Need wavelengths? Yes/No." |
| Wavelength detection | isinstance + getattr | getattr only (protocol-based) |
| Parallel augmentation bug | Present | Fixed |
| `lambda_axis` workaround | 6 operators | 0 (controller injects wavelengths) |
| Valid `_requires_wavelengths` values | `True` / `False` | `True` / `"optional"` / `False` |

---

### 8.6 My Point of View

This design is the right one, and I'm confident in it — not because it's elegant (though it is), but because the evidence is overwhelming. The `Augmenter` class is a historical artifact that no longer serves a purpose. The numbers tell the story: 0 isinstance checks in production, 24/26 subclasses ignoring `apply_on`, the method name indirection adding zero value. This isn't a matter of taste; the code is telling us what to do.

**What I genuinely like about this design:**

The single strongest aspect is the insight that augmenter-vs-transformer is a controller concern. This is not an obvious realization — it requires understanding how the controller dispatch actually works (by pipeline keyword, not by operator type). Once you see it, the entire `Augmenter` hierarchy becomes transparently unnecessary. The fact that `SavitzkyGolay` can already be used in a `sample_augmentation` step today without inheriting from `Augmenter` proves the point better than any design argument.

The protocol-based wavelength detection (`getattr` instead of `isinstance`) is also genuinely good engineering. It decouples the operator from the framework — any operator from any library can opt into wavelength injection by adding a single class attribute. This matters for extensibility and for third-party integrations.

**Where I have reservations:**

1. **The `"optional"` string value is a code smell.** Using `Union[bool, str]` for `_requires_wavelengths` where the string is specifically `"optional"` is not great. A proper enum (`WavelengthRequirement.REQUIRED`, `.OPTIONAL`, `.NONE`) or a dedicated pair of flags would be cleaner. I kept it as a string because it's simpler and matches the "minimal complexity" philosophy, but I acknowledge it's slightly unprincipled. If the codebase grows more requirement levels later (e.g., `"preferred"` — use if available but don't warn), the string approach will become messy.

2. **The `random_state` reproducibility question is unresolved.** The document recommends `rng = np.random.default_rng(self.random_state)` at the top of every `transform()` call. This gives reproducible-per-call behavior, which is what the current `Augmenter` does (via global seeding). But it means that calling `transform()` twice on the same operator with the same data produces **identical** output — which is wrong for augmentation, where you want variety. The `SampleAugmentationController` works around this by cloning the operator for each augmented sample, which resets the RNG. But if someone uses the operator standalone in a loop without cloning, they'll get identical augmentations. This is a known limitation of the current design too, but it's worth flagging that we're not fixing it — just not making it worse.

3. **The scope is large for what seems like a simple refactoring.** 35 operators + 2 controllers + tests + webapp node definitions. The mechanical parts (rename `augment` → `transform`, drop `apply_on`) are straightforward but tedious. The risk isn't in any single change; it's in the sheer number of files touched and the possibility of missing one. The phased approach mitigates this, but it still requires discipline.

**What this enables that the document doesn't emphasize enough:**

The biggest downstream win is for the synthesis-augmentation convergence described in [synthesis_augmentation_refactoring.md](synthesis_augmentation_refactoring.md). That document’s Section 7.4 shows the generator using a dual dispatch pattern:

```python
for augmenter in self._effect_chain:
    if isinstance(augmenter, SpectraTransformerMixin):
        A = augmenter.transform(A, wavelengths=self.wavelengths)
    else:
        A = augmenter.transform(A)
```

After this refactoring, this simplifies to the same pattern the controllers use: check `_requires_wavelengths`, conditionally pass wavelengths. More importantly, it means the generator, the pipeline controllers, and standalone usage all call the same method the same way. That's real unification — not at the class hierarchy level, but at the call site level, which is where it actually matters.

It also unblocks the "synthesis as pipeline" idea (see [synthesis_as_pipeline.md](../_internal/synthesis_as_pipeline.md)). If augmenters are just transformers, then a synthesis pipeline can freely mix generative steps and augmentation steps without type gymnastics. The pipeline system already knows how to handle `TransformerMixin` instances — after this refactoring, all augmenters are `TransformerMixin` instances.

---

## Section 9: Migration Guide for Users

### 9.1 If You Have Custom Augmenters Inheriting from `Augmenter`

**Before (deprecated):**
```python
from nirs4all.operators.augmentation import Augmenter

class MyAugmenter(Augmenter):
    def __init__(self, apply_on="samples", random_state=None, *, copy=True, my_param=1.0):
        super().__init__(apply_on, random_state, copy=copy)
        self.my_param = my_param

    def augment(self, X, apply_on="samples"):
        # Your transformation logic
        return X_transformed
```

**After:**
```python
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np

class MyAugmenter(TransformerMixin, BaseEstimator):
    def __init__(self, my_param=1.0, random_state=None):
        self.my_param = my_param
        self.random_state = random_state

    def fit(self, X, y=None):
        self._rng = np.random.default_rng(self.random_state)
        return self

    def transform(self, X):
        # Your transformation logic (use self._rng for randomness)
        return X_transformed
```

### 9.2 If You Have Custom Operators Inheriting from `SpectraTransformerMixin`

**Before:**
```python
class MyWavelengthOp(SpectraTransformerMixin):
    def transform_with_wavelengths(self, X, wavelengths):
        # Use wavelengths
        return X_transformed
```

**After:**
```python
class MyWavelengthOp(SpectraTransformerMixin):
    def transform(self, X, wavelengths=None):
        # Use wavelengths (same logic)
        return X_transformed
```

Just rename the method. The signature changes from `(X, wavelengths)` to `(X, wavelengths=None)`.

### 9.3 If You Were Passing `lambda_axis` to Augmenters

**Before:**
```python
LinearBaselineDrift(lambda_axis=wavelengths, offset_range=(-0.1, 0.1))
```

**After:**
```python
LinearBaselineDrift(offset_range=(-0.1, 0.1))  # wavelengths injected automatically
```

The controller now injects wavelengths automatically. Remove the `lambda_axis` parameter.

### 9.4 If You Were Passing `apply_on` to Operators That Ignored It

Most operators ignored `apply_on`. Only `GaussianAdditiveNoise` and `MultiplicativeNoise` actually use it. For all other operators, remove the parameter:

**Before:**
```python
BandPerturbation(apply_on="samples", n_bands=3)
```

**After:**
```python
BandPerturbation(n_bands=3)
```

**My honest assessment of risk:**

Low. The changes are mechanical and well-defined. The phased approach means each commit is testable in isolation. The breaking changes are acceptable given the codebase's "no backward compatibility" policy. The parallel augmentation bug fix is the most valuable single change — it's a real correctness issue today, not a theoretical one.

The one thing I'd push back on is the timeline. This should be done **before** the synthesis externalization, not after. The synthesis refactoring creates new augmenters and reorganizes the generator — doing that on top of the old `Augmenter` hierarchy and then migrating everything again would be wasted effort. Clean the foundation first, then build on it.
