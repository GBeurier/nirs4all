# Implementation Roadmap v2: Operator Refactoring & Controller-Managed Variation

**Date:** February 2026
**Status:** Roadmap
**Supersedes:** implementation_roadmap.md (v1)

---

## Executive Summary

This roadmap refactors the operator/controller architecture to achieve:

1. **Pure TransformerMixin operators** — Operators just transform features. No application logic (`apply_on`), no copy management.
2. **Controller-managed variation** — The `SampleAugmentationController` manages HOW transformations are applied (per-sample, global, per-group).
3. **Full sklearn compatibility** — All operators use standard `fit(X, y=None, **kwargs)` and `transform(X, **kwargs)` signatures.
4. **Hybrid performance model** — Performance-optimized operators can handle variation internally; others are handled by the controller.

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Naming Decision: `apply_on` → `variation_scope`](#2-naming-decision-apply_on--variation_scope)
3. [Phase 1: SpectraTransformerMixin Simplification](#3-phase-1-spectratransformermixin-simplification)
4. [Phase 2: Augmenter Base Class Elimination](#4-phase-2-augmenter-base-class-elimination)
5. [Phase 3: SampleAugmentationController Variation Management](#5-phase-3-sampleaugmentationcontroller-variation-management)
6. [Phase 4: Parallel Augmentation & Wavelength Fixes](#6-phase-4-parallel-augmentation--wavelength-fixes)
7. [Phase 5: Synthesis Transfer & Externalization](#7-phase-5-synthesis-transfer--externalization)
8. [Phase 6: Webapp & Documentation](#8-phase-6-webapp--documentation)
9. [Task Dependency Graph](#9-task-dependency-graph)
10. [Verification Checklist](#10-verification-checklist)

---

## 1. Design Principles

### 1.1 Operators Are Pure Transformers

**Before (current):**
```python
class GaussianAdditiveNoise(Augmenter):
    def __init__(self, apply_on="samples", random_state=None, *, copy=True, sigma=0.01):
        super().__init__(apply_on, random_state, copy=copy)
        self.sigma = sigma

    def augment(self, X, apply_on="samples"):
        if apply_on == "global":
            scale = np.std(X) * self.sigma
            noise = self.random_gen.normal(0, scale, size=X.shape)
        else:  # per-sample
            stds = np.std(X, axis=1, keepdims=True)
            noise = self.random_gen.normal(0, 1, size=X.shape) * stds * self.sigma
        return X + noise
```

**After (target):**
```python
class GaussianAdditiveNoise(TransformerMixin, BaseEstimator):
    def __init__(self, sigma: float = 0.01, random_state: int = None):
        self.sigma = sigma
        self.random_state = random_state

    def fit(self, X, y=None, **kwargs):
        self._rng = np.random.default_rng(self.random_state)
        return self

    def transform(self, X, **kwargs):
        rng = getattr(self, '_rng', np.random.default_rng(self.random_state))
        # Always per-sample: each row gets unique noise based on its own std
        stds = np.std(X, axis=1, keepdims=True)
        noise = rng.normal(0, 1, size=X.shape) * stds * self.sigma
        return X + noise
```

**Key changes:**
- No `apply_on` parameter — the operator doesn't decide HOW it's applied
- No `copy` parameter — operators always return new data, caller decides about memory
- No `augment()` method — standard sklearn `transform()` only
- RNG created in `fit()`, used in `transform()`

### 1.2 Controller Manages Application Strategy

The `SampleAugmentationController` decides how to apply transformations:

| `variation_scope` | Controller Behavior |
|-------------------|---------------------|
| `"sample"` | Each sample is transformed independently with unique random state |
| `"batch"` | All samples transformed together, same random pattern shared |
| `"group:<column>"` | Samples grouped by metadata column, each group gets unique random state |
| `"class"` | Samples grouped by target class (y), each class gets unique random state |

### 1.3 Hybrid Performance Model

Some operators can handle variation internally more efficiently than N separate instantiations.

**Protocol for performance-optimized operators:**

```python
class GaussianAdditiveNoise(TransformerMixin, BaseEstimator):
    # Class-level capability declaration
    _supports_variation_scope: ClassVar[bool] = True

    def __init__(self, sigma: float = 0.01, random_state: int = None,
                 variation_scope: str = "sample"):  # OPTIONAL param
        self.sigma = sigma
        self.random_state = random_state
        self.variation_scope = variation_scope

    def transform(self, X, **kwargs):
        rng = getattr(self, '_rng', np.random.default_rng(self.random_state))

        if self.variation_scope == "batch":
            # Same noise pattern for all samples
            scale = np.std(X) * self.sigma
            noise_row = rng.normal(0, scale, size=(1, X.shape[1]))
            noise = np.broadcast_to(noise_row, X.shape)
        else:  # "sample" (default)
            # Per-sample noise
            stds = np.std(X, axis=1, keepdims=True)
            noise = rng.normal(0, 1, size=X.shape) * stds * self.sigma

        return X + noise
```

**Controller logic:**

```python
def _apply_transformer(self, transformer, X, target_samples, variation_scope):
    supports_internal = getattr(transformer, '_supports_variation_scope', False)

    if supports_internal:
        # Performance path: operator handles variation internally
        cloned = clone(transformer)
        cloned.set_params(variation_scope=variation_scope)
        cloned.fit(X_train)
        return cloned.transform(X)
    else:
        # Fallback path: controller handles per-sample instantiation
        if variation_scope == "sample":
            results = []
            for i, sample in enumerate(X):
                cloned = clone(transformer)
                cloned.set_params(random_state=base_seed + i)  # Unique seed
                cloned.fit(X_train)
                results.append(cloned.transform(sample.reshape(1, -1)))
            return np.vstack(results)
        else:  # "batch"
            cloned = clone(transformer)
            cloned.fit(X_train)
            return cloned.transform(X)
```

### 1.4 Full sklearn Compatibility

All operators must pass sklearn's `check_estimator()`:

```python
# This must work for ALL operators
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import clone

op = GaussianAdditiveNoise(sigma=0.01, random_state=42)
check_estimator(op)  # Passes all sklearn checks

# Cloning works
op2 = clone(op)
assert op2.get_params() == op.get_params()

# get_params/set_params work
params = op.get_params()
op.set_params(sigma=0.02)
```

Wavelengths are passed via `**kwargs`, which sklearn ignores for standard transformers:

```python
# Wavelength-aware operator
op.fit(X, wavelengths=wl)  # sklearn ignores extra kwargs
op.transform(X, wavelengths=wl)

# Standard sklearn transformer (e.g., StandardScaler)
scaler.fit(X, wavelengths=wl)  # wavelengths ignored
scaler.transform(X, wavelengths=wl)  # wavelengths ignored
```

---

## 2. Naming Decision: `apply_on` → `variation_scope`

### 2.1 Problem with `apply_on`

The current name `apply_on` is ambiguous:
- "Apply on samples" could mean "apply to sample axis" (matrix operation)
- "Apply on features" could mean feature-wise operation
- Doesn't clearly convey the randomization/variation semantics

### 2.2 Alternatives Considered

| Name | Pros | Cons |
|------|------|------|
| `apply_on` | Current, users familiar | Ambiguous, doesn't convey variation |
| `variation_scope` | Clear: scope of random variation | New term to learn |
| `random_scope` | Clear for stochastic ops | Misleading for deterministic transforms |
| `sample_grouping` | Clear grouping semantics | Doesn't convey transformation aspect |
| `transform_scope` | Generic | Too similar to "transform" method |
| `scope` | Short | Too generic without context |

### 2.3 Decision: `variation_scope`

**Chosen name: `variation_scope`**

Values:
- `"sample"` — Each sample gets unique variation (default for augmentation)
- `"batch"` — All samples share the same variation parameters
- `"group:<column>"` — Variation shared within metadata groups (e.g., `"group:batch_id"`)
- `"class"` — Variation shared within target class groups

**Rationale:**
1. "Variation" captures both stochastic (random noise) and deterministic (baseline shift with fixed params) cases
2. "Scope" is a familiar concept (variable scope, block scope) indicating boundaries
3. Works at both controller level and operator level (for hybrid model)

### 2.4 Migration Path

```yaml
# BEFORE (current syntax with apply_on)
sample_augmentation:
  transformers: [GaussianAdditiveNoise(apply_on="samples", sigma=0.01)]
  count: 5

# AFTER - Option 1: Step-level variation_scope (applies to all transformers)
sample_augmentation:
  variation_scope: "sample"
  transformers: [GaussianAdditiveNoise(sigma=0.01)]
  count: 5

# AFTER - Option 2: Per-transformer variation_scope
sample_augmentation:
  count: 5
  transformers:
    - GaussianAdditiveNoise(sigma=0.01)  # Uses default "sample"
    - transformer: LinearBaselineDrift()
      variation_scope: "batch"  # All samples get same drift

# AFTER - Option 3: Mixed (step default + per-transformer override)
sample_augmentation:
  variation_scope: "sample"  # Default for all
  count: 5
  transformers:
    - GaussianAdditiveNoise(sigma=0.01)  # Uses step default "sample"
    - transformer: LinearBaselineDrift()
      variation_scope: "batch"  # Override for this one
```

---

## 3. Phase 1: SpectraTransformerMixin Simplification

**Goal:** Simplify wavelength handling to be fully sklearn-compatible via `**kwargs`.

**Duration:** 2-3 days
**Dependencies:** None (can start immediately)

### 3.1 Current State

```python
# operators/base/spectra_mixin.py
class SpectraTransformerMixin(TransformerMixin, BaseEstimator):
    _requires_wavelengths: bool = True

    def transform(self, X, wavelengths=None):
        # Validation and delegation
        if wavelengths is None and self._requires_wavelengths:
            raise ValueError("requires wavelengths")
        return self.transform_with_wavelengths(X, wavelengths)

    @abc.abstractmethod
    def transform_with_wavelengths(self, X, wavelengths):
        """Subclasses implement this."""
        pass
```

### 3.2 Target State

```python
# operators/base/spectra_mixin.py
class SpectraTransformerMixin(TransformerMixin, BaseEstimator):
    """Mixin for operators that can use wavelength information.

    Wavelengths are passed via kwargs for full sklearn compatibility:
        op.fit(X, wavelengths=wl)
        op.transform(X, wavelengths=wl)

    Set `_requires_wavelengths` to control behavior:
        - True: wavelengths required, raise if not provided
        - False: wavelengths ignored
        - "optional": use if provided, fallback to indices otherwise
    """
    _requires_wavelengths: Union[bool, str] = True

    def fit(self, X, y=None, **kwargs):
        """Fit the transformer. Override in subclass if needed."""
        wavelengths = kwargs.get('wavelengths')
        self._validate_wavelengths(wavelengths, X.shape[1])
        self._wavelengths = wavelengths
        return self

    def transform(self, X, **kwargs):
        """Transform X. Subclasses must implement _transform_impl."""
        wavelengths = kwargs.get('wavelengths', getattr(self, '_wavelengths', None))
        self._validate_wavelengths(wavelengths, X.shape[1])
        return self._transform_impl(X, wavelengths)

    def _validate_wavelengths(self, wavelengths, n_features):
        if self._requires_wavelengths is True and wavelengths is None:
            raise ValueError(f"{self.__class__.__name__} requires wavelengths")
        if self._requires_wavelengths == "optional" and wavelengths is None:
            # Will use np.arange(n_features) as fallback in _transform_impl
            pass
        if wavelengths is not None and len(wavelengths) != n_features:
            raise ValueError(f"wavelengths length {len(wavelengths)} != features {n_features}")

    @abc.abstractmethod
    def _transform_impl(self, X, wavelengths):
        """Implement transformation logic. wavelengths may be None if optional."""
        pass
```

### 3.3 Changes Required

| # | File | Change | LOC |
|---|------|--------|-----|
| 1.1 | `operators/base/spectra_mixin.py` | Rewrite as shown above. Remove `transform_with_wavelengths`. Add `_transform_impl` abstract method. | ~50 |
| 1.2 | `operators/augmentation/environmental.py` | `TemperatureAugmenter`, `MoistureAugmenter`: rename `transform_with_wavelengths` → `_transform_impl` | ~10 |
| 1.3 | `operators/augmentation/scattering.py` | `ParticleSizeAugmenter`, `EMSCDistortionAugmenter`: same | ~10 |
| 1.4 | `operators/augmentation/edge_artifacts.py` | 5 augmenters: same | ~25 |
| 1.5 | `controllers/transforms/transformer.py` | Update `_needs_wavelengths()` → `_get_wavelength_requirement()`. Pass wavelengths via kwargs instead of positional. | ~30 |
| 1.6 | Tests | Update all test calls from `transform_with_wavelengths(X, wl)` → `transform(X, wavelengths=wl)` | ~40 |

### 3.4 Testing Strategy

**Before changes:**
```bash
pytest tests/unit/operators/base/test_spectra_mixin.py -v
pytest tests/unit/operators/augmentation/ -v
```

**After changes:**
1. All existing tests pass (with updated method names)
2. New test: sklearn compatibility check
   ```python
   def test_sklearn_compatibility():
       from sklearn.utils.estimator_checks import parametrize_with_checks
       # Note: full check_estimator may need X,y generation customization
       op = TemperatureAugmenter(...)
       clone(op)  # Must work
       op.get_params()  # Must work
       op.set_params(...)  # Must work
   ```
3. New test: wavelengths via kwargs
   ```python
   def test_wavelengths_via_kwargs():
       op = TemperatureAugmenter(...)
       op.fit(X, wavelengths=wl)
       result = op.transform(X, wavelengths=wl)
       # Also test without wavelengths in transform (uses cached)
       result2 = op.transform(X)
       np.testing.assert_array_equal(result, result2)
   ```

### 3.5 Commit Boundary

Single atomic commit: all 9 operators + mixin + controller + tests.

**Rationale:** Changing the abstract method signature breaks all subclasses. Must change together.

### 3.6 Rollback Plan

If issues discovered post-merge:
1. Revert the atomic commit
2. No partial state possible (all-or-nothing change)

---

## 4. Phase 2: Augmenter Base Class Elimination

**Goal:** Convert all `Augmenter` subclasses to plain `TransformerMixin`/`SpectraTransformerMixin`. Remove `apply_on`, `copy`, and `augment()`.

**Duration:** 3-4 days
**Dependencies:** Phase 1 complete

### 4.1 Current Augmenter Inventory

**26 operators inheriting from `Augmenter`:**

| Category | Operators | Count |
|----------|-----------|-------|
| Noise | `GaussianAdditiveNoise`, `MultiplicativeNoise` | 2 |
| Baseline | `LinearBaselineDrift`, `PolynomialBaselineDrift` | 2 |
| Wavelength | `WavelengthShift`, `WavelengthStretch`, `LocalWavelengthWarp`, `SmoothMagnitudeWarp` | 4 |
| Features | `BandPerturbation`, `GaussianSmoothingJitter`, `UnsharpSpectralMask` | 3 |
| Masking | `BandMasking`, `ChannelDropout` | 2 |
| Artifacts | `SpikeNoise`, `LocalClipping` | 2 |
| Combinations | `MixupAugmenter`, `LocalMixupAugmenter`, `ScatterSimulationMSC` | 3 |
| Splines | `Spline_Smoothing`, `Spline_X_Perturbations`, `Spline_Y_Perturbations`, `Spline_X_Simplification`, `Spline_Curve_Simplification` | 5 |
| Random | `Rotate_Translate`, `Random_X_Operation` | 2 |
| Identity | `IdentityAugmenter` | 1 |

### 4.2 Migration Categories

**Category A: Need wavelengths (become SpectraTransformerMixin)**

| Operator | Has `lambda_axis` | Set `_requires_wavelengths` |
|----------|-------------------|---------------------------|
| `LinearBaselineDrift` | Yes | `"optional"` |
| `PolynomialBaselineDrift` | Yes | `"optional"` |
| `WavelengthShift` | Yes | `"optional"` |
| `WavelengthStretch` | Yes | `"optional"` |
| `LocalWavelengthWarp` | Yes | `"optional"` |
| `SmoothMagnitudeWarp` | Yes | `"optional"` |
| `ScatterSimulationMSC` | No (future tilt) | `"optional"` |

**Category B: No wavelengths needed, support `variation_scope` internally (stay optimized)**

| Operator | Benefit of internal handling |
|----------|----------------------------|
| `GaussianAdditiveNoise` | Batch noise generation is efficient |
| `MultiplicativeNoise` | Batch gain generation is efficient |
| `BandPerturbation` | Can generate band perturbations in batch |
| `SpikeNoise` | Batch spike placement is efficient |

**Category C: No wavelengths, no internal variation handling (simple TransformerMixin)**

All remaining operators (splines, masking, combinations, etc.)

### 4.3 Migration Pattern for Each Category

**Category A — SpectraTransformerMixin:**

```python
# BEFORE
class LinearBaselineDrift(Augmenter):
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 offset_range=(-0.1, 0.1), slope_range=(-0.001, 0.001),
                 lambda_axis=None):
        super().__init__(apply_on, random_state, copy=copy)
        self.offset_range = offset_range
        self.slope_range = slope_range
        self.lambda_axis = lambda_axis

    def augment(self, X, apply_on="samples"):
        lambdas = self.lambda_axis if self.lambda_axis is not None else np.arange(X.shape[1])
        # ... implementation

# AFTER
class LinearBaselineDrift(SpectraTransformerMixin):
    _requires_wavelengths = "optional"

    def __init__(self, offset_range=(-0.1, 0.1), slope_range=(-0.001, 0.001),
                 random_state=None):
        self.offset_range = offset_range
        self.slope_range = slope_range
        self.random_state = random_state

    def fit(self, X, y=None, **kwargs):
        self._rng = np.random.default_rng(self.random_state)
        super().fit(X, y, **kwargs)  # Handles wavelength caching
        return self

    def _transform_impl(self, X, wavelengths):
        rng = getattr(self, '_rng', np.random.default_rng(self.random_state))
        lambdas = wavelengths if wavelengths is not None else np.arange(X.shape[1])
        # ... implementation (same math, uses lambdas)
```

**Category B — TransformerMixin with optional `variation_scope`:**

```python
# AFTER
class GaussianAdditiveNoise(TransformerMixin, BaseEstimator):
    _supports_variation_scope = True  # Controller checks this

    def __init__(self, sigma: float = 0.01, random_state: int = None,
                 smoothing_kernel_width: int = 1,
                 variation_scope: str = "sample"):  # Optional performance param
        self.sigma = sigma
        self.random_state = random_state
        self.smoothing_kernel_width = smoothing_kernel_width
        self.variation_scope = variation_scope

    def fit(self, X, y=None, **kwargs):
        self._rng = np.random.default_rng(self.random_state)
        return self

    def transform(self, X, **kwargs):
        rng = getattr(self, '_rng', np.random.default_rng(self.random_state))

        if self.variation_scope == "batch":
            # Same noise for all samples
            scale = np.std(X) * self.sigma
            noise_pattern = rng.normal(0, scale, size=(1, X.shape[1]))
            noise = np.tile(noise_pattern, (X.shape[0], 1))
        else:  # "sample" (default)
            # Per-sample noise
            stds = np.std(X, axis=1, keepdims=True)
            noise = rng.normal(0, 1, size=X.shape) * stds * self.sigma

        if self.smoothing_kernel_width > 1:
            noise = _smooth_batch(noise, self.smoothing_kernel_width)

        return X + noise
```

**Category C — Plain TransformerMixin:**

```python
# AFTER
class ChannelDropout(TransformerMixin, BaseEstimator):
    def __init__(self, dropout_rate: float = 0.1, random_state: int = None):
        self.dropout_rate = dropout_rate
        self.random_state = random_state

    def fit(self, X, y=None, **kwargs):
        self._rng = np.random.default_rng(self.random_state)
        return self

    def transform(self, X, **kwargs):
        rng = getattr(self, '_rng', np.random.default_rng(self.random_state))
        mask = rng.random(X.shape) > self.dropout_rate
        return X * mask
```

### 4.4 Changes Required

| # | File | Change | LOC |
|---|------|--------|-----|
| 2.1 | `operators/augmentation/spectral.py` | Migrate all spectral augmenters (noise, baseline, wavelength, features, masking, artifacts, combinations) | ~400 |
| 2.2 | `operators/augmentation/splines.py` | Migrate 5 spline augmenters | ~150 |
| 2.3 | `operators/augmentation/random.py` | Migrate 2 random augmenters | ~80 |
| 2.4 | `operators/augmentation/abc_augmenter.py` | Delete `Augmenter` and `IdentityAugmenter` classes | -120 |
| 2.5 | `operators/augmentation/__init__.py` | Remove `Augmenter`, `IdentityAugmenter` exports | ~5 |
| 2.6 | Tests | Update all test calls, add `variation_scope` tests | ~100 |

### 4.5 Breaking Changes

| Change | Impact | Migration |
|--------|--------|-----------|
| `apply_on` parameter removed | User configs with `apply_on` break | Config migration: move `apply_on` to step-level `variation_scope` |
| `copy` parameter removed | Minor, rarely used explicitly | None needed, just remove |
| `augment()` method removed | Any code calling `.augment()` breaks | Change to `.transform()` |
| `lambda_axis` parameter removed | User configs break | Config migration: remove (controller provides wavelengths) |
| `IdentityAugmenter` deleted | Configs using it break | Remove from configs (no-op anyway) |

### 4.6 Testing Strategy

**Per-operator tests:**
```python
def test_operator_sklearn_compatible():
    op = GaussianAdditiveNoise(sigma=0.01, random_state=42)
    # Clone works
    op2 = clone(op)
    # get_params/set_params work
    assert op.get_params()['sigma'] == 0.01
    op.set_params(sigma=0.02)
    assert op.get_params()['sigma'] == 0.02

def test_operator_transform_shape():
    op = GaussianAdditiveNoise(sigma=0.01, random_state=42)
    X = np.random.randn(100, 50)
    op.fit(X)
    X_out = op.transform(X)
    assert X_out.shape == X.shape

def test_operator_deterministic():
    X = np.random.randn(100, 50)
    op1 = GaussianAdditiveNoise(sigma=0.01, random_state=42)
    op2 = GaussianAdditiveNoise(sigma=0.01, random_state=42)
    op1.fit(X)
    op2.fit(X)
    np.testing.assert_array_equal(op1.transform(X), op2.transform(X))

def test_variation_scope_sample():
    X = np.random.randn(10, 50)
    op = GaussianAdditiveNoise(sigma=0.01, random_state=42, variation_scope="sample")
    op.fit(X)
    X_out = op.transform(X)
    # Each row should have different noise
    row_stds = np.std(X_out - X, axis=1)
    assert len(np.unique(np.round(row_stds, 6))) > 1

def test_variation_scope_batch():
    X = np.random.randn(10, 50)
    op = GaussianAdditiveNoise(sigma=0.01, random_state=42, variation_scope="batch")
    op.fit(X)
    X_out = op.transform(X)
    # All rows should have same noise pattern
    noise = X_out - X
    for i in range(1, 10):
        np.testing.assert_array_almost_equal(noise[0], noise[i])
```

### 4.7 Commit Strategy

Split into 3 sub-commits for reviewability:

1. **Commit 2a:** `spectral.py` operators (largest file)
2. **Commit 2b:** `splines.py` + `random.py` operators
3. **Commit 2c:** Delete `abc_augmenter.py`, update exports, cleanup

---

## 5. Phase 3: SampleAugmentationController Variation Management

**Goal:** Move variation logic from operators to controller. Implement hybrid performance model.

**Duration:** 4-5 days
**Dependencies:** Phase 2 complete

### 5.1 Current Controller Flow

```
SampleAugmentationController.execute()
    ├── _execute_standard() or _execute_balanced()
    │   ├── Calculate augmentation counts per sample
    │   ├── Build transformer→samples mapping
    │   └── _emit_augmentation_steps()
    │       ├── _emit_augmentation_steps_parallel() [if joblib available]
    │       │   └── ThreadPoolExecutor: clone, fit, transform batch
    │       └── _emit_augmentation_steps_sequential()
    │           └── Delegate to TransformerMixinController
```

### 5.2 Target Controller Flow

```
SampleAugmentationController.execute()
    ├── Parse `variation_scope` from step config (default: "sample")
    ├── For each transformer:
    │   ├── Check if transformer._supports_variation_scope
    │   │   ├── If True: Performance path
    │   │   │   └── Set variation_scope on operator, call once with batch
    │   │   └── If False: Fallback path
    │   │       ├── If scope=="sample": Create N clones, one per sample
    │   │       ├── If scope=="batch": Create 1 clone, call with batch
    │   │       └── If scope=="group:col": Create 1 clone per group
    │   └── Add augmented samples to dataset
```

### 5.3 Configuration Schema

```yaml
sample_augmentation:
  # Step-level settings
  count: 5                      # Number of augmented copies per sample
  variation_scope: "sample"     # Default for all transformers
  random_state: 42              # Reproducibility
  selection: "random"           # How to select transformers: "random" | "all" | "sequential"

  # Transformers - multiple syntaxes supported
  transformers:
    # Syntax 1: Direct operator (uses step-level variation_scope)
    - GaussianAdditiveNoise(sigma=0.01)

    # Syntax 2: Dict with per-transformer override
    - transformer: LinearBaselineDrift()
      variation_scope: "batch"  # Override for this transformer only

    # Syntax 3: Dict for sklearn operators (no internal variation support)
    - transformer: StandardScaler()
      variation_scope: "batch"  # Must be batch (no random component)
```

### 5.4 Implementation Details

**5.4.1 Parsing `variation_scope`:**

```python
def _parse_variation_scope(self, config, transformer_spec):
    """Get variation_scope for a transformer, with inheritance."""
    step_scope = config.get("variation_scope", "sample")

    if isinstance(transformer_spec, dict):
        return transformer_spec.get("variation_scope", step_scope)
    else:
        return step_scope
```

**5.4.2 Detecting operator capability:**

```python
def _supports_internal_variation(self, transformer):
    """Check if transformer can handle variation_scope internally."""
    return getattr(transformer, '_supports_variation_scope', False)
```

**5.4.3 Performance path (internal handling):**

```python
def _apply_with_internal_variation(self, transformer, X_train, X_augment,
                                    wavelengths, variation_scope):
    """Apply transformer that handles variation internally."""
    cloned = clone(transformer)
    cloned.set_params(variation_scope=variation_scope)

    if wavelengths is not None:
        cloned.fit(X_train, wavelengths=wavelengths)
        return cloned.transform(X_augment, wavelengths=wavelengths)
    else:
        cloned.fit(X_train)
        return cloned.transform(X_augment)
```

**5.4.4 Fallback path (controller handles variation):**

```python
def _apply_with_controller_variation(self, transformer, X_train, X_augment,
                                      wavelengths, variation_scope, base_seed):
    """Apply transformer with controller-managed variation."""

    if variation_scope == "batch":
        # One transformation for all samples
        cloned = clone(transformer)
        if base_seed is not None:
            cloned.set_params(random_state=base_seed)
        if wavelengths is not None:
            cloned.fit(X_train, wavelengths=wavelengths)
            return cloned.transform(X_augment, wavelengths=wavelengths)
        else:
            cloned.fit(X_train)
            return cloned.transform(X_augment)

    elif variation_scope == "sample":
        # One transformation per sample (N clones)
        results = []
        for i in range(X_augment.shape[0]):
            cloned = clone(transformer)
            if base_seed is not None:
                cloned.set_params(random_state=base_seed + i)
            sample = X_augment[i:i+1]  # Keep 2D shape
            if wavelengths is not None:
                cloned.fit(X_train, wavelengths=wavelengths)
                results.append(cloned.transform(sample, wavelengths=wavelengths))
            else:
                cloned.fit(X_train)
                results.append(cloned.transform(sample))
        return np.vstack(results)

    elif variation_scope.startswith("group:"):
        # One transformation per group
        column = variation_scope.split(":", 1)[1]
        groups = self._get_sample_groups(column)
        results = np.empty_like(X_augment)
        for group_id, indices in groups.items():
            cloned = clone(transformer)
            if base_seed is not None:
                cloned.set_params(random_state=base_seed + hash(group_id) % (2**31))
            group_X = X_augment[indices]
            if wavelengths is not None:
                cloned.fit(X_train, wavelengths=wavelengths)
                results[indices] = cloned.transform(group_X, wavelengths=wavelengths)
            else:
                cloned.fit(X_train)
                results[indices] = cloned.transform(group_X)
        return results
```

### 5.5 Changes Required

| # | File | Change | LOC |
|---|------|--------|-----|
| 3.1 | `controllers/data/sample_augmentation.py` | Add `variation_scope` parsing, capability detection, apply methods | ~200 |
| 3.2 | `controllers/data/sample_augmentation.py` | Refactor `_emit_augmentation_steps_parallel` to use new apply methods | ~100 |
| 3.3 | `controllers/data/sample_augmentation.py` | Refactor `_emit_augmentation_steps_sequential` to use new apply methods | ~80 |
| 3.4 | `pipeline/config/schema.py` (if exists) | Add `variation_scope` to step schema | ~20 |
| 3.5 | Tests | Comprehensive tests for all variation modes | ~200 |

### 5.6 Testing Strategy

**Test matrix:**

| Operator Type | `variation_scope` | Expected Behavior |
|---------------|-------------------|-------------------|
| Internal support | `"sample"` | Operator handles, batch call |
| Internal support | `"batch"` | Operator handles, batch call |
| No internal support, stochastic | `"sample"` | N clones, per-sample call |
| No internal support, stochastic | `"batch"` | 1 clone, batch call |
| No internal support, deterministic | `"sample"` | 1 clone, batch call (no variation anyway) |
| sklearn StandardScaler | `"batch"` | 1 clone, batch call |

**Integration test:**

```python
def test_sample_augmentation_variation_scope():
    dataset = create_test_dataset(100)  # 100 samples

    config = {
        "sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01)],
            "count": 3,
            "variation_scope": "sample",
            "random_state": 42
        }
    }

    controller = SampleAugmentationController()
    controller.execute(config, dataset, ...)

    # Should have 100 * 3 = 300 augmented samples
    assert dataset.n_samples() == 100 + 300

    # Each augmented sample should have unique noise
    augmented = dataset.x({"augmented": True})
    # ... verify variation
```

### 5.7 Commit Strategy

Single commit: controller refactoring is a cohesive change.

---

## 6. Phase 4: Parallel Augmentation & Wavelength Fixes

**Goal:** Fix bugs in parallel path, ensure wavelength-aware operators work in all paths.

**Duration:** 2 days
**Dependencies:** Phase 3 complete

### 6.1 Current Bugs

1. **Wavelength not passed in parallel path:** `_emit_augmentation_steps_parallel()` calls `fit()` and `transform()` without wavelengths.

2. **No variation_scope support in parallel path:** Current parallel path treats all operators the same.

### 6.2 Fix Implementation

```python
def _emit_augmentation_steps_parallel(self, active_transformers, transformers,
                                       context, dataset, runtime_context, loaded_binaries):
    """Parallel execution with variation_scope and wavelength support."""

    # Cache wavelengths per source
    wavelengths_cache = {}  # source_idx -> wavelengths or _MISSING sentinel

    def get_wavelengths(source_idx, operator):
        if source_idx in wavelengths_cache:
            wl = wavelengths_cache[source_idx]
            return None if wl is _MISSING else wl

        req = self._get_wavelength_requirement(operator)
        if req == "none":
            wavelengths_cache[source_idx] = _MISSING
            return None

        try:
            wl = dataset.wavelengths_nm(source_idx)
            wavelengths_cache[source_idx] = wl
            return wl
        except ValueError:
            if req == "required":
                raise
            wavelengths_cache[source_idx] = _MISSING
            return None

    def process_transformer(args):
        trans_idx, sample_ids, variation_scope = args
        transformer = transformers[trans_idx]

        # Determine application strategy
        if self._supports_internal_variation(transformer):
            return self._process_internal_variation(
                transformer, sample_ids, variation_scope,
                get_wavelengths, train_data, all_origin_data
            )
        else:
            return self._process_controller_variation(
                transformer, sample_ids, variation_scope,
                get_wavelengths, train_data, all_origin_data, base_seed
            )

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=min(len(active_transformers), 16)) as executor:
        futures = {
            executor.submit(process_transformer, (idx, samples, scope)): (idx, samples)
            for idx, samples, scope in active_transformers
        }
        # ... collect results
```

### 6.3 Changes Required

| # | File | Change | LOC |
|---|------|--------|-----|
| 4.1 | `controllers/data/sample_augmentation.py` | Add wavelength caching with `_get_wavelength_requirement` | ~40 |
| 4.2 | `controllers/data/sample_augmentation.py` | Refactor `_emit_augmentation_steps_parallel` for variation_scope | ~100 |
| 4.3 | `controllers/transforms/transformer.py` | Expose `_get_wavelength_requirement` as module-level helper | ~10 |
| 4.4 | Tests | Test parallel path with wavelength-aware operators | ~60 |

### 6.4 Commit Strategy

Single commit: bug fix + parallel path enhancement.

---

## 7. Phase 5: Synthesis Transfer & Externalization

**Goal:** Create new augmenters from synthesis mechanisms, refactor generator, externalize synthesis module.

**Duration:** 5-7 days
**Dependencies:** Phase 4 complete

*(This phase content is largely the same as the original roadmap Stages 2-3, with adjustments for the new operator patterns)*

### 7.1 New Augmentation Operators

| Operator | Base Class | `_supports_variation_scope` |
|----------|------------|---------------------------|
| `PathLengthAugmenter` | `TransformerMixin` | Yes |
| `BatchEffectAugmenter` | `SpectraTransformerMixin` | Yes |
| `InstrumentalBroadeningAugmenter` | `SpectraTransformerMixin` | Yes |
| `HeteroscedasticNoiseAugmenter` | `TransformerMixin` | Yes |
| `DeadBandAugmenter` | `TransformerMixin` | Yes |

Each follows the new pattern (no `apply_on`, optional `variation_scope` for internal handling).

### 7.2 Generator Refactoring

The generator delegates to augmenters:

```python
class SyntheticNIRSGenerator:
    def _build_effect_chain(self):
        chain = []
        if self.config.path_length_variation > 0:
            chain.append(PathLengthAugmenter(
                path_length_std=self.config.path_length_std,
                random_state=self._derive_seed('path_length'),
                variation_scope="sample"  # Generator always uses sample-level
            ))
        # ... more operators
        return chain

    def generate(self, n_samples):
        # Phase A: Generative
        C = self.generate_concentrations(n_samples)
        A = self._apply_beer_lambert(C)

        # Phase B: Effect chain
        for op in self._effect_chain:
            op.fit(A, wavelengths=self.wavelengths)
            A = op.transform(A, wavelengths=self.wavelengths)

        return A
```

### 7.3 Synthesis Externalization

Move `nirs4all/synthesis/` → `nirs4all/synthesis/` with grouped structure.

---

## 8. Phase 6: Webapp & Documentation

**Goal:** Update webapp node definitions, documentation, examples.

**Duration:** 2-3 days
**Dependencies:** Phase 5 complete

### 8.1 Webapp Node Updates

**Removed parameters:**
- `apply_on` from all augmenter nodes
- `copy` from all augmenter nodes
- `lambda_axis` from 6 operators

**Added parameters:**
- `variation_scope` to step-level `sample_augmentation` configuration

**New nodes:**
- 5 new augmenters from Phase 5

### 8.2 Documentation Updates

| Document | Change |
|----------|--------|
| User guide | New `variation_scope` concept, migration from `apply_on` |
| API reference | Updated operator signatures |
| CHANGELOG | Breaking changes list |
| Examples | Updated configs |

---

## 9. Task Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: SpectraTransformerMixin                      │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐   │
│  │ 1.1 Mixin   │──▶│ 1.2-1.4     │──▶│ 1.5         │──▶│ 1.6 Tests       │   │
│  │ rewrite     │   │ 9 operators │   │ Controller  │   │ update          │   │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────────┘   │
│                           All changes atomic (single commit)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 2: Augmenter Elimination                        │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                         │
│  │ 2.1         │   │ 2.2         │   │ 2.3         │   Can be parallelized   │
│  │ spectral.py │   │ splines.py  │   │ random.py   │◀──(same pattern)        │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘                         │
│         │                 │                 │                                 │
│         └────────────────┬┴─────────────────┘                                 │
│                          ▼                                                    │
│                  ┌─────────────┐   ┌─────────────┐                            │
│                  │ 2.4 Delete  │──▶│ 2.6 Tests   │                            │
│                  │ Augmenter   │   │ update      │                            │
│                  └─────────────┘   └─────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 3: Controller variation_scope                   │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐             │
│  │ 3.1 Parse       │──▶│ 3.2-3.3 Apply   │──▶│ 3.5 Tests       │             │
│  │ variation_scope │   │ methods         │   │                 │             │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 4: Parallel & Wavelength Fixes                  │
│  ┌─────────────────────────────────────────┐                                 │
│  │ 4.1-4.4 Fix parallel path, add         │                                 │
│  │ wavelength caching, tests              │                                 │
│  └─────────────────────────────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 5: Synthesis Transfer                           │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐                     │
│  │ 5.1 New     │   │ 5.2 Gen     │   │ 5.3 External-   │   Can be            │
│  │ augmenters  │──▶│ refactor    │──▶│ ization         │   parallelized:     │
│  └─────────────┘   └─────────────┘   └─────────────────┘   5.1.(a-e)         │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 6: Webapp & Docs                                │
│  ┌─────────────┐   ┌─────────────┐                                           │
│  │ 6.1 Webapp  │   │ 6.2 Docs    │   Can be parallelized                     │
│  │ nodes       │   │ update      │                                           │
│  └─────────────┘   └─────────────┘                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.1 Parallelization Opportunities

| Tasks | Can Parallelize? | Notes |
|-------|-----------------|-------|
| Phase 1 tasks | No | Atomic change, all-or-nothing |
| Phase 2.1, 2.2, 2.3 | Yes | Same pattern, different files |
| Phase 3 tasks | No | Sequential within controller |
| Phase 4 tasks | No | Bug fix, cohesive change |
| Phase 5.1.a-e (new ops) | Yes | Independent operators |
| Phase 6.1, 6.2 | Yes | Webapp and docs independent |

### 9.2 Critical Path

```
Phase 1 (3d) → Phase 2 (4d) → Phase 3 (5d) → Phase 4 (2d) → Phase 5 (6d) → Phase 6 (3d)

Total: ~23 working days (with parallelization: ~18 days)
```

### 9.3 Resource Allocation

| Phase | Assignee | Duration | Parallelizable With |
|-------|----------|----------|---------------------|
| 1 | Dev A | 3 days | - |
| 2.1 | Dev A | 2 days | 2.2, 2.3 |
| 2.2 | Dev B | 1 day | 2.1, 2.3 |
| 2.3 | Dev B | 0.5 day | 2.1, 2.2 |
| 2.4-2.6 | Dev A+B | 1 day | - |
| 3 | Dev A | 5 days | - |
| 4 | Dev A | 2 days | - |
| 5.1 | Dev B | 3 days | (5 operators parallel) |
| 5.2 | Dev A | 2 days | - |
| 5.3 | Dev A | 1 day | - |
| 6.1 | Dev B | 1.5 days | 6.2 |
| 6.2 | Dev A | 1.5 days | 6.1 |

---

## 10. Verification Checklist

### 10.1 Phase 1 Verification

- [ ] `transform_with_wavelengths` method removed from all operators
- [ ] All operators use `transform(X, **kwargs)` signature
- [ ] `_transform_impl(X, wavelengths)` implemented in all SpectraTransformerMixin subclasses
- [ ] Controller passes wavelengths via kwargs
- [ ] All tests pass: `pytest tests/unit/operators/ tests/unit/controllers/`
- [ ] sklearn `clone()` works for all operators

### 10.2 Phase 2 Verification

- [ ] `Augmenter` base class deleted
- [ ] `IdentityAugmenter` deleted
- [ ] No `augment()` method in any operator
- [ ] No `apply_on` parameter in any operator constructor
- [ ] No `copy` parameter in any operator constructor
- [ ] No `lambda_axis` parameter in any operator constructor
- [ ] `_supports_variation_scope = True` on optimized operators
- [ ] All tests pass

### 10.3 Phase 3 Verification

- [ ] `variation_scope` parsed from step config
- [ ] per-transformer `variation_scope` override works
- [ ] Internal variation path works (operator handles)
- [ ] Controller variation path works (N clones for "sample")
- [ ] `"group:<column>"` variation works
- [ ] All tests pass

### 10.4 Phase 4 Verification

- [ ] Wavelengths passed in parallel path
- [ ] Wavelength caching works
- [ ] variation_scope works in parallel path
- [ ] Parallel tests pass with wavelength-aware operators

### 10.5 Phase 5 Verification

- [ ] 5 new augmenters created and tested
- [ ] Generator uses augmenters for effect chain
- [ ] Generator output statistically equivalent (golden test)
- [ ] Synthesis tests pass
- [ ] Synthesis externalized to `nirs4all/synthesis/`

### 10.6 Phase 6 Verification

- [ ] Webapp node definitions updated
- [ ] `npm run validate:nodes` passes
- [ ] Documentation updated
- [ ] Examples work with new API
- [ ] CHANGELOG documents breaking changes

---

## Appendix A: Breaking Changes Summary

| Change | Impact | Migration |
|--------|--------|-----------|
| `apply_on` removed from operators | Config syntax change | Move to step-level `variation_scope` |
| `copy` removed from operators | Minor | Remove from configs |
| `lambda_axis` removed from operators | Config syntax change | Remove from configs (auto-injected) |
| `augment()` method removed | API change | Use `transform()` |
| `transform_with_wavelengths()` removed | API change | Use `transform(X, wavelengths=wl)` |
| `IdentityAugmenter` deleted | Config change | Remove from configs |
| `Augmenter` base class deleted | Custom subclasses break | Inherit from `TransformerMixin` |
| `nirs4all.synthesis` moved | Import change | Use `nirs4all.synthesis` |

## Appendix B: Configuration Migration Examples

**Before (current):**
```yaml
sample_augmentation:
  transformers:
    - GaussianAdditiveNoise(apply_on="samples", sigma=0.01)
    - LinearBaselineDrift(apply_on="global", lambda_axis=[1000, 1001, ...])
  count: 5
```

**After (target):**
```yaml
sample_augmentation:
  variation_scope: "sample"  # Default for all, was apply_on
  transformers:
    - GaussianAdditiveNoise(sigma=0.01)  # Uses step default
    - transformer: LinearBaselineDrift()
      variation_scope: "batch"  # Override, was apply_on="global"
  count: 5
  # lambda_axis removed - controller provides wavelengths automatically
```
