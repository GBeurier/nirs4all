# Implementation Roadmap: Operator Refactoring, Augmentation Transfer & Synthesis Externalization

**Date:** February 2026
**Status:** Roadmap
**Source documents:**
- [Operator Base Class Design v2](operator_base_class_design.md)
- [Synthesis Externalization & Augmentation Transfer](synthesis_augmentation_refactoring.md)

---

## Table of Contents

1. [Dependency Chain & Execution Order](#1-dependency-chain--execution-order)
2. [Stage 1: Operator Base Class Refactoring](#2-stage-1-operator-base-class-refactoring)
3. [Stage 2: Synthesis Mechanism Transfer to Augmentation](#3-stage-2-synthesis-mechanism-transfer-to-augmentation)
4. [Stage 3: Synthesis Externalization](#4-stage-3-synthesis-externalization)
5. [Stage 4: Webapp Alignment & Documentation](#5-stage-4-webapp-alignment--documentation)
6. [Detailed Review: Maintainability & Performance](#6-detailed-review-maintainability--performance)

---

## 1. Dependency Chain & Execution Order

The two design documents describe three logical work streams with strict ordering dependencies:

```
Stage 1: Operator Base Class Refactoring (operator_base_class_design.md)
   |
   |  Eliminates Augmenter base class, simplifies SpectraTransformerMixin,
   |  fixes parallel augmentation bug, protocol-based wavelength detection
   |
   v
Stage 2: Synthesis Mechanism Transfer (synthesis_augmentation_refactoring.md §4.2)
   |
   |  New augmenters (PathLength, BatchEffect, InstrumentalBroadening,
   |  HeteroscedasticNoise, DeadBand), enhance existing augmenters,
   |  refactor generator to delegate all effects to augmenters
   |
   v
Stage 3: Synthesis Externalization (synthesis_augmentation_refactoring.md §4.1)
   |
   |  Move synthesis/ → synthesis/, update all imports,
   |  hard break (no backward compat alias)
   |
   v
Stage 4: Webapp Alignment & Documentation
      Update node definitions, examples, user docs
```

**Why this order:**
- Stage 1 must come first because Stage 2 creates new augmenters — those should be built on the clean hierarchy, not on the old `Augmenter` base class.
- Stage 2 must precede Stage 3 because the generator refactoring (delegating to augmenters) is a behavioral change that needs testing before the mechanical move.
- Stage 3 is purely mechanical (rename imports) and should happen on stable code.
- Stage 4 is follow-up alignment work.

---

## 2. Stage 1: Operator Base Class Refactoring

**Goal:** Eliminate the `Augmenter` base class and `transform_with_wavelengths` indirection. Establish a single, clean operator interface.

**Outcome:** All operators use `transform(X)` or `transform(X, wavelengths=None)`. One protocol flag (`_requires_wavelengths`) for wavelength injection. Parallel augmentation bug fixed.

### Phase 1.1: SpectraTransformerMixin + Controller + 9 Wavelength-Aware Operators

**Rationale:** These changes are atomic — changing `SpectraTransformerMixin.transform()` to raise `NotImplementedError` immediately breaks the 9 subclasses that override `transform_with_wavelengths()`. All must change together.

**Changes:**

| # | File | Change |
|---|------|--------|
| 1 | `operators/base/spectra_mixin.py` | Remove `transform_with_wavelengths()` abstract method. Change `transform()` to `raise NotImplementedError`. Add `_requires_wavelengths: Union[bool, str] = True` type annotation. Support `"optional"` value. |
| 2 | `operators/augmentation/environmental.py` | `TemperatureAugmenter`: rename `transform_with_wavelengths(self, X, wavelengths)` → `transform(self, X, wavelengths=None)`. Same for `MoistureAugmenter`. |
| 3 | `operators/augmentation/scattering.py` | `ParticleSizeAugmenter`, `EMSCDistortionAugmenter`: same rename. |
| 4 | `operators/augmentation/edge_artifacts.py` | `DetectorRollOffAugmenter`, `StrayLightAugmenter`, `EdgeCurvatureAugmenter`, `TruncatedPeakAugmenter`, `EdgeArtifactsAugmenter`: same rename. |
| 5 | `controllers/transforms/transformer.py` | Replace `_needs_wavelengths()` with `_wavelength_requirement()` returning `"required"` / `"optional"` / `"none"`. Add `_get_wavelengths(dataset, source_index, operator_name, requirement)` helper. Update all call sites (fit/transform branching). |
| 6 | `controllers/data/sample_augmentation.py` | Fix `_emit_augmentation_steps_parallel()`: (a) Add wavelength resolution with `_MISSING` sentinel cache using new `_wavelength_requirement()` helper. (b) Pass wavelengths to `fit()` and `transform()` for wavelength-aware operators. (c) No changes needed for `apply_on` \u2014 controllers never reference it (grep-verified: zero occurrences in controllers/). |
| 7 | Tests: `test_spectra_mixin.py`, `test_environmental.py`, `test_scattering.py`, `test_edge_artifacts.py`, `test_transformer_wavelengths.py`, `test_spectra_transformer_pipeline.py` | Replace `transform_with_wavelengths(X, wl)` calls with `transform(X, wavelengths=wl)`. Update assertions. |

**Commit boundary:** Single commit. All 9 operators + mixin + controller + tests change atomically.

**Testing strategy for this phase:**
1. Before migration: verify existing tests in `tests/unit/operators/augmentation/` and `tests/unit/operators/base/` pass
2. After migration: replace `transform_with_wavelengths(X, wl)` calls with `transform(X, wavelengths=wl)` in tests
3. Parallel sample augmentation with wavelength-aware operator: now works (was buggy before)

**Verification:**
- `pytest tests/unit/operators/augmentation/` — all augmenter unit tests pass
- `pytest tests/unit/operators/base/` — mixin tests pass
- `pytest tests/unit/controllers/` — controller tests pass
- `pytest tests/integration/augmentation/` — integration tests pass

### Phase 1.2: Migrate 7 `lambda_axis` / Wavelength-Dependent Operators to SpectraTransformerMixin

**Rationale:** These operators need wavelengths for correct math but either use the ad-hoc `lambda_axis` constructor parameter or need wavelengths for new functionality. After migration, the controller automatically provides wavelengths.

**Changes per operator:**

| Operator | File | Base class change | Parameters removed | New flag |
|----------|------|-------------------|--------------------|----------|
| `LinearBaselineDrift` | `spectral.py` | `Augmenter` → `SpectraTransformerMixin` | `lambda_axis`, `apply_on`, `copy` | `_requires_wavelengths = "optional"` |
| `PolynomialBaselineDrift` | `spectral.py` | Same | Same | Same |
| `WavelengthShift` | `spectral.py` | Same | Same | Same |
| `WavelengthStretch` | `spectral.py` | Same | Same | Same |
| `LocalWavelengthWarp` | `spectral.py` | Same | Same | Same |
| `SmoothMagnitudeWarp` | `spectral.py` | Same | Same | Same |
| `ScatterSimulationMSC` | `spectral.py` | `Augmenter` → `SpectraTransformerMixin` | `apply_on`, `copy` | `_requires_wavelengths = "optional"` |

**Note on `ScatterSimulationMSC`:** This operator is being enhanced in Phase 2.2 to add a wavelength-dependent tilt component (`gamma * wavelength`). To support this, it must become `SpectraTransformerMixin` now. The `"optional"` flag allows it to work without wavelengths (tilt disabled) for backward compatibility.

**Migration pattern for each:**
1. Change base class to `SpectraTransformerMixin`
2. Set `_requires_wavelengths = "optional"` (they fall back to `np.arange(n_features)`)
3. Remove `lambda_axis`, `apply_on`, `copy` from `__init__` (none of them use `apply_on` or `copy`)
4. Rename `augment(self, X, apply_on="samples")` → `transform(self, X, wavelengths=None)`
5. Replace `self.lambda_axis` with `wavelengths` parameter (keep `np.arange` fallback)
6. Replace `self.random_gen` with local `rng = np.random.default_rng(self.random_state)`
7. Remove `super().__init__(...)` call

**Commit boundary:** Single commit for all 7 operators + their tests.

**Testing strategy for this phase:**
1. Before migration: verify existing tests pass, identify any missing test coverage
2. After migration: equivalence test (same output when `wavelengths` matches old `lambda_axis`)
3. Fallback test: works without wavelengths (uses `np.arange`)
4. Pipeline test: used in `sample_augmentation` step with automatic wavelength injection
5. For `ScatterSimulationMSC`: verify tilt=0 (no wavelengths) produces same output as pre-migration

### Phase 1.3: Delete `Augmenter`, Migrate 18 Non-Wavelength Operators

**Changes:**

**A. Operators that USE `apply_on` (2 operators):**

| Operator | File | Change |
|----------|------|--------|
| `GaussianAdditiveNoise` | `spectral.py` | `Augmenter` → `TransformerMixin, BaseEstimator`. Keep `apply_on` and `random_state` as regular params. Rename `augment` → `transform`. Add `fit(self, X, y=None): return self`. |
| `MultiplicativeNoise` | `spectral.py` | Same pattern. |

**B. Operators that IGNORE `apply_on` (16 operators):**

| Operator | File |
|----------|------|
| `BandPerturbation` | `spectral.py` |
| `GaussianSmoothingJitter` | `spectral.py` |
| `UnsharpSpectralMask` | `spectral.py` |
| `BandMasking` | `spectral.py` |
| `ChannelDropout` | `spectral.py` |
| `SpikeNoise` | `spectral.py` |
| `LocalClipping` | `spectral.py` |
| `MixupAugmenter` | `spectral.py` |
| `LocalMixupAugmenter` | `spectral.py` |
| `Spline_Smoothing` | `splines.py` |
| `Spline_X_Perturbations` | `splines.py` |
| `Spline_Y_Perturbations` | `splines.py` |
| `Spline_X_Simplification` | `splines.py` |
| `Spline_Curve_Simplification` | `splines.py` |
| `Rotate_Translate` | `random.py` |
| `Random_X_Operation` | `random.py` |

**Pattern for each (group B):**
1. Change base class to `TransformerMixin, BaseEstimator`
2. Remove `apply_on` and `copy` from `__init__` (unused)
3. Keep `random_state` as regular param
4. Rename `augment(self, X, apply_on="samples")` → `transform(self, X)`
5. Replace `self.random_gen` with local `rng = np.random.default_rng(self.random_state)`
6. Add `fit(self, X, y=None): return self`

**Note on spline augmenters:** The spline operators in `splines.py` do NOT have `lambda_axis` parameters (verified via code inspection). They use `np.arange(n_features)` as their x-axis, which is pure index-based spline fitting. They remain as plain `TransformerMixin` and simply drop the unused `apply_on` parameter.

**Note on `IdentityAugmenter`:** This operator is not used anywhere in the codebase (verified via grep). It only exists in exports and the webapp node registry. Per the codebase's "no dead code" philosophy, it will be deleted entirely rather than migrated. The webapp node registry will be updated to remove it.

**Testing strategy for this phase:**
1. Before migration: check for existing tests, note any gaps
2. After migration: same output given same input + random_state
3. `transform(X)` works, `clone()` works, `get_params()`/`set_params()` round-trips
4. Used in `sample_augmentation` pipeline step: works

**Commit boundary:** Can be split into sub-commits by file (`spectral.py`, `splines.py`, `random.py`).

### Phase 1.4: Cleanup

| # | Task |
|---|------|
| 1 | Delete `Augmenter` class and `IdentityAugmenter` class from `abc_augmenter.py` |
| 2 | Remove `Augmenter` and `IdentityAugmenter` from `augmentation/__init__.py` exports |
| 3 | Remove `IdentityAugmenter` from `operators/__init__.py` and `operators/transforms/__init__.py` exports |
| 4 | Grep for `import.*Augmenter` — fix remaining references |
| 5 | Grep for `lambda_axis` — confirm zero remaining |
| 6 | Grep for `transform_with_wavelengths` — confirm zero remaining |
| 7 | Grep for `\.augment(` — confirm zero remaining calls |
| 8 | Grep for `\.copy` and `self.copy` — confirm zero remaining in augmenters |
| 9 | Run full `pytest tests/` |
| 10 | Run examples: `cd examples && ./run.sh -q` |
| 11 | Update webapp: remove `IdentityAugmenter` from node registry, update scripts |

**Commit boundary:** Single cleanup commit.

### Phase 1.5: Sample Augmentation Controller Behavior Preservation

**Context:** The `Augmenter` base class currently provides `apply_on` and `copy` parameters, and manages `random_gen` (RNG). After removing `Augmenter`, we must ensure the Sample Augmentation Controller still works correctly.

**Key Finding: Controllers Do NOT Reference `apply_on`**

Grep verification confirms: `apply_on` appears ZERO times in any controller file. The `apply_on` parameter is purely internal to the operator's math:

```python
# GaussianAdditiveNoise.augment() - current implementation
if apply_on == "global":
    scale = np.std(X) * self.sigma       # one std for entire matrix
else:
    stds = np.std(X, axis=1, keepdims=True)  # per-sample std
```

**After refactoring**, `apply_on` remains a regular constructor parameter stored as `self.apply_on` and used internally by `transform()`. The controller never reads it.

**Key Finding: Cloning Behavior Provides Variation**

The Sample Augmentation Controller's cloning behavior already provides the expected augmentation variation:

1. **Parallel path** (`_emit_augmentation_steps_parallel()`):
   - Creates ONE clone per (transformer, source, processing) combination
   - Each clone's `__init__` (or `fit()`) creates a fresh RNG from `random_state`
   - The SAME clone transforms ALL samples assigned to that transformer in one batch call
   - Per-sample variation comes from the RNG advancing during batch generation (see below)

2. **Sequential path** (`_emit_augmentation_steps_sequential()`):
   - Delegates to `TransformerMixinController._execute_for_sample_augmentation()`
   - Creates ONE clone per (source, processing) combination (cached in `fitted_transformers_cache`)
   - Same batch processing behavior

**How Per-Sample Variation Works (CRITICAL):**

Per-sample variation does NOT come from cloning per sample. It comes from batch RNG calls:

```python
# GaussianAdditiveNoise.augment() uses batch random generation
noise = self.random_gen.normal(0, 1, size=X.shape)  # shape = (n_samples, n_features)
#                                      ^^^^^
# This generates DIFFERENT random values for each sample in the batch
```

When `transform(X)` is called with a batch of 100 samples, the RNG generates 100 * n_features random values. Each sample gets a unique random pattern.

**Requirement: Batch RNG Pattern**

All stochastic operators MUST use batch-sized random generation:

| ✅ Correct | ❌ Wrong |
|-----------|----------|
| `rng.normal(0, 1, size=X.shape)` | `for i in range(n_samples): rng.normal(0, 1, size=(n_features,))` |
| `rng.uniform(0, 1, size=(n_samples, 1))` | `rng.uniform(0, 1)` (scalar, same for all samples) |

The wrong pattern would give all samples in a batch the same random values (scalar) or require explicit loops (slower).

**No Controller Changes Needed for `apply_on`**

The Sample Augmentation Controller does NOT need modification for `apply_on` behavior:
- Before: Operator stores `self.apply_on`, `Augmenter.transform()` passes it to `augment()`
- After: Operator stores `self.apply_on`, `transform()` uses it directly internally

The controller just calls `fit()` and `transform()` — standard sklearn pattern.

### Phase 1 RNG Contract

All stochastic operators follow this pattern after migration:

```python
def __init__(self, random_state=None, ...):
    self.random_state = random_state
    # Other params...

def fit(self, X, y=None):
    # Create RNG at fit() time - this is when the clone is ready
    self._rng = np.random.default_rng(self.random_state)
    return self

def transform(self, X):
    # Use fit-time RNG, fallback for unfitted direct calls
    rng = self._rng if hasattr(self, "_rng") else np.random.default_rng(self.random_state)

    # CRITICAL: Generate batch-sized random arrays for per-sample variation
    noise = rng.normal(0, self.sigma, size=X.shape)  # NOT size=(n_features,)

    return X + noise
```

**Contract:**
- `random_state=42`, single `transform()` call → deterministic output
- `random_state=42`, multiple `transform()` calls on same instance → different results (RNG advances)
- `random_state=42`, `clone()` then `fit()` then `transform()` → same as first cloned instance (RNG resets to seed)
- `random_state=None` → non-deterministic (entropy-seeded)

**Per-Sample Variation Mechanism:**
- The Sample Augmentation Controller clones operators per (transformer, source, processing), NOT per sample
- Per-sample variation comes from batch RNG calls: `rng.normal(size=X.shape)` generates unique values per sample
- This is the correct and expected behavior for augmentation

### Phase 1 Wavelength Unit Handling

The `SpectroDataset` stores spectra with a `HeaderUnit` that indicates the x-axis type:
- `HeaderUnit.WAVELENGTH` ("nm") — wavelength in nanometers
- `HeaderUnit.WAVENUMBER` ("cm-1") — wavenumber in cm⁻¹
- `HeaderUnit.INDEX` ("index") — integer indices
- `HeaderUnit.NONE` / `HeaderUnit.TEXT` — no numeric x-axis

**Decision:** The controller converts all spectral axis values to nanometers before passing to operators. Operators always receive wavelengths in nm (or `None` if unavailable).

**Implementation in controller:**
```python
def _extract_wavelengths(dataset, source_index, operator_name):
    """Extract wavelengths, converting to nm if needed."""
    header_unit = dataset.header_unit(source_index)
    raw_values = dataset.feature_headers(source_index)  # numeric values

    if header_unit == HeaderUnit.WAVENUMBER:
        # Convert cm⁻¹ to nm: nm = 10_000_000 / cm⁻¹
        return 10_000_000 / raw_values
    elif header_unit in (HeaderUnit.WAVELENGTH, HeaderUnit.INDEX):
        return raw_values
    else:
        raise ValueError(f"{operator_name} requires numeric wavelengths")
```

**Rationale:** Operators implement physics in wavelength space. Wavenumber data is common in FTIR but the physics (e.g., H-bond shift regions) is defined in nm. A single conversion point in the controller keeps operators simple.

---

## 3. Stage 2: Synthesis Mechanism Transfer to Augmentation

**Goal:** Every spectral transformation used in synthesis generation is also available as a standalone augmentation operator. The generator becomes a clean orchestrator: generative steps (Beer-Lambert, concentrations) + a chain of augmenter operators.

**Depends on:** Stage 1 complete (clean operator hierarchy).

### Phase 2.1: New Augmentation Operators

Create the following new operators in `operators/augmentation/`:

#### 2.1.1 `PathLengthAugmenter`

| Aspect | Detail |
|--------|--------|
| **File** | New file `operators/augmentation/path_length.py` or add to `spectral.py` |
| **Base class** | `TransformerMixin, BaseEstimator` (wavelength-independent) |
| **Physics** | `X_aug = X * L_i` where `L_i ~ N(1.0, sigma)` per sample |
| **Parameters** | `path_length_std: float = 0.05`, `random_state: int = None` |
| **Generator equivalent** | `generator._apply_path_length()` (lines 658-671) |

#### 2.1.2 `BatchEffectAugmenter`

| Aspect | Detail |
|--------|--------|
| **File** | New file `operators/augmentation/batch_effects.py` |
| **Base class** | `SpectraTransformerMixin` (`_requires_wavelengths = "optional"`) |
| **Physics** | `X_aug = gain * X + offset(lambda)`, where `gain ~ N(1.0, gain_std)`, `offset` is a smooth polynomial |
| **Parameters** | `n_batches: int = 3`, `offset_std: float = 0.02`, `gain_std: float = 0.03`, `polynomial_degree: int = 2`, `random_state: int = None` |
| **Generator equivalent** | `generator.generate_batch_effects()` (lines 868-898) |

#### 2.1.3 `InstrumentalBroadeningAugmenter`

| Aspect | Detail |
|--------|--------|
| **File** | New file `operators/augmentation/instrumental.py` |
| **Base class** | `SpectraTransformerMixin` (`_requires_wavelengths = True`) |
| **Physics** | Gaussian convolution: `X_aug = gaussian_filter1d(X, sigma_pts)` where `sigma_pts = FWHM / (2*sqrt(2*ln(2))) / wavelength_step` |
| **Parameters** | `fwhm_nm: float = 10.0`, `fwhm_range: Tuple[float, float] = None`, `random_state: int = None` |
| **Generator equivalent** | `generator._apply_instrumental_response()` (lines 781-800) |
| **Dependency** | `scipy.ndimage.gaussian_filter1d` |

#### 2.1.4 `HeteroscedasticNoiseAugmenter`

| Aspect | Detail |
|--------|--------|
| **File** | Add to `spectral.py` or new `operators/augmentation/noise.py` |
| **Base class** | `TransformerMixin, BaseEstimator` (signal-dependent, not wavelength-dependent) |
| **Physics** | `sigma_i = base_sigma + signal_dependent * |X_i|`, `X_aug = X + N(0, sigma_i)` |
| **Parameters** | `base_sigma: float = 0.001`, `signal_dependent_sigma: float = 0.005`, `random_state: int = None` |
| **Generator equivalent** | `generator._add_noise()` (lines 802-819) |

#### 2.1.5 `DeadBandAugmenter`

| Aspect | Detail |
|--------|--------|
| **File** | Add to `spectral.py` alongside `SpikeNoise` and `LocalClipping` |
| **Base class** | `TransformerMixin, BaseEstimator` |
| **Physics** | Sets contiguous wavelength regions to a constant (simulates detector failure) |
| **Parameters** | `n_bands_range: Tuple[int, int] = (1, 2)`, `width_range: Tuple[int, int] = (3, 10)`, `fill_value: float = 0.0`, `random_state: int = None` |
| **Generator equivalent** | Dead band portion of `generator._add_artifacts()` (lines 821-866) |

### Phase 2.2: Enhance Existing Augmenters

| Operator | Enhancement | Reason |
|----------|-------------|--------|
| `ScatterSimulationMSC` | Add `tilt_range: Tuple[float, float] = (0.0, 0.0)` parameter for wavelength-dependent tilt component. When non-zero, adds `gamma * wavelength` to the scatter model. Base class already changed to `SpectraTransformerMixin` in Phase 1.2. | Generator's `_apply_scatter()` includes tilt that the augmenter lacks. |
| `SpikeNoise` | Verify feature parity with synthesis `_add_artifacts()` spike generation. Ensure spike shape/width options match. | Synthesis may have more sophisticated spike shapes. |
| `LocalClipping` | Add saturation ceiling option: `saturation_value: float = None`. When set, clips at this value instead of computing from local statistics. | Synthesis `_add_artifacts()` has explicit saturation behavior. |

**Note:** `ScatterSimulationMSC` was migrated to `SpectraTransformerMixin` in Phase 1.2 to enable this enhancement. The tilt feature requires wavelengths, so the base class change is a prerequisite.

### Phase 2.3: Refactor Generator to Delegate

**Target:** `synthesis/generator.py` (or `synthesis/core/generator.py` after Stage 3)

Refactor the `generate()` method so that inline transformation methods delegate to augmentation operators. The generator becomes an orchestrator with two phases:

**Phase A — Generative (pure synthesis, stays inline):**
1. `generate_concentrations()` — sample composition
2. `_apply_beer_lambert()` — linear mixing

**Phase B — Effect chain (delegates to operators):**

Replace each inline method with an operator:

| Generator method | Replacement operator | Notes |
|-----------------|---------------------|-------|
| `_apply_path_length()` | `PathLengthAugmenter` | New (Phase 2.1.1) |
| `_generate_baseline()` | `PolynomialBaselineDrift` | Existing (migrated in Stage 1) |
| `_apply_global_slope()` | `LinearBaselineDrift` | Existing (migrated in Stage 1) |
| `_apply_scatter()` | `ScatterSimulationMSC` | Existing (enhanced in Phase 2.2) |
| `generate_batch_effects()` | `BatchEffectAugmenter` | New (Phase 2.1.2) |
| `_apply_wavelength_shift()` | `WavelengthShift` + `WavelengthStretch` | Existing (migrated in Stage 1) |
| `_apply_instrumental_response()` | `InstrumentalBroadeningAugmenter` | New (Phase 2.1.3) |
| `_add_noise()` | `HeteroscedasticNoiseAugmenter` | New (Phase 2.1.4) |
| `_add_artifacts()` | `SpikeNoise` + `DeadBandAugmenter` + `LocalClipping` | Existing + new |

**Operators that stay synthesis-only (not transferred):**
- `_apply_multi_sensor_stitching()` — acquisition process, not perturbation
- `_apply_multi_scan_averaging()` — noise reduction, not augmentation
- `generate_concentrations()` — this IS the data
- `_apply_beer_lambert()` — core generative model

**Generator structure after refactoring:**

```python
class SyntheticNIRSGenerator:
    def __init__(self, ...):
        # Build effect chain from config
        self._effect_chain = self._build_effect_chain()

    def _build_effect_chain(self):
        """Build the ordered list of operators from config."""
        chain = []
        if self.config.path_length_variation > 0:
            chain.append(PathLengthAugmenter(path_length_std=...))
        if self.config.baseline_enabled:
            chain.append(PolynomialBaselineDrift(degree=3, ...))
        # ... etc for each configurable effect
        return chain

    def generate(self, n_samples):
        # Phase A: Generative (pure synthesis)
        C = self.generate_concentrations(n_samples)
        A = self._apply_beer_lambert(C)

        # Phase B: Effect chain (all operators)
        for op in self._effect_chain:
            req = getattr(op, '_requires_wavelengths', False)
            if req:
                A = op.transform(A, wavelengths=self.wavelengths)
            else:
                A = op.transform(A)

        # Synthesis-only steps (remain inline)
        if self.include_multi_sensor:
            A = self._apply_multi_sensor_stitching(A)
        if self.include_multi_scan:
            A = self._apply_multi_scan_averaging(A)

        return A
```

**Verification:**
- Statistical equivalence test: generator output distributions (mean, std, spectral shape) must be statistically indistinguishable before/after refactoring with same random seed
- Run all synthesis tests: `pytest tests/unit/synthesis/`
- Run examples: `cd examples && ./run.sh -q`

### Phase 2.4: Tests for New Operators

Each new operator needs:

| Test category | Description |
|--------------|-------------|
| **Shape preservation** | `X_out.shape == X_in.shape` |
| **Determinism** | Same `random_state` → same output |
| **Stochasticity** | `random_state=None` → different outputs on repeated calls |
| **Edge cases** | Single sample, single feature, all-zeros input |
| **Clone compatibility** | `sklearn.clone(op)` works, `get_params()`/`set_params()` round-trips |
| **Pipeline integration** | Works in `sample_augmentation` step |
| **Equivalence** | Output matches the generator's inline implementation for same parameters |

---

## 4. Stage 3: Synthesis Externalization

**Goal:** Move `nirs4all/synthesis/` to `nirs4all/synthesis/` as a top-level package.

**Depends on:** Stage 2 complete (generator refactored, tests passing).

### Phase 3.1: Create New Package Structure

Per the design document's decision (D1 = Option B, grouped):

```
nirs4all/synthesis/
    __init__.py              (re-export public API)
    core/
        __init__.py
        generator.py         (SyntheticNIRSGenerator)
        builder.py           (SyntheticDatasetBuilder)
        config.py            (complexity presets, dataclasses)
        validation.py        (realism scorecard)
        accelerated.py       (GPU acceleration)
    components/
        __init__.py
        components.py        (NIRBand, SpectralComponent, ComponentLibrary)
        _bands.py            (48+ predefined components)
        wavenumber.py        (wavelength utilities)
        procedural.py        (procedural generation from functional groups)
    effects/
        __init__.py
        environmental.py     (TemperatureConfig, MoistureConfig)
        scattering.py        (ParticleSizeConfig, EMSCConfig)
        instruments.py       (InstrumentArchetype, MultiSensorConfig)
        detectors.py         (DetectorSpectralResponse, noise models)
    products/
        __init__.py
        domains.py           (application domain priors)
        products.py          (product templates)
        _aggregates.py       (predefined compositions)
        prior.py             (domain priors)
    inference/
        __init__.py
        fitter.py            (RealDataFitter)
        reconstruction/      (7+ files, physical forward model fitting)
    export/
        __init__.py
        exporter.py          (CSV/folder export)
        metadata.py          (metadata generation)
        targets.py           (target generation)
        sources.py           (multi-source support)
        benchmarks.py        (benchmark datasets)
```

### Phase 3.2: Move Files and Update Imports

**Hard break** (per codebase philosophy and design decision D2):

1. Move all files from `synthesis/` to `synthesis/` with the grouped structure above
2. Delete `synthesis/` entirely (no backward compat alias)
3. Bulk-update all import paths:
   - `from nirs4all.synthesis` → `from nirs4all.synthesis` (or subpackage)
   - Grep-verify: `from nirs4all.synthesis` returns zero hits after

**Files requiring import updates:**

| File | Import change |
|------|--------------|
| `api/generate.py` | All lazy imports: `from nirs4all.synthesis.core import SyntheticDatasetBuilder` etc. |
| `nirs4all-webapp/api/synthesis.py` | `from nirs4all.synthesis import SyntheticDatasetBuilder, available_components, ...` |
| `tests/unit/synthesis/*.py` | Move to `tests/unit/synthesis/` and update imports |
| `examples/*.py` | Update any `from nirs4all.synthesis` imports |
| `docs/` | Update documentation references |

### Phase 3.3: Verification

| Check | Command |
|-------|---------|
| All synthesis tests pass | `pytest tests/unit/synthesis/` |
| API layer works | `pytest tests/integration/` (exercises `nirs4all.generate()`) |
| Examples work | `cd examples && ./run.sh -q` |
| No remaining old imports | `grep -r "from nirs4all.synthesis" --include="*.py"` returns nothing |
| Webapp starts | `npm run dev:api` in webapp dir |

---

## 5. Stage 4: Webapp Alignment & Documentation

### Phase 4.1: Webapp Node Definition Updates

Update `nirs4all-webapp/src/data/nodes/definitions/augmentation/transforms.json`:

**Nodes to remove:**
- `IdentityAugmenter` — deleted from library (no longer exists)

**Parameters to remove from existing nodes:**
- `lambda_axis` from all augmenters that had it (6 spectral operators)
- `apply_on` from all augmenters that ignored it (keep for `GaussianAdditiveNoise`, `MultiplicativeNoise`)
- `copy` from all former `Augmenter` subclasses

**Parameters to add:**
- `ScatterSimulationMSC`: add `tilt_range` parameter
- `LocalClipping`: add `saturation_value` parameter

**New nodes to add:**
- `PathLengthAugmenter`
- `BatchEffectAugmenter`
- `InstrumentalBroadeningAugmenter`
- `HeteroscedasticNoiseAugmenter`
- `DeadBandAugmenter`

**classPath updates** for any operators whose import paths changed.

**Scripts to update:**
- `scripts/generate_component_library.py` — remove `IdentityAugmenter` entry
- `scripts/generate_extended_registry.py` — remove `IdentityAugmenter` entry

**Validation:** `npm run validate:nodes`

### Phase 4.2: Documentation Updates

| Document | Change |
|----------|--------|
| `docs/source/user_guide/augmentation/augmentations.md` | Remove `Augmenter` base class docs. Update `SpectraTransformerMixin` examples (remove `transform_with_wavelengths`). Add new operators. |
| `CLAUDE.md` | Update `SpectraTransformerMixin` code example to use `transform()` instead of `transform_with_wavelengths()`. Update `synthesis/` references to `synthesis/`. |
| `CHANGELOG.md` | Document breaking changes: `Augmenter` deleted, `IdentityAugmenter` deleted, `transform_with_wavelengths` removed, `lambda_axis` removed from 6 operators, `apply_on` removed from most operators, synthesis moved to `nirs4all/synthesis/`. |
| Section 9 of operator_base_class_design.md | Can be adapted into a user-facing migration guide if needed. |

### Phase 4.3: Example Updates

- Verify all examples in `examples/user/` and `examples/developer/` still work
- Update any examples that import from `nirs4all.synthesis` directly
- Update examples that construct augmenters with `apply_on=` or `lambda_axis=`

---

## 6. Detailed Review: Maintainability & Performance

### 6.1 Maintainability Assessment

#### 6.1.1 Stage 1: Operator Hierarchy — Positive Impact

**Reduction in conceptual surface area:**
- Before: 3 method names (`transform`, `augment`, `transform_with_wavelengths`), 2 base classes, 2 abstract methods
- After: 1 method name (`transform`), 1 optional mixin (`SpectraTransformerMixin`), 0 abstract methods
- New operator authors have one decision: "does this need wavelengths?"

**Protocol-based detection (`_requires_wavelengths` via `getattr`):**
- Decouples operators from the framework — any operator can opt into wavelength injection without inheriting `SpectraTransformerMixin`
- The `"optional"` value for `_requires_wavelengths` is a pragmatic choice. It avoids a custom enum import dependency in every operator file. If the codebase later needs more granularity (e.g., `"preferred"`), this could be refactored to an enum, but adding an enum now for 3 values would be over-engineering.

**Risk: Phase 1.1 atomicity.** Changing the mixin and all 9 subclasses in one commit is a large diff. However, the alternative (a temporary shim in `SpectraTransformerMixin` that supports both `transform_with_wavelengths` and `transform` overrides) adds dead code that the codebase philosophy prohibits. The atomic approach is correct — the diff is large but each change is trivial (method rename).

#### 6.1.2 Stage 1: `lambda_axis` Migration — Moderate Risk

**Behavioral change:** After migration, the controller injects wavelengths automatically. Users who constructed `LinearBaselineDrift(lambda_axis=my_wavelengths)` will need to remove the parameter. The operator will receive wavelengths from the dataset.

**Edge case:** If the operator is used outside a pipeline (standalone `op.transform(X)`), wavelengths won't be injected. The `"optional"` flag means the operator falls back to `np.arange(n_features)`, which matches the current behavior when `lambda_axis=None`. This is correct but worth documenting.

#### 6.1.3 Stage 2: New Augmenters — Low Risk

These are additive changes. Each new operator is independent and testable in isolation. The only coupling is the generator refactoring (Phase 2.3), which is the most complex change in this stage.

#### 6.1.4 Stage 2: Generator Refactoring — High Complexity

The generator's `generate()` method is ~240 lines with intricate control flow (conditional flags, batch effects, multi-sensor, multi-scan). Refactoring it to use an effect chain requires:
- Mapping each config parameter to the correct operator constructor argument
- Preserving the exact execution order (effects are order-dependent in physics)
- Ensuring the RNG state progression is identical (for reproducibility with same seed)

**Recommendation:** Write a "golden test" before refactoring: generate 100 spectra with a fixed seed, save the output. After refactoring, verify the output is bitwise identical. If RNG ordering changes (because operators create their own RNGs), accept statistical equivalence rather than bitwise identity, and document the seed-compatibility break.

**Generator effect chain order preservation is critical.** The physics demands a specific order (Beer-Lambert → path length → baseline → scatter → wavelength shift → broadening → noise). The `_build_effect_chain()` method must enforce this order explicitly, not rely on dict ordering or configuration file order. A simple list construction with explicit `if` guards (as shown in the design) is the correct approach.

#### 6.1.5 Stage 3: Externalization — Low Risk, High Tedium

This is a mechanical operation. The risk is in missing an import somewhere. Mitigation:
- Use IDE refactoring or `sed`-based bulk rename
- Grep verification before committing
- Full test suite after

**The grouped package structure (Option B) is the right call for 30+ files.** The flat structure would make the `synthesis/` directory overwhelming. The groupings (`core/`, `components/`, `effects/`, `products/`, `inference/`, `export/`) match the conceptual domains well.

#### 6.1.6 Cross-Stage: Webapp Coupling

The webapp node definitions are pure JSON/TypeScript and independent of the Python library. They only need updating when:
- Parameter names change (`lambda_axis` removed, `apply_on` removed)
- New operators are added
- Import paths change (`classPath` field)

This is a separate concern that can lag behind the Python changes without breaking the webapp (it just won't reflect the new operators until updated).

### 6.2 Performance Assessment

#### 6.2.1 Stage 1: Controller Wavelength Detection

**Before:** `isinstance(op, SpectraTransformerMixin) and getattr(op, '_requires_wavelengths', False)` — one isinstance check + one getattr.

**After:** `getattr(op, '_requires_wavelengths', False)` — one getattr.

**Performance change:** Negligible. The `isinstance` check was already fast. Removing it makes the detection marginally faster but this is not a hot path.

#### 6.2.2 Stage 1: RNG Change

**Before:** `Augmenter.transform()` seeds the global RNG (`np.random.seed()`), then `augment()` uses `self.random_gen`.

**After:** Each `transform()` call creates a local `np.random.default_rng(self.random_state)`.

**Performance concern:** `np.random.default_rng()` allocates a new generator object on each call. For operators called in a tight loop (e.g., `SampleAugmentationController` processing thousands of samples), this could add overhead.

**Mitigation:** The RNG contract (Section "Phase 1 RNG Contract") creates the RNG in `fit()` and reuses it in `transform()`. Since the controller clones operators per sample, `fit()` is called once per clone, and `transform()` once per clone. The allocation happens once per sample, not once per feature. This is acceptable — the overhead of `default_rng()` (~1μs) is negligible compared to the actual transform computation (typically 10-1000μs per sample).

**The global seeding removal is a net positive for performance** in multi-threaded scenarios. Global `np.random.seed()` is thread-unsafe and can cause contention. Local RNGs are thread-local and safe for the `ThreadPoolExecutor` used in parallel augmentation.

#### 6.2.3 Stage 1: Parallel Augmentation Fix

**Before:** Parallel path doesn't pass wavelengths → wavelength-aware operators fail.

**After:** Parallel path resolves wavelengths once per source (cached), passes to fit/transform.

**Performance impact:** The wavelength cache (`wavelengths_cache`) adds one dict lookup per source per operator. Since source counts are typically 1-3, this is negligible. The cache prevents redundant `dataset.wavelengths_nm()` calls.

#### 6.2.4 Stage 2: Generator Effect Chain vs. Inline Methods

**Before:** Each effect is an inline method call on `self` — zero allocation overhead, direct array manipulation.

**After:** Each effect is an operator instance in a list — one method dispatch per operator, one `getattr` check for `_requires_wavelengths`.

**Performance concern:** The generator may create 10-15 operator instances during `__init__()`. Each `transform()` call has the overhead of:
1. Method dispatch (Python method call ~100ns)
2. `_requires_wavelengths` check if using the `getattr` protocol
3. Local RNG creation (if stochastic)

For a typical generation run (1000+ samples), the per-operator overhead is ~1μs, totaling ~15μs per generation call. The actual computation per operator (matrix multiplication, convolution, noise generation) is 1-100ms. **The overhead is negligible** — less than 0.01% of total generation time.

**Memory:** Each operator stores its parameters (~100 bytes each). The effect chain of 15 operators adds ~1.5KB to the generator's memory footprint. Irrelevant.

#### 6.2.5 Stage 3: Import Path Changes

Moving modules to a new package path has zero runtime performance impact. Python's import system caches modules after first import.

### 6.3 Risk Matrix Summary

| Change | Complexity | Risk | Mitigation |
|--------|-----------|------|------------|
| Phase 1.1: Mixin + 9 operators + controller | Medium | Medium | Atomic commit, comprehensive tests |
| Phase 1.2: 7 wavelength-dependent migrations | Low | Low | Equivalence tests per operator |
| Phase 1.3: 18 Augmenter migrations | High (tedium) | Low | Mechanical pattern, per-file commits |
| Phase 1.4: Cleanup + IdentityAugmenter removal | Low | Low | Grep verification |
| Phase 2.1: 5 new operators | Medium | Low | Independent, additive |
| Phase 2.2: 2 operator enhancements | Low | Low | Backward-compatible parameter additions |
| Phase 2.3: Generator refactoring | High | High | Golden test, statistical equivalence |
| Phase 3: Synthesis externalization | Medium (tedium) | Medium | Bulk rename, grep verify, full tests |
| Phase 4: Webapp + docs | Low | Low | Independent, can lag |

### 6.4 Design Decisions (Resolved)

The following questions were resolved during review:

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| D1 | **Spline augmenters and `lambda_axis`** | Splines do NOT have `lambda_axis`. They remain as plain `TransformerMixin`. | Code inspection confirmed splines use `np.arange(n_features)` for index-based fitting. No wavelength dependency. |
| D2 | **`ScatterSimulationMSC` base class** | Migrate to `SpectraTransformerMixin` with `_requires_wavelengths = "optional"` in Phase 1.2. | Phase 2.2 adds wavelength-dependent tilt (`gamma * wavelength`). Must have wavelength support now. |
| D3 | **`IdentityAugmenter` placement** | Delete entirely. | Grep confirmed it's not used anywhere in the codebase (only exports and webapp registry). Per "no dead code" philosophy. |
| D4 | **Test coverage approach** | Integrate testing into each phase. Check existing tests before migration, add missing coverage as part of migration work. | Avoids test redundancy. Each phase's testing section specifies the strategy. |
| D5 | **HeaderUnit handling** | Controller converts wavenumber (cm⁻¹) to nm before passing to operators. Operators always receive nm. | Single conversion point keeps operators simple. Physics is defined in wavelength space. |
| D6 | **Commit style for Phase 1.1** | Keep atomic (single commit). | Large diff but correct. Each change is trivial (method rename). No temporary shims per codebase philosophy. |