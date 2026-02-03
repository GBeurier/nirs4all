# Design Document: Synthesis Externalization & Augmentation Transfer

**Roadmap items:**
- `[Synthesis] Externalize synthesis module out of data`
- `[Augmentations] Transfer synthesis mechanisms (temperature, machine, noise, etc.) to sample_augmentations operators`

**Status:** Investigation & Design
**Date:** February 2026

---

## Table of Contents

1. [Current State of Transformations in Synthesis](#1-current-state-of-transformations-in-synthesis)
2. [Current State of Externalization of Synthesis](#2-current-state-of-externalization-of-synthesis)
3. [Current State of Sample Augmentation and SpectraTransformerMixin](#3-current-state-of-sample-augmentation-and-spectratransformermixin)
4. [Objectives: Detailed Reformulations](#4-objectives-detailed-reformulations)
5. [Design of Physical/Chemical/Optical Transformations for Augmentation](#5-design-of-physicalchemicaloptical-transformations-for-augmentation)
6. [Detailed Review and Open Questions](#6-detailed-review-and-open-questions)
7. [Point of View](#7-point-of-view)

---

## 1. Current State of Transformations in Synthesis

### 1.1 Overview

The synthesis module (`nirs4all/synthesis/`) implements a physics-informed forward model for generating synthetic NIR spectra. The generation pipeline applies transformations sequentially, each simulating a real physical, instrumental, or environmental phenomenon.

### 1.2 The Generation Chain

The core generation model in `SyntheticNIRSGenerator` (`generator.py`) follows Beer-Lambert law:

```
A_i(lambda) = L_i * SUM_k( c_ik * epsilon_k(lambda) ) + baseline_i + scatter_i + noise_i
```

The sequential transformation chain is:

| Step | Transformation | Method | Physics/Phenomenon |
|------|---------------|--------|-------------------|
| 1 | Concentration generation | `generate_concentrations()` | Sample composition (Dirichlet, uniform, lognormal, correlated) |
| 2 | Beer-Lambert law | `_apply_beer_lambert()` | Linear mixing: `A = C @ E` (concentrations x absorptivities) |
| 3 | Path length variation | `_apply_path_length()` | Optical path differences: `A *= L_i` |
| 4 | Baseline drift | `_generate_baseline()` | Polynomial baseline: `b0 + b1*x + b2*x^2 + b3*x^3` |
| 5 | Global slope | `_apply_global_slope()` | Typical NIR upward trend from scattering |
| 6 | Scatter effects | `_apply_scatter()` | Multiplicative/additive MSC-like: `alpha*A + beta + gamma*x` |
| 7 | Batch effects | `generate_batch_effects()` | Session-to-session offset and gain drift |
| 8 | Wavelength shift/stretch | `_apply_wavelength_shift()` | Calibration errors via interpolation |
| 9 | Instrumental broadening | `_apply_instrumental_response()` | Gaussian convolution (FWHM-based ILS) |
| 10 | Noise | `_add_noise()` | Heteroscedastic: `sigma = base + signal_dep * |A|` |
| 11 | Multi-sensor stitching | `_apply_multi_sensor_stitching()` | Detector junction blending artifacts |
| 12 | Multi-scan averaging | `_apply_multi_scan_averaging()` | SNR improvement: `~sqrt(n_scans)` |
| 13 | **Temperature effects** | via `TemperatureAugmenter` | Band shifts, intensity changes, broadening |
| 14 | **Moisture effects** | via `MoistureAugmenter` | Free/bound water balance |
| 15 | **Particle size scattering** | via `ParticleSizeAugmenter` | Wavelength-dependent: `lambda^(-n)` |
| 16 | **EMSC distortion** | via `EMSCDistortionAugmenter` | Polynomial scatter model |
| 17 | **Edge artifacts** | via edge artifact augmenters | Roll-off, stray light, curvature, truncated peaks |
| 18 | Random artifacts | `_add_artifacts()` | Spikes, dead bands, saturation |

### 1.3 Critical Finding: Partial Delegation to Augmentation

Steps 13-17 already delegate to augmentation operators. The generator instantiates augmenters directly:

```python
# In generator.py ~lines 400-545:
from nirs4all.operators.augmentation import (
    TemperatureAugmenter, MoistureAugmenter,
    ParticleSizeAugmenter, EMSCDistortionAugmenter,
    DetectorRollOffAugmenter, StrayLightAugmenter,
    EdgeCurvatureAugmenter, TruncatedPeakAugmenter,
)
self._temperature_op = TemperatureAugmenter(...)
self._moisture_op = MoistureAugmenter(...)
self._particle_op = ParticleSizeAugmenter(...)
# etc.
```

This is important: the synthesis module is NOT reimplementing these effects but reusing augmentation operators. However, steps 3-10 and step 18 remain internal to the generator.

### 1.4 Transformations Implemented Only in Synthesis

The following transformations have **no augmentation operator equivalent**:

| Transformation | Implementation | Augmentation Equivalent |
|---------------|---------------|------------------------|
| Path length variation | `generator._apply_path_length()` | None |
| Polynomial baseline (degree 3) | `generator._generate_baseline()` | Partial: `PolynomialBaselineDrift` exists but not reused |
| Global slope | `generator._apply_global_slope()` | Partial: `LinearBaselineDrift` exists but not reused |
| MSC-like scatter | `generator._apply_scatter()` | Partial: `ScatterSimulationMSC` exists but not reused |
| Wavelength shift/stretch | `generator._apply_wavelength_shift()` | `WavelengthShift` and `WavelengthStretch` exist but not reused |
| Instrumental broadening | `generator._apply_instrumental_response()` | None |
| Multi-sensor stitching | `generator._apply_multi_sensor_stitching()` | None |
| Multi-scan averaging | `generator._apply_multi_scan_averaging()` | None |
| Heteroscedastic noise | `generator._add_noise()` | `GaussianAdditiveNoise` exists but is simpler (not heteroscedastic) |
| Random artifacts | `generator._add_artifacts()` | `SpikeNoise` and `LocalClipping` exist but incomplete |
| Batch effects | `generator.generate_batch_effects()` | None |

### 1.5 Configuration System in Synthesis

Effects are configured through dataclass hierarchies in `synthesis/`:

- **`environmental.py`**: `TemperatureConfig`, `TemperatureEffectParams`, `MoistureConfig`
- **`scattering.py`**: `ParticleSizeConfig`, `ParticleSizeDistribution`, `EMSCConfig`, `ScatteringEffectsConfig`
- **`instruments.py`**: `InstrumentArchetype`, `MultiSensorConfig`, `MultiScanConfig`, `EdgeArtifactsConfig`, `NoiseModelConfig`
- **`detectors.py`**: `DetectorSpectralResponse`, noise models (shot, thermal, read, 1/f)
- **`config.py`**: `COMPLEXITY_PARAMS` (simple/realistic/complex presets)

These configs are translated to augmenter parameters at instantiation time in `generator.py`.

### 1.6 Spectral Components and Forward Model

The synthesis module also contains a spectral component library system:

- **`components.py`**: `NIRBand` (Voigt profiles), `SpectralComponent` (multi-band compounds), `ComponentLibrary`
- **`_bands.py`**: 48+ predefined components (water, protein, lipid, starch, cellulose, etc.)
- **`wavenumber.py`**: Wavelength/wavenumber conversion, NIR zone classification, overtone calculation
- **`procedural.py`**: Procedural generation from functional groups (C-H, O-H, N-H)
- **`domains.py`**: Application domain priors (agriculture, dairy, pharma, food)
- **`products.py`**: Product templates with realistic compositions
- **`_aggregates.py`**: Predefined aggregate compositions (wheat, milk, cheese, tablet)

These are pure synthesis concerns and have no augmentation equivalent (they define *what* spectra look like, not *how* they vary).

---

## 2. Current State of Externalization of Synthesis

### 2.1 Current Module Location

The synthesis module currently resides at:

```
nirs4all/nirs4all/synthesis/
    __init__.py          (exports 70+ symbols)
    generator.py         (core generator, ~1200 lines)
    builder.py           (fluent builder, ~800 lines)
    components.py        (spectral components)
    _bands.py            (48+ predefined bands)
    wavenumber.py        (wavelength utilities)
    procedural.py        (procedural generation)
    domains.py           (application domain priors)
    products.py          (product templates)
    _aggregates.py       (aggregate compositions)
    environmental.py     (temperature/moisture configs)
    scattering.py        (scattering effect configs)
    instruments.py       (instrument archetypes)
    detectors.py         (detector models)
    config.py            (complexity presets)
    validation.py        (realism scorecard)
    fitter.py            (real data fitting)
    metadata.py          (metadata generation)
    targets.py           (target generation)
    sources.py           (multi-source support)
    exporter.py          (CSV/folder export)
    accelerated.py       (GPU acceleration)
    prior.py             (domain priors)
    benchmarks.py        (benchmark datasets)
    reconstruction/      (physical forward model fitting, 7+ files)
```

This is nested inside the `data` package alongside loaders, parsers, and dataset management - modules that handle *real* data input. Synthesis generates *new* data, which is a fundamentally different concern.

### 2.2 API Layer: Already Externalized

The public API is fully decoupled via `nirs4all/api/generate.py`:

```python
# nirs4all/api/__init__.py
from .generate import generate_namespace as generate  # line 54

# nirs4all/__init__.py
from .api import generate  # line 59
```

Users access synthesis via `nirs4all.generate()` and never import from `nirs4all.synthesis` directly. All imports in `api/generate.py` are lazy:

```python
def regression(...):
    from nirs4all.synthesis import SyntheticDatasetBuilder  # lazy
    builder = SyntheticDatasetBuilder(...)
```

### 2.3 Internal Import Coupling

Only `api/generate.py` imports from `nirs4all.synthesis`. The coupling is:

| Import in api/generate.py | Used for |
|--------------------------|---------|
| `SyntheticDatasetBuilder` | All builder-based functions |
| `SyntheticNIRSGenerator` | Direct generator access |
| `ProductGenerator` | Product template generation |
| `CategoryGenerator` | Multi-template generation |
| `RealDataFitter` | From-template generation |
| `generate_multi_source` | Multi-source datasets |

Beyond the API layer, synthesis is imported by:
- Test files in `tests/unit/synthesis/`
- Example files in `examples/`
- The webapp backend in `nirs4all-webapp/api/synthesis.py`

### 2.4 What Has NOT Been Done

The module itself still lives under `synthesis/`. No `nirs4all/synthesis/` top-level package exists. The roadmap task "Externalize synthesis module out of data" refers to physically moving the module to its own top-level package, which has not happened.

### 2.5 The `reconstruction` Submodule

The `synthesis/reconstruction/` submodule implements a physical signal-chain reconstruction workflow with:
- `CanonicalForwardModel`, `InstrumentModel`, `EnvironmentalEffectsModel`
- `VariableProjectionSolver`, `GlobalCalibrator`
- `ParameterDistributionFitter`, `ReconstructionGenerator`

This is tightly coupled to synthesis and should move together.

---

## 3. Current State of Sample Augmentation and SpectraTransformerMixin

### 3.1 SpectraTransformerMixin Architecture

The `SpectraTransformerMixin` (`operators/base/spectra_mixin.py`) provides wavelength-aware operator support:

```python
class SpectraTransformerMixin(TransformerMixin, BaseEstimator):
    _requires_wavelengths: bool = True

    def transform(self, X, wavelengths=None):
        if wavelengths is None and self._requires_wavelengths:
            raise ValueError(...)
        return self.transform_with_wavelengths(X, wavelengths)

    @abstractmethod
    def transform_with_wavelengths(self, X, wavelengths):
        ...
```

The `TransformerMixinController` (priority 10) detects these instances and automatically passes wavelengths:

```python
@staticmethod
def _needs_wavelengths(operator):
    return (isinstance(operator, SpectraTransformerMixin) and
            getattr(operator, '_requires_wavelengths', False))

@staticmethod
def _extract_wavelengths(dataset, source_index, operator_name):
    return dataset.wavelengths_nm(source_index)
```

### 3.2 Augmenter Base Class

The `Augmenter` base class (`operators/augmentation/abc_augmenter.py`) extends sklearn's `TransformerMixin`:

```python
class Augmenter(TransformerMixin, BaseEstimator):
    def __init__(self, apply_on="samples", random_state=None, copy=True):
        ...
    def transform(self, X, y=None):
        return self.augment(X, apply_on=self.apply_on)
    @abstractmethod
    def augment(self, X, apply_on="samples"):
        ...
```

Note: `Augmenter` does NOT inherit from `SpectraTransformerMixin`. Wavelength-aware augmenters inherit from `SpectraTransformerMixin` directly, bypassing the `Augmenter` base class.

### 3.3 Complete Augmenter Inventory

| Category | Augmenters | Wavelength-Aware | Base Class |
|----------|-----------|------------------|------------|
| **Base** | `Augmenter`, `IdentityAugmenter` | No | `Augmenter` |
| **Random** | `Random_X_Operation`, `Rotate_Translate` | No | `Augmenter` |
| **Splines** | `Spline_Smoothing`, `Spline_X_Perturbations`, `Spline_Y_Perturbations`, `Spline_Curve_Simplification`, `Spline_X_Simplification` | No | `Augmenter` |
| **Spectral Noise** | `GaussianAdditiveNoise`, `MultiplicativeNoise` | No | `Augmenter` |
| **Spectral Baseline** | `LinearBaselineDrift`, `PolynomialBaselineDrift` | No (uses `lambda_axis` param) | `Augmenter` |
| **Spectral Wavelength** | `WavelengthShift`, `WavelengthStretch`, `LocalWavelengthWarp`, `SmoothMagnitudeWarp` | No | `Augmenter` |
| **Spectral Features** | `BandPerturbation`, `GaussianSmoothingJitter`, `UnsharpSpectralMask` | No | `Augmenter` |
| **Spectral Masking** | `BandMasking`, `ChannelDropout` | No | `Augmenter` |
| **Spectral Artifacts** | `SpikeNoise`, `LocalClipping` | No | `Augmenter` |
| **Spectral Combinations** | `MixupAugmenter`, `LocalMixupAugmenter`, `ScatterSimulationMSC` | No | `Augmenter` |
| **Environmental** | `TemperatureAugmenter`, `MoistureAugmenter` | **Yes** | `SpectraTransformerMixin` |
| **Scattering** | `ParticleSizeAugmenter`, `EMSCDistortionAugmenter` | **Yes** | `SpectraTransformerMixin` |
| **Edge Artifacts** | `DetectorRollOffAugmenter`, `StrayLightAugmenter`, `EdgeCurvatureAugmenter`, `TruncatedPeakAugmenter`, `EdgeArtifactsAugmenter` | **Yes** | `SpectraTransformerMixin` |

**Total: ~35 augmenters**, of which **9 are wavelength-aware** (all physical-effects augmenters).

### 3.4 Sample Augmentation Controller

The `SampleAugmentationController` (`controllers/data/sample_augmentation.py`) handles:

- **Standard mode**: Generate N augmented copies per sample
- **Balanced mode**: Augment to balance class distribution (target_size or max_factor)
- **Delegation**: Emits `run_step` events that trigger `TransformerMixinController` per transformer

Pipeline syntax:
```python
{"sample_augmentation": {
    "transformers": [TemperatureAugmenter(temperature_range=(-5, 5)), ...],
    "count": 2,
}}
```

### 3.5 Two Parallel Hierarchies

There are effectively two base class hierarchies for augmenters:

1. **`Augmenter`** (from `abc_augmenter.py`): Non-wavelength-aware, simple `augment(X)` interface
2. **`SpectraTransformerMixin`** (from `spectra_mixin.py`): Wavelength-aware, `transform_with_wavelengths(X, wl)` interface

The spectral augmenters in `spectral.py` (noise, baseline, wavelength distortions) use the `Augmenter` base. The physics-based augmenters (environmental, scattering, edge) use `SpectraTransformerMixin`. This creates an inconsistency: baseline drift augmenters handle wavelength information via a `lambda_axis` constructor parameter rather than the standard `SpectraTransformerMixin` pattern.

---

## 4. Objectives: Detailed Reformulations

### 4.1 Task 1: Externalize Synthesis Module

**Roadmap:** `[Synthesis] Externalize synthesis module out of data`

**Reformulated objective:**

Move `nirs4all/synthesis/` to `nirs4all/synthesis/` as a top-level package, establishing synthesis as a first-class citizen of the library alongside `data`, `pipeline`, `operators`, and `controllers`.

**Scope:**
1. Create `nirs4all/synthesis/` with all files from `synthesis/`
2. Update all import paths (`from nirs4all.synthesis` -> `from nirs4all.synthesis`)
3. Update `api/generate.py` lazy imports
4. Update tests (move `tests/unit/synthesis/` -> `tests/unit/synthesis/`)
5. Update examples and documentation
6. Update webapp backend `synthesis.py` imports

**Why this matters:**
- The `data` package should only handle real data: loading, parsing, format detection, dataset management
- Synthesis is a data *creation* concern, not a data *storage/management* concern
- Aligns with the already-externalized API layer (`nirs4all.generate`)
- Reduces cognitive load: `data/` has 30+ synthesis files alongside loaders/parsers

### 4.2 Task 2: Transfer Synthesis Mechanisms to Augmentation Operators

**Roadmap:** `[Augmentations] Transfer synthesis mechanisms (temperature, machine, noise, etc.) to sample_augmentations operators`

**Reformulated objective:**

Ensure that every spectral transformation used in synthesis generation is also available as a standalone augmentation operator in `operators/augmentation/`, so that:
1. Users can apply the same physical transformations to real data for augmentation
2. The synthesis generator reuses these operators rather than implementing effects inline
3. A single source of truth exists for each physical phenomenon

**What "transfer" means concretely:**

For effects **already transferred** (synthesis already delegates to augmenters):
- Temperature, moisture, particle size, EMSC, edge artifacts: done

For effects **not yet transferred** (still inline in generator.py):
- Path length variation -> new `PathLengthAugmenter`
- Polynomial baseline drift -> reuse existing `PolynomialBaselineDrift` (update generator)
- Global slope -> reuse existing `LinearBaselineDrift` (update generator)
- MSC-like scatter -> reuse existing `ScatterSimulationMSC` or enhance it
- Wavelength shift/stretch -> reuse existing `WavelengthShift`/`WavelengthStretch`
- Instrumental broadening -> new `InstrumentalBroadeningAugmenter`
- Heteroscedastic noise -> new `HeteroscedasticNoiseAugmenter` or enhance `GaussianAdditiveNoise`
- Batch effects -> new `BatchEffectAugmenter`
- Multi-sensor stitching -> new `MultiSensorStitchingAugmenter` (complex, may remain synthesis-only)
- Multi-scan averaging -> new `MultiScanAveragingAugmenter` (may remain synthesis-only)
- Random artifacts -> enhance existing `SpikeNoise` + new augmenters for dead bands and saturation

---

## 5. Design of Physical/Chemical/Optical Transformations for Augmentation

### 5.1 Classification of Transformations

All spectral transformations in NIRS can be categorized by their physical origin:

#### A. Chemical / Compositional

| Phenomenon | Physical Basis | In Synthesis | In Augmentation |
|-----------|---------------|-------------|----------------|
| Concentration variation | Beer-Lambert: `A = c * epsilon * L` | `generate_concentrations()` | N/A (this IS the data) |
| Non-linear interactions | Saturation, hydrogen bonding | `with_nonlinear_targets()` | N/A |
| Path length variation | Optical path differences from sample geometry | `_apply_path_length()` | **Missing** |

Path length is the only chemical/compositional effect that makes sense as an augmenter. It models the real variation in sample thickness, cuvette differences, or fiber optic contact pressure.

**Proposed: `PathLengthAugmenter(SpectraTransformerMixin)`**
```
X_aug = X * L_i, where L_i ~ N(1.0, sigma)
```

#### B. Scattering / Light-Matter Interaction

| Phenomenon | Physical Basis | In Synthesis | In Augmentation |
|-----------|---------------|-------------|----------------|
| Multiplicative scatter | MSC model: `X_obs = a + b * X_true` | `_apply_scatter()` | `ScatterSimulationMSC` (partial) |
| Particle size effects | Mie/Rayleigh: wavelength-dependent `lambda^(-n)` | Config -> `ParticleSizeAugmenter` | `ParticleSizeAugmenter` |
| EMSC distortion | `x = a + b*x_ref + c1*lambda + c2*lambda^2` | Config -> `EMSCDistortionAugmenter` | `EMSCDistortionAugmenter` |
| Global slope | Scattering trend across NIR range | `_apply_global_slope()` | `LinearBaselineDrift` (not reused) |

`ScatterSimulationMSC` exists but the generator doesn't use it. The generator's `_apply_scatter()` includes a tilt component that `ScatterSimulationMSC` lacks. Either enhance the augmenter or have the generator reuse it.

#### C. Environmental / Physical Conditions

| Phenomenon | Physical Basis | In Synthesis | In Augmentation |
|-----------|---------------|-------------|----------------|
| Temperature | H-bond weakening shifts O-H bands, broadens peaks | Config -> `TemperatureAugmenter` | `TemperatureAugmenter` |
| Moisture | Free/bound water equilibrium shifts 1410-1460 nm | Config -> `MoistureAugmenter` | `MoistureAugmenter` |
| Batch effects | Session drift (baseline, gain) | `generate_batch_effects()` | **Missing** |

Temperature and moisture effects are already transferred. Batch effects model systematic differences between measurement sessions (different day, different operator, instrument warm-up state).

**Proposed: `BatchEffectAugmenter(SpectraTransformerMixin)`**
```
X_aug = gain_batch * X + offset_batch(lambda)
offset ~ smooth polynomial, gain ~ N(1.0, 0.03)
```

#### D. Instrumental / Hardware

| Phenomenon | Physical Basis | In Synthesis | In Augmentation |
|-----------|---------------|-------------|----------------|
| Detector roll-off | Sensitivity drops at wavelength edges | Config -> `DetectorRollOffAugmenter` | `DetectorRollOffAugmenter` |
| Stray light | `T_obs = (T + s) / (1 + s)` | Config -> `StrayLightAugmenter` | `StrayLightAugmenter` |
| Edge curvature | Optical aberrations at boundaries | Config -> `EdgeCurvatureAugmenter` | `EdgeCurvatureAugmenter` |
| Truncated peaks | Band centers outside measured range | Config -> `TruncatedPeakAugmenter` | `TruncatedPeakAugmenter` |
| Instrumental broadening | ILS convolution (Gaussian FWHM) | `_apply_instrumental_response()` | **Missing** |
| Wavelength calibration shift | Wavelength axis offset | `_apply_wavelength_shift()` | `WavelengthShift` (not reused) |
| Wavelength stretch | Wavelength axis compression/expansion | `_apply_wavelength_shift()` | `WavelengthStretch` (not reused) |
| Multi-sensor stitching | Junction artifacts between detectors | `_apply_multi_sensor_stitching()` | **Missing** (complex) |
| Multi-scan averaging | Noise reduction from repeated scans | `_apply_multi_scan_averaging()` | **Missing** (inverse operation) |

Edge artifacts are already transferred. Instrumental broadening is an important missing piece - it models the finite resolution of the spectrometer. Wavelength shift/stretch augmenters exist but aren't reused by synthesis.

**Proposed: `InstrumentalBroadeningAugmenter(SpectraTransformerMixin)`**
```
X_aug = gaussian_filter1d(X, sigma_pts)
sigma_pts = FWHM / (2 * sqrt(2 * ln(2))) / wavelength_step
```

#### E. Noise / Signal Quality

| Phenomenon | Physical Basis | In Synthesis | In Augmentation |
|-----------|---------------|-------------|----------------|
| Heteroscedastic noise | Shot + thermal: `sigma = base + k*|A|` | `_add_noise()` | **Missing** (only homoscedastic exists) |
| Gaussian noise | Detector thermal noise | (part of above) | `GaussianAdditiveNoise` |
| Multiplicative noise | Gain fluctuations | (part of above) | `MultiplicativeNoise` |
| Spikes | Electronic glitches | `_add_artifacts()` | `SpikeNoise` |
| Dead bands | Detector failures | `_add_artifacts()` | **Missing** |
| Saturation | ADC/detector clipping | `_add_artifacts()` | `LocalClipping` (partial) |

The key missing piece is heteroscedastic noise, which is physically more realistic than uniform Gaussian noise: detector noise scales with signal intensity (photon shot noise).

**Proposed: `HeteroscedasticNoiseAugmenter(SpectraTransformerMixin)`**
```
sigma_i = base_sigma + signal_dependent_sigma * |X_i|
X_aug = X + N(0, sigma_i)
```

### 5.2 Priority Classification

**Already done (synthesis delegates to augmenters):**
- Temperature effects
- Moisture effects
- Particle size scattering
- EMSC distortion
- Detector roll-off, stray light, edge curvature, truncated peaks

**Reuse existing augmenters (update generator to delegate):**
- Polynomial baseline drift -> `PolynomialBaselineDrift`
- Wavelength shift/stretch -> `WavelengthShift`, `WavelengthStretch`
- Linear slope -> `LinearBaselineDrift`
- MSC-like scatter -> `ScatterSimulationMSC` (may need tilt enhancement)

**New augmenters needed:**
- `PathLengthAugmenter` - Simple multiplicative factor
- `BatchEffectAugmenter` - Session drift simulation
- `InstrumentalBroadeningAugmenter` - Gaussian convolution (ILS)
- `HeteroscedasticNoiseAugmenter` - Signal-dependent noise
- `DeadBandAugmenter` - Spectral region dropout

**Possibly synthesis-only (don't transfer):**
- Multi-sensor stitching (too complex for augmentation, requires detector config)
- Multi-scan averaging (this is noise *reduction*, not augmentation)
- Concentration generation (this IS the data, not a transformation)

### 5.3 Wavelength-Awareness Decision

New augmenters should be wavelength-aware (`SpectraTransformerMixin`) when the effect depends on the wavelength axis:
- `InstrumentalBroadeningAugmenter`: Yes (FWHM depends on wavelength step)
- `BatchEffectAugmenter`: Yes (offset is wavelength-dependent polynomial)
- `HeteroscedasticNoiseAugmenter`: Optional (noise can be signal-dependent without wavelength)
- `PathLengthAugmenter`: No (pure multiplicative, wavelength-independent)
- `DeadBandAugmenter`: Optional (can use index-based or wavelength-based regions)

### 5.4 Existing Augmenters to Enhance

Some existing augmenters need enhancement to match synthesis capabilities:

1. **`PolynomialBaselineDrift`**: Currently uses `lambda_axis` constructor param instead of `SpectraTransformerMixin`. Should be migrated to wavelength-aware pattern for consistency.

2. **`LinearBaselineDrift`**: Same issue - uses `lambda_axis` param. Should be `SpectraTransformerMixin`.

3. **`ScatterSimulationMSC`**: Missing wavelength-dependent tilt component that synthesis has. Add tilt parameter.

4. **`GaussianAdditiveNoise`**: Could add optional signal-dependent mode.

5. **`SpikeNoise`**: Check feature parity with synthesis artifact generation.

---

## 6. Detailed Review and Open Questions

### 6.1 Architectural Questions

**Q1: Should the generator become a pure composition of augmenters?**

Currently, the generator has a mix of:
- Inline transformations (steps 3-10, 18)
- Delegated transformations (steps 13-17)

The roadmap suggests transferring ALL mechanisms to augmenters. But some operations (Beer-Lambert, concentration generation) are fundamentally *generative*, not *augmentative*. The generator would become:
1. Generate pure spectra via Beer-Lambert (generative)
2. Apply augmenters sequentially (transformative)

This is a clean separation. The remaining question is whether the generator should instantiate augmenters internally or receive them as a list.

**Q2: Should baseline/slope augmenters become wavelength-aware?**

`PolynomialBaselineDrift` and `LinearBaselineDrift` currently use a `lambda_axis` constructor parameter. Migrating them to `SpectraTransformerMixin` would:
- **Pro**: Consistent API, automatic wavelength injection from dataset
- **Con**: Breaking change for users who construct them without wavelengths
- **Compromise**: Set `_requires_wavelengths = "optional"` (already supported via False default)

**Q3: Should we unify `Augmenter` and `SpectraTransformerMixin` base classes?**

Currently there are two hierarchies:
- `Augmenter(TransformerMixin, BaseEstimator)` with `augment(X)` abstract method
- `SpectraTransformerMixin(TransformerMixin, BaseEstimator)` with `transform_with_wavelengths(X, wl)` abstract method

This creates confusion. Options:
- **Option A**: Keep separate (current approach) - simple augmenters don't need wavelength mechanism
- **Option B**: Make all augmenters inherit `SpectraTransformerMixin` with `_requires_wavelengths = False` - unified but heavier
- **Option C**: Create `AugmenterMixin` that optionally composes with wavelength support

**Q4: How should the webapp handle new augmenters?**

The webapp's `SpectraSynthesis` page calls synthesis directly. New augmenters would need:
- Node definitions in `nirs4all-webapp/src/data/nodes/definitions/`
- Parameter UI renderers
- Documentation/tooltips

### 6.2 Design Decisions Required

**D1: Package structure for externalized synthesis**

Option A (flat):
```
nirs4all/synthesis/
    __init__.py
    generator.py
    builder.py
    ...
```

Option B (grouped):
```
nirs4all/synthesis/
    core/           (generator, builder, config)
    components/     (bands, library, procedural)
    effects/        (environmental, scattering, instruments)
    products/       (templates, categories)
    inference/      (fitter, reconstruction/)
    export/         (exporter, validation)
```

Option A is simpler and matches the current structure. Option B provides better organization for the 30+ files but adds navigational overhead.

>> ANSWER: Option B

**D2: Backward compatibility for `data.synthetic` imports**

Options:
- **Hard break**: Remove `data.synthetic`, update all imports -> cleanest but disruptive
- **Deprecation alias**: Keep `data.synthetic.__init__.py` that re-exports from `synthesis` with deprecation warnings -> safe but adds maintenance
- **No alias**: Only the API path `nirs4all.generate` remains -> acceptable if no one imports from `data.synthetic` directly

Since the `CLAUDE.md` says "Never keep dead code, obsolete code or deprecated code", a hard break seems appropriate for this codebase's philosophy.

>> ANSWER: Hard Break

**D3: Which effects remain synthesis-only?**

Multi-sensor stitching and multi-scan averaging are complex instrumental operations that don't naturally fit the "augment existing data" paradigm. They're more about simulating how instruments *acquire* data than how data *varies*. These could remain synthesis-only without violating the design intent.

>> ANSWER: Keep them synthesis-only

### 6.3 Migration Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| Move `synthesis/` -> `synthesis/` | High: breaks all imports | Sed-based bulk rename, test coverage |
| Generator uses augmenters for baseline/slope | Low: internal change | Unit tests for equivalence |
| New augmenters | Low: additive change | Standard testing |
| Migrate baseline augmenters to SpectraTransformerMixin | Medium: API change | `_requires_wavelengths = False` default |
| Webapp node registry updates | Medium: requires UI work | Validate node registry after changes |

### 6.4 Testing Strategy

1. **Equivalence tests**: Verify that generator output is statistically identical after switching to augmenter delegation
2. **Augmenter unit tests**: Each new augmenter needs standalone tests
3. **Integration tests**: Run full `nirs4all.generate()` pipeline after changes
4. **Webapp tests**: Run `npm run validate:nodes` after node registry updates
5. **Example validation**: Run `./run.sh -q` to verify examples still work

---

## 7. Point of View

### 7.1 Assessment of Current Architecture

The current architecture reveals a well-thought-out but partially-completed design evolution. The synthesis module has grown organically into a comprehensive physics simulator (30+ files, 48+ spectral components, 7 generation phases). Its placement under `data/` was likely a pragmatic early choice ("synthetic data is data"), but the module's scope has far outgrown that framing.

The most interesting architectural insight is that the synthesis team already recognized the value of reusable physical operators: steps 13-17 delegate to augmentation operators. This partial delegation shows the intent, but leaves the generator in a hybrid state where some transformations are inline methods and others are operator instances. This inconsistency makes the generator harder to maintain and prevents users from accessing all synthesis-quality physical simulations as standalone augmenters.

### 7.2 Recommended Approach

I recommend executing both tasks in a specific order:

**Phase 1: Transfer mechanisms first (Task 2)**

Before moving the module, complete the augmenter transfer:

1. Create new wavelength-aware augmenters: `PathLengthAugmenter`, `BatchEffectAugmenter`, `InstrumentalBroadeningAugmenter`, `HeteroscedasticNoiseAugmenter`
2. Enhance existing augmenters: add tilt to `ScatterSimulationMSC`, ensure parameter parity
3. Refactor `generator.py` to delegate all effect-application steps to augmenters
4. Keep Beer-Lambert, concentration generation, and component library as pure generative logic

The generator would become:
```python
class SyntheticNIRSGenerator:
    def generate(self, n_samples):
        # Generative steps (pure synthesis, no operator equivalent)
        C = self.generate_concentrations(n_samples)
        A = self._apply_beer_lambert(C)

        # All effect steps use augmenters
        for augmenter in self._effect_chain:
            if isinstance(augmenter, SpectraTransformerMixin):
                A = augmenter.transform(A, wavelengths=self.wavelengths)
            else:
                A = augmenter.transform(A)

        return A
```

The dual dispatch reflects the two base class hierarchies (`Augmenter` with `transform(X)` vs `SpectraTransformerMixin` with `transform(X, wavelengths)`). This makes the generator a clean orchestrator that composes well-defined operators.

**Phase 2: Externalize module (Task 1)**

After the transfer is complete and tested:

1. Create `nirs4all/synthesis/` with the current flat structure (Option A from D1)
2. Move all files, update imports with a bulk rename
3. No backward compatibility alias (per codebase philosophy)
4. Update tests, examples, webapp, documentation

This order is safer because:
- The augmenter transfer can be tested without breaking any existing imports
- Moving the module is a mechanical operation once the code is stable
- Users benefit immediately from new augmenters even before the module moves

### 7.3 On the Base Class Question

The dual-hierarchy (`Augmenter` vs `SpectraTransformerMixin`) is the most significant design issue uncovered. My recommendation: **do not unify them now**.

The `Augmenter` base class provides `apply_on` semantics ("samples"/"global"/"features") that `SpectraTransformerMixin` doesn't need. Forcing all augmenters into `SpectraTransformerMixin` would add unnecessary complexity to simple noise/distortion operators.

Instead, gradually migrate augmenters that would benefit from wavelength information:
- Baseline augmenters (`PolynomialBaselineDrift`, `LinearBaselineDrift`) should become `SpectraTransformerMixin` with `_requires_wavelengths = False`
- Simple augmenters (`GaussianAdditiveNoise`, `MultiplicativeNoise`, `MixupAugmenter`) should stay as `Augmenter`

This pragmatic approach avoids a large refactoring while improving consistency where it matters.

### 7.4 On Multi-Sensor and Multi-Scan

Multi-sensor stitching and multi-scan averaging should remain synthesis-only. These are *acquisition* processes, not *perturbation* processes:
- Stitching creates new data from multiple detector reads (constructive)
- Averaging reduces noise by combining repeated measurements (constructive)
- Augmentation applies perturbation to existing data (destructive/additive)

Forcing these into the augmentation framework would distort the concept of "augmentation."

>> ANSWER: Totally right

### 7.5 Impact on the Broader Roadmap

This work directly enables several downstream roadmap items:
- **"Create example synth pipelines x synth datasets"**: Clean augmenters make it easy to compose realistic effect chains
- **"[Pipeline] as a GridSearchCV or FineTuner"**: Augmenters as searchable parameters in generation
- **"[Generator] add in-place/internal generation > branches"**: Generator as orchestrator of augmenters makes this natural
- **"[Operators] Reintroduce operators tests"**: New augmenters need tests, providing test infrastructure

The externalization also establishes the pattern for other potential future extractions (e.g., if `data/detection/` grows enough to warrant its own module).

### 7.6 Summary

| Item | Recommendation |
|------|---------------|
| Execution order | Transfer mechanisms first, then externalize |
| New augmenters needed | 4-5 (PathLength, BatchEffect, InstrumentalBroadening, HeteroscedasticNoise, DeadBand) |
| Existing augmenters to enhance | 3-4 (ScatterSimulationMSC, PolynomialBaselineDrift, LinearBaselineDrift) |
| Base class unification | Not now; gradual migration |
| Backward compatibility | Hard break (no alias) |
| Module structure | Flat (match current structure) |
| Multi-sensor/multi-scan | Remain synthesis-only |
| Generator refactoring | Become orchestrator of augmenter chain |
