# Synthetic Spectra Generator: Integration Inventory

This document provides a comprehensive inventory of the transformations, noises, and effects used by the synthetic NIRS spectra generator, with an analysis of what can be integrated into the nirs4all operators API, what is already integrated, and what remains to be done.

## Executive Summary

The synthetic generator (`nirs4all.synthesis`) implements ~35 distinct spectral transformations organized into:
- **Scattering effects** (5 classes) - NOT yet TransformerMixin
- **Environmental effects** (3 classes) - NOT yet TransformerMixin
- **Augmentation operators** (20+ classes) - Already TransformerMixin-compliant
- **Instrument effects** (1 simulator) - Complex, likely stays standalone

**Integration Status:**
| Category | Total Methods | Already Integrated | Can Be Integrated | Cannot Be Integrated |
|----------|---------------|-------------------|-------------------|---------------------|
| Scattering | 5 | 1 (partial) | 4 | 0 |
| Environmental | 3 | 0 | 3 | 0 |
| Augmentation | 20+ | 20+ | 0 | 0 |
| Instrument | 6+ | 0 | 2 | 4 |

---

## 1. Augmentation Operators (Already Integrated)

**Location:** `nirs4all/operators/augmentation/`

These are **fully TransformerMixin-compliant** and usable in nirs4all pipelines. They inherit from `Augmenter(TransformerMixin, BaseEstimator)`.

### Base Class
- **Augmenter** ([abc_augmenter.py](nirs4all/operators/augmentation/abc_augmenter.py)) - Abstract base providing sklearn compatibility
  - Parameters: `apply_on` ("samples", "features", "subsets", "global"), `random_state`, `copy`
  - Methods: `fit()`, `transform()`, `fit_transform()`, `augment()` (abstract)

### Noise and Distortion ([spectral.py](nirs4all/operators/augmentation/spectral.py))

| Class | Effect | Formula | Parameters |
|-------|--------|---------|------------|
| `GaussianAdditiveNoise` | Additive Gaussian noise | `X + N(0, σ)` | `sigma`, `smoothing_kernel_width` |
| `MultiplicativeNoise` | Random gain | `X * (1 + ε)` | `sigma_gain`, `per_wavelength` |

### Baseline Transformations

| Class | Effect | Formula | Parameters |
|-------|--------|---------|------------|
| `LinearBaselineDrift` | Linear baseline | `X + a + b*λ` | `offset_range`, `slope_range` |
| `PolynomialBaselineDrift` | Polynomial baseline | `X + Σ(cᵢ*λⁱ)` | `degree`, `coeff_ranges` |

### Wavelength Axis Distortions

| Class | Effect | Description | Parameters |
|-------|--------|-------------|------------|
| `WavelengthShift` | Spectral shift | Interpolate at λ + shift | `shift_range` |
| `WavelengthStretch` | Scale wavelength axis | Compress/expand | `stretch_range` |
| `LocalWavelengthWarp` | Non-linear warp | Spline-based distortion | `n_control_points`, `max_shift` |
| `SmoothMagnitudeWarp` | Smooth gain curve | `X * gain(λ)` | `n_control_points`, `gain_range` |

### Band Perturbation and Smoothing

| Class | Effect | Description | Parameters |
|-------|--------|-------------|------------|
| `BandPerturbation` | Local band modification | Gain + offset specific bands | `n_bands`, `bandwidth_range`, `gain_range`, `offset_range` |
| `GaussianSmoothingJitter` | Variable smoothing | Gaussian blur with random σ | `sigma_range`, `kernel_width` |
| `UnsharpSpectralMask` | Sharpening | `X + k*(X - smooth(X))` | `amount_range`, `sigma` |

### Spectral Masking and Dropout

| Class | Effect | Description | Parameters |
|-------|--------|-------------|------------|
| `BandMasking` | Mask spectral bands | Zero or interpolate | `n_bands_range`, `bandwidth_range`, `mode` |
| `ChannelDropout` | Drop wavelengths | Zero or interpolate | `dropout_prob`, `mode` |

### Artifacts

| Class | Effect | Description | Parameters |
|-------|--------|-------------|------------|
| `SpikeNoise` | Sharp peaks | Add spikes | `n_spikes_range`, `amplitude_range` |
| `LocalClipping` | Saturation | Clip local regions | `n_regions`, `width_range` |

### Sample Combinations

| Class | Effect | Description | Parameters |
|-------|--------|-------------|------------|
| `MixupAugmenter` | Mixup augmentation | `λ*X₁ + (1-λ)*X₂` | `alpha` |
| `LocalMixupAugmenter` | Neighbor mixup | Mixup with k-NN | `alpha`, `k_neighbors` |

### Scattering Simulation

| Class | Effect | Description | Parameters |
|-------|--------|-------------|------------|
| `ScatterSimulationMSC` | MSC-style scatter | `a + b*X` | `a_range`, `b_range`, `reference_mode` |

**Status:** ✅ Fully integrated. All 20+ classes are usable in nirs4all pipelines.

---

## 2. Scattering Effects (NOT Integrated)

**Location:** `nirs4all/synthesis/scattering.py`

These are standalone classes used internally by the generator. They are **NOT TransformerMixin** and cannot be used in pipelines.

### Classes and Their Functions

| Class | Purpose | Physics Model | Pipeline-Compatible? |
|-------|---------|---------------|---------------------|
| `ParticleSizeSimulator` | Particle size effects on scattering | Wavelength-dependent baseline (`λ^(-exp)`), path length effects | **No** |
| `EMSCTransformSimulator` | EMSC-style scatter distortions | `x = a + b*m + d*λ + e*λ² + ...` | **No** |
| `ScatteringCoefficientGenerator` | Kubelka-Munk S(λ) | `S(λ) ∝ λ^(-exp)` | **No** |
| `ScatteringEffectsSimulator` | Combined simulator | Combines particle + EMSC | **No** |

### Scattering Models Implemented

```python
class ScatteringModel(Enum):
    EMSC = "emsc"               # Extended Multiplicative Scatter Correction style
    RAYLEIGH = "rayleigh"       # Rayleigh-like (λ⁻⁴ dependence)
    MIE_APPROX = "mie_approx"   # Simplified Mie approximation
    KUBELKA_MUNK = "kubelka_munk"  # K-M scattering coefficient model
    POLYNOMIAL = "polynomial"   # Polynomial baseline scattering
```

### Convenience Functions

These are simple wrappers that could be converted to TransformerMixin:

| Function | Effect | Can Be Integrated? |
|----------|--------|-------------------|
| `apply_particle_size_effects()` | Apply particle size effects | ✅ Yes |
| `apply_emsc_distortion()` | Apply EMSC-style distortion | ✅ Yes |
| `simulate_snv_correctable_scatter()` | SNV-correctable scatter | ✅ Yes |
| `simulate_msc_correctable_scatter()` | MSC-correctable scatter | ✅ Yes |

### Integration Analysis

**Can be integrated as TransformerMixin:**

1. **ParticleSizeEffect** (new)
   - Wraps `ParticleSizeSimulator` logic
   - Parameters: `mean_size_um`, `std_size_um`, `wavelength_exponent`, `size_effect_strength`
   - Note: Requires wavelengths, which isn't standard for TransformerMixin

2. **EMSCDistortion** (new)
   - Wraps `EMSCTransformSimulator` logic
   - Parameters: `multiplicative_std`, `additive_std`, `polynomial_order`
   - Similar to existing `ScatterSimulationMSC` but more physics-based

3. **SNVCorrectableScatter** (new)
   - Simple multiplicative + additive scatter
   - Parameters: `intensity`

4. **MSCCorrectableScatter** (new)
   - Simple a + b*X transformation
   - Similar to existing `ScatterSimulationMSC`

**Challenges:**
- Some effects require `wavelengths` array, which standard sklearn transformers don't have access to
- Solution: Pass wavelengths via `fit()` or store in constructor

---

## 3. Environmental Effects (NOT Integrated)

**Location:** `nirs4all/synthesis/environmental.py`

These simulate temperature and moisture effects on spectra. They are **NOT TransformerMixin** and cannot be used in pipelines.

### Classes

| Class | Purpose | Physics | Pipeline-Compatible? |
|-------|---------|---------|---------------------|
| `TemperatureEffectSimulator` | Temperature-induced changes | Peak shifts, intensity changes, broadening | **No** |
| `MoistureEffectSimulator` | Water activity effects | Free/bound water band shifts | **No** |
| `EnvironmentalEffectsSimulator` | Combined | Temperature + moisture | **No** |

### Temperature Effects by Spectral Region

The simulator uses literature-based parameters for different NIR regions:

| Region | Wavelength (nm) | Shift/°C (nm) | Intensity Change/°C | Broadening/°C |
|--------|-----------------|---------------|---------------------|---------------|
| O-H 1st overtone | 1400-1520 | -0.30 | -0.2% | 0.1% |
| O-H combination | 1900-2000 | -0.40 | -0.3% | 0.12% |
| C-H 1st overtone | 1650-1780 | -0.05 | -0.05% | 0.08% |
| N-H 1st overtone | 1490-1560 | -0.20 | -0.15% | 0.1% |
| Water (free) | 1380-1420 | -0.10 | +0.3% | 0.08% |
| Water (bound) | 1440-1500 | -0.35 | -0.4% | 0.15% |

### Convenience Functions

| Function | Can Be Integrated? |
|----------|-------------------|
| `apply_temperature_effects()` | ✅ Yes |
| `apply_moisture_effects()` | ✅ Yes |
| `simulate_temperature_series()` | ✅ Yes (generates multiple samples) |

### Integration Analysis

**Can be integrated as TransformerMixin:**

1. **TemperatureEffect** (new)
   - Parameters: `temperature`, `reference_temperature`, `temperature_variation`, `enable_shift`, `enable_intensity`, `enable_broadening`
   - Effect: Apply region-specific temperature-induced spectral changes
   - Challenge: Requires wavelengths array

2. **MoistureEffect** (new)
   - Parameters: `water_activity`, `moisture_content`, `free_water_fraction`
   - Effect: Shift water bands based on free/bound water ratio
   - Challenge: Requires wavelengths array

3. **EnvironmentalEffect** (new, combined)
   - Combines temperature + moisture in correct order

**Why These Are Valuable for Pipelines:**
- Train models robust to temperature variations (critical for handheld/field instruments)
- Simulate moisture-induced spectral changes for food/agricultural applications
- Data augmentation for domain adaptation

---

## 4. Instrument Effects (Partially Integrable)

**Location:** `nirs4all/synthesis/instruments.py`

These simulate instrument-specific effects. Complex simulators are NOT suitable for TransformerMixin, but some simple effects can be extracted.

### InstrumentSimulator Effects

| Effect | Method | Integrable as Augmenter? |
|--------|--------|-------------------------|
| Spectral resolution (broadening) | `_apply_instrumental_broadening()` | ✅ Yes |
| Wavelength calibration errors | `_apply_wavelength_effects()` | ✅ Yes (similar to `WavelengthShift`) |
| Multi-sensor stitching | `_apply_multi_sensor_effects()` | ❌ No (too complex) |
| Multi-scan averaging | `_apply_multi_scan_averaging()` | ❌ No (generates multiple then averages) |
| Detector noise | `_apply_detector_noise()` | ✅ Partial (shot, thermal, 1/f noise) |
| Stray light | `_apply_stray_light()` | ✅ Yes (simple offset) |
| Photometric range | `_apply_photometric_range()` | ✅ Yes (clipping) |

### Can Be Integrated

1. **SpectralBroadening** (new)
   - Effect: Apply Gaussian convolution to simulate instrument resolution
   - Parameters: `fwhm_nm`, `wavelengths`

2. **DetectorNoise** (new)
   - Effect: Add realistic detector noise (shot + thermal + read + 1/f)
   - Parameters: `shot_factor`, `thermal_factor`, `flicker_factor`, `base_noise`

3. **StrayLight** (new)
   - Effect: Add stray light offset
   - Parameters: `stray_level`, `variation`

### Cannot Be Integrated Easily

- **Multi-sensor stitching**: Too complex, involves multiple detector ranges with different characteristics
- **Multi-scan averaging**: Generates internal samples then averages - not a simple transform

---

## 5. Other Generator Components

### Component Library (`components.py`)

Pure generation, NOT transformation:
- `NIRBand` - Voigt profile peak shapes
- `SpectralComponent` - Collection of bands representing a chemical component
- `ComponentLibrary` - Library of components

**Status:** Not integrable as operators (used for spectrum generation, not transformation).

### Procedural Generation (`procedural.py`)

Physics-based component generation:
- Overtone calculations with anharmonicity
- Combination band relationships
- Functional group libraries (hydroxyl, amine, methyl, etc.)

**Status:** Not integrable as operators (generation, not transformation).

### Detectors (`detectors.py`)

Detector response simulation:
- Response curves for different detector types
- Noise models

**Status:** Partially integrable (noise models could become augmenters).

---

## 6. Overlap Analysis: Generator vs Operators

| Effect | In Generator | In Operators | Notes |
|--------|-------------|--------------|-------|
| Gaussian noise | Implicit in detector sim | `GaussianAdditiveNoise` | ✅ Covered |
| Multiplicative scatter | EMSC simulator | `MultiplicativeNoise`, `ScatterSimulationMSC` | ✅ Covered |
| Baseline drift | Generator adds polynomial | `LinearBaselineDrift`, `PolynomialBaselineDrift` | ✅ Covered |
| Wavelength shift | Instrument calibration | `WavelengthShift` | ✅ Covered |
| Wavelength stretch | - | `WavelengthStretch` | ✅ Covered |
| Particle size effects | `ParticleSizeSimulator` | - | ❌ **Gap** |
| EMSC distortion | `EMSCTransformSimulator` | `ScatterSimulationMSC` (simpler) | ⚠️ Partial |
| Temperature effects | `TemperatureEffectSimulator` | - | ❌ **Gap** |
| Moisture effects | `MoistureEffectSimulator` | - | ❌ **Gap** |
| Spectral broadening | Instrument sim | `GaussianSmoothingJitter` (different purpose) | ⚠️ Partial |
| Shot noise | Detector sim | - | ❌ **Gap** |
| 1/f noise | Detector sim | - | ❌ **Gap** |

---

## 7. Integration Recommendations

### Priority 1: High Value, Low Effort

These can be created by wrapping existing logic:

1. **TemperatureAugmenter**
   ```python
   class TemperatureAugmenter(Augmenter):
       def __init__(self, temperature_range=(-5, 5), reference_temperature=25,
                    wavelengths=None, random_state=None):
           ...
   ```
   - Requires: Passing wavelengths at construction or fitting

2. **MoistureAugmenter**
   ```python
   class MoistureAugmenter(Augmenter):
       def __init__(self, water_activity_range=(0.3, 0.8),
                    moisture_content_range=(0.05, 0.20), wavelengths=None, random_state=None):
           ...
   ```

3. **ParticleSizeAugmenter**
   ```python
   class ParticleSizeAugmenter(Augmenter):
       def __init__(self, mean_size_um=50, size_variation_um=15,
                    wavelengths=None, random_state=None):
           ...
   ```

### Priority 2: Medium Value, Medium Effort

4. **EMSCDistortionAugmenter** - More physics-based than `ScatterSimulationMSC`
5. **DetectorNoiseAugmenter** - Realistic shot/thermal/1/f noise
6. **SpectralResolutionAugmenter** - Simulate different instrument resolutions

### Priority 3: Architecture Considerations

**Wavelength Handling:**
The main challenge is that many effects require wavelength information. Options:

A. **Pass wavelengths at construction** (recommended)
   ```python
   aug = TemperatureAugmenter(wavelengths=np.arange(900, 2500, 2))
   ```

B. **Extract from SpectroDataset metadata**
   ```python
   def fit(self, X, y=None, dataset=None):
       if dataset is not None:
           self.wavelengths = dataset.wavelengths
   ```

C. **Infer from X shape** (least accurate)
   ```python
   def fit(self, X, y=None):
       n_features = X.shape[1]
       self.wavelengths = np.linspace(900, 2500, n_features)
   ```

### Alternative: Make Generator Use Operators

Instead of extracting operators from the generator, the generator could be refactored to USE operators:

```python
# Current approach (internal simulation)
class SyntheticNIRSGenerator:
    def _apply_environmental_effects(self, spectra, wavelengths):
        # Internal implementation
        ...

# Proposed approach (use operators)
class SyntheticNIRSGenerator:
    def __init__(self, ...):
        self.augmentation_pipeline = [
            TemperatureAugmenter(temperature_range=(-5, 5), wavelengths=self.wavelengths),
            MoistureAugmenter(water_activity_range=(0.3, 0.8), wavelengths=self.wavelengths),
            ParticleSizeAugmenter(mean_size_um=50, wavelengths=self.wavelengths),
        ]

    def generate(self, n_samples):
        spectra = self._generate_base_spectra(n_samples)
        for aug in self.augmentation_pipeline:
            spectra = aug.transform(spectra)
        return spectra
```

---

## 8. Work Estimation

### New Augmenters to Create

| Augmenter | Effort | Lines of Code | Dependencies |
|-----------|--------|---------------|--------------|
| TemperatureAugmenter | Medium | ~150 | scipy.ndimage |
| MoistureAugmenter | Medium | ~100 | numpy |
| ParticleSizeAugmenter | Medium | ~120 | scipy.ndimage |
| EMSCDistortionAugmenter | Low | ~80 | numpy |
| DetectorNoiseAugmenter | Medium | ~100 | numpy |
| SpectralResolutionAugmenter | Low | ~50 | scipy.ndimage |

**Total estimated new code:** ~600 lines

### Refactoring Required

| Task | Effort |
|------|--------|
| Add wavelength handling to Augmenter base class | Low |
| Create tests for new augmenters | Medium |
| Update generator to optionally use operators | High |
| Documentation | Medium |

---

## 9. Conclusion

### What's Already Integrated
- All basic augmentation operators (20+ classes) are TransformerMixin-compliant
- Basic scatter simulation (`ScatterSimulationMSC`)

### What Can Be Integrated
- Temperature effects (high value for instrument robustness)
- Moisture effects (high value for food/agriculture)
- Particle size effects (high value for powder samples)
- Advanced scatter simulation (EMSC-style)
- Detector noise models

### What Cannot Be Integrated
- Component generation (NIRBand, SpectralComponent) - these CREATE spectra, don't transform
- Multi-sensor stitching - too complex for simple transformer pattern
- Full instrument simulation - requires too much state and configuration

### Recommended Path Forward

1. **Phase 1:** Create `TemperatureAugmenter`, `MoistureAugmenter`, `ParticleSizeAugmenter`
2. **Phase 2:** Add wavelength-aware base class or protocol
3. **Phase 3:** Refactor generator to optionally use operator pipeline
4. **Phase 4:** Add detector noise augmenters

This approach allows:
- Using effects in training pipelines for data augmentation
- Maintaining the generator as a standalone tool for synthetic data creation
- Progressive integration without breaking changes
