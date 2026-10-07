# Augmentation Operators Reference

An augmentation operator perturbs spectra. The sample-augmentation node creates origin-linked training copies; the feature-augmentation node creates representations. Choose the physical change you expect at deployment before selecting noise, drift, wavelength warp or masking. Check that the target remains valid.

## Choose a perturbation that resembles real acquisition variability

Read {doc}`nodes/sample_augmentation` for the worked result figure, expected dimensions and exercises. Choose a family below to compare the enumerated operators.

## Same worked recipe in JSON, YAML and Python

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "pipeline": [
    {
      "split": {
        "class": "sklearn.model_selection.KFold",
        "params": {
          "n_splits": 3
        }
      }
    },
    {
      "sample_augmentation": {
        "transformers": [
          {
            "class": "nirs4all.operators.augmentation.GaussianAdditiveNoise",
            "params": {
              "sigma": 0.01
            }
          }
        ],
        "count": 2,
        "selection": "random",
        "random_state": 42
      }
    },
    {
      "model": {
        "class": "sklearn.cross_decomposition.PLSRegression",
        "params": {
          "n_components": 2
        }
      }
    }
  ]
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
pipeline:
- split:
    class: sklearn.model_selection.KFold
    params:
      n_splits: 3
- sample_augmentation:
    transformers:
    - class: nirs4all.operators.augmentation.GaussianAdditiveNoise
      params:
        sigma: 0.01
    count: 2
    selection: random
    random_state: 42
- model:
    class: sklearn.cross_decomposition.PLSRegression
    params:
      n_components: 2
```
:::

:::{tab-item} Python
:sync: python

```python
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from nirs4all.operators.augmentation import GaussianAdditiveNoise

pipeline = [
    KFold(n_splits=3),
    {"sample_augmentation": {
        "transformers": [GaussianAdditiveNoise(sigma=0.01)],
        "count": 2, "selection": "random", "random_state": 42,
    }},
    {"model": PLSRegression(n_components=2)},
]
```
:::

::::

These are Python SDK operators. Native R/Octave/WASM recipes use their own method IDs and facade; see {doc}`/guide/languages`.

---

## Noise

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `GaussianAdditiveNoise` | `sigma=0.01`, `smoothing_kernel_width=1`, `variation_scope="sample"` | Adds per-sample Gaussian noise scaled by sample std |
| `MultiplicativeNoise` | `sigma_gain=0.05`, `per_wavelength=False`, `variation_scope="sample"` | Multiplies spectra by random gain factor: X * (1 + epsilon) |
| `SpikeNoise` | `random_state=None` | Adds spike artifacts to random spectral positions |
| `HeteroscedasticNoiseAugmenter` | `random_state=None`, `variation_scope="sample"` | Signal-dependent detector noise (noise scales with signal magnitude) |

---

## Baseline Drift

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `LinearBaselineDrift` | `offset_range=(-0.1, 0.1)`, `slope_range=(-0.001, 0.001)` | Adds a linear baseline drift: X + a + b * lambda |
| `PolynomialBaselineDrift` | `degree=3`, `coeff_ranges=None` | Adds a polynomial baseline drift of given degree |

---

## Wavelength Distortion

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `WavelengthShift` | `shift_range=(-2.0, 2.0)` | Shifts the wavelength axis by a random amount (simulates calibration drift) |
| `WavelengthStretch` | `stretch_range=(0.99, 1.01)` | Stretches or compresses the wavelength axis around its center |
| `LocalWavelengthWarp` | `n_control_points=5`, `max_shift=1.0` | Non-linear warp of the wavelength axis using spline control points |

---

## Spectral Distortion

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `SmoothMagnitudeWarp` | `n_control_points=5`, `gain_range=(0.9, 1.1)` | Multiplies the spectrum by a smooth spline-based gain curve |
| `BandPerturbation` | `n_bands=3`, `bandwidth_range=(5, 20)`, `gain_range=(0.9, 1.1)`, `offset_range=(-0.01, 0.01)` | Perturbs random wavelength bands with gain and offset |
| `GaussianSmoothingJitter` | `sigma_range=(0.5, 2.0)`, `kernel_width=11` | Applies Gaussian smoothing with random sigma per sample |
| `UnsharpSpectralMask` | `amount_range=(0.1, 0.5)`, `sigma=1.0`, `kernel_width=11` | Spectral sharpening via unsharp masking: X + k * (X - smooth(X)) |
| `BandMasking` | `n_bands_range=(1, 3)`, `bandwidth_range=(5, 20)`, `mode="interp"` | Masks random spectral bands; mode: `"zero"` or `"interp"` |
| `ChannelDropout` | `random_state=None` | Randomly zeroes out individual spectral channels |
| `LocalClipping` | `random_state=None` | Clips local regions of the spectrum |

---

## Mixup

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `MixupAugmenter` | `random_state=None` | Blends samples with KNN neighbors (global mixup) |
| `LocalMixupAugmenter` | `random_state=None` | Band-local mixup with KNN neighbors |
| `ScatterSimulationMSC` | `random_state=None` | Simple MSC-style scatter simulation between samples |

---

## Physical / Instrumental

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `PathLengthAugmenter` | `path_length_std=0.05`, `min_path_length=0.5`, `variation_scope="sample"` | Simulates optical path length variation via multiplicative scaling |
| `BatchEffectAugmenter` | `offset_std=0.02`, `slope_std=0.01`, `gain_std=0.03`, `variation_scope="sample"` | Simulates batch/session effects with wavelength-dependent offset and gain |
| `InstrumentalBroadeningAugmenter` | `random_state=None`, `variation_scope="sample"` | Simulates spectral resolution broadening via Gaussian convolution |
| `DeadBandAugmenter` | `random_state=None` | Simulates dead spectral bands from detector saturation or failure |

---

## Environmental

These operators require wavelength information and inherit from `SpectraTransformerMixin`.

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `TemperatureAugmenter` | `temperature_delta=5.0`, `temperature_range=None`, `reference_temperature=25.0`, `enable_shift=True`, `enable_intensity=True`, `enable_broadening=True`, `region_specific=True` | Simulates temperature-induced spectral changes (peak shifts, intensity, broadening) |
| `MoistureAugmenter` | `random_state=None` | Simulates moisture/water activity effects on NIR spectra |

---

## Scattering

These operators require wavelength information.

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `ParticleSizeAugmenter` | `mean_size_um=50.0`, `size_variation_um=15.0`, `size_range_um=None`, `wavelength_exponent=1.5`, `size_effect_strength=0.1`, `include_path_length=True` | Simulates particle size effects on scattering (wavelength-dependent baseline) |
| `EMSCDistortionAugmenter` | `random_state=None` | Applies EMSC-style polynomial scattering distortions |

---

## Edge Artifacts

These operators require wavelength information and simulate detector boundary effects.

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `DetectorRollOffAugmenter` | `detector_model="generic_nir"`, `effect_strength=1.0`, `noise_amplification=0.02`, `include_baseline_distortion=True` | Simulates detector sensitivity roll-off at spectral edges |
| `StrayLightAugmenter` | `random_state=None` | Simulates stray light effects (peak truncation at edges) |
| `EdgeCurvatureAugmenter` | `random_state=None` | Simulates edge curvature/baseline bending from optical aberrations |
| `TruncatedPeakAugmenter` | `random_state=None` | Adds truncated peaks at spectral boundaries |
| `EdgeArtifactsAugmenter` | `random_state=None` | Combined edge artifacts augmenter (applies multiple edge effects) |

Available detector models for `DetectorRollOffAugmenter`:
- `"ingaas_standard"` -- Standard InGaAs (1000-1600 nm optimal)
- `"ingaas_extended"` -- Extended InGaAs (1100-2200 nm optimal)
- `"pbs"` -- PbS detector (1000-2800 nm optimal)
- `"silicon_ccd"` -- Silicon CCD (400-900 nm optimal)
- `"generic_nir"` -- Generic NIR detector (900-1700 nm optimal)

---

## Spline-Based

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `Spline_Smoothing` | `random_state=None` | Applies a smoothing spline to each spectrum |
| `Spline_X_Perturbations` | `random_state=None` | Perturbs the wavelength (x) axis via B-spline control points |
| `Spline_Y_Perturbations` | `random_state=None` | Perturbs the intensity (y) axis via B-spline control points |
| `Spline_X_Simplification` | `random_state=None` | Simplifies the spectrum along the x-axis using spline resampling |
| `Spline_Curve_Simplification` | `random_state=None` | Simplifies the spectrum along the curve length using spline resampling |

---

## Random Geometric

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `Rotate_Translate` | `p_range=2`, `y_factor=3` | Rotation and translation augmentation: adds piecewise linear distortion |
| `Random_X_Operation` | `random_state=None` | Random multiplicative/additive operations on spectra |

---

## See Also

- {doc}`../reference/transforms` -- Preprocessing transforms reference
- {doc}`../reference/pipeline_keywords` -- Pipeline keyword syntax (including `sample_augmentation` and `feature_augmentation`)

## What the perturbations mean

```{figure} /assets/guide/augmentation.svg
:alt: The original synthetic spectrum shares peak positions with versions perturbed by additive noise and by a gain plus offset.

Illustrative acquisition changes. Copies share the original sample's identity
and target assumptions; they do not add independent validation evidence.
```

| Mechanism | Operators | Interpretation and important limit |
|---|---|---|
| Detector noise | GaussianAdditiveNoise, HeteroscedasticNoiseAugmenter | Constant-relative versus signal-dependent uncertainty; too much noise erases weak peaks |
| Gain/scatter | MultiplicativeNoise, PathLengthAugmenter, ScatterSimulationMSC | Alters intensity without moving peaks; may conflict with a concentration target if used beyond plausible acquisition variability |
| Baseline | LinearBaselineDrift, PolynomialBaselineDrift, BatchEffectAugmenter | Adds nuisance offset/slope/curvature; batch-wide perturbations differ from independent sample noise |
| Calibration drift | WavelengthShift, WavelengthStretch, LocalWavelengthWarp | Moves spectral features; shift/scale/warp have different geometry and boundary behavior |
| Local amplitude | SmoothMagnitudeWarp, BandPerturbation | Alters broad envelopes or particular bands; can change the chemical meaning of a band |
| Resolution | GaussianSmoothingJitter, InstrumentalBroadeningAugmenter, UnsharpSpectralMask | Blur or sharpen local detail; broadening is a more direct resolution model than arbitrary sharpening |
| Missing/saturated channels | BandMasking, ChannelDropout, DeadBandAugmenter, LocalClipping, SpikeNoise | Simulates failures rather than an alternative physical sample; zero masking and interpolation produce different signatures |
| Inter-sample mixing | MixupAugmenter, LocalMixupAugmenter | Requires a meaningful mixed-target assumption; nearest-neighbor structure and source origins matter |
| Spline perturbations | Spline_Smoothing, Spline_X_Perturbations, Spline_Y_Perturbations, Spline_X_Simplification, Spline_Curve_Simplification | Changes curve smoothness, axis, amplitude or representation; distinguish displacement from simplification |
| Environmental changes | TemperatureAugmenter, MoistureAugmenter | Models effects associated with temperature/water; those effects may also affect the true target |
| Particle scattering | ParticleSizeAugmenter, EMSCDistortionAugmenter | Empirical wavelength-dependent scattering; not a complete physical scattering simulation |
| Detector edges | DetectorRollOffAugmenter, StrayLightAugmenter, EdgeCurvatureAugmenter, TruncatedPeakAugmenter, EdgeArtifactsAugmenter | Models edge sensitivity, flattening, curvature or truncated peaks; detector range and wavelength units are essential |
| General geometry | Rotate_Translate, Random_X_Operation | Broad numerical transformations; justify their relation to actual instrument variability |

These mechanisms are hypotheses about deployment variability. Use an unchanged
baseline and the same independent-unit folds to test each hypothesis. A visibly
plausible curve does not establish target-label validity or improved accuracy.

## Sample versus feature augmentation

```{figure} /assets/guide/sample_augmentation.svg
:alt: Training copies keep origin identities while validation remains unchanged.

Educational workflow result; read the accompanying explanation for scope and interpretation.
```

Sample augmentation increases training observations; feature augmentation
adds representations of each observation. Prediction must replay the feature
construction needed by the fitted model, while training-only sample generation
is skipped. The container, rather than the augmentation class name alone,
determines workflow semantics. See {doc}`nodes/sample_augmentation` and
{doc}`nodes/feature_augmentation` for layout, counts and replay rules.

Augmented rows must remain with their original in every train/validation split.
If an operator combines several origins, its relation must be accounted for;
arbitrary random splitting after augmentation creates leakage. The reported
sample count should distinguish independent samples from generated rows.

## Diagnose an augmentation experiment

Inspect before/after overlays, peak positions, signal ranges and target
distribution. Check that held-out rows are unchanged and original-to-copy
relations are recorded. Compare per-origin errors, not only errors averaged
over generated rows. If augmentation helps one instrument but harms another,
report the subgroup effect and revisit the assumed perturbation distribution.
For parameter and method details use `nirs4all.operators.augmentation` under
{doc}`/api/modules`.
