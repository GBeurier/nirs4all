# `sample_augmentation`

`sample_augmentation` creates extra training samples. It runs during training and is skipped during prediction.

## DAG-ML execution

With `engine="dag-ml"`, augmentation trains on original and synthetic training rows. Validation and test scores still cover the original rows only. Without a cross-validator, DAG-ML fits once and reports no CV score. With a cross-validator, stateless augmenters can generate rows before splitting; balanced or data-dependent augmenters generate separate rows within each fold's training partition and during refit. Repetition datasets retain group-aware folds.

Consecutive augmentation steps work with or without CV. Augmentation steps can also be separated by preprocessing steps. For balanced or data-dependent augmentation, each CV fold materializes its own ordered augmentation and preprocessing chain, including transformed validation spectra; refit uses the full training partition. The fitted refit preprocessing is replayed when the result is exported. `exclude` can appear before augmentation, or after a preprocessing step following stateless augmentation. Supported branch combinations include duplication branches merged as features, mean fusion of regression models, and `by_metadata` separation with `merge: concat`.

An exported `by_metadata` separation model needs the same metadata column at prediction time. For example, pass `{"X": X_new, "metadata": {"group": group_values}}` to `nirs4all.predict`; a bare feature matrix does not identify which branch model applies to each row.

Balanced augmentation is fit within each DAG-ML training fold. Legacy nirs4all performs augmentation before creating CV folds, so its CV score may differ even when the final refit and test predictions agree.

An `exclude` between fold-local augmentation stages, or after a transform following fold-local augmentation, still requires a fold-specific exclusion view and is rejected explicitly.

## Simple YAML

```yaml
pipeline:
  - sample_augmentation:
      class: nirs4all.operators.augmentation.GaussianAdditiveNoise
      params:
        sigma: 0.01

  - model:
      class: sklearn.cross_decomposition.PLSRegression
```

Sequential augmenters:

```yaml
pipeline:
  - sample_augmentation:
      - class: nirs4all.operators.augmentation.GaussianAdditiveNoise
        params:
          sigma: 0.01
      - class: nirs4all.operators.augmentation.WavelengthShift
        params:
          shift_range: [-1.0, 1.0]
```

Advanced dictionary form:

```yaml
pipeline:
  - sample_augmentation:
      transformers:
        - class: nirs4all.operators.augmentation.GaussianAdditiveNoise
          params:
            sigma: 0.01
      count: 3
      selection: random
```

## JSON

```json
{
  "pipeline": [
    {
      "sample_augmentation": {
        "class": "nirs4all.operators.augmentation.GaussianAdditiveNoise",
        "params": {
          "sigma": 0.01
        }
      }
    }
  ]
}
```

## Python

```python
from nirs4all.operators.augmentation import GaussianAdditiveNoise, WavelengthShift

pipeline = [
    {"sample_augmentation": [
        GaussianAdditiveNoise(sigma=0.01),
        WavelengthShift(shift_range=(-1.0, 1.0)),
    ]},
    {"model": model},
]
```

## Common Augmenters

| Family | Examples |
| --- | --- |
| Noise/drift | `GaussianAdditiveNoise`, `MultiplicativeNoise`, `SpikeNoise`, `LinearBaselineDrift`, `PolynomialBaselineDrift` |
| Wavelength | `WavelengthShift`, `WavelengthStretch`, `LocalWavelengthWarp` |
| Spectral masking/warping | `BandPerturbation`, `BandMasking`, `ChannelDropout`, `SmoothMagnitudeWarp`, `LocalClipping` |
| Mixup | `MixupAugmenter`, `LocalMixupAugmenter` |
| Physical/instrument | `PathLengthAugmenter`, `BatchEffectAugmenter`, `InstrumentalBroadeningAugmenter`, `DeadBandAugmenter` |

See {doc}`/reference/augmentations`.
