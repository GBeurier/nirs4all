# `sample_augmentation`

`sample_augmentation` creates extra training samples. It runs during training and is skipped during prediction.

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
