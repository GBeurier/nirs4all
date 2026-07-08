# Preprocessing and Transformer Nodes

Preprocessing nodes transform `X`. They are usually sklearn-compatible transformers or NIRS4ALL spectral transforms.

## Syntax Summary

| Syntax | Use when |
| --- | --- |
| serialized operator | Most portable YAML/JSON form. |
| `preprocessing` wrapper | You want to make the intent explicit or group operators. |
| direct transformer | Python-only object pipeline. |
| `force_layout` | A transformer requires a specific 2D/3D layout. |

## YAML

Direct serialized transform:

```yaml
pipeline:
  - class: nirs4all.operators.transforms.StandardNormalVariate
  - class: nirs4all.operators.transforms.SavitzkyGolay
    params:
      window_length: 15
      polyorder: 2
      deriv: 1
```

Explicit `preprocessing` wrapper:

```yaml
pipeline:
  - preprocessing:
      - class: nirs4all.operators.transforms.StandardNormalVariate
      - class: sklearn.preprocessing.StandardScaler
```

## JSON

```json
{
  "pipeline": [
    {
      "class": "nirs4all.operators.transforms.StandardNormalVariate"
    },
    {
      "class": "nirs4all.operators.transforms.SavitzkyGolay",
      "params": {
        "window_length": 15,
        "polyorder": 2,
        "deriv": 1
      }
    }
  ]
}
```

## Python

```python
from nirs4all.operators.transforms import SNV, SavitzkyGolay

pipeline = [
    SNV(),
    SavitzkyGolay(window_length=15, polyorder=2, deriv=1),
]
```

## Common NIRS Transforms

| Family | Examples |
| --- | --- |
| Scatter/normalization | `StandardNormalVariate`, `SNV`, `RobustStandardNormalVariate`, `RNV`, `MultiplicativeScatterCorrection`, `MSC`, `ExtendedMultiplicativeScatterCorrection`, `EMSC` |
| Smoothing/derivatives | `SavitzkyGolay`, `FirstDerivative`, `SecondDerivative`, `NorrisWilliams`, `Gaussian` |
| Baseline | `Baseline`, `Detrend`, `ASLSBaseline`, `AirPLS`, `ArPLS`, `SNIP`, `RollingBall`, `BEADS` |
| Signal conversion | `ToAbsorbance`, `FromAbsorbance`, `ReflectanceToAbsorbance`, `KubelkaMunk`, `SignalTypeConverter` |
| Feature selection | `CARS`, `MCUVE`, `FlexiblePCA`, `FlexibleSVD` |
| Resampling/alignment | `Resampler`, `CropTransformer`, `ResampleTransformer` |

See {doc}`/reference/transforms` and {doc}`/user_guide/preprocessing/index`.
