# `auto_transfer_preproc`

`auto_transfer_preproc` runs transfer-preprocessing selection logic. It is an advanced node for workflows where a preprocessing choice is selected from candidate transforms for transfer robustness.

## YAML

```yaml
pipeline:
  - auto_transfer_preproc:
      candidates:
        - class: nirs4all.operators.transforms.StandardNormalVariate
        - class: nirs4all.operators.transforms.MultiplicativeScatterCorrection
        - class: nirs4all.operators.transforms.SavitzkyGolay
          params:
            deriv: 1
      metric: distance_reduction
      use_augmentation: true

  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components: 10
```

## JSON

```json
{
  "pipeline": [
    {
      "auto_transfer_preproc": {
        "candidates": [
          {
            "class": "nirs4all.operators.transforms.StandardNormalVariate"
          },
          {
            "class": "nirs4all.operators.transforms.MultiplicativeScatterCorrection"
          }
        ],
        "metric": "distance_reduction",
        "use_augmentation": true
      }
    }
  ]
}
```

## Python

```python
from nirs4all.operators.transforms import SNV, MSC, SavitzkyGolay

pipeline = [
    {"auto_transfer_preproc": {
        "candidates": [SNV(), MSC(), SavitzkyGolay(deriv=1)],
        "metric": "distance_reduction",
        "use_augmentation": True,
    }},
    {"model": model},
]
```

See {doc}`/user_guide/deployment/retrain_transfer` for transfer workflows.
