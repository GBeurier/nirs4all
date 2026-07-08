# `concat_transform`

`concat_transform` applies several transforms to the same input and concatenates their outputs horizontally into one feature matrix.

Use it when the next model should see one matrix containing several transformed views.

## YAML

```yaml
pipeline:
  - concat_transform:
      - class: nirs4all.operators.transforms.StandardNormalVariate
      - class: nirs4all.operators.transforms.SavitzkyGolay
        params:
          deriv: 1
      - class: sklearn.decomposition.PCA
        params:
          n_components: 20

  - model:
      class: sklearn.linear_model.Ridge
```

Nested concatenation:

```yaml
pipeline:
  - concat_transform:
      - class: nirs4all.operators.transforms.StandardNormalVariate
      - concat_transform:
          - class: sklearn.decomposition.PCA
            params:
              n_components: 20
          - class: sklearn.decomposition.TruncatedSVD
            params:
              n_components: 20
```

## JSON

```json
{
  "pipeline": [
    {
      "concat_transform": [
        {
          "class": "nirs4all.operators.transforms.StandardNormalVariate"
        },
        {
          "class": "sklearn.decomposition.PCA",
          "params": {
            "n_components": 20
          }
        }
      ]
    }
  ]
}
```

## Python

```python
from sklearn.decomposition import PCA
from nirs4all.operators.transforms import SNV, SavitzkyGolay

pipeline = [
    {"concat_transform": [SNV(), SavitzkyGolay(deriv=1), PCA(n_components=20)]},
    {"model": model},
]
```

## Difference from `feature_augmentation`

| Node | Output |
| --- | --- |
| `concat_transform` | One feature matrix with columns concatenated. |
| `feature_augmentation` | Multiple feature views/preprocessing branches tracked by the runtime. |
