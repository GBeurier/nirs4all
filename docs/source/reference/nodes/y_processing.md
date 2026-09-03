# `y_processing`

`y_processing` applies a transformer to the target `y` before model fitting. Predictions are inverse-transformed back to the original target scale when the transformer supports `inverse_transform()`.

## YAML

```yaml
pipeline:
  - y_processing:
      class: sklearn.preprocessing.MinMaxScaler

  - split:
      class: sklearn.model_selection.KFold
      params:
        n_splits: 5

  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components: 10
```

Multiple target transforms:

```yaml
pipeline:
  - y_processing:
      - class: sklearn.preprocessing.StandardScaler
      - class: nirs4all.operators.transforms.RangeDiscretizer
        params:
          n_bins: 5
```

## JSON

```json
{
  "pipeline": [
    {
      "y_processing": {
        "class": "sklearn.preprocessing.MinMaxScaler"
      }
    }
  ]
}
```

## Python

```python
from sklearn.preprocessing import MinMaxScaler

pipeline = [
    {"y_processing": MinMaxScaler()},
    {"model": model},
]
```

## Rules

- Use target transforms that preserve the target semantics required by your model.
- For regression scaling, choose transformers with `inverse_transform()`.
- `IntegerKBinsDiscretizer` and `RangeDiscretizer` are available for target discretization workflows.

See {doc}`/reference/transforms`.
