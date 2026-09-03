# Generator Keywords

Generator keywords expand one pipeline specification into many concrete pipeline variants. They can appear anywhere a value, operator, parameter, or list of steps is expected.

## Quick Reference

| Keyword | Purpose |
| --- | --- |
| `_or_` | Alternatives. |
| `_range_` | Linear numeric range. |
| `_log_range_` | Log-spaced numeric range. |
| `_grid_` | Parameter grid. |
| `_zip_` | Position-wise combination. |
| `_chain_` | Ordered chain construction. |
| `_sample_` | Random sample from a set. |
| `_cartesian_` | Explicit cartesian product. |
| `pick` | Unordered selection size for alternatives. |
| `arrange` | Ordered selection size for alternatives. |
| `then_pick`, `then_arrange` | Secondary selection after first expansion. |
| `count` | Limit/sample number of generated configs. |
| `_seed_`, `_weights_` | Reproducibility and weighted sampling. |
| `_mutex_`, `_requires_`, `_depends_on_`, `_exclude_` | Constraints between choices. |
| `_preset_` | Reference registered presets. |
| `_tags_`, `_metadata_` | Attach annotations to generated choices. |

## Cartesian Preprocessing Search

This is the common answer to "how do I make a cartesian of preprocessing pipelines?"

```yaml
pipeline:
  - _cartesian_:
      scatter:
        _or_:
          - class: nirs4all.operators.transforms.StandardNormalVariate
          - class: nirs4all.operators.transforms.MultiplicativeScatterCorrection
          - null
      derivative:
        _or_:
          - null
          - class: nirs4all.operators.transforms.SavitzkyGolay
            params:
              window_length: 15
              polyorder: 2
              deriv: 1
      scale:
        _or_:
          - class: sklearn.preprocessing.StandardScaler
          - class: sklearn.preprocessing.MinMaxScaler

  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components:
          _range_: [5, 15, 5]
```

This expands all combinations of `scatter x derivative x scale x n_components`.

## Simpler Alternatives

Use `_or_` for one choice:

```yaml
pipeline:
  - _or_:
      - class: nirs4all.operators.transforms.StandardNormalVariate
      - class: nirs4all.operators.transforms.MultiplicativeScatterCorrection
      - class: nirs4all.operators.transforms.FirstDerivative

  - model:
      class: sklearn.cross_decomposition.PLSRegression
```

Use `_range_` for a numeric parameter:

```yaml
pipeline:
  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components:
          _range_: [2, 20, 2]
```

Use `_grid_` for parameter grid syntax:

```yaml
pipeline:
  - model:
      class: sklearn.ensemble.RandomForestRegressor
      params:
        _grid_:
          n_estimators: [100, 300]
          max_depth: [null, 8, 16]
          random_state: [42]
```

Use `_zip_` when values must move together:

```yaml
pipeline:
  - class: nirs4all.operators.transforms.SavitzkyGolay
    params:
      _zip_:
        window_length: [9, 15, 21]
        polyorder: [2, 3, 3]
```

## Pick vs Arrange

Use `pick` when order does not matter:

```yaml
pipeline:
  - concat_transform:
      _or_:
        - class: nirs4all.operators.transforms.StandardNormalVariate
        - class: nirs4all.operators.transforms.MultiplicativeScatterCorrection
        - class: nirs4all.operators.transforms.FirstDerivative
      pick: 2
```

Use `arrange` when order matters:

```yaml
pipeline:
  - preprocessing:
      _or_:
        - class: nirs4all.operators.transforms.StandardNormalVariate
        - class: nirs4all.operators.transforms.SavitzkyGolay
        - class: sklearn.preprocessing.StandardScaler
      arrange: 2
```

## JSON

```json
{
  "pipeline": [
    {
      "_or_": [
        {
          "class": "nirs4all.operators.transforms.StandardNormalVariate"
        },
        {
          "class": "nirs4all.operators.transforms.MultiplicativeScatterCorrection"
        }
      ]
    },
    {
      "model": {
        "class": "sklearn.cross_decomposition.PLSRegression",
        "params": {
          "n_components": {
            "_range_": [2, 10, 2]
          }
        }
      }
    }
  ]
}
```

## Python

```python
pipeline = [
    {"_or_": [
        {"class": "nirs4all.operators.transforms.StandardNormalVariate"},
        {"class": "nirs4all.operators.transforms.MultiplicativeScatterCorrection"},
    ]},
    {"model": {
        "class": "sklearn.cross_decomposition.PLSRegression",
        "params": {"n_components": {"_range_": [2, 10, 2]}},
    }},
]
```

See {doc}`/reference/generator_keywords` for the full generator API and validation helpers.
