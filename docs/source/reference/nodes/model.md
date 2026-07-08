# `model`

`model` marks a supervised estimator step. During cross-validation, the model is fit on the training fold and evaluated on the validation fold. During refit, the selected configuration is trained on the full training partition.

## Syntax Summary

| Syntax | Use when |
| --- | --- |
| `{"model": {"class": "...", "params": {...}}}` | Portable YAML/JSON config. |
| `{"model": "module.ClassName"}` | Portable config with default parameters. |
| `{"model": estimator}` | Python object pipeline. |
| bare estimator object | Python only; auto-detected when it has `predict()`. |

## YAML

```yaml
pipeline:
  - class: sklearn.preprocessing.MinMaxScaler

  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components: 10
```

Multiple models can be generated with `_or_`:

```yaml
pipeline:
  - class: sklearn.preprocessing.StandardScaler
  - model:
      _or_:
        - class: sklearn.cross_decomposition.PLSRegression
          params:
            n_components: 8
        - class: sklearn.ensemble.RandomForestRegressor
          params:
            n_estimators: 200
            random_state: 42
```

## JSON

```json
{
  "pipeline": [
    {
      "class": "sklearn.preprocessing.MinMaxScaler"
    },
    {
      "model": {
        "class": "sklearn.cross_decomposition.PLSRegression",
        "params": {
          "n_components": 10
        }
      }
    }
  ]
}
```

## Python

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import MinMaxScaler

pipeline = [
    MinMaxScaler(),
    {"model": PLSRegression(n_components=10)},
]
```

## Built-in Model Families

NIRS4ALL accepts:

- sklearn regressors/classifiers with `fit()` and `predict()`;
- built-in NIRS models such as `PLSDA`, `AOMPLSRegressor`, `POPPLSRegressor`, `OPLS`, `MBPLS`, `DiPLS`, `SparsePLS`, and `TabPFNNIRSRegressor`;
- optional TensorFlow, PyTorch, JAX, and AutoGluon models when their extras are installed;
- `MetaModel` and `ResidualModel` for stacking/residual workflows.

See {doc}`/reference/models` and {doc}`/reference/operator_catalog`.

## Prediction and Export

After training:

```python
result = nirs4all.run("pipeline.yaml", "dataset.yaml")
best = result.best
result.export("exports/best_model.n4a")
```

Use {doc}`/reference/public_interfaces` for prediction and retraining examples.
