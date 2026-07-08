# `residual`

`residual` routes a residual-model operator. It is used for workflows that fit a base model and then train a residual learner on what the base model did not explain.

## YAML

```yaml
pipeline:
  - residual:
      class: nirs4all.operators.models.ResidualModel
      params:
        base_model:
          class: sklearn.cross_decomposition.PLSRegression
          params:
            n_components: 10
        residual_model:
          class: sklearn.ensemble.RandomForestRegressor
          params:
            n_estimators: 200
            random_state: 42
```

## JSON

```json
{
  "pipeline": [
    {
      "residual": {
        "class": "nirs4all.operators.models.ResidualModel",
        "params": {
          "base_model": {
            "class": "sklearn.cross_decomposition.PLSRegression",
            "params": {
              "n_components": 10
            }
          },
          "residual_model": {
            "class": "sklearn.ensemble.RandomForestRegressor",
            "params": {
              "n_estimators": 200,
              "random_state": 42
            }
          }
        }
      }
    }
  ]
}
```

## Python

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from nirs4all.operators.models import ResidualModel

pipeline = [
    {"residual": ResidualModel(
        base_model=PLSRegression(n_components=10),
        residual_model=RandomForestRegressor(n_estimators=200, random_state=42),
    )}
]
```

You can also place a `ResidualModel` inside `model` when you want the model controller to dispatch it as the supervised model.
