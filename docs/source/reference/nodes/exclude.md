# `exclude`

`exclude` removes flagged samples from training. Prediction samples are not excluded because NIRS4ALL should return a prediction for every provided prediction sample.

## YAML

```yaml
pipeline:
  - exclude:
      class: nirs4all.operators.filters.YOutlierFilter
      params:
        method: iqr
        threshold: 1.5

  - model:
      class: sklearn.cross_decomposition.PLSRegression
```

Multiple filters with mode:

```yaml
pipeline:
  - exclude:
      - class: nirs4all.operators.filters.YOutlierFilter
        params:
          method: iqr
      - class: nirs4all.operators.filters.XOutlierFilter
        params:
          method: mahalanobis
          threshold: 3.0
    mode: any
```

`mode: any` excludes a sample if any filter flags it. `mode: all` excludes only samples flagged by all filters.

## JSON

```json
{
  "pipeline": [
    {
      "exclude": [
        {
          "class": "nirs4all.operators.filters.YOutlierFilter",
          "params": {
            "method": "iqr"
          }
        },
        {
          "class": "nirs4all.operators.filters.XOutlierFilter",
          "params": {
            "method": "mahalanobis",
            "threshold": 3.0
          }
        }
      ],
      "mode": "any"
    }
  ]
}
```

## Python

```python
from nirs4all.operators.filters import XOutlierFilter, YOutlierFilter

pipeline = [
    {
        "exclude": [
            YOutlierFilter(method="iqr"),
            XOutlierFilter(method="mahalanobis", threshold=3.0),
        ],
        "mode": "any",
    },
    {"model": model},
]
```

Use {doc}`tag` when you want labels without removing samples.
