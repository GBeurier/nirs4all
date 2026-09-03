# `tag`

`tag` marks samples without removing them. Tags can be used for analysis, charts, or separation branches.

## YAML

```yaml
pipeline:
  - tag:
      class: nirs4all.operators.filters.YOutlierFilter
      params:
        method: iqr
        threshold: 1.5
        tag_name: y_outlier_iqr

  - branch:
      by_tag: y_outlier_iqr
      values:
        clean: false
        outliers: true
```

Multiple filters:

```yaml
pipeline:
  - tag:
      - class: nirs4all.operators.filters.YOutlierFilter
        params:
          method: iqr
      - class: nirs4all.operators.filters.XOutlierFilter
        params:
          method: mahalanobis
          threshold: 3.0
```

## JSON

```json
{
  "pipeline": [
    {
      "tag": {
        "class": "nirs4all.operators.filters.YOutlierFilter",
        "params": {
          "method": "iqr",
          "threshold": 1.5,
          "tag_name": "y_outlier_iqr"
        }
      }
    }
  ]
}
```

## Python

```python
from nirs4all.operators.filters import YOutlierFilter

pipeline = [
    {"tag": YOutlierFilter(method="iqr", threshold=1.5, tag_name="y_outlier_iqr")},
]
```

## Common Filters

`YOutlierFilter`, `XOutlierFilter`, `SpectralQualityFilter`, `HighLeverageFilter`, and `MetadataFilter`.

See {doc}`exclude` and {doc}`branch`.
