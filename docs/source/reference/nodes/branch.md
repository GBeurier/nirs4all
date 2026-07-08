# `branch`

`branch` creates parallel paths in the pipeline. There are two branch modes:

- **duplication**: every branch sees the same samples and applies different steps;
- **separation**: samples or sources are partitioned by tag, metadata, filter, or source.

## Duplication Branches

Use duplication when you want to compare or combine preprocessing/model strategies.

```yaml
pipeline:
  - split:
      class: sklearn.model_selection.KFold
      params:
        n_splits: 5
        shuffle: true
        random_state: 42

  - branch:
      snv:
        - class: nirs4all.operators.transforms.StandardNormalVariate
      msc:
        - class: nirs4all.operators.transforms.MultiplicativeScatterCorrection
      derivative:
        - class: nirs4all.operators.transforms.FirstDerivative

  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components: 10
```

Anonymous list syntax:

```yaml
pipeline:
  - branch:
      - - class: nirs4all.operators.transforms.StandardNormalVariate
      - - class: nirs4all.operators.transforms.MultiplicativeScatterCorrection
```

## Separation by Tag

Use `tag` first, then split samples by tag value.

```yaml
pipeline:
  - tag:
      class: nirs4all.operators.filters.YOutlierFilter
      params:
        method: iqr
        tag_name: y_outlier_iqr

  - branch:
      by_tag: y_outlier_iqr
      values:
        clean: false
        outliers: true
      steps:
        clean:
          - class: nirs4all.operators.transforms.StandardNormalVariate
        outliers:
          - class: nirs4all.operators.transforms.RobustStandardNormalVariate

  - merge: concat
```

## Separation by Metadata

```yaml
pipeline:
  - branch:
      by_metadata: site
      steps:
        site_a:
          - class: nirs4all.operators.transforms.StandardNormalVariate
        site_b:
          - class: nirs4all.operators.transforms.MultiplicativeScatterCorrection

  - merge: concat
```

## Separation by Source

Use this for multi-source datasets when each source needs its own steps.

```yaml
pipeline:
  - branch:
      by_source: true
      steps:
        NIR:
          - class: nirs4all.operators.transforms.StandardNormalVariate
          - class: nirs4all.operators.transforms.FirstDerivative
        markers:
          - class: sklearn.preprocessing.StandardScaler

  - merge:
      sources: concat

  - model:
      class: sklearn.cross_decomposition.PLSRegression
```

## JSON

```json
{
  "pipeline": [
    {
      "branch": {
        "snv": [
          {
            "class": "nirs4all.operators.transforms.StandardNormalVariate"
          }
        ],
        "msc": [
          {
            "class": "nirs4all.operators.transforms.MultiplicativeScatterCorrection"
          }
        ]
      }
    },
    {
      "merge": "features"
    }
  ]
}
```

## Python

```python
from nirs4all.operators.transforms import MSC, SNV

pipeline = [
    {"branch": {
        "snv": [SNV()],
        "msc": [MSC()],
    }},
    {"merge": "features"},
    {"model": model},
]
```

## What Usually Comes Next

| Branch type | Common next node |
| --- | --- |
| Duplication, combine features | `{"merge": "features"}` |
| Duplication, stacking | `{"merge": "predictions"}` |
| Separation by tag/metadata/filter/source | `{"merge": "concat"}` |
| Multi-source source fusion | `{"merge": {"sources": "concat"}}` |

See {doc}`merge`.
