# `split`

`split` creates or loads cross-validation folds. A splitter step can also be a bare sklearn-compatible splitter object in Python.

## Syntax Summary

| Syntax | Meaning |
| --- | --- |
| `{"split": {"class": "...", "params": {...}}}` | Portable explicit splitter. |
| `{"split": "folds.csv"}` | Load precomputed folds from file. |
| bare splitter object | Python-only direct splitter step. |
| serialized splitter with `class` | Portable direct splitter step. |

## YAML

```yaml
pipeline:
  - split:
      class: sklearn.model_selection.KFold
      params:
        n_splits: 5
        shuffle: true
        random_state: 42

  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components: 10
```

Equivalent direct serialized splitter:

```yaml
pipeline:
  - class: sklearn.model_selection.ShuffleSplit
    params:
      n_splits: 5
      test_size: 0.25
      random_state: 42
```

Load fold indices:

```yaml
pipeline:
  - split: folds/my_folds.csv
  - model:
      class: sklearn.cross_decomposition.PLSRegression
```

## JSON

```json
{
  "pipeline": [
    {
      "split": {
        "class": "sklearn.model_selection.KFold",
        "params": {
          "n_splits": 5,
          "shuffle": true,
          "random_state": 42
        }
      }
    }
  ]
}
```

## Python

```python
from sklearn.model_selection import KFold

pipeline = [
    KFold(n_splits=5, shuffle=True, random_state=42),
    {"model": model},
]
```

## Supported Splitter Families

- sklearn splitters such as `KFold`, `ShuffleSplit`, `GroupKFold`, `StratifiedKFold`, and `TimeSeriesSplit`;
- NIRS splitters such as `KennardStoneSplitter`, `SPXYSplitter`, `SPXYFold`, `SPXYGFold`, `KMeansSplitter`, `SPlitSplitter`, `SystematicCircularSplitter`, `KBinsStratifiedSplitter`, and `BinnedStratifiedGroupKFold`;
- custom objects with a compatible `split(X, ...)` method.

See {doc}`/reference/splitters`.
