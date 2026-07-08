# Serialized Operator Nodes

Use a serialized operator node when a pipeline must be portable across YAML, JSON, Python, R, Julia, JavaScript, or a runtime wrapper.

## Supported Forms

| Form | YAML/JSON | Python object pipeline | Meaning |
| --- | --- | --- | --- |
| `class` | Yes | Yes | Import a class and instantiate it. |
| string import path | Yes | Yes | Short form for `class` with default parameters. |
| `function` | Yes | Yes | Import a callable. |
| `instance` | Internal/advanced | Internal/advanced | Restore a serialized instance produced by NIRS4ALL internals. |
| direct object | No | Yes | Pass an already-instantiated Python object. |
| `params` | Yes | Yes | Constructor keyword arguments. |
| `name` | Yes | Yes | Human-readable label used in traces/reports. |
| `force_layout` | Yes | Yes | Force data layout for this step. |

Valid `force_layout` values are `2d`, `2d_interleaved`, `3d`, and `3d_transpose`.

## YAML

```yaml
pipeline:
  - class: sklearn.preprocessing.StandardScaler

  - class: sklearn.decomposition.PCA
    params:
      n_components: 20
    name: pca_20

  - class: nirs4all.operators.transforms.StandardNormalVariate
    force_layout: 2d
```

## JSON

```json
{
  "pipeline": [
    {
      "class": "sklearn.preprocessing.StandardScaler"
    },
    {
      "class": "sklearn.decomposition.PCA",
      "params": {
        "n_components": 20
      },
      "name": "pca_20"
    }
  ]
}
```

## Python

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pipeline = [
    StandardScaler(),
    PCA(n_components=20),
]
```

Equivalent portable form:

```python
pipeline = [
    {"class": "sklearn.preprocessing.StandardScaler"},
    {
        "class": "sklearn.decomposition.PCA",
        "params": {"n_components": 20},
    },
]
```

## Notes

- Class names are imported at runtime, so use import paths that exist in the execution environment.
- For multi-language configs, prefer serialized forms rather than Python-only object instances.
- For supervised estimators, wrap the estimator in `model` unless you intentionally rely on model auto-detection.

See {doc}`model`, {doc}`preprocessing`, and {doc}`/reference/operator_catalog`.
