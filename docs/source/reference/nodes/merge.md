# `merge`: decide what to bring back together

After branching, choose **what** to combine. Features, predictions and separated
observations have different meanings. The argument determines the operation.

## Choose the result you need

```{figure} /assets/guide/merge.svg
:alt: Feature merges add columns, prediction merges create model-output columns, separation merges restore rows and source merges fuse modalities.
:width: 100%

Read the output axis: a merge can widen a matrix, build stacking inputs or restore observation rows.
```


| You have | You want | Write | Example output |
|---|---|---|---|
| Two feature branches, each 48 × 31 | One learner using both | `merge: features` | 48 × 62 |
| Two single-target base regressors | A combiner of their predictions | `merge: predictions` | 48 × 2 |
| Two 31-feature branches, each with a regressor | Both features and predictions | `merge: all` | 48 × 64 |
| Different observation cohorts | Original observation order | `merge: concat` | 48 rows, original width |
| 31 spectral + three marker features | Matrix for a classical learner | `merge: {sources: concat}` | 48 × 34 |
| Equal-width sources | A source axis | `merge: {sources: stack}` | Source-axis representation |
| Sources for a source-aware model | Named blocks retained | `merge: {sources: dict}` | Named source blocks |

Rows match by observation identity. Do not independently sort sources and then
assume that matching row positions represent the same specimens.

## Worked example: combine predictions safely

SNV + PLS and standardization + Ridge may make complementary errors. A final
Ridge learns to combine their predictions. This is **stacking**: its features
are model outputs, not wavelengths.

:::{note}
These tabs express the same **Python SDK workflow**. JSON and YAML are recipe
files; Python can also use sklearn/nirs4all objects. R, Octave and WASM native
pipeline facades do not execute this host-controller node directly.
For a recipe that runs in those languages, use {doc}`/guide/languages`.
:::

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "pipeline": [
    {
      "split": {
        "class": "sklearn.model_selection.KFold",
        "params": {
          "n_splits": 3,
          "shuffle": true,
          "random_state": 17
        }
      }
    },
    {
      "branch": {
        "pls": [
          {
            "class": "nirs4all.operators.transforms.StandardNormalVariate"
          },
          {
            "model": {
              "class": "sklearn.cross_decomposition.PLSRegression",
              "params": {
                "n_components": 3
              }
            }
          }
        ],
        "ridge": [
          {
            "class": "sklearn.preprocessing.StandardScaler"
          },
          {
            "model": {
              "class": "sklearn.linear_model.Ridge",
              "params": {
                "alpha": 1.0
              }
            }
          }
        ]
      }
    },
    {
      "merge": "predictions"
    },
    {
      "model": {
        "class": "sklearn.linear_model.Ridge",
        "params": {
          "alpha": 0.1
        }
      }
    }
  ]
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
pipeline:
- split:
    class: sklearn.model_selection.KFold
    params:
      n_splits: 3
      shuffle: true
      random_state: 17
- branch:
    pls:
    - class: nirs4all.operators.transforms.StandardNormalVariate
    - model:
        class: sklearn.cross_decomposition.PLSRegression
        params:
          n_components: 3
    ridge:
    - class: sklearn.preprocessing.StandardScaler
    - model:
        class: sklearn.linear_model.Ridge
        params:
          alpha: 1.0
- merge: predictions
- model:
    class: sklearn.linear_model.Ridge
    params:
      alpha: 0.1
```
:::

:::{tab-item} Python
:sync: python

```python
import tempfile
import numpy as np
import nirs4all
from nirs4all.data.dataset import SpectroDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from nirs4all.operators.transforms import SNV

rng = np.random.default_rng(17)
X = rng.normal(size=(48, 31))
y = 2 * X[:, 10] - X[:, 20] + rng.normal(scale=0.1, size=48)
dataset = SpectroDataset("worked_node")
dataset.add_samples(X[:40], {"partition": "train"},
                    headers=[str(i) for i in range(31)])
dataset.add_samples(X[40:], {"partition": "test"})
dataset.add_targets(y.reshape(-1, 1))


pipeline = [
    KFold(n_splits=3, shuffle=True, random_state=17),
    {"branch": {
        "pls": [SNV(), {"model": PLSRegression(n_components=3)}],
        "ridge": [StandardScaler(), {"model": Ridge(alpha=1.0)}],
    }},
    {"merge": "predictions"},
    {"model": Ridge(alpha=0.1)},
]
with tempfile.TemporaryDirectory() as workspace:
    result = nirs4all.run(pipeline, dataset, engine="dag-ml",
                         workspace_path=workspace, save_charts=False, verbose=0)
    print("CV validation score:", result.cv_best_score)
    print("Selected model test RMSE:", result.best_rmse)
    result.close()
```
:::

::::


**What to expect:** each base model provides one prediction column, giving the
combiner two columns for the same observation IDs. For training rows, the values
come from models that excluded the corresponding row from fitting: **out-of-fold
(OOF) predictions**. New observations pass through the saved base models before
the final Ridge predicts.

OOF prevents teaching the combiner with overly optimistic predictions on rows a
base model already learned. Choosing the architecture still consumes validation
information. Keep an outer holdout for an independent final claim.
The synthetic example explains the mechanism without promising a better score.

## Combine features instead

These branches contain transforms only. The learner comes after the merge.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "pipeline": [
    {
      "split": {
        "class": "sklearn.model_selection.KFold",
        "params": {
          "n_splits": 3,
          "shuffle": true,
          "random_state": 17
        }
      }
    },
    {
      "branch": {
        "normalized": [
          {
            "class": "nirs4all.operators.transforms.StandardNormalVariate"
          }
        ],
        "smoothed": [
          {
            "class": "nirs4all.operators.transforms.SavitzkyGolay",
            "params": {
              "window_length": 7,
              "polyorder": 2,
              "deriv": 0
            }
          }
        ]
      }
    },
    {
      "merge": "features"
    },
    {
      "model": {
        "class": "sklearn.linear_model.Ridge",
        "params": {
          "alpha": 1.0
        }
      }
    }
  ]
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
pipeline:
- split:
    class: sklearn.model_selection.KFold
    params:
      n_splits: 3
      shuffle: true
      random_state: 17
- branch:
    normalized:
    - class: nirs4all.operators.transforms.StandardNormalVariate
    smoothed:
    - class: nirs4all.operators.transforms.SavitzkyGolay
      params:
        window_length: 7
        polyorder: 2
        deriv: 0
- merge: features
- model:
    class: sklearn.linear_model.Ridge
    params:
      alpha: 1.0
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.operators.transforms import SavitzkyGolay
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

pipeline = [{'split': KFold(n_splits=3, shuffle=True, random_state=17)},
 {'branch': {'normalized': [SNV()],
             'smoothed': [SavitzkyGolay(window_length=7, polyorder=2, deriv=0)]}},
 {'merge': 'features'},
 {'model': Ridge(alpha=1.0)}]
```
:::

::::


**Expected result:** 62 columns: all normalized columns followed by all smoothed
columns. One Ridge learns from that matrix. Compare against each single view
before concluding that extra columns help.

## Select particular blocks

This fragment selects branches 0 and 1 and keeps pre-branch features. For two
31-column branches and 31 originals, the result has 93 columns.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "merge": {
    "features": {
      "branches": [
        0,
        1
      ]
    },
    "include_original": true,
    "output_as": "features",
    "on_missing": "error",
    "on_shape_mismatch": "error"
  }
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
merge:
  features:
    branches:
    - 0
    - 1
  include_original: true
  output_as: features
  on_missing: error
  on_shape_mismatch: error
```
:::

:::{tab-item} Python
:sync: python

```python
merge_step = {'merge': {'features': {'branches': [0, 1]},
           'include_original': True,
           'output_as': 'features',
           'on_missing': 'error',
           'on_shape_mismatch': 'error'}}
```
:::

::::


You can specify `features` and `predictions` in the same mapping. Retaining
original inputs changes model width, scaling balance and required capacity.

## Select and name the source fusion

For a two-source dataset whose declared names are `source_0` (31 spectral
features) and `source_1` (three laboratory markers), this fragment chooses those
two blocks explicitly. It operates on sources rather than duplication branches.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "merge": {
    "sources": {
      "strategy": "concat",
      "sources": [
        "source_0",
        "source_1"
      ],
      "on_incompatible": "error",
      "output_name": "spectra_and_markers",
      "preserve_source_info": true
    }
  }
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
merge:
  sources:
    strategy: concat
    sources:
    - source_0
    - source_1
    on_incompatible: error
    output_name: spectra_and_markers
    preserve_source_info: true
```
:::

:::{tab-item} Python
:sync: python

```python
merge_step = {'merge': {'sources': {'strategy': 'concat',
                       'sources': ['source_0', 'source_1'],
                       'on_incompatible': 'error',
                       'output_name': 'spectra_and_markers',
                       'preserve_source_info': True}}}
```
:::

::::

**Expected result:** 34 columns for each shared observation, named
`spectra_and_markers`. Source names identify provenance; they are not inferred
from the scientific meaning of column values.

| Source-merge field | Choices / purpose |
|---|---|
| `strategy` | `concat`, `stack`, `dict` |
| `sources` | `all`, source indices, or declared names |
| `on_incompatible` | `error`, `flatten`, `pad`, `truncate` for incompatible stack widths |
| `output_name` | Label of merged source |
| `preserve_source_info` | Retain input-source provenance metadata |

`flatten` falls back to column concatenation; `pad` introduces zero-filled
positions; `truncate` discards features to match the shorter width. Prefer
`error` until you have chosen and documented the scientific meaning of a fallback.

## Enumerate merge options

| Key | Controls | Starting choice |
|---|---|---|
| `features` | All branch features, indices, or `{branches: [...]}` | `all` or explicit indices |
| `predictions` | Base predictions, indices, per-branch selection | `all` for a small stack |
| `sources` | `concat`, `stack`, `dict`, or source mapping | `concat` for matrix models |
| `concat` | Reassemble separated rows when true | `merge: concat` shorthand |
| `include_original` | Retain inputs from before branching | False unless needed |
| `on_missing` | `error`, `warn`, `skip` | `error` during development |
| `on_shape_mismatch` | Incompatible output shapes | `error` |
| `output_as` | Output representation | `features` here |
| `unsafe` | Disable OOF reconstruction | Leave false |
| `source_names` | Labels for retained source outputs | Meaningful modality names |

`merge_sources` and `merge_predictions` are aliases handled by the same controller.
Advanced per-branch prediction selection includes `best`, `top_k`, metrics and
aggregation. See {doc}`/reference/pipeline_keywords` for exact mappings.

## Common mistakes

- `concat` restores separated **rows**; `features` joins branch **columns**.
- `stack` needs compatible source widths and a consumer that accepts that axis.
- `unsafe: true` does not repair missing OOF predictions. Fix the fold plan or
  missing base model outputs before trusting evaluation.
- Silently skipping missing branches changes the meaning of model columns.

Executable fusion and stacking companion:
[D08](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D08_documented_multisource_stacking.py).
Start with {doc}`branch` if merge inputs remain unclear.
