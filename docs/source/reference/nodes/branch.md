# `branch`: let the same samples follow different paths

A spectrum may contain useful information after normalization **and** smoothing.
`branch` keeps both paths. You decide later whether to compare them, join their
features, or combine their predictions.

A branch is a named list of steps. Each path starts from the incoming data;
preprocessing in one path does not overwrite another path.

## Worked example: two views, one Ridge model

```{figure} /assets/guide/branch.svg
:alt: The same 48 observations with 31 features go to SNV and smoothing independently, then merge into 62 columns before Ridge.
:width: 100%

The merge joins columns: 31 normalized + 31 smoothed = 62 features. Both paths preserve observation identity.
```

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
import tempfile
import numpy as np
import nirs4all
from nirs4all.data.dataset import SpectroDataset
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from nirs4all.operators.transforms import SNV, SavitzkyGolay

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
        "normalized": [SNV()],
        "smoothed": [SavitzkyGolay(window_length=7, polyorder=2, deriv=0)],
    }},
    {"merge": "features"},
    {"model": Ridge(alpha=1.0)},
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


**What to expect:** one fused model recipe. Ridge is fitted once per fold on a
62-column input. The eight test observations receive predictions. SNV and
smoothing both see the incoming spectra; this is not SNV followed by smoothing.
The example selects `engine="dag-ml"` and requires that graph runtime installed.
See {doc}`/guide/start` for setup.

## Three reasons to branch

| Mode | Input per path | Purpose | Join |
|---|---|---|---|
| Named/list duplication | Same observations and sources | Complementary preprocessing or models | `merge: features` or `merge: predictions` |
| `by_metadata`, `by_tag`, `by_filter` | Different observation subsets | Separate sites or flagged cohorts | `merge: concat` restores rows |
| `by_source` | Modalities of the same sample | Spectra and laboratory markers need different transforms | `merge: {sources: concat}` |

**Without a merge**, subsequent steps continue on each active branch. A model
immediately after two preprocessing branches fits separate models. With a
feature merge before the model, it fits one model on both blocks.

## Compare paths instead of fusing them

Omit the merge to apply the same Ridge estimator to both representations:

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
 {'model': Ridge(alpha=1.0)}]
```
:::

::::


**Expected result:** two branch-specific model results per fold. Predictions
remain separate. For a stack, put models inside each path and use the prediction
merge explained in {doc}`merge`.

## Give each modality its own preprocessing

For a dataset with spectra first and laboratory markers second, default source
names are `source_0` and `source_1`. Use your declared names if you assigned them.

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
        "by_source": true,
        "steps": {
          "source_0": [
            {
              "class": "nirs4all.operators.transforms.StandardNormalVariate"
            }
          ],
          "source_1": [
            {
              "class": "sklearn.preprocessing.StandardScaler"
            }
          ]
        }
      }
    },
    {
      "merge": {
        "sources": "concat"
      }
    },
    {
      "model": {
        "class": "sklearn.cross_decomposition.PLSRegression",
        "params": {
          "n_components": 3
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
    by_source: true
    steps:
      source_0:
      - class: nirs4all.operators.transforms.StandardNormalVariate
      source_1:
      - class: sklearn.preprocessing.StandardScaler
- merge:
    sources: concat
- model:
    class: sklearn.cross_decomposition.PLSRegression
    params:
      n_components: 3
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

pipeline = [{'split': KFold(n_splits=3, shuffle=True, random_state=17)},
 {'branch': {'by_source': True,
             'steps': {'source_0': [SNV()], 'source_1': [StandardScaler()]}}},
 {'merge': {'sources': 'concat'}},
 {'model': PLSRegression(n_components=3)}]
```
:::

::::


**Expected result:** 31 spectral + three marker columns make a 34-column matrix.
Physical-sample IDs must align across sources. This fragment needs that two-source
dataset; the [D08 multisource example](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D08_documented_multisource_stacking.py)
provides complete data construction and prediction replay.

## Route observations using metadata

With a `site` metadata column containing `site_a` and `site_b`, each cohort can
receive different preprocessing. The following fragment assumes equal output
widths and uses the Python legacy lane for metadata separation.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "pipeline": [
    {
      "branch": {
        "by_metadata": "site",
        "steps": {
          "site_a": [
            {
              "class": "nirs4all.operators.transforms.StandardNormalVariate"
            }
          ],
          "site_b": [
            {
              "class": "sklearn.preprocessing.StandardScaler"
            }
          ]
        }
      }
    },
    {
      "merge": "concat"
    },
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
      "model": {
        "class": "sklearn.cross_decomposition.PLSRegression",
        "params": {
          "n_components": 3
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
- branch:
    by_metadata: site
    steps:
      site_a:
      - class: nirs4all.operators.transforms.StandardNormalVariate
      site_b:
      - class: sklearn.preprocessing.StandardScaler
- merge: concat
- split:
    class: sklearn.model_selection.KFold
    params:
      n_splits: 3
      shuffle: true
      random_state: 17
- model:
    class: sklearn.cross_decomposition.PLSRegression
    params:
      n_components: 3
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

pipeline = [{'branch': {'by_metadata': 'site',
             'steps': {'site_a': [SNV()], 'site_b': [StandardScaler()]}}},
 {'merge': 'concat'},
 {'split': KFold(n_splits=3, shuffle=True, random_state=17)},
 {'model': PLSRegression(n_components=3)}]
```
:::

::::


**Expected result:** observation rows are reassembled by identity. Site A's
columns are not joined side by side with site B's columns.

## Route using an existing quality tag

A tag records a flag without removing rows. This example flags spectra whose
variance is below 0.8 in the synthetic data's arbitrary units, then gives them a
different transform. The threshold is a demonstration, not a universal spectral
quality criterion. Use this fragment with the synthetic dataset above and the
Python legacy lane.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "pipeline": [
    {
      "tag": {
        "class": "nirs4all.operators.filters.SpectralQualityFilter",
        "params": {
          "min_variance": 0.8,
          "tag_name": "low_variance"
        }
      }
    },
    {
      "branch": {
        "by_tag": "low_variance",
        "values": {
          "ordinary": false,
          "flagged": true
        },
        "steps": {
          "ordinary": [
            {
              "class": "nirs4all.operators.transforms.StandardNormalVariate"
            }
          ],
          "flagged": [
            {
              "class": "sklearn.preprocessing.StandardScaler"
            }
          ]
        }
      }
    },
    {
      "merge": "concat"
    },
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
      "model": {
        "class": "sklearn.cross_decomposition.PLSRegression",
        "params": {
          "n_components": 3
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
- tag:
    class: nirs4all.operators.filters.SpectralQualityFilter
    params:
      min_variance: 0.8
      tag_name: low_variance
- branch:
    by_tag: low_variance
    values:
      ordinary: false
      flagged: true
    steps:
      ordinary:
      - class: nirs4all.operators.transforms.StandardNormalVariate
      flagged:
      - class: sklearn.preprocessing.StandardScaler
- merge: concat
- split:
    class: sklearn.model_selection.KFold
    params:
      n_splits: 3
      shuffle: true
      random_state: 17
- model:
    class: sklearn.cross_decomposition.PLSRegression
    params:
      n_components: 3
```
:::

:::{tab-item} Python
:sync: python

```python
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from nirs4all.operators.transforms import SNV
from nirs4all.operators.filters import SpectralQualityFilter

pipeline = [
    {"tag": SpectralQualityFilter(min_variance=0.8, tag_name="low_variance")},
    {"branch": {
        "by_tag": "low_variance",
        "values": {"ordinary": False, "flagged": True},
        "steps": {"ordinary": [SNV()], "flagged": [StandardScaler()]},
    }},
    {"merge": "concat"},
    KFold(n_splits=3, shuffle=True, random_state=17),
    {"model": PLSRegression(n_components=3)},
]
```
:::

::::

**Expected result:** each observation enters exactly one path, then row reassembly
restores the original 48-row order and 31-column width. Here the tag uses
**true = failed the quality check**, whereas the filter's keep mask uses
**true = passed**. No rows are deleted. A variance-based decision can be computed
from new spectra without knowing their concentration.

## Route directly from a filter's pass/fail result

Use `by_filter` when you do not need a stored tag. It applies the same criterion
and routes to default branch names `passing` and `failing`.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "pipeline": [
    {
      "branch": {
        "by_filter": {
          "class": "nirs4all.operators.filters.SpectralQualityFilter",
          "params": {
            "min_variance": 0.8,
            "tag_name": "low_variance"
          }
        },
        "steps": {
          "passing": [
            {
              "class": "nirs4all.operators.transforms.StandardNormalVariate"
            }
          ],
          "failing": [
            {
              "class": "sklearn.preprocessing.StandardScaler"
            }
          ]
        }
      }
    },
    {
      "merge": "concat"
    },
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
      "model": {
        "class": "sklearn.cross_decomposition.PLSRegression",
        "params": {
          "n_components": 3
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
- branch:
    by_filter:
      class: nirs4all.operators.filters.SpectralQualityFilter
      params:
        min_variance: 0.8
        tag_name: low_variance
    steps:
      passing:
      - class: nirs4all.operators.transforms.StandardNormalVariate
      failing:
      - class: sklearn.preprocessing.StandardScaler
- merge: concat
- split:
    class: sklearn.model_selection.KFold
    params:
      n_splits: 3
      shuffle: true
      random_state: 17
- model:
    class: sklearn.cross_decomposition.PLSRegression
    params:
      n_components: 3
```
:::

:::{tab-item} Python
:sync: python

```python
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from nirs4all.operators.transforms import SNV
from nirs4all.operators.filters import SpectralQualityFilter

pipeline = [
    {"branch": {
        "by_filter": SpectralQualityFilter(min_variance=0.8),
        "steps": {"passing": [SNV()], "failing": [StandardScaler()]},
    }},
    {"merge": "concat"},
    KFold(n_splits=3, shuffle=True, random_state=17),
    {"model": PLSRegression(n_components=3)},
]
```
:::

::::

**Expected result:** two complementary observation subsets are reassembled into
48 rows, each still with 31 features. This is routing, not exclusion. The fixed
variance filter needs no learned threshold; learned outlier filters have fitted
state and require a deliberate validation/deployment protocol. Optional `names`
changes the two pass/fail branch labels.

## Enumerate all routing forms

| Form | Required input | Check before deployment |
|---|---|---|
| Named mapping | Lists of branch steps | Use meaningful report names |
| List of lists | Anonymous branch steps | Names help when the recipe grows |
| `by_metadata` | Metadata column | New rows need known routable values |
| `by_tag` | Earlier {doc}`tag` | The tag must be computable for unlabeled rows |
| `by_filter` | Filter definitions | Check overlaps and unmatched rows |
| `by_source` | Multiple declared sources | Match `steps` to source names |

A target-outlier tag can diagnose training data but usually cannot route a new
sample whose target is unknown. Tiny separated cohorts may lack sufficient rows
for the requested CV folds or PLS components.

## Avoid surprises

- `_or_` compares recipes; `branch` keeps paths in one workflow.
- Nested branches multiply complexity. Confirm a simple branch/merge first.
- New metadata values require a routing decision; they are not automatically
  interchangeable with training sites.

Runnable sources: [D01 branching basics](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D01_branching_basics.py),
[D06 separation branches](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D06_separation_branches.py).
Full schema: {doc}`/reference/pipeline_keywords`.
