# `feature_augmentation`: retain several views of the same observations

Raw spectra show absolute signal. SNV reduces within-spectrum offset and scale.
A derivative emphasizes local changes. Retain all three instead of overwriting
the last representation each time.

This node creates **feature views**. Every view describes the same observation;
there is no new specimen and no new target measurement.

## Worked example: raw, normalized and derivative views

```{figure} /assets/guide/feature_augmentation.svg
:alt: Raw 48 by 31 spectra are kept alongside independent SNV and first-derivative views of the same 48 observations.
:width: 100%

With action extend, raw plus two new views gives three views. A view-axis representation is 48 × 3 × 31.
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
      "feature_augmentation": [
        {
          "class": "nirs4all.operators.transforms.StandardNormalVariate"
        },
        {
          "class": "nirs4all.operators.transforms.SavitzkyGolay",
          "params": {
            "window_length": 7,
            "polyorder": 2,
            "deriv": 1
          }
        }
      ],
      "action": "extend"
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
- feature_augmentation:
  - class: nirs4all.operators.transforms.StandardNormalVariate
  - class: nirs4all.operators.transforms.SavitzkyGolay
    params:
      window_length: 7
      polyorder: 2
      deriv: 1
  action: extend
- model:
    class: sklearn.cross_decomposition.PLSRegression
    params:
      n_components: 3
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
from sklearn.cross_decomposition import PLSRegression
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
    {"feature_augmentation": [
        SNV(),
        SavitzkyGolay(window_length=7, polyorder=2, deriv=1),
    ], "action": "extend"},
    {"model": PLSRegression(n_components=3)},
]
with tempfile.TemporaryDirectory() as workspace:
    result = nirs4all.run(pipeline, dataset, engine="legacy",
                         workspace_path=workspace, save_charts=False, verbose=0)
    print("CV validation score:", result.cv_best_score)
    print("Selected model test RMSE:", result.best_rmse)
    result.close()
```
:::

::::


**What to expect:** raw, SNV and Savitzky–Golay first-derivative views remain.
There are 48 observation identities, 31 feature positions and three views.
The PLS matrix input flattens these equal-width views to 93 columns. A tensor
model can preserve the view axis with an appropriate supported layout.
The example selects the Python legacy controller and writes `action` explicitly.
The number of model recipes depends on the search topology, not simply the view count.

## Enumerate action modes

Suppose two views already exist (raw and normalized), and you request two new
transforms (smoothing and derivative).

| `action` | Inputs transformed | Old views retained? | Resulting count |
|---|---|---|---:|
| `extend` | Base/first view | All | 2 old + 2 new = 4 |
| `add` | Each of two old views | All | 2 old + 4 new = 6 |
| `replace` | Each of two old views | None | 4 new |

`extend` avoids duplicating an existing processing; the table assumes distinct
new views. Operations use the set of views present when the node begins. The
operations within a node do not themselves form a sequential chain.
Always write `action`: the historical implementation uses `add` when omitted,
despite older descriptions calling `extend` the default.

## Compare combinations of views

`pick: 2` chooses every pair of SNV, smoothing and derivative. `extend` also
keeps raw input in each candidate.

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
      "feature_augmentation": {
        "_or_": [
          {
            "class": "nirs4all.operators.transforms.StandardNormalVariate"
          },
          {
            "class": "nirs4all.operators.transforms.SavitzkyGolay",
            "params": {
              "window_length": 7,
              "polyorder": 2,
              "deriv": 0
            }
          },
          {
            "class": "nirs4all.operators.transforms.SavitzkyGolay",
            "params": {
              "window_length": 7,
              "polyorder": 2,
              "deriv": 1
            }
          }
        ],
        "pick": 2
      },
      "action": "extend"
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
- feature_augmentation:
    _or_:
    - class: nirs4all.operators.transforms.StandardNormalVariate
    - class: nirs4all.operators.transforms.SavitzkyGolay
      params:
        window_length: 7
        polyorder: 2
        deriv: 0
    - class: nirs4all.operators.transforms.SavitzkyGolay
      params:
        window_length: 7
        polyorder: 2
        deriv: 1
    pick: 2
  action: extend
- model:
    class: sklearn.cross_decomposition.PLSRegression
    params:
      n_components: 3
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.operators.transforms import SavitzkyGolay
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

pipeline = [{'split': KFold(n_splits=3, shuffle=True, random_state=17)},
 {'feature_augmentation': {'_or_': [SNV(),
                                    SavitzkyGolay(window_length=7, polyorder=2, deriv=0),
                                    SavitzkyGolay(window_length=7, polyorder=2, deriv=1)],
                           'pick': 2},
  'action': 'extend'},
 {'model': PLSRegression(n_components=3)}]
```
:::

::::


**Expected result:** three candidate recipes: raw + SNV + smoothing; raw + SNV +
derivative; raw + smoothing + derivative. Each has three 31-column views.
The first example is one experiment with three views; this is three experiments.

## View, column or observation?

| Node | Adds | More observation rows? |
|---|---|---|
| `feature_augmentation` | Named representations | No |
| {doc}`concat_transform` | Columns in one matrix | No |
| {doc}`sample_augmentation` | Synthetic training observations | Yes, during training |
| {doc}`generators` | Alternative recipes | No, within each recipe |

## Common mistakes

- Extra views do not create independent biological replication.
- Repeated `add` grows quickly: two views and three transforms yield eight views.
- PCA and other reductions produce unequal widths. Prefer `concat_transform`
  for unequal-width blocks consumed as a matrix.
- A 3D layout does not teach a matrix-only estimator to accept tensors.

Runnable [U02 tutorial](https://github.com/GBeurier/nirs4all/blob/main/examples/user/03_preprocessing/U02_feature_augmentation.py).
See {doc}`operator_step` for layouts and
{doc}`/reference/multimodal_execution_matrix` for supported shapes.
