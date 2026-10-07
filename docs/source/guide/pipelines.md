# 5. Build a pipeline you can explain

**Your goal:** understand every node's effect, generate a small comparison, then build feature fusion, stacking and a two-source residual model. A pipeline grows by adding a reasoned operation, not by copying a complicated graph.

## The clickable node map

Click a family to see its parameters, equivalent code tabs, an illustration and the expected output. The {doc}`complete node catalogue </reference/nodes/index>` lists individual operations.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 1 · Transform measurements
:link: /reference/nodes/preprocessing
:link-type: doc

SNV, scatter correction, smoothing, scaling and dimensionality reduction. What changes in X?
:::

:::{grid-item-card} 2 · Transform the target
:link: /reference/nodes/y_processing
:link-type: doc

Scale y for fitting; return predictions in original scientific units.
:::

:::{grid-item-card} 3 · Split and validate
:link: /reference/nodes/split
:link-type: doc

Choose training/validation membership, preserve groups and score held-out observations.
:::

:::{grid-item-card} 4 · Fit a model
:link: /reference/nodes/model
:link-type: doc

Regressors, classifiers, PLS, Ridge and training/search settings.
:::

:::{grid-item-card} 5 · Generate candidates
:link: /reference/nodes/generators
:link-type: doc

Enumerate alternatives, ranges, grids, paired choices and constrained combinations.
:::

:::{grid-item-card} 6 · Branch and merge
:link: /reference/nodes/branch
:link-type: doc

Send observations or sources through several paths, then join features or predictions.
:::

:::{grid-item-card} 7 · Tag and exclude rows
:link: /reference/nodes/tag
:link-type: doc

Identify observations with a tag; remove permitted training rows with an exclusion.
:::

:::{grid-item-card} 8 · Augment or concatenate
:link: /reference/nodes/sample_augmentation
:link-type: doc

Create derived training observations, extra feature views or concatenated transforms.
:::

:::{grid-item-card} 9 · Work with repeated measurements
:link: /reference/nodes/repetitions
:link-type: doc

Reshape repeated scans while retaining their specimen relationships.
:::
::::

Additional node pages: {doc}`merge </reference/nodes/merge>`, {doc}`exclude </reference/nodes/exclude>`, {doc}`concat_transform </reference/nodes/concat_transform>`, {doc}`charts </reference/nodes/charts>`, {doc}`auto-transfer </reference/nodes/auto_transfer_preproc>`, and {doc}`residual model </reference/nodes/residual>`.

## Read inputs and outputs before choosing a node

| Operation | Input | Expected result |
|---|---|---|
| SNV/MSC | Spectral rows | Corrected spectral rows; same dimensions |
| PCA or variable selection | Feature matrix | Fewer or selected feature columns |
| `y_processing` | Training targets | Transformed targets; predictions inversely transformed |
| Splitter | Observation/group identities | Training and validation memberships |
| Model | Prepared X and training y | Learned relation; one output per requested target |
| Generator | Recipe with alternatives | Several separate concrete recipes |
| `branch` | Shared rows or named sources | Several paths inside one recipe |
| `merge: features` | Aligned transformed feature blocks | More columns, same observation IDs |
| `merge: predictions` | Held-out base-model outputs | Prediction columns for a final learner |
| `tag` / `exclude` | Observations and a rule | An annotation / permitted removal from fitting |
| Augmentation | Training rows and perturbation | Derived rows or extra views, with origins retained |
| Repetition conversion | Related measurements | A changed representation of the same specimens |
| Chart | Inputs or result | An inspectable figure; no model improvement by itself |

## 1 · Compare two preprocessors and three model sizes

SNV normalizes each row using its own mean and spread. MSC compares each spectrum to a reference estimated from training spectra. They are alternatives here. PLS `n_components` chooses how many target-related directions the model uses.

```{figure} /assets/guide/generators.svg
:alt: Two preprocessing choices times three PLS component counts produce six separate candidate recipes.

**Read the expansion before fitting.** SNV and MSC each pair with 2, 4 and 6 components. Six candidates × three folds = eighteen fold-model fits, followed by refit of the selected recipe. This is a search, not a six-model ensemble.
```

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "pipeline": [
    {
      "_or_": [
        {
          "class": "nirs4all.operators.transforms.StandardNormalVariate"
        },
        {
          "class": "nirs4all.operators.transforms.MultiplicativeScatterCorrection"
        }
      ]
    },
    {
      "class": "sklearn.model_selection.KFold",
      "params": {
        "n_splits": 3,
        "shuffle": true,
        "random_state": 17
      }
    },
    {
      "model": {
        "class": "sklearn.cross_decomposition.PLSRegression",
        "params": {
          "n_components": {
            "_range_": [
              2,
              6,
              2
            ]
          }
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
- _or_:
  - class: nirs4all.operators.transforms.StandardNormalVariate
  - class: nirs4all.operators.transforms.MultiplicativeScatterCorrection
- class: sklearn.model_selection.KFold
  params:
    n_splits: 3
    shuffle: true
    random_state: 17
- model:
    class: sklearn.cross_decomposition.PLSRegression
    params:
      n_components:
        _range_:
        - 2
        - 6
        - 2
```

:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.operators.transforms import SNV, MSC
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression

pipeline = [
    {"_or_": [SNV(), MSC()]},
    KFold(n_splits=3, shuffle=True, random_state=17),
    {"model": {"class": PLSRegression,
               "params": {"n_components": {"_range_": [2, 6, 2]}}}},
]
# Run against the dataset from the first Python SDK exercise:
# result = nirs4all.run(pipeline, (X, y), engine="dag-ml", refit=True)
```

:::

:::{tab-item} R
:sync: r

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

:::{tab-item} Octave / MATLAB
:sync: matlab

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

:::{tab-item} WASM / JavaScript
:sync: javascript

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

::::

**Expected result:** six candidate recipes, each evaluated with the same folds and target. `_range_: [2, 6, 2]` includes 2, 4 and 6. Keep the component count within the feature rank and smallest training-fold size. Use the synthetic SDK arrays from {doc}`start`, or supply your actual dataset.

### Enumerate the generator vocabulary

| Need | Keyword | What to expect |
|---|---|---|
| Try A or B | `_or_` | One separate candidate per alternative |
| Integer/linear values | `_range_` | A start/end/step sweep |
| Logarithmic values | `_log_range_` | Multiplicative-scale choices |
| Combine parameter settings | `_grid_` | Cartesian parameter combinations |
| Combine pipeline stages | `_cartesian_` | Every compatible stage combination |
| Keep matched settings together | `_zip_` | Paired choices, instead of all combinations |
| Join ordered specifications | `_chain_` | Concatenated ordered expansions |
| Sample a space | `_sample_`, `_seed_` | A reproducible bounded sample |
| Remove incompatible choices | `_mutex_`, `_requires_`, `_depends_on_`, `_exclude_` | Explicit compatibility rules |
| Reuse a named definition | `_preset_` | Expanded registered settings |

The {doc}`generator page </reference/nodes/generators>` shows each mechanism; the {doc}`exact keyword reference </reference/generator_keywords>` gives schemas and defaults. Some advanced expansions have runtime-specific support; inspect the generated values and preflight the chosen engine.

## 2 · Keep two views inside one recipe

Now use **both** normalization and smoothing as inputs to one model. Two paths see the same 31-column spectra. SNV returns 31 columns and smoothing returns 31, so `merge: features` produces 62 columns for Ridge. Forty rows train the recipe; eight separate rows test its refit.

```{figure} /assets/guide/fusion.svg
:alt: Feature fusion concatenates aligned source or branch feature columns; prediction fusion joins base predictions before a meta-model.

**Two kinds of combination.** Early fusion joins feature blocks before fitting one model. Stacking joins held-out prediction columns before fitting a final learner. Count the columns and keep the observation IDs aligned.
```

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
      params: {n_splits: 3, shuffle: true, random_state: 17}
  - branch:
      normalized:
        - class: nirs4all.operators.transforms.StandardNormalVariate
      smoothed:
        - class: nirs4all.operators.transforms.SavitzkyGolay
          params: {window_length: 7, polyorder: 2, deriv: 0}
  - merge: features
  - model:
      class: sklearn.linear_model.Ridge
      params: {alpha: 1.0}
```

:::

:::{tab-item} Python
:sync: python

```python
import tempfile
import numpy as np
import nirs4all
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.transforms import SNV, SavitzkyGolay

rng = np.random.default_rng(17)
X = rng.normal(size=(48, 31))
y = 2 * X[:, 10] - X[:, 20] + rng.normal(scale=0.1, size=48)

dataset = SpectroDataset("doc_branch_feature_join")
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
    result = nirs4all.run(
        pipeline=pipeline, dataset=dataset, engine="dag-ml", refit=True,
        workspace_path=workspace, save_charts=False, verbose=0,
    )
    print(result.execution_engine, result.best_rmse)
    result.close()
```

:::

:::{tab-item} R
:sync: r

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

:::{tab-item} Octave / MATLAB
:sync: matlab

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

:::{tab-item} WASM / JavaScript
:sync: javascript

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

::::

**Expected result:** the model receives 62 columns for every observation. More columns do not guarantee lower held-out error. Compare this recipe with the simpler baseline under the same validation protocol.

The Python tab is self-contained; JSON/YAML describe the same recipe, with the same data generated in Python. Fully qualified class paths construct Python objects. For native R/Octave/WASM training, use the supported finite recipe in {doc}`languages` or the {doc}`native multimodal tutorial </user_guide/data/methods_multimodal_u07>`.

## 3 · Combine models by stacking

Put a PLS model in the SNV path and Ridge in the MSC path. For each held-out row, each path contributes one prediction. `merge: predictions` therefore makes **two** input columns for a final Ridge learner, rather than 62 spectral columns.

The next Python tab continues the preceding example and reuses `X`, `y` and its imports. It creates a fresh dataset for this separate experiment. Its JSON/YAML tabs declare exactly the same stacking recipe.

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
            "class": "nirs4all.operators.transforms.MultiplicativeScatterCorrection"
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
    - class: nirs4all.operators.transforms.MultiplicativeScatterCorrection
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
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from nirs4all.operators.transforms import MSC

stacking_data = SpectroDataset("doc_stacking")
stacking_data.add_samples(X[:40], {"partition": "train"},
                          headers=[str(i) for i in range(31)])
stacking_data.add_samples(X[40:], {"partition": "test"})
stacking_data.add_targets(y.reshape(-1, 1))

stacking = [
    KFold(n_splits=3, shuffle=True, random_state=17),
    {"branch": {
        "pls": [SNV(), {"model": PLSRegression(n_components=3)}],
        "ridge": [MSC(), {"model": Ridge(alpha=1.0)}],
    }},
    {"merge": "predictions"},
    {"model": Ridge(alpha=0.1)},
]

with tempfile.TemporaryDirectory() as workspace:
    result = nirs4all.run(
        pipeline=stacking, dataset=stacking_data, engine="dag-ml", refit=True,
        workspace_path=workspace, save_charts=False, verbose=0,
    )
    archive = result.export(Path(workspace) / "stacked.n4a")
    replay = nirs4all.predict(archive, X[40:])
    assert len(replay.y_pred) == 8
    print(result.execution_engine, result.best_rmse)
    result.close()
```

:::

:::{tab-item} R
:sync: r

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

:::{tab-item} Octave / MATLAB
:sync: matlab

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

:::{tab-item} WASM / JavaScript
:sync: javascript

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

::::

**Expected result:** two base predictors and a meta-predictor are learned; exported replay returns eight values for the eight test rows. Meta-training inputs must be predictions made without training the base model on that observation. In-sample base predictions give the final learner overly favorable inputs.

This exercise uses duplication branches, default prediction merge and one downstream learner. Combining source routing with additional prediction branches is a richer graph and may be refused by DAG-ML. Choose a documented supported composition; {doc}`/reference/nodes/merge` explains the available forms.

## 4 · Add a second source and a residual correction

Use 31 spectral measurements and three laboratory markers for the same specimens. Normalize the spectral source, standardize marker columns, and merge the two sources into **34 features**. PLS makes a first estimate; Ridge learns what remains unexplained using held-out residuals. The final prediction is the base estimate plus its learned correction.

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
        "class": "nirs4all.operators.models.ResidualModel",
        "params": {
          "base": {
            "class": "sklearn.cross_decomposition.PLSRegression",
            "params": {
              "n_components": 3
            }
          },
          "learner": {
            "class": "sklearn.linear_model.Ridge",
            "params": {
              "alpha": 1.0
            }
          },
          "gate": false
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
      params: {n_splits: 3, shuffle: true, random_state: 17}
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
      class: nirs4all.operators.models.ResidualModel
      params:
        base:
          class: sklearn.cross_decomposition.PLSRegression
          params: {n_components: 3}
        learner:
          class: sklearn.linear_model.Ridge
          params: {alpha: 1.0}
        gate: false
```

:::

:::{tab-item} Python
:sync: python

```python
import tempfile
from pathlib import Path
import numpy as np
import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.models import ResidualModel
from nirs4all.operators.transforms import SNV

rng = np.random.default_rng(17)
spectra = rng.normal(size=(48, 31))
markers = rng.normal(size=(48, 3))
y = (2 * spectra[:, 10] - spectra[:, 20] + markers[:, 0]
     + rng.normal(scale=0.1, size=48))

dataset = SpectroDataset("doc_multimodal_residual")
dataset.add_samples(
    [spectra[:40], markers[:40]], {"partition": "train"},
    headers=[[str(i) for i in range(31)],
             ["marker_0", "marker_1", "marker_2"]],
)
dataset.add_samples([spectra[40:], markers[40:]], {"partition": "test"})
dataset.add_targets(y.reshape(-1, 1))

pipeline = [
    KFold(n_splits=3, shuffle=True, random_state=17),
    {"branch": {"by_source": True, "steps": {
        "source_0": [SNV()],
        "source_1": [StandardScaler()],
    }}},
    {"merge": {"sources": "concat"}},
    {"model": ResidualModel(
        base=PLSRegression(n_components=3),
        learner=Ridge(alpha=1.0), gate=False,
    )},
]

with tempfile.TemporaryDirectory() as workspace:
    result = nirs4all.run(
        pipeline=pipeline, dataset=dataset, engine="dag-ml", refit=True,
        workspace_path=workspace, save_charts=False, verbose=0,
    )
    archive = result.export(Path(workspace) / "multimodal-residual.n4a")
    # Supply raw columns in the original source order; replay owns preprocessing.
    raw_new = np.column_stack([spectra[40:], markers[40:]])
    replay = nirs4all.predict(archive, raw_new)
    assert len(replay.y_pred) == 8
    assert np.isfinite(replay.y_pred).all()
    print(result.execution_engine, result.best_rmse)
    result.close()
```

:::

:::{tab-item} R
:sync: r

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

:::{tab-item} Octave / MATLAB
:sync: matlab

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

:::{tab-item} WASM / JavaScript
:sync: javascript

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

::::

**Expected result:** the final exported predictor keeps source preprocessing, the PLS base and the Ridge correction together. Replay consumes raw source columns in their original order, and returns eight finite predictions. The two sources must describe the same observations; real files should be joined by sample IDs before loading.

### Run all three advanced checkpoints

Download {download}`D08 documented workflows <../../../examples/developer/01_advanced_pipelines/D08_documented_multisource_stacking.py>`. It constructs synthetic data and runs feature fusion, stacking, and multisource residual learning, including export/replay checks. The final recipe is also available as {download}`YAML <../../../examples/configs/pipelines/documented_multisource_residual.yaml>` and {download}`JSON <../../../examples/configs/pipelines/documented_multisource_residual.json>`.

The examples demonstrate the data flow. They are not evidence that a more complex model improves a real spectroscopy problem. Follow {doc}`evaluation` to choose the test that answers your deployment question.

**Checkpoint:** explain why a generator makes separate candidates, a feature branch makes more input columns, and stacking makes prediction columns. Draw the final model's inputs before proceeding.
