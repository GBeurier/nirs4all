# Generators: turn an experiment plan into concrete recipes

You have spectra and a measured concentration. Should you normalize each spectrum
with SNV, or standardize each wavelength across samples? Should PLS retain two,
three or four components? A **generator** writes those choices once and creates
one ordinary pipeline for each combination. Choose the final recipe from
validation results.

The generator creates experiments to compare. It does not average predictions,
add observations or join feature columns.

## Read the expansion before running it

```{figure} /assets/guide/generators.svg
:alt: Two preprocessing choices crossed with three PLS component counts create six independent pipelines.
:width: 100%

Two choices × three choices = six recipes. Each recipe gets the same data and folds, so only preprocessing and PLS complexity change.
```


| Recipe | Preprocessing | PLS components | What changes? |
|---|---|---:|---|
| 1 | SNV | 2 | Normalize each spectrum; a small latent model |
| 2 | SNV | 3 | Same normalized spectra; one more component |
| 3 | SNV | 4 | Same normalized spectra; two more components |
| 4 | StandardScaler | 2 | Standardize each wavelength; a small latent model |
| 5 | StandardScaler | 3 | Same standardized data; one more component |
| 6 | StandardScaler | 4 | Same standardized data; two more components |

SNV removes each spectrum's mean and rescales that spectrum's variation.
StandardScaler instead learns a mean and scale for **each feature** from training
observations. PLS components summarize relationships between features and the
measured target. More components increase flexibility and can increase overfitting.

## Try the six-recipe experiment

The Python tab creates 48 synthetic observations with 31 features, including an
8-observation holdout. Save either configuration as `pipeline.json` or
`pipeline.yaml`; the Python tab is a complete standalone example. These data
illustrate software behavior, not instrument performance.

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
      "_or_": [
        {
          "class": "nirs4all.operators.transforms.StandardNormalVariate"
        },
        {
          "class": "sklearn.preprocessing.StandardScaler"
        }
      ]
    },
    {
      "model": {
        "class": "sklearn.cross_decomposition.PLSRegression",
        "params": {
          "n_components": {
            "_range_": [
              2,
              4
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
- split:
    class: sklearn.model_selection.KFold
    params:
      n_splits: 3
      shuffle: true
      random_state: 17
- _or_:
  - class: nirs4all.operators.transforms.StandardNormalVariate
  - class: sklearn.preprocessing.StandardScaler
- model:
    class: sklearn.cross_decomposition.PLSRegression
    params:
      n_components:
        _range_:
        - 2
        - 4
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
from nirs4all.operators.transforms import SNV
from nirs4all.pipeline.config import PipelineConfigs

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
    {"_or_": [SNV(), StandardScaler()]},
    {"model": {
        "class": PLSRegression,
        "params": {"n_components": {"_range_": [2, 4]}},
    }},
]
configs = PipelineConfigs(pipeline)
assert len(configs.steps) == 6
for number, steps in enumerate(configs.steps, 1):
    transform = steps[1]
    name = transform if isinstance(transform, str) else transform["class"]
    print(number, name.split(".")[-1],
          steps[2]["model"]["params"]["n_components"])
with tempfile.TemporaryDirectory() as workspace:
    result = nirs4all.run(pipeline, dataset, engine="legacy",
                         workspace_path=workspace, save_charts=False, verbose=0)
    print("Best validation score:", result.cv_best_score)
    print("Selected model test RMSE:", result.best_rmse)
    result.close()
```
:::

::::


**What to expect:** the preview prints the six rows listed above. Every recipe
keeps 48 observation identities and 31 features. Three CV folds × six candidates
require 18 fold model fits, before optional final refit. The best score depends
on the data; the first candidate is not automatically the winner.

## Choose the generator that matches your question

| Your question | Use | Worked explanation |
|---|---|---|
| Which one of these transforms? | `_or_` | {ref}`generator-keyword-or` |
| How many PLS components? | `_range_` | {ref}`generator-keyword-range` |
| Which strength over several orders of magnitude? | `_log_range_` | {ref}`generator-keyword-log-range` |
| Every combination of model parameters? | `_grid_` | {ref}`generator-keyword-grid` |
| Only matched parameter pairs? | `_zip_` | {ref}`generator-keyword-zip` |
| Every scatter/smoothing/derivative combination? | `_cartesian_` | {ref}`generator-keyword-cartesian` |
| A few randomly drawn values? | `_sample_` | {ref}`generator-keyword-sample` |
| A deliberate order of candidate recipes? | `_chain_` | {ref}`generator-keyword-chain` |
| Unordered sets of parallel views? | `pick` | {ref}`generator-keyword-pick` |
| Ordered sequences of transformations? | `arrange` | {ref}`generator-keyword-arrange` |
| A reproducible subset of a large search? | `count`, `_seed_` | {ref}`generator-keyword-count` |
| Prevent inappropriate combinations? | `_mutex_`, `_requires_`, `_exclude_` | {ref}`generator-keyword-mutex` |

## Alternatives, branches or extra views?

| What you want | Result | Node |
|---|---|---|
| Compare SNV **against** smoothing | Two independent candidate recipes | Generator `_or_` |
| Give one model SNV **and** smoothed features | One wider matrix | {doc}`concat_transform` or {doc}`branch` + {doc}`merge` |
| Keep raw, SNV and derivative as named views | Three representations of the same samples | {doc}`feature_augmentation` |
| Fit base models, then combine their predictions | One stacking workflow | {doc}`branch` + prediction {doc}`merge` |

A generator **inside a duplication branch** can create more simultaneous
branches. At pipeline level it creates independently evaluated variants. Learn
the top-level example before nesting searches into branches.

## Common mistakes

1. **Counting sums instead of products.** Two transforms and three component
   counts produce six recipes. Four additional smoothing choices produce 24.
2. **Treating `_range_` like Python's `range`.** The upper bound is inclusive:
   `[2, 4]` means 2, 3 and 4.
3. **Using `arrange` for parallel views.** Three operations taken two at a time
   give three unordered pairs with `pick`, six ordered pairs with `arrange`.
4. **Assuming `_zip_` rejects unequal lengths.** It stops at the shortest list.
   Keep lists equally long to avoid losing planned candidates.
5. **Calling `_chain_` a transform chain.** It enumerates candidates in order.
   Use a list in `preprocessing` to apply SNV and then smoothing.
6. **Picking with the test set.** Preserve an untouched final evaluation cohort.

Continue with {doc}`/reference/generator_keywords` for exact inputs and outputs.
Runnable source: [D01 generator syntax](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/02_generators/D01_generator_syntax.py).
