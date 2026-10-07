# `auto_transfer_preproc`: choose preprocessing for an instrument change

A calibration built on one instrument may encounter a shifted baseline or scale
on another. The same concentration can then appear spectrally different.
`auto_transfer_preproc` compares preprocessing candidates using source and target
cohorts and recommends a representation that better aligns the domains while
preserving useful signal.

This selects preprocessing. It does not guarantee that instruments become
interchangeable or that concentration predictions improve.

## Read the adaptation experiment

```{figure} /assets/guide/auto_transfer_preproc.svg
:alt: Source-domain and target-adaptation spectra enter preprocessing selection; one saved recommendation is applied before the learner.
:width: 100%

The target cohort participates in selection. Reserve a separate untouched cohort for an independent transfer-performance measurement.
```


The source domain is the calibration instrument. The target domain is the new
instrument. A target cohort used to choose preprocessing becomes an **adaptation
cohort**, even if its dataset partition is called `test`.

## Worked example: select one representation

The synthetic example uses 40 source observations and eight target-adaptation
observations. Their spectra are multiplied by 1.2 and shifted by 0.4 while the
underlying concentrations stay fixed, simulating an instrument change. Its
printed score is descriptive for this demonstration: those
eight rows are involved in selection and are not an independent final test.

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
      "auto_transfer_preproc": {
        "preset": "fast",
        "source_partition": "train",
        "target_partition": "test",
        "apply_recommendation": true,
        "top_k": 1,
        "use_augmentation": false,
        "n_components": 3,
        "verbose": 0
      }
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
- auto_transfer_preproc:
    preset: fast
    source_partition: train
    target_partition: test
    apply_recommendation: true
    top_k: 1
    use_augmentation: false
    n_components: 3
    verbose: 0
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
import tempfile
import numpy as np
import nirs4all
from nirs4all.data.dataset import SpectroDataset
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression

rng = np.random.default_rng(17)
X = rng.normal(size=(48, 31))
y = 2 * X[:, 10] - X[:, 20] + rng.normal(scale=0.1, size=48)
# Simulate a new instrument: concentrations stay fixed, spectra shift.
X[40:] = 1.2 * X[40:] + 0.4
dataset = SpectroDataset("worked_node")
dataset.add_samples(X[:40], {"partition": "train"},
                    headers=[str(i) for i in range(31)])
dataset.add_samples(X[40:], {"partition": "test"})
dataset.add_targets(y.reshape(-1, 1))


pipeline = [
    {"auto_transfer_preproc": {
        "preset": "fast",
        "source_partition": "train", "target_partition": "test",
        "apply_recommendation": True, "top_k": 1,
        "use_augmentation": False, "n_components": 3, "verbose": 0,
    }},
    KFold(n_splits=3, shuffle=True, random_state=17),
    {"model": PLSRegression(n_components=3)},
]
with tempfile.TemporaryDirectory() as workspace:
    result = nirs4all.run(pipeline, dataset, engine="legacy",
                         workspace_path=workspace, save_charts=False, verbose=0)
    print("CV validation score:", result.cv_best_score)
    print("Target-adaptation RMSE (not an independent test):", result.best_rmse)
    result.close()
```
:::

::::


**What to expect:** the selector ranks candidate preprocessing, applies its
recommendation, and then PLS predicts in that selected feature space. There are
still 48 observations. The chosen transform and its output width depend on the
candidates and data. No fixed winner or numerical improvement is promised.
To inspect recommendations without changing model input, set
`apply_recommendation: false` and inspect the analysis results.

## Enumerate selection presets

| Preset | What it adds | Choose when |
|---|---|---|
| `fast` | Single preprocessing candidates | First diagnosis; short runtime |
| `balanced` | Sequential preprocessing combinations | Single transforms are insufficient |
| `thorough` | Combination and feature augmentation analysis | Several views may be useful |
| `full` | All stages, including supervised validation | Appropriate labels and validation protocol exist |
| `exhaustive` | Deeper search | A planned research/benchmark study |

Here, selector **stacking** means combinations of preprocessing operations;
it should not be confused with prediction stacking in {doc}`merge`.

## Options that change your experiment

| Field | Default | What you choose |
|---|---|---|
| `preset` | `fast` | Search effort/stages |
| `source_partition` | `train` | Source-domain observations |
| `target_partition` | `test` | Target-adaptation observations |
| `apply_recommendation` | True | Apply, or only record recommendation |
| `top_k` | 1 | Number of selected recommendations |
| `use_augmentation` | False | Retain several selected feature representations |
| `n_components` | 10 | PCA size used in transfer metrics |
| `preprocessing_spec` | Unset | Your generator-defined candidate search |
| `metric_weights` | Unset | Relative influence of transfer metrics |
| `verbose` | 1 | 0 silent, 1 progress, 2 details |

Overrides `run_stage2`, `stage2_top_k`, `run_stage3` and `run_stage4` control stages.
Inspect the stage's label requirements before enabling it. A metric measuring
closer distributions is not the same as error on concentration prediction.

## Retain two recommendations as views

This changes the output feature representation; a matrix learner sees a wider
input when several views are retained.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "pipeline": [
    {
      "auto_transfer_preproc": {
        "preset": "balanced",
        "top_k": 2,
        "use_augmentation": true,
        "apply_recommendation": true
      }
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
- auto_transfer_preproc:
    preset: balanced
    top_k: 2
    use_augmentation: true
    apply_recommendation: true
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

pipeline = [{'auto_transfer_preproc': {'preset': 'balanced',
                            'top_k': 2,
                            'use_augmentation': True,
                            'apply_recommendation': True}},
 {'split': KFold(n_splits=3, shuffle=True, random_state=17)},
 {'model': PLSRegression(n_components=3)}]
```
:::

::::


**Expected result:** the two selected preprocessing recommendations contribute
views to the downstream learner. Their dimensions depend on the selected
operators. The saved recommendation is reused on new observations; inference
does not rerun the selection for every batch.

## Common mistakes

- Calling target-adaptation rows an untouched test set after using them to choose
  preprocessing. Hold another cohort apart for final evaluation.
- Assuming every stage is unsupervised. The larger presets can use targets.
- Using older `candidates` or `metric: distance_reduction` snippets. The public
  selector configuration here uses `preprocessing_spec` and `metric_weights`.
- Treating distribution alignment as proof of analyte prediction accuracy.

Runnable selector analysis:
[D01 transfer analysis](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/04_transfer_learning/D01_transfer_analysis.py).
Pipeline/replay evidence:
[auto-transfer parity tests](https://github.com/GBeurier/nirs4all/blob/main/tests/integration/parity/test_auto_transfer_preproc.py).
Continue with {doc}`/user_guide/deployment/retrain_transfer` for a full adaptation protocol.
