# Repetitions: several measurements are still one physical sample

You scan each of 30 materials four times. The file has 120 rows, but the experiment
has 30 independent materials. A random row split could put one material's scans
in both training and validation, producing misleadingly easy predictions.

If you only need safe splitting and one reported prediction per material, start
with dataset `repetition: Sample_ID` and {doc}`/user_guide/data/aggregation`.
The nodes here instead **reshape** repeated measurements into input blocks.

## Choose where repetitions belong

```{figure} /assets/guide/repetitions.svg
:alt: 120 scan rows from 30 materials with four scans each become four 30-row sources or one 30-row source with four views.
:width: 100%

Repetition reshaping changes the row meaning from scan to physical material. Thirty materials remain thirty independent materials.
```


| Node | Output for 30 materials × four scans × 31 features | Choose when |
|---|---|---|
| `rep_to_sources` | Four sources, each 30 × 31 | Each repetition needs its own source path |
| `rep_to_pp` | One source, 30 × 4 × 31 | A view-aware model should retain repeated scans |
| `rep_fusion` | An explicit relation-aware representation | Sources have unequal counts or missing observations |

## Worked example: one feature block per repetition

The Python example creates 24 training materials and six held-out materials,
each with four scans. Every material's scans stay in the same partition.
Reshape **before** the splitter because reshaping changes row identities.

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
      "rep_to_sources": {
        "column": "Sample_ID",
        "expected_reps": 4,
        "on_unequal": "error",
        "source_names": "rep_{i}"
      }
    },
    {
      "merge": {
        "sources": "concat"
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
- rep_to_sources:
    column: Sample_ID
    expected_reps: 4
    on_unequal: error
    source_names: rep_{i}
- merge:
    sources: concat
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
base = rng.normal(size=(30, 31))
X = np.repeat(base, 4, axis=0) + rng.normal(scale=0.05, size=(120, 31))
y = np.repeat(2 * base[:, 10] - base[:, 20], 4)
ids = np.repeat(np.arange(30), 4)
dataset = SpectroDataset("four_scans_per_material")
dataset.add_samples(X[:96], {"partition": "train"})
dataset.add_samples(X[96:], {"partition": "test"})
dataset.add_metadata(ids.reshape(-1, 1), headers=["Sample_ID"])
dataset.add_targets(y.reshape(-1, 1))

pipeline = [
    {"rep_to_sources": {
        "column": "Sample_ID", "expected_reps": 4,
        "on_unequal": "error", "source_names": "rep_{i}",
    }},
    {"merge": {"sources": "concat"}},
    KFold(n_splits=3, shuffle=True, random_state=17),
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


**What to expect:** 120 scan rows become 30 material rows. Four 31-column sources
are joined into 124 columns. CV now splits the 24 training materials, not the
96 scans. The six held-out materials receive six predictions.
`rep_0` means the first stored scan of each material; it does not mean instrument 0.

The historical `rep_to_sources` and `rep_to_pp` controllers skip reshaping during
prediction. New input must already have the trained source/view representation.
Do not submit a fresh 120-row scan file and assume inference will group it.

## Keep scans on a view axis instead

This fragment needs the same repetition-aware dataset. It creates four named
views per material. A matrix learner flattens four equal-width views to 124
columns; a tensor model can preserve their separate axis.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "pipeline": [
    {
      "rep_to_pp": {
        "column": "Sample_ID",
        "expected_reps": 4,
        "on_unequal": "error",
        "pp_names": "{pp}_scan{i}"
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
- rep_to_pp:
    column: Sample_ID
    expected_reps: 4
    on_unequal: error
    pp_names: '{pp}_scan{i}'
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

pipeline = [{'rep_to_pp': {'column': 'Sample_ID',
                'expected_reps': 4,
                'on_unequal': 'error',
                'pp_names': '{pp}_scan{i}'}},
 {'split': KFold(n_splits=3, shuffle=True, random_state=17)},
 {'model': PLSRegression(n_components=3)}]
```
:::

::::


**Expected result:** one source, 30 material rows, four views, 31 features per
view. This is a structural transformation, not an average over scans.

## Enumerate repetition options

| Field | Choices / default | Meaning |
|---|---|---|
| `column` | Metadata name; omitted uses dataset aggregation column | Which scans describe the same material |
| `expected_reps` | Positive integer; omitted infers count | Validate your acquisition protocol |
| `on_unequal` | `error` (default), `pad`, `drop`, `truncate` | Policy for unequal counts |
| `source_names` | Template `rep_{i}` or explicit list | Labels for `rep_to_sources` |
| `pp_names` | Template with `{i}` and/or `{pp}` | Labels for `rep_to_pp`; lists unsupported |
| `preserve_order` | True by default | Stored row order determines repetition slots |
| `aggregate_metadata` | `first`, `validate`, `drop` | Handle metadata differences within a material |

## Unequal counts are a scientific choice

For counts 4, 4 and 3 across three materials:

| Policy | What happens | Consequence |
|---|---|---|
| `error` | Stop and report mismatch | You can inspect the acquisition issue |
| `pad` | Add missing values to the short group | Model must handle missing input |
| `drop` | Remove incomplete groups | Population changes |
| `truncate` | Keep three scans per group | Two acquired scans are discarded |

Start with `error`. Do not silently pad or drop observations just to make array
shapes agree. Grouping by equal target values (`column: y`) exists historically,
but different materials can share a concentration and new targets are unknown.
Use a physical-sample identifier.

## When `rep_fusion` is appropriate

A NIR source may have four scans while a chemistry source has one assay. An
ordinary scan-row matrix cannot express that relationship safely. `rep_fusion`
materializes a representation from **raw relation-aware multisource data**.
It requires `RawMultiSourceDataset` staging and a `RepresentationPlan`; the
ordinary `SpectroDataset` used above is deliberately rejected.

A representation selection fragment is:

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "rep_fusion": {
    "representation": "per_source_aggregate"
  }
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
rep_fusion:
  representation: per_source_aggregate
```
:::

:::{tab-item} Python
:sync: python

```python
fusion_step = {'rep_fusion': {'representation': 'per_source_aggregate'}}
```
:::

::::


**Expected result:** the staged source observations are aggregated per physical
sample with lineage retained. `masked_stack` preserves slots and presence masks;
other supported representations and source policies are explained in
{doc}`/user_guide/data/heterogeneous_repetitions`. Run this materialization before
branching, and retain its plan for prediction. Do not combine it with another
repetition reshaping mechanism in the same recipe.

Implementation-backed examples:
[repetition tests](https://github.com/nirs4all/nirs4all/blob/main/tests/unit/controllers/data/test_repetition.py),
[relation-fusion tests](https://github.com/nirs4all/nirs4all/blob/main/tests/unit/controllers/data/test_rep_fusion.py).
