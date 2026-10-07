# 10. Run the same recipe in your language

**Your goal:** fit StandardScaler → Ridge, reload its learned state, and compare predictions in Python, R, Octave or WASM. Use the twelve-row download and `ridge.recipe.json` from {doc}`start`.

You do not have to learn Python to follow the shared workflow. JSON/YAML express the recipe. Your language provides file access, arrays and the actual execution call. The native methods implement the numerical operations.

## Understand what is shared

| Item | Same scientific meaning | Language-specific detail |
|---|---|---|
| Dataset | Rows are observations; columns are ordered features | Lists, matrices, structs or a typed dataset |
| Recipe | Standardize features, compare two Ridge alphas | JSON parsing and function spelling |
| Fitting | Training-fold learning, validation, selection, refit | Python wheel's embedded dispatcher, other CPU hosts' CLI, or browser WASM |
| Prediction | Apply saved learned state to new inputs | CPU package loader or browser package loader |
| Result | Values attached to sample IDs and target names | Dictionary/list/struct/object access |

## Fit the two candidates

In the JSON/YAML tabs, `steps` are ordered nodes and `candidates` are two separate fits. Save the JSON as `ridge.recipe.json` and run the language tab. If you prefer YAML, parse it to the same mapping before the API call; `run_pipeline` does not promise to read a YAML filename automatically.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "steps": [
    {
      "method_id": "preprocessing.scaling.standard_scale",
      "role": "transformer",
      "params": {}
    },
    {
      "method_id": "models.regularized.ridge",
      "role": "regressor",
      "params": {
        "scale_x": false
      }
    }
  ],
  "candidates": [
    {
      "alpha": 0.1
    },
    {
      "alpha": 1.0
    }
  ]
}
```

:::

:::{tab-item} YAML
:sync: yaml

```yaml
steps:
  - method_id: preprocessing.scaling.standard_scale
    role: transformer
    params: {}
  - method_id: models.regularized.ridge
    role: regressor
    params: {scale_x: false}
candidates:
  - {alpha: 0.1}
  - {alpha: 1.0}
```

:::

:::{tab-item} Python
:sync: python

```python
import json
from pathlib import Path
from nirs4all_core import run_pipeline

data = json.loads(Path("dataset.json").read_text())
recipe = json.loads(Path("ridge.recipe.json").read_text())
model = run_pipeline(data, recipe)
model.export("ridge.native.json")
print(model.outcome["effective_plan"]["variants"])
```

:::

:::{tab-item} R
:sync: r

```r
library(nirs4all)
data <- jsonlite::fromJSON("dataset.json", simplifyVector = FALSE)
recipe <- jsonlite::fromJSON("ridge.recipe.json", simplifyVector = FALSE)
model <- nirs4all_run_pipeline(data, recipe)
nirs4all_pipeline_export(model, "ridge.native.json")
```

:::

:::{tab-item} Octave / MATLAB
:sync: matlab

```matlab
model = nirs4all.runPipeline(fileread('dataset.json'), ...
                            fileread('ridge.recipe.json'));
model.export('ridge.native.json');
```

:::

:::{tab-item} WASM / JavaScript
:sync: javascript

```javascript
import {runBrowserPipeline} from 'nirs4all';

const data = await (await fetch('./dataset.json')).json();
const pipeline = await (await fetch('./ridge.recipe.json')).json();
const model = await runBrowserPipeline(data, {pipeline});
localStorage.setItem('ridge-model', model.export());
console.log(model.outcome.effective_plan.variants);
```

:::

::::

Standardization is fitted once in each training fold, then the candidate model is fitted on its transformed training inputs. Selection chooses the smaller held-out error; refit uses the selected settings on the complete training cohort. The selected alphas are not averaged.

## Reload and predict new observations

Use the downloadable `predict.json` from {doc}`start`. Its two rows have the same seven-feature order as training and no targets. The JSON/YAML tabs below illustrate the **shape** of a one-row inference document, with arbitrary values; the execution tabs use the actual downloaded two-row file.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "x": [
    [
      0.1,
      0.2,
      0.3,
      0.4,
      0.5,
      0.6,
      0.7
    ]
  ],
  "sample_ids": [
    "new:0"
  ]
}
```

:::

:::{tab-item} YAML
:sync: yaml

```yaml
x:
  - [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
sample_ids: ["new:0"]
```

:::

:::{tab-item} Python
:sync: python

```python
import json
from pathlib import Path
from nirs4all_core import NativePipeline

model = NativePipeline.load("ridge.native.json")
fresh = json.loads(Path("predict.json").read_text())
output = model.predict(fresh["x"], sample_ids=fresh["sample_ids"])
block = output["outputs"][0]["predictions"][0]
assert block["sample_ids"] == fresh["sample_ids"]
assert {event["phase"] for event in output["lineage"]} == {"PREDICT"}
print(block)
```

:::

:::{tab-item} R
:sync: r

```r
library(nirs4all)
model <- nirs4all_pipeline_load("ridge.native.json")
fresh <- jsonlite::fromJSON("predict.json")
output <- nirs4all_pipeline_predict(model, fresh$x,
                                   sample_ids = fresh$sample_ids)
print(output$outputs[[1]]$predictions[[1]])
```

:::

:::{tab-item} Octave / MATLAB
:sync: matlab

```matlab
model = nirs4all.NativePipeline.load('ridge.native.json');
fresh = jsondecode(fileread('predict.json'));
output = model.predict(fresh.x, 'sampleIds', fresh.sample_ids);
disp(output.outputs);
```

:::

:::{tab-item} WASM / JavaScript
:sync: javascript

```javascript
import {loadBrowserPipeline} from 'nirs4all';
import {dataset} from '@nirs4all/io-wasm/public-dataset';

const model = await loadBrowserPipeline(localStorage.getItem('ridge-model'));
const training = await (await fetch('./dataset.json')).json();
const fresh = await (await fetch('./predict.json')).json();
// Preserve the trained wavelength/source declarations, replace sample values.
const input = structuredClone(dataset(training).toJSON());
input.origin_ids = [...fresh.sample_ids];
input.fold_ids = fresh.sample_ids.map(() => null);
input.dataset.sample_ids = [...fresh.sample_ids];
input.dataset.y = null;
input.dataset.groups = null;
input.dataset.partitions = {dtype: '<U7', shape: [fresh.x.length],
  values: fresh.sample_ids.map(() => 'predict')};
const source = input.dataset.sources[0];
source.sample_ids = [...fresh.sample_ids];
source.array.shape[0] = fresh.x.length;
source.array.values = fresh.x;
const output = await model.predict(input);
console.log(output.outputs[0].predictions[0]);
```

:::

::::

**Expected result:** the returned block has two sample IDs and two prediction rows. In Python, the assertion confirms a prediction-only replay. The values should match a replay of the same exported model in a compatible consumer, within a stated numerical tolerance.

### Why the WASM tab has more dataset code

The browser consumer takes a typed dataset, preserving the trained source, feature names, coordinates and units. The example copies those declarations and replaces sample IDs, values and partitions. It removes y because new measurements have no observed labels. This procedure is appropriate for the complete, single-source fixture; richer datasets should be assembled through IO with explicit source-presence masks.

CPU export and browser export use their corresponding loaders. Similar method IDs do not mean that these two package formats are interchangeable. See {doc}`deployment` for supported model transport.

## Keep matrices as matrices

| Host | Practical rule | Typical failure |
|---|---|---|
| Python | X has shape `(observations, features)` | A one-dimensional row loses the sample axis |
| R | Subset rows with `drop = FALSE` | A one-row matrix becomes a vector |
| Octave/MATLAB | Rows are observations, columns are features | A transposed matrix changes model inputs |
| JavaScript | Keep nested row arrays or an explicit typed shape | Flattened numbers lose their row boundaries |
| JSON/YAML | Keep `steps` and `candidates` as arrays, even with one item | A singleton object replaces an array |

R's `simplifyVector = FALSE` preserves arrays of recipe objects as lists. For Octave struct construction, use cell arrays for `steps` and `candidates`. In all languages, preserve booleans as booleans: `false` is different from the string `"false"`.

## A Node CPU consumer

For a CPU package produced in another language, Node uses `runPipeline` and `NativePipeline`. The loader takes **exact JSON text**, retaining large integer fingerprints without parsing/re-encoding them. This is a CPU example; it requires Node and the Core CLI.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

This box documents the Node CPU API. Python, R and Octave equivalents are in the reload box above. JSON/YAML describe inputs rather than executing the loader.

:::

:::{tab-item} YAML
:sync: yaml

This box documents the Node CPU API. Python, R and Octave equivalents are in the reload box above. JSON/YAML describe inputs rather than executing the loader.

:::

:::{tab-item} Python
:sync: python

This box documents the Node CPU API. Python, R and Octave equivalents are in the reload box above. JSON/YAML describe inputs rather than executing the loader.

:::

:::{tab-item} R
:sync: r

This box documents the Node CPU API. Python, R and Octave equivalents are in the reload box above. JSON/YAML describe inputs rather than executing the loader.

:::

:::{tab-item} Octave / MATLAB
:sync: matlab

This box documents the Node CPU API. Python, R and Octave equivalents are in the reload box above. JSON/YAML describe inputs rather than executing the loader.

:::

:::{tab-item} WASM / JavaScript
:sync: javascript

```javascript
import {readFile} from 'node:fs/promises';
import {runPipeline, NativePipeline} from 'nirs4all';

const readJSON = async path => JSON.parse(await readFile(path, 'utf8'));
const model = await runPipeline(await readJSON('dataset.json'),
                                await readJSON('ridge.recipe.json'));
await model.export('ridge.native.json');
const loaded = await NativePipeline.load(await readFile('ridge.native.json', 'utf8'));
const fresh = await readJSON('predict.json');
const output = await loaded.predict(fresh.x, {sampleIds: fresh.sample_ids});
console.log(output.outputs[0].predictions[0]);
```

:::

::::

## Before using more operators

The shared finite recipe has independently compared StandardScaler → Ridge and raw PLS examples. Check the {doc}`operator catalogue </reference/operator_catalog>` before adding a method, role, mask or model family. The {doc}`node catalogue </reference/nodes/index>` links user explanations to supported runtime details.

Advanced Python recipes use actual sklearn-compatible operators. A Python class path cannot import sklearn inside R or WASM. Use a supported native method recipe, or keep that model in its documented Python host.

Implementation witnesses: [Python native pipeline tests](https://github.com/GBeurier/nirs4all-core/blob/main/bindings/python/tests/test_native_pipeline.py), [R facade](https://github.com/GBeurier/nirs4all-r/blob/main/R/native_pipeline.R), [Octave NativePipeline](https://github.com/GBeurier/nirs4all-core/blob/main/bindings/matlab/%2Bnirs4all/NativePipeline.m), and [browser replay tests](https://github.com/GBeurier/nirs4all-core/blob/main/bindings/wasm/tests/browser-native-pipeline.test.js).

**Checkpoint:** run the same data/recipe in your preferred host, reload the model and keep values with their sample IDs. Continue to {doc}`tutorials` for the advanced learning sequence.
