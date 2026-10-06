# 1. Install and run your first workflow

A first workflow loads a dataset, evaluates candidate models on held-out folds, selects one candidate, refits it on the training cohort and saves a predictor. New-data prediction uses the saved predictor without fitting again.

## Install your product

::::{tab-set}
:sync-group: language

:::{tab-item} Python SDK
:sync: python

```bash
python -m pip install nirs4all
nirs4all --test-install
```

Follow {doc}`/getting_started/quickstart` for general Python pipelines. Install optional model backends only for the controllers you use.
:::

:::{tab-item} Python Core
:sync: core-python

```bash
python -m pip install 'nirs4all-core[all]'
```

The native workflow also needs the matching Core CLI and Methods shared library.
:::

:::{tab-item} R
:sync: r

```r
install.packages("nirs4all", repos = c(
  "https://gbeurier.r-universe.dev", "https://cloud.r-project.org"))
library(nirs4all)
nirs4all_upstreams()
```

Install the native upstream bindings and Core CLI for the native workflow.
:::

:::{tab-item} JS/TS/WASM
:sync: javascript

```bash
npm install nirs4all @nirs4all/methods @nirs4all/io-wasm dag-ml-wasm
```

Use Node 20 or later for the Node example. Dataset construction is asynchronous; await `dataset` before passing its result to a workflow. Initialize WASM once per process or worker. Browser and Node hosts use the same public imports; loading and file access differ.
:::

:::{tab-item} MATLAB/Octave
:sync: matlab

Download the matching MATLAB/Octave release ZIP and add its root to your path. Install Methods and the native CLI independently. The ZIP contains the namespace and standalone tests. Licensed MATLAB and Octave qualification are reported separately.
:::

:::{tab-item} Rust / CLI
:sync: cli

```bash
cargo install --locked nirs4all --bin nirs4all-core-archive
```

The SDK CLI additionally requires the Python `nirs4all` package. Use the matching release cohort when testing artifact transport.
:::
::::

## Configure the native runtime

For CPU examples set `NIRS4ALL_CORE_CLI` to the installed `nirs4all-core-archive` executable and `N4M_LIBRARY_PATH` to the installed Methods shared library. Use absolute paths. These paths describe your runtime; they are not part of a portable dataset or model recipe. Do not point an installed-package tutorial at an undeclared sibling checkout.

## A common dataset

Download {download}`dataset.json <../_downloads/dense-workflow.dataset.json>` and {download}`predict.json <../_downloads/dense-workflow.predict.json>` into an empty directory. The data are synthetic: twelve training observations, seven spectral features and one numeric target. They demonstrate the contract, not scientific performance. Prediction has two new sample IDs and no targets.

::::{tab-set}
:sync-group: language

:::{tab-item} Python Core
:sync: core-python

```python
import json
from pathlib import Path
import nirs4all_core as core

workflow = core.run(core.dataset(Path("dataset.json")),
                    components=[1, 2], archive="model.n4a")
core.export(workflow, "saved-workflow")
loaded = core.load("saved-workflow")
new = json.loads(Path("predict.json").read_text())
prediction = core.predict(loaded, new["x"], sample_ids=new["sample_ids"])
print(prediction)
```
:::

:::{tab-item} R native
:sync: r

```r
library(nirs4all)
workflow <- nirs4all_native_run(
  nirs4all_dataset("dataset.json"), "model.n4a", components = c(1L, 2L))
nirs4all_native_export(workflow, "saved-workflow")
loaded <- nirs4all_native_load("saved-workflow")
new <- jsonlite::fromJSON("predict.json")
print(nirs4all_native_predict(loaded, new$x, sample_ids = new$sample_ids))
```
:::

:::{tab-item} JS/TS/WASM
:sync: javascript

```javascript
import {readFile, writeFile} from 'node:fs/promises';
import {dataset, run, predict, exportWorkflow, load} from 'nirs4all';

const training = await dataset(JSON.parse(await readFile('dataset.json', 'utf8')));
const workflow = await run(training, {components: [1, 2]});
await writeFile('saved-workflow.json', JSON.stringify(exportWorkflow(workflow)));
const loaded = await load(JSON.parse(await readFile('saved-workflow.json', 'utf8')));
const fresh = JSON.parse(await readFile('predict.json', 'utf8'));
const predictionData = await dataset({spectra: fresh.x}, {sampleIds: fresh.sample_ids});
console.log(await predict(loaded, predictionData));
```

This is a Node example; use `fetch` and browser storage for the same JSON values in a browser. The JSON returned by `exportWorkflow` preserves the workflow record and archive bytes. Persist it using the host's file/storage API. It is not the CPU export directory layout.
:::

:::{tab-item} MATLAB/Octave
:sync: matlab

```matlab
workflow = nirs4all.run(nirs4all.dataset('dataset.json'), ...
    'components', [1 2], 'archive', 'model.n4a');
nirs4all.export(workflow, 'saved-workflow');
loaded = nirs4all.load('saved-workflow');
new = jsondecode(fileread('predict.json'));
prediction = nirs4all.predict(loaded, new.x, 'sampleIds', new.sample_ids);
```
:::

:::{tab-item} CLI
:sync: cli

```bash
nirs4all workflow run dataset.json --components 1 2 --archive model.n4a
nirs4all workflow predict --archive model.n4a --input predict.json
```

`workflow run` writes the selected-model archive directly. `workflow export` separately reads and copies an exported workflow directory created through the language export API.
:::
::::

## What to inspect

Check candidate scores and the selected variant, OOF sample identities and the refit artifact. Save the outcome with the experiment when you need to compare later runs. Compare predictions after reload, not just the file's existence. See {doc}`evaluation`, {doc}`results` and {doc}`deployment`.

For a first multimodal workflow, follow {doc}`/user_guide/data/methods_multimodal_u07`; it introduces NIR, image, series and metadata sources, learned encoders, early fusion and portable native state. The shape/missing-data profile is explicit in {doc}`datasets`.
