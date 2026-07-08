# Hello World: One Dataset File, One Pipeline File

This is the shortest useful NIRS4ALL workflow. A dataset is described in YAML or JSON. A pipeline is described in YAML or JSON. NIRS4ALL runs both files and returns scores, predictions, and an exportable model bundle.

## Language Support at a Glance

| Feature | Python | CLI | R | Julia | JavaScript/TypeScript |
| --- | --- | --- | --- | --- | --- |
| Validate dataset config | Yes | Yes | Via shell/process | Via shell/process | Via shell/process |
| Validate pipeline config | Yes | Yes | Via shell/process | Via shell/process | Via shell/process |
| Run training from YAML/JSON | Yes | No native `run` command yet | Via Python bridge/wrapper | Via Python bridge/wrapper | Via process/runtime wrapper |
| Use native operators as objects | Yes | No | Through Python bridge | Through Python bridge | No native object API in this repo |
| Export `.n4a` bundle | Yes | Workspace/artifact tooling only | Via Python bridge/wrapper | Via Python bridge/wrapper | Via process/runtime wrapper |
| Select `dag-ml` engine | `engine="dag-ml"` or `N4A_ENGINE=dag-ml` | Environment only for wrapped Python run | Same wrapped call | Same wrapped call | Same wrapped call |

The portable part is the configuration pair below. Language wrappers should keep those files unchanged.

## 1. Describe the Dataset

Create `dataset.yaml`:

```yaml
name: wheat_protein
task_type: regression

train_x: data/Xcal.csv
train_y: data/Ycal.csv
train_group: data/Mcal.csv

test_x: data/Xval.csv
test_y: data/Yval.csv
test_group: data/Mval.csv

global_params:
  delimiter: ";"
  has_header: true
  header_unit: cm-1
  signal_type: absorbance
```

The same config in JSON:

```json
{
  "name": "wheat_protein",
  "task_type": "regression",
  "train_x": "data/Xcal.csv",
  "train_y": "data/Ycal.csv",
  "train_group": "data/Mcal.csv",
  "test_x": "data/Xval.csv",
  "test_y": "data/Yval.csv",
  "test_group": "data/Mval.csv",
  "global_params": {
    "delimiter": ";",
    "has_header": true,
    "header_unit": "cm-1",
    "signal_type": "absorbance"
  }
}
```

## 2. Describe the Pipeline

Create `pipeline.yaml`:

```yaml
pipeline:
  - class: sklearn.preprocessing.MinMaxScaler
    params:
      feature_range: [0, 1]

  - class: sklearn.model_selection.ShuffleSplit
    params:
      n_splits: 5
      test_size: 0.25
      random_state: 42

  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components: 10
```

The same config in JSON:

```json
{
  "pipeline": [
    {
      "class": "sklearn.preprocessing.MinMaxScaler",
      "params": {
        "feature_range": [0, 1]
      }
    },
    {
      "class": "sklearn.model_selection.ShuffleSplit",
      "params": {
        "n_splits": 5,
        "test_size": 0.25,
        "random_state": 42
      }
    },
    {
      "model": {
        "class": "sklearn.cross_decomposition.PLSRegression",
        "params": {
          "n_components": 10
        }
      }
    }
  ]
}
```

## 3. Run It

::::{tab-set}

:::{tab-item} Python
```python
import nirs4all

result = nirs4all.run(
    pipeline="pipeline.yaml",
    dataset="dataset.yaml",
    name="hello_world_pls",
    random_state=42,
    verbose=1,
)

print(f"Best score: {result.best_score:.4f}")
print(f"Best RMSE: {result.best_rmse:.4f}")

bundle = result.export("exports/hello_world.n4a")
print(bundle)
```
:::

:::{tab-item} R
```r
library(reticulate)

nirs4all <- import("nirs4all")

result <- nirs4all$run(
  pipeline = "pipeline.yaml",
  dataset = "dataset.yaml",
  name = "hello_world_pls",
  random_state = 42L,
  verbose = 1L
)

print(result$best_score)
print(result$best_rmse)
result$export("exports/hello_world.n4a")
```
:::

:::{tab-item} Julia
```julia
using PythonCall

nirs4all = pyimport("nirs4all")

result = nirs4all.run(
    pipeline="pipeline.yaml",
    dataset="dataset.yaml",
    name="hello_world_pls",
    random_state=42,
    verbose=1,
)

println(result.best_score)
println(result.best_rmse)
result.export("exports/hello_world.n4a")
```
:::

:::{tab-item} JavaScript
```javascript
import { spawnSync } from "node:child_process";

const code = `
import nirs4all
result = nirs4all.run(
    pipeline="pipeline.yaml",
    dataset="dataset.yaml",
    name="hello_world_pls",
    random_state=42,
    verbose=1,
)
print(result.best_score)
print(result.best_rmse)
result.export("exports/hello_world.n4a")
`;

const run = spawnSync("python", ["-c", code], { stdio: "inherit" });
if (run.status !== 0) process.exit(run.status);
```
:::

:::{tab-item} Shell
```bash
nirs4all dataset validate dataset.yaml
nirs4all config validate pipeline.yaml --type pipeline --check-imports

python - <<'PY'
import nirs4all

result = nirs4all.run(
    pipeline="pipeline.yaml",
    dataset="dataset.yaml",
    name="hello_world_pls",
    random_state=42,
)
print(result.best_score)
result.export("exports/hello_world.n4a")
PY
```
:::

::::

## 4. Select the Runtime Engine

The default engine is resolved by `nirs4all.run()`. To request the dag-ml backend for covered pipeline shapes:

```python
result = nirs4all.run(
    pipeline="pipeline.yaml",
    dataset="dataset.yaml",
    engine="dag-ml",
    random_state=42,
)
```

Or set it for the process:

```bash
N4A_ENGINE=dag-ml python train.py
```

If the dag-ml backend is unavailable, or the requested pipeline shape is outside current dag-ml coverage, NIRS4ALL warns and falls back to the legacy engine for catchable unsupported cases.

## Next Clicks

- {doc}`/reference/nodes/index` lists every pipeline node keyword and where to use it.
- {doc}`/reference/nodes/merge` explains source, branch, prediction, and feature merges.
- {doc}`/reference/nodes/generators` explains cartesian and parameter-search syntax.
- {doc}`/reference/public_interfaces` lists the public Python API, CLI commands, and runtime switches.
