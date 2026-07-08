# Public Interfaces and Runtime

NIRS4ALL has one portable workflow contract and several user surfaces around it.

The portable contract is:

```text
dataset.yaml or dataset.json
pipeline.yaml or pipeline.json
```

The public Python API runs that contract directly. The CLI currently validates and manages datasets, configs, workspaces, and artifacts; it does not expose a native training `run` subcommand in this repository.

## Python API

| Function/class | Purpose | Typical input | Typical output |
| --- | --- | --- | --- |
| `nirs4all.run(...)` | Train/evaluate one or many pipelines on one or many datasets | Pipeline spec + dataset spec | `RunResult` |
| `nirs4all.predict(...)` | Predict from a stored chain or exported bundle | `chain_id` or model bundle + data | `PredictResult` |
| `nirs4all.explain(...)` | Generate SHAP explanations | Model/bundle + data | `ExplainResult` |
| `nirs4all.retrain(...)` | Retrain from an existing result or bundle | Source + new data | `RunResult` |
| `nirs4all.session(...)` | Share runner/workspace resources across calls | Optional pipeline and runner kwargs | `Session` |
| `nirs4all.load_session(...)` | Load an exported `.n4a` bundle for prediction | Bundle path | `Session` |
| `nirs4all.generate(...)` | Generate synthetic NIRS data | Synthetic parameters | `SpectroDataset` or arrays |
| `result.export(...)` | Export a trained pipeline bundle | Output path | `.n4a` path |

## Train from Config Files

::::{tab-set}

:::{tab-item} Python
```python
import nirs4all

result = nirs4all.run(
    pipeline="pipeline.yaml",
    dataset="dataset.yaml",
    name="run_from_files",
    random_state=42,
)

print(result.best_score)
result.export("exports/model.n4a")
```
:::

:::{tab-item} R
```r
library(reticulate)
nirs4all <- import("nirs4all")

result <- nirs4all$run(
  pipeline = "pipeline.yaml",
  dataset = "dataset.yaml",
  name = "run_from_files",
  random_state = 42L
)

result$export("exports/model.n4a")
```
:::

:::{tab-item} Julia
```julia
using PythonCall
nirs4all = pyimport("nirs4all")

result = nirs4all.run(
    pipeline="pipeline.yaml",
    dataset="dataset.yaml",
    name="run_from_files",
    random_state=42,
)

result.export("exports/model.n4a")
```
:::

:::{tab-item} JavaScript
```javascript
import { spawnSync } from "node:child_process";

const code = `
import nirs4all
r = nirs4all.run("pipeline.yaml", "dataset.yaml", name="run_from_files", random_state=42)
print(r.best_score)
r.export("exports/model.n4a")
`;

const run = spawnSync("python", ["-c", code], { stdio: "inherit" });
if (run.status !== 0) process.exit(run.status);
```
:::

::::

## Predict from an Exported Bundle

::::{tab-set}

:::{tab-item} Python
```python
import nirs4all

pred = nirs4all.predict(
    model="exports/model.n4a",
    data="new_dataset.yaml",
)

df = pred.to_dataframe()
print(df.head())
```
:::

:::{tab-item} R
```r
library(reticulate)
nirs4all <- import("nirs4all")

pred <- nirs4all$predict(
  model = "exports/model.n4a",
  data = "new_dataset.yaml"
)

print(pred$to_dataframe())
```
:::

:::{tab-item} Julia
```julia
using PythonCall
nirs4all = pyimport("nirs4all")

pred = nirs4all.predict(
    model="exports/model.n4a",
    data="new_dataset.yaml",
)

println(pred.to_dataframe())
```
:::

:::{tab-item} JavaScript
```javascript
import { spawnSync } from "node:child_process";

const code = `
import nirs4all
pred = nirs4all.predict(model="exports/model.n4a", data="new_dataset.yaml")
print(pred.to_dataframe())
`;

const run = spawnSync("python", ["-c", code], { stdio: "inherit" });
if (run.status !== 0) process.exit(run.status);
```
:::

::::

## Runtime Engine

`nirs4all.run()` accepts an `engine` selector:

| Engine | Meaning |
| --- | --- |
| `None` | Use the package default resolved by `resolve_engine(...)`. |
| `"legacy"` | Use the in-process Python orchestrator. |
| `"dag-ml"` | Request the dag-ml backend for covered shapes; catchable unsupported/unavailable cases warn and fall back to legacy. |
| `"dual"` | Reserved; not implemented as a public side-by-side mode. |

The pipeline language is broader than current dag-ml native coverage. Requesting `engine="dag-ml"` is safe for user workflows because catchable unsupported shapes and unavailable dag-ml runtime dependencies warn and re-run on the legacy engine. A genuine dag-ml runtime/operator bug still propagates as an error.

```python
result = nirs4all.run(
    pipeline="pipeline.yaml",
    dataset="dataset.yaml",
    engine="dag-ml",
    random_state=42,
)
```

Environment switches:

| Variable | Effect |
| --- | --- |
| `N4A_ENGINE=dag-ml` | Request dag-ml for calls that do not pass `engine=` explicitly. |
| `N4A_NATIVE_RESULTS=/path/to/results` | For dag-ml runs, request native result output. |

## CLI Commands

Use `nirs4all <group> <command> --help` for exact options.

| Group | Command | Purpose |
| --- | --- | --- |
| root | `--test-install` | Print dependency/install diagnostics. |
| root | `--test-integration` | Run the packaged integration smoke test. |
| root | `--version` | Print installed package version. |
| `config` | `validate <file>` | Validate a pipeline or dataset config. |
| `config` | `schema pipeline|dataset` | Print the JSON schema. |
| `dataset` | `validate <file>` | Validate a dataset config/folder. |
| `dataset` | `inspect <file>` | Show sources, task type, aggregation, and optional detected file parameters. |
| `dataset` | `export <file>` | Normalize a dataset config to YAML or JSON. |
| `dataset` | `diff <a> <b>` | Compare two dataset configs. |
| `workspace` | `init <path>` | Create workspace directories and store. |
| `workspace` | `list-runs` | List recorded runs. |
| `workspace` | `query-best` | Query top predictions by metric. |
| `workspace` | `filter` | Filter prediction rows by dataset and scores. |
| `workspace` | `stats` | Show workspace score statistics. |
| `workspace` | `list-library` | List saved templates/library items. |
| `artifacts` | `list-orphaned` | List artifacts not referenced by manifests. |
| `artifacts` | `cleanup` | Dry-run or delete orphaned artifacts. |
| `artifacts` | `stats` | Show artifact storage statistics. |
| `artifacts` | `purge` | Delete all artifacts for a dataset when forced. |

Example:

```bash
nirs4all dataset validate dataset.yaml
nirs4all dataset inspect dataset.yaml --detect
nirs4all config validate pipeline.yaml --type pipeline --check-imports
nirs4all workspace init workspace
nirs4all workspace query-best --workspace workspace --metric test_score -n 5
```

## Result Objects

`RunResult` exposes:

| Accessor | Purpose |
| --- | --- |
| `best`, `final`, `cv_best` | Best/final prediction entries. |
| `best_score`, `best_rmse`, `best_r2`, `best_accuracy` | Convenience score fields. |
| `top(n=...)` | Top prediction rows. |
| `filter(...)` | Filter prediction rows. |
| `models` | Per-model refit result handles when available. |
| `export(path, format="n4a")` | Export the selected model bundle. |

See also {doc}`api/session`, {doc}`predictions_api`, and {doc}`/api/module_api`.
