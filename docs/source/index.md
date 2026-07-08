# NIRS4ALL Documentation

NIRS4ALL runs spectroscopy machine-learning workflows from two portable files:

- a **dataset configuration** (`dataset.yaml` or `dataset.json`) that describes where spectra, targets, metadata, sources, folds, and loading options live;
- a **pipeline configuration** (`pipeline.yaml` or `pipeline.json`) that describes the DAG nodes to execute: preprocessing, splitters, branching, merging, models, charts, export, and runtime options.

The same YAML/JSON contract is the stable user surface across Python and language wrappers. Python also exposes native objects for users who want direct sklearn/nirs4all composition.

## Feature Availability by Language

| Surface | Status in this repository | Reads dataset/pipeline YAML/JSON | Native object API | Train/evaluate | Predict/export | Best entry point |
| --- | --- | --- | --- | --- | --- | --- |
| Python | Native, complete public API | Yes | Yes, sklearn + nirs4all objects | `nirs4all.run(...)` | `nirs4all.predict(...)`, `result.export(...)` | {doc}`getting_started/hello_world` |
| CLI | Native for validation, dataset inspection, workspace, artifacts | Yes, for validation/inspection | No | No `run` command in this repo yet | Workspace/artifact management only | {doc}`reference/cli` |
| R | Portable wrapper pattern | Yes | Through `reticulate` or a binding wrapper | Calls the Python runtime today | Calls the Python runtime today | {doc}`getting_started/hello_world` |
| Julia | Portable wrapper pattern | Yes | Through `PythonCall` or a binding wrapper | Calls the Python runtime today | Calls the Python runtime today | {doc}`getting_started/hello_world` |
| JavaScript/TypeScript | Portable process/runtime wrapper pattern | Yes | No native JS API in this repo | Calls a Python/runtime process today | Calls a Python/runtime process today | {doc}`getting_started/hello_world` |
| dag-ml runtime | Selectable from Python | Yes, for covered shapes | No Python object construction | `engine="dag-ml"` with legacy fallback | Native results optional; bundle export is bridged | {doc}`reference/public_interfaces` |

:::{note}
The language-independent contract is the pair of config files. Native R, Julia, or JavaScript package APIs can wrap that contract without changing user pipelines.
:::

## Hello World

Start with one page: {doc}`getting_started/hello_world`.

It shows the smallest complete workflow:

1. Write `dataset.yaml`.
2. Write `pipeline.yaml`.
3. Run the same files from Python, R, Julia, JavaScript, or a shell wrapper.
4. Read the best score and export a `.n4a` bundle.

## Task Lookup

| Question | Go to |
| --- | --- |
| How do I describe my files? | {doc}`reference/configuration` |
| What can I put in a pipeline? | {doc}`reference/nodes/index` |
| How do I merge two sources? | {doc}`reference/nodes/merge` |
| How do I create a cartesian preprocessing search? | {doc}`reference/nodes/generators` |
| How do I branch by source, tag, metadata, or filter? | {doc}`reference/nodes/branch` |
| How do I add sample or feature augmentation? | {doc}`reference/nodes/sample_augmentation`, {doc}`reference/nodes/feature_augmentation` |
| Which operators/models/splitters exist? | {doc}`reference/operator_catalog` |
| Which public API/CLI/runtime commands exist? | {doc}`reference/public_interfaces` |
| What is the extended Python API? | {doc}`user_guide/python/index` |

```{toctree}
:maxdepth: 3
:caption: Documentation

getting_started/index
concepts/index
user_guide/index
reference/index
developer/index
examples/index
api/modules
```

## Install

```bash
pip install nirs4all
```

Optional extras:

```bash
pip install "nirs4all[viz,explain]"
pip install "nirs4all[torch]"
pip install "nirs4all[tensorflow]"
pip install "nirs4all[jax]"
pip install "nirs4all[all]"
```

Verify the installation:

```bash
nirs4all --test-install
```
