# nirs4all: from data to deployed predictions

<div align="center" style="margin-bottom: 20px;">
<img src="_static/nirs4all_logo.png" width="300" alt="NIRS4ALL Logo">
</div>

The common user and developer guide follows the same workflow in Python, R, JavaScript/WASM, MATLAB/Octave, Rust and the CLI: load data, compare recipes, select and refit a model, save fitted state, and predict.

Start with a task and choose a supported language and runtime profile. The {doc}`interface matrix <guide/interfaces>` identifies each product's capabilities and the qualified directions for transferring fitted models.

::::{grid} 2
:gutter: 3

:::{grid-item-card} Common user and developer guide
:link: guide/index
:link-type: doc

Chapters 0–13: installation, data, pipelines, evaluation, results, deployment, methods, language interfaces and contribution.
:::

:::{grid-item-card} Complete tutorials
:link: guide/tutorials
:link-type: doc

Run the same documented workflow in Python, R, Node/WASM, MATLAB/Octave and the CLI.
:::

:::{grid-item-card} Interfaces and portability
:link: guide/interfaces
:link-type: doc

Choose the Python SDK or portable Core, then check the supported data, artifact and runtime profile.
:::

:::{grid-item-card} Legacy — Python
:link: legacy/index
:link-type: doc

Previous Python guides, concepts, examples, developer documentation and detailed API references.
:::

::::

## Start with your next task

| Task | Guide |
|---|---|
| Install and run a first pipeline | {doc}`guide/start` |
| Load spectra, metadata or multiple modalities | {doc}`guide/datasets` |
| Compare pipelines and evaluate predictions | {doc}`guide/pipelines` and {doc}`guide/evaluation` |
| Inspect results and saved workspaces | {doc}`guide/results` |
| Save fitted state and predict in another process or language | {doc}`guide/deployment` |
| Choose a language interface | {doc}`guide/languages` |
| Contribute a controller, reader or binding | {doc}`guide/contributing` |
| Check exact API names and options | {doc}`guide/reference` |

```{toctree}
:maxdepth: 3
:caption: Documentation

guide/index
legacy/index
```
