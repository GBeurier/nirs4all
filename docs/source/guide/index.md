# User and developer guide

Start with a task, then select the language you use. The common chapters describe datasets, evaluation, fitted state and results; the language chapters explain arrays, runtimes and resource ownership. Language tabs retain your selection while navigating in the same browser session.

The Python SDK and the portable Core product share upstream contracts, but their public return objects and supported workflows differ. Choose a documented profile from the interface matrix before transferring a fitted model. A shared function name alone does not establish portability.

```{toctree}
:maxdepth: 2

interfaces
start
principles
tasks
datasets
pipelines
evaluation
results
deployment
catalog
languages
tutorials
contributing
reference
```

| Your next task | Start here |
|---|---|
| Install and execute a complete first workflow | {doc}`start` |
| Load spectra together with metadata, images or sequences | {doc}`datasets` |
| Compare recipes, create folds or search hyperparameters | {doc}`pipelines` and {doc}`evaluation` |
| Inspect saved scores, predictions and experiments | {doc}`results` |
| Predict in another process or language | {doc}`deployment` |
| Add a controller, format reader or binding | {doc}`contributing` |
| Look up an exact parameter, tag or command | {doc}`reference` |

The existing detailed guides and API URLs remain available. This task-oriented guide connects them through the same data → evaluation → selection → refit → prediction lifecycle.
