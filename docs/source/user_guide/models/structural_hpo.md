# Search optional scaling and Ridge versus PLS

Use the existing pipeline alternatives with `run(tuning=...)` to search both
the recipe and the parameter active for its model. This first structural
profile supports one dense source, one regression target, optional
`StandardScaler`, and `Ridge` versus `PLSRegression(scale=False)`. Declare
`GroupKFold` and the metadata column that identifies groups. All trials use
the same training cohort and folds; external test targets do not select a
recipe.

```python
import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

pipeline = [
    {"_or_": [None, StandardScaler()]},
    {"split": GroupKFold(3), "group_by": "batch"},
    {"model": {"_or_": [Ridge(), PLSRegression(scale=False)]}},
]
tuning = {
    "engine": "n4m", "sampler": "random", "seed": 17,
    "n_trials": 8, "metric": "rmse", "direction": "minimize",
    "space": {
        "model.alpha": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
        "model.n_components": {"type": "int", "low": 1, "high": 3},
    },
}

# dataset is an ordinary SpectroDataset with a complete "batch" metadata column.
with nirs4all.run(
    pipeline, dataset, engine="dag-ml", tuning=tuning,
    random_state=17, refit=True, save_charts=False,
) as result:
    print(result.tuning_best_params)
    print(result.tuning_best_value)
    archive = result.export("structural-winner.n4a")

# X_new contains only the new dense features; no training targets are required.
prediction = nirs4all.predict(archive, X_new, engine="dag-ml")
```

The complete runnable example is
`examples/user/04_models/U17_structural_hpo_ridge_pls.py`. It builds grouped
training and test data, exports the selected predictor, removes the training
workspace, and predicts again from the archive.

## Recipe and parameter identity

The declarations describe four recipes: raw or scaled features, followed by
Ridge or PLS. DAG-ML expands the existing Cartesian generator, assigns native
recipe identities and validates the ordered catalogue. The SDK does not
enumerate and schedule four independent Python pipelines.

Methods uses its native conditional activation: `model.alpha` is active only
for Ridge recipes; `model.n_components` only for PLS recipes. Public
`model__alpha` and `model__n_components` spellings normalize to the same
dotted paths. Both axes must be declared. Component values must be positive
integers no larger than the feature count or any training fold's row count.
Alpha values must be finite and nonnegative.

`tuning_best_params` and each `tuning_result.trials` entry contain only active public
parameters. `structural_tuning_evidence` retains the native recipe catalogue
binding, actual trial identities, scores and selected parameters. Its
`__recipe__` selector is internal evidence; do not put it in the public search
space or use `force_params`. Inactive placeholder values remain in the native
optimizer history for compatibility and never become fit arguments.

Every candidate owns its fitted transforms and cache. Nothing learned by one
recipe initializes another recipe. Native score reports determine the winner;
native winner resolution produces the exact pruned graph for CV and REFIT.
The final archive contains the fitted selected predictor, including its
preprocessing. Archive prediction performs no fit or optimizer search.

## Stop and resume

Supply a durable `storage` URI and `study_name`. The paired checkpoint stores
the native DAG history and native N4M optimizer state together.

```python
from pathlib import Path
from nirs4all.pipeline.dagml.cancellation import DagRunCancelled

tuning.update(storage=Path("study").resolve().as_uri(), study_name="ridge-pls")
tuning["progress_callback"] = lambda event: len(event["checkpoint"]["trials"]) < 2
try:
    nirs4all.run(pipeline, dataset, engine="dag-ml", tuning=tuning,
                 random_state=17, save_charts=False)
except DagRunCancelled:
    pass

tuning.pop("progress_callback")
tuning["resume"] = True
result = nirs4all.run(pipeline, dataset, engine="dag-ml", tuning=tuning,
                     random_state=17, save_charts=False)
```

Returning `False` from the progress callback stops after a durable terminal
trial. `n_trials` is the total budget, including resumed trials. Keep the
pipeline order, constructor settings, parameter axes, groups, folds, training
values, targets, objective and seeds unchanged. Catalogue and native activity
masks are bound to the checkpoint. An incompatible or altered pair is refused
before fitting, without overwriting its bytes. A fixed-topology checkpoint
cannot become a structural checkpoint. Extending the trial budget is allowed.

Before asking another trial, Methods compares the loaded optimizer's complete
ordered search space, categorical ordering, conditional constraints and
normalized options with a fresh optimizer built from the requested contract.
This includes seeds, sampler and pruner settings, even for recipes not yet
observed in the history. Active proposed values must also belong to their
declared domains before any candidate callback runs. Structural resume requires
the native `Optimizer.configuration_matches()` capability in both the Methods
binding and library; an older installation is refused explicitly. The native
checkpoint bytes and existing fixed-topology checkpoint format stay unchanged.

## Current limits

This profile requires `engine="dag-ml"`, tuning engine `n4m`, minimizing RMSE,
and winner REFIT. Sequential execution supports existing native samplers and
pruning. Parallel execution requires explicit `sampler="random"` without a
pruner; each candidate runs in an isolated Python process. The existing
execution resource controls apply to both trials and final training.

Other transformers or models, additional pipeline steps, multi-source or
multi-target data, generated or augmented views, fit controls, calibration,
`force_params`, custom `run(cache=...)`, and training through a session are
refused. Use the result's existing `.export()` method for the winner archive.
An older DAG-ML build without the native catalogue and winner helpers fails
explicitly. Ordinary fixed-estimator tuning, Optuna and generators without
tuning keep their existing paths.
