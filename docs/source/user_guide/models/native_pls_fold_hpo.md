# Native PLS HPO within each outer fold

Set `finetune_params.scope="fold"` with
`native_profile="n4m.pls_role_pipeline.v1"` to select PLS parameters separately
inside each outer training fold. DAG-ML schedules independent native studies,
and Methods performs fitting and prediction. The SDK materializes the declared
splitter's sample-ID memberships; it does not run a Python optimizer or compute
the training scores.

```python
import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

pipeline = [
    KFold(n_splits=3, shuffle=True, random_state=17),
    {
        "model": PLSRegression(n_components=1, scale=True),
        "finetune_params": {
            "engine": "n4m",
            "scope": "fold",
            "approach": "grouped",
            "sampler": "random",
            "pruner": "none",
            "n_trials": 4,
            "seed": 6,
            "metric": "rmse",
            "direction": "minimize",
            "model_params": {
                "n_components": ["int", 1, 3],
                "scale": [False, True],
            },
        },
        "refit_params": {"n_components": 2, "scale": False},
    },
]

with nirs4all.run(
    pipeline,
    {"X": X, "y": y, "sample_ids": sample_ids, "target_names": ["response"]},
    engine="native",
    native_profile="n4m.pls_role_pipeline.v1",
    save_charts=False,
    methods_library_path=library_path,
) as result:
    outer_cv_rmse = result.cv_best_score
    studies = result.tuning_scope_results
    refit_search_params = result.tuning_best_params
    refit_inner_rmse = result.tuning_best_value
    resume_package = result.tuning_resume_package
    archive = result.export("fold-tuned-pls.n4a")
```

For each outer fold, the study sees only its outer-train sample IDs. The same
declared splitter is applied to that pool to build the inner folds. Inner OOF
RMSE selects the local winner. A fresh model with those parameters is fitted
on all outer-train rows and predicts outer-validation rows. Those external
predictions produce `cv_best_score`; outer-validation targets never select
that fold's winner.

REFIT runs a separate study over the full declared training universe. Its
inner OOF winner supplies `tuning_best_params` and `tuning_best_value`.
`refit_params` then overrides the winning recipe for the final fitted model.
The exported RAW artifact contains that final REFIT state. No outer-fold
model weights are reused. In the example, inspection reports two components
and `scale=False`, even if the REFIT search selected different parameters.

`n_trials` is the total budget for **each study**, including resumed history.
Three outer folds plus REFIT therefore run four studies. With a budget of
four, that is up to sixteen trials, each evaluated on its own inner folds.
The local budgets, native checkpoints, parameter identities and lineage are
kept separately; they are not one campaign-wide search history.

## Splitters and group identity

The fold profile accepts exact sklearn `KFold` and `GroupKFold` declarations.
Shuffling requires an explicit `random_state`. Inner membership uses the
same configuration applied to the scoped pool, preserving sklearn's actual
splits rather than replacing them with a modulo-based split.

For grouped data, supply a complete group vector and use `GroupKFold` directly:

```python
from sklearn.model_selection import GroupKFold

pipeline[0] = GroupKFold(n_splits=3)
dataset = {
    "X": X,
    "y": y,
    "sample_ids": sample_ids,
    "groups": group_ids,
    "target_names": ["response"],
}
```

Run that pipeline and dataset with the same options above. All rows of a group
remain on the same fold side, including within each outer-train study. Each
scoped pool must contain enough groups and training rows for the declared
splitter and PLS component count. Missing or partial groups are refused.
Declaring groups with `KFold` is also refused. This profile adds no separate
`inner_cv` setting, resampling splitter, or `approach="individual"` mode.

## Study results and resume

`tuning_scope_results` returns an independent snapshot containing
`outer_scopes` and `refit_scope`. Each scope records its phase, outer fold ID
(null for REFIT), signed inner fold set, effective parameter fingerprint,
winner identity, typed `winner_params` and complete `resume_state`. Winner
parameters contain only searched keys. Fixed constructor, train and refit
settings remain in the signed recipe. Public `scale` values are booleans;
the native categorical ledger uses index 0 for `False` and 1 for `True`.
Phase JSON is `FIT_CV` or `REFIT`; scope ID suffixes retain lowercase
`fit_cv`/`refit`. The package's required root coordinator `relations` table
attests group/origin/exclusion identity without carrying X/y buffers.

For metadata inspection through DAG-ML, use
`dag_ml.methods_hpo_fold_state_from_package(resume_package)`. It returns a
native-validated `MethodsFoldHpoState` whose `to_dict()` snapshots are
independent. The reader validates the containing package without FIT or HPO.

To continue all studies from two trials to four, retain the complete
`result.tuning_resume_package`, set `n_trials=4`, and pass the package as
`finetune_params.resume_package`. The containing Package V2 is the resume
authority; a standalone N4MOPT checkpoint is insufficient. Changing features,
targets, fold membership, groups, train/refit controls or study scope invalidates
it. Exchanging histories between scopes is refused even when a containing
package fingerprint is recomputed.

## Inspect and replay the saved final model

```python
inspection = nirs4all.inspect_portable_predictor_archive_v2(
    archive, methods_library_path=library_path,
)
assert inspection["models"][0]["model_params"] == {
    "n_components": 2, "scale": False,
}
prediction = nirs4all.predict(
    archive,
    {"X": heldout_X, "sample_ids": heldout_ids},
    engine="native",
    methods_library_path=library_path,
)
assert prediction.metadata["training_performed"] is False
```

Inspection and prediction hydrate the saved final state without FIT or HPO.
They require matching installed native packages and no training arrays or
heldout targets. Omitting `scope`, or setting `scope="campaign"`, retains the
existing campaign behavior. Calls without the explicit native profile retain
the historical N4MM lane. This feature does not add warm-start, a new model
family, or the separate multimodal N-D profile.
The existing SNV → Savitzky–Golay → PLS recipe remains available through
this explicit profile; every inner, outer and REFIT fit preserves its declared
preprocessing steps.

The required scientific gate is
`NIRS4ALL_REQUIRE_NATIVE_PLS_FOLD_HPO=1`, with
`NIRS4ALL_CORE_LIVE_METHODS_LIBRARY` selecting the exact Methods library.
`NIRS4ALL_NATIVE_PLS_INSTALLED_PYTHON` selects the installed-wheel Python for
detached `-I` replay when the parent test process uses a different environment.
