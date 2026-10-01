# Native PLS phase controls

Select the closed Methods RolePipeline profile explicitly to execute PLS
parameters during cross-validation, native HPO and the final refit. Methods
performs the numerical work; DAG-ML schedules the phases and owns signed
execution evidence. Matching DAG-ML and Core native packages are required.

```python
import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

result = nirs4all.run(
    [
        KFold(n_splits=3, shuffle=True, random_state=17),
        {
            "model": PLSRegression(n_components=1, scale=True),
            "train_params": {"n_components": 2},
            "finetune_params": {
                "engine": "n4m",
                "approach": "grouped",
                "sampler": "random",
                "pruner": "none",
                "n_trials": 4,
                "seed": 6,
                "metric": "rmse",
                "direction": "minimize",
                "model_params": {"scale": [False, True]},
            },
            "refit_params": {"n_components": 3, "scale": False},
        },
    ],
    {"X": X, "y": y, "sample_ids": sample_ids},
    engine="native",
    native_profile="n4m.pls_role_pipeline.v1",
    save_charts=False,
    methods_library_path=library_path,
)
archive = result.export("pls-winner.n4a")
result.close()

inspection = nirs4all.inspect_portable_predictor_archive_v2(
    archive, methods_library_path=library_path,
)
assert inspection["models"][0]["model_params"] == {
    "n_components": 3, "scale": False,
}
prediction = nirs4all.predict(
    archive,
    {"X": heldout_X, "sample_ids": heldout_ids},
    engine="native",
    methods_library_path=library_path,
)
```

Constructor values provide the base. `train_params` overrides that base for CV;
each trial overrides only its searched keys. A train key and a search axis
cannot own the same parameter. `refit_params` overrides the selected trial for
the final model. Consequently, the CV score measures the trial's CV recipe,
while archive inspection reports the effective final refit recipe.

The executable model keys are `n_components` (a positive integer up to
2,147,483,647) and `scale` (a Python boolean). `scale` controls native `scale_x`
and `scale_y` together. Search accepts `n_components: ["int", 1, 3]`,
`scale: [False, True]`, or both. Boolean choices are normalized to this order
without changing the caller's declaration. HPO returns booleans for
`model.scale`. Equal best CV scores use DAG-ML's deterministic selection;
`result.tuning_best_params` describes that selected trial. The native optimizer's
original incumbent remains in the checkpoint, with the same best score.
The profile keeps centering enabled and the native NIPALS solver. Keep
`copy=True`, `max_iter=500` and `tol=1e-6` on the declaration; other values,
`epochs`, `warm_start`, and native flag aliases are refused.

Supported recipes are PLS alone or `StandardNormalVariate()` followed by
`SavitzkyGolay(window_length=5, polyorder=2)` and PLS. Savitzky–Golay smoothing
requires an odd window from 3 to 501, degree below the window, `deriv=0`,
`delta=1.0`, and `copy=True`. The recipe order remains fixed. This profile
accepts one model and explicit array data; local and nested native HPO are
outside this profile.

`result.tuning_resume_package` retains the complete native checkpoint. Reuse
it as `finetune_params["resume_package"]` with an increased trial budget.
Changing data, folds, recipe, train/refit controls, or profile invalidates the
checkpoint. Saved `.n4a` files retain the closed recipe and opaque native
states; inspection imports and validates them without training. Prediction
hydrates the saved final model and performs no fit or optimizer operation.
Calls without `native_profile` retain the historical N4MM profile.

The opt-in qualification gate is
`NIRS4ALL_REQUIRE_NATIVE_PLS_PHASE_CONTROLS=1`. Select the actual Methods
library with `NIRS4ALL_CORE_LIVE_METHODS_LIBRARY`. Detached replay requires an
installed SDK with the same production bytes; use
`NIRS4ALL_NATIVE_PLS_INSTALLED_PYTHON` to select its Python executable when
the parent test process uses a separate environment.
