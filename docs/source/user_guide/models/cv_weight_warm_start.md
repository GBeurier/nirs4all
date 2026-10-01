# Initialize SGD REFIT from an explicit CV fold

The general `engine="dag-ml"` path can initialize a fresh
`sklearn.linear_model.SGDRegressor` for full-train REFIT from the weights learned
on one explicitly named native CV fold. This is an optional, closed profile for
dense numeric, single-source, single-target regression.

```python
import nirs4all
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold

result = nirs4all.run(
    [
        KFold(3, shuffle=True, random_state=31),
        {
            "model": SGDRegressor(
                loss="squared_error", penalty="l2", alpha=0.01,
                learning_rate="constant", eta0=0.01,
                random_state=19, shuffle=False, max_iter=2, tol=None,
            ),
            "refit_params": {
                "warm_start": True,
                "warm_start_fold": "fold1",
                "max_iter": 3,
            },
        },
    ],
    (X, y),
    engine="dag-ml",
    workspace_path="workspace/sgd-warm-start",
    save_charts=False,
    verbose=0,
)
archive = result.export("sgd-warm-start.n4a")
result.close()
prediction = nirs4all.predict(archive, X_new, engine="dag-ml")
```

`X` and `y` are the training matrix and numeric target vector; `X_new` has the
same feature columns in the same order. Native fold identifiers are zero based:
`fold1` names the second split produced by the declared splitter. It names the
source fold, independently of its score. Every CV fold starts with a fresh
estimator. Only the requested fold's completed `coef_` and `intercept_` are
retained temporarily, and REFIT fits all native full-train rows, including the
rows held out by the source fold. Test-partition rows remain outside REFIT.

REFIT receives detached arrays through sklearn's public
`fit(coef_init=..., intercept_init=...)` parameters. Its optimization counter
starts again under sklearn's `fit()` semantics. This transfers weights;
optimizer counters, private buffers and RNG state are not continued. Changing
`max_iter` or `tol` for REFIT is supported. Structural parameters such as
`alpha`, the loss, regularization and learning-rate schedule must remain
identical to the source CV fit.

## Supported inputs and refusals

The model must be exactly `SGDRegressor`, with `loss="squared_error"`,
`penalty="l2"`, `average=False`, `early_stopping=False`, a fixed integer
`random_state`, and `learning_rate="constant"` or `"invscaling"`. `alpha` and
`power_t` must be finite and nonnegative; `eta0` must be finite and positive.
Use finite dense float32 or float64 features, a finite numeric target vector,
and positive fit budgets.

An explicit native selector such as `warm_start_fold="fold1"` is required.
Native identifiers follow `foldN`, where `N` is the zero-based split index.
`best`, `last`, missing selectors, underscored aliases such as `fold_1`, and
nonexistent folds refuse with a diagnostic. The profile does
not accept feature or target transformations, augmentation, multiple sources
or targets, residual learners, stacking, estimator subclasses, averaged SGD,
early stopping, or other estimators such as PLS and Ridge. Local
`finetune_params` and incompatible REFIT parameter changes refuse before the
REFIT fit. A missing or incompatible snapshot never silently becomes a cold
start. Omitting the warm-start request preserves the existing cold-start path.

## Native HPO and persistence

For callers supplying native DSL contracts and production operator callbacks,
the public `nirs4all.run_host_hpo_search()` surface schedules CV-only proposals
and returns the native selected parameters. Keep candidate callback stores
separate, close their temporary weights when each candidate finishes, and run
the selected recipe through `nirs4all.run()` to capture its own CV fold before
REFIT. Native DAG scoring and selection remain authoritative. Generic numeric
`run(tuning=...)` uses a separate explicit scoring-data profile; this guide
does not extend it to CV-weight transfer or fold ranking.

The final predictor carries `nirs4all.cv-weight-transfer.v1` provenance bound
to the native run, controller, node, variant, source fold, effective recipe,
input representation, weight digests and CV/REFIT budgets. Temporary snapshots
are consumed and cleared on success, failure or cancellation. The existing
Python host `.n4a` archive preserves this provenance and the final REFIT
predictor without persisting CV estimators or training rows. Archive prediction
and loaded-session prediction replay the fitted model without FIT or HPO.
These remain trusted Python host archives, rather than portable Core archives.

The runnable example is
`examples/user/04_models/U16_cv_weight_warm_start.py`. It also removes the
training workspace before archive replay.
