# AOM Models

AOM models are spectroscopy-specific estimators that search over linear
preprocessing operators inside the calibration model. In nirs4all, they can use
the same split protocol as the outer pipeline, including user-defined splitters
such as `KFold`, `GroupKFold`, `StratifiedGroupKFold`, or custom splitters.

Use the runnable example at
`examples/user/04_models/U07_aom_panoply.py` for a complete comparison of
AOM-PLS, AOM-Ridge, `AOMRidgeAutoSelector`, `AOMRidgeBlender`, and FastAOM.

## Which AOM Model To Use

| Model | Best Use | Split-Aware Parameter |
|-------|----------|-----------------------|
| `AOMRidgeBlender` | Best general AOM-Ridge recipe; convex blend of several candidate variants | `outer_cv` and `inner_cv` |
| `AOMRidgeAutoSelector` | Pick one best AOM-Ridge variant instead of blending | `outer_cv` and `inner_cv` |
| `AOMRidgeRegressor` | Fast single AOM-Ridge model, good first production baseline | `cv` |
| `AOMPLSRegressor` | PLS-compatible AOM model with operator-bank selection | `cv_splitter` and `cv` |
| `FastAOMPLSRidge` | Faster chain-screened AOM family when runtime matters | no pipeline-fold injection required |

## Pipeline Split Reuse

For AOM models with internal validation, set
`train_params.use_pipeline_folds_for_aom` on the model step:

```python
from sklearn.model_selection import KFold
from nirs4all.operators.models import AOMRidgeBlender

pipeline = [
    KFold(n_splits=4, shuffle=True, random_state=42),
    {
        "model": AOMRidgeBlender(outer_cv=4, inner_cv=4, random_state=42),
        "name": "AOMRidge-blender",
        "train_params": {"use_pipeline_folds_for_aom": "required"},
    },
]
```

The same pattern works with grouped splits:

```python
from sklearn.model_selection import GroupKFold
from nirs4all.operators.models import AOMRidgeAutoSelector

pipeline = [
    {"split": GroupKFold(n_splits=4), "group_by": "batch_id"},
    {
        "model": AOMRidgeAutoSelector(outer_cv=4, inner_cv=4, random_state=42),
        "name": "AOMRidge-auto",
        "train_params": {"use_pipeline_folds_for_aom": "required"},
    },
]
```

`required` is recommended for grouped or stratified protocols. It raises if the
pipeline folds cannot be mapped into the estimator. The default `auto` policy
uses pipeline folds when possible and silently falls back to the estimator's own
CV when no compatible folds exist.

## What nirs4all Injects

When the model is recognised as an AOM estimator, nirs4all builds a precomputed
splitter in the coordinate system of the current training slice and injects it
as follows:

| Estimator Capability | Injection |
|----------------------|-----------|
| exposes `cv_splitter` | sets `cv_splitter=<pipeline splitter>` and `cv=n_splits` |
| exposes `outer_cv` | sets `outer_cv=<pipeline splitter>` |
| exposes both `outer_cv` and `inner_cv` | sets both to the pipeline splitter |
| exposes `cv` only | sets `cv=<pipeline splitter>` |
| exposes `external_folds` | sets AOM-lib style validation folds and `selection="external"` |

For `AOMRidgeAutoSelector` and `AOMRidgeBlender`, each candidate sees only the
candidate-local subset of the pipeline folds. This keeps nested AOM tuning inside
the same user-defined protocol without letting candidate validation rows leak
into candidate training.

## Practical Candidate Set

The default Blender candidates are the strongest paper-style set, but they are
heavier. For a fast user example, start with a compact candidate list:

```python
QUICK_AOM_RIDGE_CANDIDATES = [
    {
        "label": "Ridge-identity",
        "selection": "superblock",
        "operator_bank": "identity",
        "block_scaling": "none",
    },
    {
        "label": "AOMRidge-global-compact",
        "selection": "global",
        "operator_bank": "compact",
        "block_scaling": "none",
    },
    {
        "label": "AOMRidge-global-compact-snv",
        "selection": "global",
        "operator_bank": "compact",
        "block_scaling": "none",
        "branch_preproc": "snv",
    },
]
```

Then pass it to the aggregators:

```python
from nirs4all.operators.models import AOMRidgeBlender

model = AOMRidgeBlender(
    candidates=QUICK_AOM_RIDGE_CANDIDATES,
    outer_cv=4,
    inner_cv=4,
    outer_cv_kind="kfold",
    regularizer=0.01,
    random_state=42,
)
```

For final benchmark runs, omit `candidates` to use the full headline set.

## Limitations

Pipeline-fold forwarding is disabled when a fit-influence policy resamples the
training rows before model fitting, because the original fold indices no longer
align with `X_train`. With `use_pipeline_folds_for_aom="required"`, this raises
instead of falling back.

Repeated AOM-PLS CV (`repeats > 1`) is not compatible with a fixed pipeline
splitter. Use `repeats=1` when the pipeline split must be authoritative.
