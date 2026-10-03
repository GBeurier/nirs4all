# Search source subsets, preprocessing chains and Ridge versus PLS

Use the existing pipeline alternatives with `run(tuning=...)` to search both
the recipe and the parameter active for its model. This structural profile
supports dense features, one regression target, fixed preprocessing
alternatives, and `Ridge` versus `PLSRegression(scale=False)`. For multiple
aligned dense sources, also declare the ordered source subsets to concatenate. Declare
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

## Fixed ordered preprocessing chains

Declare exactly one `None` alternative for raw features and at least one
nonempty preprocessing branch. Each branch can be a single operator instance
or a flat ordered list of instances. The allowed operators are
`StandardScaler`, `SNV` (the `StandardNormalVariate` alias) and
`SavitzkyGolay`. Chains can contain more than two operators, including repeated
operators with distinct constructor settings. Nested lists, nested generators,
model steps and `None` inside a chain are refused.

```python
from nirs4all.operators.transforms import SNV, SavitzkyGolay

pipeline[0] = {
    "_or_": [
        None,
        StandardScaler(),
        [SNV(), SavitzkyGolay(window_length=5, polyorder=2)],
    ],
}
tuning["n_trials"] = 12
```

These declarations describe six native recipes: three preprocessing branches
paired with the two model choices. The SNV and Savitzky-Golay operators remain
one ordered branch; their order and every constructor parameter belong to its
native recipe identity. Changing either changes the bound catalogue. The SDK
does not enumerate the Cartesian product in Python. A trial budget does not
guarantee that a random sampler visits every declared recipe.

`examples/user/04_models/U18_structural_hpo_preprocessing_chains.py` runs this
search on U17's deterministic 48-row fixture, with 36 training rows, 12 external
test rows and three grouped CV folds. It exports the selected fitted chain,
removes the training workspace and verifies archive prediction.

Constructor settings are fixed for the search; only the two model parameters
below are tuning axes. All supported preprocessing operators preserve the
feature width. Their additional profile constraints are checked before any
candidate is fitted:

| Operator | Constructor requirements |
| --- | --- |
| `StandardScaler` | `copy`, `with_mean` and `with_std` must be booleans. |
| `SNV` / `StandardNormalVariate` | `axis=1`, a nonnegative integer `ddof`, boolean `with_mean` and `with_std`, and `copy=True`. When `with_std=True`, `ddof` must be smaller than the feature width. |
| `SavitzkyGolay` | Positive integer `window_length` no larger than the feature width; integer `polyorder` between zero and `window_length - 1`; nonnegative integer `deriv`; finite, nonzero `delta`; and `copy=True`. |

Savitzky-Golay accepts even windows, negative `delta`, and derivative orders
greater than `polyorder` according to the existing transform's semantics.
These constructor values are strictly serialized as finite JSON values.

## Ordered source subsets before preprocessing

For an ordinary multi-source `SpectroDataset`, prepend a source-choice stage
using the existing explicit source-merge syntax:

```python
pipeline.insert(0, {
    "_or_": [
        {"merge": {"sources": {"strategy": "concat", "sources": [0]}}},
        {"merge": {"sources": {"strategy": "concat", "sources": [0, 2]}}},
        {"merge": {"sources": {"strategy": "concat", "sources": [2, 0]}}},
    ],
})
```

The first choice selects source zero. The second concatenates source zero
then source two; the third keeps the reverse order. Numeric indices and their
`source_<index>` aliases are accepted. Each explicit selection must be a
nonempty list without repeated sources. The input sources must be complete,
aligned two-dimensional numeric blocks with shared samples and targets.
Images, ragged series, missing-source masks, and source-local learned adapters
are outside this structural profile. Typed `MultimodalDataset` cohorts and their
`MultimodalSpectroDataset` adapters are refused here; their axis, unit and schema
contracts require the separate typed multimodal execution path.

Each source subset is followed by the entire selected preprocessing chain,
then the chosen model. For example, SNV runs across the selected concatenated
features, and StandardScaler learns statistics from that fold's training
rows. Preprocessing is not fitted separately on each original block. Every
declared source subset must have enough features for the SG window, SNV
`ddof`, and every proposed PLS component count. These constraints are checked
before catalogue creation or FIT.

With raw/scaled preprocessing and Ridge/PLS, the three source choices above
declare twelve recipes in one native Cartesian generator. An existing
sklearn `ColumnTransformer` selects concrete ordered columns; its constructor,
the complete original source layout, source choice, and subsequent operator
order belong to the signed native catalogue. Source subset selection adds no
public numeric tuning axis. Random sampling can revisit recipes without
covering every alternative.

`examples/user/04_models/U19_structural_hpo_source_subsets.py` demonstrates
this workflow using three deterministic fixture blocks of widths 6, 3 and 4,
grouped CV, winner REFIT, and archive replay after removing the workspace.
These blocks are test data, not a new multimodal dataset generator.

The winner archive retains the selected projection and the original complete
input layout. Predict with matching raw source blocks, or with their full
ordered concatenation. Include all original source columns, even those the
winner drops; do not preselect the winner's subset yourself. Changed block
counts and block widths are refused before projection or estimator prediction.
When a dataset exposes `source_name(index)`, source names and order are checked
as well. Anonymous blocks of equal width have no independent identity: an
exchange of these blocks cannot be detected. Preserve their original order.
A single flat matrix must also have the complete original width and column
order; it carries no separate source names or block-boundary information.

Changing a deliberately unused source does not affect candidate scores or
winner predictions. Its raw values still belong to the complete signed input
identity, so changing them invalidates an existing checkpoint. This protects
resume even when a source appears only in an unvisited recipe or is dropped
by every declared choice.

## Recipe and parameter identity

The optional-scaling declarations above describe four recipes: raw or scaled
features, followed by Ridge or PLS. The chain example adds two more recipes.
DAG-ML expands the existing Cartesian generator, assigns native
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

Every candidate owns its fitted transforms and cache. Each chain executes in
the declared order, fitting transforms only on that fold's training rows.
Nothing learned by one recipe initializes another recipe. Native score reports
determine the winner; native winner resolution produces the exact pruned graph
for CV and REFIT.
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
pipeline order, constructor settings, source choices and complete input layout,
parameter axes, groups, folds, training values, targets, objective and seeds
unchanged. Catalogue and native activity
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

## Typed modality alternatives and weighted early fusion

Typed NIR, image, series and mixed-metadata inputs can use a separate structural
profile with the existing `MultimodalRegressor`. Declare an ordered subset of
encoders for each alternative; the native catalogue selects a complete
early-fusion Ridge recipe and tunes its `model__alpha`.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from nirs4all.operators.models.multimodal import MultimodalRegressor, TensorPCA

pipeline = [GroupKFold(3), {"model": {"_or_": [
    MultimodalRegressor(
        transformers={"nir": StandardScaler()},
        model=Ridge(), backend="methods",
    ),
    MultimodalRegressor(
        transformers={
            "image": TensorPCA(n_components=2, random_state=17),
            "nir": StandardScaler(),
        },
        source_weights={"image": 0.5, "nir": 1.0},
        model=Ridge(), backend="methods",
    ),
]}}]
tuning = {
    "engine": "n4m", "sampler": "random", "seed": 17,
    "n_trials": 8, "metric": "rmse", "direction": "minimize",
    "space": {"model__alpha": {
        "type": "float", "low": 0.01, "high": 10.0, "log": True,
    }},
}
```

Pass the complete aligned four-source typed cohort to
`nirs4all.run(..., engine="dag-ml", refit=True)`. Encoder insertion order defines
fusion order. Omitted encoders never fit; a selected encoder with weight zero
still fits. Weights are finite, nonnegative declarations, default to one, and
may only name selected sources. Modality choices and weights remain fixed within
each declared alternative; this profile's numeric search space contains only
`model__alpha`.

The profile uses the existing U07 encoder families: NIR `StandardScaler`, image
and series `TensorPCA`, and the supported mixed-metadata `ColumnTransformer`.
It requires `GroupKFold(3)`, complete fixed-shape sources, one regression target,
sequential execution and winner REFIT. PCA component counts must fit the raw
source width and every training fold. Matching native Methods and DAG-ML support
is required; Python supplies declarations and raw buffers rather than computing
candidate fits or enumerating the native recipe catalogue.

The signed catalogue and resume contract retain all four raw source schemas and
their content, including held-out rows and excluded sources. The exported winner
also requires the complete original raw input contract when predicting new
cohorts. Its saved selected encoders and Ridge state replay without FIT or HPO.
`U20_structural_hpo_typed_modalities.py` demonstrates these declarations and
workspace-independent archive replay using U07's deterministic test fixture.

## Early versus learned late fusion

Declare alternative sequences with the existing `_or_` syntax. An early
sequence contains one Methods `MultimodalRegressor`. A learned late sequence
contains two to four named branches, a prediction merge, and an ordinary Ridge
meta-model. Each branch contains one Methods `MultimodalRegressor` selecting
exactly the source named by that branch.

```python
# early_model selects ordered NIR/image encoders, while nir_model and
# image_model select exactly their own single raw source. All use
# MultimodalRegressor(..., backend="methods") with the U07 encoder families.
pipeline = [GroupKFold(3), {"_or_": [
    [{"model": early_model}],
    [
        {"branch": {
            "nir": [{"model": nir_model}],
            "image": [{"model": image_model}],
        }},
        {"merge": "predictions"},
        {"model": Ridge(alpha=1.0)},
    ],
]}]
tuning = {
    "engine": "n4m", "sampler": "random", "seed": 17,
    "n_trials": 8, "metric": "rmse", "direction": "minimize", "n_jobs": 1,
    "space": {
        "early.alpha": [0.1, 1.0, 10.0],
        "late.nir.alpha": [0.1, 1.0, 10.0],
        "late.image.alpha": [0.1, 1.0, 10.0],
        "late.meta.alpha": [0.1, 1.0, 10.0],
    },
}
```

Declare exactly the alpha axes used by the alternative sequences. Early alpha
is active only for early recipes. Each `late.<source>.alpha` is active only
when that source has a late branch, and `late.meta.alpha` only for late recipes.
The corresponding double-underscore spellings normalize to the same dotted
paths. Encoder constructors, source subsets, branch order and source weights
remain explicit declarations. The native catalogue chooses the topology and
Methods proposes only its active numeric parameters.

For late fusion, DAG-ML declares grouped two-fold inner OOF inside every
outer training fold. Each encoder and branch predictor learns only from that
inner training scope. The meta-model learns from those inner held-out
predictions, then predicts the outer validation cohort through branch
predictors refitted on the outer training rows. Winner REFIT uses its own
grouped inner OOF on the complete training cohort before fitting the terminal
meta-model and complete branch predictors. External test targets never train
the meta-model or select the recipe.

The SDK turns named branches into an ordered native list. That order also
defines the meta-model prediction columns and belongs to the signed topology.
PCA components must fit every actual native inner training scope, outer training
scope and full REFIT scope; the native catalogue checks these bounds before
model callbacks. This profile requires complete four-source U07 inputs, one
regression target named `y`, deterministic `GroupKFold(3)`, `refit=True`,
`n_jobs=1`, and no pruner. Ordinary sklearn Ridge in the public meta declaration
lowers to a native Methods Ridge; Python does not fit a sklearn meta-model.

Stop/resume retains the complete topology, conditional axes, groups, targets,
raw schemas and all raw values, including excluded sources and test rows.
The winner archive contains precisely its selected branch encoder/predictor
states and native meta-model state. Held-out REFIT predictions retain their
Test partition. With a train-only cohort, meta-model REFIT captures artifacts
without inventing a final training score. It replays a new complete raw cohort
without FIT, HPO or the training workspace. Always provide all four original
raw sources, even for an excluded modality.

`examples/user/04_models/U21_structural_hpo_early_late.py` searches early fusion,
late fusion with two, three and four branches, and reversed two-branch order
on the existing U07 fixture. It exports the selected winner, deletes training
state, and verifies archive replay. This phase qualifies the native Python
path; it does not extend classification, missing or ragged inputs, deep
learning, or the R/Octave/WASM replay matrix for these mixed-state topologies.

## Dense profile limits

The dense profile above requires `engine="dag-ml"`, tuning engine `n4m`, minimizing RMSE,
and winner REFIT. Sequential execution supports existing native samplers and
pruning. Parallel execution requires explicit `sampler="random"` without a
pruner; each candidate runs in an isolated Python process. The existing
execution resource controls apply to both trials and final training.

Other transformers or models, additional steps outside the declared branches,
multi-target data, non-dense or incomplete source blocks, source selection without
the explicit source-choice stage, generated or augmented views, fit controls, calibration,
`force_params`, custom `run(cache=...)`, and training through a session are
refused. Use the result's existing `.export()` method for the winner archive.
An older DAG-ML build without the native catalogue and winner helpers fails
explicitly. Ordinary fixed-estimator tuning, Optuna and generators without
tuning keep their existing paths.
