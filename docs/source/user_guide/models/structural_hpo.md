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
serial execution or the bounded native parallel profile below, and winner REFIT.
PCA component counts must fit the raw
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
serial execution or the bounded native parallel profile below, and no pruner.
Ordinary sklearn Ridge in the public meta declaration
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

## Bounded parallel typed campaigns

The typed early-fusion and early/learned-late declarations above also accept
`n_jobs=2`, `3` or `4`. Use the native random sampler, no pruner, one numerical
CPU thread per candidate and no GPU devices:

```python
parallel_tuning = {**tuning, "sampler": "random", "pruner": None, "n_jobs": 2}
with nirs4all.run(
    pipeline, complete_four_source_cohort, tuning=parallel_tuning,
    engine="dag-ml", refit=True, cpu_threads=1, gpu_devices=[],
) as result:
    archive = result.export("parallel-winner.n4a")
    candidate_audit = result.structural_tuning_candidate_audit
```

A matching Methods build must report actual schema-v1 build capabilities through
`n4m.build_capabilities()`, with `blas`, `openmp` and `cuda` all false. A CUDA build
is refused even if no CUDA device is visible. Missing or malformed build evidence
fails before the optimizer is created. This feature never changes environment
variables, disables an accelerated backend or infers its build from the requested
thread count. Upgrade Methods and DAG-ML together for this profile.

DAG-ML admits bounded candidate windows and owns all worker execution, grouped
folds, nested OOF, scoring and winner selection. Each candidate has distinct
controller/model handles over the complete signed raw inputs. The SDK supplies
raw buffers and declarations; it creates no Python pool and computes no fits.
Serial `n_jobs=1` retains its previous request and checkpoint representation.
Automatic or negative worker counts, more than four workers, adaptive samplers,
pruning, GPU requests and multiple numerical threads are refused for this lane.
Generated views, deep learning and R/Octave/WASM/browser parallel execution are
outside this profile. Memory grows with admitted candidates and their nested
models; this admission bound does not enforce a process memory ceiling or promise
speedup on every dataset.

Cancellation takes effect after an admitted window has joined. Its successful
siblings remain terminal in the paired native/optimizer checkpoint; a failed
candidate has no fabricated score. A worker failure closes every candidate owner
and propagates after joined sibling outcomes are retained. Resume keeps terminal
trials and re-executes pending RUNNING trials from fresh model state with their
original proposal IDs and parameters. This is campaign resume, rather than partial
numerical continuation. The native request signs the actual sequential build
profile, worker count and per-candidate resource declaration; changing them, the
complete cohort or the topology refuses resume before model callbacks.

`structural_tuning_candidate_audit` contains the candidates executed by the current
invocation, sorted by trial index, with their signed recipe IDs and separate raw
and meta-owner event lists captured after resource closure. Owner lists are not a
global chronology across workers. Winner REFIT and archive replay retain their
ordinary training/replay evidence. A selected mixed N4MF/N4ME closure predicts new
complete raw cohorts without FIT/HPO, including after deleting the training study
and workspace.

`examples/user/04_models/U22_parallel_structural_hpo.py` runs the U21 declarations
with a bounded native campaign and exports/replays its winner. U20 declarations
use the same tuning and resource controls.

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


## Native typed classification with early or learned late fusion

`examples/user/04_models/U23_structural_hpo_classification.py` reuses U07's
four raw modalities as a deterministic software fixture. It selects early
fusion or two to four named singleton branches followed by a genuine Methods
PLS-logistic head. Every learned encoder and classifier runs in Methods;
DAG-ML owns recipe generation, grouped inner OOF, metrics and winner refit.

Declare `MultimodalClassifier(..., model=n4m.roles.PLSLogistic(n_components=1,
max_iter=500), backend="methods")`. Fixed public runs use
`[GroupKFold(3), {"model": classifier}]`, or the same fixed named late
sequence without an `_or_` site; neither creates an optimizer. A directly fitted classifier requires
explicit IO-derived `source_schemas`; `predict_proba` column `j` corresponds
exactly to `classes_[j]`. Learned late fusion keeps the existing syntax:

```python
pipeline = [GroupKFold(3), {"_or_": [
    [{"model": early}],
    [{"branch": {"image": [{"model": image}], "nir": [{"model": nir}]}},
     {"merge": "predictions"}, {"model": PLSLogistic(n_components=1, max_iter=500)}],
]}]
```

Each named branch must select exactly its own source. All four original IO
source signatures remain required, including excluded sources. Native late
fusion consumes the `probabilities` ports in declared branch order, followed
by signed class-column order. The scored/deployed output is the distinct
`y_hat` label port. Its output request declares the actual contiguous native
IDs as text; original typed labels remain signed in the graph and state wrapper
and are decoded at the public return boundary. No probability matrix replaces
public label predictions.

The mandatory component axes are `early.n_components`,
`late.<source>.n_components` and `late.meta.n_components` for the declared
alternatives. Counts are positive integer domains; `max_iter` is a fixed signed
head parameter. The default objective is maximizing native accuracy;
`balanced_accuracy` and support-weighted `f1` also use native label scoring.
Classification requires `n_jobs=1`, no pruner and `refit=True`; HPO05 regression
worker admission does not admit parallel classification.

One sorted typed vocabulary is learned from Train labels only. Homogeneous
strings and int64 labels retain their original types, including noncontiguous
integer IDs. Mixed types, floating labels, missing labels and heldout-only
classes are refused. The closed native wire budgets admit two to 65536 classes
and UTF-8 string labels of at most 1 MiB each, including empty strings.
Native preflight enforces the 16,777,216-cell class matrix budget, including
concatenated late probability columns, and checks every actual outer, inner and
full-refit training scope for all declared classes and component bounds before
FIT or optimizer construction. The SDK signs identities and decodes exact
finite integral class IDs; it never constructs inner folds or computes scores.

The portable winner retains N4MC raw classifier states and the complete
PLS-logistic meta state. Export does not fit again. Heldout rows remain Test;
a Train-only late refit captures its deployment artifact without fabricating
terminal predictions or a training metric. Fresh PREDICT replay produces actual
payloads and original labels after validating the saved vocabulary and source
identities, without FIT/HPO or a training workspace.

These source declarations and software fixtures require the matching Methods
ABI 2.17 and DAG classification build. Qualification is a separate release gate;
this guide makes no claim that an older installed runtime supports the profile.

## Native structural Torch regression on CPU

`U24_structural_hpo_torch.py` declares early and learned late fusion through the
same `_or_` sequences. Its model declarations use `MultimodalRegressor` with
ordered passthrough sources and `DagMLTorchEstimator`. The exact importable
`structural_mlp` factory creates a new Torch `Flatten → Linear → ReLU → Linear`
module for every fit. The final late estimator is ordinary `Ridge(solver='svd')`.

```python
model = MultimodalRegressor(
    transformers={"image": None, "nir": None},
    model=DagMLTorchEstimator(
        factory_path="nirs4all.operators.models.pytorch.mlp.structural_mlp",
        factory_params={"hidden_units": 8}, force_layout="2d", device="cpu",
        task_type="regression", epochs=3, batch_size=12, patience=3,
        optimizer="Adam", loss="MSELoss", lr=0.01,
    ),
)
pipeline = [GroupKFold(3), {"_or_": [
    [{"model": model}],
    [{"branch": {
        "image": [{"model": image_only_model}],
        "nir": [{"model": nir_only_model}],
    }}, {"merge": "predictions"}, {"model": Ridge(alpha=1.0, solver="svd")}],
]}]
```

This closed profile requires four complete aligned raw numeric 2D sources,
finite mono-y regression targets, deterministic outer `GroupKFold(3)`,
`refit=True`, `n_jobs=1`, `sampler='random'`, no pruner, `cpu_threads=1` and no GPU
devices. Early source subsets preserve declared feature order; each late branch
selects exactly its named source, with two to four branches. All four original
source schemas remain signed, including excluded sources. Arbitrary Torch
module templates, other factories, missing sources, generated/augmented views,
classification and DL parallelism are refused before model callbacks.

The exact required conditional axes are `early.lr`, `late.<name>.lr` and
`late.meta.alpha`, for the nodes actually declared. Learning rates lie in
`[1e-6, 0.1]`; Ridge alpha lies in `[0, 1e6]`. Architecture/training controls remain
signed declarations: hidden width 1–128, epochs 1–100, batch size 1–1024 and
patience 1–100. The existing Torch loop shuffles Train batches, receives no
validation cohort, performs every declared epoch and applies no early stopping.
The signature records this exact policy. Its random state derives from the
native task identity before module initialization and batch ordering.

DAG-ML owns topology expansion, proposals, three grouped outer folds, two
native grouped inner OOF folds, full-training REFIT OOF, native RMSE and winner
selection. Torch learns only from that task's Train rows. The meta Ridge learns
only from native branch OOF predictions, never branch training predictions or
Test rows. Admission bounds raw/selected buffers to 16,777,216 cells, the model
to 1,000,000 parameters and the conservative training-work estimate to
100,000,000 units before callbacks.

All raw sources and scalar targets must also stay finite after the actual
float32 conversion consumed by Torch. Finite float64 values outside that range
are refused before compilation, proposals or fitting, including excluded
sources and heldout targets. The scalar target's actual nonempty name, such as
`concentration`, is signed in the profile and retained through outputs,
capture and fresh replay; it is not renamed to `y`.

The selected graph is exported with `allow_host_sidecar`. This archive carries
all actual fitted Torch branches and the learned Ridge as Python joblib
sidecars; fresh replay executes their signed native PREDICT graph without FIT,
HPO or a training workspace. Treat joblib archives as trusted Python content.
Each REFIT sidecar is bound to its native artifact/node/variant, fit scope,
source identities, effective controls and learned-state fingerprint. Matching
dimensions alone do not permit replacing a branch or another fit's state.
This is a Python CPU host profile, with no portable Methods/Core numerical
state or R/WASM Torch execution claim. U24's `--fusion early` and `--fusion late`
options constrain the declaration to one real topology while native HPO still
tunes its numeric parameters. Its deterministic arrays are software fixtures.

`cpu_threads=1` signs sequential native scheduling; it does not promise a cap on
Torch's internal numerical threads. These source declarations require the
matching DAG/SDK adapters and Torch dependency. Whole-phase review and release
qualification are separate gates; an older installed runtime is not claimed
to support this profile.
