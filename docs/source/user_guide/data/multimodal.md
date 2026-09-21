# Raw multimodal pipelines

The Python DAG host accepts aligned spectra, images, time series and mixed
metadata through `nirs4all_io.MultimodalDataset`. Images and series keep their
raw dimensions until their pipeline encoder runs. The synthetic example is a
software demonstration; it does not require a real corpus.

This capability currently requires the matching development builds of
`nirs4all`, `nirs4all-io` and `dag-ml`. It is not provided by the older published
packages bearing the same versions. Build wheels from these sources and install
them together; do not infer support from version numbers alone.

## Run the demonstration

From the nirs4all repository, with its development environment installed:

```bash
python examples/user/02_data_handling/U07_multimodal.py --output /tmp/multimodal
```

The script writes `dataset.json`, `pipeline.json`, `tuning.json`, a paired search
checkpoint, `multimodal.n4a` and `report.json`. It trains from the round-tripped
JSON declarations. The report includes native CV/test scores, selected parameters,
eight trial records, twelve new predictions, elapsed time, Python allocation peak
(not total resident memory) and archive size.

To exercise cancellation and resume, use a new output directory:

```bash
python examples/user/02_data_handling/U07_multimodal.py --output /tmp/mm-resume --stop-after 2
python examples/user/02_data_handling/U07_multimodal.py --output /tmp/mm-resume --resume
python examples/user/02_data_handling/U07_multimodal.py --replay /tmp/mm-resume/multimodal.n4a --output /tmp/mm-predict
```

`--search grid` instead evaluates all eight combinations through the existing
native generator. Its run is separate from the resumable random-search profile.

For classification and multiple regression targets, run the second example:

```bash
python examples/user/02_data_handling/U08_multimodal_targets.py --output /tmp/mm-targets --case all
```

`--case classification` uses string labels, `--case regression` uses two complete
targets named `concentration` and `moisture`, and `--case masked` uses partially
observed versions of those targets. These names describe synthetic variables.
All cases use three grouped folds. Classification and masked regression each
run two durable trials; `--resume` reuses their completed search history.

Each case directory contains `dataset.json`, `pipeline.json`,
`prediction_dataset.json`, `multimodal.n4a` and `report.json`, plus the tuning
declaration and checkpoint when applicable. Reports preserve native scores,
per-target metrics, observed target counts and twelve new predictions. The script
reloads each archive with fitting forbidden and checks that predictions agree
exactly. `examples/run.sh` includes U07 through U11; its plot switches have no
effect on these examples, which write artifacts to fresh temporary directories.

## Describe inputs

```python
from nirs4all_io import MultimodalDataset, TensorSource

dataset = MultimodalDataset(
    {
        "nir": TensorSource(spectra, spectral_ids, representation_id="signal_1d",
                            axis_units={"wavelength": "nm"},
                            axis_coordinates={"wavelength": wavelengths}),
        "image": TensorSource(images, image_ids, representation_id="rgb_image"),
        "series": TensorSource(series, series_ids, representation_id="series_mv",
                               axis_units={"time": "s"}, axis_coordinates={"time": times}),
        "metadata": TensorSource(metadata, metadata_ids, representation_id="tabular_mixed",
                                 feature_names=["measurement", "category"]),
    },
    sample_ids=observation_ids, y=targets, groups=subject_ids,
    partitions=partitions,  # train/test; use predict for new unlabelled inputs
)
```

Every observation has a unique string ID. Repetitions have different observation
IDs and share a group ID. IO reorders each source to the canonical observation
order; the default `source_alignment="strict"` rejects missing, extra or duplicate
IDs. A group cannot span train and test.
Use an explicit grouped splitter such as `GroupKFold(3)`.

Source arrays are copied and exposed read-only. Units are supplied by the caller;
no units, resampling, alignment by nearest time, or imputation are inferred.
`dataset.to_dict()` and `MultimodalDataset.from_dict(payload)` preserve the raw
arrays, dtypes, identities and axis metadata through JSON or YAML.

## Handle absent modalities

`source_alignment="left"` explicitly allows a source to contain a subset of the
canonical sample IDs. IO aligns it to that order and records absent rows as
`False` in `TensorSource.presence_mask`. Extra and duplicate IDs still fail.
Alternatively, supply a boolean `presence_mask` with a full source buffer.
`dataset.source_presence(rows=None)` returns the named, read-only masks.
These describe entire source observations, not individual missing pixels.

Models reject absent sources by default. Opt in with
`missing_source_policy="zero_with_indicator"` on `MultimodalRegressor` or
`MultimodalClassifier`. Each encoder fits and transforms only present rows.
The model receives zero embeddings for absent rows and one presence column per
source, including when every row is present. Source weights multiply both the
embedding and its indicator. This is an explicit modelling choice, not a
reconstruction of missing measurements.

Both early and intermediate fusion accept this policy. With partial targets,
each target's encoder uses the intersection of observed target and present
source rows. A source with no such training rows is rejected. At inference,
an entire modality may be absent: retain its name, dtype and trailing dimensions
in a zero-row `TensorSource` and use left alignment. Its fitted encoder is not
called. Presence patterns may change without changing the archive's input schema.
Late fusion and upstream preprocessing outside the multimodal model currently
require complete sources.

The third synthetic example combines incomplete sources with two partial targets,
tunes the grouped pipeline, and predicts new observations with no images:

```bash
python examples/user/02_data_handling/U09_multimodal_missing_sources.py --output /tmp/mm-missing --stop-after 1
python examples/user/02_data_handling/U09_multimodal_missing_sources.py --output /tmp/mm-missing --resume
python examples/user/02_data_handling/U09_multimodal_missing_sources.py --fusion intermediate --output /tmp/mm-missing-mbpls
```

When calling the sklearn operator directly, pass
`source_masks=dataset.source_presence()` to `fit`, `predict` or `predict_proba`.
Public `nirs4all.run()` and archive prediction pass the aligned masks automatically.

## Declare targets

`y` may have shape `(n_samples,)` or `(n_samples, n_targets)`. Use `target_names`
to declare their order, for example `target_names=["concentration", "moisture"]`,
and `task_type="regression"` or `task_type="classification"` to declare the task.
Names survive native scoring and archive replay. Unlabelled prediction datasets
use `y=None`; they can retain `target_names` and `task_type` without target values.

For classification, use `MultimodalClassifier(transformers=..., model=...)` with
a sklearn-compatible classifier, such as `LogisticRegression`. Public
`nirs4all.predict()` returns the original string or numeric class labels, including
non-contiguous numeric labels. The fitted operator exposes `classes_` and offers
`predict_proba()` when its model supports probabilities; probability columns
follow `classes_`. The public archive prediction API currently returns labels,
not a probability matrix. Classification currently has a single target.

Complete multiple regression targets use `MultimodalRegressor` with its default
`target_policy="complete"` and a model supporting multiple outputs, such as Ridge.
Predictions preserve the target dimension. Complete multiple outputs are also
qualified with intermediate MBPLS fusion and native late fusion. Late fusion is
qualified for single-target classification as well.

For partially observed regression targets, declare both the mask and the policy:

```python
dataset = MultimodalDataset(
    sources, sample_ids=sample_ids, y=targets,
    target_names=["concentration", "moisture"],
    target_mask=observed,  # bool, exactly the same shape as targets; True means observed
    task_type="regression",  # mandatory when any target cell is unobserved
    groups=groups, partitions=partitions,
)
model = MultimodalRegressor(
    transformers=transformers, model=Ridge(alpha=1.0), target_policy="per_target",
)
```

Each target gets an independent clone of every encoder and the final model.
Within each training fold, that clone only fits rows observed for that target;
supervised encoders never receive hidden target values. Fitted operators expose
`target_models_` and `target_counts_`. Predictions still cover every requested row
and target. There is no imputation of missing target values.

Masks must be boolean and match `y` exactly. Observed cells must be finite;
masked cells may contain NaN, infinity or hidden finite values. The JSON codec
preserves those values and the mask without invalid JSON numbers. Every target
needs observed training rows, and scored target partitions need observed values.
Native metrics use only observed cells and retain named per-target metrics;
the scalar regression metric is their unweighted mean. A report's `row_count`
counts prediction rows, not the number of observed cells for each target.

The default `target_policy="complete"` rejects partial masks. Partial targets
currently require the early or intermediate `MultimodalRegressor` path; masked
classification, late fusion and group-level score aggregation are rejected.

## Encode, fuse and tune

`MultimodalRegressor(transformers={source_name: transformer, ...}, model=...)`
clones and fits each source transformer on the current training fold. For
example, use `StandardScaler` for spectra, `TensorPCA` for raw fixed-shape image
or series tensors, and `ColumnTransformer` for mixed metadata.

- `fusion="early"` concatenates the resulting numeric encodings.
- `fusion="intermediate"` sends a list of encoded blocks to a multiblock model,
  such as `MBPLS`. Use `standardize=False` if source weights should retain their
  effect through MBPLS.
- Late fusion uses named `by_source` branches, `merge="predictions"` and a
  downstream model. DAG supplies nested grouped OOF predictions to its meta-model.

```python
result = nirs4all.run(
    [GroupKFold(3), {"model": model}], dataset, engine="dag-ml",
    tuning={
        "engine": "n4m", "sampler": "random", "seed": 17, "n_trials": 8,
        "space": {
            "transformers__image__n_components": [2, 4],
            "source_weights__image": [0.5, 1.0],
            "model__alpha": [0.1, 1.0],
        },
        "storage": "file:///absolute/study-directory", "study_name": "example",
    },
    save_charts=False,
)
result.export("multimodal.n4a")
prediction = nirs4all.predict("multimodal.n4a", new_raw_dataset)
result.close()
```

The durable profile covers one `MultimodalRegressor` or `MultimodalClassifier`,
or a complete late-fusion pipeline as described below. Topology is fixed and
proposals use random N4M search. Regression
supports complete or partially observed targets and native regression metrics;
classification supports accuracy and balanced accuracy, defaulting to maximizing
balanced accuracy. It rejects pruning and queued `force_params`.
Regression minimizes RMSE, MSE and MAE, and maximizes R² by default. An explicit
`direction` overrides that default. Classification requires non-overlapping
validation folds; repeated and overlapping holdouts are rejected before fitting
because averaging class indices is not a valid classification reduction.
DAG owns trial execution, scores and selection. The winner then
uses the existing native CV/refit path. Public tuning results use canonical
dotted parameter paths. Trial budget is total, including failed trials, and can
be increased on resume.

The tuning objective averages metrics across folds. `RunResult.cv_best_score`
uses the pooled out-of-fold predictions. These values can differ with unequal
fold sizes or nonlinear metrics such as RMSE and balanced accuracy; reports
retain both values and their native score evidence.
These CV values participate in hyperparameter selection; use the independent
test partition to evaluate the selected pipeline.

`tuning.progress_callback(event)` may return `False` between trials. Cancellation
raises `MultimodalTuningStopped`; a model failure is checkpointed and its original
error propagates. `resume=True` continues from terminal history. A crash during
an unfinished trial returns to the preceding checkpoint and may repeat that
unfinished trial. Changing training buffers, labels, schemas, folds or search
semantics refuses resume. Target names, task type and training masks are part of
the search identity. Source presence is also fingerprinted. Hidden target/source
values and held-out test data are excluded from search; modifying them does not
create a new search objective.

The host resets the Python/NumPy RNG for each native task using a stable seed
derived from `run(random_state=...)`, otherwise the tuning seed, otherwise zero.
This makes supported sklearn operators using the global RNG reproducible across
resume even when a progress callback consumes randomness. Explicit operator
seeds remain effective. Custom generators and GPU determinism are not guaranteed
by this policy.

## Tune a late-fusion ensemble

Use the existing `by_source` branches followed by `{"merge": "predictions"}`
and a meta-model. Pass a global `tuning` mapping to `nirs4all.run()`:

```python
tuning = {
    "engine": "n4m", "sampler": "random", "seed": 17, "n_trials": 4,
    "space": {
        "branches.image.0.n_components": [2, 4],
        "branches.nir.1.alpha": [0.1, 1.0],
        "branches.metadata.0.numeric.with_mean": [False, True],
        "meta.alpha": [0.1, 1.0],
    },
    "storage": "file:///absolute/late-study", "study_name": "late-fusion",
}
```

`branches.<source>.<step_index>.<parameter>` addresses the named source and
zero-based position in its public step list. Positions include `None` entries;
those entries themselves cannot be tuned. `meta.<parameter>` addresses the
final estimator. Nested sklearn parameters accept dots or `__`, as with
`branches.metadata.0.numeric__with_mean`. The example assumes a PCA at image
position 0, Ridge at NIRS position 1 and a metadata `ColumnTransformer` with
a transformer named `numeric` at position 0.

Every candidate recomputes all branch encoders, base models and inner OOF
predictions through the native scheduler. The meta-model fits those inner
predictions and is scored on the outer validation folds. Training and test
identities remain separate. Search paths and their destination nodes are part
of the checkpoint identity; changing their routing refuses resume.

The returned `RunResult` represents the selected **ensemble**. Its direct
`export()` saves that ensemble even if an individual base model scores better.
The profile requires complete sources and targets, instantiated sklearn
operators, and plain model steps. Model-local `finetune_params`, `train_params`
and `refit_params` are refused in this global search profile. Single-target
classification and complete multi-target regression use the same graph.

The synthetic U10 example writes the recipe, cohort, checkpoint, report and
archive, then predicts twelve new observations with fit forbidden:

```bash
python examples/user/02_data_handling/U10_multimodal_late_tuning.py --output /tmp/mm-late --stop-after 2
python examples/user/02_data_handling/U10_multimodal_late_tuning.py --output /tmp/mm-late --resume
```

## Executable synthetic data providers

Pass an IO `DataProvider` directly as `dataset`. DAG-ML executes its source node
once in `PLAN`, before constructing cross-validation folds. The callback returns
an IO `MultimodalDataset`; IO validates raw shapes, identities and alignment.
The provider does not learn from observations. Learned augmentation belongs in
a fold-scoped training controller.

```python
from nirs4all_io import DataProvider

def generate_cohort(*, seed, params, context):
    # Reuse a scientific generator and return aligned raw TensorSource objects
    # inside a MultimodalDataset, with explicit groups and partitions.
    return make_synthetic_cohort(seed=seed, n_samples=params["n_samples"])

provider = DataProvider(
    generate_cohort, provider_id="my.synthetic.cohort", provider_version="1",
    params={"n_samples": 64}, seed=17,
)
result = nirs4all.run(pipeline, provider, save_charts=False)
```

The provider's root seed is separate from the model's `random_state`. Native
PLAN derives and records the effective task seed passed to the callback.
`provider.materialize()` is also available outside a run and uses the provider's
root seed directly; pass an explicit `seed` to reproduce a native realization.
Callbacks must use their seed and parameters deterministically. Change
`provider_version` when changing the recipe's implementation.

For partial production, supply `base=fixed_cohort` and return a mapping with
`sample_ids` plus `sources` and/or `y`. IO aligns it to the base's fixed universe.
Adding sources is allowed; replacing existing sources or targets requires
`replace_sources=("image",)` or `replace_targets=True`. Groups and partitions
remain explicit and the base is not mutated.

After materialization, `provider.get(sample_ids)` and
`provider.batches(batch_size, sample_ids=training_ids)` return aligned cohort
views without calling the generator again. A batch iterator retains its cursor:

```python
batches = provider.batches(16)
first = next(batches)
state = batches.state_dict()  # JSON-serializable

restored = DataProvider(generate_cohort, provider_id="my.synthetic.cohort",
                        provider_version="1", params={"n_samples": 64}, seed=17)
restored.load_state_dict(state["provider"])
remaining = restored.batches(16)
remaining.load_state_dict(state)
```

Restoring the provider explicitly regenerates and verifies the recorded content
before publishing it. Restoring the iterator checks the recipe, content,
selection and batch configuration before accepting its cursor. This is a finite,
materialized provider: batching does not make generation itself out-of-core.
One run uses one fixed realization across folds and HPO trials; a changed recipe
invalidates HPO resume. Dynamic row creation, learned generators, infinite
streams and native per-epoch regeneration are outside this profile. A model
controller may iterate the fixed cohort for its epochs.

`nirs4all_io.provider_adapters.SklearnProviderAdapter(provider, source="nir")`
exposes `.arrays(training_ids)` as `(X, y)` for an ordinary estimator's `fit`.
Its `.batches(...)` supplies arrays for caller-managed `partial_fit` when the
estimator actually provides that method. N-D sources require an explicit
encoder; omitting `source` returns a mapping for source-aware estimators.

The optional `TorchMapDataset` and `TorchIterableDataset` classes in the same
module implement PyTorch's real dataset interfaces. Pass `sample_ids=training_ids`
to constrain a loader to a fold. Iterable workers shard sample positions without
duplication. `return_metadata=True` retains IDs, groups, partitions and masks;
tuple mode refuses to discard missingness masks. Use
`collate_fn=collate_provider_samples` for mixed categorical/numeric sources.
Adapters capture the realized cohort, not the generator callback. Their dataset
values are independent of worker count; iterable delivery order may differ.
The provider cursor checkpoint does not capture DataLoader prefetch queues or a
model/optimizer checkpoint. PyTorch is imported only when explicitly requested.

The source recipe, effective seed, native lineage and content fingerprint are
retained in run metadata and exported host models. Prediction uses the fitted
model and explicitly supplied new inputs; it never executes the training
provider. U11 demonstrates full and partial production, reuses the existing NIRS
synthesis API, restores a batch cursor and exports/replays a model:

```bash
python examples/user/02_data_handling/U11_multimodal_data_provider.py --output /tmp/mm-provider
# With the optional CPU/GPU PyTorch dependency installed:
python examples/user/02_data_handling/U11_multimodal_data_provider.py --output /tmp/mm-provider-torch --torch-workers 2
```

## Archive and scope

The archive contains fitted encoders, fusion and final model, raw input contracts,
selected recipe and exact host dependency versions. Replay accepts new observation
IDs and source mapping order but rejects source, dimension, dtype, feature name,
axis, coordinate or unit changes. It performs the DAG PREDICT phase without fit
and does not require the training workspace.

These are Python host archives containing trusted joblib objects, not portable
Core/ONNX/WASM model archives. Replay requires the declared package versions and
Python major/minor version. Inputs have fixed trailing dimensions, with optional
source and target masks as described above. Ragged inputs and cross-language
replay are not qualified. Classification encodings and inferred task types depend
only on training targets; a held-out class absent from that vocabulary is refused.

See the [installation qualification](../../../multimodal_installation_qualification.md)
for the independent wheel installation and replay with fit and source-directory
access forbidden, and `scripts/verify_multimodal_archive.py` to repeat that check.
The [targets and missing-source qualification](../../../multimodal_targets_and_missing_sources_qualification.md)
records the U08/U09 checks and their current validation status.
The [late-fusion tuning qualification](../../../multimodal_late_tuning_qualification.md)
records U10, durable whole-stack search and its independent installation proof.
