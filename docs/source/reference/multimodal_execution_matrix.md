# Public execution matrix

This matrix describes the current multimodal development candidate. **The new
profiles below await the complete-phase review and installed-package test run.**
Existing test files and earlier release results are not proof that this candidate
passes. Its qualification must cover integrations, official examples and fresh
process archive replay with matching DAG-ML, IO, Methods and Core builds.

`Existing` means an established public route with regression witnesses.
`New` means source and witnesses have been written in this phase. Both require
the final qualification. `Refused` identifies a combination outside the stated
profile; it is not retried through the legacy engine. Each row is bounded by its
own contract: support for two rows does not imply support for their combination.

## Python training and replay

Unless stated otherwise, these routes use `nirs4all.run(..., engine="dag-ml")`.
Python estimator archives require a trusted producer and the recorded Python
dependencies. They do not become R/WASM models by using the `.n4a` extension.

| Public journey | Code status | Supported contract and explicit boundary | Regression or new integration witnesses |
| --- | --- | --- | --- |
| Simple NIRS regression, CV or full training | Existing | Dense array/dict/directory inputs, fold-local transforms, native scores and captured replay. Explicit engine selection stays strict. | `test_general_dag_run.py`, `test_full_train_dag_run.py`, `test_general_dag_replay.py` |
| Parallel branches and concatenated features | Existing | Declared branches and source layout; export retains the fitted transforms and exact concatenation order. Unconsumed merge options and ambiguous flat replay are refused. | `test_source_concat_public.py`, `test_named_dag_stacking.py` |
| Complete multisource early/intermediate fusion | Existing | Typed signal/image/series/metadata sources keep their dimensions until each encoder. Train/Test and groups remain explicit; unknown IDs, changed units/axes or incompatible dimensions fail. | `test_multimodal_dagml.py`, `test_multimodal_targets.py` |
| Operator/parameter generators | Existing; new `by_source` capture route | Native enumeration and selection. Captured `by_source` operator choices preserve per-source producer identities. Mixing operator and parameter generation in that new capture route is refused. | `test_attested_by_source_packages.py`; generator parity cases |
| Ranked `by_source` selection, Python or CLI | New | Original native ranking, a signed requested rank and a real CV/REFIT capture for each rank. Repeated CV work is intentional. Dynamic HPO-study top-k is refused. | `test_attested_by_source_packages.py` |
| Feature/sample augmentation on supported dense pipelines | Existing | Train-only augmentation and repetition/origin lineage; feature `extend`, `add` and `replace` retain their replay layout. This does not enable augmentation of named Torch inputs or partial-source late stacks. | `test_repeated_feature_augmentation.py`, `test_dagml_cv_augmented_train_view.py`, `test_dagml_full_train_augmentation.py` |
| Prediction stacking | Existing | Native inner OOF joins and learned meta-model, with genuine group/fold scopes. Training predictions cannot silently replace OOF features. | `test_source_stacking_native.py`, `test_resampled_dag_stacking.py`, `test_dagml_shared_stacking.py` |
| Target transformations | Existing | Target transforms remain fold-local and captured for replay on supported complete-data routes. Separate target-transform steps are refused by the new partial late and experimental-unit profiles. | `test_general_dag_run.py`, `test_residual_target_preprocessing.py` |
| Partial-source early/intermediate fusion | Existing | Explicit `zero_with_indicator`, per-source presence and optional `per_target` regression. Encoders fit only present/observed rows; target values are never imputed. | `test_multimodal_missing_sources.py`, `test_multimodal_partial_targets.py` |
| Partial-source late regression/classification | New | Two to four branches, true inner OOF, one learned fusion model, global stack HPO, complete mono-target classification or per-target regression. Classes must remain available in each fitting scope. Branch-local HPO, mixed task types and partial class labels are refused. | `test_multimodal_late_partial.py`, `test_incomplete_source_stacking_execution.py` |
| Upstream preprocessing before partial `by_source` | New | Ordinary reconstructible X transforms are cloned per source and fitted inside each native source/target training intersection. Whole-stack HPO addresses the expanded chain; selected REFIT and replay apply it once. Global-fit/transfer/structured operations stay refused. Experimental units require explicit weighted support in every upstream and branch component before compilation/ask. | `test_incomplete_source_stacking_upstream.py`; `test_source_stacking_upstream.py` (unit) |
| Late archive capture and cold prediction | New | Exact REFIT components, masks, vocabularies and per-target row identities; target-free prediction can omit an entire declared source. Original native artifact identities and serialized-file hashes are checked separately before unpickling. | `test_multimodal_late_missing_export.py`, `test_attested_by_source_packages.py` |
| Ragged series via `SequenceSummary` | Existing; new domain constraints | Packed variable lengths, fixed channels; explicit minimum length and inclusive channel bounds are checked at fit and predict. No implicit interpolation, clipping or padding; time coordinates do not weight the summary. | `test_multimodal_series_domain.py`; official U12 |
| Independent experimental units | New | Explicit unit/repetition identities, native scope-local equal-unit weights and unit-level scores, while OOF/predictions stay observation-level. Weighted sklearn components are required; no automatic substitution of split groups. | `test_multimodal_experimental_units.py` |
| Joint named Torch fusion | New | CPU, two to four complete numeric fixed-shape sources, raw axes passed to user `forward(**inputs)`, one complete numeric target, Adam/MSE, native CV/REFIT and Python replay. Ragged joint tensors, missing modalities, mixed metadata, other frameworks and portable Torch weights are outside this profile. | `test_named_torch_training.py`, `test_named_torch_nd.py` |
| Finite data provider | Existing | The provider supplies a bounded dataset/view under the native run/fold lifecycle. Prediction replay does not regenerate the training data. Infinite streams and checkpointed mid-epoch continuation are separate work. | `test_data_provider.py`, `test_data_provider_views_probe.py`; official U11/U12 |

The named files live under `tests/integration/api/`, except the augmentation and
residual-target parity witnesses under `tests/integration/parity/`. The
[multimodal guide](../user_guide/data/multimodal.md) gives input and model syntax;
[partial late fusion](../user_guide/data/multimodal_late_partial.md) describes its
mask and archive rules. See [pipeline keywords](pipeline_keywords.md) for the
established generator, branch and model-local tuning syntax.

## Native portability

| Public journey | Code status | Boundary |
| --- | --- | --- |
| Methods Archive V2 and four-modality U07 replay | Existing | Real exported Methods states, native DAG scheduling and exact byte identities. Python/R/WASM role adapters have bounded supported recipes. |
| Native full-refit Archive V3 with N4ME states | New | A genuine new REFIT produces V3; DAG owns semantics and Core owns ZIP/inventory validation. `NativeMethodsRefitResult.from_package_json`, `export`, `load_archive` and `predict` expose the fitted child in Python. This does not add every N4ME method to `nirs4all.run(engine="native")`. |
| Native PLS RolePipeline in V3 | New | The persisted native controller and its selected recipe survive export/replay. Host Python/R wrappers cannot impersonate this controller. |
| Autonomous R ZIP replay | New | Public R APIs call native Core ZIP validation and DAG replay without Python. Direct Methods replay covers native Role V2/V3 and N4ME V3; the separate process adapter covers the explicit U07 recipe in V2. A process V3 route requires a genuine compatible producer and is not qualified by the direct Methods witnesses. |
| Web full-refit and prediction replay | New | Native DAG initial full-refit and replay operations schedule the browser controller. The fitted browser pipeline is an explicit JSON sidecar, not a native RAW model. Existing cross-validation preprocessing/branches remain host-controlled on native folds. |
| Web grouped folds and observation lineage | New | Explicit group/origin/repetition/augmentation roles pass from input metadata to DAG data. Native GroupKFold uses declared groups; missing identities or unsupported combinations are refused. |
| Torch/joblib in R or WASM, arbitrary mixed-language nodes | Refused | Deferred portability profiles; no conversion or host-runtime fallback is implied. |

The candidate's release checks must exercise independent numerical oracles,
unequal repetitions and leakage refusals, exact input-contract refusals, and
fresh-process replay with FIT/HPO disabled. Small deterministic multimodal
datasets used for these checks are test fixtures. The existing NIRS synthetic
generator remains a separate product feature.
