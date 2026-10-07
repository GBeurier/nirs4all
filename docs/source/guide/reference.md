# 13. Find a node, parameter, example or solution

**Use this page as an index.** Start with the user explanation and worked example; open the exact reference when you need spelling, defaults or availability. The {doc}`node catalogue </reference/nodes/index>` is the visual entry point for pipeline operations.

## Pipeline nodes, one family at a time

| What you want to do | User page with code and figure | Exact reference |
|---|---|---|
| Read the basic step forms | {doc}`operator step </reference/nodes/operator_step>` | {doc}`/reference/pipeline_syntax` |
| Change spectral/feature values | {doc}`preprocessing </reference/nodes/preprocessing>` | {doc}`/reference/transforms` |
| Change y for fitting and restore units | {doc}`target processing </reference/nodes/y_processing>` | {doc}`/reference/pipeline_keywords` |
| Fit a regressor or classifier | {doc}`model </reference/nodes/model>` | {doc}`/reference/models` |
| Compare alternatives before fitting | {doc}`generators </reference/nodes/generators>` | {doc}`/reference/generator_keywords` |
| Hold out rows/groups for validation | {doc}`split </reference/nodes/split>` | {doc}`/reference/splitters` |
| Duplicate paths or route sources/rows | {doc}`branch </reference/nodes/branch>` | {doc}`/user_guide/pipelines/branching` |
| Join features, predictions or row subsets | {doc}`merge </reference/nodes/merge>` | {doc}`/user_guide/pipelines/merging` |
| Annotate observations | {doc}`tag </reference/nodes/tag>` | {doc}`/reference/pipeline_keywords` |
| Exclude permitted training rows | {doc}`exclude </reference/nodes/exclude>` | {doc}`/reference/filters` |
| Make derived training measurements | {doc}`sample augmentation </reference/nodes/sample_augmentation>` | {doc}`/reference/augmentations` |
| Retain extra feature views | {doc}`feature augmentation </reference/nodes/feature_augmentation>` | {doc}`/user_guide/preprocessing/overview` |
| Concatenate transformed columns | {doc}`concat_transform </reference/nodes/concat_transform>` | {doc}`/reference/pipeline_keywords` |
| Reshape related scans | {doc}`repetitions </reference/nodes/repetitions>` | {doc}`/user_guide/data/heterogeneous_repetitions` |
| Add inspection figures | {doc}`charts </reference/nodes/charts>` | {doc}`/user_guide/visualization/prediction_charts` |
| Choose transfer-oriented preprocessing | {doc}`auto-transfer </reference/nodes/auto_transfer_preproc>` | {doc}`/reference/pipeline_keywords` |
| Learn a correction to a base predictor | {doc}`residual model </reference/nodes/residual>` | {doc}`/reference/models` |

## Data and modalities

| Question | Worked guide/example | What to inspect |
|---|---|---|
| How do I load arrays, files or partitions? | {doc}`datasets`, {doc}`/user_guide/data/loading_data` | Rows, features, target and partitions |
| How do I combine spectrum, image, series and metadata? | {doc}`/user_guide/data/methods_multimodal_u07` | Source encoders, aligned IDs and fusion |
| What if a modality is missing? | {doc}`/user_guide/data/multimodal_late_partial` | Presence masks and supported fusion policy |
| What if a target is missing? | {doc}`/user_guide/data/multimodal` | Observed-target masks and per-target fitting |
| What if series have different lengths? | {doc}`/user_guide/data/multimodal` | Offsets, time coordinates and explicit encoder |
| What if I have repeated scans? | {doc}`/user_guide/data/aggregation` | Specimen groups, origin IDs and scoring unit |
| What combinations can my runtime execute? | {doc}`/reference/multimodal_execution_matrix` | Actual source/model/export combination |

Executable source companions include [U07 multimodal](https://github.com/GBeurier/nirs4all/blob/main/examples/user/02_data_handling/U07_multimodal.py), [U08 targets](https://github.com/GBeurier/nirs4all/blob/main/examples/user/02_data_handling/U08_multimodal_targets.py), [U09 missing sources](https://github.com/GBeurier/nirs4all/blob/main/examples/user/02_data_handling/U09_multimodal_missing_sources.py), [U12 ragged series](https://github.com/GBeurier/nirs4all/blob/main/examples/user/02_data_handling/U12_multimodal_ragged_series.py), and [U13 late missing sources](https://github.com/GBeurier/nirs4all/blob/main/examples/user/02_data_handling/U13_multimodal_late_missing_sources.py).

## Search, training and selection

| I need… | Start here | Meaning |
|---|---|---|
| A small explicit comparison | {doc}`pipelines` | Generate separate candidate recipes |
| Model-local parameter search | {doc}`/user_guide/models/hyperparameter_tuning` | Tune one model in its allowed training scope |
| Native fold-safe PLS search | {doc}`/user_guide/models/native_pls_fold_hpo` | Keep fitted preprocessing inside training folds |
| Structural search | {doc}`/user_guide/models/structural_hpo` | Compare chains, source subsets and fusion structures |
| Durable native tuning and resume | {doc}`/user_guide/models/native_tuning_conformal` | Retain the real checkpoint/search identity |
| Three training-parameter scopes | {ref}`three-training-parameter-scopes` | Trial controls, ordinary fitting and refit overrides |
| Retraining or weight continuation | {doc}`/user_guide/deployment/retrain_transfer` | New fitting campaign versus model-specific continuation |
| Exact tuning keyword spelling | {doc}`/reference/pipeline_keywords` | Nested fields, defaults and runtime availability |

**Search changes the comparison, not the amount of independent evidence.** Keep the final test cohort outside candidate selection. Resume the same search only with compatible data, recipe, objective and checkpoint.

## Scores, uncertainty and stored results

| I need… | Open… | Check |
|---|---|---|
| RMSE, R² or classification metric definition | {doc}`evaluation`, {doc}`/reference/metrics` | Unit, direction, valid count and aggregation |
| CV selection versus refit test score | {doc}`/user_guide/scoring_and_refit` | Which observations produced each score |
| Diagnostic plots | {doc}`results`, {doc}`/user_guide/visualization/prediction_charts` | Partition, target names and readable values |
| Conformal intervals | {doc}`/user_guide/models/native_tuning_conformal` | Separate calibration/test data and achieved coverage |
| A fixed-model robustness audit | {ref}`planned-robustness-campaigns` | Scenario, severity, reference data and metric changes |
| Stored prediction queries | {doc}`/reference/predictions_api` | Run, candidate, fold, partition and sample IDs |
| Workspace/session inspection | {doc}`results`, {doc}`interop` | Stored state versus open runtime resources |
| Workspace/storage schema | {doc}`/reference/workspace`, {doc}`/reference/storage` | Actual producer format and selected-model links |

## Deployment and language help

- {doc}`languages`: one shared native recipe with actual Python/R/Octave/WASM calls.
- {doc}`deployment`: export, cold reload, raw-input prediction and diagnosis.
- {doc}`interop`: moving a supported native predictor or full workspace.
- {doc}`/user_guide/deployment/export_bundles`: complete fitted bundle.
- {doc}`/reference/public_interfaces`: exact full Python SDK surface.
- {doc}`/reference/cli`: command groups, arguments and output files.

A recipe, a fitted model, an optimizer checkpoint and a workspace contain different things. Use the loader corresponding to the producing API. Keep exact native package text when large signed integers are present.

## Diagnose a problem by what you see

| Symptom | Inspect first | Concrete next action |
|---|---|---|
| Missing runtime or method | Language package, CLI/library path and versions | Follow {doc}`interfaces` and the installed runtime setup |
| Wrong array dimensions | Observation/feature axes and smallest fold size | Print shapes before fitting |
| Source schema mismatch | Source IDs, feature order, units and coordinates | Compare with the saved model declaration |
| Group/fold refusal | Which scans belong to one specimen | Keep the independent unit together; use explicit supported folds |
| Resume mismatch | Data, space, metric and actual checkpoint | Start a new study if the experiment changed |
| Model export cannot load | Original producer, bytes and supported consumer | Use the matching loader/profile |
| Correct values on wrong rows | Sample IDs across sources and output | Join by IDs and retain ordering |
| Unexpected fit during prediction | Whether fitted state was loaded | Call the prediction API on the saved predictor |

Further help: {doc}`/user_guide/troubleshooting/faq`, {doc}`/user_guide/troubleshooting/dataset_troubleshooting`, {doc}`/user_guide/troubleshooting/migration`.

```{dropdown} For validators, forms and advanced tools

The documentation build emits the <a href="../_static/keyword-registry.json">keyword inventory</a>, <a href="../_static/keyword-registry.schema.json">keyword schema</a>, <a href="../_static/tuning-summary.schema.json">tuning summary schema</a> and <a href="../_static/robustness-summary.schema.json">robustness summary schema</a>.

The public registry includes exact paths, value schemas, scopes, engine support and documentation anchors. A schema checks document structure; runtime checks additionally establish data identity, fitted state and supported execution. Consult {doc}`/reference/native_capability_preflight` for refusals.
```
