# Choose a pipeline node by what you want to do

A pipeline is a recipe: start with observations, prepare useful features,
evaluate a model on held-out observations, and save the resulting predictor.
Each **node** performs one step. Click a node below for its worked recipe,
explanatory figure, expected output and common mistakes.

New to pipelines? Read {doc}`operator_step`, {doc}`preprocessing`, {doc}`split`
and {doc}`model` first. Then add views, paths or searches as your question requires.

## Start with the ordinary learning workflow

| Your goal | Node | What changes |
|---|---|---|
| Describe an algorithm and its settings | {doc}`operator_step` | Declares the operator |
| Normalize, smooth or reduce features | {doc}`preprocessing` | Feature values or columns |
| Define cross-validation observations | {doc}`split` | Which rows train/validate each fit |
| Predict concentration or class | {doc}`model` | Learns a predictor |
| Scale/encode targets and invert predictions | {doc}`y_processing` | Target representation |
| Inspect spectra, targets and folds | {doc}`charts` | Adds a diagnostic figure |

## Add information without confusing rows and columns

| Your goal | Node | What changes |
|---|---|---|
| Retain raw, normalized and derivative views | {doc}`feature_augmentation` | View axis |
| Join several transformed feature blocks | {doc}`concat_transform` | Column count |
| Create perturbed training observations | {doc}`sample_augmentation` | Training row count |
| Annotate unusual observations | {doc}`tag` | Metadata, no deletion |
| Remove flagged rows from training | {doc}`exclude` | Fitting population |
| Handle several scans of each material | {doc}`repetitions` | Grouping or repetition representation |

## Compose more advanced experiments

| Your goal | Node | What to learn next |
|---|---|---|
| Keep several preprocessing/model paths | {doc}`branch` | Duplication, separation and source routing |
| Join features, model predictions or rows | {doc}`merge` | Output axes and OOF stacking |
| Compare many alternative recipes | {doc}`generators` | Two transforms × three complexities = six recipes |
| Learn errors left by a base model | {doc}`residual` | Base prediction plus correction |
| Adapt preprocessing to another instrument | {doc}`auto_transfer_preproc` | Adaptation cohort and recommendations |

## Complete spelling index

| Spellings | Explanation |
|---|---|
| `class`, `function`, `instance`, direct object, import-path string | {doc}`operator_step` |
| `params`, `name`, `force_layout` | {doc}`operator_step` |
| `model` | {doc}`model` |
| `split`, direct splitter object | {doc}`split` |
| `preprocessing`, bare transformer, sequential subpipeline list | {doc}`preprocessing` |
| `y_processing` | {doc}`y_processing` |
| `feature_augmentation`, `action` | {doc}`feature_augmentation` |
| `concat_transform` | {doc}`concat_transform` |
| `sample_augmentation` | {doc}`sample_augmentation` |
| `tag`, `exclude` | {doc}`tag`, {doc}`exclude` |
| `branch`, `by_source`, `by_metadata`, `by_tag`, `by_filter` | {doc}`branch` |
| `merge`, `merge_sources`, `merge_predictions` | {doc}`merge` |
| `rep_to_sources`, `rep_to_pp`, `rep_fusion` | {doc}`repetitions` |
| `residual` | {doc}`residual` |
| `auto_transfer_preproc` | {doc}`auto_transfer_preproc` |
| `chart_2d`, `chart_3d`, `y_chart`, `chart_y` | {doc}`charts` |
| `fold_chart`, `chart_fold`, `fold_*` | {doc}`charts` |
| `spectra_dist`, `spectral_distribution`, `spectra_envelope` | {doc}`charts` |
| `augment_chart`, `augmentation_chart`, `augment_details_chart`, `augmentation_details_chart` | {doc}`charts` |
| `exclusion_chart`, `chart_exclusion` | {doc}`charts` |
| `_or_`, `_range_`, `_log_range_`, `_grid_`, `_zip_`, `_cartesian_`, `_chain_`, `_sample_` | {doc}`generators` |
| `pick`, `arrange`, `then_pick`, `then_arrange`, `count` | {doc}`/reference/generator_keywords` |
| `_seed_`, `_weights_`, `_mutex_`, `_requires_`, `_exclude_`, `_preset_` | {doc}`/reference/generator_keywords` |
| `_tags_`, `_metadata_`, `_depends_on_` | {doc}`/reference/generator_keywords` (advanced/reserved limits) |

## Choose the execution environment once

A recipe file describes intent. The runtime must actually support its algorithms
and topology. The worked tabs identify which route is used:

| Route | Appropriate scope |
|---|---|
| Native language facade | Qualified finite method recipes in Python/R/Octave/WASM |
| Python `dag-ml` | Supported host operators and graph workflows |
| Python `legacy` | Historical host controllers explicitly selected in examples |

These are not interchangeable claims of portability. Complex Python controller
recipes cannot be made native simply by changing JSON keys. Start with
{doc}`/guide/languages` for complete examples in your language. Consult
{doc}`/reference/native_capability_preflight` or
{doc}`/reference/multimodal_execution_matrix` when extending the topology.

```{toctree}
:maxdepth: 1
:hidden:

operator_step
model
split
preprocessing
y_processing
feature_augmentation
sample_augmentation
concat_transform
auto_transfer_preproc
tag
exclude
branch
merge
repetitions
residual
charts
generators
```

Full schemas and internal parser details remain in {doc}`/reference/pipeline_keywords`.
The {doc}`/reference/operator_catalog` lists algorithms, distinct from workflow nodes.
