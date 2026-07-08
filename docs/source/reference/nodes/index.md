# Pipeline Nodes Reference

This section answers the practical question: **what can I put in `pipeline.yaml` or `pipeline.json`?**

NIRS4ALL parses each pipeline entry as one of these node types:

- a serialized operator (`class`, `function`, `instance`, `module`, `object`, `pipeline`);
- a workflow keyword (`model`, `split`, `branch`, `merge`, `feature_augmentation`, and others);
- a generator keyword (`_or_`, `_range_`, `_cartesian_`, and related search syntax);
- a direct Python object, when you build the pipeline in Python instead of YAML/JSON.

:::{note}
This table describes the full NIRS4ALL pipeline language. The `dag-ml` engine currently covers a subset of pipeline shapes natively. When you request `engine="dag-ml"`, catchable unsupported shapes or unavailable runtime dependencies warn and fall back to the legacy Python engine. Genuine dag-ml runtime errors are not silently swallowed.
:::

```{toctree}
:maxdepth: 1

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

## Complete Node Table

| Node / keyword | Runs in YAML/JSON | Runs as Python object | Purpose | Dedicated page |
| --- | --- | --- | --- | --- |
| `class` | Yes | N/A | Instantiate a Python class by import path. | {doc}`operator_step` |
| `function` | Yes | N/A | Reference a callable by import path. | {doc}`operator_step` |
| `instance` | Yes | N/A | Restore a serialized runtime instance when produced by NIRS4ALL internals. | {doc}`operator_step` |
| direct operator | No | Yes | Use an already-created sklearn/nirs4all object. | {doc}`operator_step` |
| string import path | Yes | Yes | Short serialized class/function syntax such as `sklearn.preprocessing.StandardScaler`. | {doc}`operator_step` |
| `model` | Yes | Yes | Mark an estimator as the supervised model step. | {doc}`model` |
| `split` | Yes | Yes | Create CV folds from a splitter object or load fold files. | {doc}`split` |
| splitter object | No | Yes | Any sklearn-compatible object with `split(X, ...)`. | {doc}`split` |
| `preprocessing` | Yes | Yes | Explicit preprocessing wrapper for one operator or a list of operators. | {doc}`preprocessing` |
| bare transformer | Yes, via `class` | Yes | Apply an sklearn-compatible transformer to X. | {doc}`preprocessing` |
| `y_processing` | Yes | Yes | Transform targets during training and inverse-transform predictions. | {doc}`y_processing` |
| `feature_augmentation` | Yes | Yes | Create multiple feature views from preprocessing operators. | {doc}`feature_augmentation` |
| `sample_augmentation` | Yes | Yes | Add augmented training samples; skipped during prediction. | {doc}`sample_augmentation` |
| `concat_transform` | Yes | Yes | Apply several transforms and concatenate their feature outputs. | {doc}`concat_transform` |
| `auto_transfer_preproc` | Yes | Yes | Select transfer preprocessing automatically for transfer workflows. | {doc}`auto_transfer_preproc` |
| `tag` | Yes | Yes | Mark samples for later analysis/branching without removing them. | {doc}`tag` |
| `exclude` | Yes | Yes | Exclude flagged training samples. | {doc}`exclude` |
| `branch` | Yes | Yes | Create duplication or separation branches. | {doc}`branch` |
| subpipeline list | Yes | Yes | Nested list of steps, usually inside `branch`. | {doc}`branch` |
| `merge` | Yes | Yes | Merge branch features, predictions, separation branches, or sources. | {doc}`merge` |
| `merge_sources` | Yes | Yes | Source-merge alias handled by the merge controller. | {doc}`merge` |
| `merge_predictions` | Yes | Yes | Prediction-merge alias handled by the merge controller. | {doc}`merge` |
| `rep_to_sources` | Yes | Yes | Convert repetitions into multi-source layout. | {doc}`repetitions` |
| `rep_to_pp` | Yes | Yes | Convert repetitions into preprocessing-pipeline views. | {doc}`repetitions` |
| `rep_fusion` | Yes | Yes | Materialize relation-aware repetition/source fusion. | {doc}`repetitions` |
| `residual` | Yes | Yes | Fit a base model plus residual learner. | {doc}`residual` |
| `chart_2d`, `chart_3d` | Yes | Yes | Save spectral scatter/visualization charts. | {doc}`charts` |
| `y_chart`, `chart_y` | Yes | Yes | Save target-distribution charts. | {doc}`charts` |
| `fold_chart`, `chart_fold`, `fold_*` | Yes | Yes | Save fold visualization charts. | {doc}`charts` |
| `spectra_dist`, `spectral_distribution`, `spectra_envelope` | Yes | Yes | Save spectral envelope/distribution charts. | {doc}`charts` |
| `augment_chart`, `augmentation_chart` | Yes | Yes | Save augmentation comparison charts. | {doc}`charts` |
| `augment_details_chart`, `augmentation_details_chart` | Yes | Yes | Save detailed augmentation charts. | {doc}`charts` |
| `exclusion_chart`, `chart_exclusion` | Yes | Yes | Save included/excluded sample charts. | {doc}`charts` |
| `force_layout` | Yes | Yes | Metadata key that forces `2d`, `2d_interleaved`, `3d`, or `3d_transpose` layout for a step. | {doc}`operator_step` |
| `name` | Yes | Yes | Metadata key for display/report names. | {doc}`operator_step` |
| `_or_` | Yes | Yes | Expand alternatives. | {doc}`generators` |
| `_range_` | Yes | Yes | Expand numeric ranges. | {doc}`generators` |
| `_log_range_` | Yes | Yes | Expand log-spaced numeric ranges. | {doc}`generators` |
| `_grid_` | Yes | Yes | Expand a parameter grid. | {doc}`generators` |
| `_zip_` | Yes | Yes | Expand parallel choices position-by-position. | {doc}`generators` |
| `_chain_` | Yes | Yes | Build ordered chains from generated components. | {doc}`generators` |
| `_sample_` | Yes | Yes | Randomly sample generated choices. | {doc}`generators` |
| `_cartesian_` | Yes | Yes | Explicit cartesian product expansion. | {doc}`generators` |
| `pick`, `arrange`, `then_pick`, `then_arrange`, `count` | Yes | Yes | Selection modifiers for generated alternatives. | {doc}`generators` |
| `_mutex_`, `_requires_`, `_depends_on_`, `_exclude_` | Yes | Yes | Constraints on generated combinations. | {doc}`generators` |
| `_preset_`, `_tags_`, `_metadata_`, `_seed_`, `_weights_` | Yes | Yes | Generator presets, annotations, reproducibility, and sampling weights. | {doc}`generators` |

## Parser Priority

When a step is a dictionary, NIRS4ALL applies this order:

1. Serialization operators first: `class`, `function`, `module`, `object`, `pipeline`, `instance`.
2. Priority workflow keywords: `model`, `preprocessing`, `feature_augmentation`, `auto_transfer_preproc`, `concat_transform`, `y_processing`, `sample_augmentation`, `branch`.
3. The first non-reserved key is used as a workflow keyword and routed to matching controllers.
4. If nothing matches, the dictionary is treated as a serialized component.

Reserved metadata keys are not selected as node keywords: `params`, `metadata`, `steps`, `name`, `finetune_params`, `train_params`, `refit_params`, `fit_on_all`, `force_layout`, `na_policy`, and `fill_value`.
