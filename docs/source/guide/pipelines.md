# 5. Write a pipeline and search its variants

A pipeline is an ordered recipe with explicit branching, data selection and learning scopes. Specify operators, model parameters, splits and target transformations in one place; use generators to expand alternatives rather than manually copying complete pipelines.

## SDK recipe

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from nirs4all.operators.transforms import SNV

pipeline = [
    SNV(),
    KFold(n_splits=3, shuffle=True, random_state=17),
    {"model": PLSRegression, "n_components": {"_range_": [2, 8, 2]}},
]
```

Check the operator parameter mapping before adding a generator. A generator inside a model step must address the parameter accepted by that model; the complete syntax and nested paths are in {doc}`/reference/pipeline_syntax` and {doc}`/reference/generator_keywords`. The configuration is inspectable before any model is fitted.

## Branch, select and merge

A duplication branch applies alternative recipes to the same observations. A separation branch routes subsets or sources according to explicit selectors. Feature concatenation combines transformed inputs. Prediction stacking consumes held-out base-model predictions; it must not substitute in-sample training predictions. Preserve sample identity at every join.

`y_processing` changes targets and requires the corresponding prediction inverse transform. Filters, exclusions and augmentation affect fitting scope and origin relations. Marking a row with `tag` differs from removing it with `exclude`. Model-local search differs from selecting complete pipeline/architecture variants.

## Choose a generator

| Need | Generator vocabulary |
|---|---|
| Choose alternatives | `_or_` and selection modifiers |
| Sweep integer/linear values | `_range_` |
| Sweep logarithmic values | `_log_range_` |
| Cartesian parameter combinations | `_grid_`, `_cartesian_` |
| Keep paired choices together | `_zip_` |
| Concatenate ordered specifications | `_chain_` |
| Sample a search space | `_sample_`, `_seed_` |
| Restrict incompatible combinations | `_mutex_`, `_requires_`, `_depends_on_`, `_exclude_` |
| Reuse registered configurations | `_preset_` |

The {doc}`generator reference </reference/generator_keywords>` documents exact types, defaults, selection semantics and constraints. The {doc}`pipeline keyword reference </reference/pipeline_keywords>` derives lifecycle fields and nested value schemas from the public registry.

## Search and resume

Global tuning evaluates a search space over a workflow. Model-local `finetune_params` searches a learner inside its permitted training scope. Resume requires the same deterministic contract and compatible persistent evidence; changing the space, objective or data identity is not ordinary continuation. Trial counts describe a target total where documented, not necessarily additional trials.

See {doc}`/user_guide/models/hyperparameter_tuning`, {doc}`/user_guide/models/native_pls_fold_hpo`, {doc}`/user_guide/models/structural_hpo` and {doc}`/user_guide/models/native_tuning_conformal`. Host learners and native catalog methods have different artifact contracts even when their task or parameters are similar.
