# Branching and Merging

Branching lets you run multiple sub-pipelines inside a single pipeline.
Merging recombines their outputs. Together, they enable preprocessing
comparison, multi-source handling, and model stacking -- all without writing
separate scripts.

---

## Why Branch?

A single preprocessing chain is rarely optimal. You may want to:

- Compare SNV, MSC, and Detrend to see which suits your data best.
- Apply different preprocessing to different feature sources (NIR vs.
  chemical markers).
- Build a stacking ensemble where several base models feed a meta-model.

Without branching, you would write and run separate pipelines for each
strategy. Branching lets you express all of these within one pipeline
definition.

---

## Two Families of Branches

nirs4all distinguishes between **duplication** branches and **separation**
branches. The difference is whether every branch sees the same samples or a
disjoint subset.

### Duplication branches

All samples go to every branch. Each branch applies its own preprocessing
and (optionally) its own model. This is the default when you pass a list of
sub-pipelines:

```python
{"branch": [
    [SNV(), PLSRegression(10)],
    [MSC(), RandomForestRegressor()],
]}
```

Both branches receive the full dataset. They run in parallel (conceptually)
and produce independent predictions.

You can also name branches for clearer reporting:

```python
{"branch": {
    "snv_pls": [SNV(), PLSRegression(10)],
    "msc_rf":  [MSC(), RandomForestRegressor()],
}}
```

### Separation branches

Samples are split into disjoint groups. Each group is processed by its own
sub-pipeline. This is useful when different subsets of your data require
different treatment.

There are several ways to define the split:

**By metadata column** -- one branch per unique value:

```python
{"branch": {"by_metadata": "site"}}
```

**By tag** -- split on a computed tag (e.g., outlier detection):

```python
{"branch": {"by_tag": "y_outlier_iqr", "values": {
    "clean": False,
    "outliers": True,
}}}
```

**By source** -- per-source preprocessing for multi-source datasets:

```python
{"branch": {"by_source": True, "steps": {
    "NIR":     [SNV(), PLSRegression(10)],
    "markers": [MinMaxScaler(), Ridge()],
}}}
```

**By filter** -- arbitrary sample selection:

```python
{"branch": {"by_filter": SampleFilter(...)}}
```

---

## Merging

Every branch step should eventually be followed by a merge step that
recombines the outputs. The merge strategy depends on the branch family.

### Merge strategies for duplication branches

| Strategy        | What it does                                           |
|-----------------|--------------------------------------------------------|
| `"predictions"` | Collect OOF predictions from each branch and use them as features for a downstream model. This is the basis of **stacking**. |
| `"features"`    | Collect transformed feature matrices from each branch and concatenate them. |
| `"all"`         | Merge all available features (transformed and original). |

### Merge strategy for separation branches

| Strategy   | What it does                                               |
|------------|------------------------------------------------------------|
| `"concat"` | Reassemble the disjoint sample subsets in their original order, producing a single prediction vector. |

```python
# After a separation branch
{"merge": "concat"}
```

:::{note}
If you omit the merge step after duplication branches, each branch is
evaluated independently and its predictions are recorded separately. A merge
step is only required when you need to feed branch outputs into a subsequent
model (stacking) or combine them into a single feature set.
:::

---

## The Stacking Pattern

Stacking is a powerful ensemble technique. The idea: train several base models,
then train a meta-model on their predictions. The key challenge is avoiding
data leakage -- the meta-model must not see base-model predictions that were
produced on the same samples the base model trained on.

nirs4all handles this automatically through OOF predictions. When you merge
with `"predictions"`, each branch's out-of-fold predictions become features
for the meta-model. Because OOF predictions are produced by models that never
saw the corresponding samples during training, the meta-model's training data
is leak-free.

```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"branch": [
        [SNV(), PLSRegression(10)],
        [MSC(), RandomForestRegressor()],
    ]},
    {"merge": "predictions"},
    {"model": Ridge()},           # meta-model
]
```

**What happens step by step:**

```
1. Data is split into 5 folds
2. Branch 0: SNV + PLS trained on each fold, OOF predictions collected
3. Branch 1: MSC + RF trained on each fold, OOF predictions collected
4. Merge: OOF predictions from both branches become a 2-column feature matrix
5. Meta-model (Ridge) is trained on the merged OOF predictions
6. Final predictions combine all three models
```

---

## Data Flow Diagram

The following diagram shows how data moves through a branching + merging
pipeline:

```
                     +-- Branch 0 --+
                     |  [SNV, PLS]  |
Input Dataset  -->  branch         merge  -->  Meta-model  -->  Result
                     |  [MSC, RF]  |
                     +-- Branch 1 --+

Duplication branches:
  - Both branches receive all samples
  - merge "predictions" collects OOF predictions

Separation branches:
  - Each branch receives a disjoint subset
  - merge "concat" reassembles the full sample set
```

---

## Combining Branch Types

You can nest or chain different branch types. For example, you might first
separate by source, apply per-source preprocessing, merge the sources, and
then create duplication branches for different models:

```python
pipeline = [
    # Per-source preprocessing
    {"branch": {"by_source": True, "steps": {
        "NIR":     [SNV()],
        "markers": [MinMaxScaler()],
    }}},
    {"merge": {"sources": "concat"}},

    # Model comparison
    ShuffleSplit(n_splits=5),
    {"branch": [
        [PLSRegression(10)],
        [RandomForestRegressor()],
    ]},
]
```

---

## When to Use Branching vs. Generators

Both branching and generators ({doc}`generators`) let you compare multiple
strategies. The difference:

| Feature    | Branching                              | Generators                      |
|------------|----------------------------------------|---------------------------------|
| Execution  | Sub-pipelines run within one variant   | Each variant is a separate run  |
| Results    | Branches share folds and state         | Variants are independent        |
| Combining  | Merge enables stacking / concatenation | Variants are ranked, best wins  |
| Best for   | Ensembles, multi-source, stacking      | Hyperparameter search, ablation |

Use branching when you want to **combine** results. Use generators when you
want to **compare** alternatives and pick the best one.

---

## Next Steps

- {doc}`pipelines` -- understand step types and execution order.
- {doc}`cross_validation` -- learn about OOF predictions that make stacking
  safe.
- {doc}`generators` -- express a search space of pipeline variants.
- {doc}`/user_guide/pipelines/branching` -- complete branching syntax and
  examples.
- {doc}`/user_guide/pipelines/stacking` -- detailed stacking patterns.
- {doc}`/user_guide/pipelines/merging` -- merge strategy reference.
