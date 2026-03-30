# Generators

Generators let you define a **search space** inside a single pipeline
definition. Instead of writing twenty pipelines by hand to compare different
preprocessing methods or sweep hyperparameter values, you describe the space
once and let nirs4all expand it into concrete variants.

---

## Why Generators?

Real NIRS projects involve many choices:

- Which preprocessing? (SNV, MSC, Detrend, derivatives, combinations)
- How many PLS components? (5, 10, 15, 20, ...)
- Which model family? (PLS, Random Forest, SVM, ...)

Without generators, you would write a separate pipeline for each combination.
With generators, you express the search space declaratively, and the library
creates every combination for you.

---

## How Generators Work

A generator keyword is a special dictionary key that nirs4all recognises
during pipeline normalisation. Before any training happens, the pipeline
definition is **expanded**: each generator keyword is replaced with concrete
values, producing a list of fully-specified pipeline variants.

```
Pipeline with generators       Expansion           Execution

[{_or_: [SNV, MSC]},    -->  [SNV, PLS(5)]    -->  train + score
 PLS({_range_: [5,10]})]     [SNV, PLS(10)]   -->  train + score
                              [MSC, PLS(5)]    -->  train + score
                              [MSC, PLS(10)]   -->  train + score
```

All variants are executed (optionally in parallel with `n_jobs`), and the
results are ranked by cross-validation score. `result.top(n)` shows the best
performers.

---

## The `_or_` Keyword

The most common generator. It creates one variant per alternative:

```python
pipeline = [
    {"_or_": [SNV(), MSC(), Detrend()]},
    PLSRegression(n_components=10),
]
# Expands to 3 variants: SNV+PLS, MSC+PLS, Detrend+PLS
```

`_or_` accepts classes, instances, or dicts. Each item in the list becomes a
step in one of the expanded variants.

---

## The `_range_` Keyword

Creates a linear sweep of numeric values. Useful for parameters like
`n_components`, window sizes, and polynomial orders:

```python
pipeline = [
    SNV(),
    PLSRegression(n_components={"_range_": [5, 25, 5]}),
]
# Expands to 5 variants: n_components = 5, 10, 15, 20, 25
```

The three arguments are `[start, stop, step]`. The range is inclusive on
both ends. If the step is omitted, it defaults to 1.

---

## Other Generator Keywords

nirs4all provides several additional generator keywords for different search
patterns. Each is covered briefly here; see
{doc}`/reference/generator_keywords` for full syntax and examples.

### `_log_range_` -- logarithmic sweep

Spaces values on a logarithmic scale. Ideal for regularisation parameters and
learning rates that span several orders of magnitude:

```python
{"alpha": {"_log_range_": [1e-4, 1e0, 5]}}
# Values: 0.0001, 0.001, 0.01, 0.1, 1.0
```

### `_grid_` -- Cartesian product of parameters

Generates every combination of the specified parameter values for a single
step. This is the classic grid search:

```python
{"_grid_": {"n_components": [5, 10, 15], "scale": [True, False]},
 "model": PLSRegression}
# 6 variants: 3 values x 2 values
```

### `_cartesian_` -- Cartesian product of pipeline stages

Like `_grid_`, but operates across pipeline stages rather than parameters of a
single step:

```python
{"_cartesian_": [
    {"_or_": [SNV(), MSC()]},
    {"_or_": [PLSRegression(10), RandomForestRegressor()]},
]}
# 4 variants: 2 preprocessors x 2 models
```

### `_zip_` -- parallel paired iteration

Pairs values element-by-element. The lists must have the same length:

```python
{"_zip_": {"n_components": [5, 10, 15], "scale": [True, False, True]}}
# 3 variants: (5, True), (10, False), (15, True)
```

### `_chain_` -- sequential ordered choices

Like `_or_`, but guarantees evaluation in the order listed. Useful when order
matters for reporting:

```python
{"_chain_": [config_a, config_b, config_c]}
# 3 variants, evaluated in that exact order
```

### `_sample_` -- random distribution sampling

Samples values from a statistical distribution. Useful for random search in
large parameter spaces:

```python
{"alpha": {"_sample_": {
    "distribution": "log_uniform",
    "from": 1e-4, "to": 1e-1, "num": 20,
}}}
# 20 randomly sampled values between 0.0001 and 0.1
```

---

## Combining Generators

Generators compose naturally. Place multiple generators in the same pipeline,
and nirs4all generates the Cartesian product of all of them:

```python
pipeline = [
    {"_or_": [SNV(), MSC(), Detrend()]},     # 3 alternatives
    PLSRegression(n_components={"_range_": [5, 15, 5]}),  # 3 values
]
# 3 x 3 = 9 total variants
```

You can also control the combination logic with modifier keywords like
`pick` (select a subset), `count` (limit total variants), and `arrange`
(control combination strategy). See {doc}`/reference/generator_keywords` for
the full modifier reference.

---

## Parallel Execution

When you have many variants, running them in parallel can significantly reduce
total time. Use the `n_jobs` parameter:

```python
result = nirs4all.run(
    pipeline=pipeline,
    dataset=dataset,
    n_jobs=-1,   # use all CPU cores
)
```

Each variant runs independently, using `joblib` for process-based parallelism.

---

## Viewing Results

After execution, the `RunResult` gives you access to ranked results:

```python
# Best overall
print(result.best_score)

# Top 5 variants
for entry in result.top(5):
    print(entry["pipeline_name"], entry["rmse"])
```

`result.top(n)` returns the N best-performing variants, ranked by the primary
metric (RMSECV for regression, balanced accuracy for classification).

---

## Generators vs. Branching

Generators and branching ({doc}`branching_and_merging`) both let you explore
multiple strategies, but they serve different purposes:

| Aspect         | Generators                            | Branching                         |
|----------------|---------------------------------------|-----------------------------------|
| Goal           | Compare alternatives, pick the best   | Combine results (stacking, merge) |
| Independence   | Variants run independently            | Branches share state and folds    |
| Output         | Ranked list of configurations         | Merged predictions or features    |

Use generators when you are **searching** for the best configuration. Use
branching when you want to **combine** multiple models or preprocessing
strategies into one prediction.

---

## Next Steps

- {doc}`pipelines` -- understand pipeline structure and step types.
- {doc}`cross_validation` -- learn how variants are scored and ranked.
- {doc}`branching_and_merging` -- combine branch outputs instead of ranking.
- {doc}`/reference/generator_keywords` -- complete keyword reference.
- {doc}`/user_guide/pipelines/generators` -- detailed usage guide with
  examples.
