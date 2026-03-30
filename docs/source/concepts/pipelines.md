# Pipelines

A **pipeline** is the central abstraction in nirs4all. It describes, as an
ordered list of steps, everything that should happen to your data -- from
preprocessing through model training.

```python
result = nirs4all.run(
    pipeline=[MinMaxScaler(), PLSRegression(10)],
    dataset="sample_data/regression",
)
```

The pipeline above has two steps: a scaler and a model. nirs4all takes care
of loading data, running cross-validation, fitting the model on each fold,
collecting predictions, and ranking results.

---

## What Is a Pipeline?

A pipeline is a Python list. Each element is a **step**. Steps are processed
in order, from left to right. Early steps transform the data; later steps
evaluate or model it.

```
Step 1          Step 2          Step 3         Step 4            Step 5
Scale X  -->  Preprocess  -->  Scale y  -->  Define folds  -->  Train model
```

The list can be short (two steps) or long (ten or more steps with branches,
augmentation, and multiple models). The engine treats every list the same way:
parse each step, find a controller that knows how to execute it, and run it.

---

## Step Types

Every step in a pipeline falls into one of four categories.

### Transformers -- modify features (X)

Any object that follows the sklearn `TransformerMixin` interface
(`fit` / `transform`) can be used directly:

```python
from sklearn.preprocessing import MinMaxScaler
from nirs4all.operators.transforms import SNV

pipeline = [MinMaxScaler(), SNV()]
```

During training, the transformer is fitted on the training partition and then
applied to all partitions. During prediction, the saved fitted transformer is
loaded and applied without re-fitting.

### Y-processing -- modify targets (y)

Wrap a transformer in a `y_processing` dict to apply it to the target vector
rather than to the feature matrix:

```python
{"y_processing": MinMaxScaler()}
```

This is useful when training neural networks or models that are sensitive to
target scale. nirs4all automatically inverse-transforms predictions back to
the original scale before computing metrics.

### Splitters -- define cross-validation folds

A splitter step determines how the training partition is divided into folds:

```python
from sklearn.model_selection import ShuffleSplit

ShuffleSplit(n_splits=5, test_size=0.2)
```

The splitter writes fold indices into the dataset. Subsequent model steps
iterate over those folds. If no splitter is specified, nirs4all uses a
default strategy.

### Models -- train a predictor

A model step trains a predictor on each fold and collects predictions:

```python
from sklearn.cross_decomposition import PLSRegression

{"model": PLSRegression(n_components=10)}
```

The `model` keyword is optional for common sklearn estimators -- nirs4all
recognises them automatically. The keyword is required when you need to
attach additional options such as `train_params` or `refit_params`.

---

## How Steps Are Specified

There are two ways to add a step to a pipeline.

### Bare instances

Place an object directly in the list. nirs4all infers its role from its type:

```python
pipeline = [
    MinMaxScaler(),              # transformer
    ShuffleSplit(n_splits=5),    # splitter
    PLSRegression(10),           # model
]
```

### Dict-wrapped with a keyword

Wrap an object in a dictionary to be explicit about its role or to pass extra
configuration:

```python
pipeline = [
    {"y_processing": MinMaxScaler()},
    {"model": PLSRegression(10)},
]
```

The keyword tells nirs4all which controller should handle the step. Common
keywords include `model`, `y_processing`, `tag`, `exclude`, `branch`, `merge`,
`sample_augmentation`, and `feature_augmentation`. A complete list is
available in {doc}`/reference/pipeline_syntax`.

---

## Execution Order

When `nirs4all.run()` is called, steps execute in the order they appear.
A typical sequence looks like this:

```
 1. Preprocessing transforms (modify X)
 2. Y-processing (modify y)
 3. Splitter (write fold indices)
 4. For each fold:
      a. Fit transforms on train subset, apply to train + val (+ test)
      b. Fit model on train subset
      c. Predict on val and test subsets
      d. Store per-fold predictions and metrics
 5. Pool out-of-fold predictions and rank variants
 6. Refit the best variant on the full training set
```

Pre-splitter transforms are fitted once on the whole training partition.
Post-splitter transforms are fitted per fold. Placing a transform before or
after the splitter changes whether it sees all training data or only one
fold's training subset.

:::{note}
Steps before the splitter are executed once. Steps after the splitter are
executed K times (once per fold). The splitter itself runs once.
:::

---

## Controllers

Behind the scenes, every step is dispatched to a **controller** -- a class
that knows how to execute that particular kind of step. You rarely interact
with controllers directly, but understanding that they exist helps explain why
nirs4all can handle sklearn transformers, PyTorch models, branch/merge logic,
and augmentation operators using the same pipeline list.

When a step is encountered:

1. The step is parsed into a normalised form.
2. All registered controllers are checked. Each controller has a `matches()`
   method that says whether it can handle the step.
3. The matching controller with the highest priority (lowest number) is
   selected.
4. The controller executes the step, updating the dataset and the execution
   context.

For example, `MinMaxScaler()` is matched by the `TransformerMixinController`,
while `{"model": PLSRegression(10)}` is matched by the
`SklearnModelController`.

:::{tip}
If you build a custom operator that follows the sklearn `TransformerMixin` or
`RegressorMixin` interface, existing controllers will handle it automatically.
:::

---

## Putting It Together

Here is a realistic pipeline that uses all four step types:

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.cross_decomposition import PLSRegression
from nirs4all.operators.transforms import SNV

pipeline = [
    SNV(),                                   # transformer
    {"y_processing": MinMaxScaler()},        # y-processing
    ShuffleSplit(n_splits=5, test_size=0.2), # splitter
    {"model": PLSRegression(n_components=10)}, # model
]
```

The data flows through SNV normalisation, then the targets are scaled, then
five folds are defined, and finally PLS regression is trained and evaluated
on each fold.

---

## Next Steps

- {doc}`datasets` -- understand the data container that pipelines operate on.
- {doc}`cross_validation` -- learn how fold-based evaluation works.
- {doc}`branching_and_merging` -- run parallel sub-pipelines.
- {doc}`generators` -- express many pipeline variants in one definition.
- {doc}`/reference/pipeline_syntax` -- complete keyword and syntax reference.
