# Tutorial: From Hello World to Stacking

This tutorial builds progressively from a minimal pipeline to advanced model stacking.
Each section introduces one new concept on top of the previous one. All examples use
synthetic data generated on the fly, so you can run them without any external files.

Estimated time: 20 minutes.

---

## 1. Hello World

The simplest possible nirs4all pipeline needs three things: data, a model, and
`nirs4all.run()`.

```python
import nirs4all
from sklearn.cross_decomposition import PLSRegression

dataset = nirs4all.generate(n_samples=200, random_state=42)
result = nirs4all.run(
    pipeline=[PLSRegression(n_components=5)],
    dataset=dataset,
    verbose=1,
)
print(f"RMSE: {result.best_rmse:.4f}")
```

`nirs4all.generate()` creates a synthetic NIRS dataset with realistic spectral
characteristics. The pipeline is a plain Python list containing a single model --
nirs4all auto-detects the step type. After training, the `result` object gives you
`best_rmse`, `best_r2`, and other score accessors.

---

## 2. Add Preprocessing

Raw spectra benefit from preprocessing before the model sees them. Add standard
scaling and Standard Normal Variate (SNV) scatter correction.

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from nirs4all.operators.transforms import SNV

dataset = nirs4all.generate(n_samples=200, random_state=42)

pipeline = [
    MinMaxScaler(),
    SNV(),
    PLSRegression(n_components=10),
]

result = nirs4all.run(pipeline=pipeline, dataset=dataset, verbose=1)
print(f"RMSE: {result.best_rmse:.4f}")
```

Steps execute in order: `MinMaxScaler` scales features to [0, 1], `SNV` removes
scatter effects, and then `PLSRegression` trains on the cleaned spectra. You can
chain as many preprocessing steps as needed.

---

## 3. Add Cross-Validation

Without a splitter, the model trains once on the training set. Adding a
cross-validation splitter gives you reliable performance estimates across
multiple train/validation splits.

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from nirs4all.operators.transforms import SNV

dataset = nirs4all.generate(n_samples=200, random_state=42)

pipeline = [
    MinMaxScaler(),
    SNV(),
    ShuffleSplit(n_splits=5, test_size=0.25, random_state=42),
    {"model": PLSRegression(n_components=10)},
]

result = nirs4all.run(pipeline=pipeline, dataset=dataset, verbose=1)
print(f"RMSE: {result.best_rmse:.4f}")
print(f"R2:   {result.best_r2:.4f}")
```

`ShuffleSplit` creates 5 random train/validation splits. Each fold trains a
separate model and predicts on held-out samples. The reported scores are
aggregated across all folds, giving a more trustworthy estimate than a single
split.

:::{note}
Once a splitter is present, wrapping the model with `{"model": ...}` makes the
pipeline structure explicit. Auto-detection also works, but the dict form is
recommended for clarity.
:::

---

## 4. Compare Preprocessings with `_or_`

Instead of writing three separate pipelines to compare SNV, MSC, and Detrend,
use the `_or_` generator keyword. It expands into one pipeline variant per
alternative.

```python
import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from nirs4all.operators.transforms import SNV, MSC, Detrend

dataset = nirs4all.generate(n_samples=200, random_state=42)

pipeline = [
    {"_or_": [SNV, MSC, Detrend]},
    ShuffleSplit(n_splits=5, test_size=0.25, random_state=42),
    {"model": PLSRegression(n_components=10)},
]

result = nirs4all.run(pipeline=pipeline, dataset=dataset, verbose=1)

print("Top 3 configurations:")
for pred in result.top(n=3, display_metrics=["rmse", "r2"]):
    print(f"  {pred.get('preprocessings', '')}: "
          f"RMSE={pred.get('rmse', 0):.4f}, "
          f"R2={pred.get('r2', 0):.4f}")
```

`_or_` takes a list of classes (not instances). nirs4all instantiates each one,
creating 3 pipeline variants that are all evaluated. `result.top(3)` returns the
best 3, ranked by validation score by default.

:::{tip}
Pass classes (e.g. `SNV`) to `_or_`, not instances (e.g. `SNV()`). nirs4all
handles instantiation internally so it can apply generator combinations
correctly.
:::

---

## 5. Tune Hyperparameters with `_range_`

The `_range_` generator creates a linear sweep of numeric values. Use it to
search over PLS component counts without writing a loop.

```python
import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from nirs4all.operators.transforms import SNV

dataset = nirs4all.generate(n_samples=200, random_state=42)

pipeline = [
    SNV(),
    ShuffleSplit(n_splits=5, test_size=0.25, random_state=42),
    {"n_components": {"_range_": [2, 20, 3]}, "model": PLSRegression},
]

result = nirs4all.run(pipeline=pipeline, dataset=dataset, verbose=1)

print("Top 3 component counts:")
for pred in result.top(n=3, display_metrics=["rmse"]):
    print(f"  n_components={pred.get('model_name', '')}: "
          f"RMSE={pred.get('rmse', 0):.4f}")
```

`{"_range_": [2, 20, 3]}` generates the values `[2, 5, 8, 11, 14, 17, 20]`
(inclusive end, step of 3). Each value is used as `n_components` for a separate
`PLSRegression` variant. Note that the model is passed as a class (`PLSRegression`)
rather than an instance, because nirs4all constructs each variant with the
swept parameter.

You can combine `_or_` on preprocessing with `_range_` on model parameters to
search a larger space. nirs4all evaluates the full Cartesian product of all
generator expansions.

---

## 6. Branch and Stack

The most powerful pattern in nirs4all is model stacking: run multiple
sub-pipelines in parallel, merge their out-of-fold predictions, and train a
meta-model on top.

```python
import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from nirs4all.operators.transforms import SNV, MSC

dataset = nirs4all.generate(n_samples=200, random_state=42)

pipeline = [
    ShuffleSplit(n_splits=5, test_size=0.25, random_state=42),
    {"branch": [
        [SNV(), PLSRegression(n_components=10)],
        [MSC(), RandomForestRegressor(n_estimators=50, random_state=42)],
    ]},
    {"merge": "predictions"},
    {"model": Ridge()},
]

result = nirs4all.run(pipeline=pipeline, dataset=dataset, verbose=1)
print(f"Stacked RMSE: {result.best_rmse:.4f}")
print(f"Stacked R2:   {result.best_r2:.4f}")
```

Here is what happens step by step:

1. **Branch** -- the data is duplicated and sent to two independent sub-pipelines.
   The first applies SNV then PLS; the second applies MSC then Random Forest.
2. **Merge** -- `"predictions"` collects out-of-fold predictions from each branch
   and stacks them as new features. During cross-validation, only held-out
   predictions are used, avoiding data leakage.
3. **Meta-model** -- `Ridge()` learns how to combine the two sets of predictions
   into a final output.

This is classic model stacking and often outperforms any single model.

---

## 7. Export and Predict

Once you have a trained pipeline you are happy with, export it to a `.n4a` bundle
and use it later on new data.

```python
import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from nirs4all.operators.transforms import SNV

# Train
dataset = nirs4all.generate(n_samples=200, random_state=42)
pipeline = [
    MinMaxScaler(),
    SNV(),
    ShuffleSplit(n_splits=5, test_size=0.25, random_state=42),
    {"model": PLSRegression(n_components=10)},
]
result = nirs4all.run(pipeline=pipeline, dataset=dataset, verbose=1)

# Export the best model
result.export("my_model.n4a")

# Later: load and predict on new samples
new_data = nirs4all.generate(n_samples=50, random_state=99)
predictions = nirs4all.predict(model="my_model.n4a", data=new_data)
print(f"Predicted {len(predictions)} samples")
print(f"First 5 predictions: {predictions.y_pred[:5]}")
```

The `.n4a` bundle contains the full preprocessing chain and the trained model.
`nirs4all.predict()` replays every step in the correct order, so new data
receives the exact same transformations that were applied during training.

---

## 8. Next Steps

You now know the core workflow: generate or load data, build a pipeline, run it,
inspect results, and export. Here are some directions to explore next.

- {doc}`concepts` -- understand SpectroDataset, pipeline execution flow, and how
  results are structured
- {doc}`/user_guide/preprocessing/overview` -- all available NIRS transforms
  (SNV, MSC, derivatives, smoothing, wavelength selection)
- {doc}`/user_guide/pipelines/generators` -- the full generator system (`_or_`,
  `_range_`, `_grid_`, `_cartesian_`, `_sample_`, and more)
- {doc}`/examples/index` -- runnable examples organized by topic
- {doc}`/reference/pipeline_syntax` -- every pipeline keyword in one place
