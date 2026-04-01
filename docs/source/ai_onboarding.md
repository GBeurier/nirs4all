# AI Coding Assistant Onboarding

**Short reference for writing nirs4all pipelines and outputting results.**

This guide is intentionally narrow. It covers:
- how to write pipelines,
- how to expand them with generators,
- how to run them,
- how to inspect/export results,
- how to chart them with `PredictionAnalyzer`.

It does **not** cover SHAP, synthetic data generation, retraining, session internals, or architecture.

**Version**: 0.8.6 | **Python**: 3.11+ | **License**: CeCILL-2.1

---

## 1. Fast Path

Use this as the default pattern:

```python
import nirs4all
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from nirs4all.operators.transforms import FirstDerivative
from nirs4all.visualization.predictions import PredictionAnalyzer
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

pipeline = [
    SNV(),
    FirstDerivative(),
    ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
    PLSRegression(n_components=10),
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="snv_d1_pls",
    random_state=42,
    verbose=1,
)

print(result.best["model_name"])
print(result.best_rmse)
print(result.best_r2)

analyzer = PredictionAnalyzer(result.predictions)
analyzer.plot_top_k(k=5, rank_metric="rmse")
analyzer.plot_heatmap(
    x_var="model_name",
    y_var="preprocessings",
    rank_metric="rmse",
    display_metric="rmse",
)

result.export("exports/best_model.n4a")
```

Default workflow: `write pipeline -> run pipeline -> inspect RunResult -> chart with PredictionAnalyzer -> export bundle or tables`

---

## 2. `nirs4all.run()` Essentials

```python
result = nirs4all.run(
    pipeline,
    dataset,
    name="",
    verbose=1,
    random_state=None,
    refit=True,
    save_artifacts=True,
    workspace_path="workspace",
    max_generation_count=10000,
)
```

### Important arguments

| Argument | Meaning |
|---------|---------|
| `pipeline` | Pipeline definition, usually a Python list |
| `dataset` | Folder path, `(X, y)`, dict, or dataset object |
| `name` | Run name for logs/workspace |
| `verbose` | `0` quiet, `1` normal, `2` debug |
| `random_state` | Seed |
| `refit` | Refit best models on full train set after CV |
| `save_artifacts` | Persist artifacts in the workspace |
| `workspace_path` | Workspace root |
| `max_generation_count` | Guardrail for large generator expansions |

### Accepted `dataset` forms

```python
"sample_data/regression"
(X, y)
{"X": X, "y": y, "metadata": meta}
dataset_object
["data/wheat", "data/corn"]
```

If both `pipeline` and `dataset` are lists, `run()` evaluates the cartesian product.

---

## 3. Writing Pipelines

### 3.1 The default shape

Pipelines are usually plain Python lists:

```python
pipeline = [
    preprocessing_step,
    splitter_step,
    model_step,
]
```

Recommended order:

1. preprocessing,
2. split strategy,
3. model,
4. optional merge and meta-model.

### 3.2 Hard rules

- A trainable pipeline must contain at least one model step.
- Put the splitter before the first model you want scored by CV.
- Steps after a `branch` block run once per branch until you merge.
- `result.best` prefers a refit/final entry when `refit=True`.
- For visualization, use `PredictionAnalyzer` after the run.

### 3.3 Preferred style

- Use Python lists.
- Use instantiated objects for fixed steps.
- Use dict syntax only when you need generators, `branch`, `merge`, or an explicitly named `model`.
- Give important models names when you compare many variants.

Good default:

```python
pipeline = [
    SNV(),
    ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
    {"model": PLSRegression(n_components=10), "name": "PLS_10"},
]
```

### 3.4 Valid step formats

| Format | Example | Best use |
|-------|---------|----------|
| Instance | `SNV()` | Default |
| Class | `SNV` | Default constructor |
| String path | `"sklearn.preprocessing.StandardScaler"` | Config-driven pipelines |
| Dict wrapper | `{"model": PLSRegression(10)}` | Named or generated models |
| Dict block | `{"branch": {...}}` | Branching |
| Dict block | `{"merge": "features"}` | Merge branch outputs |

Examples:

```python
pipeline = [SNV(), PLSRegression(n_components=10)]
pipeline = [SNV, PLSRegression]
pipeline = ["sklearn.preprocessing.StandardScaler", "sklearn.cross_decomposition.PLSRegression"]
pipeline = [SNV(), {"model": PLSRegression(n_components=10), "name": "PLS_10"}]
```

---

## 4. Generators

Generators let one compact pipeline spec expand into many concrete pipelines.

### 4.1 Quick reference

| Generator | Use it for | Example |
|----------|------------|---------|
| `_or_` | alternatives | SNV vs MSC vs Detrend |
| `_range_` | linear numeric sweep | `n_components=5,10,15,20` |
| `_log_range_` | logarithmic sweep | `alpha` or regularization |
| `_grid_` | full cartesian parameter grid | small exhaustive search |
| `_zip_` | paired parameters | aligned `(alpha, l1_ratio)` pairs |
| `_cartesian_` | stage-by-stage pipeline combinations | preprocessing stacks |

### 4.2 `_or_`

```python
pipeline = [
    {"_or_": [SNV(), MSC(), Detrend()]},
    ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
    PLSRegression(n_components=10),
]
```

This expands to 3 pipelines.

### 4.3 `_range_`

```python
pipeline = [
    SNV(),
    ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
    {"n_components": {"_range_": [5, 20, 5]}, "model": PLSRegression},
]
```

This expands to `PLSRegression(n_components=5,10,15,20)`.

Alternative explicit form:

```python
{"n_components": {"_range_": {"from": 5, "to": 20, "step": 5}}, "model": PLSRegression}
```

### 4.4 `_log_range_`

```python
pipeline = [
    SNV(),
    {"alpha": {"_log_range_": [1e-4, 10, 8]}, "model": Ridge},
]
```

### 4.5 `_grid_`

```python
pipeline = [
    SNV(),
    {"_grid_": {
        "n_components": [5, 10, 15],
        "scale": [True, False],
    }, "model": PLSRegression},
]
```

This creates 6 model variants.

### 4.6 `_zip_`

```python
pipeline = [
    {"_zip_": {
        "alpha": [0.1, 1.0, 10.0],
        "l1_ratio": [0.2, 0.5, 0.8],
    }, "model": ElasticNet},
]
```

This creates 3 paired variants, not 9 all-combinations.

### 4.7 `_cartesian_`

```python
pipeline = [
    {"_cartesian_": [
        {"_or_": [SNV(), MSC()]},
        {"_or_": [FirstDerivative(), None]},
        {"_or_": [Detrend(), None]},
    ]},
    ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
    PLSRegression(n_components=10),
]
```

This creates all combinations of those stages.

### 4.8 Modifiers

| Modifier | Meaning | Example |
|---------|---------|---------|
| `count` | limit how many variants are kept | `{"_or_": [...], "count": 5}` |
| `pick` | choose unordered combinations | `{"_or_": [A, B, C], "pick": 2}` |
| `arrange` | choose ordered arrangements | `{"_or_": [A, B, C], "arrange": 2}` |

### 4.9 Rules of thumb

- Use `_or_` for discrete choices.
- Use `_range_` for small linear sweeps.
- Use `_log_range_` for regularization-like parameters.
- Use `_zip_` when parameter pairs should stay aligned.
- Use `_grid_` only when the full cartesian product is genuinely wanted.
- Use `max_generation_count` to prevent accidental blowups.

---

## 5. Branching And Merging

Use branching when several sub-pipelines should share upstream work but differ in preprocessing or models.

### 5.1 Compare preprocessing branches

```python
pipeline = [
    ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
    {"branch": {
        "snv": [SNV()],
        "msc": [MSC()],
        "d1": [FirstDerivative()],
    }},
    PLSRegression(n_components=10),
]
```

Shared split, separate branch contexts, then one PLS fit per branch.

### 5.2 Merge features

```python
pipeline = [
    ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
    {"branch": {
        "snv": [SNV()],
        "msc": [MSC()],
        "d1": [FirstDerivative()],
    }},
    {"merge": "features"},
    PLSRegression(n_components=10),
]
```

### 5.3 Merge predictions for stacking

```python
from nirs4all.operators.models import MetaModel

pipeline = [
    ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
    {"branch": {
        "pls": [SNV(), PLSRegression(n_components=8)],
        "ridge": [MSC(), Ridge(alpha=1.0)],
        "d1_pls": [FirstDerivative(), PLSRegression(n_components=12)],
    }},
    {"merge": "predictions"},
    {"model": MetaModel(model=Ridge(alpha=0.1)), "name": "stacker"},
]
```

Use this when the next model should learn from branch predictions rather than branch features.

---

## 6. Outputs From `RunResult`

`nirs4all.run()` returns a `RunResult`.

```python
result = nirs4all.run(pipeline, dataset)
```

### 6.1 Most useful properties

| Property | Meaning |
|---------|---------|
| `result.best` | Best prediction entry, preferring refit/final if available |
| `result.cv_best` | Best CV entry only |
| `result.final` | Final refit entry, or `None` |
| `result.best_score` | Best primary score |
| `result.best_rmse` | Best RMSE for regression |
| `result.best_r2` | Best R2 for regression |
| `result.best_accuracy` | Best accuracy for classification |
| `result.cv_best_score` | Best validation score before refit |
| `result.final_score` | Final refit test score |
| `result.num_predictions` | Number of prediction rows |
| `result.artifacts_path` | Workspace path |
| `result.predictions` | Full prediction store |

Quick inspection:

```python
print(result.summary())
print(result.best["model_name"])
print(result.best["metric"])
print(result.best_score)
print(result.cv_best_score)
print(result.final_score)
```

### 6.2 Rank and filter results

```python
top5 = result.top(5, rank_metric="rmse")
for row in top5:
    print(row["model_name"], row["test_score"], row.get("preprocessings"))

top_per_dataset = result.top(3, rank_metric="rmse", group_by="dataset_name")
top_per_combo = result.top(2, rank_metric="rmse", group_by=["dataset_name", "model_name"])

pls_test_rows = result.filter(model_name="PLSRegression", partition="test")
branch_rows = result.filter(branch_name="snv")
```

### 6.3 Export tables

Export prediction metadata:

```python
df = result.predictions.to_dataframe()   # Polars DataFrame
df.write_csv("exports/all_predictions.csv")
```

Export the arrays from one best result:

```python
from nirs4all.data.predictions import Predictions

best = result.best
Predictions.save_predictions_to_csv(
    y_true=best.get("y_true"),
    y_pred=best.get("y_pred"),
    filepath="exports/best_prediction_arrays.csv",
)
```

### 6.4 Export a deployable bundle

```python
bundle_path = result.export("exports/best_model.n4a")
print(bundle_path)

top3 = result.top(3, rank_metric="rmse")
result.export("exports/runner_up.n4a", source=top3[1])
```

### 6.5 Predict later from the bundle

```python
predict_result = nirs4all.predict(
    model="exports/best_model.n4a",
    data=X_new,
)

print(predict_result.shape)
print(predict_result.to_list()[:5])

predict_df = predict_result.to_dataframe()
predict_df.to_csv("exports/new_predictions.csv", index=False)
```

---

## 7. Charts With `PredictionAnalyzer`

Use `PredictionAnalyzer` on `result.predictions`.

```python
from nirs4all.visualization.predictions import PredictionAnalyzer

analyzer = PredictionAnalyzer(result.predictions)
```

To save figures automatically:

```python
analyzer = PredictionAnalyzer(
    result.predictions,
    save=True,
    output_dir="exports/figures",
)
```

Saved files are PNGs named like:

```text
<dataset>_<chart_type>_<counter>.png
```

### 7.1 Main chart methods

| Method | Use it for | Typical args |
|-------|-------------|--------------|
| `plot_top_k()` | compare the best few regression models | `k`, `rank_metric` |
| `plot_heatmap()` | compare two categorical dimensions | `x_var`, `y_var`, `rank_metric`, `display_metric` |
| `plot_candlestick()` | score spread by category | `variable`, `display_metric` |
| `plot_histogram()` | overall score distribution | `display_metric` |
| `plot_confusion_matrix()` | top classification models | `k`, `rank_metric` |
| `branch_summary()` | table summary for branches | `metrics` |
| `plot_branch_comparison()` | mean branch performance | `display_metric` |
| `plot_branch_boxplot()` | branch spread | `display_metric` |
| `plot_branch_heatmap()` | branch by fold or model | `y_var`, `display_metric` |

### 7.2 Core examples

Top-k regression comparison:

```python
analyzer.plot_top_k(k=5, rank_metric="rmse")
analyzer.plot_top_k(k=5, rank_metric="r2", rank_partition="val", display_partition="all")
```

Heatmaps:

```python
analyzer.plot_heatmap(
    x_var="model_name",
    y_var="preprocessings",
    rank_metric="rmse",
    display_metric="rmse",
)

analyzer.plot_heatmap(
    x_var="model_name",
    y_var="dataset_name",
    rank_metric="r2",
    display_metric="r2",
    display_partition="test",
)
```

Candlestick and histogram:

```python
analyzer.plot_candlestick(variable="model_name", display_metric="rmse")
analyzer.plot_candlestick(variable="preprocessings", display_metric="r2")
analyzer.plot_histogram(display_metric="rmse")
```

Classification confusion matrices:

```python
analyzer.plot_confusion_matrix(
    k=4,
    rank_metric="balanced_accuracy",
    display_partition="test",
)

analyzer.plot_confusion_matrix(
    k=4,
    rank_metric="balanced_accuracy",
    display_metric=["balanced_accuracy", "accuracy"],
)
```

Branch outputs:

```python
summary = analyzer.branch_summary(metrics=["rmse", "r2"])
print(summary.to_markdown())

analyzer.plot_branch_comparison(display_metric="rmse")
analyzer.plot_branch_boxplot(display_metric="rmse")
analyzer.plot_branch_heatmap(y_var="fold_id", display_metric="rmse")
```

### 7.3 Ranking vs display

The charting rule to remember:
- rank on validation,
- display on test.

Example:

```python
analyzer.plot_heatmap(
    x_var="model_name",
    y_var="preprocessings",
    rank_metric="rmse",
    rank_partition="val",
    display_metric="r2",
    display_partition="test",
)
```

### 7.4 Aggregation for repeated samples

If several rows belong to the same physical sample, aggregate charts by the repetition column:

```python
analyzer = PredictionAnalyzer(result.predictions, default_aggregate="ID")
analyzer.plot_top_k(k=5)
analyzer.plot_heatmap(x_var="model_name", y_var="preprocessings")
analyzer.plot_top_k(k=5, aggregate="ID")
analyzer.plot_candlestick(variable="model_name", aggregate="ID")
```

If `save=False`, chart methods return figure objects, so you can still call `fig.savefig(...)` manually.

---

## 8. Rules That Avoid Most Mistakes

- Start with a Python list pipeline unless you specifically need config files.
- Put the CV splitter before the first model you want scored.
- Use `_or_` for alternatives and `_range_` for simple numeric sweeps.
- Use `_zip_` when parameters are paired; use `_grid_` only when you truly want the full cartesian product.
- Use `result.top()` and `PredictionAnalyzer` immediately after a run instead of guessing which pipeline won.
- For regression charts, rank by `rmse` or `r2`. For classification charts, rank by `accuracy` or `balanced_accuracy`.
- Remember that `result.best` may be a final/refit entry; use `result.cv_best` when you want the pre-refit winner.
- Export metadata with `result.predictions.to_dataframe()`. Export deployable models with `result.export("file.n4a")`.
- If samples are repeated, aggregate charts by the repetition column such as `ID`.

This is the narrow reference to follow when the task is: write a pipeline, run it, compare the outputs, and export the result.
