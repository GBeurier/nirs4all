# Predictions

Predictions are the core output of nirs4all. Every time you run a pipeline, nirs4all evaluates each model on each cross-validation fold and partition, storing the results as **prediction records** in the workspace. These records contain scalar scores for ranking, full arrays for visualization, and links to the trained model chain for export and replay.

## The Prediction Lifecycle

```
Train               Store                    Query              Export             Predict
nirs4all.run() ---> store.duckdb +        -> result.top(n) ---> result.export() -> nirs4all.predict()
                    arrays/*.parquet         result.filter()    "model.n4a"        (new data)
                    (metadata, arrays,
                     chains, artifacts)
```

1. **Train** -- `nirs4all.run()` executes your pipeline, fits models on each fold, and evaluates on train/val/test partitions.
2. **Store** -- Prediction scores and metadata are persisted in DuckDB (`store.duckdb`), while dense arrays (y_true, y_pred) are stored in Parquet sidecar files (`arrays/`).
3. **Query** -- Use `result.top(n)`, `result.filter()`, `result.best_rmse`, etc. to find the best models.
4. **Export** -- Export the best model as a `.n4a` bundle for sharing or deployment.
5. **Predict** -- Apply the trained model to new data with `nirs4all.predict()`.

## Quick Start

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

# 1. Train
result = nirs4all.run(
    pipeline=[MinMaxScaler(), PLSRegression(10)],
    dataset="sample_data/regression",
)

# 2. Check results
print(f"Best RMSE: {result.best_rmse:.4f}")
print(f"Best R2: {result.best_r2:.4f}")

# 3. Export best model
result.export("best_model.n4a")

# 4. Predict on new data
preds = nirs4all.predict(model="best_model.n4a", data=X_new)
print(preds.values)
```

## Documentation Sections

```{toctree}
:maxdepth: 1

understanding_predictions
making_predictions
session_api
analyzing_results
exporting_models
advanced_predictions
```

::::{grid} 2
:gutter: 3

:::{grid-item-card} 📘 Understanding Predictions
:link: understanding_predictions
:link-type: doc

What predictions are, how they are stored, key concepts (chains, partitions, scores)

+++
{bdg-primary}`Fundamentals`
:::

:::{grid-item-card} 🎯 Making Predictions
:link: making_predictions
:link-type: doc

How to predict on new data from a RunResult, exported bundle, or chain ID

+++
{bdg-success}`Core Usage`
:::

:::{grid-item-card} 🔄 Session API
:link: session_api
:link-type: doc

Stateful workflows for production pipelines, model persistence, and iterative experimentation

+++
{bdg-info}`Stateful`
:::

:::{grid-item-card} 📊 Analyzing Results
:link: analyzing_results
:link-type: doc

Querying, filtering, ranking, and visualizing prediction results

+++
{bdg-primary}`Analysis`
:::

:::{grid-item-card} 📦 Exporting Models
:link: exporting_models
:link-type: doc

Exporting models as bundles, scripts, or configs for sharing and deployment

+++
{bdg-warning}`Deployment`
:::

:::{grid-item-card} 🚀 Advanced Predictions
:link: advanced_predictions
:link-type: doc

Transfer learning, retraining, SHAP explanations, batch prediction patterns

+++
{bdg-danger}`Advanced`
:::

::::

## See Also

- {doc}`/reference/predictions_api` -- API reference for PredictionResultsList and PredictionResult
- {doc}`/reference/pipeline_syntax` -- Pipeline syntax reference
