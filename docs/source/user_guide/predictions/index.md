# Predictions

Predictions are the core output of nirs4all. Every time you run a pipeline, nirs4all evaluates each model on each cross-validation fold and partition, storing the results as **prediction records** in the workspace. These records contain scalar scores for ranking, full arrays for visualization, and links to the trained model chain for export and replay.

## The Prediction Lifecycle

```
Train               Store                Query              Export             Predict
nirs4all.run() ---> store.duckdb ------> result.top(n) ---> result.export() -> nirs4all.predict()
                    (predictions,         result.filter()    "model.n4a"        (new data)
                     chains,
                     artifacts)
```

1. **Train** -- `nirs4all.run()` executes your pipeline, fits models on each fold, and evaluates on train/val/test partitions.
2. **Store** -- Every prediction record (scores, arrays, chain reference) is persisted in the workspace database (`store.duckdb`).
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

| Section | What You Will Learn |
|---------|-------------------|
| [Understanding Predictions](understanding_predictions.md) | What predictions are, how they are stored, key concepts (chains, partitions, scores) |
| [Making Predictions](making_predictions.md) | How to predict on new data from a RunResult, exported bundle, or chain ID |
| [Analyzing Results](analyzing_results.md) | Querying, filtering, ranking, and visualizing prediction results |
| [Exporting Models](exporting_models.md) | Exporting models as bundles, scripts, or configs for sharing and deployment |
| [Advanced Predictions](advanced_predictions.md) | Transfer learning, retraining, SHAP explanations, batch prediction patterns |

## See Also

- {doc}`/reference/predictions_api` -- API reference for PredictionResultsList and PredictionResult
- {doc}`/reference/pipeline_syntax` -- Pipeline syntax reference
