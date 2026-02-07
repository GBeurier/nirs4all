# Prediction and Model Reuse

```{note}
This page has been superseded by the consolidated **Predictions** documentation. Please refer to the new guides:

- **[Making Predictions](/user_guide/predictions/making_predictions.md)** -- How to predict on new data from a RunResult, exported bundle, or chain ID
- **[Exporting Models](/user_guide/predictions/exporting_models.md)** -- Export formats (.n4a, .n4a.py, .json), bundle anatomy, sharing models
- **[Advanced Predictions](/user_guide/predictions/advanced_predictions.md)** -- Transfer learning, retraining, SHAP explanations
- **[Predictions Overview](/user_guide/predictions/index.md)** -- Full navigation to all prediction documentation
```

## Quick Reference

The primary prediction workflows are:

### Predict from an Exported Bundle

```python
import nirs4all

# Train and export
result = nirs4all.run(pipeline, dataset)
result.export("model.n4a")

# Predict on new data
preds = nirs4all.predict(model="model.n4a", data=X_new)
print(preds.values)
```

### Predict from a Chain ID

```python
preds = nirs4all.predict(chain_id="abc123", data=X_new, workspace_path="workspace")
```

### Transfer Learning

```python
retrained = nirs4all.retrain(
    source="model.n4a",
    data=new_data,
    mode="transfer",
)
```

For the complete guide, see {doc}`/user_guide/predictions/index`.

```{seealso}
**Related Examples:**
- [U01: Save, Load, Predict](../../../examples/user/06_deployment/U01_save_load_predict.py) - Model persistence and prediction
- [U04: sklearn Integration](../../../examples/user/06_deployment/U04_sklearn_integration.py) - NIRSPipeline as sklearn estimator
```
