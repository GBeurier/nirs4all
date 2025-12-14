# Retrain and Transfer Learning

This guide covers retraining trained pipelines on new data, including full retrain, transfer learning, and fine-tuning modes.

## Overview

The retrain feature allows you to:

- **Full retrain**: Train from scratch with the same pipeline structure
- **Transfer**: Reuse preprocessing artifacts while training a new model
- **Finetune**: Continue training an existing model with additional data
- **Extract & Modify**: Get pipeline structure for inspection and modification

## Retrain Modes

| Mode | Preprocessing | Model | Use Case |
|------|---------------|-------|----------|
| `full` | Train new | Train new | New calibration set |
| `transfer` | Use existing | Train new | Apply preprocessing to new domain |
| `finetune` | Use existing | Continue training | Add more data to existing model |

## Basic Usage

### Full Retrain

Train everything from scratch using the same pipeline structure:

```python
from nirs4all.pipeline import PipelineRunner
from nirs4all.data import DatasetConfigs

runner = PipelineRunner(save_files=True, verbose=0)

# Train initial model
predictions, _ = runner.run(pipeline_config, dataset_config)
best_pred = predictions.top(n=1, rank_partition="test")[0]

# Full retrain on new data
new_data = DatasetConfigs(['path/to/new_calibration'])

retrained_preds, _ = runner.retrain(
    source=best_pred,
    dataset=new_data,
    mode='full',
    dataset_name='new_calibration',
    verbose=0
)

print(f"Retrained RMSE: {retrained_preds.top(n=1)[0]['rmse']:.4f}")
```

### Transfer Mode

Reuse preprocessing artifacts (scalers, SNV, etc.) while training a new model:

```python
# Transfer: reuse preprocessing, train new model
transfer_preds, _ = runner.retrain(
    source=best_pred,
    dataset=new_data,
    mode='transfer',
    dataset_name='transfer_test',
    verbose=0
)
```

This is useful when:
- Your preprocessing is well-optimized for the spectral domain
- You want to apply the same preprocessing to a different target variable
- You're doing machine/instrument transfer calibration

### Transfer with Different Model

Replace the model type during transfer:

```python
from sklearn.ensemble import GradientBoostingRegressor

new_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

transfer_preds, _ = runner.retrain(
    source=best_pred,
    dataset=new_data,
    mode='transfer',
    new_model=new_model,
    dataset_name='transfer_new_model',
    verbose=0
)
```

### Finetune Mode

Continue training an existing model (most effective with neural networks):

```python
# Finetune: continue training with additional epochs
finetune_preds, _ = runner.retrain(
    source=best_pred,
    dataset=new_data,
    mode='finetune',
    epochs=10,
    dataset_name='finetune_test',
    verbose=0
)
```

**Note**: Fine-tuning is most effective with neural network models. For sklearn models like PLSRegression, fine-tuning is equivalent to retraining since they don't support incremental learning.

## Retrain Sources

The `retrain()` method accepts various sources:

```python
# From prediction dict
runner.retrain(best_prediction, new_data, mode='full')

# From folder path
runner.retrain("runs/2024-12-14_wheat/pipeline_abc123/", new_data, mode='transfer')

# From bundle file
runner.retrain("exports/wheat_model.n4a", new_data, mode='transfer')

# From model ID
runner.retrain(model_id, new_data, mode='full')
```

## Extract and Modify

Get the pipeline structure for inspection or modification:

```python
# Extract pipeline
extracted = runner.extract(best_pred)

# Inspect
print(f"Number of steps: {len(extracted)}")
print(f"Model step index: {extracted.model_step_index}")
print(f"Preprocessing chain: {extracted.preprocessing_chain}")

# View steps
for i, step in enumerate(extracted.steps):
    print(f"Step {i}: {step}")
```

### Modify and Run

```python
from sklearn.ensemble import RandomForestRegressor

# Replace model
extracted.set_model(RandomForestRegressor(n_estimators=100))

# Run modified pipeline
modified_preds, _ = runner.run(
    pipeline=extracted.steps,
    dataset=new_data,
    pipeline_name='modified_pipeline'
)
```

## Fine-grained Step Control

For advanced use cases, you can control each step individually:

```python
from nirs4all.pipeline import StepMode

# Define step modes
step_modes = [
    StepMode(step_index=1, mode='predict'),  # Use existing scaler
    StepMode(step_index=2, mode='predict'),  # Use existing y_processing
    StepMode(step_index=3, mode='train'),    # Retrain preprocessing
    # Model step will follow overall mode
]

controlled_preds, _ = runner.retrain(
    source=best_pred,
    dataset=new_data,
    mode='full',
    step_modes=step_modes,
    dataset_name='controlled_retrain',
    verbose=0
)
```

### StepMode Options

| Mode | Description |
|------|-------------|
| `'train'` | Train this step from scratch |
| `'predict'` | Use existing artifact (no retraining) |
| `'skip'` | Skip this step entirely |

## Use Cases

### 1. Seasonal Recalibration

Update your model with new season's samples:

```python
# Load previous best model
previous_model = predictions_db.get_best_for_dataset('wheat_2023')

# Retrain with new season's data
new_season = DatasetConfigs(['data/wheat_2024'])

updated_preds, _ = runner.retrain(
    source=previous_model,
    dataset=new_season,
    mode='full',
    dataset_name='wheat_2024'
)
```

### 2. Machine Transfer

Apply preprocessing from reference instrument to new instrument:

```python
# Model trained on Machine A
machine_a_model = best_prediction

# New data from Machine B
machine_b_data = DatasetConfigs(['data/machine_b'])

# Transfer preprocessing, train new model for Machine B
transfer_preds, _ = runner.retrain(
    source=machine_a_model,
    dataset=machine_b_data,
    mode='transfer',
    dataset_name='machine_b_calibration'
)
```

### 3. Multi-target Prediction

Use same preprocessing for different target variables:

```python
# Original model for protein
protein_model = best_prediction

# Same spectra, different target (moisture)
moisture_data = DatasetConfigs({
    'X': 'spectra.csv',
    'Y': 'moisture_values.csv'
})

# Transfer preprocessing, train for moisture
moisture_preds, _ = runner.retrain(
    source=protein_model,
    dataset=moisture_data,
    mode='transfer',
    dataset_name='moisture_prediction'
)
```

### 4. A/B Testing Models

Compare different models with same preprocessing:

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

models = [
    PLSRegression(n_components=10),
    GradientBoostingRegressor(n_estimators=100),
    RandomForestRegressor(n_estimators=100),
]

results = {}
for model in models:
    preds, _ = runner.retrain(
        source=best_pred,
        dataset=test_data,
        mode='transfer',
        new_model=model,
        dataset_name=f'test_{model.__class__.__name__}'
    )
    best = preds.top(n=1, rank_partition="test")[0]
    results[model.__class__.__name__] = best['rmse']

print("Model Comparison:")
for name, rmse in sorted(results.items(), key=lambda x: x[1]):
    print(f"  {name}: RMSE = {rmse:.4f}")
```

## Neural Network Fine-tuning

For TensorFlow/PyTorch models, fine-tuning supports additional options:

```python
finetune_preds, _ = runner.retrain(
    source=best_nicon_model,
    dataset=new_data,
    mode='finetune',
    epochs=20,
    learning_rate=0.0001,  # Lower LR for fine-tuning
    freeze_layers=['conv1', 'conv2'],  # Freeze early layers
    dataset_name='finetune_nicon'
)
```

## Best Practices

1. **Validate preprocessing compatibility**: Ensure new data has same wavelength range
2. **Check feature dimensions**: New data must have same number of features
3. **Use transfer mode wisely**: Best when preprocessing is well-optimized
4. **Start with full retrain**: When in doubt, retrain everything
5. **Compare modes**: Test different modes to find what works best

## Troubleshooting

### Missing Artifacts

```python
# Error: Artifact not found
# Solution: Ensure original model was trained with save_files=True
runner = PipelineRunner(save_files=True, verbose=0)
```

### Feature Mismatch

```python
# Error: Feature dimension mismatch
# Solution: Verify new data has same number of features
print(f"Expected features: {best_pred['n_features']}")
print(f"New data features: {new_data.shape[1]}")
```

### Mode Not Suitable

```python
# Finetune with sklearn model = just retraining
# Use transfer or full mode instead for sklearn models
runner.retrain(source, data, mode='transfer')  # Better for sklearn
```

## See Also

- [Prediction and Model Reuse](prediction_model_reuse.md) - Basic prediction workflows
- [Export and Bundles](export_bundles.md) - Export models for deployment
- [Q33 Example](../examples/Q33_retrain_transfer.py) - Complete retrain examples
- [Migration Guide](migration_guide.md) - Upgrade from older versions
