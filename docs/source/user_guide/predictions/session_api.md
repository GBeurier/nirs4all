# Session API - Stateful Workflows

The Session API provides a stateful interface for multi-step machine learning workflows. Unlike the functional `nirs4all.run()` API, sessions maintain state across multiple operations, making them ideal for production pipelines, iterative experimentation, and model deployment workflows.

## When to Use Sessions

Choose the Session API when you need:

- **Multi-step workflows**: Train → predict → retrain → save in one coherent object
- **Interactive experimentation**: Keep a trained model in memory for quick predictions
- **Production pipelines**: Consistent configuration across training and prediction phases
- **Model persistence**: Save/load trained models with their complete state
- **Resource efficiency**: Share a PipelineRunner across multiple operations

For one-off experiments or quick analysis, use the simpler `nirs4all.run()` functional API.

## Basic Usage

### Creating a Session

A session is created with a pipeline definition and optional configuration:

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

# Define your pipeline
pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"model": PLSRegression(n_components=10)},
]

# Create a session
session = nirs4all.Session(
    pipeline=pipeline,
    name="WheatProteinModel",
    verbose=1,
    save_artifacts=True
)

print(f"Session: {session.name}")
print(f"Status: {session.status}")  # 'initialized'
print(f"Pipeline steps: {len(session.pipeline)}")
```

### Training with a Session

Once created, train the session on a dataset:

```python
# Train the pipeline
result = session.run(dataset="sample_data/regression")

print(f"Best RMSE: {result.best_rmse:.4f}")
print(f"Best R²: {result.best_r2:.4f}")
print(f"Session status: {session.status}")  # 'trained'
print(f"Is trained: {session.is_trained}")  # True
```

The `run()` method returns a standard `RunResult` object with all predictions and metrics.

### Making Predictions

After training, predict on new data:

```python
import numpy as np

# Generate new data (in practice, load from file)
X_new = np.random.randn(20, 100)

# Predict using the trained session
predictions = session.predict(X_new)

print(f"Predictions shape: {predictions.shape}")
print(f"Model name: {predictions.model_name}")
print(f"First 5 predictions: {predictions.values[:5]}")
```

The session automatically:
- Selects the best model from training
- Applies all preprocessing steps in the correct order
- Performs ensemble averaging across CV folds
- Returns predictions in a convenient `PredictResult` object

### Saving and Loading Sessions

Save the trained session to a `.n4a` bundle:

```python
from pathlib import Path

# Save session
save_path = Path("exports/wheat_model.n4a")
save_path.parent.mkdir(parents=True, exist_ok=True)
session.save(save_path)

print(f"Session saved: {save_path}")
print(f"File size: {save_path.stat().st_size / 1024:.1f} KB")
```

Load a saved session for later use:

```python
import nirs4all

# Load session from bundle
loaded_session = nirs4all.load_session("exports/wheat_model.n4a")

print(f"Loaded: {loaded_session.name}")
print(f"Status: {loaded_session.status}")  # 'trained'
print(f"Ready for prediction: {loaded_session.is_trained}")  # True

# Predict immediately
predictions = loaded_session.predict(X_new)
```

## Session Features

### Stateful Pipeline Management

Sessions maintain the complete state of your pipeline:

```python
# Check session state at any time
print(f"Status: {session.status}")  # 'initialized', 'trained', or 'error'
print(f"Is trained: {session.is_trained}")  # Boolean
print(f"Pipeline: {session.pipeline}")  # List of steps
print(f"Workspace: {session.workspace_path}")  # Path to workspace
```

After training, the session holds:
- Fitted preprocessing artifacts (scalers, transformers)
- Trained model weights for each CV fold
- Training metrics and prediction results
- Workspace references for artifact loading

### Automatic Workspace Context

Sessions automatically manage workspace resources:

```python
# Specify workspace location
session = nirs4all.Session(
    pipeline=pipeline,
    workspace_path="my_workspace/",
    name="Production_Model_v1"
)

# All artifacts saved to workspace
result = session.run(dataset)

# Session knows where to find artifacts
predictions = session.predict(X_new)
```

If no workspace is specified, a default `workspace/` directory is created.

### Training History

Sessions track their run history:

```python
# Initial training
result1 = session.run("sample_data/regression")

# Retrain on new data
result2 = session.retrain("sample_data/new_batch", mode="transfer")

# Check history
print(f"Total runs: {len(session.history)}")
for i, entry in enumerate(session.history):
    print(f"Run {i+1}: {entry['dataset']} -> RMSE {entry['best_score']:.4f}")
```

Each history entry records:
- Dataset used
- Best score achieved
- Number of predictions generated
- Retrain mode (if applicable)

### Context Manager Pattern

Use sessions as context managers for automatic resource cleanup:

```python
# Session auto-closes when block exits
with nirs4all.session(pipeline=pipeline, name="ContextDemo", verbose=1) as sess:
    result = sess.run("sample_data/regression")
    predictions = sess.predict(X_new)
    print(f"Best score: {result.best_score:.4f}")
# Session resources released here

# Also works for resource sharing (no pipeline)
with nirs4all.session(verbose=0, save_artifacts=False) as shared:
    # Multiple runs share the same runner
    r1 = nirs4all.run(pipeline1, dataset1, session=shared)
    r2 = nirs4all.run(pipeline2, dataset2, session=shared)
```

## Complete Example

Here's a full workflow showing session creation through deployment:

```python
import nirs4all
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from nirs4all.operators.transforms import StandardNormalVariate as SNV

# 1. Define pipeline
pipeline = [
    MinMaxScaler(),                              # X scaling
    ShuffleSplit(n_splits=5, test_size=0.2),    # 5-fold CV
    SNV(),                                        # SNV preprocessing
    {"y_processing": StandardScaler()},          # Target scaling
    {"model": PLSRegression(n_components=15)},   # PLS model
]

# 2. Create and train session
session = nirs4all.Session(
    pipeline=pipeline,
    name="WheatProtein_Production_v2",
    verbose=1,
    workspace_path="production_workspace/",
    random_state=42
)

print("Training session...")
result = session.run(dataset="data/wheat_samples/")

# 3. Validate results
print(f"\nTraining complete:")
print(f"  Predictions: {result.num_predictions}")
print(f"  Best RMSE: {result.best_rmse:.4f}")
print(f"  Best R²: {result.best_r2:.4f}")
print(f"  Status: {session.status}")

# 4. Test predictions
X_validation = np.load("data/validation_set.npy")
validation_preds = session.predict(X_validation)

print(f"\nValidation predictions:")
print(f"  Shape: {validation_preds.shape}")
print(f"  Mean: {validation_preds.values.mean():.2f}")
print(f"  Std: {validation_preds.values.std():.2f}")

# 5. Save for deployment
save_path = "exports/wheat_protein_v2.n4a"
session.save(save_path)
print(f"\nModel exported: {save_path}")

# 6. Simulate deployment: load and predict
deployed_session = nirs4all.load_session(save_path)
X_new_sample = np.random.randn(1, 100)  # New sample from sensor
prediction = deployed_session.predict(X_new_sample)

print(f"\nDeployment test:")
print(f"  Input shape: {X_new_sample.shape}")
print(f"  Prediction: {prediction.values[0]:.2f}")
```

## Advanced Usage

### Retraining

Update a trained session with new data:

```python
# Initial training
session = nirs4all.Session(pipeline=pipeline, name="Adaptive")
result = session.run("data/initial_batch/")

# New data arrives
new_result = session.retrain(
    dataset="data/new_batch/",
    mode="transfer"  # or "full", "finetune"
)

print(f"Initial RMSE: {result.best_rmse:.4f}")
print(f"Retrained RMSE: {new_result.best_rmse:.4f}")
```

Retrain modes:
- `"full"`: Complete retraining from scratch
- `"transfer"`: Transfer learning (keep preprocessing, retrain model)
- `"finetune"`: Fine-tuning (update model weights incrementally)

### Multiple Sessions

Manage multiple models in parallel:

```python
# Create sessions for different targets
protein_session = nirs4all.Session(pipeline=protein_pipeline, name="Protein")
moisture_session = nirs4all.Session(pipeline=moisture_pipeline, name="Moisture")

# Train in parallel
protein_result = protein_session.run("data/wheat/")
moisture_result = moisture_session.run("data/wheat/")

# Predict with both
protein_pred = protein_session.predict(X_new)
moisture_pred = moisture_session.predict(X_new)

print(f"Protein: {protein_pred.values[0]:.2f}%")
print(f"Moisture: {moisture_pred.values[0]:.2f}%")
```

### Resource Sharing Mode

Use a session without a pipeline to share resources across multiple `nirs4all.run()` calls:

```python
# Context manager for shared runner
with nirs4all.session(verbose=1, workspace_path="shared_workspace/") as shared:
    # Try different pipelines with shared configuration
    r1 = nirs4all.run(pipeline=[MinMaxScaler(), PLSRegression(5)],
                      dataset="data/", session=shared)
    r2 = nirs4all.run(pipeline=[MinMaxScaler(), PLSRegression(10)],
                      dataset="data/", session=shared)
    r3 = nirs4all.run(pipeline=[MinMaxScaler(), PLSRegression(15)],
                      dataset="data/", session=shared)

    # Compare results
    print(f"PLS(5):  RMSE = {r1.best_rmse:.4f}")
    print(f"PLS(10): RMSE = {r2.best_rmse:.4f}")
    print(f"PLS(15): RMSE = {r3.best_rmse:.4f}")
```

This shares the PipelineRunner instance, reducing overhead for multiple runs.

## API Reference

### Creating Sessions

**`nirs4all.Session(pipeline, name, **kwargs)`**

Create a stateful session for pipeline management.

**Parameters:**
- `pipeline` (list): Pipeline definition (list of steps)
- `name` (str): Session name for identification
- `verbose` (int): Verbosity level (0-3). Default: 1
- `save_artifacts` (bool): Save artifacts to workspace. Default: True
- `workspace_path` (str|Path): Workspace directory path
- `random_state` (int): Random seed for reproducibility
- `plots_visible` (bool): Show plots during training

**Returns:** Session object

---

**`nirs4all.session(pipeline, name, **kwargs)`**

Context manager for creating a session (same parameters as above).

**Returns:** Context manager yielding Session

---

**`nirs4all.load_session(path)`**

Load a session from a saved `.n4a` bundle.

**Parameters:**
- `path` (str|Path): Path to `.n4a` bundle file

**Returns:** Session object ready for prediction

### Session Methods

**`session.run(dataset, **kwargs)`**

Train the session's pipeline on a dataset.

**Parameters:**
- `dataset` (str|Path|tuple|dict): Dataset to train on
  - Path to data folder: `"sample_data/regression"`
  - NumPy arrays: `(X, y)` or `{"X": X, "y": y}`
- `plots_visible` (bool): Show plots during training

**Returns:** RunResult with predictions and metrics

---

**`session.predict(dataset, **kwargs)`**

Make predictions using the trained pipeline.

**Parameters:**
- `dataset` (str|Path|array): Data to predict on
  - Path to data folder
  - NumPy array X
  - Dict with 'X' key

**Returns:** PredictResult with predictions

**Raises:** ValueError if session not trained

---

**`session.retrain(dataset, mode="full", **kwargs)`**

Retrain the pipeline on new data.

**Parameters:**
- `dataset` (str|Path|tuple|dict): New dataset
- `mode` (str): Retrain mode - "full", "transfer", or "finetune"

**Returns:** RunResult from retraining

**Raises:** ValueError if session not trained

---

**`session.save(path)`**

Save the trained session to a `.n4a` bundle file.

**Parameters:**
- `path` (str|Path): Output path (e.g., `"models/my_model.n4a"`)

**Returns:** Path to saved bundle

**Raises:** ValueError if session not trained

### Session Properties

**`session.name`** → str
- Session name

**`session.pipeline`** → list
- Pipeline definition (list of steps)

**`session.status`** → str
- Current status: `"initialized"`, `"trained"`, or `"error"`

**`session.is_trained`** → bool
- Whether the pipeline has been trained or loaded from bundle

**`session.history`** → list[dict]
- List of run history entries with dataset, scores, and metadata

**`session.workspace_path`** → Path
- Path to the workspace directory

**`session.runner`** → PipelineRunner
- The shared PipelineRunner instance (created lazily)

## Session Lifecycle

```
┌─────────────┐
│ initialized │ ← nirs4all.Session(pipeline=...) or nirs4all.load_session()
└──────┬──────┘
       │ session.run(dataset)
       ▼
┌─────────────┐
│   trained   │ ← Ready for predict(), retrain(), save()
└──────┬──────┘
       │ session.save() or session.retrain()
       ▼
┌─────────────┐
│   saved/    │ ← Persistent bundle file
│  updated    │   Can load and return to trained state
└─────────────┘
```

**Methods available by state:**

| State | Available Methods |
|-------|-------------------|
| `initialized` | `run()` |
| `trained` | `predict()`, `retrain()`, `save()`, `run()` (retrain) |
| `loaded` (from bundle) | `predict()`, `retrain()`, `save()`, `run()` (retrain) |
| `error` | None (must recreate session) |

## Session vs Functional API

| Feature | Session API | Functional API (`nirs4all.run()`) |
|---------|-------------|-----------------------------------|
| State management | ✅ Stateful | ❌ Stateless |
| Multi-step workflows | ✅ Native | ❌ Manual |
| Model persistence | ✅ `session.save()` | ⚠️ `result.export()` |
| Prediction | ✅ `session.predict()` | ⚠️ `nirs4all.predict(result.best)` |
| Retraining | ✅ `session.retrain()` | ⚠️ `nirs4all.retrain()` |
| Resource sharing | ✅ Context manager | ❌ None |
| Code verbosity | Lower for workflows | Lower for single runs |
| Best for | Production, iterative work | Quick experiments, one-off analysis |

**Recommendation:**
- Use **Session API** for production pipelines, model deployment, and iterative experimentation
- Use **Functional API** for quick analysis, one-off experiments, and Jupyter notebooks

## Common Patterns

### Production Pipeline

```python
# config.py
PRODUCTION_PIPELINE = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=10, test_size=0.15, random_state=42),
    SNV(),
    {"model": PLSRegression(n_components=20)},
]

# train.py
import nirs4all
from config import PRODUCTION_PIPELINE

session = nirs4all.Session(
    pipeline=PRODUCTION_PIPELINE,
    name="Production_Wheat_v3",
    workspace_path="/data/production_workspace/",
    random_state=42,
    verbose=2
)

result = session.run(dataset="/data/wheat_samples/")
print(f"Training complete: RMSE = {result.best_rmse:.4f}")

session.save("/models/wheat_v3.n4a")
print("Model deployed")

# predict.py (deployed system)
import nirs4all
import numpy as np

session = nirs4all.load_session("/models/wheat_v3.n4a")

def predict_sample(spectra: np.ndarray) -> float:
    """Predict protein content from NIR spectra."""
    result = session.predict(spectra)
    return result.values[0]
```

### Iterative Experimentation

```python
# Quick iteration on model hyperparameters
session = nirs4all.Session(pipeline=base_pipeline, name="Experiment")

# Try different component counts
for n_comp in [5, 10, 15, 20]:
    session.pipeline[-1] = {"model": PLSRegression(n_components=n_comp)}
    result = session.run("data/train/")
    print(f"n_components={n_comp}: RMSE={result.best_rmse:.4f}")
```

### Multi-Model Deployment

```python
# Load multiple models for ensemble prediction
models = {
    "pls": nirs4all.load_session("models/pls_model.n4a"),
    "rf": nirs4all.load_session("models/rf_model.n4a"),
    "nicon": nirs4all.load_session("models/nicon_model.n4a"),
}

# Ensemble prediction
X_new = load_sample()
predictions = {name: sess.predict(X_new).values[0]
               for name, sess in models.items()}

ensemble_pred = np.mean(list(predictions.values()))
print(f"Ensemble prediction: {ensemble_pred:.2f}")
```

## See Also

- {doc}`making_predictions` - Prediction fundamentals
- {doc}`../deployment/export_bundles` - Bundle format and export options
- {doc}`advanced_predictions` - Retraining and transfer learning
- {doc}`/reference/predictions_api` - PredictResult API reference

```{seealso}
**Related Examples:**
- [D01: Session Workflow](../../../examples/developer/06_internals/D01_session_workflow.py) - Complete session API workflow guide
- [U03: Workspace Management](../../../examples/user/06_deployment/U03_workspace_management.py) - Session context and workspace structure
- [U01: Save, Load, Predict](../../../examples/user/06_deployment/U01_save_load_predict.py) - Basic prediction workflow
```
