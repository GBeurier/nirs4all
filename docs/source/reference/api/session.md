# Session API Reference

Complete API reference for the Session class and related functions.

## Session Class

```{eval-rst}
.. autoclass:: nirs4all.api.session.Session
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __enter__, __exit__, __repr__
```

## Module Functions

### nirs4all.session()

```{eval-rst}
.. autofunction:: nirs4all.api.session.session
```

Context manager for creating a session with automatic resource cleanup.

**Signature:**
```python
nirs4all.session(
    pipeline: Optional[List[Any]] = None,
    name: str = "",
    **kwargs: Any
) -> Generator[Session, None, None]
```

**Parameters:**
- `pipeline` (list, optional): Pipeline definition for stateful mode. If provided, the session can use `run()`, `predict()`, `retrain()`, and `save()` methods.
- `name` (str): Session name for identification. Default: `""`.
- `**kwargs`: Additional arguments passed to the Session constructor and ultimately to PipelineRunner.

**Common kwargs:**
- `verbose` (int): Verbosity level (0-3). Default: 1
  - 0: Silent
  - 1: Basic progress
  - 2: Detailed step info
  - 3: Debug output
- `save_artifacts` (bool): Whether to save model artifacts. Default: True
- `workspace_path` (str|Path): Path to workspace directory. Default: `"workspace/"`
- `random_state` (int): Random seed for reproducibility
- `plots_visible` (bool): Show plots during training. Default: False

**Yields:**
- Session object that will be automatically closed when the context exits

**Example (Resource Sharing Mode):**
```python
with nirs4all.session(verbose=2, save_artifacts=True) as s:
    r1 = nirs4all.run(pipeline1, data1, session=s)
    r2 = nirs4all.run(pipeline2, data2, session=s)
    print(f"PLS: {r1.best_score:.4f}, RF: {r2.best_score:.4f}")
```

**Example (Stateful Pipeline Mode):**
```python
with nirs4all.session(pipeline=my_pipeline, name="Demo", verbose=1) as sess:
    result = sess.run("sample_data/regression")
    predictions = sess.predict(X_new)
    print(f"Best score: {result.best_score:.4f}")
```

---

### nirs4all.load_session()

```{eval-rst}
.. autofunction:: nirs4all.api.session.load_session
```

Load a session from a saved `.n4a` bundle file.

**Signature:**
```python
nirs4all.load_session(path: Union[str, Path]) -> Session
```

**Parameters:**
- `path` (str|Path): Path to the `.n4a` bundle file to load

**Returns:**
- Session object ready for prediction, with status set to `"trained"`

**Raises:**
- `FileNotFoundError`: If the bundle file does not exist

**Example:**
```python
import nirs4all

# Load a saved session
session = nirs4all.load_session("exports/wheat_model.n4a")

print(f"Loaded: {session.name}")
print(f"Status: {session.status}")  # 'trained'
print(f"Is trained: {session.is_trained}")  # True

# Predict immediately
predictions = session.predict(X_new)
```

**Notes:**
- The loaded session maintains a reference to the bundle file for predictions
- The session can predict, retrain, or be saved to a new location
- The original bundle file is not modified by the loaded session

## Session Methods

### __init__()

Initialize a new session.

**Signature:**
```python
Session(
    pipeline: Optional[List[Any]] = None,
    name: str = "",
    **runner_kwargs: Any
)
```

**Parameters:**
- `pipeline` (list, optional): Pipeline definition for stateful mode. If provided, enables `run()`, `predict()`, `retrain()`, and `save()` methods.
- `name` (str): Name for the session/pipeline. Default: `"Session"`
- `**runner_kwargs`: Arguments passed to the underlying PipelineRunner

**Runner kwargs:**
- `verbose` (int): Verbosity level (0-3)
- `save_artifacts` (bool): Save artifacts to workspace
- `workspace_path` (str|Path): Workspace directory
- `random_state` (int): Random seed
- `plots_visible` (bool): Show plots during training
- See PipelineRunner documentation for complete list

**Example:**
```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

pipeline = [MinMaxScaler(), {"model": PLSRegression(10)}]

session = nirs4all.Session(
    pipeline=pipeline,
    name="WheatProtein",
    verbose=1,
    workspace_path="my_workspace/",
    random_state=42
)
```

---

### run()

Train the session's pipeline on a dataset.

**Signature:**
```python
session.run(
    dataset: Union[str, Path, Any],
    *,
    plots_visible: bool = False,
    **kwargs: Any
) -> RunResult
```

**Parameters:**
- `dataset`: Dataset to train on. Accepts:
  - Path to data folder (str|Path): `"sample_data/regression"`
  - NumPy arrays (tuple): `(X, y)`
  - Dict: `{"X": X, "y": y}`
  - DatasetConfigs object
- `plots_visible` (bool): Whether to show plots during training. Default: False
- `**kwargs`: Additional arguments passed to `runner.run()`

**Valid kwargs:**
- `pipeline_name` (str): Override pipeline name
- `dataset_name` (str): Override dataset name
- `max_generation_count` (int): Maximum generator variants to evaluate

**Returns:**
- RunResult object with predictions, metrics, and model references

**Raises:**
- `ValueError`: If no pipeline was provided to the session
- `Exception`: Propagates any training errors (sets session status to `"error"`)

**Side Effects:**
- Sets `session.status` to `"trained"` on success or `"error"` on failure
- Updates `session.history` with training record
- Stores result in `session._last_result` for prediction

**Example:**
```python
session = nirs4all.Session(pipeline=pipeline, name="Demo")

# Train on folder
result = session.run(dataset="data/wheat/")

# Train on arrays
result = session.run(dataset=(X_train, y_train))

# Train with plots
result = session.run(dataset="data/wheat/", plots_visible=True)

print(f"Best RMSE: {result.best_rmse:.4f}")
print(f"Session status: {session.status}")
```

---

### predict()

Make predictions using the trained pipeline.

**Signature:**
```python
session.predict(
    dataset: Union[str, Path, Any],
    **kwargs: Any
) -> PredictResult
```

**Parameters:**
- `dataset`: Data to predict on. Accepts:
  - Path to data folder (str|Path)
  - NumPy array: X
  - Dict with 'X' key: `{"X": X}`
  - DatasetConfigs object
- `**kwargs`: Additional arguments for prediction

**Returns:**
- PredictResult with predictions array and metadata

**Raises:**
- `ValueError`: If session has not been trained (call `session.run()` first)
- `ValueError`: If no trained model is available

**Behavior:**
- Uses the best model from the last training run
- For loaded sessions, uses the bundle file directly
- Applies all preprocessing steps automatically
- Performs ensemble averaging for cross-validation models

**Example:**
```python
import numpy as np

# Train session
session = nirs4all.Session(pipeline=pipeline, name="Demo")
result = session.run("data/train/")

# Predict on new data
X_new = np.random.randn(20, 100)
predictions = session.predict(X_new)

print(f"Shape: {predictions.shape}")
print(f"Values: {predictions.values}")
print(f"Model: {predictions.model_name}")

# Predict on folder
predictions = session.predict("data/test/")
```

---

### retrain()

Retrain the pipeline on new data.

**Signature:**
```python
session.retrain(
    dataset: Union[str, Path, Any],
    mode: str = "full",
    **kwargs: Any
) -> RunResult
```

**Parameters:**
- `dataset`: New dataset to train on (same formats as `run()`)
- `mode` (str): Retrain mode. Options:
  - `"full"`: Complete retraining from scratch
  - `"transfer"`: Transfer learning (keep preprocessing, retrain model)
  - `"finetune"`: Fine-tuning (update model weights incrementally)
- `**kwargs`: Additional arguments for retraining

**Returns:**
- RunResult from the retraining operation

**Raises:**
- `ValueError`: If session has not been trained (call `session.run()` first)
- `ValueError`: If no trained model is available for retraining

**Side Effects:**
- Updates `session._last_result` with new training result
- Appends entry to `session.history` with retrain mode

**Example:**
```python
# Initial training
session = nirs4all.Session(pipeline=pipeline, name="Adaptive")
result1 = session.run("data/batch1/")
print(f"Initial RMSE: {result1.best_rmse:.4f}")

# New data arrives - retrain
result2 = session.retrain("data/batch2/", mode="transfer")
print(f"Retrained RMSE: {result2.best_rmse:.4f}")

# Check history
for i, entry in enumerate(session.history):
    print(f"Run {i+1}: {entry.get('mode', 'run')} - {entry['best_score']:.4f}")
```

---

### save()

Save the trained session to a `.n4a` bundle file.

**Signature:**
```python
session.save(path: Union[str, Path]) -> Path
```

**Parameters:**
- `path` (str|Path): Output path for the `.n4a` bundle file

**Returns:**
- Path object pointing to the saved bundle file

**Raises:**
- `ValueError`: If session has not been trained (call `session.run()` first)

**Bundle Contents:**
- Pipeline definition (chain of steps)
- Fitted preprocessing artifacts (scalers, transformers, etc.)
- Trained model weights for all CV folds
- Metadata (name, configuration, training info)
- Artifact checksums for integrity verification

**Example:**
```python
from pathlib import Path

# Train and save
session = nirs4all.Session(pipeline=pipeline, name="Production_v1")
result = session.run("data/train/")

save_path = Path("exports/production_v1.n4a")
save_path.parent.mkdir(parents=True, exist_ok=True)

bundle_path = session.save(save_path)
print(f"Saved: {bundle_path}")
print(f"Size: {bundle_path.stat().st_size / 1024:.1f} KB")

# Load and use
loaded = nirs4all.load_session(bundle_path)
predictions = loaded.predict(X_new)
```

---

### close()

Clean up session resources.

**Signature:**
```python
session.close() -> None
```

**Parameters:**
- None

**Returns:**
- None

**Behavior:**
- Releases the PipelineRunner instance
- Called automatically when exiting a context manager
- Safe to call multiple times

**Example:**
```python
# Manual cleanup
session = nirs4all.Session(pipeline=pipeline)
result = session.run(dataset)
session.close()

# Automatic cleanup with context manager
with nirs4all.session(pipeline=pipeline) as sess:
    result = sess.run(dataset)
# close() called automatically here
```

## Session Properties

### name

Get the session name.

**Type:** `str`

**Example:**
```python
session = nirs4all.Session(pipeline=pipeline, name="Production_v2")
print(session.name)  # "Production_v2"
```

---

### pipeline

Get the pipeline definition.

**Type:** `Optional[List[Any]]`

**Returns:**
- List of pipeline steps, or None if created in resource-sharing mode

**Example:**
```python
session = nirs4all.Session(pipeline=[MinMaxScaler(), PLSRegression(10)])
print(f"Steps: {len(session.pipeline)}")
for i, step in enumerate(session.pipeline):
    print(f"  {i+1}. {type(step).__name__}")
```

---

### status

Get the current session status.

**Type:** `str`

**Possible Values:**
- `"initialized"`: Session created but not yet trained
- `"trained"`: Training completed successfully
- `"error"`: Training encountered an error

**Example:**
```python
session = nirs4all.Session(pipeline=pipeline)
print(session.status)  # "initialized"

result = session.run(dataset)
print(session.status)  # "trained"
```

---

### is_trained

Check if the pipeline has been trained or loaded from a bundle.

**Type:** `bool`

**Returns:**
- `True` if the session has been trained via `run()` or loaded via `load_session()`
- `False` otherwise

**Example:**
```python
session = nirs4all.Session(pipeline=pipeline)
print(session.is_trained)  # False

session.run(dataset)
print(session.is_trained)  # True

# Also true for loaded sessions
loaded = nirs4all.load_session("model.n4a")
print(loaded.is_trained)  # True
```

---

### history

Get the run history for this session.

**Type:** `List[Dict[str, Any]]`

**Returns:**
- List of dictionaries, one per `run()` or `retrain()` call
- Each entry contains:
  - `dataset` (str): Dataset identifier
  - `best_score` (float): Best score from that run
  - `num_predictions` (int): Number of predictions generated
  - `mode` (str): Retrain mode (for `retrain()` calls)

**Example:**
```python
session = nirs4all.Session(pipeline=pipeline, name="Iterative")

# Run 1
r1 = session.run("data/batch1/")

# Run 2 (retrain)
r2 = session.retrain("data/batch2/", mode="transfer")

# Check history
print(f"Total runs: {len(session.history)}")
for i, entry in enumerate(session.history, 1):
    mode = entry.get('mode', 'initial')
    score = entry['best_score']
    dataset = entry['dataset']
    print(f"{i}. {mode}: {dataset} -> {score:.4f}")
```

---

### workspace_path

Get the workspace path from the runner.

**Type:** `Optional[Path]`

**Returns:**
- Path to the workspace directory, or None if runner not yet created

**Example:**
```python
session = nirs4all.Session(
    pipeline=pipeline,
    workspace_path="my_workspace/"
)

print(session.workspace_path)  # Path("my_workspace/")

# Access runner to ensure it's created
runner = session.runner
print(session.workspace_path)  # Now guaranteed to be set
```

---

### runner

Get or create the shared PipelineRunner instance.

**Type:** `PipelineRunner`

**Returns:**
- The shared PipelineRunner instance (created lazily on first access)

**Behavior:**
- The runner is created on first access with the kwargs provided to the Session constructor
- Subsequent accesses return the same instance
- The runner persists across multiple `run()` calls

**Example:**
```python
session = nirs4all.Session(pipeline=pipeline, verbose=1)

# Runner created on first access
runner = session.runner
print(type(runner))  # <class 'nirs4all.pipeline.runner.PipelineRunner'>

# Same instance on subsequent access
runner2 = session.runner
assert runner is runner2  # True
```

## Special Methods

### __enter__() and __exit__()

Context manager protocol support.

**Signature:**
```python
def __enter__(self) -> Session:
    ...

def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    ...
```

**Behavior:**
- `__enter__()`: Returns the session itself
- `__exit__()`: Calls `close()` to release resources

**Example:**
```python
with nirs4all.Session(pipeline=pipeline, name="Context") as sess:
    result = sess.run(dataset)
    # Session automatically closed on exit
```

---

### __repr__()

String representation of the session.

**Signature:**
```python
def __repr__(self) -> str:
    ...
```

**Returns:**
- Human-readable string describing the session

**Format:**
- With pipeline: `Session(name='...', status='...', steps=N)`
- Without pipeline: `Session(active/idle, kwargs=[...])`

**Example:**
```python
session = nirs4all.Session(pipeline=[MinMaxScaler(), PLSRegression(10)], name="Demo")
print(session)  # Session(name='Demo', status='initialized', steps=2)

session.run(dataset)
print(session)  # Session(name='Demo', status='trained', steps=2)

resource_session = nirs4all.Session(verbose=1)
print(resource_session)  # Session(idle, kwargs=['verbose'])
```

## Complete Examples

### Basic Session Workflow

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

# Create session
pipeline = [MinMaxScaler(), {"model": PLSRegression(10)}]
session = nirs4all.Session(pipeline=pipeline, name="BasicDemo", verbose=1)

# Train
result = session.run("sample_data/regression")
print(f"RMSE: {result.best_rmse:.4f}")

# Predict
predictions = session.predict(X_new)
print(f"Predictions: {predictions.values}")

# Save
session.save("exports/basic_model.n4a")
```

### Context Manager Usage

```python
import nirs4all

# Automatic cleanup
with nirs4all.session(pipeline=pipeline, name="ContextDemo") as sess:
    result = sess.run("sample_data/regression")
    predictions = sess.predict(X_new)
    print(f"Best score: {result.best_score:.4f}")
# Resources released here

# Resource sharing across multiple runs
with nirs4all.session(verbose=0, save_artifacts=False) as shared:
    r1 = nirs4all.run(pipeline1, dataset, session=shared)
    r2 = nirs4all.run(pipeline2, dataset, session=shared)
    print(f"PLS: {r1.best_score:.4f}")
    print(f"RF: {r2.best_score:.4f}")
```

### Session State Management

```python
import nirs4all

session = nirs4all.Session(pipeline=pipeline, name="StateDemo")

# Check initial state
print(f"Status: {session.status}")  # 'initialized'
print(f"Is trained: {session.is_trained}")  # False

# Train
result = session.run(dataset)
print(f"Status: {session.status}")  # 'trained'
print(f"Is trained: {session.is_trained}")  # True

# Predict only after training
if session.is_trained:
    predictions = session.predict(X_new)

# Check history
print(f"Runs: {len(session.history)}")
for entry in session.history:
    print(f"  {entry['dataset']}: {entry['best_score']:.4f}")
```

### Load and Predict Pattern

```python
import nirs4all
import numpy as np

# Load a saved session
session = nirs4all.load_session("exports/deployed_model.n4a")

# Verify it's ready
assert session.is_trained, "Model not trained"
print(f"Loaded model: {session.name}")

# Predict on new samples
X_samples = np.random.randn(10, 100)
predictions = session.predict(X_samples)

print(f"Predicted {len(predictions)} samples")
print(f"Mean prediction: {predictions.values.mean():.2f}")
print(f"Std prediction: {predictions.values.std():.2f}")
```

## See Also

- {doc}`/user_guide/predictions/session_api` - Session API user guide
- {doc}`/user_guide/predictions/making_predictions` - Prediction workflows
- {doc}`/user_guide/deployment/export_bundles` - Bundle format details
- {doc}`predictions_api` - PredictResult reference
