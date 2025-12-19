# Outputs vs Artifacts: Serialization Architecture

## Overview

The nirs4all serialization system distinguishes between two types of saved files:

1. **Artifacts** - Internal binary objects (models, transformers, scalers) stored in content-addressed storage
2. **Outputs** - Human-readable files (charts, reports, CSV) stored in organized directories

> **Note:** This document describes the artifacts architecture overview. For the complete
> artifacts system v2 with branching, stacking, cleanup utilities, and CLI tools, see
> [Artifacts System v2](./artifacts_system_v2.md).

## Architecture: "Return, Don't Save"

To ensure clean separation of concerns and testability, controllers **do not save files directly**. Instead, they return a `StepOutput` object containing the data to be saved. The `PipelineExecutor` handles the actual file I/O.

### Artifacts (Internal Binary Storage)

**Purpose:** Deduplicated storage of trained models and transformers
**Location:** `results/artifacts/objects/<hash[:2]>/<hash>.<ext>`
**Method:** Return `StepOutput(artifacts={...})`
**Respects save_artifacts flag:** ✅ YES

**What gets stored as artifacts:**
- Trained models (sklearn, keras, pytorch, catboost, lightgbm)
- Fitted transformers (scalers, preprocessors)
- Fitted splitters (cross-validation objects)
- Resampler objects

**Benefits:**
- Automatic deduplication (identical objects stored once)
- Content-addressed (SHA256) for integrity
- Referenced in manifest.yaml for pipeline replay
- Space-efficient (e.g., 25% reduction in tests)

**Example:**
```python
# In model controller
from nirs4all.pipeline.execution.result import StepOutput

return context, StepOutput(
    artifacts={"model": trained_model}
)
# Executor saves to: results/artifacts/objects/ab/abc123...joblib
```

### Outputs (Human-Readable Files)

**Purpose:** User-accessible files for viewing and sharing
**Location:** `results/outputs/<dataset>_<pipeline>/<filename>`
**Method:** Return `StepOutput(outputs=[...])`
**Respects save_charts flag:** ✅ YES

**What gets stored as outputs:**
- Charts (PNG images)
- Reports (CSV, TXT)
- Summaries
- Exported predictions

**Benefits:**
- Easy to find and open
- Organized by dataset and pipeline
- Readable names (e.g., `2D_Chart.png`)
- Can be copied, shared, or included in papers

**Example:**
```python
# In chart controller
from nirs4all.pipeline.execution.result import StepOutput

# Generate chart data
img_png_binary = ...

return context, StepOutput(
    outputs=[(img_png_binary, "2D_Chart", "png")]
)
# Executor saves to: results/outputs/regression_Q1_47be36/2D_Chart.png
```

## Directory Structure

```
results/
├── artifacts/                    # Binary artifacts (models, transformers)
│   └── objects/
│       ├── ab/
│       │   └── abc123...joblib   # Content-addressed storage
│       └── cd/
│           └── cdef456...pkl
│
├── outputs/                      # Human-readable outputs
│   ├── regression_Q1_47be36/
│   │   ├── 2D_Chart.png         # ← Easy to find!
│   │   ├── Y_distribution_train_test.png
│   │   └── fold_visualization_3folds_train.png
│   └── classification_Q1_c3abeb/
│       ├── 2D_Chart.png
│       └── fold_visualization_traintest_split_train.png
│
├── pipelines/                    # Pipeline manifests
│   └── <uid>/
│       └── manifest.yaml         # References artifacts by hash
│
└── datasets/                     # Dataset indexes
    └── <dataset-name>/
        └── index.yaml
```

## save_artifacts / save_charts Flag Behavior

The `save_artifacts` and `save_charts` parameters in `PipelineRunner` control artifacts and outputs separately:

```python
# Save everything (default)
runner = PipelineRunner(save_artifacts=True, save_charts=True)

# Save only artifacts (models, transformers)
runner = PipelineRunner(save_artifacts=True, save_charts=False)

# Save only charts (visualizations)
runner = PipelineRunner(save_artifacts=False, save_charts=True)

# Dry run - no files saved
runner = PipelineRunner(save_artifacts=False, save_charts=False)
```

When `save_artifacts=False`:
- ✅ **Executor** skips saving artifacts
- ✅ Pipeline can still run and generate predictions
- ❌ Models won't be reloadable for predict mode

When `save_charts=False`:
- ✅ **Executor** skips saving outputs (charts, reports)
- ✅ Pipeline runs faster
- ✅ No chart files created

## Code Examples

### Saving a Chart (Output)

```python
def execute(self, step_info, dataset, context, runtime_context, ...):
    # Generate chart
    fig, ax = plt.subplots()
    ax.plot(data)

    # Save to buffer
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300)
    img_png_binary = img_buffer.getvalue()

    # Return StepOutput
    return context, StepOutput(
        outputs=[(img_png_binary, "2D_Chart", "png")]
    )
```

### Saving a Model (Artifact)

```python
def execute(self, step_info, dataset, context, runtime_context, ...):
    # Train model
    model.fit(X, y)

    # Return StepOutput
    return context, StepOutput(
        artifacts={"model": model}
    )
```

## Migration Notes

### Before (Old System)

Controllers called `saver.save_output()` or `saver.persist_artifact()` directly:
- ❌ Coupled to file system
- ❌ Hard to test without mocking I/O
- ❌ Inconsistent return types

### After (New System)

Controllers return `StepOutput`:
- ✅ Decoupled from I/O
- ✅ Easy to test (check returned object)
- ✅ Consistent return type (`StepOutput`)

## Finding Your Files

### Charts and Reports (Outputs)

```bash
# All outputs organized by pipeline
results/outputs/
├── regression_Q1_47be36/
│   ├── 2D_Chart.png              # ← Your charts are here!
│   ├── 3D_Chart.png
│   ├── Y_distribution_train_test.png
│   └── fold_visualization_3folds_train.png
```

### Models and Transformers (Artifacts)

```bash
# Check the manifest for artifact references
results/pipelines/<uid>/manifest.yaml

# Artifacts are in content-addressed storage
results/artifacts/objects/ab/abc123...joblib  # Model file
results/artifacts/objects/cd/cdef456...pkl    # Scaler file
```

## Best Practices

1. **For human viewing** (charts, reports) → Return in `outputs` list
2. **For pipeline replay** (models, transformers) → Return in `artifacts` dict
3. **Disable saving for tests** → Set `save_artifacts=False, save_charts=False`
4. **Check outputs directory** → `results/outputs/<dataset>_<pipeline>/`

## Implementation Details

### StepOutput Class

```python
@dataclass
class StepOutput:
    """Standardized output from a controller execution."""
    # Internal binaries (models, transformers)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    # User outputs (charts, reports)
    # List of tuples: (data_object, filename_hint, type_hint)
    outputs: List[Tuple[Any, str, str]] = field(default_factory=list)
```

### PipelineExecutor

The executor handles the actual saving:

```python
# In PipelineExecutor._execute_steps
for output_data, name, ext in step_result.outputs:
    self.saver.save_output(name=name, data=output_data, extension=ext)

for name, artifact in step_result.artifacts.items():
    self.saver.persist_artifact(step_number, name, artifact)
```

## FAQ

**Q: Why not store charts as artifacts?**
A: Charts are outputs meant for human viewing, not pipeline replay. They don't need deduplication or content-addressing.

**Q: Where did my charts go after the refactoring?**
A: Check `results/outputs/<dataset>_<pipeline>/` instead of the old artifact storage.

**Q: Can I disable chart saving?**
A: Yes! Set `save_artifacts=False, save_charts=False` when creating `PipelineRunner`.

**Q: What if two pipelines generate the same chart?**
A: Each pipeline gets its own outputs directory, so charts won't conflict.

**Q: Can I extract artifacts to readable locations?**
A: Models are binary - not human-readable. Use the manifest or prediction system to load them.

## Summary

| Feature | Artifacts | Outputs |
|---------|-----------|---------|
| **Purpose** | Internal binary objects | Human-readable files |
| **Location** | `artifacts/objects/` | `outputs/<dataset>_<pipeline>/` |
| **Names** | Hash-based | Human-readable |
| **Deduplication** | ✅ Yes | ❌ No |
| **Easy to find** | ❌ No | ✅ Yes |
| **Respects save flag** | save_artifacts | save_charts |
| **Examples** | Models, transformers | Charts, reports |

The new system provides the best of both worlds: efficient storage for internal objects and easy access for human outputs!
