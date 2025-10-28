# Outputs vs Artifacts: Serialization Architecture

## Overview

The nirs4all serialization system distinguishes between two types of saved files:

1. **Artifacts** - Internal binary objects (models, transformers, scalers) stored in content-addressed storage
2. **Outputs** - Human-readable files (charts, reports, CSV) stored in organized directories

## Architecture

### Artifacts (Internal Binary Storage)

**Purpose:** Deduplicated storage of trained models and transformers
**Location:** `results/artifacts/objects/<hash[:2]>/<hash>.<ext>`
**Method:** `runner.saver.persist_artifact()`
**Respects save_files flag:** ✅ YES

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
artifact = runner.saver.persist_artifact(
    step_number=runner.step_number,
    name="RandomForest_model.joblib",
    obj=trained_model,
    format_hint='sklearn_joblib'
)
# Saved to: results/artifacts/objects/ab/abc123...joblib
```

### Outputs (Human-Readable Files)

**Purpose:** User-accessible files for viewing and sharing
**Location:** `results/outputs/<dataset>_<pipeline>/<filename>`
**Method:** `runner.saver.save_output()`
**Respects save_files flag:** ✅ YES

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
output_path = runner.saver.save_output(
    step_number=runner.step_number,
    name="2D_Chart",
    data=img_png_binary,
    extension='.png'
)
# Saved to: results/outputs/regression_Q1_47be36/2D_Chart.png
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

## save_files Flag Behavior

The `save_files` parameter in `PipelineRunner` now correctly controls BOTH artifacts and outputs:

```python
# Save everything (default)
runner = PipelineRunner(save_files=True)

# Dry run - no files saved
runner = PipelineRunner(save_files=False)
```

When `save_files=False`:
- ✅ **persist_artifact()** returns metadata but doesn't save files
- ✅ **save_output()** returns None and doesn't create files
- ✅ Pipeline can still run and generate predictions
- ✅ No disk space used

## Code Examples

### Saving a Chart (Output)

```python
def execute(self, step, operator, dataset, context, runner, source, mode, loaded_binaries, prediction_store):
    # Generate chart
    fig, ax = plt.subplots()
    ax.plot(data)

    # Save to buffer
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300)
    img_png_binary = img_buffer.getvalue()

    # Save as human-readable output
    output_path = runner.saver.save_output(
        step_number=runner.step_number,
        name="2D_Chart",
        data=img_png_binary,
        extension='.png'
    )

    if output_path:
        print(f"📊 Chart saved to: {output_path}")

    return context, []
```

### Saving a Model (Artifact)

```python
def _save_model(self, model, model_name, runner):
    artifact = runner.saver.persist_artifact(
        step_number=runner.step_number,
        name=f"{model_name}.joblib",
        obj=model,
        format_hint='sklearn_joblib'
    )

    # Artifact contains:
    # {
    #   "hash": "sha256:abc123...",
    #   "name": "RandomForest.joblib",
    #   "path": "objects/ab/abc123...joblib",
    #   "format": "sklearn_joblib",
    #   "size": 15360,
    #   "step": 3
    # }

    return artifact
```

## Migration Notes

### Before (Old System)

Charts were saved with `persist_artifact()`:
- ❌ Stored in content-addressed storage
- ❌ Hard to find (`objects/ab/abc123...png`)
- ❌ No human-readable organization
- ✅ But deduplicated if identical

### After (New System)

Charts are saved with `save_output()`:
- ✅ Stored in organized directories
- ✅ Easy to find (`outputs/dataset_pipeline/Chart.png`)
- ✅ Human-readable names
- ✅ Respects save_files flag
- ⚠️ No deduplication (acceptable for outputs)

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

1. **For human viewing** (charts, reports) → Use `save_output()`
2. **For pipeline replay** (models, transformers) → Use `persist_artifact()`
3. **Disable saving for tests** → Set `save_files=False`
4. **Check outputs directory** → `results/outputs/<dataset>_<pipeline>/`

## Implementation Details

### SimulationSaver Class

```python
class SimulationSaver:
    def __init__(self, base_path, save_files=True):
        self.save_files = save_files  # ← Controls saving behavior

    def persist_artifact(self, step_number, name, obj, format_hint=None):
        if not self.save_files:
            return {"skipped": True, "reason": "save_files=False"}
        # ... save to artifacts/objects/

    def save_output(self, step_number, name, data, extension):
        if not self.save_files:
            return None
        # ... save to outputs/<dataset>_<pipeline>/
```

### Updated Controllers

**Chart Controllers:**
- `op_spectra_charts.py` - Uses `save_output()` for 2D/3D charts
- `op_y_chart.py` - Uses `save_output()` for Y distribution charts
- `op_fold_charts.py` - Uses `save_output()` for fold visualization

**Model Controllers:**
- `base_model_controller.py` - Uses `persist_artifact()` for trained models
- `op_transformermixin.py` - Uses `persist_artifact()` for fitted transformers
- `op_split.py` - Uses `persist_artifact()` for splitters

## FAQ

**Q: Why not store charts as artifacts?**
A: Charts are outputs meant for human viewing, not pipeline replay. They don't need deduplication or content-addressing.

**Q: Where did my charts go after the refactoring?**
A: Check `results/outputs/<dataset>_<pipeline>/` instead of the old artifact storage.

**Q: Can I disable chart saving?**
A: Yes! Set `save_files=False` when creating `PipelineRunner`.

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
| **Respects save_files** | ✅ Yes | ✅ Yes |
| **Examples** | Models, transformers | Charts, reports |

The new system provides the best of both worlds: efficient storage for internal objects and easy access for human outputs!
