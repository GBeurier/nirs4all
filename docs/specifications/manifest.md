# Serialization & Manifest Architecture

## Overview

The nirs4all serialization system uses **content-addressed storage** with **UID-based pipeline manifests** to manage binary artifacts efficiently. This architecture eliminates redundancy, enables deduplication, and provides a clean, filesystem-based approach without requiring a database.

## Architecture Principles

1. **Content-Addressed Storage**: Artifacts are stored once and referenced by their SHA256 hash
2. **UID-Based Pipelines**: Each pipeline gets a unique ID and a single manifest.yaml file
3. **No Database**: Pure YAML/JSON/binary files for simplicity and portability
4. **Framework-Aware Serialization**: Automatic detection and optimal format selection
5. **Backward Compatibility**: Supports loading from legacy metadata.json format

## Filesystem Structure

```
results/
├── artifacts/
│   └── objects/
│       ├── ab/
│       │   ├── abc123def456...pkl     # Deduplicated binary artifacts
│       │   └── abc789...joblib
│       ├── cd/
│       │   └── cdef012...keras
│       └── ...
│
├── pipelines/
│   ├── <uid-1>/
│   │   └── manifest.yaml              # Single file per pipeline
│   ├── <uid-2>/
│   │   └── manifest.yaml
│   └── ...
│
└── datasets/
    ├── <dataset-name-1>/
    │   └── index.yaml                 # Maps pipeline names → UIDs
    ├── <dataset-name-2>/
    │   └── index.yaml
    └── ...
```

### Content-Addressed Storage (`artifacts/objects/`)

Artifacts are stored using git-style sharding:
- **Path**: `objects/<hash[:2]>/<hash>.<ext>`
- **Deduplication**: Identical objects share the same hash → same file
- **Integrity**: SHA256 ensures data integrity
- **Extensions**: .pkl, .joblib, .keras, .pt, .json, .cbm, .txt (framework-specific)

**Example:**
```python
# Two pipelines use the same StandardScaler
# It's persisted once: objects/ab/abc123...pkl
# Both manifests reference: sha256:abc123...
```

### Pipeline Manifests (`pipelines/<uid>/manifest.yaml`)

Each pipeline has **one YAML file** containing all metadata:

```yaml
uid: "a1b2c3d4-e5f6-4789-abcd-ef0123456789"
name: "svm_baseline"
dataset: "corn_m5"
created_at: "2025-10-13T23:00:00Z"
version: "1.0"

pipeline:
  - MinMaxScaler: {}
  - y_processing:
      MinMaxScaler: {}
  - RepeatedKFold:
      n_splits: 3
      n_repeats: 2
  - model:
      name: "SVM_Linear"
      params: {C: 1.0}

metadata:
  config_name: "config_abc123"
  total_runtime: 45.2
  best_model: "SVM_Linear_1"

artifacts:
  - hash: "sha256:abc123def456..."
    name: "MinMaxScaler_0_0.pkl"
    path: "objects/ab/abc123def456...pkl"
    format: "sklearn_pickle"
    size: 2048
    step: 0
    saved_at: "2025-10-13T23:00:05Z"

  - hash: "sha256:def789abc012..."
    name: "SVM_Linear_1.joblib"
    path: "objects/de/def789abc012...joblib"
    format: "sklearn_joblib"
    size: 15360
    step: 3
    saved_at: "2025-10-13T23:00:15Z"

predictions:
  - id: "pred_001"
    model_name: "SVM_Linear_1"
    partition: "test"
    metrics:
      rmse: 0.245
      r2: 0.985
```

**Benefits:**
- ✅ Single file per pipeline (not 2-3 files)
- ✅ Human-readable YAML
- ✅ Easy deletion: `rm -rf pipelines/<uid>`
- ✅ Contains all metadata in one place
- ✅ Artifact references with integrity hashes

### Dataset Indexes (`datasets/<name>/index.yaml`)

Maps user-friendly pipeline names to UIDs:

```yaml
# datasets/corn_m5/index.yaml
pipelines:
  svm_baseline: "a1b2c3d4-e5f6-4789-abcd-ef0123456789"
  pls_optimized: "b2c3d4e5-f6a7-5890-bcde-f01234567890"
  rf_ensemble: "c3d4e5f6-a7b8-6901-cdef-012345678901"

created_at: "2025-10-01T10:00:00Z"
updated_at: "2025-10-13T23:00:00Z"
```

**Benefits:**
- ✅ Easy lookup: Name → UID → Manifest
- ✅ Per-dataset organization
- ✅ Simple YAML file (no database)

## Framework-Aware Serialization

### Supported Frameworks

| Framework | Detection | Format | Extension | Notes |
|-----------|-----------|--------|-----------|-------|
| **scikit-learn** | `BaseEstimator` | joblib (preferred) or pickle | `.joblib`, `.pkl` | Handles all sklearn objects |
| **TensorFlow/Keras** | `keras.Model` | Keras format | `.keras` | Native TF format |
| **PyTorch** | `torch.nn.Module` | State dict | `.pt` | Saves state_dict, not full model |
| **XGBoost** | `xgboost.Booster` | JSON | `.json` | Human-readable JSON format |
| **CatBoost** | `CatBoost` | CBM binary | `.cbm` | Native CatBoost format |
| **LightGBM** | `lgb.Booster` | Text | `.txt` | Human-readable text |
| **NumPy** | `np.ndarray` | NPY | `.npy` | Native NumPy format (no pickle) |
| **Generic** | Fallback | cloudpickle or pickle | `.pkl` | For custom objects |

### Serializer API

```python
from nirs4all.utils.serializer import persist, load

# Persist an object
artifact = persist(
    obj=trained_model,
    artifacts_dir=Path("results/artifacts/objects"),
    name="SVM_Linear_1",
    format_hint="sklearn"  # Optional, auto-detected if None
)
# Returns ArtifactMeta dict:
# {
#     "hash": "sha256:abc123...",
#     "name": "SVM_Linear_1.joblib",
#     "path": "objects/ab/abc123...joblib",
#     "format": "sklearn_joblib",
#     "size": 15360,
#     "saved_at": "2025-10-13T23:00:15Z"
# }

# Load an object
loaded_model = load(
    artifact_meta=artifact,
    results_dir=Path("results")
)
```

### Automatic Format Selection

The serializer automatically selects the best format:

```python
def _detect_framework(obj: Any) -> Optional[str]:
    """Detect ML framework from object type."""
    obj_type = type(obj).__module__

    if 'sklearn' in obj_type:
        return 'sklearn'
    elif 'tensorflow' in obj_type or 'keras' in obj_type:
        return 'tensorflow'
    elif 'torch' in obj_type:
        return 'pytorch'
    elif 'xgboost' in obj_type:
        return 'xgboost'
    elif 'catboost' in obj_type:
        return 'catboost'
    elif 'lightgbm' in obj_type:
        return 'lightgbm'
    elif 'numpy' in obj_type:
        return 'numpy'
    else:
        return None  # Generic pickle/cloudpickle
```

## Workflow Examples

### Training Workflow

```python
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.pipeline.manifest_manager import ManifestManager

# 1. Create runner (automatically creates manifest)
runner = PipelineRunner(save_files=True)
predictions, _ = runner.run(pipeline_config, dataset_config)

# Behind the scenes:
# - ManifestManager creates UID and manifest
# - Controllers persist artifacts using serializer
# - Runner appends artifact metadata to manifest
# - Dataset index updated with name → UID mapping
```

**What happens internally:**
1. `ManifestManager.create_pipeline()` generates UID
2. Controller executes and trains model
3. `runner.saver.persist_artifact()` saves to content-addressed storage
4. Returns `ArtifactMeta` to runner
5. `ManifestManager.append_artifacts()` adds to manifest
6. `ManifestManager.register_in_dataset()` updates index

### Prediction Workflow

```python
# 1. Get best prediction object (contains pipeline reference)
best = predictions.top(n=1, rank_partition="test")[0]

# 2. Create predictor and predict
predictor = PipelineRunner()
new_predictions, _ = predictor.predict(best, new_dataset)

# Behind the scenes:
# - Runner extracts pipeline UID from prediction object
# - ManifestManager loads manifest
# - BinaryLoader loads artifacts from manifest
# - Pipeline replayed with loaded artifacts
```

**What happens internally:**
1. `prepare_replay()` extracts pipeline info
2. `ManifestManager.load_manifest(uid)` reads YAML
3. `BinaryLoader.from_manifest()` creates loader
4. `serializer.load()` deserializes each artifact
5. Pipeline runs in prediction mode

### Garbage Collection

```bash
# Show statistics
python scripts/gc_artifacts.py --stats

# Dry run (show what would be deleted)
python scripts/gc_artifacts.py

# Actually delete orphaned artifacts
python scripts/gc_artifacts.py --force
```

**What it does:**
1. Scans all artifacts in `objects/` directory
2. Scans all manifests to find referenced artifacts
3. Computes set difference → orphaned artifacts
4. Optionally deletes orphans and shows space savings

## Backward Compatibility

The system supports loading from **legacy metadata.json** format:

```python
# Old format: results/<dataset>/<config>/
#   - pipeline.json
#   - metadata.json
#   - binaries: {step: [filename, ...]}
#   - <filename>.pkl (raw pickle files)

# New system detects this and converts on-the-fly:
if not manifest_path.exists():
    # Load legacy metadata.json
    metadata = json.loads(metadata_file.read_text())

    # Convert to artifact format
    artifacts = []
    for step_key, binary_list in metadata['binaries'].items():
        step_num = int(step_key.split('_')[0])
        for binary_filename in binary_list:
            artifacts.append({
                "name": binary_filename,
                "step": step_num,
                "path": str(config_dir / binary_filename),
                "format": "legacy_pickle",
                "hash": "",
                "size": file_size
            })

    # Create BinaryLoader with converted artifacts
    binary_loader = BinaryLoader(artifacts, results_dir)
```

## Benefits of This Architecture

### 1. Content Deduplication
- **Problem**: Old system saved identical models multiple times
- **Solution**: SHA256 hash ensures each unique object stored once
- **Result**: In our tests, 147 artifacts serve 195 pipelines (25% space savings)

### 2. Clean Deletion
- **Problem**: Old system had scattered files across directories
- **Solution**: Pipeline deletion is `rm -rf pipelines/<uid>`
- **Result**: Artifacts remain for reuse by other pipelines

### 3. No Database
- **Problem**: Databases add complexity and portability issues
- **Solution**: Pure YAML/JSON/binary files
- **Result**: Easy to backup, version control manifests, move between systems

### 4. Human-Readable Metadata
- **Problem**: Binary pickle files are opaque
- **Solution**: YAML manifests are human-readable
- **Result**: Easy debugging, inspection, and manual editing if needed

### 5. Framework Optimization
- **Problem**: All models pickled regardless of framework
- **Solution**: Auto-detect and use native formats
- **Result**: Better performance, smaller files, cross-version compatibility

### 6. Integrity Verification
- **Problem**: No way to verify artifact corruption
- **Solution**: SHA256 hashes in manifests
- **Result**: Can detect corrupted artifacts

## Migration Path

### For Existing Pipelines

Existing pipelines with `metadata.json` **still work** via backward compatibility layer:

```python
# System automatically detects old format
# Converts on-the-fly without modifying files
# New pipelines use manifest system
# Old pipelines remain usable
```

### For New Pipelines

All new training automatically uses manifest system:

```python
runner = PipelineRunner(save_files=True)
predictions, _ = runner.run(pipeline_config, dataset_config)
# → Creates: results/pipelines/<uid>/manifest.yaml
# → Creates: results/datasets/<name>/index.yaml
# → Saves artifacts to: results/artifacts/objects/<hash>.<ext>
```

## Implementation Status

✅ **Phase 1**: Core serializer and manifest manager
✅ **Phase 2**: Runner integration and BinaryLoader refactor
✅ **Phase 3**: All controllers updated (7 controllers)
✅ **Phase 4**: Utilities (history, SHAP, model_builder)
✅ **Phase 5**: Testing (108 tests passing), GC script, documentation

**Test Coverage:**
- 23 serializer tests (framework detection, hashing, persist/load)
- 27 manifest manager tests (CRUD, indexing, UIDs)
- 7 integration tests (Phase 2)
- 15 comprehensive integration tests (real pipelines)
- 40+ additional tests (controllers, preprocessing, etc.)

**Real-World Validation:**
- Q5_predict.py example: ✅ All 3 prediction methods working
- Backward compatibility: ✅ Legacy metadata.json loading works
- Deduplication: ✅ 147 artifacts serving 195 pipelines
- GC script: ✅ Correctly identifies 0 orphans

## Future Enhancements

### Potential Improvements

1. **Compression**: Compress large artifacts (>1MB) with gzip/lz4
2. **Remote Storage**: Support S3/Azure Blob for artifacts
3. **Versioning**: Track artifact versions when object changes
4. **Metadata**: Store training metrics in manifest
5. **Search**: Add artifact search by framework, size, date
6. **Web UI**: Visual pipeline and artifact explorer

### Non-Goals

- ❌ No database (keep it simple)
- ❌ No embedded `_runtime_instance` (clean separation)
- ❌ No pickle for models (use framework-native formats)

## Manifest Schema v2 (Artifacts Refactoring)

The v2 schema extends the manifest format with enhanced artifact tracking for
branching pipelines, stacking/meta-models, and fold-aware artifacts.

### Schema Detection

The schema version is detected by the `artifacts` section structure:

```python
def detect_schema_version(manifest: Dict[str, Any]) -> str:
    """Detect manifest schema version."""
    artifacts = manifest.get("artifacts", {})
    if isinstance(artifacts, dict) and artifacts.get("schema_version") == "2.0":
        return "2.0"
    elif isinstance(artifacts, list):
        return "1.0"
    else:
        return "1.0"  # Legacy format
```

### v2 Schema Structure

```yaml
schema_version: "2.0"
uid: "a1b2c3d4-e5f6-4789-abcd-ef0123456789"
pipeline_id: "0001_pls_abc123"
name: "pls_baseline"
dataset: "corn_m5"
created_at: "2025-12-12T10:00:00Z"

artifacts:
  schema_version: "2.0"
  items:
    - artifact_id: "0001:0:all"
      content_hash: "sha256:abc123def456789..."
      path: "transformer_StandardScaler_abc123.pkl"
      pipeline_id: "0001"
      branch_path: []
      step_index: 0
      fold_id: null
      artifact_type: transformer
      class_name: StandardScaler
      depends_on: []
      format: joblib
      format_version: "sklearn==1.5.0"
      nirs4all_version: "0.9.0"
      size_bytes: 2048
      created_at: "2025-12-12T10:00:00Z"
      params: {}

predictions: [...]
```

### v2 Artifact Record Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `artifact_id` | string | ✅ | Deterministic ID: `<pipeline>:<branch...>:<step>:<fold>` |
| `content_hash` | string | ✅ | SHA256 hash with `sha256:` prefix |
| `path` | string | ✅ | Relative path to binary file |
| `pipeline_id` | string | ✅ | Pipeline identifier |
| `branch_path` | list[int] | ✅ | Branch indices (empty for shared) |
| `step_index` | int | ✅ | Pipeline step index |
| `fold_id` | int \| null | ✅ | CV fold ID (null for shared) |
| `artifact_type` | string | ✅ | One of: model, transformer, splitter, encoder, meta_model |
| `class_name` | string | ✅ | Python class name |
| `depends_on` | list[string] | ✅ | Dependency artifact IDs |
| `format` | string | ✅ | Serialization format (joblib, pkl, keras, pt, etc.) |
| `format_version` | string | ❌ | Library version info |
| `nirs4all_version` | string | ❌ | nirs4all version at creation |
| `size_bytes` | int | ❌ | File size in bytes |
| `created_at` | string | ❌ | ISO8601 timestamp |
| `params` | object | ❌ | Additional metadata |
| `meta_config` | object | ❌ | Meta-model configuration (for stacking) |

### Artifact ID Format

The artifact ID encodes the execution context:

```
<pipeline_id>:<branch_path...>:<step_index>:<fold_id|all>

Examples:
  "0001:3:all"        → Pipeline 0001, step 3, shared across folds
  "0001:0:3:0"        → Pipeline 0001, branch 0, step 3, fold 0
  "0001:0:2:5:all"    → Pipeline 0001, nested branch [0,2], step 5, shared
```

**Parsing:**
```python
def parse_artifact_id(artifact_id: str) -> Dict[str, Any]:
    parts = artifact_id.split(":")
    pipeline_id = parts[0]
    fold_str = parts[-1]
    step_index = int(parts[-2])
    branch_path = [int(p) for p in parts[1:-2]]
    fold_id = None if fold_str == "all" else int(fold_str)
    return {
        "pipeline_id": pipeline_id,
        "branch_path": branch_path,
        "step_index": step_index,
        "fold_id": fold_id
    }
```

### Branch-Aware Artifacts

Artifacts track their branch context:

```yaml
# Shared (pre-branch) artifact
- artifact_id: "0001:0:all"
  branch_path: []
  step_index: 0

# Branch 0 artifact
- artifact_id: "0001:0:2:all"
  branch_path: [0]
  step_index: 2

# Branch 1 artifact
- artifact_id: "0001:1:2:all"
  branch_path: [1]
  step_index: 2

# Nested branch [0, 1] artifact
- artifact_id: "0001:0:1:3:all"
  branch_path: [0, 1]
  step_index: 3
```

### Fold-Specific Artifacts

Per-fold models for CV averaging:

```yaml
# Fold-specific models
- artifact_id: "0001:3:0"
  fold_id: 0
  step_index: 3

- artifact_id: "0001:3:1"
  fold_id: 1
  step_index: 3

- artifact_id: "0001:3:2"
  fold_id: 2
  step_index: 3

# Shared transformer (fold_id: null)
- artifact_id: "0001:0:all"
  fold_id: null
  step_index: 0
```

### Meta-Model Configuration

For stacking, artifacts include source model references:

```yaml
- artifact_id: "0001:5:all"
  artifact_type: meta_model
  class_name: Ridge
  depends_on:
    - "0001:3:all"
    - "0001:4:all"
  meta_config:
    source_models:
      - artifact_id: "0001:3:all"
        feature_index: 0
      - artifact_id: "0001:4:all"
        feature_index: 1
    feature_columns:
      - "PLSRegression_pred"
      - "RandomForest_pred"
```

### v1 to v2 Compatibility

The `ArtifactLoader` handles both formats transparently:

```python
def import_from_manifest(self, manifest: Dict[str, Any]) -> None:
    """Import artifacts from v1 or v2 manifest."""
    artifacts = manifest.get("artifacts", {})

    if isinstance(artifacts, dict) and artifacts.get("schema_version") == "2.0":
        # v2 format: artifacts.items
        for item in artifacts.get("items", []):
            record = ArtifactRecord.from_dict(item)
            self._records[record.artifact_id] = record
    elif isinstance(artifacts, list):
        # v1 format: convert to ArtifactRecord
        for item in artifacts:
            record = self._convert_v1_to_record(item)
            self._records[record.artifact_id] = record
```

### Centralized Storage (v2)

v2 uses centralized storage per dataset:

```
workspace/
├── binaries/                     # v2 centralized storage
│   ├── corn_m5/
│   │   ├── model_PLSRegression_a1b2c3.joblib
│   │   └── transformer_StandardScaler_d4e5f6.pkl
│   └── wheat_protein/
│       └── ...
│
└── runs/                         # Lightweight manifests
    └── corn_m5/
        └── 20251210_abc123/
            └── 0001_pls/
                └── manifest.yaml
```

**Migration from v1:**
- v1 artifacts in `results/artifacts/objects/` continue to work
- New runs use `workspace/binaries/<dataset>/`
- Loader checks both locations

### Deduplication (v2)

Content-addressed storage with deduplication:

```yaml
# Two manifests reference same file via content_hash:
# Manifest 1:
- artifact_id: "0001:3:all"
  content_hash: "sha256:abc123..."
  path: "model_PLSRegression_abc123.joblib"

# Manifest 2:
- artifact_id: "0002:3:all"
  content_hash: "sha256:abc123..."  # Same hash
  path: "model_PLSRegression_abc123.joblib"  # Same file

# Only ONE file on disk in binaries/
```

### Dependency Graph

Artifacts track dependencies for proper loading order:

```yaml
artifacts:
  items:
    - artifact_id: "0001:0:all"
      depends_on: []

    - artifact_id: "0001:1:all"
      depends_on: ["0001:0:all"]

    - artifact_id: "0001:2:all"
      depends_on: ["0001:0:all", "0001:1:all"]
```

**Loading with dependencies:**
```python
loader = ArtifactLoader(workspace, dataset)
loader.import_from_manifest(manifest)

# Load artifact and all dependencies in topological order
all_artifacts = loader.load_with_dependencies("0001:2:all")
# Returns: {"0001:0:all": scaler, "0001:1:all": pca, "0001:2:all": model}
```

## References

- **Serializer**: `nirs4all/utils/serializer.py`
- **Manifest Manager**: `nirs4all/pipeline/manifest_manager.py`
- **Binary Loader**: `nirs4all/pipeline/binary_loader.py`
- **Artifact Registry**: `nirs4all/pipeline/storage/artifacts/registry.py`
- **Artifact Loader**: `nirs4all/pipeline/storage/artifacts/loader.py`
- **Runner**: `nirs4all/pipeline/runner.py`
- **GC Script**: `scripts/gc_artifacts.py`
- **Tests**: `tests/test_serializer.py`, `tests/test_manifest_manager.py`, `tests/unit/pipeline/storage/artifacts/`
- **User Guide**: [Artifacts System v2](../reference/artifacts_system_v2.md)
- **API Reference**: [Storage API](../api/storage.md)
