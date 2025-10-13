# Serialization & Artifact Management Refactoring

**Goal**: Eliminate redundant serialization code, remove `_runtime_instance` embedding, and centralize all binary artifact handling through a single, framework-aware serializer API with content-addressed storage and UID-based pipeline manifests.

**Breaking Change**: This refactoring intentionally breaks backward compatibility to clean the codebase. Existing pipeline JSON files with embedded `_runtime_instance` will become invalid and must be regenerated.

**Architecture**: No database - pure YAML/JSON/binary files. Content-addressed artifacts for deduplication. UID-based pipelines with single manifest files.

---

## Quick Overview

### Current Problems
- 13+ modules independently use `pickle.dumps/loads`
- `_runtime_instance` bloats pipeline JSON with non-serializable objects
- No framework awareness (TF/PyTorch models pickled)
- No content deduplication (identical models saved multiple times)
- Many folders with just 2 files (pipeline.json + metadata.json)

### Solution
```
results/
├── artifacts/objects/<hash[:2]>/<hash>.<ext>  # Deduplicated binaries
├── pipelines/<uid>/manifest.yaml              # Single file per pipeline
└── datasets/<name>/index.yaml                 # Name → UID mapping
```

**Benefits:**
- ✅ Single manifest.yaml per pipeline (not 2+ files)
- ✅ Easy deletion: `rm -rf pipelines/<uid>`
- ✅ No database (YAML/JSON only)
- ✅ Content-addressed deduplication
- ✅ Human-readable manifests
- ✅ Simple garbage collection script

### Workflow Example

**Training:**
```python
# 1. Create pipeline UID
uid = manifest_manager.create_pipeline("svm_baseline", "corn_m5", config)

# 2. Train - controller persists artifacts
transformer = StandardScaler().fit(X)
artifact = persist(transformer, artifacts_dir, "StandardScaler_0")
# → Writes to: artifacts/objects/ab/abc123...pkl

# 3. Update manifest with artifact references
manifest["artifacts"].append(artifact)
manifest_manager.save_manifest(uid, manifest)

# 4. Register in dataset index
# → Updates: datasets/corn_m5/index.yaml
```

**Prediction:**
```python
# 1. Lookup pipeline UID
uid = manifest_manager.get_pipeline_uid("corn_m5", "svm_baseline")

# 2. Load manifest
manifest = manifest_manager.load_manifest(uid)

# 3. Load artifacts
for artifact in manifest["artifacts"]:
    obj = load(artifact, results_dir)
    # → Reads from: artifacts/objects/ab/abc123...pkl
```

**Cleanup:**
```bash
# Delete pipeline (artifacts remain for reuse)
rm -rf results/pipelines/<uid>

# Clean up unused artifacts
python scripts/gc_artifacts.py --dry-run
python scripts/gc_artifacts.py --force  # Actually delete
```

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Target Architecture](#target-architecture)
3. [Implementation Roadmap](#implementation-roadmap)
4. [Detailed Task List](#detailed-task-list)
5. [Testing Strategy](#testing-strategy)
6. [Migration Notes](#migration-notes)

---

## Current State Analysis

### Identified Serialization Sites

#### Pipeline Serialization & Configuration
- **`nirs4all/pipeline/serialization.py`**
  - ❌ `serialize_component()` embeds `_runtime_instance` (live Python objects in JSON)
  - `deserialize_component()` loads from `_runtime_instance`
  - Helper functions: `_changed_kwargs()`, `_resolve_type()`

- **`nirs4all/pipeline/config.py`**
  - `PipelineConfigs._preprocess_steps()` - merges params into standard format
  - ❌ `PipelineConfigs.serializable_steps()` - strips `_runtime_instance` but allows it
  - `PipelineConfigs.get_hash()` - generates MD5 hash from serialized steps

#### Binary Artifact Storage
- **`nirs4all/pipeline/io.py`** (SimulationSaver)
  - ❌ `save_binary()` - deprecated but still present, uses `pickle.dump`
  - ❌ `save_files()` - accepts arbitrary objects, pickles them with `pickle.HIGHEST_PROTOCOL`
  - `save_json()`, `save_file()` - text/JSON saving (OK)
  - `get_metadata()`, `_save_metadata()` - metadata management

#### Binary Loading
- **`nirs4all/pipeline/binary_loader.py`**
  - ❌ `get_step_binaries()` - loads with `pickle.load()` for `.pkl` files
  - Falls back to `pickle.load()` for unknown file types (unsafe)

#### Pipeline Execution & Orchestration
- **`nirs4all/pipeline/runner.py`**
  - `PipelineRunner._execute_controller()` - calls `controller.execute()` expecting `(context, binaries)`
  - ❌ Returns list of `(filename, bytes)` tuples from controllers
  - Calls `saver.save_files()` with raw binaries
  - ❌ Uses `deserialize_component()` which loads `_runtime_instance`
  - `prepare_replay()` - loads metadata and creates `BinaryLoader`

#### Pipeline History & Bundles
- **`nirs4all/pipeline/history.py`**
  - ❌ `save_fitted_operations()` - `pickle.dump(self.fitted_operations, f)`
  - ❌ `create_pipeline_bundle()` - pickles dataset and fitted operations
  - ❌ `save_pickle()`, `_save_pickle_bundle()`, `_save_zip_bundle()` - heavy pickle usage

#### Controllers (Transformers & Models)
- **`nirs4all/controllers/models/base_model_controller.py`**
  - ❌ `_binarize_model()` - returns `pickle.dumps(model)`
  - ❌ `_load_model_from_binaries()` - uses `pickle.loads(model_binary)`
  - ❌ Returns `binaries.append((f"{model_id}.pkl", self._binarize_model(model)))`

- **`nirs4all/controllers/models/model_controller_helper.py`**
  - ❌ `is_model_serializable()` - uses `pickle.dumps()` to check

- **`nirs4all/controllers/sklearn/op_transformermixin.py`**
  - ❌ `execute()` - `transformer_binary = pickle.dumps(transformer)`
  - ❌ Returns `fitted_transformers.append((f"{new_operator_name}.pkl", transformer_binary))`
  - ❌ Checks for `_runtime_instance` in step config

- **`nirs4all/controllers/sklearn/op_y_transformermixin.py`**
  - ❌ `execute()` - `transformer_binary = pickle.dumps(transformer)`
  - ❌ Returns `fitted_transformers = [(f"y_{operator_name}.pkl", transformer_binary)]`

- **`nirs4all/controllers/dataset/op_resampler.py`**
  - ❌ `execute()` - `resampler_binary = pickle.dumps(resampler)`
  - ❌ Returns `fitted_resamplers.append((f"{new_operator_name}.pkl", resampler_binary))`

#### Utility Functions
- **`nirs4all/utils/shap_analyzer.py`**
  - ❌ `save_results()` - `pickle.dump(results, f)`
  - ❌ `load_results()` - `pickle.load(f)`

- **`nirs4all/utils/model_builder.py`**
  - ❌ `_load_model_from_file()` - uses `pickle.load(f)` for `.pkl` files

#### Data Management (OK - No Changes Needed)
- **`nirs4all/dataset/predictions.py`**
  - ✅ Uses JSON serialization for predictions (numpy arrays → JSON strings)
  - ✅ Uses `hashlib.sha256` for content-based IDs
  - ✅ CSV export via Polars DataFrame

### Problems with Current Approach

1. **Redundant Pickling**: 13+ modules independently use `pickle.dumps/loads`
2. **`_runtime_instance` Pollution**: Pipeline JSON bloated with non-serializable objects
3. **No Framework Awareness**: TensorFlow/PyTorch models pickled instead of using native formats
4. **Unsafe Loading**: Binary loader defaults to `pickle.load()` for unknown types
5. **No Content Hashing**: Binaries saved by step number, not content hash (no deduplication)
6. **Metadata Inconsistency**: Each module manages its own binary metadata differently
7. **Hard to Audit**: Scattered serialization logic makes security review difficult

---

## Target Architecture

### Core Components

```
nirs4all/
├── utils/
│   └── serializer.py          # NEW: Central artifact manager
├── pipeline/
│   ├── manifest_manager.py    # NEW: Manage pipeline manifests and UIDs
│   ├── io.py                  # MODIFIED: Use serializer, remove save_binary
│   ├── binary_loader.py       # MODIFIED: Load via manifest + serializer
│   ├── runner.py              # MODIFIED: Handle artifact metadata + manifests
│   ├── serialization.py       # MODIFIED: Remove _runtime_instance
│   ├── config.py              # MODIFIED: Strict validation
│   └── history.py             # MODIFIED: Artifact references only
└── controllers/
    ├── models/
    │   ├── base_model_controller.py    # MODIFIED: Use serializer
    │   └── model_controller_helper.py  # MODIFIED: Use serializer
    └── sklearn/
        ├── op_transformermixin.py      # MODIFIED: Use serializer
        ├── op_y_transformermixin.py    # MODIFIED: Use serializer
        └── ...
```

### New Filesystem Structure

```
results/
├── artifacts/
│   └── objects/
│       └── <hash[:2]>/              # Git-style sharding (e.g., "ab/")
│           └── <sha256>.<ext>       # Deduplicated binary files
│
├── pipelines/
│   └── <pipeline_uid>/              # UUID for each pipeline
│       └── manifest.yaml            # Single file with ALL pipeline info
│
└── datasets/
    └── <dataset_name>/
        └── index.yaml               # Maps pipeline names → UIDs
```

**Key Benefits**:
- ✅ **Single file per pipeline** - `manifest.yaml` replaces pipeline.json + metadata.json
- ✅ **Easy deletion** - `rm -rf pipelines/<uid>` removes entire pipeline
- ✅ **No database** - Pure YAML/JSON/binary files
- ✅ **Deduplication** - Content-addressed artifacts prevent duplicate storage
- ✅ **Human-readable** - YAML manifests, dataset indexes

### New Artifact Flow

```
Training:
  1. PipelineRunner creates UID for new pipeline
  2. Controller.execute()
     → serializer.persist(obj, artifacts_dir, name)
       → Computes SHA256 hash
       → Writes to: artifacts/objects/<hash[:2]>/<hash>.<ext>
       → Returns: artifact_meta = { hash, name, format, size, path }
     → Returns: (context, [artifact_meta, ...])
  3. Runner collects all artifacts
  4. ManifestManager.create_manifest()
     → Writes: pipelines/<uid>/manifest.yaml
       - Pipeline config
       - Training metadata
       - Artifact references (by hash)
       - Prediction history
  5. ManifestManager.register_in_dataset()
     → Updates: datasets/<dataset>/index.yaml

Prediction:
  1. ManifestManager.get_pipeline_uid(dataset, name)
     → Reads: datasets/<dataset>/index.yaml
  2. ManifestManager.load_manifest(uid)
     → Reads: pipelines/<uid>/manifest.yaml
  3. BinaryLoader created with artifact metadata
  4. BinaryLoader.get_step_binaries(step_id)
     → serializer.load(artifact_meta)
       → Reads from artifacts/objects/<hash[:2]>/<hash>.<ext>
       → Returns loaded object
```

### Manifest YAML Schema

```yaml
# pipelines/<uid>/manifest.yaml

# Pipeline identity
uid: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
name: "svm_baseline"
dataset: "corn_m5"
created_at: "2025-10-13T10:30:00Z"
version: "1.0"

# Pipeline configuration
pipeline:
  steps:
    - class: "sklearn.preprocessing.StandardScaler"
      params: {}
    - class: "sklearn.svm.SVC"
      params: {kernel: "rbf", C: 1.0}

# Training metadata
metadata:
  n_samples: 80
  n_features: 100
  accuracy: 0.92
  duration: 5.2
  # ... other metadata fields

# Artifact references (hash-based, deduplicated)
artifacts:
  - step: 0
    name: "StandardScaler_0"
    hash: "sha256:abc123..."
    format: "sklearn_pickle"
    size: 2048
    path: "../../artifacts/objects/ab/abc123...pkl"

  - step: 1
    name: "SVC_1_model"
    hash: "sha256:def456..."
    format: "sklearn_pickle"
    size: 10240
    path: "../../artifacts/objects/de/def456...pkl"

# Prediction history
predictions:
  - id: "pred_001"
    timestamp: "2025-10-13T11:00:00Z"
    input_hash: "sha256:xyz789..."
    output_hash: "sha256:abc789..."
```

### Dataset Index YAML Schema

```yaml
# datasets/<dataset_name>/index.yaml

dataset: "corn_m5"
created_at: "2025-10-13T09:00:00Z"

pipelines:
  svm_baseline: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
  rf_tuned: "b2c3d4e5-f6a7-8901-bcde-f12345678901"
  nn_deep: "c3d4e5f6-a7b8-9012-cdef-123456789012"
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Create `nirs4all/utils/serializer.py` with core API
- [ ] Create `nirs4all/pipeline/manifest_manager.py` with UID management
- [ ] Add unit tests for serializer and manifest manager
- [ ] Create filesystem structure (artifacts/, pipelines/, datasets/)

### Phase 2: Pipeline Core (Week 2)
- [ ] Update `PipelineRunner` to use UIDs and manifests
- [ ] Update `BinaryLoader` to load via manifest + serializer
- [ ] Remove `_runtime_instance` from `serialization.py`
- [ ] Update `PipelineConfigs` validation
- [ ] Update `SimulationSaver` to use new structure

### Phase 3: Controllers (Week 2-3)
- [ ] Update model controllers to return artifact metadata
- [ ] Update sklearn controllers to return artifact metadata
- [ ] Update dataset controllers to return artifact metadata
- [ ] Remove all direct pickle usage

### Phase 4: Utilities & History (Week 3)
- [ ] Update `pipeline/history.py` to use manifests
- [ ] Update `utils/shap_analyzer.py` to use serializer
- [ ] Update `utils/model_builder.py` to use serializer
- [ ] Remove deprecated code

### Phase 5: Testing & Cleanup (Week 3-4)
- [ ] Integration tests for manifest workflow
- [ ] Add garbage collection script for orphaned artifacts
- [ ] Update documentation
- [ ] Final cleanup and review

---

## Detailed Task List

### ✅ Phase 1: Foundation & Core Serializer

#### 1.1 Create Central Serializer Module
- [ ] **Create `nirs4all/utils/serializer.py`**
  - [ ] Define `ArtifactMeta` TypedDict or dataclass
  - [ ] Implement `persist(obj, artifacts_dir, name, format_hint) -> ArtifactMeta`
  - [ ] Implement `load(artifact_meta, artifacts_dir) -> Any`
  - [ ] Implement `to_bytes(obj, format_hint) -> (bytes, str)`
  - [ ] Implement `from_bytes(data, format) -> Any`
  - [ ] Implement `compute_hash(data) -> str` (SHA256)
  - [ ] Implement `is_serializable(obj) -> bool`
  - [ ] Add framework detection logic:
    - [ ] Detect sklearn objects (BaseEstimator)
    - [ ] Detect TensorFlow models (keras.Model)
    - [ ] Detect PyTorch models (torch.nn.Module)
    - [ ] Detect numpy arrays
    - [ ] Fallback to cloudpickle/pickle
  - [ ] Add format handlers:
    - [ ] `_persist_sklearn(obj, path)` - use cloudpickle or joblib
    - [ ] `_persist_tensorflow(obj, path)` - use model.save() to .keras or SavedModel
    - [ ] `_persist_pytorch(obj, path)` - use torch.save(state_dict)
    - [ ] `_persist_generic(obj, path)` - cloudpickle fallback
    - [ ] Corresponding `_load_*` methods
  - [ ] Add directory/zip support for TF SavedModel format
  - [ ] Error handling and validation

**Example API:**
```python
from typing import Any, Dict, Optional, Union
from pathlib import Path
import hashlib
import cloudpickle

class ArtifactMeta(TypedDict):
    hash: str         # SHA256 hash (without "sha256:" prefix)
    name: str         # Original name
    path: str         # Relative path from results root
    format: str       # 'sklearn_pickle', 'tensorflow_h5', 'pytorch_pt', etc.
    size: int         # Bytes
    saved_at: str     # ISO timestamp

def persist(
    obj: Any,
    artifacts_dir: Union[str, Path],
    name: str,
    format_hint: Optional[str] = None
) -> ArtifactMeta:
    """
    Persist object to content-addressed storage.

    Args:
        obj: Object to persist
        artifacts_dir: Path to results/artifacts/objects/ directory
        name: Original name (for metadata only)
        format_hint: Optional format hint ('sklearn', 'tensorflow', etc.)

    Returns:
        ArtifactMeta with hash, path, format, size
    """
    # 1. Serialize to bytes
    data, format = to_bytes(obj, format_hint)

    # 2. Compute SHA256 hash
    hash_value = compute_hash(data)

    # 3. Determine extension
    ext = _format_to_extension(format)

    # 4. Create sharded path: artifacts/objects/<hash[:2]>/<hash>.<ext>
    shard_dir = artifacts_dir / hash_value[:2]
    shard_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = shard_dir / f"{hash_value}.{ext}"

    # 5. Write file (if not exists - deduplication)
    if not artifact_path.exists():
        artifact_path.write_bytes(data)

    # 6. Return metadata
    return {
        "hash": f"sha256:{hash_value}",
        "name": name,
        "path": f"../../artifacts/objects/{hash_value[:2]}/{hash_value}.{ext}",
        "format": format,
        "size": len(data),
        "saved_at": datetime.now(timezone.utc).isoformat()
    }

def load(
    artifact_meta: ArtifactMeta,
    results_root: Union[str, Path]
) -> Any:
    """Load object from artifact metadata."""
    # Resolve path relative to results root
    artifact_path = Path(results_root) / "artifacts" / "objects" / artifact_meta["hash"].split(":")[-1][:2] / Path(artifact_meta["path"]).name

    data = artifact_path.read_bytes()
    return from_bytes(data, artifact_meta["format"])
```

#### 1.2 Create Manifest Manager
- [ ] **Create `nirs4all/pipeline/manifest_manager.py`**
  - [ ] Implement `ManifestManager` class
  - [ ] Implement `create_pipeline(name, dataset, config) -> uid`
  - [ ] Implement `save_manifest(uid, manifest_data)`
  - [ ] Implement `load_manifest(uid) -> dict`
  - [ ] Implement `update_manifest(uid, updates)`
  - [ ] Implement `delete_pipeline(uid, dataset)`
  - [ ] Implement `get_pipeline_uid(dataset, name) -> uid`
  - [ ] Implement `list_pipelines(dataset) -> List[str]`
  - [ ] Implement `register_in_dataset(dataset, name, uid)`
  - [ ] Implement `unregister_from_dataset(dataset, name)`

**Example API:**
```python
from pathlib import Path
import uuid
import yaml
from typing import Dict, List, Optional

class ManifestManager:
    """Manage pipeline manifests and dataset indexes using YAML."""

    def __init__(self, results_dir: Union[str, Path]):
        self.results_dir = Path(results_dir)
        self.artifacts_dir = self.results_dir / "artifacts" / "objects"
        self.pipelines_dir = self.results_dir / "pipelines"
        self.datasets_dir = self.results_dir / "datasets"

        # Ensure directories exist
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.pipelines_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    def create_pipeline(
        self,
        name: str,
        dataset: str,
        pipeline_config: dict
    ) -> str:
        """Create new pipeline, returns UID."""
        uid = str(uuid.uuid4())
        pipeline_dir = self.pipelines_dir / uid
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "uid": uid,
            "name": name,
            "dataset": dataset,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
            "pipeline": pipeline_config,
            "metadata": {},
            "artifacts": [],
            "predictions": []
        }

        self.save_manifest(uid, manifest)
        self.register_in_dataset(dataset, name, uid)

        return uid

    def save_manifest(self, uid: str, manifest: dict):
        """Save manifest YAML file."""
        manifest_path = self.pipelines_dir / uid / "manifest.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    def load_manifest(self, uid: str) -> dict:
        """Load manifest YAML file."""
        manifest_path = self.pipelines_dir / uid / "manifest.yaml"
        with open(manifest_path, "r") as f:
            return yaml.safe_load(f)

    def get_pipeline_uid(self, dataset: str, pipeline_name: str) -> Optional[str]:
        """Get pipeline UID from dataset index."""
        index_path = self.datasets_dir / dataset / "index.yaml"
        if not index_path.exists():
            return None

        with open(index_path, "r") as f:
            index = yaml.safe_load(f)

        return index.get("pipelines", {}).get(pipeline_name)

    def register_in_dataset(self, dataset: str, pipeline_name: str, uid: str):
        """Register pipeline in dataset index."""
        dataset_dir = self.datasets_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        index_path = dataset_dir / "index.yaml"

        if index_path.exists():
            with open(index_path, "r") as f:
                index = yaml.safe_load(f)
        else:
            index = {
                "dataset": dataset,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "pipelines": {}
            }

        index["pipelines"][pipeline_name] = uid

        with open(index_path, "w") as f:
            yaml.dump(index, f, default_flow_style=False)

    def delete_pipeline(self, uid: str):
        """Delete pipeline manifest (artifacts remain for GC)."""
        manifest = self.load_manifest(uid)
        dataset = manifest["dataset"]
        name = manifest["name"]

        # Remove from dataset index
        self.unregister_from_dataset(dataset, name)

        # Delete pipeline folder
        pipeline_dir = self.pipelines_dir / uid
        shutil.rmtree(pipeline_dir)
```

#### 1.3 Unit Tests for Serializer & Manifest Manager
- [ ] **Create `tests/test_serializer.py`**
  - [ ] Test `compute_hash()` determinism
  - [ ] Test `persist()` + `load()` round-trip for:
    - [ ] sklearn StandardScaler
    - [ ] sklearn RandomForestClassifier
    - [ ] numpy arrays
    - [ ] simple dicts/lists
    - [ ] Custom objects (if supported)
  - [ ] Test framework detection logic
  - [ ] Test error cases:
    - [ ] Unserializable objects
    - [ ] Invalid paths
    - [ ] Corrupted artifacts
  - [ ] Test `is_serializable()` coverage
  - [ ] Test content deduplication (same object → same hash)

#### 1.3 Update SimulationSaver
- [ ] **Modify `nirs4all/pipeline/io.py`**
  - [ ] Add `persist_artifact(step_number, substep_number, name, obj) -> ArtifactMeta`
    ```python
    def persist_artifact(
        self,
        step_number: int,
        substep_number: int,
        name: str,
        obj: Any
    ) -> ArtifactMeta:
        """Persist artifact via serializer and update metadata."""
        from nirs4all.utils.serializer import persist

        artifact = persist(
            obj,
            base_dir=self.current_path,
            name=name,
            format_hint=None
        )

        # Update metadata
        step_id = str(step_number)
        if step_id not in self._metadata["binaries"]:
            self._metadata["binaries"][step_id] = []

        self._metadata["binaries"][step_id].append(artifact)
        self._save_metadata()

        return artifact
    ```
  - [ ] **Delete `save_binary()` method entirely**
  - [ ] Modify `save_files()` to detect objects vs bytes:
    - [ ] If bytes/str → write directly
    - [ ] If object → call `persist_artifact()`
  - [ ] Update `_metadata` schema to use artifact format
  - [ ] Update `get_metadata()` return type
  - [ ] Add validation for `binaries` structure

- [ ] **Update Tests**
  - [ ] Test `persist_artifact()` creates correct metadata
  - [ ] Test artifact deduplication (same content → same id)
  - [ ] Test metadata.json schema compliance

### ✅ Phase 2: Pipeline Core Updates

#### 2.1 Update PipelineRunner to use Manifests
- [ ] **Modify `nirs4all/pipeline/runner.py`**
  - [ ] Add `ManifestManager` integration in `__init__`:
    ```python
    from nirs4all.pipeline.manifest_manager import ManifestManager

    def __init__(self, ...):
        # ... existing code ...
        self.manifest_manager = ManifestManager(results_dir)
        self.pipeline_uid = None  # Set during training
    ```
  - [ ] Update training workflow to create pipeline on start:
    ```python
    def run(self, ...):
        if self.mode == "train":
            # Create new pipeline with UID
            self.pipeline_uid = self.manifest_manager.create_pipeline(
                name=pipeline_name,
                dataset=dataset_name,
                pipeline_config=self.pipeline_config
            )
    ```
  - [ ] Update `_execute_controller()` to collect artifact metadata:
    ```python
    # Controllers now return (context, artifacts: List[ArtifactMeta])
    context, artifacts = controller.execute(...)

    if self.mode == "train" and artifacts:
        # Add artifacts to manifest
        manifest = self.manifest_manager.load_manifest(self.pipeline_uid)
        manifest["artifacts"].extend(artifacts)
        self.manifest_manager.save_manifest(self.pipeline_uid, manifest)
    ```
  - [ ] Remove `_runtime_instance` checks in `run_step()`:
    - [ ] Delete: `if '_runtime_instance' in step[key]: operator = step[key]['_runtime_instance']`
    - [ ] Delete: `if '_runtime_instance' in step: operator = step['_runtime_instance']`
  - [ ] Update `prepare_replay()` to load from manifest:
    ```python
    def prepare_replay(self, dataset_name, pipeline_name):
        # Get pipeline UID from dataset index
        uid = self.manifest_manager.get_pipeline_uid(dataset_name, pipeline_name)

        # Load manifest
        manifest = self.manifest_manager.load_manifest(uid)

        # Create BinaryLoader with artifacts from manifest
        self.binary_loader = BinaryLoader(manifest["artifacts"], self.results_dir)
    ```
  - [ ] Update metadata saving to include UID
  - [ ] Update docstrings for new contract

#### 2.2 Update BinaryLoader to use Manifest Artifacts
- [ ] **Modify `nirs4all/pipeline/binary_loader.py`**
  - [ ] Change `__init__` to accept artifact list from manifest:
    ```python
    def __init__(self, artifacts: List[ArtifactMeta], results_dir: Path):
        """Initialize with artifacts from manifest."""
        self.results_dir = results_dir
        self.artifacts_by_step = {}

        # Group artifacts by step number
        for artifact in artifacts:
            step = artifact.get("step")
            if step not in self.artifacts_by_step:
                self.artifacts_by_step[step] = []
            self.artifacts_by_step[step].append(artifact)
    ```
  - [ ] Update `get_step_binaries(step_id)` to use serializer:
    ```python
    def get_step_binaries(self, step_id: int) -> List[Tuple[str, Any]]:
        if step_id not in self.artifacts_by_step:
            return []

        from nirs4all.utils.serializer import load

        artifacts = self.artifacts_by_step[step_id]
        loaded = []

        for artifact in artifacts:
            try:
                obj = load(artifact, self.results_dir)
                loaded.append((artifact["name"], obj))
            except Exception as e:
                warnings.warn(f"Failed to load artifact {artifact['hash']}: {e}")

        return loaded
    ```
  - [ ] **Remove all `pickle.load()` calls**
  - [ ] Remove fallback to `pickle.load()` for unknown types
  - [ ] Add explicit error for unsupported formats
  - [ ] Update `from_pipeline_path()` classmethod (or deprecate)

- [ ] **Update Tests**
  - [ ] Test loading artifacts with different formats
  - [ ] Test error handling for missing/corrupted artifacts
  - [ ] Test step grouping logic

        return loaded
    ```
  - [ ] **Remove all `pickle.load()` calls**
  - [ ] Remove fallback to `pickle.load()` for unknown types
  - [ ] Add explicit error for unsupported formats
  - [ ] Update `from_pipeline_path()` classmethod

- [ ] **Update Tests**
  - [ ] Test loading artifacts with different formats
  - [ ] Test error handling for missing/corrupted artifacts
  - [ ] Test backward compatibility handling (if any)

#### 2.3 Remove `_runtime_instance` from Serialization
- [ ] **Modify `nirs4all/pipeline/serialization.py`**
  - [ ] Update `serialize_component()` to raise error on runtime instances:
    ```python
    def serialize_component(obj: Any, include_runtime: bool = False) -> Any:
        # ... existing trivial cases ...

        if include_runtime:
            raise TypeError(
                "Runtime instance embedding is no longer supported. "
                "Use class + params or persist as artifact separately. "
                f"Got: {type(obj)}"
            )

        # Remove all `def_serialized["_runtime_instance"] = obj` lines
        # ... rest of logic ...
    ```
  - [ ] Update `deserialize_component()` to ignore `_runtime_instance`:
    ```python
    # Simply skip if present (for graceful handling)
    if isinstance(blob, dict) and "_runtime_instance" in blob:
        warnings.warn("Found _runtime_instance in config (deprecated)")
        # Continue with class/function deserialization
    ```
  - [ ] Update docstrings to document new behavior
  - [ ] Add clear error messages for unsupported patterns

- [ ] **Modify `nirs4all/pipeline/config.py`**
  - [ ] Update `PipelineConfigs.__init__` to use `include_runtime=False`:
    ```python
    self.steps = serialize_component(self.steps, include_runtime=False)
    ```
  - [ ] Update `serializable_steps()` to be no-op (already clean)
  - [ ] Add validation for artifact references format

- [ ] **Update Tests**
  - [ ] Test that runtime instances cause errors
  - [ ] Test that configs remain valid JSON
  - [ ] Test deserialization ignores old `_runtime_instance`

#### 2.4 Update Controller Base Contract
- [ ] **Document new controller contract**
  - [ ] Create `docs/CONTROLLER_CONTRACT.md` explaining:
    - New return type: `(context, List[ArtifactMeta])`
    - When to use `serializer.persist()` vs returning bytes
    - Artifact naming conventions
  - [ ] Update controller base class docstring

### ✅ Phase 3: Update All Controllers

#### 3.1 Update Model Controllers
- [ ] **Modify `nirs4all/controllers/models/base_model_controller.py`**
  - [ ] Replace `_binarize_model()` with `_persist_model()`:
    ```python
    def _persist_model(
        self,
        model: Any,
        name: str,
        step: int,
        results_dir: Path
    ) -> ArtifactMeta:
        """Persist model and return artifact metadata."""
        from nirs4all.utils.serializer import persist

        artifacts_dir = results_dir / "artifacts" / "objects"

        artifact = persist(
            model,
            artifacts_dir=artifacts_dir,
            name=name,
            format_hint=None  # Auto-detect framework
        )

        # Add step number for manifest
        artifact["step"] = step

        return artifact
    ```
  - [ ] Replace `_load_model_from_binaries()`:
    ```python
    def _load_model_from_binaries(
        self,
        loaded_binaries: List[Tuple[str, Any]]
    ) -> Any:
        """Extract model from loaded binaries."""
        # loaded_binaries already contains deserialized objects
        for name, obj in loaded_binaries:
            if name.endswith('_model') or 'model' in name.lower():
                return obj
        raise ValueError("No model found in loaded binaries")
    ```
  - [ ] Update `train()` to return artifact metadata:
    ```python
    artifact = self._persist_model(model, model_id, step, runner.results_dir)
    artifacts.append(artifact)  # Return List[ArtifactMeta]
    ```
  - [ ] Remove `import pickle` statements
  - [ ] Update return type annotations
  - [ ] Update docstrings

- [ ] **Modify `nirs4all/controllers/models/model_controller_helper.py`**
  - [ ] Replace `is_model_serializable()`:
    ```python
    def is_model_serializable(self, model: Any) -> bool:
        """Check if model can be serialized."""
        from nirs4all.utils.serializer import is_serializable
        return is_serializable(model)
    ```
  - [ ] Remove `import pickle`
  - [ ] Update tests

#### 3.2 Update Sklearn Controllers
- [ ] **Modify `nirs4all/controllers/sklearn/op_transformermixin.py`**
  - [ ] Remove `_runtime_instance` check:
    ```python
    # DELETE these lines:
    # if isinstance(model_obj, dict) and '_runtime_instance' in model_obj:
    #     model_obj = model_obj['_runtime_instance']
    ```
  - [ ] Replace pickling in `execute()`:
    ```python
    from nirs4all.utils.serializer import persist

    # OLD CODE (DELETE):
    # transformer_binary = pickle.dumps(transformer)
    # fitted_transformers.append((f"{new_operator_name}.pkl", transformer_binary))

    # NEW CODE:
    artifacts_dir = runner.results_dir / "artifacts" / "objects"
    artifact = persist(
        transformer,
        artifacts_dir=artifacts_dir,
        name=new_operator_name,
        format_hint='sklearn'
    )
    artifact["step"] = runner.step_number  # Add step info
    fitted_transformers.append(artifact)

    # Return List[ArtifactMeta] instead of List[(name, bytes)]
    return context, fitted_transformers
    ```
  - [ ] Update return type annotation: `Tuple[Dict, List[ArtifactMeta]]`
  - [ ] Remove `import pickle`
  - [ ] Update tests

- [ ] **Modify `nirs4all/controllers/sklearn/op_y_transformermixin.py`**
  - [ ] Same changes as `op_transformermixin.py`
  - [ ] Update `execute()` method
  - [ ] Remove pickle usage
  - [ ] Update tests

#### 3.3 Update Dataset Controllers
- [ ] **Modify `nirs4all/controllers/dataset/op_resampler.py`**
  - [ ] Replace pickling in `execute()`:
    ```python
    from nirs4all.utils.serializer import persist

    # OLD CODE (DELETE):
    # resampler_binary = pickle.dumps(resampler)
    # fitted_resamplers.append((f"{new_operator_name}.pkl", resampler_binary))

    # NEW CODE:
    artifacts_dir = runner.results_dir / "artifacts" / "objects"
    artifact = persist(
        resampler,
        artifacts_dir=artifacts_dir,
        name=new_operator_name,
        format_hint='sklearn'
    )
    artifact["step"] = runner.step_number
    fitted_resamplers.append(artifact)
    ```
  - [ ] Remove `import pickle`
  - [ ] Update return type: `Tuple[Dict, List[ArtifactMeta]]`
  - [ ] Update tests

#### 3.4 Update Other Controllers (if any)
- [ ] Scan `nirs4all/controllers/` for other pickle usage
- [ ] Apply same pattern to any remaining controllers
- [ ] Verify all controllers return `List[ArtifactMeta]`

### ✅ Phase 4: Update Utilities & History

#### 4.1 Update Pipeline History
- [ ] **Modify `nirs4all/pipeline/history.py`**
  - [ ] Replace `save_fitted_operations()`:
    ```python
    def save_fitted_operations(
        self,
        filepath: Union[str, Path],
        base_dir: Union[str, Path]
    ):
        """Save fitted operations as artifacts and record IDs."""
        from nirs4all.utils.serializer import persist

        artifact_refs = {}
        for step_id, operation in self.fitted_operations.items():
            artifact = persist(
                operation,
                base_dir=base_dir,
                name=f"fitted_op_{step_id}",
                format_hint=None
            )
            artifact_refs[step_id] = artifact

        # Save artifact references as JSON
        with open(filepath, 'w') as f:
            json.dump(artifact_refs, f, indent=2)
    ```
  - [ ] Update `create_pipeline_bundle()` to use artifacts
  - [ ] Update `load_pipeline_bundle()` to load via serializer
  - [ ] Remove all `pickle.dump/load` calls
  - [ ] Update `save_pickle()` to use artifact system
  - [ ] Remove `_save_pickle_bundle()` or convert to artifacts
  - [ ] Update docstrings

- [ ] **Update Tests**
  - [ ] Test bundle creation with artifacts
  - [ ] Test bundle loading
  - [ ] Test backward compatibility (if needed)

#### 4.2 Update SHAP Analyzer
- [ ] **Modify `nirs4all/utils/shap_analyzer.py`**
  - [ ] Replace `save_results()`:
    ```python
    def save_results(
        self,
        results: Dict[str, Any],
        output_path: str,
        base_dir: Optional[str] = None
    ):
        """Save SHAP results as artifact."""
        from nirs4all.utils.serializer import persist

        artifact = persist(
            results,
            base_dir=base_dir or Path(output_path).parent,
            name=Path(output_path).stem,
            format_hint='pickle'
        )
        print(f"💾 Results saved to: {artifact['path']}")
        return artifact
    ```
  - [ ] Replace `load_results()`:
    ```python
    @staticmethod
    def load_results(input_path: str) -> Dict[str, Any]:
        """Load SHAP results from artifact."""
        from nirs4all.utils.serializer import load

        # If input_path is artifact metadata, load directly
        # Otherwise, construct metadata from path
        # ... implementation ...
    ```
  - [ ] Remove `import pickle`
  - [ ] Update tests

#### 4.3 Update Model Builder
- [ ] **Modify `nirs4all/utils/model_builder.py`**
  - [ ] Replace `_load_model_from_file()` pickle section:
    ```python
    elif ext == '.pkl':
        from nirs4all.utils.serializer import load

        # Construct artifact metadata from file
        artifact = {
            "path": model_path,
            "format": "pickle",
            # ... other fields ...
        }
        model = load(artifact, base_dir=Path(model_path).parent)
        return model
    ```
  - [ ] Or deprecate `.pkl` loading entirely (since BC not required)
  - [ ] Add warning for legacy `.pkl` files
  - [ ] Update tests

### ✅ Phase 5: Testing, Documentation & Cleanup

#### 5.1 Integration Tests
- [ ] **Create `tests/integration/test_artifact_workflow.py`**
  - [ ] Test full training pipeline:
    - [ ] Train with sklearn transformer
    - [ ] Verify artifacts created in `binaries/`
    - [ ] Verify `metadata.json` schema
    - [ ] Verify artifact content hashing
  - [ ] Test prediction pipeline:
    - [ ] Load from saved artifacts
    - [ ] Verify predictions match training run
  - [ ] Test with different frameworks:
    - [ ] sklearn models
    - [ ] TensorFlow models (if available)
    - [ ] PyTorch models (if available)
  - [ ] Test artifact deduplication
  - [ ] Test error recovery

#### 5.2 Create Garbage Collection Script
- [ ] **Create `scripts/gc_artifacts.py`**
  - [ ] Implement `find_orphaned_artifacts()`:
    ```python
    def find_orphaned_artifacts(results_dir: Path) -> Set[str]:
        """Find artifacts not referenced by any pipeline manifest."""
        artifacts_dir = results_dir / "artifacts" / "objects"
        pipelines_dir = results_dir / "pipelines"

        # Collect all artifact hashes from filesystem
        all_hashes = set()
        for hash_prefix_dir in artifacts_dir.iterdir():
            if hash_prefix_dir.is_dir():
                for artifact_file in hash_prefix_dir.iterdir():
                    # Extract hash from filename (remove extension)
                    hash_value = artifact_file.stem
                    all_hashes.add(hash_value)

        # Collect referenced hashes from all manifests
        referenced_hashes = set()
        for pipeline_dir in pipelines_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue

            manifest_path = pipeline_dir / "manifest.yaml"
            if manifest_path.exists():
                manifest = yaml.safe_load(manifest_path.read_text())
                for artifact in manifest.get("artifacts", []):
                    # Extract hash from "sha256:abc123..." format
                    hash_value = artifact["hash"].split(":")[-1]
                    referenced_hashes.add(hash_value)

        return all_hashes - referenced_hashes

    def cleanup_orphaned_artifacts(results_dir: Path, dry_run: bool = True):
        """Remove orphaned artifacts."""
        orphans = find_orphaned_artifacts(results_dir)

        print(f"Found {len(orphans)} orphaned artifacts")
        total_size = 0

        for hash_value in orphans:
            # Find file with this hash (may have different extensions)
            artifact_dir = results_dir / "artifacts" / "objects" / hash_value[:2]
            for artifact_file in artifact_dir.glob(f"{hash_value}.*"):
                size = artifact_file.stat().st_size
                total_size += size

                if dry_run:
                    print(f"Would delete: {artifact_file} ({size} bytes)")
                else:
                    artifact_file.unlink()
                    print(f"Deleted: {artifact_file} ({size} bytes)")

        print(f"Total space: {total_size / 1024 / 1024:.2f} MB")
        if dry_run:
            print("\nRun with --no-dry-run to actually delete files")
    ```
  - [ ] Add CLI interface:
    ```python
    if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser(description="Clean up orphaned artifacts")
        parser.add_argument("--results-dir", default="./results", help="Path to results directory")
        parser.add_argument("--no-dry-run", action="store_true", help="Actually delete files")

        args = parser.parse_args()
        cleanup_orphaned_artifacts(Path(args.results_dir), dry_run=not args.no_dry_run)
    ```

- [ ] **Add to CLI commands**
  - [ ] Add `nirs4all gc-artifacts` command
  - [ ] Add `--dry-run` flag (default)
  - [ ] Add `--force` flag to actually delete
  - [ ] Show space savings estimate

#### 5.3 Update Documentation
- [ ] **Update `docs/` directory**
  - [ ] Create `docs/SERIALIZATION_ARCHITECTURE.md`
  - [ ] Update `docs/NIRS4ALL_INTEGRATION_SUMMARY.md`
  - [ ] Update `docs/NIRS4ALL_FORMAT.md` with artifact schema
  - [ ] Add examples of artifact usage
  - [ ] Document breaking changes
  - [ ] Add migration guide (optional)

- [ ] **Update README.md**
  - [ ] Add note about breaking changes
  - [ ] Update installation requirements (if needed)
  - [ ] Add artifact management examples

#### 5.3 Update Documentation
- [ ] **Update `docs/` directory**
  - [ ] Create `docs/MANIFEST_ARCHITECTURE.md` documenting:
    - [ ] Filesystem structure (artifacts/, pipelines/, datasets/)
    - [ ] Manifest YAML schema
    - [ ] Dataset index YAML schema
    - [ ] Content-addressed storage benefits
    - [ ] UID-based pipeline management
  - [ ] Update `docs/NIRS4ALL_INTEGRATION_SUMMARY.md`
  - [ ] Update `docs/NIRS4ALL_FORMAT.md` with manifest format
  - [ ] Add examples of manifest usage
  - [ ] Document breaking changes
  - [ ] Add CLI commands documentation

- [ ] **Update README.md**
  - [ ] Add note about breaking changes
  - [ ] Update installation requirements (if needed)
  - [ ] Add artifact management examples
  - [ ] Add garbage collection instructions

- [ ] **Update Docstrings**
  - [ ] Serializer module comprehensive docstrings
  - [ ] ManifestManager comprehensive docstrings
  - [ ] Controller execute() method contracts
  - [ ] Runner artifact handling

#### 5.4 Code Cleanup
- [ ] **Remove deprecated code**
  - [ ] Search for `import pickle` across codebase
  - [ ] Verify no direct pickle usage remains
  - [ ] Remove `save_binary()` from io.py
  - [ ] Remove commented-out code
  - [ ] Remove unused imports

- [ ] **Linting & Formatting**
  - [ ] Run black formatter on modified files
  - [ ] Run flake8/pylint checks
  - [ ] Fix type hints
  - [ ] Update `.gitignore` if needed

- [ ] **Verify No Regressions**
  - [ ] Run full test suite
  - [ ] Check CLI commands still work
  - [ ] Verify examples still run
  - [ ] Check webapp integration (if applicable)

#### 5.5 Final Review
- [ ] Code review checklist:
  - [ ] No `import pickle` except in serializer
  - [ ] No `_runtime_instance` in code
  - [ ] All controllers use artifact metadata with hashes
  - [ ] All pipelines have UIDs and manifests
  - [ ] Dataset indexes properly maintained
  - [ ] All tests passing
  - [ ] Documentation updated
  - [ ] Breaking changes documented
  - [ ] Error messages helpful
  - [ ] Performance acceptable
  - [ ] GC script works correctly
  - [ ] All controllers use artifact metadata
  - [ ] All tests passing
  - [ ] Documentation updated
  - [ ] Breaking changes documented
  - [ ] Error messages helpful
  - [ ] Performance acceptable

---

## Testing Strategy

### Unit Tests
```python
# tests/test_serializer.py
def test_persist_load_roundtrip_sklearn():
    from sklearn.preprocessing import StandardScaler
    from nirs4all.utils.serializer import persist, load

    scaler = StandardScaler()
    scaler.fit([[0], [1]])

    artifact = persist(scaler, base_dir=tmp_path, name="test_scaler")
    loaded = load(artifact, base_dir=tmp_path)

    assert loaded.mean_ == scaler.mean_
    assert artifact["format"] == "pickle"
    assert len(artifact["id"]) == 64  # SHA256

def test_content_hash_determinism():
    # Same object should produce same hash
    pass

def test_framework_detection():
    # Test sklearn, TF, PyTorch, numpy detection
    pass
```

### Integration Tests
```python
# tests/integration/test_pipeline_artifacts.py
def test_train_save_load_predict():
    # Train pipeline → save artifacts
    # Load artifacts → predict
    # Compare predictions
    pass

def test_metadata_schema():
    # Verify metadata.json structure
    pass
```

### Performance Tests
```python
def test_artifact_storage_overhead():
    # Measure space/time overhead vs raw pickle
    pass
```

---

## Migration Notes

### For Users (Breaking Changes)

**⚠️ Important**: Existing pipelines saved before this refactoring will NOT work.

**Action Required**:
1. Re-run training for all pipelines
2. Old `results/<dataset>/<pipeline>/` structure will be replaced with:
   - `results/pipelines/<uid>/manifest.yaml`
   - `results/artifacts/objects/<hash[:2]>/<hash>.<ext>`
   - `results/datasets/<dataset>/index.yaml`
3. Pipeline JSON files will no longer embed `_runtime_instance`
4. All fitted models/transformers stored as content-addressed artifacts

**What Changed**:
- **Filesystem structure**: UID-based pipelines instead of dataset/name hierarchy
- **Single manifest**: `manifest.yaml` replaces `pipeline.json` + `metadata.json`
- **Content-addressed storage**: Artifacts deduplicated by SHA256 hash
- **Dataset indexes**: YAML files map pipeline names → UIDs
- **Easy deletion**: `rm -rf pipelines/<uid>` removes entire pipeline
- **No database**: Pure YAML/JSON/binary files

**Benefits**:
- ✅ Simpler codebase (no redundant pickle code)
- ✅ Deduplication (identical models stored once)
- ✅ Easy cleanup (delete folder = done)
- ✅ Efficient storage (single manifest vs multiple files)
- ✅ Content verification (SHA256 hashes)
- ✅ Human-readable manifests (YAML)

### Example: Old vs New Structure

**Old Structure:**
```
results/
└── corn_m5/
    └── svm_baseline/
        ├── pipeline.json
        ├── metadata.json
        └── binaries/
            ├── 0_StandardScaler.pkl
            └── 1_SVC_model.pkl
```

**New Structure:**
```
results/
├── artifacts/
│   └── objects/
│       ├── ab/
│       │   └── abc123...pkl  # StandardScaler (deduplicated)
│       └── de/
│           └── def456...pkl  # SVC model
├── pipelines/
│   └── a1b2c3d4-e5f6-7890-abcd-ef1234567890/
│       └── manifest.yaml  # All pipeline info in one file
└── datasets/
    └── corn_m5/
        └── index.yaml  # Maps "svm_baseline" → UID
```

### Migration Script (Not Recommended)

**Best practice**: Just re-run training with new code.

If you absolutely need to migrate existing pipelines (not recommended since breaking BC is acceptable):

```python
# scripts/migrate_old_pipelines.py
"""
Migrate old pipeline structure to new manifest-based structure.
WARNING: This is complex and error-prone. Re-running training is better.
"""
import json
import yaml
import shutil
from pathlib import Path
from nirs4all.utils.serializer import persist, compute_hash

def migrate_pipeline(old_path: Path, results_dir: Path):
    # Load old pipeline.json and metadata.json
    # Create new UID
    # Persist binaries as content-addressed artifacts
    # Create manifest.yaml
    # Update dataset index
    pass
```

---

## Progress Tracking

### Week 1 Progress
- [ ] Phase 1 Complete: Foundation & Core Serializer
  - [ ] serializer.py implemented and tested
  - [ ] SimulationSaver updated

### Week 2 Progress
- [ ] Phase 2 Complete: Pipeline Core Updates
  - [ ] PipelineRunner updated
  - [ ] BinaryLoader updated
  - [ ] _runtime_instance removed

### Week 3 Progress
- [ ] Phase 3 Complete: All Controllers Updated
  - [ ] Model controllers
  - [ ] Sklearn controllers
  - [ ] Dataset controllers
- [ ] Phase 4 Complete: Utilities & History
  - [ ] Pipeline history
  - [ ] SHAP analyzer
  - [ ] Model builder

### Week 4 Progress
- [ ] Phase 5 Complete: Testing & Documentation
  - [ ] All tests passing
  - [ ] Documentation complete
  - [ ] Code cleanup done
  - [ ] Ready for merge

---

## Success Criteria

✅ **Definition of Done**:
1. No `import pickle` outside of `nirs4all/utils/serializer.py`
2. No `_runtime_instance` in codebase
3. All controllers return `List[ArtifactMeta]`
4. All pipelines use UID-based structure with `manifest.yaml`
5. Dataset indexes properly maintained in `datasets/<name>/index.yaml`
6. Content-addressed artifacts in `artifacts/objects/<hash[:2]>/<hash>.<ext>`
7. All tests passing (unit + integration)
8. Documentation updated with new architecture
9. Pipeline train/predict workflows functional
10. Performance acceptable (< 10% overhead vs current)
11. GC script correctly identifies and removes orphaned artifacts
12. Easy per-pipeline deletion works (`rm -rf pipelines/<uid>`)

---

## CLI Commands Reference

### Pipeline Management
```bash
# List all pipelines for a dataset
nirs4all list-pipelines --dataset corn_m5

# Show pipeline details
nirs4all show-pipeline <uid>

# Delete pipeline (keeps artifacts for GC)
nirs4all delete-pipeline <uid>
# Or delete by name
nirs4all delete-pipeline --dataset corn_m5 --name svm_baseline
```

### Artifact Management
```bash
# Show artifact statistics
nirs4all artifacts stats

# List orphaned artifacts
nirs4all gc-artifacts --dry-run

# Clean up orphaned artifacts (actual deletion)
nirs4all gc-artifacts --force

# Show artifacts for a pipeline
nirs4all show-artifacts <uid>
```

### Migration Tools
```bash
# Verify manifest integrity
nirs4all verify-manifests

# Rebuild dataset indexes
nirs4all rebuild-indexes
```

---

## Risk Mitigation

### Risks
1. **TensorFlow SavedModel complexity** - May require directory zipping
2. **PyTorch state dict** - Need to store architecture separately
3. **Large model files** - Storage space concerns
4. **Performance overhead** - Content hashing may slow saves

### Mitigations
1. Implement lazy loading for large artifacts
2. Add compression for large binaries
3. Cache artifact hashes
4. Profile and optimize hot paths
5. Add progress indicators for large operations

---

## Notes

- This is a **breaking change** by design - embrace it for cleaner code
- Focus on correct implementation over backward compatibility
- Test thoroughly with real datasets before merging
- Consider adding `--legacy-mode` flag if absolutely needed (not recommended)
- Use this opportunity to improve error messages and logging

---

## Questions / Decisions Needed

- [ ] Decide on compression strategy for large artifacts
- [ ] Decide on artifact retention/cleanup policy
- [ ] Decide on TensorFlow format (SavedModel vs .keras vs .h5)
- [ ] Decide on PyTorch format (state_dict only vs full model)
- [ ] Decide if cloudpickle is acceptable or use joblib for sklearn
- [ ] Decide on artifact versioning strategy (if needed)

---

**Last Updated**: October 13, 2025
**Status**: Ready for implementation
**Estimated Effort**: 3-4 weeks (1 engineer)
