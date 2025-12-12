# Artifacts System Refactoring - Specifications Document

## 1. Refactoring Objectives

### 1.1 Primary Goals

1. **Deterministic Identification**: Replace iterator-based naming with stable, content-derived artifact IDs that are reproducible across runs and replay operations.

2. **Branch-Aware Storage**: Native support for pipeline branching where artifacts can be shared (pre-branch) or branch-specific.

3. **Stacking Compatibility**: Enable meta-models to reference source model artifacts with clear dependency graphs.

4. **Replay & Transfer**: Reliable artifact loading during predict mode, transfer learning, and pipeline replay—without relying on run-order reconstruction.

5. **Maintenance Simplification**: Eliminate legacy backward-compatibility code paths and reduce architectural complexity.

### 1.2 Secondary Goals

- **Global deduplication**: Same binary content stored once per dataset, across all runs
- **Cleanup utilities**: Functions to delete orphaned artifacts not referenced by any manifest
- Improve debugging with meaningful artifact metadata
- Enable future features: remote storage, artifact search

---

## 2. Current State Analysis

### 2.1 Architecture Overview

The current artifact management system consists of:

| Component | Location | Role |
|-----------|----------|------|
| `ManifestManager` | `pipeline/storage/manifest_manager.py` | Manages YAML manifests with pipeline config, artifacts list, predictions |
| `ArtifactManager` | `pipeline/storage/artifacts/manager.py` | Persists binary artifacts (models, transformers) |
| `artifact_persistence` | `pipeline/storage/artifacts/artifact_persistence.py` | Framework-aware serialization/deserialization |
| `BinaryLoader` | `pipeline/storage/artifacts/binary_loader.py` | Loads artifacts by step number for predict/explain mode |
| `SimulationSaver` | `pipeline/storage/io.py` | Manages run directory structure, file I/O |

### 2.2 Current Filesystem Structure

```
workspace/runs/YYYY-MM-DD_dataset/
├── _binaries/                          # Artifacts mixed with runs (PROBLEM)
│   ├── StandardScaler_a1b2c3.pkl
│   ├── PLSRegression_d4e5f6.joblib
│   └── ...
├── 0001_pls_abc123/
│   ├── manifest.yaml                   # Pipeline manifest with artifact refs
│   ├── metrics.json
│   └── predictions.csv
├── 0002_rf_def456/
└── predictions.json                    # Global predictions database
```

**Problems with current structure:**
- Artifacts duplicated across runs on same dataset
- No global deduplication
- Run directories contain both metadata and binaries

### 2.3 Current Artifact Metadata Structure

```python
class ArtifactMeta(TypedDict):
    hash: str           # SHA256 hash with "sha256:" prefix
    name: str           # Original name for reference
    path: str           # Relative path: "<ClassName>_<short_hash>.<ext>"
    format: str         # Serialization format (joblib, pickle, keras, etc.)
    format_version: str # Library version (e.g., 'sklearn==1.3.0')
    nirs4all_version: str
    size: int
    saved_at: str       # ISO timestamp
    step: int           # Pipeline step number
    branch_id: Optional[int]
    branch_name: Optional[str]
```

### 2.4 Identified Problems

#### Problem 1: Step-Based Identification is Fragile

The current system identifies artifacts by `step` number (0, 1, 2...). This creates problems:

```python
# Current: Artifacts indexed by step number
artifacts_by_step: Dict[int, List[ArtifactMeta]] = {
    0: [scaler_artifact],
    1: [splitter_artifact],
    2: [model_artifact]
}
```

**Issues:**
- Step numbers depend on execution order
- Generator expansion changes step counts between runs
- Branching creates parallel execution paths with conflicting step numbers
- Replay mode must reconstruct the exact step sequence

#### Problem 2: Iterator-Based Naming is Non-Deterministic

```python
# Current filename: <ClassName>_<short_hash>.<ext>
# Example: StandardScaler_a1b2c3.pkl
```

The naming relies on:
- Sequential iterator during training (now removed but legacy artifacts exist)
- Class name extraction (ambiguous for wrapper classes)
- Short hash collision potential

#### Problem 3: Branch Artifacts Lack Context Hierarchy

Current branch support stores `branch_id` and `branch_name` in metadata, but:
- No parent-child relationship for branching
- No clear distinction between shared (pre-branch) and branch-specific artifacts
- BinaryLoader uses tuple keys `(step, branch_id)` which doesn't scale for nested branches

#### Problem 4: Stacking Dependencies Not Tracked

Meta-models need to reference source models, but the current system has:
- No dependency graph between artifacts
- No way to express "meta-model M depends on models A, B, C"
- Source model ordering is critical for feature column matching but not enforced

#### Problem 5: Backward Compatibility Overhead

The codebase maintains multiple loading paths:
- `legacy_pickle` format for old `metadata.json` files
- Path format detection (`/` in path vs flat filename)
- `_binaries/` vs `artifacts/objects/` directory structures

### 2.5 Current Usage Patterns

#### Training Mode
```python
# In model controller
artifact = persist(model, artifacts_dir, "model", branch_id=ctx.branch_id)
artifact['step'] = runtime_ctx.step_number
manager.append_artifacts(pipeline_id, [artifact])
```

#### Predict Mode
```python
# BinaryLoader groups by (step, branch_id)
loader = BinaryLoader.from_manifest(manifest, results_dir)
binaries = loader.get_step_binaries(step_num, branch_id=ctx.branch_id)
```

---

## 3. Proposed Solution

### 3.1 Core Concept: Execution Path as Artifact Identity

Replace step numbers with an **execution path** that captures the full context:

```python
execution_path = {
    "pipeline_id": "0001_pls_abc123",
    "branch_path": [0, 2],      # Branch indices if nested
    "step_index": 3,            # Logical step within branch
    "operation": "fit",         # fit, transform, predict
    "fold_id": 0                # CV fold (None for non-folded)
}
```

**Artifact ID** is derived from this path:
```python
artifact_id = f"{pipeline_id}:{':'.join(map(str, branch_path))}:{step_index}:{fold_id or 'all'}"
# Example: "0001_pls_abc123:0:2:3:0"
```

### 3.2 New Artifact Metadata Structure

```python
@dataclass
class ArtifactRecord:
    """Complete artifact metadata for manifest storage."""

    # Identification
    artifact_id: str              # Unique, deterministic ID
    content_hash: str             # SHA256 of binary content

    # Location
    path: str                     # Relative path in _binaries/

    # Context
    pipeline_id: str              # Parent pipeline
    branch_path: List[int]        # Branch hierarchy (empty = pre-branch)
    step_index: int               # Logical step in execution
    fold_id: Optional[int]        # CV fold (None = shared across folds)

    # Classification
    artifact_type: str            # "model", "transformer", "splitter", "encoder"
    class_name: str               # "PLSRegression", "StandardScaler"

    # Dependencies
    depends_on: List[str]         # List of artifact_ids this depends on

    # Serialization
    format: str                   # "joblib", "pickle", "keras", etc.
    format_version: str           # Library version
    nirs4all_version: str

    # Metadata
    size_bytes: int
    created_at: str               # ISO timestamp
    params: Dict[str, Any]        # Hyperparameters for models
```

### 3.3 Filename Convention

**New format**: `<type>_<class>_<short_hash>.<ext>`

```
model_PLSRegression_a1b2c3.joblib
transformer_StandardScaler_d4e5f6.pkl
splitter_ShuffleSplit_789abc.pkl
encoder_LabelEncoder_def012.pkl
```

Benefits:
- Type prefix enables quick filtering (`model_*`)
- Class name provides semantic meaning
- Short hash (6 chars) ensures uniqueness
- Content-addressed deduplication preserved

### 3.4 New Filesystem Architecture

**Key change:** Artifacts are stored **centrally per dataset**, separate from runs.

```
workspace/
├── binaries/                                    # CENTRALIZED ARTIFACT STORAGE
│   ├── corn_m5/                                 # Per-dataset artifact pool
│   │   ├── model_PLSRegression_a1b2c3.joblib
│   │   ├── model_PLSRegression_d4e5f6.joblib   # Different params → different hash
│   │   ├── transformer_StandardScaler_789abc.pkl
│   │   ├── transformer_SNV_def012.pkl
│   │   └── splitter_ShuffleSplit_345678.pkl
│   └── wheat_protein/
│       └── ...                                  # Other dataset's artifacts
│
└── runs/                                        # LIGHTWEIGHT RUN METADATA
    └── corn_m5/                                 # Per-dataset runs
        ├── 20251210_a1b2c3/                     # Format: YYYYMMDD_uniqueid
        │   ├── 0001_pls/
        │   │   └── manifest.yaml                # References artifacts by hash
        │   ├── 0002_rf/
        │   │   └── manifest.yaml
        │   └── predictions.json
        └── 20251212_d4e5f6/                     # Another run, same dataset
            ├── 0001_pls/
            │   └── manifest.yaml                # May reference SAME artifacts!
            └── predictions.json
```

**Benefits:**
- **Global deduplication**: Same content hash = same file, across ALL runs
- **Lightweight runs**: Only manifests and predictions, no binaries
- **Easy cleanup**: Delete run folder without losing shared artifacts
- **Cross-run sharing**: Identical models reused automatically

**Deduplication logic:**
```python
def persist(self, content: bytes, dataset: str, ...) -> str:
    content_hash = hashlib.sha256(content).hexdigest()[:12]
    filename = f"{artifact_type}_{class_name}_{content_hash}.{ext}"
    path = workspace / "binaries" / dataset / filename

    if not path.exists():  # Only write if new
        path.write_bytes(content)

    return filename  # Same hash → same filename → no duplicate
```

### 3.5 Dependency Graph for Stacking

```yaml
# In manifest.yaml
artifacts:
  - artifact_id: "0001:3:all"
    artifact_type: model
    class_name: PLSRegression
    depends_on: ["0001:0:all", "0001:1:all"]  # Scaler and splitter

  - artifact_id: "0001:4:all"
    artifact_type: model
    class_name: RandomForestRegressor
    depends_on: ["0001:0:all", "0001:1:all"]

  - artifact_id: "0001:5:all"
    artifact_type: meta_model
    class_name: Ridge
    depends_on: ["0001:3:all", "0001:4:all"]  # Source models
    meta_config:
      source_models:
        - {artifact_id: "0001:3:all", feature_index: 0}
        - {artifact_id: "0001:4:all", feature_index: 1}
      feature_columns: ["PLSRegression_pred", "RandomForestRegressor_pred"]
```

### 3.6 New Component Architecture

```
pipeline/storage/artifacts/
├── __init__.py
├── artifact_registry.py       # NEW: Central registry with ID generation
├── artifact_persistence.py    # Serialization (mostly unchanged)
├── artifact_loader.py         # NEW: Replaces BinaryLoader with path-aware loading
├── manager.py                 # Simplified, delegates to registry
└── types.py                   # NEW: ArtifactRecord, ArtifactType enum
```

#### ArtifactRegistry (New)

```python
class ArtifactRegistry:
    """Central registry for artifact management."""

    def generate_id(
        self,
        pipeline_id: str,
        branch_path: List[int],
        step_index: int,
        fold_id: Optional[int] = None
    ) -> str:
        """Generate deterministic artifact ID."""

    def register(
        self,
        artifact: ArtifactRecord,
        content: bytes
    ) -> ArtifactRecord:
        """Register artifact with deduplication."""

    def resolve(
        self,
        artifact_id: str
    ) -> ArtifactRecord:
        """Resolve ID to artifact record."""

    def resolve_dependencies(
        self,
        artifact_id: str
    ) -> List[ArtifactRecord]:
        """Get all dependencies (transitive)."""
```

#### ArtifactLoader (New)

```python
class ArtifactLoader:
    """Load artifacts by ID or execution context."""

    def load_by_id(self, artifact_id: str) -> Any:
        """Load single artifact by ID."""

    def load_for_step(
        self,
        step_index: int,
        branch_path: List[int],
        fold_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Load all artifacts for a step context."""

    def load_with_dependencies(
        self,
        artifact_id: str
    ) -> Dict[str, Any]:
        """Load artifact and all its dependencies."""
```

### 3.7 Manifest Schema Evolution

```yaml
# manifest.yaml v2.0
schema_version: "2.0"
uid: "a1b2c3d4-e5f6-4789-abcd-ef0123456789"
pipeline_id: "0001_pls_abc123"
name: "pls_baseline"
dataset: "corn_m5"
created_at: "2025-12-12T10:00:00Z"

# New artifact section
artifacts:
  registry_version: 1
  items:
    - artifact_id: "0001:0:all"
      content_hash: "sha256:abc123..."
      path: "transformer_StandardScaler_abc123.pkl"
      artifact_type: transformer
      class_name: StandardScaler
      branch_path: []
      step_index: 0
      depends_on: []
      format: joblib
      format_version: "sklearn==1.5.0"
      size_bytes: 2048

    - artifact_id: "0001:0:3:0"
      content_hash: "sha256:def456..."
      path: "model_PLSRegression_def456_b0.joblib"
      artifact_type: model
      class_name: PLSRegression
      branch_path: [0]
      step_index: 3
      fold_id: 0
      depends_on: ["0001:0:all", "0001:0:2:all"]
      format: joblib
      format_version: "sklearn==1.5.0"
      size_bytes: 15360
      params:
        n_components: 10

predictions: [...]
```

### 3.8 Implementation Strategy

> **No backward compatibility.** This is a clean-slate implementation. All legacy code will be deleted.

#### Phase 1: New System Implementation
- Implement `ArtifactRegistry` and `ArtifactLoader`
- Delete `ArtifactManager` and `BinaryLoader` entirely
- New filesystem structure only

#### Phase 2: Controller Updates
- Update all controllers to use new registry
- Generate artifact IDs using execution context
- Track dependencies in model controllers

#### Phase 3: Legacy Removal
- Delete `legacy_pickle` format handling
- Delete `metadata.json` support
- Delete old directory structure code (`_binaries/`, `artifacts/objects/`)
- Update all tests

### 3.9 API Changes

#### Training
```python
# Before
artifact = persist(model, artifacts_dir, "model")
artifact['step'] = step_number
manager.append_artifacts(pipeline_id, [artifact])

# After
artifact_id = registry.generate_id(
    pipeline_id=pipeline_id,
    branch_path=ctx.selector.branch_path,
    step_index=ctx.state.step_number,
    fold_id=ctx.selector.fold_id
)
record = registry.register(
    artifact_id=artifact_id,
    artifact_type="model",
    class_name=model.__class__.__name__,
    content=serialize(model),
    depends_on=upstream_artifacts,
    params=model.get_params()
)
```

#### Prediction
```python
# Before
loader = BinaryLoader.from_manifest(manifest, results_dir)
binaries = loader.get_step_binaries(step_num, branch_id=branch_id)

# After
loader = ArtifactLoader.from_manifest(manifest, results_dir)
artifacts = loader.load_for_step(
    step_index=step_num,
    branch_path=branch_path,
    fold_id=fold_id
)
```

---

## 4. Rationale

### 4.1 Why Execution Path Instead of Step Number?

| Criterion | Step Number | Execution Path |
|-----------|-------------|----------------|
| Determinism | ❌ Changes with generators | ✅ Stable |
| Branching | ❌ Conflicts | ✅ Native support |
| Replay | ❌ Requires reconstruction | ✅ Direct lookup |
| Debugging | ❌ Opaque | ✅ Self-documenting |

### 4.2 Why Dependency Tracking?

**Stacking requires it:**
```python
# Meta-model needs to know which source models to load
meta_model.source_models = ["pls_model", "rf_model"]

# Without dependency tracking:
# - Must manually specify source models in config
# - No validation that sources exist
# - No automatic loading of preprocessing chain

# With dependency tracking:
# - Load meta-model → automatically load sources
# - Validate dependency graph at load time
# - GC can trace what's still needed
```

**Transfer learning benefits:**
```python
# Transfer from pipeline A to B
# Dependencies tell us what preprocessing to apply
transfer_artifacts = loader.load_with_dependencies("pipeline_a:model_artifact")
```

### 4.3 Why Branch Path Instead of Single branch_id?

Nested branching is coming:
```python
pipeline = [
    {"branch": [
        [SNV(), {"branch": [PLS(5), PLS(10)]}],  # Nested branch
        [MSC(), PLS(10)],
    ]},
]
```

Single `branch_id` cannot represent `[0, 1]` (branch 1 under branch 0).

### 4.4 Why Keep Flat Storage with Suffixes?

Considered: subdirectory per branch (`branch_0/`, `branch_1/`)

Rejected because:
- Deduplication harder across branches
- More filesystem operations
- Complicates GC (must scan all subdirs)

Flat storage with suffixes (`_b0`, `_b1`) provides:
- Single directory to scan
- Easy deduplication (same hash = same file)
- Suffix encodes branch path

---

## 5. Global Deduplication

### 5.1 How It Works

Artifacts are stored in `/workspace/binaries/<dataset_name>/` and identified by content hash:

```python
# Two runs on same dataset, same PLS model with same params
Run 1: trains PLSRegression(n_components=10) → hash: a1b2c3d4e5f6
Run 2: trains PLSRegression(n_components=10) → hash: a1b2c3d4e5f6 (SAME!)

# Only ONE file exists:
# workspace/binaries/corn_m5/model_PLSRegression_a1b2c3d4e5f6.joblib

# Both manifests reference same file:
# runs/corn_m5/20251210_xxx/0001/manifest.yaml → path: model_PLSRegression_a1b2c3d4e5f6.joblib
# runs/corn_m5/20251212_yyy/0001/manifest.yaml → path: model_PLSRegression_a1b2c3d4e5f6.joblib
```

### 5.2 Hash Computation

```python
def compute_content_hash(content: bytes) -> str:
    """Compute SHA-256 hash, truncated to 12 chars for filename."""
    return hashlib.sha256(content).hexdigest()[:12]
```

**Why SHA-256 truncated to 12 chars?**
- 12 hex chars = 48 bits = 281 trillion combinations
- Collision probability negligible for typical dataset sizes
- Keeps filenames readable

---

## 6. Artifact Cleanup Utilities

### 6.1 Orphan Detection and Deletion

```python
class ArtifactRegistry:
    """Cleanup methods for artifact management."""

    def find_orphaned_artifacts(self, dataset: str) -> List[str]:
        """
        Find artifacts not referenced by any manifest.

        Scans all manifests in workspace/runs/<dataset>/**/manifest.yaml
        and compares with files in workspace/binaries/<dataset>/.

        Returns:
            List of orphaned artifact filenames
        """

    def delete_orphaned_artifacts(
        self,
        dataset: str,
        dry_run: bool = True
    ) -> Tuple[List[str], int]:
        """
        Delete artifacts not referenced by any manifest.

        Args:
            dataset: Dataset name
            dry_run: If True, only report what would be deleted

        Returns:
            (deleted_files, bytes_freed)
        """

    def delete_pipeline_artifacts(self, pipeline_id: str) -> int:
        """Delete all artifacts for a specific pipeline."""

    def cleanup_failed_run(self, run_path: Path) -> int:
        """Clean up artifacts from a failed run (auto-called on exception)."""
```

### 6.2 CLI Interface

```bash
# List orphaned artifacts
nirs4all artifacts list-orphaned --dataset corn_m5

# Delete orphaned artifacts (dry run)
nirs4all artifacts cleanup --dataset corn_m5

# Delete orphaned artifacts (force)
nirs4all artifacts cleanup --dataset corn_m5 --force

# Show storage statistics
nirs4all artifacts stats --dataset corn_m5
# Output:
#   Total artifacts: 145
#   Referenced: 120
#   Orphaned: 25 (12.3 MB)
#   Deduplication savings: 45% (89 unique files for 145 references)

# Delete all artifacts for a dataset
nirs4all artifacts purge --dataset corn_m5 --force
```

### 6.3 Automatic Cleanup on Failure

```python
# In runner.py
try:
    result = self._execute_pipeline(config)
except Exception as e:
    # Remove artifacts created during this failed run
    self.artifact_registry.cleanup_failed_run(self.run_path)
    raise
```

---

## 7. Design Decisions

### 7.1 fold_id is Part of Artifact ID

**Decision:** Yes, `fold_id` is fundamental for models.

**Rationale:**
- CV-trained models require per-fold artifacts
- Prediction can average fold models: `y_pred = mean([model_fold0.predict(X), ...])`
- Essential for proper cross-validation semantics

```python
# Artifact ID format includes fold:
artifact_id = f"{pipeline_id}:{branch_path}:{step_index}:{fold_id}"
# Example: "0001_pls:0:3:0" (branch 0, step 3, fold 0)
# Example: "0001_pls::2:all" (no branch, step 2, shared across folds)
```

### 7.2 Delete Artifacts on Pipeline Failure

**Decision:** Yes, delete artifacts when a pipeline fails.

**Rationale:**
- Failed pipelines produce incomplete/invalid artifacts
- Artifacts are optional (`save_files` runner parameter)
- Cleanup is automatic + manual tools provided

### 7.3 Artifact Versioning Supported

**Decision:** Yes, support artifact versioning (v1, v2, etc.).

**Rationale:**
- Enables A/B testing of model versions
- Useful for incremental training/fine-tuning

```python
@dataclass
class ArtifactRecord:
    # ... existing fields ...
    version: int = 1  # Artifact version
```

### 7.4 Local Storage Only

**Decision:** All storage is local. No remote storage.

**Rationale:**
- Implementation phase, not production
- Remote storage deferred to future

---

## 8. References

- [Manifest Specification](../specifications/manifest.md)
- [Outputs vs Artifacts](../reference/outputs_vs_artifacts.md)
- [Metamodel Stacking Strategy](./METAMODEL_STACKING_STRATEGY.md)
- [Branching Documentation](../reference/branching.md)
