# Artifacts System v2 - Complete Reference

## Overview

The nirs4all artifacts system v2 provides a comprehensive solution for managing
binary artifacts (trained models, fitted transformers, splitters) throughout the
machine learning pipeline lifecycle. It features:

- **Deterministic Identification**: Stable artifact IDs based on execution context
- **Content-Addressed Storage**: Automatic deduplication with SHA256 hashing
- **Branch-Aware Artifacts**: Native support for pipeline branching
- **Stacking Support**: Meta-model dependency tracking
- **Fold-Aware Storage**: Per-fold model persistence for CV averaging
- **Cleanup Utilities**: CLI tools for orphan detection and deletion

## Key Concepts

### Artifact ID Format

Artifacts are identified by a deterministic ID that encodes the execution context:

```
<pipeline_id>:<branch_path>:<step_index>:<fold_id>

Examples:
- "0001_pls:3:all"         # Pipeline 0001_pls, step 3, shared across folds
- "0001_pls:0:3:0"         # Pipeline 0001_pls, branch 0, step 3, fold 0
- "0001_pls:0:2:5:all"     # Nested branch [0, 2], step 5, shared
```

### Artifact Types

| Type | Value | Description |
|------|-------|-------------|
| **MODEL** | `model` | Trained ML models (sklearn, keras, pytorch, etc.) |
| **TRANSFORMER** | `transformer` | Fitted transformers (scalers, preprocessors) |
| **SPLITTER** | `splitter` | Fitted cross-validation splitters |
| **ENCODER** | `encoder` | Fitted label encoders/target transformers |
| **META_MODEL** | `meta_model` | Stacking meta-models with source references |

### Filesystem Structure

```
workspace/
├── binaries/                     # Centralized artifact storage
│   ├── corn_m5/                  # Per-dataset artifact pool
│   │   ├── model_PLSRegression_a1b2c3.joblib
│   │   ├── transformer_StandardScaler_d4e5f6.pkl
│   │   └── ...
│   └── wheat_protein/
│       └── ...
│
└── runs/                         # Lightweight run metadata
    └── corn_m5/
        ├── 20251210_abc123/
        │   └── 0001_pls/
        │       └── manifest.yaml  # References artifacts by hash
        └── ...
```

## Manifest Schema v2

Manifests use the v2 schema for artifact tracking:

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
      size_bytes: 2048

predictions: [...]
```

## Deduplication

Artifacts with identical content share the same file through content-addressed storage:

```python
# Two pipelines train same model on same data
# → Same content hash → Same file on disk

# Manifest 1 references: model_PLSRegression_a1b2c3.joblib
# Manifest 2 references: model_PLSRegression_a1b2c3.joblib
# Only ONE file exists in workspace/binaries/
```

Benefits:
- **Space savings**: ~25-40% reduction in storage
- **Fast training**: Skip serialization if file exists
- **Clean manifests**: References only, no duplication

## Branch-Aware Artifacts

### Shared vs Branch-Specific

Artifacts can be:
- **Shared (pre-branch)**: `branch_path: []` - Applied before branching
- **Branch-specific**: `branch_path: [0]` - Specific to branch 0
- **Nested branch**: `branch_path: [0, 2]` - Branch 2 within branch 0

### Example: Branching Pipeline

```python
pipeline = [
    ShuffleSplit(n_splits=3),
    MinMaxScaler(),              # Shared artifact
    {"branch": [
        [SNV()],                 # Branch 0 artifact
        [MSC()],                 # Branch 1 artifact
    ]},
    PLSRegression(10),           # Separate artifact per branch
]
```

Results in:
```
Artifacts:
- 0001:0:all (MinMaxScaler, shared)
- 0001:0:2:all (SNV, branch 0)
- 0001:1:2:all (MSC, branch 1)
- 0001:0:3:all (PLS, branch 0)
- 0001:1:3:all (PLS, branch 1)
```

## CV Fold Models

Per-fold model artifacts enable ensemble prediction:

```python
# Training creates per-fold artifacts:
# - 0001:3:0 (fold 0 model)
# - 0001:3:1 (fold 1 model)
# - 0001:3:2 (fold 2 model)

# Load for ensemble prediction:
from nirs4all.pipeline.storage.artifacts import ArtifactLoader

loader = ArtifactLoader(workspace_path, dataset)
loader.import_from_manifest(manifest)

fold_models = loader.load_fold_models(step_index=3)
# Returns: [(0, model_0), (1, model_1), (2, model_2)]

# Ensemble prediction:
predictions = [model.predict(X) for _, model in fold_models]
ensemble_pred = np.mean(predictions, axis=0)
```

## Stacking / Meta-Model Support

Meta-models track their source model dependencies:

```yaml
artifacts:
  items:
    - artifact_id: "0001:3:all"
      artifact_type: model
      class_name: PLSRegression
      # ...

    - artifact_id: "0001:4:all"
      artifact_type: model
      class_name: RandomForestRegressor
      # ...

    - artifact_id: "0001:5:all"
      artifact_type: meta_model
      class_name: Ridge
      depends_on: ["0001:3:all", "0001:4:all"]
      meta_config:
        source_models:
          - artifact_id: "0001:3:all"
            feature_index: 0
          - artifact_id: "0001:4:all"
            feature_index: 1
        feature_columns: ["PLSRegression_pred", "RandomForestRegressor_pred"]
```

### Loading Meta-Models

```python
loader = ArtifactLoader(workspace_path, dataset)
loader.import_from_manifest(manifest)

# Load meta-model with all source models
meta_model, source_models, feature_columns = loader.load_meta_model_with_sources(
    "0001:5:all"
)

# source_models: [("0001:3:all", pls_model), ("0001:4:all", rf_model)]
# feature_columns: ["PLSRegression_pred", "RandomForestRegressor_pred"]
```

## Cleanup Utilities

### CLI Commands

```bash
# List orphaned artifacts (not referenced by any manifest)
nirs4all artifacts list-orphaned --dataset corn_m5

# Show storage statistics
nirs4all artifacts stats --dataset corn_m5
# Output:
#   Files on disk:     145
#   Registered refs:   195
#   Deduplication:     25.6%
#   Orphaned:          12 files (5.2 MB)

# Delete orphaned artifacts (dry run)
nirs4all artifacts cleanup --dataset corn_m5

# Delete orphaned artifacts (force)
nirs4all artifacts cleanup --dataset corn_m5 --force

# Purge ALL artifacts for a dataset (destructive!)
nirs4all artifacts purge --dataset corn_m5 --force
```

### Programmatic API

```python
from nirs4all.pipeline.storage.artifacts import ArtifactRegistry

registry = ArtifactRegistry(workspace_path, dataset)

# Find orphans
orphans = registry.find_orphaned_artifacts()
print(f"Found {len(orphans)} orphaned files")

# Get stats
stats = registry.get_stats()
print(f"Deduplication ratio: {stats['deduplication_ratio']:.1%}")

# Delete orphans
deleted, bytes_freed = registry.delete_orphaned_artifacts(dry_run=False)
print(f"Freed {bytes_freed / 1024:.1f} KB")

# Auto-cleanup failed runs
registry.start_run()
try:
    # ... pipeline execution ...
    registry.end_run()
except Exception:
    registry.cleanup_failed_run()  # Removes current run's artifacts
    raise
```

## API Reference

### ArtifactRegistry

```python
class ArtifactRegistry:
    """Central registry for artifact management."""

    def __init__(self, workspace: Path, dataset: str):
        """Initialize registry for a dataset."""

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
        obj: Any,
        artifact_id: str,
        artifact_type: ArtifactType,
        depends_on: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> ArtifactRecord:
        """Register and persist an artifact with deduplication."""

    def register_meta_model(
        self,
        obj: Any,
        artifact_id: str,
        source_model_ids: List[str],
        feature_columns: Optional[List[str]] = None
    ) -> ArtifactRecord:
        """Register a stacking meta-model with source references."""

    def resolve(self, artifact_id: str) -> Optional[ArtifactRecord]:
        """Resolve artifact ID to record."""

    def get_dependencies(self, artifact_id: str) -> List[str]:
        """Get direct dependencies of an artifact."""

    def resolve_dependencies(self, artifact_id: str) -> List[ArtifactRecord]:
        """Get all transitive dependencies in topological order."""

    # Cleanup methods
    def find_orphaned_artifacts(self) -> List[str]:
        """Find artifacts not referenced by any manifest."""

    def delete_orphaned_artifacts(self, dry_run: bool = True) -> Tuple[List[str], int]:
        """Delete orphaned artifacts."""

    def cleanup_failed_run(self) -> int:
        """Clean up artifacts from a failed run."""

    def purge_dataset_artifacts(self, confirm: bool = False) -> Tuple[int, int]:
        """Delete ALL artifacts for this dataset."""

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
```

### ArtifactLoader

```python
class ArtifactLoader:
    """Load artifacts from manifest and binaries."""

    def __init__(
        self,
        workspace: Path,
        dataset: str,
        results_dir: Optional[Path] = None,
        cache_size: int = 100
    ):
        """Initialize loader with optional LRU cache."""

    def import_from_manifest(self, manifest: Dict[str, Any]) -> None:
        """Import artifact records from manifest."""

    def load_by_id(self, artifact_id: str) -> Any:
        """Load single artifact by ID (with caching)."""

    def load_for_step(
        self,
        step_index: int,
        branch_path: Optional[List[int]] = None,
        fold_id: Optional[int] = None
    ) -> List[Tuple[str, Any]]:
        """Load all artifacts for a step context."""

    def load_fold_models(
        self,
        step_index: int,
        branch_path: Optional[List[int]] = None
    ) -> List[Tuple[int, Any]]:
        """Load all fold-specific models for CV averaging."""

    def load_with_dependencies(self, artifact_id: str) -> Dict[str, Any]:
        """Load artifact and all its dependencies."""

    def load_meta_model_with_sources(
        self,
        artifact_id: str,
        validate_branch: bool = False
    ) -> Tuple[Any, List[Tuple[str, Any]], List[str]]:
        """Load meta-model with source models and feature columns."""

    # Cache management
    def preload_artifacts(self, artifact_ids: Optional[List[str]] = None) -> int:
        """Warm cache with specified or all artifacts."""

    def set_cache_size(self, new_size: int) -> None:
        """Dynamically resize the cache."""

    def clear_cache(self) -> None:
        """Clear the artifact cache."""

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
```

### ArtifactRecord

```python
@dataclass
class ArtifactRecord:
    """Complete artifact metadata."""

    # Identification
    artifact_id: str
    content_hash: str

    # Location
    path: str

    # Context
    pipeline_id: str
    branch_path: List[int] = field(default_factory=list)
    step_index: int = 0
    fold_id: Optional[int] = None

    # Classification
    artifact_type: ArtifactType = ArtifactType.MODEL
    class_name: str = ""

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    # Meta-model config (for stacking)
    meta_config: Optional[MetaModelConfig] = None

    # Serialization
    format: str = "joblib"
    format_version: str = ""
    nirs4all_version: str = ""

    # Metadata
    size_bytes: int = 0
    created_at: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    # Computed properties
    @property
    def is_branch_specific(self) -> bool: ...
    @property
    def is_fold_specific(self) -> bool: ...
    @property
    def is_meta_model(self) -> bool: ...
    @property
    def short_hash(self) -> str: ...
```

## Best Practices

### 1. Always Use save_artifacts=True for Production

```python
runner = PipelineRunner(save_artifacts=True)  # Artifacts saved
runner = PipelineRunner(save_artifacts=False, save_charts=False) # No artifacts (for testing)
```

### 2. Regular Cleanup

Run cleanup periodically to free disk space:

```bash
# Check for orphans weekly
nirs4all artifacts stats

# Clean up orphans
nirs4all artifacts cleanup --force
```

### 3. Use Named Branches for Clarity

```python
# Good: Named branches
{"branch": {
    "snv": [SNV()],
    "msc": [MSC()],
}}

# OK: Anonymous branches
{"branch": [
    [SNV()],
    [MSC()],
]}
```

### 4. Preload Artifacts for Batch Prediction

```python
loader = ArtifactLoader(workspace_path, dataset)
loader.import_from_manifest(manifest)

# Preload all artifacts into cache before prediction loop
loader.preload_artifacts()

# Fast predictions (all from cache)
for sample in samples:
    model = loader.load_by_id("0001:3:all")
    predictions.append(model.predict(sample))
```

## Troubleshooting

### Artifact Not Found

```
FileNotFoundError: Artifact not found: /workspace/binaries/dataset/model_xyz.joblib
```

**Cause**: The binary file was deleted but manifest still references it.

**Solution**:
1. Re-run training to regenerate artifacts
2. Or update manifest to remove missing references

### Hash Mismatch

```
ValueError: Content hash mismatch for artifact 0001:3:all
```

**Cause**: Artifact file was modified after saving.

**Solution**: Delete the corrupted artifact and re-run training.

### Missing Dependencies for Meta-Model

```
ValueError: Cannot register meta-model: missing source model dependencies: 0001:3:all
```

**Cause**: Trying to register a meta-model before its source models.

**Solution**: Register source models first, then meta-model.

## See Also

- [Outputs vs Artifacts](./outputs_vs_artifacts.md) - Overview of serialization architecture
- [Manifest Specification](../specifications/manifest.md) - Full manifest schema
- [Branching Reference](./branching.md) - Pipeline branching documentation
- [Q5_predict.py](../../examples/Q5_predict.py) - Prediction example
- [Q18_stacking.py](../../examples/Q18_stacking.py) - Stacking example
- [Q30_branching.py](../../examples/Q30_branching.py) - Branching example
