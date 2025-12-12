# Storage API Reference

This document provides the API reference for the nirs4all artifact storage system.

## Module: `nirs4all.pipeline.storage.artifacts`

### Types and Enums

#### ArtifactType

```python
class ArtifactType(Enum):
    """Classification of artifact types for serialization strategy."""

    MODEL = "model"           # Trained ML models
    TRANSFORMER = "transformer"  # Fitted transformers
    SPLITTER = "splitter"     # Fitted CV splitters
    ENCODER = "encoder"       # Label encoders
    META_MODEL = "meta_model"  # Stacking meta-models
```

**Usage:**
```python
from nirs4all.pipeline.storage.artifacts import ArtifactType

record = registry.register(
    model, artifact_id, ArtifactType.MODEL
)
```

---

#### MetaModelConfig

```python
@dataclass
class MetaModelConfig:
    """Configuration for meta-model source tracking."""

    source_models: List[SourceModelRef]  # References to source models
    feature_columns: List[str] = field(default_factory=list)  # Feature names
```

**Fields:**
- `source_models`: List of `SourceModelRef` with artifact_id and feature_index
- `feature_columns`: Ordered list of feature column names in meta-model input

**Serialization:**
```python
config = MetaModelConfig(
    source_models=[
        SourceModelRef(artifact_id="0001:3:all", feature_index=0),
        SourceModelRef(artifact_id="0001:4:all", feature_index=1),
    ],
    feature_columns=["PLSRegression_pred", "RandomForest_pred"]
)
config_dict = config.to_dict()
config_back = MetaModelConfig.from_dict(config_dict)
```

---

#### SourceModelRef

```python
@dataclass
class SourceModelRef:
    """Reference to a source model in stacking."""

    artifact_id: str       # ID of the source model artifact
    feature_index: int = 0  # Column index in meta-model feature matrix
```

---

#### ArtifactRecord

```python
@dataclass
class ArtifactRecord:
    """Complete artifact metadata record."""
```

**Core Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `artifact_id` | `str` | Deterministic ID (pipeline:branch:step:fold) |
| `content_hash` | `str` | SHA256 hash for deduplication |
| `path` | `str` | Relative path to artifact file |
| `pipeline_id` | `str` | Pipeline identifier |
| `branch_path` | `List[int]` | Branch indices (empty for shared) |
| `step_index` | `int` | Pipeline step index |
| `fold_id` | `Optional[int]` | CV fold index (None for shared) |
| `artifact_type` | `ArtifactType` | Type classification |
| `class_name` | `str` | Python class name of artifact |
| `depends_on` | `List[str]` | Dependency artifact IDs |
| `meta_config` | `Optional[MetaModelConfig]` | Meta-model configuration |
| `format` | `str` | Serialization format ("joblib", "pkl") |
| `format_version` | `str` | Library version (e.g., "sklearn==1.5.0") |
| `nirs4all_version` | `str` | nirs4all version at creation |
| `size_bytes` | `int` | File size in bytes |
| `created_at` | `str` | ISO8601 timestamp |
| `params` | `Dict[str, Any]` | Additional metadata |

**Properties:**

```python
record.is_branch_specific  # bool: True if branch_path is non-empty
record.is_fold_specific    # bool: True if fold_id is not None
record.is_meta_model       # bool: True if artifact_type is META_MODEL
record.short_hash          # str: First 8 chars of content_hash
```

**Serialization:**
```python
record_dict = record.to_dict()
record_back = ArtifactRecord.from_dict(record_dict)
```

---

### DependencyGraph

```python
class DependencyGraph:
    """DAG for artifact dependencies (topological ordering)."""
```

**Methods:**

```python
def __init__(self) -> None:
    """Initialize empty dependency graph."""

def add_node(self, node: str) -> None:
    """Add a node to the graph."""

def add_edge(self, from_node: str, to_node: str) -> None:
    """Add a dependency edge (from_node depends on to_node)."""

def topological_sort(self) -> List[str]:
    """Return nodes in dependency order (dependencies first)."""

def get_dependencies(self, node: str) -> Set[str]:
    """Get direct dependencies of a node."""

def get_all_dependencies(self, node: str) -> Set[str]:
    """Get all transitive dependencies of a node."""
```

**Usage:**
```python
graph = DependencyGraph()
graph.add_node("model_a")
graph.add_node("model_b")
graph.add_node("meta_model")
graph.add_edge("meta_model", "model_a")  # meta depends on a
graph.add_edge("meta_model", "model_b")  # meta depends on b

order = graph.topological_sort()
# ["model_a", "model_b", "meta_model"]
```

---

### ArtifactRegistry

```python
class ArtifactRegistry:
    """Central registry for artifact management and persistence."""
```

#### Constructor

```python
def __init__(
    self,
    workspace: Path,
    dataset: str,
    run_id: Optional[str] = None,
    pipeline_id: Optional[str] = None
) -> None:
    """
    Initialize registry for a dataset.

    Args:
        workspace: Path to workspace root
        dataset: Dataset name
        run_id: Optional run identifier for cleanup tracking
        pipeline_id: Optional pipeline identifier for ID generation
    """
```

#### ID Generation

```python
def generate_id(
    self,
    pipeline_id: str,
    branch_path: List[int],
    step_index: int,
    fold_id: Optional[int] = None
) -> str:
    """
    Generate deterministic artifact ID.

    Args:
        pipeline_id: Pipeline identifier
        branch_path: Branch indices (empty list for shared)
        step_index: Step index in pipeline
        fold_id: Optional fold index for CV

    Returns:
        Formatted artifact ID (e.g., "0001:0:3:all")

    Examples:
        >>> registry.generate_id("0001", [], 3, None)
        "0001:3:all"
        >>> registry.generate_id("0001", [0], 3, 2)
        "0001:0:3:2"
    """
```

#### Registration

```python
def register(
    self,
    obj: Any,
    artifact_id: str,
    artifact_type: ArtifactType,
    depends_on: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    skip_if_exists: bool = True
) -> ArtifactRecord:
    """
    Register and persist an artifact with content-addressed deduplication.

    Args:
        obj: The object to serialize (model, transformer, etc.)
        artifact_id: Deterministic ID for the artifact
        artifact_type: Classification of artifact type
        depends_on: List of dependency artifact IDs
        params: Additional metadata to store
        skip_if_exists: If True, skip serialization if hash exists

    Returns:
        ArtifactRecord with complete metadata

    Raises:
        ValueError: If depends_on contains unregistered IDs
    """
```

```python
def register_meta_model(
    self,
    obj: Any,
    artifact_id: str,
    source_model_ids: List[str],
    feature_columns: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None
) -> ArtifactRecord:
    """
    Register a stacking meta-model with source model references.

    Args:
        obj: The meta-model object
        artifact_id: Deterministic ID
        source_model_ids: IDs of source models used for stacking
        feature_columns: Feature column names in order
        params: Additional metadata

    Returns:
        ArtifactRecord with meta_config populated

    Raises:
        ValueError: If any source_model_id is not registered
    """
```

#### Resolution

```python
def resolve(self, artifact_id: str) -> Optional[ArtifactRecord]:
    """
    Resolve artifact ID to record.

    Args:
        artifact_id: The artifact ID to resolve

    Returns:
        ArtifactRecord if found, None otherwise
    """
```

```python
def resolve_dependencies(self, artifact_id: str) -> List[ArtifactRecord]:
    """
    Get all transitive dependencies in topological order.

    Args:
        artifact_id: The artifact ID to resolve dependencies for

    Returns:
        List of ArtifactRecords in dependency order
    """
```

#### Cleanup

```python
def find_orphaned_artifacts(self) -> List[str]:
    """
    Find artifacts not referenced by any manifest.

    Returns:
        List of orphaned file paths
    """
```

```python
def delete_orphaned_artifacts(
    self,
    dry_run: bool = True
) -> Tuple[List[str], int]:
    """
    Delete orphaned artifacts.

    Args:
        dry_run: If True, only report what would be deleted

    Returns:
        Tuple of (deleted_paths, bytes_freed)
    """
```

```python
def cleanup_failed_run(self) -> int:
    """
    Clean up artifacts from the current run (for error recovery).

    Returns:
        Number of artifacts deleted
    """
```

```python
def purge_dataset_artifacts(
    self,
    confirm: bool = False
) -> Tuple[int, int]:
    """
    Delete ALL artifacts for this dataset.

    Args:
        confirm: Must be True to proceed (safety check)

    Returns:
        Tuple of (files_deleted, bytes_freed)

    Raises:
        ValueError: If confirm is False
    """
```

```python
def get_stats(self) -> Dict[str, Any]:
    """
    Get storage statistics.

    Returns:
        Dict with keys:
        - files_on_disk: int
        - registered_refs: int
        - deduplication_ratio: float
        - orphaned_count: int
        - orphaned_bytes: int
        - total_bytes: int
    """
```

#### Lifecycle

```python
def start_run(self) -> None:
    """Mark start of a training run (for cleanup tracking)."""

def end_run(self) -> None:
    """Mark successful end of a training run."""
```

---

### ArtifactLoader

```python
class ArtifactLoader:
    """Load artifacts from manifest and binaries with LRU caching."""
```

#### Constructor

```python
def __init__(
    self,
    workspace: Path,
    dataset: str,
    results_dir: Optional[Path] = None,
    cache_size: int = 100
) -> None:
    """
    Initialize loader.

    Args:
        workspace: Path to workspace root
        dataset: Dataset name
        results_dir: Optional path to run results directory
        cache_size: Maximum number of artifacts in LRU cache
    """
```

#### Import

```python
def import_from_manifest(self, manifest: Dict[str, Any]) -> None:
    """
    Import artifact records from a manifest.

    Args:
        manifest: Parsed manifest dict with artifacts.items

    Notes:
        Handles both v1 and v2 manifest formats.
    """
```

#### Loading

```python
def load_by_id(self, artifact_id: str) -> Any:
    """
    Load artifact by ID with caching.

    Args:
        artifact_id: The artifact ID to load

    Returns:
        Deserialized artifact object

    Raises:
        KeyError: If artifact_id not found
        FileNotFoundError: If binary file missing
    """
```

```python
def load_for_step(
    self,
    step_index: int,
    branch_path: Optional[List[int]] = None,
    fold_id: Optional[int] = None
) -> List[Tuple[str, Any]]:
    """
    Load all artifacts for a step context.

    Args:
        step_index: Pipeline step index
        branch_path: Optional branch path filter
        fold_id: Optional fold ID filter

    Returns:
        List of (artifact_id, artifact) tuples
    """
```

```python
def load_fold_models(
    self,
    step_index: int,
    branch_path: Optional[List[int]] = None
) -> List[Tuple[int, Any]]:
    """
    Load all fold-specific models for CV averaging.

    Args:
        step_index: Pipeline step index
        branch_path: Optional branch path filter

    Returns:
        List of (fold_id, model) tuples sorted by fold_id
    """
```

```python
def load_with_dependencies(self, artifact_id: str) -> Dict[str, Any]:
    """
    Load artifact and all its dependencies.

    Args:
        artifact_id: The artifact ID to load

    Returns:
        Dict mapping artifact_id to loaded object
        (in topological order for dependencies)
    """
```

```python
def load_meta_model_with_sources(
    self,
    artifact_id: str,
    validate_branch: bool = False
) -> Tuple[Any, List[Tuple[str, Any]], List[str]]:
    """
    Load meta-model with source models.

    Args:
        artifact_id: Meta-model artifact ID
        validate_branch: If True, verify source models match meta branch

    Returns:
        Tuple of:
        - meta_model: The loaded meta-model
        - sources: List of (artifact_id, model) for source models
        - feature_columns: List of feature column names

    Raises:
        KeyError: If artifact_id not found
        ValueError: If not a meta-model artifact
    """
```

#### Cache Management

```python
def preload_artifacts(
    self,
    artifact_ids: Optional[List[str]] = None
) -> int:
    """
    Warm cache with artifacts.

    Args:
        artifact_ids: IDs to preload, or None for all

    Returns:
        Number of artifacts loaded
    """
```

```python
def set_cache_size(self, new_size: int) -> None:
    """
    Dynamically resize cache.

    Args:
        new_size: New maximum cache size
    """
```

```python
def clear_cache(self) -> None:
    """Clear the artifact cache."""
```

```python
def get_cache_info(self) -> Dict[str, Any]:
    """
    Get cache statistics.

    Returns:
        Dict with keys:
        - size: Current number of cached items
        - max_size: Maximum cache size
        - hits: Number of cache hits
        - misses: Number of cache misses
        - hit_rate: hits / (hits + misses)
    """
```

---

## Module: `nirs4all.pipeline.storage.artifacts.utils`

### ID Functions

```python
def generate_artifact_id(
    pipeline_id: str,
    branch_path: List[int],
    step_index: int,
    fold_id: Optional[int] = None
) -> str:
    """
    Generate artifact ID from components.

    Format: <pipeline_id>:<branch0>:<branch1>:...<step_index>:<fold_id|all>
    """
```

```python
def parse_artifact_id(artifact_id: str) -> Dict[str, Any]:
    """
    Parse artifact ID into components.

    Returns:
        Dict with keys: pipeline_id, branch_path, step_index, fold_id
    """
```

### Filename Functions

```python
def generate_artifact_filename(
    artifact_type: ArtifactType,
    class_name: str,
    content_hash: str,
    format: str = "joblib"
) -> str:
    """
    Generate filename for artifact storage.

    Format: <type>_<class_name>_<short_hash>.<ext>
    Example: "model_PLSRegression_a1b2c3d4.joblib"
    """
```

```python
def parse_artifact_filename(filename: str) -> Dict[str, Any]:
    """
    Parse artifact filename into components.

    Returns:
        Dict with keys: artifact_type, class_name, short_hash, format
    """
```

### Hash Functions

```python
def compute_content_hash(obj: Any) -> str:
    """
    Compute SHA256 hash of serialized object.

    Returns:
        Hash string prefixed with "sha256:"
    """
```

```python
def verify_content_hash(path: Path, expected_hash: str) -> bool:
    """
    Verify file matches expected hash.

    Returns:
        True if hash matches, False otherwise
    """
```

---

## See Also

- [Artifacts System v2](../reference/artifacts_system_v2.md) - User guide
- [Manifest Specification](../specifications/manifest.md) - Schema reference
- [Workspace API](./workspace.md) - Workspace management API
