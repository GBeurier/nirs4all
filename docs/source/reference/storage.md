# Storage API Reference

This document provides the API reference for the nirs4all artifact storage system (V3).

The V3 artifacts system uses **operator chains** for complete execution path tracking, enabling deterministic artifact IDs that work correctly with branching, multi-source, stacking, and cross-validation.

## Module: `nirs4all.pipeline.storage.artifacts`

### Types and Enums

#### ArtifactType

```python
class ArtifactType(str, Enum):
    """Classification of artifact types.

    Each type has specific handling:
    - model: Trained ML models (sklearn, tensorflow, pytorch, etc.)
    - transformer: Fitted preprocessors (scalers, feature extractors)
    - splitter: Train/test split configuration (for reproducibility)
    - encoder: Label encoders, y-scalers
    - meta_model: Stacking meta-models with source model dependencies
    """

    MODEL = "model"
    TRANSFORMER = "transformer"
    SPLITTER = "splitter"
    ENCODER = "encoder"
    META_MODEL = "meta_model"
```

**Usage:**
```python
from nirs4all.pipeline.storage.artifacts import ArtifactType

record = registry.register_with_chain(
    model, chain, ArtifactType.MODEL, step_index=3
)
```

---

#### MetaModelConfig

```python
@dataclass
class MetaModelConfig:
    """Configuration for meta-model source tracking.

    Stores the ordered source models that feed into a stacking meta-model,
    along with their feature column mapping.

    Attributes:
        source_models: Ordered list of source model artifact IDs with feature indices
        feature_columns: Feature column names in the meta-model input order
    """

    source_models: List[Dict[str, Any]] = field(default_factory=list)
    feature_columns: List[str] = field(default_factory=list)
```

**Fields:**
- `source_models`: List of dicts with `artifact_id` and `feature_index` keys
- `feature_columns`: Ordered list of feature column names in meta-model input

**Serialization:**
```python
config = MetaModelConfig(
    source_models=[
        {"artifact_id": "0001_pls$abc123:all", "feature_index": 0},
        {"artifact_id": "0001_rf$def456:all", "feature_index": 1},
    ],
    feature_columns=["PLSRegression_pred", "RandomForest_pred"]
)
config_dict = config.to_dict()
config_back = MetaModelConfig.from_dict(config_dict)
```

---

#### ArtifactRecord

```python
@dataclass
class ArtifactRecord:
    """Complete artifact metadata record (V3).

    V3 Format:
        artifact_id: "{pipeline_id}${chain_hash}:{fold_id}"
        chain_path: Full operator chain path string
    """
```

**Core Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `artifact_id` | `str` | **V3 format**: `{pipeline_id}${chain_hash}:{fold_id}` |
| `content_hash` | `str` | SHA256 hash for deduplication |
| `path` | `str` | Relative path to artifact file in binaries/ |
| `chain_path` | `str` | **V3**: Full operator chain path (e.g., `s1.MinMaxScaler>s3.PLS[br=0]`) |
| `source_index` | `Optional[int]` | **V3**: Multi-source index (None for single source) |
| `pipeline_id` | `str` | Pipeline identifier (e.g., `0001_pls_abc123`) |
| `branch_path` | `List[int]` | Branch indices (empty for shared) |
| `step_index` | `int` | Pipeline step index |
| `substep_index` | `Optional[int]` | Substep for `[model1, model2]` syntax |
| `fold_id` | `Optional[int]` | CV fold index (None for shared) |
| `artifact_type` | `ArtifactType` | Type classification |
| `class_name` | `str` | Python class name of artifact |
| `custom_name` | `str` | User-defined name for the artifact |
| `depends_on` | `List[str]` | Dependency artifact IDs |
| `meta_config` | `Optional[MetaModelConfig]` | Meta-model configuration |
| `format` | `str` | Serialization format (`joblib`, `pickle`, `keras`, etc.) |
| `format_version` | `str` | Library version (e.g., `sklearn==1.5.0`) |
| `nirs4all_version` | `str` | nirs4all version at creation |
| `size_bytes` | `int` | File size in bytes |
| `created_at` | `str` | ISO8601 timestamp |
| `params` | `Dict[str, Any]` | Model hyperparameters |
| `version` | `int` | Schema version (3 for V3) |

**Properties:**

```python
record.short_hash          # str: First 12 chars of content_hash
record.chain_hash          # str: Chain hash from artifact ID
record.is_branch_specific  # bool: True if branch_path is non-empty
record.is_fold_specific    # bool: True if fold_id is not None
record.is_source_specific  # bool: True if source_index is not None
record.is_meta_model       # bool: True if artifact_type is META_MODEL
```

**Methods:**

```python
record.get_branch_path_str()  # str: Colon-separated branch indices
record.get_fold_str()         # str: Fold ID or "all"
record.matches_context(step_index=3, branch_path=[0], fold_id=1)  # bool

record_dict = record.to_dict()
record_back = ArtifactRecord.from_dict(record_dict)
```

---

### DependencyGraph

```python
class DependencyGraph:
    """Tracks artifact dependencies for stacking and transfer.

    Maintains a directed graph where edges represent "depends on" relationships.
    Supports transitive dependency resolution with cycle detection.
    """
```

**Methods:**

```python
def __init__(self) -> None:
    """Initialize empty dependency graph."""

def add_dependency(self, artifact_id: str, depends_on: str) -> None:
    """Add a dependency relationship."""

def add_dependencies(self, artifact_id: str, depends_on: List[str]) -> None:
    """Add multiple dependencies at once."""

def get_dependencies(self, artifact_id: str) -> List[str]:
    """Get direct dependencies of an artifact."""

def get_dependents(self, artifact_id: str) -> List[str]:
    """Get artifacts that directly depend on this artifact."""

def resolve_dependencies(
    self,
    artifact_id: str,
    max_depth: int = 100
) -> List[str]:
    """Get all transitive dependencies (topologically sorted)."""

def remove_artifact(self, artifact_id: str) -> None:
    """Remove an artifact and its edges from the graph."""

def clear(self) -> None:
    """Clear all dependencies."""
```

**Usage:**
```python
graph = DependencyGraph()
graph.add_dependency("meta_model", "model_a")
graph.add_dependency("meta_model", "model_b")

# Get load order (dependencies first)
order = graph.resolve_dependencies("meta_model")
# ["model_a", "model_b"]
```

---

### ArtifactRegistry

```python
class ArtifactRegistry:
    """Central registry for artifact management (V3).

    Provides:
    - Chain-based ID generation for complete execution path tracking
    - Content-addressed storage with deduplication
    - Dependency graph for stacking/transfer
    - Cleanup utilities
    """
```

#### Constructor

```python
def __init__(
    self,
    workspace: Path,
    dataset: str,
    manifest_manager: Optional[Any] = None,
    pipeline_id: str = ""
) -> None:
    """
    Initialize registry for a dataset.

    Args:
        workspace: Path to workspace root
        dataset: Dataset name
        manifest_manager: Optional ManifestManager for manifest updates
        pipeline_id: Pipeline identifier for V3 ID generation
    """
```

#### ID Generation (V3)

```python
def generate_id(
    self,
    chain: Union[OperatorChain, str],
    fold_id: Optional[int] = None,
    pipeline_id: Optional[str] = None
) -> str:
    """
    Generate deterministic V3 artifact ID from operator chain.

    V3 Format: {pipeline_id}${chain_hash}:{fold_id}

    Args:
        chain: OperatorChain or chain path string
        fold_id: CV fold (None for shared)
        pipeline_id: Pipeline identifier (uses self.pipeline_id if None)

    Returns:
        V3 Artifact ID string

    Examples:
        >>> registry.generate_id(chain, fold_id=0)
        '0001_pls$a1b2c3d4e5f6:0'
        >>> registry.generate_id("s1.MinMaxScaler>s3.PLS", fold_id=None)
        '0001_pls$7f8e9d0c1b2a:all'
    """
```

#### Registration (V3 Chain-Based)

```python
def register_with_chain(
    self,
    obj: Any,
    chain: Union[OperatorChain, str],
    artifact_type: ArtifactType,
    step_index: int,
    branch_path: Optional[List[int]] = None,
    source_index: Optional[int] = None,
    fold_id: Optional[int] = None,
    substep_index: Optional[int] = None,
    depends_on: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    meta_config: Optional[MetaModelConfig] = None,
    format_hint: Optional[str] = None,
    custom_name: Optional[str] = None,
    pipeline_id: Optional[str] = None
) -> ArtifactRecord:
    """
    Register and persist an artifact using V3 chain-based identification.

    This is the primary registration method for V3. It generates a deterministic
    artifact ID from the operator chain and stores the chain path for later lookup.

    Args:
        obj: Object to persist (model, transformer, etc.)
        chain: OperatorChain or chain path string
        artifact_type: Classification (model, transformer, etc.)
        step_index: Pipeline step index (1-based)
        branch_path: List of branch indices (empty for non-branching)
        source_index: Multi-source index (None for single source)
        fold_id: CV fold (None for shared artifacts)
        substep_index: Substep index for [model1, model2]
        depends_on: List of artifact IDs this depends on
        params: Model parameters for inspection
        meta_config: Meta-model configuration (for stacking)
        format_hint: Optional serialization format hint
        custom_name: User-defined name for the artifact
        pipeline_id: Override pipeline ID

    Returns:
        ArtifactRecord with full metadata

    Raises:
        ValueError: If object cannot be serialized or if meta-model
            dependencies are missing
    """
```

#### Registration (Legacy Compatibility)

```python
def register(
    self,
    obj: Any,
    artifact_id: str,
    artifact_type: ArtifactType,
    depends_on: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    meta_config: Optional[MetaModelConfig] = None,
    format_hint: Optional[str] = None,
    custom_name: Optional[str] = None,
    chain_path: str = "",
    source_index: Optional[int] = None
) -> ArtifactRecord:
    """
    Register and persist an artifact with pre-generated ID.

    Note: For new code, use register_with_chain() which generates IDs from OperatorChain.

    Args:
        obj: Object to persist (model, transformer, etc.)
        artifact_id: Pre-generated artifact ID (V3 format: pipeline$hash:fold)
        artifact_type: Classification (model, transformer, etc.)
        depends_on: List of artifact IDs this depends on
        params: Model parameters for inspection
        meta_config: Meta-model configuration (for stacking)
        format_hint: Optional serialization format hint
        custom_name: User-defined name for the artifact
        chain_path: V3 operator chain path (required for full traceability)
        source_index: Multi-source index (None for single source)

    Returns:
        ArtifactRecord with full metadata
    """
```

```python
def register_meta_model(
    self,
    obj: Any,
    artifact_id: str,
    source_model_ids: List[str],
    feature_columns: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    format_hint: Optional[str] = None
) -> ArtifactRecord:
    """
    Register a stacking meta-model with source model references.

    Automatically creates MetaModelConfig with ordered source model references,
    sets up dependency tracking, and validates source models exist.

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
    """Resolve artifact ID to record."""

def resolve_by_hash(self, content_hash: str) -> Optional[ArtifactRecord]:
    """Resolve content hash to artifact record."""

def get_by_chain(
    self,
    chain: Union[OperatorChain, str],
    fold_id: Optional[int] = None
) -> Optional[ArtifactRecord]:
    """Get artifact by exact chain path match (V3)."""

def get_chain_prefix(
    self,
    prefix: str,
    branch_path: Optional[List[int]] = None,
    source_index: Optional[int] = None
) -> List[ArtifactRecord]:
    """Get all artifacts whose chain path starts with the given prefix (V3)."""

def get_dependencies(self, artifact_id: str) -> List[str]:
    """Get direct dependencies of an artifact."""

def resolve_dependencies(self, artifact_id: str) -> List[ArtifactRecord]:
    """Get all transitive dependencies as records (topological order)."""

def get_artifacts_for_step(
    self,
    pipeline_id: str,
    step_index: int,
    branch_path: Optional[List[int]] = None,
    fold_id: Optional[int] = None
) -> List[ArtifactRecord]:
    """Get all artifacts for a specific step context."""

def get_fold_models(
    self,
    pipeline_id: str,
    step_index: int,
    branch_path: Optional[List[int]] = None
) -> List[ArtifactRecord]:
    """Get all fold-specific model artifacts for CV averaging."""
```

#### Loading

```python
def load_artifact(self, record: ArtifactRecord) -> Any:
    """Load artifact binary from disk."""
```

#### Import/Export

```python
def import_from_manifest(
    self,
    manifest: Dict[str, Any],
    results_dir: Path
) -> None:
    """Import artifact records from a manifest."""

def export_to_manifest(self) -> Dict[str, Any]:
    """Export registry to manifest V3 format."""

def get_all_records(self) -> List[ArtifactRecord]:
    """Get all registered artifacts."""
```

#### Cleanup

```python
def find_orphaned_artifacts(self, scan_all_manifests: bool = True) -> List[str]:
    """Find artifacts not referenced by any manifest."""

def delete_orphaned_artifacts(
    self,
    dry_run: bool = True,
    scan_all_manifests: bool = True
) -> Tuple[List[str], int]:
    """Delete orphaned artifacts. Returns (deleted_files, bytes_freed)."""

def delete_pipeline_artifacts(
    self,
    pipeline_id: str,
    delete_files: bool = False
) -> int:
    """Delete all artifacts for a specific pipeline."""

def cleanup_failed_run(self) -> int:
    """Clean up artifacts from a failed run."""

def purge_dataset_artifacts(
    self,
    confirm: bool = False
) -> Tuple[int, int]:
    """Delete ALL artifacts for this dataset. Returns (files_deleted, bytes_freed)."""

def get_stats(self, scan_all_manifests: bool = True) -> Dict[str, Any]:
    """
    Get storage statistics.

    Returns:
        Dict with keys:
        - total_artifacts: Number of registered artifacts
        - unique_files: Number of unique binary files
        - total_size_bytes: Total size of all artifacts
        - deduplication_ratio: Ratio of saved space
        - by_type: Count by artifact type
        - orphaned_count: Number of orphaned files
        - disk_usage_bytes: Actual disk usage
    """
```

#### Lifecycle

```python
def start_run(self) -> None:
    """Start tracking a new run for cleanup purposes."""

def end_run(self) -> None:
    """End run tracking (successful completion)."""
```

---

### ArtifactLoader

```python
class ArtifactLoader:
    """Load artifacts using V3 chain-based identification.

    Supports:
    - Direct loading by V3 artifact ID (pipeline$hash:fold)
    - Chain path-based loading for deterministic replay
    - Context-based loading (step/branch/source/fold)
    - Dependency resolution for stacking meta-models
    - Per-fold model loading for cross-validation ensemble
    - LRU caching to avoid redundant I/O
    """
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

#### Factory Methods

```python
@classmethod
def from_manifest(
    cls,
    manifest: Dict[str, Any],
    results_dir: Path
) -> 'ArtifactLoader':
    """Create an ArtifactLoader from a pipeline manifest."""
```

#### V3 Chain-Based Loading

```python
def load_by_chain(
    self,
    chain: str,
    fold_id: Optional[int] = None
) -> Optional[Any]:
    """Load artifact by exact chain path match."""

def load_by_chain_prefix(
    self,
    prefix: str,
    branch_path: Optional[List[int]] = None,
    source_index: Optional[int] = None
) -> List[Tuple[str, Any]]:
    """Load all artifacts whose chain path starts with the given prefix."""

def get_record_by_chain(self, chain_path: str) -> Optional[ArtifactRecord]:
    """Get artifact record by chain path."""
```

#### Primary Loading Methods

```python
def load_by_id(self, artifact_id: str) -> Any:
    """
    Load a single artifact by its V3 ID.

    Uses LRU cache to avoid redundant disk I/O.

    Args:
        artifact_id: V3 artifact identifier (pipeline$hash:fold)

    Returns:
        Deserialized artifact object

    Raises:
        KeyError: If artifact_id not found
        FileNotFoundError: If artifact file missing
    """

def load_for_step(
    self,
    step_index: int,
    branch_path: Optional[List[int]] = None,
    source_index: Optional[int] = None,
    fold_id: Optional[int] = None,
    pipeline_id: Optional[str] = None
) -> List[Tuple[str, Any]]:
    """Load all artifacts for a step context."""

def load_with_dependencies(self, artifact_id: str) -> Dict[str, Any]:
    """Load artifact and all transitive dependencies (topological order)."""

def load_fold_models(
    self,
    step_index: int,
    branch_path: Optional[List[int]] = None,
    pipeline_id: Optional[str] = None
) -> List[Tuple[int, Any]]:
    """Load all fold-specific models for CV averaging, sorted by fold_id."""

def load_meta_model_with_sources(
    self,
    artifact_id: str,
    validate_branch: bool = True
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

#### Artifact ID-Based Loading

```python
def load_by_artifact_id(self, artifact_id: str) -> Tuple[str, Any]:
    """
    Load a single artifact by its deterministic artifact_id.

    Returns:
        Tuple of (name, loaded_object) where name is built from
        custom_name if available, otherwise from class_name.
    """

def get_step_binaries_by_artifact_ids(
    self,
    artifact_ids: List[str]
) -> List[Tuple[str, Any]]:
    """Load multiple artifacts by their deterministic artifact_ids."""

def find_artifact_by_custom_name(
    self,
    custom_name: str,
    step_index: Optional[int] = None,
    fold_id: Optional[int] = None,
    branch_path: Optional[List[int]] = None
) -> Optional[ArtifactRecord]:
    """Find an artifact by its custom_name (reverse lookup)."""
```

#### Legacy Compatibility

```python
def get_step_binaries(
    self,
    step_id: int,
    branch_id: Optional[int] = None,
    branch_path: Optional[List[int]] = None
) -> List[Tuple[str, Any]]:
    """Legacy-compatible method for loading step binaries."""

def has_binaries_for_step(
    self,
    step_number: int,
    substep_number: Optional[int] = None,
    branch_id: Optional[int] = None
) -> bool:
    """Check if binaries exist for a specific step."""
```

#### Cache Management

```python
def preload_artifacts(
    self,
    artifact_ids: Optional[List[str]] = None,
    artifact_types: Optional[List[ArtifactType]] = None
) -> int:
    """Warm cache with artifacts. Returns number loaded."""

def set_cache_size(self, new_size: int) -> None:
    """Dynamically resize cache."""

def clear_cache(self) -> None:
    """Clear the artifact cache."""

def get_cache_info(self) -> Dict[str, Any]:
    """
    Get cache statistics.

    Returns:
        Dict with keys:
        - cached_count: Current number of cached items
        - max_size: Maximum cache size
        - hits, misses: Cache hit/miss counts
        - hit_rate: hits / (hits + misses)
        - total_artifacts: Total artifacts registered
        - artifacts_by_type: Count by type
    """
```

---

## Module: `nirs4all.pipeline.storage.artifacts.utils`

### V3 ID Functions

```python
def generate_artifact_id_v3(
    pipeline_id: str,
    chain: Union[OperatorChain, str],
    fold_id: Optional[int] = None
) -> str:
    """Generate V3 artifact ID from chain.

    Format: {pipeline_id}${chain_hash}:{fold_id}
    """

def parse_artifact_id_v3(artifact_id: str) -> Tuple[str, str, Optional[int]]:
    """Parse V3 artifact ID into (pipeline_id, chain_hash, fold_id)."""

def is_v3_artifact_id(artifact_id: str) -> bool:
    """Check if artifact ID is V3 format (contains '$')."""

def compute_chain_hash(chain: Union[OperatorChain, str]) -> str:
    """Compute deterministic hash from chain path."""
```

### Filename Functions

```python
def generate_filename(
    artifact_type: str,
    class_name: str,
    content_hash: str,
    extension: str = "joblib"
) -> str:
    """
    Generate filename for artifact storage.

    Format: <type>_<class_name>_<short_hash>.<ext>
    Example: "model_PLSRegression_a1b2c3d4e5f6.joblib"
    """

def parse_filename(filename: str) -> Optional[Tuple[str, str, str]]:
    """Parse artifact filename into (artifact_type, class_name, short_hash)."""
```

### Hash Functions

```python
def compute_content_hash(content: bytes) -> str:
    """Compute SHA256 hash of binary content. Returns "sha256:..." format."""

def get_short_hash(content_hash: str, length: int = 12) -> str:
    """Extract short hash from full content hash."""
```

### Path Functions

```python
def get_binaries_path(workspace: Path, dataset: str) -> Path:
    """Get centralized binaries directory for a dataset.

    Returns: workspace/binaries/<dataset>/
    """
```

### Validation

```python
def validate_artifact_id(artifact_id: str) -> bool:
    """Validate artifact ID format (V3 only)."""

def extract_pipeline_id_from_artifact_id(artifact_id: str) -> str:
    """Extract pipeline ID from artifact ID."""

def extract_fold_id_from_artifact_id(artifact_id: str) -> Optional[int]:
    """Extract fold ID from artifact ID (None if "all")."""
```

---

## See Also

- [Workspace Architecture](./workspace.md) - Workspace directory structure
- [Developer Guide: Artifacts](/developer/artifacts) - User guide for artifacts
- [Pipeline Syntax](/reference/pipeline_syntax) - Pipeline configuration reference
