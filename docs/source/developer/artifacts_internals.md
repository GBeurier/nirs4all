# Artifacts Developer Guide

This guide covers internal implementation details and extension points for the nirs4all artifacts system. For user-facing documentation, see {doc}`artifacts`.

## Architecture Overview

```
nirs4all/pipeline/storage/artifacts/
├── __init__.py           # Public API exports
├── types.py              # ArtifactType, ArtifactRecord, MetaModelConfig
├── registry.py           # ArtifactRegistry (registration, deduplication)
├── loader.py             # ArtifactLoader (loading, caching)
├── graph.py              # DependencyGraph (topological sorting)
└── utils.py              # ID generation, hashing, filename utilities
```

## Adding New Artifact Types

### Step 1: Add to ArtifactType Enum

Edit `nirs4all/pipeline/storage/artifacts/types.py`:

```python
class ArtifactType(Enum):
    """Classification of artifact types."""

    MODEL = "model"
    TRANSFORMER = "transformer"
    SPLITTER = "splitter"
    ENCODER = "encoder"
    META_MODEL = "meta_model"
    # Add new type here:
    FEATURE_SELECTOR = "feature_selector"  # New!
```

### Step 2: Update Serialization Detection

Edit `nirs4all/utils/serializer.py`:

```python
def _detect_artifact_type(obj: Any) -> str:
    """Detect artifact type from object."""
    if hasattr(obj, 'get_support') and hasattr(obj, 'fit_transform'):
        return 'feature_selector'
    # ... existing detection ...
```

### Step 3: Update Pipeline Controllers

Ensure the relevant controller uses the new type:

```python
record = registry.register(
    obj=fitted_selector,
    artifact_id=artifact_id,
    artifact_type=ArtifactType.FEATURE_SELECTOR,
)
```

### Step 4: Add Unit Tests

```python
def test_feature_selector_type():
    """Test FEATURE_SELECTOR type serialization."""
    assert ArtifactType.FEATURE_SELECTOR.value == "feature_selector"

    record = ArtifactRecord(
        artifact_id="0001:0:all",
        content_hash="sha256:abc123",
        path="feature_selector_SelectKBest_abc123.pkl",
        pipeline_id="0001",
        artifact_type=ArtifactType.FEATURE_SELECTOR,
        class_name="SelectKBest"
    )

    d = record.to_dict()
    assert d["artifact_type"] == "feature_selector"
```

## Custom Serialization Formats

### Adding a New Format

1. **Add format handler** in `nirs4all/utils/serializer.py`:

```python
def _save_onnx(obj, path: Path) -> None:
    """Save ONNX model."""
    import onnx
    onnx.save(obj, str(path))

def _load_onnx(path: Path) -> Any:
    """Load ONNX model."""
    import onnx
    return onnx.load(str(path))

FORMAT_HANDLERS = {
    # ... existing handlers ...
    "onnx": (_save_onnx, _load_onnx, ".onnx"),
}
```

2. **Update format detection**:

```python
def _detect_format(obj: Any) -> str:
    """Detect best serialization format."""
    if hasattr(obj, 'SerializeToString'):  # ONNX models
        return "onnx"
    # ... existing detection ...
```

## Dependency Graph Implementation

### Topological Sorting

The `DependencyGraph` ensures artifacts are loaded in the correct order:

```python
class DependencyGraph:
    """DAG for artifact dependencies."""

    def __init__(self):
        self._nodes: Set[str] = set()
        self._edges: Dict[str, Set[str]] = defaultdict(set)

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add dependency: from_node depends on to_node."""
        self._nodes.add(from_node)
        self._nodes.add(to_node)
        self._edges[from_node].add(to_node)

    def topological_sort(self) -> List[str]:
        """Return nodes in dependency order (Kahn's algorithm)."""
        in_degree = {node: 0 for node in self._nodes}
        for node, deps in self._edges.items():
            for dep in deps:
                in_degree[dep] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for dependent in self._get_dependents(node):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._nodes):
            raise ValueError("Cycle detected in dependency graph")

        return result
```

### Cycle Detection

The registry prevents cycles when registering dependencies:

```python
def register(self, ..., depends_on: List[str] = None):
    """Register artifact with dependency validation."""
    if depends_on:
        # Check all dependencies exist
        for dep_id in depends_on:
            if not self.resolve(dep_id):
                raise ValueError(f"Unknown dependency: {dep_id}")

        # Check for cycles
        temp_graph = self._graph.copy()
        for dep_id in depends_on:
            temp_graph.add_edge(artifact_id, dep_id)

        try:
            temp_graph.topological_sort()
        except ValueError:
            raise ValueError(f"Cycle detected when adding {artifact_id}")
```

## LRU Cache Implementation

```python
class LRUCache:
    """Least Recently Used cache with statistics."""

    def __init__(self, max_size: int = 100):
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get item, moving to end (most recent)."""
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """Add item, evicting oldest if at capacity."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = value
```

## Testing Patterns

### Unit Test Structure

```python
# tests/unit/pipeline/storage/artifacts/test_registry.py

import pytest
from pathlib import Path
from nirs4all.pipeline.storage.artifacts import (
    ArtifactRegistry,
    ArtifactType,
)

@pytest.fixture
def temp_workspace(tmp_path):
    """Create temporary workspace structure."""
    binaries = tmp_path / "workspace" / "binaries" / "test_dataset"
    binaries.mkdir(parents=True)
    return tmp_path / "workspace"

@pytest.fixture
def registry(temp_workspace):
    """Create registry for testing."""
    return ArtifactRegistry(temp_workspace, "test_dataset")

class TestArtifactRegistry:
    def test_register_model(self, registry):
        """Test basic model registration."""
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit([[1], [2]], [1, 2])

        record = registry.register(
            obj=model,
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.MODEL
        )

        assert record.artifact_id == "0001:0:all"
        assert record.artifact_type == ArtifactType.MODEL
        assert record.class_name == "LinearRegression"
        assert record.content_hash.startswith("sha256:")

    def test_deduplication(self, registry):
        """Test that identical objects share same file."""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit([[1, 2], [3, 4]])

        record1 = registry.register(scaler, "0001:0:all", ArtifactType.TRANSFORMER)
        record2 = registry.register(scaler, "0002:0:all", ArtifactType.TRANSFORMER)

        assert record1.content_hash == record2.content_hash
        assert record1.path == record2.path
```

### Integration Test Structure

```python
# tests/integration/artifacts/test_artifact_flow.py

class TestArtifactFlow:
    """Integration tests for training → prediction artifact flow."""

    @pytest.fixture
    def sample_pipeline(self):
        return [
            {"ShuffleSplit": {"n_splits": 2, "test_size": 0.2}},
            {"MinMaxScaler": {}},
            {"PLSRegression": {"n_components": 3}},
        ]

    def test_train_and_predict(self, temp_workspace, sample_pipeline, sample_data):
        """Test artifacts from training can be loaded for prediction."""
        runner = PipelineRunner(
            workspace=temp_workspace,
            save_artifacts=True
        )

        predictions, outputs = runner.run(sample_pipeline, sample_data)
        manifest = outputs.manifest

        loader = ArtifactLoader(temp_workspace, sample_data["dataset"])
        loader.import_from_manifest(manifest)

        for artifact_id in loader.list_artifact_ids():
            obj = loader.load_by_id(artifact_id)
            assert obj is not None
```

## Debugging Tips

### Inspecting Artifacts

```python
from nirs4all.pipeline.storage.artifacts import ArtifactRegistry

registry = ArtifactRegistry(workspace_path, dataset)

for artifact_id, record in registry._records.items():
    print(f"{artifact_id}: {record.class_name} ({record.artifact_type.value})")
    print(f"  Hash: {record.short_hash}")
    print(f"  Dependencies: {record.depends_on}")
```

### Cache Statistics

```python
loader = ArtifactLoader(workspace, dataset)
loader.import_from_manifest(manifest)

# Load some artifacts...
loader.load_by_id("0001:0:all")
loader.load_by_id("0001:0:all")  # Cache hit

info = loader.get_cache_info()
print(f"Cache size: {info['size']}/{info['max_size']}")
print(f"Hit rate: {info['hit_rate']:.1%}")
```

### Orphan Detection

```python
registry = ArtifactRegistry(workspace, dataset)

orphans = registry.find_orphaned_artifacts()
print(f"Orphaned files: {len(orphans)}")

stats = registry.get_stats()
print(f"Files on disk: {stats['files_on_disk']}")
print(f"Deduplication: {stats['deduplication_ratio']:.1%}")
```

## CLI Implementation

The artifacts CLI commands are in `nirs4all/cli/commands/artifacts.py`:

```python
import click
from pathlib import Path
from nirs4all.pipeline.storage.artifacts import ArtifactRegistry

@click.group(name="artifacts")
def artifacts_cli():
    """Manage pipeline artifacts."""
    pass

@artifacts_cli.command()
@click.option("--dataset", required=True, help="Dataset name")
@click.option("--workspace", default="workspace", help="Workspace path")
def stats(dataset: str, workspace: str):
    """Show artifact storage statistics."""
    registry = ArtifactRegistry(Path(workspace), dataset)
    stats = registry.get_stats()

    click.echo(f"Files on disk:     {stats['files_on_disk']}")
    click.echo(f"Registered refs:   {stats['registered_refs']}")
    click.echo(f"Deduplication:     {stats['deduplication_ratio']:.1%}")

@artifacts_cli.command(name="cleanup")
@click.option("--dataset", required=True)
@click.option("--workspace", default="workspace")
@click.option("--force", is_flag=True, help="Actually delete files")
def cleanup(dataset: str, workspace: str, force: bool):
    """Delete orphaned artifacts."""
    registry = ArtifactRegistry(Path(workspace), dataset)
    deleted, bytes_freed = registry.delete_orphaned_artifacts(dry_run=not force)

    if force:
        click.echo(f"Deleted {len(deleted)} files, freed {bytes_freed / 1024:.1f} KB")
    else:
        click.echo(f"Would delete {len(deleted)} files")
        click.echo("Run with --force to delete")
```

## ArtifactRecord Reference

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

## See Also

- {doc}`artifacts` - User guide for artifacts system
- {doc}`/reference/storage` - Storage API reference
- {doc}`architecture` - Pipeline architecture overview
