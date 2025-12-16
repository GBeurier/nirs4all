# Developer Guide: Artifacts System

This guide covers internal implementation details and extension points for the
nirs4all artifacts system.

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

Edit `nirs4all/utils/serializer.py` to handle the new type:

```python
def _detect_artifact_type(obj: Any) -> str:
    """Detect artifact type from object."""
    # Add detection logic for new type
    if hasattr(obj, 'get_support') and hasattr(obj, 'fit_transform'):
        # sklearn feature selectors
        return 'feature_selector'
    # ... existing detection ...
```

### Step 3: Update Pipeline Controllers

Ensure the relevant controller uses the new type when registering:

```python
# In the controller that handles feature selection
record = registry.register(
    obj=fitted_selector,
    artifact_id=artifact_id,
    artifact_type=ArtifactType.FEATURE_SELECTOR,  # Use new type
)
```

### Step 4: Add Unit Tests

Create tests in `tests/unit/pipeline/storage/artifacts/test_types.py`:

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

    restored = ArtifactRecord.from_dict(d)
    assert restored.artifact_type == ArtifactType.FEATURE_SELECTOR
```

## Extending the Registry

### Custom Serialization Formats

To add a new serialization format:

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

# Register in FORMAT_HANDLERS
FORMAT_HANDLERS = {
    # ... existing handlers ...
    "onnx": (_save_onnx, _load_onnx, ".onnx"),
}
```

2. **Update format detection** in `_detect_format()`:

```python
def _detect_format(obj: Any) -> str:
    """Detect best serialization format."""
    if hasattr(obj, 'SerializeToString'):  # ONNX models
        return "onnx"
    # ... existing detection ...
```

### Custom Hash Functions

The default uses SHA256. To use a different hash:

```python
# In utils.py
def compute_content_hash(obj: Any, algorithm: str = "sha256") -> str:
    """Compute hash of serialized object."""
    import hashlib

    # Serialize to bytes
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    content = buf.getvalue()

    # Compute hash
    hasher = hashlib.new(algorithm)
    hasher.update(content)
    return f"{algorithm}:{hasher.hexdigest()}"
```

## Understanding the Dependency Graph

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
        """Return nodes in dependency order using Kahn's algorithm."""
        # Calculate in-degree for each node
        in_degree = {node: 0 for node in self._nodes}
        for node, deps in self._edges.items():
            for dep in deps:
                in_degree[dep] += 1

        # Start with nodes that have no dependencies
        queue = [n for n, d in in_degree.items() if d == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # Reduce in-degree for dependent nodes
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
        temp_graph.add_node(artifact_id)
        for dep_id in depends_on:
            temp_graph.add_edge(artifact_id, dep_id)

        try:
            temp_graph.topological_sort()
        except ValueError:
            raise ValueError(f"Cycle detected when adding {artifact_id}")
```

## LRU Cache Implementation

The `ArtifactLoader` uses an LRU cache for performance:

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
                self._cache.popitem(last=False)  # Remove oldest
            self._cache[key] = value

    def resize(self, new_size: int) -> None:
        """Resize cache, evicting if needed."""
        self._max_size = new_size
        while len(self._cache) > new_size:
            self._cache.popitem(last=False)
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
    ArtifactRecord
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

import pytest
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.pipeline.storage.artifacts import ArtifactLoader

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
        """Test that artifacts from training can be loaded for prediction."""
        runner = PipelineRunner(
            workspace=temp_workspace,
            save_files=True
        )

        # Train
        predictions, outputs = runner.run(sample_pipeline, sample_data)

        # Get manifest
        manifest = outputs.manifest

        # Load artifacts for prediction
        loader = ArtifactLoader(temp_workspace, sample_data["dataset"])
        loader.import_from_manifest(manifest)

        # Verify all artifacts loadable
        for artifact_id in loader.list_artifact_ids():
            obj = loader.load_by_id(artifact_id)
            assert obj is not None
```

## Debugging Tips

### Inspecting Artifacts

```python
from nirs4all.pipeline.storage.artifacts import ArtifactRegistry

registry = ArtifactRegistry(workspace_path, dataset)

# List all registered artifacts
for artifact_id, record in registry._records.items():
    print(f"{artifact_id}: {record.class_name} ({record.artifact_type.value})")
    print(f"  Hash: {record.short_hash}")
    print(f"  Dependencies: {record.depends_on}")
```

### Cache Statistics

```python
from nirs4all.pipeline.storage.artifacts import ArtifactLoader

loader = ArtifactLoader(workspace, dataset)
loader.import_from_manifest(manifest)

# Load some artifacts...
loader.load_by_id("0001:0:all")
loader.load_by_id("0001:1:all")
loader.load_by_id("0001:0:all")  # Cache hit

# Check cache stats
info = loader.get_cache_info()
print(f"Cache size: {info['size']}/{info['max_size']}")
print(f"Hit rate: {info['hit_rate']:.1%}")
```

### Orphan Detection

```python
registry = ArtifactRegistry(workspace, dataset)

# Find orphans
orphans = registry.find_orphaned_artifacts()
print(f"Orphaned files: {len(orphans)}")

# Get detailed stats
stats = registry.get_stats()
print(f"Files on disk: {stats['files_on_disk']}")
print(f"Registered refs: {stats['registered_refs']}")
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
    click.echo(f"Orphaned:          {stats['orphaned_count']} files "
               f"({stats['orphaned_bytes'] / 1024:.1f} KB)")

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
        click.echo(f"Would delete {len(deleted)} files ({bytes_freed / 1024:.1f} KB)")
        click.echo("Run with --force to delete")
```

## Related Documentation

- [Artifacts System v2](../reference/artifacts_system_v2.md) - User guide
- [Storage API](../api/storage.md) - API reference
- [Manifest Specification](../specifications/manifest.md) - Schema docs
