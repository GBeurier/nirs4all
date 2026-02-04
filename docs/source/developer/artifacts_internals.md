# Artifacts Developer Guide

This guide covers internal implementation details and extension points for the nirs4all artifacts system. For user-facing documentation, see {doc}`artifacts`.

## Architecture Overview

The storage system has two layers:

1. **WorkspaceStore** (`nirs4all.pipeline.storage.workspace_store`) -- DuckDB-backed central store for all structured data and artifact management.
2. **Artifacts subpackage** (`nirs4all.pipeline.storage.artifacts/`) -- Low-level artifact utilities (types, serialization, operator chains) used by WorkspaceStore and the execution engine.

```
nirs4all/pipeline/storage/
├── __init__.py               # Public API exports (WorkspaceStore, ChainBuilder, etc.)
├── workspace_store.py        # DuckDB-backed central store
├── store_schema.py           # DuckDB table schema creation
├── store_queries.py          # SQL query constants
├── store_protocol.py         # StoreProtocol abstract interface
├── chain_builder.py          # ChainBuilder (ExecutionTrace -> chain dict)
├── chain_replay.py           # Standalone replay_chain() function
├── library.py                # PipelineLibrary (template management)
└── artifacts/
    ├── __init__.py            # Subpackage exports
    ├── types.py               # ArtifactType enum
    ├── artifact_registry.py   # ArtifactRegistry (used by execution engine)
    ├── artifact_loader.py     # ArtifactLoader (loading, caching)
    ├── artifact_persistence.py # Low-level persist/load functions
    ├── operator_chain.py      # OperatorChain (execution path tracking)
    └── utils.py               # Hashing, filename, ID utilities
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

Ensure the relevant controller returns the artifact in `StepOutput`:

```python
return context, StepOutput(
    artifacts={"feature_selector": fitted_selector}
)
```

The execution engine will call `store.save_artifact()` with the appropriate type.

### Step 4: Add Unit Tests

```python
def test_feature_selector_artifact_roundtrip(tmp_path):
    """Test FEATURE_SELECTOR artifact save/load via WorkspaceStore."""
    from sklearn.feature_selection import SelectKBest
    from nirs4all.pipeline.storage import WorkspaceStore

    store = WorkspaceStore(tmp_path / "workspace")
    selector = SelectKBest(k=5)
    selector.fit([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], [0, 1])

    artifact_id = store.save_artifact(
        obj=selector,
        operator_class="sklearn.feature_selection.SelectKBest",
        artifact_type="feature_selector",
        format="joblib"
    )

    loaded = store.load_artifact(artifact_id)
    assert isinstance(loaded, SelectKBest)
    store.close()
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
# tests/unit/pipeline/storage/test_workspace_store.py

import pytest
from pathlib import Path
from nirs4all.pipeline.storage import WorkspaceStore

@pytest.fixture
def store(tmp_path):
    """Create a WorkspaceStore for testing."""
    s = WorkspaceStore(tmp_path / "workspace")
    yield s
    s.close()

class TestWorkspaceStore:
    def test_run_lifecycle(self, store):
        """Test basic run lifecycle."""
        run_id = store.begin_run("test", config={}, datasets=[])
        store.complete_run(run_id, summary={"total": 1})

        run = store.get_run(run_id)
        assert run is not None
        assert run["status"] == "completed"

    def test_artifact_deduplication(self, store):
        """Test that identical objects share the same file."""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        id1 = store.save_artifact(scaler, "StandardScaler", "transformer", "joblib")
        id2 = store.save_artifact(scaler, "StandardScaler", "transformer", "joblib")

        assert id1 == id2  # Content-addressed deduplication
```

### Integration Test Structure

```python
# tests/integration/artifacts/test_artifact_flow.py

class TestArtifactFlow:
    """Integration tests for training -> prediction artifact flow."""

    def test_train_creates_store_with_artifacts(self, workspace_path, dataset):
        """Test that training creates store.duckdb and artifacts."""
        runner = PipelineRunner(
            workspace_path=workspace_path,
            save_artifacts=True,
            verbose=0,
        )
        predictions, _ = runner.run(PipelineConfigs(pipeline), dataset)

        assert (workspace_path / "store.duckdb").exists()
        artifacts = list((workspace_path / "artifacts").rglob("*.joblib"))
        assert len(artifacts) >= 1
```

## Debugging Tips

### Inspecting the Store

```python
from nirs4all.pipeline.storage import WorkspaceStore

store = WorkspaceStore(workspace_path)

# List all runs
runs = store.list_runs()
print(runs)

# Get top predictions
top = store.top_predictions(n=5, metric="val_score")
print(top)

# Inspect a chain
chain = store.get_chain(chain_id)
print(f"Model: {chain['model_class']}")
print(f"Steps: {len(chain['steps'])}")
print(f"Fold artifacts: {chain['fold_artifacts']}")

store.close()
```

### DuckDB Direct Queries

For advanced debugging, query the DuckDB store directly:

```python
import duckdb

conn = duckdb.connect(str(workspace_path / "store.duckdb"))

# Count records
print(conn.execute("SELECT COUNT(*) FROM predictions").fetchone())

# Inspect artifacts
print(conn.execute(
    "SELECT artifact_id, operator_class, ref_count, size_bytes FROM artifacts"
).fetchall())

conn.close()
```

### Cleanup Diagnostics

```python
store = WorkspaceStore(workspace_path)

# Check for unreferenced artifacts
orphan_count = store.gc_artifacts()
print(f"Removed {orphan_count} orphaned artifacts")

# Reclaim disk space
store.vacuum()

store.close()
```

## See Also

- {doc}`artifacts` - User guide for artifacts system
- {doc}`/reference/storage` - Storage API reference
- {doc}`architecture` - Pipeline architecture overview
