# Virtual Data Management System Design

## Overview

This document presents a complete design for a virtual data management system in nirs4all where **all operations are views on immutable base data**. Only explicit duplication during branching creates actual copies.

**Core Principle**: "The dataset keeps all transformations, augmentations, sources, and the logic of feature/sample organization - providing views on demand with the right structure (merge/subset/etc.)"

---

## 1. Problem Statement

### Current Architecture Issues

| Problem | Current Behavior | Impact |
|---------|------------------|--------|
| **Branching copies** | `copy.deepcopy(dataset._features.sources)` in `branch.py:1517` | O(n_branches × data_size) memory |
| **Merge destroys data** | `dataset._features.sources = merged_data` | Original sources lost permanently |
| **Eager evaluation** | All transforms computed immediately | No caching, repeated work |
| **No lineage tracking** | Processing names encode history ("raw_SNV_001") | Limited traceability, hard to debug |

### Memory Impact Example

```
Dataset: 500 samples × 1000 features × float64 = 4 MB
Pipeline with 10 branches:
  - Current: 10 × deepcopy = 40 MB (+ copies for each post-branch step)
  - Proposed: 4 MB base + views (near-zero overhead)
```

---

## 2. Design Principles

### 2.1 Immutability by Default

- **Raw data is never mutated** after creation
- All "modifications" create new logical references, not physical copies
- Base data enforced read-only: `data.flags.writeable = False`

### 2.2 Lazy Evaluation

- **Transforms are recorded, not executed** until data is accessed
- A transform is a node in a DAG, not an immediate computation
- Computation happens at "materialization boundaries" (model training, export)

### 2.3 Views, Not Copies

- **Branching creates view specifications**, not data copies
- Merging creates a virtual view combining multiple branches
- Only explicit `mode="duplicate"` creates actual copies

### 2.4 Composability

- Views can be composed: `view.with_partition("train").with_fold(0)`
- Composition is lazy - specifications accumulate, computation deferred

---

## 3. Architecture

### 3.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DatasetContext                            │
│    (bundles everything for one pipeline execution)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │FeatureBlockStore│  │  SampleRegistry │  │   TargetStore   │  │
│  │                 │  │  (Indexer)      │  │                 │  │
│  │ Immutable CoW   │  │                 │  │  y versions:    │  │
│  │ blocks with     │  │  Polars DF:     │  │  - "raw"        │  │
│  │ lineage hash    │  │  - sample_id    │  │  - "scaled"     │  │
│  │                 │  │  - partition    │  │  - "encoded"    │  │
│  │ Content-        │  │  - fold_id      │  │                 │  │
│  │ addressed       │  │  - tags         │  │  Fitted         │  │
│  │ deduplication   │  │  - excluded     │  │  transformers   │  │
│  │                 │  │  - origin       │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    TransformGraph (DAG)                      ││
│  │                                                              ││
│  │  Records all transforms as nodes WITHOUT computing them      ││
│  │  Each node: parent_ids, transform_spec, cached_result?       ││
│  │                                                              ││
│  │  [Block_raw] ──SNV──> [Node_snv] ──PLS──> [Node_pls]         ││
│  │       └──────MSC──> [Node_msc] ──PLS──> [Node_pls2]          ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      ViewResolver                            ││
│  │                                                              ││
│  │  ViewSpec (declarative) → lazy slice → materialized array    ││
│  │                                                              ││
│  │  1. Check cache                                              ││
│  │  2. Walk transform graph                                     ││
│  │  3. Compute only what's needed                               ││
│  │  4. Cache result if beneficial                               ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  MaterializationCache                        ││
│  │                                                              ││
│  │  LRU cache for computed views                                ││
│  │  Key: hash(ViewSpec + layout + transform_lineage)            ││
│  │  Eviction: memory pressure, LRU, entry count                 ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

```
User Request: dataset.x(partition="train", fold_id=0)
                           │
                           ▼
                    ┌─────────────┐
                    │  ViewSpec   │  Declarative: what to access
                    │  created    │  (no computation yet)
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ ViewResolver│  Checks cache first
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
        Cache Hit?                 Cache Miss
              │                         │
              ▼                         ▼
        Return cached         Walk TransformGraph
                                       │
                                       ▼
                              Materialize needed
                              transforms only
                                       │
                                       ▼
                              Cache result
                                       │
                                       ▼
                              Return array
```

---

## 4. Core Components

### 4.1 FeatureBlock

**Purpose**: Immutable storage unit for spectral data with lineage tracking.

```
FeatureBlock
├── block_id: str           # Content-addressed hash (unique identifier)
├── _data: np.ndarray       # 3D: (samples, processings, features) - READ ONLY
├── headers: Tuple[str]     # Wavelengths/feature names (immutable)
├── header_unit: str        # "nm", "cm-1", "text", "index"
├── processing_ids: Tuple   # Names for each processing slot
├── parent_id: Optional[str]# Lineage: who created this block
├── transform_info: Optional# What transform created this block
├── lineage_hash: str       # Deterministic hash of parent + transform
├── _ref_count: int         # For garbage collection
└── _is_cow: bool           # Is this sharing memory with another block?
```

**Key Properties**:
- Data array is ALWAYS read-only (`data.flags.writeable = False`)
- Content-addressed: same data → same block_id (deduplication)
- Lineage hash enables cache lookup by transformation history

**Copy-on-Write (CoW) Semantics**:
```
                    Original Block
                    ┌──────────────┐
                    │ data: [...]  │  ref_count = 1
                    └──────────────┘
                           │
          After derive() with CoW optimization:
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  │                  ▼
   Original Block          │            New Block
   ┌──────────────┐        │       ┌──────────────┐
   │ data: [...]  │◄───────┴───────│ data: VIEW   │
   │ ref_count=2  │   SHARED       │ _is_cow=True │
   └──────────────┘   MEMORY       └──────────────┘
                      (until mutation)
```

### 4.2 FeatureBlockStore

**Purpose**: Central registry for all blocks with lifecycle management.

```
FeatureBlockStore
├── _blocks: Dict[block_id → FeatureBlock]
├── _lineage_index: Dict[lineage_hash → block_id]  # For deduplication
├── _parent_index: Dict[parent_id → Set[child_ids]] # For lineage traversal
└── _lock: RLock  # Thread safety

Methods:
├── register_source(data, headers, ...) → block_id
│   └── Creates new source block, deduplicates if same content
├── register_transform(parent_id, data, transform_info) → block_id
│   └── Creates derived block, links to parent
├── register_slice(parent_id, sample_indices, feature_indices) → block_id
│   └── Creates sliced view (virtual source)
├── get(block_id) → FeatureBlock
├── get_by_lineage(lineage_hash) → Optional[FeatureBlock]
├── get_lineage(block_id) → List[FeatureBlock]  # Full chain to root
├── release(block_id)  # Decrement ref_count
└── gc() → int  # Remove blocks with ref_count=0
```

**Deduplication Flow**:
```
register_source(data) called
         │
         ▼
   Compute data_hash
         │
         ▼
   Compute lineage_hash
         │
         ▼
   lineage_hash in _lineage_index?
         │
    ┌────┴────┐
    │         │
   YES       NO
    │         │
    ▼         ▼
 Return    Create new block
 existing  Register in both indexes
 block_id  Return new block_id
```

### 4.3 TransformGraph (DAG)

**Purpose**: Records all transforms as a directed acyclic graph WITHOUT executing them.

```
TransformNode
├── node_id: str
├── parent_ids: Tuple[str]      # Input blocks or nodes
├── transform_spec: TransformSpec
│   ├── transform_class: str    # "sklearn.preprocessing.MinMaxScaler"
│   ├── transform_params: Dict  # Constructor params (frozen)
│   └── fit_params: Dict        # Params used during fit
├── sample_indices: Optional    # For partitioned transforms
├── feature_indices: Optional   # For virtual sources
├── _cached_block_id: Optional  # If already materialized
└── _fitted_state: Optional     # Fitted transformer object

TransformGraph
├── _nodes: Dict[node_id → TransformNode]
├── _roots: Set[str]  # Nodes with no parents (source blocks)

Methods:
├── add_transform(parent_id, transform_spec) → node_id
│   └── Records transform WITHOUT executing
├── get_lineage(node_id) → List[TransformNode]
├── is_materialized(node_id) → bool
├── materialize(node_id, resolver) → block_id
└── validate_acyclic()  # Prevent circular dependencies
```

**Example Graph**:
```
Pipeline: [SNV(), {"branch": [[PLS(5)], [PLS(10)]]}, {"merge": "predictions"}]

Graph:
                    ┌─────────────┐
                    │ Block: raw  │ (source)
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Node: SNV   │ transform_spec = SNV()
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       ┌─────────────┐           ┌─────────────┐
       │ Node: PLS_5 │           │ Node: PLS_10│
       │ branch=[0]  │           │ branch=[1]  │
       └──────┬──────┘           └──────┬──────┘
              │                         │
              └────────────┬────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Node: Merge │ virtual concatenation
                    └─────────────┘
```

### 4.4 ViewSpec

**Purpose**: Declarative, immutable specification of what data to access.

```
ViewSpec (frozen dataclass, hashable)
├── block_ids: Tuple[str]           # Which blocks to include
├── transform_node_id: Optional[str] # Or a transform node (lazy)
├── virtual_source_names: Tuple[str] # Named feature subsets
│
├── sample_indices: Optional[Tuple[int]]  # Explicit sample filter
├── partition: Optional[str]        # "train", "test", "all"
├── fold_id: Optional[int]          # CV fold number
├── fold_role: Optional[str]        # "train" or "val"
├── branch_path: Tuple[int]         # e.g., (0, 1) for nested branches
├── processing_filter: Optional[Tuple[str]]
│
├── include_augmented: bool = True
├── include_excluded: bool = False
├── layout: str = "2d"
└── concat_sources: bool = True

Methods:
├── with_partition(p) → ViewSpec    # Immutable update
├── with_fold(id, role) → ViewSpec
├── with_branch(path) → ViewSpec
├── with_samples(indices) → ViewSpec
├── compose(other) → ViewSpec       # View of view
├── cache_key() → str               # For caching
└── is_empty() → bool
```

**Composition Example**:
```python
# Build view incrementally
base = ViewSpec(block_ids=("nir_block",))
train = base.with_partition("train")
fold0 = train.with_fold(0, "train")
branch0 = fold0.with_branch((0,))

# Or compose
final = base.compose(ViewSpec(partition="train"))
            .compose(ViewSpec(fold_id=0, fold_role="train"))
            .compose(ViewSpec(branch_path=(0,)))
```

### 4.5 VirtualSourceSpec

**Purpose**: Define feature subsets (wavelength regions) without physical copies.

```
VirtualSourceSpec (frozen dataclass)
├── parent_block_id: str
├── name: str                       # "VIS", "NIR1", "protein_region"
├── feature_indices: Tuple[int]     # Which features to include
│   OR
├── wavelength_range: Tuple[float, float]  # (start_nm, end_nm)
├── header_subset: Tuple[str]       # Corresponding headers

Methods:
├── get_view(block_store) → np.ndarray
│   └── Returns parent.data[:, :, feature_indices]  # numpy VIEW, not copy
└── materialize(block_store) → FeatureBlock
    └── Creates actual sliced block if needed
```

**Virtual Region System**:
```
Original Block: 2100 features (400-2500 nm)
┌────────────────────────────────────────────────────────────────┐
│ data: (100, 2, 2100)                                           │
└────────────────────────────────────────────────────────────────┘
                               │
                   No data copied - just indices stored
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
VirtualSourceSpec:       VirtualSourceSpec:    VirtualSourceSpec:
name="VIS"              name="NIR1"           name="NIR2"
indices=[0:300]         indices=[300:700]     indices=[700:2100]
(400-700 nm)            (700-1100 nm)         (1100-2500 nm)
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                               ▼
                     On access: return VIEW
                     parent.data[:, :, indices]
```

### 4.6 ViewResolver

**Purpose**: Lazy evaluation engine that materializes views on demand.

```
ViewResolver
├── _block_store: FeatureBlockStore
├── _transform_graph: TransformGraph
├── _sample_registry: SampleRegistry (Indexer)
├── _cache: MaterializationCache
└── _pending: Dict[str, Future]  # For async resolution

Methods:
├── resolve(view: ViewSpec, layout="2d") → np.ndarray
│   └── Main entry point - triggers lazy evaluation
├── _check_cache(view) → Optional[np.ndarray]
├── _resolve_sample_indices(view) → np.ndarray
├── _materialize_transform(node_id, indices) → np.ndarray
├── _materialize_virtual_sources(specs, indices) → np.ndarray
└── _materialize_blocks(block_ids, indices) → np.ndarray
```

**Resolution Algorithm**:
```
resolve(view: ViewSpec) → np.ndarray:

    1. CACHE CHECK
       cache_key = view.cache_key() + layout
       if cache_key in cache:
           return cache[cache_key]

    2. RESOLVE SAMPLE INDICES
       indices = sample_registry.query(
           partition=view.partition,
           fold_id=view.fold_id,
           fold_role=view.fold_role,
           include_augmented=view.include_augmented,
           include_excluded=view.include_excluded,
           branch_path=view.branch_path
       )
       if view.sample_indices:
           indices = intersect(indices, view.sample_indices)

    3. MATERIALIZE DATA
       if view.transform_node_id:
           # Lazy transform - walk graph
           data = materialize_transform(view.transform_node_id, indices)
       elif view.virtual_source_names:
           # Virtual sources - return views
           data = materialize_virtual_sources(view.virtual_source_names, indices)
       else:
           # Direct block access
           data = materialize_blocks(view.block_ids, indices)

    4. APPLY LAYOUT
       if layout == "2d" and data.ndim == 3:
           data = data.reshape(data.shape[0], -1)

    5. CACHE IF BENEFICIAL
       if should_cache(data, computation_time):
           cache[cache_key] = data

    6. RETURN
       return data
```

**Transform Materialization (Recursive)**:
```
materialize_transform(node_id, sample_indices) → np.ndarray:

    node = transform_graph.get(node_id)

    # Already computed?
    if node._cached_block_id:
        block = block_store.get(node._cached_block_id)
        return block.data[sample_indices]

    # Recursively materialize parents
    parent_data = []
    for parent_id in node.parent_ids:
        if is_transform_node(parent_id):
            parent_data.append(materialize_transform(parent_id, sample_indices))
        else:
            # It's a block_id
            block = block_store.get(parent_id)
            parent_data.append(block.data[sample_indices])

    # Apply transform
    input_data = parent_data[0] if len(parent_data) == 1 else parent_data

    if node._fitted_state:
        # Prediction mode - use fitted transformer
        result = node._fitted_state.transform(input_data)
    else:
        # Training mode - fit and transform
        transformer = instantiate(node.transform_spec)
        result = transformer.fit_transform(input_data)
        node._fitted_state = transformer

    # Cache intermediate result if frequently accessed
    if should_cache_intermediate(node_id):
        block_id = block_store.register_transform(node.parent_ids[0], result, ...)
        node._cached_block_id = block_id

    return result
```

### 4.7 MaterializationCache

**Purpose**: LRU cache for materialized views with memory pressure handling.

```
CacheConfig
├── max_memory_mb: float = 500.0
├── max_entries: int = 100
├── eviction_policy: str = "lru"
├── cache_threshold_ms: float = 10.0   # Only cache if computation > 10ms
└── cache_max_size_mb: float = 100.0   # Don't cache arrays > 100MB

MaterializationCache
├── _config: CacheConfig
├── _cache: OrderedDict[key → (np.ndarray, size_mb)]
├── _total_size_mb: float
├── _lock: RLock
├── _hits: int
├── _misses: int

Methods:
├── get(key) → Optional[np.ndarray]
├── put(key, data) → None
├── invalidate_prefix(prefix) → int
├── clear() → None
└── stats() → Dict
```

**Eviction Strategy**:
```
put(key, data):
    size_mb = data.nbytes / (1024^2)

    # Don't cache if too large
    if size_mb > config.cache_max_size_mb:
        return

    # Evict until there's room
    while total_size + size_mb > max_memory_mb OR len(cache) >= max_entries:
        oldest_key = cache.popitem(last=False)  # LRU: remove oldest
        total_size -= oldest_size

    # Store
    cache[key] = (data, size_mb)
    total_size += size_mb
```

---

## 5. Branching Model

### 5.1 Implicit Branching (Default: View-Based)

**Current approach**:
```python
# branch.py:1515-1519
def _snapshot_features(self, dataset):
    return copy.deepcopy(dataset._features.sources)  # FULL COPY
```

**Proposed approach**:
```python
# No data copy - just ViewSpec creation
def _create_branch_view(self, base_view: ViewSpec, branch_id: int) -> ViewSpec:
    return base_view.with_branch(branch_id)
```

**Diagram - Current vs Proposed**:
```
CURRENT (Hard Duplication):
┌──────────────┐
│  Base Data   │  4 MB
└──────┬───────┘
       │ deepcopy
       ├─────────────────┬─────────────────┐
       ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Branch 0   │  │   Branch 1   │  │   Branch 2   │
│    4 MB      │  │    4 MB      │  │    4 MB      │
└──────────────┘  └──────────────┘  └──────────────┘
                        Total: 16 MB


PROPOSED (View-Based):
┌──────────────┐
│  Base Data   │  4 MB (immutable)
└──────┬───────┘
       │ ViewSpec
       ├─────────────────┬─────────────────┐
       ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  ViewSpec    │  │  ViewSpec    │  │  ViewSpec    │
│  branch=[0]  │  │  branch=[1]  │  │  branch=[2]  │
│  ~100 bytes  │  │  ~100 bytes  │  │  ~100 bytes  │
└──────────────┘  └──────────────┘  └──────────────┘
                        Total: 4 MB + 300 bytes
```

### 5.2 Explicit Duplication

For cases where actual copies are needed (rare):

```python
# Pipeline syntax
{"branch": [...], "mode": "duplicate"}

# Implementation
def _create_branch_with_duplication(self, base_view, branch_id):
    # Force materialization and create new block
    data = resolver.resolve(base_view, force_compute=True)
    new_block_id = block_store.register_source(
        data.copy(),  # EXPLICIT COPY
        metadata={"duplicated_for_branch": branch_id}
    )
    return ViewSpec(block_ids=(new_block_id,))
```

### 5.3 Virtual Merge

**Current merge** destroys original sources:
```python
# merge.py - current behavior
dataset._features.sources = merged_data  # ORIGINAL LOST
```

**Proposed virtual merge**:
```
After branching:
Branch 0: ViewSpec(transform_node_id="snv_pls_0", branch=[0])
Branch 1: ViewSpec(transform_node_id="msc_pls_1", branch=[1])

After {"merge": "features"}:
MergeViewSpec:
  - child_views: [branch_0_view, branch_1_view]
  - mode: "concat_features"

Original branches STILL ACCESSIBLE:
  dataset.x(branch=[0])  # Returns branch 0 data
  dataset.x(branch=[1])  # Returns branch 1 data
  dataset.x()            # Returns merged data (computed lazily)
```

---

## 6. API Design

### 6.1 Backward Compatible X() Interface

The existing `dataset.x()` API remains unchanged:

```python
# Existing code continues to work
X = dataset.x(partition="train")
X_3d = dataset.x(partition="train", layout="3d")
X_fold = dataset.x(partition="train", fold_id=0)
```

**Internal change**: `x()` now builds a ViewSpec and calls ViewResolver:

```python
def x(self, partition="all", layout="2d", fold_id=None, **kwargs):
    view = ViewSpec(
        block_ids=self._current_block_ids,
        partition=partition,
        fold_id=fold_id,
        layout=layout,
        **kwargs
    )
    return self._resolver.resolve(view)
```

### 6.2 New View API

Explicit view creation for advanced use cases:

```python
# Create a lazy view
view = dataset.view(partition="train", fold_id=0)

# Views are lazy - no computation yet
X = view.x()    # Computation happens here
y = view.y()    # Same sample indices, no recomputation of index query

# View composition
full_view = (dataset.view()
    .with_partition("train")
    .with_fold(0)
    .with_branch([0, 1]))

# Multiple access from same view (efficient)
view = dataset.view(partition="train")
X_2d = view.x(layout="2d")
X_3d = view.x(layout="3d")  # Reuses resolved indices
```

### 6.3 Virtual Source Management API

```python
# Create wavelength regions (no data copy)
dataset.virtual.create_region(
    source="NIR",
    name="protein",
    start=1500,
    end=1800,
    unit="nm"
)
dataset.virtual.create_region("NIR", "water", 1400, 1450)

# Access by region name
X_protein = dataset.x(region="protein")   # View into NIR source
X_water = dataset.x(region="water")

# Original still accessible
X_full = dataset.x()

# List virtual sources
dataset.virtual.list()  # ["protein", "water"]

# Remove (trivial - just deletes VirtualSourceSpec)
dataset.virtual.remove("protein")
dataset.virtual.clear()  # Remove all
```

### 6.4 Branch API with Duplication Control

```python
# Default: view-based branching (NO copy)
pipeline = [
    {"branch": [
        [SNV(), PLSRegression(10)],
        [MSC(), PLSRegression(10)],
    ]},
    {"merge": "predictions"}
]

# Explicit duplication when needed (RARE)
pipeline = [
    {"branch": [
        [SNV(), PLSRegression(10)],
        [MSC(), PLSRegression(10)],
    ], "mode": "duplicate"},  # Creates actual copies
    {"merge": "predictions"}
]

# Access branch data after merge (still available!)
result = nirs4all.run(pipeline, dataset)
X_branch0 = dataset.x(branch=[0])  # First branch preprocessing
X_branch1 = dataset.x(branch=[1])  # Second branch preprocessing
X_merged = dataset.x()             # Merged result
```

---

## 7. Bottlenecks and Pitfalls

### 7.1 Transform Chain Explosion

**Problem**: Deep transform graphs can be slow to traverse during resolution.

```
Raw → SNV → SG → PLS → Detrend → Normalize → PCA → ...
      50 nodes deep = 50 recursive calls
```

**Mitigation**: Periodic materialization checkpoints

```python
# In TransformGraph
CHECKPOINT_INTERVAL = 10  # Materialize every 10 nodes

def add_transform(...):
    ...
    if len(self.get_lineage(node_id)) % CHECKPOINT_INTERVAL == 0:
        self._mark_for_checkpoint(node_id)

# During resolution, checkpointed nodes are pre-materialized
```

### 7.2 Cache Memory Pressure

**Problem**: Many views cached → memory exhaustion

**Mitigation**: Multi-level eviction strategy

```python
class MemoryPressureHandler:
    def check_and_handle(self):
        memory_usage = get_process_memory()

        if memory_usage > CRITICAL_THRESHOLD:
            cache.clear()              # Nuclear option
            gc.collect()
        elif memory_usage > WARNING_THRESHOLD:
            cache.evict_oldest(50%)    # Partial eviction
        elif memory_usage > SOFT_THRESHOLD:
            cache.evict_oldest(25%)    # Gentle eviction
```

### 7.3 Accidental Materialization in Loops

**Problem**: Repeated resolution in loops is expensive

```python
# BAD: Resolves view 5 times
for fold_id in range(5):
    X = dataset.x(partition="train", fold_id=fold_id)
    # ... training

# GOOD: Pre-build views
views = [dataset.view(partition="train", fold_id=i) for i in range(5)]
for view in views:
    X = view.x()  # Resolves with caching
```

**Documentation**: Clearly document this pattern.

### 7.4 In-Place Mutation

**Problem**: If base data is mutated, all views see wrong data

```python
# This MUST fail
block = store.get(block_id)
block.data[0, 0] = 999  # ValueError: assignment destination is read-only
```

**Mitigation**: Already handled by `data.flags.writeable = False`

### 7.5 Circular Dependencies in Transform Graph

**Problem**: Circular transforms cause infinite recursion

```python
# A depends on B, B depends on A
node_a = TransformNode(parent_ids=("node_b",), ...)
node_b = TransformNode(parent_ids=("node_a",), ...)
```

**Mitigation**: DAG validation on every `add_transform()`

```python
def add_transform(self, parent_id, ...):
    # Check for cycles before adding
    if self._would_create_cycle(parent_id, new_node_id):
        raise CircularDependencyError(...)
```

### 7.6 Stale Cache After Data Reload

**Problem**: External file changes, cache returns old data

```python
dataset.load("data.csv")
view = dataset.view()
X = view.x()  # Cached

# File changes externally...
dataset.reload()  # Should invalidate cache

X = view.x()  # Should return new data, not stale cache
```

**Mitigation**: Cache invalidation on reload

```python
def reload(self):
    self._cache.invalidate_all()
    self._load_data()
```

### 7.7 Complexity Tradeoffs Summary

| Aspect | Current (Simple) | Proposed (Virtual) |
|--------|------------------|-------------------|
| Mental Model | Direct arrays | Views + resolution |
| Debugging | Easy (inspect arrays) | Harder (trace graph) |
| Memory | High (copies) | Low (views) |
| CPU | Low (no resolution) | Higher (lazy eval) |
| Cache Management | None | Complex (eviction) |
| Branch Isolation | Natural (copies) | Requires discipline |

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Immutable Storage)

**Goal**: Create immutable block storage without breaking existing code.

**Components**:
- `FeatureBlock` with CoW semantics
- `FeatureBlockStore` registry
- Content-addressed hashing

**Files**:
- Create: `nirs4all/data/_virtual/base_block.py`
- Create: `nirs4all/data/_virtual/block_store.py`
- Modify: `nirs4all/data/dataset.py` (internal wiring only)

**Validation**: Existing `dataset.x()` still works, all tests pass.

### Phase 2: View System

**Goal**: Lazy evaluation engine.

**Components**:
- `ViewSpec` (frozen dataclass)
- `ViewResolver` with recursive materialization
- `MaterializationCache` with LRU eviction

**Files**:
- Create: `nirs4all/data/_virtual/view_spec.py`
- Create: `nirs4all/data/_virtual/view_resolver.py`
- Create: `nirs4all/data/_virtual/cache.py`

**Validation**: Views resolve correctly, caching works.

### Phase 3: Transform Graph

**Goal**: Lazy transform tracking.

**Components**:
- `TransformNode` and `TransformGraph`
- Integration with ViewResolver

**Files**:
- Create: `nirs4all/data/_virtual/transform_graph.py`

**Validation**: Transforms recorded without execution, lazy materialization works.

### Phase 4: Virtual Sources

**Goal**: Feature-level views.

**Components**:
- `VirtualSourceSpec`
- Wavelength region API
- Integration with existing source handling

**Files**:
- Create: `nirs4all/data/_virtual/virtual_source.py`

**Validation**: Regions created without copies, access returns views.

### Phase 5: View-Based Branching

**Goal**: Replace deepcopy branching with views.

**Changes**:
- Modify `BranchController` to use views by default
- Add `mode="duplicate"` for explicit copies
- Update `MergeController` for virtual merge
- Remove `_snapshot_features` / `_restore_features` pattern

**Files**:
- Modify: `nirs4all/controllers/data/branch.py` (major refactor)
- Modify: `nirs4all/controllers/data/merge.py`

**Validation**: Branching tests pass, memory usage reduced.

### Phase 6: Migration & Cleanup

**Goal**: Production-ready.

**Tasks**:
- Deprecate old patterns
- Update all controllers
- Performance optimization
- Full test suite
- Documentation

---

## 9. Validation Plan

### Unit Tests

```
tests/unit/data/_virtual/
├── test_feature_block.py      # Immutability, CoW, lineage
├── test_block_store.py        # Registration, deduplication, GC
├── test_view_spec.py          # Composition, hashing, serialization
├── test_transform_graph.py    # DAG operations, cycle detection
├── test_cache.py              # LRU eviction, memory limits
├── test_view_resolver.py      # Resolution, caching
└── test_virtual_source.py     # Region creation, views
```

### Integration Tests

```
tests/integration/pipeline/
├── test_virtual_branching.py  # View-based branching
├── test_virtual_merge.py      # Non-destructive merge
├── test_memory_usage.py       # Memory profiling
└── test_backward_compat.py    # All existing code works
```

### Memory Profiling

```bash
# Compare memory before/after for branching scenarios
python -m memory_profiler examples/developer/D01_branching_basics.py

# Expected: 80%+ memory reduction for multi-branch pipelines
```

---

## 10. Alignment with v2 Design

This design aligns with existing v2 design documents in `docs/_internal/nirs4all_v2_design/`:

| v2 Design Component | This Design |
|---------------------|-------------|
| `FeatureBlock` (02_data_layer.md) | ✓ Same concept, enhanced with CoW |
| `FeatureBlockStore` | ✓ Matches v2 registry pattern |
| `ViewSpec` | ✓ Extends v2 concept with composition |
| `ViewResolver` | ✓ Lazy evaluation as specified |
| `SampleRegistry` | ✓ Uses existing Indexer infrastructure |

**Key Additions**:
- `TransformGraph` for lazy transform tracking (not in v2)
- `VirtualSourceSpec` for feature-level views (requested in this task)
- Explicit `mode="duplicate"` for rare cases needing copies

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Block** | Immutable data container with lineage tracking |
| **View** | Lazy specification of data to access (no computation) |
| **Materialization** | Converting a view into actual numpy array |
| **CoW** | Copy-on-Write: share memory until mutation |
| **Lineage** | Chain of transformations from source to current state |
| **Virtual Source** | Feature subset defined by indices (no copy) |
| **Transform Node** | DAG node representing a recorded (not executed) transform |

---

## Appendix B: Decision Log

| Decision | Rationale | Alternative Considered |
|----------|-----------|------------------------|
| Content-addressed blocks | Enables deduplication | UUID-based IDs |
| Frozen dataclass for ViewSpec | Hashable for caching | Mutable with manual hash |
| DAG for transforms | Lazy evaluation, lineage | Eager execution |
| LRU cache eviction | Simple, effective | LFU, ARC |
| Default view-based branching | Memory efficiency | Always duplicate |

---

*Document version: 1.0*
*Created: 2025-01-20*
*Related documents:*
- `source_extraction_merge_analysis.md`
- `feature_based_branching_analysis.md`
- `docs/_internal/nirs4all_v2_design/02_data_layer.md`
