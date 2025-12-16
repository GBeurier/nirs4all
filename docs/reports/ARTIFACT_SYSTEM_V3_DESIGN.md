# Artifact System V3 - Comprehensive Redesign

**Date**: December 15, 2025
**Author**: Technical Design
**Status**: Proposed Design - Ready for Review

---

## 1. Design Goals

### 1.1 Primary Objectives

1. **Complete Chain Tracking**: Every artifact knows its full execution path from input to output
2. **Deterministic Replay**: Same trace always produces identical artifact loading
3. **Unified Handling**: All edge cases (branching, multi-source, stacking, bundles) use the same mechanism
4. **Clean Separation**: Training persists, prediction loads - no ambiguity
5. **Backward Compatible**: Existing manifests can be migrated

### 1.2 Edge Cases That Must Work

| Case | Example | Current | V3 Target |
|------|---------|---------|-----------|
| Simple pipeline | `[Scaler, Split, Model]` | ✅ | ✅ |
| Multi-fold CV | `ShuffleSplit(n=5)` | ✅ | ✅ |
| Multi-source | 3 X arrays | ✅ Train, ❌ Reload | ✅ |
| Branching | `branch[[SNV], [MSC]]` | ⚠️ | ✅ |
| Branch + Multi-source | 2 branches × 3 sources | ❌ | ✅ |
| Nested branches | `branch[[..., branch[[...]]]]` | ❌ | ✅ |
| Subpipeline models | `[model1, model2]` | ⚠️ | ✅ |
| Meta-model stacking | `MetaModel(source="all")` | ⚠️ | ✅ |
| Loaded bundle | `load_bundle(...)` | ⚠️ | ✅ |
| Transfer learning | Freeze + finetune | ✅ | ✅ |

---

## 2. Core Concepts

### 2.1 Operator Path (New Concept)

The **Operator Path** is the fundamental identifier for any artifact. It encodes the complete chain of operators that produced this artifact.

```
OperatorPath := "{pipeline}:{chain}:{fold}"

Where:
  pipeline := pipeline_id (e.g., "0001_pls")
  chain := operator_chain (e.g., "s1.MinMaxScaler[0]>s3.SNV[0,1]>s4.PLS")
  fold := fold_id or "all"
```

**Examples:**

```
# Single transformer at step 1, source 0
0001_pls:s1.MinMaxScaler[src=0]:all

# Branched transformer at step 3, branch [0], source 1
0001_pls:s1.MinMaxScaler[src=0]>s3.SNV[br=0,src=1]:all

# Model at step 4, branch [0], fold 0
0001_pls:s1.MinMaxScaler>s3.SNV[br=0]>s4.PLS[br=0]:0

# Meta-model depending on branch predictions
0001_pls:s4.PLS[br=0]+s4.RF[br=1]>s5.Ridge:all

# Nested branch: branch[0] -> nested branch[1]
0001_pls:s3[br=0]>s3.1[br=1]>s4.PLS[br=0,1]:0
```

### 2.2 Operator Node (Building Block)

An `OperatorNode` represents one operator in the chain:

```python
@dataclass
class OperatorNode:
    step_index: int               # Pipeline step (1-based)
    substep_index: Optional[int]  # For nested operators (0-based)
    operator_class: str           # "MinMaxScaler", "PLS", etc.
    branch_path: List[int]        # Branch indices [0], [0, 1], etc.
    source_index: Optional[int]   # For multi-source transformers
    fold_id: Optional[int]        # For fold-specific models

    def to_key(self) -> str:
        """Compact key for this node."""
        parts = [f"s{self.step_index}"]
        if self.substep_index is not None:
            parts[0] += f".{self.substep_index}"
        parts.append(self.operator_class)

        attrs = []
        if self.branch_path:
            attrs.append(f"br={','.join(map(str, self.branch_path))}")
        if self.source_index is not None:
            attrs.append(f"src={self.source_index}")
        if attrs:
            parts.append(f"[{';'.join(attrs)}]")

        return "".join(parts)
```

### 2.3 Operator Chain (Full Path)

An `OperatorChain` is an ordered sequence of `OperatorNode` objects:

```python
@dataclass
class OperatorChain:
    nodes: List[OperatorNode]

    def to_path(self) -> str:
        """Full path string: node1>node2>node3"""
        return ">".join(node.to_key() for node in self.nodes)

    def append(self, node: OperatorNode) -> "OperatorChain":
        """Return new chain with node appended."""
        return OperatorChain(nodes=self.nodes + [node])

    def filter_branch(self, target_branch_path: List[int]) -> "OperatorChain":
        """Return chain with only nodes matching branch path."""
        filtered = [
            n for n in self.nodes
            if not n.branch_path or self._branch_matches(n.branch_path, target_branch_path)
        ]
        return OperatorChain(nodes=filtered)

    @staticmethod
    def _branch_matches(node_path: List[int], target_path: List[int]) -> bool:
        """Check if node's branch path is prefix of or equal to target."""
        return node_path == target_path[:len(node_path)]
```

---

## 3. Artifact ID V3 Format

### 3.1 New Format

```
{pipeline_id}${chain_hash}:{fold_id}

Where:
  pipeline_id := string (e.g., "0001_pls")
  chain_hash := sha256(chain_path)[:12] (12 hex chars)
  fold_id := integer or "all"
```

**Examples:**
```
0001_pls$a1b2c3d4e5f6:all    # Shared transformer
0001_pls$7f8e9d0c1b2a:0     # Fold 0 model
0001_pls$3c4d5e6f7a8b:1     # Fold 1 model
```

### 3.2 Rationale

1. **Deterministic**: Same chain → same hash
2. **Compact**: 12 chars enough for collision-free IDs within a pipeline
3. **Reversible**: Full chain stored in metadata, hash for quick lookup
4. **Source-agnostic**: Hash includes all discriminating info

### 3.3 Backward Compatibility

```python
def migrate_artifact_id(old_id: str) -> str:
    """Convert v2 ID to v3 format."""
    pipeline_id, branch_path, step_index, fold_id, sub_index = parse_v2_id(old_id)

    # Reconstruct chain from v2 components
    chain = OperatorChain(nodes=[
        OperatorNode(
            step_index=step_index,
            substep_index=sub_index,
            operator_class="Unknown",  # Will be enriched from manifest
            branch_path=branch_path,
            source_index=None,
            fold_id=fold_id
        )
    ])

    chain_hash = compute_chain_hash(chain.to_path())
    return f"{pipeline_id}${chain_hash}:{fold_id or 'all'}"
```

---

## 4. Revised Data Structures

### 4.1 ArtifactRecord V3

```python
@dataclass
class ArtifactRecordV3:
    # Identity
    artifact_id: str                    # V3 format: pipeline$hash:fold
    content_hash: str                   # SHA256 of binary content

    # Chain info (NEW)
    operator_chain: OperatorChain       # Full path to this artifact
    chain_path: str                     # Serialized chain: "s1.Scaler>s3.SNV[br=0]"

    # Operator info
    step_index: int                     # Pipeline step (1-based)
    substep_index: Optional[int]        # Nested operator index
    operator_class: str                 # Class name
    operator_name: str                  # Instance name (e.g., "MinMaxScaler_1")

    # Context info
    pipeline_id: str                    # Pipeline identifier
    branch_path: List[int]              # Branch indices
    source_index: Optional[int]         # Multi-source index (NEW)
    fold_id: Optional[int]              # Fold number or None for shared

    # Type and dependencies
    artifact_type: ArtifactType         # MODEL, TRANSFORMER, etc.
    depends_on: List[str]               # Artifact IDs this depends on

    # Storage
    path: str                           # Relative path in binaries dir
    format: str                         # "joblib", "h5", etc.
    format_version: str                 # Framework version
    size_bytes: int

    # Metadata
    params: Dict[str, Any]              # Operator params
    nirs4all_version: str
    created_at: str                     # ISO timestamp
    version: int                        # Record version (3)
```

### 4.2 ExecutionStep V3

```python
@dataclass
class ExecutionStepV3:
    # Identity
    step_id: str                        # Unique ID: "{trace_id}:step:{step_index}"
    step_index: int                     # Pipeline step (1-based)

    # Operator info
    operator_type: str                  # "transform", "model", "branch", etc.
    operator_class: str                 # Class name
    operator_config: Dict[str, Any]     # Original config

    # Chain tracking (NEW)
    input_chain: OperatorChain          # Chain up to this step's input
    output_chains: List[OperatorChain]  # Chains produced by this step

    # Branching (REVISED)
    parent_branch_path: List[int]       # Branch context entering this step
    produces_branches: bool             # True if this is a branch operator
    branch_outputs: Dict[int, "ExecutionStepV3"]  # Sub-steps per branch (NEW)

    # Multi-source (NEW)
    source_count: int                   # Number of X sources
    source_steps: Dict[int, "ExecutionStepV3"]  # Sub-steps per source

    # Execution
    execution_mode: StepExecutionMode   # TRAIN, PREDICT, SKIP
    artifacts: StepArtifactsV3          # Artifacts produced

    # Metadata
    duration_ms: float
    metadata: Dict[str, Any]
```

### 4.3 StepArtifacts V3

```python
@dataclass
class StepArtifactsV3:
    # Artifacts by role
    primary_artifacts: Dict[str, str]   # {chain_path: artifact_id}
    fold_artifacts: Dict[int, Dict[str, str]]  # {fold: {chain_path: artifact_id}}

    # Quick lookups
    all_artifact_ids: List[str]         # All artifact IDs at this step
    by_branch: Dict[Tuple[int, ...], List[str]]  # {branch_path: [artifact_ids]}
    by_source: Dict[int, List[str]]     # {source_idx: [artifact_ids]}

    def get_artifacts(
        self,
        branch_path: Optional[List[int]] = None,
        source_index: Optional[int] = None,
        fold_id: Optional[int] = None
    ) -> List[str]:
        """Get artifact IDs matching the given filters."""
        result = self.all_artifact_ids

        if branch_path is not None:
            branch_key = tuple(branch_path)
            result = [a for a in result if a in self.by_branch.get(branch_key, [])]

        if source_index is not None:
            result = [a for a in result if a in self.by_source.get(source_index, [])]

        if fold_id is not None:
            fold_chains = self.fold_artifacts.get(fold_id, {})
            result = [a for a in result if a in fold_chains.values()]

        return result
```

---

## 5. Revised Trace Recording

### 5.1 TraceRecorder V3

```python
class TraceRecorderV3:
    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.trace = ExecutionTraceV3(pipeline_id=pipeline_id)
        self._chain_stack: List[OperatorChain] = [OperatorChain(nodes=[])]
        self._branch_stack: List[List[int]] = [[]]

    def push_chain(self, node: OperatorNode):
        """Push new node onto the chain stack."""
        current = self._chain_stack[-1]
        extended = current.append(node)
        self._chain_stack.append(extended)

    def pop_chain(self) -> OperatorChain:
        """Pop and return the current chain."""
        return self._chain_stack.pop()

    def current_chain(self) -> OperatorChain:
        """Get current operator chain without modifying stack."""
        return self._chain_stack[-1]

    def enter_branch(self, branch_id: int):
        """Enter a branch context."""
        current = self._branch_stack[-1].copy()
        current.append(branch_id)
        self._branch_stack.append(current)

    def exit_branch(self) -> List[int]:
        """Exit current branch context."""
        return self._branch_stack.pop()

    def current_branch_path(self) -> List[int]:
        """Get current branch path."""
        return self._branch_stack[-1].copy()

    def start_step(
        self,
        step_index: int,
        operator_type: str,
        operator_class: str,
        operator_config: Dict[str, Any],
        source_index: Optional[int] = None,
        substep_index: Optional[int] = None
    ) -> ExecutionStepV3:
        """Start recording a step."""
        node = OperatorNode(
            step_index=step_index,
            substep_index=substep_index,
            operator_class=operator_class,
            branch_path=self.current_branch_path(),
            source_index=source_index,
            fold_id=None  # Set later if fold-specific
        )

        self.push_chain(node)

        step = ExecutionStepV3(
            step_index=step_index,
            operator_type=operator_type,
            operator_class=operator_class,
            operator_config=operator_config,
            input_chain=self._chain_stack[-2],  # Before push
            output_chains=[],
            parent_branch_path=self.current_branch_path(),
            source_count=1,
            artifacts=StepArtifactsV3()
        )

        self._current_step = step
        return step

    def record_artifact(
        self,
        artifact_id: str,
        chain: OperatorChain,
        is_primary: bool = False,
        fold_id: Optional[int] = None,
        branch_path: Optional[List[int]] = None,
        source_index: Optional[int] = None
    ):
        """Record an artifact for the current step."""
        step = self._current_step
        chain_path = chain.to_path()

        if fold_id is not None:
            if fold_id not in step.artifacts.fold_artifacts:
                step.artifacts.fold_artifacts[fold_id] = {}
            step.artifacts.fold_artifacts[fold_id][chain_path] = artifact_id
        elif is_primary:
            step.artifacts.primary_artifacts[chain_path] = artifact_id

        step.artifacts.all_artifact_ids.append(artifact_id)

        if branch_path:
            branch_key = tuple(branch_path)
            if branch_key not in step.artifacts.by_branch:
                step.artifacts.by_branch[branch_key] = []
            step.artifacts.by_branch[branch_key].append(artifact_id)

        if source_index is not None:
            if source_index not in step.artifacts.by_source:
                step.artifacts.by_source[source_index] = []
            step.artifacts.by_source[source_index].append(artifact_id)

        step.output_chains.append(chain)

    def end_step(self):
        """End current step and add to trace."""
        self.trace.add_step(self._current_step)
        self.pop_chain()
        self._current_step = None
```

### 5.2 Branch Recording (The Key Fix)

```python
class BranchControllerV3:
    def execute(self, step_info, dataset, context, runtime_context, ...):
        recorder = runtime_context.trace_recorder

        # Start parent branch step
        parent_step = recorder.start_step(
            step_index=runtime_context.step_number,
            operator_type="branch",
            operator_class="BranchOperator",
            operator_config=step_info.config
        )
        parent_step.produces_branches = True

        branch_defs = self._parse_branch_definitions(step_info)

        for branch_idx, branch_def in enumerate(branch_defs):
            # Enter branch context
            recorder.enter_branch(branch_idx)
            branch_path = recorder.current_branch_path()

            # Create sub-step record for this branch
            branch_step = ExecutionStepV3(
                step_index=runtime_context.step_number,
                operator_type="branch_content",
                operator_class=f"Branch_{branch_idx}",
                parent_branch_path=branch_path[:-1],  # Parent's path
                ...
            )

            for substep_idx, substep in enumerate(branch_def.steps):
                # Record each substep INDIVIDUALLY
                substep_step = recorder.start_step(
                    step_index=runtime_context.step_number,
                    substep_index=substep_idx,
                    operator_type=substep.operator_type,
                    operator_class=substep.operator_class,
                    operator_config=substep.config
                )

                # Execute substep
                result = runtime_context.step_runner.execute(substep, ...)

                # Record artifacts with full chain
                for artifact in result.artifacts:
                    recorder.record_artifact(
                        artifact_id=artifact.artifact_id,
                        chain=recorder.current_chain(),
                        branch_path=branch_path,
                        ...
                    )

                recorder.end_step()

            # Store branch sub-step in parent
            parent_step.branch_outputs[branch_idx] = branch_step

            # Exit branch context
            recorder.exit_branch()

        recorder.end_step()
```

---

## 6. Revised Artifact Registry

### 6.1 ArtifactRegistry V3

```python
class ArtifactRegistryV3:
    def __init__(self, binaries_dir: Path, pipeline_id: str):
        self.binaries_dir = binaries_dir
        self.pipeline_id = pipeline_id
        self._artifacts: Dict[str, ArtifactRecordV3] = {}
        self._by_chain: Dict[str, str] = {}  # chain_path -> artifact_id
        self._dependency_graph = DependencyGraphV3()

    def generate_id(self, chain: OperatorChain, fold_id: Optional[int] = None) -> str:
        """Generate deterministic artifact ID from chain."""
        chain_path = chain.to_path()
        chain_hash = hashlib.sha256(chain_path.encode()).hexdigest()[:12]
        fold_str = str(fold_id) if fold_id is not None else "all"
        return f"{self.pipeline_id}${chain_hash}:{fold_str}"

    def register(
        self,
        obj: Any,
        chain: OperatorChain,
        artifact_type: ArtifactType,
        fold_id: Optional[int] = None,
        depends_on: Optional[List[str]] = None,
        **metadata
    ) -> ArtifactRecordV3:
        """Register an artifact with full chain tracking."""
        artifact_id = self.generate_id(chain, fold_id)
        chain_path = chain.to_path()

        # Check for deduplication
        if chain_path in self._by_chain:
            existing_id = self._by_chain[chain_path]
            if existing_id in self._artifacts:
                return self._artifacts[existing_id]

        # Compute content hash
        content_hash = self._compute_content_hash(obj)

        # Check for content deduplication
        existing_path = self._find_by_content_hash(content_hash)

        if existing_path:
            path = existing_path  # Reuse existing file
        else:
            path = self._persist_object(obj, artifact_id, artifact_type)

        # Extract last node for step info
        last_node = chain.nodes[-1] if chain.nodes else None

        record = ArtifactRecordV3(
            artifact_id=artifact_id,
            content_hash=content_hash,
            operator_chain=chain,
            chain_path=chain_path,
            step_index=last_node.step_index if last_node else 0,
            substep_index=last_node.substep_index if last_node else None,
            operator_class=last_node.operator_class if last_node else "",
            operator_name=metadata.get("operator_name", ""),
            pipeline_id=self.pipeline_id,
            branch_path=last_node.branch_path if last_node else [],
            source_index=last_node.source_index if last_node else None,
            fold_id=fold_id,
            artifact_type=artifact_type,
            depends_on=depends_on or [],
            path=path,
            **self._get_format_info(obj)
        )

        self._artifacts[artifact_id] = record
        self._by_chain[chain_path] = artifact_id

        # Register dependencies
        if depends_on:
            for dep_id in depends_on:
                self._dependency_graph.add_dependency(artifact_id, dep_id)

        return record

    def get_by_chain(self, chain: OperatorChain) -> Optional[ArtifactRecordV3]:
        """Get artifact by exact chain match."""
        chain_path = chain.to_path()
        artifact_id = self._by_chain.get(chain_path)
        return self._artifacts.get(artifact_id) if artifact_id else None

    def get_chain_prefix(
        self,
        chain_prefix: OperatorChain,
        branch_path: Optional[List[int]] = None,
        source_index: Optional[int] = None
    ) -> List[ArtifactRecordV3]:
        """Get all artifacts whose chain starts with the given prefix."""
        prefix_path = chain_prefix.to_path()
        results = []

        for chain_path, artifact_id in self._by_chain.items():
            if chain_path.startswith(prefix_path):
                record = self._artifacts[artifact_id]

                # Apply filters
                if branch_path is not None:
                    if record.branch_path != branch_path:
                        continue

                if source_index is not None:
                    if record.source_index != source_index:
                        continue

                results.append(record)

        return results
```

---

## 7. Revised Artifact Loader

### 7.1 ArtifactLoader V3

```python
class ArtifactLoaderV3:
    def __init__(self, binaries_dir: Path, manifest: ManifestV3):
        self.binaries_dir = binaries_dir
        self.manifest = manifest
        self._cache = LRUCache(maxsize=100)
        self._artifacts_by_chain = self._build_chain_index()

    def _build_chain_index(self) -> Dict[str, ArtifactRecordV3]:
        """Build index from chain_path to artifact record."""
        index = {}
        for record in self.manifest.artifacts:
            index[record.chain_path] = record
        return index

    def load_by_chain(
        self,
        chain: OperatorChain,
        fold_id: Optional[int] = None
    ) -> Any:
        """Load artifact by exact chain match."""
        chain_path = chain.to_path()
        if fold_id is not None:
            chain_path = f"{chain_path}:fold={fold_id}"

        if chain_path in self._cache:
            return self._cache[chain_path]

        record = self._artifacts_by_chain.get(chain_path)
        if not record:
            raise ArtifactNotFoundError(f"No artifact for chain: {chain_path}")

        obj = self._load_from_path(record.path, record.format)
        self._cache[chain_path] = obj
        return obj

    def load_for_step(
        self,
        step_index: int,
        branch_path: Optional[List[int]] = None,
        source_index: Optional[int] = None,
        substep_index: Optional[int] = None
    ) -> List[Tuple[str, Any]]:
        """Load all artifacts for a step with filters."""
        results = []

        for chain_path, record in self._artifacts_by_chain.items():
            if record.step_index != step_index:
                continue

            if substep_index is not None and record.substep_index != substep_index:
                continue

            if branch_path is not None and record.branch_path != branch_path:
                continue

            if source_index is not None and record.source_index != source_index:
                continue

            obj = self.load_by_id(record.artifact_id)
            results.append((record.operator_name, obj))

        return results

    def load_chain_artifacts(
        self,
        chain: OperatorChain,
        branch_path: Optional[List[int]] = None
    ) -> Dict[int, List[Tuple[str, Any]]]:
        """Load all artifacts in a chain, organized by step."""
        results = {}

        # Filter chain by branch if needed
        if branch_path:
            chain = chain.filter_branch(branch_path)

        for node in chain.nodes:
            step_artifacts = self.load_for_step(
                step_index=node.step_index,
                branch_path=node.branch_path or branch_path,
                source_index=node.source_index,
                substep_index=node.substep_index
            )
            results[node.step_index] = step_artifacts

        return results

    def load_fold_models(
        self,
        chain: OperatorChain,
        fold_ids: List[int]
    ) -> Dict[int, Any]:
        """Load per-fold models for a given chain."""
        models = {}
        chain_path = chain.to_path()

        for record in self._artifacts_by_chain.values():
            if not record.chain_path.startswith(chain_path):
                continue
            if record.artifact_type != ArtifactType.MODEL:
                continue
            if record.fold_id in fold_ids:
                models[record.fold_id] = self.load_by_id(record.artifact_id)

        return models
```

---

## 8. Revised Minimal Predictor

### 8.1 MinimalArtifactProvider V3

```python
class MinimalArtifactProviderV3(ArtifactProvider):
    def __init__(
        self,
        minimal_pipeline: MinimalPipelineV3,
        artifact_loader: ArtifactLoaderV3,
        target_chain: Optional[OperatorChain] = None
    ):
        self.minimal_pipeline = minimal_pipeline
        self.loader = artifact_loader
        self.target_chain = target_chain
        self._artifact_cache: Dict[str, Any] = {}

    def get_artifacts_for_step(
        self,
        step_index: int,
        branch_path: Optional[List[int]] = None,
        source_index: Optional[int] = None
    ) -> List[Tuple[str, Any]]:
        """Get artifacts for a step, respecting target chain context."""
        # Get step info from minimal pipeline
        step = self.minimal_pipeline.get_step(step_index)
        if not step:
            return []

        # Use target chain's branch path if not specified
        effective_branch = branch_path
        if effective_branch is None and self.target_chain:
            # Extract branch from target chain for this step
            for node in self.target_chain.nodes:
                if node.step_index == step_index:
                    effective_branch = node.branch_path
                    break

        # Load from artifacts map
        artifacts = step.artifacts.get_artifacts(
            branch_path=effective_branch,
            source_index=source_index
        )

        results = []
        for artifact_id in artifacts:
            if artifact_id in self._artifact_cache:
                obj = self._artifact_cache[artifact_id]
            else:
                obj = self.loader.load_by_id(artifact_id)
                self._artifact_cache[artifact_id] = obj

            name = self._get_operator_name(artifact_id, step)
            results.append((name, obj))

        return results

    def get_chain_for_prediction(
        self,
        prediction_record: Dict[str, Any]
    ) -> OperatorChain:
        """Reconstruct the operator chain for a prediction."""
        chain_path = prediction_record.get("chain_path")
        if chain_path:
            return OperatorChain.from_path(chain_path)

        # Fallback: reconstruct from trace
        trace = self.minimal_pipeline.trace
        branch_path = prediction_record.get("branch_path", [])

        nodes = []
        for step in trace.steps:
            if self._step_on_path(step, branch_path):
                nodes.append(OperatorNode(
                    step_index=step.step_index,
                    operator_class=step.operator_class,
                    branch_path=step.parent_branch_path,
                    ...
                ))

        return OperatorChain(nodes=nodes)
```

### 8.2 MinimalPredictor V3

```python
class MinimalPredictorV3:
    def __init__(
        self,
        minimal_pipeline: MinimalPipelineV3,
        artifact_loader: ArtifactLoaderV3
    ):
        self.minimal_pipeline = minimal_pipeline
        self.loader = artifact_loader

    def predict(
        self,
        dataset: SpectroDataset,
        prediction_record: Dict[str, Any]
    ) -> np.ndarray:
        """Execute minimal pipeline for prediction."""
        # Get target chain from prediction record
        target_chain = self._get_target_chain(prediction_record)
        target_branch = prediction_record.get("branch_path", [])

        # Create artifact provider with target context
        provider = MinimalArtifactProviderV3(
            minimal_pipeline=self.minimal_pipeline,
            artifact_loader=self.loader,
            target_chain=target_chain
        )

        # Execute steps on the target chain
        context = ExecutionContext()
        context.selector = context.selector.with_branch(
            branch_path=target_branch
        )

        for step in self.minimal_pipeline.steps:
            # Skip steps not on target chain
            if not self._step_on_chain(step, target_chain):
                continue

            # Get artifacts for this step
            artifacts = provider.get_artifacts_for_step(
                step_index=step.step_index,
                branch_path=target_branch
            )

            # Execute step with artifacts
            result = step.controller.execute(
                step_info=step.config,
                dataset=dataset,
                context=context,
                loaded_binaries=artifacts,
                mode="predict"
            )

            context = result.updated_context
            dataset = result.dataset

        return dataset.y

    def _get_target_chain(self, prediction_record: Dict[str, Any]) -> OperatorChain:
        """Extract or reconstruct target chain from prediction record."""
        # V3 predictions include chain_path directly
        if "chain_path" in prediction_record:
            return OperatorChain.from_path(prediction_record["chain_path"])

        # Fallback for v2 records
        branch_path = prediction_record.get("branch_path", [])
        model_step = prediction_record.get("model_step_number")

        # Build chain from trace
        nodes = []
        for step in self.minimal_pipeline.trace.steps:
            if step.step_index <= model_step:
                if self._branch_matches(step.parent_branch_path, branch_path):
                    nodes.append(OperatorNode(
                        step_index=step.step_index,
                        operator_class=step.operator_class,
                        branch_path=step.parent_branch_path
                    ))

        return OperatorChain(nodes=nodes)
```

---

## 9. Controller Integration

### 9.1 BaseModelController V3

```python
class BaseModelControllerV3(BaseModelController):
    def _persist_model(
        self,
        runtime_context: RuntimeContext,
        model: Any,
        fold_id: Optional[int] = None,
        **kwargs
    ) -> ArtifactRecordV3:
        registry = runtime_context.artifact_registry
        recorder = runtime_context.trace_recorder

        # Get current chain from recorder
        chain = recorder.current_chain()

        # Add model node to chain
        model_node = OperatorNode(
            step_index=runtime_context.step_number,
            substep_index=runtime_context.substep_number,
            operator_class=model.__class__.__name__,
            branch_path=recorder.current_branch_path(),
            source_index=None,
            fold_id=fold_id
        )
        full_chain = chain.append(model_node)

        # Collect dependencies (preprocessing chain)
        depends_on = self._collect_dependencies(recorder, runtime_context.step_number)

        # Register with full chain
        record = registry.register(
            obj=model,
            chain=full_chain,
            artifact_type=ArtifactType.MODEL,
            fold_id=fold_id,
            depends_on=depends_on,
            operator_name=kwargs.get("custom_name", model.__class__.__name__)
        )

        # Record in trace
        recorder.record_artifact(
            artifact_id=record.artifact_id,
            chain=full_chain,
            is_primary=(fold_id is None),
            fold_id=fold_id,
            branch_path=recorder.current_branch_path()
        )

        return record

    def _collect_dependencies(
        self,
        recorder: TraceRecorderV3,
        up_to_step: int
    ) -> List[str]:
        """Collect artifact IDs this model depends on."""
        deps = []
        for step in recorder.trace.steps:
            if step.step_index >= up_to_step:
                break
            if step.artifacts.primary_artifacts:
                deps.extend(step.artifacts.primary_artifacts.values())
        return deps
```

### 9.2 TransformerMixinController V3

```python
class TransformerMixinControllerV3(TransformerMixinController):
    def _persist_transformer(
        self,
        runtime_context: RuntimeContext,
        transformer: Any,
        name: str,
        source_index: Optional[int] = None
    ) -> ArtifactRecordV3:
        registry = runtime_context.artifact_registry
        recorder = runtime_context.trace_recorder

        # Get current chain
        chain = recorder.current_chain()

        # Add transformer node
        transformer_node = OperatorNode(
            step_index=runtime_context.step_number,
            substep_index=runtime_context.substep_number,
            operator_class=transformer.__class__.__name__,
            branch_path=recorder.current_branch_path(),
            source_index=source_index,  # NEW: track source
            fold_id=None
        )
        full_chain = chain.append(transformer_node)

        # Register
        record = registry.register(
            obj=transformer,
            chain=full_chain,
            artifact_type=ArtifactType.TRANSFORMER,
            operator_name=name
        )

        # Record in trace
        recorder.record_artifact(
            artifact_id=record.artifact_id,
            chain=full_chain,
            branch_path=recorder.current_branch_path(),
            source_index=source_index
        )

        return record
```

### 9.3 MetaModelController V3

```python
class MetaModelControllerV3(MetaModelController):
    def _persist_meta_model(
        self,
        runtime_context: RuntimeContext,
        meta_model: Any,
        source_chains: List[OperatorChain],
        fold_id: Optional[int] = None
    ) -> ArtifactRecordV3:
        registry = runtime_context.artifact_registry
        recorder = runtime_context.trace_recorder

        # Build meta-model chain from source chains
        # Format: source1_chain + source2_chain > meta_model
        combined_input = "+".join(c.to_path() for c in source_chains)

        meta_node = OperatorNode(
            step_index=runtime_context.step_number,
            operator_class=meta_model.__class__.__name__,
            branch_path=[],  # Meta-models typically at root
            fold_id=fold_id
        )

        # Custom chain format for meta-models
        meta_chain = OperatorChain(nodes=[meta_node])
        meta_chain_path = f"{combined_input}>{meta_node.to_key()}"

        # Collect source artifact IDs as dependencies
        depends_on = []
        for source_chain in source_chains:
            source_artifacts = registry.get_by_chain(source_chain)
            if source_artifacts:
                depends_on.append(source_artifacts.artifact_id)

        record = registry.register(
            obj=meta_model,
            chain=meta_chain,
            artifact_type=ArtifactType.META_MODEL,
            fold_id=fold_id,
            depends_on=depends_on
        )

        return record
```

---

## 10. Migration Strategy

### 10.1 Manifest Migration

```python
def migrate_manifest_v2_to_v3(manifest_v2: dict) -> ManifestV3:
    """Migrate v2 manifest to v3 format."""
    artifacts_v3 = []

    for item in manifest_v2.get("artifacts", {}).get("items", []):
        # Parse v2 artifact ID
        v2_id = item["artifact_id"]
        pipeline_id, branch_path, step_index, fold_id, sub_index = parse_v2_id(v2_id)

        # Reconstruct chain from v2 data
        node = OperatorNode(
            step_index=step_index,
            substep_index=sub_index,
            operator_class=item.get("class_name", "Unknown"),
            branch_path=branch_path,
            source_index=None,  # Not tracked in v2
            fold_id=fold_id
        )
        chain = OperatorChain(nodes=[node])

        # Create v3 record
        record = ArtifactRecordV3(
            artifact_id=f"{pipeline_id}${compute_chain_hash(chain.to_path())}:{fold_id or 'all'}",
            content_hash=item.get("content_hash", ""),
            operator_chain=chain,
            chain_path=chain.to_path(),
            step_index=step_index,
            substep_index=sub_index,
            operator_class=item.get("class_name", ""),
            operator_name=item.get("custom_name", ""),
            pipeline_id=pipeline_id,
            branch_path=branch_path,
            source_index=None,
            fold_id=fold_id,
            artifact_type=ArtifactType(item.get("artifact_type", "model")),
            depends_on=item.get("depends_on", []),
            path=item["path"],
            format=item.get("format", "joblib"),
            format_version=item.get("format_version", ""),
            size_bytes=item.get("size_bytes", 0),
            params=item.get("params", {}),
            nirs4all_version=item.get("nirs4all_version", ""),
            created_at=item.get("created_at", ""),
            version=3
        )

        artifacts_v3.append(record)

    return ManifestV3(artifacts=artifacts_v3, ...)
```

### 10.2 Backward Compatibility Layer

```python
class ArtifactLoaderCompat:
    """Loader that supports both v2 and v3 formats."""

    def __init__(self, binaries_dir: Path, manifest: dict):
        self.version = manifest.get("artifacts", {}).get("schema_version", "1.0")

        if self.version.startswith("3"):
            self._loader = ArtifactLoaderV3(binaries_dir, manifest)
        else:
            # Migrate on-the-fly
            manifest_v3 = migrate_manifest_v2_to_v3(manifest)
            self._loader = ArtifactLoaderV3(binaries_dir, manifest_v3)

    def load_by_id(self, artifact_id: str) -> Any:
        # Handle both v2 and v3 ID formats
        if "$" in artifact_id:
            return self._loader.load_by_id(artifact_id)
        else:
            # v2 ID - need to map to v3
            return self._load_v2_artifact(artifact_id)
```

---

## 11. Prediction Records V3

### 11.1 Enhanced Prediction Record

```python
@dataclass
class PredictionRecordV3:
    # Identity
    prediction_id: str
    pipeline_uid: str

    # Chain tracking (NEW)
    chain_path: str                    # Full operator chain
    model_artifact_id: str             # V3 artifact ID

    # Context
    branch_path: List[int]
    fold_id: Optional[int]
    fold_weights: Optional[Dict[int, float]]

    # For averaging
    component_chains: Optional[List[str]]  # For avg/w_avg predictions

    # Legacy compatibility
    model_step_number: int
    op_counter: int                    # Deprecated in v3

    # Prediction data
    sample_indices: List[int]
    predictions: np.ndarray
    y_true: Optional[np.ndarray]
    metrics: Dict[str, float]

    # Timestamps
    created_at: str
```

### 11.2 Prediction DB Integration

```python
class PredictionDBV3:
    def get_predictions_for_chain(
        self,
        chain: OperatorChain,
        fold_id: Optional[int] = None
    ) -> List[PredictionRecordV3]:
        """Get all predictions made by models on this chain."""
        chain_path = chain.to_path()
        return [
            p for p in self._predictions
            if p.chain_path.startswith(chain_path)
            and (fold_id is None or p.fold_id == fold_id)
        ]

    def get_predictions_for_meta_model(
        self,
        source_chains: List[OperatorChain]
    ) -> Dict[str, List[PredictionRecordV3]]:
        """Get predictions from all source models for stacking."""
        result = {}
        for chain in source_chains:
            chain_path = chain.to_path()
            result[chain_path] = self.get_predictions_for_chain(chain)
        return result
```

---

## 12. Summary of Changes

### 12.1 New Concepts

| Concept | Purpose |
|---------|---------|
| `OperatorNode` | Single operator in execution path |
| `OperatorChain` | Full path of operators |
| `chain_path` | String serialization of chain |
| `chain_hash` | Deterministic ID from chain |

### 12.2 Modified Components

| Component | Key Changes |
|-----------|-------------|
| `ArtifactRecord` | Added `operator_chain`, `chain_path`, `source_index` |
| `ExecutionStep` | Added `input_chain`, `output_chains`, `branch_outputs` |
| `StepArtifacts` | Added `by_branch`, `by_source` indexes |
| `TraceRecorder` | Added chain stack, branch stack |
| `ArtifactRegistry` | Chain-based registration and lookup |
| `ArtifactLoader` | Chain-based loading and filtering |
| `MinimalPredictor` | Target chain context for prediction |
| `PredictionRecord` | Added `chain_path`, `component_chains` |

### 12.3 Removed/Deprecated

| Item | Reason |
|------|--------|
| `operation_count` for naming | Replaced by chain path |
| `_find_transformer_by_class` fallback | No longer needed with chain |
| `sub_index` in artifact ID | Part of chain now |
| Manual branch path assembly | Automatic from chain stack |

---

## 13. Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Implement `OperatorNode` and `OperatorChain`
- [ ] Implement `TraceRecorderV3` with chain/branch stacks
- [ ] Update `ArtifactRecordV3` dataclass
- [ ] Implement manifest migration

### Phase 2: Registry & Recording (Week 3-4)
- [ ] Implement `ArtifactRegistryV3`
- [ ] Update `BaseModelController` to use chains
- [ ] Update `TransformerMixinController` to use chains
- [ ] Update `BranchController` to record substeps

### Phase 3: Loading & Prediction (Week 5-6)
- [ ] Implement `ArtifactLoaderV3`
- [ ] Implement `MinimalArtifactProviderV3`
- [ ] Implement `MinimalPredictorV3`
- [ ] Update prediction records

### Phase 4: Edge Cases (Week 7-8)
- [ ] Implement `MetaModelControllerV3`
- [ ] Implement bundle import chain merging
- [ ] Test all edge cases from Section 1.2
- [ ] Performance optimization

### Phase 5: Cleanup (Week 9)
- [ ] Remove deprecated code
- [ ] Update all examples
- [ ] Update documentation
- [ ] Final testing

---

## 14. Validation Criteria

### 14.1 Test Cases

All these must pass before V3 is complete:

```python
def test_simple_pipeline_reload():
    """Standard pipeline: Scaler → Split → Model."""
    pass

def test_multisource_reload():
    """3 X sources with per-source transformers."""
    pass

def test_branching_reload():
    """2 branches, each with transformers and model."""
    pass

def test_branching_multisource_reload():
    """2 branches × 3 sources."""
    pass

def test_nested_branches_reload():
    """Branch within branch."""
    pass

def test_subpipeline_models_reload():
    """[model1, model2] at same step."""
    pass

def test_meta_model_stacking():
    """MetaModel collecting predictions from branches."""
    pass

def test_bundle_import():
    """Load pre-trained bundle and extend."""
    pass

def test_chain_determinism():
    """Same pipeline always produces same chains."""
    pass

def test_migration_v2_to_v3():
    """V2 manifests load correctly in V3."""
    pass
```

### 14.2 Performance Benchmarks

- Chain generation: < 1ms per artifact
- Chain lookup: O(1) with hash index
- Migration: < 5s for 1000 artifacts
- Memory: Chain objects share node references

---

## 15. Conclusion

The V3 artifact system addresses all identified issues through the fundamental concept of **Operator Chains**:

1. **Every artifact knows its full path**: No more ambiguous ID parsing
2. **Branch substeps are recorded individually**: Complete trace fidelity
3. **Multi-source is tracked explicitly**: `source_index` in every node
4. **Deterministic replay guaranteed**: Same chain → same artifacts
5. **Meta-models have explicit dependencies**: Source chains tracked
6. **Bundles merge cleanly**: Import as prefix chains

The chain-based approach is more verbose but provides the complete information needed for reliable artifact management in complex pipelines.
