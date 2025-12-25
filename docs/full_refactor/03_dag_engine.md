# NIRS4ALL v2.0: DAG Execution Engine

**Author**: GitHub Copilot (Claude Opus 4.5)
**Date**: December 25, 2025
**Status**: Design Proposal (Revised per Critical Review)
**Document**: 3 of 5

---

## Table of Contents

1. [Overview](#overview)
2. [DAG Model](#dag-model)
3. [Node Types](#node-types)
   - [TransformNode](#transformnode)
   - [ModelNode](#modelnode)
   - [ForkNode and JoinNode](#forknode-and-joinnode)
   - [SplitterNode](#splitternode)
   - [MidFusionNode](#midfusionnode)
   - [LayoutAwareModelNode](#layoutawaremodelnode)
4. [DAG Builder](#dag-builder)
5. [Execution Engine](#execution-engine)
6. [Branching and Merging](#branching-and-merging)
7. [Fork/Join Synchronization](#forkjoin-synchronization)
8. [Cross-Validation as Fork/Join](#cross-validation-as-forkjoin)
9. [Generator Expansion](#generator-expansion)
10. [Operator Serialization](#operator-serialization)
11. [Artifact Management](#artifact-management)
12. [Prediction Store Integration](#prediction-store-integration)

---

## Overview

The DAG Execution Engine compiles pipeline syntax into a directed acyclic graph and executes it with support for parallelism, caching, and deterministic replay.

### Design Goals

1. **DAG-First**: All pipelines are graphs, even linear ones
2. **Pre-Execution Expansion**: Generators expand before execution with **configurable limits**
3. **Parallelizable**: Independent nodes can execute in parallel with barrier synchronization
4. **Cacheable**: Hash-based artifact identification
5. **Reproducible**: Deterministic execution order with consistent node sorting
6. **Minimal Execution**: Prediction mode runs only required nodes
7. **Fail-Fast**: Clear error propagation with context

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      Pipeline Execution                          │
│                                                                  │
│  [Pipeline Syntax]                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────┐                                            │
│  │   DAG Builder   │  Parse → Expand Generators → Build Graph   │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │  Executable DAG │  Nodes + Edges + Metadata                  │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │ Execution Engine│  Topological Sort → Execute → Collect      │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  [RunResult: predictions, artifacts, metrics]                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## DAG Model

### Exceptions and Warnings

```python
from nirs4all.exceptions import NIRSError


class DAGError(NIRSError):
    """Base exception for DAG-related errors."""
    pass


class GeneratorExplosionError(DAGError):
    """Raised when generator expansion exceeds safe limits.

    This prevents runaway memory consumption from unbounded
    _or_, _range_, or _grid_ expansions.
    """
    pass


class GeneratorExplosionWarning(UserWarning):
    """Warning issued when generator expansion is large but within limits.

    Triggered when variant count exceeds warn_threshold (default 100)
    but is still under max_variants limit.
    """
    pass


class DAGCycleError(DAGError):
    """Raised when the DAG contains cycles."""
    pass


class DAGValidationError(DAGError):
    """Raised when DAG structure is invalid."""
    pass


class BranchSyncError(DAGError):
    """Raised when fork/join synchronization fails."""
    pass
```

### Core Structures

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum


class NodeType(Enum):
    """Types of DAG nodes."""
    SOURCE = "source"           # Initial data source
    TRANSFORM = "transform"     # Feature transformation
    Y_TRANSFORM = "y_transform" # Target transformation
    SPLITTER = "splitter"       # CV fold assignment
    MODEL = "model"             # Model training/prediction
    FORK = "fork"               # Branch point
    JOIN = "join"               # Merge point
    AUGMENT = "augment"         # Sample augmentation
    FEATURE_AUG = "feature_aug" # Feature augmentation
    FILTER = "filter"           # Sample filtering
    SINK = "sink"               # Terminal node


@dataclass
class DAGNode:
    """A node in the execution DAG.

    Attributes:
        node_id: Unique identifier
        node_type: Type classification
        operator: The operator instance or config
        operator_class: Fully qualified class name
        operator_params: Constructor parameters
        metadata: Additional node metadata
        is_dynamic: Created during execution (not compile-time)
        resource_requirements: Hardware requirements for scheduling
    """
    node_id: str
    node_type: NodeType
    operator: Any
    operator_class: str
    operator_params: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_dynamic: bool = False
    resource_requirements: Optional["NodeResources"] = None


@dataclass
class NodeResources:
    """Resource requirements for node execution.

    Used by the ExecutionEngine to schedule nodes appropriately:
    - GPU nodes wait for GPU availability
    - Memory-heavy nodes may run sequentially
    - CPU-only nodes can be parallelized freely
    """
    requires_gpu: bool = False
    gpu_memory_mb: int = 0             # Estimated GPU memory needed
    cpu_memory_mb: int = 0             # Estimated CPU memory needed
    preferred_device: str = "auto"     # "cpu", "cuda:0", "auto"
    parallelizable: bool = True        # Can run alongside other nodes?

    @classmethod
    def from_operator(cls, operator: Any) -> "NodeResources":
        """Infer resource requirements from operator.

        Heuristics:
        - PyTorch/TensorFlow models: requires_gpu=True
        - Large transforms (PCA with many components): high memory
        - Simple scalers: low requirements
        """
        requires_gpu = False
        gpu_memory = 0

        # Check for PyTorch
        if hasattr(operator, "to") and hasattr(operator, "parameters"):
            requires_gpu = True
            gpu_memory = 500  # Default estimate

        # Check for TensorFlow
        if hasattr(operator, "build") and "tensorflow" in str(type(operator)):
            requires_gpu = True
            gpu_memory = 500

        # Check for explicit attribute
        if hasattr(operator, "requires_gpu"):
            requires_gpu = operator.requires_gpu

        return cls(
            requires_gpu=requires_gpu,
            gpu_memory_mb=gpu_memory
        )


class GPUResourceManager:
    """Manages GPU resources for pipeline execution.

    Handles:
    - GPU availability detection
    - Memory tracking and limits
    - Automatic CPU fallback when GPU unavailable
    - Multi-GPU distribution

    Thread Safety:
        Uses locks for GPU allocation/deallocation. Multiple threads
        can safely request GPU resources.

    Usage:
        gpu_mgr = GPUResourceManager(fallback_to_cpu=True)

        # Check availability
        if gpu_mgr.is_available():
            device = gpu_mgr.allocate(memory_mb=500)
            try:
                model.to(device)
                result = model.forward(X)
            finally:
                gpu_mgr.release(device)

        # Or use context manager
        with gpu_mgr.device(memory_mb=500) as device:
            model.to(device)
            result = model.forward(X)
    """

    def __init__(
        self,
        fallback_to_cpu: bool = True,
        memory_fraction: float = 0.9,  # Max fraction of GPU memory to use
        preferred_gpus: Optional[List[int]] = None,
        log_allocation: bool = True
    ):
        """Initialize GPU resource manager.

        Args:
            fallback_to_cpu: If True, use CPU when GPU unavailable
            memory_fraction: Maximum fraction of GPU memory to use (0.0-1.0)
            preferred_gpus: List of GPU indices to use (None = all available)
            log_allocation: If True, log GPU allocation/release events
        """
        import threading

        self.fallback_to_cpu = fallback_to_cpu
        self.memory_fraction = memory_fraction
        self.preferred_gpus = preferred_gpus
        self.log_allocation = log_allocation

        self._lock = threading.Lock()
        self._allocated: Dict[str, int] = {}  # device -> allocated MB
        self._gpu_info = self._detect_gpus()

    def _detect_gpus(self) -> Dict[int, Dict[str, Any]]:
        """Detect available GPUs and their properties."""
        gpus = {}

        # Try PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    if self.preferred_gpus is None or i in self.preferred_gpus:
                        props = torch.cuda.get_device_properties(i)
                        gpus[i] = {
                            "name": props.name,
                            "total_memory_mb": props.total_memory // (1024 * 1024),
                            "backend": "torch"
                        }
                return gpus
        except ImportError:
            pass

        # Try TensorFlow
        try:
            import tensorflow as tf
            physical_gpus = tf.config.list_physical_devices('GPU')
            for i, gpu in enumerate(physical_gpus):
                if self.preferred_gpus is None or i in self.preferred_gpus:
                    # TF doesn't easily expose memory, estimate 8GB
                    gpus[i] = {
                        "name": gpu.name,
                        "total_memory_mb": 8192,  # Default estimate
                        "backend": "tensorflow"
                    }
            return gpus
        except ImportError:
            pass

        return gpus

    def is_available(self) -> bool:
        """Check if any GPU is available."""
        return len(self._gpu_info) > 0

    def get_free_memory(self, gpu_id: int = 0) -> int:
        """Get available memory on GPU in MB."""
        if gpu_id not in self._gpu_info:
            return 0

        total = self._gpu_info[gpu_id]["total_memory_mb"]
        usable = int(total * self.memory_fraction)
        allocated = self._allocated.get(f"cuda:{gpu_id}", 0)

        return usable - allocated

    def allocate(self, memory_mb: int) -> str:
        """Allocate GPU memory, return device string.

        Args:
            memory_mb: Requested memory in MB

        Returns:
            Device string ("cuda:0", "cuda:1", or "cpu")

        Raises:
            RuntimeError: If no GPU available and fallback_to_cpu=False
        """
        with self._lock:
            # Find GPU with enough memory
            for gpu_id in sorted(self._gpu_info.keys()):
                if self.get_free_memory(gpu_id) >= memory_mb:
                    device = f"cuda:{gpu_id}"
                    self._allocated[device] = self._allocated.get(device, 0) + memory_mb

                    if self.log_allocation:
                        import logging
                        logging.getLogger("nirs4all.gpu").debug(
                            f"Allocated {memory_mb}MB on {device}"
                        )

                    return device

            # No GPU available
            if self.fallback_to_cpu:
                if self.log_allocation:
                    import logging
                    logging.getLogger("nirs4all.gpu").info(
                        f"GPU unavailable, falling back to CPU"
                    )
                return "cpu"

            raise RuntimeError(
                f"No GPU with {memory_mb}MB available. "
                f"Available GPUs: {self._gpu_info}. "
                f"Set fallback_to_cpu=True to use CPU instead."
            )

    def release(self, device: str, memory_mb: Optional[int] = None) -> None:
        """Release GPU memory."""
        if device == "cpu":
            return

        with self._lock:
            if device in self._allocated:
                if memory_mb:
                    self._allocated[device] = max(0, self._allocated[device] - memory_mb)
                else:
                    self._allocated[device] = 0

                if self.log_allocation:
                    import logging
                    logging.getLogger("nirs4all.gpu").debug(
                        f"Released memory on {device}"
                    )

    @contextmanager
    def device(self, memory_mb: int) -> Generator[str, None, None]:
        """Context manager for GPU allocation."""
        device = self.allocate(memory_mb)
        try:
            yield device
        finally:
            self.release(device, memory_mb)

    def summary(self) -> Dict[str, Any]:
        """Get summary of GPU resources."""
        return {
            "gpus": self._gpu_info,
            "allocated": self._allocated,
            "available": {
                f"cuda:{i}": self.get_free_memory(i)
                for i in self._gpu_info
            }
        }


@dataclass
class DAGEdge:
    """An edge connecting two nodes.

    Attributes:
        source_id: Source node ID
        target_id: Target node ID
        edge_type: Type of connection
        view_spec: Data view specification for this edge
        metadata: Additional edge metadata
    """
    source_id: str
    target_id: str
    edge_type: str = "data"
    view_spec: Optional[ViewSpec] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutableDAG:
    """Complete executable DAG structure.

    Attributes:
        dag_id: Unique identifier for this DAG
        nodes: All nodes in the DAG
        edges: All edges connecting nodes
        source_nodes: Entry point nodes (no incoming edges)
        sink_nodes: Terminal nodes (no outgoing edges)
        metadata: DAG-level metadata
    """
    dag_id: str
    nodes: Dict[str, DAGNode]
    edges: List[DAGEdge]
    source_nodes: List[str]
    sink_nodes: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_node(self, node_id: str) -> DAGNode:
        return self.nodes[node_id]

    def get_predecessors(self, node_id: str) -> List[str]:
        """Get nodes that feed into this node."""
        return [e.source_id for e in self.edges if e.target_id == node_id]

    def get_successors(self, node_id: str) -> List[str]:
        """Get nodes that this node feeds into."""
        return [e.target_id for e in self.edges if e.source_id == node_id]

    def topological_order(self) -> List[str]:
        """Return nodes in topological execution order."""
        from collections import deque

        in_degree = {nid: 0 for nid in self.nodes}
        for edge in self.edges:
            in_degree[edge.target_id] += 1

        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        order = []

        while queue:
            node_id = queue.popleft()
            order.append(node_id)
            for succ in self.get_successors(node_id):
                in_degree[succ] -= 1


class CheckpointManager:
    """Manages checkpoints for long-running pipelines.

    Enables:
    - Resuming from failures
    - Incremental execution
    - Progress tracking
    - Resource-efficient reruns

    Checkpoints are saved after each node completes, including:
    - Node execution state
    - DatasetContext snapshot
    - Collected artifacts
    - Prediction store state

    Usage:
        # Enable checkpointing
        engine = ExecutionEngine(
            checkpoint_manager=CheckpointManager(
                checkpoint_dir=Path("./checkpoints"),
                checkpoint_every=5,  # Every 5 nodes
            )
        )

        # Resume from failure
        result = engine.execute(dag, context, resume=True)

        # Or resume manually
        checkpoint = CheckpointManager.load_latest("./checkpoints")
        result = engine.resume(dag, checkpoint)
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_every: int = 10,
        keep_last_n: int = 3,
        compress: bool = True
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            checkpoint_every: Save checkpoint every N nodes
            keep_last_n: Keep only last N checkpoints (0 = keep all)
            compress: If True, compress checkpoints with gzip
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_every = checkpoint_every
        self.keep_last_n = keep_last_n
        self.compress = compress
        self._node_count = 0

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def should_checkpoint(self) -> bool:
        """Check if a checkpoint should be saved now."""
        self._node_count += 1
        return self._node_count % self.checkpoint_every == 0

    def save(
        self,
        dag_id: str,
        completed_nodes: List[str],
        context_snapshot: Dict[str, Any],
        artifacts: Dict[str, Any],
        prediction_store_state: Dict[str, Any]
    ) -> Path:
        """Save checkpoint.

        Returns:
            Path to saved checkpoint file
        """
        import json
        import gzip
        from datetime import datetime

        checkpoint = {
            "dag_id": dag_id,
            "completed_nodes": completed_nodes,
            "context_snapshot": context_snapshot,
            "artifacts": artifacts,
            "prediction_store": prediction_store_state,
            "timestamp": datetime.now().isoformat(),
            "node_count": self._node_count
        }

        filename = f"checkpoint_{dag_id}_{self._node_count:06d}.json"
        if self.compress:
            filename += ".gz"

        path = self.checkpoint_dir / filename

        if self.compress:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(checkpoint, f)
        else:
            with open(path, "w") as f:
                json.dump(checkpoint, f, indent=2)

        # Cleanup old checkpoints
        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints(dag_id)

        return path

    def _cleanup_old_checkpoints(self, dag_id: str) -> None:
        """Remove old checkpoints, keeping only last N."""
        pattern = f"checkpoint_{dag_id}_*.json*"
        checkpoints = sorted(self.checkpoint_dir.glob(pattern))

        if len(checkpoints) > self.keep_last_n:
            for old in checkpoints[:-self.keep_last_n]:
                old.unlink()

    @classmethod
    def load_latest(cls, checkpoint_dir: Path, dag_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load most recent checkpoint.

        Args:
            checkpoint_dir: Directory containing checkpoints
            dag_id: Optional DAG ID to filter by

        Returns:
            Checkpoint dict or None if no checkpoint found
        """
        import json
        import gzip

        pattern = f"checkpoint_{dag_id or '*'}_*.json*"
        checkpoints = sorted(Path(checkpoint_dir).glob(pattern))

        if not checkpoints:
            return None

        latest = checkpoints[-1]

        if latest.suffix == ".gz":
            with gzip.open(latest, "rt", encoding="utf-8") as f:
                return json.load(f)
        else:
            with open(latest) as f:
                return json.load(f)

    def get_remaining_nodes(
        self,
        dag: ExecutableDAG,
        checkpoint: Dict[str, Any]
    ) -> List[str]:
        """Get nodes that still need to be executed."""
        completed = set(checkpoint["completed_nodes"])
        all_nodes = dag.topological_order()
        return [n for n in all_nodes if n not in completed]
                if in_degree[succ] == 0:
                    queue.append(succ)

        return order

    def get_execution_levels(self) -> List[List[str]]:
        """Group nodes by execution level for parallelization."""
        levels = []
        remaining = set(self.nodes.keys())
        executed = set()

        while remaining:
            level = []
            for node_id in remaining:
                preds = set(self.get_predecessors(node_id))
                if preds <= executed:
                    level.append(node_id)

            levels.append(level)
            executed.update(level)
            remaining -= set(level)

        return levels
```

### DAG Visualization

```
Example Pipeline:
[MinMaxScaler, {"branch": [[SNV()], [MSC()]]}, PLSRegression]

Compiled DAG:
                    ┌─────────┐
                    │ SOURCE  │
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │ SCALE   │ MinMaxScaler
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │  FORK   │ branch
                    └──┬───┬──┘
                       │   │
              ┌────────┘   └────────┐
              │                     │
         ┌────▼────┐           ┌────▼────┐
         │ TRANS_0 │ SNV       │ TRANS_1 │ MSC
         └────┬────┘           └────┬────┘
              │                     │
         ┌────▼────┐           ┌────▼────┐
         │ MODEL_0 │ PLS       │ MODEL_1 │ PLS
         └────┬────┘           └────┬────┘
              │                     │
              └────────┬────────────┘
                       │
                  ┌────▼────┐
                  │  JOIN   │ implicit merge
                  └────┬────┘
                       │
                  ┌────▼────┐
                  │  SINK   │
                  └─────────┘
```

---

## Node Types

### TransformNode

Applies sklearn-compatible transformations to features.

```python
@dataclass
class TransformNode(DAGNode):
    """Feature transformation node.

    Executes a TransformerMixin operator on feature data.
    Fits on training partition, transforms all partitions.
    """
    node_type: NodeType = NodeType.TRANSFORM

    def execute(
        self,
        context: DatasetContext,
        input_view: ViewSpec,
        mode: str = "train",
        artifacts: Optional[Dict[str, Any]] = None
    ) -> Tuple[DatasetContext, Dict[str, Any]]:
        """Execute transformation.

        Args:
            context: Current dataset context
            input_view: View specification for input data
            mode: "train" or "predict"
            artifacts: Pre-loaded artifacts for predict mode

        Returns:
            Tuple of (updated context, new artifacts)
        """
        transformer = self.operator

        if mode == "train":
            # Fit on training data
            X_train = context.resolver.materialize_X(
                input_view.with_partition("train"),
                layout="2d"
            )
            transformer.fit(X_train)

            # Transform all data
            X_all = context.resolver.materialize_X(
                input_view.for_transform(),
                layout="2d"
            )
            X_transformed = transformer.transform(X_all)

            # Register new block
            new_block_id = context.block_store.register_transform(
                parent_id=context.active_block_ids[0],
                data=X_transformed.reshape(X_all.shape[0], 1, -1),
                transform_info=TransformInfo(
                    transform_class=f"{transformer.__class__.__module__}.{transformer.__class__.__name__}",
                    transform_params=transformer.get_params(),
                    target_processing=transformer.__class__.__name__
                )
            )

            # Update context
            new_context = replace(
                context,
                active_block_ids=[new_block_id]
            )

            return new_context, {"transformer": transformer}

        else:  # predict mode
            transformer = artifacts["transformer"]
            X_all = context.resolver.materialize_X(input_view, layout="2d")
            X_transformed = transformer.transform(X_all)

            new_block_id = context.block_store.register_transform(
                parent_id=context.active_block_ids[0],
                data=X_transformed.reshape(X_all.shape[0], 1, -1),
                transform_info=TransformInfo(
                    transform_class=f"{transformer.__class__.__module__}.{transformer.__class__.__name__}",
                    transform_params=transformer.get_params()
                )
            )

            new_context = replace(context, active_block_ids=[new_block_id])
            return new_context, {}
```

### ModelNode

Trains or applies ML models with cross-validation.

```python
@dataclass
class ModelNode(DAGNode):
    """Model training/prediction node.

    Handles:
    - Per-fold model training
    - Prediction generation
    - VirtualModel creation for aggregated predictions
    """
    node_type: NodeType = NodeType.MODEL

    def execute(
        self,
        context: DatasetContext,
        input_view: ViewSpec,
        prediction_store: PredictionStore,
        mode: str = "train",
        artifacts: Optional[Dict[str, Any]] = None
    ) -> Tuple[DatasetContext, Dict[str, Any]]:
        """Execute model training or prediction.

        Args:
            context: Dataset context
            input_view: Input data view
            prediction_store: Where to store predictions
            mode: "train" or "predict"
            artifacts: Pre-loaded artifacts for predict mode

        Returns:
            Tuple of (context, artifacts including virtual_model)
        """
        model_class = self.operator.__class__

        if mode == "train":
            fold_models = []
            fold_weights = []

            for fold_id, (train_idx, val_idx) in enumerate(context.folds):
                # Clone model for this fold
                fold_model = clone(self.operator)

                # Get fold data
                train_view = input_view.with_fold(fold_id, "train")
                val_view = input_view.with_fold(fold_id, "val")

                X_train, y_train = context.resolver.materialize(train_view)
                X_val, y_val = context.resolver.materialize(val_view)

                # Train
                fold_model.fit(X_train, y_train)

                # Predict
                y_pred_train = fold_model.predict(X_train)
                y_pred_val = fold_model.predict(X_val)

                # Score
                val_score = self._compute_score(y_val, y_pred_val, context)

                # Store predictions
                prediction_store.add_prediction(
                    fold_id=fold_id,
                    partition="train",
                    y_true=y_train,
                    y_pred=y_pred_train,
                    sample_indices=train_idx.tolist(),
                    model_name=model_class.__name__,
                    train_score=self._compute_score(y_train, y_pred_train, context)
                )
                prediction_store.add_prediction(
                    fold_id=fold_id,
                    partition="val",
                    y_true=y_val,
                    y_pred=y_pred_val,
                    sample_indices=val_idx.tolist(),
                    model_name=model_class.__name__,
                    val_score=val_score
                )

                fold_models.append(fold_model)
                fold_weights.append(val_score)

            # Create virtual model
            virtual_model = VirtualModel(
                fold_models=fold_models,
                fold_weights=fold_weights,
                aggregation="weighted_mean"
            )

            return context, {
                "fold_models": fold_models,
                "virtual_model": virtual_model
            }

        else:  # predict mode
            virtual_model = artifacts["virtual_model"]
            X = context.resolver.materialize_X(input_view)
            y_pred = virtual_model.predict(X)

            prediction_store.add_prediction(
                partition="prediction",
                y_pred=y_pred,
                model_name=model_class.__name__
            )

            return context, {}

    def _compute_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        context: DatasetContext
    ) -> float:
        """Compute appropriate score based on task type."""
        from sklearn.metrics import mean_squared_error, accuracy_score

        if context.target_store.task_type.is_classification:
            return accuracy_score(y_true, y_pred)
        else:
            return mean_squared_error(y_true, y_pred)


@dataclass
class VirtualModel:
    """Aggregated model from CV folds with configurable aggregation.

    Represents the ensemble of fold models with weighted prediction.
    Supports multiple aggregation strategies and quality-aware weighting.
    """
    fold_models: List[Any]
    fold_scores: List[float]              # Validation scores per fold
    fold_sizes: Optional[List[int]] = None  # Training set sizes per fold
    aggregation: "AggregationStrategy" = field(default_factory=lambda: WeightedMeanStrategy())
    task_type: str = "regression"         # "regression" or "classification"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate aggregated predictions."""
        predictions = np.array([m.predict(X) for m in self.fold_models])
        return self.aggregation.aggregate(
            predictions,
            scores=self.fold_scores,
            sizes=self.fold_sizes,
            task_type=self.task_type
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate aggregated probabilities (classification)."""
        if not all(hasattr(m, 'predict_proba') for m in self.fold_models):
            raise AttributeError("Not all fold models support predict_proba")
        probas = np.array([m.predict_proba(X) for m in self.fold_models])
        return self.aggregation.aggregate(
            probas,
            scores=self.fold_scores,
            sizes=self.fold_sizes,
            task_type="classification"
        )

    def predict_all_folds(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all fold models."""
        return np.array([m.predict(X) for m in self.fold_models])

    def prediction_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Compute prediction uncertainty (std across folds)."""
        predictions = self.predict_all_folds(X)
        return predictions.std(axis=0)

    @property
    def primary_model(self) -> Any:
        """Get the best performing fold model for SHAP."""
        if not self.fold_scores:
            return self.fold_models[0]
        # Higher score = better (assuming R² or accuracy)
        best_idx = np.argmax(self.fold_scores)
        return self.fold_models[best_idx]


### Mixed Task Types in Pipelines

NIRS4ALL supports **mixed task types** within a single pipeline, enabling advanced workflows
like discretization-based classification on regression targets. The system maintains:

1. **Global task type**: Determined by the final model's output or explicitly set
2. **Local task types**: Individual models can have different task types

#### Task Type Detection

```python
class TaskType(Enum):
    """Pipeline task types."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    AUTO = "auto"  # Detect from data/model


def detect_task_type(model: Any, y: Optional[np.ndarray] = None) -> TaskType:
    """Detect task type from model or target.

    Priority:
    1. Explicit model attribute (model.task_type)
    2. sklearn is_classifier/is_regressor
    3. PyTorch/TF model output shape
    4. Target dtype analysis (int/categorical = classification)
    5. Default to regression
    """
    # 1. Explicit attribute
    if hasattr(model, 'task_type'):
        return TaskType(model.task_type)

    # 2. sklearn detection
    from sklearn.base import is_classifier, is_regressor
    if is_classifier(model):
        return TaskType.CLASSIFICATION
    if is_regressor(model):
        return TaskType.REGRESSION

    # 3. Check for predict_proba (classification indicator)
    if hasattr(model, 'predict_proba') and not hasattr(model, 'predict'):
        return TaskType.CLASSIFICATION

    # 4. Analyze target if available
    if y is not None:
        if np.issubdtype(y.dtype, np.integer):
            n_unique = len(np.unique(y))
            if n_unique < min(20, len(y) * 0.05):  # Heuristic: < 20 or < 5% unique
                return TaskType.CLASSIFICATION
        if y.dtype == object or np.issubdtype(y.dtype, np.str_):
            return TaskType.CLASSIFICATION

    # 5. Default
    return TaskType.REGRESSION
```

#### Mixed Task Type Pipeline Example

```python
# Workflow: Discretize regression target → classify → map back to values
from nirs4all.operators.transforms import QuantileDiscretizer, ClassToValueMapper

pipeline = [
    MinMaxScaler(),

    # Discretize continuous target into classes
    {"y_processing": QuantileDiscretizer(n_bins=5)},  # task_type → classification locally

    # Train classifier on discretized targets
    {"model": RandomForestClassifier(n_estimators=100)},  # Uses classification metrics

    # Optional: Map class predictions back to regression values
    {"post_processing": ClassToValueMapper(strategy="bin_center")},  # task_type → regression
]

# Global task_type determined by final output:
# - If ClassToValueMapper present → regression (continuous output)
# - If absent → classification (class output)
```

#### Task Type Context Propagation

```python
@dataclass
class TaskTypeContext:
    """Tracks task type through pipeline execution."""
    global_task_type: TaskType = TaskType.AUTO
    current_task_type: TaskType = TaskType.AUTO  # May differ locally
    task_history: List[Tuple[str, TaskType]] = field(default_factory=list)

    def update(self, node_id: str, task_type: TaskType) -> "TaskTypeContext":
        """Update context when task type changes."""
        new_history = self.task_history + [(node_id, task_type)]
        return replace(
            self,
            current_task_type=task_type,
            task_history=new_history
        )

    def finalize(self) -> TaskType:
        """Determine final global task type."""
        if self.global_task_type != TaskType.AUTO:
            return self.global_task_type
        # Use last model's task type
        for node_id, tt in reversed(self.task_history):
            if tt != TaskType.AUTO:
                return tt
        return TaskType.REGRESSION


# In ModelNode.execute():
def execute(self, context, ...):
    # Detect this model's task type
    local_task_type = detect_task_type(self.operator, y_train)

    # Update context
    context = replace(
        context,
        task_type_context=context.task_type_context.update(
            self.node_id, local_task_type
        )
    )

    # Use appropriate metrics based on local task type
    if local_task_type == TaskType.CLASSIFICATION:
        scorer = accuracy_score
    else:
        scorer = r2_score
    ...
```

#### Result Handling for Mixed Tasks

```python
@dataclass
class RunResult:
    """Pipeline execution result with mixed task support."""
    predictions: np.ndarray
    task_type: TaskType  # Final/global task type
    task_history: List[Tuple[str, TaskType]]  # Full history

    @property
    def is_classification(self) -> bool:
        return self.task_type == TaskType.CLASSIFICATION

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    def score(self, y_true: np.ndarray) -> float:
        """Compute appropriate score based on task type."""
        if self.is_classification:
            return accuracy_score(y_true, self.predictions)
        return r2_score(y_true, self.predictions)
```

This design:
- **Preserves v1 flexibility**: Mixed classifiers/regressors work as before
- **Auto-detects when possible**: Reduces user configuration burden
- **Tracks lineage**: Task type changes are recorded for debugging
- **Chooses appropriate metrics**: Each model uses metrics matching its task type


class AggregationStrategy(Protocol):
    """Protocol for fold aggregation strategies."""

    def aggregate(
        self,
        predictions: np.ndarray,
        scores: Optional[List[float]] = None,
        sizes: Optional[List[int]] = None,
        task_type: str = "regression"
    ) -> np.ndarray:
        """Aggregate predictions from multiple folds."""
        ...


class MeanStrategy:
    """Simple mean aggregation."""

    def aggregate(self, predictions, scores=None, sizes=None, task_type="regression"):
        return predictions.mean(axis=0)


class WeightedMeanStrategy:
    """Weighted mean by validation score."""

    def __init__(self, invert_for_error: bool = True):
        self.invert_for_error = invert_for_error

    def aggregate(self, predictions, scores=None, sizes=None, task_type="regression"):
        if scores is None:
            return predictions.mean(axis=0)

        weights = np.array(scores)
        # For error metrics (MSE, MAE), lower is better → invert
        if self.invert_for_error and np.mean(weights) > 1:
            weights = 1 / (weights + 1e-10)
        # Normalize
        weights = weights / weights.sum()

        return np.average(predictions, axis=0, weights=weights)


class MedianStrategy:
    """Median aggregation (robust to outlier folds)."""

    def aggregate(self, predictions, scores=None, sizes=None, task_type="regression"):
        return np.median(predictions, axis=0)


class VoteStrategy:
    """Voting for classification."""

    def aggregate(self, predictions, scores=None, sizes=None, task_type="regression"):
        from scipy import stats
        mode_result = stats.mode(predictions, axis=0, keepdims=False)
        return mode_result.mode
```

### ForkNode and JoinNode

Handle branching and merging in pipelines.

#### Branch Naming Convention

NIRS4ALL uses a hierarchical naming scheme for nested branches to maintain clear lineage:

```
Level 0:  branch_0, branch_1, branch_2, ...
Level 1:  branch_0-0, branch_0-1, branch_1-0, branch_1-1, ...
Level 2:  branch_0-0-0, branch_0-0-1, branch_0-1-0, ...
```

- **Top-level branches**: Simple numeric suffix (`branch_0`, `branch_1`)
- **Nested branches**: Parent name + hyphen + child index (`branch_0-0`, `branch_0-1`)
- **Custom names**: User-provided names are prefixed to maintain hierarchy (`preprocessing-snv-0`, `preprocessing-msc-1`)

This naming scheme ensures:
1. **Unique identification**: Every node has a globally unique path
2. **Lineage tracking**: Parent branches can be inferred from the name
3. **Artifact organization**: Artifacts are stored in hierarchical directories
4. **Debugging clarity**: Log messages show full branch path

```python
# Example: Nested branch naming in practice
pipeline = [
    {"branch": [                              # Creates branch_0, branch_1
        [
            {"branch": [                      # Creates branch_0-0, branch_0-1
                [SNV()],
                [MSC()]
            ]},
            {"merge": "features"}
        ],
        [Detrend()]
    ]},
    {"merge": "predictions"}
]

# Resulting branch names in execution:
# - branch_0 -> branch_0-0 (SNV)
# - branch_0 -> branch_0-1 (MSC)
# - branch_1 (Detrend)
```

```python
@dataclass
class ForkNode(DAGNode):
    """Branch point that creates parallel execution paths.

    A ForkNode creates N output edges, each leading to a separate
    branch of the pipeline. The fork can be:
    - Explicit: {"branch": [[step1], [step2]]}
    - Source-based: {"source_branch": {...}}
    - Generator-based: {"_or_": [...]}
    """
    node_type: NodeType = NodeType.FORK
    branch_count: int = 0
    branch_names: List[str] = field(default_factory=list)

    def execute(
        self,
        context: DatasetContext,
        input_view: ViewSpec,
        mode: str = "train",
        artifacts: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DatasetContext, ViewSpec]]:
        """Execute fork, creating branch contexts.

        Returns:
            List of (context, view) tuples, one per branch
        """
        branch_outputs = []

        for i in range(self.branch_count):
            # Create branch-specific context
            branch_context = replace(
                context,
                metadata={
                    **context.metadata,
                    "branch_id": i,
                    "branch_name": self.branch_names[i] if i < len(self.branch_names) else f"branch_{i}"
                }
            )

            # Create branch-specific view
            branch_view = replace(
                input_view,
                metadata={
                    **input_view.metadata,
                    "branch_id": i
                }
            )

            branch_outputs.append((branch_context, branch_view))

        return branch_outputs


@dataclass
class JoinNode(DAGNode):
    """Merge point that combines parallel execution paths.

    A JoinNode has N input edges (from branches) and 1 output edge.
    Merge strategies:
    - "features": Concatenate transformed features
    - "predictions": Use OOF predictions as meta-features
    - "best": Select best-performing branch
    """
    node_type: NodeType = NodeType.JOIN
    merge_strategy: str = "features"

    def execute(
        self,
        contexts: List[DatasetContext],
        views: List[ViewSpec],
        prediction_store: Optional[PredictionStore] = None,
        mode: str = "train",
        artifacts: Optional[Dict[str, Any]] = None
    ) -> Tuple[DatasetContext, ViewSpec, Dict[str, Any]]:
        """Execute merge, combining branch outputs.

        Args:
            contexts: Contexts from each branch
            views: Views from each branch
            prediction_store: For prediction-based merging
            mode: "train" or "predict"
            artifacts: Pre-loaded artifacts

        Returns:
            Tuple of (merged context, merged view, artifacts)
        """
        if self.merge_strategy == "features":
            return self._merge_features(contexts, views, mode)
        elif self.merge_strategy == "predictions":
            return self._merge_predictions(contexts, views, prediction_store, mode)
        elif self.merge_strategy == "best":
            return self._select_best(contexts, views, prediction_store)
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")

    def _merge_features(
        self,
        contexts: List[DatasetContext],
        views: List[ViewSpec],
        mode: str
    ) -> Tuple[DatasetContext, ViewSpec, Dict[str, Any]]:
        """Merge by concatenating features from all branches."""
        # Collect all active block IDs
        all_block_ids = []
        for ctx in contexts:
            all_block_ids.extend(ctx.active_block_ids)

        # Use first context as base
        merged_context = replace(
            contexts[0],
            active_block_ids=all_block_ids,
            metadata={
                **contexts[0].metadata,
                "merged_from": [ctx.metadata.get("branch_name") for ctx in contexts]
            }
        )

        # Create merged view
        merged_view = replace(
            views[0],
            block_ids=tuple(all_block_ids)
        )

        return merged_context, merged_view, {}

    def _merge_predictions(
        self,
        contexts: List[DatasetContext],
        views: List[ViewSpec],
        prediction_store: PredictionStore,
        mode: str
    ) -> Tuple[DatasetContext, ViewSpec, Dict[str, Any]]:
        """Merge by using OOF predictions as meta-features.

        Raises:
            DAGValidationError: If CV is not enabled when using prediction merge.
                Using predictions without CV would cause data leakage because
                train predictions would be used as features for training.
        """
        # Validate that CV was used - critical to prevent data leakage
        for i, ctx in enumerate(contexts):
            if not ctx.has_cv_folds:
                raise DAGValidationError(
                    f"merge_strategy='predictions' requires cross-validation. "
                    f"Branch '{ctx.metadata.get('branch_name', i)}' has no CV folds. "
                    f"Add a Splitter before branching or use merge_strategy='features'. "
                    f"Without CV, train predictions would leak into meta-features."
                )

        # Get OOF predictions from each branch model
        branch_predictions = []

        for i, ctx in enumerate(contexts):
            branch_name = ctx.metadata.get("branch_name", f"branch_{i}")

            # Get OOF predictions for this branch
            oof_preds = prediction_store.get_oof_predictions(
                branch_name=branch_name,
                partition="val"
            )

            if oof_preds:
                # Reconstruct full OOF prediction array
                oof_array = self._reconstruct_oof(oof_preds, ctx)
                branch_predictions.append(oof_array)

        if not branch_predictions:
            raise ValueError("No OOF predictions found for merging")

        # Stack predictions as meta-features
        X_meta = np.column_stack(branch_predictions)

        # Create new block for meta-features
        meta_block_id = contexts[0].block_store.register_source(
            data=X_meta.reshape(X_meta.shape[0], 1, -1),
            source_name="meta_predictions",
            metadata={"merge_type": "predictions"}
        )

        merged_context = replace(
            contexts[0],
            active_block_ids=[meta_block_id]
        )

        merged_view = replace(
            views[0],
            block_ids=(meta_block_id,)
        )

        return merged_context, merged_view, {"X_meta": X_meta}

    def _reconstruct_oof(
        self,
        oof_preds: List[Dict],
        context: DatasetContext
    ) -> np.ndarray:
        """Reconstruct OOF predictions in original sample order."""
        n_samples = context.n_samples
        oof = np.zeros(n_samples)

        for pred in oof_preds:
            indices = pred["sample_indices"]
            y_pred = pred["y_pred"]
            oof[indices] = y_pred

        return oof
```

### SplitterNode

Assigns cross-validation folds.

```python
@dataclass
class SplitterNode(DAGNode):
    """Cross-validation fold assignment node.

    Takes an sklearn splitter and assigns fold_id to samples
    in the sample registry.
    """
    node_type: NodeType = NodeType.SPLITTER

    def execute(
        self,
        context: DatasetContext,
        input_view: ViewSpec,
        mode: str = "train",
        artifacts: Optional[Dict[str, Any]] = None
    ) -> Tuple[DatasetContext, Dict[str, Any]]:
        """Execute fold assignment."""
        if mode == "predict":
            # No fold assignment in predict mode
            return context, {}

        splitter = self.operator

        # Get training data for splitting
        X = context.resolver.materialize_X(
            input_view.with_partition("train")
        )
        y = context.target_store.get(version="raw")

        # Get groups if GroupKFold
        groups = None
        if hasattr(splitter, 'get_n_splits'):
            try:
                # Check if groups are required
                import inspect
                sig = inspect.signature(splitter.split)
                if 'groups' in sig.parameters:
                    group_col = context.sample_registry._df.filter(
                        pl.col("partition") == "train"
                    )["group"].to_numpy()
                    if group_col is not None:
                        groups = group_col
            except:
                pass

        # Assign folds
        context.sample_registry.assign_folds(
            splitter=splitter,
            X=X,
            y=y,
            groups=groups
        )

        return context, {"splitter": splitter}
```

### MidFusionNode

Combines intermediate representations from neural networks.

```python
@dataclass
class MidFusionNode(DAGNode):
    """Mid-level fusion node for neural network tower concatenation.

    Extracts intermediate representations from trained models,
    concatenates them, and trains a fusion head. This enables
    learning combined representations while preserving source-specific
    patterns captured by individual models.

    Supports:
    - TensorFlow/Keras: Model layer extraction
    - PyTorch: Module indexing
    - Configurable truncation point
    - Freezing base models during fusion training
    """
    node_type: NodeType = NodeType.JOIN
    truncate_at: Union[int, str] = -2  # Second to last layer
    freeze_base: bool = True
    fusion_head: Optional[Any] = None
    fusion_strategy: Literal["concat", "add", "attention", "gated"] = "concat"

    def execute(
        self,
        branch_contexts: List[DatasetContext],
        branch_artifacts: List[Dict[str, Any]],
        view: ViewSpec,
        prediction_store: PredictionStore,
        mode: str = "train"
    ) -> Tuple[DatasetContext, Dict[str, Any]]:
        """Execute mid-fusion.

        Args:
            branch_contexts: Contexts from each branch (with trained models)
            branch_artifacts: Model artifacts from each branch
            view: Current data view
            prediction_store: For collecting fusion predictions
            mode: "train" or "predict"

        Returns:
            Tuple of (merged context, fusion artifacts)
        """
        if mode == "train":
            return self._train_fusion(
                branch_contexts, branch_artifacts, view, prediction_store
            )
        else:
            return self._predict_fusion(
                branch_contexts, view, prediction_store
            )

    def _train_fusion(
        self,
        contexts: List[DatasetContext],
        artifacts: List[Dict[str, Any]],
        view: ViewSpec,
        prediction_store: PredictionStore
    ) -> Tuple[DatasetContext, Dict[str, Any]]:
        """Train fusion head on intermediate representations."""
        # Extract base models from artifacts
        base_models = []
        for art in artifacts:
            if "virtual_model" in art:
                base_models.append(art["virtual_model"].primary_model)
            elif "model" in art:
                base_models.append(art["model"])
            else:
                raise ValueError("No model found in branch artifacts")

        # Truncate models to extract representations
        truncated_models = []
        for model in base_models:
            truncated = self._truncate_model(model)
            if self.freeze_base:
                self._freeze_model(truncated)
            truncated_models.append(truncated)

        # Extract representations for training data
        representations = []
        for ctx, trunc_model in zip(contexts, truncated_models):
            X = ctx.resolver.materialize_X(view.with_partition("train"))
            h = self._get_representation(trunc_model, X)
            representations.append(h)

        # Combine representations
        H = self._combine_representations(representations)

        # Get targets
        y = contexts[0].target_store.get(version="raw")

        # Train fusion head
        fusion_head = clone(self.fusion_head) if self.fusion_head else self._default_head(H.shape[1])
        fusion_head.fit(H, y)

        # Store predictions
        y_pred = fusion_head.predict(H)
        prediction_store.add_prediction(
            partition="train",
            y_pred=y_pred,
            y_true=y,
            model_name="mid_fusion"
        )

        # Validation predictions (per-fold if CV)
        if contexts[0].folds:
            for fold_id, (train_idx, val_idx) in enumerate(contexts[0].folds):
                val_view = view.with_fold(fold_id, "val")

                val_reps = []
                for ctx, trunc_model in zip(contexts, truncated_models):
                    X_val = ctx.resolver.materialize_X(val_view)
                    h_val = self._get_representation(trunc_model, X_val)
                    val_reps.append(h_val)

                H_val = self._combine_representations(val_reps)
                y_pred_val = fusion_head.predict(H_val)
                y_val = contexts[0].target_store.get(sample_indices=val_idx)

                prediction_store.add_prediction(
                    partition="val",
                    fold_id=fold_id,
                    y_pred=y_pred_val,
                    y_true=y_val,
                    sample_indices=val_idx.tolist(),
                    model_name="mid_fusion"
                )

        return contexts[0], {
            "base_models": base_models,
            "truncated_models": truncated_models,
            "fusion_head": fusion_head
        }

    def _combine_representations(
        self,
        representations: List[np.ndarray]
    ) -> np.ndarray:
        """Combine representations using configured strategy."""
        if self.fusion_strategy == "concat":
            return np.concatenate(representations, axis=1)
        elif self.fusion_strategy == "add":
            # Pad to same size if needed
            max_dim = max(r.shape[1] for r in representations)
            padded = [
                np.pad(r, ((0, 0), (0, max_dim - r.shape[1])))
                for r in representations
            ]
            return np.sum(padded, axis=0)
        elif self.fusion_strategy == "attention":
            return self._attention_fusion(representations)
        elif self.fusion_strategy == "gated":
            return self._gated_fusion(representations)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

    def _attention_fusion(
        self,
        representations: List[np.ndarray]
    ) -> np.ndarray:
        """Attention-based fusion (learns attention weights)."""
        # Stack representations: (n_samples, n_branches, hidden_dim)
        # For now, simple average with learned weights
        stacked = np.stack(representations, axis=1)
        # Placeholder: equal weights (would be learnable in NN)
        weights = np.ones(len(representations)) / len(representations)
        return np.einsum('ijk,j->ik', stacked, weights)

    def _truncate_model(self, model: Any) -> Any:
        """Truncate model at specified layer."""
        # TensorFlow/Keras
        if hasattr(model, 'layers'):
            import tensorflow as tf
            if isinstance(self.truncate_at, int):
                layer = model.layers[self.truncate_at]
            else:
                layer = model.get_layer(self.truncate_at)
            return tf.keras.Model(inputs=model.input, outputs=layer.output)

        # PyTorch
        if hasattr(model, 'children'):
            import torch.nn as nn
            layers = list(model.children())
            if isinstance(self.truncate_at, int):
                if self.truncate_at < 0:
                    layers = layers[:self.truncate_at]
                else:
                    layers = layers[:self.truncate_at + 1]
            return nn.Sequential(*layers)

        raise ValueError(
            f"Cannot truncate model of type {type(model)}. "
            "Mid-fusion requires TensorFlow or PyTorch models."
        )

    def _freeze_model(self, model: Any) -> None:
        """Freeze model weights."""
        if hasattr(model, 'trainable'):
            model.trainable = False
        elif hasattr(model, 'parameters'):
            for param in model.parameters():
                param.requires_grad = False

    def _get_representation(
        self,
        truncated_model: Any,
        X: np.ndarray
    ) -> np.ndarray:
        """Get intermediate representation from truncated model."""
        if hasattr(truncated_model, 'predict'):
            return truncated_model.predict(X, verbose=0)
        elif hasattr(truncated_model, 'forward'):
            import torch
            with torch.no_grad():
                X_t = torch.from_numpy(X).float()
                if hasattr(truncated_model, 'device'):
                    X_t = X_t.to(truncated_model.device)
                return truncated_model(X_t).cpu().numpy()
        else:
            raise ValueError(f"Cannot get representation from {type(truncated_model)}")

    def _default_head(self, input_dim: int) -> Any:
        """Create default fusion head."""
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)
```

### LayoutAwareModelNode

Model node that respects model-declared input specifications.

```python
@dataclass
class LayoutAwareModelNode(DAGNode):
    """Model node with automatic layout resolution.

    Inspects the model for input specifications (ModelInputSpec)
    and automatically constructs appropriate input format.
    Supports multi-head models with dictionary inputs.
    """
    node_type: NodeType = NodeType.MODEL

    def execute(
        self,
        context: DatasetContext,
        input_view: ViewSpec,
        prediction_store: PredictionStore,
        mode: str = "train",
        artifacts: Optional[Dict[str, Any]] = None
    ) -> Tuple[DatasetContext, Dict[str, Any]]:
        """Execute model with layout-aware input construction."""
        from nirs4all.data import (
            LayoutResolver, LayoutSpec, ModelWithInputSpec, LAYOUT_SKLEARN
        )

        model = self.operator

        # Determine layout from model or default
        if isinstance(model, ModelWithInputSpec):
            input_spec = model.get_input_spec()
            layout = input_spec.to_layout_spec()

            # Validate required sources are present
            if input_spec.required_sources:
                available = set(
                    context.block_store.get(bid).metadata.get("source")
                    for bid in context.active_block_ids
                )
                missing = set(input_spec.required_sources) - available
                if missing:
                    raise ValueError(
                        f"Model requires sources {input_spec.required_sources} "
                        f"but only {available} are available. Missing: {missing}"
                    )
        else:
            # Default to flat 2D for sklearn-style models
            layout = LAYOUT_SKLEARN

        # Create layout resolver
        resolver = LayoutResolver(context.block_store, context.sample_registry)

        if mode == "train":
            return self._train_with_layout(
                context, input_view, layout, resolver, prediction_store
            )
        else:
            return self._predict_with_layout(
                context, input_view, layout, resolver, artifacts, prediction_store
            )

    def _train_with_layout(
        self,
        context: DatasetContext,
        view: ViewSpec,
        layout: LayoutSpec,
        resolver: LayoutResolver,
        prediction_store: PredictionStore
    ) -> Tuple[DatasetContext, Dict[str, Any]]:
        """Train model with layout-resolved inputs."""
        fold_models = []
        fold_scores = []

        for fold_id, (train_idx, val_idx) in enumerate(context.folds):
            # Clone model for this fold
            fold_model = clone(self.operator)

            # Resolve inputs with layout
            train_view = view.with_fold(fold_id, "train")
            val_view = view.with_fold(fold_id, "val")

            X_train = resolver.resolve(train_view, layout)
            X_val = resolver.resolve(val_view, layout)
            y_train = context.target_store.get(sample_indices=train_idx)
            y_val = context.target_store.get(sample_indices=val_idx)

            # Train
            fold_model.fit(X_train, y_train)

            # Predict
            y_pred_val = fold_model.predict(X_val)

            # Score
            score = self._compute_score(y_val, y_pred_val, context)

            # Store predictions
            prediction_store.add_prediction(
                partition="val",
                fold_id=fold_id,
                y_pred=y_pred_val,
                y_true=y_val,
                sample_indices=val_idx.tolist(),
                val_score=score
            )

            fold_models.append(fold_model)
            fold_scores.append(score)

        # Create virtual model
        virtual_model = VirtualModel(
            fold_models=fold_models,
            fold_scores=fold_scores
        )

        return context, {
            "virtual_model": virtual_model,
            "layout": layout
        }
```

---

## DAG Builder

The DAG Builder compiles pipeline syntax into an executable DAG.

### Interface

```python
class DAGBuilder:
    """Builds executable DAG from pipeline syntax.

    Handles:
    - Step parsing and normalization
    - Generator expansion (pre-execution)
    - Node creation and linking
    - Fork/join inference
    """

    def __init__(self):
        self._node_counter = 0
        self._nodes: Dict[str, DAGNode] = {}
        self._edges: List[DAGEdge] = []

    def build(
        self,
        pipeline: List[Any],
        context: Optional[DatasetContext] = None,
        max_variants: int = 1000,
        warn_threshold: int = 100
    ) -> ExecutableDAG:
        """Build DAG from pipeline definition.

        Args:
            pipeline: List of pipeline steps
            context: Optional context for dynamic resolution
            max_variants: Maximum number of generator variants allowed.
                Exceeding raises GeneratorExplosionError. Default 1000.
            warn_threshold: Threshold for GeneratorExplosionWarning.
                Default 100.

        Returns:
            Executable DAG

        Raises:
            GeneratorExplosionError: If generator expansion exceeds max_variants.
        """
        # Phase 1: Expand generators (with limits)
        expanded = self._expand_generators(pipeline, max_variants, warn_threshold)

        # Phase 2: Parse steps
        parsed_steps = [self._parse_step(step) for step in expanded]

        # Phase 3: Build nodes
        self._build_nodes(parsed_steps)

        # Phase 4: Link nodes
        self._link_nodes()

        # Phase 5: Validate DAG
        self._validate()

        return ExecutableDAG(
            dag_id=self._generate_dag_id(),
            nodes=self._nodes,
            edges=self._edges,
            source_nodes=self._find_sources(),
            sink_nodes=self._find_sinks()
        )

    def _validate(self) -> None:
        """Validate DAG structure.

        Raises:
            DAGCycleError: If cycle detected
            DAGValidationError: If structure is invalid
        """
        # Cycle detection via DFS
        self._detect_cycles()

        # Other validations
        self._validate_sources()
        self._validate_sinks()
        self._validate_fork_join_pairing()

    def _detect_cycles(self) -> None:
        """Detect cycles using DFS with three-color marking.

        Colors:
        - WHITE (0): Not visited
        - GRAY (1): Currently in recursion stack
        - BLACK (2): Fully processed

        If we encounter a GRAY node during DFS, a cycle exists.

        Raises:
            DAGCycleError: If cycle is detected, with the cycle path
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in self._nodes}
        path = []

        def dfs(node_id: str) -> Optional[List[str]]:
            color[node_id] = GRAY
            path.append(node_id)

            # Get successors
            successors = [
                e.target_id for e in self._edges
                if e.source_id == node_id
            ]

            for succ in successors:
                if color[succ] == GRAY:
                    # Found cycle - extract cycle path
                    cycle_start = path.index(succ)
                    cycle = path[cycle_start:] + [succ]
                    return cycle
                elif color[succ] == WHITE:
                    result = dfs(succ)
                    if result:
                        return result

            path.pop()
            color[node_id] = BLACK
            return None

        for node_id in self._nodes:
            if color[node_id] == WHITE:
                cycle = dfs(node_id)
                if cycle:
                    raise DAGCycleError(
                        f"Cycle detected in DAG: {' -> '.join(cycle)}. "
                        f"Check for circular dependencies in pipeline definition."
                    )

    def _expand_generators(
        self,
        pipeline: List[Any],
        max_variants: int = 1000,
        warn_threshold: int = 100
    ) -> List[Any]:
        """Expand all generator syntax before building.

        Generators (_or_, _range_, etc.) are expanded to explicit
        branch structures.

        Args:
            pipeline: The pipeline steps to expand.
            max_variants: Maximum number of variants allowed. Exceeding this
                raises GeneratorExplosionError. Default 1000.
            warn_threshold: Threshold for warning about large expansions.
                Default 100.

        Returns:
            Expanded pipeline with generators converted to branch structures.

        Raises:
            GeneratorExplosionError: If total variants exceed max_variants.

        Note:
            Use `estimate_variants()` to preview expansion size before building.
        """
        expanded = []
        total_variants = 0

        for step in pipeline:
            if self._is_generator(step):
                # Convert generator to explicit branch
                variants = self._expand_generator_step(step)
                variant_count = len(variants)
                total_variants += variant_count

                # Check limits BEFORE adding to expanded
                if total_variants > max_variants:
                    raise GeneratorExplosionError(
                        f"Generator expansion would create {total_variants} variants, "
                        f"exceeding limit of {max_variants}. "
                        f"Use estimate_variants() to preview, or increase max_variants "
                        f"if intentional."
                    )

                if total_variants > warn_threshold:
                    import warnings
                    warnings.warn(
                        f"Generator expansion creating {total_variants} variants. "
                        f"Consider reducing complexity or using subset sampling.",
                        GeneratorExplosionWarning
                    )

                expanded.append({
                    "branch": [[v] for v in variants],
                    "_generator_source": step,  # Keep for debugging
                    "_variant_count": variant_count  # Track for diagnostics
                })
            elif isinstance(step, dict) and "branch" in step:
                # Recursively expand generators in branch definitions
                branch_def = step["branch"]
                if self._is_generator(branch_def):
                    variants = self._expand_generator_step(branch_def)
                    step = {"branch": [[v] for v in variants]}
                else:
                    # Expand generators within each branch
                    expanded_branches = []
                    for branch in branch_def:
                        if isinstance(branch, list):
                            expanded_branches.append(
                                self._expand_generators(branch, max_variants, warn_threshold)
                            )
                        else:
                            expanded_branches.append([branch])
                    step = {"branch": expanded_branches}
                expanded.append(step)
            else:
                expanded.append(step)

        return expanded

    def estimate_variants(self, pipeline: List[Any]) -> int:
        """Preview the total number of variants without building.

        Use this before build() to check if generator expansion is reasonable.

        Args:
            pipeline: The pipeline definition to analyze.

        Returns:
            Estimated total number of pipeline variants.

        Example:
            >>> builder = DAGBuilder()
            >>> n = builder.estimate_variants(pipeline)
            >>> print(f"Pipeline will expand to {n} variants")
            >>> if n > 100:
            ...     print("Consider reducing complexity")
        """
        count = 0
        for step in pipeline:
            if self._is_generator(step):
                count += len(self._expand_generator_step(step))
            elif isinstance(step, dict) and "branch" in step:
                branch_def = step["branch"]
                if self._is_generator(branch_def):
                    count += len(self._expand_generator_step(branch_def))
                elif isinstance(branch_def, list):
                    for branch in branch_def:
                        if isinstance(branch, list):
                            count += self.estimate_variants(branch)
        return max(count, 1)  # At least 1 variant (the base pipeline)

    def _is_generator(self, step: Any) -> bool:
        """Check if step is a generator."""
        if isinstance(step, dict):
            return any(k in step for k in ("_or_", "_range_", "_grid_"))
        return False

    def _expand_generator_step(self, step: Dict) -> List[Any]:
        """Expand a single generator step."""
        if "_or_" in step:
            return step["_or_"]
        elif "_range_" in step:
            start, end, step_size = step["_range_"]
            return list(range(start, end, step_size))
        elif "_grid_" in step:
            import itertools
            params = step["_grid_"]
            keys = list(params.keys())
            values = list(params.values())
            return [
                dict(zip(keys, combo))
                for combo in itertools.product(*values)
            ]
        return [step]

    def _parse_step(self, step: Any) -> ParsedStep:
        """Parse a step into standardized form."""
        # Normalize different syntaxes
        if isinstance(step, type):
            # Class reference: MinMaxScaler
            return ParsedStep(
                operator=step(),
                keyword="preprocessing",
                operator_class=f"{step.__module__}.{step.__name__}",
                params={}
            )
        elif hasattr(step, 'fit') or hasattr(step, 'transform'):
            # Instance: MinMaxScaler()
            return ParsedStep(
                operator=step,
                keyword="preprocessing",
                operator_class=f"{step.__class__.__module__}.{step.__class__.__name__}",
                params=step.get_params() if hasattr(step, 'get_params') else {}
            )
        elif isinstance(step, dict):
            # Dict with keyword: {"model": PLSRegression()}
            return self._parse_dict_step(step)
        else:
            raise ValueError(f"Unknown step format: {step}")

    def _parse_dict_step(self, step: Dict) -> ParsedStep:
        """Parse dictionary step."""
        KEYWORDS = {
            "model", "meta_model", "preprocessing", "transform",
            "y_processing", "feature_augmentation", "sample_augmentation",
            "branch", "merge", "source_branch", "merge_sources",
            "splitter", "resampler", "feature_selection",
            "outlier_excluder", "sample_filter"
        }


### Controller Auto-Identification

The DAGBuilder creates nodes from parsed steps, but **node execution is delegated to controllers**
identified at runtime via the **ControllerRegistry**. This means:

1. **Users write operators directly** - no need to specify controllers:
   ```python
   # User just provides sklearn/torch/jax operators
   pipeline = [SNV(), PLSRegression(n_components=10)]
   ```

2. **Controllers are matched automatically** via `matches()` introspection:
   ```python
   class ControllerRegistry:
       """Priority-based controller dispatch."""

       controllers: List[Type[OperatorController]]  # Sorted by priority

       def get_controller(self, node: DAGNode) -> OperatorController:
           """Find appropriate controller for node's operator."""
           for controller_cls in self.controllers:
               if controller_cls.matches(
                   step=node.metadata.get("step_config"),
                   operator=node.operator,
                   keyword=node.metadata.get("keyword", "")
               ):
                   return controller_cls()

           raise ControllerNotFoundError(
               f"No controller found for operator: {node.operator}. "
               f"Ensure it implements TransformerMixin, or has fit/predict methods, "
               f"or register a custom controller."
           )
   ```

3. **Matching uses interface introspection**:
   ```python
   # TransformController.matches()
   @classmethod
   def matches(cls, step, operator, keyword) -> bool:
       return isinstance(operator, TransformerMixin)

   # SklearnModelController.matches()
   @classmethod
   def matches(cls, step, operator, keyword) -> bool:
       return (
           isinstance(operator, BaseEstimator) and
           hasattr(operator, 'predict') and
           not isinstance(operator, TransformerMixin)
       )

   # TorchModelController.matches()
   @classmethod
   def matches(cls, step, operator, keyword) -> bool:
       import torch.nn as nn
       return isinstance(operator, nn.Module)
   ```

4. **Priority resolves conflicts** (lower value = higher priority):
   ```python
   CONTROLLER_REGISTRY = [
       (TFModelController, priority=4),      # TensorFlow models
       (TorchModelController, priority=5),   # PyTorch models
       (SklearnModelController, priority=6), # sklearn predictors
       (TransformController, priority=10),   # sklearn transformers
       # Custom controllers can have priority 1-3 to override defaults
   ]
   ```

This design preserves v1 behavior where users can mix sklearn, TensorFlow, PyTorch,
and JAX operators freely - the appropriate controller is selected automatically.

        for keyword in KEYWORDS:
            if keyword in step:
                operator = step[keyword]
                return ParsedStep(
                    operator=operator,
                    keyword=keyword,
                    operator_class=self._get_operator_class(operator),
                    params=self._get_operator_params(operator),
                    step_config=step
                )

        # Check for serialized format
        if "class" in step:
            return self._parse_serialized_step(step)

        raise ValueError(f"Unknown step format: {step}")

    def _build_nodes(self, parsed_steps: List[ParsedStep]) -> None:
        """Build nodes from parsed steps."""
        for step in parsed_steps:
            node = self._create_node(step)
            self._nodes[node.node_id] = node

    def _create_node(self, step: ParsedStep) -> DAGNode:
        """Create appropriate node type for step."""
        node_id = self._next_node_id()

        NODE_TYPE_MAP = {
            "preprocessing": NodeType.TRANSFORM,
            "transform": NodeType.TRANSFORM,
            "y_processing": NodeType.Y_TRANSFORM,
            "model": NodeType.MODEL,
            "meta_model": NodeType.MODEL,
            "splitter": NodeType.SPLITTER,
            "branch": NodeType.FORK,
            "source_branch": NodeType.FORK,
            "merge": NodeType.JOIN,
            "merge_sources": NodeType.JOIN,
            "sample_augmentation": NodeType.AUGMENT,
            "feature_augmentation": NodeType.FEATURE_AUG,
            "sample_filter": NodeType.FILTER,
            "outlier_excluder": NodeType.FILTER,
        }

        node_type = NODE_TYPE_MAP.get(step.keyword, NodeType.TRANSFORM)

        return DAGNode(
            node_id=node_id,
            node_type=node_type,
            operator=step.operator,
            operator_class=step.operator_class,
            operator_params=step.params,
            metadata={"keyword": step.keyword}
        )

    def _link_nodes(self) -> None:
        """Create edges between nodes."""
        node_list = list(self._nodes.values())

        for i in range(len(node_list) - 1):
            current = node_list[i]
            next_node = node_list[i + 1]

            if current.node_type == NodeType.FORK:
                # Fork creates multiple output edges
                self._create_fork_edges(current, next_node)
            elif current.node_type == NodeType.JOIN:
                # Join has single output
                self._edges.append(DAGEdge(
                    source_id=current.node_id,
                    target_id=next_node.node_id
                ))
            else:
                # Normal sequential edge
                self._edges.append(DAGEdge(
                    source_id=current.node_id,
                    target_id=next_node.node_id
                ))

    def _next_node_id(self) -> str:
        """Generate next node ID."""
        self._node_counter += 1
        return f"node_{self._node_counter:03d}"


@dataclass
class ParsedStep:
    """Standardized step representation."""
    operator: Any
    keyword: str
    operator_class: str
    params: Dict[str, Any]
    step_config: Optional[Dict] = None
```

---

## Execution Engine

The Execution Engine runs the compiled DAG.

### Interface

```python
class ExecutionEngine:
    """Executes compiled DAG with support for parallelism.

    Handles:
    - Topological ordering
    - Level-based parallelization
    - Artifact management
    - Prediction collection

    Thread Safety:
        When parallel=True, the engine uses ThreadPoolExecutor for concurrent
        node execution. Thread safety is ensured by:

        1. **Copy-on-Write contexts**: Each parallel branch receives a shallow
           copy of the DatasetContext. Modifications create new FeatureBlocks
           rather than mutating shared data. See Data Layer Thread-Safety
           section in 02_data_layer.md.

        2. **Thread-safe FeatureBlockStore**: Block registration uses locks;
           blocks themselves are immutable.

        3. **Isolated prediction stores**: Each branch writes to its own
           prediction buffer; JoinNode aggregates them atomically.

        4. **Barrier synchronization**: ForkBarrier ensures all branches
           complete before JoinNode proceeds, preventing race conditions
           on merge operations.

        For CPU-bound transformations (e.g., heavy preprocessing), consider
        parallel=False with external parallelization (joblib) for better
        performance due to Python's GIL.

    GIL Considerations:
        Python's Global Interpreter Lock (GIL) limits true parallelism for
        CPU-bound Python code. The engine provides two parallelism strategies:

        1. **ThreadPoolExecutor** (default, parallel_backend="threads"):
           - Best for I/O-bound tasks or when operators release the GIL
           - NumPy, scikit-learn, and most C-extensions release the GIL
           - Low overhead for task spawning
           - Shared memory access (no pickling)

        2. **ProcessPoolExecutor** (parallel_backend="processes"):
           - Best for pure Python CPU-bound code
           - True parallelism, bypasses GIL
           - Higher overhead (pickling, memory duplication)
           - Use when transforms are Python-heavy

        Recommendation:
        - For sklearn/NumPy pipelines: Use threads (default)
        - For custom Python transforms: Use processes
        - For mixed workloads: Use threads, let NumPy parallelize internally
    """

    def __init__(
        self,
        parallel: bool = False,
        n_workers: int = 4,
        parallel_backend: str = "threads",  # "threads" or "processes"
        cache_dir: Optional[Path] = None,
        node_timeout: Optional[float] = None  # Timeout in seconds per node
    ):
        self.parallel = parallel
        self.n_workers = n_workers
        self.parallel_backend = parallel_backend
        self.cache_dir = cache_dir
        self.node_timeout = node_timeout
        self._executor = None


class NodeTimeoutError(DAGError):
    """Raised when a node exceeds its execution timeout."""
    def __init__(self, node_id: str, timeout: float):
        super().__init__(f"Node '{node_id}' exceeded timeout of {timeout}s")
        self.node_id = node_id
        self.timeout = timeout

    def _get_executor(self):
        """Get appropriate executor based on backend."""
        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

        if self.parallel_backend == "processes":
            return ProcessPoolExecutor(max_workers=self.n_workers)
        else:
            return ThreadPoolExecutor(max_workers=self.n_workers)

    def execute(
        self,
        dag: ExecutableDAG,
        context: DatasetContext,
        mode: str = "train",
        artifacts: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> ExecutionResult:
        """Execute DAG on context.

        Args:
            dag: Compiled DAG to execute
            context: Initial dataset context
            mode: "train" or "predict"
            artifacts: Pre-loaded artifacts for predict mode

        Returns:
            ExecutionResult with predictions, artifacts, metrics
        """
        prediction_store = PredictionStore()
        collected_artifacts = {}

        # Get execution order
        if self.parallel:
            levels = dag.get_execution_levels()
            for level in levels:
                if len(level) > 1:
                    results = self._execute_parallel(
                        dag, level, context, mode, artifacts, prediction_store
                    )
                else:
                    result = self._execute_node(
                        dag, level[0], context, mode, artifacts, prediction_store
                    )
                    context = result.context
                    collected_artifacts[level[0]] = result.artifacts
        else:
            for node_id in dag.topological_order():
                result = self._execute_node(
                    dag, node_id, context, mode, artifacts, prediction_store
                )
                context = result.context
                collected_artifacts[node_id] = result.artifacts

        return ExecutionResult(
            context=context,
            prediction_store=prediction_store,
            artifacts=collected_artifacts,
            dag=dag
        )

    def _execute_node(
        self,
        dag: ExecutableDAG,
        node_id: str,
        context: DatasetContext,
        mode: str,
        artifacts: Optional[Dict],
        prediction_store: PredictionStore
    ) -> NodeResult:
        """Execute a single node."""
        node = dag.get_node(node_id)

        # Get input view from incoming edges
        input_view = self._resolve_input_view(dag, node_id, context)

        # Get node-specific artifacts
        node_artifacts = None
        if artifacts and node_id in artifacts:
            node_artifacts = artifacts[node_id]

        # Execute based on node type
        if node.node_type == NodeType.TRANSFORM:
            new_context, new_artifacts = self._execute_transform(
                node, context, input_view, mode, node_artifacts
            )
        elif node.node_type == NodeType.MODEL:
            new_context, new_artifacts = self._execute_model(
                node, context, input_view, prediction_store, mode, node_artifacts
            )
        elif node.node_type == NodeType.FORK:
            return self._execute_fork(
                dag, node, context, input_view, mode, artifacts, prediction_store
            )
        elif node.node_type == NodeType.JOIN:
            new_context, new_artifacts = self._execute_join(
                dag, node, context, mode, prediction_store, artifacts
            )
        elif node.node_type == NodeType.SPLITTER:
            new_context, new_artifacts = self._execute_splitter(
                node, context, input_view, mode, node_artifacts
            )
        else:
            raise ValueError(f"Unknown node type: {node.node_type}")

        return NodeResult(
            context=new_context,
            artifacts=new_artifacts
        )

    def _execute_fork(
        self,
        dag: ExecutableDAG,
        node: DAGNode,
        context: DatasetContext,
        input_view: ViewSpec,
        mode: str,
        artifacts: Dict,
        prediction_store: PredictionStore
    ) -> NodeResult:
        """Execute fork and all downstream branch nodes."""
        branch_definitions = node.operator
        branch_results = []

        for branch_id, branch_steps in enumerate(branch_definitions):
            # Create branch context
            branch_context = replace(
                context,
                metadata={
                    **context.metadata,
                    "branch_id": branch_id,
                    "branch_path": context.metadata.get("branch_path", []) + [branch_id]
                }
            )

            # Build sub-DAG for branch
            branch_dag = DAGBuilder().build(branch_steps, branch_context)

            # Execute branch
            branch_result = self.execute(
                branch_dag,
                branch_context,
                mode,
                artifacts
            )

            branch_results.append(branch_result)

        return NodeResult(
            context=context,  # Original context
            artifacts={"branch_results": branch_results},
            branch_results=branch_results
        )

    def _execute_parallel(
        self,
        dag: ExecutableDAG,
        node_ids: List[str],
        context: DatasetContext,
        mode: str,
        artifacts: Dict,
        prediction_store: PredictionStore
    ) -> List[NodeResult]:
        """Execute multiple nodes in parallel."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(
                    self._execute_node,
                    dag, node_id, context, mode, artifacts, prediction_store
                )
                for node_id in node_ids
            ]
            return [f.result() for f in futures]


@dataclass
class ExecutionResult:
    """Result of DAG execution."""
    context: DatasetContext
    prediction_store: PredictionStore
    artifacts: Dict[str, Dict[str, Any]]
    dag: ExecutableDAG

    @property
    def predictions(self) -> PredictionStore:
        return self.prediction_store

    def get_best(self, metric: str = "val_score") -> Dict:
        return self.prediction_store.get_best(metric=metric)


@dataclass
class NodeResult:
    """Result of single node execution."""
    context: DatasetContext
    artifacts: Dict[str, Any]
    branch_results: Optional[List["ExecutionResult"]] = None
```

---

## DAG Visualization

Complex pipelines with branches and merges can be difficult to understand. The DAG engine provides visualization tools:

### Graphviz Export

```python
class ExecutableDAG:
    def to_graphviz(
        self,
        show_shapes: bool = True,
        show_timing: bool = False,
        highlight_path: Optional[List[str]] = None
    ) -> str:
        """Export DAG to Graphviz DOT format.

        Args:
            show_shapes: Include data shapes at each node
            show_timing: Include execution timing (after execution)
            highlight_path: Node IDs to highlight (e.g., error path)

        Returns:
            DOT format string

        Example:
            dot = dag.to_graphviz(show_shapes=True)
            # Render with: dot -Tpng dag.dot -o dag.png
        """
        lines = ["digraph Pipeline {", "  rankdir=TB;", "  node [shape=box];"]

        for node in self.nodes.values():
            label = f"{node.name}\\n{node.node_type.value}"
            if show_shapes and hasattr(node, "output_shape"):
                label += f"\\n{node.output_shape}"
            if show_timing and hasattr(node, "execution_time"):
                label += f"\\n{node.execution_time:.2f}s"

            style = ""
            if highlight_path and node.node_id in highlight_path:
                style = ', style=filled, fillcolor=red'

            lines.append(f'  "{node.node_id}" [label="{label}"{style}];')

        for edge in self.edges:
            lines.append(f'  "{edge.source}" -> "{edge.target}";')

        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Export DAG to Mermaid.js format for web display.

        Returns:
            Mermaid format string

        Example:
            mermaid = dag.to_mermaid()
            # Embed in markdown: ```mermaid\\n{mermaid}\\n```
        """
        lines = ["graph TD"]
        for node in self.nodes.values():
            shape_start, shape_end = ("[", "]")  # Rectangle
            if node.node_type == NodeType.FORK:
                shape_start, shape_end = ("{", "}")  # Diamond
            elif node.node_type == NodeType.JOIN:
                shape_start, shape_end = ("((", "))")  # Circle

            lines.append(f"  {node.node_id}{shape_start}{node.name}{shape_end}")

        for edge in self.edges:
            lines.append(f"  {edge.source} --> {edge.target}")

        return "\n".join(lines)

    def visualize(
        self,
        output_path: Optional[Path] = None,
        format: str = "png",
        view: bool = True
    ) -> Optional[Path]:
        """Render and optionally display DAG visualization.

        Requires graphviz to be installed (pip install graphviz).

        Args:
            output_path: Where to save the image
            format: Output format (png, svg, pdf)
            view: Open the image after rendering

        Returns:
            Path to rendered image, or None if view-only
        """
        try:
            import graphviz
            dot = graphviz.Source(self.to_graphviz())
            if output_path:
                dot.render(output_path.stem, directory=output_path.parent,
                          format=format, view=view, cleanup=True)
                return output_path.with_suffix(f".{format}")
            elif view:
                dot.view()
            return None
        except ImportError:
            raise ImportError(
                "graphviz package required for visualization. "
                "Install with: pip install graphviz"
            )
```

### Interactive HTML Export

```python
def to_html(self, output_path: Path) -> None:
    """Export DAG to interactive HTML with D3.js visualization.

    The HTML file includes:
    - Zoomable/pannable DAG view
    - Click on nodes to see details
    - Color-coded node types
    - Execution timing (if available)
    """
    template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <script src="https://unpkg.com/@hpcc-js/wasm/dist/index.min.js"></script>
        <script src="https://unpkg.com/d3-graphviz@4/build/d3-graphviz.min.js"></script>
    </head>
    <body>
        <div id="graph"></div>
        <script>
            d3.select("#graph").graphviz().renderDot(`{dot}`);
        </script>
    </body>
    </html>
    '''
    html = template.format(dot=self.to_graphviz())
    output_path.write_text(html)
```

---

## Debug Mode (Synchronous Execution)

For debugging complex pipelines, a synchronous execution mode disables all parallelism and lazy evaluation:

```python
class ExecutionEngine:
    def execute(
        self,
        dag: ExecutableDAG,
        context: DatasetContext,
        mode: str = "train",
        debug: bool = False,  # Enable debug mode
        **kwargs
    ) -> ExecutionResult:
        """Execute DAG with optional debug mode.

        When debug=True:
        1. Parallelism is disabled (sequential execution)
        2. Lazy views are materialized immediately
        3. Verbose logging at each step
        4. Data shapes printed after each node
        5. Stack traces are not wrapped (raw exceptions)
        6. Checkpoints saved after each node

        This makes debugging much easier at the cost of performance.
        """
        if debug:
            return self._execute_debug_mode(dag, context, mode, **kwargs)
        else:
            return self._execute_normal(dag, context, mode, **kwargs)

    def _execute_debug_mode(
        self,
        dag: ExecutableDAG,
        context: DatasetContext,
        mode: str,
        **kwargs
    ) -> ExecutionResult:
        """Sequential debug execution with verbose output."""
        print(f"[DEBUG] Starting pipeline execution in debug mode")
        print(f"[DEBUG] Initial data shape: X={context.x.shape}, y={context.y.shape}")
        print(f"[DEBUG] DAG has {len(dag.nodes)} nodes")
        print()

        for i, node_id in enumerate(dag.topological_order()):
            node = dag.nodes[node_id]
            print(f"[DEBUG] Step {i+1}/{len(dag.nodes)}: {node.name} ({node.node_type.value})")

            # Materialize lazy views before execution
            if hasattr(context, '_pending_views'):
                context._materialize_all()
                print(f"[DEBUG]   Materialized {len(context._pending_views)} pending views")

            try:
                # Execute node
                start_time = time.time()
                result = self._execute_node(dag, node_id, context, mode, **kwargs)
                elapsed = time.time() - start_time

                # Update context
                context = result.context

                # Print diagnostics
                print(f"[DEBUG]   Execution time: {elapsed:.3f}s")
                print(f"[DEBUG]   Output shape: X={context.x.shape}")
                if hasattr(result, 'predictions') and result.predictions:
                    print(f"[DEBUG]   Predictions collected: {len(result.predictions)}")

                # Save checkpoint
                checkpoint_path = kwargs.get('checkpoint_dir', Path('./debug_checkpoints'))
                checkpoint_path.mkdir(exist_ok=True)
                self._save_checkpoint(context, checkpoint_path / f"step_{i:03d}_{node_id}.pkl")
                print(f"[DEBUG]   Checkpoint saved")
                print()

            except Exception as e:
                print(f"[DEBUG] ❌ ERROR at step {i+1}: {node.name}")
                print(f"[DEBUG]   Exception: {type(e).__name__}: {e}")
                print(f"[DEBUG]   Data shape at failure: X={context.x.shape}")
                print(f"[DEBUG]   Node config: {node}")
                print()
                # Re-raise raw exception (no wrapping in debug mode)
                raise

        print(f"[DEBUG] ✓ Pipeline completed successfully")
        return ExecutionResult(...)
```

### Fast-Path for Linear Pipelines

Simple linear pipelines (no branching, no generators) can skip DAG overhead:

```python
class ExecutionEngine:
    def execute(self, dag: ExecutableDAG, context: DatasetContext, **kwargs):
        # Check if fast-path is applicable
        if self._is_linear_pipeline(dag) and not kwargs.get('force_dag'):
            return self._execute_fast_path(dag, context, **kwargs)
        else:
            return self._execute_dag(dag, context, **kwargs)

    def _is_linear_pipeline(self, dag: ExecutableDAG) -> bool:
        """Check if DAG is a simple linear chain.

        Returns True if:
        - No ForkNodes or JoinNodes
        - Each node has exactly one input and one output
        - No generators were expanded
        """
        for node in dag.nodes.values():
            if node.node_type in (NodeType.FORK, NodeType.JOIN):
                return False
            if len(dag.get_predecessors(node.node_id)) > 1:
                return False
            if len(dag.get_successors(node.node_id)) > 1:
                return False
        return True

    def _execute_fast_path(
        self,
        dag: ExecutableDAG,
        context: DatasetContext,
        **kwargs
    ) -> ExecutionResult:
        """Direct sequential execution without DAG machinery.

        For small datasets and simple pipelines, this avoids the overhead
        of lock management, barrier synchronization, and view resolution.
        """
        X, y = context.x, context.y

        for node in dag.topological_order():
            op = dag.nodes[node].operator

            if hasattr(op, 'fit_transform'):
                X = op.fit_transform(X, y)
            elif hasattr(op, 'fit') and hasattr(op, 'transform'):
                op.fit(X, y)
                X = op.transform(X)
            elif hasattr(op, 'fit') and hasattr(op, 'predict'):
                # Model node - handle CV inline
                ...

        return ExecutionResult(...)
```

---

## Fork/Join Synchronization

### Barrier Synchronization Protocol

ForkNode and JoinNode use explicit barrier synchronization to ensure all branches complete before merging:

```python
class BranchStatus(Enum):
    """Status of a branch execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BranchContext:
    """Context for a single branch within a fork."""
    branch_id: int
    branch_name: str
    status: BranchStatus = BranchStatus.PENDING
    context: Optional[DatasetContext] = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class ForkErrorPolicy(Enum):
    """Error handling policy for fork/join operations.

    Defines how the pipeline handles branch failures:

    FAIL_FAST (default):
        Stop immediately when any branch fails.
        Best for: Development, debugging, critical pipelines.

    CONTINUE_ALL:
        Continue executing remaining branches even if some fail.
        Best for: Exploration, comparing many variants.

    REQUIRE_MAJORITY:
        Continue if majority of branches succeed.
        Best for: Ensemble models, voting classifiers.

    REQUIRE_ONE:
        Continue if at least one branch succeeds.
        Best for: Fallback patterns, optional enhancements.

    BEST_EFFORT:
        Always continue, use whatever branches succeeded.
        Best for: Non-critical exploration, robustness testing.
    """
    FAIL_FAST = "fail_fast"
    CONTINUE_ALL = "continue_all"
    REQUIRE_MAJORITY = "require_majority"
    REQUIRE_ONE = "require_one"
    BEST_EFFORT = "best_effort"

    def should_continue(
        self,
        succeeded: int,
        failed: int,
        total: int
    ) -> bool:
        """Check if execution should continue given branch results."""
        if self == ForkErrorPolicy.FAIL_FAST:
            return failed == 0
        elif self == ForkErrorPolicy.CONTINUE_ALL:
            return True  # Always continue
        elif self == ForkErrorPolicy.REQUIRE_MAJORITY:
            return succeeded > total // 2
        elif self == ForkErrorPolicy.REQUIRE_ONE:
            return succeeded >= 1
        elif self == ForkErrorPolicy.BEST_EFFORT:
            return True
        return False

    def should_abort_early(self, failed: int, remaining: int, total: int) -> bool:
        """Check if should abort before all branches complete."""
        if self == ForkErrorPolicy.FAIL_FAST:
            return failed > 0
        elif self == ForkErrorPolicy.REQUIRE_MAJORITY:
            # Abort if majority is impossible
            succeeded_so_far = total - failed - remaining
            max_possible = succeeded_so_far + remaining
            return max_possible <= total // 2
        elif self == ForkErrorPolicy.REQUIRE_ONE:
            # Abort if all remaining branches are unlikely to help
            return False  # Never abort early
        return False
    """Barrier synchronization for fork/join.

    Ensures all branches complete before join proceeds.
    Supports timeout and flexible error handling via ForkErrorPolicy.
    """

    def __init__(
        self,
        branch_ids: List[int],
        timeout: Optional[float] = None,
        error_policy: ForkErrorPolicy = ForkErrorPolicy.FAIL_FAST
    ):
        self.branch_ids = branch_ids
        self.timeout = timeout
        self.error_policy = error_policy
        self._statuses: Dict[int, BranchContext] = {
            bid: BranchContext(branch_id=bid, branch_name=f"branch_{bid}")
            for bid in branch_ids
        }
        self._completed = threading.Event()
        self._lock = threading.Lock()

    def mark_started(self, branch_id: int) -> None:
        """Mark branch as started."""
        with self._lock:
            self._statuses[branch_id].status = BranchStatus.RUNNING
            self._statuses[branch_id].start_time = datetime.now()

    def mark_completed(
        self,
        branch_id: int,
        context: DatasetContext
    ) -> None:
        """Mark branch as successfully completed."""
        with self._lock:
            self._statuses[branch_id].status = BranchStatus.COMPLETED
            self._statuses[branch_id].context = context
            self._statuses[branch_id].end_time = datetime.now()
            self._check_all_done()

    def mark_failed(self, branch_id: int, error: Exception) -> None:
        """Mark branch as failed."""
        with self._lock:
            self._statuses[branch_id].status = BranchStatus.FAILED
            self._statuses[branch_id].error = error
            self._statuses[branch_id].end_time = datetime.now()
            if not self.continue_on_error:
                self._completed.set()  # Trigger early exit
            else:
                self._check_all_done()

    def wait_for_all(self) -> Dict[int, BranchContext]:
        """Wait for all branches to complete.

        Returns:
            Dict mapping branch_id to BranchContext

        Raises:
            TimeoutError: If timeout exceeded
            BranchExecutionError: If any branch failed and continue_on_error=False
        """
        completed = self._completed.wait(timeout=self.timeout)

        if not completed:
            # Cancel running branches
            for status in self._statuses.values():
                if status.status == BranchStatus.RUNNING:
                    status.status = BranchStatus.CANCELLED
            raise TimeoutError(f"Fork barrier timed out after {self.timeout}s")

        # Check for failures
        failed = {
            bid: ctx for bid, ctx in self._statuses.items()
            if ctx.status == BranchStatus.FAILED
        }
        if failed and not self.continue_on_error:
            raise BranchExecutionError(
                {bid: ctx.error for bid, ctx in failed.items()}
            )

        return self._statuses

    def _check_all_done(self) -> None:
        """Check if all branches are done (completed or failed)."""
        all_done = all(
            s.status in (BranchStatus.COMPLETED, BranchStatus.FAILED)
            for s in self._statuses.values()
        )
        if all_done:
            self._completed.set()
```

### JoinNode with Synchronization

```python
@dataclass
class JoinNode(DAGNode):
    """Merge point with barrier synchronization."""
    node_type: NodeType = NodeType.JOIN
    merge_strategy: str = "features"
    timeout: Optional[float] = None
    continue_on_error: bool = False

    def execute(
        self,
        barrier: ForkBarrier,
        prediction_store: Optional[PredictionStore] = None,
        mode: str = "train",
        artifacts: Optional[Dict[str, Any]] = None
    ) -> Tuple[DatasetContext, ViewSpec, Dict[str, Any]]:
        """Execute merge after all branches complete.

        Args:
            barrier: Fork barrier with branch contexts
            prediction_store: For prediction-based merging
            mode: "train" or "predict"
            artifacts: Pre-loaded artifacts

        Returns:
            Tuple of (merged context, merged view, artifacts)
        """
        # Wait for all branches
        branch_results = barrier.wait_for_all()

        # Filter to completed branches only
        completed = {
            bid: ctx for bid, ctx in branch_results.items()
            if ctx.status == BranchStatus.COMPLETED
        }

        if not completed:
            raise DAGExecutionError(
                "All branches failed, nothing to merge",
                node_id=self.node_id
            )

        contexts = [ctx.context for ctx in completed.values()]
        views = [ctx.context.current_view for ctx in completed.values()]

        if self.merge_strategy == "features":
            return self._merge_features(contexts, views, mode)
        elif self.merge_strategy == "predictions":
            return self._merge_predictions(contexts, views, prediction_store, mode)
        elif self.merge_strategy == "best":
            return self._select_best(contexts, views, prediction_store)
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")
```

### Execution Order Guarantees

When branches have different lengths, the execution engine maintains deterministic ordering:

```python
class ExecutionEngine:
    def _execute_fork(self, fork_node, context, ...):
        # Create barrier
        barrier = ForkBarrier(
            branch_ids=list(range(fork_node.branch_count)),
            timeout=self.fork_timeout,
            continue_on_error=self.continue_on_error
        )

        if self.parallel:
            # Parallel execution with barrier sync
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {}
                for branch_id, branch_def in enumerate(fork_node.branches):
                    future = executor.submit(
                        self._execute_branch_with_barrier,
                        branch_id, branch_def, context, barrier
                    )
                    futures[branch_id] = future

                # Wait for all (barrier handles sync)
                branch_results = barrier.wait_for_all()
        else:
            # Sequential execution (still uses barrier for consistency)
            for branch_id, branch_def in enumerate(fork_node.branches):
                self._execute_branch_with_barrier(
                    branch_id, branch_def, context, barrier
                )
            branch_results = barrier.wait_for_all()

        return branch_results
```

---

## Cross-Validation as Fork/Join

Cross-validation is modeled as an implicit fork/join pattern.

### Conceptual Model

```
                    ┌─────────────────┐
                    │   SplitterNode  │
                    │  assigns folds  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │ Fold 0  │          │ Fold 1  │          │ Fold 2  │
   │ Train   │          │ Train   │          │ Train   │
   └────┬────┘          └────┬────┘          └────┬────┘
        │                    │                    │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │ Fold 0  │          │ Fold 1  │          │ Fold 2  │
   │ Predict │          │ Predict │          │ Predict │
   └────┬────┘          └────┬────┘          └────┬────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  VirtualModel   │
                    │   aggregation   │
                    └─────────────────┘
```

### Implementation

```python
class FoldAwareMixin:
    """Mixin for nodes that need fold-aware execution."""

    def execute_per_fold(
        self,
        context: DatasetContext,
        input_view: ViewSpec,
        executor_fn: Callable
    ) -> List[Tuple[np.ndarray, np.ndarray, Any]]:
        """Execute function for each fold.

        Args:
            context: Dataset context with folds assigned
            input_view: Base view specification
            executor_fn: Function(X_train, y_train, X_val) -> (y_pred_val, model)

        Returns:
            List of (y_pred_val, sample_indices, model) per fold
        """
        results = []

        for fold_id, (train_idx, val_idx) in enumerate(context.folds):
            train_view = input_view.with_fold(fold_id, "train")
            val_view = input_view.with_fold(fold_id, "val")

            X_train, y_train = context.resolver.materialize(train_view)
            X_val, y_val = context.resolver.materialize(val_view)

            y_pred_val, model = executor_fn(X_train, y_train, X_val)

            results.append((y_pred_val, val_idx, model))

        return results

    def aggregate_fold_models(
        self,
        fold_models: List[Any],
        fold_scores: List[float],
        strategy: str = "weighted_mean"
    ) -> VirtualModel:
        """Create virtual model from fold models."""
        return VirtualModel(
            fold_models=fold_models,
            fold_weights=fold_scores,
            aggregation=strategy
        )
```

---

## Generator Expansion

Generator syntax (`_or_`, `_range_`, `_grid_`) expands at DAG build time into explicit branch structures. This section documents the expansion limits and safeguards.

### Safeguards

The critical review identified that unbounded generator expansion can cause memory exhaustion. The following safeguards are implemented:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_variants` | 1000 | Hard limit on total variants. Exceeding raises `GeneratorExplosionError` |
| `warn_threshold` | 100 | Threshold for `GeneratorExplosionWarning` |

### Preview Before Build

Always preview expansion size before building large pipelines:

```python
builder = DAGBuilder()

# Preview expansion
n_variants = builder.estimate_variants(pipeline)
print(f"Pipeline will expand to {n_variants} variants")

if n_variants > 100:
    print("Consider using subset sampling or reducing grid size")

# Build with custom limits if needed
dag = builder.build(
    pipeline,
    max_variants=500,      # Custom limit
    warn_threshold=50      # Earlier warning
)
```

### Generator Types

| Generator | Syntax | Expansion |
|-----------|--------|-----------|
| `_or_` | `{"_or_": [A, B, C]}` | 3 variants |
| `_range_` | `{"_range_": [1, 10, 2]}` | 5 variants (1, 3, 5, 7, 9) |
| `_grid_` | `{"_grid_": {"a": [1,2], "b": [3,4]}}` | 4 variants (cartesian product) |

### Strategies for Large Searches

If you need more than 1000 variants:

1. **Subset Sampling**: Use `count` parameter with `_or_`:
   ```python
   {"_or_": large_list, "count": 10}  # Random sample of 10
   ```

2. **Sequential Iteration**: Run multiple smaller searches:
   ```python
   for chunk in np.array_split(search_space, 10):
       run_pipeline(chunk)
   ```

3. **Optuna Integration**: Use Optuna for intelligent search:
   ```python
   {"tuner": OptunaTuner(n_trials=100)}  # Smart sampling
   ```

---

## Operator Serialization

Custom operators (transformers, models) must serialize for bundling and portability. This section defines the serialization protocol.

### Serialization Risks and Version Management

**Critical Warning**: Serializing arbitrary Python objects with `pickle` or `joblib` is inherently fragile across:
- Different Python versions
- Different library versions
- Different platforms

The serialization system must enforce strict version checking:

```python
@dataclass
class SerializationManifest:
    """Version manifest for serialized bundle.

    When loading a bundle, versions are checked and warnings/errors
    raised for incompatibilities.
    """
    python_version: str              # e.g., "3.11.4"
    nirs4all_version: str            # e.g., "2.0.0"
    sklearn_version: str             # e.g., "1.3.0"
    numpy_version: str               # e.g., "1.25.0"
    optional_deps: Dict[str, str]    # {"torch": "2.0.1", "tensorflow": "2.13.0"}
    serialization_format: str        # "sklearn", "onnx", "pickle"
    created_at: datetime
    platform: str                    # e.g., "linux-x86_64"

    def check_compatibility(self) -> List[str]:
        """Check if current environment is compatible.

        Returns:
            List of warning/error messages (empty if compatible)
        """
        import sys
        import sklearn
        import numpy as np
        import nirs4all

        issues = []

        # Python major.minor must match
        current_py = f"{sys.version_info.major}.{sys.version_info.minor}"
        saved_py = ".".join(self.python_version.split(".")[:2])
        if current_py != saved_py:
            issues.append(
                f"ERROR: Python version mismatch. "
                f"Bundle: {self.python_version}, Current: {sys.version}"
            )

        # sklearn major.minor should match
        current_sklearn = ".".join(sklearn.__version__.split(".")[:2])
        saved_sklearn = ".".join(self.sklearn_version.split(".")[:2])
        if current_sklearn != saved_sklearn:
            issues.append(
                f"WARNING: sklearn version mismatch. "
                f"Bundle: {self.sklearn_version}, Current: {sklearn.__version__}. "
                f"Predictions may differ."
            )

        # nirs4all major should match
        current_nirs = nirs4all.__version__.split(".")[0]
        saved_nirs = self.nirs4all_version.split(".")[0]
        if current_nirs != saved_nirs:
            issues.append(
                f"ERROR: nirs4all major version mismatch. "
                f"Bundle: {self.nirs4all_version}, Current: {nirs4all.__version__}"
            )

        return issues


def load_bundle_with_checks(path: Path) -> "ModelBundle":
    """Load bundle with strict version checking.

    Raises:
        VersionMismatchError: If critical versions don't match
        VersionWarning: If non-critical versions differ (logged but continues)
    """
    bundle = ModelBundle.load(path)
    manifest = bundle.manifest

    issues = manifest.check_compatibility()

    errors = [i for i in issues if i.startswith("ERROR")]
    warnings = [i for i in issues if i.startswith("WARNING")]

    for warning in warnings:
        logging.warning(warning)

    if errors:
        raise VersionMismatchError(
            f"Cannot load bundle due to version incompatibilities:\n" +
            "\n".join(errors) +
            "\n\nTo force load (may cause incorrect predictions), use:\n"
            "  load_bundle(path, ignore_version_errors=True)"
        )

    return bundle
```

**Recommended Serialization Strategy by Operator Type**:

| Operator Type | Recommended Format | Rationale |
|--------------|-------------------|-----------|
| sklearn estimators | sklearn JSON + fitted arrays | Portable, version-tolerant |
| PyTorch models | ONNX or TorchScript | Cross-platform, version-stable |
| TensorFlow models | SavedModel or ONNX | Standard format |
| Custom Python | Parameter dict + code hash | Requires matching code version |
| Preprocessing transforms | Parameter dict only | Stateless, easily recreated |

### Serialization Protocol

Operators are serialized using multiple strategies based on complexity:

```python
from typing import Protocol, Dict, Any, Type
from abc import abstractmethod


class OperatorSerializer(Protocol):
    """Protocol for operator serialization."""

    @abstractmethod
    def can_serialize(self, operator: Any) -> bool:
        """Check if this serializer handles the operator."""
        ...

    @abstractmethod
    def serialize(self, operator: Any) -> Dict[str, Any]:
        """Serialize operator to dict representation."""
        ...

    @abstractmethod
    def deserialize(self, data: Dict[str, Any]) -> Any:
        """Reconstruct operator from dict representation."""
        ...


class SklearnSerializer:
    """Serializer for sklearn-compatible operators."""

    def can_serialize(self, operator: Any) -> bool:
        return hasattr(operator, "get_params") and hasattr(operator, "set_params")

    def serialize(self, operator: Any) -> Dict[str, Any]:
        return {
            "class": f"{operator.__class__.__module__}.{operator.__class__.__name__}",
            "params": operator.get_params(deep=True),
            "fitted_attrs": self._get_fitted_attrs(operator),
            "sklearn_version": sklearn.__version__  # Track version
        }

    def _get_fitted_attrs(self, operator: Any) -> Dict[str, Any]:
        """Extract fitted attributes (those ending with underscore)."""
        fitted = {}
        for attr in dir(operator):
            if attr.endswith("_") and not attr.startswith("_"):
                try:
                    value = getattr(operator, attr)
                    if self._is_serializable(value):
                        fitted[attr] = value
                except Exception:
                    pass
        return fitted

    def _is_serializable(self, value: Any) -> bool:
        """Check if value can be serialized to JSON/numpy."""
        import numpy as np
        return isinstance(value, (int, float, str, bool, list, dict, np.ndarray))

    def deserialize(self, data: Dict[str, Any]) -> Any:
        from importlib import import_module

        module_path, class_name = data["class"].rsplit(".", 1)
        module = import_module(module_path)
        cls = getattr(module, class_name)

        operator = cls(**data["params"])

        # Restore fitted attributes
        for attr, value in data.get("fitted_attrs", {}).items():
            setattr(operator, attr, value)

        return operator


class JoblibSerializer:
    """Fallback serializer using joblib for complex objects.

    SECURITY WARNING:
        Joblib/pickle can execute arbitrary code during deserialization.
        This is a known security risk when loading bundles from untrusted sources.

        Mitigations:
        1. Always validate bundle source before loading
        2. Use `safe_load=True` to reject joblib-serialized operators
        3. Prefer SklearnSerializer or ONNXSerializer when possible
        4. Consider using cryptographic signatures for bundle verification

        For research sharing, prefer exporting with `format="sklearn"` which
        only stores parameters, not executable code.
    """

    def can_serialize(self, operator: Any) -> bool:
        return True  # Always try as fallback

    def serialize(self, operator: Any) -> Dict[str, Any]:
        import joblib
        import base64
        import io
        import warnings

        warnings.warn(
            f"Serializing {type(operator).__name__} with joblib. "
            f"This may pose security risks when sharing bundles. "
            f"Consider using sklearn-compatible operators instead.",
            SecurityWarning
        )

        buffer = io.BytesIO()
        joblib.dump(operator, buffer)

        return {
            "class": f"{operator.__class__.__module__}.{operator.__class__.__name__}",
            "joblib_bytes": base64.b64encode(buffer.getvalue()).decode(),
            "serialization_method": "joblib"  # Mark for safe_load checks
        }

    def deserialize(
        self,
        data: Dict[str, Any],
        safe_load: bool = False
    ) -> Any:
        """Deserialize operator from joblib bytes.

        Args:
            data: Serialized data dict
            safe_load: If True, refuse to load joblib-serialized operators

        Raises:
            SecurityError: If safe_load=True and operator uses joblib
        """
        if safe_load and data.get("serialization_method") == "joblib":
            raise SecurityError(
                f"Refusing to load {data['class']} serialized with joblib. "
                f"This operator contains executable code which could be malicious. "
                f"Use safe_load=False to override (at your own risk) or "
                f"request the bundle author to export with format='sklearn'."
            )

        import joblib
        import base64
        import io

        buffer = io.BytesIO(base64.b64decode(data["joblib_bytes"]))
        return joblib.load(buffer)


class SecurityWarning(UserWarning):
    """Warning for security-sensitive operations."""
    pass


class ONNXSerializer:
    """Serializer using ONNX format for cross-platform compatibility.

    ONNX (Open Neural Network Exchange) provides:
    - Cross-platform portability (Windows, Linux, macOS)
    - Version stability (models work across library versions)
    - Hardware acceleration (ONNX Runtime supports GPU, CPU, TPU)
    - Language interoperability (Python, C++, C#, Java)

    Supported Operators:
    - sklearn estimators (via skl2onnx)
    - PyTorch models (via torch.onnx.export)
    - TensorFlow models (via tf2onnx)
    - Keras models (via keras2onnx or tf2onnx)

    Limitations:
    - Some sklearn estimators not fully supported (check skl2onnx docs)
    - Custom Python code cannot be serialized
    - Dynamic shapes may require explicit input shapes

    Usage:
        serializer = ONNXSerializer()

        if serializer.can_serialize(model):
            data = serializer.serialize(model, input_shape=(None, 100))
            # ... save data to bundle ...

            # Later:
            loaded_model = serializer.deserialize(data)
            predictions = loaded_model.run(X)
    """

    def __init__(self, opset_version: int = 15):
        """Initialize ONNX serializer.

        Args:
            opset_version: ONNX opset version (default: 15)
        """
        self.opset_version = opset_version

    def can_serialize(self, operator: Any) -> bool:
        """Check if operator can be serialized to ONNX."""
        # sklearn
        if hasattr(operator, "get_params") and hasattr(operator, "fit"):
            try:
                import skl2onnx
                return True
            except ImportError:
                pass

        # PyTorch
        if hasattr(operator, "forward") and hasattr(operator, "parameters"):
            try:
                import torch
                return isinstance(operator, torch.nn.Module)
            except ImportError:
                pass

        # TensorFlow/Keras
        if hasattr(operator, "call") and hasattr(operator, "build"):
            try:
                import tensorflow as tf
                return isinstance(operator, (tf.keras.Model, tf.Module))
            except ImportError:
                pass

        return False

    def serialize(
        self,
        operator: Any,
        input_shape: Optional[Tuple[Optional[int], ...]] = None,
        input_dtype: str = "float32"
    ) -> Dict[str, Any]:
        """Serialize operator to ONNX format.

        Args:
            operator: Model to serialize
            input_shape: Input tensor shape (use None for dynamic dims)
            input_dtype: Input data type

        Returns:
            Dict with ONNX bytes and metadata
        """
        import base64

        # Determine framework and serialize
        if hasattr(operator, "get_params"):
            onnx_bytes = self._serialize_sklearn(operator, input_shape)
            framework = "sklearn"
        elif hasattr(operator, "forward"):
            onnx_bytes = self._serialize_pytorch(operator, input_shape, input_dtype)
            framework = "pytorch"
        elif hasattr(operator, "call"):
            onnx_bytes = self._serialize_tensorflow(operator, input_shape)
            framework = "tensorflow"
        else:
            raise ValueError(f"Cannot serialize {type(operator)} to ONNX")

        return {
            "class": f"{operator.__class__.__module__}.{operator.__class__.__name__}",
            "onnx_bytes": base64.b64encode(onnx_bytes).decode(),
            "serialization_method": "onnx",
            "opset_version": self.opset_version,
            "framework": framework,
            "input_shape": list(input_shape) if input_shape else None,
            "input_dtype": input_dtype
        }

    def _serialize_sklearn(
        self,
        operator: Any,
        input_shape: Optional[Tuple[Optional[int], ...]]
    ) -> bytes:
        """Serialize sklearn model to ONNX."""
        import skl2onnx
        from skl2onnx.common.data_types import FloatTensorType
        import numpy as np

        # Infer input shape from fitted attributes if not provided
        if input_shape is None:
            if hasattr(operator, "n_features_in_"):
                input_shape = (None, operator.n_features_in_)
            else:
                raise ValueError("input_shape required for sklearn model without n_features_in_")

        initial_type = [("input", FloatTensorType(list(input_shape)))]

        onnx_model = skl2onnx.convert_sklearn(
            operator,
            initial_types=initial_type,
            target_opset=self.opset_version
        )

        return onnx_model.SerializeToString()

    def _serialize_pytorch(
        self,
        operator: Any,
        input_shape: Tuple[Optional[int], ...],
        input_dtype: str
    ) -> bytes:
        """Serialize PyTorch model to ONNX."""
        import torch
        import io

        if input_shape is None:
            raise ValueError("input_shape required for PyTorch model")

        # Create dummy input
        shape = tuple(s if s is not None else 1 for s in input_shape)
        dtype = getattr(torch, input_dtype)
        dummy_input = torch.zeros(shape, dtype=dtype)

        buffer = io.BytesIO()
        torch.onnx.export(
            operator,
            dummy_input,
            buffer,
            opset_version=self.opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {i: f"dim_{i}" for i, s in enumerate(input_shape) if s is None},
                "output": {0: "batch"}
            }
        )

        return buffer.getvalue()

    def _serialize_tensorflow(
        self,
        operator: Any,
        input_shape: Optional[Tuple[Optional[int], ...]]
    ) -> bytes:
        """Serialize TensorFlow model to ONNX."""
        import tf2onnx
        import tensorflow as tf

        if input_shape is None:
            # Try to infer from model
            if hasattr(operator, "input_shape"):
                input_shape = operator.input_shape
            else:
                raise ValueError("input_shape required for TensorFlow model")

        # Create input spec
        input_spec = [tf.TensorSpec(input_shape, tf.float32, name="input")]

        onnx_model, _ = tf2onnx.convert.from_keras(
            operator,
            input_signature=input_spec,
            opset=self.opset_version
        )

        return onnx_model.SerializeToString()

    def deserialize(self, data: Dict[str, Any]) -> "ONNXInferenceSession":
        """Deserialize ONNX model to inference session.

        Returns:
            ONNXInferenceSession wrapper for predictions
        """
        import base64
        import onnxruntime as ort

        onnx_bytes = base64.b64decode(data["onnx_bytes"])
        session = ort.InferenceSession(onnx_bytes)

        return ONNXInferenceSession(
            session=session,
            input_name=session.get_inputs()[0].name,
            output_name=session.get_outputs()[0].name,
            original_class=data["class"]
        )


@dataclass
class ONNXInferenceSession:
    """Wrapper for ONNX Runtime inference session.

    Provides sklearn-compatible predict interface.
    """
    session: Any  # onnxruntime.InferenceSession
    input_name: str
    output_name: str
    original_class: str

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference on input data."""
        import numpy as np

        # Ensure float32
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: X}
        )

        return outputs[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Run inference returning probabilities (if available)."""
        import numpy as np

        if X.dtype != np.float32:
            X = X.astype(np.float32)

        # Try to get probability output
        output_names = [o.name for o in self.session.get_outputs()]

        if len(output_names) > 1:
            # Second output is usually probabilities
            outputs = self.session.run(output_names[:2], {self.input_name: X})
            return outputs[1]

        return self.predict(X)


class SecurityError(Exception):
    """Raised when a security check fails."""
    pass
```

### Serialization Registry

```python
class SerializationRegistry:
    """Registry of serializers with priority ordering."""

    def __init__(self):
        self._serializers: List[OperatorSerializer] = [
            SklearnSerializer(),      # Try sklearn first
            # TensorFlowSerializer(),  # Framework-specific
            # PyTorchSerializer(),
            JoblibSerializer(),       # Fallback
        ]

    def serialize(self, operator: Any) -> Dict[str, Any]:
        """Serialize operator using first matching serializer."""
        for serializer in self._serializers:
            if serializer.can_serialize(operator):
                data = serializer.serialize(operator)
                data["_serializer"] = serializer.__class__.__name__
                return data
        raise ValueError(f"No serializer found for {type(operator)}")

    def deserialize(self, data: Dict[str, Any]) -> Any:
        """Deserialize using recorded serializer."""
        serializer_name = data.get("_serializer", "JoblibSerializer")
        for serializer in self._serializers:
            if serializer.__class__.__name__ == serializer_name:
                return serializer.deserialize(data)
        raise ValueError(f"Unknown serializer: {serializer_name}")
```

### UX: Flexible Operator Specification

Preserving v1's flexible syntax, operators can be specified multiple ways:

```python
# All equivalent - normalized during parsing
MinMaxScaler                                # Class reference
MinMaxScaler()                              # Instance
{"class": "sklearn.preprocessing.MinMaxScaler"}  # Dict with class path
"sklearn.preprocessing.MinMaxScaler"        # String path
{"preprocessing": MinMaxScaler()}           # Keyword wrapper

# YAML equivalent
preprocessing:
  class: sklearn.preprocessing.MinMaxScaler
  params:
    feature_range: [0, 1]
```

The `DAGBuilder._parse_step()` method normalizes all forms to a canonical `ParsedStep`:

```python
@dataclass
class ParsedStep:
    """Normalized step representation."""
    operator_class: str          # Fully qualified class name
    operator_params: Dict        # Constructor parameters
    keyword: Optional[str]       # Step keyword (model, preprocessing, etc.)
    operator_instance: Any       # Resolved instance (lazy)
```

---

## Artifact Management

### ArtifactManager

```python
class ArtifactManager:
    """Manages artifacts with lineage tracking.

    Artifacts are stored with:
    - Unique ID based on node + lineage
    - Serialized binary (joblib)
    - Metadata JSON
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._manifest: Dict[str, ArtifactRecord] = {}

    def save(
        self,
        node_id: str,
        name: str,
        obj: Any,
        lineage_hash: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Save artifact and return artifact ID."""
        artifact_id = f"{node_id}_{name}_{lineage_hash[:8]}"

        # Save binary
        artifact_path = self.base_dir / f"{artifact_id}.joblib"
        joblib.dump(obj, artifact_path)

        # Save metadata
        meta_path = self.base_dir / f"{artifact_id}.json"
        record = ArtifactRecord(
            artifact_id=artifact_id,
            node_id=node_id,
            name=name,
            lineage_hash=lineage_hash,
            path=str(artifact_path),
            metadata=metadata or {}
        )
        with open(meta_path, 'w') as f:
            json.dump(record.to_dict(), f)

        self._manifest[artifact_id] = record
        return artifact_id

    def load(self, artifact_id: str) -> Any:
        """Load artifact by ID."""
        artifact_path = self.base_dir / f"{artifact_id}.joblib"
        return joblib.load(artifact_path)

    def get_by_lineage(self, lineage_hash: str) -> Optional[str]:
        """Get artifact ID by lineage hash (for cache lookup)."""
        for record in self._manifest.values():
            if record.lineage_hash == lineage_hash:
                return record.artifact_id
        return None


@dataclass
class ArtifactRecord:
    """Metadata for a stored artifact."""
    artifact_id: str
    node_id: str
    name: str
    lineage_hash: str
    path: str
    metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "artifact_id": self.artifact_id,
            "node_id": self.node_id,
            "name": self.name,
            "lineage_hash": self.lineage_hash,
            "path": self.path,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
```

---

## Prediction Store Integration

The DAG engine integrates with a Polars-backed prediction store.

### PredictionStore

```python
class PredictionStore:
    """Polars-backed prediction storage.

    Collects predictions during DAG execution and provides
    ranking, filtering, and aggregation.
    """

    def __init__(self):
        self._df: pl.DataFrame = pl.DataFrame(schema=PREDICTION_SCHEMA)
        self._arrays: Dict[str, np.ndarray] = {}

    def add_prediction(
        self,
        partition: str,
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        fold_id: Optional[int] = None,
        sample_indices: Optional[List[int]] = None,
        model_name: str = "",
        branch_id: Optional[int] = None,
        branch_path: Optional[List[int]] = None,
        val_score: Optional[float] = None,
        test_score: Optional[float] = None,
        **metadata
    ) -> str:
        """Add prediction to store."""
        pred_id = self._generate_id()

        # Store arrays separately
        self._arrays[f"{pred_id}_pred"] = y_pred
        if y_true is not None:
            self._arrays[f"{pred_id}_true"] = y_true

        # Add row to DataFrame
        row = {
            "id": pred_id,
            "partition": partition,
            "fold_id": fold_id,
            "sample_indices": sample_indices or [],
            "model_name": model_name,
            "branch_id": branch_id,
            "branch_path": branch_path or [],
            "val_score": val_score,
            "test_score": test_score,
            **metadata
        }

        self._df = pl.concat([self._df, pl.DataFrame([row])])
        return pred_id

    def get_best(
        self,
        metric: str = "val_score",
        ascending: bool = True,
        **filters
    ) -> Optional[Dict]:
        """Get best prediction by metric."""
        df = self._apply_filters(self._df, filters)

        if len(df) == 0:
            return None

        sorted_df = df.sort(metric, descending=not ascending)
        best_row = sorted_df.head(1).to_dicts()[0]

        return self._hydrate(best_row)

    def get_oof_predictions(
        self,
        model_name: Optional[str] = None,
        branch_name: Optional[str] = None
    ) -> List[Dict]:
        """Get out-of-fold predictions for reconstruction."""
        filters = {"partition": "val"}
        if model_name:
            filters["model_name"] = model_name
        if branch_name:
            filters["branch_name"] = branch_name

        df = self._apply_filters(self._df, filters)
        return [self._hydrate(row) for row in df.to_dicts()]

    def _hydrate(self, row: Dict) -> Dict:
        """Add array data to row dict."""
        pred_id = row["id"]
        row["y_pred"] = self._arrays.get(f"{pred_id}_pred")
        row["y_true"] = self._arrays.get(f"{pred_id}_true")
        return row
```

---

## Next Document

**Document 4: API Layer Design** covers:
- Static functions (`run`, `predict`, `explain`)
- sklearn-compatible estimators
- Result objects and convenience methods
- CLI and configuration file support
