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
    """
    node_id: str
    node_type: NodeType
    operator: Any
    operator_class: str
    operator_params: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_dynamic: bool = False


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
        """Merge by using OOF predictions as meta-features."""
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
        cache_dir: Optional[Path] = None
    ):
        self.parallel = parallel
        self.n_workers = n_workers
        self.parallel_backend = parallel_backend
        self.cache_dir = cache_dir
        self._executor = None

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


class ForkBarrier:
    """Barrier synchronization for fork/join.

    Ensures all branches complete before join proceeds.
    Supports timeout and partial failure handling.
    """

    def __init__(
        self,
        branch_ids: List[int],
        timeout: Optional[float] = None,
        continue_on_error: bool = False
    ):
        self.branch_ids = branch_ids
        self.timeout = timeout
        self.continue_on_error = continue_on_error
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
    """Fallback serializer using joblib for complex objects."""

    def can_serialize(self, operator: Any) -> bool:
        return True  # Always try as fallback

    def serialize(self, operator: Any) -> Dict[str, Any]:
        import joblib
        import base64
        import io

        buffer = io.BytesIO()
        joblib.dump(operator, buffer)

        return {
            "class": f"{operator.__class__.__module__}.{operator.__class__.__name__}",
            "joblib_bytes": base64.b64encode(buffer.getvalue()).decode()
        }

    def deserialize(self, data: Dict[str, Any]) -> Any:
        import joblib
        import base64
        import io

        buffer = io.BytesIO(base64.b64decode(data["joblib_bytes"]))
        return joblib.load(buffer)
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
