"""
Execution Trace V3 - Records the exact path through pipeline that produced a prediction.

This module provides the core data structures for recording execution traces,
which enable deterministic prediction replay and pipeline extraction.

V3 improvements:
- OperatorChain tracking for complete execution path
- Per-branch and per-source artifact indexing
- Support for nested branches and multi-source pipelines
- Chain-based artifact lookup for deterministic replay

Key Classes:
    - StepArtifacts: Artifacts produced by a single step with V3 indexes
    - ExecutionStep: Record of a single step's execution with chain tracking
    - ExecutionTrace: Complete trace of a pipeline execution path

Architecture:
    During training, each step execution is recorded in the trace:
    1. Step starts -> record step_index, operator info, input chain
    2. Step completes -> record artifacts and output chains
    3. Model produces prediction -> trace_id is attached to prediction

    During prediction, the trace is used to:
    1. Identify the minimal set of steps needed
    2. Load the correct artifacts for each step via chain lookup
    3. Execute only required steps via existing controllers
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from nirs4all.pipeline.storage.artifacts.operator_chain import OperatorChain


class StepExecutionMode(str, Enum):
    """Mode of step execution.

    Attributes:
        TRAIN: Step fitted on data (creates new artifacts)
        PREDICT: Step uses pre-fitted artifacts
        SKIP: Step was skipped (no-op)
    """

    TRAIN = "train"
    PREDICT = "predict"
    SKIP = "skip"

    def __str__(self) -> str:
        return self.value


@dataclass
class StepArtifacts:
    """Artifacts produced by a single step (V3).

    Records all artifacts created during step execution, with V3 indexes
    for efficient lookup by chain path, branch, source, and fold.

    Attributes:
        artifact_ids: List of artifact IDs produced by this step
        primary_artifact_id: Main artifact (e.g., model) if applicable
        fold_artifact_ids: Per-fold artifacts for CV models

        # V3 indexes
        primary_artifacts: Map of chain_path to artifact_id for shared artifacts
        by_branch: Artifacts indexed by branch path tuple
        by_source: Artifacts indexed by source index
        by_chain: Artifacts indexed by chain path

        metadata: Additional artifact metadata (types, paths, etc.)
    """

    artifact_ids: List[str] = field(default_factory=list)
    primary_artifact_id: Optional[str] = None
    fold_artifact_ids: Dict[int, str] = field(default_factory=dict)

    # V3 indexes
    primary_artifacts: Dict[str, str] = field(default_factory=dict)
    by_branch: Dict[Tuple[int, ...], List[str]] = field(default_factory=dict)
    by_source: Dict[int, List[str]] = field(default_factory=dict)
    by_chain: Dict[str, str] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization.

        Returns:
            Dictionary suitable for manifest storage
        """
        # Convert tuple keys to string for YAML compatibility
        by_branch_serialized = {
            ".".join(str(b) for b in k): v
            for k, v in self.by_branch.items()
        }

        return {
            "artifact_ids": self.artifact_ids,
            "primary_artifact_id": self.primary_artifact_id,
            "fold_artifact_ids": self.fold_artifact_ids,
            "primary_artifacts": self.primary_artifacts,
            "by_branch": by_branch_serialized,
            "by_source": self.by_source,
            "by_chain": self.by_chain,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepArtifacts":
        """Create StepArtifacts from dictionary.

        Args:
            data: Dictionary from manifest

        Returns:
            StepArtifacts instance
        """
        # Handle fold_artifact_ids with potential string keys from YAML
        fold_artifacts = data.get("fold_artifact_ids", {})
        if fold_artifacts:
            fold_artifacts = {int(k): v for k, v in fold_artifacts.items()}

        # Handle by_branch with string keys from YAML
        by_branch_raw = data.get("by_branch", {})
        by_branch: Dict[Tuple[int, ...], List[str]] = {}
        for k, v in by_branch_raw.items():
            if isinstance(k, str):
                branch_tuple = tuple(int(b) for b in k.split(".")) if k else ()
            else:
                branch_tuple = tuple(k) if k else ()
            by_branch[branch_tuple] = v

        # Handle by_source with potential string keys
        by_source_raw = data.get("by_source", {})
        by_source = {int(k): v for k, v in by_source_raw.items()} if by_source_raw else {}

        return cls(
            artifact_ids=data.get("artifact_ids", []),
            primary_artifact_id=data.get("primary_artifact_id"),
            fold_artifact_ids=fold_artifacts,
            primary_artifacts=data.get("primary_artifacts", {}),
            by_branch=by_branch,
            by_source=by_source,
            by_chain=data.get("by_chain", {}),
            metadata=data.get("metadata", {}),
        )

    def add_artifact(
        self,
        artifact_id: str,
        is_primary: bool = False,
        chain_path: Optional[str] = None,
        branch_path: Optional[List[int]] = None,
        source_index: Optional[int] = None,
    ) -> None:
        """Add an artifact ID to this step's artifacts (V3).

        Args:
            artifact_id: The artifact ID to add
            is_primary: Whether this is the primary artifact
            chain_path: V3 operator chain path
            branch_path: Branch path for indexing
            source_index: Source index for multi-source indexing
        """
        if artifact_id not in self.artifact_ids:
            self.artifact_ids.append(artifact_id)

        if is_primary:
            self.primary_artifact_id = artifact_id

        # V3 indexing
        if chain_path:
            self.by_chain[chain_path] = artifact_id
            if is_primary:
                self.primary_artifacts[chain_path] = artifact_id

        if branch_path is not None:
            branch_key = tuple(branch_path)
            if branch_key not in self.by_branch:
                self.by_branch[branch_key] = []
            if artifact_id not in self.by_branch[branch_key]:
                self.by_branch[branch_key].append(artifact_id)

        if source_index is not None:
            if source_index not in self.by_source:
                self.by_source[source_index] = []
            if artifact_id not in self.by_source[source_index]:
                self.by_source[source_index].append(artifact_id)

    def add_fold_artifact(
        self,
        fold_id: int,
        artifact_id: str,
        chain_path: Optional[str] = None,
        branch_path: Optional[List[int]] = None,
    ) -> None:
        """Add a fold-specific artifact.

        Args:
            fold_id: CV fold index
            artifact_id: Artifact ID for this fold
            chain_path: V3 operator chain path
            branch_path: Branch path for indexing
        """
        self.fold_artifact_ids[fold_id] = artifact_id
        self.add_artifact(
            artifact_id,
            is_primary=False,
            chain_path=chain_path,
            branch_path=branch_path,
        )

    def get_artifacts_for_branch(
        self,
        branch_path: List[int]
    ) -> List[str]:
        """Get artifact IDs matching a branch path.

        Includes artifacts from:
        - Exact branch match
        - Empty branch (shared/pre-branch)
        - Parent branches (for nested branches)

        Args:
            branch_path: Target branch path

        Returns:
            List of matching artifact IDs
        """
        results: List[str] = []
        target_tuple = tuple(branch_path)

        for branch_key, ids in self.by_branch.items():
            # Include if exact match, empty (shared), or prefix
            if not branch_key:  # Empty = shared
                results.extend(ids)
            elif branch_key == target_tuple:
                results.extend(ids)
            elif len(branch_key) < len(target_tuple) and target_tuple[:len(branch_key)] == branch_key:
                # Parent branch
                results.extend(ids)

        # Deduplicate while preserving order
        seen = set()
        return [x for x in results if not (x in seen or seen.add(x))]

    def get_artifacts_for_source(self, source_index: int) -> List[str]:
        """Get artifact IDs for a specific source.

        Args:
            source_index: Source index to filter

        Returns:
            List of artifact IDs for that source
        """
        return self.by_source.get(source_index, []).copy()

    def get_artifact_by_chain(self, chain_path: str) -> Optional[str]:
        """Get artifact ID by exact chain path match.

        Args:
            chain_path: Operator chain path

        Returns:
            Artifact ID or None if not found
        """
        return self.by_chain.get(chain_path)

    def merge(self, other: "StepArtifacts") -> None:
        """Merge another StepArtifacts into this one.

        Used when multiple substeps share the same step_index and their
        artifacts need to be combined in the artifact_map.

        Args:
            other: StepArtifacts to merge into this one
        """
        # Merge artifact_ids
        for artifact_id in other.artifact_ids:
            if artifact_id not in self.artifact_ids:
                self.artifact_ids.append(artifact_id)

        # Primary artifact: keep existing if set, otherwise use other
        if not self.primary_artifact_id and other.primary_artifact_id:
            self.primary_artifact_id = other.primary_artifact_id

        # Merge fold_artifact_ids (other takes precedence for conflicts)
        for fold_id, artifact_id in other.fold_artifact_ids.items():
            if fold_id not in self.fold_artifact_ids:
                self.fold_artifact_ids[fold_id] = artifact_id

        # Merge primary_artifacts
        for chain_path, artifact_id in other.primary_artifacts.items():
            if chain_path not in self.primary_artifacts:
                self.primary_artifacts[chain_path] = artifact_id

        # Merge by_branch
        for branch_key, ids in other.by_branch.items():
            if branch_key not in self.by_branch:
                self.by_branch[branch_key] = []
            for artifact_id in ids:
                if artifact_id not in self.by_branch[branch_key]:
                    self.by_branch[branch_key].append(artifact_id)

        # Merge by_source
        for source_idx, ids in other.by_source.items():
            if source_idx not in self.by_source:
                self.by_source[source_idx] = []
            for artifact_id in ids:
                if artifact_id not in self.by_source[source_idx]:
                    self.by_source[source_idx].append(artifact_id)

        # Merge by_chain
        for chain_path, artifact_id in other.by_chain.items():
            if chain_path not in self.by_chain:
                self.by_chain[chain_path] = artifact_id

        # Merge metadata
        self.metadata.update(other.metadata)


@dataclass
class ExecutionStep:
    """Record of a single step's execution in the trace (V3).

    Captures all information needed to replay this step during prediction,
    including operator configuration, execution mode, and produced artifacts.

    V3 additions:
    - input_chain: Operator chain up to this step's input
    - output_chains: Chains produced by this step (for branching)
    - source_count: Number of X sources at this step
    - produces_branches: Whether this is a branch operator

    Attributes:
        step_index: 1-based step number in the pipeline
        operator_type: Type of operation (e.g., "transform", "model", "splitter")
        operator_class: Class name of the operator (e.g., "PLSRegression", "SNV")
        operator_config: Serialized operator configuration
        execution_mode: How the step was executed (train/predict/skip)
        artifacts: Artifacts produced by this step
        branch_path: Branch indices if in a branch context
        branch_name: Human-readable branch name
        duration_ms: Execution duration in milliseconds
        metadata: Additional step-specific metadata

        # V3 chain tracking
        input_chain_path: Serialized operator chain up to this step's input
        output_chain_paths: List of chains produced by this step
        source_count: Number of X sources processed
        produces_branches: True if this is a branch operator
        substep_index: Index within substep (for [model1, model2])
    """

    step_index: int
    operator_type: str = ""
    operator_class: str = ""
    operator_config: Dict[str, Any] = field(default_factory=dict)
    execution_mode: StepExecutionMode = StepExecutionMode.TRAIN
    artifacts: StepArtifacts = field(default_factory=StepArtifacts)
    branch_path: List[int] = field(default_factory=list)
    branch_name: str = ""
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # V3 chain tracking
    input_chain_path: str = ""
    output_chain_paths: List[str] = field(default_factory=list)
    source_count: int = 1
    produces_branches: bool = False
    substep_index: Optional[int] = None

    # V4 shape tracking
    # Input/output shapes are 2D layout (samples, features)
    input_shape: Optional[Tuple[int, int]] = None
    output_shape: Optional[Tuple[int, int]] = None
    # Features shape is 3D per-source: List of (samples, processings, features) per source
    input_features_shape: Optional[List[Tuple[int, int, int]]] = None
    output_features_shape: Optional[List[Tuple[int, int, int]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization.

        Returns:
            Dictionary suitable for manifest storage
        """
        return {
            "step_index": self.step_index,
            "operator_type": self.operator_type,
            "operator_class": self.operator_class,
            "operator_config": self.operator_config,
            "execution_mode": str(self.execution_mode),
            "artifacts": self.artifacts.to_dict(),
            "branch_path": self.branch_path,
            "branch_name": self.branch_name,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "input_chain_path": self.input_chain_path,
            "output_chain_paths": self.output_chain_paths,
            "source_count": self.source_count,
            "produces_branches": self.produces_branches,
            "substep_index": self.substep_index,
            "input_shape": list(self.input_shape) if self.input_shape else None,
            "output_shape": list(self.output_shape) if self.output_shape else None,
            "input_features_shape": [list(s) for s in self.input_features_shape] if self.input_features_shape else None,
            "output_features_shape": [list(s) for s in self.output_features_shape] if self.output_features_shape else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionStep":
        """Create ExecutionStep from dictionary.

        Args:
            data: Dictionary from manifest

        Returns:
            ExecutionStep instance
        """
        # Handle execution_mode enum
        mode_value = data.get("execution_mode", "train")
        if isinstance(mode_value, str):
            execution_mode = StepExecutionMode(mode_value)
        else:
            execution_mode = mode_value

        # Handle artifacts
        artifacts_data = data.get("artifacts", {})
        if isinstance(artifacts_data, dict):
            artifacts = StepArtifacts.from_dict(artifacts_data)
        else:
            artifacts = StepArtifacts()

        # Parse shape fields
        input_shape = data.get("input_shape")
        output_shape = data.get("output_shape")
        input_features_shape = data.get("input_features_shape")
        output_features_shape = data.get("output_features_shape")

        return cls(
            step_index=data.get("step_index", 0),
            operator_type=data.get("operator_type", ""),
            operator_class=data.get("operator_class", ""),
            operator_config=data.get("operator_config", {}),
            execution_mode=execution_mode,
            artifacts=artifacts,
            branch_path=data.get("branch_path", []),
            branch_name=data.get("branch_name", ""),
            duration_ms=data.get("duration_ms", 0.0),
            metadata=data.get("metadata", {}),
            input_chain_path=data.get("input_chain_path", ""),
            output_chain_paths=data.get("output_chain_paths", []),
            source_count=data.get("source_count", 1),
            produces_branches=data.get("produces_branches", False),
            substep_index=data.get("substep_index"),
            input_shape=tuple(input_shape) if input_shape else None,
            output_shape=tuple(output_shape) if output_shape else None,
            input_features_shape=[tuple(s) for s in input_features_shape] if input_features_shape else None,
            output_features_shape=[tuple(s) for s in output_features_shape] if output_features_shape else None,
        )

    def has_artifacts(self) -> bool:
        """Check if this step produced any artifacts.

        Returns:
            True if the step has at least one artifact
        """
        return len(self.artifacts.artifact_ids) > 0

    def add_output_chain(self, chain_path: str) -> None:
        """Add an output chain path to this step.

        Args:
            chain_path: Operator chain path to add
        """
        if chain_path and chain_path not in self.output_chain_paths:
            self.output_chain_paths.append(chain_path)


@dataclass
class ExecutionTrace:
    """Complete trace of a pipeline execution path.

    Records the exact sequence of steps and artifacts that produced a prediction,
    enabling deterministic replay for prediction, transfer, and export.

    The trace is controller-agnostic: it records what happened without encoding
    specific controller logic, so any controller (existing or custom) can be
    replayed using the same infrastructure.

    Attributes:
        trace_id: Unique identifier for this trace
        pipeline_uid: Parent pipeline UID
        created_at: ISO timestamp of trace creation
        steps: Ordered list of execution steps
        model_step_index: Index of the model step that produced predictions
        fold_weights: Per-fold weights for CV ensemble (None for single model)
        preprocessing_chain: Summary of preprocessing steps for quick reference
        metadata: Additional trace metadata (e.g., dataset info, run parameters)
    """

    trace_id: str = field(default_factory=lambda: str(uuid4())[:12])
    pipeline_uid: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    steps: List[ExecutionStep] = field(default_factory=list)
    model_step_index: Optional[int] = None
    fold_weights: Optional[Dict[int, float]] = None
    preprocessing_chain: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization.

        Returns:
            Dictionary suitable for manifest storage
        """
        return {
            "trace_id": self.trace_id,
            "pipeline_uid": self.pipeline_uid,
            "created_at": self.created_at,
            "steps": [step.to_dict() for step in self.steps],
            "model_step_index": self.model_step_index,
            "fold_weights": self.fold_weights,
            "preprocessing_chain": self.preprocessing_chain,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionTrace":
        """Create ExecutionTrace from dictionary.

        Args:
            data: Dictionary from manifest

        Returns:
            ExecutionTrace instance
        """
        steps = [
            ExecutionStep.from_dict(step_data)
            for step_data in data.get("steps", [])
        ]

        # Handle fold_weights with potential string keys from YAML
        fold_weights = data.get("fold_weights")
        if fold_weights is not None:
            fold_weights = {int(k): float(v) for k, v in fold_weights.items()}

        return cls(
            trace_id=data.get("trace_id", str(uuid4())[:12]),
            pipeline_uid=data.get("pipeline_uid", ""),
            created_at=data.get("created_at", ""),
            steps=steps,
            model_step_index=data.get("model_step_index"),
            fold_weights=fold_weights,
            preprocessing_chain=data.get("preprocessing_chain", ""),
            metadata=data.get("metadata", {}),
        )

    def add_step(self, step: ExecutionStep) -> None:
        """Add a step to the trace.

        Args:
            step: ExecutionStep to add
        """
        self.steps.append(step)

    def get_step(self, step_index: int) -> Optional[ExecutionStep]:
        """Get a step by its index.

        Args:
            step_index: 1-based step index to find

        Returns:
            ExecutionStep or None if not found
        """
        for step in self.steps:
            if step.step_index == step_index:
                return step
        return None

    def get_steps_before(self, step_index: int) -> List[ExecutionStep]:
        """Get all steps before a given step index.

        Args:
            step_index: 1-based step index (exclusive)

        Returns:
            List of steps with step_index < given index
        """
        return [s for s in self.steps if s.step_index < step_index]

    def get_steps_up_to_model(self) -> List[ExecutionStep]:
        """Get all steps up to and including the model step.

        Returns:
            List of steps needed to reproduce the prediction
        """
        if self.model_step_index is None:
            return self.steps.copy()
        return [s for s in self.steps if s.step_index <= self.model_step_index]

    def get_artifact_ids(self) -> List[str]:
        """Get all artifact IDs in this trace.

        Returns:
            List of all artifact IDs across all steps
        """
        artifact_ids = []
        for step in self.steps:
            artifact_ids.extend(step.artifacts.artifact_ids)
        return artifact_ids

    def get_artifacts_by_step(self, step_index: int) -> Optional[StepArtifacts]:
        """Get artifacts for a specific step.

        Args:
            step_index: 1-based step index

        Returns:
            StepArtifacts or None if step not found
        """
        step = self.get_step(step_index)
        return step.artifacts if step else None

    def get_model_artifact_id(self) -> Optional[str]:
        """Get the primary model artifact ID.

        Returns:
            Model artifact ID or None if no model step
        """
        if self.model_step_index is None:
            return None
        step = self.get_step(self.model_step_index)
        if step and step.artifacts:
            return step.artifacts.primary_artifact_id
        return None

    def get_fold_artifact_ids(self) -> Dict[int, str]:
        """Get per-fold model artifact IDs.

        Returns:
            Dictionary of fold_id -> artifact_id
        """
        if self.model_step_index is None:
            return {}
        step = self.get_step(self.model_step_index)
        if step and step.artifacts:
            return step.artifacts.fold_artifact_ids.copy()
        return {}

    def set_model_step(
        self,
        step_index: int,
        fold_weights: Optional[Dict[int, float]] = None
    ) -> None:
        """Set the model step index and optional fold weights.

        Args:
            step_index: Index of the model step
            fold_weights: Optional per-fold weights for CV
        """
        self.model_step_index = step_index
        self.fold_weights = fold_weights

    def finalize(
        self,
        preprocessing_chain: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Finalize the trace with summary information.

        Call this after all steps have been recorded to add summary info.

        Args:
            preprocessing_chain: Summary string of preprocessing (e.g., "SNV>SG>MinMax")
            metadata: Additional metadata to merge
        """
        if preprocessing_chain:
            self.preprocessing_chain = preprocessing_chain
        if metadata:
            self.metadata.update(metadata)

    def __repr__(self) -> str:
        n_steps = len(self.steps)
        n_artifacts = len(self.get_artifact_ids())
        return (
            f"ExecutionTrace(id={self.trace_id!r}, "
            f"steps={n_steps}, artifacts={n_artifacts}, "
            f"model_step={self.model_step_index})"
        )
