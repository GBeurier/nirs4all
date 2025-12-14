"""
Execution Trace - Records the exact path through pipeline that produced a prediction.

This module provides the core data structures for recording execution traces,
which enable deterministic prediction replay and pipeline extraction.

The execution trace is controller-agnostic: it records what steps were executed
and what artifacts were produced, without encoding specific controller logic.

Key Classes:
    - StepArtifacts: Artifacts produced by a single step
    - ExecutionStep: Record of a single step's execution
    - ExecutionTrace: Complete trace of a pipeline execution path

Architecture:
    During training, each step execution is recorded in the trace:
    1. Step starts -> record step_index, operator info
    2. Step completes -> record artifacts and output info
    3. Model produces prediction -> trace_id is attached to prediction

    During prediction, the trace is used to:
    1. Identify the minimal set of steps needed
    2. Load the correct artifacts for each step
    3. Execute only required steps via existing controllers
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


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
    """Artifacts produced by a single step.

    Records all artifacts created during step execution, enabling
    artifact injection during prediction replay.

    Attributes:
        artifact_ids: List of artifact IDs produced by this step
        primary_artifact_id: Main artifact (e.g., model) if applicable
        fold_artifact_ids: Per-fold artifacts for CV models
        metadata: Additional artifact metadata (types, paths, etc.)
    """

    artifact_ids: List[str] = field(default_factory=list)
    primary_artifact_id: Optional[str] = None
    fold_artifact_ids: Dict[int, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization.

        Returns:
            Dictionary suitable for manifest storage
        """
        return {
            "artifact_ids": self.artifact_ids,
            "primary_artifact_id": self.primary_artifact_id,
            "fold_artifact_ids": self.fold_artifact_ids,
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

        return cls(
            artifact_ids=data.get("artifact_ids", []),
            primary_artifact_id=data.get("primary_artifact_id"),
            fold_artifact_ids=fold_artifacts,
            metadata=data.get("metadata", {}),
        )

    def add_artifact(self, artifact_id: str, is_primary: bool = False) -> None:
        """Add an artifact ID to this step's artifacts.

        Args:
            artifact_id: The artifact ID to add
            is_primary: Whether this is the primary artifact
        """
        if artifact_id not in self.artifact_ids:
            self.artifact_ids.append(artifact_id)
        if is_primary:
            self.primary_artifact_id = artifact_id

    def add_fold_artifact(self, fold_id: int, artifact_id: str) -> None:
        """Add a fold-specific artifact.

        Args:
            fold_id: CV fold index
            artifact_id: Artifact ID for this fold
        """
        self.fold_artifact_ids[fold_id] = artifact_id
        if artifact_id not in self.artifact_ids:
            self.artifact_ids.append(artifact_id)


@dataclass
class ExecutionStep:
    """Record of a single step's execution in the trace.

    Captures all information needed to replay this step during prediction,
    including operator configuration, execution mode, and produced artifacts.

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
        )

    def has_artifacts(self) -> bool:
        """Check if this step produced any artifacts.

        Returns:
            True if the step has at least one artifact
        """
        return len(self.artifacts.artifact_ids) > 0


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
