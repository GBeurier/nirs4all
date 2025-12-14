"""
Trace Recorder - Records execution traces during pipeline execution.

This module provides the TraceRecorder class which is responsible for
building ExecutionTrace objects during pipeline execution.

The recorder is designed to be controller-agnostic: it records step execution
and artifact creation without knowing about specific controller types.

Usage:
    1. Create a TraceRecorder at the start of pipeline execution
    2. Call start_step() when a step begins
    3. Call record_artifact() when artifacts are created
    4. Call end_step() when a step completes
    5. Call finalize() to get the completed trace
"""

import time
from typing import Any, Dict, List, Optional

from nirs4all.pipeline.trace.execution_trace import (
    ExecutionStep,
    ExecutionTrace,
    StepArtifacts,
    StepExecutionMode,
)


class TraceRecorder:
    """Records execution traces during pipeline execution.

    Builds an ExecutionTrace by recording step starts, artifact creations,
    and step completions. Designed for use within the pipeline executor.

    Attributes:
        trace: The ExecutionTrace being built
        current_step: The step currently being executed
        step_start_time: Time when current step started (for duration)

    Example:
        >>> recorder = TraceRecorder(pipeline_uid="0001_pls_abc123")
        >>> recorder.start_step(step_index=1, operator_type="transform", operator_class="SNV")
        >>> recorder.record_artifact(artifact_id="0001:1:all", is_primary=False)
        >>> recorder.end_step()
        >>> recorder.start_step(step_index=2, operator_type="model", operator_class="PLSRegression")
        >>> recorder.record_artifact(artifact_id="0001:2:0", fold_id=0)
        >>> recorder.record_artifact(artifact_id="0001:2:1", fold_id=1)
        >>> recorder.end_step(is_model=True, fold_weights={0: 0.52, 1: 0.48})
        >>> trace = recorder.finalize(preprocessing_chain="SNV>MinMax")
    """

    def __init__(
        self,
        pipeline_uid: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize trace recorder.

        Args:
            pipeline_uid: Pipeline UID for the trace
            metadata: Optional initial metadata
        """
        self.trace = ExecutionTrace(
            pipeline_uid=pipeline_uid,
            metadata=metadata or {}
        )
        self.current_step: Optional[ExecutionStep] = None
        self.step_start_time: float = 0.0

    def start_step(
        self,
        step_index: int,
        operator_type: str = "",
        operator_class: str = "",
        operator_config: Optional[Dict[str, Any]] = None,
        execution_mode: StepExecutionMode = StepExecutionMode.TRAIN,
        branch_path: Optional[List[int]] = None,
        branch_name: str = ""
    ) -> None:
        """Start recording a new step.

        Args:
            step_index: 1-based step index
            operator_type: Type of operator (e.g., "transform", "model")
            operator_class: Class name of operator
            operator_config: Serialized operator configuration
            execution_mode: Train/predict/skip mode
            branch_path: Branch indices if in branch context
            branch_name: Human-readable branch name
        """
        # Finalize previous step if still open
        if self.current_step is not None:
            self._finalize_current_step()

        self.current_step = ExecutionStep(
            step_index=step_index,
            operator_type=operator_type,
            operator_class=operator_class,
            operator_config=operator_config or {},
            execution_mode=execution_mode,
            branch_path=branch_path or [],
            branch_name=branch_name,
        )
        self.step_start_time = time.time()

    def record_artifact(
        self,
        artifact_id: str,
        is_primary: bool = False,
        fold_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an artifact created during the current step.

        Args:
            artifact_id: The artifact ID
            is_primary: Whether this is the primary artifact
            fold_id: CV fold ID if fold-specific artifact
            metadata: Additional artifact metadata
        """
        if self.current_step is None:
            raise RuntimeError("Cannot record artifact: no step is active")

        if fold_id is not None:
            self.current_step.artifacts.add_fold_artifact(fold_id, artifact_id)
        else:
            self.current_step.artifacts.add_artifact(artifact_id, is_primary)

        if metadata:
            self.current_step.artifacts.metadata.update(metadata)

    def add_step_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the current step.

        Args:
            key: Metadata key
            value: Metadata value
        """
        if self.current_step is not None:
            self.current_step.metadata[key] = value

    def end_step(
        self,
        is_model: bool = False,
        fold_weights: Optional[Dict[int, float]] = None,
        skip_trace: bool = False
    ) -> None:
        """End the current step and add it to the trace.

        Args:
            is_model: Whether this is the model step
            fold_weights: Per-fold weights for CV models
            skip_trace: If True, don't add this step to the trace
        """
        if self.current_step is None:
            return

        self._finalize_current_step()

        if not skip_trace:
            self.trace.add_step(self.current_step)

            if is_model:
                self.trace.set_model_step(
                    step_index=self.current_step.step_index,
                    fold_weights=fold_weights
                )

        self.current_step = None

    def _finalize_current_step(self) -> None:
        """Finalize the current step by calculating duration."""
        if self.current_step is not None:
            end_time = time.time()
            self.current_step.duration_ms = (end_time - self.step_start_time) * 1000

    def mark_step_skipped(self, step_index: int) -> None:
        """Record that a step was skipped.

        Args:
            step_index: Index of the skipped step
        """
        skip_step = ExecutionStep(
            step_index=step_index,
            execution_mode=StepExecutionMode.SKIP,
        )
        self.trace.add_step(skip_step)

    def finalize(
        self,
        preprocessing_chain: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExecutionTrace:
        """Finalize and return the completed trace.

        Args:
            preprocessing_chain: Summary string of preprocessing
            metadata: Additional metadata to merge

        Returns:
            The completed ExecutionTrace
        """
        # Finalize any open step
        if self.current_step is not None:
            self._finalize_current_step()
            self.trace.add_step(self.current_step)
            self.current_step = None

        self.trace.finalize(preprocessing_chain, metadata)
        return self.trace

    @property
    def trace_id(self) -> str:
        """Get the trace ID.

        Returns:
            Trace ID string
        """
        return self.trace.trace_id

    def get_current_step_index(self) -> Optional[int]:
        """Get the current step index.

        Returns:
            Current step index or None if no step active
        """
        return self.current_step.step_index if self.current_step else None

    def has_model_step(self) -> bool:
        """Check if a model step has been recorded.

        Returns:
            True if model step index is set
        """
        return self.trace.model_step_index is not None
