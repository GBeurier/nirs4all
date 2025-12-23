"""
Trace Recorder V3 - Records execution traces during pipeline execution.

This module provides the TraceRecorder class which is responsible for
building ExecutionTrace objects during pipeline execution.

V3 improvements:
- Chain stack for tracking operator chain through execution
- Branch stack for automatic branch path management
- Proper recording of branch substeps as individual steps
- Support for multi-source artifact tracking

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

from nirs4all.pipeline.storage.artifacts.operator_chain import (
    OperatorChain,
    OperatorNode,
)
from nirs4all.pipeline.trace.execution_trace import (
    ExecutionStep,
    ExecutionTrace,
    StepArtifacts,
    StepExecutionMode,
)


class TraceRecorder:
    """Records execution traces during pipeline execution (V3).

    Builds an ExecutionTrace by recording step starts, artifact creations,
    and step completions. Designed for use within the pipeline executor.

    V3 improvements:
    - Maintains a chain stack for tracking full operator chain
    - Maintains a branch stack for automatic branch path management
    - Tracks source index for multi-source pipelines
    - Records branch substeps individually

    Attributes:
        trace: The ExecutionTrace being built
        current_step: The step currently being executed
        step_start_time: Time when current step started (for duration)
        pipeline_id: Pipeline identifier for chain generation

    Example:
        >>> recorder = TraceRecorder(pipeline_uid="0001_pls_abc123")
        >>> recorder.start_step(step_index=1, operator_type="transform", operator_class="SNV")
        >>> recorder.record_artifact(artifact_id="0001$abc123:all", chain_path="s1.SNV")
        >>> recorder.end_step()
        >>> recorder.enter_branch(0)
        >>> recorder.start_step(step_index=3, operator_type="transform", operator_class="PLS")
        >>> recorder.record_artifact(artifact_id="0001$def456:0", chain_path="s1.SNV>s3.PLS[br=0]")
        >>> recorder.end_step(is_model=True)
        >>> recorder.exit_branch()
        >>> trace = recorder.finalize(preprocessing_chain="SNV>MinMax")
    """

    def __init__(
        self,
        pipeline_uid: str = "",
        pipeline_id: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize trace recorder.

        Args:
            pipeline_uid: Pipeline UID for the trace
            pipeline_id: Pipeline ID for chain generation
            metadata: Optional initial metadata
        """
        self.trace = ExecutionTrace(
            pipeline_uid=pipeline_uid,
            metadata=metadata or {}
        )
        self.pipeline_id = pipeline_id or pipeline_uid.split("_")[0] if pipeline_uid else ""
        self.current_step: Optional[ExecutionStep] = None
        self.step_start_time: float = 0.0

        # V3: Chain and branch stacks
        self._chain_stack: List[OperatorChain] = [OperatorChain(pipeline_id=self.pipeline_id)]
        self._branch_stack: List[List[int]] = [[]]

    # =========================================================================
    # V3 Chain Management
    # =========================================================================

    def current_chain(self) -> OperatorChain:
        """Get current operator chain without modifying stack.

        Returns:
            Current OperatorChain
        """
        return self._chain_stack[-1].copy()

    def push_chain(self, node: OperatorNode) -> OperatorChain:
        """Push new node onto the chain stack.

        Creates a new chain with the node appended and pushes it.

        Args:
            node: OperatorNode to append

        Returns:
            The new extended chain
        """
        current = self._chain_stack[-1]
        extended = current.append(node)
        self._chain_stack.append(extended)
        return extended

    def pop_chain(self) -> OperatorChain:
        """Pop and return the current chain.

        Returns:
            The popped OperatorChain

        Raises:
            RuntimeError: If trying to pop the root chain
        """
        if len(self._chain_stack) <= 1:
            raise RuntimeError("Cannot pop root chain")
        return self._chain_stack.pop()

    def reset_chain_to(self, chain: OperatorChain) -> None:
        """Reset chain stack to a specific chain.

        Useful when entering a new branch context.

        Args:
            chain: Chain to reset to
        """
        # Keep only root and the new chain
        self._chain_stack = [self._chain_stack[0], chain]

    # =========================================================================
    # V3 Branch Management
    # =========================================================================

    def current_branch_path(self) -> List[int]:
        """Get current branch path.

        Returns:
            Copy of current branch path
        """
        return self._branch_stack[-1].copy()

    def enter_branch(self, branch_id: int) -> List[int]:
        """Enter a branch context.

        Args:
            branch_id: Branch index to enter

        Returns:
            New branch path after entering
        """
        current = self._branch_stack[-1].copy()
        current.append(branch_id)
        self._branch_stack.append(current)
        return current

    def exit_branch(self) -> List[int]:
        """Exit current branch context.

        Returns:
            The exited branch path

        Raises:
            RuntimeError: If not in a branch context
        """
        if len(self._branch_stack) <= 1:
            raise RuntimeError("Cannot exit root branch context")
        return self._branch_stack.pop()

    def in_branch(self) -> bool:
        """Check if currently in a branch context.

        Returns:
            True if in a branch
        """
        return len(self._branch_stack) > 1 or bool(self._branch_stack[-1])

    # =========================================================================
    # Step Recording
    # =========================================================================

    def start_step(
        self,
        step_index: int,
        operator_type: str = "",
        operator_class: str = "",
        operator_config: Optional[Dict[str, Any]] = None,
        execution_mode: StepExecutionMode = StepExecutionMode.TRAIN,
        branch_path: Optional[List[int]] = None,
        branch_name: str = "",
        source_count: int = 1,
        produces_branches: bool = False,
        substep_index: Optional[int] = None,
    ) -> ExecutionStep:
        """Start recording a new step (V3).

        Args:
            step_index: 1-based step index
            operator_type: Type of operator (e.g., "transform", "model")
            operator_class: Class name of operator
            operator_config: Serialized operator configuration
            execution_mode: Train/predict/skip mode
            branch_path: Branch indices (uses current if None)
            branch_name: Human-readable branch name
            source_count: Number of X sources at this step
            produces_branches: Whether this is a branch operator
            substep_index: Index within substep

        Returns:
            The created ExecutionStep
        """
        # Finalize previous step if still open
        if self.current_step is not None:
            self._finalize_current_step()

        # Use provided branch_path or current from stack
        effective_branch_path = branch_path if branch_path is not None else self.current_branch_path()

        # Build input chain path from current chain
        input_chain = self.current_chain()

        self.current_step = ExecutionStep(
            step_index=step_index,
            operator_type=operator_type,
            operator_class=operator_class,
            operator_config=operator_config or {},
            execution_mode=execution_mode,
            branch_path=effective_branch_path,
            branch_name=branch_name,
            source_count=source_count,
            produces_branches=produces_branches,
            substep_index=substep_index,
            input_chain_path=input_chain.to_path(),
        )
        self.step_start_time = time.time()

        return self.current_step

    def record_artifact(
        self,
        artifact_id: str,
        is_primary: bool = False,
        fold_id: Optional[int] = None,
        chain_path: Optional[str] = None,
        branch_path: Optional[List[int]] = None,
        source_index: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an artifact created during the current step (V3).

        Args:
            artifact_id: The artifact ID
            is_primary: Whether this is the primary artifact
            fold_id: CV fold ID if fold-specific artifact
            chain_path: V3 operator chain path
            branch_path: Branch path for indexing
            source_index: Source index for multi-source
            metadata: Additional artifact metadata
        """
        if self.current_step is None:
            raise RuntimeError("Cannot record artifact: no step is active")

        # Use current branch path if not provided
        if branch_path is None:
            branch_path = self.current_branch_path()

        if fold_id is not None:
            self.current_step.artifacts.add_fold_artifact(
                fold_id,
                artifact_id,
                chain_path=chain_path,
                branch_path=branch_path,
            )
        else:
            self.current_step.artifacts.add_artifact(
                artifact_id,
                is_primary=is_primary,
                chain_path=chain_path,
                branch_path=branch_path,
                source_index=source_index,
            )

        # Track output chain
        if chain_path:
            self.current_step.add_output_chain(chain_path)

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

    def record_input_shapes(
        self,
        input_shape: Optional[tuple] = None,
        features_shape: Optional[List[tuple]] = None
    ) -> None:
        """Record input shapes for the current step.

        Args:
            input_shape: 2D layout shape (samples, features)
            features_shape: List of 3D shapes per source (samples, processings, features)
        """
        if self.current_step is not None:
            if input_shape is not None:
                self.current_step.input_shape = tuple(input_shape)
            if features_shape is not None:
                self.current_step.input_features_shape = [tuple(s) for s in features_shape]

    def record_output_shapes(
        self,
        output_shape: Optional[tuple] = None,
        features_shape: Optional[List[tuple]] = None
    ) -> None:
        """Record output shapes for the current step.

        Args:
            output_shape: 2D layout shape (samples, features)
            features_shape: List of 3D shapes per source (samples, processings, features)
        """
        if self.current_step is not None:
            if output_shape is not None:
                self.current_step.output_shape = tuple(output_shape)
            if features_shape is not None:
                self.current_step.output_features_shape = [tuple(s) for s in features_shape]

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
            branch_path=self.current_branch_path(),
        )
        self.trace.add_step(skip_step)

    # =========================================================================
    # V3 Branch Step Recording
    # =========================================================================

    def start_branch_step(
        self,
        step_index: int,
        branch_count: int,
        operator_config: Optional[Dict[str, Any]] = None,
    ) -> ExecutionStep:
        """Start recording a branch step.

        Args:
            step_index: Step index of the branch
            branch_count: Number of branches
            operator_config: Branch configuration

        Returns:
            The created ExecutionStep for the branch
        """
        return self.start_step(
            step_index=step_index,
            operator_type="branch",
            operator_class="Branch",
            operator_config=operator_config or {"branch_count": branch_count},
            produces_branches=True,
        )

    def start_branch_substep(
        self,
        parent_step_index: int,
        branch_id: int,
        operator_type: str,
        operator_class: str,
        substep_index: int = 0,
        operator_config: Optional[Dict[str, Any]] = None,
        branch_name: Optional[str] = None,
    ) -> ExecutionStep:
        """Start recording a substep within a branch.

        Note: This method assumes enter_branch() has already been called for
        this branch, so current_branch_path() already includes the branch_id.

        Args:
            parent_step_index: Parent branch step index
            branch_id: Branch index this substep belongs to (for metadata only)
            operator_type: Type of operator
            operator_class: Class name of operator
            substep_index: Index within the branch's substeps
            operator_config: Operator configuration
            branch_name: Human-readable branch name

        Returns:
            The created ExecutionStep
        """
        # Use current_branch_path() directly - enter_branch() already pushed the branch_id
        current_path = self.current_branch_path()

        # Create operator node for this substep
        node = OperatorNode(
            step_index=parent_step_index,
            operator_class=operator_class,
            branch_path=current_path,
            substep_index=substep_index,
        )

        # Push onto chain
        self.push_chain(node)

        return self.start_step(
            step_index=parent_step_index,
            operator_type=operator_type,
            operator_class=operator_class,
            operator_config=operator_config,
            branch_path=current_path,
            branch_name=branch_name,
            substep_index=substep_index,
        )

    # =========================================================================
    # Finalization
    # =========================================================================

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

    # =========================================================================
    # Utilities
    # =========================================================================

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

    def build_chain_for_artifact(
        self,
        step_index: int,
        operator_class: str,
        source_index: Optional[int] = None,
        fold_id: Optional[int] = None,
        substep_index: Optional[int] = None,
    ) -> OperatorChain:
        """Build an operator chain for an artifact.

        Creates a chain based on current context plus the specified operator.

        Args:
            step_index: Step index of the operator
            operator_class: Class name of the operator
            source_index: Source index for multi-source
            fold_id: Fold ID for CV models
            substep_index: Substep index within step

        Returns:
            OperatorChain for the artifact
        """
        node = OperatorNode(
            step_index=step_index,
            operator_class=operator_class,
            branch_path=self.current_branch_path(),
            source_index=source_index,
            fold_id=fold_id,
            substep_index=substep_index,
        )
        return self.current_chain().append(node)
