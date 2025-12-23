"""
Trace-Based Extractor - Extract minimal pipeline from execution trace.

This module provides the TraceBasedExtractor class which extracts the minimal
set of pipeline steps needed to replay a prediction from an ExecutionTrace.

The extractor is controller-agnostic: it uses the recorded trace information
to identify which steps are needed, without encoding knowledge of specific
controller types. This enables the same infrastructure to work with any
controller (existing or custom).

Key Features:
    - Extracts minimal pipeline (only steps needed for prediction)
    - Builds artifact map for step-based artifact injection
    - Creates dependency graph for step execution order
    - Supports branch-aware extraction for multi-branch pipelines

Usage:
    >>> from nirs4all.pipeline.trace import TraceBasedExtractor, ExecutionTrace
    >>>
    >>> # Load trace from manifest
    >>> trace = manifest_manager.load_execution_trace(pipeline_uid, trace_id)
    >>>
    >>> # Extract minimal pipeline
    >>> extractor = TraceBasedExtractor()
    >>> minimal = extractor.extract(trace, full_pipeline_steps)
    >>>
    >>> # Use minimal pipeline for prediction
    >>> for step_info in minimal.steps:
    ...     artifacts = minimal.get_artifacts_for_step(step_info.step_index)
    ...     # Execute step with artifacts
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from nirs4all.pipeline.trace.execution_trace import (
    ExecutionTrace,
    ExecutionStep,
    StepArtifacts,
    StepExecutionMode,
)


logger = logging.getLogger(__name__)


@dataclass
class MinimalPipelineStep:
    """A step in the minimal pipeline for prediction replay.

    Contains the step configuration and metadata needed to replay
    the step during prediction, without encoding controller-specific logic.

    Attributes:
        step_index: 1-based step index from original pipeline
        step_config: The pipeline step configuration (dict or object)
        execution_mode: How to execute this step (train/predict/skip)
        artifacts: Artifacts for this step (from trace)
        operator_type: Type of operator (for logging/debugging)
        operator_class: Class name of operator
        branch_path: Branch path if in branch context
        branch_name: Human-readable branch name
        depends_on: Indices of steps this step depends on
    """

    step_index: int
    step_config: Any = None
    execution_mode: StepExecutionMode = StepExecutionMode.PREDICT
    artifacts: StepArtifacts = field(default_factory=StepArtifacts)
    operator_type: str = ""
    operator_class: str = ""
    branch_path: List[int] = field(default_factory=list)
    branch_name: str = ""
    substep_index: Optional[int] = None
    depends_on: Set[int] = field(default_factory=set)

    def has_artifacts(self) -> bool:
        """Check if this step has associated artifacts.

        Returns:
            True if artifacts are available for this step
        """
        return len(self.artifacts.artifact_ids) > 0

    def get_artifact_ids(self) -> List[str]:
        """Get all artifact IDs for this step.

        Returns:
            List of artifact IDs
        """
        return list(self.artifacts.artifact_ids)

    def get_artifact_by_chain(self, chain_path: str) -> Optional[str]:
        """Get artifact ID by V3 chain path.

        Args:
            chain_path: Operator chain path

        Returns:
            Artifact ID or None if not found
        """
        return self.artifacts.get_artifact_by_chain(chain_path)

    def get_artifacts_by_chain(self) -> Dict[str, str]:
        """Get all artifacts indexed by chain path.

        Returns:
            Dict mapping chain_path to artifact_id
        """
        return dict(self.artifacts.by_chain) if self.artifacts.by_chain else {}


@dataclass
class MinimalPipeline:
    """Minimal pipeline extracted from an execution trace.

    Contains only the steps needed to replay a prediction, with artifact
    mappings for each step. Used by MinimalPredictor for efficient prediction.

    Attributes:
        trace_id: ID of the source execution trace
        pipeline_uid: UID of the parent pipeline
        steps: Ordered list of minimal steps to execute
        artifact_map: Mapping of step_index to list of (artifact_id, step_artifacts)
        model_step_index: Index of the model step
        fold_weights: Per-fold weights for CV ensemble
        preprocessing_chain: Summary of preprocessing steps
        metadata: Additional metadata from trace
    """

    trace_id: str = ""
    pipeline_uid: str = ""
    steps: List[MinimalPipelineStep] = field(default_factory=list)
    artifact_map: Dict[int, StepArtifacts] = field(default_factory=dict)
    model_step_index: Optional[int] = None
    fold_weights: Optional[Dict[int, float]] = None
    preprocessing_chain: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_step(self, step_index: int) -> Optional[MinimalPipelineStep]:
        """Get a step by its index.

        Args:
            step_index: 1-based step index

        Returns:
            MinimalPipelineStep or None if not found
        """
        for step in self.steps:
            if step.step_index == step_index:
                return step
        return None

    def get_artifacts_for_step(self, step_index: int) -> Optional[StepArtifacts]:
        """Get artifacts for a specific step.

        Args:
            step_index: 1-based step index

        Returns:
            StepArtifacts or None if not found
        """
        return self.artifact_map.get(step_index)

    def has_step(self, step_index: int) -> bool:
        """Check if a step is included in the minimal pipeline.

        Args:
            step_index: 1-based step index

        Returns:
            True if step is included
        """
        return any(s.step_index == step_index for s in self.steps)

    def get_step_count(self) -> int:
        """Get the number of steps in the minimal pipeline.

        Returns:
            Number of steps
        """
        return len(self.steps)

    def get_artifact_ids(self) -> List[str]:
        """Get all artifact IDs in the minimal pipeline.

        Returns:
            List of all artifact IDs across all steps
        """
        artifact_ids = []
        for artifacts in self.artifact_map.values():
            artifact_ids.extend(artifacts.artifact_ids)
        return artifact_ids

    def get_step_indices(self) -> List[int]:
        """Get all step indices in execution order.

        Returns:
            List of step indices
        """
        return [s.step_index for s in self.steps]

    def get_artifact_by_chain(self, chain_path: str) -> Optional[str]:
        """Get artifact ID by V3 chain path across all steps.

        Args:
            chain_path: Operator chain path

        Returns:
            Artifact ID or None if not found
        """
        for step_artifacts in self.artifact_map.values():
            artifact_id = step_artifacts.get_artifact_by_chain(chain_path)
            if artifact_id:
                return artifact_id
        return None

    def get_all_chain_paths(self) -> Dict[str, str]:
        """Get all artifacts indexed by chain path.

        Returns:
            Dict mapping chain_path to artifact_id
        """
        chain_map = {}
        for step_artifacts in self.artifact_map.values():
            if step_artifacts.by_chain:
                chain_map.update(step_artifacts.by_chain)
        return chain_map

    def __repr__(self) -> str:
        n_steps = len(self.steps)
        n_artifacts = len(self.get_artifact_ids())
        return (
            f"MinimalPipeline(trace_id={self.trace_id!r}, "
            f"steps={n_steps}, artifacts={n_artifacts}, "
            f"model_step={self.model_step_index})"
        )


class TraceBasedExtractor:
    """Extract minimal pipeline from execution trace.

    The extractor analyzes an ExecutionTrace to determine which steps
    are needed for prediction replay and builds a MinimalPipeline with
    the correct artifact mappings.

    The extractor is controller-agnostic: it uses trace metadata
    to identify steps without encoding knowledge of controller types.

    Attributes:
        include_skipped: Whether to include skipped steps in minimal pipeline
        preserve_order: Whether to preserve original step order

    Example:
        >>> extractor = TraceBasedExtractor()
        >>> trace = manifest_manager.load_execution_trace(pipeline_uid, trace_id)
        >>> minimal = extractor.extract(trace, full_pipeline_steps)
        >>> print(f"Minimal pipeline has {minimal.get_step_count()} steps")
    """

    def __init__(
        self,
        include_skipped: bool = False,
        preserve_order: bool = True
    ):
        """Initialize trace-based extractor.

        Args:
            include_skipped: Whether to include steps that were skipped
            preserve_order: Whether to preserve original step execution order
        """
        self.include_skipped = include_skipped
        self.preserve_order = preserve_order

    def extract(
        self,
        trace: ExecutionTrace,
        full_pipeline: Optional[List[Any]] = None,
        up_to_model: bool = True
    ) -> MinimalPipeline:
        """Extract minimal pipeline from execution trace.

        Analyzes the trace to determine which steps are needed for prediction
        and builds a MinimalPipeline with artifact mappings.

        Args:
            trace: ExecutionTrace to extract from
            full_pipeline: Optional full pipeline steps (for step configs)
            up_to_model: If True, only include steps up to model step

        Returns:
            MinimalPipeline with steps and artifact mappings
        """
        minimal = MinimalPipeline(
            trace_id=trace.trace_id,
            pipeline_uid=trace.pipeline_uid,
            model_step_index=trace.model_step_index,
            fold_weights=trace.fold_weights,
            preprocessing_chain=trace.preprocessing_chain,
            metadata=trace.metadata.copy() if trace.metadata else {}
        )

        # Get steps to include
        if up_to_model and trace.model_step_index is not None:
            trace_steps = trace.get_steps_up_to_model()
        else:
            trace_steps = trace.steps

        # Determine if model is in a branch (has non-empty branch_path)
        # If not (feature merge case), we need to deduplicate branch steps
        model_in_branch = False
        if trace.model_step_index is not None:
            model_step = trace.get_step(trace.model_step_index)
            if model_step and model_step.branch_path:
                model_in_branch = True

        # Track which step_indices we've already added for deduplication
        # This is needed for feature merge where branch steps should only run once
        added_step_indices: Set[int] = set()

        # Build minimal steps from trace
        for exec_step in trace_steps:
            # Skip if mode is SKIP and we're not including skipped
            if exec_step.execution_mode == StepExecutionMode.SKIP and not self.include_skipped:
                continue

            # Get step config from full pipeline if available
            step_config = None
            if full_pipeline and 0 < exec_step.step_index <= len(full_pipeline):
                step_config = full_pipeline[exec_step.step_index - 1]

            # Deduplication for feature merge case:
            # When model is NOT in a branch, branch steps should only appear once
            # so the branch controller can iterate all branches internally
            is_branch_step = (
                isinstance(step_config, dict) and "branch" in step_config
            )
            if not model_in_branch and is_branch_step:
                if exec_step.step_index in added_step_indices:
                    # Skip this duplicate branch step entry, but still merge artifacts
                    if exec_step.has_artifacts():
                        if exec_step.step_index in minimal.artifact_map:
                            minimal.artifact_map[exec_step.step_index].merge(exec_step.artifacts)
                        else:
                            from copy import deepcopy
                            minimal.artifact_map[exec_step.step_index] = deepcopy(exec_step.artifacts)
                    continue

            # Create minimal step
            minimal_step = MinimalPipelineStep(
                step_index=exec_step.step_index,
                step_config=step_config,
                execution_mode=StepExecutionMode.PREDICT,  # Force predict mode
                artifacts=exec_step.artifacts,
                operator_type=exec_step.operator_type,
                operator_class=exec_step.operator_class,
                branch_path=list(exec_step.branch_path),
                branch_name=exec_step.branch_name
            )

            minimal.steps.append(minimal_step)
            added_step_indices.add(exec_step.step_index)

            # Add to artifact map, merging if step_index already exists
            if exec_step.has_artifacts():
                if exec_step.step_index in minimal.artifact_map:
                    minimal.artifact_map[exec_step.step_index].merge(exec_step.artifacts)
                else:
                    from copy import deepcopy
                    minimal.artifact_map[exec_step.step_index] = deepcopy(exec_step.artifacts)

        # Sort by step index if preserving order
        if self.preserve_order:
            minimal.steps.sort(key=lambda s: s.step_index)

        logger.debug(
            f"Extracted minimal pipeline: {minimal.get_step_count()} steps, "
            f"{len(minimal.get_artifact_ids())} artifacts"
        )

        return minimal

    def extract_for_step(
        self,
        trace: ExecutionTrace,
        target_step_index: int,
        full_pipeline: Optional[List[Any]] = None
    ) -> MinimalPipeline:
        """Extract minimal pipeline up to a specific step.

        Useful for partial prediction or when targeting a specific model
        in a multi-model pipeline.

        Args:
            trace: ExecutionTrace to extract from
            target_step_index: Target step index (inclusive)
            full_pipeline: Optional full pipeline steps

        Returns:
            MinimalPipeline with steps up to target
        """
        minimal = MinimalPipeline(
            trace_id=trace.trace_id,
            pipeline_uid=trace.pipeline_uid,
            preprocessing_chain=trace.preprocessing_chain,
            metadata=trace.metadata.copy() if trace.metadata else {}
        )

        # Get steps up to target
        trace_steps = [s for s in trace.steps if s.step_index <= target_step_index]

        # Check if target is the model step
        target_step = trace.get_step(target_step_index)
        if target_step and target_step.operator_type in ("model", "meta_model"):
            minimal.model_step_index = target_step_index
            minimal.fold_weights = trace.fold_weights

        for exec_step in trace_steps:
            if exec_step.execution_mode == StepExecutionMode.SKIP and not self.include_skipped:
                continue

            step_config = None
            if full_pipeline and 0 < exec_step.step_index <= len(full_pipeline):
                step_config = full_pipeline[exec_step.step_index - 1]

            minimal_step = MinimalPipelineStep(
                step_index=exec_step.step_index,
                step_config=step_config,
                execution_mode=StepExecutionMode.PREDICT,
                artifacts=exec_step.artifacts,
                operator_type=exec_step.operator_type,
                operator_class=exec_step.operator_class,
                branch_path=list(exec_step.branch_path),
                branch_name=exec_step.branch_name
            )

            minimal.steps.append(minimal_step)

            if exec_step.has_artifacts():
                if exec_step.step_index in minimal.artifact_map:
                    minimal.artifact_map[exec_step.step_index].merge(exec_step.artifacts)
                else:
                    from copy import deepcopy
                    minimal.artifact_map[exec_step.step_index] = deepcopy(exec_step.artifacts)

        if self.preserve_order:
            minimal.steps.sort(key=lambda s: s.step_index)

        return minimal

    def extract_for_branch(
        self,
        trace: ExecutionTrace,
        branch_path: List[int],
        full_pipeline: Optional[List[Any]] = None
    ) -> MinimalPipeline:
        """Extract minimal pipeline for a specific branch.

        Includes shared steps (before branching) plus branch-specific steps.

        Args:
            trace: ExecutionTrace to extract from
            branch_path: Branch path to extract (e.g., [0] for first branch)
            full_pipeline: Optional full pipeline steps

        Returns:
            MinimalPipeline with steps for the specified branch
        """
        minimal = MinimalPipeline(
            trace_id=trace.trace_id,
            pipeline_uid=trace.pipeline_uid,
            model_step_index=trace.model_step_index,
            fold_weights=trace.fold_weights,
            preprocessing_chain=trace.preprocessing_chain,
            metadata=trace.metadata.copy() if trace.metadata else {}
        )

        # Get steps to include (up to model or all)
        if trace.model_step_index is not None:
            trace_steps = trace.get_steps_up_to_model()
        else:
            trace_steps = trace.steps

        for exec_step in trace_steps:
            if exec_step.execution_mode == StepExecutionMode.SKIP and not self.include_skipped:
                continue

            # Include step if:
            # 1. It has no branch (shared/pre-branch step)
            # 2. It's on the target branch path (exact or nested within)
            include_step = False
            if not exec_step.branch_path:
                include_step = True  # Shared step
            elif exec_step.branch_path == branch_path:
                include_step = True  # Exact branch match
            elif self._is_prefix_branch(branch_path, exec_step.branch_path):
                include_step = True  # Target is prefix of step's path (nested branch)
            elif self._is_parent_branch(branch_path, exec_step.branch_path):
                include_step = True  # Step is a parent branch of target

            if not include_step:
                continue

            # Get step config - for branch substeps, extract the individual substep config
            step_config = None
            if full_pipeline and 0 < exec_step.step_index <= len(full_pipeline):
                parent_config = full_pipeline[exec_step.step_index - 1]
                # If this is a substep inside a branch, extract the specific substep config
                if exec_step.branch_path and exec_step.branch_name:
                    substep_config = self._extract_substep_config(
                        parent_config, exec_step.branch_name, exec_step.substep_index
                    )
                    if substep_config is not None:
                        step_config = substep_config
                    else:
                        # Fallback to parent config if extraction fails
                        step_config = parent_config
                else:
                    step_config = parent_config

            minimal_step = MinimalPipelineStep(
                step_index=exec_step.step_index,
                step_config=step_config,
                execution_mode=StepExecutionMode.PREDICT,
                artifacts=exec_step.artifacts,
                operator_type=exec_step.operator_type,
                operator_class=exec_step.operator_class,
                branch_path=list(exec_step.branch_path),
                branch_name=exec_step.branch_name,
                substep_index=exec_step.substep_index
            )

            minimal.steps.append(minimal_step)

            if exec_step.has_artifacts():
                # Merge artifacts for substeps with the same step_index
                if exec_step.step_index in minimal.artifact_map:
                    minimal.artifact_map[exec_step.step_index].merge(exec_step.artifacts)
                else:
                    from copy import deepcopy
                    minimal.artifact_map[exec_step.step_index] = deepcopy(exec_step.artifacts)

        if self.preserve_order:
            minimal.steps.sort(key=lambda s: s.step_index)

        return minimal

    def extract_for_branch_name(
        self,
        trace: ExecutionTrace,
        branch_name: str,
        full_pipeline: Optional[List[Any]] = None
    ) -> MinimalPipeline:
        """Extract minimal pipeline for a specific branch by name.

        More reliable than extract_for_branch for nested branches where
        branch_id doesn't map directly to branch_path. Uses branch_name
        for matching since it's unique and stored in both predictions and trace.

        Includes shared steps (before branching) plus branch-specific steps.

        Args:
            trace: ExecutionTrace to extract from
            branch_name: Branch name to match (e.g., "branch_0_branch_0")
            full_pipeline: Optional full pipeline steps

        Returns:
            MinimalPipeline with steps for the specified branch
        """
        minimal = MinimalPipeline(
            trace_id=trace.trace_id,
            pipeline_uid=trace.pipeline_uid,
            model_step_index=trace.model_step_index,
            fold_weights=trace.fold_weights,
            preprocessing_chain=trace.preprocessing_chain,
            metadata=trace.metadata.copy() if trace.metadata else {}
        )

        # Get steps to include (up to model or all)
        if trace.model_step_index is not None:
            trace_steps = trace.get_steps_up_to_model()
        else:
            trace_steps = trace.steps

        for exec_step in trace_steps:
            if exec_step.execution_mode == StepExecutionMode.SKIP and not self.include_skipped:
                continue

            # Include step if:
            # 1. It has no branch (shared/pre-branch step)
            # 2. It has matching branch_name
            include_step = False
            if not exec_step.branch_path:
                include_step = True  # Shared step
            elif exec_step.branch_name == branch_name:
                include_step = True  # Exact branch name match

            if not include_step:
                continue

            # Get step config - for branch substeps, extract the individual substep config
            step_config = None
            if full_pipeline and 0 < exec_step.step_index <= len(full_pipeline):
                parent_config = full_pipeline[exec_step.step_index - 1]
                # If this is a substep inside a branch, extract the specific substep config
                if exec_step.branch_path and exec_step.branch_name:
                    substep_config = self._extract_substep_config(
                        parent_config, exec_step.branch_name, exec_step.substep_index
                    )
                    if substep_config is not None:
                        step_config = substep_config
                    else:
                        # Fallback to parent config if extraction fails
                        step_config = parent_config
                else:
                    step_config = parent_config

            minimal_step = MinimalPipelineStep(
                step_index=exec_step.step_index,
                step_config=step_config,
                execution_mode=StepExecutionMode.PREDICT,
                artifacts=exec_step.artifacts,
                operator_type=exec_step.operator_type,
                operator_class=exec_step.operator_class,
                branch_path=list(exec_step.branch_path),
                branch_name=exec_step.branch_name,
                substep_index=exec_step.substep_index
            )

            minimal.steps.append(minimal_step)

            if exec_step.has_artifacts():
                # Merge artifacts for substeps with the same step_index
                if exec_step.step_index in minimal.artifact_map:
                    minimal.artifact_map[exec_step.step_index].merge(exec_step.artifacts)
                else:
                    # Create a copy to avoid modifying the original trace
                    from copy import deepcopy
                    minimal.artifact_map[exec_step.step_index] = deepcopy(exec_step.artifacts)

        if self.preserve_order:
            minimal.steps.sort(key=lambda s: s.step_index)

        return minimal

    def get_required_artifact_ids(
        self,
        trace: ExecutionTrace,
        up_to_model: bool = True
    ) -> List[str]:
        """Get list of artifact IDs required for prediction.

        Useful for pre-loading artifacts or validating artifact availability.

        Args:
            trace: ExecutionTrace to analyze
            up_to_model: If True, only include artifacts up to model step

        Returns:
            List of artifact IDs needed for prediction
        """
        if up_to_model and trace.model_step_index is not None:
            steps = trace.get_steps_up_to_model()
        else:
            steps = trace.steps

        artifact_ids = []
        for step in steps:
            if step.execution_mode != StepExecutionMode.SKIP:
                artifact_ids.extend(step.artifacts.artifact_ids)

        return artifact_ids

    def get_step_dependency_graph(
        self,
        trace: ExecutionTrace
    ) -> Dict[int, Set[int]]:
        """Build dependency graph from execution trace.

        The dependency graph maps each step to the set of steps it depends on.
        This is inferred from the trace execution order and branch structure.

        Args:
            trace: ExecutionTrace to analyze

        Returns:
            Dictionary mapping step_index to set of dependency step indices
        """
        dependencies: Dict[int, Set[int]] = {}
        prev_step_by_branch: Dict[Tuple[int, ...], int] = {}

        for step in trace.steps:
            step_idx = step.step_index
            branch_key = tuple(step.branch_path) if step.branch_path else ()
            dependencies[step_idx] = set()

            # Depend on previous step in same branch
            if branch_key in prev_step_by_branch:
                dependencies[step_idx].add(prev_step_by_branch[branch_key])

            # If in a branch, also depend on last pre-branch step
            if branch_key and () in prev_step_by_branch:
                # Find the last shared step before branching
                shared_step = prev_step_by_branch[()]
                if shared_step not in dependencies[step_idx]:
                    dependencies[step_idx].add(shared_step)

            # Update previous step for this branch
            prev_step_by_branch[branch_key] = step_idx

        return dependencies

    def _extract_substep_config(
        self,
        branch_step_config: Any,
        branch_name: str,
        substep_index: Optional[int]
    ) -> Any:
        """Extract individual substep config from a branch step.

        When a step is inside a branch, the full_pipeline contains the branch
        step config (e.g., {'branch': {'ridge': [...], 'pls': [...]}}).
        This method extracts the specific substep config for the given branch
        and substep index.

        Args:
            branch_step_config: The branch step configuration from full_pipeline
            branch_name: Name of the branch (e.g., 'pls', 'branch_0')
            substep_index: Index of substep within the branch (0-based)

        Returns:
            The extracted substep config, or None if extraction fails
        """
        if not isinstance(branch_step_config, dict):
            return None

        if "branch" not in branch_step_config:
            return None

        branches = branch_step_config["branch"]
        if not branches:
            return None

        # Handle dict branches (named): {'branch': {'ridge': [...], 'pls': [...]}}
        if isinstance(branches, dict):
            # Try exact name match first
            if branch_name in branches:
                branch_steps = branches[branch_name]
            else:
                # For generated names like 'branch_0', try to match by index
                # E.g., 'branch_0' -> index 0 -> first branch
                branch_keys = list(branches.keys())
                if branch_name.startswith("branch_"):
                    try:
                        idx = int(branch_name.split("_")[1])
                        if 0 <= idx < len(branch_keys):
                            branch_steps = branches[branch_keys[idx]]
                        else:
                            return None
                    except (ValueError, IndexError):
                        return None
                else:
                    return None
        # Handle list branches: {'branch': [[...], [...]]}
        elif isinstance(branches, list):
            # Extract branch index from name (e.g., 'branch_0' -> 0)
            if branch_name.startswith("branch_"):
                try:
                    idx = int(branch_name.split("_")[1])
                    if 0 <= idx < len(branches):
                        branch_steps = branches[idx]
                    else:
                        return None
                except (ValueError, IndexError):
                    return None
            else:
                return None
        else:
            return None

        # Now extract substep from branch_steps
        if substep_index is not None and isinstance(branch_steps, list):
            if 0 <= substep_index < len(branch_steps):
                return branch_steps[substep_index]
            else:
                logger.warning(
                    f"Substep index {substep_index} out of range for branch "
                    f"'{branch_name}' with {len(branch_steps)} steps"
                )
                return None

        return None

    def _is_prefix_branch(
        self,
        prefix_path: List[int],
        full_path: List[int]
    ) -> bool:
        """Check if prefix_path is a prefix of full_path.

        Used to match nested branches where we want a specific outer branch
        but any inner branch. For example, [0] is a prefix of [0, 0] and [0, 1].

        Args:
            prefix_path: The potential prefix path
            full_path: The full branch path to check

        Returns:
            True if prefix_path is a proper prefix of full_path
        """
        if len(prefix_path) >= len(full_path):
            return False
        return full_path[:len(prefix_path)] == prefix_path

    def _is_parent_branch(
        self,
        target_path: List[int],
        check_path: List[int]
    ) -> bool:
        """Check if check_path is a parent of target_path.

        Args:
            target_path: Target branch path
            check_path: Path to check if it's a parent

        Returns:
            True if check_path is a prefix of target_path
        """
        if len(check_path) >= len(target_path):
            return False
        return target_path[:len(check_path)] == check_path

    def validate_trace_for_prediction(
        self,
        trace: ExecutionTrace
    ) -> Tuple[bool, List[str]]:
        """Validate that a trace has all information needed for prediction.

        Checks that:
        - Model step is recorded
        - All steps up to model have recorded artifacts (if applicable)
        - No critical information is missing

        Args:
            trace: ExecutionTrace to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check model step
        if trace.model_step_index is None:
            issues.append("No model step recorded in trace")

        # Check steps have artifacts where expected
        steps_needing_artifacts = {"transform", "model", "y_processing",
                                   "feature_augmentation", "concat_transform",
                                   "feature_selection"}

        for step in trace.steps:
            if step.execution_mode == StepExecutionMode.SKIP:
                continue

            # Check if this type typically needs artifacts
            if step.operator_type in steps_needing_artifacts:
                if not step.has_artifacts():
                    issues.append(
                        f"Step {step.step_index} ({step.operator_type}/{step.operator_class}) "
                        f"has no recorded artifacts"
                    )

        is_valid = len(issues) == 0
        return is_valid, issues
