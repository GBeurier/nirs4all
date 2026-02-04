"""Chain builder for converting ExecutionTrace to WorkspaceStore chain format.

Bridges the trace/artifact system and the DuckDB WorkspaceStore by converting
an ExecutionTrace (recorded during pipeline execution) into the chain dict
format expected by ``store.save_chain()``.

Example:
    >>> from nirs4all.pipeline.storage.chain_builder import ChainBuilder
    >>> chain_builder = ChainBuilder(trace, artifact_registry)
    >>> chain_data = chain_builder.build()
    >>> chain_id = store.save_chain(pipeline_id=pipeline_id, **chain_data)
"""

from typing import Any, Dict, List, Optional

from nirs4all.pipeline.trace.execution_trace import (
    ExecutionStep,
    ExecutionTrace,
    StepExecutionMode,
)


class ChainBuilder:
    """Converts an ExecutionTrace into the chain dict for WorkspaceStore.save_chain().

    The builder extracts the ordered sequence of non-skipped steps,
    identifies the model step, collects fold and shared artifact IDs,
    and produces a chain descriptor ready for DuckDB persistence.

    Args:
        trace: Finalized ExecutionTrace from TraceRecorder.
        artifact_registry: ArtifactRegistry that holds artifact records
            produced during this pipeline execution.  Used to map
            chain-path based artifact IDs to WorkspaceStore artifact IDs.

    Example:
        >>> builder = ChainBuilder(trace, artifact_registry)
        >>> chain_data = builder.build()
        >>> store.save_chain(pipeline_id=pid, **chain_data)
    """

    def __init__(
        self,
        trace: ExecutionTrace,
        artifact_registry: Any = None,
    ) -> None:
        self._trace = trace
        self._artifact_registry = artifact_registry

    def build(self) -> Dict[str, Any]:
        """Build the chain dict from the execution trace.

        Returns:
            Dictionary with keys matching ``WorkspaceStore.save_chain()``
            parameters: ``steps``, ``model_step_idx``, ``model_class``,
            ``preprocessings``, ``fold_strategy``, ``fold_artifacts``,
            ``shared_artifacts``, ``branch_path``, ``source_index``.
        """
        steps: List[Dict[str, Any]] = []
        fold_artifacts: Dict[str, str] = {}
        shared_artifacts: Dict[str, str] = {}
        model_class = ""
        model_step_idx = 0
        branch_path: Optional[List[int]] = None
        source_index: Optional[int] = None

        for step in self._trace.steps:
            if step.execution_mode == StepExecutionMode.SKIP:
                continue

            step_dict = {
                "step_idx": step.step_index,
                "operator_class": step.operator_class,
                "params": step.operator_config or {},
                "artifact_id": step.artifacts.primary_artifact_id,
                "stateless": not bool(step.artifacts.artifact_ids),
            }
            steps.append(step_dict)

            # Track branch_path from step context
            if step.branch_path:
                branch_path = step.branch_path

            # Collect artifacts
            self._collect_artifacts(step, fold_artifacts, shared_artifacts)

        # Identify model step
        if self._trace.model_step_index is not None:
            model_step_idx = self._trace.model_step_index
            model_step = self._find_step(model_step_idx)
            if model_step is not None:
                model_class = model_step.operator_class

        # Determine fold strategy
        fold_strategy = "per_fold" if fold_artifacts else "shared"

        # Preprocessing chain summary
        preprocessings = self._trace.preprocessing_chain or ""

        return {
            "steps": steps,
            "model_step_idx": model_step_idx,
            "model_class": model_class,
            "preprocessings": preprocessings,
            "fold_strategy": fold_strategy,
            "fold_artifacts": fold_artifacts,
            "shared_artifacts": shared_artifacts,
            "branch_path": branch_path,
            "source_index": source_index,
        }

    def _collect_artifacts(
        self,
        step: ExecutionStep,
        fold_artifacts: Dict[str, str],
        shared_artifacts: Dict[str, str],
    ) -> None:
        """Collect fold and shared artifact IDs from a step.

        Fold artifacts come from model steps with per-fold training.
        Shared artifacts come from transformers that are fitted once.

        Args:
            step: Execution step to inspect.
            fold_artifacts: Mutable dict to accumulate fold artifact IDs.
            shared_artifacts: Mutable dict to accumulate shared artifact IDs.
        """
        artifacts = step.artifacts
        step_idx_str = str(step.step_index)

        # Fold-specific artifacts (from CV model steps)
        if artifacts.fold_artifact_ids:
            for fold_id, artifact_id in artifacts.fold_artifact_ids.items():
                fold_key = f"fold_{fold_id}"
                fold_artifacts[fold_key] = artifact_id

        # Primary artifact as shared (non-fold) artifact
        if artifacts.primary_artifact_id and not artifacts.fold_artifact_ids:
            shared_artifacts[step_idx_str] = artifacts.primary_artifact_id

    def _find_step(self, step_index: int) -> Optional[ExecutionStep]:
        """Find a step by index in the trace.

        Args:
            step_index: Step index to search for.

        Returns:
            ExecutionStep or None if not found.
        """
        for step in self._trace.steps:
            if step.step_index == step_index:
                return step
        return None
