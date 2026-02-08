"""Chain builder for converting ExecutionTrace to WorkspaceStore chain format.

Bridges the trace/artifact system and the DuckDB WorkspaceStore by converting
an ExecutionTrace (recorded during pipeline execution) into the chain dict
format expected by ``store.save_chain()``.

When a pipeline has multiple model steps, the builder produces one chain
per model step so that each prediction can be resolved independently.

Example:
    >>> from nirs4all.pipeline.storage.chain_builder import ChainBuilder
    >>> chain_builder = ChainBuilder(trace, artifact_registry)
    >>> for chain_data in chain_builder.build_all():
    ...     store.save_chain(pipeline_id=pipeline_id, **chain_data)
"""

from typing import Any, Dict, List, Optional

from nirs4all.pipeline.trace.execution_trace import (
    ExecutionStep,
    ExecutionTrace,
    StepExecutionMode,
    normalize_fold_key,
)


class ChainBuilder:
    """Converts an ExecutionTrace into chain dicts for WorkspaceStore.save_chain().

    The builder extracts the ordered sequence of non-skipped steps,
    identifies model steps, collects fold and shared artifact IDs,
    and produces chain descriptors ready for DuckDB persistence.

    When the trace has multiple model steps, ``build_all()`` produces
    one chain per model.  ``build()`` returns the chain for the trace's
    primary ``model_step_index`` (backward-compatible).

    Args:
        trace: Finalized ExecutionTrace from TraceRecorder.
        artifact_registry: ArtifactRegistry that holds artifact records
            produced during this pipeline execution.

    Example:
        >>> builder = ChainBuilder(trace, artifact_registry)
        >>> for chain_data in builder.build_all():
        ...     store.save_chain(pipeline_id=pid, **chain_data)
    """

    def __init__(
        self,
        trace: ExecutionTrace,
        artifact_registry: Any = None,
    ) -> None:
        self._trace = trace
        self._artifact_registry = artifact_registry

    def build(self) -> Dict[str, Any]:
        """Build one chain dict for the trace's primary model step.

        Returns:
            Dictionary with keys matching ``WorkspaceStore.save_chain()``
            parameters.
        """
        chains = self.build_all()
        if not chains:
            return self._empty_chain()
        # Return the chain matching the trace's primary model_step_index
        if self._trace.model_step_index is not None:
            for chain in chains:
                if chain["model_step_idx"] == self._trace.model_step_index:
                    return chain
        return chains[-1]

    def build_all(self) -> List[Dict[str, Any]]:
        """Build one chain dict per model step found in the trace.

        Returns:
            List of chain dicts, one per model step.
        """
        # Collect all non-skipped steps
        all_steps: List[Dict[str, Any]] = []
        # Collect shared artifacts with their branch context for per-model filtering
        shared_artifact_entries: List[tuple[ExecutionStep, List[str]]] = []
        model_steps: List[ExecutionStep] = []
        branch_path: Optional[List[int]] = None

        for step in self._trace.steps:
            if step.execution_mode == StepExecutionMode.SKIP:
                continue

            step_dict = {
                "step_idx": step.step_index,
                "operator_class": step.operator_class,
                "params": step.operator_config or {},
                "artifact_ids": list(step.artifacts.artifact_ids),
                "stateless": not bool(step.artifacts.artifact_ids),
            }
            all_steps.append(step_dict)

            if step.branch_path:
                branch_path = step.branch_path

            # Is this a model step (has fold artifacts)?
            if step.artifacts.fold_artifact_ids:
                model_steps.append(step)
            elif step.artifacts.artifact_ids:
                shared_artifact_entries.append((step, list(step.artifacts.artifact_ids)))

        if not model_steps:
            # No model steps found — build a single chain with trace's hint
            shared_artifacts = self._collect_shared_artifacts(shared_artifact_entries)
            return [self._build_single_chain(all_steps, shared_artifacts, branch_path)]

        # Build one chain per model step
        preprocessings = self._trace.preprocessing_chain or ""
        chains: List[Dict[str, Any]] = []

        for model_step in model_steps:
            fold_artifacts: Dict[str, str] = {}
            for fold_id, artifact_id in model_step.artifacts.fold_artifact_ids.items():
                fold_artifacts[normalize_fold_key(fold_id)] = artifact_id

            # Include steps up to this model step
            chain_steps = [s for s in all_steps if s["step_idx"] <= model_step.step_index]

            # Build shared_artifacts for THIS model only, filtering by branch path
            shared_artifacts = self._collect_shared_artifacts(
                shared_artifact_entries, model_step.branch_path
            )

            chains.append({
                "steps": chain_steps,
                "model_step_idx": model_step.step_index,
                "model_class": model_step.operator_class,
                "preprocessings": preprocessings,
                "fold_strategy": "per_fold" if fold_artifacts else "shared",
                "fold_artifacts": fold_artifacts,
                "shared_artifacts": shared_artifacts,
                "branch_path": model_step.branch_path or branch_path,
                "source_index": None,
            })

        return chains

    @staticmethod
    def _is_branch_compatible(
        artifact_branch_path: Optional[List[int]],
        model_branch_path: Optional[List[int]],
    ) -> bool:
        """Check if an artifact's branch path is on the same execution path as a model.

        An artifact is compatible if:
        - It has no branch path (executed before any branching)
        - The model has no branch path (e.g., after merge) - include all artifacts
        - Its branch path is a prefix of or equal to the model's branch path
        """
        if artifact_branch_path is None:
            return True
        if not model_branch_path:  # None or empty list (e.g., post-merge model)
            return True
        n = len(artifact_branch_path)
        return artifact_branch_path == model_branch_path[:n]

    @staticmethod
    def _collect_shared_artifacts(
        entries: List[tuple["ExecutionStep", List[str]]],
        model_branch_path: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Collect shared artifacts, optionally filtering by branch compatibility.

        When a step has multi-source artifacts (``by_source`` with more than
        one source), a ``_source_map`` metadata key is added to the result so
        that the resolver can build a source-aware artifact provider.
        """
        shared: Dict[str, Any] = {}
        source_map: Dict[str, Dict[str, List[str]]] = {}
        for step, aids in entries:
            if model_branch_path is not None and not ChainBuilder._is_branch_compatible(step.branch_path, model_branch_path):
                continue
            key = str(step.step_index)
            if key not in shared:
                shared[key] = []
            shared[key].extend(aids)
            # Record per-source grouping for multi-source steps
            if step.artifacts.by_source and len(step.artifacts.by_source) > 1:
                source_map[key] = {
                    str(si): list(art_ids)
                    for si, art_ids in step.artifacts.by_source.items()
                }
        if source_map:
            shared["_source_map"] = source_map
        return shared

    def _build_single_chain(
        self,
        all_steps: List[Dict[str, Any]],
        shared_artifacts: Dict[str, List[str]],
        branch_path: Optional[List[int]],
    ) -> Dict[str, Any]:
        """Build a single chain when there are no model steps with fold artifacts."""
        model_class = ""
        model_step_idx = 0
        if self._trace.model_step_index is not None:
            model_step_idx = self._trace.model_step_index
            model_step = self._find_step(model_step_idx)
            if model_step is not None:
                model_class = model_step.operator_class

        return {
            "steps": all_steps,
            "model_step_idx": model_step_idx,
            "model_class": model_class,
            "preprocessings": self._trace.preprocessing_chain or "",
            "fold_strategy": "shared",
            "fold_artifacts": {},
            "shared_artifacts": shared_artifacts,
            "branch_path": branch_path,
            "source_index": None,
        }

    def _empty_chain(self) -> Dict[str, Any]:
        """Return an empty chain dict."""
        return {
            "steps": [],
            "model_step_idx": 0,
            "model_class": "",
            "preprocessings": "",
            "fold_strategy": "shared",
            "fold_artifacts": {},
            "shared_artifacts": {},
            "branch_path": None,
            "source_index": None,
        }

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
