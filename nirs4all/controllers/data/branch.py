"""
Unified Branch Controller for pipeline branching.

This controller enables two types of branching:

1. **Duplication branches** (original behavior):
   - All samples processed by all branches
   - Each branch applies different preprocessing
   - Used for comparing preprocessing strategies

2. **Separation branches** (new in v2):
   - Samples partitioned into non-overlapping subsets
   - Each sample processed by exactly one branch
   - Used for per-group models (by tag, metadata, source, or filter)

Syntax detection:
    - List syntax: `{"branch": [[A], [B]]}` -> duplication branches
    - Dict with by_* keys: `{"branch": {"by_tag": "outlier"}}` -> separation branches

Examples:
    >>> # Duplication branch: compare SNV vs MSC
    >>> pipeline = [
    ...     {"branch": [
    ...         [SNV(), PCA(n_components=10)],
    ...         [MSC(), FirstDerivative()],
    ...     ]},
    ...     PLSRegression(n_components=5),  # Runs on BOTH branches
    ... ]
    >>>
    >>> # Separation branch by tag: different models for outliers vs inliers
    >>> pipeline = [
    ...     {"tag": YOutlierFilter()},  # Creates 'y_outlier_iqr' tag
    ...     {"branch": {
    ...         "by_tag": "y_outlier_iqr",
    ...         "values": {
    ...             "outliers": True,
    ...             "inliers": False,
    ...         },
    ...         "steps": [PLS(n_components=5)],
    ...     }},
    ... ]
    >>>
    >>> # Separation branch by metadata: per-site models
    >>> pipeline = [
    ...     {"branch": {
    ...         "by_metadata": "site",
    ...         "steps": [SNV(), PLS(10)],
    ...     }},
    ...     {"merge": "concat"},  # Reassemble predictions
    ... ]
    >>>
    >>> # Separation branch by source: per-source preprocessing
    >>> pipeline = [
    ...     {"branch": {
    ...         "by_source": True,
    ...         "steps": {
    ...             "NIR": [SNV(), SavitzkyGolay()],
    ...             "markers": [VarianceThreshold()],
    ...         },
    ...     }},
    ...     {"merge": {"sources": "concat"}},
    ... ]
"""

import copy
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING

import numpy as np

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger
from nirs4all.pipeline.config.generator import (
    expand_spec,
    is_generator_node,
)
from nirs4all.pipeline.execution.result import StepOutput

logger = get_logger(__name__)

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
    from nirs4all.pipeline.steps.parser import ParsedStep


# Separation branch keywords (indicating separation mode)
SEPARATION_KEYWORDS = {"by_tag", "by_metadata", "by_filter", "by_source"}


@register_controller
class BranchController(OperatorController):
    """Unified controller for pipeline branching.

    Implements both duplication and separation branching mechanisms:

    **Duplication branches** (all samples to all branches):
        - Creates independent context copies for each branch
        - Each branch sees all samples
        - Executes branch steps sequentially within each branch
        - Post-branch steps iterate over all branch contexts

    **Separation branches** (samples partitioned):
        - Partitions samples into non-overlapping subsets
        - Each sample goes to exactly one branch
        - Supports by_tag, by_metadata, by_filter, by_source modes
        - Use {"merge": "concat"} to reassemble

    Attributes:
        priority: Controller priority (5 = high, executes early).
    """

    priority = 5

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Check if the step matches the branch controller.

        Args:
            step: Original step configuration
            operator: Deserialized operator
            keyword: Step keyword

        Returns:
            True if keyword is "branch" or "source_branch" (backward compatibility)
        """
        return keyword in ("branch", "source_branch")

    @classmethod
    def use_multi_source(cls) -> bool:
        """Branch controller supports multi-source datasets."""
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Branch controller executes in prediction mode to reconstruct branches."""
        return True

    def execute(
        self,
        step_info: "ParsedStep",
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple["ExecutionContext", StepOutput]:
        """Execute the branch step.

        Detects branch mode (duplication vs separation) and delegates
        to the appropriate execution method.

        Args:
            step_info: Parsed step containing branch definitions
            dataset: Dataset to operate on
            context: Pipeline execution context
            runtime_context: Runtime infrastructure context
            source: Data source index
            mode: Execution mode ("train" or "predict")
            loaded_binaries: Pre-loaded binary objects for prediction mode
            prediction_store: External prediction store for model predictions

        Returns:
            Tuple of (updated_context, StepOutput with collected artifacts)
        """
        # Handle backward compatibility: source_branch keyword -> by_source
        if "source_branch" in step_info.original_step:
            raw_def = self._convert_source_branch_syntax(step_info.original_step)
        else:
            raw_def = step_info.original_step.get("branch", {})

        branch_mode = self._detect_branch_mode(raw_def)

        if branch_mode == "separation":
            return self._execute_separation_branch(
                step_info=step_info,
                raw_def=raw_def,
                dataset=dataset,
                context=context,
                runtime_context=runtime_context,
                source=source,
                mode=mode,
                loaded_binaries=loaded_binaries,
                prediction_store=prediction_store,
            )
        else:
            return self._execute_duplication_branch(
                step_info=step_info,
                dataset=dataset,
                context=context,
                runtime_context=runtime_context,
                source=source,
                mode=mode,
                loaded_binaries=loaded_binaries,
                prediction_store=prediction_store,
            )

    def _convert_source_branch_syntax(self, original_step: Dict[str, Any]) -> Dict[str, Any]:
        """Convert legacy source_branch syntax to new by_source syntax.

        Old syntax:
            {"source_branch": {"NIR": [steps], "markers": [steps]}}
            {"source_branch": "auto"}

        New syntax:
            {"by_source": True, "steps": {"NIR": [steps], "markers": [steps]}}

        Args:
            original_step: Original step containing source_branch key

        Returns:
            Converted raw_def dict suitable for by_source processing
        """
        source_branch_def = original_step.get("source_branch", {})

        logger.warning(
            "The 'source_branch' keyword is deprecated. "
            "Use {'branch': {'by_source': True, 'steps': {...}}} instead. "
            "See migration guide for details."
        )

        if source_branch_def == "auto" or source_branch_def is True:
            return {"by_source": True, "steps": {}}

        if isinstance(source_branch_def, dict):
            # Extract special keys
            steps = {}
            for key, value in source_branch_def.items():
                if not key.startswith("_"):
                    steps[key] = value

            return {"by_source": True, "steps": steps}

        if isinstance(source_branch_def, list):
            # List indexed by source position
            steps = {str(i): v for i, v in enumerate(source_branch_def)}
            return {"by_source": True, "steps": steps}

        # Default
        return {"by_source": True, "steps": {}}

    def _detect_branch_mode(self, raw_def: Any) -> str:
        """Detect whether this is a duplication or separation branch.

        Args:
            raw_def: Raw branch definition from step

        Returns:
            "duplication" or "separation"
        """
        # List syntax is always duplication
        if isinstance(raw_def, list):
            return "duplication"

        # Check for separation keywords
        if isinstance(raw_def, dict):
            for key in SEPARATION_KEYWORDS:
                if key in raw_def:
                    return "separation"

            # Check for legacy "by" key patterns (for backward compatibility during migration)
            by_value = raw_def.get("by")
            if by_value in ("outlier_excluder", "sample_partitioner", "metadata_partitioner"):
                # These are handled by their own controllers (to be deleted in Phase 4)
                # but if encountered here, treat as error
                raise ValueError(
                    f"Legacy branch syntax 'by: {by_value}' is no longer supported. "
                    f"Use the new unified branch syntax instead. "
                    f"See migration guide for details."
                )

        # Default to duplication (dict with named branches, generator syntax, etc.)
        return "duplication"

    # =========================================================================
    # Duplication Branch Execution (Original Behavior)
    # =========================================================================

    def _execute_duplication_branch(
        self,
        step_info: "ParsedStep",
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple["ExecutionContext", StepOutput]:
        """Execute duplication branch (all samples to all branches).

        This is the original branch behavior where each branch receives
        all samples and applies different preprocessing.
        """
        # Get branch definitions from step
        branch_defs = self._parse_branch_definitions(step_info)

        if not branch_defs:
            logger.warning("No branch definitions found, skipping branch step")
            return context, StepOutput()

        n_branches = len(branch_defs)

        # V3: Start branch step recording in trace
        recorder = runtime_context.trace_recorder
        if recorder is not None:
            recorder.start_branch_step(
                step_index=runtime_context.step_number,
                branch_count=n_branches,
                operator_config={"branch_definitions": len(branch_defs), "mode": "duplication"},
            )

        # In predict/explain mode, filter to only the target branch
        target_branch_id = None
        target_branch_name = None
        if mode in ("predict", "explain") and hasattr(runtime_context, 'target_model') and runtime_context.target_model:
            target_branch_id = runtime_context.target_model.get("branch_id")
            target_branch_name = runtime_context.target_model.get("branch_name")

            if target_branch_id is not None:
                if target_branch_id < len(branch_defs):
                    branch_defs = [branch_defs[target_branch_id]]
                    logger.info(f"Predict mode: executing only branch {target_branch_id} ({target_branch_name or 'unnamed'})")
                else:
                    raise ValueError(
                        f"Target branch_id={target_branch_id} not found in pipeline. "
                        f"Pipeline has {n_branches} branches (0-{n_branches-1}). "
                        f"The model may have been trained with a different branch configuration."
                    )
            else:
                logger.info(f"Creating {n_branches} duplication branches (predict mode, no target branch specified)")
        else:
            logger.info(f"Creating {n_branches} duplication branches")

        # Store the initial context as a snapshot point
        initial_context = context.copy()
        initial_processing = copy.deepcopy(context.selector.processing)

        # Snapshot the dataset's feature state before branching
        initial_features_snapshot = self._snapshot_features(dataset)

        # V3: Snapshot the chain state before branching
        initial_chain = recorder.current_chain() if recorder else None

        # Initialize list to collect branch contexts
        branch_contexts: List[Dict[str, Any]] = []
        all_artifacts = []

        # Execute each branch
        for idx, branch_def in enumerate(branch_defs):
            # Use original branch_id if we're in predict mode and have filtered
            if target_branch_id is not None:
                branch_id = target_branch_id
            else:
                branch_id = idx

            branch_name = branch_def.get("name", f"branch_{branch_id}")
            branch_steps = branch_def.get("steps", [])

            logger.info(f"  Branch {branch_id}: {branch_name}")

            # V3: Enter branch context in trace recorder
            if recorder is not None:
                recorder.enter_branch(branch_id)
                if initial_chain is not None:
                    recorder.reset_chain_to(initial_chain)

            # Create isolated context for this branch
            branch_context = initial_context.copy()

            # Build branch_path by appending to parent's branch_path
            parent_branch_path = context.selector.branch_path or []
            new_branch_path = parent_branch_path + [branch_id]

            branch_context.selector = branch_context.selector.with_branch(
                branch_id=branch_id,
                branch_name=branch_name,
                branch_path=new_branch_path
            )

            # Reset processing to initial state for this branch
            branch_context.selector.processing = copy.deepcopy(initial_processing)

            # Restore dataset features to initial state for this branch
            self._restore_features(dataset, initial_features_snapshot)

            # Reset artifact load counter for this branch
            if runtime_context:
                runtime_context.artifact_load_counter = {}

            # In predict/explain mode, load branch-specific binaries
            branch_binaries = loaded_binaries
            if mode in ("predict", "explain") and runtime_context.artifact_loader:
                branch_binaries = runtime_context.artifact_loader.get_step_binaries(
                    runtime_context.step_number, branch_id=branch_id
                )
                if not branch_binaries:
                    branch_binaries = loaded_binaries

            # Execute branch steps sequentially
            for substep_idx, substep in enumerate(branch_steps):
                if runtime_context.step_runner:
                    runtime_context.substep_number = substep_idx

                    # Record substep in trace before execution
                    if recorder is not None:
                        op_type, op_class = self._extract_substep_info(substep)
                        recorder.start_branch_substep(
                            parent_step_index=runtime_context.step_number,
                            branch_id=branch_id,
                            operator_type=op_type,
                            operator_class=op_class,
                            substep_index=substep_idx,
                            branch_name=branch_name,
                        )

                        self._record_dataset_shapes(
                            dataset, branch_context, runtime_context, is_input=True
                        )

                    result = runtime_context.step_runner.execute(
                        step=substep,
                        dataset=dataset,
                        context=branch_context,
                        runtime_context=runtime_context,
                        loaded_binaries=branch_binaries,
                        prediction_store=prediction_store
                    )

                    if recorder is not None:
                        self._record_dataset_shapes(
                            dataset, result.updated_context, runtime_context, is_input=False
                        )
                        is_model = op_type in ("model", "meta_model")
                        recorder.end_step(is_model=is_model)

                    branch_context = result.updated_context
                    all_artifacts.extend(result.artifacts)

            # V3: Snapshot the chain state BEFORE exiting branch context
            branch_chain_snapshot = recorder.current_chain() if recorder else None

            # V3: Exit branch context in trace recorder
            if recorder is not None:
                recorder.exit_branch()

            # Snapshot features AFTER branch processing completes
            branch_features_snapshot = self._snapshot_features(dataset)

            # Store the final context for this branch
            branch_contexts.append({
                "branch_id": branch_id,
                "name": branch_name,
                "context": branch_context,
                "generator_choice": branch_def.get("generator_choice"),
                "features_snapshot": branch_features_snapshot,
                "chain_snapshot": branch_chain_snapshot,
                "branch_mode": "duplication",
            })

            logger.success(f"  Branch {branch_id} ({branch_name}) completed")

        # V3: End branch step in trace
        if recorder is not None:
            recorder.end_step()

        # Store branch contexts in custom dict for post-branch iteration
        existing_branches = context.custom.get("branch_contexts", [])
        if existing_branches:
            new_branch_contexts = self._multiply_branch_contexts(
                existing_branches, branch_contexts
            )
        else:
            new_branch_contexts = branch_contexts

        # Update context with branch contexts
        result_context = context.copy()
        result_context.custom["branch_contexts"] = new_branch_contexts
        result_context.custom["in_branch_mode"] = True
        result_context.custom["branch_type"] = "duplication"

        # Collect generator choices from branches for serialization
        branch_generator_choices = [
            {"branch": branch_def.get("generator_choice")}
            for branch_def in branch_defs
            if branch_def.get("generator_choice") is not None
        ]

        logger.success(f"Duplication branch step completed with {len(new_branch_contexts)} branch(es)")

        return result_context, StepOutput(
            artifacts=all_artifacts,
            metadata={
                "branch_count": len(new_branch_contexts),
                "branch_mode": "duplication",
                "branch_generator_choices": branch_generator_choices
            }
        )

    # =========================================================================
    # Separation Branch Execution (New in v2)
    # =========================================================================

    def _execute_separation_branch(
        self,
        step_info: "ParsedStep",
        raw_def: Dict[str, Any],
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple["ExecutionContext", StepOutput]:
        """Execute separation branch (samples partitioned).

        Dispatches to the appropriate handler based on separation type.
        """
        if "by_tag" in raw_def:
            return self._execute_by_tag(
                raw_def, dataset, context, runtime_context,
                source, mode, loaded_binaries, prediction_store
            )
        elif "by_metadata" in raw_def:
            return self._execute_by_metadata(
                raw_def, dataset, context, runtime_context,
                source, mode, loaded_binaries, prediction_store
            )
        elif "by_filter" in raw_def:
            return self._execute_by_filter(
                raw_def, dataset, context, runtime_context,
                source, mode, loaded_binaries, prediction_store
            )
        elif "by_source" in raw_def:
            return self._execute_by_source(
                raw_def, dataset, context, runtime_context,
                source, mode, loaded_binaries, prediction_store
            )
        else:
            raise ValueError(
                f"Unknown separation branch type. "
                f"Expected one of: by_tag, by_metadata, by_filter, by_source"
            )

    def _execute_by_tag(
        self,
        raw_def: Dict[str, Any],
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple["ExecutionContext", StepOutput]:
        """Execute separation branch by tag values.

        Syntax:
            {"branch": {
                "by_tag": "tag_name",
                "values": {"branch1": True, "branch2": False},  # Optional
                "steps": [...],  # Shared steps for all branches
            }}
        """
        from nirs4all.controllers.data.branch_utils import (
            parse_value_condition,
            group_samples_by_value_mapping,
        )

        tag_name = raw_def["by_tag"]
        value_mapping = raw_def.get("values")
        steps = raw_def.get("steps", [])

        # Check if tag exists
        if not dataset.has_tag(tag_name):
            available_tags = dataset.tags
            raise ValueError(
                f"Tag '{tag_name}' not found in dataset. "
                f"Available tags: {available_tags}. "
                f"Ensure the tag is created with {{'tag': Filter()}} before branching."
            )

        logger.info(f"Creating separation branches by tag '{tag_name}'")

        # Get tag values for all samples (training partition in train mode)
        selector = context.selector.copy()
        if mode == "train":
            selector = context.with_partition("train").selector
            selector.include_augmented = False

        sample_indices = dataset._indexer.x_indices(
            selector, include_augmented=False, include_excluded=False
        )

        if len(sample_indices) == 0:
            logger.warning("No samples found for separation branch")
            return context, StepOutput()

        # Get tag values for these samples
        tag_values = dataset.get_tag(tag_name, selector={"indices": sample_indices.tolist()})

        # Build value mapping if not provided
        if value_mapping is None:
            # Auto-detect unique values
            unique_values = sorted(set(tag_values))
            value_mapping = {str(v): v for v in unique_values}
            logger.info(f"  Auto-detected {len(unique_values)} unique tag values: {unique_values}")

        # Group samples by value mapping
        groups = group_samples_by_value_mapping(tag_values.tolist(), value_mapping)

        return self._execute_separation_branches(
            groups=groups,
            sample_indices=sample_indices,
            steps=steps,
            dataset=dataset,
            context=context,
            runtime_context=runtime_context,
            source=source,
            mode=mode,
            loaded_binaries=loaded_binaries,
            prediction_store=prediction_store,
            separation_type="by_tag",
            separation_key=tag_name,
        )

    def _execute_by_metadata(
        self,
        raw_def: Dict[str, Any],
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple["ExecutionContext", StepOutput]:
        """Execute separation branch by metadata column.

        Syntax:
            {"branch": {
                "by_metadata": "column_name",
                "values": {"branch1": ["A", "B"], "branch2": ["C"]},  # Optional grouping
                "steps": [...],  # Shared steps for all branches
                "min_samples": 10,  # Optional minimum samples per branch
            }}
        """
        from nirs4all.controllers.data.branch_utils import (
            parse_value_condition,
            group_samples_by_value_mapping,
        )

        column = raw_def["by_metadata"]
        value_mapping = raw_def.get("values")
        steps = raw_def.get("steps", [])
        min_samples = raw_def.get("min_samples", 1)

        # Check if metadata column exists
        metadata = dataset.metadata
        if metadata is None or column not in metadata.columns:
            available_cols = list(metadata.columns) if metadata is not None else []
            raise ValueError(
                f"Metadata column '{column}' not found. "
                f"Available columns: {available_cols}"
            )

        logger.info(f"Creating separation branches by metadata column '{column}'")

        # Get sample indices
        selector = context.selector.copy()
        if mode == "train":
            selector = context.with_partition("train").selector
            selector.include_augmented = False

        sample_indices = dataset._indexer.x_indices(
            selector, include_augmented=False, include_excluded=False
        )

        if len(sample_indices) == 0:
            logger.warning("No samples found for separation branch")
            return context, StepOutput()

        # Get metadata values for these samples
        column_values = metadata[column].values[sample_indices]

        # Build value mapping if not provided
        if value_mapping is None:
            unique_values = sorted(set(column_values))
            value_mapping = {str(v): [v] for v in unique_values}
            logger.info(f"  Auto-detected {len(unique_values)} unique values: {unique_values[:10]}{'...' if len(unique_values) > 10 else ''}")

        # Group samples - for metadata, values in mapping are lists of allowed values
        groups: Dict[str, List[int]] = {}
        for branch_name, allowed_values in value_mapping.items():
            if not isinstance(allowed_values, (list, tuple, set)):
                allowed_values = [allowed_values]
            allowed_set = set(allowed_values)

            indices = [
                i for i, val in enumerate(column_values)
                if val in allowed_set
            ]
            groups[branch_name] = indices

        # Filter out branches with too few samples
        skipped = []
        for branch_name in list(groups.keys()):
            if len(groups[branch_name]) < min_samples:
                skipped.append((branch_name, len(groups[branch_name])))
                del groups[branch_name]

        if skipped:
            logger.warning(f"  Skipped {len(skipped)} branches below min_samples={min_samples}: {skipped}")

        return self._execute_separation_branches(
            groups=groups,
            sample_indices=sample_indices,
            steps=steps,
            dataset=dataset,
            context=context,
            runtime_context=runtime_context,
            source=source,
            mode=mode,
            loaded_binaries=loaded_binaries,
            prediction_store=prediction_store,
            separation_type="by_metadata",
            separation_key=column,
        )

    def _execute_by_filter(
        self,
        raw_def: Dict[str, Any],
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple["ExecutionContext", StepOutput]:
        """Execute separation branch by filter result.

        Creates two branches: one for samples passing the filter, one for failing.

        Syntax:
            {"branch": {
                "by_filter": FilterInstance(),  # or serialized filter dict
                "steps": [...],  # Shared steps for all branches
                "names": ["passing", "failing"],  # Optional branch names
            }}
        """
        from nirs4all.operators.filters.base import SampleFilter
        from nirs4all.pipeline.steps.deserializer import StepDeserializer

        filter_spec = raw_def["by_filter"]
        steps = raw_def.get("steps", [])
        names = raw_def.get("names", ["passing", "failing"])

        # Deserialize filter if needed
        if isinstance(filter_spec, dict):
            deserializer = StepDeserializer()
            filter_obj = deserializer.deserialize(filter_spec)
        elif isinstance(filter_spec, SampleFilter):
            filter_obj = filter_spec
        else:
            raise ValueError(
                f"by_filter expects a SampleFilter instance or dict, got {type(filter_spec).__name__}"
            )

        logger.info(f"Creating separation branches by filter {filter_obj.__class__.__name__}")

        # Get sample indices
        selector = context.selector.copy()
        if mode == "train":
            selector = context.with_partition("train").selector
            selector.include_augmented = False

        sample_indices = dataset._indexer.x_indices(
            selector, include_augmented=False, include_excluded=False
        )

        if len(sample_indices) == 0:
            logger.warning("No samples found for separation branch")
            return context, StepOutput()

        # Get X and y for filter
        X = dataset.x(selector, layout="2d", concat_source=True, include_augmented=False, include_excluded=False)
        y = dataset.y(selector, include_augmented=False, include_excluded=False)

        # Fit and apply filter
        filter_obj.fit(X, y)
        mask = filter_obj.get_mask(X, y)  # True = passing, False = failing

        # Create groups
        passing_indices = [i for i, m in enumerate(mask) if m]
        failing_indices = [i for i, m in enumerate(mask) if not m]

        groups = {
            names[0]: passing_indices,
            names[1]: failing_indices,
        }

        logger.info(f"  Filter result: {len(passing_indices)} passing, {len(failing_indices)} failing")

        # Persist filter for prediction mode
        all_artifacts = []
        if mode == "train":
            artifact = (filter_obj, f"branch_filter_{runtime_context.next_op()}", "sklearn")
            all_artifacts.append(artifact)

        result_context, output = self._execute_separation_branches(
            groups=groups,
            sample_indices=sample_indices,
            steps=steps,
            dataset=dataset,
            context=context,
            runtime_context=runtime_context,
            source=source,
            mode=mode,
            loaded_binaries=loaded_binaries,
            prediction_store=prediction_store,
            separation_type="by_filter",
            separation_key=filter_obj.__class__.__name__,
        )

        # Merge artifacts
        output.artifacts.extend(all_artifacts)
        return result_context, output

    def _execute_by_source(
        self,
        raw_def: Dict[str, Any],
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple["ExecutionContext", StepOutput]:
        """Execute per-source branching.

        This absorbs the functionality of SourceBranchController.

        Syntax:
            {"branch": {
                "by_source": True,  # or "auto"
                "steps": {  # Per-source steps
                    "NIR": [SNV(), SavitzkyGolay()],
                    "markers": [VarianceThreshold()],
                },
            }}

            # Or with shared steps
            {"branch": {
                "by_source": True,
                "steps": [MinMaxScaler()],  # Applied to all sources
            }}
        """
        steps = raw_def.get("steps", {})

        # Validate multi-source dataset
        n_sources = dataset.n_sources
        if n_sources == 0:
            raise ValueError(
                "by_source requires a dataset with feature sources. "
                "No sources found in dataset."
            )

        if n_sources == 1:
            logger.warning(
                "by_source called on single-source dataset. "
                "This is effectively a no-op. Consider removing this step."
            )

        # Get source names
        source_names = self._get_source_names(dataset, n_sources)
        logger.info(f"Creating separation branches by source: {n_sources} sources")

        # Parse steps configuration
        if isinstance(steps, dict):
            # Per-source steps: {"NIR": [...], "markers": [...]}
            source_steps = steps
        elif isinstance(steps, list):
            # Shared steps for all sources
            source_steps = {name: steps for name in source_names}
        else:
            source_steps = {name: [] for name in source_names}

        # V3: Start branch step recording in trace
        recorder = runtime_context.trace_recorder
        if recorder is not None:
            recorder.start_branch_step(
                step_index=runtime_context.step_number,
                branch_count=n_sources,
                operator_config={"by_source": True, "n_sources": n_sources, "mode": "separation"},
            )

        # Store initial state
        initial_context = context.copy()
        initial_processing = copy.deepcopy(context.selector.processing)

        # Get current processing chains for all sources
        current_processing = list(context.selector.processing) if context.selector.processing else []
        while len(current_processing) < n_sources:
            src_idx = len(current_processing)
            current_processing.append(dataset.features_processings(src_idx))

        # Track new processing chains per source
        new_processing_per_source: List[List[str]] = [list(p) for p in current_processing]

        # Execute per-source pipelines
        source_contexts: List[Dict[str, Any]] = []
        all_artifacts = []

        for src_idx, src_name in enumerate(source_names):
            branch_steps = source_steps.get(src_name, [])

            logger.info(f"  Source '{src_name}' (index {src_idx}): {self._get_step_names(branch_steps) or '[passthrough]'}")

            # V3: Enter source context in trace recorder
            if recorder is not None:
                recorder.enter_branch(src_idx)

            # Create context with processing for only this source
            source_specific_processing = []
            for i in range(n_sources):
                if i == src_idx:
                    source_specific_processing.append(list(current_processing[i]))
                else:
                    source_specific_processing.append([])  # Empty = skip this source

            source_context = initial_context.copy()
            source_context = source_context.with_processing(source_specific_processing)
            source_context.custom["_current_source_idx"] = src_idx
            source_context.custom["_current_source_name"] = src_name

            # Set branch info
            parent_branch_path = context.selector.branch_path or []
            new_branch_path = parent_branch_path + [src_idx]
            source_context.selector = source_context.selector.with_branch(
                branch_id=src_idx,
                branch_name=src_name,
                branch_path=new_branch_path
            )

            # Get source-specific binaries for prediction mode
            source_binaries = loaded_binaries
            if mode in ("predict", "explain") and runtime_context.artifact_loader:
                source_binaries = runtime_context.artifact_loader.get_step_binaries(
                    runtime_context.step_number, branch_id=src_idx
                )
                if not source_binaries:
                    source_binaries = loaded_binaries

            # Execute source pipeline steps
            for substep_idx, substep in enumerate(branch_steps):
                if runtime_context.step_runner:
                    runtime_context.substep_number = substep_idx
                    result = runtime_context.step_runner.execute(
                        step=substep,
                        dataset=dataset,
                        context=source_context,
                        runtime_context=runtime_context,
                        loaded_binaries=source_binaries,
                        prediction_store=prediction_store
                    )
                    source_context = result.updated_context
                    all_artifacts.extend(result.artifacts)

            # V3: Exit source context in trace recorder
            if recorder is not None:
                recorder.exit_branch()

            # Update new processing for this source from the context
            if source_context.selector.processing and len(source_context.selector.processing) > src_idx:
                new_processing_per_source[src_idx] = list(source_context.selector.processing[src_idx])

            # Store the source context
            source_contexts.append({
                "branch_id": src_idx,
                "source_id": src_idx,
                "name": src_name,
                "source_name": src_name,
                "context": source_context,
                "features_snapshot": None,
                "pipeline_steps": branch_steps,
                "branch_mode": "separation",
            })

            logger.success(f"  Source '{src_name}' processing completed")

        # V3: End branch step in trace
        if recorder is not None:
            recorder.end_step()

        # Build updated context with combined processing from all sources
        result_context = context.copy()
        result_context = result_context.with_processing(new_processing_per_source)

        # Store source contexts for later merge operations
        result_context.custom["source_branch_contexts"] = source_contexts
        result_context.custom["branch_contexts"] = source_contexts
        result_context.custom["in_source_branch_mode"] = True
        result_context.custom["in_branch_mode"] = True
        result_context.custom["branch_type"] = "separation"
        result_context.custom["separation_type"] = "by_source"

        # Build metadata
        metadata = {
            "branch_count": n_sources,
            "branch_mode": "separation",
            "separation_type": "by_source",
            "n_sources": n_sources,
            "source_names": source_names,
        }

        logger.success(f"Source branch step completed: {n_sources} sources processed")

        return result_context, StepOutput(
            artifacts=all_artifacts,
            metadata=metadata
        )

    def _execute_separation_branches(
        self,
        groups: Dict[str, List[int]],
        sample_indices: np.ndarray,
        steps: List[Any],
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None,
        separation_type: str = "unknown",
        separation_key: str = "",
    ) -> Tuple["ExecutionContext", StepOutput]:
        """Common execution logic for separation branches.

        Args:
            groups: Dict mapping branch names to lists of local indices (into sample_indices)
            sample_indices: Array of actual sample indices in the dataset
            steps: Pipeline steps to execute in each branch
            ... (other args same as execute)
        """
        n_branches = len(groups)
        if n_branches == 0:
            logger.warning("No branches after grouping, skipping separation branch")
            return context, StepOutput()

        # V3: Start branch step recording in trace
        recorder = runtime_context.trace_recorder
        if recorder is not None:
            recorder.start_branch_step(
                step_index=runtime_context.step_number,
                branch_count=n_branches,
                operator_config={
                    "separation_type": separation_type,
                    "separation_key": separation_key,
                    "mode": "separation",
                },
            )

        logger.info(f"  Creating {n_branches} separation branches ({separation_type})")

        # Store initial context as snapshot
        initial_context = context.copy()
        initial_processing = copy.deepcopy(context.selector.processing)
        initial_features_snapshot = self._snapshot_features(dataset)

        # V3: Snapshot chain state
        initial_chain = recorder.current_chain() if recorder else None

        # In predict/explain mode, filter to target branch if specified
        target_branch_id = None
        if mode in ("predict", "explain") and hasattr(runtime_context, 'target_model') and runtime_context.target_model:
            target_branch_id = runtime_context.target_model.get("branch_id")

        # Process each group/branch
        branch_contexts: List[Dict[str, Any]] = []
        all_artifacts = []

        for branch_id, (branch_name, local_indices) in enumerate(groups.items()):
            # Skip if not target branch in predict mode
            if target_branch_id is not None and branch_id != target_branch_id:
                continue

            # Convert local indices to actual sample indices
            branch_sample_indices = sample_indices[local_indices] if local_indices else np.array([], dtype=int)

            logger.info(f"    Branch {branch_id} '{branch_name}': {len(branch_sample_indices)} samples")

            # V3: Enter branch context
            if recorder is not None:
                recorder.enter_branch(branch_id)
                if initial_chain is not None:
                    recorder.reset_chain_to(initial_chain)

            # Restore features to initial state
            self._restore_features(dataset, initial_features_snapshot)

            # Create isolated context for this branch
            branch_context = initial_context.copy()

            # Build branch_path
            parent_branch_path = context.selector.branch_path or []
            new_branch_path = parent_branch_path + [branch_id]

            branch_context.selector = branch_context.selector.with_branch(
                branch_id=branch_id,
                branch_name=branch_name,
                branch_path=new_branch_path
            )
            branch_context.selector.processing = copy.deepcopy(initial_processing)

            # Store sample partition info for downstream controllers
            branch_context.custom["sample_partition"] = {
                "sample_indices": branch_sample_indices.tolist() if isinstance(branch_sample_indices, np.ndarray) else list(branch_sample_indices),
                "n_samples": len(branch_sample_indices),
                "separation_type": separation_type,
                "separation_key": separation_key,
            }

            # Reset artifact counter
            if runtime_context:
                runtime_context.artifact_load_counter = {}

            # Get branch-specific binaries
            branch_binaries = loaded_binaries
            if mode in ("predict", "explain") and runtime_context.artifact_loader:
                branch_binaries = runtime_context.artifact_loader.get_step_binaries(
                    runtime_context.step_number, branch_id=branch_id
                )
                if not branch_binaries:
                    branch_binaries = loaded_binaries

            # Execute branch steps
            for substep_idx, substep in enumerate(steps):
                if runtime_context.step_runner:
                    runtime_context.substep_number = substep_idx

                    if recorder is not None:
                        op_type, op_class = self._extract_substep_info(substep)
                        recorder.start_branch_substep(
                            parent_step_index=runtime_context.step_number,
                            branch_id=branch_id,
                            operator_type=op_type,
                            operator_class=op_class,
                            substep_index=substep_idx,
                            branch_name=branch_name,
                        )

                    result = runtime_context.step_runner.execute(
                        step=substep,
                        dataset=dataset,
                        context=branch_context,
                        runtime_context=runtime_context,
                        loaded_binaries=branch_binaries,
                        prediction_store=prediction_store
                    )

                    if recorder is not None:
                        is_model = op_type in ("model", "meta_model")
                        recorder.end_step(is_model=is_model)

                    branch_context = result.updated_context
                    all_artifacts.extend(result.artifacts)

            # V3: Snapshot chain and exit branch
            branch_chain_snapshot = recorder.current_chain() if recorder else None
            if recorder is not None:
                recorder.exit_branch()

            # Snapshot features after processing
            branch_features_snapshot = self._snapshot_features(dataset)

            # Store branch context
            branch_contexts.append({
                "branch_id": branch_id,
                "name": branch_name,
                "context": branch_context,
                "features_snapshot": branch_features_snapshot,
                "chain_snapshot": branch_chain_snapshot,
                "branch_mode": "separation",
                "sample_indices": branch_sample_indices.tolist() if isinstance(branch_sample_indices, np.ndarray) else list(branch_sample_indices),
            })

            logger.success(f"    Branch {branch_id} '{branch_name}' completed")

        # V3: End branch step
        if recorder is not None:
            recorder.end_step()

        # Handle nested branching
        existing_branches = context.custom.get("branch_contexts", [])
        if existing_branches:
            new_branch_contexts = self._multiply_branch_contexts(
                existing_branches, branch_contexts
            )
        else:
            new_branch_contexts = branch_contexts

        # Update result context
        result_context = context.copy()
        result_context.custom["branch_contexts"] = new_branch_contexts
        result_context.custom["in_branch_mode"] = True
        result_context.custom["branch_type"] = "separation"
        result_context.custom["separation_type"] = separation_type

        logger.success(f"Separation branch step completed with {len(new_branch_contexts)} branch(es)")

        return result_context, StepOutput(
            artifacts=all_artifacts,
            metadata={
                "branch_count": len(new_branch_contexts),
                "branch_mode": "separation",
                "separation_type": separation_type,
                "separation_key": separation_key,
            }
        )

    # =========================================================================
    # Branch Definition Parsing (for duplication branches)
    # =========================================================================

    def _parse_branch_definitions(
        self,
        step_info: "ParsedStep"
    ) -> List[Dict[str, Any]]:
        """Parse branch definitions from step configuration.

        Supports multiple syntaxes:
            - List of lists: [[step1, step2], [step3]]
            - Dict with names: {"snv_pca": [SNV(), PCA()], "msc": [MSC()]}
            - List of dicts: [{"name": "a", "steps": [...]}, ...]
            - Generator syntax: {"_or_": [SNV(), MSC()]} or {"_range_": [...]}

        Args:
            step_info: Parsed step containing branch definitions

        Returns:
            Normalized list of branch definitions with 'name' and 'steps'
        """
        raw_def = step_info.original_step.get("branch", [])

        if not raw_def:
            return []

        # Case 0: Generator syntax - expand before processing
        if isinstance(raw_def, dict) and is_generator_node(raw_def):
            return self._expand_generator_branches(raw_def)

        # Case 1: Dict with named branches {"name": [steps], ...}
        if isinstance(raw_def, dict):
            # Skip separation keywords (handled separately)
            if any(k in raw_def for k in SEPARATION_KEYWORDS):
                return []

            # Check if any value contains generator syntax
            expanded_branches = []
            for name, steps in raw_def.items():
                if name.startswith("_"):  # Skip internal keys
                    continue
                if isinstance(steps, dict) and is_generator_node(steps):
                    expanded_steps = expand_spec(steps)
                    for i, exp_step in enumerate(expanded_steps):
                        branch_name = f"{name}_{self._generate_step_name(exp_step, i)}"
                        expanded_branches.append({
                            "name": branch_name,
                            "steps": exp_step if isinstance(exp_step, list) else [exp_step],
                            "generator_choice": exp_step
                        })
                else:
                    expanded_branches.append({
                        "name": name,
                        "steps": steps if isinstance(steps, list) else [steps]
                    })
            return expanded_branches

        # Case 2: List format
        if isinstance(raw_def, list):
            result = []
            for i, item in enumerate(raw_def):
                # Sub-case 2a: Dict with explicit name and steps
                if isinstance(item, dict) and "steps" in item:
                    steps = item["steps"]
                    if isinstance(steps, dict) and is_generator_node(steps):
                        expanded_steps = expand_spec(steps)
                        for j, exp_step in enumerate(expanded_steps):
                            result.append({
                                "name": f"{item.get('name', f'branch_{i}')}_{j}",
                                "steps": exp_step if isinstance(exp_step, list) else [exp_step],
                                "generator_choice": exp_step
                            })
                    else:
                        result.append({
                            "name": item.get("name", f"branch_{i}"),
                            "steps": steps
                        })
                # Sub-case 2b: Dict with generator syntax (inside list)
                elif isinstance(item, dict) and is_generator_node(item):
                    expanded = expand_spec(item)
                    for j, exp_item in enumerate(expanded):
                        branch_name = self._generate_step_name(exp_item, i * 100 + j)
                        result.append({
                            "name": branch_name,
                            "steps": exp_item if isinstance(exp_item, list) else [exp_item],
                            "generator_choice": exp_item
                        })
                # Sub-case 2c: List of steps (anonymous branch)
                elif isinstance(item, list):
                    expanded_list = self._expand_list_with_generators(item)
                    if len(expanded_list) > 1:
                        for j, exp_item in enumerate(expanded_list):
                            result.append({
                                "name": f"branch_{i}_{j}",
                                "steps": exp_item,
                                "generator_choice": exp_item
                            })
                    else:
                        result.append({
                            "name": f"branch_{i}",
                            "steps": expanded_list[0] if expanded_list else item
                        })
                # Sub-case 2d: Single step (wrap in list)
                else:
                    result.append({
                        "name": f"branch_{i}",
                        "steps": [item]
                    })
            return result

        # Fallback: treat as single branch with single step
        return [{"name": "branch_0", "steps": [raw_def]}]

    def _expand_generator_branches(
        self,
        generator_node: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Expand a generator node into branch definitions."""
        expanded = expand_spec(generator_node)

        result = []
        for i, item in enumerate(expanded):
            branch_name = self._generate_step_name(item, i)
            if isinstance(item, list):
                steps = item
            else:
                steps = [item]

            result.append({
                "name": branch_name,
                "steps": steps,
                "generator_choice": item
            })

        return result

    def _expand_list_with_generators(
        self,
        items: List[Any]
    ) -> List[List[Any]]:
        """Expand a list that may contain generator nodes."""
        from itertools import product

        expanded_items = []
        for item in items:
            if isinstance(item, dict) and is_generator_node(item):
                expanded_items.append(expand_spec(item))
            else:
                expanded_items.append([item])

        result = []
        for combo in product(*expanded_items):
            result.append(list(combo))

        return result if result else [items]

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_source_names(
        self,
        dataset: "SpectroDataset",
        n_sources: int
    ) -> List[str]:
        """Get source names from dataset."""
        try:
            source_names = []
            for i in range(n_sources):
                if hasattr(dataset, 'source_name'):
                    name = dataset.source_name(i)
                    if name:
                        source_names.append(name)
                        continue
                source_names.append(f"source_{i}")
            return source_names
        except Exception:
            return [f"source_{i}" for i in range(n_sources)]

    def _get_step_names(self, steps: List[Any]) -> str:
        """Get human-readable names for a list of steps."""
        if not steps:
            return ""

        names = []
        for step in steps:
            if hasattr(step, "__class__"):
                names.append(step.__class__.__name__)
            elif isinstance(step, dict):
                keys = [k for k in step.keys() if not k.startswith("_")]
                if keys:
                    names.append(keys[0])
            else:
                names.append(str(step)[:20])

        return ", ".join(names)

    def _generate_step_name(self, step: Any, index: int) -> str:
        """Generate a human-readable name for a step or list of steps."""
        if isinstance(step, list):
            names = [self._get_single_step_name(s) for s in step]
            names = [n for n in names if n]
            if names:
                return "_".join(names[:3])
            return f"branch_{index}"

        return self._get_single_step_name(step) or f"branch_{index}"

    def _get_single_step_name(self, step: Any) -> Optional[str]:
        """Extract a short name from a single step."""
        if step is None:
            return None

        if isinstance(step, str):
            return step

        if isinstance(step, dict):
            if "name" in step:
                return step["name"]
            if "class" in step:
                class_name = step["class"]
                if isinstance(class_name, str) and "." in class_name:
                    return class_name.split(".")[-1]
                return str(class_name).split(".")[-1].replace("'>", "")
            if "model" in step:
                return self._get_single_step_name(step["model"])
            if "preprocessing" in step:
                return self._get_single_step_name(step["preprocessing"])
            keys = [k for k in step.keys() if not k.startswith("_")]
            if keys:
                return keys[0]

        if hasattr(step, "__class__"):
            return step.__class__.__name__

        return None

    def _multiply_branch_contexts(
        self,
        existing: List[Dict[str, Any]],
        new: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Multiply existing branch contexts with new branches (for nesting)."""
        result = []
        flattened_id = 0

        for parent in existing:
            parent_id = parent["branch_id"]
            parent_name = parent["name"]
            parent_context = parent["context"]
            parent_branch_path = parent_context.selector.branch_path or [parent_id]

            for child in new:
                child_id = child["branch_id"]
                child_name = child["name"]
                child_context = child["context"]

                combined_context = child_context.copy()
                combined_branch_path = parent_branch_path + [child_id]

                combined_context.selector = combined_context.selector.with_branch(
                    branch_id=flattened_id,
                    branch_name=f"{parent_name}_{child_name}",
                    branch_path=combined_branch_path
                )

                result.append({
                    "branch_id": flattened_id,
                    "name": f"{parent_name}_{child_name}",
                    "context": combined_context,
                    "parent_branch_id": parent_id,
                    "child_branch_id": child_id,
                    "branch_path": combined_branch_path,
                    "features_snapshot": child.get("features_snapshot"),
                    "chain_snapshot": child.get("chain_snapshot"),
                    "branch_mode": child.get("branch_mode", "duplication"),
                    "sample_indices": child.get("sample_indices"),
                })
                flattened_id += 1

        return result

    def _snapshot_features(self, dataset: "SpectroDataset") -> List[Any]:
        """Create a deep copy of the dataset's feature sources."""
        return copy.deepcopy(dataset._features.sources)

    def _restore_features(
        self,
        dataset: "SpectroDataset",
        snapshot: List[Any]
    ) -> None:
        """Restore the dataset's feature sources from a snapshot."""
        dataset._features.sources = copy.deepcopy(snapshot)

    def _record_dataset_shapes(
        self,
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        is_input: bool = True
    ) -> None:
        """Record dataset shapes to the execution trace for branch substeps."""
        try:
            X_2d = dataset.x(context.selector, layout="2d", include_excluded=False)
            if isinstance(X_2d, list):
                layout_shape = (X_2d[0].shape[0], sum(x.shape[1] for x in X_2d))
            else:
                layout_shape = X_2d.shape

            X_3d = dataset.x(context.selector, layout="3d", concat_source=False, include_excluded=False)
            if not isinstance(X_3d, list):
                X_3d = [X_3d]

            features_shapes = [x.shape for x in X_3d]

            if is_input:
                runtime_context.record_input_shapes(
                    input_shape=layout_shape,
                    features_shape=features_shapes
                )
            else:
                runtime_context.record_output_shapes(
                    output_shape=layout_shape,
                    features_shape=features_shapes
                )

        except Exception:
            pass

    def _extract_substep_info(self, step: Any) -> tuple:
        """Extract operator type and class from a branch substep."""
        if isinstance(step, dict):
            type_keywords = [
                'preprocessing', 'y_processing', 'feature_augmentation',
                'sample_augmentation', 'concat_transform', 'model',
                'meta_model', 'branch', 'merge', 'source_branch',
                'merge_sources', 'name', 'tag', 'exclude'
            ]
            for kw in type_keywords:
                if kw in step:
                    operator = step[kw]
                    if kw == 'name':
                        if 'model' in step:
                            operator = step['model']
                            kw = 'model'
                        else:
                            continue
                    op_class = self._get_operator_class_name(operator)
                    return kw, op_class

            if 'class' in step:
                class_path = step['class']
                if '.' in class_path:
                    op_class = class_path.rsplit('.', 1)[-1]
                else:
                    op_class = class_path
                return 'transform', op_class

            return 'config', 'Config'

        if isinstance(step, str):
            if '.' in step:
                op_class = step.rsplit('.', 1)[-1]
            else:
                op_class = step
            return 'transform', op_class

        if isinstance(step, type):
            return 'transform', step.__name__
        elif hasattr(step, '__class__'):
            return 'transform', type(step).__name__

        return 'operator', str(type(step).__name__)

    def _get_operator_class_name(self, operator: Any) -> str:
        """Get a human-readable class name from an operator."""
        if operator is None:
            return 'None'

        if isinstance(operator, list):
            if len(operator) == 0:
                return 'Empty'
            if len(operator) == 1:
                return self._get_operator_class_name(operator[0])
            names = [self._get_operator_class_name(op) for op in operator[:3]]
            suffix = f"... (+{len(operator)-3})" if len(operator) > 3 else ""
            return ', '.join(names) + suffix

        if isinstance(operator, str):
            if '.' in operator:
                return operator.rsplit('.', 1)[-1]
            return operator

        if isinstance(operator, dict):
            if 'class' in operator:
                class_path = operator['class']
                if '.' in class_path:
                    return class_path.rsplit('.', 1)[-1]
                return class_path
            return 'Config'

        if isinstance(operator, type):
            return operator.__name__

        return type(operator).__name__
