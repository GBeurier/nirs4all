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

import contextlib
import copy
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger
from nirs4all.pipeline.config.generator import (
    expand_spec,
    has_nested_generator_keywords,
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
            True if keyword is "branch"
        """
        return keyword == "branch"

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
        loaded_binaries: list[tuple[str, Any]] | None = None,
        prediction_store: Any | None = None
    ) -> tuple["ExecutionContext", StepOutput]:
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
        loaded_binaries: list[tuple[str, Any]] | None = None,
        prediction_store: Any | None = None
    ) -> tuple["ExecutionContext", StepOutput]:
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

        # Extract parallel execution configuration
        parallel_config = self._get_parallel_config(step_info)

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

        # Determine CoW mode from cache config
        use_cow = self._use_cow_snapshots(runtime_context)

        # Store the initial context as a snapshot point
        initial_context = context.copy()
        initial_processing = copy.deepcopy(context.selector.processing)

        # Snapshot the dataset's feature state before branching
        initial_features_snapshot = self._snapshot_features(dataset, use_cow=use_cow)

        # V3: Snapshot the chain state before branching
        initial_chain = recorder.current_chain() if recorder else None

        # Initialize list to collect branch contexts
        branch_contexts: list[dict[str, Any]] = []
        all_artifacts: list[Any] = []

        # Determine if parallel execution should be used
        use_parallel = self._should_use_parallel_execution(
            parallel_config, branch_defs, mode, n_branches
        )

        logger.info(f"Parallel config: enabled={parallel_config['enabled']}, n_jobs={parallel_config['n_jobs']}, use_parallel={use_parallel}, n_branches={n_branches}, mode={mode}")

        if use_parallel:
            # Parallel execution
            n_jobs = parallel_config["n_jobs"]
            branch_contexts, all_artifacts = self._execute_branches_parallel(
                branch_defs=branch_defs,
                dataset=dataset,
                initial_context=initial_context,
                initial_processing=initial_processing,
                initial_features_snapshot=initial_features_snapshot,
                initial_chain=initial_chain,
                context=context,
                runtime_context=runtime_context,
                loaded_binaries=loaded_binaries,
                prediction_store=prediction_store,
                recorder=recorder,
                mode=mode,
                use_cow=use_cow,
                n_jobs=n_jobs,
            )
        else:
            # Sequential execution (original behavior)
            branch_contexts, all_artifacts = self._execute_branches_sequential(
                branch_defs=branch_defs,
                dataset=dataset,
                initial_context=initial_context,
                initial_processing=initial_processing,
                initial_features_snapshot=initial_features_snapshot,
                initial_chain=initial_chain,
                context=context,
                runtime_context=runtime_context,
                loaded_binaries=loaded_binaries,
                prediction_store=prediction_store,
                recorder=recorder,
                mode=mode,
                use_cow=use_cow,
                target_branch_id=target_branch_id,
            )

        # V3: End branch step in trace
        if recorder is not None:
            recorder.end_step()

        # Release the initial snapshot (no longer needed after all branches executed)
        self._release_snapshot(initial_features_snapshot, use_cow=use_cow)

        # Store branch contexts in custom dict for post-branch iteration
        existing_branches = context.custom.get("branch_contexts", [])
        new_branch_contexts = self._multiply_branch_contexts(existing_branches, branch_contexts) if existing_branches else branch_contexts

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
        raw_def: dict[str, Any],
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: list[tuple[str, Any]] | None = None,
        prediction_store: Any | None = None
    ) -> tuple["ExecutionContext", StepOutput]:
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
                "Unknown separation branch type. "
                "Expected one of: by_tag, by_metadata, by_filter, by_source"
            )

    def _execute_by_tag(
        self,
        raw_def: dict[str, Any],
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: list[tuple[str, Any]] | None = None,
        prediction_store: Any | None = None
    ) -> tuple["ExecutionContext", StepOutput]:
        """Execute separation branch by tag values.

        Syntax:
            {"branch": {
                "by_tag": "tag_name",
                "values": {"branch1": True, "branch2": False},  # Optional
                "steps": [...],  # Shared steps for all branches
            }}
        """
        from nirs4all.controllers.data.branch_utils import (
            group_samples_by_value_mapping,
            parse_value_condition,
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
        raw_def: dict[str, Any],
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: list[tuple[str, Any]] | None = None,
        prediction_store: Any | None = None
    ) -> tuple["ExecutionContext", StepOutput]:
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
            group_samples_by_value_mapping,
            parse_value_condition,
        )

        column = raw_def["by_metadata"]
        value_mapping = raw_def.get("values")
        steps = raw_def.get("steps", [])
        min_samples = raw_def.get("min_samples", 1)

        # Check if metadata column exists
        metadata_df = dataset.metadata()
        if metadata_df is None or column not in metadata_df.columns:
            available_cols = list(metadata_df.columns) if metadata_df is not None else []
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
        column_values = metadata_df[column].to_numpy()[sample_indices]

        # Build value mapping if not provided
        if value_mapping is None:
            unique_values = sorted(set(column_values))
            value_mapping = {str(v): [v] for v in unique_values}
            logger.info(f"  Auto-detected {len(unique_values)} unique values: {unique_values[:10]}{'...' if len(unique_values) > 10 else ''}")

        # Group samples - for metadata, values in mapping are lists of allowed values
        groups: dict[str, list[int]] = {}
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
        raw_def: dict[str, Any],
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: list[tuple[str, Any]] | None = None,
        prediction_store: Any | None = None
    ) -> tuple["ExecutionContext", StepOutput]:
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
        raw_def: dict[str, Any],
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: list[tuple[str, Any]] | None = None,
        prediction_store: Any | None = None
    ) -> tuple["ExecutionContext", StepOutput]:
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
            source_steps = dict.fromkeys(source_names, steps)
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
        new_processing_per_source: list[list[str]] = [list(p) for p in current_processing]

        # Execute per-source pipelines
        source_contexts: list[dict[str, Any]] = []
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
        groups: dict[str, list[int]],
        sample_indices: np.ndarray,
        steps: list[Any],
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: list[tuple[str, Any]] | None = None,
        prediction_store: Any | None = None,
        separation_type: str = "unknown",
        separation_key: str = "",
    ) -> tuple["ExecutionContext", StepOutput]:
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

        # Determine CoW mode from cache config
        use_cow = self._use_cow_snapshots(runtime_context)

        logger.info(f"  Creating {n_branches} separation branches ({separation_type})")

        # Store initial context as snapshot
        initial_context = context.copy()
        initial_processing = copy.deepcopy(context.selector.processing)
        initial_features_snapshot = self._snapshot_features(dataset, use_cow=use_cow)

        # V3: Snapshot chain state
        initial_chain = recorder.current_chain() if recorder else None

        # In predict/explain mode, filter to target branch if specified
        target_branch_id = None
        if mode in ("predict", "explain") and hasattr(runtime_context, 'target_model') and runtime_context.target_model:
            target_branch_id = runtime_context.target_model.get("branch_id")

        # Process each group/branch
        branch_contexts: list[dict[str, Any]] = []
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
            self._restore_features(dataset, initial_features_snapshot, use_cow=use_cow)

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
            branch_features_snapshot = self._snapshot_features(dataset, use_cow=use_cow)

            # Store branch context
            branch_contexts.append({
                "branch_id": branch_id,
                "name": branch_name,
                "context": branch_context,
                "features_snapshot": branch_features_snapshot,
                "chain_snapshot": branch_chain_snapshot,
                "branch_mode": "separation",
                "sample_indices": branch_sample_indices.tolist() if isinstance(branch_sample_indices, np.ndarray) else list(branch_sample_indices),
                "use_cow": use_cow,
            })

            logger.success(f"    Branch {branch_id} '{branch_name}' completed")

        # V3: End branch step
        if recorder is not None:
            recorder.end_step()

        # Release the initial snapshot (no longer needed after all branches executed)
        self._release_snapshot(initial_features_snapshot, use_cow=use_cow)

        # Handle nested branching
        existing_branches = context.custom.get("branch_contexts", [])
        new_branch_contexts = self._multiply_branch_contexts(existing_branches, branch_contexts) if existing_branches else branch_contexts

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
    ) -> list[dict[str, Any]]:
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

            # Reserved keys for branch configuration (not branch names)
            reserved_keys = {"parallel", "n_jobs"}

            # Check if any value contains generator syntax
            expanded_branches = []
            for name, steps in raw_def.items():
                if name.startswith("_"):  # Skip internal keys
                    continue
                if name in reserved_keys:  # Skip configuration keys
                    continue
                if isinstance(steps, dict) and (is_generator_node(steps) or has_nested_generator_keywords(steps)):
                    expanded_steps = expand_spec(steps)
                    for i, exp_step in enumerate(expanded_steps):
                        branch_name = f"{name}_{self._generate_step_name(exp_step, i)}"
                        expanded_branches.append({
                            "name": branch_name,
                            "steps": exp_step if isinstance(exp_step, list) else [exp_step],
                            "generator_choice": exp_step
                        })
                elif isinstance(steps, list) and any(
                    isinstance(s, dict) and (is_generator_node(s) or has_nested_generator_keywords(s)) for s in steps
                ):
                    # Steps list contains generator nodes — expand them
                    expanded_list = self._expand_list_with_generators(steps)
                    logger.info(f"Expanded branch '{name}' with generators into {len(expanded_list)} variants")
                    for i, exp_steps in enumerate(expanded_list):
                        branch_name = f"{name}_{i}" if len(expanded_list) > 1 else name
                        expanded_branches.append({
                            "name": branch_name,
                            "steps": exp_steps,
                            "generator_choice": exp_steps
                        })
                else:
                    expanded_branches.append({
                        "name": name,
                        "steps": steps if isinstance(steps, list) else [steps]
                    })
            logger.info(f"Parsed {len(expanded_branches)} total branches from dict definition")
            return expanded_branches

        # Case 2: List format
        if isinstance(raw_def, list):
            result = []
            for i, item in enumerate(raw_def):
                # Sub-case 2a: Dict with explicit name and steps
                if isinstance(item, dict) and "steps" in item:
                    steps = item["steps"]
                    if isinstance(steps, dict) and (is_generator_node(steps) or has_nested_generator_keywords(steps)):
                        expanded_steps = expand_spec(steps)
                        for j, exp_step in enumerate(expanded_steps):
                            result.append({
                                "name": f"{item.get('name', f'branch_{i}')}_{j}",
                                "steps": exp_step if isinstance(exp_step, list) else [exp_step],
                                "generator_choice": exp_step
                            })
                    elif isinstance(steps, list) and any(
                        isinstance(s, dict) and (is_generator_node(s) or has_nested_generator_keywords(s)) for s in steps
                    ):
                        expanded_list = self._expand_list_with_generators(steps)
                        base_name = item.get('name', f'branch_{i}')
                        for j, exp_steps in enumerate(expanded_list):
                            result.append({
                                "name": f"{base_name}_{j}" if len(expanded_list) > 1 else base_name,
                                "steps": exp_steps,
                                "generator_choice": exp_steps
                            })
                    else:
                        result.append({
                            "name": item.get("name", f"branch_{i}"),
                            "steps": steps
                        })
                # Sub-case 2b: Dict with generator syntax (inside list)
                elif isinstance(item, dict) and (is_generator_node(item) or has_nested_generator_keywords(item)):
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
        generator_node: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Expand a generator node into branch definitions."""
        expanded = expand_spec(generator_node)

        result = []
        for i, item in enumerate(expanded):
            branch_name = self._generate_step_name(item, i)
            steps = item if isinstance(item, list) else [item]

            result.append({
                "name": branch_name,
                "steps": steps,
                "generator_choice": item
            })

        return result

    def _expand_list_with_generators(
        self,
        items: list[Any]
    ) -> list[list[Any]]:
        """Expand a list that may contain generator nodes.

        Uses a two-phase approach:
        1. Phase 1: Expand top-level generator nodes (like _cartesian_, _or_)
           and take Cartesian product, flattening list results.
        2. Phase 2: For each result from phase 1, expand any steps with nested
           generator keywords (like model_params._range_) into additional variants.
        """
        from itertools import product

        # Phase 1: Expand top-level generators only
        expanded_items = []
        for item in items:
            if isinstance(item, dict) and is_generator_node(item):
                expanded_items.append(expand_spec(item))
            else:
                expanded_items.append([item])

        phase1_results = []
        for combo in product(*expanded_items):
            # Flatten: if a generator produced a list of steps, splice them in
            flat = []
            for element in combo:
                if isinstance(element, list):
                    flat.extend(element)
                else:
                    flat.append(element)
            phase1_results.append(flat)

        if not phase1_results:
            phase1_results = [items]

        # Phase 2: Expand nested generators in each phase 1 result
        final_results = []
        for step_list in phase1_results:
            # Check if any step has nested generator keywords
            has_nested = any(
                isinstance(step, dict) and has_nested_generator_keywords(step)
                for step in step_list
            )

            if not has_nested:
                final_results.append(step_list)
                continue

            # Expand nested generators per step and take Cartesian product
            nested_expansions = []
            for step in step_list:
                if isinstance(step, dict) and has_nested_generator_keywords(step):
                    nested_expansions.append(expand_spec(step))
                else:
                    nested_expansions.append([step])

            for combo in product(*nested_expansions):
                final_results.append(list(combo))

        return final_results if final_results else [items]

    # =========================================================================
    # Helper Methods
    # =========================================================================

    @staticmethod
    def _update_best_refit_chains(
        best_chains: dict,
        prediction_store: Any,
        n_pred_before: int,
        branch_steps: list,
    ) -> None:
        """Update the best-refit-chains accumulator after a branch variant.

        Extracts predictions added during this branch variant, groups by
        model_name, computes average val_score per model, and updates the
        accumulator if this variant produced a better score.

        Each model's entry stores only the preprocessing steps + its own
        model step (not other models in the same branch variant).

        Args:
            best_chains: Accumulator dict (model_name -> BestChainEntry).
            prediction_store: Predictions instance with new entries appended.
            n_pred_before: Buffer length before this branch variant started.
            branch_steps: Expanded steps for this branch variant.
        """
        import copy

        from nirs4all.data.predictions import _infer_ascending
        from nirs4all.pipeline.config.context import BestChainEntry

        n_pred_after = prediction_store.num_predictions
        if n_pred_after <= n_pred_before:
            return

        new_entries = prediction_store.slice_after(n_pred_before).iter_entries()

        # Separate branch_steps into preprocessing and model steps.
        # Model steps are dicts with "model" key.
        preprocessing_steps: list = []
        model_steps_by_name: dict[str, dict] = {}
        for step in branch_steps:
            if isinstance(step, dict) and "model" in step:
                name = step.get("name") or type(step["model"]).__name__
                model_steps_by_name[name] = step
            else:
                preprocessing_steps.append(step)

        # Group prediction entries by model_name
        model_scores: dict[str, list[float]] = {}
        model_params: dict[str, dict] = {}
        model_metric: dict[str, str] = {}

        for entry in new_entries:
            model_name = entry.get("model_name")
            val_score = entry.get("val_score")
            if model_name is None or val_score is None:
                continue

            model_name = str(model_name)
            if model_name not in model_scores:
                model_scores[model_name] = []
                model_params[model_name] = {}
                model_metric[model_name] = entry.get("metric", "rmse")

            model_scores[model_name].append(float(val_score))

            # Capture best_params (same for all folds in unified mode)
            bp = entry.get("best_params")
            if bp and not model_params[model_name]:
                if isinstance(bp, str):
                    import json
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        model_params[model_name] = json.loads(bp)
                elif isinstance(bp, dict):
                    model_params[model_name] = bp

        # Update accumulator for each model
        for model_name, scores in model_scores.items():
            avg_score = sum(scores) / len(scores)
            metric = model_metric.get(model_name, "rmse")
            ascending = _infer_ascending(metric)

            existing = best_chains.get(model_name)
            if existing is not None:
                is_better = (
                    (ascending and avg_score < existing.avg_val_score)
                    or (not ascending and avg_score > existing.avg_val_score)
                )
                if not is_better:
                    continue

            # Build model-specific steps: preprocessing + only this model's step
            model_step = model_steps_by_name.get(model_name)
            if model_step is not None:
                model_specific_steps = preprocessing_steps + [model_step]
            else:
                # Fallback: store full branch_steps (shouldn't happen)
                model_specific_steps = branch_steps

            best_chains[model_name] = BestChainEntry(
                model_name=model_name,
                avg_val_score=avg_score,
                branch_steps=copy.deepcopy(model_specific_steps),
                best_params=model_params.get(model_name, {}),
                metric=metric,
            )

    def _use_cow_snapshots(self, runtime_context: "RuntimeContext") -> bool:
        """Check if CoW snapshots are enabled via CacheConfig."""
        if runtime_context is None:
            return False
        cache_config = getattr(runtime_context, "cache_config", None)
        if cache_config is None:
            return False
        return getattr(cache_config, "use_cow_snapshots", False)

    def _get_source_names(
        self,
        dataset: "SpectroDataset",
        n_sources: int
    ) -> list[str]:
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

    def _get_step_names(self, steps: list[Any]) -> str:
        """Get human-readable names for a list of steps."""
        filtered_names = [n for n in (self._get_single_step_name(s) for s in steps) if n is not None]
        return " > ".join(filtered_names) if filtered_names else ""

    def _generate_step_name(self, step: Any, index: int) -> str:
        """Generate a human-readable name for a step or list of steps."""
        if isinstance(step, list):
            filtered_names = [n for n in (self._get_single_step_name(s) for s in step) if n is not None]
            if filtered_names:
                return "_".join(filtered_names[:3])
            return f"branch_{index}"

        return self._get_single_step_name(step) or f"branch_{index}"

    def _get_single_step_name(self, step: Any) -> str | None:
        """Extract a short name from a single step, including non-default parameters."""
        if step is None:
            return None

        if isinstance(step, (int, float, bool)):
            return str(step)

        if isinstance(step, str):
            return step.split(".")[-1] if "." in step else step

        if isinstance(step, dict):
            # Handle explicit name (always takes priority)
            if "name" in step:
                return str(step["name"])

            # Handle serialized class with params
            if "class" in step:
                class_name = step["class"]
                # Extract short class name
                short_name = class_name.split(".")[-1] if isinstance(class_name, str) and "." in class_name else str(class_name).split(".")[-1].replace("'>", "")

                # Format parameters if present
                params = step.get("params", {})
                if params and isinstance(params, dict):
                    # Filter out default/common parameters and format compactly
                    param_strs = []
                    for key, value in params.items():
                        if key.startswith("_"):  # Skip private params
                            continue
                        # Format value compactly
                        if isinstance(value, (int, float, bool)):
                            param_strs.append(f"{key}={value}")
                        elif isinstance(value, str) and len(value) < 15:
                            param_strs.append(f"{key}='{value}'")

                    if param_strs:
                        # Limit to 3 most important params
                        return f"{short_name}({', '.join(param_strs[:3])})"

                return short_name

            # Handle wrapped model or preprocessing
            if "model" in step:
                return self._get_single_step_name(step["model"])
            if "preprocessing" in step:
                return self._get_single_step_name(step["preprocessing"])

            # Fallback to first non-private key
            keys = [k for k in step if not k.startswith("_")]
            if keys:
                return str(keys[0])

        # For operator instances, use format_operator_with_params to show parameters
        if hasattr(step, "__class__"):
            from nirs4all.utils.operator_formatting import format_operator_with_params
            return str(format_operator_with_params(step))

        return None

    def _multiply_branch_contexts(
        self,
        existing: list[dict[str, Any]],
        new: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
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

    def _snapshot_features(self, dataset: "SpectroDataset", use_cow: bool = False) -> list[Any]:
        """Create a snapshot of the dataset's feature sources.

        Args:
            dataset: The dataset to snapshot.
            use_cow: If True, use copy-on-write shared references instead of
                deep-copying.  CoW avoids allocating full copies for branches
                that only read the feature arrays (e.g. model-training steps).

        Returns:
            A snapshot representation.  With CoW this is a lightweight list of
            ``(SharedBlocks, processing_ids, headers, header_unit)`` tuples.
            Without CoW it is a deep-copied list of ``FeatureSource`` objects.
        """
        if use_cow:
            snapshot = []
            for source in dataset._features.sources:
                snapshot.append((
                    source._storage.ensure_shared().acquire(),
                    source._processing_mgr.processing_ids,  # returns a copy
                    list(source._header_mgr.headers) if source._header_mgr.headers else None,
                    source._header_mgr.header_unit,
                ))
            return snapshot
        return copy.deepcopy(dataset._features.sources)

    def _restore_features(
        self,
        dataset: "SpectroDataset",
        snapshot: list[Any],
        use_cow: bool = False,
    ) -> None:
        """Restore the dataset's feature sources from a snapshot.

        Args:
            dataset: The dataset to restore into.
            snapshot: Snapshot created by ``_snapshot_features``.
            use_cow: Must match the flag used when the snapshot was taken.
        """
        if use_cow:
            for source, (shared, proc_ids, headers, header_unit) in zip(
                dataset._features.sources, snapshot, strict=False
            ):
                source._storage.restore_from_shared(shared.acquire())
                source._processing_mgr.reset_processings(proc_ids)
                source._header_mgr.set_headers(headers, unit=header_unit)
        else:
            dataset._features.sources = copy.deepcopy(snapshot)

    def _release_snapshot(self, snapshot: list[Any], use_cow: bool = False) -> None:
        """Release shared references in a CoW snapshot.

        Should be called when a branch snapshot is no longer needed to allow
        prompt memory reclamation.

        Args:
            snapshot: Snapshot created by ``_snapshot_features``.
            use_cow: Must match the flag used when the snapshot was taken.
        """
        if not use_cow:
            return
        for item in snapshot:
            shared = item[0]
            shared.release()

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
            layout_shape = (X_2d[0].shape[0], sum(x.shape[1] for x in X_2d)) if isinstance(X_2d, list) else X_2d.shape

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
                'meta_model', 'branch', 'merge',
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
                op_class = class_path.rsplit('.', 1)[-1] if '.' in class_path else class_path
                return 'transform', op_class

            return 'config', 'Config'

        if isinstance(step, str):
            op_class = step.rsplit('.', 1)[-1] if '.' in step else step
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
                class_path = str(operator['class'])
                if '.' in class_path:
                    return class_path.rsplit('.', 1)[-1]
                return class_path
            return 'Config'

        if isinstance(operator, type):
            return operator.__name__

        return type(operator).__name__

    # =========================================================================
    # Parallel Execution Support
    # =========================================================================

    def _get_parallel_config(self, step_info: "ParsedStep") -> dict[str, Any]:
        """Extract parallel execution configuration from step.

        Returns:
            Dict with keys:
            - enabled: bool - whether parallel execution is enabled
            - n_jobs: int - number of parallel workers (1 = sequential, -1 = auto)
        """
        original_step = step_info.original_step

        # Check for parallel config at top level (for backwards compatibility)
        parallel_enabled = original_step.get("parallel", False)
        n_jobs = original_step.get("n_jobs", None)

        # If branch is a dict (named branches), check inside the branch dict
        branch_def = original_step.get("branch", {})
        if isinstance(branch_def, dict):
            # Override with branch-level parallel config if present
            parallel_enabled = branch_def.get("parallel", parallel_enabled)
            n_jobs = branch_def.get("n_jobs", n_jobs)

        # If n_jobs is explicitly set, use it
        if n_jobs is not None:
            return {"enabled": True, "n_jobs": int(n_jobs)}

        # If parallel is True, use auto-detection
        if parallel_enabled:
            return {"enabled": True, "n_jobs": -1}

        # Default: sequential execution
        return {"enabled": False, "n_jobs": 1}

    def _should_use_parallel_execution(
        self,
        parallel_config: dict[str, Any],
        branch_defs: list[dict[str, Any]],
        mode: str,
        n_branches: int
    ) -> bool:
        """Determine if parallel execution should be used.

        Checks:
        1. Parallel execution is enabled in config
        2. More than 1 branch to execute
        3. Not in predict/explain mode (predict needs sequential for proper artifact loading)
        4. No branches contain incompatible models (nested parallelization, GPU, etc.)

        Returns:
            True if parallel execution is safe and beneficial
        """
        if not parallel_config["enabled"]:
            return False

        if n_branches <= 1:
            return False

        # Predict/explain mode needs careful handling of artifact loading
        # For now, disable parallel execution in these modes
        if mode in ("predict", "explain"):
            logger.debug("Parallel execution disabled: predict/explain mode")
            return False

        # Check each branch for incompatible models
        for idx, branch_def in enumerate(branch_defs):
            should_disable, reason = self._check_branch_parallelization_safety(
                branch_def["steps"]
            )
            if should_disable:
                logger.warning(
                    f"Parallel execution disabled: Branch {idx} '{branch_def['name']}': {reason}"
                )
                return False

        return True

    def _check_branch_parallelization_safety(
        self, branch_steps: list[Any]
    ) -> tuple[bool, str | None]:
        """Check if a branch contains models that conflict with parallel execution.

        Returns:
            (should_disable, reason) tuple
            - should_disable: True if parallel execution should be disabled
            - reason: Human-readable explanation
        """
        for step in branch_steps:
            # Check dict-wrapped models
            if isinstance(step, dict):
                model = step.get("model")
                if model is not None:
                    should_disable, reason = self._check_model_parallelization_safety(model)
                    if should_disable:
                        return True, reason

            # Check direct model instances
            elif hasattr(step, 'fit') and hasattr(step, 'predict'):
                should_disable, reason = self._check_model_parallelization_safety(step)
                if should_disable:
                    return True, reason

        return False, None

    def _check_model_parallelization_safety(
        self, model: Any
    ) -> tuple[bool, str | None]:
        """Check if a model has internal parallelization or GPU usage.

        Returns:
            (should_disable, reason) tuple
        """
        model_class_name = model.__class__.__name__
        model_module = model.__class__.__module__

        # Check for n_jobs > 1 (sklearn models with internal parallelization)
        if hasattr(model, 'n_jobs'):
            n_jobs = getattr(model, 'n_jobs', 1)
            if n_jobs is not None and n_jobs != 1:
                return True, f"{model_class_name} uses n_jobs={n_jobs}"

        # Check for neural network frameworks (often use GPU or threads)
        neural_net_frameworks = ['torch', 'tensorflow', 'keras', 'jax', 'theano']
        if any(framework in model_module for framework in neural_net_frameworks):
            return True, f"Neural network model ({model_class_name})"

        # Check for GPU-related attributes
        gpu_indicators = ['device', 'gpu_id', 'cuda', 'gpu']
        if any(hasattr(model, attr) for attr in gpu_indicators):
            device = getattr(model, 'device', None)
            if device is not None and 'cuda' in str(device).lower():
                return True, f"{model_class_name} uses GPU (device={device})"

        # Check for models with known parallelization issues
        problematic_models = [
            'TabPFNRegressor', 'TabPFNClassifier',  # Uses internal parallelization
            'CatBoostRegressor', 'CatBoostClassifier',  # Thread pool
            'LGBMRegressor', 'LGBMClassifier',  # OpenMP threads
        ]
        if model_class_name in problematic_models:
            return True, f"{model_class_name} has internal parallelization"

        return False, None

    def _execute_branches_sequential(
        self,
        branch_defs: list[dict[str, Any]],
        dataset: "SpectroDataset",
        initial_context: "ExecutionContext",
        initial_processing: list[Any],
        initial_features_snapshot: list[Any],
        initial_chain: Any,
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        loaded_binaries: list[tuple[str, Any]] | None,
        prediction_store: Any | None,
        recorder: Any,
        mode: str,
        use_cow: bool,
        target_branch_id: int | None = None,
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        """Execute branches sequentially (original behavior).

        This is the existing for-loop logic extracted into a method.
        """
        branch_contexts: list[dict[str, Any]] = []
        all_artifacts = []

        # Execute each branch
        for idx, branch_def in enumerate(branch_defs):
            # Use original branch_id if we're in predict mode and have filtered
            branch_id = target_branch_id if target_branch_id is not None else idx

            branch_name = branch_def.get("name", f"branch_{branch_id}")
            branch_steps = branch_def.get("steps", [])

            # Track prediction count before this branch variant for accumulator
            n_pred_before = prediction_store.num_predictions if prediction_store is not None else 0

            chain_str = self._get_step_names(branch_steps)
            logger.info(f"  Branch {branch_id}/{len(branch_defs)}: {branch_name}: {chain_str or '[empty]'}")

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
            self._restore_features(dataset, initial_features_snapshot, use_cow=use_cow)

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

            # Skip expensive shape recording when there are many branches
            # (_record_dataset_shapes calls dataset.x() 4x per substep)
            record_shapes = recorder is not None and len(branch_defs) <= 20

            try:
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

                            if record_shapes:
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
                            if record_shapes:
                                self._record_dataset_shapes(
                                    dataset, result.updated_context, runtime_context, is_input=False
                                )
                            is_model = op_type in ("model", "meta_model")
                            recorder.end_step(is_model=is_model)

                        branch_context = result.updated_context
                        all_artifacts.extend(result.artifacts)
            except Exception as e:
                if mode in ("predict", "explain"):
                    raise
                logger.warning(f"  Branch {branch_id} ({branch_name}) failed: {e}")
                if recorder is not None:
                    recorder.exit_branch()
                continue

            # V3: Snapshot the chain state BEFORE exiting branch context
            branch_chain_snapshot = recorder.current_chain() if recorder else None

            # V3: Exit branch context in trace recorder
            if recorder is not None:
                recorder.exit_branch()

            # Snapshot features AFTER branch processing completes
            branch_features_snapshot = self._snapshot_features(dataset, use_cow=use_cow)

            # Store the final context for this branch
            branch_contexts.append({
                "branch_id": branch_id,
                "name": branch_name,
                "context": branch_context,
                "generator_choice": branch_def.get("generator_choice"),
                "features_snapshot": branch_features_snapshot,
                "chain_snapshot": branch_chain_snapshot,
                "branch_mode": "duplication",
                "use_cow": use_cow,
            })

            logger.success(f"  Branch {branch_id} ({branch_name}) completed")

            # Accumulate best preprocessing chain per model for refit
            if (
                runtime_context
                and runtime_context.best_refit_chains is not None
                and prediction_store is not None  # FIX: Check 'is not None' instead of truthy test, because empty Predictions has __len__ -> bool=False
                and mode == "train"
            ):
                self._update_best_refit_chains(
                    runtime_context.best_refit_chains,
                    prediction_store,
                    n_pred_before,
                    branch_steps,
                )

        return branch_contexts, all_artifacts

    def _execute_branches_parallel(
        self,
        branch_defs: list[dict[str, Any]],
        dataset: "SpectroDataset",
        initial_context: "ExecutionContext",
        initial_processing: list[Any],
        initial_features_snapshot: list[Any],
        initial_chain: Any,
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        loaded_binaries: list[tuple[str, Any]] | None,
        prediction_store: Any | None,
        recorder: Any,
        mode: str,
        use_cow: bool,
        n_jobs: int,
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        """Execute branches in parallel using joblib.

        Uses chunk-based parallelism: creates only n_workers copies of the
        dataset (not n_branches copies).  Each worker processes a chunk of
        branches sequentially, using CoW snapshots for state isolation
        within the chunk.  This avoids the massive overhead of deep-copying
        the dataset per branch.

        Args:
            n_jobs: Number of parallel workers (-1 = auto-detect)
        """
        import multiprocessing

        from joblib import Parallel, delayed

        # Determine effective n_jobs
        if n_jobs == -1:
            n_jobs = min(len(branch_defs), multiprocessing.cpu_count())
        n_jobs = min(n_jobs, len(branch_defs))

        logger.info(f"Parallel execution: {len(branch_defs)} branches with {n_jobs} workers (chunk-based)")

        # Split branches into n_jobs chunks
        chunk_size = (len(branch_defs) + n_jobs - 1) // n_jobs
        branch_chunks: list[list[tuple[int, dict[str, Any]]]] = []
        for i in range(0, len(branch_defs), chunk_size):
            chunk = [(idx, bdef) for idx, bdef in enumerate(branch_defs[i:i + chunk_size], start=i)]
            branch_chunks.append(chunk)

        logger.info(f"Split into {len(branch_chunks)} chunks (avg {chunk_size} branches/worker)")

        parent_branch_path = context.selector.branch_path or []

        def _execute_branch_chunk_worker(
            chunk: list[tuple[int, dict[str, Any]]],
            dataset_copy: "SpectroDataset",
            initial_context_copy: "ExecutionContext",
            initial_processing_copy: list[Any],
            runtime_context_copy: "RuntimeContext",
        ) -> list[dict[str, Any]]:
            """Execute a chunk of branches in a single worker thread.

            Uses CoW snapshot/restore for efficient state isolation
            between branches within the chunk.
            """
            import copy as copy_module

            from nirs4all.data.predictions import Predictions

            # Recreate step_runner in this worker (not picklable)
            if runtime_context_copy and runtime_context_copy.step_runner is None:
                from nirs4all.pipeline.steps.step_runner import StepRunner
                runtime_context_copy.step_runner = StepRunner()

            # Take initial snapshot of the worker's dataset copy for CoW restore
            worker_initial_snapshot = self._snapshot_features(dataset_copy, use_cow=True)

            results = []
            for branch_id, branch_def in chunk:
                branch_name = branch_def.get("name", f"branch_{branch_id}")
                branch_steps = branch_def.get("steps", [])
                local_predictions = Predictions()

                try:
                    # Restore dataset to initial state via CoW (lightweight)
                    self._restore_features(dataset_copy, worker_initial_snapshot, use_cow=True)

                    # Create isolated context for this branch
                    branch_context = initial_context_copy.copy()
                    new_branch_path = parent_branch_path + [branch_id]
                    branch_context.selector = branch_context.selector.with_branch(
                        branch_id=branch_id,
                        branch_name=branch_name,
                        branch_path=new_branch_path,
                    )
                    branch_context.selector.processing = copy_module.deepcopy(initial_processing_copy)

                    if runtime_context_copy:
                        runtime_context_copy.artifact_load_counter = {}

                    # Execute branch steps
                    for substep_idx, substep in enumerate(branch_steps):
                        if runtime_context_copy and runtime_context_copy.step_runner:
                            runtime_context_copy.substep_number = substep_idx
                            step_result = runtime_context_copy.step_runner.execute(
                                step=substep,
                                dataset=dataset_copy,
                                context=branch_context,
                                runtime_context=runtime_context_copy,
                                loaded_binaries=None,
                                prediction_store=local_predictions,
                            )
                            branch_context = step_result.updated_context

                    results.append({
                        "success": True,
                        "branch_id": branch_id,
                        "branch_name": branch_name,
                        "predictions": local_predictions.iter_entries(),
                        "context_selector": {
                            "processing": branch_context.selector.processing,
                            "branch_id": branch_context.selector.branch_id,
                            "branch_name": branch_context.selector.branch_name,
                            "branch_path": branch_context.selector.branch_path,
                        },
                        "generator_choice": branch_def.get("generator_choice"),
                        "branch_steps": branch_steps,
                    })

                except Exception as e:
                    import traceback
                    logger.warning(f"  Branch {branch_id} ({branch_name}) failed: {e}")
                    results.append({
                        "success": False,
                        "branch_id": branch_id,
                        "branch_name": branch_name,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    })

            # Release worker snapshot
            self._release_snapshot(worker_initial_snapshot, use_cow=True)
            return results

        # Temporarily clear unpicklable objects from runtime_context for deep copy
        original_attrs = {}
        unpicklable_keys = ["store", "artifact_registry", "trace_recorder", "step_runner", "artifact_loader"]
        if runtime_context:
            for key in unpicklable_keys:
                original_attrs[key] = getattr(runtime_context, key, None)
                setattr(runtime_context, key, None)

        # Create only n_workers copies (not n_branches copies)
        worker_args = []
        for chunk in branch_chunks:
            dataset_copy = copy.deepcopy(dataset)
            initial_context_copy = copy.deepcopy(initial_context)
            initial_processing_copy = copy.deepcopy(initial_processing)
            runtime_context_copy = copy.deepcopy(runtime_context)
            if runtime_context_copy:
                runtime_context_copy.store = None
                runtime_context_copy.artifact_registry = None
                runtime_context_copy.trace_recorder = None
            worker_args.append((chunk, dataset_copy, initial_context_copy, initial_processing_copy, runtime_context_copy))

        # Restore original runtime_context attributes
        if runtime_context:
            for key, val in original_attrs.items():
                setattr(runtime_context, key, val)

        logger.info(f"Created {len(worker_args)} worker copies (instead of {len(branch_defs)})")

        # Execute chunks in parallel using threading backend
        chunk_results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(_execute_branch_chunk_worker)(*args)
            for args in worker_args
        )

        # Flatten chunk results and collect
        branch_contexts = []
        all_artifacts: list[Any] = []

        for chunk_result_list in chunk_results:
            for result in chunk_result_list:
                if not result["success"]:
                    logger.error(
                        f"Branch {result['branch_id']} ({result['branch_name']}) failed:\n"
                        f"{result['error']}\n{result.get('traceback', '')}"
                    )
                    continue

                n_pred_before = prediction_store.num_predictions if prediction_store is not None else 0

                # Merge predictions into main store
                if prediction_store is not None and result["predictions"]:
                    prediction_store.extend_from_list(result["predictions"])

                # Reconstruct branch context
                branch_context_dict = {
                    "branch_id": result["branch_id"],
                    "name": result["branch_name"],
                    "context": initial_context.copy(),
                    "generator_choice": result.get("generator_choice"),
                    "features_snapshot": None,
                    "chain_snapshot": None,
                    "branch_mode": "duplication",
                    "use_cow": use_cow,
                }
                ctx_sel = result["context_selector"]
                branch_context_dict["context"].selector.processing = ctx_sel["processing"]
                branch_context_dict["context"].selector.branch_id = ctx_sel["branch_id"]
                branch_context_dict["context"].selector.branch_name = ctx_sel["branch_name"]
                branch_context_dict["context"].selector.branch_path = ctx_sel["branch_path"]
                branch_contexts.append(branch_context_dict)

                # Accumulate best preprocessing chain per model for refit
                if (
                    runtime_context
                    and runtime_context.best_refit_chains is not None
                    and prediction_store is not None
                    and mode == "train"
                    and result.get("branch_steps")
                ):
                    self._update_best_refit_chains(
                        runtime_context.best_refit_chains,
                        prediction_store,
                        n_pred_before,
                        result["branch_steps"],
                    )

        total_predictions = prediction_store.num_predictions if prediction_store is not None else 0
        logger.info(
            f"Parallel branch execution complete: {len(branch_contexts)} branches processed, "
            f"{total_predictions} total predictions in store"
        )

        return branch_contexts, all_artifacts
