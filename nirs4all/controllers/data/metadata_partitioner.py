"""
Metadata Partitioner Controller for metadata-based branching.

This controller partitions the dataset into multiple branches based on a
metadata column. Unlike copy branches (where all branches see all samples),
this controller creates non-overlapping sample sets - each sample exists in
exactly ONE branch.

For example, with column="site":
    - Branch "site_A": Contains ONLY samples where metadata["site"] == "A"
    - Branch "site_B": Contains ONLY samples where metadata["site"] == "B"
    - Branch "site_C": Contains ONLY samples where metadata["site"] == "C"

This enables training separate models for different data subsets (e.g., per-site,
per-variety, per-instrument models) and combining their predictions via stacking.

Example:
    >>> pipeline = [
    ...     MinMaxScaler(),
    ...     {
    ...         "branch": [PLS(5), RF(100), XGB()],
    ...         "by": "metadata_partitioner",
    ...         "column": "site",
    ...         "cv": ShuffleSplit(n_splits=3),
    ...         "min_samples": 20,  # Skip branches with < 20 samples
    ...     },
    ...     {"merge": "predictions"},
    ...     Ridge(),
    ... ]
"""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Union, TYPE_CHECKING

import numpy as np

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger
from nirs4all.pipeline.execution.result import StepOutput

logger = get_logger(__name__)

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
    from nirs4all.pipeline.steps.parser import ParsedStep


@dataclass
class MetadataPartitionConfig:
    """Configuration for metadata partitioning.

    Attributes:
        column: Metadata column name to partition by.
        branch_steps: Pipeline steps to execute in each branch.
        cv: Cross-validation splitter for per-branch CV.
        min_samples: Minimum samples required per branch. Branches with
            fewer samples are skipped.
        group_values: Optional dict mapping branch names to lists of values
            to group together. E.g., {"others": ["C", "D", "E"]} groups
            values C, D, E into a single "others" branch.
    """
    column: str
    branch_steps: List[Any]
    cv: Optional[Any] = None
    min_samples: int = 1
    group_values: Optional[Dict[str, List[Any]]] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.column:
            raise ValueError("column must be specified for metadata_partitioner")
        if self.min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {self.min_samples}")


def _parse_metadata_partition_config(step: Dict[str, Any]) -> MetadataPartitionConfig:
    """Parse metadata partitioner configuration from step dict.

    Args:
        step: Step configuration dict with keys:
            - branch: List of steps to run in each branch
            - by: Must be "metadata_partitioner"
            - column: Metadata column name
            - cv: Optional CV splitter
            - min_samples: Optional minimum samples per branch
            - group_values: Optional value grouping dict

    Returns:
        MetadataPartitionConfig instance.

    Raises:
        ValueError: If required keys are missing or invalid.
    """
    branch_def = step.get("branch", [])

    # Handle case where branch is the full config dict
    if isinstance(branch_def, dict):
        column = branch_def.get("column") or step.get("column")
        cv = branch_def.get("cv") or step.get("cv")
        min_samples = branch_def.get("min_samples", step.get("min_samples", 1))
        group_values = branch_def.get("group_values") or step.get("group_values")
        branch_steps = branch_def.get("steps", [])
    else:
        # branch is the list of steps directly
        column = step.get("column")
        cv = step.get("cv")
        min_samples = step.get("min_samples", 1)
        group_values = step.get("group_values")
        branch_steps = branch_def

    if not column:
        raise ValueError(
            "metadata_partitioner requires 'column' parameter. "
            "Specify the metadata column to partition by. "
            "Example: {'branch': [...], 'by': 'metadata_partitioner', 'column': 'site'}"
        )

    return MetadataPartitionConfig(
        column=column,
        branch_steps=branch_steps,
        cv=cv,
        min_samples=min_samples,
        group_values=group_values,
    )


def _build_partition_groups(
    unique_values: List[Any],
    group_values: Optional[Dict[str, List[Any]]]
) -> Dict[str, List[Any]]:
    """Build partition groups from unique values and grouping config.

    Args:
        unique_values: List of unique values in the metadata column.
        group_values: Optional dict mapping group names to value lists.

    Returns:
        Dict mapping partition names to lists of values in that partition.
    """
    if group_values is None:
        # Each unique value becomes its own partition
        return {str(v): [v] for v in unique_values}

    # Build grouped partitions
    partitions = {}
    grouped_values = set()

    for group_name, values in group_values.items():
        partitions[group_name] = values
        grouped_values.update(values)

    # Add remaining ungrouped values as individual partitions
    for v in unique_values:
        if v not in grouped_values:
            partitions[str(v)] = [v]

    return partitions


@register_controller
class MetadataPartitionerController(OperatorController):
    """Controller for metadata-based branching via partitioning.

    This controller creates branches by partitioning samples based on a
    metadata column. Each branch contains a disjoint subset of samples
    where the metadata column equals specific value(s).

    Key behaviors:
        - Each branch contains a disjoint subset of samples
        - Per-branch cross-validation is supported
        - Branches with too few samples can be skipped (min_samples)
        - Values can be grouped into combined branches (group_values)
        - Models train and predict only on their partition

    Attributes:
        priority: Controller priority (set to 3 to run before other controllers).
    """

    priority = 3  # High priority to catch this branch type early

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Check if the step matches the metadata_partitioner branch pattern.

        Matches:
            {"branch": [...], "by": "metadata_partitioner", "column": "..."}

        Args:
            step: Original step configuration
            operator: Deserialized operator
            keyword: Step keyword

        Returns:
            True if this is a metadata_partitioner branch definition.
        """
        if keyword != "branch":
            return False

        if isinstance(step, dict):
            # Check for 'by' key at step level
            if step.get("by") == "metadata_partitioner":
                return True
            # Check for 'by' key inside branch dict
            branch_def = step.get("branch", {})
            if isinstance(branch_def, dict) and branch_def.get("by") == "metadata_partitioner":
                return True

        return False

    @classmethod
    def use_multi_source(cls) -> bool:
        """Metadata partitioner operates on dataset level."""
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Metadata partitioner should execute in prediction mode.

        In prediction mode, we need to route samples to the correct
        branch based on their metadata value.
        """
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
        """Execute the metadata partitioner branch step.

        Creates branches based on metadata column values, with each branch
        containing only samples matching specific value(s).

        In prediction mode, samples are routed to the correct branch based
        on their metadata value. Each sample is processed by the branch
        that matches its metadata value.

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
        # Parse configuration
        config = _parse_metadata_partition_config(step_info.original_step)

        # In prediction/explain mode, use sample routing logic
        if mode in ("predict", "explain"):
            return self._execute_prediction_mode(
                step_info=step_info,
                dataset=dataset,
                context=context,
                runtime_context=runtime_context,
                config=config,
                source=source,
                mode=mode,
                loaded_binaries=loaded_binaries,
                prediction_store=prediction_store,
            )

        logger.info(f"Creating metadata partitioner branches by column '{config.column}'")

        # Store initial context as snapshot
        initial_context = context.copy()
        initial_processing = copy.deepcopy(context.selector.processing)

        # Snapshot dataset features
        initial_features = self._snapshot_features(dataset)

        # Get metadata column values
        metadata = dataset.metadata
        if metadata is None or config.column not in metadata.columns:
            available_cols = list(metadata.columns) if metadata is not None else []
            raise ValueError(
                f"Metadata column '{config.column}' not found. "
                f"Available columns: {available_cols}"
            )

        column_values = metadata[config.column].values

        # Get training sample indices
        train_context = context.with_partition("train")
        train_selector = train_context.selector.copy()
        train_selector.include_augmented = False

        train_sample_indices = dataset._indexer.x_indices(
            train_selector, include_augmented=False, include_excluded=False
        )

        if len(train_sample_indices) == 0:
            logger.warning("No training samples found, skipping metadata partitioner")
            return context, StepOutput()

        # Get unique values and build partition groups
        # Use all samples for determining unique values (not just train)
        unique_values = sorted(set(column_values))
        partition_groups = _build_partition_groups(unique_values, config.group_values)

        logger.info(f"  Found {len(unique_values)} unique values in '{config.column}'")
        logger.info(f"  Creating {len(partition_groups)} partition(s)")

        # V3: Start branch step recording in trace
        recorder = runtime_context.trace_recorder
        if recorder is not None:
            recorder.start_branch_step(
                step_index=runtime_context.step_number,
                branch_count=len(partition_groups),
                operator_config={
                    "by": "metadata_partitioner",
                    "column": config.column,
                    "partitions": list(partition_groups.keys()),
                },
            )

        # In predict/explain mode, filter to target branch if specified
        target_branch_id = None
        target_branch_name = None
        if mode in ("predict", "explain") and hasattr(runtime_context, 'target_model') and runtime_context.target_model:
            target_branch_id = runtime_context.target_model.get("branch_id")
            target_branch_name = runtime_context.target_model.get("branch_name")

        # Create branch contexts
        branch_contexts: List[Dict[str, Any]] = []
        all_artifacts = []
        skipped_branches = []

        branch_id = 0
        for partition_name, partition_values in partition_groups.items():
            # Skip if not target branch in predict mode
            if target_branch_id is not None and branch_id != target_branch_id:
                branch_id += 1
                continue

            # Find sample indices for this partition
            partition_mask = np.isin(column_values, partition_values)
            partition_indices = np.where(partition_mask)[0]

            # Filter to training samples only for min_samples check
            train_partition_indices = np.intersect1d(partition_indices, train_sample_indices)
            n_train_samples = len(train_partition_indices)

            if n_train_samples < config.min_samples:
                logger.warning(
                    f"  Skipping partition '{partition_name}': {n_train_samples} samples "
                    f"< min_samples={config.min_samples}"
                )
                skipped_branches.append({
                    "name": partition_name,
                    "values": partition_values,
                    "n_samples": n_train_samples,
                    "reason": "min_samples",
                })
                branch_id += 1
                continue

            logger.info(
                f"  Partition '{partition_name}': {n_train_samples} train samples "
                f"(values: {partition_values})"
            )

            # V3: Enter branch context in trace recorder
            if recorder is not None:
                recorder.enter_branch(branch_id)

            # Restore dataset features to initial state for this branch
            self._restore_features(dataset, initial_features)

            # Create isolated context for this branch
            branch_context = initial_context.copy()

            # Build branch_path by appending to parent's branch_path
            parent_branch_path = context.selector.branch_path or []
            new_branch_path = parent_branch_path + [branch_id]

            branch_context.selector = branch_context.selector.with_branch(
                branch_id=branch_id,
                branch_name=partition_name,
                branch_path=new_branch_path
            )

            # Reset processing to initial state for this branch
            branch_context.selector.processing = copy.deepcopy(initial_processing)

            # Store metadata partition info
            branch_context.custom["metadata_partition"] = {
                "sample_indices": partition_indices.tolist(),
                "train_sample_indices": train_partition_indices.tolist(),
                "partition_value": partition_name,
                "partition_values": partition_values,
                "column": config.column,
                "n_samples": len(partition_indices),
                "n_train_samples": n_train_samples,
            }

            # Store CV splitter if provided (for per-branch CV)
            if config.cv is not None:
                branch_context.custom["per_branch_cv"] = config.cv

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

            # Apply per-branch CV if specified (before executing branch steps)
            # This creates independent folds within this partition
            if config.cv is not None and mode == "train":
                cv_step = {"split": config.cv}
                if runtime_context.step_runner:
                    logger.debug(f"  Applying per-branch CV for partition '{partition_name}'")
                    runtime_context.substep_number = -1  # Mark as CV step

                    # Record CV substep in trace
                    if recorder is not None:
                        recorder.start_branch_substep(
                            parent_step_index=runtime_context.step_number,
                            branch_id=branch_id,
                            operator_type="split",
                            operator_class=config.cv.__class__.__name__,
                            substep_index=-1,  # Special index for CV
                            branch_name=partition_name,
                        )

                    cv_result = runtime_context.step_runner.execute(
                        step=cv_step,
                        dataset=dataset,
                        context=branch_context,
                        runtime_context=runtime_context,
                        loaded_binaries=branch_binaries,
                        prediction_store=prediction_store
                    )

                    if recorder is not None:
                        recorder.end_step(is_model=False)

                    branch_context = cv_result.updated_context
                    all_artifacts.extend(cv_result.artifacts)

            # Execute branch steps sequentially
            for substep_idx, substep in enumerate(config.branch_steps):
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
                            branch_name=partition_name,
                        )

                    result = runtime_context.step_runner.execute(
                        step=substep,
                        dataset=dataset,
                        context=branch_context,
                        runtime_context=runtime_context,
                        loaded_binaries=branch_binaries,
                        prediction_store=prediction_store
                    )

                    # End substep recording
                    if recorder is not None:
                        is_model = op_type in ("model", "meta_model")
                        recorder.end_step(is_model=is_model)

                    branch_context = result.updated_context
                    all_artifacts.extend(result.artifacts)

            # Snapshot features AFTER branch processing completes
            branch_features_snapshot = self._snapshot_features(dataset)

            # V3: Snapshot chain state before exiting branch
            branch_chain_snapshot = recorder.current_chain() if recorder else None

            # V3: Exit branch context in trace recorder
            if recorder is not None:
                recorder.exit_branch()

            # Store the final context for this branch
            branch_contexts.append({
                "branch_id": branch_id,
                "name": partition_name,
                "context": branch_context,
                "partition_info": {
                    "values": partition_values,
                    "n_samples": len(partition_indices),
                    "n_train_samples": n_train_samples,
                    "sample_indices": partition_indices.tolist(),
                    "train_sample_indices": train_partition_indices.tolist(),
                },
                "features_snapshot": branch_features_snapshot,
                "chain_snapshot": branch_chain_snapshot,
            })

            logger.success(f"  Partition '{partition_name}' (branch {branch_id}) completed")
            branch_id += 1

        # V3: End branch step in trace
        if recorder is not None:
            recorder.end_step()

        # Handle nested branching (multiply with existing branches)
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
        result_context.custom["metadata_partitioner_active"] = True
        result_context.custom["metadata_partitioner_config"] = {
            "column": config.column,
            "group_values": config.group_values,
            "min_samples": config.min_samples,
        }

        # Build metadata
        metadata_info = {
            "branch_count": len(new_branch_contexts),
            "metadata_partitioner": True,
            "column": config.column,
            "partitions": [bc["name"] for bc in branch_contexts],
            "skipped_branches": skipped_branches,
        }

        logger.success(
            f"Metadata partitioner completed with {len(new_branch_contexts)} branch(es)"
            + (f" ({len(skipped_branches)} skipped)" if skipped_branches else "")
        )

        return result_context, StepOutput(
            artifacts=all_artifacts,
            metadata=metadata_info
        )

    def _snapshot_features(self, dataset: "SpectroDataset") -> List[Any]:
        """Create a deep copy of dataset features for branch isolation."""
        return copy.deepcopy(dataset._features.sources)

    def _restore_features(
        self,
        dataset: "SpectroDataset",
        snapshot: List[Any]
    ) -> None:
        """Restore dataset features from snapshot."""
        dataset._features.sources = copy.deepcopy(snapshot)

    def _multiply_branch_contexts(
        self,
        existing: List[Dict[str, Any]],
        new: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Multiply existing branch contexts with new ones for nested branching.

        Creates Cartesian product: each existing branch × each new branch.

        Args:
            existing: List of existing branch context dicts
            new: List of new branch context dicts

        Returns:
            Combined list of branch contexts
        """
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

                # Create combined context
                combined_context = child_context.copy()

                # Build nested branch_path: parent_path + child_id
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
                    "partition_info": child.get("partition_info", {}),
                    "features_snapshot": child.get("features_snapshot"),
                    "chain_snapshot": child.get("chain_snapshot"),
                })
                flattened_id += 1

        return result

    def _extract_substep_info(self, step: Any) -> Tuple[str, str]:
        """Extract operator type and class from a branch substep.

        Args:
            step: The substep configuration (dict, class, or instance)

        Returns:
            Tuple of (operator_type, operator_class)
        """
        # Handle dict steps with keywords
        if isinstance(step, dict):
            type_keywords = [
                'preprocessing', 'y_processing', 'feature_augmentation',
                'sample_augmentation', 'concat_transform', 'model',
                'meta_model', 'branch', 'merge', 'source_branch',
                'merge_sources', 'name'
            ]
            for kw in type_keywords:
                if kw in step:
                    operator = step[kw]
                    if kw == 'name':
                        # For {'name': 'X', 'model': Y}, look for actual operator
                        if 'model' in step:
                            operator = step['model']
                            kw = 'model'
                        else:
                            continue
                    op_class = self._get_operator_class_name(operator)
                    return kw, op_class

            # Check for 'class' key (serialized format)
            if 'class' in step:
                class_path = step['class']
                if '.' in class_path:
                    op_class = class_path.rsplit('.', 1)[-1]
                else:
                    op_class = class_path
                return 'transform', op_class

            return 'config', 'Config'

        # Handle string (class path)
        if isinstance(step, str):
            if '.' in step:
                op_class = step.rsplit('.', 1)[-1]
            else:
                op_class = step
            return 'transform', op_class

        # Handle class or instance
        if isinstance(step, type):
            return 'transform', step.__name__
        elif hasattr(step, '__class__'):
            return 'transform', type(step).__name__

        return 'operator', str(type(step).__name__)

    def _get_operator_class_name(self, operator: Any) -> str:
        """Get a human-readable class name from an operator.

        Args:
            operator: The operator (class, instance, string, or list)

        Returns:
            Human-readable class name string
        """
        if operator is None:
            return 'None'

        if isinstance(operator, list):
            if len(operator) == 0:
                return 'Empty'
            if len(operator) == 1:
                return self._get_operator_class_name(operator[0])
            # Multiple operators - join names
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

    # =========================================================================
    # Phase 4: Prediction Mode Sample Routing
    # =========================================================================

    def _execute_prediction_mode(
        self,
        step_info: "ParsedStep",
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        config: MetadataPartitionConfig,
        source: int = -1,
        mode: str = "predict",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None,
    ) -> Tuple["ExecutionContext", StepOutput]:
        """Execute metadata partitioner in prediction mode with sample routing.

        In prediction mode, each sample is routed to the correct branch based
        on its metadata value. The appropriate branch's transformers and models
        are applied to each sample subset.

        Routing Algorithm:
        1. Read metadata column values for all prediction samples
        2. Group samples by their partition (using same grouping as training)
        3. For each partition, apply branch-specific transformers
        4. Combine results back in original sample order

        Args:
            step_info: Parsed step containing branch definitions
            dataset: Dataset to operate on
            context: Pipeline execution context
            runtime_context: Runtime infrastructure context
            config: Parsed partition configuration
            source: Data source index
            mode: Execution mode ("predict" or "explain")
            loaded_binaries: Pre-loaded binary objects for this step
            prediction_store: External prediction store

        Returns:
            Tuple of (updated_context, StepOutput)

        Raises:
            ValueError: If metadata column is missing or samples cannot be routed.
        """
        logger.info(
            f"Metadata partitioner (predict mode): routing by column '{config.column}'"
        )

        # Get metadata column values
        metadata = dataset.metadata
        if metadata is None:
            raise ValueError(
                f"Dataset has no metadata. Cannot route samples by column '{config.column}' "
                f"in prediction mode. Ensure prediction data includes the same metadata "
                f"column used during training."
            )

        if config.column not in metadata.columns:
            available_cols = list(metadata.columns)
            raise ValueError(
                f"Metadata column '{config.column}' not found in prediction data. "
                f"Available columns: {available_cols}. "
                f"Ensure prediction data includes the same metadata column used during training."
            )

        column_values = metadata[config.column].values
        n_samples = len(column_values)

        # Get unique values and build partition groups (same logic as training)
        unique_values = sorted(set(column_values))
        partition_groups = _build_partition_groups(unique_values, config.group_values)

        logger.info(
            f"  Found {len(unique_values)} unique values, "
            f"mapping to {len(partition_groups)} partition(s)"
        )

        # Load partition routing info from trace/manifest if available
        stored_partitions = self._load_partition_routing_info(runtime_context)

        # Validate partitions match training if stored info is available
        if stored_partitions:
            training_partitions = set(stored_partitions.keys())
            prediction_partitions = set(partition_groups.keys())

            # Check for unknown partitions in prediction data
            unknown_partitions = prediction_partitions - training_partitions
            if unknown_partitions:
                unknown_samples = []
                for partition_name in unknown_partitions:
                    partition_values = partition_groups[partition_name]
                    mask = np.isin(column_values, partition_values)
                    unknown_samples.extend(np.where(mask)[0].tolist())

                logger.warning(
                    f"  {len(unknown_samples)} samples have metadata values "
                    f"not seen during training: {unknown_partitions}. "
                    f"These samples will use fallback routing."
                )

        # Store initial context and features
        initial_context = context.copy()
        initial_processing = copy.deepcopy(context.selector.processing)
        initial_features = self._snapshot_features(dataset)

        # V3: Start branch step recording in trace
        recorder = runtime_context.trace_recorder
        if recorder is not None:
            recorder.start_branch_step(
                step_index=runtime_context.step_number,
                branch_count=len(partition_groups),
                operator_config={
                    "by": "metadata_partitioner",
                    "column": config.column,
                    "partitions": list(partition_groups.keys()),
                    "prediction_mode": True,
                },
            )

        # Process each partition
        branch_contexts: List[Dict[str, Any]] = []
        all_artifacts = []
        processed_samples: Dict[int, int] = {}  # sample_idx -> branch_id

        branch_id = 0
        for partition_name, partition_values in partition_groups.items():
            # Find sample indices for this partition
            partition_mask = np.isin(column_values, partition_values)
            partition_indices = np.where(partition_mask)[0]

            if len(partition_indices) == 0:
                logger.debug(f"  Partition '{partition_name}': no samples in prediction data")
                branch_id += 1
                continue

            logger.info(
                f"  Partition '{partition_name}': {len(partition_indices)} samples"
            )

            # Track which samples go to which branch
            for idx in partition_indices:
                processed_samples[idx] = branch_id

            # V3: Enter branch context in trace recorder
            if recorder is not None:
                recorder.enter_branch(branch_id)

            # Restore dataset features to initial state for this branch
            self._restore_features(dataset, initial_features)

            # Create isolated context for this branch
            branch_context = initial_context.copy()

            # Build branch_path
            parent_branch_path = context.selector.branch_path or []
            new_branch_path = parent_branch_path + [branch_id]

            branch_context.selector = branch_context.selector.with_branch(
                branch_id=branch_id,
                branch_name=partition_name,
                branch_path=new_branch_path
            )

            # Reset processing to initial state
            branch_context.selector.processing = copy.deepcopy(initial_processing)

            # Store metadata partition info for downstream controllers
            branch_context.custom["metadata_partition"] = {
                "sample_indices": partition_indices.tolist(),
                "train_sample_indices": partition_indices.tolist(),  # Same in predict mode
                "partition_value": partition_name,
                "partition_values": partition_values,
                "column": config.column,
                "n_samples": len(partition_indices),
                "n_train_samples": len(partition_indices),
                "prediction_mode": True,
            }

            # Reset artifact load counter for this branch
            if runtime_context:
                runtime_context.artifact_load_counter = {}

            # Load branch-specific artifacts
            branch_binaries = loaded_binaries
            if runtime_context.artifact_loader:
                branch_binaries = runtime_context.artifact_loader.get_step_binaries(
                    runtime_context.step_number, branch_id=branch_id
                )
                if not branch_binaries:
                    branch_binaries = loaded_binaries

            # Execute branch steps
            for substep_idx, substep in enumerate(config.branch_steps):
                if runtime_context.step_runner:
                    runtime_context.substep_number = substep_idx

                    # Record substep in trace
                    if recorder is not None:
                        op_type, op_class = self._extract_substep_info(substep)
                        recorder.start_branch_substep(
                            parent_step_index=runtime_context.step_number,
                            branch_id=branch_id,
                            operator_type=op_type,
                            operator_class=op_class,
                            substep_index=substep_idx,
                            branch_name=partition_name,
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

            # Snapshot features after branch processing
            branch_features_snapshot = self._snapshot_features(dataset)

            # V3: Exit branch
            if recorder is not None:
                recorder.exit_branch()

            # Store branch context
            branch_contexts.append({
                "branch_id": branch_id,
                "name": partition_name,
                "context": branch_context,
                "partition_info": {
                    "values": partition_values,
                    "n_samples": len(partition_indices),
                    "sample_indices": partition_indices.tolist(),
                },
                "features_snapshot": branch_features_snapshot,
            })

            logger.success(
                f"  Partition '{partition_name}' (branch {branch_id}) completed"
            )
            branch_id += 1

        # V3: End branch step
        if recorder is not None:
            recorder.end_step()

        # Check for unprocessed samples (missing partitions)
        unprocessed = [i for i in range(n_samples) if i not in processed_samples]
        if unprocessed:
            logger.warning(
                f"  {len(unprocessed)} samples were not processed "
                f"(no matching partition). Sample indices: {unprocessed[:10]}..."
            )

        # Update result context
        result_context = context.copy()
        result_context.custom["branch_contexts"] = branch_contexts
        result_context.custom["in_branch_mode"] = True
        result_context.custom["metadata_partitioner_active"] = True
        result_context.custom["metadata_partitioner_config"] = {
            "column": config.column,
            "group_values": config.group_values,
            "min_samples": config.min_samples,
        }
        result_context.custom["sample_routing"] = {
            "processed_samples": processed_samples,
            "n_total_samples": n_samples,
            "n_processed": len(processed_samples),
            "n_unprocessed": len(unprocessed),
        }

        # Build metadata
        metadata_info = {
            "branch_count": len(branch_contexts),
            "metadata_partitioner": True,
            "prediction_mode": True,
            "column": config.column,
            "partitions": [bc["name"] for bc in branch_contexts],
            "sample_routing": {
                "n_total": n_samples,
                "n_processed": len(processed_samples),
                "n_unprocessed": len(unprocessed),
            },
        }

        logger.success(
            f"Metadata partitioner (predict mode) completed: "
            f"{len(branch_contexts)} branch(es), "
            f"{len(processed_samples)}/{n_samples} samples routed"
        )

        return result_context, StepOutput(
            artifacts=all_artifacts,
            metadata=metadata_info
        )

    def _load_partition_routing_info(
        self,
        runtime_context: "RuntimeContext"
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Load partition routing info from training trace/manifest.

        Attempts to load the partition configuration used during training
        to validate prediction data and handle missing partitions.

        Args:
            runtime_context: Runtime context with artifact loader

        Returns:
            Dict mapping partition names to their configuration, or None
            if not available.
        """
        if not runtime_context:
            return None

        # Try to get from trace
        if hasattr(runtime_context, 'trace') and runtime_context.trace:
            trace = runtime_context.trace
            step_idx = runtime_context.step_number

            step = trace.get_step(step_idx)
            if step and step.metadata:
                partition_info = step.metadata.get("partitions")
                if partition_info:
                    return {name: {"name": name} for name in partition_info}

        # Try to get from artifact loader metadata
        if hasattr(runtime_context, 'artifact_loader') and runtime_context.artifact_loader:
            loader = runtime_context.artifact_loader
            if hasattr(loader, 'get_step_metadata'):
                step_meta = loader.get_step_metadata(runtime_context.step_number)
                if step_meta and "partitions" in step_meta:
                    partitions = step_meta["partitions"]
                    return {name: {"name": name} for name in partitions}

        return None
