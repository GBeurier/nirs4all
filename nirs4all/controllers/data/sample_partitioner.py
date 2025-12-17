"""
Sample Partitioner Controller for sample-based branching.

This controller partitions the dataset into multiple branches based on a
sample filter (e.g., outlier detection). Unlike OutlierExcluderController
which excludes samples from training, this controller creates separate
branches where each branch contains a different subset of samples.

For example, with Y-outlier detection:
    - Branch "outliers": Contains ONLY the outlier samples
    - Branch "inliers": Contains ONLY the non-outlier samples

This enables training separate models for different data subsets and
comparing their performance.

Example:
    >>> pipeline = [
    ...     ShuffleSplit(n_splits=5),
    ...     {"branch": {
    ...         "by": "sample_partitioner",
    ...         "filter": {"method": "y_outlier", "threshold": 3.0},
    ...     }},
    ...     PLSRegression(n_components=10),
    ... ]
"""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING

import numpy as np

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger
from nirs4all.operators.filters.base import SampleFilter
from nirs4all.operators.filters.x_outlier import XOutlierFilter
from nirs4all.operators.filters.y_outlier import YOutlierFilter
from nirs4all.pipeline.execution.result import StepOutput

logger = get_logger(__name__)

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
    from nirs4all.pipeline.steps.parser import ParsedStep


def _create_partition_filter(filter_config: Dict[str, Any]) -> SampleFilter:
    """
    Create a sample filter from a filter configuration.

    Args:
        filter_config: Dict with 'method' and method-specific parameters.
                      Supported methods:
                      - "y_outlier": Y value outlier detection (IQR-based)
                      - "x_outlier": X value outlier detection
                      - "isolation_forest": Isolation Forest on X
                      - "mahalanobis": Mahalanobis distance on X
                      - "lof": Local Outlier Factor on X

    Returns:
        SampleFilter: Configured filter instance.

    Raises:
        ValueError: If method is not recognized.
    """
    method = filter_config.get("method", "y_outlier")

    # Y-based outlier detection
    if method == "y_outlier":
        threshold = filter_config.get("threshold", 1.5)
        return YOutlierFilter(method="iqr", threshold=threshold)

    # X-based outlier detection methods
    x_methods = {
        "x_outlier": "isolation_forest",
        "isolation_forest": "isolation_forest",
        "mahalanobis": "mahalanobis",
        "robust_mahalanobis": "robust_mahalanobis",
        "lof": "lof",
        "leverage": "pca_leverage",
        "pca_residual": "pca_residual",
    }

    if method in x_methods:
        filter_method = x_methods[method]
        filter_kwargs = {
            "method": filter_method,
            "reason": f"partition_{method}",
        }

        if "contamination" in filter_config:
            filter_kwargs["contamination"] = filter_config["contamination"]
        if "threshold" in filter_config:
            filter_kwargs["threshold"] = filter_config["threshold"]
        if "n_components" in filter_config:
            filter_kwargs["n_components"] = filter_config["n_components"]
        if "random_state" in filter_config:
            filter_kwargs["random_state"] = filter_config["random_state"]

        return XOutlierFilter(**filter_kwargs)

    valid_methods = ["y_outlier"] + list(x_methods.keys())
    raise ValueError(
        f"Unknown partition method '{method}'. "
        f"Valid methods: {valid_methods}"
    )


def _generate_branch_names(filter_config: Dict[str, Any]) -> Tuple[str, str]:
    """
    Generate branch names for the two partitions.

    Args:
        filter_config: Filter configuration dict.

    Returns:
        Tuple of (outliers_name, inliers_name).
    """
    method = filter_config.get("method", "y_outlier")

    # Allow custom names
    if "branch_names" in filter_config:
        names = filter_config["branch_names"]
        if len(names) >= 2:
            return (names[0], names[1])

    # Default names based on method
    if method == "y_outlier":
        return ("y_outliers", "y_inliers")
    elif method in ("x_outlier", "isolation_forest", "mahalanobis", "lof", "leverage", "pca_residual"):
        return ("x_outliers", "x_inliers")
    else:
        return ("outliers", "inliers")


@register_controller
class SamplePartitionerController(OperatorController):
    """
    Controller for sample-based branching via partitioning.

    This controller creates two branches by partitioning samples based on
    a filter (e.g., outlier detection). Each branch contains a different
    subset of samples:
        - "outliers" branch: samples where filter returns False (outliers)
        - "inliers" branch: samples where filter returns True (non-outliers)

    Unlike OutlierExcluderController which only excludes from training,
    this controller truly partitions the samples so each branch trains
    and predicts only on its subset.

    Key behaviors:
        - Each branch contains a disjoint subset of samples
        - Samples are partitioned, not excluded
        - Models train and predict only on their partition
        - Supports Y-outlier and X-outlier detection methods

    Attributes:
        priority: Controller priority (set to 3 to run before outlier excluder).
    """

    priority = 3  # Higher priority than OutlierExcluderController (4)

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """
        Check if the step matches the sample_partitioner branch pattern.

        Matches:
            {"branch": {"by": "sample_partitioner", "filter": {...}}}

        Args:
            step: Original step configuration
            operator: Deserialized operator
            keyword: Step keyword

        Returns:
            True if this is a sample_partitioner branch definition.
        """
        if keyword != "branch":
            return False

        if isinstance(step, dict) and "branch" in step:
            branch_def = step["branch"]
            if isinstance(branch_def, dict) and branch_def.get("by") == "sample_partitioner":
                return True

        return False

    @classmethod
    def use_multi_source(cls) -> bool:
        """Sample partitioner operates on dataset level."""
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """
        Sample partitioner should execute in prediction mode.

        In prediction mode, we need to reconstruct the branch contexts
        and apply the same sample partitioning.
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
        """
        Execute the sample partitioner branch step.

        Creates two branches: one for outliers and one for inliers.
        Each branch contains only its subset of samples.

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
        # Parse filter config from step
        branch_def = step_info.original_step.get("branch", {})
        filter_config = branch_def.get("filter", {"method": "y_outlier"})

        logger.info("Creating sample partitioner branches")

        # Store initial context as snapshot
        initial_context = context.copy()
        initial_processing = copy.deepcopy(context.selector.processing)

        # Snapshot dataset features
        initial_features = self._snapshot_features(dataset)

        # Get training sample indices
        train_context = context.with_partition("train")
        train_selector = train_context.selector.copy()
        train_selector.include_augmented = False

        train_sample_indices = dataset._indexer.x_indices(
            train_selector, include_augmented=False, include_excluded=False
        )

        if len(train_sample_indices) == 0:
            logger.warning("No training samples found, skipping partitioner")
            return context, StepOutput()

        # Get X and Y data for filter
        train_X = dataset.x(train_selector, layout="2d", concat_source=True)
        train_y = dataset.y(train_selector)

        # Create and fit filter
        filter_obj = _create_partition_filter(filter_config)
        filter_obj.fit(train_X, train_y)

        # Get mask: True = inlier (keep), False = outlier (remove)
        mask = filter_obj.get_mask(train_X, train_y)

        # Compute partition indices
        outlier_indices = train_sample_indices[~mask]
        inlier_indices = train_sample_indices[mask]

        n_outliers = len(outlier_indices)
        n_inliers = len(inliers_indices := inlier_indices)  # alias for clarity
        n_total = len(train_sample_indices)

        logger.info(f"  Partition: {n_outliers} outliers, {n_inliers} inliers "
              f"({100 * n_outliers / n_total:.1f}% / {100 * n_inliers / n_total:.1f}%)")

        # Generate branch names
        outliers_name, inliers_name = _generate_branch_names(filter_config)

        # In predict/explain mode, filter to target branch if specified
        target_branch_id = None
        if mode in ("predict", "explain") and hasattr(runtime_context, 'target_model') and runtime_context.target_model:
            target_branch_id = runtime_context.target_model.get("branch_id")

        # Create branch contexts
        branch_contexts: List[Dict[str, Any]] = []
        all_artifacts = []

        # Branch 0: Outliers
        if target_branch_id is None or target_branch_id == 0:
            branch_context_outliers = initial_context.copy()
            branch_context_outliers.selector = branch_context_outliers.selector.with_branch(
                branch_id=0,
                branch_name=outliers_name
            )
            branch_context_outliers.selector.processing = copy.deepcopy(initial_processing)

            # Store sample partition info
            branch_context_outliers.custom["sample_partition"] = {
                "sample_indices": outlier_indices.tolist() if isinstance(outlier_indices, np.ndarray) else list(outlier_indices),
                "partition_type": "outliers",
                "n_samples": n_outliers,
                "filter_config": filter_config,
            }

            branch_contexts.append({
                "branch_id": 0,
                "name": outliers_name,
                "context": branch_context_outliers,
                "partition_info": {
                    "type": "outliers",
                    "n_samples": n_outliers,
                    "sample_indices": outlier_indices.tolist() if isinstance(outlier_indices, np.ndarray) else list(outlier_indices),
                }
            })

            logger.info(f"  Branch 0: {outliers_name} ({n_outliers} samples)")

        # Branch 1: Inliers
        if target_branch_id is None or target_branch_id == 1:
            self._restore_features(dataset, initial_features)

            branch_context_inliers = initial_context.copy()
            branch_context_inliers.selector = branch_context_inliers.selector.with_branch(
                branch_id=1,
                branch_name=inliers_name
            )
            branch_context_inliers.selector.processing = copy.deepcopy(initial_processing)

            # Store sample partition info
            branch_context_inliers.custom["sample_partition"] = {
                "sample_indices": inlier_indices.tolist() if isinstance(inlier_indices, np.ndarray) else list(inlier_indices),
                "partition_type": "inliers",
                "n_samples": n_inliers,
                "filter_config": filter_config,
            }

            branch_contexts.append({
                "branch_id": 1 if target_branch_id is None else target_branch_id,
                "name": inliers_name,
                "context": branch_context_inliers,
                "partition_info": {
                    "type": "inliers",
                    "n_samples": n_inliers,
                    "sample_indices": inlier_indices.tolist() if isinstance(inlier_indices, np.ndarray) else list(inlier_indices),
                }
            })

            logger.info(f"  Branch 1: {inliers_name} ({n_inliers} samples)")

        # Persist filter for prediction mode
        if mode == "train" and runtime_context.saver is not None:
            artifact = runtime_context.saver.persist_artifact(
                step_number=runtime_context.step_number,
                name=f"partition_filter_{runtime_context.next_op()}",
                obj=filter_obj,
                format_hint='sklearn',
            )
            all_artifacts.append(artifact)

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
        result_context.custom["sample_partitioner_active"] = True

        logger.success(f"Sample partitioner completed with {len(new_branch_contexts)} branch(es)")

        return result_context, StepOutput(
            artifacts=all_artifacts,
            metadata={
                "branch_count": len(new_branch_contexts),
                "sample_partitioner": True,
                "n_outliers": n_outliers,
                "n_inliers": n_inliers,
            }
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
        """
        Multiply existing branch contexts with new ones for nested branching.

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

            for child in new:
                child_id = child["branch_id"]
                child_name = child["name"]
                child_context = child["context"]

                # Create combined context
                combined_context = child_context.copy()
                combined_context.selector.branch_id = flattened_id
                combined_context.selector.branch_name = f"{parent_name}_{child_name}"

                result.append({
                    "branch_id": flattened_id,
                    "name": f"{parent_name}_{child_name}",
                    "context": combined_context,
                    "parent_branch_id": parent_id,
                    "child_branch_id": child_id,
                    "partition_info": child.get("partition_info", {}),
                })
                flattened_id += 1

        return result
