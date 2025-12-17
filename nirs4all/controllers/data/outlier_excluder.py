"""
Outlier Excluder Controller for sample-based branching.

This controller enables creating multiple branches with different outlier
exclusion strategies. Each branch trains on a different subset of samples
based on the outlier detection method applied.

This is useful for comparing how different outlier handling approaches
affect model performance without creating separate pipeline runs.

Example:
    >>> pipeline = [
    ...     ShuffleSplit(n_splits=5),
    ...     {"branch": {
    ...         "by": "outlier_excluder",
    ...         "strategies": [
    ...             None,  # No exclusion (baseline)
    ...             {"method": "isolation_forest", "contamination": 0.05},
    ...             {"method": "mahalanobis", "threshold": 3.0},
    ...             {"method": "leverage", "threshold": 2.0},
    ...             {"method": "lof", "contamination": 0.05},
    ...         ],
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
from nirs4all.pipeline.execution.result import StepOutput

logger = get_logger(__name__)

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
    from nirs4all.pipeline.steps.parser import ParsedStep


@dataclass
class OutlierExclusionResult:
    """Result of applying an outlier exclusion strategy."""

    strategy_name: str
    n_total: int
    n_excluded: int
    excluded_indices: List[int]
    exclusion_mask: np.ndarray  # True = keep, False = exclude


def _create_outlier_filter(strategy: Dict[str, Any]) -> SampleFilter:
    """
    Create an outlier filter from a strategy specification.

    Supports methods from XOutlierFilter plus additional high-level shortcuts.

    Args:
        strategy: Strategy dict with 'method' and method-specific parameters.
                 Supported methods:
                 - "isolation_forest": Isolation Forest anomaly detection
                 - "mahalanobis": Mahalanobis distance from centroid
                 - "robust_mahalanobis": Robust Mahalanobis (MinCovDet)
                 - "leverage": High leverage points (alias for pca_leverage)
                 - "lof": Local Outlier Factor
                 - "pca_residual": Q-statistic from PCA reconstruction

    Returns:
        SampleFilter: Configured outlier filter instance.

    Raises:
        ValueError: If method is not recognized.
    """
    method = strategy.get("method", "isolation_forest")

    # Map high-level method names to XOutlierFilter methods
    method_mapping = {
        "isolation_forest": "isolation_forest",
        "mahalanobis": "mahalanobis",
        "robust_mahalanobis": "robust_mahalanobis",
        "leverage": "pca_leverage",  # Alias for high leverage detection
        "lof": "lof",
        "pca_residual": "pca_residual",
        "pca_leverage": "pca_leverage",
    }

    if method not in method_mapping:
        valid_methods = list(method_mapping.keys())
        raise ValueError(
            f"Unknown outlier method '{method}'. "
            f"Valid methods: {valid_methods}"
        )

    filter_method = method_mapping[method]

    # Build filter kwargs from strategy
    filter_kwargs = {
        "method": filter_method,
        "reason": f"outlier_{method}",
    }

    # Map strategy parameters to filter parameters
    if "contamination" in strategy:
        filter_kwargs["contamination"] = strategy["contamination"]
    if "threshold" in strategy:
        filter_kwargs["threshold"] = strategy["threshold"]
    if "n_components" in strategy:
        filter_kwargs["n_components"] = strategy["n_components"]
    if "random_state" in strategy:
        filter_kwargs["random_state"] = strategy["random_state"]

    return XOutlierFilter(**filter_kwargs)


def _strategy_to_name(strategy: Optional[Dict[str, Any]], idx: int) -> str:
    """
    Generate a human-readable name for an outlier strategy.

    Args:
        strategy: Strategy dict or None for baseline.
        idx: Strategy index for fallback naming.

    Returns:
        str: Branch name like "baseline", "if_0.05", "mahal_3.0", etc.
    """
    if strategy is None:
        return "baseline"

    method = strategy.get("method", "unknown")

    # Create abbreviated name
    method_abbrev = {
        "isolation_forest": "if",
        "mahalanobis": "mahal",
        "robust_mahalanobis": "rmahal",
        "leverage": "lev",
        "pca_leverage": "lev",
        "lof": "lof",
        "pca_residual": "pca_q",
    }

    abbrev = method_abbrev.get(method, method[:4])

    # Add threshold/contamination to name
    if "contamination" in strategy:
        return f"{abbrev}_{strategy['contamination']}"
    elif "threshold" in strategy:
        return f"{abbrev}_{strategy['threshold']}"
    else:
        return f"{abbrev}_{idx}"


@register_controller
class OutlierExcluderController(OperatorController):
    """
    Controller for sample-based branching with outlier exclusion strategies.

    This controller creates multiple branches, each with a different outlier
    exclusion strategy. Samples identified as outliers are excluded from
    training in that branch, but predictions still cover all samples.

    Key behaviors:
        - Each branch applies a different outlier detection method
        - Outlier detection runs on training data only
        - Exclusion is per-branch (tracked in context, not in indexer)
        - Predictions include exclusion metadata for analysis
        - Branch 0 with None strategy serves as baseline

    Attributes:
        priority: Controller priority (set to 4 to run before regular branch controller).
    """

    priority = 4  # Higher priority than BranchController (5)

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """
        Check if the step matches the outlier excluder branch pattern.

        Matches:
            {"branch": {"by": "outlier_excluder", "strategies": [...]}}

        Args:
            step: Original step configuration
            operator: Deserialized operator
            keyword: Step keyword

        Returns:
            True if this is an outlier_excluder branch definition.
        """
        if keyword != "branch":
            return False

        # Check if it's the outlier_excluder pattern
        if isinstance(step, dict) and "branch" in step:
            branch_def = step["branch"]
            if isinstance(branch_def, dict) and branch_def.get("by") == "outlier_excluder":
                return True

        return False

    @classmethod
    def use_multi_source(cls) -> bool:
        """Outlier excluder operates on dataset level."""
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """
        Outlier excluder should execute in prediction mode.

        In prediction mode, we need to reconstruct the branch contexts
        but NOT apply sample exclusion (we predict on all samples).
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
        Execute the outlier excluder branch step.

        Creates branches for each outlier exclusion strategy. In train mode,
        applies outlier detection and marks exclusions. In predict mode,
        reconstructs branch contexts without applying exclusions.

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
        # Parse strategies from step
        branch_def = step_info.original_step.get("branch", {})
        strategies = branch_def.get("strategies", [])

        if not strategies:
            logger.warning("No outlier strategies defined, skipping outlier excluder")
            return context, StepOutput()

        n_strategies = len(strategies)

        # In predict/explain mode, filter to target branch if specified
        target_branch_id = None
        if mode in ("predict", "explain") and hasattr(runtime_context, 'target_model') and runtime_context.target_model:
            target_branch_id = runtime_context.target_model.get("branch_id")
            if target_branch_id is not None:
                if target_branch_id >= n_strategies:
                    raise ValueError(
                        f"Target branch_id={target_branch_id} not found. "
                        f"Pipeline has {n_strategies} strategies (0-{n_strategies-1})."
                    )
                strategies = [strategies[target_branch_id]]
                logger.info(f"Predict mode: using outlier strategy {target_branch_id}")

        logger.info(f"Creating {len(strategies)} outlier excluder branches")

        # Store initial context as snapshot
        initial_context = context.copy()
        initial_processing = copy.deepcopy(context.selector.processing)

        # Snapshot dataset features
        initial_features = self._snapshot_features(dataset)

        # Get training data for outlier detection
        train_X = None
        train_y = None
        train_sample_indices = None

        if mode == "train":
            train_context = context.with_partition("train")
            train_selector = train_context.selector.copy()
            train_selector.include_augmented = False

            train_sample_indices = dataset._indexer.x_indices(
                train_selector, include_augmented=False, include_excluded=False
            )

            if len(train_sample_indices) > 0:
                train_X = dataset.x(train_selector, layout="2d", concat_source=True)
                train_y = dataset.y(train_selector)

        # Process each strategy
        branch_contexts: List[Dict[str, Any]] = []
        all_artifacts = []
        initial_op_count = runtime_context.operation_count

        for idx, strategy in enumerate(strategies):
            # Determine branch_id (preserve original if in predict mode with filter)
            if target_branch_id is not None:
                branch_id = target_branch_id
                runtime_context.operation_count = initial_op_count + branch_id
            else:
                branch_id = idx

            # Generate branch name from strategy
            branch_name = _strategy_to_name(strategy, branch_id)

            logger.info(f"  Branch {branch_id}: {branch_name}")

            # Create isolated context for this branch
            branch_context = initial_context.copy()
            branch_context.selector = branch_context.selector.with_branch(
                branch_id=branch_id,
                branch_name=branch_name
            )
            branch_context.selector.processing = copy.deepcopy(initial_processing)

            # Restore dataset features
            self._restore_features(dataset, initial_features)

            # Apply outlier exclusion in train mode only
            exclusion_info = {
                "strategy": strategy,
                "n_excluded": 0,
                "exclusion_rate": 0.0,
                "excluded_indices": [],
            }

            if mode == "train" and strategy is not None and train_X is not None and len(train_X) > 0:
                try:
                    result = self._apply_outlier_strategy(
                        strategy=strategy,
                        X=train_X,
                        y=train_y,
                        sample_indices=train_sample_indices,
                        branch_id=branch_id,
                        branch_name=branch_name,
                    )

                    exclusion_info.update({
                        "n_excluded": result.n_excluded,
                        "exclusion_rate": result.n_excluded / result.n_total if result.n_total > 0 else 0.0,
                        "excluded_indices": result.excluded_indices,
                    })

                    # Store exclusion mask in branch context for use by models
                    branch_context.custom["outlier_exclusion"] = {
                        "mask": result.exclusion_mask,  # True = keep
                        "sample_indices": train_sample_indices.tolist() if isinstance(train_sample_indices, np.ndarray) else train_sample_indices,
                        "n_excluded": result.n_excluded,
                        "strategy": strategy,
                    }

                    logger.info(f"    Excluded {result.n_excluded}/{result.n_total} samples "
                          f"({100 * exclusion_info['exclusion_rate']:.1f}%)")

                    # Persist the fitted filter for prediction mode reference
                    if runtime_context.saver is not None:
                        filter_obj = _create_outlier_filter(strategy)
                        filter_obj.fit(train_X, train_y)
                        artifact = runtime_context.saver.persist_artifact(
                            step_number=runtime_context.step_number,
                            name=f"outlier_filter_b{branch_id}_{runtime_context.next_op()}",
                            obj=filter_obj,
                            format_hint='sklearn',
                            branch_id=branch_id,
                            branch_name=branch_name
                        )
                        all_artifacts.append(artifact)

                except Exception as e:
                    logger.warning(f"Outlier detection failed: {e}")
                    # Continue with no exclusion on failure

            elif strategy is None:
                logger.info("    No exclusion (baseline)")

            # Store exclusion info in context for downstream use
            branch_context.custom["exclusion_info"] = exclusion_info

            # Store branch context
            branch_contexts.append({
                "branch_id": branch_id,
                "name": branch_name,
                "context": branch_context,
                "exclusion_info": exclusion_info,
            })

            logger.success(f"  Branch {branch_id} ({branch_name}) completed")

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
        result_context.custom["outlier_excluder_active"] = True

        logger.success(f"Outlier excluder completed with {len(new_branch_contexts)} branch(es)")

        return result_context, StepOutput(
            artifacts=all_artifacts,
            metadata={
                "branch_count": len(new_branch_contexts),
                "outlier_excluder": True,
            }
        )

    def _apply_outlier_strategy(
        self,
        strategy: Dict[str, Any],
        X: np.ndarray,
        y: Optional[np.ndarray],
        sample_indices: np.ndarray,
        branch_id: int,
        branch_name: str,
    ) -> OutlierExclusionResult:
        """
        Apply an outlier detection strategy and return exclusion results.

        Args:
            strategy: Strategy configuration dict
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,) or None
            sample_indices: Original sample indices in dataset
            branch_id: Branch identifier
            branch_name: Branch name

        Returns:
            OutlierExclusionResult with exclusion details
        """
        n_samples = len(X)

        # Create and fit filter
        filter_obj = _create_outlier_filter(strategy)
        filter_obj.fit(X, y)

        # Get mask (True = keep, False = exclude)
        mask = filter_obj.get_mask(X, y)

        # Get excluded indices
        excluded_mask = ~mask
        excluded_sample_indices = sample_indices[excluded_mask].tolist() if isinstance(sample_indices, np.ndarray) else [
            sample_indices[i] for i, ex in enumerate(excluded_mask) if ex
        ]

        return OutlierExclusionResult(
            strategy_name=branch_name,
            n_total=n_samples,
            n_excluded=int(np.sum(excluded_mask)),
            excluded_indices=excluded_sample_indices,
            exclusion_mask=mask,
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

                # Merge exclusion info
                parent_exclusion = parent.get("exclusion_info", {})
                child_exclusion = child.get("exclusion_info", {})

                result.append({
                    "branch_id": flattened_id,
                    "name": f"{parent_name}_{child_name}",
                    "context": combined_context,
                    "parent_branch_id": parent_id,
                    "child_branch_id": child_id,
                    "exclusion_info": {
                        "parent": parent_exclusion,
                        "child": child_exclusion,
                    }
                })
                flattened_id += 1

        return result
