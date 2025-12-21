"""Merge operator configuration for branch and source merging.

This module provides configuration dataclasses and enums for the MergeController,
which handles combining branch outputs (features and/or predictions) and
exiting branch mode.

The merge operator is the core primitive for all branch combination operations.
It provides:
- Feature merging from branches (horizontal concatenation)
- Prediction merging with OOF reconstruction (data leakage prevention)
- Per-branch model selection and aggregation strategies
- Mixed merging (features from some branches, predictions from others)

Example:
    >>> # Simple feature merge
    >>> {"merge": "features"}
    >>>
    >>> # Prediction merge with OOF safety
    >>> {"merge": "predictions"}
    >>>
    >>> # Mixed merge with per-branch control
    >>> {"merge": {
    ...     "predictions": [
    ...         {"branch": 0, "select": "best", "metric": "rmse"},
    ...         {"branch": 1, "aggregate": "mean"}
    ...     ],
    ...     "features": [2]
    ... }}
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import warnings


class MergeMode(Enum):
    """What to merge from branches.

    Attributes:
        FEATURES: Merge feature matrices from branches.
        PREDICTIONS: Merge model predictions from branches (with OOF reconstruction).
        ALL: Merge both features and predictions from all branches.
    """

    FEATURES = "features"
    PREDICTIONS = "predictions"
    ALL = "all"


class SelectionStrategy(Enum):
    """How to select models within a branch for prediction merging.

    When a branch contains multiple models, this controls which models'
    predictions are included in the merge.

    Attributes:
        ALL: Include all models in the branch (default).
        BEST: Single best model by specified metric.
        TOP_K: Top K models by specified metric.
        EXPLICIT: Explicit list of model names.
    """

    ALL = "all"
    BEST = "best"
    TOP_K = "top_k"
    EXPLICIT = "explicit"


class AggregationStrategy(Enum):
    """How to aggregate predictions from selected models within a branch.

    After model selection, this controls how the selected predictions
    are combined into features for the merged output.

    Attributes:
        SEPARATE: Keep each model's predictions as separate features (default).
            Results in N features (one per selected model).
        MEAN: Simple average of all selected model predictions.
            Results in 1 feature.
        WEIGHTED_MEAN: Weighted average by validation score.
            Results in 1 feature.
        PROBA_MEAN: Average class probabilities (classification only).
            Results in K features (one per class).
    """

    SEPARATE = "separate"
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    PROBA_MEAN = "proba_mean"


class ShapeMismatchStrategy(Enum):
    """How to handle shape mismatches during 3D feature merging.

    This strategy only applies when using 3D layout for features, where
    the number of processings must be aligned across branches. In 2D layout
    (the default), features are simply flattened and concatenated horizontally,
    so different feature dimensions across branches is expected and normal.

    Example:
        - Branch 0: (200 samples, 500 features) from MinMaxScaler
        - Branch 1: (200 samples, 4 processings, 20 features) from multi-processing

        In 2D layout: concatenates to (200, 500 + 4*20 = 580) - no error
        In 3D layout: needs alignment strategy since processings differ

    Attributes:
        ERROR: Raise an error on shape mismatch (default, strictest).
        ALLOW: Flatten to 2D and concatenate regardless of differences.
        PAD: Pad shorter branches with zeros to match longest processings.
        TRUNCATE: Truncate longer branches to match shortest processings.
    """

    ERROR = "error"
    ALLOW = "allow"
    PAD = "pad"
    TRUNCATE = "truncate"


@dataclass
class BranchPredictionConfig:
    """Configuration for prediction collection from a single branch.

    This dataclass specifies how to collect and process predictions
    from a specific branch during merge operations.

    Attributes:
        branch: Branch index or name to collect from.
        select: Model selection strategy.
            - "all" (default): All models in branch
            - "best": Single best model by metric
            - {"top_k": N}: Top N models by metric
            - ["ModelA", "ModelB"]: Explicit model names
        metric: Metric for selection (rmse, mae, r2, accuracy, f1).
            Default is task-appropriate (rmse for regression, accuracy for classification).
        aggregate: How to combine predictions from selected models.
            - "separate" (default): Each model is a separate feature
            - "mean": Simple average of predictions
            - "weighted_mean": Weight by validation score
            - "proba_mean": Average class probabilities (classification)
        weight_metric: Metric for weighted aggregation (default: same as `metric`).
        proba: Use class probabilities instead of predictions (classification only).
        sources: Source filter for multi-source datasets.
            - "all" (default): Include all sources
            - List of source indices or names

    Example:
        >>> # Best model from branch 0 by RMSE
        >>> BranchPredictionConfig(branch=0, select="best", metric="rmse")
        >>>
        >>> # Top 3 models from branch 1, averaged
        >>> BranchPredictionConfig(
        ...     branch=1,
        ...     select={"top_k": 3},
        ...     metric="r2",
        ...     aggregate="mean"
        ... )
        >>>
        >>> # Explicit models with weighted average
        >>> BranchPredictionConfig(
        ...     branch="spectral_path",
        ...     select=["PLS", "RF"],
        ...     aggregate="weighted_mean",
        ...     weight_metric="r2"
        ... )
    """

    branch: Union[int, str]
    select: Union[str, Dict[str, Any], List[str]] = "all"
    metric: Optional[str] = None
    aggregate: str = "separate"
    weight_metric: Optional[str] = None
    proba: bool = False
    sources: Union[str, List[Union[int, str]]] = "all"

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate aggregate
        valid_aggregates = ("separate", "mean", "weighted_mean", "proba_mean")
        if self.aggregate not in valid_aggregates:
            raise ValueError(
                f"aggregate must be one of {valid_aggregates}, got '{self.aggregate}'"
            )

        # Validate select format
        if isinstance(self.select, dict):
            if "top_k" not in self.select:
                raise ValueError(
                    "dict select must contain 'top_k' key, "
                    f"got keys: {list(self.select.keys())}"
                )
            top_k = self.select["top_k"]
            if not isinstance(top_k, int) or top_k < 1:
                raise ValueError(
                    f"top_k must be a positive integer, got {top_k}"
                )
        elif isinstance(self.select, str):
            if self.select not in ("all", "best"):
                raise ValueError(
                    f"string select must be 'all' or 'best', got '{self.select}'"
                )
        elif isinstance(self.select, list):
            if not all(isinstance(s, str) for s in self.select):
                raise ValueError(
                    "list select must contain only string model names"
                )
            if len(self.select) == 0:
                raise ValueError("list select cannot be empty")

        # Validate metric if provided
        valid_metrics = ("rmse", "mae", "r2", "mse", "accuracy", "f1", "auc", "log_loss")
        if self.metric is not None and self.metric not in valid_metrics:
            raise ValueError(
                f"metric must be one of {valid_metrics}, got '{self.metric}'"
            )

        if self.weight_metric is not None and self.weight_metric not in valid_metrics:
            raise ValueError(
                f"weight_metric must be one of {valid_metrics}, got '{self.weight_metric}'"
            )

        # Validate proba_mean requires proba=True
        if self.aggregate == "proba_mean" and not self.proba:
            warnings.warn(
                "aggregate='proba_mean' requires proba=True. Setting proba=True automatically.",
                UserWarning,
                stacklevel=2
            )
            object.__setattr__(self, 'proba', True)

    def get_selection_strategy(self) -> SelectionStrategy:
        """Get the selection strategy enum for this configuration.

        Returns:
            SelectionStrategy enum value based on select field.
        """
        if isinstance(self.select, str):
            if self.select == "all":
                return SelectionStrategy.ALL
            elif self.select == "best":
                return SelectionStrategy.BEST
        elif isinstance(self.select, dict):
            return SelectionStrategy.TOP_K
        elif isinstance(self.select, list):
            return SelectionStrategy.EXPLICIT
        return SelectionStrategy.ALL

    def get_aggregation_strategy(self) -> AggregationStrategy:
        """Get the aggregation strategy enum for this configuration.

        Returns:
            AggregationStrategy enum value based on aggregate field.
        """
        return AggregationStrategy(self.aggregate)


@dataclass
class MergeConfig:
    """Configuration for branch merging operations.

    This dataclass provides complete configuration for the MergeController,
    controlling what data is collected from branches and how it is combined.

    Attributes:
        collect_features: Whether to collect features from branches.
        feature_branches: Which branches to collect features from.
            - "all" (default): All branches
            - List of branch indices: [0, 2] for specific branches
        collect_predictions: Whether to collect predictions from branches.
        prediction_branches: Legacy simple mode: which branches for predictions.
            Use `prediction_configs` for advanced per-branch control.
        prediction_configs: Advanced per-branch prediction configuration.
            Takes precedence over prediction_branches when set.
        model_filter: Legacy: global model filter (simple mode).
            List of model names to include.
        use_proba: Legacy: global proba setting for classification.
        include_original: Include pre-branch features in merged output.
            When True, original features are prepended to merged features.
        on_missing: How to handle missing branches or predictions.
            - "error" (default): Raise an error
            - "warn": Log warning and skip
            - "skip": Silent skip
        on_shape_mismatch: Reserved for 3D layout feature merging.
            In 2D layout (default), features are flattened and concatenated
            horizontally, so different feature dimensions is normal and this
            parameter has no effect. For future 3D layout support:
            - "error": Raise error if processings differ
            - "allow": Flatten to 2D and concatenate
            - "pad": Pad shorter processings with zeros
            - "truncate": Truncate longer to match shortest
        unsafe: If True, DISABLE OOF reconstruction for predictions.
            ⚠️ CAUSES DATA LEAKAGE - only for rapid prototyping.
        output_as: Where to put merged output.
            - "features" (default): Single concatenated feature matrix
            - "sources": Each branch becomes a separate source
            - "dict": Keep as structured dict for multi-head models
        source_names: Custom names for output sources (when output_as="sources").
            If not provided, uses "branch_0", "branch_1", etc.

    Example:
        >>> # Simple feature merge
        >>> MergeConfig(collect_features=True)
        >>>
        >>> # Prediction merge with OOF
        >>> MergeConfig(collect_predictions=True)
        >>>
        >>> # Mixed merge with per-branch control
        >>> MergeConfig(
        ...     collect_predictions=True,
        ...     prediction_configs=[
        ...         BranchPredictionConfig(branch=0, select="best"),
        ...         BranchPredictionConfig(branch=1, aggregate="mean")
        ...     ],
        ...     collect_features=True,
        ...     feature_branches=[2]
        ... )
        >>>
        >>> # Unsafe mode (with warning)
        >>> MergeConfig(collect_predictions=True, unsafe=True)
    """

    collect_features: bool = False
    feature_branches: Union[str, List[int]] = "all"
    collect_predictions: bool = False
    prediction_branches: Union[str, List[int]] = "all"
    prediction_configs: Optional[List[BranchPredictionConfig]] = None
    model_filter: Optional[List[str]] = None
    use_proba: bool = False
    include_original: bool = False
    on_missing: str = "error"
    on_shape_mismatch: str = "error"
    unsafe: bool = False
    output_as: str = "features"
    source_names: Optional[List[str]] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate on_missing
        valid_on_missing = ("error", "warn", "skip")
        if self.on_missing not in valid_on_missing:
            raise ValueError(
                f"on_missing must be one of {valid_on_missing}, got '{self.on_missing}'"
            )

        # Validate on_shape_mismatch
        valid_shape_strategies = ("error", "allow", "pad", "truncate")
        if self.on_shape_mismatch not in valid_shape_strategies:
            raise ValueError(
                f"on_shape_mismatch must be one of {valid_shape_strategies}, "
                f"got '{self.on_shape_mismatch}'"
            )

        # Validate output_as
        valid_output_as = ("features", "sources", "dict")
        if self.output_as not in valid_output_as:
            raise ValueError(
                f"output_as must be one of {valid_output_as}, got '{self.output_as}'"
            )

        # Validate unsafe usage
        if self.unsafe and self.collect_predictions:
            warnings.warn(
                "⚠️ MergeConfig: unsafe=True disables OOF reconstruction. "
                "Training predictions will be used directly, causing DATA LEAKAGE. "
                "Do NOT use for final model evaluation.",
                UserWarning,
                stacklevel=2
            )

        # Validate source_names
        if self.source_names is not None and self.output_as != "sources":
            warnings.warn(
                "source_names is only used when output_as='sources'. "
                "It will be ignored with current output_as setting.",
                UserWarning,
                stacklevel=2
            )

    def has_per_branch_config(self) -> bool:
        """Check if using advanced per-branch prediction configuration.

        Returns:
            True if prediction_configs is set and non-empty.
        """
        return self.prediction_configs is not None and len(self.prediction_configs) > 0

    def get_prediction_configs(
        self,
        n_branches: int
    ) -> List[BranchPredictionConfig]:
        """Get prediction configurations, normalizing legacy format if needed.

        Converts legacy simple mode (prediction_branches + model_filter + use_proba)
        to per-branch configurations for uniform processing.

        Args:
            n_branches: Total number of branches available.

        Returns:
            List of BranchPredictionConfig for each branch to collect from.
        """
        # If advanced config is set, use it directly
        if self.has_per_branch_config():
            return self.prediction_configs

        # Convert legacy format to per-branch configs
        # Resolve branch indices
        if self.prediction_branches == "all":
            branch_indices = list(range(n_branches))
        else:
            branch_indices = self.prediction_branches

        configs = []
        for branch_idx in branch_indices:
            config = BranchPredictionConfig(
                branch=branch_idx,
                select=self.model_filter if self.model_filter else "all",
                proba=self.use_proba,
                aggregate="separate"
            )
            configs.append(config)

        return configs

    def get_feature_branches(self, n_branches: int) -> List[int]:
        """Get list of branch indices to collect features from.

        Args:
            n_branches: Total number of branches available.

        Returns:
            List of branch indices.
        """
        if self.feature_branches == "all":
            return list(range(n_branches))
        return list(self.feature_branches)

    def get_merge_mode(self) -> MergeMode:
        """Determine the merge mode based on configuration.

        Returns:
            MergeMode enum value.
        """
        if self.collect_features and self.collect_predictions:
            return MergeMode.ALL
        elif self.collect_features:
            return MergeMode.FEATURES
        elif self.collect_predictions:
            return MergeMode.PREDICTIONS
        else:
            raise ValueError(
                "Invalid MergeConfig: neither collect_features nor collect_predictions is True"
            )

    def get_shape_mismatch_strategy(self) -> ShapeMismatchStrategy:
        """Get the shape mismatch strategy enum.

        Returns:
            ShapeMismatchStrategy enum value.
        """
        return ShapeMismatchStrategy(self.on_shape_mismatch)
