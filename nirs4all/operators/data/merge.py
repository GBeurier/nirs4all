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


class BranchType(Enum):
    """Type of branch based on sample handling.

    Attributes:
        COPY: All branches see all samples (default branching behavior).
        METADATA_PARTITIONER: Branches partition samples by metadata column.
        SAMPLE_PARTITIONER: Branches partition samples by filter (e.g., outlier).
    """

    COPY = "copy"
    METADATA_PARTITIONER = "metadata_partitioner"
    SAMPLE_PARTITIONER = "sample_partitioner"


class DisjointSelectionCriterion(Enum):
    """Criterion for selecting top-N models in disjoint branch merge.

    When branches have different model counts, we select top-N models
    from each branch based on this criterion.

    Attributes:
        MSE: Select by lowest Mean Squared Error (default for regression).
        RMSE: Select by lowest Root Mean Squared Error.
        MAE: Select by lowest Mean Absolute Error.
        R2: Select by highest R² score.
        ORDER: Select first N in definition order (no ranking).
    """

    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"
    ORDER = "order"


@dataclass
class DisjointBranchInfo:
    """Information about a single branch in a disjoint merge.

    Captures per-branch statistics and model selection details for
    comprehensive merge metadata.

    Attributes:
        n_samples: Number of samples in this branch partition.
        sample_ids: List of sample indices belonging to this branch.
        n_models_original: Original number of models in the branch.
        n_models_selected: Number of models selected for merge.
        selected_models: List of selected model details with name, score, column.
        dropped_models: List of dropped model details with name, score.
    """

    n_samples: int
    sample_ids: List[int]
    n_models_original: int = 0
    n_models_selected: int = 0
    selected_models: List[Dict[str, Any]] = field(default_factory=list)
    dropped_models: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "n_samples": self.n_samples,
            "sample_ids": self.sample_ids,
            "n_models_original": self.n_models_original,
            "n_models_selected": self.n_models_selected,
            "selected_models": self.selected_models,
            "dropped_models": self.dropped_models,
        }


@dataclass
class DisjointMergeMetadata:
    """Complete metadata for a disjoint sample branch merge.

    This dataclass captures all information about a disjoint merge operation
    for logging, debugging, and downstream use. Matches the specification
    in docs/reports/disjoint_sample_branch_merging.md Section 6.

    Attributes:
        merge_type: Always "disjoint_samples" for disjoint merges.
        n_columns: Number of output columns (prediction features).
        select_by: Selection criterion used (mse, rmse, mae, r2, order).
        branches: Per-branch information as Dict[branch_name, DisjointBranchInfo].
        column_mapping: Maps output column index to per-branch model names.
            Example: {0: {"red": "RF", "blue": "PLS"}, 1: {"red": "PLS", "blue": "RF"}}
        is_heterogeneous: True if different branches have different models per column.
        feature_dim: Feature dimension (for feature merges).

    Example:
        >>> metadata = DisjointMergeMetadata(
        ...     merge_type="disjoint_samples",
        ...     n_columns=2,
        ...     select_by="mse",
        ...     branches={
        ...         "red": DisjointBranchInfo(n_samples=50, sample_ids=[...], ...),
        ...         "blue": DisjointBranchInfo(n_samples=100, sample_ids=[...], ...),
        ...     },
        ...     column_mapping={
        ...         0: {"red": "RF", "blue": "PLS"},
        ...         1: {"red": "PLS", "blue": "RF"},
        ...     },
        ... )
    """

    merge_type: str = "disjoint_samples"
    n_columns: int = 0
    select_by: str = "mse"
    branches: Dict[str, "DisjointBranchInfo"] = field(default_factory=dict)
    column_mapping: Dict[int, Dict[str, str]] = field(default_factory=dict)
    is_heterogeneous: bool = False
    feature_dim: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization/logging.

        Returns:
            Dictionary representation suitable for YAML/JSON serialization.
        """
        return {
            "merge_type": self.merge_type,
            "n_columns": self.n_columns,
            "select_by": self.select_by,
            "branches": {
                name: info.to_dict() for name, info in self.branches.items()
            },
            "column_mapping": self.column_mapping,
            "is_heterogeneous": self.is_heterogeneous,
            "feature_dim": self.feature_dim,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisjointMergeMetadata":
        """Create from dictionary representation.

        Args:
            data: Dictionary with metadata fields.

        Returns:
            DisjointMergeMetadata instance.
        """
        branches = {}
        if "branches" in data:
            for name, info_dict in data["branches"].items():
                branches[name] = DisjointBranchInfo(
                    n_samples=info_dict.get("n_samples", 0),
                    sample_ids=info_dict.get("sample_ids", []),
                    n_models_original=info_dict.get("n_models_original", 0),
                    n_models_selected=info_dict.get("n_models_selected", 0),
                    selected_models=info_dict.get("selected_models", []),
                    dropped_models=info_dict.get("dropped_models", []),
                )

        return cls(
            merge_type=data.get("merge_type", "disjoint_samples"),
            n_columns=data.get("n_columns", 0),
            select_by=data.get("select_by", "mse"),
            branches=branches,
            column_mapping=data.get("column_mapping", {}),
            is_heterogeneous=data.get("is_heterogeneous", False),
            feature_dim=data.get("feature_dim"),
        )

    def get_branch_summary(self) -> str:
        """Get a summary string for logging.

        Returns:
            Human-readable summary of branch statistics.
        """
        parts = []
        for name, info in self.branches.items():
            parts.append(f"'{name}' ({info.n_samples} samples)")
        return ", ".join(parts)

    def get_column_mapping_summary(self) -> List[str]:
        """Get column mapping summary for logging.

        Returns:
            List of strings describing each column's model mapping.
        """
        summaries = []
        for col_idx, mapping in sorted(self.column_mapping.items()):
            model_names = set(mapping.values())
            if len(model_names) == 1:
                # Homogeneous column - all branches use same model
                summaries.append(f"Column {col_idx}: {list(model_names)[0]} (all branches)")
            else:
                # Heterogeneous column
                mapping_str = ", ".join(f"{branch}: {model}" for branch, model in mapping.items())
                summaries.append(f"Column {col_idx}: {mapping_str}")
        return summaries

    def log_summary(self, logger_func) -> None:
        """Log merge summary using provided logger function.

        Args:
            logger_func: Logger function (e.g., logger.info)
        """
        logger_func(
            f"Merging {len(self.branches)} disjoint branches: {self.get_branch_summary()}"
        )
        total_samples = sum(info.n_samples for info in self.branches.values())
        logger_func(f"Output: {total_samples} samples × {self.n_columns} columns")

    def log_warnings(self, logger_warning_func) -> None:
        """Log warnings for heterogeneous columns and dropped models.

        Args:
            logger_warning_func: Logger warning function (e.g., logger.warning)
        """
        # Log model count asymmetry and dropped models
        model_counts = {
            name: info.n_models_original for name, info in self.branches.items()
        }
        unique_counts = set(model_counts.values())

        if len(unique_counts) > 1:
            logger_warning_func(
                f"Model count differs across branches. Using N={self.n_columns} columns (minimum)."
            )
            for name, info in self.branches.items():
                if info.dropped_models:
                    dropped_names = [m.get("name", "?") for m in info.dropped_models]
                    selected_names = [m.get("name", "?") for m in info.selected_models]
                    logger_warning_func(
                        f"  Branch '{name}': selected {selected_names}; dropped {dropped_names}"
                    )
                else:
                    selected_names = [m.get("name", "?") for m in info.selected_models]
                    logger_warning_func(f"  Branch '{name}': all models selected")

        # Log heterogeneous column mapping
        if self.is_heterogeneous:
            logger_warning_func("Column mapping is heterogeneous:")
            for summary in self.get_column_mapping_summary():
                logger_warning_func(f"  {summary}")


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


class SourceMergeStrategy(Enum):
    """How to combine features from multiple data sources.

    Used by the `merge_sources` keyword to control how multi-source
    datasets are unified into a single feature space.

    Attributes:
        CONCAT: Horizontal concatenation of all source features (default).
            Results in 2D array: (samples, sum_of_all_source_features).
            Different feature dimensions per source is expected.
        STACK: Stack sources along a new axis to create 3D tensor.
            Results in 3D array: (samples, n_sources, n_features).
            Requires all sources to have the same feature dimension.
        DICT: Keep sources as a structured dictionary.
            Results in Dict[str, ndarray] for multi-input models.
            Each source is accessible by name.
    """

    CONCAT = "concat"
    STACK = "stack"
    DICT = "dict"


class SourceIncompatibleStrategy(Enum):
    """How to handle incompatible source shapes during stacking.

    When using `stack` strategy with sources that have different feature
    dimensions or processing counts, this controls the resolution.

    Attributes:
        ERROR: Raise an error on incompatible shapes (default, strictest).
        FLATTEN: Force 2D concatenation instead of stacking.
        PAD: Pad shorter sources with zeros to match longest.
        TRUNCATE: Truncate longer sources to match shortest.
    """

    ERROR = "error"
    FLATTEN = "flatten"
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
        >>>
        >>> # Disjoint branch merge with n_columns override
        >>> MergeConfig(
        ...     collect_predictions=True,
        ...     n_columns=2,
        ...     select_by="mse"
        ... )
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
    output_as: str = "features"  # Default to "features" for backward compatibility
    source_names: Optional[List[str]] = None
    # Disjoint sample branch merge options (Phase 2)
    n_columns: Optional[int] = None  # Force output column count for disjoint prediction merge
    select_by: str = "mse"  # Criterion for selecting top-N models (mse, rmse, mae, r2, order)

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

        # Validate n_columns
        if self.n_columns is not None and self.n_columns < 1:
            raise ValueError(
                f"n_columns must be >= 1, got {self.n_columns}"
            )

        # Validate select_by
        valid_select_by = ("mse", "rmse", "mae", "r2", "order")
        if self.select_by not in valid_select_by:
            raise ValueError(
                f"select_by must be one of {valid_select_by}, got '{self.select_by}'"
            )

    def get_selection_criterion(self) -> "DisjointSelectionCriterion":
        """Get the selection criterion enum for disjoint branch merging.

        Returns:
            DisjointSelectionCriterion enum value.
        """
        return DisjointSelectionCriterion(self.select_by)

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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize merge configuration to a dictionary.

        Used for saving merge configuration to manifest for reproducibility
        in prediction mode and bundle export.

        Returns:
            Dictionary representation suitable for YAML/JSON serialization.
        """
        result = {
            "collect_features": self.collect_features,
            "collect_predictions": self.collect_predictions,
            "include_original": self.include_original,
            "on_missing": self.on_missing,
            "on_shape_mismatch": self.on_shape_mismatch,
            "unsafe": self.unsafe,
            "output_as": self.output_as,
        }

        if self.feature_branches != "all":
            result["feature_branches"] = self.feature_branches

        if self.prediction_branches != "all":
            result["prediction_branches"] = self.prediction_branches

        if self.prediction_configs:
            result["prediction_configs"] = [
                {
                    "branch": pc.branch,
                    "select": pc.select,
                    "metric": pc.metric,
                    "aggregate": pc.aggregate,
                    "weight_metric": pc.weight_metric,
                    "proba": pc.proba,
                    "sources": pc.sources,
                }
                for pc in self.prediction_configs
            ]

        if self.model_filter:
            result["model_filter"] = self.model_filter

        if self.use_proba:
            result["use_proba"] = self.use_proba

        if self.source_names:
            result["source_names"] = self.source_names

        # Disjoint branch merge options
        if self.n_columns is not None:
            result["n_columns"] = self.n_columns

        if self.select_by != "mse":  # Only serialize non-default
            result["select_by"] = self.select_by

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MergeConfig":
        """Create MergeConfig from a dictionary.

        Used for loading merge configuration from manifest in prediction mode.

        Args:
            data: Dictionary representation of merge configuration.

        Returns:
            MergeConfig instance.
        """
        prediction_configs = None
        if "prediction_configs" in data:
            prediction_configs = [
                BranchPredictionConfig(
                    branch=pc["branch"],
                    select=pc.get("select", "all"),
                    metric=pc.get("metric"),
                    aggregate=pc.get("aggregate", "separate"),
                    weight_metric=pc.get("weight_metric"),
                    proba=pc.get("proba", False),
                    sources=pc.get("sources", "all"),
                )
                for pc in data["prediction_configs"]
            ]

        return cls(
            collect_features=data.get("collect_features", False),
            feature_branches=data.get("feature_branches", "all"),
            collect_predictions=data.get("collect_predictions", False),
            prediction_branches=data.get("prediction_branches", "all"),
            prediction_configs=prediction_configs,
            model_filter=data.get("model_filter"),
            use_proba=data.get("use_proba", False),
            include_original=data.get("include_original", False),
            on_missing=data.get("on_missing", "error"),
            on_shape_mismatch=data.get("on_shape_mismatch", "error"),
            unsafe=data.get("unsafe", False),
            output_as=data.get("output_as", "sources"),
            source_names=data.get("source_names"),
            n_columns=data.get("n_columns"),
            select_by=data.get("select_by", "mse"),
        )


@dataclass
class SourceMergeConfig:
    """Configuration for merging multi-source dataset features.

    This dataclass provides configuration for the `merge_sources` keyword,
    which combines features from multiple data sources (e.g., NIR, markers,
    Raman) into a unified feature space.

    Unlike branch merging (`merge`), source merging operates on the data
    provenance dimension—combining features that originated from different
    sensors, instruments, or data modalities.

    Attributes:
        strategy: How to combine source features.
            - "concat" (default): Horizontal concatenation (2D result)
            - "stack": Stack along new axis (3D result, requires uniform shapes)
            - "dict": Keep as structured dictionary (for multi-input models)
        sources: Which sources to include.
            - "all" (default): Include all available sources
            - List of source indices: [0, 1] for specific sources
            - List of source names: ["NIR", "markers"] for named sources
        on_incompatible: How to handle incompatible shapes (for stack strategy).
            - "error" (default): Raise error if shapes don't match
            - "flatten": Fall back to 2D concat
            - "pad": Pad shorter with zeros
            - "truncate": Truncate longer to match shortest
        output_name: Name for the merged output source (default: "merged").
        preserve_source_info: Whether to store source metadata for debugging.

    Example:
        >>> # Simple concatenation (default)
        >>> {"merge_sources": "concat"}
        >>>
        >>> # Stack for 3D models (requires same feature count per source)
        >>> {"merge_sources": {"strategy": "stack"}}
        >>>
        >>> # Selective sources with fallback on shape mismatch
        >>> {"merge_sources": {
        ...     "strategy": "stack",
        ...     "sources": ["NIR", "MIR"],
        ...     "on_incompatible": "flatten"
        ... }}
        >>>
        >>> # Dict output for multi-head models
        >>> {"merge_sources": {"strategy": "dict"}}
    """

    strategy: str = "concat"
    sources: Union[str, List[Union[int, str]]] = "all"
    on_incompatible: str = "error"
    output_name: str = "merged"
    preserve_source_info: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate strategy
        valid_strategies = ("concat", "stack", "dict")
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"strategy must be one of {valid_strategies}, got '{self.strategy}'"
            )

        # Validate on_incompatible
        valid_incompatible = ("error", "flatten", "pad", "truncate")
        if self.on_incompatible not in valid_incompatible:
            raise ValueError(
                f"on_incompatible must be one of {valid_incompatible}, "
                f"got '{self.on_incompatible}'"
            )

        # Validate sources
        if isinstance(self.sources, list):
            if len(self.sources) == 0:
                raise ValueError("sources list cannot be empty")

    def get_strategy(self) -> SourceMergeStrategy:
        """Get the merge strategy as an enum.

        Returns:
            SourceMergeStrategy enum value.
        """
        return SourceMergeStrategy(self.strategy)

    def get_incompatible_strategy(self) -> SourceIncompatibleStrategy:
        """Get the incompatible handling strategy as an enum.

        Returns:
            SourceIncompatibleStrategy enum value.
        """
        return SourceIncompatibleStrategy(self.on_incompatible)

    def get_source_indices(self, available_sources: List[str]) -> List[int]:
        """Resolve source specification to indices.

        Args:
            available_sources: List of available source names.

        Returns:
            List of source indices to include.

        Raises:
            ValueError: If a specified source is not found.
        """
        if self.sources == "all":
            return list(range(len(available_sources)))

        indices = []
        for source in self.sources:
            if isinstance(source, int):
                if source < 0 or source >= len(available_sources):
                    raise ValueError(
                        f"Source index {source} out of range. "
                        f"Available: 0-{len(available_sources) - 1}. "
                        f"[Error: MERGE-E031]"
                    )
                indices.append(source)
            elif isinstance(source, str):
                if source not in available_sources:
                    raise ValueError(
                        f"Source name '{source}' not found. "
                        f"Available: {available_sources}. "
                        f"[Error: MERGE-E031]"
                    )
                indices.append(available_sources.index(source))
            else:
                raise ValueError(
                    f"Source must be int or str, got {type(source).__name__}"
                )

        return indices

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary.

        Returns:
            Dictionary representation for manifest storage.
        """
        result = {
            "strategy": self.strategy,
            "on_incompatible": self.on_incompatible,
            "output_name": self.output_name,
            "preserve_source_info": self.preserve_source_info,
        }

        if self.sources != "all":
            result["sources"] = self.sources

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceMergeConfig":
        """Create config from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            SourceMergeConfig instance.
        """
        return cls(
            strategy=data.get("strategy", "concat"),
            sources=data.get("sources", "all"),
            on_incompatible=data.get("on_incompatible", "error"),
            output_name=data.get("output_name", "merged"),
            preserve_source_info=data.get("preserve_source_info", True),
        )


@dataclass
class SourceBranchConfig:
    """Configuration for source branching operations.

    This dataclass provides configuration for the `source_branch` keyword,
    which creates per-source pipeline execution paths. Each source in a
    multi-source dataset gets its own independent processing pipeline.

    Unlike regular branching (`branch`), which creates parallel paths that
    all process the same data, source branching assigns each source to a
    specific processing pipeline based on its name or index.

    Attributes:
        source_pipelines: Mapping of source names/indices to their pipeline steps.
            - Dict[str, List]: Named sources to steps mapping
            - Dict[int, List]: Source indices to steps mapping
            - "auto": Apply same steps to all sources independently
        default_pipeline: Default pipeline for sources not explicitly specified.
            Applied when a source is not listed in source_pipelines.
            If None, unspecified sources are passed through unchanged.
        merge_after: Whether to automatically merge sources after branching.
            - True (default): Automatically call merge_sources after
            - False: Keep sources separate (user must merge manually)
        merge_strategy: Strategy for auto-merge (when merge_after=True).
            - "concat" (default): Horizontal concatenation
            - "stack": Stack along source axis
            - "dict": Keep as dictionary

    Example:
        >>> # Different preprocessing per source
        >>> {"source_branch": {
        ...     "NIR": [SNV(), SavitzkyGolay()],
        ...     "markers": [VarianceThreshold(), MinMaxScaler()],
        ...     "Raman": [BaselineCorrection(), StandardScaler()]
        ... }}
        >>>
        >>> # Source branching with default fallback
        >>> {"source_branch": {
        ...     "NIR": [SNV()],
        ...     "_default_": [MinMaxScaler()]  # Applied to other sources
        ... }}
        >>>
        >>> # Automatic same-preprocessing per source (isolates sources)
        >>> {"source_branch": "auto"}
        >>>
        >>> # Source branching without auto-merge
        >>> {"source_branch": {
        ...     "NIR": [SNV()],
        ...     "markers": [StandardScaler()],
        ...     "_merge_after_": False  # Disable auto-merge
        ... }}
    """

    source_pipelines: Union[str, Dict[Union[str, int], List[Any]]] = field(
        default_factory=dict
    )
    default_pipeline: Optional[List[Any]] = None
    merge_after: bool = True
    merge_strategy: str = "concat"

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate merge_strategy
        valid_strategies = ("concat", "stack", "dict")
        if self.merge_strategy not in valid_strategies:
            raise ValueError(
                f"merge_strategy must be one of {valid_strategies}, "
                f"got '{self.merge_strategy}'"
            )

        # Validate source_pipelines format
        if isinstance(self.source_pipelines, str):
            if self.source_pipelines != "auto":
                raise ValueError(
                    f"string source_pipelines must be 'auto', got '{self.source_pipelines}'"
                )
        elif isinstance(self.source_pipelines, dict):
            # Extract special keys
            if "_default_" in self.source_pipelines:
                self.default_pipeline = self.source_pipelines.pop("_default_")
            if "_merge_after_" in self.source_pipelines:
                self.merge_after = self.source_pipelines.pop("_merge_after_")
            if "_merge_strategy_" in self.source_pipelines:
                self.merge_strategy = self.source_pipelines.pop("_merge_strategy_")

            # Validate remaining keys are valid source references
            for key in self.source_pipelines.keys():
                if not isinstance(key, (str, int)):
                    raise ValueError(
                        f"source_pipelines keys must be str or int, got {type(key).__name__}"
                    )

    def is_auto_mode(self) -> bool:
        """Check if using automatic source branching.

        Returns:
            True if source_pipelines is "auto".
        """
        return self.source_pipelines == "auto"

    def get_pipeline_for_source(
        self,
        source_name: str,
        source_index: int
    ) -> Optional[List[Any]]:
        """Get pipeline steps for a specific source.

        Args:
            source_name: Name of the source.
            source_index: Index of the source.

        Returns:
            List of pipeline steps for this source, or None if passthrough.
        """
        if self.is_auto_mode():
            # Auto mode: return empty list (passthrough with isolation)
            return []

        # Check by name first, then by index (both int and string form)
        if isinstance(self.source_pipelines, dict):
            # Check by source name (e.g., "NIR", "markers")
            if source_name in self.source_pipelines:
                return self.source_pipelines[source_name]
            # Check by integer index (e.g., 0, 1, 2)
            if source_index in self.source_pipelines:
                return self.source_pipelines[source_index]
            # Check by string index (e.g., "0", "1", "2") - for list-indexed syntax
            str_index = str(source_index)
            if str_index in self.source_pipelines:
                return self.source_pipelines[str_index]

        # Fall back to default
        return self.default_pipeline

    def get_all_source_mappings(
        self,
        available_sources: List[str]
    ) -> Dict[str, List[Any]]:
        """Get pipeline mapping for all available sources.

        Args:
            available_sources: List of available source names.

        Returns:
            Dict mapping source names to their pipeline steps.
        """
        result = {}

        if self.is_auto_mode():
            # Auto mode: each source gets empty pipeline (isolation only)
            for source in available_sources:
                result[source] = []
        elif isinstance(self.source_pipelines, dict):
            for idx, source in enumerate(available_sources):
                pipeline = self.get_pipeline_for_source(source, idx)
                if pipeline is not None:
                    result[source] = pipeline
                elif self.default_pipeline is not None:
                    result[source] = self.default_pipeline
                else:
                    # No pipeline specified and no default: passthrough
                    result[source] = []

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary.

        Returns:
            Dictionary representation for manifest storage.
        """
        result = {
            "merge_after": self.merge_after,
            "merge_strategy": self.merge_strategy,
        }

        if self.is_auto_mode():
            result["source_pipelines"] = "auto"
        else:
            # Serialize pipeline references (not the actual objects)
            result["source_pipelines"] = {
                str(k): "..." for k in self.source_pipelines.keys()
            }

        if self.default_pipeline is not None:
            result["default_pipeline"] = "..."

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceBranchConfig":
        """Create config from dictionary.

        Note: This is primarily for metadata reconstruction. The actual
        pipeline steps must be restored from the manifest/artifacts.

        Args:
            data: Dictionary representation.

        Returns:
            SourceBranchConfig instance (with placeholder pipelines).
        """
        source_pipelines = data.get("source_pipelines", {})
        if isinstance(source_pipelines, str) and source_pipelines == "auto":
            source_pipelines = "auto"
        else:
            # Placeholder for actual pipeline reconstruction
            source_pipelines = {}

        return cls(
            source_pipelines=source_pipelines,
            default_pipeline=None,  # Must be reconstructed
            merge_after=data.get("merge_after", True),
            merge_strategy=data.get("merge_strategy", "concat"),
        )
