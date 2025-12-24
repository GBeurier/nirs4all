"""Merge Controller for branch combination and exit.

This controller is the CORE PRIMITIVE for all branch combination operations.
It handles:
1. Exiting branch mode (always, unconditionally)
2. Collecting features and/or predictions from branches
3. Enforcing OOF (out-of-fold) safety when predictions are involved
4. Creating a unified dataset for subsequent steps

Phase 1 Implementation:
- Controller registration and matching
- Configuration parsing for all syntax variants
- Branch validation utilities

Phase 3 Implementation:
- Feature collection and concatenation
- Shape mismatch handling

Phase 4 Implementation:
- Model discovery from prediction store
- OOF prediction reconstruction via TrainingSetReconstructor
- Unsafe mode with prominent warnings
- Simple prediction merge syntax

Phase 5 Implementation:
- Per-branch model selection strategies (all, best, top_k, explicit)
- Per-branch aggregation strategies (separate, mean, weighted_mean, proba_mean)
- Model ranking by validation metrics
- Advanced per-branch prediction configuration

Phase 6 Implementation:
- Mixed merging (features from some branches, predictions from others)
- Asymmetric branch detection and handling (models in some, not others)
- Different feature dimensions per branch handling
- Different model counts per branch handling
- Improved error messages with resolution suggestions (MERGE-E010, MERGE-E011)

Phase 8 Implementation:
- Prediction mode support for merge steps
- Bundle export support
- Full train/predict cycle

Phase 9 Implementation:
- Source merge (merge_sources keyword) for multi-source datasets
- Source merge strategies: concat, stack, dict
- Source incompatibility handling: error, flatten, pad, truncate
- Prediction merge (merge_predictions keyword) for late fusion
- Error codes: MERGE-E024, MERGE-E030, MERGE-E031

Example:
    >>> # Simple feature merge
    >>> pipeline = [
    ...     {"branch": [[SNV()], [MSC()]]},
    ...     {"merge": "features"},
    ...     PLSRegression(n_components=10)
    ... ]
    >>>
    >>> # Prediction stacking
    >>> pipeline = [
    ...     {"branch": [[SNV(), PLS()], [MSC(), RF()]]},
    ...     {"merge": "predictions"},
    ...     {"model": Ridge()}
    ... ]
    >>>
    >>> # Source merge for multi-source datasets
    >>> pipeline = [
    ...     SNV(),  # Applied to all sources
    ...     {"merge_sources": "concat"},  # Combine NIR + markers
    ...     {"model": PLS()}
    ... ]
    >>>
    >>> # Late fusion without branches
    >>> pipeline = [
    ...     SNV(),
    ...     {"model": PLS()},
    ...     {"model": RF()},
    ...     {"merge_predictions": "all"},  # Combine predictions
    ...     {"model": Ridge()}
    ... ]

Keywords: "merge", "merge_sources", "merge_predictions"
Priority: 5 (same as BranchController)
"""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

import numpy as np

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.controllers.shared import ModelSelector, PredictionAggregator
from nirs4all.core.logging import get_logger
from nirs4all.operators.data.merge import (
    MergeConfig,
    MergeMode,
    BranchPredictionConfig,
    BranchType,
    DisjointSelectionCriterion,
    DisjointBranchInfo,
    DisjointMergeMetadata,
    SelectionStrategy,
    AggregationStrategy,
    ShapeMismatchStrategy,
    SourceMergeConfig,
    SourceMergeStrategy,
    SourceIncompatibleStrategy,
)
from nirs4all.pipeline.execution.result import StepOutput

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.data._features.feature_source import FeatureSource
    from nirs4all.data.predictions import Predictions
    from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
    from nirs4all.pipeline.steps.parser import ParsedStep

logger = get_logger(__name__)


# =============================================================================
# Phase 5: Model Selection and Prediction Aggregation Utilities
# NOTE: ModelSelector and PredictionAggregator have been moved to
# nirs4all.controllers.shared to avoid code duplication between
# MergeController and MetaModelController (Phase 2 Stacking Restoration).
# They are imported from the shared module and re-exported here for
# backward compatibility.
# =============================================================================


# =============================================================================
# Phase 6: Asymmetric Branch Detection and Handling
# =============================================================================

@dataclass
class BranchAnalysisResult:
    """Result of analyzing branch asymmetry.

    Attributes:
        branch_id: Numeric identifier of the branch.
        branch_name: Name of the branch (if named).
        has_models: Whether the branch contains trained models.
        model_names: List of model names in this branch.
        model_count: Number of models in this branch.
        feature_dim: Feature dimension from this branch (or None if not extracted).
        has_features: Whether the branch has feature snapshots.
    """
    branch_id: int
    branch_name: Optional[str]
    has_models: bool
    model_names: List[str]
    model_count: int
    feature_dim: Optional[int]
    has_features: bool


@dataclass
class AsymmetryReport:
    """Report on asymmetry across branches.

    Provides detailed analysis of how branches differ, helping users
    understand and resolve merge configuration issues.

    Attributes:
        is_asymmetric: Whether any asymmetry was detected.
        has_model_asymmetry: Some branches have models, others don't.
        has_model_count_asymmetry: Branches have different model counts.
        has_feature_dim_asymmetry: Branches have different feature dimensions.
        branches_with_models: List of branch IDs that have models.
        branches_without_models: List of branch IDs without models.
        model_counts: Dict mapping branch_id to model count.
        feature_dims: Dict mapping branch_id to feature dimension.
        summary: Human-readable summary of asymmetry.
    """
    is_asymmetric: bool
    has_model_asymmetry: bool
    has_model_count_asymmetry: bool
    has_feature_dim_asymmetry: bool
    branches_with_models: List[int]
    branches_without_models: List[int]
    model_counts: Dict[int, int]
    feature_dims: Dict[int, Optional[int]]
    summary: str


# =============================================================================
# Phase 2 (Disjoint Sample Branch Merging): Detection and Analysis
# =============================================================================

@dataclass
class DisjointBranchAnalysis:
    """Analysis result for disjoint sample branches.

    Attributes:
        is_disjoint: Whether branches have disjoint sample sets.
        branch_type: Type of disjoint branching (metadata_partitioner, sample_partitioner).
        branch_sample_counts: Dict mapping branch_id to sample count.
        branch_sample_indices: Dict mapping branch_id to list of sample indices.
        total_samples: Total unique samples across all branches.
        partition_column: Metadata column used for partitioning (if metadata_partitioner).
    """
    is_disjoint: bool
    branch_type: Optional[BranchType]
    branch_sample_counts: Dict[int, int]
    branch_sample_indices: Dict[int, List[int]]
    total_samples: int
    partition_column: Optional[str] = None


@dataclass
class DisjointMergeResult:
    """Result of disjoint sample branch merge.

    Attributes:
        merged_array: The merged prediction or feature array (n_total_samples, n_columns).
        n_columns: Number of output columns.
        select_by: Selection criterion used.
        branch_info: Per-branch information about selection and merging.
        column_mapping: Mapping of output columns to per-branch models.
    """
    merged_array: np.ndarray
    n_columns: int
    select_by: str
    branch_info: Dict[str, Any]
    column_mapping: Dict[int, Dict[str, str]]


def is_disjoint_branch(branch_context: Dict[str, Any]) -> bool:
    """Check if a branch context indicates disjoint sample branching.

    A disjoint branch has a 'sample_partition' or 'partition_info' key
    that indicates samples were partitioned (not copied) across branches.

    Args:
        branch_context: A single branch context dictionary.

    Returns:
        True if this branch is part of a disjoint sample partition.
    """
    # Get context object safely
    context = branch_context.get("context")
    custom = getattr(context, "custom", {}) if context else {}

    # Check for sample_partition key (from SamplePartitionerController)
    if "sample_partition" in custom:
        return True

    # Check for metadata_partition key (from MetadataPartitionerController)
    if "metadata_partition" in custom:
        return True

    # Check for partition_info key (from both partitioners)
    partition_info = branch_context.get("partition_info")
    if partition_info and "sample_indices" in partition_info:
        return True

    return False


def detect_disjoint_branches(
    branch_contexts: List[Dict[str, Any]]
) -> DisjointBranchAnalysis:
    """Detect if branches represent disjoint sample partitions.

    Examines branch contexts to determine if they were created by a
    partitioning controller (metadata_partitioner or sample_partitioner).

    Args:
        branch_contexts: List of branch context dictionaries.

    Returns:
        DisjointBranchAnalysis with detection results.
    """
    if not branch_contexts:
        return DisjointBranchAnalysis(
            is_disjoint=False,
            branch_type=None,
            branch_sample_counts={},
            branch_sample_indices={},
            total_samples=0,
        )

    # Check if any branch has partition info
    has_disjoint = False
    branch_type = None
    branch_sample_counts = {}
    branch_sample_indices = {}
    partition_column = None
    all_sample_indices = set()

    for bc in branch_contexts:
        branch_id = bc["branch_id"]
        context = bc.get("context")
        partition_info = bc.get("partition_info", {})

        # Check for partition indicators in context.custom
        custom = context.custom if context else {}

        sample_indices = None

        # Check for sample_partition (SamplePartitionerController)
        if "sample_partition" in custom:
            has_disjoint = True
            branch_type = BranchType.SAMPLE_PARTITIONER
            sample_indices = custom["sample_partition"].get("sample_indices", [])

        # Check for metadata_partition (MetadataPartitionerController)
        elif "metadata_partition" in custom:
            has_disjoint = True
            branch_type = BranchType.METADATA_PARTITIONER
            sample_indices = custom["metadata_partition"].get("sample_indices", [])
            partition_column = custom["metadata_partition"].get("column")

        # Check partition_info (fallback, from both controllers)
        elif "sample_indices" in partition_info:
            has_disjoint = True
            # Determine type from partition_info
            if partition_info.get("type") in ("outliers", "inliers"):
                branch_type = BranchType.SAMPLE_PARTITIONER
            else:
                branch_type = BranchType.METADATA_PARTITIONER
            sample_indices = partition_info.get("sample_indices", [])

        if sample_indices is not None:
            branch_sample_counts[branch_id] = len(sample_indices)
            branch_sample_indices[branch_id] = sample_indices
            all_sample_indices.update(sample_indices)

    # If no disjoint branches found, return non-disjoint result
    if not has_disjoint:
        return DisjointBranchAnalysis(
            is_disjoint=False,
            branch_type=BranchType.COPY,
            branch_sample_counts={},
            branch_sample_indices={},
            total_samples=0,
        )

    return DisjointBranchAnalysis(
        is_disjoint=True,
        branch_type=branch_type,
        branch_sample_counts=branch_sample_counts,
        branch_sample_indices=branch_sample_indices,
        total_samples=len(all_sample_indices),
        partition_column=partition_column,
    )


class AsymmetricBranchAnalyzer:
    """Utility class for analyzing branch asymmetry.

    Detects and reports on asymmetry across branches, providing
    detailed information for error messages and resolution suggestions.

    Phase 6 Features:
    - Detect model presence asymmetry (some have models, some don't)
    - Detect model count asymmetry (different numbers of models)
    - Detect feature dimension asymmetry
    - Generate resolution suggestions for mixed merge
    """

    def __init__(
        self,
        branch_contexts: List[Dict[str, Any]],
        prediction_store: Optional[Any],
        context: "ExecutionContext",
    ):
        """Initialize the analyzer.

        Args:
            branch_contexts: List of branch context dictionaries.
            prediction_store: Prediction storage for model discovery.
            context: Execution context.
        """
        self.branch_contexts = branch_contexts
        self.prediction_store = prediction_store
        self.context = context
        self._analysis_cache: Dict[int, BranchAnalysisResult] = {}

    def analyze_branch(self, branch_idx: int) -> BranchAnalysisResult:
        """Analyze a single branch for its characteristics.

        Args:
            branch_idx: Branch index to analyze.

        Returns:
            BranchAnalysisResult with branch characteristics.
        """
        if branch_idx in self._analysis_cache:
            return self._analysis_cache[branch_idx]

        branch_ctx = None
        for bc in self.branch_contexts:
            if bc["branch_id"] == branch_idx:
                branch_ctx = bc
                break

        if branch_ctx is None:
            # Return empty result for missing branch
            result = BranchAnalysisResult(
                branch_id=branch_idx,
                branch_name=None,
                has_models=False,
                model_names=[],
                model_count=0,
                feature_dim=None,
                has_features=False,
            )
            self._analysis_cache[branch_idx] = result
            return result

        # Extract branch info
        branch_id = branch_ctx["branch_id"]
        branch_name = branch_ctx.get("name")

        # Check for features
        snapshot = branch_ctx.get("features_snapshot")
        has_features = snapshot is not None and len(snapshot) > 0

        # Estimate feature dimension from snapshot
        feature_dim = None
        if has_features:
            try:
                total_features = 0
                for feature_source in snapshot:
                    # Handle different feature source types
                    # Check for FeatureSource (has num_2d_features property)
                    if hasattr(feature_source, 'num_2d_features'):
                        total_features += feature_source.num_2d_features
                    # Fallback to numpy-like shape attribute
                    elif hasattr(feature_source, 'shape'):
                        shape = feature_source.shape
                        if len(shape) >= 2:
                            # shape is (samples, processings, features) or (samples, features)
                            total_features += int(np.prod(shape[1:]))
                feature_dim = int(total_features) if total_features > 0 else None
            except Exception:
                feature_dim = None

        # Discover models in this branch
        model_names = []
        if self.prediction_store is not None:
            current_step = getattr(self.context.state, 'step_number', float('inf'))

            filter_kwargs = {
                'branch_id': branch_id,
                'partition': 'val',
                'load_arrays': False,
            }

            predictions = self.prediction_store.filter_predictions(**filter_kwargs)

            # Filter by step
            predictions = [
                p for p in predictions
                if p.get('step_idx', 0) < current_step
            ]

            model_names = sorted(set(p.get('model_name') for p in predictions if p.get('model_name')))

        result = BranchAnalysisResult(
            branch_id=branch_id,
            branch_name=branch_name,
            has_models=len(model_names) > 0,
            model_names=model_names,
            model_count=len(model_names),
            feature_dim=feature_dim,
            has_features=has_features,
        )

        self._analysis_cache[branch_idx] = result
        return result

    def analyze_all(self) -> AsymmetryReport:
        """Analyze all branches for asymmetry.

        Returns:
            AsymmetryReport with comprehensive asymmetry analysis.
        """
        # Analyze all branches
        analyses = []
        for bc in self.branch_contexts:
            branch_id = bc["branch_id"]
            analyses.append(self.analyze_branch(branch_id))

        if not analyses:
            return AsymmetryReport(
                is_asymmetric=False,
                has_model_asymmetry=False,
                has_model_count_asymmetry=False,
                has_feature_dim_asymmetry=False,
                branches_with_models=[],
                branches_without_models=[],
                model_counts={},
                feature_dims={},
                summary="No branches to analyze.",
            )

        # Detect model presence asymmetry
        branches_with_models = [a.branch_id for a in analyses if a.has_models]
        branches_without_models = [a.branch_id for a in analyses if not a.has_models]
        has_model_asymmetry = len(branches_with_models) > 0 and len(branches_without_models) > 0

        # Detect model count asymmetry
        model_counts = {a.branch_id: a.model_count for a in analyses}
        unique_counts = set(model_counts.values())
        has_model_count_asymmetry = len(unique_counts) > 1

        # Detect feature dimension asymmetry
        feature_dims = {a.branch_id: a.feature_dim for a in analyses}
        known_dims = [d for d in feature_dims.values() if d is not None]
        has_feature_dim_asymmetry = len(set(known_dims)) > 1 if known_dims else False

        is_asymmetric = has_model_asymmetry or has_model_count_asymmetry or has_feature_dim_asymmetry

        # Build summary
        summary_parts = []
        if has_model_asymmetry:
            summary_parts.append(
                f"Model presence asymmetry: branches {branches_with_models} have models, "
                f"branches {branches_without_models} have only features"
            )
        if has_model_count_asymmetry:
            counts_str = ", ".join(f"branch {k}: {v} models" for k, v in model_counts.items())
            summary_parts.append(f"Model count asymmetry: {counts_str}")
        if has_feature_dim_asymmetry:
            dims_str = ", ".join(f"branch {k}: {v} features" for k, v in feature_dims.items() if v is not None)
            summary_parts.append(f"Feature dimension asymmetry: {dims_str}")

        summary = "; ".join(summary_parts) if summary_parts else "Branches are symmetric"

        return AsymmetryReport(
            is_asymmetric=is_asymmetric,
            has_model_asymmetry=has_model_asymmetry,
            has_model_count_asymmetry=has_model_count_asymmetry,
            has_feature_dim_asymmetry=has_feature_dim_asymmetry,
            branches_with_models=branches_with_models,
            branches_without_models=branches_without_models,
            model_counts=model_counts,
            feature_dims=feature_dims,
            summary=summary,
        )

    def suggest_mixed_merge(self) -> Optional[str]:
        """Suggest a mixed merge configuration for asymmetric branches.

        Returns:
            Suggested merge configuration string, or None if not applicable.
        """
        report = self.analyze_all()

        if not report.has_model_asymmetry:
            return None

        # Build suggestion
        predictions_part = f'"predictions": {report.branches_with_models}'
        features_part = f'"features": {report.branches_without_models}'

        return (
            f'Consider mixed merge: {{"merge": {{{predictions_part}, {features_part}}}}}\n'
            f"This collects OOF predictions from branches with models and features from branches without."
        )


class MergeConfigParser:
    """Parser for merge step configurations.

    Handles all syntax variants and normalizes them to MergeConfig.

    Supported syntaxes:
        - Simple string: "features", "predictions", "all"
        - Dict with keys: {"features": ..., "predictions": ..., ...}
        - Legacy format: {"predictions": [0, 1]}
        - Per-branch format: {"predictions": [{"branch": 0, ...}]}
    """

    @classmethod
    def parse(cls, raw_config: Any) -> MergeConfig:
        """Parse raw merge configuration into MergeConfig.

        Args:
            raw_config: The value from {"merge": raw_config}

        Returns:
            Normalized MergeConfig instance.

        Raises:
            ValueError: If configuration format is invalid.
        """
        if isinstance(raw_config, str):
            return cls._parse_simple_string(raw_config)
        elif isinstance(raw_config, dict):
            return cls._parse_dict(raw_config)
        elif isinstance(raw_config, MergeConfig):
            return raw_config
        else:
            raise ValueError(
                f"Invalid merge config type: {type(raw_config).__name__}. "
                f"Expected string, dict, or MergeConfig."
            )

    @classmethod
    def _parse_simple_string(cls, mode_str: str) -> MergeConfig:
        """Parse simple string mode: "features", "predictions", or "all".

        Args:
            mode_str: One of "features", "predictions", "all"

        Returns:
            MergeConfig for the specified mode.

        Raises:
            ValueError: If mode_str is not recognized.
        """
        # Simple string syntax uses output_as="features" by default (legacy behavior)
        # This concatenates all features horizontally into a single feature matrix
        if mode_str == "features":
            return MergeConfig(collect_features=True, output_as="features")
        elif mode_str == "predictions":
            return MergeConfig(collect_predictions=True, output_as="features")
        elif mode_str == "all":
            return MergeConfig(collect_features=True, collect_predictions=True, output_as="features")
        else:
            raise ValueError(
                f"Unknown merge mode: '{mode_str}'. "
                f"Expected 'features', 'predictions', or 'all'."
            )

    @classmethod
    def _parse_dict(cls, config_dict: Dict[str, Any]) -> MergeConfig:
        """Parse dictionary configuration.

        Handles:
            - {"features": ...}: Feature collection config
            - {"predictions": ...}: Prediction collection config
            - Global options: include_original, on_missing, unsafe, output_as
            - Per-branch prediction configs

        Args:
            config_dict: Dictionary configuration

        Returns:
            MergeConfig for the specified configuration.
        """
        config = MergeConfig()

        # Parse features configuration
        if "features" in config_dict:
            config.collect_features = True
            feat_spec = config_dict["features"]
            config.feature_branches = cls._parse_branch_spec(feat_spec)

        # Parse predictions configuration
        if "predictions" in config_dict:
            config.collect_predictions = True
            pred_spec = config_dict["predictions"]
            config = cls._parse_predictions_spec(config, pred_spec)

        # Parse global options
        config.include_original = config_dict.get("include_original", False)
        config.on_missing = config_dict.get("on_missing", "error")
        config.on_shape_mismatch = config_dict.get("on_shape_mismatch", "error")
        config.unsafe = config_dict.get("unsafe", False)
        config.output_as = config_dict.get("output_as", "features")
        config.source_names = config_dict.get("source_names")

        # Parse disjoint sample branch merge options (Phase 2)
        config.n_columns = config_dict.get("n_columns")
        config.select_by = config_dict.get("select_by", "mse")

        # Validate at least one collection mode is enabled
        if not config.collect_features and not config.collect_predictions:
            raise ValueError(
                "Merge config must specify at least one of 'features' or 'predictions'. "
                f"Got keys: {list(config_dict.keys())}"
            )

        return config

    @classmethod
    def _parse_branch_spec(
        cls,
        spec: Union[str, List[int], Dict[str, Any]]
    ) -> Union[str, List[int]]:
        """Parse branch specification for features.

        Args:
            spec: Branch specification:
                - "all": All branches
                - [0, 1, 2]: Specific branch indices
                - {"branches": [0, 1]}: Dict with branches key

        Returns:
            "all" or list of branch indices.
        """
        if spec == "all" or spec is True:
            return "all"
        elif isinstance(spec, list):
            # Validate all are integers
            if not all(isinstance(i, int) for i in spec):
                raise ValueError(
                    f"Branch indices must be integers, got: {spec}"
                )
            return spec
        elif isinstance(spec, dict):
            if "branches" in spec:
                return cls._parse_branch_spec(spec["branches"])
            else:
                return "all"
        else:
            raise ValueError(
                f"Invalid branch specification: {spec}. "
                f"Expected 'all', list of indices, or dict with 'branches' key."
            )

    @classmethod
    def _parse_predictions_spec(
        cls,
        config: MergeConfig,
        pred_spec: Union[str, List, Dict]
    ) -> MergeConfig:
        """Parse predictions specification.

        Handles:
            - "all": All predictions from all branches
            - [0, 1, 2]: Simple branch indices (legacy)
            - [{"branch": 0, ...}]: Per-branch configuration (advanced)
            - {"branches": [...], "models": [...], ...}: Dict format

        Args:
            config: MergeConfig to update
            pred_spec: Predictions specification

        Returns:
            Updated MergeConfig.
        """
        if pred_spec == "all" or pred_spec is True:
            config.prediction_branches = "all"
            return config

        elif isinstance(pred_spec, list):
            # Check if it's a list of branch configs or branch indices
            if len(pred_spec) == 0:
                raise ValueError("Predictions branch list cannot be empty")

            # Detect if this is a list of per-branch configs (dicts with keys)
            # vs a list of branch indices (integers)
            has_dicts = any(isinstance(item, dict) for item in pred_spec)

            if has_dicts:
                # Per-branch configuration: all items must be dicts with 'branch' key
                config.prediction_configs = [
                    cls._parse_branch_prediction_config(item)
                    for item in pred_spec
                ]
            else:
                # Legacy: list of branch indices
                if not all(isinstance(i, int) for i in pred_spec):
                    raise ValueError(
                        f"Prediction branch indices must be integers, got: {pred_spec}"
                    )
                config.prediction_branches = pred_spec
            return config

        elif isinstance(pred_spec, dict):
            # Dict format with branches, models, proba keys
            if "branches" in pred_spec:
                config.prediction_branches = cls._parse_branch_spec(
                    pred_spec["branches"]
                )
            if "models" in pred_spec:
                config.model_filter = pred_spec["models"]
            if "proba" in pred_spec:
                config.use_proba = pred_spec["proba"]
            return config

        else:
            raise ValueError(
                f"Invalid predictions specification: {pred_spec}. "
                f"Expected 'all', list of indices, list of configs, or dict."
            )

    @classmethod
    def _parse_branch_prediction_config(
        cls,
        item: Dict[str, Any]
    ) -> BranchPredictionConfig:
        """Parse a single per-branch prediction configuration.

        Args:
            item: Dict with 'branch' key and optional select, metric, aggregate, etc.

        Returns:
            BranchPredictionConfig instance.

        Raises:
            ValueError: If 'branch' key is missing.
        """
        if "branch" not in item:
            raise ValueError(
                f"Per-branch prediction config must have 'branch' key, "
                f"got: {list(item.keys())}"
            )

        return BranchPredictionConfig(
            branch=item["branch"],
            select=item.get("select", "all"),
            metric=item.get("metric"),
            aggregate=item.get("aggregate", "separate"),
            weight_metric=item.get("weight_metric"),
            proba=item.get("proba", False),
            sources=item.get("sources", "all"),
        )


@register_controller
class MergeController(OperatorController):
    """Controller for merging branch outputs and exiting branch mode.

    This controller is the CORE PRIMITIVE for branch combination. It:
    1. Collects features and/or predictions from specified branches
    2. Performs horizontal concatenation of features
    3. Performs OOF reconstruction for predictions (mandatory unless unsafe=True)
    4. Creates a unified "merged" processing in the dataset
    5. ALWAYS clears branch contexts and exits branch mode

    Supported Keywords:
        - "merge": Branch merging (features/predictions/both)
        - "merge_sources": Source merging (multi-source datasets) [Phase 9]
        - "merge_predictions": Prediction-only late fusion [Phase 9]

    OOF Safety:
        When predictions are merged, OOF reconstruction is MANDATORY by default.
        This prevents data leakage when the merged output is used for training.
        Set `unsafe=True` to disable OOF (generates prominent warnings).

    Relationship to MetaModel:
        MetaModel internally uses MergeController for data preparation, then
        trains the meta-learner. Users can achieve the same result with:
            {"merge": "predictions"}, {"model": Ridge()}
        which is equivalent to:
            {"model": MetaModel(Ridge())}

    Attributes:
        priority: Controller priority (5 = same as BranchController).
        SUPPORTED_KEYWORDS: Set of keywords this controller handles.
    """

    priority = 5
    SUPPORTED_KEYWORDS = {"merge", "merge_sources", "merge_predictions"}

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Check if the step matches the merge controller.

        Args:
            step: Original step configuration
            operator: Deserialized operator
            keyword: Step keyword

        Returns:
            True if keyword is one of the supported merge keywords.
        """
        return keyword in cls.SUPPORTED_KEYWORDS

    @classmethod
    def use_multi_source(cls) -> bool:
        """Merge controller supports multi-source datasets."""
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Merge controller should execute in prediction mode."""
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
        """Execute the merge step with keyword dispatch.

        Dispatches to appropriate handler based on the step keyword:
        - "merge": Branch merging (features/predictions/both)
        - "merge_sources": Source merging (Phase 9, not yet implemented)
        - "merge_predictions": Prediction-only late fusion (Phase 9, not yet implemented)

        Phase 2 implementation provides:
        - Configuration parsing
        - Branch validation
        - Branch mode exit
        - Keyword dispatch framework

        Subsequent phases will add:
        - Feature collection (Phase 3)
        - Prediction OOF reconstruction (Phase 4)
        - Per-branch selection/aggregation (Phase 5)
        - Source merge implementation (Phase 9)

        Args:
            step_info: Parsed step containing merge configuration
            dataset: Dataset to operate on
            context: Pipeline execution context
            runtime_context: Runtime infrastructure context
            source: Data source index
            mode: Execution mode ("train" or "predict")
            loaded_binaries: Pre-loaded binary objects for prediction mode
            prediction_store: External prediction store for model predictions

        Returns:
            Tuple of (updated_context, StepOutput)

        Raises:
            ValueError: If not in branch mode or configuration is invalid.
            NotImplementedError: If merge_sources or merge_predictions called (Phase 9).
        """
        # Determine which keyword was used
        keyword = step_info.keyword

        # Dispatch to appropriate handler
        if keyword == "merge":
            return self._execute_branch_merge(
                step_info, dataset, context, runtime_context,
                source, mode, loaded_binaries, prediction_store
            )
        elif keyword == "merge_sources":
            return self._execute_source_merge(
                step_info, dataset, context, runtime_context,
                source, mode, loaded_binaries, prediction_store
            )
        elif keyword == "merge_predictions":
            return self._execute_prediction_merge(
                step_info, dataset, context, runtime_context,
                source, mode, loaded_binaries, prediction_store
            )
        else:
            raise ValueError(
                f"Unknown merge keyword: '{keyword}'. "
                f"Supported: {self.SUPPORTED_KEYWORDS}"
            )

    def _execute_branch_merge(
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
        """Execute branch merge operation.

        Combines outputs from multiple branches and exits branch mode.

        Phase 8 Enhancement:
        In prediction mode, if branch_contexts are not available (because branches
        were already processed), we reconstruct the merge from loaded metadata.
        The merge step doesn't persist binary artifacts - it combines features/predictions
        that were already transformed by upstream branch steps.

        Args:
            step_info: Parsed step containing merge configuration
            dataset: Dataset to operate on
            context: Pipeline execution context
            runtime_context: Runtime infrastructure context
            source: Data source index
            mode: Execution mode ("train" or "predict")
            loaded_binaries: Pre-loaded binary objects for prediction mode
            prediction_store: External prediction store for model predictions

        Returns:
            Tuple of (updated_context, StepOutput)
        """
        # Parse configuration
        raw_config = step_info.original_step.get("merge")
        config = MergeConfigParser.parse(raw_config)

        # Check for source_branch mode (different from regular branch mode)
        in_source_branch_mode = context.custom.get("in_source_branch_mode", False)
        source_branch_contexts = context.custom.get("source_branch_contexts", [])

        if in_source_branch_mode:
            return self._execute_source_branch_merge(
                step_info=step_info,
                dataset=dataset,
                context=context,
                runtime_context=runtime_context,
                source=source,
                mode=mode,
                config=config,
                source_contexts=source_branch_contexts,
                loaded_binaries=loaded_binaries,
                prediction_store=prediction_store,
            )

        # Validate branch mode
        branch_contexts = context.custom.get("branch_contexts", [])
        in_branch_mode = context.custom.get("in_branch_mode", False)

        # Phase 8: Handle prediction mode without branch_contexts
        # In predict mode, branches are processed but contexts may not be available
        # because the executor has already iterated through branches. We handle this
        # by checking if we're in predict mode and if branch_contexts are empty.
        if mode in ("predict", "explain") and not branch_contexts and not in_branch_mode:
            return self._execute_branch_merge_predict_mode(
                step_info=step_info,
                dataset=dataset,
                context=context,
                runtime_context=runtime_context,
                source=source,
                config=config,
                loaded_binaries=loaded_binaries,
                prediction_store=prediction_store,
            )

        if not branch_contexts and not in_branch_mode:
            raise ValueError(
                "merge requires active branch contexts. "
                "Use merge only after a branch step. "
                "[Error: MERGE-E020]"
            )

        n_branches = len(branch_contexts)
        logger.info(f"Merge step: mode={config.get_merge_mode().value}, branches={n_branches}")

        # Phase 2: Detect disjoint sample branches
        # Disjoint branches (from metadata_partitioner or sample_partitioner) require
        # special merge logic: row concatenation instead of horizontal concatenation
        disjoint_analysis = detect_disjoint_branches(branch_contexts)

        if disjoint_analysis.is_disjoint:
            logger.info(
                f"  Disjoint sample branching detected: {disjoint_analysis.branch_type.value}, "
                f"{disjoint_analysis.total_samples} total samples across {n_branches} branches"
            )

            # Phase 4: In prediction mode, handle disjoint merge specially
            # Samples were already routed to branches, we just need to
            # collect results back in sample order
            if mode in ("predict", "explain"):
                return self._execute_disjoint_branch_merge_predict_mode(
                    step_info=step_info,
                    dataset=dataset,
                    context=context,
                    runtime_context=runtime_context,
                    source=source,
                    config=config,
                    branch_contexts=branch_contexts,
                    disjoint_analysis=disjoint_analysis,
                    loaded_binaries=loaded_binaries,
                    prediction_store=prediction_store,
                )

            return self._execute_disjoint_branch_merge(
                step_info=step_info,
                dataset=dataset,
                context=context,
                runtime_context=runtime_context,
                source=source,
                mode=mode,
                config=config,
                branch_contexts=branch_contexts,
                disjoint_analysis=disjoint_analysis,
                loaded_binaries=loaded_binaries,
                prediction_store=prediction_store,
            )

        # Validate branch indices
        self._validate_branches(config, branch_contexts)

        # Log configuration (Phase 6: enhanced with asymmetric analysis)
        self._log_config(
            config=config,
            n_branches=n_branches,
            branch_contexts=branch_contexts,
            prediction_store=prediction_store,
            context=context,
        )

        # Collect merged data
        merged_parts = []
        merge_info = {}

        # Phase 3: Feature merging
        if config.collect_features:
            feature_branches = config.get_feature_branches(n_branches)
            # When output_as="sources", preserve the preprocessing dimension (3D layout)
            # Otherwise flatten to 2D for horizontal concatenation
            preserve_preprocessing = config.output_as == "sources"
            features_list, feature_info = self._collect_features(
                dataset=dataset,
                branch_contexts=branch_contexts,
                branch_indices=feature_branches,
                on_missing=config.on_missing,
                on_shape_mismatch=config.on_shape_mismatch,
                preserve_preprocessing=preserve_preprocessing,
            )

            if features_list:
                merged_parts.extend(features_list)
                merge_info["feature_shapes"] = feature_info.get("shapes", [])
                merge_info["feature_branches_used"] = feature_info.get("branches_used", [])
                logger.info(
                    f"  Collected features from {len(features_list)} branches: "
                    f"shapes={feature_info.get('shapes', [])}"
                )

        # Phase 4: Prediction merging
        if config.collect_predictions:
            predictions_array, pred_info = self._collect_predictions(
                dataset=dataset,
                context=context,
                branch_contexts=branch_contexts,
                config=config,
                prediction_store=prediction_store,
                mode=mode,
            )

            if predictions_array is not None and predictions_array.size > 0:
                merged_parts.append(predictions_array)
                merge_info["prediction_shape"] = predictions_array.shape
                merge_info["prediction_models_used"] = pred_info.get("models_used", [])
                merge_info["prediction_branches_used"] = pred_info.get("branches_used", [])
                merge_info["oof_reconstruction"] = pred_info.get("oof_reconstruction", True)
                logger.info(
                    f"  Collected predictions: shape={predictions_array.shape}, "
                    f"models={pred_info.get('models_used', [])}"
                )

        # Include original pre-branch features if requested
        if config.include_original:
            original_features = self._get_original_features(dataset, context)
            if original_features is not None:
                # Prepend original features
                merged_parts.insert(0, original_features)
                merge_info["include_original"] = True
                merge_info["original_shape"] = original_features.shape
                logger.info(
                    f"  Prepended original features: shape={original_features.shape}"
                )

        # Check if this is a source branch merge (branches came from source_branch)
        is_source_branch_merge = context.custom.get("in_source_branch_mode", False)

        # Handle output_as strategy
        if merged_parts:
            if config.output_as == "sources":
                # Each branch becomes a separate source
                # For source_branch: restore original sources with new features
                # For regular branch: create sources from branches
                merge_info["merged_shapes"] = [p.shape for p in merged_parts]

                for idx, part in enumerate(merged_parts):
                    source_name = f"merged_{idx}"
                    if config.source_names and idx < len(config.source_names):
                        source_name = config.source_names[idx]
                    elif is_source_branch_merge and idx < len(branch_contexts):
                        # Use original source name for source_branch merges
                        source_name = branch_contexts[idx].get("name", f"source_{idx}")

                    dataset.add_merged_features(
                        features=part,
                        processing_name=source_name,
                        source=idx
                    )
                    logger.info(f"  Source {idx} ({source_name}): shape={part.shape}")

                logger.info(f"  Merged {len(merged_parts)} branches → {len(merged_parts)} sources")

            elif config.output_as == "dict":
                # Store as structured dictionary in context for multi-input models
                merged_dict = {}
                for idx, part in enumerate(merged_parts):
                    source_name = f"branch_{idx}"
                    if config.source_names and idx < len(config.source_names):
                        source_name = config.source_names[idx]
                    elif is_source_branch_merge and idx < len(branch_contexts):
                        source_name = branch_contexts[idx].get("name", f"source_{idx}")
                    merged_dict[source_name] = part

                merge_info["merged_dict_keys"] = list(merged_dict.keys())

                # Store in context for downstream multi-input models
                result_context = context.copy()
                result_context.custom["merged_sources_dict"] = merged_dict
                logger.info(f"  Merged as dict with keys: {list(merged_dict.keys())}")

            else:  # output_as == "features" - legacy behavior
                # Concatenate all parts horizontally into single feature matrix
                merged_features = np.concatenate(merged_parts, axis=1)
                merge_info["merged_shape"] = merged_features.shape
                logger.info(f"  Final merged shape: {merged_features.shape}")

                # Store merged features in dataset
                processing_name = "merged"
                if config.source_names and len(config.source_names) > 0:
                    processing_name = config.source_names[0]

                dataset.add_merged_features(
                    features=merged_features,
                    processing_name=processing_name,
                    source=0  # Primary source for merged features
                )

                # Remove other sources - output_as="features" consolidates to single source
                if dataset.features_sources() > 1:
                    dataset.keep_sources(0)
                    logger.info(f"  Consolidated to single source with shape {merged_features.shape}")
        else:
            logger.warning(
                "No features collected during merge. "
                "Dataset features unchanged."
            )

        # ALWAYS exit branch mode (both regular and source_branch)
        result_context = context.copy()
        result_context.custom["branch_contexts"] = []
        result_context.custom["in_branch_mode"] = False
        result_context.custom["source_branch_contexts"] = []
        result_context.custom["in_source_branch_mode"] = False

        # Update context processing to match the new dataset processing names
        # This is critical for subsequent transformers to correctly identify which
        # processings to operate on after a merge
        n_sources = dataset.features_sources()
        new_processing = []
        for sd_idx in range(n_sources):
            src_processings = list(dataset.features_processings(sd_idx))
            new_processing.append(src_processings)
        result_context = result_context.with_processing(new_processing)

        # Build metadata with serialized config for prediction mode reproducibility
        metadata = {
            "merge_mode": config.get_merge_mode().value,
            "feature_branches": (
                config.get_feature_branches(n_branches)
                if config.collect_features else []
            ),
            "prediction_branches": (
                [pc.branch for pc in config.get_prediction_configs(n_branches)]
                if config.collect_predictions else []
            ),
            "include_original": config.include_original,
            "output_as": config.output_as,
            # Phase 8: Store serialized config for prediction mode
            "merge_config": config.to_dict(),
            **merge_info,  # Include merge details
        }

        # Add unsafe warning to metadata if applicable
        if config.unsafe:
            metadata["unsafe_merge"] = True
            logger.warning(
                "⚠️ UNSAFE MERGE: OOF reconstruction disabled for predictions. "
                "Training predictions are used directly, causing DATA LEAKAGE. "
                "Do NOT use for final model evaluation. "
                "Set unsafe=False (default) for production pipelines. "
                "[Error: MERGE-E025]"
            )

        logger.success(
            f"Merge step completed: exited branch mode. "
            f"Features={config.collect_features}, Predictions={config.collect_predictions}"
            f"{' [UNSAFE]' if config.unsafe else ''}"
        )

        return result_context, StepOutput(metadata=metadata)

    def _execute_source_branch_merge(
        self,
        step_info: "ParsedStep",
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int,
        mode: str,
        config: MergeConfig,
        source_contexts: List[Dict[str, Any]],
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple["ExecutionContext", StepOutput]:
        """Execute merge for source_branch mode.

        This handles merging after source_branch, where each source was processed
        independently. Unlike regular branch merge, this collects features from
        the dataset's sources directly (not from branch snapshots).

        Args:
            step_info: Parsed step containing merge configuration
            dataset: Dataset with processed sources
            context: Pipeline execution context
            runtime_context: Runtime infrastructure context
            source: Data source index
            mode: Execution mode ("train" or "predict")
            config: Parsed merge configuration
            source_contexts: List of source context dictionaries
            loaded_binaries: Pre-loaded binary objects for prediction mode
            prediction_store: External prediction store

        Returns:
            Tuple of (updated_context, StepOutput)
        """
        n_sources = dataset.n_sources
        logger.info(f"Source branch merge: {n_sources} sources, output_as={config.output_as}")

        # Collect features from each source
        merged_parts = []
        source_shapes = []
        source_names = []

        # When output_as="sources", preserve the preprocessing dimension (3D layout)
        # Otherwise flatten to 2D for horizontal concatenation
        preserve_preprocessing = config.output_as == "sources"
        layout = "3d" if preserve_preprocessing else "2d"

        for src_idx in range(n_sources):
            try:
                # Get features for this source using current processing
                X = dataset.x(
                    selector=context.selector,
                    layout=layout,
                    concat_source=False,
                    include_augmented=True,
                    include_excluded=False
                )

                # X is a list of per-source arrays
                if isinstance(X, list) and src_idx < len(X):
                    features = X[src_idx]
                elif not isinstance(X, list) and src_idx == 0:
                    features = X
                else:
                    logger.warning(f"Source {src_idx} not found in dataset output")
                    continue

                merged_parts.append(features)
                source_shapes.append(features.shape)

                # Get source name from contexts or generate default
                if src_idx < len(source_contexts):
                    name = source_contexts[src_idx].get("source_name", f"source_{src_idx}")
                else:
                    name = f"source_{src_idx}"
                source_names.append(name)

                logger.info(f"  Source {src_idx} ({name}): shape={features.shape}")

            except Exception as e:
                logger.warning(f"Failed to collect features from source {src_idx}: {e}")
                continue

        if not merged_parts:
            logger.warning("No source features collected during merge")
            result_context = context.copy()
            result_context.custom["in_source_branch_mode"] = False
            result_context.custom["source_branch_contexts"] = []
            return result_context, StepOutput(metadata={"error": "no_features"})

        merge_info = {
            "source_shapes": source_shapes,
            "source_names": source_names,
            "n_sources": len(merged_parts),
        }

        # Apply output_as strategy
        if config.output_as == "sources":
            # Keep as separate sources - already in the right format
            # Just store merged info for metadata
            for idx, (part, name) in enumerate(zip(merged_parts, source_names)):
                processing_name = f"merged_{name}"
                if config.source_names and idx < len(config.source_names):
                    processing_name = config.source_names[idx]

                dataset.add_merged_features(
                    features=part,
                    processing_name=processing_name,
                    source=idx
                )
            logger.info(f"  Kept {len(merged_parts)} sources with shapes {source_shapes}")

        elif config.output_as == "dict":
            # Store as dictionary for multi-input models
            merged_dict = {name: part for name, part in zip(source_names, merged_parts)}
            merge_info["merged_dict_keys"] = source_names

            result_context = context.copy()
            result_context.custom["merged_sources_dict"] = merged_dict
            logger.info(f"  Stored as dict with keys: {source_names}")

        else:  # output_as == "features"
            # Concatenate all sources into single feature matrix
            merged_features = np.concatenate(merged_parts, axis=1)
            merge_info["merged_shape"] = merged_features.shape

            dataset.add_merged_features(
                features=merged_features,
                processing_name="merged",
                source=0
            )
            logger.info(f"  Concatenated to shape {merged_features.shape}")

        # Exit source branch mode
        result_context = context.copy()
        result_context.custom["in_source_branch_mode"] = False
        result_context.custom["source_branch_contexts"] = []

        # Update context processing to match the new dataset processing names
        # This is critical for subsequent transformers to correctly identify which
        # processings to operate on
        n_sources = dataset.features_sources()
        new_processing = []
        for sd_idx in range(n_sources):
            src_processings = list(dataset.features_processings(sd_idx))
            new_processing.append(src_processings)
        result_context = result_context.with_processing(new_processing)

        metadata = {
            "source_branch_merge": True,
            "output_as": config.output_as,
            "merge_config": config.to_dict(),
            **merge_info,
        }

        logger.success(f"Source branch merge completed: {len(merged_parts)} sources → output_as={config.output_as}")

        return result_context, StepOutput(metadata=metadata)

    # =========================================================================
    # Phase 2: Disjoint Sample Branch Merging
    # =========================================================================

    def _execute_disjoint_branch_merge(
        self,
        step_info: "ParsedStep",
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int,
        mode: str,
        config: MergeConfig,
        branch_contexts: List[Dict[str, Any]],
        disjoint_analysis: DisjointBranchAnalysis,
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple["ExecutionContext", StepOutput]:
        """Execute merge for disjoint sample branches.

        Disjoint branches partition samples such that each sample exists in
        exactly ONE branch. This requires different merge semantics:

        Feature merge: Validate equal feature dimensions, then concatenate rows
            by sample_id to reconstruct full dataset.

        Prediction merge: Select top-N models per branch (where N is the minimum
            model count or explicitly specified), then reconstruct OOF predictions
            by sample_id.

        Args:
            step_info: Parsed step containing merge configuration
            dataset: Dataset to operate on
            context: Pipeline execution context
            runtime_context: Runtime infrastructure context
            source: Data source index
            mode: Execution mode ("train" or "predict")
            config: Parsed merge configuration with n_columns and select_by
            branch_contexts: List of branch context dictionaries
            disjoint_analysis: Analysis result from detect_disjoint_branches()
            loaded_binaries: Pre-loaded binary objects for prediction mode
            prediction_store: External prediction store

        Returns:
            Tuple of (updated_context, StepOutput)

        Raises:
            ValueError: If feature dimensions differ across branches (for features merge)
            ValueError: If n_columns exceeds minimum model count across branches
        """
        n_branches = len(branch_contexts)
        n_total_samples = disjoint_analysis.total_samples

        logger.info(
            f"Disjoint branch merge: {n_branches} branches, "
            f"{n_total_samples} total samples, "
            f"type={disjoint_analysis.branch_type.value}"
        )

        merge_info: Dict[str, Any] = {
            "disjoint_merge": True,
            "branch_type": disjoint_analysis.branch_type.value,
            "n_branches": n_branches,
            "n_total_samples": n_total_samples,
        }

        merged_features = None
        merged_predictions = None

        # ===== FEATURE MERGE =====
        if config.collect_features:
            merged_features, feature_info = self._collect_disjoint_features(
                dataset=dataset,
                branch_contexts=branch_contexts,
                disjoint_analysis=disjoint_analysis,
                config=config,
            )
            merge_info.update(feature_info)

        # ===== PREDICTION MERGE =====
        if config.collect_predictions:
            merged_predictions, pred_info = self._collect_disjoint_predictions(
                dataset=dataset,
                context=context,
                branch_contexts=branch_contexts,
                disjoint_analysis=disjoint_analysis,
                config=config,
                prediction_store=prediction_store,
                mode=mode,
            )
            merge_info.update(pred_info)

        # Combine merged parts
        merged_parts = []
        if merged_features is not None:
            merged_parts.append(merged_features)
        if merged_predictions is not None:
            merged_parts.append(merged_predictions)

        # Include original features if requested
        if config.include_original:
            original = self._get_original_features(dataset, context)
            if original is not None:
                merged_parts.insert(0, original)
                merge_info["include_original"] = True
                merge_info["original_shape"] = original.shape

        if not merged_parts:
            raise ValueError(
                "Disjoint branch merge resulted in empty output. "
                "Check that branches have features or predictions. "
                "[Error: MERGE-E040]"
            )

        # Validate trainability of merged result
        self._validate_merged_trainability(merged_parts[0], merge_info)

        # Concatenate horizontally if multiple parts
        if len(merged_parts) == 1:
            final_merged = merged_parts[0]
        else:
            final_merged = np.concatenate(merged_parts, axis=1)

        merge_info["merged_shape"] = final_merged.shape
        logger.info(f"  Final merged shape: {final_merged.shape}")

        # Store merged features in dataset
        processing_name = "merged"
        if config.source_names and len(config.source_names) > 0:
            processing_name = config.source_names[0]

        dataset.add_merged_features(
            features=final_merged,
            processing_name=processing_name,
            source=0
        )

        # Remove other sources - disjoint merge consolidates to single source
        if dataset.features_sources() > 1:
            dataset.keep_sources(0)

        # Exit branch mode
        result_context = context.copy()
        result_context.custom["branch_contexts"] = []
        result_context.custom["in_branch_mode"] = False
        result_context.custom["metadata_partitioner_active"] = False
        result_context.custom["sample_partitioner_active"] = False

        # Update context processing
        n_sources = dataset.features_sources()
        new_processing = []
        for sd_idx in range(n_sources):
            src_processings = list(dataset.features_processings(sd_idx))
            new_processing.append(src_processings)
        result_context = result_context.with_processing(new_processing)

        # Build metadata
        metadata = {
            "merge_mode": config.get_merge_mode().value,
            "disjoint_merge": True,
            "branch_type": disjoint_analysis.branch_type.value,
            "partition_column": disjoint_analysis.partition_column,
            "merge_config": config.to_dict(),
            **merge_info,
        }

        logger.success(
            f"Disjoint branch merge completed: {n_branches} branches → "
            f"shape={final_merged.shape}"
        )

        return result_context, StepOutput(metadata=metadata)

    def _execute_disjoint_branch_merge_predict_mode(
        self,
        step_info: "ParsedStep",
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int,
        config: MergeConfig,
        branch_contexts: List[Dict[str, Any]],
        disjoint_analysis: DisjointBranchAnalysis,
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple["ExecutionContext", StepOutput]:
        """Execute disjoint merge in prediction mode.

        In prediction mode, samples have already been routed to their
        respective branches and processed. This method reconstructs
        the merged output by collecting features/predictions from each
        branch in sample order.

        For feature merge:
            - Each branch has processed its subset of samples
            - Collect features and reconstruct in original sample order

        For prediction merge:
            - Models have already generated predictions for their samples
            - Collect predictions and reconstruct in original sample order

        Args:
            step_info: Parsed step info
            dataset: Dataset with branch-processed samples
            context: Execution context
            runtime_context: Runtime context
            source: Source index
            config: Merge configuration
            branch_contexts: List of branch context dicts
            disjoint_analysis: Disjoint branch analysis
            loaded_binaries: Not used (merge has no artifacts)
            prediction_store: Prediction storage

        Returns:
            Tuple of (updated_context, StepOutput)
        """
        n_branches = len(branch_contexts)
        n_total_samples = disjoint_analysis.total_samples

        logger.info(
            f"Disjoint branch merge (predict mode): {n_branches} branches, "
            f"{n_total_samples} samples"
        )

        merge_info: Dict[str, Any] = {
            "disjoint_merge": True,
            "prediction_mode": True,
            "branch_type": disjoint_analysis.branch_type.value,
            "n_branches": n_branches,
            "n_total_samples": n_total_samples,
        }

        # For feature merge, reconstruct features from branch snapshots
        if config.collect_features:
            try:
                merged_features, feature_info = self._collect_disjoint_features(
                    dataset=dataset,
                    branch_contexts=branch_contexts,
                    disjoint_analysis=disjoint_analysis,
                    config=config,
                )
                merge_info.update(feature_info)

                # Store merged features
                processing_name = "merged"
                if config.source_names and len(config.source_names) > 0:
                    processing_name = config.source_names[0]

                dataset.add_merged_features(
                    features=merged_features,
                    processing_name=processing_name,
                    source=0
                )

                logger.info(f"  Merged features (predict): shape={merged_features.shape}")

            except Exception as e:
                logger.warning(f"Could not merge disjoint features in predict mode: {e}")

        # For prediction merge, collect predictions from prediction store
        if config.collect_predictions and prediction_store is not None:
            try:
                # Get predictions from test partition (predict mode)
                predictions_array = self._collect_disjoint_predictions_predict_mode(
                    dataset=dataset,
                    context=context,
                    branch_contexts=branch_contexts,
                    disjoint_analysis=disjoint_analysis,
                    config=config,
                    prediction_store=prediction_store,
                )

                if predictions_array is not None:
                    merge_info["prediction_shape"] = predictions_array.shape

                    # Add predictions to merged features if also collecting features
                    if config.collect_features:
                        merged_features = np.concatenate([merged_features, predictions_array], axis=1)
                        dataset.add_merged_features(
                            features=merged_features,
                            processing_name="merged",
                            source=0
                        )
                    else:
                        dataset.add_merged_features(
                            features=predictions_array,
                            processing_name="merged_predictions",
                            source=0
                        )

                    logger.info(f"  Merged predictions (predict): shape={predictions_array.shape}")

            except Exception as e:
                logger.warning(f"Could not collect predictions in predict mode: {e}")

        # Exit branch mode
        result_context = context.copy()
        result_context.custom["branch_contexts"] = []
        result_context.custom["in_branch_mode"] = False
        result_context.custom["metadata_partitioner_active"] = False
        result_context.custom["sample_partitioner_active"] = False

        # Update context processing
        n_sources = dataset.features_sources()
        new_processing = []
        for sd_idx in range(n_sources):
            src_processings = list(dataset.features_processings(sd_idx))
            new_processing.append(src_processings)
        result_context = result_context.with_processing(new_processing)

        # Build metadata
        metadata = {
            "merge_mode": config.get_merge_mode().value,
            "disjoint_merge": True,
            "prediction_mode": True,
            "branch_type": disjoint_analysis.branch_type.value,
            "partition_column": disjoint_analysis.partition_column,
            "merge_config": config.to_dict(),
            **merge_info,
        }

        logger.success(
            f"Disjoint branch merge (predict mode) completed: {n_branches} branches"
        )

        return result_context, StepOutput(metadata=metadata)

    def _collect_disjoint_predictions_predict_mode(
        self,
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        branch_contexts: List[Dict[str, Any]],
        disjoint_analysis: DisjointBranchAnalysis,
        config: MergeConfig,
        prediction_store: Any,
    ) -> Optional[np.ndarray]:
        """Collect predictions from disjoint branches in predict mode.

        In predict mode, models have already generated predictions for
        their respective sample subsets. This method collects those
        predictions and reconstructs them in original sample order.

        Args:
            dataset: Dataset for sample info
            context: Execution context
            branch_contexts: List of branch contexts
            disjoint_analysis: Disjoint branch analysis
            config: Merge configuration
            prediction_store: Prediction storage

        Returns:
            Merged predictions array or None
        """
        n_total_samples = disjoint_analysis.total_samples
        branch_sample_indices = disjoint_analysis.branch_sample_indices

        # Query prediction store for test partition predictions
        filter_kwargs = {
            'partition': 'test',
            'load_arrays': True,
        }

        predictions = prediction_store.filter_predictions(**filter_kwargs)

        if not predictions:
            logger.debug("No test predictions found in prediction store")
            return None

        # Group predictions by branch
        branch_predictions: Dict[int, List[Dict[str, Any]]] = {}
        for pred in predictions:
            branch_id = pred.get('branch_id', 0)
            if branch_id not in branch_predictions:
                branch_predictions[branch_id] = []
            branch_predictions[branch_id].append(pred)

        # Determine output shape
        # Find number of models/columns from first branch with predictions
        n_columns = 1
        for preds in branch_predictions.values():
            model_names = set(p.get('model_name') for p in preds if p.get('model_name'))
            if model_names:
                n_columns = len(model_names)
                break

        # Initialize output array
        merged = np.full((n_total_samples, n_columns), np.nan)

        # Collect predictions from each branch
        for branch_id, sample_indices in branch_sample_indices.items():
            if branch_id not in branch_predictions:
                logger.debug(f"Branch {branch_id} has no predictions")
                continue

            preds = branch_predictions[branch_id]

            # Group by model
            model_preds: Dict[str, np.ndarray] = {}
            for pred in preds:
                model_name = pred.get('model_name', 'model')
                y_pred = pred.get('y_pred')
                if y_pred is not None:
                    y_pred = np.asarray(y_pred).flatten()
                    if model_name not in model_preds:
                        model_preds[model_name] = y_pred
                    else:
                        # Average if multiple predictions for same model
                        model_preds[model_name] = np.mean(
                            [model_preds[model_name], y_pred], axis=0
                        )

            # Map predictions to output columns
            for col_idx, (model_name, y_pred) in enumerate(model_preds.items()):
                if col_idx >= n_columns:
                    break

                # Map predictions to sample indices
                for local_idx, global_idx in enumerate(sample_indices):
                    if global_idx < n_total_samples and local_idx < len(y_pred):
                        merged[global_idx, col_idx] = y_pred[local_idx]

        # Check for unfilled samples
        nan_count = np.sum(np.isnan(merged))
        if nan_count > 0:
            logger.warning(
                f"Disjoint prediction merge (predict): {nan_count} values are NaN"
            )

        return merged

    def _collect_disjoint_features(
        self,
        dataset: "SpectroDataset",
        branch_contexts: List[Dict[str, Any]],
        disjoint_analysis: DisjointBranchAnalysis,
        config: MergeConfig,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Collect features from disjoint sample branches.

        For disjoint branches, features are concatenated VERTICALLY (row-wise)
        by sample_id, not horizontally. All branches must produce the same
        feature dimension or an error is raised.

        Args:
            dataset: Dataset for sample information
            branch_contexts: List of branch context dictionaries
            disjoint_analysis: Analysis of disjoint branches
            config: Merge configuration

        Returns:
            Tuple of (merged_features, info_dict) where merged_features has
            shape (n_total_samples, n_features).

        Raises:
            ValueError: If feature dimensions differ across branches.
        """
        n_total_samples = disjoint_analysis.total_samples
        feature_dims: Dict[str, int] = {}
        branch_features: Dict[int, Tuple[np.ndarray, List[int]]] = {}

        # Collect features from each branch
        for bc in branch_contexts:
            branch_id = bc["branch_id"]
            branch_name = bc.get("name", f"branch_{branch_id}")
            sample_indices = disjoint_analysis.branch_sample_indices.get(branch_id, [])

            if not sample_indices:
                logger.warning(f"Branch {branch_name} has no sample indices, skipping")
                continue

            # Extract features from branch snapshot
            snapshot = bc.get("features_snapshot")
            if snapshot is None:
                raise ValueError(
                    f"Branch '{branch_name}' has no feature snapshot for disjoint merge. "
                    f"[Error: MERGE-E041]"
                )

            try:
                features = self._extract_features_from_snapshot(
                    snapshot=snapshot,
                    expected_samples=len(sample_indices),
                    branch_idx=branch_id,
                    layout="2d",
                )
            except ValueError as e:
                raise ValueError(
                    f"Failed to extract features from branch '{branch_name}': {e}. "
                    f"[Error: MERGE-E041]"
                ) from e

            feature_dim = features.shape[1]
            feature_dims[branch_name] = feature_dim
            branch_features[branch_id] = (features, sample_indices)

            logger.debug(
                f"  Branch '{branch_name}': {len(sample_indices)} samples, "
                f"{feature_dim} features"
            )

        # Validate feature dimensions are equal
        unique_dims = set(feature_dims.values())
        if len(unique_dims) > 1:
            dims_str = ", ".join(f"'{k}': {v}" for k, v in feature_dims.items())
            raise ValueError(
                f"Cannot merge features from disjoint branches with different "
                f"feature dimensions: {{{dims_str}}}. "
                f"Ensure all branches apply identical transformations. "
                f"[Error: MERGE-E042]"
            )

        if not unique_dims:
            raise ValueError(
                "No features collected from any disjoint branch. "
                "[Error: MERGE-E041]"
            )

        n_features = unique_dims.pop()

        # Reconstruct full feature matrix by sample_id
        merged = np.full((n_total_samples, n_features), np.nan)

        for branch_id, (features, sample_indices) in branch_features.items():
            for local_idx, global_idx in enumerate(sample_indices):
                if global_idx < n_total_samples:
                    merged[global_idx] = features[local_idx]

        # Check for any unfilled samples
        nan_rows = np.any(np.isnan(merged), axis=1)
        n_unfilled = np.sum(nan_rows)
        if n_unfilled > 0:
            logger.warning(
                f"Disjoint feature merge: {n_unfilled} samples have NaN values. "
                f"This may indicate sample coverage gaps."
            )

        # Phase 3: Build comprehensive metadata for feature merge
        # Build per-branch info (for features, no model selection)
        branches_info: Dict[str, DisjointBranchInfo] = {}
        for branch_id, (features, sample_indices) in branch_features.items():
            # Get branch name from feature_dims keys (we saved branch_name -> dim)
            branch_name = None
            for bc in branch_contexts:
                if bc["branch_id"] == branch_id:
                    branch_name = bc.get("name", f"branch_{branch_id}")
                    break
            if branch_name is None:
                branch_name = f"branch_{branch_id}"

            branches_info[branch_name] = DisjointBranchInfo(
                n_samples=len(sample_indices),
                sample_ids=sample_indices,
                n_models_original=0,  # Feature merge, no models
                n_models_selected=0,
                selected_models=[],
                dropped_models=[],
            )

        # Build feature merge metadata
        disjoint_metadata = DisjointMergeMetadata(
            merge_type="disjoint_samples",
            n_columns=0,  # 0 for feature merge (not prediction columns)
            select_by="",  # Not applicable for feature merge
            branches=branches_info,
            column_mapping={},  # Not applicable for feature merge
            is_heterogeneous=False,
            feature_dim=n_features,
        )

        # Phase 3: Use structured logging from metadata
        disjoint_metadata.log_summary(logger.info)

        info = {
            "feature_dims": feature_dims,
            "feature_dim": n_features,
            "feature_branches_used": list(branch_features.keys()),
            "feature_merged_shape": merged.shape,
            # Phase 3: Add structured metadata
            "disjoint_metadata": disjoint_metadata.to_dict(),
        }

        return merged, info

    def _collect_disjoint_predictions(
        self,
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        branch_contexts: List[Dict[str, Any]],
        disjoint_analysis: DisjointBranchAnalysis,
        config: MergeConfig,
        prediction_store: Optional[Any],
        mode: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Collect predictions from disjoint sample branches.

        For disjoint branches, predictions are collected per-branch and then
        reconstructed by sample_id. When branches have different model counts,
        we select top-N models from each branch based on the selection criterion.

        Algorithm:
        1. Determine N (output column count) from n_columns or min(model_counts)
        2. Select top-N models per branch based on select_by criterion
        3. Reconstruct OOF predictions by sample_id

        Args:
            dataset: Dataset for sample information
            context: Execution context
            branch_contexts: List of branch context dictionaries
            disjoint_analysis: Analysis of disjoint branches
            config: Merge configuration with n_columns and select_by
            prediction_store: Prediction storage
            mode: Execution mode

        Returns:
            Tuple of (merged_predictions, info_dict) where merged_predictions
            has shape (n_total_samples, N).

        Raises:
            ValueError: If n_columns exceeds minimum model count
            ValueError: If no predictions found in any branch
        """
        if prediction_store is None:
            raise ValueError(
                "prediction_store is required for disjoint prediction merge. "
                "[Error: MERGE-E043]"
            )

        n_total_samples = disjoint_analysis.total_samples
        select_by = config.select_by

        # Step 1: Discover models in each branch and their scores
        branch_models: Dict[int, List[Dict[str, Any]]] = {}
        branch_sample_indices = disjoint_analysis.branch_sample_indices

        for bc in branch_contexts:
            branch_id = bc["branch_id"]
            branch_name = bc.get("name", f"branch_{branch_id}")

            # Discover models in this branch
            model_names = self._discover_branch_models(
                prediction_store=prediction_store,
                branch_id=branch_id,
                context=context,
                model_filter=config.model_filter,
            )

            if not model_names:
                logger.warning(f"Branch '{branch_name}' has no models, skipping")
                continue

            # Get model scores for ranking
            model_infos = []
            for model_name in model_names:
                score = self._get_model_score(
                    prediction_store=prediction_store,
                    model_name=model_name,
                    branch_id=branch_id,
                    metric=select_by,
                    context=context,
                )
                model_infos.append({
                    "name": model_name,
                    "score": score,
                    "branch_id": branch_id,
                    "branch_name": branch_name,
                })

            branch_models[branch_id] = model_infos
            logger.debug(
                f"  Branch '{branch_name}': {len(model_infos)} models"
            )

        if not branch_models:
            raise ValueError(
                "No model predictions found in any disjoint branch. "
                "[Error: MERGE-E043]"
            )

        # Step 2: Determine N (output column count)
        model_counts = {bid: len(models) for bid, models in branch_models.items()}
        min_model_count = min(model_counts.values())
        max_model_count = max(model_counts.values())

        if config.n_columns is not None:
            n_columns = config.n_columns
            if n_columns > min_model_count:
                raise ValueError(
                    f"n_columns={n_columns} exceeds minimum model count "
                    f"({min_model_count}) across branches. "
                    f"Model counts: {model_counts}. "
                    f"[Error: MERGE-E044]"
                )
        else:
            n_columns = min_model_count

        logger.info(
            f"  Disjoint prediction merge: N={n_columns} columns "
            f"(model counts: {model_counts}, select_by='{select_by}')"
        )

        # Step 3: Select top-N models per branch
        selected_per_branch: Dict[int, List[Dict[str, Any]]] = {}
        dropped_per_branch: Dict[int, List[Dict[str, Any]]] = {}
        column_mapping: Dict[int, Dict[str, str]] = {i: {} for i in range(n_columns)}

        for branch_id, model_infos in branch_models.items():
            branch_name = model_infos[0]["branch_name"] if model_infos else f"branch_{branch_id}"

            if len(model_infos) == n_columns:
                # No selection needed
                selected = model_infos
                dropped = []
            else:
                # Rank by score and select top-N
                if select_by == "order":
                    # First N in definition order
                    selected = model_infos[:n_columns]
                    dropped = model_infos[n_columns:]
                elif select_by in ("r2",):
                    # Higher is better
                    sorted_models = sorted(
                        model_infos,
                        key=lambda m: m["score"] if m["score"] is not None else float('-inf'),
                        reverse=True
                    )
                    selected = sorted_models[:n_columns]
                    dropped = sorted_models[n_columns:]
                else:
                    # Lower is better (mse, rmse, mae)
                    sorted_models = sorted(
                        model_infos,
                        key=lambda m: m["score"] if m["score"] is not None else float('inf'),
                    )
                    selected = sorted_models[:n_columns]
                    dropped = sorted_models[n_columns:]

            selected_per_branch[branch_id] = selected
            dropped_per_branch[branch_id] = dropped

            # Build column mapping
            for col_idx, model_info in enumerate(selected):
                column_mapping[col_idx][branch_name] = model_info["name"]

        # Step 4: Collect OOF predictions for selected models
        merged = np.full((n_total_samples, n_columns), np.nan)

        for branch_id, selected_models in selected_per_branch.items():
            sample_indices = branch_sample_indices.get(branch_id, [])

            for col_idx, model_info in enumerate(selected_models):
                model_name = model_info["name"]

                # Get OOF predictions for this model
                oof_predictions = self._get_branch_oof_predictions(
                    dataset=dataset,
                    context=context,
                    prediction_store=prediction_store,
                    model_name=model_name,
                    branch_id=branch_id,
                    sample_indices=sample_indices,
                    mode=mode,
                )

                if oof_predictions is not None:
                    for local_idx, global_idx in enumerate(sample_indices):
                        if global_idx < n_total_samples and local_idx < len(oof_predictions):
                            merged[global_idx, col_idx] = oof_predictions[local_idx]

        # Check for unfilled predictions
        nan_count = np.sum(np.isnan(merged))
        if nan_count > 0:
            logger.warning(
                f"Disjoint prediction merge: {nan_count} values are NaN "
                f"({100 * nan_count / merged.size:.1f}% of total). "
                f"This may indicate incomplete OOF coverage."
            )

        # Phase 3: Build comprehensive metadata using DisjointMergeMetadata
        # Build per-branch info
        branches_info: Dict[str, DisjointBranchInfo] = {}
        for branch_id, selected_models in selected_per_branch.items():
            # Get branch name
            branch_name = selected_models[0]["branch_name"] if selected_models else f"branch_{branch_id}"

            # Get sample indices for this branch
            sample_indices = branch_sample_indices.get(branch_id, [])

            # Build selected model details with column mapping
            selected_model_details = []
            for col_idx, model_info in enumerate(selected_models):
                selected_model_details.append({
                    "name": model_info["name"],
                    "score": model_info["score"],
                    "column": col_idx,
                })

            # Build dropped model details
            dropped_model_details = []
            for model_info in dropped_per_branch.get(branch_id, []):
                dropped_model_details.append({
                    "name": model_info["name"],
                    "score": model_info["score"],
                })

            branches_info[branch_name] = DisjointBranchInfo(
                n_samples=len(sample_indices),
                sample_ids=sample_indices,
                n_models_original=model_counts.get(branch_id, 0),
                n_models_selected=len(selected_models),
                selected_models=selected_model_details,
                dropped_models=dropped_model_details,
            )

        # Check if column mapping is heterogeneous (different models in same column for different branches)
        is_heterogeneous = False
        for col_idx, mapping in column_mapping.items():
            if len(set(mapping.values())) > 1:
                is_heterogeneous = True
                break

        # Build the full metadata object
        disjoint_metadata = DisjointMergeMetadata(
            merge_type="disjoint_samples",
            n_columns=n_columns,
            select_by=select_by,
            branches=branches_info,
            column_mapping=column_mapping,
            is_heterogeneous=is_heterogeneous,
        )

        # Phase 3: Use structured logging from metadata
        disjoint_metadata.log_summary(logger.info)
        if is_heterogeneous or max_model_count > min_model_count:
            disjoint_metadata.log_warnings(logger.warning)

        # Build info dict with both legacy fields and new metadata
        info = {
            "prediction_n_columns": n_columns,
            "prediction_select_by": select_by,
            "prediction_model_counts": model_counts,
            "prediction_branches_used": list(selected_per_branch.keys()),
            "prediction_column_mapping": column_mapping,
            "prediction_merged_shape": merged.shape,
            "selected_models": {
                bid: [m["name"] for m in models]
                for bid, models in selected_per_branch.items()
            },
            "dropped_models": {
                bid: [m["name"] for m in models]
                for bid, models in dropped_per_branch.items()
                if models
            },
            # Phase 3: Add structured metadata
            "disjoint_metadata": disjoint_metadata.to_dict(),
        }

        logger.info(
            f"  Collected predictions: {len(selected_per_branch)} branches → "
            f"shape={merged.shape}"
        )

        return merged, info

    def _get_model_score(
        self,
        prediction_store: Any,
        model_name: str,
        branch_id: int,
        metric: str,
        context: "ExecutionContext",
    ) -> Optional[float]:
        """Get validation score for a model.

        Args:
            prediction_store: Prediction storage
            model_name: Name of the model
            branch_id: Branch ID
            metric: Metric name (mse, rmse, mae, r2)
            context: Execution context

        Returns:
            Score value, or None if not available
        """
        try:
            current_step = getattr(context.state, 'step_number', float('inf'))

            filter_kwargs = {
                'model_name': model_name,
                'branch_id': branch_id,
                'partition': 'val',
                'load_arrays': False,
            }

            predictions = prediction_store.filter_predictions(**filter_kwargs)
            predictions = [
                p for p in predictions
                if p.get('step_idx', 0) < current_step
            ]

            if not predictions:
                return None

            # Aggregate scores across folds
            scores = []
            for pred in predictions:
                metrics = pred.get('metrics', {})
                if metric in metrics:
                    scores.append(metrics[metric])
                # Try uppercase/lowercase variants
                elif metric.upper() in metrics:
                    scores.append(metrics[metric.upper()])
                elif metric.lower() in metrics:
                    scores.append(metrics[metric.lower()])

            if scores:
                return float(np.mean(scores))

            return None

        except Exception as e:
            logger.debug(f"Failed to get score for {model_name}: {e}")
            return None

    def _get_branch_oof_predictions(
        self,
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        prediction_store: Any,
        model_name: str,
        branch_id: int,
        sample_indices: List[int],
        mode: str,
    ) -> Optional[np.ndarray]:
        """Get OOF predictions for a model in a disjoint branch.

        For disjoint branches, we need predictions only for the samples
        in this branch's partition.

        Args:
            dataset: Dataset for sample info
            context: Execution context
            prediction_store: Prediction storage
            model_name: Model name
            branch_id: Branch ID
            sample_indices: Sample indices for this branch
            mode: Execution mode

        Returns:
            1D array of predictions for the branch's samples, or None
        """
        try:
            current_step = getattr(context.state, 'step_number', float('inf'))
            n_branch_samples = len(sample_indices)

            # Get validation predictions (OOF)
            filter_kwargs = {
                'model_name': model_name,
                'branch_id': branch_id,
                'partition': 'val',
                'load_arrays': True,
            }

            predictions = prediction_store.filter_predictions(**filter_kwargs)
            predictions = [
                p for p in predictions
                if p.get('step_idx', 0) < current_step
            ]

            if not predictions:
                logger.debug(f"No OOF predictions for {model_name} in branch {branch_id}")
                return None

            # Build sample_id to prediction mapping
            sample_to_pred: Dict[int, List[float]] = {}

            for pred in predictions:
                y_pred = pred.get('y_pred')
                pred_sample_indices = pred.get('sample_indices')

                if y_pred is None:
                    continue

                y_pred = np.asarray(y_pred).flatten()

                if pred_sample_indices is not None:
                    if hasattr(pred_sample_indices, 'tolist'):
                        pred_sample_indices = pred_sample_indices.tolist()
                    for i, sid in enumerate(pred_sample_indices):
                        if i < len(y_pred):
                            if sid not in sample_to_pred:
                                sample_to_pred[sid] = []
                            sample_to_pred[sid].append(y_pred[i])

            # Build output array aligned with branch sample indices
            result = np.full(n_branch_samples, np.nan)
            for local_idx, global_idx in enumerate(sample_indices):
                if global_idx in sample_to_pred:
                    # Average if multiple predictions (across folds)
                    result[local_idx] = np.mean(sample_to_pred[global_idx])

            return result

        except Exception as e:
            logger.warning(f"Failed to get OOF predictions for {model_name}: {e}")
            return None

    def _validate_merged_trainability(
        self,
        merged: np.ndarray,
        merge_info: Dict[str, Any],
    ) -> None:
        """Validate that merged predictions can train a meta-model.

        Checks for:
        1. Non-finite values (NaN, Inf)
        2. Minimum sample count

        Args:
            merged: Merged prediction/feature array
            merge_info: Merge info dict (for error context)

        Raises:
            ValueError: If validation fails
        """
        MIN_SAMPLES = 10

        # Check for non-finite values
        non_finite_mask = ~np.isfinite(merged)
        non_finite_count = np.sum(non_finite_mask)

        if non_finite_count > 0:
            non_finite_pct = 100 * non_finite_count / merged.size
            if non_finite_pct > 50:
                raise ValueError(
                    f"Merged predictions contain {non_finite_count} non-finite values "
                    f"({non_finite_pct:.1f}% of total). Cannot train meta-model on invalid data. "
                    f"[Error: MERGE-E045]"
                )
            else:
                logger.warning(
                    f"Merged predictions contain {non_finite_count} non-finite values "
                    f"({non_finite_pct:.1f}%). These will be imputed with column means."
                )
                # Impute NaN with column means
                for col in range(merged.shape[1]):
                    col_data = merged[:, col]
                    mask = ~np.isfinite(col_data)
                    if np.any(mask):
                        col_mean = np.nanmean(col_data)
                        if np.isfinite(col_mean):
                            merged[mask, col] = col_mean
                        else:
                            merged[mask, col] = 0.0

        # Check minimum samples
        n_samples = merged.shape[0]
        if n_samples < MIN_SAMPLES:
            raise ValueError(
                f"Merged predictions have only {n_samples} samples. "
                f"Minimum {MIN_SAMPLES} required for meta-model training. "
                f"[Error: MERGE-E046]"
            )

    def _validate_branches(
        self,
        config: MergeConfig,
        branch_contexts: List[Dict[str, Any]]
    ) -> None:
        """Validate that specified branch indices exist.

        Args:
            config: Merge configuration
            branch_contexts: Available branch contexts

        Raises:
            ValueError: If any specified branch index is invalid.
        """
        n_branches = len(branch_contexts)
        available_indices = set(range(n_branches))
        available_names = {
            bc.get("name", f"branch_{bc['branch_id']}"): bc["branch_id"]
            for bc in branch_contexts
        }

        # Validate feature branches
        if config.collect_features and config.feature_branches != "all":
            # At this point, feature_branches is List[int] (not "all")
            feature_branch_list = config.feature_branches
            assert isinstance(feature_branch_list, list)  # Type narrowing
            self._validate_branch_indices(
                feature_branch_list,
                available_indices,
                available_names,
                "feature_branches"
            )

        # Validate prediction branches
        if config.collect_predictions:
            if config.has_per_branch_config():
                # At this point, prediction_configs is List[BranchPredictionConfig]
                assert config.prediction_configs is not None  # Type narrowing
                for pc in config.prediction_configs:
                    self._validate_branch_reference(
                        pc.branch,
                        available_indices,
                        available_names,
                        f"prediction_config[branch={pc.branch}]"
                    )
            elif config.prediction_branches != "all":
                # At this point, prediction_branches is List[int] (not "all")
                prediction_branch_list = config.prediction_branches
                assert isinstance(prediction_branch_list, list)  # Type narrowing
                self._validate_branch_indices(
                    prediction_branch_list,
                    available_indices,
                    available_names,
                    "prediction_branches"
                )

    def _validate_branch_indices(
        self,
        indices: List[int],
        available_indices: set,
        available_names: Dict[str, int],
        context_name: str
    ) -> None:
        """Validate a list of branch indices.

        Args:
            indices: List of branch indices to validate
            available_indices: Set of valid branch indices
            available_names: Map of branch names to indices
            context_name: Name for error context

        Raises:
            ValueError: If any index is invalid.
        """
        for idx in indices:
            self._validate_branch_reference(
                idx, available_indices, available_names, context_name
            )

    def _validate_branch_reference(
        self,
        ref: Union[int, str],
        available_indices: set,
        available_names: Dict[str, int],
        context_name: str
    ) -> None:
        """Validate a single branch reference (index or name).

        Args:
            ref: Branch index (int) or name (str)
            available_indices: Set of valid branch indices
            available_names: Map of branch names to indices
            context_name: Name for error context

        Raises:
            ValueError: If reference is invalid.
        """
        if isinstance(ref, int):
            if ref not in available_indices:
                raise ValueError(
                    f"Invalid branch index in {context_name}: {ref}. "
                    f"Available indices: {sorted(available_indices)}. "
                    f"[Error: MERGE-E021]"
                )
        elif isinstance(ref, str):
            if ref not in available_names:
                raise ValueError(
                    f"Invalid branch name in {context_name}: '{ref}'. "
                    f"Available names: {list(available_names.keys())}. "
                    f"[Error: MERGE-E021]"
                )
        else:
            raise ValueError(
                f"Branch reference must be int or str, got {type(ref).__name__}: {ref}"
            )

    def _log_config(
        self,
        config: MergeConfig,
        n_branches: int,
        branch_contexts: Optional[List[Dict[str, Any]]] = None,
        prediction_store: Optional[Any] = None,
        context: Optional["ExecutionContext"] = None,
    ) -> None:
        """Log merge configuration for debugging.

        Phase 6: Enhanced logging for mixed merge and asymmetric scenarios.

        Args:
            config: Merge configuration
            n_branches: Number of available branches
            branch_contexts: Optional branch contexts for asymmetric analysis
            prediction_store: Optional prediction store for model discovery
            context: Optional execution context
        """
        mode = config.get_merge_mode()

        # Phase 6: Log mixed merge detection
        if config.collect_features and config.collect_predictions:
            logger.info("  Mixed merge detected: collecting both features and predictions")

        if config.collect_features:
            feat_branches = config.get_feature_branches(n_branches)
            logger.info(f"  Features: collecting from branches {feat_branches}")

        if config.collect_predictions:
            if config.has_per_branch_config():
                # Type narrowing: has_per_branch_config() guarantees prediction_configs is not None
                assert config.prediction_configs is not None
                for pc in config.prediction_configs:
                    logger.info(
                        f"  Predictions: branch={pc.branch}, "
                        f"select={pc.select}, aggregate={pc.aggregate}"
                    )
            else:
                pred_branches = config.prediction_branches
                logger.info(
                    f"  Predictions: collecting from branches {pred_branches}, "
                    f"models={config.model_filter or 'all'}"
                )

        if config.include_original:
            logger.info("  Including original pre-branch features")

        if config.output_as != "features":
            logger.info(f"  Output target: {config.output_as}")

        # Phase 6: Log asymmetric branch analysis if context available
        if branch_contexts and prediction_store and context:
            try:
                analyzer = AsymmetricBranchAnalyzer(
                    branch_contexts=branch_contexts,
                    prediction_store=prediction_store,
                    context=context,
                )
                report = analyzer.analyze_all()

                if report.is_asymmetric:
                    logger.info(f"  Asymmetric branches: {report.summary}")

                    if report.has_model_asymmetry and not (config.collect_features and config.collect_predictions):
                        # User is not using mixed merge but branches are asymmetric
                        suggestion = analyzer.suggest_mixed_merge()
                        if suggestion:
                            logger.warning(
                                f"⚠️ Asymmetric branches detected but not using mixed merge. "
                                f"Some branches may not contribute to the result. "
                                f"{suggestion}"
                            )
            except Exception as e:
                # Don't fail the pipeline due to analysis errors
                logger.debug(f"Asymmetric analysis failed: {e}")

    # =========================================================================
    # Phase 3: Feature Extraction and Collection
    # =========================================================================

    def _collect_features(
        self,
        dataset: "SpectroDataset",
        branch_contexts: List[Dict[str, Any]],
        branch_indices: List[int],
        on_missing: str = "error",
        on_shape_mismatch: str = "error",
        preserve_preprocessing: bool = False,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Collect features from specified branches.

        Extracts features from each branch's feature snapshot. By default, features
        are extracted in 2D layout (samples, features) and are horizontally
        concatenated during merge. When preserve_preprocessing=True, features are
        extracted in 3D layout (samples, processings, features) to preserve the
        preprocessing dimension.

        Args:
            dataset: Dataset (used to get sample count for validation).
            branch_contexts: List of branch context dictionaries.
            branch_indices: List of branch indices to collect features from.
            on_missing: How to handle missing snapshots:
                - "error": Raise ValueError
                - "warn": Log warning and skip
                - "skip": Silent skip
            on_shape_mismatch: Reserved for future 3D layout support.
                In 2D layout (current default), features are simply concatenated
                and this parameter has no effect. Will be used when 3D layout
                is needed and number of processings must align:
                - "error": Raise ValueError if processings differ
                - "allow": Allow different processings (flatten to 2D)
                - "pad": Pad shorter to match longest processings
                - "truncate": Truncate longer to match shortest
            preserve_preprocessing: If True, preserve the preprocessing dimension
                by extracting in 3D layout. Used when output_as="sources".

        Returns:
            Tuple of (features_list, info_dict) where:
                - features_list: List of numpy arrays (2D or 3D), one per branch
                - info_dict: Dictionary with collection metadata

        Raises:
            ValueError: If branch is missing and on_missing="error", or
                if sample counts don't match.
        """
        features_list = []
        shapes = []
        branches_used = []
        expected_samples = dataset.num_samples
        layout = "3d" if preserve_preprocessing else "2d"

        for branch_idx in branch_indices:
            # Find branch context by index
            branch_ctx = self._get_branch_context(branch_contexts, branch_idx)

            if branch_ctx is None:
                msg = f"Branch {branch_idx} not found in branch contexts. [Error: MERGE-E021]"
                if on_missing == "error":
                    raise ValueError(msg)
                elif on_missing == "warn":
                    logger.warning(msg + " Skipping.")
                    continue
                else:  # skip
                    continue

            # Extract features from snapshot
            snapshot = branch_ctx.get("features_snapshot")
            if snapshot is None:
                msg = f"Branch {branch_idx} has no feature snapshot. [Error: MERGE-E001]"
                if on_missing == "error":
                    raise ValueError(msg)
                elif on_missing == "warn":
                    logger.warning(msg + " Skipping.")
                    continue
                else:  # skip
                    continue

            # Extract features from snapshot (2D or 3D based on preserve_preprocessing)
            try:
                features = self._extract_features_from_snapshot(
                    snapshot=snapshot,
                    expected_samples=expected_samples,
                    branch_idx=branch_idx,
                    layout=layout,
                )
            except ValueError as e:
                msg = f"Failed to extract features from branch {branch_idx}: {e}"
                if on_missing == "error":
                    raise ValueError(msg) from e
                elif on_missing == "warn":
                    logger.warning(msg + " Skipping.")
                    continue
                else:  # skip
                    continue

            features_list.append(features)
            shapes.append(features.shape)
            branches_used.append(branch_idx)

            logger.debug(
                f"Extracted features from branch {branch_idx}: "
                f"shape={features.shape}"
            )

        # Validate sample counts match
        if features_list:
            n_samples_list = [f.shape[0] for f in features_list]
            if len(set(n_samples_list)) > 1:
                raise ValueError(
                    f"Sample count mismatch across branches: {n_samples_list}. "
                    f"All branches must have the same number of samples. "
                    f"[Error: MERGE-E003]"
                )

        # Note: Shape mismatch checking is NOT performed for 2D feature collection.
        # In 2D layout, features from different branches are simply concatenated
        # horizontally, so different feature dimensions across branches is expected
        # and normal behavior. Shape mismatch handling (pad/truncate) only applies
        # when using 3D layout where the number of processings must align.
        # The on_shape_mismatch parameter is reserved for future 3D layout support.

        info = {
            "shapes": shapes,
            "branches_used": branches_used,
        }

        return features_list, info

    def _extract_features_from_snapshot(
        self,
        snapshot: List["FeatureSource"],
        expected_samples: int,
        branch_idx: int,
        layout: str = "2d",
    ) -> np.ndarray:
        """Extract features from a branch's feature snapshot.

        The snapshot is a list of FeatureSource objects (one per data source).
        Each FeatureSource contains a 3D array of shape (samples, processings, features).

        Args:
            snapshot: List of FeatureSource objects from branch context.
            expected_samples: Expected number of samples.
            branch_idx: Branch index (for error messages).
            layout: Feature layout to extract:
                - "2d": Flatten to (n_samples, processings * features)
                - "3d": Preserve as (n_samples, processings, features)

        Returns:
            If layout="2d": 2D numpy array of shape (n_samples, total_features)
                containing all features from all sources and processings,
                concatenated horizontally.
            If layout="3d": 3D numpy array of shape (n_samples, processings, features)
                preserving the preprocessing dimension.

        Raises:
            ValueError: If snapshot is empty, sample count mismatches, or
                extraction fails.
        """
        if not snapshot:
            raise ValueError(
                f"Branch {branch_idx} snapshot is empty (no feature sources)"
            )

        source_features = []

        for src_idx, feature_source in enumerate(snapshot):
            # Get number of samples in this source
            n_samples = feature_source.num_samples
            if n_samples != expected_samples:
                raise ValueError(
                    f"Branch {branch_idx} source {src_idx} has {n_samples} samples, "
                    f"expected {expected_samples}. [Error: MERGE-E003]"
                )

            # Get all sample indices
            sample_indices = list(range(n_samples))

            # Extract features with specified layout
            try:
                features = feature_source.x(indices=sample_indices, layout=layout)
            except Exception as e:
                raise ValueError(
                    f"Failed to extract features from branch {branch_idx} "
                    f"source {src_idx}: {e}"
                ) from e

            if features.size == 0:
                logger.warning(
                    f"Branch {branch_idx} source {src_idx} has empty features "
                    f"(shape: {features.shape})"
                )
                continue

            source_features.append(features)

        if not source_features:
            raise ValueError(
                f"Branch {branch_idx} has no extractable features "
                f"(all sources empty)"
            )

        # Concatenate all source features
        if len(source_features) == 1:
            return source_features[0]

        # For 2D layout, concatenate horizontally along axis=1
        # For 3D layout, concatenate along feature axis (axis=2)
        concat_axis = 1 if layout == "2d" else 2
        return np.concatenate(source_features, axis=concat_axis)

    def _get_branch_context(
        self,
        branch_contexts: List[Dict[str, Any]],
        branch_ref: Union[int, str]
    ) -> Optional[Dict[str, Any]]:
        """Get a branch context by index or name.

        Args:
            branch_contexts: List of branch context dictionaries.
            branch_ref: Branch index (int) or name (str).

        Returns:
            Branch context dictionary, or None if not found.
        """
        if isinstance(branch_ref, int):
            for bc in branch_contexts:
                if bc["branch_id"] == branch_ref:
                    return bc
        elif isinstance(branch_ref, str):
            for bc in branch_contexts:
                if bc.get("name") == branch_ref:
                    return bc
        return None

    def _handle_shape_mismatch(
        self,
        features_list: List[np.ndarray],
        strategy: str,
        branches_used: List[int]
    ) -> List[np.ndarray]:
        """Handle feature dimension mismatches across branches.

        Args:
            features_list: List of 2D feature arrays.
            strategy: How to handle mismatches:
                - "error": Raise ValueError
                - "pad": Pad shorter with zeros
                - "truncate": Truncate longer to shortest
            branches_used: List of branch indices (for error messages).

        Returns:
            List of 2D feature arrays with consistent feature dimensions.

        Raises:
            ValueError: If strategy is "error" and dimensions differ.
        """
        if len(features_list) <= 1:
            return features_list

        feature_dims = [f.shape[1] for f in features_list]

        # Check if all dimensions are the same
        if len(set(feature_dims)) == 1:
            return features_list

        # Dimensions differ - apply strategy
        if strategy == "error":
            raise ValueError(
                f"Feature dimension mismatch across branches. "
                f"Branches {branches_used} have dimensions {feature_dims}. "
                f"Set on_shape_mismatch='allow' to concatenate anyway, "
                f"'pad' to zero-pad, or 'truncate' to truncate. "
                f"[Error: MERGE-E002]"
            )

        elif strategy == "pad":
            max_features = max(feature_dims)
            padded_list = []
            for i, features in enumerate(features_list):
                if features.shape[1] < max_features:
                    pad_width = max_features - features.shape[1]
                    padded = np.pad(
                        features,
                        ((0, 0), (0, pad_width)),
                        mode='constant',
                        constant_values=0
                    )
                    logger.info(
                        f"  Padded branch {branches_used[i]} from "
                        f"{features.shape[1]} to {max_features} features"
                    )
                    padded_list.append(padded)
                else:
                    padded_list.append(features)
            return padded_list

        elif strategy == "truncate":
            min_features = min(feature_dims)
            truncated_list = []
            for i, features in enumerate(features_list):
                if features.shape[1] > min_features:
                    logger.warning(
                        f"  Truncating branch {branches_used[i]} from "
                        f"{features.shape[1]} to {min_features} features"
                    )
                    truncated_list.append(features[:, :min_features])
                else:
                    truncated_list.append(features)
            return truncated_list

        # Default: allow (no modification)
        return features_list

    def _get_original_features(
        self,
        dataset: "SpectroDataset",
        context: "ExecutionContext"
    ) -> Optional[np.ndarray]:
        """Get original pre-branch features from dataset.

        Retrieves the features that were present before branching started.
        This uses the context's pre_branch_features_snapshot if available,
        otherwise falls back to current dataset features.

        Args:
            dataset: The dataset.
            context: Execution context (may contain pre-branch snapshot).

        Returns:
            2D numpy array of original features, or None if unavailable.
        """
        # Check if context has a pre-branch snapshot stored
        pre_branch_snapshot = context.custom.get("pre_branch_features_snapshot")

        if pre_branch_snapshot is not None:
            try:
                return self._extract_features_from_snapshot(
                    snapshot=pre_branch_snapshot,
                    expected_samples=dataset.num_samples,
                    branch_idx=-1  # -1 indicates original features
                )
            except Exception as e:
                logger.warning(
                    f"Failed to extract pre-branch features: {e}. "
                    f"Falling back to current dataset features."
                )

        # Fallback: get current dataset features
        # Note: This may not be ideal as current features are from a specific branch
        try:
            X = dataset.x(selector={}, layout="2d", concat_source=True)
            # X could be ndarray or list[ndarray] depending on settings
            if isinstance(X, list):
                if len(X) == 0:
                    return None
                X = np.concatenate(X, axis=1) if len(X) > 1 else X[0]
            return X
        except Exception as e:
            logger.warning(f"Failed to get original features: {e}")
            return None

    # =========================================================================
    # Phase 4 & 5: Prediction Collection with Per-Branch Configuration
    # =========================================================================

    def _collect_predictions(
        self,
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        branch_contexts: List[Dict[str, Any]],
        config: MergeConfig,
        prediction_store: Optional["Predictions"],
        mode: str = "train",
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Collect predictions from specified branches with per-branch control.

        Orchestrates model discovery, selection, aggregation, and OOF reconstruction.
        Supports Phase 5 per-branch configuration for selection and aggregation
        strategies.

        Phase 5 Features:
        - Model selection per branch: all, best, top_k, explicit list
        - Aggregation per branch: separate, mean, weighted_mean, proba_mean
        - Model ranking by validation metrics

        Args:
            dataset: Dataset for sample information.
            context: Execution context with branch/fold info.
            branch_contexts: List of branch context dictionaries.
            config: Merge configuration.
            prediction_store: Prediction storage containing model predictions.
            mode: Execution mode ("train" or "predict").

        Returns:
            Tuple of (predictions_array, info_dict) where:
                - predictions_array: 2D array (n_samples, n_features) or None
                - info_dict: Dictionary with collection metadata

        Raises:
            ValueError: If no predictions found or prediction store unavailable.
        """
        if prediction_store is None:
            raise ValueError(
                "prediction_store is required for prediction merge. "
                "Ensure models were trained in the specified branches. "
                "[Error: MERGE-E010]"
            )

        n_branches = len(branch_contexts)

        # Get prediction configs (Phase 5 or legacy)
        prediction_configs = config.get_prediction_configs(n_branches)

        # Initialize model selector for Phase 5 features
        model_selector = ModelSelector(
            prediction_store=prediction_store,
            context=context,
        )

        # Collect predictions per branch with selection and aggregation
        all_branch_predictions: List[np.ndarray] = []
        all_models_used: List[str] = []
        branches_used: List[int] = []
        selection_info: List[Dict[str, Any]] = []

        for branch_config in prediction_configs:
            branch_ref = branch_config.branch
            actual_idx = self._resolve_branch_index(branch_contexts, branch_ref)
            branch_ctx = self._get_branch_context(branch_contexts, actual_idx)

            if branch_ctx is None:
                if config.on_missing == "error":
                    raise ValueError(
                        f"Branch {branch_ref} not found for prediction collection. "
                        f"[Error: MERGE-E021]"
                    )
                logger.warning(f"Branch {branch_ref} not found, skipping predictions.")
                continue

            branch_id = branch_ctx["branch_id"]

            # Discover all models in this branch
            available_models = self._discover_branch_models(
                prediction_store=prediction_store,
                branch_id=branch_id,
                context=context,
                model_filter=None,  # Don't pre-filter; selection does that
            )

            if not available_models:
                if config.on_missing == "error":
                    # Phase 6: Provide detailed error with asymmetric branch analysis
                    analyzer = AsymmetricBranchAnalyzer(
                        branch_contexts=branch_contexts,
                        prediction_store=prediction_store,
                        context=context,
                    )
                    report = analyzer.analyze_all()

                    if report.has_model_asymmetry and branch_id in report.branches_without_models:
                        suggestion = analyzer.suggest_mixed_merge()
                        raise ValueError(
                            f"No model predictions found in branch {branch_ref}. "
                            f"This branch has only features (no trained models). "
                            f"{report.summary}. "
                            f"\n\n{suggestion}\n"
                            f"[Error: MERGE-E011]"
                        )
                    else:
                        raise ValueError(
                            f"No model predictions found in branch {branch_ref}. "
                            f"Ensure models were trained in this branch before merge. "
                            f"[Error: MERGE-E010]"
                        )
                logger.warning(f"No models found in branch {branch_ref}, skipping.")
                continue

            # Phase 5: Apply model selection
            selected_models = model_selector.select_models(
                available_models=available_models,
                config=branch_config,
                branch_id=branch_id,
            )

            if not selected_models:
                logger.warning(
                    f"No models selected from branch {branch_ref} after applying "
                    f"selection strategy: {branch_config.select}. Skipping."
                )
                continue

            logger.info(
                f"  Branch {branch_id}: selected {len(selected_models)}/{len(available_models)} models "
                f"using strategy '{branch_config.get_selection_strategy().value}'"
            )

            # Collect OOF predictions for selected models
            branch_predictions = self._collect_branch_predictions(
                dataset=dataset,
                context=context,
                prediction_store=prediction_store,
                model_names=selected_models,
                branch_id=branch_id,
                config=config,
                mode=mode,
            )

            if branch_predictions is None or len(branch_predictions) == 0:
                logger.warning(f"No predictions collected from branch {branch_id}")
                continue

            # Phase 5: Apply aggregation strategy
            aggregation_strategy = branch_config.get_aggregation_strategy()

            if aggregation_strategy != AggregationStrategy.SEPARATE:
                # Get model scores for weighted aggregation
                model_scores = None
                if aggregation_strategy == AggregationStrategy.WEIGHTED_MEAN:
                    metric = branch_config.weight_metric or branch_config.metric or "rmse"
                    model_scores = model_selector.get_model_scores(
                        model_names=selected_models,
                        metric=metric,
                        branch_id=branch_id,
                    )

                # Aggregate predictions
                aggregated = PredictionAggregator.aggregate(
                    predictions=branch_predictions,
                    strategy=aggregation_strategy,
                    model_scores=model_scores,
                    proba=branch_config.proba,
                    metric=branch_config.weight_metric or branch_config.metric,
                )

                logger.info(
                    f"  Branch {branch_id}: aggregated {len(selected_models)} models "
                    f"using '{aggregation_strategy.value}' → shape {aggregated.shape}"
                )

                all_branch_predictions.append(aggregated)
            else:
                # Keep predictions separate (each model = 1 feature)
                separate = PredictionAggregator.aggregate(
                    predictions=branch_predictions,
                    strategy=AggregationStrategy.SEPARATE,
                )
                all_branch_predictions.append(separate)

            all_models_used.extend(selected_models)
            branches_used.append(actual_idx)
            selection_info.append({
                "branch": actual_idx,
                "available_models": len(available_models),
                "selected_models": selected_models,
                "selection_strategy": branch_config.get_selection_strategy().value,
                "aggregation_strategy": aggregation_strategy.value,
            })

        if not all_branch_predictions:
            if config.on_missing == "error":
                # Phase 6: Use asymmetric analyzer for better error messages
                analyzer = AsymmetricBranchAnalyzer(
                    branch_contexts=branch_contexts,
                    prediction_store=prediction_store,
                    context=context,
                )
                report = analyzer.analyze_all()

                if report.has_model_asymmetry:
                    # Provide resolution suggestion for asymmetric branches
                    suggestion = analyzer.suggest_mixed_merge()
                    raise ValueError(
                        f"No model predictions found in any specified branch. "
                        f"Asymmetric branches detected: {report.summary}. "
                        f"\n\n{suggestion}\n"
                        f"[Error: MERGE-E011]"
                    )
                else:
                    raise ValueError(
                        f"No model predictions found in any specified branch. "
                        f"Ensure models were trained in the specified branches before merge. "
                        f"[Error: MERGE-E010]"
                    )
            logger.warning("No predictions collected from any branch.")
            return None, {"models_used": [], "branches_used": []}

        # Concatenate all branch predictions horizontally
        predictions = np.concatenate(all_branch_predictions, axis=1)

        info = {
            "models_used": all_models_used,
            "branches_used": branches_used,
            "oof_reconstruction": not config.unsafe,
            "n_features": predictions.shape[1],
            "selection_info": selection_info,
        }

        return predictions, info

    def _collect_branch_predictions(
        self,
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        prediction_store: "Predictions",
        model_names: List[str],
        branch_id: int,
        config: MergeConfig,
        mode: str = "train",
    ) -> Optional[Dict[str, np.ndarray]]:
        """Collect predictions for specified models from a single branch.

        Returns a dictionary mapping model names to their prediction arrays,
        suitable for per-branch aggregation.

        Args:
            dataset: Dataset for sample information.
            context: Execution context.
            prediction_store: Prediction storage.
            model_names: List of model names to collect.
            branch_id: Branch identifier.
            config: Merge configuration.
            mode: Execution mode.

        Returns:
            Dictionary mapping model names to prediction arrays (n_samples,),
            or None if no predictions found.
        """
        if config.unsafe:
            return self._collect_branch_predictions_unsafe(
                dataset=dataset,
                context=context,
                prediction_store=prediction_store,
                model_names=model_names,
                branch_id=branch_id,
            )
        else:
            return self._collect_branch_predictions_oof(
                dataset=dataset,
                context=context,
                prediction_store=prediction_store,
                model_names=model_names,
                branch_id=branch_id,
                config=config,
            )

    def _collect_branch_predictions_oof(
        self,
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        prediction_store: "Predictions",
        model_names: List[str],
        branch_id: int,
        config: MergeConfig,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Collect OOF predictions for models in a single branch.

        Uses TrainingSetReconstructor for proper OOF reconstruction.

        Args:
            dataset: Dataset for sample information.
            context: Execution context.
            prediction_store: Prediction storage.
            model_names: List of model names.
            branch_id: Branch identifier.
            config: Merge configuration.

        Returns:
            Dictionary mapping model names to prediction arrays.
        """
        from nirs4all.controllers.models.stacking import (
            TrainingSetReconstructor,
            ReconstructorConfig,
        )
        from nirs4all.operators.models.meta import StackingConfig, CoverageStrategy

        # Create stacking config with IMPUTE_MEAN to handle incomplete coverage
        # This is more lenient than STRICT and allows merge to work with
        # samples that may not have predictions from all folds
        stacking_config = StackingConfig(
            coverage_strategy=CoverageStrategy.IMPUTE_MEAN,
        )
        reconstructor_config = ReconstructorConfig(
            log_warnings=True,
            validate_fold_alignment=False,  # Allow fold mismatch for branch merge
        )

        model_predictions = {}

        for model_name in model_names:
            try:
                reconstructor = TrainingSetReconstructor(
                    prediction_store=prediction_store,
                    source_model_names=[model_name],
                    stacking_config=stacking_config,
                    reconstructor_config=reconstructor_config,
                )

                result = reconstructor.reconstruct(
                    dataset=dataset,
                    context=context,
                    use_proba=config.use_proba,
                )

                # Combine train (OOF) and test predictions
                n_total = dataset.num_samples
                combined = np.full(n_total, np.nan)

                # Get train and test sample indices
                # IMPORTANT: Use include_augmented=False for train because OOF predictions
                # are only available for original (non-augmented) samples
                train_context = context.with_partition('train')
                train_ids = dataset._indexer.x_indices(
                    train_context.selector,
                    include_augmented=False,
                    include_excluded=False
                )

                test_context = context.with_partition('test')
                test_ids = dataset._indexer.x_indices(
                    test_context.selector,
                    include_augmented=False,
                    include_excluded=False
                )

                # Fill train (OOF) predictions
                if result.X_train_meta.size > 0:
                    train_preds = result.X_train_meta[:, 0] if result.X_train_meta.ndim > 1 else result.X_train_meta
                    if len(train_preds) == len(train_ids):
                        for i, sample_id in enumerate(train_ids):
                            combined[sample_id] = train_preds[i]

                # Fill test predictions
                if result.X_test_meta.size > 0:
                    test_preds = result.X_test_meta[:, 0] if result.X_test_meta.ndim > 1 else result.X_test_meta
                    if len(test_preds) == len(test_ids):
                        for i, sample_id in enumerate(test_ids):
                            combined[sample_id] = test_preds[i]

                # Propagate predictions from base samples to their augmented versions
                # Augmented samples should have the same prediction as their origin
                base_sample_ids = list(train_ids) + list(test_ids)
                if base_sample_ids:
                    augmented_ids = dataset._indexer._augmentation_tracker.get_augmented_for_origins(
                        base_sample_ids
                    )
                    for aug_id in augmented_ids:
                        origin_id = dataset._indexer._augmentation_tracker.get_origin_for_sample(aug_id)
                        if origin_id is not None and not np.isnan(combined[origin_id]):
                            combined[aug_id] = combined[origin_id]

                model_predictions[model_name] = combined

            except Exception as e:
                logger.warning(
                    f"Failed to collect OOF predictions for model '{model_name}' "
                    f"in branch {branch_id}: {e}"
                )
                continue

        return model_predictions if model_predictions else None

    def _collect_branch_predictions_unsafe(
        self,
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        prediction_store: "Predictions",
        model_names: List[str],
        branch_id: int,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Collect predictions WITHOUT OOF reconstruction (UNSAFE).

        ⚠️ WARNING: This causes DATA LEAKAGE when used for training.

        Args:
            dataset: Dataset for sample information.
            context: Execution context.
            prediction_store: Prediction storage.
            model_names: List of model names.
            branch_id: Branch identifier.

        Returns:
            Dictionary mapping model names to prediction arrays.
        """
        logger.warning(
            "⚠️ UNSAFE PREDICTION COLLECTION: Using training predictions directly. "
            "This causes DATA LEAKAGE - do NOT use for final model evaluation!"
        )

        current_step = getattr(context.state, 'step_number', float('inf'))
        n_total = dataset.num_samples

        # Get sample indices for both partitions
        train_context = context.with_partition('train')
        train_ids = dataset._indexer.x_indices(
            train_context.selector,
            include_augmented=True,
            include_excluded=False
        )

        test_context = context.with_partition('test')
        test_ids = dataset._indexer.x_indices(
            test_context.selector,
            include_augmented=False,
            include_excluded=False
        )

        model_predictions = {}

        for model_name in model_names:
            combined = np.full(n_total, np.nan)

            # Collect train partition predictions
            train_preds = self._get_unsafe_partition_predictions(
                prediction_store=prediction_store,
                model_name=model_name,
                partition="train",
                current_step=current_step,
            )

            if train_preds:
                for pred in train_preds:
                    y_pred = pred.get('y_pred')
                    sample_indices = pred.get('sample_indices')

                    if y_pred is None:
                        continue

                    y_pred = np.asarray(y_pred).flatten()

                    if sample_indices is not None:
                        if hasattr(sample_indices, 'tolist'):
                            sample_indices = sample_indices.tolist()
                        for i, sid in enumerate(sample_indices):
                            if i < len(y_pred) and int(sid) < n_total:
                                combined[int(sid)] = y_pred[i]

            # Collect test partition predictions
            test_preds = self._get_unsafe_partition_predictions(
                prediction_store=prediction_store,
                model_name=model_name,
                partition="test",
                current_step=current_step,
            )

            if test_preds:
                # Aggregate test predictions across folds
                test_aggregated: Dict[int, List[float]] = {}

                for pred in test_preds:
                    y_pred = pred.get('y_pred')
                    sample_indices = pred.get('sample_indices')

                    if y_pred is None:
                        continue

                    y_pred = np.asarray(y_pred).flatten()

                    if sample_indices is not None:
                        if hasattr(sample_indices, 'tolist'):
                            sample_indices = sample_indices.tolist()
                        for i, sid in enumerate(sample_indices):
                            if i < len(y_pred):
                                sample_idx = int(sid)
                                if sample_idx not in test_aggregated:
                                    test_aggregated[sample_idx] = []
                                test_aggregated[sample_idx].append(y_pred[i])

                # Average across folds
                for sample_idx, values in test_aggregated.items():
                    if sample_idx < n_total:
                        combined[sample_idx] = np.mean(values)

            # Replace remaining NaN with 0
            combined = np.nan_to_num(combined, nan=0.0)
            model_predictions[model_name] = combined

        return model_predictions if model_predictions else None

    def _get_unsafe_partition_predictions(
        self,
        prediction_store: "Predictions",
        model_name: str,
        partition: str,
        current_step: Union[int, float],
    ) -> List[Dict[str, Any]]:
        """Get predictions for a model/partition without OOF.

        Helper for unsafe prediction collection.

        Args:
            prediction_store: Prediction storage.
            model_name: Model name.
            partition: Partition name.
            current_step: Current step for filtering.

        Returns:
            List of prediction dictionaries.
        """
        filter_kwargs = {
            'model_name': model_name,
            'partition': partition,
            'load_arrays': True,
        }

        predictions = prediction_store.filter_predictions(**filter_kwargs)

        # Filter by step
        return [
            p for p in predictions
            if p.get('step_idx', 0) < current_step
        ]

    def _discover_branch_models(
        self,
        prediction_store: "Predictions",
        branch_id: int,
        context: "ExecutionContext",
        model_filter: Optional[List[str]] = None,
    ) -> List[str]:
        """Discover models that have predictions in a branch.

        Queries the prediction store for models that ran in the specified
        branch and returns their names.

        Args:
            prediction_store: Prediction storage.
            branch_id: Branch ID to search for.
            context: Execution context for step filtering.
            model_filter: Optional list of model names to include.

        Returns:
            List of model names with predictions in the branch.
        """
        current_step = getattr(context.state, 'step_number', float('inf'))

        # Query prediction store for validation predictions in this branch
        filter_kwargs = {
            'branch_id': branch_id,
            'partition': 'val',
            'load_arrays': False,
        }

        predictions = prediction_store.filter_predictions(**filter_kwargs)

        # Filter by step (only include predictions from earlier steps)
        predictions = [
            p for p in predictions
            if p.get('step_idx', 0) < current_step
        ]

        # If no predictions with branch filter, try pre-branch models
        # (models trained before branch was created have branch_id=None)
        if not predictions:
            filter_kwargs_no_branch = {
                'partition': 'val',
                'load_arrays': False,
            }
            predictions = prediction_store.filter_predictions(**filter_kwargs_no_branch)
            predictions = [
                p for p in predictions
                if p.get('step_idx', 0) < current_step and p.get('branch_id') is None
            ]

        # Extract unique model names
        model_names = set()
        for pred in predictions:
            model_name = pred.get('model_name')
            if model_name:
                model_names.add(model_name)

        # Apply model filter if specified
        if model_filter:
            model_names = model_names.intersection(set(model_filter))

        return sorted(model_names)

    def _resolve_branch_index(
        self,
        branch_contexts: List[Dict[str, Any]],
        branch_ref: Union[int, str]
    ) -> int:
        """Resolve a branch reference to its numeric index.

        Args:
            branch_contexts: List of branch context dictionaries.
            branch_ref: Branch index (int) or name (str).

        Returns:
            Numeric branch index.

        Raises:
            ValueError: If branch not found.
        """
        if isinstance(branch_ref, int):
            return branch_ref
        elif isinstance(branch_ref, str):
            for bc in branch_contexts:
                if bc.get("name") == branch_ref:
                    return bc["branch_id"]
            raise ValueError(f"Branch name '{branch_ref}' not found")
        else:
            raise ValueError(f"Invalid branch reference type: {type(branch_ref)}")

    # =========================================================================
    # Phase 8: Prediction Mode Support
    # =========================================================================

    def _execute_branch_merge_predict_mode(
        self,
        step_info: "ParsedStep",
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int,
        config: MergeConfig,
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None,
    ) -> Tuple["ExecutionContext", StepOutput]:
        """Execute merge in prediction mode without active branch contexts.

        In prediction mode, branches have already been processed by the executor
        which iterates through each branch and applies their transformers. The merge
        step's job in predict mode is to:

        1. For feature merge: The dataset already has the merged/transformed features
           from branch processing. We just need to mark branch mode as exited.

        2. For prediction merge: Collect predictions from the prediction store
           using the same configuration that was used during training.

        The key insight is that merge doesn't persist artifacts itself - it orchestrates
        the combination of outputs from branches. In predict mode, this orchestration
        has already happened through the branch iteration in the executor.

        Args:
            step_info: Parsed step info
            dataset: Dataset with branch-transformed features
            context: Execution context
            runtime_context: Runtime context
            source: Source index
            config: Parsed merge configuration
            loaded_binaries: Not used (merge has no artifacts)
            prediction_store: For prediction collection

        Returns:
            Updated context and StepOutput with prediction mode metadata
        """
        logger.info(
            f"Merge step (predict mode): mode={config.get_merge_mode().value}"
        )

        merged_parts = []
        merge_info = {"prediction_mode": True}

        # In predict mode for feature merge:
        # The features are already in the dataset from branch processing.
        # The executor has iterated through branches and applied transformers.
        # We just need to collect the current features as the "merged" output.
        if config.collect_features:
            # Get current features from dataset
            # In predict mode, the branch iteration has already produced transformed features
            try:
                # Try to get existing merged features if already created
                current_features = dataset.x(
                    selector=context.selector,
                    layout="2d",
                    concat_source=True
                )
                if isinstance(current_features, list):
                    current_features = np.concatenate(current_features, axis=1)

                if current_features is not None and current_features.size > 0:
                    merged_parts.append(current_features)
                    merge_info["feature_shape"] = current_features.shape
                    logger.info(
                        f"  Collected features for prediction: shape={current_features.shape}"
                    )
            except Exception as e:
                logger.warning(f"Could not extract features in predict mode: {e}")

        # In predict mode for prediction merge:
        # We need to collect predictions from models that ran during prediction
        if config.collect_predictions and prediction_store is not None:
            try:
                predictions_array = self._collect_predictions_predict_mode(
                    dataset=dataset,
                    context=context,
                    config=config,
                    prediction_store=prediction_store,
                )

                if predictions_array is not None and predictions_array.size > 0:
                    merged_parts.append(predictions_array)
                    merge_info["prediction_shape"] = predictions_array.shape
                    logger.info(
                        f"  Collected predictions for prediction: shape={predictions_array.shape}"
                    )
            except Exception as e:
                logger.warning(f"Could not collect predictions in predict mode: {e}")

        # Include original features if configured
        if config.include_original:
            original = self._get_original_features(dataset, context)
            if original is not None:
                merged_parts.insert(0, original)
                merge_info["original_shape"] = original.shape

        # Combine all parts
        if merged_parts:
            merged_features = np.concatenate(merged_parts, axis=1)
            merge_info["merged_shape"] = merged_features.shape
            logger.info(f"  Final merged shape (predict): {merged_features.shape}")

            # Store in dataset
            processing_name = "merged"
            if config.source_names and len(config.source_names) > 0:
                processing_name = config.source_names[0]

            dataset.add_merged_features(
                features=merged_features,
                processing_name=processing_name,
                source=0
            )

        # Exit branch mode (if any residual state)
        result_context = context.copy()
        result_context.custom["branch_contexts"] = []
        result_context.custom["in_branch_mode"] = False

        # Build metadata
        metadata = {
            "merge_mode": config.get_merge_mode().value,
            "prediction_mode": True,
            "output_as": config.output_as,
            **merge_info,
        }

        logger.success(
            f"Merge step (predict mode) completed. "
            f"Features={config.collect_features}, Predictions={config.collect_predictions}"
        )

        return result_context, StepOutput(metadata=metadata)

    def _collect_predictions_predict_mode(
        self,
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        config: MergeConfig,
        prediction_store: "Predictions",
    ) -> Optional[np.ndarray]:
        """Collect predictions in prediction mode.

        In predict mode, models have already generated predictions which are
        stored in the prediction store. We collect and aggregate them according
        to the merge configuration.

        Args:
            dataset: Dataset for sample info
            context: Execution context
            config: Merge configuration
            prediction_store: Prediction storage

        Returns:
            Aggregated predictions array or None
        """
        # Get model names from config
        model_filter = config.model_filter

        # Query prediction store for test partition predictions
        filter_kwargs = {
            'partition': 'test',
            'load_arrays': True,
        }

        predictions = prediction_store.filter_predictions(**filter_kwargs)

        if not predictions:
            logger.debug("No test predictions found in prediction store")
            return None

        # Group by model
        model_predictions: Dict[str, List[np.ndarray]] = {}
        for pred in predictions:
            model_name = pred.get('model_name')
            if model_name is None:
                continue

            # Apply model filter if specified
            if model_filter and model_name not in model_filter:
                continue

            y_pred = pred.get('y_pred')
            if y_pred is not None:
                y_pred = np.asarray(y_pred)
                if model_name not in model_predictions:
                    model_predictions[model_name] = []
                model_predictions[model_name].append(y_pred)

        if not model_predictions:
            logger.debug("No matching predictions after filtering")
            return None

        # Aggregate predictions per model (average across folds)
        aggregated = []
        for model_name, pred_list in model_predictions.items():
            if len(pred_list) == 1:
                model_pred = pred_list[0]
            else:
                # Average across folds
                try:
                    stacked = np.stack([p.flatten() for p in pred_list], axis=0)
                    model_pred = np.mean(stacked, axis=0)
                except Exception:
                    model_pred = pred_list[0]

            # Ensure 1D
            model_pred = model_pred.flatten()
            aggregated.append(model_pred.reshape(-1, 1))

        if not aggregated:
            return None

        return np.hstack(aggregated)

    def _execute_source_merge(
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
        """Execute source merge operation (Phase 9).

        Combines features from multiple data sources in a multi-source dataset.
        This is distinct from branch merging - it operates on the data provenance
        dimension (different sensors/instruments) rather than pipeline execution
        dimension (parallel processing paths).

        Supports three merge strategies:
        - concat: Horizontal concatenation (2D result)
        - stack: Stack along new axis (3D result, requires uniform shapes)
        - dict: Keep as structured dictionary (for multi-input models)

        Args:
            step_info: Parsed step containing merge configuration
            dataset: Dataset to operate on
            context: Pipeline execution context
            runtime_context: Runtime infrastructure context
            source: Data source index
            mode: Execution mode ("train" or "predict")
            loaded_binaries: Pre-loaded binary objects for prediction mode
            prediction_store: External prediction store for model predictions

        Returns:
            Tuple of (updated_context, StepOutput)

        Raises:
            ValueError: If dataset has only one source (warning in single-source case).
        """
        # Parse configuration
        raw_config = step_info.original_step.get("merge_sources")
        config = self._parse_source_merge_config(raw_config)

        # Validate multi-source dataset
        n_sources = dataset.n_sources

        if n_sources == 0:
            raise ValueError(
                "merge_sources requires a dataset with feature sources. "
                "No sources found in dataset. "
                "[Error: MERGE-E024]"
            )

        if n_sources == 1:
            # Single source - warn but don't fail
            logger.warning(
                "merge_sources called on single-source dataset. "
                "This is a no-op - the dataset already has unified features. "
                "Consider removing this step. [Warning: MERGE-E024]"
            )
            return context.copy(), StepOutput(metadata={
                "source_merge": "no-op",
                "n_sources": 1,
                "reason": "single_source_dataset",
            })

        # Get source names for logging and selection
        source_names = self._get_source_names(dataset, n_sources)

        logger.info(
            f"Source merge: strategy={config.strategy}, "
            f"sources={config.sources}, n_sources={n_sources}"
        )

        # Resolve source indices
        try:
            source_indices = config.get_source_indices(source_names)
        except ValueError as e:
            raise ValueError(str(e))

        if len(source_indices) < 2:
            logger.warning(
                f"Only {len(source_indices)} source(s) selected for merge. "
                "Merge requires at least 2 sources to be meaningful."
            )

        # Collect features from each source
        source_features, source_info = self._collect_source_features(
            dataset=dataset,
            context=context,
            source_indices=source_indices,
            source_names=source_names,
        )

        if not source_features:
            raise ValueError(
                "No features collected from any source. "
                "[Error: MERGE-E030]"
            )

        # Apply merge strategy
        strategy = config.get_strategy()

        if strategy == SourceMergeStrategy.CONCAT:
            merged_features, merge_info = self._merge_sources_concat(
                source_features=source_features,
                source_indices=source_indices,
                source_names=source_names,
            )
        elif strategy == SourceMergeStrategy.STACK:
            merged_features, merge_info = self._merge_sources_stack(
                source_features=source_features,
                source_indices=source_indices,
                source_names=source_names,
                on_incompatible=config.get_incompatible_strategy(),
            )
        elif strategy == SourceMergeStrategy.DICT:
            merged_features, merge_info = self._merge_sources_dict(
                source_features=source_features,
                source_indices=source_indices,
                source_names=source_names,
            )
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")

        # Store merged features in dataset
        # For dict strategy, we need special handling
        if strategy == SourceMergeStrategy.DICT:
            # Dict strategy - store reference in context for downstream use
            result_context = context.copy()
            result_context.custom["merged_sources_dict"] = merged_features
            result_context.custom["source_merge_applied"] = True

            logger.info(
                f"Source merge (dict) completed: {len(merged_features)} sources preserved"
            )
        else:
            # Array strategies (concat/stack) - update dataset
            processing_name = config.output_name

            # Store as merged features
            if isinstance(merged_features, np.ndarray):
                dataset.add_merged_features(
                    features=merged_features,
                    processing_name=processing_name,
                    source=0  # Primary source for merged features
                )

            result_context = context.copy()
            result_context.custom["source_merge_applied"] = True

            if merged_features is not None:
                shape_str = str(merged_features.shape) if hasattr(merged_features, 'shape') else 'dict'
                logger.info(
                    f"Source merge ({config.strategy}) completed: shape={shape_str}"
                )

        # Build metadata
        metadata = {
            "merge_sources_strategy": config.strategy,
            "sources_used": [source_names[i] for i in source_indices],
            "source_indices": source_indices,
            "n_sources_merged": len(source_indices),
            "output_name": config.output_name,
            # Store config for prediction mode
            "source_merge_config": config.to_dict(),
            **merge_info,
        }

        return result_context, StepOutput(metadata=metadata)

    def _parse_source_merge_config(
        self,
        raw_config: Any
    ) -> SourceMergeConfig:
        """Parse source merge configuration.

        Handles multiple syntax formats:
        - Simple string: "concat", "stack", "dict"
        - Dict with options: {"strategy": "stack", "sources": [...]}
        - Already parsed SourceMergeConfig

        Args:
            raw_config: Raw configuration from step

        Returns:
            Normalized SourceMergeConfig instance
        """
        if isinstance(raw_config, str):
            # Simple strategy string
            return SourceMergeConfig(strategy=raw_config)
        elif isinstance(raw_config, dict):
            # Dict configuration
            return SourceMergeConfig(
                strategy=raw_config.get("strategy", "concat"),
                sources=raw_config.get("sources", "all"),
                on_incompatible=raw_config.get("on_incompatible", "error"),
                output_name=raw_config.get("output_name", "merged"),
                preserve_source_info=raw_config.get("preserve_source_info", True),
            )
        elif isinstance(raw_config, SourceMergeConfig):
            return raw_config
        else:
            raise ValueError(
                f"Invalid merge_sources config type: {type(raw_config).__name__}. "
                f"Expected string, dict, or SourceMergeConfig."
            )

    def _get_source_names(
        self,
        dataset: "SpectroDataset",
        n_sources: int
    ) -> List[str]:
        """Get source names from dataset.

        Args:
            dataset: The dataset
            n_sources: Number of sources

        Returns:
            List of source names (generates default names if not available)
        """
        # Try to get source names from feature accessor
        try:
            source_names = []
            for i in range(n_sources):
                # Check if there's a name stored
                processings = dataset.features_processings(i)
                # Use first processing name or generate default
                if processings:
                    source_names.append(f"source_{i}")
                else:
                    source_names.append(f"source_{i}")
            return source_names
        except Exception:
            return [f"source_{i}" for i in range(n_sources)]

    def _collect_source_features(
        self,
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        source_indices: List[int],
        source_names: List[str],
    ) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
        """Collect features from specified sources.

        Args:
            dataset: The dataset
            context: Execution context
            source_indices: Which sources to collect
            source_names: Names for logging

        Returns:
            Tuple of (source_features dict, info dict)
        """
        source_features = {}
        shapes = {}

        for src_idx in source_indices:
            try:
                # Get features for this source
                # Use concat_source=False to get per-source data
                X = dataset.x(
                    selector=context.selector,
                    layout="2d",
                    concat_source=False,
                    include_augmented=True,
                    include_excluded=False
                )

                # X might be list or single array
                if isinstance(X, list):
                    if src_idx < len(X):
                        features = X[src_idx]
                    else:
                        logger.warning(
                            f"Source index {src_idx} out of range "
                            f"(got {len(X)} sources). Skipping."
                        )
                        continue
                else:
                    # Single source - only valid for index 0
                    if src_idx == 0:
                        features = X
                    else:
                        logger.warning(
                            f"Source index {src_idx} requested but dataset "
                            f"returned single array. Skipping."
                        )
                        continue

                source_features[src_idx] = features
                shapes[src_idx] = features.shape
                logger.debug(
                    f"Collected source {src_idx} ({source_names[src_idx]}): "
                    f"shape={features.shape}"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to collect features from source {src_idx}: {e}"
                )
                continue

        info = {
            "source_shapes": shapes,
            "sources_collected": len(source_features),
        }

        return source_features, info

    def _merge_sources_concat(
        self,
        source_features: Dict[int, np.ndarray],
        source_indices: List[int],
        source_names: List[str],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Merge sources by horizontal concatenation.

        Args:
            source_features: Dict mapping source index to feature array
            source_indices: Source indices in order
            source_names: Source names for logging

        Returns:
            Tuple of (merged 2D array, info dict)
        """
        arrays = []
        feature_counts = []

        for src_idx in source_indices:
            if src_idx in source_features:
                arr = source_features[src_idx]
                # Ensure 2D
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                elif arr.ndim > 2:
                    # Flatten to 2D
                    arr = arr.reshape(arr.shape[0], -1)
                arrays.append(arr)
                feature_counts.append(arr.shape[1])

        if not arrays:
            raise ValueError(
                "No arrays to concatenate. All sources failed to collect. "
                "[Error: MERGE-E030]"
            )

        # Validate sample counts match
        sample_counts = [arr.shape[0] for arr in arrays]
        if len(set(sample_counts)) > 1:
            raise ValueError(
                f"Sample count mismatch across sources: {sample_counts}. "
                f"All sources must have the same number of samples. "
                f"[Error: MERGE-E030]"
            )

        # Concatenate horizontally
        merged = np.concatenate(arrays, axis=1)

        info = {
            "merged_shape": merged.shape,
            "feature_counts_per_source": feature_counts,
            "total_features": merged.shape[1],
        }

        return merged, info

    def _merge_sources_stack(
        self,
        source_features: Dict[int, np.ndarray],
        source_indices: List[int],
        source_names: List[str],
        on_incompatible: SourceIncompatibleStrategy,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Merge sources by stacking along new axis (3D result).

        Args:
            source_features: Dict mapping source index to feature array
            source_indices: Source indices in order
            source_names: Source names for logging
            on_incompatible: How to handle shape mismatches

        Returns:
            Tuple of (merged 3D array or 2D fallback, info dict)
        """
        arrays = []
        feature_dims = []

        for src_idx in source_indices:
            if src_idx in source_features:
                arr = source_features[src_idx]
                # Ensure 2D first
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                elif arr.ndim > 2:
                    arr = arr.reshape(arr.shape[0], -1)
                arrays.append(arr)
                feature_dims.append(arr.shape[1])

        if not arrays:
            raise ValueError(
                "No arrays to stack. All sources failed to collect. "
                "[Error: MERGE-E030]"
            )

        # Check if shapes are compatible for stacking
        shapes_compatible = len(set(feature_dims)) == 1

        if not shapes_compatible:
            logger.warning(
                f"Source feature dimensions differ: {feature_dims}. "
                f"Cannot stack directly (requires uniform dimensions)."
            )

            if on_incompatible == SourceIncompatibleStrategy.ERROR:
                raise ValueError(
                    f"Cannot stack sources with different feature dimensions: {feature_dims}. "
                    f"Use on_incompatible='flatten' to fall back to 2D concat, "
                    f"or 'pad'/'truncate' to align dimensions. "
                    f"[Error: MERGE-E030]"
                )
            elif on_incompatible == SourceIncompatibleStrategy.FLATTEN:
                logger.info("Falling back to 2D concatenation due to shape mismatch")
                return self._merge_sources_concat(
                    source_features, source_indices, source_names
                )
            elif on_incompatible == SourceIncompatibleStrategy.PAD:
                max_features = max(feature_dims)
                padded_arrays = []
                for arr in arrays:
                    if arr.shape[1] < max_features:
                        padding = np.zeros((arr.shape[0], max_features - arr.shape[1]))
                        arr = np.hstack([arr, padding])
                    padded_arrays.append(arr)
                arrays = padded_arrays
                logger.info(f"Padded all sources to {max_features} features")
            elif on_incompatible == SourceIncompatibleStrategy.TRUNCATE:
                min_features = min(feature_dims)
                truncated_arrays = [arr[:, :min_features] for arr in arrays]
                arrays = truncated_arrays
                logger.info(f"Truncated all sources to {min_features} features")

        # Stack along axis 1 to create (samples, sources, features)
        merged = np.stack(arrays, axis=1)

        info = {
            "merged_shape": merged.shape,
            "n_sources_stacked": len(arrays),
            "features_per_source": arrays[0].shape[1] if arrays else 0,
            "shape_adjustment": on_incompatible.value if not shapes_compatible else None,
        }

        return merged, info

    def _merge_sources_dict(
        self,
        source_features: Dict[int, np.ndarray],
        source_indices: List[int],
        source_names: List[str],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Keep sources as structured dictionary.

        Args:
            source_features: Dict mapping source index to feature array
            source_indices: Source indices in order
            source_names: Source names for keys

        Returns:
            Tuple of (dict mapping source names to arrays, info dict)
        """
        result = {}
        shapes = {}

        for src_idx in source_indices:
            if src_idx in source_features:
                name = source_names[src_idx]
                arr = source_features[src_idx]
                # Ensure 2D
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                elif arr.ndim > 2:
                    arr = arr.reshape(arr.shape[0], -1)
                result[name] = arr
                shapes[name] = arr.shape

        info = {
            "source_shapes": shapes,
            "n_sources": len(result),
            "output_format": "dict",
        }

        return result, info

    def _execute_prediction_merge(
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
        """Execute prediction-only merge operation (late fusion).

        Late fusion of predictions without branch context requirements.
        Useful for combining predictions from multiple models without
        requiring branch mode. Unlike `{"merge": "predictions"}` which
        requires active branch contexts, this operates on the prediction
        store directly.

        Use cases:
        - Combine predictions from sequential models (not in branches)
        - Late fusion after separate model training phases
        - Ensemble of predictions for final output

        Args:
            step_info: Parsed step containing merge configuration
            dataset: Dataset to operate on
            context: Pipeline execution context
            runtime_context: Runtime infrastructure context
            source: Data source index
            mode: Execution mode ("train" or "predict")
            loaded_binaries: Pre-loaded binary objects for prediction mode
            prediction_store: External prediction store for model predictions

        Returns:
            Tuple of (updated_context, StepOutput)
        """
        if prediction_store is None:
            raise ValueError(
                "merge_predictions requires prediction_store. "
                "Ensure models were trained before this step. "
                "[Error: MERGE-E010]"
            )

        # Parse configuration
        raw_config = step_info.original_step.get("merge_predictions", {})

        # Handle simple string vs dict config
        if isinstance(raw_config, str):
            if raw_config == "all":
                model_filter = None
            else:
                model_filter = [raw_config]
            aggregation = "separate"
        elif isinstance(raw_config, dict):
            model_filter = raw_config.get("models")
            aggregation = raw_config.get("aggregate", "separate")
        else:
            model_filter = None
            aggregation = "separate"

        logger.info(
            f"Prediction merge: models={model_filter or 'all'}, "
            f"aggregate={aggregation}"
        )

        # Discover available models from prediction store
        current_step = getattr(context.state, 'step_number', float('inf'))

        filter_kwargs = {
            'partition': 'val',
            'load_arrays': False,
        }

        predictions = prediction_store.filter_predictions(**filter_kwargs)
        predictions = [
            p for p in predictions
            if p.get('step_idx', 0) < current_step
        ]

        available_models = sorted(set(
            p.get('model_name') for p in predictions if p.get('model_name')
        ))

        if not available_models:
            raise ValueError(
                "No model predictions found in prediction store. "
                "Ensure models were trained before merge_predictions. "
                "[Error: MERGE-E010]"
            )

        # Apply model filter
        if model_filter:
            selected_models = [m for m in available_models if m in model_filter]
            if not selected_models:
                logger.warning(
                    f"No models matched filter {model_filter}. "
                    f"Available: {available_models}"
                )
        else:
            selected_models = available_models

        logger.info(f"  Selected {len(selected_models)} models for prediction merge")

        # Collect predictions using OOF reconstruction
        # Use empty branch_contexts since we're not in branch mode
        config = MergeConfig(
            collect_predictions=True,
            prediction_branches="all",
            model_filter=selected_models,
            unsafe=False,  # Always use OOF for safety
        )

        # Create synthetic branch context for the reconstructor
        # This allows reuse of existing prediction collection logic
        n_samples = dataset.num_samples
        model_predictions: Dict[str, np.ndarray] = {}

        for model_name in selected_models:
            try:
                from nirs4all.controllers.models.stacking import (
                    TrainingSetReconstructor,
                    ReconstructorConfig,
                )
                from nirs4all.operators.models.meta import StackingConfig, CoverageStrategy

                stacking_config = StackingConfig(
                    coverage_strategy=CoverageStrategy.IMPUTE_MEAN,
                )
                reconstructor_config = ReconstructorConfig(
                    log_warnings=True,
                    validate_fold_alignment=False,
                )

                reconstructor = TrainingSetReconstructor(
                    prediction_store=prediction_store,
                    source_model_names=[model_name],
                    stacking_config=stacking_config,
                    reconstructor_config=reconstructor_config,
                )

                result = reconstructor.reconstruct(
                    dataset=dataset,
                    context=context,
                    use_proba=False,
                )

                # Combine train (OOF) and test predictions
                combined = np.full(n_samples, np.nan)

                # Get partition indices
                # IMPORTANT: Use include_augmented=False for train because OOF predictions
                # are only available for original (non-augmented) samples
                train_context = context.with_partition('train')
                train_ids = dataset._indexer.x_indices(
                    train_context.selector,
                    include_augmented=False,
                    include_excluded=False
                )

                test_context = context.with_partition('test')
                test_ids = dataset._indexer.x_indices(
                    test_context.selector,
                    include_augmented=False,
                    include_excluded=False
                )

                # Fill train (OOF) predictions
                if result.X_train_meta.size > 0:
                    train_preds = result.X_train_meta[:, 0] if result.X_train_meta.ndim > 1 else result.X_train_meta
                    if len(train_preds) == len(train_ids):
                        for i, sample_id in enumerate(train_ids):
                            combined[sample_id] = train_preds[i]

                # Fill test predictions
                if result.X_test_meta.size > 0:
                    test_preds = result.X_test_meta[:, 0] if result.X_test_meta.ndim > 1 else result.X_test_meta
                    if len(test_preds) == len(test_ids):
                        for i, sample_id in enumerate(test_ids):
                            combined[sample_id] = test_preds[i]

                # Propagate predictions from base samples to their augmented versions
                # Augmented samples should have the same prediction as their origin
                base_sample_ids = list(train_ids) + list(test_ids)
                if base_sample_ids:
                    augmented_ids = dataset._indexer._augmentation_tracker.get_augmented_for_origins(
                        base_sample_ids
                    )
                    for aug_id in augmented_ids:
                        origin_id = dataset._indexer._augmentation_tracker.get_origin_for_sample(aug_id)
                        if origin_id is not None and not np.isnan(combined[origin_id]):
                            combined[aug_id] = combined[origin_id]

                model_predictions[model_name] = combined
                logger.debug(f"  Collected predictions from model '{model_name}'")

            except Exception as e:
                logger.warning(
                    f"Failed to collect predictions from model '{model_name}': {e}"
                )
                continue

        if not model_predictions:
            raise ValueError(
                "Failed to collect predictions from any model. "
                "[Error: MERGE-E010]"
            )

        # Aggregate predictions based on strategy
        if aggregation == "separate":
            merged = PredictionAggregator.aggregate(
                predictions=model_predictions,
                strategy=AggregationStrategy.SEPARATE,
            )
        elif aggregation == "mean":
            merged = PredictionAggregator.aggregate(
                predictions=model_predictions,
                strategy=AggregationStrategy.MEAN,
            )
        elif aggregation == "weighted_mean":
            # Get model scores for weighting (use validation scores from store)
            model_selector = ModelSelector(
                prediction_store=prediction_store,
                context=context,
            )
            model_scores = model_selector.get_model_scores(
                model_names=selected_models,
                metric="rmse",
                branch_id=-1,  # No branch context
            )
            merged = PredictionAggregator.aggregate(
                predictions=model_predictions,
                strategy=AggregationStrategy.WEIGHTED_MEAN,
                model_scores=model_scores,
                metric="rmse",
            )
        else:
            # Default to separate
            merged = PredictionAggregator.aggregate(
                predictions=model_predictions,
                strategy=AggregationStrategy.SEPARATE,
            )

        # Store merged predictions as features
        dataset.add_merged_features(
            features=merged,
            processing_name="merged_predictions",
            source=0
        )

        # Build metadata
        metadata = {
            "merge_predictions": True,
            "models_used": selected_models,
            "aggregation": aggregation,
            "n_features": merged.shape[1],
            "merged_shape": merged.shape,
        }

        logger.success(
            f"Prediction merge completed: {len(selected_models)} models, "
            f"shape={merged.shape}"
        )

        return context.copy(), StepOutput(metadata=metadata)


# =============================================================================
# Phase 7: Static merge_branches method for MetaModel integration
# =============================================================================

    @classmethod
    def merge_branches(
        cls,
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        config: MergeConfig,
        prediction_store: Optional[Any] = None,
        mode: str = "train",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Static method for programmatic merge (used by MetaModel).

        This class method allows MetaModelController to delegate to merge logic
        without going through the full step execution machinery. It provides
        the core branch merging functionality without modifying the context
        or requiring a step_info object.

        This is the key integration point for Phase 7: MetaModel Refactoring.

        Args:
            dataset: SpectroDataset with sample data.
            context: Execution context with branch_contexts and state.
            config: MergeConfig specifying what to merge.
            prediction_store: Prediction storage for model predictions.
                Required if config.collect_predictions is True.
            mode: Execution mode ("train" or "predict").

        Returns:
            Tuple of (merged_features, info_dict) where:
                - merged_features: 2D numpy array (n_samples, n_features)
                - info_dict: Dictionary with merge metadata including:
                    - "merged_shape": Shape of merged features
                    - "feature_branches_used": List of branch indices for features
                    - "prediction_branches_used": List of branch indices for predictions
                    - "models_used": List of model names (if predictions)
                    - "oof_reconstruction": Whether OOF was used (if predictions)
                    - "unsafe_merge": True if unsafe mode was used

        Raises:
            ValueError: If not in branch mode or config is invalid.
            ValueError: If prediction_store is None but predictions requested.

        Example:
            >>> from nirs4all.controllers.data.merge import MergeController
            >>> from nirs4all.operators.data.merge import MergeConfig
            >>>
            >>> # Called from MetaModelController
            >>> config = MergeConfig(
            ...     collect_predictions=True,
            ...     prediction_branches="all",
            ... )
            >>> merged_X, info = MergeController.merge_branches(
            ...     dataset=dataset,
            ...     context=context,
            ...     config=config,
            ...     prediction_store=prediction_store,
            ... )
            >>> meta_model.fit(merged_X, y)

        Note:
            Unlike execute(), this method does NOT:
            - Exit branch mode (caller must handle this if needed)
            - Modify the context
            - Add merged features to the dataset
            - Return a StepOutput

            It simply performs the merge computation and returns the result.
        """
        # Create a controller instance for internal methods
        controller = cls()

        # Validate branch mode
        branch_contexts = context.custom.get("branch_contexts", [])
        in_branch_mode = context.custom.get("in_branch_mode", False)

        if not branch_contexts and not in_branch_mode:
            raise ValueError(
                "merge_branches requires active branch contexts. "
                "Use only after a branch step. "
                "[Error: MERGE-E020]"
            )

        n_branches = len(branch_contexts)

        # Validate branch indices in config
        controller._validate_branches(config, branch_contexts)

        # Log configuration
        controller._log_config(
            config=config,
            n_branches=n_branches,
            branch_contexts=branch_contexts,
            prediction_store=prediction_store,
            context=context,
        )

        merged_parts = []
        info: Dict[str, Any] = {}

        # Collect features if requested
        if config.collect_features:
            feature_branches = config.get_feature_branches(n_branches)
            features_list, feature_info = controller._collect_features(
                dataset=dataset,
                branch_contexts=branch_contexts,
                branch_indices=feature_branches,
                on_missing=config.on_missing,
                on_shape_mismatch=config.on_shape_mismatch,
            )

            if features_list:
                merged_parts.extend(features_list)
                info["feature_shapes"] = feature_info.get("shapes", [])
                info["feature_branches_used"] = feature_info.get("branches_used", [])
                logger.debug(
                    f"merge_branches: Collected features from {len(features_list)} branches"
                )

        # Collect predictions if requested
        if config.collect_predictions:
            predictions_array, pred_info = controller._collect_predictions(
                dataset=dataset,
                context=context,
                branch_contexts=branch_contexts,
                config=config,
                prediction_store=prediction_store,
                mode=mode,
            )

            if predictions_array is not None and predictions_array.size > 0:
                merged_parts.append(predictions_array)
                info["prediction_shape"] = predictions_array.shape
                info["prediction_models_used"] = pred_info.get("models_used", [])
                info["prediction_branches_used"] = pred_info.get("branches_used", [])
                info["oof_reconstruction"] = pred_info.get("oof_reconstruction", True)
                info["models_used"] = pred_info.get("models_used", [])
                logger.debug(
                    f"merge_branches: Collected predictions: shape={predictions_array.shape}"
                )

        # Include original pre-branch features if requested
        if config.include_original:
            original_features = controller._get_original_features(dataset, context)
            if original_features is not None:
                merged_parts.insert(0, original_features)
                info["include_original"] = True
                info["original_shape"] = original_features.shape

        # Concatenate all parts
        if not merged_parts:
            raise ValueError(
                "merge_branches resulted in empty output - check configuration. "
                "[Error: MERGE-E012]"
            )

        merged_features = np.concatenate(merged_parts, axis=1)
        info["merged_shape"] = merged_features.shape

        # Add unsafe warning if applicable
        if config.unsafe:
            info["unsafe_merge"] = True
            logger.warning(
                "⚠️ UNSAFE MERGE: OOF reconstruction disabled. "
                "Training predictions used directly, causing DATA LEAKAGE."
            )

        logger.info(
            f"merge_branches completed: shape={merged_features.shape}"
            f"{' [UNSAFE]' if config.unsafe else ''}"
        )

        return merged_features, info

    @classmethod
    def build_config_from_meta_model(
        cls,
        meta_operator: Any,
        context: "ExecutionContext",
        branch_contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> MergeConfig:
        """Build MergeConfig from MetaModel operator parameters.

        Translates MetaModel configuration to an equivalent MergeConfig
        for use with merge_branches(). This enables MetaModel to delegate
        to the centralized merge logic.

        This is a helper for Phase 7: MetaModel Refactoring.

        Args:
            meta_operator: MetaModel operator instance with configuration.
            context: Execution context with branch info.
            branch_contexts: Optional branch contexts for branch resolution.

        Returns:
            MergeConfig equivalent to the MetaModel's configuration.

        Example:
            >>> config = MergeController.build_config_from_meta_model(
            ...     meta_operator=meta_model,
            ...     context=context,
            ... )
            >>> merged_X, info = MergeController.merge_branches(
            ...     dataset=dataset,
            ...     context=context,
            ...     config=config,
            ...     prediction_store=prediction_store,
            ... )
        """
        from nirs4all.operators.models.meta import BranchScope

        config = MergeConfig(collect_predictions=True)

        # Map source_models
        source_models = getattr(meta_operator, 'source_models', 'all')
        if source_models == "all":
            config.prediction_branches = "all"
        elif isinstance(source_models, list):
            config.model_filter = source_models

        # Map branch_scope
        stacking_config = getattr(meta_operator, 'stacking_config', None)
        if stacking_config is not None:
            branch_scope = getattr(stacking_config, 'branch_scope', BranchScope.CURRENT_ONLY)
            if branch_scope == BranchScope.ALL_BRANCHES:
                config.prediction_branches = "all"
            elif branch_scope == BranchScope.CURRENT_ONLY:
                # Keep current branch only
                current_branch_id = getattr(context.selector, 'branch_id', None)
                if current_branch_id is not None and branch_contexts:
                    config.prediction_branches = [current_branch_id]
                else:
                    config.prediction_branches = "all"

        # Map use_proba
        config.use_proba = getattr(meta_operator, 'use_proba', False)

        # Map include_features if present (new capability)
        if getattr(meta_operator, 'include_features', False):
            config.collect_features = True
            config.include_original = True

        return config


# Expose parser and utilities for testing
__all__ = [
    "MergeController",
    "MergeConfigParser",
    "ModelSelector",
    "PredictionAggregator",
    "AsymmetricBranchAnalyzer",
    "BranchAnalysisResult",
    "AsymmetryReport",
    "SourceMergeConfig",
    # Phase 2: Disjoint sample branch merging
    "DisjointBranchAnalysis",
    "DisjointMergeResult",
    "is_disjoint_branch",
    "detect_disjoint_branches",
    # Phase 3: Disjoint merge metadata
    "DisjointBranchInfo",
    "DisjointMergeMetadata",
]
