"""
Branch Validator for Meta-Model Stacking (Phase 4).

This module provides validation logic for stacking in branched pipelines,
including support for:
- Preprocessing branches (same samples, different features)
- Sample partitioner branches (different sample subsets)
- Outlier excluder branches (same samples, different exclusions)
- Generator syntax branches (same samples, model variants)

The validator ensures that stacking is only performed in compatible
scenarios and provides clear error messages for unsupported cases.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import warnings

from .exceptions import (
    IncompatibleBranchTypeError,
    CrossPartitionStackingError,
    NestedBranchStackingError,
    FoldMismatchAcrossBranchesError,
    DisjointSampleSetsError,
    GeneratorSyntaxStackingWarning,
)

if TYPE_CHECKING:
    from nirs4all.data.predictions import Predictions
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext


class BranchType(Enum):
    """Types of branching in nirs4all pipelines."""

    NONE = "none"                           # No branching
    PREPROCESSING = "preprocessing"         # {"branch": {...}}
    SAMPLE_PARTITIONER = "sample_partitioner"  # {"branch": {"by": "sample_partitioner"}}
    METADATA_PARTITIONER = "metadata_partitioner"  # {"branch": {"by": "metadata_partitioner", "column": ...}}
    OUTLIER_EXCLUDER = "outlier_excluder"   # {"branch": {"by": "outlier_excluder"}}
    GENERATOR = "generator"                 # {"_or_": [...]} or generator syntax
    NESTED = "nested"                       # Multiple levels of branching
    UNKNOWN = "unknown"                     # Unrecognized branch type


class StackingCompatibility(Enum):
    """Compatibility level for stacking with a branch type."""

    COMPATIBLE = "compatible"               # Fully supported
    COMPATIBLE_WITH_WARNINGS = "compatible_with_warnings"  # Supported but with caveats
    WITHIN_PARTITION_ONLY = "within_partition_only"  # Only within same partition
    NOT_SUPPORTED = "not_supported"         # Not currently supported


@dataclass
class BranchInfo:
    """Information about branch context for stacking validation."""

    branch_type: BranchType
    branch_id: Optional[int] = None
    branch_name: Optional[str] = None
    branch_path: List[int] = field(default_factory=list)
    partition_info: Optional[Dict[str, Any]] = None
    exclusion_info: Optional[Dict[str, Any]] = None
    sample_indices: Optional[List[int]] = None
    n_samples: Optional[int] = None
    is_nested: bool = False
    nesting_depth: int = 0


@dataclass
class BranchValidationResult:
    """Result of branch validation for stacking."""

    is_valid: bool
    compatibility: StackingCompatibility
    branch_info: BranchInfo
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source_filter_hint: Optional[Dict[str, Any]] = None

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)


class BranchValidator:
    """Validates branch contexts for meta-model stacking.

    This validator checks that the current branch context is compatible
    with stacking and provides clear error messages for unsupported cases.

    Supported scenarios:
    - No branching: Fully compatible
    - Preprocessing branches: Stack within branch
    - Outlier excluder branches: Stack within branch (all samples have predictions)
    - Sample partitioner branches: Stack within partition only

    Unsupported or limited scenarios:
    - Cross-partition stacking with sample_partitioner
    - Deeply nested branching (depth > 2)
    - Generator syntax with large variant counts

    Example:
        >>> validator = BranchValidator(prediction_store)
        >>> result = validator.validate(context, source_model_names)
        >>> if not result.is_valid:
        ...     raise ValueError(result.errors[0])
    """

    # Maximum supported nesting depth
    MAX_NESTING_DEPTH = 2

    # Maximum variants before warning for generator syntax
    MAX_GENERATOR_VARIANTS_WARNING = 10

    def __init__(
        self,
        prediction_store: 'Predictions',
        log_warnings: bool = True
    ):
        """Initialize branch validator.

        Args:
            prediction_store: Predictions storage for analyzing branch data.
            log_warnings: If True, emit Python warnings for non-critical issues.
        """
        self.prediction_store = prediction_store
        self.log_warnings = log_warnings

    def validate(
        self,
        context: 'ExecutionContext',
        source_model_names: List[str],
        dataset: Optional['SpectroDataset'] = None
    ) -> BranchValidationResult:
        """Validate branch context for stacking compatibility.

        Args:
            context: Current execution context with branch info.
            source_model_names: List of source model names to validate.
            dataset: Optional dataset for sample index validation.

        Returns:
            BranchValidationResult with validation status and any errors.
        """
        # Extract branch information
        branch_info = self._extract_branch_info(context)

        # Initialize result
        result = BranchValidationResult(
            is_valid=True,
            compatibility=StackingCompatibility.COMPATIBLE,
            branch_info=branch_info
        )

        # No branching - fully compatible
        if branch_info.branch_type == BranchType.NONE:
            return result

        # Check nesting depth
        if branch_info.nesting_depth > self.MAX_NESTING_DEPTH:
            result.add_error(
                f"Deeply nested branching (depth={branch_info.nesting_depth}) "
                f"is not fully supported for stacking. "
                f"Maximum supported depth is {self.MAX_NESTING_DEPTH}."
            )
            result.compatibility = StackingCompatibility.NOT_SUPPORTED
            return result

        # Validate based on branch type
        if branch_info.branch_type == BranchType.SAMPLE_PARTITIONER:
            self._validate_sample_partitioner(context, source_model_names, result)
        elif branch_info.branch_type == BranchType.METADATA_PARTITIONER:
            self._validate_metadata_partitioner(context, source_model_names, result)
        elif branch_info.branch_type == BranchType.OUTLIER_EXCLUDER:
            self._validate_outlier_excluder(context, source_model_names, result)
        elif branch_info.branch_type == BranchType.PREPROCESSING:
            self._validate_preprocessing_branch(context, source_model_names, result)
        elif branch_info.branch_type == BranchType.GENERATOR:
            self._validate_generator_syntax(context, source_model_names, result)
        elif branch_info.branch_type == BranchType.NESTED:
            self._validate_nested_branching(context, source_model_names, result)

        # Validate fold alignment across source models
        if result.is_valid and source_model_names:
            self._validate_fold_alignment_in_branch(
                context, source_model_names, result
            )

        # Emit warnings if configured
        if self.log_warnings:
            for warning in result.warnings:
                warnings.warn(warning)

        return result

    def _extract_branch_info(self, context: 'ExecutionContext') -> BranchInfo:
        """Extract branch information from execution context.

        Args:
            context: Execution context with branch data.

        Returns:
            BranchInfo with detected branch type and metadata.
        """
        selector = context.selector
        custom = context.custom

        branch_id = getattr(selector, 'branch_id', None)
        branch_name = getattr(selector, 'branch_name', None)
        branch_path = getattr(selector, 'branch_path', None) or []

        # Detect branch type from custom context
        if custom.get('metadata_partitioner_active'):
            branch_type = BranchType.METADATA_PARTITIONER
            partition_info = custom.get('metadata_partition', {})
            sample_indices = partition_info.get('sample_indices', [])
            n_samples = partition_info.get('n_samples', len(sample_indices))

            return BranchInfo(
                branch_type=branch_type,
                branch_id=branch_id,
                branch_name=branch_name,
                branch_path=list(branch_path),
                partition_info=partition_info,
                sample_indices=sample_indices,
                n_samples=n_samples,
                is_nested=len(branch_path) > 1,
                nesting_depth=len(branch_path)
            )

        if custom.get('sample_partitioner_active'):
            branch_type = BranchType.SAMPLE_PARTITIONER
            partition_info = custom.get('sample_partition', {})
            sample_indices = partition_info.get('sample_indices', [])
            n_samples = partition_info.get('n_samples', len(sample_indices))

            return BranchInfo(
                branch_type=branch_type,
                branch_id=branch_id,
                branch_name=branch_name,
                branch_path=list(branch_path),
                partition_info=partition_info,
                sample_indices=sample_indices,
                n_samples=n_samples,
                is_nested=len(branch_path) > 1,
                nesting_depth=len(branch_path)
            )

        if custom.get('outlier_excluder_active'):
            exclusion_info = custom.get('outlier_exclusion', {}) or custom.get('exclusion_info', {})

            return BranchInfo(
                branch_type=BranchType.OUTLIER_EXCLUDER,
                branch_id=branch_id,
                branch_name=branch_name,
                branch_path=list(branch_path),
                exclusion_info=exclusion_info,
                is_nested=len(branch_path) > 1,
                nesting_depth=len(branch_path)
            )

        if custom.get('in_branch_mode'):
            # Generic branching (preprocessing or generator)
            branch_contexts = custom.get('branch_contexts', [])

            # Check if this looks like generator syntax
            if self._looks_like_generator_syntax(branch_contexts):
                return BranchInfo(
                    branch_type=BranchType.GENERATOR,
                    branch_id=branch_id,
                    branch_name=branch_name,
                    branch_path=list(branch_path),
                    is_nested=len(branch_path) > 1,
                    nesting_depth=len(branch_path)
                )

            # Check for nested branching
            if len(branch_path) > 1:
                return BranchInfo(
                    branch_type=BranchType.NESTED,
                    branch_id=branch_id,
                    branch_name=branch_name,
                    branch_path=list(branch_path),
                    is_nested=True,
                    nesting_depth=len(branch_path)
                )

            # Default to preprocessing branch
            return BranchInfo(
                branch_type=BranchType.PREPROCESSING,
                branch_id=branch_id,
                branch_name=branch_name,
                branch_path=list(branch_path),
                is_nested=len(branch_path) > 1,
                nesting_depth=len(branch_path)
            )

        # No branching
        if branch_id is None:
            return BranchInfo(branch_type=BranchType.NONE)

        # Branch context present but type unknown
        return BranchInfo(
            branch_type=BranchType.UNKNOWN,
            branch_id=branch_id,
            branch_name=branch_name,
            branch_path=list(branch_path),
            nesting_depth=len(branch_path) if branch_path else 0
        )

    def _looks_like_generator_syntax(
        self,
        branch_contexts: List[Dict[str, Any]]
    ) -> bool:
        """Check if branch contexts look like generator syntax.

        Generator syntax typically creates many branches with similar names
        like 'variant_0', 'variant_1', etc.

        Args:
            branch_contexts: List of branch context dictionaries.

        Returns:
            True if this looks like generator syntax.
        """
        if len(branch_contexts) <= 1:
            return False

        # Check for variant-like naming patterns
        names = [bc.get('name', '') for bc in branch_contexts]
        variant_patterns = ['variant_', 'or_', 'gen_', 'config_']

        for pattern in variant_patterns:
            matching = sum(1 for n in names if n.startswith(pattern))
            if matching == len(names):
                return True

        return False

    def _validate_sample_partitioner(
        self,
        context: 'ExecutionContext',
        source_model_names: List[str],
        result: BranchValidationResult
    ) -> None:
        """Validate sample partitioner branching for stacking.

        Sample partitioner creates disjoint sample sets, so stacking
        is only valid within the same partition.

        Args:
            context: Execution context.
            source_model_names: Source model names.
            result: Validation result to update.
        """
        result.compatibility = StackingCompatibility.WITHIN_PARTITION_ONLY

        current_branch_id = result.branch_info.branch_id
        current_partition = result.branch_info.partition_info or {}
        partition_type = current_partition.get('partition_type', 'unknown')

        # Get sample indices for current partition
        current_samples = set(result.branch_info.sample_indices or [])

        if not current_samples:
            result.add_warning(
                f"Sample partitioner branch '{partition_type}' has no sample "
                f"indices recorded. Stacking will proceed but may have issues."
            )

        # Check source models are from same partition
        for model_name in source_model_names:
            model_preds = self.prediction_store.filter_predictions(
                model_name=model_name,
                load_arrays=False
            )

            for pred in model_preds:
                pred_branch_id = pred.get('branch_id')
                pred_samples = set(pred.get('sample_indices', []))

                # Skip if no sample indices
                if not pred_samples:
                    continue

                # Check if from different branch
                if pred_branch_id is not None and pred_branch_id != current_branch_id:
                    # Check sample overlap
                    if current_samples and pred_samples:
                        overlap = current_samples & pred_samples
                        overlap_ratio = len(overlap) / max(len(current_samples), 1)

                        if overlap_ratio < 0.1:  # Less than 10% overlap
                            result.add_error(
                                f"Source model '{model_name}' is from a different "
                                f"partition with disjoint samples (only {100*overlap_ratio:.1f}% overlap). "
                                f"Cross-partition stacking is not supported with sample_partitioner. "
                                f"Stack only with models from the current '{partition_type}' partition."
                            )
                            return

        # Add hint for source filtering
        result.source_filter_hint = {
            'branch_id': current_branch_id,
            'partition_type': partition_type
        }

        result.add_warning(
            f"Stacking within sample_partitioner partition '{partition_type}'. "
            f"Only models from the same partition will be used as sources."
        )

    def _validate_metadata_partitioner(
        self,
        context: 'ExecutionContext',
        source_model_names: List[str],
        result: BranchValidationResult
    ) -> None:
        """Validate metadata partitioner branching for stacking.

        Metadata partitioner creates disjoint sample sets based on metadata
        column values. Stacking is only valid within the same partition.

        Args:
            context: Execution context.
            source_model_names: Source model names.
            result: Validation result to update.
        """
        result.compatibility = StackingCompatibility.WITHIN_PARTITION_ONLY

        current_branch_id = result.branch_info.branch_id
        current_partition = result.branch_info.partition_info or {}
        partition_value = current_partition.get('partition_value', 'unknown')
        column_name = current_partition.get('column', 'unknown')

        # Get sample indices for current partition
        current_samples = set(result.branch_info.sample_indices or [])

        if not current_samples:
            result.add_warning(
                f"Metadata partitioner branch '{partition_value}' has no sample "
                f"indices recorded. Stacking will proceed but may have issues."
            )

        # Check source models are from same partition
        for model_name in source_model_names:
            model_preds = self.prediction_store.filter_predictions(
                model_name=model_name,
                load_arrays=False
            )

            for pred in model_preds:
                pred_branch_id = pred.get('branch_id')
                pred_samples = set(pred.get('sample_indices', []))

                # Skip if no sample indices
                if not pred_samples:
                    continue

                # Check if from different branch
                if pred_branch_id is not None and pred_branch_id != current_branch_id:
                    # Check sample overlap
                    if current_samples and pred_samples:
                        overlap = current_samples & pred_samples
                        overlap_ratio = len(overlap) / max(len(current_samples), 1)

                        if overlap_ratio < 0.1:  # Less than 10% overlap
                            result.add_error(
                                f"Source model '{model_name}' is from a different "
                                f"metadata partition with disjoint samples "
                                f"(only {100*overlap_ratio:.1f}% overlap). "
                                f"Cross-partition stacking is not supported with metadata_partitioner. "
                                f"Stack only with models from the current '{column_name}={partition_value}' partition."
                            )
                            return

        # Add hint for source filtering
        result.source_filter_hint = {
            'branch_id': current_branch_id,
            'partition_value': partition_value,
            'column': column_name
        }

        result.add_warning(
            f"Stacking within metadata_partitioner partition '{column_name}={partition_value}'. "
            f"Only models from the same partition will be used as sources."
        )

    def _validate_outlier_excluder(
        self,
        context: 'ExecutionContext',
        source_model_names: List[str],
        result: BranchValidationResult
    ) -> None:
        """Validate outlier excluder branching for stacking.

        Outlier excluder creates branches where all samples have predictions
        (some just weren't used in training). This is fully compatible with stacking.

        Args:
            context: Execution context.
            source_model_names: Source model names.
            result: Validation result to update.
        """
        result.compatibility = StackingCompatibility.COMPATIBLE

        exclusion_info = result.branch_info.exclusion_info or {}
        n_excluded = exclusion_info.get('n_excluded', 0)
        strategy = exclusion_info.get('strategy', {})
        method = strategy.get('method', 'unknown') if strategy else 'baseline'

        if n_excluded > 0:
            result.add_warning(
                f"Stacking with outlier_excluder branch (method='{method}', "
                f"excluded={n_excluded} samples from training). "
                f"All samples have predictions, but some were not used in training."
            )

    def _validate_preprocessing_branch(
        self,
        context: 'ExecutionContext',
        source_model_names: List[str],
        result: BranchValidationResult
    ) -> None:
        """Validate preprocessing branching for stacking.

        Preprocessing branches have the same samples but different features.
        This is fully compatible with stacking within the branch.

        Args:
            context: Execution context.
            source_model_names: Source model names.
            result: Validation result to update.
        """
        result.compatibility = StackingCompatibility.COMPATIBLE

        branch_name = result.branch_info.branch_name or f"branch_{result.branch_info.branch_id}"

        result.add_warning(
            f"Stacking within preprocessing branch '{branch_name}'. "
            f"Only models from this branch will be used as sources."
        )

    def _validate_generator_syntax(
        self,
        context: 'ExecutionContext',
        source_model_names: List[str],
        result: BranchValidationResult
    ) -> None:
        """Validate generator syntax for stacking.

        Generator syntax creates multiple model variants. This is compatible
        but may result in high-dimensional meta-features.

        Args:
            context: Execution context.
            source_model_names: Source model names.
            result: Validation result to update.
        """
        result.compatibility = StackingCompatibility.COMPATIBLE_WITH_WARNINGS

        n_sources = len(source_model_names)
        if n_sources > self.MAX_GENERATOR_VARIANTS_WARNING:
            result.add_warning(
                f"Generator syntax created {n_sources} model variants for stacking. "
                f"This may result in high-dimensional meta-features. "
                f"Consider using TopKByMetricSelector to limit sources."
            )

    def _validate_nested_branching(
        self,
        context: 'ExecutionContext',
        source_model_names: List[str],
        result: BranchValidationResult
    ) -> None:
        """Validate nested branching for stacking.

        Nested branching may have limited support depending on the
        combination of branch types.

        Args:
            context: Execution context.
            source_model_names: Source model names.
            result: Validation result to update.
        """
        depth = result.branch_info.nesting_depth
        path = result.branch_info.branch_path

        if depth > self.MAX_NESTING_DEPTH:
            result.add_error(
                f"Nested branching depth {depth} exceeds maximum supported ({self.MAX_NESTING_DEPTH}). "
                f"Consider simplifying the pipeline structure."
            )
            result.compatibility = StackingCompatibility.NOT_SUPPORTED
            return

        result.compatibility = StackingCompatibility.COMPATIBLE_WITH_WARNINGS
        result.add_warning(
            f"Stacking with nested branching (depth={depth}, path={path}). "
            f"Only models from the current branch path will be used as sources."
        )

    def _validate_fold_alignment_in_branch(
        self,
        context: 'ExecutionContext',
        source_model_names: List[str],
        result: BranchValidationResult
    ) -> None:
        """Validate that fold structures are consistent within branch.

        Args:
            context: Execution context.
            source_model_names: Source model names.
            result: Validation result to update.
        """
        current_branch_id = result.branch_info.branch_id
        current_step = context.state.step_number

        fold_counts: Dict[str, int] = {}

        for model_name in source_model_names:
            filter_kwargs = {
                'model_name': model_name,
                'partition': 'val',
                'load_arrays': False,
            }
            if current_branch_id is not None:
                filter_kwargs['branch_id'] = current_branch_id

            preds = self.prediction_store.filter_predictions(**filter_kwargs)
            preds = [p for p in preds if p.get('step_idx', 0) < current_step]

            # Count unique fold IDs (excluding avg/w_avg)
            fold_ids = {
                str(p.get('fold_id'))
                for p in preds
                if str(p.get('fold_id', '')) not in ('avg', 'w_avg', 'None', '')
            }

            if fold_ids:
                fold_counts[model_name] = len(fold_ids)

        # Check consistency
        if fold_counts:
            unique_counts = set(fold_counts.values())
            if len(unique_counts) > 1:
                result.add_warning(
                    f"Source models have different fold counts: "
                    f"{dict(fold_counts)}. This may affect OOF reconstruction."
                )

    def validate_sample_alignment(
        self,
        source_model_names: List[str],
        expected_sample_indices: List[int],
        context: 'ExecutionContext'
    ) -> BranchValidationResult:
        """Validate that source models have predictions for expected samples.

        This is particularly important for sample_partitioner branches where
        different partitions have different samples.

        Args:
            source_model_names: List of source model names.
            expected_sample_indices: Expected sample indices (from current partition).
            context: Execution context.

        Returns:
            Validation result with any sample alignment issues.
        """
        branch_info = self._extract_branch_info(context)
        result = BranchValidationResult(
            is_valid=True,
            compatibility=StackingCompatibility.COMPATIBLE,
            branch_info=branch_info
        )

        expected_set = set(expected_sample_indices)
        current_branch_id = getattr(context.selector, 'branch_id', None)
        current_step = context.state.step_number

        for model_name in source_model_names:
            filter_kwargs = {
                'model_name': model_name,
                'partition': 'val',
                'load_arrays': False,
            }
            if current_branch_id is not None:
                filter_kwargs['branch_id'] = current_branch_id

            preds = self.prediction_store.filter_predictions(**filter_kwargs)
            preds = [p for p in preds if p.get('step_idx', 0) < current_step]

            # Collect all sample indices from predictions
            found_samples = set()
            for pred in preds:
                sample_indices = pred.get('sample_indices', [])
                if sample_indices is not None:
                    if hasattr(sample_indices, 'tolist'):
                        sample_indices = sample_indices.tolist()
                    found_samples.update(sample_indices)

            if found_samples:
                overlap = expected_set & found_samples
                overlap_ratio = len(overlap) / max(len(expected_set), 1)

                if overlap_ratio < 0.5:  # Less than 50% overlap
                    result.add_error(
                        f"Source model '{model_name}' has low sample overlap "
                        f"({100*overlap_ratio:.1f}%). Expected {len(expected_set)} samples, "
                        f"found predictions for {len(found_samples)} samples with "
                        f"{len(overlap)} overlapping."
                    )

        return result


def detect_branch_type(context: 'ExecutionContext') -> BranchType:
    """Detect the type of branching from execution context.

    Convenience function for quick branch type detection.

    Args:
        context: Execution context with branch info.

    Returns:
        Detected BranchType enum value.
    """
    custom = context.custom

    if custom.get('metadata_partitioner_active'):
        return BranchType.METADATA_PARTITIONER
    if custom.get('sample_partitioner_active'):
        return BranchType.SAMPLE_PARTITIONER
    if custom.get('outlier_excluder_active'):
        return BranchType.OUTLIER_EXCLUDER
    if custom.get('in_branch_mode'):
        branch_path = getattr(context.selector, 'branch_path', None) or []
        if len(branch_path) > 1:
            return BranchType.NESTED
        return BranchType.PREPROCESSING

    branch_id = getattr(context.selector, 'branch_id', None)
    if branch_id is None:
        return BranchType.NONE

    return BranchType.UNKNOWN


def is_stacking_compatible(context: 'ExecutionContext') -> bool:
    """Quick check if stacking is compatible with current context.

    Args:
        context: Execution context.

    Returns:
        True if stacking is likely compatible.
    """
    branch_type = detect_branch_type(context)

    # Always compatible
    if branch_type in (BranchType.NONE, BranchType.PREPROCESSING,
                        BranchType.OUTLIER_EXCLUDER):
        return True

    # Compatible within partition
    if branch_type in (BranchType.SAMPLE_PARTITIONER, BranchType.METADATA_PARTITIONER):
        return True  # But only within partition

    # Limited support
    if branch_type == BranchType.GENERATOR:
        return True

    # Check nesting depth
    if branch_type == BranchType.NESTED:
        branch_path = getattr(context.selector, 'branch_path', None) or []
        return len(branch_path) <= BranchValidator.MAX_NESTING_DEPTH

    return True


def is_disjoint_branch(context: 'ExecutionContext') -> bool:
    """Check if the current branch context represents disjoint sample branching.

    Disjoint branches partition samples into non-overlapping sets, where each
    sample exists in exactly ONE branch. This is in contrast to copy branches
    where all branches see all samples.

    Disjoint branch types:
        - METADATA_PARTITIONER: Branches by metadata column value
        - SAMPLE_PARTITIONER: Branches by outlier status

    Copy branch types:
        - PREPROCESSING: All branches see all samples
        - GENERATOR: All branches see all samples (model variants)

    Args:
        context: Execution context with branch info.

    Returns:
        True if current context represents a disjoint sample branch.
    """
    branch_type = detect_branch_type(context)

    # Disjoint branch types
    if branch_type in (BranchType.METADATA_PARTITIONER, BranchType.SAMPLE_PARTITIONER):
        return True

    # Check for explicit markers in context.custom
    custom = context.custom

    # metadata_partition indicates disjoint samples by metadata column
    if custom.get('metadata_partition') is not None:
        return True

    # sample_partition indicates disjoint samples by filter (outliers/inliers)
    if custom.get('sample_partition') is not None:
        return True

    return False


def get_disjoint_branch_info(context: 'ExecutionContext') -> Optional[Dict[str, Any]]:
    """Get information about the disjoint branch if applicable.

    Args:
        context: Execution context with branch info.

    Returns:
        Dict with partition info, or None if not a disjoint branch.
        Keys may include:
            - partition_type: "metadata" or "sample"
            - column: Metadata column name (for metadata partitioner)
            - partition_value: Value(s) for this partition
            - sample_indices: List of sample indices in this partition
            - n_samples: Number of samples in this partition
    """
    if not is_disjoint_branch(context):
        return None

    custom = context.custom

    # Check for metadata_partition
    metadata_partition = custom.get('metadata_partition')
    if metadata_partition is not None:
        return {
            'partition_type': 'metadata',
            'column': metadata_partition.get('column'),
            'partition_value': metadata_partition.get('partition_value'),
            'partition_values': metadata_partition.get('partition_values'),
            'sample_indices': metadata_partition.get('sample_indices', []),
            'train_sample_indices': metadata_partition.get('train_sample_indices', []),
            'n_samples': metadata_partition.get('n_samples', 0),
            'n_train_samples': metadata_partition.get('n_train_samples', 0),
        }

    # Check for sample_partition
    sample_partition = custom.get('sample_partition')
    if sample_partition is not None:
        return {
            'partition_type': 'sample',
            'partition_value': sample_partition.get('partition_type'),  # e.g., "outliers" or "inliers"
            'sample_indices': sample_partition.get('sample_indices', []),
            'n_samples': sample_partition.get('n_samples', 0),
            'filter_config': sample_partition.get('filter_config'),
        }

    return None
