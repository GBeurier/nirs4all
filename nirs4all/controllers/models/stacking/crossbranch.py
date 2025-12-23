"""
Cross-Branch Stacking Support (Phase 7).

This module provides support for stacking across multiple branches,
allowing meta-models to use predictions from models in different
preprocessing branches.

Key Features:
1. BranchScope.ALL_BRANCHES support - stack across all branches
2. Feature alignment validation - ensure samples match across branches
3. Branch compatibility detection - identify compatible vs incompatible branches
4. Cross-branch prediction aggregation - combine predictions from multiple branches

Compatibility Matrix:
| Branch Type         | Cross-Branch Stacking | Notes                           |
|---------------------|----------------------|----------------------------------|
| Preprocessing       | ✅ Supported          | Same samples, different features |
| Generator (_or_)    | ✅ Supported          | Same samples, model variants     |
| Outlier Excluder    | ✅ Supported          | Same samples, different training |
| Sample Partitioner  | ❌ Not Supported      | Different samples per partition  |

Example:
    >>> validator = CrossBranchValidator(prediction_store)
    >>> result = validator.validate_cross_branch_stacking(
    ...     source_candidates=candidates,
    ...     context=context
    ... )
    >>> if result.is_compatible:
    ...     aligned_features = validator.align_branch_features(...)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import warnings
import numpy as np

from .branch_validator import BranchType, detect_branch_type
from .exceptions import (
    IncompatibleBranchSamplesError,
    BranchFeatureAlignmentError,
    CrossPartitionStackingError,
)

if TYPE_CHECKING:
    from nirs4all.data.predictions import Predictions
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext
    from nirs4all.operators.models.selection import ModelCandidate


class CrossBranchCompatibility(Enum):
    """Compatibility level for cross-branch stacking."""

    COMPATIBLE = "compatible"
    COMPATIBLE_WITH_ALIGNMENT = "compatible_with_alignment"
    INCOMPATIBLE_SAMPLES = "incompatible_samples"
    INCOMPATIBLE_PARTITIONS = "incompatible_partitions"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class BranchPredictionInfo:
    """Information about predictions from a specific branch.

    Attributes:
        branch_id: Unique branch identifier.
        branch_name: Human-readable branch name.
        model_names: List of model names in this branch.
        sample_indices: Set of sample indices with predictions.
        n_samples: Number of samples.
        n_folds: Number of folds.
        branch_type: Type of branching.
    """
    branch_id: int
    branch_name: Optional[str]
    model_names: List[str]
    sample_indices: Set[int]
    n_samples: int
    n_folds: int
    branch_type: BranchType = BranchType.UNKNOWN


@dataclass
class CrossBranchValidationResult:
    """Result of cross-branch stacking validation.

    Attributes:
        is_compatible: Whether cross-branch stacking is possible.
        compatibility: Detailed compatibility level.
        branches: Dict of BranchPredictionInfo by branch_id.
        common_samples: Set of samples present in all branches.
        alignment_issues: List of alignment problems found.
        warnings: List of warning messages.
        errors: List of error messages.
    """
    is_compatible: bool = True
    compatibility: CrossBranchCompatibility = CrossBranchCompatibility.NOT_APPLICABLE
    branches: Dict[int, BranchPredictionInfo] = field(default_factory=dict)
    common_samples: Set[int] = field(default_factory=set)
    alignment_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        """Add an error and mark as incompatible."""
        self.errors.append(message)
        self.is_compatible = False

    @property
    def total_models(self) -> int:
        """Total number of models across all branches."""
        return sum(len(b.model_names) for b in self.branches.values())


class CrossBranchValidator:
    """Validates and supports cross-branch stacking.

    This validator checks that stacking across multiple branches is feasible
    and provides utilities for aligning predictions from different branches.

    Attributes:
        prediction_store: Predictions storage.
        log_warnings: Whether to emit Python warnings.
    """

    def __init__(
        self,
        prediction_store: 'Predictions',
        log_warnings: bool = True
    ):
        """Initialize cross-branch validator.

        Args:
            prediction_store: Predictions storage.
            log_warnings: Whether to emit Python warnings.
        """
        self.prediction_store = prediction_store
        self.log_warnings = log_warnings

    def validate_cross_branch_stacking(
        self,
        source_candidates: List['ModelCandidate'],
        context: 'ExecutionContext',
        dataset: Optional['SpectroDataset'] = None
    ) -> CrossBranchValidationResult:
        """Validate cross-branch stacking feasibility.

        Checks that all branches have compatible samples and predictions
        can be properly aligned for stacking.

        Args:
            source_candidates: List of candidate source models.
            context: Execution context.
            dataset: Optional dataset for sample validation.

        Returns:
            CrossBranchValidationResult with compatibility info.
        """
        result = CrossBranchValidationResult()

        # Group candidates by branch
        branch_groups = self._group_by_branch(source_candidates)

        if len(branch_groups) <= 1:
            result.compatibility = CrossBranchCompatibility.NOT_APPLICABLE
            result.add_warning("Only one branch found, cross-branch stacking not needed")
            return result

        # Collect branch info
        for branch_id, candidates in branch_groups.items():
            # Skip None branch_id (shouldn't happen with ALL_BRANCHES scope)
            if branch_id is None:
                continue
            branch_info = self._collect_branch_info(branch_id, candidates, context)
            result.branches[branch_id] = branch_info

        # Check for sample partitioner branches
        has_sample_partitioner = any(
            b.branch_type == BranchType.SAMPLE_PARTITIONER
            for b in result.branches.values()
        )

        if has_sample_partitioner:
            result.compatibility = CrossBranchCompatibility.INCOMPATIBLE_PARTITIONS
            result.add_error(
                "Cross-branch stacking not supported with sample_partitioner branches. "
                "Different partitions have disjoint sample sets."
            )
            return result

        # Check sample alignment across branches
        all_sample_sets = [b.sample_indices for b in result.branches.values()]

        if not all_sample_sets:
            result.add_warning("No sample indices found in any branch")
            result.compatibility = CrossBranchCompatibility.COMPATIBLE
            return result

        # Check if all sample sets are empty - this can happen if sample_indices
        # aren't stored in prediction metadata. In this case, proceed with warning.
        if all(len(s) == 0 for s in all_sample_sets):
            result.add_warning(
                "No sample indices available for cross-branch validation. "
                "Proceeding with cross-branch stacking."
            )
            result.compatibility = CrossBranchCompatibility.COMPATIBLE
            return result

        # Find common samples across all branches
        # Filter out empty sets before intersection
        non_empty_sets = [s for s in all_sample_sets if len(s) > 0]
        if non_empty_sets:
            result.common_samples = set.intersection(*non_empty_sets)
        else:
            result.common_samples = set()

        # Check sample coverage
        total_samples = set.union(*all_sample_sets) if all_sample_sets else set()
        coverage_ratio = len(result.common_samples) / max(len(total_samples), 1)

        if coverage_ratio < 0.9:  # Less than 90% common
            if coverage_ratio < 0.5:  # Less than 50% - likely incompatible
                result.compatibility = CrossBranchCompatibility.INCOMPATIBLE_SAMPLES
                result.add_error(
                    f"Branches have low sample overlap ({coverage_ratio:.1%}). "
                    f"Cross-branch stacking requires compatible sample sets."
                )
                return result
            else:
                result.compatibility = CrossBranchCompatibility.COMPATIBLE_WITH_ALIGNMENT
                result.add_warning(
                    f"Branches have {coverage_ratio:.1%} sample overlap. "
                    f"Some samples may be dropped during cross-branch stacking."
                )
        else:
            result.compatibility = CrossBranchCompatibility.COMPATIBLE

        # Check fold alignment
        fold_counts = {b.branch_id: b.n_folds for b in result.branches.values()}
        unique_fold_counts = set(fold_counts.values())

        if len(unique_fold_counts) > 1:
            result.alignment_issues.append(
                f"Different fold counts across branches: {fold_counts}"
            )
            result.add_warning(
                f"Branches have different fold counts: {fold_counts}. "
                f"OOF reconstruction may be affected."
            )

        # Emit warnings
        if self.log_warnings:
            for warning in result.warnings:
                warnings.warn(warning)

        return result

    def get_cross_branch_sources(
        self,
        source_candidates: List['ModelCandidate'],
        context: 'ExecutionContext'
    ) -> List['ModelCandidate']:
        """Get source models from all branches for cross-branch stacking.

        Filters and orders candidates for cross-branch stacking,
        ensuring proper handling of branch-specific models.

        Args:
            source_candidates: All candidate source models.
            context: Execution context.

        Returns:
            Filtered and ordered list of candidates for cross-branch stacking.
        """
        current_step = context.state.step_number

        # Filter to earlier steps only
        valid_candidates = [
            c for c in source_candidates
            if c.step_idx < current_step
        ]

        # Remove duplicates (same model appearing in multiple folds)
        seen = set()
        unique_candidates = []
        for c in valid_candidates:
            key = (c.model_name, c.branch_id)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        # Sort by branch_id then step_idx for consistent ordering
        unique_candidates.sort(key=lambda c: (
            c.branch_id or 0,
            c.step_idx,
            c.model_name
        ))

        return unique_candidates

    def align_branch_features(
        self,
        branch_features: Dict[int, np.ndarray],
        branch_sample_indices: Dict[int, List[int]],
        target_sample_indices: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align features from multiple branches to common sample order.

        Combines features from different branches into a single feature matrix,
        aligning samples to a common order.

        Args:
            branch_features: Dict mapping branch_id to feature matrix.
            branch_sample_indices: Dict mapping branch_id to sample indices.
            target_sample_indices: Target sample order for output.

        Returns:
            Tuple of (aligned_features, valid_mask).

        Raises:
            BranchFeatureAlignmentError: If alignment fails.
        """
        n_samples = len(target_sample_indices)
        target_id_to_pos = {int(sid): pos for pos, sid in enumerate(target_sample_indices)}

        # Calculate total features
        n_total_features = sum(f.shape[1] if f.ndim > 1 else 1 for f in branch_features.values())

        # Initialize output
        aligned = np.full((n_samples, n_total_features), np.nan)
        valid_mask = np.ones(n_samples, dtype=bool)

        feat_col = 0
        alignment_issues = []

        for branch_id in sorted(branch_features.keys()):
            features = branch_features[branch_id]
            indices = branch_sample_indices.get(branch_id, [])

            n_branch_features = features.shape[1] if features.ndim > 1 else 1

            if len(indices) != len(features):
                alignment_issues.append(
                    f"Branch {branch_id}: sample count mismatch "
                    f"({len(indices)} indices, {len(features)} features)"
                )
                continue

            # Map features to target positions
            for i, sample_idx in enumerate(indices):
                pos = target_id_to_pos.get(int(sample_idx))
                if pos is not None:
                    if features.ndim == 1:
                        aligned[pos, feat_col] = features[i]
                    else:
                        aligned[pos, feat_col:feat_col + n_branch_features] = features[i]

            feat_col += n_branch_features

        # Update valid mask for samples with missing data
        valid_mask = ~np.isnan(aligned).any(axis=1)

        if alignment_issues:
            raise BranchFeatureAlignmentError(
                expected_features=n_total_features,
                branch_features={
                    bid: f.shape[1] if f.ndim > 1 else 1
                    for bid, f in branch_features.items()
                },
                alignment_issues=alignment_issues
            )

        return aligned, valid_mask

    def _group_by_branch(
        self,
        candidates: List['ModelCandidate']
    ) -> Dict[Optional[int], List['ModelCandidate']]:
        """Group candidates by branch_id.

        Args:
            candidates: List of model candidates.

        Returns:
            Dict mapping branch_id to list of candidates.
        """
        groups: Dict[Optional[int], List['ModelCandidate']] = {}

        for c in candidates:
            branch_id = c.branch_id
            if branch_id not in groups:
                groups[branch_id] = []
            groups[branch_id].append(c)

        return groups

    def _collect_branch_info(
        self,
        branch_id: int,
        candidates: List['ModelCandidate'],
        context: 'ExecutionContext'
    ) -> BranchPredictionInfo:
        """Collect information about predictions in a branch.

        Args:
            branch_id: Branch identifier.
            candidates: Candidates from this branch.
            context: Execution context.

        Returns:
            BranchPredictionInfo with branch metadata.
        """
        model_names = list(dict.fromkeys(c.model_name for c in candidates))
        sample_indices: Set[int] = set()
        fold_ids: Set[str] = set()
        branch_name = None

        for c in candidates:
            if c.branch_name and not branch_name:
                branch_name = c.branch_name

        # Get sample indices from predictions
        for model_name in model_names:
            preds = self.prediction_store.filter_predictions(
                model_name=model_name,
                branch_id=branch_id,
                partition='val',
                load_arrays=False
            )

            for pred in preds:
                indices = pred.get('sample_indices', [])
                if indices is not None:
                    if hasattr(indices, 'tolist'):
                        indices = indices.tolist()
                    sample_indices.update(int(i) for i in indices)

                fold_id = pred.get('fold_id')
                if fold_id and str(fold_id) not in ('avg', 'w_avg'):
                    fold_ids.add(str(fold_id))

        # Detect branch type from context or candidates
        branch_type = BranchType.PREPROCESSING
        if candidates:
            first_candidate = candidates[0]
            # Check for sample_partitioner markers
            if hasattr(first_candidate, 'partition_type'):
                branch_type = BranchType.SAMPLE_PARTITIONER

        return BranchPredictionInfo(
            branch_id=branch_id,
            branch_name=branch_name,
            model_names=model_names,
            sample_indices=sample_indices,
            n_samples=len(sample_indices),
            n_folds=len(fold_ids),
            branch_type=branch_type
        )


def validate_all_branches_scope(
    prediction_store: 'Predictions',
    source_candidates: List['ModelCandidate'],
    context: 'ExecutionContext'
) -> CrossBranchValidationResult:
    """Convenience function for validating BranchScope.ALL_BRANCHES.

    Args:
        prediction_store: Predictions storage.
        source_candidates: List of candidate source models.
        context: Execution context.

    Returns:
        CrossBranchValidationResult with compatibility info.
    """
    validator = CrossBranchValidator(
        prediction_store=prediction_store,
        log_warnings=True
    )
    return validator.validate_cross_branch_stacking(
        source_candidates=source_candidates,
        context=context
    )
