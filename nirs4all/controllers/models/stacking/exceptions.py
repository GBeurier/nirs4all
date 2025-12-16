"""
Meta-Model Exceptions - Error classes for meta-model stacking.

This module provides specific exception classes for meta-model prediction
and serialization errors.

Exception Hierarchy:
    MetaModelError (base)
    ├── MetaModelPredictionError (prediction failures)
    │   ├── MissingSourceModelError (source model not found)
    │   ├── SourcePredictionError (source model prediction failed)
    │   ├── FeatureOrderMismatchError (column order doesn't match)
    │   └── BranchMismatchError (branch context incompatible)
    └── MetaModelSerializationError (persistence failures)
        └── MissingDependencyError (dependency not serialized)
"""

from typing import Any, Dict, List, Optional


class MetaModelError(Exception):
    """Base exception for meta-model errors.

    All meta-model specific exceptions inherit from this class,
    allowing catch-all handling of meta-model issues.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary with additional context.
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


# =============================================================================
# Prediction Errors
# =============================================================================


class MetaModelPredictionError(MetaModelError):
    """Base exception for meta-model prediction errors.

    Raised when prediction mode fails for a meta-model.
    """
    pass


class MissingSourceModelError(MetaModelPredictionError):
    """Raised when a source model binary is not found.

    This occurs when trying to load a meta-model for prediction but one
    or more of its source model dependencies cannot be found in the
    artifact store.

    Attributes:
        source_model_id: The artifact ID of the missing source model.
        meta_model_id: The artifact ID of the meta-model.
        all_missing: List of all missing source model IDs (if multiple).
    """

    def __init__(
        self,
        source_model_id: str,
        meta_model_id: str,
        all_missing: Optional[List[str]] = None
    ):
        self.source_model_id = source_model_id
        self.meta_model_id = meta_model_id
        self.all_missing = all_missing or [source_model_id]

        missing_str = ", ".join(self.all_missing)
        message = (
            f"Source model(s) not found for meta-model {meta_model_id}: "
            f"{missing_str}. "
            f"Ensure all source models are persisted before the meta-model."
        )
        super().__init__(
            message,
            {
                "source_model_id": source_model_id,
                "meta_model_id": meta_model_id,
                "missing_count": len(self.all_missing)
            }
        )


class SourcePredictionError(MetaModelPredictionError):
    """Raised when a source model fails to produce predictions.

    This occurs when a source model is loaded successfully but fails
    during prediction, preventing the meta-model from constructing
    its input features.

    Attributes:
        source_model_id: The artifact ID of the failing source model.
        source_model_name: The name of the failing source model.
        original_error: The underlying exception that caused the failure.
    """

    def __init__(
        self,
        source_model_id: str,
        source_model_name: str,
        original_error: Optional[Exception] = None
    ):
        self.source_model_id = source_model_id
        self.source_model_name = source_model_name
        self.original_error = original_error

        message = (
            f"Source model '{source_model_name}' ({source_model_id}) "
            f"failed to produce predictions"
        )
        if original_error:
            message += f": {str(original_error)}"

        super().__init__(
            message,
            {
                "source_model_id": source_model_id,
                "source_model_name": source_model_name,
                "original_error_type": type(original_error).__name__ if original_error else None
            }
        )


class FeatureOrderMismatchError(MetaModelPredictionError):
    """Raised when feature columns don't match expected order.

    This occurs when the predictions from source models don't align
    with the feature column order that was used during meta-model training.

    Attributes:
        expected_columns: Expected feature column names in order.
        actual_columns: Actual feature column names found.
        meta_model_id: The artifact ID of the meta-model.
    """

    def __init__(
        self,
        expected_columns: List[str],
        actual_columns: List[str],
        meta_model_id: str
    ):
        self.expected_columns = expected_columns
        self.actual_columns = actual_columns
        self.meta_model_id = meta_model_id

        message = (
            f"Feature column mismatch for meta-model {meta_model_id}. "
            f"Expected {len(expected_columns)} columns: {expected_columns[:3]}..., "
            f"got {len(actual_columns)}: {actual_columns[:3]}..."
        )
        super().__init__(
            message,
            {
                "expected_count": len(expected_columns),
                "actual_count": len(actual_columns),
                "meta_model_id": meta_model_id
            }
        )


class BranchMismatchError(MetaModelPredictionError):
    """Raised when prediction branch doesn't match training branch.

    This occurs when attempting to use a meta-model trained in one branch
    for prediction in a different, incompatible branch context.

    Attributes:
        training_branch: Branch context during training.
        prediction_branch: Branch context during prediction.
        meta_model_id: The artifact ID of the meta-model.
    """

    def __init__(
        self,
        training_branch: Dict[str, Any],
        prediction_branch: Dict[str, Any],
        meta_model_id: str
    ):
        self.training_branch = training_branch
        self.prediction_branch = prediction_branch
        self.meta_model_id = meta_model_id

        train_str = f"id={training_branch.get('branch_id')}, name={training_branch.get('branch_name')}"
        pred_str = f"id={prediction_branch.get('branch_id')}, name={prediction_branch.get('branch_name')}"

        message = (
            f"Branch context mismatch for meta-model {meta_model_id}. "
            f"Trained in branch ({train_str}), "
            f"but attempting prediction in branch ({pred_str}). "
            f"Meta-models must be used in the same branch context as training."
        )
        super().__init__(
            message,
            {
                "training_branch_id": training_branch.get('branch_id'),
                "prediction_branch_id": prediction_branch.get('branch_id'),
                "meta_model_id": meta_model_id
            }
        )


class NoSourcePredictionsError(MetaModelPredictionError):
    """Raised when no source model predictions are available.

    This occurs when the prediction store doesn't contain any predictions
    from source models, making it impossible to construct meta-features.

    Attributes:
        expected_sources: List of expected source model names.
        meta_model_id: The artifact ID of the meta-model.
    """

    def __init__(
        self,
        expected_sources: List[str],
        meta_model_id: str
    ):
        self.expected_sources = expected_sources
        self.meta_model_id = meta_model_id

        source_str = ", ".join(expected_sources[:5])
        if len(expected_sources) > 5:
            source_str += f"... ({len(expected_sources)} total)"

        message = (
            f"No predictions found from source models for meta-model {meta_model_id}. "
            f"Expected predictions from: {source_str}. "
            f"Ensure source model steps run before the meta-model step."
        )
        super().__init__(
            message,
            {
                "expected_sources": expected_sources,
                "meta_model_id": meta_model_id
            }
        )


# =============================================================================
# Serialization Errors
# =============================================================================


class MetaModelSerializationError(MetaModelError):
    """Base exception for meta-model serialization errors.

    Raised when persistence or loading of meta-model artifacts fails.
    """
    pass


class MissingDependencyError(MetaModelSerializationError):
    """Raised when a dependency is not serialized.

    This occurs when attempting to serialize a meta-model but one of
    its source model dependencies hasn't been persisted yet.

    Attributes:
        dependency_id: The artifact ID of the missing dependency.
        meta_model_id: The artifact ID of the meta-model.
        dependency_name: Optional name of the missing dependency.
    """

    def __init__(
        self,
        dependency_id: str,
        meta_model_id: str,
        dependency_name: Optional[str] = None
    ):
        self.dependency_id = dependency_id
        self.meta_model_id = meta_model_id
        self.dependency_name = dependency_name

        name_str = f" ({dependency_name})" if dependency_name else ""
        message = (
            f"Cannot serialize meta-model {meta_model_id}: "
            f"dependency {dependency_id}{name_str} not found in artifact registry. "
            f"Ensure source models are registered before the meta-model."
        )
        super().__init__(
            message,
            {
                "dependency_id": dependency_id,
                "meta_model_id": meta_model_id,
                "dependency_name": dependency_name
            }
        )


class InvalidMetaModelArtifactError(MetaModelSerializationError):
    """Raised when a meta-model artifact is invalid or corrupted.

    This occurs when loading a meta-model artifact that doesn't contain
    the required fields or has inconsistent data.

    Attributes:
        artifact_id: The artifact ID being loaded.
        validation_errors: List of validation errors found.
    """

    def __init__(
        self,
        artifact_id: str,
        validation_errors: List[str]
    ):
        self.artifact_id = artifact_id
        self.validation_errors = validation_errors

        errors_str = "; ".join(validation_errors[:3])
        if len(validation_errors) > 3:
            errors_str += f"... ({len(validation_errors)} total errors)"

        message = (
            f"Invalid meta-model artifact {artifact_id}: {errors_str}"
        )
        super().__init__(
            message,
            {
                "artifact_id": artifact_id,
                "error_count": len(validation_errors)
            }
        )


# =============================================================================
# Branching Errors (Phase 4)
# =============================================================================


class BranchingError(MetaModelError):
    """Base exception for branching-related errors in meta-model stacking."""
    pass


class IncompatibleBranchTypeError(BranchingError):
    """Raised when stacking is attempted with incompatible branch type.

    This occurs when the branch context is not suitable for stacking,
    such as attempting cross-partition stacking with sample_partitioner.

    Attributes:
        branch_type: Type of branching detected.
        reason: Explanation of why stacking is incompatible.
        suggestions: List of possible solutions.
    """

    def __init__(
        self,
        branch_type: str,
        reason: str,
        suggestions: Optional[List[str]] = None
    ):
        self.branch_type = branch_type
        self.reason = reason
        self.suggestions = suggestions or []

        message = (
            f"Stacking incompatible with branch type '{branch_type}': {reason}"
        )
        if self.suggestions:
            message += "\nSuggestions: " + "; ".join(self.suggestions)

        super().__init__(
            message,
            {
                "branch_type": branch_type,
                "reason": reason
            }
        )


class CrossPartitionStackingError(BranchingError):
    """Raised when attempting cross-partition stacking with sample_partitioner.

    Sample partitioner creates disjoint sample sets, making cross-partition
    stacking impossible without data leakage or sample misalignment.

    Attributes:
        partition_a: Name of first partition.
        partition_b: Name of second partition.
        n_samples_a: Number of samples in partition A.
        n_samples_b: Number of samples in partition B.
    """

    def __init__(
        self,
        partition_a: str,
        partition_b: str,
        n_samples_a: int,
        n_samples_b: int
    ):
        self.partition_a = partition_a
        self.partition_b = partition_b
        self.n_samples_a = n_samples_a
        self.n_samples_b = n_samples_b

        message = (
            f"Cannot stack across sample partitions '{partition_a}' ({n_samples_a} samples) "
            f"and '{partition_b}' ({n_samples_b} samples). "
            f"Sample partitioner creates disjoint sample sets. "
            f"Stacking must be performed within each partition separately."
        )
        super().__init__(
            message,
            {
                "partition_a": partition_a,
                "partition_b": partition_b,
                "n_samples_a": n_samples_a,
                "n_samples_b": n_samples_b
            }
        )


class NestedBranchStackingError(BranchingError):
    """Raised when stacking is attempted with complex nested branching.

    Complex nested branching scenarios may not be fully supported
    in meta-model stacking due to sample alignment challenges.

    Attributes:
        branch_depth: Depth of nesting.
        branch_path: Full path of branch IDs.
        reason: Explanation of the issue.
    """

    def __init__(
        self,
        branch_depth: int,
        branch_path: List[int],
        reason: str
    ):
        self.branch_depth = branch_depth
        self.branch_path = branch_path
        self.reason = reason

        path_str = " → ".join(str(b) for b in branch_path)
        message = (
            f"Stacking with nested branching (depth={branch_depth}, path=[{path_str}]) "
            f"may have limited support: {reason}. "
            f"Consider simplifying the pipeline structure or flattening branches "
            f"before stacking."
        )
        super().__init__(
            message,
            {
                "branch_depth": branch_depth,
                "branch_path": branch_path,
                "reason": reason
            }
        )


class FoldMismatchAcrossBranchesError(BranchingError):
    """Raised when source models from different branches have mismatched folds.

    For stacking to work correctly, all source models must have compatible
    fold structures (same number of folds, same sample assignments per fold).

    Attributes:
        fold_structures: Dict mapping branch_id to fold count.
        affected_models: List of affected model names.
    """

    def __init__(
        self,
        fold_structures: Dict[int, int],
        affected_models: List[str]
    ):
        self.fold_structures = fold_structures
        self.affected_models = affected_models

        structure_str = ", ".join(
            f"branch {bid}: {n_folds} folds"
            for bid, n_folds in fold_structures.items()
        )
        models_str = ", ".join(affected_models[:5])
        if len(affected_models) > 5:
            models_str += f"... ({len(affected_models)} total)"

        message = (
            f"Fold structure mismatch across branches: {structure_str}. "
            f"Affected models: {models_str}. "
            f"All source models must use the same cross-validation configuration."
        )
        super().__init__(
            message,
            {
                "fold_structures": fold_structures,
                "affected_models_count": len(affected_models)
            }
        )


class DisjointSampleSetsError(BranchingError):
    """Raised when source models have disjoint sample sets.

    This typically occurs with sample_partitioner where different branches
    have different samples, making OOF reconstruction impossible.

    Attributes:
        source_model: Name of the source model with disjoint samples.
        expected_samples: Number of expected samples.
        found_samples: Number of samples found in predictions.
        overlap_ratio: Ratio of overlapping samples.
    """

    def __init__(
        self,
        source_model: str,
        expected_samples: int,
        found_samples: int,
        overlap_ratio: float
    ):
        self.source_model = source_model
        self.expected_samples = expected_samples
        self.found_samples = found_samples
        self.overlap_ratio = overlap_ratio

        message = (
            f"Source model '{source_model}' has disjoint sample set: "
            f"expected {expected_samples} samples, found {found_samples} "
            f"({100 * overlap_ratio:.1f}% overlap). "
            f"This typically occurs with sample_partitioner branches. "
            f"Stack only with models from the same partition."
        )
        super().__init__(
            message,
            {
                "source_model": source_model,
                "expected_samples": expected_samples,
                "found_samples": found_samples,
                "overlap_ratio": overlap_ratio
            }
        )


class GeneratorSyntaxStackingWarning(BranchingError):
    """Raised as warning for generator syntax with stacking.

    Generator syntax creates multiple model variants which may complicate
    stacking behavior. This is a warning, not a hard error.

    Attributes:
        generator_type: Type of generator used.
        n_variants: Number of variants generated.
    """

    def __init__(
        self,
        generator_type: str,
        n_variants: int
    ):
        self.generator_type = generator_type
        self.n_variants = n_variants

        message = (
            f"Generator syntax '{generator_type}' created {n_variants} model variants. "
            f"The meta-model will stack on all variants which may increase "
            f"dimensionality significantly. Consider using explicit source_models "
            f"parameter or a TopKByMetricSelector to limit sources."
        )
        super().__init__(
            message,
            {
                "generator_type": generator_type,
                "n_variants": n_variants
            }
        )


# =============================================================================
# Multi-Level Stacking Errors (Phase 7)
# =============================================================================


class MultiLevelStackingError(MetaModelError):
    """Base exception for multi-level stacking errors."""
    pass


class CircularDependencyError(MultiLevelStackingError):
    """Raised when circular dependencies are detected in multi-level stacking.

    This occurs when a meta-model attempts to use itself or a model that
    depends on it as a source, creating a circular dependency.

    Attributes:
        source_model: Name of the source model causing the cycle.
        meta_model: Name of the current meta-model.
        dependency_chain: List of model names forming the cycle.
    """

    def __init__(
        self,
        source_model: str,
        meta_model: str,
        dependency_chain: List[str]
    ):
        self.source_model = source_model
        self.meta_model = meta_model
        self.dependency_chain = dependency_chain

        chain_str = " -> ".join(dependency_chain)
        message = (
            f"Circular dependency detected in multi-level stacking: {chain_str}. "
            f"Meta-model '{meta_model}' cannot use '{source_model}' as a source "
            f"because it creates a dependency cycle."
        )
        super().__init__(
            message,
            {
                "source_model": source_model,
                "meta_model": meta_model,
                "chain_length": len(dependency_chain)
            }
        )


class MaxStackingLevelExceededError(MultiLevelStackingError):
    """Raised when the maximum stacking level is exceeded.

    This occurs when attempting to create more stacking levels than allowed,
    which could lead to overfitting or excessive computational cost.

    Attributes:
        current_level: The level that was attempted.
        max_level: The maximum allowed level.
        source_models: Source models that caused the level increase.
    """

    def __init__(
        self,
        current_level: int,
        max_level: int,
        source_models: Optional[List[str]] = None
    ):
        self.current_level = current_level
        self.max_level = max_level
        self.source_models = source_models or []

        sources_str = ", ".join(self.source_models[:5]) if self.source_models else "unknown"
        if len(self.source_models) > 5:
            sources_str += f"... ({len(self.source_models)} total)"

        message = (
            f"Maximum stacking level exceeded: attempted level {current_level} "
            f"but maximum is {max_level}. "
            f"Source models: {sources_str}. "
            f"Increase max_level in StackingConfig or simplify the stacking hierarchy."
        )
        super().__init__(
            message,
            {
                "current_level": current_level,
                "max_level": max_level,
                "source_models_count": len(self.source_models)
            }
        )


class InconsistentLevelError(MultiLevelStackingError):
    """Raised when source models have inconsistent stacking levels.

    This occurs when a meta-model tries to use source models from
    incompatible levels, which could indicate a pipeline configuration issue.

    Attributes:
        expected_levels: Expected level(s) for source models.
        found_levels: Actual levels found in source models.
        problematic_models: Models with unexpected levels.
    """

    def __init__(
        self,
        expected_levels: List[int],
        found_levels: Dict[str, int],
        problematic_models: List[str]
    ):
        self.expected_levels = expected_levels
        self.found_levels = found_levels
        self.problematic_models = problematic_models

        expected_str = ", ".join(str(lv) for lv in expected_levels)
        models_str = ", ".join(f"{m}(L{lv})" for m, lv in list(found_levels.items())[:5])

        message = (
            f"Inconsistent stacking levels in source models. "
            f"Expected levels {expected_str}, found: {models_str}. "
            f"Ensure source models are from appropriate stacking levels."
        )
        super().__init__(
            message,
            {
                "expected_levels": expected_levels,
                "problematic_count": len(problematic_models)
            }
        )


# =============================================================================
# Cross-Branch Stacking Errors (Phase 7)
# =============================================================================


class CrossBranchStackingError(BranchingError):
    """Base exception for cross-branch stacking errors."""
    pass


class IncompatibleBranchSamplesError(CrossBranchStackingError):
    """Raised when branches have incompatible sample sets for cross-branch stacking.

    This occurs when attempting ALL_BRANCHES stacking with branches that have
    different sample indices (e.g., sample_partitioner branches).

    Attributes:
        branches: Dict mapping branch_id to sample counts.
        overlap_matrix: Optional matrix showing sample overlap between branches.
    """

    def __init__(
        self,
        branches: Dict[int, int],
        overlap_matrix: Optional[Dict[tuple, float]] = None
    ):
        self.branches = branches
        self.overlap_matrix = overlap_matrix

        branches_str = ", ".join(f"branch_{bid}: {n} samples" for bid, n in branches.items())

        message = (
            f"Cannot perform cross-branch stacking: branches have incompatible "
            f"sample sets. {branches_str}. "
            f"Cross-branch stacking requires all branches to have the same samples. "
            f"Use BranchScope.CURRENT_ONLY for sample_partitioner branches."
        )
        super().__init__(
            message,
            {
                "branch_count": len(branches),
                "sample_counts": list(branches.values())
            }
        )


class BranchFeatureAlignmentError(CrossBranchStackingError):
    """Raised when cross-branch features cannot be aligned.

    This occurs when attempting to combine predictions from multiple branches
    but the feature matrices cannot be properly aligned due to different
    preprocessing or ordering.

    Attributes:
        expected_features: Expected number of features.
        branch_features: Dict mapping branch_id to feature count.
        alignment_issues: List of specific alignment problems.
    """

    def __init__(
        self,
        expected_features: int,
        branch_features: Dict[int, int],
        alignment_issues: Optional[List[str]] = None
    ):
        self.expected_features = expected_features
        self.branch_features = branch_features
        self.alignment_issues = alignment_issues or []

        features_str = ", ".join(
            f"branch_{bid}: {n}" for bid, n in branch_features.items()
        )
        issues_str = "; ".join(self.alignment_issues[:3]) if self.alignment_issues else ""

        message = (
            f"Cannot align features across branches for cross-branch stacking. "
            f"Expected {expected_features} features, got: {features_str}. "
            f"{issues_str}"
        )
        super().__init__(
            message,
            {
                "expected_features": expected_features,
                "branch_count": len(branch_features),
                "issues_count": len(self.alignment_issues)
            }
        )

