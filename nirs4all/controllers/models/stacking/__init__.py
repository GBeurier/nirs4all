"""
Stacking subpackage for meta-model training set reconstruction.

This module provides the TrainingSetReconstructor class and supporting utilities
for building meta-model training features from out-of-fold predictions.

Phase 2 Classes:
    TrainingSetReconstructor: Main class for OOF prediction collection and reconstruction.
    FoldAlignmentValidator: Validates fold structure consistency across source models.
    ValidationResult: Container for validation errors and warnings.
    ReconstructionResult: Container for reconstructed training set data.

Phase 3 Classes (Serialization & Prediction Mode):
    SourceModelReference: Reference to a source model with feature mapping.
    MetaModelArtifact: Complete artifact for meta-model persistence.
    MetaModelSerializer: Handles serialization/deserialization of meta-model artifacts.

Phase 4 Classes (Branching Integration):
    BranchValidator: Validates branch contexts for stacking compatibility.
    BranchType: Enum for branch type detection.
    BranchInfo: Information about branch context.
    BranchValidationResult: Result of branch validation.

Phase 5 Classes (Classification Support):
    StackingTaskType: Enum for task types (regression, binary, multiclass).
    ClassificationInfo: Information about classification task detected from predictions.
    TaskTypeDetector: Detects task type from prediction metadata.
    ClassificationFeatureExtractor: Extracts probability features for classification.
    FeatureNameGenerator: Creates descriptive feature names for meta-features.
    MetaFeatureInfo: Information about generated meta-features for importance tracking.

Phase 7 Classes (Advanced Features):
    MultiLevelValidator: Validates multi-level stacking hierarchies.
    ModelLevelInfo: Information about a model's stacking level.
    LevelValidationResult: Result of multi-level validation.
    CrossBranchValidator: Validates cross-branch stacking compatibility.
    CrossBranchValidationResult: Result of cross-branch validation.
    BranchPredictionInfo: Information about predictions from a branch.

Exception Classes:
    MetaModelError: Base exception for meta-model errors.
    MetaModelPredictionError: Base exception for prediction errors.
    MissingSourceModelError: Source model binary not found.
    SourcePredictionError: Source model prediction failed.
    FeatureOrderMismatchError: Feature columns don't match expected order.
    BranchMismatchError: Branch context incompatible.
    MetaModelSerializationError: Base exception for serialization errors.
    MissingDependencyError: Dependency not serialized.
    BranchingError: Base exception for branching errors.
    IncompatibleBranchTypeError: Branch type not compatible with stacking.
    CrossPartitionStackingError: Cross-partition stacking attempted.
    NestedBranchStackingError: Nested branching too deep.
    FoldMismatchAcrossBranchesError: Folds don't align across branches.
    DisjointSampleSetsError: Sample sets don't overlap.
    MultiLevelStackingError: Base exception for multi-level stacking errors.
    CircularDependencyError: Circular dependencies detected in stacking.
    MaxStackingLevelExceededError: Maximum stacking level exceeded.
    CrossBranchStackingError: Base exception for cross-branch stacking errors.
    IncompatibleBranchSamplesError: Branches have incompatible samples.
    BranchFeatureAlignmentError: Feature alignment failed across branches.

Example:
    >>> from nirs4all.controllers.models.stacking import TrainingSetReconstructor
    >>> from nirs4all.operators.models.meta import StackingConfig
    >>>
    >>> reconstructor = TrainingSetReconstructor(
    ...     prediction_store=predictions,
    ...     source_models=["PLS", "RF"],
    ...     stacking_config=StackingConfig()
    ... )
    >>> result = reconstructor.reconstruct(dataset, context)
"""

from .reconstructor import (
    TrainingSetReconstructor,
    FoldAlignmentValidator,
    ValidationResult,
    ReconstructionResult,
)
from .config import (
    ReconstructorConfig,
)
from .serialization import (
    SourceModelReference,
    MetaModelArtifact,
    MetaModelSerializer,
    stacking_config_to_dict,
    stacking_config_from_dict,
)
from .branch_validator import (
    BranchValidator,
    BranchType,
    BranchInfo,
    BranchValidationResult,
    StackingCompatibility,
    detect_branch_type,
    is_stacking_compatible,
    is_disjoint_branch,
    get_disjoint_branch_info,
)
from .classification import (
    StackingTaskType,
    ClassificationInfo,
    TaskTypeDetector,
    ClassificationFeatureExtractor,
    FeatureNameGenerator,
    MetaFeatureInfo,
    build_meta_feature_info,
)
# Phase 7: Multi-Level Stacking
from .multilevel import (
    MultiLevelValidator,
    ModelLevelInfo,
    LevelValidationResult,
    validate_multi_level_stacking,
    detect_stacking_level,
)
# Phase 7: Cross-Branch Stacking
from .crossbranch import (
    CrossBranchValidator,
    CrossBranchValidationResult,
    CrossBranchCompatibility,
    BranchPredictionInfo,
    validate_all_branches_scope,
)
from .exceptions import (
    MetaModelError,
    MetaModelPredictionError,
    MissingSourceModelError,
    SourcePredictionError,
    FeatureOrderMismatchError,
    BranchMismatchError,
    NoSourcePredictionsError,
    MetaModelSerializationError,
    MissingDependencyError,
    InvalidMetaModelArtifactError,
    # Phase 4 - Branching Exceptions
    BranchingError,
    IncompatibleBranchTypeError,
    CrossPartitionStackingError,
    NestedBranchStackingError,
    FoldMismatchAcrossBranchesError,
    DisjointSampleSetsError,
    GeneratorSyntaxStackingWarning,
    # Phase 7 - Multi-Level Stacking Exceptions
    MultiLevelStackingError,
    CircularDependencyError,
    MaxStackingLevelExceededError,
    InconsistentLevelError,
    # Phase 7 - Cross-Branch Stacking Exceptions
    CrossBranchStackingError,
    IncompatibleBranchSamplesError,
    BranchFeatureAlignmentError,
)

__all__ = [
    # Phase 2 - Reconstruction
    'TrainingSetReconstructor',
    'FoldAlignmentValidator',
    'ValidationResult',
    'ReconstructionResult',
    'ReconstructorConfig',
    # Phase 3 - Serialization
    'SourceModelReference',
    'MetaModelArtifact',
    'MetaModelSerializer',
    'stacking_config_to_dict',
    'stacking_config_from_dict',
    # Phase 4 - Branch Validation
    'BranchValidator',
    'BranchType',
    'BranchInfo',
    'BranchValidationResult',
    'StackingCompatibility',
    'detect_branch_type',
    'is_stacking_compatible',
    'is_disjoint_branch',
    'get_disjoint_branch_info',
    # Phase 5 - Classification Support
    'StackingTaskType',
    'ClassificationInfo',
    'TaskTypeDetector',
    'ClassificationFeatureExtractor',
    'FeatureNameGenerator',
    'MetaFeatureInfo',
    'build_meta_feature_info',
    # Phase 7 - Multi-Level Stacking
    'MultiLevelValidator',
    'ModelLevelInfo',
    'LevelValidationResult',
    'validate_multi_level_stacking',
    'detect_stacking_level',
    # Phase 7 - Cross-Branch Stacking
    'CrossBranchValidator',
    'CrossBranchValidationResult',
    'CrossBranchCompatibility',
    'BranchPredictionInfo',
    'validate_all_branches_scope',
    # Phase 3 - Exceptions
    'MetaModelError',
    'MetaModelPredictionError',
    'MissingSourceModelError',
    'SourcePredictionError',
    'FeatureOrderMismatchError',
    'BranchMismatchError',
    'NoSourcePredictionsError',
    'MetaModelSerializationError',
    'MissingDependencyError',
    'InvalidMetaModelArtifactError',
    # Phase 4 - Branching Exceptions
    'BranchingError',
    'IncompatibleBranchTypeError',
    'CrossPartitionStackingError',
    'NestedBranchStackingError',
    'FoldMismatchAcrossBranchesError',
    'DisjointSampleSetsError',
    'GeneratorSyntaxStackingWarning',
    # Phase 7 - Multi-Level Stacking Exceptions
    'MultiLevelStackingError',
    'CircularDependencyError',
    'MaxStackingLevelExceededError',
    'InconsistentLevelError',
    # Phase 7 - Cross-Branch Stacking Exceptions
    'CrossBranchStackingError',
    'IncompatibleBranchSamplesError',
    'BranchFeatureAlignmentError',
]
