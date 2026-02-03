"""Data manipulation controllers.

Controllers for data manipulation operators (branch, merge, resampler, augmentation,
feature selection, exclusion, tagging, repetition transformation).
"""

from .feature_augmentation import FeatureAugmentationController
from .sample_augmentation import SampleAugmentationController
from .resampler import ResamplerController
from .feature_selection import FeatureSelectionController
from .concat_transform import ConcatAugmentationController
from .auto_transfer_preproc import AutoTransferPreprocessingController
from .exclude import ExcludeController
from .tag import TagController
from .merge import MergeController, MergeConfigParser
from .repetition import RepToSourcesController, RepToPPController
from .branch import BranchController
from .branch_utils import (
    parse_value_condition,
    group_samples_by_value_mapping,
    validate_disjoint_conditions,
)

__all__ = [
    # Controllers
    "FeatureAugmentationController",
    "SampleAugmentationController",
    "ResamplerController",
    "FeatureSelectionController",
    "ConcatAugmentationController",
    "AutoTransferPreprocessingController",
    "ExcludeController",
    "TagController",
    "MergeController",
    "MergeConfigParser",
    "RepToSourcesController",
    "RepToPPController",
    "BranchController",
    # Branch utilities
    "parse_value_condition",
    "group_samples_by_value_mapping",
    "validate_disjoint_conditions",
]
