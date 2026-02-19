"""Data manipulation controllers.

Controllers for data manipulation operators (branch, merge, resampler, augmentation,
feature selection, exclusion, tagging, repetition transformation).
"""

from .auto_transfer_preproc import AutoTransferPreprocessingController
from .branch import BranchController
from .branch_utils import (
    group_samples_by_value_mapping,
    parse_value_condition,
    validate_disjoint_conditions,
)
from .concat_transform import ConcatAugmentationController
from .exclude import ExcludeController
from .feature_augmentation import FeatureAugmentationController
from .feature_selection import FeatureSelectionController
from .merge import MergeConfigParser, MergeController
from .repetition import RepToPPController, RepToSourcesController
from .resampler import ResamplerController
from .sample_augmentation import SampleAugmentationController
from .tag import TagController

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
