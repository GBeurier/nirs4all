"""Feature management components and utilities."""

from .feature_constants import (
    FeatureLayout,
    HeaderUnit,
    LayoutType,
    HeaderUnitType,
    DEFAULT_PROCESSING,
    normalize_layout,
    normalize_header_unit,
)
from .array_storage import ArrayStorage
from .processing_manager import ProcessingManager
from .header_manager import HeaderManager
from .layout_transformer import LayoutTransformer
from .update_strategy import UpdateStrategy, ReplacementOperation, AdditionOperation
from .augmentation_handler import AugmentationHandler
from .feature_source import FeatureSource

__all__ = [
    # Constants and enums
    "FeatureLayout",
    "HeaderUnit",
    "LayoutType",
    "HeaderUnitType",
    "DEFAULT_PROCESSING",
    "normalize_layout",
    "normalize_header_unit",
    # Main class
    "FeatureSource",
    # Internal components
    "ArrayStorage",
    "ProcessingManager",
    "HeaderManager",
    "LayoutTransformer",
    "UpdateStrategy",
    "ReplacementOperation",
    "AdditionOperation",
    "AugmentationHandler",
]
