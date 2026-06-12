"""
SpectroDataset - A specialized dataset API for spectroscopy data.

This module provides zero-copy, multi-source aware dataset management
with transparent versioning and fine-grained indexing capabilities.
"""

# Provide backward-compatible imports for feature components
from ._features import (
    FeatureLayout,
    FeatureSource,
    HeaderUnit,
    normalize_header_unit,
    normalize_layout,
)
from .config import DatasetConfigs
from .dataset import SpectroDataset
from .fit_influence import (
    FitInfluenceError,
    FitInfluenceMode,
    FitInfluencePolicy,
    FitInfluenceResolution,
    resolve_fit_influence,
)

# Parser utilities
from .parsers import (
    ConfigNormalizer,
    normalize_config,
)
from .predictions import MergeReport, PredictionResult, PredictionResultsList, Predictions
from .relation_replay_manifest import (
    RelationReplayManifest,
    RelationReplayManifestError,
    build_relation_replay_manifest,
)

# Schema types (Phase 1 - new in refactoring)
from .schema import (
    ColumnConfig,
    ConfigValidator,
    DatasetConfigSchema,
    FileConfig,
    LoadingParams,
    PartitionConfig,
    TaskType,
    ValidationError,
    ValidationResult,
    ValidationWarning,
)

# Signal type management
from .signal_type import (
    SignalType,
    SignalTypeDetector,
    SignalTypeInput,
    detect_signal_type,
    normalize_signal_type,
)

__all__ = [
    "SpectroDataset",
    "DatasetConfigs",
    "MergeReport",
    "Predictions",
    "PredictionResult",
    "PredictionResultsList",
    "PredictionAnalyzer",
    "FeatureSource",
    "FeatureLayout",
    "HeaderUnit",
    "normalize_layout",
    "normalize_header_unit",
    # Fit influence
    "FitInfluenceError",
    "FitInfluenceMode",
    "FitInfluencePolicy",
    "FitInfluenceResolution",
    "resolve_fit_influence",
    # Relational replay
    "RelationReplayManifest",
    "RelationReplayManifestError",
    "build_relation_replay_manifest",
    # Signal type
    "SignalType",
    "SignalTypeInput",
    "normalize_signal_type",
    "SignalTypeDetector",
    "detect_signal_type",
    # Schema (new)
    "DatasetConfigSchema",
    "FileConfig",
    "ColumnConfig",
    "PartitionConfig",
    "LoadingParams",
    "TaskType",
    "ConfigValidator",
    "ValidationResult",
    "ValidationError",
    "ValidationWarning",
    # Parsers (new)
    "ConfigNormalizer",
    "normalize_config",
]


def __getattr__(name: str):
    """Lazily expose PredictionAnalyzer (PEP 562).

    PredictionAnalyzer lives in the visualization layer, which imports matplotlib
    at module load. Importing it lazily keeps ``import nirs4all`` from pulling
    matplotlib in through the data layer; it loads on first attribute access.
    """
    if name == "PredictionAnalyzer":
        from ..visualization.predictions import PredictionAnalyzer
        return PredictionAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
