"""Storage and persistence layer for pipeline artifacts and results.

This module handles all I/O operations including:
- Pipeline artifact storage (binary models, scalers, etc.)
- Manifest and metadata management
- Workspace exports and imports
- Prediction resolution and saving
- Pipeline template library
"""

from .io import SimulationSaver
from .io_writer import PipelineWriter
from .io_exporter import WorkspaceExporter
from .io_resolver import TargetResolver, PredictionResolver  # PredictionResolver is alias for TargetResolver
from .manifest_manager import ManifestManager
from .library import PipelineLibrary

__all__ = [
    'SimulationSaver',
    'PipelineWriter',
    'WorkspaceExporter',
    'TargetResolver',
    'PredictionResolver',  # Deprecated alias for TargetResolver
    'ManifestManager',
    'PipelineLibrary',
]
