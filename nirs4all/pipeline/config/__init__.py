"""Configuration and domain objects for pipeline execution.

This module contains core domain objects that define pipeline structure,
execution context, and configuration expansion logic.
"""

from .pipeline_config import PipelineConfigs
from .context import (
    ExecutionContext,
    DataSelector,
    PipelineState,
    StepMetadata,
    RuntimeContext,
    ArtifactProvider,
    MapArtifactProvider,
    LoaderArtifactProvider,
)
from .component_serialization import serialize_component
from .generator import expand_spec, count_combinations

__all__ = [
    'PipelineConfigs',
    'ExecutionContext',
    'DataSelector',
    'PipelineState',
    'StepMetadata',
    'RuntimeContext',
    'ArtifactProvider',
    'MapArtifactProvider',
    'LoaderArtifactProvider',
    'serialize_component',
    'expand_spec',
    'count_combinations',
]
