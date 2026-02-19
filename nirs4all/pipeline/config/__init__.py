"""Configuration and domain objects for pipeline execution.

This module contains core domain objects that define pipeline structure,
execution context, and configuration expansion logic.
"""

from .component_serialization import serialize_component
from .context import (
    ArtifactProvider,
    DataSelector,
    ExecutionContext,
    ExecutionPhase,
    LoaderArtifactProvider,
    MapArtifactProvider,
    PipelineState,
    RuntimeContext,
    StepMetadata,
)
from .generator import count_combinations, expand_spec
from .pipeline_config import PipelineConfigs

__all__ = [
    'PipelineConfigs',
    'ExecutionContext',
    'ExecutionPhase',
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
