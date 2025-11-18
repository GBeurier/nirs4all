"""Configuration and domain objects for pipeline execution.

This module contains core domain objects that define pipeline structure,
execution context, and configuration expansion logic.
"""

from .config import PipelineConfigs
from .context import (
    ExecutionContext,
    DataSelector,
    PipelineState,
    StepMetadata
)
from .component_serialization import serialize_component
from .generator import expand_spec, count_combinations

__all__ = [
    'PipelineConfigs',
    'ExecutionContext',
    'DataSelector',
    'PipelineState',
    'StepMetadata',
    'serialize_component',
    'expand_spec',
    'count_combinations',
]
