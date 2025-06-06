"""
Pipeline module for nirs4all package.

This module contains pipeline classes for processing workflows.
"""

# Import main pipeline classes
from .fitted_pipeline import FittedPipeline
from .pipeline import Pipeline
from .pipeline_config import PipelineConfig
from .pipeline_context import PipelineContext, ScopeState
from .pipeline_history import PipelineHistory, PipelineExecution, StepExecution
from .pipeline_operation import PipelineOperation
# from ..operations.operator_controller import PipelineOperatorWrapper  # Not found
from .pipeline_runner import PipelineRunner
from .pipeline_tree import PipelineTree

# Import the presets dictionary from operation_presets.py
from ..operations import operation_presets

__all__ = [
    'FittedPipeline',
    'Pipeline',
    'PipelineConfig',
    'PipelineContext',
    'PipelineHistory',
    'PipelineOperation',
    # 'PipelineOperatorWrapper',  # Not found
    'PipelineRunner',
    'PipelineTree',
    'ScopeState',
    'PipelineExecution',
    'StepExecution',
    'operation_presets',
]
