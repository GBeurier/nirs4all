"""
Pipeline module for nirs4all package.

This module contains pipeline classes for processing workflows.
"""

# Import main pipeline classes
from .config import PipelineConfigs
# from .pipeline_context import PipelineContext, ScopeState
from .history import PipelineHistory, PipelineExecution, StepExecution
from .operation import PipelineOperation
# from ..operations.operator_controller import PipelineOperatorWrapper  # Not found
from .runner import PipelineRunner
# from .pipeline_tree import PipelineTree

# Import the presets dictionary from operation_presets.py
# from ..operations import operation_presets

__all__ = [
    # 'FittedPipeline','Pipeline',
    'PipelineConfigs',
    # 'PipelineContext',
    'PipelineHistory',
    'PipelineOperation',
    # 'PipelineOperatorWrapper',  # Not found
    'PipelineRunner',
    # 'PipelineTree',
    # 'ScopeState',
    'PipelineExecution',
    'StepExecution',
    # 'operation_presets',
]
