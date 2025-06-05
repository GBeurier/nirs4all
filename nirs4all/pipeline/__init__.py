"""
Pipeline module for nirs4all package.

This module contains pipeline classes for processing workflows.
"""

# Import main pipeline classes
from .FittedPipeline import FittedPipeline
from .Pipeline import Pipeline
from .PipelineConfig import PipelineConfig
from .PipelineContext import PipelineContext, ScopeState
from .PipelineHistory import PipelineHistory, PipelineExecution, StepExecution
from .PipelineOperation import PipelineOperation
from ..operations.OperatorController import PipelineOperatorWrapper
from .PipelineRunner import PipelineRunner
from .PipelineTree import PipelineTree

# Import the presets dictionary from OperationPresets.py
from ..operations import OperationPresets

__all__ = [
    'FittedPipeline',
    'Pipeline',
    'PipelineConfig',
    'PipelineContext',
    'PipelineHistory',
    'PipelineOperation',
    'PipelineOperatorWrapper',
    'PipelineRunner',
    'PipelineTree',
    'ScopeState',
    'PipelineExecution',
    'StepExecution',
    'OperationPresets',
]
