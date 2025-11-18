"""Pipeline execution module for nirs4all."""
from .result import StepResult, ExecutionResult, OrchestrationResult, ArtifactMeta
from .executor import PipelineExecutor
from .orchestrator import PipelineOrchestrator

__all__ = [
    'StepResult',
    'ExecutionResult',
    'OrchestrationResult',
    'ArtifactMeta',
    'PipelineExecutor',
    'PipelineOrchestrator',
]
