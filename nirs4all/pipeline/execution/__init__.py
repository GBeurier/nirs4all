"""Pipeline execution module for nirs4all."""
# Importing a result/parser must not initialize executors that import controllers
# again. Resolve the public conveniences when used, after package initialization.
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .builder import ExecutorBuilder
    from .executor import PipelineExecutor
    from .orchestrator import PipelineOrchestrator
    from .result import ArtifactMeta, StepResult

_LAZY_EXPORTS = {
    'ExecutorBuilder': ('.builder', 'ExecutorBuilder'),
    'PipelineExecutor': ('.executor', 'PipelineExecutor'),
    'PipelineOrchestrator': ('.orchestrator', 'PipelineOrchestrator'),
    'ArtifactMeta': ('.result', 'ArtifactMeta'),
    'StepResult': ('.result', 'StepResult'),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = [
    'StepResult',
    'ArtifactMeta',
    'PipelineExecutor',
    'PipelineOrchestrator',
    'ExecutorBuilder',
]

