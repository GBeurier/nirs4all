"""Pipeline step processing module for nirs4all."""
# Importing a result/parser must not initialize executors that import controllers
# again. Resolve the public conveniences when used, after package initialization.
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .parser import ParsedStep, StepParser
    from .router import ControllerRouter
    from .step_runner import StepRunner

_LAZY_EXPORTS = {
    'ParsedStep': ('.parser', 'ParsedStep'),
    'StepParser': ('.parser', 'StepParser'),
    'ControllerRouter': ('.router', 'ControllerRouter'),
    'StepRunner': ('.step_runner', 'StepRunner'),
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
    'StepParser',
    'ParsedStep',
    'ControllerRouter',
    'StepRunner',
]
