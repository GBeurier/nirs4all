"""
OperatorRegistry.py
A registry for operator controllers in the nirs4all pipeline.
"""

from .operator_controller import OperatorController

CONTROLLER_REGISTRY = []
def register_controller(wrapper_cls: OperatorController):
    """Decorator to register a wrapper class."""
    CONTROLLER_REGISTRY.append(wrapper_cls)
    CONTROLLER_REGISTRY.sort(key=lambda c: c.priority)
    return wrapper_cls
