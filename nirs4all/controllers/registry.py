"""
OperatorRegistry.py
A registry for operator controllers in the nirs4all pipeline.
"""

from typing import Type
from .controller import OperatorController

CONTROLLER_REGISTRY = []
def register_controller(operator_cls: Type[OperatorController]):
    """Decorator to register a controller class."""
    global CONTROLLER_REGISTRY
    if not issubclass(operator_cls, OperatorController):
        raise TypeError(f"Operator class {operator_cls.__name__} must inherit from OperatorController")

    # Check if controller is already registered (avoid duplicates)
    if any(c.__name__ == operator_cls.__name__ and c.__module__ == operator_cls.__module__
           for c in CONTROLLER_REGISTRY):
        # print(f"Controller {operator_cls.__name__} already registered, skipping...")
        return operator_cls

    # print(f"Registering controller: {operator_cls.__name__}")
    CONTROLLER_REGISTRY.append(operator_cls)
    CONTROLLER_REGISTRY.sort(key=lambda c: c.priority)
    # print(f"Registry now has {len(CONTROLLER_REGISTRY)} controllers: {[c.__name__ for c in CONTROLLER_REGISTRY]}")
    return operator_cls

def reset_registry():
    """Reset the controller registry."""
    global CONTROLLER_REGISTRY
    CONTROLLER_REGISTRY = []
    # print("Controller registry has been reset.")
