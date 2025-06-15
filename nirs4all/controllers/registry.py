"""
OperatorRegistry.py
A registry for operator controllers in the nirs4all pipeline.
"""

from .controller import OperatorController

CONTROLLER_REGISTRY = []
def register_controller(operator_cls: OperatorController):
    """Decorator to register a controller class."""
    print(f"Registering controller: {operator_cls.__name__}")
    CONTROLLER_REGISTRY.append(operator_cls)
    CONTROLLER_REGISTRY.sort(key=lambda c: c.priority)
    print(f"Registry now has {len(CONTROLLER_REGISTRY)} controllers: {[c.__name__ for c in CONTROLLER_REGISTRY]}")
    return operator_cls
