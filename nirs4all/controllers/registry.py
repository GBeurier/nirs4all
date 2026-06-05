"""
OperatorRegistry.py
A registry for operator controllers in the nirs4all pipeline.
"""

from .controller import OperatorController

# The registry is a single shared list object that is mutated IN PLACE for its
# whole lifetime and never rebound -- callers (e.g. StepRouter) and the module-level
# re-exports in __init__.py capture this object by reference, so rebinding it would
# leave them pointing at a stale list. register_controller/reset_registry therefore
# only ever .append()/.sort()/.clear() it; no `global` reassignment.
CONTROLLER_REGISTRY: list[type[OperatorController]] = []
def register_controller(operator_cls: type[OperatorController]):
    """Decorator to register a controller class."""
    if not issubclass(operator_cls, OperatorController):
        raise TypeError(f"Operator class {operator_cls.__name__} must inherit from OperatorController")

    # Check if controller is already registered (avoid duplicates)
    if any(c.__name__ == operator_cls.__name__ and c.__module__ == operator_cls.__module__
           for c in CONTROLLER_REGISTRY):
        # print(f"Controller {operator_cls.__name__} already registered, skipping...")
        return operator_cls

    # print(f"Registering controller: {operator_cls.__name__}")
    CONTROLLER_REGISTRY.append(operator_cls)
    # Secondary sort by class name makes same-priority ordering deterministic
    # (independent of controller import order), so routing is reproducible.
    CONTROLLER_REGISTRY.sort(key=lambda c: (c.priority, c.__name__))
    # print(f"Registry now has {len(CONTROLLER_REGISTRY)} controllers: {[c.__name__ for c in CONTROLLER_REGISTRY]}")
    return operator_cls

def reset_registry():
    """Reset the controller registry IN PLACE.

    Clears the shared list rather than rebinding it, so references already held
    by the StepRouter and the ``CONTROLLER_REGISTRY`` re-exports stay valid and
    observe subsequent re-registrations.
    """
    CONTROLLER_REGISTRY.clear()
