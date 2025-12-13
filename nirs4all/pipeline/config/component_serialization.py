import inspect
from enum import Enum
from typing import Any, get_type_hints, get_origin, get_args, Annotated, Union
import importlib
import json

# Simple alias dictionary for common transformations
build_aliases = {
    # Add common aliases here if needed
}


def _is_meta_estimator(obj) -> bool:
    """Check if object is a stacking/voting meta-estimator.

    Args:
        obj: Object to check.

    Returns:
        True if object is a meta-estimator (has estimators and final_estimator).
    """
    try:
        from sklearn.ensemble import (
            StackingRegressor, StackingClassifier,
            VotingRegressor, VotingClassifier
        )
        meta_types = (StackingRegressor, StackingClassifier,
                      VotingRegressor, VotingClassifier)
        return isinstance(obj, meta_types)
    except ImportError:
        return False


def _is_meta_estimator_class(cls) -> bool:
    """Check if class is a meta-estimator type.

    Args:
        cls: Class to check.

    Returns:
        True if class is a meta-estimator type.
    """
    try:
        from sklearn.ensemble import (
            StackingRegressor, StackingClassifier,
            VotingRegressor, VotingClassifier
        )
        return cls in (StackingRegressor, StackingClassifier,
                       VotingRegressor, VotingClassifier)
    except ImportError:
        return False


def _serialize_meta_estimator(obj) -> dict:
    """Serialize a stacking/voting meta-estimator with nested estimators.

    Handles StackingRegressor, StackingClassifier, VotingRegressor, VotingClassifier
    by recursively serializing their base estimators and final_estimator.

    Args:
        obj: Meta-estimator instance to serialize.

    Returns:
        Dictionary with class path, estimators, final_estimator, and other params.
    """
    result = {
        "class": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
        "params": {}
    }

    # Serialize each base estimator as (name, serialized_estimator) pairs
    if hasattr(obj, 'estimators') and obj.estimators is not None:
        result["params"]["estimators"] = [
            [name, serialize_component(est)]
            for name, est in obj.estimators
        ]

    # Serialize final_estimator (for stacking models)
    if hasattr(obj, 'final_estimator') and obj.final_estimator is not None:
        result["params"]["final_estimator"] = serialize_component(obj.final_estimator)

    # Add other changed params (cv, n_jobs, passthrough, etc.)
    other_params = _changed_kwargs(obj)
    for key in ['estimators', 'final_estimator']:
        other_params.pop(key, None)  # Already handled above
    if other_params:
        result["params"].update(serialize_component(other_params))

    return result


def _deserialize_meta_estimator(cls, params: dict) -> Any:
    """Reconstruct a meta-estimator with nested estimators.

    Args:
        cls: Meta-estimator class (StackingRegressor, etc.).
        params: Serialized parameters including estimators and final_estimator.

    Returns:
        Instantiated meta-estimator with deserialized nested estimators.
    """
    deserialized_params = {}

    # Handle estimators list of [name, estimator_config] tuples
    if "estimators" in params:
        deserialized_params["estimators"] = [
            (name, deserialize_component(est_config))
            for name, est_config in params["estimators"]
        ]

    # Handle final_estimator
    if "final_estimator" in params:
        deserialized_params["final_estimator"] = deserialize_component(
            params["final_estimator"]
        )

    # Deserialize other params normally
    for key, value in params.items():
        if key not in ["estimators", "final_estimator"]:
            deserialized_params[key] = deserialize_component(value)

    return cls(**deserialized_params)

def serialize_component(obj: Any) -> Any:
    """
    Return something that json.dumps can handle.

    Normalizes all syntaxes to canonical form for hash-based uniqueness.
    All instances serialize to their internal module paths with only non-default parameters.
    """

    if obj is None or isinstance(obj, (bool, int, float)):
        return obj

    if isinstance(obj, str):
        # Normalize string module paths to internal module paths for hash consistency
        # e.g., "sklearn.preprocessing.StandardScaler" → "sklearn.preprocessing._data.StandardScaler"
        if "." in obj and not obj.endswith(('.pkl', '.h5', '.keras', '.joblib', '.pt', '.pth')):
            try:
                # Try to import and get canonical internal module path
                mod_name, _, cls_name = obj.rpartition(".")
                mod = importlib.import_module(mod_name)
                cls = getattr(mod, cls_name)
                # Return canonical form (internal module path)
                return f"{cls.__module__}.{cls.__qualname__}"
            except (ImportError, AttributeError):
                # If import fails, pass through as-is (e.g., controller names, invalid paths)
                pass
        return obj

    # Handle Enum instances - serialize as class path + value
    if isinstance(obj, Enum):
        return {
            "enum": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
            "value": obj.value
        }

    if isinstance(obj, dict):
        return {k: serialize_component(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_component(x) for x in obj]
    if isinstance(obj, tuple):
        # Convert tuples to lists for YAML/JSON compatibility
        # Hyperparameter range specifications like ('int', min, max) become ['int', min, max]
        return [serialize_component(x) for x in obj]

    if inspect.isclass(obj):
        return f"{obj.__module__}.{obj.__qualname__}"

    # Special handling for stacking/ensemble meta-estimators
    # Must be checked BEFORE generic instance serialization
    if _is_meta_estimator(obj):
        return _serialize_meta_estimator(obj)

    # Handle numpy arrays and other array-like objects
    # Convert to list for JSON/YAML serialization
    if hasattr(obj, '__array__') or (hasattr(obj, 'tolist') and hasattr(obj, 'shape')):
        try:
            return obj.tolist()
        except (AttributeError, TypeError):
            pass

    params = _changed_kwargs(obj)

    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        func_serialized = {
            "function": f"{obj.__module__}.{obj.__name__}"
        }
        if params:
            func_serialized["params"] = serialize_component(params)

        # Store framework as string (not runtime instance) for JSON serialization
        if hasattr(obj, 'framework'):
            func_serialized["framework"] = obj.framework

        return func_serialized

    def_serialized = f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"

    if params:
        def_serialized = {
            "class": def_serialized,
            "params": serialize_component(params),
        }

    return def_serialized


def deserialize_component(blob: Any, infer_type: Any = None) -> Any:
    """Turn the output of serialize_component back into live objects."""
    # --- trivial cases ------------------------------------------------------ #
    if blob is None or isinstance(blob, (bool, int, float)):
        # Type validation - int and float are considered compatible for numeric values
        if infer_type is not None and infer_type is not type(None):
            if not isinstance(blob, infer_type):
                # Allow int/float cross-compatibility for numeric types
                if not (isinstance(blob, (int, float)) and infer_type in (int, float)):
                    # Debug-level info only - the value is still returned as-is
                    pass  # Removed verbose warning - type mismatch is handled gracefully
        return blob

    if isinstance(blob, str):
        if blob in build_aliases:
            blob = build_aliases[blob]
        try:
            # try to import the module and get the class or function
            # Safety check for empty or invalid strings
            if not blob or "." not in blob:
                return blob

            mod_name, _, cls_or_func_name = blob.rpartition(".")

            # Safety check for empty module name
            if not mod_name:
                return blob

            mod = importlib.import_module(mod_name)
            cls_or_func = getattr(mod, cls_or_func_name)

            # Try to instantiate without parameters
            try:
                return cls_or_func()
            except TypeError as e:
                # If instantiation fails due to missing required parameters,
                # check if there are required parameters without defaults
                if inspect.isclass(cls_or_func):
                    sig = inspect.signature(cls_or_func.__init__)
                    required_params = [
                        name for name, param in sig.parameters.items()
                        if name != "self" and param.default is inspect._empty
                    ]
                    if required_params:
                        raise TypeError(
                            f"Cannot deserialize {blob} from string representation: "
                            f"class requires parameters {required_params} but none were provided. "
                            f"This usually means the serialization failed to capture required parameters. "
                            f"Original error: {e}"
                        ) from e
                raise
        except (ImportError, AttributeError):
            return blob

    if isinstance(blob, list):
        if infer_type is not None and isinstance(infer_type, type):
            if issubclass(infer_type, tuple):
                return tuple(deserialize_component(x) for x in blob)
            # Handle numpy array deserialization
            try:
                import numpy as np
                is_numpy_type = (infer_type is np.ndarray or
                                 (hasattr(infer_type, '__module__') and
                                  infer_type.__module__ == 'numpy' and
                                  infer_type.__name__ == 'ndarray'))
                if is_numpy_type:
                    return np.array(blob)
            except ImportError:
                pass
        return [deserialize_component(x) for x in blob]

    if isinstance(blob, dict):
        # Handle Enum deserialization
        if "enum" in blob and "value" in blob:
            enum_path = blob["enum"]
            mod_name, _, enum_name = enum_path.rpartition(".")
            try:
                mod = importlib.import_module(mod_name)
                enum_cls = getattr(mod, enum_name)
                return enum_cls(blob["value"])
            except (ImportError, AttributeError, ValueError) as e:
                print(f"Failed to deserialize enum {enum_path}: {e}")
                return blob

        if any(key in blob for key in ("class", "function", "instance")):
            key = "class" if "class" in blob else "function" if "function" in blob else "instance"

            # Safety check for empty or None values
            if not blob[key] or not isinstance(blob[key], str):
                print(f"Invalid {key} value in blob: {blob[key]}")
                return blob

            mod_name, _, cls_or_func_name = blob[key].rpartition(".")

            # Safety check for empty module name
            if not mod_name:
                print(f"Empty module name for {key}: {blob[key]}")
                return blob

            try:
                mod = importlib.import_module(mod_name)
                cls_or_func = getattr(mod, cls_or_func_name)
            except (ImportError, AttributeError):
                print(f"Failed to import {blob[key]}")
                return blob

            # Special handling for meta-estimators (stacking/voting)
            # Must deserialize nested estimators properly
            if key == "class" and _is_meta_estimator_class(cls_or_func) and "params" in blob:
                return _deserialize_meta_estimator(cls_or_func, blob["params"])

            params = {}
            if "params" in blob:
                # print(blob)
                for k, v in blob["params"].items():
                    # resolved_type = _resolve_type(cls_or_func, k)
                    # print(k, v, resolved_type)
                    params[k] = deserialize_component(v, _resolve_type(cls_or_func, k))

            try:
                # Special handling for model factory functions with @framework decorator
                # These need dataset-dependent parameters (like input_shape) so we return
                # them as dict for controllers to instantiate
                if key == "function" and hasattr(cls_or_func, 'framework'):
                    # Return dict for controller instantiation (no runtime instance)
                    return {
                        "type": "function",
                        "func": cls_or_func,
                        "framework": cls_or_func.framework,
                        "params": params
                    }

                if key == "class" or key == "instance" or key == "function":
                    return cls_or_func(**params)

                # Fallback for other cases
                if len(params) == 0:
                    return cls_or_func
                else:
                    return {
                        key: cls_or_func,
                        "params": params
                    }

            except TypeError:
                print(f"Failed to instantiate {cls_or_func} with params {params}")
                sig = inspect.signature(cls_or_func)
                allowed = {n for n in sig.parameters if n != "self"}
                filtered = {k: v for k, v in params.items() if k in allowed}

                # Check again if this is a model factory function
                if hasattr(cls_or_func, 'framework'):
                    # Return dict for controller instantiation
                    return {
                        "type": "function",
                        "func": cls_or_func,
                        "framework": cls_or_func.framework,
                        "params": filtered
                    }

                return cls_or_func(**filtered)

        return {k: deserialize_component(v) for k, v in blob.items()}

    # should not reach here
    return blob


def _changed_kwargs(obj):
    """Return {param: value} for every __init__ param whose current
    value differs from its default."""
    sig = inspect.signature(obj.__class__.__init__)
    out = {}

    # Check if object is a Flax module to skip internal fields like 'parent'
    is_flax_module = False
    try:
        import flax.linen as nn
        if isinstance(obj, nn.Module):
            is_flax_module = True
    except ImportError:
        pass

    # Get params dict if available (standard sklearn API)
    obj_params = {}
    if hasattr(obj, 'get_params'):
        try:
            obj_params = obj.get_params(deep=False)
        except Exception:
            pass

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        if is_flax_module and name == 'parent':
            continue

        default = param.default if param.default is not inspect._empty else None

        try:
            current = getattr(obj, name)
        except AttributeError:
            # Try to get from get_params() if available
            if name in obj_params:
                current = obj_params[name]
            else:
                # fall back to what's in cvargs if it exists
                current = obj.__dict__.get("cvargs", {}).get(name, default)

        # Handle comparison with numpy arrays and other array-like objects
        try:
            is_different = current != default
            # For numpy arrays and similar, convert boolean array to single boolean
            if hasattr(is_different, '__iter__') and not isinstance(is_different, str):
                # any(is_different) means at least one element differs
                is_different = any(is_different) if hasattr(is_different, '__len__') else True
        except (ValueError, TypeError):
            # If comparison fails (e.g., array vs None), consider them different
            is_different = True

        if is_different:
            if isinstance(current, tuple):
                current = list(current)
            # out[name] = (current, current_type)
            out[name] = current
    return out


def _resolve_type(obj_or_cls: Any, name: str) -> Union[type, Any, None]:
    """Resolve the type of a parameter in a class or function
    based on its signature or type hints.
    If the parameter is not found, return None.
    If the parameter has a default value, return its type.
    If the parameter has no default value, return the type of the
    attribute with the same name in the class or instance
    If the parameter is not found in the signature or type hints,
    return None.
    """
    if obj_or_cls is None:
        return None

    cls = obj_or_cls if inspect.isclass(obj_or_cls) else obj_or_cls.__class__
    sig = inspect.signature(cls.__init__)

    if name in sig.parameters:
        if sig.parameters[name].default is inspect._empty:
            if sig.parameters[name].annotation is not inspect._empty:
                # print(f"Using annotation for {name}: {sig.parameters[name].annotation}")
                ann = sig.parameters[name].annotation
                while get_origin(ann) is Annotated:
                    ann = get_args(ann)[0]
                origin = get_origin(ann)

                if origin is not None:
                    return origin
                else:
                    return ann
            else:
                if hasattr(obj_or_cls, name):
                    return type(getattr(obj_or_cls, name))
                else:
                    return None
        else:
            return type(sig.parameters[name].default)

    class_hints = get_type_hints(cls, include_extras=True)
    if name in class_hints:
        return class_hints[name]

    init_hints = get_type_hints(cls.__init__, include_extras=True)
    init_hints.pop('return', None)
    if name in init_hints:
        return init_hints[name]

    if not inspect.isclass(obj_or_cls) and hasattr(obj_or_cls, name):
        return type(getattr(obj_or_cls, name))

    return None
