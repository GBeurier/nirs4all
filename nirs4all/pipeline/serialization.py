import inspect
from typing import Any, get_type_hints, get_origin, get_args, Annotated, Union
import importlib
import json

# Simple alias dictionary for common transformations
build_aliases = {
    # Add common aliases here if needed
}

def serialize_component(obj: Any, include_runtime: bool = True) -> Any:
    """Return something that json.dumps can handle."""

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k: serialize_component(v, include_runtime) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_component(x, include_runtime) for x in obj]
    if isinstance(obj, tuple):
        # Preserve tuples that look like hyperparameter range specifications
        # e.g., ('int', min, max) or ('float', min, max)
        if (len(obj) == 3 and isinstance(obj[0], str) and
                obj[0] in ['int', 'float'] and
                isinstance(obj[1], (int, float)) and
                isinstance(obj[2], (int, float))):
            return obj  # Preserve tuple for hyperparameter ranges
        else:
            return [serialize_component(x, include_runtime) for x in obj]

    if inspect.isclass(obj):
        return f"{obj.__module__}.{obj.__qualname__}"

    params = _changed_kwargs(obj)

    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        func_serialized = {
            "function": f"{obj.__module__}.{obj.__name__}"
        }
        if params:
            func_serialized["params"] = serialize_component(params, include_runtime)
        return func_serialized

    def_serialized = f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"

    if params:
        def_serialized = {
            "class": def_serialized,
            "params": serialize_component(params),
        }
        if include_runtime:
            def_serialized["_runtime_instance"] = obj
    elif include_runtime:
        def_serialized = {
            "class": def_serialized,
            "_runtime_instance": obj
        }

    return def_serialized



def deserialize_component(blob: Any, infer_type: Any = None) -> Any:
    """Turn the output of serialize_component back into live objects."""
    # --- trivial cases ------------------------------------------------------ #
    if blob is None or isinstance(blob, (bool, int, float)):
        if infer_type is not None and infer_type is not type(None):
            if not isinstance(blob, infer_type):
                print(f"Type mismatch: {type(blob)} != {infer_type}")
                return blob
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
            return cls_or_func()
        except (ImportError, AttributeError):
            return blob

    if isinstance(blob, list):
        if infer_type is not None and isinstance(infer_type, type):
            if issubclass(infer_type, tuple):
                return tuple(deserialize_component(x) for x in blob)
        return [deserialize_component(x) for x in blob]

    if isinstance(blob, dict):
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

            params = {}
            if "params" in blob:
                # print(blob)
                for k, v in blob["params"].items():
                    # resolved_type = _resolve_type(cls_or_func, k)
                    # print(k, v, resolved_type)
                    params[k] = deserialize_component(v, _resolve_type(cls_or_func, k))

            try:
                if key == "class" or key == "instance" or key == "function":
                    return cls_or_func(**params)
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
                return cls_or_func(**filtered)

        return {k: deserialize_component(v) for k, v in blob.items()}

    # should not reach here
    return blob


def _changed_kwargs(obj):
    """Return {param: value} for every __init__ param whose current
    value differs from its default."""
    sig = inspect.signature(obj.__class__.__init__)
    out = {}

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        default = param.default if param.default is not inspect._empty else None

        try:
            current = getattr(obj, name)
        except AttributeError:
            # fall back to what's in cvargs if it exists
            current = obj.__dict__.get("cvargs", {}).get(name, default)

        # Handle comparison with numpy arrays and other array-like objects
        try:
            is_different = current != default
            # For numpy arrays and similar, convert boolean array to single boolean
            if hasattr(is_different, '__iter__') and not isinstance(is_different, str):
                is_different = not all(is_different) if hasattr(is_different, '__len__') else True
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
