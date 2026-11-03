"""Lossless nested sklearn constructor values within the existing JSON DSL.

Plain parameters retain their historical JSON representation. Nested estimators
reuse the library's component serializer rather than becoming repr strings, and
non-finite floats (e.g. XGBoost's ``missing=nan``), which strict JSON cannot
carry, are marked so they round-trip exactly.
This is a trusted library configuration format, not a safe untrusted-code loader.
"""

from __future__ import annotations

import math
from dataclasses import is_dataclass
from typing import Any

import numpy as np

_COMPONENT = "__nirs4all_constructor_component_v1__"


def encode_constructor_value(value: Any) -> Any:
    """Encode nested estimators while preserving ordinary parameter mappings."""
    if isinstance(value, np.ndarray):
        return encode_constructor_value(value.tolist())
    if isinstance(value, np.generic):
        return encode_constructor_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {_COMPONENT: {"kind": "float", "value": repr(value)}}
    if not isinstance(value, type) and (
        callable(getattr(value, "get_params", None)) or is_dataclass(value)
    ):
        from nirs4all.pipeline.config.component_serialization import serialize_component

        return {_COMPONENT: {"kind": "component", "value": encode_constructor_value(serialize_component(value))}}
    if isinstance(value, dict):
        if _COMPONENT in value or any(not isinstance(key, str) for key in value):
            # Escape marker collisions and keys JSON objects would stringify.
            return {_COMPONENT: {"kind": "mapping", "value": [[key, encode_constructor_value(item)] for key, item in value.items()]}}
        return {key: encode_constructor_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_constructor_value(item) for item in value]
    return value


def decode_constructor_value(value: Any) -> Any:
    """Restore only explicitly marked components, never interpret plain strings."""
    if isinstance(value, dict):
        if set(value) == {_COMPONENT}:
            payload = value[_COMPONENT]
            if not isinstance(payload, dict) or set(payload) != {"kind", "value"}:
                raise ValueError("Malformed nested constructor component")
            if payload["kind"] == "mapping":
                return {key: decode_constructor_value(item) for key, item in payload["value"]}
            if payload["kind"] == "float":
                if payload["value"] not in ("nan", "inf", "-inf"):
                    raise ValueError("Malformed non-finite constructor float")
                return float(payload["value"])
            if payload["kind"] == "component":
                from nirs4all.pipeline.config.component_serialization import deserialize_component

                return deserialize_component(decode_constructor_value(payload["value"]))
            raise ValueError("Unknown nested constructor component kind")
        return {key: decode_constructor_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_constructor_value(item) for item in value]
    return value
