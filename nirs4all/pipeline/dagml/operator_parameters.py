"""Lossless nested sklearn constructor values within the existing JSON DSL.

Plain parameters retain their historical JSON representation. Nested estimators
reuse the library's component serializer rather than becoming repr strings.
This is a trusted library configuration format, not a safe untrusted-code loader.
"""

from __future__ import annotations

from typing import Any

_COMPONENT = "__nirs4all_constructor_component_v1__"


def encode_constructor_value(value: Any) -> Any:
    """Encode nested estimators while preserving ordinary parameter mappings."""
    if not isinstance(value, type) and callable(getattr(value, "get_params", None)):
        from nirs4all.pipeline.config.component_serialization import serialize_component

        return {_COMPONENT: {"kind": "component", "value": serialize_component(value)}}
    if isinstance(value, dict):
        if _COMPONENT in value:
            # Escape an ordinary user mapping which happens to use our marker.
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
            if payload["kind"] == "component":
                from nirs4all.pipeline.config.component_serialization import deserialize_component

                return deserialize_component(payload["value"])
            raise ValueError("Unknown nested constructor component kind")
        return {key: decode_constructor_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_constructor_value(item) for item in value]
    return value
