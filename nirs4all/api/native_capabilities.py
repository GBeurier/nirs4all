"""Executable native capability matrix for the stable public lifecycle APIs.

The matrix is descriptive only: it records the fail-closed native profile and
does not select an execution engine or route a request.  Consumers can use it
to decide whether to call a native API, explicitly activate a plugin, or stop
before an unsupported operation reaches a broader runtime.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import resources
from typing import Any, Literal, TypeAlias, cast

from jsonschema import Draft202012Validator

CapabilityDisposition: TypeAlias = Literal["native", "plugin", "refused"]

NATIVE_CAPABILITY_MATRIX_SCHEMA_ID = "https://nirs4all.org/schemas/native-capability-matrix/v1"
NATIVE_CAPABILITY_MATRIX_FILENAME = "native_capability_matrix.v1.json"
NATIVE_CAPABILITY_OPERATIONS = (
    "run",
    "predict",
    "session",
    "load_session",
    "save",
    "export",
    "retrain",
    "explain",
    "generate",
)
NATIVE_CAPABILITY_DISPOSITIONS: tuple[CapabilityDisposition, ...] = ("native", "plugin", "refused")


_NATIVE_CAPABILITY_MATRIX_SCHEMA: dict[str, Any] = {
    "$id": NATIVE_CAPABILITY_MATRIX_SCHEMA_ID,
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "additionalProperties": False,
    "properties": {
        "matrix_id": {"const": "nirs4all.native-capability-matrix"},
        "operations": {
            "additionalProperties": False,
            "properties": {
                operation: {"$ref": "#/$defs/operation"} for operation in NATIVE_CAPABILITY_OPERATIONS
            },
            "required": list(NATIVE_CAPABILITY_OPERATIONS),
            "type": "object",
        },
        "profile": {"const": "native"},
        "schema_version": {"const": 1},
        "source_ledgers": {
            "additionalProperties": False,
            "properties": {
                "compatibility": {"const": "docs/compatibility.json"},
                "lifecycle": {"const": "docs/source/reference/public_interfaces.md#native-lifecycle-capability-matrix"},
            },
            "required": ["compatibility", "lifecycle"],
            "type": "object",
        },
    },
    "required": ["matrix_id", "operations", "profile", "schema_version", "source_ledgers"],
    "title": "nirs4all native public API capability matrix",
    "type": "object",
    "$defs": {
        "operation": {
            "additionalProperties": False,
            "allOf": [
                {
                    "if": {
                        "properties": {"disposition": {"const": "plugin"}},
                        "required": ["disposition"],
                    },
                    "then": {"required": ["plugin"]},
                    "else": {"not": {"required": ["plugin"]}},
                }
            ],
            "properties": {
                "boundary": {"minLength": 1, "type": "string"},
                "disposition": {"enum": list(NATIVE_CAPABILITY_DISPOSITIONS)},
                "fallback": {"const": "forbidden"},
                "plugin": {
                    "additionalProperties": False,
                    "properties": {
                        "activation": {"const": "explicit"},
                        "id": {"minLength": 1, "type": "string"},
                    },
                    "required": ["activation", "id"],
                    "type": "object",
                },
                "public_api": {"minLength": 1, "pattern": "^nirs4all(?:\\.|$)", "type": "string"},
            },
            "required": ["boundary", "disposition", "fallback", "public_api"],
            "type": "object",
        }
    },
}


def native_capability_matrix_schema() -> dict[str, Any]:
    """Return a detached JSON Schema for the native capability matrix."""

    return cast(dict[str, Any], json.loads(json.dumps(_NATIVE_CAPABILITY_MATRIX_SCHEMA, sort_keys=True)))


def validate_native_capability_matrix(payload: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` unless *payload* satisfies the native matrix contract."""

    if not isinstance(payload, Mapping):
        raise TypeError("native capability matrix must be a mapping")

    validator = Draft202012Validator(_NATIVE_CAPABILITY_MATRIX_SCHEMA)
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda error: (tuple(str(part) for part in error.absolute_path), error.message),
    )
    if errors:
        rendered = "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}" for error in errors
        )
        raise ValueError(f"invalid native capability matrix: {rendered}")


def get_native_capability_matrix() -> dict[str, Any]:
    """Load and validate the packaged native capability matrix.

    A new decoded mapping is returned for every call so callers cannot mutate a
    shared authority record in process.
    """

    text = resources.files("nirs4all").joinpath(NATIVE_CAPABILITY_MATRIX_FILENAME).read_text(encoding="utf-8")
    payload = cast(dict[str, Any], json.loads(text))
    validate_native_capability_matrix(payload)
    return payload


__all__ = [
    "CapabilityDisposition",
    "NATIVE_CAPABILITY_DISPOSITIONS",
    "NATIVE_CAPABILITY_MATRIX_FILENAME",
    "NATIVE_CAPABILITY_MATRIX_SCHEMA_ID",
    "NATIVE_CAPABILITY_OPERATIONS",
    "get_native_capability_matrix",
    "native_capability_matrix_schema",
    "validate_native_capability_matrix",
]
