"""Contract coverage for the fail-closed native lifecycle capability matrix."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from nirs4all.api.native_capabilities import (
    NATIVE_CAPABILITY_OPERATIONS,
    get_native_capability_matrix,
    native_capability_matrix_schema,
    validate_native_capability_matrix,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_packaged_native_capability_matrix_is_complete_and_fail_closed() -> None:
    """The installed authority covers every lifecycle operation without fallback."""

    matrix = get_native_capability_matrix()
    schema = native_capability_matrix_schema()

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(matrix, schema)

    assert set(matrix["operations"]) == set(NATIVE_CAPABILITY_OPERATIONS)
    assert {name: entry["disposition"] for name, entry in matrix["operations"].items()} == {
        "run": "native",
        "predict": "native",
        "session": "native",
        "load_session": "native",
        "save": "native",
        "export": "native",
        "retrain": "native",
        "explain": "plugin",
        "generate": "refused",
    }
    assert all(entry["fallback"] == "forbidden" for entry in matrix["operations"].values())
    assert "legacy" not in json.dumps(matrix).lower()

    compatibility_path = REPOSITORY_ROOT / matrix["source_ledgers"]["compatibility"]
    lifecycle_reference = matrix["source_ledgers"]["lifecycle"]
    lifecycle_path, _, lifecycle_anchor = lifecycle_reference.partition("#")
    lifecycle_text = (REPOSITORY_ROOT / lifecycle_path).read_text(encoding="utf-8")

    assert json.loads(compatibility_path.read_text(encoding="utf-8"))["owner"] == "nirs4all compatibility ledger"
    assert f"## {lifecycle_anchor.replace('-', ' ')}" in lifecycle_text.lower()
    assert "get_native_capability_matrix" in lifecycle_text


def test_native_capability_matrix_requires_explicit_plugin_and_forbids_fallback() -> None:
    """A plugin cannot be implied and no operation can add a fallback route."""

    missing_plugin = copy.deepcopy(get_native_capability_matrix())
    missing_plugin["operations"]["explain"].pop("plugin")
    with pytest.raises(ValueError, match="plugin"):
        validate_native_capability_matrix(missing_plugin)

    fallback_route = copy.deepcopy(get_native_capability_matrix())
    fallback_route["operations"]["generate"]["fallback"] = "legacy"
    with pytest.raises(ValueError, match="forbidden"):
        validate_native_capability_matrix(fallback_route)


def test_native_capability_matrix_returns_a_detached_document() -> None:
    """Consumers cannot mutate the packaged authority record in process."""

    first = get_native_capability_matrix()
    first["operations"]["run"]["boundary"] = "mutated"

    assert get_native_capability_matrix()["operations"]["run"]["boundary"] != "mutated"
