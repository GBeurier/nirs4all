"""Contract coverage for the fail-closed native lifecycle capability matrix."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

import nirs4all
from nirs4all.api.native_capabilities import (
    NATIVE_CAPABILITY_DISPOSITIONS,
    NATIVE_CAPABILITY_OPERATIONS,
    get_native_capability_matrix,
    native_capability_matrix_schema,
    validate_native_capability_matrix,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_FORMS: dict[str, dict[str, str]] = {
    "run": {"portable_methods": "native", "unsupported_request": "refused"},
    "predict": {
        "in_memory_result_or_session": "native",
        "archive_v2": "native",
        "archive_v3": "native",
        "workspace_publication": "refused",
    },
    "session": {"native_methods": "native"},
    "load_session": {"archive_v2": "native", "archive_v3": "native", "non_archive_path": "refused"},
    "save": {"archive_v2": "native", "archive_v3": "native"},
    "export": {"archive_v2": "native", "archive_v3": "native"},
    "retrain": {
        "full_in_memory": "native",
        "transfer": "refused",
        "finetune": "refused",
        "archive_source": "refused",
    },
    "explain": {"native_request": "refused"},
    "generate": {"native_request": "refused"},
}


def _resolve_public_api(path: str) -> object:
    """Resolve a matrix public surface from the installed top-level package."""

    target: object = nirs4all
    for attribute in path.split(".")[1:]:
        target = getattr(target, attribute)
    return target


def test_packaged_native_capability_matrix_is_complete_and_fail_closed() -> None:
    """The installed authority records every concrete lifecycle form."""

    matrix = get_native_capability_matrix()
    schema = native_capability_matrix_schema()

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(matrix, schema)

    assert set(NATIVE_CAPABILITY_DISPOSITIONS) == {"native", "plugin", "refused"}
    assert set(matrix["operations"]) == set(NATIVE_CAPABILITY_OPERATIONS)
    assert {
        operation: {form_name: form["disposition"] for form_name, form in entry["forms"].items()}
        for operation, entry in matrix["operations"].items()
    } == EXPECTED_FORMS
    assert all(
        form["fallback"] == "forbidden"
        for entry in matrix["operations"].values()
        for form in entry["forms"].values()
    )
    assert all(
        form["disposition"] != "plugin"
        for entry in matrix["operations"].values()
        for form in entry["forms"].values()
    )
    assert "legacy" not in json.dumps(matrix).lower()

    for entry in matrix["operations"].values():
        for form in entry["forms"].values():
            assert callable(_resolve_public_api(form["public_api"]))

    evidence = matrix["evidence_sources"]
    assert "source_ledgers" not in matrix
    assert evidence["compatibility_context"]["role"] == "parity_tolerance_context_only"
    assert (REPOSITORY_ROOT / evidence["compatibility_context"]["path"]).is_file()
    assert all((REPOSITORY_ROOT / path).is_file() for path in evidence["runtime_tests"]["paths"])

    lifecycle = evidence["lifecycle_reference"]
    lifecycle_text = (REPOSITORY_ROOT / lifecycle["path"]).read_text(encoding="utf-8")
    assert f"## {lifecycle['anchor'].replace('-', ' ')}" in lifecycle_text.lower()
    assert "get_native_capability_matrix" in lifecycle_text
    assert "parity-tolerance context only" in lifecycle_text


def test_native_capability_matrix_requires_forms_callable_plugins_and_scoped_evidence() -> None:
    """Unsupported shortcuts cannot turn into implied routes or semantic evidence."""

    missing_forms = copy.deepcopy(get_native_capability_matrix())
    missing_forms["operations"]["run"] = {}
    with pytest.raises(ValueError, match="forms"):
        validate_native_capability_matrix(missing_forms)

    missing_callable_plugin = copy.deepcopy(get_native_capability_matrix())
    explain_form = missing_callable_plugin["operations"]["explain"]["forms"]["native_request"]
    explain_form["disposition"] = "plugin"
    explain_form["plugin"] = {"activation": "explicit", "id": "shap"}
    with pytest.raises(ValueError, match="callable_api"):
        validate_native_capability_matrix(missing_callable_plugin)

    fallback_route = copy.deepcopy(get_native_capability_matrix())
    fallback_route["operations"]["generate"]["forms"]["native_request"]["fallback"] = "legacy"
    with pytest.raises(ValueError, match="forbidden"):
        validate_native_capability_matrix(fallback_route)

    semantic_compatibility_ledger = copy.deepcopy(get_native_capability_matrix())
    semantic_compatibility_ledger["evidence_sources"]["compatibility_context"]["role"] = "semantic_authority"
    with pytest.raises(ValueError, match="parity_tolerance_context_only"):
        validate_native_capability_matrix(semantic_compatibility_ledger)


def test_native_capability_matrix_returns_a_detached_document() -> None:
    """Consumers cannot mutate the packaged authority record in process."""

    first = get_native_capability_matrix()
    first["operations"]["run"]["forms"]["portable_methods"]["boundary"] = "mutated"

    assert get_native_capability_matrix()["operations"]["run"]["forms"]["portable_methods"]["boundary"] != "mutated"
