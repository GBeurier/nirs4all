"""Compatibility ledger machine-readable authority checks."""

from __future__ import annotations

import json

from ._authority import COMPATIBILITY_JSON, load_compatibility_ledger, validate_compatibility_ledger


def test_compatibility_json_is_valid_json() -> None:
    with COMPATIBILITY_JSON.open("r", encoding="utf-8") as handle:
        assert json.load(handle)["schema_version"] == 1


def test_compatibility_json_matches_live_parity_authority() -> None:
    validate_compatibility_ledger(load_compatibility_ledger())


def test_compatibility_json_publishes_marker_policy() -> None:
    policy = load_compatibility_ledger()["marker_policy"]
    assert policy["schema"] == "nirs4all.pyref.marker_policy.v1"
    assert policy["xfail"]["sanctioned_modules"] == ["test_conformance_dual_engine.py"]
