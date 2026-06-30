"""Compatibility ledger machine-readable authority checks."""

from __future__ import annotations

import json

from ._authority import COMPATIBILITY_JSON, load_compatibility_ledger, validate_compatibility_ledger


def test_compatibility_json_is_valid_json() -> None:
    with COMPATIBILITY_JSON.open("r", encoding="utf-8") as handle:
        assert json.load(handle)["schema_version"] == 1


def test_compatibility_json_matches_live_parity_authority() -> None:
    validate_compatibility_ledger(load_compatibility_ledger())
