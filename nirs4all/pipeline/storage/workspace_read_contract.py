"""Published read-only projection contracts for WorkspaceStore schema v5.

The contract is a distribution resource so native consumers can use the same
bounded SQL projections without importing ``WorkspaceStore`` or recreating its
private schema knowledge. It is intentionally limited to bounded run,
pipeline, and explicitly directed chain-ranking projections. Studio's
high-level results policy, arrays, artifacts, predictions and mutations need
their own contracts.

The results-summary contract is deliberately a separate resource: it composes
the stable low-level read projection with the ranking and serialization policy
of the compact Studio results response without changing the read-v1 bytes.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, Final

from .store_schema import SCHEMA_VERSION

WORKSPACE_STORE_READ_CONTRACT_RESOURCE: Final = "contracts/workspace_store_read_v1.json"
WORKSPACE_STORE_READ_CONTRACT_SCHEMA_ID: Final = "nirs4all.workspace-store-read.v1"
WORKSPACE_STORE_READ_CONTRACT_SCHEMA_VERSION: Final = 1
WORKSPACE_STORE_RESULTS_SUMMARY_CONTRACT_RESOURCE: Final = "contracts/workspace_store_results_summary_v1.json"
WORKSPACE_STORE_RESULTS_SUMMARY_CONTRACT_SCHEMA_ID: Final = "nirs4all.workspace-store-results-summary.v1"
WORKSPACE_STORE_RESULTS_SUMMARY_CONTRACT_SCHEMA_VERSION: Final = 1


def workspace_store_read_contract() -> dict[str, Any]:
    """Return the version-checked, read-only WorkspaceStore contract.

    Consumers must refuse a database whose ``PRAGMA user_version`` differs
    from the contract's exact store schema version before executing its query.
    """
    resource = files(__package__).joinpath(WORKSPACE_STORE_READ_CONTRACT_RESOURCE)
    contract: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    if contract.get("schema_id") != WORKSPACE_STORE_READ_CONTRACT_SCHEMA_ID:
        raise RuntimeError("workspace read contract has an unexpected schema id")
    if contract.get("schema_version") != WORKSPACE_STORE_READ_CONTRACT_SCHEMA_VERSION:
        raise RuntimeError("workspace read contract has an unsupported schema version")
    if contract.get("workspace_store_schema_version") != SCHEMA_VERSION:
        raise RuntimeError("workspace read contract does not match the installed WorkspaceStore schema version")
    return contract


def workspace_store_results_summary_contract() -> dict[str, Any]:
    """Return the version-checked Store-v5 results-summary contract.

    This policy is self-contained except for its explicit reference to the
    ``studio_chain_ranked_v1`` projection in :func:`workspace_store_read_contract`.
    It owns the complete, paged source query plus selection, synthetic-refit,
    metric-direction and JSON/null serialization semantics needed by a native
    consumer of the fixed ``n=5`` summary surface.
    """
    resource = files(__package__).joinpath(WORKSPACE_STORE_RESULTS_SUMMARY_CONTRACT_RESOURCE)
    contract: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    if contract.get("schema_id") != WORKSPACE_STORE_RESULTS_SUMMARY_CONTRACT_SCHEMA_ID:
        raise RuntimeError("workspace results-summary contract has an unexpected schema id")
    if contract.get("schema_version") != WORKSPACE_STORE_RESULTS_SUMMARY_CONTRACT_SCHEMA_VERSION:
        raise RuntimeError("workspace results-summary contract has an unsupported schema version")
    if contract.get("workspace_store_schema_version") != SCHEMA_VERSION:
        raise RuntimeError("workspace results-summary contract does not match the installed WorkspaceStore schema version")

    dependency = contract.get("dependencies", {}).get("workspace_store_read")
    read_contract = workspace_store_read_contract()
    if dependency != {
        "schema_id": read_contract["schema_id"],
        "schema_version": read_contract["schema_version"],
        "projection": "studio_chain_ranked_v1",
    }:
        raise RuntimeError("workspace results-summary contract has an incompatible read-contract dependency")
    return contract


__all__ = [
    "WORKSPACE_STORE_READ_CONTRACT_RESOURCE",
    "WORKSPACE_STORE_READ_CONTRACT_SCHEMA_ID",
    "WORKSPACE_STORE_READ_CONTRACT_SCHEMA_VERSION",
    "WORKSPACE_STORE_RESULTS_SUMMARY_CONTRACT_RESOURCE",
    "WORKSPACE_STORE_RESULTS_SUMMARY_CONTRACT_SCHEMA_ID",
    "WORKSPACE_STORE_RESULTS_SUMMARY_CONTRACT_SCHEMA_VERSION",
    "workspace_store_read_contract",
    "workspace_store_results_summary_contract",
]
