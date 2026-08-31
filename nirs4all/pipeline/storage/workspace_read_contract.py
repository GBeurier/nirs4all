"""Published read-only projection contract for WorkspaceStore schema v5.

The contract is a distribution resource so native consumers can use the same
bounded SQL projection without importing ``WorkspaceStore`` or recreating its
private schema knowledge. It is intentionally limited to run summaries; array,
artifact, prediction and mutation surfaces need their own contracts.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, Final

from .store_schema import SCHEMA_VERSION

WORKSPACE_STORE_READ_CONTRACT_RESOURCE: Final = "contracts/workspace_store_read_v1.json"
WORKSPACE_STORE_READ_CONTRACT_SCHEMA_ID: Final = "nirs4all.workspace-store-read.v1"
WORKSPACE_STORE_READ_CONTRACT_SCHEMA_VERSION: Final = 1


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
        raise RuntimeError(
            "workspace read contract does not match the installed WorkspaceStore schema version"
        )
    return contract


__all__ = [
    "WORKSPACE_STORE_READ_CONTRACT_RESOURCE",
    "WORKSPACE_STORE_READ_CONTRACT_SCHEMA_ID",
    "WORKSPACE_STORE_READ_CONTRACT_SCHEMA_VERSION",
    "workspace_store_read_contract",
]
