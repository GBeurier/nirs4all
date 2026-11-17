"""Bounded, transactional model metadata reads owned by the workspace library."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from nirs4all.pipeline.storage.store_queries import GET_CHAIN, build_chain_summary_query
from nirs4all.pipeline.storage.store_schema import SCHEMA_VERSION


def read_model_catalogue(workspace_path: str | Path, *, max_models: int | None = None) -> list[dict[str, Any]]:
    """Read chain summaries and metadata without opening a writable store.

    One read-only transaction includes committed WAL data while another owner
    uses the workspace. Unsupported schemas are refused without migrations,
    model deserialization or array reconciliation. SQLite may maintain its
    coordination files; persisted data is never written by this reader.
    The existing chain-summary SQL remains the source of score meaning.

    Args:
        workspace_path: Workspace directory holding ``store.sqlite``.
        max_models: Optional host-supplied bound on the number of chains; ``None``
            reads every chain. A catalogue larger than the bound is refused, never truncated.

    Returns:
        One ``{"summary", "chain"}`` record per chain.
    """
    if max_models is not None and (type(max_models) is not int or max_models < 1):
        raise ValueError("max_models must be a positive integer or None")
    database = Path(workspace_path) / "store.sqlite"
    if not database.is_file():
        raise FileNotFoundError(f"WorkspaceStore database not found: {database}")
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise RuntimeError(f"Model catalogue requires WorkspaceStore schema {SCHEMA_VERSION}, got {version}")
        query, params = build_chain_summary_query()
        cursor = connection.execute(query, params)
        rows = cursor.fetchall() if max_models is None else cursor.fetchmany(max_models + 1)
        if max_models is not None and len(rows) > max_models:
            raise ValueError(f"Model catalogue exceeds the host bound of {max_models} models")
        result = []
        for row in rows:
            summary = dict(row)
            raw_chain = connection.execute(GET_CHAIN, [summary["chain_id"]]).fetchone()
            if raw_chain is None:
                raise RuntimeError("Model catalogue chain is missing")
            chain = dict(raw_chain)
            for field in ("steps", "fold_artifacts", "shared_artifacts", "branch_path", "relation_replay_manifest"):
                value = chain.get(field)
                chain[field] = json.loads(value) if value is not None else None
            result.append({"summary": summary, "chain": chain})
        return result
    finally:
        connection.close()
