"""Bounded, transactional model metadata reads owned by the workspace library."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from nirs4all.pipeline.storage.store_queries import GET_CHAIN, build_chain_summary_query
from nirs4all.pipeline.storage.store_schema import SCHEMA_VERSION


def read_model_catalogue(workspace_path: str | Path, *, max_models: int = 10000) -> list[dict[str, Any]]:
    """Read chain summaries and metadata without opening a writable store.

    One read-only transaction includes committed WAL data while another owner
    uses the workspace. Unsupported schemas are refused without migrations,
    model deserialization or array reconciliation. SQLite may maintain its
    coordination files; persisted data is never written by this reader.
    The existing chain-summary SQL remains the source of score meaning.
    """
    if type(max_models) is not int or not 0 < max_models <= 10000:
        raise ValueError("max_models must be an integer between 1 and 10000")
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
        rows = connection.execute(query, params).fetchmany(max_models + 1)
        if len(rows) > max_models:
            raise ValueError("Model catalogue exceeds the bounded response capacity")
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
