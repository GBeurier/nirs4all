"""Bounded, immutable model metadata reads owned by the workspace library."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from nirs4all.pipeline.storage.store_queries import GET_CHAIN, build_chain_summary_query
from nirs4all.pipeline.storage.store_schema import SCHEMA_VERSION


def read_model_catalogue(workspace_path: str | Path, *, max_models: int = 10000) -> list[dict[str, Any]]:
    """Read chain summaries and metadata without opening a writable store.

    This follows the immutable Studio Store-v5 read contract: active journals,
    unsupported schemas and changes during a read are refused. No schema
    migration, model deserialization, array reconciliation or file creation
    occurs. The existing chain-summary SQL remains the source of score meaning.
    """
    if type(max_models) is not int or not 0 < max_models <= 10000:
        raise ValueError("max_models must be an integer between 1 and 10000")
    database = Path(workspace_path) / "store.sqlite"
    if not database.is_file():
        raise FileNotFoundError(f"WorkspaceStore database not found: {database}")
    sidecars = [Path(f"{database}{suffix}") for suffix in ("-wal", "-shm", "-journal")]

    def signature() -> tuple[int, int, int, int]:
        if any(path.exists() for path in sidecars):
            raise RuntimeError("Model catalogue refuses an active SQLite journal")
        stat = database.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    before = signature()
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
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
        if signature() != before:
            raise RuntimeError("Model catalogue detected a database change during immutable read")
