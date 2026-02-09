"""
Integration test configuration — per-test workspace isolation.

Each integration test gets its own temporary workspace directory so that
DuckDB stores (``store.duckdb``) do not accumulate state across tests
within the same xdist worker.  This prevents foreign-key constraint
errors that arise when unrelated tests share a single database file.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_workspace(tmp_path, monkeypatch):
    """Give each integration test its own workspace directory."""
    workspace = tmp_path / "_test_workspace"
    workspace.mkdir()
    monkeypatch.setenv("NIRS4ALL_WORKSPACE", str(workspace))

    # Reset the module-level singleton so get_active_workspace()
    # re-reads the (now per-test) environment variable.
    import nirs4all.workspace as ws

    monkeypatch.setattr(ws, "_active_workspace", None)
