"""
Unit test configuration — per-test workspace isolation.

Each unit test gets its own temporary workspace directory so that
SQLite stores (``store.sqlite``) do not accumulate lock contention
when thousands of tests run in a single process.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_workspace(tmp_path, monkeypatch):
    """Give each unit test its own workspace directory."""
    workspace = tmp_path / "_test_workspace"
    workspace.mkdir()
    monkeypatch.setenv("NIRS4ALL_WORKSPACE", str(workspace))

    # Reset the module-level singleton so get_active_workspace()
    # re-reads the (now per-test) environment variable.
    import nirs4all.workspace as ws

    monkeypatch.setattr(ws, "_active_workspace", None)
