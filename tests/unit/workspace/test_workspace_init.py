"""Unit tests for nirs4all/workspace/__init__.py (D09).

Covers: get_active_workspace(), set_active_workspace(), reset_active_workspace(),
and the resolution-order contract (explicit > env-var > default).
"""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clean_workspace_state(monkeypatch):
    """Restore workspace singleton and env-var after every test."""
    import nirs4all.workspace as ws

    # Save original state
    original_active = ws._active_workspace
    original_env = os.environ.get("NIRS4ALL_WORKSPACE")

    yield

    # Restore state
    ws._active_workspace = original_active
    if original_env is None:
        os.environ.pop("NIRS4ALL_WORKSPACE", None)
    else:
        os.environ["NIRS4ALL_WORKSPACE"] = original_env

# ---------------------------------------------------------------------------
# get_active_workspace() resolution order
# ---------------------------------------------------------------------------

class TestGetActiveWorkspace:

    def test_returns_path_object(self, tmp_path):
        import nirs4all.workspace as ws
        ws.set_active_workspace(tmp_path)
        result = ws.get_active_workspace()
        assert isinstance(result, Path)

    def test_explicit_set_takes_highest_priority(self, tmp_path, monkeypatch):
        import nirs4all.workspace as ws
        monkeypatch.setenv("NIRS4ALL_WORKSPACE", str(tmp_path / "env_dir"))
        explicit = tmp_path / "explicit_dir"
        ws.set_active_workspace(explicit)
        assert ws.get_active_workspace() == explicit.resolve()

    def test_env_var_used_when_no_explicit_set(self, tmp_path, monkeypatch):
        import nirs4all.workspace as ws
        ws._active_workspace = None
        env_path = tmp_path / "from_env"
        monkeypatch.setenv("NIRS4ALL_WORKSPACE", str(env_path))
        assert ws.get_active_workspace() == env_path

    def test_default_is_cwd_workspace_when_nothing_set(self, monkeypatch):
        import nirs4all.workspace as ws
        ws._active_workspace = None
        monkeypatch.delenv("NIRS4ALL_WORKSPACE", raising=False)
        expected = Path.cwd() / "workspace"
        assert ws.get_active_workspace() == expected

    def test_multiple_calls_return_same_path(self, tmp_path):
        import nirs4all.workspace as ws
        ws.set_active_workspace(tmp_path)
        assert ws.get_active_workspace() == ws.get_active_workspace()

# ---------------------------------------------------------------------------
# set_active_workspace()
# ---------------------------------------------------------------------------

class TestSetActiveWorkspace:

    def test_accepts_string_path(self, tmp_path):
        import nirs4all.workspace as ws
        ws.set_active_workspace(str(tmp_path))
        assert ws.get_active_workspace() == tmp_path.resolve()

    def test_accepts_path_object(self, tmp_path):
        import nirs4all.workspace as ws
        ws.set_active_workspace(tmp_path)
        assert ws.get_active_workspace() == tmp_path.resolve()

    def test_sets_env_variable(self, tmp_path):
        import nirs4all.workspace as ws
        ws.set_active_workspace(tmp_path)
        assert os.environ.get("NIRS4ALL_WORKSPACE") == str(tmp_path.resolve())

    def test_resolves_relative_paths(self):
        import nirs4all.workspace as ws
        ws.set_active_workspace("some/relative/path")
        result = ws.get_active_workspace()
        assert result.is_absolute()

    def test_overwrite_with_new_path(self, tmp_path):
        import nirs4all.workspace as ws
        path1 = tmp_path / "ws1"
        path2 = tmp_path / "ws2"
        ws.set_active_workspace(path1)
        ws.set_active_workspace(path2)
        assert ws.get_active_workspace() == path2.resolve()

# ---------------------------------------------------------------------------
# reset_active_workspace()
# ---------------------------------------------------------------------------

class TestResetActiveWorkspace:

    def test_reset_clears_explicit_path(self, tmp_path):
        import nirs4all.workspace as ws
        ws.set_active_workspace(tmp_path)
        ws.reset_active_workspace()
        assert ws._active_workspace is None

    def test_reset_removes_env_variable(self, tmp_path):
        import nirs4all.workspace as ws
        ws.set_active_workspace(tmp_path)
        ws.reset_active_workspace()
        assert "NIRS4ALL_WORKSPACE" not in os.environ

    def test_get_after_reset_returns_default(self, monkeypatch):
        import nirs4all.workspace as ws
        monkeypatch.delenv("NIRS4ALL_WORKSPACE", raising=False)
        ws._active_workspace = None
        # Explicitly set, then reset
        ws.set_active_workspace("/some/explicit/path")
        ws.reset_active_workspace()
        assert ws.get_active_workspace() == Path.cwd() / "workspace"

    def test_reset_is_idempotent(self):
        import nirs4all.workspace as ws
        ws.reset_active_workspace()
        ws.reset_active_workspace()
        assert ws._active_workspace is None

# ---------------------------------------------------------------------------
# __all__ exports
# ---------------------------------------------------------------------------

class TestWorkspaceModuleExports:

    def test_all_is_defined(self):
        import nirs4all.workspace as ws
        assert hasattr(ws, "__all__")

    def test_public_functions_in_all(self):
        import nirs4all.workspace as ws
        for name in ("get_active_workspace", "set_active_workspace", "reset_active_workspace"):
            assert name in ws.__all__, f"{name} missing from __all__"

    def test_functions_callable(self):
        import nirs4all.workspace as ws
        assert callable(ws.get_active_workspace)
        assert callable(ws.set_active_workspace)
        assert callable(ws.reset_active_workspace)
