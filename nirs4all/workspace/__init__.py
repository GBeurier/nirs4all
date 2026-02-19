"""
Workspace management for nirs4all.

Provides:
- Active workspace path management for run output storage.
"""

import os
from pathlib import Path
from typing import Optional

# Global active workspace path
_active_workspace: Path | None = None

def get_active_workspace() -> Path:
    """Get the active workspace path.

    Resolution order:
    1. Explicitly set via set_active_workspace()
    2. NIRS4ALL_WORKSPACE environment variable
    3. ./workspace in current working directory (default)

    Returns:
        Path to the active workspace directory.
    """
    global _active_workspace

    if _active_workspace is not None:
        return _active_workspace

    env_workspace = os.environ.get("NIRS4ALL_WORKSPACE")
    if env_workspace:
        return Path(env_workspace)

    return Path.cwd() / "workspace"

def set_active_workspace(path: str | Path) -> None:
    """Set the active workspace path.

    This also sets the NIRS4ALL_WORKSPACE environment variable to ensure
    child processes and other modules use the same workspace.

    Args:
        path: Path to the workspace directory.
    """
    global _active_workspace
    _active_workspace = Path(path).resolve()
    os.environ["NIRS4ALL_WORKSPACE"] = str(_active_workspace)

def reset_active_workspace() -> None:
    """Reset the active workspace to the default.

    Clears the explicitly set workspace and removes the environment variable,
    causing get_active_workspace() to return the default ./workspace path.
    """
    global _active_workspace
    _active_workspace = None
    os.environ.pop("NIRS4ALL_WORKSPACE", None)

__all__ = [
    "get_active_workspace",
    "set_active_workspace",
    "reset_active_workspace",
]
