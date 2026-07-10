"""Transition helpers for legacy workspace detection and conversion guidance."""

from __future__ import annotations

import shlex
import sqlite3
import warnings
from dataclasses import dataclass
from pathlib import Path

TARGET_WORKSPACE_FORMAT = "nirs4all-workspace-v2"


@dataclass(frozen=True)
class WorkspaceFormatInfo:
    """Stat-only workspace format diagnosis used by the transition release."""

    path: Path
    format: str
    conversion_required: bool
    message: str
    conversion_command: str | None = None


def _quote_path(path: Path | str) -> str:
    return shlex.quote(str(path))


def default_conversion_output(path: Path) -> Path:
    """Return the default fresh output directory for a converted workspace."""
    return path.with_name(f"{path.name}-workspace-v2")


def build_conversion_command(path: Path | str, output: Path | str | None = None) -> str:
    """Build the recommended no-in-place conversion command."""
    source = Path(path)
    target = Path(output) if output is not None else default_conversion_output(source)
    return (
        "nirs4all workspace convert "
        f"{_quote_path(source)} --output {_quote_path(target)} --verify"
    )


def _sqlite_has_prediction_arrays(path: Path) -> bool:
    uri = f"file:{path.as_posix()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as conn:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='prediction_arrays'"
            ).fetchone()
            return row is not None
    except sqlite3.DatabaseError:
        return False


def _has_legacy_runs_tree(path: Path) -> bool:
    runs_dir = path / "runs"
    if not runs_dir.is_dir():
        return False
    return any(runs_dir.glob("*/*/manifest.yaml"))


def inspect_workspace_format(path: Path | str) -> WorkspaceFormatInfo:
    """Inspect a workspace or artifact without opening it through WorkspaceStore."""
    root = Path(path)
    command = build_conversion_command(root)

    if root.is_file():
        if root.suffix in {".n4a", ".py"} or root.name.endswith(".n4a.py"):
            return WorkspaceFormatInfo(
                path=root,
                format="legacy-artifact",
                conversion_required=True,
                message="Legacy nirs4all artifact detected; convert it before using V1 runtimes.",
                conversion_command=command,
            )
        return WorkspaceFormatInfo(root, "unknown-file", False, "File is not a recognized nirs4all workspace artifact.")

    sqlite_path = root / "store.sqlite"
    duckdb_path = root / "store.duckdb"

    if duckdb_path.exists() and not sqlite_path.exists():
        return WorkspaceFormatInfo(
            path=root,
            format="duckdb-workspace",
            conversion_required=True,
            message=(
                "Legacy DuckDB workspace detected. This transition build can still "
                "open it after migration, but users should convert it into a fresh "
                f"{TARGET_WORKSPACE_FORMAT} workspace before switching runtimes."
            ),
            conversion_command=command,
        )

    if sqlite_path.exists():
        if _sqlite_has_prediction_arrays(sqlite_path):
            return WorkspaceFormatInfo(
                path=root,
                format="sqlite-workspace-legacy-arrays",
                conversion_required=True,
                message=(
                    "Workspace uses the legacy prediction_arrays table. Convert it "
                    f"to {TARGET_WORKSPACE_FORMAT} before publishing or sharing it."
                ),
                conversion_command=command,
            )
        return WorkspaceFormatInfo(root, "sqlite-workspace-v2", False, "Workspace is already in the V1 SQLite format.")

    if _has_legacy_runs_tree(root):
        return WorkspaceFormatInfo(
            path=root,
            format="fs-runs-legacy",
            conversion_required=True,
            message=(
                "Legacy filesystem run manifests detected. This format cannot be "
                "opened in-place by V1 runtimes without risking a blank store; "
                "convert it into a fresh workspace first."
            ),
            conversion_command=command,
        )

    return WorkspaceFormatInfo(root, "new-or-empty", False, "No existing nirs4all store was detected.")


def warn_if_legacy_workspace(path: Path | str) -> WorkspaceFormatInfo:
    """Warn with a concrete conversion command when *path* is a legacy workspace."""
    info = inspect_workspace_format(path)
    if info.conversion_required:
        command = info.conversion_command or build_conversion_command(info.path)
        warnings.warn(f"{info.message} Conversion command: {command}", RuntimeWarning, stacklevel=2)
    return info


__all__ = [
    "TARGET_WORKSPACE_FORMAT",
    "WorkspaceFormatInfo",
    "build_conversion_command",
    "default_conversion_output",
    "inspect_workspace_format",
    "warn_if_legacy_workspace",
]
