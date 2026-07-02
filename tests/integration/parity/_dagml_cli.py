"""dag-ml-cli discovery shared by parity tests."""

from __future__ import annotations

import os
from pathlib import Path


def dagml_cli_path() -> Path:
    """Return the preferred dag-ml-cli candidate for the current checkout layout."""
    explicit = os.environ.get("N4A_DAGML_CLI")
    if explicit:
        return Path(explicit).expanduser()

    workspace = Path(__file__).resolve().parents[3].parent
    candidates = [
        workspace / "RC-v1-dagml" / "target" / "release" / "dag-ml-cli",
        workspace / "RC-v1-dagml" / "target" / "debug" / "dag-ml-cli",
        workspace / "dag-ml" / "target" / "release" / "dag-ml-cli",
        workspace / "dag-ml" / "target" / "debug" / "dag-ml-cli",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])
