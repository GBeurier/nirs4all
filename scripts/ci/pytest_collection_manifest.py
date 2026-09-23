"""Record collected node IDs before a per-file pytest process runs tests."""

import json
import os
from pathlib import Path


def pytest_collection_finish(session):
    """Persist collection before native test code can crash the process."""
    manifest = os.environ.get("N4A_PYTEST_MANIFEST")
    if manifest:
        Path(manifest).write_text(
            json.dumps([item.nodeid for item in session.items]), encoding="utf-8"
        )
