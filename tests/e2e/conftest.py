from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--artifacts-dir",
        action="store",
        default=None,
        help="directory where cross-repository e2e artifacts are written",
    )


@pytest.fixture()
def artifacts_dir(pytestconfig: pytest.Config, tmp_path: Path) -> Path:
    configured = pytestconfig.getoption("--artifacts-dir") or os.environ.get("N4A_E2E_ARTIFACTS_DIR")
    root = Path(configured) if configured else tmp_path / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root
