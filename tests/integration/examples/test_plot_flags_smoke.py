"""Smoke tests for example plot flag semantics."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("flag", ["--plots", "--show"])
def test_example_plot_flags_generate_charts(flag: str, tmp_path: Path) -> None:
    """User examples should generate charts for both save and show requests."""
    repo_root = Path(__file__).resolve().parents[3]
    examples_dir = repo_root / "examples"
    launcher = examples_dir / "ci_example_launcher.py"
    example = Path("user/01_getting_started/U02_basic_regression.py")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["NIRS4ALL_EXAMPLE_FAST"] = "1"
    env["NIRS4ALL_WORKSPACE"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, str(launcher), str(example), flag],
        cwd=examples_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert list((tmp_path / "figures").glob("*.png"))
