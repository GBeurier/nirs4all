"""CONFORMANCE: the canonical user examples run on the explicit rollback lane.

The shipped ``examples/user/`` tutorials are the public-facing proof that
``nirs4all.run`` works end to end. These historical tutorials intentionally pin
``engine="legacy"`` while the native-first tutorials are developed separately,
so this test executes each script once and verifies that explicit rollback.

Each example is run as a SUBPROCESS (the scripts parse argv at import and import
matplotlib), with ``cwd=examples/`` (the scripts use dataset paths relative to
``examples/``), ``MPLBACKEND=Agg`` (headless), and the venv python. A non-zero
exit fails the test with the captured stderr tail. If a script needs an optional
dependency that is not installed (TF/keras/torch/jax), the run is SKIPPED, not
failed.

Slow: 4 examples. Gated by ``slow``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.parity, pytest.mark.slow]

# project root: parents[0]=parity, [1]=integration, [2]=tests, [3]=project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES_DIR = _PROJECT_ROOT / "examples"

# The 4 canonical examples (paths relative to examples/).
_EXAMPLES = (
    "user/01_getting_started/U01_hello_world.py",
    "user/01_getting_started/U02_basic_regression.py",
    "user/01_getting_started/U03_basic_classification.py",
    "user/03_preprocessing/U01_preprocessing_basics.py",
)

# Optional-dependency import errors that should SKIP rather than fail.
_OPTIONAL_DEP_MARKERS = (
    "No module named 'tensorflow'",
    "No module named 'keras'",
    "No module named 'torch'",
    "No module named 'jax'",
    "TensorFlow is not available",
    "JAX is not available",
    "PyTorch is not available",
)


@pytest.mark.parametrize("example", _EXAMPLES, ids=lambda p: Path(p).stem)
def test_example_runs_on_explicit_legacy_engine(example: str) -> None:
    """The explicitly legacy-pinned historical example exits cleanly."""
    script = _EXAMPLES_DIR / example
    assert script.exists(), f"canonical example missing: {script}"

    env = dict(os.environ)
    env.pop("N4A_ENGINE", None)
    env["MPLBACKEND"] = "Agg"  # headless: no interactive backend in CI
    env["PYTHONIOENCODING"] = "utf-8"
    pythonpath = [str(_PROJECT_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(_EXAMPLES_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if proc.returncode != 0:
        combined = proc.stderr + proc.stdout
        if any(marker in combined for marker in _OPTIONAL_DEP_MARKERS):
            pytest.skip(f"{example} on explicit legacy engine: optional dependency not installed")
        pytest.fail(
            f"{example} on explicit legacy engine exited {proc.returncode}:\n"
            f"--- stderr tail ---\n{proc.stderr[-2000:]}"
        )
