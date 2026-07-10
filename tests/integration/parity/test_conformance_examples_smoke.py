"""CONFORMANCE: the canonical user examples run on BOTH engines.

The shipped ``examples/user/`` tutorials are the public-facing proof that
``nirs4all.run`` works end to end. This pins that the four canonical examples
exit cleanly on the legacy engine AND the dag-ml engine — the engine selector is
honored via ``$N4A_ENGINE`` (precedence: explicit arg > env > default), so a
plain example script picks up the engine under test without code changes.

Each example is run as a SUBPROCESS (the scripts parse argv at import and import
matplotlib), with ``cwd=examples/`` (the scripts use dataset paths relative to
``examples/``), ``MPLBACKEND=Agg`` (headless), and the venv python. A non-zero
exit fails the test with the captured stderr tail. If a script needs an optional
dependency that is not installed (TF/keras/torch/jax), the run is SKIPPED, not
failed.

Slow: 4 examples × 2 engines. Gated by ``slow``.
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


@pytest.mark.parametrize("engine", ["legacy", "dag-ml"])
@pytest.mark.parametrize("example", _EXAMPLES, ids=lambda p: Path(p).stem)
def test_example_runs_on_engine(example: str, engine: str) -> None:
    """The example script exits 0 under ``engine`` (skips on a missing optional dep)."""
    script = _EXAMPLES_DIR / example
    assert script.exists(), f"canonical example missing: {script}"

    env = dict(os.environ)
    env["N4A_ENGINE"] = engine
    env["MPLBACKEND"] = "Agg"  # headless: no interactive backend in CI
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(_PROJECT_ROOT) + (os.pathsep + current_pythonpath if current_pythonpath else "")

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
            pytest.skip(f"{example} on engine={engine}: optional dependency not installed")
        pytest.fail(
            f"{example} on engine={engine} exited {proc.returncode}:\n"
            f"--- stderr tail ---\n{proc.stderr[-2000:]}"
        )
