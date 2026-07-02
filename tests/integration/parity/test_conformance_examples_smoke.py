"""CONFORMANCE: the canonical user examples run or refuse as classified.

The shipped ``examples/user/`` tutorials are the public-facing proof that
``nirs4all.run`` works end to end. This pins that the four canonical examples
exit cleanly on the legacy engine. On the V1 dag-ml engine, examples whose
pipeline shapes are not yet covered must fail with an exact structured refusal
classification, not with an unclassified crash or silent legacy fallback.

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

_DAGML_REFUSAL_LEDGER: dict[str, dict[str, object]] = {
    "user/01_getting_started/U02_basic_regression.py": {
        "markers": (
            "[run/unsupported_shape]",
            "runs ONE model per pipeline",
            "3 top-level {'model': ...} steps",
        ),
        "unsupported_capability": "public example uses multiple top-level model steps instead of a dag-ml model sweep",
        "owner": "dag-ml native coverage",
        "rationale": "The example remains a public legacy tutorial, but dag-ml V1 must refuse the multi-model shape explicitly instead of silently refitting legacy.",
        "coverage_evidence": "tests/integration/parity/test_dagml_cli_runner.py::test_dagml_engine_coverage_boundary",
    },
    "user/01_getting_started/U03_basic_classification.py": {
        "marker_sets": (
            (
                "[run/unsupported_shape]",
                "runs ONE model per pipeline",
                "2 top-level {'model': ...} steps",
            ),
            (
                "[run/unsupported_shape]",
                "feature_augmentation",
                "followed by another X-side step",
                "3D data-plane",
            ),
        ),
        "unsupported_capability": "public example composition may include an optional second model, then uses feature_augmentation followed by a downstream X transform that requires the processing-axis/3D dataplane",
        "owner": "dag-ml native coverage",
        "rationale": (
            "Simple ShuffleSplit CV and classification model sweeps are covered natively; this ledger entry tracks "
            "the richer public-example composition until optional multi-model tutorial shapes and processing-axis "
            "feature_augmentation followed by downstream transforms are supported."
        ),
        "coverage_evidence": "tests/integration/parity/test_dagml_cli_runner.py::test_feature_augmentation_3d_shapes_fail_loud and tests/integration/parity/test_dagml_cli_runner.py::test_public_run_engine_dagml_classification_sweep_selects_balanced_accuracy_winner",
    },
}


def _refusal_marker_sets(entry: dict[str, object]) -> tuple[tuple[str, ...], ...]:
    markers = entry.get("markers")
    if markers is not None:
        assert isinstance(markers, tuple) and markers
        assert all(isinstance(marker, str) and marker for marker in markers)
        return (markers,)

    marker_sets = entry.get("marker_sets")
    assert isinstance(marker_sets, tuple) and marker_sets
    for marker_set in marker_sets:
        assert isinstance(marker_set, tuple) and marker_set
        assert all(isinstance(marker, str) and marker for marker in marker_set)
    return marker_sets


def test_dagml_example_refusal_ledger_is_explicit() -> None:
    """Every accepted dag-ml example refusal is ledgered as a named V1 coverage gap."""
    assert set(_DAGML_REFUSAL_LEDGER) < set(_EXAMPLES)
    for example, entry in _DAGML_REFUSAL_LEDGER.items():
        marker_sets = _refusal_marker_sets(entry)
        assert isinstance(entry["unsupported_capability"], str) and entry["unsupported_capability"]
        assert isinstance(entry["owner"], str) and entry["owner"]
        assert isinstance(entry["rationale"], str) and entry["rationale"]
        assert isinstance(entry["coverage_evidence"], str) and entry["coverage_evidence"]
        if any("ShuffleSplit" in " ".join(markers) for markers in marker_sets):
            assert "simple shufflesplit" in entry["rationale"].lower()


@pytest.mark.parametrize("engine", ["legacy", "dag-ml"])
@pytest.mark.parametrize("example", _EXAMPLES, ids=lambda p: Path(p).stem)
def test_example_runs_on_engine(example: str, engine: str) -> None:
    """The example exits 0, or dag-ml refuses with the exact documented classification."""
    script = _EXAMPLES_DIR / example
    assert script.exists(), f"canonical example missing: {script}"

    env = dict(os.environ)
    env["N4A_ENGINE"] = engine
    env["MPLBACKEND"] = "Agg"  # headless: no interactive backend in CI
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
            pytest.skip(f"{example} on engine={engine}: optional dependency not installed")
        refusal = _DAGML_REFUSAL_LEDGER.get(example) if engine == "dag-ml" else None
        if refusal is not None:
            if any(all(marker in combined for marker in markers) for markers in _refusal_marker_sets(refusal)):
                return
        pytest.fail(
            f"{example} on engine={engine} exited {proc.returncode}:\n"
            f"--- stderr tail ---\n{proc.stderr[-2000:]}"
        )
