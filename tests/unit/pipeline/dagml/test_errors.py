"""Failure messages retain the actionable cause from a DAG-ML subprocess."""

import pytest

from nirs4all.pipeline.dagml.errors import _cli_child_error, _raise_run_failure


def test_cli_failure_keeps_nested_native_and_adapter_causes() -> None:
    stdout = (
        "Error: process DSL CV+refit bundle capture failed\n\n"
        "Caused by:\n"
        "    0: FIT_CV execution failed\n"
        "    1: JAX model fit failed\n"
        "ValueError: low-level adapter cause\n"
    )
    outcome = {"returncode": 1, "stdout": stdout, "results": []}

    with pytest.raises(RuntimeError) as exc_info:
        _raise_run_failure(outcome, "dag-ml residual model run failed")

    message = str(exc_info.value)
    assert "rc=1" in message
    assert "Error: process DSL CV+refit bundle capture failed" in message
    assert "Caused by:\n    0: FIT_CV execution failed" in message
    assert "1: JAX model fit failed" in message
    assert "ValueError: low-level adapter cause" in message


def test_oversized_cli_output_keeps_first_and_last_causes() -> None:
    stdout = "Error: top-level failure\n" + "x" * 20000 + "\nCaused by: deepest cause"
    detail = _cli_child_error(stdout)

    assert detail.startswith("Error: top-level failure")
    assert "CLI output characters omitted" in detail
    assert detail.endswith("Caused by: deepest cause")
    assert len(detail) < len(stdout)
