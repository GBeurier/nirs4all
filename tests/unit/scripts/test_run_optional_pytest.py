"""Tests for the optional pytest CI slice runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_runner() -> ModuleType:
    script_path = Path(__file__).parents[3] / "scripts" / "ci" / "run_optional_pytest.py"
    spec = importlib.util.spec_from_file_location("run_optional_pytest", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "exit_code",
    [
        pytest.ExitCode.TESTS_FAILED,
        pytest.ExitCode.INTERRUPTED,
        pytest.ExitCode.INTERNAL_ERROR,
        pytest.ExitCode.USAGE_ERROR,
    ],
)
def test_normalize_exit_code_preserves_failures(exit_code: pytest.ExitCode) -> None:
    runner = _load_runner()

    assert runner.normalize_exit_code(exit_code) == int(exit_code)


def test_normalize_exit_code_accepts_only_an_empty_slice(capsys: pytest.CaptureFixture[str]) -> None:
    runner = _load_runner()

    assert runner.normalize_exit_code(pytest.ExitCode.NO_TESTS_COLLECTED) == 0
    assert "selected no tests" in capsys.readouterr().out


def test_main_forwards_arguments_and_preserves_selected_test_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner()
    received: list[list[str] | None] = []

    def fake_pytest_main(args: list[str] | None) -> pytest.ExitCode:
        received.append(args)
        return pytest.ExitCode.TESTS_FAILED

    monkeypatch.setattr(runner.pytest, "main", fake_pytest_main)

    assert runner.main(["-m", "torch", "tests/unit/"]) == int(pytest.ExitCode.TESTS_FAILED)
    assert received == [["-m", "torch", "tests/unit/"]]
