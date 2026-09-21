"""Smoke tests for example plot flag semantics."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def _load_launcher():
    launcher = Path(__file__).resolve().parents[3] / "examples" / "ci_example_launcher.py"
    spec = spec_from_file_location("nirs4all_ci_example_launcher", launcher)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fast_mode_preserves_categorical_domains_used_by_conditions() -> None:
    """Runtime caps must not remove labels referenced by conditional axes."""
    launcher = _load_launcher()
    optimized = launcher._optimize_object(
        {
            "model_params": {
                "kernel": ["linear", "rbf", "poly"],
                "gamma": {"type": "float_log", "min": 1e-4, "max": 10.0, "when": {"kernel": ["rbf", "poly"]}},
                "degree": {"type": "int", "min": 2, "max": 4, "when": {"kernel": "poly"}},
            }
        }
    )

    assert optimized["model_params"]["kernel"] == ["linear", "rbf", "poly"]


def test_launcher_only_skips_tabpfn_specific_environment_errors() -> None:
    launcher = _load_launcher()

    TabPFNLicenseError = type("TabPFNLicenseError", (RuntimeError,), {"__module__": "tabpfn.errors"})
    wrapped = RuntimeError("pipeline failed")
    wrapped.__cause__ = TabPFNLicenseError("license required")

    assert "TabPFNLicenseError" in launcher._qualification_skip_reason(wrapped)
    assert launcher._qualification_skip_reason(RuntimeError("real bug")) is None

    DagMlRuntimeError = type("DagMlRuntimeError", (RuntimeError,), {"__module__": "_dag_ml"})
    native_wrapped = DagMlRuntimeError("python callback: TabPFN requires a one-time license acceptance to download model weights")
    assert "license acceptance required" in launcher._qualification_skip_reason(native_wrapped)
    assert launcher._qualification_skip_reason(DagMlRuntimeError("unrelated callback bug")) is None


def test_legacy_qualification_reports_declared_dag_only_provider_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_launcher()
    error = ValueError("DataProvider requires the general DAG-ML profile; use engine='dag-ml' or omit engine")

    monkeypatch.setenv("NIRS4ALL_EXAMPLE_ENGINE", "legacy")
    assert "DAG-ML-only" in launcher._qualification_skip_reason(error)

    monkeypatch.setenv("NIRS4ALL_EXAMPLE_ENGINE", "dag-ml")
    assert launcher._qualification_skip_reason(error) is None


def test_fast_launcher_can_force_backend_for_cross_engine_qualification(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_launcher()
    nirs4all = importlib.import_module("nirs4all")
    run_api = importlib.import_module("nirs4all.api.run")
    runner_module = importlib.import_module("nirs4all.pipeline.runner")
    captured: dict[str, object] = {}

    def fake_run(pipeline, dataset, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(run_api, "run", fake_run)
    monkeypatch.setattr(nirs4all, "run", fake_run)
    monkeypatch.setattr(runner_module.PipelineRunner, "__init__", runner_module.PipelineRunner.__init__)
    monkeypatch.setattr(runner_module.PipelineRunner, "run", runner_module.PipelineRunner.run)
    monkeypatch.setenv("NIRS4ALL_EXAMPLE_ENGINE", "legacy")

    launcher._patch_nirs4all_fast_mode()
    nirs4all.run([], {})
    assert captured["engine"] == "legacy"

    captured.clear()
    nirs4all.run([], {}, engine="dag-ml")
    assert captured["engine"] == "dag-ml"


def test_launcher_preserves_direct_script_sibling_imports(tmp_path: Path) -> None:
    """Launcher execution should expose the script directory on ``sys.path``."""
    repo_root = Path(__file__).resolve().parents[3]
    launcher = repo_root / "examples" / "ci_example_launcher.py"
    helper = tmp_path / "sibling_helper.py"
    example = tmp_path / "example.py"
    helper.write_text("VALUE = 'sibling-import-ok'\n", encoding="utf-8")
    example.write_text("from sibling_helper import VALUE\nprint(VALUE)\n", encoding="utf-8")

    env = os.environ.copy()
    env["NIRS4ALL_EXAMPLE_FAST"] = "0"
    result = subprocess.run(
        [sys.executable, str(launcher), str(example)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == "sibling-import-ok"


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
    env["N4A_ENGINE"] = "legacy"

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
