"""Tests for the module-level explain API."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from nirs4all.api.explain import explain, explain_preflight
from nirs4all.pipeline.dagml.rt import RtError


class _FakeRunner:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def explain(self, **kwargs):
        return (
            {
                "shap_values": np.array([[0.1, 0.2]]),
                "feature_names": ["MIR:1000", "NIRS:1000"],
                "expected_value": 0.0,
                "explainer_type": "kernel",
                "explanation_level": "source_aggregate",
                "feature_lineage": {
                    "MIR:1000": {
                        "source_id": "MIR",
                        "representation": "per_source_aggregate",
                    }
                },
                "lineage_warning": "Explained features are per-source aggregates.",
            },
            str(self.output_dir),
        )


class _FakeSession:
    def __init__(self, output_dir: Path):
        self.runner = _FakeRunner(output_dir)


def test_explain_preserves_relation_lineage_from_runner(tmp_path: Path) -> None:
    """The public API forwards relation explanation metadata into ExplainResult."""
    result = explain(
        {"model_name": "PLS"},
        np.array([[1.0, 2.0]]),
        session=_FakeSession(tmp_path),
        plots_visible=False,
        engine="legacy",
    )

    assert result.explanation_level == "source_aggregate"
    assert result.feature_names == ["MIR:1000", "NIRS:1000"]
    assert result.get_feature_lineage("MIR:1000")["source_id"] == "MIR"
    assert result.lineage_warning == "Explained features are per-source aggregates."


def test_explain_native_default_refuses_before_runner_or_session_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default profile never reaches the Python SHAP runner or user data."""
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.delenv("N4A_EXPLAIN_PLUGIN", raising=False)
    explain_module = importlib.import_module("nirs4all.api.explain")

    class _RunnerMustNotBeConstructed:
        def __init__(self, **kwargs):
            raise AssertionError("native explain refusal constructed PipelineRunner")

    class _SessionMustNotBeRead:
        @property
        def runner(self):
            raise AssertionError("native explain refusal read Session.runner")

    monkeypatch.setattr(explain_module, "PipelineRunner", _RunnerMustNotBeConstructed)

    with pytest.raises(RtError) as caught:
        explain(
            "missing-model.n4a",
            {"X": object()},
            session=_SessionMustNotBeRead(),
            plots_visible=False,
        )

    assert caught.value.to_dict() == {
        "verb": "explain",
        "cause": "unsupported_capability",
        "message": "engine='dag-ml' has no callable explain contract in the installed nirs4all Core/DAG-ML adapter",
        "mitigation": "select a supported explain plugin when one is installed, or select engine='legacy' explicitly",
        "unsupported_capability": "native_explain",
    }


def test_explain_plugin_and_fallback_selections_fail_closed() -> None:
    """Plugin absence and fallback permission are explicit typed refusals."""
    with pytest.raises(RtError) as plugin_error:
        explain("missing.n4a", np.empty((0, 0)), plugin="shap-provider")
    assert plugin_error.value.unsupported_capability == "explain_plugin"

    with pytest.raises(RtError) as fallback_error:
        explain(
            "missing.n4a",
            np.empty((0, 0)),
            engine="legacy",
            allow_fallback=True,
        )
    assert fallback_error.value.unsupported_capability == "implicit_legacy_fallback"


def test_explain_preflight_is_side_effect_free_and_reports_legacy_contract() -> None:
    """Callers can inspect the decision before providing model or data."""
    import nirs4all

    assert nirs4all.explain.preflight is explain_preflight
    native = explain_preflight(engine="native")
    assert native.executable is False
    assert native.lane == "native"
    assert native.contract is None

    legacy = explain_preflight(engine="legacy")
    assert legacy.executable is True
    assert legacy.lane == "legacy"
    assert legacy.contract == "nirs4all.python.pipeline_runner.shap"
