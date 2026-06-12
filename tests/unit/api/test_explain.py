"""Tests for the module-level explain API."""

from pathlib import Path

import numpy as np

from nirs4all.api.explain import explain


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
    )

    assert result.explanation_level == "source_aggregate"
    assert result.feature_names == ["MIR:1000", "NIRS:1000"]
    assert result.get_feature_lineage("MIR:1000")["source_id"] == "MIR"
    assert result.lineage_warning == "Explained features are per-source aggregates."
