"""SHAP explains the captured predictor in original units without retraining."""

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nirs4all import explain
from nirs4all.pipeline.dagml.general_archive import load_general_archive


@pytest.mark.parametrize("kind", ["regression", "multi_target", "classification"])
def test_captured_shap_additivity_and_no_predictor_fit(kind, tmp_path, monkeypatch):
    import nirs4all

    rng = np.random.default_rng(192)
    X = rng.normal(size=(24, 4)).astype(np.float32)
    y = X @ np.array([1.0, -2.0, 3.0, 0.5])
    pipeline = [StandardScaler()]
    output_index = 0
    if kind == "classification":
        y = (y > 0).astype(int)
        pipeline.append(LogisticRegression())
        output_index = 1
    else:
        if kind == "multi_target":
            y = np.column_stack([y, -2 * y + 3])
            output_index = 1
        pipeline.extend([{"y_processing": StandardScaler()}, Ridge()])
    result = nirs4all.run(pipeline, (X, y), verbose=0, save_charts=False, workspace_path=tmp_path / "workspace")
    try:
        archive = result.export(tmp_path / "model.n4a")
        captured = load_general_archive(archive)["artifact"]["estimator"]
        expected = captured.estimator.predict_proba(X[:4])[:, output_index] if kind == "classification" else np.asarray(captured.predict(X[:4])).reshape(4, -1)[:, output_index]
        monkeypatch.setattr(Pipeline, "fit", lambda *args, **kwargs: pytest.fail("predictor retrained"))
        monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.__init__", lambda *args, **kwargs: pytest.fail("legacy runner created"))
        explanation = explain(
            archive, X[:4], name="samples", session=None, verbose=0, plots_visible=False,
            n_samples=4, explainer_type="auto", visualizations=[], output_index=output_index,
        )
        np.testing.assert_allclose(explanation.values.sum(axis=1) + explanation.base_value, expected, rtol=1e-5, atol=1e-5)
        assert explanation.shape == (4, 4)
        assert explanation.explanation_level == "raw_observation"
        assert f"output {output_index}" in explanation.lineage_warning
    finally:
        result.close()


def test_general_explain_best_dict_reopens_workspace_without_training(tmp_path, monkeypatch):
    import nirs4all

    X = np.random.default_rng(28).normal(size=(24, 3)).astype(np.float32)
    result = nirs4all.run(
        [StandardScaler(), {"y_processing": StandardScaler()}, Ridge()], (X, 2 * X[:, 0] + 12),
        save_charts=False, verbose=0, workspace_path=tmp_path / "workspace",
    )
    selected = result.best
    expected = nirs4all.predict(selected, X[:3]).y_pred
    result.close()
    monkeypatch.setattr(Pipeline, "fit", lambda *args, **kwargs: pytest.fail("predictor retrained"))
    output = tmp_path / "explanation"
    explanation = explain(selected, X[:3], verbose=0, plots_visible=False, n_samples=3, visualizations=[], output_dir=output)
    np.testing.assert_allclose(explanation.values.sum(axis=1) + explanation.base_value, expected, rtol=1e-5, atol=1e-5)
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["source_type"] == "workspace"
    assert provenance["artifact_scope"] == "full_training_refit"
    assert "archive_fingerprint" not in provenance


def test_general_explain_preflight_selects_callable_builtin(monkeypatch):
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.delenv("N4A_EXPLAIN_PLUGIN", raising=False)
    decision = explain.preflight()
    assert decision.executable and decision.lane == "plugin"
    assert decision.contract == "nirs4all.python.shap.v1"


@pytest.mark.parametrize("relation_level", [None, "per_source_aggregate", "sample_aggregate"])
def test_general_explain_preserves_headers_and_relation_level(tmp_path, relation_level):
    import nirs4all
    from nirs4all.data.dataset import SpectroDataset

    X = np.random.default_rng(72).normal(size=(20, 3)).astype(np.float32)
    result = nirs4all.run([Ridge()], (X, X[:, 0]), save_charts=False, verbose=0, workspace_path=tmp_path / "workspace")
    dataset = SpectroDataset("held-out")
    names = ["MIR:1000", "MIR:1100", "MIR:1200"]
    dataset.add_samples(X[:3], headers=names, header_unit="text", indexes={"partition": "test"})
    if relation_level is not None:
        dataset._relation_materialization_manifest = {
            "representation": relation_level, "headers": names, "shape": [3, 3],
            "source_ids": ["MIR"], "fingerprint": "recorded-materialization",
        }
    try:
        explained = explain(result, dataset, visualizations=[], plots_visible=False, verbose=0, n_samples=3)
        assert explained.feature_names == names
        expected_level = {None: "raw_observation", "per_source_aggregate": "source_aggregate", "sample_aggregate": "sample_aggregate"}[relation_level]
        assert explained.explanation_level == expected_level
        if relation_level is not None:
            assert explained.feature_lineage[names[0]]["source_id"] == "MIR"
            assert explained.feature_lineage[names[0]]["materialization_fingerprint"] == "recorded-materialization"
    finally:
        result.close()


def test_general_explain_session_rejects_replaced_archive_before_pickle(tmp_path, monkeypatch):
    import joblib

    import nirs4all

    X = np.random.default_rng(13).normal(size=(20, 3)).astype(np.float32)
    result = nirs4all.run([Ridge()], (X, X[:, 0]), save_charts=False, verbose=0, workspace_path=tmp_path / "workspace")
    try:
        archive = result.export(tmp_path / "model.n4a")
        with nirs4all.load_session(archive) as session:
            explained = explain(archive, X[:2], session=session, n_samples=3, visualizations=[], plots_visible=False, verbose=0)
            assert explained.shape == (2, 3)
            archive.write_bytes(b"replaced archive")
            monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("changed archive was deserialized"))
            with pytest.raises(ValueError, match="source archive changed"):
                explain(archive, X[:2], session=session, plots_visible=False)
    finally:
        result.close()


def test_general_explain_scientific_error_never_retries_legacy(tmp_path, monkeypatch):
    import nirs4all
    from nirs4all.visualization.analysis.shap import ShapAnalyzer

    X = np.random.default_rng(41).normal(size=(20, 3)).astype(np.float32)
    result = nirs4all.run([Ridge()], (X, X[:, 0]), save_charts=False, verbose=0, workspace_path=tmp_path / "workspace")
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("scientific SHAP failure")

    monkeypatch.setattr(ShapAnalyzer, "explain_model", fail_once)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.__init__", lambda *args, **kwargs: pytest.fail("legacy retry"))
    try:
        with pytest.raises(RuntimeError, match="scientific SHAP failure"):
            explain(result, X[:2], plots_visible=False, visualizations=[])
        assert calls == [1]
    finally:
        result.close()


@pytest.mark.parametrize("visualization_options", [{"visualizations": ["summary"]}, {}])
def test_general_explain_writes_table_and_provenance_before_optional_plot(tmp_path, monkeypatch, visualization_options):
    import nirs4all

    rng = np.random.default_rng(10)
    X = rng.normal(size=(18, 3)).astype(np.float32)
    result = nirs4all.run([Ridge()], (X, X[:, 0]), save_charts=False, verbose=0, workspace_path=tmp_path / "workspace")
    output = tmp_path / "explanation"
    from nirs4all.visualization.analysis.shap import ShapAnalyzer
    original_plot = ShapAnalyzer.plot_summary

    def checked_plot(self, **kwargs):
        assert (output / "summary.html").is_file()
        assert (output / "shap_values.csv").is_file()
        return original_plot(self, **kwargs)

    monkeypatch.setattr(ShapAnalyzer, "plot_summary", checked_plot)
    try:
        explanation = explain(
            result, X[:3], name="samples", session=None, verbose=0, plots_visible=False,
            n_samples=3, explainer_type="auto", output_dir=output,
            feature_names=["a, quoted", "b", "c"], **visualization_options,
        )
        assert explanation.visualizations["summary"].is_file()
        assert "<table" in explanation.visualizations["summary_table"].read_text()
        provenance = json.loads((output / "provenance.json").read_text())
        assert provenance["training_performed"] is False
        assert provenance["artifact_integrity_verified"] is True
        import pandas as pd

        assert pd.read_csv(output / "input_features.csv").columns.tolist() == ["a, quoted", "b", "c"]
    finally:
        result.close()
