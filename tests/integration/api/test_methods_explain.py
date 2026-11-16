"""The Methods explain backend over captured general predictors."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nirs4all import explain
from nirs4all.operators.transforms import SNV, SavitzkyGolay
from nirs4all.pipeline.dagml.general_archive import load_general_archive

pytestmark = [pytest.mark.methods, pytest.mark.sklearn]
pytest.importorskip("n4m.explain")


def _spectra(rows: int = 30, width: int = 48, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 1.0, width)
    peaks = rng.normal(size=(rows, 2))
    X = 1.0 + peaks[:, :1] * np.exp(-((axis - 0.3) / 0.05) ** 2) + peaks[:, 1:] * np.exp(-((axis - 0.7) / 0.08) ** 2) + 0.01 * rng.normal(size=(rows, width))
    return X.astype(np.float32).astype(np.float64), 3.0 * peaks[:, 0] - peaks[:, 1]


def _train(pipeline, X, y, tmp_path):
    import nirs4all

    result = nirs4all.run(pipeline, (X, y), verbose=0, save_charts=False, workspace_path=tmp_path / "workspace")
    archive = result.export(tmp_path / "model.n4a")
    result.close()
    return archive, load_general_archive(archive)["artifact"]["estimator"]


def _terminal(captured):
    inner = captured.estimator
    return inner.steps[-1][1] if isinstance(inner, Pipeline) else inner


def _validate_envelope(envelope: dict, n_rows: int, n_groups: int) -> None:
    assert envelope["schema"] == "nirs4all.explanation.v1"
    assert envelope["rows"]["count"] == n_rows == len(envelope["rows"]["sample_indices"])
    assert len(envelope["groups"]) == n_groups
    assert np.asarray(envelope["attributions"]).shape[:2] == (n_rows, n_groups)
    assert np.asarray(envelope["expected_value"]).shape[0] == n_rows
    assert len(envelope["axis"]["values"]) == len(envelope["feature_names"])
    assert envelope["axis"]["unit"] is None or envelope["axis"]["kind"] in {"wavelength", "wavenumber"}
    schema_path = os.environ.get("NIRS4ALL_EXPLANATION_V1_SCHEMA")
    if schema_path:
        import jsonschema

        jsonschema.validate(envelope, json.loads(Path(schema_path).read_text(encoding="utf-8")))


def test_grouped_permutation_explains_complete_preprocessing_pipeline(tmp_path):
    X, y = _spectra()
    archive, captured = _train([SNV(), SavitzkyGolay(window_length=7, polyorder=2), PLSRegression(n_components=3)], X, y, tmp_path)
    options = {"explainer_backend": "methods", "explainer_type": "permutation", "group_width": 8, "n_permutations": 4, "seed": 1, "n_samples": 10}
    output = tmp_path / "explanation"
    first = explain(archive, X[:6], verbose=0, plots_visible=False, visualizations=["summary"], output_dir=output, **options)
    second = explain(archive, X[:6], verbose=0, plots_visible=False, visualizations=[], **options)

    expected = np.asarray(captured.predict(X[:6]), dtype=float).reshape(6, -1)[:, 0]
    np.testing.assert_allclose(first.values.sum(axis=1) + first.base_value, expected, rtol=1e-8, atol=1e-8)
    np.testing.assert_array_equal(first.values, second.values)
    assert first.shape == (6, 6)
    assert first.explainer_type == "permutation_shap"
    envelope = first.to_explanation_v1()
    assert first.feature_names[0] == f"{envelope['feature_names'][0]}–{envelope['feature_names'][7]}"
    _validate_envelope(envelope, 6, 6)
    assert envelope["engine"]["name"] == "nirs4all-methods"
    assert (envelope["group_aggregation"], envelope["attribution_space"], envelope["exactness"]) == ("group_game", "raw_input", "sampled")
    # Background rows are drawn from the supplied rows: min(n_samples, 6).
    assert envelope["settings"]["n_background"] == 6 and envelope["settings"]["seed"] == 1
    assert json.loads((output / "explanation_v1.json").read_text()) == envelope
    assert first.visualizations["summary"].is_file()
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["explainer_backend"] == "methods"

    auto = explain(archive, X[:3], verbose=0, plots_visible=False, visualizations=[], explainer_backend="methods", n_samples=5)
    assert auto.explainer_type == "permutation_shap"
    assert auto.shape == (3, 3)  # 48 features exceed the default width of 20.
    with pytest.raises(ValueError, match="applies preprocessing before its estimator"):
        explain(archive, X[:3], verbose=0, plots_visible=False, visualizations=[], explainer_backend="methods", explainer_type="linear")


def test_transformed_features_linear_shap_explains_terminal_estimator(tmp_path):
    X, y = _spectra()
    archive, captured = _train([StandardScaler(), Ridge(alpha=1.0)], X, y, tmp_path)
    explained = explain(archive, X[:5], verbose=0, plots_visible=False, visualizations=[], explainer_backend="methods",
                        explainer_type="linear", attribution_space="transformed_features")
    expected = np.asarray(captured.predict(X[:5]), dtype=float).reshape(5, -1)[:, 0]
    np.testing.assert_allclose(explained.values.sum(axis=1) + explained.base_value, expected, rtol=1e-9, atol=1e-9)
    assert explained.explainer_type == "linear_shap"
    assert explained.explanation_level == "transformed_feature"
    assert explained.feature_names[:2] == ["transformed_0", "transformed_1"]
    envelope = explained.to_explanation_v1()
    _validate_envelope(envelope, 5, X.shape[1])
    assert envelope["axis"]["kind"] == "feature_index" and envelope["axis"]["unit"] is None
    assert envelope["attribution_space"] == "transformed_features"
    assert "StandardScaler" in envelope["settings"]["preprocessing"]


def test_raw_linear_shap_folds_affine_target_inverse(tmp_path):
    X, y = _spectra(rows=24, width=12)
    archive, captured = _train([{"y_processing": StandardScaler()}, Ridge(alpha=0.3)], X, 10.0 * y + 50.0, tmp_path)
    explained = explain(archive, X[:4], verbose=0, plots_visible=False, visualizations=[], explainer_backend="methods", explainer_type="linear", n_samples=100)

    def predict(rows):
        return np.asarray(captured.predict(rows), dtype=float).reshape(len(rows), -1)[:, 0]

    origin = predict(X[:1])
    slopes = np.array([predict(X[:1] + np.eye(X.shape[1])[j])[0] - origin[0] for j in range(X.shape[1])])
    # n_samples exceeds the supplied rows, so the background is all of X[:4].
    np.testing.assert_allclose(explained.values, (X[:4] - X[:4].mean(axis=0)) * slopes, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(explained.values.sum(axis=1) + explained.base_value, predict(X[:4]), rtol=1e-9, atol=1e-8)
    assert explained.to_explanation_v1()["output"] == {"index": 0, "label": None, "space": "target"}


def test_tree_shap_matches_shap_and_keeps_source_row_indices(tmp_path):
    shap = pytest.importorskip("shap")
    X, y = _spectra(rows=40, width=10)
    archive, captured = _train([RandomForestRegressor(n_estimators=6, max_depth=4, random_state=0)], X, y, tmp_path)
    data = X[:10]
    selected = explain(archive, data, verbose=0, plots_visible=False, visualizations=[], explainer_backend="methods", sample_indices=[7, 2, 5])
    direct = explain(archive, data[[7, 2, 5]], verbose=0, plots_visible=False, visualizations=[], explainer_backend="methods", explainer_type="tree")

    assert selected.explainer_type == "tree_shap"  # auto selects the exact exporter
    np.testing.assert_allclose(selected.values, direct.values, rtol=1e-12, atol=1e-12)
    assert selected.to_explanation_v1()["rows"]["sample_indices"] == [7, 2, 5]
    assert direct.to_explanation_v1()["rows"]["sample_indices"] == [0, 1, 2]
    reference = shap.TreeExplainer(_terminal(captured), feature_perturbation="tree_path_dependent").shap_values(data[[7, 2, 5]].astype(np.float32))
    np.testing.assert_allclose(selected.values, reference, rtol=1e-6, atol=1e-9)
    with pytest.raises(ValueError, match="sample_indices must be integers"):
        explain(archive, data, verbose=0, plots_visible=False, visualizations=[], explainer_backend="methods", sample_indices=[10])


def test_tree_shap_explains_selected_class_probability(tmp_path):
    X, y = _spectra(rows=40, width=8)
    labels = (y > np.median(y)).astype(int)
    archive, captured = _train([RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)], X, labels, tmp_path)
    explained = explain(archive, X[:6], verbose=0, plots_visible=False, visualizations=[], explainer_backend="methods",
                        explainer_type="tree", output_index=1)
    probabilities = captured.estimator.predict_proba(X[:6])[:, 1]
    np.testing.assert_allclose(explained.values.sum(axis=1) + explained.base_value, probabilities, rtol=1e-9, atol=1e-9)
    envelope = explained.to_explanation_v1()
    assert envelope["output"]["space"] == "class_probability" and envelope["output"]["index"] == 1
    assert np.asarray(envelope["attributions"]).shape == (6, 8, 2)
    with pytest.raises(ValueError, match="class probabilities are not an affine"):
        explain(archive, X[:2], verbose=0, plots_visible=False, visualizations=[], explainer_backend="methods", explainer_type="linear")


def test_default_shap_backend_records_python_shap_envelope(tmp_path):
    shap = pytest.importorskip("shap")
    X, y = _spectra(rows=20, width=4)
    archive, _ = _train([Ridge()], X, y, tmp_path)
    explained = explain(archive, X[:6], verbose=0, plots_visible=False, visualizations=[], n_samples=5, sample_indices=[4, 1])
    envelope = explained.to_explanation_v1()
    _validate_envelope(envelope, 2, 4)
    assert envelope["engine"] == {"name": "python-shap", "version": shap.__version__}
    assert (envelope["kind"], envelope["exactness"], envelope["group_aggregation"]) == ("kernel_shap", "sampled", "none")
    assert envelope["rows"]["sample_indices"] == [4, 1]
    assert envelope["axis"]["kind"] == "feature_index"
    assert envelope["additivity_max_abs_error"] < 1e-6


def test_waterfall_figure_decomposes_the_prediction_with_disjoint_bins(tmp_path, monkeypatch):
    pytest.importorskip("shap")
    from nirs4all.visualization.analysis.shap import ShapAnalyzer

    X, y = _spectra(rows=24, width=30)
    archive, captured = _train([Ridge(alpha=0.3)], X, y, tmp_path)
    seen = []
    original = ShapAnalyzer.plot_waterfall_binned

    def checked(self, **kwargs):
        binned, _ = self._aggregate_shap_bins(self.shap_values[0], kwargs["feature_labels"])
        seen.append((self.bin_size, self.bin_stride, self.bin_aggregation, float(self.base_value + binned.sum())))
        return original(self, **kwargs)

    monkeypatch.setattr(ShapAnalyzer, "plot_waterfall_binned", checked)
    explain(archive, X[:3], verbose=0, plots_visible=False, explainer_backend="methods", explainer_type="linear",
            visualizations=["waterfall"], output_dir=tmp_path / "figures")
    prediction = float(np.asarray(captured.predict(X[:1]), dtype=float).reshape(-1)[0])
    size, stride, aggregation, end = seen[0]
    assert (stride, aggregation) == (size, "sum")
    assert end == pytest.approx(prediction, rel=1e-9, abs=1e-9)
