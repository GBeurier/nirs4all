"""A real DAG phase fits all train rows exactly once, never the held-out rows."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nirs4all.data import SpectroDataset
from nirs4all.pipeline.dagml.full_train import NoSplitEvaluationWarning, run_full_train


@pytest.mark.parametrize("has_test", [False, True])
@pytest.mark.parametrize("scale_target", [False, True])
def test_full_training_matches_one_sklearn_fit_and_retains_replay(has_test, scale_target, monkeypatch, tmp_path):
    from nirs4all.pipeline.dagml.native_results import read_native_results, write_native_results

    rng = np.random.default_rng(190)
    X = rng.normal(size=(24, 5))
    y = X @ np.arange(1.0, 6.0) + rng.normal(scale=0.1, size=24)
    dataset = SpectroDataset("no-split")
    dataset.add_samples(X[:18], {"partition": "train"})
    dataset.add_targets(y[:18])
    if has_test:
        dataset.add_samples(X[18:] + 20, {"partition": "test"})
        dataset.add_targets(y[18:])
    train_x = dataset.x({"partition": "train"}, layout="2d")
    train_y = np.asarray(dataset.y({"partition": "train"}), dtype=float).ravel()
    target_transform = StandardScaler().fit(train_y.reshape(-1, 1)) if scale_target else None
    fit_y = target_transform.transform(train_y.reshape(-1, 1)).ravel() if target_transform is not None else train_y
    expected = make_pipeline(StandardScaler(), Ridge(alpha=0.7)).fit(train_x, fit_y)
    fitted_rows = []
    original_fit = Ridge.fit

    def fit(model, fit_x, fit_targets, **kwargs):
        fitted_rows.append(len(fit_x))
        return original_fit(model, fit_x, fit_targets, **kwargs)

    monkeypatch.setattr(Ridge, "fit", fit)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("legacy execution"))
    pipeline = [StandardScaler()]
    if scale_target:
        pipeline.append({"y_processing": StandardScaler()})
    pipeline.append(Ridge(alpha=0.7))
    with pytest.warns(NoSplitEvaluationWarning, match="No splitter"):
        result = run_full_train(pipeline, dataset)
    assert fitted_rows == [18]
    assert np.isnan(result.cv_best_score)
    assert result._dagml_refit_artifacts
    for row in result.predictions.filter_predictions(load_arrays=True):
        partition = "test" if row["partition"] == "val" else row["partition"]
        predicted = np.asarray(expected.predict(dataset.x({"partition": partition}, layout="2d")), dtype=float)
        if target_transform is not None:
            predicted = target_transform.inverse_transform(predicted.reshape(-1, 1)).ravel()
        np.testing.assert_allclose(row["y_pred"], predicted, rtol=1e-10, atol=1e-10)
    assert {frame["lineage"]["phase"] for frame in result._dagml_node_results} == {"REFIT"}
    assert {report["partition"] for report in result._dagml_score_set["reports"]} <= {"final", "test"}
    assert all(report.get("fold_id") is None for report in result._dagml_score_set["reports"])
    result._dagml_results_dir = write_native_results(result, result._dagml_score_set, tmp_path)
    restored = read_native_results(result._dagml_results_dir)
    assert restored["score_set"] == result._dagml_score_set
    for row in restored["predictions"].filter_predictions():
        evaluation = row["result_metadata"]["evaluation"]
        assert evaluation["cross_validation"] is False
        assert evaluation["training_scope"] == "resubstitution"
        assert evaluation["test_used_for_validation"] is has_test
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("export retrained"))
    assert result.export_model(tmp_path / "full-train.joblib").is_file()


def test_full_training_classifier_uses_native_classification_scores():
    rng = np.random.default_rng(22)
    X = rng.normal(size=(30, 4))
    y = (X[:, 0] > 0).astype(int)
    dataset = SpectroDataset("no-split-classification")
    dataset.add_samples(X[:24], {"partition": "train"})
    dataset.add_targets(y[:24])
    dataset.add_samples(X[24:], {"partition": "test"})
    dataset.add_targets(y[24:])
    with pytest.warns(NoSplitEvaluationWarning):
        result = run_full_train([StandardScaler(), LogisticRegression()], dataset, metric="balanced_accuracy", task_type="classification")
    assert 0.0 <= result.best_score <= 1.0
    assert np.isnan(result.cv_best_score)
    assert "balanced_accuracy" in next(report for report in result._dagml_score_set["reports"] if report["partition"] == "test")["metrics"]


def test_full_training_preserves_storage_order_for_seeded_random_forest():
    rng = np.random.default_rng(39)
    X = rng.normal(size=(18, 4))
    y = X[:, 0] ** 2 + X[:, 1]
    dataset = SpectroDataset("no-split-order")
    dataset.add_samples(X, {"partition": "train"})
    dataset.add_targets(y)
    model = RandomForestRegressor(n_estimators=7, max_depth=3, random_state=25, n_jobs=1)
    expected = make_pipeline(StandardScaler(), model).fit(
        dataset.x({"partition": "train"}, layout="2d"), np.asarray(dataset.y({"partition": "train"}), dtype=float).ravel(),
    ).predict(dataset.x({"partition": "train"}, layout="2d"))
    with pytest.warns(NoSplitEvaluationWarning):
        result = run_full_train([StandardScaler(), model], dataset)
    row = result.predictions.filter_predictions(load_arrays=True)[0]
    assert row["sample_indices"] == list(range(18))
    np.testing.assert_array_equal(row["y_pred"], expected)
