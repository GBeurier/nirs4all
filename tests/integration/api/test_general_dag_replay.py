"""Captured host estimators replay under PREDICT only, including unlabelled X."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from nirs4all.data import SpectroDataset
from nirs4all.pipeline.dagml.full_train import NoSplitEvaluationWarning, run_full_train
from nirs4all.pipeline.dagml.general_replay import predict_captured_artifact
from nirs4all.pipeline.dagml.identity import mint_identity
from nirs4all.pipeline.dagml.native_results import read_native_results, write_native_results


@pytest.mark.parametrize("target_width", [1, 2])
@pytest.mark.parametrize("scale_target", [False, True])
def test_captured_replay_uses_native_predict_without_labels_or_fit(target_width, scale_target, monkeypatch, tmp_path):
    rng = np.random.default_rng(71)
    X = rng.normal(size=(26, 4))
    y = np.column_stack((X[:, 0] + 2 * X[:, 1], X[:, 2] - X[:, 3]))[:, :target_width]
    training = SpectroDataset("captured-training")
    training.add_samples(X, {"partition": "train"})
    training.add_targets(y)
    pipeline = [StandardScaler()]
    if scale_target:
        pipeline.append({"y_processing": StandardScaler()})
    pipeline.append(Ridge(alpha=0.4))
    with pytest.warns(NoSplitEvaluationWarning):
        trained = run_full_train(pipeline, training)
    directory = write_native_results(trained, trained._dagml_score_set, tmp_path)
    # This reader verifies bytes before loading the captured estimator.
    artifact = read_native_results(directory)["artifacts"][0]
    prediction = SpectroDataset("captured-unlabelled")
    prediction.add_samples(rng.normal(size=(17, 4)) + 4, {"partition": "test"})
    expected = np.asarray(artifact["estimator"].predict(prediction.x({}, layout="2d")), dtype=float).reshape(17, target_width)
    if artifact["y_transform"] is not None:
        expected = artifact["y_transform"].inverse_transform(expected)
    input_ids = mint_identity(prediction).observation_ids()
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("prediction fitted a model"))
    monkeypatch.setattr(StandardScaler, "fit", lambda *args, **kwargs: pytest.fail("prediction fitted a transformer"))
    monkeypatch.setattr(prediction, "y", lambda *args, **kwargs: pytest.fail("prediction required labels"))
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("legacy execution"))
    names = ["y"] if target_width == 1 else ["target_a", "target_b"]
    # A training splitter in the saved topology must not create any folds or fit.
    values, evidence = predict_captured_artifact(artifact, prediction, pipeline=[KFold(3), *pipeline], target_names=names)
    # Callback replay may change array contiguity before sklearn reaches the
    # platform BLAS.  The 2e-6 bound covers observed arm64 last-bit variation
    # while the fit prohibition and native evidence below enforce semantics.
    np.testing.assert_allclose(values.reshape(17, target_width), expected, rtol=2e-6, atol=2e-6)
    assert evidence["phase"] == "PREDICT"
    assert evidence["training_performed"] is False
    assert evidence["sample_ids"] == input_ids
    assert evidence["predict_cohort"]["role"] == "inference"
    assert evidence["predict_cohort"].get("target_content_fingerprint") is None
    assert evidence["scores"] is None
    assert {node["lineage"]["phase"] for node in evidence["node_results"]} == {"PREDICT"}
    assert all(not node.get("regression_targets") for node in evidence["node_results"])
    assert all(not node["artifacts"] for node in evidence["node_results"])


def test_captured_replay_does_not_retry_a_prediction_failure(monkeypatch):
    dataset = SpectroDataset("predict-failure")
    dataset.add_samples(np.ones((3, 2)), {"partition": "test"})
    model = Ridge().fit(np.arange(8).reshape(4, 2), np.arange(4))
    calls = []

    def fail_prediction(*args, **kwargs):
        calls.append("predict")
        raise RuntimeError("sentinel prediction failure")

    monkeypatch.setattr(model, "predict", fail_prediction)
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("prediction retried fit"))
    with pytest.raises(Exception, match="sentinel prediction failure"):
        predict_captured_artifact({"estimator": model}, dataset, pipeline=[Ridge()])
    assert calls == ["predict"]
