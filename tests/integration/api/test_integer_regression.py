"""DAG consumes actual explicitly declared regression measurements, not label codes."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


@pytest.mark.parametrize("with_cv", [False, True])
def test_integer_regression_keeps_measurements_metrics_and_replay(tmp_path, monkeypatch, with_cv):
    import nirs4all
    from nirs4all.data import DatasetConfigs

    rng = np.random.default_rng(631)
    X = rng.normal(size=(24, 4))
    y = np.arange(101., 125.)
    configs = DatasetConfigs({"train_x": X, "train_y": y}, task_type="regression")
    dataset = configs.get_dataset_at(0)
    np.testing.assert_array_equal(dataset.y({}).ravel(), y)
    pipeline = [KFold(3), Ridge()] if with_cv else [Ridge()]
    result = nirs4all.run(pipeline, dataset, save_artifacts=False)
    assert result.execution_engine == "dag-ml"
    assert all(row["task_type"] == "regression" for row in result.predictions.filter_predictions())
    assert all(row["metric"] == "rmse" for row in result.predictions.filter_predictions())
    final = result.predictions.filter_predictions(fold_id="final", partition="train", load_arrays=True)[0]
    np.testing.assert_array_equal(np.sort(final["y_true"].ravel()), y)
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("export/replay fitted"))
    archive = result.export(tmp_path / "regression.n4a")
    predicted = nirs4all.predict(archive, X).y_pred
    expected = result._dagml_refit_artifacts[0]["estimator"].predict(X.astype(np.float32))
    np.testing.assert_array_equal(predicted, expected)
    result.close()
