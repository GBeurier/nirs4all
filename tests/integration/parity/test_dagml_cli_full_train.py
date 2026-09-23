"""No-splitter REFIT parity across DAG-ML's Python and CLI mechanisms."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nirs4all.data.config import DatasetConfigs
from nirs4all.pipeline.dagml.full_train import NoSplitEvaluationWarning

from ._datasets import dataset_path


@pytest.mark.parity
def test_no_splitter_cli_matches_direct_train_only_refit_and_archive(tmp_path, monkeypatch) -> None:
    import nirs4all

    from ._dagml_cli import dagml_cli_path

    cli = dagml_cli_path()
    if not cli.exists():
        pytest.skip(f"dag-ml-cli binary not built at {cli}")
    path = dataset_path("regression")
    dataset = DatasetConfigs(path).get_dataset_at(0)
    train_x = dataset.x({"partition": "train"}, layout="2d")
    train_y = dataset.y({"partition": "train"})
    test_x = dataset.x({"partition": "test"}, layout="2d")
    test_y = np.asarray(dataset.y({"partition": "test"})).ravel()
    oracle = PLSRegression(n_components=5).fit(train_x, train_y)
    oracle_pred = np.asarray(oracle.predict(test_x)).ravel()
    oracle_rmse = float(np.sqrt(np.mean((test_y - oracle_pred) ** 2)))

    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0")
    monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    with pytest.warns(NoSplitEvaluationWarning, match="No splitter"):
        result = nirs4all.run([PLSRegression(n_components=5)], path, engine="dag-ml", save_artifacts=False)
    assert result.execution_engine == "dag-ml"
    assert np.isnan(result.cv_best_score)
    assert result.best_rmse == pytest.approx(oracle_rmse, abs=1e-5)
    assert {frame["lineage"]["phase"] for frame in result._dagml_node_results} == {"REFIT"}
    assert len(result._dagml_refit_artifacts) == 1
    np.testing.assert_allclose(
        np.asarray(result._dagml_refit_artifacts[0]["estimator"].predict(test_x)).ravel(),
        oracle_pred, atol=1e-5,
    )
    archive = result.export(tmp_path / "cli_full_train.n4a")
    np.testing.assert_allclose(
        np.asarray(nirs4all.predict(archive, test_x).y_pred).ravel(), oracle_pred, atol=1e-5,
    )


@pytest.mark.parity
def test_no_splitter_cli_in_memory_training_has_no_fabricated_validation(monkeypatch) -> None:
    import nirs4all

    from ._dagml_cli import dagml_cli_path

    cli = dagml_cli_path()
    if not cli.exists():
        pytest.skip(f"dag-ml-cli binary not built at {cli}")
    rng = np.random.default_rng(87)
    x = rng.normal(size=(18, 8))
    y = rng.normal(size=(18, 1))
    oracle = make_pipeline(StandardScaler(), Ridge(alpha=0.5)).fit(x, y)
    oracle_rmse = float(np.sqrt(np.mean((y.ravel() - np.asarray(oracle.predict(x)).ravel()) ** 2)))

    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0")
    monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    with pytest.warns(NoSplitEvaluationWarning, match="No splitter"):
        result = nirs4all.run([StandardScaler(), Ridge(alpha=0.5)], (x, y), engine="dag-ml", save_artifacts=False)
    assert np.isnan(result.cv_best_score)
    assert np.isnan(result.best_rmse)
    train_row = next(row for row in result.predictions.filter_predictions(load_arrays=True) if row["partition"] == "train")
    assert train_row["train_score"] == pytest.approx(oracle_rmse, abs=1e-6)
    assert result.per_dataset[next(iter(result.per_dataset))]["evaluation"]["validation_source"] is None
    assert {frame["lineage"]["phase"] for frame in result._dagml_node_results} == {"REFIT"}
