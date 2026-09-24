"""Public legacy/DAG-ML parity for transfer-guided preprocessing."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.analysis import get_base_preprocessings
from nirs4all.data.config import DatasetConfigs
from nirs4all.operators.transforms.scalers import StandardNormalVariate
from nirs4all.pipeline.dagml.full_train import NoSplitEvaluationWarning

from ._dagml_cli import dagml_cli_path
from ._datasets import dataset_path


def _test_prediction(result) -> np.ndarray:
    rows = [row for row in result.predictions.filter_predictions(load_arrays=True) if row["partition"] == "test"]
    final = [row for row in rows if row.get("fold_id") == "final"]
    rows = final or rows
    assert len(rows) == 1
    return np.asarray(rows[0]["y_pred"]).ravel()


@pytest.mark.parity
@pytest.mark.parametrize("apply_recommendation", [False, True])
def test_auto_transfer_preproc_no_cv_matches_legacy_direct_model_and_archive(
    tmp_path, monkeypatch, apply_recommendation: bool,
) -> None:
    cli = dagml_cli_path()
    if not cli.exists():
        pytest.skip(f"dag-ml-cli binary not built at {cli}")
    path = dataset_path("regression")
    dataset = DatasetConfigs(path).get_dataset_at(0)
    train_x = np.asarray(dataset.x({"partition": "train"}, layout="2d"))
    train_y = np.asarray(dataset.y({"partition": "train"}))
    test_x = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
    test_y = np.asarray(dataset.y({"partition": "test"})).ravel()
    pipeline = [
        {"auto_transfer_preproc": {
            "preset": "fast", "apply_recommendation": apply_recommendation, "verbose": 0,
        }},
        PLSRegression(3),
    ]
    legacy = nirs4all.run(pipeline, path, engine="legacy", save_artifacts=False, verbose=0)
    legacy_pred = _test_prediction(legacy)
    if apply_recommendation:
        selected = get_base_preprocessings()["area_norm"]
        selected.fit(train_x)
        train_x = selected.transform(train_x)
        test_x_direct = selected.transform(test_x)
    else:
        test_x_direct = test_x
    direct = np.asarray(PLSRegression(3).fit(train_x, train_y).predict(test_x_direct)).ravel()
    np.testing.assert_allclose(legacy_pred, direct, atol=1e-6)

    for mode in ("1", "0"):
        monkeypatch.setenv("N4A_DAGML_INPROCESS", mode)
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
        with pytest.warns(NoSplitEvaluationWarning, match="No splitter"):
            native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)
        np.testing.assert_allclose(_test_prediction(native), direct, atol=1e-6)
        assert native.best_rmse == pytest.approx(float(np.sqrt(np.mean((direct - test_y) ** 2))), abs=1e-6)
        assert any(
            "allow_fit_cv_all_observations_view" in frame["lineage"]["unsafe_flags"]
            for frame in native._dagml_node_results
        )
        archive = native.export(tmp_path / f"auto_transfer_{apply_recommendation}_{mode}.n4a")
        np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, test_x).y_pred).ravel(), direct, atol=1e-6)


@pytest.mark.parity
def test_auto_transfer_preproc_cv_matches_legacy_inprocess_and_cli(tmp_path, monkeypatch) -> None:
    cli = dagml_cli_path()
    if not cli.exists():
        pytest.skip(f"dag-ml-cli binary not built at {cli}")
    path = dataset_path("regression")
    pipeline = [
        {"auto_transfer_preproc": {"preset": "fast", "apply_recommendation": True, "verbose": 0}},
        KFold(2),
        PLSRegression(3),
    ]
    legacy = nirs4all.run(pipeline, path, engine="legacy", save_artifacts=False, verbose=0)
    for mode in ("1", "0"):
        monkeypatch.setenv("N4A_DAGML_INPROCESS", mode)
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
        native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)
        assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-6)
        np.testing.assert_allclose(_test_prediction(native), _test_prediction(legacy), atol=1e-6)
        archive = native.export(tmp_path / f"auto_transfer_cv_{mode}.n4a")
        x_test = DatasetConfigs(path).get_dataset_at(0).x({"partition": "test"}, layout="2d")
        np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel(), _test_prediction(legacy), atol=1e-6)


@pytest.mark.parity
@pytest.mark.parametrize("use_augmentation", [False, True])
def test_auto_transfer_top_k_replays_legacy_selection_and_archive(tmp_path, monkeypatch, use_augmentation: bool) -> None:
    cli = dagml_cli_path()
    if not cli.exists():
        pytest.skip(f"dag-ml-cli binary not built at {cli}")
    path = dataset_path("regression")
    pipeline = [
        {"auto_transfer_preproc": {
            "preset": "fast", "top_k": 2, "use_augmentation": use_augmentation,
            "apply_recommendation": True, "verbose": 0,
        }},
        PLSRegression(3),
    ]
    legacy = nirs4all.run(pipeline, path, engine="legacy", save_artifacts=False, verbose=0)
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0")
    monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    with pytest.warns(NoSplitEvaluationWarning, match="No splitter"):
        native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)
    np.testing.assert_allclose(_test_prediction(native), _test_prediction(legacy), atol=1e-6)
    archive = native.export(tmp_path / f"auto_transfer_top_k_{use_augmentation}.n4a")
    x_test = DatasetConfigs(path).get_dataset_at(0).x({"partition": "test"}, layout="2d")
    np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel(), _test_prediction(legacy), atol=1e-6)


@pytest.mark.parity
def test_auto_transfer_multi_source_selects_jointly_and_replays_source_local_transforms(tmp_path, monkeypatch) -> None:
    cli = dagml_cli_path()
    if not cli.exists():
        pytest.skip(f"dag-ml-cli binary not built at {cli}")
    path = dataset_path("multi")
    pipeline = [
        {"auto_transfer_preproc": {"preset": "fast", "apply_recommendation": True, "verbose": 0}},
        PLSRegression(3),
    ]
    legacy = nirs4all.run(pipeline, path, engine="legacy", save_artifacts=False, verbose=0)
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0")
    monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    with pytest.warns(NoSplitEvaluationWarning, match="No splitter"):
        native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)
    np.testing.assert_allclose(_test_prediction(native), _test_prediction(legacy), atol=1e-6)
    archive = native.export(tmp_path / "auto_transfer_multi_source.n4a")
    x_test = DatasetConfigs(path).get_dataset_at(0).x({"partition": "test"}, layout="2d")
    np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel(), _test_prediction(legacy), atol=1e-6)


@pytest.mark.parity
@pytest.mark.parametrize("top_k", [1, 2])
def test_feature_augmentation_before_auto_transfer_preserves_processing_lanes(tmp_path, monkeypatch, top_k: int) -> None:
    cli = dagml_cli_path()
    if not cli.exists():
        pytest.skip(f"dag-ml-cli binary not built at {cli}")
    path = dataset_path("regression")
    pipeline = [
        {"feature_augmentation": [StandardNormalVariate()]},
        {"auto_transfer_preproc": {
            "preset": "fast", "apply_recommendation": True, "top_k": top_k,
            "use_augmentation": top_k > 1, "verbose": 0,
        }},
        PLSRegression(3),
    ]
    legacy = nirs4all.run(pipeline, path, engine="legacy", save_artifacts=False, verbose=0)
    test_x = DatasetConfigs(path).get_dataset_at(0).x({"partition": "test"}, layout="2d")
    for mode in ("1", "0"):
        monkeypatch.setenv("N4A_DAGML_INPROCESS", mode)
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
        with pytest.warns(NoSplitEvaluationWarning, match="No splitter"):
            native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)
        np.testing.assert_allclose(_test_prediction(native), _test_prediction(legacy), atol=1e-6)
        archive = native.export(tmp_path / f"transfer_after_feature_augmentation_{top_k}_{mode}.n4a")
        np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, test_x).y_pred).ravel(), _test_prediction(legacy), atol=1e-6)
