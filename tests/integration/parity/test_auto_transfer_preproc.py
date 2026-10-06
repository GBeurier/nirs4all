"""Public legacy/DAG-ML parity for transfer-guided preprocessing."""

from copy import deepcopy

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.analysis import TransferPreprocessingSelector, get_base_preprocessings
from nirs4all.data.config import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.transforms.scalers import StandardNormalVariate
from nirs4all.pipeline.dagml.full_train import NoSplitEvaluationWarning
from nirs4all.pipeline.dagml.identity import mint_identity

from ._dagml_cli import dagml_cli_path
from ._datasets import dataset_path


def _test_prediction(result) -> np.ndarray:
    rows = [row for row in result.predictions.filter_predictions(load_arrays=True) if row["partition"] == "test"]
    final = [row for row in rows if row.get("fold_id") == "final"]
    rows = final or rows
    assert len(rows) == 1
    return np.asarray(rows[0]["y_pred"]).ravel()


def _direct_transfer_oracles(dataset, *, apply_recommendation=True, top_k=1, use_augmentation=False):
    """Fit independent train-only models under each backend's transform batch contract."""
    ids = list(dataset.index_column("sample", {}))
    train_ids = list(dataset.index_column("sample", {"partition": "train"}))
    test_ids = list(dataset.index_column("sample", {"partition": "test"}))
    assert set(train_ids).isdisjoint(test_ids)
    rows = {sample: row for row, sample in enumerate(ids)}
    train_rows = [rows[sample] for sample in train_ids]
    test_rows = [rows[sample] for sample in test_ids]
    raw = np.asarray(dataset.x({}, layout="2d", include_augmented=False))
    train, test = raw[train_rows], raw[test_rows]
    y_train = np.asarray(dataset.y({"partition": "train"}))
    y_test = np.asarray(dataset.y({"partition": "test"}))
    spec = None
    if apply_recommendation:
        selected = TransferPreprocessingSelector(preset="fast", verbose=0).fit(train, test, y_train, y_test)
        spec = selected.to_pipeline_spec(top_k=top_k, use_augmentation=use_augmentation)
    augment = isinstance(spec, dict)
    names = spec["feature_augmentation"] if augment else [spec] if isinstance(spec, str) else spec or []
    replay_cohort = SpectroDataset("transfer_replay_oracle")
    replay_cohort.add_samples(test, {"partition": "test"})
    replay_ids = mint_identity(replay_cohort).observation_ids()
    replay_order = np.array(sorted(range(len(test)), key=lambda row: replay_ids[row]))
    predictions, feature_blocks = {}, {}
    for backend in ("legacy", "native", "native_replay", "native_rowwise"):
        predict_input = raw if backend == "legacy" else test[replay_order] if backend == "native_replay" else test
        current_train, current_predict = train, predict_input
        train_outputs, predict_outputs = [current_train], [current_predict]
        for name in names:
            if augment:
                current_train, current_predict = train, predict_input
            for component in name.split(">"):
                transform = deepcopy(get_base_preprocessings()[component])
                transform.fit(current_train)
                current_train = np.asarray(transform.transform(current_train))
                current_predict = (
                    np.vstack([transform.transform(row[None, :]) for row in current_predict])
                    if backend == "native_rowwise" else np.asarray(transform.transform(current_predict))
                )
            if backend == "legacy":
                # SpectroDataset materializes complete stored-row blocks as float32.
                current_predict = current_predict.astype(np.float32)
                current_train = current_predict[train_rows]
            if augment:
                train_outputs.append(current_train)
                predict_outputs.append(current_predict)
        if augment:
            current_train, current_predict = np.hstack(train_outputs), np.hstack(predict_outputs)
        test_features = current_predict[test_rows] if backend == "legacy" else current_predict
        if backend == "native_replay":
            # PREDICT resolves its declared wire identities in lexical order;
            # restore original caller row order before comparing predictions.
            test_features = test_features[np.argsort(replay_order)]
        feature_blocks[backend] = (current_train, test_features)
        predictions[backend] = np.asarray(PLSRegression(3).fit(current_train, y_train).predict(test_features)).ravel()
    # Batch-size-dependent float32 rounding is bounded before model amplification.
    feature_bound = 64 * np.finfo(np.float32).eps * max(1., float(np.max(np.abs(raw))))
    for backend in ("legacy", "native_replay", "native_rowwise"):
        for left, right in zip(feature_blocks[backend], feature_blocks["native"], strict=True):
            assert left.shape == right.shape
            assert np.max(np.abs(left - right)) <= feature_bound
    return predictions, test_ids


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
    direct, test_ids = _direct_transfer_oracles(dataset, apply_recommendation=apply_recommendation)
    np.testing.assert_allclose(legacy_pred, direct["legacy"], atol=1e-6)
    legacy_rows = legacy.predictions.filter_predictions(partition="test", load_arrays=True)
    assert all(list(row["sample_indices"]) == test_ids for row in legacy_rows)

    for mode in ("1", "0"):
        monkeypatch.setenv("N4A_DAGML_INPROCESS", mode)
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
        with pytest.warns(NoSplitEvaluationWarning, match="No splitter"):
            native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)
        np.testing.assert_allclose(_test_prediction(native), direct["native"], atol=1e-6)
        assert native.best_rmse == pytest.approx(float(np.sqrt(np.mean((direct["native"] - test_y) ** 2))), abs=1e-6)
        assert any(
            "allow_fit_cv_all_observations_view" in frame["lineage"]["unsafe_flags"]
            for frame in native._dagml_node_results
        )
        archive = native.export(tmp_path / f"auto_transfer_{apply_recommendation}_{mode}.n4a")
        np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, test_x).y_pred).ravel(), direct["native_replay"], atol=1e-6)


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
def test_auto_transfer_top_k_replays_legacy_selection_and_archive(tmp_path, monkeypatch, record_property, use_augmentation: bool) -> None:
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
    dataset = DatasetConfigs(path).get_dataset_at(0)
    direct, _ = _direct_transfer_oracles(dataset, top_k=2, use_augmentation=use_augmentation)
    legacy_pred, native_pred = _test_prediction(legacy), _test_prediction(native)
    np.testing.assert_allclose(legacy_pred, direct["legacy"], atol=1e-6)
    np.testing.assert_allclose(native_pred, direct["native"], atol=1e-6)
    # Preserve strict backend-to-oracle checks, then allow only independently
    # measured model amplification of the bounded float32 feature difference.
    precision_envelope = np.abs(direct["native"] - direct["legacy"]) + 2e-6
    assert np.all(np.abs(native_pred - legacy_pred) <= precision_envelope)
    record_property("independent_float32_prediction_envelope_max", float(precision_envelope.max()))
    record_property("independent_rowwise_to_replay_max", float(np.max(np.abs(direct["native_rowwise"] - direct["native_replay"]))))
    archive = native.export(tmp_path / f"auto_transfer_top_k_{use_augmentation}.n4a")
    x_test = DatasetConfigs(path).get_dataset_at(0).x({"partition": "test"}, layout="2d")
    np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel(), direct["native_replay"], atol=1e-6)


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
