"""Public legacy/DAG-ML parity for neural model layout and TensorFlow controls."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

pytest.importorskip("dag_ml")
try:
    import torch
except ImportError:
    torch = None


if torch is not None:
    class FlatOnlyModule(torch.nn.Module):
        framework = "pytorch"

        def __init__(self) -> None:
            super().__init__()
            # A Conv1d child makes the default PyTorch layout 3D. The model's
            # actual forward contract intentionally requires the 2D override.
            self.conv = torch.nn.Conv1d(1, 1, 1)
            self.linear = torch.nn.Linear(16, 1)
            self.last_shape: tuple[int, ...] | None = None

        def forward(self, features):
            self.last_shape = tuple(features.shape[1:])
            if features.ndim != 2:
                raise ValueError("this model requires force_layout='2d'")
            return self.linear(features)


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.torch
@pytest.mark.parity
def test_torch_force_layout_2d_matches_legacy_and_replays(tmp_path, monkeypatch, mechanism: str) -> None:
    pytest.importorskip("torch")
    import nirs4all

    rng = np.random.default_rng(23)
    x = rng.normal(size=(10, 16)).astype(np.float32)
    y = rng.normal(size=(10, 1)).astype(np.float32)
    pipeline = [KFold(2), {"model": FlatOnlyModule(), "force_layout": "2d", "train_params": {"epochs": 1, "batch_size": 4}}]

    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", save_charts=False)
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", save_charts=False)
    fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001 - inspect refit layout
    assert fitted.model_.last_shape == (16,)
    expected = np.asarray(fitted.predict(x[:2])).reshape(-1)
    archive = result.export(tmp_path / "force_layout_2d.n4a")
    np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, x[:2]).y_pred).reshape(-1), expected)
    result.close()


@pytest.mark.parity
def test_cv_without_refit_cli_and_in_process_have_same_native_predictions(monkeypatch) -> None:
    from ._dagml_cli import dagml_cli_path

    cli = dagml_cli_path()
    if not cli.exists():
        pytest.skip(f"dag-ml-cli binary not built at {cli}")
    monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    import nirs4all

    rng = np.random.default_rng(26)
    x = rng.normal(size=(12, 8))
    y = 0.5 * x[:, 0] + 0.1 * x[:, 2]
    pipeline = [KFold(2), Ridge(alpha=0.5)]
    snapshots = []
    for mode in ("1", "0"):
        monkeypatch.setenv("N4A_DAGML_INPROCESS", mode)
        result = nirs4all.run(pipeline, (x, y), engine="dag-ml", refit=False, save_charts=False)
        reports = {report["fold_id"]: report["metrics"]["rmse"] for report in result._dagml_score_set["reports"]}  # noqa: SLF001
        rows = {
            row["fold_id"]: (np.asarray(row["sample_indices"]), np.asarray(row["y_pred"]))
            for row in result.predictions.filter_predictions(load_arrays=True)
        }
        snapshots.append((reports, rows))
        result.close()
    in_process, subprocess = snapshots
    assert in_process[0] == subprocess[0]
    assert in_process[1].keys() == subprocess[1].keys() == {"0", "1", "avg"}
    for fold in in_process[1]:
        np.testing.assert_array_equal(in_process[1][fold][0], subprocess[1][fold][0])
        np.testing.assert_allclose(in_process[1][fold][1], subprocess[1][fold][1], rtol=1e-12)


@pytest.mark.tensorflow
@pytest.mark.parity
def test_tensorflow_nested_compile_fit_controls_match_legacy_and_replay(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    pytest.importorskip("tensorflow")
    import nirs4all
    from nirs4all.operators.models.tensorflow.nicon import customizable_decon

    rng = np.random.default_rng(24)
    x = rng.uniform(0, 1, (8, 64)).astype(np.float32)
    y = rng.uniform(0, 1, (8, 1)).astype(np.float32)
    pipeline = [
        KFold(2),
        {"model": customizable_decon, "train_params": {"compile": {"loss": "mae"}, "fit": {"epochs": 1, "batch_size": 4, "verbose": 0}}},
    ]
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", save_charts=False)
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", save_charts=False)
    fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001 - inspect refit model
    assert fitted.model_.loss == "mae"
    assert len(fitted.model_.history.epoch) == 1
    expected = np.asarray(fitted.predict(x[:2])).reshape(-1)
    archive = result.export(tmp_path / "tensorflow_nested_controls.n4a")
    np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, x[:2]).y_pred).reshape(-1), expected, rtol=1e-6, atol=1e-6)
    result.close()


@pytest.mark.tensorflow
@pytest.mark.parity
def test_tensorflow_custom_callbacks_are_not_a_working_legacy_run_contract(tmp_path, monkeypatch) -> None:
    """The public legacy serializer turns callback instances into strings."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    tf = pytest.importorskip("tensorflow")
    import nirs4all
    from nirs4all.operators.models.tensorflow.nicon import customizable_decon
    from nirs4all.pipeline.dagml.rt import RtError

    class Probe(tf.keras.callbacks.Callback):
        pass

    rng = np.random.default_rng(2401)
    x = rng.uniform(0, 1, (8, 64)).astype(np.float32)
    y = rng.uniform(0, 1, (8, 1)).astype(np.float32)
    pipeline = [
        KFold(2),
        {"model": customizable_decon, "train_params": {
            "epochs": 1, "batch_size": 4, "custom_callbacks": [Probe()],
        }},
    ]
    with pytest.raises(RuntimeError, match="'str' object has no attribute 'set_model'"):
        nirs4all.run(pipeline, (x, y), engine="legacy", workspace_path=tmp_path / "legacy", save_charts=False, verbose=0)
    with pytest.raises(RtError, match="not JSON serializable"):
        nirs4all.run(pipeline, (x, y), engine="dag-ml", workspace_path=tmp_path / "dagml", save_charts=False, verbose=0)


@pytest.mark.tensorflow
@pytest.mark.parity
def test_tensorflow_cyclic_lr_legacy_callback_runs_but_cannot_assign_keras3_lr(tmp_path, monkeypatch) -> None:
    """The current Keras variable rejects legacy backend.set_value, leaving LR unchanged."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    tf = pytest.importorskip("tensorflow")
    if not isinstance(tf.keras.optimizers.Adam().learning_rate.dtype, str):
        pytest.skip("this Keras version does not have the Keras 3 learning-rate variable")
    import nirs4all
    from nirs4all.controllers.models.tensorflow.config import TensorFlowCallbackFactory
    from nirs4all.operators.models.tensorflow.nicon import customizable_decon
    from nirs4all.pipeline.dagml.rt import RtError

    made = []
    original = TensorFlowCallbackFactory.create_cyclic_lr

    def record(train_params, verbose=0):
        callback = original(train_params, verbose)
        made.append(callback)
        return callback

    monkeypatch.setattr(TensorFlowCallbackFactory, "create_cyclic_lr", staticmethod(record))
    rng = np.random.default_rng(2402)
    x = rng.uniform(0, 1, (8, 64)).astype(np.float32)
    y = rng.uniform(0, 1, (8, 1)).astype(np.float32)
    pipeline = [
        KFold(2),
        {"model": customizable_decon, "train_params": {
            "epochs": 1, "batch_size": 2, "verbose": 0, "best_model_memory": False,
            "cyclic_lr": True, "cyclic_lr_params": {"base_lr": 0.0001, "max_lr": 0.001, "step_size": 1},
        }},
    ]
    result = nirs4all.run(pipeline, (x, y), engine="legacy", workspace_path=tmp_path / "legacy", save_charts=False, verbose=0)
    assert np.isfinite(result.cv_best_score)
    assert [callback.clr_iterations for callback in made] == [2, 2, 4]
    assert [float(callback.model.optimizer.learning_rate.numpy()) for callback in made] == pytest.approx([0.001] * 3)
    result.close()
    with pytest.raises(RtError, match="cyclic_lr"):
        nirs4all.run(pipeline, (x, y), engine="dag-ml", workspace_path=tmp_path / "dagml", save_charts=False, verbose=0)


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parity
def test_cv_without_refit_uses_native_scores_and_no_refit_artifact(monkeypatch, mechanism: str) -> None:
    import nirs4all

    rng = np.random.default_rng(25)
    x = rng.normal(size=(12, 8))
    y = 0.7 * x[:, 0] - 0.2 * x[:, 1] + 0.05
    pipeline = [KFold(2), Ridge(alpha=1.0)]
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", refit=False, save_charts=False)
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", refit=False, save_charts=False)
    assert result.execution_engine == "dag-ml"
    assert result.per_dataset and all(metadata["refit_enabled"] is False for metadata in result.per_dataset.values())
    assert np.isfinite(result.cv_best_score)
    np.testing.assert_allclose(result.cv_best_score, legacy.cv_best_score, rtol=1e-6)
    assert result._dagml_refit_artifacts == []  # noqa: SLF001 - no fitted REFIT identity
    assert {row["partition"] for row in result.predictions.filter_predictions(load_arrays=False)} == {"val"}
    assert all((frame.get("result") or frame).get("lineage", {}).get("phase") != "REFIT" for frame in result._dagml_node_results)  # noqa: SLF001
    result.close()


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parity
def test_model_parameter_sweep_without_refit_keeps_all_cv_variants(monkeypatch, mechanism: str) -> None:
    import nirs4all

    rng = np.random.default_rng(29)
    x = rng.normal(size=(16, 8))
    y = rng.normal(size=16)
    pipeline = [KFold(2), {"model": Ridge, "_grid_": {"alpha": [0.01, 1000.0]}}]
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", refit=False, save_charts=False)
    legacy_cv = {
        row["config_name"]: row["val_score"]
        for row in legacy.predictions.filter_predictions()
        if row["fold_id"] == "avg" and row["partition"] == "val"
    }
    assert len(legacy_cv) == 2
    legacy_best = legacy.cv_best_score
    legacy.close()

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", refit=False, save_charts=False)
    assert result._dagml_refit_artifacts == []  # noqa: SLF001
    native_cv = {
        row["config_name"]: row["val_score"]
        for row in result.predictions.filter_predictions()
        if row["fold_id"] == "avg" and row["partition"] == "val"
    }
    assert native_cv.keys() == legacy_cv.keys()
    for name, score in native_cv.items():
        np.testing.assert_allclose(score, legacy_cv[name], rtol=1e-6)
    np.testing.assert_allclose(result.cv_best_score, legacy_best, rtol=1e-6)
    assert {row["partition"] for row in result.predictions.filter_predictions()} == {"val"}
    result.close()


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parity
def test_repetition_cv_without_refit_keeps_grouped_validation(monkeypatch, mechanism: str) -> None:
    import nirs4all
    from nirs4all.data.config import DatasetConfigs

    from ._datasets import PARSER_FIXTURES

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    dataset = DatasetConfigs(str(PARSER_FIXTURES["aggregate_mean"]), repetition="sample_id")
    result = nirs4all.run([KFold(2), Ridge(alpha=1.0)], dataset, engine="dag-ml", refit=False, save_charts=False)
    assert np.isfinite(result.cv_best_score)
    assert result._dagml_refit_artifacts == []  # noqa: SLF001
    assert {row["partition"] for row in result.predictions.filter_predictions()} == {"val"}
    assert result.per_dataset and all(metadata["refit_enabled"] is False for metadata in result.per_dataset.values())
    result.close()


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parity
def test_operator_sweep_without_refit_keeps_all_cv_variants(monkeypatch, mechanism: str) -> None:
    import nirs4all

    rng = np.random.default_rng(31)
    x = rng.normal(size=(16, 8))
    y = 0.3 * x[:, 0] - 0.5 * x[:, 2] + rng.normal(size=16) * 0.1
    pipeline = [{"_or_": [StandardScaler(), MinMaxScaler()]}, KFold(2), Ridge(alpha=0.5)]
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", refit=False, save_charts=False)
    legacy_best = legacy.cv_best_score
    legacy_names = {row["config_name"] for row in legacy.predictions.filter_predictions()}
    legacy.close()

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", refit=False, save_charts=False)
    assert result._dagml_refit_artifacts == []  # noqa: SLF001
    assert {row["config_name"] for row in result.predictions.filter_predictions()} == legacy_names
    assert {row["partition"] for row in result.predictions.filter_predictions()} == {"val"}
    # Legacy fits StandardScaler globally before splitting; the two independent
    # sklearn oracles distinguish that behavior from leakage-safe fold-local CV.
    direct = np.empty_like(y)
    global_direct = np.empty_like(y)
    globally_scaled = StandardScaler().fit_transform(x)
    for train, val in KFold(2).split(x):
        direct[val] = make_pipeline(StandardScaler(), Ridge(alpha=0.5)).fit(x[train], y[train]).predict(x[val])
        global_direct[val] = Ridge(alpha=0.5).fit(globally_scaled[train], y[train]).predict(globally_scaled[val])
    np.testing.assert_allclose(result.cv_best_score, np.sqrt(np.mean((direct - y) ** 2)), rtol=1e-5)
    np.testing.assert_allclose(legacy_best, np.sqrt(np.mean((global_direct - y) ** 2)), rtol=1e-5)
    result.close()


@pytest.mark.parametrize("refit_option,enabled", [(None, False), ({}, False), ([], False), ({"top_k": 1}, True), ([{"top_k": 1}], True)])
@pytest.mark.parity
def test_legacy_equivalent_refit_spellings_use_native_on_off(refit_option, enabled: bool) -> None:
    import nirs4all

    rng = np.random.default_rng(33)
    x = rng.normal(size=(12, 6))
    y = 0.3 * x[:, 0] + rng.normal(size=12) * 0.01
    pipeline = [KFold(2), Ridge(alpha=0.5)]
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", refit=refit_option, save_charts=False)
    legacy_final = sum(row["fold_id"] == "final" for row in legacy.predictions.filter_predictions())
    legacy.close()

    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", refit=refit_option, save_charts=False)
    native_final = sum(row["fold_id"] == "final" for row in result.predictions.filter_predictions())
    assert bool(legacy_final) is enabled
    assert bool(native_final) is enabled
    assert bool(result._dagml_refit_artifacts) is enabled  # noqa: SLF001
    result.close()


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parity
def test_top_two_parameter_variants_refit_natively(tmp_path, monkeypatch, mechanism: str) -> None:
    import nirs4all

    rng = np.random.default_rng(29)
    x = rng.normal(size=(16, 8))
    y = rng.normal(size=16)
    pipeline = [KFold(2), {"model": Ridge, "_grid_": {"alpha": [0.01, 1.0, 1000.0]}}]
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", refit={"top_k": 2}, save_charts=False)
    legacy_final = {row["config_name"] for row in legacy.predictions.filter_predictions() if row["fold_id"] == "final"}
    legacy_cv = legacy.cv_best_score
    assert len(legacy_final) == 2
    legacy.close()

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", refit={"top_k": 2}, save_charts=False)
    native_final = {row["config_name"] for row in result.predictions.filter_predictions() if row["fold_id"] == "final"}
    assert native_final == legacy_final
    assert len(result._dagml_refit_artifacts) == 2  # noqa: SLF001
    np.testing.assert_allclose(result.cv_best_score, legacy_cv, rtol=1e-6)
    selected_id = result.per_dataset[next(iter(result.per_dataset))]["selected_refit_variant_ids"][0]
    primary = next(
        artifact for artifact in result._dagml_refit_artifacts  # noqa: SLF001
        if artifact["artifact_id"].endswith(f":nirs4all:refit:{selected_id}")
    )
    expected = primary["estimator"].predict(x[:3])
    archive = result.export(tmp_path / "top_two.n4a")
    np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, x[:3]).y_pred).reshape(-1), expected)
    model_path = result.export_model(tmp_path / "top_two.joblib")
    import joblib

    np.testing.assert_allclose(joblib.load(model_path).predict(x[:3]), expected)
    result.close()


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parity
def test_top_two_operator_variants_refit_on_pruned_plans(tmp_path, monkeypatch, mechanism: str) -> None:
    import nirs4all

    rng = np.random.default_rng(37)
    x = rng.normal(size=(16, 8))
    y = 0.3 * x[:, 0] - 0.5 * x[:, 2]
    pipeline = [{"_or_": [StandardScaler(), MinMaxScaler(), RobustScaler()]}, KFold(2), Ridge(alpha=0.5)]
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", refit={"top_k": 2}, save_charts=False)
    legacy_final = {row["config_name"] for row in legacy.predictions.filter_predictions() if row["fold_id"] == "final"}
    assert len(legacy_final) == 2
    legacy.close()

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", refit={"top_k": 2}, save_charts=False)
    native_final = {row["config_name"] for row in result.predictions.filter_predictions() if row["fold_id"] == "final"}
    assert native_final == legacy_final
    assert len(result._dagml_refit_artifacts) == 2  # noqa: SLF001
    selected_id = result.per_dataset[next(iter(result.per_dataset))]["selected_refit_variant_ids"][0]
    primary = next(
        artifact for artifact in result._dagml_refit_artifacts  # noqa: SLF001
        if artifact["artifact_id"].endswith(f":nirs4all:refit:{selected_id}")
    )
    expected = primary["estimator"].predict(x[:3])
    archive = result.export(tmp_path / "top_two_operators.n4a")
    np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, x[:3]).y_pred).reshape(-1), expected)
    result.close()
