"""Target preprocessing before ResidualModel remains scoped to its base model."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nirs4all
from nirs4all.data import DatasetConfigs
from nirs4all.operators.models.residual import ResidualModel

from ._datasets import dataset_path


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("target_chain", [False, True])
def test_residual_target_preprocessing_refit_and_replay(tmp_path, monkeypatch, mechanism: str, target_chain: bool) -> None:
    """Legacy accepts this prefix; DAG applies it once to the base target."""
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    pipeline = [
        KFold(2, shuffle=True, random_state=1),
        {"y_processing": StandardScaler()},
        *([{"y_processing": MinMaxScaler()}] if target_chain else []),
        {"model": ResidualModel(base=PLSRegression(n_components=2), learner=Ridge(), gate=False)},
    ]
    source = dataset_path("regression")
    legacy = nirs4all.run(
        pipeline, source, engine="legacy", refit=False,
        workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    native = nirs4all.run(
        pipeline, source, engine="dag-ml", refit=True,
        workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(native.cv_best_score)
    assert np.isfinite(native.best_rmse)
    artifacts = {artifact["controller_id"]: artifact for artifact in native._dagml_refit_artifacts}
    assert artifacts["controller:nirs4all.model"]["y_transform"] is not None
    assert artifacts["controller:nirs4all.residual_learner"]["y_transform"] is None
    archive = native.export(tmp_path / "residual_y_preprocessing.n4a")
    dataset = DatasetConfigs(source).get_dataset_at(0)
    expected = np.asarray(dataset.y({"partition": "test"})).ravel()
    replay = nirs4all.predict(archive, dataset.x({"partition": "test"}, layout="2d"))
    assert np.sqrt(np.mean((expected - np.asarray(replay.y_pred).ravel()) ** 2)) == pytest.approx(native.best_rmse, abs=1e-5)
    native.close()


@pytest.mark.parity
def test_residual_learner_train_params_reach_native_fit(tmp_path, monkeypatch) -> None:
    """The legacy submodel receives train_params as estimator overrides."""
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    pipeline = [
        KFold(2, shuffle=True, random_state=1),
        {"model": ResidualModel(
            base=PLSRegression(n_components=2), learner=Ridge(alpha=0.1), gate=False,
            train_params={"alpha": 4.0},
        )},
    ]
    legacy = nirs4all.run(
        pipeline, dataset_path("regression"), engine="legacy", refit=False,
        workspace_path=tmp_path / "legacy-params", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    native = nirs4all.run(
        pipeline, dataset_path("regression"), engine="dag-ml", refit=True,
        workspace_path=tmp_path / "native-params", save_artifacts=False, save_charts=False, verbose=0,
    )
    learner = next(artifact["estimator"] for artifact in native._dagml_refit_artifacts if artifact["controller_id"] == "controller:nirs4all.residual_learner")
    assert learner.get_params()["alpha"] == pytest.approx(4.0)
    assert learner._nirs4all_training_controls["model_params"]["alpha"] == pytest.approx(4.0)
    native.close()


@pytest.mark.torch
@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_residual_torch_learner_train_params_refit_and_replay(tmp_path, monkeypatch, mechanism: str) -> None:
    """A framework learner uses the ordinary DAG host adapter inside a residual graph."""
    pytest.importorskip("torch")
    from nirs4all.operators.models.pytorch.nicon import customizable_decon

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    rng = np.random.default_rng(12)
    x = rng.uniform(0, 1, (16, 64)).astype(np.float32)
    y = rng.uniform(0, 1, (16, 1)).astype(np.float32)
    dataset = tmp_path / "data"
    dataset.mkdir()
    for name, values in (("Xcal", x[:12]), ("Ycal", y[:12]),
                         ("Xval", x[12:]), ("Yval", y[12:])):
        np.savetxt(dataset / f"{name}.csv", values, delimiter=";")
    pipeline = [
        KFold(2, shuffle=True, random_state=1),
        {"model": ResidualModel(
            base=PLSRegression(n_components=2), learner=customizable_decon,
            gate=False, train_params={"epochs": 1, "batch_size": 4},
        )},
    ]
    legacy = nirs4all.run(pipeline, str(dataset), engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False,
                          save_charts=False, verbose=0)
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    native = nirs4all.run(pipeline, str(dataset), engine="dag-ml", allow_fallback=False,
                         workspace_path=tmp_path / "native", save_artifacts=False,
                         save_charts=False, verbose=0)
    try:
        assert native.execution_engine == "dag-ml"
        assert np.isfinite(native.cv_best_score)
        artifacts = {artifact["controller_id"]: artifact["estimator"] for artifact in native._dagml_refit_artifacts}
        learner = artifacts["controller:nirs4all.residual_learner"]
        assert learner.epochs == 1
        assert learner.batch_size == 4
        features = x[:3]
        expected = (
            np.asarray(artifacts["controller:nirs4all.model"].predict(features)).ravel()
            + np.asarray(learner.predict(features)).ravel()
        )
        archive = native.export(tmp_path / "residual_torch.n4a")
        np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, features).y_pred).ravel(), expected, atol=1e-5)
    finally:
        native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("merge_mode", ["features", "all", "selected"])
def test_residual_after_duplication_feature_merge_refit_and_replay(tmp_path, monkeypatch, mechanism: str, merge_mode: str) -> None:
    """Reuse the fold-local feature-merge transformer before both residual stages."""
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    branches = (
        [[StandardScaler(), {"model": Ridge(alpha=0.1)}], [MinMaxScaler(), {"model": Ridge(alpha=1.0)}]]
        if merge_mode == "all" else [[StandardScaler()], [MinMaxScaler()]]
    )
    merge = {"features": [0]} if merge_mode == "selected" else merge_mode
    pipeline = [
        KFold(2, shuffle=True, random_state=1),
        {"branch": branches},
        {"merge": merge},
        {"model": ResidualModel(base=PLSRegression(n_components=2), learner=Ridge(), gate=False)},
    ]
    source = dataset_path("regression")
    legacy = nirs4all.run(
        pipeline, source, engine="legacy", refit=False,
        workspace_path=tmp_path / "legacy-branch", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    native = nirs4all.run(
        pipeline, source, engine="dag-ml", refit=True,
        workspace_path=tmp_path / "native-branch", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(native.cv_best_score)
    assert np.isfinite(native.best_rmse)
    archive = native.export(tmp_path / "residual_branch_features.n4a")
    dataset = DatasetConfigs(source).get_dataset_at(0)
    expected = np.asarray(dataset.y({"partition": "test"})).ravel()
    replay = nirs4all.predict(archive, dataset.x({"partition": "test"}, layout="2d"))
    assert np.sqrt(np.mean((expected - np.asarray(replay.y_pred).ravel()) ** 2)) == pytest.approx(native.best_rmse, abs=1e-5)
    native.close()
