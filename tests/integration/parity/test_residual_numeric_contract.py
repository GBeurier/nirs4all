"""Numeric and archive contracts for native residual calibration and fusion."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.data import DatasetConfigs
from nirs4all.operators.models.residual import ResidualModel

from ._datasets import dataset_path


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("gate,threshold", [(0.35, 0.0), ("auto", 1.0), (True, 1.0), (None, 1.0)])
def test_residual_lambda_and_rli_threshold_match_refit_and_replay(tmp_path, monkeypatch, mechanism: str, gate, threshold: float) -> None:
    """REFIT predictions and replay use the core-calibrated scalar exactly."""
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")

    source = dataset_path("regression")
    operator = ResidualModel(
        base=PLSRegression(n_components=2), learner=Ridge(alpha=1.0),
        lam=0.6, gate=gate, rli_threshold=threshold,
    )
    pipeline = [KFold(2, shuffle=True, random_state=1), {"model": operator}]
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
    replay_contract = native.per_dataset[next(iter(native.per_dataset))]["residual_replay"]
    assert replay_contract["lambda"] == pytest.approx(0.6)
    if gate is True or gate is None or gate == "auto":
        assert replay_contract["gate"] == pytest.approx(0.0)
        assert all(record["gate"] == pytest.approx(0.0) for record in replay_contract["gate_records"])
    else:
        assert replay_contract["gate"] == pytest.approx(gate)

    artifacts = {artifact["controller_id"]: artifact for artifact in native._dagml_refit_artifacts}
    dataset = DatasetConfigs(source).get_dataset_at(0)
    features = dataset.x({"partition": "test"}, layout="2d")
    targets = np.asarray(dataset.y({"partition": "test"})).ravel()
    base = np.asarray(artifacts["controller:nirs4all.model"]["estimator"].predict(features)).ravel()
    learner = np.asarray(artifacts["controller:nirs4all.residual_learner"]["estimator"].predict(features)).ravel()
    expected = base + replay_contract["lambda"] * replay_contract["gate"] * learner
    assert np.sqrt(np.mean((targets - expected) ** 2)) == pytest.approx(native.best_rmse, abs=1e-5)
    archive = native.export(tmp_path / "residual_numeric.n4a")
    replay = nirs4all.predict(archive, features)
    np.testing.assert_allclose(np.asarray(replay.y_pred).ravel(), expected, atol=1e-6)
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("lam", [0.0, -0.5])
def test_residual_zero_and_negative_lambda_replay(tmp_path, monkeypatch, mechanism: str, lam: float) -> None:
    """The public residual operator preserves a signed scalar through refit and replay."""
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")

    source = dataset_path("regression")
    operator = ResidualModel(base=PLSRegression(n_components=2), learner=Ridge(alpha=1), lam=lam, gate=False)
    pipeline = [
        KFold(2, shuffle=True, random_state=1),
        {"model": operator},
    ]
    legacy = nirs4all.run(
        pipeline, source, engine="legacy", refit=False,
        workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(legacy.cv_best_score)
    legacy_rows = [row for row in legacy.predictions.filter_predictions(model_name=operator.name, load_arrays=True)
                   if row["partition"] == "val" and np.asarray(row["y_pred"]).size]
    assert legacy_rows and all(np.isfinite(row["val_score"]) for row in legacy_rows)
    legacy.close()

    native = nirs4all.run(
        pipeline, source, engine="dag-ml", refit=True,
        workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0,
    )
    contract = native.per_dataset[next(iter(native.per_dataset))]["residual_replay"]
    assert contract["lambda"] == pytest.approx(lam)
    assert contract["gate"] == pytest.approx(1.0)
    artifacts = {artifact["controller_id"]: artifact for artifact in native._dagml_refit_artifacts}
    dataset = DatasetConfigs(source).get_dataset_at(0)
    features = dataset.x({"partition": "test"}, layout="2d")
    base = np.asarray(artifacts["controller:nirs4all.model"]["estimator"].predict(features)).ravel()
    learner = np.asarray(artifacts["controller:nirs4all.residual_learner"]["estimator"].predict(features)).ravel()
    expected = base + lam * learner
    archive = native.export(tmp_path / "residual_signed_lambda.n4a")
    replay = nirs4all.predict(archive, features)
    np.testing.assert_allclose(np.asarray(replay.y_pred).ravel(), expected, atol=1e-5)
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_residual_nonlinear_base_and_custom_name_replay(tmp_path, monkeypatch, mechanism: str) -> None:
    """A named residual model can use a nonlinear base and retain its fusion in archives."""
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")

    source = dataset_path("regression")
    operator = ResidualModel(
        base=RandomForestRegressor(n_estimators=8, max_depth=3, random_state=3),
        learner=Ridge(alpha=1), lam=0.5, gate=False, name="named_residual_rf",
    )
    assert operator.name == "named_residual_rf"
    pipeline = [KFold(2, shuffle=True, random_state=1), {"model": operator}]
    legacy = nirs4all.run(
        pipeline, source, engine="legacy", refit=False,
        workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(legacy.cv_best_score)
    legacy_rows = [row for row in legacy.predictions.filter_predictions(model_name=operator.name, load_arrays=True)
                   if row["partition"] == "val" and np.asarray(row["y_pred"]).size]
    assert legacy_rows and all(np.isfinite(row["val_score"]) for row in legacy_rows)
    legacy.close()

    native = nirs4all.run(
        pipeline, source, engine="dag-ml", refit=True,
        workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0,
    )
    contract = native.per_dataset[next(iter(native.per_dataset))]["residual_replay"]
    assert contract["lambda"] == pytest.approx(0.5)
    artifacts = {artifact["controller_id"]: artifact for artifact in native._dagml_refit_artifacts}
    dataset = DatasetConfigs(source).get_dataset_at(0)
    features = dataset.x({"partition": "test"}, layout="2d")
    base = np.asarray(artifacts["controller:nirs4all.model"]["estimator"].predict(features)).ravel()
    learner = np.asarray(artifacts["controller:nirs4all.residual_learner"]["estimator"].predict(features)).ravel()
    expected = base + 0.5 * learner
    replay = nirs4all.predict(native.export(tmp_path / "named_residual_rf.n4a"), features)
    np.testing.assert_allclose(np.asarray(replay.y_pred).ravel(), expected, atol=5e-5)
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_residual_nonlinear_learner_signed_gate_replays(tmp_path, monkeypatch, mechanism: str) -> None:
    """The residual learner may itself be a nonlinear host estimator."""
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")

    source = dataset_path("regression")
    operator = ResidualModel(
        base=PLSRegression(n_components=2),
        learner=RandomForestRegressor(n_estimators=8, max_depth=3, random_state=2),
        lam=-0.5, gate=0.4, name="named_rf_learner",
    )
    pipeline = [KFold(2, shuffle=True, random_state=11), {"model": operator}]
    with nirs4all.run(pipeline, source, engine="legacy", refit=False,
                      workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0) as legacy:
        assert np.isfinite(legacy.cv_best_score)
        rows = [row for row in legacy.predictions.filter_predictions(model_name=operator.name, load_arrays=True)
                if row["partition"] == "val" and np.asarray(row["y_pred"]).size]
        assert rows and all(np.isfinite(row["val_score"]) for row in rows)

    with nirs4all.run(pipeline, source, engine="dag-ml", allow_fallback=False, refit=True,
                      workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0) as native:
        assert np.isfinite(native.cv_best_score)
        contract = native.per_dataset[next(iter(native.per_dataset))]["residual_replay"]
        assert contract["lambda"] == pytest.approx(-0.5)
        assert contract["gate"] == pytest.approx(0.4)
        artifacts = {artifact["controller_id"]: artifact["estimator"] for artifact in native._dagml_refit_artifacts}
        assert isinstance(artifacts["controller:nirs4all.residual_learner"], RandomForestRegressor)
        dataset = DatasetConfigs(source).get_dataset_at(0)
        features = dataset.x({"partition": "test"}, layout="2d")
        base = np.asarray(artifacts["controller:nirs4all.model"].predict(features)).ravel()
        learner = np.asarray(artifacts["controller:nirs4all.residual_learner"].predict(features)).ravel()
        expected = base - 0.2 * learner
        replay = nirs4all.predict(native.export(tmp_path / "named_rf_learner.n4a"), features)
        np.testing.assert_allclose(np.asarray(replay.y_pred).ravel(), expected, atol=5e-5)
