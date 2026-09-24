"""A metadata feature join feeds both stages of the native residual graph."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.data import DatasetConfigs
from nirs4all.operators.models.residual import ResidualModel

from ._datasets import dataset_path


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_residual_after_metadata_feature_join_refits_and_replays(tmp_path, monkeypatch, mechanism: str) -> None:
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    source = dataset_path("with_metadata")
    pipeline = [
        KFold(2, shuffle=True, random_state=1),
        {"branch": {"by_metadata": "group", "steps": [StandardScaler()]}},
        {"merge": "concat"},
        {"model": ResidualModel(base=PLSRegression(n_components=2), learner=Ridge(), gate=False)},
    ]
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
    assert artifacts["controller:nirs4all.model"]["estimator"].metadata_key == "group"
    assert artifacts["controller:nirs4all.residual_learner"]["estimator"].metadata_key == "group"
    archive = native.export(tmp_path / "residual_metadata_concat.n4a")
    dataset = DatasetConfigs(source).get_dataset_at(0)
    full_train_mean = np.asarray(dataset.x({"partition": "train"}, layout="2d")).mean(axis=0)
    for artifact in artifacts.values():
        for _, chain in artifact["estimator"].chain.branches:
            np.testing.assert_allclose(chain.steps[0].mean_, full_train_mean, rtol=1e-5, atol=1e-5)
    features = dataset.x({"partition": "test"}, layout="2d")
    metadata = dataset.metadata_column("group", {"partition": "test"})
    replay = nirs4all.predict(archive, {"X": features, "metadata": {"group": metadata}})
    targets = np.asarray(dataset.y({"partition": "test"})).ravel()
    replay_rmse = np.sqrt(np.mean((targets - np.asarray(replay.y_pred).ravel()) ** 2))
    assert replay_rmse == pytest.approx(native.best_rmse, abs=1e-5)
    with pytest.raises((ValueError, RuntimeError), match="metadata|group"):
        nirs4all.predict(archive, features)
    native.close()
