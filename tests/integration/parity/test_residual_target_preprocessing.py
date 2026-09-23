"""Target preprocessing before ResidualModel remains scoped to its base model."""

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
def test_residual_target_preprocessing_refit_and_replay(tmp_path, monkeypatch, mechanism: str) -> None:
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
