"""ResidualModel after a shared by-source preprocessing and concat merge."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.data import DatasetConfigs
from nirs4all.operators.models.residual import ResidualModel
from nirs4all.operators.transforms import StandardNormalVariate

from ._datasets import dataset_path


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_residual_after_by_source_concat_refit_and_replay(tmp_path, monkeypatch, mechanism: str) -> None:
    """Both residual stages apply source-local X transforms within each fold."""
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    source = dataset_path("multi")
    pipeline = [
        KFold(2, shuffle=True, random_state=1),
        {"branch": {"by_source": True, "steps": [StandardNormalVariate()]}},
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
    archive = native.export(tmp_path / "residual_by_source.n4a")
    dataset = DatasetConfigs(source).get_dataset_at(0)
    features = dataset.x({"partition": "test"}, layout="2d")
    targets = np.asarray(dataset.y({"partition": "test"})).ravel()
    replay = nirs4all.predict(archive, features)
    assert np.sqrt(np.mean((targets - np.asarray(replay.y_pred).ravel()) ** 2)) == pytest.approx(native.best_rmse, abs=1e-5)
    native.close()
