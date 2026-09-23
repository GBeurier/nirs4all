"""Branch model predictions become native OOF features for ResidualModel."""

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
@pytest.mark.parametrize("gate", [False, "auto"])
def test_residual_after_prediction_feature_join_is_oof_safe_and_replays(tmp_path, monkeypatch, mechanism: str, gate) -> None:
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    source = dataset_path("regression")
    pipeline = [
        KFold(2, shuffle=True, random_state=1),
        {"branch": [
            [StandardScaler(), {"model": Ridge(alpha=0.5)}],
            [MinMaxScaler(), {"model": Ridge(alpha=1.0)}],
        ]},
        {"merge": "predictions"},
        {"model": ResidualModel(base=PLSRegression(n_components=2), learner=Ridge(), gate=gate,
                                rli_threshold=1.0 if gate == "auto" else 0.0)},
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
    if gate == "auto":
        replay_contract = native.per_dataset[next(iter(native.per_dataset))]["residual_replay"]
        assert replay_contract["gate"] == pytest.approx(0.0)
    archive = native.export(tmp_path / "residual_prediction_features.n4a")
    dataset = DatasetConfigs(source).get_dataset_at(0)
    features = dataset.x({"partition": "test"}, layout="2d")
    targets = np.asarray(dataset.y({"partition": "test"})).ravel()
    replay = nirs4all.predict(archive, features)
    replay_rmse = np.sqrt(np.mean((targets - np.asarray(replay.y_pred).ravel()) ** 2))
    assert replay_rmse == pytest.approx(native.best_rmse, abs=1e-5)
    native.close()
