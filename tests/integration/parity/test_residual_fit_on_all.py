"""ResidualModel with an explicit all-observation preprocessing fit scope."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.models.residual import ResidualModel

from ._dagml_cli import dagml_cli_path


def _dataset() -> tuple[SpectroDataset, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(53)
    features = rng.normal(size=(34, 8))
    features[30:] += 5  # Fit-on-all has a visible effect on the scaler mean.
    targets = 2 * features[:, 0] - features[:, 1] + 0.1 * rng.normal(size=34)
    dataset = SpectroDataset("residual_fit_on_all")
    dataset.add_samples(features[:30], {"partition": "train"}, headers=[str(index) for index in range(8)])
    dataset.add_samples(features[30:], {"partition": "test"})
    dataset.add_targets(targets.reshape(-1, 1))
    return dataset, features, features[30:]


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
def test_residual_preprocessing_fit_on_all_runs_and_replays(
    tmp_path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    """A native all-observation view reaches the residual graph's X transform."""
    if mechanism == "cli":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "pyo3" else "0")
    pipeline = [
        {"preprocessing": StandardScaler(), "fit_on_all": True},
        KFold(2, shuffle=True, random_state=1),
        {"model": ResidualModel(base=PLSRegression(n_components=2),
                                learner=Ridge(alpha=0.3), gate=False)},
    ]
    legacy_data, _, _ = _dataset()
    legacy = nirs4all.run(pipeline, legacy_data, engine="legacy", allow_fallback=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    native_data, features, x_test = _dataset()
    native = nirs4all.run(pipeline, native_data, engine="dag-ml", allow_fallback=False,
                          workspace_path=tmp_path / "native", save_artifacts=True, save_charts=False, verbose=0)
    try:
        assert np.isfinite(legacy.best_rmse)
        assert np.isfinite(native.best_rmse)
        assert native.execution_engine == "dag-ml"
        assert len(native._dagml_refit_artifacts) == 2
        base = native._dagml_refit_artifacts[0]
        fitted_chain = base["estimator"].steps[0][1].transformer
        np.testing.assert_allclose(fitted_chain.steps[0].mean_, features.mean(axis=0), atol=1e-6)
        assert not np.allclose(fitted_chain.steps[0].mean_, features[:30].mean(axis=0), atol=1e-3)
        archive = native.export(tmp_path / "residual_fit_on_all.n4a")
        replay = nirs4all.predict(archive, x_test)
        y_test = np.asarray(native_data.y({"partition": "test"})).ravel()
        replay_rmse = np.sqrt(np.mean((y_test - np.asarray(replay.y_pred).ravel()) ** 2))
        assert replay_rmse == pytest.approx(native.best_rmse, abs=1e-5)
    finally:
        native.close()
        legacy.close()
