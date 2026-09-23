"""A residual prediction join keeps usable model scopes on small train sets."""

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


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_residual_prediction_join_pls_capacity_refit_and_replay(tmp_path, monkeypatch, mechanism: str) -> None:
    """Twenty train rows work in legacy; DAG selects safe nested fit scopes."""
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")

    rng = np.random.default_rng(922)
    features = rng.normal(size=(24, 8))
    targets = 2 * features[:, 0] - features[:, 1] + 0.1 * rng.normal(size=24)

    def dataset() -> SpectroDataset:
        result = SpectroDataset("residual_capacity_20")
        result.add_samples(features[:20], {"partition": "train"}, headers=[str(index) for index in range(8)])
        result.add_samples(features[20:], {"partition": "test"})
        result.add_targets(targets.reshape(-1, 1))
        return result

    def pipeline() -> list:
        return [
            KFold(2, shuffle=True, random_state=1),
            {"y_processing": StandardScaler()},
            {"branch": [[{"model": Ridge(alpha=1.0)}], [{"model": Ridge(alpha=2.0)}]]},
            {"merge": "predictions"},
            {"model": ResidualModel(base=PLSRegression(n_components=2), learner=Ridge(), gate="auto")},
        ]

    legacy = nirs4all.run(pipeline(), dataset(), engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    native = nirs4all.run(pipeline(), dataset(), engine="dag-ml", allow_fallback=False, refit=True,
                          workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0)
    try:
        assert native.execution_engine == "dag-ml"
        assert np.isfinite(native.cv_best_score)
        archive = native.export(tmp_path / "capacity_20.n4a")
        predicted = np.asarray(nirs4all.predict(archive, features[20:]).y_pred).ravel()
        assert np.sqrt(np.mean((targets[20:] - predicted) ** 2)) == pytest.approx(native.best_rmse, abs=1e-5)
    finally:
        native.close()
