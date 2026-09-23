"""Legacy ResidualModel syntax reaches the native graph and replays its result."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.models.residual import ResidualModel

from ._dagml_cli import dagml_cli_path


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("syntax", ["direct", "residual_instance"])
def test_residual_public_instance_forms_refit_and_replay(tmp_path, monkeypatch, mechanism: str, syntax: str) -> None:
    """Both legacy instance forms retain a native residual fusion and archive."""
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")

    rng = np.random.default_rng(917)
    features = rng.normal(size=(28, 7))
    targets = 2 * features[:, 0] - features[:, 1] + 0.1 * rng.normal(size=28)

    def dataset() -> SpectroDataset:
        result = SpectroDataset("residual_public_instance")
        result.add_samples(features[:24], {"partition": "train"}, headers=[str(index) for index in range(7)])
        result.add_samples(features[24:], {"partition": "test"})
        result.add_targets(targets.reshape(-1, 1))
        return result

    def pipeline() -> list:
        operator = ResidualModel(base=Ridge(alpha=1.0), learner=Ridge(alpha=1.0), gate=False)
        step = operator if syntax == "direct" else {"residual": operator}
        return [KFold(2, shuffle=True, random_state=1), step]

    legacy = nirs4all.run(pipeline(), dataset(), engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    native = nirs4all.run(pipeline(), dataset(), engine="dag-ml", allow_fallback=False, refit=True,
                          workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0)
    try:
        assert native.execution_engine == "dag-ml"
        assert np.isfinite(native.cv_best_score)
        assert np.isfinite(native.best_rmse)
        archive = native.export(tmp_path / "residual_instance.n4a")
        predicted = np.asarray(nirs4all.predict(archive, features[24:]).y_pred).ravel()
        assert np.sqrt(np.mean((targets[24:] - predicted) ** 2)) == pytest.approx(native.best_rmse, abs=1e-5)
    finally:
        native.close()
