"""Public augmentation compositions that legacy already executes."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.data.config import DatasetConfigs
from nirs4all.operators.augmentation import GaussianAdditiveNoise
from nirs4all.operators.transforms.scalers import StandardNormalVariate
from nirs4all.pipeline.dagml.full_train import NoSplitEvaluationWarning

from ._dagml_cli import dagml_cli_path
from ._datasets import dataset_path

pytestmark = pytest.mark.parity


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("with_splitter", [False, True])
def test_feature_branch_before_sample_augmentation_matches_legacy_and_replays(
    tmp_path, monkeypatch: pytest.MonkeyPatch, mechanism: str, with_splitter: bool,
) -> None:
    """A branch feature merge remains replayable after train-only augmentation."""
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")

    path = dataset_path("regression")
    pipeline = [
        {"branch": [[StandardNormalVariate()], [StandardScaler()]]},
        {"merge": "features"},
        {"sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01)],
            "count": 1, "selection": "all", "random_state": 42,
        }},
    ]
    if with_splitter:
        pipeline.append(KFold(n_splits=2, shuffle=True, random_state=42))
    pipeline.append({"model": PLSRegression(n_components=3)})

    legacy = nirs4all.run(pipeline, path, engine="legacy", allow_fallback=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    if with_splitter:
        native = nirs4all.run(pipeline, path, engine="dag-ml", allow_fallback=False,
                              workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0)
    else:
        with pytest.warns(NoSplitEvaluationWarning):
            native = nirs4all.run(pipeline, path, engine="dag-ml", allow_fallback=False,
                                  workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0)
    try:
        assert native.execution_engine == "dag-ml"
        assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-8)
        if with_splitter:
            assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-8)

        archive = tmp_path / "feature_branch_before_augmentation.n4a"
        native.export(archive)
        dataset = DatasetConfigs(path).get_dataset_at(0)
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        y_test = np.asarray(dataset.y({"partition": "test"}))
        replay = nirs4all.predict(archive, x_test)
        assert root_mean_squared_error(y_test, np.asarray(replay.y_pred)) == pytest.approx(native.best_rmse, abs=1e-8)
    finally:
        native.close()
        legacy.close()


def test_model_checkpoint_before_augmentation_keeps_both_legacy_models(tmp_path) -> None:
    """Record a real legacy-successful order for the remaining native checkpoint gap."""
    pipeline = [
        KFold(n_splits=2, shuffle=True, random_state=42),
        {"model": PLSRegression(n_components=3)},
        {"sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01)],
            "count": 1, "selection": "all", "random_state": 42,
        }},
        {"model": Ridge()},
    ]
    legacy = nirs4all.run(pipeline, dataset_path("regression"), engine="legacy", allow_fallback=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    try:
        assert legacy.get_models() == ["PLSRegression", "Ridge"]
        assert np.isfinite(legacy.cv_best_score)
        assert np.isfinite(legacy.best_rmse)
    finally:
        legacy.close()

    try:
        native = nirs4all.run(pipeline, dataset_path("regression"), engine="dag-ml", allow_fallback=False,
                              workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0)
    except Exception as exc:
        if "[run/unsupported_shape]" in str(exc):
            pytest.xfail("DAG-ML still needs native sequential model checkpoints across augmentation")
        raise
    try:
        assert native.get_models() == legacy.get_models()
    finally:
        native.close()
