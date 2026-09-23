"""Representative public residual base/learner operator pairs."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.data import DatasetConfigs
from nirs4all.operators.models.residual import ResidualModel

from ._dagml_cli import dagml_cli_path
from ._datasets import dataset_path


def _operator(kind: str):
    if kind == "pls":
        return PLSRegression(n_components=2)
    if kind == "forest":
        return RandomForestRegressor(n_estimators=8, max_depth=3, random_state=3)
    return Ridge(alpha=1.0)


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("base,learner", [
    ("ridge", "pls"), ("ridge", "forest"),
    ("pls", "pls"), ("pls", "forest"),
    ("forest", "pls"), ("forest", "forest"),
])
def test_residual_public_operator_pair_cv_refit_archive(tmp_path, monkeypatch, base, learner, mechanism):
    """Every qualified pair runs in legacy and replays its native REFIT result."""
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")

    source = dataset_path("regression")
    model = ResidualModel(base=_operator(base), learner=_operator(learner), lam=0.5, gate=False)
    pipeline = [KFold(2, shuffle=True, random_state=1), {"model": model}]
    legacy = nirs4all.run(pipeline, source, engine="legacy", allow_fallback=False, refit=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    try:
        assert np.isfinite(legacy.cv_best_score)
        rows = [row for row in legacy.predictions.filter_predictions(model_name=model.name, load_arrays=True)
                if row["partition"] == "val" and np.asarray(row["y_pred"]).size]
        assert rows and all(np.isfinite(row["val_score"]) for row in rows)
    finally:
        legacy.close()

    native = nirs4all.run(pipeline, source, engine="dag-ml", allow_fallback=False, refit=True,
                          workspace_path=tmp_path / "native", save_artifacts=True, save_charts=False, verbose=0)
    try:
        assert native.execution_engine == "dag-ml"
        assert np.isfinite(native.cv_best_score)
        features = DatasetConfigs(source).get_dataset_at(0).x({"partition": "test"}, layout="2d")
        artifacts = {artifact["controller_id"]: artifact for artifact in native._dagml_refit_artifacts}
        base_prediction = np.asarray(artifacts["controller:nirs4all.model"]["estimator"].predict(features)).ravel()
        learner_prediction = np.asarray(artifacts["controller:nirs4all.residual_learner"]["estimator"].predict(features)).ravel()
        expected = base_prediction + 0.5 * learner_prediction
        archive = native.export(tmp_path / "operator_pair.n4a")
        np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, features).y_pred).ravel(), expected, rtol=1e-5, atol=1e-4)
    finally:
        native.close()
