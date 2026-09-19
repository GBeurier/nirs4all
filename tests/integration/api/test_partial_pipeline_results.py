"""Late model failures preserve completed model evidence, never failed folds."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class FailsOnLargerFold(Ridge):
    """First KFold split fits 12 rows; the second 13-row fit must fail."""

    fitted_sizes = []

    def fit(self, X, y, sample_weight=None):
        type(self).fitted_sizes.append(len(X))
        if len(X) == 13:
            raise RuntimeError("deliberate failure on second fold")
        return super().fit(X, y, sample_weight=sample_weight)


@pytest.mark.parametrize("cross_validate", [False, True])
def test_late_failure_preserves_completed_model_and_replay(tmp_path, cross_validate):
    import nirs4all
    from nirs4all.pipeline.storage import WorkspaceStore

    rng = np.random.default_rng(832)
    X = rng.normal(size=(25, 4)).astype(np.float32)
    y = 2 * X[:, 0] - X[:, 1]
    steps = [StandardScaler()]
    if cross_validate:
        steps.append(KFold(n_splits=2))
    steps.extend([Ridge(), PLSRegression(n_components=99)])
    workspace = tmp_path / "workspace"
    with pytest.raises(RuntimeError, match="n_components"):
        nirs4all.run(steps, (X, y), engine="legacy", workspace_path=workspace,
                     verbose=0, save_charts=False, refit=False)
    with WorkspaceStore.open_readonly(workspace) as store:
        pipeline = store.list_pipelines().row(0, named=True)
        assert pipeline["status"] == "failed" and "n_components" in pipeline["error"]
        chains = store.query_chain_summaries().to_dicts()
        assert len(chains) == 1 and chains[0]["model_class"] == "Ridge"
        assert set(store.query_predictions()["model_class"].to_list()) == {"Ridge"}
        chain_id = chains[0]["chain_id"]
        archive = store.export_chain(chain_id, tmp_path / "completed-ridge.n4a")
        if cross_validate:
            # Preprocessing is fitted before the splitter, matching the pipeline.
            scaler = StandardScaler().fit(X)
            transformed = scaler.transform(X)
            expected = np.mean([
                Ridge().fit(transformed[train], y[train]).predict(transformed)
                for train, _ in KFold(n_splits=2).split(X)
            ], axis=0)
        else:
            expected = make_pipeline(StandardScaler(), Ridge()).fit(X, y).predict(X)
        actual = store.replay_chain(chain_id, X)
        np.testing.assert_allclose(np.asarray(actual).ravel(), expected, atol=1e-5)
    predicted = nirs4all.predict(model=archive, data=X, engine="legacy").y_pred
    np.testing.assert_allclose(np.asarray(predicted).ravel(), expected, atol=1e-5)


def test_failed_models_first_fold_is_not_attached_to_completed_model(tmp_path):
    import nirs4all
    from nirs4all.pipeline.storage import WorkspaceStore

    X = np.random.default_rng(833).normal(size=(25, 4))
    y = X[:, 0] - X[:, 1]
    FailsOnLargerFold.fitted_sizes = []
    with pytest.raises(RuntimeError, match="deliberate failure"):
        nirs4all.run([KFold(n_splits=2), Ridge(), FailsOnLargerFold()], (X, y),
                     engine="legacy", workspace_path=tmp_path, verbose=0, save_charts=False, refit=False)
    assert 12 in FailsOnLargerFold.fitted_sizes and 13 in FailsOnLargerFold.fitted_sizes
    with WorkspaceStore.open_readonly(tmp_path) as store:
        chains = store.query_chain_summaries().to_dicts()
        assert len(chains) == 1 and chains[0]["model_class"] == "Ridge"
        predictions = store.query_predictions().to_dicts()
        assert predictions and {row["model_class"] for row in predictions} == {"Ridge"}
        assert {row["chain_id"] for row in predictions} == {chains[0]["chain_id"]}
        assert {row["fold_id"] for row in predictions} >= {"0", "1"}
        assert store.list_pipelines().row(0, named=True)["status"] == "failed"
