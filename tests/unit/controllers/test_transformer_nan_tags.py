"""Missing-value aware operators receive the original NaNs under modern sklearn."""
import numpy as np
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from nirs4all.controllers.transforms.transformer import TransformerMixinController


def test_modern_nan_tags_are_respected():
    assert TransformerMixinController._allows_nan(MissingIndicator())
    assert TransformerMixinController._allows_nan(SimpleImputer())
    assert TransformerMixinController._allows_nan(StandardScaler())
    assert not TransformerMixinController._allows_nan(Ridge())

    class LegacyOperator:
        def _more_tags(self):
            return {"allow_nan": True}

    assert TransformerMixinController._allows_nan(LegacyOperator())


def test_missing_indicator_sees_actual_nan_values_and_replays(tmp_path):
    import nirs4all
    from nirs4all.pipeline.storage import WorkspaceStore

    X = np.arange(80, dtype=np.float32).reshape(20, 4)
    X[::2, 0] = np.nan
    X[::3, 2] = np.nan
    y = np.isnan(X[:, 0]).astype(float) + 2 * np.isnan(X[:, 2]) + np.linspace(0, .1, 20)
    data = {"train_x": X, "train_y": y, "na_policy": "ignore"}
    workspace = tmp_path / "workspace"
    nirs4all.run([MissingIndicator(), Ridge()], data, workspace_path=workspace,
                 engine="legacy", verbose=0)
    expected = Ridge().fit(MissingIndicator().fit_transform(X), y).predict(MissingIndicator().fit_transform(X))
    with WorkspaceStore(workspace) as store:
        rows = store.query_predictions().to_dicts()
        chain_id = next(row["chain_id"] for row in rows if row["partition"] == "train")
        np.testing.assert_allclose(store.replay_chain(chain_id, X), expected, atol=1e-6)
