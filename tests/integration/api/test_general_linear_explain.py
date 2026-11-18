"""The linear explainer explains a complete captured chain exactly or refuses it."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.operators.transforms import SavitzkyGolay, StandardNormalVariate
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


def _chain(tmp_path, preprocessing):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 24)).cumsum(axis=1)
    y = X[:, 4] - 0.5 * X[:, 15] + rng.normal(scale=0.1, size=40)
    nirs4all.run([*preprocessing, KFold(3), PLSRegression(3)], {"X": X, "y": y}, engine="dag-ml",
                 allow_fallback=False, workspace_path=str(tmp_path), save_artifacts=True, verbose=0)
    with WorkspaceStore(tmp_path) as store:
        chain = store.query_predictions()["chain_id"][0]
    return {"chain_id": chain, "workspace_path": str(tmp_path)}, X


def _explain(model, X, explainer_type, tmp_path):
    return nirs4all.explain(model=model, data=X, engine="dag-ml", verbose=0, plots_visible=False,
                            explainer_type=explainer_type, background_samples=40, workspace_path=str(tmp_path),
                            visualizations=[], sample_indices=list(range(8)))


def test_linear_explainer_is_exact_for_a_linear_chain(tmp_path):
    model, X = _chain(tmp_path, [StandardScaler(), SavitzkyGolay(window_length=7, polyorder=2)])
    linear = _explain(model, X, "linear", tmp_path)
    assert linear.explainer_type == "LinearExplainer"
    assert linear.explanation["exactness"] == "exact"
    scale = max(1.0, float(np.abs(linear.shap_values).sum(axis=1).max()))
    assert linear.explanation["additivity_max_abs_error"] < 1e-6 * scale
    # Interventional SHAP of feature j: f(x) - f(x with x_j at its background mean).
    reference = X.mean(axis=0)
    rows = X[:8]
    replaced = np.repeat(rows, X.shape[1], axis=0)
    columns = np.tile(np.arange(X.shape[1]), len(rows))
    replaced[np.arange(len(replaced)), columns] = reference[columns]
    predict = nirs4all.predict(model=model, data=np.vstack([rows, replaced, reference[None]]), engine="dag-ml", verbose=0)
    predicted = np.asarray(predict.y_pred).reshape(-1)
    expected = predicted[:8, None] - predicted[8:-1].reshape(8, X.shape[1])
    np.testing.assert_allclose(linear.shap_values, expected, atol=1e-6 * scale)
    np.testing.assert_allclose(linear.base_value, predicted[-1], atol=1e-6 * scale)


def test_linear_explainer_refuses_a_nonlinear_chain(tmp_path):
    model, X = _chain(tmp_path, [StandardNormalVariate()])
    with pytest.raises(ValueError, match="linear in its raw inputs"):
        _explain(model, X, "linear", tmp_path)
