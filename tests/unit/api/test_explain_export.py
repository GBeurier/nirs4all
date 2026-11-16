"""Model exports for the Methods linear and path-dependent TreeSHAP kernels."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from nirs4all.api.explain_export import affine_coefficients, export_tree_ensemble, is_tree_ensemble


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(7)
    # float32-representable inputs: scikit-learn trees compare in float32.
    X = rng.normal(size=(60, 6)).astype(np.float32).astype(np.float64)
    y = 2.0 * X[:, 0] - X[:, 1] * X[:, 2] + 0.5 * np.sin(X[:, 3]) + 0.1 * rng.normal(size=60)
    return X, y


@pytest.fixture
def n4m_explain():
    return pytest.importorskip("n4m.explain")


def _shap_tree(model, X):
    shap = pytest.importorskip("shap")
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    values = explainer.shap_values(X)
    values = np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)
    return values.reshape(len(X), X.shape[1], -1), np.asarray(explainer.expected_value, dtype=float).reshape(-1)


def _tree_shap(n4m_explain, model, X, *, probabilities=False):
    ensemble = export_tree_ensemble(model, probabilities=probabilities)
    rows = X.astype(ensemble.input_dtype).astype(np.float64)
    return n4m_explain.tree_shap(rows, n4m_explain.TreeEnsemble(**ensemble.ensemble_fields()))


@pytest.mark.methods
@pytest.mark.parametrize("model", [
    DecisionTreeRegressor(max_depth=5, random_state=0),
    RandomForestRegressor(n_estimators=8, max_depth=4, random_state=0),
    ExtraTreesRegressor(n_estimators=6, max_depth=4, random_state=0),
    GradientBoostingRegressor(n_estimators=12, max_depth=3, learning_rate=0.2, subsample=0.8, random_state=0),
], ids=["decision_tree", "random_forest", "extra_trees", "gradient_boosting"])
def test_sklearn_regressor_export_matches_shap_tree_explainer(n4m_explain, regression_data, model):
    X, y = regression_data
    model.fit(X, y)
    result = _tree_shap(n4m_explain, model, X[:15])
    expected_values, expected_base = _shap_tree(model, X[:15])
    np.testing.assert_allclose(result.predictions[:, 0], model.predict(X[:15]), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(result.attributions, expected_values, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(result.expected[:, 0], expected_base[0], rtol=1e-6, atol=1e-9)
    assert result.kind == "tree_shap"
    assert result.additivity_max_abs_error < 1e-9


@pytest.mark.methods
def test_sklearn_forest_classifier_probability_export_matches_shap(n4m_explain, regression_data):
    X, y = regression_data
    labels = np.digitize(y, np.quantile(y, [0.33, 0.66]))
    model = RandomForestClassifier(n_estimators=6, max_depth=4, random_state=0).fit(X, labels)
    result = _tree_shap(n4m_explain, model, X[:10], probabilities=True)
    expected_values, expected_base = _shap_tree(model, X[:10])
    np.testing.assert_allclose(result.predictions, model.predict_proba(X[:10]), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(result.attributions, expected_values, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(result.expected[0], expected_base, rtol=1e-6, atol=1e-9)
    assert not is_tree_ensemble(model, probabilities=False)


@pytest.mark.methods
def test_xgboost_export_matches_shap_tree_explainer(n4m_explain, regression_data):
    xgboost = pytest.importorskip("xgboost")
    X, y = regression_data
    model = xgboost.XGBRegressor(n_estimators=10, max_depth=3, learning_rate=0.3, base_score=0.4).fit(X, y)
    result = _tree_shap(n4m_explain, model, X[:12])
    expected_values, _ = _shap_tree(model, X[:12])
    np.testing.assert_allclose(result.predictions[:, 0], model.predict(X[:12]), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result.attributions, expected_values, rtol=1e-5, atol=1e-6)


@pytest.mark.methods
def test_lightgbm_export_matches_shap_tree_explainer(n4m_explain, regression_data):
    lightgbm = pytest.importorskip("lightgbm")
    X, y = regression_data
    model = lightgbm.LGBMRegressor(n_estimators=10, num_leaves=7, min_child_samples=5, verbose=-1).fit(X, y)
    result = _tree_shap(n4m_explain, model, X[:12])
    expected_values, _ = _shap_tree(model, X[:12])
    np.testing.assert_allclose(result.predictions[:, 0], model.predict(X[:12]), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(result.attributions, expected_values, rtol=1e-6, atol=1e-9)


def test_tree_export_refuses_unsupported_models(regression_data):
    X, y = regression_data
    with pytest.raises(ValueError, match="not a supported regression tree ensemble"):
        export_tree_ensemble(KNeighborsRegressor().fit(X, y), probabilities=False)
    custom_init = GradientBoostingRegressor(n_estimators=3, init=LinearRegression()).fit(X, y)
    with pytest.raises(ValueError, match="non-constant offset"):
        export_tree_ensemble(custom_init, probabilities=False)
    assert not is_tree_ensemble(Ridge(), probabilities=False)


@pytest.mark.parametrize("model", [Ridge(alpha=0.5), PLSRegression(n_components=3)], ids=["ridge", "pls"])
def test_affine_coefficients_reproduce_predictions(regression_data, model):
    X, y = regression_data
    model.fit(X, y)
    coefficients, intercept = affine_coefficients(model, X)
    assert coefficients.shape == (X.shape[1], 1)
    np.testing.assert_allclose(X @ coefficients + intercept, np.asarray(model.predict(X)).reshape(-1, 1), rtol=1e-9, atol=1e-9)


class _ColumnCoefficients:
    """An affine estimator storing coef_ as (n_features, n_outputs)."""

    def __init__(self, coef, offset, transform=None):
        self.coef_ = coef
        self.offset = offset
        self.transform = transform or (lambda value: value)

    def predict(self, X):
        return self.transform(X @ self.coef_ + self.offset)


def test_affine_coefficients_resolve_orientation_and_refuse_nonlinear_predictions(regression_data):
    X, _ = regression_data
    coef = np.arange(12, dtype=float).reshape(6, 2) / 10
    coefficients, intercept = affine_coefficients(_ColumnCoefficients(coef, np.array([1.0, -2.0])), X)
    np.testing.assert_allclose(coefficients, coef)
    np.testing.assert_allclose(intercept, [1.0, -2.0])
    with pytest.raises(ValueError, match="does not reproduce"):
        affine_coefficients(_ColumnCoefficients(coef, 0.0, np.exp), X)
    with pytest.raises(ValueError, match="exposes no coef_"):
        affine_coefficients(KNeighborsRegressor().fit(X, X[:, 0]), X)


@pytest.mark.methods
@pytest.mark.parametrize("estimator", [Ridge(alpha=0.5), PLSRegression(n_components=3)], ids=["ridge", "pls"])
def test_linear_shap_matches_shap_linear_explainer(n4m_explain, regression_data, estimator):
    shap = pytest.importorskip("shap")
    X, y = regression_data
    estimator.fit(X, y)
    background = X[:40]
    coefficients, intercept = affine_coefficients(estimator, background)
    result = n4m_explain.linear_shap(X[40:], coefficients, reference=background.mean(axis=0), intercept=intercept)
    explainer = shap.LinearExplainer((coefficients[:, 0], float(intercept[0])), background)
    np.testing.assert_allclose(result.attributions[:, :, 0], explainer.shap_values(X[40:]), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(result.predictions[:, 0], np.asarray(estimator.predict(X[40:])).reshape(-1), rtol=1e-9, atol=1e-9)
    if isinstance(estimator, Ridge):
        direct = shap.LinearExplainer(estimator, background)
        np.testing.assert_allclose(result.attributions[:, :, 0], direct.shap_values(X[40:]), rtol=1e-6, atol=1e-9)


@pytest.mark.methods
def test_xgboost_thresholds_keep_float32_precision_at_split_values(n4m_explain, regression_data):
    """Rows lying exactly on a split must follow XGBoost's float32 comparison."""
    xgboost = pytest.importorskip("xgboost")
    X, y = regression_data
    model = xgboost.XGBRegressor(n_estimators=6, max_depth=3, learning_rate=0.3).fit(X, y)
    ensemble = export_tree_ensemble(model, probabilities=False)
    internal = ensemble.feature >= 0
    thresholds = ensemble.threshold[internal]
    np.testing.assert_array_equal(thresholds, thresholds.astype(np.float32).astype(np.float64))
    rows = np.repeat(X[:1], internal.sum(), axis=0)
    rows[np.arange(len(rows)), ensemble.feature[internal]] = thresholds
    result = _tree_shap(n4m_explain, model, rows)
    expected_values, _ = _shap_tree(model, rows)
    np.testing.assert_allclose(result.predictions[:, 0], model.predict(rows), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result.attributions, expected_values, rtol=1e-5, atol=1e-6)


@pytest.mark.methods
def test_xgboost_export_uses_the_early_stopping_slice(n4m_explain, regression_data):
    xgboost = pytest.importorskip("xgboost")
    X, y = regression_data
    noise = np.random.default_rng(3).normal(size=len(y))
    model = xgboost.XGBRegressor(n_estimators=40, max_depth=3, learning_rate=0.3, early_stopping_rounds=2)
    model.fit(X[:40], y[:40], eval_set=[(X[40:], noise[40:])], verbose=False)
    ensemble = export_tree_ensemble(model, probabilities=False)
    assert len(ensemble.tree_offsets) - 1 == model.best_iteration + 1
    result = _tree_shap(n4m_explain, model, X[:10])
    np.testing.assert_allclose(result.predictions[:, 0], model.predict(X[:10]), rtol=1e-5, atol=1e-6)
