"""Numerical and estimator-contract regressions for the scientific PLS audit."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from nirs4all.operators.models.sklearn import KOPLS, MBPLS, IntervalPLS, KernelPLS, RobustPLS, SparsePLS
from nirs4all.operators.models.sklearn.fckpls import FCKPLS, FractionalConvFeaturizer


def _data():
    X = np.random.default_rng(8).normal(size=(60, 12)) * np.arange(1, 13)
    y = 50 + X[:, 0] - 0.2 * X[:, 3] + 0.5 * X[:, 5]
    return X, y


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("weighting", ["huber", "tukey"])
def test_robust_pls_removes_contaminated_intercept(backend, weighting):
    if backend == "jax":
        pytest.importorskip("jax")
    rng = np.random.default_rng(23)
    X = rng.normal(size=(100, 8))
    clean_y = 50 + X[:, :5].sum(axis=1)
    contaminated_y = clean_y.copy()
    contaminated_y[:10] += 15
    X_test = rng.normal(size=(200, 8))
    y_test = 50 + X_test[:, :5].sum(axis=1)
    model = RobustPLS(n_components=5, weighting=weighting, backend=backend).fit(X, contaminated_y)
    assert abs(np.mean(model.predict(X_test) - y_test)) < 0.15
    assert np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2)) < 0.3
    assert abs(model.y_mean_[0] - contaminated_y.mean()) > 1


@pytest.mark.parametrize("kind", ["kopls", "kernelpls"])
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_unscaled_centered_kernel_preserves_target_offset(kind, backend):
    if backend == "jax":
        pytest.importorskip("jax")
    X, y = _data()
    params = {"kernel": "linear", "n_components": 3, "backend": backend}
    if kind == "kopls":
        factory = KOPLS
        params.update(scale=False, n_ortho_components=0)
    else:
        factory = KernelPLS
        params.update(scale_y=False)
    model = factory(**params).fit(X[:48], y[:48])
    shifted = factory(**params).fit(X[:48], y[:48] + 100)
    np.testing.assert_allclose(shifted.predict(X[48:]), model.predict(X[48:]) + 100, atol=1e-7)
    assert abs(np.mean(model.predict(X[48:]))) > 40


def test_kopls_reports_effective_predictive_rank():
    X, y = _data()
    model = KOPLS(n_components=5, kernel="linear", n_ortho_components=0).fit(X, y)
    assert model.n_components_ == model.transform(X).shape[1] == 1


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_sparse_pls_scale_toggle_changes_model_and_matches_manual_scaling(backend):
    if backend == "jax":
        pytest.importorskip("jax")
    X, y = _data()
    scaled = SparsePLS(n_components=2, alpha=1, scale=True, backend=backend).fit(X, y)
    unscaled = SparsePLS(n_components=2, alpha=1, scale=False, backend=backend).fit(X, y)
    assert np.max(np.abs(scaled.predict(X) - unscaled.predict(X))) > 0.1
    np.testing.assert_array_equal(unscaled._X_std, np.ones((1, X.shape[1])))
    x_mean, x_std = X.mean(axis=0), X.std(axis=0)
    y_mean, y_std = y.mean(), y.std()
    manual = SparsePLS(n_components=2, alpha=1, scale=False, backend=backend).fit((X - x_mean) / x_std, (y - y_mean) / y_std)
    np.testing.assert_allclose(scaled.predict(X), manual.predict((X - x_mean) / x_std) * y_std + y_mean, atol=1e-7)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_sparse_selected_features_and_scores_are_available_and_consistent(backend):
    if backend == "jax":
        pytest.importorskip("jax")
    X, y = _data()
    model = SparsePLS(n_components=3, alpha=1, backend=backend).fit(X, y)
    selected = model.get_selected_features()
    assert len(selected) > 0
    assert np.all(np.any(np.asarray(model.coef_)[selected] != 0, axis=-1))
    residual = (X - np.asarray(model._X_mean)) / np.asarray(model._X_std)
    expected = []
    for component in range(3):
        score = residual @ np.asarray(model._W)[:, component]
        score /= np.linalg.norm(score)
        expected.append(score)
        residual -= np.outer(score, np.asarray(model._P)[:, component])
    np.testing.assert_allclose(model.transform(X), np.column_stack(expected), atol=1e-7)


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("backend,multiblock", [("numpy", False), ("numpy", True), ("jax", False)])
def test_mbpls_transform_reproduces_training_scores(standardize, backend, multiblock):
    if backend == "jax":
        pytest.importorskip("jax")
    X, y = _data()
    inputs = [X[:, :5], X[:, 5:]] if multiblock else X
    model = MBPLS(n_components=3, standardize=standardize, backend=backend).fit(inputs, y)
    np.testing.assert_allclose(model.transform(inputs), model._T, atol=1e-7)
    if not standardize and not multiblock:
        np.testing.assert_array_equal(model._X_std, np.ones((1, X.shape[1])))


@pytest.mark.parametrize("sequences", [((0., 1.), (2.,)), ([0., 1.], [2.])])
def test_fractional_featurizer_can_be_cloned(sequences):
    model = FractionalConvFeaturizer(alphas=sequences[0], sigmas=sequences[1])
    replica = clone(model)
    X, _ = _data()
    np.testing.assert_allclose(model.fit_transform(X), replica.fit_transform(X))


@pytest.mark.parametrize("mode", ["same", "valid"])
def test_fractional_jax_matches_numpy_feature_bank_and_predictions(mode):
    pytest.importorskip("jax")
    rng = np.random.default_rng(27)
    X = rng.normal(size=(30, 40))
    y = X[:, :4].sum(axis=1)
    params = {"alphas": (0., 1.), "sigmas": (2.,), "kernel_size": 15, "mode": mode, "n_components": 2}
    numpy_model = FCKPLS(**params, backend="numpy").fit(X[:24], y[:24])
    jax_model = FCKPLS(**params, backend="jax").fit(X[:24], y[:24])
    from nirs4all.operators.models.sklearn.fckpls import _get_cached_jax_fckpls
    X_proc = numpy_model.x_scaler_.transform(X)
    actual_features = np.asarray(_get_cached_jax_fckpls()["apply_filter_bank"](X_proc, jax_model.featurizer_.kernels_, mode=mode))
    np.testing.assert_allclose(actual_features, numpy_model.featurizer_.transform(X_proc), atol=1e-7)
    assert numpy_model.n_features_out_ == jax_model.n_features_out_
    np.testing.assert_allclose(numpy_model.predict(X[24:]), jax_model.predict(X[24:]), atol=1e-7)
    np.testing.assert_allclose(numpy_model.transform(X), jax_model.transform(X), atol=1e-7)


def test_jax_last_short_interval_uses_its_actual_columns():
    jnp = pytest.importorskip("jax.numpy")
    from nirs4all.operators.models.sklearn.ipls import _get_jax_ipls_functions
    rng = np.random.default_rng(12)
    X = rng.normal(size=(60, 103))
    y = X[:, 90:103].sum(axis=1)[:, None]
    funcs = _get_jax_ipls_functions()
    result = np.asarray(funcs["eval_all_intervals"](jnp.asarray(X), jnp.asarray(y), jnp.array([0, 30, 60, 90]), jnp.array([30, 60, 90, 103]), 5, 3, 30))
    actual_last = np.asarray(funcs["eval_all_intervals"](jnp.asarray(X[:, 90:103]), jnp.asarray(y), jnp.array([0]), jnp.array([13]), 5, 3, 13))
    np.testing.assert_allclose(result[-1], actual_last[0], atol=1e-7)


@pytest.mark.parametrize("mode", ["single", "forward", "backward"])
def test_interval_pls_invalid_scoring_never_returns_a_plausible_model(mode):
    X, y = _data()
    with pytest.raises(ValueError):
        IntervalPLS(n_components=2, n_intervals=3, mode=mode, scoring="neg_rmse_typo").fit(X, y)


def test_jax_interval_pls_rejects_unsupported_scoring():
    pytest.importorskip("jax")
    X, y = _data()
    with pytest.raises(ValueError, match="scoring"):
        IntervalPLS(n_components=2, n_intervals=3, scoring="neg_mean_absolute_error", backend="jax").fit(X, y)


def test_interval_pls_jax_and_numpy_use_the_same_scaling_and_short_interval():
    pytest.importorskip("jax")
    X, y = _data()
    params = {"n_components": 2, "interval_width": 5, "mode": "single", "cv": 3}
    numpy_model = IntervalPLS(**params, backend="numpy").fit(X, y)
    jax_model = IntervalPLS(**params, backend="jax").fit(X, y)
    np.testing.assert_allclose(jax_model.interval_scores_, numpy_model.interval_scores_, atol=1e-7)
    assert jax_model.selected_regions_ == numpy_model.selected_regions_
    np.testing.assert_allclose(jax_model.predict(X), numpy_model.predict(X), atol=1e-7)


def test_interval_pls_does_not_swallow_runtime_cv_errors(monkeypatch):
    from nirs4all.operators.models.sklearn import ipls
    def fail(*args, **kwargs):
        raise RuntimeError("CV failure witness")
    monkeypatch.setattr(ipls, "cross_val_score", fail)
    X, y = _data()
    with pytest.raises(RuntimeError, match="CV failure witness"):
        IntervalPLS(n_components=2, n_intervals=3).fit(X, y)


@pytest.mark.parametrize("estimator", [RobustPLS, KOPLS, KernelPLS, SparsePLS, MBPLS, IntervalPLS, FCKPLS])
def test_scientific_estimators_reject_unknown_tuning_keys_and_unfitted_prediction(estimator):
    model = estimator()
    with pytest.raises(ValueError, match="n_componets"):
        model.set_params(n_componets=3)
    assert model.set_params(n_components=2) is model
    assert model.n_components == 2
    X, _ = _data()
    with pytest.raises(NotFittedError):
        model.predict(X)


@pytest.mark.parametrize("scale", [False, True])
def test_robust_pls_respects_disabled_centering(scale):
    X, y = _data()
    model = RobustPLS(n_components=3, center=False, scale=scale).fit(X, y)
    np.testing.assert_array_equal(model.x_mean_, np.zeros(X.shape[1]))
    np.testing.assert_array_equal(model.y_mean_, np.zeros(1))
    assert np.isfinite(model.predict(X)).all()
