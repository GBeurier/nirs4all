"""Numerical references for AOM PLS selection, scores and fitted state."""
import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

from nirs4all.operators.models._aom_nirs.fast.models._common import ridge_on_scores
from nirs4all.operators.models._aom_nirs.fast.models.fast_aom_pls_ridge import FastAOMConfig, FastAOMPLSRidge
from nirs4all.operators.models._aom_nirs.pls.classification import AOMPLSDAClassifier
from nirs4all.operators.models._aom_nirs.pls.metrics import balanced_accuracy
from nirs4all.operators.models._aom_nirs.pls.operators import ExplicitMatrixOperator, IdentityOperator
from nirs4all.operators.models._aom_nirs.pls.scorers import CriterionConfig, approx_press_regression
from nirs4all.operators.models._aom_nirs.pls.selection import select


@pytest.mark.parametrize('gamma', [None, 2.])
def test_score_ridge_loo_matches_explicit_refits(gamma):
    rng = np.random.default_rng(4)
    T = rng.normal(size=(20, 4))
    T -= T.mean(axis=0)
    y = T[:, 0] + rng.normal(size=20)
    y -= y.mean()
    grid = [0., .01, .1, 1., 10., 100.]
    losses = []
    for alpha in grid:
        residuals = []
        for i in range(len(y)):
            keep = np.arange(len(y)) != i
            xm, ym = T[keep].mean(axis=0), y[keep].mean()
            tx = T[keep] - xm
            penalties = np.ones(4) if gamma is None else np.arange(1, 5)**gamma
            coef = np.linalg.solve(tx.T @ tx + np.diag(alpha * penalties), tx.T @ (y[keep] - ym))
            residuals.append(y[i] - ((T[i] - xm) @ coef + ym))
        losses.append(np.mean(np.square(residuals)))
    coef, alpha, loss = ridge_on_scores(T, y, grid, gamma)
    assert alpha == grid[np.argmin(losses)]
    assert loss == pytest.approx(min(losses))
    assert alpha > 0
    assert np.isfinite(coef).all()


@pytest.mark.parametrize('engine', ['simpls_covariance', 'simpls_materialized', 'nipals_materialized', 'nipals_adjoint', 'pls_standard'])
@pytest.mark.parametrize('nonidentity', [False, True])
def test_classifier_training_transform_and_calibrator_agree(engine, nonidentity):
    X = np.random.default_rng(5).normal(size=(45, 6))
    labels = np.where(X[:, 0] + .5 * X[:, 2] > 0, 'positive', 'negative')
    operator = ExplicitMatrixOperator(np.diag(np.arange(1, 7))) if nonidentity else IdentityOperator()
    model = AOMPLSDAClassifier(n_components=3, engine=engine, criterion='covariance',
                             operator_bank=[operator]).fit(X, labels)
    transformed = model.transform(X)
    np.testing.assert_allclose(transformed, model.x_scores_, atol=1e-10)
    np.testing.assert_allclose(model.predict_proba(X), model._calibrator.predict_proba(model.x_scores_), atol=1e-10)
    assert model.score(X, labels) == balanced_accuracy_score(labels, model.predict(X))


def test_classifier_numpy_component_count_and_scaling_reference():
    X = np.random.default_rng(6).normal(size=(45, 6)) * np.array([1, 10, 100, .01, .1, 2])
    labels = np.where(X[:, 0] > 0, 'positive', 'negative')
    options = {'n_components': np.int64(3), 'criterion': 'covariance', 'operator_bank': [IdentityOperator()]}
    scaled = AOMPLSDAClassifier(scale=True, **options).fit(X, labels)
    reference = AOMPLSDAClassifier(**options).fit((X - X.mean(axis=0))/X.std(axis=0, ddof=1), labels)
    assert scaled.n_components_ == 3
    np.testing.assert_allclose(scaled.predict_proba(X), reference.predict_proba((X-X.mean(axis=0))/X.std(axis=0, ddof=1)), atol=1e-10)
    assert not np.allclose(scaled.coef_, AOMPLSDAClassifier(**options).fit(X, labels).coef_)


def test_soft_multioutput_direction_is_target_permutation_invariant():
    X = np.random.default_rng(6).normal(size=(70, 5))
    X -= X.mean(axis=0)
    Y = np.c_[np.random.default_rng(7).normal(size=70) * .001, X[:, 2]]
    Y -= Y.mean(axis=0)
    args = {'operators': [IdentityOperator()], 'engine': 'nipals_adjoint', 'selection': 'soft',
                'n_components_max': 1, 'criterion': CriterionConfig(kind='covariance')}
    first = select(X, Y, **args).result
    swapped = select(X, Y[:, ::-1], **args).result
    np.testing.assert_allclose(first.coef(), swapped.coef()[:, ::-1], atol=1e-10)
    direction = np.linalg.svd(X.T @ Y, full_matrices=False)[0][:, 0]
    score = X @ direction
    loading = X.T @ score / (score @ score)
    y_loading = Y.T @ score / (score @ score)
    expected = np.outer(direction, y_loading) / (loading @ direction)
    np.testing.assert_allclose(first.coef(), expected, atol=1e-10)


@pytest.mark.parametrize('mode', ['soft', 'superblock', 'active_superblock'])
def test_auto_prefix_cv_matches_independent_fold_refits(mode):
    X = np.random.default_rng(7).normal(size=(36, 6))
    y = X[:, 0] + np.random.default_rng(8).normal(size=36)
    X -= X.mean(axis=0)
    y -= y.mean()
    criterion = CriterionConfig(kind='cv', cv=3, random_state=9)
    args = {'operators': [IdentityOperator()], 'engine': 'simpls_covariance', 'selection': mode, 'criterion': criterion}
    curves = []
    for k in range(1, 5):
        rmses = []
        for train, val in KFold(3, shuffle=True, random_state=9).split(X):
            xm, ym = X[train].mean(axis=0), y[train].mean()
            result = select(X[train]-xm, y[train]-ym, n_components_max=k, auto_prefix=False, **args).result
            residual = y[val] - ((X[val]-xm) @ result.coef()).ravel() - ym
            rmses.append(np.sqrt(np.mean(residual**2)))
        curves.append(np.mean(rmses))
    automatic = select(X, y, n_components_max=4, auto_prefix=True, **args)
    np.testing.assert_allclose(automatic.diagnostics['score_curve'], curves, atol=1e-10)
    assert automatic.n_components_selected == np.argmin(curves) + 1


def test_press_latent_leverage_matches_explicit_ols_loo_with_rank_deficiency():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(20, 40))
    X -= X.mean(axis=0)
    y = X[:, 0] + rng.normal(size=20)
    y -= y.mean()
    coefs, scores, expected = [], [], []
    for k in [1, 3]:
        Z = np.eye(40)[:, :k]
        T = X @ Z
        c = np.linalg.lstsq(T, y, rcond=None)[0]
        coefs.append(Z @ c)
        scores.append(np.c_[T, T[:, :1]])
        errors = []
        for i in range(20):
            keep = np.arange(20) != i
            fitted = Ridge(alpha=0.).fit(T[keep], y[keep])
            errors.append(y[i] - fitted.predict(T[[i]])[0])
        expected.append(np.sum(np.square(errors)))
    actual = approx_press_regression(X, y, coefs, scores)
    np.testing.assert_allclose(actual, expected, rtol=1e-10)
    assert all(np.isfinite(actual))


def test_fast_constant_to_varying_refit_matches_fresh_instance():
    X = np.random.default_rng(4).normal(size=(25, 6))
    y = X[:, 0] + .3 * X[:, 2]
    config = FastAOMConfig(model='single_chain', n_components=2, rank=6, max_chain_depth=1,
                           top_global=3, use_snv=False)
    model = FastAOMPLSRidge(config).fit(X, np.full(25, 3.))
    model.fit(X, y)
    fresh = FastAOMPLSRidge(config).fit(X, y)
    np.testing.assert_allclose(model.predict(X), fresh.predict(X))
    assert np.std(model.predict(X)) > .1


@pytest.mark.parametrize(('truth', 'prediction', 'expected'), [(['a', 'b'], ['b', 'a'], 0.), ([.2, .8], [.8, .2], 0.), (['a', 'b'], ['a', 'b'], 1.)])
def test_balanced_accuracy_preserves_original_labels(truth, prediction, expected):
    assert balanced_accuracy(truth, prediction) == expected


def test_soft_zero_target_stops_extraction_without_invented_direction():
    X = np.random.default_rng(4).normal(size=(20, 5))
    result = select(X, np.zeros(20), [IdentityOperator()], 'simpls_covariance',
                    'soft', 3, CriterionConfig(kind='approx_press'))
    assert result.result.n_components == 0


def test_score_ridge_numpy_lambda_sequence():
    X = np.random.default_rng(4).normal(size=(15, 2))
    result = ridge_on_scores(X - X.mean(axis=0), np.arange(15)-7., np.array([.1, 1., 10.]))
    assert np.isfinite(result[2])


def test_soft_mixed_operator_directions_do_not_depend_on_target_order():
    X = np.random.default_rng(4).normal(size=(50, 6))
    X -= X.mean(axis=0)
    Y = np.c_[X[:, 0] + .3*X[:, 4], X[:, 3] - X[:, 2], .2*X[:, 1]]
    operators = [IdentityOperator(), ExplicitMatrixOperator(np.diag([1., .1, 4., 8., .2, 3.]))]
    options = {'operators': operators, 'engine': 'nipals_adjoint', 'selection': 'soft',
               'n_components_max': 3, 'criterion': CriterionConfig(kind='covariance')}
    first = select(X, Y, **options).result
    swapped = select(X, Y[:, ::-1], **options).result
    np.testing.assert_allclose(first.coef(), swapped.coef()[:, ::-1], atol=1e-10)
