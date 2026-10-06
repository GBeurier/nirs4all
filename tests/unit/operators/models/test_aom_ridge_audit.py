"""Independent ridge, distance, CV and persistence references for AOM models."""
import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from nirs4all.operators.models._aom_nirs.pls.operators import IdentityOperator
from nirs4all.operators.models._aom_nirs.ridge.estimators import AOMRidgeRegressor
from nirs4all.operators.models._aom_nirs.ridge.kernelizer import AOMKernelizer
from nirs4all.operators.models._aom_nirs.ridge.local_ridge import _local_predict, _topk_indices_per_row
from nirs4all.operators.models._aom_nirs.ridge.mkr_estimator import (
    AOMMultiKernelRidge,
    _inner_cv_rmse_alpha,
    _select_alpha_with_one_se,
)
from nirs4all.operators.models._aom_nirs.ridge.multi_branch_mkl import (
    AOMMultiBranchMKL,
    cross_branch_kernel,
    fit_branches_and_kernels,
)
from nirs4all.operators.models._aom_nirs.ridge.tabpfn_candidate import TabPFNCandidate


class RidgeCandidate(TabPFNCandidate):
    """Network-free inner estimator to isolate wrapper persistence behavior."""

    def _make_estimator(self):
        return Ridge(alpha=.3)


@pytest.mark.parametrize('query', [1., -1.])
def test_local_neighbors_and_predictions_match_euclidean_ridge(query):
    X = np.array([-6., -.9, -.1, .1, .9, 5., 6., 7.])[:, None]
    y = X[:, 0]**2
    kernel = X @ X.T
    cross = np.array([[query]]) @ X.T
    neighbors = _topk_indices_per_row(cross, 2, np.diag(kernel))[0]
    expected = np.argsort(np.abs(X[:, 0]-query))[:2]
    assert set(neighbors) == set(expected)
    reference = Ridge(alpha=.3, fit_intercept=False).fit(X[expected], y[expected])
    actual = _local_predict(kernel, cross, y[:, None], np.zeros(1), 2, .3)
    np.testing.assert_allclose(actual.ravel(), reference.predict([[query]]), atol=1e-10)


@pytest.mark.parametrize('center', [False, True])
@pytest.mark.parametrize('multioutput', [False, True])
def test_multibranch_identity_matches_sklearn_ridge(center, multioutput):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 5)) + 8
    query = rng.normal(size=(6, 5)) + 8
    y = X[:, 0] - .5 * X[:, 2] + rng.normal(size=30) * .2
    if multioutput:
        y = np.c_[y, 2*y + .1 * X[:, 3]]
    model = AOMMultiBranchMKL(branches=('none',), operator_bank=[IdentityOperator()],
                               center=center, alphas=[.1, 1., 10.], cv=3).fit(X, y)
    Xc = X - X.mean(axis=0) if center else X
    factor = len(X) / np.sum(Xc**2)
    reference = Ridge(alpha=model.alpha_/factor, fit_intercept=center).fit(X, y)
    np.testing.assert_allclose(model.predict(query), reference.predict(query), rtol=1e-9, atol=1e-9)
    if center:
        shifted = clone(model).fit(X+17, y)
        np.testing.assert_allclose(shifted.predict(query+17), model.predict(query), atol=1e-9)


def test_branch_centering_reuses_training_post_transform_means():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(25, 6)) + 3
    query = rng.normal(size=(3, 6)) + 7
    branches = ('none', 'snv')
    kernels, fitted, factors = fit_branches_and_kernels(X, branches, [IdentityOperator()], center=True)
    cross = cross_branch_kernel(query, X, branches, [IdentityOperator()],
                                {'none': .5, 'snv': .5}, fitted, factors)
    reference = np.zeros_like(cross)
    for name in branches:
        train = X if name == 'none' else (X-X.mean(axis=1, keepdims=True))/X.std(axis=1, keepdims=True)
        test = query if name == 'none' else (query-query.mean(axis=1, keepdims=True))/query.std(axis=1, keepdims=True)
        train_centered, test_centered = train-train.mean(axis=0), test-train.mean(axis=0)
        factor = len(X) / np.sum(train_centered**2)
        np.testing.assert_allclose(kernels[name], train_centered @ train_centered.T * factor, atol=1e-10)
        reference += .5 * (test_centered @ train_centered.T) * factor
    np.testing.assert_allclose(cross, reference, atol=1e-10)


@pytest.mark.parametrize('strategy', ['uniform', 'manual', 'kta', 'softmax_cv'])
def test_multikernel_numpy_alpha_grid_and_fold_se(strategy):
    X = np.random.default_rng(6).normal(size=(30, 5))
    y = X[:, 0] + np.random.default_rng(7).normal(size=30)
    grid = np.array([.001, .01, .1, 1., 10., 100.])
    model = AOMMultiKernelRidge(operator_bank=[IdentityOperator()], alphas=grid,
                               weight_strategy=strategy, weight_init=[1.],
                               one_se_rule=True, alpha_cv_n_splits=3, random_state=8).fit(X, y)
    blocks = model.kernelizer_.K_train_blocks_
    summary, folds = _inner_cv_rmse_alpha(blocks, y-y.mean(), model.eta_, grid,
                                         KFold(3, shuffle=True, random_state=8), return_per_fold=True)
    best = np.argmin(summary)
    threshold = summary[best] + np.std(folds[:, best], ddof=1)/np.sqrt(3)
    expected = grid[summary <= threshold].max()
    assert model.alpha_ == expected
    np.testing.assert_allclose(model.predict(X), model.predict_dual(X), atol=1e-10)


def test_one_se_uses_folds_and_numeric_alpha_order():
    grid = np.array([1., 100., 10., .1])
    folds = np.array([[.1, .4, .11, .2], [.1, .4, .11, .2], [.1, .4, .11, .2]])
    assert _select_alpha_with_one_se(folds.mean(axis=0), grid, True, folds) == (1., 0)
    noisy = folds.copy()
    noisy[:, 0] = [0., .1, .2]
    assert _select_alpha_with_one_se(noisy.mean(axis=0), grid, True, noisy) == (10., 2)


@pytest.mark.parametrize('mode', ['superblock', 'global', 'active_superblock', 'mkl', 'branch_global'])
def test_trimmed_cv_scoring_matches_pooled_sklearn_reference(mode):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 4))
    y = X[:, 0] + rng.normal(size=30) * .1
    y[0] += 40
    grid = np.array([.1, 1., 10.])
    model = AOMRidgeRegressor(selection=mode, operator_bank=[IdentityOperator()],
                              branches=('none',), alphas=grid, scoring='rmse_pooled_trimmed',
                              block_scaling='none', cv=3, random_state=5).fit(X, y)
    scores = []
    for alpha in grid:
        errors = []
        for tr, va in KFold(3, shuffle=True, random_state=5).split(X):
            reference = Ridge(alpha=alpha).fit(X[tr], y[tr])
            errors.extend(y[va]-reference.predict(X[va]))
        residuals = np.sort(np.abs(errors))[:29]
        scores.append(np.sqrt(np.mean(residuals**2)))
    assert model.alpha_ == grid[np.argmin(scores)]
    assert model.diagnostics_['cv_min_score'] == pytest.approx(min(scores))


@pytest.mark.parametrize('mode', ['superblock', 'global', 'active_superblock', 'mkl', 'branch_global'])
def test_explicit_alpha_grid_skips_adaptive_repeated_cv(mode, monkeypatch):
    X = np.random.default_rng(4).normal(size=(24, 4))
    y = X[:, 0]
    model = AOMRidgeRegressor(selection=mode, operator_bank=[IdentityOperator()],
                              branches=('none',), alphas=[1e3, 1e4, 1e5], cv=3, max_grid_expansions=2)
    calls = []
    original = model._build_alpha_grid_from_data
    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(model, '_build_alpha_grid_from_data', tracked)
    model.fit(X, y)
    assert len(calls) == 1
    assert model.diagnostics_['grid_expansions'] == 0


def test_auto_expansions_count_only_evaluated_grids(monkeypatch):
    X = np.random.default_rng(4).normal(size=(24, 4))
    y = X[:, 0]
    model = AOMRidgeRegressor(operator_bank=[IdentityOperator()], alpha_grid_size=5,
                              max_grid_expansions=2, cv=3)
    calls = []
    original = model._build_alpha_grid_from_data
    def tracked(*args, **kwargs):
        grid = original(*args, **kwargs)
        calls.append(grid)
        return grid
    monkeypatch.setattr(model, '_build_alpha_grid_from_data', tracked)
    model.fit(X, y)
    assert model.diagnostics_['grid_expansions'] == len(calls)-1 == 2
    assert not np.array_equal(calls[0], calls[-1])


def test_kernelizer_clone_preserves_zero_trace_policy_and_threshold():
    original = AOMKernelizer([IdentityOperator()], zero_trace_policy='drop', zero_trace_threshold=.001)
    copied = clone(original)
    assert copied.zero_trace_policy == original.zero_trace_policy
    assert copied.zero_trace_threshold == original.zero_trace_threshold
    with pytest.raises(ValueError, match='all blocks dropped'):
        copied.fit(np.ones((5, 4)))


def test_tabpfn_wrapper_pickle_retains_predictions_without_network():
    X = np.random.default_rng(4).normal(size=(25, 8))
    y = X[:, 0] + .3 * X[:, 2]
    model = RidgeCandidate(max_features=4, max_samples=20).fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict(X), model.predict(X))
