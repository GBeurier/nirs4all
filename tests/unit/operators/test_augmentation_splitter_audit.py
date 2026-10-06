"""Independent numerical and lifecycle regressions for scientific operators."""
import random

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.stats import chi2
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from nirs4all.operators.augmentation.edge_artifacts import EdgeArtifactsAugmenter
from nirs4all.operators.augmentation.environmental import MoistureAugmenter, TemperatureAugmenter
from nirs4all.operators.augmentation.spectral import ScatterSimulationMSC, WavelengthShift, WavelengthStretch
from nirs4all.operators.augmentation.synthesis import InstrumentalBroadeningAugmenter
from nirs4all.operators.data.repetition import RepetitionConfig
from nirs4all.operators.filters.x_outlier import XOutlierFilter
from nirs4all.operators.splitters.splitters import (
    KBinsStratifiedSplitter,
    KMeansSplitter,
    SPlitSplitter,
    SPXYSplitter,
    SystematicCircularSplitter,
)


@pytest.mark.parametrize('factory', [WavelengthShift, WavelengthStretch, TemperatureAugmenter,
                                    MoistureAugmenter, InstrumentalBroadeningAugmenter])
@pytest.mark.parametrize('order', ['descending', 'permuted'])
def test_wavelength_effects_are_channel_order_invariant(factory, order):
    wl = np.linspace(1100, 2500, 301)
    X = np.array([1 + np.exp(-((wl - center) / 40)**2) for center in (1440, 1930)])
    original = X.copy()
    permutation = np.arange(len(wl))[::-1] if order == 'descending' else np.random.default_rng(3).permutation(len(wl))
    increasing = factory(random_state=7).fit_transform(X, wavelengths=wl)
    reordered = factory(random_state=7).fit_transform(X[:, permutation], wavelengths=wl[permutation])
    np.testing.assert_allclose(reordered, increasing[:, permutation], atol=1e-12)
    np.testing.assert_array_equal(X, original)
    assert np.isfinite(reordered).all()


def test_wavelength_shift_against_linear_interpolation_reference():
    wl = np.linspace(1000, 1100, 51)
    X = np.array([wl, 2 * wl])
    result = WavelengthShift(shift_range=(4, 4)).fit_transform(X[:, ::-1], wavelengths=wl[::-1])
    expected = np.array([np.interp(wl - 4, wl, row) for row in X])
    np.testing.assert_allclose(result[:, ::-1], expected)


@pytest.mark.parametrize('X', [np.random.default_rng(4).normal(size=(30, 3)), np.ones((30, 3))])
def test_kmeans_exact_disjoint_exhaustive_split(X):
    train, test = next(KMeansSplitter(test_size=.3, random_state=5).split(X))
    assert len(train) == len(set(train)) == 21
    assert len(test) == 9
    np.testing.assert_array_equal(np.sort(np.r_[train, test]), np.arange(30))
    centers = KMeans(n_clusters=21, n_init=10, random_state=5).fit(X).cluster_centers_
    available = list(range(30))
    expected = []
    for center in centers:
        chosen = min(available, key=lambda idx: np.linalg.norm(X[idx] - center))
        expected.append(chosen)
        available.remove(chosen)
    np.testing.assert_array_equal(train, expected)


@pytest.mark.parametrize('fraction', [.5, .25, 1/3])
def test_split_seed_ratio_and_global_rng(fraction):
    X = np.random.default_rng(8).normal(size=(40, 5))
    state = np.random.get_state()
    first = next(SPlitSplitter(fraction, random_state=9).split(X))
    second = next(SPlitSplitter(fraction, random_state=9).split(X))
    np.testing.assert_array_equal(first[1], second[1])
    assert len(first[1]) == int(np.ceil(40 * fraction))
    np.testing.assert_array_equal(np.sort(np.r_[first[0], first[1]]), np.arange(40))
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


@pytest.mark.parametrize('fraction', [.3, .4, .15, .6, 0, np.nan])
def test_split_refuses_unsupported_ratios(fraction):
    with pytest.raises(ValueError, match='test_size must be 1/r'):
        next(SPlitSplitter(fraction).split(np.arange(100).reshape(20, 5)))


def test_spxy_pca_uses_original_target_distance():
    X = np.random.default_rng(4).normal(size=(30, 5))
    y = X[:, 0]**2 + .3 * X[:, 1]
    Xt = PCA(3).fit_transform(X)
    dx, dy = cdist(Xt, Xt), cdist(y[:, None], y[:, None])
    distances = dx / dx.max() + dy / dy.max()
    selected = list(np.unravel_index(np.argmax(distances), distances.shape))
    remaining = [i for i in range(30) if i not in selected]
    while len(selected) < 21:
        next_idx = max(remaining, key=lambda i: min(distances[j, i] for j in selected))
        selected.append(next_idx)
        remaining.remove(next_idx)
    train, test = next(SPXYSplitter(.3, pca_components=3).split(X, y))
    np.testing.assert_array_equal(train, selected)
    np.testing.assert_array_equal(test, remaining)


def test_circular_split_does_not_modify_python_global_rng():
    X = np.arange(100).reshape(20, 5)
    state = random.getstate()
    train, test = next(SystematicCircularSplitter(random_state=12).split(X, np.arange(20)))
    assert random.getstate() == state
    repeated = next(SystematicCircularSplitter(random_state=12).split(X, np.arange(20)))
    np.testing.assert_array_equal(train, repeated[0])
    np.testing.assert_array_equal(np.sort(np.r_[train, test]), np.arange(20))


def test_edge_current_params_and_seeded_clones():
    wl = np.linspace(1000, 2500, 301)
    X = np.ones((5, len(wl)))
    aug = EdgeArtifactsAugmenter(random_state=12)
    aug.transform(X, wavelengths=wl)
    aug.set_params(random_state=18, overall_strength=.4, detector_roll_off=False)
    changed = aug.transform(X, wavelengths=wl)
    np.testing.assert_allclose(changed, clone(aug).transform(X, wavelengths=wl))
    assert not hasattr(aug, '_detector_aug')
    different = clone(aug).set_params(random_state=19).transform(X, wavelengths=wl)
    assert not np.allclose(changed, different)


def test_repetition_explicit_lists_fail_clearly_and_templates_work():
    with pytest.raises(ValueError, match='pp_names lists are unsupported'):
        RepetitionConfig(pp_names=['a', 'b'])
    assert RepetitionConfig(pp_names='{pp}_r{i}').get_pp_name(1, 'snv') == 'snv_r1'


@pytest.mark.parametrize('mode', ['self', 'global_mean'])
def test_scatter_reference_analytic_model(mode):
    X = np.array([[1., 2., 3.], [3., 5., 8.]])
    query = X + .3
    aug = ScatterSimulationMSC(mode, a_range=(.2, .2), b_range=(1.4, 1.4)).fit(X)
    reference = query if mode == 'self' else X.mean(axis=0)
    np.testing.assert_allclose(aug.transform(query), query + .2 + .4 * reference)


def test_scatter_refuses_unknown_reference():
    with pytest.raises(ValueError, match='reference_mode'):
        ScatterSimulationMSC('unknown').fit(np.ones((3, 4)))


@pytest.mark.parametrize('method', ['mahalanobis', 'pca_residual', 'pca_leverage'])
def test_outlier_statistical_defaults_are_preserved(method):
    X = np.random.default_rng(3).normal(size=(80, 5))
    filt = XOutlierFilter(method=method, n_components=3).fit(X)
    expected = np.sqrt(chi2.ppf(.975, 3)) if method == 'mahalanobis' else np.percentile(filt._distances_, 95)
    assert filt.threshold_ == pytest.approx(expected)


def test_sparse_bins_have_explicit_supported_input_limitation():
    X = np.arange(100).reshape(20, 5)
    y = np.r_[np.zeros(19), 100.]
    with pytest.raises(ValueError, match='least populated class'):
        next(KBinsStratifiedSplitter().split(X, y))
