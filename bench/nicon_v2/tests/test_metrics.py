import numpy as np
import pytest

from nicon_v2 import metrics as m


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_rmse_against_reference(seed):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=64)
    pred = y + rng.normal(scale=0.1, size=64)
    expected = float(np.sqrt(np.mean((y - pred) ** 2)))
    assert m.rmse(y, pred) == pytest.approx(expected)


def test_rmse_zero_when_perfect():
    y = np.linspace(0.0, 1.0, 16)
    assert m.rmse(y, y.copy()) == pytest.approx(0.0)


def test_r2_matches_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.normal(size=128)
    pred = y + rng.normal(scale=0.5, size=128)
    assert m.r2(y, pred) == pytest.approx(sklearn.r2_score(y, pred))


def test_r2_constant_truth_returns_nan():
    y = np.zeros(8)
    pred = np.zeros(8)
    assert np.isnan(m.r2(y, pred))


def test_gaussian_nll_homoscedastic():
    rng = np.random.default_rng(0)
    y = rng.normal(size=2048)
    mu = np.zeros_like(y)
    sigma = np.ones_like(y)
    expected = float(0.5 * np.mean(y ** 2 + np.log(2.0 * np.pi)))
    assert m.gaussian_nll(y, mu, sigma) == pytest.approx(expected, rel=1e-3)


def test_coverage_at_alpha_perfect():
    y = np.linspace(0.0, 1.0, 100)
    lo = y - 0.01
    hi = y + 0.01
    assert m.coverage_at_alpha(y, lo, hi) == pytest.approx(1.0)


def test_relative_rmsep_signed_change():
    assert m.relative_rmsep(1.0, 2.0) == pytest.approx(-0.5)
    assert m.relative_rmsep(3.0, 2.0) == pytest.approx(0.5)
    assert m.relative_rmsep(1.0, 0.0) is None
    assert m.relative_rmsep(1.0, None) is None


def test_rmse_shape_mismatch_raises():
    with pytest.raises(ValueError):
        m.rmse(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_mae_shape_mismatch_raises():
    with pytest.raises(ValueError):
        m.mae(np.array([1.0]), np.array([1.0, 2.0]))


def test_mae_matches_reference():
    rng = np.random.default_rng(0)
    y = rng.normal(size=64)
    pred = y + rng.normal(scale=0.5, size=64)
    expected = float(np.mean(np.abs(y - pred)))
    assert m.mae(y, pred) == pytest.approx(expected)


def test_metrics_empty_input_returns_nan():
    empty = np.array([], dtype=float)
    assert np.isnan(m.rmse(empty, empty))
    assert np.isnan(m.mae(empty, empty))
    assert np.isnan(m.r2(empty, empty))
    assert np.isnan(m.bias(empty, empty))
    assert np.isnan(m.gaussian_nll(empty, empty, empty))
    assert np.isnan(m.coverage_at_alpha(empty, empty, empty))
    assert np.isnan(m.width_at_alpha(empty, empty))


def test_bias_signed():
    y = np.array([1.0, 2.0, 3.0])
    pred = y + 0.5
    assert m.bias(y, pred) == pytest.approx(0.5)
    pred2 = y - 0.3
    assert m.bias(y, pred2) == pytest.approx(-0.3)


def test_width_at_alpha_mean_width():
    lo = np.array([0.0, 1.0, 2.0])
    hi = np.array([1.0, 3.0, 6.0])
    expected = float(np.mean(hi - lo))
    assert m.width_at_alpha(lo, hi) == pytest.approx(expected)


def test_coverage_at_alpha_partial():
    y = np.array([0.5, 1.5, 2.5, 3.5])
    lo = np.array([0.0, 1.0, 3.0, 4.0])
    hi = np.array([1.0, 2.0, 4.0, 5.0])
    # samples 0, 1 covered; sample 2 (2.5) not in [3, 4]; sample 3 (3.5) not in [4, 5]
    assert m.coverage_at_alpha(y, lo, hi) == pytest.approx(0.5)


def test_gaussian_nll_clamps_small_sigma():
    # sigma=0 would normally crash; we clip to 1e-6
    y = np.array([0.0])
    mu = np.array([0.0])
    sigma = np.array([0.0])
    val = m.gaussian_nll(y, mu, sigma)
    assert np.isfinite(val)
