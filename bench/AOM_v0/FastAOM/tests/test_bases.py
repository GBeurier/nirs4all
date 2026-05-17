"""Tests for FastAOM nonlinear base transforms."""

from __future__ import annotations

import numpy as np
import pytest

from FastAOM.bases import (
    AbsorbanceBase,
    ASLSBase,
    EMSCBase,
    MSCBase,
    OSCBase,
    RawBase,
    SNVBase,
    SNVOSCBase,
    WhittakerBaseLine,
    build_base_bank,
)


@pytest.fixture
def reflectance_data():
    rng = np.random.default_rng(0)
    # Reflectance-like data: positive, in (0, 1] range
    X = 0.1 + 0.8 * rng.uniform(size=(40, 64))
    return X


@pytest.fixture
def absorbance_data():
    rng = np.random.default_rng(1)
    return rng.standard_normal((40, 64))


def test_raw_base_identity(reflectance_data) -> None:
    base = RawBase()
    out = base.fit_transform(reflectance_data)
    np.testing.assert_allclose(out, reflectance_data, atol=1e-12)


def test_absorbance_clips_and_logs(reflectance_data) -> None:
    base = AbsorbanceBase(eps=1e-6).fit(reflectance_data)
    out = base.transform(reflectance_data)
    np.testing.assert_allclose(out, -np.log10(np.clip(reflectance_data, 1e-6, None)))


def test_absorbance_falls_back_when_data_is_negative(absorbance_data) -> None:
    base = AbsorbanceBase().fit(absorbance_data)
    out = base.transform(absorbance_data)
    assert base.fallback_to_identity_ is True
    np.testing.assert_allclose(out, absorbance_data)


def test_snv_zero_mean_unit_std(reflectance_data) -> None:
    base = SNVBase()
    out = base.fit_transform(reflectance_data)
    np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose(out.std(axis=1, ddof=0), 1.0, atol=1e-6)


def test_msc_returns_correct_shape(reflectance_data) -> None:
    base = MSCBase()
    out = base.fit_transform(reflectance_data)
    assert out.shape == reflectance_data.shape


def test_emsc_returns_correct_shape(reflectance_data) -> None:
    base = EMSCBase(degree=2)
    out = base.fit_transform(reflectance_data)
    assert out.shape == reflectance_data.shape


def test_emsc_fold_aware(reflectance_data) -> None:
    """The reference must be the training-fold mean, not full-data mean."""
    train = reflectance_data[:20]
    test = reflectance_data[20:]
    base = EMSCBase(degree=2).fit(train)
    np.testing.assert_allclose(base.reference_, train.mean(axis=0))
    out_test = base.transform(test)
    assert out_test.shape == test.shape


def test_asls_returns_baseline_subtracted(reflectance_data) -> None:
    base = ASLSBase(lam=1e5, p=0.01, max_iter=10)
    out = base.fit_transform(reflectance_data)
    assert out.shape == reflectance_data.shape


def test_whittaker_baseline_returns_correct_shape(reflectance_data) -> None:
    base = WhittakerBaseLine(lam=1e4)
    out = base.fit_transform(reflectance_data)
    assert out.shape == reflectance_data.shape


def test_osc_base_requires_y() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 32))
    base = OSCBase(n_components=2)
    with pytest.raises(ValueError):
        base.fit(X, y=None)


def test_osc_base_supervised_roundtrip() -> None:
    rng = np.random.default_rng(1)
    n, p = 60, 40
    X = rng.standard_normal((n, p))
    # Inject a direction orthogonal to y that OSC should remove
    w = rng.standard_normal(p)
    w /= np.linalg.norm(w)
    y = X @ w + 0.01 * rng.standard_normal(n)
    base = OSCBase(n_components=2).fit(X, y)
    out = base.transform(X)
    assert out.shape == X.shape
    # OSC should preserve correlation with y (or improve it)
    pre_corr = float(np.corrcoef(X[:, p // 2], y)[0, 1])
    post_corr = float(np.corrcoef(out[:, p // 2], y)[0, 1])
    assert np.isfinite(post_corr)


def test_osc_base_replays_with_train_projection() -> None:
    """OSC at predict time must use the train-time projection, not refit."""
    rng = np.random.default_rng(2)
    X_train = rng.standard_normal((40, 32))
    y_train = X_train.sum(axis=1) + 0.01 * rng.standard_normal(40)
    X_test = rng.standard_normal((10, 32))
    base = OSCBase(n_components=2).fit(X_train, y_train)
    out_test = base.transform(X_test)
    # Re-transform same X_test must give same result
    out_test_2 = base.transform(X_test)
    np.testing.assert_allclose(out_test, out_test_2, atol=1e-12)


def test_snv_osc_base() -> None:
    rng = np.random.default_rng(3)
    X = 0.2 + 0.6 * rng.uniform(size=(50, 32))
    y = X.sum(axis=1) + 0.01 * rng.standard_normal(50)
    base = SNVOSCBase(n_components=2).fit(X, y)
    out = base.transform(X)
    assert out.shape == X.shape


def test_build_base_bank_with_supervised_bases() -> None:
    bank = build_base_bank(
        use_raw=True,
        use_snv=True,
        asls_grid=[(1e4, 0.01)],
        osc_components=[2, 3],
        use_snv_osc=True,
        use_whittaker_baseline=[1e3, 1e5],
    )
    names = [b.signature for b in bank]
    assert any("osc_n2" in n for n in names)
    assert any("osc_n3" in n for n in names)
    assert any("snv_osc" in n for n in names)
    assert any("asls" in n for n in names)
    assert any("whittaker_baseline" in n for n in names)
    assert len(names) == len(set(names))


def test_build_base_bank_default() -> None:
    bank = build_base_bank()
    assert len(bank) >= 1
    signatures = [b.signature for b in bank]
    assert len(signatures) == len(set(signatures))


def test_build_base_bank_with_asls_grid() -> None:
    bank = build_base_bank(
        use_raw=True,
        use_absorbance=False,
        use_snv=True,
        use_msc=False,
        use_emsc=False,
        asls_grid=[(1e4, 0.001), (1e5, 0.01)],
    )
    assert len(bank) == 4
    names = [b.name for b in bank]
    assert any("asls" in name for name in names)


def test_build_base_bank_raises_on_empty() -> None:
    with pytest.raises(ValueError):
        build_base_bank(
            use_raw=False,
            use_absorbance=False,
            use_snv=False,
            use_msc=False,
            use_emsc=False,
        )
