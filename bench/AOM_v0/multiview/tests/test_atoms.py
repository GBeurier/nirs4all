"""Tests for the canonical Phase-11 atoms surfaced in :mod:`multiview.atoms`."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from multiview import LazyV2AOM as TopLevelLazyV2AOM
from multiview.atoms import (
    AOMMoEMultiK,
    AOMMoERegressor,
    AOMPLSRegressor,
    LazyV2AOM,
)


def _block_signal(n=120, p=96, signal=(20, 60), seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    s, e = signal
    weights = rng.standard_normal(e - s)
    y = X[:, s:e] @ weights + 0.05 * rng.standard_normal(n)
    return X, y


class TestPublicSurface:
    def test_top_level_re_export_is_same_object(self):
        assert TopLevelLazyV2AOM is LazyV2AOM

    def test_lazy_v2_aom_is_sklearn_regressor(self):
        from sklearn.base import is_regressor

        assert is_regressor(LazyV2AOM())

    def test_canonical_atoms_have_fit_predict(self):
        for cls in (LazyV2AOM, AOMMoEMultiK, AOMMoERegressor, AOMPLSRegressor):
            inst = cls()
            assert hasattr(inst, "fit"), f"{cls.__name__} missing fit"
            assert hasattr(inst, "predict"), f"{cls.__name__} missing predict"


class TestLazyV2AOM:
    def test_fit_predict_shape(self):
        X, y = _block_signal(n=80, p=60)
        est = LazyV2AOM(max_components=4, K=3, random_state=0)
        est.fit(X, y)
        pred = est.predict(X)
        assert pred.shape == y.shape
        assert hasattr(est, "estimator_")
        assert isinstance(est.estimator_, AOMPLSRegressor)

    def test_predict_recovers_signal(self):
        X, y = _block_signal(n=160, p=80, signal=(20, 50), seed=1)
        est = LazyV2AOM(max_components=8, K=3, random_state=1)
        est.fit(X, y)
        pred = est.predict(X)
        # On-train R^2 must be substantially positive on a clean signal.
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        assert r2 > 0.5, f"on-train R^2={r2:.3f} unexpectedly low"

    def test_clone_preserves_hyperparameters(self):
        original = LazyV2AOM(
            max_components=12, K=5, bank_name="compact",
            include_global=False, random_state=42,
        )
        cloned = clone(original)
        assert cloned.get_params() == original.get_params()
        # Clone must not carry fitted state.
        assert not hasattr(cloned, "estimator_")

    def test_get_set_params_roundtrip(self):
        est = LazyV2AOM()
        params = est.get_params()
        for key in (
            "max_components", "K", "bank_name", "strategy",
            "include_global", "engine", "selection", "criterion",
            "random_state",
        ):
            assert key in params, f"missing parameter: {key}"
        est.set_params(max_components=20, K=7)
        assert est.max_components == 20
        assert est.K == 7

    def test_matches_original_factory(self):
        """LazyV2AOM(...) must produce the same predictions as the
        ``_build_lazy_v2_aom`` factory it replaced.

        The factory built the bank from `p=X.shape[1]` and instantiated
        AOMPLSRegressor with the V2 configuration; LazyV2AOM does the
        same in :meth:`fit`. This test exercises the equivalence.
        """
        from multiview.views import ViewBuilder

        X, y = _block_signal(n=100, p=64, seed=2)
        seed = 7
        max_components = 6

        # Replicate the original factory inline.
        bank = ViewBuilder.combined(
            bank_name="compact", K=3, strategy="equal_width",
            include_global=True,
        ).build(p=X.shape[1])
        factory_est = AOMPLSRegressor(
            n_components="auto", max_components=max_components,
            engine="simpls_covariance", selection="global",
            criterion="holdout", operator_bank=bank, random_state=seed,
        )
        factory_est.fit(X, y)
        factory_pred = factory_est.predict(X)

        lazy_est = LazyV2AOM(
            max_components=max_components, K=3, bank_name="compact",
            strategy="equal_width", include_global=True,
            engine="simpls_covariance", selection="global",
            criterion="holdout", random_state=seed,
        )
        lazy_est.fit(X, y)
        lazy_pred = lazy_est.predict(X)

        np.testing.assert_allclose(lazy_pred, factory_pred, rtol=1e-10, atol=1e-10)


class TestCanonicalAtomConfigurations:
    """Smoke-tests for the four Phase-11 atom configurations."""

    def test_multiK_3_5_7(self):
        X, y = _block_signal(n=100, p=64, seed=3)
        est = AOMMoEMultiK(K_list=(3, 5, 7), per_expert_components=4, random_state=0)
        est.fit(X, y)
        assert est.predict(X).shape == y.shape

    def test_moe_preproc_soft(self):
        X, y = _block_signal(n=100, p=64, seed=4)
        est = AOMMoERegressor(
            expert_layout="per_preproc", routing="soft",
            bank_name="compact", per_expert_components=4, random_state=0,
        )
        est.fit(X, y)
        assert est.predict(X).shape == y.shape

    def test_aom_pls_compact(self):
        X, y = _block_signal(n=100, p=64, seed=5)
        est = AOMPLSRegressor(
            n_components="auto", max_components=4,
            engine="simpls_covariance", selection="global",
            criterion="holdout", operator_bank="compact", random_state=0,
        )
        est.fit(X, y)
        assert est.predict(X).shape == y.shape


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lazy_v2_aom_seed_determinism(seed):
    """Two LazyV2AOM with the same seed must produce identical predictions."""
    X, y = _block_signal(n=60, p=80, signal=(10, 30), seed=seed)
    a = LazyV2AOM(max_components=4, K=3, random_state=seed).fit(X, y).predict(X)
    b = LazyV2AOM(max_components=4, K=3, random_state=seed).fit(X, y).predict(X)
    np.testing.assert_allclose(a, b, rtol=0, atol=0)
