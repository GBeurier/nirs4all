"""Tests for the FastAOM model family on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from aompls.banks import compact_bank
from aompls.operators import (
    DetrendProjectionOperator,
    FiniteDifferenceOperator,
    IdentityOperator,
    SavitzkyGolayOperator,
)
from FastAOM.bases import MSCBase, OSCBase, RawBase, SNVBase, SNVOSCBase
from FastAOM.chain_generator import ChainGenerationConfig, generate_chains
from FastAOM.grammar import default_grammar
from FastAOM.lowrank import fit_lowrank_bases
from FastAOM.models import (
    FastAOMConfig,
    FastAOMPLSRidge,
    HardAOMChainPLSRidge,
    SingleChainPLSRidge,
    SoftAOMChainPLSRidge,
    SparseMultiKernelRidge,
)
from FastAOM.operator_chain import OperatorChain
from FastAOM.screening import diversity_topk, fast_covariance_screen


def _make_data(seed: int = 0, n_train: int = 60, n_test: int = 30, p: int = 80):
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test
    X = 0.2 + 0.6 * rng.uniform(size=(n_total, p))
    # Construct y as a smoothed-derivative signal of X plus a linear baseline.
    kernel = np.zeros(p)
    kernel[40:60] = np.linspace(0.0, 1.0, 20)
    y = X @ kernel + 0.05 * rng.standard_normal(n_total)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    return X_train, X_test, y_train, y_test


def _make_candidates(X_train, y_train, p):
    bases = [RawBase(), SNVBase(), MSCBase()]
    lowrank = fit_lowrank_bases(bases, X_train, y_train, rank=30)
    primitive_bank = compact_bank(p=p)
    chains = generate_chains(
        primitive_bank,
        default_grammar(max_depth=3),
        ChainGenerationConfig(max_depth=3, include_identity_chain=True),
        feature_dim=p,
    )
    cand = fast_covariance_screen(lowrank, chains)
    finalists = diversity_topk(cand, top_k_global=40, top_k_per_family=2)
    cand_pairs = [(c.base_index, c.chain) for c in finalists]
    return bases, lowrank, finalists, cand_pairs


def test_single_chain_pls_ridge_runs() -> None:
    X_tr, X_te, y_tr, y_te = _make_data()
    chain = OperatorChain([SavitzkyGolayOperator(11, 2, 1, p=X_tr.shape[1])])
    model = SingleChainPLSRidge(base=RawBase(), chain=chain, n_components=8)
    model.fit(X_tr, y_tr)
    yhat = model.predict(X_te)
    assert yhat.shape == y_te.shape
    rmse = float(np.sqrt(np.mean((y_te - yhat) ** 2)))
    # Random baseline is ~std(y_te); we want at least a modest improvement.
    baseline_rmse = float(np.sqrt(np.mean((y_te - y_tr.mean()) ** 2)))
    assert rmse < baseline_rmse, f"single-chain RMSE {rmse:.4f} >= baseline {baseline_rmse:.4f}"


def test_single_chain_pls_ridge_cv_n_components() -> None:
    """When ``cv_n_components=True`` the model picks an integer k in {1..max}."""
    X_tr, X_te, y_tr, y_te = _make_data(seed=7, n_train=100)
    chain = OperatorChain([SavitzkyGolayOperator(11, 2, 1, p=X_tr.shape[1])])
    model = SingleChainPLSRidge(
        base=RawBase(),
        chain=chain,
        n_components=12,
        cv_n_components=True,
        cv_folds=5,
        cv_random_state=0,
    )
    model.fit(X_tr, y_tr)
    # Selected k must be in [1, 12]
    assert 1 <= model.n_components_ <= 12
    # And model must still beat mean baseline on test
    yhat = model.predict(X_te)
    rmse = float(np.sqrt(np.mean((y_te - yhat) ** 2)))
    baseline = float(np.sqrt(np.mean((y_te - y_tr.mean()) ** 2)))
    assert rmse < baseline


def test_single_chain_pls_ridge_supervised_base_requires_y() -> None:
    """OSCBase requires ``y`` at fit time. The ``SingleChainPLSRidge.fit``
    must thread ``y`` through ``base.fit_transform`` — otherwise OSC
    raises and supervised variants are unusable."""
    X_tr, X_te, y_tr, y_te = _make_data(seed=11, n_train=80)
    chain = OperatorChain([SavitzkyGolayOperator(11, 2, 1, p=X_tr.shape[1])])
    model = SingleChainPLSRidge(base=OSCBase(n_components=2), chain=chain, n_components=8)
    # Must not raise — y is threaded
    model.fit(X_tr, y_tr)
    yhat = model.predict(X_te)
    assert yhat.shape == y_te.shape
    assert np.all(np.isfinite(yhat))


def test_single_chain_pls_ridge_with_snv_osc_base() -> None:
    X_tr, X_te, y_tr, y_te = _make_data(seed=12, n_train=80)
    chain = OperatorChain([SavitzkyGolayOperator(11, 2, 0, p=X_tr.shape[1])])
    model = SingleChainPLSRidge(base=SNVOSCBase(n_components=2), chain=chain, n_components=6)
    model.fit(X_tr, y_tr)
    yhat = model.predict(X_te)
    assert yhat.shape == y_te.shape
    assert np.all(np.isfinite(yhat))


def test_single_chain_pls_ridge_cv_n_components_handles_tiny_n() -> None:
    """If n < 2*cv_folds, fall back to max n_components without crashing."""
    X_tr, X_te, y_tr, y_te = _make_data(seed=8, n_train=8)
    chain = OperatorChain([SavitzkyGolayOperator(11, 2, 1, p=X_tr.shape[1])])
    model = SingleChainPLSRidge(
        base=RawBase(),
        chain=chain,
        n_components=5,
        cv_n_components=True,
        cv_folds=5,
    )
    model.fit(X_tr, y_tr)
    assert model.n_components_ > 0


def test_single_chain_pls_ridge_component_shrinkage() -> None:
    X_tr, _, y_tr, _ = _make_data()
    chain = OperatorChain([SavitzkyGolayOperator(11, 2, 1, p=X_tr.shape[1])])
    model = SingleChainPLSRidge(
        base=RawBase(),
        chain=chain,
        n_components=8,
        component_shrinkage_gamma=1.0,
    )
    model.fit(X_tr, y_tr)
    assert model.n_components_ > 0


def test_hard_aom_chain_pls_ridge_fits_and_predicts() -> None:
    X_tr, X_te, y_tr, y_te = _make_data()
    bases, lowrank, finalists, cand_pairs = _make_candidates(X_tr, y_tr, X_tr.shape[1])
    model = HardAOMChainPLSRidge(
        bases=bases,
        lowrank_bases=lowrank,
        candidates=cand_pairs,
        n_components=8,
    )
    model.fit(X_train=X_tr, y_train=y_tr)
    assert model.n_components_ > 0
    yhat = model.predict(X_te)
    assert yhat.shape == y_te.shape
    rmse = float(np.sqrt(np.mean((y_te - yhat) ** 2)))
    baseline = float(np.sqrt(np.mean((y_te - y_tr.mean()) ** 2)))
    assert rmse < baseline


def test_hard_aom_chain_records_per_component_chain() -> None:
    X_tr, _, y_tr, _ = _make_data()
    bases, lowrank, finalists, cand_pairs = _make_candidates(X_tr, y_tr, X_tr.shape[1])
    model = HardAOMChainPLSRidge(
        bases=bases,
        lowrank_bases=lowrank,
        candidates=cand_pairs,
        n_components=4,
    )
    model.fit(X_train=X_tr, y_train=y_tr)
    assert len(model.components_) == model.n_components_
    for comp in model.components_:
        assert isinstance(comp.chain, OperatorChain)
        assert 0 <= comp.base_index < len(bases)


def test_sparse_multi_kernel_ridge_fits_and_predicts() -> None:
    X_tr, X_te, y_tr, y_te = _make_data()
    bases, lowrank, finalists, cand_pairs = _make_candidates(X_tr, y_tr, X_tr.shape[1])
    model = SparseMultiKernelRidge(
        bases=bases,
        lowrank_bases=lowrank,
        candidates=cand_pairs,
        max_chains=4,
    )
    model.fit(X_train=X_tr, y_train=y_tr)
    assert model.theta_.size > 0
    assert np.all(model.theta_ >= 0)
    yhat = model.predict(X_te)
    assert yhat.shape == y_te.shape
    rmse = float(np.sqrt(np.mean((y_te - yhat) ** 2)))
    baseline = float(np.sqrt(np.mean((y_te - y_tr.mean()) ** 2)))
    assert rmse < baseline


def test_soft_aom_chain_pls_ridge_fits_and_predicts() -> None:
    X_tr, X_te, y_tr, y_te = _make_data()
    bases, lowrank, finalists, cand_pairs = _make_candidates(X_tr, y_tr, X_tr.shape[1])
    model = SoftAOMChainPLSRidge(
        bases=bases,
        lowrank_bases=lowrank,
        candidates=cand_pairs,
        n_components=4,
        rho=0.02,
        max_mixture_size=3,
    )
    model.fit(X_train=X_tr, y_train=y_tr)
    assert model.n_components_ > 0
    yhat = model.predict(X_te)
    assert yhat.shape == y_te.shape
    rmse = float(np.sqrt(np.mean((y_te - yhat) ** 2)))
    baseline = float(np.sqrt(np.mean((y_te - y_tr.mean()) ** 2)))
    assert rmse < baseline


def test_fast_aom_pls_ridge_meta_estimator_default() -> None:
    X_tr, X_te, y_tr, y_te = _make_data()
    cfg = FastAOMConfig(
        model="hard_aom_chain",
        primitive_bank="compact",
        max_chain_depth=3,
        rank=30,
        top_global=40,
        top_per_base=15,
        top_per_family=2,
        n_components=6,
    )
    est = FastAOMPLSRidge(config=cfg)
    est.fit(X_tr, y_tr)
    yhat = est.predict(X_te)
    rmse = float(np.sqrt(np.mean((y_te - yhat) ** 2)))
    baseline = float(np.sqrt(np.mean((y_te - y_tr.mean()) ** 2)))
    assert rmse < baseline
    assert "timings_s" in est.diagnostics_
    assert est.diagnostics_["n_chains_enumerated"] > 0
    assert est.diagnostics_["n_finalists"] > 0


def test_sparse_mkr_does_not_infinite_loop_on_redundant_candidates() -> None:
    """When NNLS-refit zeroes out a freshly selected chain, that chain must be
    blacklisted; otherwise the greedy loop re-selects it forever."""
    rng = np.random.default_rng(123)
    n, p = 60, 80
    # Construct two near-identical signals so NNLS will drop redundant ones
    X = rng.standard_normal((n, p))
    y = X[:, 0] + 0.5 * X[:, 1] + 0.05 * rng.standard_normal(n)
    bases = [RawBase()]
    lowrank = fit_lowrank_bases(bases, X, y, rank=30)
    # Many slightly-different chains — most will be near-duplicates after NNLS
    chains = [
        OperatorChain([SavitzkyGolayOperator(w, 2, 0, p=p)])
        for w in (5, 7, 9, 11, 15, 21)
    ]
    cand_pairs = [(0, ch) for ch in chains]
    model = SparseMultiKernelRidge(
        bases=bases,
        lowrank_bases=lowrank,
        candidates=cand_pairs,
        max_chains=10,  # > number of useful chains
        nnls_iters=50,
        tol=1e-8,
    )
    # The fit must terminate (not loop forever)
    model.fit(X_train=X, y_train=y)
    assert model.theta_.size <= len(chains)


@pytest.mark.parametrize("model", ["single_chain", "soft_aom_chain", "sparse_mkr"])
def test_fast_aom_pls_ridge_supports_all_models(model: str) -> None:
    X_tr, X_te, y_tr, y_te = _make_data()
    cfg = FastAOMConfig(
        model=model,
        primitive_bank="compact",
        max_chain_depth=3,
        rank=25,
        top_global=20,
        top_per_base=10,
        top_per_family=2,
        n_components=4,
        sparse_mkr_max_chains=3,
        soft_max_mixture_size=2,
        soft_rho=0.05,
    )
    est = FastAOMPLSRidge(config=cfg)
    est.fit(X_tr, y_tr)
    yhat = est.predict(X_te)
    assert yhat.shape == y_te.shape
    assert np.all(np.isfinite(yhat))
