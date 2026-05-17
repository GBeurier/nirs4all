"""End-to-end FastAOM PLS-Ridge orchestrator.

Bundles the four-stage pipeline:

  1. Build a small bank of nonlinear bases (``RawBase``, ``SNVBase``, ...).
  2. Build a primitive operator bank and enumerate linear chains up
     to depth 4 under the FastAOM grammar.
  3. Run fast covariance screening on the centred bases (truncated SVD)
     and keep the top-``K`` candidates per family / per base / globally.
  4. Fit a chosen AOM-model on the surviving candidate pool.

The class is sklearn-style with ``fit`` and ``predict``. The model
choice is controlled by ``model``: ``"single_chain"``,
``"hard_aom_chain"``, ``"soft_aom_chain"``, or ``"sparse_mkr"``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from nirs4all.operators.models._aom_nirs.fast.bases import BaseTransform, build_base_bank
from nirs4all.operators.models._aom_nirs.fast.chain_generator import ChainGenerationConfig, generate_chains
from nirs4all.operators.models._aom_nirs.fast.grammar import ChainGrammar, default_grammar
from nirs4all.operators.models._aom_nirs.fast.lowrank import LowRankBase, fit_lowrank_bases
from nirs4all.operators.models._aom_nirs.fast.operator_chain import OperatorChain
from nirs4all.operators.models._aom_nirs.fast.screening import (
    ScreeningCandidate,
    diversity_topk,
    fast_covariance_screen,
)

from .hard_aom_chain_pls_ridge import HardAOMChainPLSRidge
from .single_chain_pls_ridge import SingleChainPLSRidge
from .sparse_multi_kernel_ridge import SparseMultiKernelRidge
from .soft_aom_chain_pls_ridge import SoftAOMChainPLSRidge


ModelChoice = Literal["single_chain", "hard_aom_chain", "soft_aom_chain", "sparse_mkr"]


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass
class FastAOMConfig:
    """Configuration for :class:`FastAOMPLSRidge`.

    Attributes:
        model: Final AOM model to fit on the surviving candidate pool.
        primitive_bank: Bank name to draw chain primitives from (e.g.
            ``"compact"`` or ``"default"`` from ``aompls.banks``).
        max_chain_depth: Maximum chain depth.
        chain_beam_width: Optional beam-width cap for chain generation.
        rank: Truncated SVD rank for the low-rank evaluator.
        top_global: Global top-K for the diversity filter.
        top_per_base: Per-base top-K.
        top_per_family: Per-family-tag top-K (diversity).
        n_components: Maximum number of latent PLS components in the
            final AOM model.
        lambdas: Ridge ``lambda_0`` grid.
        component_shrinkage_gamma: Component-wise shrinkage exponent
            (``lambda_h = lambda_0 * h ** gamma``).
        soft_rho: L1 strength for the soft-AOM variant.
        sparse_mkr_max_chains: Maximum number of chains for sparse
            multi-kernel Ridge.
        use_raw / use_absorbance / use_snv / use_msc / use_emsc / asls_grid:
            Toggles for the nonlinear base bank.
        random_state: Reproducibility seed (currently used for tie-breaking).
    """

    model: ModelChoice = "hard_aom_chain"
    primitive_bank: str = "compact"
    max_chain_depth: int = 3
    chain_beam_width: Optional[int] = None
    rank: int = 200
    top_global: int = 200
    top_per_base: Optional[int] = 60
    top_per_family: Optional[int] = 6
    n_components: int = 15
    lambdas: Tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    component_shrinkage_gamma: Optional[float] = None
    soft_rho: float = 0.05
    soft_max_mixture_size: Optional[int] = 4
    sparse_mkr_max_chains: int = 8
    use_raw: bool = True
    use_absorbance: bool = False
    use_snv: bool = True
    use_msc: bool = False
    use_emsc: bool = False
    asls_grid: Optional[Tuple[Tuple[float, float], ...]] = None
    osc_components: Optional[Tuple[int, ...]] = None
    use_snv_osc: bool = False
    whittaker_baseline_lam: Optional[Tuple[float, ...]] = None
    random_state: int = 0
    # CV-based n_components selection (currently only honoured by
    # ``SingleChainPLSRidge`` — the AOM/MKR variants have their own component
    # selection policies).
    cv_n_components: bool = False
    cv_folds: int = 5

    def __post_init__(self) -> None:
        _validate_positive_int("n_components", self.n_components)
        _validate_positive_int("rank", self.rank)
        _validate_positive_int("top_global", self.top_global)
        _validate_positive_int("max_chain_depth", self.max_chain_depth)
        _validate_positive_int("sparse_mkr_max_chains", self.sparse_mkr_max_chains)


class FastAOMPLSRidge(BaseEstimator, RegressorMixin):
    """End-to-end FastAOM PLS-Ridge regressor.

    The estimator exposes the standard sklearn interface ``fit /
    predict`` and stores rich diagnostics in :attr:`diagnostics_`
    (timings, candidate counts, selected chains, etc.) for the
    benchmark scripts to log.
    """

    def __init__(self, config: Optional[FastAOMConfig] = None) -> None:
        self.config = config or FastAOMConfig()

    # ------------------------------------------------------------------ steps

    def _build_bases(self) -> List[BaseTransform]:
        cfg = self.config
        return build_base_bank(
            use_raw=cfg.use_raw,
            use_absorbance=cfg.use_absorbance,
            use_snv=cfg.use_snv,
            use_msc=cfg.use_msc,
            use_emsc=cfg.use_emsc,
            asls_grid=list(cfg.asls_grid) if cfg.asls_grid else None,
            osc_components=list(cfg.osc_components) if cfg.osc_components else None,
            use_snv_osc=cfg.use_snv_osc,
            use_whittaker_baseline=list(cfg.whittaker_baseline_lam) if cfg.whittaker_baseline_lam else None,
        )

    def _build_chains(self, feature_dim: int) -> List[OperatorChain]:
        from nirs4all.operators.models._aom_nirs.pls.banks import bank_by_name
        primitive_bank = bank_by_name(self.config.primitive_bank, p=feature_dim)
        cfg = ChainGenerationConfig(
            max_depth=self.config.max_chain_depth,
            include_identity_chain=True,
            beam_width=self.config.chain_beam_width,
        )
        grammar = default_grammar(max_depth=self.config.max_chain_depth)
        return generate_chains(primitive_bank, grammar, cfg, feature_dim=feature_dim)

    # ------------------------------------------------------------------ fit

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FastAOMPLSRidge":
        cfg = self.config
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y row counts must match")

        # Constant-y guard: fall back to the mean predictor.
        y_var = float(np.var(y))
        if y_var < 1e-12:
            self.diagnostics_ = {
                "fallback_constant_y": True,
                "y_mean": float(y.mean()),
                "n_chains_enumerated": 0,
                "n_candidates_total": 0,
                "n_finalists": 0,
                "model": cfg.model,
                "primitive_bank": cfg.primitive_bank,
                "timings_s": {"total": 0.0},
            }
            self._constant_prediction = float(y.mean())
            return self

        timings: dict = {}
        t0 = time.perf_counter()

        # 1. Bases
        bases = self._build_bases()
        timings["bases"] = time.perf_counter() - t0

        # 2. Chains
        t1 = time.perf_counter()
        chains = self._build_chains(feature_dim=X.shape[1])
        timings["chain_generation"] = time.perf_counter() - t1
        chain_count = len(chains)

        # 3. Low-rank evaluator
        t2 = time.perf_counter()
        lowrank = fit_lowrank_bases(bases, X, y, rank=cfg.rank, random_state=cfg.random_state)
        timings["lowrank_fit"] = time.perf_counter() - t2

        # 4. Screening
        t3 = time.perf_counter()
        candidates = fast_covariance_screen(lowrank, chains)
        timings["screening"] = time.perf_counter() - t3

        finalists = diversity_topk(
            candidates,
            top_k_global=cfg.top_global,
            top_k_per_family=cfg.top_per_family,
            top_k_per_base=cfg.top_per_base,
        )
        cand_pairs = [(c.base_index, c.chain) for c in finalists]
        if not cand_pairs:
            raise RuntimeError("Screening produced no finalist candidates")

        # 5. Final AOM model
        t4 = time.perf_counter()
        if cfg.model == "single_chain":
            best = finalists[0]
            model = SingleChainPLSRidge(
                base=bases[best.base_index],
                chain=best.chain,
                n_components=cfg.n_components,
                lambdas=cfg.lambdas,
                component_shrinkage_gamma=cfg.component_shrinkage_gamma,
                cv_n_components=cfg.cv_n_components,
                cv_folds=cfg.cv_folds,
                cv_random_state=cfg.random_state,
            )
            # Refit base from scratch on the *fold*, ensuring it has the same training-fold params
            # as the screening step. SingleChainPLSRidge re-fits internally.
            model.fit(X, y)
        elif cfg.model == "hard_aom_chain":
            model = HardAOMChainPLSRidge(
                bases=bases,
                lowrank_bases=lowrank,
                candidates=cand_pairs,
                n_components=cfg.n_components,
                lambdas=cfg.lambdas,
                component_shrinkage_gamma=cfg.component_shrinkage_gamma,
            )
            model.fit(X_train=X, y_train=y)
        elif cfg.model == "soft_aom_chain":
            model = SoftAOMChainPLSRidge(
                bases=bases,
                lowrank_bases=lowrank,
                candidates=cand_pairs,
                n_components=cfg.n_components,
                rho=cfg.soft_rho,
                lambdas=cfg.lambdas,
                component_shrinkage_gamma=cfg.component_shrinkage_gamma,
                max_mixture_size=cfg.soft_max_mixture_size,
            )
            model.fit(X_train=X, y_train=y)
        elif cfg.model == "sparse_mkr":
            model = SparseMultiKernelRidge(
                bases=bases,
                lowrank_bases=lowrank,
                candidates=cand_pairs,
                max_chains=cfg.sparse_mkr_max_chains,
                lambdas=cfg.lambdas,
            )
            model.fit(X_train=X, y_train=y)
        else:
            raise ValueError(f"Unknown model choice: {cfg.model!r}")
        timings["model_fit"] = time.perf_counter() - t4
        timings["total"] = time.perf_counter() - t0

        # Per-component selected chains (for the AOM variants that have them).
        per_component_chains: List[dict] = []
        if hasattr(model, "components_"):
            comps = getattr(model, "components_")
            for h, comp in enumerate(comps):
                if hasattr(comp, "chain"):
                    entry = {
                        "component": h,
                        "base_index": int(getattr(comp, "base_index", -1)),
                        "base_name": getattr(comp, "base_name", ""),
                        "chain_signature": comp.chain.signature,
                        "score": float(getattr(comp, "score", 0.0)),
                    }
                    # Sparse MKR also has a per-chain ``theta`` weight; record it.
                    if hasattr(comp, "theta"):
                        entry["theta"] = float(getattr(comp, "theta", 0.0))
                    per_component_chains.append(entry)
                elif hasattr(comp, "chains"):  # SoftAOMComponent has a list of chains
                    per_component_chains.append({
                        "component": h,
                        "base_indices": [int(b) for b in comp.base_indices],
                        "base_names": list(getattr(comp, "base_names", [])),
                        "chain_signatures": [c.signature for c in comp.chains],
                        "weights": [float(w) for w in comp.weights],
                    })

        # Persist diagnostics
        self.diagnostics_ = {
            "n_chains_enumerated": chain_count,
            "n_candidates_total": len(candidates),
            "n_finalists": len(finalists),
            "model": cfg.model,
            "primitive_bank": cfg.primitive_bank,
            "rank": cfg.rank,
            "max_chain_depth": cfg.max_chain_depth,
            "selected_chain_signatures": [c.chain.signature for c in finalists[:25]],
            "selected_base_indices": [c.base_index for c in finalists[:25]],
            "per_component_chains": per_component_chains,
            "timings_s": timings,
            "top_finalist_score": float(finalists[0].score) if finalists else 0.0,
        }
        self.bases_ = bases
        self.lowrank_bases_ = lowrank
        self.finalists_ = finalists
        self.model_ = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self, "_constant_prediction"):
            return np.full(np.asarray(X).shape[0], self._constant_prediction, dtype=float)
        if not hasattr(self, "model_"):
            raise RuntimeError("FastAOMPLSRidge.predict called before fit")
        return self.model_.predict(X)
