"""Block-sparse and classic MB-PLS-AOM selection algorithms.

Two algorithms (see DESIGN_MBPLS.md §3 / §3a for full math):

- `fit_block_sparse_aom`: hard-gated AOM-MBPLS where only the winning block
  participates at each LV. Block-supported components, AOM-style locality.
  Deflation: the winning block's columns are reduced by `t_a . p_{k*,a}^T`.
- `fit_classic_mbpls_aom`: Westerhuis-style super-score from all blocks +
  per-block deflation. AOM principle is in per-LV operator selection within
  each block.

Both produce `MBPLSResult` with the standard PLS coefficient assembly
`B = Z . pinv(P^T Z) . Q^T`.

Leakage-free CV scoring is delegated to the existing `cv_score_regression`
in `aompls.scorers` via a `_fp` closure that refits the prefix per fold —
the same pattern as `aompls.selection._criterion_score_at_indices`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from aompls.operators import ComposedOperator, LinearSpectralOperator
from aompls.scorers import (
    CriterionConfig,
    cv_score_regression,
    holdout_score_regression,
)


_PINV_EPS = 1e-10


@dataclass
class MBPLSResult:
    """Outcome of a block-sparse or classic MB-PLS-AOM fit.

    Attributes:
        Z: raw weights, shape (p, K_max). Each column lives in feature space
           and is block-supported (zero outside the winning block) for
           block-sparse; sums of block-supported columns for classic.
        P: block loadings, shape (p, K_max). Block-supported (zero outside
           the winning block) for block-sparse; multi-block (sum of block
           contributions) for classic.
        Q: y loadings, shape (q, K_max).
        T: super-scores, shape (n, K_max).
        op_indices: bank indices selected at each LV (block-sparse) or the
            list of per-LV {block_idx: op_idx} dicts (classic).
        block_winners: list of (block_idx, bank_op_idx) for block-sparse;
            for classic, list of dicts.
        n_components: actual K_max committed (may be less than requested).
        diagnostics: free-form info (algorithm name, criterion, scores).
    """

    Z: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    T: np.ndarray
    op_indices: List
    block_winners: List
    n_components: int
    diagnostics: dict = field(default_factory=dict)

    def coef(self) -> np.ndarray:
        """Return regression coefficient `B = Z . pinv(P^T Z) . Q^T`."""
        return self.coef_prefix(self.n_components)

    def coef_prefix(self, k: int) -> np.ndarray:
        """Coefficient using only the first k components."""
        if k <= 0:
            return np.zeros((self.Z.shape[0], self.Q.shape[0]))
        k = min(k, self.n_components)
        Z_K = self.Z[:, :k]
        P_K = self.P[:, :k]
        Q_K = self.Q[:, :k]
        PtZ = P_K.T @ Z_K
        reg = PtZ + _PINV_EPS * np.eye(k)
        return Z_K @ np.linalg.pinv(reg) @ Q_K.T


# ---------------------------------------------------------------------------
# Inner kernel: fit block-sparse with fixed operator sequence
# ---------------------------------------------------------------------------


def _fit_block_sparse_fixed(
    *,
    Xc: np.ndarray,
    yc: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    op_to_block: Sequence[int],
    block_masks: Dict[int, np.ndarray],
    indices: Sequence[int],
) -> MBPLSResult:
    """Fit the block-sparse algorithm with a pre-determined operator sequence.

    Builds the residual matrices block-sparsely as described in
    DESIGN_MBPLS.md §3.4: maintain a single `X_res`; at each LV, deflate
    only the winning block's features by `t_a . p_a^T` where `p_a` is the
    full-X loading masked to the winning block. The standard PLS coefficient
    formula `B = Z . pinv(P^T Z) . Q^T` recovers prediction-space rotations
    from the block-supported raw weights `Z` and block-supported loadings `P`.
    """
    n, p = Xc.shape
    yc2 = yc.reshape(-1, 1) if yc.ndim == 1 else yc.copy()
    q = yc2.shape[1]
    K_max = len(indices)

    X_res = Xc.copy()
    y_res = yc2.copy()

    Z = np.zeros((p, K_max))
    P = np.zeros((p, K_max))
    Q = np.zeros((q, K_max))
    T = np.zeros((n, K_max))
    block_winners: List[Tuple[int, int]] = []

    n_committed = 0
    for a, b in enumerate(indices):
        op = operators[int(b)]
        S = X_res.T @ y_res
        w_cov = op.apply_cov(S)
        w_vec = w_cov.ravel() if w_cov.ndim == 1 else w_cov[:, 0]
        w_norm = np.linalg.norm(w_vec)
        if w_norm < 1e-12:
            break
        w_vec = w_vec / w_norm
        r_vec = op.adjoint_vec(w_vec)
        t_a = X_res @ r_vec
        t_norm = float(t_a @ t_a) + 1e-12
        if t_norm < 1e-12:
            break
        k_star = int(op_to_block[int(b)])
        m_kstar = block_masks[k_star]
        p_a_full = X_res.T @ t_a / t_norm
        p_a = m_kstar * p_a_full
        q_a = y_res.T @ t_a / t_norm
        Z[:, a] = r_vec
        P[:, a] = p_a
        Q[:, a] = q_a.ravel()
        T[:, a] = t_a
        block_winners.append((k_star, int(b)))
        # Block-sparse deflation: only the winning block's features are reduced.
        X_res = X_res - np.outer(t_a, p_a)
        y_res = y_res - np.outer(t_a, q_a.ravel())
        n_committed += 1

    return MBPLSResult(
        Z=Z[:, :n_committed],
        P=P[:, :n_committed],
        Q=Q[:, :n_committed],
        T=T[:, :n_committed],
        op_indices=[int(b) for b in indices[:n_committed]],
        block_winners=block_winners[:n_committed],
        n_components=n_committed,
    )


# ---------------------------------------------------------------------------
# Candidate scoring (leakage-free CV / holdout)
# ---------------------------------------------------------------------------


def _score_block_sparse_indices(
    *,
    Xc: np.ndarray,
    yc: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    op_to_block: Sequence[int],
    block_masks: Dict[int, np.ndarray],
    indices: Sequence[int],
    criterion: CriterionConfig,
) -> float:
    """Score a fixed block-sparse prefix under the given criterion.

    Mirrors `aompls.selection._criterion_score_at_indices` for the cv /
    holdout paths: each fold re-centers training data, refits the operators,
    runs the algorithm with `indices` on the train fold, predicts on val.

    To avoid an `aompls.scorers._rmse` broadcasting bug (when y_va is 2D
    and y_pred is 1D, `(y_va - y_pred)` broadcasts to a square matrix
    giving wildly wrong RMSE), we pass 1D yc into the scorer and reshape
    back to 2D inside `_fp` only if `q > 1`.
    """
    if not indices:
        return float("inf")
    yc_arr = np.asarray(yc)
    if yc_arr.ndim == 2 and yc_arr.shape[1] == 1:
        yc_for_scoring = yc_arr.ravel()
    else:
        yc_for_scoring = yc_arr

    def _fp(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray) -> np.ndarray:
        x_mean = X_tr.mean(axis=0)
        if y_tr.ndim > 1:
            y_mean = y_tr.mean(axis=0)
        else:
            y_mean = float(y_tr.mean())
        Xtc = X_tr - x_mean
        ytc = y_tr - y_mean
        for b in set(int(i) for i in indices):
            operators[b].fit(Xtc)
        res = _fit_block_sparse_fixed(
            Xc=Xtc, yc=ytc, operators=operators,
            op_to_block=op_to_block, block_masks=block_masks,
            indices=list(indices),
        )
        if res.n_components == 0:
            n_va = X_va.shape[0]
            if isinstance(y_mean, np.ndarray):
                return np.broadcast_to(y_mean, (n_va, y_mean.shape[0])).copy()
            return np.full(n_va, y_mean)
        coef = res.coef()
        Xv = X_va - x_mean
        pred = Xv @ coef
        if pred.ndim == 2 and pred.shape[1] == 1:
            pred = pred.ravel()
        return pred + y_mean

    if criterion.kind in ("cv", "hybrid"):
        return float(cv_score_regression(
            Xc, yc_for_scoring, _fp,
            n_splits=criterion.cv,
            random_state=criterion.random_state,
            cv_splitter=criterion.cv_splitter,
        ))
    if criterion.kind == "holdout":
        return float(holdout_score_regression(
            Xc, yc_for_scoring, _fp,
            fraction=criterion.holdout_fraction,
            random_state=criterion.random_state,
        ))
    raise ValueError(f"unsupported criterion for block-sparse: {criterion.kind!r}")


# ---------------------------------------------------------------------------
# Block-sparse AOM-MBPLS public API
# ---------------------------------------------------------------------------


def fit_block_sparse_aom(
    Xc: np.ndarray,
    yc: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    op_to_block: Sequence[int],
    block_masks: Dict[int, np.ndarray],
    n_components_max: int,
    criterion: CriterionConfig,
    auto_prefix: bool = True,
) -> MBPLSResult:
    """Greedy block-sparse AOM-MBPLS with optional auto-prefix.

    Greedy selection: at each LV, score every operator candidate against the
    current residual and pick the best. After `n_components_max` LVs are
    committed, optionally re-score each prefix `1..K_max` and pick the
    smallest k whose score is the minimum (auto-prefix). Auto-prefix matches
    the production AOM-PLS POP behaviour.

    Args:
        Xc: centered training data, shape (n, p).
        yc: centered target, shape (n,) or (n, q).
        operators: bank of strict-linear operators. Each operator must
            include the block mask in its action (e.g. via
            `ComposedOperator([preproc, mask])`). Pure block masks are
            allowed (V1 layout). Identity / bare preproc must be filtered
            out before calling — use `derive_block_metadata`.
        op_to_block: same length as `operators`; entry b is the block
            index for operator b.
        block_masks: mapping `block_idx -> 1-D mask array of length p`.
        n_components_max: upper bound on number of components.
        criterion: CriterionConfig (kind, cv, random_state, cv_splitter,
            holdout_fraction).
        auto_prefix: if True, score every prefix and pick the smallest k
            with lowest score. Default True.
    """
    op_indices: List[int] = []
    candidate_scores_per_lv: List[Dict[int, float]] = []

    for a in range(n_components_max):
        scores: Dict[int, float] = {}
        for b in range(len(operators)):
            score = _score_block_sparse_indices(
                Xc=Xc, yc=yc, operators=operators,
                op_to_block=op_to_block, block_masks=block_masks,
                indices=op_indices + [b],
                criterion=criterion,
            )
            scores[b] = score
        candidate_scores_per_lv.append(dict(scores))
        if not scores:
            break
        best_b = min(scores.items(), key=lambda kv: kv[1])[0]
        op_indices.append(best_b)

    final_indices = list(op_indices)
    prefix_scores: List[float] = []
    if auto_prefix and op_indices:
        for k in range(1, len(op_indices) + 1):
            sc = _score_block_sparse_indices(
                Xc=Xc, yc=yc, operators=operators,
                op_to_block=op_to_block, block_masks=block_masks,
                indices=op_indices[:k],
                criterion=criterion,
            )
            prefix_scores.append(sc)
        if prefix_scores:
            best_k = int(np.argmin(prefix_scores)) + 1
            final_indices = op_indices[:best_k]

    res = _fit_block_sparse_fixed(
        Xc=Xc, yc=yc, operators=operators,
        op_to_block=op_to_block, block_masks=block_masks,
        indices=final_indices,
    )
    res.diagnostics.update({
        "algorithm": "block_sparse_aom",
        "candidate_scores_per_lv": candidate_scores_per_lv,
        "prefix_scores": prefix_scores,
        "n_components_max_visited": len(op_indices),
    })
    return res


# ---------------------------------------------------------------------------
# Helper: derive (operators, op_to_block, block_masks) from a ViewBuilder bank
# ---------------------------------------------------------------------------


def derive_block_metadata(
    bank: Sequence[LinearSpectralOperator],
    blocks: Sequence[Tuple[int, int]],
    p: int,
) -> Tuple[List[LinearSpectralOperator], List[int], Dict[int, np.ndarray]]:
    """From a ViewBuilder bank, derive (filtered_bank, op_to_block, block_masks).

    Filters out non-block-aware operators (identity, raw preproc) — block-sparse
    requires every operator to be associated with a single block. Mapping:

    - `BlockMaskOperator` -> its own `start, end` block.
    - `ComposedOperator([preproc, BlockMaskOperator])` -> the inner mask's block.
    - Identity / bare preproc are skipped (no block association).

    Returns:
        `(operators_kept, op_to_block_indices, block_masks)`.
    """
    from .views import BlockMaskOperator

    block_to_idx: Dict[Tuple[int, int], int] = {
        (s, e): k for k, (s, e) in enumerate(blocks)
    }
    masks: Dict[int, np.ndarray] = {}
    for k, (s, e) in enumerate(blocks):
        m = np.zeros(p, dtype=float)
        m[s:e] = 1.0
        masks[k] = m

    kept: List[LinearSpectralOperator] = []
    op_blocks: List[int] = []
    for op in bank:
        if isinstance(op, BlockMaskOperator):
            key = (op.start, op.end)
            if key in block_to_idx:
                kept.append(op)
                op_blocks.append(block_to_idx[key])
        elif isinstance(op, ComposedOperator):
            inner_masks = [
                inner for inner in op.operators if isinstance(inner, BlockMaskOperator)
            ]
            if len(inner_masks) == 1:
                key = (inner_masks[0].start, inner_masks[0].end)
                if key in block_to_idx:
                    kept.append(op)
                    op_blocks.append(block_to_idx[key])

    return kept, op_blocks, masks
