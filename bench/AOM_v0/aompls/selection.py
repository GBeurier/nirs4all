"""Operator selection policies.

Five policies are supported:

- `none`: no selection; the bank must be a singleton.
- `global`: AOM. One operator picked for all components.
- `per_component`: POP. Per-component operator pursuit.
- `soft`: experimental convex mixture (covariance objective only).
- `superblock`: concatenate transformed views and run PLS on the wide matrix.

Each policy returns a `SelectionResult` with the chosen operator sequence,
the final `NIPALSResult`, and a per-candidate score table for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .nipals import (
    NIPALSResult,
    nipals_adjoint,
    nipals_materialized_fixed,
    nipals_materialized_per_component,
)
from .operators import IdentityOperator, LinearSpectralOperator
from .scorers import (
    CriterionConfig,
    covariance_score,
    cv_score_regression,
    holdout_score_regression,
    approx_press_regression,
)
from .simpls import (
    simpls_covariance,
    simpls_materialized_fixed,
    simpls_materialized_per_component,
    superblock_simpls,
)


@dataclass
class SelectionResult:
    """Outcome of a selection policy run."""

    result: NIPALSResult
    operator_indices: List[int]
    operator_names: List[str]
    operator_scores: dict = field(default_factory=dict)
    n_components_selected: int = 0
    diagnostics: dict = field(default_factory=dict)


def _resolve_engine(engine: str, X: np.ndarray, Y: np.ndarray, ops: Sequence[LinearSpectralOperator],
                    op_indices: Sequence[int], n_components: int, orthogonalization: str) -> NIPALSResult:
    """Dispatch to the requested engine."""
    if engine == "pls_standard":
        from .nipals import nipals_pls_standard
        return nipals_pls_standard(X, Y, n_components)
    if engine == "nipals_materialized":
        if len(set(op_indices)) == 1:
            return nipals_materialized_fixed(X, Y, ops[op_indices[0]], n_components)
        return nipals_materialized_per_component(X, Y, ops, op_indices, n_components)
    if engine == "nipals_adjoint":
        return nipals_adjoint(X, Y, ops, op_indices, n_components)
    if engine == "simpls_materialized":
        if len(set(op_indices)) == 1 and orthogonalization == "transformed":
            return simpls_materialized_fixed(X, Y, ops[op_indices[0]], n_components)
        return simpls_materialized_per_component(
            X, Y, ops, op_indices, n_components, orthogonalization=orthogonalization
        )
    if engine == "simpls_covariance":
        return simpls_covariance(
            X, Y, ops, op_indices, n_components, orthogonalization=orthogonalization
        )
    raise ValueError(f"unknown engine: {engine!r}")


def _fit_predict_factory(engine: str, ops: Sequence[LinearSpectralOperator],
                          op_indices_for_eval, n_components: int, orthogonalization: str):
    """Return a callable `(X_train, y_train, X_val) -> y_val_hat` for CV."""

    def _fit_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray) -> np.ndarray:
        x_mean = X_tr.mean(axis=0)
        y_mean = y_tr.mean(axis=0) if y_tr.ndim > 1 else float(y_tr.mean())
        Xc = X_tr - x_mean
        yc = y_tr - y_mean
        if op_indices_for_eval is None:
            op_idx = [0] * n_components
        else:
            op_idx = list(op_indices_for_eval)
        res = _resolve_engine(engine, Xc, yc, ops, op_idx, n_components, orthogonalization)
        if res.n_components == 0:
            return np.full(X_va.shape[0], y_mean)
        coef = res.coef()
        Xv = X_va - x_mean
        pred = Xv @ coef
        if pred.ndim == 2 and pred.shape[1] == 1:
            pred = pred.ravel()
        return pred + y_mean

    return _fit_predict


# ---------------------------------------------------------------------------
# Score auto-prefix helper
# ---------------------------------------------------------------------------


def _auto_prefix_score(
    engine: str,
    operators: Sequence[LinearSpectralOperator],
    op_indices: Sequence[int],
    Xc: np.ndarray,
    yc: np.ndarray,
    n_components: int,
    criterion: CriterionConfig,
    orthogonalization: str,
) -> Tuple[int, float, np.ndarray]:
    """Score every prefix `k = 1..n_components` and return the argmin.

    Returns `(best_k, best_score, all_scores)`. For criterion `cv`/`holdout`,
    the engine is refit per prefix on each fold. For `approx_press`, all
    prefixes are evaluated from a single full-fit pass.
    """
    if criterion.kind == "approx_press":
        # Fit once on the full data, then evaluate every prefix coefficient.
        x_mean = Xc.mean(axis=0)
        y_mean = yc.mean(axis=0) if yc.ndim > 1 else float(yc.mean())
        Xcc = Xc - x_mean
        ycc = yc - y_mean
        res = _resolve_engine(engine, Xcc, ycc, operators, list(op_indices), n_components, orthogonalization)
        coef_list = []
        for k in range(1, res.n_components + 1):
            coef_list.append(res.coef_prefix(k))
        scores = np.full(n_components, np.inf)
        press = approx_press_regression(Xcc, ycc, coef_list)
        for i, val in enumerate(press):
            scores[i] = val
        if scores.size == 0 or np.all(np.isinf(scores)):
            return 1, float("inf"), scores
        best_k = int(np.argmin(scores)) + 1
        return best_k, float(scores[best_k - 1]), scores
    if criterion.kind in ("cv", "hybrid"):
        # Fit once with the maximum prefix, then evaluate every prefix on the
        # same fold via NIPALSResult.coef_prefix(k). This matches production
        # AOM-PLS and avoids the silent shape mismatch where len(op_indices)
        # > prefix_k caused the engine to raise and the holdout helper to
        # return +inf.
        scores = np.full(n_components, np.inf)

        def _fp_fold_full_fit(X_tr, y_tr, X_va):
            x_mean = X_tr.mean(axis=0)
            y_mean = y_tr.mean(axis=0) if y_tr.ndim > 1 else float(y_tr.mean())
            Xtc = X_tr - x_mean
            ytc = y_tr - y_mean
            res = _resolve_engine(
                engine, Xtc, ytc, operators, list(op_indices), n_components, orthogonalization
            )
            Xv = X_va - x_mean
            preds_per_prefix = []
            for k in range(1, res.n_components + 1):
                coef_k = res.coef_prefix(k)
                pred_k = Xv @ coef_k
                if pred_k.ndim == 2 and pred_k.shape[1] == 1:
                    pred_k = pred_k.ravel()
                preds_per_prefix.append(pred_k + y_mean)
            return preds_per_prefix

        rmses_per_prefix = _cv_score_per_prefix(Xc, yc, _fp_fold_full_fit, criterion.cv, criterion.random_state, n_components, repeats=criterion.repeats)
        for k_idx, val in enumerate(rmses_per_prefix):
            scores[k_idx] = val
        if scores.size == 0 or np.all(np.isinf(scores)):
            return 1, float("inf"), scores
        best_k = int(np.argmin(scores)) + 1
        return best_k, float(scores[best_k - 1]), scores
    if criterion.kind == "holdout":
        scores = np.full(n_components, np.inf)

        def _fp_holdout_global_centered(X_tr, y_tr, X_va):
            # Production fits NIPALS on the GLOBALLY centered holdout-train
            # (no per-fold re-centering). The estimator already passed centered
            # `Xc, yc`, so X_tr / y_tr are already centered; we run the engine
            # directly on them and predict on X_va in the same coordinate frame.
            res = _resolve_engine(
                engine, X_tr, y_tr, operators, list(op_indices), n_components, orthogonalization
            )
            preds_per_prefix = []
            for k in range(1, res.n_components + 1):
                coef_k = res.coef_prefix(k)
                pred_k = X_va @ coef_k
                if pred_k.ndim == 2 and pred_k.shape[1] == 1:
                    pred_k = pred_k.ravel()
                preds_per_prefix.append(pred_k)
            return preds_per_prefix

        rmses = _holdout_score_per_prefix(
            Xc, yc, _fp_holdout_global_centered, criterion.holdout_fraction,
            criterion.holdout_seed, n_components
        )
        for k_idx, val in enumerate(rmses):
            scores[k_idx] = val
        if scores.size == 0 or np.all(np.isinf(scores)):
            return 1, float("inf"), scores
        best_k = int(np.argmin(scores)) + 1
        return best_k, float(scores[best_k - 1]), scores
    if criterion.kind == "covariance":
        # Covariance scoring uses S = X^T y; not a prefix selector.
        return n_components, 0.0, np.zeros(n_components)
    raise ValueError(f"unknown criterion: {criterion.kind!r}")


def _safe_residual(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Element-wise residual that resists 1D/2D shape mismatches.

    `y_true` may arrive as `(n, 1)` (the estimator reshapes y to 2D) while
    `y_pred` is 1D. A naive `y_true - y_pred` would broadcast to `(n, n)`.
    We coerce both to identical shape before subtracting.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim == 2 and yt.shape[1] == 1:
        yt = yt.ravel()
    if yp.ndim == 2 and yp.shape[1] == 1:
        yp = yp.ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch in holdout residual: y_true {yt.shape} vs y_pred {yp.shape}")
    return (yt - yp).ravel()


def _cv_score_per_prefix(
    Xc: np.ndarray, yc: np.ndarray, fit_predict_per_prefix, n_splits: int, random_state: int, n_components: int,
    repeats: int = 1
) -> List[float]:
    """K-fold CV scoring that fits once per fold and evaluates all prefixes.

    When `repeats > 1`, runs `repeats` independent K-fold splits with
    different random seeds derived from `random_state` and returns the
    mean per-prefix RMSE averaged across all (repeat * fold) blocks. This
    reduces selection variance by sqrt(repeats) at the cost of `repeats`
    times more fits.
    """
    from sklearn.model_selection import KFold
    total_blocks = max(1, int(repeats)) * n_splits
    fold_rmses = np.full((total_blocks, n_components), np.inf)
    block = 0
    for r in range(max(1, int(repeats))):
        seed = int(random_state) + r
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in kf.split(Xc):
            X_tr, X_va = Xc[train_idx], Xc[val_idx]
            y_tr, y_va = yc[train_idx], yc[val_idx]
            try:
                preds_per_prefix = fit_predict_per_prefix(X_tr, y_tr, X_va)
            except Exception:
                block += 1
                continue
            for k_idx, pred_k in enumerate(preds_per_prefix):
                try:
                    diff = _safe_residual(y_va, pred_k)
                except ValueError:
                    continue
                fold_rmses[block, k_idx] = float(np.sqrt(np.mean(diff * diff)))
            block += 1
    return [float(np.mean(fold_rmses[:, k])) for k in range(n_components)]


def _holdout_score_per_prefix(
    Xc: np.ndarray, yc: np.ndarray, fit_predict_per_prefix, fraction: float, random_state: int, n_components: int
) -> List[float]:
    """Single-split holdout scoring that fits once and evaluates all prefixes.

    Uses the legacy `np.random.RandomState` with seed 42 by default (matches
    the production AOM-PLS), `n_ho = max(3, n // 5)` so the validation block is
    at least 3 samples. The first `n_ho` permuted indices are the validation
    set, the remainder is the training set — same convention as production.
    """
    rng = np.random.RandomState(int(random_state))
    n = Xc.shape[0]
    perm = rng.permutation(n)
    n_val = max(3, n // 5)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    X_tr, X_va = Xc[train_idx], Xc[val_idx]
    y_tr, y_va = yc[train_idx], yc[val_idx]
    rmses = [float("inf")] * n_components
    try:
        preds_per_prefix = fit_predict_per_prefix(X_tr, y_tr, X_va)
    except Exception:
        return rmses
    for k_idx, pred_k in enumerate(preds_per_prefix):
        try:
            diff = _safe_residual(y_va, pred_k)
        except ValueError:
            continue
        rmses[k_idx] = float(np.sqrt(np.mean(diff * diff)))
    return rmses


# ---------------------------------------------------------------------------
# Global selection (AOM)
# ---------------------------------------------------------------------------


def select_global(
    Xc: np.ndarray,
    yc: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    engine: str,
    n_components_max: int,
    criterion: CriterionConfig,
    orthogonalization: str,
    auto_prefix: bool,
) -> SelectionResult:
    """AOM: pick a single operator for the full model.

    Each operator is evaluated by the criterion at the requested prefix(es).
    For `auto_prefix=True`, the score is the best prefix score; the chosen
    `n_components` is that prefix.
    """
    operator_scores: dict = {}
    best_op = 0
    best_score = float("inf")
    best_k = n_components_max
    best_score_curve: Optional[np.ndarray] = None

    if criterion.kind == "covariance":
        # Use S = X^T y to score operators (PLS1 friendly).
        S = Xc.T @ yc.reshape(-1, 1) if yc.ndim == 1 else Xc.T @ yc
        for b, op in enumerate(operators):
            op.fit(Xc)
            sc = covariance_score(op.apply_cov(S))
            operator_scores[op.name] = sc
            if sc < best_score:
                best_score = sc
                best_op = b
                best_k = n_components_max
        op_indices = [best_op] * n_components_max
        result = _resolve_engine(engine, Xc, yc, operators, op_indices, n_components_max, orthogonalization)
        return SelectionResult(
            result=result,
            operator_indices=[best_op] * result.n_components,
            operator_names=[operators[best_op].name] * result.n_components,
            operator_scores=operator_scores,
            n_components_selected=result.n_components,
            diagnostics={"selection": "global", "criterion": criterion.kind},
        )
    # Hybrid: prescreen by covariance, then CV-evaluate the top-m operators.
    candidates = list(range(len(operators)))
    if criterion.kind == "hybrid":
        S = Xc.T @ yc.reshape(-1, 1) if yc.ndim == 1 else Xc.T @ yc
        cov_scores = np.array([covariance_score(operators[b].apply_cov(S)) for b in candidates])
        top_m = max(1, min(criterion.prescreen_top_m, len(candidates)))
        order = np.argsort(cov_scores)
        candidates = [int(candidates[i]) for i in order[:top_m]]
    # Evaluate each candidate with the prefix scorer.
    all_curves = {}
    for b in candidates:
        op_indices = [b] * n_components_max
        if auto_prefix:
            k_b, sc_b, curve = _auto_prefix_score(
                engine, operators, op_indices, Xc, yc, n_components_max, criterion, orthogonalization
            )
        else:
            # Fixed n_components: still score the candidate honestly with the
            # requested criterion (Codex code review, HIGH #1).
            sc_b, _ = _criterion_score_at_indices(
                Xc, yc, operators, engine, op_indices, criterion, orthogonalization
            )
            curve = np.zeros(n_components_max)
            k_b = n_components_max
        operator_scores[operators[b].name] = float(sc_b)
        all_curves[b] = (k_b, sc_b, curve)
        if sc_b < best_score:
            best_score = sc_b
            best_op = b
            best_k = k_b
            best_score_curve = curve
    one_se_applied = False
    if criterion.one_se_rule and best_score_curve is not None and len(all_curves) > 0:
        # Pragmatic one-SE rule: use the variability of the winning curve
        # across prefixes as a noise proxy. SE := std(curve)/sqrt(len(curve)).
        # Then pick the (b, k) pair with the smallest `k` (and smallest `b`
        # as a tiebreak) whose mean score is within `best_score + SE`. This
        # shrinks selection toward fewer components and identity-like
        # operators when the score surface is flat.
        curve_arr = np.asarray(best_score_curve, dtype=float)
        finite = curve_arr[np.isfinite(curve_arr)]
        if len(finite) >= 2:
            se = float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
            threshold = float(best_score) + se
            # Search every (b, k) pair within threshold; pick smallest k, then
            # smallest b. This is the global one-SE shrinkage.
            best_simple_k = best_k
            best_simple_b = best_op
            best_simple_score = best_score
            for b, (k_b_cur, sc_b_cur, curve_b) in all_curves.items():
                curve_b_arr = np.asarray(curve_b, dtype=float)
                for k_cand in range(1, min(len(curve_b_arr) + 1, best_k + 1)):
                    sc_cand = curve_b_arr[k_cand - 1] if k_cand - 1 < len(curve_b_arr) else float("inf")
                    if not np.isfinite(sc_cand) or sc_cand > threshold:
                        continue
                    # Simpler = smaller k, then smaller b.
                    if (k_cand < best_simple_k or
                        (k_cand == best_simple_k and b < best_simple_b)):
                        best_simple_k = k_cand
                        best_simple_b = b
                        best_simple_score = float(sc_cand)
            if best_simple_k < best_k or best_simple_b != best_op:
                one_se_applied = True
                best_k = best_simple_k
                best_op = best_simple_b
                best_score = best_simple_score
                best_score_curve = np.asarray(all_curves[best_op][2], dtype=float)
    op_indices = [best_op] * best_k
    result = _resolve_engine(engine, Xc, yc, operators, op_indices, best_k, orthogonalization)
    diag = {
        "selection": "global",
        "criterion": criterion.kind,
        "best_score": float(best_score),
        "score_curve": None if best_score_curve is None else best_score_curve.tolist(),
        "candidates": [int(c) for c in candidates],
        "one_se_applied": bool(one_se_applied),
    }
    return SelectionResult(
        result=result,
        operator_indices=[best_op] * result.n_components,
        operator_names=[operators[best_op].name] * result.n_components,
        operator_scores=operator_scores,
        n_components_selected=result.n_components,
        diagnostics=diag,
    )


# ---------------------------------------------------------------------------
# Per-component selection (POP)
# ---------------------------------------------------------------------------


def select_per_component(
    Xc: np.ndarray,
    yc: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    engine: str,
    n_components_max: int,
    criterion: CriterionConfig,
    orthogonalization: str,
    auto_prefix: bool,
) -> SelectionResult:
    """POP: greedy operator pursuit, one operator per component.

    At component `a`, every operator candidate is evaluated using the current
    state. The candidate that yields the lowest criterion score is selected;
    its component is committed and we proceed to the next.
    """
    operator_scores: dict = {}
    op_indices: List[int] = []
    op_names: List[str] = []
    yc2 = yc.reshape(-1, 1) if yc.ndim == 1 else yc
    # Iteratively grow the operator sequence using the engine of choice.
    for a in range(n_components_max):
        best_b = 0
        best_score = float("inf")
        candidate_scores: dict = {}
        if criterion.kind == "covariance":
            # Compute the deflated cross-covariance from the already-committed
            # operator sequence, then score each candidate by `||A_b S_res||`.
            # This matches AOM's covariance proxy and is leakage-safe.
            if len(op_indices) == 0:
                S_res = Xc.T @ yc2
            else:
                # Run the partial engine for already-selected components.
                partial_res = _resolve_engine(
                    engine, Xc, yc, operators, op_indices, len(op_indices), orthogonalization
                )
                if partial_res.n_components == 0:
                    S_res = Xc.T @ yc2
                else:
                    V = partial_res.diagnostics.get("basis")
                    if V is not None:
                        V = np.asarray(V)[:, : partial_res.n_components]
                        S_full = Xc.T @ yc2
                        S_res = S_full - V @ (V.T @ S_full)
                    else:
                        # NIPALS engines do not emit a Gram-Schmidt basis, so
                        # we deflate via residual matrices: X_res^T Y_res with
                        # X_res = X - T P^T and Y_res = Y - T Q^T (Codex review,
                        # MEDIUM).
                        T = partial_res.T
                        P = partial_res.P
                        Q = partial_res.Q
                        X_res = Xc - T @ P.T
                        Y_res = yc2 - T @ Q.T
                        S_res = X_res.T @ Y_res
            for b, op in enumerate(operators):
                op.fit(Xc)
                sc_b = covariance_score(op.apply_cov(S_res))
                candidate_scores[op.name] = sc_b
                if sc_b < best_score:
                    best_score = sc_b
                    best_b = b
        else:
            for b in range(len(operators)):
                partial_indices = op_indices + [b]
                sc_b, _ = _criterion_score_at_indices(
                    Xc, yc, operators, engine, partial_indices, criterion, orthogonalization
                )
                candidate_scores[operators[b].name] = sc_b
                if sc_b < best_score:
                    best_score = sc_b
                    best_b = b
        operator_scores[f"component_{a + 1}"] = candidate_scores
        op_indices.append(best_b)
        op_names.append(operators[best_b].name)
        # Optional auto-prefix early stop: if the running score does not
        # improve for many components we may prune. Not required for default
        # behaviour; we stop adding components after n_components_max.
    final_k = n_components_max
    one_se_applied = False
    if auto_prefix and criterion.kind != "covariance":
        # Score every prefix to choose final k.
        candidate_k_scores = np.full(n_components_max, np.inf)
        for k in range(1, n_components_max + 1):
            sc_k, _ = _criterion_score_at_indices(
                Xc, yc, operators, engine, op_indices[:k], criterion, orthogonalization
            )
            candidate_k_scores[k - 1] = sc_k
        final_k = int(np.argmin(candidate_k_scores)) + 1
        if criterion.one_se_rule:
            # Pragmatic one-SE: pick smallest k whose score is within
            # `best + std(curve)/sqrt(n_k)` of the optimum. Same noise proxy
            # as `select_global`.
            finite = candidate_k_scores[np.isfinite(candidate_k_scores)]
            if len(finite) >= 2:
                best_score_pop = float(candidate_k_scores[final_k - 1])
                se = float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
                threshold = best_score_pop + se
                for k_cand in range(1, final_k):
                    if np.isfinite(candidate_k_scores[k_cand - 1]) and candidate_k_scores[k_cand - 1] <= threshold:
                        if k_cand < final_k:
                            final_k = k_cand
                            one_se_applied = True
                            break
    result = _resolve_engine(engine, Xc, yc, operators, op_indices[:final_k], final_k, orthogonalization)
    diag = {
        "selection": "per_component",
        "criterion": criterion.kind,
        "operator_sequence": op_indices[:final_k],
        "one_se_applied": bool(one_se_applied),
    }
    return SelectionResult(
        result=result,
        operator_indices=op_indices[:final_k],
        operator_names=op_names[:final_k],
        operator_scores=operator_scores,
        n_components_selected=final_k,
        diagnostics=diag,
    )


def _criterion_score_at_indices(
    Xc: np.ndarray,
    yc: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    engine: str,
    indices: Sequence[int],
    criterion: CriterionConfig,
    orthogonalization: str,
) -> Tuple[float, Optional[np.ndarray]]:
    """Score a fixed operator-sequence under the chosen criterion (regression)."""
    K = len(indices)
    if K == 0:
        return float("inf"), None
    if criterion.kind == "approx_press":
        x_mean = Xc.mean(axis=0)
        y_mean = yc.mean(axis=0) if yc.ndim > 1 else float(yc.mean())
        Xcc = Xc - x_mean
        ycc = yc - y_mean
        res = _resolve_engine(engine, Xcc, ycc, operators, list(indices), K, orthogonalization)
        if res.n_components == 0:
            return float("inf"), None
        press = approx_press_regression(Xcc, ycc, [res.coef()])
        return float(press[0]), None
    if criterion.kind == "covariance":
        x_mean = Xc.mean(axis=0)
        y_mean = yc.mean(axis=0) if yc.ndim > 1 else float(yc.mean())
        Xcc = Xc - x_mean
        ycc = yc - y_mean
        res = _resolve_engine(engine, Xcc, ycc, operators, list(indices), K, orthogonalization)
        if res.n_components == 0:
            return float("inf"), None
        coef = res.coef()
        pred = Xcc @ coef
        return float(np.sqrt(np.mean((ycc.reshape(pred.shape) - pred) ** 2))), None
    if criterion.kind in ("cv", "hybrid"):
        def _fp(X_tr, y_tr, X_va):
            x_mean = X_tr.mean(axis=0)
            y_mean = y_tr.mean(axis=0) if y_tr.ndim > 1 else float(y_tr.mean())
            Xtc = X_tr - x_mean
            ytc = y_tr - y_mean
            res = _resolve_engine(engine, Xtc, ytc, operators, list(indices), K, orthogonalization)
            if res.n_components == 0:
                return np.full(X_va.shape[0], y_mean)
            coef = res.coef()
            Xv = X_va - x_mean
            pred = Xv @ coef
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred.ravel()
            return pred + y_mean
        return cv_score_regression(Xc, yc, _fp, criterion.cv, criterion.random_state), None
    if criterion.kind == "holdout":
        def _fp(X_tr, y_tr, X_va):
            x_mean = X_tr.mean(axis=0)
            y_mean = y_tr.mean(axis=0) if y_tr.ndim > 1 else float(y_tr.mean())
            Xtc = X_tr - x_mean
            ytc = y_tr - y_mean
            res = _resolve_engine(engine, Xtc, ytc, operators, list(indices), K, orthogonalization)
            if res.n_components == 0:
                return np.full(X_va.shape[0], y_mean)
            coef = res.coef()
            Xv = X_va - x_mean
            pred = Xv @ coef
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred.ravel()
            return pred + y_mean
        return holdout_score_regression(Xc, yc, _fp, criterion.holdout_fraction, criterion.random_state), None
    raise ValueError(f"unknown criterion {criterion.kind!r}")


# ---------------------------------------------------------------------------
# Soft mixture (experimental)
# ---------------------------------------------------------------------------


def select_soft(
    Xc: np.ndarray,
    yc: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    engine: str,
    n_components_max: int,
    criterion: CriterionConfig,
    orthogonalization: str,
    gate: str = "softmax",
    temperature: float = 1.0,
) -> SelectionResult:
    """Experimental soft mixture: weight operators by a covariance-softmax.

    For each component, compute candidate covariance scores, derive weights
    via softmax (or sparsemax), and apply the convex combination of operators.
    Soft mixture often degenerates to hard selection on covariance objectives;
    this is intentionally exposed as experimental.
    """
    yc2 = yc.reshape(-1, 1) if yc.ndim == 1 else yc
    Tres = Xc.copy()
    Yres = yc2.copy()
    n, p = Xc.shape
    q = yc2.shape[1]
    Z = np.zeros((p, n_components_max))
    P = np.zeros((p, n_components_max))
    Q = np.zeros((q, n_components_max))
    T = np.zeros((n, n_components_max))
    weights_table = []
    op_indices_per_component: List[int] = []
    for a in range(n_components_max):
        S = Tres.T @ Yres
        scores = np.array([covariance_score(op.apply_cov(S)) for op in operators])
        adj = -scores  # higher is better
        if gate == "sparsemax":
            w = _sparsemax(adj / max(temperature, 1e-9))
        else:
            w = _softmax(adj / max(temperature, 1e-9))
        # Combined direction in original space
        z = np.zeros(p)
        for b, op in enumerate(operators):
            if w[b] < 1e-10:
                continue
            r = op.apply_cov(S) if S.ndim == 1 else op.apply_cov(S)
            r = r if r.ndim == 1 else r[:, 0]
            r_norm = np.linalg.norm(r)
            if r_norm > 1e-12:
                z = z + w[b] * op.adjoint_vec(r / r_norm)
        z_norm = np.linalg.norm(z)
        if z_norm < 1e-12:
            break
        z = z / z_norm
        t = Tres @ z
        t_norm = np.linalg.norm(t)
        if t_norm < 1e-12:
            break
        # Normalisation similar to SIMPLS for prediction consistency.
        Z[:, a] = z
        T[:, a] = t
        p_load = Tres.T @ t / (t @ t)
        q_load = Yres.T @ t / (t @ t)
        P[:, a] = p_load
        Q[:, a] = q_load
        Tres = Tres - np.outer(t, p_load)
        Yres = Yres - np.outer(t, q_load)
        weights_table.append(w.tolist())
        op_indices_per_component.append(int(np.argmax(w)))
    K = len([t for t in T.T if np.linalg.norm(t) > 0])
    res = NIPALSResult(
        Z=Z[:, :K],
        P=P[:, :K],
        Q=Q[:, :K],
        T=T[:, :K],
        R=[Z[:, a].copy() for a in range(K)],
        operator_indices=op_indices_per_component[:K],
        operator_names=[operators[i].name for i in op_indices_per_component[:K]],
        diagnostics={"engine": "soft_mixture", "weights": weights_table},
    )
    return SelectionResult(
        result=res,
        operator_indices=op_indices_per_component[:K],
        operator_names=[operators[i].name for i in op_indices_per_component[:K]],
        operator_scores={"weights": weights_table},
        n_components_selected=K,
        diagnostics={"selection": "soft", "gate": gate, "criterion": criterion.kind},
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def _sparsemax(z: np.ndarray) -> np.ndarray:
    """Project z onto the probability simplex (Martins & Astudillo, 2016)."""
    z_sorted = np.sort(z)[::-1]
    cumsum = np.cumsum(z_sorted)
    rho = 0
    for i, zi in enumerate(z_sorted):
        if 1 + (i + 1) * zi > cumsum[i]:
            rho = i + 1
    tau = (cumsum[rho - 1] - 1) / rho
    return np.maximum(z - tau, 0.0)


# ---------------------------------------------------------------------------
# Superblock selection
# ---------------------------------------------------------------------------


def select_superblock(
    Xc: np.ndarray,
    yc: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    n_components_max: int,
    criterion: CriterionConfig,
    block_weights: Optional[Sequence[float]] = None,
) -> SelectionResult:
    """Concatenate operator views and run SIMPLS on the wide block.

    Returns a `SelectionResult` whose `coef()` lives in the **original**
    feature space (`(p, q)` shape), so estimator predictions are valid on
    plain `(n, p)` matrices.
    """
    res, groups = superblock_simpls(Xc, yc, operators, n_components_max, block_weights=block_weights)
    coef = res.coef()
    if coef.ndim == 1:
        coef = coef.reshape(-1, 1)
    importance = {
        operators[b].name: float(np.linalg.norm(coef))
        for b in range(len(operators))
    }
    op_names = [op.name for op in operators]
    weights = res.diagnostics.get("block_weights", [1.0] * len(operators))
    return SelectionResult(
        result=res,
        operator_indices=list(range(len(operators))),
        operator_names=op_names,
        operator_scores={"group_importance": importance, "block_weights": list(weights)},
        n_components_selected=res.n_components,
        diagnostics={
            "selection": "superblock",
            "groups": groups.tolist(),
            "block_weights": list(weights),
            "original_feature_space": True,
        },
    )


# ---------------------------------------------------------------------------
# Active superblock selection
# ---------------------------------------------------------------------------


def _frobenius_block_weights(
    Xc: np.ndarray, operators: Sequence[LinearSpectralOperator]
) -> List[float]:
    """Per-block weight `1 / (||X A^T||_F + eps)` for Frobenius balancing."""
    eps = 1e-9
    weights: List[float] = []
    for op in operators:
        op.fit(Xc)
        Xb = op.transform(Xc)
        norm = float(np.linalg.norm(Xb)) + eps
        weights.append(1.0 / norm)
    # Optional sqrt(n) factor for sensible relative magnitudes; documented above.
    n = float(Xc.shape[0]) ** 0.5
    return [w * n for w in weights]


def _select_active_bank(
    Xc: np.ndarray,
    yc: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    active_top_m: int,
    diversity_threshold: float,
) -> Tuple[List[int], List[float], List[np.ndarray]]:
    """Pick a diverse subset of operators by covariance score and response cosine.

    Returns indices into `operators`, raw covariance scores, and the response
    arrays for the selected operators.
    """
    yc2 = yc.reshape(-1, 1) if yc.ndim == 1 else yc
    S = Xc.T @ yc2
    items = []
    identity_idx = None
    for b, op in enumerate(operators):
        op.fit(Xc)
        if op.name == "identity" and identity_idx is None:
            identity_idx = b
        resp = op.apply_cov(S)
        score = -float(np.linalg.norm(resp))
        items.append((score, np.asarray(resp).ravel(), b))
    items.sort(key=lambda t: t[0])
    selected_idx: List[int] = []
    selected_scores: List[float] = []
    selected_responses: List[np.ndarray] = []
    # Always include identity if present and we have room.
    if identity_idx is not None and active_top_m > 1:
        for sc, resp, b in items:
            if b == identity_idx:
                selected_idx.append(b)
                selected_scores.append(sc)
                selected_responses.append(resp)
                break
    for sc, resp, b in items:
        if b in selected_idx:
            continue
        if any(_cosine(resp, prev) >= diversity_threshold for prev in selected_responses):
            continue
        selected_idx.append(b)
        selected_scores.append(sc)
        selected_responses.append(resp)
        if len(selected_idx) >= active_top_m:
            break
    return selected_idx, selected_scores, selected_responses


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(abs(a @ b) / (na * nb))


def select_active_superblock(
    Xc: np.ndarray,
    yc: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    n_components_max: int,
    criterion: CriterionConfig,
    active_top_m: int = 20,
    diversity_threshold: float = 0.98,
    block_scaling: str = "frobenius",
) -> SelectionResult:
    """Active Superblock PLS: covariance-screen + diversity prune + weighted SIMPLS.

    1. Screen operators by `-||A_b S||` (covariance proxy).
    2. Prune redundant responses (`response_cosine >= diversity_threshold`).
    3. Keep up to `active_top_m` operators (identity always retained).
    4. Compute per-block weights (`frobenius` by default) so high-gain blocks
       do not dominate solely by amplitude.
    5. Concatenate `[alpha_b X A_b^T]_b`, run SIMPLS, map back to original
       feature space via `Z_orig[:, a] = sum_b alpha_b A_b^T Z_wide_b[:, a]`.

    Returns a `SelectionResult` with `coef_` in the `(p, q)` original space.
    """
    selected_idx, selected_scores, selected_responses = _select_active_bank(
        Xc, yc, operators, active_top_m=active_top_m, diversity_threshold=diversity_threshold
    )
    active_ops = [operators[b] for b in selected_idx]
    if block_scaling == "none":
        weights = [1.0] * len(active_ops)
    else:
        weights = _frobenius_block_weights(Xc, active_ops)
    res, groups = superblock_simpls(Xc, yc, active_ops, n_components_max, block_weights=weights)
    coef = res.coef()
    if coef.ndim == 1:
        coef = coef.reshape(-1, 1)
    importance = {
        op.name: float(np.linalg.norm(coef))
        for op in active_ops
    }
    return SelectionResult(
        result=res,
        operator_indices=list(selected_idx),
        operator_names=[op.name for op in active_ops],
        operator_scores={
            "active_operator_scores": {op.name: float(s) for op, s in zip(active_ops, selected_scores)},
            "group_importance": importance,
            "block_weights": list(weights),
        },
        n_components_selected=res.n_components,
        diagnostics={
            "selection": "active_superblock",
            "active_operator_indices": list(selected_idx),
            "active_operator_names": [op.name for op in active_ops],
            "active_operator_scores": {op.name: float(s) for op, s in zip(active_ops, selected_scores)},
            "block_weights": list(weights),
            "group_importance": importance,
            "active_top_m": active_top_m,
            "diversity_threshold": diversity_threshold,
            "block_scaling": block_scaling,
            "groups": groups.tolist(),
            "original_feature_space": True,
        },
    )


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def select(
    Xc: np.ndarray,
    yc: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    engine: str,
    selection: str,
    n_components_max: int,
    criterion: CriterionConfig,
    orthogonalization: str = "auto",
    auto_prefix: bool = True,
) -> SelectionResult:
    """Run the requested selection policy and return the result."""
    if orthogonalization == "auto":
        if selection in ("none", "global"):
            orthogonalization = "transformed"
        else:
            orthogonalization = "original"
    if selection == "none":
        if len(operators) != 1:
            raise ValueError("selection='none' requires a singleton bank")
        op_indices = [0] * n_components_max
        if auto_prefix and criterion.kind != "covariance":
            best_k, best_score, scores = _auto_prefix_score(
                engine, operators, op_indices, Xc, yc, n_components_max, criterion, orthogonalization
            )
            op_indices = [0] * best_k
        result = _resolve_engine(engine, Xc, yc, operators, op_indices, len(op_indices), orthogonalization)
        return SelectionResult(
            result=result,
            operator_indices=[0] * result.n_components,
            operator_names=[operators[0].name] * result.n_components,
            operator_scores={operators[0].name: 0.0},
            n_components_selected=result.n_components,
            diagnostics={"selection": "none"},
        )
    if selection == "global":
        return select_global(Xc, yc, operators, engine, n_components_max, criterion, orthogonalization, auto_prefix)
    if selection == "per_component":
        return select_per_component(
            Xc, yc, operators, engine, n_components_max, criterion, orthogonalization, auto_prefix
        )
    if selection == "soft":
        return select_soft(Xc, yc, operators, engine, n_components_max, criterion, orthogonalization)
    if selection == "superblock":
        return select_superblock(Xc, yc, operators, n_components_max, criterion)
    if selection == "active_superblock":
        return select_active_superblock(Xc, yc, operators, n_components_max, criterion)
    raise ValueError(f"unknown selection: {selection!r}")
