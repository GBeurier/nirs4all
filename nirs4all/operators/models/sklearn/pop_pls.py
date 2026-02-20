"""POP-PLS: Per-Operator-Per-component PLS regressor for nirs4all.

A sklearn-compatible implementation of POP-PLS that selects a potentially
DIFFERENT linear preprocessing operator for EACH PLS component. This is the
true multi-operator mode: each component independently chooses the operator
that best improves the model, then extracts that component through the
selected operator's lens via NIPALS deflation.

Compared to AOM-PLS (which selects ONE operator for ALL components), POP-PLS
can adapt its preprocessing strategy as successive residuals reveal different
spectral features, e.g. component 1 might use SG 1st derivative to capture
slope information while component 3 uses detrending for baseline correction.

Mathematical formulation
------------------------
Let X in R^{n x p} be the input matrix and y in R^n the response vector.
Given an operator bank {A_b}_{b=1..B} of p x p linear operators:

For k = 1..K_max:
  1. Cross-covariance: c_k = X_res^T @ y_res
  2. For each operator b in the bank:
     - Adjoint trick: g_b = A_b^T @ c_k
     - Normalize: w_hat_b = g_b / ||g_b||
     - Forward pass: a_w_b = A_b @ w_hat_b, then w_b = a_w_b / ||a_w_b||
     - Candidate score: t_b = X_res @ w_b
     - Build prefix model B_k, evaluate PRESS on full training data
  3. Select: b* = argmin_b PRESS(prefix_k)
  4. Extract component k using operator b*: NIPALS deflation
  5. Record gamma_[k, b*] = 1.0

Built-in model selection via PRESS (no Optuna or holdout needed):
  PRESS (Predicted Residual Error Sum of Squares) approximates leave-one-out
  cross-validation using the PLS hat matrix: PRESS = sum((e_i / (1 - h_ii))^2).
  This uses ALL training data for selection (no holdout split), providing
  stable, deterministic operator and component-count decisions. The n_orth
  OPLS pre-filter is auto-tuned by comparing PRESS across candidates.

References
----------
- de Jong, S. (1993). SIMPLS: An alternative approach to partial least
  squares regression. Chemometrics and Intelligent Laboratory Systems.
- Martens, M. & Martens, H. (2001). Multivariate Analysis of Quality:
  An Introduction. Wiley.
- Allen, D.M. (1974). The Relationship Between Variable Selection and
  Data Augmentation and a Method for Prediction. Technometrics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from nirs4all.operators.models.sklearn.aom_pls import (
    ComposedOperator,
    DetrendProjectionOperator,
    IdentityOperator,
    LinearOperator,
    SavitzkyGolayOperator,
    _opls_prefilter,
)

# =============================================================================
# POP-PLS Operator Bank
# =============================================================================

def pop_pls_operator_bank() -> list[LinearOperator]:
    """Build a dedicated operator bank optimized for POP-PLS.

    Unlike the AOM bank (which has many parametric variants for global search),
    this bank is COMPACT and DIVERSE: ~9 operators, each capturing a
    qualitatively different spectral transformation. This design exploits
    POP-PLS's per-component selection: instead of many similar operators
    competing (adding noise to selection), a few diverse operators each serve
    a clear role at different stages of NIPALS deflation.

    Design principles:
    - Maximum diversity: each operator has a qualitatively different effect
    - Spanning mild to aggressive: from identity to second derivatives
    - Covering the key NIRS preprocessing families: smoothing, derivatives,
      detrending, and compositions thereof
    - Compact (9 operators) to minimize selection noise on small holdouts
      or PRESS evaluations

    Returns
    -------
    operators : list of LinearOperator
        Compact, diverse operator bank for POP-PLS (~9 operators).
    """
    return [
        # Identity — recovers standard PLS, always a contender for early
        # components where dominant features need no preprocessing.
        IdentityOperator(),
        # Smoothing — reduces high-frequency noise while preserving spectral
        # shape. Good for early components where dominant variation is broad.
        SavitzkyGolayOperator(window=21, polyorder=2, deriv=0),
        # 1st derivative, narrow window — enhances sharp features, removes
        # baseline. The workhorse of NIRS preprocessing. High spectral
        # resolution but amplifies noise.
        SavitzkyGolayOperator(window=11, polyorder=2, deriv=1),
        # 1st derivative, medium window — broader feature enhancement with
        # better noise suppression. Balances resolution and stability.
        SavitzkyGolayOperator(window=21, polyorder=2, deriv=1),
        # 1st derivative, wide window — maximum noise suppression derivative.
        # Good for late components where residuals are noisy.
        SavitzkyGolayOperator(window=31, polyorder=2, deriv=1),
        # 2nd derivative — removes both baseline and linear trends. Isolates
        # peak curvature. Good for overlapping spectral bands.
        SavitzkyGolayOperator(window=21, polyorder=3, deriv=2),
        # Linear detrend — removes linear baseline drift without computing
        # derivatives. Complementary to derivative approaches.
        DetrendProjectionOperator(degree=1),
        # Quadratic detrend — removes curved baseline caused by scattering
        # effects. Handles more complex baselines than linear detrend.
        DetrendProjectionOperator(degree=2),
        # Combined 1st derivative + detrend — maximum baseline removal with
        # feature enhancement. Good for complex baseline scenarios where
        # neither derivative nor detrend alone suffices.
        ComposedOperator(
            first=SavitzkyGolayOperator(window=15, polyorder=2, deriv=1),
            second=DetrendProjectionOperator(degree=1),
        ),
    ]

# =============================================================================
# POP-PLS Core Algorithms
# =============================================================================

def _compute_weight(op: LinearOperator, c_k: NDArray, eps: float) -> NDArray | None:
    """Compute PLS weight vector via the adjoint trick for a given operator.

    Returns None if the operator produces degenerate results.
    """
    g_b = op.apply_adjoint(c_k)
    g_norm = np.linalg.norm(g_b)
    if g_norm < eps:
        return None
    w_hat_b = g_b / g_norm
    a_w = op.apply(w_hat_b.reshape(1, -1)).ravel()
    a_w_norm = np.linalg.norm(a_w)
    if a_w_norm < eps:
        return None
    return np.asarray(a_w / a_w_norm)

def _compute_prefix_B(W: NDArray, P: NDArray, Q: NDArray, k: int) -> NDArray:
    """Compute prefix regression coefficient matrix B_k = W @ inv(P^T W) @ Q^T."""
    PtW = P[:, :k].T @ W[:, :k]
    try:
        R_k = W[:, :k] @ np.linalg.inv(PtW)
    except np.linalg.LinAlgError:
        R_k = W[:, :k] @ np.linalg.pinv(PtW)
    return np.asarray(R_k @ Q[:, :k].T)

def _poppls_fit_greedy(
    X: NDArray, Y: NDArray, operators: list[LinearOperator],
    n_components: int, n_orth: int, eps: float = 1e-12,
) -> dict:
    """Greedy POP-PLS extraction using training R^2 criterion.

    Used when auto_select=False. Each component greedily selects the operator
    with highest R^2(y_res, t_b) on training data. All extractable components
    are returned without model selection.
    """
    n, p = X.shape
    q = Y.shape[1]
    B = len(operators)

    P_orth = None
    if n_orth > 0:
        X, P_orth, _ = _opls_prefilter(X, Y[:, 0], n_orth)

    W = np.zeros((p, n_components), dtype=np.float64)
    T = np.zeros((n, n_components), dtype=np.float64)
    P = np.zeros((p, n_components), dtype=np.float64)
    Q = np.zeros((q, n_components), dtype=np.float64)
    Gamma = np.zeros((n_components, B), dtype=np.float64)
    component_operators = []

    X_res = X.copy()
    Y_res = Y.copy()
    n_extracted = 0

    for k in range(n_components):
        c_k = X_res.T @ Y_res
        if q == 1:
            c_k = c_k[:, 0]
            y_score = Y_res[:, 0]
        else:
            u, s, vt = np.linalg.svd(c_k, full_matrices=False)
            c_k = u[:, 0] * s[0]
            y_score = Y_res @ vt[0]

        c_norm = np.linalg.norm(c_k)
        if c_norm < eps:
            break

        y_norm_sq = np.dot(y_score, y_score)
        if y_norm_sq < eps:
            break

        best_r2 = -np.inf
        best_b = 0
        best_w = None
        best_t = None

        for b, op in enumerate(operators):
            w_b = _compute_weight(op, c_k, eps)
            if w_b is None:
                continue
            t_b = X_res @ w_b
            t_norm_sq = np.dot(t_b, t_b)
            if t_norm_sq < eps:
                continue
            cov_yt = np.dot(y_score, t_b)
            r2 = cov_yt ** 2 / (y_norm_sq * t_norm_sq)
            if r2 > best_r2:
                best_r2 = r2
                best_b = b
                best_w = w_b
                best_t = t_b

        if best_w is None or best_t is None:
            break

        t_k = best_t
        tt = t_k @ t_k
        if tt < eps:
            break

        p_k = (X_res.T @ t_k) / tt
        q_k = (Y_res.T @ t_k) / tt

        W[:, k] = best_w
        T[:, k] = t_k
        P[:, k] = p_k
        Q[:, k] = q_k
        Gamma[k, best_b] = 1.0
        component_operators.append(operators[best_b].name)
        n_extracted = k + 1

        X_res -= np.outer(t_k, p_k)
        Y_res -= np.outer(t_k, q_k)

    W = W[:, :n_extracted]
    T = T[:, :n_extracted]
    P = P[:, :n_extracted]
    Q = Q[:, :n_extracted]
    Gamma = Gamma[:n_extracted]

    B_coefs = np.zeros((n_extracted, p, q), dtype=np.float64)
    for k in range(n_extracted):
        B_coefs[k] = _compute_prefix_B(W, P, Q, k + 1)

    return {
        "n_extracted": n_extracted, "k_selected": n_extracted,
        "W": W, "T": T, "P": P, "Q": Q, "Gamma": Gamma,
        "B_coefs": B_coefs, "P_orth": P_orth,
        "component_operators": component_operators,
    }

def _poppls_holdout_pass(
    X_train: NDArray, Y_train: NDArray, operators: list[LinearOperator],
    n_components: int, n_orth: int, X_val: NDArray, Y_val: NDArray,
    eps: float = 1e-12,
) -> dict | None:
    """Single holdout pass for external validation data.

    At each component, selects the operator that minimizes the prefix RMSE on
    the external validation set. Tracks the best prefix count across all
    components. Uses patience-based early stopping.

    Returns None if no components could be extracted.
    """
    n, p = X_train.shape
    q = Y_train.shape[1]
    B = len(operators)

    # OPLS pre-filter
    P_orth = None
    X_t = X_train.copy()
    X_v = X_val.copy()
    if n_orth > 0:
        X_t, P_orth, _ = _opls_prefilter(X_t, Y_train[:, 0], n_orth)
        for j in range(P_orth.shape[1]):
            p_o = P_orth[:, j]
            t_o = X_v @ p_o
            X_v = X_v - np.outer(t_o, p_o)

    max_comp = min(n - 1, p, n_components)
    if max_comp <= 0:
        return None

    W = np.zeros((p, max_comp), dtype=np.float64)
    P = np.zeros((p, max_comp), dtype=np.float64)
    Q = np.zeros((q, max_comp), dtype=np.float64)
    operator_indices: list[int] = []

    X_res = X_t.copy()
    Y_res = Y_train.copy()
    n_extracted = 0

    # Baseline: no components (predict 0 in centered space)
    baseline_rmse = np.sqrt(np.mean(Y_val ** 2))
    best_rmse = baseline_rmse
    best_k = 0
    patience = 5
    no_improve = 0

    for k in range(max_comp):
        c_k = X_res.T @ Y_res
        if q == 1:
            c_k = c_k[:, 0]
        else:
            u, s, vt = np.linalg.svd(c_k, full_matrices=False)
            c_k = u[:, 0] * s[0]

        c_norm = np.linalg.norm(c_k)
        if c_norm < eps:
            break

        # Evaluate each operator using holdout prefix RMSE
        best_op_rmse = np.inf
        best_op_b = None
        best_op_artifacts = None

        for b, op in enumerate(operators):
            w_b = _compute_weight(op, c_k, eps)
            if w_b is None:
                continue
            t_b = X_res @ w_b
            tt = np.dot(t_b, t_b)
            if tt < eps:
                continue

            p_b = (X_res.T @ t_b) / tt
            q_b = (Y_res.T @ t_b) / tt

            # Build temporary prefix model and evaluate on holdout
            W_tmp = np.column_stack([W[:, :k], w_b]) if k > 0 else w_b.reshape(-1, 1)
            P_tmp = np.column_stack([P[:, :k], p_b]) if k > 0 else p_b.reshape(-1, 1)
            Q_tmp = np.column_stack([Q[:, :k], q_b]) if k > 0 else q_b.reshape(-1, 1)

            B_k = _compute_prefix_B(W_tmp, P_tmp, Q_tmp, k + 1)
            y_pred_val = X_v @ B_k
            rmse = np.sqrt(np.mean((Y_val - y_pred_val) ** 2))

            if rmse < best_op_rmse:
                best_op_rmse = rmse
                best_op_b = b
                best_op_artifacts = (w_b, t_b, p_b, q_b)

        if best_op_b is None or best_op_artifacts is None:
            break

        w_k, t_k, p_k, q_k = best_op_artifacts

        W[:, k] = w_k
        P[:, k] = p_k
        Q[:, k] = q_k
        operator_indices.append(best_op_b)
        n_extracted = k + 1

        # NIPALS deflation
        X_res -= np.outer(t_k, p_k)
        Y_res -= np.outer(t_k, q_k)

        # Track best prefix
        if best_op_rmse < best_rmse:
            best_rmse = best_op_rmse
            best_k = n_extracted
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if n_extracted == 0:
        return None

    return {
        "n_orth": n_orth,
        "k_selected": best_k,
        "operator_indices": operator_indices,
        "best_rmse": best_rmse,
    }

def _poppls_press_pass(
    X: NDArray, Y: NDArray, operators: list[LinearOperator],
    n_components: int, n_orth: int, eps: float = 1e-12,
) -> dict | None:
    """PRESS-based POP-PLS: operator and component selection via LOO-CV.

    Uses the PRESS (Predicted Residual Error Sum of Squares) statistic for
    per-component operator selection and component count determination. No
    holdout split is needed — PRESS approximates leave-one-out cross-validation
    from the full training data using the PLS hat matrix leverages:

        PRESS_k = sum_i (e_i / (1 - h_ii))^2

    where h_ii = sum_j t_{ij}^2 / ||t_j||^2 are the leverage values from the
    orthogonal PLS score matrix.

    Since PRESS uses ALL training data, it provides more stable operator and
    component selections than holdout-based approaches, especially for small
    datasets. The selection is also deterministic (no random splits).

    Returns a complete model dictionary (no refit phase needed since all data
    is used during selection).
    """
    n, p = X.shape
    q = Y.shape[1]
    B_ops = len(operators)

    # OPLS pre-filter
    P_orth = None
    if n_orth > 0:
        X, P_orth, _ = _opls_prefilter(X, Y[:, 0], n_orth)

    # Limit components: n-2 to ensure leverages stay < 1
    max_comp = min(n - 2, p, n_components)
    if max_comp <= 0:
        return None

    W = np.zeros((p, max_comp), dtype=np.float64)
    T_scores = np.zeros((n, max_comp), dtype=np.float64)
    P_load = np.zeros((p, max_comp), dtype=np.float64)
    Q_load = np.zeros((q, max_comp), dtype=np.float64)
    Gamma = np.zeros((max_comp, B_ops), dtype=np.float64)
    operator_indices: list[int] = []
    component_operators: list[str] = []

    # Store X before NIPALS deflation (needed for prefix predictions)
    X_orig = X.copy()

    X_res = X.copy()
    Y_res = Y.copy()
    n_extracted = 0

    # Baseline PRESS: predicting zero (centered data). For LOO with mean-only
    # model: e_{-i} = y_i * n/(n-1), so PRESS_0 = (n/(n-1))^2 * ||Y||_F^2
    baseline_press = (n / (n - 1)) ** 2 * np.sum(Y ** 2)
    best_press = baseline_press
    best_k = 0

    # Cumulative leverages from extracted components.
    # Since NIPALS scores are orthogonal: h_ii = sum_j t_{ij}^2 / ||t_j||^2
    h_cumulative = np.zeros(n, dtype=np.float64)

    patience = 5
    no_improve = 0

    for k in range(max_comp):
        c_k = X_res.T @ Y_res
        if q == 1:
            c_k = c_k[:, 0]
        else:
            u, s, vt = np.linalg.svd(c_k, full_matrices=False)
            c_k = u[:, 0] * s[0]

        if np.linalg.norm(c_k) < eps:
            break

        best_op_press = np.inf
        best_op_b = None
        best_op_artifacts = None
        best_op_h_increment = None

        for b, op in enumerate(operators):
            w_b = _compute_weight(op, c_k, eps)
            if w_b is None:
                continue
            t_b = X_res @ w_b
            tt = np.dot(t_b, t_b)
            if tt < eps:
                continue

            p_b = (X_res.T @ t_b) / tt
            q_b = (Y_res.T @ t_b) / tt

            # Build temporary prefix model
            W_tmp = np.column_stack([W[:, :k], w_b]) if k > 0 else w_b.reshape(-1, 1)
            P_tmp = np.column_stack([P_load[:, :k], p_b]) if k > 0 else p_b.reshape(-1, 1)
            Q_tmp = np.column_stack([Q_load[:, :k], q_b]) if k > 0 else q_b.reshape(-1, 1)

            # Prefix prediction on original (pre-deflation) X
            B_k = _compute_prefix_B(W_tmp, P_tmp, Q_tmp, k + 1)
            y_pred = X_orig @ B_k
            residuals = Y - y_pred

            # Leverage increment from candidate component
            h_increment = t_b ** 2 / tt
            h_total = h_cumulative + h_increment

            # Clip leverages for numerical stability (avoid division by ~0)
            h_clipped = np.minimum(h_total, 1 - 1e-6)

            # PRESS: sum_i (e_i / (1 - h_ii))^2
            press = np.sum((residuals / (1 - h_clipped[:, np.newaxis])) ** 2)

            if press < best_op_press:
                best_op_press = press
                best_op_b = b
                best_op_artifacts = (w_b, t_b, p_b, q_b)
                best_op_h_increment = h_increment

        if best_op_b is None or best_op_artifacts is None or best_op_h_increment is None:
            break

        # Accept this component
        w_k, t_k, p_k, q_k = best_op_artifacts
        W[:, k] = w_k
        T_scores[:, k] = t_k
        P_load[:, k] = p_k
        Q_load[:, k] = q_k
        Gamma[k, best_op_b] = 1.0
        operator_indices.append(best_op_b)
        component_operators.append(operators[best_op_b].name)
        n_extracted = k + 1
        h_cumulative += best_op_h_increment

        # NIPALS deflation
        X_res -= np.outer(t_k, p_k)
        Y_res -= np.outer(t_k, q_k)

        # Track best PRESS
        if best_op_press < best_press:
            best_press = best_op_press
            best_k = n_extracted
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if n_extracted == 0:
        return None

    # Trim to extracted components
    W = W[:, :n_extracted]
    T_scores = T_scores[:, :n_extracted]
    P_load = P_load[:, :n_extracted]
    Q_load = Q_load[:, :n_extracted]
    Gamma = Gamma[:n_extracted]

    # Compute all prefix B coefficients
    B_coefs = np.zeros((n_extracted, p, q), dtype=np.float64)
    for k_idx in range(n_extracted):
        B_coefs[k_idx] = _compute_prefix_B(W, P_load, Q_load, k_idx + 1)

    return {
        "n_extracted": n_extracted,
        "k_selected": best_k,
        "W": W, "T": T_scores, "P": P_load, "Q": Q_load,
        "Gamma": Gamma, "B_coefs": B_coefs, "P_orth": P_orth,
        "component_operators": component_operators,
        "best_press": best_press,
    }

def _poppls_refit(
    X: NDArray, Y: NDArray, operators: list[LinearOperator],
    config: dict, eps: float = 1e-12,
) -> dict:
    """Refit on full data using the determined operator sequence."""
    n, p = X.shape
    q = Y.shape[1]
    B = len(operators)

    n_orth = config["n_orth"]
    k_selected = config["k_selected"]
    op_indices = config["operator_indices"]

    # OPLS pre-filter on full data
    P_orth = None
    if n_orth > 0:
        X, P_orth, _ = _opls_prefilter(X, Y[:, 0], n_orth)

    W = np.zeros((p, k_selected), dtype=np.float64)
    T = np.zeros((n, k_selected), dtype=np.float64)
    P = np.zeros((p, k_selected), dtype=np.float64)
    Q = np.zeros((q, k_selected), dtype=np.float64)
    Gamma = np.zeros((k_selected, B), dtype=np.float64)
    component_operators: list[str] = []

    X_res = X.copy()
    Y_res = Y.copy()
    n_extracted = 0

    for k in range(k_selected):
        op_idx = op_indices[k]
        op = operators[op_idx]

        c_k = X_res.T @ Y_res
        if q == 1:
            c_k = c_k[:, 0]
        else:
            u, s, vt = np.linalg.svd(c_k, full_matrices=False)
            c_k = u[:, 0] * s[0]

        c_norm = np.linalg.norm(c_k)
        if c_norm < eps:
            break

        w_k = _compute_weight(op, c_k, eps)
        if w_k is None:
            break

        t_k = X_res @ w_k
        tt = np.dot(t_k, t_k)
        if tt < eps:
            break

        p_k = (X_res.T @ t_k) / tt
        q_k = (Y_res.T @ t_k) / tt

        W[:, k] = w_k
        T[:, k] = t_k
        P[:, k] = p_k
        Q[:, k] = q_k
        Gamma[k, op_idx] = 1.0
        component_operators.append(op.name)
        n_extracted = k + 1

        X_res -= np.outer(t_k, p_k)
        Y_res -= np.outer(t_k, q_k)

    # Trim
    W = W[:, :n_extracted]
    T = T[:, :n_extracted]
    P = P[:, :n_extracted]
    Q = Q[:, :n_extracted]
    Gamma = Gamma[:n_extracted]

    B_coefs = np.zeros((n_extracted, p, q), dtype=np.float64)
    for k in range(n_extracted):
        B_coefs[k] = _compute_prefix_B(W, P, Q, k + 1)

    return {
        "n_extracted": n_extracted, "k_selected": n_extracted,
        "W": W, "T": T, "P": P, "Q": Q, "Gamma": Gamma,
        "B_coefs": B_coefs, "P_orth": P_orth,
        "component_operators": component_operators,
    }

def _poppls_fit_numpy(
    X: NDArray, Y: NDArray, operators: list[LinearOperator],
    n_components: int, n_orth: int, auto_select: bool,
    X_val: NDArray | None = None, Y_val: NDArray | None = None,
    random_state: int | None = None,
) -> dict:
    """Fit POP-PLS model with automatic model selection.

    Selection strategy depends on available data:

    1. **auto_select=False**: Greedy extraction with R^2 criterion, no model
       selection. Uses all data directly.

    2. **auto_select=True, external validation provided**: Per-component
       holdout selection using the external validation set, with n_orth
       auto-tuning. Refit on full data with the selected configuration.

    3. **auto_select=True, no external validation** (default): PRESS-based
       selection using the full training set. PRESS (Predicted Residual Error
       Sum of Squares) approximates leave-one-out CV via the PLS hat matrix,
       providing stable, deterministic selection without any holdout split.
       No refit phase needed since PRESS uses all data.
    """
    if not auto_select:
        return _poppls_fit_greedy(X, Y, operators, n_components, n_orth)

    n, p = X.shape
    q = Y.shape[1]
    B = len(operators)
    eps = 1e-12

    has_external_val = X_val is not None and Y_val is not None and X_val.shape[0] > 0

    # Auto-tune n_orth: search [0..5] when n_orth=0, [0..n_orth] otherwise
    n_orth_candidates = list(range(n_orth + 1)) if n_orth > 0 else [0, 1, 2, 3, 4, 5]

    if has_external_val:
        assert X_val is not None and Y_val is not None
        # External validation path: holdout selection + refit.
        # Run per-component selection using external validation data for
        # each n_orth candidate, then refit best config on full data.
        best_cfg = None
        best_rmse = np.inf
        for n_orth_cand in n_orth_candidates:
            cfg = _poppls_holdout_pass(
                X, Y, operators, n_components, n_orth_cand,
                X_val, Y_val, eps,
            )
            if cfg is not None and cfg["best_rmse"] < best_rmse:
                best_rmse = cfg["best_rmse"]
                best_cfg = cfg

        if best_cfg is None or best_cfg["k_selected"] == 0:
            return {
                "n_extracted": 0, "k_selected": 0,
                "W": np.empty((p, 0)), "T": np.empty((n, 0)),
                "P": np.empty((p, 0)), "Q": np.empty((q, 0)),
                "Gamma": np.empty((0, B)), "B_coefs": np.empty((0, p, q)),
                "P_orth": None, "component_operators": [],
            }

        return _poppls_refit(X, Y, operators, best_cfg, eps)

    else:
        # Internal selection: PRESS-based (no holdout split needed).
        # For each n_orth candidate, run full PRESS-based POP-PLS on all
        # data. Select the n_orth with lowest PRESS. The resulting model
        # IS the final model (no refit needed).
        candidates = []
        for n_orth_cand in n_orth_candidates:
            result = _poppls_press_pass(
                X, Y, operators, n_components, n_orth_cand, eps,
            )
            if result is not None and result["k_selected"] > 0:
                candidates.append(result)

        if candidates:
            best_model = min(candidates, key=lambda r: r["best_press"])
            return best_model

        # No components improve over baseline — return empty model
        return {
            "n_extracted": 0, "k_selected": 0,
            "W": np.empty((p, 0)), "T": np.empty((n, 0)),
            "P": np.empty((p, 0)), "Q": np.empty((q, 0)),
            "Gamma": np.empty((0, B)), "B_coefs": np.empty((0, p, q)),
            "P_orth": None, "component_operators": [],
        }

# =============================================================================
# POPPLSRegressor
# =============================================================================

class POPPLSRegressor(BaseEstimator, RegressorMixin):
    """Per-Operator-Per-component PLS regressor.

    Selects a potentially DIFFERENT preprocessing operator for EACH PLS
    component, choosing the operator that best reduces prediction error.
    This is the true multi-operator mode: each component independently
    adapts its preprocessing to the spectral features visible in the residual
    at that stage of deflation.

    Includes built-in model selection via PRESS (Predicted Residual Error
    Sum of Squares) or external validation data, with automatic n_orth
    tuning, eliminating the need for external hyperparameter tuning.

    Parameters
    ----------
    n_components : int, default=27
        Maximum number of PLS components to extract. When auto_select=True,
        the actual number used for prediction is determined automatically
        by PRESS minimization.
    operator_bank : list of LinearOperator or None, default=None
        Explicit list of operators. If None, uses pop_pls_operator_bank()
        which provides a compact bank of ~9 diverse operators optimized
        for per-component selection.
    n_orth : int, default=0
        OPLS orthogonal pre-filter control. When auto_select=True and n_orth=0,
        the algorithm auto-searches over [0, 1, 2, 3, 4, 5]. When n_orth > 0,
        it searches [0, 1, ..., n_orth]. When auto_select=False, n_orth is used
        directly without search.
    center : bool, default=True
        Whether to center X and Y (subtract mean).
    scale : bool, default=False
        Whether to scale X and Y to unit variance per column.
        WARNING: per-column scaling destroys spectral shape and cripples
        SG/detrend operators. Only enable if your data is not spectral.
    auto_select : bool, default=True
        If True, automatically select operator sequence, component count,
        and n_orth. Uses PRESS (LOO-CV approximation) when no external
        validation is provided, or holdout RMSE with external validation.
        If False, uses greedy R^2 criterion without model selection.
    random_state : int or None, default=None
        Random state (currently unused since PRESS is deterministic, but
        kept for API compatibility).

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    n_components_ : int
        Actual number of components extracted.
    k_selected_ : int
        Selected number of components for prediction.
    gamma_ : ndarray of shape (n_components_, n_blocks)
        Per-component operator selection matrix. Each row has exactly one
        1.0 entry indicating which operator was selected for that component.
    coef_ : ndarray of shape (n_features, n_targets)
        Regression coefficients using selected components.
    block_names_ : list of str
        Names of operators in the bank.
    component_operators_ : list of str
        Operator name selected for each component (length n_components_).
    selected_n_orth_ : int
        The n_orth value selected by auto-tuning (only set when auto_select=True).

    Notes
    -----
    **Why POP-PLS does not need Optuna (unlike AOM-PLS)**

    AOM-PLS selects ONE operator for ALL components. Since the optimal
    operator is unknown a priori, AOM-PLS relies on external search
    (Optuna, grid search) to find the globally best operator.

    POP-PLS eliminates this external search by making all decisions
    internally:

    1. **Operator selection** is per-component: at each deflation step, the
       operator minimizing PRESS is chosen. PRESS approximates leave-one-out
       CV using the PLS hat matrix leverages, providing stable evaluation
       on the FULL training set (no holdout variance).

    2. **Component count** is selected by tracking PRESS across prefixes.
       Early stopping with patience prevents extracting unnecessary components.

    3. **n_orth** is auto-tuned by comparing PRESS across candidates
       [0, 1, 2, 3, 4, 5].

    4. **No refit phase** is needed: since PRESS uses all training data,
       the model built during selection IS the final model.

    5. **Dedicated operator bank**: pop_pls_operator_bank() provides ~9
       diverse operators (identity, smoothing, derivatives, detrending,
       compositions) instead of AOM's ~38 parametric variants. Fewer,
       more diverse operators reduce selection noise.

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.pop_pls import POPPLSRegressor
    >>> import numpy as np
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(100, 200)
    >>> y = X[:, :5].sum(axis=1) + 0.1 * rng.randn(100)
    >>> model = POPPLSRegressor(n_components=10)
    >>> model.fit(X, y)
    POPPLSRegressor(n_components=10)
    >>> preds = model.predict(X)
    >>> # Each component may use a different operator:
    >>> print(model.get_component_operators())

    See Also
    --------
    AOMPLSRegressor : Single-operator selection for all components.
    SIMPLS : Standard SIMPLS regressor.
    """

    _webapp_meta = {
        "category": "pls",
        "tier": "advanced",
        "tags": ["pls", "pop-pls", "per-component", "preprocessing", "auto-select", "regression"],
    }

    _estimator_type = "regressor"

    def __init__(
        self,
        n_components: int = 27,
        operator_bank: list[LinearOperator] | None = None,
        n_orth: int = 0,
        center: bool = True,
        scale: bool = False,
        auto_select: bool = True,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.operator_bank = operator_bank
        self.n_orth = n_orth
        self.center = center
        self.scale = scale
        self.auto_select = auto_select
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_val: ArrayLike | None = None,
        y_val: ArrayLike | None = None,
    ) -> POPPLSRegressor:
        """Fit the POP-PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        X_val : array-like of shape (n_val, n_features), optional
            Validation data for operator and prefix selection. When provided
            (together with y_val), used directly instead of PRESS-based
            internal selection.
        y_val : array-like of shape (n_val,) or (n_val, n_targets), optional
            Validation targets.

        Returns
        -------
        self : POPPLSRegressor
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        self._y_1d = y.ndim == 1
        if self._y_1d:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Limit components by data dimensions
        max_components = min(n_samples - 1, n_features)
        n_comp = min(self.n_components, max_components)

        # Center and optionally scale
        if self.center:
            self.x_mean_ = X.mean(axis=0)
            self.y_mean_ = y.mean(axis=0)
        else:
            self.x_mean_ = np.zeros(n_features, dtype=np.float64)
            self.y_mean_ = np.zeros(y.shape[1], dtype=np.float64)

        if self.scale:
            self.x_std_ = X.std(axis=0, ddof=1)
            self.y_std_ = y.std(axis=0, ddof=1)
            self.x_std_ = np.where(self.x_std_ < 1e-10, 1.0, self.x_std_)
            self.y_std_ = np.where(self.y_std_ < 1e-10, 1.0, self.y_std_)
        else:
            self.x_std_ = np.ones(n_features, dtype=np.float64)
            self.y_std_ = np.ones(y.shape[1], dtype=np.float64)

        X_centered = (X - self.x_mean_) / self.x_std_
        Y_centered = (y - self.y_mean_) / self.y_std_

        # Initialize operator bank
        operators = self.operator_bank if self.operator_bank is not None else pop_pls_operator_bank()
        if not any(isinstance(op, IdentityOperator) for op in operators):
            operators = [IdentityOperator()] + list(operators)
        self.operators_ = list(operators)
        for op in self.operators_:
            op.initialize(n_features)
        self.block_names_ = [op.name for op in self.operators_]

        # Center/scale validation data
        X_val_c = None
        Y_val_c = None
        if X_val is not None and y_val is not None:
            X_v = np.asarray(X_val, dtype=np.float64)
            y_v = np.asarray(y_val, dtype=np.float64)
            if self._y_1d and y_v.ndim == 1:
                y_v = y_v.reshape(-1, 1)
            X_val_c = (X_v - self.x_mean_) / self.x_std_
            Y_val_c = (y_v - self.y_mean_) / self.y_std_

        # Fit
        artifacts = _poppls_fit_numpy(
            X_centered, Y_centered, self.operators_, n_comp, self.n_orth,
            self.auto_select, X_val_c, Y_val_c, self.random_state,
        )

        # Unpack artifacts
        self.n_components_ = artifacts["n_extracted"]
        self.k_selected_ = artifacts["k_selected"]
        self._W = artifacts["W"]
        self._T = artifacts["T"]
        self._P = artifacts["P"]
        self._Q = artifacts["Q"]
        self.gamma_ = artifacts["Gamma"]
        self._B_coefs = artifacts["B_coefs"]
        self._P_orth = artifacts["P_orth"]
        self.component_operators_ = artifacts["component_operators"]

        # Store regression coefficients for selected prefix
        if self.n_components_ > 0:
            B_selected = self._B_coefs[self.k_selected_ - 1]
            self.coef_ = B_selected * self.y_std_[np.newaxis, :] / self.x_std_[:, np.newaxis]
        else:
            self.coef_ = np.zeros((n_features, y.shape[1]), dtype=np.float64)

        return self

    def predict(
        self,
        X: ArrayLike,
        n_components: int | None = None,
    ) -> NDArray[np.floating]:
        """Predict using the POP-PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        n_components : int, optional
            Number of components to use. If None, uses k_selected_.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.
        """
        check_is_fitted(self, ["x_mean_", "x_std_", "y_mean_", "y_std_", "_B_coefs"])

        X = np.asarray(X, dtype=np.float64)
        X_centered = (X - self.x_mean_) / self.x_std_

        # Apply OPLS filter
        if self._P_orth is not None:
            for j in range(self._P_orth.shape[1]):
                p_o = self._P_orth[:, j]
                t_o = X_centered @ p_o
                X_centered = X_centered - np.outer(t_o, p_o)

        if n_components is None:
            n_components = self.k_selected_
        n_components = min(n_components, self.n_components_)

        y_pred: NDArray[np.floating]
        if n_components == 0:
            y_pred = np.full((X.shape[0], len(self.y_mean_)), self.y_mean_, dtype=np.float64)
        else:
            B_k = self._B_coefs[n_components - 1]
            y_pred_std = X_centered @ B_k
            y_pred = y_pred_std * self.y_std_ + self.y_mean_

        if self._y_1d:
            y_pred = y_pred.ravel()
        return y_pred

    def transform(self, X: ArrayLike) -> NDArray[np.floating]:
        """Transform X to score space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Returns
        -------
        T : ndarray of shape (n_samples, k_selected_)
            X scores.
        """
        check_is_fitted(self, ["x_mean_", "x_std_", "_W"])

        X = np.asarray(X, dtype=np.float64)
        X_centered = (X - self.x_mean_) / self.x_std_

        if self._P_orth is not None:
            for j in range(self._P_orth.shape[1]):
                p_o = self._P_orth[:, j]
                t_o = X_centered @ p_o
                X_centered = X_centered - np.outer(t_o, p_o)

        return np.asarray(X_centered @ self._W[:, :self.k_selected_])

    def get_block_weights(self) -> NDArray[np.floating]:
        """Get per-component operator selection matrix.

        Returns
        -------
        gamma : ndarray of shape (n_components_, n_blocks)
            Selection matrix. Each row has exactly one 1.0 entry indicating
            which operator was selected for that component.
        """
        check_is_fitted(self, ["gamma_"])
        return np.array(self.gamma_, copy=True)

    def get_preprocessing_report(self) -> list[dict]:
        """Get a human-readable report of per-component operator selections.

        Returns
        -------
        report : list of dict
            One entry per component with fields: 'component', 'blocks'
            (list of {name, weight} dicts for non-zero blocks).
        """
        check_is_fitted(self, ["gamma_", "block_names_"])
        report: list[dict] = []
        for k in range(self.n_components_):
            blocks: list[dict[str, str | float]] = []
            for b, name in enumerate(self.block_names_):
                if self.gamma_[k, b] > 1e-6:
                    blocks.append({"name": name, "weight": float(self.gamma_[k, b])})
            blocks.sort(key=lambda x: float(x["weight"]), reverse=True)
            report.append({"component": k + 1, "blocks": blocks})
        return report

    def get_component_operators(self) -> list[str]:
        """Get the operator name selected for each component.

        Returns
        -------
        operators : list of str
            Operator name for each extracted component (length n_components_).
        """
        check_is_fitted(self, ["component_operators_"])
        return list(self.component_operators_)

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            "n_components": self.n_components,
            "operator_bank": self.operator_bank,
            "n_orth": self.n_orth,
            "center": self.center,
            "scale": self.scale,
            "auto_select": self.auto_select,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> POPPLSRegressor:
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        return (
            f"POPPLSRegressor(n_components={self.n_components}, "
            f"n_orth={self.n_orth}, auto_select={self.auto_select})"
        )
