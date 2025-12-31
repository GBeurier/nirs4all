"""Interval PLS (iPLS) regressor for nirs4all.

A sklearn-compatible implementation of Interval PLS for wavelength interval
selection in spectroscopic data. iPLS evaluates PLS models on contiguous
wavelength windows to identify optimal spectral regions.

Supports both NumPy (CPU) and JAX (GPU/TPU) backends.

References
----------
- Norgaard, L., Saudland, A., Wagner, J., Nielsen, J. P., Munck, L., &
  Engelsen, S. B. (2000). Interval partial least-squares regression
  (iPLS): A comparative chemometric study with an example from
  near-infrared spectroscopy. Applied Spectroscopy, 54(3), 413-419.
- Leardi, R., & Nørgaard, L. (2004). Sequential application of backward
  interval partial least squares and genetic algorithms for the
  selection of relevant spectral regions. Journal of Chemometrics,
  18(11), 486-497.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted


def _check_jax_available():
    """Check if JAX is available for GPU acceleration."""
    try:
        import jax  # noqa: F401
        return True
    except ImportError:
        return False


# =============================================================================
# NumPy Backend Implementation
# =============================================================================

def _ipls_fit_numpy(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    n_components: int,
    n_intervals: int,
    interval_width: int | None,
    cv: int,
    scoring: str,
    mode: str,
    combination_method: str,
) -> tuple[
    NDArray[np.floating],  # interval_scores (CV scores for each interval)
    NDArray[np.int_],      # interval_starts (start indices)
    NDArray[np.int_],      # interval_ends (end indices)
    list[int],             # selected_intervals (best intervals)
    list[tuple[int, int]], # selected_regions (start, end tuples)
    NDArray[np.floating],  # coef_ (final coefficients)
    PLSRegression,         # final_pls_ (fitted PLS model)
]:
    """Fit iPLS model using NumPy backend.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training data.
    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    n_components : int
        Number of PLS components.
    n_intervals : int
        Number of equal-width intervals to divide X into.
    interval_width : int or None
        Fixed width for each interval. If None, uses n_intervals.
    cv : int
        Number of cross-validation folds.
    scoring : str
        Scoring metric for cross-validation.
    mode : str
        Selection mode: 'single', 'forward', 'backward'.
    combination_method : str
        How to combine intervals: 'best' or 'union'.

    Returns
    -------
    interval_scores : ndarray of shape (n_intervals,)
        Cross-validation scores for each interval.
    interval_starts : ndarray of shape (n_intervals,)
        Start indices for each interval.
    interval_ends : ndarray of shape (n_intervals,)
        End indices for each interval.
    selected_intervals : list of int
        Indices of selected intervals.
    selected_regions : list of tuples
        (start, end) pairs for selected regions.
    coef_ : ndarray of shape (n_selected_features, n_targets)
        Regression coefficients for selected features.
    final_pls_ : PLSRegression
        Fitted PLS model on selected features.
    """
    n_samples, n_features = X.shape

    # Determine interval boundaries
    if interval_width is not None:
        # Fixed width intervals
        n_intervals_actual = int(np.ceil(n_features / interval_width))
        interval_starts = np.array([i * interval_width for i in range(n_intervals_actual)])
        interval_ends = np.minimum(interval_starts + interval_width, n_features)
    else:
        # Equal-count intervals
        n_intervals_actual = n_intervals
        interval_width_calc = n_features // n_intervals
        remainder = n_features % n_intervals

        interval_starts = np.zeros(n_intervals_actual, dtype=np.int_)
        interval_ends = np.zeros(n_intervals_actual, dtype=np.int_)

        current_start = 0
        for i in range(n_intervals_actual):
            interval_starts[i] = current_start
            # Distribute remainder across first intervals
            width = interval_width_calc + (1 if i < remainder else 0)
            interval_ends[i] = current_start + width
            current_start = interval_ends[i]

    # Evaluate each interval
    interval_scores = np.zeros(n_intervals_actual, dtype=np.float64)

    for i in range(n_intervals_actual):
        start_idx = interval_starts[i]
        end_idx = interval_ends[i]
        X_interval = X[:, start_idx:end_idx]

        # Limit n_components to interval width
        n_comp_interval = min(n_components, end_idx - start_idx, n_samples - 1)
        if n_comp_interval < 1:
            interval_scores[i] = -np.inf
            continue

        try:
            pls = PLSRegression(n_components=n_comp_interval)
            scores = cross_val_score(pls, X_interval, y, cv=cv, scoring=scoring)
            interval_scores[i] = np.mean(scores)
        except Exception:
            interval_scores[i] = -np.inf

    # Select intervals based on mode
    if mode == 'single':
        # Select only the best interval
        best_idx = np.argmax(interval_scores)
        selected_intervals = [int(best_idx)]

    elif mode == 'forward':
        # Forward selection: add intervals until performance decreases
        selected_intervals = []
        remaining = list(range(n_intervals_actual))
        best_score = -np.inf

        while remaining:
            best_candidate = None
            best_candidate_score = -np.inf

            for candidate in remaining:
                # Try adding this interval
                test_intervals = selected_intervals + [candidate]

                # Combine features from selected intervals
                feature_mask = np.zeros(n_features, dtype=bool)
                for idx in test_intervals:
                    feature_mask[interval_starts[idx]:interval_ends[idx]] = True
                X_selected = X[:, feature_mask]

                n_selected_features = X_selected.shape[1]
                n_comp = min(n_components, n_selected_features, n_samples - 1)
                if n_comp < 1:
                    continue

                try:
                    pls = PLSRegression(n_components=n_comp)
                    scores = cross_val_score(pls, X_selected, y, cv=cv, scoring=scoring)
                    score = np.mean(scores)

                    if score > best_candidate_score:
                        best_candidate_score = score
                        best_candidate = candidate
                except Exception:
                    continue

            if best_candidate is not None and best_candidate_score > best_score:
                selected_intervals.append(best_candidate)
                remaining.remove(best_candidate)
                best_score = best_candidate_score
            else:
                break

        if not selected_intervals:
            # Fallback to best single interval
            best_idx = np.argmax(interval_scores)
            selected_intervals = [int(best_idx)]

    elif mode == 'backward':
        # Backward elimination: start with all, remove until performance decreases
        selected_intervals = list(range(n_intervals_actual))

        # Get initial score with all intervals
        feature_mask = np.ones(n_features, dtype=bool)
        n_comp = min(n_components, n_features, n_samples - 1)
        pls = PLSRegression(n_components=n_comp)
        scores = cross_val_score(pls, X, y, cv=cv, scoring=scoring)
        best_score = np.mean(scores)

        improved = True
        while len(selected_intervals) > 1 and improved:
            improved = False
            worst_interval = None
            best_after_removal = best_score

            for candidate in selected_intervals:
                # Try removing this interval
                test_intervals = [i for i in selected_intervals if i != candidate]

                # Combine features from remaining intervals
                feature_mask = np.zeros(n_features, dtype=bool)
                for idx in test_intervals:
                    feature_mask[interval_starts[idx]:interval_ends[idx]] = True
                X_selected = X[:, feature_mask]

                n_selected_features = X_selected.shape[1]
                n_comp = min(n_components, n_selected_features, n_samples - 1)
                if n_comp < 1:
                    continue

                try:
                    pls = PLSRegression(n_components=n_comp)
                    scores = cross_val_score(pls, X_selected, y, cv=cv, scoring=scoring)
                    score = np.mean(scores)

                    if score > best_after_removal:
                        best_after_removal = score
                        worst_interval = candidate
                except Exception:
                    continue

            if worst_interval is not None and best_after_removal > best_score:
                selected_intervals.remove(worst_interval)
                best_score = best_after_removal
                improved = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Build final model on selected intervals
    if combination_method == 'union':
        # Use union of all selected interval features
        feature_mask = np.zeros(n_features, dtype=bool)
        for idx in selected_intervals:
            feature_mask[interval_starts[idx]:interval_ends[idx]] = True
    else:  # 'best'
        # Use only the best interval
        best_idx = selected_intervals[0]
        feature_mask = np.zeros(n_features, dtype=bool)
        feature_mask[interval_starts[best_idx]:interval_ends[best_idx]] = True

    X_final = X[:, feature_mask]
    n_final_features = X_final.shape[1]
    n_comp_final = min(n_components, n_final_features, n_samples - 1)

    final_pls = PLSRegression(n_components=n_comp_final)
    final_pls.fit(X_final, y)

    # Build selected regions list
    selected_regions = [(int(interval_starts[i]), int(interval_ends[i]))
                        for i in selected_intervals]

    # Get coefficients
    coef_ = final_pls.coef_

    return (
        interval_scores,
        interval_starts,
        interval_ends,
        selected_intervals,
        selected_regions,
        coef_,
        final_pls,
    )


def _ipls_predict_numpy(
    X: NDArray[np.floating],
    interval_starts: NDArray[np.int_],
    interval_ends: NDArray[np.int_],
    selected_intervals: list[int],
    combination_method: str,
    final_pls: PLSRegression,
) -> NDArray[np.floating]:
    """Predict using fitted iPLS model with NumPy.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data to predict.
    interval_starts : ndarray
        Start indices for each interval.
    interval_ends : ndarray
        End indices for each interval.
    selected_intervals : list of int
        Indices of selected intervals.
    combination_method : str
        How intervals were combined.
    final_pls : PLSRegression
        Fitted PLS model.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Predicted values.
    """
    n_features = X.shape[1]

    if combination_method == 'union':
        feature_mask = np.zeros(n_features, dtype=bool)
        for idx in selected_intervals:
            feature_mask[interval_starts[idx]:interval_ends[idx]] = True
    else:
        best_idx = selected_intervals[0]
        feature_mask = np.zeros(n_features, dtype=bool)
        feature_mask[interval_starts[best_idx]:interval_ends[best_idx]] = True

    X_selected = X[:, feature_mask]
    return final_pls.predict(X_selected)


# =============================================================================
# JAX Backend Implementation (Optimized with vmap and JIT)
# =============================================================================

# Global cache for JIT-compiled JAX functions
_JAX_IPLS_CACHE: dict | None = None


def _build_jax_ipls_functions():
    """Build and JIT-compile all JAX iPLS functions.

    Returns a dictionary of compiled functions. This is called once and cached.
    The functions use vmap for parallel evaluation across intervals and CV folds.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from functools import partial as functools_partial

    # Enable float64 for numerical precision
    jax.config.update("jax_enable_x64", True)

    # =========================================================================
    # Core PLS Implementation (NIPALS with fixed array sizes)
    # =========================================================================

    def _nipals_pls_fixed(X: jax.Array, y: jax.Array, n_components: int):
        """NIPALS PLS algorithm with pre-allocated arrays.

        Uses lax.fori_loop with index-based updates for JIT compatibility.

        Parameters
        ----------
        X : jax.Array of shape (n_samples, n_features)
            Centered predictor matrix.
        y : jax.Array of shape (n_samples, 1)
            Centered response vector.
        n_components : int
            Number of PLS components.

        Returns
        -------
        B : jax.Array of shape (n_features, 1)
            Regression coefficients.
        """
        n_samples, n_features = X.shape
        n_targets = y.shape[1] if y.ndim > 1 else 1

        # Pre-allocate arrays
        W = jnp.zeros((n_features, n_components))
        P = jnp.zeros((n_features, n_components))
        Q = jnp.zeros((n_targets, n_components))

        def nipals_iteration(a, carry):
            E, F, W, P, Q = carry

            # Weight vector: w = E.T @ f / ||E.T @ f||
            f = F[:, 0] if F.ndim > 1 else F
            w = E.T @ f
            w_norm = jnp.linalg.norm(w)
            w = jnp.where(w_norm > 1e-12, w / w_norm, jnp.zeros_like(w))

            # Score: t = E @ w
            t = E @ w
            tt = jnp.dot(t, t)
            tt = jnp.maximum(tt, 1e-12)

            # X loading: p = E.T @ t / (t.T @ t)
            p = (E.T @ t) / tt

            # Y loading: q = F.T @ t / (t.T @ t)
            q = (F.T @ t) / tt

            # Store at index a
            W = W.at[:, a].set(w)
            P = P.at[:, a].set(p)
            Q = Q.at[:, a].set(q.ravel() if q.ndim > 0 else q)

            # Deflate
            E = E - jnp.outer(t, p)
            F = F - jnp.outer(t, q.ravel() if q.ndim > 0 else q)

            return (E, F, W, P, Q)

        # Run NIPALS iterations using fori_loop
        init_carry = (X, y, W, P, Q)

        def body_fn(a, carry):
            return nipals_iteration(a, carry)

        (_, _, W, P, Q) = lax.fori_loop(0, n_components, body_fn, init_carry)

        # Compute regression coefficients: B = W @ inv(P.T @ W) @ Q.T
        PtW = P.T @ W
        # Use pseudoinverse for numerical stability
        R = W @ jnp.linalg.pinv(PtW)
        B = R @ Q.T

        return B

    # =========================================================================
    # CV Fold Scoring
    # =========================================================================

    def _score_single_fold(
        X: jax.Array,
        y: jax.Array,
        n_components: int,
        fold_idx: jax.Array,
        fold_size: int,
        n_samples: int,
    ) -> jax.Array:
        """Compute R2 score for a single CV fold.

        Uses jnp.where with size for JIT-compatible indexing.
        """
        # Compute fold boundaries
        test_start = fold_idx * fold_size
        test_end = jnp.minimum((fold_idx + 1) * fold_size, n_samples)

        # Create indices for train/test split using masking
        indices = jnp.arange(n_samples)
        is_test = (indices >= test_start) & (indices < test_end)
        is_train = ~is_test

        # Extract train and test sets
        train_indices = jnp.where(is_train, size=n_samples - fold_size)[0]
        test_indices = jnp.where(is_test, size=fold_size)[0]

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        # Center based on training data
        X_mean = jnp.mean(X_train, axis=0)
        y_mean = jnp.mean(y_train, axis=0)

        X_train_c = X_train - X_mean
        y_train_c = y_train - y_mean
        X_test_c = X_test - X_mean

        # Fit PLS
        B = _nipals_pls_fixed(X_train_c, y_train_c, n_components)

        # Predict
        y_pred = X_test_c @ B + y_mean

        # R2 score
        y_test_flat = y_test.ravel()
        y_pred_flat = y_pred.ravel()

        ss_res = jnp.sum((y_test_flat - y_pred_flat) ** 2)
        ss_tot = jnp.sum((y_test_flat - jnp.mean(y_test_flat)) ** 2)
        r2 = 1.0 - ss_res / jnp.maximum(ss_tot, 1e-12)

        return r2

    def _cv_score(
        X: jax.Array,
        y: jax.Array,
        n_components: int,
        cv_folds: int,
    ) -> jax.Array:
        """Compute mean CV score using vmap over folds.

        Parameters
        ----------
        X : jax.Array of shape (n_samples, n_features)
            Feature matrix.
        y : jax.Array of shape (n_samples, 1)
            Target vector.
        n_components : int
            Number of PLS components.
        cv_folds : int
            Number of CV folds.

        Returns
        -------
        mean_r2 : jax.Array
            Mean R2 score across folds.
        """
        n_samples = X.shape[0]
        fold_size = n_samples // cv_folds

        # Use vmap to parallelize over folds
        fold_indices = jnp.arange(cv_folds)
        score_fold = functools_partial(
            _score_single_fold,
            X, y, n_components,
            fold_size=fold_size,
            n_samples=n_samples,
        )
        scores = jax.vmap(score_fold)(fold_indices)

        return jnp.mean(scores)

    # =========================================================================
    # Interval Evaluation
    # =========================================================================

    def _eval_single_interval_static(
        X: jax.Array,
        y: jax.Array,
        start_idx: jax.Array,
        end_idx: jax.Array,
        n_components: int,
        cv_folds: int,
        max_interval_width: int,
    ) -> jax.Array:
        """Evaluate a single interval with CV.

        Uses max_interval_width for static shape compilation.
        """
        n_samples = X.shape[0]

        # Extract interval using dynamic_slice with static size
        # Pad to max_interval_width for consistent shapes
        X_interval = lax.dynamic_slice(
            X,
            (0, start_idx),
            (n_samples, max_interval_width)
        )

        # Create mask for valid features
        actual_width = end_idx - start_idx
        valid_mask = jnp.arange(max_interval_width) < actual_width

        # Mask out invalid features
        X_interval = jnp.where(valid_mask[None, :], X_interval, 0.0)

        # Limit components
        n_comp_actual = min(n_components, max_interval_width, n_samples - 1)
        n_comp_actual = max(n_comp_actual, 1)

        return _cv_score(X_interval, y, n_comp_actual, cv_folds)

    # JIT-compile the interval evaluation with static arguments
    @functools_partial(jax.jit, static_argnums=(4, 5, 6))
    def eval_all_intervals_jit(
        X: jax.Array,
        y: jax.Array,
        interval_starts: jax.Array,
        interval_ends: jax.Array,
        n_components: int,
        cv_folds: int,
        max_interval_width: int,
    ) -> jax.Array:
        """Evaluate all intervals using vmap.

        Parameters
        ----------
        X : jax.Array of shape (n_samples, n_features)
            Feature matrix.
        y : jax.Array of shape (n_samples, 1)
            Target vector.
        interval_starts : jax.Array of shape (n_intervals,)
            Start indices for each interval.
        interval_ends : jax.Array of shape (n_intervals,)
            End indices for each interval.
        n_components : int
            Number of PLS components (static).
        cv_folds : int
            Number of CV folds (static).
        max_interval_width : int
            Maximum interval width for static shapes (static).

        Returns
        -------
        scores : jax.Array of shape (n_intervals,)
            CV scores for each interval.
        """
        def eval_interval(start_end):
            start, end = start_end[0], start_end[1]
            return _eval_single_interval_static(
                X, y, start, end, n_components, cv_folds, max_interval_width
            )

        # Stack starts and ends for vmap
        interval_bounds = jnp.stack([interval_starts, interval_ends], axis=1)

        # Use vmap for parallel evaluation
        scores = jax.vmap(eval_interval)(interval_bounds)

        return scores

    # =========================================================================
    # Parallel Interval Evaluation (Fixed Width - For Maximum Performance)
    # =========================================================================

    @functools_partial(jax.jit, static_argnums=(3, 4, 5))
    def eval_intervals_parallel(
        X: jax.Array,
        y: jax.Array,
        interval_indices: jax.Array,
        interval_width: int,
        n_components: int,
        cv_folds: int,
    ) -> jax.Array:
        """Evaluate intervals with fixed width using full vmap parallelization.

        This is the fastest path when all intervals have the same width.
        Uses vmap for true parallel evaluation on GPU/TPU.

        Parameters
        ----------
        X : jax.Array of shape (n_samples, n_features)
            Feature matrix.
        y : jax.Array of shape (n_samples, 1)
            Target vector.
        interval_indices : jax.Array of shape (n_intervals,)
            Start indices for each interval.
        interval_width : int
            Fixed width for all intervals (static for JIT).
        n_components : int
            Number of PLS components (static for JIT).
        cv_folds : int
            Number of CV folds (static for JIT).

        Returns
        -------
        scores : jax.Array of shape (n_intervals,)
            CV scores for each interval.
        """
        n_samples = X.shape[0]

        # Limit n_components based on interval width
        n_comp_actual = min(n_components, interval_width, n_samples - 1)
        n_comp_actual = max(n_comp_actual, 1)

        def eval_single(start_idx):
            X_interval = lax.dynamic_slice(
                X,
                (0, start_idx),
                (n_samples, interval_width)
            )
            return _cv_score(X_interval, y, n_comp_actual, cv_folds)

        # Fully parallel evaluation with vmap
        scores = jax.vmap(eval_single)(interval_indices)
        return scores

    # =========================================================================
    # Combination Scoring (for forward/backward selection)
    # =========================================================================

    @functools_partial(jax.jit, static_argnums=(5, 6))
    def score_interval_combination(
        X: jax.Array,
        y: jax.Array,
        interval_mask: jax.Array,
        interval_starts: jax.Array,
        interval_ends: jax.Array,
        n_components: int,
        cv_folds: int,
    ) -> jax.Array:
        """Score a combination of intervals.

        Parameters
        ----------
        X : jax.Array of shape (n_samples, n_features)
            Feature matrix.
        y : jax.Array of shape (n_samples, 1)
            Target vector.
        interval_mask : jax.Array of shape (n_intervals,)
            Boolean mask indicating which intervals to include.
        interval_starts : jax.Array of shape (n_intervals,)
            Start indices for each interval.
        interval_ends : jax.Array of shape (n_intervals,)
            End indices for each interval.
        n_components : int
            Number of PLS components (static).
        cv_folds : int
            Number of CV folds (static).

        Returns
        -------
        score : jax.Array
            CV score for the combined intervals.
        """
        n_samples, n_features = X.shape

        # Build feature mask from interval mask
        feature_indices = jnp.arange(n_features)

        def is_in_selected_interval(feat_idx):
            # Check if feature is in any selected interval
            in_interval = (feat_idx >= interval_starts) & (feat_idx < interval_ends)
            return jnp.any(in_interval & interval_mask)

        feature_mask = jax.vmap(is_in_selected_interval)(feature_indices)

        # Count selected features
        n_selected = jnp.sum(feature_mask)

        # Extract selected features (using gather with mask)
        # Use n_features as max size for consistent compilation
        selected_indices = jnp.where(feature_mask, size=n_features, fill_value=0)[0]

        X_selected = X[:, selected_indices]

        # Mask out invalid features beyond n_selected
        valid_mask = jnp.arange(n_features) < n_selected
        X_selected = jnp.where(valid_mask[None, :], X_selected, 0.0)

        # Compute score - use static n_components (capped at compile time)
        n_comp = min(n_components, n_features, n_samples - 1)
        n_comp = max(n_comp, 1)

        # The score function handles the actual feature count internally
        # We pass the full padded array but only use valid features
        return _cv_score(X_selected, y, n_comp, cv_folds)

    # =========================================================================
    # Final PLS Fit (for prediction)
    # =========================================================================

    @functools_partial(jax.jit, static_argnums=(2,))
    def pls_fit_final(
        X: jax.Array,
        y: jax.Array,
        n_components: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Fit PLS model and return coefficients and means.

        Returns
        -------
        B : regression coefficients
        X_mean : feature means
        y_mean : target mean
        """
        X_mean = jnp.mean(X, axis=0)
        y_mean = jnp.mean(y, axis=0)

        X_centered = X - X_mean
        y_centered = y - y_mean

        B = _nipals_pls_fixed(X_centered, y_centered, n_components)

        return B, X_mean, y_mean

    @jax.jit
    def pls_predict(
        X: jax.Array,
        B: jax.Array,
        X_mean: jax.Array,
        y_mean: jax.Array,
    ) -> jax.Array:
        """Predict with fitted PLS model."""
        X_centered = X - X_mean
        return X_centered @ B + y_mean

    # Return all functions in a dictionary
    return {
        'eval_all_intervals': eval_all_intervals_jit,
        'eval_intervals_parallel': eval_intervals_parallel,
        'score_combination': score_interval_combination,
        'pls_fit': pls_fit_final,
        'pls_predict': pls_predict,
    }


def _get_jax_ipls_functions():
    """Get cached JAX iPLS functions (with lazy initialization)."""
    global _JAX_IPLS_CACHE
    if _JAX_IPLS_CACHE is None:
        _JAX_IPLS_CACHE = _build_jax_ipls_functions()
    return _JAX_IPLS_CACHE


def _ipls_fit_jax(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    n_intervals: int,
    interval_width: int | None,
    cv: int,
    mode: str,
    combination_method: str,
) -> tuple:
    """Fit iPLS model using optimized JAX backend.

    This implementation uses JIT-compiled functions with vmap for
    parallel interval and CV fold evaluation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training data.
    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    n_components : int
        Number of PLS components.
    n_intervals : int
        Number of intervals.
    interval_width : int or None
        Fixed interval width (if None, computed from n_intervals).
    cv : int
        Number of CV folds.
    mode : str
        Selection mode ('single', 'forward', 'backward').
    combination_method : str
        How to combine intervals ('best', 'union').

    Returns
    -------
    interval_scores : ndarray
        CV scores for each interval.
    interval_starts : ndarray
        Start indices for each interval.
    interval_ends : ndarray
        End indices for each interval.
    selected_intervals : list of int
        Indices of selected intervals.
    selected_regions : list of tuple
        (start, end) pairs for selected regions.
    coef_ : ndarray
        Regression coefficients.
    B_jax : jax.Array
        JAX coefficients for prediction.
    X_mean : jax.Array
        Feature means.
    y_mean : jax.Array
        Target mean.
    feature_mask : ndarray
        Boolean mask for selected features.
    """
    import jax.numpy as jnp

    funcs = _get_jax_ipls_functions()

    n_samples, n_features = X.shape

    # Ensure y is 2D
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Convert to JAX arrays
    X_jax = jnp.asarray(X)
    y_jax = jnp.asarray(y)

    # Compute interval boundaries
    if interval_width is not None:
        n_intervals_actual = int(np.ceil(n_features / interval_width))
        interval_starts = np.array([i * interval_width for i in range(n_intervals_actual)])
        interval_ends = np.minimum(interval_starts + interval_width, n_features)
    else:
        n_intervals_actual = n_intervals
        base_width = n_features // n_intervals
        remainder = n_features % n_intervals

        interval_starts = np.zeros(n_intervals_actual, dtype=np.int64)
        interval_ends = np.zeros(n_intervals_actual, dtype=np.int64)

        current = 0
        for i in range(n_intervals_actual):
            interval_starts[i] = current
            width = base_width + (1 if i < remainder else 0)
            interval_ends[i] = current + width
            current = interval_ends[i]

    # Check if all intervals have the same width (enables parallel vmap)
    widths = interval_ends - interval_starts
    uniform_width = np.all(widths == widths[0])
    max_interval_width = int(np.max(widths))

    # Convert interval bounds to JAX
    starts_jax = jnp.asarray(interval_starts)
    ends_jax = jnp.asarray(interval_ends)

    # Evaluate all intervals
    if uniform_width and interval_width is None:
        # Use parallel vmap path (faster)
        interval_scores_jax = funcs['eval_intervals_parallel'](
            X_jax, y_jax, starts_jax,
            int(widths[0]), n_components, cv
        )
    else:
        # Use vmap path with max width for static shapes
        interval_scores_jax = funcs['eval_all_intervals'](
            X_jax, y_jax, starts_jax, ends_jax,
            n_components, cv, max_interval_width
        )

    # Block until computation is done (for accurate timing)
    interval_scores_jax.block_until_ready()

    interval_scores = np.asarray(interval_scores_jax)

    # Selection logic (Python control flow)
    if mode == 'single':
        best_idx = int(np.argmax(interval_scores))
        selected_intervals = [best_idx]

    elif mode == 'forward':
        # Forward selection using JAX for scoring combinations
        selected_intervals = []
        remaining = list(range(n_intervals_actual))
        best_score = -np.inf

        while remaining:
            best_candidate = None
            best_candidate_score = -np.inf

            # Evaluate adding each remaining interval
            for candidate in remaining:
                test_intervals = selected_intervals + [candidate]

                # Create interval mask
                interval_mask = np.zeros(n_intervals_actual, dtype=bool)
                for idx in test_intervals:
                    interval_mask[idx] = True

                # Score with JAX
                mask_jax = jnp.asarray(interval_mask)
                score = float(funcs['score_combination'](
                    X_jax, y_jax, mask_jax, starts_jax, ends_jax, n_components, cv
                ))

                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = candidate

            if best_candidate is not None and best_candidate_score > best_score:
                selected_intervals.append(best_candidate)
                remaining.remove(best_candidate)
                best_score = best_candidate_score
            else:
                break

        if not selected_intervals:
            best_idx = int(np.argmax(interval_scores))
            selected_intervals = [best_idx]

    elif mode == 'backward':
        # Backward elimination using JAX for scoring
        selected_intervals = list(range(n_intervals_actual))

        # Get initial score with all intervals
        all_mask = jnp.ones(n_intervals_actual, dtype=bool)
        best_score = float(funcs['score_combination'](
            X_jax, y_jax, all_mask, starts_jax, ends_jax, n_components, cv
        ))

        improved = True
        while len(selected_intervals) > 1 and improved:
            improved = False
            worst_interval = None
            best_after_removal = best_score

            for candidate in selected_intervals:
                # Try removing this interval
                test_intervals = [i for i in selected_intervals if i != candidate]

                interval_mask = np.zeros(n_intervals_actual, dtype=bool)
                for idx in test_intervals:
                    interval_mask[idx] = True

                mask_jax = jnp.asarray(interval_mask)
                score = float(funcs['score_combination'](
                    X_jax, y_jax, mask_jax, starts_jax, ends_jax, n_components, cv
                ))

                if score > best_after_removal:
                    best_after_removal = score
                    worst_interval = candidate

            if worst_interval is not None and best_after_removal > best_score:
                selected_intervals.remove(worst_interval)
                best_score = best_after_removal
                improved = True

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Build feature mask
    feature_mask = np.zeros(n_features, dtype=bool)
    if combination_method == 'union':
        for idx in selected_intervals:
            feature_mask[interval_starts[idx]:interval_ends[idx]] = True
    else:  # 'best'
        best_idx = selected_intervals[0]
        feature_mask[interval_starts[best_idx]:interval_ends[best_idx]] = True

    # Build selected regions list
    selected_regions = [
        (int(interval_starts[i]), int(interval_ends[i]))
        for i in selected_intervals
    ]

    # Fit final PLS model using JAX
    X_final = X[:, feature_mask]
    n_final_features = X_final.shape[1]
    n_comp_final = min(n_components, n_final_features, n_samples - 1)

    X_final_jax = jnp.asarray(X_final)
    B_jax, X_mean_jax, y_mean_jax = funcs['pls_fit'](X_final_jax, y_jax, n_comp_final)

    # Convert coefficients to numpy
    coef_ = np.asarray(B_jax)

    return (
        interval_scores,
        interval_starts,
        interval_ends,
        selected_intervals,
        selected_regions,
        coef_,
        B_jax,
        X_mean_jax,
        y_mean_jax,
        feature_mask,
    )


# =============================================================================
# IntervalPLS Estimator Class
# =============================================================================

class IntervalPLS(BaseEstimator, RegressorMixin):
    """Interval Partial Least Squares (iPLS) regressor.

    iPLS evaluates PLS models on contiguous wavelength intervals to identify
    optimal spectral regions for prediction. This is particularly useful for
    NIR spectroscopy where not all wavelengths contribute equally to the
    prediction.

    The algorithm divides the spectrum into intervals and evaluates each
    interval (or combination of intervals) using cross-validation. Different
    selection modes are available:
    - 'single': Select only the best performing interval
    - 'forward': Iteratively add intervals that improve performance
    - 'backward': Start with all intervals and remove those that don't help

    Parameters
    ----------
    n_components : int, default=5
        Number of PLS components to extract for each interval model.
    n_intervals : int, default=10
        Number of equal-width intervals to divide X into.
    interval_width : int, optional
        Fixed width for each interval. If specified, overrides n_intervals.
    cv : int, default=5
        Number of cross-validation folds for interval evaluation.
    scoring : str, default='r2'
        Scoring metric for cross-validation. Supports sklearn metrics
        like 'r2', 'neg_mean_squared_error', etc.
    mode : {'single', 'forward', 'backward'}, default='forward'
        Interval selection mode:
        - 'single': Use only the best single interval
        - 'forward': Forward selection of intervals
        - 'backward': Backward elimination of intervals
    combination_method : {'best', 'union'}, default='union'
        How to combine selected intervals for the final model:
        - 'best': Use only the single best interval
        - 'union': Use union of all selected intervals
    backend : str, default='numpy'
        Computational backend:
        - 'numpy': NumPy backend (CPU only, default)
        - 'jax': JAX backend (supports GPU/TPU acceleration)
        Note: JAX backend accelerates interval evaluation but final
        model fitting uses sklearn for compatibility.

    Attributes
    ----------
    n_features_in\_ : int
        Number of features seen during fit.
    n_components\_ : int
        Actual number of components used in final model.
    interval_scores\_ : ndarray of shape (n_intervals,)
        Cross-validation scores for each interval.
    interval_starts\_ : ndarray of shape (n_intervals,)
        Start indices for each interval.
    interval_ends\_ : ndarray of shape (n_intervals,)
        End indices for each interval.
    n_intervals\_ : int
        Actual number of intervals.
    selected_intervals\_ : list of int
        Indices of selected intervals.
    selected_regions\_ : list of tuple
        (start, end) pairs for selected spectral regions.
    coef\_ : ndarray of shape (n_selected_features, n_targets)
        Regression coefficients for selected features.
    feature_mask\_ : ndarray of shape (n_features,)
        Boolean mask indicating selected features.

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.ipls import IntervalPLS
    >>> import numpy as np
    >>> # Generate sample spectral data
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 200)  # 200 wavelengths
    >>> y = X[:, 50:70].sum(axis=1) + 0.1 * np.random.randn(100)  # Signal in 50-70
    >>> # Fit iPLS to find informative regions
    >>> model = IntervalPLS(n_components=5, n_intervals=10, mode='forward')
    >>> model.fit(X, y)
    IntervalPLS(n_components=5, n_intervals=10, mode='forward')
    >>> # See which intervals were selected
    >>> print(f"Selected intervals: {model.selected_intervals_}")
    >>> print(f"Selected regions: {model.selected_regions_}")
    >>> # Predict
    >>> predictions = model.predict(X)

    Notes
    -----
    iPLS is particularly effective for NIR spectroscopy because:
    1. Different spectral regions contain different chemical information
    2. Some regions may be dominated by noise or uninformative signals
    3. Selecting optimal intervals can improve both prediction and interpretation

    The JAX backend provides acceleration for interval evaluation when using
    GPU/TPU, which is beneficial when evaluating many intervals.

    See Also
    --------
    sklearn.cross_decomposition.PLSRegression : Standard PLS regression.
    SIMPLS : SIMPLS algorithm implementation.

    References
    ----------
    - Norgaard, L., et al. (2000). Interval partial least-squares
      regression (iPLS): A comparative chemometric study with an example
      from near-infrared spectroscopy. Applied Spectroscopy, 54(3), 413-419.
    """

    # Explicitly declare estimator type for sklearn compatibility (e.g., StackingRegressor)
    _estimator_type = "regressor"

    def __init__(
        self,
        n_components: int = 5,
        n_intervals: int = 10,
        interval_width: int | None = None,
        cv: int = 5,
        scoring: str = 'r2',
        mode: Literal['single', 'forward', 'backward'] = 'forward',
        combination_method: Literal['best', 'union'] = 'union',
        backend: str = 'numpy',
    ):
        """Initialize IntervalPLS regressor.

        Parameters
        ----------
        n_components : int, default=5
            Number of PLS components.
        n_intervals : int, default=10
            Number of intervals to divide spectrum into.
        interval_width : int, optional
            Fixed interval width (overrides n_intervals).
        cv : int, default=5
            Number of cross-validation folds.
        scoring : str, default='r2'
            Scoring metric for CV.
        mode : str, default='forward'
            Selection mode ('single', 'forward', 'backward').
        combination_method : str, default='union'
            How to combine intervals ('best', 'union').
        backend : str, default='numpy'
            Computational backend ('numpy' or 'jax').
        """
        self.n_components = n_components
        self.n_intervals = n_intervals
        self.interval_width = interval_width
        self.cv = cv
        self.scoring = scoring
        self.mode = mode
        self.combination_method = combination_method
        self.backend = backend

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "IntervalPLS":
        """Fit the IntervalPLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (e.g., spectral data).
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : IntervalPLS
            Fitted estimator.

        Raises
        ------
        ValueError
            If backend is invalid or mode is invalid.
        ImportError
            If backend is 'jax' and JAX is not installed.
        """
        # Validate backend
        if self.backend not in ('numpy', 'jax'):
            raise ValueError(
                f"backend must be 'numpy' or 'jax', got '{self.backend}'"
            )

        if self.backend == 'jax' and not _check_jax_available():
            raise ImportError(
                "JAX is required for IntervalPLS with backend='jax'. "
                "Install it with: pip install jax\n"
                "For GPU support: pip install jax[cuda12]"
            )

        if self.mode not in ('single', 'forward', 'backward'):
            raise ValueError(
                f"mode must be 'single', 'forward', or 'backward', got '{self.mode}'"
            )

        if self.combination_method not in ('best', 'union'):
            raise ValueError(
                f"combination_method must be 'best' or 'union', "
                f"got '{self.combination_method}'"
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Handle 1D y
        self._y_1d = y.ndim == 1
        if self._y_1d:
            y_fit = y.reshape(-1, 1)
        else:
            y_fit = y

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Fit using the appropriate backend
        if self.backend == 'jax':
            # Full JAX backend - uses JIT-compiled functions with vmap
            (
                interval_scores,
                interval_starts,
                interval_ends,
                selected_intervals,
                selected_regions,
                coef_,
                self._B_jax,
                self._X_mean_jax,
                self._y_mean_jax,
                feature_mask,
            ) = _ipls_fit_jax(
                X, y_fit, self.n_components, self.n_intervals,
                self.interval_width, self.cv, self.mode,
                self.combination_method,
            )

            self.interval_scores_ = interval_scores
            self.interval_starts_ = interval_starts
            self.interval_ends_ = interval_ends
            self.feature_mask_ = feature_mask

            # Create a sklearn PLS for compatibility (used by predict if needed)
            X_final = X[:, feature_mask]
            n_final_features = X_final.shape[1]
            n_comp_final = min(self.n_components, n_final_features, n_samples - 1)
            final_pls = PLSRegression(n_components=n_comp_final)
            final_pls.fit(X_final, y_fit)
            self._final_pls = final_pls
            self._use_jax_predict = True

        else:
            # Pure NumPy backend
            (
                interval_scores,
                interval_starts,
                interval_ends,
                selected_intervals,
                selected_regions,
                coef_,
                final_pls,
            ) = _ipls_fit_numpy(
                X, y_fit, self.n_components, self.n_intervals,
                self.interval_width, self.cv, self.scoring,
                self.mode, self.combination_method,
            )

            self.interval_scores_ = interval_scores
            self.interval_starts_ = interval_starts
            self.interval_ends_ = interval_ends
            self._final_pls = final_pls
            self._use_jax_predict = False

            # Build feature mask
            self.feature_mask_ = np.zeros(n_features, dtype=bool)
            if self.combination_method == 'union':
                for idx in selected_intervals:
                    start = self.interval_starts_[idx]
                    end = self.interval_ends_[idx]
                    self.feature_mask_[start:end] = True
            else:
                best_idx = selected_intervals[0]
                start = self.interval_starts_[best_idx]
                end = self.interval_ends_[best_idx]
                self.feature_mask_[start:end] = True

        # Store common results
        self.n_intervals_ = len(self.interval_starts_)
        self.selected_intervals_ = selected_intervals
        self.selected_regions_ = selected_regions
        self.coef_ = coef_
        self.n_components_ = self._final_pls.n_components

        return self

    def predict(
        self,
        X: ArrayLike,
    ) -> NDArray[np.floating]:
        """Predict using the fitted IntervalPLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.
        """
        check_is_fitted(self, ['feature_mask_', '_final_pls'])

        X = np.asarray(X, dtype=np.float64)

        # Use JAX prediction if available and was used for fitting
        if getattr(self, '_use_jax_predict', False) and self.backend == 'jax':
            import jax.numpy as jnp

            funcs = _get_jax_ipls_functions()
            X_selected = X[:, self.feature_mask_]
            X_jax = jnp.asarray(X_selected)

            y_pred_jax = funcs['pls_predict'](
                X_jax, self._B_jax, self._X_mean_jax, self._y_mean_jax
            )
            y_pred = np.asarray(y_pred_jax)
        else:
            # NumPy prediction
            y_pred = _ipls_predict_numpy(
                X,
                self.interval_starts_,
                self.interval_ends_,
                self.selected_intervals_,
                self.combination_method,
                self._final_pls,
            )

        # Flatten if single target
        if self._y_1d:
            y_pred = y_pred.ravel()

        return y_pred

    def transform(
        self,
        X: ArrayLike,
    ) -> NDArray[np.floating]:
        """Transform X to selected feature space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Returns
        -------
        X_selected : ndarray of shape (n_samples, n_selected_features)
            Selected features only.
        """
        check_is_fitted(self, ['feature_mask_'])

        X = np.asarray(X, dtype=np.float64)
        return X[:, self.feature_mask_]

    def get_interval_info(self) -> dict:
        """Get detailed information about intervals and selection.

        Returns
        -------
        info : dict
            Dictionary containing:
            - 'n_intervals': Number of intervals
            - 'interval_scores': CV scores for each interval
            - 'interval_ranges': List of (start, end) for each interval
            - 'selected_intervals': Indices of selected intervals
            - 'selected_regions': (start, end) pairs for selected regions
            - 'n_selected_features': Total number of selected features
        """
        check_is_fitted(self, ['interval_scores_', 'selected_intervals_'])

        return {
            'n_intervals': self.n_intervals_,
            'interval_scores': self.interval_scores_.copy(),
            'interval_ranges': [
                (int(self.interval_starts_[i]), int(self.interval_ends_[i]))
                for i in range(self.n_intervals_)
            ],
            'selected_intervals': self.selected_intervals_.copy(),
            'selected_regions': self.selected_regions_.copy(),
            'n_selected_features': int(self.feature_mask_.sum()),
        }

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'n_components': self.n_components,
            'n_intervals': self.n_intervals,
            'interval_width': self.interval_width,
            'cv': self.cv,
            'scoring': self.scoring,
            'mode': self.mode,
            'combination_method': self.combination_method,
            'backend': self.backend,
        }

    def set_params(self, **params) -> "IntervalPLS":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : IntervalPLS
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"IntervalPLS(n_components={self.n_components}, "
            f"n_intervals={self.n_intervals}, mode='{self.mode}', "
            f"backend='{self.backend}')"
        )
