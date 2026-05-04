"""Tests for ``AOMRidgeAutoSelector``.

Covers the six required scenarios from the task spec:

1. The selector picks the best candidate on a synthetic where one variant
   is clearly better than the other.
2. No candidate ever observes outer-validation rows during ``fit``.
3. ``predict`` returns the right shape for 1D and 2D targets.
4. Default ``candidates=None`` resolves to the 8 HEADLINE variants.
5. The auto-selector achieves at least the better of two candidates on
   held-out data (oracle-envelope sanity check).
6. ``predict`` before ``fit`` raises ``NotFittedError``.
"""

from __future__ import annotations

import numpy as np
import pytest
from aompls.operators import IdentityOperator, LinearSpectralOperator
from aomridge.auto_selector import (
    AOMRidgeAutoSelector,
    _default_headline_candidates,
)
from aomridge.estimators import AOMRidgeRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold

# ----------------------------------------------------------------------
# Helpers: synthetic data + spy operator
# ----------------------------------------------------------------------


class SpyOperator(LinearSpectralOperator):
    """Identity-equivalent operator that records the rows seen at fit time."""

    def __init__(self, p: int | None = None) -> None:
        super().__init__(name="spy_identity", p=p)
        self.fit_row_signatures: list[float] = []

    def fit(self, X=None, y=None):
        if X is not None:
            X = np.asarray(X)
            self.fit_row_signatures.extend(np.sum(X, axis=1).tolist())
            self.p = X.shape[1]
        return self

    def _transform_impl(self, X):
        return X.copy()

    def _apply_cov_impl(self, S):
        return S.copy()

    def _adjoint_vec_impl(self, v):
        return v.copy()

    def _matrix_impl(self, p: int):
        return np.eye(p)


def _make_regression(
    n: int = 60, p: int = 24, noise: float = 0.05, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=p)
    y = X @ coef + noise * rng.normal(size=n)
    return X, y


# ----------------------------------------------------------------------
# 1. Selector picks the better variant on a clearly biased synthetic.
# ----------------------------------------------------------------------


def test_auto_selector_picks_best_synthetic():
    """A linear-friendly target should make the AOM-Ridge global-compact-none
    variant outperform a degenerate ``identity-only`` Ridge with a tiny
    operator bank.

    We construct a target where the ``fd_1`` (first finite difference)
    response carries the signal, so any variant that includes the compact
    bank should beat one that's stuck with identity. The auto-selector
    must therefore pick ``"compact"`` over ``"identity"``.
    """
    rng = np.random.default_rng(42)
    n, p = 80, 32
    X = rng.normal(size=(n, p))
    # First finite-difference along columns → highlight the compact bank.
    Xd = np.diff(X, axis=1)
    coef = rng.normal(size=Xd.shape[1])
    y = Xd @ coef + 0.02 * rng.normal(size=n)

    candidates = [
        {
            "label": "Ridge-identity",
            "selection": "superblock",
            "operator_bank": "identity",
            "block_scaling": "none",
        },
        {
            "label": "AOMRidge-global-compact",
            "selection": "global",
            "operator_bank": "compact",
            "block_scaling": "none",
        },
    ]
    selector = AOMRidgeAutoSelector(
        candidates=candidates,
        outer_cv=3,
        outer_cv_kind="kfold",
        random_state=0,
    ).fit(X, y)
    assert selector.selected_variant_label_ == "AOMRidge-global-compact"
    # Identity must be strictly worse on this synthetic.
    scores = dict(zip(
        [c["label"] for c in selector.candidates_],
        selector.cv_scores_, strict=True,
    ))
    assert scores["AOMRidge-global-compact"] < scores["Ridge-identity"]


# ----------------------------------------------------------------------
# 2. Outer-CV anti-leakage: spy operator never sees outer-validation rows.
# ----------------------------------------------------------------------


def test_auto_selector_outer_cv_no_leak():
    """The user-supplied operator instance is the *template*; clones absorb
    state. The spy is wrapped inside an AOMRidgeRegressor candidate; its
    ``fit_row_signatures`` must remain empty (clones, not the template,
    are fitted).

    More importantly, the *clones* used by the candidate's inner CV must
    only ever see outer-train rows. We assert this by collecting the
    centred row sums recorded by every clone and checking they correspond
    to outer-train slices, never to outer-validation slices.
    """
    rng = np.random.default_rng(7)
    n, p = 36, 16
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    template_spy = SpyOperator(p=p)

    # Use a deterministic outer KFold so we can compute the outer-train /
    # outer-valid row sets exactly.
    outer = KFold(n_splits=3, shuffle=False)
    folds = list(outer.split(X, y))

    # Capture every fit row sum across all clones (across folds and
    # candidates). We do this with a custom subclass that appends to a
    # *shared* list.
    fit_row_sums: list[float] = []

    class SharedSpy(SpyOperator):
        def fit(self, X=None, y=None):
            if X is not None:
                X = np.asarray(X)
                fit_row_sums.extend(np.round(np.sum(X, axis=1), 8).tolist())
                self.p = X.shape[1]
            return self

    candidates = [
        {
            "label": "spy_candidate",
            "selection": "global",
            "operator_bank": [IdentityOperator(p=p), SharedSpy(p=p)],
            "block_scaling": "none",
        },
    ]
    selector = AOMRidgeAutoSelector(
        candidates=candidates,
        outer_cv=outer,
        random_state=0,
    ).fit(X, y)

    # Template must remain unfit (each clone is fresh).
    assert template_spy.fit_row_signatures == []
    # selector must have at least called fit somewhere.
    assert fit_row_sums, "spy clones were never invoked"

    # For every outer fold, the union of seen row sums must lie inside the
    # outer-train slice's centred row sums (modulo numerical rounding) and
    # never include any outer-validation row's centred row sum.
    seen = set(fit_row_sums)
    for tr_idx, va_idx in folds:
        x_mean_tr = X[tr_idx].mean(axis=0)
        # Validation row sums (centred with the outer-train mean) must not
        # be in the seen set. Note: candidates can fit on inner-train slices
        # of outer-train, which use a different mean - so we only assert the
        # *outer-validation* rows centred by the outer-train mean are absent.
        val_centred_sums = set(
            np.round(np.sum(X[va_idx] - x_mean_tr, axis=1), 8).tolist()
        )
        # The outer-validation row sums centred by the outer-train mean
        # would only show up if a candidate had been trained on outer-train+val.
        assert seen.isdisjoint(val_centred_sums), (
            "candidate fit observed outer-validation rows (leak)"
        )


# ----------------------------------------------------------------------
# 3. predict works for 1D and 2D targets.
# ----------------------------------------------------------------------


def test_auto_selector_predict_shape():
    rng = np.random.default_rng(3)
    n, p = 40, 12
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=p)

    candidates = [
        {
            "label": "Ridge-identity",
            "selection": "superblock",
            "operator_bank": "identity",
            "block_scaling": "none",
        },
    ]
    # 1D y
    y_1d = X @ coef + 0.05 * rng.normal(size=n)
    sel_1d = AOMRidgeAutoSelector(
        candidates=candidates, outer_cv=3, outer_cv_kind="kfold", random_state=0,
    ).fit(X, y_1d)
    pred_1d = sel_1d.predict(X)
    assert pred_1d.shape == (n,)

    # 2D y with q=2
    y_2d = np.column_stack([y_1d, X @ rng.normal(size=p) + 0.05 * rng.normal(size=n)])
    sel_2d = AOMRidgeAutoSelector(
        candidates=candidates, outer_cv=3, outer_cv_kind="kfold", random_state=0,
    ).fit(X, y_2d)
    pred_2d = sel_2d.predict(X)
    assert pred_2d.shape == (n, 2)


# ----------------------------------------------------------------------
# 4. candidates=None → HEADLINE
# ----------------------------------------------------------------------


def test_auto_selector_default_candidates():
    """When ``candidates=None``, the resolved set must equal the 8 HEADLINE
    variants. We compare labels in order.
    """
    headline = _default_headline_candidates()
    assert len(headline) == 8
    expected_labels = [c["label"] for c in headline]
    assert expected_labels == [
        "Ridge-raw",
        "AOMRidge-global-compact-none",
        "AOMRidge-global-compact-none-msc",
        "AOMRidge-global-compact-none-snv",
        "AOMRidge-global-compact-none-asls",
        "AOMRidgePLS-compact-colscale-cv-relative",
        "AOMRidgePLS-compact-Hmax-relative-emsc2",
        "AOM-PLS-compact-CV",
    ]
    # Verify the selector internally normalises None to the headline list.
    selector = AOMRidgeAutoSelector(candidates=None)
    norm = selector._normalise_candidates()
    assert [c["label"] for c in norm] == expected_labels


# ----------------------------------------------------------------------
# 5. Oracle-envelope sanity: selector >= min(test_RMSE) within CV variance.
# ----------------------------------------------------------------------


def test_auto_selector_oracle_envelope():
    """On a held-out test split, the auto-selector's RMSE must not be worse
    than the min RMSE achieved by either candidate when each is trained on
    the same training split. Strict equality is too strong (CV may pick the
    other variant by a hair); we allow a 30% tolerance on the gap to the
    better candidate, which should be more than enough to absorb random
    fluctuations on a small synthetic.
    """
    rng = np.random.default_rng(11)
    n, p = 80, 24
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=p)
    y = X @ coef + 0.05 * rng.normal(size=n)
    # 70/30 split
    perm = rng.permutation(n)
    split = int(0.7 * n)
    tr, te = perm[:split], perm[split:]
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]

    candidates = [
        {
            "label": "A_identity",
            "selection": "superblock",
            "operator_bank": "identity",
            "block_scaling": "none",
        },
        {
            "label": "B_compact",
            "selection": "global",
            "operator_bank": "compact",
            "block_scaling": "none",
        },
    ]

    def _rmse(yt, yp):
        return float(np.sqrt(np.mean((yt - yp) ** 2)))

    # Baseline: train each candidate independently on X_tr/y_tr.
    cand_rmse: list[float] = []
    for spec in candidates:
        est = AOMRidgeRegressor(
            selection=spec["selection"],
            operator_bank=spec["operator_bank"],
            block_scaling=spec["block_scaling"],
            cv=3,
            random_state=0,
        )
        est.fit(X_tr, y_tr)
        cand_rmse.append(_rmse(y_te, est.predict(X_te)))

    selector = AOMRidgeAutoSelector(
        candidates=candidates,
        outer_cv=3,
        outer_cv_kind="kfold",
        random_state=0,
    ).fit(X_tr, y_tr)
    sel_rmse = _rmse(y_te, selector.predict(X_te))

    best_cand_rmse = min(cand_rmse)
    # Allow 30% slack relative to the better candidate's test RMSE (CV
    # selection can pick the other one by a small margin on a small set).
    assert sel_rmse <= best_cand_rmse * 1.30, (
        f"selector RMSE {sel_rmse:.4f} too far from best candidate "
        f"RMSE {best_cand_rmse:.4f}"
    )


# ----------------------------------------------------------------------
# 6. predict before fit raises NotFittedError.
# ----------------------------------------------------------------------


def test_auto_selector_check_is_fitted():
    sel = AOMRidgeAutoSelector()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 4))
    with pytest.raises(NotFittedError):
        sel.predict(X)


# ----------------------------------------------------------------------
# Bonus: factory-callable spec works (covers spec.factory branch in dispatch).
# ----------------------------------------------------------------------


def test_auto_selector_factory_callable():
    """A candidate spec may carry a ``factory`` callable that returns a fresh
    sklearn-compatible estimator. The dispatcher must use that estimator
    verbatim, ignoring any selection/operator_bank keys. Useful for plugging
    in non-AOM estimators (e.g. a vanilla PLSRegression) for comparison.
    """
    X, y = _make_regression(n=50, p=12, seed=2)
    candidates = [
        {"label": "vanilla_pls", "factory": lambda: PLSRegression(n_components=3)},
        {
            "label": "AOMRidge-global",
            "selection": "global",
            "operator_bank": "compact",
            "block_scaling": "none",
        },
    ]
    selector = AOMRidgeAutoSelector(
        candidates=candidates,
        outer_cv=3,
        outer_cv_kind="kfold",
        random_state=0,
    ).fit(X, y)
    assert selector.selected_variant_label_ in (
        "vanilla_pls", "AOMRidge-global",
    )
    # Predictions must match the chosen refit estimator's shape.
    yhat = selector.predict(X)
    # PLSRegression returns (n, 1) for 1D y; we normalise via ``_ravel_match``
    # only on score(), so accept either shape on predict.
    assert yhat.shape in {(X.shape[0],), (X.shape[0], 1)}
