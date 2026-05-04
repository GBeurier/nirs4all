"""Outer-CV variant selector for AOM-Ridge candidate estimators.

The selector takes a list of candidate variant specs, runs an *outer* K-fold
cross-validation for every candidate, and refits the best-scoring candidate
on the full training set. The headline use case is to close the gap between
the best fixed AOM-Ridge variant (~60% wins) and the per-dataset oracle
(~84% wins) on the curated 38-dataset cohort.

Anti-leakage invariant
----------------------

Every outer fold builds a *fresh* estimator from the candidate spec, fits
it on the outer-train slice ONLY, and scores predictions on the
outer-validation slice. Any *inner* CV used by a candidate (e.g. the
AOMRidgeRegressor alpha grid search, or the AOMRidgePLSCV ``(H, alpha)``
grid) sees only the outer-train rows because the inner splitter is
materialised against ``X[outer_train_idx]``.

The final ``predict`` delegates to ``self.refit_estimator_``, an
estimator built from the chosen spec and refit on the FULL training set
so it can apply branch preprocessing or any other state stored at fit
time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

from .branches import fit_transform_branch, make_branch_preproc

VariantSpec = dict[str, Any]


# ----------------------------------------------------------------------
# Outer CV resolution
# ----------------------------------------------------------------------


def _resolve_outer_cv(
    cv: int | object,
    kind: str,
    repeats: int,
    random_state: int | None,
) -> object:
    """Return an sklearn-compatible outer splitter.

    ``cv`` may be an integer (number of splits) or any object exposing
    ``split(X, y)``. When integer, ``kind`` decides:

    - ``"spxy"``: ``SPXYFold`` from nirs4all (single repeat) when
      available, else ``KFold(shuffle=True)``;
    - ``"kfold"``: shuffled ``KFold``;
    - ``"spxy_repeated"``: ``RepeatedSPXYFold`` with ``repeats`` repeats.

    Pre-built splitter objects are returned unchanged.
    """
    if hasattr(cv, "split"):
        return cv
    if not isinstance(cv, int):
        raise TypeError("outer_cv must be an integer or an sklearn-compatible splitter")
    if cv < 2:
        raise ValueError("outer_cv must be >= 2 when integer")
    if kind == "kfold":
        return KFold(n_splits=cv, shuffle=True, random_state=random_state)
    if kind == "spxy_repeated":
        from .cv import RepeatedSPXYFold
        return RepeatedSPXYFold(
            n_splits=cv, n_repeats=max(1, int(repeats)), random_state=random_state,
        )
    if kind == "spxy":
        try:
            from nirs4all.operators.splitters import SPXYFold
            return SPXYFold(n_splits=cv, random_state=random_state)
        except Exception:
            return KFold(n_splits=cv, shuffle=True, random_state=random_state)
    raise ValueError(
        f"unknown outer_cv_kind {kind!r}; expected 'spxy', 'kfold', or 'spxy_repeated'"
    )


# ----------------------------------------------------------------------
# Candidate dispatch
# ----------------------------------------------------------------------


def _dispatch_candidate(
    spec: VariantSpec, seed: int, inner_cv: int | object,
) -> tuple[BaseEstimator, str | None]:
    """Build a fresh estimator from a spec dict.

    Returns ``(estimator, branch_preproc_name)``. The branch preprocessor
    is applied externally by the caller (``fit_transform`` on training rows,
    ``transform`` on validation rows) so the candidate estimator never sees
    raw rows that would leak between folds.

    Recognised dispatch keys (mirrors the benchmark runner conventions):

    - ``selection == "ridge_pls"``: build :class:`AOMRidgePLS` (fixed
      ``n_components`` / ``ridge_alpha``) or :class:`AOMRidgePLSCV` (with
      grid keys ``n_components_grid`` / ``ridge_alpha_grid``).
    - ``selection == "aom_pls"``: build :class:`aompls.AOMPLSRegressor`.
    - any other ``selection`` value: build :class:`AOMRidgeRegressor`.

    The ``factory`` key takes precedence: when present and callable, it is
    invoked with no arguments to produce the estimator and the rest of the
    spec is ignored (except ``branch_preproc`` and ``label``).
    """
    factory = spec.get("factory")
    branch = spec.get("branch_preproc")
    if callable(factory):
        return factory(), branch

    extra = dict(spec.get("extra", {}))
    selection = spec.get("selection", "superblock")
    operator_bank = spec.get("operator_bank", "compact")
    block_scaling = spec.get("block_scaling", "rms")

    if selection == "ridge_pls":
        from .aom_ridge_pls import AOMRidgePLS, AOMRidgePLSCV

        base_kwargs = {
            "operator_bank": operator_bank,
            "block_scaling": block_scaling,
            "random_state": seed,
        }
        cv_kind = extra.pop("cv_kind", None)
        cv_splits = int(extra.pop("cv_splits", 0)) or 0
        cv_repeats = int(extra.pop("cv_repeats", 1)) or 1
        if "n_components_grid" in extra or "ridge_alpha_grid" in extra:
            if cv_kind == "spxy_repeated":
                if cv_splits < 2:
                    raise ValueError(
                        "cv_splits must be >= 2 when cv_kind='spxy_repeated'"
                    )
                from .cv import RepeatedSPXYFold
                cv_for_inner: int | object = RepeatedSPXYFold(
                    n_splits=cv_splits, n_repeats=cv_repeats, random_state=seed,
                )
            elif cv_splits > 0:
                cv_for_inner = cv_splits
            else:
                cv_for_inner = inner_cv
            est: BaseEstimator = AOMRidgePLSCV(
                **base_kwargs,
                n_components_grid=tuple(
                    extra.pop("n_components_grid", (2, 5, 10, 15, 20))
                ),
                ridge_alpha_grid=extra.pop(
                    "ridge_alpha_grid", np.logspace(-4.0, 4.0, 9).tolist(),
                ),
                cv=cv_for_inner,
                **extra,
            )
        else:
            est = AOMRidgePLS(**base_kwargs, cv=inner_cv, **extra)
        return est, branch

    if selection == "aom_pls":
        from aompls.estimators import AOMPLSRegressor

        max_components = int(extra.pop("max_components", 30))
        cv_inner = int(extra.pop("cv", 3))
        est = AOMPLSRegressor(
            n_components="auto",
            max_components=max_components,
            operator_bank=operator_bank,
            cv=cv_inner,
            random_state=seed,
            **extra,
        )
        return est, branch

    from .estimators import AOMRidgeRegressor

    kwargs = {
        "selection": selection,
        "operator_bank": operator_bank,
        "block_scaling": block_scaling,
        "cv": inner_cv,
        "random_state": seed,
    }
    kwargs.update(extra)
    return AOMRidgeRegressor(**kwargs), branch


def _apply_branch(
    branch: str | None,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_other: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit branch on train rows and apply to both train and an optional view.

    Returns ``(X_train_out, X_other_out)``. When ``branch`` is falsy or
    resolves to ``"none"`` the inputs are returned unchanged (cast to
    ``float``).
    """
    if not branch:
        out_tr = np.asarray(X_train, dtype=float)
        out_ot = None if X_other is None else np.asarray(X_other, dtype=float)
        return out_tr, out_ot
    preproc = make_branch_preproc(branch)
    if preproc is None:
        out_tr = np.asarray(X_train, dtype=float)
        out_ot = None if X_other is None else np.asarray(X_other, dtype=float)
        return out_tr, out_ot
    Xtr_proc = fit_transform_branch(
        preproc, np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=float),
    )
    if X_other is None:
        return Xtr_proc, None
    Xot_proc = np.asarray(
        preproc.transform(np.asarray(X_other, dtype=float)), dtype=float,
    )
    return Xtr_proc, Xot_proc


# ----------------------------------------------------------------------
# Default candidate set: the 8 HEADLINE_VARIANTS from the bench
# ----------------------------------------------------------------------


def _default_headline_candidates() -> list[VariantSpec]:
    """Return spec dicts for the 8 HEADLINE variants.

    Imported lazily so that the auto_selector module stays decoupled from the
    benchmark runner script (which lives in ``benchmarks/`` and is not part
    of the importable package).
    """
    return [
        {
            "label": "Ridge-raw",
            "selection": "superblock",
            "operator_bank": "identity",
            "block_scaling": "none",
        },
        {
            "label": "AOMRidge-global-compact-none",
            "selection": "global",
            "operator_bank": "compact",
            "block_scaling": "none",
        },
        {
            "label": "AOMRidge-global-compact-none-msc",
            "selection": "global",
            "operator_bank": "compact",
            "block_scaling": "none",
            "branch_preproc": "msc",
        },
        {
            "label": "AOMRidge-global-compact-none-snv",
            "selection": "global",
            "operator_bank": "compact",
            "block_scaling": "none",
            "branch_preproc": "snv",
        },
        {
            "label": "AOMRidge-global-compact-none-asls",
            "selection": "global",
            "operator_bank": "compact",
            "block_scaling": "none",
            "branch_preproc": "asls",
        },
        {
            "label": "AOMRidgePLS-compact-colscale-cv-relative",
            "selection": "ridge_pls",
            "operator_bank": "compact",
            "block_scaling": "frobenius",
            "extra": {
                "n_components_grid": (2, 3, 5, 7, 10, 15, 20, 30),
                "ridge_alpha_grid": np.logspace(-4.0, 4.0, 25).tolist(),
                "cv_kind": "spxy_repeated",
                "cv_splits": 3,
                "cv_repeats": 3,
                "ridge_alpha_mode": "relative_to_score_variance",
                "selection_rule": "1se",
                "scoring": "rmse_mean",
                "column_scaling": True,
            },
        },
        {
            "label": "AOMRidgePLS-compact-Hmax-relative-emsc2",
            "selection": "ridge_pls",
            "operator_bank": "compact",
            "block_scaling": "frobenius",
            "branch_preproc": "emsc2",
            "extra": {
                "n_components_grid": (2, 3, 5, 7, 10, 15, 20, 30),
                "ridge_alpha_grid": np.logspace(-4.0, 4.0, 25).tolist(),
                "cv_kind": "spxy_repeated",
                "cv_splits": 3,
                "cv_repeats": 3,
                "ridge_alpha_mode": "relative_to_score_variance",
                "selection_rule": "1se",
                "scoring": "rmse_mean",
            },
        },
        {
            "label": "AOM-PLS-compact-CV",
            "selection": "aom_pls",
            "operator_bank": "compact",
            "block_scaling": "none",
            "extra": {"max_components": 30, "cv": 3},
        },
    ]


# ----------------------------------------------------------------------
# Outer-CV scoring helpers (top-level so joblib can pickle them)
# ----------------------------------------------------------------------


def _ravel_match(arr: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Reshape ``arr`` so its trailing dims match ``ref``.

    Handles candidates that emit 1D predictions when the target is 2D
    single-column, and vice versa. The actual squared-error computation is
    invariant under this reshape but normalising shapes lets us use
    a single code path.
    """
    a = np.asarray(arr, dtype=float)
    r = np.asarray(ref, dtype=float)
    if a.shape == r.shape:
        return a
    if a.ndim == 1 and r.ndim == 2 and r.shape[1] == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2 and a.shape[1] == 1 and r.ndim == 1:
        return a.ravel()
    return a.reshape(r.shape)


def _score_candidate(
    spec: VariantSpec,
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    scoring: str,
) -> tuple[float, list[float]]:
    """Compute the outer-CV score for one candidate spec.

    Returns ``(summary_score, per_fold_rmse)`` where ``summary_score`` is
    either the mean of fold RMSEs (``scoring="rmse_mean"``) or the pooled
    RMSE across folds (``scoring="mse_pooled"``).
    """
    per_fold: list[float] = []
    pooled_sse = 0.0
    pooled_n = 0
    for tr_idx, va_idx in folds:
        X_tr_raw, X_va_raw = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        # Inner CV passed into candidates: a small KFold seeded by the
        # outer seed. The candidate's own spec may override (e.g. via
        # cv_kind="spxy_repeated"); we never let a candidate look at the
        # outer-validation slice because we only pass it ``X[tr_idx]``.
        est, branch = _dispatch_candidate(spec, seed=seed, inner_cv=3)
        X_tr, X_va = _apply_branch(branch, X_tr_raw, y_tr, X_va_raw)
        est.fit(X_tr, y_tr)
        y_pred = est.predict(X_va)
        y_pred = _ravel_match(y_pred, y_va)
        diff = (y_pred - y_va).ravel()
        sse = float(np.sum(diff * diff))
        n = int(diff.size)
        per_fold.append(float(np.sqrt(sse / max(1, n))))
        pooled_sse += sse
        pooled_n += n
    if scoring == "rmse_mean":
        summary = float(np.mean(per_fold)) if per_fold else float("inf")
    elif scoring == "mse_pooled":
        summary = float(np.sqrt(pooled_sse / max(1, pooled_n)))
    else:
        raise ValueError("scoring must be 'rmse_mean' or 'mse_pooled'")
    return summary, per_fold


# ----------------------------------------------------------------------
# Estimator
# ----------------------------------------------------------------------


class AOMRidgeAutoSelector(BaseEstimator, RegressorMixin):
    """Outer-CV variant selector over a list of AOM-Ridge candidates.

    For each candidate variant in ``candidates``, run K-fold outer CV
    (default ``SPXYFold(3)``) and compute the mean RMSE across folds. The
    candidate with the lowest mean RMSE is selected and refit on the full
    training set. ``predict()`` delegates to the refit estimator.

    Parameters
    ----------
    candidates : sequence of variant specs (dicts) or callables, or ``None``
        Each spec is a dict with at minimum ``label`` and either:

        - ``factory``: a callable returning a fresh sklearn-compatible
          estimator (the rest of the spec is ignored), or
        - ``selection`` / ``operator_bank`` / ``block_scaling`` /
          ``branch_preproc`` / ``extra``: dispatched to
          :class:`AOMRidgeRegressor`, :class:`AOMRidgePLS`,
          :class:`AOMRidgePLSCV`, or :class:`AOMPLSRegressor`.

        When ``None`` (default), uses the 8 HEADLINE variants.
    outer_cv : int or splitter
        K-fold outer CV. Integer is interpreted by ``outer_cv_kind``.
    outer_cv_kind : {"spxy", "kfold", "spxy_repeated"}
        Splitter style when ``outer_cv`` is integer.
    outer_cv_repeats : int
        Repeats for ``outer_cv_kind="spxy_repeated"`` (ignored otherwise).
    scoring : {"rmse_mean", "mse_pooled"}
        Aggregation across outer folds.
    random_state : int
        Seed forwarded to splitters and inner CVs.
    n_jobs : int
        joblib parallelism over candidates. ``1`` runs serially, ``-1``
        uses all cores.

    Attributes
    ----------
    selected_variant_label_ : str
        Label of the chosen candidate.
    selected_variant_index_ : int
        Index of the chosen candidate inside ``self.candidates_``.
    cv_scores_ : list of float
        Mean RMSE per candidate across outer folds.
    cv_per_fold_scores_ : list of list of float
        Per-fold RMSE matrix ``[n_candidates][n_folds]``.
    candidates_ : list of dict
        Resolved candidate specs (with at least ``label``).
    refit_estimator_ : sklearn-compatible estimator
        The chosen candidate refit on the full training set. ``predict``
        and ``score`` delegate to it.
    refit_branch_preproc_ : transformer or None
        Branch preprocessor fitted on the full training set, applied to
        new rows before delegating to ``refit_estimator_``.
    """

    def __init__(
        self,
        candidates: Sequence[VariantSpec | Callable[[], BaseEstimator]] | None = None,
        outer_cv: int | object = 3,
        outer_cv_kind: str = "spxy",
        outer_cv_repeats: int = 1,
        scoring: str = "rmse_mean",
        random_state: int = 0,
        n_jobs: int = 1,
    ) -> None:
        self.candidates = candidates
        self.outer_cv = outer_cv
        self.outer_cv_kind = outer_cv_kind
        self.outer_cv_repeats = outer_cv_repeats
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------
    # Spec normalisation
    # ------------------------------------------------------------------

    def _normalise_candidates(self) -> list[VariantSpec]:
        if self.candidates is None:
            return _default_headline_candidates()
        out: list[VariantSpec] = []
        for i, c in enumerate(self.candidates):
            if callable(c):
                spec = {"label": f"candidate_{i}", "factory": c}
            elif isinstance(c, dict):
                spec = dict(c)
                spec.setdefault("label", f"candidate_{i}")
            else:
                raise TypeError(
                    f"candidate {i} must be a dict spec or a callable factory; "
                    f"got {type(c).__name__}"
                )
            out.append(spec)
        if not out:
            raise ValueError("candidates must be non-empty")
        return out

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> AOMRidgeAutoSelector:
        if self.scoring not in ("rmse_mean", "mse_pooled"):
            raise ValueError("scoring must be 'rmse_mean' or 'mse_pooled'")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        y_arr = np.asarray(y, dtype=float)
        if y_arr.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        candidates = self._normalise_candidates()
        cv_obj = _resolve_outer_cv(
            self.outer_cv,
            kind=self.outer_cv_kind,
            repeats=self.outer_cv_repeats,
            random_state=self.random_state,
        )
        folds = list(cv_obj.split(X, y_arr))
        if not folds:
            raise ValueError("outer CV produced no folds")

        scoring = self.scoring
        seed = int(self.random_state)

        if int(self.n_jobs) == 1:
            results = [
                _score_candidate(spec, X, y_arr, folds, seed, scoring)
                for spec in candidates
            ]
        else:
            results = Parallel(n_jobs=int(self.n_jobs), backend="loky")(
                delayed(_score_candidate)(spec, X, y_arr, folds, seed, scoring)
                for spec in candidates
            )

        cv_scores = [float(r[0]) for r in results]
        per_fold_scores = [list(r[1]) for r in results]

        best_idx = int(np.argmin(cv_scores))
        best_spec = candidates[best_idx]

        # Refit the winning candidate on the full training set with a fresh
        # estimator. Branch preprocessing is fitted on the full training rows.
        refit_est, branch = _dispatch_candidate(best_spec, seed=seed, inner_cv=3)
        refit_branch_preproc = None
        X_refit = X
        if branch:
            refit_branch_preproc = make_branch_preproc(branch)
            if refit_branch_preproc is not None:
                X_refit = fit_transform_branch(
                    refit_branch_preproc, np.asarray(X, dtype=float),
                    np.asarray(y_arr, dtype=float),
                )
        refit_est.fit(X_refit, y_arr)

        self.candidates_ = candidates
        self.cv_scores_ = cv_scores
        self.cv_per_fold_scores_ = per_fold_scores
        self.selected_variant_index_ = best_idx
        self.selected_variant_label_ = str(best_spec.get("label", f"candidate_{best_idx}"))
        self.refit_estimator_ = refit_est
        self.refit_branch_preproc_ = refit_branch_preproc
        self.n_features_in_ = int(X.shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "refit_estimator_")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if self.refit_branch_preproc_ is not None:
            X = np.asarray(self.refit_branch_preproc_.transform(X), dtype=float)
        return self.refit_estimator_.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        check_is_fitted(self, "refit_estimator_")
        from sklearn.metrics import r2_score

        y_pred = self.predict(X)
        y_arr = np.asarray(y, dtype=float)
        y_pred = _ravel_match(y_pred, y_arr)
        return float(r2_score(y_arr, y_pred, multioutput="uniform_average"))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> dict:
        """Return a JSON-serialisable summary of the selection."""
        check_is_fitted(self, "refit_estimator_")
        diag: dict[str, Any] = {
            "model": "AOMRidgeAutoSelector",
            "selected_variant_label": self.selected_variant_label_,
            "selected_variant_index": int(self.selected_variant_index_),
            "scoring": self.scoring,
            "outer_cv_kind": self.outer_cv_kind,
            "outer_cv_repeats": int(self.outer_cv_repeats),
            "candidate_labels": [
                str(c.get("label", f"candidate_{i}"))
                for i, c in enumerate(self.candidates_)
            ],
            "cv_scores": [float(s) for s in self.cv_scores_],
            "cv_per_fold_scores": [
                [float(x) for x in row] for row in self.cv_per_fold_scores_
            ],
        }
        # Surface the refit estimator's diagnostics when available.
        get_diag = getattr(self.refit_estimator_, "get_diagnostics", None)
        if callable(get_diag):
            diag["refit_diagnostics"] = get_diag()
        return diag


# Re-export sklearn's clone for callers needing a fresh estimator from a spec
__all__ = ["AOMRidgeAutoSelector", "clone"]
