"""FCKResidualRegressor — do-no-harm residual on top of an OOF teacher.

Trains a teacher (e.g. AOM-PLS or Ridge) with 5-fold inner CV on the
training set to obtain out-of-fold predictions ``z_train_oof``, then trains
``FCKStatic + ridge_head`` on the residuals ``y_train - z_train_oof``.
A held-out calibration partition selects a shrinkage coefficient
``s* in {0, 0.25, 0.5, 0.75, 1.0}`` that minimises
``RMSE(y_val, z_val + s * nn_residual_val)``. ``s = 0`` is always in the
grid as the do-no-harm fallback (test prediction collapses to the teacher
alone). The catastrophic-loss diagnostic is computed at
predict-time via ``self.last_diagnostics_`` for the run-level aggregator.

Usage::

    from sklearn.linear_model import Ridge
    from sklearn.cross_decomposition import PLSRegression
    from nirs4all.operators.transforms import FCKStaticTransformer
    from bench.fck_pls.fck_residual import FCKResidualRegressor

    teacher = PLSRegression(n_components=10)
    model = FCKResidualRegressor(teacher=teacher,
                                 fck=FCKStaticTransformer(),
                                 residual_head=Ridge(alpha=1.0))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(model.shrinkage_s_star_)

The class follows the sklearn estimator protocol (clone-friendly, no
state in ``__init__``). Shrinkage CV uses a single held-out partition
deterministic from ``(random_state, val_fraction)`` for compute
parity with R21 nicon_v2 — see
``bench/nicon_v2/docs/B_PLAN_2026-05.md`` §2.2 for the rationale.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


class FCKResidualRegressor(BaseEstimator, RegressorMixin):
    """Teacher + (FCKStatic + ridge head) residual stack with shrinkage CV.

    Parameters
    ----------
    teacher : sklearn-compatible regressor
        The "do-no-harm" baseline. The residual model is trained to predict
        ``y - z_train_oof`` where ``z_train_oof`` is the teacher's 5-fold
        out-of-fold prediction on ``X_train``. The teacher is also
        re-fitted on the full training set at the end of ``fit`` for
        test-time prediction. Cloned via ``sklearn.base.clone`` before
        fitting, so passing a single instance is safe.
    fck : sklearn-compatible transformer
        The FCK feature extractor. Defaults to
        :class:`nirs4all.operators.transforms.FCKStaticTransformer` with
        the standard 16-filter bank.
    residual_head : sklearn-compatible regressor
        Linear (or other) head fit on ``fck.transform(X), residuals``.
        Defaults to :class:`sklearn.linear_model.Ridge` with ``alpha=1.0``.
    shrinkage_grid : sequence of float, default=(0.0, 0.25, 0.5, 0.75, 1.0)
        Candidate shrinkage coefficients. ``0.0`` must be present so the
        do-no-harm fallback exists; this is enforced.
    oof_n_folds : int, default=5
        Number of folds for the teacher's out-of-fold predictions on the
        training set. Increase for finer residuals at the cost of training
        time.
    val_fraction : float, default=0.2
        Fraction of the training set held out for shrinkage selection.
        Disjoint from the test set by construction.
    random_state : int, default=0
        Controls the OOF fold permutation and the held-out shrinkage
        partition. Same value across calls produces identical splits.
    catastrophic_threshold : float, default=0.5
        Threshold on ``(final_train_rmse / teacher_oof_rmse) - 1`` used to
        flag selections that look catastrophic on the calibration split.

    Attributes
    ----------
    teacher_ : fitted teacher (on full train)
    fck_ : fitted FCKStaticTransformer
    residual_head_ : fitted residual_head
    shrinkage_s_star_ : float
        Selected shrinkage coefficient.
    shrinkage_inner_rmse_per_s_ : dict[float, float]
        Calibration-partition RMSE for each candidate ``s``.
    catastrophic_ : bool
        Set to ``True`` when the selected ``s*`` produced a calibration
        RMSE above ``catastrophic_threshold * teacher_only_rmse``.
    last_diagnostics_ : dict
        Convenience aggregate of the above for downstream logging.
    """

    def __init__(
        self,
        teacher,
        fck=None,
        residual_head=None,
        shrinkage_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        oof_n_folds: int = 5,
        val_fraction: float = 0.2,
        random_state: int = 0,
        catastrophic_threshold: float = 0.5,
    ):
        self.teacher = teacher
        self.fck = fck
        self.residual_head = residual_head
        self.shrinkage_grid = shrinkage_grid
        self.oof_n_folds = oof_n_folds
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.catastrophic_threshold = catastrophic_threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_residual_pipeline(self):
        # Lazy default for fck so the dependency is only required when used.
        if self.fck is None:
            from nirs4all.operators.transforms import FCKStaticTransformer
            fck = FCKStaticTransformer()
        else:
            fck = clone(self.fck)
        head = clone(self.residual_head) if self.residual_head is not None else Ridge(alpha=1.0)
        return Pipeline([("fck", fck), ("head", head)])

    def _validate_grid(self) -> tuple[float, ...]:
        grid = tuple(float(s) for s in self.shrinkage_grid)
        if 0.0 not in grid:
            raise ValueError("shrinkage_grid must include 0.0 as the do-no-harm fallback.")
        return grid

    def _oof_teacher_predictions(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if self.oof_n_folds < 2:
            raise ValueError("oof_n_folds must be >= 2.")
        kf = KFold(n_splits=self.oof_n_folds, shuffle=True, random_state=self.random_state)
        z_oof = np.zeros(n, dtype=float)
        for tr_idx, va_idx in kf.split(np.arange(n)):
            teacher_k = clone(self.teacher)
            teacher_k.fit(X[tr_idx], y[tr_idx])
            z_oof[va_idx] = np.asarray(teacher_k.predict(X[va_idx]), dtype=float).ravel()
        return z_oof

    def _calibration_split(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(n)
        n_val = max(1, int(round(self.val_fraction * n)))
        return idx[:n_val]

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        grid = self._validate_grid()

        # 1. OOF teacher predictions on the full training set.
        z_train_oof = self._oof_teacher_predictions(X, y)
        residuals = y - z_train_oof

        # 2. Fit the residual pipeline on the full training set on residuals.
        self.residual_pipeline_ = self._build_residual_pipeline()
        self.residual_pipeline_.fit(X, residuals)

        # 3. Calibration split: pick s* on the held-out partition.
        val_idx = self._calibration_split(X.shape[0])
        nn_val = np.asarray(
            self.residual_pipeline_.predict(X[val_idx]), dtype=float
        ).ravel()
        z_val = z_train_oof[val_idx]
        y_val = y[val_idx]
        teacher_val_rmse = float(np.sqrt(np.mean((y_val - z_val) ** 2)))
        rmse_per_s: dict[float, float] = {}
        for s in grid:
            rmse_s = float(np.sqrt(np.mean((y_val - (z_val + s * nn_val)) ** 2)))
            rmse_per_s[s] = rmse_s
        s_star = min(grid, key=lambda s: rmse_per_s[s])
        final_val_rmse = rmse_per_s[s_star]
        catastrophic = bool(
            teacher_val_rmse > 0
            and (final_val_rmse / teacher_val_rmse - 1.0) > self.catastrophic_threshold
        )

        # 4. Fit the production teacher on the full training set for test-time use.
        self.teacher_ = clone(self.teacher)
        self.teacher_.fit(X, y)

        # 5. Expose attributes.
        self.shrinkage_s_star_ = float(s_star)
        self.shrinkage_inner_rmse_per_s_ = {float(k): float(v) for k, v in rmse_per_s.items()}
        self.teacher_calibration_rmse_ = teacher_val_rmse
        self.calibration_rmse_at_s_star_ = final_val_rmse
        self.catastrophic_ = catastrophic
        self.last_diagnostics_ = {
            "shrinkage_s_star": self.shrinkage_s_star_,
            "shrinkage_inner_rmse_per_s": self.shrinkage_inner_rmse_per_s_,
            "teacher_calibration_rmse": self.teacher_calibration_rmse_,
            "calibration_rmse_at_s_star": self.calibration_rmse_at_s_star_,
            "catastrophic": self.catastrophic_,
        }
        return self

    def predict(self, X):
        if not hasattr(self, "teacher_"):
            raise RuntimeError("FCKResidualRegressor.fit must be called before predict.")
        X = np.asarray(X, dtype=float)
        z_test = np.asarray(self.teacher_.predict(X), dtype=float).ravel()
        nn_test = np.asarray(self.residual_pipeline_.predict(X), dtype=float).ravel()
        return z_test + self.shrinkage_s_star_ * nn_test
