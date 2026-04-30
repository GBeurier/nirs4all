"""Phase 5 / H12 stacking — V1c-best + PLS via Ridge meta-learner.

Out-of-fold (OOF) predictions from each base learner are stacked into a small
feature matrix; a Ridge meta-learner is fitted on those features (with the
ground-truth `y_train` as target). At inference time, all base learners are
refit on the **full** training set and the meta-learner is applied to their
test-time predictions concatenated as features.

This mirrors the AOM-PLS / AOM-Ridge stacking workflow used in
``bench/AOM_v0/aompls/`` and is the canonical sklearn-style stacked ensemble
recipe (Wolpert 1992; Mehmood 2024).

Base learners exposed:

* ``ridge``  — :class:`nicon_v2.models.baseline.RidgeBaseline`
* ``pls``    — :class:`nicon_v2.models.baseline.PLSBaseline`
* ``v1c``    — V1c-concat-bjerrum (Phase 1c best CNN)
* ``v1a_head`` — V1a-head-only (Phase 1a accepted)

The stacker accepts an arbitrary list of base learners; the smoke variant
combines `ridge + pls + v1c`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from ..training import (
    StandardXProcessor,
    StandardYProcessor,
    TrainConfig,
    pick_device,
    predict_torch_regressor,
    set_global_seed,
    train_torch_regressor,
)
from .baseline import PLSBaseline, RidgeBaseline
from .searched_baseline import SearchedPLS, SearchedRidge


@dataclass
class StackingConfig:
    base_learners: tuple[str, ...] = ("ridge", "pls", "v1c")
    # Codex round 3 finding #3: extend α grid to 1e4 (Rice was selecting upper bound 100).
    meta_alphas: tuple[float, ...] = field(default_factory=lambda: tuple(10.0 ** k for k in np.linspace(-3.0, 4.0, 15)))
    n_folds: int = 5
    seed: int = 0
    cnn_v1c_kwargs: dict = field(default_factory=lambda: {
        "norm": "layer", "use_concat_derivatives": True,
    })
    cnn_train_epochs: int = 200
    cnn_train_patience: int = 20
    cnn_use_bjerrum: bool = True
    # Codex round 3 finding #1: optional SPXY-aware splitter for NIR data.
    splitter_kind: str = "kfold"  # {"kfold", "spxy"}


def _make_oof_splitter(kind: str, n_splits: int, seed: int):
    """Construct the outer / inner CV splitter (Codex round 3 #1).

    `"kfold"` → sklearn's shuffled KFold.
    `"spxy"`  → ``nirs4all.operators.splitters.SPXYFold`` (NIR-aware).
    """
    if kind == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if kind == "spxy":
        try:
            from nirs4all.operators.splitters import SPXYFold
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "splitter_kind='spxy' requires nirs4all to be importable"
            ) from exc
        return SPXYFold(n_splits=n_splits, random_state=seed)
    raise ValueError(f"unknown splitter_kind: {kind!r}")


def _build_base_estimator(name: str, seed: int, cfg: StackingConfig):
    if name == "ridge":
        return _SklearnEstimatorAdapter(RidgeBaseline(seed=seed))
    if name == "pls":
        return _SklearnEstimatorAdapter(PLSBaseline(seed=seed))
    if name == "v1c":
        return _CNNV1cAdapter(seed=seed, cfg=cfg)
    if name == "v1a_head":
        return _CNNV1aHeadAdapter(seed=seed, cfg=cfg)
    if name == "aom_ridge":
        return _AOMRidgeAdapter(seed=seed, cfg=cfg)
    if name == "searched_ridge":
        return _SklearnEstimatorAdapter(SearchedRidge(seed=seed))
    if name == "searched_pls":
        return _SklearnEstimatorAdapter(SearchedPLS(seed=seed))
    raise ValueError(f"unknown base learner: {name!r}")


class _AOMRidgeAdapter:
    """Adapter around ``AOMRidgeRegressor`` (read-only import from `bench/AOM_v0/Ridge/aomridge`).

    Uses the ``superblock`` selection on the ``compact`` operator bank with ``rms``
    block scaling — the highest-leverage variant per `bench/AOM_v0/Ridge/docs/IMPLEMENTATION_LOG.md`.
    """

    def __init__(self, seed: int, cfg: StackingConfig):
        self.seed = seed
        self.cfg = cfg
        self._estimator = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_AOMRidgeAdapter":
        from aomridge.estimators import AOMRidgeRegressor

        self._estimator = AOMRidgeRegressor(
            selection="superblock",
            operator_bank="compact",
            block_scaling="rms",
            alphas="auto",
            cv=3,
            random_state=self.seed,
        ).fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError("AOMRidge adapter not fitted")
        return self._estimator.predict(X).ravel()


class _SklearnEstimatorAdapter:
    """Uniform fit/predict wrapper around a RidgeBaseline / PLSBaseline."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SklearnEstimatorAdapter":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)


class _CNNV1cAdapter:
    """V1c CNN (concat-deriv + Bjerrum) wrapped as a fit/predict estimator."""

    def __init__(self, seed: int, cfg: StackingConfig):
        self.seed = seed
        self.cfg = cfg
        self._x_proc: StandardXProcessor | None = None
        self._y_proc: StandardYProcessor | None = None
        self._model = None
        self._device = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_CNNV1cAdapter":
        from ..augmentation import AugmentationPlan, BjerrumConfig, CMixupConfig
        from .v1c_gap_backbone import build_nicon_v1c

        set_global_seed(self.seed)
        device = pick_device("auto")
        self._device = device

        x_proc = StandardXProcessor().fit(X)
        y_proc = StandardYProcessor().fit(y)
        Xs = x_proc.transform(X)
        ys = y_proc.transform(y)

        n_features = Xs.shape[1]
        model = build_nicon_v1c((1, n_features), params=self.cfg.cnn_v1c_kwargs).to(device)

        config = TrainConfig(
            seed=self.seed,
            device=device.type,
            batch_size=min(32, max(8, Xs.shape[0] // 8)),
            epochs=self.cfg.cnn_train_epochs,
            patience=self.cfg.cnn_train_patience,
        )
        if self.cfg.cnn_use_bjerrum:
            plan = AugmentationPlan(bjerrum=BjerrumConfig(enabled=True), cmixup=CMixupConfig(enabled=False))
            bjer, cmix, sigma_y = plan.build(Xs, ys, seq_len=n_features, device=device)
            config.augmenter = bjer

        model, _ = train_torch_regressor(model, Xs, ys, config)
        self._x_proc = x_proc
        self._y_proc = y_proc
        self._model = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or self._x_proc is None or self._y_proc is None:
            raise RuntimeError("V1c adapter not fitted")
        Xs = self._x_proc.transform(X)
        pred_scaled = predict_torch_regressor(self._model, Xs, device=self._device)
        return self._y_proc.inverse_transform(pred_scaled)


class _CNNV1aHeadAdapter(_CNNV1cAdapter):
    """V1a-head-only adapter (no concat-deriv, no Bjerrum)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_CNNV1aHeadAdapter":
        from .v1a_minimal_repair import build_nicon_v1a_head_only

        set_global_seed(self.seed)
        device = pick_device("auto")
        self._device = device

        x_proc = StandardXProcessor().fit(X)
        y_proc = StandardYProcessor().fit(y)
        Xs = x_proc.transform(X)
        ys = y_proc.transform(y)

        n_features = Xs.shape[1]
        model = build_nicon_v1a_head_only((1, n_features)).to(device)
        config = TrainConfig(
            seed=self.seed,
            device=device.type,
            batch_size=min(32, max(8, Xs.shape[0] // 8)),
            epochs=self.cfg.cnn_train_epochs,
            patience=self.cfg.cnn_train_patience,
        )
        model, _ = train_torch_regressor(model, Xs, ys, config)
        self._x_proc = x_proc
        self._y_proc = y_proc
        self._model = model
        return self


@dataclass
class StackedRegressor:
    """OOF-stacked ensemble. Trains base learners K-fold on `(X_train, y_train)` to produce
    OOF predictions, fits a Ridge meta on those OOFs, then refits each base on the full
    train and applies the meta at predict time.
    """

    cfg: StackingConfig
    selected_meta_alpha_: float = float("nan")
    base_models_: list[Any] = field(default_factory=list)
    meta_: Ridge | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackedRegressor":
        n = X.shape[0]
        m = len(self.cfg.base_learners)
        oof_preds = np.zeros((n, m), dtype=float)

        outer_splitter = _make_oof_splitter(self.cfg.splitter_kind, self.cfg.n_folds, self.cfg.seed)
        for fold_idx, (tr, va) in enumerate(outer_splitter.split(X, y)):
            for j, learner_name in enumerate(self.cfg.base_learners):
                est = _build_base_estimator(learner_name, seed=self.cfg.seed + fold_idx, cfg=self.cfg)
                est.fit(X[tr], y[tr])
                oof_preds[va, j] = est.predict(X[va])

        # Pick meta α via inner CV on the OOF feature matrix (Codex round 3 #3).
        from sklearn.linear_model import Ridge as SkRidge
        inner_n = max(2, min(self.cfg.n_folds, n // 4))
        inner_splitter = _make_oof_splitter("kfold", inner_n, self.cfg.seed)
        cv_rmse = []
        for alpha in self.cfg.meta_alphas:
            errs = []
            for tr, va in inner_splitter.split(oof_preds, y):
                meta = SkRidge(alpha=alpha).fit(oof_preds[tr], y[tr])
                errs.append(float(np.sqrt(np.mean((meta.predict(oof_preds[va]) - y[va]) ** 2))))
            cv_rmse.append(float(np.mean(errs)))
        best = int(np.argmin(cv_rmse))
        self.selected_meta_alpha_ = float(self.cfg.meta_alphas[best])
        self.meta_ = SkRidge(alpha=self.selected_meta_alpha_).fit(oof_preds, y)

        # Refit base learners on the full training set.
        self.base_models_ = []
        for learner_name in self.cfg.base_learners:
            est = _build_base_estimator(learner_name, seed=self.cfg.seed, cfg=self.cfg)
            est.fit(X, y)
            self.base_models_.append((learner_name, est))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.meta_ is None or not self.base_models_:
            raise RuntimeError("StackedRegressor not fitted")
        feats = np.column_stack([est.predict(X) for _, est in self.base_models_])
        return self.meta_.predict(feats)

    @property
    def hyperparams(self) -> dict:
        return {
            "model": "Stacked",
            "base_learners": list(self.cfg.base_learners),
            "selected_meta_alpha": self.selected_meta_alpha_,
            "alpha_at_boundary": (
                self.selected_meta_alpha_ == self.cfg.meta_alphas[0]
                or self.selected_meta_alpha_ == self.cfg.meta_alphas[-1]
            ),
            "n_folds": self.cfg.n_folds,
            "splitter_kind": self.cfg.splitter_kind,
            "cnn_v1c_kwargs": self.cfg.cnn_v1c_kwargs,
            "cnn_use_bjerrum": self.cfg.cnn_use_bjerrum,
            "alpha_grid_min": float(min(self.cfg.meta_alphas)),
            "alpha_grid_max": float(max(self.cfg.meta_alphas)),
        }
