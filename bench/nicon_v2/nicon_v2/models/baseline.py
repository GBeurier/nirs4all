"""Phase 0 baselines.

* :class:`RidgeBaseline` — sklearn ``Ridge`` with ``StandardScaler`` and α via 5-fold CV.
* :class:`PLSBaseline`   — sklearn ``PLSRegression`` with auto n-components via 5-fold CV.
* :func:`build_nicon_torch` — wraps the existing PyTorch ``nicon`` factory from `nirs4all`.
* :func:`build_decon_torch` — wraps the existing PyTorch ``decon`` factory from `nirs4all`.

All baselines fit on raw arrays and expose ``fit / predict`` so the benchmark runner can
treat them uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .. import CODE_VERSION  # noqa: F401  (stamp on serialised models if needed)


# ---------------------------------------------------------------------------
# Ridge baseline
# ---------------------------------------------------------------------------


@dataclass
class RidgeBaseline:
    """sklearn Ridge with StandardScaler. α chosen by 5-fold CV from a logspace grid.

    The CV is feature-fold-local: scaler is refit on each fold's train split, never on
    validation data, to avoid leakage.
    """

    alphas: tuple[float, ...] = field(default_factory=lambda: tuple(10.0 ** k for k in np.linspace(-3.0, 3.0, 13)))
    cv_splits: int = 5
    seed: int = 0
    selected_alpha_: float = float("nan")
    pipeline_: tuple[StandardScaler, Ridge] | None = None
    n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeBaseline":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_ = X.shape[1]
        kf = KFold(n_splits=self.cv_splits, shuffle=True, random_state=self.seed)
        cv_rmse = []
        for alpha in self.alphas:
            errs = []
            for train_idx, val_idx in kf.split(X):
                scaler = StandardScaler().fit(X[train_idx])
                Xt = scaler.transform(X[train_idx])
                Xv = scaler.transform(X[val_idx])
                model = Ridge(alpha=alpha).fit(Xt, y[train_idx])
                pred = model.predict(Xv)
                errs.append(float(np.sqrt(np.mean((pred - y[val_idx]) ** 2))))
            cv_rmse.append(float(np.mean(errs)))
        best = int(np.argmin(cv_rmse))
        self.selected_alpha_ = float(self.alphas[best])
        scaler = StandardScaler().fit(X)
        model = Ridge(alpha=self.selected_alpha_).fit(scaler.transform(X), y)
        self.pipeline_ = (scaler, model)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("RidgeBaseline not fitted")
        scaler, model = self.pipeline_
        return model.predict(scaler.transform(np.asarray(X, dtype=float)))

    @property
    def hyperparams(self) -> dict[str, Any]:
        return {
            "model": "Ridge",
            "selected_alpha": self.selected_alpha_,
            "cv_splits": self.cv_splits,
            "alpha_grid": list(self.alphas),
            "scaler": "StandardScaler",
        }


# ---------------------------------------------------------------------------
# PLS baseline
# ---------------------------------------------------------------------------


@dataclass
class PLSBaseline:
    """sklearn PLSRegression with auto n-components from a small CV grid."""

    n_components_grid: tuple[int, ...] = (1, 2, 3, 5, 7, 10, 12, 15, 20, 25)
    cv_splits: int = 5
    seed: int = 0
    selected_n_components_: int = 0
    pipeline_: tuple[StandardScaler, PLSRegression] | None = None
    n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSBaseline":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_ = X.shape[1]
        max_components = min(X.shape[0] - 1, X.shape[1])
        candidates = tuple(c for c in self.n_components_grid if c <= max_components)
        if not candidates:
            candidates = (1,)
        kf = KFold(n_splits=self.cv_splits, shuffle=True, random_state=self.seed)
        cv_rmse = []
        for n_comp in candidates:
            errs = []
            for train_idx, val_idx in kf.split(X):
                scaler = StandardScaler().fit(X[train_idx])
                Xt = scaler.transform(X[train_idx])
                Xv = scaler.transform(X[val_idx])
                pls = PLSRegression(n_components=n_comp).fit(Xt, y[train_idx])
                pred = pls.predict(Xv).ravel()
                errs.append(float(np.sqrt(np.mean((pred - y[val_idx]) ** 2))))
            cv_rmse.append(float(np.mean(errs)))
        best = int(np.argmin(cv_rmse))
        self.selected_n_components_ = int(candidates[best])
        scaler = StandardScaler().fit(X)
        model = PLSRegression(n_components=self.selected_n_components_).fit(scaler.transform(X), y)
        self.pipeline_ = (scaler, model)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("PLSBaseline not fitted")
        scaler, model = self.pipeline_
        return model.predict(scaler.transform(np.asarray(X, dtype=float))).ravel()

    @property
    def hyperparams(self) -> dict[str, Any]:
        return {
            "model": "PLSRegression",
            "selected_n_components": self.selected_n_components_,
            "cv_splits": self.cv_splits,
            "n_components_grid": list(self.n_components_grid),
            "scaler": "StandardScaler",
        }


# ---------------------------------------------------------------------------
# Torch baselines wrapping the upstream nirs4all builders
# ---------------------------------------------------------------------------


def build_nicon_torch(input_shape: tuple[int, int], params: dict | None = None) -> nn.Module:
    """Build the upstream ``nirs4all`` PyTorch ``nicon`` model. Read-only import.

    ``input_shape = (channels, sequence_length)``.
    """
    from nirs4all.operators.models.pytorch.nicon import _build_nicon

    return _build_nicon(input_shape, params or {}, num_classes=1)


def build_decon_torch(input_shape: tuple[int, int], params: dict | None = None) -> nn.Module:
    from nirs4all.operators.models.pytorch.nicon import _build_decon

    return _build_decon(input_shape, params or {}, num_classes=1)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def cuda_peak_mb() -> float:
    if not torch.cuda.is_available():
        return float("nan")
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
