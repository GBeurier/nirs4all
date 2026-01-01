"""
Utilities for FCK-PLS Torch experiments.

- Cross-validation search
- Hyperparameter grids
- Visualization helpers
- Model analysis tools
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

from fckpls_torch import FCKPLSTorch, TrainConfig


# =============================================================================
# CV Search
# =============================================================================

@dataclass
class CVResult:
    """Result from one CV run."""
    params: Dict[str, Any]
    fold_scores_rmse: List[float]
    fold_scores_r2: List[float]
    mean_rmse: float
    std_rmse: float
    mean_r2: float
    std_r2: float


class FCKPLSCVSearch:
    """
    Lightweight cross-validation grid search for FCK-PLS Torch.

    Usage:
        search = FCKPLSCVSearch(
            param_grid={"n_kernels": [8, 16], "n_components": [5, 10, 15]},
            n_splits=5,
        )
        search.fit(X, y)
        best_model = search.refit_best(X, y)
    """

    def __init__(
        self,
        param_grid: Dict[str, Sequence[Any]],
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        scorer: str = "rmse",  # "rmse" or "r2"
        verbose: int = 1,
        base_params: Optional[Dict[str, Any]] = None,
    ):
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.scorer = scorer
        self.verbose = verbose
        self.base_params = base_params or {}

        self.results_: List[CVResult] = []
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self.cv_results_df_: Optional[pd.DataFrame] = None

    def _iter_grid(self) -> Iterable[Dict[str, Any]]:
        """Iterate over all parameter combinations."""
        keys = list(self.param_grid.keys())
        vals = [list(self.param_grid[k]) for k in keys]

        def rec(i: int, cur: Dict[str, Any]):
            if i == len(keys):
                yield dict(cur)
                return
            for v in vals[i]:
                cur[keys[i]] = v
                yield from rec(i + 1, cur)

        yield from rec(0, {})

    def _to_2d(self, y: np.ndarray) -> np.ndarray:
        if y.ndim == 1:
            return y.reshape(-1, 1)
        return y

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FCKPLSCVSearch":
        """Run CV search over parameter grid."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        best_score = float("inf") if self.scorer == "rmse" else float("-inf")
        best_params = None

        param_list = list(self._iter_grid())
        n_params = len(param_list)

        for p_i, params in enumerate(param_list):
            if self.verbose:
                print(f"\n[{p_i+1}/{n_params}] params={params}")

            fold_rmses = []
            fold_r2s = []

            for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
                X_tr, X_va = X[tr_idx], X[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]

                # Create model with combined params
                all_params = {**self.base_params, **params}
                model = FCKPLSTorch(**all_params)

                try:
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_va)

                    rmse = np.sqrt(mean_squared_error(y_va, y_pred))
                    r2 = r2_score(y_va.ravel(), y_pred.ravel())

                    fold_rmses.append(rmse)
                    fold_r2s.append(r2)

                    if self.verbose:
                        print(f"  fold={fold}: RMSE={rmse:.4f}, R²={r2:.4f}")

                except Exception as e:
                    print(f"  fold={fold}: ERROR - {e}")
                    fold_rmses.append(np.nan)
                    fold_r2s.append(np.nan)

            mean_rmse = np.nanmean(fold_rmses)
            std_rmse = np.nanstd(fold_rmses)
            mean_r2 = np.nanmean(fold_r2s)
            std_r2 = np.nanstd(fold_r2s)

            self.results_.append(CVResult(
                params=dict(params),
                fold_scores_rmse=fold_rmses,
                fold_scores_r2=fold_r2s,
                mean_rmse=mean_rmse,
                std_rmse=std_rmse,
                mean_r2=mean_r2,
                std_r2=std_r2,
            ))

            if self.verbose:
                print(f"  mean_rmse={mean_rmse:.4f}±{std_rmse:.4f}, mean_r2={mean_r2:.4f}±{std_r2:.4f}")

            # Check if best
            if self.scorer == "rmse":
                if mean_rmse < best_score:
                    best_score = mean_rmse
                    best_params = dict(params)
            else:
                if mean_r2 > best_score:
                    best_score = mean_r2
                    best_params = dict(params)

        self.best_params_ = best_params
        self.best_score_ = best_score

        # Build results DataFrame
        rows = []
        for r in self.results_:
            row = dict(r.params)
            row["mean_rmse"] = r.mean_rmse
            row["std_rmse"] = r.std_rmse
            row["mean_r2"] = r.mean_r2
            row["std_r2"] = r.std_r2
            rows.append(row)
        self.cv_results_df_ = pd.DataFrame(rows)

        if self.verbose:
            print(f"\nBest: {self.best_params_} ({self.scorer}={self.best_score_:.4f})")

        return self

    def refit_best(self, X: np.ndarray, y: np.ndarray) -> FCKPLSTorch:
        """Refit model with best parameters on full data."""
        if self.best_params_ is None:
            raise RuntimeError("Call fit() first.")

        all_params = {**self.base_params, **self.best_params_}
        model = FCKPLSTorch(**all_params)
        model.fit(X, y)
        return model


# =============================================================================
# Default Parameter Grids
# =============================================================================

GRID_QUICK = {
    "n_kernels": [8, 16],
    "n_components": [5, 10],
    "ridge_lambda": [1e-3],
}

GRID_STANDARD = {
    "version": ["v1", "v2"],
    "n_kernels": [8, 16, 32],
    "kernel_size": [15, 31],
    "n_components": [5, 10, 15],
    "ridge_lambda": [1e-4, 1e-3, 1e-2],
}

GRID_V1_FULL = {
    "version": ["v1"],
    "n_kernels": [8, 16, 32],
    "kernel_size": [15, 31, 51],
    "n_components": [5, 10, 15, 20],
    "ridge_lambda": [1e-5, 1e-4, 1e-3, 1e-2],
    "init_mode": ["random", "fractional"],
}

GRID_V2_FULL = {
    "version": ["v2"],
    "n_kernels": [8, 16, 32],
    "kernel_size": [15, 31, 51],
    "n_components": [5, 10, 15, 20],
    "ridge_lambda": [1e-5, 1e-4, 1e-3, 1e-2],
    "alpha_max": [2.0],
    "tau": [0.5, 1.0, 2.0],
}


# =============================================================================
# Analysis Tools
# =============================================================================

def analyze_kernels(model: FCKPLSTorch) -> Dict[str, Any]:
    """Analyze learned kernels from FCK-PLS Torch model."""
    kernels = model.get_kernels()  # (K, ks)

    analysis = {
        "n_kernels": kernels.shape[0],
        "kernel_size": kernels.shape[1],
        "kernel_means": kernels.mean(axis=1),
        "kernel_stds": kernels.std(axis=1),
        "kernel_l1_norms": np.abs(kernels).sum(axis=1),
        "kernel_l2_norms": np.sqrt((kernels ** 2).sum(axis=1)),
    }

    # Check symmetry (derivative-like kernels should be antisymmetric)
    mid = kernels.shape[1] // 2
    left = kernels[:, :mid]
    right = np.flip(kernels[:, mid+1:], axis=1)
    if left.shape[1] == right.shape[1]:
        symmetry = np.abs(left + right).mean(axis=1)  # 0 = antisymmetric
        analysis["antisymmetry_score"] = symmetry

    # If v2, add alpha/sigma info
    if model.version == "v2" and hasattr(model.model_.extractor, "get_alphas"):
        analysis["alphas"] = model.model_.extractor.get_alphas()
        analysis["sigmas"] = model.model_.extractor.get_sigmas()

    return analysis


def compare_kernels_to_reference(
    model: FCKPLSTorch,
    reference_alphas: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
    reference_sigma: float = 3.0,
) -> Dict[str, Any]:
    """
    Compare learned kernels to reference fractional kernels.

    Useful to see if learned kernels resemble standard fractional derivatives.
    """
    from fckpls_torch import FractionalKernelBank

    learned = model.get_kernels()
    kernel_size = learned.shape[1]

    # Build reference kernels
    ref_bank = FractionalKernelBank(
        n_kernels=len(reference_alphas),
        kernel_size=kernel_size,
        alpha_init=list(reference_alphas),
        sigma_init=[reference_sigma] * len(reference_alphas),
    )
    reference = ref_bank.get_kernels()

    # Compute correlations
    correlations = np.zeros((learned.shape[0], reference.shape[0]))
    for i, lk in enumerate(learned):
        for j, rk in enumerate(reference):
            # Normalize and compute correlation
            lk_n = lk / (np.linalg.norm(lk) + 1e-8)
            rk_n = rk / (np.linalg.norm(rk) + 1e-8)
            correlations[i, j] = np.dot(lk_n, rk_n)

    # Best match for each learned kernel
    best_matches = correlations.argmax(axis=1)
    best_corrs = correlations.max(axis=1)

    return {
        "correlations": correlations,
        "best_match_idx": best_matches,
        "best_match_alpha": [reference_alphas[i] for i in best_matches],
        "best_correlation": best_corrs,
        "reference_alphas": list(reference_alphas),
    }


def plot_training_history(model: FCKPLSTorch, ax=None):
    """Plot training loss history."""
    import matplotlib.pyplot as plt

    if not model.training_history_:
        print("No training history available")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    epochs = [h["epoch"] for h in model.training_history_]
    losses = [h["loss"] for h in model.training_history_]
    mses = [h["mse"] for h in model.training_history_]
    regs = [h["reg"] for h in model.training_history_]

    ax.plot(epochs, losses, "b-", label="Total Loss", linewidth=2)
    ax.plot(epochs, mses, "g--", label="MSE", linewidth=1.5)
    ax.plot(epochs, regs, "r:", label="Regularization", linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_kernel_comparison(
    model: FCKPLSTorch,
    reference_alphas: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
    n_show: int = 8,
):
    """Plot learned kernels alongside reference fractional kernels."""
    import matplotlib.pyplot as plt

    comparison = compare_kernels_to_reference(model, reference_alphas)
    learned = model.get_kernels()

    n_show = min(n_show, learned.shape[0])

    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))

    kernel_size = learned.shape[1]
    x = np.arange(kernel_size) - kernel_size // 2

    for i in range(n_show):
        # Learned kernel
        ax = axes[0, i]
        ax.plot(x, learned[i], "b-", linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"Learned {i+1}")
        if i == 0:
            ax.set_ylabel("Learned")

        # Matching reference
        ax = axes[1, i]
        best_alpha = comparison["best_match_alpha"][i]
        best_corr = comparison["best_correlation"][i]

        # Rebuild reference kernel
        from fckpls_torch import FractionalKernelBank
        ref = FractionalKernelBank(1, kernel_size, alpha_init=[best_alpha], sigma_init=[3.0])
        ref_kernel = ref.get_kernels()[0]

        ax.plot(x, ref_kernel, "r-", linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"α={best_alpha:.1f} (r={best_corr:.2f})")
        if i == 0:
            ax.set_ylabel("Reference")

    plt.suptitle("Learned vs Reference Fractional Kernels")
    plt.tight_layout()
    plt.show()
