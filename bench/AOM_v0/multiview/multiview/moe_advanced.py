"""Advanced MoE variants: per-sample routing + multi-K ensemble.

Three enhancements over the constant-gate `AOMMoERegressor`:

1. `AOMMoEPerSampleRouting` — replaces the dataset-level NNLS gate with
   a per-sample classifier trained on `(X, OOF residuals)` to predict the
   best expert per sample. **Rejected on smoke tests** — argmax labels
   are too noisy on small datasets.

2. `AOMMoEStacked` — preferred per-sample approach. Trains a Ridge meta on
   `[X_subsample | OOF_expert_predictions]` so the meta-model sees both
   raw features and per-expert predictions. Equivalent to a 2-layer
   architecture where layer 1 is the expert pool and layer 2 is a
   regularised linear combiner with per-sample feature awareness.

3. `AOMMoEMultiK` — fits multiple `AOMMoERegressor` instances at different
   K values (e.g. K=3, K=5, K=7) and averages their predictions. Hedges
   against the K-parameter being dataset-dependent.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone  # noqa: F401
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .moe import AOMMoERegressor, _make_block_view, _make_op_view
from .views import _equal_width_blocks


class AOMMoEPerSampleRouting(BaseEstimator, RegressorMixin):
    """MoE with a per-sample gate that picks the best expert per test sample.

    Pipeline:

    1. Build K experts (per_view: K equal-width blocks; per_preproc: K
       operators in `bank_name`).
    2. Generate OOF predictions per expert via K-fold on training data.
    3. For each sample, identify the best expert (argmin |y - OOF_pred|).
    4. Train a gate classifier on `X → best_expert_label`.
    5. Refit each expert on full training data.
    6. At predict: gate predicts soft probabilities; blend expert outputs.

    Parameters mirror `AOMMoERegressor`. Additional:
        gate_classifier: "logreg" or "rf".
        gate_use_residuals: if True, append OOF residual stats to gate
            features (richer signal at the cost of a bigger gate input).
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        expert_layout: str = "per_view",
        K: int = 3,
        bank_name: str = "compact",
        per_expert_components: int = 10,
        n_oof_folds: int = 3,
        random_state: int = 0,
        gate_classifier: str = "logreg",
        gate_use_residuals: bool = False,
    ) -> None:
        self.expert_layout = expert_layout
        self.K = K
        self.bank_name = bank_name
        self.per_expert_components = per_expert_components
        self.n_oof_folds = n_oof_folds
        self.random_state = random_state
        self.gate_classifier = gate_classifier
        self.gate_use_residuals = gate_use_residuals

    def _build_views(self, p: int):
        if self.expert_layout == "per_view":
            blocks = _equal_width_blocks(p, self.K)
            return [
                (f"view_{k}", _make_block_view(s, e))
                for k, (s, e) in enumerate(blocks)
            ]
        if self.expert_layout == "per_preproc":
            from aompls.banks import bank_by_name
            bank = bank_by_name(self.bank_name, p=p)
            return [(op.name, _make_op_view(op)) for op in bank]
        raise ValueError(f"unknown expert_layout: {self.expert_layout!r}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AOMMoEPerSampleRouting":
        start = time.perf_counter()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        views = self._build_views(p)
        E = len(views)

        # Generate OOF predictions per expert via K-fold.
        kf = KFold(n_splits=self.n_oof_folds, shuffle=True, random_state=self.random_state)
        oof_pred = np.zeros((n, E), dtype=float)
        for train_idx, val_idx in kf.split(X):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr = y[train_idx]
            for e, (_name, view_fn) in enumerate(views):
                X_tr_e = view_fn(X_tr)
                X_va_e = view_fn(X_va)
                k = min(self.per_expert_components, max(1, X_tr.shape[0] - 1), X_tr_e.shape[1])
                try:
                    pls = PLSRegression(n_components=k)
                    pls.fit(X_tr_e, y_tr)
                    oof_pred[val_idx, e] = pls.predict(X_va_e).ravel()
                except Exception:
                    oof_pred[val_idx, e] = float(y_tr.mean())

        # Per-sample best-expert label = argmin |y - OOF[:,e]|.
        residuals = oof_pred - y[:, None]
        best_expert_per_sample = np.argmin(np.abs(residuals), axis=1)

        # Train gate.
        gate_input = X.copy()
        if self.gate_use_residuals:
            # Append per-expert OOF residual statistics — richer signal.
            res_abs = np.abs(residuals)
            gate_input = np.column_stack([gate_input, res_abs])

        scaler = StandardScaler()
        gate_input_s = scaler.fit_transform(gate_input)

        # If only one class is present (one expert dominates everywhere),
        # fall back to the constant gate (simple AOMMoERegressor logic).
        unique = np.unique(best_expert_per_sample)
        if len(unique) == 1:
            gate = None
            constant_argmax = int(unique[0])
        else:
            if self.gate_classifier == "logreg":
                gate = LogisticRegression(
                    max_iter=2000, multi_class="multinomial",
                    class_weight="balanced", random_state=self.random_state,
                )
            elif self.gate_classifier == "rf":
                gate = RandomForestClassifier(
                    n_estimators=200, max_depth=4, min_samples_leaf=2,
                    class_weight="balanced", random_state=self.random_state, n_jobs=1,
                )
            else:
                raise ValueError(f"unknown gate_classifier: {self.gate_classifier!r}")
            gate.fit(gate_input_s, best_expert_per_sample)
            constant_argmax = None

        # Refit each expert on full training data.
        full_experts = []
        for _name, view_fn in views:
            X_full = view_fn(X)
            k = min(self.per_expert_components, max(1, n - 1), X_full.shape[1])
            pls = PLSRegression(n_components=k)
            pls.fit(X_full, y)
            full_experts.append(pls)

        self.views_ = views
        self.full_experts_ = full_experts
        self.gate_ = gate
        self.gate_scaler_ = scaler
        self.constant_argmax_ = constant_argmax
        self.gate_classes_ = unique
        self.fit_time_s_ = float(time.perf_counter() - start)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "full_experts_"):
            raise RuntimeError("Estimator not fitted")
        X = np.asarray(X, dtype=float)
        E = len(self.views_)
        # Expert predictions
        preds = np.zeros((X.shape[0], E), dtype=float)
        for e, ((_name, view_fn), pls) in enumerate(zip(self.views_, self.full_experts_)):
            preds[:, e] = pls.predict(view_fn(X)).ravel()

        # Per-sample gate weights
        if self.gate_ is None:
            # Constant gate fallback (one expert dominates).
            return preds[:, self.constant_argmax_]

        gate_input = X.copy()
        if self.gate_use_residuals:
            # At predict time we don't have OOF residuals; pad with zeros.
            gate_input = np.column_stack([gate_input, np.zeros((X.shape[0], E))])
        gate_input_s = self.gate_scaler_.transform(gate_input)
        gate_probs_partial = self.gate_.predict_proba(gate_input_s)
        # Map partial probs (over self.gate_classes_) to full E-vector.
        gate_probs = np.zeros((X.shape[0], E))
        for col_idx, class_idx in enumerate(self.gate_classes_):
            gate_probs[:, int(class_idx)] = gate_probs_partial[:, col_idx]
        # Soft mixture: per-sample weighted blend.
        return (gate_probs * preds).sum(axis=1)


class AOMMoEStacked(BaseEstimator, RegressorMixin):
    """Per-sample routing via Ridge meta on `[X | OOF_expert_predictions]`.

    Replaces the noisy argmax-label classifier from
    `AOMMoEPerSampleRouting` with a regression-based stacker. The Ridge
    meta sees both raw features and per-expert predictions, so it can
    learn per-sample blending without needing crisp expert labels.

    To control the dimensionality of the X part of the meta input, X
    is reduced to its top-`x_pca_components` PCA scores before
    concatenation. This keeps the meta-input width bounded
    (`x_pca_components + n_experts`) regardless of original `p`.

    Parameters
    ----------
    expert_layout : str
        "per_view" (K equal-width blocks) or "per_preproc" (`bank_name` ops).
    K : int
        Number of equal-width blocks (per_view only).
    bank_name : str
        Operator bank (per_preproc only).
    per_expert_components : int
        n_components for each PLS expert.
    n_oof_folds : int
        K-fold count for OOF prediction generation.
    x_pca_components : int
        PCA components of X to include in the Ridge meta input.
    meta_alpha : float
        Ridge regularisation for the meta-model.
    random_state : int
        Seed for fold splits.
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        expert_layout: str = "per_view",
        K: int = 3,
        bank_name: str = "compact",
        per_expert_components: int = 10,
        n_oof_folds: int = 3,
        x_pca_components: int = 5,
        meta_alpha: float = 1.0,
        random_state: int = 0,
    ) -> None:
        self.expert_layout = expert_layout
        self.K = K
        self.bank_name = bank_name
        self.per_expert_components = per_expert_components
        self.n_oof_folds = n_oof_folds
        self.x_pca_components = x_pca_components
        self.meta_alpha = meta_alpha
        self.random_state = random_state

    def _build_views(self, p: int):
        if self.expert_layout == "per_view":
            blocks = _equal_width_blocks(p, self.K)
            return [(f"view_{k}", _make_block_view(s, e)) for k, (s, e) in enumerate(blocks)]
        if self.expert_layout == "per_preproc":
            from aompls.banks import bank_by_name
            bank = bank_by_name(self.bank_name, p=p)
            return [(op.name, _make_op_view(op)) for op in bank]
        raise ValueError(f"unknown expert_layout: {self.expert_layout!r}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AOMMoEStacked":
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge

        start = time.perf_counter()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        views = self._build_views(p)
        E = len(views)

        # Generate OOF expert predictions
        kf = KFold(n_splits=self.n_oof_folds, shuffle=True, random_state=self.random_state)
        oof = np.zeros((n, E), dtype=float)
        for tr, va in kf.split(X):
            for e, (_n, vfn) in enumerate(views):
                Xtr_e = vfn(X[tr])
                Xva_e = vfn(X[va])
                k = min(self.per_expert_components, max(1, len(tr) - 1), Xtr_e.shape[1])
                try:
                    pls = PLSRegression(n_components=k)
                    pls.fit(Xtr_e, y[tr])
                    oof[va, e] = pls.predict(Xva_e).ravel()
                except Exception:
                    oof[va, e] = float(y[tr].mean())

        # Reduce X via PCA to bound meta-input dimensionality.
        n_pca = min(self.x_pca_components, p, n - 1)
        self._pca = PCA(n_components=n_pca, random_state=self.random_state)
        X_pca = self._pca.fit_transform(X)

        # Meta input: [X_pca | oof_expert_predictions]
        meta_X = np.column_stack([X_pca, oof])
        self._meta = Ridge(alpha=self.meta_alpha, fit_intercept=True)
        self._meta.fit(meta_X, y)

        # Refit experts on full data
        full_experts = []
        for _name, vfn in views:
            Xv = vfn(X)
            k = min(self.per_expert_components, max(1, n - 1), Xv.shape[1])
            pls = PLSRegression(n_components=k)
            pls.fit(Xv, y)
            full_experts.append(pls)

        self.views_ = views
        self.full_experts_ = full_experts
        self.fit_time_s_ = float(time.perf_counter() - start)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "full_experts_"):
            raise RuntimeError("Estimator not fitted")
        X = np.asarray(X, dtype=float)
        E = len(self.views_)
        # Expert predictions
        preds = np.zeros((X.shape[0], E), dtype=float)
        for e, ((_n, vfn), pls) in enumerate(zip(self.views_, self.full_experts_)):
            preds[:, e] = pls.predict(vfn(X)).ravel()
        # PCA + meta
        X_pca = self._pca.transform(X)
        meta_X = np.column_stack([X_pca, preds])
        return self._meta.predict(meta_X).ravel()


class AOMMoEMultiK(BaseEstimator, RegressorMixin):
    """Average predictions of `AOMMoERegressor` at multiple K values.

    Hedges against K-parameter dataset dependence. Equal-weights average
    of K_list MoE instances. Cheap if K_list is small (3-5 values) since
    each MoE is independent.

    With ``per_expert_components="auto"``, the per-expert PLS depth scales
    with the training-set size: 10 for n<200, 15 for n<1000, 25 otherwise.
    Bigger datasets benefit from more latent variables.
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        K_list: tuple = (3, 5, 7),
        per_expert_components=10,
        random_state: int = 0,
    ) -> None:
        self.K_list = K_list
        self.per_expert_components = per_expert_components
        self.random_state = random_state

    def _resolve_components(self, n: int) -> int:
        if self.per_expert_components == "auto":
            if n < 200:
                return 10
            if n < 1000:
                return 15
            return 25
        return int(self.per_expert_components)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AOMMoEMultiK":
        start = time.perf_counter()
        n = X.shape[0]
        comp = self._resolve_components(n)
        self._models = []
        for k in self.K_list:
            mdl = AOMMoERegressor(
                expert_layout="per_view", routing="soft", K=int(k),
                per_expert_components=comp,
                random_state=self.random_state,
            )
            mdl.fit(X, y)
            self._models.append(mdl)
        self.fit_time_s_ = float(time.perf_counter() - start)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_models"):
            raise RuntimeError("Estimator not fitted")
        preds = np.column_stack([m.predict(X) for m in self._models])
        return preds.mean(axis=1)


class MultiVariantMeanEnsemble(BaseEstimator, RegressorMixin):
    """Mean of test predictions across heterogeneous base estimators.

    Each base is fit on the full training data; their test-time predictions
    are averaged with equal weights. Cheap, robust, and benefits from
    uncorrelated errors when bases use different mechanisms.

    Parameters
    ----------
    bases : list of (name, estimator) tuples
        Base sklearn estimators. Each is `clone`d on `fit`.
    """

    _estimator_type = "regressor"

    def __init__(self, bases=None) -> None:
        self.bases = bases

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiVariantMeanEnsemble":
        if not self.bases:
            raise ValueError("bases must be a non-empty list")
        start = time.perf_counter()
        self._fitted = []
        for name, est in self.bases:
            e = clone(est)
            e.fit(X, y)
            self._fitted.append((name, e))
        self.fit_time_s_ = float(time.perf_counter() - start)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_fitted"):
            raise RuntimeError("Estimator not fitted")
        preds = np.column_stack([np.asarray(e.predict(X)).ravel() for _n, e in self._fitted])
        return preds.mean(axis=1)
