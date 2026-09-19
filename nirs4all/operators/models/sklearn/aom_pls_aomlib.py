"""Historical AOMPLSAomlibRegressor class path backed by nirs4all-methods.

The native global AOM selector owns operator/component selection and prediction.
The old external ``aompls`` package is no longer required. Unsupported historical
selection/preprocessing options raise explicitly instead of changing meaning.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

_AOMPLS_IMPORT_HINT = (
    "AOMPLSAomlibRegressor requires the nirs4all-methods native Python wheel. "
    "Install a compatible nirs4all-methods version in the Python environment "
    "running this application; its wheel includes the native library."
)


class AOMPLSAomlibRegressor(RegressorMixin, BaseEstimator):
    """Native global AOM-PLS selector with the historical nirs4all class path.

    ``n_components`` bounds component selection; ``cv`` controls internal folds.
    ``selection='cv'``/``'kfold'`` uses deterministic contiguous folds unless a
    random_state is supplied, in which case shuffled KFold is used. Explicit
    external_folds lists must partition the training samples exactly once.
    ``center`` controls native X/Y centering, with scaling disabled explicitly.

    Legacy one-SE selection, SPXY/holdout modes and built-in preprocessing are
    not provided by this native selector and are rejected. Use explicit pipeline
    preprocessing steps. Their constructor fields remain only so existing saved
    configurations get clear errors, rather than silently losing parameters.
    """

    _webapp_meta = {
        "category": "pls",
        "tier": "advanced",
        "tags": ["pls", "aom-pls", "aom_lib", "regression", "cpp-backend"],
    }

    _estimator_type = "regressor"
    # Availability checks can validate the lazy backend without fitting a model.
    _required_imports = ("n4m.model_selection.aom_search.AOMPLSRegressor",)
    _dependency_installation_hint = _AOMPLS_IMPORT_HINT

    def __init__(
        self,
        n_components: int = 15,
        selection: str = "cv",
        cv: int = 5,
        one_se: bool = False,
        preprocessing: str | None = None,
        random_state: int | None = None,
        osc_n_components: int = 1,
        asls_lam: float = 1e5,
        asls_p: float = 0.01,
        asls_n_iter: int = 10,
        center: bool = True,
        external_folds: Any | None = None,
    ) -> None:
        self.n_components = n_components
        self.selection = selection
        self.cv = cv
        self.one_se = one_se
        self.preprocessing = preprocessing
        self.random_state = random_state
        self.osc_n_components = osc_n_components
        self.asls_lam = asls_lam
        self.asls_p = asls_p
        self.asls_n_iter = asls_n_iter
        self.center = center
        self.external_folds = external_folds

    def _make_backend(self, n_samples: int) -> Any:
        """Translate supported configuration to the installed public binding."""
        if self.selection not in ("cv", "kfold", "external"):
            raise ValueError(f"Unsupported selection mode {self.selection!r} for nirs4all-methods AOM-PLS; use cv, kfold or external.")
        if self.one_se:
            raise ValueError("one_se=True is not supported by nirs4all-methods AOM-PLS.")
        if self.preprocessing not in (None, "none"):
            raise ValueError("Built-in preprocessing is not supported by nirs4all-methods AOM-PLS; add explicit pipeline preprocessing steps.")
        if (self.osc_n_components, self.asls_lam, self.asls_p, self.asls_n_iter) != (1, 1e5, 0.01, 10):
            raise ValueError("Legacy OSC/ASLS parameters are unsupported; configure an explicit pipeline preprocessing step.")
        if self.selection != "external" and self.external_folds is not None:
            raise ValueError("external_folds requires selection='external'.")
        if self.selection == "external" and self.external_folds is None:
            raise ValueError("selection='external' requires external_folds.")
        try:
            from n4m.model_selection.aom_search import AOMPLSRegressor
        except ImportError as exc:
            raise ImportError(_AOMPLS_IMPORT_HINT) from exc

        fold_ids = None
        folds = int(self.cv)
        if self.external_folds is not None:
            fold_ids = np.full(n_samples, -1, dtype=np.int32)
            folds = len(self.external_folds)
            if folds < 2:
                raise ValueError("external_folds must contain at least two nonempty validation folds.")
            for fold, indices in enumerate(self.external_folds):
                values = np.asarray(indices)
                if values.ndim != 1 or not values.size or not np.issubdtype(values.dtype, np.integer):
                    raise ValueError("external_folds must contain nonempty integer index lists.")
                if np.any(values < 0) or np.any(values >= n_samples) or len(np.unique(values)) != len(values) or np.any(fold_ids[values] != -1):
                    raise ValueError("external_folds must partition training samples exactly once.")
                fold_ids[values] = fold
            if np.any(fold_ids == -1):
                raise ValueError("external_folds must partition training samples exactly once.")
        elif self.random_state is not None:
            fold_ids = np.empty(n_samples, dtype=np.int32)
            for fold, (_, validation) in enumerate(KFold(folds, shuffle=True, random_state=self.random_state).split(np.empty(n_samples))):
                fold_ids[validation] = fold
        return AOMPLSRegressor(max_components=int(self.n_components), cv=folds, fold_ids=fold_ids,
                               center_x=bool(self.center), center_y=bool(self.center), scale_x=False, scale_y=False)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_val: ArrayLike | None = None,  # noqa: ARG002 - kept for API symmetry
        y_val: ArrayLike | None = None,  # noqa: ARG002 - kept for API symmetry
    ) -> AOMPLSAomlibRegressor:
        """Fit through the installed nirs4all-methods C ABI binding.

        Args:
            X: Training spectra of shape ``(n_samples, n_features)``.
            y: Target values of shape ``(n_samples,)``. Multivariate ``y`` is
                not supported by the compact PLS1 backend and is reshaped to
                1D after squeezing trailing singleton dimensions.
            X_val: Unused. Kept for API symmetry with other nirs4all wrappers
                that accept an optional validation set; the native backend
                performs operator/K selection internally.
            y_val: Unused. See ``X_val``.

        Returns:
            ``self``, with diagnostic attributes populated.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim > 1:
            y_arr = np.squeeze(y_arr)
            if y_arr.ndim != 1:
                raise ValueError(
                    "AOMPLSAomlibRegressor only supports 1D targets "
                    f"(PLS1). Got y with shape {np.shape(y)}."
                )

        backend = self._make_backend(X_arr.shape[0])
        started = perf_counter()
        backend.fit(X_arr, y_arr)
        self.fit_time_s_ = perf_counter() - started

        self._backend = backend
        self.n_features_in_ = X_arr.shape[1]
        self.coef_ = np.asarray(backend.coef_, dtype=np.float64)
        self.intercept_ = float(backend.intercept_)
        self.backend_ = "nirs4all-methods"
        self.native_diagnostics_ = backend.get_diagnostics()
        kinds = backend.operator_kinds_
        self.bank_names_ = [f"native_operator_{index}_kind_{int(kind)}" for index, kind in enumerate(kinds)]
        fold_ids = np.asarray(backend.result_["fold_ids"], dtype=np.int32)
        self.fold_indices_ = [np.flatnonzero(fold_ids == index).tolist() for index in range(int(backend.cv))]
        self.one_se_applied_ = False
        self.n_components_selected_ = int(backend.selected_n_components_)
        self.selected_operator_index_ = int(backend.result_["selected_operator_index"])
        self.selected_operator_sequence_ = [self.bank_names_[self.selected_operator_index_]]
        self.selected_operator_scores_ = np.asarray(backend.result_["rmse_curves"], dtype=np.float64)

        return self

    def predict(self, X: ArrayLike) -> NDArray[np.floating]:
        """Predict target values for ``X``.

        Args:
            X: Spectra of shape ``(n_samples, n_features)``.

        Returns:
            1D array of float predictions of shape ``(n_samples,)``.
        """
        check_is_fitted(self, ["_backend", "coef_"])
        X_arr = np.asarray(X, dtype=np.float64)
        preds = self._backend.predict(X_arr)
        return np.asarray(preds, dtype=np.float64).ravel()

    def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002 - sklearn signature
        """Return constructor parameters for sklearn compatibility."""
        return {
            "n_components": self.n_components,
            "selection": self.selection,
            "cv": self.cv,
            "one_se": self.one_se,
            "preprocessing": self.preprocessing,
            "random_state": self.random_state,
            "osc_n_components": self.osc_n_components,
            "asls_lam": self.asls_lam,
            "asls_p": self.asls_p,
            "asls_n_iter": self.asls_n_iter,
            "center": self.center,
            "external_folds": self.external_folds,
        }

    def set_params(self, **params: Any) -> AOMPLSAomlibRegressor:
        """Set constructor parameters and return ``self``."""
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(
                    f"Invalid parameter '{key}' for AOMPLSAomlibRegressor."
                )
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"AOMPLSAomlibRegressor(n_components={self.n_components}, "
            f"selection='{self.selection}', cv={self.cv}, "
            f"one_se={self.one_se}, preprocessing={self.preprocessing!r})"
        )
