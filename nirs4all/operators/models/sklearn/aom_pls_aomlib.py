"""AOM-PLS wrapper around the dedicated ``aompls`` (AOM_lib) C++/Eigen backend.

This module exposes :class:`AOMPLSAomlibRegressor`, a thin sklearn-compatible
wrapper that delegates fitting and prediction to the
``aompls.AOMPLSCompact`` estimator shipped in ``bench/AOM_lib/python/src``.
The legacy pure-Python implementation in
:mod:`nirs4all.operators.models.sklearn.aom_pls` is kept untouched; this
wrapper is added to let nirs4all pipelines drive the dedicated
implementation used for the Talanta AOM-PLS submission.

Paper terminology mapping
-------------------------
==============================  ==================================
Paper term                      Code parameter
==============================  ==================================
Compact operator bank           built into ``AOMPLSCompact`` (PLS1)
Number of latent variables K    ``n_components`` -> ``max_components``
CV-based operator/K selector    ``selection="cv"`` -> ``cv_mode="kfold"``
SPXY selector                   ``selection="spxy"`` -> ``cv_mode="spxy"``
Hold-out selector               ``selection="holdout"`` -> ``cv_mode="holdout"``
One standard-error rule         ``one_se`` -> ``one_se_rule``
Optional one-shot preprocessing ``preprocessing`` -> ``preproc``
==============================  ==================================

The wrapper imports ``aompls`` lazily so that environments without the
compiled extension on ``PYTHONPATH`` can still import nirs4all. If
:meth:`AOMPLSAomlibRegressor.fit` is invoked without ``aompls`` available,
a clear :class:`ImportError` is raised pointing the user at the source
checkout or the future PyPI package.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

_AOMPLS_IMPORT_HINT = (
    "The 'aompls' package (AOM_lib C++/Eigen backend) is required for "
    "AOMPLSAomlibRegressor. Install it from the repository checkout with\n"
    "    PYTHONPATH=bench/AOM_lib/python/src python ...\n"
    "or wait for the upcoming PyPI release ('pip install aompls')."
)


def _selection_to_cv_mode(selection: str) -> str:
    """Map the nirs4all-facing ``selection`` argument to AOM_lib's ``cv_mode``.

    Args:
        selection: One of ``"cv"``, ``"kfold"``, ``"spxy"``, ``"holdout"`` or
            ``"external"``. ``"cv"`` is an alias for ``"kfold"``.

    Returns:
        The corresponding ``cv_mode`` string expected by ``AOMPLSCompact``.

    Raises:
        ValueError: If ``selection`` is not a recognised mode.
    """
    mapping = {
        "cv": "kfold",
        "kfold": "kfold",
        "spxy": "spxy",
        "holdout": "holdout",
        "external": "external",
    }
    if selection not in mapping:
        raise ValueError(
            f"Unknown selection mode '{selection}'. Expected one of "
            f"{sorted(mapping)}."
        )
    return mapping[selection]


class AOMPLSAomlibRegressor(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper around ``aompls.AOMPLSCompact`` (AOM_lib).

    Provides a nirs4all-friendly facade for the dedicated C++/Eigen
    implementation of AOM-PLS (PLS1, compact operator bank) that backs the
    Talanta paper. The wrapper handles lazy import of the ``aompls``
    extension, parameter translation between paper terminology and the C++
    backend, and mirrors the diagnostic attributes exposed by the legacy
    pure-Python :class:`AOMPLSRegressor`.

    Args:
        n_components: Maximum number of PLS components extracted during CV
            scoring and refit. Paper symbol: ``K_max``.
        selection: Operator/component selection strategy. ``"cv"`` (alias
            ``"kfold"``) uses K-fold CV, ``"spxy"`` uses SPXY folds,
            ``"holdout"`` uses a single split, ``"external"`` requires
            pre-computed folds via ``external_folds``.
        cv: Number of CV folds (ignored when ``selection="external"``).
        one_se: When ``True``, apply the one-standard-error parsimony rule
            on top of the CV operator/K selection.
        preprocessing: Optional one-shot preprocessing applied before AOM.
            ``None`` means no preprocessing; otherwise one of the strings
            accepted by ``AOMPLSCompact.preproc`` (e.g. ``"asls"``,
            ``"snv"``, ``"osc"``, ``"snv+osc"``, ``"asls+osc"``).
        random_state: Seed for the C++ shuffler used during CV. Defaults to
            ``0`` (matching ``AOMPLSCompact``).
        osc_n_components: Number of OSC components when ``preprocessing``
            includes ``"osc"``.
        asls_lam: AsLS lambda hyperparameter (used when ``preprocessing``
            includes ``"asls"``).
        asls_p: AsLS asymmetry weight.
        asls_n_iter: AsLS number of iterations.
        center: Whether to mean-center X and y before AOM. Defaults to
            ``True``; strongly recommended for spectral data.
        external_folds: Pre-computed test indices per fold. Required when
            ``selection="external"``.

    Attributes:
        n_features_in_: Number of features seen at fit time.
        n_components_selected_: Number of PLS components retained after CV.
        selected_operator_sequence_: List of length 1 containing the name of
            the operator picked by the CV selector. The list shape mirrors
            the per-component sequence exposed by the legacy POP-PLS-style
            wrappers so downstream code can treat both wrappers uniformly.
        selected_operator_scores_: Stacked CV RMSE curves per operator
            (shape ``(n_operators, K_max)``).
        coef_: Regression coefficients in the original feature space.
        intercept_: Regression intercept.
        bank_names_: Names of the operators in the compact bank.
        fold_indices_: Test indices per CV fold returned by the backend.
        fit_time_s_: Wall-clock fit time reported by the C++ backend.

    Examples:
        >>> from nirs4all.operators.models.sklearn import AOMPLSAomlibRegressor
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X = rng.standard_normal((40, 80))
        >>> y = X[:, :3].sum(axis=1) + 0.1 * rng.standard_normal(40)
        >>> model = AOMPLSAomlibRegressor(n_components=8, cv=3).fit(X, y)
        >>> preds = model.predict(X)
        >>> preds.shape
        (40,)
    """

    _webapp_meta = {
        "category": "pls",
        "tier": "advanced",
        "tags": ["pls", "aom-pls", "aom_lib", "regression", "cpp-backend"],
    }

    _estimator_type = "regressor"

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

    def _make_backend(self) -> Any:
        """Lazily import ``aompls`` and instantiate the backend estimator.

        Returns:
            An unfitted ``aompls.AOMPLSCompact`` instance with parameters
            translated from this wrapper's configuration.

        Raises:
            ImportError: If the ``aompls`` package cannot be imported.
        """
        try:
            from aompls import AOMPLSCompact
        except ImportError as exc:  # pragma: no cover - exercised in tests via patching
            raise ImportError(_AOMPLS_IMPORT_HINT) from exc

        cv_mode = _selection_to_cv_mode(self.selection)
        preproc = "none" if self.preprocessing is None else str(self.preprocessing)
        random_state = 0 if self.random_state is None else int(self.random_state)

        return AOMPLSCompact(
            max_components=int(self.n_components),
            n_folds=int(self.cv),
            cv_mode=cv_mode,
            one_se_rule=bool(self.one_se),
            random_state=random_state,
            preproc=preproc,
            osc_n_components=int(self.osc_n_components),
            asls_lam=float(self.asls_lam),
            asls_p=float(self.asls_p),
            asls_n_iter=int(self.asls_n_iter),
            center=bool(self.center),
            external_folds=self.external_folds,
        )

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_val: ArrayLike | None = None,  # noqa: ARG002 - kept for API symmetry
        y_val: ArrayLike | None = None,  # noqa: ARG002 - kept for API symmetry
    ) -> AOMPLSAomlibRegressor:
        """Fit AOM-PLS via the AOM_lib C++ backend.

        Args:
            X: Training spectra of shape ``(n_samples, n_features)``.
            y: Target values of shape ``(n_samples,)``. Multivariate ``y`` is
                not supported by the compact PLS1 backend and is reshaped to
                1D after squeezing trailing singleton dimensions.
            X_val: Unused. Kept for API symmetry with other nirs4all wrappers
                that accept an optional validation set; the AOM_lib backend
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

        backend = self._make_backend()
        backend.fit(X_arr, y_arr)

        self._backend = backend
        self.n_features_in_ = X_arr.shape[1]
        self.coef_ = np.asarray(backend.coef_, dtype=np.float64)
        self.intercept_ = float(backend.intercept_)
        self.x_mean_ = np.asarray(backend.x_mean_, dtype=np.float64)
        self.y_mean_ = float(backend.y_mean_)
        self.bank_names_ = list(backend.bank_names_)
        self.fold_indices_ = [list(map(int, f)) for f in backend.fold_indices_]
        self.fit_time_s_ = float(backend.fit_time_s_)
        self.one_se_applied_ = bool(backend.one_se_applied_)

        # Diagnostics mirroring the legacy AOMPLSRegressor / POPPLSRegressor wrappers.
        self.n_components_selected_ = int(backend.n_components_)
        selected_name = str(backend.selected_operator_name_)
        self.selected_operator_index_ = int(backend.selected_operator_index_)
        self.selected_operator_sequence_ = [selected_name]
        self.selected_operator_scores_ = np.asarray(
            backend.rmse_curves_, dtype=np.float64
        )

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
