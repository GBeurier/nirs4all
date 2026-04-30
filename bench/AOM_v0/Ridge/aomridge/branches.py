"""Branch preprocessing factory for AOM-Ridge.

Branches are sklearn-style transformers applied *before* feature centering
and the AOM dual Ridge. Most are non-linear (per-sample normalisation), so
their parameters (e.g. the MSC reference spectrum, the EMSC mean spectrum,
or the OSC orthogonal projection) must be fitted on the training fold only
- never on the union of train + validation rows.

Two flavours of branches are supported:

- *Stateless* branches (e.g. ``"snv"``, ``"asls"``): every spectrum is
  normalised in isolation. ``fit_transform`` and ``transform`` produce the
  same result, no training-side state leaks.
- *Fitted* branches (``"msc"``, ``"emsc1"``, ``"emsc2"``, all OSC variants
  and any pipeline containing them): training-fold means or supervised
  projections are stored at fit time and replayed at transform time. The
  fold-local CV path must therefore call ``fit_transform`` on the training
  rows and ``transform`` on the validation rows.

The strict-linear AOM operator bank stays the same; what changes is that
the fold-local CV path can now wrap the bank inside a per-branch
``fit_transform`` step that the validation rows go through unchanged.
"""

from __future__ import annotations

import numpy as np

# Import the existing standalone preprocessors from aompls.
from aompls.preprocessing import (
    ASLSBaseline,
    ExtendedMSC,
    MultiplicativeScatterCorrection,
    OrthogonalSignalCorrection,
    PreprocessingPipeline,
    StandardNormalVariate,
)

BRANCH_NONE = "none"
BRANCH_SNV = "snv"
BRANCH_MSC = "msc"
BRANCH_OSC = "osc"
BRANCH_OSC1 = "osc1"
BRANCH_OSC2 = "osc2"
BRANCH_OSC3 = "osc3"
BRANCH_EMSC1 = "emsc1"
BRANCH_EMSC2 = "emsc2"
BRANCH_ASLS = "asls"
BRANCH_ASLS_SOFT = "asls_soft"
BRANCH_ASLS_MEDIUM = "asls_medium"
BRANCH_ASLS_HARD = "asls_hard"
BRANCH_SNV_OSC = "snv_osc"
BRANCH_MSC_OSC = "msc_osc"
BRANCH_SNV_ASLS = "snv_asls"
BRANCH_MSC_ASLS = "msc_asls"

VALID_BRANCHES = (
    BRANCH_NONE,
    BRANCH_SNV,
    BRANCH_MSC,
    BRANCH_OSC,
    BRANCH_OSC1,
    BRANCH_OSC2,
    BRANCH_OSC3,
    BRANCH_EMSC1,
    BRANCH_EMSC2,
    BRANCH_ASLS,
    BRANCH_ASLS_SOFT,
    BRANCH_ASLS_MEDIUM,
    BRANCH_ASLS_HARD,
    BRANCH_SNV_OSC,
    BRANCH_MSC_OSC,
    BRANCH_SNV_ASLS,
    BRANCH_MSC_ASLS,
)

# Stateless branches do not store any training-time parameters; ``fit_transform``
# and ``transform`` behave identically. Only SNV qualifies — even ASLS, which
# computes a fresh per-sample baseline and has no training-time state, is
# implemented via a class with a non-trivial ``fit`` we still want to call
# explicitly on the training rows for clarity.
_STATELESS = frozenset({BRANCH_SNV})


def make_branch_preproc(branch: str):
    """Return a fresh sklearn-style transformer for ``branch``.

    ``"none"`` returns ``None`` (the caller must skip the transform step).

    For all other branches this returns a fresh transformer instance with
    the standard ``fit(X[, y]) / transform(X) / fit_transform(X[, y])``
    interface so the fold-local CV path can fit on training rows only.
    """
    if branch == BRANCH_NONE:
        return None
    if branch == BRANCH_SNV:
        return StandardNormalVariate()
    if branch == BRANCH_MSC:
        return MultiplicativeScatterCorrection()
    if branch in (BRANCH_OSC, BRANCH_OSC2):
        return OrthogonalSignalCorrection(n_components=2)
    if branch == BRANCH_OSC1:
        return OrthogonalSignalCorrection(n_components=1)
    if branch == BRANCH_OSC3:
        return OrthogonalSignalCorrection(n_components=3)
    if branch == BRANCH_EMSC1:
        return ExtendedMSC(degree=1)
    if branch == BRANCH_EMSC2:
        return ExtendedMSC(degree=2)
    if branch == BRANCH_ASLS:
        return ASLSBaseline()
    if branch == BRANCH_ASLS_SOFT:
        return ASLSBaseline(lam=1e4)
    if branch == BRANCH_ASLS_MEDIUM:
        return ASLSBaseline(lam=1e6)
    if branch == BRANCH_ASLS_HARD:
        return ASLSBaseline(lam=1e8)
    if branch == BRANCH_SNV_OSC:
        return PreprocessingPipeline(
            [StandardNormalVariate(), OrthogonalSignalCorrection(n_components=2)]
        )
    if branch == BRANCH_MSC_OSC:
        return PreprocessingPipeline(
            [MultiplicativeScatterCorrection(), OrthogonalSignalCorrection(n_components=2)]
        )
    if branch == BRANCH_SNV_ASLS:
        return PreprocessingPipeline([StandardNormalVariate(), ASLSBaseline()])
    if branch == BRANCH_MSC_ASLS:
        return PreprocessingPipeline([MultiplicativeScatterCorrection(), ASLSBaseline()])
    raise ValueError(f"unknown branch {branch!r}; expected one of {VALID_BRANCHES}")


def is_stateless(branch: str) -> bool:
    """Return ``True`` only for branches whose ``transform`` ignores training data.

    Stateless branches normalise each sample independently and do not need
    fold-local refitting for correctness. They are still wrapped in
    ``fit_transform`` so the CV path treats them uniformly.
    """
    return branch in _STATELESS


def fit_transform_branch(preproc, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Call ``preproc.fit_transform(X[, y])``, falling back to ``fit_transform(X)``.

    Supervised branches such as ``OrthogonalSignalCorrection`` require ``y``
    at fit time; unsupervised ones (SNV, MSC, EMSC, ASLS) ignore ``y``. The
    aompls ``PreprocessingPipeline`` tries both signatures internally, but
    direct estimators raise ``TypeError`` if called with an unexpected ``y``.
    This helper centralises the dual-signature handling.
    """
    if preproc is None:
        return np.asarray(X, dtype=float)
    Xa = np.asarray(X, dtype=float)
    if y is None:
        try:
            return np.asarray(preproc.fit_transform(Xa), dtype=float)
        except TypeError:
            return np.asarray(preproc.fit_transform(Xa, None), dtype=float)
    try:
        return np.asarray(preproc.fit_transform(Xa, y), dtype=float)
    except TypeError:
        return np.asarray(preproc.fit_transform(Xa), dtype=float)


def apply_branch_train(
    branch: str, X_tr: np.ndarray, y_tr: np.ndarray | None = None,
):
    """Fit a fresh branch transformer on ``X_tr`` and return ``(preproc, X_tr_proc)``.

    For ``"none"`` returns ``(None, X_tr)`` unchanged. ``y_tr`` is forwarded
    to the underlying estimator when provided so supervised branches
    (e.g. OSC) can fit their projection on the training fold.
    """
    preproc = make_branch_preproc(branch)
    if preproc is None:
        return None, np.asarray(X_tr, dtype=float)
    X_proc = fit_transform_branch(preproc, X_tr, y_tr)
    return preproc, X_proc


def apply_branch_transform(preproc, X: np.ndarray) -> np.ndarray:
    """Apply an already-fitted branch transformer to ``X``.

    Passes ``X`` through unchanged when ``preproc is None`` (branch="none").
    """
    if preproc is None:
        return np.asarray(X, dtype=float)
    return np.asarray(preproc.transform(np.asarray(X, dtype=float)), dtype=float)
