"""Split-aware inner CV for AOM-Ridge.

Adapts the inner CV splitter to the *test split protocol* of the dataset so
the alpha selected on training folds corresponds to a fold geometry that
mirrors the eventual train/test partition. Core rationale: the default
``SPXYFold`` blends training rows by joint X/Y distance, but on Y-extrapolation
holdouts (e.g. the AMYLOSE / BEER hard splits) the test set sits *outside*
the training Y range. A CV that pretends to interpolate ranks variants on
the wrong inductive task. ``YBlockedKFold`` rotates Y-strata as test folds,
mimicking that extrapolation. ``GroupKFold`` is used when the dataset name
encodes a grouped holdout.

The detector is purely heuristic: dataset names from the AOM_v0 cohort
already encode the protocol (``*_YbasedSplit``, ``*_byCultivar_*``, ``*_KS``
etc.). When the name has no signal we fall back to SPXY which is the
project's default NIRS-aware inner CV.

Public surface:

- ``detect_split_kind(dataset_name, X_train, y_train, **metadata)``
- ``make_inner_cv(split_kind, n_splits, n_repeats, random_state, groups)``
- ``YBlockedKFold(n_splits, random_state)``

The Y-blocked splitter never observes test labels: at fit time it sees only
the *training* y vector and partitions it into quantile bins. This is
leakage-safe by construction.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import GroupKFold

# ----------------------------------------------------------------------
# Heuristic dataset-name detectors
# ----------------------------------------------------------------------

_Y_BASED_RE = re.compile(r"y[ _-]?based?[ _-]?split", re.IGNORECASE)
_GROUPED_PATTERNS = (
    re.compile(r"groupSampleID", re.IGNORECASE),
    re.compile(r"byCultivar", re.IGNORECASE),
    re.compile(r"byManure", re.IGNORECASE),
    re.compile(r"byspecies", re.IGNORECASE),
    re.compile(r"bySite", re.IGNORECASE),
    re.compile(r"_grp[0-9]", re.IGNORECASE),
)


def detect_split_kind(
    dataset_name: str,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    **metadata: object,
) -> str:
    """Detect the split protocol encoded in ``dataset_name``.

    Returns one of ``"y_based"``, ``"grouped"``, or ``"spxy"`` (default).

    The detector first tries dataset-name regex matching, then falls back to
    metadata cues (``groups`` keyword forces ``"grouped"``). Test-time labels
    are never read.
    """
    if dataset_name is None:
        dataset_name = ""
    name = str(dataset_name)
    if _Y_BASED_RE.search(name):
        return "y_based"
    for pat in _GROUPED_PATTERNS:
        if pat.search(name):
            return "grouped"
    if metadata.get("groups") is not None:
        return "grouped"
    return "spxy"


# ----------------------------------------------------------------------
# Y-Blocked K-Fold splitter
# ----------------------------------------------------------------------


class YBlockedKFold:
    """Rotate Y-quantile blocks as validation folds.

    Suited for Y-extrapolation holdouts: each validation fold contains rows
    whose y values fall in one quantile band, and training contains all
    other rows. This forces the inner CV to evaluate ranks under the same
    extrapolation regime as the dataset's outer test split.

    Parameters
    ----------
    n_splits : int, default=3
        Number of Y-quantile bands (and folds). Must be >= 2.
    random_state : int or None
        Tie-breaker seed used when y has duplicate values; the splitter is
        otherwise deterministic.

    Notes
    -----
    The split is leakage-safe: every sample belongs to exactly one validation
    fold, train and validation indices are disjoint, and the bins are derived
    from the *training* y vector that ``split`` is called with.
    """

    def __init__(self, n_splits: int = 3, random_state: int | None = None) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        self.n_splits = int(n_splits)
        self.random_state = random_state

    def get_n_splits(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if y is None:
            raise ValueError("YBlockedKFold requires y to bin samples by quantile")
        y_arr = np.asarray(y).ravel().astype(float)
        n = y_arr.shape[0]
        if X is not None and np.asarray(X).shape[0] != n:
            raise ValueError("X and y must have the same number of rows")
        if self.n_splits > n:
            raise ValueError(
                f"n_splits={self.n_splits} exceeds n_samples={n}"
            )

        rng = np.random.default_rng(self.random_state)
        # Stable jitter for tie-breaking: keeps assignment deterministic for
        # a given random_state without changing the underlying y ordering.
        jitter = rng.uniform(-1e-12, 1e-12, size=n)
        order = np.argsort(y_arr + jitter, kind="stable")

        # Equal-size bins by quantile rank: bin index = (rank * n_splits) // n.
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(n)
        bins = (ranks * self.n_splits) // n  # in [0, n_splits)

        for fold_idx in range(self.n_splits):
            valid_mask = bins == fold_idx
            valid_idx = np.where(valid_mask)[0]
            train_idx = np.where(~valid_mask)[0]
            yield train_idx, valid_idx


# ----------------------------------------------------------------------
# Inner CV factory
# ----------------------------------------------------------------------


def make_inner_cv(
    split_kind: str,
    n_splits: int = 3,
    n_repeats: int = 1,
    random_state: int | None = 0,
    groups: np.ndarray | None = None,
) -> object:
    """Build an inner CV splitter matched to ``split_kind``.

    - ``"y_based"`` -> ``YBlockedKFold(n_splits, random_state)``
    - ``"grouped"`` -> ``sklearn.model_selection.GroupKFold(n_splits)`` if
      ``groups`` is provided; otherwise falls back to SPXY.
    - ``"spxy"`` -> ``SPXYFold`` (or repeated equivalent if ``n_repeats > 1``).

    ``n_repeats`` is reserved for SPXY; raises if combined with ``"y_based"``
    or ``"grouped"`` since those splitters are deterministic given the data.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    if split_kind == "y_based":
        if n_repeats != 1:
            raise ValueError("YBlockedKFold does not support n_repeats > 1")
        return YBlockedKFold(n_splits=n_splits, random_state=random_state)

    if split_kind == "grouped":
        if groups is not None:
            if n_repeats != 1:
                raise ValueError("GroupKFold does not support n_repeats > 1")
            cv = GroupKFold(n_splits=n_splits)
            # Bind groups so the splitter can be passed as a plain ``cv``
            # without callers re-supplying them inside scoring loops.
            return _GroupKFoldWithGroups(cv, np.asarray(groups))
        # fallthrough to SPXY when groups are unavailable
        split_kind = "spxy"

    if split_kind == "spxy":
        from nirs4all.operators.splitters import SPXYFold

        if n_repeats == 1:
            return SPXYFold(n_splits=n_splits, random_state=random_state)
        return _RepeatedSPXYFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    raise ValueError(
        f"unknown split_kind {split_kind!r}; expected 'y_based', 'grouped', or 'spxy'"
    )


class _GroupKFoldWithGroups:
    """Adapter that pre-binds ``groups`` to ``GroupKFold.split``."""

    def __init__(self, splitter: GroupKFold, groups: np.ndarray) -> None:
        self._splitter = splitter
        self._groups = groups
        self.n_splits = splitter.n_splits

    def get_n_splits(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> int:
        return int(
            self._splitter.get_n_splits(
                X, y, groups if groups is not None else self._groups
            )
        )

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        g = groups if groups is not None else self._groups
        if g.shape[0] != np.asarray(X).shape[0]:
            raise ValueError(
                "GroupKFold groups length must match number of samples"
            )
        for tr, va in self._splitter.split(X, y, g):
            yield np.asarray(tr), np.asarray(va)


class _RepeatedSPXYFold:
    """Yield SPXY folds with multiple seeds; only used when n_repeats > 1."""

    def __init__(self, n_splits: int, n_repeats: int, random_state: int | None) -> None:
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def get_n_splits(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> int:
        return self.n_splits * self.n_repeats

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        from nirs4all.operators.splitters import SPXYFold

        base = 0 if self.random_state is None else int(self.random_state)
        for r in range(self.n_repeats):
            cv = SPXYFold(n_splits=self.n_splits, random_state=base + r * 1000)
            yield from cv.split(X, y, groups)
