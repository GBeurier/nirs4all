"""Repeated cross-validation helpers for AOM-Ridge.

Provides ``RepeatedSPXYFold``: a thin wrapper that runs an SPXY-based fold
generator (``nirs4all.operators.splitters.SPXYFold`` when available, falling
back to ``sklearn.model_selection.RepeatedKFold``) ``n_repeats`` times with
distinct ``random_state`` values to stabilise selection on small / sensitive
datasets such as AMYLOSE.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold


def _try_import_spxyfold():
    try:
        from nirs4all.operators.splitters import SPXYFold
    except Exception:
        return None
    return SPXYFold


class RepeatedSPXYFold:
    """Run ``SPXYFold`` (or ``KFold``) ``n_repeats`` times with varying seeds.

    Falls back to ``sklearn.model_selection.KFold(shuffle=True)`` when nirs4all
    is not importable in the current environment.

    Parameters
    ----------
    n_splits : int
        Number of folds per repeat (default 3).
    n_repeats : int
        Number of independent SPXY runs (default 3). Each run uses a distinct
        ``random_state`` derived from the constructor seed.
    random_state : int
        Seed for the per-repeat random states.
    """

    def __init__(
        self, n_splits: int = 3, n_repeats: int = 3, random_state: int | None = 0
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if n_repeats < 1:
            raise ValueError("n_repeats must be >= 1")
        self.n_splits = int(n_splits)
        self.n_repeats = int(n_repeats)
        self.random_state = random_state
        self._spxy_cls = _try_import_spxyfold()

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits * self.n_repeats

    def split(self, X, y=None, groups=None):
        # ``SPXYFold`` is a deterministic distance-based splitter: changing
        # ``random_state`` only affects the optional PCA step (no PCA by
        # default), so feeding the unmodified ``(X, y)`` n_repeats times
        # returns identical folds every time. To produce genuinely different
        # SPXY-flavoured folds we (a) permute the row order per-repeat with
        # a seeded RNG, and (b) add a tiny per-repeat Gaussian jitter to the
        # *copy* of X used by the splitter so the distance matrix differs
        # slightly across repeats. The jitter is sized to the per-feature
        # standard deviation scaled by 1e-3, which is small enough that the
        # SPXY geometry is preserved (the same broad partition emerges) but
        # large enough to break ties / shift borderline assignments. The
        # resulting (train, valid) indices are mapped back to the original
        # row positions of ``X`` so callers see consistent indexing.
        X_arr = np.asarray(X, dtype=float)
        y_arr = None if y is None else np.asarray(y)
        n_rows = X_arr.shape[0]
        # Per-feature scale used to size the jitter; falls back to 1.0 for
        # constant columns to avoid zero noise. SPXY's fold assignment is
        # purely distance-based and therefore robust to small perturbations:
        # we use a jitter on the order of 1% of the per-feature std so that
        # borderline assignments shift across repeats while the broad SPXY
        # geometry is preserved.
        col_std = np.std(X_arr, axis=0, ddof=0)
        col_std = np.where(col_std > 0.0, col_std, 1.0)
        if y_arr is not None and y_arr.ndim == 1:
            y_std = float(np.std(y_arr, ddof=0)) or 1.0
        elif y_arr is not None:
            y_std = np.std(y_arr, axis=0, ddof=0)
            y_std = np.where(y_std > 0.0, y_std, 1.0)
        else:
            y_std = None
        jitter_scale = 1e-2
        for repeat_idx in range(self.n_repeats):
            base = 0 if self.random_state is None else int(self.random_state)
            rng = np.random.default_rng(base + repeat_idx)
            seed = int(rng.integers(0, 2**31 - 1))
            perm = rng.permutation(n_rows)
            jitter_X = jitter_scale * col_std * rng.standard_normal(X_arr.shape)
            X_perturbed = X_arr + jitter_X
            X_perm = X_perturbed[perm]
            if y_arr is None:
                y_perm = None
            else:
                jitter_y = jitter_scale * y_std * rng.standard_normal(y_arr.shape)
                y_perm = (y_arr + jitter_y)[perm]
            if self._spxy_cls is not None:
                splitter = self._spxy_cls(n_splits=self.n_splits, random_state=seed)
            else:
                splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
            for tr, va in splitter.split(X_perm, y_perm):
                yield perm[np.asarray(tr)], perm[np.asarray(va)]
