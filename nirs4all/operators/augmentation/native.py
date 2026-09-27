"""n4m augmenter roles as nirs4all sample augmenters.

``n4m.roles`` exposes the native augmentations through one role interface,
``NativeAugmenter`` (``augment(X, axis=None)``, seeded by a parameter).
``sample_augmentation`` accepts those roles directly; :func:`as_augmenter`
gives them the transformer face that step uses.
"""

import sys
from typing import Any

import numpy as np
from sklearn.base import clone

from nirs4all.operators.base.spectra_mixin import SpectraTransformerMixin

# Manifest axis requirement -> SpectraTransformerMixin._requires_wavelengths.
_AXIS_NEED: dict[str, bool | str] = {"required": True, "optional": "optional", "none": False}


class NativeRoleAugmenter(SpectraTransformerMixin):
    """An ``n4m.roles`` augmenter, run natively.

    Seed priority: an explicit ``random_state`` is the base seed; otherwise
    the role's own ``seed`` when the caller set it; otherwise 0, the native
    default. :func:`as_augmenter` copies an explicit role seed into
    ``random_state``, so ``sample_augmentation`` keeps it and derives a base
    seed from its step ``random_state`` only for roles without one. Each
    ``transform`` call draws with base seed + call index, so repeated
    augmentations of the same rows differ while a run stays reproducible; the
    seeds used are recorded in ``seeds_``. The wavelengths of the dataset are
    passed as the role's axis when the method uses one.

    Args:
        role: The n4m role instance (for example ``n4m.roles.GaussianNoise()``).
        random_state: Base seed; None falls back to the role's ``seed``.
    """

    def __init__(self, role: Any, random_state: int | None = None):
        self.role = role
        self.random_state = random_state

    @property
    def _requires_wavelengths(self) -> bool | str:  # type: ignore[override]
        return _AXIS_NEED[self.role.input_requirements()["axis"]]

    @property
    def base_seed(self) -> int | None:
        """Effective base seed (None when the role takes no seed)."""
        params = self.role.get_params()
        if "seed" not in params:
            return None
        for seed in (self.random_state, params["seed"]):
            if seed is not None:
                return int(seed)
        return 0

    def fit(self, X, y=None, **kwargs):
        super().fit(X, y, **kwargs)
        self._calls = 0
        self.seeds_: list[int] = []
        return self

    def _transform_impl(self, X: np.ndarray, wavelengths) -> np.ndarray:
        role = self.role
        base = self.base_seed
        if base is not None:
            role = clone(role).set_params(seed=base + self._calls)
            self.seeds_.append(base + self._calls)
        self._calls += 1
        axis = wavelengths if self._requires_wavelengths else None
        out: np.ndarray = role.augment(np.asarray(X, dtype=np.float64), axis=axis)
        return out


def as_augmenter(obj: Any) -> Any:
    """``obj``, or an n4m augmenter role wrapped as a sample augmenter.

    An explicit role seed becomes the adapter's ``random_state``, so the
    ``sample_augmentation`` step seed only reaches roles without one.
    """
    # A role instance implies n4m.roles is loaded; nothing to import otherwise.
    roles = sys.modules.get("n4m.roles")
    if roles is not None and isinstance(obj, roles.NativeAugmenter):
        obj = NativeRoleAugmenter(obj)
    if isinstance(obj, NativeRoleAugmenter) and obj.random_state is None and obj.role.get_params().get("seed") is not None:
        obj = clone(obj).set_params(random_state=obj.role.get_params()["seed"])
    return obj
