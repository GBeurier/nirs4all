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

    Each ``transform`` call draws with the next seed (base seed + call
    index), so repeated augmentations of the same rows differ while a run
    stays reproducible. The wavelengths of the dataset are passed as the
    role's axis when the method uses one.

    Args:
        role: The n4m role instance (for example ``n4m.roles.GaussianNoise()``).
        random_state: Base seed; defaults to the role's ``seed`` parameter.
    """

    def __init__(self, role: Any, random_state: int | None = None):
        self.role = role
        self.random_state = random_state

    @property
    def _requires_wavelengths(self) -> bool | str:  # type: ignore[override]
        return _AXIS_NEED[self.role.input_requirements()["axis"]]

    def fit(self, X, y=None, **kwargs):
        super().fit(X, y, **kwargs)
        self._calls = 0
        return self

    def _transform_impl(self, X: np.ndarray, wavelengths) -> np.ndarray:
        role = self.role
        params = role.get_params()
        if "seed" in params:
            base = self.random_state if self.random_state is not None else params["seed"]
            role = clone(role).set_params(seed=int(base) + self._calls)
        self._calls += 1
        axis = wavelengths if self._requires_wavelengths else None
        out: np.ndarray = role.augment(np.asarray(X, dtype=np.float64), axis=axis)
        return out


def as_augmenter(obj: Any) -> Any:
    """``obj``, or an n4m augmenter role wrapped as a sample augmenter."""
    # A role instance implies n4m.roles is loaded; nothing to import otherwise.
    roles = sys.modules.get("n4m.roles")
    if roles is not None and isinstance(obj, roles.NativeAugmenter):
        return NativeRoleAugmenter(obj)
    return obj
