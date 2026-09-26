"""n4m sample-filter roles as nirs4all sample filters.

``n4m.roles`` exposes every native sample filter (Y/X outliers, leverage,
spectral quality) through one role interface, ``NativeSampleFilter``
(``fit`` / ``get_mask``). The exclude, tag and branch controllers accept
those roles directly; :func:`as_sample_filter` gives them the
:class:`SampleFilter` face those controllers use.
"""

import sys
from typing import Any

import numpy as np

from .base import SampleFilter


class NativeRoleFilter(SampleFilter):
    """An ``n4m.roles`` sample filter, fitted and applied natively.

    Args:
        role: The n4m role instance (for example ``n4m.roles.YOutlierFilter()``).
        reason: Exclusion reason; defaults to the role's class name.
        tag_name: Tag name for the tag controller.
    """

    def __init__(self, role: Any, reason: str | None = None, tag_name: str | None = None):
        super().__init__(reason=reason, tag_name=tag_name)
        self.role = role

    @property
    def exclusion_reason(self) -> str:
        return self.reason if self.reason is not None else type(self.role).__name__

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "NativeRoleFilter":
        self.role.fit(X, y)
        return self

    def get_mask(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        mask: np.ndarray = self.role.get_mask(X, y)
        return mask


def as_sample_filter(obj: Any) -> SampleFilter | None:
    """``obj`` as a SampleFilter: itself, an n4m sample-filter role wrapped, else None."""
    if isinstance(obj, SampleFilter):
        return obj
    # A role instance implies n4m.roles is loaded; nothing to import otherwise.
    roles = sys.modules.get("n4m.roles")
    if roles is not None and isinstance(obj, roles.NativeSampleFilter):
        return NativeRoleFilter(obj)
    return None
