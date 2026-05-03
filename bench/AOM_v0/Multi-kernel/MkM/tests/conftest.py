"""Shared fixtures for AOM-MKM tests.

Adds ``bench/aom_v0/Multi-kernel``, ``bench/aom_v0/Multi-kernel/MKR``,
``bench/aom_v0/Multi-kernel/MkM``, and the preserved Ridge fallback to
``sys.path`` so that the ``aompls``, ``aomridge``, and ``mkm`` packages can be
imported when pytest is launched from the repository root without setting
``PYTHONPATH``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_MKM_ROOT = _HERE.parent.parent              # bench/AOM_v0/Multi-kernel/MkM
_AOM_ROOT = _MKM_ROOT.parent                 # bench/AOM_v0/Multi-kernel
_MKR_ROOT = _AOM_ROOT / "MKR"

# IMPORTANT: do NOT add bench/AOM_v0/Ridge/ — its `aomridge/` package is
# the older copy without top_k_active. The Multi-kernel/MKR/aomridge/
# is the canonical source.
for path in (_MKR_ROOT, _MKM_ROOT, _AOM_ROOT):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)
