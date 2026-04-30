"""Shared fixtures for AOM-Ridge tests.

Adds ``bench/aom_v0/Multi-kernel`` and ``bench/aom_v0/Multi-kernel/MKR`` to ``sys.path`` so that the
``aompls`` and ``aomridge`` packages can be imported when pytest is launched
from the repository root without setting ``PYTHONPATH``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_RIDGE_ROOT = _HERE.parent.parent           # bench/aom_v0/Multi-kernel/MKR
_AOM_ROOT = _RIDGE_ROOT.parent              # bench/aom_v0/Multi-kernel
for path in (_RIDGE_ROOT, _AOM_ROOT):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)
