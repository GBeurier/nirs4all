"""Shared fixtures and ``sys.path`` setup for FastAOM tests.

Adds ``bench/AOM_v0`` to ``sys.path`` so ``aompls`` and ``FastAOM`` are
importable as top-level packages when pytest is invoked from the repository
root without explicit ``PYTHONPATH`` configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_FASTAOM_ROOT = _HERE.parent.parent  # bench/AOM_v0/FastAOM
_AOM_ROOT = _FASTAOM_ROOT.parent     # bench/AOM_v0
for path in (_FASTAOM_ROOT, _AOM_ROOT):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)
