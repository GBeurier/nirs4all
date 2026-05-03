"""Shared fixtures for AOM-BLUP tests.

Adds ``bench/aom_v0/Multi-kernel``, ``bench/aom_v0/Multi-kernel/MKR``,
``bench/aom_v0/Multi-kernel/MkM``, ``bench/aom_v0/Multi-kernel/Blup``, and the
preserved Ridge fallback to ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_BLUP_ROOT = _HERE.parent.parent             # bench/AOM_v0/Multi-kernel/Blup
_AOM_ROOT = _BLUP_ROOT.parent                # bench/AOM_v0/Multi-kernel
_MKR_ROOT = _AOM_ROOT / "MKR"
_MKM_ROOT = _AOM_ROOT / "MkM"

# Do NOT add bench/AOM_v0/Ridge/ — see MkM/tests/conftest.py for rationale.
for path in (_MKR_ROOT, _MKM_ROOT, _BLUP_ROOT, _AOM_ROOT):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)
