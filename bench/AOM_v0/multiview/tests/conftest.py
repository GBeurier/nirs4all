"""Shared fixtures for AOM-multiview tests.

Adds ``bench/AOM_v0`` and ``bench/AOM_v0/multiview`` to ``sys.path`` so that
the ``aompls`` and ``multiview`` packages can be imported when pytest is
launched from the repository root.

Order matters: ``_MULTIVIEW_ROOT`` must come *before* ``_AOM_ROOT`` in
``sys.path`` because ``bench/AOM_v0/multiview/`` has its own outer
``__init__.py`` (added 2026-05-07 for dispatch-level package resolution),
which would otherwise shadow the inner ``multiview`` package and break
``from multiview.<sub> import ...`` test imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_MULTIVIEW_ROOT = _HERE.parent.parent           # bench/AOM_v0/multiview
_AOM_ROOT = _MULTIVIEW_ROOT.parent              # bench/AOM_v0
# Insert _AOM_ROOT first, then _MULTIVIEW_ROOT — the latter ends up at
# sys.path[0] so ``import multiview`` resolves to the inner package.
for path in (_AOM_ROOT, _MULTIVIEW_ROOT):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)
