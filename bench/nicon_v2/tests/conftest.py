"""Pytest setup for nicon_v2 — adds the bench package to sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS = Path(__file__).resolve()
_BENCH_ROOT = _TESTS.parent.parent  # bench/nicon_v2
for path in (_BENCH_ROOT, _BENCH_ROOT.parent):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)
