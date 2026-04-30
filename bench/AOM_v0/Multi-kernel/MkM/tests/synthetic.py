"""Synthetic data generators for MKM tests.

Re-uses the consolidated MKR synthetic generators (``synthetic_mkr.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure consolidated MKR tests are on path so we can import synthetic_mkr.
_MKR_TESTS = Path(__file__).resolve().parents[2] / "MKR" / "tests"
if str(_MKR_TESTS) not in sys.path:
    sys.path.insert(0, str(_MKR_TESTS))

from synthetic_mkr import SyntheticDataset, make_R1, make_R2, make_R3  # noqa: E402

__all__ = ["SyntheticDataset", "make_R1", "make_R2", "make_R3"]
