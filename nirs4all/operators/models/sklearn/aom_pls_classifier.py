"""AOM-PLS-DA classifier — re-export of the canonical aom-nirs implementation.

Once ``aom-nirs`` is on PyPI this file will switch to
``from aom_nirs.pls import AOMPLSDAClassifier as AOMPLSClassifier``.
"""

from __future__ import annotations

from nirs4all.operators.models._aom_nirs.pls.classification import (
    AOMPLSDAClassifier as AOMPLSClassifier,
)

__all__ = ["AOMPLSClassifier"]
