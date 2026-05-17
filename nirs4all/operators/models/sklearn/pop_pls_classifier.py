"""POP-PLS-DA classifier — re-export of the canonical aom-nirs implementation."""

from __future__ import annotations

from nirs4all.operators.models._aom_nirs.pls.classification import (
    POPPLSDAClassifier as POPPLSClassifier,
)

__all__ = ["POPPLSClassifier"]
