"""POP-PLS public entry point for `nirs4all`.

Re-exports the canonical implementation from
``nirs4all.operators.models._aom_nirs.pls`` (vendored copy of the
``aom-nirs`` package). POP-PLS is a *per-component* AOM-PLS variant:
each PLS component is allowed to pick a different preprocessing
operator from the bank, selected via approximate PRESS.

In the Talanta paper POP-PLS is reported as a negative ablation
(median RMSEP ratio ~1.37 vs PLS-default); it is kept here for
reproducibility of the supplement and for users who want to explore
per-component selection.
"""

from __future__ import annotations

from nirs4all.operators.models._aom_nirs.pls.estimators import POPPLSRegressor

__all__ = ["POPPLSRegressor"]
