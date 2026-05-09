"""AOM-multiview package.

Multi-view extension of `bench/AOM_v0/aompls/`: wavelength-block masks,
preproc × block view banks, and (Phase 2+) AOM-MBPLS / AOM-MoE selection
policies. See `bench/AOM_v0/multiview/docs/DESIGN_VIEWS.md` for the design
rationale and Codex review disposition.

The canonical Phase-11 atoms (``LazyV2AOM``, ``AOMMoEMultiK``,
``AOMMoERegressor``, ``AOMPLSRegressor``) are re-exported through
:mod:`.atoms` so they can be imported as ``from multiview import LazyV2AOM``.
"""

from .atoms import AOMMoEMultiK, AOMMoERegressor, AOMPLSRegressor, LazyV2AOM
from .views import BlockMaskOperator, ViewBuilder

__all__ = [
    "AOMMoEMultiK",
    "AOMMoERegressor",
    "AOMPLSRegressor",
    "BlockMaskOperator",
    "LazyV2AOM",
    "ViewBuilder",
]
