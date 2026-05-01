"""AOM-multiview package.

Multi-view extension of `bench/AOM_v0/aompls/`: wavelength-block masks,
preproc × block view banks, and (Phase 2+) AOM-MBPLS / AOM-MoE selection
policies. See `bench/AOM_v0/multiview/docs/DESIGN_VIEWS.md` for the design
rationale and Codex review disposition.
"""

from .views import BlockMaskOperator, ViewBuilder

__all__ = ["BlockMaskOperator", "ViewBuilder"]
