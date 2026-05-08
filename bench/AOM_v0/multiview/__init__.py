"""Outer namespace for `bench.AOM_v0.multiview`.

Added 2026-05-07 by Agent A as the first half of the D-A-002 triage
agreed under Codex round 6: the registry entry for
``AdaptiveSuperLearner-bigN-guarded`` (and ``-recipe-nnls``) declared
``module: bench.AOM_v0.multiview.adaptive_super_learner``, which (a)
points at a file that does not exist and (b) cannot resolve at all
because this directory was missing an ``__init__.py``. The actual class
is defined at ``bench/AOM_v0/multiview/multiview/super_learner.py``,
so the corrected path is ``bench.AOM_v0.multiview.multiview.super_learner``.

Without this file, all 36 fits of ``AdaptiveSuperLearner-bigN-guarded``
in the D-A-001 partial run failed at import (``ModuleNotFoundError:
No module named 'bench.AOM_v0.multiview.adaptive_super_learner'``).
The registry/yaml side of the fix is owned by Agent C and is proposed
in the corresponding SYNC.md entry.
"""
