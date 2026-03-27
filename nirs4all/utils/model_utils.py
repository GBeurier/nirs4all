"""Backward-compatibility shim for nirs4all.utils.model_utils.

This module was split into:
- nirs4all.controllers.models.utilities (ModelControllerUtils)
- nirs4all.data.ensemble_utils (EnsembleUtils)

This shim prevents ModuleNotFoundError from stale .pyc bytecode caches
that still reference the old import path.
"""

import warnings

warnings.warn(
    "nirs4all.utils.model_utils is deprecated. "
    "Use nirs4all.controllers.models.utilities or nirs4all.data.ensemble_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

from nirs4all.controllers.models.utilities import ModelControllerUtils as ModelUtils  # noqa: F401, E402
from nirs4all.data.ensemble_utils import EnsembleUtils  # noqa: F401, E402
