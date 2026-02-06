"""Refit parameter resolution utilities.

Provides :func:`resolve_refit_params` which merges ``refit_params`` on
top of ``train_params`` for the refit execution phase.  Parameters not
specified in ``refit_params`` inherit from ``train_params``.
"""

from __future__ import annotations

from typing import Any


def resolve_refit_params(model_config: dict[str, Any]) -> dict[str, Any]:
    """Merge ``refit_params`` on top of ``train_params``.

    Produces the effective training parameters for the refit phase.
    ``refit_params`` values override ``train_params`` on conflicts;
    unspecified parameters inherit from ``train_params``.

    Special ``refit_params`` keys (stored but not acted on until Phase 4):
        - ``warm_start`` (bool): Whether to warm-start from CV weights.
        - ``warm_start_fold`` (str): Which fold to warm-start from
          (``"best"``, ``"last"``, or ``"fold_N"``).

    Args:
        model_config: Model step configuration dict, typically containing
            ``"train_params"`` and/or ``"refit_params"`` keys.

    Returns:
        Merged parameter dict.  If neither ``train_params`` nor
        ``refit_params`` is present, returns an empty dict.

    Example:
        >>> config = {
        ...     "train_params": {"verbose": 0, "n_jobs": 4},
        ...     "refit_params": {"verbose": 1, "warm_start": True},
        ... }
        >>> resolve_refit_params(config)
        {'verbose': 1, 'n_jobs': 4, 'warm_start': True}
    """
    train_params = model_config.get("train_params", {}) or {}
    refit_params = model_config.get("refit_params", {}) or {}

    # Start from train_params, override with refit_params
    merged = {**train_params, **refit_params}
    return merged
