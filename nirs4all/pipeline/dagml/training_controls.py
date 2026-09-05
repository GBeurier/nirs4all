"""Historical sklearn training controls, applied inside each native fit scope.

These are estimator set_params overrides, not arbitrary fit keyword arguments.
The historical controller ignored unknown keys; the general DAG host diagnoses
them instead. CV-weight warm starts and specialized controller policies require
their own owners and are not emulated by setting a similarly named parameter.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from nirs4all.core.logging import get_logger

from .operator_parameters import decode_constructor_value, encode_constructor_value

logger = get_logger(__name__)


def encode_training_controls(value: Any, *, name: str) -> dict[str, Any]:
    """Encode controls without the lossy repr fallback used by old metadata."""
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a mapping with string keys")
    encoded: dict[str, Any] = json.loads(json.dumps(encode_constructor_value(dict(value)), allow_nan=False))
    return encoded


def effective_training_controls(metadata: Mapping[str, Any], phase: str) -> dict[str, Any]:
    """Resolve refit-over-train precedence without mutating graph metadata."""
    train = dict(metadata.get("nirs4all_train_params") or {})
    refit = dict(metadata.get("nirs4all_refit_params") or {})
    if phase == "REFIT":
        train.update(refit)
    controls: dict[str, Any] = decode_constructor_value(train)
    return controls


def apply_model_training_controls(model: Any, metadata: Mapping[str, Any], phase: str) -> dict[str, Any]:
    """Apply recognized estimator overrides after candidate selection, before fit."""
    from .operator_routing import _coerce_one

    controls = effective_training_controls(metadata, phase)
    verbose = controls.pop("verbose", 0)
    if type(verbose) is not int or verbose < 0:
        raise ValueError("train/refit_params.verbose must be a non-negative integer")
    refit = metadata.get("nirs4all_refit_params") or {}
    if phase == "REFIT" and (refit.get("warm_start") or "warm_start_fold" in refit):
        raise NotImplementedError("refit warm-start requires captured CV-weight transfer; a fresh estimator is not equivalent")
    reserved = {"reset_gpu", "fit_influence", "use_pipeline_folds_for_aom"} & controls.keys()
    if reserved:
        raise NotImplementedError(f"training controls require their specialized controller owner: {sorted(reserved)}")
    defaults = model.get_params(deep=True) if controls and callable(getattr(model, "get_params", None)) else {}
    unknown = sorted(controls.keys() - defaults.keys())
    if unknown:
        raise ValueError(f"unrecognized training parameters for {type(model).__name__}: {unknown}; these would have been ignored by the historical sklearn controller")
    if controls:
        model.set_params(**{key: _coerce_one(value, defaults.get(key)) for key, value in controls.items()})
    return {"schema": "nirs4all.model-training-controls.v1", "phase": phase,
            "model_params": encode_training_controls(controls, name="effective model parameters"), "verbose": verbose}


def report_model_training_controls(evidence: Mapping[str, Any], model: Any, sample_count: int) -> None:
    """Honor controller verbosity without forcing it into estimator parameters."""
    if evidence["verbose"] > 0:
        logger.info(f"DAG {evidence['phase']}: fitted {type(model).__name__} on {sample_count} training rows; "
                    f"training overrides={evidence['model_params']}")
