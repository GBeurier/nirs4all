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


def validate_training_control_declarations(value: Any) -> None:
    """Validate configuration shape before public dispatch, without fitting."""
    if isinstance(value, list):
        for step in value:
            validate_training_control_declarations(step)
    elif isinstance(value, dict):
        metadata = {}
        for key in ("train_params", "refit_params"):
            if key in value:
                if "model" not in value and value.get("framework") != "autogluon":
                    raise ValueError(f"{key} must belong to a model step")
                metadata[f"nirs4all_{key}"] = encode_training_controls(value[key], name=key)
        model = value.get("model")
        if metadata:
            from sklearn.base import clone

            from .autogluon_estimator import autogluon_step_estimator
            from .framework_estimator import DagMLFrameworkEstimator, framework_model_params
            from .torch_estimator import DagMLTorchEstimator, torch_model_params

            autogluon = autogluon_step_estimator(value)
            if autogluon is not None:
                model = autogluon
            else:
                torch_params = torch_model_params(model)
                if torch_params is not None:
                    model = DagMLTorchEstimator(**torch_params)
                else:
                    framework_params = framework_model_params(model)
                    if framework_params is not None:
                        model = DagMLFrameworkEstimator(**framework_params)

            apply_model_training_controls(clone(model), metadata, "FIT_CV")
            apply_model_training_controls(clone(model), metadata, "REFIT")
        for key, child in value.items():
            if key not in {"params", "train_params", "refit_params", "finetune_params"}:
                validate_training_control_declarations(child)


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
    from nirs4all.controllers.models.pipeline_cv import is_aom_estimator

    from .operator_routing import _coerce_one

    controls = effective_training_controls(metadata, phase)
    explicit_verbose = "verbose" in controls
    verbose = controls.pop("verbose", 0)
    if type(verbose) is not int or verbose < 0:
        raise ValueError("train/refit_params.verbose must be a non-negative integer")
    from .framework_estimator import DagMLFrameworkEstimator

    if isinstance(model, DagMLFrameworkEstimator) and model.framework == "tensorflow":
        for nested_name, estimator_name, flat_keys in (
            ("compile", "compile_params", {"optimizer", "loss", "metrics", "learning_rate", "lr"}),
            ("fit", "fit_params", {"epochs", "batch_size", "validation_split"}),
        ):
            if nested_name not in controls:
                continue
            nested = controls.pop(nested_name)
            if not isinstance(nested, Mapping) or any(not isinstance(key, str) for key in nested):
                raise TypeError(f"train/refit_params.{nested_name} must be a mapping with string keys")
            merged = dict(nested)
            for key in flat_keys & controls.keys():
                merged[key] = controls.pop(key)
            if nested_name == "fit" and explicit_verbose:
                merged["verbose"] = verbose
            controls[estimator_name] = merged
    refit = metadata.get("nirs4all_refit_params") or {}
    if phase == "REFIT" and (refit.get("warm_start") or "warm_start_fold" in refit):
        raise NotImplementedError("refit warm-start requires captured CV-weight transfer; a fresh estimator is not equivalent")
    pipeline_fold_policy = controls.pop("use_pipeline_folds_for_aom", "auto")
    if pipeline_fold_policy != "auto" and not is_aom_estimator(model):
        raise ValueError("train/refit_params.use_pipeline_folds_for_aom requires an AOM estimator")
    reserved = {"reset_gpu", "fit_influence"} & controls.keys()
    if reserved:
        raise NotImplementedError(f"training controls require their specialized controller owner: {sorted(reserved)}")
    from .autogluon_estimator import DagMLAutoGluonEstimator

    if isinstance(model, DagMLAutoGluonEstimator):
        model.fit_params = {**(model.fit_params or {}), **controls}
        return {"schema": "nirs4all.model-training-controls.v1", "phase": phase,
                "model_params": encode_training_controls(controls, name="effective model parameters"),
                "pipeline_fold_policy_for_aom": pipeline_fold_policy, "verbose": verbose}
    defaults = model.get_params(deep=True) if controls and callable(getattr(model, "get_params", None)) else {}
    unknown = sorted(controls.keys() - defaults.keys())
    if unknown:
        raise ValueError(f"unrecognized training parameters for {type(model).__name__}: {unknown}; these would have been ignored by the historical sklearn controller")
    if controls:
        model.set_params(**{key: _coerce_one(value, defaults.get(key)) for key, value in controls.items()})
    return {"schema": "nirs4all.model-training-controls.v1", "phase": phase,
            "model_params": encode_training_controls(controls, name="effective model parameters"),
            "pipeline_fold_policy_for_aom": pipeline_fold_policy, "verbose": verbose}


def apply_pipeline_folds_to_model(
    model: Any,
    metadata: Mapping[str, Any],
    phase: str,
    fit_ids: list[str],
) -> bool:
    """Apply the materialized outer FoldSet to an AOM model's current fit scope."""
    from nirs4all.controllers.models.pipeline_cv import (
        PrecomputedFoldSplitter,
        apply_pipeline_folds_to_aom_estimator,
        is_aom_estimator,
    )

    if not is_aom_estimator(model):
        return False
    policy = effective_training_controls(metadata, phase).get("use_pipeline_folds_for_aom", "auto")
    fold_set = metadata.get("nirs4all_pipeline_fold_set")
    if not isinstance(fold_set, Mapping):
        return apply_pipeline_folds_to_aom_estimator(
            model, None, policy=policy, unavailable_reason="the DAG model task did not carry its materialized FoldSet"
        )
    active_positions = {str(sample_id): position for position, sample_id in enumerate(fit_ids)}
    fold_universe = {str(sample_id) for sample_id in fold_set.get("sample_ids", [])}
    unknown_fit_ids = set(active_positions) - fold_universe
    if unknown_fit_ids:
        return apply_pipeline_folds_to_aom_estimator(
            model,
            None,
            policy=policy,
            unavailable_reason="augmented or branch-local rows are outside the materialized pipeline FoldSet",
        )
    local_folds: list[tuple[list[int], list[int]]] = []
    for fold in fold_set.get("folds", []):
        if not isinstance(fold, Mapping):
            continue
        local_train = [active_positions[str(sample_id)] for sample_id in fold.get("train_sample_ids", []) if str(sample_id) in active_positions]
        local_validation = [active_positions[str(sample_id)] for sample_id in fold.get("validation_sample_ids", []) if str(sample_id) in active_positions]
        if local_train and local_validation:
            local_folds.append((local_train, local_validation))
    candidate = PrecomputedFoldSplitter.from_folds(
        local_folds,
        n_samples=len(fit_ids),
        label=f"dag-ml:{phase.lower()}",
    )
    splitter: PrecomputedFoldSplitter | None = candidate if candidate.get_n_splits() >= 2 else None
    return apply_pipeline_folds_to_aom_estimator(
        model,
        splitter,
        policy=policy,
        unavailable_reason="fewer than two pipeline folds remain in the current DAG fit scope",
    )


def report_model_training_controls(evidence: Mapping[str, Any], model: Any, sample_count: int) -> None:
    """Honor controller verbosity without forcing it into estimator parameters."""
    if evidence["verbose"] > 0:
        logger.info(f"DAG {evidence['phase']}: fitted {type(model).__name__} on {sample_count} training rows; "
                    f"training overrides={evidence['model_params']}")
