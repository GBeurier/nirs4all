"""Validated archive inspection and callback-free replay for native PLS."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .native_pls_phase_controls import NATIVE_PLS_PHASE_CONTROLLER, NATIVE_PLS_PHASE_PROFILE
from .raw_replay_lowerer import RawArrayMethodsReplayCompiler, _package_document, _require_native_methods_package


def is_native_pls_phase_package(package: Mapping[str, Any]) -> bool:
    """Recognize the dedicated owner; validation still happens natively."""

    bundle = package.get("execution_bundle")
    records = bundle.get("refit_artifacts") if isinstance(bundle, Mapping) else None
    return isinstance(records, list) and any(
        isinstance(record, Mapping) and isinstance(record.get("artifact"), Mapping)
        and record["artifact"].get("controller_id") == NATIVE_PLS_PHASE_CONTROLLER
        for record in records
    )


def read_native_pls_phase_archive(path: str | Path, *, methods_library_path: str) -> tuple[Any, list[dict[str, Any]]]:
    """Validate Core/package/RAW links, then inspect officially imported states."""

    from nirs4all.api.portable_archive import read_portable_predictor_archive_v2

    package = read_portable_predictor_archive_v2(path)
    document = _package_document(package)
    _require_native_methods_package(document, native_profile=NATIVE_PLS_PHASE_PROFILE)
    dag_ml = importlib.import_module("dag_ml")
    inspect = getattr(dag_ml, "inspect_methods_role_pipeline_params", None)
    if not callable(inspect):
        raise ImportError("native PLS archive inspection requires matching DAG-ML native state inspection")
    bundle = document["execution_bundle"]
    models = []
    for record in bundle["refit_artifacts"]:
        artifact = record["artifact"]
        payload = bundle["raw_artifact_payloads"][artifact["id"]]
        # Transport Vec<u8> only: Methods interprets the model itself.
        if isinstance(payload, list):
            payload = bytes(payload)
        if not isinstance(payload, bytes):
            raise ValueError("native PLS package has a non-byte RAW payload transport")
        model = inspect(payload, methods_library_path)
        if not isinstance(model, Mapping) or model.get("native_profile") != NATIVE_PLS_PHASE_PROFILE:
            raise ValueError("saved native PLS state does not match the selected phase profile")
        if model.get("node_id") != record.get("node_id"):
            raise ValueError("saved native PLS state does not match the package node")
        models.append(dict(model))
    return package, models


def predict_native_pls_phase_archive(
    path: str | Path, data: Any, *, methods_library_path: str, outcome_id: str, run_id: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reopen the validated archive and replay current explicitly named rows."""

    from .core_archive_replay import _decode_prediction, _normalize_dataset
    from .fit_identity import normalize_predict_identity

    package, models = read_native_pls_phase_archive(path, methods_library_path=methods_library_path)
    X, sample_ids = _normalize_dataset(data)
    identity = normalize_predict_identity(X, sample_ids=sample_ids, require_explicit_sample_ids=True)
    compiled = RawArrayMethodsReplayCompiler(
        package, native_profile=NATIVE_PLS_PHASE_PROFILE, methods_library_path=methods_library_path,
        outcome_id=outcome_id, run_id=run_id,
    ).compile_replay(None, X, mode="predict", identity_frame=identity)
    dag_ml = importlib.import_module("dag_ml")
    execute = getattr(dag_ml, "replay_loaded_methods_predictor_package", None)
    if not callable(execute):
        raise ImportError("native PLS archive replay requires matching DAG-ML native Methods replay")
    outcome = execute(
        package, compiled.request, compiled.data_envelopes, compiled.methods_inputs,
        methods_library_path=methods_library_path, outcome_id=outcome_id, run_id=run_id,
    )
    document = outcome.to_dict() if hasattr(outcome, "to_dict") else outcome
    targets = tuple(_package_document(package)["output_bindings"][0]["target_names"])
    values = _decode_prediction(document, sample_ids, target_names=targets)
    return values, {
        "engine": "core-native", "archive_path": str(path), "archive_schema_version": 2,
        "native_profile": NATIVE_PLS_PHASE_PROFILE, "sample_ids": list(sample_ids), "target_names": list(targets),
        "native_predictor_descriptors": models, "outcome_id": outcome_id, "run_id": run_id,
        "training_performed": False,
    }
