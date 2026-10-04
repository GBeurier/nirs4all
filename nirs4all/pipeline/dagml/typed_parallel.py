"""Admit bounded native typed campaigns using actual Methods build evidence."""

from __future__ import annotations

import copy
import importlib
from typing import Any, cast

from .resources import current_execution_resources, normalize_execution_resources
from .tuning_contracts import DagMLTuningSpec


def typed_parallel_execution(spec: DagMLTuningSpec, run_options: dict[str, Any]) -> dict[str, Any] | None:
    """Return the signed native profile, leaving serial declarations untouched.

    The native scheduler owns candidate windows and joins. This admission check
    neither creates workers nor changes the process environment or native state.
    """
    if spec.n_jobs == 1:
        return None
    if type(spec.n_jobs) is not int or spec.n_jobs not in {2, 3, 4}:
        raise ValueError("parallel typed structural tuning requires n_jobs=2, 3 or 4")
    if spec.sampler != "random" or spec.pruner not in {None, "none"}:
        raise ValueError("parallel typed structural tuning requires sampler='random' without pruning")
    current = current_execution_resources()
    resources = normalize_execution_resources(
        cpu_threads=run_options.get("cpu_threads", current.cpu_threads),
        gpu_devices=run_options.get("gpu_devices", current.gpu_devices),
    )
    if resources.cpu_threads != 1 or resources.gpu_devices:
        raise ValueError("parallel typed structural tuning requires cpu_threads=1 and no GPU devices")
    try:
        methods = importlib.import_module("n4m")
        capability_type = getattr(methods, "BuildCapabilities", None)
        read = getattr(methods, "build_capabilities", None)
        if not isinstance(capability_type, type) or not callable(read):
            raise RuntimeError("Methods build capability evidence is unavailable")
        evidence = read()
        if type(evidence) is not capability_type:
            raise RuntimeError("Methods build capability evidence is malformed")
        evidence = cast(Any, evidence)
        if (
            type(evidence.schema_version) is not int
            or evidence.schema_version != 1
            or any(type(getattr(evidence, key, None)) is not bool for key in ("blas", "openmp", "cuda"))
            or type(evidence.sequential_cpu) is not bool
        ):
            raise RuntimeError("Methods build capability evidence is malformed")
    except (ImportError, AttributeError, TypeError) as error:
        raise RuntimeError("Methods build capability evidence is unavailable or malformed") from error
    if evidence.blas or evidence.openmp or evidence.cuda or not evidence.sequential_cpu:
        raise RuntimeError("parallel typed structural tuning requires an actual sequential CPU Methods build")
    return {
        "schema_version": 1,
        "profile": "methods_sequential_cpu_v1",
        "workers": spec.n_jobs,
        "cpu_threads": 1,
        "gpu_devices": [],
        "methods_build": {"schema_version": 1, "blas": evidence.blas, "openmp": evidence.openmp, "cuda": evidence.cuda},
    }


def closed_candidate_audit(controller: Any, *, trial_index: int, recipe_id: str) -> dict[str, Any]:
    """Snapshot closed candidate owners without inventing cross-owner chronology."""
    if not controller.closed:
        raise RuntimeError("candidate audit requires closed controller resources after the native join")
    owners = [controller.raw] if hasattr(controller, "raw") else [controller]
    if getattr(controller, "meta", None) is not None:
        owners.append(controller.meta)
    if not all(owner.closed for owner in owners):
        raise RuntimeError("candidate audit requires every controller owner to be closed")
    return {
        "trial_index": trial_index,
        "recipe_id": recipe_id,
        "closed": True,
        "owners": [{"controller_id": owner.controller_id, "closed": owner.closed, "events": copy.deepcopy(owner.audit)} for owner in owners],
    }
