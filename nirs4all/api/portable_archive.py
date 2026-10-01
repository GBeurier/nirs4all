"""Explicit portable Package V2 archive transport and callback replay.

Core owns the archive container; DAG-ML owns package semantics, trust and
replay. This surface does not deserialize Python models or reinterpret the
separate closed N4MM prediction lane.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from nirs4all.pipeline.dagml.native_archive_replay import write_methods_archive_v2


def write_portable_predictor_archive_v2(
    path: str | Path,
    *,
    archive_id: str,
    outcome: Any,
    package: Any,
) -> dict[str, str]:
    """Write captured portable Package V2 artifacts without another fit.

    DAG-ML assembles the complete native payloads and Core validates/writes
    them. RAW ``methods_role_pipeline`` artifacts retain their controller
    trust contract; they are not converted to N4MM or Python sidecars.

    Args:
        path: Destination ``.n4a`` archive.
        archive_id: Explicit archive identity.
        outcome: Captured DAG-ML ``TrainingOutcome`` matching the package.
        package: Captured DAG-ML ``PortablePredictorPackage`` V2.

    Returns:
        Core's archive identity and SHA-256 reference.
    """

    return write_methods_archive_v2(path, archive_id=archive_id, outcome=outcome, package=package)


def read_portable_predictor_archive_v2(path: str | Path) -> Any:
    """Read a typed Package V2 after Core and DAG-ML validate the archive.

    Core validates bounded ZIP storage and returns opaque members. DAG-ML
    then checks the outcome/package links, companion members, full RAW
    artifact coverage and controller contracts. No callback or model loader
    runs while reading. The returned package is ready for explicit DAG-ML
    replay; reading does not grant trust to its controllers.

    Args:
        path: Existing portable Core Archive V2.

    Returns:
        A validated DAG-ML ``PortablePredictorPackage``.

    Raises:
        ImportError: A required matching Core or DAG-ML facade is absent.
        ValueError: The archive or native payload semantics are invalid.
    """

    core = importlib.import_module("nirs4all_core")
    dag_ml = importlib.import_module("dag_ml")
    reader = getattr(core, "read_archive_v2_payloads", None)
    validator = getattr(dag_ml, "validate_archive_v2_portable_payloads", None)
    if not callable(reader) or not callable(validator):
        raise ImportError("portable Archive V2 reading requires matching Core and DAG-ML payload validation facades")
    payloads = reader(str(path))
    if not isinstance(payloads, Mapping) or set(payloads) != {"manifest", "members"}:
        raise ValueError("Core Archive V2 reader returned an invalid payload transport")
    manifest = payloads["manifest"]
    members = payloads["members"]
    if not isinstance(manifest, Mapping) or not isinstance(members, Mapping):
        raise ValueError("Core Archive V2 reader returned invalid manifest or members")
    package_bytes = members.get("dagml/portable_predictor_package.json")
    if not isinstance(package_bytes, bytes):
        raise ValueError("Core Archive V2 reader did not return the portable package bytes")
    package = dag_ml.PortablePredictorPackage(package_bytes.decode("utf-8"))
    validator(manifest, package, members)
    return package


def inspect_portable_predictor_archive_v2(
    path: str | Path, *, methods_library_path: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect saved parameters of the closed native PLS phase profile.

    Core validates the container, DAG-ML validates package/RAW bindings, and
    Methods imports each fitted state and checks its native parameters. No fit
    or prediction runs. Other controller profiles are explicitly refused.

    Returns:
        The native profile, archive path, a ``models`` list with validated
        ``steps`` and effective REFIT ``model_params``, and
        ``training_performed=False``.
    """

    from nirs4all.pipeline.dagml.core_archive_replay import _resolve_methods_library_identity
    from nirs4all.pipeline.dagml.native_pls_phase_controls import NATIVE_PLS_PHASE_PROFILE
    from nirs4all.pipeline.dagml.native_pls_phase_replay import read_native_pls_phase_archive

    library_path, _ = _resolve_methods_library_identity(methods_library_path)
    _, models = read_native_pls_phase_archive(path, methods_library_path=library_path)
    return {
        "native_profile": NATIVE_PLS_PHASE_PROFILE, "archive_path": str(path),
        "models": models, "training_performed": False,
    }


def replay_portable_predictor_archive_v2(
    path: str | Path,
    request: Any,
    data_envelopes: Any,
    trusted_manifests: Any,
    op_callback: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    outcome_id: str,
    run_id: str,
    artifact_callback: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> Any:
    """Replay an archive on an explicitly signed current cohort, without fit.

    The archive is reopened and fully validated before DAG-ML sees callbacks.
    DAG-ML validates the signed PREDICT request, named current data envelopes
    and explicitly trusted controller manifests before hydration. Its driver
    schedules prediction and releases artifact states, including on errors.
    Controller callbacks must implement that read-only replay contract;
    this facade supplies no training callback, fallback or Python model load.

    Args:
        path: Existing portable Core Archive V2.
        request: Signed current-cohort DAG-ML PREDICT replay request.
        data_envelopes: Named, signed current-cohort data envelopes.
        trusted_manifests: Explicitly trusted controller manifests for replay.
        op_callback: Trusted controller's PREDICT operation callback.
        outcome_id: Explicit replay outcome identity.
        run_id: Explicit replay run identity.
        artifact_callback: Trusted controller's hydration/release callback.

    Returns:
        The native DAG-ML replay outcome, including aligned predictions.
    """

    if trusted_manifests is None:
        raise ValueError("portable archive replay requires explicit trusted controller manifests")
    package = read_portable_predictor_archive_v2(path)
    dag_ml = importlib.import_module("dag_ml")
    return dag_ml.replay_loaded_predictor_package(
        package,
        request,
        data_envelopes,
        {},
        op_callback,
        outcome_id=outcome_id,
        run_id=run_id,
        artifact_callback=artifact_callback,
        trusted_controller_manifests=trusted_manifests,
    )


__all__ = [
    "inspect_portable_predictor_archive_v2",
    "read_portable_predictor_archive_v2",
    "replay_portable_predictor_archive_v2",
    "write_portable_predictor_archive_v2",
]
