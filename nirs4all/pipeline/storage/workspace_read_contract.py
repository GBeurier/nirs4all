"""Published read-only projection contracts for WorkspaceStore schema v5.

The contract is a distribution resource so native consumers can use the same
bounded SQL projections without importing ``WorkspaceStore`` or recreating its
private schema knowledge. It is intentionally limited to bounded run,
pipeline, and explicitly directed chain-ranking projections. Studio's
high-level results policy, arrays, artifacts, predictions and mutations need
their own contracts.

The results-summary contract is deliberately a separate resource: it composes
the stable low-level read projection with the ranking and serialization policy
of the compact Studio results response without changing the read-v1 bytes.

The run-detail HTTP contract is a cutover manifest plus a Python oracle for the
inputs owned by this distribution. It deliberately leaves Studio catalogue,
presentation/runtime compatibility, and legacy-manifest policies external.
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any, Final, cast

from .store_schema import SCHEMA_VERSION

WORKSPACE_STORE_READ_CONTRACT_RESOURCE: Final = "contracts/workspace_store_read_v1.json"
WORKSPACE_STORE_READ_CONTRACT_SCHEMA_ID: Final = "nirs4all.workspace-store-read.v1"
WORKSPACE_STORE_READ_CONTRACT_SCHEMA_VERSION: Final = 1
WORKSPACE_STORE_RESULTS_SUMMARY_CONTRACT_RESOURCE: Final = "contracts/workspace_store_results_summary_v1.json"
WORKSPACE_STORE_RESULTS_SUMMARY_CONTRACT_SCHEMA_ID: Final = "nirs4all.workspace-store-results-summary.v1"
WORKSPACE_STORE_RESULTS_SUMMARY_CONTRACT_SCHEMA_VERSION: Final = 1
STUDIO_RUN_DETAIL_HTTP_CONTRACT_RESOURCE: Final = "contracts/studio_run_detail_http_v1.json"
STUDIO_RUN_DETAIL_HTTP_CONTRACT_SCHEMA_ID: Final = "nirs4all.studio-run-detail-http.v1"
STUDIO_RUN_DETAIL_HTTP_CONTRACT_SCHEMA_VERSION: Final = 1


def workspace_store_read_contract() -> dict[str, Any]:
    """Return the version-checked, read-only WorkspaceStore contract.

    Consumers must refuse a database whose ``PRAGMA user_version`` differs
    from the contract's exact store schema version before executing its query.
    """
    resource = files(__package__).joinpath(WORKSPACE_STORE_READ_CONTRACT_RESOURCE)
    contract: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    if contract.get("schema_id") != WORKSPACE_STORE_READ_CONTRACT_SCHEMA_ID:
        raise RuntimeError("workspace read contract has an unexpected schema id")
    if contract.get("schema_version") != WORKSPACE_STORE_READ_CONTRACT_SCHEMA_VERSION:
        raise RuntimeError("workspace read contract has an unsupported schema version")
    if contract.get("workspace_store_schema_version") != SCHEMA_VERSION:
        raise RuntimeError("workspace read contract does not match the installed WorkspaceStore schema version")
    return contract


def workspace_store_results_summary_contract() -> dict[str, Any]:
    """Return the version-checked Store-v5 results-summary contract.

    This policy is self-contained except for its explicit reference to the
    ``studio_chain_ranked_v1`` projection in :func:`workspace_store_read_contract`.
    It owns the complete, paged source query plus selection, synthetic-refit,
    metric-direction and JSON/null serialization semantics needed by a native
    consumer of the fixed ``n=5`` summary surface.
    """
    resource = files(__package__).joinpath(WORKSPACE_STORE_RESULTS_SUMMARY_CONTRACT_RESOURCE)
    contract: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    if contract.get("schema_id") != WORKSPACE_STORE_RESULTS_SUMMARY_CONTRACT_SCHEMA_ID:
        raise RuntimeError("workspace results-summary contract has an unexpected schema id")
    if contract.get("schema_version") != WORKSPACE_STORE_RESULTS_SUMMARY_CONTRACT_SCHEMA_VERSION:
        raise RuntimeError("workspace results-summary contract has an unsupported schema version")
    if contract.get("workspace_store_schema_version") != SCHEMA_VERSION:
        raise RuntimeError("workspace results-summary contract does not match the installed WorkspaceStore schema version")

    dependency = contract.get("dependencies", {}).get("workspace_store_read")
    read_contract = workspace_store_read_contract()
    if dependency != {
        "schema_id": read_contract["schema_id"],
        "schema_version": read_contract["schema_version"],
        "projection": "studio_chain_ranked_v1",
    }:
        raise RuntimeError("workspace results-summary contract has an incompatible read-contract dependency")
    return contract


def studio_run_detail_http_contract() -> dict[str, Any]:
    """Return the version-checked run-detail HTTP composition contract.

    The contract publishes complete Store-v5 owner inputs, but intentionally
    forbids route selection until the separately owned Studio and legacy
    policies named in ``cutover.blocked_on`` are versioned and qualified.
    """
    resource = files(__package__).joinpath(STUDIO_RUN_DETAIL_HTTP_CONTRACT_RESOURCE)
    contract: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    if contract.get("schema_id") != STUDIO_RUN_DETAIL_HTTP_CONTRACT_SCHEMA_ID:
        raise RuntimeError("studio run-detail HTTP contract has an unexpected schema id")
    if contract.get("schema_version") != STUDIO_RUN_DETAIL_HTTP_CONTRACT_SCHEMA_VERSION:
        raise RuntimeError("studio run-detail HTTP contract has an unsupported schema version")
    if contract.get("workspace_store_schema_version") != SCHEMA_VERSION:
        raise RuntimeError("studio run-detail HTTP contract does not match the installed WorkspaceStore schema version")

    dependency = contract.get("dependencies", {}).get("workspace_store_read")
    read_contract = workspace_store_read_contract()
    if dependency != {
        "schema_id": read_contract["schema_id"],
        "schema_version": read_contract["schema_version"],
        "projection": "studio_run_detail_v1",
    }:
        raise RuntimeError("studio run-detail HTTP contract has an incompatible read-contract dependency")
    runtime_dependency = contract.get("dependencies", {}).get("pipeline_runtime")
    if runtime_dependency != {
        "owner_method": "WorkspaceStore.get_studio_run_detail_runtime_v1",
        "source_table": "pipelines",
        "required_columns": ["pipeline_id", "run_id", "created_at"],
        "optional_columns": [
            "engine",
            "engine_requested",
            "engine_diagnostics",
            "runtime_manifest",
            "fallback_policy",
            "native_result_refs",
        ],
        "optional_column_selection": "fixed_allowlist_present_column_or_sql_null_alias",
        "absent_optional_column": "null_with_absent_in_store_v5_provenance",
        "present_text_columns": ["engine", "engine_requested"],
        "present_json_shapes": {
            "engine_diagnostics": "array_or_null",
            "runtime_manifest": "object_or_null",
            "fallback_policy": "object_or_null",
            "native_result_refs": "array_or_null",
        },
        "malformed_or_wrong_shape": "reject",
        "non_finite_numbers": "replace_with_null_recursively",
        "ordering": "pipeline_created_at_desc_then_pipeline_id_asc",
    }:
        raise RuntimeError("studio run-detail HTTP contract has an incompatible runtime projection")
    splitter_output = contract.get("owner_output", {}).get("pipeline_splitters", {})
    if splitter_output.get("materialization") != "derived_by_owner_oracle_before_consumer_boundary" or splitter_output.get("consumer_reimplementation") != "forbidden":
        raise RuntimeError("studio run-detail HTTP contract does not preserve owner-only splitter materialization")
    if contract.get("cutover", {}).get("route_selection") != "forbidden":
        raise RuntimeError("studio run-detail HTTP contract must remain fail-closed until all external policies are proven")
    return contract


def _studio_splitter_config_payload(expanded_config: Any) -> dict[str, Any] | None:
    """Serialize the library-owned splitter projection for one pipeline."""
    from nirs4all.pipeline.analysis.splitter_config import extract_splitter_config

    splitter = extract_splitter_config(expanded_config)
    if splitter is None:
        return None
    return {
        "splitter_class": splitter.splitter_class,
        "reference": splitter.reference,
        "n_splits": splitter.n_splits,
        "shuffle": splitter.shuffle,
        "random_state": splitter.random_state,
        "test_size": splitter.test_size,
        "group_by": splitter.group_by,
    }


def studio_run_detail_http_inputs_v1(workspace_path: str | Path, run_id: str) -> dict[str, Any] | None:
    """Return the Store/library-owned inputs for Studio run-detail HTTP.

    This is not the complete HTTP response. Dataset linking, Studio's CV label
    vocabulary, runtime compatibility aliases, rerun readiness, and the legacy
    manifest branch remain external exactly as declared by
    :func:`studio_run_detail_http_contract`.
    """
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    studio_run_detail_http_contract()
    database = Path(workspace_path) / "store.sqlite"
    before = database.stat()
    before_signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    sidecars = tuple(Path(f"{database}{suffix}") for suffix in ("-wal", "-shm", "-journal"))
    try:
        run_detail = WorkspaceStore.get_studio_run_detail_v1(workspace_path, run_id)
        if run_detail is None:
            return None
        runtime = WorkspaceStore.get_studio_run_detail_runtime_v1(workspace_path, run_id)
        if runtime is None:
            raise RuntimeError("studio run-detail runtime input disappeared during owner composition")

        raw_pipelines = run_detail.get("pipelines")
        if not isinstance(raw_pipelines, list) or not all(isinstance(pipeline, dict) for pipeline in raw_pipelines):
            raise RuntimeError("studio run-detail owner input requires an ordered pipeline list")
        pipelines = cast(list[dict[str, Any]], raw_pipelines)
        pipeline_ids = [pipeline.get("pipeline_id") for pipeline in pipelines]

        raw_pipeline_runtime = runtime.get("pipeline_runtime")
        provenance = runtime.get("runtime_column_provenance")
        if (
            not isinstance(raw_pipeline_runtime, list)
            or not all(isinstance(row, dict) for row in raw_pipeline_runtime)
            or [row.get("pipeline_id") for row in raw_pipeline_runtime] != pipeline_ids
            or not isinstance(provenance, dict)
        ):
            raise RuntimeError("studio run-detail runtime input does not align with the owner pipeline order")

        pipeline_splitters = [
            {
                "pipeline_id": pipeline.get("pipeline_id"),
                "splitter": _studio_splitter_config_payload(pipeline.get("expanded_config")),
            }
            for pipeline in pipelines
        ]
        results = [
            {
                "id": pipeline.get("pipeline_id", ""),
                "run_id": pipeline.get("run_id", ""),
                "dataset": pipeline.get("dataset_name", ""),
                "pipeline_config": pipeline.get("name", ""),
                "pipeline_config_id": pipeline.get("pipeline_id", ""),
                "created_at": pipeline.get("created_at") or "",
                "best_score": pipeline.get("best_val"),
                "best_test_score": pipeline.get("best_test"),
                "metric": pipeline.get("metric", ""),
                "status": pipeline.get("status", ""),
                "duration_ms": pipeline.get("duration_ms"),
                "format": "store",
            }
            for pipeline in pipelines
        ]
        return {
            "source_branch": "store_v5",
            "run_detail": run_detail,
            "pipeline_splitters": pipeline_splitters,
            "pipeline_runtime": raw_pipeline_runtime,
            "runtime_column_provenance": provenance,
            "results": results,
            "results_count": len(results),
        }
    finally:
        after = database.stat()
        after_signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if any(path.exists() for path in sidecars):
            raise RuntimeError("studio run-detail owner composition detected an active SQLite journal")
        if after_signature != before_signature:
            raise RuntimeError("studio run-detail owner composition detected a database change")


__all__ = [
    "WORKSPACE_STORE_READ_CONTRACT_RESOURCE",
    "WORKSPACE_STORE_READ_CONTRACT_SCHEMA_ID",
    "WORKSPACE_STORE_READ_CONTRACT_SCHEMA_VERSION",
    "WORKSPACE_STORE_RESULTS_SUMMARY_CONTRACT_RESOURCE",
    "WORKSPACE_STORE_RESULTS_SUMMARY_CONTRACT_SCHEMA_ID",
    "WORKSPACE_STORE_RESULTS_SUMMARY_CONTRACT_SCHEMA_VERSION",
    "STUDIO_RUN_DETAIL_HTTP_CONTRACT_RESOURCE",
    "STUDIO_RUN_DETAIL_HTTP_CONTRACT_SCHEMA_ID",
    "STUDIO_RUN_DETAIL_HTTP_CONTRACT_SCHEMA_VERSION",
    "studio_run_detail_http_contract",
    "studio_run_detail_http_inputs_v1",
    "workspace_store_read_contract",
    "workspace_store_results_summary_contract",
]
