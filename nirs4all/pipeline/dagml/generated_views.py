"""Run-local IO view buffers keyed by DAG-ML's native data-view handles.

This is an internal bridge for DATA-PROV-01 qualification. The public run path
continues to refuse generated views until the node resolver consumes this store
and the native training identity includes the exact view content receipts.
"""

from __future__ import annotations

import copy
from typing import Any


class GeneratedViewStore:
    """Materialize a scheduler-requested IO view and retain it under its handle."""

    def __init__(self, provider: Any) -> None:
        recipe = provider.recipe()
        if not recipe["params"]["_io_assembly"].get("view_generation"):
            raise ValueError("GeneratedViewStore requires an IO provider with generate_view")
        if "_dag_ml_view" in recipe["context"]:
            raise ValueError("DataProvider context reserves _dag_ml_view for the native view scope")
        provider.state_dict()  # PLAN must be complete and unchanged before any view callback.
        self._provider = provider
        self._context = copy.deepcopy(recipe["context"])
        self._views: dict[int, tuple[dict[str, Any], tuple[str, ...], Any, dict[str, Any]]] = {}

    def __call__(self, call: dict[str, Any]) -> dict[str, Any]:
        """Answer the native callback with an IO checkpoint and retain its buffers."""
        if set(call) != {"request", "handle"}:
            raise ValueError("Native data-view callback must supply request and handle")
        request, handle = call["request"], call["handle"]
        if not isinstance(request, dict) or not isinstance(handle, dict):
            raise TypeError("Native data-view request and handle must be mappings")
        if handle.get("kind") != "data_view" or type(handle.get("handle")) is not int or handle["handle"] <= 0:
            raise ValueError("Native data-view callback requires a positive data_view handle")
        if handle["handle"] in self._views:
            raise ValueError("Native data-view handle was reused")
        view = request.get("view")
        if not isinstance(view, dict):
            raise ValueError("Native data-view request has no selector")
        binding = request.get("binding")
        if not isinstance(binding, dict):
            raise ValueError("Native data-view request has no binding")
        if (view.get("source_ids") != binding.get("source_ids")
                or view.get("columns") is not None
                or view.get("branch_view") is not None
                or view.get("extra") != {}
                or view.get("include_augmented") is not False
                or view.get("include_excluded") is not False):
            raise ValueError("Generated IO view does not support this native selector")
        ids = view.get("sample_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
            raise ValueError("Native data-view request requires unique ordered sample IDs")
        key, seed = request.get("view_key"), request.get("view_seed")
        if not isinstance(key, str) or not key.strip() or type(seed) is not int or seed < 0:
            raise ValueError("Native data-view request requires a stable key and seed")
        context = copy.deepcopy(self._context)
        context["_dag_ml_view"] = {
            "phase": request.get("phase"),
            "partition": view.get("partition"),
            "fold_id": view.get("fold_id"),
            "source_ids": view.get("source_ids"),
        }
        cohort = self._provider.materialize_view(ids, seed=seed, context=context, view_key=key)
        state = self._provider.view_state_dict(cohort)
        if state["view_key"] != key or state["sample_ids"] != ids:
            raise ValueError("IO view checkpoint does not match the native request")
        frozen_handle = copy.deepcopy(handle)
        self._views[handle["handle"]] = (frozen_handle, tuple(ids), cohort, state)
        return {
            "handle": frozen_handle,
            "view_key": key,
            "sample_ids": list(ids),
            "schema_fingerprint": state["schema_fingerprint"],
            "content_fingerprint": state["fingerprint"],
        }

    def resolve(self, handle: dict[str, Any], sample_ids: list[str]) -> Any:
        """Return the exact attested cohort for a native view handle and ID order."""
        if not isinstance(handle, dict) or type(handle.get("handle")) is not int:
            raise ValueError("Generated view resolution requires a native handle")
        record = self._views.get(handle["handle"])
        if record is None:
            raise ValueError("Unknown generated data-view handle")
        expected_handle, expected_ids, cohort, state = record
        if handle != expected_handle or tuple(sample_ids) != expected_ids:
            raise ValueError("Generated data-view handle or ordered IDs do not match")
        if self._provider.view_state_dict(cohort) != state:
            raise ValueError("Generated data-view content changed after receipt")
        return cohort
