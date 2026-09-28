"""Run-local IO view buffers keyed by DAG-ML's native data-view handles.

This is an internal bridge for DATA-PROV-01 qualification. The public run path
continues to refuse generated views until the node resolver consumes this store
and the native training identity includes the exact view content receipts.
"""

from __future__ import annotations

import copy
from typing import Any

_SELECTOR_FIELDS = (
    "sample_ids", "partition", "fold_id", "source_ids", "columns",
    "branch_view", "include_augmented", "include_excluded", "extra",
)


def _selector_fields(view: dict[str, Any]) -> dict[str, Any]:
    """Compare a native selector including absent optional fields as null."""
    return {name: view.get(name) for name in _SELECTOR_FIELDS}


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
        self._views: dict[int, tuple[dict[str, Any], tuple[str, ...], Any, dict[str, Any], dict[str, Any]]] = {}

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
        if request.get("input_name") != binding.get("input_name"):
            raise ValueError("Native data-view input name does not match its binding")
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
        self._views[handle["handle"]] = (frozen_handle, tuple(ids), cohort, state, copy.deepcopy(request))
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
        expected_handle, expected_ids, cohort, state, _request = record
        if handle != expected_handle or tuple(sample_ids) != expected_ids:
            raise ValueError("Generated data-view handle or ordered IDs do not match")
        if self._provider.view_state_dict(cohort) != state:
            raise ValueError("Generated data-view content changed after receipt")
        return cohort

    def _task_binding_for(self, handle: dict[str, Any], sample_ids: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
        self.resolve(handle, sample_ids)
        stored_handle, stored_ids, _cohort, state, request = self._views[handle["handle"]]
        receipt = {
            "handle": copy.deepcopy(stored_handle),
            "view_key": request["view_key"],
            "sample_ids": list(stored_ids),
            "schema_fingerprint": state["schema_fingerprint"],
            "content_fingerprint": state["fingerprint"],
        }
        return copy.deepcopy(request), receipt

    def bind_task(self, task: dict[str, Any]) -> GeneratedTaskViews:
        """Bind a native task's named view handles to its exact IO cohorts."""
        return GeneratedTaskViews(self, task)


class GeneratedTaskViews:
    """Read generated features only through a task's attested native view handles.

    Callers must name both the input and partition. A task may carry different
    views for train and validation, or multiple views containing the same ID.
    This object never chooses a view from the sample ID alone.
    """

    def __init__(self, store: GeneratedViewStore, task: dict[str, Any]) -> None:
        views, handles = task.get("data_views"), task.get("input_handles")
        if not isinstance(views, dict) or not views or not isinstance(handles, dict):
            raise ValueError("Generated task requires native data views and input handles")
        receipts = task.get("data_view_receipts")
        if not isinstance(receipts, dict) or set(receipts) != set(views):
            raise ValueError("Generated task requires a native receipt for every data view")
        self._store = store
        self._task = copy.deepcopy(task)
        self._scopes: dict[tuple[str, str], tuple[dict[str, Any], list[str]]] = {}
        self._scope_keys: dict[tuple[str, str], str] = {}
        self._read_batches: dict[str, list[list[str]]] = {}
        seen_handles: set[int] = set()
        if any(handle.get("kind") == "data_view" and key not in views
               for key, handle in handles.items() if isinstance(handle, dict)):
            raise ValueError("Generated task has an unbound native data-view handle")
        for key, view in views.items():
            if not isinstance(key, str) or not key.startswith("data:") or not isinstance(view, dict):
                raise ValueError("Generated task contains an invalid data view")
            handle = handles.get(key)
            if not isinstance(handle, dict) or handle.get("kind") != "data_view":
                raise ValueError(f"Generated task has no native data-view handle for {key!r}")
            ids, partition = view.get("sample_ids"), view.get("partition")
            if (not isinstance(ids, list) or not ids
                    or any(not isinstance(item, str) or not item for item in ids)
                    or len(ids) != len(set(ids))
                    or not isinstance(partition, str) or not partition):
                raise ValueError(f"Generated task has an invalid selector for {key!r}")
            request, expected_receipt = store._task_binding_for(handle, ids)
            if receipts[key] != expected_receipt:
                raise ValueError(f"Generated task receipt does not match IO view {key!r}")
            input_name = request["input_name"]
            native_key = f"data:{input_name}"
            if key not in (native_key, f"{native_key}:validation", f"{native_key}:test"):
                raise ValueError(f"Generated task data-view key {key!r} does not match its native binding")
            if key.endswith(":validation") and key != native_key and partition not in ("fold_validation", "predict"):
                raise ValueError("Generated task validation handle has the wrong partition")
            if key.endswith(":test") and key != native_key and partition != "predict":
                raise ValueError("Generated task test handle has the wrong partition")
            node_plan = task.get("node_plan")
            if (not isinstance(node_plan, dict)
                    or task.get("run_id") != request.get("run_id")
                    or node_plan.get("node_id") != request.get("node_id")
                    or task.get("phase") != request.get("phase")
                    or task.get("fold_id") != request.get("fold_id")
                    or task.get("variant_id") != request.get("variant_id")
                    or _selector_fields(view) != _selector_fields(request["view"])):
                raise ValueError(f"Generated task scope or selector changed for {key!r}")
            scope = (input_name, partition)
            if scope in self._scopes:
                raise ValueError(f"Generated task has ambiguous native view scope {scope!r}")
            if handle["handle"] in seen_handles:
                raise ValueError("Generated task reuses one native data-view handle")
            seen_handles.add(handle["handle"])
            self._scopes[scope] = (copy.deepcopy(handle), list(ids))
            self._scope_keys[scope] = key

    def validate_task(self, task: dict[str, Any]) -> None:
        """Refuse reuse of this binding for a different native task."""
        if task != self._task:
            raise ValueError("Generated task views are bound to a different native task")

    def take(self, input_name: str, partition: str, sample_ids: list[str]) -> Any:
        """Return typed IO rows from one named view, preserving requested order.

        A subset is allowed because a controller may read a source's present
        rows only. The full native view is reattested on every read, including
        after the task was bound. Mixed train/validation reads must call this
        method separately for each partition before assembling their outputs.
        """
        scope = (input_name, partition)
        if scope not in self._scopes:
            raise ValueError(f"Generated task has no native view for {scope!r}")
        handle, full_ids = self._scopes[scope]
        if (not isinstance(sample_ids, list) or not sample_ids
                or any(not isinstance(item, str) or not item for item in sample_ids)
                or len(sample_ids) != len(set(sample_ids))
                or not set(sample_ids).issubset(full_ids)):
            raise ValueError("Generated task read must use unique IDs within its native view")
        cohort = self._store.resolve(handle, full_ids)
        selected = cohort.take(sample_ids)
        self._read_batches.setdefault(self._scope_keys[scope], []).append(list(sample_ids))
        return selected

    def consumed_data_views(self) -> dict[str, dict[str, Any]]:
        """Report successful materialization reads, including presence checks.

        This ledger does not prove which rows the fitted estimator used.
        """
        receipts = self._task["data_view_receipts"]
        return {
            key: {
                "receipt": copy.deepcopy(receipts[key]),
                "read_batches": copy.deepcopy(batches),
            }
            for key, batches in sorted(self._read_batches.items())
        }

    def feature_blocks(
        self, input_name: str, partition: str, sample_ids: list[str],
        *, source_names: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Read typed feature blocks from one explicitly named native view.

        The source order comes from the frozen IO schema unless the controller
        asks for a unique named subset. Missing-source masks travel with the
        same selected rows. Targets are deliberately absent: they remain bound
        to the PLAN cohort held by the host resolver.
        """
        cohort = self.take(input_name, partition, sample_ids)
        names = tuple(cohort.sources) if source_names is None else source_names
        if not names or len(names) != len(set(names)) or any(name not in cohort.sources for name in names):
            raise ValueError("Generated feature read requires unique declared source names")
        blocks = cohort.source_values(source_names=names)
        presence = cohort.source_presence()
        result: dict[str, Any] = {
            "feature_set_id": "features",
            "observation_ids": list(sample_ids),
            "source_names": names,
            "blocks": blocks,
        }
        if any(not presence[name].all() for name in names):
            result["source_masks"] = {name: presence[name] for name in names}
        return result
