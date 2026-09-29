"""Run-local IO view buffers keyed by DAG-ML's native data-view handles.

This internal bridge supports qualified concrete-model CV run profiles.
The host reports generated-view digests and model-call evidence to DAG-ML;
model export remains closed pending a replay contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
from typing import Any, SupportsIndex

import numpy as np

_SELECTOR_FIELDS = (
    "sample_ids", "partition", "fold_id", "source_ids", "columns",
    "branch_view", "include_augmented", "include_excluded", "extra",
)
_BINDING_IDENTITY_FIELDS = (
    "schema_fingerprint", "plan_fingerprint", "relation_fingerprint",
    "output_representation", "request_id", "input_name", "feature_set_id",
    "source_ids", "metadata",
)


def qualified_generated_model_pipeline(pipeline: Any) -> bool:
    """Whether a pipeline has a qualified generated-view model/transform shape."""
    from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

    return (
        isinstance(pipeline, list) and len(pipeline) in (2, 3)
        and type(pipeline[0]) in (KFold, GroupKFold, StratifiedKFold, StratifiedGroupKFold)
        and (not getattr(pipeline[0], "shuffle", False)
             or type(getattr(pipeline[0], "random_state", None)) is int)
        and (len(pipeline) == 2 or (
            not isinstance(pipeline[1], (type, dict, str))
            and callable(getattr(pipeline[1], "fit", None))
            and callable(getattr(pipeline[1], "transform", None))
        ))
        and type(pipeline[-1]) is dict and set(pipeline[-1]) == {"model"}
        and not isinstance(pipeline[-1]["model"], type)
        and callable(getattr(pipeline[-1]["model"], "fit", None))
        and callable(getattr(pipeline[-1]["model"], "predict", None))
    )


def qualified_generated_by_source_pipeline(pipeline: Any) -> bool:
    """Accept the existing distinct by-source preprocessing and concat shape."""
    from .detect import _detect_by_source_distinct_preproc_concat

    if not isinstance(pipeline, list) or len(pipeline) != 4:
        return False
    branch = pipeline[1]
    if not isinstance(branch, dict) or set(branch) != {"branch"}:
        return False
    criterion = branch["branch"]
    if not isinstance(criterion, dict) or set(criterion) != {"by_source", "steps"} or criterion["by_source"] is not True:
        return False
    source_steps = criterion["steps"]
    if not isinstance(source_steps, dict) or len(source_steps) < 2:
        return False
    if not qualified_generated_model_pipeline([pipeline[0], pipeline[-1]]):
        return False
    if _detect_by_source_distinct_preproc_concat(pipeline, len(source_steps)) is None:
        return False
    return all(
        isinstance(steps, list) and steps and all(
            not isinstance(step, (type, dict, str))
            and callable(getattr(step, "fit", None))
            and callable(getattr(step, "transform", None))
            for step in steps
        )
        for steps in source_steps.values()
    )


def _selector_fields(view: dict[str, Any]) -> dict[str, Any]:
    """Compare a native selector including absent optional fields as null."""
    return {name: view.get(name) for name in _SELECTOR_FIELDS}


def _model_value_descriptor(value: Any) -> Any:
    """Describe model-call arguments without coercing their dtype or layout."""
    from nirs4all_io.ragged import RaggedSeriesBatch

    if isinstance(value, RaggedSeriesBatch):
        return {
            "kind": "ragged_series",
            "values": _model_value_descriptor(value.values),
            "offsets": _model_value_descriptor(value.offsets),
            "time_coordinates": _model_value_descriptor(value.time_coordinates),
        }
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.hasobject:
            return {
                "kind": "array", "dtype": array.dtype.str, "shape": list(array.shape),
                "items": [_model_value_descriptor(item) for item in array.flat],
            }
        return {
            "kind": "array", "dtype": array.dtype.str, "shape": list(array.shape),
            "content": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return _model_value_descriptor(np.asarray(value))
    if isinstance(value, (list, tuple)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list",
                "items": [_model_value_descriptor(item) for item in value]}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Generated model input mappings require string keys")
        return {"kind": "mapping", "items": {key: _model_value_descriptor(value[key]) for key in sorted(value)}}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _model_value_descriptor(np.asarray(value))
    raise TypeError(f"Unsupported generated model input type: {type(value).__name__}")


def _model_value_fingerprint(value: Any) -> str:
    descriptor = _model_value_descriptor(value)
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_model_rows(value: Any, count: int, label: str) -> None:
    """Reject a reported ID list that does not match the supplied row buffers."""
    from nirs4all_io.ragged import RaggedSeriesBatch

    if isinstance(value, RaggedSeriesBatch):
        rows = len(value)
    elif isinstance(value, np.ndarray):
        if value.ndim == 0:
            raise ValueError(f"Generated {label} must have a row dimension")
        rows = value.shape[0]
    elif isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"Generated {label} cannot be empty")
        if all(isinstance(item, (np.ndarray, RaggedSeriesBatch)) for item in value):
            for item in value:
                _require_model_rows(item, count, label)
            return
        rows = len(value)
    else:
        raise TypeError(f"Generated {label} has unsupported row buffer type {type(value).__name__}")
    if rows != count:
        raise ValueError(f"Generated {label} has {rows} rows for {count} native sample IDs")


def _require_option_rows(options: dict[str, Any], count: int) -> None:
    for key, value in options.items():
        if isinstance(value, dict):
            _require_option_rows(value, count)
        elif isinstance(value, (np.ndarray, list, tuple)):
            _require_model_rows(value, count, f"option {key}")


class GeneratedViewStore:
    """Materialize a scheduler-requested IO view and retain it under its handle."""

    def __reduce_ex__(self, protocol: SupportsIndex) -> Any:
        raise TypeError("live generated data view stores cannot be serialized")

    def __init__(self, provider: Any, *, _provider_lock: Any = None) -> None:
        self._provider_lock = _provider_lock if _provider_lock is not None else threading.RLock()
        with self._provider_lock:
            recipe = provider.recipe()
            if not recipe["params"]["_io_assembly"].get("view_generation"):
                raise ValueError("GeneratedViewStore requires an IO provider with generate_view")
            if "_dag_ml_view" in recipe["context"]:
                raise ValueError("DataProvider context reserves _dag_ml_view for the native view scope")
            provider.state_dict()  # PLAN must be complete and unchanged before any view callback.
        self._provider = provider
        self._context = copy.deepcopy(recipe["context"])
        self._views: dict[int, tuple[dict[str, Any], tuple[str, ...], Any, dict[str, Any], dict[str, Any]]] = {}
        self._by_key: dict[str, tuple[tuple[str, ...], int, dict[str, Any], dict[str, Any], Any, dict[str, Any]]] = {}

    def for_trial(self) -> GeneratedViewStore:
        """Give one HPO candidate an isolated native-handle namespace."""
        return GeneratedViewStore(self._provider, _provider_lock=self._provider_lock)

    def provider_for_worker(self) -> Any:
        """Transfer the PLAN-complete provider, never the live receipt/buffer store."""
        with self._provider_lock:
            if self._views or self._by_key:
                raise ValueError("generated worker requires a fresh view store")
            self._provider.state_dict()
            return self._provider

    def _context_for(self, view: dict[str, Any]) -> dict[str, Any]:
        context: dict[str, Any] = copy.deepcopy(self._context)
        context["_dag_ml_view"] = {
            "partition": view.get("partition"),
            "fold_id": view.get("fold_id"),
            "source_ids": view.get("source_ids"),
        }
        return context

    def recheck_record(self, record: dict[str, Any]) -> dict[str, str]:
        """Regenerate a saved HPO view and compare its actual IO content."""
        with self._provider_lock:
            return self._recheck_record_locked(record)

    def _recheck_record_locked(self, record: dict[str, Any]) -> dict[str, str]:
        if not isinstance(record, dict) or not isinstance(record.get("view"), dict):
            raise ValueError("Generated HPO resume requires a native view record")
        view = record["view"]
        ids, key, seed = view.get("sample_ids"), record.get("view_key"), record.get("view_seed")
        if (not isinstance(ids, list) or not ids or any(not isinstance(item, str) or not item for item in ids)
                or len(ids) != len(set(ids)) or not isinstance(key, str) or not key.strip()
                or type(seed) is not int or seed < 0):
            raise ValueError("Generated HPO resume record has invalid IDs, key or seed")
        cohort = self._provider.materialize_view(ids, seed=seed, context=self._context_for(view), view_key=key)
        state = self._provider.view_state_dict(cohort)
        if (state["view_key"] != key or state["sample_ids"] != ids
                or state["schema_fingerprint"] != record.get("schema_fingerprint")
                or state["fingerprint"] != record.get("content_fingerprint")):
            raise ValueError("Generated HPO resume view content differs from its saved checkpoint")
        return {"schema_fingerprint": state["schema_fingerprint"], "content_fingerprint": state["fingerprint"]}

    def __call__(self, call: dict[str, Any]) -> dict[str, Any]:
        """Answer the native callback with an IO checkpoint and retain its buffers."""
        with self._provider_lock:
            return self._call_locked(call)

    def _call_locked(self, call: dict[str, Any]) -> dict[str, Any]:
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
        native_extra = view.get("extra")
        binding_metadata = binding.get("metadata")
        metadata_extra = {
            name: binding_metadata[name]
            for name in ("source_index", "feature_axes")
            if isinstance(binding_metadata, dict) and name in binding_metadata
        }
        allowed_extra = {
            "feature_set_id": binding.get("feature_set_id"),
            "include_augmented_cv_train_predictions": False,
        }
        feature_extra = {"feature_set_id": binding.get("feature_set_id")}
        refit_extra = {**feature_extra, "include_augmented_refit_predictions": False}
        allowed_phase_extra = (
            ({}, feature_extra, allowed_extra) if request.get("phase") == "FIT_CV"
            else ({}, feature_extra, refit_extra) if request.get("phase") == "REFIT"
            else ({}, feature_extra)
        )
        extra_valid = isinstance(native_extra, dict) and (
            native_extra in allowed_phase_extra
            or bool(metadata_extra) and native_extra in ({**base, **metadata_extra} for base in allowed_phase_extra)
        )
        unsupported = [
            name for name, invalid in (
                ("source_ids", view.get("source_ids") != binding.get("source_ids")),
                ("columns", view.get("columns") is not None),
                ("branch_view", view.get("branch_view") is not None),
                ("extra", not extra_valid),
                ("include_augmented", view.get("include_augmented") is not False
                 and not (view.get("include_augmented") is True
                          and view.get("partition") in {"fold_train", "full_train"})),
                ("include_excluded", view.get("include_excluded") is not False
                 and not (view.get("include_excluded") is True
                          and view.get("partition") in {"fold_validation", "predict"})),
            ) if invalid
        ]
        if unsupported:
            raise ValueError("Generated IO view does not support this native selector: " + ", ".join(unsupported))
        ids = view.get("sample_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
            raise ValueError("Native data-view request requires unique ordered sample IDs")
        key, seed = request.get("view_key"), request.get("view_seed")
        if not isinstance(key, str) or not key.strip() or type(seed) is not int or seed < 0:
            raise ValueError("Native data-view request requires a stable key and seed")
        selector = _selector_fields(view)
        binding_identity = {name: binding.get(name) for name in _BINDING_IDENTITY_FIELDS}
        prior = self._by_key.get(key)
        if prior is None:
            # Phase is intentionally absent: the native key can be reused
            # across execution phases for this same partition and selector.
            context = self._context_for(view)
            cohort = self._provider.materialize_view(ids, seed=seed, context=context, view_key=key)
            state = self._provider.view_state_dict(cohort)
            self._by_key[key] = (tuple(ids), seed, copy.deepcopy(selector), copy.deepcopy(binding_identity), cohort, state)
        else:
            prior_ids, prior_seed, prior_selector, prior_binding, cohort, state = prior
            if (prior_ids != tuple(ids) or prior_seed != seed
                    or prior_selector != selector or prior_binding != binding_identity):
                raise ValueError("Native data-view key was reused with a different selector or binding")
            if self._provider.view_state_dict(cohort) != state:
                raise ValueError("Generated data-view content changed after key reuse")
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
        with self._provider_lock:
            return self._resolve_locked(handle, sample_ids)

    def _resolve_locked(self, handle: dict[str, Any], sample_ids: list[str]) -> Any:
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
        with self._provider_lock:
            self._resolve_locked(handle, sample_ids)
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
        self._model_calls: dict[str, list[dict[str, Any]]] = {}
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
            raise ValueError(f"Generated task has no native view for {scope!r}; available={sorted(self._scopes)!r}")
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
                **({"model_calls": copy.deepcopy(self._model_calls[key])} if key in self._model_calls else {}),
            }
            for key, batches in sorted(self._read_batches.items())
        }

    def record_model_call(
        self, operation: str, input_name: str, partition: str, sample_ids: list[str],
        features: Any, *, options: dict[str, Any] | None = None, targets: Any = None,
    ) -> None:
        """Record the exact X/options and optional y supplied to a model call.

        This is host-reported boundary evidence, not native recomputation of
        Python-owned buffers. Each row must have been read from this task view.
        """
        if operation not in {"fit", "predict", "predict_proba"}:
            raise ValueError("Unknown generated model operation")
        if (operation == "fit") != (targets is not None):
            raise ValueError("Generated fit requires targets and prediction forbids them")
        if operation == "fit":
            phase = self._task.get("phase")
            required_partition = {"FIT_CV": "fold_train", "REFIT": "full_train"}.get(phase) if isinstance(phase, str) else None
            if partition != required_partition:
                raise ValueError("Generated fit requires its phase's native training partition")
        scope = (input_name, partition)
        key = self._scope_keys.get(scope)
        if key is None or not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Generated model call requires a native view and unique rows")
        def in_read_order(batch: list[str]) -> bool:
            wanted = iter(sample_ids)
            next_id = next(wanted, None)
            for seen in batch:
                if seen == next_id:
                    next_id = next(wanted, None)
                    if next_id is None:
                        return True
            return False
        if not any(in_read_order(batch) for batch in self._read_batches.get(key, [])):
            raise ValueError("Generated model call rows were not read in order from their native view")
        _require_model_rows(features, len(sample_ids), "features")
        _require_option_rows(options or {}, len(sample_ids))
        if targets is not None:
            _require_model_rows(targets, len(sample_ids), "targets")
        call: dict[str, Any] = {
            "operation": operation,
            "sample_ids": list(sample_ids),
            "input_fingerprint": _model_value_fingerprint({"features": features, "options": options or {}}),
        }
        if targets is not None:
            call["target_fingerprint"] = _model_value_fingerprint(targets)
        self._model_calls.setdefault(key, []).append(call)

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
