"""Attest fixed IO cohort tables through the native data-view callback protocol.

This provider selects existing rows; it never generates data. Each receipt is
issued after reading its actual named source buffer and rechecked before use.
"""

from __future__ import annotations

import copy
import threading
from typing import Any, cast

import numpy as np

from .generated_views import (
    _BINDING_IDENTITY_FIELDS,
    GeneratedTaskViews,
    GeneratedViewStore,
    _model_value_fingerprint,
    _selector_fields,
)
from .resolver import MaterializationResolver
from .tuning_contracts import tcv1_sha256


class FixedCohortViewStore:
    """Retain real source tables behind native handles for one fixed-cohort run."""

    def __init__(self, resolver: MaterializationResolver, model_input: dict[str, Any], envelope: dict[str, Any]) -> None:
        from nirs4all.data.multimodal import MultimodalSpectroDataset

        from .envelope import source_ids, source_order

        dataset = resolver._dataset  # noqa: SLF001 -- this provider shares the scientific host's resolver
        if not isinstance(dataset, MultimodalSpectroDataset) or getattr(dataset, "_generated_view_store", None) is not None:
            raise ValueError("fixed named views require an explicit complete IO cohort")
        if resolver._fold_feature_views is not None or resolver._augmented_observation_ids:  # noqa: SLF001 -- fixed views cannot substitute host rows
            raise ValueError("fixed named views cannot contain augmented or fold-transformed source rows")
        self._resolver = resolver
        self._ports = {port["name"]: copy.deepcopy(port) for port in model_input["ports"]}
        names, native_ids = source_order(dataset), source_ids(dataset)
        layout = envelope["plan"]["source_layout"]
        if (len(self._ports) != len(model_input["ports"]) or set(self._ports) != set(names)
                or layout.get("source_order") != names or layout.get("source_ids") != native_ids):
            raise ValueError("fixed named views disagree with the native cohort source layout")
        self._source_indices = {name: index for index, name in enumerate(names)}
        self._source_ids = dict(zip(names, native_ids, strict=True))
        if any(port["metadata"]["source_id"] != self._source_ids[name] for name, port in self._ports.items()):
            raise ValueError("fixed named views disagree with captured source IDs")
        self._envelope_identity = {key: envelope.get(key) for key in ("schema_fingerprint", "plan_fingerprint", "relation_fingerprint")}
        self._views: dict[int, dict[str, Any]] = {}
        self._by_key: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def _read(self, name: str, ids: list[str]) -> tuple[np.ndarray, str, str]:
        from nirs4all.data.multimodal import MultimodalSpectroDataset

        resolved = self._resolver.resolve_source_block(ids, self._source_indices[name], include_augmented=False)
        values = np.asarray(resolved["values"])
        metadata = self._ports[name]["metadata"]
        if (resolved.get("observation_ids") != ids or values.shape != (len(ids), *metadata["feature_shape"])
                or str(values.dtype) != metadata["dtype"] or not np.isfinite(values).all()):
            raise ValueError("fixed named view changed its actual ordered rows, dtype, shape or finite values")
        dataset = self._resolver._dataset  # noqa: SLF001 -- IO owns the source descriptors and buffer fingerprint
        if not isinstance(dataset, MultimodalSpectroDataset):
            raise ValueError("fixed named view lost its original explicit IO cohort")
        descriptor = dataset.cohort.sources[name].schema_descriptor(name)
        schema_fingerprint = tcv1_sha256({"source_id": self._source_ids[name], "source": descriptor, "port": self._ports[name]})
        content_fingerprint = _model_value_fingerprint({
            "source_id": self._source_ids[name], "source_name": name, "sample_ids": ids, "features": values,
            "io_source_fingerprint": dataset.content_hash(source_index=self._source_indices[name]),
        })
        return values, schema_fingerprint, content_fingerprint

    def __call__(self, call: dict[str, Any]) -> dict[str, Any]:
        """Materialize a scheduler-selected source before emitting its receipt."""
        with self._lock:
            if not isinstance(call, dict) or set(call) != {"request", "handle"}:
                raise ValueError("fixed named callback requires a native request and handle")
            request, handle = call["request"], call["handle"]
            if (not isinstance(request, dict) or not isinstance(handle, dict) or handle.get("kind") != "data_view"
                    or type(handle.get("handle")) is not int or handle["handle"] <= 0 or handle["handle"] in self._views):
                raise ValueError("fixed named callback requires a fresh positive native data-view handle")
            name = request.get("input_name")
            binding, view = request.get("binding"), request.get("view")
            if not isinstance(name, str) or name not in self._ports or not isinstance(binding, dict) or not isinstance(view, dict):
                raise ValueError("fixed named callback requires its declared source port")
            if (binding.get("input_name") != name or binding.get("source_ids") != [self._source_ids[name]]
                    or view.get("source_ids") != binding["source_ids"] or binding.get("output_representation") != self._ports[name]["accepted_representations"][0]
                    or any(binding.get(key) != value for key, value in self._envelope_identity.items())
                    or (binding.get("metadata") or {}).get("source_name") != name):
                raise ValueError("fixed named callback changed source, representation or envelope binding")
            phase, partition = request.get("phase"), view.get("partition")
            # Native FIT_CV supplies the external Test companion as a non-fit
            # predict view. It keeps its own receipt/rows; model-call validation
            # still permits FIT only on fold_train (full_train for REFIT).
            allowed = {"FIT_CV": {"fold_train", "fold_validation", "predict"}, "REFIT": {"full_train", "predict"}, "PREDICT": {"predict"}}
            extra = view.get("extra")
            if (not isinstance(phase, str) or not isinstance(partition, str) or partition not in allowed.get(phase, set())
                    or view.get("columns") is not None or view.get("branch_view") is not None
                    or not isinstance(extra, dict) or set(extra) - {"feature_set_id", "include_augmented_cv_train_predictions", "include_augmented_refit_predictions"}
                    or extra.get("feature_set_id", binding.get("feature_set_id")) != binding.get("feature_set_id")
                    or any(value is not False for key, value in extra.items() if key != "feature_set_id")
                    or type(view.get("include_augmented")) is not bool or type(view.get("include_excluded")) is not bool
                    or view.get("include_augmented") is True and partition not in {"fold_train", "full_train"}
                    or view.get("include_excluded") is True and partition not in {"fold_validation", "predict"}):
                raise ValueError("fixed named callback does not support this source selector")
            ids, key, seed = view.get("sample_ids"), request.get("view_key"), request.get("view_seed")
            if (not isinstance(ids, list) or not ids or any(not isinstance(item, str) or not item for item in ids)
                    or len(ids) != len(set(ids)) or not isinstance(key, str) or not key.strip() or type(seed) is not int or seed < 0):
                raise ValueError("fixed named callback requires ordered unique sample IDs and a native key/seed")
            values, schema_fingerprint, content_fingerprint = self._read(name, ids)
            identity = {"sample_ids": ids, "seed": seed, "selector": _selector_fields(view),
                        "binding": {field: binding.get(field) for field in _BINDING_IDENTITY_FIELDS},
                        "schema_fingerprint": schema_fingerprint, "content_fingerprint": content_fingerprint}
            if key in self._by_key and self._by_key[key] != identity:
                raise ValueError("fixed named view key changed its selector, source or actual content")
            self._by_key[key] = copy.deepcopy(identity)
            receipt = {"handle": copy.deepcopy(handle), "view_key": key, "sample_ids": list(ids),
                       "schema_fingerprint": schema_fingerprint, "content_fingerprint": content_fingerprint}
            retained = np.array(values, copy=True, order="C")
            retained.setflags(write=False)
            self._views[handle["handle"]] = {"request": copy.deepcopy(request), "receipt": copy.deepcopy(receipt),
                                             "values": retained, "buffer_fingerprint": _model_value_fingerprint(retained)}
            return receipt

    def _resolve_values(self, handle: dict[str, Any], ids: list[str]) -> tuple[str, np.ndarray]:
        if not isinstance(handle, dict) or type(handle.get("handle")) is not int:
            raise ValueError("fixed named read requires its native handle")
        record = self._views.get(handle["handle"])
        if record is None or record["receipt"]["handle"] != handle or record["receipt"]["sample_ids"] != ids:
            raise ValueError("fixed named read changed its handle or ordered sample IDs")
        name = record["request"]["input_name"]
        _, schema_fingerprint, content_fingerprint = self._read(name, ids)
        if (record["receipt"]["schema_fingerprint"] != schema_fingerprint or record["receipt"]["content_fingerprint"] != content_fingerprint
                or record["buffer_fingerprint"] != _model_value_fingerprint(record["values"])):
            raise ValueError("fixed named view changed its IO schema or actual buffers after receipt")
        return name, record["values"]

    def _task_binding_for(self, handle: dict[str, Any], ids: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._lock:
            self._resolve_values(handle, ids)
            record = self._views[handle["handle"]]
            return copy.deepcopy(record["request"]), copy.deepcopy(record["receipt"])

    def bind_task(self, task: dict[str, Any]) -> FixedCohortTaskViews:
        """Validate every native receipt before the scientific callback runs."""
        with self._lock:
            return FixedCohortTaskViews(self, task)


class FixedCohortTaskViews(GeneratedTaskViews):
    """Reuse native task/receipt validation and reads ledger for fixed tables."""

    def __init__(self, store: FixedCohortViewStore, task: dict[str, Any]) -> None:
        # GeneratedTaskViews uses this structural store interface, without an
        # IO generator: _task_binding_for attests each handle's real buffer.
        super().__init__(cast(GeneratedViewStore, store), task)
        bindings = task["node_plan"].get("data_bindings")
        if not isinstance(bindings, list) or len(bindings) != len(store._ports):  # noqa: SLF001 -- every fixed port needs its native binding
            raise ValueError("fixed named task requires exactly one binding per source port")
        for (name, _), (handle, ids) in self._scopes.items():
            request, _ = store._task_binding_for(handle, ids)  # noqa: SLF001 -- compare the materialized native binding before numerical use
            matched = [binding for binding in bindings if isinstance(binding, dict) and binding.get("input_name") == name]
            if len(matched) != 1 or matched[0] != request["binding"]:
                raise ValueError("fixed named task changed its materialized source binding")

    def take(self, input_name: str, partition: str, sample_ids: list[str]) -> Any:
        raise ValueError("fixed named views expose only their attested source table")

    def consumed_data_views(self) -> dict[str, dict[str, Any]]:
        """Reattest the fixed buffers before reporting successful model reads."""
        store = cast(FixedCohortViewStore, self._store)
        with store._lock:  # noqa: SLF001 -- validate the shared receipt registry after the callback
            for handle, ids in self._scopes.values():
                store._resolve_values(handle, ids)  # noqa: SLF001 -- prevent a mutated buffer from becoming successful evidence
        return super().consumed_data_views()

    def feature_blocks(
        self, input_name: str, partition: str, sample_ids: list[str], *, source_names: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Read the complete ordered single-source buffer attached to this port."""
        scope = (input_name, partition)
        if scope not in self._scopes or source_names != (input_name,):
            raise ValueError("fixed named read requires its exact port, source and partition")
        handle, full_ids = self._scopes[scope]
        if sample_ids != full_ids:
            raise ValueError("fixed named read cannot substitute its native ordered rows")
        store = cast(FixedCohortViewStore, self._store)
        with store._lock:  # noqa: SLF001 -- this bound reader shares its provider's receipt registry
            name, values = store._resolve_values(handle, full_ids)  # noqa: SLF001 -- reattest before exposing the buffer
        if name != input_name:
            raise ValueError("fixed named handle belongs to another source port")
        self._read_batches.setdefault(self._scope_keys[scope], []).append(list(sample_ids))
        binding = self._task["node_plan"]["data_bindings"]
        feature_set_id = next(item["feature_set_id"] for item in binding if item["input_name"] == name)
        return {"feature_set_id": feature_set_id, "observation_ids": list(sample_ids), "source_names": (name,), "blocks": [values]}
