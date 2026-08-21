"""Native Methods N4MM replay callbacks for a loaded DAG-ML package.

This module is deliberately narrow.  It is the Python host adapter for an
already-validated, portable ``n4m_model`` artifact: DAG-ML owns package,
identity, replay scheduling and output validation; :mod:`pls4all` owns N4MM
import and numerical prediction.  It neither decodes model bytes nor
reimplements PLS.

The callback pair matches ``dag_ml.replay_loaded_predictor_package``'s
``op_callback`` / ``artifact_callback`` contracts.  It supports a target-free
``PREDICT`` task only.  FIT/CV/REFIT, host sidecars, transforms and unknown
artifact kinds are intentionally refused rather than routed through legacy
``PipelineRunner``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .node_runner import _build_result, _train_predict_ids

if TYPE_CHECKING:
    from .resolver import MaterializationResolver


class MethodsPortableReplayError(RuntimeError):
    """A portable Methods replay cannot be executed by this host adapter."""


@dataclass
class _HydratedModel:
    context: Any
    model: Any

    def close(self) -> None:
        try:
            self.model.close()
        finally:
            self.context.close()


class MethodsN4mmReplayCallbacks:
    """Own invocation-local N4MM handles for a Methods-only PREDICT replay.

    ``fallback`` is used only for non-model graph nodes (the normal DAG-ML
    passthrough data nodes).  A model without one exact hydrated N4MM handle is
    always refused.  ``context_type`` and ``model_type`` are injectable solely
    for host-contract tests; production resolves them from the official
    ``pls4all`` binding lazily.
    """

    def __init__(
        self,
        resolver: MaterializationResolver,
        *,
        target_names_by_node: dict[str, list[str]],
        fallback: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        context_type: type[Any] | None = None,
        model_type: type[Any] | None = None,
    ) -> None:
        if (context_type is None) != (model_type is None):
            raise ValueError("context_type and model_type must be supplied together")
        if context_type is None:
            try:
                from pls4all import Context, Model
            except ImportError as error:  # pragma: no cover - depends on optional native wheel
                raise MethodsPortableReplayError(
                    "native Methods replay requires the published pls4all binding"
                ) from error
            context_type, model_type = Context, Model
        self._resolver = resolver
        self._target_names_by_node = {
            node_id: list(target_names)
            for node_id, target_names in target_names_by_node.items()
        }
        self._fallback = fallback
        self._context_type = context_type
        self._model_type = model_type
        self._next_handle = 1
        self._models: dict[int, _HydratedModel] = {}

    @property
    def active_handle_count(self) -> int:
        """Number of invocation-local N4MM imports currently retained."""

        return len(self._models)

    def artifact_callback(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Hydrate/release one raw N4MM payload on behalf of DAG-ML."""

        operation = event.get("operation")
        if operation == "hydrate":
            request = event.get("request")
            payload = event.get("payload")
            if not isinstance(request, dict) or not isinstance(payload, (bytes, bytearray, list)):
                raise MethodsPortableReplayError("invalid DAG-ML native artifact hydration event")
            artifact = request.get("artifact")
            controller_id = request.get("controller_id")
            if (
                not isinstance(artifact, dict)
                or artifact.get("kind") != "n4m_model"
                or not isinstance(controller_id, str)
            ):
                raise MethodsPortableReplayError(
                    "Methods portable replay only hydrates n4m_model artifacts"
                )
            try:
                raw_payload = bytes(payload)
            except (TypeError, ValueError) as error:
                raise MethodsPortableReplayError("N4MM payload is not byte-addressable") from error
            context = self._context_type()
            try:
                model = self._model_type.from_bytes(context, raw_payload)
            except BaseException:
                context.close()
                raise
            handle = self._next_handle
            self._next_handle += 1
            self._models[handle] = _HydratedModel(context=context, model=model)
            return {
                "handle": handle,
                "kind": "model",
                "owner_controller": controller_id,
            }
        if operation == "release":
            handle = event.get("handle")
            if not isinstance(handle, dict) or not isinstance(handle.get("handle"), int):
                raise MethodsPortableReplayError("invalid DAG-ML native artifact release event")
            hydrated = self._models.pop(handle["handle"], None)
            if hydrated is not None:
                hydrated.close()
            return None
        raise MethodsPortableReplayError("unknown DAG-ML native artifact callback operation")

    def op_callback(self, task: dict[str, Any]) -> dict[str, Any]:
        """Run one target-free PREDICT task using its exact N4MM handle."""

        node_plan = task.get("node_plan")
        if not isinstance(node_plan, dict):
            raise MethodsPortableReplayError("DAG-ML task is missing node_plan")
        if task.get("phase") != "PREDICT":
            raise MethodsPortableReplayError("Methods portable replay supports PREDICT only")
        if node_plan.get("kind") != "model":
            if self._fallback is None:
                raise MethodsPortableReplayError(
                    "Methods portable replay cannot execute a non-model node without a host callback"
                )
            return self._fallback(task)

        node_id = node_plan.get("node_id")
        controller_id = node_plan.get("controller_id")
        if not isinstance(node_id, str) or not isinstance(controller_id, str):
            raise MethodsPortableReplayError("DAG-ML model task has no stable node/controller identity")
        handles = task.get("input_handles")
        if not isinstance(handles, dict):
            raise MethodsPortableReplayError("DAG-ML model task has no artifact input handle map")
        candidates = [
            value
            for value in handles.values()
            if isinstance(value, dict)
            and value.get("owner_controller") == controller_id
            and value.get("kind") in {"model", "artifact"}
            and isinstance(value.get("handle"), int)
            and value["handle"] in self._models
        ]
        if len(candidates) != 1:
            raise MethodsPortableReplayError(
                "Methods portable PREDICT requires exactly one hydrated N4MM model handle"
            )
        _train_ids, sample_ids = _train_predict_ids(task)
        if not sample_ids:
            raise MethodsPortableReplayError("Methods portable PREDICT received no sample identities")
        features = self._resolver.resolve_features(sample_ids, include_augmented=False)["values"]
        values = np.asarray(self._models[candidates[0]["handle"]].model.predict(
            self._models[candidates[0]["handle"]].context,
            features,
        ), dtype=float)
        values = values.reshape(len(sample_ids), -1)
        target_names = self._target_names_by_node.get(node_id)
        if target_names is None or len(target_names) != values.shape[1]:
            raise MethodsPortableReplayError(
                "Methods portable PREDICT target schema does not match the N4MM model output"
            )
        prediction = {
            "prediction_id": f"pred:{node_id}:PREDICT:portable",
            "producer_node": node_id,
            "partition": "final",
            "fold_id": None,
            "sample_ids": sample_ids,
            "values": values.tolist(),
            "target_names": target_names,
        }
        return _build_result(task, [prediction], [], {}, [])


__all__ = ["MethodsN4mmReplayCallbacks", "MethodsPortableReplayError"]
