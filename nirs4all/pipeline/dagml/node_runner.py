"""Node runner — execute one dag-ml ``NodeTask`` with real nirs4all operators + data.

This is the heart of the host-controller side of the migration: it replaces the shipped
conformance adapter's hash-synthesized features with the **real** ``SpectroDataset`` rows
(via :class:`MaterializationResolver`) and the **real** operator (via ``route_graph_node``),
mirroring the dag-ml ``NodeTask``/``NodeResult`` JSON contract exactly (verified against
``dag-ml/examples/adapters/sklearn_process_controller.py``).

Frame contract (per phase): ``data_views`` is keyed by ``data:<input>`` with a
``partition`` — FIT_CV gets ``fold_train`` + a sibling ``…:validation`` (``fold_validation``)
and predicts the validation samples (``partition: validation``); REFIT gets ``full_train``,
fits, persists a model artifact, predicts the train samples (``partition: final``); PREDICT
reloads the artifact and predicts the ``predict`` view. The ``NodeTask`` carries ``node_plan``
(params, kind) but **not** the operator class, so the runner takes a ``node_lookup`` that maps
``node_id`` → the compiled graph node (which carries the operator) — both produced nirs4all
side by ``build_dagml_plan``.

**Scope (honest):** only ``model``/``tuner`` nodes produce predictions (matching dag-ml);
``transform``/``y_transform`` nodes are passthrough output-handles here. Real cross-node
feature chaining (e.g. SNV→PLS, where the model must see *transformed* features) is the
unresolved A3 gap — the process-adapter delivers only ``sample_ids`` per node, so a model
after a transform would re-fetch raw features. Until that is designed, this runner is
numerically correct for **model-on-raw-features** graphs.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .operator_routing import route_graph_node

if TYPE_CHECKING:
    from .resolver import MaterializationResolver

_PREDICTION_PARTITION = {"FIT_CV": "validation", "REFIT": "final", "PREDICT": "final", "EXPLAIN": "final"}


def _view_by_partition(task: dict[str, Any], partition: str) -> dict[str, Any] | None:
    for view in task.get("data_views", {}).values():
        if view.get("partition") == partition:
            return cast("dict[str, Any]", view)
    return None


def _sample_ids(view: dict[str, Any] | None) -> list[str]:
    if view is None or not view.get("sample_ids"):
        raise ValueError("data view is missing sample_ids")
    return list(view["sample_ids"])


def _variant_overrides(task: dict[str, Any], node_id: str) -> dict[str, Any]:
    """Collect per-node generated param overrides from the task's variant, if any."""
    overrides: dict[str, Any] = {}
    variant = task.get("variant")
    if variant is None:
        return overrides
    for choice in variant.get("choices", {}).values():
        for override in choice.get("param_overrides", []):
            if override.get("node_id") == node_id:
                overrides.update(override.get("params", {}))
    return overrides


def _train_predict_ids(task: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(train_sample_ids, predict_sample_ids) for the task's phase, by partition."""
    phase = task["phase"]
    if phase == "FIT_CV":
        return _sample_ids(_view_by_partition(task, "fold_train")), _sample_ids(_view_by_partition(task, "fold_validation"))
    if phase == "REFIT":
        train = _sample_ids(_view_by_partition(task, "full_train"))
        return train, train
    if phase == "PREDICT":
        return [], _sample_ids(_view_by_partition(task, "predict"))
    raise ValueError(f"unsupported phase {phase!r}")


def _artifact_id(node_id: str, variant_label: str) -> str:
    return f"artifact:{node_id}:nirs4all:refit:{variant_label}"


def run_model_node(
    task: dict[str, Any],
    resolver: MaterializationResolver,
    node_lookup: Callable[[str], dict[str, Any]],
    model_store: MutableMapping[str, Any],
) -> dict[str, Any]:
    """Execute a model-kind ``NodeTask`` with the real operator + real data; return a ``NodeResult``."""
    node_plan = task["node_plan"]
    node_id = node_plan["node_id"]
    controller_id = node_plan["controller_id"]
    phase = task["phase"]
    variant_label = task.get("variant_id") or "base"
    fold_label = task.get("fold_id") or "nofold"

    train_ids, predict_ids = _train_predict_ids(task)
    artifact_id = _artifact_id(node_id, variant_label)

    estimator: Any
    if phase == "PREDICT":
        estimator = model_store[task.get("replay_artifact_id", artifact_id)]
    else:
        estimator = route_graph_node(node_lookup(node_id), variant_overrides=_variant_overrides(task, node_id))
        x_train = np.asarray(resolver.resolve_features(train_ids)["values"], dtype=float)
        y_train = np.asarray(resolver.resolve_targets(train_ids)["values"], dtype=float)
        estimator.fit(x_train, y_train)

    x_predict = np.asarray(resolver.resolve_features(predict_ids)["values"], dtype=float)
    y_hat = np.asarray(estimator.predict(x_predict), dtype=float).reshape(len(predict_ids), -1)
    predictions = [
        {
            "prediction_id": f"pred:{node_id}:{phase}:{variant_label}:{fold_label}",
            "producer_node": node_id,
            "partition": _PREDICTION_PARTITION[phase],
            "fold_id": task.get("fold_id") if phase == "FIT_CV" else None,
            "sample_ids": predict_ids,
            "values": y_hat.tolist(),
            "target_names": ["y"],
        }
    ]

    artifacts: list[dict[str, Any]] = []
    artifact_handles: dict[str, Any] = {}
    if phase == "REFIT":
        model_store[artifact_id] = estimator
        artifacts.append({"id": artifact_id, "kind": "sklearn_estimator", "controller_id": controller_id, "backend": "joblib"})
        artifact_handles[artifact_id] = {"handle": artifact_id, "kind": "model", "owner_controller": controller_id}

    return {
        "node_id": node_id,
        "predictions": predictions,
        "artifacts": artifacts,
        "artifact_handles": artifact_handles,
        "lineage": {
            "record_id": f"lineage:{node_id}:{phase}:{variant_label}:{fold_label}",
            "run_id": task.get("run_id"),
            "node_id": node_id,
            "phase": phase,
            "controller_id": controller_id,
            "fold_id": task.get("fold_id"),
            "metrics": {"nirs4all_adapter": 1.0},
        },
    }


def run_node(
    task: dict[str, Any],
    resolver: MaterializationResolver,
    node_lookup: Callable[[str], dict[str, Any]],
    model_store: MutableMapping[str, Any],
) -> dict[str, Any]:
    """Dispatch a ``NodeTask`` by node kind.

    ``model``/``tuner`` execute and emit predictions; ``transform``/``y_transform`` are
    passthrough (no predictions) — real cross-node feature chaining is the deferred A3 gap.
    """
    kind = task["node_plan"]["kind"]
    if kind in ("model", "tuner"):
        return run_model_node(task, resolver, node_lookup, model_store)
    return {"node_id": task["node_plan"]["node_id"], "predictions": [], "artifacts": [], "artifact_handles": {}}
