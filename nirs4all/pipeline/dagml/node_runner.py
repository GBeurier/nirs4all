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

import hashlib
from collections.abc import Callable, MutableMapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from sklearn.pipeline import make_pipeline

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


def _upstream_x_chain(node_id: str, edges: list[dict[str, Any]] | None) -> list[str]:
    """Ordered upstream node ids feeding ``node_id``'s ``x`` input (furthest-upstream first).

    Walks the linear data-edge chain backward so the model node can apply the real
    preprocessing (e.g. SNV) before fitting — the process-adapter delivers only sample_ids
    per node, so cross-node feature flow is reconstructed here. Linear chains only.
    """
    incoming: dict[str, str] = {}
    for edge in edges or []:
        if edge["target"]["port_name"] == "x" and edge["contract"]["kind"] == "data":
            incoming[edge["target"]["node_id"]] = edge["source"]["node_id"]
    chain: list[str] = []
    current = incoming.get(node_id)
    while current is not None:
        chain.append(current)
        current = incoming.get(current)
    chain.reverse()
    return chain


def _stable_handle(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "big") & 0x7FFFFFFFFFFFFFFF


def _output_handles(task: dict[str, Any], handle: int) -> dict[str, Any]:
    """Output port handles by node kind (model→out+oof, join→out+prediction, else out+x_out)."""
    controller_id = task["node_plan"]["controller_id"]
    kind = task["node_plan"].get("kind")
    outputs = {"out": {"handle": handle, "kind": "data", "owner_controller": controller_id}}
    if kind in ("model", "tuner"):
        outputs["oof"] = {"handle": handle, "kind": "prediction", "owner_controller": controller_id}
    elif kind == "prediction_join":
        outputs["prediction"] = {"handle": handle, "kind": "prediction", "owner_controller": controller_id}
    else:
        outputs["x_out"] = {"handle": handle, "kind": "data", "owner_controller": controller_id}
    return outputs


def _build_result(task: dict[str, Any], predictions: list[dict[str, Any]], artifacts: list[dict[str, Any]], artifact_handles: dict[str, Any], regression_targets: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Assemble the schema-complete ``NodeResult`` the runtime validates (outputs + full lineage).

    ``regression_targets`` (the real ``y_true`` for the predicted samples) lets dag-ml score the
    predictions natively into the bundle's ScoreSet — emitted only when we predict labelled samples.
    """
    node_plan = task["node_plan"]
    node_id = node_plan["node_id"]
    phase = task["phase"]
    variant_label = task.get("variant_id") or "base"
    fold_label = task.get("fold_id") or "nofold"
    metrics = {"nirs4all_adapter": 1.0}
    if predictions and predictions[0]["values"]:
        flat = [row[0] for row in predictions[0]["values"]]
        metrics["prediction_mean"] = float(sum(flat) / len(flat))
    return {
        "node_id": node_id,
        "outputs": _output_handles(task, _stable_handle(f"{node_id}:{phase}:{variant_label}:{fold_label}")),
        "predictions": predictions,
        "shape_deltas": [],
        "artifacts": artifacts,
        "artifact_handles": artifact_handles,
        "regression_targets": regression_targets or [],
        "lineage": {
            "record_id": f"lineage:{node_id}:{phase}:{variant_label}:{fold_label}",
            "run_id": task["run_id"],
            "node_id": node_id,
            "phase": phase,
            "controller_id": node_plan["controller_id"],
            "controller_version": node_plan["controller_version"],
            "variant_id": task.get("variant_id"),
            "fold_id": task.get("fold_id"),
            "branch_path": task.get("branch_path", []),
            "input_lineage": [],
            "artifact_refs": artifacts,
            "params_fingerprint": node_plan["params_fingerprint"],
            "data_model_shape_fingerprint": None,
            "aggregation_policy_fingerprint": None,
            "seed": task.get("seed"),
            "unsafe_flags": [],
            "metrics": metrics,
        },
    }


def run_model_node(
    task: dict[str, Any],
    resolver: MaterializationResolver,
    node_lookup: Callable[[str], dict[str, Any]],
    model_store: MutableMapping[int, Any],
    edges: list[dict[str, Any]] | None = None,
    y_transform_node: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a model-kind ``NodeTask`` with the real operator + real data; return a ``NodeResult``.

    When ``edges`` are supplied, the model node reconstructs its upstream X-transform chain
    (e.g. SNV→PLS) into a leakage-safe sklearn ``Pipeline`` fit on fold-train only — the
    process-adapter delivers only sample_ids per node, so the chain is applied here. A
    ``y_transform_node`` (a nirs4all ``y_processing`` step — a floating graph node with no edge
    to the model) is fit on fold-train ``y``, applied before fitting, and **inverse-transformed**
    over the predictions, exactly reproducing nirs4all target scaling.
    """
    node_plan = task["node_plan"]
    node_id = node_plan["node_id"]
    controller_id = node_plan["controller_id"]
    phase = task["phase"]
    variant_label = task.get("variant_id") or "base"
    fold_label = task.get("fold_id") or "nofold"

    train_ids, predict_ids = _train_predict_ids(task)
    artifact_id = _artifact_id(node_id, variant_label)
    artifact_handle = _stable_handle(artifact_id)

    estimator: Any
    y_transform: Any
    if phase == "PREDICT":
        bundle = model_store[artifact_handle]
        estimator, y_transform = bundle["estimator"], bundle["y_transform"]
    else:
        model = route_graph_node(node_lookup(node_id), variant_overrides=_variant_overrides(task, node_id))
        upstream = [route_graph_node(node_lookup(upstream_id)) for upstream_id in _upstream_x_chain(node_id, edges)]
        estimator = make_pipeline(*upstream, model) if upstream else model
        y_transform = route_graph_node(y_transform_node) if y_transform_node is not None else None
        x_train = np.asarray(resolver.resolve_features(train_ids)["values"], dtype=float)
        y_train = np.asarray(resolver.resolve_targets(train_ids)["values"], dtype=float)
        y_fit = y_transform.fit_transform(y_train.reshape(-1, 1)).ravel() if y_transform is not None else y_train
        estimator.fit(x_train, y_fit)

    x_predict = np.asarray(resolver.resolve_features(predict_ids)["values"], dtype=float)
    y_pred = np.asarray(estimator.predict(x_predict), dtype=float).reshape(len(predict_ids), -1)
    y_hat = np.asarray(y_transform.inverse_transform(y_pred), dtype=float).reshape(len(predict_ids), -1) if y_transform is not None else y_pred
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

    # Emit the real y_true (original scale) for the predicted samples so dag-ml scores natively.
    true_y = resolver.resolve_targets(predict_ids)["values"]
    regression_targets = [
        {
            "level": "sample",
            "unit_ids": [{"level": "sample", "id": sample_id} for sample_id in predict_ids],
            "values": [[float(value)] for value in true_y],
            "target_names": ["y"],
        }
    ]

    artifacts: list[dict[str, Any]] = []
    artifact_handles: dict[str, Any] = {}
    if phase == "REFIT":
        model_store[artifact_handle] = {"estimator": estimator, "y_transform": y_transform}
        artifacts.append({"id": artifact_id, "kind": "sklearn_estimator", "controller_id": controller_id, "backend": "joblib"})
        artifact_handles[artifact_id] = {"handle": artifact_handle, "kind": "model", "owner_controller": controller_id}

    return _build_result(task, predictions, artifacts, artifact_handles, regression_targets)


def run_node(
    task: dict[str, Any],
    resolver: MaterializationResolver,
    node_lookup: Callable[[str], dict[str, Any]],
    model_store: MutableMapping[int, Any],
    edges: list[dict[str, Any]] | None = None,
    y_transform_node: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch a ``NodeTask`` by node kind.

    ``model``/``tuner`` execute and (with ``edges``/``y_transform_node``) apply their upstream
    X-transform chain + target scaling; ``transform``/``y_transform`` are passthrough
    output-handles (the model node reconstructs the chain), so each preprocessing step is
    applied exactly once, at the model node.
    """
    kind = task["node_plan"]["kind"]
    if kind in ("model", "tuner"):
        return run_model_node(task, resolver, node_lookup, model_store, edges, y_transform_node)
    return _build_result(task, [], [], {})
