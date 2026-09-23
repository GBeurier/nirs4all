"""Replay independent no-CV source outputs through the captured DAG package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .envelope import build_envelope
from .general_replay import _prediction_content_fingerprint
from .identity import mint_identity
from .node_runner import _build_result, _stable_handle, _train_predict_ids

if TYPE_CHECKING:
    from collections.abc import Mapping


def predict_initial_refit_output(
    package: dict[str, Any], model: Any, spectro: Any, topology: Mapping[str, Any],
    output_binding_id: str, target_names: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run a fresh PREDICT cohort against a no-splitter REFIT package and its host sidecars."""
    import dag_ml

    identity = mint_identity(spectro)
    storage_ids = identity.observation_ids()
    if not storage_ids:
        raise ValueError("initial full-refit replay requires prediction rows")
    current_envelope = build_envelope(spectro, identity)
    envelope = dag_ml.attach_predict_cohort_to_envelope(package["training_envelope"], {
        "role": "inference", "relations": current_envelope["coordinator_relations"],
        "target_names": target_names,
        "data_content_fingerprint": _prediction_content_fingerprint(spectro),
        "target_content_fingerprint": None,
    }).to_dict()
    by_binding = {entry["output_binding_id"]: entry for entry in topology["outputs"]}
    selected = by_binding[output_binding_id]
    native_output_id = selected["dagml_output_id"]
    by_node = {entry["producer_node"]: entry for entry in topology["outputs"]}
    native_bindings = {entry["node_id"]: entry for entry in package["outputs"]}
    artifact_handles = {
        artifact["record"]["artifact"]["id"]: {
            "handle": _stable_handle(f"initial-refit:{package['package_fingerprint']}:{artifact['record']['artifact']['id']}"),
            "kind": "model", "owner_controller": artifact["record"]["controller_id"],
        }
        for artifact in package["artifacts"] if artifact["load_mode"] == "host_sidecar"
    }
    blocks = model._source_blocks(np.asarray(spectro.x({}, layout="2d")))  # noqa: SLF001 -- captured source topology
    positions = {sample_id: index for index, sample_id in enumerate(storage_ids)}

    def callback(task: dict[str, Any]) -> dict[str, Any]:
        if task["phase"] != "PREDICT":
            raise ValueError("initial full-refit package cannot execute a training phase during replay")
        node_id = task["node_plan"]["node_id"]
        entry = by_node.get(node_id)
        if entry is None:
            return _build_result(task, [], [], {})
        _, ids = _train_predict_ids(task)
        source_index = entry["source_index"]
        x = blocks[source_index][[positions[sample_id] for sample_id in ids]]
        prediction = np.asarray(model.members[source_index].predict_numeric(x), dtype=float).reshape(len(ids), -1)
        if prediction.shape != (len(ids), len(target_names)):
            raise ValueError("initial full-refit prediction width differs from target names")
        block = {
            "prediction_id": f"pred:{node_id}:initial:PREDICT", "producer_node": node_id,
            "producer_port": native_bindings[node_id]["port_name"],
            "partition": "final", "fold_id": None, "sample_ids": ids,
            "values": prediction.tolist(), "target_names": target_names,
        }
        return _build_result(task, [block], [], {})

    replay = dag_ml.replay_initial_full_refit_in_process(
        package, envelope, callback, artifact_handles, [native_output_id],
        f"run:nirs4all:initial-refit-predict:{package['package_fingerprint'][:16]}",
    )
    output = replay["replay_outcome"]["outputs"][0]
    if output["output_id"] != native_output_id:
        raise ValueError("initial full-refit replay returned a different output")
    prediction = output["prediction"]
    ids = prediction["sample_ids"]
    if len(ids) != len(set(ids)) or set(ids) != set(storage_ids) or prediction["target_names"] != target_names:
        raise ValueError("initial full-refit replay output identities or targets differ from input")
    by_id = dict(zip(ids, prediction["values"], strict=True))
    values = np.asarray([by_id[sample_id] for sample_id in storage_ids], dtype=float)
    from nirs4all.pipeline.dagml.target_capture import CapturedTargetTransform

    transform = model.members[selected["source_index"]].y_transform
    if isinstance(transform, CapturedTargetTransform):
        values = np.asarray(transform.decode(values))
    if len(target_names) == 1:
        values = values.ravel()
    return values, {
        "engine": "dag-ml", "execution_profile": "initial_full_refit_package_replay",
        "phase": "PREDICT", "package_fingerprint": package["package_fingerprint"],
        "replay_outcome": replay["replay_outcome"], "sample_ids": storage_ids,
        "target_names": target_names, "cross_validation": False, "training_performed": False,
    }
