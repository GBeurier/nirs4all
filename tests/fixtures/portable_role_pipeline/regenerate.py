"""Requalify the historical five-model capture with the installed native cohort.

Run with the original qualification receipt as the sole training-data source.
DAG-ML owns scoring, selection and contract fingerprints; Methods owns fits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import dag_ml
import numpy as np
from n4m.roles import RolePipeline

SOURCE_SHA256 = "6191402f61ef5e0227ef1bd47ed44eac5f6721cbcc8d4509ed0614f0737aae35"


class _CaptureController:
    """Execute the original numeric Ridge recipes under their signed manifest."""

    def __init__(self, source: dict[str, Any]) -> None:
        self.ids = source["sampleIds"]
        self.rows = source["sourceRows"]
        self.targets = dict(zip(self.ids, source["target"], strict=True))
        self.models: dict[int, Any] = {}
        self.payloads: dict[str, bytes] = {}

    def features(self, task: dict[str, Any], *, outer: bool = False) -> tuple[list[str], np.ndarray, list[str]]:
        blocks: list[Any] = []
        names: list[str] = []
        if task.get("prediction_inputs"):
            suffix = ":outer" if task["phase"] == "FIT_CV" else ":refit"
            entries = [(key, value) for key, value in sorted(task["prediction_inputs"].items()) if key.endswith(suffix) == outer]
            sample_ids = entries[0][1]["sample_ids"]
            for key, value in entries:
                assert value["sample_ids"] == sample_ids
                if not outer:
                    assert value["partition"] == "validation"
                    if task["phase"] == "FIT_CV":
                        assert task["fold_id"] not in value.get("fold_ids", [])
                blocks.append(value["values"])
                names.extend(f"{key.removesuffix(suffix)}/{column}" for column in range(value["prediction_width"]))
        else:
            partition = "fold_validation" if outer else "fold_train" if task["phase"] == "FIT_CV" else "full_train"
            entries = [(key, view) for key, view in sorted(task["data_views"].items()) if view["partition"] == partition]
            sample_ids = entries[0][1]["sample_ids"]
            for key, view in entries:
                assert view["sample_ids"] == sample_ids
                [source_id] = view["source_ids"]
                block = np.asarray([self.rows[source_id][self.ids.index(sample)] for sample in sample_ids])
                blocks.append(block)
                names.extend(f"{key.removesuffix(':validation')}/{column}" for column in range(block.shape[1]))
        return sample_ids, np.column_stack(blocks), names

    def operator(self, task: dict[str, Any]) -> dict[str, Any]:
        node, phase = task["node_plan"], task["phase"]
        assert phase in {"FIT_CV", "REFIT"}
        assert node["controller_id"] == "controller:methods.wasm.regression"
        train_ids, X, names = self.features(task)
        steps = [{"class": "n4m:models.regularized.ridge", "params": node["params"]}]
        model = RolePipeline(steps).fit(X, np.asarray([self.targets[sample] for sample in train_ids]))
        artifacts, handles = [], {}
        try:
            if phase == "FIT_CV":
                sample_ids, valid, valid_names = self.features(task, outer=True)
                assert set(train_ids).isdisjoint(sample_ids)
            elif task.get("prediction_inputs") and any(key.endswith(":refit") for key in task["prediction_inputs"]):
                sample_ids, valid, valid_names = self.features(task, outer=True)
            else:
                sample_ids, valid, valid_names = train_ids, X, names
            assert names == valid_names
            if phase == "REFIT":
                saved = {
                    "schema": "dagml.methods.regression.v1", "node_id": node["node_id"],
                    "params_fingerprint": node["params_fingerprint"], "target_names": ["y"],
                    "steps": steps, "feature_names": names,
                    "states": [list(state) for _method, state, _training_rows in model.export_states()],
                }
                payload = json.dumps(saved, separators=(",", ":")).encode()
                fingerprint = hashlib.sha256(payload).hexdigest()
                artifact_id = ":".join(["artifact:methods", task["run_id"], node["node_id"], task["variant_id"] or "base", "refit"])
                artifacts = [{
                    "id": artifact_id, "kind": "methods_role_pipeline", "controller_id": node["controller_id"],
                    "backend": "raw", "uri": f"artifacts/{fingerprint}.json", "content_fingerprint": fingerprint,
                    "size_bytes": len(payload), "plugin": "dagml.methods.wasm.regression", "plugin_version": node["controller_version"],
                }]
                handle = len(self.models) + 1
                handles[artifact_id] = {"handle": handle, "kind": "model", "owner_controller": node["controller_id"]}
                self.models[handle] = model
                self.payloads[artifact_id] = payload
            return {
                "node_id": node["node_id"], "outputs": {}, "artifacts": artifacts, "artifact_handles": handles,
                "predictions": [{
                    "producer_node": node["node_id"], "partition": "validation" if phase == "FIT_CV" else "final",
                    "fold_id": task["fold_id"], "sample_ids": sample_ids,
                    "values": np.asarray(model.predict(valid)).reshape(-1, 1).tolist(), "target_names": ["y"],
                }],
                "regression_targets": [{
                    "level": "sample", "unit_ids": [{"level": "sample", "id": sample} for sample in sample_ids],
                    "values": [[self.targets[sample]] for sample in sample_ids], "target_names": ["y"],
                }] if phase == "FIT_CV" else [],
                "lineage": {
                    "record_id": ":".join(["lineage:methods-wasm", task["run_id"], node["node_id"], phase, task["variant_id"] or "base", task["fold_id"] or "full"]),
                    "run_id": task["run_id"], "node_id": node["node_id"], "phase": phase,
                    "controller_id": node["controller_id"], "controller_version": node["controller_version"],
                    "variant_id": task["variant_id"], "fold_id": task["fold_id"], "branch_path": task["branch_path"],
                    "input_lineage": [], "artifact_refs": artifacts, "params_fingerprint": node["params_fingerprint"],
                    "data_model_shape_fingerprint": None, "aggregation_policy_fingerprint": None,
                    "seed": task["seed"], "unsafe_flags": [], "metrics": {}, "loss_attestations": [], "early_stopping_records": [],
                },
            }
        finally:
            if model not in self.models.values():
                model.close()

    def artifact(self, event: dict[str, Any]) -> bytes:
        assert event["operation"] == "export"
        return self.payloads[event["artifact_id"]]

    def close(self) -> None:
        for model in self.models.values():
            model.close()


def regenerate(source_path: Path, output_path: Path) -> None:
    raw = source_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("original qualification receipt SHA-256 disagrees")
    source = json.loads(raw)
    archive = source["completeArchive"]
    historical = json.loads(archive["outcomeJson"])
    controller = _CaptureController(source)
    training = None
    try:
        training = dag_ml.execute_training(
            archive["requestJson"], archive["trainingEnvelopes"], source["envelope"]["coordinator_relations"],
            historical["training_influence"], controller.operator, artifact_callback=controller.artifact,
            outcome_id=historical["outcome_id"], run_id=historical["run_id"], bundle_id=historical["execution_bundle"]["bundle_id"],
        )
        current = training.outcome.to_dict()
        assert current["selected_variant_id"] == historical["selected_variant_id"]
        for before, after in zip(historical["oof_averages"], current["oof_averages"], strict=True):
            assert before["predictions"]["producer_node"] == after["predictions"]["producer_node"]
            assert before["predictions"]["unit_ids"] == after["predictions"]["unit_ids"]
            np.testing.assert_allclose(after["predictions"]["values"], before["predictions"]["values"], rtol=1e-12, atol=1e-12)
        package = training.export_portable_predictor_package(
            json.loads(archive["packageJson"])["package_id"], fitted_artifact_mode="portable_required", artifact_load_mode="native_portable",
        )
        replay = json.loads(archive["replayRequestJson"])
        replay["source_outcome_fingerprint"] = current["outcome_fingerprint"]
        # The installed facade exports this API; its shipped stub omits it.
        sign_replay = getattr(dag_ml, "sign_training_replay_request")
        capture = {
            "training_request_json": archive["requestJson"], "package_json": package.json(), "outcome_json": training.outcome_json(),
            "trusted_manifests": [source["manifest"]], "data_envelopes": archive["predictEnvelopes"],
            "replay_request_json": sign_replay(replay).json(), "heldout_rows": archive["heldoutRows"],
            "expected_prediction": archive["replay"]["outputs"][0]["predictions"][0],
        }
        output_path.write_text(json.dumps(capture, separators=(",", ":")) + "\n")
    finally:
        if training is not None:
            training.detach()
        controller.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    regenerate(args.source_receipt, args.output)
