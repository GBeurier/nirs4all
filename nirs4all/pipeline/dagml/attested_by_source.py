"""Native multi-output training captures and their exact host-sidecar bindings."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import dag_ml
import numpy as np

from nirs4all.core.metrics import is_higher_better
from nirs4all.pipeline.dagml_bridge import controller_manifests

from .identity import IdentityMap
from .in_process_runner import _capture_refit_artifacts, _load_subprocess_refit_artifacts
from .node_runner import run_node
from .raw_training_lowerer import _array_content_fingerprint, _core_relation_fingerprint, _data_contracts_from_campaign, _output_request_for_node, _training_influence_manifest
from .resolver import MaterializationResolver
from .training_contracts import DagMLTrainingRequestSpec, assemble_training_request


def bind_source_output_contract(dsl: dict[str, Any], dataset: Any, identity: IdentityMap, sample_ints: list[int], source_names: list[str]) -> None:
    """Sign the public typed vocabulary and experimental-unit design before compilation."""
    dsl.setdefault("metadata", {})["by_source_source_order"] = list(source_names)
    from .envelope import _numeric_feature_axis

    blocks = dataset.x({"sample": sample_ints}, "3d", concat_source=False)
    widths = [int(np.asarray(block).reshape(len(sample_ints), -1).shape[1]) for block in blocks]
    if len(widths) != len(source_names) or any(width <= 0 for width in widths):
        raise ValueError("source input widths must align with the declared source order")
    dsl["metadata"]["by_source_input_widths"] = widths
    dsl["metadata"]["by_source_feature_axes"] = [_numeric_feature_axis(dataset, index) for index in range(len(widths))]
    resolver = MaterializationResolver(dataset, identity)
    target = resolver.resolve_targets([identity.to_wire(sample) for sample in sample_ints])
    from .envelope import target_names
    from .experimental_units import apply_experimental_unit_contract

    names = target.get("target_names", ["y"] if len(target_names(dataset)) == 1 else target_names(dataset))
    apply_experimental_unit_contract(dsl, dataset, identity, sample_ints, target_values=target["values"], target_names=names)
    if not dataset.is_classification:
        return
    matrix = np.asarray(target["values"], dtype=float).reshape(len(sample_ints), -1)
    if matrix.shape[1] != 1 or not np.isfinite(matrix).all():
        raise ValueError("independent classification requires complete mono-y labels")
    native_ids = np.unique(matrix[:, 0]).tolist()
    decoder = resolver.target_decoder()
    from .target_capture import CapturedTargetTransform

    labels = getattr(CapturedTargetTransform(None, decoder), "classes_", None)
    if labels is None:
        # No external mapping: native labels themselves remain numeric, as in the original callback.
        public = native_ids
        kind = "native_numeric"
    else:
        public = [label.item() if isinstance(label, np.generic) else label for label in labels]
        if len(public) != len(native_ids) or native_ids != list(map(float, range(len(public)))):
            raise ValueError("typed class vocabulary differs from the native encoded label columns")
        if all(type(label) is str for label in public):
            kind = "str"
        elif all(type(label) is int and -(1 << 63) <= label < (1 << 63) for label in public):
            kind = "int64"
        else:
            raise ValueError("public class labels must be homogeneous strings or signed int64 integers")
    dsl.setdefault("metadata", {})["by_source_class_labels"] = {
        "schema_version": 1, "label_type": kind, "native_ids": native_ids, "labels": public,
    }


def _source_dsl_nodes(steps: list[dict[str, Any]]) -> Any:
    for step in steps:
        yield step
        for branch in step.get("branches", []):
            yield from _source_dsl_nodes(branch.get("steps", []))
        yield from _source_dsl_nodes(step.get("tail", []))


def _native_selection(outcome: dict[str, Any]) -> dict[str, Any]:
    selections = outcome["execution_bundle"]["selections"]
    if not isinstance(selections, dict) or len(selections) != 1:
        raise ValueError("independent source training requires exactly one native selection decision")
    return next(iter(selections.values()))


def validate_source_package_bindings(packages: list[dict[str, Any]], artifacts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Verify exact original native artifact identities before touching sidecar bytes."""
    records: dict[str, dict[str, Any]] = {}
    for package in packages:
        dag_ml.PortablePredictorPackage(package)
        bindings = package["artifact_bindings"]
        refits = package["execution_bundle"]["refit_artifacts"]
        by_id = {record["artifact"]["id"]: record for record in refits}
        if len(by_id) != len(refits) or len(bindings) != len(by_id) or {item["artifact_id"] for item in bindings} != set(by_id):
            raise ValueError("source package bindings must biject with original native REFIT records")
        for binding in bindings:
            artifact_id = binding["artifact_id"]
            if binding["load_mode"] != "host_sidecar" or artifact_id in records:
                raise ValueError("source package requires unique host-sidecar bindings across selected ranks")
            records[artifact_id] = by_id[artifact_id]
    actual = {item["artifact_id"]: item for item in artifacts}
    if len(actual) != len(artifacts) or set(actual) != set(records):
        raise ValueError("source sidecars must biject with PackageArtifactBinding, without missing or extra models")
    for artifact_id, record in records.items():
        item = actual[artifact_id]
        descriptor = record["artifact"]
        if (item.get("controller_id") != descriptor["controller_id"] or item.get("kind") != descriptor["kind"]
                or item.get("producer_node", record["node_id"]) != record["node_id"]):
            raise ValueError("source sidecar producer/controller differs from its original native REFIT")
    return records


def validate_source_archive_before_model(archive: Any, manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Validate packages and every declared carrier hash before unpickling a model."""
    ref = manifest.get("dagml_source_training_capture_ref")
    member = "dagml_source_training_capture.json"
    if ref is None:
        if member in archive.namelist():
            raise ValueError("archive has an undeclared source training capture")
        return None
    if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"} or ref["path"] != member
            or archive.getinfo(member).file_size > 64 * 1024 * 1024):
        raise ValueError("archive source training capture is invalid or oversized")
    raw = archive.read(member)
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("archive source training capture hash differs")
    capture = json.loads(raw)
    if not isinstance(capture, dict) or set(capture) != {"schema_version", "captures", "sidecars", "outputs"} or capture["schema_version"] != 1:
        raise ValueError("archive source training capture schema is invalid")
    if not isinstance(capture["captures"], list) or not capture["captures"]:
        raise ValueError("archive requires original nonempty native training captures")
    packages = []
    for index, item in enumerate(capture["captures"], 1):
        if not isinstance(item, dict) or set(item) != {"outcome", "package"}:
            raise ValueError("archive source capture must retain original outcome and package")
        outcome, package = item["outcome"], item["package"]
        dag_ml.TrainingOutcome(outcome)
        dag_ml.PortablePredictorPackage(package)
        if (package["training_outcome"]["outcome_fingerprint"] != outcome["outcome_fingerprint"]
                or package["execution_bundle"] != outcome["execution_bundle"]
                or _native_selection(outcome).get("requested_rank", 1) != index):
            raise ValueError("archive package does not match the original ranked training outcome")
        packages.append(package)
    sidecars = capture["sidecars"]
    records = validate_source_package_bindings(packages, sidecars)
    from .native_results import _validate_portable_uri

    paths = set()
    for item in sidecars:
        path = _validate_portable_uri(item.get("uri"))
        expected_path = f"dagml_source_sidecar_{hashlib.sha256(item['artifact_id'].encode()).hexdigest()}.joblib"
        if path != expected_path or path in paths or item.get("native_record") != records[item["artifact_id"]]:
            raise ValueError("archive source sidecar mapping differs from native records")
        paths.add(path)
        info = archive.getinfo(path)
        if info.file_size != item.get("size_bytes") or info.file_size > 512 * 1024 * 1024:
            raise ValueError("archive source sidecar has invalid size")
        digest = hashlib.sha256()
        with archive.open(path) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != item.get("serialization_sha256"):
            raise ValueError("archive source sidecar hash differs")
    if {name for name in archive.namelist() if name.startswith("dagml_source_sidecar_")} != paths:
        raise ValueError("archive contains undeclared source sidecars")
    expected_outputs = []
    for rank, package in enumerate(packages, 1):
        for binding in package["output_bindings"]:
            expected_outputs.append({"rank": rank, "output_id": binding["binding_id"], "producer_node": binding["node_id"],
                                     "prediction_kind": binding["prediction_kind"], "target_names": binding["target_names"],
                                     "class_labels": binding["class_labels"]})
    if capture["outputs"] != expected_outputs:
        raise ValueError("archive named output contract differs from its native packages")
    expanded = []
    for native, package in zip(expected_outputs, [item for item in packages for _ in item["output_bindings"]], strict=True):
        graph = package["effective_plan"]["graph_plan"]["graph"]
        node = next(node for node in graph["nodes"] if node["id"] == native["producer_node"])
        index = node.get("metadata", {}).get("source_index")
        order = graph.get("metadata", {}).get("by_source_source_order")
        if type(index) is not int or not isinstance(order, list) or not 0 <= index < len(order) or native["output_id"] != f"output:source_{index}":
            raise ValueError("archive output source identity differs from its signed graph")
        members = [artifact_id for artifact_id, record in records.items() if record["node_id"] == node["id"]
                   and artifact_id in {binding["artifact_id"] for binding in package["artifact_bindings"]}]
        if len(members) != 1:
            raise ValueError("archive output must bind exactly one original native sidecar")
        expanded.append({**native, "source_index": index, "source_id": order[index], "artifact_id": members[0],
                         "output_binding_id": native["output_id"] if native["rank"] == 1 else f"{native['output_id']}:rank:{native['rank']}"})
    topology = manifest.get("dagml_independent_output_topology")
    if topology is not None:
        metadata = packages[0]["effective_plan"]["graph_plan"]["graph"]["metadata"]
        widths, axes = metadata["by_source_input_widths"], metadata["by_source_feature_axes"]
        expected_sources = [{"source_id": name, "source_index": index, "output_binding_id": f"output:source_{index}", "feature_width": widths[index],
                             **({"feature_axis_cm1": axes[index]} if axes[index] is not None else {})}
                            for index, name in enumerate(metadata["by_source_source_order"])]
        if topology.get("outputs") != expected_sources:
            raise ValueError("archive source input contract differs from its signed graph")
    if topology is not None and topology.get("ranked_outputs") != expanded:
        raise ValueError("archive ranked output topology differs from native package identities")
    selected = manifest.get("dagml_selected_source")
    if topology is None and selected not in expanded:
        raise ValueError("archive selected output differs from native package identities")
    return capture


def _cli_training(*, request: dict[str, Any], envelopes: dict[str, Any], relations: dict[str, Any], influence: dict[str, Any],
                  graph: dict[str, Any], dataset_path: str, dataset_pickle: str | None, workdir: Path,
                  cli: str, python: str, suffix: str) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    from .cli_runner import write_launcher_shim

    workdir.mkdir(parents=True, exist_ok=False)
    documents = {"request": request, "data_envelopes": envelopes, "relations": relations, "influence": influence, "graph": graph}
    for name, document in documents.items():
        (workdir / f"{name}.json").write_text(json.dumps(document), encoding="utf-8")
    capture = workdir / "results.jsonl"
    sidecars = workdir / "refit_artifacts"
    sidecars.mkdir()
    fitted_x = workdir / "fitted_x"
    fitted_x.mkdir()
    shim = write_launcher_shim(workdir / "n4a_adapter", python)
    env = {**os.environ, "N4A_DAGML_DATASET_PATH": dataset_path, "N4A_DAGML_GRAPH_PATH": str(workdir / "graph.json"),
           "N4A_DAGML_RESULT_CAPTURE": str(capture), "N4A_DAGML_REFIT_ARTIFACT_DIR": str(sidecars),
           "N4A_DAGML_FITTED_X_DIR": str(fitted_x), "N4A_RANDOM_STATE": str(request["options"]["seed"])}
    env.pop("N4A_DAGML_SAMPLE_META_PATH", None)
    env.pop("N4A_DAGML_DATASET_PICKLE", None)
    if dataset_pickle is not None:
        env["N4A_DAGML_DATASET_PICKLE"] = dataset_pickle
    output = workdir / "native_training.json"
    command = [cli, "execute-training", "--request", str(workdir / "request.json"), "--data-envelopes", str(workdir / "data_envelopes.json"),
               "--relations", str(workdir / "relations.json"), "--influence", str(workdir / "influence.json"), "--adapter", str(shim),
               "--output", str(output), "--outcome-id", f"outcome:{suffix}", "--run-id", "run:nirs4all-by-source-auto-models",
               "--bundle-id", f"bundle:{suffix}", "--package-id", f"predictor:{suffix}"]
    terminal = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    frames = [json.loads(line) for line in capture.read_text().splitlines() if line.strip()] if capture.exists() else []
    if terminal.returncode:
        raise RuntimeError(f"native by_source execute_training failed ({terminal.returncode}): {terminal.stderr}\n{terminal.stdout}")
    payload = json.loads(output.read_text())
    outcome, package = payload["outcome"], payload["portable_package"]
    dag_ml.TrainingOutcome(outcome)
    if package is not None:
        # The native package/declarations constrain the set before any sidecar is deserialized.
        declared = []
        for frame in frames:
            result = frame.get("result") if frame.get("type") == "result" else frame
            for descriptor in result.get("artifacts", []) if isinstance(result, dict) else []:
                declared.append({"artifact_id": descriptor["id"], "controller_id": descriptor["controller_id"],
                                 "kind": descriptor["kind"], "producer_node": result.get("node_id", frame.get("node_id"))})
        validate_source_package_bindings([package], declared)
        from .in_process_runner import _refit_artifact_path

        if set(sidecars.iterdir()) != {_refit_artifact_path(sidecars, item["artifact_id"]) for item in declared}:
            raise ValueError("subprocess sidecar files do not biject with original native PackageArtifactBinding")
        artifacts = _load_subprocess_refit_artifacts(frames, sidecars)
    else:
        artifacts = []
    return outcome, package, frames, artifacts


def execute_attested_by_source_cv(*, dsl: dict[str, Any], envelope: dict[str, Any], graph: dict[str, Any], spectro: Any,
                                  identity: IdentityMap, folds: list[tuple[list[int], list[int]]], source_names: list[str],
                                  selection_metric: str, random_state: int | None = None, refit_top_k: int = 1, refit: bool = True,
                                  group_by_sample: dict[int, str] | None = None, cli: str | None = None, python: str | None = None,
                                  dataset_path: str = "", dataset_pickle: str | None = None, workdir: Path | None = None) -> dict[str, Any]:
    """Execute one signed native training request per selected rank, retaining every original capture.

    Each extra rank repeats native candidate CV. Python never ranks scores or
    synthesizes a TrainingOutcome from another scheduler's execution bundle.
    """
    from nirs4all.data.multimodal import MultimodalSpectroDataset

    if not folds or len(source_names) < 2 or type(refit_top_k) is not int or refit_top_k < 1 or not refit and refit_top_k != 1:
        raise ValueError("attested source training requires CV, multiple sources and positive refit ranks")
    models = sorted([node for node in graph["nodes"] if node["kind"] == "model"], key=lambda node: node.get("metadata", {}).get("source_index", -1))
    indices = [node.get("metadata", {}).get("source_index") for node in models]
    if any(type(index) is not int for index in indices) or set(indices) != set(range(len(source_names))):
        raise ValueError("source model nodes must cover every indexed source")
    manifests = controller_manifests(dsl)
    artifact = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, manifests)
    if artifact.graph.to_dict() != graph:
        raise ValueError("attested source graph differs from the executable DSL")
    graph = copy.deepcopy(graph)
    if any(node["kind"] == "generator" for node in _source_dsl_nodes(dsl.get("steps", []))):
        graph.setdefault("metadata", {})["training_operator_source_dsl"] = copy.deepcopy(dsl)
    models = [next(node for node in models if node["metadata"]["source_index"] == index) for index in range(len(source_names))]
    campaign = artifact.campaign_template.to_dict()
    seed = random_state if random_state is not None else 12345
    campaign["root_seed"] = seed
    signed = copy.deepcopy(envelope)
    pool = spectro.index_column("sample", {"partition": "train"})
    signed["relation_fingerprint"] = _core_relation_fingerprint(signed["coordinator_relations"], dag_ml)
    signed["data_content_fingerprint"] = (spectro.content_hash(sample_rows=pool) if isinstance(spectro, MultimodalSpectroDataset)
                                          else _array_content_fingerprint("X", spectro.x({"partition": "train"}, layout="2d")))
    signed["target_content_fingerprint"] = _array_content_fingerprint("y", spectro.y({"partition": "train"}))
    from .envelope import build_envelope, target_names

    test = spectro.index_column("sample", {"partition": "test"})
    if test:
        test_envelope = build_envelope(spectro, identity, sample_ints=test)
        signed.update(dag_ml.attach_predict_cohort_to_envelope(signed, {
            "role": "external_test", "relations": test_envelope["coordinator_relations"], "target_names": target_names(spectro),
            "data_content_fingerprint": (spectro.content_hash(sample_rows=test) if isinstance(spectro, MultimodalSpectroDataset)
                                         else _array_content_fingerprint("X", spectro.x({"partition": "test"}, layout="2d"))),
            "target_content_fingerprint": _array_content_fingerprint("y", spectro.y({"partition": "test"})),
        }).to_dict())
    envelopes, identities = _data_contracts_from_campaign(campaign, signed)
    resolver = MaterializationResolver(spectro, identity)
    targets = resolver.resolve_targets([identity.to_wire(sample) for sample in pool])
    names = targets.get("target_names", ["y"] if len(target_names(spectro)) == 1 else target_names(spectro))
    classes = None
    if spectro.is_classification:
        matrix = np.asarray(targets["values"], dtype=float).reshape(len(pool), -1)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("by_source classification requires complete finite encoded training labels")
        classes = [[str(float(value)) for value in np.unique(matrix[:, column])] for column in range(matrix.shape[1])]
    outputs = []
    for index, node in enumerate(models):
        output = _output_request_for_node(graph, node["id"], target_names=names, class_labels=classes)
        output["output_id"] = f"output:source_{index}"
        outputs.append(output)
    nodes = {node["id"]: node for node in graph["nodes"]}
    y_node = next((node for node in graph["nodes"] if node["kind"] == "y_transform"), None)
    captures: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    scores = None
    first_training, first_package = None, None
    for rank in range(1, refit_top_k + 1):
        request = assemble_training_request(DagMLTrainingRequestSpec(
            request_id=f"training:{dsl['id']}:rank:{rank}", plan_id=f"plan:{dsl['id']}", graph=graph, campaign=campaign,
            controller_manifests=manifests, data_identities=identities, output_requests=outputs, selection_metric=selection_metric,
            selection_objective="maximize" if is_higher_better(selection_metric) else "minimize", selection_output_id=outputs[0]["output_id"],
            selection_required_metric_level=campaign.get("aggregation_policy", {}).get("selection_metric_level", "sample"), selection_evaluation_scope="oof",
            selection_requested_rank=rank if refit_top_k > 1 else None, seed=seed, refit=refit, cv_artifacts="discard", fitted_artifacts="allow_host_sidecar"))
        if len(dag_ml.TrainingRequest(request).project().to_dict()["plan"]["variants"]) < refit_top_k:
            raise ValueError("refit_top_k exceeds the concrete native variant inventory")
        influence = _training_influence_manifest(graph, campaign, folds, identity, group_by_sample=group_by_sample or {}, selection_metric=selection_metric, refit=refit)
        local_frames: list[dict[str, Any]] = []
        store: dict[int, Any] = {}

        def callback(task: dict[str, Any], store: dict[int, Any] = store,
                     local_frames: list[dict[str, Any]] = local_frames) -> dict[str, Any]:
            result = run_node(task, resolver, nodes.__getitem__, store, graph.get("edges", []), y_node, graph_metadata=graph.get("metadata"))
            local_frames.append(result)
            return result

        suffix = f"{dsl['id']}:rank:{rank}"
        if cli is None:
            training = dag_ml.execute_training(request, envelopes, signed["coordinator_relations"], influence, callback,
                                              outcome_id=f"outcome:{suffix}", run_id=f"run:{dsl['id']}", bundle_id=f"bundle:{suffix}")
            outcome = training.outcome.to_dict()
            package_object = training.export_portable_predictor_package(f"predictor:{suffix}") if refit else None
            package = package_object.to_dict() if package_object is not None else None
            if rank == 1:
                first_training, first_package = training, package_object
            local_artifacts = _capture_refit_artifacts(local_frames, store)
        else:
            if python is None or workdir is None:
                raise ValueError("native CLI training requires an explicit worker Python and isolated workspace")
            outcome, package, local_frames, local_artifacts = _cli_training(request=request, envelopes=envelopes, relations=signed["coordinator_relations"],
                influence=influence, graph=graph, dataset_path=dataset_path, dataset_pickle=dataset_pickle,
                workdir=workdir / f"native-training-rank-{rank}", cli=cli, python=python, suffix=suffix)
        if package is not None:
            validate_source_package_bindings([package], local_artifacts)
            if captures and _native_selection(outcome)["ranked_candidates"] != _native_selection(captures[0]["outcome"])["ranked_candidates"]:
                raise ValueError("native candidate ranking changed between selected refit ranks")
            captures.append({"outcome": outcome, "package": package})
        for average in [*outcome.get("oof_averages", []), *outcome.get("ensemble_averages", [])]:
            local_frames.append({"variant_id": outcome["selected_variant_id"], "aggregated_predictions": [average["predictions"]], "regression_targets": [average["y_true"]]})
        for variant in outcome.get("variant_oof_averages", []):
            for average in variant["oof_averages"]:
                local_frames.append({"variant_id": variant["variant_id"], "aggregated_predictions": [average["predictions"]], "regression_targets": [average["y_true"]]})
        frames.extend(local_frames)
        artifacts.extend(local_artifacts)
        current = outcome["score_set"]
        if scores is None:
            scores = copy.deepcopy(current)
        else:
            for report in current["reports"]:
                if report not in scores["reports"]:
                    scores["reports"].append(report)
    if captures:
        validate_source_package_bindings([item["package"] for item in captures], artifacts)
    return {"captures": captures, "training_result": first_training, "training_outcome": captures[0]["outcome"] if captures else outcome,
            "portable_package": first_package, "portable_package_dict": captures[0]["package"] if captures else None,
            "scores": scores, "results": frames, "refit_artifacts": artifacts,
            "selected_refit_variant_ids": [item["outcome"]["selected_variant_id"] for item in captures]}
