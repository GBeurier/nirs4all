"""Attested CV capture for independent source-local model outputs."""

from __future__ import annotations

import copy
from typing import Any

import dag_ml

from nirs4all.core.metrics import is_higher_better
from nirs4all.pipeline.dagml_bridge import controller_manifests

from .identity import IdentityMap
from .in_process_runner import _capture_refit_artifacts
from .node_runner import run_node
from .raw_training_lowerer import (
    _array_content_fingerprint,
    _core_relation_fingerprint,
    _data_contracts_from_campaign,
    _output_request_for_node,
    _training_influence_manifest,
)
from .resolver import MaterializationResolver
from .training_contracts import DagMLTrainingRequestSpec, assemble_training_request


def execute_attested_by_source_cv(
    *,
    dsl: dict[str, Any],
    envelope: dict[str, Any],
    graph: dict[str, Any],
    spectro: Any,
    identity: IdentityMap,
    folds: list[tuple[list[int], list[int]]],
    source_names: list[str],
    selection_metric: str,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Train all source outputs once via native ``execute_training``.

    This path keeps its signed request and outcome from the actual training
    execution. It supports one selected refit variant and no fold augmentation;
    callers must route other campaign shapes through their existing runner.
    """
    if not folds or len(source_names) < 2:
        raise ValueError("attested source training requires CV folds and multiple sources")
    models = [node for node in graph["nodes"] if node["kind"] == "model"]
    source_indexes = [node.get("metadata", {}).get("source_index") for node in models]
    if source_indexes != list(range(len(source_names))):
        raise ValueError("source model nodes must be ordered and indexed by source")
    manifests = controller_manifests()
    artifact = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, manifests)
    if artifact.graph.to_dict() != graph:
        raise ValueError("attested source graph differs from the executable DSL")
    campaign = artifact.campaign_template.to_dict()
    seed = random_state if random_state is not None else 12345
    campaign["root_seed"] = seed
    signed_envelope = copy.deepcopy(envelope)
    signed_envelope["relation_fingerprint"] = _core_relation_fingerprint(
        signed_envelope["coordinator_relations"], dag_ml,
    )
    signed_envelope["data_content_fingerprint"] = _array_content_fingerprint(
        "X", spectro.x({"partition": "train"}, layout="2d"),
    )
    signed_envelope["target_content_fingerprint"] = _array_content_fingerprint(
        "y", spectro.y({"partition": "train"}),
    )
    data_envelopes, data_identities = _data_contracts_from_campaign(campaign, signed_envelope)
    from .envelope import target_names

    names = target_names(spectro)
    if len(names) == 1:
        # The host model callback emits the established single-target name.
        names = ["y"]
    output_requests = []
    for index, node in enumerate(models):
        output = _output_request_for_node(graph, node["id"], target_names=names)
        output["output_id"] = f"output:source_{index}"
        output_requests.append(output)
    request = assemble_training_request(DagMLTrainingRequestSpec(
        request_id=f"training:{dsl['id']}", plan_id=f"plan:{dsl['id']}",
        graph=graph, campaign=campaign, controller_manifests=manifests,
        data_identities=data_identities, output_requests=output_requests,
        selection_metric=selection_metric,
        selection_objective="maximize" if is_higher_better(selection_metric) else "minimize",
        selection_output_id=output_requests[0]["output_id"],
        selection_required_metric_level="sample", selection_evaluation_scope="oof",
        seed=seed, cv_artifacts="discard", fitted_artifacts="allow_host_sidecar",
    ))
    influence = _training_influence_manifest(
        graph, campaign, folds, identity, group_by_sample={}, selection_metric=selection_metric,
    )
    resolver = MaterializationResolver(spectro, identity)
    nodes = {node["id"]: node for node in graph["nodes"]}
    y_transform_node = next((node for node in graph["nodes"] if node["kind"] == "y_transform"), None)
    store: dict[int, Any] = {}
    frames: list[dict[str, Any]] = []

    def callback(task: dict[str, Any]) -> dict[str, Any]:
        result = run_node(task, resolver, nodes.__getitem__, store, graph.get("edges", []), y_transform_node)
        frames.append(result)
        return result

    training = dag_ml.execute_training(
        request, data_envelopes, signed_envelope["coordinator_relations"], influence, callback,
        outcome_id=f"outcome:{dsl['id']}", run_id=f"run:{dsl['id']}", bundle_id=f"bundle:{dsl['id']}",
    )
    package = training.export_portable_predictor_package(f"predictor:{dsl['id']}")
    if {item["binding"]["binding_id"] for item in training.outputs} != {
        output["output_id"] for output in output_requests
    }:
        raise ValueError("attested source training omitted a named output")
    return {
        "training_result": training,
        "portable_package": package,
        "scores": training.score_set,
        "results": frames,
        "refit_artifacts": _capture_refit_artifacts(frames, store),
    }
