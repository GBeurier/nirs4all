"""Early/late public grammar, ordered lowering and native preflight boundaries."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from nirs4all.data.multimodal import MultimodalSpectroDataset
from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component
from nirs4all.pipeline.dagml.identity import mint_identity
from nirs4all.pipeline.dagml.methods_multimodal import source_schemas_from_cohort
from nirs4all.pipeline.dagml.raw_training_lowerer import _training_influence_manifest
from nirs4all.pipeline.dagml.structural_topology import INNER_SPLITS, declared_topologies, is_topology_choice, lower_topology_choices
from nirs4all.pipeline.dagml.structural_tuning import _prepare_structure, validate_structural_profile

_PATH = Path(__file__).resolve().parents[4] / "examples/user/04_models/U21_structural_hpo_early_late.py"
_SPEC = importlib.util.spec_from_file_location("early_late_unit_example", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


def test_training_influence_distinguishes_oof_meta_models_from_raw_and_ordinary_models() -> None:
    cohort = example.make_dataset()
    identity = mint_identity(MultimodalSpectroDataset(cohort))
    graph = {"nodes": [
        {"id": "raw", "kind": "model", "metadata": {"controller_id": "controller:methods.python.multimodal"}},
        {"id": "meta", "kind": "model", "metadata": {"controller_id": "controller:methods.python.regression"}},
        {"id": "ordinary", "kind": "model", "metadata": {"controller_id": "controller:methods.python.regression"}},
        {"id": "legacy", "kind": "model", "metadata": {"controller_id": "controller:methods.ridge"}},
    ], "edges": [
        {"source": {"node_id": "raw"}, "target": {"node_id": "meta"}, "contract": {"kind": "prediction", "requires_oof": True}},
        {"source": {"node_id": "raw"}, "target": {"node_id": "ordinary"}, "contract": {"kind": "prediction", "requires_oof": False}},
    ]}
    campaign = {"data_bindings": {"raw": [{"relation_fingerprint": "a" * 64}]}}
    manifest = _training_influence_manifest(graph, campaign, [([0, 1], [2])], identity,
                                            group_by_sample={0: "g0", 1: "g1", 2: "g2"}, selection_metric="rmse")
    for node, kind in {"raw": "model_fit", "ordinary": "model_fit", "meta": "trained_meta_aggregation", "legacy": "trained_meta_aggregation"}.items():
        entries = [entry for entry in manifest["entries"] if entry["node_id"] == node]
        assert len(entries) == 2 and {entry["kind"] for entry in entries} == {kind}
        scoped = {entry["scope_id"]: entry for entry in entries}
        assert scoped["fit_cv:fold0"]["physical_sample_ids"] == sorted(cohort.sample_ids[:2])
        assert scoped["fit_cv:fold0"]["group_ids"] == ["g0", "g1"]
        assert scoped["refit:full"]["physical_sample_ids"] == sorted(cohort.sample_ids[:3])
        assert scoped["refit:full"]["group_ids"] == ["g0", "g1", "g2"]


def test_public_sequence_roundtrip_preserves_native_order_and_distinct_logical_ids() -> None:
    pipeline = deserialize_component(json.loads(json.dumps(serialize_component(example.make_pipeline()))))
    steps, splitter = validate_structural_profile(pipeline)
    assert is_topology_choice(steps) and type(splitter) is GroupKFold
    schemas = source_schemas_from_cohort(example.make_dataset())
    dsl, bindings, sinks = lower_topology_choices(steps, splitter, schemas)
    generator = dsl["pipeline"][0]
    assert generator["mode"] == "cartesian" and len(generator["stages"]) == 1
    branches = generator["stages"][0]["branches"]
    assert len(branches) == 5 and len(sinks) == 5
    ids = []
    for index, branch in enumerate(branches):
        if index == 0:
            raw_nodes = branch["steps"]
            assert raw_nodes[0]["operator"]["recipe"]["source_order"] == ["nir", "image"]
            assert sinks[index] == raw_nodes[0]["id"]
        else:
            duplication, meta = branch["steps"]
            order = list(pipeline[1]["_or_"][index][0]["branch"])
            assert [child["id"] for child in duplication["branches"]] == order
            raw_nodes = [child["steps"][0] for child in duplication["branches"]]
            assert [node["operator"]["recipe"]["source_order"] for node in raw_nodes] == [[name] for name in order]
            assert meta["operator"]["source_order"] == order
            assert meta["sources"] == [node["id"] for node in raw_nodes]
            assert meta["inner_cv"] == {"kind": "group_kfold", "n_splits": INNER_SPLITS}
            assert meta["metadata"]["stacking_oof_execution"] == "nested_oof_v1"
            assert meta["metadata"]["stacking_refit_oof"] == "partitioned_inner_v1"
            assert meta["operator"]["steps"][0]["methodId"] == "models.regularized.ridge"
            assert sinks[index] == meta["id"]
            ids.append(meta["id"])
        for node in raw_nodes:
            ids.append(node["id"])
            assert node["operator"]["source_schemas"] == schemas == node["params"]["source_schemas"]
            assert set(node["operator"]["source_schemas"]) == {"nir", "image", "series", "metadata"}
    assert len(ids) == len(set(ids)) == 16
    assert set(bindings) == {"early.alpha", "late.nir.alpha", "late.image.alpha", "late.series.alpha", "late.metadata.alpha", "late.meta.alpha"}
    assert {binding["node_id"] for options in bindings.values() for binding in options} == set(ids)
    # Sorting JSON object keys cannot reorder the lowered branch/source lists.
    assert json.loads(json.dumps(dsl, sort_keys=True)) == dsl


@pytest.mark.parametrize("mutation", ["not_sequence", "wrong_merge", "one_branch", "wrong_source", "nested_branch", "sklearn_backend", "extra_step", "bad_meta", "shuffle"])
def test_unsupported_topology_refused_without_native_callbacks(mutation: str) -> None:
    pipeline = example.make_pipeline()
    sequence = pipeline[1]["_or_"][1]
    if mutation == "not_sequence":
        pipeline[1]["_or_"][0] = pipeline[1]["_or_"][0][0]
    elif mutation == "wrong_merge":
        sequence[1] = {"merge": "features"}
    elif mutation == "one_branch":
        sequence[0]["branch"].pop("image")
    elif mutation == "wrong_source":
        sequence[0]["branch"]["nir"][0]["model"] = example.make_model(("image",))
    elif mutation == "nested_branch":
        sequence[0]["branch"]["nir"].append({"model": example.make_model(("nir",))})
    elif mutation == "sklearn_backend":
        sequence[0]["branch"]["nir"][0]["model"].backend = "sklearn"
    elif mutation == "extra_step":
        sequence.append({"model": Ridge()})
    elif mutation == "bad_meta":
        sequence[2]["model"] = Ridge(positive=True)
    else:
        pipeline[0] = GroupKFold(3, shuffle=True, random_state=17)
    with pytest.raises((ValueError, TypeError)):
        declared_topologies(pipeline[1:], pipeline[0])


@pytest.mark.parametrize("control", ["parallel", "pruning", "refit", "axis", "alpha"])
def test_invalid_campaign_controls_fail_before_native_catalogue(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, control: str) -> None:
    import dag_ml

    tuning = example.make_tuning(tmp_path / "study")
    options: dict[str, Any] = {"refit": True}
    if control == "parallel":
        tuning["n_jobs"] = 2
    elif control == "pruning":
        tuning["pruner"] = "median"
    elif control == "refit":
        options["refit"] = False
    elif control == "axis":
        tuning["space"]["late.weight"] = [0.5, 1.0]
    else:
        tuning["space"]["late.meta.alpha"] = [-1.0, 1.0]
    monkeypatch.setattr(dag_ml, "prepare_host_hpo_topology_catalogue", lambda *a, **k: pytest.fail("invalid declaration reached native catalogue"))
    with pytest.raises((ValueError, TypeError)):
        _prepare_structure(example.make_pipeline(), example.make_dataset(), tuning, options)


def test_dotted_and_sklearn_alpha_axes_bind_the_same_native_catalogue(tmp_path: Path) -> None:
    tuning = example.make_tuning(tmp_path / "study")
    aliased = copy.deepcopy(tuning)
    aliased["space"] = {path.replace(".", "__"): declaration for path, declaration in tuning["space"].items()}
    prepared = _prepare_structure(example.make_pipeline(), example.make_dataset(), tuning, {"random_state": 17})
    alternate = _prepare_structure(example.make_pipeline(), example.make_dataset(), aliased, {"random_state": 17})
    expected_groups = {prepared["identity"].to_wire(sample): str(prepared["dataset"].cohort.groups[sample]) for sample in prepared["pool"]}
    fold_set = prepared["dsl"]["split_invocation"]["fold_set"]
    assert fold_set["sample_groups"] == expected_groups
    assert set(fold_set["sample_ids"]) == set(expected_groups)
    records = prepared["envelope"]["coordinator_relations"]["records"]
    assert {record["sample_id"]: record["group_id"] for record in records} == expected_groups
    for fold in fold_set["folds"]:
        assert not {expected_groups[sample] for sample in fold["train_sample_ids"]}.intersection(expected_groups[sample] for sample in fold["validation_sample_ids"])
    assert prepared["catalogue"] == alternate["catalogue"]
    assert prepared["catalogue"]["schema_version"] == 2
    assert len({entry["variant_label"] for entry in prepared["catalogue"]["entries"]}) == 5
    for entry in prepared["catalogue"]["entries"]:
        assert entry["target_node"] in {node["id"] for node in entry["graph"]["nodes"]}
        active = set(entry["parameter_bindings"])
        if "early.alpha" in active:
            assert active == {"early.alpha"}
        else:
            assert "late.meta.alpha" in active and "early.alpha" not in active
