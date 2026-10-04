"""Signed vocabulary, strict classification declarations and preflight transport."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component
from nirs4all.pipeline.dagml.methods_classification import classification_vocabulary, classifier_head, decode_labels, encode_labels
from nirs4all.pipeline.dagml.methods_multimodal import source_schemas_from_cohort
from nirs4all.pipeline.dagml.structural_classification import classification_output, lower_classifier_choices, validate_count_space
from nirs4all.pipeline.dagml.structural_tuning import _prepare_structure, validate_structural_profile

_PATH = Path(__file__).resolve().parents[4] / "examples/user/04_models/U23_structural_hpo_classification.py"
_SPEC = importlib.util.spec_from_file_location("classification_unit_example", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


@pytest.mark.parametrize("labels", [["blue", "amber", "violet"], [103, -9, 17], [(1 << 62) + 3, -9, 17]])
def test_typed_train_vocabulary_roundtrip_never_floatcasts_original_labels(labels: list) -> None:
    values = [labels[2], labels[0], labels[1], labels[2]]
    vocabulary = classification_vocabulary(values, [0, 1, 2])
    assert vocabulary["label_names"] == sorted(labels)
    assert vocabulary["class_labels"] == [0, 1, 2]
    actual = decode_labels(encode_labels(values, vocabulary).astype(float), vocabulary)
    assert actual.tolist() == values
    assert all(type(value) is (str if isinstance(labels[0], str) else int) for value in actual.tolist())


def test_vocabulary_never_learns_a_heldout_only_class() -> None:
    vocabulary = classification_vocabulary(["b", "a", "only-test"], [0, 1])
    assert vocabulary["label_names"] == ["a", "b"]
    with pytest.raises(ValueError, match="training-only"):
        encode_labels(["only-test"], vocabulary)


@pytest.mark.parametrize("labels", [[1, "1"], [True, False], [0.0, 1.0], [None, "a"]])
def test_native_label_types_refuse_ambiguous_or_missing_values(labels: list) -> None:
    with pytest.raises(ValueError, match="homogeneous"):
        classification_vocabulary(labels, [0, 1])


@pytest.mark.parametrize("values", [[-1], [3], [0.5], [float("nan")], [True]])
def test_decoder_refuses_invalid_native_class_ids(values: list) -> None:
    with pytest.raises(ValueError, match="invalid"):
        decode_labels(values, {"schema_version": 1, "class_labels": [0, 1, 2], "label_names": [-9, 17, 103]})


@pytest.mark.parametrize("head", [Ridge(), LogisticRegression()])
def test_native_head_is_not_a_relabelled_regressor_or_generic_logistic(head: object) -> None:
    with pytest.raises(ValueError, match=r"n4m\.roles\.PLSLogistic"):
        classifier_head(head)


def test_public_roundtrip_preserves_probability_ports_and_declared_order() -> None:
    pipeline = deserialize_component(json.loads(json.dumps(serialize_component(example.make_pipeline()))))
    steps, splitter = validate_structural_profile(pipeline)
    schemas = source_schemas_from_cohort(example.make_dataset())
    vocabulary = {"schema_version": 1, "class_labels": [0, 1, 2], "label_names": ["amber", "blue", "violet"]}
    dsl, bindings, sinks = lower_classifier_choices(steps, splitter, schemas, vocabulary)
    ids = []
    for branch in dsl["pipeline"][0]["stages"][0]["branches"]:
        if branch["steps"][0]["kind"] == "branch":
            duplication, meta = branch["steps"]
            raw = [child["steps"][0] for child in duplication["branches"]]
            assert meta["operator"]["source_order"] == [child["id"] for child in duplication["branches"]]
            assert meta["source_ports"] == {node["id"]: "probabilities" for node in raw}
            assert meta["prediction_output_ports"] == ["y_hat", "probabilities"]
            assert meta["operator"]["steps"][0]["methodId"] == "models.classification.pls_logistic"
            assert meta["metadata"]["stacking_refit_oof"] == "partitioned_inner_v1"
            ids.append(meta["id"])
        else:
            raw = branch["steps"]
        for node in raw:
            assert node["prediction_output_ports"] == ["y_hat", "probabilities"]
            assert node["operator"]["source_schemas"] == schemas and set(schemas) == {"nir", "image", "series", "metadata"}
            assert node["operator"]["classification"] == vocabulary == node["params"]["classification"]
            ids.append(node["id"])
    assert len(ids) == len(set(ids)) == 16 and len(sinks) == 5
    assert set(bindings) == {"early.n_components", "late.nir.n_components", "late.image.n_components", "late.series.n_components", "late.metadata.n_components", "late.meta.n_components"}
    graph = {"nodes": [{"id": "meta", "kind": "model"}]}
    output = classification_output(graph, "meta", vocabulary)
    assert output["port_name"] == "y_hat" and output["prediction_kind"] == "class_label"
    assert output["class_labels"] == [["0", "1", "2"]]


@pytest.mark.parametrize("declaration", [[1.0, 2.0], ("float", 1, 2), {"type": "float", "low": 1, "high": 2}, [0, 1], [True, 2]])
def test_components_require_genuine_integer_domains(declaration: object) -> None:
    with pytest.raises(ValueError):
        validate_count_space(declaration)


@pytest.mark.parametrize("control", ["parallel", "pruner", "direction", "refit", "axis", "components"])
def test_unsupported_classification_refuses_before_native_catalogue(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, control: str) -> None:
    import dag_ml

    tuning = example.make_tuning(tmp_path / "study")
    options = {}
    if control == "parallel":
        tuning["n_jobs"] = 2
    elif control == "pruner":
        tuning["pruner"] = "median"
    elif control == "direction":
        tuning["direction"] = "minimize"
    elif control == "refit":
        options["refit"] = False
    elif control == "axis":
        tuning["space"]["late.weight"] = [1]
    else:
        tuning["space"]["late.meta.n_components"] = [1.0]
    monkeypatch.setattr(dag_ml, "prepare_host_hpo_topology_catalogue", lambda *a, **k: pytest.fail("invalid profile reached native catalogue"))
    with pytest.raises((ValueError, TypeError)):
        _prepare_structure(example.make_pipeline(), example.make_dataset(), tuning, options)


def test_envelope_signs_exact_train_labels_groups_and_alias_bindings(tmp_path: Path) -> None:
    cohort = example.make_dataset(numeric=True)
    tuning = example.make_tuning(tmp_path / "study")
    prepared = _prepare_structure(example.make_pipeline(), cohort, tuning, {})
    targets = prepared["dsl"]["metadata"]["classification_targets"]
    expected = {prepared["identity"].to_wire(sample): int(prepared["encoded_targets"][sample]) for sample in prepared["pool"]}
    assert targets == {**prepared["classification"], "sample_labels": expected}
    folds = prepared["dsl"]["split_invocation"]["fold_set"]
    assert set(folds["sample_groups"]) == set(expected)
    assert all(sample not in expected for sample, partition in zip(cohort.sample_ids, cohort.partitions, strict=True) if partition == "test")
    aliased = copy.deepcopy(tuning)
    aliased["space"] = {name.replace(".", "__"): declaration for name, declaration in tuning["space"].items()}
    alternate = _prepare_structure(example.make_pipeline(), cohort, aliased, {})
    assert alternate["catalogue"] == prepared["catalogue"]


def test_empty_string_labels_are_valid_and_wire_resource_budgets_are_explicit() -> None:
    vocabulary = classification_vocabulary(["", "named"], [0, 1])
    assert decode_labels([0, 1], vocabulary).tolist() == ["", "named"]
    with pytest.raises(ValueError, match="byte budget"):
        classification_vocabulary(["x" * ((1 << 20) + 1), "a"], [0, 1])
    with pytest.raises(ValueError, match="65536"):
        classification_vocabulary(list(range(65537)), list(range(65537)))
