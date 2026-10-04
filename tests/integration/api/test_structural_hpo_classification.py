"""Actual serial native classification, independent OOF oracle and portable replay."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nirs4all_io import MultimodalDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import nirs4all
from nirs4all.api.portable_archive import read_portable_predictor_archive_v2
from nirs4all.operators.models.multimodal import MultimodalClassifier, TensorPCA
from nirs4all.pipeline.dagml.methods_classification import classifier_recipe
from nirs4all.pipeline.dagml.methods_multimodal import source_schemas_from_cohort
from nirs4all.pipeline.dagml.structural_tuning import _prepare_structure
from tests.integration.api.classification_oracle import topology_probabilities
from tests.integration.api.test_structural_hpo_preprocessing_chains import _enqueue_recipes

_PATH = Path(__file__).resolve().parents[3] / "examples/user/04_models/U23_structural_hpo_classification.py"
_SPEC = importlib.util.spec_from_file_location("classification_integration_example", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


@pytest.fixture(autouse=True)
def native_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("legacy execution"))
    monkeypatch.setattr("nirs4all.pipeline.dagml.run_paths._run_model_on_precomputed_matrix", lambda *a, **k: pytest.fail("host CV execution"))


def _run(pipeline: Any, cohort: Any, root: Path, tuning: dict[str, Any] | None = None) -> Any:
    return nirs4all.run(pipeline, cohort, tuning=tuning, engine="dag-ml", workspace_path=root,
                        random_state=17, refit=True, verbose=0, save_charts=False)


def _without_host_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    for cls in (MultimodalClassifier, StandardScaler, OneHotEncoder, TensorPCA):
        monkeypatch.setattr(cls, "fit", lambda *a, **k: pytest.fail("production Python FIT"))


def _observe(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    from dag_ml.multimodal_classification import ClassificationTopologyController

    controllers = []
    original, operation = ClassificationTopologyController.__init__, ClassificationTopologyController.operator
    def create(controller: Any, *args: Any, **kwargs: Any) -> None:
        original(controller, *args, **kwargs)
        controller._test_packets = []
        controllers.append(controller)
    def execute(controller: Any, task: Any) -> Any:
        result = operation(controller, task)
        controller._test_packets.append({"task": copy.deepcopy(task), "result": copy.deepcopy(result)})
        return result
    monkeypatch.setattr(ClassificationTopologyController, "__init__", create)
    monkeypatch.setattr(ClassificationTopologyController, "operator", execute)
    return controllers


def _sequence(entry: dict[str, Any], pipeline: list[Any]) -> list[Any]:
    nodes = {node["id"]: node for node in entry["graph"]["nodes"] if node["kind"] == "model"}
    bindings = entry["parameter_bindings"]
    matched = []
    for sequence in pipeline[1]["_or_"]:
        if len(sequence) == 1:
            if set(bindings) != {"early.n_components"} or set(nodes) != {entry["target_node"]}:
                continue
            node = nodes[entry["target_node"]]
            if (bindings["early.n_components"] == {"node_id": node["id"], "param_path": "model__n_components"}
                    and node["operator"]["recipe"] == classifier_recipe(sequence[0]["model"], allow_source_selection=True)):
                matched.append(sequence)
        else:
            branches = sequence[0]["branch"]
            if set(bindings) != {"late.meta.n_components", *(f"late.{name}.n_components" for name in branches)} or len(nodes) != len(branches) + 1:
                continue
            meta = nodes[entry["target_node"]]
            if meta["operator"]["source_order"] != list(branches) or bindings["late.meta.n_components"] != {"node_id": meta["id"], "param_path": "n_components"}:
                continue
            for name, branch in branches.items():
                binding = bindings[f"late.{name}.n_components"]
                if (binding["param_path"] != "model__n_components"
                        or nodes[binding["node_id"]]["operator"]["recipe"] != classifier_recipe(branch[0]["model"], allow_source_selection=True)):
                    break
            else:
                matched.append(sequence)
    assert len(matched) == 1, (entry["recipe_id"], len(matched))
    return matched[0]


@pytest.mark.parametrize("n_classes,numeric", [(2, False), (3, False), (2, True), (3, True)])
def test_direct_native_classifier_and_fixed_run_preserve_typed_labels_and_probabilities(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, n_classes: int, numeric: bool) -> None:
    cohort = example.make_dataset(n_classes=n_classes, numeric=numeric)
    train = np.flatnonzero(np.asarray(cohort.partitions) == "train")
    heldout = np.flatnonzero(np.asarray(cohort.partitions) == "test")
    model = example.make_model(("nir", "image", "series", "metadata"))
    with monkeypatch.context() as strict:
        for cls in (StandardScaler, OneHotEncoder, TensorPCA):
            strict.setattr(cls, "fit", lambda *a, **k: pytest.fail("direct native used Python encoder FIT"))
        model.fit([source.values[train] for source in cohort.sources.values()], cohort.y[train], source_schemas=source_schemas_from_cohort(cohort))
        blocks = [source.values[heldout] for source in cohort.sources.values()]
        actual, probabilities = model.predict(blocks), model.predict_proba(blocks)
    names = sorted(set(cohort.y[train].tolist()))
    assert model.classes_.tolist() == names
    np.testing.assert_array_equal(actual, np.asarray(names)[probabilities.argmax(axis=1)])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
    vocabulary = {"schema_version": 1, "class_labels": list(range(n_classes)), "label_names": names}
    expected = topology_probabilities([{"model": model}], cohort, train, cohort, heldout, {"early.n_components": 1}, vocabulary)
    np.testing.assert_allclose(probabilities, expected, rtol=2e-7, atol=2e-7)
    model.close()
    with monkeypatch.context() as strict:
        _without_host_fits(strict)
        result = _run([GroupKFold(3), {"model": example.make_model(("nir", "image", "series", "metadata"))}], cohort, tmp_path / "fixed")
    with result:
        assert result.classes_.tolist() == names and result.native_profile.endswith("classification.v1")
        assert result.methods_multimodal_training_request["options"]["outputs"][0]["port_name"] == "y_hat"
        rows = result.predictions.filter_predictions(partition="test", fold_id="final")
        assert len(rows) == 1 and set(rows[0]["sample_indices"]) == set(heldout)
        np.testing.assert_array_equal(rows[0]["y_pred"], np.asarray(names)[expected.argmax(axis=1)])


@pytest.mark.parametrize("metric", ["accuracy", "balanced_accuracy", "f1"])
def test_every_topology_matches_independent_train_only_grouped_probability_oracle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, metric: str) -> None:
    cohort, pipeline = example.make_dataset(), example.make_pipeline()
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 5, "metric": metric}
    prepared = _prepare_structure(pipeline, cohort, tuning, {"random_state": 17})
    catalogue, vocabulary = prepared["catalogue"], prepared["classification"]
    _enqueue_recipes(monkeypatch, catalogue, catalogue["entries"])
    observed = _observe(monkeypatch)
    train = np.flatnonzero(np.asarray(cohort.partitions) == "train")
    folds = list(GroupKFold(3).split(np.zeros((len(train), 1)), groups=np.asarray(cohort.groups)[train]))
    recipes = {entry["recipe_id"]: entry for entry in catalogue["entries"]}
    names = np.asarray(vocabulary["label_names"])
    scorer = accuracy_score if metric == "accuracy" else balanced_accuracy_score if metric == "balanced_accuracy" else lambda y, p: f1_score(y, p, average="weighted")
    with monkeypatch.context() as strict:
        _without_host_fits(strict)
        result = _run(pipeline, cohort, tmp_path / "workspace", tuning)
    with result:
        evidence = result.structural_tuning_evidence
        assert {trial["params"]["__recipe__"] for trial in evidence["trials"]} == set(recipes)
        for trial in evidence["trials"]:
            entry = recipes[trial["params"]["__recipe__"]]
            assert set(trial["params"]) == {"__recipe__", *entry["parameter_bindings"]}
            sequence = _sequence(entry, pipeline)
            scores = {f"fold{index}": scorer(cohort.y[train[heldout]], names[topology_probabilities(sequence, cohort,
                train[fit], cohort, train[heldout], trial["params"], vocabulary).argmax(axis=1)]) for index, (fit, heldout) in enumerate(folds)}
            assert trial["objective_fold_scores"] == pytest.approx(scores, abs=1e-12)
        assert result.classification_probability_blocks
        for block in result.classification_probability_blocks:
            assert block["target_names"] == ["class:0", "class:1", "class:2"]
            np.testing.assert_allclose(np.asarray(block["values"]).sum(axis=1), 1, atol=1e-12)
        winner = recipes[evidence["selected_params"]["__recipe__"]]
        archive = result.export(tmp_path / "winner.n4a")
        package = read_portable_predictor_archive_v2(archive).to_dict()
        assert {record["node_id"] for record in package["execution_bundle"]["refit_artifacts"]} == {node["id"] for node in winner["graph"]["nodes"] if node["kind"] == "model"}
        assert {record["artifact"]["kind"] for record in package["execution_bundle"]["refit_artifacts"]} <= {"methods_multimodal_classifier_pipeline", "methods_role_classifier_pipeline"}
    events = [event for controller in observed for event in controller.audit if event["operation"] == "fit"]
    test_ids = {sample for sample, partition in zip(cohort.sample_ids, cohort.partitions, strict=True) if partition == "test"}
    assert events and all(not set(event["sample_ids"]).intersection(test_ids) for event in events)
    for event in events:
        if "source_order" in event:
            assert set(event["raw_shapes"]) == set(event["source_order"])


@pytest.mark.parametrize("fixed", [False, True])
def test_missing_class_scope_is_refused_before_fit_and_optimizer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fixed: bool) -> None:
    from dag_ml import DagMlRuntimeError
    from dag_ml.multimodal_classification import ClassificationTopologyController
    from n4m.model_selection.optimizer import Optimizer

    source = example.make_dataset()
    labels = np.asarray(["rare" if group == source.groups[0] else "common" for group in source.groups])
    cohort = MultimodalDataset(source.sources, sample_ids=source.sample_ids, y=labels, groups=source.groups,
        partitions=source.partitions, task_type="classification", name=source.name)
    monkeypatch.setattr(ClassificationTopologyController, "operator", lambda *a, **k: pytest.fail("missing class reached FIT"))
    monkeypatch.setattr(Optimizer, "__init__", lambda *a, **k: pytest.fail("missing class reached optimizer"))
    pipeline = [GroupKFold(3), {"model": example.make_model(("nir", "image", "series", "metadata"))}] if fixed else example.make_pipeline()
    with pytest.raises(DagMlRuntimeError, match="class|vocabulary"):
        _run(pipeline, cohort, tmp_path / "invalid", None if fixed else example.make_tuning(tmp_path / "study"))


@pytest.mark.parametrize("train_only", [False, True])
def test_forced_late_archive_preserves_test_scope_or_artifact_only_refit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, train_only: bool) -> None:
    cohort, new = example.make_dataset(numeric=True, train_only=train_only), example.make_dataset(29, prediction=True)
    pipeline = example.make_pipeline()
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 1}
    prepared = _prepare_structure(pipeline, cohort, tuning, {})
    entry = next(entry for entry in prepared["catalogue"]["entries"] if _sequence(entry, pipeline) is pipeline[1]["_or_"][2])
    _enqueue_recipes(monkeypatch, prepared["catalogue"], [entry])
    observed = _observe(monkeypatch)
    with _run(pipeline, cohort, tmp_path / "workspace", tuning) as result:
        outcome = result._methods_multimodal_outcome.to_dict()
        producer = entry["target_node"]
        bound = outcome["outputs"][0]
        assert bound["binding"]["node_id"] == producer
        reports = [report for report in outcome["score_set"]["reports"] if report["producer_node"] == producer and report.get("fold_id") is None]
        if train_only:
            assert bound["artifact_only"] is True and not reports
            assert all(bound[surface] == [] for surface in ("predictions", "observation_predictions", "aggregated_predictions"))
        else:
            ids = {sample for sample, partition in zip(cohort.sample_ids, cohort.partitions, strict=True) if partition == "test"}
            blocks = bound["predictions"] + bound["aggregated_predictions"]
            assert blocks and all(block["partition"] == "test" and block.get("fold_id") is None for block in blocks)
            assert bound["refit_test_cohort"]["role"] == "external_test" and set(bound["refit_test_cohort"]["physical_sample_ids"]) == ids
            assert reports and all(report["partition"] == "test" for report in reports)
            callbacks = [block for controller in observed for packet in controller._test_packets if packet["task"]["phase"] == "REFIT"
                         for block in packet["result"].get("predictions", []) if block["producer_node"] == producer]
            assert callbacks and all(block["partition"] == "test" and set(block["sample_ids"]) == ids for block in callbacks)
        train = np.flatnonzero(np.asarray(cohort.partitions) == "train")
        expected = topology_probabilities(pipeline[1]["_or_"][2], cohort, train, new, np.arange(len(new.sample_ids)), result.tuning_best_params, prepared["classification"])
        archive = result.export(tmp_path / "late.n4a")
        actual = nirs4all.predict(archive, new, engine="dag-ml")
        np.testing.assert_array_equal(actual.y_pred.ravel(), np.asarray(prepared["classification"]["label_names"])[expected.argmax(axis=1)])
        assert actual.metadata["training_performed"] is False


@pytest.mark.parametrize("mutation", ["labels", "order", "excluded"])
def test_resume_drift_refuses_before_callbacks_and_preserves_checkpoint_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str) -> None:
    from dag_ml import DagMlRuntimeError
    from dag_ml.multimodal_classification import ClassificationTopologyController

    from tests.integration.api.test_methods_multimodal_u07 import _replace

    cohort, pipeline = example.make_dataset(), example.make_pipeline()
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 1}
    with _run(pipeline, cohort, tmp_path / "first", tuning):
        pass
    before = {path: path.read_bytes() for path in (tmp_path / "study").rglob("*") if path.is_file()}
    if mutation == "labels":
        labels = np.asarray(["renamed" if label == "amber" else label for label in cohort.y])
        cohort = MultimodalDataset(cohort.sources, sample_ids=cohort.sample_ids, y=labels, groups=cohort.groups,
            partitions=cohort.partitions, task_type="classification", name=cohort.name)
    elif mutation == "order":
        pipeline[1]["_or_"][1] = example.make_late(("image", "nir"))
    else:
        values = np.array(cohort.sources["series"].values, copy=True)
        values[0, 0, 0] += 0.1
        cohort = _replace(cohort, "series", values)
    monkeypatch.setattr(ClassificationTopologyController, "operator", lambda *a, **k: pytest.fail("resume drift reached callback"))
    with pytest.raises((ValueError, RuntimeError, DagMlRuntimeError)):
        _run(pipeline, cohort, tmp_path / "resume", {**tuning, "resume": True})
    assert before and {path: path.read_bytes() for path in before} == before


def test_fixed_learned_late_uses_native_preflight_and_no_optimizer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from n4m.model_selection.optimizer import Optimizer

    cohort, new = example.make_dataset(), example.make_dataset(29, prediction=True)
    sequence = example.make_late(("image", "nir"))
    params = {"late.image.n_components": 1, "late.nir.n_components": 1, "late.meta.n_components": 1}
    names = sorted(set(cohort.y.tolist()))
    vocabulary = {"schema_version": 1, "class_labels": [0, 1, 2], "label_names": names}
    train = np.flatnonzero(np.asarray(cohort.partitions) == "train")
    expected = topology_probabilities(sequence, cohort, train, new, np.arange(len(new.sample_ids)), params, vocabulary)
    monkeypatch.setattr(Optimizer, "__init__", lambda *a, **k: pytest.fail("fixed native run created an optimizer"))
    with monkeypatch.context() as strict:
        _without_host_fits(strict)
        result = _run([GroupKFold(3), *sequence], cohort, tmp_path / "fixed-late")
    with result:
        archive = result.export(tmp_path / "fixed-late.n4a")
        actual = nirs4all.predict(archive, new, engine="dag-ml")
        np.testing.assert_array_equal(actual.y_pred.ravel(), np.asarray(names)[expected.argmax(axis=1)])


def test_large_original_int64_labels_survive_fixed_archive_replay(tmp_path: Path) -> None:
    source = example.make_dataset(n_classes=2, numeric=True)
    label_map = {-9: -(1 << 62) + 5, 17: (1 << 62) + 3}
    labels = np.asarray([label_map[int(label)] for label in source.y], dtype=np.int64)
    cohort = MultimodalDataset(source.sources, sample_ids=source.sample_ids, y=labels, groups=source.groups,
        partitions=source.partitions, task_type="classification", name=source.name)
    new = example.make_dataset(29, prediction=True)
    with _run([GroupKFold(3), {"model": example.make_model(("nir",))}], cohort, tmp_path / "large-labels") as result:
        assert result.classes_.tolist() == sorted(label_map.values())
        archive = result.export(tmp_path / "large-labels.n4a")
    actual = nirs4all.predict(archive, new, engine="dag-ml")
    assert actual.y_pred.dtype.kind == "i" and set(actual.y_pred.ravel().tolist()) <= set(label_map.values())
    assert actual.metadata["classification"]["label_names"] == sorted(label_map.values())
