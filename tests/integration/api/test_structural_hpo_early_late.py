"""Native conditional topology HPO and independent grouped nested-OOF oracle."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import nirs4all
from nirs4all.api.portable_archive import read_portable_predictor_archive_v2
from nirs4all.operators.models.multimodal import MultimodalRegressor, TensorPCA
from nirs4all.pipeline.dagml.cancellation import DagRunCancelled
from nirs4all.pipeline.dagml.methods_multimodal import recipe_from_estimator
from nirs4all.pipeline.dagml.structural_topology import INNER_SPLITS
from nirs4all.pipeline.dagml.structural_tuning import _prepare_structure
from tests.integration.api import test_methods_multimodal_u07 as fixed
from tests.integration.api import test_structural_hpo_preprocessing_chains as chains

_PATH = Path(__file__).resolve().parents[3] / "examples/user/04_models/U21_structural_hpo_early_late.py"
_SPEC = importlib.util.spec_from_file_location("early_late_integration_example", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


@pytest.fixture(autouse=True)
def native_execution_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("legacy scheduler executed"))
    monkeypatch.setattr("nirs4all.pipeline.dagml.run_paths._run_model_on_precomputed_matrix", lambda *a, **k: pytest.fail("host CV scheduler executed"))


def _run(pipeline: Any, cohort: Any, root: Path, tuning: dict[str, Any]) -> Any:
    return nirs4all.run(pipeline, cohort, tuning=tuning, engine="dag-ml", workspace_path=root,
                        random_state=17, refit=True, verbose=0, save_charts=False)


def _branch_prediction(model: Any, cohort: Any, train: Any, prediction: Any, rows: Any) -> np.ndarray:
    """Independent sklearn declarations, including train-only category/PCA FIT."""
    encoded_train, encoded_prediction = [], []
    for name, transformer in model.transformers.items():
        encoder = clone(transformer).fit(cohort.sources[name].values[train], cohort.y[train])
        weight = (model.source_weights or {}).get(name, 1.0)
        encoded_train.append(np.asarray(encoder.transform(cohort.sources[name].values[train]), dtype=float) * weight)
        encoded_prediction.append(np.asarray(encoder.transform(prediction.sources[name].values[rows]), dtype=float) * weight)
    estimator = clone(model.model).fit(np.concatenate(encoded_train, axis=1), cohort.y[train])
    return np.asarray(estimator.predict(np.concatenate(encoded_prediction, axis=1))).reshape(-1)


def _oracle(sequence: list[Any], cohort: Any, train: Any, prediction: Any, rows: Any, params: dict[str, Any]) -> np.ndarray:
    """Fit a separate nested grouped OOF stack; never reuse native states."""
    if len(sequence) == 1:
        model = clone(sequence[0]["model"]).set_params(model__alpha=params["early.alpha"])
        return _branch_prediction(model, cohort, train, prediction, rows)
    branches = sequence[0]["branch"]
    oof = np.empty((len(train), len(branches)))
    projected = []
    folds = list(GroupKFold(INNER_SPLITS).split(np.zeros((len(train), 1)), groups=np.asarray(cohort.groups)[train]))
    for column, (name, steps) in enumerate(branches.items()):
        model = clone(steps[0]["model"]).set_params(model__alpha=params[f"late.{name}.alpha"])
        for inner_train, inner_validation in folds:
            assert not set(np.asarray(cohort.groups)[train[inner_train]]).intersection(np.asarray(cohort.groups)[train[inner_validation]])
            oof[inner_validation, column] = _branch_prediction(model, cohort, train[inner_train], cohort, train[inner_validation])
        projected.append(_branch_prediction(model, cohort, train, prediction, rows))
    meta = clone(sequence[2]["model"]).set_params(alpha=params["late.meta.alpha"]).fit(oof, cohort.y[train])
    return np.asarray(meta.predict(np.column_stack(projected))).reshape(-1)


def _sequence_for_entry(entry: dict[str, Any], pipeline: list[Any]) -> list[Any]:
    # Use compiler-resolved destinations and exact declarations. Native node
    # spelling is deliberately not an association contract.
    nodes = {node["id"]: node for node in entry["graph"]["nodes"] if node["kind"] == "model"}
    bindings = entry["parameter_bindings"]
    matches = []
    for sequence in pipeline[1]["_or_"]:
        if len(sequence) == 1:
            if set(bindings) != {"early.alpha"} or set(nodes) != {entry["target_node"]}:
                continue
            node = nodes[entry["target_node"]]
            expected_recipe = recipe_from_estimator(sequence[0]["model"], allow_source_selection=True)
            if (bindings["early.alpha"] == {"node_id": node["id"], "param_path": "model__alpha"}
                    and node["operator"]["type"] == "N4mMultimodalPipeline" and node["operator"]["recipe"] == expected_recipe):
                matches.append(sequence)
            continue
        branches = sequence[0]["branch"]
        axes = {f"late.{name}.alpha" for name in branches} | {"late.meta.alpha"}
        if set(bindings) != axes or len(nodes) != len(branches) + 1:
            continue
        meta = nodes[entry["target_node"]]
        expected_meta = {"type": "N4mRolePipeline", "source_order": list(branches), "steps": [{
            "methodId": "models.regularized.ridge",
            "params": {"alpha": float(sequence[2]["model"].alpha), "center_x": True, "center_y": True, "scale_x": False},
        }]}
        if (bindings["late.meta.alpha"] != {"node_id": meta["id"], "param_path": "alpha"}
                or meta["operator"] != expected_meta):
            continue
        active_nodes = {meta["id"]}
        for name, steps in branches.items():
            binding = bindings[f"late.{name}.alpha"]
            node = nodes[binding["node_id"]]
            if (binding["param_path"] != "model__alpha" or node["operator"]["type"] != "N4mMultimodalPipeline"
                    or node["operator"]["recipe"] != recipe_from_estimator(steps[0]["model"], allow_source_selection=True)):
                break
            active_nodes.add(node["id"])
        else:
            if active_nodes == set(nodes):
                matches.append(sequence)
    assert len(matches) == 1, f"Expected one exact public declaration for native recipe {entry['recipe_id']}, found {len(matches)}"
    return matches[0]


def _observe(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    from dag_ml.multimodal_methods import MethodsMultimodalController
    from dag_ml.multimodal_topology import MethodsTopologyController

    observed = []
    for cls in (MethodsMultimodalController, MethodsTopologyController):
        original = cls.__init__

        def create(controller: Any, *args: Any, _original: Any = original, **kwargs: Any) -> None:
            _original(controller, *args, **kwargs)
            controller._qualification_results = []
            observed.append(controller)

        monkeypatch.setattr(cls, "__init__", create)
        original_operator = cls.operator

        def operator(controller: Any, task: dict[str, Any], _original: Any = original_operator) -> dict[str, Any]:
            result = _original(controller, task)
            controller._qualification_results.append({"phase": task["phase"], "node_id": task["node_plan"]["node_id"], "result": copy.deepcopy(result)})
            return result

        monkeypatch.setattr(cls, "operator", operator)
    return observed


def _forbid_sklearn_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    for cls in (Ridge, StandardScaler, TensorPCA, OneHotEncoder, MultimodalRegressor):
        monkeypatch.setattr(cls, "fit", lambda *a, **k: pytest.fail("production performed Python encoder/model FIT"))


def _output_blocks(outcome: dict[str, Any], producer: str) -> list[dict[str, Any]]:
    return [block for output in outcome["outputs"]
            for surface in ("predictions", "aggregated_predictions") for block in output[surface]
            if block["producer_node"] == producer]


def _block_sample_ids(block: dict[str, Any]) -> set[str]:
    if "sample_ids" in block:
        return set(block["sample_ids"])
    return {unit["id"] for unit in block["unit_ids"] if unit["level"] == "sample"}


def _assert_heldout_rows_remain_test(result: Any, cohort: Any, producer: str, controllers: list[Any]) -> None:
    heldout = np.flatnonzero(np.asarray(cohort.partitions) == "test")
    heldout_ids = {cohort.sample_ids[index] for index in heldout}
    outcome = result._methods_multimodal_outcome.to_dict()
    assert len(outcome["outputs"]) == 1 and outcome["outputs"][0]["binding"]["node_id"] == producer
    assert result.structural_tuning_training_request["options"]["outputs"][0]["node_id"] == producer
    # FinalRefit names the deployed fitted state. Genuine meta held-out rows
    # retain Test scope in bound outputs, callbacks and native reports.
    blocks = _output_blocks(outcome, producer)
    callback_blocks = [block for controller in controllers for packet in controller._qualification_results
                       for block in packet["result"].get("predictions", []) if block["producer_node"] == producer]
    blocks.extend(callback_blocks)
    terminal = [block for controller in controllers for packet in controller._qualification_results if packet["phase"] == "REFIT"
                for block in packet["result"].get("predictions", [])
                if block["producer_node"] == producer and block.get("fold_id") is None and _block_sample_ids(block) & heldout_ids]
    assert terminal and all(block["partition"] == "test" and _block_sample_ids(block) == heldout_ids for block in terminal)
    for block in blocks:
        if _block_sample_ids(block) & heldout_ids:
            assert block["partition"] == "test"
    reports = [report for report in outcome["score_set"]["reports"]
               if report["producer_node"] == producer and report.get("fold_id") is None]
    assert any(report["partition"] == "test" for report in reports)
    terminal_node = next(node for node in outcome["effective_plan"]["graph_plan"]["graph"]["nodes"] if node["id"] == producer)
    if terminal_node["operator"]["type"] == "N4mRolePipeline":
        assert all(report["partition"] == "test" for report in reports)
        bound = outcome["outputs"][0]
        assert "artifact_only" not in bound
        assert bound["refit_test_cohort"]["role"] == "external_test"
        assert set(bound["refit_test_cohort"]["physical_sample_ids"]) == heldout_ids
        bound_blocks = _output_blocks(outcome, producer)
        assert bound_blocks and all(block["partition"] == "test" and block.get("fold_id") is None
                                    and _block_sample_ids(block) == heldout_ids for block in bound_blocks)
    rows = result.predictions.filter_predictions(partition="test", fold_id="final")
    assert len(rows) == 1 and set(rows[0]["sample_indices"]) == set(heldout)
    for partition in ("train", "val"):
        for row in result.predictions.filter_predictions(partition=partition):
            indices = row.get("sample_indices")
            assert not set(indices if indices is not None else []).intersection(heldout)


def test_each_declared_topology_matches_independent_nested_grouped_oof_and_exact_winner_closure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cohort, pipeline = example.make_dataset(), example.make_pipeline()
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 5}
    prepared = _prepare_structure(pipeline, cohort, tuning, {"random_state": 17})
    catalogue = prepared["catalogue"]
    assert catalogue["schema_version"] == 2 and len(catalogue["entries"]) == 5
    chains._enqueue_recipes(monkeypatch, catalogue, catalogue["entries"])
    controllers = _observe(monkeypatch)
    train = np.flatnonzero(np.asarray(cohort.partitions) == "train")
    test = np.flatnonzero(np.asarray(cohort.partitions) == "test")
    folds = list(GroupKFold(3).split(np.zeros((len(train), 1)), groups=np.asarray(cohort.groups)[train]))
    recipes = {entry["recipe_id"]: entry for entry in catalogue["entries"]}
    with monkeypatch.context() as strict:
        _forbid_sklearn_fit(strict)
        result = _run(pipeline, cohort, tmp_path / "workspace", tuning)
    with result:
        evidence = result.structural_tuning_evidence
        assert {trial["params"]["__recipe__"] for trial in evidence["trials"]} == set(recipes)
        for trial in evidence["trials"]:
            entry = recipes[trial["params"]["__recipe__"]]
            assert set(trial["params"]) == {"__recipe__", *entry["parameter_bindings"]}
            sequence = _sequence_for_entry(entry, pipeline)
            expected = {f"fold{index}": float(np.sqrt(np.mean((_oracle(sequence, cohort, train[fit], cohort, train[heldout], trial["params"]) - cohort.y[train[heldout]]) ** 2)))
                        for index, (fit, heldout) in enumerate(folds)}
            assert trial["objective_fold_scores"] == pytest.approx(expected, rel=2e-7, abs=2e-7)
        winner = recipes[evidence["selected_params"]["__recipe__"]]
        _assert_heldout_rows_remain_test(result, cohort, winner["target_node"], controllers)
        expected = _oracle(_sequence_for_entry(winner, pipeline), cohort, train, cohort, test[::-1], evidence["selected_params"])
        with monkeypatch.context() as strict:
            _forbid_sklearn_fit(strict)
            archive = result.export(tmp_path / "winner.n4a")
            actual = nirs4all.predict(archive, cohort.take([cohort.sample_ids[index] for index in test[::-1]]), engine="dag-ml")
        np.testing.assert_allclose(actual.y_pred.ravel(), expected, rtol=2e-7, atol=2e-7)
        assert actual.metadata["training_performed"] is False
        package = read_portable_predictor_archive_v2(archive).to_dict()
        records = package["execution_bundle"]["refit_artifacts"]
        models = [node for node in winner["graph"]["nodes"] if node["kind"] == "model"]
        assert {record["node_id"] for record in records} == {node["id"] for node in models}
        assert len(records) == len(models)
        assert {record["artifact"]["kind"] for record in records} <= {"methods_multimodal_pipeline", "methods_role_pipeline"}
        assert package["output_bindings"][0]["node_id"] == winner["target_node"]
    audit = [event for controller in controllers for event in controller.audit]
    raw_fits = [event for event in audit if event["operation"] == "fit" and "source_order" in event]
    assert raw_fits and any(len(event["sample_ids"]) < min(len(train[fit]) for fit, _ in folds) for event in raw_fits)
    for event in raw_fits:
        assert not set(event["sample_ids"]).intersection(cohort.sample_ids[index] for index in test)
        if len(event["source_order"]) == 1:
            assert set(event["raw_shapes"]) == set(event["source_order"])
    assert any(event["operation"] == "fit" and "steps" in event for event in audit)


def test_train_only_late_refit_captures_artifacts_without_fabricating_terminal_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from n4m import MultimodalPipeline
    from n4m.roles import RolePipeline

    complete = example.make_dataset()
    cohort = complete.take([sample for sample, partition in zip(complete.sample_ids, complete.partitions, strict=True) if partition == "train"])
    new, pipeline = example.make_dataset(29, prediction=True), example.make_pipeline()
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 1}
    prepared = _prepare_structure(pipeline, cohort, tuning, {})
    entry = next(entry for entry in prepared["catalogue"]["entries"] if _sequence_for_entry(entry, pipeline) is pipeline[1]["_or_"][2])
    chains._enqueue_recipes(monkeypatch, prepared["catalogue"], [entry])
    controllers = _observe(monkeypatch)
    with monkeypatch.context() as strict:
        _forbid_sklearn_fit(strict)
        result = _run(pipeline, cohort, tmp_path / "workspace", tuning)
    with result:
        producer = entry["target_node"]
        outcome = result._methods_multimodal_outcome.to_dict()
        assert outcome["refit"]["status"] == "completed"
        assert len(outcome["outputs"]) == 1
        bound = outcome["outputs"][0]
        assert bound["binding"]["node_id"] == producer and bound["artifact_only"] is True
        assert "refit_test_cohort" not in bound
        assert all(bound[surface] == [] for surface in ("predictions", "observation_predictions", "aggregated_predictions"))
        assert all(block["partition"] not in {"test", "final"} for block in _output_blocks(outcome, producer))
        reports = [report for report in outcome["score_set"]["reports"] if report["producer_node"] == producer]
        assert reports and all(report["partition"] not in {"test", "final"} and report.get("fold_id") is not None for report in reports)
        assert result.predictions.filter_predictions(fold_id="final") == []
        refit_packets = [packet for controller in controllers for packet in controller._qualification_results
                         if packet["phase"] == "REFIT" and packet["node_id"] == producer]
        assert len(refit_packets) == 1 and not refit_packets[0]["result"].get("predictions")
        refit = [event for event in result.methods_multimodal_audit
                 if event["operation"] == "fit" and event["node"] == producer and event.get("fold") is None]
        assert len(refit) == 1 and set(refit[0]["sample_ids"]) == set(cohort.sample_ids)
        expected = _oracle(pipeline[1]["_or_"][2], cohort, np.arange(len(cohort.sample_ids)), new,
                           np.arange(len(new.sample_ids)), result.tuning_best_params)
        archive = result.export(tmp_path / "train-only-late.n4a")
        package = read_portable_predictor_archive_v2(archive).to_dict()
        records = package["execution_bundle"]["refit_artifacts"]
        assert len(records) == 3 and {record["node_id"] for record in records} == {node["id"] for node in entry["graph"]["nodes"] if node["kind"] == "model"}
        assert {record["artifact"]["kind"] for record in records} == {"methods_multimodal_pipeline", "methods_role_pipeline"}
    shutil.rmtree(tmp_path / "study")
    if (tmp_path / "workspace").exists():
        shutil.rmtree(tmp_path / "workspace")
    with monkeypatch.context() as strict:
        _forbid_sklearn_fit(strict)
        for cls in (MultimodalPipeline, RolePipeline):
            strict.setattr(cls, "fit", lambda *a, **k: pytest.fail("artifact-only REFIT replay reached native FIT"))
        actual = nirs4all.predict(archive, new, engine="dag-ml")
    np.testing.assert_allclose(actual.y_pred.ravel(), expected, rtol=2e-7, atol=2e-7)
    assert actual.metadata["training_performed"] is False


@pytest.mark.parametrize("mutation", ["alternative_order", "branch_order", "branch_source", "meta_alpha", "weight", "schema", "excluded_train", "excluded_test", "targets", "groups", "axis"])
def test_resume_rejects_complete_input_or_topology_mutations_before_model_callbacks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str) -> None:
    from dag_ml import DagMlRuntimeError
    from dag_ml.multimodal_methods import MethodsMultimodalController
    from dag_ml.multimodal_topology import MethodsTopologyController
    from nirs4all_io import MultimodalDataset

    cohort, pipeline = example.make_dataset(), example.make_pipeline()
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 5}
    prepared = _prepare_structure(pipeline, cohort, tuning, {})
    early = next(entry for entry in prepared["catalogue"]["entries"] if set(entry["parameter_bindings"]) == {"early.alpha"})
    chains._enqueue_recipes(monkeypatch, prepared["catalogue"], [early])
    with pytest.raises(DagRunCancelled):
        _run(pipeline, cohort, tmp_path / "workspace", {**tuning, "progress_callback": lambda event: len(event["checkpoint"]["trials"]) < 1})
    paths = list((tmp_path / "study").rglob("*"))
    before = {path: path.read_bytes() for path in paths if path.is_file()}
    assert before
    late = pipeline[1]["_or_"][1]
    if mutation == "alternative_order":
        pipeline[1]["_or_"] = list(reversed(pipeline[1]["_or_"]))
    elif mutation == "branch_order":
        late[0]["branch"] = dict(reversed(list(late[0]["branch"].items())))
    elif mutation == "branch_source":
        pipeline[1]["_or_"][1] = example.make_late(("nir", "series"))
    elif mutation == "meta_alpha":
        late[2]["model"].alpha = 2.0
    elif mutation == "weight":
        late[0]["branch"]["image"][0]["model"].source_weights = {"image": 0.0}
    elif mutation == "schema":
        cohort = fixed._replace(cohort, "metadata", cohort.sources["metadata"].values, feature_names=["modified", "category"])
    elif mutation in {"excluded_train", "excluded_test"}:
        values = np.array(cohort.sources["series"].values, copy=True)
        partition = "train" if mutation.endswith("train") else "test"
        values[np.flatnonzero(np.asarray(cohort.partitions) == partition)[0], 0, 0] += 0.1
        cohort = fixed._replace(cohort, "series", values)
    elif mutation in {"targets", "groups"}:
        changed = np.array(cohort.y, copy=True) if mutation == "targets" else list(cohort.groups)
        changed[0] = changed[0] + 1 if mutation == "targets" else "changed-group"
        cohort = MultimodalDataset(dict(cohort.sources), sample_ids=cohort.sample_ids,
                                   y=changed if mutation == "targets" else cohort.y,
                                   groups=changed if mutation == "groups" else cohort.groups,
                                   partitions=cohort.partitions, target_names=cohort.target_names, name=cohort.name)
    else:
        tuning["space"]["late.meta.alpha"] = [0.1, 1.0]
    for cls in (MethodsMultimodalController, MethodsTopologyController):
        monkeypatch.setattr(cls, "operator", lambda *a, **k: pytest.fail("resume mismatch reached model callback"))
    with pytest.raises((ValueError, RuntimeError, DagMlRuntimeError)) as refused:
        _run(pipeline, cohort, tmp_path / "resume", {**tuning, "resume": True})
    if mutation in {"excluded_train", "excluded_test", "targets"}:
        assert isinstance(refused.value, DagMlRuntimeError) and "binding mismatch" in str(refused.value)
    assert {path: path.read_bytes() for path in before} == before


def test_native_pca_preflight_refuses_invalid_inner_training_bound_without_callbacks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dag_ml import DagMlRuntimeError
    from dag_ml.multimodal_topology import MethodsTopologyController

    cohort, pipeline = example.make_dataset(), example.make_pipeline()
    train = np.flatnonzero(np.asarray(cohort.partitions) == "train")
    outer = list(GroupKFold(3).split(np.zeros((len(train), 1)), groups=np.asarray(cohort.groups)[train]))
    inner_sizes = [len(inner_train) for fit, _ in outer for inner_train, _ in GroupKFold(INNER_SPLITS).split(np.zeros((len(fit), 1)), groups=np.asarray(cohort.groups)[train[fit]])]
    components = min(inner_sizes) + 1
    assert components <= min(len(fit) for fit, _ in outer)
    pipeline[1]["_or_"][1][0]["branch"]["image"][0]["model"].transformers["image"].n_components = components
    monkeypatch.setattr(MethodsTopologyController, "operator", lambda *a, **k: pytest.fail("invalid PCA reached FIT"))
    with pytest.raises(DagMlRuntimeError, match="PCA components exceed raw width or native training scope"):
        _prepare_structure(pipeline, cohort, example.make_tuning(tmp_path / "study"), {})


def test_stop_resume_matches_continuous_native_conditional_history(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cohort, pipeline = example.make_dataset(), example.make_pipeline()
    tuning = {**example.make_tuning(tmp_path / "continuous-study"), "n_trials": 4}
    with _run(pipeline, cohort, tmp_path / "continuous", tuning) as complete:
        expected = [(trial["params"], trial["score"]) for trial in complete.structural_tuning_evidence["trials"]]
    resumed = {**tuning, "storage": (tmp_path / "resumed-study").resolve().as_uri()}
    with pytest.raises(DagRunCancelled):
        _run(pipeline, cohort, tmp_path / "first", {**resumed, "progress_callback": lambda event: len(event["checkpoint"]["trials"]) < 2})
    with _run(pipeline, cohort, tmp_path / "second", {**resumed, "resume": True}) as result:
        actual = [(trial["params"], trial["score"]) for trial in result.structural_tuning_evidence["trials"]]
        assert actual == expected


def test_fresh_installed_late_winner_replays_exact_mixed_closure_without_fit_or_hpo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    installed = os.environ.get("NIRS4ALL_EARLY_LATE_INSTALLED_PYTHON")
    if not installed:
        if os.environ.get("NIRS4ALL_REQUIRE_EARLY_LATE_INSTALLED") == "1":
            pytest.fail("mandatory installed topology proof requires NIRS4ALL_EARLY_LATE_INSTALLED_PYTHON")
        pytest.skip("fresh installed Python is supplied by the qualification gate")
    cohort, new, pipeline = example.make_dataset(), example.make_dataset(29, prediction=True), example.make_pipeline()
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 1}
    prepared = _prepare_structure(pipeline, cohort, tuning, {})
    entry = next(entry for entry in prepared["catalogue"]["entries"] if _sequence_for_entry(entry, pipeline) is pipeline[1]["_or_"][2])
    chains._enqueue_recipes(monkeypatch, prepared["catalogue"], [entry])
    train = np.flatnonzero(np.asarray(cohort.partitions) == "train")
    controllers = _observe(monkeypatch)
    with _run(pipeline, cohort, tmp_path / "workspace", tuning) as result:
        _assert_heldout_rows_remain_test(result, cohort, entry["target_node"], controllers)
        expected = _oracle(pipeline[1]["_or_"][2], cohort, train, new, np.arange(len(new.sample_ids)), result.tuning_best_params)
        archive = result.export(tmp_path / "late.n4a")
        document = read_portable_predictor_archive_v2(archive).to_dict()
        assert len(document["execution_bundle"]["refit_artifacts"]) == 3
        assert {record["artifact"]["kind"] for record in document["execution_bundle"]["refit_artifacts"]} == {"methods_multimodal_pipeline", "methods_role_pipeline"}
    if (tmp_path / "workspace").exists():
        shutil.rmtree(tmp_path / "workspace")
    shutil.rmtree(tmp_path / "study")
    (tmp_path / "cohort.json").write_text(json.dumps(new.to_dict(), allow_nan=False))
    root = Path(nirs4all.__file__).resolve().parent
    sources = ["pipeline/dagml/structural_topology.py", "pipeline/dagml/structural_tuning.py", "pipeline/dagml/methods_multimodal.py"]
    evidence = {"prediction": expected.tolist(), "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}}
    (tmp_path / "expected.json").write_text(json.dumps(evidence, allow_nan=False))
    script = textwrap.dedent("""\
        import hashlib, json, pathlib, sys
        import dag_ml, numpy as np, nirs4all
        from n4m import MultimodalPipeline
        from n4m.roles import RolePipeline
        from n4m.model_selection.optimizer import Optimizer
        from nirs4all_io import MultimodalDataset
        from nirs4all.pipeline.dagml.host_search_checkpoint import HostSearchOptimizer
        expected = json.loads(pathlib.Path(sys.argv[3]).read_text())
        root = pathlib.Path(nirs4all.__file__).resolve().parent
        assert 'site-packages' in root.parts, root
        for name, digest in expected['source_sha256'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
        archive = pathlib.Path(sys.argv[1])
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected['archive_sha256']
        def forbidden(*args, **kwargs):
            raise AssertionError('late winner replay reached FIT/HPO')
        MultimodalPipeline.fit = forbidden
        RolePipeline.fit = forbidden
        Optimizer.__init__ = forbidden
        Optimizer.load = classmethod(forbidden)
        HostSearchOptimizer.__init__ = forbidden
        for name in ('run_host_hpo_search_in_process', 'execute_training', 'prepare_host_hpo_topology_catalogue', 'resolve_host_hpo_structural_winner'):
            setattr(dag_ml, name, forbidden)
        original_replay = dag_ml.replay_loaded_predictor_package
        def checked_replay(*args, **kwargs):
            outcome = original_replay(*args, **kwargs)
            outputs = outcome.to_dict()['outputs']
            assert len(outputs) == 1
            assert 'artifact_only' not in outputs[0] and 'refit_test_cohort' not in outputs[0]
            blocks = outputs[0]['predictions'] + outputs[0]['aggregated_predictions']
            assert blocks and all(block['partition'] == 'final' and block.get('fold_id') is None for block in blocks)
            return outcome
        dag_ml.replay_loaded_predictor_package = checked_replay
        nirs4all.run = forbidden
        cohort = MultimodalDataset.from_dict(json.loads(pathlib.Path(sys.argv[2]).read_text()))
        result = nirs4all.predict(archive, cohort, engine='dag-ml')
        np.testing.assert_allclose(result.y_pred.ravel(), expected['prediction'], rtol=2e-7, atol=2e-7)
        assert result.metadata['training_performed'] is False
        malformed = MultimodalDataset({name:source for name,source in cohort.sources.items() if name != 'series'},
                                      sample_ids=cohort.sample_ids, partitions=cohort.partitions, name=cohort.name)
        try:
            nirs4all.predict(archive, malformed, engine='dag-ml')
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError('archive accepted removal of excluded raw source')
        print(json.dumps({'installed':str(root),'training_performed':False,'mixed_closure':True}))
    """)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    process = subprocess.run([installed, "-I", "-c", script, str(archive), str(tmp_path / "cohort.json"), str(tmp_path / "expected.json")],
                             cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=180, check=False)
    assert process.returncode == 0, process.stderr[-6000:]
    assert json.loads(process.stdout.splitlines()[-1])["mixed_closure"] is True
