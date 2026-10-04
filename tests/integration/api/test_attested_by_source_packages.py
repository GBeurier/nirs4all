"""Actual native multi-output training, ranked capture and installed archive replay."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import dag_ml
import numpy as np
import pytest
from nirs4all_io import MultimodalDataset, TensorSource
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nirs4all
from nirs4all.pipeline.bundle.loader import BundleLoader


def _data(kind):
    rng = np.random.default_rng(409)
    size = 84
    latent = rng.normal(size=(size, 3))
    ids = [f"mm09.{index}" for index in range(size)]
    sources = {f"sensor{index}": TensorSource(latent + index / 9 + rng.normal(scale=0.04, size=latent.shape), ids,
                                            representation_id="tabular_numeric") for index in range(2)}
    if kind == "regression":
        y = latent @ [2, -0.7, 0.4]
    else:
        axis = np.array(["fruit🧬", "fruité", "fruit水"]) if kind == "string" else np.array([-(2**62), 2**53 + 17, 2**62 - 1], dtype=np.int64)
        y = axis[np.arange(size) % 3]
    return MultimodalDataset(sources, sample_ids=ids, y=y, target_names=["y"],
        task_type="regression" if kind == "regression" else "classification", groups=[f"fold-group-{index // 6}" for index in range(size)],
        partitions=["train"] * 72 + ["test"] * 12, name=f"mm09-{kind}")


def _pipeline(dataset, mode):
    def head():
        return Ridge(alpha=0.4) if dataset.task_type == "regression" else LogisticRegression(C=0.7, max_iter=1000)
    bodies = {name: ([StandardScaler()] if mode == "static" else [{"_or_": [StandardScaler(), MinMaxScaler()]}]) + [{"model": head()}]
              for name in dataset.sources}
    return [GroupKFold(3), {"branch": {"by_source": True, "steps": bodies}}, {"merge": "auto"}]


def _assert_generator_package_influence(capture):
    """Rust validates genuine union training evidence and a selected-only deployment."""
    from nirs4all.pipeline.dagml.training_contracts import tcv1_fingerprint_without, tcv1_sha256

    package = capture["package"]
    assert package["training_influence"] == capture["outcome"]["training_influence"]
    selected = set(package["predictor_node_ids"])
    influence_nodes = {entry["node_id"] for entry in package["training_influence"]["entries"] if entry.get("node_id")}
    assert selected < influence_nodes, "generator training must retain evaluated alternatives"
    assert selected == {node["id"] for node in package["effective_plan"]["graph_plan"]["graph"]["nodes"]}

    foreign = json.loads(json.dumps(package))
    foreign["training_influence"]["entries"][0]["node_id"] += ".foreign"
    foreign["training_influence"]["manifest_fingerprint"] = tcv1_fingerprint_without(foreign["training_influence"], "manifest_fingerprint")
    foreign["training_outcome"]["training_influence_fingerprint"] = foreign["training_influence"]["manifest_fingerprint"]
    foreign["package_fingerprint"] = tcv1_fingerprint_without(foreign, "package_fingerprint")
    with pytest.raises(dag_ml.DagMlValidationError, match="influence references a node outside predictor closure"):
        dag_ml.PortablePredictorPackage(foreign)

    # Rebuild all graph/plan/template/bundle anchors through the native builder;
    # these refusals must reach compiler pruning, not a stale-hash guard.
    for mutate_dsl in (True, False):
        changed = json.loads(json.dumps(package))
        old_plan = changed["effective_plan"]
        graph = old_plan["graph_plan"]["graph"]
        if mutate_dsl:
            graph["metadata"]["training_operator_source_dsl"]["metadata"]["foreign_marker"] = "changed"
        else:
            graph["nodes"][0]["metadata"]["foreign_marker"] = "changed"
        # The retained campaign binds the entire training union. For the
        # graph-hash builder only, restrict bindings to its selected nodes;
        # preserve the original campaign/variants in the tampered package.
        selected_campaign = json.loads(json.dumps(old_plan["campaign"]))
        node_ids = {node["id"] for node in graph["nodes"]}
        selected_campaign["data_bindings"] = {node: bindings for node, bindings in selected_campaign["data_bindings"].items() if node in node_ids}
        rebuilt_graph = dag_ml.build_execution_plan(old_plan["id"], graph, selected_campaign, list(old_plan["controller_manifests"].values())).to_dict()
        rebuilt = old_plan
        rebuilt["graph_fingerprint"] = rebuilt_graph["graph_fingerprint"]
        dag_ml.ExecutionPlan(rebuilt)
        changed["effective_plan"] = rebuilt
        changed["template"]["graph"] = graph
        changed["template"]["template_fingerprint"] = tcv1_fingerprint_without(changed["template"], "template_fingerprint")
        changed["training_outcome"]["effective_plan_fingerprint"] = tcv1_sha256(rebuilt)
        changed["execution_bundle"]["graph_fingerprint"] = rebuilt["graph_fingerprint"]
        changed["training_outcome"]["execution_bundle_fingerprint"] = tcv1_sha256(changed["execution_bundle"])
        changed["package_fingerprint"] = tcv1_fingerprint_without(changed, "package_fingerprint")
        with pytest.raises(dag_ml.DagMlValidationError, match="operator predictor graph or variant inventory differs from signed native pruning"):
            dag_ml.PortablePredictorPackage(changed)


@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
@pytest.mark.parametrize("kind", ["regression", "string", "int64"])
@pytest.mark.parametrize("mode", ["static", "generator", "top_k"])
def test_real_by_source_training_retains_native_packages_and_ranked_replay(tmp_path, monkeypatch, mechanism, kind, mode):
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "pyo3" else "0")
    if mechanism == "cli":
        cli = os.environ.get("N4A_DAGML_CLI")
        assert cli and Path(cli).is_file(), "MM09 CLI gate requires the fresh installed candidate CLI"
    monkeypatch.setattr("nirs4all.pipeline.dagml.run_paths.run_cv_refit_bundle", lambda *args, **kwargs: pytest.fail("old bundle capture route executed"))
    dataset = _data(kind)
    top_k = 2 if mode == "top_k" else 1
    result = nirs4all.run(_pipeline(dataset, mode), dataset, engine="dag-ml", random_state=43, refit={"top_k": top_k},
                         save_artifacts=True, save_charts=False, verbose=0, workspace_path=tmp_path / "workspace")
    try:
        captures = result._dagml_source_training_captures
        assert len(captures) == top_k
        assert len({item["outcome"]["selected_variant_id"] for item in captures}) == top_k
        expected_outputs = {"output:source_0", "output:source_1"}
        for rank, item in enumerate(captures, 1):
            dag_ml.TrainingOutcome(item["outcome"])
            dag_ml.PortablePredictorPackage(item["package"])
            if mode != "static":
                _assert_generator_package_influence(item)
            decision = next(iter(item["outcome"]["execution_bundle"]["selections"].values()))
            assert decision.get("requested_rank", 1) == rank
            assert decision["selected_candidate_id"] == decision["ranked_candidates"][rank - 1]["candidate_id"]
            assert {output["binding_id"] for output in item["package"]["output_bindings"]} == expected_outputs
            assert all(output["prediction_kind"] == ("regression_point" if kind == "regression" else "class_label")
                       for output in item["package"]["output_bindings"])
            if rank > 1:
                assert decision["ranked_candidates"] == next(iter(captures[0]["outcome"]["execution_bundle"]["selections"].values()))["ranked_candidates"]
        archive = result.export(tmp_path / "all.n4a")
        loader = BundleLoader(archive)
        expected_ids = tuple(f"output:source_{source}" + (f":rank:{rank}" if rank > 1 else "") for rank in range(1, top_k + 1) for source in range(2))
        assert loader.named_outputs == expected_ids
        blocks = [np.asarray(source.values)[72:] for source in dataset.sources.values()]
        matrix = np.concatenate(blocks, axis=1)
        replay = loader.predict_outputs(matrix)
        with zipfile.ZipFile(archive) as contents:
            capture = json.loads(contents.read("dagml_source_training_capture.json"))
            assert len(capture["captures"]) == top_k
            assert len(capture["sidecars"]) == 2 * top_k
        native_records = {item["artifact_id"]: item for item in result._dagml_refit_artifacts}
        from nirs4all.api.result import _DagmlExportedModel

        for output in loader._named_output_model().native_outputs:
            artifact = native_records[output["artifact_id"]]
            oracle = _DagmlExportedModel(artifact["estimator"], artifact["y_transform"]).predict(blocks[output["source_index"]])
            if kind == "regression":
                np.testing.assert_allclose(replay[output["output_binding_id"]], oracle, atol=1e-12, rtol=1e-12)
            else:
                np.testing.assert_array_equal(replay[output["output_binding_id"]], oracle)
                assert set(np.asarray(oracle).ravel()).issubset(set(dataset.y.ravel()))
        # Independent sklearn train-only reference. Selected encoder declarations
        # are read from the captured chain, but its learned means/weights are never reused.
        from sklearn.base import clone

        from nirs4all.pipeline.dagml.node_runner import _FrozenTransform

        all_sources = [np.asarray(source.values) for source in dataset.sources.values()]
        for output in loader._named_output_model().native_outputs:
            artifact = native_records[output["artifact_id"]]
            fitted = artifact["estimator"]
            encoders = []
            assert isinstance(fitted, Pipeline)
            for _name, step in fitted.steps[:-1]:
                if isinstance(step, _FrozenTransform):
                    declared = step.transformer.steps
                else:
                    declared = [step]
                for encoder in declared:
                    assert isinstance(encoder, (StandardScaler, MinMaxScaler))
                    encoders.append(clone(encoder))
            head = Ridge(alpha=0.4) if kind == "regression" else LogisticRegression(C=0.7, max_iter=1000)
            # Numeric public targets pass through TargetConverter's float32
            # storage before the native resolver exposes float64 values.
            training_y = dataset.y[:72].ravel()
            if kind == "regression":
                training_y = training_y.astype(np.float32).astype(np.float64)
            independent = make_pipeline(*encoders, head).fit(all_sources[output["source_index"]][:72], training_y)
            reference = independent.predict(all_sources[output["source_index"]][72:])
            actual = np.asarray(replay[output["output_binding_id"]]).ravel()
            if kind == "regression":
                np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=1e-10)
            else:
                np.testing.assert_array_equal(actual, reference)
        selected = next(row for row in result.predictions.filter_predictions(load_arrays=True) if row.get("partition") == "test" and row.get("fold_id") == "final")
        selected_archive = result.export(tmp_path / "selected.n4a", source=selected)
        selected_loader = BundleLoader(selected_archive)
        selected_index = int(selected["branch_id"])
        selected_prediction = np.asarray(selected["y_pred"]).ravel()
        if kind != "regression":
            # Stored score rows carry encoded classes; archive prediction
            # restores the original typed labels in the train-only vocabulary.
            codes = selected_prediction.astype(np.int64)
            np.testing.assert_array_equal(selected_prediction, codes)
            selected_prediction = np.unique(dataset.y[:72])[codes]
        np.testing.assert_array_equal(np.asarray(selected_loader.predict(blocks[selected_index])).ravel(), selected_prediction)
    finally:
        result.close()


def test_ranked_source_archive_cold_installed_replay_never_fits(tmp_path, monkeypatch):
    python = os.environ.get("NIRS4ALL_BY_SOURCE_INSTALLED_PYTHON")
    if not python:
        if os.environ.get("NIRS4ALL_REQUIRE_BY_SOURCE_INSTALLED") == "1":
            pytest.fail("MM09 requires an explicit fresh installed child for cold replay")
        pytest.skip("opt-in installed MM09 replay child not configured")
    dataset = _data("int64")
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    result = nirs4all.run(_pipeline(dataset, "top_k"), dataset, engine="dag-ml", random_state=43, refit={"top_k": 2},
                         save_artifacts=True, save_charts=False, verbose=0, workspace_path=tmp_path / "workspace")
    archive = result.export(tmp_path / "cold.n4a")
    matrix = np.concatenate([np.asarray(source.values)[72:] for source in dataset.sources.values()], axis=1)
    expected = {key: np.asarray(value).reshape(-1).tolist() for key, value in BundleLoader(archive).predict_outputs(matrix).items()}
    result.close()
    shutil.rmtree(tmp_path / "workspace")
    data_path = tmp_path / "inputs.npy"
    np.save(data_path, matrix)
    program = '''import json,sys,numpy as np,nirs4all,dag_ml
from sklearn.linear_model import LogisticRegression,Ridge
from nirs4all.pipeline.bundle.loader import BundleLoader
from nirs4all.pipeline.dagml import node_runner
expected_root=sys.prefix
assert nirs4all.__file__.startswith(expected_root) and dag_ml.__file__.startswith(expected_root)
def forbidden(*a,**k): raise AssertionError("FIT or search during cold replay")
LogisticRegression.fit=Ridge.fit=dag_ml.execute_training=forbidden
loader=BundleLoader(sys.argv[1])
values={key:nirs4all.predict(sys.argv[1],np.load(sys.argv[2]),output=key).y_pred for key in loader.named_outputs}
print(json.dumps({key: np.asarray(value).reshape(-1).tolist() for key,value in values.items()}))
'''
    terminal = subprocess.run([python, "-I", "-c", program, str(archive), str(data_path)], capture_output=True, text=True, check=False)
    assert terminal.returncode == 0, terminal.stdout + terminal.stderr
    assert json.loads(terminal.stdout.splitlines()[-1]) == expected


@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
@pytest.mark.parametrize("kind", ["regression", "string", "int64"])
def test_native_source_packages_preserve_experimental_units(tmp_path, monkeypatch, mechanism, kind):
    """Unequal repetition counts affect native FIT, never split identities."""
    from collections import Counter

    from sklearn.pipeline import Pipeline

    from nirs4all.pipeline.dagml import attested_by_source

    rng = np.random.default_rng(149)
    repeats = [1, 2, 3, 4] * 4
    units = np.repeat(np.arange(16), repeats)
    latent = rng.normal(size=(16, 3))
    ids = [f"unit-observation.{row}" for row in range(len(units))]
    if kind == "regression":
        y = (latent @ [1.4, -.8, .2])[units]
    else:
        labels = np.asarray(["classé", "class🧬", "class水"]) if kind == "string" else np.asarray([-(2**62), 2**53 + 17, 2**62 - 1], dtype=np.int64)
        y = labels[(units % 3)]
    sources = {f"sensor{index}": TensorSource(latent[units] + rng.normal(scale=.2, size=(len(units), 3)), ids,
                representation_id="tabular_numeric") for index in range(2)}
    cohort = MultimodalDataset(sources, sample_ids=ids, y=y, target_names=["y"],
        task_type="regression" if kind == "regression" else "classification",
        partitions=["train" if unit < 12 else "test" for unit in units],
        groups=[f"split_batch_{unit // 2}" for unit in units],
        independent_unit_ids=[f"unit:{unit}" for unit in units],
        repetition_ids=[f"repeat_{index}" for count in repeats for index in range(count)], name="mm09-unit-design")
    tasks = []
    weighted_calls = []
    current_task = [None]
    original = attested_by_source.run_node

    def observe(task, *args, **kwargs):
        if task["phase"] in {"FIT_CV", "REFIT"} and task["node_plan"]["kind"] == "model":
            tasks.append(json.loads(json.dumps(task)))
        current_task[0] = task
        try:
            return original(task, *args, **kwargs)
        finally:
            current_task[0] = None

    def record_weighted_fit(component, x, weights):
        task = current_task[0]
        if task is None:  # Independent oracle fits below have no native task.
            return
        influence = task["fit_influence"]
        assert len(x) == len(influence["fit_sample_ids"])
        np.testing.assert_array_equal(weights, influence["row_weights"])
        weighted_calls.append((task["phase"], task["node_plan"]["node_id"], component))

    encoder_fit = StandardScaler.fit
    head_class = Ridge if kind == "regression" else LogisticRegression
    head_fit = head_class.fit

    def observe_encoder_fit(self, x, y=None, sample_weight=None):
        record_weighted_fit("encoder", x, sample_weight)
        return encoder_fit(self, x, y, sample_weight=sample_weight)

    def observe_head_fit(self, x, y, sample_weight=None):
        record_weighted_fit("head", x, sample_weight)
        return head_fit(self, x, y, sample_weight=sample_weight)

    monkeypatch.setattr(StandardScaler, "fit", observe_encoder_fit)
    monkeypatch.setattr(head_class, "fit", observe_head_fit)
    monkeypatch.setattr(attested_by_source, "run_node", observe)
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "pyo3" else "0")
    if mechanism == "cli":
        cli = os.environ.get("N4A_DAGML_CLI")
        assert cli and Path(cli).is_file(), "MM09 CLI unit gate requires current candidate CLI"
    # MM03 admits model-owned weighted preprocessing. By-source _or_ only
    # generates separate X transforms, so this weighted profile uses fixed
    # Pipeline models; ranked preprocessing remains explicitly unsupported.
    bodies = {}
    for source in cohort.sources:
        head = Ridge(alpha=.4) if kind == "regression" else LogisticRegression(C=.7, max_iter=1000)
        bodies[source] = [{"model": Pipeline([("encoder", StandardScaler()), ("head", head)])}]
    weighted_pipeline = [GroupKFold(3), {"branch": {"by_source": True, "steps": bodies}}, {"merge": "auto"}]
    result = nirs4all.run(weighted_pipeline, cohort, engine="dag-ml", random_state=43,
                         refit={"top_k": 1}, save_artifacts=True, save_charts=False, verbose=0, workspace_path=tmp_path / "workspace")
    try:
        captures = result._dagml_source_training_captures
        assert len(captures) == 1
        for capture in captures:
            plan = capture["outcome"]["effective_plan"]
            descriptor = plan["graph_plan"]["graph"]["metadata"]["experimental_unit"]
            assert descriptor["fit_influence_policy"] == "equal_sample_influence"
            assert descriptor["independent_unit_ids"] == [f"unit:{unit}" for unit in units if unit < 12]
            decision = next(iter(capture["outcome"]["execution_bundle"]["selections"].values()))
            assert decision["metric_level"] == "group"
            assert plan["campaign"]["aggregation_policy"]["grouping_key"] == {"kind": "relation_metadata", "key": "independent_unit_id"}
            assert all(output["prediction_level"] == "sample" for output in capture["package"]["output_bindings"])
            group_reports = [report for report in capture["outcome"]["score_set"]["reports"] if report["level"] == "group"]
            assert group_reports and all(report["grouping_key"] == plan["campaign"]["aggregation_policy"]["grouping_key"] for report in group_reports)
        if mechanism == "pyo3":
            assert tasks and any(task["phase"] == "REFIT" for task in tasks)
            assert len(weighted_calls) == 2 * len(tasks)
            assert Counter(component for _phase, _node, component in weighted_calls) == {"encoder": len(tasks), "head": len(tasks)}
            for task in tasks:
                influence = task["fit_influence"]
                counts = Counter(influence["independent_unit_ids"])
                expected = [1. / counts[unit] for unit in influence["independent_unit_ids"]]
                np.testing.assert_allclose(influence["row_weights"], expected, rtol=0, atol=0)
                assert len(influence["fit_sample_ids"]) == len(expected)
        # Archive replay and numerical reference consume genuine fitted state;
        # weights in the independent reference are counted only by this test.
        archive = result.export(tmp_path / "unit-models.n4a")
        loader = BundleLoader(archive)
        matrix = np.concatenate([source.values[units >= 12] for source in sources.values()], axis=1)
        replay = loader.predict_outputs(matrix)
        from sklearn.base import clone

        from nirs4all.pipeline.dagml.node_runner import _FrozenTransform

        train = units < 12
        counts = Counter(units[train])
        weights = np.asarray([1. / counts[unit] for unit in units[train]])
        artifacts = {item["artifact_id"]: item for item in result._dagml_refit_artifacts}
        cold_expected = {}
        for output in loader._named_output_model().native_outputs:
            estimator = artifacts[output["artifact_id"]]["estimator"]
            assert isinstance(estimator, Pipeline)
            encoders = [encoder for _name, step in estimator.steps[:-1]
                        for encoder in (step.transformer.steps if isinstance(step, _FrozenTransform) else [step])]
            assert len(encoders) == 1 and isinstance(encoders[0], (StandardScaler, MinMaxScaler))
            # MinMaxScaler has no sample_weight; the closed native admission
            # refuses it for weighted scopes. The supported choices are StandardScaler.
            assert isinstance(encoders[0], StandardScaler)
            head = Ridge(alpha=.4) if kind == "regression" else LogisticRegression(C=.7, max_iter=1000)
            encoder = clone(encoders[0]).fit(sources[f"sensor{output['source_index']}"].values[train], sample_weight=weights)
            training_y = y[train].astype(np.float32).astype(np.float64) if kind == "regression" else y[train]
            head.fit(encoder.transform(sources[f"sensor{output['source_index']}"].values[train]), training_y, sample_weight=weights)
            expected = head.predict(encoder.transform(sources[f"sensor{output['source_index']}"].values[~train]))
            # Public array prediction materializes float32 inputs. Match that
            # real transport in the independent cold oracle, retaining the
            # existing float64 direct-loader oracle and its strict tolerance.
            cold_x = sources[f"sensor{output['source_index']}"].values[~train].astype(np.float32)
            cold_expected[output["output_binding_id"]] = head.predict(encoder.transform(cold_x))
            actual = np.asarray(replay[output["output_binding_id"]]).ravel()
            if kind == "regression":
                np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
            else:
                np.testing.assert_array_equal(actual, expected)
    finally:
        result.close()

    # The mandatory installed gate also replays this weighted archive in a
    # fresh process after training state is closed, with every FIT path forbidden.
    python = os.environ.get("NIRS4ALL_BY_SOURCE_INSTALLED_PYTHON")
    if not python and os.environ.get("NIRS4ALL_REQUIRE_BY_SOURCE_INSTALLED") == "1":
        pytest.fail("MM03 weighted archive requires the fresh installed replay child")
    if python:
        data_path = tmp_path / "weighted-inputs.npy"
        np.save(data_path, matrix.astype(np.float32))
        program = '''import json,sys,numpy as np,nirs4all,dag_ml
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from nirs4all.pipeline.bundle.loader import BundleLoader
assert nirs4all.__file__.startswith(sys.prefix) and dag_ml.__file__.startswith(sys.prefix)
def forbidden(*a,**k): raise AssertionError("FIT or search during weighted cold replay")
Pipeline.fit=StandardScaler.fit=LogisticRegression.fit=Ridge.fit=dag_ml.execute_training=forbidden
loader=BundleLoader(sys.argv[1])
values={key:nirs4all.predict(sys.argv[1],np.load(sys.argv[2]),output=key).y_pred for key in loader.named_outputs}
print(json.dumps({key:np.asarray(value).reshape(-1).tolist() for key,value in values.items()}))
'''
        terminal = subprocess.run([python, "-I", "-B", "-c", program, str(archive), str(data_path)],
                                  capture_output=True, text=True, check=False)
        assert terminal.returncode == 0, terminal.stdout + terminal.stderr
        cold_actual = json.loads(terminal.stdout.splitlines()[-1])
        assert set(cold_actual) == set(cold_expected)
        for key, expected in cold_expected.items():
            if kind == "regression":
                np.testing.assert_allclose(cold_actual[key], np.asarray(expected).ravel(), atol=1e-10, rtol=1e-10)
            else:
                np.testing.assert_array_equal(cold_actual[key], np.asarray(expected).ravel())

    # Keep the original generator combination as a public refusal witness.
    # Both component estimators support weights, but separate native fitting
    # nodes cannot consume the model-owned MM03 influence contract.
    unsupported = _pipeline(cohort, "top_k")
    for body in unsupported[1]["branch"]["steps"].values():
        body[0]["_or_"] = [StandardScaler(), StandardScaler(with_std=False)]
    before_calls = len(weighted_calls)
    before_tasks = len(tasks)
    with pytest.raises(dag_ml.DagMlRuntimeError, match="weighted preprocessing inside the model owner; separate fitting nodes"):
        nirs4all.run(unsupported, cohort, engine="dag-ml", random_state=43,
                     refit={"top_k": 2}, save_artifacts=True, save_charts=False,
                     verbose=0, workspace_path=tmp_path / "unsupported-workspace")
    assert len(weighted_calls) == before_calls and len(tasks) == before_tasks


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "controller", "kind", "producer"])
def test_original_package_sidecar_bijection_rejects_tampering(tmp_path, monkeypatch, mutation):
    """Use a real native package; changing its host attachment never retrains it."""
    from copy import deepcopy

    from nirs4all.pipeline.dagml.attested_by_source import validate_source_package_bindings

    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    dataset = _data("regression")
    result = nirs4all.run(_pipeline(dataset, "static"), dataset, engine="dag-ml", random_state=43,
                         save_artifacts=True, save_charts=False, verbose=0, workspace_path=tmp_path / "workspace")
    try:
        packages = [item["package"] for item in result._dagml_source_training_captures]
        descriptors = [{key: value for key, value in artifact.items() if key not in {"estimator", "y_transform"}}
                       for artifact in result._dagml_refit_artifacts]
        assert validate_source_package_bindings(packages, descriptors)
        changed = deepcopy(descriptors)
        if mutation == "missing":
            changed.pop()
        elif mutation == "extra":
            changed.append({**changed[0], "artifact_id": "artifact:undeclared"})
        elif mutation == "duplicate":
            changed.append(deepcopy(changed[0]))
        else:
            changed[0][{"controller": "controller_id", "kind": "kind", "producer": "producer_node"}[mutation]] = "tampered"
        with pytest.raises(ValueError, match="biject|producer/controller"):
            validate_source_package_bindings(packages, changed)
    finally:
        result.close()


def test_archive_carrier_corruption_is_refused_before_unpickling(tmp_path, monkeypatch):
    """An original native package cannot authorize changed executable carrier bytes."""
    from nirs4all.pipeline.dagml.attested_by_source import validate_source_archive_before_model

    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    dataset = _data("string")
    result = nirs4all.run(_pipeline(dataset, "static"), dataset, engine="dag-ml", random_state=43,
                         save_artifacts=True, save_charts=False, verbose=0, workspace_path=tmp_path / "workspace")
    try:
        archive = result.export(tmp_path / "original.n4a")
        bad = tmp_path / "corrupt.n4a"
        corrupted = 0
        with zipfile.ZipFile(archive) as original, zipfile.ZipFile(bad, "w") as changed:
            for name in original.namelist():
                content = original.read(name)
                if name.startswith("dagml_source_sidecar_"):
                    content += b"corrupt"
                    corrupted += 1
                changed.writestr(name, content)
        assert corrupted == 2
        import joblib

        monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("unpickled carrier before hash validation"))
        with zipfile.ZipFile(bad) as changed:
            manifest = json.loads(changed.read("manifest.json"))
            with pytest.raises(ValueError, match="size|hash"):
                validate_source_archive_before_model(changed, manifest)
        with pytest.raises(ValueError, match="size|hash"):
            BundleLoader(bad)
    finally:
        result.close()
