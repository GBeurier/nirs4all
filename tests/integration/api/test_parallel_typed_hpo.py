"""Native typed worker windows, nested scientific oracle and durable resume."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

import nirs4all
from nirs4all.pipeline.dagml.cancellation import DagRunCancelled
from nirs4all.pipeline.dagml.structural_tuning import _prepare_structure
from tests.integration.api import test_methods_multimodal_u07 as fixed
from tests.integration.api import test_structural_hpo_early_late as late
from tests.integration.api import test_structural_hpo_preprocessing_chains as chains
from tests.integration.api import test_structural_hpo_typed_modalities as typed


@pytest.fixture(autouse=True)
def native_execution_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("legacy scheduler executed"))
    monkeypatch.setattr("nirs4all.pipeline.dagml.run_paths._run_model_on_precomputed_matrix", lambda *a, **k: pytest.fail("host CV scheduler executed"))


def _run(example: Any, directory: Path, tuning: dict[str, Any], *, options: dict[str, Any] | None = None) -> Any:
    resources = {"cpu_threads": 1, "gpu_devices": [], **(options or {})}
    return nirs4all.run(example.make_pipeline(), example.make_dataset(), tuning=tuning, engine="dag-ml", workspace_path=directory, refit=True, random_state=17, save_charts=False, verbose=0, **resources)


def _audit(result: Any, expected_indices: list[int]) -> None:
    audits = result.structural_tuning_candidate_audit
    assert [record["trial_index"] for record in audits] == expected_indices
    terminal = result.structural_tuning_evidence["checkpoint"]["trials"]
    for record in audits:
        trial = terminal[record["trial_index"]]
        params = trial.get("evidence", trial)["params"]
        assert record["recipe_id"] == params["__recipe__"] and record["closed"] is True
        assert all(owner["closed"] is True and owner["events"] for owner in record["owners"])
        for owner in record["owners"]:
            assert any(event["operation"] == "dispose" for event in owner["events"])
    signed = result.structural_tuning_search_request["request"]["optimizer_descriptor"]["parallel_execution"]
    assert signed["cpu_threads"] == 1 and signed["gpu_devices"] == []
    assert signed["methods_build"] == {"schema_version": 1, "blas": False, "openmp": False, "cuda": False}


def _expanded_overlap_fixture() -> Any:
    """Enlarge the existing test fixture so real native calls have measurable duration."""
    from nirs4all_io import MultimodalDataset, TensorSource

    base = late.example.make_dataset()
    repeats = 64
    ids = [f"overlap.{repeat}.{sample}" for repeat in range(repeats) for sample in base.sample_ids]
    groups = [f"replica.{repeat}.{group}" for repeat in range(repeats) for group in base.groups]
    sources = {}
    for name, source in base.sources.items():
        values = np.tile(source.values, (repeats,) + (1,) * (source.values.ndim - 1))
        coordinates = dict(source.axis_coordinates or {})
        if name == "nir":
            values = np.tile(values, (1, 6))[:, :128]
            values += 0.001 * np.sin(np.arange(len(ids))[:, None] + np.arange(128)[None, :] / 11)
            coordinates["wavelength"] = np.linspace(900, 1700, 128)
        values.flags.writeable = False
        sources[name] = TensorSource(
            values, ids, representation_id=source.representation_id, axes=source.axes,
            feature_names=source.feature_names, axis_units=source.axis_units,
            axis_coordinates=coordinates,
        )
    target = np.tile(base.y, repeats)
    target.flags.writeable = False
    return MultimodalDataset(
        sources, sample_ids=ids, y=target, groups=groups,
        partitions=list(base.partitions) * repeats, target_names=base.target_names,
        name="native_overlap_test_fixture",
    )


@pytest.mark.parametrize("family", ["typed", "topology"])
def test_actual_native_hpo_fit_overlap_has_serialized_negative_control(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, family: str, record_property: Any,
) -> None:
    helper = os.environ.get("N4M_NATIVE_FIT_WITNESS_HELPER")
    library = os.environ.get("N4M_NATIVE_FIT_WITNESS_LIBRARY")
    if not helper or not library:
        if os.environ.get("NIRS4ALL_REQUIRE_NATIVE_FIT_WITNESS") == "1":
            pytest.fail("mandatory native HPO overlap proof requires the real Methods test probe and helper")
        pytest.skip("native test probe is opt-in; release qualification requires this witness")
    spec = importlib.util.spec_from_file_location("typed_hpo_native_fit_witness", helper)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    example = typed.example if family == "typed" else late.example
    cohort, pipeline = _expanded_overlap_fixture(), example.make_pipeline()
    base = {**example.make_tuning(tmp_path / "base-study"), "n_jobs": 2, "n_trials": 2}
    prepared = _prepare_structure(pipeline, cohort, base, {})
    entry = next(
        item for item in prepared["catalogue"]["entries"]
        if (family == "typed" and "nir" in next(
            node["operator"]["recipe"]["source_order"] for node in item["graph"]["nodes"] if node["kind"] == "model"
        )) or (family == "topology" and {"late.nir.alpha", "late.image.alpha", "late.meta.alpha"}.issubset(item["parameter_bindings"]))
    )
    before = {name: hashlib.sha256(
        json.dumps(source.values.tolist(), ensure_ascii=False).encode()
        if source.values.dtype.kind in "OUS" else source.values.tobytes()
    ).hexdigest() for name, source in cohort.sources.items()}
    trials, snapshots = [], []
    for serialized in (False, True):
        directory = tmp_path / ("serialized" if serialized else "concurrent")
        with monkeypatch.context() as patch:
            chains._enqueue_recipes(patch, prepared["catalogue"], [entry, entry])
            with module.witness(patch, serialized=serialized, capacity=65536) as probe:
                with nirs4all.run(
                    pipeline, cohort, tuning={**base, "storage": (directory / "study").resolve().as_uri()},
                    engine="dag-ml", workspace_path=directory / "workspace", refit=True,
                    cpu_threads=1, gpu_devices=[], random_state=17, save_charts=False, verbose=0,
                ) as result:
                    _audit(result, [0, 1])
                    trials.append(copy.deepcopy(result.structural_tuning_evidence["trials"]))
                snapshot = probe.snapshot()
                assert snapshot["active"] == snapshot["inflight"] == 0
                assert snapshot["overflow"] is False and snapshot["total_calls"] > 0
                assert len(snapshot["records"]) == snapshot["total_calls"]
                assert all(record["entered_ns"] < record["exited_ns"] for record in snapshot["records"])
                snapshots.append(snapshot)
    assert snapshots[0]["peak_active"] >= 2, "No overlapping in-flight native FIT calls observed through actual HPO windows"
    assert len({record["native_thread"] for record in snapshots[0]["records"]}) >= 2
    assert snapshots[1]["peak_active"] == 1, "Native probe did not detect serialized negative control"
    assert snapshots[0]["total_calls"] == snapshots[1]["total_calls"]
    assert trials[0] == trials[1]
    assert before == {name: hashlib.sha256(
        json.dumps(source.values.tolist(), ensure_ascii=False).encode()
        if source.values.dtype.kind in "OUS" else source.values.tobytes()
    ).hexdigest() for name, source in cohort.sources.items()}
    record_property("real_hpo_native_fit_witness_concurrent_and_serialized", json.dumps(snapshots))


@pytest.mark.parametrize("family,workers", [("typed", 2), ("topology", 2), ("topology", 3), ("topology", 4)])
def test_all_declared_choices_match_serial_and_independent_grouped_oracle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, family: str, workers: int) -> None:
    example = typed.example if family == "typed" else late.example
    cohort, pipeline = example.make_dataset(), example.make_pipeline()
    base = example.make_tuning(tmp_path / "serial-study")
    prepared = _prepare_structure(pipeline, cohort, base, {})
    entries = prepared["catalogue"]["entries"]
    base = {**base, "n_trials": len(entries)}
    input_hashes = {name: hashlib.sha256(source.values.tobytes()).hexdigest() for name, source in cohort.sources.items()}
    with ExitStack() as runs:
        # Each optimizer must receive its own complete queue. Restore ask
        # between runs so a second wrapper cannot inherit an exhausted one.
        with monkeypatch.context() as serial_patch:
            chains._enqueue_recipes(serial_patch, prepared["catalogue"], entries)
            serial = runs.enter_context(_run(example, tmp_path / "serial", base))
        with monkeypatch.context() as parallel_patch:
            chains._enqueue_recipes(parallel_patch, prepared["catalogue"], entries)
            parallel = runs.enter_context(
                _run(
                    example,
                    tmp_path / "parallel",
                    {
                        **base,
                        "storage": (tmp_path / "parallel-study").resolve().as_uri(),
                        "n_jobs": workers,
                    },
                )
            )
        assert "parallel_execution" not in serial.structural_tuning_search_request["request"]["optimizer_descriptor"]
        assert not hasattr(serial, "structural_tuning_candidate_audit")
        declared_recipes = [entry["recipe_id"] for entry in entries]
        for candidate in (serial, parallel):
            assert [trial["params"]["__recipe__"] for trial in candidate.structural_tuning_evidence["trials"]] == declared_recipes
        assert parallel.structural_tuning_evidence["trials"] == serial.structural_tuning_evidence["trials"]
        assert parallel.tuning_best_params == serial.tuning_best_params
        assert parallel.structural_tuning_evidence["selected_trial_index"] == serial.structural_tuning_evidence["selected_trial_index"]
        _audit(parallel, list(range(len(entries))))
        train = np.flatnonzero(np.asarray(cohort.partitions) == "train")
        folds = list(GroupKFold(3).split(np.zeros((len(train), 1)), groups=np.asarray(cohort.groups)[train]))
        for trial in parallel.structural_tuning_evidence["trials"]:
            entry = next(item for item in entries if item["recipe_id"] == trial["params"]["__recipe__"])
            if family == "topology":
                sequence = late._sequence_for_entry(entry, pipeline)
            else:
                recipe = next(node["operator"]["recipe"] for node in entry["graph"]["nodes"] if node["kind"] == "model")
                declaration = next(model for model in pipeline[1]["model"]["_or_"] if fixed.recipe_from_estimator(model, allow_source_selection=True) == recipe)
            scores = {}
            for fold_index, (fit_rows, validation_rows) in enumerate(folds):
                fit, validation = train[fit_rows], train[validation_rows]
                if family == "topology":
                    prediction = late._oracle(sequence, cohort, fit, cohort, validation, trial["params"])
                else:
                    declaration = copy.deepcopy(declaration)
                    declaration.set_params(model__alpha=trial["params"]["model.alpha"])
                    prediction, _fitted = fixed._oracle(declaration, cohort, fit, validation)
                score = float(np.sqrt(mean_squared_error(np.asarray(cohort.y)[validation], prediction)))
                scores[f"fold{fold_index}"] = score
            assert trial["objective_fold_scores"] == pytest.approx(scores, rel=2e-7, abs=2e-7)
            assert trial["score"] == pytest.approx(float(np.mean(list(scores.values()))), rel=2e-7, abs=2e-7)
        new = example.make_dataset(29, prediction=True)
        left = serial.export(tmp_path / "serial.n4a")
        right = parallel.export(tmp_path / "parallel.n4a")
        np.testing.assert_array_equal(nirs4all.predict(left, new, engine="dag-ml").y_pred, nirs4all.predict(right, new, engine="dag-ml").y_pred)
    assert {name: hashlib.sha256(source.values.tobytes()).hexdigest() for name, source in cohort.sources.items()} == input_hashes


@pytest.mark.parametrize("workers,budget", [(2, 1), (3, 2), (4, 2), (4, 5)])
def test_short_and_partial_windows_return_only_budgeted_candidate_audits(tmp_path: Path, workers: int, budget: int) -> None:
    tuning = {**late.example.make_tuning(tmp_path / "study"), "n_jobs": workers, "n_trials": budget}
    with _run(late.example, tmp_path / "workspace", tuning) as result:
        _audit(result, list(range(budget)))
        assert len(result.tuning_result.trials) == budget


def test_cancellation_joins_admitted_siblings_and_resume_retains_terminal_history(tmp_path: Path) -> None:
    tuning = {**late.example.make_tuning(tmp_path / "study"), "n_jobs": 2, "n_trials": 5}
    with pytest.raises(DagRunCancelled):
        _run(late.example, tmp_path / "stopped", {**tuning, "progress_callback": lambda event: len(event["checkpoint"]["trials"]) < 1})
    path = tmp_path / "study/structural-early-late.n4mopt.json"
    initial = json.loads(path.read_text())["native_checkpoint"]["trials"]
    assert len(initial) == 2 and all(record["state"] == "complete" for record in initial)
    with (
        _run(late.example, tmp_path / "resumed", {**tuning, "resume": True}) as resumed,
        _run(
            late.example,
            tmp_path / "continuous",
            {
                **tuning,
                "storage": (tmp_path / "continuous-study").resolve().as_uri(),
            },
        ) as continuous,
    ):
        assert json.loads(path.read_text())["native_checkpoint"]["trials"][:2] == initial
        assert resumed.structural_tuning_evidence["trials"] == continuous.structural_tuning_evidence["trials"]
        _audit(resumed, [2, 3, 4])


@pytest.mark.parametrize("mutation", ["workers", "threads", "cuda_build"])
def test_resume_resource_or_profile_change_refused_before_model_callbacks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str) -> None:
    from dag_ml.multimodal_methods import MethodsMultimodalController
    from dag_ml.multimodal_topology import MethodsTopologyController
    from n4m import BuildCapabilities

    from nirs4all.pipeline.dagml import typed_parallel

    tuning = {**late.example.make_tuning(tmp_path / "study"), "n_jobs": 2, "n_trials": 4}
    with pytest.raises(DagRunCancelled):
        _run(late.example, tmp_path / "first", {**tuning, "progress_callback": lambda event: len(event["checkpoint"]["trials"]) < 1})
    paths = [path for path in (tmp_path / "study").rglob("*") if path.is_file()]
    before = {path: path.read_bytes() for path in paths}
    for cls in (MethodsMultimodalController, MethodsTopologyController):
        monkeypatch.setattr(cls, "operator", lambda *a, **k: pytest.fail("profile mismatch reached model callback"))
    options: dict[str, Any] = {}
    if mutation == "workers":
        tuning["n_jobs"] = 3
    elif mutation == "threads":
        options["cpu_threads"] = 2
    else:
        monkeypatch.setattr(typed_parallel.importlib.import_module("n4m"), "build_capabilities", lambda: BuildCapabilities(schema_version=1, blas=False, openmp=False, cuda=True))
    with pytest.raises(Exception, match="(?i)(binding|checkpoint|contract|parallel typed|sequential CPU)"):
        _run(late.example, tmp_path / "resume", {**tuning, "resume": True}, options=options)
    assert {path: path.read_bytes() for path in paths} == before


def test_failed_worker_retains_successful_sibling_and_closes_every_owner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dag_ml.multimodal_methods import MethodsMultimodalController

    controllers = late._observe(monkeypatch)
    original = MethodsMultimodalController.operator
    injected = False

    def fail_after_real_fit(controller: Any, task: dict[str, Any]) -> Any:
        nonlocal injected
        result = original(controller, task)
        trial = ((task.get("variant") or {}).get("choices", {}).get("host_hpo", {}).get("value") or {}).get("trial_index")
        if trial == 0 and task["phase"] == "FIT_CV" and not injected:
            injected = True
            raise RuntimeError("injected typed candidate failure after actual FIT")
        return result

    monkeypatch.setattr(MethodsMultimodalController, "operator", fail_after_real_fit)
    tuning = {**late.example.make_tuning(tmp_path / "study"), "n_jobs": 2, "n_trials": 4}
    with pytest.raises(Exception, match="injected typed candidate failure"):
        _run(late.example, tmp_path / "failed", tuning)
    path = tmp_path / "study/structural-early-late.n4mopt.json"
    saved = json.loads(path.read_text())["native_checkpoint"]["trials"]
    assert [record["state"] for record in saved] == ["failed", "complete"]
    assert injected and controllers and all(controller.closed for controller in controllers)
    with _run(late.example, tmp_path / "resume", {**tuning, "resume": True}) as resumed:
        _audit(resumed, [2, 3])
        assert json.loads(path.read_text())["native_checkpoint"]["trials"][:2] == saved
    assert all(controller.closed for controller in controllers)


@pytest.mark.parametrize("point", ["before", "after"])
def test_real_process_death_preserves_atomic_pair_and_pending_proposal_identity(tmp_path: Path, point: str) -> None:
    directory = tmp_path / "interrupted"
    directory.mkdir()
    script = textwrap.dedent("""
        import json, os, sys
        from pathlib import Path
        from tests.integration.api import test_parallel_typed_hpo as test
        root, point = Path(sys.argv[1]), sys.argv[2]
        replace = os.replace
        def terminate_at_replace(source, destination):
            payload = json.loads(Path(source).read_text()) if str(destination).endswith('.n4mopt.json') else {}
            if payload.get('native_checkpoint', {}).get('trials'):
                if point == 'before': os._exit(73)
                replace(source, destination)
                os._exit(73)
            replace(source, destination)
        os.replace = terminate_at_replace
        tuning = {**test.late.example.make_tuning(root/'study'), 'n_jobs':2, 'n_trials':4}
        test._run(test.late.example, root/'workspace', tuning)
    """)
    child = subprocess.run([sys.executable, "-c", script, str(directory), point], capture_output=True, text=True, check=False, timeout=120)
    assert child.returncode == 73, child.stdout + child.stderr
    path = directory / "study/structural-early-late.n4mopt.json"
    pair = json.loads(path.read_text())
    frontier = len(pair["native_checkpoint"]["trials"])
    assert frontier == (0 if point == "before" else 1)
    tuning = {**late.example.make_tuning(directory / "study"), "n_jobs": 2, "n_trials": 4, "resume": True}
    with (
        _run(late.example, directory / "resumed", tuning) as resumed,
        _run(
            late.example,
            tmp_path / "continuous",
            {
                **tuning,
                "resume": False,
                "storage": (tmp_path / "continuous-study").resolve().as_uri(),
            },
        ) as continuous,
    ):
        assert resumed.structural_tuning_evidence["trials"] == continuous.structural_tuning_evidence["trials"]
        assert json.loads(path.read_text())["native_checkpoint"]["trials"][:frontier] == pair["native_checkpoint"]["trials"]
        _audit(resumed, list(range(frontier, 4)))


def test_fresh_installed_parallel_late_winner_reuses_mixed_portable_closure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original = late.example.make_tuning
    monkeypatch.setattr(late.example, "make_tuning", lambda *a, **k: {**original(*a, **k), "n_jobs": 2})
    late.test_fresh_installed_late_winner_replays_exact_mixed_closure_without_fit_or_hpo(monkeypatch, tmp_path)
