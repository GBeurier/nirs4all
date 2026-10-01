"""Public global/local composition uses native, training-only source scopes."""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nirs4all_io import MultimodalDataset
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, StratifiedKFold

import nirs4all
from nirs4all.operators.models.multimodal import MultimodalClassifier, TensorPCA
from nirs4all.pipeline.dagml import host_finetune
from nirs4all.pipeline.dagml.multimodal_tuning import MultimodalTuningStopped
from tests.integration.api.test_multimodal_dagml import _cohort, _model
from tests.integration.api.test_multimodal_late_fusion import _cohort as _late_cohort
from tests.integration.api.test_multimodal_late_fusion import _pipeline as _late_pipeline
from tests.integration.api.test_multimodal_targets import _classification_cohort, _classifier
from tests.integration.api.test_multimodal_tuning import _stop_after


def _recipe(late: bool, engine: str = "n4m") -> tuple[list[Any], Any, dict[str, Any]]:
    local = {"engine": engine, "sampler": "random", "seed": 23, "n_trials": 2,
             "approach": "grouped", "eval_mode": "mean", "model_params": {"alpha" if late else "model__alpha": [0.1, 1.0]}}
    if not late:
        return [GroupKFold(3), {"model": _model(), "finetune_params": local,
                               "train_params": {"model__tol": 0.003}, "refit_params": {"model__alpha": 9.0}}], _cohort(), {
            "transformers__image__n_components": [1, 2],
        }
    pipeline = _late_pipeline()
    branch = pipeline[1]["branch"]["steps"]["image"]
    branch[-1] = {"model": branch[-1], "finetune_params": local,
                  "train_params": {"tol": 0.003}, "refit_params": {"alpha": 9.0}}
    return pipeline, _late_cohort(), {"branches.image.0.n_components": [1, 2]}


def _run(pipeline: list[Any], cohort: Any, space: dict[str, Any], workspace: Path, **tuning: Any) -> Any:
    return nirs4all.run(
        pipeline, cohort, tuning={"engine": "n4m", "sampler": "random", "seed": 19,
                                 "n_trials": 2, "space": space, **tuning},
        engine="dag-ml", workspace_path=workspace, save_charts=False,
        verbose=0, random_state=19, refit=True, save_artifacts=True,
    )


@pytest.fixture(autouse=True)
def native_scheduler_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("legacy scheduler executed"))
    monkeypatch.setattr("nirs4all.pipeline.dagml.run_paths._run_model_on_precomputed_matrix", lambda *a, **k: pytest.fail("Python CV loop executed"))


def _permuted_sources(cohort: Any) -> MultimodalDataset:
    rng = np.random.default_rng(607)
    sources = {}
    for name, source in cohort.sources.items():
        order = rng.permutation(len(source.sample_ids))
        sources[name] = replace(source, values=np.asarray(source.values)[order],
                                sample_ids=tuple(source.sample_ids[row] for row in order),
                                presence_mask=np.asarray(source.presence_mask)[order])
    return MultimodalDataset(sources, sample_ids=cohort.sample_ids, y=cohort.y,
                             groups=cohort.groups, partitions=cohort.partitions,
                             task_type=cohort.task_type, name=cohort.name)


@pytest.mark.parametrize("local_engine", ["optuna", "n4m"])
def test_public_nested_classifier_uses_native_accuracy_and_replays_original_labels_without_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_engine: str,
) -> None:
    original = _classification_cohort()
    # These original labels vary within a plant. This case uses an ordinary
    # stratified cohort, rather than aggregating mixed labels per plant; grouped
    # inner-fold ownership remains exercised by the regression cases below.
    cohort = MultimodalDataset(original.sources, sample_ids=original.sample_ids,
                               y=original.y, partitions=original.partitions,
                               task_type="classification", name=original.name)
    local = {"engine": local_engine, "sampler": "random", "seed": 23, "n_trials": 2,
             "approach": "grouped", "eval_mode": "mean", "model_params": {"model__C": [0.1, 1.0]}}
    pipeline = [StratifiedKFold(2, shuffle=True, random_state=31),
                {"model": _classifier(), "finetune_params": local,
                 "train_params": {"model__tol": 0.003}, "refit_params": {"model__C": 3.0}}]
    searches: list[dict[str, Any]] = []
    search = host_finetune.run_scoped_finetune

    def capture(*args: Any, **kwargs: Any) -> Any:
        dataset = kwargs["scoped_dataset"]
        assert tuple(dataset.source_names) == ("nir", "image", "series", "metadata")
        assert set(dataset.sample_ids) <= set(cohort.sample_ids[:12])
        answer = search(*args, **kwargs)
        searches.append(answer[1])
        return answer

    monkeypatch.setattr(host_finetune, "run_scoped_finetune", capture)
    result = _run(pipeline, cohort, {"transformers__image__n_components": [1, 2]}, tmp_path / "workspace")
    try:
        assert [trial.state for trial in result.tuning_result.trials] == ["COMPLETE"] * 2
        assert result.tuning_result.tuning.metric == "balanced_accuracy"
        assert result.tuning_result.tuning.direction == "maximize"
        assert result.tuning_best_value == max(trial.value for trial in result.tuning_result.trials)
        assert searches
        assert {evidence["scope"]["phase"] for evidence in searches} >= {"FIT_CV", "REFIT"}
        for evidence in searches:
            assert evidence["optimizer"]["name"] == local_engine
            assert evidence["evaluation"]["outer_validation_used"] is False
            assert evidence["evaluation"]["test_used"] is False
            assert evidence["evaluation"]["inner_fold_count"] == 2
            assert len(evidence["trials"]) == 2
            assert all(0 <= trial["score"] <= 1 for trial in evidence["trials"])
            if evidence["scope"]["phase"] == "REFIT":
                assert evidence["effective_selected_model_params"]["model__C"] == 3.0
        prediction = _cohort(prediction=True)
        archive = result.export(tmp_path / "nested-classifier.n4a")
        expected = nirs4all.predict(archive, prediction, engine="dag-ml").y_pred
        assert set(expected) <= set(cohort.y)
    finally:
        result.close()
    shutil.rmtree(tmp_path / "workspace")
    monkeypatch.setattr(MultimodalClassifier, "fit", lambda *a, **k: pytest.fail("classifier archive prediction fit a model"))
    monkeypatch.setattr(LogisticRegression, "fit", lambda *a, **k: pytest.fail("classifier archive prediction fit LogisticRegression"))
    monkeypatch.setattr(TensorPCA, "fit", lambda *a, **k: pytest.fail("classifier archive prediction fit an encoder"))
    monkeypatch.setattr(host_finetune, "run_scoped_finetune", lambda *a, **k: pytest.fail("classifier archive prediction opened local HPO"))
    replay = nirs4all.predict(archive, prediction, engine="dag-ml")
    np.testing.assert_array_equal(replay.y_pred, expected)
    assert replay.metadata["training_performed"] is False


@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("local_engine", ["optuna", "n4m"])
def test_public_nested_search_fits_fresh_inner_source_recipes_and_replays_without_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, late: bool, local_engine: str,
) -> None:
    import dag_ml

    pipeline, cohort, space = _recipe(late, local_engine)
    supplied = copy.deepcopy(pipeline)
    calls: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    original_search = host_finetune.run_scoped_finetune
    original_fit = TensorPCA.fit
    original_native = dag_ml.run_host_hpo_search_in_process
    global_requests: list[dict[str, Any]] = []

    def record_native(*args: Any, **kwargs: Any) -> Any:
        request = args[3]
        if "nested_local_hpo" in request["optimizer_descriptor"]:
            global_requests.append(copy.deepcopy(request))
        return original_native(*args, **kwargs)

    def record_search(*args: Any, **kwargs: Any) -> Any:
        dataset = kwargs["scoped_dataset"]
        assert dataset is not None
        ids = tuple(dataset.sample_ids)
        names = tuple(dataset.source_names)
        assert names == (("image",) if late else ("nir", "image", "series", "metadata"))
        assert kwargs["source_index"] == (0 if late else None)
        assert set(ids) <= {sample for sample, partition in zip(cohort.sample_ids, cohort.partitions, strict=True) if partition == "train"}
        scope = copy.deepcopy(kwargs["scope"])
        assert len(scope["training_sample_ids"]) == len(ids)
        assert tuple(scope["source_names"]) == names
        assert len(scope["recipe_fingerprint"]) == len(scope["data_content_fingerprint"]) == 64
        # This is an assertion oracle, not execution: native DAG owns actual
        # candidate folds, callback order, scores and selection.
        expected = {
            frozenset(int(ids[row].rsplit("-", 1)[1]) for row in train)
            for train, _ in GroupKFold(3).split(np.zeros((len(ids), 1)), groups=dataset.cohort.groups)
        }
        capture = {"scope": scope, "ids": ids, "expected": expected, "fits": []}
        calls.append(capture)
        active.append(capture)
        try:
            best, evidence = original_search(*args, **kwargs)
            capture["evidence"] = evidence
            return best, evidence
        finally:
            active.pop()

    def record_fit(self: TensorPCA, X: Any, y: Any = None) -> Any:
        values = np.asarray(X)
        if active:
            rows = frozenset(int(value) for value in values.reshape(len(values), -1)[:, 0])
            assert rows in active[-1]["expected"], "encoder reused outer-fit statistics or received inner-validation rows"
            assert len(rows) < len(active[-1]["ids"])
            active[-1]["fits"].append((values.ndim, rows, self.n_components))
        return original_fit(self, X, y)

    monkeypatch.setattr(host_finetune, "run_scoped_finetune", record_search)
    monkeypatch.setattr(TensorPCA, "fit", record_fit)
    monkeypatch.setattr(dag_ml, "run_host_hpo_search_in_process", record_native)
    result = _run(pipeline, cohort, space, tmp_path / "workspace")
    try:
        assert [trial.state for trial in result.tuning_result.trials] == ["COMPLETE"] * 2
        assert calls
        global_calls = [call for call in calls if call["scope"]["variant_id"].startswith("host_hpo:trial:")]
        assert {call["scope"]["variant_id"] for call in global_calls} == {"host_hpo:trial:0000000000", "host_hpo:trial:0000000001"}
        assert {call["scope"]["phase"] for call in calls} >= {"FIT_CV", "REFIT"}
        for call in calls:
            assert call["fits"]
            assert {rows for _, rows, _ in call["fits"]} == call["expected"]
            assert {rank for rank, _, _ in call["fits"]} == ({4} if late else {3, 4})
            assert call["evidence"]["evaluation"]["outer_validation_used"] is False
            assert call["evidence"]["evaluation"]["test_used"] is False
            assert len(call["evidence"]["trials"]) == 2
        target_step = pipeline[1]["branch"]["steps"]["image"][-1] if late else pipeline[-1]
        original_step = supplied[1]["branch"]["steps"]["image"][-1] if late else supplied[-1]
        assert target_step["finetune_params"] == original_step["finetune_params"]
        assert target_step["train_params"] == original_step["train_params"]
        assert not any("__dagml_inner_splitter" in step.get("finetune_params", {})
                       for step in ([target_step] if not late else pipeline[1]["branch"]["steps"]["image"]) if isinstance(step, dict))
        assert not hasattr(target_step["model"], "coef_")
        assert len(global_requests) == 1
        profile = global_requests[0]["optimizer_descriptor"]["nested_local_hpo"]
        assert profile["profile"] == "raw_source_recipe_inner_cv_v1"
        assert len(profile["graph_fingerprint"]) == 64
        prediction = _late_cohort(prediction=True) if late else _cohort(prediction=True)
        archive = result.export(tmp_path / "nested.n4a")
        expected_prediction = nirs4all.predict(archive, prediction, engine="dag-ml").y_pred
        refit_calls = [call for call in calls if call["scope"]["phase"] == "REFIT"]
        assert refit_calls
        assert all(call["evidence"]["effective_selected_model_params"]["alpha" if late else "model__alpha"] == 9.0 for call in refit_calls)
    finally:
        result.close()
    shutil.rmtree(tmp_path / "workspace")
    monkeypatch.setattr(Ridge, "fit", lambda *a, **k: pytest.fail("archive prediction fit a model"))
    monkeypatch.setattr(TensorPCA, "fit", lambda *a, **k: pytest.fail("archive prediction fit an encoder"))
    monkeypatch.setattr(host_finetune, "run_scoped_finetune", lambda *a, **k: pytest.fail("archive prediction opened local HPO"))
    np.testing.assert_array_equal(nirs4all.predict(archive, prediction, engine="dag-ml").y_pred, expected_prediction)


@pytest.mark.parametrize("late", [False, True])
def test_parallel_nested_candidates_match_serial_parameters_scores_and_predictions(tmp_path: Path, late: bool) -> None:
    pipeline, cohort, space = _recipe(late)
    serial = _run(pipeline, cohort, space, tmp_path / "serial")
    parallel = _run(pipeline, cohort, space, tmp_path / "parallel", n_jobs=2)
    try:
        assert [trial.params for trial in serial.tuning_result.trials] == [trial.params for trial in parallel.tuning_result.trials]
        assert [trial.value for trial in serial.tuning_result.trials] == [trial.value for trial in parallel.tuning_result.trials]
        assert serial.tuning_best_params == parallel.tuning_best_params
        prediction = _late_cohort(prediction=True) if late else _cohort(prediction=True)
        a = serial.export(tmp_path / "serial.n4a")
        b = parallel.export(tmp_path / "parallel.n4a")
        np.testing.assert_array_equal(nirs4all.predict(a, prediction, engine="dag-ml").y_pred,
                                      nirs4all.predict(b, prediction, engine="dag-ml").y_pred)
    finally:
        serial.close()
        parallel.close()


@pytest.mark.parametrize("late", [False, True])
def test_nested_search_aligns_independently_permuted_raw_source_rows(tmp_path: Path, late: bool) -> None:
    pipeline, cohort, space = _recipe(late)
    baseline = _run(pipeline, cohort, space, tmp_path / "baseline")
    permuted = _run(pipeline, _permuted_sources(cohort), space, tmp_path / "permuted")
    try:
        assert [trial.params for trial in baseline.tuning_result.trials] == [trial.params for trial in permuted.tuning_result.trials]
        assert [trial.value for trial in baseline.tuning_result.trials] == [trial.value for trial in permuted.tuning_result.trials]
        prediction = _late_cohort(prediction=True) if late else _cohort(prediction=True)
        a = baseline.export(tmp_path / "baseline.n4a")
        b = permuted.export(tmp_path / "permuted.n4a")
        np.testing.assert_array_equal(nirs4all.predict(a, prediction, engine="dag-ml").y_pred,
                                      nirs4all.predict(b, _permuted_sources(prediction), engine="dag-ml").y_pred)
    finally:
        baseline.close()
        permuted.close()


@pytest.mark.parametrize("late", [False, True])
def test_nested_resume_matches_continuous_and_keeps_completed_trial(tmp_path: Path, late: bool) -> None:
    pipeline, cohort, space = _recipe(late)
    study = tmp_path / "study"
    options = {"storage": study.as_uri(), "study_name": "nested"}
    with pytest.raises(MultimodalTuningStopped):
        _run(pipeline, cohort, space, tmp_path / "stopped", **options, progress_callback=_stop_after(1, []))
    before = json.loads((study / "nested.n4mopt.json").read_text())["native_checkpoint"]["trials"]
    resumed = _run(pipeline, cohort, space, tmp_path / "resumed", **options, resume=True)
    continuous = _run(pipeline, cohort, space, tmp_path / "continuous")
    try:
        assert json.loads((study / "nested.n4mopt.json").read_text())["native_checkpoint"]["trials"][:1] == before
        assert [trial.params for trial in resumed.tuning_result.trials] == [trial.params for trial in continuous.tuning_result.trials]
        assert [trial.value for trial in resumed.tuning_result.trials] == [trial.value for trial in continuous.tuning_result.trials]
    finally:
        resumed.close()
        continuous.close()


@pytest.mark.parametrize("mutation", ["local_seed", "local_budget", "ancestor_recipe"])
def test_nested_resume_changed_local_contract_refuses_without_fit_or_checkpoint_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    pipeline, cohort, space = _recipe(False)
    study = tmp_path / "study"
    options = {"storage": study.as_uri(), "study_name": "nested"}
    with pytest.raises(MultimodalTuningStopped):
        _run(pipeline, cohort, space, tmp_path / "stopped", **options, progress_callback=_stop_after(1, []))
    if mutation == "ancestor_recipe":
        pipeline[-1]["model"].set_params(transformers__image__random_state=99)
    else:
        pipeline[-1]["finetune_params"]["seed" if mutation == "local_seed" else "n_trials"] += 1
    before = {path.name: path.read_bytes() for path in study.iterdir() if path.is_file()}
    monkeypatch.setattr(Ridge, "fit", lambda *a, **k: pytest.fail("incompatible resume reached fit"))
    monkeypatch.setattr(TensorPCA, "fit", lambda *a, **k: pytest.fail("incompatible resume reached encoder fit"))
    with pytest.raises(Exception, match="host HPO checkpoint objective/graph/controller/data/fold binding mismatch"):
        _run(pipeline, cohort, space, tmp_path / "rejected", **options, resume=True)
    assert {path.name: path.read_bytes() for path in study.iterdir() if path.is_file()} == before


@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("kind", ["model", "train", "forced", "ancestor", "nested_dictionary"])
def test_parameter_owner_overlap_refuses_before_study_or_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, late: bool, kind: str,
) -> None:
    pipeline, cohort, _ = _recipe(late)
    step = pipeline[1]["branch"]["steps"]["image"][-1] if late else pipeline[-1]
    prefix = "branches.image.1." if late else ""
    key = "alpha" if late else "model__alpha"
    path = key.replace("__", ".")
    if kind == "train":
        step["finetune_params"]["train_params"] = {"tol" if late else "model__tol": [0.001, 0.003]}
        path = "tol" if late else "model.tol"
    elif kind == "forced":
        step["finetune_params"]["force_params"] = {"tol" if late else "model__tol": 0.001}
        path = "tol" if late else "model.tol"
    elif kind == "ancestor":
        path = "model" if not late else "alpha.child"
    elif kind == "nested_dictionary":
        if late:
            step["finetune_params"]["model_params"] = {"alpha": {"child": [0.1, 1.0]}}
            path = "alpha"
        else:
            step["finetune_params"]["model_params"] = {"model": {"alpha": [0.1, 1.0]}}
    monkeypatch.setattr("nirs4all.pipeline.dagml.multimodal_tuning.HostSearchOptimizer.__init__",
                        lambda *a, **k: pytest.fail("parameter overlap opened global optimizer"))
    monkeypatch.setattr(Ridge, "fit", lambda *a, **k: pytest.fail("parameter overlap reached fit"))
    monkeypatch.setattr(TensorPCA, "fit", lambda *a, **k: pytest.fail("parameter overlap reached encoder fit"))
    with pytest.raises(ValueError, match="overlap.*one search owner"):
        _run(pipeline, cohort, {prefix + path: [0.1, 1.0]}, tmp_path / "rejected")


@pytest.mark.parametrize("guard", ["meta_local", "deterministic_local", "generated", "fit_on_all", "augmentation", "join", "missing_policy"])
def test_unsupported_nested_profiles_refuse_before_optimizer_or_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, guard: str,
) -> None:
    late = guard in {"meta_local", "fit_on_all", "augmentation", "join", "missing_policy"}
    pipeline, cohort, space = _recipe(late)
    if guard == "meta_local":
        pipeline[-1] = {"model": pipeline[-1], "finetune_params": {"engine": "n4m", "n_trials": 2, "model_params": {"alpha": [0.1, 1.0]}}}
    elif guard == "deterministic_local":
        pipeline[-1]["finetune_params"]["engine"] = "dag-ml"
    elif guard == "generated":
        cohort._generated_view_store = object()
    elif guard == "missing_policy":
        pipeline[1]["branch"]["missing_source_policy"] = "zero_with_indicator"
    else:
        branch = pipeline[1]["branch"]["steps"]["image"]
        branch[0] = ({"preprocessing": branch[0], "fit_on_all": True} if guard == "fit_on_all" else
                     {"sample_augmentation": []} if guard == "augmentation" else {"merge": "features"})
    monkeypatch.setattr("nirs4all.pipeline.dagml.multimodal_tuning.HostSearchOptimizer.__init__",
                        lambda *a, **k: pytest.fail("unsupported nested profile opened global optimizer"))
    monkeypatch.setattr(Ridge, "fit", lambda *a, **k: pytest.fail("unsupported nested profile reached fit"))
    monkeypatch.setattr(TensorPCA, "fit", lambda *a, **k: pytest.fail("unsupported nested profile reached encoder fit"))
    with pytest.raises((ValueError, NotImplementedError), match="meta-model HPO|host search profile|generated|model steps|missing_source_policy|unsupported|whole-stack|multimodal tuning requires one multimodal model or by_source branches"):
        _run(pipeline, cohort, space, tmp_path / "rejected")
