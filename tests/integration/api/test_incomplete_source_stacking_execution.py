"""Real native grouped nested OOF, search, masked fits and cold public replay."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nirs4all_io import MultimodalDataset, TensorSource
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.pipeline.dagml import node_runner


def _cohort(*, classification: bool, source_count: int = 2, prediction: bool = False,
            hidden: float = np.nan, integer_labels: bool = False) -> MultimodalDataset:
    size = 8 if prediction else 108
    rows = np.arange(size)
    ids = [f"{'new' if prediction else 'train'}-{row}" for row in rows]
    rng = np.random.default_rng(301 if prediction else 300)
    latent = rng.normal(size=(size, 3))
    sources = {}
    for index in range(source_count):
        values = latent + index / 7 + rng.normal(scale=0.05, size=latent.shape)
        present = (rows + index) % (11 + index * 2) != 0
        if prediction:
            present[:] = False if index == 0 else True
            present[0] = False  # all sources absent: genuine meta prediction remains admitted
        values[~present] = hidden
        sources[f"source{index}"] = TensorSource(values, ids, representation_id="signal_1d", presence_mask=present)
    mask = np.ones(size if classification else (size, 2), dtype=bool)
    if classification:
        vocabulary = np.array([-(2 ** 62), 2 ** 53 + 17, 2 ** 62 - 1], dtype=np.int64) if integer_labels else np.array(["class🧬", "classé", "class水"])
        y = vocabulary[rows % 3]
        names = ["class"]
    else:
        y = np.column_stack([latent @ [1.5, 0.4, -0.7], latent @ [-0.2, 2, 0.9]])
        mask = np.column_stack([rows % 7 != 0, rows % 5 != 0])
        y[~mask] = 1e15 if np.isnan(hidden) else hidden
        names = ["sugar", "protein"]
    return MultimodalDataset(
        sources, sample_ids=ids, y=None if prediction else y, target_mask=None if prediction else mask,
        target_names=names, task_type="classification" if classification else "regression",
        groups=None if prediction else [f"group-{row // 6}" for row in rows],
        partitions=["predict"] * size if prediction else ["train"] * 96 + ["test"] * 12,
        name="incomplete-class" if classification else "incomplete-targets",
    )


def _pipeline(cohort: MultimodalDataset) -> list[Any]:
    classifier = cohort.task_type == "classification"
    def head():
        return LogisticRegression(C=0.7, max_iter=1000, solver="lbfgs") if classifier else Ridge(alpha=0.4)
    return [GroupKFold(3), {"branch": {"by_source": True,
        "missing_source_policy": "zero_with_indicator", "target_policy": "complete" if classifier else "per_target",
        "steps": {name: [StandardScaler(), head()] for name in reversed(tuple(cohort.sources))}}},
        {"merge": "predictions"}, LogisticRegression(C=0.9, max_iter=1000, solver="lbfgs") if classifier else Ridge(alpha=0.5)]


def _run(cohort: MultimodalDataset, workspace: Path, *, tuning: dict[str, Any] | None = None):
    return nirs4all.run(_pipeline(cohort), cohort, engine="dag-ml", random_state=41, refit=True,
                        save_artifacts=True, save_charts=False, verbose=0, workspace_path=workspace, tuning=tuning)


@pytest.fixture(autouse=True)
def forbid_fallback(monkeypatch):
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("legacy scheduler executed"))
    monkeypatch.setattr("nirs4all.pipeline.dagml.run_paths._run_model_on_precomputed_matrix", lambda *args, **kwargs: pytest.fail("Python fold scheduler executed"))


def _params(task, node):
    params = dict(node["params"])
    for choice in (task.get("variant") or {}).get("choices", {}).values():
        for override in choice.get("param_overrides", []):
            if override["node_id"] == node["id"]:
                params.update(override["params"])
    return params


def _oracle_observer(monkeypatch, cohort, *, upstream_factories=()):
    """Independent sklearn fits use actual native scopes, never product fit loops."""
    ledger, scopes = [], []
    raw, meta = node_runner.run_model_node, node_runner.run_meta_model_node
    classifier = cohort.task_type == "classification"

    def raw_observe(task, resolver, lookup, store, *args, **kwargs):
        result = raw(task, resolver, lookup, store, *args, **kwargs)
        node = lookup(task["node_plan"]["node_id"])
        name = node["metadata"].get("prediction_availability_source")
        if name is None or task["phase"] == "PREDICT":
            return result
        fit_view = next(view for view in task["data_views"].values() if view["partition"] in {"fold_train", "full_train"})
        native_ids = fit_view["sample_ids"]
        native_rows = [resolver._identity.to_int(sample) for sample in native_ids]
        observed = cohort.target_mask[native_rows].reshape(len(native_rows), -1)
        present = cohort.sources[name].presence_mask[native_rows]
        selected = present & observed.any(axis=1)
        fit_ids = [sample for sample, keep in zip(native_ids, selected, strict=True) if keep]
        fit_rows = np.asarray(native_rows)[selected]
        targets = np.asarray(resolver.resolve_targets(fit_ids)["values"]).reshape(len(fit_ids), -1)
        X = cohort.sources[name].values[fit_rows]
        params = _params(task, node)
        # Follow signed data edges independently of the product chain walker;
        # reconstruct oracle prefix parameters including native HPO overrides.
        incoming = {edge["target"]["node_id"]: edge["source"]["node_id"]
                    for edge in (args[0] if args else kwargs.get("edges", [])) or []
                    if edge["target"]["port_name"] in {"x", "x_original"} and edge["contract"]["kind"] == "data"}
        ancestors = []
        current = incoming.get(node["id"])
        while current is not None:
            ancestors.append(lookup(current))
            current = incoming.get(current)
        ancestors.reverse()
        assert len(ancestors) >= len(upstream_factories)
        prefixes = []
        for factory, declared in zip(upstream_factories, ancestors, strict=False):
            operator = declared["operator"]
            operator = operator["class"] if isinstance(operator, dict) else operator
            assert operator == factory.__module__ + "." + factory.__qualname__
            prefix_params = _params(task, declared)
            # JSON represents tuple constructor parameters as lists. Restore
            # the independent sklearn factory's declared defaults, not a
            # product decoder or learned value.
            defaults = factory().get_params(deep=False)
            prefix_params = {key: tuple(value) if isinstance(value, list) and isinstance(defaults.get(key), tuple) else value
                             for key, value in prefix_params.items()}
            prefixes.append((factory, prefix_params))
        heads = []
        for column in range(1 if classifier else targets.shape[1]):
            valid = np.ones(len(fit_ids), dtype=bool) if classifier else observed[selected, column]
            head = LogisticRegression(**params) if classifier else Ridge(**params)
            oracle = make_pipeline(*(factory(**params) for factory, params in prefixes), StandardScaler(), head).fit(X[valid], targets[valid, column])
            heads.append(oracle)
            scopes.append({"node": node["id"], "phase": task["phase"], "fold": task.get("fold_id"),
                           "target": column, "ids": [sample for sample, keep in zip(fit_ids, valid, strict=True) if keep]})
        for block in result["predictions"]:
            rows = [resolver._identity.to_int(sample) for sample in block["sample_ids"]]
            assert cohort.sources[name].presence_mask[rows].all()
            features = cohort.sources[name].values[rows]
            probabilities = block.get("producer_port") == "proba"
            expected = heads[0].predict_proba(features) if probabilities else np.column_stack([head.predict(features) for head in heads])
            np.testing.assert_allclose(block["values"], expected, rtol=1e-10, atol=1e-10)
            if probabilities:
                assert block["target_names"] == [json.dumps(float(label)) for label in heads[0].classes_]
            for sample, values in zip(block["sample_ids"], np.asarray(expected), strict=True):
                ledger.append((node["id"], task.get("variant_id"), block["fold_id"], block["partition"],
                               block.get("producer_port", "oof"), sample, values.copy(), frozenset(fit_ids)))
        return result

    def meta_observe(task, resolver, lookup, store, *args, **kwargs):
        result = meta(task, resolver, lookup, store, *args, **kwargs)
        node = lookup(task["node_plan"]["node_id"])
        if not node["metadata"].get("prediction_availability_meta") or task["phase"] == "PREDICT":
            return result
        ordered_nodes = node["metadata"]["prediction_source_order"]
        ports = node["metadata"].get("prediction_source_ports", {})

        def joined(suffix):
            specs = []
            for producer in ordered_nodes:
                key = producer + "." + ports.get(producer, "oof") + (":" + suffix if suffix else "")
                specs.append(task["prediction_inputs"][key])
            ids = specs[0]["sample_ids"]
            blocks = []
            for spec, name in zip(specs, cohort.sources, strict=True):
                assert spec["sample_ids"] == ids
                rows = [resolver._identity.to_int(sample) for sample in ids]
                present = cohort.sources[name].presence_mask[rows]
                np.testing.assert_array_equal(spec["source_presence"], present)
                values = np.asarray(spec["values"])
                np.testing.assert_array_equal(spec["feature_validity_masks"], np.broadcast_to(present[:, None], values.shape))
                np.testing.assert_array_equal(values[~present], 0)
                for row, sample in enumerate(ids):
                    if not present[row]:
                        continue
                    candidates = [item for item in ledger if item[0] == spec["producer_node"]
                                  and item[1] == task.get("variant_id") and item[4] == ports.get(spec["producer_node"], "oof")
                                  and item[5] == sample and np.allclose(item[6], values[row], rtol=1e-10, atol=1e-10)]
                    assert candidates, "native joined feature has no independently checked producer evidence"
                    if suffix is None:
                        assert any(item[3] == "validation" and sample not in item[7] for item in candidates), "meta FIT received in-sample producer evidence"
                blocks.append(np.column_stack([values, present.astype(float)]))
            return ids, np.column_stack(blocks)

        fit_ids, features = joined(None)
        target = resolver.resolve_targets(fit_ids)
        y = np.asarray(target["values"]).reshape(len(fit_ids), -1)
        mask = np.asarray(target.get("validity_masks", np.ones(y.shape, dtype=bool)))
        params = _params(task, node)
        heads = []
        for column in range(y.shape[1]):
            valid = np.ones(len(y), dtype=bool) if classifier else mask[:, column]
            estimator = LogisticRegression(**params) if classifier else Ridge(**params)
            heads.append(estimator.fit(features[valid], y[valid, column]))
        for block in result["predictions"]:
            suffix = "outer" if block["partition"] == "validation" else "test" if task["phase"] == "FIT_CV" else "refit"
            ids, X = joined(suffix)
            assert block["sample_ids"] == ids
            expected = heads[0].predict_proba(X) if block.get("producer_port") == "proba" else np.column_stack([head.predict(X) for head in heads])
            np.testing.assert_allclose(block["values"], expected, rtol=1e-10, atol=1e-10)
        return result

    monkeypatch.setattr(node_runner, "run_model_node", raw_observe)
    monkeypatch.setattr(node_runner, "run_meta_model_node", meta_observe)
    return scopes


def _meta_run(result):
    # Public multi-run export defaults to global CV-best, which can be a base.
    # This witness specifically exports the native learned whole stack.
    matches = [run for run in result.runs if any(
        artifact["controller_id"] == "controller:nirs4all.meta_model"
        for artifact in run._dagml_refit_artifacts)]
    assert len(matches) == 1, "expected exactly one genuine native meta REFIT run"
    return matches[0]


def _assert_independent_scores(result, classification):
    reports = {report["prediction_id"]: report for report in result._dagml_score_set["reports"]}
    checked = 0
    for frame in result._dagml_node_results:
        if frame.get("type") != "result" and "node_id" not in frame:
            continue
        node = frame["result"] if frame.get("type") == "result" else frame
        # Native avg/ensemble evidence frames contain aggregated_predictions only;
        # this oracle independently checks the original per-fold producer blocks.
        for block in node.get("predictions", []):
            report = reports.get(block["prediction_id"])
            if report is None or block.get("producer_port", "oof") != "oof":
                continue
            target = next(target for target in node["regression_targets"]
                          if [unit["id"] for unit in target["unit_ids"]] == block["sample_ids"])
            truth, predicted = np.asarray(target["values"]), np.asarray(block["values"])
            if classification:
                assert report["metrics"]["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(truth[:, 0], predicted[:, 0]), abs=1e-12)
                if "f1" in report["metrics"]:
                    assert report["metrics"]["f1"] == pytest.approx(f1_score(truth[:, 0], predicted[:, 0], average="weighted"), abs=1e-12)
            else:
                mask = np.asarray(target.get("validity_masks", np.ones(truth.shape, dtype=bool)))
                expected = []
                for column, name in enumerate(target["target_names"]):
                    valid = mask[:, column]
                    metric = root_mean_squared_error(truth[valid, column], predicted[valid, column])
                    assert report["metrics"]["rmse:" + name] == pytest.approx(metric, rel=1e-10, abs=1e-12)
                    expected.append(metric)
                assert report["metrics"]["rmse"] == pytest.approx(np.mean(expected), rel=1e-10, abs=1e-12)
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("source_count", [2, 3, 4])
def test_real_grouped_nested_fits_match_independent_masked_oracle(classification, source_count, tmp_path, monkeypatch):
    cohort = _cohort(classification=classification, source_count=source_count)
    scopes = _oracle_observer(monkeypatch, cohort)
    result = _run(cohort, tmp_path / "workspace")
    try:
        assert scopes and any("inner" in (scope["fold"] or "") for scope in scopes)
        assert any(scope["phase"] == "REFIT" for scope in scopes)
        assert all(set(scope["ids"]) <= set(cohort.sample_ids[:96]) for scope in scopes)
        meta = next(item for item in result._dagml_refit_artifacts if item["controller_id"] == "controller:nirs4all.meta_model")
        fitted = meta["estimator"]
        assert fitted.n_features_in_ == source_count * (4 if classification else 3)
        origin = meta["late_partial_refit_origin"]
        assert origin["source_order"] == list(cohort.sources)
        assert origin["target_policy"] == ("complete" if classification else "per_target")
        assert np.isfinite(result.cv_best_score)
        _assert_independent_scores(result, classification)
    finally:
        result.close()


@pytest.mark.parametrize("classification", [False, True])
def test_whole_stack_hpo_and_hidden_values_preserve_real_evidence(classification, tmp_path, monkeypatch):
    cohort = _cohort(classification=classification)
    _oracle_observer(monkeypatch, cohort)
    parameter = "C" if classification else "alpha"
    tuning = {"engine": "n4m", "sampler": "random", "seed": 41, "n_jobs": 1, "n_trials": 2,
              "metric": "balanced_accuracy" if classification else "rmse", "direction": "maximize" if classification else "minimize",
              "storage": (tmp_path / "study").as_uri(), "study_name": "partial", "space": {"meta." + parameter: [0.3, 0.9]}}
    result = _run(cohort, tmp_path / "workspace", tuning=tuning)
    try:
        assert len(result.tuning_result.trials) == 2
        assert all(trial.state == "COMPLETE" and trial.diagnostics["test_used"] is False for trial in result.tuning_result.trials)
        assert result.tuning_best_params["meta." + parameter] in [0.3, 0.9]
    finally:
        result.close()


@pytest.mark.parametrize("classification,integer_labels", [(False, False), (True, False), (True, True)])
def test_export_load_cold_replay_keeps_typed_labels_and_entire_absent_source(classification, integer_labels, tmp_path):
    cohort = _cohort(classification=classification, integer_labels=integer_labels)
    prediction = _cohort(classification=classification, integer_labels=integer_labels, prediction=True)
    result = _run(cohort, tmp_path / "workspace")
    try:
        archive = _meta_run(result).export(tmp_path / "partial.n4a")
        expected = nirs4all.predict(archive, prediction)
    finally:
        result.close()
    payload = tmp_path / "prediction.json"
    payload.write_text(json.dumps(prediction.to_dict()), encoding="utf-8")
    child = os.environ.get("NIRS4ALL_PARTIAL_LATE_INSTALLED_PYTHON")
    if not child:
        if os.environ.get("NIRS4ALL_REQUIRE_PARTIAL_LATE_INSTALLED") == "1":
            pytest.fail("mandatory installed partial-cohort replay Python is unavailable")
        pytest.skip("opt-in installed partial-cohort replay child is unavailable")
    script = """import json,sys,numpy as np,nirs4all
from nirs4all_io import MultimodalDataset
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.preprocessing import StandardScaler
for cls in (Ridge,LogisticRegression,StandardScaler):
    cls.fit=lambda *a,**k: (_ for _ in ()).throw(AssertionError('cold replay attempted FIT'))
cohort=MultimodalDataset.from_dict(json.load(open(sys.argv[2])))
result=nirs4all.predict(sys.argv[1],cohort)
json.dump({'values':np.asarray(result.y_pred).tolist(),'sample_ids':result.metadata['sample_ids'],'training_performed':result.metadata['training_performed']},open(sys.argv[3],'w'))
"""
    output = tmp_path / "cold.json"
    completed = subprocess.run([child, "-I", "-c", script, str(archive), str(payload), str(output)], capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    actual = json.loads(output.read_text())
    assert actual["sample_ids"] == list(prediction.sample_ids)
    assert actual["training_performed"] is False
    if classification:
        np.testing.assert_array_equal(actual["values"], expected.y_pred)
        assert set(np.asarray(expected.y_pred).reshape(-1)) <= set(np.asarray(cohort.y).reshape(-1))
    else:
        np.testing.assert_allclose(actual["values"], expected.y_pred, rtol=1e-12, atol=1e-12)


def test_all_scope_admission_refuses_empty_intersection_before_any_fit_or_trial(tmp_path, monkeypatch):
    cohort = _cohort(classification=False)
    source = cohort.sources["source0"]
    presence = np.zeros(len(cohort), dtype=bool)
    broken = TensorSource(source.values, source.sample_ids, representation_id="signal_1d", presence_mask=presence)
    invalid = MultimodalDataset({**cohort.sources, "source0": broken}, sample_ids=cohort.sample_ids,
                               y=cohort.y, target_names=cohort.target_names, target_mask=cohort.target_mask,
                               task_type="regression", groups=cohort.groups, partitions=cohort.partitions)
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("inadmissible native scope reached model FIT"))
    monkeypatch.setattr(StandardScaler, "fit", lambda *args, **kwargs: pytest.fail("inadmissible native scope reached encoder FIT"))
    with pytest.raises(Exception, match="availability|observed|intersection|present"):
        _run(invalid, tmp_path / "workspace")


@pytest.mark.parametrize("classification", [False, True])
def test_hidden_source_and_target_cells_cannot_change_native_oof(classification, tmp_path):
    before = _run(_cohort(classification=classification), tmp_path / "before")
    after = _run(_cohort(classification=classification, hidden=1e25), tmp_path / "after")
    try:
        for left, right in zip(before.runs, after.runs, strict=True):
            for fold in range(3):
                expected = left.predictions.filter_predictions(fold_id=str(fold), partition="val", load_arrays=True)[0]
                actual = right.predictions.filter_predictions(fold_id=str(fold), partition="val", load_arrays=True)[0]
                assert expected["sample_indices"] == actual["sample_indices"]
                np.testing.assert_array_equal(expected["y_pred"], actual["y_pred"])
    finally:
        before.close()
        after.close()


@pytest.mark.parametrize("classification", [False, True])
def test_partial_cohort_checkpoint_resumes_the_exact_whole_stack(classification, tmp_path):
    from nirs4all.pipeline.dagml.multimodal_tuning import MultimodalTuningStopped
    from tests.integration.api.test_multimodal_tuning import _stop_after

    cohort = _cohort(classification=classification)
    parameter = "C" if classification else "alpha"
    study = tmp_path / "study"
    options = {"engine": "n4m", "sampler": "random", "seed": 41, "n_jobs": 1, "n_trials": 2,
               "metric": "f1" if classification else "rmse", "direction": "maximize" if classification else "minimize",
               "storage": study.as_uri(), "study_name": "partial", "space": {"meta." + parameter: [0.3, 0.9]}}
    with pytest.raises(MultimodalTuningStopped):
        _run(cohort, tmp_path / "stopped", tuning={**options, "progress_callback": _stop_after(1, [])})
    checkpoint = study / "partial.n4mopt.json"
    prefix = json.loads(checkpoint.read_text(encoding="utf-8"))["native_checkpoint"]["trials"]
    resumed = _run(cohort, tmp_path / "resumed", tuning={**options, "resume": True})
    continuous = _run(cohort, tmp_path / "continuous", tuning={**options, "storage": (tmp_path / "continuous-study").as_uri()})
    try:
        assert [trial.to_dict() for trial in resumed.tuning_result.trials] == [trial.to_dict() for trial in continuous.tuning_result.trials]
        assert resumed.tuning_best_params == continuous.tuning_best_params
        assert json.loads(checkpoint.read_text(encoding="utf-8"))["native_checkpoint"]["trials"][:1] == prefix
    finally:
        resumed.close()
        continuous.close()


def test_outer_validation_labels_never_enter_their_inner_encoders_or_meta_fit(tmp_path):
    cohort = _cohort(classification=False)
    validation = next(GroupKFold(3).split(np.zeros((96, 1)), groups=cohort.groups[:96]))[1]
    targets = cohort.y.copy()
    targets[validation] += 1e7
    poisoned = MultimodalDataset(cohort.sources, sample_ids=cohort.sample_ids, y=targets,
                                 target_names=cohort.target_names, target_mask=cohort.target_mask,
                                 task_type="regression", groups=cohort.groups, partitions=cohort.partitions)
    before, after = _run(cohort, tmp_path / "before"), _run(poisoned, tmp_path / "after")
    try:
        for left, right in zip(before.runs, after.runs, strict=True):
            expected = left.predictions.filter_predictions(fold_id="0", partition="val", load_arrays=True)[0]
            actual = right.predictions.filter_predictions(fold_id="0", partition="val", load_arrays=True)[0]
            assert expected["sample_indices"] == actual["sample_indices"]
            np.testing.assert_array_equal(expected["y_pred"], actual["y_pred"])
    finally:
        before.close()
        after.close()
