"""Classification evidence is tied to real fitted native tasks, never a refit substitute."""

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from nirs4all.pipeline.dagml.native_vote import _ensemble_record, capture_vote_evidence, collect_vote_evidence, project_vote_evidence


class FittedClassifier(ClassifierMixin, BaseEstimator):
    classes_ = np.array([0., 2.])

    def fit(self, *args):
        raise AssertionError("evidence collection must never fit")

    def predict_proba(self, x):
        positive = np.asarray(x)[:, 0] / 10
        return np.column_stack([1 - positive, positive])


def test_capture_retains_native_fold_proba_ids_and_separate_training_views():
    resolver = SimpleNamespace(
        _dataset=SimpleNamespace(repetition="physical_sample", aggregate_method="vote"),
        partition_wire_ids=lambda part: {"train": ["a", "b", "c"], "test": ["d"]}[part],
        resolve_targets=lambda ids: {"values": [0. if identity in {"a", "b"} else 2. for identity in ids]},
    )
    indices = {"a": 1, "b": 2, "c": 8, "d": 9}
    materialized = []

    def features(ids, include_augmented):
        assert not include_augmented
        materialized.append(ids)
        return np.array([[indices[identity]] for identity in ids])

    def predict(ids, include_augmented):
        return [[0. if identity in {"a", "b"} else 2.] for identity in ids]

    task = {"phase": "FIT_CV", "fold_id": "fold0", "variant_id": None, "run_id": "real-run",
            "node_plan": {"node_id": "model", "params_fingerprint": "native-params"}}
    store = {101: {"estimator": "unrelated"}}
    capture_vote_evidence(task, resolver, FittedClassifier(),
                          [{"partition": "validation", "sample_ids": ["c"], "values": [[2.]]}],
                          ["b", "a"], features, predict, store)
    records = {record["partition"]: record for record in collect_vote_evidence(store)}
    assert records["train"]["sample_ids"] == ["b", "a"]
    assert records["validation"]["sample_ids"] == ["c"]
    assert records["ensemble_train"]["sample_ids"] == ["a", "b", "c"]
    assert records["test"]["sample_ids"] == ["d"]
    np.testing.assert_allclose(records["test"]["y_proba"], [[.1, .9]])
    assert records["test"]["classes"] == [0., 2.]
    assert all(record["training_performed_for_evidence"] is False for record in records.values())
    assert 101 in store and len(materialized) == 4


def _record(classes, proba, pred=2.):
    return {"node_id": "model", "variant_id": None, "phase": "FIT_CV", "fold_id": "fold0",
            "partition": "test", "sample_ids": ["sample"], "classes": classes,
            "y_true": [2.], "y_pred": [pred], "y_proba": proba}


def test_fold_probability_alignment_uses_class_identity_not_column_position():
    first = _record([0., 2.], [[.1, .9]])
    second = _record([0., 1., 2.], [[.1, .8, .1]], 1.)
    result = _ensemble_record([first, second], None, np.array([0., 1., 2.]), oof=False)
    np.testing.assert_allclose(result["y_proba"], [[.1, .4, .5]])
    assert result["y_pred"] == [2.]
    assert result["sample_ids"] == ["sample"]


def test_hard_vote_does_not_invent_probabilities():
    records = [_record([0., 2.], None, label) for label in (2., 0., 2.)]
    result = _ensemble_record(records, None, np.array([0., 2.]), oof=False)
    assert result["y_pred"] == [2.]
    assert result["y_proba"] is None


def test_probability_ensemble_refuses_misaligned_sample_or_target_identity():
    first = _record([0., 2.], [[.1, .9]])
    second = {**first, "sample_ids": ["different"]}
    with pytest.raises(ValueError, match="identities"):
        _ensemble_record([first, second], None, np.array([0., 2.]), oof=False)


def test_projection_refuses_evidence_disagreeing_with_native_block():
    record = {**_record([0., 2.], [[.1, .9]]), "phase": "REFIT", "fold_id": None}
    entry = {"fold_id": "final", "partition": "test", "sample_indices": [4], "y_true": [2.], "y_pred": [0.]}
    result = SimpleNamespace(predictions=SimpleNamespace(iter_entries=lambda: [entry]))
    with pytest.raises(ValueError, match="disagrees"):
        project_vote_evidence(result, [record], SimpleNamespace(to_int=lambda _: 4))


def test_subprocess_sidecar_stays_out_of_protocol_and_is_drained_once(tmp_path):
    import io
    import json

    from nirs4all.pipeline.dagml.process_adapter import _capture_vote_sidecar, run_jsonl_loop

    task = {"node_plan": {"node_id": "model"}, "phase": "FIT_CV", "fold_id": "fold0"}
    record = _record([0., 2.], [[.1, .9]])
    store = {("classification_evidence", "model", None, "FIT_CV", "fold0"): [record], 7: "artifact"}
    capture = tmp_path / "capture.jsonl"

    def handler(current):
        _capture_vote_sidecar(current, store, str(capture))
        return {"node_id": "model", "predictions": []}

    output = io.StringIO()
    run_jsonl_loop(io.StringIO(json.dumps({"type": "task", "task": task}) + "\n"), output, handler)
    assert json.loads(output.getvalue())["type"] == "result"
    assert "classification_evidence" not in output.getvalue()
    assert json.loads(capture.read_text())["records"] == [record]
    _capture_vote_sidecar(task, store, str(capture))
    assert len(capture.read_text().splitlines()) == 1
    assert store == {7: "artifact"}


@pytest.mark.parametrize("version", [1, 2])
def test_subprocess_router_separates_host_evidence_from_native_results(tmp_path, monkeypatch, version):
    import json

    from nirs4all.pipeline.dagml import cli_runner, in_process_runner

    record = _record([0., 2.], [[.1, .9]])
    native = {"node_id": "model", "predictions": []}
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0")
    cli = tmp_path / "cli"
    cli.touch()
    (tmp_path / "bundle.json").write_text(json.dumps({"scores": {"reports": []}}))
    monkeypatch.setattr(cli_runner, "run_cv_refit_bundle", lambda **kwargs: {
        "returncode": 0, "results": [native, {"type": "nirs4all_classification_evidence", "schema_version": version, "records": [record]}],
    })
    kwargs = {"dsl": {}, "envelope": {}, "graph": {}, "dataset_path": "", "workdir": tmp_path, "dagml_cli": str(cli), "venv_python": "unused"}
    if version != 1:
        with pytest.raises(ValueError, match="invalid classification evidence sidecar"):
            in_process_runner.run_cv_refit_bundle_router(**kwargs)
    else:
        outcome = in_process_runner.run_cv_refit_bundle_router(**kwargs)
        assert outcome["results"] == [native]
        assert outcome["classification_evidence"] == [record]
        assert outcome["scores"] == {"reports": []}


def test_real_native_vote_preserves_all_fold_and_final_arrays_without_extra_fits(tmp_path, monkeypatch):
    import copy
    import sys

    from sklearn.neighbors import KNeighborsClassifier

    import nirs4all
    from nirs4all.pipeline.dagml import run_paths
    from nirs4all.pipeline.dagml.run_paths import _run_repetition_concrete
    from tests.integration.parity import cases_aggregation_reps  # noqa: F401 - register the immutable corpus cases
    from tests.integration.parity._conformance_helpers import make_dataset
    from tests.integration.parity._registry import all_cases

    case = next(case for case in all_cases() if case.name == "aggregation_classification_vote")
    configs = make_dataset(case)
    legacy = nirs4all.run(case.pipeline, configs, engine="legacy", verbose=0, save_charts=False, workspace_path=tmp_path / "oracle")
    native_fits = []
    original_fit = KNeighborsClassifier.fit
    original_run = run_paths.run_cv_refit_bundle
    native_scores = []

    def capture_run(**kwargs):
        outcome = original_run(**kwargs)
        native_scores.append(copy.deepcopy(outcome["scores"]))
        return outcome

    def counted_fit(self, x, y):
        native_fits.append(len(x))
        return original_fit(self, x, y)

    monkeypatch.setattr(KNeighborsClassifier, "fit", counted_fit)
    monkeypatch.setattr(run_paths, "run_cv_refit_bundle", capture_run)
    native = _run_repetition_concrete(case.pipeline, configs.get_dataset_at(0), "", "", sys.executable,
                                      tmp_path / "native", "balanced_accuracy", "classification")
    assert len(native_fits) == 4  # three native CV tasks + one native REFIT, no evidence refit
    assert native_fits[-1] == 178
    assert native.best_score == pytest.approx(legacy.best_score, abs=1e-12)
    assert native.best_accuracy == legacy.best_accuracy
    assert native.cv_best_score == legacy.cv_best_score
    assert native._dagml_score_set == native_scores[0]
    expected = {(str(row["fold_id"]), row["partition"]): row for row in legacy.predictions.iter_entries()}
    observed = {(str(row["fold_id"]), row["partition"]): row for row in native.predictions.iter_entries()}
    assert len(expected) == len(observed) == 34
    for key, reference in expected.items():
        row = observed[key]
        assert row["n_samples"] == reference["n_samples"]
        positions = {sample: index for index, sample in enumerate(reference["sample_indices"])}
        order = [positions[sample] for sample in row["sample_indices"]] if positions else slice(None)
        np.testing.assert_array_equal(row["y_pred"], np.asarray(reference["y_pred"]).ravel()[order], err_msg=str(key))
        np.testing.assert_array_equal(row["y_true"], np.asarray(reference["y_true"]).ravel()[order], err_msg=str(key))
        expected_proba = np.asarray(reference["y_proba"])[order]
        if expected_proba.shape[1] < row["y_proba"].shape[1]:
            expected_proba = np.pad(expected_proba, ((0, 0), (0, row["y_proba"].shape[1] - expected_proba.shape[1])))
        np.testing.assert_allclose(row["y_proba"], expected_proba, atol=1e-14, rtol=1e-14, err_msg=str(key))
        if key[0].endswith("_agg"):
            assert row["result_metadata"]["aggregate_evidence"]["selection_score"] is False
        else:
            assert row["result_metadata"]["classification_evidence"]["training_performed_for_evidence"] is False
            assert row["metadata"]["physical_sample_id"] == row["result_metadata"]["classification_evidence"]["sample_ids"]
