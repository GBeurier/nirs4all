"""Actual native scopes, independent encoder oracle, HPO and installed cold replay."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

import nirs4all
from nirs4all.pipeline.dagml import node_runner
from tests.integration.api.test_incomplete_source_stacking_execution import (
    _assert_independent_scores,
    _cohort,
    _meta_run,
    _oracle_observer,
    _pipeline,
)
from tests.integration.api.test_incomplete_source_stacking_execution import (
    forbid_fallback as forbid_fallback,
)


def _run_prefix(cohort, workspace: Path, *, tuning=None):
    # The public prefix is shared syntax; learned states belong to each source
    # and each actual native inner/outer/REFIT target intersection.
    return nirs4all.run([MinMaxScaler(), *_pipeline(cohort)], cohort, engine="dag-ml", random_state=41,
                        refit=True, save_artifacts=True, save_charts=False, verbose=0,
                        workspace_path=workspace, tuning=tuning)


def _assert_real_prefix_fit_scopes(monkeypatch, cohort):
    """Check real encoder inputs/state, even when affine downstream steps cancel."""
    raw, fit = node_runner.run_model_node, MinMaxScaler.fit
    captured = None

    def witnessed_fit(estimator, X, y=None, **kwargs):
        fitted = fit(estimator, X, y, **kwargs)
        if captured is not None:
            captured.append((np.asarray(X).copy(), estimator.data_min_.copy(), estimator.data_max_.copy()))
        return fitted

    def witnessed_raw(task, resolver, lookup, store, *args, **kwargs):
        nonlocal captured
        node = lookup(task["node_plan"]["node_id"])
        name = node["metadata"].get("prediction_availability_source")
        if name is None or task["phase"] == "PREDICT":
            return raw(task, resolver, lookup, store, *args, **kwargs)
        view = next(view for view in task["data_views"].values() if view["partition"] in {"fold_train", "full_train"})
        rows = np.asarray([resolver._identity.to_int(sample) for sample in view["sample_ids"]])
        present = cohort.sources[name].presence_mask[rows]
        observed = cohort.target_mask[rows].reshape(len(rows), -1)
        expected = [cohort.sources[name].values[rows[present & observed[:, column]]]
                    for column in range(observed.shape[1])]
        captured = []
        try:
            result = raw(task, resolver, lookup, store, *args, **kwargs)
            assert len(captured) == len(expected)
            for (actual, minima, maxima), X in zip(captured, expected, strict=True):
                np.testing.assert_array_equal(actual, X)
                np.testing.assert_array_equal(minima, X.min(axis=0))
                np.testing.assert_array_equal(maxima, X.max(axis=0))
            return result
        finally:
            captured = None

    monkeypatch.setattr(MinMaxScaler, "fit", witnessed_fit)
    monkeypatch.setattr(node_runner, "run_model_node", witnessed_raw)


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("source_count", [2, 4])
def test_upstream_encoder_matches_independent_train_only_oracle_in_all_native_grouped_scopes(classification, source_count, tmp_path, monkeypatch):
    cohort = _cohort(classification=classification, source_count=source_count)
    _assert_real_prefix_fit_scopes(monkeypatch, cohort)
    scopes = _oracle_observer(monkeypatch, cohort, upstream_factories=(MinMaxScaler,))
    result = _run_prefix(cohort, tmp_path / "workspace")
    try:
        assert any("inner" in (scope["fold"] or "") for scope in scopes)
        assert any(scope["phase"] == "REFIT" for scope in scopes)
        _assert_independent_scores(result, classification)
        bases = [artifact for artifact in result._dagml_refit_artifacts
                 if artifact.get("late_partial_refit_origin", {}).get("source_name") is not None]
        assert len(bases) == source_count
        for artifact in bases:
            estimator = artifact["estimator"]
            if classification:
                assert isinstance(estimator.steps[0][1], MinMaxScaler)
                assert len(estimator.steps) == 3
            else:
                assert isinstance(estimator.chain_template[0], MinMaxScaler)
                assert len(estimator.chain_template) == 2
    finally:
        result.close()


@pytest.mark.parametrize("classification", [False, True])
def test_prefix_whole_stack_hpo_and_selected_refit_remain_native(classification, tmp_path, monkeypatch):
    cohort = _cohort(classification=classification)
    _oracle_observer(monkeypatch, cohort, upstream_factories=(MinMaxScaler,))
    parameter = "C" if classification else "alpha"
    options = {"engine": "n4m", "sampler": "random", "seed": 41, "n_jobs": 1, "n_trials": 2,
        "metric": "balanced_accuracy" if classification else "rmse",
        "direction": "maximize" if classification else "minimize",
        "storage": (tmp_path / "study").as_uri(), "study_name": "upstream",
        "space": {"branches.source0.0.clip": [False, True], "meta." + parameter: [0.3, 0.9]}}
    result = _run_prefix(cohort, tmp_path / "workspace", tuning=options)
    try:
        assert len(result.tuning_result.trials) == 2
        assert all(trial.state == "COMPLETE" and trial.diagnostics["test_used"] is False
                   for trial in result.tuning_result.trials)
        assert result.tuning_best_params["branches.source0.0.clip"] in [False, True]
        _assert_independent_scores(result, classification)
    finally:
        result.close()


@pytest.mark.parametrize("classification,integer_labels", [(False, False), (True, False), (True, True)])
def test_prefix_cold_archive_replay_retains_original_typed_labels_and_absent_source(classification, integer_labels, tmp_path):
    cohort = _cohort(classification=classification, integer_labels=integer_labels)
    prediction = _cohort(classification=classification, integer_labels=integer_labels, prediction=True)
    result = _run_prefix(cohort, tmp_path / "workspace")
    try:
        archive = _meta_run(result).export(tmp_path / "upstream.n4a")
        expected = nirs4all.predict(archive, prediction)
    finally:
        result.close()
    child = os.environ.get("NIRS4ALL_PARTIAL_LATE_INSTALLED_PYTHON")
    if not child:
        if os.environ.get("NIRS4ALL_REQUIRE_PARTIAL_LATE_INSTALLED") == "1":
            pytest.fail("mandatory installed partial-cohort replay Python is unavailable")
        pytest.skip("opt-in installed partial-cohort replay child is unavailable")
    payload = tmp_path / "prediction.json"
    payload.write_text(json.dumps(prediction.to_dict()), encoding="utf-8")
    output = tmp_path / "cold.json"
    script = """import json,sys,numpy as np,nirs4all
from nirs4all_io import MultimodalDataset
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.preprocessing import StandardScaler,MinMaxScaler
for cls in (Ridge,LogisticRegression,StandardScaler,MinMaxScaler):
    cls.fit=lambda *a,**k: (_ for _ in ()).throw(AssertionError('cold replay attempted FIT'))
cohort=MultimodalDataset.from_dict(json.load(open(sys.argv[2])))
result=nirs4all.predict(sys.argv[1],cohort)
json.dump({'values':np.asarray(result.y_pred).tolist(),'sample_ids':result.metadata['sample_ids'],
           'training_performed':result.metadata['training_performed']},open(sys.argv[3],'w'))
"""
    completed = subprocess.run([child, "-I", "-c", script, str(archive), str(payload), str(output)],
                               capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    actual = json.loads(output.read_text())
    assert actual["training_performed"] is False and actual["sample_ids"] == list(prediction.sample_ids)
    if classification:
        np.testing.assert_array_equal(actual["values"], expected.y_pred)
        assert set(np.asarray(expected.y_pred).reshape(-1)) <= set(np.asarray(cohort.y).reshape(-1))
    else:
        np.testing.assert_allclose(actual["values"], expected.y_pred, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("classification", [False, True])
def test_mutated_learned_prefix_cannot_reseal_the_original_refit_capture(classification, tmp_path):
    from nirs4all.pipeline.dagml.multimodal_contracts import validate_late_partial_refit_origin

    result = _run_prefix(_cohort(classification=classification), tmp_path / "workspace")
    try:
        artifact = next(artifact for artifact in result._dagml_refit_artifacts
                        if artifact.get("late_partial_refit_origin", {}).get("source_name") is not None)
        validate_late_partial_refit_origin(artifact)
        estimator = artifact["estimator"]
        name = artifact["late_partial_refit_origin"]["source_name"]
        chain = estimator if classification else estimator.model_.target_models_[0].transformers_[name]
        prefix = chain.steps[0][1]
        assert isinstance(prefix, MinMaxScaler) and hasattr(prefix, "data_min_")
        prefix.data_min_[0] += 0.125
        with pytest.raises(ValueError, match="state|fingerprint|REFIT"):
            validate_late_partial_refit_origin(artifact)
    finally:
        result.close()


def test_weighted_native_prefix_receives_true_unit_weights_on_each_source_target_intersection(tmp_path, monkeypatch):
    from tests.integration.api.test_multimodal_experimental_units import (
        _assert_native_scopes,
        _Observer,
        _weight_oracle,
    )
    from tests.integration.api.test_multimodal_experimental_units import (
        _cohort as units_cohort,
    )
    from tests.integration.api.test_multimodal_experimental_units import (
        _pipeline as units_pipeline,
    )

    cohort = units_cohort(partial=True)
    observer = _Observer()
    observer.install(monkeypatch)
    from sklearn.preprocessing import StandardScaler

    original = StandardScaler.fit
    fits = []

    def witnessed_fit(estimator, X, y=None, sample_weight=None):
        fitted = original(estimator, X, y, sample_weight=sample_weight)
        if estimator.with_mean is False:
            assert sample_weight is not None, "upstream FIT silently dropped native unit weights"
            fits.append((np.asarray(X).copy(), np.asarray(sample_weight).copy()))
        return fitted

    monkeypatch.setattr(StandardScaler, "fit", witnessed_fit)
    result = nirs4all.run([StandardScaler(with_mean=False), *units_pipeline(partial=True)], cohort,
        engine="dag-ml", random_state=41, refit=True, save_artifacts=True, save_charts=False,
        verbose=0, workspace_path=tmp_path / "workspace")
    try:
        _assert_native_scopes(cohort, observer.tasks, partial=True)
        assert fits
        for X, weights in fits:
            source = next(source for source in cohort.sources.values() if source.values.shape[1] == X.shape[1])
            matches = (np.asarray(source.values)[None, :, :] == X[:, None, :]).all(axis=2)
            assert (matches.sum(axis=1) == 1).all()
            rows = matches.argmax(axis=1)
            assert source.presence_mask[rows].all()
            ids = [cohort.sample_ids[row] for row in rows]
            np.testing.assert_allclose(weights, _weight_oracle(cohort, ids), rtol=1e-14, atol=1e-14)
    finally:
        result.close()


def test_unweighted_prefix_on_experimental_units_is_refused_before_any_fit_or_native_ask(tmp_path, monkeypatch):
    import dag_ml
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    from tests.integration.api.test_multimodal_experimental_units import _cohort as units_cohort
    from tests.integration.api.test_multimodal_experimental_units import _pipeline as units_pipeline

    def forbidden(*args, **kwargs):
        pytest.fail("unweighted prefix reached FIT or native HPO ask")

    from functools import wraps

    def forbid_fit(method):
        @wraps(method)
        def blocked(*args, **kwargs):
            return forbidden(*args, **kwargs)
        return blocked

    for cls in (MinMaxScaler, StandardScaler, Ridge):
        monkeypatch.setattr(cls, "fit", forbid_fit(cls.fit))
    monkeypatch.setattr(dag_ml, "run_host_hpo_search_in_process", forbidden)
    with pytest.raises(ValueError, match="MinMaxScaler.*sample_weight"):
        nirs4all.run([MinMaxScaler(), *units_pipeline(partial=True)], units_cohort(partial=True),
            engine="dag-ml", random_state=41, refit=True, save_artifacts=True, save_charts=False,
            verbose=0, workspace_path=tmp_path / "workspace")
