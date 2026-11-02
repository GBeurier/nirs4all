"""Progress events describe actual CV tasks and preserve scientific results."""
import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.pipeline.dagml.progress import ExecutionProgress, execution_progress, observe_progress


def test_native_sweep_reports_all_real_folds_and_refit_without_changing_scores(tmp_path):
    rng = np.random.default_rng(32)
    x = rng.normal(size=(36, 8))
    y = 3 * x[:, 0] + x[:, 1]
    pipeline = [KFold(3), {"model": PLSRegression(), "_range_": [1, 3, 1], "param": "n_components"}]
    options = {"engine": "dag-ml", "allow_fallback": False, "save_artifacts": False, "verbose": 0}
    baseline = nirs4all.run(pipeline, (x, y), workspace_path=tmp_path / "baseline", **options)
    events = []
    with observe_progress(events.append):
        observed = nirs4all.run(pipeline, (x, y), workspace_path=tmp_path / "observed", **options)
    assert observed.cv_best_score == pytest.approx(baseline.cv_best_score, rel=1e-12)
    assert execution_progress() is None
    assert any(event.get("phase") == "REFIT" for event in events)
    finished = [event for event in events if "Fitting:" in event["message"] and "(completed)" in event["message"]]
    assert len(finished) == 9
    assert finished[-1]["completed_cv_fits"] == finished[-1]["total_cv_fits"] == 9
    assert {event["current_fold"] for event in finished} == {1, 2, 3}
    assert {event["current_variant"] for event in finished} == {1, 2, 3}
    assert all(event["operator"] == "PLSRegression" for event in finished)
    values = [event["progress"] for event in events if event["progress"] is not None]
    assert values == sorted(values)
    assert 0 < values[-1] < 100


def test_progress_scope_resets_after_scientific_failure():
    with pytest.raises(RuntimeError, match="operator failed"), observe_progress(lambda _: None):
        raise RuntimeError("operator failed")
    assert execution_progress() is None


def _bundle(folds, models, tuner=False):
    """A minimal DSL/graph pair exposing only what the observer reads."""
    dsl = {"split_invocation": {"fold_set": {"folds": [{"fold_id": f"fold-{index}"} for index in range(folds)]}}}
    nodes = [{"id": f"model:{index}", "kind": "model"} for index in range(models)]
    nodes += [{"id": "tuner:0", "kind": "tuner"}] if tuner else []
    return dsl, {"nodes": [{"id": "transform:0", "kind": "transform"}, *nodes]}


def _fit_cv(observer, bundle, folds, variant, model, fold):
    task = {"phase": "FIT_CV", "variant_id": variant, "fold_id": f"fold-{fold}", "node_plan": {"node_id": f"model:{model}", "kind": "model"}}
    observer.task(bundle, folds, task, {"operator": "sklearn.PLSRegression"}, completed=False)
    observer.task(bundle, folds, task, {"operator": "sklearn.PLSRegression"}, completed=True)


def _run_dispatch(observer, variants, folds, models, *, fits=None):
    """Drive one dispatch the way the in-process runner does: one bundle per variant."""
    observer.configure(variants, "data")
    done = 0
    for variant in range(variants):
        bundle, labels = observer.open_bundle(*_bundle(folds, models))
        for fold in range(folds):
            for model in range(models):
                if fits is not None and done >= fits:
                    return
                _fit_cv(observer, bundle, labels, f"v{variant}", model, fold)
                done += 1


def test_repeated_unannounced_dispatch_makes_the_global_total_unknown():
    events = []
    with observe_progress(events.append) as observer:
        _run_dispatch(observer, 2, 3, 1)
        assert events[-1]["total_cv_fits"] == 6
        observer.configure(2, "second")
        assert not observer.completed and not observer.variants
        assert observer.total_folds == 0
        assert events[-1]["progress"] is None
        assert events[-1]["total_cv_fits"] is None
        assert events[-1]["completed_cv_fits"] == 6


def test_total_counts_every_model_node_of_a_dispatch():
    events = []
    with observe_progress(events.append) as observer:
        _run_dispatch(observer, 2, 3, 2)
    assert events[-1]["total_cv_fits"] == 12 and events[-1]["completed_cv_fits"] == 12
    assert all(event["progress"] is not None for event in events if event["total_cv_fits"])
    values = [event["progress"] for event in events if event["progress"] is not None]
    assert values == sorted(values) and values[-1] == 99.0
    halfway = next(event for event in events if event["completed_cv_fits"] == 6)
    assert halfway["progress"] == pytest.approx(50.0)


def test_model_node_count_drift_or_tuner_makes_the_total_unknown():
    with observe_progress(lambda _: None) as observer:
        observer.configure(2, "data")
        observer.open_bundle(*_bundle(3, 1))
        assert observer.reliable_total
        observer.open_bundle(*_bundle(3, 2))
        assert not observer.reliable_total
    with observe_progress(lambda _: None) as observer:
        observer.configure(1, "data")
        observer.open_bundle(*_bundle(3, 1, tuner=True))
        assert not observer.reliable_total


def test_announced_dispatches_share_one_denominator_and_never_move_backwards():
    events = []
    with observe_progress(events.append) as observer:
        observer.plan_dispatches(4)
        for _ in range(4):
            _run_dispatch(observer, 3, 3, 1)
            if len(events) and observer.dispatches == 1:
                first = next(event for event in reversed(events) if event["completed_cv_fits"] == 9)
                # The first model of four is a quarter of the work, not 99 % of it.
                assert first["progress"] == pytest.approx(25.0)
                assert first["total_cv_fits"] == 36
    assert events[-1]["completed_cv_fits"] == events[-1]["total_cv_fits"] == 36
    assert events[-1]["progress"] == 99.0
    assert all(event["total_cv_fits"] in (0, 36) for event in events)
    values = [event["progress"] for event in events if event["progress"] is not None]
    assert values == sorted(values)
    assert "Model 4/4" in events[-1]["message"]


def test_later_dispatch_larger_than_the_estimate_keeps_the_bar_monotone():
    events = []
    with observe_progress(events.append) as observer:
        observer.plan_dispatches(2)
        _run_dispatch(observer, 1, 3, 1)
        _run_dispatch(observer, 1, 3, 3)
    values = [event["progress"] for event in events if event["progress"] is not None]
    assert values == sorted(values)
    assert events[-1]["total_cv_fits"] == 12 and events[-1]["completed_cv_fits"] == 12


def test_nested_announcement_is_ignored_and_a_finished_plan_can_be_extended():
    with observe_progress(lambda _: None) as observer:
        observer.plan_dispatches(3)
        observer.plan_dispatches(3)  # a child batch inside the first dispatch: already counted
        assert observer.planned_dispatches == 3
        for _ in range(3):
            _run_dispatch(observer, 1, 2, 1)
        observer.plan_dispatches(2)  # a second run observed by the same scope
        assert observer.planned_dispatches == 5
        assert observer.reliable_total


def test_dispatch_without_countable_fits_is_dropped_from_the_plan():
    events = []
    with observe_progress(events.append) as observer:
        observer.plan_dispatches(3)
        observer.configure(1, "data")  # e.g. the native Methods lane: no bundles, no per-fit tasks
        _run_dispatch(observer, 1, 2, 1)
        _run_dispatch(observer, 1, 2, 1)
    assert events[-1]["total_cv_fits"] == events[-1]["completed_cv_fits"] == 4
    assert events[-1]["progress"] == 99.0


def test_bookkeeping_failures_degrade_to_unknown_progress_without_raising():
    events = []
    with observe_progress(events.append) as observer:
        observer.configure(1, "data")
        assert observer.open_bundle("not a dsl", None) == (1, {})
        assert not observer.reliable_total
        observer.task(1, {}, {"phase": "FIT_CV"}, {}, completed=True)
        observer.plan_dispatches("many")
        observer.emit("still alive")
    assert events[-1]["message"] == "still alive" and events[-1]["progress"] is None
    assert events[-1]["total_cv_fits"] is None


def test_open_bundle_failure_keeps_the_bundle_counter_consistent():
    with observe_progress(lambda _: None) as observer:
        first = observer.open_bundle({"split_invocation": None}, {"nodes": [None]})
        second = observer.open_bundle(*_bundle(2, 1))
    assert first == (1, {}) and second[0] == 2


def test_telemetry_failure_inside_a_real_run_does_not_change_scores(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 6))
    y = 2 * x[:, 0] - x[:, 2]
    options = {"engine": "dag-ml", "allow_fallback": False, "save_artifacts": False, "verbose": 0}
    pipeline = [KFold(3), PLSRegression(2)]
    baseline = nirs4all.run(pipeline, (x, y), workspace_path=tmp_path / "baseline", **options)

    def broken_total(self):
        raise RuntimeError("bookkeeping bug")

    monkeypatch.setattr(ExecutionProgress, "total_fits", property(broken_total))
    events = []
    with observe_progress(events.append):
        observed = nirs4all.run(pipeline, (x, y), workspace_path=tmp_path / "observed", **options)
    assert observed.cv_best_score == pytest.approx(baseline.cv_best_score, rel=1e-12)
    assert execution_progress() is None


def test_sequential_models_report_one_progress_scale_across_all_dispatches(tmp_path):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(36, 8))
    y = 3 * x[:, 0] + x[:, 1]
    pipeline = [KFold(3), {"model": PLSRegression(), "_range_": [1, 2, 1], "param": "n_components"}, Ridge(), PLSRegression(2)]
    options = {"engine": "dag-ml", "allow_fallback": False, "save_artifacts": False, "verbose": 0}
    events = []
    with observe_progress(events.append):
        nirs4all.run(pipeline, (x, y), workspace_path=tmp_path / "sequence", **options)
    finished = [event for event in events if "Fitting:" in event["message"] and "(completed)" in event["message"]]
    # PLS sweep: 2 variants x 3 folds; Ridge and PLS: 1 variant x 3 folds each.
    assert len(finished) == 12
    assert all(event["progress"] is not None for event in finished)
    assert finished[-1]["completed_cv_fits"] == finished[-1]["total_cv_fits"] == 12
    # The sweep is the first of three dispatches: its end is a third of the work, never 99 % of it.
    sweep_end = next(event for event in finished if event["completed_cv_fits"] == 6)
    assert sweep_end["progress"] < 50
    # Later dispatches replace the average-size estimate by their real size; the bar never goes backwards.
    values = [event["progress"] for event in events if event["progress"] is not None]
    assert values == sorted(values) and values[-1] == 99.0
    assert any("Model 3/3" in event["message"] for event in events)
