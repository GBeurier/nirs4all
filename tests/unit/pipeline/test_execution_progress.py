"""Progress events describe actual CV tasks and preserve scientific results."""
import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.pipeline.dagml.progress import execution_progress, observe_progress


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


def test_repeated_dispatch_resets_local_counters_and_unknown_global_total():
    events = []
    with observe_progress(events.append) as observer:
        observer.configure(3, "first")
        observer.bundle = 1
        observer.completed.add((1, "model", "base", "fold"))
        observer.variants[(1, "base")] = 1
        observer.active_models[(1, "base")] = "model"
        observer.total_folds = 3
        observer.configure(2, "second")
        assert not observer.completed and not observer.variants and not observer.active_models
        assert observer.total_folds == 0
        assert events[-1]["progress"] is None
        assert events[-1]["completed_cv_fits"] == 0


def test_progress_callback_failure_does_not_interrupt_scientific_execution():
    def broken_callback(_):
        raise OSError("Closed telemetry stream")
    with observe_progress(broken_callback) as observer:
        observer.configure(2, "Dataset \U0001f600" * 300)
        observer.emit("Refitting selected model")
    assert execution_progress() is None
