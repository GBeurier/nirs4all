"""Library child results cannot finish a Studio-owned parent run."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def test_existing_parent_run_retains_external_lifecycle(tmp_path):
    import nirs4all
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    with WorkspaceStore(tmp_path) as store:
        parent = store.begin_run(name="Studio multi-pipeline", config={}, datasets=[])
    X = np.arange(60, dtype=float).reshape(20, 3)
    for alpha in (0.5, 1.0):
        result = nirs4all.run(
            [KFold(n_splits=2), Ridge(alpha=alpha)], (X, X[:, 0]),
            workspace_path=tmp_path, store_run_id=parent, save_artifacts=False,
        )
        assert {item["run_id"] for item in result.per_dataset.values()} == {parent}
        with WorkspaceStore(tmp_path) as store:
            assert store.get_run(parent)["status"] == "running"
    with WorkspaceStore(tmp_path) as store:
        assert store.query_predictions().height > result.num_predictions
        store.complete_run(parent, {"owner": "studio"})
        assert store.get_run(parent)["status"] == "completed"


def test_bad_parent_is_rejected_before_operator_fit(tmp_path, monkeypatch):
    import nirs4all

    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("invalid parent reached training"))
    X = np.arange(60, dtype=float).reshape(20, 3)
    with pytest.raises(ValueError, match="existing running run"):
        nirs4all.run([KFold(n_splits=2), Ridge()], (X, X[:, 0]), workspace_path=tmp_path, store_run_id="absent")


def test_cancellation_stops_before_next_native_fit_and_preserves_parent(tmp_path, monkeypatch):
    import nirs4all
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    with WorkspaceStore(tmp_path) as store:
        parent = store.begin_run(name="cancelled child", config={}, datasets=[])
    fitted = []
    original_fit = Ridge.fit

    def fit(model, *args, **kwargs):
        fitted.append(True)
        return original_fit(model, *args, **kwargs)

    monkeypatch.setattr(Ridge, "fit", fit)
    X = np.arange(60, dtype=float).reshape(20, 3)
    with pytest.raises(Exception, match="cancelled"):
        nirs4all.run(
            [KFold(n_splits=3), Ridge()], (X, X[:, 0]), workspace_path=tmp_path,
            store_run_id=parent, should_stop=lambda: bool(fitted),
        )
    assert len(fitted) == 1
    with WorkspaceStore(tmp_path) as store:
        assert store.get_run(parent)["status"] == "running"
        assert store.query_predictions().height == 0
    fitted.clear()
    nirs4all.run([KFold(n_splits=2), Ridge()], (X, X[:, 0]), save_artifacts=False)
    assert len(fitted) == 3  # cancellation context was reset after the failed invocation
