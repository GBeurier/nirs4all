"""General API routing is decided before execution, never by runtime fallback."""

import importlib

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from nirs4all.api.run_selection import select_run_engine


@pytest.fixture(autouse=True)
def clear_selector(monkeypatch):
    monkeypatch.delenv("N4A_ENGINE", raising=False)


def portable_request():
    return [KFold(3), {"model": PLSRegression(2)}], {"X": np.ones((12, 4)), "y": np.arange(12), "sample_ids": [str(i) for i in range(12)]}


def test_portable_selection_does_not_execute_splitter(monkeypatch):
    pipeline, data = portable_request()
    monkeypatch.setattr(pipeline[0], "split", lambda *args, **kwargs: pytest.fail("preflight executed a splitter"))
    assert select_run_engine(None, pipeline, data) == "native"


@pytest.mark.parametrize("dataset", ["data/folder", (np.ones((12, 4)), np.arange(12)), {"X": np.ones((12, 4)), "y": np.arange(12)}])
def test_general_dataset_forms_select_dagml(dataset):
    pipeline, _ = portable_request()
    assert select_run_engine(None, pipeline, dataset) == "dag-ml"


def test_general_preprocessing_selects_dagml():
    pipeline, data = portable_request()
    assert select_run_engine(None, [MinMaxScaler(), *pipeline], data) == "dag-ml"


@pytest.mark.parametrize("option", [{"save_charts": True}, {"cache": {}}, {"project": "project"}, {"refit": False}, {"tuning": {}}, {"runner_kwargs": {"n_jobs": 2}}])
def test_nonportable_options_select_general_dag_profile(option):
    assert select_run_engine(None, *portable_request(), **option) == "dag-ml"


@pytest.mark.parametrize("engine", ["native", "dag-ml", "legacy", "dual"])
def test_explicit_engine_never_changes(engine):
    assert select_run_engine(engine, [], "data/folder") == engine


def test_environment_selector_is_respected(monkeypatch):
    monkeypatch.setenv("N4A_ENGINE", "native")
    assert select_run_engine(None, [], "data/folder") == "native"


def test_session_workspace_configuration_participates_without_creating_runner(tmp_path):
    from nirs4all.api.session import Session

    with Session(workspace_path=tmp_path, verbose=0) as session:
        assert select_run_engine(None, *portable_request(), session=session) == "dag-ml"
        assert session._runner is None


@pytest.mark.parametrize("owner", ["native", "dag-ml"])
def test_existing_session_owner_is_not_changed_by_request_shape(owner):
    from unittest.mock import Mock

    session = Mock(execution_engine=owner)
    assert select_run_engine(None, [], "general-data", session=session) == owner
    assert select_run_engine("legacy", [], "general-data", session=session) == "legacy"


def test_session_run_selects_general_profile_before_delegating(monkeypatch):
    from nirs4all.api.session import Session

    calls = []
    run_module = importlib.import_module("nirs4all.api.run")
    monkeypatch.setattr(run_module, "run", lambda *args, **kwargs: calls.append(kwargs))
    with Session([MinMaxScaler(), *portable_request()[0]], verbose=0) as session:
        session.run(portable_request()[1])
    assert calls[0]["engine"] == "dag-ml"
    assert "save_charts" not in calls[0]


def test_closed_session_fails_during_selection():
    from nirs4all.api.session import Session, SessionClosedError

    session = Session()
    session.close()
    with pytest.raises(SessionClosedError):
        select_run_engine(None, *portable_request(), session=session)


def test_portable_runtime_failure_does_not_retry_on_general_or_legacy(monkeypatch):
    import nirs4all

    native = importlib.import_module("nirs4all.api.native_archive_training")
    backend = importlib.import_module("nirs4all.pipeline.dagml.run_backend")
    calls = []

    def fail(*args, **kwargs):
        calls.append("native")
        raise RuntimeError("native witness failure")

    monkeypatch.setattr(native, "run_native_methods_archive", fail)
    monkeypatch.setattr(backend, "run_via_dagml", lambda *args, **kwargs: pytest.fail("runtime fallback"))
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("legacy fallback"))
    with pytest.raises(RuntimeError, match="native witness failure"):
        nirs4all.run(*portable_request())
    assert calls == ["native"]
