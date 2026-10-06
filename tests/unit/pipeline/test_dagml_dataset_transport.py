"""Live native data needs no CLI transport; subprocess runs retain exact identity."""

import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from nirs4all.data import SpectroDataset
from nirs4all.pipeline.dagml import cli_runner, dataset, in_process_runner, run_backend
from nirs4all.pipeline.dagml.identity import mint_identity


def _data():
    x = np.arange(24, dtype=float).reshape(12, 2)
    return x, x[:, 0] * 0.7 + x[:, 1] * 0.2


@pytest.mark.parametrize("file_backed", [False, True])
@pytest.mark.parametrize("cross_validate", [False, True])
def test_native_run_never_prepares_subprocess_dataset(monkeypatch, tmp_path, file_backed, cross_validate):
    monkeypatch.delenv("N4A_DAGML_INPROCESS", raising=False)
    x, y = _data()
    if file_backed:
        np.savetxt(tmp_path / "train_x.csv", x, delimiter=";")
        np.savetxt(tmp_path / "train_y.csv", y, delimiter=";")
        supplied = str(tmp_path)
    else:
        supplied = (x, y)

    def forbidden(*args, **kwargs):
        raise AssertionError("native execution must not reload or pickle a CLI dataset")

    monkeypatch.setattr(run_backend, "_dataset_inputs", forbidden)
    pipeline = ([KFold(3)] if cross_validate else []) + [Ridge(alpha=0.1)]
    result = run_backend.run_via_dagml(pipeline, supplied, save_charts=False, save_artifacts=False, verbose=0)
    assert result.execution_engine == "dag-ml"
    assert result.num_predictions > 0
    assert result.final
    assert dataset._DATASET_TRANSPORT_PREPARED.get() is True
    np.testing.assert_array_equal(x, _data()[0])
    np.testing.assert_array_equal(y, _data()[1])


class _StopDispatch(Exception):
    pass


@pytest.mark.parametrize("mode", ["explicit_cli", "unavailable_extension", "generated"])
def test_subprocess_preparation_retains_host_identity(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(run_backend, "preflight_dagml_backend", lambda *args, **kwargs: None)
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mode == "explicit_cli" else "1")
    monkeypatch.setattr(in_process_runner, "_dagml_extension_loads", lambda: mode != "unavailable_extension")
    supplied = SpectroDataset("transport")
    x, y = _data()
    supplied.add_samples(x, indexes={"partition": "train"})
    supplied.add_targets(y)
    if mode == "generated":
        supplied._generated_view_store = object()

    def stop(pipeline, spectro, base_dir, dataset_arg, host_pickle, *args, **kwargs):
        assert dataset._DATASET_TRANSPORT_PREPARED.get() is True
        assert host_pickle is not None
        with open(host_pickle, "rb") as stream:
            transported = pickle.load(stream)
        assert mint_identity(transported).fingerprint == mint_identity(spectro).fingerprint
        np.testing.assert_array_equal(transported.x({"partition": "train"}), spectro.x({"partition": "train"}))
        raise _StopDispatch

    monkeypatch.setattr(run_backend, "_dispatch_run", stop)
    with pytest.raises(_StopDispatch):
        run_backend.run_via_dagml([KFold(3), Ridge()], supplied, workdir=tmp_path, save_charts=False, verbose=0)
    assert dataset._DATASET_TRANSPORT_PREPARED.get() is True


@pytest.mark.parametrize("full_train", [False, True])
def test_late_cli_dispatch_refuses_before_writing_or_spawning(monkeypatch, tmp_path, full_train):
    monkeypatch.setattr(cli_runner, "named_model_input_spec", lambda value: None)
    kwargs = {"dsl": {}, "envelope": {}, "graph": {}, "dataset_path": "not-prepared", "workdir": tmp_path / "cli", "dagml_cli": "unused", "venv_python": "unused"}
    token = dataset._DATASET_TRANSPORT_PREPARED.set(False)
    try:
        if full_train:
            kwargs["training_sample_ids"] = []
        call = cli_runner.run_refit_phase_cli if full_train else cli_runner.run_cv_refit_bundle
        with pytest.raises(RuntimeError, match="changed to CLI"):
            call(**kwargs)
        assert not kwargs["workdir"].exists()
    finally:
        dataset._DATASET_TRANSPORT_PREPARED.reset(token)


def test_transport_context_is_nested_and_thread_local():
    outer = dataset._DATASET_TRANSPORT_PREPARED.set(False)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(dataset._DATASET_TRANSPORT_PREPARED.get).result() is True
        inner = dataset._DATASET_TRANSPORT_PREPARED.set(True)
        try:
            dataset._require_prepared_dataset_transport()
        finally:
            dataset._DATASET_TRANSPORT_PREPARED.reset(inner)
        with pytest.raises(RuntimeError, match="changed to CLI"):
            dataset._require_prepared_dataset_transport()
    finally:
        dataset._DATASET_TRANSPORT_PREPARED.reset(outer)


@pytest.mark.parametrize("late_change", ["mode", "extension"])
def test_by_source_late_cli_dispatch_refuses_before_writes(monkeypatch, tmp_path, late_change):
    from nirs4all.pipeline.dagml import attested_by_source

    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    x = np.arange(48, dtype=float).reshape(12, 4)
    supplied = SpectroDataset("by-source-transport")
    supplied.add_samples([x[:, :2], x[:, 2:]], indexes={"partition": "train"})
    supplied.add_targets(x[:, 0] * 0.7 + x[:, 1] * 0.2)

    class ModeFlipKFold(KFold):
        def split(self, *args, **kwargs):
            monkeypatch.setenv("N4A_DAGML_INPROCESS", "0")
            yield from super().split(*args, **kwargs)

    if late_change == "extension":
        checks = iter([True, True, False])
        monkeypatch.setattr(in_process_runner, "_dagml_extension_loads", lambda: next(checks))
    splitter = ModeFlipKFold(3) if late_change == "mode" else KFold(3)
    pipeline = [splitter, {"branch": {"by_source": True, "steps": {f"source_{i}": [{"model": Ridge()}] for i in range(2)}}}, {"merge": "auto"}]

    def forbidden(*args, **kwargs):
        raise AssertionError("unprepared by_source dataset must not reach subprocess launch")

    monkeypatch.setattr(attested_by_source.subprocess, "run", forbidden)
    with pytest.raises(RuntimeError, match="changed to CLI"):
        run_backend.run_via_dagml(pipeline, supplied, workdir=tmp_path, dagml_cli=str(run_backend._default_dagml_cli()), save_charts=False, save_artifacts=False, verbose=0)
    assert not list(tmp_path.rglob("request.json"))
    assert not list(tmp_path.rglob("n4a_adapter"))
    assert dataset._DATASET_TRANSPORT_PREPARED.get() is True
