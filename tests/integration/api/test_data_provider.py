"""Real native provider preparation, grouped training and independent replay."""

from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nirs4all_io import DataProvider, MultimodalDataset
from sklearn.model_selection import GroupKFold

import nirs4all
from nirs4all.data.multimodal import MultimodalSpectroDataset
from nirs4all.operators.models.multimodal import TensorPCA
from nirs4all.pipeline.dagml.cancellation import DagRunCancelled
from nirs4all.pipeline.dagml.multimodal_tuning import MultimodalTuningStopped
from tests.integration.api.test_multimodal_dagml import _cohort, _model
from tests.integration.api.test_multimodal_tuning import _checkpoint, _tuning


def _generate(*, seed: int, params: dict[str, Any], context: dict[str, Any]) -> MultimodalDataset:
    cohort = _cohort(unequal_groups=True)
    return MultimodalDataset(
        cohort.sources, sample_ids=cohort.sample_ids,
        y=cohort.y + np.random.default_rng(seed).normal(scale=params.get("noise", 0.01), size=len(cohort.sample_ids)),
        groups=cohort.groups, partitions=cohort.partitions, name=cohort.name,
    )


def _provider(**kwargs: Any) -> DataProvider:
    return DataProvider(_generate, provider_id="qualification.synthetic", seed=19, **kwargs)


def _run(provider: DataProvider, directory: Path, **kwargs: Any) -> Any:
    return nirs4all.run(
        [GroupKFold(3), _model()], provider, workspace_path=directory,
        random_state=19, verbose=0, save_charts=False, **kwargs,
    )


@pytest.fixture(autouse=True)
def no_legacy_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("Legacy scheduler executed"))


def test_native_source_executes_once_before_grouped_fits_and_survives_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def generate(**kwargs: Any) -> MultimodalDataset:
        calls.append(kwargs["seed"])
        return _generate(**kwargs)

    provider = DataProvider(generate, provider_id="qualification.synthetic", seed=19)
    fits: list[frozenset[int]] = []
    original_fit = TensorPCA.fit

    def observe(self: TensorPCA, X: Any, y: Any = None) -> Any:
        assert len(calls) == 1
        values = np.asarray(X)
        fits.append(frozenset(int(value) for value in values.reshape(len(values), -1)[:, 0]))
        return original_fit(self, X, y)

    monkeypatch.setattr(TensorPCA, "fit", observe)
    result = _run(provider, tmp_path / "workspace")
    try:
        assert len(calls) == 1
        cohort = provider.cohort
        groups = np.asarray(cohort.groups)[:12]
        expected = [frozenset(train) for train, _ in GroupKFold(3).split(np.zeros((12, 1)), groups=groups)] + [frozenset(range(12))]
        assert sorted(fits, key=lambda ids: (len(ids), sorted(ids))) == sorted(expected * 2, key=lambda ids: (len(ids), sorted(ids)))
        evidence = next(iter(result.per_dataset.values()))["data_provider_evidence"]
        assert evidence["execution"]["task_seed"] == calls[0]
        assert evidence["execution"]["metadata"]["content_fingerprint"] == provider.fingerprint
        wrapped = MultimodalSpectroDataset(cohort)
        assert pickle.loads(pickle.dumps(wrapped))._data_provider_evidence == evidence
        archive = result.export(tmp_path / "provider.n4a")
        with zipfile.ZipFile(archive) as bundle:
            manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["multimodal_host"]["data_provider"] == evidence
        monkeypatch.setattr(DataProvider, "materialize", lambda *a, **k: pytest.fail("Replay regenerated training data"))
        monkeypatch.setattr(TensorPCA, "fit", lambda *a, **k: pytest.fail("Replay fitted an encoder"))
        prediction = nirs4all.predict(archive, _cohort(prediction=True))
        assert len(prediction.values) == 5
        assert len(calls) == 1
    finally:
        result.close()


def test_partial_provider_adds_raw_modalities_to_fixed_base(tmp_path: Path) -> None:
    complete = _cohort()
    base = MultimodalDataset(
        {"nir": complete.sources["nir"]}, sample_ids=complete.sample_ids,
        y=complete.y, groups=complete.groups, partitions=complete.partitions, name=complete.name,
    )

    def generate(**kwargs: Any) -> dict[str, Any]:
        return {"sample_ids": complete.sample_ids, "sources": {key: value for key, value in complete.sources.items() if key != "nir"}}

    provider = DataProvider(generate, provider_id="qualification.partial", base=base)
    result = _run(provider, tmp_path)
    try:
        assert np.isfinite(result.best_rmse)
        assert set(provider.cohort.sources) == {"nir", "image", "series", "metadata"}
        np.testing.assert_array_equal(provider.cohort.sources["nir"].values, base.sources["nir"].values)
        np.testing.assert_array_equal(provider.cohort.y, base.y)
    finally:
        result.close()


@pytest.mark.parametrize("engine", ["native", "legacy", "dual"])
def test_unsupported_engine_refuses_before_provider_execution(engine: str, tmp_path: Path) -> None:
    def forbidden(**kwargs: Any) -> Any:
        pytest.fail("Unsupported engine executed the provider")

    with pytest.raises(ValueError, match="DataProvider requires"):
        _run(DataProvider(forbidden, provider_id="qualification.refused"), tmp_path, engine=engine)


def test_cancelled_run_refuses_before_provider_execution(tmp_path: Path) -> None:
    def forbidden(**kwargs: Any) -> Any:
        pytest.fail("Cancelled run executed the provider")

    with pytest.raises(DagRunCancelled, match="cancelled"):
        _run(DataProvider(forbidden, provider_id="qualification.cancelled"), tmp_path, should_stop=lambda: True)


def test_cancellation_during_generation_retains_public_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stop = False

    def generate(**kwargs: Any) -> MultimodalDataset:
        nonlocal stop
        stop = True
        return _generate(**kwargs)

    monkeypatch.setattr(TensorPCA, "fit", lambda *a, **k: pytest.fail("Cancelled provider reached training"))
    with pytest.raises(DagRunCancelled, match="cancelled"):
        _run(DataProvider(generate, provider_id="qualification.cancelled"), tmp_path, should_stop=lambda: stop)


def test_provider_hpo_resume_matches_continuous_and_binds_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "study"
    tuning = {**_tuning(directory), "n_trials": 2}
    with pytest.raises(MultimodalTuningStopped):
        _run(_provider(), tmp_path / "stopped", tuning={**tuning, "progress_callback": lambda event: len(event["checkpoint"]["trials"]) < 1})
    prefix = _checkpoint(directory)["native_checkpoint"]["trials"]
    resumed = _run(_provider(), tmp_path / "resumed", tuning={**tuning, "resume": True})
    continuous = _run(_provider(), tmp_path / "continuous", tuning={**_tuning(tmp_path / "other-study"), "n_trials": 2})
    try:
        assert [trial.to_dict() for trial in resumed.tuning_result.trials] == [trial.to_dict() for trial in continuous.tuning_result.trials]
        assert _checkpoint(directory)["native_checkpoint"]["trials"][:1] == prefix
        assert resumed.tuning_best_params == continuous.tuning_best_params
        resumed_archive, continuous_archive = resumed.export(tmp_path / "resumed.n4a"), continuous.export(tmp_path / "continuous.n4a")
        np.testing.assert_allclose(nirs4all.predict(resumed_archive, _cohort(prediction=True)).values,
                                   nirs4all.predict(continuous_archive, _cohort(prediction=True)).values, rtol=0, atol=0)
        before = (directory / "multimodal.n4mopt.json").read_bytes()
        monkeypatch.setattr(TensorPCA, "fit", lambda *a, **k: pytest.fail("Changed provider recipe reached fit"))
        with pytest.raises(Exception, match="checkpoint .* binding mismatch"):
            _run(_provider(provider_version="2"), tmp_path / "changed", tuning={**tuning, "resume": True})
        assert (directory / "multimodal.n4mopt.json").read_bytes() == before
    finally:
        resumed.close()
        continuous.close()


def test_provider_lineage_is_retained_by_late_fusion_export(tmp_path: Path) -> None:
    from tests.integration.api.test_multimodal_late_fusion import _pipeline

    result = nirs4all.run(_pipeline(), _provider(), workspace_path=tmp_path / "workspace",
                         random_state=19, verbose=0, save_charts=False)
    try:
        ensemble = next(view for view in result.runs if any(item.get("producer_node") == "merge:stack" for item in view.per_dataset.values()))
        archive = ensemble.export(tmp_path / "late.n4a")
        with zipfile.ZipFile(archive) as bundle:
            manifest = json.loads(bundle.read("manifest.json"))
        contract = manifest["multimodal_host"]
        assert contract["selected_model"]["fusion"] == "late_oof"
        assert contract["data_provider"]["recipe"]["provider_id"] == "qualification.synthetic"
        assert len(nirs4all.predict(archive, _cohort(prediction=True)).values) == 5
    finally:
        result.close()


def test_should_stop_ends_search_at_saved_trial_boundary(tmp_path: Path) -> None:
    stop = False
    directory = tmp_path / "study"

    def progress(event: dict[str, Any]) -> bool:
        nonlocal stop
        stop = len(event["checkpoint"]["trials"]) >= 1
        return True

    tuning = {**_tuning(directory), "n_trials": 2}
    with pytest.raises(DagRunCancelled, match="checkpoint saved"):
        _run(_provider(), tmp_path / "stopped", should_stop=lambda: stop,
             tuning={**tuning, "progress_callback": progress})
    saved = _checkpoint(directory)["native_checkpoint"]["trials"]
    assert len(saved) == 1 and saved[0]["state"] == "complete"
    resumed = _run(_provider(), tmp_path / "resumed", tuning={**tuning, "resume": True})
    try:
        assert len(resumed.tuning_result.trials) == 2
        assert _checkpoint(directory)["native_checkpoint"]["trials"][:1] == saved
    finally:
        resumed.close()
