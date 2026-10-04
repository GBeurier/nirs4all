"""Fixed raw IO tensors reach native joint training and prediction-only replay."""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import dag_ml
import numpy as np
import pytest
from nirs4all_io import MultimodalDataset, TensorSource
from sklearn.model_selection import GroupKFold

import nirs4all
from nirs4all.controllers.models.torch_model import PyTorchModelController
from nirs4all.operators.models.multimodal import MultimodalRegressor
from nirs4all.pipeline.dagml.named_torch_estimator import DagMLNamedTorchEstimator
from nirs4all.pipeline.dagml.torch_estimator import DagMLTorchEstimator
from tests.fixtures import named_torch_nd
from tests.integration.api.test_named_torch_training import _Observer

pytestmark = pytest.mark.torch
_SHAPES = {"tabular_numeric": (3,), "signal_1d": (5,), "gray_image": (2, 3), "rgb_image": (2, 3, 3),
           "mc_image": (2, 3, 2), "multispectral_image": (2, 3, 4), "series_mv": (5, 2)}
_TYPES = {"tabular_numeric": "table", "signal_1d": "dense_signal", "gray_image": "gray_image", "rgb_image": "image_rgb",
          "mc_image": "multichannel_image", "multispectral_image": "multichannel_image", "series_mv": "time_series"}


def _cohort(representation: str, *, prediction: bool = False, four: bool = False) -> MultimodalDataset:
    rng = np.random.default_rng(804 if prediction else 803)
    rows = 5 if prediction else 16
    ids = [f"{'new' if prediction else 'row'}_{index:02d}" for index in range(rows)]
    declarations = {"nir": "signal_1d", "sensor": representation}
    if four:
        declarations.update(series="series_mv", clinical="tabular_numeric")
    sources = {}
    target = np.zeros(rows, dtype=np.float32)
    for index, (name, kind) in enumerate(declarations.items()):
        dtype = np.float64 if name == "sensor" else np.float32
        values = rng.normal(size=(rows, *_SHAPES[kind])).astype(dtype)
        target += (index + 1) * values.reshape(rows, -1)[:, 0].astype(np.float32)
        order = rng.permutation(rows)
        metadata: dict[str, Any] = {}
        if kind == "signal_1d":
            metadata = {"axis_units": {"wavelength": "nm"}, "axis_coordinates": {"wavelength": np.arange(_SHAPES[kind][0]) + 1100}}
        elif kind == "series_mv":
            metadata = {"axis_units": {"time": "s"}, "axis_coordinates": {"time": np.arange(_SHAPES[kind][0]) / 4}}
        sources[name] = TensorSource(values[order], [ids[row] for row in order], representation_id=kind, **metadata)
    return MultimodalDataset(sources, sample_ids=ids, y=None if prediction else target,
        groups=None if prediction else [f"subject_{index // 2}" for index in range(rows)],
        partitions=["predict"] * rows if prediction else ["train"] * 12 + ["test"] * 4,
        target_names=["concentration"], task_type="regression", name="native_tensor_prediction" if prediction else "native_tensor_training")


def _run(cohort: Any, root: Path, *, cv: bool = True) -> Any:
    torch = DagMLTorchEstimator(factory_path="tests.fixtures.named_torch_nd.joint_factory", factory_params={"hidden_units": 3},
        device="cpu", task_type="regression", epochs=2, batch_size=4, patience=2, lr=0.01)
    model = MultimodalRegressor(dict.fromkeys(cohort.sources, "passthrough"), model=torch, fusion="intermediate")
    pipeline = [GroupKFold(3), {"model": model}] if cv else [{"model": model}]
    return nirs4all.run(pipeline, cohort, engine="dag-ml", refit=True, random_state=31, cpu_threads=1, gpu_devices=[],
                        save_artifacts=True, save_charts=False, workspace_path=root, verbose=0)


@pytest.fixture(autouse=True)
def native_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.delenv("N4A_NATIVE_RESULTS", raising=False)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("legacy scheduler executed"))
    monkeypatch.setattr(dag_ml, "run_host_hpo_search_in_process", lambda *a, **k: pytest.fail("unexpected HPO"))
    # Reuse observation of real native tasks/receipts, replacing only its
    # independent numerical oracle with one that understands actual raw axes.
    observer_module = importlib.import_module("tests.integration.api.test_named_torch_training")
    monkeypatch.setattr(observer_module.oracle, "fit_reference", named_torch_nd.fit_reference)
    monkeypatch.setattr(observer_module.oracle, "predict_reference", named_torch_nd.predict_reference)


@pytest.mark.parametrize("representation", list(_SHAPES))
@pytest.mark.parametrize("cv", [False, True])
def test_actual_fixed_io_shapes_train_jointly_and_replay_without_fit(
    representation: str, cv: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    training = _cohort(representation)
    observer = _Observer(monkeypatch, training)
    result = _run(training, tmp_path / "workspace", cv=cv)
    try:
        fits = [record for record in observer.records if record["task"]["phase"] in {"FIT_CV", "REFIT"}]
        assert len(fits) == (4 if cv else 1)
        if cv:
            test_ids = [sample for sample, partition in zip(training.sample_ids, training.partitions, strict=True) if partition == "test"]
            assert test_ids
            for record in fits:
                if record["task"]["phase"] != "FIT_CV":
                    continue
                task, response = record["task"], record["response"]
                assert set(record["fit_ids"]).isdisjoint(test_ids)
                test_predictions = [block for block in response["predictions"] if block["partition"] == "test"]
                assert len(test_predictions) == 1 and test_predictions[0]["sample_ids"] == test_ids
                for name in training.sources:
                    key = f"data:{name}:test"
                    view = task["data_views"][key]
                    assert view["partition"] == "predict" and view["sample_ids"] == test_ids
                    assert view["include_augmented"] is False
                    assert task["data_view_receipts"][key]["sample_ids"] == test_ids
                    evidence = response["consumed_data_views"][key]
                    assert evidence["receipt"] == task["data_view_receipts"][key]
                    assert evidence["read_batches"] and all(batch == test_ids for batch in evidence["read_batches"])
                    assert evidence["model_calls"] and all(call["operation"] == "predict" for call in evidence["model_calls"])
                    assert all(call["sample_ids"] == test_ids and "target_fingerprint" not in call for call in evidence["model_calls"])
                    fit_evidence = response["consumed_data_views"][f"data:{name}"]
                    fit_calls = [call for call in fit_evidence["model_calls"] if call["operation"] == "fit"]
                    assert len(fit_calls) == 1 and fit_calls[0]["sample_ids"] == record["fit_ids"]
        refit = next(record for record in fits if record["task"]["phase"] == "REFIT")
        origin = result._dagml_refit_artifacts[0]["named_refit_origin"]
        port = next(port for port in origin["model_input"]["ports"] if port["name"] == "sensor")
        assert port["accepted_types"] == [_TYPES[representation]]
        assert port["accepted_representations"] == [representation]
        assert port["rank"] == len(_SHAPES[representation]) + 1
        assert port["metadata"]["feature_shape"] == list(_SHAPES[representation])
        for record in fits:
            module = record["estimator"].model_
            assert module.seen_shapes and all(shapes["sensor"] == _SHAPES[representation] for shapes in module.seen_shapes)
        prediction = _cohort(representation, prediction=True)
        expected = named_torch_nd.predict_reference(refit["reference"], {name: source.values for name, source in prediction.sources.items()})
        archive = result.export(tmp_path / "fixed-tensors.n4a")
        monkeypatch.setattr(DagMLNamedTorchEstimator, "fit", lambda *a, **k: pytest.fail("replay fitted a model"))
        monkeypatch.setattr(PyTorchModelController, "_train_model", lambda *a, **k: pytest.fail("replay trained weights"))
        tasks = observer.observe_prediction(monkeypatch)
        actual = nirs4all.predict(archive, prediction, engine="dag-ml", verbose=0)
        assert len(tasks) == 1 and tasks[0]["phase"] == "PREDICT"
        np.testing.assert_allclose(actual.values.reshape(-1, 1), expected, rtol=2e-6, atol=2e-6)
    finally:
        result.close()


@pytest.mark.parametrize("change", ["shape", "dtype", "units", "coordinates"])
def test_raw_tensor_schema_changes_are_refused_before_numeric_prediction(change: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = _run(_cohort("rgb_image", four=True), tmp_path / "workspace", cv=False)
    try:
        archive = result.export(tmp_path / "original.n4a")
    finally:
        result.close()
    prediction = _cohort("rgb_image", prediction=True, four=True)
    sources = dict(prediction.sources)
    source = sources["sensor" if change in {"shape", "dtype"} else "series"]
    if change == "shape":
        sources["sensor"] = TensorSource(source.values.transpose(0, 2, 1, 3), source.sample_ids, representation_id="rgb_image")
    elif change == "dtype":
        sources["sensor"] = TensorSource(source.values.astype(np.float32), source.sample_ids, representation_id="rgb_image")
    else:
        sources["series"] = TensorSource(source.values, source.sample_ids, representation_id="series_mv",
            axis_units={"time": "ms" if change == "units" else "s"},
            axis_coordinates={"time": np.arange(5) / (5 if change == "coordinates" else 4)})
    changed = MultimodalDataset(sources, sample_ids=prediction.sample_ids, y=None, partitions=prediction.partitions,
                               target_names=prediction.target_names, task_type="regression")
    monkeypatch.setattr(DagMLNamedTorchEstimator, "predict", lambda *a, **k: pytest.fail("changed IO schema reached numeric forward"))
    with pytest.raises(ValueError, match="(?i)(schema|contract)"):
        nirs4all.predict(archive, changed, engine="dag-ml", verbose=0)


def test_four_actual_modalities_replay_in_a_fresh_installed_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    python = os.environ.get("NIRS4ALL_NAMED_TORCH_INSTALLED_PYTHON")
    if not python:
        if os.environ.get("NIRS4ALL_REQUIRE_NAMED_TORCH_INSTALLED") == "1":
            pytest.fail("NIRS4ALL_NAMED_TORCH_INSTALLED_PYTHON is mandatory")
        pytest.skip("fresh installed interpreter is mandatory during qualification")
    training = _cohort("rgb_image", four=True)
    observer = _Observer(monkeypatch, training)
    result = _run(training, tmp_path / "workspace")
    try:
        archive = result.export(tmp_path / "joint-native-shapes.n4a")
        prediction = _cohort("rgb_image", prediction=True, four=True)
        refit = next(record for record in observer.records if record["task"]["phase"] == "REFIT")
        expected = named_torch_nd.predict_reference(refit["reference"], {name: source.values for name, source in prediction.sources.items()})
        inputs = tmp_path / "inputs.json"
        inputs.write_text(json.dumps(prediction.to_dict()))
        clean = tmp_path / "clean"
        clean.mkdir()
        script = textwrap.dedent("""\
            import importlib,json,pathlib,sys,types
            import nirs4all,dag_ml
            from nirs4all_io import MultimodalDataset
            prefix=pathlib.Path(sys.prefix).resolve()
            for name in ['nirs4all','dag_ml','dag_ml._dag_ml','nirs4all_io',
                         'nirs4all.pipeline.dagml.named_torch_estimator',
                         'nirs4all.pipeline.dagml.general_replay','nirs4all.pipeline.dagml.fixed_cohort_views']:
                assert pathlib.Path(importlib.import_module(name).__file__).resolve().is_relative_to(prefix),name
            resource=pathlib.Path(sys.argv[3])/'tests'
            for name,folder in [('tests',resource),('tests.fixtures',resource/'fixtures')]:
                package=types.ModuleType(name);package.__path__=[str(folder)];sys.modules[name]=package
            import tests.fixtures.named_torch_nd
            from nirs4all.pipeline.dagml.named_torch_estimator import DagMLNamedTorchEstimator
            from nirs4all.controllers.models.torch_model import PyTorchModelController
            from nirs4all.pipeline.dagml.fixed_cohort_views import FixedCohortViewStore
            from nirs4all.pipeline import PipelineRunner
            def forbidden(*a,**k):raise AssertionError('FIT/HPO entered during archive prediction')
            DagMLNamedTorchEstimator.fit=forbidden;PyTorchModelController._train_model=forbidden
            PipelineRunner.run=forbidden;dag_ml.run_host_hpo_search_in_process=forbidden
            original=FixedCohortViewStore.__call__;receipts=[]
            def materialize(store,call):
                assert call['request']['phase']=='PREDICT'
                receipt=original(store,call);receipts.append(receipt);return receipt
            FixedCohortViewStore.__call__=materialize
            cohort=MultimodalDataset.from_dict(json.loads(pathlib.Path(sys.argv[2]).read_text()))
            result=nirs4all.predict(sys.argv[1],cohort,engine='dag-ml',verbose=0)
            assert len(receipts)==4
            assert all(receipt['sample_ids']==list(cohort.sample_ids) for receipt in receipts)
            print(json.dumps({'values':result.values.reshape(-1,1).tolist(),'receipts':len(receipts)}))
        """)
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["N4A_DAGML_INPROCESS"] = "1"
        child = subprocess.run([python, "-I", "-B", "-c", script, str(archive), str(inputs), str(Path(__file__).parents[3])],
                               cwd=clean, env=environment, capture_output=True, text=True, timeout=180, check=True)
        report = json.loads(child.stdout.strip().splitlines()[-1])
        assert report["receipts"] == 4
        np.testing.assert_allclose(report["values"], expected, rtol=2e-6, atol=2e-6)
    finally:
        result.close()
