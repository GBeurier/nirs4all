"""SDK mixed metadata preserves categories with the public Methods 1.3.2 binding."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

TAG_HELPER_SHA256 = "bee669e75773c568171ef5ced6c8575a4fead0265b4426856ceba3cb4251ef38"


def _require_public_binding():
    import n4m
    import n4m.roles._multimodal as binding

    assert hashlib.sha256(Path(binding.__file__).read_bytes()).hexdigest() == TAG_HELPER_SHA256
    assert n4m.version() == "1.3.2+abi.2.17.0"
    assert n4m.abi_version() == (2, 17, 0)


def _case(classification, selected):
    from n4m.roles import PLSLogistic
    from nirs4all_io import MultimodalDataset, TensorSource

    from nirs4all.operators.models.multimodal import MultimodalClassifier

    path = Path(__file__).resolve().parents[3] / "examples/user/02_data_handling/U07_multimodal.py"
    spec = importlib.util.spec_from_file_location("mixed_metadata_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    def cohort(prediction):
        original = example.make_cohort(29 if prediction else 17, prediction=prediction)
        values = original.sources["metadata"].values.astype(object)
        values[:, 1] = np.resize(["A" * 32, "é", "猫"], len(values))
        if prediction:
            values[0, 1] = "A" * 32 + "unseen猫"
        values = values.astype(str)  # Different widths remain genuine raw inputs.
        old = original.sources["metadata"]
        sources = dict(original.sources)
        sources["metadata"] = TensorSource(values, original.sample_ids, representation_id="tabular_mixed", feature_names=old.feature_names)
        y = None if prediction else np.where(np.asarray(original.y) > np.median(original.y), "classe猫", "classeé") if classification else original.y
        return MultimodalDataset(sources, sample_ids=original.sample_ids, y=y, groups=original.groups,
                                 partitions=original.partitions, name=original.name)

    training, prediction = cohort(False), cohort(True)
    model = example.make_pipeline(backend="methods")[-1]["model"]
    if selected:
        model.transformers = {name: model.transformers[name] for name in ("nir", "metadata")}
    if classification:
        model = MultimodalClassifier(transformers=model.transformers, model=PLSLogistic(n_components=2, max_iter=100), backend="methods")
    return training, prediction, model


def _oracle(model, training, prediction, rows, classification):
    encoded_train, encoded_predict = [], []
    for name, transform in model.transformers.items():
        fitted = clone(transform).fit(training.sources[name].values[rows], training.y[rows])
        encoded_train.append(fitted.transform(training.sources[name].values[rows]))
        encoded_predict.append(fitted.transform(prediction.sources[name].values))
    train_matrix, predict_matrix = np.hstack(encoded_train), np.hstack(encoded_predict)
    targets = np.asarray(training.y)[rows]
    if classification:
        classes = np.unique(targets)
        reference = clone(model.model).fit(train_matrix, np.searchsorted(classes, targets))
        expected = classes[np.asarray(reference.predict(predict_matrix), dtype=int)]
        probabilities = reference.predict_proba(predict_matrix)
    else:
        reference = clone(model.model).fit(train_matrix, targets)
        expected, probabilities = reference.predict(predict_matrix), None
    return expected, predict_matrix, probabilities


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_direct_native_fit_predict_and_transform_preserve_unknown_text(classification, selected):
    from nirs4all.pipeline.dagml.methods_multimodal import methods_input_blocks, source_schemas_from_cohort

    _require_public_binding()
    training, prediction, model = _case(classification, selected)
    rows = np.flatnonzero(np.asarray(training.partitions) == "train")
    schemas = {name: source_schemas_from_cohort(training)[name] for name in model.transformers}
    saved = deepcopy(schemas)
    raw = {name: training.sources[name].values[rows] for name in model.transformers}
    new = {name: prediction.sources[name].values for name in model.transformers}
    expected, encoded, probabilities = _oracle(model, training, prediction, rows, classification)
    assert raw["metadata"].dtype.kind == new["metadata"].dtype.kind == "U"
    assert raw["metadata"].dtype != new["metadata"].dtype
    try:
        model.fit(list(raw.values()), training.y[rows], source_schemas=schemas)
        state = model.native_pipeline_.export_state()
        actual = model.predict(list(new.values()))
        if classification:
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_allclose(model.predict_proba(list(new.values())), probabilities, atol=2e-7, rtol=2e-7)
            assert np.isfinite(model.decision_function(list(new.values()))).all()
        else:
            np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)
        transformed = model.native_pipeline_.transform(methods_input_blocks(new, schemas))
        np.testing.assert_allclose(transformed[:, -4:], encoded[:, -4:], atol=2e-8, rtol=2e-8)
        np.testing.assert_array_equal(transformed[0, -3:], np.zeros(3))
        assert new["metadata"][0, 1] == "A" * 32 + "unseen猫"
        assert schemas == saved == model.source_schemas_
        assert model.native_pipeline_.export_state() == state
        invalid = dict(new)
        invalid["metadata"] = new["metadata"].astype(object)
        invalid["metadata"][0, 1] = 7
        with pytest.raises(TypeError, match="categorical cells"):
            model.predict(list(invalid.values()))
        with pytest.raises(TypeError, match="categorical cells"):
            model.fit(list(invalid.values()), np.resize(training.y[rows], len(invalid["metadata"])), source_schemas=schemas)
        assert model.native_pipeline_.export_state() == state
    finally:
        model.close()


def test_captured_unicode_schema_still_refuses_object_storage_without_truncation():
    from nirs4all.pipeline.dagml.methods_multimodal import source_schemas_from_cohort

    _require_public_binding()
    training, prediction, model = _case(False, True)
    rows = np.flatnonzero(np.asarray(training.partitions) == "train")
    schemas = {name: source_schemas_from_cohort(training)[name] for name in model.transformers}
    schemas["metadata"]["dtype"] = str(training.sources["metadata"].values.dtype)
    descriptor = json.loads(schemas["metadata"]["identity"])
    descriptor["dtype"] = schemas["metadata"]["dtype"]
    schemas["metadata"]["identity"] = json.dumps(descriptor, sort_keys=True)
    saved = deepcopy(schemas)
    try:
        model.fit([training.sources[name].values[rows] for name in model.transformers], training.y[rows], source_schemas=schemas)
        state = model.native_pipeline_.export_state()
        with pytest.raises(ValueError, match="mixed metadata dtype differs"):
            model.predict([prediction.sources[name].values.astype(object) if name == "metadata" else prediction.sources[name].values for name in model.transformers])
        assert schemas == saved
        assert model.native_pipeline_.export_state() == state
    finally:
        model.close()


def _cold(archive, declaration):
    import dag_ml
    import n4m
    from nirs4all_io import MultimodalDataset

    import nirs4all

    _require_public_binding()

    def forbidden(*args, **kwargs):
        raise AssertionError("FIT/HPO forbidden during mixed-text cold replay")

    n4m.MultimodalPipeline.fit = n4m.MultimodalClassifierPipeline.fit = forbidden
    dag_ml.execute_training = dag_ml.run_host_hpo_search_in_process = forbidden
    prediction = MultimodalDataset.from_dict(json.loads(Path(declaration).read_text()))
    result = nirs4all.predict(archive, prediction, verbose=0)
    assert result.metadata["training_performed"] is False
    print(json.dumps({"values": np.asarray(result.y_pred).reshape(-1).tolist()}, ensure_ascii=False))


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_portable_and_fresh_process_replay_preserve_mixed_categories(tmp_path, monkeypatch, classification, selected):
    import nirs4all
    from nirs4all.pipeline.dagml.methods_multimodal import source_schemas_from_cohort

    _require_public_binding()
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    training, prediction, model = _case(classification, selected)
    saved_schemas = deepcopy(source_schemas_from_cohort(training))
    train_rows = np.flatnonzero(np.asarray(training.partitions) == "train")
    expected, _, _ = _oracle(model, training, prediction, train_rows, classification)
    workspace = tmp_path / "workspace"
    pipeline = [GroupKFold(3), {"model": model}]
    tuning = None
    if selected:
        pipeline = [GroupKFold(3), {"_or_": [[{"model": model}]]}]
        tuning = {"engine": "n4m", "sampler": "random", "seed": 17, "n_trials": 1,
                  "space": {"early.n_components" if classification else "early.alpha": [2 if classification else 1.0]},
                  "storage": (tmp_path / "study").as_uri(), "study_name": "mixed"}
    with nirs4all.run(pipeline, training, tuning=tuning, engine="dag-ml", workspace_path=workspace,
                     verbose=0, save_charts=False, random_state=17) as result:
        archive = result.export(tmp_path / "mixed.n4a")
    assert source_schemas_from_cohort(training) == saved_schemas
    actual = nirs4all.predict(archive, prediction, verbose=0).y_pred
    if classification:
        np.testing.assert_array_equal(np.asarray(actual).reshape(-1), expected)
    else:
        np.testing.assert_allclose(np.asarray(actual).reshape(-1), expected, atol=2e-7, rtol=2e-7)
    declaration = tmp_path / "prediction.json"
    declaration.write_text(json.dumps(prediction.to_dict(), ensure_ascii=False))
    if workspace.exists():
        shutil.rmtree(workspace)
    assert not workspace.exists()
    environment = {name: value for name, value in os.environ.items() if name not in {"PYTHONPATH", "PYTHONHOME"}}
    # An isolated interpreter imports the active public binding anew. The final
    # installed SDK wheel gate is separately owned by the release runner.
    code = "import runpy,sys; module=runpy.run_path(sys.argv[1]); module['_cold'](*sys.argv[2:])"
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", code, __file__, str(archive), str(declaration)],
                               cwd=tmp_path, env=environment, capture_output=True, text=True, check=False, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    cold_values = json.loads(completed.stdout.strip().splitlines()[-1])["values"]
    np.testing.assert_array_equal(cold_values, np.asarray(actual).reshape(-1))
