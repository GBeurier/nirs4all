"""Real typed native alternatives, selected-only encoder FIT and portable replay."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

import nirs4all
from nirs4all.api.portable_archive import read_portable_predictor_archive_v2
from nirs4all.pipeline.dagml.cancellation import DagRunCancelled
from nirs4all.pipeline.dagml.methods_multimodal import recipe_from_estimator
from nirs4all.pipeline.dagml.structural_tuning import _prepare_structure
from tests.integration.api import test_methods_multimodal_u07 as fixed
from tests.integration.api import test_structural_hpo_preprocessing_chains as chains

_PATH = Path(__file__).resolve().parents[3] / "examples/user/04_models/U20_structural_hpo_typed_modalities.py"
_SPEC = importlib.util.spec_from_file_location("typed_structural_integration_example", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


@pytest.fixture(autouse=True)
def native_execution_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("legacy scheduler executed"))
    monkeypatch.setattr("nirs4all.pipeline.dagml.run_paths._run_model_on_precomputed_matrix", lambda *a, **k: pytest.fail("host CV scheduler executed"))


def _run(pipeline: Any, cohort: Any, directory: Path, tuning: dict[str, Any]) -> Any:
    return nirs4all.run(pipeline, cohort, tuning=tuning, engine="dag-ml", workspace_path=directory, random_state=17, refit=True, verbose=0, save_charts=False)


def _observe_native(monkeypatch: pytest.MonkeyPatch) -> tuple[list[Any], list[Any]]:
    from dag_ml.multimodal_methods import MethodsMultimodalController
    from n4m import MultimodalPipeline

    controllers, fitted = [], []
    original_init, original_fit = MethodsMultimodalController.__init__, MultimodalPipeline.fit

    def create(controller: Any, *args: Any, **kwargs: Any) -> None:
        original_init(controller, *args, **kwargs)
        controllers.append(controller)

    def fit(model: Any, sources: Any, targets: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_fit(model, sources, targets, *args, **kwargs)
        fitted.append({"model": model, "source_order": list(sources), "raw_shapes": {name: list(np.shape(values)) for name, values in sources.items()}})
        return result

    monkeypatch.setattr(MethodsMultimodalController, "__init__", create)
    monkeypatch.setattr(MultimodalPipeline, "fit", fit)
    return controllers, fitted


def test_each_typed_recipe_matches_fold_oracle_and_fits_selected_encoders_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cohort, pipeline = example.make_dataset(), example.make_pipeline()
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 4}
    prepared = _prepare_structure(pipeline, cohort, tuning, {"random_state": 17})
    catalogue = prepared["catalogue"]
    assert len(catalogue["entries"]) == 4
    assert len({entry["variant_label"] for entry in catalogue["entries"]}) == 4
    controllers, fitted = _observe_native(monkeypatch)
    chains._enqueue_recipes(monkeypatch, catalogue, catalogue["entries"])
    train = np.flatnonzero(np.asarray(cohort.partitions) == "train")
    test = np.flatnonzero(np.asarray(cohort.partitions) == "test")
    folds = list(GroupKFold(3).split(train[:, None], groups=np.asarray(cohort.groups)[train]))
    models = {json.dumps(recipe_from_estimator(model, allow_source_selection=True), sort_keys=True): model for model in pipeline[-1]["model"]["_or_"]}
    recipes = {entry["recipe_id"]: entry for entry in catalogue["entries"]}
    with _run(pipeline, cohort, tmp_path / "workspace", tuning) as result:
        evidence = result.structural_tuning_evidence
        assert {trial["params"]["__recipe__"] for trial in evidence["trials"]} == set(recipes)
        for trial in evidence["trials"]:
            node = recipes[trial["params"]["__recipe__"]]["graph"]["nodes"][0]
            assert node["params"]["recipe"] == node["operator"]["recipe"]
            assert len(node["operator"]["source_schemas"]) == 4
            model = clone(models[json.dumps(node["operator"]["recipe"], sort_keys=True)])
            model.set_params(model__alpha=trial["params"]["model.alpha"])
            scores = {f"fold{number}": float(np.sqrt(np.mean((fixed._oracle(model, cohort, train[fit], train[heldout])[0] - cohort.y[train[heldout]]) ** 2))) for number, (fit, heldout) in enumerate(folds)}
            assert trial["objective_fold_scores"] == pytest.approx(scores, rel=2e-7, abs=2e-7)
        winner = recipes[evidence["selected_params"]["__recipe__"]]["graph"]["nodes"][0]
        model = clone(models[json.dumps(winner["operator"]["recipe"], sort_keys=True)])
        model.set_params(model__alpha=evidence["selected_params"]["model.alpha"])
        archive = result.export(tmp_path / "winner.n4a")
        prediction_rows = test[::-1]
        expected = fixed._oracle(model, cohort, train, prediction_rows)[0]
        actual = nirs4all.predict(archive, cohort.take([cohort.sample_ids[index] for index in prediction_rows]), engine="dag-ml")
        np.testing.assert_allclose(actual.y_pred.ravel(), expected, rtol=2e-7, atol=2e-7)
        assert actual.metadata["training_performed"] is False
        package = read_portable_predictor_archive_v2(archive).to_dict()
        artifacts = package["execution_bundle"]["refit_artifacts"]
        assert len(artifacts) == 1 and artifacts[0]["artifact"]["kind"] == "methods_multimodal_pipeline"
    fit_events = [event for controller in controllers for event in controller.audit if event["operation"] == "fit"]
    assert len(fit_events) == len(fitted) == 4 * 3 + 3 + 1
    for event in fit_events:
        names = event["source_order"]
        assert set(event["raw_shapes"]) == set(names) == set(event["source_weights"]) == set(event["recipe"]["encoders"])
        assert not set(event["sample_ids"]).intersection(cohort.sample_ids[index] for index in test)
        expected_rows = train if event["fold"] is None else train[folds[int(event["fold"][4:])][0]]
        assert set(event["sample_ids"]) == {cohort.sample_ids[index] for index in expected_rows}
    assert {tuple(entry["source_order"]) for entry in fitted} == {("nir",), ("nir", "image"), ("image", "nir"), ("image", "series", "metadata")}
    for entry in fitted:
        with pytest.raises(RuntimeError, match="closed"):
            entry["model"].export_state()


def test_zero_weight_is_not_a_modality_exclusion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pipeline = [GroupKFold(3), {"model": {"_or_": [example.make_model(("nir", "image"), {"image": 0.0})]}}]
    controllers, fitted = _observe_native(monkeypatch)
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 1}
    with _run(pipeline, example.make_dataset(), tmp_path / "workspace", tuning):
        pass
    events = [event for controller in controllers for event in controller.audit if event["operation"] == "fit"]
    assert events and all(event["source_order"] == ["nir", "image"] and event["source_weights"]["image"] == 0 for event in events)
    assert fitted and all(entry["source_order"] == ["nir", "image"] for entry in fitted)


@pytest.mark.parametrize("mutation", ["selection", "order", "weight", "schema", "excluded_train", "excluded_test"])
def test_resume_rejects_changed_recipe_or_complete_raw_input_before_callbacks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str) -> None:
    from dag_ml import DagMlRuntimeError
    from dag_ml.multimodal_methods import MethodsMultimodalController

    cohort, pipeline = example.make_dataset(), example.make_pipeline()
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 4}
    prepared = _prepare_structure(pipeline, cohort, tuning, {})
    chains._enqueue_recipes(monkeypatch, prepared["catalogue"], prepared["catalogue"]["entries"][:1])
    with pytest.raises(DagRunCancelled):
        _run(pipeline, cohort, tmp_path / "workspace", {**tuning, "progress_callback": lambda event: len(event["checkpoint"]["trials"]) < 1})
    checkpoint = tmp_path / "study/structural-typed-modalities.n4mopt.json"
    before = checkpoint.read_bytes()
    if mutation == "selection":
        pipeline[-1]["model"]["_or_"][0] = example.make_model(("nir", "image"))
    elif mutation == "order":
        model = pipeline[-1]["model"]["_or_"][1]
        model.transformers = dict(reversed(list(model.transformers.items())))
    elif mutation == "weight":
        pipeline[-1]["model"]["_or_"][0].source_weights = {"nir": 2.0}
    elif mutation == "schema":
        cohort = fixed._replace(cohort, "metadata", cohort.sources["metadata"].values, feature_names=["different", "category"])
    else:
        values = np.array(cohort.sources["series"].values, copy=True)
        partition = "test" if mutation == "excluded_test" else "train"
        row = np.flatnonzero(np.asarray(cohort.partitions) == partition)[0]
        values[row, 0, 0] += 0.5
        cohort = fixed._replace(cohort, "series", values)
    monkeypatch.setattr(MethodsMultimodalController, "operator", lambda *a, **k: pytest.fail("resume mismatch reached native callback"))
    with pytest.raises((ValueError, RuntimeError, DagMlRuntimeError)) as refused:
        _run(pipeline, cohort, tmp_path / "resume", {**tuning, "resume": True})
    if mutation in {"excluded_train", "excluded_test"}:
        assert isinstance(refused.value, DagMlRuntimeError)
        assert "binding mismatch" in str(refused.value)
    assert checkpoint.read_bytes() == before


def test_fresh_installed_typed_winner_requires_full_raw_contract_without_fit(tmp_path: Path) -> None:
    installed = os.environ.get("NIRS4ALL_STRUCTURAL_HPO_INSTALLED_PYTHON")
    if not installed:
        if os.environ.get("NIRS4ALL_REQUIRE_STRUCTURAL_HPO_INSTALLED") == "1":
            pytest.fail("mandatory installed proof requires NIRS4ALL_STRUCTURAL_HPO_INSTALLED_PYTHON")
        pytest.skip("fresh installed Python is supplied by the qualification gate")
    model = example.make_model(("image", "nir"), {"image": 0.5})
    pipeline = [GroupKFold(3), {"model": {"_or_": [model]}}]
    training, new = example.make_dataset(), example.make_dataset(29, prediction=True)
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 1}
    train = np.flatnonzero(np.asarray(training.partitions) == "train")
    with _run(pipeline, training, tmp_path / "workspace", tuning) as result:
        model.set_params(model__alpha=result.tuning_best_params["model.alpha"])
        # Evaluate the independent fixed sklearn declaration on a combined
        # aligned training/new cohort through its ordinary public transforms.
        encoded_train, encoded_new = [], []
        for name, transformer in model.transformers.items():
            fitted = clone(transformer).fit(training.sources[name].values[train], training.y[train])
            weight = (model.source_weights or {}).get(name, 1.0)
            encoded_train.append(fitted.transform(training.sources[name].values[train]) * weight)
            encoded_new.append(fitted.transform(new.sources[name].values) * weight)
        predictor = clone(model.model).fit(np.concatenate(encoded_train, axis=1), training.y[train])
        expected = predictor.predict(np.concatenate(encoded_new, axis=1)).tolist()
        archive = result.export(tmp_path / "typed.n4a")
    if (tmp_path / "workspace").exists():
        shutil.rmtree(tmp_path / "workspace")
    shutil.rmtree(tmp_path / "study")
    (tmp_path / "cohort.json").write_text(json.dumps(new.to_dict(), allow_nan=False))
    package_root = Path(nirs4all.__file__).resolve().parent
    sources = ["pipeline/dagml/methods_multimodal.py", "pipeline/dagml/structural_multimodal.py", "pipeline/dagml/structural_tuning.py"]
    evidence = {"prediction": expected, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "source_sha256": {name: hashlib.sha256((package_root / name).read_bytes()).hexdigest() for name in sources}}
    (tmp_path / "expected.json").write_text(json.dumps(evidence, allow_nan=False))
    script = textwrap.dedent("""\
        import hashlib, json, pathlib, sys
        import dag_ml, numpy as np, nirs4all
        from n4m import MultimodalPipeline
        from n4m.model_selection.optimizer import Optimizer
        from nirs4all_io import MultimodalDataset
        from nirs4all.pipeline.dagml.host_search_checkpoint import HostSearchOptimizer
        expected = json.loads(pathlib.Path(sys.argv[3]).read_text())
        root = pathlib.Path(nirs4all.__file__).resolve().parent
        assert 'site-packages' in root.parts, root
        for name, digest in expected['source_sha256'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
        archive = pathlib.Path(sys.argv[1])
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected['archive_sha256']
        def forbidden(*args, **kwargs):
            raise AssertionError('typed archive replay reached FIT/HPO')
        MultimodalPipeline.fit = forbidden
        Optimizer.__init__ = forbidden
        Optimizer.load = classmethod(forbidden)
        HostSearchOptimizer.__init__ = forbidden
        for name in ('run_host_hpo_search_in_process', 'execute_training', 'prepare_host_hpo_structural_catalogue', 'resolve_host_hpo_structural_winner'):
            setattr(dag_ml, name, forbidden)
        nirs4all.run = forbidden
        cohort = MultimodalDataset.from_dict(json.loads(pathlib.Path(sys.argv[2]).read_text()))
        result = nirs4all.predict(archive, cohort, engine='dag-ml')
        np.testing.assert_allclose(result.y_pred.ravel(), expected['prediction'], rtol=2e-7, atol=2e-7)
        assert result.metadata['training_performed'] is False
        malformed = MultimodalDataset({name:source for name,source in cohort.sources.items() if name != 'series'},
                                      sample_ids=cohort.sample_ids, partitions=cohort.partitions, name=cohort.name)
        try:
            nirs4all.predict(archive, malformed, engine='dag-ml')
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError('archive accepted removal of an excluded raw source')
        print(json.dumps({'installed':str(root),'training_performed':False,'full_raw_contract':True}))
    """)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    process = subprocess.run(
        [installed, "-I", "-c", script, str(archive), str(tmp_path / "cohort.json"), str(tmp_path / "expected.json")], cwd=tmp_path, env=environment, capture_output=True, text=True, check=False, timeout=180
    )
    assert process.returncode == 0, process.stderr[-6000:]
    assert json.loads(process.stdout.splitlines()[-1])["full_raw_contract"] is True
