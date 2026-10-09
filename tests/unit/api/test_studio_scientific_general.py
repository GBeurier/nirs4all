"""The v2 host contract uses real general DAG results and bounded JSON."""

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.api import studio_scientific_general as general
from nirs4all.pipeline.config.component_serialization import serialize_component
from tests.integration.api.test_multimodal_late_missing import _late_pipeline
from tests.integration.api.test_multimodal_ragged import _ragged_cohort


def _request(tmp_path, classification=False):
    rng = np.random.default_rng(14)
    X = rng.normal(size=(30, 8))
    y = X[:, 0] * 2 - X[:, 1]
    if classification:
        y = (y > np.median(y)).astype(int)
    splitter = StratifiedKFold(3) if classification else KFold(3)
    model = LogisticRegression() if classification else Ridge()
    return {
        "schema": general.STUDIO_GENERAL_JOB_SCHEMA,
        "operation": "run", "job_id": "job-general",
        "pipeline": serialize_component([StandardScaler(), splitter, model]),
        "dataset": {"X": X.tolist(), "y": y.tolist()},
        "options": {"workspace_path": str(tmp_path), "name": "general", "project": "studio"},
    }


@pytest.mark.parametrize("classification", [False, True])
def test_general_contract_executes_canonical_pipeline_and_real_workspace(tmp_path, monkeypatch, classification):
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    # Source checkout qualification only; installed-prefix attestation is
    # independently exercised by the installed host integration gate.
    monkeypatch.setattr(general, "_ambient_runtime_preflight", lambda: None)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("implicit legacy host"))
    response = general.studio_scientific_job_v2(_request(tmp_path, classification))
    json.dumps(response, allow_nan=False)
    assert response["schema"] == general.STUDIO_GENERAL_RESULT_SCHEMA
    assert response["engine"] == "dag-ml"
    assert response["result"]["native_score_sets_available"]
    assert response["result"]["prediction_count"] > 0
    assert response["result"]["evaluations"] == []
    assert np.isfinite(response["result"]["validation_score"])
    with WorkspaceStore(tmp_path) as store:
        for run_id in response["result"]["run_ids"]:
            assert store.get_run(run_id)["status"] == "completed"
        assert store.query_predictions().height == response["result"]["prediction_count"]


def test_general_contract_executes_typed_ragged_cohort_with_missing_sources(tmp_path, monkeypatch):
    from nirs4all_io import MultimodalDataset, RaggedSeriesSource

    monkeypatch.setattr(general, "_ambient_runtime_preflight", lambda: None)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("implicit legacy host"))
    cohort = _ragged_cohort(missing=True)
    request = _request(tmp_path)
    request["pipeline"] = serialize_component(_late_pipeline())
    request["dataset"] = {"schema": general.STUDIO_MULTIMODAL_DATASET_SCHEMA, "cohort": cohort.to_dict()}

    restored = general._inline_dataset_arrays(request["dataset"])
    assert isinstance(restored, MultimodalDataset)
    assert tuple(restored.sources) == tuple(cohort.sources)
    assert restored.sample_ids == cohort.sample_ids
    assert isinstance(restored.sources["series"], RaggedSeriesSource)
    np.testing.assert_array_equal(restored.sources["series"].presence_mask, cohort.sources["series"].presence_mask)
    np.testing.assert_array_equal(restored.sources["series"].time_coordinates, cohort.sources["series"].time_coordinates)

    response = general.studio_scientific_job_v2(request)
    assert response["engine"] == "dag-ml"
    assert response["result"]["run_ids"] == []
    assert response["result"]["native_results_dirs"] == []
    archive = Path(response["result"]["archive_path"])
    assert archive.is_file() and archive.parent == tmp_path / "exports"
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
    assert manifest["multimodal_host"]["source_presence"]["source_names"] == list(cohort.sources)
    from nirs4all.operators.transforms import SequenceSummary

    for estimator in (SequenceSummary, StandardScaler, Ridge):
        monkeypatch.setattr(estimator, "fit", lambda *args, **kwargs: pytest.fail("archive replay attempted fitting"))
    replay = nirs4all.predict(archive, restored)
    assert replay.y_pred.shape == (len(cohort),)
    assert np.isfinite(replay.y_pred).all()
    assert replay.metadata["training_performed"] is False
    assert response["result"]["native_score_sets_available"]
    assert np.isfinite(response["result"]["validation_score"])


@pytest.mark.parametrize("cohort", [None, {}, {"schema": "wrong"}])
def test_general_typed_cohort_refuses_invalid_payload_before_runtime(tmp_path, monkeypatch, cohort):
    request = _request(tmp_path)
    request["dataset"] = {"schema": general.STUDIO_MULTIMODAL_DATASET_SCHEMA, "cohort": cohort}
    monkeypatch.setattr(general, "_ambient_runtime_preflight", lambda: pytest.fail("runtime touched before dataset validation"))
    with pytest.raises(general.StudioScientificJobError) as error:
        general.studio_scientific_job_v2(request)
    assert error.value.code == "invalid_dataset"


@pytest.mark.parametrize("dtype,shape", [("U500000000", [1]), ("U1", [0, 1000000000])])
def test_general_typed_cohort_bounds_declared_array_allocation_before_runtime(tmp_path, monkeypatch, dtype, shape):
    request = _request(tmp_path)
    request["dataset"] = {
        "schema": general.STUDIO_MULTIMODAL_DATASET_SCHEMA,
        "cohort": {"sources": [{"array": {"dtype": dtype, "shape": shape, "values": ["a"] if shape == [1] else []}}]},
    }
    with pytest.raises(ValueError, match="allocation exceeds budget"):
        general._validate_multimodal_array_budget(request["dataset"]["cohort"])
    monkeypatch.setattr(general, "_ambient_runtime_preflight", lambda: pytest.fail("runtime touched before allocation validation"))
    with pytest.raises(general.StudioScientificJobError) as error:
        general.studio_scientific_job_v2(request)
    assert error.value.code == "invalid_dataset"


@pytest.mark.parametrize("amplification", ["values", "alignment"])
def test_general_typed_cohort_refuses_array_allocation_amplification(tmp_path, monkeypatch, amplification):
    request = _request(tmp_path)
    if amplification == "values":
        cohort = {"sources": [{"array": {"dtype": "U1000000", "shape": [1], "values": ["a"] * 1000}}]}
    else:
        cohort = {
            "source_alignment": "left",
            "sample_ids": [f"sample-{index}" for index in range(1000)],
            "sources": [{"array": {"dtype": "float64", "shape": [0, 1000000], "values": []}}],
        }
    request["dataset"] = {"schema": general.STUDIO_MULTIMODAL_DATASET_SCHEMA, "cohort": cohort}
    with pytest.raises(ValueError, match="Multimodal array"):
        general._validate_multimodal_array_budget(cohort)
    monkeypatch.setattr(general, "_ambient_runtime_preflight", lambda: pytest.fail("runtime touched before allocation validation"))
    with pytest.raises(general.StudioScientificJobError) as error:
        general.studio_scientific_job_v2(request)
    assert error.value.code == "invalid_dataset"


@pytest.mark.parametrize("change,code", [
    ({"engine": "legacy"}, "engine_forbidden"),
    ({"allow_fallback": True}, "engine_forbidden"),
    ({"extra": "unknown"}, "invalid_shape"),
    ({"pipeline": [{"function": "os.system", "params": {"command": "must-not-execute"}}]}, "operator_package_forbidden"),
    ({"pipeline": [{"class": "user_plugin.Model"}]}, "operator_package_forbidden"),
    ({"pipeline": [{"class": ""}]}, "invalid_operator"),
    ({"options": {"workspace_path": "relative"}}, "workspace_required"),
    ({"options": {"workspace_path": "/workspace", "engine": "legacy"}}, "unknown_option"),
])
def test_invalid_general_request_fails_before_runtime_or_operator_instantiation(tmp_path, monkeypatch, change, code):
    request = {**_request(tmp_path), **change}
    monkeypatch.setattr(general, "_ambient_runtime_preflight", lambda: pytest.fail("runtime touched before validation"))
    monkeypatch.setattr(general, "deserialize_component", lambda *args: pytest.fail("operator instantiated before validation"))
    with pytest.raises(general.StudioScientificJobError) as error:
        general.studio_scientific_job_v2(request)
    assert error.value.code == code


def test_general_operator_failure_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(general, "_ambient_runtime_preflight", lambda: None)
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("operator failure")

    monkeypatch.setattr(general, "_run_strict_product", fail)
    with pytest.raises(RuntimeError, match="operator failure"):
        general.studio_scientific_job_v2(_request(tmp_path))
    assert len(calls) == 1
    assert calls[0]["engine"] == "dag-ml"
    assert calls[0]["allow_fallback"] is False


@pytest.mark.parametrize("operator", ["TabPFNRegressor", "TabPFNClassifier"])
def test_product_tabpfn_documents_do_not_import_optional_dependency(monkeypatch, operator):
    monkeypatch.setattr(general, "import_module", lambda *args: pytest.fail("document validation imported optional code"))
    monkeypatch.setattr(general, "deserialize_component", lambda *args: pytest.fail("document validation instantiated code"))
    general.validate_studio_pipeline_config([
        {"model": {"class": f"tabpfn.{operator}", "params": {"n_estimators": 8}}},
    ])


@pytest.mark.parametrize("operator", ["tabpfn.UnknownEstimator", "tabpfn.utils.download", "tabpfn_other.TabPFNRegressor"])
def test_product_tabpfn_authorization_does_not_authorize_package(operator):
    with pytest.raises(general.StudioScientificJobError) as error:
        general.validate_studio_pipeline_config([{"class": operator}])
    assert error.value.code == "operator_package_forbidden"


@pytest.mark.parametrize("operator", ["TabPFNRegressor", "TabPFNClassifier"])
def test_available_tabpfn_preflight_does_not_construct_estimator(monkeypatch, operator):
    def must_not_construct():
        pytest.fail("preflight constructed estimator")

    monkeypatch.setattr(general, "import_module", lambda name: SimpleNamespace(**{operator: must_not_construct}))
    general._preflight_optional_product_operators([{"model": {"class": f"tabpfn.{operator}"}}])


@pytest.mark.parametrize("package, operator", [("lightgbm", "LGBMClassifier"), ("xgboost", "XGBRegressor"), ("catboost", "CatBoostClassifier")])
def test_missing_preset_dependency_fails_before_construction(monkeypatch, package, operator):
    def missing(name):
        assert name == package
        raise ModuleNotFoundError(package)

    monkeypatch.setattr(general, "import_module", missing)
    with pytest.raises(general.StudioScientificJobError) as raised:
        general._preflight_optional_product_operators([{"model": {"class": f"{package}.{operator}"}}])
    assert raised.value.code == "dependency_missing"
    assert package in str(raised.value)


@pytest.mark.parametrize("operator", ["TabPFNRegressor", "TabPFNClassifier"])
@pytest.mark.parametrize("failure", [ModuleNotFoundError("No module named 'tabpfn'"), AttributeError("operator unavailable")])
def test_missing_tabpfn_execution_fails_before_instantiation_or_run(tmp_path, monkeypatch, operator, failure):
    request = _request(tmp_path)
    request["pipeline"] = [{"model": {"class": f"tabpfn.{operator}"}}]
    monkeypatch.setattr(general, "_ambient_runtime_preflight", lambda: None)

    def missing(name):
        assert name == "tabpfn"
        raise failure

    monkeypatch.setattr(general, "import_module", missing)
    monkeypatch.setattr(general, "deserialize_component", lambda *args: pytest.fail("missing dependency reached construction"))
    monkeypatch.setattr(general, "_run_strict_product", lambda *args, **kwargs: pytest.fail("missing dependency reached execution"))
    with pytest.raises(general.StudioScientificJobError, match="preset can still be imported and edited") as error:
        general.studio_scientific_job_v2(request)
    assert error.value.code == "dependency_missing"
