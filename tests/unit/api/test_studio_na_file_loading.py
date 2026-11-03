"""Studio run and preview use one IO assembly for missing file observations."""
import numpy as np
import pytest

from nirs4all.api import studio_scientific_general as general


@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("policy", ["ignore", "remove_sample", "replace"])
def test_studio_file_materialization_keeps_spectral_target_and_metadata_alignment(tmp_path, categorical, policy):
    (tmp_path / "x.csv").write_text("w\n1\n2\n3\n4\n")
    (tmp_path / "y.csv").write_text("y\na\n\nb\na\n" if categorical else "y\n10\n\n30\n40\n")
    (tmp_path / "m.csv").write_text("label\none\ntwo\nthree\nfour\n")
    config = {
        "train_x": str(tmp_path / "x.csv"), "train_y": str(tmp_path / "y.csv"), "train_group": str(tmp_path / "m.csv"),
        "global_params": {"has_header": True, "delimiter": ";"},
        "train_y_params": {"na": {"policy": policy, "fill": {"method": "value", "fill_value": "unknown" if categorical else 7}}},
    }
    loaded = general._materialize_file_dataset(config)
    keep = [1, 3, 4] if policy == "remove_sample" else [1, 2, 3, 4]
    np.testing.assert_array_equal(loaded.x({}), np.asarray(keep).reshape(-1, 1))
    targets = np.asarray(loaded.y({"y": "raw"})).ravel()
    assert len(targets) == len(keep)
    if not categorical and policy == "remove_sample":
        np.testing.assert_array_equal(targets, [10, 30, 40])
    elif not categorical:
        # IO never imputes targets: ignore and replace keep the missing value.
        np.testing.assert_array_equal(targets, [10, np.nan, 30, 40])
    np.testing.assert_array_equal(np.asarray(loaded.metadata({})).ravel(), ["one", "three", "four"] if policy == "remove_sample" else ["one", "two", "three", "four"])


def test_studio_file_materialization_refuses_strict_na_without_legacy_fallback(tmp_path, monkeypatch):
    (tmp_path / "x.csv").write_text("w\n1\n2\n")
    (tmp_path / "y.csv").write_text("y\n1\n\n")
    monkeypatch.setattr("nirs4all.data.loaders.loader.load_XY", lambda *a, **kw: pytest.fail("legacy file loader"))
    with pytest.raises(ValueError, match="NA"):
        general._materialize_file_dataset({"train_x": str(tmp_path / "x.csv"), "train_y": str(tmp_path / "y.csv"), "train_y_params": {"na": {"policy": "abort"}}})


def test_studio_array_inputs_keep_existing_materialization():
    config = {"train_x": [[1, 2], [3, 4]], "train_y": [10, 20]}
    assert general._materialize_file_dataset(config) is config


def test_real_studio_training_accepts_canonical_na_file_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(general, "_ambient_runtime_preflight", lambda: None)
    monkeypatch.setattr("nirs4all.data.loaders.loader.load_XY", lambda *a, **kw: pytest.fail("legacy file loader"))
    x_path, y_path = tmp_path / "x.csv", tmp_path / "y.csv"
    x_path.write_text("w;v\n" + "".join(f"{index};{index % 2}\n" for index in range(24)))
    y_path.write_text("class\n" + "".join("\n" if index in {3, 8} else ("a\n" if index % 2 else "b\n") for index in range(24)))
    response = general.studio_scientific_job_v2({
        "schema": general.STUDIO_GENERAL_JOB_SCHEMA, "operation": "run", "job_id": "na-file-training",
        "pipeline": ["sklearn.preprocessing.StandardScaler", {"class": "sklearn.model_selection.StratifiedKFold", "params": {"n_splits": 3}}, "sklearn.linear_model.LogisticRegression"],
        "dataset": {"train_x": str(x_path), "train_y": str(y_path), "train_y_params": {"na": {"policy": "remove_sample"}}, "task_type": "binary_classification"},
        "options": {"workspace_path": str(tmp_path / "workspace"), "project": "na-regression", "verbose": 0, "save_charts": False},
    })
    assert response["engine"] == "dag-ml"
    assert response["result"]["run_ids"]
    assert response["result"]["native_score_sets_available"]
    assert np.isfinite(response["result"]["validation_score"])


@pytest.mark.parametrize("policy", ["auto", "ignore"])
def test_studio_training_keeps_missing_targets_out_of_training_and_predicts_them(tmp_path, monkeypatch, policy):
    # Studio no longer rewrites the training target policy: the SDK excludes the gaps itself.
    monkeypatch.setattr(general, "_ambient_runtime_preflight", lambda: None)
    x_path, y_path = tmp_path / "x.csv", tmp_path / "y.csv"
    rng = np.random.default_rng(3)
    features = rng.normal(size=(24, 3))
    x_path.write_text("a;b;c\n" + "".join(";".join(map(str, row)) + "\n" for row in features))
    y_path.write_text("y\n" + "".join("\n" if index in {3, 8} else f"{features[index].sum()}\n" for index in range(24)))
    response = general.studio_scientific_job_v2({
        "schema": general.STUDIO_GENERAL_JOB_SCHEMA, "operation": "run", "job_id": f"missing-targets-{policy}",
        "pipeline": [{"class": "sklearn.model_selection.KFold", "params": {"n_splits": 3}}, "sklearn.linear_model.Ridge"],
        "dataset": {"train_x": str(x_path), "train_y": str(y_path), "train_y_params": {"na": {"policy": policy}}},
        "options": {"workspace_path": str(tmp_path / "workspace"), "verbose": 0, "save_charts": False},
    })
    assert np.isfinite(response["result"]["validation_score"])
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    with WorkspaceStore(tmp_path / "workspace") as store:
        predictions = store.query_predictions(partition="prediction").to_dicts()
    assert [row["n_samples"] for row in predictions] == [2]
