"""No-splitter REFIT parity across DAG-ML's Python and CLI mechanisms."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nirs4all.data.config import DatasetConfigs
from nirs4all.pipeline.dagml.full_train import NoSplitEvaluationWarning
from nirs4all.pipeline.dagml.host_container import BUNDLE_MANIFEST_MEMBER, open_n4a_source
from nirs4all.pipeline.dagml.rt import RtError
from tests.unit.pipeline.dagml.host_container_helpers import reseal_host_member

from ._datasets import dataset_path


@pytest.mark.parity
def test_no_splitter_cli_matches_direct_train_only_refit_and_archive(tmp_path, monkeypatch) -> None:
    import nirs4all

    from ._dagml_cli import dagml_cli_path

    cli = dagml_cli_path()
    if not cli.exists():
        pytest.skip(f"dag-ml-cli binary not built at {cli}")
    path = dataset_path("regression")
    dataset = DatasetConfigs(path).get_dataset_at(0)
    train_x = dataset.x({"partition": "train"}, layout="2d")
    train_y = dataset.y({"partition": "train"})
    test_x = dataset.x({"partition": "test"}, layout="2d")
    test_y = np.asarray(dataset.y({"partition": "test"})).ravel()
    oracle = PLSRegression(n_components=5).fit(train_x, train_y)
    oracle_pred = np.asarray(oracle.predict(test_x)).ravel()
    oracle_rmse = float(np.sqrt(np.mean((test_y - oracle_pred) ** 2)))

    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0")
    monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    with pytest.warns(NoSplitEvaluationWarning, match="No splitter"):
        result = nirs4all.run([PLSRegression(n_components=5)], path, engine="dag-ml", save_artifacts=False)
    assert result.execution_engine == "dag-ml"
    assert np.isnan(result.cv_best_score)
    assert result.best_rmse == pytest.approx(oracle_rmse, abs=1e-5)
    assert {frame["lineage"]["phase"] for frame in result._dagml_node_results} == {"REFIT"}
    assert len(result._dagml_refit_artifacts) == 1
    np.testing.assert_allclose(
        np.asarray(result._dagml_refit_artifacts[0]["estimator"].predict(test_x)).ravel(),
        oracle_pred, atol=1e-5,
    )
    archive = result.export(tmp_path / "cli_full_train.n4a")
    np.testing.assert_allclose(
        np.asarray(nirs4all.predict(archive, test_x).y_pred).ravel(), oracle_pred, atol=1e-5,
    )


@pytest.mark.parity
def test_no_splitter_cli_in_memory_training_has_no_fabricated_validation(monkeypatch) -> None:
    import nirs4all

    from ._dagml_cli import dagml_cli_path

    cli = dagml_cli_path()
    if not cli.exists():
        pytest.skip(f"dag-ml-cli binary not built at {cli}")
    rng = np.random.default_rng(87)
    x = rng.normal(size=(18, 8))
    y = rng.normal(size=(18, 1))
    oracle = make_pipeline(StandardScaler(), Ridge(alpha=0.5)).fit(x, y)
    oracle_rmse = float(np.sqrt(np.mean((y.ravel() - np.asarray(oracle.predict(x)).ravel()) ** 2)))

    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0")
    monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    with pytest.warns(NoSplitEvaluationWarning, match="No splitter"):
        result = nirs4all.run([StandardScaler(), Ridge(alpha=0.5)], (x, y), engine="dag-ml", save_artifacts=False)
    assert np.isnan(result.cv_best_score)
    assert np.isnan(result.best_rmse)
    train_row = next(row for row in result.predictions.filter_predictions(load_arrays=True) if row["partition"] == "train")
    assert train_row["train_score"] == pytest.approx(oracle_rmse, abs=1e-6)
    assert result.per_dataset[next(iter(result.per_dataset))]["evaluation"]["validation_source"] is None
    assert {frame["lineage"]["phase"] for frame in result._dagml_node_results} == {"REFIT"}


@pytest.mark.parity
def test_no_splitter_cli_by_source_auto_matches_independent_source_models(tmp_path, monkeypatch) -> None:
    import nirs4all

    from ._dagml_cli import dagml_cli_path

    cli = dagml_cli_path()
    if not cli.exists():
        pytest.skip(f"dag-ml-cli binary not built at {cli}")
    path = dataset_path("multi")
    dataset = DatasetConfigs(path).get_dataset_at(0)
    train_blocks = dataset.x({"partition": "train"}, "3d", concat_source=False)
    test_blocks = dataset.x({"partition": "test"}, "3d", concat_source=False)
    y_train = np.asarray(dataset.y({"partition": "train"}))
    source_names = [f"source_{index}" for index in range(len(train_blocks))]
    pipeline = [
        {"branch": {"by_source": True, "steps": {
            name: [{"model": Ridge(alpha=1.0)}] for name in source_names
        }}},
        {"merge": "auto"},
    ]
    legacy = nirs4all.run(pipeline, path, engine="legacy", save_artifacts=False, verbose=0)
    assert legacy.num_predictions > 0
    results = []
    for mode in ("1", "0"):
        monkeypatch.setenv("N4A_DAGML_INPROCESS", mode)
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
        with pytest.warns(NoSplitEvaluationWarning, match="No splitter"):
            results.append(nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0))
    for mode, result in zip(("in_process", "cli"), results, strict=True):
        assert np.isnan(result.cv_best_score)
        assert len(result._dagml_refit_artifacts) == len(source_names)
        assert {frame["lineage"]["phase"] for frame in result._dagml_node_results} == {"REFIT"}
        rows = [row for row in result.predictions.filter_predictions(load_arrays=True) if row["partition"] == "test"]
        assert {row["branch_name"] for row in rows} == set(source_names)
        for index, name in enumerate(source_names):
            expected = Ridge(alpha=1.0).fit(
                np.asarray(train_blocks[index]).reshape(len(y_train), -1), y_train,
            ).predict(np.asarray(test_blocks[index]).reshape(len(test_blocks[index]), -1))
            row = next(row for row in rows if row["branch_name"] == name)
            np.testing.assert_allclose(np.asarray(row["y_pred"]).ravel(), np.asarray(expected).ravel(), atol=1e-6)
        archive = tmp_path / f"by_source_auto_{mode}.n4a"
        result.export(archive)
        from nirs4all.pipeline.bundle.loader import BundleLoader

        loader = BundleLoader(archive)
        output_ids = tuple(f"output:source_{index}" for index in range(len(source_names)))
        assert loader.named_outputs == output_ids
        with open_n4a_source(archive) as archive_source:
            archive_file = archive_source.require_host()
            archive_manifest = archive_file.bundle_manifest
            topology = archive_manifest["dagml_independent_output_topology"]
            package_ref = archive_manifest["dagml_initial_full_refit_package_ref"]
            package_bytes = archive_file.read(package_ref["path"])
            package = json.loads(package_bytes)
        assert package["schema_version"] == 1
        assert len(package["outputs"]) == len(source_names)
        assert {entry["dagml_output_id"] for entry in topology["outputs"]} == {
            binding["output_id"] for binding in package["outputs"]
        }
        assert topology["schema_id"] == "dag-ml.host_independent_outputs.v1"
        assert [(entry["source_id"], entry["output_binding_id"]) for entry in topology["outputs"]] == list(zip(source_names, output_ids, strict=True))
        assert all(len(entry["feature_axis_cm1"]) == entry["feature_width"] for entry in topology["outputs"])
        full_x = np.asarray(dataset.x({"partition": "test"}, "2d"))
        with pytest.raises(ValueError, match="multiple named outputs"):
            loader.predict(full_x)
        all_outputs = loader.predict_outputs(full_x)
        assert set(all_outputs) == set(output_ids)
        named_sources = {
            "sample_ids": [f"sample_{row}" for row in range(len(full_x))],
            "sources": {},
        }
        for index, name in enumerate(source_names):
            block = np.asarray(test_blocks[index]).reshape(len(test_blocks[index]), -1)
            order = np.arange(len(block))[::-1] if index % 2 == 0 else np.arange(len(block))
            named_sources["sources"][name] = {
                "sample_ids": [named_sources["sample_ids"][row] for row in order],
                "values": block[order],
                "feature_axis_cm1": topology["outputs"][index]["feature_axis_cm1"],
            }
        named_outputs = loader.predict_outputs(named_sources)
        assert set(named_outputs) == set(output_ids)
        mismatched_axis = {"sample_ids": named_sources["sample_ids"], "sources": {
            name: dict(payload) for name, payload in named_sources["sources"].items()
        }}
        mismatched_axis["sources"][source_names[0]]["feature_axis_cm1"] = list(reversed(
            named_sources["sources"][source_names[0]]["feature_axis_cm1"],
        ))
        with pytest.raises(ValueError, match="spectral axis differs"):
            loader.predict_outputs(mismatched_axis)
        with pytest.raises(ValueError, match="spectral axis differs"):
            nirs4all.predict(archive, mismatched_axis, output=output_ids[0])
        for index, name in enumerate(source_names):
            expected = Ridge(alpha=1.0).fit(
                np.asarray(train_blocks[index]).reshape(len(y_train), -1), y_train,
            ).predict(np.asarray(test_blocks[index]).reshape(len(test_blocks[index]), -1))
            np.testing.assert_allclose(np.asarray(all_outputs[output_ids[index]]).ravel(), np.asarray(expected).ravel(), atol=1e-6)
            np.testing.assert_allclose(np.asarray(named_outputs[output_ids[index]]).ravel(), np.asarray(expected).ravel(), atol=1e-6)
            np.testing.assert_allclose(np.asarray(loader.predict_output(output_ids[index], named_sources)).ravel(), np.asarray(expected).ravel(), atol=1e-6)
            np.testing.assert_allclose(np.asarray(loader.predict_output(output_ids[index], full_x)).ravel(), np.asarray(expected).ravel(), atol=1e-6)
        with pytest.raises(ValueError, match="unknown named output"):
            loader.predict_output("unknown", full_x)
        with pytest.raises(ValueError, match="no named output"):
            nirs4all.predict(archive, full_x, output=source_names[0])
        corrupt_archive = tmp_path / f"by_source_swapped_{mode}.n4a"
        manifest = copy.deepcopy(archive_manifest)
        manifest["dagml_independent_output_topology"]["outputs"][0]["source_id"] = "other_source"
        reseal_host_member(archive, corrupt_archive, BUNDLE_MANIFEST_MEMBER, json.dumps(manifest).encode())
        with pytest.raises(ValueError, match="disagrees with its named-output manifest"):
            BundleLoader(corrupt_archive).predict_outputs(full_x)
        corrupt_package_archive = tmp_path / f"by_source_package_corrupt_{mode}.n4a"
        # Reseal only the container inventory; the bundle manifest keeps the original package digest.
        reseal_host_member(archive, corrupt_package_archive, package_ref["path"], package_bytes + b" ")
        from nirs4all.pipeline.dagml.general_archive import load_general_archive

        with pytest.raises(ValueError, match="package fingerprint mismatch"):
            load_general_archive(corrupt_package_archive)
        forged_package_archive = tmp_path / f"by_source_package_forged_{mode}.n4a"
        forged_package = json.loads(package_bytes)
        forged_package["outputs"][0]["output_id"] = "output:forged"
        forged_bytes = json.dumps(forged_package, sort_keys=True, separators=(",", ":")).encode()
        forged_manifest = copy.deepcopy(archive_manifest)
        forged_manifest["dagml_initial_full_refit_package_ref"]["sha256"] = hashlib.sha256(forged_bytes).hexdigest()
        # Re-sign the forged package in the bundle manifest and reseal the container around both members.
        reseal_host_member(archive, forged_package_archive, package_ref["path"], forged_bytes)
        reseal_host_member(forged_package_archive, forged_package_archive, BUNDLE_MANIFEST_MEMBER, json.dumps(forged_manifest).encode())
        with pytest.raises(ValueError, match="fingerprint"):
            load_general_archive(forged_package_archive)
        with pytest.raises(ValueError, match="multiple named outputs"):
            nirs4all.predict(archive, full_x)
        public_selected = nirs4all.predict(archive, full_x, output=output_ids[1])
        public_named = nirs4all.predict(archive, named_sources, output=output_ids[1])
        assert public_selected.metadata["execution_profile"] == "initial_full_refit_package_replay"
        assert public_named.metadata["execution_profile"] == "initial_full_refit_package_replay"
        assert public_named.metadata["package_fingerprint"] == package["package_fingerprint"]
        np.testing.assert_allclose(
            np.asarray(public_named.y_pred).ravel(),
            np.asarray(named_outputs[output_ids[1]]).ravel(), atol=1e-4,
        )
        assert public_named.metadata["phase"] == "PREDICT"
        assert public_named.metadata["source_sample_ids"] == named_sources["sample_ids"]
        from nirs4all.pipeline.dagml.dataset import _materialize_dataset

        # The public array path materializes a SpectroDataset before DAG replay;
        # compare against the same materialized features, not the raw float64 input.
        public_x = np.asarray(_materialize_dataset(full_x).x({}, layout="2d"))
        np.testing.assert_allclose(
            np.asarray(public_selected.y_pred).ravel(),
            np.asarray(loader.predict_output(output_ids[1], public_x)).ravel(),
            atol=1e-4,
        )
        assert public_selected.metadata["selected_output"] == output_ids[1]
        with pytest.raises(RtError, match="independent source predictions"):
            result.export(tmp_path / f"legacy_refit_{mode}.n4a", compatibility="legacy-refit")
        selected = next(row for row in rows if row["branch_name"] == source_names[1])
        selected_archive = tmp_path / f"by_source_selected_{mode}.n4a"
        result.export(selected_archive, source=selected)
        with open_n4a_source(selected_archive) as archive_source:
            manifest = archive_source.require_host().bundle_manifest
        assert manifest["dagml_native_export_shape"] == "independent_by_source_selected"
        assert manifest["dagml_selected_source"]["name"] == source_names[1]
        assert manifest["dagml_selected_source"]["index"] == 1
        selected_x = np.asarray(test_blocks[1]).reshape(len(test_blocks[1]), -1)
        selected_prediction = BundleLoader(selected_archive).predict(selected_x)
        full_x = np.asarray(dataset.x({"partition": "test"}, "2d"))
        full_prediction = BundleLoader(selected_archive).predict(full_x)
        expected_selected = Ridge(alpha=1.0).fit(
            np.asarray(train_blocks[1]).reshape(len(y_train), -1), y_train,
        ).predict(selected_x)
        np.testing.assert_allclose(np.asarray(selected_prediction).ravel(), np.asarray(expected_selected).ravel(), atol=1e-6)
        np.testing.assert_allclose(np.asarray(full_prediction).ravel(), np.asarray(expected_selected).ravel(), atol=1e-6)
        with pytest.raises(ValueError, match="source= must identify"):
            result.export(tmp_path / f"forged_{mode}.n4a", source={"id": "foreign"})


@pytest.mark.parity
@pytest.mark.parametrize("with_preprocessing", [False, True])
def test_legacy_by_source_auto_exports_selected_fitted_model_cold(tmp_path, with_preprocessing: bool) -> None:
    """Default-best and explicit-source exports retain real source-local fitted state."""
    import nirs4all
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    from .test_dagml_cli_runner import _two_source_distinct_dataset

    dataset = _two_source_distinct_dataset()
    train_sources = [values.copy() for values in dataset.x({"partition": "train"}, "2d", concat_source=False)]
    test_sources = [values.copy() for values in dataset.x({"partition": "test"}, "2d", concat_source=False)]
    targets = np.asarray(dataset.y({"partition": "train"})).ravel()
    prefix = [StandardScaler()] if with_preprocessing else []
    pipeline = [
        {"branch": {"by_source": True, "steps": {
            "source_0": [*prefix, {"model": Ridge(alpha=1.0)}],
            "source_1": [*prefix, {"model": Ridge(alpha=1.0)}],
        }}},
        {"merge": "auto"},
    ]
    legacy = nirs4all.run(
        pipeline, dataset, engine="legacy", workspace_path=tmp_path / "workspace",
        save_charts=False, verbose=0,
    )
    try:
        selected_best = legacy.final or legacy.best
        explicit = next(row for row in legacy.predictions.iter_entries()
                        if row.get("model_artifact_id") and row["branch_id"] != selected_best["branch_id"])
        selections = [("best", selected_best), ("explicit", explicit)]
        archives = [(name, legacy.export(tmp_path / f"{name}.n4a", **({"source": selected} if name == "explicit" else {})))
                    for name, selected in selections]
        for name, selected in selections:
            destination = tmp_path / f"{name}.n4a.py"
            destination.write_bytes(b"previous valid portable script")
            with pytest.raises(NotImplementedError, match="scripts do not support branching"):
                legacy.export(destination, format="n4a.py", **({"source": selected} if name == "explicit" else {}))
            assert destination.read_bytes() == b"previous valid portable script"
    finally:
        legacy.close()
    with WorkspaceStore(tmp_path / "workspace") as store:
        for (name, selected), (_, archive) in zip(selections, archives, strict=True):
            branch = selected["branch_id"]
            oracle = make_pipeline(StandardScaler(), Ridge(alpha=1.0)) if with_preprocessing else Ridge(alpha=1.0)
            oracle.fit(train_sources[branch], targets)
            expected = oracle.predict(test_sources[branch]).ravel()
            fitted = store.load_artifact(selected["model_artifact_id"])
            assert fitted.n_features_in_ == train_sources[branch].shape[1]
            model_x = oracle[0].transform(test_sources[branch]) if with_preprocessing else test_sources[branch]
            np.testing.assert_allclose(np.asarray(fitted.predict(model_x)).ravel(), expected, rtol=1e-5, atol=1e-5)
            inputs = tmp_path / f"{name}.npz"
            np.savez(inputs, raw_x=test_sources[branch], expected=expected)
            probe = r'''
import pathlib, sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from nirs4all.pipeline.bundle import BundleLoader
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore
archive, inputs, workspace = sys.argv[1:]
data = np.load(inputs)
blocked = pathlib.Path(workspace).resolve()
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = pathlib.Path(args[0]).resolve()
        if path == blocked or blocked in path.parents:
            raise AssertionError('Selected-source export read its workspace')
sys.addaudithook(audit)
def no_fit(*args, **kwargs):
    raise AssertionError('Selected-source export attempted FIT or opened a Store')
WorkspaceStore.__init__ = no_fit
Ridge.fit = StandardScaler.fit = no_fit
with BundleLoader(archive) as loader:
    np.testing.assert_allclose(loader.predict(data['raw_x']), data['expected'], rtol=1e-5, atol=1e-5)
'''
            subprocess.run([sys.executable, "-B", "-c", probe, str(archive), str(inputs), str(tmp_path / "workspace")], check=True)


@pytest.mark.parity
def test_no_splitter_cli_by_metadata_concat_matches_legacy_and_archive(tmp_path, monkeypatch) -> None:
    import nirs4all

    from ._dagml_cli import dagml_cli_path

    cli = dagml_cli_path()
    if not cli.exists():
        pytest.skip(f"dag-ml-cli binary not built at {cli}")
    path = dataset_path("with_metadata")
    pipeline = [
        {"branch": {"by_metadata": "group", "steps": [{"model": Ridge(alpha=1.0)}]}},
        {"merge": "concat"},
    ]
    legacy = nirs4all.run(pipeline, path, engine="legacy", save_artifacts=False, verbose=0)
    results = []
    for mode in ("1", "0"):
        monkeypatch.setenv("N4A_DAGML_INPROCESS", mode)
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
        with pytest.warns(NoSplitEvaluationWarning, match="No splitter"):
            results.append(nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0))
    dataset = DatasetConfigs(path).get_dataset_at(0)
    x_train = np.asarray(dataset.x({"partition": "train"}, layout="2d"))
    y_train = np.asarray(dataset.y({"partition": "train"}))
    x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
    y_test = np.asarray(dataset.y({"partition": "test"}))
    train_groups = np.asarray(dataset.metadata_column("group", {"partition": "train"})).ravel()
    test_groups = np.asarray(dataset.metadata_column("group", {"partition": "test"})).ravel()
    expected = np.empty(len(test_groups))
    expected_train = np.empty(len(train_groups))
    for group in np.unique(train_groups):
        model = Ridge(alpha=1.0).fit(x_train[train_groups == group], y_train[train_groups == group])
        expected[test_groups == group] = np.asarray(model.predict(x_test[test_groups == group])).ravel()
        expected_train[train_groups == group] = np.asarray(model.predict(x_train[train_groups == group])).ravel()
    expected_rmse = root_mean_squared_error(y_test, expected)
    legacy_train = [row for row in legacy.predictions.filter_predictions(load_arrays=True) if row["partition"] == "train"]
    assert len(legacy_train) == len(np.unique(train_groups))
    for result in results:
        assert np.isnan(result.cv_best_score)
        assert result.best_rmse == pytest.approx(expected_rmse, abs=1e-6)
        assert len(result._dagml_refit_artifacts) == len(np.unique(train_groups))
        assert {frame["lineage"]["phase"] for frame in result._dagml_node_results} == {"REFIT"}
        native_train = next(row for row in result.predictions.filter_predictions(load_arrays=True) if row["partition"] == "train")
        np.testing.assert_allclose(np.asarray(native_train["y_pred"]).ravel(), expected_train, atol=1e-6)
    archive = results[-1].export(tmp_path / "cli_by_metadata_full_train.n4a")
    replay = nirs4all.predict(archive, {"X": x_test, "metadata": {"group": test_groups}}, engine="legacy")
    np.testing.assert_allclose(np.asarray(replay.y_pred).ravel(), expected, atol=1e-6)
