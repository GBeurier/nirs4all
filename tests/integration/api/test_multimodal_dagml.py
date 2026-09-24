"""Public four-modality training and captured replay through real DAG-ML."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nirs4all_io import MultimodalDataset, TensorSource
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import nirs4all
from nirs4all.operators.models.multimodal import MultimodalRegressor, TensorPCA
from nirs4all.operators.models.sklearn.mbpls import MBPLS


def _cohort(*, prediction: bool = False, unequal_groups: bool = False) -> MultimodalDataset:
    rng = np.random.default_rng(181 if prediction else 79)
    rows = 5 if prediction else 16
    ids = [f"{'new' if prediction else 'train'}-{index:02d}" for index in range(rows)]
    latent = rng.normal(size=rows)
    nir = latent[:, None] + rng.normal(scale=0.2, size=(rows, 6))
    image = latent[:, None, None, None] + rng.normal(scale=0.1, size=(rows, 2, 2, 3))
    series = latent[:, None, None] + rng.normal(scale=0.15, size=(rows, 5, 2))
    # These exact sentinels identify the raw rows actually delivered to fit.
    image[:, 0, 0, 0] = np.arange(rows)
    series[:, 0, 0] = np.arange(rows)
    metadata = np.array([[float(index) / rows, "new" if prediction else "a" if index % 2 else "b"] for index in range(rows)], dtype=object)
    permutation = rng.permutation(rows)
    sources = {
        "nir": TensorSource(nir, ids, representation_id="signal_1d", axis_units={"wavelength": "nm"}, axis_coordinates={"wavelength": np.linspace(900, 1700, 6)}),
        "image": TensorSource(image[permutation], [ids[index] for index in permutation], representation_id="rgb_image"),
        "series": TensorSource(series, ids, representation_id="series_mv", axis_units={"time": "s"}, axis_coordinates={"time": np.arange(5)}),
        "metadata": TensorSource(metadata, ids, representation_id="tabular_mixed", feature_names=["age", "batch"]),
    }
    group_indices = list(np.repeat(np.arange(6), [1, 2, 3, 1, 2, 3])) + [6, 6, 7, 7] if unequal_groups else [index // 2 for index in range(rows)]
    return MultimodalDataset(
        sources, sample_ids=ids,
        y=None if prediction else 2.1 * latent + np.arange(rows) * 0.03,
        groups=None if prediction else [f"plant-{index}" for index in group_indices],
        partitions=["predict"] * rows if prediction else ["train"] * 12 + ["test"] * 4,
        name="new-raw-inputs" if prediction else "four-modality-training",
    )


def _model() -> MultimodalRegressor:
    return MultimodalRegressor(
        {
            "nir": StandardScaler(),
            "image": TensorPCA(n_components=2, random_state=19),
            "series": TensorPCA(n_components=2, random_state=19),
            "metadata": ColumnTransformer([
                ("numeric", StandardScaler(), [0]),
                ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [1]),
            ]),
        },
        model=Ridge(alpha=0.2),
        source_weights={"image": 0.7},
    )


def _run(cohort: MultimodalDataset, workspace: Path, *, model: MultimodalRegressor | None = None, grid: dict[str, list[Any]] | None = None) -> Any:
    step: dict[str, Any] = {"model": _model() if model is None else model}
    if grid is not None:
        step["_grid_"] = grid
    return nirs4all.run(
        [GroupKFold(3), step], cohort,
        engine="dag-ml", refit=True, save_artifacts=True, save_charts=False,
        verbose=0, random_state=19, workspace_path=workspace,
    )


@pytest.mark.parametrize("unequal_groups", [False, True])
def test_public_run_fits_raw_encoders_only_on_each_native_fold(unequal_groups: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cohort = _cohort(unequal_groups=unequal_groups)
    seen: list[tuple[int, frozenset[int]]] = []
    original_fit = TensorPCA.fit

    def observe_fit(self: TensorPCA, X: Any, y: Any = None) -> TensorPCA:
        values = np.asarray(X)
        seen.append((values.ndim, frozenset(int(value) for value in values.reshape(len(values), -1)[:, 0])))
        return original_fit(self, X, y)

    monkeypatch.setattr(TensorPCA, "fit", observe_fit)
    result = _run(cohort, tmp_path / "workspace")
    groups = np.asarray(cohort.groups)[:12]
    expected_fits = [frozenset(train.tolist()) for train, _ in GroupKFold(3).split(np.zeros((12, 1)), groups=groups)]
    expected_fits.append(frozenset(range(12)))
    for rank in (3, 4):
        actual = [rows for seen_rank, rows in seen if seen_rank == rank]
        assert sorted(actual, key=lambda rows: (len(rows), sorted(rows))) == sorted(expected_fits, key=lambda rows: (len(rows), sorted(rows)))
    assert all(not rows & {12, 13, 14, 15} for _, rows in seen)
    assert np.isfinite(result.best_rmse)
    assert result._dagml_score_set is not None
    artifacts = result._dagml_refit_artifacts
    assert len(artifacts) == 1
    saved = artifacts[0]["estimator"]._model
    assert set(saved.source_names_) == {"nir", "image", "series", "metadata"}
    assert artifacts[0]["estimator"].source_names == saved.source_names_
    assert saved.input_shapes_["image"] == (2, 2, 3)
    assert saved.input_shapes_["series"] == (5, 2)
    assert saved.transformers_["image"].pca_.n_samples_ == 12
    assert saved.transformers_["series"].pca_.n_samples_ == 12
    np.testing.assert_allclose(saved.transformers_["nir"].mean_, cohort.sources["nir"].values[:12].mean(axis=0), atol=1e-12)


def test_public_archive_replays_reordered_raw_sources_in_fresh_process_without_fit(tmp_path: Path) -> None:
    workspace = tmp_path / "training-workspace"
    trained = _run(_cohort(), workspace)
    prediction = _cohort(prediction=True)
    captured = trained._dagml_refit_artifacts[0]["estimator"]
    expected = captured.predict([prediction.sources[name].values for name in captured.source_names])
    archive = trained.export(tmp_path / "multimodal.n4a")
    assert archive.is_file()
    clean = tmp_path / "clean-process"
    clean.mkdir()
    inputs = {
        "sample_ids": list(prediction.sample_ids),
        "sources": [
            {"name": name, "values": source.values.tolist(), "representation_id": source.representation_id,
             "sample_ids": list(source.sample_ids), "dtype": str(source.values.dtype), "feature_names": source.feature_names,
             "axis_units": dict(source.axis_units), "axis_coordinates": dict(source.axis_coordinates)}
            for name, source in reversed(list(prediction.sources.items()))
        ],
    }
    (clean / "inputs.json").write_text(json.dumps(inputs), encoding="utf-8")
    shutil.rmtree(workspace)
    assert not workspace.exists()
    script = r'''
import json
import sys
import numpy as np
import nirs4all
from nirs4all_io import MultimodalDataset, TensorSource
from nirs4all.operators.models.multimodal import MultimodalRegressor, TensorPCA
from nirs4all.pipeline import PipelineRunner
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def forbidden(*args, **kwargs):
    raise AssertionError("archive prediction attempted training or legacy execution")

for cls in (MultimodalRegressor, TensorPCA, PCA, Ridge, OneHotEncoder, StandardScaler):
    cls.fit = forbidden
PipelineRunner.run = forbidden
payload = json.loads(open("inputs.json", encoding="utf-8").read())
sources = {
    source["name"]: TensorSource(
        np.asarray(source["values"], dtype=source["dtype"]), source["sample_ids"], representation_id=source["representation_id"],
        feature_names=source["feature_names"], axis_units=source["axis_units"], axis_coordinates=source["axis_coordinates"],
    )
    for source in payload["sources"]
}
cohort = MultimodalDataset(sources, sample_ids=payload["sample_ids"], partitions=["predict"] * len(payload["sample_ids"]))
result = nirs4all.predict(model=sys.argv[1], data=cohort)
assert result.metadata["engine"] == "dag-ml"
assert result.metadata["phase"] == "PREDICT"
assert result.metadata["training_performed"] is False
assert result.metadata["sample_ids"] == payload["sample_ids"]
assert result.metadata["scores"] is None
assert result.metadata["artifact_integrity_verified"] is True
assert {node["lineage"]["phase"] for node in result.metadata["node_results"]} == {"PREDICT"}
np.save("predictions.npy", result.y_pred)
'''
    env = dict(os.environ)
    # Preserve dependency paths while making a different cwd meaningful.
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.pathsep.join(str(Path(entry).resolve()) for entry in env["PYTHONPATH"].split(os.pathsep) if entry)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(archive.resolve())], cwd=clean, env=env,
        capture_output=True, text=True, timeout=90, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    np.testing.assert_allclose(np.load(clean / "predictions.npy").reshape(-1), np.asarray(expected).reshape(-1), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("corruption", ["missing_source", "changed_image_shape", "changed_units", "changed_coordinates", "changed_feature_names"])
def test_public_replay_refuses_incompatible_raw_sources(corruption: str, tmp_path: Path) -> None:
    trained = _run(_cohort(), tmp_path / "workspace")
    archive = trained.export(tmp_path / "multimodal.n4a")
    prediction = _cohort(prediction=True)
    sources = dict(prediction.sources)
    if corruption == "missing_source":
        del sources["image"]
    elif corruption == "changed_image_shape":
        image = sources["image"]
        sources["image"] = TensorSource(image.values[:, :1], image.sample_ids, representation_id="rgb_image")
    elif corruption in {"changed_units", "changed_coordinates"}:
        nir = sources["nir"]
        sources["nir"] = TensorSource(
            nir.values, nir.sample_ids, representation_id="signal_1d",
            axis_units={"wavelength": "um" if corruption == "changed_units" else "nm"},
            axis_coordinates={"wavelength": np.linspace(901 if corruption == "changed_coordinates" else 900, 1700, 6)},
        )
    else:
        metadata = sources["metadata"]
        sources["metadata"] = TensorSource(metadata.values, metadata.sample_ids, representation_id="tabular_mixed", feature_names=["different_age", "batch"])
    invalid = MultimodalDataset(sources, sample_ids=prediction.sample_ids, partitions=["predict"] * len(prediction))
    with pytest.raises(Exception, match="(?i)(source|shape|schema mismatch)"):
        nirs4all.predict(model=archive, data=invalid)


def test_native_grid_applies_eight_variants_and_selection_ignores_final_test_labels(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Native grid enumeration uses canonical alphabetical parameter order.
    grid: dict[str, list[Any]] = {
        "model__alpha": [0.1, 1.0],
        "source_weights__image": [0.5, 1.0],
        "transformers__image__n_components": [1, 2],
    }
    expected_combinations = set(product(*grid.values()))
    fit_calls: list[tuple[tuple[float, float, int], int]] = []
    original_fit = MultimodalRegressor.fit

    def observe_fit(self: MultimodalRegressor, X: Any, y: Any) -> MultimodalRegressor:
        params = self.get_params(deep=True)
        applied = (float(params["model__alpha"]), float(params["source_weights__image"]), int(params["transformers__image__n_components"]))
        fitted = original_fit(self, X, y)
        assert fitted.model_.alpha == applied[0]
        assert fitted.source_weights_["image"] == applied[1]
        assert fitted.transformers_["image"].n_components_ == applied[2]
        fit_calls.append((applied, len(y)))
        return fitted

    monkeypatch.setattr(MultimodalRegressor, "fit", observe_fit)
    training = _cohort(unequal_groups=True)
    first = _run(training, tmp_path / "baseline", grid=grid)
    baseline_calls = list(fit_calls)
    refits = [params for params, rows in baseline_calls if rows == 12]
    assert len(refits) == 1
    cv_calls = Counter(params for params, rows in baseline_calls if rows == 8)
    assert set(cv_calls) == expected_combinations
    assert all(count == 3 for params, count in cv_calls.items() if params != refits[0])
    # The current native selection path reruns the winner's CV before REFIT.
    # Permit its complete fold repetitions without requiring redundant work.
    assert cv_calls[refits[0]] >= 3 and cv_calls[refits[0]] % 3 == 0

    reports = [report for report in first._dagml_score_set["reports"] if report["partition"] == "validation" and report.get("fold_id") not in {"avg", "w_avg"}]
    assert len(reports) == 24
    variants = {report["variant_id"] for report in reports}
    assert len(variants) == 8
    scores = {}
    for variant in variants:
        folds = [report for report in reports if report["variant_id"] == variant]
        assert {report["fold_id"] for report in folds} == {"fold0", "fold1", "fold2"}
        assert all(np.isfinite(report["metrics"]["rmse"]) for report in folds)
        scores[variant] = np.sqrt(sum(report["metrics"]["mse"] * report["row_count"] for report in folds) / sum(report["row_count"] for report in folds))
    refit_lineages = [node["lineage"] for node in first._dagml_node_results if node.get("lineage", {}).get("phase") == "REFIT"]
    assert len(refit_lineages) == 1
    selected = refit_lineages[0]["variant_id"]
    assert selected == min(scores, key=lambda variant: scores[variant])
    assert first.cv_best_score == pytest.approx(scores[selected])
    saved = first._dagml_refit_artifacts[0]["estimator"]._model
    assert (saved.model_.alpha, saved.source_weights_["image"], saved.transformers_["image"].n_components_) == refits[0]

    changed_y = np.array(training.y, copy=True)
    changed_y[12:] += np.array([1000, -2000, 3000, -4000])
    changed = MultimodalDataset(training.sources, sample_ids=training.sample_ids, y=changed_y, groups=training.groups, partitions=training.partitions, name=training.name)
    fit_calls.clear()
    second = _run(changed, tmp_path / "perturbed-test-labels", grid=grid)
    assert fit_calls == baseline_calls
    second_reports = [report for report in second._dagml_score_set["reports"] if report["partition"] == "validation" and report.get("fold_id") not in {"avg", "w_avg"}]
    assert {(report["variant_id"], report["fold_id"]): report["metrics"]["rmse"] for report in second_reports} == pytest.approx({(report["variant_id"], report["fold_id"]): report["metrics"]["rmse"] for report in reports})
    second_refit = [node["lineage"]["variant_id"] for node in second._dagml_node_results if node.get("lineage", {}).get("phase") == "REFIT"]
    assert second_refit == [selected]
    assert second.best_rmse != pytest.approx(first.best_rmse)
    prediction = _cohort(prediction=True)
    first_archive = first.export(tmp_path / "selected.n4a")
    second_archive = second.export(tmp_path / "selected-perturbed.n4a")
    np.testing.assert_allclose(
        nirs4all.predict(first_archive, prediction).y_pred,
        nirs4all.predict(second_archive, prediction).y_pred,
        rtol=1e-12, atol=1e-12,
    )


def test_intermediate_mbpls_receives_four_encoded_blocks_and_replays_without_fit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = _model().set_params(model=MBPLS(n_components=2, standardize=False), fusion="intermediate")
    observed: list[tuple[tuple[int, ...], ...]] = []
    original_fit = MBPLS.fit

    def observe_fit(self: MBPLS, X: Any, y: Any) -> MBPLS:
        assert isinstance(X, list)
        assert len(X) == 4
        assert all(np.asarray(block).ndim == 2 for block in X)
        observed.append(tuple(np.shape(block) for block in X))
        original_fit(self, X, y)
        return self

    monkeypatch.setattr(MBPLS, "fit", observe_fit)
    result = _run(_cohort(unequal_groups=True), tmp_path / "intermediate", model=model)
    assert len(observed) == 4
    assert sorted(shapes[0][0] for shapes in observed) == [8, 8, 8, 12]
    captured = result._dagml_refit_artifacts[0]["estimator"]
    saved = captured._model
    assert saved.fusion_ == "intermediate"
    assert saved.model_._is_multiblock is True
    assert saved.model_._block_sizes == [saved.output_widths_[name] for name in saved.source_names_]
    prediction = _cohort(prediction=True)
    expected = captured.predict([prediction.sources[name].values for name in captured.source_names])
    archive = result.export(tmp_path / "intermediate.n4a")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("intermediate archive replay attempted fit")

    monkeypatch.setattr(MBPLS, "fit", forbidden)
    monkeypatch.setattr(MultimodalRegressor, "fit", forbidden)
    monkeypatch.setattr(TensorPCA, "fit", forbidden)
    replay = nirs4all.predict(archive, prediction)
    assert replay.metadata["engine"] == "dag-ml"
    assert replay.metadata["phase"] == "PREDICT"
    assert replay.metadata["training_performed"] is False
    np.testing.assert_allclose(replay.y_pred.reshape(-1), expected.reshape(-1), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("damage", ["dependency_version", "model_payload"])
def test_multimodal_archive_refuses_incompatibility_before_deserializing(damage: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import joblib

    result = _run(_cohort(), tmp_path / "workspace")
    archive = result.export(tmp_path / "original.n4a")
    with zipfile.ZipFile(archive) as bundle:
        members = {name: bundle.read(name) for name in bundle.namelist()}
    manifest = json.loads(members["manifest.json"])
    contract = manifest["multimodal_host"]
    assert set(contract["input_schema"]) == {"nir", "image", "series", "metadata"}
    assert all(source["shape"][0] is None for source in contract["input_schema"].values())
    assert {"nirs4all", "nirs4all-io", "dag-ml", "dag-ml-data", "numpy", "scikit-learn", "joblib"} <= set(contract["dependencies"])
    if damage == "dependency_version":
        contract["dependencies"]["numpy"] = "0.0.0-incompatible"
        members["manifest.json"] = json.dumps(manifest).encode()
        expected_error: type[Exception] = ImportError
        message = "requires numpy==0.0.0-incompatible"
    else:
        artifact_name = next(name for name in members if name.startswith("artifacts/") and name.endswith(".joblib"))
        members[artifact_name] += b"corrupted-model-payload"
        expected_error = ValueError
        message = "artifact content fingerprint mismatch"
    damaged = tmp_path / "damaged.n4a"
    with zipfile.ZipFile(damaged, "w") as bundle:
        for name, contents in members.items():
            bundle.writestr(name, contents)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("incompatible or corrupted archive reached joblib.load")

    monkeypatch.setattr(joblib, "load", forbidden)
    with pytest.raises(expected_error, match=message):
        nirs4all.predict(damaged, _cohort(prediction=True))
