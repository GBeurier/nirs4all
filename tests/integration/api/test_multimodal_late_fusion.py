"""Raw source branches feed identity-aligned, native nested OOF stacking."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nirs4all_io import MultimodalDataset, TensorSource
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import nirs4all
from nirs4all.operators.models.multimodal import TensorPCA


def _cohort(*, poison: list[int] | None = None, prediction: bool = False) -> MultimodalDataset:
    rng = np.random.default_rng(403 if prediction else 401)
    size = 7 if prediction else 30
    ids = tuple(f"{'new' if prediction else 'sample'}-{index:02d}" for index in range(size))
    latent = rng.normal(size=(size, 3))
    nir = latent[:, :1] + rng.normal(scale=0.2, size=(size, 4))
    images = latent[:, 1, None, None, None] + rng.normal(scale=0.2, size=(size, 2, 2, 3))
    series = latent[:, 2, None, None] + rng.normal(scale=0.2, size=(size, 5, 2))
    # Exact row sentinels let the test inspect the real encoder fit scopes.
    images[:, 0, 0, 0] = np.arange(size)
    series[:, 0, 0] = np.arange(size)
    metadata = np.asarray([[float(index), "new" if prediction else "a" if index % 2 else "b"] for index in range(size)], dtype=object)
    permutation = rng.permutation(size)
    sources = {
        "nir": TensorSource(nir, ids, representation_id="signal_1d", axis_units={"wavelength": "nm"}, axis_coordinates={"wavelength": [900, 1000, 1100, 1200]}),
        "image": TensorSource(images[permutation], [ids[index] for index in permutation], representation_id="rgb_image"),
        "series": TensorSource(series, ids, representation_id="series_mv", axis_units={"time": "s"}, axis_coordinates={"time": [0, 1, 2, 3, 4]}),
        "metadata": TensorSource(metadata, ids, representation_id="tabular_mixed", feature_names=["age", "cultivar"]),
    }
    y = latent @ [2.0, 3.0, -1.5] + np.arange(size) * 0.01
    if poison is not None:
        y[poison] += 100_000
    groups = [f"plant-{group}" for group in np.repeat(np.arange(12), [1, 2, 3, 4, 1, 2, 3, 4, 1, 3, 3, 3])]
    return MultimodalDataset(
        sources, sample_ids=ids, y=None if prediction else y,
        groups=None if prediction else groups,
        partitions=["predict"] * size if prediction else ["train"] * 24 + ["test"] * 6,
        name="new-late-fusion-inputs" if prediction else "raw-late-fusion",
    )


def _pipeline() -> list[Any]:
    # Source declaration order deliberately differs from the cohort order.
    branches = {
        "series": [TensorPCA(n_components=2), Ridge(alpha=0.3)],
        "metadata": [ColumnTransformer([
            ("numeric", StandardScaler(), [0]),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [1]),
        ]), Ridge(alpha=0.4)],
        "image": [TensorPCA(n_components=2), Ridge(alpha=0.2)],
        "nir": [StandardScaler(), Ridge(alpha=0.1)],
    }
    return [GroupKFold(3), {"branch": {"by_source": True, "steps": branches}}, {"merge": "predictions"}, Ridge(alpha=0.5)]


def _run(cohort: MultimodalDataset, workspace: Path) -> Any:
    return nirs4all.run(
        _pipeline(), cohort, engine="dag-ml", refit=True, save_artifacts=True,
        save_charts=False, verbose=0, random_state=17, workspace_path=workspace,
    )


def test_native_late_fusion_preserves_raw_ranks_identity_and_inner_groups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nirs4all.pipeline.dagml import run_paths

    cohort = _cohort()
    fits: list[tuple[int, frozenset[int]]] = []
    campaigns: list[dict[str, Any]] = []
    original_fit = TensorPCA.fit
    original_run = run_paths.run_cv_refit_bundle

    def record_fit(self: TensorPCA, X: Any, y: Any = None) -> TensorPCA:
        values = np.asarray(X)
        fits.append((values.ndim, frozenset(int(value) for value in values.reshape(len(values), -1)[:, 0])))
        return original_fit(self, X, y)

    def record_campaign(*args: Any, **kwargs: Any) -> Any:
        campaigns.append({"dsl": kwargs["dsl"], "graph": kwargs["graph"]})
        return original_run(*args, **kwargs)

    monkeypatch.setattr(TensorPCA, "fit", record_fit)
    monkeypatch.setattr(run_paths, "run_cv_refit_bundle", record_campaign)
    monkeypatch.setattr(run_paths, "_run_model_on_precomputed_matrix", lambda *args, **kwargs: pytest.fail("Python CV loop executed"))
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("legacy scheduler executed"))
    result = _run(cohort, tmp_path / "training")
    assert len(campaigns) == 1
    assert campaigns[0]["dsl"]["inner_cv"] == {"kind": "group_kfold", "n_splits": 2}
    assert len(result.runs) == 5
    assert {rank for rank, _ in fits} == {3, 4}
    assert any(len(rows) < 16 for _, rows in fits)
    assert all(rows <= set(range(24)) for _, rows in fits)
    group_rows = {
        group: {index for index, value in enumerate(cohort.groups[:24]) if value == group}
        for group in cohort.groups[:24]
    }
    for _, rows in fits:
        assert all(not rows.intersection(members) or members <= rows for members in group_rows.values())
    layout = result.per_dataset[cohort.name]["source_stacking"]["layout"]
    assert layout["kind"] == "typed_source_blocks"
    assert [source["source_name"] for source in layout["sources"]] == list(cohort.sources)
    assert "total_columns" not in layout
    model_nodes = [node for node in campaigns[0]["graph"]["nodes"] if node["kind"] == "model" and node["id"] != "merge:stack"]
    assert {node["metadata"]["source_name"] for node in model_nodes} == set(cohort.sources)
    outer_folds = list(GroupKFold(3).split(np.zeros((24, 1)), groups=cohort.groups[:24]))
    for child in result.runs:
        assert np.isfinite(child.cv_best_score)
        for fold, (_training, validation) in enumerate(outer_folds):
            rows = child.predictions.filter_predictions(fold_id=str(fold), partition="val", load_arrays=True)
            assert len(rows) == 1
            assert set(rows[0]["sample_indices"]) == set(validation)
    meta = result.runs[-1]
    fitted_meta = next(item["estimator"] for item in meta._dagml_refit_artifacts if item["controller_id"] == "controller:nirs4all.meta_model")
    assert fitted_meta.n_features_in_ == 4
    assert result._dagml_score_set is not None
    # Keep a small authoritative handoff for inspecting native export bindings.
    (tmp_path / "late-fusion-evidence.json").write_text(json.dumps({
        "layout": layout, "nodes": model_nodes,
        "artifacts": [{key: value for key, value in item.items() if key not in {"estimator", "y_transform"}} for item in meta._dagml_refit_artifacts],
    }, default=str), encoding="utf-8")
    result.close()


def test_late_fusion_outer_validation_targets_never_enter_its_meta_fit(tmp_path: Path) -> None:
    cohort = _cohort()
    validation = next(GroupKFold(3).split(np.zeros((24, 1)), groups=cohort.groups[:24]))[1]
    before = _run(cohort, tmp_path / "before")
    after = _run(_cohort(poison=validation.tolist()), tmp_path / "poisoned")
    for left, right in zip(before.runs, after.runs, strict=True):
        expected = left.predictions.filter_predictions(fold_id="0", partition="val", load_arrays=True)[0]
        actual = right.predictions.filter_predictions(fold_id="0", partition="val", load_arrays=True)[0]
        assert expected["sample_indices"] == actual["sample_indices"]
        np.testing.assert_array_equal(expected["y_pred"], actual["y_pred"])
    before.close()
    after.close()


def test_late_fusion_archive_replays_new_raw_sources_without_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cohort = _cohort()
    result = _run(cohort, tmp_path / "workspace")
    original = _cohort(prediction=True).take(["new-05", "new-01", "new-06", "new-00"])
    predicted = MultimodalDataset(
        dict(reversed(list(original.sources.items()))), sample_ids=original.sample_ids,
        partitions=["predict"] * len(original), name="new-reordered-sources",
    )
    base_predictions = [
        child._dagml_refit_artifacts[0]["estimator"].predict(predicted.sources[name].values)
        for name, child in zip(cohort.sources, result.runs[:4], strict=True)
    ]
    meta = result.runs[-1]
    fitted_meta = next(item["estimator"] for item in meta._dagml_refit_artifacts if item["controller_id"] == "controller:nirs4all.meta_model")
    expected = fitted_meta.predict(np.column_stack(base_predictions))

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("export or replay fitted an encoder/model or ran legacy scheduling")

    for cls in (Ridge, TensorPCA, OneHotEncoder, StandardScaler):
        monkeypatch.setattr(cls, "fit", forbidden)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", forbidden)
    archive = meta.export(tmp_path / "raw-late-fusion.n4a")
    result.close()
    shutil.rmtree(tmp_path / "workspace")
    replay = nirs4all.predict(archive, predicted)
    np.testing.assert_array_equal(replay.y_pred, expected)
    assert replay.metadata["sample_ids"] == list(predicted.sample_ids)
    assert replay.metadata["phase"] == "PREDICT"
    assert replay.metadata["training_performed"] is False
    assert replay.metadata["artifact_integrity_verified"] is True
    assert replay.metadata["scores"] is None


def test_late_fusion_requires_enough_outer_training_groups_before_fitting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = _cohort()
    cohort = MultimodalDataset(
        original.sources, sample_ids=original.sample_ids, y=original.y,
        partitions=original.partitions, groups=["first"] * 12 + ["second"] * 12 + ["test"] * 6,
    )
    pipeline = _pipeline()
    pipeline[0] = GroupKFold(2)
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("invalid group plan reached fitting"))
    monkeypatch.setattr(TensorPCA, "fit", lambda *args, **kwargs: pytest.fail("invalid group plan reached encoding"))
    with pytest.raises(Exception, match="at least two groups"):
        nirs4all.run(pipeline, cohort, engine="dag-ml", workspace_path=tmp_path, save_artifacts=False, save_charts=False, verbose=0)


def test_late_fusion_rejects_wrong_source_names_before_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _pipeline()
    branches = pipeline[1]["branch"]["steps"]
    branches["wrong-source"] = branches.pop("nir")
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("invalid source binding reached fitting"))
    with pytest.raises(ValueError, match="names must exactly match"):
        nirs4all.run(pipeline, _cohort(), engine="dag-ml", workspace_path=tmp_path, save_artifacts=False, save_charts=False, verbose=0)


def test_nested_transformer_with_local_callable_is_refused_before_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _pipeline()
    pipeline[1]["branch"]["steps"]["metadata"][0] = ColumnTransformer([("unsafe", FunctionTransformer(lambda X: X), [0])])
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("unreconstructible transformer reached fitting"))
    with pytest.raises(Exception, match="cannot route ColumnTransformer"):
        nirs4all.run(pipeline, _cohort(), engine="dag-ml", workspace_path=tmp_path, save_artifacts=False, save_charts=False, verbose=0)
