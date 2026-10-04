"""Captured late-fusion replay preserves zero/presence semantics and layout."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from nirs4all.api.result import _DagmlExportedModel, _DagmlNativeStackingModel
from nirs4all.pipeline.bundle import write_single_model_bundle
from nirs4all.pipeline.dagml.general_archive import load_general_archive
from nirs4all.pipeline.dagml.multimodal_contracts import archive_metadata


def _fitted_stack(width: int = 1) -> _DagmlNativeStackingModel:
    x = np.arange(8, dtype=float).reshape(-1, 1)
    bases = []
    names = ("z_signal", "a_image")
    for index, name in enumerate(names):
        y = np.column_stack([100.0 * (index + 1) + (column + 2) * x[:, 0] for column in range(width)])
        transform = StandardScaler().fit(y)
        model = Ridge(alpha=0.2).fit(x, transform.transform(y))
        model.multimodal_source_name = name
        model.multimodal_missing_source_policy = "zero_with_indicator"
        model.multimodal_prediction_width = width
        bases.append(_DagmlExportedModel(model, transform))
    features = np.random.default_rng(403).normal(size=(12, len(names) * (width + 1)))
    meta = Ridge(alpha=0.4).fit(features, features[:, :width])
    meta.multimodal_missing_source_policy = "zero_with_indicator"
    meta.multimodal_source_names = names
    return _DagmlNativeStackingModel(bases, _DagmlExportedModel(meta, None), names)


@pytest.mark.parametrize("width", [1, 2])
def test_absent_base_is_zero_after_inverse_target_transform_and_never_called(
    width: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _fitted_stack(width)
    signal = np.asarray([[2.0], [np.nan], [4.0]])
    image = np.full((3, 1), np.nan)
    masks = {"a_image": np.zeros(3, dtype=bool), "z_signal": np.asarray([True, False, True])}
    expected = np.zeros((3, 2 * (width + 1)))
    expected[[0, 2], :width] = model.base_members[0].predict_numeric(signal[[0, 2]])
    expected[:, width] = masks["z_signal"]
    monkeypatch.setattr(model.base_members[1].estimator, "predict", lambda *_: pytest.fail("all-absent source reached its estimator"))
    actual = model._meta_features([signal, image], source_masks=masks)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(model.predict_numeric([signal, image], source_masks=masks), model.meta_member.predict_numeric(expected))


def test_presence_columns_remain_when_prediction_inputs_are_complete() -> None:
    model = _fitted_stack(2)
    blocks = [np.asarray([[2.0], [4.0]]), np.asarray([[3.0], [5.0]])]
    features = model._meta_features(blocks)
    assert features.shape == (2, 6)
    np.testing.assert_array_equal(features[:, [2, 5]], np.ones((2, 2)))
    np.testing.assert_array_equal(features[:, :2], model.base_members[0].predict_numeric(blocks[0]))
    np.testing.assert_array_equal(features[:, 3:5], model.base_members[1].predict_numeric(blocks[1]))


@pytest.mark.parametrize("corruption", ["policy", "order", "width", "meta_width", "base_name"])
def test_incoherent_fitted_layout_is_refused_before_prediction(corruption: str) -> None:
    model = _fitted_stack()
    if corruption == "policy":
        del model.base_members[0].estimator.multimodal_missing_source_policy
    elif corruption == "order":
        model.meta_member.estimator.multimodal_source_names = tuple(reversed(model.source_names))
    elif corruption == "width":
        model.base_members[0].estimator.multimodal_prediction_width = True
    elif corruption == "meta_width":
        model.meta_member.estimator.n_features_in_ = 2
    else:
        model.base_members[0].estimator.multimodal_source_name = "wrong"
    with pytest.raises(ValueError, match="native stacking"):
        # Invalid state must fail before even examining malformed feature inputs.
        model.predict(None)
    with pytest.raises(ValueError, match="native stacking"):
        _DagmlNativeStackingModel(model.base_members, model.meta_member, model.source_names)


def test_strict_stacking_keeps_prediction_only_columns_and_refuses_masks() -> None:
    x = np.arange(8, dtype=float).reshape(-1, 1)
    members = [_DagmlExportedModel(Ridge(alpha=alpha).fit(x, x[:, 0]), None) for alpha in (0.2, 0.4)]
    features = np.column_stack([member.predict_numeric(x) for member in members])
    meta = _DagmlExportedModel(Ridge(alpha=0.5).fit(features, x[:, 0]), None)
    model = _DagmlNativeStackingModel(members, meta, ["signal", "image"])
    np.testing.assert_array_equal(model._meta_features([x, x]), features)
    with pytest.raises(ValueError, match="zero_with_indicator"):
        model.predict([x, x], source_masks={"signal": np.ones(8, dtype=bool), "image": np.ones(8, dtype=bool)})


def _rewrite_manifest(source: Path, destination: Path, change: Any) -> Path:
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(destination, "w") as modified:
        for item in original.infolist():
            data = original.read(item.filename)
            if item.filename == "manifest.json":
                manifest = json.loads(data)
                change(manifest)
                data = json.dumps(manifest).encode()
            modified.writestr(item, data)
    return destination


@pytest.mark.parametrize("field", ["missing_source_policy", "source_names", "prediction_widths", "meta_feature_layout", "missing_contract", "recipe"])
def test_archive_manifest_must_attest_exact_fitted_presence_layout(tmp_path: Path, field: str) -> None:
    model = _fitted_stack()
    model.multimodal_input_schema = {name: {"source_id": name} for name in model.source_names}
    archive = write_single_model_bundle(
        model, tmp_path / "original.n4a", model_label="late_missing",
        provenance={"source_type": "dagml_native", **archive_metadata(model)},
    )
    loaded = load_general_archive(archive)
    assert loaded["artifact_integrity_verified"]
    layout = loaded["manifest"]["multimodal_host"]["source_presence"]
    assert layout["source_names"] == ["z_signal", "a_image"]
    assert layout["prediction_widths"] == [1, 1]
    assert layout["meta_feature_width"] == 4

    def corrupt(manifest: dict[str, Any]) -> None:
        host = manifest["multimodal_host"]
        if field == "missing_contract":
            del host["source_presence"]
        elif field == "recipe":
            host["selected_model"]["missing_source_policy"] = "error"
        else:
            host["source_presence"][field] = "corrupted"

    changed = _rewrite_manifest(archive, tmp_path / "changed.n4a", corrupt)
    with pytest.raises(ValueError, match="native stacking archive"):
        load_general_archive(changed)


@pytest.mark.parametrize("width, all_series_absent", [(1, False), (1, True), (2, False), (2, True)])
def test_native_late_missing_export_replays_new_rows_without_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int, all_series_absent: bool,
) -> None:
    from nirs4all_io import DataProvider, MultimodalDataset
    from sklearn.preprocessing import OneHotEncoder

    import nirs4all
    from nirs4all.operators.models.multimodal import TensorPCA
    from nirs4all.operators.transforms import SequenceSummary
    from tests.integration.api.test_multimodal_late_missing import _meta, _run
    from tests.integration.api.test_multimodal_ragged import _ragged_cohort, _replace_series

    cohort = _ragged_cohort(missing=True)
    if width == 2:
        cohort = MultimodalDataset(
            cohort.sources, sample_ids=cohort.sample_ids, y=np.column_stack([cohort.y, -2 * cohort.y + 30]),
            target_names=["first", "second"], task_type="regression", groups=cohort.groups,
            partitions=cohort.partitions,
        )
    workspace = tmp_path / "training"
    result = _run(cohort, workspace)
    try:
        child = _meta(result)
        artifacts = child._dagml_refit_artifacts
        meta = next(item["estimator"] for item in artifacts if item["controller_id"] == "controller:nirs4all.meta_model")
        bases = {item["estimator"].multimodal_source_name: item for item in artifacts
                 if getattr(item["estimator"], "multimodal_source_name", None) is not None}
        assert set(bases) == set(meta.multimodal_source_names)
        new = _ragged_cohort(prediction=True, missing=True)
        source = new.sources["series"]
        mask = np.zeros(len(new), dtype=bool) if all_series_absent else source.presence_mask
        values = source.values.values.copy()
        for row in np.flatnonzero(~mask):
            values[source.offsets[row]:source.offsets[row + 1]] = np.nan
        new = _replace_series(new, values=values, presence_mask=mask)
        new = new.take([new.sample_ids[index] for index in [4, 1, 3, 0]])
        new = MultimodalDataset(
            new.sources, sample_ids=new.sample_ids, target_names=cohort.target_names,
            partitions=new.partitions, task_type="regression",
        )
        expected_parts = []
        for name in meta.multimodal_source_names:
            source = new.sources[name]
            presence = source.presence_mask
            part = np.zeros((len(new), width))
            if presence.any():
                block = source.values
                observed = block.take_rows(presence) if hasattr(block, "take_rows") else block[presence]
                captured = bases[name]
                member = _DagmlExportedModel(captured["estimator"], captured["y_transform"], late_partial_refit_artifact=captured)
                part[presence] = member.predict_numeric(observed).reshape(int(presence.sum()), width)
            expected_parts.append(np.column_stack([part, presence.astype(float)]))
        expected = meta.predict(np.column_stack(expected_parts))

        def forbidden(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("export/replay executed fitting, generation or legacy scheduling")

        for cls in (Ridge, StandardScaler, OneHotEncoder, TensorPCA, SequenceSummary):
            monkeypatch.setattr(cls, "fit", forbidden)
        monkeypatch.setattr(DataProvider, "materialize", forbidden)
        monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", forbidden)
        if all_series_absent:
            monkeypatch.setattr(SequenceSummary, "transform", forbidden)
        archive = child.export(tmp_path / "late-missing.n4a")
        with zipfile.ZipFile(archive) as bundle:
            layout = json.loads(bundle.read("manifest.json"))["multimodal_host"]["source_presence"]
        assert layout["source_names"] == list(meta.multimodal_source_names)
        assert layout["prediction_widths"] == [width] * len(bases)
        assert layout["meta_feature_width"] == len(bases) * (width + 1)
    finally:
        result.close()
    shutil.rmtree(workspace)
    reordered = MultimodalDataset(
        dict(reversed(list(new.sources.items()))), sample_ids=new.sample_ids,
        target_names=cohort.target_names, partitions=new.partitions, task_type="regression",
    )
    with pytest.raises(ValueError, match="original target names and ordered named sources"):
        nirs4all.predict(archive, reordered)
    replay = nirs4all.predict(archive, MultimodalDataset.from_dict(new.to_dict()))
    np.testing.assert_array_equal(np.asarray(replay.y_pred).reshape(len(new), width), np.asarray(expected).reshape(len(new), width))
    assert replay.metadata["phase"] == "PREDICT"
    assert replay.metadata["sample_ids"] == list(new.sample_ids)
    assert replay.metadata["training_performed"] is False
    assert replay.metadata["artifact_integrity_verified"] is True
    assert replay.metadata["scores"] is None


def test_partial_source_component_archive_export_refuses_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nirs4all.pipeline.dagml.rt import RtError
    from tests.integration.api.test_multimodal_late_missing import _run
    from tests.integration.api.test_multimodal_ragged import _ragged_cohort

    result = _run(_ragged_cohort(missing=True), tmp_path / "training")
    try:
        source_run = next(run for run in result.runs if len(run._dagml_refit_artifacts) == 1
                          and run._dagml_refit_artifacts[0]["late_partial_refit_origin"]["source_name"] is not None)
        output = tmp_path / "must-not-exist" / "component.n4a"

        def forbidden(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("unsupported standalone partial component reached the bundle writer")

        monkeypatch.setattr("nirs4all.pipeline.bundle.write_single_model_bundle", forbidden)
        with pytest.raises(RtError, match="partial-source component cannot be exported alone"):
            source_run.export(output)
        assert not output.exists()
        assert not output.parent.exists()
    finally:
        result.close()
