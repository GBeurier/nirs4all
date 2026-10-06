"""Host and public execution regressions for DGA-01 through DGA-05 and DGC-03.

Exercise real dataset mutation, expansion, dispatch and fold construction;
stop at the native execution boundary in host tests and run the public DAG lane.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nirs4all.api  # noqa: F401 - initialize public exports before importing backend modules
from nirs4all.api.result import RunResult
from nirs4all.data import SpectroDataset
from nirs4all.operators.augmentation import GaussianAdditiveNoise
from nirs4all.operators.models.meta import MetaModel
from nirs4all.pipeline.dagml import run_backend, run_paths
from nirs4all.pipeline.dagml.dataset import _materialize_dataset
from nirs4all.pipeline.dagml.errors import DagMlUnsupported
from nirs4all.pipeline.dagml.folds import _build_folds, lower_fold_file_holdout
from nirs4all.pipeline.dagml.result import _native_variant_config_map


@pytest.fixture
def spectro() -> SpectroDataset:
    rng = np.random.default_rng(19)
    features = rng.normal(size=(24, 6))
    dataset = SpectroDataset("audit-run")
    dataset.add_samples(features, {"partition": "train"})
    dataset.add_targets(features[:, 0] - features[:, 1])
    dataset.add_metadata(np.repeat(np.arange(6), 4), headers=["group"])
    dataset.set_folds([(list(range(12)), list(range(12, 24)))])
    return dataset


@pytest.mark.parametrize("levels", [1, 2])
def test_native_stacking_consumes_only_prediction_features(spectro: SpectroDataset, levels: int) -> None:
    import nirs4all

    pipeline: list[Any] = [
        KFold(3),
        {"model": Ridge(1)},
        {"model": Ridge(2)},
        {"model": MetaModel(model=Ridge(0.1)), "name": "FirstMeta"},
    ]
    if levels == 2:
        pipeline.append({"model": MetaModel(model=Ridge(0.2), source_models=["FirstMeta"]), "name": "SecondMeta"})
    result = nirs4all.run(pipeline, spectro, engine="dag-ml", refit=True, save_artifacts=False, verbose=0, save_charts=False)
    assert np.isfinite(result.cv_best_score)
    meta_artifacts = [artifact for artifact in result._dagml_refit_artifacts if artifact["controller_id"] == "controller:nirs4all.meta_model"]
    assert len(meta_artifacts) == levels
    # Raw data has six columns; each meta learner must fit only producer predictions.
    assert [artifact["estimator"].n_features_in_ for artifact in meta_artifacts] == ([2] if levels == 1 else [2, 1])


def test_live_dataset_augmentation_does_not_change_caller(spectro: SpectroDataset) -> None:
    original_x = spectro.x({}, layout="2d").copy()
    original_y = spectro.y({}).copy()
    original_hash = spectro.content_hash()
    for _ in range(2):
        working = _materialize_dataset(spectro)
        run_paths._apply_sample_augmentation(
            {"sample_augmentation": {"transformers": [GaussianAdditiveNoise(sigma=0.01)], "count": 1}},
            working,
        )
        assert working.num_samples > 24
        assert spectro.num_samples == 24
        assert spectro.content_hash() == original_hash
        np.testing.assert_array_equal(spectro.x({}, layout="2d"), original_x)
        np.testing.assert_array_equal(spectro.y({}), original_y)
    # Reusing the caller for plain CV must still cover precisely the original rows.
    folds = _build_folds(KFold(3), _materialize_dataset(spectro), list(range(24)), set())
    assert sorted(sample for _, validation in folds for sample in validation) == list(range(24))


def test_live_dataset_holdout_does_not_repartition_caller(spectro: SpectroDataset, tmp_path: Path) -> None:
    fold_file = tmp_path / "holdout.json"
    fold_file.write_text(json.dumps([{"train": list(range(16)), "val": list(range(16, 24))}]))
    working = _materialize_dataset(spectro)
    _, train = lower_fold_file_holdout([{"split": str(fold_file)}, {"model": Ridge()}], working)
    assert train == list(range(16))
    assert working.index_column("sample", {"partition": "test"}) == list(range(16, 24))
    working.set_folds([])
    working.add_metadata_column("only_working", np.arange(24))
    assert spectro.index_column("sample", {"partition": "train"}) == list(range(24))
    assert spectro.index_column("sample", {"partition": "test"}) == []
    assert spectro.folds == [(list(range(12)), list(range(12, 24)))]
    assert spectro.metadata_columns == ["group"]


@pytest.mark.parametrize("with_text", [False, True])
def test_mapping_metadata_preserves_column_types(with_text: bool) -> None:
    metadata: dict[str, Any] = {
        "temperature": np.array([20.5, np.nan, 22.0]),
        "count": np.array([1, 2, 10]),
        "enabled": np.array([True, False, True]),
    }
    if with_text:
        metadata["batch"] = ["a", "b", "a"]
    dataset = _materialize_dataset({"X": np.ones((3, 2)), "y": np.arange(3), "metadata": metadata})
    for name, values in metadata.items():
        actual = dataset.metadata_column(name)
        np.testing.assert_array_equal(actual, values)
        if isinstance(values, np.ndarray):
            assert actual.dtype.kind == values.dtype.kind


def test_mapping_metadata_rejects_misaligned_columns() -> None:
    with pytest.raises(ValueError, match="metadata columns must match X row count"):
        _materialize_dataset({"X": np.ones((3, 2)), "metadata": {"batch": ["a", "b"]}})


@pytest.mark.parametrize("prebuilt", [False, True])
@pytest.mark.parametrize("serialized", [False, True])
def test_sibling_sweep_recovers_parameters_and_excludes_annotations(prebuilt: bool, serialized: bool) -> None:
    from nirs4all.pipeline.config.pipeline_config import PipelineConfigs

    model: Any = {"class": "sklearn.cross_decomposition.PLSRegression", "params": {"n_components": 8, "scale": False}} if serialized else PLSRegression(scale=False)
    pipeline: Any = [KFold(3), {"model": model, "n_components": {"_range_": [1, 3, 1]}, "name": "PLS"}]
    if prebuilt:
        pipeline = PipelineConfigs(pipeline)
    recovered = run_backend._native_param_variant_model_params(pipeline, "")
    assert [params["n_components"] for params in recovered] == [1, 2, 3]
    assert all(params["scale"] is False for params in recovered)
    assert all("name" not in params and "model" not in params for params in recovered)
    names = run_backend._derive_variant_config_names(pipeline, "")
    assert len(recovered) == len(names)
    catalog = [
        {"variant_id": f"variant:{index}", "choices": {"model": {"param_overrides": [{"params": {"n_components": index + 1}}]}}}
        for index in range(3)
    ]
    assert run_paths._native_param_config_map_from_catalog(catalog, names, recovered) == dict(zip(["variant:0", "variant:1", "variant:2"], names, strict=True))
    assert run_paths._native_param_winner_config_name([{"estimator": PLSRegression(n_components=3, scale=False)}], names, recovered) == names[2]


@pytest.mark.parametrize("refit", [False, True])
def test_aggregation_stacking_respects_refit_request(
    spectro: SpectroDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refit: bool,
) -> None:
    pipeline = [
        KFold(3),
        {"branch": [[{"model": PLSRegression(2)}], [{"model": Ridge()}]]},
        {"merge": {"predictions": [{"branch": 0, "aggregate": "mean"}, {"branch": 1, "aggregate": "mean"}]}},
        {"model": Ridge(alpha=0.5)},
    ]
    calls: list[dict[str, Any]] = []
    sentinel = object()

    def capture_stack(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(run_backend, "_run_stacking_branch", capture_stack)
    if not refit:
        with pytest.raises(DagMlUnsupported, match="refit=False.*CV-only"):
            run_backend._dispatch_run(pipeline, spectro, tmp_path, "dataset", None, "unused", None, refit=False)
        assert calls == []
    else:
        result = run_backend._dispatch_run(pipeline, spectro, tmp_path, "dataset", None, "unused", None, refit=True)
        assert result is sentinel
        assert calls[0]["refit"] is True


class _BeforeNativeExecution(Exception):
    """Stop after real host fold construction, before native compilation."""


@pytest.mark.parametrize("path", ["before_branch", "inside_merge", "interleaved"])
@pytest.mark.parametrize("split_form", ["bare", "dict", "grouped_dict"])
def test_checkpoint_paths_parse_splitters_and_keep_group_constraints(
    spectro: SpectroDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str, split_form: str,
) -> None:
    splitter: Any = KFold(3)
    if split_form == "dict":
        splitter = {"split": splitter}
    elif split_form == "grouped_dict":
        splitter = {"split": GroupKFold(3), "group_by": "group"}
    first, last = {"model": PLSRegression(2)}, {"model": Ridge()}
    branches = [[StandardScaler()], []]
    captured: list[Any] = []

    def capture_folds(split_step: Any, dataset: Any, pool: Any, excluded: Any) -> Any:
        folds = _build_folds(split_step, dataset, pool, excluded)
        captured.extend(folds)
        raise _BeforeNativeExecution

    monkeypatch.setattr(run_paths, "_build_folds", capture_folds)
    with pytest.raises(_BeforeNativeExecution):
        if path == "interleaved":
            pipeline = [splitter, {"exclude": None}, first, last]
            run_paths._run_interleaved_augmentation_checkpoints(pipeline, spectro, "dataset", "unused", "unused", tmp_path, "rmse", "regression", "", None, False)
        else:
            pipeline = [splitter, first, {"branch": branches}, last] if path == "before_branch" else [splitter, {"branch": branches}, first, {"merge": "features"}, last]
            runner = run_paths._run_checkpoint_before_duplication_branch if path == "before_branch" else run_paths._run_checkpoint_inside_duplication_feature_merge
            runner(pipeline, branches, first, last, spectro, "dataset", "unused", "unused", tmp_path, "rmse", "regression", None, "", None, False)
    assert len(captured) == 3
    assert sorted(sample for _, validation in captured for sample in validation) == list(range(24))
    if split_form == "grouped_dict":
        groups = spectro.metadata_column("group")
        for train, validation in captured:
            assert set(groups[train]).isdisjoint(groups[validation])
    else:
        assert captured == [(train.tolist(), validation.tolist()) for train, validation in KFold(3).split(np.arange(24))]


@pytest.mark.parametrize("winner_name", [None, "config2"])
def test_variant_labels_ignore_summary_and_untagged_reports(winner_name: str | None) -> None:
    scores = {"reports": [
        {"partition": "validation", "fold_id": "fold0", "variant_id": "winner"},
        {"partition": "validation", "fold_id": "w_avg", "variant_id": None},
        {"partition": "validation", "fold_id": "avg", "variant_id": "summary-only"},
        {"partition": "validation", "fold_id": "fold1", "variant_id": None},
        {"partition": "validation", "fold_id": "fold1", "variant_id": "winner"},
        {"partition": "test", "fold_id": None, "variant_id": "test-only"},
        {"partition": "validation", "fold_id": "fold0", "variant_id": "loser"},
    ]}
    expected = {"winner": "config1", "loser": "config2"} if winner_name is None else {"winner": "config2", "loser": "config1"}
    assert _native_variant_config_map(scores, ["config1", "config2"], winner_name) == expected


def test_public_augmentation_then_plain_cv_preserves_input(spectro: SpectroDataset) -> None:
    from nirs4all import run

    original_hash = spectro.content_hash()
    augmentation = {"sample_augmentation": {"transformers": [GaussianAdditiveNoise(sigma=0.01)], "count": 1}}
    for prefix in ([augmentation], []):
        result = run([*prefix, KFold(3), {"model": Ridge()}], spectro, engine="dag-ml", save_artifacts=False, save_charts=False, verbose=0)
        assert isinstance(result, RunResult)
        try:
            assert result.num_predictions > 0
            assert spectro.num_samples == 24
            assert spectro.content_hash() == original_hash
        finally:
            result.close()


@pytest.mark.parametrize("refit", [False, True])
def test_public_sibling_sweep_labels_every_variant(spectro: SpectroDataset, refit: bool) -> None:
    from nirs4all import run

    pipeline = [KFold(3), {"model": PLSRegression(), "n_components": {"_range_": [1, 3, 1]}}]
    names = run_backend._derive_variant_config_names(pipeline, "")
    result = run(pipeline, spectro, engine="dag-ml", refit=refit, save_artifacts=False, save_charts=False, verbose=0)
    assert isinstance(result, RunResult)
    try:
        rows = result.predictions.filter_predictions(load_arrays=True)
        cv_rows = [row for row in rows if str(row["fold_id"]) in {"0", "1", "2"} and row["partition"] == "val"]
        assert len(cv_rows) == 9
        assert {row["config_name"] for row in cv_rows} == set(names)
        features = spectro.x({"partition": "train"}, layout="2d")
        targets = spectro.y({"partition": "train"}).reshape(-1)
        for component_count, config_name in enumerate(names, start=1):
            for fold_index, (train, validation) in enumerate(KFold(3).split(features)):
                model = PLSRegression(n_components=component_count).fit(features[train], targets[train])
                expected_rmse = np.sqrt(np.mean((model.predict(features[validation]).reshape(-1) - targets[validation]) ** 2))
                row = next(row for row in cv_rows if row["config_name"] == config_name and str(row["fold_id"]) == str(fold_index))
                np.testing.assert_allclose(row["val_score"], expected_rmse, rtol=1e-9)
        assert any(row["fold_id"] == "final" for row in rows) is refit
        if not refit:
            assert result.per_dataset[spectro.name]["refit_enabled"] is False
    finally:
        result.close()


@pytest.mark.parametrize("inside_merge", [False, True])
def test_public_dict_split_checkpoint_branches(spectro: SpectroDataset, inside_merge: bool) -> None:
    from nirs4all import run

    first, last = {"model": PLSRegression(2)}, {"model": Ridge()}
    branch = {"branch": [[StandardScaler()], [MinMaxScaler()]]}
    body = [branch, first, {"merge": "features"}, last] if inside_merge else [first, branch, last]
    result = run([{"split": KFold(3)}, *body], spectro, engine="dag-ml", refit=False, save_artifacts=False, save_charts=False, verbose=0)
    assert isinstance(result, RunResult)
    try:
        rows = result.predictions.filter_predictions(load_arrays=True)
        assert rows and all(row["fold_id"] != "final" for row in rows)
        assert len(result.per_dataset[spectro.name]["checkpoint_producers"]) == (4 if inside_merge else 3)
    finally:
        result.close()
