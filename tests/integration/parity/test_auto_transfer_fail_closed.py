"""Transfer search failures cannot publish a predictor from a reduced search."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import nirs4all
import nirs4all.analysis.selector as selector_module
from nirs4all.analysis import (
    TransferCandidateRecord,
    TransferPreprocessingSelector,
    TransferResult,
    TransferSelectionError,
    TransferSelectionResults,
    get_base_preprocessings,
)
from nirs4all.controllers.data.auto_transfer_preproc import AutoTransferPreprocessingController
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.bundle.generator import write_single_model_bundle
from nirs4all.pipeline.bundle.loader import BundleLoader
from nirs4all.pipeline.config.context import ExecutionPhase
from nirs4all.pipeline.dagml.auto_transfer import DagMLAutoTransferPreprocessor, transfer_selection_metadata
from nirs4all.pipeline.dagml.host_container import open_n4a_source

from ._dagml_cli import dagml_cli_path

pytestmark = pytest.mark.parity

_SG_FAILURE = "injected applicable SG15d1 candidate failure"


@pytest.fixture
def transfer_views() -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic cohorts with enough channels for the real SG15d1."""
    random = np.random.default_rng(1951)
    return random.normal(size=(24, 31)), random.normal(loc=0.15, size=(12, 31))


def _selector(**kwargs) -> TransferPreprocessingSelector:
    base = get_base_preprocessings()
    return TransferPreprocessingSelector(
        preset=None,
        preprocessings={name: base[name] for name in ("snv", "savgol15_d1", "gaussian")},
        n_components=3,
        k_neighbors=3,
        verbose=0,
        **kwargs,
    )


def _inject_sg_failure(monkeypatch) -> list[str]:
    original = selector_module.apply_pipeline
    reached: list[str] = []

    def apply_pipeline(features, transforms):
        if any(getattr(transform, "window_length", None) == 15 and getattr(transform, "deriv", None) == 1 for transform in transforms):
            reached.append("savgol15_d1")
            raise RuntimeError(_SG_FAILURE)
        return original(features, transforms)

    monkeypatch.setattr(selector_module, "apply_pipeline", apply_pipeline)
    return reached


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_applicable_sg_failure_retains_every_candidate_and_refuses_selection(transfer_views, monkeypatch, n_jobs: int) -> None:
    reached = _inject_sg_failure(monkeypatch)
    selector = _selector(n_jobs=n_jobs)
    with pytest.raises(TransferSelectionError, match=_SG_FAILURE) as error:
        selector.fit(*transfer_views)
    assert reached
    assert selector.results_ is None
    records = {record.name: record for record in error.value.candidate_records}
    assert set(records) == {"snv", "savgol15_d1", "gaussian"}
    assert records["savgol15_d1"].status == "failed"
    assert records["savgol15_d1"].exception_type == "RuntimeError"
    assert records["savgol15_d1"].message == _SG_FAILURE
    assert records["snv"].status == records["gaussian"].status == "completed"
    assert selector.candidate_records_ == error.value.candidate_records


@pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf])
def test_non_finite_candidate_score_refuses_selection(transfer_views, monkeypatch, score: float) -> None:
    monkeypatch.setattr(selector_module, "compute_transfer_score", lambda *args, **kwargs: score)
    selector = _selector(n_jobs=2)
    with pytest.raises(TransferSelectionError) as error:
        selector.fit(*transfer_views)
    assert selector.results_ is None
    assert {record.name for record in error.value.candidate_records} == {"snv", "savgol15_d1", "gaussian"}
    assert all(record.status == "non_finite" and record.score is None for record in error.value.candidate_records)
    assert all(record.message for record in error.value.candidate_records)


def test_declared_applicability_excludes_before_transform_computation(transfer_views, monkeypatch) -> None:
    reached = _inject_sg_failure(monkeypatch)
    source, target = (features[:, :9] for features in transfer_views)

    def sg_applicability(source_features, target_features):
        assert source_features.shape[1] == target_features.shape[1] == 9
        return "declared SG15 window requires at least 15 channels"

    selector = _selector(n_jobs=2, applicability_checks={"savgol15_d1": sg_applicability})
    results = selector.fit(source, target)
    assert reached == []
    assert results.complete is True
    results.require_complete()
    records = {record.name: record for record in results.candidate_records}
    assert records["savgol15_d1"].status == "excluded"
    assert records["savgol15_d1"].reason == "declared SG15 window requires at least 15 channels"
    assert records["snv"].status == records["gaussian"].status == "completed"
    assert {result.name for result in results.ranking} == {"snv", "gaussian"}


def test_applicability_check_exception_is_failure_not_exclusion(transfer_views) -> None:
    def broken_applicability(source, target):
        raise RuntimeError("declared applicability provider failed")

    selector = _selector(n_jobs=1, applicability_checks={"savgol15_d1": broken_applicability})
    with pytest.raises(TransferSelectionError, match="declared applicability provider failed") as error:
        selector.fit(*transfer_views)
    failed = next(record for record in error.value.candidate_records if record.name == "savgol15_d1")
    assert failed.status == "failed"
    assert failed.exception_type == "RuntimeError"
    assert failed.reason is None


def test_failed_refit_clears_previous_successful_result(transfer_views, monkeypatch) -> None:
    selector = _selector(n_jobs=1)
    previous = selector.fit(*transfer_views)
    assert previous.complete
    assert all(record.status == "completed" for record in selector.candidate_records_)
    _inject_sg_failure(monkeypatch)
    with pytest.raises(TransferSelectionError):
        selector.fit(*transfer_views)
    assert selector.results_ is None
    assert len(selector.candidate_records_) == 3
    assert sum(record.status == "failed" for record in selector.candidate_records_) == 1


@pytest.mark.parametrize("conversion", ["best", "top_k", "to_preprocessing_list", "to_pipeline_spec"])
def test_incomplete_selection_cannot_be_converted_to_executable_recommendation(conversion: str) -> None:
    completed = TransferResult("snv", "single", ["snv"], 0.9, {}, 1.0)
    result = TransferSelectionResults(
        ranking=[completed], raw_metrics={}, complete=False,
        candidate_records=[TransferCandidateRecord("savgol15_d1", "single", "failed", exception_type="RuntimeError", message=_SG_FAILURE)],
    )
    with pytest.raises(TransferSelectionError):
        if conversion == "best":
            _ = result.best
        else:
            getattr(result, conversion)()


@pytest.mark.parametrize("engine", ["legacy", "dag-ml"])
@pytest.mark.parametrize("failure", ["sg_exception", "non_finite"])
def test_production_run_refuses_incomplete_selection_before_model_fit(transfer_views, monkeypatch, tmp_path, engine: str, failure: str) -> None:
    if failure == "sg_exception":
        _inject_sg_failure(monkeypatch)
        message = _SG_FAILURE
    else:
        monkeypatch.setattr(selector_module, "compute_transfer_score", lambda *args, **kwargs: np.nan)
        message = "non_finite"
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    model_fit_calls = []

    def unexpected_model_fit(*args, **kwargs):
        model_fit_calls.append(True)
        raise AssertionError("model fit must not follow an incomplete transfer search")

    monkeypatch.setattr(PLSRegression, "fit", unexpected_model_fit)
    source, target = transfer_views
    dataset = SpectroDataset("transfer_failure_refusal")
    dataset.add_samples(source, {"partition": "train"})
    dataset.add_samples(target, {"partition": "test"})
    dataset.add_targets(np.arange(len(source) + len(target), dtype=float).reshape(-1, 1))
    with pytest.raises(Exception, match=message):  # noqa: PT011 -- engine envelopes preserve the original candidate failure
        nirs4all.run(
            [{"auto_transfer_preproc": {"preset": "fast", "verbose": 0}}, PLSRegression(3)],
            dataset, engine=engine, allow_fallback=False, workspace_path=tmp_path / engine,
            save_artifacts=False, save_charts=False, verbose=0,
        )
    assert model_fit_calls == []


def _exclude_declared_sg15(source: np.ndarray, target: np.ndarray) -> str:
    """A caller-declared search restriction, retained separately from failures."""
    return "SG15d1 explicitly outside this qualification search"


@pytest.mark.parametrize("mode", ["1", "0"])
def test_native_archive_preserves_completed_candidate_records(transfer_views, monkeypatch, tmp_path, mode: str) -> None:
    cli = dagml_cli_path()
    if mode == "0" and not cli.exists():
        pytest.skip(f"dag-ml-cli binary not built at {cli}")
    monkeypatch.setenv("N4A_DAGML_INPROCESS", mode)
    monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    source, target = transfer_views
    dataset = SpectroDataset("transfer_record_archive")
    dataset.add_samples(source, {"partition": "train"})
    dataset.add_samples(target, {"partition": "test"})
    dataset.add_targets(np.concatenate((source[:, 0], target[:, 0])).reshape(-1, 1))
    config = {
        "preset": "fast", "top_k": 2, "use_augmentation": True, "verbose": 0,
    }
    result = nirs4all.run(
        [{"auto_transfer_preproc": config}, PLSRegression(3)], dataset,
        engine="dag-ml", allow_fallback=False, workspace_path=tmp_path / "workspace",
        save_charts=False, verbose=0,
    )
    try:
        expected = {"artifacts": [
            {"artifact_id": artifact["artifact_id"], "selections": transfer_selection_metadata(artifact["estimator"])}
            for artifact in result._dagml_refit_artifacts
        ]}
        assert expected["artifacts"]
        for artifact in expected["artifacts"]:
            assert artifact["selections"]
            for selection in artifact["selections"]:
                recommendation = selection["recommendation"]
                assert recommendation["complete"] is True
                records = {record["name"]: record for record in recommendation["candidate_records"]}
                assert records["savgol15_d1"]["status"] == "completed"
                assert records.keys() == get_base_preprocessings().keys()
                assert all(record["status"] == "completed" for record in records.values())
        archive = result.export(tmp_path / "complete_search.n4a")
        manifest = json.loads((result._dagml_results_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["transfer_preprocessing_selection"] == expected
        with open_n4a_source(archive) as source_archive:
            stored = source_archive.require_host().bundle_manifest
            assert stored["transfer_preprocessing_selection"] == expected

        def forbidden_selection(*args, **kwargs):
            raise AssertionError("archive replay must not repeat transfer selection")

        monkeypatch.setattr(TransferPreprocessingSelector, "fit", forbidden_selection)
        replay = nirs4all.predict(archive, target)
        assert np.isfinite(np.asarray(replay.y_pred)).all()
    finally:
        result.close()


def test_fitted_host_chain_preserves_declared_exclusions_in_archive(transfer_views, monkeypatch, tmp_path) -> None:
    source, target = transfer_views
    preprocessing = DagMLAutoTransferPreprocessor({
        "preset": "fast", "verbose": 0,
        "applicability_checks": {"savgol15_d1": _exclude_declared_sg15},
    })
    preprocessing.fit_with_views({"train": (source, None), "test": (target, None)})
    model = PLSRegression(3).fit(preprocessing.transform(source), source[:, 0])
    fitted = make_pipeline(preprocessing, model)
    selections = transfer_selection_metadata(fitted)
    assert len(selections) == 1
    recommendation = selections[0]["recommendation"]
    assert recommendation["complete"] is True
    records = {record["name"]: record for record in recommendation["candidate_records"]}
    assert records["savgol15_d1"]["status"] == "excluded"
    assert records["savgol15_d1"]["reason"] == _exclude_declared_sg15(source, target)
    assert any(record["status"] == "completed" for record in records.values())
    metadata = {"artifacts": [{"artifact_id": "qualification:fitted-host-chain", "selections": selections}]}
    archive = write_single_model_bundle(fitted, tmp_path / "declared_exclusions.n4a", provenance={"transfer_preprocessing_selection": metadata})
    with open_n4a_source(archive) as stored:
        assert stored.require_host().bundle_manifest["transfer_preprocessing_selection"] == metadata

    def forbidden_selection(*args, **kwargs):
        raise AssertionError("saved host inference must not repeat transfer selection")

    monkeypatch.setattr(TransferPreprocessingSelector, "fit", forbidden_selection)
    np.testing.assert_array_equal(BundleLoader(archive).predict(target), fitted.predict(target))


def test_fit_selected_refuses_incomplete_recommendation(transfer_views) -> None:
    preprocessing = DagMLAutoTransferPreprocessor()
    recommendation = {"pipeline_spec": "snv", "complete": False, "candidate_records": []}
    with pytest.raises(TransferSelectionError, match="incomplete"):
        preprocessing.fit_selected(recommendation, transfer_views[0])
    assert not hasattr(preprocessing, "recommendation_")


def test_failed_selected_name_clears_previous_fitted_recommendation(transfer_views) -> None:
    source, target = transfer_views
    preprocessing = DagMLAutoTransferPreprocessor({"preset": "fast", "verbose": 0})
    preprocessing.fit_with_views({"train": (source, None), "test": (target, None)})
    recommendation = {**preprocessing.recommendation_, "pipeline_spec": "undeclared_owner_preprocessing"}
    completed = recommendation["candidate_records"]
    assert completed and all(record["status"] == "completed" for record in completed)
    with pytest.raises(TransferSelectionError, match="unknown transfer preprocessing") as error:
        preprocessing.fit_selected(recommendation, source)
    assert sum(record.status == "completed" for record in error.value.candidate_records) == len(completed)
    assert any(record.status == "failed" and record.stage == "recommendation" for record in error.value.candidate_records)
    assert not hasattr(preprocessing, "recommendation_")
    with pytest.raises(NotFittedError):
        preprocessing.transform(source)


def _candidate_failure_dataset(transfer_views) -> SpectroDataset:
    source, target = transfer_views
    dataset = SpectroDataset("transfer_candidate_outcomes")
    dataset.add_samples(source, {"partition": "train"})
    dataset.add_samples(target, {"partition": "test"})
    dataset.add_targets(np.concatenate((source[:, 0], target[:, 0])).reshape(-1, 1))
    return dataset


def _invalid_pca_selection() -> dict:
    return {"auto_transfer_preproc": {"preset": "fast", "verbose": 0, "preprocessing_spec": {"_or_": [PCA(n_components=1000), "snv"]}}}


def test_parallel_variants_propagate_real_candidate_failure_with_records(transfer_views, tmp_path) -> None:
    """The real loky boundary must retain an error whose text names n_components."""
    pipeline = [{"_or_": [_invalid_pca_selection(), {"auto_transfer_preproc": {"preset": "fast", "verbose": 0}}]}, PLSRegression(3)]
    with pytest.raises(TransferSelectionError, match="n_components") as error:
        nirs4all.run(
            pipeline, _candidate_failure_dataset(transfer_views), engine="legacy", n_jobs=2,
            refit=False, workspace_path=tmp_path, save_artifacts=False, save_charts=False, verbose=0,
        )
    assert error.value.candidate_records
    assert any(record.status == "failed" and record.exception_type == "ValueError" and record.message is not None and "n_components" in record.message for record in error.value.candidate_records)
    assert error.value.__cause__ is not None
    assert "n_components" in str(error.value.__cause__)


def test_continue_on_error_cannot_publish_partial_pipeline_after_candidate_failure(transfer_views, tmp_path, monkeypatch) -> None:
    model_fits = []

    def forbidden_fit(*args, **kwargs):
        model_fits.append(True)
        raise AssertionError("an incomplete search cannot reach model fitting")

    monkeypatch.setattr(PLSRegression, "fit", forbidden_fit)
    with pytest.raises(TransferSelectionError, match="n_components") as error:
        nirs4all.run(
            [{"_or_": [_invalid_pca_selection()]}, PLSRegression(3)], _candidate_failure_dataset(transfer_views),
            engine="legacy", continue_on_error=True, workspace_path=tmp_path,
            save_artifacts=False, save_charts=False, verbose=0,
        )
    assert error.value.candidate_records
    assert model_fits == []


def test_generator_supervised_validation_uses_stored_transform_objects(transfer_views) -> None:
    source, target = transfer_views
    selector = TransferPreprocessingSelector(
        preset=None, preprocessing_spec={"_or_": [StandardScaler(), PCA(n_components=3)]},
        n_components=3, k_neighbors=3, n_jobs=1, run_stage4=True,
        stage4_models=["ridge"], stage4_cv_folds=2, stage4_top_k=2, verbose=0,
    )
    result = selector.fit(source, target, source[:, 0])
    assert result.complete
    assert len(result.ranking) == 2
    assert all(candidate.signal_score is not None and np.isfinite(candidate.signal_score) for candidate in result.ranking)
    validation = [record for record in result.candidate_records if record.stage == "validation:ridge"]
    assert len(validation) == 2
    assert all(record.status == "completed" and record.score is not None and np.isfinite(record.score) for record in validation)


@pytest.mark.parametrize("stacking", [False, True])
def test_refit_candidate_failure_cannot_return_successful_cv_result(transfer_views, tmp_path, monkeypatch, stacking: bool) -> None:
    """A valid CV pass cannot hide an applicable-candidate error during REFIT."""
    original_train = AutoTransferPreprocessingController._execute_train_mode
    original_apply = selector_module.apply_pipeline
    phase = {"current": None}
    observed = []

    def train_mode(controller, config, dataset, context, runtime_context, source=-1):
        phase["current"] = runtime_context.phase
        observed.append(runtime_context.phase)
        try:
            return original_train(controller, config, dataset, context, runtime_context, source)
        finally:
            phase["current"] = None

    def apply_pipeline(features, transforms):
        if phase["current"] == ExecutionPhase.REFIT and any(
            getattr(transform, "window_length", None) == 15 and getattr(transform, "deriv", None) == 1 for transform in transforms
        ):
            raise RuntimeError(_SG_FAILURE)
        return original_apply(features, transforms)

    monkeypatch.setattr(AutoTransferPreprocessingController, "_execute_train_mode", train_mode)
    monkeypatch.setattr(selector_module, "apply_pipeline", apply_pipeline)
    pipeline = [{"auto_transfer_preproc": {"preset": "fast", "verbose": 0}}, KFold(2)]
    if stacking:
        pipeline.extend([{"branch": [[PLSRegression(2)], [PLSRegression(3)]]}, {"merge": "predictions"}, PLSRegression(2)])
    else:
        pipeline.append(PLSRegression(3))
    with pytest.raises(TransferSelectionError, match=_SG_FAILURE) as error:
        nirs4all.run(
            pipeline,
            _candidate_failure_dataset(transfer_views), engine="legacy", refit=True,
            workspace_path=tmp_path, save_charts=False, verbose=0,
        )
    assert ExecutionPhase.CV in observed and ExecutionPhase.REFIT in observed
    assert any(record.name == "savgol15_d1" and record.status == "failed" for record in error.value.candidate_records)


def test_declared_generation_failure_cannot_be_skipped_with_continue_on_error(transfer_views, tmp_path, monkeypatch) -> None:
    model_fits = []

    def forbidden_fit(*args, **kwargs):
        model_fits.append(True)
        raise AssertionError("an incomplete declared generator cannot reach model fitting")

    monkeypatch.setattr(PLSRegression, "fit", forbidden_fit)
    config = {"preset": "balanced", "verbose": 0, "preprocessing_spec": {"_or_": [StandardScaler(), PCA(n_components=3)]}}
    with pytest.raises(TransferSelectionError) as error:
        nirs4all.run(
            [{"auto_transfer_preproc": config}, PLSRegression(3)], _candidate_failure_dataset(transfer_views),
            engine="legacy", continue_on_error=True, workspace_path=tmp_path,
            save_artifacts=False, save_charts=False, verbose=0,
        )
    assert model_fits == []
    assert any(record.status == "completed" for record in error.value.candidate_records)
    assert any(record.status == "failed" for record in error.value.candidate_records)




@pytest.mark.parametrize("engine", ["legacy", "dag-ml"])
def test_invalid_serialized_candidate_cannot_be_skipped(transfer_views, tmp_path, monkeypatch, engine: str) -> None:
    """Constructor failures retain their cause before any downstream fit."""
    dataset = _candidate_failure_dataset(transfer_views)
    fitted = []

    def forbidden_fit(*args, **kwargs):
        fitted.append(True)
        raise AssertionError("A failed transfer specification must not train a model")

    monkeypatch.setattr(PLSRegression, "fit", forbidden_fit)
    if engine == "dag-ml":
        monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    pipeline = [{"auto_transfer_preproc": {
        "preprocessing_spec": {"_or_": [{"class": "sklearn.decomposition.PCA", "params": {"bogus": 1}}]},
        "verbose": 0,
    }}, PLSRegression(2)]
    # DAG rejects malformed constructors in its public preflight, before a
    # selector exists. Legacy also reaches its guarded step-parser boundary.
    expected_error = TransferSelectionError if engine == "legacy" else ValueError
    with pytest.raises(expected_error, match="bogus") as error:
        nirs4all.run(pipeline, dataset, engine=engine, continue_on_error=True, workspace_path=tmp_path / engine, verbose=0)
    assert not fitted
    if engine == "legacy":
        assert error.value.candidate_records[0].stage == "specification"
        assert "bogus" in error.value.candidate_records[0].message


@pytest.mark.parametrize("pipeline_type", ["single", "stacked", "augmented"])
def test_generated_recommendation_canonical_recipe_preserves_exact_parameters(transfer_views, pipeline_type: str) -> None:
    """Names alone cannot distinguish these real same-class generated candidates."""
    from copy import deepcopy

    from nirs4all.analysis.transfer_utils import resolve_transfer_preprocessing
    from nirs4all.pipeline.config.component_serialization import serialize_component

    source, target = transfer_views
    spec = {"_or_": [StandardScaler(with_mean=False), StandardScaler(with_std=False)]}
    if pipeline_type != "single":
        spec["arrange" if pipeline_type == "stacked" else "pick"] = 2
    results = TransferPreprocessingSelector(
        preset=None, preprocessing_spec=spec, n_components=3, k_neighbors=3, n_jobs=1, verbose=0,
    ).fit(source, target)
    candidates = [candidate for candidate in results.ranking if candidate.pipeline_type == pipeline_type]
    assert candidates
    identities = set()
    for candidate in candidates:
        selected = TransferSelectionResults([candidate], results.raw_metrics, candidate_records=results.candidate_records)
        # Existing display conversion remains available without changing names.
        assert selected.to_pipeline_spec() == candidate.name
        recipe = json.loads(json.dumps(selected.to_pipeline_spec(resolve_transforms=True)))
        name, components = resolve_transfer_preprocessing(recipe)
        identities.add(name)
        actual = source.copy()
        for _, transform in components:
            actual = transform.fit_transform(actual)
        expected = source.copy()
        if pipeline_type == "augmented":
            expected = np.hstack([deepcopy(transform).fit_transform(source) for transform in candidate.transforms])
        else:
            for transform in candidate.transforms:
                expected = deepcopy(transform).fit_transform(expected)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
        assert serialize_component(components[0][1]) == recipe
    assert len(identities) == len(candidates)


@pytest.mark.parametrize("use_augmentation", [False, True])
def test_same_class_selected_recipes_fit_all_channels_without_losing_parameters(transfer_views, use_augmentation: bool) -> None:
    """The source-local DAG fit consumes a JSON selection made on the joint view."""
    from nirs4all.pipeline.config.component_serialization import serialize_component

    source, target = transfer_views
    config = {"preset": "fast", "top_k": 2, "use_augmentation": use_augmentation, "verbose": 0,
              "preprocessing_spec": {"_or_": [StandardScaler(with_mean=False), StandardScaler(with_std=False)]}}
    shared = DagMLAutoTransferPreprocessor(config).select_with_views({"train": (source, None), "test": (target, None)})
    recommendation = json.loads(json.dumps(shared.recommendation_))
    specs = recommendation["pipeline_spec"]["feature_augmentation"] if use_augmentation else recommendation["pipeline_spec"]
    assert len(specs) == 2 and specs[0] != specs[1]
    train_x, predict_x = source[:, :7], target[:, :7]
    local = DagMLAutoTransferPreprocessor(config)
    local.set_input_channels((3, 4))
    local.fit_selected(recommendation, train_x)
    expected_parts = [predict_x[:, :3], predict_x[:, 3:]] if use_augmentation else []
    for channel in ([0] if use_augmentation else [0, 1]):
        train = train_x[:, :3] if channel == 0 else train_x[:, 3:]
        predict = predict_x[:, :3] if channel == 0 else predict_x[:, 3:]
        for recipe in specs:
            params = recipe.get("params", {})
            transform = StandardScaler(**params).fit(train)
            transformed = transform.transform(predict)
            if use_augmentation:
                expected_parts.append(transformed)
            else:
                train, predict = transform.transform(train), transformed
        if not use_augmentation:
            expected_parts.append(predict)
    np.testing.assert_allclose(local.transform(predict_x), np.hstack(expected_parts), rtol=1e-5, atol=1e-8)
    chains = local._chains if use_augmentation else local._channel_chains[0]
    assert [serialize_component(chain[0]) for chain in chains] == specs
