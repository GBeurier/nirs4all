"""Real native unit influence and scoring, independently compared to sklearn."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from collections import Counter
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dag_ml._dag_ml import DagMlRuntimeError
from nirs4all_io import MultimodalDataset, TensorSource
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

import nirs4all
from nirs4all.operators.models.multimodal import MultimodalClassifier, MultimodalRegressor


def _cohort(*, partial: bool = False, conflicting_truth: bool = False,
            target_mask_profile: str = "partial", grouped: bool = True) -> MultimodalDataset:
    rng = np.random.default_rng(817)
    repeats = [1, 2, 3, 4] * 4
    units = np.repeat(np.arange(16), repeats)
    scan = np.concatenate([np.arange(count) for count in repeats])
    ids = [f"observation.{index:03d}" for index in range(len(units))]
    latent = rng.normal(size=(16, 3))
    nir = latent[units] + rng.normal(scale=.7, size=(len(units), 3))
    metadata = latent[units, :2] + rng.normal(scale=.5, size=(len(units), 2))
    y = latent[units] @ [2., -1.5, .7]
    if partial:
        y = np.column_stack([y, latent[units] @ [-.5, 2., 1.]])
    if conflicting_truth:
        y[2] += 10.  # The two observations of unit.01 now contradict.
    sources = {
        "nir": TensorSource(nir, ids, representation_id="signal_1d", presence_mask=scan != 2 if partial else None),
        "metadata": TensorSource(metadata, ids, representation_id="tabular_numeric", presence_mask=scan != 1 if partial else None),
    }
    target_mask = np.column_stack([scan != 1, scan != 2]) if partial else None
    if partial and target_mask_profile == "complete":
        target_mask = np.ones((len(units), 2), dtype=bool)
    elif partial and target_mask_profile == "all_masked":
        assert target_mask is not None
        target_mask[scan == 3] = False
    elif target_mask_profile != "partial":
        raise ValueError("unknown experimental-unit test target mask profile")
    return MultimodalDataset(
        sources, sample_ids=ids, y=y, task_type="regression", target_names=["a", "b"] if partial else ["a"],
        target_mask=target_mask,
        partitions=["train" if unit < 12 else "test" for unit in units],
        groups=[f"split_batch.{unit // 2:02d}" for unit in units] if grouped else None,
        independent_unit_ids=[f"unit.{unit:02d}" for unit in units], repetition_ids=[f"scan.{index}" for index in scan],
        name="explicit_experimental_units",
    )


def _pipeline(*, split: bool = True, partial: bool = False) -> list[Any]:
    prefix: list[Any] = [GroupKFold(3)] if split else []
    if partial:
        return [*prefix, {"branch": {"by_source": True, "steps": {
            "nir": [StandardScaler(), Ridge(alpha=.2)],
            "metadata": [StandardScaler(), Ridge(alpha=.3)],
        }, "missing_source_policy": "zero_with_indicator", "target_policy": "per_target"}},
            {"merge": "predictions"}, Ridge(alpha=.4)]
    return [*prefix, {"model": MultimodalRegressor(
        {"nir": StandardScaler(), "metadata": StandardScaler()}, Ridge(alpha=.4))}]


class _Observer:
    def __init__(self) -> None:
        self.tasks: list[dict[str, Any]] = []
        self.results: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from nirs4all.pipeline.dagml import full_train, in_process_runner

        for module in (in_process_runner, full_train):
            original = module.run_node

            def observe(task: dict[str, Any], *args: Any, _original: Any = original, **kwargs: Any) -> Any:
                tracked = (task["phase"] in {"FIT_CV", "REFIT"}
                           and task["node_plan"]["controller_id"] in {"controller:nirs4all.model", "controller:nirs4all.meta_model"})
                if tracked:
                    self.tasks.append(deepcopy(task))
                result = _original(task, *args, **kwargs)
                if tracked:
                    self.results.append((deepcopy(task), deepcopy(result)))
                return result

            monkeypatch.setattr(module, "run_node", observe)


def _artifacts(result: Any) -> list[dict[str, Any]]:
    runs = getattr(result, "runs", None) or [result]
    return [artifact for run in runs for artifact in run._dagml_refit_artifacts]


def _weight_oracle(cohort: Any, ids: list[str], target: int | None = None) -> np.ndarray:
    """Test-only counting, independent of the native influence implementation."""
    positions = {sample: row for row, sample in enumerate(cohort.sample_ids)}
    units = [cohort.independent_unit_ids[positions[sample]] for sample in ids]
    active = np.ones(len(ids), dtype=bool) if target is None else cohort.target_mask[[positions[sample] for sample in ids], target]
    counts = Counter(unit for unit, observed in zip(units, active, strict=True) if observed)
    return np.asarray([1. / counts[unit] if observed else 0. for unit, observed in zip(units, active, strict=True)])


def _assert_native_scopes(cohort: Any, tasks: list[dict[str, Any]], *, partial: bool) -> None:
    assert tasks and any(task["phase"] == "REFIT" for task in tasks)
    positions = {sample: row for row, sample in enumerate(cohort.sample_ids)}
    for task in tasks:
        influence = task["fit_influence"]
        ids = influence["fit_sample_ids"]
        assert ids and len(ids) == len(set(ids))
        assert all(cohort.partitions[positions[sample]] == "train" for sample in ids)
        assert influence["independent_unit_ids"] == [cohort.independent_unit_ids[positions[sample]] for sample in ids]
        np.testing.assert_allclose(influence["row_weights"], _weight_oracle(cohort, ids), rtol=0, atol=0)
        if partial:
            assert influence["target_names"] == ["a", "b"]
            np.testing.assert_allclose(influence["target_row_weights"], np.column_stack([
                _weight_oracle(cohort, ids, target) for target in range(2)
            ]), rtol=0, atol=0)


@pytest.fixture(autouse=True)
def _native_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("legacy scheduler executed"))


@pytest.mark.parametrize("split", [False, True])
def test_public_weighted_encoders_refit_native_group_scores_and_replay(split: bool, tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    installed_python = os.environ.get("NIRS4ALL_EXPERIMENTAL_UNITS_INSTALLED_PYTHON")
    if os.environ.get("NIRS4ALL_REQUIRE_EXPERIMENTAL_UNITS_INSTALLED") == "1" and not installed_python:
        pytest.fail("NIRS4ALL_EXPERIMENTAL_UNITS_INSTALLED_PYTHON is mandatory")
    cohort, observer = _cohort(), _Observer()
    observer.install(monkeypatch)
    result = nirs4all.run(_pipeline(split=split), cohort, engine="dag-ml", refit=True, save_artifacts=True,
                         save_charts=False, verbose=0, random_state=23, workspace_path=tmp_path / "training")
    try:
        _assert_native_scopes(cohort, observer.tasks, partial=False)
        assert any(task["phase"] == "FIT_CV" for task in observer.tasks) is split
        artifact = next(artifact for artifact in _artifacts(result) if artifact["controller_id"] == "controller:nirs4all.model")
        fitted = artifact["estimator"]
        influence = fitted._nirs4all_fit_influence
        ids = influence["fit_sample_ids"]
        rows = [cohort.sample_ids.index(sample) for sample in ids]
        weights = _weight_oracle(cohort, ids)
        # The internal captured wrapper uses its signed source order; dict
        # serialization may order the model's transformer names differently.
        assert set(fitted.source_names) == set(cohort.sources)
        raw = cohort.source_values(source_names=fitted.source_names)
        encoders = [StandardScaler().fit(block[rows], sample_weight=weights) for block in raw]
        encoded_train = np.hstack([encoder.transform(block[rows]) for encoder, block in zip(encoders, raw, strict=True)])
        truth = np.asarray(cohort.y, dtype=np.float32).astype(np.float64)
        oracle = Ridge(alpha=.4).fit(encoded_train, truth[rows], sample_weight=weights)
        encoded_all = np.hstack([encoder.transform(block) for encoder, block in zip(encoders, raw, strict=True)])
        expected = oracle.predict(encoded_all)
        np.testing.assert_allclose(np.asarray(fitted.predict(raw)).ravel(), expected, rtol=1e-10, atol=1e-10)
        reports = [report for report in result._dagml_score_set["reports"] if report["level"] == "group"]
        unit_key = {"kind": "relation_metadata", "key": "independent_unit_id"}
        assert reports and all(report["grouping_key"] == unit_key for report in reports)
        test_reports = [report for report in reports if report["partition"] == "test" and report.get("fold_id") is None]
        assert len(test_reports) == 1 and test_reports[0]["row_count"] == 4
        test_units = sorted({cohort.independent_unit_ids[row] for row, partition in enumerate(cohort.partitions) if partition == "test"})
        errors = [np.mean(expected[[row for row, unit in enumerate(cohort.independent_unit_ids) if unit == wanted]])
                  - truth[cohort.independent_unit_ids.index(wanted)] for wanted in test_units]
        np.testing.assert_allclose(test_reports[0]["metrics"]["rmse"], np.sqrt(np.mean(np.square(errors))), rtol=1e-10, atol=1e-10)
        run = next(run for run in (getattr(result, "runs", None) or [result])
                   if any(candidate is artifact for candidate in run._dagml_refit_artifacts))
        archive = run.export(tmp_path / "weighted.n4a")
    finally:
        result.close()
    prediction = MultimodalDataset(cohort.sources, sample_ids=cohort.sample_ids, partitions=["predict"] * len(cohort),
                                  task_type="regression", target_names=["a"],
                                  independent_unit_ids=cohort.independent_unit_ids, repetition_ids=cohort.repetition_ids)
    # Both replay paths must work after the training workspace is gone.
    shutil.rmtree(tmp_path / "training")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("weighted replay attempted FIT")

    monkeypatch.setattr(Ridge, "fit", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    replay = nirs4all.predict(archive, prediction, engine="dag-ml", verbose=0)
    np.testing.assert_allclose(np.asarray(replay.y_pred).ravel(), expected, rtol=1e-10, atol=1e-10)
    assert replay.metadata["training_performed"] is False and replay.metadata["scores"] is None
    if installed_python:
        _assert_installed_replay(installed_python, Path(archive), prediction, expected, tmp_path)


def _assert_installed_replay(python: str, archive: Path, cohort: MultimodalDataset,
                             expected: np.ndarray, directory: Path) -> None:
    """No training data, source checkout or parent estimator reaches the child."""
    inputs = directory / "predict-inputs.json"
    inputs.write_text(json.dumps(cohort.to_dict()), encoding="utf-8")
    clean = directory / "clean-process"
    clean.mkdir()
    script = textwrap.dedent("""\
        import importlib, json, pathlib, sys
        import numpy as np
        import dag_ml, nirs4all
        from nirs4all_io import MultimodalDataset
        from nirs4all.pipeline import PipelineRunner
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        prefix = pathlib.Path(sys.prefix).resolve()
        for name in ('nirs4all', 'nirs4all_io', 'dag_ml', 'dag_ml._dag_ml',
                     'nirs4all.pipeline.dagml.experimental_units',
                     'nirs4all.pipeline.dagml.general_replay'):
            path = pathlib.Path(importlib.import_module(name).__file__).resolve()
            assert path.is_relative_to(prefix), (name, str(path))
        def forbidden(*args, **kwargs):
            raise AssertionError('FIT/HPO during independent-unit archive replay')
        Ridge.fit = forbidden
        StandardScaler.fit = forbidden
        PipelineRunner.run = forbidden
        dag_ml.run_host_hpo_search_in_process = forbidden
        cohort = MultimodalDataset.from_dict(json.loads(pathlib.Path(sys.argv[2]).read_text()))
        assert cohort.y is None and cohort.independent_unit_ids and cohort.repetition_ids
        assert set(cohort.partitions) == {'predict'}
        result = nirs4all.predict(sys.argv[1], cohort, engine='dag-ml', verbose=0)
        assert result.metadata['training_performed'] is False
        assert result.metadata['scores'] is None
        print(json.dumps({'values': np.asarray(result.y_pred).ravel().tolist(),
                          'sample_ids': list(cohort.sample_ids),
                          'independent_unit_ids': list(cohort.independent_unit_ids),
                          'repetition_ids': list(cohort.repetition_ids)}))
    """)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["N4A_DAGML_INPROCESS"] = "1"
    child = subprocess.run([python, "-I", "-B", "-c", script, str(archive.resolve()), str(inputs.resolve())],
                           cwd=clean, env=environment, capture_output=True, text=True, timeout=180, check=False)
    assert child.returncode == 0, child.stdout + "\n" + child.stderr
    report = json.loads(child.stdout.strip().splitlines()[-1])
    assert report["sample_ids"] == list(cohort.sample_ids)
    assert report["independent_unit_ids"] == list(cohort.independent_unit_ids)
    assert report["repetition_ids"] == list(cohort.repetition_ids)
    np.testing.assert_allclose(report["values"], expected, rtol=1e-10, atol=1e-10)


def test_partial_late_targets_consume_native_source_target_intersection_weights(tmp_path: Path,
                                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    cohort, observer = _cohort(partial=True), _Observer()
    observer.install(monkeypatch)
    result = nirs4all.run(_pipeline(partial=True), cohort, engine="dag-ml", refit=True, save_artifacts=True,
                         save_charts=False, verbose=0, random_state=23, workspace_path=tmp_path / "partial")
    try:
        _assert_native_scopes(cohort, observer.tasks, partial=True)
        artifacts = {artifact["late_partial_refit_origin"]["source_name"]: artifact
                     for artifact in _artifacts(result) if artifact.get("late_partial_refit_origin")}
        assert set(artifacts) == {"nir", "metadata", None}
        positions = {sample: row for row, sample in enumerate(cohort.sample_ids)}
        for name, alpha in (("nir", .2), ("metadata", .3)):
            artifact, block = artifacts[name], cohort.source_values(source_names=[name])[0]
            influence = artifact["estimator"]._nirs4all_fit_influence
            ids = influence["fit_sample_ids"]
            rows = [positions[sample] for sample in ids]
            assert np.asarray(cohort.sources[name].presence_mask)[rows].all()
            for target in range(2):
                active = cohort.target_mask[rows, target]
                target_rows = np.asarray(rows)[active]
                weights = _weight_oracle(cohort, ids, target)[active]
                encoder = StandardScaler().fit(block[target_rows], sample_weight=weights)
                # Public TargetConverter stores regression targets as float32;
                # model callbacks then widen those exact stored values.
                truth = np.asarray(cohort.y, dtype=np.float32).astype(np.float64)
                head = Ridge(alpha=alpha).fit(encoder.transform(block[target_rows]), truth[target_rows, target], sample_weight=weights)
                actual = artifact["estimator"].predict(block[target_rows])[:, target]
                np.testing.assert_allclose(actual, head.predict(encoder.transform(block[target_rows])), rtol=1e-10, atol=1e-10)
        assert all(report.get("grouping_key") == {"kind": "relation_metadata", "key": "independent_unit_id"}
                   for report in result._dagml_score_set["reports"] if report["level"] == "group")
    finally:
        result.close()


@pytest.mark.parametrize("target_mask_profile", ["complete", "all_masked"])
def test_per_target_late_native_oof_weights_align_actual_observed_fit_rows(
    target_mask_profile: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real native scopes; independent weighted Ridge on genuine delivered OOF."""
    from nirs4all.pipeline.dagml.node_runner import _PerTargetLateEstimator

    cohort = _cohort(partial=True, target_mask_profile=target_mask_profile)
    observer = _Observer()
    observer.install(monkeypatch)
    records: list[dict[str, Any]] = []
    original_fit = _PerTargetLateEstimator.fit

    def observe_fit(self: Any, X: Any, y: Any, *, target_mask: Any, sample_weight: Any = None) -> Any:
        records.append({"task": deepcopy(observer.tasks[-1]), "estimator": self,
                        "X": np.array(X, copy=True), "y": np.array(y, copy=True),
                        "mask": np.array(target_mask, copy=True), "weights": np.array(sample_weight, copy=True)})
        return original_fit(self, X, y, target_mask=target_mask, sample_weight=sample_weight)

    monkeypatch.setattr(_PerTargetLateEstimator, "fit", observe_fit)
    result = nirs4all.run(_pipeline(partial=True), cohort, engine="dag-ml", refit=True, save_artifacts=True,
                         save_charts=False, verbose=0, random_state=23, workspace_path=tmp_path / target_mask_profile)
    try:
        _assert_native_scopes(cohort, observer.tasks, partial=True)
        meta_artifact = next(artifact for artifact in _artifacts(result)
                             if artifact["controller_id"] == "controller:nirs4all.meta_model")
        metadata = meta_artifact["late_partial_refit_origin"]["graph_node"]["metadata"]
        producers = metadata["prediction_source_order"]
        assert len(producers) == 2 and len(set(producers)) == 2
        positions = {sample: row for row, sample in enumerate(cohort.sample_ids)}
        meta_records = [record for record in records
                        if record["task"]["node_plan"]["controller_id"] == "controller:nirs4all.meta_model"]
        assert meta_records and {record["task"]["phase"] for record in meta_records} == {"FIT_CV", "REFIT"}
        for record in records:
            assert record["mask"].any(axis=1).all()
            assert record["X"].shape[0] == len(record["task"]["fit_influence"]["fit_sample_ids"])
            np.testing.assert_array_equal(record["weights"], record["task"]["fit_influence"]["target_row_weights"])
        for record in meta_records:
            task = record["task"]
            inputs = {spec["producer_node"]: spec for key, spec in task["prediction_inputs"].items()
                      if not key.endswith((":outer", ":test", ":refit", ":predict"))}
            assert set(inputs) == set(producers)
            universe = inputs[producers[0]]["sample_ids"]
            assert all(inputs[producer]["sample_ids"] == universe for producer in producers)
            rows = np.asarray([positions[sample] for sample in universe])
            observed = cohort.target_mask[rows].any(axis=1)
            fit_ids = [sample for sample, active in zip(universe, observed, strict=True) if active]
            assert task["fit_influence"]["fit_sample_ids"] == fit_ids
            columns = []
            for producer in producers:
                spec = inputs[producer]
                presence = np.asarray(spec["source_presence"], dtype=bool)
                validity = np.asarray(spec["feature_validity_masks"], dtype=bool)
                values = np.asarray(spec["values"])
                np.testing.assert_array_equal(validity, np.broadcast_to(presence[:, None], values.shape))
                columns.append(np.column_stack([values, presence.astype(float)]))
            full_features = np.column_stack(columns)
            np.testing.assert_array_equal(record["X"], full_features[observed])
            fit_rows = rows[observed]
            for target in range(2):
                active = cohort.target_mask[fit_rows, target]
                weights = _weight_oracle(cohort, fit_ids, target)
                truth = np.asarray(cohort.y, dtype=np.float32).astype(np.float64)
                oracle = Ridge(alpha=.4).fit(full_features[observed][active], truth[fit_rows[active], target],
                                            sample_weight=weights[active])
                np.testing.assert_allclose(record["estimator"].predict(full_features)[:, target],
                                           oracle.predict(full_features), rtol=1e-10, atol=1e-10)
            if target_mask_profile == "complete":
                assert observed.all()
                np.testing.assert_array_equal(record["weights"], np.repeat(
                    np.asarray(task["fit_influence"]["row_weights"])[:, None], 2, axis=1))
        if target_mask_profile == "all_masked":
            refit_record = next(record for record in meta_records if record["task"]["phase"] == "REFIT")
            refit_inputs = refit_record["task"]["prediction_inputs"]
            full_train_ids = next(spec["sample_ids"] for key, spec in refit_inputs.items()
                                  if not key.endswith((":refit", ":test", ":outer", ":predict")))
            assert set(full_train_ids) - set(refit_record["task"]["fit_influence"]["fit_sample_ids"])
            # Missing labels affect FIT only. Native outer/Test delivery remains complete.
            for task, output in observer.results:
                if task["node_plan"]["controller_id"] != "controller:nirs4all.meta_model":
                    continue
                deliveries = [("outer", "validation"), ("test", "test")] if task["phase"] == "FIT_CV" else [("refit", "test")]
                for suffix, partition in deliveries:
                    specs = [spec for key, spec in task["prediction_inputs"].items() if key.endswith(":" + suffix)]
                    if specs:
                        block = next(block for block in output["predictions"] if block["partition"] == partition)
                        assert block["sample_ids"] == specs[0]["sample_ids"]
    finally:
        result.close()


def test_classifier_weighted_refit_preserves_full_vocabulary_and_native_unit_vote(tmp_path: Path,
                                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    original = _cohort()
    labels = np.asarray(["healthy", "mild", "severe"])
    y = labels[[int(unit.rsplit(".", 1)[1]) % 3 for unit in original.independent_unit_ids]]
    cohort = MultimodalDataset(original.sources, sample_ids=original.sample_ids, y=y, task_type="classification",
        target_names=["diagnosis"], partitions=original.partitions, groups=original.groups,
        independent_unit_ids=original.independent_unit_ids, repetition_ids=original.repetition_ids)
    observer = _Observer()
    observer.install(monkeypatch)
    model = MultimodalClassifier({"nir": StandardScaler(), "metadata": StandardScaler()},
                                LogisticRegression(C=.4, max_iter=500))
    result = nirs4all.run([GroupKFold(3), {"model": model}], cohort, engine="dag-ml", refit=True,
        save_artifacts=True, save_charts=False, verbose=0, random_state=23,
        workspace_path=tmp_path / "classification")
    try:
        _assert_native_scopes(cohort, observer.tasks, partial=False)
        artifact = next(item for item in _artifacts(result) if item["controller_id"] == "controller:nirs4all.model")
        fitted = artifact["estimator"]
        ids = fitted._nirs4all_fit_influence["fit_sample_ids"]
        rows = [cohort.sample_ids.index(sample) for sample in ids]
        weights = _weight_oracle(cohort, ids)
        assert set(fitted.source_names) == set(cohort.sources)
        blocks = cohort.source_values(source_names=fitted.source_names)
        encoders = [StandardScaler().fit(block[rows], sample_weight=weights) for block in blocks]
        features = np.hstack([encoder.transform(block) for encoder, block in zip(encoders, blocks, strict=True)])
        vocabulary = LabelEncoder().fit(cohort.y[rows])
        oracle = LogisticRegression(C=.4, max_iter=500).fit(features[rows], vocabulary.transform(cohort.y[rows]), sample_weight=weights)
        np.testing.assert_array_equal(fitted.classes_, oracle.classes_)
        np.testing.assert_allclose(fitted.predict_proba(blocks), oracle.predict_proba(features), rtol=1e-10, atol=1e-10)
        assert fitted.predict_proba(blocks).shape == (len(cohort), 3)
        reports = [report for report in result._dagml_score_set["reports"]
                   if report["level"] == "group" and report["partition"] == "test" and report.get("fold_id") is None]
        assert len(reports) == 1 and reports[0]["row_count"] == 4
        assert reports[0]["grouping_key"] == {"kind": "relation_metadata", "key": "independent_unit_id"}
        expected = oracle.predict(features)
        correct = []
        for unit in sorted({cohort.independent_unit_ids[row] for row, partition in enumerate(cohort.partitions) if partition == "test"}):
            unit_rows = [row for row, candidate in enumerate(cohort.independent_unit_ids) if candidate == unit]
            votes = Counter(expected[unit_rows])
            winner = min(votes, key=lambda label: (-votes[label], label))
            correct.append(winner == vocabulary.transform(cohort.y[unit_rows[:1]])[0])
        np.testing.assert_allclose(reports[0]["metrics"]["accuracy"], np.mean(correct), rtol=0, atol=0)
    finally:
        result.close()


@pytest.mark.parametrize("case", ["unsupported_encoder", "cross_fold_unit", "conflicting_truth"])
def test_unit_contract_refusals_precede_all_numerical_fit(case: str, tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    # Explicit split groups correctly wrap KFold and protect whole units.
    # Omit those optional groups only in the deliberately leaking fixture.
    cohort = _cohort(conflicting_truth=case == "conflicting_truth", grouped=case != "cross_fold_unit")
    pipeline = _pipeline()
    if case == "unsupported_encoder":
        pipeline[-1]["model"].set_params(transformers={"nir": PCA(1), "metadata": StandardScaler()})
    elif case == "cross_fold_unit":
        # Unequal scan counts make five ordinary folds cut real units. Verify
        # that this fixture, without the group-preserving wrapper, leaks.
        pipeline[0] = KFold(5)
        train = [row for row, partition in enumerate(cohort.partitions) if partition == "train"]
        assert any(
            {cohort.independent_unit_ids[train[row]] for row in fit}
            & {cohort.independent_unit_ids[train[row]] for row in validation}
            for fit, validation in pipeline[0].split(np.zeros((len(train), 1)))
        )
    fitted: list[str] = []
    for model in (Ridge, StandardScaler, PCA):
        original = model.fit

        @wraps(original)
        def observed(self: Any, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            fitted.append(type(self).__name__)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(model, "fit", observed)
    with pytest.raises((ValueError, RuntimeError, DagMlRuntimeError), match="sample_weight|experimental|independent.unit|unit|truth"):
        nirs4all.run(pipeline, cohort, engine="dag-ml", save_artifacts=False,
                     save_charts=False, verbose=0, workspace_path=tmp_path / case)
    assert fitted == []
