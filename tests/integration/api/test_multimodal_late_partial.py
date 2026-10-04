"""Partial late stacks preserve genuine REFIT state through trusted archives."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import textwrap
import zipfile
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest
from nirs4all_io import MultimodalDataset
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline

import nirs4all
from nirs4all.operators.models.multimodal import TensorPCA
from nirs4all.pipeline.dagml.general_archive import load_general_archive
from nirs4all.pipeline.dagml.multimodal_contracts import validate_late_partial_stack
from nirs4all.pipeline.dagml.native_results import read_native_results
from tests.integration.api.test_multimodal_late_fusion import _cohort as complete_cohort
from tests.integration.api.test_multimodal_late_fusion import _pipeline as complete_pipeline


def _cohort(*, classification: bool = False, prediction: bool = False, absent: str | None = None,
            hidden: float = 1e7, source_policy: str = "zero_with_indicator") -> MultimodalDataset:
    original = complete_cohort(prediction=prediction)
    positions = {sample: index for index, sample in enumerate(original.sample_ids)}
    sources = {}
    for column, (name, source) in enumerate(original.sources.items()):
        rows = np.asarray([positions[sample] for sample in source.sample_ids])
        presence = np.ones(len(rows), dtype=bool) if source_policy == "error" else (rows + column) % 5 != 1
        if name == absent:
            presence[:] = False
        values = np.array(source.values, copy=True)
        if name == "metadata":
            values[~presence] = [hidden, f"unobserved_{hidden}"]
        else:
            values[~presence] = hidden
        sources[name] = replace(source, values=values, presence_mask=presence)
    labels = np.asarray(["healthy", "mild", "severe"])
    target_names = ["diagnosis"] if classification else ["concentration", "moisture"]
    target_mask = None
    targets = None
    if not prediction:
        rows = np.arange(len(original))
        if classification:
            targets = labels[rows % 3]
        else:
            targets = np.column_stack([original.y, -1.7 * original.y + np.cos(rows)])
            target_mask = np.column_stack([rows % 7 != 1, rows % 7 != 2])
            targets[~target_mask] = hidden
    return MultimodalDataset(sources, sample_ids=original.sample_ids, y=targets, target_mask=target_mask,
        target_names=target_names, task_type="classification" if classification else "regression",
        groups=original.groups, partitions=original.partitions, name="late_partial_prediction" if prediction else "late_partial_training")


def _pipeline(*, classification: bool = False, source_policy: str = "zero_with_indicator") -> list[Any]:
    pipeline = complete_pipeline()
    pipeline[1]["branch"].update(missing_source_policy=source_policy,
                                  target_policy="complete" if classification else "per_target")
    if classification:
        for branch in pipeline[1]["branch"]["steps"].values():
            branch[-1] = LogisticRegression(C=0.2, max_iter=500)
        pipeline[-1] = LogisticRegression(C=0.5, max_iter=500)
    return pipeline


def _run(cohort: Any, workspace: Path, **options: Any) -> Any:
    return nirs4all.run(_pipeline(classification=cohort.task_type == "classification",
                                source_policy=options.pop("source_policy", "zero_with_indicator")),
        cohort, engine="dag-ml", refit=True, save_artifacts=True, random_state=19,
        workspace_path=workspace, verbose=0, save_charts=False, **options)


def _meta(result: Any) -> Any:
    return next(run for run in result.runs if any(item["controller_id"] == "controller:nirs4all.meta_model"
                                                for item in run._dagml_refit_artifacts))


def _aligned(source: Any, sample_ids: Any) -> np.ndarray:
    positions = {sample: index for index, sample in enumerate(source.sample_ids)}
    return np.asarray(source.values)[[positions[sample] for sample in sample_ids]]


def _presence(source: Any, sample_ids: Any) -> np.ndarray:
    positions = {sample: index for index, sample in enumerate(source.sample_ids)}
    return np.asarray(source.presence_mask)[[positions[sample] for sample in sample_ids]]


def _oracle(training: Any, prediction: Any, artifacts: list[dict[str, Any]], *, classification: bool,
            source_policy: str = "zero_with_indicator") -> tuple[np.ndarray, np.ndarray | None]:
    """Refit each source independently on the captured native row intersections."""
    templates = _pipeline(classification=classification, source_policy=source_policy)[1]["branch"]["steps"]
    positions = {sample: index for index, sample in enumerate(training.sample_ids)}
    # SpectroDataset's public target conversion stores float32; the callback
    # promotes those stored values to float64 without recovering discarded bits.
    target_values = np.asarray(training.y) if classification else np.asarray(training.y, dtype=np.float32).astype(float)
    blocks = []
    by_source = {item["late_partial_refit_origin"]["source_name"]: item for item in artifacts}
    for name in training.sources:
        artifact = by_source[name]
        origin = artifact["late_partial_refit_origin"]
        source = training.sources[name]
        raw = _aligned(source, training.sample_ids)
        new_source = prediction.sources[name]
        new_raw = _aligned(new_source, prediction.sample_ids)
        new_positions = {sample: index for index, sample in enumerate(new_source.sample_ids)}
        present = np.asarray([new_source.presence_mask[new_positions[sample]] for sample in prediction.sample_ids])
        width = len(origin["class_labels"]) if classification else len(origin["target_names"])
        values = np.zeros((len(prediction), width))
        if classification:
            ids = origin["fit_sample_ids"]
            rows = [positions[sample] for sample in ids]
            observed = np.asarray([training.sources[name].presence_mask[list(training.sources[name].sample_ids).index(sample)] for sample in ids])
            assert observed.all() and all(training.partitions[row] == "train" for row in rows)
            model = make_pipeline(*(clone(step) for step in templates[name]))
            vocabulary = origin["public_class_labels"]
            encoded = np.asarray([vocabulary.index(label) for label in target_values[rows]], dtype=float)
            model.fit(raw[rows], encoded)
            np.testing.assert_array_equal(model.classes_, origin["class_labels"])
            if present.any():
                values[present] = model.predict_proba(new_raw[present])
                np.testing.assert_allclose(artifact["estimator"].predict_proba(new_raw[present]), values[present], rtol=1e-9, atol=1e-9)
        else:
            for column, target in enumerate(origin["target_names"]):
                ids = origin["target_fit_sample_ids"][target]
                rows = [positions[sample] for sample in ids]
                expected = [sample for sample in origin["fit_sample_ids"]
                            if training.target_mask[positions[sample], column]]
                assert ids == expected and all(training.partitions[row] == "train" for row in rows)
                model = make_pipeline(*(clone(step) for step in templates[name]))
                model.fit(raw[rows], target_values[rows, column])
                if present.any():
                    values[present, column] = model.predict(new_raw[present])
            if present.any():
                np.testing.assert_allclose(artifact["estimator"].predict(new_raw[present]), values[present], rtol=1e-9, atol=1e-9)
        if source_policy == "zero_with_indicator":
            values = np.column_stack([values, present.astype(float)])
        blocks.append(values)
    meta = by_source[None]
    features = np.column_stack(blocks)
    encoded = np.asarray(meta["estimator"].predict(features)).reshape(len(prediction), -1)
    if classification:
        probabilities = meta["estimator"].predict_proba(features)
        return np.asarray(meta["y_transform"].decode(encoded)).ravel(), probabilities
    return encoded, None


@pytest.fixture(autouse=True)
def _no_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("legacy scheduler executed"))


@pytest.mark.parametrize(("classification", "source_policy"), [(False, "error"), (False, "zero_with_indicator"), (True, "zero_with_indicator")])
def test_partial_target_or_source_refit_and_cold_replay(classification: bool, source_policy: str,
                                                       tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    training = _cohort(classification=classification, source_policy=source_policy)
    result = _run(training, tmp_path / "workspace", source_policy=source_policy)
    try:
        meta = _meta(result)
        original = meta._dagml_refit_artifacts
        assert len(original) == len(training.sources) + 1
        assert all(item["late_partial_refit_origin"]["phase"] == "REFIT" for item in original)
        assert all(item["late_partial_refit_origin"]["source_order"] == list(training.sources) for item in original)
        meta._ensure_dagml_export_results()
        persisted = read_native_results(meta._dagml_results_dir)["artifacts"]
        for artifact in persisted:
            assert artifact["content_fingerprint"] == artifact["late_partial_refit_fingerprint"]
            assert len(artifact["serialization_fingerprint"]) == 64
            assert artifact["late_partial_refit_origin"]["target_names"] == list(training.target_names)
        prediction = _cohort(classification=classification, prediction=True,
                             absent="series" if source_policy == "zero_with_indicator" else None,
                             source_policy=source_policy)
        expected, probabilities = _oracle(training, prediction, persisted, classification=classification, source_policy=source_policy)
        archive = meta.export(tmp_path / "partial.n4a")
    finally:
        result.close()
    loaded = load_general_archive(archive)
    stack = loaded["artifact"]["estimator"]
    closure = validate_late_partial_stack(stack, loaded["artifact"]["late_partial_refit_contract"])
    assert closure is not None and closure["target_policy"] == ("complete" if classification else "per_target")
    assert closure["source_order"] == list(training.sources)
    blocks = [_aligned(source, prediction.sample_ids) for source in prediction.sources.values()]
    masks = {name: _presence(source, prediction.sample_ids) for name, source in prediction.sources.items()}
    if classification:
        actual_proba = stack.predict_proba_numeric(blocks, source_masks=masks)
        assert actual_proba.shape == (len(prediction), 3)
        np.testing.assert_allclose(actual_proba.sum(axis=1), 1, atol=1e-12)
        np.testing.assert_allclose(actual_proba, probabilities, rtol=1e-9, atol=1e-9)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("cold replay attempted FIT/HPO")

    for model in (Ridge, LogisticRegression, TensorPCA):
        monkeypatch.setattr(model, "fit", forbidden)
    monkeypatch.setattr("nirs4all.pipeline.dagml.multimodal_tuning.run_multimodal_tuning", forbidden)
    if source_policy == "zero_with_indicator":
        member = next(member for member in stack.base_members if member.estimator.multimodal_source_name == "series")
        monkeypatch.setattr(member, "predict_numeric", forbidden)
        monkeypatch.setattr(member, "predict_proba_numeric", forbidden)
        # The model is still attested; the spy replaces only wrapper dispatch.
        stack.predict_numeric(blocks, source_masks=masks)
    replay = nirs4all.predict(archive, prediction, engine="dag-ml", verbose=0)
    assert replay.metadata["training_performed"] is False and replay.metadata["cross_validation"] is False
    assert replay.metadata["phase"] == "PREDICT"
    assert replay.metadata["target_names"] == list(training.target_names)
    if classification:
        np.testing.assert_array_equal(replay.y_pred, expected)
    else:
        np.testing.assert_allclose(replay.y_pred, expected, rtol=1e-9, atol=1e-9)
    for relation in replay.metadata["predict_cohort"]["relations"]["records"]:
        presence = relation["metadata"]["prediction_source_presence"]
        assert set(presence) == set(training.sources)
        if source_policy == "zero_with_indicator":
            assert presence["series"] is False


def _rewrite_archive(source: Path, target: Path, mutate: Any) -> None:
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    member = next(name for name in members if name.startswith("artifacts/") and name.endswith(".joblib"))
    model = joblib.load(io.BytesIO(members[member]))
    mutate(manifest, model)
    stream = io.BytesIO()
    joblib.dump(model, stream)
    members[member] = stream.getvalue()
    manifest["artifact_integrity"][member] = "sha256:" + hashlib.sha256(members[member]).hexdigest()
    members["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(target, "w") as archive:
        for name, contents in members.items():
            archive.writestr(name, contents)


@pytest.mark.parametrize("mutation", ["weights", "schema", "source_order", "target_names", "target_ids", "policy", "vocabulary", "origin", "marker"])
def test_archive_resigned_carrier_still_requires_original_refit(mutation: str, tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    classification = mutation == "vocabulary"
    result = _run(_cohort(classification=classification), tmp_path / "workspace")
    try:
        archive = _meta(result).export(tmp_path / "original.n4a")
    finally:
        result.close()

    def mutate(manifest: Any, stack: Any) -> None:
        base = stack.base_members[0].estimator
        if mutation == "weights":
            model = base._model if classification and hasattr(base, "_model") else base
            if not classification:
                model = base.model_.target_models_[0].model_
            if hasattr(model, "steps"):
                model = model.steps[-1][1]
            model.coef_ = np.asarray(model.coef_) + 1
        elif mutation == "schema":
            base.multimodal_input_schema = deepcopy(base.multimodal_input_schema)
            first = next(iter(base.multimodal_input_schema))
            base.multimodal_input_schema[first]["units"] = {"invented": "unit"}
        elif mutation == "source_order":
            stack.source_names = tuple(reversed(stack.source_names))
        elif mutation == "target_names":
            base.multimodal_target_names = ("renamed", "moisture")
        elif mutation == "target_ids":
            base.multimodal_target_fit_sample_ids = deepcopy(base.multimodal_target_fit_sample_ids)
            base.multimodal_target_fit_sample_ids["concentration"] = []
        elif mutation == "policy":
            base.multimodal_missing_source_policy = "error"
        elif mutation == "vocabulary":
            model = base.steps[-1][1] if hasattr(base, "steps") else base
            model.classes_ = np.asarray([0.0, 1.0, 4.0])
        elif mutation == "origin":
            stack.base_members[0].late_partial_refit_artifact["late_partial_refit_origin"]["availability"]["target_validity_masks"][0][0] = False
        else:
            manifest.pop("late_partial_refit")

    changed = tmp_path / "changed.n4a"
    _rewrite_archive(archive, changed, mutate)
    monkeypatch.setattr(Ridge, "predict", lambda *a, **k: pytest.fail("unverified carrier reached prediction"))
    monkeypatch.setattr(LogisticRegression, "predict", lambda *a, **k: pytest.fail("unverified carrier reached prediction"))
    with pytest.raises(ValueError, match="(?i)(late partial|REFIT|presence|schema|closure|missing-source policies)"):
        load_general_archive(changed)


@pytest.mark.parametrize("classification", [False, True])
def test_partial_archive_replays_from_fresh_installed_interpreter(classification: bool, tmp_path: Path) -> None:
    python = os.environ.get("NIRS4ALL_PARTIAL_LATE_INSTALLED_PYTHON")
    if not python:
        if os.environ.get("NIRS4ALL_REQUIRE_PARTIAL_LATE_INSTALLED") == "1":
            pytest.fail("NIRS4ALL_PARTIAL_LATE_INSTALLED_PYTHON is mandatory")
        pytest.skip("set a fresh installed interpreter for mandatory partial late archive qualification")
    training = _cohort(classification=classification)
    prediction = _cohort(classification=classification, prediction=True, absent="series")
    result = _run(training, tmp_path / "workspace")
    try:
        meta = _meta(result)
        expected, _ = _oracle(training, prediction, meta._dagml_refit_artifacts, classification=classification)
        archive = _meta(result).export(tmp_path / "cold.n4a")
    finally:
        result.close()
    inputs = tmp_path / "inputs.joblib"
    joblib.dump(prediction, inputs)
    output = tmp_path / "output.json"
    clean = tmp_path / "clean"
    clean.mkdir()
    script = textwrap.dedent("""\
        import importlib,json,pathlib,sys
        import joblib,nirs4all,dag_ml
        from sklearn.linear_model import Ridge,LogisticRegression
        from nirs4all.operators.models.multimodal import TensorPCA
        from nirs4all.pipeline import PipelineRunner
        prefix=pathlib.Path(sys.prefix).resolve()
        for name in ('nirs4all','dag_ml','dag_ml._dag_ml','nirs4all.api.result',
                     'nirs4all.pipeline.dagml.general_replay','nirs4all.pipeline.dagml.multimodal_contracts'):
            assert pathlib.Path(importlib.import_module(name).__file__).resolve().is_relative_to(prefix),name
        def forbidden(*a,**k):raise AssertionError('FIT/HPO entered during cold PREDICT')
        Ridge.fit=LogisticRegression.fit=TensorPCA.fit=PipelineRunner.run=forbidden
        dag_ml.run_host_hpo_search_in_process=forbidden
        replay=nirs4all.predict(sys.argv[1],joblib.load(sys.argv[2]),engine='dag-ml',verbose=0)
        assert replay.metadata['phase']=='PREDICT' and replay.metadata['training_performed'] is False
        assert replay.metadata['cross_validation'] is False
        pathlib.Path(sys.argv[3]).write_text(json.dumps({'values':replay.y_pred.tolist(),
            'targets':replay.metadata['target_names'],'origin':replay.metadata['late_partial_refit_contract']}))
        """)
    completed = subprocess.run([python, "-I", "-c", script, str(archive), str(inputs), str(output)],
                               cwd=clean, text=True, capture_output=True, check=False, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    actual = json.loads(output.read_text())
    assert actual["targets"] == list(training.target_names)
    assert actual["origin"]["source_order"] == list(training.sources)
    if classification:
        np.testing.assert_array_equal(actual["values"], expected)
    else:
        np.testing.assert_allclose(actual["values"], expected, rtol=1e-9, atol=1e-9)
