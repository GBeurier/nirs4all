"""Canonical Core ``.n4a`` SDK glue, exercised against a stubbed Core bridge.

These tests cover only the SDK boundary (routing, refusals, single produce,
exact byte publication). They make no native qualification claim: the real
``nirs4all_core`` producer, inspector and replay are not exercised here.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import types
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import nirs4all
import nirs4all.api  # noqa: F401 - initialize API exports first
from nirs4all.api.native_n4a import (
    CANONICAL_N4A_NATIVE_PROFILE,
    CanonicalN4aDependencyError,
    CanonicalN4aError,
    CanonicalN4aRunResult,
    inspect_n4a,
)
from nirs4all.pipeline.dagml.core_archive_replay import CANONICAL_N4A_V1, CoreArchiveReplayError
from nirs4all.pipeline.dagml.rt import RtError


def _sealed_container() -> bytes:
    """A stand-in container whose routing manifest carries the canonical schema."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"schema": CANONICAL_N4A_V1}))
    return buffer.getvalue()


def _retrained_container() -> bytes:
    """A different sealed canonical container: what Core returns for a fresh retrain."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"schema": CANONICAL_N4A_V1}))
        archive.writestr("retrained.marker", "fresh")
    return buffer.getvalue()


SEALED = _sealed_container()
RETRAINED = _retrained_container()
RECIPE = {
    "steps": [
        {"method_id": "n4m:standard_scale", "role": "transformer", "params": {}},
        {"method_id": "n4m:pls", "role": "regressor", "params": {}},
    ],
    "candidates": [{"n_components": 2}, {"n_components": 3}],
}


def _report(variant: str | None, partition: str, fold: str | None, rmse: float) -> dict[str, Any]:
    return {"partition": partition, "fold_id": fold, "variant_id": variant, "producer_node": "model", "level": "sample", "metrics": {"rmse": rmse}}


def _outcome() -> dict[str, Any]:
    """A Core-like TrainingOutcome: variant:b wins on OOF but is not first; variant:a's retained test is misleadingly best."""

    reports = [
        _report("variant:a", "validation", "fold0", 0.52),
        _report("variant:a", "validation", "fold1", 0.48),
        _report("variant:a", "validation", "avg", 0.50),
        _report("variant:a", "test", "fold0", 0.05),  # retained test: far better than any OOF score
        _report("variant:b", "validation", "fold0", 0.31),
        _report("variant:b", "validation", "fold1", 0.29),
        _report(None, "validation", "avg", 0.30),  # the winner's untagged OOF average
        _report(None, "final", None, 0.28),
        _report(None, "test", None, 0.35),  # the refit's retained test
    ]
    return {
        "selected_variant_id": "variant:b",
        "selection_output_id": "output:main",
        "outputs": [{"binding": {"binding_id": "output:main"}}],
        "execution_bundle": {
            "selections": {
                "policy:select": {
                    "policy_id": "policy:select",
                    "selected_candidate_id": "variant:b",
                    "metric_name": "rmse",
                    "objective": "minimize",
                    "metric_level": "sample",
                    "evaluation_scope": "oof",
                    "selected_score": 0.30,
                }
            }
        },
        "score_set": {"selection_metric": "rmse", "reports": reports},
    }


def _view(archive: bytes, mutate: Any = None, container_id: str = "container:stub") -> dict[str, Any]:
    outcome = _outcome()
    if mutate is not None:
        mutate(outcome)
    return {
        "manifest": {
            "schema": CANONICAL_N4A_V1,
            "container_id": container_id,
            "physical_profile": "stub.physical.v1",
            "artifacts": [
                {"id": f"artifact:{kind}", "kind": kind, "identity": {"fingerprint": f"fp-{kind}"}}
                for kind in ("model", "run", "result", "score")
            ],
        },
        "receipt": {
            "container_sha256": hashlib.sha256(archive).hexdigest(),
            "manifest_sha256": "0" * 64,
            "byte_length": len(archive),
        },
        "training": {"outcome": outcome, "source_id": "source:stub"},
    }


_OMIT = object()


class CoreFile:
    """Recording Core handle; only the stub interprets this test container."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.closed = False
        self.close_calls = 0

    def manifest_bytes(self) -> bytes:
        assert not self.closed
        return json.dumps({
            "schema": CANONICAL_N4A_V1, "members": [], "artifacts": [],
            "entrypoints": [], "links": [], "required_features": [],
        }).encode()

    def view(self) -> dict[str, Any]:
        assert not self.closed
        return {"readability": {
            "container_sha256": hashlib.sha256(self.payload).hexdigest(),
            "training_view": {"status": "readable", "source_id": "source:stub"},
        }}

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class Bridge:
    """Recording stand-in for the Core-side canonical bridge."""

    def __init__(self) -> None:
        self.produce_calls: list[dict[str, Any]] = []
        self.inspect_calls: list[CoreFile] = []
        self.open_calls: list[CoreFile] = []
        self.predict_calls: list[dict[str, Any]] = []
        self.retrain_calls: list[dict[str, Any]] = []
        self.refuse_inspect = False
        self.sealed = SEALED
        self.retrained = RETRAINED
        self.mutate: Any = None
        # Core's replay block is sorted by sample id; the owner also returns the physical record order.
        self.block = {"a": 1.0, "b": 2.0, "c": 3.0}
        self.physical_ids: Any = ["c", "a", "b"]

    def open_n4a_file(self, path: Path, *, memory_budget_bytes: int | None = None) -> CoreFile:
        assert memory_budget_bytes is None
        if self.refuse_inspect:
            raise ValueError("n4a integrity refusal: Core refused the container")
        payload = Path(path).read_bytes()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema") != CANONICAL_N4A_V1:
            raise ValueError("n4a unsupported format: not canonical")
        handle = CoreFile(payload)
        self.open_calls.append(handle)
        return handle

    def write_n4a_file(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("routing must not call the Core writer")

    def produce_n4a_native(self, record: bytes, source_id: str, recipe_json: str, **kwargs: Any) -> dict[str, Any]:
        self.produce_calls.append({"record": record, "source_id": source_id, "recipe_json": recipe_json, **kwargs})
        return {"archive_bytes": self.sealed, "view": _view(self.sealed, self.mutate)}

    def retrain_n4a_native(self, archive_bytes: bytes | CoreFile, **kwargs: Any) -> dict[str, Any]:
        if isinstance(archive_bytes, CoreFile):
            assert not archive_bytes.closed
        self.retrain_calls.append({"archive_bytes": archive_bytes, **kwargs})
        return {"archive_bytes": self.retrained, "view": _view(self.retrained, self.mutate, "container:retrained")}

    def inspect_n4a_bytes(self, archive_bytes: CoreFile) -> dict[str, Any]:
        self.inspect_calls.append(archive_bytes)
        if self.refuse_inspect:
            raise ValueError("Core refused the container")
        assert not archive_bytes.closed
        return _view(archive_bytes.payload)

    def predict_n4a_native(self, archive_bytes: CoreFile, **kwargs: Any) -> dict[str, Any]:
        assert not archive_bytes.closed
        self.predict_calls.append({"archive_bytes": archive_bytes, **kwargs})
        replay = {"outputs": [{"predictions": [{"sample_ids": list(self.block), "values": [[value] for value in self.block.values()]}]}]}
        result: dict[str, Any] = {"replay": replay, "receipt": {"kind": "stub-receipt"}}
        if self.physical_ids is not _OMIT:
            result["sample_ids"] = self.physical_ids
        return result


@pytest.fixture
def bridge(monkeypatch: pytest.MonkeyPatch) -> Bridge:
    stub = Bridge()
    module = types.ModuleType("nirs4all_core")
    for name in ("produce_n4a_native", "inspect_n4a_bytes", "predict_n4a_native", "retrain_n4a_native", "open_n4a_file", "write_n4a_file"):
        setattr(module, name, getattr(stub, name))
    monkeypatch.setitem(sys.modules, "nirs4all_core", module)
    return stub


@pytest.fixture
def library(tmp_path: Path) -> Path:
    path = tmp_path / "libn4m.so"
    path.write_bytes(b"stub-methods-library")
    return path


@pytest.fixture
def record_file(tmp_path: Path) -> Path:
    path = tmp_path / "record.json"
    path.write_bytes(b'{"io":"record" ,  "exact":true}\n')
    return path


def _run(record_file: Path, library: Path, **overrides: Any) -> CanonicalN4aRunResult:
    arguments: dict[str, Any] = {
        "pipeline": RECIPE,
        "dataset": record_file,
        "source_id": "source:stub",
        "engine": "native",
        "native_profile": CANONICAL_N4A_NATIVE_PROFILE,
        "methods_library_path": str(library),
        "verbose": 0,
    }
    arguments.update(overrides)
    result = nirs4all.run(**arguments)
    assert isinstance(result, CanonicalN4aRunResult)
    return result


def _write_manifest_archive(path: Path, manifest: dict[str, Any]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
    return path



def test_one_produce_across_run_export_and_predict(bridge: Bridge, library: Path, record_file: Path, tmp_path: Path) -> None:
    result = _run(record_file, library)
    destination = result.export(tmp_path / "out" / "model.n4a")

    assert len(bridge.produce_calls) == 1
    assert bridge.inspect_calls == []  # export never re-inspects
    assert destination.read_bytes() == SEALED
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == hashlib.sha256(SEALED).hexdigest()
    reference = result.native_archive_reference
    assert reference == {"container_id": "container:stub", "archive_sha256": hashlib.sha256(SEALED).hexdigest()}
    assert result.per_dataset and all(info["native_profile"] == CANONICAL_N4A_NATIVE_PROFILE for info in result.per_dataset.values())

    prediction = nirs4all.predict(
        model=destination,
        data={"record": record_file, "partition": "test"},
        engine="native",
        methods_library_path=str(library),
    )
    assert len(bridge.produce_calls) == 1  # predict never trains
    assert bridge.inspect_calls == []
    assert len(bridge.open_calls) == 1
    assert bridge.predict_calls[0]["archive_bytes"] is bridge.open_calls[0]
    assert bridge.open_calls[0].closed and bridge.open_calls[0].close_calls == 1
    assert prediction.metadata["training_performed"] is False


def test_produce_receives_exact_record_recipe_and_options(bridge: Bridge, library: Path, record_file: Path) -> None:
    _run(record_file, library)
    call = bridge.produce_calls[0]
    assert call["record"] == record_file.read_bytes()  # raw bytes, never re-encoded
    assert call["source_id"] == "source:stub"
    assert json.loads(call["recipe_json"]) == RECIPE  # steps and candidates pass through untouched
    assert call["embed_training_record"] is True
    assert call["methods_library_path"] == str(library.resolve())
    assert call["methods_library_sha256"] == hashlib.sha256(b"stub-methods-library").hexdigest()
    assert call["run_id"].startswith("run:") and call["container_id"].startswith("container:")


def test_yaml_recipe_path_is_lowered_to_json_without_changes(bridge: Bridge, library: Path, record_file: Path, tmp_path: Path) -> None:
    import yaml

    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(yaml.safe_dump(RECIPE))
    _run(record_file, library, pipeline=str(recipe_path))
    assert json.loads(bridge.produce_calls[0]["recipe_json"]) == RECIPE


def test_export_failure_leaves_existing_destination_intact(
    bridge: Bridge, library: Path, record_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run(record_file, library)
    destination = tmp_path / "model.n4a"
    destination.write_bytes(b"previous-good-archive")

    def refuse(*_: Any) -> None:
        raise OSError("replace refused")

    monkeypatch.setattr("nirs4all.api.native_n4a.os.replace", refuse)
    with pytest.raises(OSError):
        result.export(destination)
    assert destination.read_bytes() == b"previous-good-archive"
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".n4a-publish-")] == []
    assert len(bridge.produce_calls) == 1


def test_export_rejects_selectors_and_other_formats(bridge: Bridge, library: Path, record_file: Path, tmp_path: Path) -> None:
    result = _run(record_file, library)
    with pytest.raises(ValueError):
        result.export(tmp_path / "m.n4a", format="bundle")
    with pytest.raises(NotImplementedError):
        result.export(tmp_path / "m.n4a", chain_id="chain")
    with pytest.raises(ValueError):
        result.export(tmp_path / "m.bin")
    assert not (tmp_path / "m.n4a").exists()


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"random_state": 1}, NotImplementedError),
        ({"tuning": {"engine": "optuna"}}, NotImplementedError),
        ({"calibration": {"alpha": 0.1}}, NotImplementedError),
        ({"workspace_path": "/tmp/ws"}, NotImplementedError),
        ({"n_jobs": 2}, NotImplementedError),
        ({"refit": False}, NotImplementedError),
        ({"project": "p"}, NotImplementedError),
        ({"save_charts": True}, NotImplementedError),
        ({"terminal_predict": {"X": [[1.0]], "sample_ids": ["a"]}}, NotImplementedError),
        ({"pipeline": [{"model": "PLSRegression"}]}, TypeError),
        ({"dataset": {"X": [[1.0]], "y": [1.0], "sample_ids": ["a"]}}, TypeError),
        ({"source_id": None}, ValueError),
        ({"engine": "dag-ml"}, ValueError),
        ({"allow_fallback": True}, ValueError),
        (
            {"pipeline": {"steps": [{"method_id": "n4m:lda", "role": "classifier", "params": {}}], "candidates": [{}]}},
            NotImplementedError,
        ),
    ],
)
def test_unsupported_options_are_refused_before_any_fit(
    bridge: Bridge, library: Path, record_file: Path, overrides: dict[str, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        _run(record_file, library, **overrides)
    assert bridge.produce_calls == []


def test_canonical_session_is_refused_explicitly(bridge: Bridge, library: Path, record_file: Path) -> None:
    with nirs4all.session() as session:
        with pytest.raises(NotImplementedError):
            _run(record_file, library, session=session)
    assert bridge.produce_calls == []


def test_missing_core_bridge_is_a_typed_refusal(
    bridge: Bridge, library: Path, record_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = types.ModuleType("nirs4all_core")  # an older Core without the canonical bridge
    monkeypatch.setitem(sys.modules, "nirs4all_core", module)
    with pytest.raises(CanonicalN4aDependencyError):
        _run(record_file, library)


def test_receipt_that_does_not_match_sealed_bytes_is_refused(bridge: Bridge, library: Path, record_file: Path) -> None:
    bridge.sealed = SEALED  # produce() still returns a view built for these bytes...
    original = bridge.produce_n4a_native

    def tampered(*args: Any, **kwargs: Any) -> dict[str, Any]:
        produced = original(*args, **kwargs)
        produced["archive_bytes"] = SEALED + b"x"
        return produced

    bridge.produce_n4a_native = tampered  # type: ignore[method-assign]
    sys.modules["nirs4all_core"].produce_n4a_native = tampered  # type: ignore[attr-defined]
    with pytest.raises(CanonicalN4aError):
        _run(record_file, library)


def test_the_untrusted_interim_v2_v3_hint_no_longer_exists() -> None:
    """Native Package V2/V3 containers are classified from the Core-opened manifest, never from a ZIP manifest peek."""
    from nirs4all.pipeline.dagml import core_archive_replay, host_container

    assert not hasattr(core_archive_replay, "detect_core_archive_version")
    assert not hasattr(host_container, "_interim_archive_version")


def test_canonical_load_session_never_reaches_bundle_loader(bridge: Bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import nirs4all.pipeline.bundle as bundle_module

    def forbidden(*_: Any, **__: Any) -> None:
        raise AssertionError("BundleLoader must not be reached")

    monkeypatch.setattr(bundle_module, "BundleLoader", forbidden)
    canonical = _write_manifest_archive(tmp_path / "canonical.n4a", {"schema": CANONICAL_N4A_V1})
    with pytest.raises(NotImplementedError):
        nirs4all.load_session(canonical)
    with pytest.raises(NotImplementedError):
        nirs4all.load_session(canonical, engine="native")


def test_canonical_predict_validates_via_core_and_refuses_corruption_without_fallback(
    bridge: Bridge, library: Path, record_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nirs4all.pipeline.bundle as bundle_module

    def forbidden(*_: Any, **__: Any) -> None:
        raise AssertionError("BundleLoader must not be reached")

    monkeypatch.setattr(bundle_module, "BundleLoader", forbidden)
    canonical = _write_manifest_archive(tmp_path / "canonical.n4a", {"schema": CANONICAL_N4A_V1})
    data = {"record": record_file, "partition": "test"}

    prediction = nirs4all.predict(model=canonical, data=data, engine="native", methods_library_path=str(library))
    call = bridge.predict_calls[0]
    assert call["entrypoint"] == "predict-external"
    assert call["source_id"] == "source:stub"  # taken from the Core-inspected view
    assert call["partition"] == "test"
    assert call["record"] == record_file.read_bytes()
    assert call["require_training_schema"] is True
    assert call["methods_library_sha256"] == hashlib.sha256(b"stub-methods-library").hexdigest()
    np.testing.assert_array_equal(prediction.y_pred, np.array([[3.0], [1.0], [2.0]]))  # owner physical order c, a, b
    assert prediction.metadata["receipt"] == {"kind": "stub-receipt"}
    assert bridge.produce_calls == []

    bridge.refuse_inspect = True
    with pytest.raises(ValueError, match="Core refused"):
        nirs4all.predict(model=canonical, data=data, engine="native", methods_library_path=str(library))
    assert len(bridge.predict_calls) == 1  # the refused container was never replayed

    bridge.refuse_inspect = False
    with pytest.raises(CoreArchiveReplayError):
        nirs4all.predict(model=canonical, data=data, engine="legacy", methods_library_path=str(library))
    with pytest.raises(NotImplementedError):
        nirs4all.predict(model=canonical, data=data, engine="native", coverage=0.9, methods_library_path=str(library))
    with pytest.raises(TypeError):
        nirs4all.predict(model=canonical, data={"X": [[1.0]], "sample_ids": ["a"]}, engine="native")
    assert len(bridge.predict_calls) == 1


def test_inspect_n4a_returns_the_core_view(bridge: Bridge, tmp_path: Path) -> None:
    path = tmp_path / "model.n4a"
    path.write_bytes(SEALED)
    view = inspect_n4a(path)
    assert view["manifest"]["container_id"] == "container:stub"
    assert bridge.inspect_calls == bridge.open_calls
    assert len(bridge.open_calls) == 1 and bridge.open_calls[0].close_calls == 1
    assert bridge.open_calls[0].payload == SEALED


def test_public_scores_follow_the_owner_selected_oof_decision(bridge: Bridge, library: Path, record_file: Path) -> None:
    result = _run(record_file, library)

    # Core's OOF decision, not the retained test (0.35 refit / 0.05 loser fold) and not report order (variant:a is first).
    assert result.best_score == 0.30
    assert result.best_rmse == 0.30
    assert math.isnan(result.best_r2) and math.isnan(result.best_accuracy)  # no test-score substitute
    assert result.selection == {
        "selected_variant_id": "variant:b",
        "selection_output_id": "output:main",
        "policy_id": "policy:select",
        "metric_name": "rmse",
        "objective": "minimize",
        "metric_level": "sample",
        "evaluation_scope": "oof",
        "selected_score": 0.30,
    }
    assert result.cv_best_score == pytest.approx(0.30)

    # The retained measurements stay on their own variant's rows and are never the public score.
    assert result.best["test_score"] == pytest.approx(0.35)
    rows = result.predictions.filter_predictions(load_arrays=False)
    refit_variants = {row["result_metadata"]["dagml_projection"]["variant_id"] for row in rows if row["fold_id"] == "final"}
    assert refit_variants == {"variant:b"}
    loser_tests = [row for row in rows if row["partition"] == "test" and row["result_metadata"]["dagml_projection"]["variant_id"] == "variant:a"]
    assert [row["test_score"] for row in loser_tests] == [pytest.approx(0.05)]


def _select_mae(outcome: dict[str, Any]) -> None:
    """Core selects on MAE; the retained final/test measurements are misleadingly far better than any OOF value."""

    for report in outcome["score_set"]["reports"]:
        report["metrics"]["mae"] = 0.02 if report["partition"] in {"final", "test"} else report["metrics"]["rmse"]
    outcome["score_set"]["selection_metric"] = "mae"
    outcome["execution_bundle"]["selections"]["policy:select"].update(metric_name="mae", selected_score=0.30)


def test_best_mae_is_the_owner_selected_oof_score_when_core_selected_mae(bridge: Bridge, library: Path, record_file: Path) -> None:
    bridge.mutate = _select_mae
    result = _run(record_file, library)

    assert result.selection["metric_name"] == "mae"
    assert result.best_mae == 0.30 and result.best_score == 0.30  # the OOF decision, not the retained 0.02
    assert math.isnan(result.best_rmse) and math.isnan(result.best_r2)  # not the selection metric: no test substitute
    assert result.best["test_score"] == pytest.approx(0.02)  # retained test stays on its row, never public


def test_best_mae_is_none_and_other_shortcuts_unchanged_when_core_selected_rmse(bridge: Bridge, library: Path, record_file: Path) -> None:
    result = _run(record_file, library)

    assert result.selection["metric_name"] == "rmse"
    assert result.best_mae is None
    assert result.best_rmse == 0.30 and result.best_score == 0.30
    assert math.isnan(result.best_r2)


def _drop(key: str) -> Any:
    return lambda outcome: outcome.pop(key)


def _decision(**changes: Any) -> Any:
    def mutate(outcome: dict[str, Any]) -> None:
        outcome["execution_bundle"]["selections"]["policy:select"].update(changes)

    return mutate


def _set(path: str, value: Any) -> Any:
    def mutate(outcome: dict[str, Any]) -> None:
        outcome[path] = value

    return mutate


def _both(*mutations: Any) -> Any:
    def mutate(outcome: dict[str, Any]) -> None:
        for mutation in mutations:
            mutation(outcome)

    return mutate


def _score_set(**changes: Any) -> Any:
    return lambda outcome: outcome["score_set"].update(changes)


def _refit_owned_by_loser(outcome: dict[str, Any]) -> None:
    for report in outcome["score_set"]["reports"]:
        if report["partition"] == "final":
            report["variant_id"] = "variant:a"


@pytest.mark.parametrize(
    "mutation",
    [
        _drop("selected_variant_id"),
        _drop("selection_output_id"),
        _set("selection_output_id", "output:unknown"),
        _set("selected_variant_id", "variant:a"),  # the decision still selects variant:b
        _decision(selected_candidate_id="variant:a"),
        _both(_set("selected_variant_id", "variant:z"), _decision(selected_candidate_id="variant:z")),  # no OOF evidence for it
        lambda outcome: outcome["execution_bundle"].update(selections={}),
        lambda outcome: outcome["execution_bundle"]["selections"].update(
            {"policy:other": dict(outcome["execution_bundle"]["selections"]["policy:select"], policy_id="policy:other")}
        ),
        _decision(policy_id="policy:renamed"),
        _decision(evaluation_scope="holdout"),
        _decision(objective="maximize"),
        _decision(selected_score=float("nan")),
        _decision(metric_level=None),
        _score_set(selection_metric="mae"),
        _both(_decision(metric_name="not_a_metric"), _score_set(selection_metric="not_a_metric")),
        _refit_owned_by_loser,
    ],
)
def test_missing_or_mismatched_owner_selection_is_refused(bridge: Bridge, library: Path, record_file: Path, mutation: Any, tmp_path: Path) -> None:
    bridge.mutate = mutation
    with pytest.raises(CanonicalN4aError):
        _run(record_file, library)
    assert len(bridge.produce_calls) == 1  # Core's single fit was not repeated or replaced by a Python winner


def test_selected_variant_projection_keeps_historical_callers_and_single_variant_runs(bridge: Bridge) -> None:
    from nirs4all.pipeline.dagml.result import _scores_to_run_result

    untagged = {"reports": [_report(None, "validation", "fold0", 0.4), _report(None, "validation", "avg", 0.4), _report(None, "final", None, 0.1), _report(None, "test", None, 0.2)]}
    explicit = _scores_to_run_result(untagged, "dataset", "pls", "rmse", "regression", selected_variant_id="variant:base")
    assert {row["result_metadata"]["dagml_projection"]["variant_id"] for row in explicit.predictions.filter_predictions(load_arrays=False)} == {"variant:base"}
    historical = _scores_to_run_result(untagged, "dataset", "pls", "rmse", "regression")
    assert {row["result_metadata"]["dagml_projection"]["variant_id"] for row in historical.predictions.filter_predictions(load_arrays=False)} == {None}


def _predict(bridge: Bridge, library: Path, record_file: Path, tmp_path: Path, **options: Any) -> Any:
    archive = tmp_path / "canonical.n4a"
    if not archive.exists():
        _write_manifest_archive(archive, {"schema": CANONICAL_N4A_V1})
    return nirs4all.predict(
        model=archive, data={"record": record_file, "partition": "test"}, engine="native", methods_library_path=str(library), **options
    )


def test_predictions_follow_the_owner_physical_order_not_the_sorted_replay_block(
    bridge: Bridge, library: Path, record_file: Path, tmp_path: Path
) -> None:
    result = _predict(bridge, library, record_file, tmp_path)

    assert result.sample_ids == ["c", "a", "b"] and result.metadata["sample_ids"] == ["c", "a", "b"]
    np.testing.assert_array_equal(result.y_pred, [[3.0], [1.0], [2.0]])
    assert result.sample_indices is None  # original indices are never invented
    # The signed owner replay and receipt are returned untouched (block still sorted by id).
    block = result.metadata["replay"]["outputs"][0]["predictions"][0]
    assert block["sample_ids"] == ["a", "b", "c"] and block["values"] == [[1.0], [2.0], [3.0]]
    assert result.metadata["receipt"] == {"kind": "stub-receipt"}

    # Against the record's true targets (in record order), positional use of the sorted block is badly wrong.
    truth = np.array([[3.1], [1.1], [2.1]])
    sorted_block = np.asarray(block["values"])
    positional_rmse = float(np.sqrt(np.mean((sorted_block - truth) ** 2)))
    aligned_rmse = float(np.sqrt(np.mean((result.y_pred - truth) ** 2)))
    assert positional_rmse == pytest.approx(math.sqrt((2.1**2 + 0.9**2 + 0.9**2) / 3))
    assert aligned_rmse == pytest.approx(0.1)

    frame = result.to_dataframe()
    assert list(frame["sample_id"]) == ["c", "a", "b"] and list(frame["y_pred"]) == [3.0, 1.0, 2.0]
    assert "sample_index" not in frame


def test_permuting_the_record_only_reorders_rows_per_sample_id(bridge: Bridge, library: Path, record_file: Path, tmp_path: Path) -> None:
    by_id: list[dict[str, float]] = []
    for order in (["c", "a", "b"], ["b", "c", "a"], ["a", "b", "c"]):
        bridge.physical_ids = order  # the same record, rows permuted; Core's block stays sorted
        result = _predict(bridge, library, record_file, tmp_path)
        assert result.sample_ids == order
        by_id.append(dict(zip(result.sample_ids, result.y_pred[:, 0].tolist(), strict=True)))
    assert by_id[0] == by_id[1] == by_id[2] == {"a": 1.0, "b": 2.0, "c": 3.0}


@pytest.mark.parametrize(
    ("physical_ids", "block"),
    [
        (_OMIT, None),  # owner did not return the physical order
        (None, None),
        ("abc", None),
        ([], None),
        (["c", "a"], None),  # missing id
        (["c", "a", "b", "d"], None),  # extra id
        (["c", "a", "z"], None),  # different set, same length
        (["c", "a", "a"], None),  # duplicate in the owner order
        (["c", "a", ""], None),  # empty id
        (["c", "a", 3], None),  # non-string id
        (["c", "a", "b"], {"a": 1.0, "b": 2.0, "b ": 3.0}),  # block ids differ from the owner order
    ],
)
def test_missing_or_inconsistent_owner_order_is_refused(
    bridge: Bridge, library: Path, record_file: Path, tmp_path: Path, physical_ids: Any, block: Any
) -> None:
    bridge.physical_ids = physical_ids
    if block is not None:
        bridge.block = block
    with pytest.raises(CanonicalN4aError):
        _predict(bridge, library, record_file, tmp_path)


def test_duplicate_ids_in_the_replay_block_are_refused(bridge: Bridge, library: Path, record_file: Path, tmp_path: Path) -> None:
    original = bridge.predict_n4a_native

    def duplicated(archive_bytes: bytes, **kwargs: Any) -> dict[str, Any]:
        result = original(archive_bytes, **kwargs)
        result["replay"]["outputs"][0]["predictions"][0]["sample_ids"] = ["a", "a", "c"]
        return result

    sys.modules["nirs4all_core"].predict_n4a_native = duplicated  # type: ignore[attr-defined]
    with pytest.raises(CanonicalN4aError, match="duplicates"):
        _predict(bridge, library, record_file, tmp_path)


@pytest.mark.parametrize(
    "options",
    [{"name": "custom"}, {"workspace_path": "/nonexistent/workspace"}, {"workspace_metadata": {"k": "v"}}, {"workspace_result_metadata": {"k": "v"}}],
)
def test_canonical_predict_refuses_non_neutral_name_and_workspace(
    bridge: Bridge, library: Path, record_file: Path, tmp_path: Path, options: dict[str, Any]
) -> None:
    with pytest.raises(NotImplementedError):
        _predict(bridge, library, record_file, tmp_path, **options)
    assert bridge.predict_calls == []  # refused before any Core call, no fake workspace

    # The neutral defaults and an explicit verbosity are honored.
    assert _predict(bridge, library, record_file, tmp_path, verbose=1).sample_ids == ["c", "a", "b"]
    with pytest.raises(ValueError):
        _predict(bridge, library, record_file, tmp_path, verbose=9)


@pytest.mark.parametrize(
    "text",
    [
        "steps: []\ncandidates: []\nsteps: []\n",  # top-level duplicate
        "steps:\n  - method_id: n4m:pls\n    role: regressor\n    params: {alpha: 0.1, alpha: 1.0}\ncandidates: [{}]\n",  # nested duplicate
    ],
)
def test_yaml_recipe_with_duplicate_keys_is_refused_without_a_fit(
    bridge: Bridge, library: Path, record_file: Path, tmp_path: Path, text: str
) -> None:
    recipe_path = tmp_path / "dup.yaml"
    recipe_path.write_text(text)
    with pytest.raises(ValueError, match="duplicate key"):
        _run(record_file, library, pipeline=str(recipe_path))
    assert bridge.produce_calls == []


# --- Canonical retrain: public routing and owner boundary (stubbed Core; no native claim) ---------------------------


NEW_RECORD = b'{"io":"new-record","exact":true}\n'


@pytest.fixture
def source_archive(tmp_path: Path) -> Path:
    path = tmp_path / "source.n4a"
    path.write_bytes(SEALED)
    return path


def _retrain(source: Any, library: Path, data: Any = NEW_RECORD, **overrides: Any) -> CanonicalN4aRunResult:
    arguments: dict[str, Any] = {"engine": "native", "methods_library_path": str(library), "verbose": 0}
    arguments.update(overrides)
    result = nirs4all.retrain(source, data, **arguments)
    assert isinstance(result, CanonicalN4aRunResult)
    return result


def test_retrain_path_runs_one_owner_action_and_returns_fresh_sealed_bytes(
    bridge: Bridge, library: Path, source_archive: Path, tmp_path: Path
) -> None:
    result = _retrain(source_archive, library)

    assert len(bridge.retrain_calls) == 1
    assert bridge.produce_calls == [] and bridge.inspect_calls == [] and bridge.predict_calls == []
    call = bridge.retrain_calls[0]
    assert call["archive_bytes"] is bridge.open_calls[0]
    assert len(bridge.open_calls) == 1 and bridge.open_calls[0].close_calls == 1
    assert bridge.open_calls[0].payload == SEALED
    assert call["record"] == NEW_RECORD  # exact bytes, never re-encoded
    assert set(call) == {
        "archive_bytes", "record", "methods_library_path", "methods_library_sha256", "run_id", "container_id", "embed_training_record",
    }  # no caller recipe, source id or seed
    assert call["embed_training_record"] is True
    assert call["methods_library_path"] == str(library.resolve())
    assert call["methods_library_sha256"] == hashlib.sha256(b"stub-methods-library").hexdigest()
    assert call["run_id"].startswith("run:") and call["container_id"].startswith("container:")

    assert result.native_archive_reference == {"container_id": "container:retrained", "archive_sha256": hashlib.sha256(RETRAINED).hexdigest()}
    assert result.canonical_view["manifest"]["container_id"] == "container:retrained"
    assert result.per_dataset and all(info["native_profile"] == CANONICAL_N4A_NATIVE_PROFILE for info in result.per_dataset.values())
    destination = result.export(tmp_path / "out" / "retrained.n4a")
    assert destination.read_bytes() == RETRAINED
    assert source_archive.read_bytes() == SEALED  # the source container is unchanged
    assert len(bridge.inspect_calls) == 0  # export never re-inspects


def test_retrain_from_a_result_leaves_the_original_result_and_bytes_unchanged(
    bridge: Bridge, library: Path, record_file: Path, tmp_path: Path
) -> None:
    original = _run(record_file, library)
    before = (original.native_archive_reference, original.selection, original.canonical_view)

    fresh = _retrain(original, library, {"record": record_file})

    assert len(bridge.produce_calls) == 1 and len(bridge.retrain_calls) == 1  # the retrain is not a second produce
    assert bridge.retrain_calls[0]["archive_bytes"] == SEALED
    assert bridge.retrain_calls[0]["record"] == record_file.read_bytes()
    assert fresh is not original
    assert fresh.native_archive_reference["archive_sha256"] == hashlib.sha256(RETRAINED).hexdigest()
    assert (original.native_archive_reference, original.selection, original.canonical_view) == before
    assert original.export(tmp_path / "original.n4a").read_bytes() == SEALED
    assert fresh.export(tmp_path / "fresh.n4a").read_bytes() == RETRAINED


def test_retrain_scores_are_the_owner_selected_oof_decision(bridge: Bridge, library: Path, source_archive: Path) -> None:
    result = _retrain(source_archive, library)
    assert result.best_score == 0.30 and result.best_rmse == 0.30  # Core's OOF decision, not the retained refit test (0.35)
    assert result.selection["selected_variant_id"] == "variant:b" and result.selection["evaluation_scope"] == "oof"
    assert result.best_mae is None and math.isnan(result.best_r2)

    bridge.mutate = _select_mae
    selected_mae = _retrain(source_archive, library)
    assert selected_mae.best_mae == 0.30 and selected_mae.best_score == 0.30  # not the retained 0.02
    assert selected_mae.best["test_score"] == pytest.approx(0.02)


def test_retrain_uses_the_public_name_for_the_dataset(bridge: Bridge, library: Path, source_archive: Path) -> None:
    assert list(_retrain(source_archive, library, name="new_cohort").per_dataset) == ["new_cohort"]
    assert list(_retrain(source_archive, library).per_dataset) == ["retrain_dataset"]


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"new_model": object()}, NotImplementedError),
        ({"epochs": 3}, NotImplementedError),
        ({"random_state": 1}, NotImplementedError),
        ({"refit": False}, NotImplementedError),
        ({"cache": object()}, NotImplementedError),
        ({"project": "p"}, NotImplementedError),
        ({"workspace_path": "/tmp/ws"}, NotImplementedError),
        ({"learning_rate": 0.1}, NotImplementedError),
        ({"save_artifacts": False}, ValueError),
        ({"verbose": 9}, ValueError),
        ({"verbose": True}, ValueError),
        ({"name": 3}, TypeError),
        ({"data": b""}, ValueError),
        ({"data": {"record": NEW_RECORD, "partition": "train"}}, TypeError),
        ({"data": {"X": [[1.0]], "y": [1.0], "sample_ids": ["a"]}}, TypeError),
        ({"data": np.zeros((2, 2))}, TypeError),
        ({"data": (np.zeros((2, 2)), np.zeros(2))}, TypeError),
    ],
)
def test_retrain_unsupported_options_and_data_are_refused_before_any_fit(
    bridge: Bridge, library: Path, source_archive: Path, overrides: dict[str, Any], error: type[Exception]
) -> None:
    arguments: dict[str, Any] = dict(overrides)
    data = arguments.pop("data", NEW_RECORD)
    with pytest.raises(error):
        _retrain(source_archive, library, data, **arguments)
    assert bridge.retrain_calls == [] and bridge.produce_calls == []


def test_retrain_refuses_a_wrong_source_type_before_any_fit(bridge: Bridge, library: Path) -> None:
    from nirs4all.api.native_n4a import retrain_canonical_n4a

    with pytest.raises(TypeError):
        retrain_canonical_n4a({"model_name": "PLS"}, NEW_RECORD, options={"methods_library_path": str(library)})
    assert bridge.retrain_calls == []


def test_canonical_retrain_never_reaches_bundle_v3_runner_or_legacy_lanes(
    bridge: Bridge, library: Path, record_file: Path, source_archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from importlib import import_module

    import nirs4all.api.native_training as native_training
    import nirs4all.pipeline as pipeline_module
    import nirs4all.pipeline.bundle as bundle_module

    retrain_module = import_module("nirs4all.api.retrain")

    def forbidden(*_: Any, **__: Any) -> None:
        raise AssertionError("a canonical retrain must not reach this lane")

    monkeypatch.setattr(retrain_module, "_bundle_training_spec", forbidden)
    monkeypatch.setattr(retrain_module, "_native_full_retrain", forbidden)
    monkeypatch.setattr(retrain_module, "require_dagml_retrain_backend", forbidden)
    monkeypatch.setattr(native_training, "refit_native_methods", forbidden)
    monkeypatch.setattr(pipeline_module, "PipelineRunner", forbidden)
    monkeypatch.setattr(bundle_module, "BundleLoader", forbidden)

    _retrain(source_archive, library)
    _retrain(_run(record_file, library), library)  # a CanonicalN4aRunResult is not a Methods V3 parent
    assert len(bridge.retrain_calls) == 2


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"engine": None}, CoreArchiveReplayError),  # the omitted selector means DAG-ML, never canonical Core
        ({"engine": "dag-ml"}, CoreArchiveReplayError),
        ({"engine": "legacy"}, CoreArchiveReplayError),
        ({"engine": None, "mode": "transfer", "plugin": "nirs4all-python-library"}, CoreArchiveReplayError),
        ({"mode": "transfer"}, RtError),
        ({"mode": "finetune"}, RtError),
        ({"allow_fallback": True}, RtError),
    ],
)
def test_canonical_retrain_respects_explicit_engine_and_mode_selection(
    bridge: Bridge, library: Path, source_archive: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any], error: type[Exception]
) -> None:
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.delenv("N4A_RETRAIN_PLUGIN", raising=False)
    with pytest.raises(error):
        _retrain(source_archive, library, **overrides)
    assert bridge.retrain_calls == [] and bridge.produce_calls == []


def test_canonical_retrain_refuses_a_session_before_any_fit(bridge: Bridge, library: Path, source_archive: Path) -> None:
    with nirs4all.session() as session:
        with pytest.raises(RtError):
            _retrain(source_archive, library, session=session)
    assert bridge.retrain_calls == []


def test_missing_retrain_bridge_refuses_retrain_only(
    bridge: Bridge, library: Path, record_file: Path, source_archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(sys.modules["nirs4all_core"], "retrain_n4a_native")  # an older Core without the retrain action
    with pytest.raises(CanonicalN4aDependencyError, match="retrain_n4a_native"):
        _retrain(source_archive, library)
    assert bridge.retrain_calls == [] and bridge.produce_calls == []

    # Unrelated canonical produce, inspect and predict keep working.
    result = _run(record_file, library)
    assert len(bridge.produce_calls) == 1
    assert inspect_n4a(source_archive)["manifest"]["container_id"] == "container:stub"
    assert _predict(bridge, library, record_file, source_archive.parent).sample_ids == ["c", "a", "b"]
    assert result.native_archive_reference["container_id"] == "container:stub"


def test_owner_retrain_refusal_propagates_without_any_fallback(bridge: Bridge, library: Path, source_archive: Path) -> None:
    def refuse(*_: Any, **__: Any) -> None:
        raise ValueError("Core refused the retrain record")

    sys.modules["nirs4all_core"].retrain_n4a_native = refuse  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="Core refused the retrain record"):
        _retrain(source_archive, library)
    assert bridge.produce_calls == []  # no separate produce or Python-side recipe
    assert len(bridge.open_calls) == 1 and bridge.open_calls[0].close_calls == 1
    assert source_archive.read_bytes() == SEALED


def test_retrain_receipt_that_does_not_match_the_sealed_bytes_is_refused(bridge: Bridge, library: Path, source_archive: Path) -> None:
    original = bridge.retrain_n4a_native

    def tampered(*args: Any, **kwargs: Any) -> dict[str, Any]:
        produced = original(*args, **kwargs)
        produced["archive_bytes"] = RETRAINED + b"x"
        return produced

    sys.modules["nirs4all_core"].retrain_n4a_native = tampered  # type: ignore[attr-defined]
    with pytest.raises(CanonicalN4aError):
        _retrain(source_archive, library)


@pytest.mark.parametrize(
    "mutation",
    [_drop("selected_variant_id"), _decision(selected_candidate_id="variant:a"), _decision(evaluation_scope="holdout"), _score_set(selection_metric="mae")],
)
def test_retrain_with_a_missing_or_mismatched_owner_selection_is_refused(
    bridge: Bridge, library: Path, source_archive: Path, mutation: Any
) -> None:
    bridge.mutate = mutation
    with pytest.raises(CanonicalN4aError):
        _retrain(source_archive, library)
    assert len(bridge.retrain_calls) == 1  # Core's fit was neither repeated nor replaced by a Python winner


@pytest.fixture
def host_inspection_file(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    """Real Core-written host evidence with inert fitted bytes; never fit/load."""
    import nirs4all_core  # noqa: F401 - required integration dependency, never skip

    from nirs4all.pipeline.bundle.constants import BUNDLE_FORMAT_VERSION
    from nirs4all.pipeline.dagml.host_container import (
        BUNDLE_CONTENT_TYPE,
        BUNDLE_MANIFEST_MEMBER,
        JOBLIB_CONTENT_TYPE,
        JSON_CONTENT_TYPE,
        HostMember,
        write_host_container,
    )

    values = {
        BUNDLE_MANIFEST_MEMBER: {"bundle_format_version": BUNDLE_FORMAT_VERSION, "source_type": "inspection_synthetic",
                                 "relation_replay_manifest": {"path": "relations.json"},
                                 "unknown_additive": {"raw": "<Object at 0xabc>", "targets": ["t:b", "t:a"]}},
        "pipeline.json": {"steps": [{"model": {"class": "CosmeticLabel"}}]},
        "train_pipeline.json": {"steps": [{"branch": [[{"model": "sklearn.linear_model.Ridge"}],
                                                       [{"y_processing": {"class": "sklearn.preprocessing.StandardScaler"}}]]}],
                                "options": {"generation": [2, 1]}, "targets": ["t:b", "t:a"]},
        "chain.json": {"unknown_chain": ["branch:right", "branch:left"]},
        "trace.json": {"trace_id": "trace:synthetic", "pipeline_uid": "pipeline:synthetic", "steps": []},
        "fold_weights.json": {"0": 1.0},
        "relations.json": {"sample_ids": ["s:2", "s:1"], "origin_ids": ["origin:2", "origin:1"]},
        "extra_scores.json": {"scores": [{"metric": "rmse", "value": 1.25, "target": "t:b",
                                         "partition": "validation", "level": "sample", "selection": "owner", "aggregation": "mean"}]},
    }
    raw = {name: json.dumps(value).encode() for name, value in values.items()}
    members = [HostMember(name, data, "provenance" if name == BUNDLE_MANIFEST_MEMBER else "recipe",
                          BUNDLE_CONTENT_TYPE if name == BUNDLE_MANIFEST_MEMBER else JSON_CONTENT_TYPE)
               for name, data in raw.items()]
    members.append(HostMember("artifacts/step_1_foldfinal_inert.joblib", b"not pickle; never hydrate", "model", JOBLIB_CONTENT_TYPE))
    path = tmp_path / "sdk-host.n4a"
    write_host_container(path, members)
    return path, raw


def test_host_json_inspection_exact_roles_no_hydration_and_default_unchanged(host_inspection_file, monkeypatch):
    import pickle

    import joblib
    import nirs4all_core as core

    from nirs4all.pipeline.bundle import loader
    from nirs4all.pipeline.dagml import general_archive, host_container

    def forbidden(*args, **kwargs):
        pytest.fail("inspection entered fitted-object hydration")

    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    monkeypatch.setattr(host_container, "load_host_joblib", forbidden)
    monkeypatch.setattr(loader, "load_host_joblib", forbidden)
    monkeypatch.setattr(general_archive, "load_general_archive", forbidden)
    monkeypatch.setattr(loader.BundleArtifactProvider, "get_artifact", forbidden)
    monkeypatch.setattr(loader.BundleArtifactProvider, "get_artifacts_for_step", forbidden)
    monkeypatch.setattr(loader.BundleArtifactProvider, "_load_artifact", forbidden)
    path, raw = host_inspection_file
    opens = []
    original = core.open_n4a_file
    def counted(*args, **kwargs):
        opens.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(core, "open_n4a_file", counted)
    with core.open_n4a_file(path) as outer:
        expected = outer.view()
        with host_container.open_n4a_source(path, core_file=outer) as source:
            assert inspect_n4a(path, source=source) == expected
            result = nirs4all.inspect_n4a(path, source=source, include_host_json=True)
            assert source.container.core_file is outer
            assert not outer.closed
        assert not outer.closed
    assert len(opens) == 1
    assert result["view"] == expected
    assert result["bundle_manifest_bytes"] == raw[host_container.BUNDLE_MANIFEST_MEMBER]
    assert result["profile"] == host_container.HOST_OWNER
    assert result["operation"] == "host.predict"
    documents = {item["member"]: item for item in result["documents"]}
    assert set(documents) == set(raw)
    for name, data in raw.items():
        assert documents[name]["bytes"] == data
        assert documents[name]["json"] == json.loads(data)
        assert documents[name]["inventory_sha256"] == hashlib.sha256(data).hexdigest()
        assert documents[name]["byte_length"] == len(data)
    assert documents["train_pipeline.json"]["role"] == "training_declaration"
    assert documents["pipeline.json"]["role"] == "prediction_config"
    assert documents["relations.json"]["role"] == "relation_replay_manifest"
    assert documents["extra_scores.json"]["role"] == "host_json"
    assert documents["trace.json"]["role"] == "trace"
    assert documents["chain.json"]["role"] == "chain"
    assert documents["fold_weights.json"]["role"] == "fold_weights"
    assert result["validation"] == {"bundle_loader_pre_model": True, "bundle_format": "checked",
                                     "independent_output_topology": "not_applicable", "source_training_capture": "not_present",
                                     "scientific_scoring": "not_evaluated", "fitted_objects": "not_loaded"}
    assert result["opaque_artifacts"] and all("bytes" not in item and "json" not in item for item in result["opaque_artifacts"])
    assert inspect_n4a(path) == expected  # unchanged default public result


def test_host_json_bundleless_inspection_reports_validation_scope(tmp_path):
    import nirs4all_core  # noqa: F401 - required, never skip

    from nirs4all.pipeline.dagml.host_container import JSON_CONTENT_TYPE, HostMember, write_host_container

    path = tmp_path / "bundleless.n4a"
    write_host_container(path, [
        HostMember("result.json", b'{"score":null,"target_ids":["t:b","t:a"]}', "result", JSON_CONTENT_TYPE),
        HostMember("dagml_source_training_capture.json", b'{"note":"bundle-less JSON inspection only; no completed run"}', "provenance", JSON_CONTENT_TYPE),
    ], entrypoint=False)
    result = inspect_n4a(path, include_host_json=True)
    assert result["operation"] is None
    assert result["bundle_manifest_bytes"] is None
    assert result["documents"][0]["role"] == "host_json"
    assert result["validation"]["bundle_loader_pre_model"] is False
    assert {item["role"] for item in result["documents"]} == {"host_json", "source_training_capture"}
    assert result["validation"]["source_training_capture"] == "not_evaluated"
    assert result["validation"]["scientific_scoring"] == "not_evaluated"


@pytest.mark.parametrize("wrong_type,payload", [(True, b'{}'), (False, b'{invalid JSON')])
def test_host_json_wrong_type_or_invalid_json_refuses(tmp_path, wrong_type, payload):
    import nirs4all_core  # noqa: F401 - required, never skip

    from nirs4all.pipeline.dagml.host_container import BLOB_CONTENT_TYPE, JSON_CONTENT_TYPE, HostMember, write_host_container

    path = tmp_path / "invalid.n4a"
    write_host_container(path, [HostMember("pipeline.json", payload, "recipe", BLOB_CONTENT_TYPE if wrong_type else JSON_CONTENT_TYPE)], entrypoint=False)
    with pytest.raises(ValueError):
        inspect_n4a(path, include_host_json=True)


def test_host_json_source_argument_is_sdk_source_only(host_inspection_file):
    import nirs4all_core as core

    path, _ = host_inspection_file
    with core.open_n4a_file(path) as source:
        with pytest.raises(TypeError, match="SDK N4aSource"):
            inspect_n4a(path, source=source, include_host_json=True)
        assert not source.closed


def test_host_json_preserves_owner_nonfinite_values(host_inspection_file):
    from nirs4all.pipeline.dagml.host_container import (
        JSON_CONTENT_TYPE,
        HostMember,
        open_n4a_source,
        update_host_container,
    )

    path, _ = host_inspection_file
    raw = b'{"scores": [NaN, Infinity, -Infinity]}'
    target = path.with_name("nonfinite.n4a")
    with open_n4a_source(path) as source:
        update_host_container(source.require_host(), target, add=[
            HostMember("nonfinite_scores.json", raw, "score", JSON_CONTENT_TYPE),
        ])
    inspected = inspect_n4a(target, include_host_json=True)
    document = next(item for item in inspected["documents"] if item["member"] == "nonfinite_scores.json")
    assert document["bytes"] == raw
    assert document["inventory_sha256"] == hashlib.sha256(raw).hexdigest()
    assert document["byte_length"] == len(raw)
    values = document["json"]["scores"]
    assert math.isnan(values[0])
    assert math.isinf(values[1]) and values[1] > 0
    assert math.isinf(values[2]) and values[2] < 0
