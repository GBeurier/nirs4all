"""Live execution witness boundaries for the strict Methods lane."""

from __future__ import annotations

import copy
import importlib
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from nirs4all.api import native_session, native_training, native_witness
from nirs4all.api.native_result import NativeMethodsRunResult
from nirs4all.api.native_session import NativeMethodsSession
from nirs4all.api.native_witness import _LiveMethodsWitness
from nirs4all.pipeline.dagml.estimator import DagMLPipelineEstimator
from nirs4all.pipeline.dagml.native_client import DagMLNativeCoverageError


class _Outcome:
    def __init__(self, document: dict[str, object]) -> None:
        self._document = document

    def to_dict(self) -> dict[str, object]:
        return dict(self._document)


class _TrainingResult:
    def __init__(self, document: dict[str, object]) -> None:
        self._document = document
        self.is_attached = True
        self.detach_calls = 0
        self.raise_once_on_detach = False

    @property
    def outcome_fingerprint(self) -> str:
        return self._document["outcome_fingerprint"]  # type: ignore[return-value]

    @property
    def outcome(self) -> _Outcome:
        return _Outcome(self._document)

    def detach(self) -> bool:
        self.detach_calls += 1
        if self.raise_once_on_detach:
            self.raise_once_on_detach = False
            raise RuntimeError("native detach failed once")
        if not self.is_attached:
            return False
        self.is_attached = False
        return True


def _score_set() -> dict[str, object]:
    return {
        "schema_version": 2,
        "selection_metric": "rmse",
        "reports": [
            {
                "producer_node": "model:methods",
                "producer_port": "oof",
                "partition": "validation",
                "fold_id": "avg",
                "level": "sample",
                "metrics": {"rmse": 0.5},
                "row_count": 2,
                "target_names": ["y"],
                "target_width": 1,
                "variant_id": "variant:base",
            }
        ],
    }


def _install_dagml_facade(monkeypatch: pytest.MonkeyPatch) -> None:
    facade = ModuleType("dag_ml")
    facade.TrainingResult = _TrainingResult
    monkeypatch.setitem(sys.modules, "dag_ml", facade)


def _estimator(tmp_path: Path, *, callback: object | None = None) -> tuple[SimpleNamespace, _TrainingResult]:
    library = tmp_path / "libn4m.so"
    library.write_bytes(b"methods")
    fingerprint = "a" * 64
    outcome: dict[str, object] = {
        "outcome_fingerprint": fingerprint,
        "score_set": _score_set(),
    }
    result = _TrainingResult(outcome)
    estimator = SimpleNamespace(
        dagml_module="dag_ml",
        native_client=None,
        native_training_execution_=SimpleNamespace(
            methods_inputs={"data:train": {"sample_ids": ["s1", "s2"]}},
            methods_library_path=str(library.resolve()),
            op_callback=callback,
        ),
        training_result_=result,
        training_outcome_=outcome,
        predictor_package_={"package_id": "package:native"},
    )

    def export_native_archive(path: Path, *, archive_id: str) -> dict[str, str]:
        path.write_bytes(b"native-archive")
        return {"archive_id": archive_id, "archive_sha256": "b" * 64}

    estimator.export_native_archive = export_native_archive
    return estimator, result


def test_live_witness_is_exact_redacted_nonserializable_and_invalidates_after_detach(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_dagml_facade(monkeypatch)
    estimator, training_result = _estimator(tmp_path)

    witness = _LiveMethodsWitness.from_estimator(estimator)  # type: ignore[arg-type]
    claim = witness.claim

    assert claim.to_dict() == {
        "schema_version": 1,
        "execution_entrypoint": "dag_ml.execute_methods_training",
        "execution_mode": "methods_callback_free",
        "outcome_fingerprint": "a" * 64,
        "methods_library_mode": "explicit_absolute",
        "portable_artifacts_required": True,
    }
    assert "libn4m" not in repr(claim)
    assert witness.is_live
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(witness)
    with pytest.raises(TypeError, match="cannot be deep-copied"):
        copy.deepcopy(witness)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(witness)

    assert witness.detach() is True
    assert witness.detach() is False
    assert training_result.detach_calls == 1
    assert not witness.is_live
    with pytest.raises(DagMLNativeCoverageError, match="no longer attached"):
        _ = witness.claim


def test_live_witness_constructor_is_internal_and_cannot_forge_a_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_dagml_facade(monkeypatch)
    estimator, training_result = _estimator(tmp_path)
    claim = _LiveMethodsWitness.from_estimator(estimator).claim  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="internal factory"):
        _LiveMethodsWitness(
            estimator,
            training_result,
            claim,
            _factory_capability=object(),
        )


def test_native_result_factory_does_not_accept_an_injected_witness_and_detects_facade_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_dagml_facade(monkeypatch)
    estimator_a, training_result_a = _estimator(tmp_path)
    witness = _LiveMethodsWitness.from_estimator(estimator_a)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="unexpected keyword argument 'live_witness'"):
        NativeMethodsRunResult.from_estimator(
            estimator_a,  # type: ignore[arg-type]
            dataset_name="native",
            model_name="MethodsPLS",
            live_witness=witness,
        )

    estimator_a.training_result_ = _TrainingResult(dict(training_result_a._document))
    with pytest.raises(DagMLNativeCoverageError, match="no longer owns the estimator TrainingResult"):
        _ = witness.claim
    assert not witness.is_live


def test_live_witness_detach_failure_keeps_ownership_for_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_dagml_facade(monkeypatch)
    estimator, training_result = _estimator(tmp_path)
    witness = _LiveMethodsWitness.from_estimator(estimator)  # type: ignore[arg-type]
    training_result.raise_once_on_detach = True

    with pytest.raises(RuntimeError, match="failed once"):
        witness.detach()
    assert witness.is_live
    assert training_result.detach_calls == 1

    assert witness.detach() is True
    assert training_result.detach_calls == 2
    assert not witness.is_live


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda estimator: setattr(estimator.native_training_execution_, "op_callback", object()), "operator callback"),
        (lambda estimator: setattr(estimator.native_training_execution_, "methods_inputs", {}), "non-empty native Methods inputs"),
        (lambda estimator: setattr(estimator.native_training_execution_, "methods_library_path", "relative/libn4m.so"), "must be absolute"),
        (lambda estimator: setattr(estimator, "native_client", object()), "default Dag-ML client"),
        (lambda estimator: setattr(estimator, "training_outcome_", {"outcome_fingerprint": "b" * 64}), "finalized training outcome"),
    ],
)
def test_live_witness_refuses_any_non_strict_or_mismatched_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutate,
    message: str,
) -> None:  # noqa: ANN001
    _install_dagml_facade(monkeypatch)
    estimator, _training_result = _estimator(tmp_path)
    mutate(estimator)

    with pytest.raises(DagMLNativeCoverageError, match=message):
        _LiveMethodsWitness.from_estimator(estimator)  # type: ignore[arg-type]


def test_native_result_and_session_detach_live_facade_but_preserve_native_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_dagml_facade(monkeypatch)
    estimator, training_result = _estimator(tmp_path)
    result = NativeMethodsRunResult.from_estimator(
        estimator,  # type: ignore[arg-type]
        dataset_name="native",
        model_name="MethodsPLS",
    )

    assert result.native_execution_claim.outcome_fingerprint == "a" * 64
    assert result.native_execution_is_live
    no_model_path = tmp_path / "forbidden.joblib"
    with pytest.raises(NotImplementedError, match="export_model is unavailable"):
        result.export_model(no_model_path)
    assert not no_model_path.exists()

    session = NativeMethodsSession([{"split": "stub"}, {"model": "stub"}])
    session._result = result  # noqa: SLF001 - exercise close ownership directly
    session.close()

    assert training_result.detach_calls == 1
    assert not result.native_execution_is_live
    with pytest.raises(DagMLNativeCoverageError, match="no longer attached"):
        _ = result.native_execution_claim
    exported = result.export(tmp_path / "portable.n4a")
    assert exported.read_bytes() == b"native-archive"


def test_native_session_close_failure_retains_result_for_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_dagml_facade(monkeypatch)
    estimator, training_result = _estimator(tmp_path)
    result = NativeMethodsRunResult.from_estimator(
        estimator,  # type: ignore[arg-type]
        dataset_name="native",
        model_name="MethodsPLS",
    )
    session = NativeMethodsSession([{"split": "stub"}, {"model": "stub"}])
    session._result = result  # noqa: SLF001 - exercise close ownership directly
    training_result.raise_once_on_detach = True

    with pytest.raises(RuntimeError, match="failed once"):
        session.close()
    assert not session.closed
    assert session.result is result
    assert result.native_execution_is_live

    session.close()
    assert session.closed
    assert training_result.detach_calls == 2


@pytest.mark.parametrize("operation", ["run", "retrain"])
def test_native_session_replacement_failure_keeps_previous_result_owned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
) -> None:
    _install_dagml_facade(monkeypatch)
    estimator, training_result = _estimator(tmp_path)
    previous = NativeMethodsRunResult.from_estimator(
        estimator,  # type: ignore[arg-type]
        dataset_name="native",
        model_name="MethodsPLS",
    )
    session = NativeMethodsSession([{"split": "stub"}, {"model": "stub"}])
    session._result = previous  # noqa: SLF001 - exercise replacement ownership directly
    training_result.raise_once_on_detach = True

    def reached(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("replacement work ran before the prior live result closed")

    if operation == "run":
        monkeypatch.setattr(native_session, "run_native_methods", reached)
        with pytest.raises(RuntimeError, match="failed once"):
            session.run({})
    else:
        monkeypatch.setattr(importlib.import_module("nirs4all.api.retrain"), "retrain", reached)
        with pytest.raises(RuntimeError, match="failed once"):
            session.retrain({"X": [], "y": [], "sample_ids": []})

    assert session.result is previous
    assert previous.native_execution_is_live


def test_native_training_projection_failure_detaches_the_fitted_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FittedEstimator:
        def __init__(self) -> None:
            self.detach_calls = 0

        def detach_native_training_result(self) -> bool:
            self.detach_calls += 1
            return True

    class FailingWitness:
        @classmethod
        def from_estimator(cls, _estimator: object) -> object:
            raise RuntimeError("witness projection failed")

    estimator = FittedEstimator()
    monkeypatch.setattr(native_training, "fit_native_pipeline", lambda *_args, **_kwargs: estimator)
    monkeypatch.setattr("nirs4all.api.native_result._LiveMethodsWitness", FailingWitness)

    with pytest.raises(RuntimeError, match="witness projection failed"):
        native_training.run_native_methods(
            [{"split": "stub"}, {"model": "stub"}],
            {"X": [[1.0], [2.0]], "y": [1.0, 2.0], "sample_ids": ["s1", "s2"]},
            save_charts=False,
        )
    assert estimator.detach_calls == 1


def test_direct_fit_estimator_has_idempotent_native_detach() -> None:
    document = {"outcome_fingerprint": "a" * 64}
    training_result = _TrainingResult(document)
    estimator = object.__new__(DagMLPipelineEstimator)
    estimator.training_result_ = training_result
    estimator.training_outcome_ = document
    estimator.predictor_package_ = {"package_id": "package:native"}

    assert estimator.detach_native_training_result() is True
    assert estimator.detach_native_training_result() is False
    assert training_result.detach_calls == 1
    assert estimator.training_outcome_ == document
    assert estimator.predictor_package_ == {"package_id": "package:native"}
