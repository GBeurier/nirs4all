"""Native-only public result projection tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nirs4all.api import native_result
from nirs4all.api.native_result import NativeMethodsRunResult
from nirs4all.api.native_witness import NativeMethodsExecutionClaim
from nirs4all.pipeline.dagml.native_client import DagMLNativeCoverageError


class _TestWitness:
    """Minimal test double for result projection tests, never production evidence."""

    def __init__(self, estimator: object) -> None:
        self._estimator = estimator
        self._live = True
        self._claim = NativeMethodsExecutionClaim(
            schema_version=1,
            execution_entrypoint="dag_ml.execute_methods_training",
            execution_mode="methods_callback_free",
            outcome_fingerprint="a" * 64,
            methods_library_mode="explicit_absolute",
            portable_artifacts_required=True,
        )

    @classmethod
    def from_estimator(cls, estimator: object) -> _TestWitness:
        return cls(estimator)

    def _claim_for_estimator(self, estimator: object) -> NativeMethodsExecutionClaim:
        if not self._live:
            raise DagMLNativeCoverageError("the live Methods witness is no longer attached")
        if estimator is not self._estimator:
            raise DagMLNativeCoverageError("the live Methods witness does not own this estimator")
        return self._claim

    def _is_live_for_estimator(self, estimator: object) -> bool:
        return self._live and estimator is self._estimator

    def detach(self) -> bool:
        if not self._live:
            return False
        self._live = False
        return True


@pytest.fixture(autouse=True)
def _patch_witness_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native_result, "_LiveMethodsWitness", _TestWitness)


class _Estimator:
    training_outcome_ = {
        "outcome_fingerprint": "a" * 64,
        "score_set": {
            "schema_version": 2,
            "selection_metric": "rmse",
            "reports": [
                {
                    "producer_node": "model:methods",
                    "producer_port": "oof",
                    "partition": "validation",
                    "fold_id": "fold0",
                    "level": "sample",
                    "metrics": {"rmse": 0.5},
                    "row_count": 2,
                    "target_names": ["y"],
                    "target_width": 1,
                    "variant_id": "variant:base",
                },
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
                },
            ],
        },
    }

    def __init__(self) -> None:
        self.calls: list[tuple[object, str]] = []

    def export_native_archive(self, path, *, archive_id):  # noqa: ANN001
        self.calls.append((path, archive_id))
        return {"archive_id": archive_id, "archive_sha256": "b" * 64}


def test_native_methods_result_projects_native_scores_and_exports_archive_v2(tmp_path) -> None:
    estimator = _Estimator()
    result = NativeMethodsRunResult.from_estimator(
        estimator,  # type: ignore[arg-type]
        dataset_name="native",
        model_name="PLSRegression",
    )

    assert result.per_dataset == {"native": {"engine": "native"}}
    assert result.cv_best_score == pytest.approx(0.5)
    assert result.native_estimator is estimator

    path = result.export(tmp_path / "model.n4a")

    assert path == tmp_path / "model.n4a"
    assert estimator.calls == [(path, "archive:" + "a" * 64)]
    assert result.native_archive_reference == {
        "archive_id": "archive:" + "a" * 64,
        "archive_sha256": "b" * 64,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"format": "joblib"}, "format='n4a'"),
        ({"source": {}}, "source=/chain_id="),
        ({"chain_id": "legacy"}, "source=/chain_id="),
        ({"compatibility": "legacy-refit"}, "never accepts"),
    ],
)
def test_native_methods_result_refuses_legacy_export_routes(tmp_path, kwargs, message) -> None:  # noqa: ANN001
    estimator = _Estimator()
    result = NativeMethodsRunResult.from_estimator(
        estimator,  # type: ignore[arg-type]
        dataset_name="native",
        model_name="PLSRegression",
    )

    with pytest.raises((ValueError, NotImplementedError), match=message):
        result.export(tmp_path / "model.n4a", **kwargs)


def test_native_methods_result_refuses_missing_outcome_export_identity() -> None:
    estimator = _Estimator()
    estimator.training_outcome_ = {"score_set": _Estimator.training_outcome_["score_set"]}

    with pytest.raises(Exception, match="outcome fingerprint"):
        NativeMethodsRunResult.from_estimator(
            estimator,  # type: ignore[arg-type]
            dataset_name="native",
            model_name="PLSRegression",
        )
