"""Installed-wheel lifecycle proof for the strict native Methods witness."""

from __future__ import annotations

import os

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from nirs4all.pipeline.dagml.native_client import DagMLNativeCoverageError

_REQUIRE_N4M = os.environ.get("NIRS4ALL_REQUIRE_N4M") == "1"

try:
    import dag_ml
    import n4m
except Exception as error:  # pragma: no cover - exact loader failures depend on the host wheel
    message = f"installed Methods witness runtime is unavailable: {error}"
    if _REQUIRE_N4M:
        pytest.fail(message, pytrace=True)
    pytest.skip(message, allow_module_level=True)

if not callable(getattr(dag_ml, "execute_methods_training", None)) or not isinstance(getattr(dag_ml, "TrainingResult", None), type):
    message = "installed dag-ml wheel does not expose the strict Methods TrainingResult surface"
    if _REQUIRE_N4M:
        pytest.fail(message, pytrace=True)
    pytest.skip(message, allow_module_level=True)

pytestmark = pytest.mark.methods


def test_installed_methods_witness_claim_closes_through_the_public_dagml_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public native route yields a live claim and closes the real wheel facade."""

    monkeypatch.delenv("N4M_LIB_PATH", raising=False)
    assert callable(n4m.library_path)

    import nirs4all

    features = np.asarray(
        [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0]],
        dtype=float,
    )
    targets = np.arange(6.0, dtype=float)
    result = nirs4all.run(
        [KFold(n_splits=3), {"model": PLSRegression(n_components=1)}],
        {"X": features, "y": targets, "sample_ids": [f"fit-{index}" for index in range(len(features))]},
        engine="native",
        save_charts=False,
    )

    assert type(result) is nirs4all.NativeMethodsRunResult
    estimator = result.native_estimator
    training_result = estimator.training_result_
    assert type(training_result) is dag_ml.TrainingResult
    assert training_result.is_attached is True
    claim = result.native_execution_claim
    assert claim.execution_entrypoint == "dag_ml.execute_methods_training"
    assert claim.execution_mode == "methods_callback_free"
    assert claim.methods_library_mode == "explicit_absolute"
    assert claim.portable_artifacts_required is True
    assert claim.outcome_fingerprint == training_result.outcome_fingerprint
    assert claim.outcome_fingerprint == training_result.outcome.to_dict()["outcome_fingerprint"]
    assert result.native_execution_is_live is True

    result.close()

    assert training_result.is_attached is False
    assert result.native_execution_is_live is False
    with pytest.raises(DagMLNativeCoverageError, match="no longer attached"):
        _ = result.native_execution_claim
    assert estimator.detach_native_training_result() is False
