"""The native Methods lane rule (``nirs4all.pipeline.dagml.methods_lane``).

``native_methods_refusal`` decides, before execution, which DAG campaigns run
through DAG-ML's callback-free Methods estimator controllers. Every clause of
the documented rule is exercised here; lane parity lives in
``tests/integration/api/test_dagml_native_methods_lane.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

roles = pytest.importorskip("n4m.roles")
native = pytest.importorskip("dag_ml._dag_ml")
if not hasattr(native, "run_cv_refit_methods_in_process"):
    pytest.skip("the installed dag-ml lacks the native Methods CV/REFIT lane", allow_module_level=True)

from nirs4all.pipeline.dagml.dataset import _materialize_dataset  # noqa: E402
from nirs4all.pipeline.dagml.identity import mint_identity  # noqa: E402
from nirs4all.pipeline.dagml.methods_lane import (  # noqa: E402
    HOST_CALLBACK_LANE,
    MIXED_LANES,
    NATIVE_METHODS_ENV_VAR,
    NATIVE_METHODS_LANE,
    lane_record,
    merge_lane_records,
    native_methods_refusal,
    native_params,
)

pytestmark = pytest.mark.methods


@pytest.fixture(scope="module")
def regression():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 12))
    return _materialize_dataset((X, X[:, 0] - X[:, 1]))


@pytest.fixture(scope="module")
def classification():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 12))
    return _materialize_dataset((X, (X[:, 0] > 0).astype(int)))


def refusal(steps, spectro, **kwargs):
    return native_methods_refusal(steps, spectro, identity=kwargs.pop("identity", None) or mint_identity(spectro), **kwargs)


def test_role_chain_is_native(regression, classification):
    assert refusal([roles.SNV(), roles.VarianceFilter(top_k=5), {"model": roles.CPPLS(n_components=2)}], regression) is None
    assert refusal([{"model": roles.CPPLS(n_components=2), "name": "cp"}], regression) is None
    assert refusal([roles.SNV(), {"model": roles.PLSLogistic(n_components=2)}], classification) is None
    assert refusal([{"model": roles.CPPLS(), "n_components": {"_range_": [1, 5, 2]}}], regression) is None
    assert refusal([{"model": roles.Ridge(), "alpha": {"_log_range_": [-3, 0, 4]}}], regression) is None


@pytest.mark.parametrize(
    ("steps", "reason"),
    [
        ([StandardScaler(), {"model": roles.CPPLS()}], "step 0 is not an n4m transformer"),
        ([{"y_processing": MinMaxScaler()}, {"model": roles.CPPLS()}], "step 0 {'y_processing'} is not an n4m transformer"),
        ([{"model": PLSRegression()}], "not an n4m regressor or classifier"),
        ([roles.SNV()], "last step is not a model step"),
        ([{"model": roles.PLSLogistic()}], "does not match the dataset task type"),
        ([{"model": roles.LWPLS()}], "retains training rows"),
        ([roles.OnPLS(), {"model": roles.CPPLS()}], "fit input a DAG node cannot supply"),
        ([{"model": roles.CPPLS(n_components=2.5)}], "no exact native int value"),
        ([{"model": roles.CPPLS(), "n_components": {"_log_range_": [0, 1, 3]}}], "does not produce exact native int"),
        ([{"model": roles.CPPLS(), "n_components": {"_range_": [1, 4, 0.5]}}], "does not produce exact native int"),
        ([{"model": roles.CPPLS(), "n_components": {"_grid_": [1, 2]}}], "is not a native parameter sweep"),
    ],
)
def test_ineligible_steps_keep_the_callback_lane(regression, steps, reason):
    assert reason in refusal(steps, regression)


def test_classification_task_needs_a_classifier(classification):
    assert "task type" in refusal([{"model": roles.CPPLS()}], classification)


def test_dataset_clauses(regression):
    steps = [roles.SNV(), {"model": roles.CPPLS()}]
    augmented = SimpleNamespace(identities=[SimpleNamespace(augmented=True)])
    assert refusal(steps, regression, identity=augmented) == "the dataset holds augmented rows"
    assert refusal(steps, regression, excluded={3}) == "excluded samples are kept in the OOF universe"

    from nirs4all.data.dataset import SpectroDataset

    rng = np.random.default_rng(5)
    held_out = SpectroDataset("held_out")
    held_out.add_samples(rng.normal(size=(20, 8)), {"partition": "train"})
    held_out.add_samples(rng.normal(size=(5, 8)), {"partition": "test"})
    held_out.add_targets(rng.normal(size=25))
    assert refusal(steps, held_out) == "the dataset has a held-out test partition"


def test_runtime_clauses(regression, monkeypatch):
    steps = [roles.SNV(), {"model": roles.CPPLS()}]
    monkeypatch.setenv(NATIVE_METHODS_ENV_VAR, "off")
    assert NATIVE_METHODS_ENV_VAR in refusal(steps, regression)
    monkeypatch.delenv(NATIVE_METHODS_ENV_VAR)
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0")
    assert "subprocess" in refusal(steps, regression)
    monkeypatch.delenv("N4A_DAGML_INPROCESS")
    monkeypatch.delattr(native, "run_cv_refit_methods_in_process")
    assert "lacks the native Methods CV/REFIT lane" in refusal(steps, regression)


def test_native_params_follow_the_binding_types():
    assert native_params(roles.CPPLS(n_components=3.0)) == {"n_components": 3, "gamma": 0.5}
    assert native_params(roles.YOutlierFilter(threshold=2)) == {"method": "iqr", "threshold": 2.0, "lower_percentile": 1.0, "upper_percentile": 99.0}
    # None keeps the native default, exactly as the n4m Python binding does.
    assert "seed" not in native_params(roles.BaggingPLS(seed=None))
    with pytest.raises(ValueError, match="no exact native double"):
        native_params(roles.Ridge(alpha=float("nan")))


def test_lane_records_merge():
    native_record = lane_record(NATIVE_METHODS_LANE)
    callback = lane_record(HOST_CALLBACK_LANE, "why")
    assert merge_lane_records([native_record, native_record]) == native_record
    assert merge_lane_records([callback, callback]) == callback
    assert merge_lane_records([native_record, callback]) == lane_record(MIXED_LANES)
