"""Real sklearn initializers and bounded CV-weight ownership contracts."""

from __future__ import annotations

import gc
import hashlib
import json
import weakref
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nirs4all.pipeline.dagml.cv_weight_transfer import (
    CVWeightTransferStore,
    TransferIdentity,
    TransferInputContract,
    WarmStartRequest,
    parse_warm_start_request,
    validate_sgd_profile,
    validate_transfer_input,
)

pytestmark = pytest.mark.sklearn


def _sgd(**overrides: Any) -> SGDRegressor:
    params = {"learning_rate": "constant", "eta0": 0.04, "alpha": 0.05,
              "max_iter": 2, "tol": None, "shuffle": False, "random_state": 19}
    params.update(overrides)
    return SGDRegressor(**params)


def _data(dtype: str = "float64") -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray([[1, 0, 2], [0, 1, -1], [-1, 0, 1], [0, -1, 1], [1, 1, 1], [-1, -1, -2]], dtype=dtype)
    y = np.asarray([2, 3, -2, -3, 5, -5], dtype="float64")
    return X, y


def _contract(dtype: str = "float64") -> TransferInputContract:
    return TransferInputContract(("source:nir",), ("nm1000", "nm1002", "nm1004"), ("target:value",), dtype, "float64")


def _identity() -> TransferIdentity:
    return TransferIdentity("run:one", "model:sgd", "controller:python.sklearn", "variant:one")


@pytest.fixture
def fitted() -> tuple[SGDRegressor, np.ndarray, np.ndarray]:
    X, y = _data()
    return _sgd().fit(X[:3], y[:3]), X, y


def test_explicit_request_and_disabled_control() -> None:
    assert parse_warm_start_request({}) is None
    assert parse_warm_start_request({"warm_start": False}) is None
    assert parse_warm_start_request({"warm_start": True, "warm_start_fold": "fold1"}) == WarmStartRequest("fold1")
    assert parse_warm_start_request({"warm_start": True, "warm_start_fold": "fold0"}) == WarmStartRequest("fold0")


@pytest.mark.parametrize("controls", [
    {"warm_start": True}, {"warm_start_fold": "fold1"},
    {"warm_start": False, "warm_start_fold": "fold1"},
    *({"warm_start": True, "warm_start_fold": value} for value in ("best", "last", "", None, 1, "fold-1", "fold01", "fold1\n", "fold_1")),
])
def test_implicit_or_invalid_selector_refused(controls: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="explicit|requires"):
        parse_warm_start_request(controls)


@pytest.mark.parametrize("flag", [1, 0, "true", None, np.bool_(True)])
def test_transfer_flag_is_strict_boolean(flag: Any) -> None:
    with pytest.raises(TypeError, match="boolean"):
        parse_warm_start_request({"warm_start": flag, "warm_start_fold": "fold1"})


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("refit", [{}, {"warm_start": False}])
def test_cold_path_preserves_ordinary_constructor_warm_start(flag: bool, refit: dict[str, Any]) -> None:
    train = {"warm_start": flag}
    assert parse_warm_start_request(refit, train_params=train) is None
    assert train == {"warm_start": flag}
    for model in (SGDRegressor(), RandomForestRegressor()):
        model.set_params(**train)
        assert model.get_params()["warm_start"] is flag
        assert not hasattr(model, "coef_") and not hasattr(model, "estimators_")


def test_explicit_transfer_allows_cold_cv_constructor_override() -> None:
    assert parse_warm_start_request({"warm_start": True, "warm_start_fold": "fold1"}, train_params={"warm_start": False}) == WarmStartRequest("fold1")


@pytest.mark.parametrize("train", [{"warm_start": True}, {"warm_start": 0}, {"warm_start_fold": "fold1"}])
def test_cv_cannot_request_transfer(train: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="refit_params only|start independently"):
        parse_warm_start_request({"warm_start": True, "warm_start_fold": "fold1"}, train_params=train)


@pytest.mark.parametrize("refit", [{}, {"warm_start": False}])
def test_transfer_fold_is_never_a_cv_constructor_control(refit: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="refit_params only"):
        parse_warm_start_request(refit, train_params={"warm_start_fold": "fold1"})


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("learning_rate", ["constant", "invscaling"])
def test_exact_public_weights_detached_and_sklearn_reference_agrees(dtype: str, learning_rate: str) -> None:
    X, y = _data(dtype)
    source = _sgd(learning_rate=learning_rate).fit(X[:3], y[:3])
    expected_coef = source.coef_.copy()
    expected_intercept = source.intercept_.copy()
    request, identity, contract = WarmStartRequest("fold1"), _identity(), _contract(dtype)
    validate_transfer_input(X, y, contract)
    with CVWeightTransferStore(identity.run_id) as store:
        assert store.capture(request, identity, "fold0", source, contract) is False
        assert store.pending_count == store.pending_bytes == 0
        assert store.capture(request, identity, "fold1", source, contract)
        assert store.pending_count == 1 and store.pending_bytes > source.coef_.nbytes
        source.coef_[:] = 999
        source.intercept_[:] = 999
        refit = _sgd(max_iter=3, warm_start=True, learning_rate=learning_rate)
        options, provenance = store.prepare_refit(request, identity, refit, contract)
        assert store.pending_count == store.pending_bytes == 0
        np.testing.assert_array_equal(options["coef_init"], expected_coef)
        np.testing.assert_array_equal(options["intercept_init"], expected_intercept)
        for key, original in (("coef_init", source.coef_), ("intercept_init", source.intercept_)):
            assert options[key].flags.owndata and options[key].flags.c_contiguous and options[key].flags.writeable
            assert not np.shares_memory(options[key], original)
        assert options["coef_init"].dtype == expected_coef.dtype
        assert options["intercept_init"].dtype == expected_intercept.dtype
        assert not hasattr(refit, "coef_") and not hasattr(refit, "t_")
        reference = _sgd(max_iter=3, warm_start=True, learning_rate=learning_rate)
        reference.fit(X, y, coef_init=expected_coef.copy(), intercept_init=expected_intercept.copy())
        refit.fit(X, y, **options)
        np.testing.assert_array_equal(refit.coef_, reference.coef_)
        np.testing.assert_array_equal(refit.intercept_, reference.intercept_)
        assert refit.t_ == 1 + refit.n_iter_ * len(X)
        assert refit.n_iter_ == 3
        cold = _sgd(max_iter=3, warm_start=True, learning_rate=learning_rate).fit(X, y)
        assert not np.array_equal(cold.coef_, refit.coef_)
        assert provenance["schema"] == "nirs4all.cv-weight-transfer.v1"
        assert provenance["source_fold_id"] == "fold1"
        assert provenance["run_id"] == identity.run_id and provenance["variant_label"] == identity.variant_label
        assert provenance["coef_sha256"] == hashlib.sha256(expected_coef.tobytes()).hexdigest()
        assert provenance["intercept_sha256"] == hashlib.sha256(expected_intercept.tobytes()).hexdigest()
        assert provenance["coef_dtype"] == expected_coef.dtype.str
        assert provenance["intercept_dtype"] == expected_intercept.dtype.str
        assert provenance["cv_fit_budget"] == {"max_iter": 2, "tol": None}
        assert provenance["refit_fit_budget"] == {"max_iter": 3, "tol": None}
        assert provenance["optimization_counter_policy"] == "reset_by_fit"
        assert provenance["training_rows_retained"] is False
        json.dumps(provenance, allow_nan=False)
        with pytest.raises(ValueError, match="no captured state"):
            store.prepare_refit(request, identity, _sgd(warm_start=True, learning_rate=learning_rate), contract)


def test_fit_budgets_may_change_without_changing_the_effective_recipe(fitted: tuple[SGDRegressor, np.ndarray, np.ndarray]) -> None:
    source, _, _ = fitted
    request = WarmStartRequest("fold1")
    with CVWeightTransferStore("run:one") as store:
        store.capture(request, _identity(), "fold1", source, _contract())
        options, provenance = store.prepare_refit(request, _identity(), _sgd(max_iter=7, tol=0.001, warm_start=True), _contract())
        np.testing.assert_array_equal(options["coef_init"], source.coef_)
        assert provenance["cv_fit_budget"] == {"max_iter": 2, "tol": None}
        assert provenance["refit_fit_budget"] == {"max_iter": 7, "tol": 0.001}
        assert store.pending_count == store.pending_bytes == 0


def test_source_estimator_and_training_rows_are_not_retained() -> None:
    X, y = _data()
    model = _sgd().fit(X, y)
    model_ref, X_ref, y_ref = weakref.ref(model), weakref.ref(X), weakref.ref(y)
    store = CVWeightTransferStore("run:one")
    store.capture(WarmStartRequest("fold1"), _identity(), "fold1", model, _contract())
    del model, X, y
    gc.collect()
    assert model_ref() is None and X_ref() is None and y_ref() is None
    options, _ = store.prepare_refit(WarmStartRequest("fold1"), _identity(), _sgd(warm_start=True), _contract())
    assert options["coef_init"].shape == (3,) and options["intercept_init"].shape == (1,)
    store.close()


@pytest.mark.parametrize("changed", [
    {"run_id": "run:other"}, {"node_id": "model:other"}, {"controller_id": "controller:other"}, {"variant_label": "variant:other"},
])
def test_run_node_controller_candidate_isolation(fitted: tuple[SGDRegressor, np.ndarray, np.ndarray], changed: dict[str, str]) -> None:
    source, _, _ = fitted
    store = CVWeightTransferStore("run:one")
    request = WarmStartRequest("fold1")
    store.capture(request, _identity(), "fold1", source, _contract())
    with pytest.raises(ValueError, match="another run|no captured state"):
        store.prepare_refit(request, replace(_identity(), **changed), _sgd(warm_start=True), _contract())
    assert store.pending_count == 1
    store.close()
    assert store.pending_count == store.pending_bytes == 0


def test_wrong_fold_and_duplicate_capture_refused(fitted: tuple[SGDRegressor, np.ndarray, np.ndarray]) -> None:
    source, _, _ = fitted
    with CVWeightTransferStore("run:one") as store:
        request = WarmStartRequest("fold1")
        store.capture(request, _identity(), "fold1", source, _contract())
        with pytest.raises(ValueError, match="already captured"):
            store.capture(request, _identity(), "fold1", source, _contract())
        with pytest.raises(ValueError, match="no captured state"):
            store.prepare_refit(WarmStartRequest("fold2"), _identity(), _sgd(warm_start=True), _contract())
        assert store.pending_count == 1


@pytest.mark.parametrize("changed", [
    {"alpha": 0.1}, {"eta0": 0.02}, {"learning_rate": "invscaling"}, {"power_t": 0.5},
    {"fit_intercept": False}, {"shuffle": True}, {"random_state": 20}, {"n_iter_no_change": 6},
])
def test_structural_recipe_change_refused(fitted: tuple[SGDRegressor, np.ndarray, np.ndarray], changed: dict[str, Any]) -> None:
    source, _, _ = fitted
    with CVWeightTransferStore("run:one") as store:
        request = WarmStartRequest("fold1")
        store.capture(request, _identity(), "fold1", source, _contract())
        refit = _sgd(warm_start=True, **changed)
        with pytest.raises(ValueError, match="recipe differs"):
            store.prepare_refit(request, _identity(), refit, _contract())
        assert not hasattr(refit, "coef_") and store.pending_count == 1


@pytest.mark.parametrize("changed", [
    {"source_names": ("source:other",)}, {"feature_names": ("nm1002", "nm1000", "nm1004")},
    {"target_names": ("target:other",)}, {"x_dtype": "float32"}, {"y_dtype": "float32"},
])
def test_representation_change_refused(fitted: tuple[SGDRegressor, np.ndarray, np.ndarray], changed: dict[str, Any]) -> None:
    source, _, _ = fitted
    with CVWeightTransferStore("run:one") as store:
        request = WarmStartRequest("fold1")
        store.capture(request, _identity(), "fold1", source, _contract())
        with pytest.raises(ValueError, match="representation differs"):
            store.prepare_refit(request, _identity(), _sgd(warm_start=True), replace(_contract(), **changed))


@pytest.mark.parametrize("overrides", [
    {"average": True}, {"average": 10}, {"early_stopping": True}, {"learning_rate": "adaptive"},
    {"learning_rate": "optimal"}, {"learning_rate": "pa1"}, {"loss": "huber"}, {"penalty": "elasticnet"},
])
def test_unsupported_sgd_capabilities_refused_before_fit(overrides: dict[str, Any]) -> None:
    with pytest.raises(NotImplementedError, match="requires|supports"):
        validate_sgd_profile(_sgd(**overrides))


@pytest.mark.parametrize("overrides", [
    {"random_state": None}, {"random_state": np.random.RandomState(19)}, {"random_state": -1},
    {"random_state": 2**32}, {"random_state": True}, {"eta0": 0}, {"alpha": np.nan},
    {"power_t": -1}, {"tol": -1}, {"max_iter": 0}, {"max_iter": 1.5}, {"shuffle": "false"},
])
def test_invalid_or_nondeterministic_parameters_refused_before_fit(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="parameter|seed range"):
        validate_sgd_profile(_sgd(**overrides))


def test_plain_estimator_profile_excludes_pipeline_ridge_and_subclass() -> None:
    class AlternateSGD(SGDRegressor):
        pass

    for model in (Ridge(), make_pipeline(StandardScaler(), _sgd()), AlternateSGD(random_state=19)):
        with pytest.raises(NotImplementedError, match="plain sklearn SGDRegressor"):
            validate_sgd_profile(model)


def test_cv_is_cold_refit_is_fresh_and_flag_is_not_the_transfer(fitted: tuple[SGDRegressor, np.ndarray, np.ndarray]) -> None:
    source, X, y = fitted
    with pytest.raises(ValueError, match="start independently"):
        validate_sgd_profile(_sgd(warm_start=True), phase="FIT_CV")
    with pytest.raises(ValueError, match="warm_start=True"):
        validate_sgd_profile(_sgd(), phase="REFIT")
    with CVWeightTransferStore("run:one") as store:
        request = WarmStartRequest("fold1")
        with pytest.raises(ValueError, match="successfully fitted"):
            store.capture(request, _identity(), "fold1", _sgd(), _contract())
        store.capture(request, _identity(), "fold1", source, _contract())
        existing = _sgd(warm_start=True).fit(X, y)
        with pytest.raises(ValueError, match="fresh REFIT"):
            store.prepare_refit(request, _identity(), existing, _contract())


def test_input_guard_rejects_partial_wrong_dtype_shape_and_width() -> None:
    X, y = _data()
    validate_transfer_input(X, y, _contract())
    validate_transfer_input(X, y.reshape(-1, 1), _contract())
    bad_pairs = [(X[:, :2], y), (X.astype("float32"), y), (X, y.astype("float32")),
                 (X, y[:2]), (X, np.column_stack([y, y])), (X.tolist(), y),
                 (X, np.ma.array(y, mask=[True, False, False, False, False, False])),
                 (X.copy(), np.asarray([np.nan, 3, -2, -3, 5, -5])),
                 (np.full_like(X, np.inf), y)]
    for bad_X, bad_y in bad_pairs:
        with pytest.raises(ValueError, match="requires|dtype|refuses"):
            validate_transfer_input(bad_X, bad_y, _contract())


@pytest.mark.parametrize("changed", [
    {"source_names": ("one", "two")}, {"target_names": ("one", "two")},
    {"feature_names": ("same", "same")}, {"feature_names": ("with\x00nul",)},
    {"x_dtype": "int64"}, {"y_dtype": "object"},
])
def test_input_metadata_guard(changed: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="one numeric|distinct|NUL|float32"):
        replace(_contract(), **changed)


def test_corrupt_fitted_weights_refused(fitted: tuple[SGDRegressor, np.ndarray, np.ndarray]) -> None:
    source, _, _ = fitted
    with CVWeightTransferStore("run:one") as store:
        original = source.coef_.copy()
        source.coef_[0] = np.nan
        with pytest.raises(ValueError, match="nonfinite fitted weights"):
            store.capture(WarmStartRequest("fold1"), _identity(), "fold1", source, _contract())
        source.coef_ = original[:2]
        with pytest.raises(ValueError, match="dimensions"):
            store.capture(WarmStartRequest("fold1"), _identity(), "fold1", source, _contract())
        assert store.pending_count == store.pending_bytes == 0


def test_limits_do_not_evict_or_partially_capture(fitted: tuple[SGDRegressor, np.ndarray, np.ndarray]) -> None:
    source, _, _ = fitted
    request = WarmStartRequest("fold1")
    with CVWeightTransferStore("run:one", max_entries=1) as store:
        store.capture(request, _identity(), "fold1", source, _contract())
        retained_bytes = store.pending_bytes
        with pytest.raises(ValueError, match="entry limit"):
            store.capture(request, replace(_identity(), node_id="model:other"), "fold1", source, _contract())
        assert store.pending_count == 1 and store.pending_bytes == retained_bytes
    with CVWeightTransferStore("run:one", max_bytes=1) as store:
        with pytest.raises(ValueError, match="byte limit"):
            store.capture(request, _identity(), "fold1", source, _contract())
        assert store.pending_count == store.pending_bytes == 0
        assert source.coef_.flags.writeable and source.intercept_.flags.writeable


def test_cleanup_on_cancellation_error_and_closed_store(fitted: tuple[SGDRegressor, np.ndarray, np.ndarray]) -> None:
    source, _, _ = fitted
    store = CVWeightTransferStore("run:one")
    with pytest.raises(RuntimeError, match="cancelled"):
        with store:
            store.capture(WarmStartRequest("fold1"), _identity(), "fold1", source, _contract())
            raise RuntimeError("cancelled")
    assert store.pending_count == store.pending_bytes == 0
    store.close()
    store.clear()
    with pytest.raises(RuntimeError, match="closed"):
        store.capture(WarmStartRequest("fold1"), _identity(), "fold1", source, _contract())
    with pytest.raises(AttributeError):
        store.run_id = "run:foreign"  # type: ignore[misc]


def test_consumption_releases_state_before_fit_error(fitted: tuple[SGDRegressor, np.ndarray, np.ndarray]) -> None:
    source, X, y = fitted
    with CVWeightTransferStore("run:one") as store:
        request = WarmStartRequest("fold1")
        store.capture(request, _identity(), "fold1", source, _contract())
        refit = _sgd(warm_start=True)
        options, _ = store.prepare_refit(request, _identity(), refit, _contract())
        assert store.pending_count == store.pending_bytes == 0
        with pytest.raises(ValueError):
            refit.fit(X[:, :2], y, **options)
        assert store.pending_count == store.pending_bytes == 0
