"""Contract tests for native n4m AOM-PLS and shared operator helpers."""

import numpy as np
import pytest
from n4m.model_selection.aom_search import AOMPLSRegressor as NativeAOMPLSRegressor
from sklearn.base import clone, is_regressor

from nirs4all.operators.models import AOMPLSRegressor as PublicAOMPLSRegressor
from nirs4all.operators.models.sklearn.aom_pls import (
    AOMPLSRegressor,
    ComposedOperator,
    DetrendProjectionOperator,
    IdentityOperator,
    LinearSpectralOperator,
    SavitzkyGolayOperator,
    default_operator_bank,
)


@pytest.fixture
def regression_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic spectral-like data and a four-fold partition."""

    rng = np.random.default_rng(42)
    X = rng.normal(size=(48, 24))
    y = 1.8 * X[:, 0] - 0.7 * X[:, 3] + 0.4 * X[:, 7]
    y += rng.normal(scale=0.02, size=X.shape[0])
    fold_ids = np.arange(X.shape[0], dtype=np.int32) % 4
    return X, y, fold_ids


class TestOperatorAdjoint:
    """Keep the linear-operator contracts used by AOM classifiers and Ridge."""

    def _check_adjoint(self, op: LinearSpectralOperator, p: int = 80) -> None:
        op.fit(np.zeros((1, p)))
        rng = np.random.default_rng(4)
        for _ in range(3):
            x = rng.normal(size=(1, p))
            y = rng.normal(size=p)
            np.testing.assert_allclose(
                np.dot(op.transform(x).ravel(), y),
                np.dot(x.ravel(), op.adjoint_vec(y)),
                rtol=0.0,
                atol=1e-8,
            )

    @pytest.mark.parametrize(
        "operator",
        [
            IdentityOperator(),
            SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=0),
            SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1),
            DetrendProjectionOperator(degree=1),
            DetrendProjectionOperator(degree=2),
            ComposedOperator(
                [
                    DetrendProjectionOperator(degree=1),
                    SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1),
                ]
            ),
        ],
    )
    def test_adjoint_identity(self, operator: LinearSpectralOperator) -> None:
        self._check_adjoint(operator)

    def test_default_bank_adjoint_identity(self) -> None:
        for operator in default_operator_bank(p=80):
            self._check_adjoint(operator)


class TestOperatorProperties:
    def test_identity_is_identity(self) -> None:
        operator = IdentityOperator().fit(np.zeros((1, 40)))
        X = np.arange(120, dtype=float).reshape(3, 40)
        np.testing.assert_array_equal(operator.transform(X), X)

    def test_detrend_removes_linear_trend(self) -> None:
        operator = DetrendProjectionOperator(degree=1).fit(np.zeros((1, 80)))
        result = operator.transform(np.linspace(0, 10, 80).reshape(1, -1))
        assert np.max(np.abs(result)) < 1e-10

    def test_smoothing_reduces_noise(self) -> None:
        operator = SavitzkyGolayOperator(window_length=11, polyorder=2).fit(np.zeros((1, 80)))
        rng = np.random.default_rng(9)
        signal = np.sin(np.linspace(0, 4 * np.pi, 80))
        noisy = signal + 0.3 * rng.normal(size=80)
        smoothed = operator.transform(noisy.reshape(1, -1)).ravel()
        assert np.std(smoothed - signal) < np.std(noisy - signal)

    def test_default_bank_contains_identity(self) -> None:
        bank = default_operator_bank(p=80)
        assert any(isinstance(operator, IdentityOperator) for operator in bank)
        assert 8 <= len(bank) <= 120


class TestNativeAOMPLSRegressor:
    def test_public_exports_are_the_n4m_estimator(self) -> None:
        assert AOMPLSRegressor is NativeAOMPLSRegressor
        assert PublicAOMPLSRegressor is NativeAOMPLSRegressor

    def test_sklearn_contract_and_clone(self) -> None:
        model = AOMPLSRegressor(
            max_components=4,
            operators=["identity", ("savgol_smooth", [5, 2])],
            cv=4,
            scale_x=False,
        )
        assert is_regressor(model)
        cloned = clone(model)
        assert cloned is not model
        assert cloned.get_params() == model.get_params()

    def test_native_coefficients_replay_training_predictions(
        self,
        regression_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X, y, fold_ids = regression_data
        model = AOMPLSRegressor(
            max_components=4,
            operators=["identity", ("savgol_smooth", [5, 2])],
            cv=4,
            fold_ids=fold_ids,
            scale_x=False,
        )

        assert model.fit(X, y) is model
        predictions = model.predict(X)

        np.testing.assert_allclose(
            predictions,
            np.asarray(model.result_["predictions"]).ravel(),
            rtol=1e-10,
            atol=1e-10,
        )
        assert model.result_["input_coefficients"].shape == (X.shape[1], 1)
        assert model.n_features_in_ == X.shape[1]
        assert 1 <= model.selected_n_components_ <= 4
        diagnostics = model.get_diagnostics()
        assert diagnostics["n_operators"] == 2
        assert diagnostics["selected_operator_index"] in {0, 1}

    def test_multioutput_prediction_shape(
        self,
        regression_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X, y, fold_ids = regression_data
        Y = np.column_stack((y, -0.5 * y + X[:, 2]))
        model = AOMPLSRegressor(
            max_components=3,
            operators=["identity"],
            cv=4,
            fold_ids=fold_ids,
        ).fit(X, Y)

        assert model.predict(X).shape == Y.shape
        assert model.result_["input_coefficients"].shape == (X.shape[1], 2)

    def test_deterministic_fit(
        self,
        regression_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X, y, fold_ids = regression_data
        kwargs = {
            "max_components": 3,
            "operators": ["identity", ("detrend_poly", [1])],
            "cv": 4,
            "fold_ids": fold_ids,
        }
        first = AOMPLSRegressor(**kwargs).fit(X, y)
        second = AOMPLSRegressor(**kwargs).fit(X, y)
        np.testing.assert_array_equal(first.predict(X), second.predict(X))
        np.testing.assert_array_equal(first.coef_, second.coef_)

    def test_invalid_fold_contract_fails_closed(
        self,
        regression_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X, y, _ = regression_data
        with pytest.raises(ValueError, match="fold_ids length"):
            AOMPLSRegressor(cv=4, fold_ids=np.arange(8)).fit(X, y)

    def test_predict_rejects_wrong_feature_count(
        self,
        regression_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X, y, fold_ids = regression_data
        model = AOMPLSRegressor(cv=4, fold_ids=fold_ids).fit(X, y)
        with pytest.raises(ValueError, match="different number of features"):
            model.predict(X[:, :-1])
