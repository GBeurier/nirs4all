"""Contract tests for native n4m POP-PLS and the Python classifier surface."""

import numpy as np
import pytest
from n4m.model_selection.aom_search import POPPLSRegressor as NativePOPPLSRegressor
from sklearn.base import clone, is_classifier, is_regressor

from nirs4all.operators.models import POPPLSRegressor as PublicPOPPLSRegressor
from nirs4all.operators.models.sklearn.pop_pls import POPPLSRegressor
from nirs4all.operators.models.sklearn.pop_pls_classifier import POPPLSClassifier


@pytest.fixture
def regression_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(48, 24))
    y = X[:, :4].sum(axis=1) + 0.05 * rng.normal(size=X.shape[0])
    return X, y, np.arange(X.shape[0], dtype=np.int32) % 4


@pytest.fixture
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 40))
    labels = np.array(["classA", "classB"])
    return X, labels[(X[:, :3].sum(axis=1) > 0).astype(int)]


@pytest.fixture
def multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(43)
    X = rng.normal(size=(90, 40))
    scores = X[:, :3].sum(axis=1)
    y = np.where(scores < -0.5, "low", np.where(scores > 0.5, "high", "mid"))
    return X, y


class TestNativePOPPLSRegressor:
    def test_public_exports_are_the_n4m_estimator(self) -> None:
        assert POPPLSRegressor is NativePOPPLSRegressor
        assert PublicPOPPLSRegressor is NativePOPPLSRegressor

    def test_sklearn_contract_and_clone(self) -> None:
        model = POPPLSRegressor(
            max_components=4,
            operators=["identity", ("savgol_smooth", [5, 2])],
            cv=4,
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
        model = POPPLSRegressor(
            max_components=4,
            operators=["identity", ("savgol_smooth", [5, 2])],
            cv=4,
            fold_ids=fold_ids,
            scale_x=False,
        )

        assert model.fit(X, y) is model
        np.testing.assert_allclose(
            model.predict(X),
            np.asarray(model.result_["predictions"]).ravel(),
            rtol=1e-10,
            atol=1e-10,
        )
        diagnostics = model.get_diagnostics()
        selected = diagnostics["selected_operator_indices"]
        assert selected.shape == (model.selected_n_components_,)
        assert set(selected.tolist()) <= {0, 1}
        assert diagnostics["n_operators"] == 2

    def test_multioutput_prediction_shape(
        self,
        regression_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X, y, fold_ids = regression_data
        Y = np.column_stack((y, X[:, 5] - y))
        model = POPPLSRegressor(
            max_components=3,
            operators=["identity"],
            cv=4,
            fold_ids=fold_ids,
        ).fit(X, Y)
        assert model.predict(X).shape == Y.shape

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
        first = POPPLSRegressor(**kwargs).fit(X, y)
        second = POPPLSRegressor(**kwargs).fit(X, y)
        np.testing.assert_array_equal(first.predict(X), second.predict(X))
        np.testing.assert_array_equal(first.coef_, second.coef_)

    def test_invalid_operator_contract_fails_closed(
        self,
        regression_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X, y, fold_ids = regression_data
        with pytest.raises(ValueError, match="unknown AOM strict operator"):
            POPPLSRegressor(
                operators=["not_an_operator"],
                cv=4,
                fold_ids=fold_ids,
            ).fit(X, y)


class TestPOPPLSClassifier:
    """The classifier remains Python-owned until n4m exposes that surface."""

    def test_binary_fit_predict(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = binary_data
        model = POPPLSClassifier(n_components=5).fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert set(predictions) <= set(y)

    def test_binary_predict_proba(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = binary_data
        probabilities = POPPLSClassifier(n_components=5).fit(X, y).predict_proba(X)
        assert probabilities.shape == (len(y), 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
        assert np.all((probabilities >= 0) & (probabilities <= 1))

    def test_multiclass_fit_predict(self, multiclass_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = multiclass_data
        model = POPPLSClassifier(n_components=5).fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert set(predictions) <= set(y)
        np.testing.assert_array_equal(model.classes_, np.unique(y))

    def test_multiclass_predict_proba(self, multiclass_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = multiclass_data
        model = POPPLSClassifier(n_components=5).fit(X, y)
        probabilities = model.predict_proba(X)
        assert probabilities.shape == (len(y), len(np.unique(y)))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)

    def test_sklearn_contract(self) -> None:
        model = POPPLSClassifier(n_components=5, max_components=8)
        assert is_classifier(model)
        cloned = clone(model)
        assert cloned is not model
        assert cloned.get_params() == model.get_params()
