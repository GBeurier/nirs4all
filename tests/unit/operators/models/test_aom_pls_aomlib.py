"""Real installed nirs4all-methods binding parity and saved-model replay."""

from __future__ import annotations

import builtins
import pickle

import numpy as np
import pytest
from n4m.model_selection.aom_search import AOMPLSRegressor
from sklearn.base import clone, is_regressor
from sklearn.datasets import make_regression

from nirs4all.operators.models.sklearn import AOMPLSAomlibRegressor  # noqa: E402


@pytest.fixture
def small_regression() -> tuple[np.ndarray, np.ndarray]:
    """Return a tiny regression dataset for fast unit tests."""
    X, y = make_regression(
        n_samples=40,
        n_features=80,
        n_informative=10,
        noise=0.1,
        random_state=0,
    )
    return X.astype(np.float64), y.astype(np.float64)


def test_instantiation_default_params() -> None:
    """Constructor stores parameters without touching the backend."""
    model = AOMPLSAomlibRegressor()
    assert model.n_components == 15
    assert model.selection == "cv"
    assert model.cv == 5
    assert model.one_se is False
    assert model.preprocessing is None


def test_fit_predict_shape(small_regression: tuple[np.ndarray, np.ndarray]) -> None:
    """fit/predict produce a 1D float array of the right length."""
    X, y = small_regression
    model = AOMPLSAomlibRegressor(n_components=6, cv=3, random_state=0)
    fitted = model.fit(X, y)
    assert fitted is model

    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)
    assert preds.dtype == np.float64
    assert np.isfinite(preds).all()


def test_fit_populates_diagnostics(small_regression: tuple[np.ndarray, np.ndarray]) -> None:
    """Diagnostic attributes mirror the legacy AOMPLSRegressor surface."""
    X, y = small_regression
    model = AOMPLSAomlibRegressor(n_components=6, cv=3, random_state=0).fit(X, y)

    assert model.n_features_in_ == X.shape[1]
    assert 1 <= model.n_components_selected_ <= 6

    assert isinstance(model.selected_operator_sequence_, list)
    assert len(model.selected_operator_sequence_) == 1
    assert isinstance(model.selected_operator_sequence_[0], str)

    assert model.selected_operator_scores_.ndim == 2
    assert model.selected_operator_scores_.shape[1] == 6  # K_max curves

    assert isinstance(model.bank_names_, list) and len(model.bank_names_) > 0
    assert model.selected_operator_index_ >= 0


def test_get_set_params_roundtrip() -> None:
    """get_params/set_params follow the sklearn contract."""
    model = AOMPLSAomlibRegressor()
    params = model.get_params()
    expected_keys = {
        "n_components",
        "selection",
        "cv",
        "one_se",
        "preprocessing",
        "random_state",
        "osc_n_components",
        "asls_lam",
        "asls_p",
        "asls_n_iter",
        "center",
        "external_folds",
    }
    assert set(params) == expected_keys

    model.set_params(n_components=7, cv=4, preprocessing="asls", one_se=True)
    assert model.n_components == 7
    assert model.cv == 4
    assert model.preprocessing == "asls"
    assert model.one_se is True

    with pytest.raises(ValueError):
        model.set_params(does_not_exist=42)


def test_selection_alias_maps_to_kfold(small_regression: tuple[np.ndarray, np.ndarray]) -> None:
    """``selection='cv'`` is an alias for the K-fold backend mode."""
    X, y = small_regression
    model = AOMPLSAomlibRegressor(n_components=4, cv=3, selection="cv", random_state=0).fit(X, y)
    # If the alias mapping was broken, fit would have raised ValueError from the C++ backend.
    assert model.n_components_selected_ >= 1


def test_unknown_selection_raises() -> None:
    """An unknown selection mode is rejected before reaching the backend."""
    model = AOMPLSAomlibRegressor(selection="nope")
    X = np.zeros((10, 5), dtype=np.float64)
    y = np.zeros(10, dtype=np.float64)
    with pytest.raises(ValueError, match="Unsupported selection mode"):
        model.fit(X, y)


def test_missing_methods_raises_clear_error(monkeypatch, small_regression):
    X, y = small_regression
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "n4m.model_selection.aom_search":
            raise ImportError("simulated missing nirs4all-methods")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="nirs4all-methods native Python wheel"):
        AOMPLSAomlibRegressor(n_components=4, cv=3).fit(X, y)


def test_matches_public_native_binding_and_pickle_replay(small_regression):
    X, y = small_regression
    model = AOMPLSAomlibRegressor(n_components=4, cv=3, random_state=42).fit(X, y)
    native = AOMPLSRegressor(max_components=4, cv=3, fold_ids=model._backend.result_["fold_ids"],
                            center_x=True, center_y=True, scale_x=False, scale_y=False).fit(X, y)
    np.testing.assert_allclose(model.predict(X), native.predict(X), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pickle.loads(pickle.dumps(model)).predict(X), model.predict(X), rtol=1e-12)
    assert model.backend_ == "nirs4all-methods"
    assert is_regressor(model)
    assert clone(model).get_params() == model.get_params()


@pytest.mark.parametrize("parameters", [
    {"one_se": True}, {"selection": "spxy"}, {"selection": "holdout"}, {"preprocessing": "asls"},
    {"osc_n_components": 2}, {"selection": "external"}, {"external_folds": [[0], [1]]},
])
def test_unsupported_legacy_options_are_never_silently_ignored(small_regression, parameters):
    X, y = small_regression
    with pytest.raises(ValueError):
        AOMPLSAomlibRegressor(**parameters).fit(X, y)


def test_external_validation_folds_are_preserved(small_regression):
    X, y = small_regression
    folds = [list(range(0, 20)), list(range(20, 40))]
    model = AOMPLSAomlibRegressor(n_components=3, selection="external", external_folds=folds).fit(X, y)
    assert model.fold_indices_ == folds
    with pytest.raises(ValueError, match="exactly once"):
        AOMPLSAomlibRegressor(selection="external", external_folds=[[0, 1], [1, 2]]).fit(X, y)
