"""Unit tests for :class:`AOMPLSAomlibRegressor`.

These tests require the ``aompls`` package (AOM_lib C++ backend) to be
importable. Invoke pytest with::

    PYTHONPATH=bench/AOM_lib/python/src pytest tests/unit/operators/models/test_aom_pls_aomlib.py

The whole module is skipped when ``aompls`` is unavailable so the rest of
the test suite keeps running on environments without the compiled
extension.
"""

from __future__ import annotations

import builtins
import sys

import numpy as np
import pytest
from sklearn.datasets import make_regression

pytest.importorskip("aompls")

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
    with pytest.raises(ValueError, match="Unknown selection mode"):
        model.fit(X, y)


def test_missing_aompls_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
    small_regression: tuple[np.ndarray, np.ndarray],
) -> None:
    """When ``aompls`` cannot be imported, fit raises an informative ImportError."""
    X, y = small_regression

    # Remove cached aompls modules and force `import aompls` to fail.
    for name in list(sys.modules):
        if name == "aompls" or name.startswith("aompls."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    original_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "aompls" or name.startswith("aompls."):
            raise ImportError("simulated missing aompls")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    model = AOMPLSAomlibRegressor(n_components=4, cv=3)
    with pytest.raises(ImportError, match="aompls"):
        model.fit(X, y)
