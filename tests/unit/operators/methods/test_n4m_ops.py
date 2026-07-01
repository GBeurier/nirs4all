"""Tests for the opt-in ``nirs4all-methods``-backed operators.

Covers five things:

1. **Packaging / import** — the ``n4m`` binding imports and ``libn4m.so``
   resolves in the intended env (the operator module reports it available).
2. **SNV parity vs the methods bit-exact fixture** — the native SNV matches
   the ``nirs4all-methods`` golden ``snv_v1.json`` fixture (and sklearn) to
   floating-point precision.
3. **SNV→PLS parity vs sklearn** — ``MethodsSNV`` → ``MethodsPLS`` matches
   ``StandardNormalVariate`` → ``PLSRegression`` within tolerance.
4. **Absent-binding diagnostic** — when the binding is absent, the tests assert
   the explicit blocker instead of skipping the proof surface.
5. **Dual-engine execution** — the operators run through the normal nirs4all
   pipeline on BOTH the legacy engine and the dag-ml engine, proving the
   sklearn-contract dispatch path works on each.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pytest

from nirs4all.operators import methods

_REQUIRE_N4M = os.environ.get("NIRS4ALL_REQUIRE_N4M") == "1"

pytestmark = pytest.mark.methods

MethodsSNV = methods.MethodsSNV
MethodsPLS = methods.MethodsPLS

# The methods repo's bit-exact parity fixtures live in the sibling checkout.
_METHODS_FIXTURES = Path("/home/delete/nirs4all/nirs4all-methods/parity/fixtures")


def _assert_useful_unavailable_status(status: dict[str, Any]) -> None:
    assert status["available"] is False
    message = str(status["message"])
    mitigation = str(status["mitigation"])
    assert "nirs4all-methods" in message
    assert "n4m" in message
    assert "MethodsSNV" in message
    assert "MethodsPLS" in message
    assert "Install a compatible `nirs4all-methods` wheel" in mitigation
    assert (
        status["snv_available"] is False
        or status["pls_available"] is False
        or "libn4m" in message
        or "abi" in message
        or "library_path" in message
    )


@pytest.fixture
def methods_available() -> bool:
    status = methods.methods_binding_status()
    if status["available"]:
        return True
    _assert_useful_unavailable_status(status)
    if _REQUIRE_N4M:
        pytest.fail(f"{status['message']} {status['mitigation']}", pytrace=True)
    return False


def _skip_unavailable_binding() -> NoReturn:
    status = methods.methods_binding_status()
    pytest.skip(f"{status['message']} {status['mitigation']}")


def _hexbe_to_array(hex_list: list[str], rows: int, cols: int) -> np.ndarray:
    """Decode an ``ieee754_binary64_be_hex`` fixture column into an array."""
    buf = b"".join(bytes.fromhex(h) for h in hex_list)
    return np.frombuffer(buf, dtype=">f8").astype(np.float64).reshape(rows, cols)


# ----------------------------------------------------------------------
# 1. Packaging / import
# ----------------------------------------------------------------------


class TestPackaging:
    def test_binding_status_consumes_installed_n4m_or_reports_blocker(self, methods_available: bool) -> None:
        status = methods.methods_binding_status()
        if not methods_available:
            with pytest.raises(ImportError, match="nirs4all-methods"):
                MethodsSNV().fit_transform(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64))
            with pytest.raises(ImportError, match="nirs4all-methods"):
                MethodsPLS(n_components=1, cv=2).fit(np.asarray([[1.0, 2.0], [2.0, 3.0]], dtype=np.float64), np.asarray([1.0, 2.0], dtype=np.float64))
            return

        assert methods.METHODS_AVAILABLE is True
        assert status["abi_version"]
        assert status["library_path"]
        assert Path(str(status["library_path"])).exists()

        X = np.asarray(
            [
                [1.0, 2.0, 4.0, 8.0],
                [2.0, 3.0, 5.0, 7.0],
                [3.0, 1.0, 4.0, 1.0],
                [5.0, 9.0, 2.0, 6.0],
                [8.0, 5.0, 9.0, 7.0],
                [2.0, 7.0, 1.0, 8.0],
            ],
            dtype=np.float64,
        )
        y = X[:, 0] - 0.25 * X[:, 2] + 0.1 * X[:, 3]
        Xs = MethodsSNV().fit_transform(X)
        np.testing.assert_allclose(Xs.mean(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(Xs.std(axis=1, ddof=0), 1.0, atol=1e-12)

        pred = MethodsPLS(n_components=2, cv=2, scale_x=True).fit(Xs, y).predict(Xs)
        assert pred.shape == (X.shape[0],)
        assert np.all(np.isfinite(pred))

    def test_methods_available(self, methods_available: bool) -> None:
        if not methods_available:
            _skip_unavailable_binding()
        assert methods.METHODS_AVAILABLE is True

    def test_libn4m_resolves(self, methods_available: bool) -> None:
        if not methods_available:
            _skip_unavailable_binding()
        import n4m

        # Loads + queries the embedded libn4m; raises if the .so cannot resolve.
        assert n4m.abi_version()
        assert Path(n4m.library_path()).exists()

    def test_operators_are_fqn_importable(self, methods_available: bool) -> None:
        if not methods_available:
            _skip_unavailable_binding()
        # The dag-ml path imports operators by fully-qualified name; the FQN
        # must resolve back to the same classes.
        import importlib

        mod = importlib.import_module(MethodsSNV.__module__)
        assert getattr(mod, MethodsSNV.__qualname__) is MethodsSNV
        assert getattr(mod, MethodsPLS.__qualname__) is MethodsPLS

    def test_params_are_json_serializable(self, methods_available: bool) -> None:
        if not methods_available:
            _skip_unavailable_binding()
        json.dumps(MethodsSNV(ddof=1).get_params())
        json.dumps(MethodsPLS(n_components=3, cv=4).get_params())

    def test_set_params_validates_unknown_keys(self, methods_available: bool) -> None:
        if not methods_available:
            _skip_unavailable_binding()
        # sklearn-contract: set_params with a real param is reflected in
        # get_params; an unknown param raises ValueError (no silent setattr).
        snv = MethodsSNV()
        snv.set_params(ddof=1, with_std=False)
        assert snv.get_params()["ddof"] == 1
        assert snv.get_params()["with_std"] is False

        pls = MethodsPLS()
        pls.set_params(n_components=7, cv=3)
        assert pls.get_params()["n_components"] == 7
        assert pls.get_params()["cv"] == 3

        with pytest.raises(ValueError):
            MethodsSNV().set_params(not_a_param=1)
        with pytest.raises(ValueError):
            MethodsPLS().set_params(bogus=2)


# ----------------------------------------------------------------------
# 2. SNV parity vs the methods bit-exact fixture
# ----------------------------------------------------------------------


class TestSNVFixtureParity:
    def test_snv_matches_methods_golden_fixture(self, methods_available: bool) -> None:
        if not methods_available:
            _skip_unavailable_binding()
        fixture_path = _METHODS_FIXTURES / "snv_v1.json"
        assert fixture_path.exists(), f"methods SNV fixture missing: {fixture_path}"
        fixture = json.loads(fixture_path.read_text())
        rows, cols = fixture["rows"], fixture["cols"]
        X = _hexbe_to_array(fixture["input_hex"], rows, cols)

        for case in fixture["cases"]:
            params = case["params"]
            expected = _hexbe_to_array(case["output_hex"], rows, cols)
            got = MethodsSNV(**params).fit_transform(X)
            # The golden output was generated by the 0.9.1 numpy reference; the
            # native engine matches it to floating-point precision.
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)

    def test_snv_matches_sklearn_default(self, methods_available: bool) -> None:
        if not methods_available:
            _skip_unavailable_binding()
        fixture_path = _METHODS_FIXTURES / "snv_v1.json"
        assert fixture_path.exists(), f"methods SNV fixture missing: {fixture_path}"
        fixture = json.loads(fixture_path.read_text())
        X = _hexbe_to_array(fixture["input_hex"], fixture["rows"], fixture["cols"])
        # Reference SNV: row-wise (x - mean) / std.
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, ddof=0, keepdims=True)
        ref = (X - mu) / sd
        got = MethodsSNV().fit_transform(X)
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)


# ----------------------------------------------------------------------
# 3. SNV → PLS parity vs sklearn
# ----------------------------------------------------------------------


class TestSNVPLSvsSklearn:
    def _data(self, methods_available: bool) -> tuple[np.ndarray, np.ndarray]:
        assert methods_available
        fixture_path = _METHODS_FIXTURES / "synthetic_small_pls2_v1.json"
        if fixture_path.exists():
            fixture = json.loads(fixture_path.read_text())

            def _arr(o: dict) -> np.ndarray:
                return np.asarray(o["values"], dtype=np.float64).reshape(o["shape"])

            X = _arr(fixture["data"]["X"])
            y = _arr(fixture["data"]["Y"])[:, 0]  # single target -> PLS1
            return X, y
        rng = np.random.default_rng(7)
        X = rng.standard_normal((60, 24))
        y = X[:, 0] - 0.4 * X[:, 5] + 0.02 * rng.standard_normal(60)
        return X, y

    def test_pls_matches_sklearn_on_common_input(self, methods_available: bool) -> None:
        if not methods_available:
            _skip_unavailable_binding()
        from sklearn.cross_decomposition import PLSRegression

        X, y = self._data(methods_available)
        # Common SNV input isolates the PLS comparison.
        Xs = MethodsSNV().fit_transform(X)

        methods_pls = MethodsPLS(n_components=3, cv=2, scale_x=True).fit(Xs, y)
        pred_methods = methods_pls.predict(Xs)

        sk = PLSRegression(n_components=3, scale=True).fit(Xs, y)
        pred_sklearn = sk.predict(Xs).ravel()

        max_diff = float(np.max(np.abs(pred_methods - pred_sklearn)))
        assert max_diff < 1e-9, f"methods vs sklearn PLS pred max|diff|={max_diff:.3e}"

    def test_full_snv_pls_pipeline_matches_sklearn(self, methods_available: bool) -> None:
        if not methods_available:
            _skip_unavailable_binding()
        from sklearn.cross_decomposition import PLSRegression

        from nirs4all.operators.transforms import StandardNormalVariate

        X, y = self._data(methods_available)

        # Methods pipeline.
        Xs_m = MethodsSNV().fit_transform(X)
        pred_m = MethodsPLS(n_components=3, cv=2, scale_x=True).fit(Xs_m, y).predict(Xs_m)

        # Pure-Python nirs4all SNV + sklearn PLS pipeline.
        Xs_p = StandardNormalVariate().fit_transform(X)
        pred_p = PLSRegression(n_components=3, scale=True).fit(Xs_p, y).predict(Xs_p).ravel()

        rmse_m = float(np.sqrt(np.mean((pred_m - y) ** 2)))
        rmse_p = float(np.sqrt(np.mean((pred_p - y) ** 2)))
        assert abs(rmse_m - rmse_p) < 1e-6
        np.testing.assert_allclose(pred_m, pred_p, rtol=0, atol=1e-8)


# ----------------------------------------------------------------------
# 4. Dual-engine execution
# ----------------------------------------------------------------------


class TestDualEngine:
    def _pipeline(self) -> list:
        from sklearn.model_selection import ShuffleSplit

        return [
            MethodsSNV(),
            ShuffleSplit(n_splits=2, random_state=0),
            {"model": MethodsPLS(n_components=5, cv=2)},
        ]

    @pytest.mark.parametrize("engine", ["legacy", "dag-ml"])
    def test_runs_under_engine(self, engine: str, methods_available: bool) -> None:
        if not methods_available:
            _skip_unavailable_binding()
        import nirs4all

        # Single-target corpus: the native engine is PLS1 (single-output).
        dataset = nirs4all.generate.regression(n_samples=80, random_state=0, target_component=0)
        result = nirs4all.run(
            pipeline=self._pipeline(),
            dataset=dataset,
            engine=engine,
            verbose=0,
        )
        assert result is not None
        assert hasattr(result, "best_rmse")
        assert np.isfinite(result.best_rmse)
        assert result.best_rmse > 0
