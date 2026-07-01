"""Focused tests for dag-ml operator routing special cases."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _isolated_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Override tests/unit's workspace fixture; this module never imports real nirs4all."""
    monkeypatch.setenv("NIRS4ALL_WORKSPACE", str(tmp_path / "_test_workspace"))


def _load_operator_routing() -> ModuleType:
    path = Path(__file__).parents[3] / "nirs4all" / "pipeline" / "dagml" / "operator_routing.py"
    spec = importlib.util.spec_from_file_location("_n4a_operator_routing_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load operator_routing.py from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_nirs4all(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, calls: list[dict[str, object]]) -> type:
    nirs4all = ModuleType("nirs4all")
    nirs4all.__path__ = []
    operators = ModuleType("nirs4all.operators")
    operators.__path__ = []
    transforms = ModuleType("nirs4all.operators.transforms")
    transforms.__path__ = []
    scalers = ModuleType("nirs4all.operators.transforms.scalers")
    methods = ModuleType("nirs4all.operators.methods")
    methods.__path__ = []
    n4m_ops = ModuleType("nirs4all.operators.methods.n4m_ops")

    class StandardNormalVariate:
        def __init__(self, axis: int = 1, with_mean: bool = True, with_std: bool = True, ddof: int = 0, copy: bool = True) -> None:
            self.axis = axis
            self.with_mean = with_mean
            self.with_std = with_std
            self.ddof = ddof
            self.copy = copy

        def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002 - sklearn-compatible signature
            return {"axis": self.axis, "with_mean": self.with_mean, "with_std": self.with_std, "ddof": self.ddof, "copy": self.copy}

    class FakeN4MSNV:
        def __init__(self, *, with_mean: bool, with_std: bool, ddof: int) -> None:
            calls.append({"with_mean": with_mean, "with_std": with_std, "ddof": ddof})

        def fit_transform(self, X: object) -> np.ndarray:
            return np.asarray(X, dtype=float) + 7.0

    class MethodsSNV:
        def __init__(self, with_mean: bool = True, with_std: bool = True, ddof: int = 0) -> None:
            self.with_mean = with_mean
            self.with_std = with_std
            self.ddof = ddof

        def fit_transform(self, X: object) -> np.ndarray:
            backend = n4m_ops._N4MSNV(with_mean=self.with_mean, with_std=self.with_std, ddof=self.ddof)
            return backend.fit_transform(X)

        def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002 - sklearn-compatible signature
            return {"with_mean": self.with_mean, "with_std": self.with_std, "ddof": self.ddof}

    StandardNormalVariate.__module__ = "nirs4all.operators.transforms.scalers"
    MethodsSNV.__module__ = "nirs4all.operators.methods.n4m_ops"
    scalers.StandardNormalVariate = StandardNormalVariate  # type: ignore[attr-defined]
    transforms.StandardNormalVariate = StandardNormalVariate  # type: ignore[attr-defined]
    n4m_ops.MethodsSNV = MethodsSNV  # type: ignore[attr-defined]
    n4m_ops._N4MSNV = FakeN4MSNV  # type: ignore[attr-defined]
    methods.MethodsSNV = MethodsSNV  # type: ignore[attr-defined]
    methods.n4m_ops = n4m_ops  # type: ignore[attr-defined]

    fake_lib = tmp_path / "libn4m.so"
    fake_lib.write_text("")
    n4m = ModuleType("n4m")
    n4m.abi_version = lambda: "test-abi"  # type: ignore[attr-defined]
    n4m.library_path = lambda: str(fake_lib)  # type: ignore[attr-defined]

    for module in (nirs4all, operators, transforms, scalers, methods, n4m_ops, n4m):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return StandardNormalVariate


def test_standard_snv_defaults_to_python_route(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    standard_snv = _install_fake_nirs4all(monkeypatch, tmp_path, calls)
    routing = _load_operator_routing()
    monkeypatch.delenv("N4A_DAGML_METHODS_SNV", raising=False)

    transform = routing.route_graph_node(
        {
            "kind": "transform",
            "operator": {"class": "nirs4all.operators.transforms.scalers.StandardNormalVariate"},
            "params": {"axis": 1, "with_mean": True, "with_std": True, "ddof": 0, "copy": True},
        }
    )

    assert isinstance(transform, standard_snv)
    assert calls == []


def test_standard_snv_opt_in_routes_to_methods_and_invokes_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    _install_fake_nirs4all(monkeypatch, tmp_path, calls)
    routing = _load_operator_routing()
    monkeypatch.setenv("N4A_DAGML_METHODS_SNV", "1")

    transform = routing.route_graph_node(
        {
            "kind": "transform",
            "operator": {"class": "nirs4all.operators.transforms.scalers.StandardNormalVariate"},
            "params": {"axis": 1, "with_mean": False, "with_std": True, "ddof": 1, "copy": True},
        }
    )

    assert type(transform).__name__ == "MethodsSNV"
    np.testing.assert_allclose(transform.fit_transform(np.asarray([[1.0, 2.0, 3.0]])), [[8.0, 9.0, 10.0]])
    assert calls == [{"with_mean": False, "with_std": True, "ddof": 1}]


def test_standard_snv_opt_in_refuses_unsafe_or_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    _install_fake_nirs4all(monkeypatch, tmp_path, calls)
    routing = _load_operator_routing()
    monkeypatch.setenv("N4A_DAGML_METHODS_SNV", "1")

    with pytest.raises(ValueError, match="axis=1"):
        routing.route_graph_node(
            {
                "kind": "transform",
                "operator": {"class": "nirs4all.operators.transforms.scalers.StandardNormalVariate"},
                "params": {"axis": 0, "with_mean": True, "with_std": True, "ddof": 0, "copy": True},
            }
        )

    sys.modules["nirs4all.operators.methods.n4m_ops"]._N4MSNV = None  # type: ignore[attr-defined]
    with pytest.raises(ImportError, match="SNV binding"):
        routing.route_graph_node(
            {
                "kind": "transform",
                "operator": {"class": "nirs4all.operators.transforms.scalers.StandardNormalVariate"},
                "params": {"axis": 1, "with_mean": True, "with_std": True, "ddof": 0, "copy": True},
            }
        )


def test_pls_short_alias_stays_sklearn_until_methods_route_has_pipeline_parity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Automatic PLSRegression -> MethodsPLS routing is intentionally not enabled.

    MethodsPLS is available as an explicit opt-in operator, but replacing a
    sklearn PLS node requires a dedicated pipeline-parity gate for single-target
    shape, scaling, fold scope, and native component-selection semantics.
    """
    calls: list[dict[str, object]] = []
    _install_fake_nirs4all(monkeypatch, tmp_path, calls)
    routing = _load_operator_routing()

    model = routing.route_operator("model", "PLSRegression", {"n_components": 2})

    assert type(model).__name__ == "PLSRegression"
    assert type(model).__module__.startswith("sklearn.cross_decomposition")
