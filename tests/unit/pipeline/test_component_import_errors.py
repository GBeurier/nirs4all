"""Executable references retain dependency errors; parameter strings stay permissive."""
import sys
from types import ModuleType

import pytest

from nirs4all.pipeline.config.component_serialization import deserialize_component
from nirs4all.pipeline.steps.parser import StepParser


@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("failure", [
    ModuleNotFoundError("No module named 'tabpfn'", name="tabpfn"),
    ImportError("cannot import name 'some_api' from 'torch'"),
    AttributeError("module 'numpy' has no attribute 'removed_api'"),
])
def test_step_import_preserves_dependency_failure(monkeypatch, as_dict, failure):
    from nirs4all.pipeline.config import component_serialization
    original = component_serialization.importlib.import_module

    def import_component(name):
        if name == "tabpfn":
            raise failure
        return original(name)

    monkeypatch.setattr(component_serialization.importlib, "import_module", import_component)
    reference = "tabpfn.TabPFNRegressor"
    component = {"class": reference} if as_dict else reference
    with pytest.raises(ValueError, match="Could not deserialize component") as caught:
        StepParser().parse({"model": component})
    assert str(failure) in str(caught.value)
    assert type(failure).__name__ in str(caught.value)
    assert sys.executable in str(caught.value)
    assert caught.value.__cause__ is failure


def test_available_public_tabpfn_reference_keeps_constructor_parameters(monkeypatch):
    class TabPFNRegressor:
        def __init__(self, *, device="auto", ignore_pretraining_limits=False):
            self.device = device
            self.ignore_pretraining_limits = ignore_pretraining_limits

    module = ModuleType("tabpfn")
    module.TabPFNRegressor = TabPFNRegressor
    monkeypatch.setitem(sys.modules, "tabpfn", module)
    parsed = StepParser().parse({"model": {
        "class": "tabpfn.TabPFNRegressor",
        "params": {"device": "cpu", "ignore_pretraining_limits": True},
    }})
    assert isinstance(parsed.operator, TabPFNRegressor)
    assert parsed.operator.device == "cpu"
    assert parsed.operator.ignore_pretraining_limits is True


def test_non_operator_string_parameters_remain_permissive():
    assert deserialize_component("missing_package.config") == "missing_package.config"
    parsed = StepParser().parse({"model": {
        "class": "sklearn.linear_model.Ridge",
        "params": {"solver": "some.custom.value"},
    }})
    assert parsed.operator.solver == "some.custom.value"
