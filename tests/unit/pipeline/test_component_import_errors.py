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


@pytest.mark.parametrize("params, rejected", [
    ({"alphaa": 3.0}, "['alphaa']"),
    ({"alpha": 2.0, "n_components": 12, "center": True}, "['center', 'n_components']"),
])
def test_unknown_constructor_parameters_are_errors_not_defaults(params, rejected):
    """A model must never silently train with defaults instead of the requested parameters."""
    with pytest.raises(ValueError, match="does not accept parameter") as caught:
        deserialize_component({"class": "sklearn.linear_model.Ridge", "params": params})
    message = str(caught.value)
    assert rejected in message
    assert "valid parameters: ['alpha', 'copy_X'" in message


def test_aom_pls_rejects_legacy_parameter_names():
    pytest.importorskip("n4m")
    with pytest.raises(ValueError, match=r"does not accept parameter\(s\) \['center', 'n_components', 'scale'\]") as caught:
        deserialize_component({"class": "nirs4all.operators.models.AOMPLSRegressor", "params": {"n_components": 12, "center": True, "scale": False}})
    assert "max_components" in str(caught.value)
    model = deserialize_component({"class": "nirs4all.operators.models.AOMPLSRegressor", "params": {"max_components": 12}})
    assert model.max_components == 12


def test_repository_pipeline_configs_use_accepted_parameters():
    """Example configs must not depend on a parameter-dropping fallback."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[3]
    declared = []

    def collect(node):
        if isinstance(node, dict):
            if isinstance(node.get("class"), str) and isinstance(node.get("params"), dict) and node["params"]:
                declared.append(node)
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    for path in sorted((root / "examples" / "configs").rglob("*.y*ml")):
        collect(yaml.safe_load(path.read_text()))
    assert declared
    for node in declared:
        deserialize_component(node)


def test_neural_factories_reject_unknown_parameters_but_ignore_injected_hints():
    from nirs4all.controllers.models.factory import ModelFactory

    def factory(input_shape, hidden_units=8):
        return {"input_shape": input_shape, "hidden_units": hidden_units}

    assert ModelFactory.prepare_and_call(factory, {"hidden_units": 64, "input_shape": (4,), "num_classes": 3}) == {"input_shape": (4,), "hidden_units": 64}
    with pytest.raises(ValueError, match=r"does not accept parameter\(s\) \['hidden_unitz'\]; valid parameters: \['hidden_units', 'input_shape'\]"):
        ModelFactory.prepare_and_call(factory, {"hidden_unitz": 64, "input_shape": (4,)})

    def bundled(input_shape, params):
        return params

    assert ModelFactory.prepare_and_call(bundled, {"input_shape": (4,), "anything": 1}) == {"anything": 1}

    class Deferred:
        def __init__(self, input_shape=None, width=4):
            self.input_shape, self.width = input_shape, width

    assert ModelFactory.reconstruct_object(Deferred(), force_params={"input_shape": (5,), "width": 9}).width == 9
    with pytest.raises(ValueError, match=r"\['widht'\]"):
        ModelFactory.reconstruct_object(Deferred(), force_params={"input_shape": (5,), "widht": 9})
