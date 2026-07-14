"""Native finetuning search-space contract tests that do not require libn4m."""

from __future__ import annotations

import pytest

from nirs4all.optimization import n4m_engine
from nirs4all.optimization.n4m_engine import N4MFinetuneManager


class _FakeConstraintKind:
    CONDITION_IN = "condition_in"
    CONDITION_NOT_IN = "condition_not_in"


class _FakeSearchSpace:
    def __init__(self) -> None:
        self.axes: list[tuple] = []
        self.constraints: list[tuple] = []

    def add_int(self, name, low, high, step=1, log=False) -> None:
        self.axes.append(("int", name, low, high, step, log))

    def add_float(self, name, low, high, step=0.0, log=False) -> None:
        self.axes.append(("float", name, low, high, step, log))

    def add_categorical(self, name, choices) -> None:
        self.axes.append(("categorical", name, tuple(choices)))

    def add_constraint(self, kind, names, labels) -> None:
        self.constraints.append((kind, tuple(names), tuple(labels)))


class _FakeTrial:
    def __init__(
        self,
        *,
        active: set[str],
        ints: dict[str, int] | None = None,
        floats: dict[str, float] | None = None,
        categories: dict[str, int] | None = None,
    ) -> None:
        self._active = active
        self._ints = ints or {}
        self._floats = floats or {}
        self._categories = categories or {}

    def is_active(self, name: str) -> bool:
        return name in self._active

    def get_int(self, name: str) -> int:
        return self._ints[name]

    def get_float(self, name: str) -> float:
        return self._floats[name]

    def get_category(self, name: str) -> tuple[int, str]:
        idx = self._categories[name]
        return idx, str(idx)


def _compile_contract(monkeypatch):
    monkeypatch.setattr(n4m_engine, "SearchSpace", _FakeSearchSpace, raising=False)
    monkeypatch.setattr(n4m_engine, "ConstraintKind", _FakeConstraintKind, raising=False)

    return N4MFinetuneManager()._compile_space(
        {
            "model_params": {
                "scale": {"type": "categorical", "options": {"std": "STD", "mm": "MM"}},
                "est": {"options": {"pls": "PLS", "ridge": "RIDGE"}},
                "est__alpha": {
                    "type": "float_log",
                    "min": 1e-4,
                    "max": 1.0,
                    "when": {"est": "ridge"},
                },
                "cfg": {
                    "mode": "fast",
                    "depth": {"type": "int", "min": 1, "max": 3},
                },
            },
            "train_params": {
                "epochs": {"type": "int", "min": 5, "max": 20, "step": 5},
                "verbose": 0,
            },
        }
    )


def test_n4m_normalize_canonicalizes_public_string_knobs_without_native_runtime():
    params = N4MFinetuneManager()._normalize(
        {
            "engine": "n4m",
            "sample": " GRID ",
            "pruner": " HyperBand ",
            "approach": " Individual ",
            "eval_mode": " AVG ",
            "direction": " MAXIMIZE ",
            "n_trials": 3,
        }
    )

    assert "engine" not in params
    assert "sample" not in params
    assert params["sampler"] == "grid"
    assert params["pruner"] == "hyperband"
    assert params["approach"] == "individual"
    assert params["eval_mode"] == "mean"
    assert params["direction"] == "maximize"
    assert params["n_trials"] == 3


def test_n4m_normalize_drops_sample_alias_when_sampler_is_explicit() -> None:
    params = N4MFinetuneManager()._normalize(
        {
            "sample": "grid",
            "sampler": " TPE ",
            "model_params": {"alpha": [0.1, 0.2]},
        }
    )

    assert "sample" not in params
    assert params["sampler"] == "tpe"


def test_n4m_compiles_public_search_space_dsl_without_native_runtime(monkeypatch):
    """Compilation covers the Optuna-facing DSL before native ask/tell starts."""
    space, slots, static_model, static_train = _compile_contract(monkeypatch)

    assert space.axes == [
        ("categorical", "scale", ("std", "mm")),
        ("categorical", "est", ("pls", "ridge")),
        ("float", "est__alpha", 0.0001, 1.0, 0.0, True),
        ("int", "cfg__depth", 1, 3, 1, False),
        ("int", "train.epochs", 5, 20, 5, False),
    ]
    assert space.constraints == [
        ("condition_in", ("est__alpha", "est"), ("", "ridge")),
    ]
    assert static_model == {"cfg__mode": "fast"}
    assert static_train == {"verbose": 0}
    assert [(slot.native, slot.origin_name, slot.kind, slot.is_train) for slot in slots] == [
        ("scale", "scale", "categorical", False),
        ("est", "est", "categorical", False),
        ("est__alpha", "est__alpha", "float", False),
        ("cfg__depth", "cfg__depth", "int", False),
        ("train.epochs", "epochs", "int", True),
    ]


def test_n4m_resolves_inactive_operator_params_without_leakage(monkeypatch):
    _, slots, static_model, static_train = _compile_contract(monkeypatch)
    manager = N4MFinetuneManager()
    flat_heads = {slot.origin_name for slot in slots if not slot.is_train and "__" not in slot.origin_name} | {key for key in static_model if "__" not in key}

    trial = _FakeTrial(
        active={"scale", "est", "cfg__depth", "train.epochs"},
        ints={"cfg__depth": 2, "train.epochs": 10},
        categories={"scale": 0, "est": 0},
    )

    model_params, train_params = manager._resolve(trial, slots, static_model, static_train, flat_heads)

    assert model_params == {"scale": "STD", "est": "PLS", "cfg": {"mode": "fast", "depth": 2}}
    assert "est__alpha" not in model_params
    assert train_params == {"verbose": 0, "epochs": 10}


def test_n4m_resolves_active_operator_subparams_as_sklearn_set_params(monkeypatch):
    _, slots, static_model, static_train = _compile_contract(monkeypatch)
    manager = N4MFinetuneManager()
    flat_heads = {slot.origin_name for slot in slots if not slot.is_train and "__" not in slot.origin_name} | {key for key in static_model if "__" not in key}

    trial = _FakeTrial(
        active={"scale", "est", "est__alpha", "cfg__depth", "train.epochs"},
        ints={"cfg__depth": 3, "train.epochs": 15},
        floats={"est__alpha": 0.1},
        categories={"scale": 1, "est": 1},
    )

    model_params, train_params = manager._resolve(trial, slots, static_model, static_train, flat_heads)

    assert model_params == {
        "scale": "MM",
        "est": "RIDGE",
        "est__alpha": 0.1,
        "cfg": {"mode": "fast", "depth": 3},
    }
    assert train_params == {"verbose": 0, "epochs": 15}


def test_n4m_force_params_encode_public_values_before_native_enqueue(monkeypatch):
    _, slots, _, _ = _compile_contract(monkeypatch)
    manager = N4MFinetuneManager()

    encoded = manager._encode_force_params(
        slots,
        {"scale": "MM", "est": "RIDGE", "cfg__depth": 2},
    )

    assert encoded == {"scale": "mm", "est": "ridge", "cfg__depth": 2}


def test_n4m_force_params_reject_unknown_or_train_keys(monkeypatch):
    _, slots, _, _ = _compile_contract(monkeypatch)
    manager = N4MFinetuneManager()

    with pytest.raises(ValueError, match="subset of sampled finetune_params.model_params"):
        manager._encode_force_params(slots, {"epochs": 10})

    with pytest.raises(ValueError, match="unknown keys: \\['missing'\\]"):
        manager._encode_force_params(slots, {"missing": 1})


def test_n4m_force_params_reject_internal_option_labels(monkeypatch):
    _, slots, _, _ = _compile_contract(monkeypatch)
    manager = N4MFinetuneManager()

    with pytest.raises(ValueError, match="public decoded choices"):
        manager._encode_force_params(slots, {"scale": "mm"})


def test_n4m_force_params_enqueue_is_fail_closed(monkeypatch):
    _, slots, _, _ = _compile_contract(monkeypatch)
    manager = N4MFinetuneManager()

    class Recorder:
        payload = None

        def enqueue(self, payload):
            self.payload = payload

    recorder = Recorder()
    manager._enqueue_force_params(recorder, slots, {"scale": "STD"})
    assert recorder.payload == {"scale": "std"}

    with pytest.raises(NotImplementedError, match="optimizer.enqueue"):
        manager._enqueue_force_params(object(), slots, {"scale": "STD"})

    class Broken:
        def enqueue(self, payload):
            raise AttributeError("old n4m binding")

    with pytest.raises(RuntimeError, match="fail-closed"):
        manager._enqueue_force_params(Broken(), slots, {"scale": "STD"})


def test_n4m_grid_sampler_uses_native_grid_enum(monkeypatch):
    class FakeSampler:
        @classmethod
        def __class_getitem__(cls, name):
            if name != "GRID":
                raise KeyError(name)
            return "native-grid"

    monkeypatch.setattr(n4m_engine, "Sampler", FakeSampler, raising=False)

    assert N4MFinetuneManager()._native_sampler("grid") == "native-grid"


def test_n4m_grid_sampler_fails_closed_when_native_enum_is_missing(monkeypatch):
    class FakeSampler:
        @classmethod
        def __class_getitem__(cls, name):
            raise KeyError(name)

    monkeypatch.setattr(n4m_engine, "Sampler", FakeSampler, raising=False)

    with pytest.raises(NotImplementedError, match="Sampler.GRID"):
        N4MFinetuneManager()._native_sampler("grid")
