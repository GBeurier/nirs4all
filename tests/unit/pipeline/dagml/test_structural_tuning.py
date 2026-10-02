"""Boundary contracts for structural HPO; numerical proof lives in public tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.dagml import structural_tuning as structural
from nirs4all.pipeline.dagml import tuning_adapters as adapters
from nirs4all.pipeline.dagml.host_search_checkpoint import HostSearchOptimizer
from nirs4all.pipeline.dagml.structural_tuning import _prepare_structure, _validate_numeric_space, is_structural_tuning_pipeline, validate_structural_profile
from nirs4all.pipeline.dagml.tuning_contracts import parse_tuning_spec, tcv1_sha256
from nirs4all.pipeline.dagml_bridge import lower_structural_hpo_pipeline


def _pipeline() -> list[Any]:
    return [{"split": GroupKFold(3), "group_by": "batch"}, {"_or_": [None, StandardScaler()]},
            {"model": {"_or_": [Ridge(), PLSRegression(scale=False)]}}]


def _spec(**controls: Any) -> Any:
    return parse_tuning_spec({"engine": "n4m", "sampler": "random", "seed": 17,
                             "space": {"model.alpha": [0.1, 1.0], "model.n_components": [1, 2]}, **controls})


def _catalogue() -> dict[str, Any]:
    # Adapter-only fixture. Public integration independently compiles and checks
    # actual native VariantIds, GraphSpecs and full catalogue fingerprints.
    return {"selector_path": "__recipe__", "entries": [
        {"recipe_id": "ridge-raw", "parameter_bindings": {"model.alpha": {"node_id": "ridge", "param_path": "alpha"}}},
        {"recipe_id": "ridge-scaled", "parameter_bindings": {"model.alpha": {"node_id": "ridge-scaled", "param_path": "alpha"}}},
        {"recipe_id": "pls-raw", "parameter_bindings": {"model.n_components": {"node_id": "pls", "param_path": "n_components"}}},
        {"recipe_id": "pls-scaled", "parameter_bindings": {"model.n_components": {"node_id": "pls-scaled", "param_path": "n_components"}}},
    ]}


class _Space:
    """Observe ABI calls only; no optimizer or numerical candidate is simulated."""

    def __init__(self) -> None:
        self.parameters: list[tuple[Any, ...]] = []
        self.constraints: list[tuple[Any, ...]] = []

    def add_int(self, *args: Any, **kwargs: Any) -> None:
        self.parameters.append(("int", args, kwargs))

    def add_float(self, *args: Any, **kwargs: Any) -> None:
        self.parameters.append(("float", args, kwargs))

    def add_categorical(self, *args: Any) -> None:
        self.parameters.append(("categorical", *args))

    def add_constraint(self, kind: Any, refs: list[str], labels: list[str]) -> None:
        self.constraints.append((kind, refs, labels))


_API = SimpleNamespace(SearchSpace=_Space, ConstraintKind=SimpleNamespace(CONDITION_IN="condition_in"))


def test_declared_two_sites_lower_to_native_generator_without_host_product() -> None:
    pipeline = _pipeline()
    before = copy.deepcopy(pipeline)
    steps, _splitter = validate_structural_profile(pipeline)
    dsl = lower_structural_hpo_pipeline(steps)
    generator = dsl["pipeline"][0]
    assert generator["kind"] == "generator" and generator["mode"] == "cartesian"
    assert [len(stage["branches"]) for stage in generator["stages"]] == [2, 2]
    assert generator["stages"][0]["branches"][0]["steps"] == []
    assert "StandardScaler" in generator["stages"][0]["branches"][1]["steps"][0]["operator"]["class"]
    models = [branch["steps"][0] for branch in generator["stages"][1]["branches"]]
    assert models[0]["params"] == before[2]["model"]["_or_"][0].get_params()
    assert models[1]["params"] == before[2]["model"]["_or_"][1].get_params()
    assert pipeline[1]["_or_"][0] is None and not hasattr(pipeline[1]["_or_"][1], "mean_")
    assert "entries" not in dsl and "variant_id" not in dsl


def test_alternative_order_is_preserved_as_a_declaration() -> None:
    pipeline = _pipeline()
    pipeline[1]["_or_"].reverse()
    pipeline[2]["model"]["_or_"].reverse()
    steps, _splitter = validate_structural_profile(pipeline)
    stages = lower_structural_hpo_pipeline(steps)["pipeline"][0]["stages"]
    assert stages[0]["branches"][1]["steps"] == []
    assert "PLSRegression" in stages[1]["branches"][0]["steps"][0]["operator"]


@pytest.mark.parametrize("mutation", [
    lambda p: p.__setitem__(0, {"split": KFold(3), "group_by": "batch"}),
    lambda p: p.__setitem__(0, GroupKFold(3)),
    lambda p: p.__setitem__(1, {"_or_": [None, MinMaxScaler()]}),
    lambda p: p.__setitem__(1, {"_or_": [StandardScaler(), StandardScaler()]}),
    lambda p: p[2]["model"].__setitem__("_or_", [Ridge(), PLSRegression(scale=True)]),
    lambda p: p[2]["model"].__setitem__("_or_", [Ridge(positive=True), PLSRegression(scale=False)]),
    lambda p: p[2].__setitem__("train_params", {"sample_weight": True}),
    lambda p: p.append(StandardScaler()),
])
def test_unsupported_declarations_refused_without_native_handles(mutation: Any) -> None:
    pipeline = _pipeline()
    mutation(pipeline)
    with pytest.raises(ValueError, match="(?i)(structural|groupkfold|scaler|pls|ridge)"):
        validate_structural_profile(pipeline)


def test_fixed_pipeline_dispatch_remains_separate() -> None:
    assert not is_structural_tuning_pipeline([GroupKFold(3), StandardScaler(), {"model": Ridge()}])
    assert is_structural_tuning_pipeline(_pipeline())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_constructor_identity_rejects_nonfinite_or_repr_only_values(value: Any) -> None:
    pipeline = _pipeline()
    pipeline[2]["model"]["_or_"][0].set_params(alpha=value)
    with pytest.raises(Exception, match="(?i)(finite|json|fingerprint)"):
        validate_structural_profile(pipeline)


@pytest.mark.parametrize("path,value", [
    ("model.n_components", [0, 2]), ("model.n_components", [True, 2]),
    ("model.n_components", {"type": "float", "low": 1, "high": 2}),
    ("model.n_components", {"type": "int", "low": 1, "high": 2, "step": 1.5}),
    ("model.n_components", ("float", 1, 2)), ("model.n_components", [1, 4]),
    ("model.alpha", [-0.1, 1.0]), ("model.alpha", ["one", "two"]),
])
def test_axis_semantics_refuse_invalid_estimator_inputs(path: str, value: Any) -> None:
    declaration: dict[str, Any] = {"engine": "n4m", "space": {"model.alpha": [0.1, 1.0], "model.n_components": [1, 2]}}
    declaration["space"][path] = value
    with pytest.raises(ValueError):
        _validate_numeric_space(parse_tuning_spec(declaration), 3)


def test_public_aliases_normalize_once_and_private_selector_is_not_public() -> None:
    spec = parse_tuning_spec({"engine": "n4m", "space": {"model__alpha": [0.1], "model__n_components": [2]}})
    _validate_numeric_space(spec, 3)
    assert set(spec.space) == {"model.alpha", "model.n_components"}
    with pytest.raises(ValueError, match="(?i)(requires only|path segment)"):
        _validate_numeric_space(_spec(space={"__recipe__": ["ridge"], "model.alpha": [0.1], "model.n_components": [2]}), 3)


@pytest.mark.parametrize("controls,options,match", [
    ({"engine": "optuna"}, {}, "requires n4m"),
    ({"force_params": {"model.alpha": 0.1}}, {}, "force_params"),
    ({"n_jobs": 2, "sampler": "tpe"}, {}, "parallel"),
    ({"n_jobs": 2, "pruner": "median"}, {}, "parallel"),
    ({"seed": -1}, {}, "unsigned 64-bit"),
    ({}, {"refit": False}, "winner refit"),
    ({}, {"session": object()}, "standalone"),
    ({}, {"cache": True}, "candidate-local"),
    ({}, {"unrecognized": True}, "unsupported structural run"),
])
def test_execution_refusals_precede_native_preparation(controls: dict[str, Any], options: dict[str, Any], match: str) -> None:
    tuning = {"engine": "n4m", "sampler": "random", "space": {"model.alpha": [0.1], "model.n_components": [2]}, **controls}
    with pytest.raises(ValueError, match=match):
        _prepare_structure(_pipeline(), object(), tuning, options)


@pytest.mark.parametrize("changed_partition", ["train", "test"])
def test_real_dense_dataset_seals_only_training_features(changed_partition: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def dataset(changed: bool) -> SpectroDataset:
        values = np.arange(54, dtype=np.float32).reshape(18, 3)
        if changed:
            values[0 if changed_partition == "train" else 12, 0] += 0.25
        spectro = SpectroDataset("structural-pool-identity")
        spectro.add_samples(values[:12], {"partition": "train"})
        spectro.add_samples(values[12:], {"partition": "test"})
        spectro.add_targets(np.linspace(0.2, 3.7, 18))
        spectro.add_metadata(np.repeat(["a", "b", "c", "external"], [4, 4, 4, 6])[:, None], headers=["batch"])
        spectro.set_task_type("regression")
        return spectro

    payloads: list[dict[str, Any]] = []
    real_fingerprint = structural.tcv1_sha256

    def observe_fingerprint(value: Any) -> str:
        if isinstance(value, dict) and set(value) == {"buffers", "targets", "groups"}:
            payloads.append(copy.deepcopy(value))
        return real_fingerprint(value)

    monkeypatch.setattr(structural, "tcv1_sha256", observe_fingerprint)
    declaration: dict[str, Any] = {"engine": "n4m", "sampler": "random", "seed": 17,
                                   "space": {"model.alpha": [0.1, 1.0], "model.n_components": [1, 2]}}
    baseline = _prepare_structure(_pipeline(), dataset(False), declaration, {})
    changed = _prepare_structure(_pipeline(), dataset(True), declaration, {})
    assert len(payloads) == 2
    assert (payloads[1]["buffers"] != payloads[0]["buffers"]) is (changed_partition == "train")
    assert payloads[1]["targets"] == payloads[0]["targets"]
    assert changed["pool"] == baseline["pool"] == list(range(12))
    # The established namespace seals the whole dataset, including held-out
    # content. Preserve that reminting while checking the training buffers.
    assert changed["identity"].fingerprint != baseline["identity"].fingerprint
    for prepared, payload in zip((baseline, changed), payloads, strict=True):
        wire_ids = [item["sample_id"] for item in payload["groups"]]
        assert wire_ids == [prepared["identity"].to_wire(sample) for sample in prepared["pool"]]
        assert [prepared["identity"].to_int(wire) for wire in wire_ids] == prepared["pool"]
        assert len(wire_ids) == len(set(wire_ids))
    assert changed["folds"] == baseline["folds"]
    assert changed["groups"] == baseline["groups"]


def test_conditions_use_native_membership_for_each_allowed_recipe() -> None:
    catalogue = _catalogue()
    space, slots = adapters._make_n4m_space(_API, _spec().space, structural_catalogue=catalogue)
    assert space.constraints == [
        ("condition_in", ["model.alpha", "__recipe__"], ["", "ridge-raw"]),
        ("condition_in", ["model.alpha", "__recipe__"], ["", "ridge-scaled"]),
        ("condition_in", ["model.n_components", "__recipe__"], ["", "pls-raw"]),
        ("condition_in", ["model.n_components", "__recipe__"], ["", "pls-scaled"]),
    ]
    assert slots[-1] == ("__recipe__", "categorical", None)
    fixed, fixed_slots = adapters._make_n4m_space(_API, _spec().space)
    assert fixed.constraints == [] and "__recipe__" not in {slot[0] for slot in fixed_slots}


def test_inactive_placeholder_never_read_as_effective_value() -> None:
    _space, slots = adapters._make_n4m_space(_API, _spec().space, structural_catalogue=_catalogue())

    class Trial:
        def is_active(self, path: str) -> bool:
            return path != "model.n_components"

        def get_category(self, path: str) -> tuple[int, Any]:
            assert path != "model.n_components", "inactive placeholder was read"
            return (0, "ridge-raw") if path == "__recipe__" else (0, 0.1)

    assert adapters._n4m_trial_params(Trial(), slots, active_only=True) == {"model.alpha": 0.1, "__recipe__": "ridge-raw"}


def _record() -> Any:
    return SimpleNamespace(id=0, status="COMPLETE", score=0.5,
                           params={"model.alpha": 0.1, "model.n_components": 0, "__recipe__": "ridge-raw"},
                           param_details={path: SimpleNamespace(active=path != "model.n_components")
                                          for path in ("model.alpha", "model.n_components", "__recipe__")})


def test_rich_trace_preserves_placeholders_and_filters_by_attested_activity() -> None:
    _space, slots = adapters._make_n4m_space(_API, _spec().space, structural_catalogue=_catalogue())
    record = _record()
    original = dict(record.params)
    assert adapters._n4m_active_record_params(record, slots) == {"model.alpha": 0.1, "__recipe__": "ridge-raw"}
    assert record.params == original and record.params["model.n_components"] == 0
    record.param_details["model.n_components"].active = "false"
    with pytest.raises(ValueError, match="native booleans"):
        adapters._n4m_active_record_params(record, slots)


def test_fixed_pair_preimage_retains_historical_bytes() -> None:
    owner = HostSearchOptimizer.__new__(HostSearchOptimizer)
    payload = {"checkpoint_fingerprint": "a" * 64, "native_checkpoint": {"trials": []}, "unrelated": 1}
    original = {"checkpoint_fingerprint": "a" * 64, "native_checkpoint": {"trials": []}}
    assert owner._pair_preimage(payload) == original
    assert tcv1_sha256(owner._pair_preimage(payload)) == tcv1_sha256(original)
    payload["structural_binding"] = {"catalogue": _catalogue(), "activation_masks": []}
    assert tcv1_sha256(owner._pair_preimage(payload)) != tcv1_sha256(original)


@pytest.mark.parametrize("tamper", ["catalogue", "activity", "fixed"])
def test_resume_refuses_tampered_structure_and_closes_loaded_native_owner(
    tamper: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    spec = _spec(storage=tmp_path.as_uri(), study_name="structural", resume=True)
    path = adapters._n4m_checkpoint_path(spec)
    assert path is not None
    record = _record()
    catalogue = _catalogue()
    saved: dict[str, Any] = {"catalogue": copy.deepcopy(catalogue), "activation_masks": [{"trial_index": 0, "active_paths": ["__recipe__", "model.alpha"]}]}
    if tamper == "catalogue":
        saved["catalogue"]["entries"].reverse()
    if tamper == "activity":
        saved["activation_masks"][0]["active_paths"].append("model.n_components")
    payload = {"checkpoint_fingerprint": "a" * 64, "native_checkpoint": {"trials": []}, "structural_binding": saved}
    payload["pair_fingerprint"] = tcv1_sha256({key: payload[key] for key in ("checkpoint_fingerprint", "native_checkpoint", "structural_binding")})
    path.write_text(json.dumps(payload), encoding="utf-8")
    original_bytes = path.read_bytes()
    loaded: list[Any] = []

    class Owner:
        closed = False

        def configuration_matches(self, expected: Any) -> bool:
            return True

        def get_trials(self) -> list[Any]:
            return [record]

        def close(self) -> None:
            self.closed = True

    def load(*args: Any) -> Any:
        owner = Owner()
        loaded.append(owner)
        return owner

    api = SimpleNamespace(**vars(_API), Sampler=SimpleNamespace(RANDOM="random"), Direction=SimpleNamespace(MINIMIZE="minimize"),
                          Optimizer=lambda space, **options: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(adapters, "_import_n4m_optimizer", lambda: api)
    monkeypatch.setattr(adapters, "_load_n4m_optimizer_checkpoint", load)
    with pytest.raises(ValueError, match="(?i)(structural|activation)"):
        HostSearchOptimizer(spec, n_folds=3, structural_catalogue=None if tamper == "fixed" else catalogue)
    assert path.read_bytes() == original_bytes
    assert all(owner.closed for owner in loaded)
    assert bool(loaded) == (tamper == "activity")


@pytest.mark.parametrize("outcome", ["match", "mismatch", "native_failure", "interrupted", "missing", "non_boolean"])
def test_resume_native_contract_precedes_history_and_closes_temporary_owner(
    outcome: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # Ownership/API contract only. Public integration creates and substitutes
    # genuine valid N4MOPT states to prove native configuration comparisons.
    spec = _spec(storage=tmp_path.as_uri(), study_name="configuration", resume=True, n_trials=128)
    catalogue = _catalogue()
    path = adapters._n4m_checkpoint_path(spec)
    assert path is not None
    payload = {"checkpoint_fingerprint": "a" * 64, "native_checkpoint": {"trials": []},
               "structural_binding": {"catalogue": catalogue, "activation_masks": []}}
    payload["pair_fingerprint"] = tcv1_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    original = path.read_bytes()
    expected_owners: list[Any] = []

    class Expected:
        def __init__(self, space: Any, **options: Any) -> None:
            assert options == {"sampler": "random", "direction": "minimize", "seed": 17}
            assert space.constraints
            self.closed = False
            expected_owners.append(self)

        def close(self) -> None:
            self.closed = True

    class Loaded:
        checked = False
        closed = False

        def configuration_matches(self, expected: Any) -> Any:
            self.checked = True
            assert expected in expected_owners and not expected.closed
            if outcome == "native_failure":
                raise RuntimeError("native comparison failed")
            if outcome == "interrupted":
                raise KeyboardInterrupt("native comparison interrupted")
            return {"match": True, "mismatch": False, "non_boolean": 1}[outcome]

        def get_trials(self) -> list[Any]:
            assert self.checked, "history was read before full native configuration validation"
            return []

        def ask(self) -> Any:
            raise AssertionError("resume validation must never ask a trial")

        def close(self) -> None:
            self.closed = True

    loaded = Loaded()
    if outcome == "missing":
        monkeypatch.setattr(loaded, "configuration_matches", None)
    api = SimpleNamespace(**vars(_API), Sampler=SimpleNamespace(RANDOM="random"), Direction=SimpleNamespace(MINIMIZE="minimize"), Optimizer=Expected)
    monkeypatch.setattr(adapters, "_import_n4m_optimizer", lambda: api)
    monkeypatch.setattr(adapters, "_load_n4m_optimizer_checkpoint", lambda *args: loaded)
    if outcome == "match":
        owner = HostSearchOptimizer(spec, n_folds=3, structural_catalogue=catalogue)
        assert owner.pending == {} and not loaded.closed
        owner.close()
    else:
        error = {"native_failure": RuntimeError, "interrupted": KeyboardInterrupt}.get(outcome, ValueError)
        with pytest.raises(error, match="(?i)(native comparison|configuration_matches|space or options contract)"):
            HostSearchOptimizer(spec, n_folds=3, structural_catalogue=catalogue)
    assert loaded.closed
    assert all(owner.closed for owner in expected_owners)
    assert len(expected_owners) == (0 if outcome == "missing" else 1)
    assert path.read_bytes() == original


def test_fixed_resume_keeps_previous_binding_and_checkpoint_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    spec = _spec(storage=tmp_path.as_uri(), study_name="fixed", resume=True)
    path = adapters._n4m_checkpoint_path(spec)
    assert path is not None
    payload = {"checkpoint_fingerprint": "a" * 64, "native_checkpoint": {"trials": []}}
    payload["pair_fingerprint"] = tcv1_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    original = path.read_bytes()
    loaded = SimpleNamespace(get_trials=lambda: [], close=lambda: None)
    # The old binding has neither Optimizer.configuration_matches nor its new
    # C ABI symbol. Ordinary fixed-topology resume must not require either.
    monkeypatch.setattr(adapters, "_import_n4m_optimizer", lambda: _API)
    monkeypatch.setattr(adapters, "_load_n4m_optimizer_checkpoint", lambda *args: loaded)
    owner = HostSearchOptimizer(spec, n_folds=3)
    assert owner.pending == {} and owner.resume_checkpoint == {"trials": []}
    owner.close()
    assert path.read_bytes() == original


@pytest.mark.parametrize("path,declaration,value,accepted", [
    ("model.alpha", [0.1, 1.0], 0.1, True),
    ("model.alpha", [0.1, 1.0], 0.5, False),
    ("model.alpha", [0.1, 1.0], True, False),
    ("model.alpha", ("float", 0.1, 1.0), 1.1, False),
    ("model.alpha", (0.1, 1.0), float("nan"), False),
    ("model.alpha", {"type": "float", "low": 0.1, "high": 1.0, "step": 0}, 0.55, True),
    ("model.alpha", {"type": "float", "low": 0.1, "high": 1.0, "step": 0.2}, 0.3, True),
    ("model.alpha", {"type": "float", "low": 0.1, "high": 1.0, "step": 0.2}, 0.4, False),
    ("model.alpha", {"type": "float_log", "low": 0.01, "high": 10.0}, 0.1, True),
    ("model.alpha", {"type": "float", "min": 0.1, "max": 1.0}, float("inf"), False),
    ("model.alpha", {"type": "categorical", "options": {"small": 0.1, "large": 1.0}}, 0.1, True),
    ("model.alpha", {"type": "categorical", "options": {"small": 0.1, "large": 1.0}}, 0.5, False),
    ("model.n_components", [1, 2], 1, True),
    ("model.n_components", [1, 2], True, False),
    ("model.n_components", [1, 2], 1.0, False),
    ("model.n_components", (1, 3), 4, False),
    ("model.n_components", ("int", 1, 3), 2, True),
    ("model.n_components", {"type": "int", "low": 1, "high": 5, "step": 2}, 3, True),
    ("model.n_components", {"type": "int", "low": 1, "high": 5, "step": 2}, 2, False),
])
def test_active_value_domain_checks_ignore_inactive_placeholders(
    path: str, declaration: Any, value: Any, accepted: bool,
) -> None:
    space = {"model.alpha": [0.1, 1.0], "model.n_components": [1, 2], path: declaration}
    owner = HostSearchOptimizer.__new__(HostSearchOptimizer)
    owner.tuning = _spec(space=space)
    owner.structural_catalogue = _catalogue()
    _space, owner.slots = adapters._make_n4m_space(_API, owner.tuning.space, structural_catalogue=owner.structural_catalogue)
    recipe = "ridge-raw" if path == "model.alpha" else "pls-raw"
    params = {"__recipe__": recipe, path: value}
    before = copy.deepcopy(params)
    if accepted:
        owner._validate_structural_params(params)
    else:
        with pytest.raises(ValueError, match=f"outside the declared search domain: {path}"):
            owner._validate_structural_params(params)
    assert params.keys() == before.keys() and params["__recipe__"] == before["__recipe__"]


def test_resumed_active_domain_checked_without_mutating_inactive_trace() -> None:
    owner = HostSearchOptimizer.__new__(HostSearchOptimizer)
    owner.tuning = _spec()
    owner.structural_catalogue = _catalogue()
    _space, owner.slots = adapters._make_n4m_space(_API, owner.tuning.space, structural_catalogue=owner.structural_catalogue)
    record = _record()
    assert owner._record_params(record) == {"__recipe__": "ridge-raw", "model.alpha": 0.1}
    record.params["model.alpha"] = 0.5
    with pytest.raises(ValueError, match="outside the declared search domain: model.alpha"):
        owner._record_params(record)
    assert record.params["model.n_components"] == 0 and record.params["model.alpha"] == 0.5


@pytest.mark.parametrize("pruner,expected", [
    (None, {}),
    ("median", {"pruner": "median", "n_startup_trials": 10, "max_resource": 0, "reduction_factor": 0}),
    ("asha", {"pruner": "asha", "n_startup_trials": 10, "max_resource": 0, "reduction_factor": 0}),
    ("racing", {"pruner": "racing", "n_startup_trials": 10, "max_resource": 0, "reduction_factor": 0}),
    ("hyperband", {"pruner": "hyperband", "n_startup_trials": 10, "max_resource": 3, "reduction_factor": 0}),
])
def test_expected_native_options_keep_pruning_contract_and_exclude_total_budget(pruner: Any, expected: dict[str, Any]) -> None:
    owner = HostSearchOptimizer.__new__(HostSearchOptimizer)
    owner.tuning = _spec(pruner=pruner, n_trials=128)
    owner.api = SimpleNamespace(Sampler=SimpleNamespace(RANDOM="random"), Direction=SimpleNamespace(MINIMIZE="minimize"),
                                Pruner=SimpleNamespace(MEDIAN="median", ASHA="asha", RACING="racing", HYPERBAND="hyperband"))
    assert owner._optimizer_options(n_folds=3) == {"sampler": "random", "direction": "minimize", "seed": 17, **expected}
