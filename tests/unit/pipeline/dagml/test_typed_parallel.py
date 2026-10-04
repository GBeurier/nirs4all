"""Fail closed on real build evidence before native typed campaign callbacks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nirs4all.pipeline.dagml import typed_parallel
from nirs4all.pipeline.dagml.tuning_contracts import DagMLTuningSpec


@dataclass(frozen=True)
class _Capabilities:
    schema_version: int = 1
    blas: bool = False
    openmp: bool = False
    cuda: bool = False

    @property
    def sequential_cpu(self) -> bool:
        return not (self.blas or self.openmp or self.cuda)


def _spec(**options: Any) -> DagMLTuningSpec:
    return DagMLTuningSpec(engine="n4m", space={"model.alpha": [0.1, 1.0]}, sampler="random", n_jobs=2, **options)


def _install(monkeypatch: pytest.MonkeyPatch, evidence: Any) -> list[str]:
    calls: list[str] = []

    def read() -> Any:
        calls.append("actual-build")
        if isinstance(evidence, BaseException):
            raise evidence
        return evidence

    api = SimpleNamespace(BuildCapabilities=_Capabilities, build_capabilities=read)
    monkeypatch.setattr(typed_parallel.importlib, "import_module", lambda name: api if name == "n4m" else pytest.fail(name))
    return calls


@pytest.mark.parametrize("workers", [2, 3, 4])
def test_parallel_profile_uses_actual_flags_and_exact_resources(monkeypatch: pytest.MonkeyPatch, workers: int) -> None:
    calls = _install(monkeypatch, _Capabilities())
    spec = _spec()
    spec = replace(spec, n_jobs=workers)
    assert typed_parallel.typed_parallel_execution(spec, {"cpu_threads": 1, "gpu_devices": []}) == {
        "schema_version": 1,
        "profile": "methods_sequential_cpu_v1",
        "workers": workers,
        "cpu_threads": 1,
        "gpu_devices": [],
        "methods_build": {"schema_version": 1, "blas": False, "openmp": False, "cuda": False},
    }
    assert calls == ["actual-build"]


def test_serial_does_not_read_capabilities_or_add_parallel_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(typed_parallel.importlib, "import_module", lambda name: pytest.fail("serial read parallel capabilities"))
    spec = replace(_spec(), n_jobs=1, sampler="tpe")
    before = spec.to_dict()
    assert "n_jobs" not in before and "parallel_execution" not in before
    assert typed_parallel.typed_parallel_execution(spec, {"cpu_threads": 3}) is None
    assert spec.to_dict() == before


@pytest.mark.parametrize("mutation", ["workers", "auto_workers", "sampler", "pruning", "threads", "gpu"])
def test_unsupported_parallel_controls_fail_before_capability_read(monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    monkeypatch.setattr(typed_parallel.importlib, "import_module", lambda name: pytest.fail("invalid controls read native capability"))
    spec, options = _spec(), {}
    if mutation == "workers":
        spec = replace(spec, n_jobs=5)
    elif mutation == "auto_workers":
        spec = replace(spec, n_jobs=-1)
    elif mutation == "sampler":
        spec = replace(spec, sampler="tpe")
    elif mutation == "pruning":
        spec = replace(spec, pruner="median")
    elif mutation == "threads":
        options["cpu_threads"] = 2
    else:
        options["gpu_devices"] = ["cuda:0"]
    with pytest.raises(ValueError, match="parallel typed"):
        typed_parallel.typed_parallel_execution(spec, options)


@pytest.mark.parametrize(
    "evidence",
    [
        _Capabilities(blas=True),
        _Capabilities(openmp=True),
        _Capabilities(cuda=True),
        _Capabilities(schema_version=2),
        _Capabilities(blas=0),
        _Capabilities(cuda="false"),
        SimpleNamespace(schema_version=1, blas=False, openmp=False, cuda=False, sequential_cpu=True),
        RuntimeError("missing compiled stamp"),
    ],
)
def test_accelerated_or_malformed_build_cannot_become_cpu_profile(monkeypatch: pytest.MonkeyPatch, evidence: Any) -> None:
    calls = _install(monkeypatch, evidence)
    with pytest.raises(RuntimeError):
        typed_parallel.typed_parallel_execution(_spec(), {})
    assert calls == ["actual-build"]


def test_missing_public_build_evidence_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(typed_parallel.importlib, "import_module", lambda name: SimpleNamespace())
    with pytest.raises(RuntimeError, match="unavailable"):
        typed_parallel.typed_parallel_execution(_spec(), {})


def test_closed_candidate_audit_separates_owners_and_freezes_events() -> None:
    raw = SimpleNamespace(controller_id="controller:methods.python.multimodal", closed=True, audit=[{"operation": "fit", "node": "raw"}, {"operation": "dispose"}])
    meta = SimpleNamespace(controller_id="controller:methods.python.regression", closed=True, audit=[{"operation": "fit", "node": "meta"}, {"operation": "dispose"}])
    controller = SimpleNamespace(raw=raw, meta=meta, closed=True)
    audit = typed_parallel.closed_candidate_audit(controller, trial_index=3, recipe_id="recipe:late")
    assert audit["trial_index"] == 3 and audit["recipe_id"] == "recipe:late"
    assert [owner["controller_id"] for owner in audit["owners"]] == [raw.controller_id, meta.controller_id]
    assert all(owner["closed"] for owner in audit["owners"])
    raw.audit[0]["node"] = "changed"
    assert audit["owners"][0]["events"][0]["node"] == "raw"
    controller.closed = False
    with pytest.raises(RuntimeError, match="closed"):
        typed_parallel.closed_candidate_audit(controller, trial_index=3, recipe_id="recipe:late")


@pytest.mark.parametrize("family", ["typed", "topology"])
def test_missing_build_evidence_stops_both_public_routes_before_optimizer_or_catalogue(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, family: str) -> None:
    import dag_ml

    from nirs4all.pipeline.dagml.host_search_checkpoint import HostSearchOptimizer
    from nirs4all.pipeline.dagml.structural_tuning import _run_structural_tuning
    from tests.unit.pipeline.dagml import test_structural_multimodal, test_structural_topology

    example = test_structural_multimodal.example if family == "typed" else test_structural_topology.example
    original = typed_parallel.importlib.import_module
    monkeypatch.setattr(typed_parallel.importlib, "import_module", lambda name: SimpleNamespace() if name == "n4m" else original(name))
    monkeypatch.setattr(HostSearchOptimizer, "__init__", lambda *a, **k: pytest.fail("missing build evidence constructed optimizer"))
    catalogue = "prepare_host_hpo_structural_catalogue" if family == "typed" else "prepare_host_hpo_topology_catalogue"
    monkeypatch.setattr(dag_ml, catalogue, lambda *a, **k: pytest.fail("missing build evidence reached native catalogue"))
    tuning = {**example.make_tuning(tmp_path / "study"), "n_jobs": 2}
    with pytest.raises(RuntimeError, match="unavailable"):
        _run_structural_tuning(example.make_pipeline(), example.make_dataset(), tuning, run_options={})
