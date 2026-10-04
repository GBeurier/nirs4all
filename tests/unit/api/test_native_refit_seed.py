"""Native full-refit admission preserves a parent's seed without substitution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from nirs4all.api.native_archive_training import NativeMethodsArchiveRunResult
from nirs4all.api.native_training import refit_native_methods
from nirs4all.pipeline.dagml.native_client import DagMLNativeCoverageError


@pytest.fixture
def seed_parent(monkeypatch: pytest.MonkeyPatch) -> NativeMethodsArchiveRunResult:
    """A contract shell checks SDK admission, without claiming native training."""
    parent = object.__new__(NativeMethodsArchiveRunResult)
    parent._native_package_contract = {
        "effective_plan": {"controller_manifests": {}, "node_plans": {}, "campaign": {}},
        "output_bindings": [{"target_names": ["target"]}],
    }
    parent._native_pipeline = []
    parent._methods_library_path = "unused-libn4m.so"
    monkeypatch.setattr("nirs4all.api.native_training.DagMLNativeClient", lambda _name: object())
    return parent


def _target_dataset() -> dict[str, Any]:
    return {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([3.0, 4.0]), "sample_ids": ["target.a", "target.b"]}


@pytest.mark.parametrize("campaign", [{}, {"root_seed": None}, {"root_seed": True}, {"root_seed": 1.0}, {"root_seed": "7"}, {"root_seed": -1}, {"root_seed": 2**64}])
def test_invalid_parent_seed_refuses_before_target_lowering(
    campaign: dict[str, Any], seed_parent: NativeMethodsArchiveRunResult, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_parent._native_package_contract["effective_plan"]["campaign"] = campaign
    snapshot = deepcopy(seed_parent._native_package_contract)

    def forbidden_lowering(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("invalid parent seed reached target compilation or native training")

    monkeypatch.setattr("nirs4all.api.native_training.lower_raw_array_training_contracts", forbidden_lowering)
    with pytest.raises(DagMLNativeCoverageError, match="explicit unsigned 64-bit parent campaign.root_seed"):
        refit_native_methods(seed_parent, _target_dataset())
    assert seed_parent._native_package_contract == snapshot


@pytest.mark.parametrize("root_seed", [0, 7, 2**64 - 1])
def test_unsigned_parent_seed_reaches_lowering_unchanged(
    root_seed: int, seed_parent: NativeMethodsArchiveRunResult, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_parent._native_package_contract["effective_plan"]["campaign"] = {"root_seed": root_seed}
    snapshot = deepcopy(seed_parent._native_package_contract)
    received: list[int] = []

    class LoweringBoundaryReached(Exception):
        """Stop at the input boundary; no synthetic successful training result."""

    def record_seed(*_args: Any, **kwargs: Any) -> None:
        received.append(kwargs["seed"])
        raise LoweringBoundaryReached

    monkeypatch.setattr("nirs4all.api.native_training.lower_raw_array_training_contracts", record_seed)
    with pytest.raises(LoweringBoundaryReached):
        refit_native_methods(seed_parent, _target_dataset())
    assert received == [root_seed]
    assert type(received[0]) is int
    assert seed_parent._native_package_contract == snapshot
