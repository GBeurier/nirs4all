"""Tests for run-local DAG-ML resource declarations."""

from __future__ import annotations

import pytest

from nirs4all.pipeline.dagml.resources import (
    DagMLExecutionResources,
    bind_execution_resources,
    current_execution_resources,
    normalize_execution_resources,
    reset_execution_resources,
)


def test_resources_are_normalized_to_closed_contract() -> None:
    resources = normalize_execution_resources(3, ["cuda:0", "cuda:1"])

    assert resources.to_contract() == {
        "cpu_threads": 3,
        "gpu_devices": ["cuda:0", "cuda:1"],
    }


@pytest.mark.parametrize(
    ("cpu_threads", "gpu_devices", "message"),
    [
        (0, (), "positive integer"),
        (True, (), "positive integer"),
        (1, ["cuda:0", "cuda:0"], "duplicates"),
        (1, ["cuda:1", "cuda:0"], "sorted"),
        (1, [""], "non-empty"),
        (1, ["gpu0"], "canonical CUDA"),
    ],
)
def test_invalid_resources_fail_before_execution(
    cpu_threads: object,
    gpu_devices: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        normalize_execution_resources(cpu_threads, gpu_devices)


def test_resource_binding_is_scoped_and_restored() -> None:
    before = current_execution_resources()
    selected = DagMLExecutionResources(cpu_threads=2, gpu_devices=("cuda:0",))

    token = bind_execution_resources(selected)
    try:
        assert current_execution_resources() == selected
    finally:
        reset_execution_resources(token)

    assert current_execution_resources() == before
