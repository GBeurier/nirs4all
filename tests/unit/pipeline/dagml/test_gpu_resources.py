"""Behavior tests for GPU allocation on DAG-ML host model tasks."""

from __future__ import annotations

import pytest

from nirs4all.pipeline.dagml.node_runner import _framework_name, _gpu_device_scope


class _DeclaredTorchEstimator:
    framework = "pytorch"


def test_framework_detection_honors_explicit_declaration() -> None:
    assert _framework_name(_DeclaredTorchEstimator()) == "pytorch"


@pytest.mark.parametrize("devices", [["cuda:0", "cuda:1"], ["gpu:0"]])
def test_gpu_scope_rejects_ambiguous_or_non_cuda_allocations(devices: list[str]) -> None:
    task = {"resources": {"cpu_threads": 1, "gpu_devices": devices}}

    with pytest.raises(ValueError), _gpu_device_scope(task, object()):
        pass


def test_gpu_scope_runs_real_pytorch_operation_on_requested_device() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA-capable PyTorch runtime")
    task = {"resources": {"cpu_threads": 1, "gpu_devices": ["cuda:0"]}}
    model = torch.nn.Linear(2, 1)

    with _gpu_device_scope(task, model):
        value = torch.ones(2, device="cuda") @ torch.ones(2, device="cuda")

    assert value.device.type == "cuda"
    assert value.device.index == 0
