"""Tests for PyTorch models."""

import pickle
from types import SimpleNamespace
from typing import Any

import pytest

from nirs4all.utils.backend import TORCH_AVAILABLE


class _RecordingLossRegistry:
    def __init__(self, *, fail: bool = False, value: Any = None) -> None:
        self.bind_calls: list[tuple[dict[str, Any], int]] = []
        self.calls: list[tuple[dict[str, Any], Any, Any, int]] = []
        self.fail = fail
        self.value = value

    def bind_training_loss(
        self,
        task: dict[str, Any],
        *,
        role_index: int,
    ) -> dict[str, Any]:
        self.bind_calls.append((task, role_index))

        def invoke(target: Any, prediction: Any) -> Any:
            if self.fail:
                raise RuntimeError("custom loss failed")
            self.calls.append((task, target, prediction, role_index))
            if self.value is not None:
                return self.value
            value = (prediction - target) ** 2
            mean = getattr(value, "mean", None)
            return mean() if callable(mean) else value

        return {
            "invoke": invoke,
            "required_attestation": {
                "phase": task["phase"],
                "loss_id": "example.loss.squared@1",
            },
        }


def _loss_task(phase: str = "FIT_CV") -> dict[str, Any]:
    return {
        "phase": phase,
        "node_plan": {"training_losses": [{}]},
        "required_loss_attestations": [
            {"phase": phase, "loss_id": "example.loss.squared@1"}
        ],
    }


class TestPyTorchLossResolution:
    """Test loss configuration without importing the optional backend."""

    @staticmethod
    def _fake_nn():
        class MSELoss:
            pass

        class L1Loss:
            pass

        class CrossEntropyLoss:
            pass

        return SimpleNamespace(
            MSELoss=MSELoss,
            L1Loss=L1Loss,
            CrossEntropyLoss=CrossEntropyLoss,
        )

    def test_known_alias_is_resolved(self):
        from nirs4all.controllers.models.torch_model import _resolve_loss_function

        nn = self._fake_nn()

        assert isinstance(_resolve_loss_function("mse", nn), nn.MSELoss)

    def test_torch_class_name_and_default_are_resolved(self):
        from nirs4all.controllers.models.torch_model import _resolve_loss_function

        nn = self._fake_nn()

        assert isinstance(_resolve_loss_function("L1Loss", nn), nn.L1Loss)
        assert isinstance(_resolve_loss_function("MSELoss", nn), nn.MSELoss)

    def test_callable_is_preserved(self):
        from nirs4all.controllers.models.torch_model import _resolve_loss_function

        custom_loss = object()

        assert _resolve_loss_function(custom_loss, self._fake_nn()) is custom_loss

    def test_unknown_loss_name_is_rejected(self):
        from nirs4all.controllers.models.torch_model import _resolve_loss_function

        with pytest.raises(ValueError, match="Unknown PyTorch loss 'not_a_loss'"):
            _resolve_loss_function("not_a_loss", self._fake_nn())

    def test_dagml_execution_adapts_prediction_target_order(self):
        from nirs4all.controllers.models.torch_model import _resolve_loss_function
        from nirs4all.pipeline.dagml import DagMLTrainingLossExecution

        registry = _RecordingLossRegistry()
        execution = DagMLTrainingLossExecution(_loss_task(), registry)
        loss_fn = _resolve_loss_function(execution, self._fake_nn())

        assert loss_fn(3.0, 1.0) == 4.0

        assert len(registry.bind_calls) == 1
        assert registry.calls[0][1:] == (1.0, 3.0, 0)
        assert execution.loss_attestations == [
            {"phase": "FIT_CV", "loss_id": "example.loss.squared@1"}
        ]

    def test_dagml_execution_is_process_local_and_attests_only_after_success(self):
        from nirs4all.pipeline.dagml import DagMLTrainingLossExecution

        execution = DagMLTrainingLossExecution(
            _loss_task("REFIT"), _RecordingLossRegistry(fail=True)
        )

        assert execution.loss_attestations == []
        with pytest.raises(RuntimeError, match="custom loss failed"):
            execution(1.0, 2.0)
        assert execution.loss_attestations == []
        with pytest.raises(TypeError, match="cannot be serialized"):
            pickle.dumps(execution)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
    def test_dagml_execution_rejects_non_finite_scalar_before_attestation(self, value: float):
        from nirs4all.pipeline.dagml import DagMLTrainingLossExecution

        execution = DagMLTrainingLossExecution(
            _loss_task("REFIT"), _RecordingLossRegistry(value=value)
        )

        with pytest.raises(ValueError, match="non-finite scalar"):
            execution(1.0, 2.0)
        assert execution.invocation_count == 0
        assert execution.loss_attestations == []

    @pytest.mark.parametrize(
        ("phase", "role_index", "message"),
        [
            ("PREDICT", 0, "only in FIT_CV or REFIT"),
            ("FIT_CV", -1, "non-negative integer"),
            ("FIT_CV", True, "non-negative integer"),
        ],
    )
    def test_dagml_execution_rejects_invalid_scope(
        self, phase: str, role_index: int, message: str
    ):
        from nirs4all.pipeline.dagml import DagMLTrainingLossExecution

        with pytest.raises(ValueError, match=message):
            DagMLTrainingLossExecution(
                _loss_task(phase), _RecordingLossRegistry(), role_index=role_index
            )

    def test_dagml_execution_rejects_malformed_binding(self):
        from nirs4all.pipeline.dagml import DagMLTrainingLossExecution

        registry = SimpleNamespace(bind_training_loss=lambda *_args, **_kwargs: {})

        with pytest.raises(ValueError, match="lacks invoke or required_attestation"):
            DagMLTrainingLossExecution(_loss_task(), registry)

    @pytest.mark.parametrize(
        ("required_loss_attestations", "error_type", "message"),
        [
            (None, TypeError, "must be a sequence"),
            ("loss@1", TypeError, "must be a sequence"),
            ([], ValueError, "require a loss attestation"),
            ([object()], TypeError, "must be a mapping"),
        ],
    )
    def test_dagml_execution_requires_task_owned_attestation(
        self,
        required_loss_attestations: Any,
        error_type: type[Exception],
        message: str,
    ):
        from nirs4all.pipeline.dagml import DagMLTrainingLossExecution

        task = _loss_task()
        if required_loss_attestations is None:
            task.pop("required_loss_attestations")
        else:
            task["required_loss_attestations"] = required_loss_attestations

        with pytest.raises(error_type, match=message):
            DagMLTrainingLossExecution(task, _RecordingLossRegistry())

    def test_dagml_execution_rejects_attestation_mismatch(self):
        from nirs4all.pipeline.dagml import DagMLTrainingLossExecution

        class _MismatchedRegistry(_RecordingLossRegistry):
            def bind_training_loss(
                self,
                task: dict[str, Any],
                *,
                role_index: int,
            ) -> dict[str, Any]:
                binding = super().bind_training_loss(task, role_index=role_index)
                binding["required_attestation"] = {
                    "phase": task["phase"],
                    "loss_id": "example.loss.other@1",
                }
                return binding

        with pytest.raises(ValueError, match="does not match the NodeTask requirement"):
            DagMLTrainingLossExecution(_loss_task(), _MismatchedRegistry())


@pytest.mark.xdist_group("gpu")
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPyTorchModels:
    """Test suite for PyTorch models."""

    def test_simple_mlp(self):
        import torch
        import torch.nn as nn

        from nirs4all.utils.backend import framework

        @framework('pytorch')
        class SimpleMLP(nn.Module):
            def __init__(self, input_shape):
                super().__init__()
                self.layer = nn.Linear(input_shape[1], 1)
            def forward(self, x):
                return self.layer(x)

        model = SimpleMLP(input_shape=(10, 5))
        x = torch.randn(1, 5)
        y = model(x)
        assert y.shape == (1, 1)

    def test_training_loop_executes_dagml_loss_and_captures_attestation(self):
        import torch
        import torch.nn as nn

        from nirs4all.controllers.models.torch_model import PyTorchModelController
        from nirs4all.pipeline.dagml import DagMLTrainingLossExecution

        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        registry = _RecordingLossRegistry()
        execution = DagMLTrainingLossExecution(_loss_task(), registry)

        trained = PyTorchModelController()._train_model(
            model,
            torch.ones((4, 1)),
            torch.ones((4, 1)),
            optimizer="SGD",
            lr=0.1,
            epochs=1,
            batch_size=2,
            loss=execution,
        )

        assert trained.weight.detach().cpu().item() > 0.0
        assert execution.invocation_count == 2
        assert len(registry.bind_calls) == 1
        assert execution.loss_attestations == [
            {"phase": "FIT_CV", "loss_id": "example.loss.squared@1"}
        ]
