"""Tests for TensorFlow models."""

import io
import os
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nirs4all.utils.backend import TF_AVAILABLE


class _RecordingLossRegistry:
    def __init__(self, implementation: Callable[[Any, Any], Any]) -> None:
        self.bind_calls: list[tuple[dict[str, Any], int]] = []
        self.calls: list[tuple[dict[str, Any], Any, Any, int]] = []
        self.implementation = implementation

    def bind_training_loss(
        self,
        task: dict[str, Any],
        *,
        role_index: int,
    ) -> dict[str, Any]:
        self.bind_calls.append((task, role_index))

        def invoke(target: Any, prediction: Any) -> Any:
            self.calls.append((task, target, prediction, role_index))
            return self.implementation(target, prediction)

        return {
            "invoke": invoke,
            "required_attestation": {
                "phase": task["phase"],
                "loss_id": "example.loss.tensorflow-squared@1",
            },
        }


def _loss_task(phase: str = "FIT_CV") -> dict[str, Any]:
    return {"phase": phase, "node_plan": {"training_losses": [{}]}}


def _train_with_dagml_loss(
    phase: str,
    *,
    nested_compile: bool = False,
) -> tuple[Any, _RecordingLossRegistry, Any]:
    import tensorflow as tf

    from nirs4all.controllers.models.tensorflow_model import TensorFlowModelController
    from nirs4all.core.task_type import TaskType
    from nirs4all.pipeline.dagml import DagMLTrainingLossExecution

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,)),
            tf.keras.layers.Dense(1, use_bias=False, kernel_initializer="zeros"),
        ]
    )
    registry = _RecordingLossRegistry(
        lambda target, prediction: tf.reduce_mean(tf.square(prediction - target))
    )
    execution = DagMLTrainingLossExecution(_loss_task(phase), registry)
    train_params: dict[str, Any] = {
        "task_type": TaskType.REGRESSION,
        "epochs": 1,
        "batch_size": 2,
        "validation_split": 0.0,
        "best_model_memory": False,
        "verbose": 0,
    }
    compile_params = {
        "optimizer": "sgd",
        "learning_rate": 0.1,
        "loss": execution,
        "metrics": [],
    }
    if nested_compile:
        compile_params["run_eagerly"] = True
        train_params["compile"] = compile_params
    else:
        train_params.update(compile_params)

    trained = TensorFlowModelController()._train_model(
        model,
        np.ones((4, 1), dtype=np.float32),
        np.ones((4, 1), dtype=np.float32),
        **train_params,
    )
    return trained, registry, execution


def test_dagml_execution_preserves_keras_target_prediction_order():
    from nirs4all.controllers.models.tensorflow_model import _resolve_loss_function
    from nirs4all.pipeline.dagml import DagMLTrainingLossExecution

    registry = _RecordingLossRegistry(
        lambda target, prediction: (prediction - target) ** 2
    )
    execution = DagMLTrainingLossExecution(_loss_task(), registry)
    loss_fn = _resolve_loss_function(execution)

    assert loss_fn(1.0, 3.0) == 4.0
    assert len(registry.bind_calls) == 1
    assert registry.calls[0][1:] == (1.0, 3.0, 0)
    assert execution.loss_attestations == [
        {"phase": "FIT_CV", "loss_id": "example.loss.tensorflow-squared@1"}
    ]


def test_regular_keras_loss_is_preserved():
    from nirs4all.controllers.models.tensorflow_model import _resolve_loss_function

    custom_loss = object()

    assert _resolve_loss_function(custom_loss) is custom_loss


@pytest.mark.tensorflow
@pytest.mark.xdist_group("tensorflow")
@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
@pytest.mark.parametrize("phase", ["FIT_CV", "REFIT"])
@pytest.mark.parametrize("nested_compile", [False, True], ids=["graph", "nested-eager"])
def test_training_loop_executes_dagml_loss_and_preserves_gradients(
    phase: str,
    nested_compile: bool,
):
    trained, registry, execution = _train_with_dagml_loss(
        phase,
        nested_compile=nested_compile,
    )

    assert trained.layers[-1].get_weights()[0].item() > 0.0
    assert execution.invocation_count >= 1
    assert len(registry.bind_calls) == 1
    assert execution.loss_attestations == [
        {"phase": phase, "loss_id": "example.loss.tensorflow-squared@1"}
    ]


@pytest.mark.tensorflow
@pytest.mark.xdist_group("tensorflow")
@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
def test_production_persistence_reloads_without_process_local_loss(
    tmp_path: Path,
):
    from nirs4all.pipeline.storage.artifacts.artifact_persistence import to_bytes

    trained, _, _ = _train_with_dagml_loss("FIT_CV")

    data, format_name = to_bytes(trained, format_hint="tensorflow")
    assert format_name == "tensorflow_keras"
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        config = archive.read("config.json")
    assert b"example.loss.tensorflow-squared@1" not in config
    assert b"DagMLTrainingLossExecution" not in config

    artifact_path = tmp_path / "model.keras"
    artifact_path.write_bytes(data)
    script = "\n".join(
        [
            "import sys",
            "from pathlib import Path",
            "import numpy as np",
            "from nirs4all.pipeline.storage.artifacts.artifact_persistence import from_bytes",
            "data = Path(sys.argv[1]).read_bytes()",
            "model = from_bytes(data, 'tensorflow_keras')",
            "prediction = model.predict(np.ones((1, 1), dtype=np.float32), verbose=0)",
            "assert prediction.shape == (1, 1)",
        ]
    )
    env = os.environ.copy()
    env.update(
        {
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(artifact_path)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
