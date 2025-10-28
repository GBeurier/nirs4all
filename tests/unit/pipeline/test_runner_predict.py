import json
import os
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from nirs4all.pipeline.runner import PipelineRunner, init_global_random_state
from nirs4all.pipeline.io import SimulationSaver
from nirs4all.pipeline.manifest_manager import ManifestManager
from nirs4all.data.predictions import Predictions
from nirs4all.utils.tab_report_manager import TabReportManager


def _create_runner_with_pipeline(tmp_path: Path, dataset_name: str = "dataset"):
    """Utility to prepare a runner with a persisted pipeline structure."""
    workspace = tmp_path / "workspace"
    runner = PipelineRunner(workspace_path=workspace, save_files=False, enable_tab_reports=False)

    run_dir = runner.runs_dir / f"2024-01-01_{dataset_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pipeline_uid = "0001_mockpipeline"
    pipeline_dir = run_dir / pipeline_uid
    pipeline_dir.mkdir(exist_ok=True)

    pipeline_steps = [{"model": "dummy"}]
    (pipeline_dir / "pipeline.json").write_text(json.dumps({"steps": pipeline_steps}), encoding="utf-8")

    manifest_manager = ManifestManager(run_dir)
    manifest_data = {
        "uid": "uid-123",
        "pipeline_id": pipeline_uid,
        "name": "mock_pipeline",
        "dataset": dataset_name,
        "created_at": "2024-01-01T00:00:00Z",
        "version": "1.0",
        "pipeline": {"steps": pipeline_steps},
        "metadata": {},
        "artifacts": [
            {
                "hash": "sha256:abc",
                "name": "artifact.bin",
                "path": "artifact.bin",
                "format": "pickle",
                "size": 1,
                "saved_at": "2024-01-01T00:00:00Z",
                "step": 0,
            }
        ],
        "predictions": [],
    }
    manifest_manager.save_manifest(pipeline_uid, manifest_data)

    selection_obj = {
        "config_path": f"{dataset_name}/{pipeline_uid}",
        "config_name": "mock_config",
        "model_name": "MockModel",
        "id": "pred-1",
        "step_idx": 0,
        "fold_id": "0",
        "weights": [0.5],
        "y_true": [0.0],
        "y_pred": [0.0],
        "pipeline_uid": pipeline_uid,
    }
    return runner, run_dir, pipeline_steps, selection_obj


def test_init_global_random_state_controls_entropy():
    # Clear any prior hash seed to ensure the function sets it.
    os.environ.pop("PYTHONHASHSEED", None)

    init_global_random_state(123)
    np_val1 = np.random.rand()
    py_val1 = random.random()

    init_global_random_state(123)
    np_val2 = np.random.rand()
    py_val2 = random.random()

    assert np_val1 == np_val2
    assert py_val1 == py_val2
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_prepare_replay_loads_pipeline_and_sets_state(tmp_path):
    runner, run_dir, pipeline_steps, selection_obj = _create_runner_with_pipeline(tmp_path)
    dataset_config = SimpleNamespace(configs=[({}, "dataset")])

    runner.saver = SimulationSaver(run_dir, save_files=False)
    steps = runner.prepare_replay(selection_obj.copy(), dataset_config)

    assert steps == pipeline_steps
    assert runner.config_path == selection_obj["config_path"]
    assert runner.target_model["model_name"] == "MockModel"
    assert "y_pred" not in runner.target_model
    assert runner.model_weights == selection_obj["weights"]
    assert runner.binary_loader is not None
    assert 0 in runner.binary_loader.artifacts_by_step


def test_prepare_replay_requires_pipeline_uid(tmp_path):
    runner, run_dir, _, selection_obj = _create_runner_with_pipeline(tmp_path)
    dataset_config = SimpleNamespace(configs=[({}, "dataset")])

    runner.saver = SimulationSaver(run_dir, save_files=False)
    selection_missing_uid = selection_obj.copy()
    selection_missing_uid.pop("pipeline_uid")

    with pytest.raises(ValueError):
        runner.prepare_replay(selection_missing_uid, dataset_config)


def test_predict_returns_best_prediction(tmp_path, monkeypatch):
    runner, _, _, selection_obj = _create_runner_with_pipeline(tmp_path)

    def fake_run_single(self, steps, config_name, dataset, config_predictions):
        config_predictions.add_prediction(
            dataset_name=dataset.name,
            config_name=self.target_model["config_name"],
            config_path=self.target_model["config_path"],
            pipeline_uid=self.target_model["pipeline_uid"],
            step_idx=self.target_model["step_idx"],
            op_counter=1,
            model_name=self.target_model["model_name"],
            model_classname="MockModel",
            fold_id=self.target_model["fold_id"],
            partition="test",
            y_true=np.array([0.1, 0.2]),
            y_pred=np.array([0.2, 0.3]),
            test_score=0.0,
            val_score=0.0,
            train_score=0.0,
            n_samples=2,
            n_features=1,
        )
        return dataset, {}

    saved_prediction = {}

    def fake_save_predictions_to_csv(*, y_true=None, y_pred=None, filepath="", prefix="", suffix=""):
        saved_prediction["y_pred"] = np.array(y_pred)
        saved_prediction["filepath"] = Path(filepath)

    monkeypatch.setattr(PipelineRunner, "_run_single", fake_run_single)
    monkeypatch.setattr(Predictions, "save_predictions_to_csv", staticmethod(fake_save_predictions_to_csv))

    X = np.random.randn(6, 2)
    y = np.random.randn(6)

    result, run_predictions = runner.predict(selection_obj.copy(), (X, y))

    assert np.allclose(result, np.array([0.2, 0.3]))
    assert run_predictions.num_predictions == 1
    assert saved_prediction["filepath"].name.startswith("Predict_")


def test_predict_all_predictions_returns_dict(tmp_path, monkeypatch):
    runner, _, _, selection_obj = _create_runner_with_pipeline(tmp_path)

    def fake_run_single(self, steps, config_name, dataset, config_predictions):
        config_predictions.add_prediction(
            dataset_name=dataset.name,
            config_name=self.target_model["config_name"],
            config_path=self.target_model["config_path"],
            pipeline_uid=self.target_model["pipeline_uid"],
            step_idx=self.target_model["step_idx"],
            op_counter=1,
            model_name=self.target_model["model_name"],
            model_classname="MockModel",
            fold_id=self.target_model["fold_id"],
            partition="test",
            y_true=np.array([0.1, 0.2]),
            y_pred=np.array([0.2, 0.3]),
            test_score=0.0,
            val_score=0.0,
            train_score=0.0,
            n_samples=2,
            n_features=1,
        )
        return dataset, {}

    save_called = {"value": False}

    def fake_save_predictions_to_csv(*args, **kwargs):  # pragma: no cover - should not be invoked
        save_called["value"] = True

    monkeypatch.setattr(PipelineRunner, "_run_single", fake_run_single)
    monkeypatch.setattr(Predictions, "save_predictions_to_csv", staticmethod(fake_save_predictions_to_csv))

    X = np.random.randn(4, 1)
    y = np.random.randn(4)

    results, run_predictions = runner.predict(selection_obj.copy(), (X, y), all_predictions=True)

    assert run_predictions.num_predictions == 1
    assert list(results.keys()) == ["prediction_dataset"]
    assert save_called["value"] is False


def test_execute_controller_appends_artifacts(tmp_path):
    runner = PipelineRunner(workspace_path=tmp_path / "workspace_exec", save_files=False, enable_tab_reports=False)
    runner.mode = "train"
    runner.pipeline_uid = "0001_pipeline"
    runner.manifest_manager = MagicMock()

    class DummyController:
        def execute(self, step, operator, dataset, context, runner_ref, source, mode, loaded_binaries, prediction_store):
            return (
                {"updated": True},
                [
                    {
                        "hash": "sha",
                        "name": "artifact.bin",
                        "path": "artifact.bin",
                        "format": "pickle",
                        "size": 1,
                        "saved_at": "now",
                        "step": runner_ref.step_number,
                    }
                ],
            )

    dataset = MagicMock()
    context = {"foo": "bar"}

    result = runner._execute_controller(DummyController(), "step", None, dataset, context)

    assert result["updated"] is True
    runner.manifest_manager.append_artifacts.assert_called_once()


def test_run_steps_with_context_list(monkeypatch):
    runner = PipelineRunner(save_files=False, enable_tab_reports=False)
    dataset = MagicMock()
    contexts = [{"idx": 0}, {"idx": 1}]
    steps = ["a", "b"]
    calls = []

    def fake_run_step(self, step, dataset_obj, context, prediction_store=None, *, is_substep=False, propagated_binaries=None):
        context["processed"] = step
        calls.append((step, context["idx"], is_substep))
        return context

    monkeypatch.setattr(PipelineRunner, "run_step", fake_run_step)

    result = runner.run_steps(steps, dataset, contexts, execution="sequential", is_substep=True)

    assert calls == [("a", 0, True), ("b", 1, True)]
    assert result == contexts[-1]


def test_run_step_none_returns_context(tmp_path):
    runner = PipelineRunner(workspace_path=tmp_path / "workspace_none", save_files=False, enable_tab_reports=False)
    dataset = MagicMock()
    context = {"value": 1}

    result = runner.run_step(None, dataset, context)

    assert result is context


def test_select_controller_raises_when_no_match(monkeypatch):
    runner = PipelineRunner(save_files=False, enable_tab_reports=False)
    monkeypatch.setattr("nirs4all.pipeline.runner.CONTROLLER_REGISTRY", [])

    with pytest.raises(TypeError):
        runner._select_controller("unknown")


def test_print_best_predictions_generates_reports(tmp_path, monkeypatch):
    runner = PipelineRunner(workspace_path=tmp_path / "workspace_reports", save_files=True, enable_tab_reports=True)
    runner.saver = SimpleNamespace(base_path=runner.workspace_path, saved=[])

    def fake_save_file(filename, content):
        runner.saver.saved.append((filename, content))
        return runner.saver.base_path / filename

    runner.saver.save_file = fake_save_file

    saved_prediction = {}

    def fake_save_predictions_to_csv(*args, y_true=None, y_pred=None, filepath="", prefix="", suffix="", **kwargs):
        if len(args) >= 3:
            path_arg = args[2]
        else:
            path_arg = filepath
        saved_prediction["filepath"] = Path(path_arg)

    monkeypatch.setattr(TabReportManager, "generate_best_score_tab_report", staticmethod(lambda partitions: ("REPORT", "CSV_DATA")))
    monkeypatch.setattr(Predictions, "save_predictions_to_csv", staticmethod(fake_save_predictions_to_csv))

    base_prediction_kwargs = dict(
        dataset_name="dataset",
        config_name="mock_config",
        config_path="dataset/0001_mockpipeline",
        pipeline_uid="0001_mockpipeline",
        step_idx=0,
        op_counter=1,
        model_name="MockModel",
        model_classname="MockModel",
        fold_id="0",
        n_samples=2,
        n_features=1,
    )

    run_predictions = Predictions()
    run_predictions.add_prediction(
        partition="val",
        y_true=np.array([0.1, 0.2]),
        y_pred=np.array([0.15, 0.25]),
        test_score=0.3,
        val_score=0.4,
        train_score=0.35,
        **base_prediction_kwargs,
    )
    run_predictions.add_prediction(
        partition="test",
        y_true=np.array([0.1, 0.2]),
        y_pred=np.array([0.2, 0.3]),
        test_score=0.3,
        val_score=0.4,
        train_score=0.33,
        **base_prediction_kwargs,
    )

    global_predictions = Predictions()
    global_predictions.add_prediction(
        partition="val",
        y_true=np.array([0.1, 0.2]),
        y_pred=np.array([0.12, 0.22]),
        test_score=0.28,
        val_score=0.5,
        train_score=0.4,
        **base_prediction_kwargs,
    )
    global_predictions.add_prediction(
        partition="test",
        y_true=np.array([0.1, 0.2]),
        y_pred=np.array([0.18, 0.28]),
        test_score=0.25,
        val_score=0.5,
        train_score=0.45,
        **base_prediction_kwargs,
    )

    dataset_stub = SimpleNamespace(name="dataset", is_regression=True)
    dataset_prediction_path = runner.workspace_path / "dataset.json"

    runner.print_best_predictions(run_predictions, global_predictions, dataset_stub, "dataset", str(dataset_prediction_path))

    assert runner.saver.saved, "Expected tab report to be saved"
    assert saved_prediction["filepath"].name.startswith("Best_prediction_")
    assert dataset_prediction_path.exists()
