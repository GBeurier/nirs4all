"""Regression coverage for audited workspace and dataset CLI contracts."""

import argparse
import json
from unittest.mock import Mock

import pytest

from nirs4all.cli.commands import dataset as dataset_cli
from nirs4all.cli.commands import workspace as workspace_cli
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


def _args(root, command, *options):
    parser = argparse.ArgumentParser()
    workspace_cli.add_workspace_commands(parser.add_subparsers())
    return parser.parse_args(["workspace", *command.split(), "--workspace", str(root), *options])


def _messages(log):
    return "\n".join(str(call.args[0]) for call in log.info.call_args_list)


@pytest.fixture
def workspace(tmp_path):
    with WorkspaceStore(tmp_path) as store:
        run = store.begin_run("CLI audit", {}, [])
        pipeline = store.begin_pipeline(run, "CLI", [], [], "dataset", "hash")
        chain = store.save_chain(pipeline, [], 0, "classifier", "", "per_fold", {}, {})
        for name, test, train, val, partition in (("LowModel", .2, .8, .1, "val"),
                                                  ("HighModel", .9, .3, .7, "val"),
                                                  ("NullModel", None, .5, .4, "val"),
                                                  ("HeldOutModel", .99, .1, .1, "test")):
            store.save_prediction(pipeline, chain, "dataset", name, name, "fold_0", partition,
                                  val, test, train, "balanced_accuracy", "classification", 4, 2, {}, {}, None, None, 0, 0)
    return tmp_path


@pytest.mark.parametrize("metric,first,second", [("balanced_accuracy", "HighModel", "LowModel"),
                                               ("r2", "HighModel", "LowModel"), ("rmse", "LowModel", "HighModel")])
def test_query_best_uses_stored_metric_direction(workspace, monkeypatch, metric, first, second):
    with WorkspaceStore(workspace) as store:
        store._ensure_open().execute("UPDATE predictions SET metric = ?", [metric])
    log = Mock()
    monkeypatch.setattr(workspace_cli, "logger", log)
    args = _args(workspace, "query-best")
    assert args.ascending is None
    args.func(args)
    output = _messages(log)
    assert output.index(first) < output.index(second) < output.index("NullModel")
    assert "HeldOutModel" not in output


def test_query_best_direction_partition_and_mixed_metric_selection(workspace, monkeypatch):
    log = Mock()
    monkeypatch.setattr(workspace_cli, "logger", log)
    args = _args(workspace, "query-best", "--ascending")
    args.func(args)
    assert _messages(log).index("LowModel") < _messages(log).index("HighModel")
    log.reset_mock()
    args = _args(workspace, "query-best", "--descending", "--partition", "test")
    assert args.ascending is False
    args.func(args)
    assert "HeldOutModel" in _messages(log) and "HighModel" not in _messages(log)
    with WorkspaceStore(workspace) as store:
        store._ensure_open().execute("UPDATE predictions SET metric = 'rmse' WHERE model_name = 'LowModel'")
    args = _args(workspace, "query-best")
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert exc.value.code == 1
    assert "--evaluation-metric" in str(log.error.call_args)
    log.reset_mock()
    args = _args(workspace, "query-best", "--evaluation-metric", "balanced_accuracy")
    args.func(args)
    assert "HighModel" in _messages(log) and "LowModel" not in _messages(log)


@pytest.mark.parametrize("options,expected", [(('--test-score', '.5'), 2), (('--train-score', '.6'), 1),
                                             (('--val-score', '.6'), 1), (('--test-score', '0'), 3),
                                             (('--test-score', '.5', '--train-score', '.6'), 0)])
def test_workspace_filters_apply_every_threshold(workspace, monkeypatch, options, expected):
    log = Mock()
    monkeypatch.setattr(workspace_cli, "logger", log)
    args = _args(workspace, "filter", *options)
    args.func(args)
    assert f"Found {expected} predictions matching criteria" in _messages(log)


@pytest.mark.parametrize("column,mean", [("train_score", "0.425"), ("val_score", "0.325"), ("test_score", "0.696667")])
def test_workspace_stats_use_selected_score_column(workspace, monkeypatch, column, mean):
    log = Mock()
    monkeypatch.setattr(workspace_cli, "logger", log)
    args = _args(workspace, "stats", "--metric", column)
    args.func(args)
    assert f"Score statistics for {column}" in _messages(log)
    assert mean in _messages(log)


@pytest.mark.parametrize("command", ["list-runs", "query-best", "filter", "stats", "list-library",
                                    "tuning list", "conformal list", "robustness list"])
def test_inspection_rejects_existing_non_workspace_without_writes(tmp_path, command):
    marker = tmp_path / "notes.txt"
    marker.write_text("preserve")
    before = set(tmp_path.iterdir())
    args = _args(tmp_path, command)
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert exc.value.code == 1
    assert set(tmp_path.iterdir()) == before
    assert marker.read_text() == "preserve"


@pytest.mark.parametrize("command", ["list-runs", "query-best", "filter", "stats", "list-library",
                                    "tuning list", "conformal list", "robustness list"])
def test_inspection_opens_current_store_without_initialization(workspace, monkeypatch, command):
    from nirs4all.pipeline.storage import workspace_store

    schema = Mock(side_effect=AssertionError("inspection must not initialize or migrate"))
    monkeypatch.setattr(workspace_store, "create_schema", schema)
    args = _args(workspace, command)
    args.func(args)
    schema.assert_not_called()
    assert not (workspace / "library").exists()


@pytest.mark.parametrize("format", ["json", "text"])
def test_dataset_warnings_preserve_code_message_and_field(tmp_path, capsys, format):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"train_x": "missing_train.csv", "train_y": "missing_targets.csv"}))
    parser = argparse.ArgumentParser()
    dataset_cli.add_dataset_commands(parser.add_subparsers())
    args = parser.parse_args(["dataset", "validate", str(config), "--format", format])
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "E204" not in output and "encoding issue" not in output
    assert "missing_train.csv" in output and "missing_targets.csv" in output
    if format == "json":
        report = json.loads(output)
        assert report["is_valid"] and report["warning_count"] == 2
        assert all(message["severity"] == "warning" for message in report["messages"])
        assert {message["context"]["field"] for message in report["messages"]} == {"train_x", "train_y"}
        assert all(message["code"] == "FILE_NOT_FOUND" for message in report["messages"])
