"""
Unit tests for the nirs4all CLI entrypoint.

Tests cover:
- --version flag
- --help flag
- subcommand help
- error handling wrapper (CLI-04)
- workspace path validation (WS-01)
- config validate command structure
"""

import json
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import jsonschema
import numpy as np
import pytest


def invoke_cli(args):
    """Invoke the nirs4all CLI main() with the given argument list.

    Args:
        args: List of string arguments (excluding the program name).

    Returns:
        Tuple of (exit_code, stdout_output) where exit_code is the integer
        exit code raised by SystemExit, or 0 if the function returned normally.
    """
    from nirs4all.cli.main import main

    with patch("sys.argv", ["nirs4all"] + args):
        try:
            main()
            return 0
        except SystemExit as exc:
            return int(exc.code) if exc.code is not None else 0


class TestCLIVersion:
    """Test --version flag."""

    def test_version_flag(self):
        """Test that --version prints version and exits with 0."""
        exit_code = invoke_cli(["--version"])
        assert exit_code == 0

    def test_version_flag_short(self):
        """Test that --version only works as --version (argparse standard)."""
        # argparse only supports --version, not -v for version
        exit_code = invoke_cli(["--version"])
        assert exit_code == 0


class TestCLIHelp:
    """Test --help flag."""

    def test_help_flag(self):
        """Test that --help exits cleanly with code 0."""
        exit_code = invoke_cli(["--help"])
        assert exit_code == 0

    def test_no_args_shows_help(self):
        """Test that no arguments shows help and exits 0."""
        exit_code = invoke_cli([])
        assert exit_code == 0


class TestCLISubcommandHelp:
    """Test subcommand help messages."""

    def test_workspace_help(self):
        """Test that 'workspace --help' exits cleanly."""
        exit_code = invoke_cli(["workspace", "--help"])
        assert exit_code == 0

    def test_config_help(self):
        """Test that 'config --help' exits cleanly."""
        exit_code = invoke_cli(["config", "--help"])
        assert exit_code == 0

    def test_workspace_init_help(self):
        """Test that 'workspace init --help' exits cleanly."""
        exit_code = invoke_cli(["workspace", "init", "--help"])
        assert exit_code == 0

    def test_config_validate_help(self):
        """Test that 'config validate --help' exits cleanly."""
        exit_code = invoke_cli(["config", "validate", "--help"])
        assert exit_code == 0

    def test_config_schema_help(self):
        """Test that 'config schema --help' exits cleanly."""
        exit_code = invoke_cli(["config", "schema", "--help"])
        assert exit_code == 0

    def test_keyword_registry_help(self):
        """Test that 'keyword-registry --help' exits cleanly."""
        exit_code = invoke_cli(["keyword-registry", "--help"])
        assert exit_code == 0

    def test_robustness_report_help(self):
        """Test that 'robustness-report --help' exits cleanly."""
        exit_code = invoke_cli(["robustness-report", "--help"])
        assert exit_code == 0

    def test_robustness_summary_schema_help(self):
        """Test that 'robustness-summary-schema --help' exits cleanly."""
        exit_code = invoke_cli(["robustness-summary-schema", "--help"])
        assert exit_code == 0

    def test_tuning_summary_schema_help(self):
        """Test that 'tuning-summary-schema --help' exits cleanly."""
        exit_code = invoke_cli(["tuning-summary-schema", "--help"])
        assert exit_code == 0

    def test_tuning_space_help(self):
        """Test that 'tuning-space --help' exits cleanly."""
        exit_code = invoke_cli(["tuning-space", "--help"])
        assert exit_code == 0

    def test_tuning_space_outputs_ordered_search_space_json(self, capsys):
        """CLI exposes the same canonical tuning-space artifact as Python."""
        payload = {
            "engine": "optuna",
            "space": {
                "ridge__alpha": [0.1, 0.2],
                "scale.with_mean": [False],
            },
            "force_params": {"ridge.alpha": 0.1},
        }

        exit_code = invoke_cli(["tuning-space", "--tuning", json.dumps(payload), "--compact"])
        output = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert output["format"] == "nirs4all.tuning.ordered_search_space"
        assert output["parameters"][0]["path"] == "ridge.alpha"
        assert output["parameters"][1]["path"] == "scale.with_mean"
        assert output["force_params"] == [
            {
                "path": "ridge.alpha",
                "segments": ["ridge", "alpha"],
                "value": 0.1,
            }
        ]

    def test_tuning_space_writes_output_file(self, tmp_path):
        """CLI can write the ordered search-space artifact to a file."""
        payload = tmp_path / "tuning.json"
        output = tmp_path / "space.json"
        payload.write_text(json.dumps({"engine": "n4m", "space": {"alpha": [0.2]}}), encoding="utf-8")

        exit_code = invoke_cli(["tuning-space", "--input", str(payload), "--output", str(output)])

        assert exit_code == 0
        artifact = json.loads(output.read_text(encoding="utf-8"))
        assert artifact["parameters"] == [
            {
                "index": 0,
                "path": "alpha",
                "segments": ["alpha"],
                "spec": [0.2],
            }
        ]

    def test_tuning_space_schema_outputs_json_schema(self, capsys):
        """CLI can publish the static ordered search-space JSON Schema."""
        exit_code = invoke_cli(["tuning-space", "--schema", "--compact"])
        output = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert output["$id"] == "https://nirs4all.org/schemas/tuning-ordered-search-space/v1"
        assert output["properties"]["format"]["const"] == "nirs4all.tuning.ordered_search_space"

    def test_tuning_space_schema_writes_output_file(self, tmp_path):
        """CLI can write the ordered search-space JSON Schema to a file."""
        output = tmp_path / "tuning-space.schema.json"

        exit_code = invoke_cli(["tuning-space", "--schema", "--output", str(output)])

        assert exit_code == 0
        artifact = json.loads(output.read_text(encoding="utf-8"))
        assert artifact["$id"] == "https://nirs4all.org/schemas/tuning-ordered-search-space/v1"

    def test_tuning_space_schema_rejects_tuning_payload(self, capsys):
        """Schema mode is static and cannot inspect a tuning payload."""
        exit_code = invoke_cli(["tuning-space", "--schema", "--tuning", "{}"])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "--schema cannot be combined" in captured.err

    def test_tuning_space_rejects_invalid_json(self, capsys):
        """Invalid JSON is rejected before any tuning contract is inspected."""
        exit_code = invoke_cli(["tuning-space", "--tuning", "{bad-json"])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "invalid tuning JSON" in captured.err

    def test_workspace_list_runs_help(self):
        """Test that 'workspace list-runs --help' exits cleanly."""
        exit_code = invoke_cli(["workspace", "list-runs", "--help"])
        assert exit_code == 0

    def test_workspace_inspect_help(self):
        """Test that 'workspace inspect --help' exits cleanly."""
        exit_code = invoke_cli(["workspace", "inspect", "--help"])
        assert exit_code == 0

    def test_workspace_inspect_legacy_artifact_prints_conversion_command(self, monkeypatch, tmp_path):
        """Transition CLI must show users the exact artifact conversion command."""
        from nirs4all.cli.commands import workspace as workspace_commands
        from nirs4all.workspace.compat import build_conversion_command

        messages: list[str] = []

        class CapturingLogger:
            def info(self, message):
                messages.append(str(message))

        monkeypatch.setattr(workspace_commands, "logger", CapturingLogger())
        artifact = tmp_path / "old-export.n4a"
        artifact.write_bytes(b"legacy artifact placeholder")

        exit_code = invoke_cli(["workspace", "inspect", str(artifact)])

        output = "\n".join(messages)
        assert exit_code == 0
        assert "Format: legacy-artifact" in output
        assert "Conversion required: True" in output
        assert build_conversion_command(artifact) in output

    def test_workspace_convert_help(self):
        """Test that 'workspace convert --help' exits cleanly."""
        exit_code = invoke_cli(["workspace", "convert", "--help"])
        assert exit_code == 0

    def test_workspace_tuning_help(self):
        """Test that 'workspace tuning --help' exits cleanly."""
        exit_code = invoke_cli(["workspace", "tuning", "--help"])
        assert exit_code == 0

    def test_workspace_tuning_export_help(self):
        """Test that 'workspace tuning export --help' exits cleanly."""
        exit_code = invoke_cli(["workspace", "tuning", "export", "--help"])
        assert exit_code == 0

    def test_workspace_conformal_help(self):
        """Test that 'workspace conformal --help' exits cleanly."""
        exit_code = invoke_cli(["workspace", "conformal", "--help"])
        assert exit_code == 0

    def test_workspace_conformal_predict_help(self):
        """Test that 'workspace conformal predict --help' exits cleanly."""
        exit_code = invoke_cli(["workspace", "conformal", "predict", "--help"])
        assert exit_code == 0

    def test_workspace_conformal_show_as_predict_result_outputs_provenance_json(self, tmp_path, capsys):
        """CLI can inspect a stored conformal result through the PredictResult payload."""
        import nirs4all

        workspace = tmp_path / "workspace"
        nirs4all.calibrate(
            y_true=[1.0, 2.0, 3.0, 4.0],
            y_pred_calibration=[1.0, 2.0, 3.0, 4.0],
            y_pred=[10.0, 20.0],
            calibration_sample_ids=["c1", "c2", "c3", "c4"],
            prediction_sample_ids=["p1", "p2"],
            coverage=0.8,
            result_metadata={
                "tuning_calibration_source": {
                    "source": "tuning.winner",
                    "score_data_role": "hpo_objective_only",
                    "score_data_used": False,
                }
            },
            workspace_path=workspace,
            workspace_conformal_id="cal-main",
        )

        exit_code = invoke_cli(["workspace", "conformal", "show", "cal-main", "--workspace", str(workspace), "--as-predict-result", "--json"])
        payload = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert payload["y_pred"] == [10.0, 20.0]
        assert payload["sample_ids"] == ["p1", "p2"]
        assert payload["intervals"]["0.8"]["lower"] == [10.0, 20.0]
        assert payload["calibrated_result_fingerprint"]
        assert payload["conformal_guarantee_status"]["status"] == "active"
        assert payload["calibration_replay_source"]["kind"] == "explicit_replayed_arrays"
        assert payload["tuning_calibration_source"] == {
            "source": "tuning.winner",
            "score_data_role": "hpo_objective_only",
            "score_data_used": False,
        }

    def test_workspace_robustness_help(self):
        """Test that 'workspace robustness --help' exits cleanly."""
        exit_code = invoke_cli(["workspace", "robustness", "--help"])
        assert exit_code == 0

    def test_workspace_robustness_list_help(self):
        """Test that 'workspace robustness list --help' exits cleanly."""
        exit_code = invoke_cli(["workspace", "robustness", "list", "--help"])
        assert exit_code == 0

    def test_workspace_robustness_show_help(self):
        """Test that 'workspace robustness show --help' exits cleanly."""
        exit_code = invoke_cli(["workspace", "robustness", "show", "--help"])
        assert exit_code == 0

    def test_workspace_robustness_export_help(self):
        """Test that 'workspace robustness export --help' exits cleanly."""
        exit_code = invoke_cli(["workspace", "robustness", "export", "--help"])
        assert exit_code == 0

    def test_workspace_robustness_from_prediction_help(self):
        """Test that 'workspace robustness from-prediction --help' exits cleanly."""
        exit_code = invoke_cli(["workspace", "robustness", "from-prediction", "--help"])
        assert exit_code == 0

    def test_workspace_robustness_evidence_help(self):
        """Test that 'workspace robustness evidence --help' exits cleanly."""
        exit_code = invoke_cli(["workspace", "robustness", "evidence", "--help"])
        assert exit_code == 0

    def test_workspace_robustness_evidence_outputs_prediction_readiness_json(self, monkeypatch, tmp_path, capsys):
        """CLI reports spectral/OOD replay readiness from stored prediction evidence."""
        from nirs4all.data.predictions import Predictions

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        records = [
            {
                "id": "ready-pred",
                "dataset_name": "wheat",
                "model_name": "PLS",
                "partition": "test",
                "y_pred": np.asarray([1.0, 2.0]),
                "X": np.asarray([[0.1, 0.2], [0.3, 0.4]]),
                "result_metadata": {
                    "robustness_evidence": {
                        "X": "prediction_arrays.X",
                        "predictor_bundle": "artifacts/pls.n4a",
                        "publisher": "test",
                    }
                },
                "calibration_replay_source": {
                    "dataset_backed": False,
                    "kind": "predict_result",
                    "requires_model_replay": False,
                    "route": "PredictResult",
                    "version": 1,
                },
                "tuning_calibration_source": {
                    "score_data_role": "hpo_objective_only",
                    "score_data_used": False,
                    "source": "tuning.winner",
                },
            },
            {
                "id": "missing-pred",
                "dataset_name": "wheat",
                "model_name": "PLS",
                "partition": "test",
                "y_pred": np.asarray([3.0]),
            },
        ]

        class FakePredictions:
            def to_dicts(self, load_arrays=True):
                assert load_arrays is True
                return list(records)

        def fake_from_workspace(path, *, dataset_name=None, load_arrays=True):
            assert Path(path) == workspace
            assert dataset_name == "wheat"
            assert load_arrays is True
            return FakePredictions()

        monkeypatch.setattr(Predictions, "from_workspace", staticmethod(fake_from_workspace))

        exit_code = invoke_cli(["workspace", "robustness", "evidence", "--workspace", str(workspace), "--dataset", "wheat", "--json"])
        payload = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert payload["count"] == 2
        assert payload["ready_count"] == 1
        by_id = {row["prediction_id"]: row for row in payload["predictions"]}
        assert by_id["ready-pred"]["ready"] is True
        assert by_id["ready-pred"]["publisher"] == "test"
        assert by_id["ready-pred"]["calibration_replay_source"]["kind"] == "predict_result"
        assert by_id["ready-pred"]["tuning_calibration_source"] == {
            "score_data_role": "hpo_objective_only",
            "score_data_used": False,
            "source": "tuning.winner",
        }
        assert by_id["ready-pred"]["spectral_replay_evidence_status"]["status"] == "ready_for_spectral_replay"
        assert by_id["missing-pred"]["ready"] is False
        assert by_id["missing-pred"]["spectral_replay_evidence_status"]["missing"] == [
            "row_aligned_executable_X_or_spectra",
            "predictor_bundle",
        ]

    def test_workspace_robustness_evidence_filters_ready_and_prediction_id(self, monkeypatch, tmp_path, capsys):
        """CLI evidence inspection can be narrowed to a single ready prediction."""
        from nirs4all.data.predictions import Predictions

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        records = [
            {
                "id": "ready-pred",
                "dataset_name": "wheat",
                "model_name": "PLS",
                "partition": "test",
                "y_pred": np.asarray([1.0]),
                "X": np.asarray([[0.1, 0.2]]),
                "result_metadata": {"robustness_evidence": {"predictor_bundle": "artifacts/pls.n4a"}},
            },
            {
                "id": "missing-pred",
                "dataset_name": "wheat",
                "model_name": "PLS",
                "partition": "test",
                "y_pred": np.asarray([3.0]),
            },
        ]

        class FakePredictions:
            def to_dicts(self, load_arrays=True):
                assert load_arrays is True
                return list(records)

        monkeypatch.setattr(Predictions, "from_workspace", staticmethod(lambda *args, **kwargs: FakePredictions()))

        exit_code = invoke_cli(
            [
                "workspace",
                "robustness",
                "evidence",
                "--workspace",
                str(workspace),
                "--id",
                "ready-pred",
                "--ready-only",
                "--json",
            ]
        )
        payload = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert payload["count"] == 1
        assert payload["ready_count"] == 1
        assert payload["predictions"][0]["prediction_id"] == "ready-pred"

    def test_workspace_robustness_evidence_reads_prediction_sidecar_published_by_add_prediction(self, tmp_path, capsys):
        """CLI evidence inspection sees real X/result_metadata sidecar rows."""
        from nirs4all.data.predictions import Predictions
        from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

        workspace = tmp_path / "workspace"
        store = WorkspaceStore(workspace)
        run_id = store.begin_run(
            "cli-sidecar-run",
            config={"metric": "rmse"},
            datasets=[{"name": "wheat"}],
        )
        pipeline_id = store.begin_pipeline(
            run_id=run_id,
            name="0001_pls",
            expanded_config=[{"model": "PLSRegression"}],
            generator_choices=[],
            dataset_name="wheat",
            dataset_hash="abc123",
        )
        chain_id = store.save_chain(
            pipeline_id=pipeline_id,
            steps=[{"step_idx": 0, "operator_class": "PLSRegression", "params": {}, "artifact_id": None, "stateless": False}],
            model_step_idx=0,
            model_class="sklearn.cross_decomposition.PLSRegression",
            preprocessings="SNV",
            fold_strategy="final",
            fold_artifacts={},
            shared_artifacts={},
        )
        predictions = Predictions(store=store)
        prediction_id = predictions.add_prediction(
            dataset_name="wheat",
            model_name="PLSRegression",
            model_classname="sklearn.cross_decomposition.PLSRegression",
            fold_id="final",
            partition="test",
            y_pred=np.asarray([1.1, 1.9], dtype=float),
            X=np.asarray([[1.0, 10.0], [2.0, 20.0]], dtype=float),
            result_metadata={
                "robustness_evidence": {
                    "X": "prediction_arrays.X",
                    "predictor_bundle": "models/pls.n4a",
                    "publisher": "nirs4all.predictions.add_prediction",
                }
            },
            sample_indices=np.asarray([10, 20], dtype=np.int64),
            metric="rmse",
            task_type="regression",
            n_samples=2,
            n_features=2,
        )
        predictions.flush(pipeline_id=pipeline_id, chain_id=chain_id)
        store.close()

        exit_code = invoke_cli(
            [
                "workspace",
                "robustness",
                "evidence",
                "--workspace",
                str(workspace),
                "--id",
                prediction_id,
                "--ready-only",
                "--json",
            ]
        )
        payload = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert payload["count"] == 1
        assert payload["ready_count"] == 1
        row = payload["predictions"][0]
        assert row["prediction_id"] == prediction_id
        assert row["publisher"] == "nirs4all.predictions.add_prediction"
        assert row["spectral_replay_evidence_status"]["status"] == "ready_for_spectral_replay"
        assert row["spectral_replay_evidence_status"]["has_executable_X_or_spectra"] is True
        assert row["spectral_replay_evidence_status"]["predictor_bundle"] == "models/pls.n4a"

    def test_workspace_convert_delegates_to_transition_tools(self, monkeypatch, tmp_path):
        """Transition CLI must call nirs4all-tools with the V1 workspace target."""
        seen: dict[str, list[str]] = {}

        tools_package = ModuleType("nirs4all_tools")
        tools_cli = ModuleType("nirs4all_tools.cli")

        def fake_tools_main(argv):
            seen["argv"] = list(argv)
            return 17

        tools_cli.main = fake_tools_main
        monkeypatch.setitem(sys.modules, "nirs4all_tools", tools_package)
        monkeypatch.setitem(sys.modules, "nirs4all_tools.cli", tools_cli)

        source = tmp_path / "legacy-workspace"
        target = tmp_path / "workspace-v2"

        exit_code = invoke_cli(
            [
                "workspace",
                "convert",
                str(source),
                "--output",
                str(target),
                "--verify",
                "--strict",
            ]
        )

        assert exit_code == 17
        assert seen["argv"] == [
            "legacy",
            "migrate",
            str(source),
            "--output",
            str(target),
            "--target",
            "nirs4all-workspace-v2",
            "--verify",
            "--strict",
        ]

    def test_transition_extra_installs_full_converter_readers(self):
        """nirs4all[transition] must be sufficient for real legacy conversion."""
        pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
        metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

        transition_deps = metadata["project"]["optional-dependencies"]["transition"]

        assert "nirs4all-tools[duckdb,parquet]>=0.0.5" in transition_deps


class TestCLIErrorHandling:
    """Test CLI error handling wrapper (CLI-04)."""

    def test_unexpected_exception_exits_1(self):
        """Test that unhandled exceptions in subcommands exit with code 1."""
        from nirs4all.cli.main import main

        def failing_func(args):
            raise RuntimeError("Unexpected failure")

        mock_args = Mock()
        mock_args.test_install = False
        mock_args.test_integration = False
        mock_args.func = failing_func

        with patch("sys.argv", ["nirs4all", "workspace", "stats"]):
            with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
                exit_code = invoke_cli([])

        # Should exit with 1 due to error handling
        assert exit_code == 1

    def test_keyboard_interrupt_exits_130(self):
        """Test that KeyboardInterrupt exits with code 130."""
        from nirs4all.cli.main import main

        def interrupting_func(args):
            raise KeyboardInterrupt()

        mock_args = Mock()
        mock_args.test_install = False
        mock_args.test_integration = False
        mock_args.func = interrupting_func

        with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
            with pytest.raises(SystemExit) as excinfo:
                with patch("sys.argv", ["nirs4all"]):
                    main()

        assert excinfo.value.code == 130

    def test_system_exit_propagates(self):
        """Test that SystemExit from subcommands is not swallowed."""
        from nirs4all.cli.main import main

        def sys_exit_func(args):
            sys.exit(42)

        mock_args = Mock()
        mock_args.test_install = False
        mock_args.test_integration = False
        mock_args.func = sys_exit_func

        with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
            with pytest.raises(SystemExit) as excinfo:
                with patch("sys.argv", ["nirs4all"]):
                    main()

        assert excinfo.value.code == 42


class TestCLIConfigValidate:
    """Test 'nirs4all config validate' command."""

    def test_validate_missing_file_exits_1(self, tmp_path):
        """Test that validating a non-existent file exits with 1."""
        nonexistent = str(tmp_path / "nonexistent.yaml")
        exit_code = invoke_cli(["config", "validate", nonexistent])
        assert exit_code == 1

    def test_validate_existing_file(self, tmp_path):
        """Test that validating an existing file runs without crashing."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("pipeline: []\n")

        # The validator may return invalid, but it should not raise unexpectedly
        # We just check it exits with 0 or 1 (not a crash)
        exit_code = invoke_cli(["config", "validate", str(config_file)])
        assert exit_code in (0, 1)


class TestCLIConfigSchema:
    """Test 'nirs4all config schema' command."""

    def test_schema_pipeline(self, capsys):
        """Test that 'config schema pipeline' outputs JSON."""
        exit_code = invoke_cli(["config", "schema", "pipeline"])
        # Should exit cleanly
        assert exit_code == 0

    def test_schema_dataset(self, capsys):
        """Test that 'config schema dataset' outputs JSON."""
        exit_code = invoke_cli(["config", "schema", "dataset"])
        assert exit_code == 0


class TestCLIKeywordRegistry:
    """Test machine-readable keyword/effect registry export."""

    def test_keyword_registry_outputs_json(self, capsys):
        """CLI exports the public keyword/effect registry as JSON."""
        exit_code = invoke_cli(["keyword-registry", "--compact"])
        payload = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert payload["schema_version"] == 1
        entries = {entry["id"]: entry for entry in payload["entries"]}
        assert "robustness.scenarios.kind" in entries
        assert entries["run.tuning.space"]["value_schema"] == {"type": "object"}
        assert entries["run.tuning.space"]["ui"]["control"] == "object"
        assert entries["run.tuning.force_params"]["changes"] == ["trial_sequence", "candidate_fit", "selection"]
        assert entries["run.tuning.force_params"]["invalidates_calibration"] == "if_predictor_changes"
        assert "public decoded syntax" in entries["run.tuning.force_params"]["summary"]
        score_data_schema = entries["run.tuning.score_data"]["value_schema"]
        workspace_metadata_schema = entries["run.tuning.workspace_metadata"]["value_schema"]
        calibration_selector_schema = entries["calibrate.calibration_data.selector"]["value_schema"]
        calibrate_result_metadata_schema = entries["calibrate.result_metadata"]["value_schema"]
        predict_calibrated_result_metadata_schema = entries["predict_calibrated.result_metadata"]["value_schema"]
        predict_workspace_metadata_schema = entries["predict.workspace_metadata"]["value_schema"]
        predict_workspace_result_metadata_schema = entries["predict.workspace_result_metadata"]["value_schema"]
        robustness_workspace_metadata_schema = entries["robustness.workspace_metadata"]["value_schema"]
        assert score_data_schema["oneOf"][0]["properties"]["metadata"] == {"$ref": "#/$defs/json_native_metadata"}
        assert workspace_metadata_schema["$ref"] == "#/$defs/json_native_mapping"
        assert calibration_selector_schema["$ref"] == "#/$defs/json_native_mapping"
        assert calibrate_result_metadata_schema["$ref"] == "#/$defs/json_native_mapping"
        assert predict_calibrated_result_metadata_schema["$ref"] == "#/$defs/json_native_mapping"
        assert predict_workspace_metadata_schema["$ref"] == "#/$defs/json_native_mapping"
        assert predict_workspace_result_metadata_schema["$ref"] == "#/$defs/json_native_mapping"
        assert robustness_workspace_metadata_schema["$ref"] == "#/$defs/json_native_mapping"
        assert workspace_metadata_schema["x-nirs4all-json-native"] is True
        assert calibration_selector_schema["x-nirs4all-json-native"] is True
        assert calibrate_result_metadata_schema["x-nirs4all-json-native"] is True
        assert predict_calibrated_result_metadata_schema["x-nirs4all-json-native"] is True
        assert predict_workspace_metadata_schema["x-nirs4all-json-native"] is True
        assert predict_workspace_result_metadata_schema["x-nirs4all-json-native"] is True
        assert robustness_workspace_metadata_schema["x-nirs4all-json-native"] is True
        jsonschema.validate({"X": [[1.0]], "y": [1.0], "metadata": {"site": "north", "nested": {"ok": [1, True, None]}}}, score_data_schema)
        jsonschema.validate({"site": "north", "nested": {"ok": [1, True, None]}}, workspace_metadata_schema)
        jsonschema.validate({"partition": "calibration", "nested": {"ok": [1, True, None]}}, calibration_selector_schema)
        jsonschema.validate({"site": "north", "nested": {"ok": [1, True, None]}}, calibrate_result_metadata_schema)
        jsonschema.validate({"site": "north", "nested": {"ok": [1, True, None]}}, predict_calibrated_result_metadata_schema)
        jsonschema.validate({"site": "north", "nested": {"ok": [1, True, None]}}, predict_workspace_metadata_schema)
        jsonschema.validate({"robustness_evidence": {"predictor_bundle": "model.n4a"}}, predict_workspace_result_metadata_schema)
        jsonschema.validate({"site": "north", "nested": {"ok": [1, True, None]}}, robustness_workspace_metadata_schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"X": [[1.0]], "y": [1.0], "metadata": {" bad": 1}}, score_data_schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"bad": object()}, workspace_metadata_schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"bad": object()}, calibration_selector_schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"bad": object()}, calibrate_result_metadata_schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"bad": object()}, predict_calibrated_result_metadata_schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"bad": object()}, predict_workspace_metadata_schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"bad": object()}, predict_workspace_result_metadata_schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"bad": object()}, robustness_workspace_metadata_schema)
        assert "invalid conformal sidecar fails validation" in entries["predict.coverage"]["summary"]

    def test_keyword_registry_outputs_schema_json(self, capsys):
        """CLI exports the registry JSON Schema for static consumers."""
        exit_code = invoke_cli(["keyword-registry", "--schema", "--compact"])
        payload = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert payload["$id"] == "https://nirs4all.org/schemas/keyword-effects/v1"
        assert payload["properties"]["schema_id"]["const"] == payload["$id"]

    def test_keyword_registry_writes_output_file(self, tmp_path, capsys):
        """CLI can write the registry artifact for CI or Studio generation."""
        output = tmp_path / "artifacts" / "keyword-registry.json"

        exit_code = invoke_cli(["keyword-registry", "--output", str(output)])
        stdout = capsys.readouterr().out
        payload = json.loads(output.read_text(encoding="utf-8"))

        assert exit_code == 0
        assert stdout == ""
        assert payload["schema_id"] == "https://nirs4all.org/schemas/keyword-effects/v1"
        entries = {entry["id"]: entry for entry in payload["entries"]}
        assert "invalid conformal sidecar fails validation" in entries["predict.coverage"]["summary"]


class TestCLIRobustnessSummarySchema:
    """Test robustness summary schema export."""

    def test_robustness_summary_schema_outputs_json(self, capsys):
        """CLI exports the robustness summary JSON Schema."""
        exit_code = invoke_cli(["robustness-summary-schema", "--compact"])
        payload = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert payload["$id"] == "https://nirs4all.org/schemas/robustness-summary/v1"
        assert payload["properties"]["format"]["const"] == "nirs4all.robustness.summary"
        guarantee_schema = payload["properties"]["conformal_guarantee_status"]
        assert guarantee_schema["type"] == ["object", "null"]
        assert guarantee_schema["properties"]["effective_engine"] == {"type": "string"}
        assert guarantee_schema["properties"]["coverage"] == {"type": "array", "items": {"type": "number"}}

    def test_robustness_summary_schema_writes_output_file(self, tmp_path, capsys):
        """CLI can write the schema artifact for CI or Studio generation."""
        output = tmp_path / "artifacts" / "robustness-summary.schema.json"

        exit_code = invoke_cli(["robustness-summary-schema", "--output", str(output)])
        stdout = capsys.readouterr().out
        payload = json.loads(output.read_text(encoding="utf-8"))

        assert exit_code == 0
        assert stdout == ""
        assert payload["$id"] == "https://nirs4all.org/schemas/robustness-summary/v1"
        guarantee_schema = payload["properties"]["conformal_guarantee_status"]
        assert guarantee_schema["type"] == ["object", "null"]
        assert guarantee_schema["properties"]["effective_engine"] == {"type": "string"}


class TestCLITuningSummarySchema:
    """Test tuning summary schema export."""

    def test_tuning_summary_schema_outputs_json(self, capsys):
        """CLI exports the tuning summary JSON Schema."""
        exit_code = invoke_cli(["tuning-summary-schema", "--compact"])
        payload = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert payload["$id"] == "https://nirs4all.org/schemas/tuning-summary/v1"
        assert payload["properties"]["format"]["const"] == "nirs4all.tuning.summary"
        persistence_schema = payload["properties"]["persistence"]
        assert persistence_schema["properties"]["storage_configured"] == {"type": "boolean"}
        assert persistence_schema["properties"]["optimizer_state_resume_supported"] == {"type": "boolean"}
        assert "persistence" in payload["required"]

    def test_tuning_summary_schema_writes_output_file(self, tmp_path, capsys):
        """CLI can write the schema artifact for CI or Studio generation."""
        output = tmp_path / "artifacts" / "tuning-summary.schema.json"

        exit_code = invoke_cli(["tuning-summary-schema", "--output", str(output)])
        stdout = capsys.readouterr().out
        payload = json.loads(output.read_text(encoding="utf-8"))

        assert exit_code == 0
        assert stdout == ""
        assert payload["$id"] == "https://nirs4all.org/schemas/tuning-summary/v1"


class TestCLIRobustnessReport:
    """Test verified robustness report artifact publication."""

    @staticmethod
    def _write_report_json(tmp_path) -> Path:
        import nirs4all

        calibrated = nirs4all.calibrate(
            y_true=[1.0, 2.0, 3.0, 4.0],
            y_pred_calibration=[1.0, 2.0, 3.0, 4.0],
            y_pred=[10.0, 20.0],
            calibration_sample_ids=["c1", "c2", "c3", "c4"],
            prediction_sample_ids=["p1", "p2"],
            coverage=0.8,
            as_predict_result=True,
        )
        report = nirs4all.robustness(
            calibrated,
            y_true=[10.0, 19.0],
            metadata={"batch": ["b1", "b2"]},
            slice_by=["batch"],
            seed=7,
        )
        return report.save_json(tmp_path / "robustness.json")

    def test_robustness_report_outputs_markdown(self, tmp_path, capsys):
        """CLI publishes a verified report JSON as Markdown on stdout."""
        report_json = self._write_report_json(tmp_path)

        exit_code = invoke_cli(["robustness-report", str(report_json)])
        stdout = capsys.readouterr().out

        assert exit_code == 0
        assert stdout.startswith("# NIRS4All robustness report\n")
        assert "## Scenario summary" in stdout
        assert "| observed | baseline | not required | 0 | 2 | 0.707106781187 | 0 | 1 | 0.5 | 0.3 | batch=b2 | 1 |" in stdout

    def test_robustness_report_writes_html_file(self, tmp_path, capsys):
        """CLI writes standalone HTML reports for CI/release artifacts."""
        report_json = self._write_report_json(tmp_path)
        output = tmp_path / "artifacts" / "robustness.html"

        exit_code = invoke_cli(["robustness-report", str(report_json), "--format", "html", "--output", str(output)])
        stdout = capsys.readouterr().out
        html = output.read_text(encoding="utf-8")

        assert exit_code == 0
        assert stdout == ""
        assert html.startswith("<!doctype html>\n")
        assert "<h2>Scenario summary</h2>" in html

    def test_robustness_report_outputs_compact_json(self, tmp_path, capsys):
        """CLI can normalize and re-emit verified report JSON."""
        report_json = self._write_report_json(tmp_path)

        exit_code = invoke_cli(["robustness-report", str(report_json), "--format", "json", "--compact"])
        payload = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert payload["metadata"]["seed"] == 7
        assert payload["fingerprint"]
        assert payload["scenarios"][0]["scenario"]["kind"] == "observed"

    def test_robustness_report_outputs_compact_summary_json(self, tmp_path, capsys):
        """CLI emits lightweight summary JSON for CI and Studio cards."""
        report_json = self._write_report_json(tmp_path)

        exit_code = invoke_cli(["robustness-report", str(report_json), "--format", "summary", "--compact"])
        payload = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert payload["format"] == "nirs4all.robustness.summary"
        assert payload["fingerprint"]
        assert payload["summary"][0]["scenario_label"] == "observed"

    def test_robustness_report_writes_parquet_directory(self, tmp_path, capsys):
        """CLI writes tabular Parquet-directory artifacts for downstream tools."""
        import polars as pl

        report_json = self._write_report_json(tmp_path)
        output = tmp_path / "robustness-report.parquet"

        exit_code = invoke_cli(["robustness-report", str(report_json), "--format", "parquet", "--output", str(output)])
        stdout = capsys.readouterr().out
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        summary = pl.read_parquet(output / "summary.parquet")

        assert exit_code == 0
        assert stdout == ""
        assert manifest["tables"]["summary"] == "summary.parquet"
        assert summary["scenario_label"].to_list() == ["observed"]

    def test_robustness_report_writes_artifact_directory(self, tmp_path, capsys):
        """CLI writes a complete robustness publication bundle."""
        report_json = self._write_report_json(tmp_path)
        output = tmp_path / "robustness-artifacts"

        exit_code = invoke_cli(["robustness-report", str(report_json), "--format", "artifacts", "--output", str(output)])
        stdout = capsys.readouterr().out
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

        assert exit_code == 0
        assert stdout == ""
        assert manifest["format"] == "nirs4all.robustness.artifact-directory"
        assert manifest["files"] == {
            "html": "report.html",
            "json": "report.json",
            "markdown": "report.md",
            "parquet": "report.parquet",
            "summary": "summary.json",
        }
        assert (output / "report.json").is_file()
        assert (output / "summary.json").is_file()
        assert (output / "report.md").is_file()
        assert (output / "report.html").is_file()
        assert (output / "report.parquet" / "manifest.json").is_file()

    def test_robustness_report_republishes_from_artifact_directory(self, tmp_path, capsys):
        """CLI accepts a verified artifact directory as input."""
        import nirs4all

        report_json = self._write_report_json(tmp_path)
        source = tmp_path / "source-artifacts"
        report = nirs4all.RobustnessReport.load_json(report_json)
        report.save_artifacts(source)

        exit_code = invoke_cli(["robustness-report", str(source), "--format", "json", "--compact"])
        payload = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert payload["fingerprint"] == report.fingerprint

        exit_code = invoke_cli(["robustness-report", str(source), "--format", "summary", "--compact"])
        summary = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert summary["fingerprint"] == report.fingerprint
        assert summary["summary"][0]["scenario_label"] == "observed"

        exit_code = invoke_cli(["robustness-report", str(source), "--format", "markdown"])
        markdown = capsys.readouterr().out

        assert exit_code == 0
        assert markdown.startswith("# NIRS4All robustness report\n")
        assert "## Scenario summary" in markdown

    def test_robustness_report_refuses_corrupted_artifact_directory(self, tmp_path):
        """CLI verifies artifact directories before republishing them."""
        import nirs4all

        report_json = self._write_report_json(tmp_path)
        source = tmp_path / "source-artifacts"
        nirs4all.RobustnessReport.load_json(report_json).save_artifacts(source)
        (source / "report.md").write_text("corrupted", encoding="utf-8")

        exit_code = invoke_cli(["robustness-report", str(source), "--format", "json"])

        assert exit_code == 1

    def test_robustness_report_parquet_requires_output(self, tmp_path):
        """CLI refuses ambiguous Parquet publication without a target directory."""
        report_json = self._write_report_json(tmp_path)

        exit_code = invoke_cli(["robustness-report", str(report_json), "--format", "parquet"])

        assert exit_code == 1

    def test_robustness_report_artifacts_requires_output(self, tmp_path):
        """CLI refuses ambiguous publication bundles without a target directory."""
        report_json = self._write_report_json(tmp_path)

        exit_code = invoke_cli(["robustness-report", str(report_json), "--format", "artifacts"])

        assert exit_code == 1


class TestCLIWorkspaceNativeArtifacts:
    """Test native tuning/conformal workspace inspection commands."""

    def test_workspace_tuning_list_and_show_json(self, tmp_path, capsys):
        """CLI can list and show persisted TuningResult records as JSON."""
        import nirs4all
        from nirs4all.pipeline.dagml.tuning_contracts import TrialResult, TuningResult, parse_tuning_spec

        workspace = tmp_path / "workspace"
        tuning = parse_tuning_spec({"engine": "optuna", "space": {"alpha": [0.1, 1.0]}, "sampler": "grid", "n_trials": 2})
        result = TuningResult(
            tuning=tuning,
            best_params={"alpha": 0.1},
            best_value=0.0,
            trials=(TrialResult(number=0, params={"alpha": 0.1}, value=0.0, state="COMPLETE", diagnostics={}),),
            optimizer="optuna",
        )
        nirs4all.save_workspace_tuning_result(workspace, result, tuning_id="cli-tune", name="CLI tuning")

        capsys.readouterr()
        assert invoke_cli(["workspace", "tuning", "list", "--workspace", str(workspace), "--json"]) == 0
        listed = json.loads(capsys.readouterr().out)
        assert listed[0]["tuning_id"] == "cli-tune"
        assert listed[0]["result_fingerprint"] == result.fingerprint

        assert invoke_cli(["workspace", "tuning", "show", "cli-tune", "--workspace", str(workspace), "--json"]) == 0
        shown = json.loads(capsys.readouterr().out)
        assert shown["fingerprint"] == result.fingerprint
        assert shown["best_params"] == {"alpha": 0.1}

        assert invoke_cli(["workspace", "tuning", "export", "cli-tune", "--workspace", str(workspace), "--compact"]) == 0
        exported_summary = json.loads(capsys.readouterr().out)
        assert exported_summary["format"] == "nirs4all.tuning.summary"
        assert exported_summary["fingerprint"] == result.fingerprint
        assert exported_summary["sampler"] == "grid"
        assert exported_summary["pruner"] is None
        assert exported_summary["seed"] is None
        assert exported_summary["trial_states"] == {"COMPLETE": 1}

        assert (
            invoke_cli(
                [
                    "workspace",
                    "tuning",
                    "export",
                    "cli-tune",
                    "--workspace",
                    str(workspace),
                    "--format",
                    "json",
                    "--compact",
                ]
            )
            == 0
        )
        exported_json = json.loads(capsys.readouterr().out)
        assert exported_json["fingerprint"] == result.fingerprint
        assert exported_json["tuning"]["space"] == {"alpha": [0.1, 1.0]}

        output = tmp_path / "exports" / "tuning-summary.json"
        assert (
            invoke_cli(
                [
                    "workspace",
                    "tuning",
                    "export",
                    "cli-tune",
                    "--workspace",
                    str(workspace),
                    "--format",
                    "summary",
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
        assert capsys.readouterr().out == ""
        written = json.loads(output.read_text(encoding="utf-8"))
        assert written["fingerprint"] == result.fingerprint
        assert written["best_params"] == {"alpha": 0.1}

    def test_workspace_conformal_list_and_show_json(self, tmp_path, capsys):
        """CLI can list and show persisted conformal result records as JSON."""
        import nirs4all

        workspace = tmp_path / "workspace"
        calibrated = nirs4all.calibrate(
            y_true=[1.0, 2.0, 3.0, 4.0],
            y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
            y_pred=[10.0, 20.0],
            calibration_sample_ids=["c1", "c2", "c3", "c4"],
            prediction_sample_ids=["p1", "p2"],
            coverage=0.8,
            workspace_path=workspace,
            workspace_conformal_id="cli-conformal",
            workspace_name="CLI conformal",
        )

        capsys.readouterr()
        assert invoke_cli(["workspace", "conformal", "list", "--workspace", str(workspace), "--json"]) == 0
        listed = json.loads(capsys.readouterr().out)
        assert listed[0]["conformal_id"] == "cli-conformal"
        assert listed[0]["result_fingerprint"] == calibrated.fingerprint

        assert invoke_cli(["workspace", "conformal", "show", "cli-conformal", "--workspace", str(workspace), "--json"]) == 0
        shown = json.loads(capsys.readouterr().out)
        assert shown["fingerprint"] == calibrated.fingerprint
        assert shown["sample_ids"] == ["p1", "p2"]

    def test_workspace_conformal_predict_json(self, tmp_path, capsys):
        """CLI can apply a persisted conformal calibrator to new point predictions."""
        import nirs4all

        workspace = tmp_path / "workspace"
        nirs4all.calibrate(
            y_true=[1.0, 2.0, 3.0, 4.0],
            y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
            y_pred=[10.0],
            calibration_sample_ids=["c1", "c2", "c3", "c4"],
            prediction_sample_ids=["p0"],
            coverage=0.8,
            workspace_path=workspace,
            workspace_conformal_id="cli-conformal",
        )

        capsys.readouterr()
        exit_code = invoke_cli(
            [
                "workspace",
                "conformal",
                "predict",
                "cli-conformal",
                "--workspace",
                str(workspace),
                "--y-pred",
                "30.0,40.0",
                "--sample-ids",
                "p1,p2",
                "--json",
            ]
        )
        payload = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert payload["sample_ids"] == ["p1", "p2"]
        assert payload["y_pred"] == [30.0, 40.0]
        assert payload["conformal_guarantee_status"]["status"] == "active"
        assert payload["intervals"]["0.8"]["lower"] == [29.6, 39.6]
        assert payload["intervals"]["0.8"]["upper"] == [30.4, 40.4]

    def test_workspace_conformal_predict_rejects_mismatched_lengths(self, tmp_path):
        """CLI refuses ambiguous prediction/sample id alignment."""
        import nirs4all

        workspace = tmp_path / "workspace"
        nirs4all.calibrate(
            y_true=[1.0, 2.0, 3.0, 4.0],
            y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
            y_pred=[10.0],
            calibration_sample_ids=["c1", "c2", "c3", "c4"],
            prediction_sample_ids=["p0"],
            coverage=0.8,
            workspace_path=workspace,
            workspace_conformal_id="cli-conformal",
        )

        exit_code = invoke_cli(
            [
                "workspace",
                "conformal",
                "predict",
                "cli-conformal",
                "--workspace",
                str(workspace),
                "--y-pred",
                "30.0,40.0",
                "--sample-ids",
                "p1",
                "--json",
            ]
        )

        assert exit_code == 1

    def test_workspace_robustness_list_and_show_json(self, tmp_path, capsys):
        """CLI can list and show persisted robustness reports as JSON."""
        import polars as pl

        import nirs4all

        workspace = tmp_path / "workspace"
        calibrated = nirs4all.calibrate(
            y_true=[1.0, 2.0, 3.0, 4.0],
            y_pred_calibration=[1.0, 2.0, 3.0, 4.0],
            y_pred=[10.0, 20.0],
            calibration_sample_ids=["c1", "c2", "c3", "c4"],
            prediction_sample_ids=["p1", "p2"],
            coverage=0.8,
            as_predict_result=True,
        )
        report = nirs4all.robustness(
            calibrated,
            y_true=[10.0, 19.0],
            metadata={"batch": ["b1", "b2"]},
            slice_by=["batch"],
        )
        nirs4all.save_workspace_robustness_report(
            workspace,
            report,
            robustness_id="cli-robustness",
            name="CLI robustness",
        )

        capsys.readouterr()
        assert invoke_cli(["workspace", "robustness", "list", "--workspace", str(workspace), "--json"]) == 0
        listed = json.loads(capsys.readouterr().out)
        assert listed[0]["robustness_id"] == "cli-robustness"
        assert listed[0]["result_fingerprint"] == report.fingerprint

        assert invoke_cli(["workspace", "robustness", "show", "cli-robustness", "--workspace", str(workspace), "--json"]) == 0
        shown = json.loads(capsys.readouterr().out)
        assert shown["fingerprint"] == report.fingerprint
        assert shown["slice_by"] == ["batch"]

        assert invoke_cli(["workspace", "robustness", "export", "cli-robustness", "--workspace", str(workspace)]) == 0
        markdown = capsys.readouterr().out
        assert markdown.startswith("# NIRS4All robustness report\n")
        assert "## Scenario summary" in markdown

        html_output = tmp_path / "exports" / "robustness.html"
        assert (
            invoke_cli(
                [
                    "workspace",
                    "robustness",
                    "export",
                    "cli-robustness",
                    "--workspace",
                    str(workspace),
                    "--format",
                    "html",
                    "--output",
                    str(html_output),
                ]
            )
            == 0
        )
        assert capsys.readouterr().out == ""
        assert "<h2>Scenario summary</h2>" in html_output.read_text(encoding="utf-8")

        assert (
            invoke_cli(
                [
                    "workspace",
                    "robustness",
                    "export",
                    "cli-robustness",
                    "--workspace",
                    str(workspace),
                    "--format",
                    "json",
                    "--compact",
                ]
            )
            == 0
        )
        exported_json = json.loads(capsys.readouterr().out)
        assert exported_json["fingerprint"] == report.fingerprint

        assert (
            invoke_cli(
                [
                    "workspace",
                    "robustness",
                    "export",
                    "cli-robustness",
                    "--workspace",
                    str(workspace),
                    "--format",
                    "summary",
                    "--compact",
                ]
            )
            == 0
        )
        exported_summary = json.loads(capsys.readouterr().out)
        assert exported_summary["fingerprint"] == report.fingerprint
        assert exported_summary["summary"][0]["scenario_label"] == "observed"

        parquet_output = tmp_path / "exports" / "robustness.parquet"
        assert (
            invoke_cli(
                [
                    "workspace",
                    "robustness",
                    "export",
                    "cli-robustness",
                    "--workspace",
                    str(workspace),
                    "--format",
                    "parquet",
                    "--output",
                    str(parquet_output),
                ]
            )
            == 0
        )
        assert capsys.readouterr().out == ""
        summary = pl.read_parquet(parquet_output / "summary.parquet")
        assert summary["scenario_label"].to_list() == ["observed"]

        artifact_output = tmp_path / "exports" / "robustness-artifacts"
        assert (
            invoke_cli(
                [
                    "workspace",
                    "robustness",
                    "export",
                    "cli-robustness",
                    "--workspace",
                    str(workspace),
                    "--format",
                    "artifacts",
                    "--output",
                    str(artifact_output),
                ]
            )
            == 0
        )
        assert capsys.readouterr().out == ""
        artifact_manifest = json.loads((artifact_output / "manifest.json").read_text(encoding="utf-8"))
        assert artifact_manifest["fingerprint"] == report.fingerprint
        assert artifact_manifest["format"] == "nirs4all.robustness.artifact-directory"
        assert (artifact_output / "report.json").is_file()
        assert (artifact_output / "summary.json").is_file()
        assert (artifact_output / "report.parquet" / "manifest.json").is_file()

        assert invoke_cli(["workspace", "robustness", "export", "cli-robustness", "--workspace", str(workspace), "--format", "parquet"]) == 1
        assert invoke_cli(["workspace", "robustness", "export", "cli-robustness", "--workspace", str(workspace), "--format", "artifacts"]) == 1

    def test_workspace_robustness_from_prediction_outputs_summary_and_forwards_save_flags(self, monkeypatch, tmp_path, capsys):
        """CLI computes a report from one workspace prediction through the public helper."""
        import nirs4all

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        calls = []

        def _fake_from_prediction(workspace_path, prediction_id, **kwargs):
            calls.append((workspace_path, prediction_id, kwargs))
            result = nirs4all.PredictResult(
                y_pred=np.asarray([10.0, 20.0], dtype=float),
                metadata={"row_metadata": [{"batch": "a"}, {"batch": "b"}]},
                sample_indices=np.asarray(["p1", "p2"], dtype=object),
            )
            return nirs4all.robustness(
                result,
                y_true=kwargs["y_true"],
                scenarios=kwargs["scenarios"],
                slice_by=kwargs["slice_by"],
                metadata=kwargs["metadata"],
                seed=kwargs["seed"],
            )

        monkeypatch.setattr(nirs4all, "robustness_from_workspace_prediction", _fake_from_prediction)

        exit_code = invoke_cli(
            [
                "workspace",
                "robustness",
                "from-prediction",
                "--workspace",
                str(workspace),
                "--prediction-id",
                "pred-001",
                "--y-true",
                "10.0,19.0",
                "--scenarios-json",
                '[{"kind":"prediction_bias","severity":0.5}]',
                "--metadata-json",
                '{"batch":["a","b"]}',
                "--slice-by",
                "batch",
                "--seed",
                "7",
                "--save-to-workspace",
                "--workspace-name",
                "CLI audit",
                "--workspace-robustness-id",
                "cli-pred-audit",
                "--workspace-metadata-json",
                '{"source":"cli"}',
                "--format",
                "summary",
                "--compact",
            ]
        )
        payload = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert payload["format"] == "nirs4all.robustness.summary"
        assert payload["summary"][0]["scenario_label"] == "prediction_bias"
        assert payload["summary"][0]["worst_slice_label"] == "batch=b"
        assert calls == [
            (
                workspace,
                "pred-001",
                {
                    "y_true": [10.0, 19.0],
                    "scenarios": [{"kind": "prediction_bias", "severity": 0.5}],
                    "slice_by": ["batch"],
                    "metadata": {"batch": ["a", "b"]},
                    "seed": 7,
                    "save_to_workspace": True,
                    "workspace_name": "CLI audit",
                    "workspace_robustness_id": "cli-pred-audit",
                    "workspace_metadata": {"source": "cli"},
                },
            )
        ]


class TestCLIWorkspaceValidation:
    """Test workspace path validation in CLI (WS-01)."""

    def test_workspace_list_runs_nonexistent_path_exits_1(self, tmp_path):
        """Test that list-runs with nonexistent workspace exits with 1."""
        nonexistent = str(tmp_path / "does_not_exist")
        exit_code = invoke_cli(["workspace", "list-runs", "--workspace", nonexistent])
        assert exit_code == 1

    def test_workspace_stats_nonexistent_path_exits_1(self, tmp_path):
        """Test that stats with nonexistent workspace exits with 1."""
        nonexistent = str(tmp_path / "does_not_exist")
        exit_code = invoke_cli(["workspace", "stats", "--workspace", nonexistent])
        assert exit_code == 1

    def test_workspace_query_best_nonexistent_path_exits_1(self, tmp_path):
        """Test that query-best with nonexistent workspace exits with 1."""
        nonexistent = str(tmp_path / "does_not_exist")
        exit_code = invoke_cli(["workspace", "query-best", "--workspace", nonexistent])
        assert exit_code == 1

    def test_workspace_filter_nonexistent_path_exits_1(self, tmp_path):
        """Test that filter with nonexistent workspace exits with 1."""
        nonexistent = str(tmp_path / "does_not_exist")
        exit_code = invoke_cli(["workspace", "filter", "--workspace", nonexistent])
        assert exit_code == 1

    def test_workspace_list_library_nonexistent_path_exits_1(self, tmp_path):
        """Test that list-library with nonexistent workspace exits with 1."""
        nonexistent = str(tmp_path / "does_not_exist")
        exit_code = invoke_cli(["workspace", "list-library", "--workspace", nonexistent])
        assert exit_code == 1

    def test_workspace_init_invalid_parent_exits_1(self, tmp_path):
        """Test that init with nonexistent parent directory exits with 1."""
        invalid_path = str(tmp_path / "nonexistent_parent" / "workspace")
        exit_code = invoke_cli(["workspace", "init", invalid_path])
        assert exit_code == 1

    def test_workspace_init_file_instead_of_dir_exits_1(self, tmp_path):
        """Test that init on an existing file exits with 1."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("content")
        exit_code = invoke_cli(["workspace", "init", str(file_path)])
        assert exit_code == 1

    def test_workspace_init_new_dir_succeeds(self, tmp_path):
        """Test that init in an existing parent directory succeeds."""
        workspace_path = tmp_path / "new_workspace"
        exit_code = invoke_cli(["workspace", "init", str(workspace_path)])
        assert exit_code == 0
        assert workspace_path.exists()
