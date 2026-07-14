"""Smoke lock for the native tuning + conformal example."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

import nirs4all


def test_u09_native_tuning_conformal_example_roundtrips_workspace(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    example_path = repo_root / "examples" / "user" / "04_models" / "U09_native_tuning_conformal.py"
    spec = importlib.util.spec_from_file_location("u09_native_tuning_conformal", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = module.main(tmp_path / "workspace")

    assert isinstance(output["result"], nirs4all.TunedSingleEstimatorConformalResult)
    assert output["result"].run.tuning_id == "u09-native-tuning"
    assert output["result"].run.tuning_result.tuning.metric == "conformal_mean_width"
    assert output["result"].run.tuning_result.trials[0].diagnostics["score_family"] == "conformal"
    assert output["result"].run.tuning_result.trials[0].diagnostics["score_extractor"] == "conformal_temporary_calibration"
    assert output["result"].run.tuning_result.trials[0].diagnostics["final_calibration_scope"] == "unmodified_by_score_data"
    assert output["restored_tuning"].to_dict() == output["result"].run.tuning_result.to_dict()
    assert output["restored_conformal"].fingerprint == output["result"].calibrated.metadata["calibrated_result_fingerprint"]
    assert output["replayed_prediction"].conformal_guarantee_status is not None
    assert output["replayed_prediction"].conformal_guarantee_status["status"] == "active"
    np.testing.assert_array_equal(output["replayed_prediction"].sample_indices, ["pred-003", "pred-004"])
    assert isinstance(output["robustness_report"], nirs4all.RobustnessReport)
    assert output["restored_robustness"].to_dict() == output["robustness_report"].to_dict()
    assert output["restored_robustness_bundle"].to_dict() == output["robustness_report"].to_dict()
    assert [row["scenario_label"] for row in output["robustness_report"].summary_rows()] == [
        "observed",
        "prediction_bias",
        "prediction_noise",
    ]
    assert output["robustness_report"].summary_artifact()["format"] == "nirs4all.robustness.summary"
    assert output["robustness_artifacts"].joinpath("manifest.json").is_file()
    assert output["robustness_artifacts"].joinpath("summary.json").is_file()
    assert output["typed_direct_calibration"].artifact.predictor_fingerprint == "u09-typed-direct"
    assert output["typed_direct_calibration"].artifact.calibration_cohort.sample_ids == (
        "cal-001",
        "cal-002",
        "cal-003",
        "cal-004",
    )
    assert output["typed_direct_calibration"].sample_ids == ("typed-pred-001",)
    assert (tmp_path / "workspace" / "store.sqlite").is_file()


def test_u10_native_pls_conformal_robustness_example_roundtrips_workspace(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    example_path = repo_root / "examples" / "user" / "04_models" / "U10_native_pls_conformal_robustness.py"
    spec = importlib.util.spec_from_file_location("u10_native_pls_conformal_robustness", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = module.main(tmp_path / "pls-workspace")

    assert isinstance(output["result"], nirs4all.TunedSingleEstimatorConformalResult)
    assert output["result"].run.tuning_id == "u10-native-pls-tuning"
    assert output["result"].run.tuning_best_params == {"model.n_components": 2}
    assert output["restored_tuning"].to_dict() == output["result"].run.tuning_result.to_dict()
    assert output["restored_conformal"].fingerprint == output["result"].calibrated.metadata["calibrated_result_fingerprint"]
    assert output["replayed_prediction"].conformal_guarantee_status["status"] == "active"
    np.testing.assert_array_equal(output["replayed_prediction"].sample_indices, ["pls-pred-003", "pls-pred-004"])
    assert isinstance(output["robustness_report"], nirs4all.RobustnessReport)
    assert output["restored_robustness"].to_dict() == output["robustness_report"].to_dict()
    assert output["restored_robustness_bundle"].to_dict() == output["robustness_report"].to_dict()
    assert [row["scenario_label"] for row in output["robustness_report"].summary_rows()] == [
        "observed",
        "prediction_bias",
        "prediction_noise",
    ]
    assert output["robustness_report"].summary_artifact()["format"] == "nirs4all.robustness.summary"
    assert output["robustness_artifacts"].joinpath("manifest.json").is_file()
    assert output["robustness_artifacts"].joinpath("summary.json").is_file()
    assert (tmp_path / "pls-workspace" / "store.sqlite").is_file()
