"""
U09 - Native tuning + conformal calibration
===========================================

Run the native ``run(tuning=...)`` subset end to end:

* tune a linear transformer→estimator pipeline with conformal-aware scoring;
* project an explicit winner prediction entry;
* calibrate conformal intervals from that winner;
* persist both the tuning trace and conformal result in one workspace;
* reload the persisted artifacts and apply the calibrator to new point predictions.
* show the standalone typed ``ConformalCalibrationData`` helper used by
  forms/bindings/Studio payload builders.
* compute, persist, publish, and reload an audit-only robustness report for the
  calibrated prediction cohort.

This example uses small in-memory arrays so it is fast enough for CI. The same
syntax applies to real NIR arrays once the calibration cohort and prediction
sample ids are explicit.

Duration: < 10 seconds
Difficulty: ★★★☆☆
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import nirs4all


def main(workspace_path: str | Path | None = None) -> dict[str, Any]:
    """Run the example and return key objects for smoke tests."""

    if workspace_path is None:
        workspace_context = tempfile.TemporaryDirectory(prefix="n4a-native-tuning-conformal-")
        workspace = Path(workspace_context.name)
    else:
        workspace_context = None
        workspace = Path(workspace_path)

    try:
        # Development data used for terminal refit.
        X_dev = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        y_dev = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)

        # Explicit optimizer scoring cohort. This keeps selection auditable and
        # avoids implicit data splitting inside the tuning surface. The tuning
        # objective below scores temporary conformal intervals on this cohort.
        X_score = np.asarray([[4.0], [5.0], [6.0]], dtype=float)
        y_score = np.asarray([4.0, 5.0, 6.0], dtype=float)

        # Explicit development calibration cohort used only by the temporary
        # conformal-aware tuning scorer. It is separate from the final
        # calibration derived from the winner projection.
        X_score_cal = np.asarray([[1.5], [2.5], [3.5], [4.5]], dtype=float)
        y_score_cal = np.asarray([1.4, 2.7, 3.4, 4.8], dtype=float)
        score_calibration_ids = ["score-cal-001", "score-cal-002", "score-cal-003", "score-cal-004"]

        # Explicit winner projection cohort. The conformal calibration below
        # derives calibration_data from this prediction entry.
        X_cal = np.asarray([[7.0], [8.0], [9.0], [10.0]], dtype=float)
        y_cal = np.asarray([7.2, 7.8, 9.3, 9.7], dtype=float)
        calibration_ids = ["cal-001", "cal-002", "cal-003", "cal-004"]

        result = nirs4all.run(
            pipeline=[
                {"name": "scale", "transform": StandardScaler()},
                {"model": Ridge()},
            ],
            dataset=(X_dev, y_dev),
            engine="dag-ml",
            workspace_path=workspace,
            tuning=nirs4all.NativeTuning(
                engine="optuna",
                space={"model.alpha": [0.1, 1.0]},
                metric="conformal_mean_width",
                direction="minimize",
                sampler="grid",
                n_trials=2,
                score_data=nirs4all.TuningScoreData(
                    X=X_score,
                    y=y_score,
                    conformal_coverage=0.8,
                    conformal_calibration=nirs4all.TuningConformalScoreCalibration(
                        X=X_score_cal,
                        y_true=y_score_cal,
                        sample_ids=score_calibration_ids,
                    ),
                ),
                workspace_tuning_id="u09-native-tuning",
                winner=nirs4all.TuningWinner(
                    X=X_cal,
                    y_true=y_cal,
                    score=0.3,
                    metric="rmse",
                    sample_ids=calibration_ids,
                    model_name="RidgeNativeTuned",
                ),
            ),
            calibration=nirs4all.TuningCalibration(
                y_pred=np.asarray([11.0, 12.0], dtype=float),
                prediction_sample_ids=["pred-001", "pred-002"],
                coverage=0.8,
                workspace_conformal_id="u09-conformal",
                workspace_metadata={"example": "U09_native_tuning_conformal"},
            ),
            verbose=0,
        )

        restored_tuning = nirs4all.load_workspace_tuning_result(workspace, "u09-native-tuning")
        restored_conformal = nirs4all.load_workspace_calibrated_result(workspace, "u09-conformal")
        replayed_prediction = nirs4all.predict_calibrated(
            restored_conformal,
            y_pred=[13.0, 14.0],
            prediction_sample_ids=["pred-003", "pred-004"],
        )
        robustness_report = replayed_prediction.robustness(
            y_true=[13.2, 13.7],
            scenarios=[
                nirs4all.RobustnessScenarioSpec(kind="observed"),
                nirs4all.RobustnessScenarioSpec(kind="prediction_bias", severity=0.1),
                nirs4all.RobustnessScenarioSpec(kind="prediction_noise", severity=0.05),
            ],
            metadata={"batch": ["external-a", "external-b"]},
            slice_by=["batch"],
            seed=7,
            workspace_path=workspace,
            workspace_robustness_id="u09-robustness",
            workspace_name="U09 robustness audit",
            workspace_metadata={"example": "U09_native_tuning_conformal"},
        )
        robustness_artifacts = workspace / "u09-robustness-artifacts"
        robustness_report.save_artifacts(robustness_artifacts)
        restored_robustness = nirs4all.load_workspace_robustness_report(workspace, "u09-robustness")
        restored_robustness_bundle = nirs4all.RobustnessReport.load_artifacts(robustness_artifacts)
        typed_direct_calibration = nirs4all.calibrate(
            calibration_data=nirs4all.ConformalCalibrationData(
                y_true=y_cal,
                y_pred=np.asarray([7.0, 8.0, 9.0, 10.0], dtype=float),
                sample_ids=calibration_ids,
                metadata={"source": ["typed_direct_example"] * len(calibration_ids)},
            ),
            y_pred=[15.0],
            prediction_sample_ids=["typed-pred-001"],
            coverage=0.8,
            predictor_fingerprint="u09-typed-direct",
        )

        print("\nNative tuning + conformal calibration complete")
        print(f"  workspace: {workspace}")
        print(f"  best params: {result.run.tuning_best_params}")
        print(f"  tuning objective: {result.run.tuning_result.tuning.metric}")
        print(f"  tuning fingerprint: {restored_tuning.fingerprint}")
        print(f"  conformal fingerprint: {restored_conformal.fingerprint}")
        print(f"  80% interval lower bounds: {replayed_prediction.interval(0.8).lower.tolist()}")
        print(f"  robustness fingerprint: {restored_robustness.fingerprint}")
        print(f"  robustness scenarios: {[row['scenario_label'] for row in robustness_report.summary_rows()]}")
        print(f"  typed helper fingerprint: {typed_direct_calibration.fingerprint}")

        return {
            "workspace": workspace,
            "result": result,
            "restored_tuning": restored_tuning,
            "restored_conformal": restored_conformal,
            "replayed_prediction": replayed_prediction,
            "robustness_report": robustness_report,
            "restored_robustness": restored_robustness,
            "restored_robustness_bundle": restored_robustness_bundle,
            "robustness_artifacts": robustness_artifacts,
            "typed_direct_calibration": typed_direct_calibration,
        }
    finally:
        if workspace_context is not None:
            workspace_context.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U09 Native tuning + conformal calibration")
    parser.add_argument("--workspace", default=None, help="Workspace directory for persisted tuning/conformal artifacts")
    args = parser.parse_args()
    main(args.workspace)
