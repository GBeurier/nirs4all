"""
U10 - Native PLS tuning + conformal calibration + robustness
============================================================

Run the current native Python E2E lane on a spectroscopy-standard
``PLSRegression`` model:

* tune ``model.n_components`` through ``run(tuning=..., engine="dag-ml")``;
* project an explicit winner prediction cohort;
* calibrate split conformal intervals;
* reload the calibrated result from the workspace;
* apply calibrated intervals to new point predictions;
* compute, persist, publish, and reload an audit-only robustness report.

The arrays are intentionally tiny and in-memory so the example stays suitable
for CI. Real NIR workflows should replace them with explicit development,
scoring, calibration, and external audit cohorts that carry physical sample ids.

Duration: < 10 seconds
Difficulty: ★★★☆☆
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

import nirs4all


def main(workspace_path: str | Path | None = None) -> dict[str, Any]:
    """Run the native PLS example and return key artifacts for smoke tests."""

    if workspace_path is None:
        workspace_context = tempfile.TemporaryDirectory(prefix="n4a-native-pls-conformal-")
        workspace = Path(workspace_context.name)
    else:
        workspace_context = None
        workspace = Path(workspace_path)

    try:
        X_dev = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 1.0, 0.0],
                [3.0, 1.0, 1.0],
                [4.0, 2.0, 0.0],
                [5.0, 2.0, 1.0],
            ],
            dtype=float,
        )
        y_dev = np.asarray([0.0, 1.0, 2.1, 3.0, 4.2, 5.1], dtype=float)

        X_score = np.asarray(
            [
                [1.5, 0.5, 1.0],
                [2.5, 1.2, 0.2],
                [4.5, 2.0, 0.8],
            ],
            dtype=float,
        )
        y_score = np.asarray([1.4, 2.7, 4.6], dtype=float)

        X_cal = np.asarray(
            [
                [6.0, 2.5, 1.0],
                [7.0, 3.0, 0.5],
                [8.0, 3.5, 1.0],
                [9.0, 4.0, 0.5],
            ],
            dtype=float,
        )
        y_cal = np.asarray([6.1, 7.1, 7.9, 9.2], dtype=float)
        calibration_ids = ["pls-cal-001", "pls-cal-002", "pls-cal-003", "pls-cal-004"]

        result = nirs4all.run(
            pipeline=[
                {"name": "scale", "transform": StandardScaler()},
                {"model": PLSRegression(scale=False)},
            ],
            dataset=(X_dev, y_dev),
            engine="dag-ml",
            workspace_path=workspace,
            tuning=nirs4all.NativeTuning(
                engine="optuna",
                space={"model.n_components": [1, 2]},
                metric="rmse",
                direction="minimize",
                sampler="grid",
                n_trials=2,
                score_data=nirs4all.TuningScoreData(
                    X=X_score,
                    y=y_score,
                    sample_ids=["pls-score-001", "pls-score-002", "pls-score-003"],
                ),
                workspace_tuning_id="u10-native-pls-tuning",
                winner=nirs4all.TuningWinner(
                    X=X_cal,
                    y_true=y_cal,
                    score=0.25,
                    metric="rmse",
                    sample_ids=calibration_ids,
                    model_name="NativePLSRegression",
                ),
            ),
            calibration=nirs4all.TuningCalibration(
                y_pred=np.asarray([10.0, 11.0], dtype=float),
                prediction_sample_ids=["pls-pred-001", "pls-pred-002"],
                coverage=0.8,
                workspace_conformal_id="u10-pls-conformal",
                workspace_metadata={"example": "U10_native_pls_conformal_robustness"},
            ),
            verbose=0,
        )

        restored_tuning = nirs4all.load_workspace_tuning_result(workspace, "u10-native-pls-tuning")
        restored_conformal = nirs4all.load_workspace_calibrated_result(workspace, "u10-pls-conformal")
        replayed_prediction = nirs4all.predict_calibrated(
            restored_conformal,
            y_pred=[12.0, 13.0],
            prediction_sample_ids=["pls-pred-003", "pls-pred-004"],
        )
        robustness_report = replayed_prediction.robustness(
            y_true=[12.2, 12.8],
            scenarios=[
                nirs4all.RobustnessScenarioSpec(kind="observed"),
                nirs4all.RobustnessScenarioSpec(kind="prediction_bias", severity=0.1),
                nirs4all.RobustnessScenarioSpec(kind="prediction_noise", severity=0.05),
            ],
            metadata={"batch": ["pls-external-a", "pls-external-b"]},
            slice_by=["batch"],
            seed=11,
            workspace_path=workspace,
            workspace_robustness_id="u10-pls-robustness",
            workspace_name="U10 PLS robustness audit",
            workspace_metadata={"example": "U10_native_pls_conformal_robustness"},
        )
        robustness_artifacts = workspace / "u10-pls-robustness-artifacts"
        robustness_report.save_artifacts(robustness_artifacts)
        restored_robustness = nirs4all.load_workspace_robustness_report(workspace, "u10-pls-robustness")
        restored_robustness_bundle = nirs4all.RobustnessReport.load_artifacts(robustness_artifacts)

        print("\nNative PLS tuning + conformal + robustness complete")
        print(f"  workspace: {workspace}")
        print(f"  best params: {result.run.tuning_best_params}")
        print(f"  conformal fingerprint: {restored_conformal.fingerprint}")
        print(f"  robustness fingerprint: {restored_robustness.fingerprint}")
        print(f"  robustness summary rows: {len(robustness_report.summary_rows())}")

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
        }
    finally:
        if workspace_context is not None:
            workspace_context.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U10 Native PLS tuning + conformal + robustness")
    parser.add_argument("--workspace", default=None, help="Workspace directory for persisted tuning/conformal/robustness artifacts")
    args = parser.parse_args()
    main(args.workspace)
