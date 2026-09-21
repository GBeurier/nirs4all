"""Synthetic multimodal classification, named outputs and partially observed targets.

Run: python examples/user/02_data_handling/U08_multimodal_targets.py --output /tmp/mm-targets

Each case uses the four raw modalities from U07 and three grouped folds. The
classification and masked-regression cases use two durable N4M trials. Reports
contain native scores and new predictions, checked again with fitting forbidden.
These fixtures demonstrate software behavior; no real corpus is required.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
from nirs4all_io import MultimodalDataset
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from U07_multimodal import make_cohort, make_pipeline

import nirs4all
from nirs4all.api.result import RunResult
from nirs4all.operators.models import MultimodalClassifier, MultimodalRegressor, TensorPCA
from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component

CASES = ("classification", "regression", "masked")


def make_target_cohort(case: str, *, prediction: bool = False) -> MultimodalDataset:
    """Reuse U07 raw inputs and declare the target contract explicitly."""
    source = make_cohort(29 if prediction else 17, prediction=prediction)
    task_type = "classification" if case == "classification" else "regression"
    target_names = ["condition"] if case == "classification" else ["concentration", "moisture"]
    targets = None
    target_mask = None
    if not prediction:
        response = np.asarray(source.y)
        if case == "classification":
            targets = np.asarray(["low", "medium", "high"])[np.digitize(response, [-0.5, 0.5])]
        else:
            targets = np.column_stack([response, 0.75 * response + 0.15 * response**2])
            if case == "masked":
                rows = np.arange(len(source))
                target_mask = np.column_stack([rows % 4 != 0, rows % 3 != 1])
                targets[~target_mask] = np.nan
    return MultimodalDataset(
        source.sources, sample_ids=source.sample_ids, y=targets,
        target_names=target_names, target_mask=target_mask, task_type=task_type,
        groups=source.groups, partitions=source.partitions, name=f"multimodal_{case}",
    )


def run_case(case: str, output: Path, *, resume: bool = False) -> dict[str, Any]:
    """Train from public JSON declarations, export, and verify prediction-only replay."""
    output.mkdir(parents=True, exist_ok=True)
    base = make_pipeline()[-1]["model"]
    model: Any = (
        MultimodalClassifier(base.transformers, LogisticRegression(C=1.0, max_iter=300))
        if case == "classification" else base.set_params(target_policy="per_target" if case == "masked" else "complete")
    )
    pipeline = [GroupKFold(3), {"model": model}]
    dataset = make_target_cohort(case)
    new_dataset = make_target_cohort(case, prediction=True)
    pipeline_file = output / "pipeline.json"
    dataset_file = output / "dataset.json"
    prediction_file = output / "prediction_dataset.json"
    pipeline_file.write_text(json.dumps(serialize_component(pipeline), indent=2, allow_nan=False) + "\n")
    dataset_file.write_text(json.dumps(dataset.to_dict(), indent=2, allow_nan=False) + "\n")
    prediction_file.write_text(json.dumps(new_dataset.to_dict(), indent=2, allow_nan=False) + "\n")
    tuning: dict[str, Any] | None = None
    if case in ("classification", "masked"):
        model_parameter = "model__C" if case == "classification" else "model__alpha"
        tuning = {
            "engine": "n4m", "sampler": "random", "seed": 17, "n_trials": 2,
            "space": {model_parameter: [0.1, 1.0], "transformers__image__n_components": [2, 3]},
            "storage": output.resolve().as_uri(), "study_name": case, "resume": resume,
        }
        (output / "tuning.json").write_text(json.dumps(tuning, indent=2) + "\n")
    result = cast(RunResult, nirs4all.run(
        pipeline=deserialize_component(json.loads(pipeline_file.read_text())),
        dataset=MultimodalDataset.from_dict(json.loads(dataset_file.read_text())),
        engine="dag-ml", tuning=tuning, random_state=17,
        workspace_path=output / "workspace", save_charts=False, verbose=0,
    ))
    try:
        archive = output / "multimodal.n4a"
        result.export(archive)
        expected = nirs4all.predict(archive, new_dataset, verbose=0)
        with ExitStack() as guards:
            for estimator in (MultimodalClassifier, MultimodalRegressor, TensorPCA, ColumnTransformer,
                              StandardScaler, OneHotEncoder, LogisticRegression, Ridge):
                guards.enter_context(patch.object(estimator, "fit", side_effect=AssertionError("Replay attempted to fit")))
            replay = nirs4all.predict(
                archive, MultimodalDataset.from_dict(json.loads(prediction_file.read_text())), verbose=0,
            )
        np.testing.assert_array_equal(replay.y_pred, expected.y_pred)
        assert replay.metadata["training_performed"] is False
        assert replay.metadata["target_names"] == list(dataset.target_names)
        assert replay.metadata["scores"] is None
        if case == "classification":
            assert set(np.asarray(replay.y_pred).ravel()) <= {"low", "medium", "high"}
        else:
            assert np.asarray(replay.y_pred).shape == (len(new_dataset), 2)
        # Preserve native reports; do not recompute scores in the example.
        assert result._dagml_score_set is not None
        score_reports = result._dagml_score_set["reports"]
        per_target_scores = [
            {
                "partition": report["partition"], "fold_id": report.get("fold_id"),
                "targets": {
                    name: {key.removesuffix(f":{name}"): value for key, value in report["metrics"].items() if key.endswith(f":{name}")}
                    for name in dataset.target_names
                },
            }
            for report in score_reports
        ]
        mask = np.asarray(dataset.target_mask).reshape(len(dataset), -1)
        report = {
            "fixture": "deterministic synthetic data", "case": case, "engine": result.execution_engine,
            "task_type": dataset.task_type, "target_names": list(dataset.target_names),
            "target_policy": getattr(model, "target_policy", None),
            "raw_shapes": {name: list(source.values.shape) for name, source in dataset.sources.items()},
            "observed_target_counts": {
                partition: dict(zip(dataset.target_names, mask[np.asarray(dataset.partitions) == partition].sum(axis=0).tolist(), strict=True))
                for partition in ("train", "test")
            },
            "cv_best_score": float(result.cv_best_score), "native_score_reports": score_reports,
            "per_target_scores": per_target_scores,
            "tuning": result.tuning_result.to_dict() if result.tuning_result is not None else None,
            "archive": archive.name, "archive_bytes": archive.stat().st_size,
            "new_predictions": np.asarray(replay.y_pred).tolist(),
            "training_performed_on_reload": replay.metadata["training_performed"],
            "replay_with_fit_forbidden": True,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return {
            "case": case, "cv_best_score": report["cv_best_score"],
            "archive": str(archive.resolve()), "report": str((output / "report.json").resolve()),
            "new_prediction_shape": list(np.asarray(replay.y_pred).shape),
            "replay_with_fit_forbidden": True,
        }
    finally:
        result.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = Path(os.environ["NIRS4ALL_WORKSPACE"]) / "multimodal_targets_demo" if "NIRS4ALL_WORKSPACE" in os.environ else Path("multimodal_targets_demo")
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--resume", action="store_true", help="Reuse the selected cases' durable search history")
    args = parser.parse_args()
    selected = CASES if args.case == "all" else (args.case,)
    reports = [run_case(case, args.output / case, resume=args.resume) for case in selected]
    (args.output / "report.json").write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n")
    print(json.dumps(reports, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
