"""Tune raw-source branches and an OOF meta-model with durable native search.

Run: python examples/user/02_data_handling/U10_multimodal_late_tuning.py --output /tmp/mm-late
Stop: add --stop-after 2; continue with the same --output and --resume.

This deterministic synthetic example uses complete sources and targets.
DAG-ML rebuilds grouped inner OOF predictions for every candidate, scores the
meta-model on the outer folds, then refits and exports the selected ensemble.
The test partition is excluded from hyperparameter selection.
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from U07_multimodal import make_cohort

import nirs4all
from nirs4all.api.result import RunResult
from nirs4all.operators.models.multimodal import TensorPCA
from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component
from nirs4all.pipeline.dagml.multimodal_tuning import MultimodalTuningStopped


def make_pipeline() -> list[Any]:
    """Declare four raw-source branches, an OOF merge and the meta-regressor."""
    branches = {
        "nir": [StandardScaler(), Ridge(alpha=1.0)],
        "image": [TensorPCA(2, random_state=17), Ridge(alpha=1.0)],
        "series": [TensorPCA(2, random_state=17), Ridge(alpha=1.0)],
        "metadata": [ColumnTransformer([
            ("numeric", StandardScaler(), [0]),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [1]),
        ]), Ridge(alpha=1.0)],
    }
    return [GroupKFold(3), {"branch": {"by_source": True, "steps": branches}}, {"merge": "predictions"}, Ridge(alpha=1.0)]


def run_demo(output: Path, *, n_trials: int = 4, resume: bool = False, stop_after: int | None = None) -> dict[str, Any]:
    """Search, optionally resume, then export and replay the winning ensemble."""
    output.mkdir(parents=True, exist_ok=True)
    dataset = make_cohort(17)
    prediction_dataset = make_cohort(29, prediction=True)
    for name, value in (
        ("pipeline", serialize_component(make_pipeline())),
        ("dataset", dataset.to_dict()), ("prediction_dataset", prediction_dataset.to_dict()),
    ):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tuning: dict[str, Any] = {
        "engine": "n4m", "sampler": "random", "seed": 17,
        "metric": "rmse", "direction": "minimize", "n_trials": n_trials,
        "storage": output.resolve().as_uri(), "study_name": "late-fusion", "resume": resume,
        "space": {
            "branches.image.0.n_components": [2, 4],
            "branches.nir.1.alpha": [0.1, 1.0],
            "branches.metadata.0.numeric.with_mean": [False, True],
            "meta.alpha": [0.1, 1.0],
        },
    }
    (output / "tuning.json").write_text(json.dumps(tuning, indent=2) + "\n", encoding="utf-8")
    if stop_after is not None:
        tuning["progress_callback"] = lambda event: len(event["checkpoint"]["trials"]) < stop_after
    try:
        result = cast(RunResult, nirs4all.run(
            deserialize_component(json.loads((output / "pipeline.json").read_text(encoding="utf-8"))),
            MultimodalDataset.from_dict(json.loads((output / "dataset.json").read_text(encoding="utf-8"))),
            tuning=tuning, engine="dag-ml", refit=True, random_state=17,
            workspace_path=output / "workspace", save_charts=False, verbose=0,
        ))
    except MultimodalTuningStopped as stopped:
        report = {"status": "cancelled", "completed_trials": len(stopped.evidence["checkpoint"]["trials"]), "resume": True}
        (output / "stopped.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    try:
        assert result.tuning_result is not None
        archive = result.export(output / "late-fusion.n4a")
        with ExitStack() as guards:
            for estimator in (TensorPCA, ColumnTransformer, StandardScaler, OneHotEncoder, Ridge):
                for method in ("fit", "fit_transform", "partial_fit"):
                    if hasattr(estimator, method):
                        guards.enter_context(patch.object(estimator, method, side_effect=AssertionError("Replay attempted to fit")))
            replay = nirs4all.predict(
                archive, MultimodalDataset.from_dict(json.loads((output / "prediction_dataset.json").read_text(encoding="utf-8"))),
            )
        assert replay.y_pred.shape == (len(prediction_dataset),)
        assert np.isfinite(replay.y_pred).all()
        assert replay.metadata["training_performed"] is False
        report = {
            "fixture": "deterministic synthetic data", "engine": result.execution_engine,
            "fusion": "late", "cv": "grouped outer CV and native inner OOF",
            "raw_shapes": {name: list(source.values.shape) for name, source in dataset.sources.items()},
            "tuning": result.tuning_result.to_dict(), "cv_best_score": result.cv_best_score,
            "archive": archive.name, "new_predictions": replay.y_pred.tolist(),
            "prediction_sample_ids": replay.metadata["sample_ids"],
            "training_performed_on_reload": False, "replay_with_fit_forbidden": True,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return {"status": "complete", "archive": str(archive), "report": str(output / "report.json"), "best_params": dict(result.tuning_best_params)}
    finally:
        result.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = Path(os.environ["NIRS4ALL_WORKSPACE"]) / "multimodal_late_tuning_demo" if "NIRS4ALL_WORKSPACE" in os.environ else Path("multimodal_late_tuning_demo")
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if args.trials < 1 or args.stop_after is not None and args.stop_after < 0:
        parser.error("--trials must be positive and --stop-after must be non-negative")
    print(json.dumps(run_demo(args.output, n_trials=args.trials, resume=args.resume, stop_after=args.stop_after), indent=2))


if __name__ == "__main__":
    main()
