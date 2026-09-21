"""Synthetic partial modalities and targets, grouped search and independent replay.

Run: python examples/user/02_data_handling/U09_multimodal_missing_sources.py --output /tmp/mm-missing

Sources contain different subsets of individuals. Each encoder learns only
from present observations with an observed target. Missing embeddings are zero
with an explicit presence indicator. New prediction inputs have no images.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
from nirs4all_io import MultimodalDataset
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from U07_multimodal import make_pipeline
from U08_multimodal_targets import make_target_cohort

import nirs4all
from nirs4all.api.result import RunResult
from nirs4all.operators.models import MultimodalRegressor, TensorPCA
from nirs4all.operators.models.sklearn.mbpls import MBPLS
from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component
from nirs4all.pipeline.dagml.multimodal_tuning import MultimodalTuningStopped


def make_missing_cohort(*, prediction: bool = False) -> MultimodalDataset:
    """Align independently incomplete sources by explicit sample identities."""
    original = make_target_cohort("masked", prediction=prediction)
    rows = np.arange(len(original))
    sources = {}
    for index, (name, source) in enumerate(original.sources.items()):
        present = rows % 5 != index
        if prediction and name == "image":
            present[:] = False
        selected = np.flatnonzero(present)[::-1]
        sources[name] = replace(
            source, values=source.values[selected],
            sample_ids=[source.sample_ids[row] for row in selected], presence_mask=None,
        )
    return MultimodalDataset(
        sources, sample_ids=original.sample_ids, source_alignment="left",
        y=original.y, target_names=original.target_names, target_mask=original.target_mask,
        task_type=original.task_type, groups=original.groups, partitions=original.partitions,
        name="multimodal_missing_sources",
    )


def run_demo(output: Path, *, fusion: str = "early", resume: bool = False, stop_after: int | None = None) -> dict[str, Any]:
    """Exercise JSON, native grouped HPO, two partial targets and captured replay."""
    output.mkdir(parents=True, exist_ok=True)
    model = make_pipeline()[-1]["model"].set_params(
        target_policy="per_target", missing_source_policy="zero_with_indicator", fusion=fusion,
    )
    if fusion == "intermediate":
        model.set_params(model=MBPLS(n_components=2, standardize=False))
    pipeline = [GroupKFold(3), {"model": model}]
    dataset, new_dataset = make_missing_cohort(), make_missing_cohort(prediction=True)
    for name, payload in (
        ("pipeline", serialize_component(pipeline)), ("dataset", dataset.to_dict()),
        ("prediction_dataset", new_dataset.to_dict()),
    ):
        (output / f"{name}.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tuning: dict[str, Any] = {
        "engine": "n4m", "sampler": "random", "seed": 17, "n_trials": 2,
        "space": {"source_weights__image": [0.5, 1.0], "transformers__image__n_components": [2, 3]},
        "storage": output.resolve().as_uri(), "study_name": "missing", "resume": resume,
    }
    (output / "tuning.json").write_text(json.dumps(tuning, indent=2) + "\n")
    if stop_after is not None:
        tuning["progress_callback"] = lambda event: len(event["checkpoint"]["trials"]) < stop_after
    try:
        result = cast(RunResult, nirs4all.run(
            deserialize_component(json.loads((output / "pipeline.json").read_text())),
            MultimodalDataset.from_dict(json.loads((output / "dataset.json").read_text())),
            tuning=tuning, engine="dag-ml", random_state=17,
            workspace_path=output / "workspace", save_charts=False, verbose=0,
        ))
    except MultimodalTuningStopped as stopped:
        return {"status": "cancelled", "completed_trials": len(stopped.evidence["checkpoint"]["trials"]), "resume": True}
    try:
        captured = result._dagml_refit_artifacts[0]["estimator"]
        expected = captured.predict(
            [new_dataset.sources[name].values for name in captured.source_names],
            source_masks=new_dataset.source_presence(),
        )
        archive = result.export(output / "multimodal.n4a")
        with ExitStack() as guards:
            for estimator in (MultimodalRegressor, TensorPCA, ColumnTransformer, StandardScaler, OneHotEncoder, Ridge, MBPLS):
                for method in ("fit", "fit_transform", "partial_fit"):
                    if hasattr(estimator, method):
                        guards.enter_context(patch.object(estimator, method, side_effect=AssertionError("Replay attempted to fit")))
            replay = nirs4all.predict(archive, MultimodalDataset.from_dict(json.loads((output / "prediction_dataset.json").read_text())))
        np.testing.assert_array_equal(replay.y_pred, expected)
        assert replay.y_pred.shape == (len(new_dataset), 2)
        assert replay.metadata["training_performed"] is False
        assert replay.metadata["scores"] is None
        assert result._dagml_score_set is not None
        partitions = np.asarray(dataset.partitions)
        report = {
            "fixture": "deterministic synthetic data", "engine": result.execution_engine,
            "fusion": fusion, "missing_source_policy": model.missing_source_policy,
            "target_policy": model.target_policy, "target_names": list(dataset.target_names),
            "present_source_counts": {
                partition: {name: int(mask[partitions == partition].sum()) for name, mask in dataset.source_presence().items()}
                for partition in ("train", "test")
            },
            "new_present_source_counts": {name: int(mask.sum()) for name, mask in new_dataset.source_presence().items()},
            "cv_best_score": result.cv_best_score, "native_score_reports": result._dagml_score_set["reports"],
            "tuning": result.tuning_result.to_dict() if result.tuning_result is not None else None,
            "archive": archive.name, "new_predictions": replay.y_pred.tolist(),
            "training_performed_on_reload": False, "replay_with_fit_forbidden": True,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return {"status": "complete", "archive": str(archive), "report": str(output / "report.json"), "cv_best_score": result.cv_best_score}
    finally:
        result.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = Path(os.environ["NIRS4ALL_WORKSPACE"]) / "multimodal_missing_demo" if "NIRS4ALL_WORKSPACE" in os.environ else Path("multimodal_missing_demo")
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--fusion", choices=("early", "intermediate"), default="early")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.output, fusion=args.fusion, resume=args.resume, stop_after=args.stop_after), indent=2))


if __name__ == "__main__":
    main()
