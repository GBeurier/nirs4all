"""Tune late fusion with absent series, resume its search and replay the winner.

Run: python examples/user/02_data_handling/U13_multimodal_late_missing_sources.py --output /tmp/mm-late-missing --stop-after 1
Resume: use the same --output with --resume instead of --stop-after.

This synthetic regression example has complete targets. Each source branch
trains only on its present observations inside native nested grouped folds.
The meta-model receives branch predictions and explicit presence indicators.
This U13 extension is under requalification on the current release baseline;
the historical 2026-09-19 results do not validate this checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
from nirs4all_io import DataProvider, MultimodalDataset, RaggedSeriesSource, TensorSource
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import nirs4all
from nirs4all.api.result import RunResult
from nirs4all.operators.transforms import SequenceSummary
from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component
from nirs4all.pipeline.dagml.multimodal_tuning import MultimodalTuningStopped


def generate_cohort(*, seed: int, params: dict[str, Any], context: dict[str, Any]) -> MultimodalDataset:
    """Declare complete targets and independently absent raw series observations."""
    count = int(params["n_samples"])
    prediction = context.get("role") == "predict"
    all_absent = bool(context.get("all_series_absent", False))
    rng = np.random.default_rng(seed)
    ids = [f"{'new-all' if all_absent else 'new' if prediction else 'train-cohort'}-{index:03d}" for index in range(count)]
    latent = rng.normal(size=(count, 2))
    wavelengths = np.linspace(950, 1150, 12)
    spectra = (latent[:, :1] * np.exp(-((wavelengths - 1000) / 35) ** 2)
               + latent[:, 1:] * np.exp(-((wavelengths - 1100) / 40) ** 2)
               + rng.normal(scale=0.1, size=(count, len(wavelengths))))
    lengths = (11 if prediction else 3) + np.arange(count) % 7
    presence = np.arange(count) % (3 if prediction else 5) != 1
    values, times = [], []
    for row, length in enumerate(lengths):
        time = np.cumsum(rng.uniform(0.1, 0.5, size=int(length)))
        series = latent[row] + rng.normal(scale=0.03, size=(int(length), 2))
        if not presence[row]:
            series[:] = np.nan  # Hidden cells never reach the branch encoder or model.
        values.append(series)
        times.append(time)
    if all_absent:
        series_source = RaggedSeriesSource(
            np.empty((0, 2)), np.asarray([0], dtype=np.int64), [],
            time_coordinates=np.empty(0), channel_names=["sensor_a", "sensor_b"], time_unit="s",
        )
    else:
        series_source = RaggedSeriesSource(
            np.concatenate(values), np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lengths)]), ids,
            time_coordinates=np.concatenate(times), channel_names=["sensor_a", "sensor_b"],
            time_unit="s", presence_mask=presence,
        )
    sources: dict[str, TensorSource | RaggedSeriesSource] = {
        "nir": TensorSource(spectra, ids, representation_id="signal_1d", axis_units={"wavelength": "nm"},
                            axis_coordinates={"wavelength": wavelengths}),
        "series": series_source,
    }
    # Archive replay retains the signed source order; new sample IDs and series
    # lengths remain independent of the training cohort.
    n_train = 2 * ((3 * count // 4) // 2)
    return MultimodalDataset(
        sources, sample_ids=ids, source_alignment="left" if all_absent else "strict",
        y=None if prediction else 2 * latent[:, 0] - 1.5 * latent[:, 1],
        target_names=["synthetic_response"], task_type="regression",
        groups=None if prediction else [f"subject-{index // 2:02d}" for index in range(count)],
        partitions=["predict"] * count if prediction else ["train"] * n_train + ["test"] * (count - n_train),
        name="late-missing-prediction" if prediction else "late-missing-provider",
    )


def make_pipeline() -> list[Any]:
    """Opt into zero/presence meta-features for the two source branches."""
    return [GroupKFold(3), {"branch": {
        "by_source": True, "missing_source_policy": "zero_with_indicator",
        "steps": {
            "nir": [StandardScaler(), Ridge(alpha=1.0)],
            "series": [SequenceSummary(), StandardScaler(), Ridge(alpha=1.0)],
        },
    }}, {"merge": "predictions"}, Ridge(alpha=1.0)]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_demo(output: Path, *, n_trials: int = 3, resume: bool = False, stop_after: int | None = None) -> dict[str, Any]:
    """Persist native search progress and export the selected complete ensemble."""
    output.mkdir(parents=True, exist_ok=True)
    provider = DataProvider(generate_cohort, provider_id="nirs4all.example.late-missing", provider_version="1",
                            params={"n_samples": 32}, seed=17)
    recipe = serialize_component(make_pipeline())
    _write_json(output / "provider_recipe.json", provider.recipe())
    _write_json(output / "pipeline.json", recipe)
    tuning: dict[str, Any] = {
        "engine": "n4m", "sampler": "random", "seed": 17, "metric": "rmse",
        "direction": "minimize", "n_trials": n_trials, "resume": resume,
        "storage": output.resolve().as_uri(), "study_name": "late-missing",
        "space": {"branches.series.0.include_length": [True, False],
                  "branches.nir.1.alpha": [0.1, 1.0], "meta.alpha": [0.1, 1.0]},
    }
    _write_json(output / "tuning.json", tuning)
    previous = json.loads((output / "stopped.json").read_text(encoding="utf-8")) if resume and (output / "stopped.json").exists() else None
    new_trials: list[int] = []
    seen_trials: set[int] | None = None

    def progress(event: dict[str, Any]) -> bool:
        nonlocal seen_trials
        trials = event["checkpoint"]["trials"]
        numbers = [trial.get("evidence", trial)["trial_index"] for trial in trials]
        # The first callback publishes the initial/resumed checkpoint before fit.
        if seen_trials is not None:
            new_trials.extend(number for number in numbers if number not in seen_trials)
        seen_trials = set(numbers)
        return stop_after is None or len(trials) < stop_after

    tuning["progress_callback"] = progress
    with threadpool_limits(limits=1):
        try:
            result = cast(RunResult, nirs4all.run(
                deserialize_component(recipe), provider, tuning=tuning, engine="dag-ml", refit=True,
                random_state=17, save_artifacts=True, save_charts=False, verbose=0, workspace_path=output / "workspace",
            ))
        except MultimodalTuningStopped as stopped:
            history = [trial.get("evidence", trial) for trial in stopped.evidence["checkpoint"]["trials"]]
            report = {"status": "cancelled", "nirs4all_version": nirs4all.__version__, "completed_trials": len(history),
                      "completed_trial_numbers": [trial["trial_index"] for trial in history],
                      "completed_trial_params": [trial["params"] for trial in history],
                      "newly_completed_trial_numbers": new_trials, "resume": True}
            _write_json(output / "dataset.json", provider.cohort.to_dict())
            _write_json(output / "stopped.json", report)
            return report
        try:
            cohort = provider.cohort
            _write_json(output / "dataset.json", cohort.to_dict())
            assert result.tuning_result is not None
            trials = list(result.tuning_result.trials)
            best_params = dict(result.tuning_best_params)
            if previous is not None:
                count = previous["completed_trials"]
                assert [trial.number for trial in trials[:count]] == previous["completed_trial_numbers"]
                assert [trial.params for trial in trials[:count]] == previous["completed_trial_params"]
                assert not set(new_trials).intersection(previous["completed_trial_numbers"])
            archive = result.export(output / "late-fusion.n4a")
            with zipfile.ZipFile(archive) as bundle:
                layout = json.loads(bundle.read("manifest.json"))["multimodal_host"]["source_presence"]
            assert layout["meta_feature_width"] == 4
            report = {
                "fixture": "deterministic synthetic data", "nirs4all_version": nirs4all.__version__, "engine": result.execution_engine,
                "task_type": "regression", "fusion": "late", "missing_source_policy": "zero_with_indicator",
                "target_names": list(cohort.target_names), "cv_best_score": result.cv_best_score,
                "test_rmse": result.best_rmse, "tuning": result.tuning_result.to_dict(),
                "newly_completed_trial_numbers": new_trials, "resume_prefix_preserved": previous is not None,
                "present_counts": {name: int(mask.sum()) for name, mask in cohort.source_presence().items()},
                "source_presence_contract": layout, "provider": result.per_dataset[cohort.name]["data_provider_evidence"],
                "archive": archive.name,
            }
        finally:
            result.close()

        for all_absent in (False, True):
            new = generate_cohort(seed=29, params={"n_samples": 8}, context={"role": "predict", "all_series_absent": all_absent})
            name = "all_absent_prediction_dataset" if all_absent else "prediction_dataset"
            _write_json(output / f"{name}.json", new.to_dict())
            restored = MultimodalDataset.from_dict(json.loads((output / f"{name}.json").read_text(encoding="utf-8")))
            with ExitStack() as guards:
                guards.enter_context(patch.object(DataProvider, "materialize", side_effect=AssertionError("Replay attempted generation")))
                for estimator in (SequenceSummary, StandardScaler, Ridge):
                    for method in ("fit", "fit_transform", "partial_fit"):
                        if hasattr(estimator, method):
                            guards.enter_context(patch.object(estimator, method, side_effect=AssertionError("Replay attempted fitting")))
                if all_absent:
                    guards.enter_context(patch.object(SequenceSummary, "transform", side_effect=AssertionError("Absent series reached its encoder")))
                expected = nirs4all.predict(archive, new)
                replay = nirs4all.predict(archive, restored)
            np.testing.assert_array_equal(replay.y_pred, expected.y_pred)
            assert replay.y_pred.shape == (len(new),) and np.isfinite(replay.y_pred).all()
            assert replay.metadata["training_performed"] is False and replay.metadata["scores"] is None
            assert replay.metadata["artifact_integrity_verified"] is True
            prefix = "all_absent_" if all_absent else ""
            report[prefix + "new_predictions"] = replay.y_pred.tolist()
            report[prefix + "prediction_sample_ids"] = replay.metadata["sample_ids"]
            report[prefix + "prediction_present_counts"] = {name: int(mask.sum()) for name, mask in new.source_presence().items()}
        report["replay_with_fit_and_generation_forbidden"] = True
        report["training_performed_on_reload"] = False
        _write_json(output / "report.json", report)
        (output / "report.txt").write_text(
            f"Synthetic regression; native nested grouped OOF; {len(trials)} completed trials.\n"
            f"CV RMSE: {report['cv_best_score']}; test RMSE: {report['test_rmse']}.\n"
            f"Selected parameters: {best_params}\n"
            f"Newly completed trial numbers: {new_trials}.\n"
            "Replay: 8 rows with some series absent and 8 rows with all series absent.\n"
            "Fit and provider generation forbidden; all-absent series encoder never called.\n",
            encoding="utf-8",
        )
    return {"status": "complete", "archive": str(archive), "report": str(output / "report.json"), "best_params": best_params}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(os.environ.get("NIRS4ALL_WORKSPACE", ".")) / "multimodal_late_missing_demo",
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if args.trials < 1 or args.stop_after is not None and args.stop_after < 1:
        parser.error("--trials and --stop-after must be positive")
    print(json.dumps(run_demo(args.output, n_trials=args.trials, resume=args.resume, stop_after=args.stop_after), indent=2))
